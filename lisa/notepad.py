from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from lisa.constitutions import ConstitutionMode
from lisa.embeddings import deterministic_embedding
from lisa.memory_system import HybridMemoryCoordinator, SkillArtifactRecord, iso_now
from safety.input_sanitizer import inspect_query_text, sanitize_structure, sanitize_text

try:
    from lisa.config import Settings
except Exception:  # pragma: no cover - import cycle defense
    Settings = Any  # type: ignore[assignment]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_constitution_text(value: ConstitutionMode | str) -> str:
    return value.value if isinstance(value, ConstitutionMode) else str(value)


class Notepad:
    """Compatibility facade backed by the new hybrid memory architecture."""

    def __init__(
        self, config_or_db_path: Settings | Path, event_bus: Any | None = None
    ):
        if hasattr(config_or_db_path, "workspace_root"):
            settings = config_or_db_path
            db_path = Path(settings.db_path)
            agent_id = str(getattr(settings, "agent_id", "lisa") or "lisa")
            redis_url = getattr(settings, "redis_url", None)
            postgres_dsn = getattr(settings, "postgres_dsn", None)
            chroma_dir = getattr(settings, "chroma_persist_dir", None)
            ttl = int(getattr(settings, "working_memory_ttl_seconds", 7200))
        else:
            settings = None
            db_path = Path(config_or_db_path)
            agent_id = "lisa"
            redis_url = None
            postgres_dsn = None
            chroma_dir = None
            ttl = 7200

        self.settings = settings
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.memory = HybridMemoryCoordinator(
            agent_id=agent_id,
            namespace=str(self.db_path.resolve()),
            redis_url=redis_url,
            postgres_dsn=postgres_dsn,
            chroma_persist_dir=Path(chroma_dir) if chroma_dir else None,
            working_ttl_seconds=ttl,
            event_bus=event_bus,
        )

    async def start(self) -> None:
        await self.memory.start()

    async def close(self) -> None:
        await self.memory.close()

    def startup_maintenance(
        self, backup_dir: Path, retention_days: int = 30, keep_latest_backups: int = 3
    ) -> dict[str, Any]:
        backup_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "hybrid_memory_active",
            "retention_days": retention_days,
            "keep_latest_backups": keep_latest_backups,
        }

    def get_constitution_state(self) -> dict[str, str | None]:
        return dict(self.memory._backend.constitution_state)

    def set_constitution_mode(self, mode: ConstitutionMode, reason: str | None) -> None:
        with self._lock:
            self.memory._backend.constitution_state = {
                "mode": mode.value,
                "reason": reason,
                "updated_at": iso_now(),
            }
            self.memory._backend.ledger_entries.append(
                {
                    "id": str(uuid4()),
                    "entry_type": "constitution_switch",
                    "constitution": mode.value,
                    "created_at": iso_now(),
                    "payload": {"mode": mode.value, "reason": reason},
                }
            )

    def log_entry(
        self,
        entry_type: str,
        payload: dict[str, Any],
        constitution: ConstitutionMode | str,
        personas: dict[str, float] | None = None,
        critique: str | None = None,
        reward: float | None = None,
    ) -> int:
        return self.log_entries_batch(
            [
                {
                    "entry_type": entry_type,
                    "payload": payload,
                    "constitution": _as_constitution_text(constitution),
                    "personas": personas or {},
                    "critique": critique,
                    "reward": reward,
                }
            ]
        )[0]

    def log_entries_batch(self, entries: list[dict[str, Any]]) -> list[int]:
        ids: list[int] = []
        with self._lock:
            for entry in entries:
                entry_id = len(self.memory._backend.ledger_entries) + 1
                payload = dict(
                    sanitize_structure(
                        entry.get("payload") or {},
                        max_string_length=2_000,
                        max_items=25,
                    )
                )
                constitution = str(
                    entry.get("constitution") or ConstitutionMode.RESTRICTED.value
                )
                row = {
                    "id": entry_id,
                    "entry_type": sanitize_text(
                        str(entry.get("entry_type") or "interaction"), max_length=64
                    ),
                    "constitution": constitution,
                    "created_at": iso_now(),
                    "payload": payload,
                }
                self.memory._backend.ledger_entries.append(row)
                ids.append(entry_id)

                if row["entry_type"] in {"task_summary", "interaction", "episode"}:
                    summary = str(
                        payload.get("task_summary")
                        or payload.get("input")
                        or payload.get("user_input")
                        or payload.get("user_message")
                        or ""
                    ).strip()
                    output = str(
                        payload.get("output")
                        or payload.get("response")
                        or payload.get("assistant_message")
                        or ""
                    ).strip()
                    combined = (
                        " ".join(part for part in (summary, output) if part).strip()
                        or summary
                        or output
                    )
                    episode_payload = {
                        "task_summary": combined,
                        "task_type": str(payload.get("task_type") or row["entry_type"]),
                        "tools_used": payload.get("tool_calls")
                        or payload.get("tools_used")
                        or payload.get("tool_results")
                        or [],
                        "success": str(payload.get("outcome") or "success").lower()
                        == "success",
                        "failure_reason": payload.get("failure_reason")
                        or payload.get("error"),
                        "skill_artifacts": payload.get("skill_artifacts")
                        or payload.get("context", []),
                        "constitution": constitution,
                        "session_id": payload.get("session_id"),
                        "task_id": payload.get("task_id") or payload.get("session_id"),
                        "persona_blend": payload.get("persona_blend")
                        or payload.get("personas")
                        or {},
                    }
                    episode_id = str(uuid4())
                    self.memory._backend.episodes.append(
                        self._episode_from_payload(episode_id, episode_payload)
                    )

                if row["entry_type"] in {"tool_call", "tool_error", "audit"}:
                    self.memory._backend.audit_events.append(self._audit_from_row(row))
        return ids

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        needle = sanitize_text(query, max_length=256).lower().strip()
        if not needle:
            return []
        _ = inspect_query_text(needle)
        with self._lock:
            rows = list(self.memory._backend.ledger_entries)
        matches = []
        for row in reversed(rows):
            haystack = json.dumps(row.get("payload", {}), ensure_ascii=False).lower()
            if needle in haystack or needle in str(row.get("entry_type", "")).lower():
                matches.append(row)
            if len(matches) >= limit:
                break
        return matches

    def search_interactions(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.search(query, limit=limit)
        output: list[dict[str, Any]] = []
        for row in rows:
            payload = row.get("payload", {})
            if not isinstance(payload, dict):
                continue
            output.append(
                {
                    "id": row["id"],
                    "timestamp": row["created_at"],
                    "user_id": str(payload.get("user_id") or ""),
                    "channel": str(
                        payload.get("channel") or payload.get("source") or ""
                    ),
                    "input": str(
                        payload.get("input")
                        or payload.get("user_input")
                        or payload.get("user_message")
                        or ""
                    ),
                    "output": str(
                        payload.get("output")
                        or payload.get("response")
                        or payload.get("assistant_message")
                        or ""
                    ),
                    "tool_calls": payload.get("tool_calls")
                    or payload.get("tools_used")
                    or [],
                    "persona_blend": payload.get("persona_blend")
                    or payload.get("personas")
                    or {},
                    "constitution": row["constitution"],
                    "outcome": str(payload.get("outcome") or "unknown"),
                    "reward": float(payload.get("reward") or 0.0),
                    "self_critique": str(
                        payload.get("self_critique") or payload.get("critique") or ""
                    ),
                    "entry_type": row["entry_type"],
                    "payload": sanitize_structure(
                        payload, max_string_length=2_000, max_items=25
                    ),
                }
            )
        return output

    def latest_entries(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self.memory._backend.ledger_entries[-limit:]))

    def latest_interaction_timestamp(self) -> datetime | None:
        for row in self.latest_entries(limit=50):
            if row.get("entry_type") in {"interaction", "task_summary", "episode"}:
                return datetime.fromisoformat(str(row["created_at"]))
        return None

    def evolution_candidates(
        self, limit: int = 6, min_reward: float = 0.35
    ) -> list[dict[str, Any]]:
        rows = []
        for row in self.latest_entries(limit=200):
            payload = row.get("payload", {})
            if not isinstance(payload, dict):
                continue
            reward = float(payload.get("reward") or 0.0)
            outcome = str(payload.get("outcome") or "success").lower()
            if reward <= float(min_reward) or outcome != "success":
                rows.append(
                    {
                        "id": row["id"],
                        "payload": payload,
                        "score": max(0.0, 1.0 - reward),
                        "created_at": row["created_at"],
                    }
                )
            if len(rows) >= limit:
                break
        return rows

    def recent_failure_interactions(
        self, limit: int = 24, min_reward: float = 0.35
    ) -> list[dict[str, Any]]:
        candidates = self.evolution_candidates(limit=limit, min_reward=min_reward)
        failures = []
        for item in candidates:
            payload = item.get("payload", {})
            if str(payload.get("outcome") or "").lower() != "success" or payload.get(
                "error"
            ):
                failures.append(item)
        return failures[:limit]

    def add_metric(self, metric: str, value: str) -> None:
        with self._lock:
            self.memory._backend.metrics.append(
                {"metric": metric, "value": value, "created_at": iso_now()}
            )

    def recent_metrics(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self.memory._backend.metrics[-limit:]))

    def register_evolution_skill_artifact(
        self,
        *,
        skill_name: str,
        description: str,
        encrypted_url: str,
        url_hash: str,
        keywords: list[str] | None = None,
        checksum: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "active",
    ) -> dict[str, Any]:
        metadata = dict(metadata or {})
        metadata.update(
            {
                "description": description,
                "encrypted_url": encrypted_url,
                "url_hash": url_hash,
                "checksum": checksum,
                "keywords": list(keywords or []),
                "status": status,
            }
        )
        record = SkillArtifactRecord(
            skill_name=skill_name,
            content=str(metadata.get("code") or ""),
            metadata=metadata,
            embedding=deterministic_embedding(
                " ".join([skill_name, description, " ".join(keywords or [])])
            ),
            active=status == "active",
        )
        with self._lock:
            self.memory._backend.skills[skill_name] = record
        return self._skill_record_to_dict(record)

    def get_evolution_skill_artifact(self, skill_name: str) -> dict[str, Any] | None:
        with self._lock:
            record = self.memory._backend.skills.get(skill_name)
        if record is None:
            return None
        return self._skill_record_to_dict(record)

    def search_evolution_skill_artifacts(
        self, query: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        needle = query.lower().strip()
        with self._lock:
            skills = list(self.memory._backend.skills.values())
        ranked = []
        for skill in skills:
            if not skill.active:
                continue
            content = (
                json.dumps(skill.metadata, ensure_ascii=False).lower()
                + " "
                + skill.content.lower()
            )
            if needle in content or needle in skill.skill_name.lower():
                ranked.append(self._skill_record_to_dict(skill))
        return ranked[:limit]

    def mark_evolution_skill_loaded(self, skill_name: str) -> None:
        with self._lock:
            record = self.memory._backend.skills.get(skill_name)
            if record is not None:
                record.metadata["last_loaded_at"] = iso_now()
                record.metadata["load_count"] = (
                    int(record.metadata.get("load_count", 0) or 0) + 1
                )
                record.metadata["last_used"] = iso_now()
                record.updated_at = iso_now()

    def disable_evolution_skill(self, skill_name: str, reason: str) -> None:
        with self._lock:
            record = self.memory._backend.skills.get(skill_name)
            if record is not None:
                record.active = False
                record.metadata["status"] = "disabled"
                record.metadata["disable_reason"] = reason
                record.updated_at = iso_now()

    def disable_evolution_skill_artifact(self, skill_name: str, reason: str) -> None:
        self.disable_evolution_skill(skill_name, reason)

    @staticmethod
    def _episode_from_payload(episode_id: str, payload: dict[str, Any]):
        from lisa.memory_system import EpisodeRecord

        return EpisodeRecord(
            id=episode_id,
            task_summary=str(payload.get("task_summary") or ""),
            task_type=str(payload.get("task_type") or "generic"),
            tools_used=list(payload.get("tools_used") or []),
            success=bool(payload.get("success", True)),
            failure_reason=payload.get("failure_reason"),
            skill_artifacts=list(payload.get("skill_artifacts") or []),
            embedding=deterministic_embedding(str(payload.get("task_summary") or "")),
            created_at=iso_now(),
            evolved_from=payload.get("evolved_from"),
            metadata={
                "constitution": payload.get("constitution"),
                "session_id": payload.get("session_id"),
                "task_id": payload.get("task_id"),
                "persona_blend": payload.get("persona_blend") or {},
            },
        )

    @staticmethod
    def _audit_from_row(row: dict[str, Any]):
        from lisa.memory_system import AuditRecord

        payload = row.get("payload", {})
        return AuditRecord(
            id=str(row["id"]),
            component=str(
                payload.get("component") or row.get("entry_type") or "notepad"
            ),
            event_type=str(
                payload.get("event_type") or row.get("entry_type") or "unknown"
            ),
            payload=dict(payload) if isinstance(payload, dict) else {"value": payload},
            session_id=str(payload.get("session_id") or "") or None,
            task_id=str(payload.get("task_id") or payload.get("trace_id") or "")
            or None,
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _skill_record_to_dict(record: SkillArtifactRecord) -> dict[str, Any]:
        return {
            "skill_name": record.skill_name,
            "description": str(record.metadata.get("description") or ""),
            "encrypted_url": str(record.metadata.get("encrypted_url") or ""),
            "url_hash": str(record.metadata.get("url_hash") or ""),
            "keywords": list(record.metadata.get("keywords") or []),
            "keywords_json": json.dumps(
                record.metadata.get("keywords") or [], ensure_ascii=False
            ),
            "checksum": str(record.metadata.get("checksum") or ""),
            "metadata": dict(record.metadata),
            "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
            "status": str(
                record.metadata.get("status")
                or ("active" if record.active else "disabled")
            ),
            "disable_reason": record.metadata.get("disable_reason"),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "last_loaded_at": record.metadata.get("last_loaded_at"),
            "load_count": int(record.metadata.get("load_count", 0) or 0),
            "active": record.active,
        }


@dataclass(slots=True)
class LedgerWrite:
    entry_type: str
    payload: dict[str, Any]
    constitution: str
    personas: dict[str, float]
    critique: str | None = None
    reward: float | None = None
    future: asyncio.Future[int] | None = None


class AsyncNotepadWriter:
    def __init__(
        self,
        notepad: Notepad,
        event_bus: Any | None = None,
        batch_size: int = 10,
        flush_interval: float = 0.1,
    ):
        self.notepad = notepad
        self.event_bus = event_bus
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: asyncio.Queue[LedgerWrite] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        starter = getattr(self.notepad, "start", None)
        if callable(starter):
            await starter()
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="hybrid-memory-writer")

    async def close(self) -> None:
        self._stop.set()
        await self.flush_pending()
        if self._task is not None:
            await self._task
            self._task = None
        closer = getattr(self.notepad, "close", None)
        if callable(closer):
            await closer()

    async def flush_pending(self) -> None:
        while not self._queue.empty():
            await asyncio.sleep(0.01)

    async def enqueue(
        self,
        entry_type: str,
        payload: dict[str, Any],
        constitution: ConstitutionMode | str,
        personas: dict[str, float] | None = None,
        critique: str | None = None,
        reward: float | None = None,
    ) -> asyncio.Future[int]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[int] = loop.create_future()
        await self._queue.put(
            LedgerWrite(
                entry_type=entry_type,
                payload=dict(payload),
                constitution=_as_constitution_text(constitution),
                personas=dict(personas or {}),
                critique=critique,
                reward=reward,
                future=future,
            )
        )
        return future

    async def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            batch: list[LedgerWrite] = []
            try:
                first = await asyncio.wait_for(
                    self._queue.get(), timeout=self.flush_interval
                )
                batch.append(first)
            except TimeoutError:
                continue
            while len(batch) < self.batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())

            row_ids = self.notepad.log_entries_batch(
                [
                    {
                        "entry_type": item.entry_type,
                        "payload": item.payload,
                        "constitution": item.constitution,
                        "personas": item.personas,
                        "critique": item.critique,
                        "reward": item.reward,
                    }
                    for item in batch
                ]
            )
            for item, row_id in zip(batch, row_ids, strict=False):
                if item.future is not None and not item.future.done():
                    item.future.set_result(row_id)


def search_notepad(
    query: str, limit: int = 10, db_path: Path | None = None
) -> list[dict[str, Any]]:
    target = Path(db_path) if db_path is not None else Path("data/lisa_notepad.db")
    notepad = Notepad(target)
    return notepad.search_interactions(query=query, limit=limit)
