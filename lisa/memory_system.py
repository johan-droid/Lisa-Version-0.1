from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lisa.constitutions import ConstitutionMode
from lisa.embeddings import EMBEDDING_DIMS, cosine_similarity, deterministic_embedding
from lisa.events import EventBus, LisaEvent
from safety.input_sanitizer import (
    sanitize_structure,
    sanitize_text,
    sanitize_user_visible_text,
)

try:
    import asyncpg  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    asyncpg = None

try:
    import chromadb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None

try:
    from redis.asyncio import Redis  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    Redis = None  # type: ignore[assignment]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utcnow().isoformat()


@dataclass(slots=True)
class WorkingMemoryState:
    task_id: str
    agent_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reflection_notes: list[str] = field(default_factory=list)
    current_plan: list[str] = field(default_factory=list)
    iteration: int = 0
    idempotency_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: str = field(default_factory=iso_now)
    updated_at: str = field(default_factory=iso_now)

    def as_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "messages": list(self.messages),
            "tool_calls": list(self.tool_calls),
            "reflection_notes": list(self.reflection_notes),
            "current_plan": list(self.current_plan),
            "iteration": self.iteration,
            "idempotency_cache": dict(self.idempotency_cache),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class EpisodeRecord:
    id: str
    task_summary: str
    task_type: str
    tools_used: list[dict[str, Any]]
    success: bool
    failure_reason: str | None
    skill_artifacts: list[dict[str, Any]]
    embedding: list[float]
    created_at: str
    evolved_from: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_search_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entry_type": "episode",
            "constitution": str(
                self.metadata.get("constitution") or ConstitutionMode.RESTRICTED.value
            ),
            "created_at": self.created_at,
            "payload": {
                "task_summary": self.task_summary,
                "task_type": self.task_type,
                "tools_used": self.tools_used,
                "success": self.success,
                "failure_reason": self.failure_reason,
                "skill_artifacts": self.skill_artifacts,
                **self.metadata,
            },
        }


@dataclass(slots=True)
class SkillArtifactRecord:
    skill_name: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float]
    created_at: str = field(default_factory=iso_now)
    updated_at: str = field(default_factory=iso_now)
    active: bool = True


@dataclass(slots=True)
class AuditRecord:
    id: str
    component: str
    event_type: str
    payload: dict[str, Any]
    session_id: str | None
    task_id: str | None
    created_at: str = field(default_factory=iso_now)


@dataclass(slots=True)
class EnrichedMemoryContext:
    task_id: str
    agent_id: str
    similar_episodes: list[dict[str, Any]]
    relevant_skills: list[dict[str, Any]]
    working_memory_key: str


@dataclass(slots=True)
class _InMemoryHybridBackend:
    working_sessions: dict[str, WorkingMemoryState] = field(default_factory=dict)
    scratchpads: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    episodes: list[EpisodeRecord] = field(default_factory=list)
    skills: dict[str, SkillArtifactRecord] = field(default_factory=dict)
    audit_events: list[AuditRecord] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    constitution_state: dict[str, str | None] = field(
        default_factory=lambda: {
            "mode": ConstitutionMode.RESTRICTED.value,
            "reason": "Default safe mode",
            "updated_at": iso_now(),
        }
    )
    ledger_entries: list[dict[str, Any]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_BACKEND_REGISTRY: dict[str, _InMemoryHybridBackend] = {}


class HybridMemoryCoordinator:
    def __init__(
        self,
        *,
        agent_id: str,
        namespace: str,
        redis_url: str | None = None,
        postgres_dsn: str | None = None,
        chroma_persist_dir: Path | None = None,
        working_ttl_seconds: int = 7200,
        event_bus: EventBus | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.namespace = namespace
        self.redis_url = redis_url
        self.postgres_dsn = postgres_dsn
        self.chroma_persist_dir = chroma_persist_dir
        self.working_ttl_seconds = working_ttl_seconds
        self.event_bus = event_bus
        self._backend = _BACKEND_REGISTRY.setdefault(
            namespace, _InMemoryHybridBackend()
        )
        self._redis = None
        self._postgres_pool = None
        self._chroma_client = None
        self._skill_collection = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        if Redis is not None and self.redis_url:
            try:  # pragma: no cover - integration path
                self._redis = Redis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception:
                self._redis = None
        if asyncpg is not None and self.postgres_dsn:
            try:  # pragma: no cover - integration path
                self._postgres_pool = await asyncpg.create_pool(
                    self.postgres_dsn, min_size=1, max_size=4
                )
                await self._initialize_postgres()
            except Exception:
                self._postgres_pool = None
        if chromadb is not None and self.chroma_persist_dir is not None:
            try:  # pragma: no cover - integration path
                self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(
                    path=str(self.chroma_persist_dir)
                )
                self._skill_collection = self._chroma_client.get_or_create_collection(
                    "skill_artifacts"
                )
            except Exception:
                self._skill_collection = None
                self._chroma_client = None
        self._started = True

    async def close(self) -> None:
        if self._postgres_pool is not None:  # pragma: no cover - integration path
            await self._postgres_pool.close()
            self._postgres_pool = None
        if self._redis is not None:  # pragma: no cover - integration path
            await self._redis.close()
            self._redis = None
        self._started = False

    async def initialize_working_memory(self, task_id: str) -> str:
        key = self.working_memory_key(task_id)
        state = WorkingMemoryState(task_id=task_id, agent_id=self.agent_id)
        async with self._backend.lock:
            self._backend.working_sessions[key] = state
            self._backend.scratchpads[task_id] = []
        await self._publish(
            "memory.working_initialized",
            {"task_id": task_id, "key": key},
            task_id=task_id,
        )
        return key

    def working_memory_key(self, task_id: str) -> str:
        return f"session:{self.agent_id}:{task_id}"

    async def get_working_memory(self, task_id: str) -> WorkingMemoryState:
        key = self.working_memory_key(task_id)
        async with self._backend.lock:
            state = self._backend.working_sessions.get(key)
            if state is None:
                state = WorkingMemoryState(task_id=task_id, agent_id=self.agent_id)
                self._backend.working_sessions[key] = state
            return state

    async def append_message(self, task_id: str, role: str, content: str) -> None:
        state = await self.get_working_memory(task_id)
        state.messages.append(
            {
                "role": sanitize_text(role, max_length=32),
                "content": sanitize_user_visible_text(content, max_length=8_000),
                "timestamp": iso_now(),
            }
        )
        state.updated_at = iso_now()
        if len(state.messages) > 50:
            del state.messages[:-50]

    async def append_tool_call(self, task_id: str, item: dict[str, Any]) -> None:
        state = await self.get_working_memory(task_id)
        state.tool_calls.append(
            sanitize_structure(item, max_string_length=4_000, max_items=25)
        )
        state.updated_at = iso_now()
        if len(state.tool_calls) > 50:
            del state.tool_calls[:-50]

    async def append_reflection(self, task_id: str, note: str) -> None:
        state = await self.get_working_memory(task_id)
        state.reflection_notes.append(
            sanitize_user_visible_text(note, max_length=2_000)
        )
        state.updated_at = iso_now()
        if len(state.reflection_notes) > 50:
            del state.reflection_notes[:-50]

    async def set_current_plan(self, task_id: str, steps: list[str]) -> None:
        state = await self.get_working_memory(task_id)
        state.current_plan = [
            sanitize_user_visible_text(step, max_length=600) for step in steps[:20]
        ]
        state.updated_at = iso_now()

    async def bump_iteration(self, task_id: str) -> int:
        state = await self.get_working_memory(task_id)
        state.iteration += 1
        state.updated_at = iso_now()
        return state.iteration

    async def get_cached_tool_result(
        self, task_id: str, idempotency_key: str
    ) -> dict[str, Any] | None:
        state = await self.get_working_memory(task_id)
        return state.idempotency_cache.get(idempotency_key)

    async def cache_tool_result(
        self, task_id: str, idempotency_key: str, result: dict[str, Any]
    ) -> None:
        state = await self.get_working_memory(task_id)
        state.idempotency_cache[idempotency_key] = sanitize_structure(
            result, max_string_length=4_000, max_items=25
        )
        state.updated_at = iso_now()
        if len(state.idempotency_cache) > 100:
            oldest = next(iter(state.idempotency_cache))
            state.idempotency_cache.pop(oldest, None)

    async def write_scratchpad(self, task_id: str, item: dict[str, Any]) -> None:
        async with self._backend.lock:
            scratchpad = self._backend.scratchpads.setdefault(task_id, [])
            scratchpad.append(
                sanitize_structure(item, max_string_length=4_000, max_items=25)
            )
            if len(scratchpad) > 64:
                del scratchpad[:-64]
            safe_item = scratchpad[-1]
        await self._publish(
            "react.scratchpad", {"task_id": task_id, "item": safe_item}, task_id=task_id
        )

    async def read_scratchpad(self, task_id: str) -> list[dict[str, Any]]:
        async with self._backend.lock:
            return list(self._backend.scratchpads.get(task_id, []))

    async def clear_task(self, task_id: str) -> None:
        key = self.working_memory_key(task_id)
        async with self._backend.lock:
            self._backend.working_sessions.pop(key, None)
            self._backend.scratchpads.pop(task_id, None)

    async def store_episode(
        self,
        *,
        task_summary: str,
        task_type: str,
        tools_used: list[dict[str, Any]],
        success: bool,
        failure_reason: str | None,
        skill_artifacts: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        evolved_from: str | None = None,
    ) -> EpisodeRecord:
        metadata = dict(metadata or {})
        episode = EpisodeRecord(
            id=str(uuid4()),
            task_summary=sanitize_user_visible_text(task_summary, max_length=8_000),
            task_type=sanitize_text(task_type, max_length=128),
            tools_used=list(
                sanitize_structure(
                    list(tools_used), max_string_length=2_000, max_items=25
                )
            ),
            success=success,
            failure_reason=sanitize_user_visible_text(
                failure_reason or "", max_length=2_000
            )
            or None,
            skill_artifacts=list(
                sanitize_structure(
                    list(skill_artifacts), max_string_length=2_000, max_items=25
                )
            ),
            embedding=deterministic_embedding(
                sanitize_user_visible_text(task_summary, max_length=8_000),
                EMBEDDING_DIMS,
            ),
            created_at=iso_now(),
            evolved_from=evolved_from,
            metadata=dict(
                sanitize_structure(metadata, max_string_length=2_000, max_items=25)
            ),
        )
        async with self._backend.lock:
            self._backend.episodes.append(episode)
            self._backend.ledger_entries.append(episode.as_search_row())
        await self._insert_episode_postgres(episode)
        await self._publish(
            "memory.episode_stored",
            {"episode_id": episode.id, "task_type": task_type, "success": success},
            task_id=str(metadata.get("task_id") or ""),
            session_id=str(metadata.get("session_id") or ""),
        )
        return episode

    async def similar_episodes(self, text: str, limit: int = 3) -> list[dict[str, Any]]:
        target = deterministic_embedding(text, EMBEDDING_DIMS)
        async with self._backend.lock:
            ranked = sorted(
                self._backend.episodes,
                key=lambda item: cosine_similarity(target, item.embedding),
                reverse=True,
            )
            return [episode.as_search_row()["payload"] for episode in ranked[:limit]]

    async def recent_failure_episodes(
        self, task_type: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        async with self._backend.lock:
            rows = [
                episode
                for episode in self._backend.episodes
                if not episode.success
                and (task_type is None or episode.task_type == task_type)
            ]
        rows.sort(key=lambda item: item.created_at, reverse=True)
        return [episode.as_search_row()["payload"] for episode in rows[:limit]]

    async def recent_success_episodes(
        self, task_type: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        async with self._backend.lock:
            rows = [
                episode
                for episode in self._backend.episodes
                if episode.success
                and (task_type is None or episode.task_type == task_type)
            ]
        rows.sort(key=lambda item: item.created_at, reverse=True)
        return [episode.as_search_row()["payload"] for episode in rows[:limit]]

    async def upsert_skill_artifact(
        self,
        *,
        skill_name: str,
        content: str,
        metadata: dict[str, Any],
        embedding_text: str | None = None,
    ) -> SkillArtifactRecord:
        record = SkillArtifactRecord(
            skill_name=skill_name,
            content=content,
            metadata=dict(metadata),
            embedding=deterministic_embedding(
                embedding_text or content, EMBEDDING_DIMS
            ),
            updated_at=iso_now(),
        )
        async with self._backend.lock:
            previous = self._backend.skills.get(skill_name)
            if previous is not None:
                record.created_at = previous.created_at
            self._backend.skills[skill_name] = record
        await self._upsert_skill_chroma(record)
        await self._publish(
            "memory.skill_upserted", {"skill_name": skill_name}, task_id=None
        )
        return record

    async def relevant_skills(self, text: str, limit: int = 3) -> list[dict[str, Any]]:
        target = deterministic_embedding(text, EMBEDDING_DIMS)
        async with self._backend.lock:
            skills = [item for item in self._backend.skills.values() if item.active]
        ranked = sorted(
            skills,
            key=lambda item: cosine_similarity(target, item.embedding),
            reverse=True,
        )
        results: list[dict[str, Any]] = []
        for skill in ranked[:limit]:
            results.append(
                {
                    "skill_name": skill.skill_name,
                    "content": skill.content,
                    "metadata": dict(skill.metadata),
                    "created_at": skill.created_at,
                    "updated_at": skill.updated_at,
                }
            )
        return results

    async def low_performing_skills(
        self, threshold: float = 0.6
    ) -> list[dict[str, Any]]:
        async with self._backend.lock:
            skills = list(self._backend.skills.values())
        results = []
        for skill in skills:
            success_rate = float(skill.metadata.get("success_rate", 0.0))
            if success_rate < threshold and skill.active:
                results.append(
                    {
                        "skill_name": skill.skill_name,
                        "content": skill.content,
                        "metadata": dict(skill.metadata),
                        "success_rate": success_rate,
                    }
                )
        results.sort(key=lambda item: item["success_rate"])
        return results

    async def replace_skill(
        self, skill_name: str, content: str, metadata: dict[str, Any]
    ) -> SkillArtifactRecord:
        metadata = dict(metadata)
        metadata["evolved_version"] = int(metadata.get("evolved_version", 0) or 0) + 1
        return await self.upsert_skill_artifact(
            skill_name=skill_name, content=content, metadata=metadata
        )

    async def disable_skill(self, skill_name: str, reason: str) -> None:
        async with self._backend.lock:
            record = self._backend.skills.get(skill_name)
            if record is not None:
                record.active = False
                record.metadata["disabled_reason"] = reason
                record.updated_at = iso_now()

    async def record_audit(
        self,
        *,
        component: str,
        event_type: str,
        payload: dict[str, Any],
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> AuditRecord:
        record = AuditRecord(
            id=str(uuid4()),
            component=sanitize_text(component, max_length=64),
            event_type=sanitize_text(event_type, max_length=128),
            payload=dict(
                sanitize_structure(payload, max_string_length=2_000, max_items=25)
            ),
            session_id=session_id,
            task_id=task_id,
        )
        async with self._backend.lock:
            self._backend.audit_events.append(record)
            self._backend.ledger_entries.append(
                {
                    "id": record.id,
                    "entry_type": "audit",
                    "constitution": str(
                        record.payload.get("constitution")
                        or ConstitutionMode.RESTRICTED.value
                    ),
                    "created_at": record.created_at,
                    "payload": {
                        "component": record.component,
                        "event_type": record.event_type,
                        **record.payload,
                    },
                }
            )
        await self._insert_audit_postgres(record)
        return record

    async def add_metric(self, metric: str, value: str) -> None:
        async with self._backend.lock:
            self._backend.metrics.append(
                {"metric": metric, "value": value, "created_at": iso_now()}
            )

    async def recent_metrics(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self._backend.lock:
            return list(reversed(self._backend.metrics[-limit:]))

    async def search_ledger(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        needle = sanitize_text(query, max_length=256).lower().strip()
        if not needle:
            return []
        async with self._backend.lock:
            rows = list(self._backend.ledger_entries)
        matches = []
        for row in reversed(rows):
            text = json.dumps(row.get("payload", {}), ensure_ascii=False).lower()
            if needle in text or needle in str(row.get("entry_type", "")).lower():
                matches.append(row)
            if len(matches) >= limit:
                break
        return matches

    async def latest_entries(self, limit: int = 10) -> list[dict[str, Any]]:
        async with self._backend.lock:
            return list(reversed(self._backend.ledger_entries[-limit:]))

    async def latest_interaction_timestamp(self) -> datetime | None:
        rows = await self.latest_entries(limit=50)
        for row in rows:
            if row.get("entry_type") in {"interaction", "task_summary", "episode"}:
                return datetime.fromisoformat(str(row["created_at"]))
        return None

    async def enrich_task(
        self, *, task_id: str, description: str
    ) -> EnrichedMemoryContext:
        await self.initialize_working_memory(task_id)
        similar = await self.similar_episodes(description, limit=3)
        skills = await self.relevant_skills(description, limit=3)
        return EnrichedMemoryContext(
            task_id=task_id,
            agent_id=self.agent_id,
            similar_episodes=similar,
            relevant_skills=skills,
            working_memory_key=self.working_memory_key(task_id),
        )

    async def get_constitution_state(self) -> dict[str, str | None]:
        async with self._backend.lock:
            return dict(self._backend.constitution_state)

    async def set_constitution_mode(
        self, mode: ConstitutionMode, reason: str | None
    ) -> None:
        async with self._backend.lock:
            self._backend.constitution_state = {
                "mode": mode.value,
                "reason": reason,
                "updated_at": iso_now(),
            }
        await self.record_audit(
            component="constitution",
            event_type="mode_changed",
            payload={"mode": mode.value, "reason": reason},
        )

    async def _publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        task_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if self.event_bus is None:
            return
        await self.event_bus.publish(
            LisaEvent(
                type=event_type,
                payload=payload,
                session_id=session_id,
                trace_id=task_id,
            )
        )

    async def _initialize_postgres(self) -> None:
        if self._postgres_pool is None:  # pragma: no cover - integration path
            return
        async with self._postgres_pool.acquire() as connection:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_episodes (
                    id UUID PRIMARY KEY,
                    task_summary TEXT NOT NULL,
                    task_type VARCHAR(50) NOT NULL,
                    tools_used JSONB NOT NULL,
                    success BOOLEAN NOT NULL,
                    failure_reason TEXT,
                    skill_artifacts JSONB NOT NULL,
                    embedding VECTOR(1536) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    evolved_from UUID NULL
                )
                """)
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_audit_events (
                    id UUID PRIMARY KEY,
                    component VARCHAR(64) NOT NULL,
                    event_type VARCHAR(128) NOT NULL,
                    payload JSONB NOT NULL,
                    session_id VARCHAR(128),
                    task_id VARCHAR(128),
                    created_at TIMESTAMP NOT NULL
                )
                """)

    async def _insert_episode_postgres(self, episode: EpisodeRecord) -> None:
        if self._postgres_pool is None:  # pragma: no cover - integration path
            return
        async with self._postgres_pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO agent_episodes (
                    id, task_summary, task_type, tools_used, success, failure_reason,
                    skill_artifacts, embedding, created_at, evolved_from
                ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7::jsonb, $8, $9, $10)
                """,
                episode.id,
                episode.task_summary,
                episode.task_type,
                json.dumps(episode.tools_used),
                episode.success,
                episode.failure_reason,
                json.dumps(episode.skill_artifacts),
                episode.embedding,
                datetime.fromisoformat(episode.created_at),
                episode.evolved_from,
            )

    async def _insert_audit_postgres(self, record: AuditRecord) -> None:
        if self._postgres_pool is None:  # pragma: no cover - integration path
            return
        async with self._postgres_pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO agent_audit_events (id, component, event_type, payload, session_id, task_id, created_at)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                """,
                record.id,
                record.component,
                record.event_type,
                json.dumps(record.payload),
                record.session_id,
                record.task_id,
                datetime.fromisoformat(record.created_at),
            )

    async def _upsert_skill_chroma(self, record: SkillArtifactRecord) -> None:
        if self._skill_collection is None:  # pragma: no cover - integration path
            return
        metadata = {
            key: value
            for key, value in record.metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        metadata.update(
            {
                "last_used": str(record.metadata.get("last_used") or ""),
                "skill_type": str(record.metadata.get("skill_type") or "generic"),
                "success_rate": float(record.metadata.get("success_rate", 0.0) or 0.0),
                "evolved_version": int(record.metadata.get("evolved_version", 0) or 0),
                "active": record.active,
            }
        )
        self._skill_collection.upsert(
            ids=[record.skill_name],
            documents=[record.content],
            embeddings=[record.embedding],
            metadatas=[metadata],
        )
