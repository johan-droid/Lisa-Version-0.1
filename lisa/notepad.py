from __future__ import annotations

import asyncio
import gzip
import json
import sqlite3
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from contextlib import suppress
from pathlib import Path
from threading import Lock
from typing import Any

from sqlite_utils import Database

from lisa.constitutions import ConstitutionMode
from lisa.events import EventBus, LisaEvent


@dataclass(slots=True)
class LedgerWrite:
    entry_type: str
    payload: dict[str, Any]
    constitution: str
    personas: dict[str, float]
    critique: str | None = None
    reward: float | None = None
    future: asyncio.Future[int] | None = None


class Notepad:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._database = Database(self.db_path)
        self._lock = Lock()
        self._fts_enabled = False
        self._interactions_fts_enabled = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self._database.enable_wal()
        with self._connect() as connection:
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    constitution TEXT NOT NULL,
                    personas TEXT NOT NULL,
                    critique TEXT,
                    reward REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    channel TEXT,
                    input TEXT,
                    output TEXT,
                    tool_calls TEXT,
                    persona_blend TEXT,
                    constitution TEXT NOT NULL,
                    outcome TEXT,
                    reward REAL,
                    self_critique TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS constitution_state (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    mode TEXT NOT NULL,
                    reason TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dashboard_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO constitution_state (singleton_id, mode, reason, updated_at)
                VALUES (1, ?, ?, ?)
                """,
                (
                    ConstitutionMode.RESTRICTED.value,
                    "Default safe mode",
                    self._utcnow(),
                ),
            )
            connection.commit()

            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS ledger_fts
                    USING fts5(entry_id UNINDEXED, content)
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS interactions_fts
                    USING fts5(
                        interaction_id UNINDEXED,
                        input,
                        output,
                        tool_calls
                    )
                    """
                )
                self._interactions_fts_enabled = True
            except sqlite3.OperationalError:
                self._interactions_fts_enabled = False

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).isoformat()

    def get_constitution_state(self) -> dict[str, str | None]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT mode, reason, updated_at FROM constitution_state WHERE singleton_id = 1"
            ).fetchone()
        return {
            "mode": row["mode"],
            "reason": row["reason"],
            "updated_at": row["updated_at"],
        }

    def set_constitution_mode(self, mode: ConstitutionMode, reason: str | None) -> None:
        updated_at = self._utcnow()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE constitution_state
                SET mode = ?, reason = ?, updated_at = ?
                WHERE singleton_id = 1
                """,
                (mode.value, reason, updated_at),
            )
            self._insert_entry(
                connection=connection,
                entry_type="constitution_switch",
                payload={"mode": mode.value, "reason": reason},
                constitution=mode.value,
                personas={},
            )
            connection.commit()

    def log_entry(
        self,
        entry_type: str,
        payload: dict[str, Any],
        constitution: ConstitutionMode | str,
        personas: dict[str, float] | None = None,
        critique: str | None = None,
        reward: float | None = None,
        ) -> int:
        constitution_value = constitution.value if isinstance(constitution, ConstitutionMode) else constitution
        return self.log_entries_batch(
            [
                LedgerWrite(
                    entry_type=entry_type,
                    payload=payload,
                    constitution=constitution_value,
                    personas=personas or {},
                    critique=critique,
                    reward=reward,
                )
            ]
        )[0]

    def log_entries_batch(self, entries: list[LedgerWrite]) -> list[int]:
        if not entries:
            return []

        with self._lock, self._connect() as connection:
            row_ids: list[int] = []
            for entry in entries:
                row_id = self._insert_entry(
                    connection=connection,
                    entry_type=entry.entry_type,
                    payload=entry.payload,
                    constitution=entry.constitution,
                    personas=entry.personas,
                    critique=entry.critique,
                    reward=entry.reward,
                )
                self._insert_interaction(
                    connection=connection,
                    entry_type=entry.entry_type,
                    payload=entry.payload,
                    constitution=entry.constitution,
                    personas=entry.personas,
                    critique=entry.critique,
                    reward=entry.reward,
                )
                row_ids.append(row_id)
            connection.commit()
        return row_ids

    def _insert_entry(
        self,
        connection: sqlite3.Connection,
        entry_type: str,
        payload: dict[str, Any],
        constitution: str,
        personas: dict[str, float],
        critique: str | None = None,
        reward: float | None = None,
    ) -> int:
        serialized_payload = json.dumps(payload, ensure_ascii=True)
        serialized_personas = json.dumps(personas, ensure_ascii=True)
        created_at = self._utcnow()
        cursor = connection.execute(
            """
            INSERT INTO ledger_entries (
                entry_type,
                payload,
                constitution,
                personas,
                critique,
                reward,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_type,
                serialized_payload,
                constitution,
                serialized_personas,
                critique,
                reward,
                created_at,
            ),
        )

        if self._fts_enabled:
            connection.execute(
                "INSERT INTO ledger_fts (entry_id, content) VALUES (?, ?)",
                (cursor.lastrowid, self._search_text(entry_type, payload)),
            )

        return int(cursor.lastrowid)

    def _insert_interaction(
        self,
        connection: sqlite3.Connection,
        entry_type: str,
        payload: dict[str, Any],
        constitution: str,
        personas: dict[str, float],
        critique: str | None = None,
        reward: float | None = None,
    ) -> int:
        interaction = self._interaction_from_entry(
            entry_type=entry_type,
            payload=payload,
            constitution=constitution,
            personas=personas,
            critique=critique,
            reward=reward,
        )
        cursor = connection.execute(
            """
            INSERT INTO interactions (
                timestamp,
                user_id,
                channel,
                input,
                output,
                tool_calls,
                persona_blend,
                constitution,
                outcome,
                reward,
                self_critique
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction["timestamp"],
                interaction["user_id"],
                interaction["channel"],
                interaction["input"],
                interaction["output"],
                interaction["tool_calls"],
                interaction["persona_blend"],
                interaction["constitution"],
                interaction["outcome"],
                interaction["reward"],
                interaction["self_critique"],
            ),
        )

        if self._interactions_fts_enabled:
            connection.execute(
                """
                INSERT INTO interactions_fts (
                    interaction_id,
                    input,
                    output,
                    tool_calls
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    cursor.lastrowid,
                    interaction["input"],
                    interaction["output"],
                    interaction["tool_calls"],
                ),
            )

        return int(cursor.lastrowid)

    @staticmethod
    def _search_text(entry_type: str, payload: dict[str, Any]) -> str:
        joined = json.dumps(payload, ensure_ascii=True)
        return f"{entry_type} {joined}"

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        with self._connect() as connection:
            if self._fts_enabled:
                try:
                    rows = connection.execute(
                        """
                        SELECT e.id, e.entry_type, e.payload, e.constitution, e.created_at
                        FROM ledger_fts f
                        JOIN ledger_entries e ON e.id = f.entry_id
                        WHERE ledger_fts MATCH ?
                        ORDER BY e.id DESC
                        LIMIT ?
                        """,
                        (query, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = self._search_without_fts(connection, query, limit)
            else:
                rows = self._search_without_fts(connection, query, limit)

        return [self._deserialize_row(row) for row in rows]

    def search_interactions(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        with self._connect() as connection:
            if self._interactions_fts_enabled:
                try:
                    rows = connection.execute(
                        """
                        SELECT i.id, i.timestamp, i.user_id, i.channel, i.input, i.output,
                               i.tool_calls, i.persona_blend, i.constitution, i.outcome,
                               i.reward, i.self_critique
                        FROM interactions_fts f
                        JOIN interactions i ON i.id = f.interaction_id
                        WHERE interactions_fts MATCH ?
                        ORDER BY i.id DESC
                        LIMIT ?
                        """,
                        (query, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            else:
                rows = connection.execute(
                    """
                    SELECT id, timestamp, user_id, channel, input, output, tool_calls,
                           persona_blend, constitution, outcome, reward, self_critique
                    FROM interactions
                    WHERE input LIKE ? OR output LIKE ? OR tool_calls LIKE ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", f"%{query}%", f"%{query}%", limit),
                    ).fetchall()
        return [self._deserialize_interaction_row(row) for row in rows]

    def latest_interaction_timestamp(self) -> datetime | None:
        with self._connect() as connection:
            row = connection.execute("SELECT MAX(timestamp) AS timestamp FROM interactions").fetchone()
        if row is None:
            return None
        value = row["timestamp"]
        if not value:
            return None
        timestamp = datetime.fromisoformat(str(value))
        if timestamp.tzinfo is None:
            timestamp = timestamp.astimezone()
        return timestamp

    def recent_failure_interactions(self, limit: int = 24, min_reward: float = 0.35) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, user_id, channel, input, output, tool_calls,
                       persona_blend, constitution, outcome, reward, self_critique
                FROM interactions
                WHERE outcome IN ('error', 'failed', 'timeout')
                   OR reward IS NULL
                   OR reward < ?
                   OR self_critique LIKE '%fail%'
                   OR self_critique LIKE '%error%'
                   OR tool_calls LIKE '%error%'
                ORDER BY id DESC
                LIMIT ?
                """,
                (min_reward, limit),
            ).fetchall()
        return [self._deserialize_interaction_row(row) for row in rows]

    @staticmethod
    def _search_without_fts(
        connection: sqlite3.Connection,
        query: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        like_query = f"%{query}%"
        return connection.execute(
            """
            SELECT id, entry_type, payload, constitution, created_at
            FROM ledger_entries
            WHERE payload LIKE ? OR entry_type LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (like_query, like_query, limit),
        ).fetchall()

    @staticmethod
    def _interaction_from_entry(
        entry_type: str,
        payload: dict[str, Any],
        constitution: str,
        personas: dict[str, float],
        critique: str | None = None,
        reward: float | None = None,
    ) -> dict[str, Any]:
        timestamp = str(payload.get("timestamp") or Notepad._utcnow())
        input_text = str(
            payload.get("input")
            or payload.get("user_input")
            or payload.get("text")
            or payload.get("message")
            or json.dumps(payload, ensure_ascii=False)
        )
        output_text = str(
            payload.get("output")
            or payload.get("response")
            or payload.get("assistant")
            or ""
        )
        tool_calls = payload.get("tool_calls") or payload.get("tool_results") or []
        persona_blend = payload.get("persona_blend") or payload.get("persona_weights") or personas or {}
        outcome = str(payload.get("outcome") or entry_type)
        self_critique = payload.get("self_critique") or critique or ""

        return {
            "timestamp": timestamp,
            "user_id": str(payload.get("user_id") or ""),
            "channel": str(payload.get("channel") or payload.get("source") or ""),
            "input": input_text,
            "output": output_text,
            "tool_calls": json.dumps(tool_calls, ensure_ascii=False),
            "persona_blend": json.dumps(persona_blend, ensure_ascii=False),
            "constitution": constitution,
            "outcome": outcome,
            "reward": float(payload.get("reward") if payload.get("reward") is not None else (reward if reward is not None else 0.0)),
            "self_critique": str(self_critique),
        }

    def recent_metrics(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT metric, value, created_at
                FROM dashboard_metrics
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_backup(self, backup_dir: Path, keep_latest: int = 3) -> Path:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"notepad-{timestamp}.db"
        shutil.copy2(self.db_path, backup_path)
        try:
            backup_path.chmod(0o600)
        except Exception:
            pass
        self._prune_backups(backup_dir, keep_latest)
        return backup_path

    def purge_old_entries(self, retention_days: int = 30, archive_dir: Path | None = None) -> dict[str, Any]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, retention_days))
        cutoff_iso = cutoff.isoformat()
        archived_rows = 0
        deleted_rows = 0
        archive_path: Path | None = None

        with self._lock, self._connect() as connection:
            interaction_rows = connection.execute(
                "SELECT * FROM interactions WHERE timestamp < ? ORDER BY timestamp ASC",
                (cutoff_iso,),
            ).fetchall()
            ledger_rows = connection.execute(
                "SELECT * FROM ledger_entries WHERE created_at < ? ORDER BY created_at ASC",
                (cutoff_iso,),
            ).fetchall()

            if archive_dir is not None and (interaction_rows or ledger_rows):
                archive_path = self._archive_rows(archive_dir, interaction_rows, ledger_rows, cutoff_iso)
                archived_rows = len(interaction_rows) + len(ledger_rows)

            if interaction_rows:
                connection.execute("DELETE FROM interactions WHERE timestamp < ?", (cutoff_iso,))
                deleted_rows += len(interaction_rows)

            if ledger_rows:
                connection.execute("DELETE FROM ledger_entries WHERE created_at < ?", (cutoff_iso,))
                deleted_rows += len(ledger_rows)

            if self._interactions_fts_enabled:
                connection.execute("DELETE FROM interactions_fts")
                connection.execute(
                    """
                    INSERT INTO interactions_fts (interaction_id, input, output, tool_calls)
                    SELECT id, input, output, tool_calls FROM interactions
                    """
                )
            if self._fts_enabled:
                connection.execute("DELETE FROM ledger_fts")
                connection.execute(
                    """
                    INSERT INTO ledger_fts (entry_id, content)
                    SELECT id, entry_type || ' ' || payload FROM ledger_entries
                    """
                )

            connection.commit()

        return {
            "archived_rows": archived_rows,
            "deleted_rows": deleted_rows,
            "archive_path": str(archive_path) if archive_path is not None else "",
        }

    def vacuum(self) -> None:
        with self._connect() as connection:
            connection.execute("VACUUM")

    def startup_maintenance(
        self,
        backup_dir: Path,
        retention_days: int = 30,
        keep_latest_backups: int = 3,
    ) -> dict[str, Any]:
        backup_path = self.create_backup(backup_dir, keep_latest_backups)
        purge_result = self.purge_old_entries(retention_days=retention_days, archive_dir=backup_dir / "archive")
        if datetime.now(timezone.utc).weekday() == 0:
            self.vacuum()
        return {"backup_path": str(backup_path), "purge_result": purge_result}

    def add_metric(self, metric: str, value: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dashboard_metrics (metric, value, created_at)
                VALUES (?, ?, ?)
                """,
                (metric, value, self._utcnow()),
            )
            connection.commit()

    def latest_entries(self, limit: int = 5) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, entry_type, payload, constitution, created_at
                FROM ledger_entries
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._deserialize_row(row) for row in rows]

    def evolution_candidates(
        self,
        limit: int = 6,
        min_reward: float = 0.35,
    ) -> list[dict[str, Any]]:
        sample_limit = max(limit * 6, 24)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, entry_type, payload, constitution, critique, reward, created_at
                FROM ledger_entries
                ORDER BY id DESC
                LIMIT ?
                """,
                (sample_limit,),
            ).fetchall()

        candidates: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload"])
            entry_type = str(row["entry_type"])
            reward = row["reward"]
            reward_value = float(reward) if reward is not None else None
            text_blobs = [
                entry_type,
                str(payload.get("user_input") or payload.get("text") or ""),
                str(payload.get("response") or ""),
                str(payload.get("error") or ""),
                str(payload.get("self_critique") or ""),
                str(payload.get("outcome") or ""),
                str(row["critique"] or ""),
            ]
            lowered = " ".join(text_blobs).lower()

            score = 0.0
            if entry_type in {"tool_error", "ledger.error"}:
                score += 3.0
            if payload.get("outcome") == "error":
                score += 2.5
            if payload.get("error"):
                score += 1.5
            if reward_value is not None and reward_value < min_reward:
                score += (min_reward - reward_value) + 1.0
            if any(keyword in lowered for keyword in ("fail", "broken", "retry", "timeout", "error")):
                score += 0.5

            if score <= 0.0:
                continue

            candidates.append(
                {
                    "id": int(row["id"]),
                    "entry_type": entry_type,
                    "constitution": row["constitution"],
                    "critique": row["critique"],
                    "reward": reward_value,
                    "created_at": row["created_at"],
                    "payload": payload,
                    "score": round(score, 3),
                }
            )

        candidates.sort(key=lambda item: (float(item["score"]), int(item["id"])), reverse=True)
        return candidates[: max(1, limit)]

    @staticmethod
    def _deserialize_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "entry_type": row["entry_type"],
            "constitution": row["constitution"],
            "created_at": row["created_at"],
            "payload": json.loads(row["payload"]),
        }

    @staticmethod
    def _deserialize_interaction_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "user_id": row["user_id"],
            "channel": row["channel"],
            "input": row["input"],
            "output": row["output"],
            "tool_calls": json.loads(row["tool_calls"] or "[]"),
            "persona_blend": json.loads(row["persona_blend"] or "{}"),
            "constitution": row["constitution"],
            "outcome": row["outcome"],
            "reward": row["reward"],
            "self_critique": row["self_critique"],
        }

    def _archive_rows(
        self,
        archive_dir: Path,
        interaction_rows: list[sqlite3.Row],
        ledger_rows: list[sqlite3.Row],
        cutoff_iso: str,
    ) -> Path:
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = archive_dir / f"notepad-archive-{timestamp}.jsonl.gz"
        with gzip.open(archive_path, "wt", encoding="utf-8") as handle:
            for row in interaction_rows:
                handle.write(json.dumps({"table": "interactions", **dict(row)}, ensure_ascii=False) + "\n")
            for row in ledger_rows:
                handle.write(json.dumps({"table": "ledger_entries", **dict(row)}, ensure_ascii=False) + "\n")
            handle.write(json.dumps({"table": "meta", "cutoff": cutoff_iso}, ensure_ascii=False) + "\n")
        try:
            archive_path.chmod(0o600)
        except Exception:
            pass
        return archive_path

    def _prune_backups(self, backup_dir: Path, keep_latest: int) -> None:
        backups = sorted(backup_dir.glob("notepad-*.db"), key=lambda path: path.stat().st_mtime, reverse=True)
        for stale in backups[max(0, keep_latest):]:
            with suppress(Exception):
                stale.unlink()


class AsyncNotepadWriter:
    def __init__(
        self,
        notepad: Notepad,
        event_bus: EventBus | None = None,
        max_queue_size: int = 1024,
        batch_size: int = 32,
        flush_interval: float = 0.05,
    ):
        self.notepad = notepad
        self.event_bus = event_bus
        self.max_queue_size = max_queue_size
        self.batch_size = max(1, min(batch_size, 10))
        self.flush_interval = max(0.1, flush_interval)
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._write_loop(), name="notepad-writer")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            with suppress(asyncio.CancelledError):
                await self._task

    async def flush_pending(self) -> None:
        batch: list[dict[str, Any]] = []
        while not self._queue.empty():
            with suppress(asyncio.QueueEmpty):
                batch.append(self._queue.get_nowait())
            if len(batch) >= self.batch_size:
                await self._flush(batch)
                batch.clear()

        if batch:
            await self._flush(batch)

    async def enqueue(
        self,
        entry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> asyncio.Future[int]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[int] = loop.create_future()
        payload = dict(entry or kwargs)
        if "entry_type" not in payload:
            raise ValueError("enqueue requires an 'entry_type'.")
        constitution_value = payload.get("constitution")
        if isinstance(constitution_value, ConstitutionMode):
            constitution_value = constitution_value.value
        elif constitution_value is None:
            constitution_value = ConstitutionMode.RESTRICTED.value
        payload["constitution"] = constitution_value
        payload.setdefault("payload", {})
        payload.setdefault("personas", payload.get("persona_blend") or payload.get("persona_weights") or {})
        payload["future"] = future
        await self._queue.put(
            payload
        )
        return future

    async def _write_loop(self) -> None:
        batch: list[dict[str, Any]] = []
        while not self._stop.is_set() or not self._queue.empty():
            item = await self._next_item()
            if item is not None:
                batch.append(item)

            while len(batch) < self.batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())

            if not batch:
                continue

            await self._flush(batch)
            batch.clear()

    async def _run(self) -> None:
        await self._write_loop()

    async def _next_item(self) -> dict[str, Any] | None:
        if self._stop.is_set() and self._queue.empty():
            return None

        try:
            return await asyncio.wait_for(self._queue.get(), timeout=self.flush_interval)
        except TimeoutError:
            return None

    async def _flush(self, batch: list[dict[str, Any]]) -> None:
        writes = [
            LedgerWrite(
                entry_type=str(item["entry_type"]),
                payload=dict(item.get("payload") or {}),
                constitution=str(item.get("constitution") or ConstitutionMode.RESTRICTED.value),
                personas=dict(item.get("personas") or {}),
                critique=item.get("critique"),
                reward=item.get("reward"),
                future=item.get("future"),
            )
            for item in batch
        ]
        try:
            row_ids = await asyncio.to_thread(self.notepad.log_entries_batch, writes)
        except Exception as exc:  # pragma: no cover - defensive recovery path
            for item in batch:
                future = item.get("future")
                if future is not None and not future.done():
                    future.set_exception(exc)
            if self.event_bus is not None:
                await self.event_bus.publish_dict(
                    "ledger.error",
                    {"message": str(exc), "batch_size": len(batch)},
                )
            return

        for item, row_id in zip(batch, row_ids, strict=True):
            future = item.get("future")
            if future is not None and not future.done():
                future.set_result(row_id)
            if self.event_bus is not None:
                await self.event_bus.publish(
                    LisaEvent(
                        type="ledger.append",
                        payload={
                            "row_id": row_id,
                            "entry_type": item["entry_type"],
                            "constitution": item.get("constitution"),
                            "payload": item.get("payload") or {},
                        },
                    )
                )


def search_notepad(query: str, limit: int = 5, db_path: Path | None = None) -> list[dict[str, Any]]:
    notepad = Notepad(db_path or Path("data/lisa_notepad.db"))
    return notepad.search_interactions(query=query, limit=limit)
