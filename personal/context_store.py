from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PersonalContextStore:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS personal_data (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    remind_at TEXT NOT NULL,
                    channel TEXT NOT NULL DEFAULT 'dashboard',
                    completed INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.commit()

    def set(self, key: str, value: Any, kind: str = "json") -> None:
        payload = json.dumps(value, ensure_ascii=False) if kind == "json" else str(value)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO personal_data (key, value, kind, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, kind = excluded.kind, updated_at = excluded.updated_at
                """,
                (key, payload, kind),
            )
            connection.commit()

    def get(self, key: str, default: Any = None) -> Any:
        with self._connect() as connection:
            row = connection.execute("SELECT value, kind FROM personal_data WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        if row["kind"] == "json":
            return json.loads(row["value"])
        return row["value"]

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value, kind FROM personal_data ORDER BY key").fetchall()
            reminders = connection.execute(
                "SELECT id, text, remind_at, channel, completed FROM reminders WHERE completed = 0 ORDER BY remind_at ASC LIMIT 10"
            ).fetchall()
        return {
            "preferences": {
                row["key"]: json.loads(row["value"]) if row["kind"] == "json" else row["value"]
                for row in rows
            },
            "reminders": [dict(row) for row in reminders],
        }

    def add_reminder(self, text: str, remind_at: str, channel: str = "dashboard") -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO reminders (text, remind_at, channel) VALUES (?, ?, ?)",
                (text, remind_at, channel),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def due_reminders(self, iso_now: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, text, remind_at, channel
                FROM reminders
                WHERE completed = 0 AND remind_at <= ?
                ORDER BY remind_at ASC
                """,
                (iso_now,),
            ).fetchall()
        return [dict(row) for row in rows]

    def complete_reminder(self, reminder_id: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE reminders SET completed = 1 WHERE id = ?", (reminder_id,))
            connection.commit()
