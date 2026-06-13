from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SOURCES = ("telegram", "slack", "whatsapp")


@dataclass(slots=True)
class ChannelAccessRecord:
    source: str
    user_id: str


class ChannelAccessController:
    def __init__(
        self, storage_path: Path, initial: dict[str, list[str]] | None = None
    ) -> None:
        self.storage_path = storage_path
        self._rules: dict[str, set[str]] = {source: set() for source in DEFAULT_SOURCES}
        self._load()
        for source, user_ids in (initial or {}).items():
            self._rules.setdefault(source, set()).update(self._normalize_ids(user_ids))
        self._persist()

    def summary(self) -> dict[str, list[str]]:
        return {
            source: sorted(user_ids)
            for source, user_ids in self._rules.items()
            if user_ids
        }

    def configured_sources(self) -> list[str]:
        return [source for source, user_ids in self._rules.items() if user_ids]

    def is_authorized(self, source: str, user_id: str) -> bool:
        rules = self._rules.get(source, set())
        if not rules:
            return True
        return user_id in rules

    def is_restricted(self, source: str) -> bool:
        return bool(self._rules.get(source))

    def grant(self, source: str, user_id: str) -> ChannelAccessRecord:
        source_key = str(source).strip().lower()
        normalized_user_id = str(user_id).strip()
        if not source_key:
            raise ValueError("source is required")
        if not normalized_user_id:
            raise ValueError("user_id is required")
        self._rules.setdefault(source_key, set()).add(normalized_user_id)
        self._persist()
        return ChannelAccessRecord(source=source_key, user_id=normalized_user_id)

    def revoke(self, source: str, user_id: str) -> bool:
        source_key = str(source).strip().lower()
        normalized_user_id = str(user_id).strip()
        users = self._rules.setdefault(source_key, set())
        if normalized_user_id not in users:
            return False
        users.remove(normalized_user_id)
        self._persist()
        return True

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        for source, raw_user_ids in payload.items():
            self._rules.setdefault(str(source).strip().lower(), set()).update(
                self._normalize_ids(raw_user_ids)
            )

    def _persist(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            source: sorted(user_ids)
            for source, user_ids in self._rules.items()
            if user_ids
        }

        # Atomic rename-under-lock
        import tempfile
        import os

        # Create temp file in same directory to ensure rename is atomic
        fd, temp_path = tempfile.mkstemp(dir=str(self.storage_path.parent), suffix=".tmp", prefix="channel_access_")
        temp_file = Path(temp_path)
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            try:
                temp_file.chmod(0o600)
            except OSError:
                pass
            os.replace(temp_path, str(self.storage_path))
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _normalize_ids(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, (tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        if value is None:
            return []
        text = str(value).strip()
        if not text:
            return []
        return [text]
