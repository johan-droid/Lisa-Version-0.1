from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class CalendarAwareness:
    path: Path | None = None
    _events: list[dict[str, Any]] = field(default_factory=list)

    def load(self) -> None:
        if self.path is None or not Path(self.path).exists():
            self._events = []
            return
        self._events = json.loads(Path(self.path).read_text(encoding="utf-8"))

    def today(self) -> list[dict[str, Any]]:
        today = datetime.now(timezone.utc).date().isoformat()
        return [event for event in self._events if str(event.get("date", "")).startswith(today)]
