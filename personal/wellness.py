from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(slots=True)
class WellnessTracker:
    start_time: datetime | None = None
    break_after_minutes: int = 120

    def start_session(self) -> None:
        self.start_time = datetime.now(timezone.utc)

    def should_check_in(self) -> bool:
        if self.start_time is None:
            return False
        return (datetime.now(timezone.utc) - self.start_time) >= timedelta(
            minutes=self.break_after_minutes
        )
