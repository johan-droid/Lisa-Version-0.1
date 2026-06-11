from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from dataclasses import field

from lisa.events import EventBus, LisaEvent
from personal.context_store import PersonalContextStore


@dataclass(slots=True)
class ReminderScheduler:
    store: PersonalContextStore
    event_bus: EventBus
    interval_seconds: int = 60
    _task: asyncio.Task[None] | None = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="lisa-reminders")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while not self._stop.is_set():
            now = datetime.now(timezone.utc).isoformat()
            due = self.store.due_reminders(now)
            for reminder in due:
                self.store.complete_reminder(int(reminder["id"]))
                await self.event_bus.publish(
                    LisaEvent(
                        type="personal.reminder",
                        payload={"reminder": reminder},
                    )
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                continue
