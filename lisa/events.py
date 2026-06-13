from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class LisaEvent:
    type: str
    payload: dict[str, Any]
    trace_id: str | None = None
    session_id: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    def __init__(self, history_size: int = 100):
        self._subscribers: set[asyncio.Queue[LisaEvent]] = set()
        self._history: list[LisaEvent] = []
        self._history_size = history_size
        self._lock = asyncio.Lock()

    async def publish(self, event: LisaEvent) -> None:
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size :]
            subscribers = list(self._subscribers)

        for queue in subscribers:
            with suppress(asyncio.QueueFull):
                queue.put_nowait(event)
                continue
            with suppress(asyncio.QueueEmpty):
                queue.get_nowait()
            with suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    async def publish_dict(
        self,
        event_type: str,
        payload: dict[str, Any],
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        await self.publish(
            LisaEvent(
                type=event_type,
                payload=payload,
                trace_id=trace_id,
                session_id=session_id,
            )
        )

    async def subscribe(self) -> asyncio.Queue[LisaEvent]:
        queue: asyncio.Queue[LisaEvent] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(queue)
            history = list(self._history)

        for event in history:
            with suppress(asyncio.QueueFull):
                queue.put_nowait(event)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[LisaEvent]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)
