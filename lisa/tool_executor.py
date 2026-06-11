from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from lisa.constitutions import ConstitutionMode
from lisa.events import EventBus, LisaEvent
from lisa.schemas import ToolCall, ToolResult
from lisa.tools import ToolRegistry


@dataclass(slots=True, order=True)
class ToolExecutionJob:
    priority: int
    sequence: int
    tool_call: ToolCall = field(compare=False)
    constitution: str = field(compare=False)
    session_id: str | None = field(default=None, compare=False)
    trace_id: str | None = field(default=None, compare=False)
    future: asyncio.Future[ToolResult] | None = field(default=None, compare=False)
    job_id: str = field(default_factory=lambda: str(uuid4()), compare=False)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        event_bus: EventBus,
        max_workers: int = 10,
        queue_size: int = 256,
    ):
        self.registry = registry
        self.event_bus = event_bus
        self.max_workers = max_workers
        self._queue: asyncio.PriorityQueue[ToolExecutionJob] = asyncio.PriorityQueue(maxsize=queue_size)
        self._sequence = itertools.count()
        self._stop = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if self._workers:
            return
        for index in range(self.max_workers):
            task = asyncio.create_task(self._run_worker(), name=f"tool-executor-{index}")
            self._workers.append(task)

    async def close(self) -> None:
        self._stop.set()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

    async def submit(
        self,
        tool_call: ToolCall,
        constitution: ConstitutionMode | str,
        priority: int = 0,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> ToolResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ToolResult] = loop.create_future()
        constitution_value = constitution.value if isinstance(constitution, ConstitutionMode) else constitution
        await self._queue.put(
            ToolExecutionJob(
                priority=priority,
                sequence=next(self._sequence),
                tool_call=tool_call,
                constitution=constitution_value,
                session_id=session_id,
                trace_id=trace_id,
                future=future,
            )
        )
        return await future

    async def execute_many(
        self,
        tool_calls: list[ToolCall],
        constitution: ConstitutionMode | str,
        priority: int = 0,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> list[ToolResult]:
        if not tool_calls:
            return []

        jobs = [
            self.submit(
                tool_call=tool_call,
                constitution=constitution,
                priority=priority,
                session_id=session_id,
                trace_id=trace_id,
            )
            for tool_call in tool_calls
        ]
        return list(await asyncio.gather(*jobs))

    async def _run_worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            await self._handle(job)

    async def _handle(self, job: ToolExecutionJob) -> None:
        await self.event_bus.publish(
            LisaEvent(
                type="tool_executor.job_started",
                payload={
                    "job_id": job.job_id,
                    "tool": job.tool_call.name,
                    "constitution": job.constitution,
                    "session_id": job.session_id,
                    "trace_id": job.trace_id,
                },
            )
        )

        try:
            output = await self.registry.invoke(
                name=job.tool_call.name,
                arguments=job.tool_call.arguments,
                constitution=ConstitutionMode(job.constitution),
                session_id=job.session_id,
                trace_id=job.trace_id,
            )
            result = ToolResult(tool=job.tool_call.name, success=True, output=output)
            await self.event_bus.publish(
                LisaEvent(
                    type="tool_executor.job_finished",
                    payload={
                        "job_id": job.job_id,
                        "tool": job.tool_call.name,
                        "session_id": job.session_id,
                        "trace_id": job.trace_id,
                        "success": True,
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - defensive tool failure path
            result = ToolResult(tool=job.tool_call.name, success=False, error=str(exc))
            await self.event_bus.publish(
                LisaEvent(
                    type="tool_executor.job_error",
                    payload={
                        "job_id": job.job_id,
                        "tool": job.tool_call.name,
                        "session_id": job.session_id,
                        "trace_id": job.trace_id,
                        "error": str(exc),
                    },
                )
            )

        if job.future is not None and not job.future.done():
            job.future.set_result(result)
