from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
from pathlib import Path

from lisa.constitutions import ConstitutionMode
from lisa.events import EventBus, LisaEvent
from lisa.sandbox import ToolSandbox
from lisa.schemas import TaskContext, ToolCall, ToolResult
from lisa.tools import ToolRegistry


@dataclass(slots=True, order=True)
class ToolExecutionJob:
    priority: int
    sequence: int
    tool_call: ToolCall = field(compare=False)
    constitution: str = field(compare=False)
    task_context: TaskContext | None = field(default=None, compare=False)
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
        self._queue: asyncio.PriorityQueue[ToolExecutionJob] = asyncio.PriorityQueue(
            maxsize=queue_size
        )
        self._sequence = itertools.count()
        self._stop = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []
        workspace_root = getattr(
            getattr(registry, "settings", None), "workspace_root", Path.cwd()
        )
        self.sandbox = ToolSandbox(Path(workspace_root))

    async def start(self) -> None:
        if self._workers:
            return
        for index in range(self.max_workers):
            task = asyncio.create_task(
                self._run_worker(), name=f"tool-executor-{index}"
            )
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
        task_context: TaskContext | None = None,
    ) -> ToolResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ToolResult] = loop.create_future()
        constitution_value = (
            constitution.value
            if isinstance(constitution, ConstitutionMode)
            else constitution
        )
        await self._queue.put(
            ToolExecutionJob(
                priority=priority,
                sequence=next(self._sequence),
                tool_call=tool_call,
                constitution=constitution_value,
                task_context=task_context,
                session_id=session_id,
                trace_id=trace_id,
                future=future,
            )
        )
        return await future

    async def execute(
        self, tool_call: ToolCall, task_context: TaskContext
    ) -> ToolResult:
        return await self.submit(
            tool_call=tool_call,
            constitution=task_context.constitution,
            priority=0,
            session_id=task_context.session_id,
            trace_id=task_context.task_id,
            task_context=task_context,
        )

    async def execute_many(
        self,
        tool_calls: list[ToolCall],
        constitution: ConstitutionMode | str,
        priority: int = 0,
        session_id: str | None = None,
        trace_id: str | None = None,
        task_context: TaskContext | None = None,
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
                task_context=task_context,
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
                    "task_id": (
                        job.task_context.task_id
                        if job.task_context is not None
                        else None
                    ),
                },
            )
        )

        try:
            task_context = job.task_context or TaskContext(
                task_id=job.session_id or job.job_id,
                agent_id="lisa",
                session_id=job.session_id,
                constitution=job.constitution,
            )
            required_level = self.sandbox.permission_for(job.tool_call.name)
            if not self.sandbox.has_permission(
                task_context.approved_levels,
                required_level,
                explicit_user_grant=task_context.explicit_user_grant,
                two_factor_confirmed=task_context.two_factor_confirmed,
            ):
                result = ToolResult(
                    tool=job.tool_call.name,
                    success=False,
                    error=f"PERMISSION_DENIED:{required_level}",
                )
                if job.future is not None and not job.future.done():
                    job.future.set_result(result)
                await self.event_bus.publish(
                    LisaEvent(
                        type="tool_executor.permission_denied",
                        payload={
                            "job_id": job.job_id,
                            "tool": job.tool_call.name,
                            "required_level": required_level,
                            "task_id": task_context.task_id,
                        },
                    )
                )
                return

            sandbox_result = self.sandbox.prepare(
                task_context.task_id, job.tool_call.name, job.tool_call.arguments
            )
            cached = None
            memory = getattr(getattr(self.registry, "notepad", None), "memory", None)
            if memory is not None:
                cached = await memory.get_cached_tool_result(
                    task_context.task_id, sandbox_result.idempotency_key
                )
            if cached is not None:
                result = ToolResult(
                    tool=job.tool_call.name,
                    success=bool(cached.get("success", True)),
                    output=cached.get("output"),
                    error=cached.get("error"),
                )
            else:
                output = await self._invoke_registry(
                    job=job,
                    task_context=task_context,
                    sandbox_result=sandbox_result,
                )
                result = ToolResult(
                    tool=job.tool_call.name, success=True, output=output
                )
                if memory is not None:
                    await memory.cache_tool_result(
                        task_context.task_id,
                        sandbox_result.idempotency_key,
                        result.model_dump(),
                    )
                    await memory.record_audit(
                        component="tool_executor",
                        event_type="tool_call",
                        payload={
                            "tool": job.tool_call.name,
                            "arguments": job.tool_call.arguments,
                            "success": True,
                            "required_level": required_level,
                            "idempotency_key": sandbox_result.idempotency_key,
                        },
                        session_id=job.session_id,
                        task_id=task_context.task_id,
                    )
            await self.event_bus.publish(
                LisaEvent(
                    type="tool_executor.job_finished",
                    payload={
                        "job_id": job.job_id,
                        "tool": job.tool_call.name,
                        "session_id": job.session_id,
                        "trace_id": job.trace_id,
                        "success": True,
                        "task_id": task_context.task_id,
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - defensive tool failure path
            result = ToolResult(tool=job.tool_call.name, success=False, error=str(exc))
            task_context = job.task_context or TaskContext(
                task_id=job.session_id or job.job_id,
                agent_id="lisa",
                session_id=job.session_id,
                constitution=job.constitution,
            )
            memory = getattr(getattr(self.registry, "notepad", None), "memory", None)
            if memory is not None:
                await memory.record_audit(
                    component="tool_executor",
                    event_type="tool_error",
                    payload={
                        "tool": job.tool_call.name,
                        "arguments": job.tool_call.arguments,
                        "error": str(exc),
                    },
                    session_id=job.session_id,
                    task_id=task_context.task_id,
                )
            await self.event_bus.publish(
                LisaEvent(
                    type="tool_executor.job_error",
                    payload={
                        "job_id": job.job_id,
                        "tool": job.tool_call.name,
                        "session_id": job.session_id,
                        "trace_id": job.trace_id,
                        "error": str(exc),
                        "task_id": task_context.task_id,
                    },
                )
            )

        if job.future is not None and not job.future.done():
            job.future.set_result(result)

    async def _invoke_registry(
        self, *, job: ToolExecutionJob, task_context: TaskContext, sandbox_result
    ) -> Any:
        try:
            return await self.registry.invoke(
                name=job.tool_call.name,
                arguments=job.tool_call.arguments,
                constitution=ConstitutionMode(job.constitution),
                session_id=job.session_id,
                trace_id=job.trace_id,
                task_id=task_context.task_id,
                idempotency_key=sandbox_result.idempotency_key,
                workspace_root_override=sandbox_result.workspace_root,
            )
        except TypeError:
            return await self.registry.invoke(
                name=job.tool_call.name,
                arguments=job.tool_call.arguments,
                constitution=ConstitutionMode(job.constitution),
                session_id=job.session_id,
                trace_id=job.trace_id,
            )
