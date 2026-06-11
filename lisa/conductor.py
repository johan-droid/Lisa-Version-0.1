from __future__ import annotations

import asyncio
import itertools
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from lisa.constitutions import ConstitutionMode
from lisa.events import EventBus, LisaEvent
from lisa.schemas import BrainTask, ChatRequest, ChatResponse, InboundMessage, ToolCall, ToolResult
from lisa.tool_executor import ToolExecutor


@dataclass(slots=True, order=True)
class ConductorJob:
    sort_index: int = field(init=False, repr=False)
    sequence: int
    priority: int = field(compare=False)
    kind: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
    future: asyncio.Future[Any] | None = field(default=None, compare=False)
    job_id: str = field(default_factory=lambda: str(uuid4()), compare=False)

    def __post_init__(self) -> None:
        # Higher numeric priority should run first, so the queue stores the
        # inverted sort key while keeping the original priority value intact.
        self.sort_index = -int(self.priority)


@dataclass(slots=True)
class ConductorSessionState:
    session_id: str
    turn_index: int = 0
    constitution: str | None = None
    persona_weights: dict[str, float] = field(default_factory=dict)
    last_context_summary: list[dict[str, Any]] = field(default_factory=list)
    last_tool_results: list[ToolResult] = field(default_factory=list)
    last_response: str | None = None


class TaskConductor:
    def __init__(
        self,
        runtime: Any,
        tool_executor: ToolExecutor,
        event_bus: EventBus,
        queue_size: int = 256,
        max_arms: int = 10,
        max_follow_ups: int = 3,
    ):
        self.runtime = runtime
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.max_arms = max_arms
        self.max_follow_ups = max_follow_ups
        self._queue: asyncio.PriorityQueue[ConductorJob] = asyncio.PriorityQueue(maxsize=queue_size)
        self._sequence = itertools.count()
        self._stop = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._arm_semaphore = asyncio.Semaphore(max_arms)
        self._arm_tasks: set[asyncio.Task[None]] = set()
        self._sessions: dict[str, ConductorSessionState] = {}
        self._snapshot_task: asyncio.Task[None] | None = None
        self._autonomous_conductor = None

    async def start(self) -> None:
        if self._worker is None:
            self._load_snapshot()
            self._worker = asyncio.create_task(self.run_forever(), name="task-conductor")
            self._snapshot_task = asyncio.create_task(self._periodic_snapshot())
            settings = getattr(self.runtime, "settings", None)
            if settings and getattr(settings, "autonomous_enabled", False):
                try:
                    from conductor.autonomous import SelfDirectedConductor
                    self._autonomous_conductor = SelfDirectedConductor(settings, self)
                    await self._autonomous_conductor.start()
                except Exception:
                    pass

    async def close(self) -> None:
        self._stop.set()
        if self._autonomous_conductor is not None:
            try:
                await self._autonomous_conductor.stop()
            except Exception:
                pass
        if self._snapshot_task is not None:
            self._snapshot_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._snapshot_task
        if self._worker is not None:
            await self._worker
        if self._arm_tasks:
            await asyncio.gather(*self._arm_tasks, return_exceptions=True)
            
        # Final state snapshot
        try:
            from utils.snapshot import save_snapshot
            state_data = {
                "sessions": self._sessions,
                "jobs": self._dump_jobs(),
            }
            save_snapshot(state_data, self.runtime.settings)
        except Exception:
            pass

    def _load_snapshot(self) -> None:
        try:
            from utils.snapshot import load_snapshot
            data = load_snapshot(self.runtime.settings)
            if data:
                sessions = data.get("sessions")
                if sessions:
                    self._sessions.update(sessions)
                jobs = data.get("jobs", [])
                for job_data in jobs:
                    job = ConductorJob(
                        priority=job_data["priority"],
                        sequence=job_data["sequence"],
                        kind=job_data["kind"],
                        payload=job_data["payload"],
                    )
                    job.job_id = job_data["job_id"]
                    try:
                        self._queue.put_nowait(job)
                    except asyncio.QueueFull:
                        pass
        except Exception:
            pass

    def _dump_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        try:
            # Safely grab queue contents
            for job in list(self._queue._queue):
                jobs.append({
                    "priority": job.priority,
                    "sequence": job.sequence,
                    "kind": job.kind,
                    "payload": job.payload,
                    "job_id": job.job_id,
                })
        except Exception:
            pass
        return jobs

    async def _periodic_snapshot(self) -> None:
        from utils.snapshot import save_snapshot
        while not self._stop.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                state_data = {
                    "sessions": self._sessions,
                    "jobs": self._dump_jobs(),
                }
                save_snapshot(state_data, self.runtime.settings)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def submit_chat(self, request: ChatRequest, priority: int = 0) -> ChatResponse:
        inbound = InboundMessage(
            source="direct",
            user_id="local",
            channel="chat",
            text=request.message,
            session_id=request.session_id,
            priority=priority,
        )
        task = BrainTask(inbound=inbound, max_tokens=request.max_tokens)
        return await self.submit_brain(task, priority=priority)

    async def submit_brain(self, task: BrainTask, priority: int = 0) -> ChatResponse:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ChatResponse] = loop.create_future()
        await self._queue.put(
            ConductorJob(
                priority=priority,
                sequence=next(self._sequence),
                kind="brain",
                payload={"task": task.model_dump(mode="json")},
                future=future,
            )
        )
        return await future

    def try_submit_message(self, message: InboundMessage) -> str | None:
        job = ConductorJob(
            priority=message.priority,
            sequence=next(self._sequence),
            kind="message",
            payload={"task": BrainTask(inbound=message, max_tokens=800).model_dump(mode="json")},
        )
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            return None
        return job.job_id

    def is_idle(self) -> bool:
        return self._queue.empty() and not self._arm_tasks

    async def _run(self) -> None:
        await self.run_forever()

    async def run_forever(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            arm = asyncio.create_task(self._handle(job), name=f"conductor-arm-{job.job_id}")
            self._arm_tasks.add(arm)
            arm.add_done_callback(self._arm_tasks.discard)

    async def _handle(self, job: ConductorJob) -> None:
        async with self._arm_semaphore:
            task: BrainTask | None = None
            prepared_task: BrainTask | None = None
            response: ChatResponse | None = None
            error: Exception | None = None
            try:
                await self.event_bus.publish(
                    LisaEvent(
                        type="conductor.job_started",
                        payload={"kind": job.kind, "priority": job.priority, "job_id": job.job_id},
                    )
                )

                if job.kind in {"brain", "message"}:
                    task = BrainTask.model_validate(job.payload["task"])
                    prepared_task = await self._prepare_brain_task(task)
                    response = await self._run_brain_loop(prepared_task)
                elif job.kind == "chat":
                    response = await self.runtime.process_chat(job.payload["request"])
                else:
                    raise ValueError(f"Unsupported job kind: {job.kind}")
            except Exception as exc:  # pragma: no cover - defensive recovery path
                error = exc
            finally:
                await self._enqueue_task_summary(
                    job=job,
                    task=task or self._task_from_job(job),
                    prepared_task=prepared_task,
                    response=response,
                    error_message=str(error) if error is not None else None,
                )
                if error is None and response is not None:
                    if job.future is not None and not job.future.done():
                        job.future.set_result(response)
                    await self.event_bus.publish(
                        LisaEvent(
                            type="conductor.job_finished",
                            payload={
                                "kind": job.kind,
                                "priority": job.priority,
                                "job_id": job.job_id,
                                "session_id": response.session_id,
                            },
                        )
                    )
                elif error is not None:
                    if job.future is not None and not job.future.done():
                        job.future.set_exception(error)
                    await self.event_bus.publish(
                        LisaEvent(
                            type="conductor.job_error",
                            payload={"kind": job.kind, "error": str(error), "job_id": job.job_id},
                        )
                    )

    async def _prepare_brain_task(self, task: BrainTask) -> BrainTask:
        inbound = task.inbound
        session_id = inbound.session_id or str(uuid4())

        await self.runtime.notepad_writer.flush_pending()

        constitution_state = self.runtime.notepad.get_constitution_state()
        constitution = task.constitution or constitution_state["mode"]
        persona_weights = task.persona_weights or self.runtime.gating.predict_blend(inbound.text)

        if task.context_summary:
            context_summary = task.context_summary
        else:
            context_entries = await asyncio.to_thread(self.runtime.notepad.search, inbound.text, 5)
            context_summary = [
                {
                    "entry_type": row["entry_type"],
                    "payload": row["payload"],
                    "constitution": row["constitution"],
                }
                for row in context_entries
            ]

        self._session(session_id).constitution = constitution
        self._session(session_id).persona_weights = persona_weights
        self._session(session_id).last_context_summary = context_summary

        return task.model_copy(
            update={
                "inbound": inbound.model_copy(update={"session_id": session_id}),
                "constitution": constitution,
                "persona_weights": persona_weights,
                "context_summary": context_summary,
            }
        )

    async def _run_brain_loop(self, task: BrainTask) -> ChatResponse:
        session_id = task.inbound.session_id or str(uuid4())
        current_task = task
        response: ChatResponse | None = None
        timeout_seconds = getattr(getattr(self.runtime, "settings", None), "tool_timeout_seconds", 30)

        for turn_index in range(self.max_follow_ups + 1):
            try:
                response = await asyncio.wait_for(
                    self.runtime.process_brain_task(current_task),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                response = ChatResponse(
                    session_id=session_id,
                    message=(
                        f"Reasoning timed out after {timeout_seconds} seconds. "
                        "Please narrow the request or retry with fewer tool steps."
                    ),
                    constitution=(current_task.constitution or self.runtime.notepad.get_constitution_state()["mode"]),
                    personas=current_task.persona_weights or self.runtime.gating.predict_blend(current_task.inbound.text),
                    tool_suggestions=[],
                    tool_calls=[],
                    used_external_model=False,
                    notes=["Brain execution timed out before a final answer was produced."],
                )
            session = self._session(session_id)
            session.turn_index = turn_index + 1
            session.constitution = response.constitution
            session.persona_weights = response.personas
            session.last_response = response.message
            session.last_tool_results = current_task.tool_results
            session.last_context_summary = current_task.context_summary

            if not response.tool_calls:
                break

            tool_results = await self._execute_tool_calls(
                tool_calls=response.tool_calls,
                constitution=response.constitution,
                session_id=session_id,
            )
            session.last_tool_results = tool_results
            if not tool_results:
                break

            follow_up_text = self._build_follow_up_text(
                inbound=current_task.inbound,
                assistant_message=response.message,
                tool_results=tool_results,
            )
            follow_up_inbound = current_task.inbound.model_copy(
                update={
                    "source": "tool",
                    "text": follow_up_text,
                    "timestamp": datetime.now(timezone.utc),
                }
            )
            current_task = current_task.model_copy(
                update={
                    "inbound": follow_up_inbound,
                    "tool_results": tool_results,
                    "follow_up": True,
                    "constitution": response.constitution,
                    "persona_weights": response.personas,
                }
            )

        assert response is not None
        return response

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        constitution: str,
        session_id: str,
    ) -> list[ToolResult]:
        results = await self.tool_executor.execute_many(
            tool_calls=tool_calls,
            constitution=ConstitutionMode(constitution),
            session_id=session_id,
        )

        await self.event_bus.publish(
            LisaEvent(
                type="conductor.tool_results",
                payload={"results": [result.model_dump() for result in results]},
            )
        )
        return results

    @staticmethod
    def _build_follow_up_text(
        inbound: InboundMessage,
        assistant_message: str,
        tool_results: list[ToolResult],
    ) -> str:
        lines = [
            "Tool execution completed. Continue the same session using the results below.",
            f"Original user message: {inbound.text}",
            f"Previous assistant message: {assistant_message}",
            "Tool results:",
        ]
        for result in tool_results:
            if result.success:
                lines.append(f"- {result.tool}: success -> {result.output}")
            else:
                lines.append(f"- {result.tool}: error -> {result.error}")
        return "\n".join(lines)

    def _session(self, session_id: str) -> ConductorSessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = ConductorSessionState(session_id=session_id)
            self._sessions[session_id] = state
        return state

    def _task_from_job(self, job: ConductorJob) -> BrainTask | None:
        if job.kind not in {"brain", "message"}:
            return None
        return BrainTask.model_validate(job.payload["task"])

    async def _enqueue_task_summary(
        self,
        job: ConductorJob,
        task: BrainTask | None,
        prepared_task: BrainTask | None,
        response: ChatResponse | None,
        error_message: str | None,
    ) -> None:
        if task is None:
            return

        inbound = (prepared_task or task).inbound
        session_id = inbound.session_id or task.inbound.session_id or job.job_id
        tool_calls = response.tool_calls if response is not None else []
        tool_results = prepared_task.tool_results if prepared_task is not None else []
        outcome = "success" if error_message is None else "error"
        summary_payload = {
            "session_id": session_id,
            "job_id": job.job_id,
            "source": inbound.source,
            "channel": inbound.channel,
            "user_id": inbound.user_id,
            "input": inbound.text,
            "user_input": inbound.text,
            "output": response.message if response is not None else None,
            "response": response.message if response is not None else None,
            "tool_calls": [call.model_dump() for call in tool_calls],
            "tool_results": [result.model_dump() for result in tool_results],
            "tools_used": [result.tool for result in tool_results] if tool_results else [call.name for call in tool_calls],
            "outcome": outcome,
            "error": error_message,
            "self_critique": self._build_self_critique(response, error_message, tool_results),
            "turn_index": self._session(session_id).turn_index,
            "constitution": response.constitution if response is not None else (prepared_task.constitution if prepared_task else task.constitution),
            "persona_blend": response.personas if response is not None else (prepared_task.persona_weights if prepared_task else task.persona_weights),
            "persona_weights": response.personas if response is not None else (prepared_task.persona_weights if prepared_task else task.persona_weights),
        }

        try:
            summary_future = await self.runtime.notepad_writer.enqueue(
                entry_type="task_summary",
                payload=summary_payload,
                constitution=(response.constitution if response is not None else (prepared_task.constitution if prepared_task else task.constitution)) or self.runtime.notepad.get_constitution_state()["mode"],
                personas=response.personas if response is not None else (prepared_task.persona_weights if prepared_task else task.persona_weights) or {},
            )
            await summary_future
        except Exception as exc:  # pragma: no cover - queue shutdown edge
            await self.event_bus.publish(
                LisaEvent(
                    type="conductor.summary_error",
                    payload={"job_id": job.job_id, "error": str(exc)},
                )
            )

    @staticmethod
    def _build_self_critique(
        response: ChatResponse | None,
        error_message: str | None,
        tool_results: list[ToolResult],
    ) -> str:
        if error_message is not None:
            return f"Task ended with error: {error_message}"
        notes = response.notes if response is not None else []
        if notes:
            return "; ".join(notes)
        if tool_results:
            failed = [result.tool for result in tool_results if not result.success]
            if failed:
                return f"Tool failures detected for: {', '.join(failed)}"
            return f"Executed {len(tool_results)} tool result(s) without explicit critique."
        return "No explicit self-critique was produced."
