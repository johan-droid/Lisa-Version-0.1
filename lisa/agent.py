from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from lisa.constitutions import ConstitutionMode, parse_constitution_command
from lisa.events import EventBus, LisaEvent
from lisa.notepad import AsyncNotepadWriter, Notepad
from lisa.memory_system import HybridMemoryCoordinator
from lisa.react_engine import ReActEngine
from lisa.router import LLMRouter
from lisa.schemas import (
    BrainTask,
    ChatRequest,
    ChatResponse,
    EnrichedTask,
    InboundMessage,
    TaskResult,
)
from personal.context_store import PersonalContextStore
from safety.input_sanitizer import (
    inspect_query_text,
    inspect_text,
    sanitize_text,
    sanitize_user_visible_text,
)


class LisaRuntime:
    def __init__(
        self,
        *,
        settings,
        notepad: Notepad,
        llm_client,
        tools,
        tool_executor,
        event_bus: EventBus,
        notepad_writer: AsyncNotepadWriter,
        gating,
        memory=None,
        llm_router: LLMRouter | None = None,
        react_engine=None,
        evolution_engine=None,
        personal_store: PersonalContextStore | None = None,
    ):
        self.settings = settings
        self.notepad = notepad
        self.llm_client = llm_client
        self.tools = tools
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.notepad_writer = notepad_writer
        self.gating = gating
        self.memory = memory or HybridMemoryCoordinator(
            agent_id=str(getattr(settings, "agent_id", "lisa") or "lisa"),
            namespace=str(
                getattr(notepad, "db_path", settings.workspace_root / "memory.db")
            ),
            redis_url=getattr(settings, "redis_url", None),
            postgres_dsn=getattr(settings, "postgres_dsn", None),
            chroma_persist_dir=getattr(settings, "chroma_persist_dir", None),
            working_ttl_seconds=int(
                getattr(settings, "working_memory_ttl_seconds", 7200)
            ),
            event_bus=event_bus,
        )
        self.llm_router = llm_router or LLMRouter(llm_client)
        self.react_engine = react_engine or ReActEngine(
            llm_router=self.llm_router,
            llm_client=llm_client,
            tool_executor=tool_executor,
            memory=self.memory,
            event_bus=event_bus,
        )
        self.evolution_engine = evolution_engine
        self.personal_store = personal_store

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        inbound = InboundMessage(
            source="direct",
            user_id="local",
            channel="chat",
            text=request.message,
            timestamp=datetime.now(timezone.utc),
            session_id=request.session_id,
            priority=0,
        )
        return await self.process_brain_task(
            BrainTask(inbound=inbound, max_tokens=request.max_tokens)
        )

    async def process_message(
        self, inbound: InboundMessage, max_tokens: int = 800
    ) -> ChatResponse:
        return await self.process_brain_task(
            BrainTask(inbound=inbound, max_tokens=max_tokens)
        )

    async def process_brain_task(self, task: BrainTask) -> ChatResponse:
        task_id = task.inbound.session_id or str(uuid4())
        enriched = await self.enrich_task(task_id=task_id, task=task)
        return await self.process_enriched_task(enriched)

    async def enrich_task(self, *, task_id: str, task: BrainTask) -> EnrichedTask:
        inbound = task.inbound.model_copy(update={"session_id": task_id})
        sanitized_text = sanitize_text(inbound.text)
        inspection = inspect_text(sanitized_text)
        query_inspection = inspect_query_text(sanitized_text)
        if inspection.suspicious:
            await self.event_bus.publish(
                LisaEvent(
                    type="security.prompt_flagged",
                    payload={
                        "task_id": task_id,
                        "risk_score": inspection.risk_score,
                        "reasons": inspection.reasons,
                    },
                    session_id=task_id,
                    trace_id=task_id,
                )
            )
        if query_inspection.suspicious:
            await self.event_bus.publish(
                LisaEvent(
                    type="security.query_flagged",
                    payload={
                        "task_id": task_id,
                        "reasons": query_inspection.reasons,
                    },
                    session_id=task_id,
                    trace_id=task_id,
                )
            )

        constitution_state = self.notepad.get_constitution_state()
        constitution = task.constitution or str(
            constitution_state["mode"] or ConstitutionMode.RESTRICTED.value
        )
        persona_weights = task.persona_weights or self.gating.predict_blend(
            sanitized_text
        )
        memory_context = await self.memory.enrich_task(
            task_id=task_id, description=sanitized_text
        )
        personal_summary = (
            self.personal_store.summary() if self.personal_store is not None else {}
        )
        route = await self.llm_router.route(
            sanitized_text,
            {
                "memory_context": memory_context.similar_episodes,
                "skill_context": memory_context.relevant_skills,
                "personal_summary": personal_summary,
            },
        )
        return EnrichedTask(
            task_id=task_id,
            agent_id=self.memory.agent_id,
            inbound=inbound.model_copy(update={"text": sanitized_text}),
            description=sanitized_text,
            max_tokens=task.max_tokens,
            constitution=constitution,
            persona_weights=persona_weights,
            memory_context=memory_context.similar_episodes,
            skill_context=memory_context.relevant_skills,
            working_memory_key=memory_context.working_memory_key,
            stress_level=task.stress_level,
            metadata={
                "task_type": route.task_type,
                "route_brain": route.brain,
                "route_reason": route.reason,
                "personal_context": personal_summary,
                "approved_levels": list(
                    inbound.metadata.get("approved_levels") or ["L0", "L1"]
                ),
                "explicit_user_grant": bool(
                    inbound.metadata.get("explicit_user_grant", False)
                ),
                "two_factor_confirmed": bool(
                    inbound.metadata.get("two_factor_confirmed", False)
                ),
            },
        )

    async def process_enriched_task(self, enriched_task: EnrichedTask) -> ChatResponse:
        constitution_command = parse_constitution_command(enriched_task.inbound.text)
        if constitution_command is not None:
            target_mode = constitution_command.target_mode
            reason = (
                constitution_command.reason
                if target_mode == ConstitutionMode.UNRESTRICTED
                else "User disabled lab mode"
            )
            if (
                target_mode == ConstitutionMode.UNRESTRICTED
                and not constitution_command.reason
            ):
                return ChatResponse(
                    session_id=enriched_task.task_id,
                    message="Unrestricted mode was not enabled because an explicit reason is required.",
                    constitution=ConstitutionMode.RESTRICTED.value,
                    personas={"guardian": 1.0},
                    tool_suggestions=[],
                    used_external_model=False,
                    notes=[
                        "Constitution switch rejected because the reason was missing."
                    ],
                )
            self.notepad.set_constitution_mode(target_mode, reason)
            return ChatResponse(
                session_id=enriched_task.task_id,
                message="Constitution updated.",
                constitution=target_mode.value,
                personas=enriched_task.persona_weights,
                tool_suggestions=[],
                used_external_model=False,
                notes=[f"Constitution switched to {target_mode.value}."],
            )

        await self.event_bus.publish(
            LisaEvent(
                type="chat.received",
                payload={
                    "source": enriched_task.inbound.source,
                    "user_id": enriched_task.inbound.user_id,
                    "channel": enriched_task.inbound.channel,
                    "task_id": enriched_task.task_id,
                },
                session_id=enriched_task.task_id,
                trace_id=enriched_task.task_id,
            )
        )

        result = await self.react_engine.run(enriched_task)
        response = self._result_to_response(enriched_task, result)
        safe_answer = response.message
        if not result.success and self.evolution_engine is not None:
            triggered = await self.evolution_engine.note_task_failure(
                str(enriched_task.metadata.get("task_type") or "react_task")
            )
            if triggered:
                response.notes.append(
                    "Emergency evolution trigger fired after repeated failures."
                )
        await self.notepad_writer.enqueue(
            entry_type="interaction",
            payload={
                "session_id": enriched_task.task_id,
                "source": enriched_task.inbound.source,
                "user_id": enriched_task.inbound.user_id,
                "channel": enriched_task.inbound.channel,
                "user_message": enriched_task.inbound.text,
                "assistant_message": safe_answer,
                "tool_calls": [tool.model_dump() for tool in result.tool_results],
                "notes": list(response.notes),
                "task_type": enriched_task.metadata.get("task_type"),
                "outcome": "success" if result.success else "error",
                "error": result.failure_reason,
                "persona_blend": enriched_task.persona_weights,
            },
            constitution=response.constitution,
            personas=enriched_task.persona_weights,
        )
        await self.event_bus.publish(
            LisaEvent(
                type="chat.responded",
                payload={
                    "message": response.message,
                    "personas": response.personas,
                    "tool_calls": [tool.model_dump() for tool in response.tool_calls],
                    "session_id": response.session_id,
                    "delivery_hints": response.delivery_hints,
                },
                session_id=response.session_id,
                trace_id=enriched_task.task_id,
            )
        )
        return response

    def _result_to_response(
        self, enriched_task: EnrichedTask, result: TaskResult
    ) -> ChatResponse:
        route_brain = str(enriched_task.metadata.get("route_brain") or "")
        used_external_model = route_brain != "tinyllama"
        if (
            route_brain == "tinyllama"
            and not self.llm_client.local_backend_ready
            and self.llm_client.external_backend_configured
        ):
            used_external_model = True
        safe_answer = sanitize_user_visible_text(result.answer, max_length=8_000)
        return ChatResponse(
            session_id=enriched_task.task_id,
            message=safe_answer,
            constitution=enriched_task.constitution,
            personas=enriched_task.persona_weights,
            tool_suggestions=[
                str(item.get("skill_name") or "")
                for item in enriched_task.skill_context
                if item.get("skill_name")
            ],
            tool_calls=[],
            used_external_model=used_external_model,
            notes=[
                f"route={route_brain}",
                f"task_type={enriched_task.metadata.get('task_type')}",
                *(
                    ["failure=" + str(result.failure_reason)]
                    if result.failure_reason
                    else []
                ),
            ],
            delivery_hints={"scratchpad_stream": True},
        )
