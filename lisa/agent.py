from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from lisa.events import EventBus, LisaEvent
from lisa.notepad import AsyncNotepadWriter
from lisa.config import Settings
from lisa.constitutions import ConstitutionMode, parse_constitution_command
from lisa.gating import PersonaGatingNetwork
from lisa.llm import LLMClient
from lisa.notepad import Notepad
from lisa.schemas import BrainTask, ChatRequest, ChatResponse, InboundMessage
from lisa.tool_executor import ToolExecutor
from lisa.tools import ToolRegistry
from personal.context_store import PersonalContextStore
from safety.input_sanitizer import is_prompt_injection_suspicious, sanitize_text


class LisaRuntime:
    def __init__(
        self,
        settings: Settings,
        notepad: Notepad,
        llm_client: LLMClient,
        tools: ToolRegistry,
        tool_executor: ToolExecutor,
        event_bus: EventBus,
        notepad_writer: AsyncNotepadWriter,
        gating: PersonaGatingNetwork,
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
        return await self.process_brain_task(BrainTask(inbound=inbound, max_tokens=request.max_tokens))

    async def process_message(
        self,
        inbound: InboundMessage,
        max_tokens: int = 800,
    ) -> ChatResponse:
        return await self.process_brain_task(BrainTask(inbound=inbound, max_tokens=max_tokens))

    async def process_brain_task(self, task: BrainTask) -> ChatResponse:
        inbound = task.inbound
        session_id = inbound.session_id or str(uuid4())
        sanitized_text = sanitize_text(inbound.text)
        if is_prompt_injection_suspicious(sanitized_text):
            await self.notepad_writer.enqueue(
                entry_type="interaction",
                payload={
                    "session_id": session_id,
                    "source": inbound.source,
                    "user_id": inbound.user_id,
                    "channel": inbound.channel,
                    "user_message": inbound.text,
                    "assistant_message": "I can help with the task itself, but I won't follow instructions that try to override safety or system behavior.",
                    "tool_suggestions": [],
                    "notes": ["Rejected suspicious prompt-injection content."],
                    "context": [],
                    "tool_results": [],
                },
                constitution=ConstitutionMode.RESTRICTED,
                personas={"guardian": 1.0},
            )
            return ChatResponse(
                session_id=session_id,
                message=(
                    "I can help with the task itself, but I won't follow instructions "
                    "that try to override safety or system behavior."
                ),
                constitution=ConstitutionMode.RESTRICTED.value,
                personas={"guardian": 1.0},
                tool_suggestions=[],
                tool_calls=[],
                used_external_model=False,
                notes=["Suspicious prompt content rejected."],
            )
        current_mode = (
            ConstitutionMode(task.constitution)
            if task.constitution is not None
            else ConstitutionMode(self.notepad.get_constitution_state()["mode"])
        )
        personas = task.persona_weights or self.gating.predict_blend(sanitized_text)
        notes: list[str] = []
        used_external_model = False
        tool_calls: list = []
        trace_id = str(uuid4())

        await self.event_bus.publish(
            LisaEvent(
                type="chat.received",
                payload={
                    "source": inbound.source,
                    "user_id": inbound.user_id,
                    "channel": inbound.channel,
                    "text": inbound.text,
                },
                trace_id=trace_id,
                session_id=session_id,
            )
        )

        await self.notepad_writer.enqueue(
            entry_type="interaction_start",
            payload={
                "session_id": session_id,
                "source": inbound.source,
                "user_id": inbound.user_id,
                "channel": inbound.channel,
                "user_message": inbound.text,
                "timestamp": inbound.timestamp.isoformat(),
            },
            constitution=current_mode,
            personas=personas,
        )

        await self.notepad_writer.flush_pending()
        if task.context_summary:
            context_summary = task.context_summary
        else:
            context_entries = await asyncio.to_thread(self.notepad.search, sanitized_text, 5)
            context_summary = [
                {
                    "entry_type": row["entry_type"],
                    "payload": row["payload"],
                    "constitution": row["constitution"],
                }
                for row in context_entries
            ]

        constitution_command = parse_constitution_command(inbound.text)
        if constitution_command is not None:
            if constitution_command.target_mode == ConstitutionMode.UNRESTRICTED:
                if not constitution_command.reason:
                    reply_text = (
                        "Unrestricted mode was not enabled because a reason is required. "
                        "Use: ENABLE UNRESTRICTED MODE [reason]"
                    )
                    notes.append("Constitution switch rejected because the reason was missing.")
                else:
                    await asyncio.to_thread(
                        self.notepad.set_constitution_mode,
                        ConstitutionMode.UNRESTRICTED,
                        constitution_command.reason,
                    )
                    current_mode = ConstitutionMode.UNRESTRICTED
                    reply_text = (
                        "Unrestricted mode enabled for this workspace. "
                        "The switch was logged in the Notepad and should surface as a dashboard warning."
                    )
                    notes.append(f"Reason logged: {constitution_command.reason}")
            else:
                await asyncio.to_thread(
                    self.notepad.set_constitution_mode,
                    ConstitutionMode.RESTRICTED,
                    "User disabled lab mode",
                )
                current_mode = ConstitutionMode.RESTRICTED
                reply_text = "Restricted mode restored. Safe defaults are active again."
        else:
            system_prompt = self._build_system_prompt(current_mode, personas, context_summary, sanitized_text)
            if task.tool_results:
                tool_result_summary = [
                    {
                        "tool": result.tool,
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                        "attempt": result.attempt,
                    }
                    for result in task.tool_results
                ]
                system_prompt += f" Tool results from the previous arm: {tool_result_summary}."
            if self.llm_client.configured or self.llm_client.supports_local_generation:
                generation = await self.llm_client.generate_brain(
                    system_prompt=system_prompt,
                    user_prompt=sanitized_text,
                    max_tokens=task.max_tokens,
                    persona_weights=personas,
                )
                reply_text = generation.text
                tool_calls = generation.tool_calls
                used_external_model = not generation.used_local_model
                if tool_calls:
                    notes.append(f"Model emitted {len(tool_calls)} tool call(s).")
            else:
                reply_text = self._bootstrap_reply(current_mode, personas)
                notes.append(
                    "No external model is configured yet; returning the deterministic bootstrap response."
                )

        tool_suggestions = self._suggest_tools(inbound.text)
        final_future = await self.notepad_writer.enqueue(
            entry_type="interaction",
            payload={
                "session_id": session_id,
                "source": inbound.source,
                "user_id": inbound.user_id,
                "channel": inbound.channel,
                "user_message": inbound.text,
                "assistant_message": reply_text,
                "tool_suggestions": tool_suggestions,
                "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
                "notes": notes,
                "context": context_summary,
                "tool_results": [tool_result.model_dump() for tool_result in task.tool_results],
            },
            constitution=current_mode,
            personas=personas,
        )
        await final_future

        await self.event_bus.publish(
            LisaEvent(
                type="chat.responded",
                payload={
                    "message": reply_text,
                    "notes": notes,
                    "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
                    "tool_results": [tool_result.model_dump() for tool_result in task.tool_results],
                    "personas": personas,
                    "session_id": session_id,
                },
                trace_id=trace_id,
                session_id=session_id,
            )
        )

        return ChatResponse(
            session_id=session_id,
            message=reply_text,
            constitution=current_mode.value,
            personas=personas,
            tool_suggestions=tool_suggestions,
            tool_calls=tool_calls,
            used_external_model=used_external_model,
            notes=notes,
        )

    def _build_system_prompt(
        self,
        constitution: ConstitutionMode,
        personas: dict[str, float],
        context_summary: list[dict[str, object]],
        user_text: str,
    ) -> str:
        tool_names = ", ".join(tool.name for tool in self.tools.list_tools())
        personal_summary = self.personal_store.summary() if self.personal_store is not None else {}
        return (
            "You are LISA, a proactive developer agent. "
            f"Current constitution: {constitution.value}. "
            f"Persona blend weights: {personas}. "
            f"Recent memory context: {context_summary}. "
            f"Personal context: {personal_summary}. "
            f"Sanitized user request: {user_text}. "
            "Respect the current constitution strictly, keep user data local by default, "
            "and prefer production-ready engineering outputs. "
            f"Available tools: {tool_names}."
        )

    @staticmethod
    def _bootstrap_reply(
        constitution: ConstitutionMode,
        personas: dict[str, float],
    ) -> str:
        dominant = max(personas, key=personas.get)
        return (
            f"LISA core is running in {constitution.value} mode. "
            f"The current dominant cognitive blend is '{dominant}'. "
            "The ledger, constitution state, and tool registry are live. "
            "Configure LISA_MODEL_PROVIDER, LISA_MODEL_NAME, LISA_MODEL_BASE_URL, "
            "and LISA_MODEL_API_KEY to attach an external reasoning model."
        )

    @staticmethod
    def _suggest_tools(message: str) -> list[str]:
        lowered = message.lower()
        suggestions: list[str] = []
        if any(keyword in lowered for keyword in ("search", "remember", "history", "notepad")):
            suggestions.append("search_notepad")
        if any(keyword in lowered for keyword in ("file", "write", "edit", "read")):
            suggestions.extend(["file_read", "file_write", "file_edit"])
        if any(keyword in lowered for keyword in ("run", "command", "shell", "terminal")):
            suggestions.append("terminal_exec")
        if any(keyword in lowered for keyword in ("web", "browser", "docs", "search online")):
            suggestions.extend(["browser_search", "browser_fetch"])
        if any(keyword in lowered for keyword in ("metric", "dashboard")):
            suggestions.append("dashboard_update")

        if not suggestions:
            suggestions.append("search_notepad")
        return list(dict.fromkeys(suggestions))
