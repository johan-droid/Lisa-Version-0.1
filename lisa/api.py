from __future__ import annotations

import logging
from contextlib import asynccontextmanager, suppress
from datetime import datetime
import asyncio
import json
import uuid
import time
import traceback
from typing import Any
import httpx
from time import perf_counter

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.websockets import WebSocketDisconnect
from fastapi import WebSocket

from lisa.agent import LisaRuntime
from lisa.conductor import TaskConductor
from lisa.config import Settings, get_settings
from lisa.constitutions import ConstitutionMode
from lisa.evolution import NightlyEvolutionScheduler
from lisa.gating import PersonaGatingNetwork
from lisa.memory_system import HybridMemoryCoordinator
from lisa.events import EventBus, LisaEvent
from lisa.llm import LLMClient
from lisa.router import LLMRouter
from lisa.react_engine import ReActEngine
from lisa.evolution_engine import EvolutionEngine
from lisa.hub import MessageHub
from lisa.notepad import AsyncNotepadWriter, Notepad
from lisa.schemas import (
    BotDispatchRequest,
    ChatRequest,
    ChatResponse,
    ConstitutionStateResponse,
    BotDispatchResponse,
    ChannelAccessRequest,
    ChannelAccessResponse,
    ChannelCapabilitiesResponse,
    DashboardMetric,
    HubAck,
    InboundMessage,
    NotepadEntry,
    NotepadSearchResponse,
    ToolInvokeRequest,
    ToolInvokeResponse,
    ToolSummary,
)
from lisa.tool_executor import ToolExecutor
from lisa.tools import ToolRegistry
from personal import (
    CalendarAwareness,
    PersonalContextStore,
    ReminderScheduler,
    StyleLearner,
    WellnessTracker,
)
from safety.admin_auth import require_admin_request
from safety.input_sanitizer import ensure_body_size
from safety.replay_guard import ReplayAttackDetected
from utils.observability import bind_runtime_context

LOGGER = logging.getLogger("lisa.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    event_bus = EventBus()

    app = FastAPI(
        title=settings.app_name, version="1.0.0", docs_url=None, redoc_url=None
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        LOGGER.error(f"Unhandled exception on {request.url}: {exc}")
        LOGGER.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": str(exc)},
        )

    memory = HybridMemoryCoordinator(
        agent_id=str(getattr(settings, "agent_id", "lisa") or "lisa"),
        namespace=str(settings.db_path.resolve()),
        redis_url=getattr(settings, "redis_url", None),
        postgres_dsn=getattr(settings, "postgres_dsn", None),
        chroma_persist_dir=getattr(settings, "chroma_persist_dir", None),
        working_ttl_seconds=int(getattr(settings, "working_memory_ttl_seconds", 7200)),
        event_bus=event_bus,
    )
    notepad = Notepad(settings, event_bus=event_bus)
    notepad.memory = memory
    notepad.startup_maintenance(
        settings.backup_dir,
        settings.notepad_retention_days,
        settings.notepad_backup_keep,
    )
    gating = PersonaGatingNetwork.load_or_initialize(settings.gating_model_path)
    llm_client = LLMClient(settings)
    notepad_writer = AsyncNotepadWriter(notepad=notepad, event_bus=event_bus)
    personal_store = (
        PersonalContextStore(settings.personal_db_path)
        if settings.enable_personal_features
        else None
    )
    reminder_scheduler = (
        ReminderScheduler(store=personal_store, event_bus=event_bus)
        if personal_store is not None
        else None
    )
    calendar_awareness = CalendarAwareness(
        path=settings.workspace_root / "data" / "calendar.json"
    )
    calendar_awareness.load()
    style_learner = StyleLearner()
    wellness_tracker = WellnessTracker()
    wellness_tracker.start_session()
    tools = ToolRegistry(
        settings=settings,
        notepad=notepad,
        llm_client=llm_client,
        event_bus=event_bus,
        notepad_writer=notepad_writer,
    )
    tool_executor = ToolExecutor(registry=tools, event_bus=event_bus, max_workers=10)
    llm_router = LLMRouter(llm_client)
    react_engine = ReActEngine(
        llm_router=llm_router,
        llm_client=llm_client,
        tool_executor=tool_executor,
        memory=memory,
        event_bus=event_bus,
    )
    evolution_engine = EvolutionEngine(
        memory=memory,
        llm_client=llm_client,
        tool_executor=tool_executor,
        event_bus=event_bus,
        agent_id=str(getattr(settings, "agent_id", "lisa") or "lisa"),
    )
    runtime = LisaRuntime(
        settings=settings,
        notepad=notepad,
        llm_client=llm_client,
        tools=tools,
        tool_executor=tool_executor,
        event_bus=event_bus,
        notepad_writer=notepad_writer,
        gating=gating,
        memory=memory,
        llm_router=llm_router,
        react_engine=react_engine,
        evolution_engine=evolution_engine,
        personal_store=personal_store,
    )
    conductor = TaskConductor(
        runtime=runtime,
        tool_executor=tool_executor,
        event_bus=event_bus,
        queue_size=settings.incoming_queue_size,
    )
    evolution_scheduler = NightlyEvolutionScheduler(
        runtime=runtime,
        conductor=conductor,
        event_bus=event_bus,
    )
    message_hub = MessageHub(
        settings=settings,
        event_bus=event_bus,
        conductor=conductor,
        personal_store=personal_store,
        capabilities_provider=lambda: [
            tool.name for tool in runtime.tools.list_tools()
        ],
    )
    wellness_task: asyncio.Task[None] | None = None

    async def _wellness_loop() -> None:
        while True:
            await asyncio.sleep(max(60, int(settings.proactive_checkin_minutes) * 30))
            if wellness_tracker.should_check_in():
                await event_bus.publish(
                    LisaEvent(
                        type="personal.wellness_checkin",
                        payload={
                            "message": "You have been coding for a while. Consider taking a short break.",
                        },
                    )
                )
                wellness_tracker.start_session()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await notepad_writer.start()
        await memory.start()
        await tool_executor.start()
        await conductor.start()
        await evolution_scheduler.start()
        if reminder_scheduler is not None:
            await reminder_scheduler.start()
        nonlocal wellness_task
        if settings.enable_personal_features:
            wellness_task = asyncio.create_task(_wellness_loop(), name="lisa-wellness")
        await message_hub.start()
        try:
            yield
        finally:
            await message_hub.close()
            if reminder_scheduler is not None:
                await reminder_scheduler.close()
            if wellness_task is not None:
                wellness_task.cancel()
                with suppress(asyncio.CancelledError):
                    await wellness_task
            await evolution_scheduler.close()
            await conductor.close()
            await tool_executor.close()
            await memory.close()
            await notepad_writer.close()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.settings = settings
    app.state.event_bus = event_bus
    app.state.notepad_writer = notepad_writer
    app.state.tool_executor = tool_executor
    app.state.conductor = conductor
    app.state.gating = gating
    app.state.memory = memory
    app.state.llm_router = llm_router
    app.state.react_engine = react_engine
    app.state.evolution_engine = evolution_engine
    app.state.message_hub = message_hub
    app.state.channel_access = message_hub.channel_access
    app.state.evolution_scheduler = evolution_scheduler
    app.state.personal_store = personal_store
    app.state.reminder_scheduler = reminder_scheduler
    app.state.calendar_awareness = calendar_awareness
    app.state.style_learner = style_learner
    app.state.wellness_tracker = wellness_tracker

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started_at = perf_counter()
        bind_runtime_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None,
        )
        response: Response | None = None
        try:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > int(
                        getattr(settings, "max_request_body_bytes", 262_144)
                    ):
                        response = JSONResponse(
                            status_code=413,
                            content={"detail": "Request body is too large."},
                        )
                        return response
                except ValueError:
                    response = JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length header."},
                    )
                    return response
            response = await call_next(request)
            return response
        finally:
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            LOGGER.info(
                "request.completed method=%s path=%s status=%s duration_ms=%s request_id=%s",
                request.method,
                request.url.path,
                getattr(response, "status_code", "error"),
                duration_ms,
                request_id,
            )
            if response is not None:
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["Referrer-Policy"] = "no-referrer"
                response.headers["Content-Security-Policy"] = (
                    "default-src 'self'; img-src 'self' data:; connect-src 'self' https: wss:; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'self'"
                )

    @app.post("/bots/connect", response_model=ChannelAccessResponse, deprecated=True)
    async def connect_bot(
        access_request: ChannelAccessRequest, raw_request: Request
    ) -> ChannelAccessResponse:
        require_admin_request(raw_request, settings)
        record = message_hub.channel_access.grant(
            access_request.source, access_request.user_id
        )
        return ChannelAccessResponse(
            source=record.source,
            user_ids=message_hub.channel_access.summary().get(record.source, []),
            updated=True,
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        state = runtime.notepad.get_constitution_state()
        capabilities = message_hub.channel_gateway.capabilities()
        local_backend = runtime.llm_client.local_backend_status()
        listener_mode = (
            "standalone"
            if settings.message_hub_enabled and settings.message_hub_start_listener
            else "embedded_only" if settings.message_hub_enabled else "disabled"
        )
        return {
            "status": "ok",
            "constitution": state["mode"],
            "components": {
                "notepad": "ok",
                "conductor_started": bool(getattr(conductor, "_workers", [])),
                "tool_executor_started": bool(getattr(tool_executor, "_workers", [])),
                "message_hub": listener_mode,
                "local_model": local_backend,
                "external_model_configured": runtime.llm_client.external_backend_configured,
                "hybrid_brain_enabled": bool(settings.hybrid_brain_enabled),
                "configured_channels": list(
                    capabilities.get("configured_channels") or []
                ),
                "evolution_enabled": bool(settings.evolution_enabled),
                "autonomous_enabled": bool(settings.autonomous_enabled),
            },
        }

    @app.post("/shutdown", deprecated=True)
    async def shutdown(request: Request) -> dict[str, str]:
        require_admin_request(request, settings, unsafe_only=True)

        import os
        import signal

        def kill_self():
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGINT)

        asyncio.create_task(asyncio.to_thread(kill_self))
        return {"status": "shutdown_initiated"}

    @app.post("/shed_memory", deprecated=True)
    async def shed_memory(request: Request) -> dict[str, str]:
        require_admin_request(request, settings, unsafe_only=True)

        # 1. Clear browser cache
        try:
            async with runtime.tools._web_cache_lock:
                runtime.tools._web_cache.clear()
        except Exception:
            pass

        # 2. Offload oldest notepad entries (maintenance)
        try:
            runtime.notepad.startup_maintenance(
                settings.backup_dir, retention_days=15, keep_latest_backups=2
            )
        except Exception:
            pass

        # 3. Reduce LLM context window from 2048 to 1024
        try:
            if hasattr(runtime.llm_client, "local_backend") and hasattr(
                runtime.llm_client.local_backend, "_llama"
            ):
                backend = runtime.llm_client.local_backend
                if backend._llama is not None:
                    from lisa.local_inference import PersonaGatedModel

                    # Free existing model to trigger GC
                    backend._llama = None
                    import gc

                    gc.collect()
                    # Reinitialize with smaller context
                    new_backend = PersonaGatedModel(
                        model_path=settings.local_model_path,
                        persona_bank=backend.persona_bank,
                        context_size=1024,
                        n_threads=settings.local_model_n_threads,
                        n_gpu_layers=settings.local_model_n_gpu_layers,
                    )
                    runtime.llm_client.local_backend = new_backend
        except Exception:
            pass

        # 4. Reset tool executor to close down any hanging worker processes
        try:
            await tool_executor.close()
            await tool_executor.start()
        except Exception:
            pass

        # 5. Pause evolution scheduler
        try:
            await evolution_scheduler.close()
        except Exception:
            pass

        return {"status": "shedding_complete"}

    @app.post("/admin/runtime/shutdown")
    async def admin_shutdown(request: Request) -> dict[str, str]:
        return await shutdown(request)

    @app.post("/admin/runtime/shed-memory")
    async def admin_shed_memory(request: Request) -> dict[str, str]:
        return await shed_memory(request)

    @app.get("/state", response_model=ConstitutionStateResponse)
    async def state() -> ConstitutionStateResponse:
        await runtime.notepad_writer.flush_pending()
        current = runtime.notepad.get_constitution_state()
        return ConstitutionStateResponse(
            mode=current["mode"],
            reason=current["reason"],
            updated_at=datetime.fromisoformat(current["updated_at"]),
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        return await conductor.submit_chat(request)

    def _extract_user_message(messages: Any) -> str:
        if isinstance(messages, str):
            return messages
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif isinstance(part, str):
                        text_parts.append(part)
                return " ".join(part for part in text_parts if part).strip()
            if isinstance(content, str):
                return content.strip()
        return ""

    def _usage_from_text(user_msg: str, reply_text: str) -> dict[str, int]:
        prompt_tokens = len(user_msg.split())
        completion_tokens = len(reply_text.split())
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _openai_tool_calls(chat_res: ChatResponse) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for index, tool_call in enumerate(chat_res.tool_calls):
            tool_calls.append(
                {
                    "id": f"call_{index}_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(
                            tool_call.arguments, ensure_ascii=False
                        ),
                    },
                }
            )
        return tool_calls

    def _chat_completion_response_payload(
        *,
        model: str,
        response_id: str,
        created_time: int,
        user_msg: str,
        chat_res: ChatResponse,
    ) -> dict[str, Any]:
        tool_calls = _openai_tool_calls(chat_res)
        finish_reason = "tool_calls" if tool_calls else "stop"
        message: dict[str, Any] = {
            "role": "assistant",
            "content": (
                chat_res.message if not tool_calls else (chat_res.message or None)
            ),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage_from_text(user_msg, chat_res.message),
        }

    def _responses_api_payload(
        *,
        model: str,
        response_id: str,
        created_time: int,
        user_msg: str,
        chat_res: ChatResponse,
    ) -> dict[str, Any]:
        output: list[dict[str, Any]] = [
            {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": chat_res.message,
                    }
                ],
            }
        ]
        for tool_call in chat_res.tool_calls:
            output.append(
                {
                    "id": f"fc_{uuid.uuid4().hex[:12]}",
                    "type": "function_call",
                    "status": "completed",
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                }
            )
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_time,
            "model": model,
            "output": output,
            "usage": _usage_from_text(user_msg, chat_res.message),
        }

    def _stream_chunks(text: str, chunk_size: int = 80) -> list[str]:
        if not text:
            return [""]
        return [
            text[index : index + chunk_size]
            for index in range(0, len(text), chunk_size)
        ]

    async def _chat_completion_stream(
        *,
        model: str,
        response_id: str,
        created_time: int,
        chat_res: ChatResponse,
    ):
        first_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
        tool_calls = _openai_tool_calls(chat_res)
        if tool_calls:
            tool_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n"
        for piece in _stream_chunks(chat_res.message):
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _responses_stream(
        *,
        model: str,
        response_id: str,
        created_time: int,
        chat_res: ChatResponse,
        user_msg: str,
    ):
        created_event = {
            "type": "response.created",
            "response": {
                "id": response_id,
                "created_at": created_time,
                "model": model,
            },
        }
        yield f"data: {json.dumps(created_event, ensure_ascii=False)}\n\n"
        for piece in _stream_chunks(chat_res.message):
            delta_event = {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "delta": piece,
            }
            yield f"data: {json.dumps(delta_event, ensure_ascii=False)}\n\n"
        for tool_call in chat_res.tool_calls:
            tool_event = {
                "type": "response.function_call",
                "response_id": response_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            yield f"data: {json.dumps(tool_event, ensure_ascii=False)}\n\n"
        completed_event = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "output": _responses_api_payload(
                    model=model,
                    response_id=response_id,
                    created_time=created_time,
                    user_msg=user_msg,
                    chat_res=chat_res,
                )["output"],
                "usage": _usage_from_text(user_msg, chat_res.message),
            },
        }
        yield f"data: {json.dumps(completed_event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        raw_body = await request.body()
        ensure_body_size(
            raw_body, max_bytes=getattr(settings, "max_request_body_bytes", 262_144)
        )
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
        messages = body.get("messages") or []
        model = body.get("model") or "lisa"
        user_msg = _extract_user_message(messages)
        if not user_msg:
            raise HTTPException(
                status_code=400, detail="No user message found in messages list."
            )

        max_tokens = body.get("max_tokens") or 800
        session_id = body.get("user")
        chat_req = ChatRequest(
            message=user_msg, session_id=session_id, max_tokens=max_tokens
        )
        chat_res = await conductor.submit_chat(chat_req)
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        if body.get("stream"):
            return StreamingResponse(
                _chat_completion_stream(
                    model=model,
                    response_id=response_id,
                    created_time=created_time,
                    chat_res=chat_res,
                ),
                media_type="text/event-stream",
            )

        return Response(
            content=json.dumps(
                _chat_completion_response_payload(
                    model=model,
                    response_id=response_id,
                    created_time=created_time,
                    user_msg=user_msg,
                    chat_res=chat_res,
                ),
                ensure_ascii=False,
            ),
            media_type="application/json",
        )

    @app.post("/v1/responses")
    async def responses_endpoint(request: Request) -> Response:
        raw_body = await request.body()
        ensure_body_size(
            raw_body, max_bytes=getattr(settings, "max_request_body_bytes", 262_144)
        )
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
        model = body.get("model") or "lisa"
        input_data = body.get("input") or []
        user_msg = _extract_user_message(input_data)
        if not user_msg:
            raise HTTPException(
                status_code=400, detail="No user input found in request."
            )

        max_tokens = body.get("max_tokens") or 800
        session_id = body.get("user")
        chat_req = ChatRequest(
            message=user_msg, session_id=session_id, max_tokens=max_tokens
        )
        chat_res = await conductor.submit_chat(chat_req)
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        if body.get("stream"):
            return StreamingResponse(
                _responses_stream(
                    model=model,
                    response_id=response_id,
                    created_time=created_time,
                    chat_res=chat_res,
                    user_msg=user_msg,
                ),
                media_type="text/event-stream",
            )

        return Response(
            content=json.dumps(
                _responses_api_payload(
                    model=model,
                    response_id=response_id,
                    created_time=created_time,
                    user_msg=user_msg,
                    chat_res=chat_res,
                ),
                ensure_ascii=False,
            ),
            media_type="application/json",
        )

    @app.post("/v1/embeddings")
    async def embeddings_endpoint(request: Request) -> dict[str, Any]:
        raw_body = await request.body()
        ensure_body_size(
            raw_body, max_bytes=getattr(settings, "max_request_body_bytes", 262_144)
        )
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
        input_data = body.get("input")
        model = body.get("model") or "auto"

        if not input_data:
            raise HTTPException(
                status_code=400, detail="Missing 'input' field in request body."
            )

        api_key = settings.freellmapi_api_key or settings.model_api_key

        if not api_key or "test" in str(settings.db_path):
            emb_vector = [0.0] * 768
            inputs = [input_data] if isinstance(input_data, str) else (input_data or [])
            data_list = []
            for idx, inp in enumerate(inputs):
                data_list.append(
                    {"object": "embedding", "index": idx, "embedding": emb_vector}
                )
            return {
                "object": "list",
                "data": data_list,
                "model": model,
                "usage": {
                    "prompt_tokens": sum(len(str(inp).split()) for inp in inputs),
                    "total_tokens": sum(len(str(inp).split()) for inp in inputs),
                },
            }

        if settings.freellmapi_embeddings_url:
            embeddings_url = settings.freellmapi_embeddings_url
        else:
            base_url = settings.freellmapi_base_url or settings.model_base_url
            if not base_url:
                raise HTTPException(
                    status_code=400, detail="No external API configured for embeddings."
                )
            base_url_str = str(base_url).rstrip("/")
            if not base_url_str.endswith("/v1"):
                embeddings_url = f"{base_url_str}/v1/embeddings"
            else:
                embeddings_url = f"{base_url_str}/embeddings"

        payload = {
            "input": input_data,
            "model": model,
        }
        if "user" in body:
            payload["user"] = body["user"]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    embeddings_url, json=payload, headers=headers
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail=exc.response.text
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch embeddings from FreeLLMAPI: {str(exc)}",
            )

    async def ingest_message(
        source: str, request: Request, response: Response
    ) -> HubAck:
        try:
            body = await request.body()
            ensure_body_size(
                body, max_bytes=getattr(settings, "max_request_body_bytes", 262_144)
            )
            message_hub_verifier = message_hub.webhook_secrets
            from safety.webhooks import verify_webhook

            try:
                verify_webhook(
                    source,
                    {key: value for key, value in request.headers.items()},
                    body,
                    url=str(request.url),
                    secrets=message_hub_verifier,
                )
            except PermissionError as exc:
                LOGGER.warning("Webhook signature verification failed: %s", exc)
                response.status_code = 403
                return HubAck(
                    accepted=False,
                    queued=False,
                    detail="Invalid signature or unauthorized webhook source.",
                )
            except ValueError as exc:
                LOGGER.warning("Invalid webhook headers or structure: %s", exc)
                response.status_code = 400
                return HubAck(
                    accepted=False,
                    queued=False,
                    detail=f"Bad request payload: {str(exc)}",
                )

            payload = {}
            if body:
                try:
                    payload = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    form = await request.form()
                    payload = dict(form)
            await message_hub._replay_guard.check_webhook(
                source=source,
                payload=payload if isinstance(payload, dict) else {},
                headers={key: value for key, value in request.headers.items()},
                body=body,
            )
            ack, status = await message_hub.ingest_message(
                source, payload if isinstance(payload, dict) else {}
            )
            response.status_code = status
            if ack.retry_after_seconds is not None:
                response.headers["Retry-After"] = str(ack.retry_after_seconds)
            return ack
        except ReplayAttackDetected as exc:
            LOGGER.warning("Replay attack detected for %s webhook: %s", source, exc)
            response.status_code = 409
            return HubAck(accepted=False, queued=False, detail=str(exc))
        except Exception as exc:
            LOGGER.exception("Internal error in webhook ingestion loop: %s", exc)
            response.status_code = 500
            return HubAck(
                accepted=False,
                queued=False,
                detail="An internal server error occurred while processing this message.",
            )

    @app.post("/ingest/{source}", response_model=HubAck, status_code=202)
    async def ingest(
        source: str, inbound: InboundMessage, response: Response, raw_request: Request
    ) -> HubAck:
        require_admin_request(raw_request, settings)
        try:
            ack, status = await message_hub.ingest_message(
                source,
                inbound.model_dump(mode="json"),
            )
        except ReplayAttackDetected as exc:
            response.status_code = 409
            return HubAck(accepted=False, queued=False, detail=str(exc))
        response.status_code = status
        return ack

    @app.post("/v1/messages/ingest/{source}", response_model=HubAck, status_code=202)
    async def multiplex_ingest(
        source: str, inbound: InboundMessage, response: Response, raw_request: Request
    ) -> HubAck:
        require_admin_request(raw_request, settings)
        try:
            ack, status = await message_hub.ingest_message(
                source,
                inbound.model_dump(mode="json"),
            )
        except ReplayAttackDetected as exc:
            response.status_code = 409
            return HubAck(accepted=False, queued=False, detail=str(exc))
        response.status_code = status
        return ack

    @app.post("/v1/messages/dispatch", response_model=BotDispatchResponse)
    async def multiplex_dispatch(
        raw_request: Request,
        payload: dict[str, Any] = Body(...),
    ) -> BotDispatchResponse:
        require_admin_request(raw_request, settings)
        dispatch_request = BotDispatchRequest.model_validate(payload)
        return await message_hub.dispatch_request(dispatch_request)

    @app.post("/telegram/webhook", response_model=HubAck, status_code=202)
    async def telegram_webhook(request: Request, response: Response) -> HubAck:
        return await ingest_message("telegram", request, response)

    @app.post("/slack/events", response_model=HubAck, status_code=202)
    async def slack_events(request: Request, response: Response) -> HubAck:
        return await ingest_message("slack", request, response)

    @app.post("/whatsapp/webhook", response_model=HubAck, status_code=202)
    async def whatsapp_webhook(request: Request, response: Response) -> HubAck:
        return await ingest_message("whatsapp", request, response)

    @app.get("/v1/channels", response_model=ChannelCapabilitiesResponse)
    async def channels() -> ChannelCapabilitiesResponse:
        capabilities = message_hub.channel_gateway.capabilities()
        return ChannelCapabilitiesResponse(
            configured_channels=list(capabilities.get("configured_channels") or []),
            supports=dict(capabilities.get("supports") or {}),
            features=list(capabilities.get("features") or []),
            access=message_hub.channel_access.summary(),
        )

    @app.post("/v1/channels/authorize", response_model=ChannelAccessResponse)
    async def authorize_channel_user(
        access_request: ChannelAccessRequest, raw_request: Request
    ) -> ChannelAccessResponse:
        require_admin_request(raw_request, settings)
        record = message_hub.channel_access.grant(
            access_request.source, access_request.user_id
        )
        return ChannelAccessResponse(
            source=record.source,
            user_ids=message_hub.channel_access.summary().get(record.source, []),
            updated=True,
        )

    @app.post("/v1/channels/revoke", response_model=ChannelAccessResponse)
    async def revoke_channel_user(
        access_request: ChannelAccessRequest, raw_request: Request
    ) -> ChannelAccessResponse:
        require_admin_request(raw_request, settings)
        updated = message_hub.channel_access.revoke(
            access_request.source, access_request.user_id
        )
        return ChannelAccessResponse(
            source=access_request.source,
            user_ids=message_hub.channel_access.summary().get(
                access_request.source, []
            ),
            updated=updated,
        )

    @app.get("/tools", response_model=list[ToolSummary])
    async def list_tools() -> list[ToolSummary]:
        return [
            ToolSummary(
                name=tool.name,
                description=tool.description,
                restricted_safe=tool.restricted_safe,
            )
            for tool in runtime.tools.list_tools()
        ]

    @app.get("/personas")
    async def personas() -> dict[str, object]:
        return {
            "weights": runtime.llm_client.persona_summary(),
            "bank_path": str(runtime.settings.persona_vectors_path),
        }

    @app.get("/gating")
    async def gating_info(text: str | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "metadata": runtime.gating.metadata(),
            "bank_path": str(runtime.settings.gating_model_path),
        }
        if text is not None:
            result["blend"] = runtime.gating.predict_blend(text)
            result["text"] = text
        return result

    @app.post("/tools/{tool_name}", response_model=ToolInvokeResponse)
    async def invoke_tool(
        tool_name: str, request: ToolInvokeRequest
    ) -> ToolInvokeResponse:
        state = runtime.notepad.get_constitution_state()
        try:
            output = await runtime.tools.invoke(
                name=tool_name,
                arguments=request.arguments,
                constitution=ConstitutionMode(state["mode"]),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (TimeoutError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ToolInvokeResponse(name=tool_name, success=True, output=output)

    @app.get("/notepad/search", response_model=NotepadSearchResponse)
    async def search_notepad(
        q: str = Query(..., min_length=1),
        limit: int = Query(default=10, ge=1, le=50),
    ) -> NotepadSearchResponse:
        await runtime.notepad_writer.flush_pending()
        rows = runtime.notepad.search(query=q, limit=limit)
        return NotepadSearchResponse(
            query=q,
            results=[
                NotepadEntry(
                    id=row["id"],
                    entry_type=row["entry_type"],
                    constitution=row["constitution"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    payload=row["payload"],
                )
                for row in rows
            ],
        )

    @app.get("/dashboard", response_model=list[DashboardMetric])
    async def dashboard(
        limit: int = Query(default=20, ge=1, le=100)
    ) -> list[DashboardMetric]:
        await runtime.notepad_writer.flush_pending()
        metrics = runtime.notepad.recent_metrics(limit=limit)
        return [
            DashboardMetric(
                metric=row["metric"],
                value=row["value"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in metrics
        ]

    @app.get("/dashboard/live", response_class=HTMLResponse)
    async def dashboard_live() -> HTMLResponse:
        return HTMLResponse(message_hub._render_dashboard_html())

    @app.get("/dashboard/snapshot")
    async def dashboard_snapshot() -> dict[str, Any]:
        return message_hub._snapshot_payload()

    @app.get("/personal")
    async def personal_summary() -> dict[str, object]:
        if personal_store is None:
            return {"enabled": False, "calendar": calendar_awareness.today()}
        return {
            "enabled": True,
            "summary": personal_store.summary(),
            "calendar": calendar_awareness.today(),
            "wellness": {"should_check_in": wellness_tracker.should_check_in()},
        }

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = await event_bus.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event.as_dict())
        except WebSocketDisconnect:
            pass
        finally:
            await event_bus.unsubscribe(queue)

    @app.websocket("/ws/dashboard")
    async def ws_dashboard(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = await event_bus.subscribe()
        await websocket.send_json(message_hub._snapshot_payload())
        try:
            while True:
                await queue.get()
                await websocket.send_json(message_hub._snapshot_payload())
        except WebSocketDisconnect:
            pass
        finally:
            await event_bus.unsubscribe(queue)

    return app


__all__ = ["create_app"]
