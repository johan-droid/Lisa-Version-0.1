from __future__ import annotations

from contextlib import asynccontextmanager, suppress
from datetime import datetime
import asyncio
import json

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.websockets import WebSocketDisconnect
from fastapi import WebSocket

from lisa.agent import LisaRuntime
from lisa.conductor import TaskConductor
from lisa.config import Settings, get_settings
from lisa.constitutions import ConstitutionMode
from lisa.evolution import NightlyEvolutionScheduler
from lisa.gating import PersonaGatingNetwork
from lisa.events import EventBus, LisaEvent
from lisa.llm import LLMClient
from lisa.hub import MessageHub
from lisa.notepad import AsyncNotepadWriter, Notepad
from lisa.schemas import (
    ChatRequest,
    ChatResponse,
    ConstitutionStateResponse,
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
from personal import CalendarAwareness, PersonalContextStore, ReminderScheduler, StyleLearner, WellnessTracker


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    event_bus = EventBus()
    notepad = Notepad(settings.db_path)
    notepad.startup_maintenance(settings.backup_dir, settings.notepad_retention_days, settings.notepad_backup_keep)
    gating = PersonaGatingNetwork.load_or_initialize(settings.gating_model_path)
    llm_client = LLMClient(settings)
    notepad_writer = AsyncNotepadWriter(notepad=notepad, event_bus=event_bus)
    personal_store = PersonalContextStore(settings.personal_db_path) if settings.enable_personal_features else None
    reminder_scheduler = ReminderScheduler(store=personal_store, event_bus=event_bus) if personal_store is not None else None
    calendar_awareness = CalendarAwareness(path=settings.workspace_root / "data" / "calendar.json")
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
    runtime = LisaRuntime(
        settings=settings,
        notepad=notepad,
        llm_client=llm_client,
        tools=tools,
        tool_executor=tool_executor,
        event_bus=event_bus,
        notepad_writer=notepad_writer,
        gating=gating,
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
        capabilities_provider=lambda: [tool.name for tool in runtime.tools.list_tools()],
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
            await notepad_writer.close()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.settings = settings
    app.state.event_bus = event_bus
    app.state.notepad_writer = notepad_writer
    app.state.tool_executor = tool_executor
    app.state.conductor = conductor
    app.state.gating = gating
    app.state.message_hub = message_hub
    app.state.evolution_scheduler = evolution_scheduler
    app.state.personal_store = personal_store
    app.state.reminder_scheduler = reminder_scheduler
    app.state.calendar_awareness = calendar_awareness
    app.state.style_learner = style_learner
    app.state.wellness_tracker = wellness_tracker

    import pydantic

    class BotConnectRequest(pydantic.BaseModel):
        security_key: str
        bot_type: str
        user_id: str

    @app.post("/bots/connect")
    async def connect_bot(request: BotConnectRequest) -> dict:
        """Connect a bot securely using a key."""
        if not settings.bot_security_key:
            raise HTTPException(status_code=500, detail="Bot security key not configured on server")
        if request.security_key != settings.bot_security_key:
            raise HTTPException(status_code=403, detail="Invalid security key")
        return {
            "status": "success",
            "message": f"Successfully connected to {request.bot_type} for user {request.user_id}",
            "connected": True
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        state = runtime.notepad.get_constitution_state()
        return {"status": "ok", "constitution": state["mode"]}

    @app.post("/shutdown")
    async def shutdown(request: Request) -> dict[str, str]:
        if request.client and request.client.host not in ("127.0.0.1", "localhost"):
            raise HTTPException(status_code=403, detail="Forbidden")
        
        import os
        import signal
        import time
        
        def kill_self():
            time.sleep(0.5)
            # Send SIGINT to trigger uvicorn shutdown
            os.kill(os.getpid(), signal.SIGINT)
            
        asyncio.create_task(asyncio.to_thread(kill_self))
        return {"status": "shutdown_initiated"}

    @app.post("/shed_memory")
    async def shed_memory(request: Request) -> dict[str, str]:
        if request.client and request.client.host not in ("127.0.0.1", "localhost"):
            raise HTTPException(status_code=403, detail="Forbidden")

        # 1. Clear browser cache
        try:
            async with runtime.tools._web_cache_lock:
                runtime.tools._web_cache.clear()
        except Exception:
            pass

        # 2. Offload oldest notepad entries (maintenance)
        try:
            runtime.notepad.startup_maintenance(settings.backup_dir, days_to_keep=15, keep_backups=2)
        except Exception:
            pass

        # 3. Reduce LLM context window from 2048 to 1024
        try:
            if hasattr(runtime.llm_client, "local_backend") and hasattr(runtime.llm_client.local_backend, "_llama"):
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

    async def ingest_message(source: str, request: Request, response: Response) -> HubAck:
        body = await request.body()
        message_hub_verifier = message_hub.webhook_secrets
        from safety.webhooks import verify_webhook

        verify_webhook(source, {key: value for key, value in request.headers.items()}, body, url=str(request.url), secrets=message_hub_verifier)
        payload = {}
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                form = await request.form()
                payload = dict(form)
        ack, status = await message_hub.ingest_message(source, payload if isinstance(payload, dict) else {})
        response.status_code = status
        if ack.retry_after_seconds is not None:
            response.headers["Retry-After"] = str(ack.retry_after_seconds)
        return ack

    @app.post("/ingest/{source}", response_model=HubAck, status_code=202)
    async def ingest(source: str, request: InboundMessage, response: Response) -> HubAck:
        return await ingest_message(source, request, response)

    @app.post("/telegram/webhook", response_model=HubAck, status_code=202)
    async def telegram_webhook(request: Request, response: Response) -> HubAck:
        return await ingest_message("telegram", request, response)

    @app.post("/slack/events", response_model=HubAck, status_code=202)
    async def slack_events(request: Request, response: Response) -> HubAck:
        return await ingest_message("slack", request, response)

    @app.post("/whatsapp/webhook", response_model=HubAck, status_code=202)
    async def whatsapp_webhook(request: Request, response: Response) -> HubAck:
        return await ingest_message("whatsapp", request, response)

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
    async def invoke_tool(tool_name: str, request: ToolInvokeRequest) -> ToolInvokeResponse:
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
    async def dashboard(limit: int = Query(default=20, ge=1, le=100)) -> list[DashboardMetric]:
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

    return app


app = create_app()
