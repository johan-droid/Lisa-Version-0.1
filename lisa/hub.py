from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

from aiohttp import WSMsgType, web

from lisa.channel_access import ChannelAccessController
from lisa.config import Settings
from lisa.events import EventBus, LisaEvent
from lisa.channels import ChannelGateway
from lisa.schemas import (
    BotDispatchRequest,
    BotDispatchResponse,
    HubAck,
    InboundMessage,
    BrainTask,
)
from interfaces.dashboard import render_dashboard_html
from personal.context_store import PersonalContextStore
from safety.input_sanitizer import ensure_body_size, sanitize_structure
from safety.replay_guard import ReplayAttackDetected, ReplayGuard
from safety.webhooks import secrets_from_mapping, verify_webhook


@dataclass(slots=True)
class DashboardMetricsState:
    window_seconds: int = 3600
    sample_limit: int = 120
    active_job_ids: set[str] = field(default_factory=set)
    token_consumption_total: int = 0
    token_consumption_by_provider: dict[str, int] = field(default_factory=dict)
    persona_blend: dict[str, float] = field(default_factory=dict)
    evolution_events: deque[datetime] = field(default_factory=deque)
    samples: deque[dict[str, Any]] = field(default_factory=deque)
    last_update: str | None = None
    last_evolution_skill: str | None = None
    last_evolution_status: str | None = None

    def observe(self, event: LisaEvent) -> None:
        now = datetime.now(timezone.utc)
        payload = event.payload

        if event.type == "conductor.job_started":
            job_id = str(payload.get("job_id") or "")
            if job_id:
                self.active_job_ids.add(job_id)
        elif event.type in {"conductor.job_finished", "conductor.job_error"}:
            job_id = str(payload.get("job_id") or "")
            if job_id:
                self.active_job_ids.discard(job_id)
        elif event.type == "external_llm.completed":
            usage = payload.get("usage") or {}
            total = int(usage.get("total_tokens") or 0)
            provider = str(payload.get("provider") or "unknown")
            self.token_consumption_total += total
            self.token_consumption_by_provider[provider] = (
                self.token_consumption_by_provider.get(provider, 0) + total
            )
        elif event.type == "chat.responded":
            personas = payload.get("personas")
            if isinstance(personas, dict) and personas:
                self.persona_blend = {
                    str(key): float(value) for key, value in personas.items()
                }
        elif event.type == "dashboard.metric":
            metric = str(payload.get("metric") or "")
            value = str(payload.get("value") or "")
            if metric == "evolution_last_skill":
                self.last_evolution_skill = value
            elif metric == "evolution_status":
                self.last_evolution_status = value
        elif event.type == "ledger.append":
            if payload.get("entry_type") in {"task_summary", "evolution_cycle"}:
                self.evolution_events.append(now)
                self._trim_evolution_window(now)
        elif event.type == "evolution.skill_registered":
            skill_name = str(payload.get("skill_name") or "")
            if skill_name:
                self.last_evolution_skill = skill_name
            self.last_evolution_status = "registered"
        elif event.type == "evolution.cycle_started":
            self.last_evolution_status = "running"
        elif event.type == "evolution.cycle_finished":
            self.last_evolution_status = str(payload.get("status") or "finished")

        self.last_update = now.isoformat()
        self._record_sample(now)

    def _trim_evolution_window(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self.evolution_events and self.evolution_events[0] < cutoff:
            self.evolution_events.popleft()

    def _record_sample(self, now: datetime) -> None:
        self._trim_evolution_window(now)
        sample = {
            "timestamp": now.isoformat(),
            "active_tasks": len(self.active_job_ids),
            "token_consumption_total": self.token_consumption_total,
            "evolution_rate": self.evolution_rate,
        }
        self.samples.append(sample)
        while len(self.samples) > self.sample_limit:
            self.samples.popleft()

    @property
    def evolution_rate(self) -> float:
        return round(len(self.evolution_events) / 60.0, 3)

    def snapshot(
        self,
        *,
        personal_context: dict[str, Any] | None = None,
        capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        persona_items = sorted(
            self.persona_blend.items(), key=lambda item: item[1], reverse=True
        )
        timeline = list(self.samples)
        return {
            "timestamp": self.last_update or datetime.now(timezone.utc).isoformat(),
            "active_tasks": len(self.active_job_ids),
            "token_consumption": {
                "total": self.token_consumption_total,
                "by_provider": dict(self.token_consumption_by_provider),
            },
            "persona_blend": dict(self.persona_blend),
            "dominant_persona": persona_items[0][0] if persona_items else None,
            "evolution_rate": self.evolution_rate,
            "last_evolution_skill": self.last_evolution_skill,
            "last_evolution_status": self.last_evolution_status,
            "capabilities": list(capabilities or []),
            "personal_context": dict(personal_context or {}),
            "charts": {
                "timeline": {
                    "labels": [sample["timestamp"] for sample in timeline],
                    "active_tasks": [sample["active_tasks"] for sample in timeline],
                    "token_consumption_total": [
                        sample["token_consumption_total"] for sample in timeline
                    ],
                    "evolution_rate": [sample["evolution_rate"] for sample in timeline],
                },
                "personas": {
                    "labels": [name for name, _ in persona_items],
                    "values": [value for _, value in persona_items],
                },
            },
        }


class MessageHub:
    def __init__(
        self,
        settings: Settings,
        event_bus: EventBus,
        conductor: Any,
        dashboard_state: DashboardMetricsState | None = None,
        personal_store: PersonalContextStore | None = None,
        capabilities_provider: Callable[[], list[str]] | None = None,
        channel_gateway: ChannelGateway | None = None,
    ):
        self.settings = settings
        self.event_bus = event_bus
        self.conductor = conductor
        self.dashboard_state = dashboard_state or DashboardMetricsState()
        self.personal_store = personal_store
        import os
        from lisa.channels import ChannelCredentials, ChannelGateway

        if channel_gateway is None:
            credentials_map = {}
            if hasattr(settings, "interface_keys") and settings.interface_keys:
                credentials_map.update(settings.interface_keys)

            credentials_map.update(
                {
                    "telegram_bot_token": settings.telegram_bot_token
                    or credentials_map.get("telegram_bot_token"),
                    "slack_bot_token": settings.slack_bot_token
                    or credentials_map.get("slack_bot_token"),
                    "whatsapp_auth_token": settings.whatsapp_bot_token
                    or credentials_map.get("whatsapp_auth_token"),
                }
            )

            for key in (
                "TWILIO_ACCOUNT_SID",
                "TWILIO_AUTH_TOKEN",
                "TWILIO_FROM_NUMBER",
                "TELEGRAM_DEFAULT_CHAT_ID",
                "SLACK_DEFAULT_CHANNEL",
                "WHATSAPP_DEFAULT_TO",
            ):
                val = os.environ.get(key)
                if val:
                    credentials_map[key.lower()] = val
            for key in (
                "WHATSAPP_ACCOUNT_SID",
                "WHATSAPP_AUTH_TOKEN",
                "WHATSAPP_FROM_NUMBER",
            ):
                val = os.environ.get(key)
                if val:
                    credentials_map[key.lower()] = val

            self.channel_gateway = ChannelGateway(
                credentials=ChannelCredentials.from_mapping(credentials_map)
            )
        else:
            self.channel_gateway = channel_gateway

        mapping = dict(getattr(settings, "interface_keys", {}) or {})
        mapping["telegram_webhook_secret"] = getattr(
            settings, "telegram_webhook_secret", None
        )
        mapping["telegram_bot_token"] = getattr(settings, "telegram_bot_token", None)
        self.webhook_secrets = secrets_from_mapping(mapping)
        self.capabilities_provider = capabilities_provider
        self.channel_access = ChannelAccessController(
            self.settings.workspace_root / "data" / "channel_access.json",
            initial={
                "telegram": list(
                    getattr(settings, "telegram_allowed_user_ids", []) or []
                ),
                "slack": list(getattr(settings, "slack_allowed_user_ids", []) or []),
                "whatsapp": list(
                    getattr(settings, "whatsapp_allowed_user_ids", []) or []
                ),
            },
        )
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._websockets: set[web.WebSocketResponse] = set()
        self._session_by_user: dict[str, str] = {}
        self._delivery_targets: dict[str, dict[str, Any]] = {}
        self._ws_lock = asyncio.Lock()
        self._replay_guard = ReplayGuard()
        self._started = False
        self._setup_routes()

    def _setup_routes(self) -> None:
        self._app.router.add_get("/", self._dashboard_page)
        self._app.router.add_get("/dashboard", self._dashboard_page)
        self._app.router.add_get("/dashboard/data", self._dashboard_data)
        self._app.router.add_get("/ws", self._dashboard_ws)
        self._app.router.add_get("/ws/dashboard", self._dashboard_ws)
        self._app.router.add_post("/telegram/webhook", self._telegram_webhook)
        self._app.router.add_post("/slack/events", self._slack_webhook)
        self._app.router.add_post("/whatsapp/webhook", self._whatsapp_webhook)
        self._app.router.add_get("/health", self._health)

    @property
    def enabled(self) -> bool:
        return self.settings.message_hub_enabled

    async def start(self) -> None:
        if self._started:
            return
        if self.enabled and self.settings.message_hub_start_listener:
            self._runner = web.AppRunner(self._app, access_log=None)
            await self._runner.setup()
            self._site = web.TCPSite(
                self._runner,
                host=self.settings.message_hub_host,
                port=self.settings.message_hub_port,
            )
            await self._site.start()
        await self._configure_platform_features()
        self._event_task = asyncio.create_task(
            self._observe_events(), name="lisa-message-hub"
        )
        self._started = True

    async def close(self) -> None:
        if self._event_task is not None:
            self._event_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._event_task
            self._event_task = None

        async with self._ws_lock:
            for websocket in list(self._websockets):
                with suppress(Exception):
                    await websocket.close()
            self._websockets.clear()

        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        self._started = False

    async def _observe_events(self) -> None:
        queue = await self.event_bus.subscribe()
        try:
            while True:
                event = await queue.get()
                self.dashboard_state.observe(event)
                if event.type == "chat.responded":
                    session_id = event.session_id or event.payload.get("session_id")
                    asyncio.create_task(
                        self._deliver_chat_response(session_id, event.payload),
                        name=f"deliver-chat-response-{session_id}",
                    )
                elif event.type == "conductor.job_started":
                    session_id = event.payload.get("session_id")
                    job_id = event.payload.get("job_id")
                    if session_id and job_id:
                        asyncio.create_task(
                            self._send_typing_indicator_loop(session_id, job_id),
                            name=f"typing-loop-{session_id}",
                        )
                await self._broadcast_snapshot()
        finally:
            await self.event_bus.unsubscribe(queue)

    async def _send_typing_indicator_loop(self, session_id: str, job_id: str) -> None:
        # Give a small delay to register target if needed, though usually already registered
        await asyncio.sleep(0.1)
        target = self._delivery_targets.get(session_id)
        if not target or target.get("channel") != "telegram":
            return

        user_id = target.get("channel_id") or target.get("user_id")
        while job_id in self.dashboard_state.active_job_ids:
            try:
                await self.channel_gateway.send_typing("telegram", user_id=user_id)
            except Exception:
                pass
            await asyncio.sleep(4.0)

    async def _broadcast_snapshot(self) -> None:
        snapshot = self._snapshot_payload()
        stale: list[web.WebSocketResponse] = []
        async with self._ws_lock:
            for websocket in list(self._websockets):
                if websocket.closed:
                    stale.append(websocket)
                    continue
                try:
                    await websocket.send_json(snapshot)
                except Exception:
                    stale.append(websocket)
            for websocket in stale:
                self._websockets.discard(websocket)

    def _register_delivery_target(
        self, inbound: InboundMessage, deliver: bool = True
    ) -> None:
        if not deliver:
            return
        session_id = inbound.session_id
        if not session_id:
            return
        target = {
            "channel": inbound.source,
            "channel_id": inbound.channel,
            "user_id": inbound.user_id,
            "reply_to_message_id": inbound.reply_to_message_id,
            "metadata": dict(inbound.metadata),
        }
        if inbound.message_id:
            target["message_id"] = inbound.message_id
        self._delivery_targets[session_id] = target

    async def _deliver_chat_response(
        self, session_id: str | None, payload: dict[str, Any]
    ) -> None:
        if not session_id:
            return
        target = self._delivery_targets.get(session_id)
        if not target:
            return

        channel = str(target.get("channel") or "direct")
        if channel == "direct":
            return

        if channel in {"telegram", "slack"}:
            target_id = target.get("channel_id") or target.get("user_id")
        else:
            target_id = target.get("user_id")
        delivery_hints = payload.get("delivery_hints")
        try:
            result = await self.channel_gateway.send_response(
                channel,
                user_id=str(target_id or ""),
                text=str(payload.get("message") or ""),
                parse_mode=(delivery_hints or {}).get("parse_mode"),
                reply_to_message_id=target.get("reply_to_message_id"),
                metadata=dict(target.get("metadata") or {}),
                delivery_hints=(
                    delivery_hints if isinstance(delivery_hints, dict) else {}
                ),
            )
            await self.event_bus.publish(
                LisaEvent(
                    type="bot.delivery",
                    payload={
                        "session_id": session_id,
                        "channel": channel,
                        "delivered": bool(result.get("delivered")),
                        "detail": result.get("detail"),
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - outbound channel failure path
            await self.event_bus.publish(
                LisaEvent(
                    type="bot.delivery_error",
                    payload={
                        "session_id": session_id,
                        "channel": channel,
                        "error": str(exc),
                    },
                )
            )

    async def ingest_message(
        self, source: str, payload: dict[str, Any]
    ) -> tuple[HubAck, int]:
        inbound = self._normalize_inbound_message(source, payload)
        await self._replay_guard.check_message(
            source=source,
            user_id=inbound.user_id,
            message_id=inbound.message_id
            or str(inbound.metadata.get("telegram_callback_query_id") or ""),
        )

        if source in {
            "telegram",
            "slack",
            "whatsapp",
        } and not self.channel_access.is_authorized(source, inbound.user_id):
            await self._deny_inbound_message(inbound)
            return (
                HubAck(
                    accepted=False,
                    queued=False,
                    job_id=None,
                    retry_after_seconds=None,
                    detail="Sender is not authorized for this channel.",
                ),
                202,
            )

        shortcut = await self._maybe_handle_provider_shortcut(inbound)
        if shortcut is not None:
            return shortcut

        self._register_delivery_target(inbound)
        job_id = self.conductor.try_submit_message(inbound)
        if job_id is None:
            return (
                HubAck(
                    accepted=False,
                    queued=False,
                    job_id=None,
                    retry_after_seconds=2,
                    detail="Queue is full. Please retry shortly.",
                ),
                202,
            )

        return (
            HubAck(
                accepted=True,
                queued=True,
                job_id=job_id,
                retry_after_seconds=None,
                detail="Message accepted and queued.",
            ),
            202,
        )

    async def dispatch_request(
        self, request: BotDispatchRequest
    ) -> BotDispatchResponse:
        inbound = InboundMessage(
            source=request.source,
            user_id=request.user_id or request.channel,
            channel=request.channel,
            text=request.text,
            session_id=request.session_id,
            priority=request.priority,
            reply_to_message_id=request.reply_to_message_id,
            metadata=dict(request.metadata),
        )
        self._register_delivery_target(inbound, deliver=request.deliver)
        response = await self.conductor.submit_brain(
            BrainTask(
                inbound=inbound,
                max_tokens=request.max_tokens,
            ),
            priority=request.priority,
        )
        delivery: dict[str, Any] = {}
        delivered = False
        if request.deliver and request.channel != "direct":
            delivery = await self.channel_gateway.send_response(
                request.channel,
                user_id=inbound.user_id,
                text=response.message,
                parse_mode=response.delivery_hints.get("parse_mode"),
                reply_to_message_id=inbound.reply_to_message_id,
                metadata=inbound.metadata,
                delivery_hints=response.delivery_hints.get(
                    request.channel, response.delivery_hints
                ),
            )
            delivered = bool(delivery.get("delivered"))
        return BotDispatchResponse(
            accepted=True,
            delivered=delivered,
            channel=request.channel,
            session_id=response.session_id,
            job_id=None,
            response=response,
            delivery=delivery,
        )

    async def _configure_platform_features(self) -> None:
        telegram_commands = [
            {"command": "start", "description": "Start or resume a chat with LISA"},
            {
                "command": "help",
                "description": "Show Telegram commands and quick actions",
            },
            {
                "command": "status",
                "description": "Show current runtime and channel status",
            },
            {"command": "new", "description": "Start a fresh session"},
            {"command": "tools", "description": "List current tool capabilities"},
        ]
        try:
            await self.channel_gateway.configure_telegram_commands(telegram_commands)
        except Exception:
            pass

    async def _deny_inbound_message(self, inbound: InboundMessage) -> None:
        target_id = self._outbound_target_for_channel(inbound)
        try:
            await self.channel_gateway.send_response(
                inbound.source,
                user_id=target_id,
                text="This channel is restricted. Ask the workspace admin to authorize your user ID.",
            )
        except Exception:
            pass

    async def _maybe_handle_provider_shortcut(
        self, inbound: InboundMessage
    ) -> tuple[HubAck, int] | None:
        if inbound.source != "telegram":
            return None

        callback_data = (
            str(inbound.metadata.get("telegram_callback_data") or "").strip().lower()
        )
        command = self._telegram_command(inbound.text)
        action = callback_data or command
        if not action:
            return None

        if action == "start":
            await self._send_telegram_shortcut(
                inbound,
                "LISA is online. Use /help for commands, /status for runtime state, /new for a fresh session, or send any task directly.",
                edit=False,
            )
            return self._shortcut_ack("Telegram onboarding delivered.")

        if action == "help":
            await self._send_telegram_shortcut(
                inbound,
                "Commands: /status, /new, /tools. You can also press the buttons below or send any coding request directly.",
                edit=bool(callback_data),
            )
            return self._shortcut_ack("Telegram help delivered.")

        if action == "status":
            active_tasks = len(self.dashboard_state.active_job_ids)
            dominant_persona = (
                self.dashboard_state.snapshot().get("dominant_persona") or "n/a"
            )
            text = (
                f"LISA status\n"
                f"Active tasks: {active_tasks}\n"
                f"Dominant persona: {dominant_persona}\n"
                f"Authorized users: {', '.join(self.channel_access.summary().get('telegram', [])) or 'open'}"
            )
            await self._send_telegram_shortcut(inbound, text, edit=bool(callback_data))
            return self._shortcut_ack("Telegram status delivered.")

        if action == "tools":
            capabilities = (
                self.capabilities_provider()
                if self.capabilities_provider is not None
                else []
            )
            text = "Available tools: " + (
                ", ".join(capabilities[:20]) if capabilities else "none registered"
            )
            await self._send_telegram_shortcut(inbound, text, edit=bool(callback_data))
            return self._shortcut_ack("Telegram tools delivered.")

        if action == "new":
            self._session_by_user[inbound.user_id] = str(uuid4())
            await self._send_telegram_shortcut(
                inbound,
                "Started a fresh session. Send the next request when ready.",
                edit=bool(callback_data),
            )
            return self._shortcut_ack("Fresh session created.")

        return None

    async def _send_telegram_shortcut(
        self, inbound: InboundMessage, text: str, *, edit: bool
    ) -> None:
        target_id = self._outbound_target_for_channel(inbound)
        delivery_hints: dict[str, Any] = {
            "reply_markup": self._telegram_shortcut_keyboard()
        }
        if edit and inbound.message_id:
            try:
                delivery_hints["edit_message_id"] = int(inbound.message_id)
            except ValueError:
                pass
        await self.channel_gateway.send_response(
            "telegram",
            user_id=target_id,
            text=text,
            metadata=dict(inbound.metadata),
            delivery_hints=delivery_hints,
        )

    @staticmethod
    def _shortcut_ack(detail: str) -> tuple[HubAck, int]:
        return (
            HubAck(
                accepted=True,
                queued=False,
                job_id=None,
                retry_after_seconds=None,
                detail=detail,
            ),
            202,
        )

    @staticmethod
    def _telegram_shortcut_keyboard() -> dict[str, Any]:
        return {
            "inline_keyboard": [
                [
                    {"text": "Status", "callback_data": "status"},
                    {"text": "Tools", "callback_data": "tools"},
                ],
                [
                    {"text": "Help", "callback_data": "help"},
                    {"text": "New Session", "callback_data": "new"},
                ],
            ]
        }

    @staticmethod
    def _telegram_command(text: str) -> str | None:
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        command = stripped.split(None, 1)[0][1:]
        if not command:
            return None
        if "@" in command:
            command = command.split("@", 1)[0]
        lowered = command.lower()
        if lowered in {"start", "help", "status", "new", "tools"}:
            return lowered
        return None

    @staticmethod
    def _outbound_target_for_channel(inbound: InboundMessage) -> str:
        if inbound.source in {"telegram", "slack"}:
            return inbound.channel
        return inbound.user_id

    async def _telegram_webhook(self, request: web.Request) -> web.Response:
        try:
            body = await request.read()
            ensure_body_size(
                body,
                max_bytes=getattr(self.settings, "max_request_body_bytes", 262_144),
            )
            verify_webhook(
                "telegram",
                dict(request.headers),
                body,
                url=str(request.url),
                secrets=self.webhook_secrets,
            )
            payload = await self._parse_payload(request, body)
            await self._replay_guard.check_webhook(
                source="telegram",
                payload=payload,
                headers=dict(request.headers),
                body=body,
            )
            ack, status = await self.ingest_message("telegram", payload)
            return self._ack_response(ack, status)
        except ReplayAttackDetected as e:
            return web.json_response(
                {"error": "Conflict", "detail": str(e)}, status=409
            )
        except PermissionError as e:
            return web.json_response(
                {"error": "Forbidden", "detail": str(e)}, status=403
            )
        except ValueError as e:
            return web.json_response(
                {"error": "Bad Request", "detail": str(e)}, status=400
            )
        except Exception as e:
            return web.json_response(
                {"error": "Internal Server Error", "detail": str(e)}, status=500
            )

    async def _slack_webhook(self, request: web.Request) -> web.Response:
        try:
            body = await request.read()
            ensure_body_size(
                body,
                max_bytes=getattr(self.settings, "max_request_body_bytes", 262_144),
            )
            verify_webhook(
                "slack",
                dict(request.headers),
                body,
                url=str(request.url),
                secrets=self.webhook_secrets,
            )
            payload = await self._parse_payload(request, body)
            await self._replay_guard.check_webhook(
                source="slack",
                payload=payload,
                headers=dict(request.headers),
                body=body,
            )
            if payload.get("type") == "url_verification" and payload.get("challenge"):
                return web.json_response(
                    {"challenge": payload["challenge"]}, status=200
                )
            event_payload = (
                payload.get("event")
                if isinstance(payload.get("event"), dict)
                else payload
            )
            ack, status = await self.ingest_message("slack", event_payload)
            return self._ack_response(ack, status)
        except ReplayAttackDetected as e:
            return web.json_response(
                {"error": "Conflict", "detail": str(e)}, status=409
            )
        except PermissionError as e:
            return web.json_response(
                {"error": "Forbidden", "detail": str(e)}, status=403
            )
        except ValueError as e:
            return web.json_response(
                {"error": "Bad Request", "detail": str(e)}, status=400
            )
        except Exception as e:
            return web.json_response(
                {"error": "Internal Server Error", "detail": str(e)}, status=500
            )

    async def _whatsapp_webhook(self, request: web.Request) -> web.Response:
        try:
            body = await request.read()
            ensure_body_size(
                body,
                max_bytes=getattr(self.settings, "max_request_body_bytes", 262_144),
            )
            verify_webhook(
                "whatsapp",
                dict(request.headers),
                body,
                url=str(request.url),
                secrets=self.webhook_secrets,
            )
            payload = await self._parse_payload(request, body)
            await self._replay_guard.check_webhook(
                source="whatsapp",
                payload=payload,
                headers=dict(request.headers),
                body=body,
            )
            ack, status = await self.ingest_message("whatsapp", payload)
            return self._ack_response(ack, status)
        except ReplayAttackDetected as e:
            return web.json_response(
                {"error": "Conflict", "detail": str(e)}, status=409
            )
        except PermissionError as e:
            return web.json_response(
                {"error": "Forbidden", "detail": str(e)}, status=403
            )
        except ValueError as e:
            return web.json_response(
                {"error": "Bad Request", "detail": str(e)}, status=400
            )
        except Exception as e:
            return web.json_response(
                {"error": "Internal Server Error", "detail": str(e)}, status=500
            )

    async def _dashboard_data(self, request: web.Request) -> web.Response:
        return web.json_response(self._snapshot_payload())

    async def _health(self, request: web.Request) -> web.Response:
        state = self.dashboard_state.snapshot(
            personal_context=(
                self.personal_store.summary() if self.personal_store is not None else {}
            ),
            capabilities=(
                self.capabilities_provider()
                if self.capabilities_provider is not None
                else []
            ),
        )
        return web.json_response(
            {
                "status": "ok",
                "active_tasks": state["active_tasks"],
                "hub_enabled": self.enabled,
            }
        )

    async def _dashboard_page(self, request: web.Request) -> web.Response:
        html = self._render_dashboard_html()
        return web.Response(text=html, content_type="text/html")

    async def _dashboard_ws(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(heartbeat=30.0)
        await websocket.prepare(request)

        async with self._ws_lock:
            self._websockets.add(websocket)

        await websocket.send_json(self._snapshot_payload())
        try:
            async for message in websocket:
                if message.type in {WSMsgType.TEXT, WSMsgType.BINARY}:
                    continue
                if message.type in {
                    WSMsgType.ERROR,
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSING,
                }:
                    break
        finally:
            async with self._ws_lock:
                self._websockets.discard(websocket)
        return websocket

    def _render_dashboard_html(self) -> str:
        return render_dashboard_html()

    def _snapshot_payload(self) -> dict[str, Any]:
        snapshot = self.dashboard_state.snapshot(
            personal_context=(
                self.personal_store.summary() if self.personal_store is not None else {}
            ),
            capabilities=(
                self.capabilities_provider()
                if self.capabilities_provider is not None
                else []
            ),
        )
        snapshot["channel_access"] = self.channel_access.summary()
        return dict(sanitize_structure(snapshot, max_string_length=2_000, max_items=40))

    def _normalize_inbound_message(
        self, source: str, payload: dict[str, Any]
    ) -> InboundMessage:
        normalized = self._coerce_provider_payload(source, payload)
        normalized = {
            key: value for key, value in normalized.items() if value is not None
        }
        inbound = InboundMessage.model_validate({**normalized, "source": source})
        if inbound.session_id is None:
            user_key = inbound.user_id or f"{source}:{inbound.channel}"
            session_id = self._session_by_user.get(user_key) or str(uuid4())
            self._session_by_user[user_key] = session_id
            inbound = inbound.model_copy(update={"session_id": session_id})
        elif inbound.user_id:
            self._session_by_user[inbound.user_id] = inbound.session_id
        return inbound

    @staticmethod
    def _coerce_provider_payload(
        source: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if source == "telegram":
            callback = (
                payload.get("callback_query")
                if isinstance(payload.get("callback_query"), dict)
                else None
            )
            if callback:
                message = (
                    callback.get("message")
                    if isinstance(callback.get("message"), dict)
                    else {}
                )
                sender = (
                    callback.get("from")
                    if isinstance(callback.get("from"), dict)
                    else {}
                )
                chat = (
                    message.get("chat") if isinstance(message.get("chat"), dict) else {}
                )
                callback_data = str(callback.get("data") or "")
                return {
                    "user_id": str(
                        sender.get("id")
                        or sender.get("username")
                        or chat.get("id")
                        or "telegram"
                    ),
                    "channel": str(chat.get("id") or "telegram"),
                    "text": callback_data or "/help",
                    "timestamp": message.get("date") or payload.get("timestamp"),
                    "session_id": payload.get("session_id"),
                    "priority": int(payload.get("priority") or 1),
                    "message_id": (
                        str(message.get("message_id"))
                        if message.get("message_id")
                        else None
                    ),
                    "reply_to_message_id": message.get("message_id"),
                    "metadata": {
                        "telegram_callback_query_id": str(callback.get("id") or ""),
                        "telegram_callback_data": callback_data,
                    },
                }
            message = (
                payload.get("message")
                if isinstance(payload.get("message"), dict)
                else payload
            )
            chat = (
                message.get("chat")
                if isinstance(message, dict) and isinstance(message.get("chat"), dict)
                else {}
            )
            sender = (
                message.get("from")
                if isinstance(message, dict) and isinstance(message.get("from"), dict)
                else {}
            )
            chat_id = chat.get("id")
            reply_to = (
                message.get("reply_to_message") if isinstance(message, dict) else None
            )
            reply_to_id = (
                reply_to.get("message_id") if isinstance(reply_to, dict) else None
            )
            return {
                "user_id": str(
                    sender.get("id")
                    or sender.get("username")
                    or chat_id
                    or payload.get("user_id")
                    or "telegram"
                ),
                "channel": str(chat_id or payload.get("channel") or "telegram"),
                "text": str(message.get("text") or payload.get("text") or ""),
                "timestamp": message.get("date") or payload.get("timestamp"),
                "session_id": payload.get("session_id"),
                "priority": int(payload.get("priority") or 1),
                "message_id": (
                    str(message.get("message_id"))
                    if isinstance(message, dict) and message.get("message_id")
                    else (
                        str(payload.get("message_id"))
                        if payload.get("message_id")
                        else None
                    )
                ),
                "reply_to_message_id": reply_to_id
                or payload.get("reply_to_message_id"),
                "metadata": dict(payload.get("metadata") or {}),
            }

        if source == "slack":
            event = (
                payload.get("event") if isinstance(payload.get("event"), dict) else {}
            )
            return {
                "user_id": str(
                    payload.get("user_id")
                    or payload.get("user")
                    or event.get("user")
                    or "slack"
                ),
                "channel": str(
                    payload.get("channel") or event.get("channel") or "slack"
                ),
                "text": str(payload.get("text") or event.get("text") or ""),
                "timestamp": payload.get("ts")
                or event.get("ts")
                or payload.get("timestamp"),
                "session_id": payload.get("session_id"),
                "priority": int(payload.get("priority") or 1),
                "message_id": str(
                    payload.get("event_id")
                    or event.get("ts")
                    or payload.get("message_id")
                    or ""
                )
                or None,
                "metadata": dict(payload.get("metadata") or {}),
            }

        if source == "whatsapp":
            sender = (
                payload.get("From")
                or payload.get("from")
                or payload.get("wa_id")
                or payload.get("user_id")
            )
            body = payload.get("Body") or payload.get("body") or payload.get("text")
            return {
                "user_id": str(sender or "whatsapp"),
                "channel": str(payload.get("channel") or "whatsapp"),
                "text": str(body or ""),
                "timestamp": payload.get("timestamp"),
                "session_id": payload.get("session_id"),
                "priority": int(payload.get("priority") or 1),
                "message_id": str(
                    payload.get("MessageSid")
                    or payload.get("SmsSid")
                    or payload.get("message_id")
                    or ""
                )
                or None,
                "metadata": dict(payload.get("metadata") or {}),
            }

        return payload

    async def _parse_payload(self, request: web.Request, body: bytes) -> dict[str, Any]:
        content_type = request.content_type or ""
        if "json" in content_type:
            try:
                data = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict):
                return data
        if request.can_read_body:
            post_data = await request.post()
            if post_data:
                return dict(post_data)
        return {}

    @staticmethod
    def _ack_response(ack: HubAck, status: int) -> web.Response:
        response = web.json_response(ack.model_dump(), status=status)
        if not ack.accepted and ack.retry_after_seconds is not None:
            response.headers["Retry-After"] = str(ack.retry_after_seconds)
        return response
