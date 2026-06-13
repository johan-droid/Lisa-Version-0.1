from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from interfaces.dashboard import render_dashboard_html
from lisa.config import Settings
from lisa.events import EventBus, LisaEvent
from lisa.hub import MessageHub


def test_dashboard_html_includes_current_and_legacy_ws_paths() -> None:
    html = render_dashboard_html()

    assert "/ws" in html
    assert "/ws/dashboard" in html
    assert "Chart.js" in html


def test_aiohttp_message_hub_handles_slack_verification_and_telegram_ingest(
    tmp_path: Path,
) -> None:
    class StubConductor:
        def __init__(self) -> None:
            self.messages = []

        def try_submit_message(self, message):
            self.messages.append(message)
            return f"job-{len(self.messages)}"

    async def run() -> None:
        settings = Settings(
            workspace_root=tmp_path,
            db_path=tmp_path / "data" / "test.db",
            skills_dir=tmp_path / "skills",
            persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
            gating_model_path=tmp_path / "data" / "gating_model.pkl",
            enable_browser_tools=False,
            message_hub_enabled=False,
            message_hub_host="localhost",
            message_hub_port=8800,
        )
        conductor = StubConductor()
        hub = MessageHub(settings=settings, event_bus=EventBus(), conductor=conductor)

        server = TestServer(hub._app)
        await server.start_server()
        client = TestClient(server)
        await client.start_server()
        try:
            verification = await client.post(
                "/slack/events",
                json={"type": "url_verification", "challenge": "abc123"},
            )
            assert verification.status == 200
            assert await verification.json() == {"challenge": "abc123"}

            ingest = await client.post(
                "/telegram/webhook",
                json={
                    "message": {
                        "from": {"id": "u-1"},
                        "chat": {"id": "c-1", "type": "private"},
                        "text": "hello lisa",
                    }
                },
            )
            assert ingest.status == 202
            body = await ingest.json()
            assert body["accepted"] is True
            assert body["queued"] is True
            assert conductor.messages
            assert conductor.messages[0].session_id is not None

            websocket = await client.ws_connect("/ws")
            try:
                snapshot = await asyncio.wait_for(websocket.receive_json(), timeout=2)
                assert "active_tasks" in snapshot
                assert "charts" in snapshot
            finally:
                await websocket.close()
        finally:
            await client.close()
            await server.close()

    asyncio.run(run())


def test_aiohttp_message_hub_delivers_chat_response_async(tmp_path: Path) -> None:
    class StubConductor:
        def __init__(self) -> None:
            self.messages = []

        def try_submit_message(self, message):
            self.messages.append(message)
            return f"job-{len(self.messages)}"

    async def run() -> None:
        settings = Settings(
            workspace_root=tmp_path,
            db_path=tmp_path / "data" / "test.db",
            skills_dir=tmp_path / "skills",
            persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
            gating_model_path=tmp_path / "data" / "gating_model.pkl",
            enable_browser_tools=False,
            message_hub_enabled=False,
            message_hub_host="127.0.0.1",
            message_hub_port=8800,
        )
        conductor = StubConductor()
        event_bus = EventBus()
        hub = MessageHub(settings=settings, event_bus=event_bus, conductor=conductor)

        # Mock send_response to record calls
        send_calls = []

        async def mock_send_response(channel, *, user_id, text, **kwargs):
            send_calls.append((channel, user_id, text))
            return {"delivered": True}

        hub.channel_gateway.send_response = mock_send_response

        server = TestServer(hub._app)
        await server.start_server()
        client = TestClient(server)
        await client.start_server()

        # Start observing events manually
        event_task = asyncio.create_task(hub._observe_events())
        try:
            # 1. Ingest a message to register a delivery target
            ingest = await client.post(
                "/telegram/webhook",
                json={
                    "message": {
                        "from": {"id": "user-abc"},
                        "chat": {"id": "chat-xyz", "type": "private"},
                        "text": "hi lisa",
                    }
                },
            )
            assert ingest.status == 202
            assert conductor.messages
            inbound = conductor.messages[0]
            session_id = inbound.session_id
            assert session_id is not None

            # 2. Publish a chat.responded event
            await event_bus.publish(
                LisaEvent(
                    type="chat.responded",
                    payload={
                        "message": "Hello from the test!",
                        "session_id": session_id,
                    },
                    session_id=session_id,
                )
            )

            # Wait a moment for async delivery task to execute
            for _ in range(20):
                if send_calls:
                    break
                await asyncio.sleep(0.1)

            assert send_calls == [("telegram", "chat-xyz", "Hello from the test!")]

        finally:
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass
            await client.close()
            await server.close()

    asyncio.run(run())
