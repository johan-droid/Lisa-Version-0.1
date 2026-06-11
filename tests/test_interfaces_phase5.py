from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from interfaces.dashboard import render_dashboard_html
from lisa.config import Settings
from lisa.events import EventBus
from lisa.hub import MessageHub


def test_dashboard_html_includes_current_and_legacy_ws_paths() -> None:
    html = render_dashboard_html()

    assert "/ws" in html
    assert "/ws/dashboard" in html
    assert "Chart.js" in html


def test_aiohttp_message_hub_handles_slack_verification_and_telegram_ingest(tmp_path: Path) -> None:
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
