from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from interfaces.dashboard import render_dashboard_html
from lisa.api import create_app
from lisa.config import Settings
from lisa.local_inference import ToolCallParser
from lisa.memory_system import HybridMemoryCoordinator
from safety.input_sanitizer import sanitize_structure, sanitize_user_visible_text


def build_security_client(
    tmp_path: Path, *, max_request_body_bytes: int = 262_144
) -> TestClient:
    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        message_hub_enabled=False,
        evolution_enabled=False,
        incoming_queue_size=16,
        admin_api_token="test-admin-token",
        telegram_webhook_secret="telegram-secret",
        max_request_body_bytes=max_request_body_bytes,
    )
    return TestClient(create_app(settings))


def test_sanitize_user_visible_text_redacts_secrets_and_strips_bidi_controls() -> None:
    text = "Bearer supersecrettoken123 \u202e hidden sk-abcdefghijklmnopqrstuvwxyz1234"
    cleaned = sanitize_user_visible_text(text)
    assert "supersecrettoken123" not in cleaned
    assert "sk-abcdefghijklmnopqrstuvwxyz1234" not in cleaned
    assert "\u202e" not in cleaned
    assert "[redacted]" in cleaned


def test_sanitize_structure_redacts_nested_secret_fields() -> None:
    cleaned = sanitize_structure(
        {
            "authorization": "Bearer abcdefghijklmnop",
            "nested": {"api_key": "secret-key", "message": "ok"},
        }
    )
    assert cleaned["authorization"] == "[redacted]"
    assert cleaned["nested"]["api_key"] == "[redacted]"
    assert cleaned["nested"]["message"] == "ok"


def test_dashboard_template_escapes_script_breakout() -> None:
    html = render_dashboard_html('"</script><script>alert(1)</script>')
    assert "</script><script>alert(1)</script>" not in html
    assert "<\\/script>" in html


def test_telegram_webhook_replay_returns_409(tmp_path: Path) -> None:
    payload = {
        "update_id": 9991,
        "message": {
            "message_id": 77,
            "text": "hello",
            "date": 1718200000,
            "chat": {"id": 12345},
            "from": {"id": 12345},
        },
    }
    with build_security_client(tmp_path) as client:
        first = client.post(
            "/telegram/webhook",
            json=payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )
        assert first.status_code == 202

        second = client.post(
            "/telegram/webhook",
            json=payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )
        assert second.status_code == 409
        assert "Replay attack detected" in second.json()["detail"]


def test_oversized_request_returns_413(tmp_path: Path) -> None:
    big_message = "a" * 500
    with build_security_client(tmp_path, max_request_body_bytes=128) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "lisa",
                "messages": [{"role": "user", "content": big_message}],
            },
        )
        assert response.status_code == 413


def test_tool_call_parser_handles_large_payload_without_backtracking() -> None:
    parser = ToolCallParser()
    huge = (
        ("x" * 40_000)
        + '<tool_call>{"name":"search_notepad","arguments":{"query":"safe"}}</tool_call>'
    )
    generation = parser.parse(huge)
    assert generation.tool_calls == []
    assert len(generation.text) <= parser.MAX_PARSE_CHARS


def test_memory_audit_redacts_secret_payloads() -> None:
    async def scenario() -> None:
        memory = HybridMemoryCoordinator(agent_id="lisa", namespace="security-test")
        await memory.start()
        await memory.record_audit(
            component="tool_executor",
            event_type="tool_call",
            payload={"authorization": "Bearer secret-token-123456", "note": "safe"},
            session_id="s1",
            task_id="t1",
        )
        rows = await memory.latest_entries(limit=5)
        await memory.close()
        assert rows[0]["payload"]["authorization"] == "[redacted]"
        assert rows[0]["payload"]["note"] == "safe"

    asyncio.run(scenario())
