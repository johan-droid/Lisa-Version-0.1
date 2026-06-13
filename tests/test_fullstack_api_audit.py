from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psutil
from fastapi.testclient import TestClient

from lisa.api import create_app
from lisa.config import Settings

ADMIN_TOKEN = "test-admin-token"


def build_audit_client(tmp_path: Path, queue_size: int = 128) -> TestClient:
    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        message_hub_enabled=False,
        evolution_enabled=False,
        incoming_queue_size=queue_size,
        admin_api_token=ADMIN_TOKEN,
        enable_unsafe_admin_endpoints=False,
    )
    return TestClient(create_app(settings))


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def test_fullstack_control_plane_endpoint_sweep(tmp_path: Path) -> None:
    with build_audit_client(tmp_path) as client:
        safe_gets = [
            "/health",
            "/state",
            "/tools",
            "/personas",
            "/gating?text=design%20a%20secure%20api",
            "/dashboard",
            "/dashboard/live",
            "/dashboard/snapshot",
            "/personal",
            "/v1/channels",
            "/notepad/search?q=missing",
        ]
        for path in safe_gets:
            response = client.get(path)
            assert response.status_code == 200, path

        chat = client.post("/chat", json={"message": "Audit the API briefly."})
        assert chat.status_code == 200
        assert chat.json()["message"]

        chat_completion = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "model": "lisa",
            },
        )
        assert chat_completion.status_code == 200
        assert chat_completion.json()["choices"][0]["message"]["role"] == "assistant"

        responses = client.post(
            "/v1/responses", json={"input": "Say hello", "model": "lisa"}
        )
        assert responses.status_code == 200
        assert responses.json()["output"][0]["role"] == "assistant"

        embeddings = client.post(
            "/v1/embeddings", json={"input": ["alpha", "beta"], "model": "auto"}
        )
        assert embeddings.status_code == 200
        assert len(embeddings.json()["data"]) == 2

        tool = client.post(
            "/tools/dashboard_update",
            json={"arguments": {"metric": "audit", "value": "ok"}},
        )
        assert tool.status_code == 200
        assert tool.json()["success"] is True

        missing_tool = client.post("/tools/not_registered", json={"arguments": {}})
        assert missing_tool.status_code == 404

        with client.websocket_connect("/ws/dashboard") as websocket:
            snapshot = websocket.receive_json()
            assert "active_tasks" in snapshot
            assert "capabilities" in snapshot


def test_admin_and_multiplexer_endpoints_are_guarded_and_functional(
    tmp_path: Path,
) -> None:
    with build_audit_client(tmp_path) as client:
        guarded_posts = [
            ("/bots/connect", {"source": "telegram", "user_id": "u1"}),
            (
                "/ingest/direct",
                {
                    "source": "direct",
                    "user_id": "u1",
                    "channel": "api",
                    "text": "hello",
                },
            ),
            (
                "/v1/messages/ingest/direct",
                {
                    "source": "direct",
                    "user_id": "u1",
                    "channel": "api",
                    "text": "hello",
                },
            ),
            ("/v1/messages/dispatch", {"channel": "direct", "text": "hello"}),
            ("/v1/channels/authorize", {"source": "telegram", "user_id": "u1"}),
            ("/v1/channels/revoke", {"source": "telegram", "user_id": "u1"}),
            ("/admin/runtime/shutdown", {}),
            ("/admin/runtime/shed-memory", {}),
        ]
        for path, payload in guarded_posts:
            response = client.post(path, json=payload)
            assert response.status_code in {403, 503}, path

        connect = client.post(
            "/bots/connect",
            json={"source": "telegram", "user_id": "u1"},
            headers=admin_headers(),
        )
        assert connect.status_code == 200
        assert connect.json()["updated"] is True

        revoke = client.post(
            "/v1/channels/revoke",
            json={"source": "telegram", "user_id": "u1"},
            headers=admin_headers(),
        )
        assert revoke.status_code == 200
        assert revoke.json()["updated"] is True

        direct_ingest = client.post(
            "/ingest/direct",
            json={
                "source": "direct",
                "user_id": "u1",
                "channel": "api",
                "text": "hello",
            },
            headers=admin_headers(),
        )
        assert direct_ingest.status_code == 202
        assert direct_ingest.json()["queued"] is True

        multiplex_ingest = client.post(
            "/v1/messages/ingest/direct",
            json={
                "source": "direct",
                "user_id": "u2",
                "channel": "api",
                "text": "hello again",
            },
            headers=admin_headers(),
        )
        assert multiplex_ingest.status_code == 202
        assert multiplex_ingest.json()["queued"] is True

        dispatch = client.post(
            "/v1/messages/dispatch",
            json={"channel": "direct", "text": "respond through the multiplexer"},
            headers=admin_headers(),
        )
        assert dispatch.status_code == 200
        assert dispatch.json()["accepted"] is True

        shutdown = client.post(
            "/admin/runtime/shutdown", json={}, headers=admin_headers()
        )
        assert shutdown.status_code == 403
        assert "disabled" in shutdown.json()["detail"].lower()

        shed_memory = client.post(
            "/admin/runtime/shed-memory", json={}, headers=admin_headers()
        )
        assert shed_memory.status_code == 403
        assert "disabled" in shed_memory.json()["detail"].lower()


def test_public_webhook_security_and_channel_access_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LISA_TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    monkeypatch.setenv("LISA_SLACK_ALLOWED_USER_IDS", "U_ALLOWED")

    with build_audit_client(tmp_path) as client:
        bad_telegram = client.post(
            "/telegram/webhook",
            json={
                "message": {"text": "hello", "from": {"id": "u1"}, "chat": {"id": "c1"}}
            },
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        )
        assert bad_telegram.status_code == 403
        assert bad_telegram.json()["accepted"] is False

        good_telegram = client.post(
            "/telegram/webhook",
            json={
                "message": {
                    "text": "/status",
                    "from": {"id": "u1"},
                    "chat": {"id": "c1"},
                }
            },
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )
        assert good_telegram.status_code == 202
        assert good_telegram.json()["accepted"] is True

        denied_slack = client.post(
            "/slack/events",
            json={"user": "U_DENIED", "channel": "C1", "text": "hello"},
        )
        assert denied_slack.status_code == 202
        assert denied_slack.json()["accepted"] is False


def test_parallel_api_stress_keeps_control_plane_responsive(tmp_path: Path) -> None:
    process = psutil.Process()

    with build_audit_client(tmp_path, queue_size=256) as client:
        before_rss = process.memory_info().rss

        def request_round(index: int) -> tuple[int, int]:
            if index % 4 == 0:
                response = client.get("/health")
            elif index % 4 == 1:
                response = client.get("/dashboard/snapshot")
            elif index % 4 == 2:
                response = client.post(
                    "/chat", json={"message": f"stress message {index}"}
                )
            else:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "lisa",
                        "messages": [
                            {"role": "user", "content": f"stress completion {index}"}
                        ],
                    },
                )
            return response.status_code, len(response.content)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(request_round, index) for index in range(40)]
            results = [future.result(timeout=60) for future in as_completed(futures)]

        after_rss = process.memory_info().rss
        post_stress_health = client.get("/health")

    assert all(status == 200 for status, _ in results)
    assert all(size > 0 for _, size in results)
    assert post_stress_health.status_code == 200
    assert after_rss - before_rss < 250 * 1024 * 1024
