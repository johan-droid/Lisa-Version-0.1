from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from lisa.agent import LisaRuntime
from lisa.conductor import ConductorJob, TaskConductor
from lisa.events import EventBus
from lisa.api import create_app
from lisa.config import Settings
from lisa.constitutions import ConstitutionMode
from lisa.gating import PersonaGatingNetwork
from lisa.evolution import EvolutionCycleResult, NightlyEvolutionScheduler
from lisa.hub import DashboardMetricsState, MessageHub
from lisa.events import LisaEvent
from lisa.local_inference import BrainGeneration, ToolCallParser
from lisa.llm import LLMClient
from lisa.notepad import Notepad
from lisa.schemas import (
    BrainTask,
    ChatRequest,
    ChatResponse,
    InboundMessage,
    ToolCall,
    ToolResult,
)
from lisa.soft_prompts import PersonaSoftPromptBank
from lisa.tool_executor import ToolExecutor
from lisa.tools import ToolSpec


def build_client(tmp_path: Path, queue_size: int = 16) -> TestClient:
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
        admin_api_token="test-admin-token",
    )
    app = create_app(settings)
    return TestClient(app)


class StubToolExecutor:
    async def execute_many(
        self,
        tool_calls: list[ToolCall],
        constitution: object,
        session_id: str | None = None,
    ) -> list[ToolResult]:
        return [
            ToolResult(
                tool=call.name,
                success=True,
                output={
                    "tool": call.name,
                    "arguments": call.arguments,
                    "session_id": session_id,
                },
            )
            for call in tool_calls
        ]


def test_health_returns_ok(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["constitution"] == "restricted"


def test_chat_logs_interaction(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/chat", json={"message": "Implement a secure API endpoint"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["constitution"] == "restricted"
        assert body["used_external_model"] is False
        assert body["personas"]["architect"] > 0

        search = client.get("/notepad/search", params={"q": "secure"})
        assert search.status_code == 200
        assert len(search.json()["results"]) >= 2


def test_conductor_writes_task_summary_to_notepad(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/chat", json={"message": "Summarize this implementation"}
        )

        assert response.status_code == 200

        search = client.get("/notepad/search", params={"q": "summary"})
        assert search.status_code == 200
        results = search.json()["results"]
        assert any(row["entry_type"] == "task_summary" for row in results)
        summary = next(row for row in results if row["entry_type"] == "task_summary")
        assert summary["payload"]["outcome"] == "success"
        assert "self_critique" in summary["payload"]


def test_v1_endpoints(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        # Test /v1/chat/completions
        chat_payload = {
            "model": "lisa-chat-model",
            "messages": [{"role": "user", "content": "How are you doing today?"}],
            "max_tokens": 500,
            "user": "test_user_session",
        }
        res_chat = client.post("/v1/chat/completions", json=chat_payload)
        assert res_chat.status_code == 200
        data_chat = res_chat.json()
        assert data_chat["object"] == "chat.completion"
        assert "choices" in data_chat
        assert len(data_chat["choices"]) == 1
        assert data_chat["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data_chat["choices"][0]["message"]
        assert "usage" in data_chat

        # Test /v1/responses
        resp_payload = {
            "model": "lisa-responses-model",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the capital of France?"}
                    ],
                }
            ],
            "max_tokens": 500,
            "user": "test_user_session_resp",
        }
        res_resp = client.post("/v1/responses", json=resp_payload)
        assert res_resp.status_code == 200
        data_resp = res_resp.json()
        assert data_resp["object"] == "response"
        assert "output" in data_resp
        assert len(data_resp["output"]) == 1
        assert data_resp["output"][0]["type"] == "message"
        assert data_resp["output"][0]["role"] == "assistant"
        assert data_resp["output"][0]["content"][0]["text"] is not None
        assert "usage" in data_resp

        # Test /v1/embeddings (using mock path since db_path has 'test' in it)
        emb_payload = {"input": ["apple", "banana"], "model": "auto"}
        res_emb = client.post("/v1/embeddings", json=emb_payload)
        assert res_emb.status_code == 200
        data_emb = res_emb.json()
        assert data_emb["object"] == "list"
        assert len(data_emb["data"]) == 2
        assert data_emb["data"][0]["object"] == "embedding"
        assert len(data_emb["data"][0]["embedding"]) == 768


def test_v1_streaming_and_live_dashboard_surface(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        stream_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "lisa-chat-model",
                "messages": [{"role": "user", "content": "stream this"}],
                "stream": True,
            },
        )
        assert stream_response.status_code == 200
        assert "chat.completion.chunk" in stream_response.text
        assert "[DONE]" in stream_response.text

        responses_stream = client.post(
            "/v1/responses",
            json={
                "model": "lisa-responses-model",
                "input": "stream this too",
                "stream": True,
            },
        )
        assert responses_stream.status_code == 200
        assert "response.created" in responses_stream.text
        assert "response.completed" in responses_stream.text

        dashboard_live = client.get("/dashboard/live")
        assert dashboard_live.status_code == 200
        assert "LISA Live Dashboard" in dashboard_live.text

        dashboard_snapshot = client.get("/dashboard/snapshot")
        assert dashboard_snapshot.status_code == 200
        assert "channel_access" in dashboard_snapshot.json()


def test_memory_layer_uses_hybrid_facade(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        memory = client.app.state.memory
        notepad = client.app.state.runtime.notepad
        assert notepad.memory is memory
        assert memory.working_memory_key("task-1") == "session:lisa:task-1"


def test_dashboard_metrics_state_tracks_live_events() -> None:
    state = DashboardMetricsState()
    state.observe(LisaEvent(type="conductor.job_started", payload={"job_id": "job-1"}))
    state.observe(LisaEvent(type="conductor.job_finished", payload={"job_id": "job-1"}))
    state.observe(
        LisaEvent(
            type="chat.responded",
            payload={"personas": {"architect": 0.75, "oracle": 0.25}},
        )
    )
    state.observe(
        LisaEvent(
            type="external_llm.completed",
            payload={"provider": "openai", "usage": {"total_tokens": 42}},
        )
    )
    state.observe(
        LisaEvent(
            type="ledger.append",
            payload={"entry_type": "evolution_cycle"},
        )
    )
    state.observe(
        LisaEvent(
            type="evolution.skill_registered",
            payload={"skill_name": "retry_guard"},
        )
    )

    snapshot = state.snapshot()
    assert snapshot["active_tasks"] == 0
    assert snapshot["token_consumption"]["total"] == 42
    assert snapshot["dominant_persona"] == "architect"
    assert snapshot["evolution_rate"] > 0
    assert snapshot["last_evolution_skill"] == "retry_guard"
    assert snapshot["last_evolution_status"] == "registered"


def test_message_hub_renders_chart_dashboard(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        message_hub_enabled=False,
    )

    class StubConductor:
        def try_submit_message(self, message: InboundMessage) -> str | None:
            return "job-1"

    hub = MessageHub(settings=settings, event_bus=EventBus(), conductor=StubConductor())
    html = hub._render_dashboard_html()

    assert "Chart.js" in html
    assert "/ws/dashboard" in html


def test_enable_unrestricted_mode_requires_reason(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post("/chat", json={"message": "ENABLE UNRESTRICTED MODE"})

        assert response.status_code == 200
        assert response.json()["constitution"] == "restricted"
        assert "not enabled" in response.json()["message"]


def test_enable_and_disable_unrestricted_mode(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        enable = client.post(
            "/chat",
            json={
                "message": "ENABLE UNRESTRICTED MODE benchmark a trusted sandbox build"
            },
        )
        assert enable.status_code == 200
        assert enable.json()["constitution"] == "unrestricted"

        state = client.get("/state")
        assert state.status_code == 200
        assert state.json()["mode"] == "unrestricted"

        disable = client.post("/chat", json={"message": "DISABLE UNRESTRICTED MODE"})
        assert disable.status_code == 200
        assert disable.json()["constitution"] == "restricted"


def test_tool_invocation_can_write_dashboard_metric(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/tools/dashboard_update",
            json={"arguments": {"metric": "tasks", "value": "1"}},
        )

        assert response.status_code == 200
        dashboard = client.get("/dashboard")
        assert dashboard.status_code == 200
        assert dashboard.json()[0]["metric"] == "tasks"


def test_websocket_receives_events(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        client.post("/chat", json={"message": "Summarize the current plan"})

        with client.websocket_connect("/ws/events") as websocket:
            first = websocket.receive_json()
            assert first["type"] in {
                "conductor.job_started",
                "chat.received",
                "ledger.append",
            }


def test_telegram_ingest_accepts_message(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/telegram/webhook",
            json={
                "source": "telegram",
                "user_id": "u1",
                "channel": "dm",
                "text": "hello lisa",
            },
        )

        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] is True
        assert body["queued"] is True
        assert body["job_id"] is not None


def test_channel_access_blocks_unauthorized_sender(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        authorize = client.post(
            "/v1/channels/authorize",
            json={"source": "telegram", "user_id": "userA"},
            headers={"Authorization": "Bearer test-admin-token"},
        )
        assert authorize.status_code == 200

        response = client.post(
            "/telegram/webhook",
            json={
                "source": "telegram",
                "user_id": "userB",
                "channel": "dm",
                "text": "hello",
            },
        )
        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] is False
        assert body["queued"] is False
        assert body["detail"] == "Sender is not authorized for this channel."

        response = client.post(
            "/telegram/webhook",
            json={
                "source": "telegram",
                "user_id": "userA",
                "channel": "dm",
                "text": "actual instruction to lisa",
            },
        )
        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] is True
        assert body["queued"] is True
        assert body["job_id"] is not None


def test_telegram_shortcuts_handle_start_and_callback_query(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/telegram/webhook",
            json={
                "source": "telegram",
                "user_id": "userA",
                "channel": "dm",
                "text": "/start",
            },
        )
        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] is True
        assert body["queued"] is False
        assert body["detail"] == "Telegram onboarding delivered."

        callback = client.post(
            "/telegram/webhook",
            json={
                "callback_query": {
                    "id": "cb-1",
                    "data": "status",
                    "from": {"id": "userA"},
                    "message": {
                        "message_id": 42,
                        "chat": {"id": "chat-1", "type": "private"},
                    },
                }
            },
        )
        assert callback.status_code == 202
        callback_body = callback.json()
        assert callback_body["accepted"] is True
        assert callback_body["queued"] is False
        assert callback_body["detail"] == "Telegram status delivered."


def test_admin_dispatch_multiplexer_requires_token_and_returns_response(
    tmp_path: Path,
) -> None:
    with build_client(tmp_path) as client:
        denied = client.post(
            "/v1/messages/dispatch",
            json={"channel": "direct", "text": "hello"},
        )
        assert denied.status_code == 403

        allowed = client.post(
            "/v1/messages/dispatch",
            json={"channel": "direct", "text": "hello"},
            headers={"Authorization": "Bearer test-admin-token"},
        )
        assert allowed.status_code == 200
        body = allowed.json()
        assert body["accepted"] is True
        assert body["response"]["message"]


def test_webhook_error_handling(tmp_path: Path, monkeypatch) -> None:
    # Test PermissionError by setting a Telegram webhook secret and passing incorrect secret
    monkeypatch.setenv("LISA_TELEGRAM_WEBHOOK_SECRET", "my_super_webhook_secret")
    with build_client(tmp_path) as client:
        response = client.post(
            "/telegram/webhook",
            json={"text": "hello"},
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong_secret"},
        )
        assert response.status_code == 403
        body = response.json()
        assert body["accepted"] is False
        assert "Invalid signature" in body["detail"]

        # Pass correct webhook token
        response = client.post(
            "/telegram/webhook",
            json={"text": "hello"},
            headers={"X-Telegram-Bot-Api-Secret-Token": "my_super_webhook_secret"},
        )
        # Should be 202 because verification passes (even if gating requires security key)
        assert response.status_code == 202


def test_conductor_backpressure_returns_none_when_full(tmp_path: Path) -> None:
    event_bus = EventBus()

    class StubRuntime:
        async def process_chat(self, request: object) -> ChatResponse:
            return ChatResponse(
                session_id="s1",
                message="ok",
                constitution="restricted",
                personas={"architect": 1.0},
                tool_suggestions=[],
                used_external_model=False,
                notes=[],
            )

        async def process_message(
            self, inbound: InboundMessage, max_tokens: int = 800
        ) -> ChatResponse:
            return ChatResponse(
                session_id=inbound.session_id or "s1",
                message="ok",
                constitution="restricted",
                personas={"architect": 1.0},
                tool_suggestions=[],
                used_external_model=False,
                notes=[],
            )

    conductor = TaskConductor(
        runtime=StubRuntime(),
        tool_executor=StubToolExecutor(),
        event_bus=event_bus,
        queue_size=1,
    )
    conductor._queue.put_nowait(
        ConductorJob(
            priority=1,
            sequence=0,
            kind="message",
            payload={
                "message": {
                    "source": "telegram",
                    "user_id": "u1",
                    "channel": "dm",
                    "text": "hello",
                    "timestamp": "2026-06-11T00:00:00Z",
                    "session_id": None,
                    "priority": 1,
                }
            },
        )
    )

    rejected = conductor.try_submit_message(
        InboundMessage(
            source="telegram", user_id="u2", channel="dm", text="second", priority=1
        )
    )
    assert rejected is None


def test_tool_invocation_retries_once(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        registry = client.app.state.runtime.tools
        attempts = {"count": 0}

        async def flaky(
            arguments: dict[str, object], context: object
        ) -> dict[str, bool]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("boom")
            return {"ok": True}

        registry._register(
            ToolSpec(
                name="flaky",
                description="flaky test tool",
                restricted_safe=True,
                handler=flaky,
            )
        )

        response = client.post("/tools/flaky", json={"arguments": {}})
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert attempts["count"] == 2


def test_persona_bank_persists_and_blends(tmp_path: Path) -> None:
    path = tmp_path / "persona_vectors.npz"
    bank = PersonaSoftPromptBank.initialize(tokens=4, dims=8, seed=7)
    bank.save(path)

    loaded = PersonaSoftPromptBank.load(path)
    assert loaded.summary()["architect"]["shape"] == [4, 8]

    blended = loaded.blend({"architect": 0.75, "oracle": 0.25})
    assert blended.shape == (4, 8)
    assert blended.dtype.name == "float32"


def test_persona_endpoint_exposes_bank_metadata(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.get("/personas")

        assert response.status_code == 200
        body = response.json()
        assert "weights" in body
        assert "architect" in body["weights"]
        assert body["bank_path"].endswith("persona_vectors.npz")


def test_gating_endpoint_exposes_prediction(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.get(
            "/gating", params={"text": "design a secure api and audit it"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["metadata"]["personas"][0] == "architect"
        assert body["blend"]["architect"] > 0
        assert abs(sum(body["blend"].values()) - 1.0) < 0.01


def test_llm_client_exports_persona_prefix_shape(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
    )
    client = LLMClient(
        settings,
        persona_bank=PersonaSoftPromptBank.initialize(tokens=3, dims=5, seed=1),
    )
    prefix = client.persona_prefix({"architect": 1.0})

    assert prefix.shape == (3, 5)


def test_gating_model_persists_and_loads(tmp_path: Path) -> None:
    path = tmp_path / "gating.pkl"
    model = PersonaGatingNetwork.initialize(max_features=32, hidden_size=8, seed=4)
    model.save(path)

    loaded = PersonaGatingNetwork.load_or_initialize(path)
    prediction = loaded.predict_blend("build a secure api endpoint")

    assert set(prediction) == {
        "architect",
        "oracle",
        "guardian",
        "evolution_engine",
        "distributed_mind",
    }
    assert abs(sum(prediction.values()) - 1.0) < 0.01


def test_tool_executor_runs_up_to_ten_calls_concurrently() -> None:
    event_bus = EventBus()

    class StubRegistry:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def invoke(
            self,
            name: str,
            arguments: dict[str, object],
            constitution: object,
            session_id: str | None = None,
            trace_id: str | None = None,
        ) -> dict[str, object]:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.05)
            self.active -= 1
            return {"name": name, "arguments": arguments}

    async def run() -> tuple[int, list[ToolResult]]:
        registry = StubRegistry()
        executor = ToolExecutor(
            registry=registry, event_bus=event_bus, max_workers=4, queue_size=16
        )
        await executor.start()
        try:
            results = await executor.execute_many(
                [
                    ToolCall(name=f"tool_{index}", arguments={"index": index})
                    for index in range(6)
                ],
                constitution="restricted",
                session_id="session",
            )
        finally:
            await executor.close()
        return registry.max_active, results

    max_active, results = asyncio.run(run())
    assert len(results) == 6
    assert max_active >= 2
    assert all(result.success for result in results)


def test_external_llm_call_tracks_usage(tmp_path: Path, monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {"message": {"content": "hello from the vault"}},
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9,
                },
            }

    class FakeClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout
            self.requests: list[dict[str, object]] = []

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self, url: str, json: dict[str, object], headers: dict[str, str]
        ) -> FakeResponse:
            self.requests.append({"url": url, "json": json, "headers": headers})
            return FakeResponse()

    monkeypatch.setattr("lisa.llm.httpx.AsyncClient", FakeClient)

    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        freellmapi_base_url="https://vault.example",
        freellmapi_api_key="vault-key",
        freellmapi_default_provider="openai",
    )
    client = LLMClient(
        settings,
        persona_bank=PersonaSoftPromptBank.initialize(tokens=3, dims=5, seed=1),
    )

    result = asyncio.run(
        client.call_external_llm(provider="openai", prompt="hi", max_tokens=32)
    )

    assert result.content == "hello from the vault"
    assert result.usage["total_tokens"] == 9


def test_tool_call_parser_extracts_xml_and_json() -> None:
    parser = ToolCallParser()
    parsed = parser.parse("""
        I will handle this in two steps.
        <tool_call>{"name":"search_notepad","arguments":{"query":"secure api"}}</tool_call>
        Then I will write the file:
        {"tool":"file_write","arguments":{"path":"notes.txt","content":"done"}}
        """)

    assert parsed.text.startswith("I will handle this in two steps.")
    assert {call.name for call in parsed.tool_calls} == {"search_notepad", "file_write"}


def test_llm_client_uses_local_backend_when_configured(tmp_path: Path) -> None:
    class StubLocalBackend:
        async def generate(self, request: object) -> BrainGeneration:
            return BrainGeneration(
                text="local result",
                tool_calls=[
                    ToolCall(name="search_notepad", arguments={"query": "secure"})
                ],
                raw_text="local result",
                used_local_model=True,
            )

    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        model_provider="local",
    )
    client = LLMClient(
        settings,
        persona_bank=PersonaSoftPromptBank.initialize(tokens=3, dims=5, seed=1),
        local_backend=StubLocalBackend(),
    )

    result = asyncio.run(
        client.generate_brain(
            system_prompt="You are LISA.",
            user_prompt="test",
            max_tokens=64,
            persona_weights={"architect": 1.0},
        )
    )

    assert result.text == "local result"
    assert result.tool_calls[0].name == "search_notepad"


def test_llm_client_can_race_local_and_external_brains_under_stress(
    tmp_path: Path,
) -> None:
    class StubLocalBackend:
        ready = True

        async def generate(self, request: object) -> BrainGeneration:
            await asyncio.sleep(0.05)
            return BrainGeneration(
                text="local fallback",
                raw_text="local fallback",
                used_local_model=True,
            )

    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        evolution_artifacts_dir=tmp_path / "data" / "evolution_artifacts",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        model_provider="local",
        freellmapi_base_url="https://vault.example",
        freellmapi_api_key="vault-key",
        freellmapi_default_provider="openai",
        hybrid_brain_enabled=True,
        hybrid_brain_stress_threshold=2,
        hybrid_brain_race_window_ms=250,
    )
    client = LLMClient(
        settings,
        persona_bank=PersonaSoftPromptBank.initialize(tokens=3, dims=5, seed=1),
        local_backend=StubLocalBackend(),
    )

    async def fake_chat_completion(**kwargs):
        await asyncio.sleep(0.01)
        return type(
            "FakeExternalResult",
            (),
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "content": '<tool_call>{"name":"search_notepad","arguments":{"query":"stress"}}</tool_call>external win',
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 6,
                    "total_tokens": 10,
                },
            },
        )()

    client._chat_completion = fake_chat_completion  # type: ignore[method-assign]

    result = asyncio.run(
        client.generate_brain(
            system_prompt="Restricted constitution.",
            user_prompt="Need a very fast answer under pressure",
            max_tokens=128,
            persona_weights={"architect": 1.0},
            stress_level=6,
        )
    )

    assert result.used_local_model is False
    assert result.tool_calls and result.tool_calls[0].name == "search_notepad"
    assert "external win" in result.text


def test_runtime_falls_back_when_local_backend_is_unavailable(tmp_path: Path) -> None:
    class StubUnavailableLocalBackend:
        ready = False
        load_error = "gguf load failed"

        async def generate(self, request: object) -> BrainGeneration:
            raise AssertionError(
                "generate should not run when the local backend is not ready"
            )

    class StubWriter:
        async def enqueue(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(1)
            return future

        async def flush_pending(self) -> None:
            return None

    class StubTools:
        def list_tools(self) -> list[object]:
            return []

    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        model_provider="local",
        local_model_path=tmp_path / "models" / "missing.gguf",
    )
    notepad = Notepad(settings.db_path)
    runtime = LisaRuntime(
        settings=settings,
        notepad=notepad,
        llm_client=LLMClient(
            settings,
            persona_bank=PersonaSoftPromptBank.initialize(tokens=3, dims=5, seed=1),
            local_backend=StubUnavailableLocalBackend(),
        ),
        tools=StubTools(),
        tool_executor=StubToolExecutor(),
        event_bus=EventBus(),
        notepad_writer=StubWriter(),
        gating=PersonaGatingNetwork.load_or_initialize(settings.gating_model_path),
        personal_store=None,
    )

    response = asyncio.run(runtime.process_chat(ChatRequest(message="hello lisa")))

    assert "hello lisa" in response.message.lower()
    assert response.constitution == "restricted"
    assert response.used_external_model is False
    assert any(note.startswith("route=") for note in response.notes)


def test_conductor_runs_tool_follow_up_loop() -> None:
    class StubNotepad:
        def get_constitution_state(self) -> dict[str, object]:
            return {
                "mode": "restricted",
                "reason": None,
                "updated_at": "2026-06-11T00:00:00+00:00",
            }

        def search(self, query: str, limit: int = 5) -> list[dict[str, object]]:
            return [
                {
                    "entry_type": "interaction",
                    "payload": {"query": query},
                    "constitution": "restricted",
                }
            ]

    class StubWriter:
        async def flush_pending(self) -> None:
            return None

    class StubGating:
        def predict_blend(self, text: str) -> dict[str, float]:
            return {
                "architect": 0.7,
                "oracle": 0.1,
                "guardian": 0.1,
                "evolution_engine": 0.05,
                "distributed_mind": 0.05,
            }

    class StubTools:
        async def invoke(
            self, name: str, arguments: dict[str, object], constitution: object
        ) -> dict[str, object]:
            return {
                "tool": name,
                "arguments": arguments,
                "constitution": getattr(constitution, "value", constitution),
            }

    class StubRuntime:
        def __init__(self) -> None:
            self.notepad = StubNotepad()
            self.notepad_writer = StubWriter()
            self.gating = StubGating()
            self.tools = StubTools()
            self.calls: list[BrainTask] = []

        async def process_brain_task(self, task: BrainTask) -> ChatResponse:
            self.calls.append(task)
            if len(self.calls) == 1:
                return ChatResponse(
                    session_id=task.inbound.session_id or "session-1",
                    message="Need a tool result.",
                    constitution="restricted",
                    personas=task.persona_weights or {"architect": 1.0},
                    tool_suggestions=["search_notepad"],
                    tool_calls=[
                        ToolCall(name="search_notepad", arguments={"query": "secure"})
                    ],
                    used_external_model=False,
                    notes=[],
                )

            assert task.follow_up is True
            assert task.tool_results
            return ChatResponse(
                session_id=task.inbound.session_id or "session-1",
                message=f"Follow-up with {task.tool_results[0].tool} complete.",
                constitution="restricted",
                personas=task.persona_weights or {"architect": 1.0},
                tool_suggestions=[],
                tool_calls=[],
                used_external_model=False,
                notes=[],
            )

    async def run() -> tuple[StubRuntime, ChatResponse]:
        runtime = StubRuntime()
        conductor = TaskConductor(
            runtime=runtime,
            tool_executor=StubToolExecutor(),
            event_bus=EventBus(),
            queue_size=4,
            max_arms=2,
        )
        await conductor.start()
        try:
            response = await conductor.submit_chat(
                ChatRequest(message="inspect the secure api")
            )
        finally:
            await conductor.close()
        return runtime, response

    runtime, response = asyncio.run(run())

    assert len(runtime.calls) == 2
    assert runtime.calls[0].context_summary[0]["entry_type"] == "interaction"
    assert runtime.calls[0].persona_weights is not None
    assert runtime.calls[1].follow_up is True
    assert runtime.calls[1].tool_results[0].tool == "search_notepad"
    assert response.message == "Follow-up with search_notepad complete."


def test_nightly_evolution_scheduler_registers_skill_after_sandbox_pass(
    tmp_path: Path,
) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        evolution_enabled=True,
        evolution_check_interval_seconds=1,
        evolution_window_start_hour=0,
        evolution_window_duration_hours=24,
        evolution_idle_after_seconds=0,
        enable_browser_tools=True,
        message_hub_enabled=False,
    )
    notepad = Notepad(settings.db_path)
    notepad.log_entry(
        entry_type="task_summary",
        payload={
            "user_input": "Build a retry guard for the nightly workflow",
            "outcome": "error",
            "error": "py_compile failed",
            "self_critique": "Need a reusable retry guard.",
            "response": "The last attempt broke on syntax.",
        },
        constitution=ConstitutionMode.RESTRICTED,
        reward=0.1,
    )

    class FakeWriter:
        def __init__(self) -> None:
            self.records: list[dict[str, object]] = []

        async def enqueue(
            self,
            entry_type: str,
            payload: dict[str, object],
            constitution: object,
            personas: dict[str, float] | None = None,
            critique: str | None = None,
            reward: float | None = None,
        ) -> asyncio.Future[int]:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[int] = loop.create_future()
            self.records.append(
                {
                    "entry_type": entry_type,
                    "payload": payload,
                    "constitution": getattr(constitution, "value", constitution),
                    "personas": personas or {},
                    "reward": reward,
                }
            )
            future.set_result(len(self.records))
            return future

    class FakeConductor:
        def is_idle(self) -> bool:
            return True

    class FakeTools:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []
            self.metrics: list[dict[str, object]] = []
            self._skills: dict[str, str] = {}

        async def invoke(
            self,
            name: str,
            arguments: dict[str, object],
            constitution: object,
        ) -> dict[str, object]:
            self.calls.append((name, arguments))
            if name == "browser_search":
                return {
                    "query": arguments["query"],
                    "results": [
                        {
                            "title": "Python py_compile documentation",
                            "url": "https://docs.python.org/3/library/py_compile.html",
                        }
                    ],
                }
            if name == "browser_fetch":
                return {
                    "url": arguments["url"],
                    "text": "py_compile compiles Python source files into bytecode.",
                }
            if name == "call_external_llm":
                return {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "content": json.dumps(
                        {
                            "name": "retry_guard",
                            "description": "A tiny helper that keeps retry handling explicit.",
                            "code": (
                                "def retry_guard(context):\n"
                                "    return {'status': 'ok', 'context_type': type(context).__name__}\n"
                            ),
                            "test_command": "python -m py_compile {path}",
                            "notes": ["Generated from recurring syntax failures."],
                            "smoke_test_arguments": {},
                        }
                    ),
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                }
            if name == "file_write":
                path = Path(str(arguments["path"]))
                content = str(arguments["content"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                return {
                    "path": str(path),
                    "bytes_written": len(content.encode("utf-8")),
                }
            if name == "terminal_exec":
                command = str(arguments["command"])
                assert "py_compile" in command
                return {"returncode": 0, "stdout": "compiled", "stderr": ""}
            if name == "add_skill":
                skill_name = str(arguments["name"])
                path = settings.skills_dir / f"{skill_name}.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                code = str(arguments["code"])
                path.write_text(code, encoding="utf-8")
                self._skills[skill_name] = code
                return {"path": str(path), "status": "saved"}
            if name == "dashboard_update":
                self.metrics.append(dict(arguments))
                return {
                    "status": "ok",
                    "metric": arguments["metric"],
                    "value": arguments["value"],
                }
            if name in self._skills:
                return {"status": "ok", "skill": name}
            raise AssertionError(f"Unexpected tool: {name}")

    async def run() -> (
        tuple[NightlyEvolutionScheduler, FakeTools, FakeWriter, EvolutionCycleResult]
    ):
        event_bus = EventBus()
        runtime = SimpleNamespace(
            settings=settings,
            notepad=notepad,
            notepad_writer=FakeWriter(),
            tools=FakeTools(),
        )
        scheduler = NightlyEvolutionScheduler(
            runtime=runtime,
            conductor=FakeConductor(),
            event_bus=event_bus,
        )
        result = await scheduler.run_once(force=True)
        return scheduler, runtime.tools, runtime.notepad_writer, result

    scheduler, tools, writer, result = asyncio.run(run())

    assert result.status == "registered", result.details
    assert result.registered is True
    assert result.test_passed is True
    assert result.skill_name is not None
    assert any(name == "call_external_llm" for name, _ in tools.calls)
    assert any(name == "add_skill" for name, _ in tools.calls)
    assert (settings.skills_dir / f"{result.skill_name}.py").exists()
    assert writer.records and writer.records[-1]["entry_type"] == "evolution_cycle"
    assert any(metric["metric"] == "evolution_last_skill" for metric in tools.metrics)
    assert any(metric["metric"] == "evolution_status" for metric in tools.metrics)
