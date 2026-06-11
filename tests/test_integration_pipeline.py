from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from lisa.api import create_app
from lisa.config import Settings
from lisa.notepad import Notepad
from lisa.tools import ToolRegistry


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        message_hub_enabled=False,
        evolution_enabled=False,
        model_provider="openai",
        model_name="gpt-4o-mini",
        model_base_url="https://example.invalid",
        model_api_key="test-key",
    )


def test_full_chat_pipeline_with_mocked_external_model(tmp_path: Path, monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, content: str) -> None:
            self._content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": self._content}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
            }

    class FakeAsyncClient:
        calls = 0

        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self,
            url: str,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> FakeResponse:
            FakeAsyncClient.calls += 1
            if FakeAsyncClient.calls == 1:
                content = json_module.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "file_write",
                                "arguments": {
                                    "path": "artifacts/hello_lisa.txt",
                                    "content": "Hello from LISA's tool executor.",
                                },
                            }
                        ]
                    }
                )
            else:
                content = "Hello! I wrote the file and completed the follow-up."
            return FakeResponse(content)

    json_module = json
    monkeypatch.setattr("lisa.llm.httpx.AsyncClient", FakeAsyncClient)

    settings = build_settings(tmp_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Hello, LISA?"})

        assert response.status_code == 200
        body = response.json()
        assert body["used_external_model"] is True
        assert body["message"] == "Hello! I wrote the file and completed the follow-up."
        assert body["tool_calls"] == []

        created_file = tmp_path / "artifacts" / "hello_lisa.txt"
        assert created_file.exists()
        assert created_file.read_text(encoding="utf-8") == "Hello from LISA's tool executor."

        tool_results = client.get("/notepad/search", params={"q": "file_write"})
        assert tool_results.status_code == 200
        assert any(row["entry_type"] == "tool_call" for row in tool_results.json()["results"])

        summaries = client.get("/notepad/search", params={"q": "task_summary"})
        assert summaries.status_code == 200
        assert any(row["payload"]["outcome"] == "success" for row in summaries.json()["results"])


def test_terminal_exec_uses_sandboxed_docker_flags(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        message_hub_enabled=False,
        evolution_enabled=False,
    )

    class FakeWriter:
        async def enqueue(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(1)
            return future

    class FakeEventBus:
        async def publish(self, event):
            return None

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.killed = False

        async def communicate(self):
            return b"ok", b""

        def kill(self) -> None:
            self.killed = True

    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("lisa.tools.shutil.which", lambda name: "docker")
    monkeypatch.setattr("lisa.tools.asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    registry = ToolRegistry(
        settings=settings,
        notepad=object(),  # not used by terminal_exec
        llm_client=object(),  # not used by terminal_exec
        event_bus=FakeEventBus(),
        notepad_writer=FakeWriter(),
    )

    async def run() -> dict[str, object]:
        return await registry.invoke(
            "terminal_exec",
            {"command": "python -c \"print('hello')\""},
            constitution="restricted",
        )

    result = asyncio.run(run())

    args = captured["args"]
    assert args[0] == "docker"
    assert "--network" in args and "none" in args
    assert "--memory" in args and "512m" in args
    assert "--cpus" in args and "1" in args
    assert result["returncode"] == 0
    assert "ok" in result["stdout"]


def test_notepad_search_falls_back_when_fts_query_has_punctuation(tmp_path: Path) -> None:
    notepad = Notepad(tmp_path / "data" / "test.db")
    notepad.log_entry(
        entry_type="interaction",
        payload={"user_message": "Hello, LISA?", "assistant_message": "Hello back."},
        constitution="restricted",
    )

    results = notepad.search("Hello, LISA?")

    assert results
    assert any(row["entry_type"] == "interaction" for row in results)
