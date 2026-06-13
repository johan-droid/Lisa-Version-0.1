from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from cryptography.fernet import Fernet

from tools.executor import EncryptedKeyVault, ProviderConfig, ToolExecutor


def test_executor_file_ops_dashboard_and_skill_registration(tmp_path: Path) -> None:
    executor = ToolExecutor(
        workspace_root=tmp_path,
        master_key=Fernet.generate_key(),
        keys_path=tmp_path / "keys.enc",
        skills_manifest_path=tmp_path / "skills_manifest.json",
        mcp_config_path=tmp_path / "mcp_servers.json",
        provider_configs={},
    )

    async def run() -> None:
        write_result = await executor.file_write("notes/demo.txt", "alpha beta gamma")
        assert write_result.success is True
        assert (tmp_path / "notes" / "demo.txt").exists()

        read_result = await executor.file_read("notes/demo.txt")
        assert read_result.data["content"] == "alpha beta gamma"

        edit_result = await executor.file_edit("notes/demo.txt", "beta", "delta")
        assert edit_result.success is True
        assert "delta" in (tmp_path / "notes" / "demo.txt").read_text(encoding="utf-8")

        metric_result = await executor.dashboard_update(metric="active_tasks", value=3)
        assert metric_result.success is True
        assert await executor.dashboard_bus.latest("active_tasks") == 3

        code = """
from tools.executor import ToolResult

async def skill(**kwargs):
    return ToolResult(success=True, data={"echo": kwargs})
"""
        skill_result = await executor.add_skill("Echo Skill", code)
        assert skill_result.success is True
        invoked = await executor.invoke("echo_skill", value=7)
        assert invoked.success is True
        assert invoked.data["echo"]["value"] == 7

    asyncio.run(run())
    manifest = json.loads(
        (tmp_path / "skills_manifest.json").read_text(encoding="utf-8")
    )
    assert "echo_skill" in manifest["skills"]


def test_encrypted_vault_round_trip(tmp_path: Path) -> None:
    master_key = Fernet.generate_key()
    vault_path = tmp_path / "keys.enc"
    vault = EncryptedKeyVault.load_or_create(vault_path, master_key)
    vault.set_provider(
        "openai",
        ProviderConfig(
            base_url="https://example.invalid",
            api_key="secret",
            model="gpt-4o-mini",
        ),
    )

    reloaded = EncryptedKeyVault.load_or_create(vault_path, master_key)
    provider = reloaded.get_provider("openai")

    assert provider is not None
    assert provider.api_key == "secret"
    assert vault_path.exists()


def test_browser_fetch_caches_and_parses(tmp_path: Path, monkeypatch) -> None:
    executor = ToolExecutor(
        workspace_root=tmp_path,
        master_key=Fernet.generate_key(),
        keys_path=tmp_path / "keys.enc",
        skills_manifest_path=tmp_path / "skills_manifest.json",
        mcp_config_path=tmp_path / "mcp_servers.json",
        provider_configs={},
    )

    calls = {"count": 0}

    class FakeResponse:
        def __init__(self, url: str, text: str):
            self.url = url
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.headers = kwargs.get("headers", {})

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, **kwargs) -> FakeResponse:
            calls["count"] += 1
            if url.endswith("/robots.txt"):
                return FakeResponse(url, "User-agent: *\nAllow: /")
            if "lite.duckduckgo.com" in url:
                return FakeResponse(
                    url,
                    """
                    <html><body>
                      <table>
                        <tr><td><a href="https://example.com/r1">Result One</a></td></tr>
                        <tr><td><a href="https://example.com/r2">Result Two</a></td></tr>
                      </table>
                    </body></html>
                    """,
                )
            return FakeResponse(
                url,
                """
                <html>
                  <head><title>Demo Page</title></head>
                  <body>
                    <main>
                      <p>Hello world</p>
                      <pre><code>print("hi")</code></pre>
                      <a href="https://example.com/next">Next</a>
                    </main>
                  </body>
                </html>
                """,
            )

    monkeypatch.setattr("tools.executor.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(
        executor, "_respect_robots", lambda parsed_url: asyncio.sleep(0)
    )
    monkeypatch.setattr(
        executor,
        "_throttle_domain",
        lambda domain, minimum_delay=0.35: asyncio.sleep(0),
    )

    async def run() -> None:
        first = await executor.browser_fetch("https://example.com/page")
        second = await executor.browser_fetch("https://example.com/page")
        search = await executor.browser_search("python async tools")

        assert first.success is True
        assert first.data["title"] == "Demo Page"
        assert "Hello world" in first.data["text"]
        assert 'print("hi")' in first.data["code_blocks"][0]
        assert second.data["title"] == "Demo Page"
        assert calls["count"] >= 1
        assert search.success is True
        assert search.data["results"][0]["title"] == "Result One"

    asyncio.run(run())


def test_terminal_exec_uses_docker_sandbox(monkeypatch, tmp_path: Path) -> None:
    executor = ToolExecutor(
        workspace_root=tmp_path,
        master_key=Fernet.generate_key(),
        keys_path=tmp_path / "keys.enc",
        skills_manifest_path=tmp_path / "skills_manifest.json",
        mcp_config_path=tmp_path / "mcp_servers.json",
        provider_configs={},
        docker_image="python:3.11-slim",
    )

    captured: dict[str, object] = {}

    class FakeContainer:
        def __init__(self) -> None:
            self.killed = False

        def wait(self, timeout: int) -> dict[str, int]:
            captured["timeout"] = timeout
            return {"StatusCode": 0}

        def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
            return b"hello\x00 world\x1b[31m!"

        def kill(self) -> None:
            self.killed = True

        def remove(self, force: bool = False) -> None:
            captured["removed"] = True

    class FakeContainerManager:
        def run(self, image: str, **kwargs):
            captured["image"] = image
            captured["kwargs"] = kwargs
            return FakeContainer()

    class FakeDockerClient:
        def __init__(self) -> None:
            self.containers = FakeContainerManager()

    monkeypatch.setattr("tools.executor.docker.from_env", lambda: FakeDockerClient())

    async def run() -> None:
        result = await executor.terminal_exec("python -c 'print(1)'", timeout=12)
        assert result.success is True
        assert "hello world" in result.data["stdout"]
        assert "\x1b" not in result.data["stdout"]
        assert captured["image"] == "python:3.11-slim"
        assert captured["kwargs"]["mem_limit"] == "100m"
        assert captured["kwargs"]["read_only"] is True
        assert captured["timeout"] == 12

    asyncio.run(run())


def test_mcp_call_uses_json_rpc(monkeypatch, tmp_path: Path) -> None:
    executor = ToolExecutor(
        workspace_root=tmp_path,
        master_key=Fernet.generate_key(),
        keys_path=tmp_path / "keys.enc",
        skills_manifest_path=tmp_path / "skills_manifest.json",
        mcp_config_path=tmp_path / "mcp_servers.json",
        provider_configs={},
    )

    executor._load_mcp_configs = lambda: {
        "filesystem": SimpleNamespace(
            name="filesystem",
            command=["python", "-c", "print('mcp')"],
            methods=["filesystem.read"],
            env={},
            cwd=None,
        )
    }

    class FakeStdin:
        def __init__(self, process: "FakeProcess") -> None:
            self.process = process
            self.buffer = []

        def write(self, data: bytes) -> None:
            self.buffer.append(data)
            request = json.loads(data.decode("utf-8").strip())
            response = (
                json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}
                )
                + "\n"
            )
            self.process.stdout.feed(response.encode("utf-8"))

        async def drain(self) -> None:
            return None

    class FakeStdout:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[bytes] = asyncio.Queue()

        def feed(self, payload: bytes) -> None:
            self.queue.put_nowait(payload)

        async def readline(self) -> bytes:
            return await self.queue.get()

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakeStdout()
            self.stdin = FakeStdin(self)
            self.stderr = FakeStdout()
            self.returncode = None

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(
        "tools.executor.asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )

    async def run() -> None:
        result = await executor.mcp_call(
            "filesystem", "filesystem.read", {"path": "notes.txt"}
        )
        assert result.success is True
        assert result.data["ok"] is True

    asyncio.run(run())


def test_executor_loads_preexisting_skills_from_manifest(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skills_dir / "echo_existing.py"
    skill_path.write_text(
        """
from tools.executor import ToolResult

async def skill(**kwargs):
    return ToolResult(success=True, data={"echo": kwargs})
""".strip(),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "skills_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "skills": {
                    "echo_existing": {
                        "name": "Echo Existing",
                        "path": str(skill_path),
                        "function": "skill",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    executor = ToolExecutor(
        workspace_root=tmp_path,
        master_key=Fernet.generate_key(),
        keys_path=tmp_path / "keys.enc",
        skills_dir=skills_dir,
        skills_manifest_path=manifest_path,
        mcp_config_path=tmp_path / "mcp_servers.json",
        provider_configs={},
    )

    async def run() -> None:
        result = await executor.invoke("echo_existing", value=42)
        assert result.success is True
        assert result.data["echo"]["value"] == 42

    asyncio.run(run())
