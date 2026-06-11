from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import inspect
import json
import os
import re
import time
import urllib.parse
import urllib.robotparser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import docker
import httpx
from bs4 import BeautifulSoup
from cachetools import LRUCache
from cryptography.fernet import Fernet, InvalidToken


ToolCallable = Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class ToolResult:
    success: bool
    data: Any = None
    error: str | None = None
    tool: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tool": self.tool,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ProviderConfig:
    base_url: str
    api_key: str
    model: str
    kind: str = "openai"
    headers: dict[str, str] = field(default_factory=dict)


class MetricEventBus:
    """Simple metric bus backed by asyncio.Event objects keyed by metric name."""

    def __init__(self) -> None:
        self._events: dict[str, list[asyncio.Event]] = {}
        self._latest: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def publish(self, metric: str, value: Any) -> None:
        async with self._lock:
            self._latest[metric] = value
            subscribers = list(self._events.get(metric, []))
        for event in subscribers:
            event.set()

    async def subscribe(self, metric: str) -> asyncio.Event:
        event = asyncio.Event()
        async with self._lock:
            self._events.setdefault(metric, []).append(event)
        return event

    async def wait_for(self, metric: str, timeout: float | None = None) -> Any:
        event = await self.subscribe(metric)
        await asyncio.wait_for(event.wait(), timeout=timeout)
        async with self._lock:
            return self._latest.get(metric)

    async def latest(self, metric: str) -> Any:
        async with self._lock:
            return self._latest.get(metric)


class EncryptedKeyVault:
    """Encrypted provider key vault backed by a Fernet blob on disk."""

    def __init__(self, path: Path, master_key: str | bytes):
        self.path = Path(path)
        self.fernet = Fernet(self._normalize_key(master_key))
        self._payload: dict[str, Any] = {"providers": {}}

    @staticmethod
    def _normalize_key(master_key: str | bytes) -> bytes:
        if isinstance(master_key, bytes):
            return master_key
        key = master_key.encode("utf-8")
        if len(key) == 44:
            return key
        raise ValueError(
            "Fernet master keys must be urlsafe base64 encoded 32-byte keys. "
            "Provide a 44-character key string."
        )

    @classmethod
    def load_or_create(cls, path: Path, master_key: str | bytes) -> "EncryptedKeyVault":
        vault = cls(path=path, master_key=master_key)
        if path.exists():
            vault.load()
        else:
            vault.save()
        return vault

    def load(self) -> None:
        if not self.path.exists():
            self._payload = {"providers": {}}
            return
        encrypted = self.path.read_bytes()
        if not encrypted:
            self._payload = {"providers": {}}
            return
        try:
            decrypted = self.fernet.decrypt(encrypted)
        except InvalidToken as exc:
            raise RuntimeError("Unable to decrypt keys.enc with the supplied master key.") from exc
        self._payload = json.loads(decrypted.decode("utf-8"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        plaintext = json.dumps(self._payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.path.write_bytes(self.fernet.encrypt(plaintext))

    def set_provider(self, name: str, config: ProviderConfig) -> None:
        providers = self._payload.setdefault("providers", {})
        providers[name] = {
            "base_url": config.base_url,
            "api_key": config.api_key,
            "model": config.model,
            "kind": config.kind,
            "headers": config.headers,
        }
        self.save()

    def get_provider(self, name: str) -> ProviderConfig | None:
        providers = self._payload.get("providers", {})
        raw = providers.get(name)
        if not isinstance(raw, dict):
            return None
        return ProviderConfig(
            base_url=str(raw.get("base_url") or ""),
            api_key=str(raw.get("api_key") or ""),
            model=str(raw.get("model") or ""),
            kind=str(raw.get("kind") or "openai"),
            headers=dict(raw.get("headers") or {}),
        )

    def list_providers(self) -> list[str]:
        providers = self._payload.get("providers", {})
        return sorted(providers.keys())


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    command: list[str]
    methods: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass(slots=True)
class _MCPProcess:
    config: MCPServerConfig
    process: asyncio.subprocess.Process
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    request_id: int = 0

    def next_request_id(self) -> int:
        self.request_id += 1
        return self.request_id


class ToolExecutor:
    def __init__(
        self,
        workspace_root: Path,
        *,
        skills_dir: Path | None = None,
        skills_manifest_path: Path | None = None,
        mcp_config_path: Path | None = None,
        keys_path: Path | None = None,
        master_key: str | bytes | None = None,
        provider_configs: dict[str, ProviderConfig] | None = None,
        dashboard_bus: MetricEventBus | None = None,
        max_concurrent: int = 10,
        timeout_seconds: int = 30,
        browser_cache_size: int = 100,
        docker_image: str = "python:3.11-slim",
        user_agent: str = "LISA/0.1",
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.skills_dir = (skills_dir or self.workspace_root / "skills").resolve()
        self.skills_manifest_path = (skills_manifest_path or self.workspace_root / "skills_manifest.json").resolve()
        self.mcp_config_path = (mcp_config_path or self.workspace_root / "mcp_servers.json").resolve()
        self.keys_path = (keys_path or self.workspace_root / "keys.enc").resolve()
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self.user_agent = user_agent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tools: dict[str, ToolCallable] = {}
        self._browser_cache: LRUCache[str, dict[str, Any]] = LRUCache(maxsize=browser_cache_size)
        self._domain_last_fetch: dict[str, float] = {}
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.dashboard_bus = dashboard_bus or MetricEventBus()
        self.provider_configs = provider_configs or {}
        self.vault: EncryptedKeyVault | None = None
        self._mcp_servers: dict[str, _MCPProcess] = {}
        self._skills_manifest: dict[str, Any] = {"skills": {}}
        self._register_default_tools()
        self._load_or_initialize_files(master_key)

    @property
    def available_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def register_tool(self, name: str, func: ToolCallable) -> None:
        self._tools[name] = func

    def register_provider(self, name: str, config: ProviderConfig) -> None:
        self.provider_configs[name] = config
        if self.vault is not None:
            self.vault.set_provider(name, config)

    def list_mcp_servers(self) -> dict[str, list[str]]:
        return {name: config.methods for name, config in self._load_mcp_configs().items()}

    def _load_or_initialize_files(self, master_key: str | bytes | None) -> None:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        if not self.skills_manifest_path.exists():
            self.skills_manifest_path.write_text(json.dumps(self._skills_manifest, indent=2), encoding="utf-8")
        else:
            try:
                self._skills_manifest = json.loads(self.skills_manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._skills_manifest = {"skills": {}}
        if not self.mcp_config_path.exists():
            self.mcp_config_path.write_text(
                json.dumps(
                    {
                        "servers": {
                            "filesystem": {
                                "command": [],
                                "methods": ["filesystem.read", "filesystem.write", "filesystem.edit"],
                                "description": "Filesystem MCP placeholder configuration.",
                            },
                            "github": {
                                "command": [],
                                "methods": ["github.search", "github.issue", "github.pull_request"],
                                "description": "GitHub MCP placeholder configuration.",
                            },
                        }
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if master_key is not None:
            self.vault = EncryptedKeyVault.load_or_create(self.keys_path, master_key)
            for provider in self.vault.list_providers():
                config = self.vault.get_provider(provider)
                if config is not None:
                    self.provider_configs[provider] = config
        elif self.keys_path.exists():
            # A key file exists, but without the master key we can still operate
            # with explicit provider configs injected at runtime.
            self.vault = None

        self._load_preexisting_skills()

    def _register_default_tools(self) -> None:
        self.register_tool("browser_fetch", self.browser_fetch)
        self.register_tool("browser_search", self.browser_search)
        self.register_tool("terminal_exec", self.terminal_exec)
        self.register_tool("mcp_call", self.mcp_call)
        self.register_tool("file_read", self.file_read)
        self.register_tool("file_write", self.file_write)
        self.register_tool("file_edit", self.file_edit)
        self.register_tool("call_external_llm", self.call_external_llm)
        self.register_tool("add_skill", self.add_skill)
        self.register_tool("dashboard_update", self.dashboard_update)

    async def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}", tool=name)
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(tool(**kwargs), timeout=self.timeout_seconds)
                return self._coerce_result(name, result)
            except Exception as exc:
                return ToolResult(success=False, error=str(exc), tool=name)

    async def execute_many(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        tasks = [asyncio.create_task(self.invoke(name, **arguments)) for name, arguments in calls]
        return list(await asyncio.gather(*tasks))

    async def browser_fetch(self, url: str, extract_text: bool = True) -> ToolResult:
        try:
            parsed = urllib.parse.urlparse(url)
            await self._respect_robots(parsed)
            await self._throttle_domain(parsed.netloc)
            if url in self._browser_cache:
                return ToolResult(success=True, data=self._browser_cache[url], tool="browser_fetch")

            headers = {"User-Agent": self.user_agent}
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True, headers=headers) as client:
                response = await client.get(url)
                response.raise_for_status()

            data = self._parse_html_page(str(response.url), response.text, extract_text=extract_text)
            self._browser_cache[url] = data
            return ToolResult(success=True, data=data, tool="browser_fetch")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="browser_fetch")

    async def browser_search(self, query: str) -> ToolResult:
        try:
            if not query.strip():
                raise ValueError("query must not be empty")
            url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote_plus(query)}"
            headers = {"User-Agent": self.user_agent}
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True, headers=headers) as client:
                response = await client.get(url)
                response.raise_for_status()
            results = self._parse_duckduckgo_lite(response.text)
            return ToolResult(success=True, data={"query": query, "results": results}, tool="browser_search")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="browser_search")

    async def terminal_exec(self, command: str, timeout: int = 30) -> ToolResult:
        try:
            result = await asyncio.to_thread(self._terminal_exec_sync, command, timeout)
            return ToolResult(success=True, data=result, tool="terminal_exec")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="terminal_exec")

    async def mcp_call(
        self,
        server: str,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> ToolResult:
        try:
            data = await self._mcp_call(server, method, params or {}, timeout or self.timeout_seconds)
            return ToolResult(success=True, data=data, tool="mcp_call")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="mcp_call")

    async def file_read(self, path: str) -> ToolResult:
        try:
            resolved = self._resolve_workspace_path(path)
            async with aiofiles.open(resolved, "r", encoding="utf-8") as handle:
                content = await handle.read()
            return ToolResult(success=True, data={"path": str(resolved), "content": content}, tool="file_read")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="file_read")

    async def file_write(self, path: str, content: str) -> ToolResult:
        try:
            resolved = self._resolve_workspace_path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(resolved, "w", encoding="utf-8") as handle:
                await handle.write(content)
            return ToolResult(
                success=True,
                data={"path": str(resolved), "bytes_written": len(content.encode("utf-8"))},
                tool="file_write",
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="file_write")

    async def file_edit(self, path: str, find: str, replace: str) -> ToolResult:
        try:
            resolved = self._resolve_workspace_path(path)
            async with aiofiles.open(resolved, "r", encoding="utf-8") as handle:
                content = await handle.read()
            if not find:
                raise ValueError("find must not be empty")
            occurrences = content.count(find)
            if occurrences == 0:
                raise ValueError("target text not found")
            updated = content.replace(find, replace)
            async with aiofiles.open(resolved, "w", encoding="utf-8") as handle:
                await handle.write(updated)
            return ToolResult(
                success=True,
                data={"path": str(resolved), "replacements": occurrences},
                tool="file_edit",
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="file_edit")

    async def call_external_llm(
        self,
        provider: str,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
    ) -> ToolResult:
        try:
            config = self.provider_configs.get(provider) or (self.vault.get_provider(provider) if self.vault else None)
            if config is None:
                raise KeyError(f"No provider config is available for '{provider}'.")
            response = await self._call_provider(config, prompt, max_tokens=max_tokens, temperature=temperature)
            total_tokens = int(response.get("usage", {}).get("total_tokens") or self._estimate_tokens(prompt, response["content"]))
            await self.dashboard_update(metric=f"llm_tokens_{provider}", value=str(total_tokens))
            return ToolResult(
                success=True,
                data={
                    "provider": provider,
                    "model": config.model,
                    "content": response["content"],
                    "usage": response.get("usage", {"total_tokens": total_tokens}),
                },
                tool="call_external_llm",
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="call_external_llm")

    async def add_skill(self, name: str, code: str) -> ToolResult:
        try:
            if not name.strip():
                raise ValueError("skill name is required")
            compile(code, f"<skill:{name}>", "exec")
            slug = self._slugify(name)
            path = self.skills_dir / f"{slug}.py"
            async with aiofiles.open(path, "w", encoding="utf-8") as handle:
                await handle.write(code)
            module = self._import_skill_module(slug, path)
            skill_callable = getattr(module, "skill", None)
            if skill_callable is None or not inspect.iscoroutinefunction(skill_callable):
                raise ValueError("skill modules must define async def skill(**kwargs) -> ToolResult")
            self.register_tool(slug, self._build_skill_wrapper(slug, skill_callable))
            self._skills_manifest.setdefault("skills", {})[slug] = {
                "name": name,
                "path": str(path),
                "function": "skill",
            }
            async with aiofiles.open(self.skills_manifest_path, "w", encoding="utf-8") as handle:
                await handle.write(json.dumps(self._skills_manifest, indent=2, ensure_ascii=False))
            return ToolResult(success=True, data={"path": str(path), "status": "registered"}, tool="add_skill")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="add_skill")

    async def dashboard_update(self, metric: str, value: Any) -> ToolResult:
        try:
            await self.dashboard_bus.publish(metric, value)
            return ToolResult(success=True, data={"metric": metric, "value": value}, tool="dashboard_update")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool="dashboard_update")

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        candidate.relative_to(self.workspace_root)
        return candidate

    async def _respect_robots(self, parsed_url: urllib.parse.ParseResult) -> None:
        if parsed_url.scheme not in {"http", "https"}:
            return
        domain = parsed_url.netloc
        parser = self._robots_cache.get(domain)
        if parser is None:
            parser = urllib.robotparser.RobotFileParser()
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            parser.set_url(robots_url)
            try:
                async with httpx.AsyncClient(timeout=min(5, self.timeout_seconds), follow_redirects=True) as client:
                    response = await client.get(robots_url, headers={"User-Agent": self.user_agent})
                if response.status_code < 400:
                    parser.parse(response.text.splitlines())
                else:
                    parser.parse([])
            except Exception:
                # Best-effort fallback: if robots cannot be fetched, keep the
                # parser permissive rather than breaking the tool outright.
                parser.parse([])
            self._robots_cache[domain] = parser

        if not parser.can_fetch(self.user_agent, parsed_url.geturl()):
            raise PermissionError(f"Robots.txt disallows fetching {parsed_url.geturl()}")

    async def _throttle_domain(self, domain: str, minimum_delay: float = 0.35) -> None:
        lock = self._domain_locks.setdefault(domain, asyncio.Lock())
        async with lock:
            last_fetch = self._domain_last_fetch.get(domain, 0.0)
            elapsed = time.monotonic() - last_fetch
            if elapsed < minimum_delay:
                await asyncio.sleep(minimum_delay - elapsed)
            self._domain_last_fetch[domain] = time.monotonic()

    @staticmethod
    def _sanitize_output(text: str) -> str:
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    def _terminal_exec_sync(self, command: str, timeout: int) -> dict[str, Any]:
        client = docker.from_env()
        container = None
        try:
            container = client.containers.run(
                self.docker_image,
                command=["sh", "-lc", command],
                detach=True,
                stdin_open=False,
                tty=False,
                network_disabled=True,
                mem_limit="100m",
                read_only=True,
                working_dir="/workspace",
                volumes={str(self.workspace_root): {"bind": "/workspace", "mode": "rw"}},
                environment={"HOME": "/tmp"},
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                pids_limit=128,
            )
            wait_result = container.wait(timeout=timeout)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            stdout = self._sanitize_output(logs)
            exit_code = int(wait_result.get("StatusCode", 1) if isinstance(wait_result, dict) else 0)
            return {"returncode": exit_code, "stdout": stdout, "stderr": ""}
        except Exception:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.kill()
            raise
        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.remove(force=True)

    async def _mcp_call(
        self,
        server: str,
        method: str,
        params: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        config = self._load_mcp_configs().get(server)
        if config is None:
            raise KeyError(f"Unknown MCP server: {server}")
        if not config.command:
            raise RuntimeError(f"MCP server '{server}' is configured but has no command.")

        process = await self._ensure_mcp_process(config)
        async with process.lock:
            request_id = process.next_request_id()
            request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            if process.process.stdin is None or process.process.stdout is None:
                raise RuntimeError("MCP stdio pipes are unavailable.")
            process.process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            await asyncio.wait_for(process.process.stdin.drain(), timeout=timeout)

            while True:
                raw = await asyncio.wait_for(process.process.stdout.readline(), timeout=timeout)
                if not raw:
                    raise RuntimeError(f"MCP server '{server}' closed the pipe unexpectedly.")
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                message = json.loads(line)
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise RuntimeError(f"MCP error: {message['error']}")
                result = message.get("result")
                return result if isinstance(result, dict) else {"value": result}

    async def _ensure_mcp_process(self, config: MCPServerConfig) -> _MCPProcess:
        existing = self._mcp_servers.get(config.name)
        if existing is not None and existing.process.returncode is None:
            return existing

        env = os.environ.copy()
        env.update(config.env)
        process = await asyncio.create_subprocess_exec(
            *config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=config.cwd or str(self.workspace_root),
            env=env,
        )
        handle = _MCPProcess(config=config, process=process)
        self._mcp_servers[config.name] = handle
        return handle

    def _load_mcp_configs(self) -> dict[str, MCPServerConfig]:
        payload = json.loads(self.mcp_config_path.read_text(encoding="utf-8"))
        servers = payload.get("servers", {})
        configs: dict[str, MCPServerConfig] = {}
        for name, raw in servers.items():
            if not isinstance(raw, dict):
                continue
            command = raw.get("command") or []
            configs[name] = MCPServerConfig(
                name=name,
                command=[str(item) for item in command],
                methods=[str(item) for item in raw.get("methods", [])],
                env={str(k): str(v) for k, v in dict(raw.get("env", {})).items()},
                cwd=str(raw["cwd"]) if raw.get("cwd") else None,
            )
        return configs

    async def _call_provider(
        self,
        config: ProviderConfig,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {config.api_key}", **config.headers}
        if config.kind == "anthropic":
            url = f"{config.base_url.rstrip('/')}/v1/messages"
            payload = {
                "model": config.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers.setdefault("anthropic-version", "2023-06-01")
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            data = response.json()
            content = "".join(part.get("text", "") for part in data.get("content", []) if isinstance(part, dict))
            usage = data.get("usage") or {}
            return {"content": content, "usage": usage}

        if config.kind == "google":
            url = (
                f"{config.base_url.rstrip('/')}/v1beta/models/"
                f"{urllib.parse.quote(config.model, safe='')}:generateContent?key={config.api_key}"
            )
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
            }
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            text = ""
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
            usage = data.get("usageMetadata") or {}
            return {"content": text, "usage": usage}

        url = f"{config.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Provider response did not contain any choices.")
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        usage = data.get("usage") or {}
        return {"content": content, "usage": usage}

    @staticmethod
    def _estimate_tokens(prompt: str, response: str) -> int:
        return max(1, len(prompt.split()) + len(response.split()))

    @staticmethod
    def _parse_html_page(url: str, html: str, *, extract_text: bool = True) -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        code_blocks = [block.get_text("\n", strip=True) for block in soup.find_all(["pre", "code"])]
        links = []
        for anchor in soup.find_all("a", href=True):
            text = anchor.get_text(" ", strip=True)
            href = anchor["href"]
            if text or href:
                links.append({"text": text, "href": href})
        main_text = ""
        if extract_text:
            for node in soup(["script", "style", "noscript"]):
                node.decompose()
            main_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        return {
            "url": url,
            "title": title,
            "text": main_text,
            "code_blocks": code_blocks,
            "links": links[:100],
        }

    @staticmethod
    def _parse_duckduckgo_lite(html: str) -> list[dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict[str, str]] = []
        seen: set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            title = anchor.get_text(" ", strip=True)
            if not title or href in seen:
                continue
            if href.startswith("/lite/") or href.startswith("javascript:"):
                continue
            container = anchor.find_parent(["tr", "td", "div"]) or anchor.parent
            snippet = ""
            if container is not None:
                snippet = re.sub(r"\s+", " ", container.get_text(" ", strip=True)).strip()
                snippet = snippet.replace(title, "", 1).strip()
            results.append({"title": title, "url": href, "snippet": snippet})
            seen.add(href)
            if len(results) >= 10:
                break

        return results

    @staticmethod
    def _coerce_result(tool: str, result: ToolResult | dict[str, Any] | Any) -> ToolResult:
        if isinstance(result, ToolResult):
            if result.tool is None:
                result.tool = tool
            return result
        if isinstance(result, dict):
            if {"success", "data", "error"}.intersection(result.keys()):
                return ToolResult(
                    success=bool(result.get("success")),
                    data=result.get("data"),
                    error=result.get("error"),
                    tool=result.get("tool", tool),
                    metadata=dict(result.get("metadata") or {}),
                )
            return ToolResult(success=True, data=result, tool=tool)
        return ToolResult(success=True, data=result, tool=tool)

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
        return slug or "skill"

    def _import_skill_module(self, slug: str, path: Path):
        module_name = f"lisa_skill_{slug}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import skill module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def _load_preexisting_skills(self) -> None:
        skills = self._skills_manifest.get("skills", {})
        if not isinstance(skills, dict):
            return

        for slug, raw in skills.items():
            if not isinstance(raw, dict):
                continue
            path_value = raw.get("path")
            if not path_value:
                continue
            path = Path(str(path_value))
            if not path.is_absolute():
                path = (self.workspace_root / path).resolve()
            if not path.exists():
                continue
            try:
                module = self._import_skill_module(str(slug), path)
            except Exception:
                continue
            skill_callable = getattr(module, "skill", None)
            if skill_callable is None or not inspect.iscoroutinefunction(skill_callable):
                continue
            self.register_tool(str(slug), self._build_skill_wrapper(str(slug), skill_callable))

    def _build_skill_wrapper(self, slug: str, skill_callable: Callable[..., Awaitable[Any]]) -> ToolCallable:
        async def _wrapped(**kwargs: Any) -> ToolResult:
            result = await skill_callable(**kwargs)
            return self._coerce_result(slug, result)

        return _wrapped
