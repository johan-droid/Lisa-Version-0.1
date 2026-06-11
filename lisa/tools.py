from __future__ import annotations

import asyncio
import json
import importlib.util
import inspect
import os
import re
import shlex
import shutil
import textwrap
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from lisa.config import Settings
from lisa.constitutions import ConstitutionMode
from lisa.events import EventBus, LisaEvent
from lisa.llm import LLMClient
from lisa.notepad import AsyncNotepadWriter, Notepad


ToolHandler = Callable[[dict[str, Any], "ToolContext"], Awaitable[Any]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    restricted_safe: bool
    handler: ToolHandler


@dataclass(slots=True)
class ToolContext:
    settings: Settings
    notepad: Notepad
    llm_client: LLMClient
    constitution: ConstitutionMode
    event_bus: EventBus
    notepad_writer: AsyncNotepadWriter
    session_id: str | None = None
    trace_id: str | None = None


class ToolRegistry:
    def __init__(
        self,
        settings: Settings,
        notepad: Notepad,
        llm_client: LLMClient,
        event_bus: EventBus,
        notepad_writer: AsyncNotepadWriter,
    ):
        self.settings = settings
        self.notepad = notepad
        self.llm_client = llm_client
        self.event_bus = event_bus
        self.notepad_writer = notepad_writer
        self._tools: dict[str, ToolSpec] = {}
        self._web_cache: OrderedDict[str, str] = OrderedDict()
        self._web_cache_lock = asyncio.Lock()
        self._mcp_servers: dict[str, "_MCPServerHandle"] = {}
        self._mcp_lock = asyncio.Lock()
        self._skills_manifest_path = self.settings.skills_dir / "skills_manifest.json"
        self._skills_archive_dir = self.settings.skills_dir / "archive"
        self._register_defaults()
        self._load_manifest_skills()

    def list_tools(self) -> list[ToolSpec]:
        return sorted(self._tools.values(), key=lambda tool: tool.name)

    def register_tool(self, spec_or_name: ToolSpec | str, handler: ToolHandler | None = None, *, description: str = "", restricted_safe: bool = False) -> None:
        if isinstance(spec_or_name, ToolSpec):
            self._register(spec_or_name)
            return
        if handler is None:
            raise ValueError("register_tool requires a handler when no ToolSpec is supplied.")
        self._register(
            ToolSpec(
                name=spec_or_name,
                description=description or f"User supplied tool {spec_or_name}",
                restricted_safe=restricted_safe,
                handler=handler,
            )
        )

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        constitution: ConstitutionMode,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Unknown tool: {name}")

        context = ToolContext(
            settings=self.settings,
            notepad=self.notepad,
            llm_client=self.llm_client,
            constitution=constitution,
            event_bus=self.event_bus,
            notepad_writer=self.notepad_writer,
            session_id=session_id,
            trace_id=trace_id,
        )
        last_error: Exception | None = None
        attempts = 2
        for attempt in range(1, attempts + 1):
            try:
                result = await asyncio.wait_for(
                    spec.handler(arguments, context),
                    timeout=self.settings.tool_timeout_seconds,
                )
                log_future = await self.notepad_writer.enqueue(
                    entry_type="tool_call",
                    payload={
                        "tool": name,
                        "arguments": arguments,
                        "result": result,
                        "attempt": attempt,
                        "session_id": session_id,
                        "trace_id": trace_id,
                    },
                    constitution=constitution,
                )
                await log_future
                await self.event_bus.publish(
                    LisaEvent(
                        type="tool.completed",
                        payload={
                            "tool": name,
                            "result": result,
                            "attempt": attempt,
                            "session_id": session_id,
                            "trace_id": trace_id,
                        },
                    )
                )
                return result
            except Exception as exc:  # pragma: no cover - retry path
                last_error = exc
                await self.notepad_writer.enqueue(
                    entry_type="tool_error",
                    payload={
                        "tool": name,
                        "arguments": arguments,
                        "error": str(exc),
                        "attempt": attempt,
                        "session_id": session_id,
                        "trace_id": trace_id,
                    },
                    constitution=constitution,
                )
                await self.event_bus.publish(
                    LisaEvent(
                        type="tool.failed",
                        payload={
                            "tool": name,
                            "error": str(exc),
                            "attempt": attempt,
                            "session_id": session_id,
                            "trace_id": trace_id,
                        },
                    )
                )
                if attempt < attempts:
                    continue
                
                # Check if this is an evolved skill and track its error
                CORE_TOOLS = {
                    "search_notepad", "dashboard_update", "file_read", "file_write",
                    "file_edit", "terminal_exec", "call_external_llm", "add_skill",
                    "rollback_skill", "mcp_call", "browser_fetch", "browser_search"
                }
                if name not in CORE_TOOLS:
                    import logging
                    logger = logging.getLogger("lisa.tools")
                    from utils.evolution_guard import track_skill_error
                    rolled_back = await asyncio.to_thread(
                        track_skill_error,
                        name,
                        str(exc),
                        self.settings.skills_dir,
                        self.settings.backup_dir
                    )
                    if rolled_back:
                        logger.warning(f"Auto-rollback triggered for skill '{name}' due to high error rate. Reloading manifest skills.")
                        self._load_manifest_skills()
                
                break

        assert last_error is not None
        raise last_error

    def _register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def _register_defaults(self) -> None:
        self._register(
            ToolSpec(
                name="search_notepad",
                description="Search the append-only SQLite action ledger.",
                restricted_safe=True,
                handler=self._search_notepad,
            )
        )
        self._register(
            ToolSpec(
                name="dashboard_update",
                description="Persist a dashboard metric for future UI display.",
                restricted_safe=True,
                handler=self._dashboard_update,
            )
        )
        self._register(
            ToolSpec(
                name="file_read",
                description="Read a text file from the workspace.",
                restricted_safe=True,
                handler=self._file_read,
            )
        )
        self._register(
            ToolSpec(
                name="file_write",
                description="Write a text file inside the workspace.",
                restricted_safe=False,
                handler=self._file_write,
            )
        )
        self._register(
            ToolSpec(
                name="file_edit",
                description="Find and replace text inside a workspace file.",
                restricted_safe=False,
                handler=self._file_edit,
            )
        )
        self._register(
            ToolSpec(
                name="terminal_exec",
                description="Run a shell command in the configured workspace.",
                restricted_safe=False,
                handler=self._terminal_exec,
            )
        )
        self._register(
            ToolSpec(
                name="call_external_llm",
                description="Call the configured OpenAI-compatible external model.",
                restricted_safe=True,
                handler=self._call_external_llm,
            )
        )
        self._register(
            ToolSpec(
                name="add_skill",
                description="Persist reusable Python code under the skills directory.",
                restricted_safe=False,
                handler=self._add_skill,
            )
        )
        self._register(
            ToolSpec(
                name="rollback_skill",
                description="Restore a previous archived version of a skill.",
                restricted_safe=False,
                handler=self._rollback_skill,
            )
        )
        self._register(
            ToolSpec(
                name="mcp_call",
                description="Placeholder for future MCP server calls.",
                restricted_safe=True,
                handler=self._mcp_call,
            )
        )
        if self.settings.enable_browser_tools:
            self._register(
                ToolSpec(
                    name="browser_fetch",
                    description="Fetch a webpage and optionally extract readable text.",
                    restricted_safe=True,
                    handler=self._browser_fetch,
                )
            )
            self._register(
                ToolSpec(
                    name="browser_search",
                    description="Run a lightweight DuckDuckGo HTML search.",
                    restricted_safe=True,
                    handler=self._browser_search,
                )
            )

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (self.settings.workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        try:
            candidate.relative_to(self.settings.workspace_root)
        except ValueError as exc:
            raise ValueError("Path must stay inside the configured workspace.") from exc

        return candidate

    def _ensure_restricted_safe_command(self, command: str, constitution: ConstitutionMode) -> None:
        if constitution == ConstitutionMode.UNRESTRICTED:
            return

        patterns = (
            r"\brm\s+-rf\b",
            r"\bdel\s+/.*/s\b",
            r"\bformat\b",
            r"\bdiskpart\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bmkfs\b",
            r"\bchmod\s+000\b",
            r">\s*/dev/null\s+2>&1",
        )
        lowered = command.lower()
        for pattern in patterns:
            if re.search(pattern, lowered):
                raise PermissionError(
                    "Restricted mode blocked a potentially destructive shell command."
                )

    async def _search_notepad(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        query = str(arguments.get("query", "")).strip()
        limit = int(arguments.get("limit", 10))
        if not query:
            raise ValueError("search_notepad requires a non-empty 'query'.")
        return await asyncio.to_thread(context.notepad.search, query, limit)

    async def _dashboard_update(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        metric = str(arguments.get("metric", "")).strip()
        value = str(arguments.get("value", "")).strip()
        if not metric or not value:
            raise ValueError("dashboard_update requires 'metric' and 'value'.")
        await asyncio.to_thread(context.notepad.add_metric, metric=metric, value=value)
        await context.event_bus.publish(
            LisaEvent(
                type="dashboard.metric",
                payload={
                    "metric": metric,
                    "value": value,
                    "session_id": context.session_id,
                    "trace_id": context.trace_id,
                },
            )
        )
        return {"status": "ok", "metric": metric, "value": value}

    async def _file_read(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        path = self._resolve_workspace_path(str(arguments.get("path", "")))
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return {"path": str(path), "content": content}

    async def _file_write(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        path = self._resolve_workspace_path(str(arguments.get("path", "")))
        content = str(arguments.get("content", ""))
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, content, encoding="utf-8")
        return {"path": str(path), "bytes_written": len(content.encode("utf-8"))}

    async def _file_edit(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        path = self._resolve_workspace_path(str(arguments.get("path", "")))
        find = str(arguments.get("find", ""))
        replace = str(arguments.get("replace", ""))
        if not find:
            raise ValueError("file_edit requires a non-empty 'find' string.")

        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        occurrences = content.count(find)
        if occurrences == 0:
            raise ValueError("No occurrences of the target text were found.")
        updated = content.replace(find, replace)
        await asyncio.to_thread(path.write_text, updated, encoding="utf-8")
        return {"path": str(path), "replacements": occurrences}

    async def _terminal_exec(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        command = str(arguments.get("command", "")).strip()
        timeout = int(arguments.get("timeout", 30))
        if not command:
            raise ValueError("terminal_exec requires a non-empty 'command'.")

        self._ensure_restricted_safe_command(command, context.constitution)

        if shutil.which("docker") is None:
            raise RuntimeError("Docker is not available on this machine.")

        process = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "512m",
            "--pids-limit",
            "128",
            "--mount",
            f"type=bind,source={self.settings.workspace_root.as_posix()},target=/workspace",
            "-w",
            "/workspace",
            "-e",
            "HOME=/tmp",
            self.settings.docker_image,
            "sh",
            "-lc",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise TimeoutError(f"Command timed out after {timeout} seconds.")

        return {
            "returncode": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }

    async def _call_external_llm(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        provider = str(arguments.get("provider", "")).strip()
        model = str(arguments.get("model", "")).strip() or None
        prompt = str(arguments.get("prompt", "")).strip()
        system_prompt = str(arguments.get("system_prompt", "")).strip() or None
        max_tokens = int(arguments.get("max_tokens", 800))
        if provider and context.settings.model_provider and provider != context.settings.model_provider:
            raise ValueError(
                f"Configured provider is '{context.settings.model_provider}', got '{provider}'."
            )
        if not prompt:
            raise ValueError("call_external_llm requires a non-empty 'prompt'.")

        result = await context.llm_client.call_external_llm(
            provider=provider or context.settings.freellmapi_default_provider,
            prompt=prompt,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt or "You are LISA's external reasoning coprocessor.",
        )
        await asyncio.to_thread(
            context.notepad.add_metric,
            metric=f"llm_tokens_{result.provider}",
            value=str(result.usage["total_tokens"]),
        )
        await context.event_bus.publish(
            LisaEvent(
                type="external_llm.completed",
                payload={
                    "provider": result.provider,
                    "model": result.model,
                    "usage": result.usage,
                    "session_id": context.session_id,
                    "trace_id": context.trace_id,
                },
            )
        )
        return {
            "provider": result.provider,
            "model": result.model,
            "content": result.content,
            "usage": result.usage,
        }

    async def _add_skill(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        name = str(arguments.get("name", "")).strip()
        code = str(arguments.get("code", "")).strip()
        body = str(arguments.get("body", "")).strip()
        if not name:
            raise ValueError("add_skill requires a non-empty 'name'.")
        if not code and not body:
            raise ValueError("add_skill requires 'code' or 'body'.")

        slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "skill"
        path = context.settings.skills_dir / f"{slug}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Take a snapshot of skills directory before modifying
        from utils.evolution_guard import snapshot_skills_dir, record_skill_deployment
        snapshot_path = await asyncio.to_thread(
            snapshot_skills_dir,
            context.settings.skills_dir,
            context.settings.backup_dir
        )

        archive_path = None
        if path.exists():
            archive_path = await asyncio.to_thread(self._archive_skill_version, slug, path)
        if not code:
            function_name = slug
            code = f"def {function_name}(context):\n{textwrap.indent(body or 'pass', '    ')}\n"
        elif "def " not in code and "async def " not in code:
            function_name = slug
            code = f"def {function_name}(context):\n{textwrap.indent(code or 'pass', '    ')}\n"
        compile(code, f"<skill:{slug}>", "exec")
        await asyncio.to_thread(path.write_text, code, encoding="utf-8")
        
        # Record skill deployment in the journal
        record_skill_deployment(slug, path, snapshot_path, context.settings.backup_dir)

        module = self._import_skill_module(slug, path)
        export_name = self._find_skill_export(module, slug)
        if export_name is None:
            raise ValueError("Skill modules must define a callable named 'skill' or matching the skill slug.")
        self.register_tool(slug, self._build_skill_wrapper(slug, getattr(module, export_name)))
        await asyncio.to_thread(self._update_skill_manifest, slug, name, path, archive_path)
        return {
            "path": str(path),
            "status": "saved",
            "archive_path": str(archive_path) if archive_path is not None else None,
            "version": self._skill_version_count(slug),
        }

    async def _rollback_skill(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        name = str(arguments.get("name", "")).strip()
        version = str(arguments.get("version", "")).strip()
        if not name:
            raise ValueError("rollback_skill requires a non-empty 'name'.")
        slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "skill"
        restored = await asyncio.to_thread(self._restore_skill_version, slug, version)
        return {"status": "rolled_back", **restored}

    async def _mcp_call(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        server_name = str(arguments.get("server_name", "")).strip()
        method = str(arguments.get("method", "")).strip()
        params = arguments.get("params", {})
        command = arguments.get("command") or arguments.get("server_command")
        if not server_name:
            server_name = str(arguments.get("name", "")).strip()
        if not command:
            raise ValueError("mcp_call requires 'command' or 'server_command'.")
        if not method:
            raise ValueError("mcp_call requires a non-empty 'method'.")
        if isinstance(command, str):
            command_parts = shlex.split(command)
        else:
            command_parts = [str(part) for part in command]
        result = await self._run_mcp_call(
            server_name=server_name or "default",
            command=command_parts,
            method=method,
            params=params if isinstance(params, dict) else {"value": params},
            timeout=int(arguments.get("timeout", context.settings.tool_timeout_seconds)),
        )
        return result

    async def _browser_fetch(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        url = str(arguments.get("url", "")).strip()
        extract_text = bool(arguments.get("extract_text", True))
        if not url:
            raise ValueError("browser_fetch requires a non-empty 'url'.")

        html: str | None = None
        async with self._web_cache_lock:
            if url in self._web_cache:
                html = self._web_cache.pop(url)
                self._web_cache[url] = html

        if html is None:
            async with httpx.AsyncClient(timeout=context.settings.external_timeout_seconds) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                html = response.text
            async with self._web_cache_lock:
                self._web_cache[url] = html
                while len(self._web_cache) > 32:
                    self._web_cache.popitem(last=False)

        result: dict[str, Any] = {"url": url, "html": html}
        if extract_text:
            result["text"] = self._html_to_text(html)
        return result

    async def _browser_search(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("browser_search requires a non-empty 'query'.")

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        async with httpx.AsyncClient(timeout=context.settings.external_timeout_seconds) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for link in soup.select("a.result__a"):
            title = link.get_text(" ", strip=True)
            href = link.get("href")
            if title and href:
                results.append({"title": title, "url": href})
            if len(results) >= 5:
                break

        return {"query": query, "results": results}

    async def _run_mcp_call(
        self,
        server_name: str,
        command: list[str],
        method: str,
        params: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        handle = await self._ensure_mcp_server(server_name, command, timeout)
        async with handle.lock:
            initialize_result = await self._mcp_initialize(handle, timeout=timeout)
            request_id = handle.next_request_id()
            response = await self._mcp_request(
                handle=handle,
                request={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
                timeout=timeout,
            )
            return {
                "server_name": server_name,
                "method": method,
                "result": response,
                "initialize": initialize_result,
            }

    async def _ensure_mcp_server(
        self,
        server_name: str,
        command: list[str],
        timeout: int,
    ) -> "_MCPServerHandle":
        async with self._mcp_lock:
            handle = self._mcp_servers.get(server_name)
            if handle is not None and handle.process.returncode is None:
                return handle

            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            handle = _MCPServerHandle(
                server_name=server_name,
                command=command,
                process=process,
                timeout=timeout,
            )
            self._mcp_servers[server_name] = handle
            return handle

    async def _mcp_initialize(self, handle: "_MCPServerHandle", timeout: int) -> dict[str, Any]:
        if handle.initialized:
            return handle.initialize_result or {}

        initialize_request = {
            "jsonrpc": "2.0",
            "id": handle.next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "LISA",
                    "version": "0.1.0",
                },
            },
        }
        response = await self._mcp_request(handle, initialize_request, timeout=timeout)
        await self._mcp_notify(
            handle,
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            },
            timeout=timeout,
        )
        handle.initialized = True
        handle.initialize_result = response
        return response

    async def _mcp_request(
        self,
        handle: "_MCPServerHandle",
        request: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        if handle.process.stdin is None or handle.process.stdout is None:
            raise RuntimeError("MCP server pipes are not available.")
        payload = json.dumps(request, ensure_ascii=True)
        handle.process.stdin.write((payload + "\n").encode("utf-8"))
        await handle.process.stdin.drain()

        expected_id = request.get("id")
        deadline = asyncio.get_running_loop().time() + timeout
        stderr_task = asyncio.create_task(self._collect_stderr(handle), name=f"mcp-stderr-{handle.server_name}")
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"MCP request timed out after {timeout} seconds.")
                try:
                    raw = await asyncio.wait_for(handle.process.stdout.readline(), timeout=remaining)
                except TimeoutError:
                    raise TimeoutError(f"MCP request timed out after {timeout} seconds.")
                if not raw:
                    stderr_output = await self._drain_task(stderr_task)
                    raise RuntimeError(
                        f"MCP server '{handle.server_name}' exited unexpectedly."
                        + (f" stderr: {stderr_output}" if stderr_output else "")
                    )
                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if message.get("id") != expected_id:
                    continue
                if "error" in message:
                    error = message["error"]
                    raise RuntimeError(f"MCP error: {error}")
                result = message.get("result")
                if isinstance(result, dict):
                    return result
                return {"value": result}
        finally:
            stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await stderr_task

    async def _mcp_notify(
        self,
        handle: "_MCPServerHandle",
        notification: dict[str, Any],
        timeout: int,
    ) -> None:
        if handle.process.stdin is None:
            raise RuntimeError("MCP server stdin is not available.")
        payload = json.dumps(notification, ensure_ascii=True)
        handle.process.stdin.write((payload + "\n").encode("utf-8"))
        await asyncio.wait_for(handle.process.stdin.drain(), timeout=timeout)

    async def _collect_stderr(self, handle: "_MCPServerHandle") -> str:
        if handle.process.stderr is None:
            return ""
        chunks: list[str] = []
        while True:
            raw = await handle.process.stderr.readline()
            if not raw:
                break
            chunks.append(raw.decode("utf-8", errors="replace").strip())
        return "\n".join(chunk for chunk in chunks if chunk)

    async def _drain_task(self, task: asyncio.Task[str]) -> str:
        with suppress(asyncio.CancelledError):
            return await task
        return ""

    @staticmethod
    def _html_to_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for node in soup(["script", "style"]):
            node.decompose()
        text = soup.get_text(" ", strip=True)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _load_manifest_skills(self) -> None:
        if not self._skills_manifest_path.exists():
            self._skills_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self._skills_manifest_path.write_text(json.dumps({"skills": {}}, indent=2), encoding="utf-8")
            return

        try:
            payload = json.loads(self._skills_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"skills": {}}

        skills = payload.get("skills", {})
        if not isinstance(skills, dict):
            return
        for slug, meta in skills.items():
            if not isinstance(meta, dict):
                continue
            path_value = meta.get("path")
            if not path_value:
                continue
            path = Path(str(path_value))
            if not path.is_absolute():
                path = (self.settings.workspace_root / path).resolve()
            if not path.exists():
                continue
            try:
                module = self._import_skill_module(str(slug), path)
            except Exception:
                continue
            export_name = self._find_skill_export(module, str(slug))
            if export_name is None:
                continue
            self.register_tool(str(slug), self._build_skill_wrapper(str(slug), getattr(module, export_name)))

    def _archive_skill_version(self, slug: str, path: Path) -> Path:
        self._skills_archive_dir.mkdir(parents=True, exist_ok=True)
        skill_archive_dir = self._skills_archive_dir / slug
        skill_archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = skill_archive_dir / f"{timestamp}.py"
        shutil.copy2(path, archive_path)
        return archive_path

    def _skill_version_count(self, slug: str) -> int:
        skill_record = self._skill_manifest().get(slug, {})
        versions = skill_record.get("versions", [])
        return len(versions) + 1

    def _update_skill_manifest(self, slug: str, name: str, path: Path, archive_path: Path | None) -> None:
        manifest = self._skill_manifest()
        record = manifest.get(slug, {"name": name, "versions": []})
        versions = list(record.get("versions", []))
        if path.exists():
            versions.append(
                {
                    "path": str(path),
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        if archive_path is not None:
            versions.append(
                {
                    "path": str(archive_path),
                    "archived_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        manifest[slug] = {
            "name": name,
            "path": str(path),
            "versions": versions[-10:],
        }
        self._skills_manifest_path.write_text(json.dumps({"skills": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")

    def _restore_skill_version(self, slug: str, version: str) -> dict[str, Any]:
        manifest = self._skill_manifest()
        record = manifest.get(slug)
        if not isinstance(record, dict):
            raise KeyError(f"Unknown skill: {slug}")
        versions = record.get("versions", [])
        if not isinstance(versions, list) or not versions:
            raise ValueError(f"No archived versions are available for skill '{slug}'.")

        selected_path: Path | None = None
        if version and version != "latest":
            for entry in versions:
                if not isinstance(entry, dict):
                    continue
                entry_path = str(entry.get("path") or "")
                if version in entry_path:
                    selected_path = Path(entry_path)
                    break
        else:
            selected_path = Path(str(record.get("path") or ""))
            if not selected_path.exists():
                selected_path = None

        if selected_path is None or not selected_path.exists():
            for entry in reversed(versions):
                if not isinstance(entry, dict):
                    continue
                entry_path = Path(str(entry.get("path") or ""))
                if entry_path.exists():
                    selected_path = entry_path
                    break

        if selected_path is None or not selected_path.exists():
            raise FileNotFoundError(f"Unable to restore skill '{slug}' from the recorded versions.")

        target_path = self.settings.skills_dir / f"{slug}.py"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_path, target_path)
        module = self._import_skill_module(slug, target_path)
        export_name = self._find_skill_export(module, slug)
        if export_name is None:
            raise ValueError(f"Restored skill '{slug}' does not expose a callable entry point.")
        self.register_tool(slug, self._build_skill_wrapper(slug, getattr(module, export_name)))
        record["path"] = str(target_path)
        self._skills_manifest_path.write_text(json.dumps({"skills": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"path": str(target_path), "restored_from": str(selected_path)}

    def _skill_manifest(self) -> dict[str, Any]:
        if not self._skills_manifest_path.exists():
            return {}
        try:
            payload = json.loads(self._skills_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        skills = payload.get("skills", {})
        return skills if isinstance(skills, dict) else {}

    @staticmethod
    def _find_skill_export(module: Any, slug: str) -> str | None:
        if hasattr(module, "skill") and callable(getattr(module, "skill")):
            return "skill"
        if hasattr(module, slug) and callable(getattr(module, slug)):
            return slug
        for name, value in vars(module).items():
            if name.startswith("_"):
                continue
            if callable(value):
                return name
        return None

    def _import_skill_module(self, slug: str, path: Path):
        module_name = f"lisa_skill_{slug}_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import skill module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def _build_skill_wrapper(self, slug: str, skill_callable: Callable[..., Any]) -> ToolHandler:
        async def _wrapped(arguments: dict[str, Any], context: ToolContext) -> Any:
            result = skill_callable(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return result

        return _wrapped


@dataclass(slots=True)
class _MCPServerHandle:
    server_name: str
    command: list[str]
    process: asyncio.subprocess.Process
    timeout: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    initialized: bool = False
    initialize_result: dict[str, Any] | None = None
    _request_counter: int = 0

    def next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter
