from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PERMISSION_LEVELS = {
    "L0": 0,
    "L1": 1,
    "L2": 2,
    "L3": 3,
}


TOOL_PERMISSION_MAP = {
    "search_notepad": "L0",
    "browser_fetch": "L0",
    "browser_search": "L0",
    "dashboard_update": "L0",
    "file_read": "L0",
    "file_write": "L1",
    "file_edit": "L1",
    "call_external_llm": "L1",
    "mcp_call": "L1",
    "add_skill": "L1",
    "rollback_skill": "L1",
    "send_message": "L1",
    "terminal_exec": "L2",
    "docker_exec": "L2",
    "install_package": "L2",
    "delete_file": "L3",
    "system_command": "L3",
    "deploy": "L3",
}


@dataclass(slots=True)
class SandboxResult:
    workspace_root: Path
    idempotency_key: str


class ToolSandbox:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.tasks_root = workspace_root / "workspace"
        self.tasks_root.mkdir(parents=True, exist_ok=True)

    def permission_for(self, tool_name: str) -> str:
        return TOOL_PERMISSION_MAP.get(tool_name, "L1")

    def has_permission(
        self,
        approved_levels: list[str],
        required_level: str,
        *,
        explicit_user_grant: bool,
        two_factor_confirmed: bool,
    ) -> bool:
        required_value = PERMISSION_LEVELS.get(required_level, 1)
        approved_value = max(
            (PERMISSION_LEVELS.get(level, 0) for level in approved_levels), default=0
        )
        if required_value <= approved_value:
            return True
        if required_level == "L2" and explicit_user_grant:
            return True
        if required_level == "L3" and two_factor_confirmed:
            return True
        return False

    def prepare(
        self, task_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> SandboxResult:
        task_root = (self.tasks_root / task_id).resolve()
        task_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "tool": tool_name,
            "arguments": arguments,
        }
        idempotency_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return SandboxResult(workspace_root=task_root, idempotency_key=idempotency_key)

    async def cleanup(self, task_id: str) -> None:
        task_root = self.tasks_root / task_id
        if not task_root.exists():
            return
        await asyncio.to_thread(self._remove_tree, task_root)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
