from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from lisa.conductor import ConductorJob, TaskConductor
from lisa.schemas import ToolCall, ToolResult


@dataclass(slots=True)
class BrainTask:
    id: str = field(default_factory=lambda: str(uuid4()))
    conversation: list[dict[str, str]] = field(default_factory=list)
    persona_blend: dict[str, float] | None = None
    constitution: str = "restricted"
    priority: int = 0
    callback_future: asyncio.Future[Any] | None = None
    max_tokens: int = 800
    tool_results: list[ToolResult] = field(default_factory=list)
    follow_up: bool = False


@dataclass(slots=True)
class ToolTask:
    id: str = field(default_factory=lambda: str(uuid4()))
    tool_call: ToolCall = field(
        default_factory=lambda: ToolCall(name="placeholder", arguments={})
    )
    constitution: str = "restricted"
    priority: int = 0
    callback_future: asyncio.Future[Any] | None = None
    session_id: str | None = None
    trace_id: str | None = None


__all__ = ["BrainTask", "ConductorJob", "TaskConductor", "ToolTask"]
