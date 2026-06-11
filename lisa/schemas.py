from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    max_tokens: int = Field(default=800, ge=64, le=4096)


class ToolCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw: str | None = None


class ToolResult(BaseModel):
    tool: str = Field(min_length=1)
    success: bool
    output: Any = None
    error: str | None = None
    attempt: int | None = None


class InboundMessage(BaseModel):
    source: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    text: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | None = None
    priority: int = Field(default=1, ge=0, le=10)
    metadata: dict[str, Any] = Field(default_factory=dict)
    reply_to_message_id: int | None = None
    message_id: str | None = None


class BrainTask(BaseModel):
    inbound: InboundMessage
    max_tokens: int = Field(default=800, ge=64, le=4096)
    constitution: str | None = None
    persona_weights: dict[str, float] | None = None
    context_summary: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    follow_up: bool = False


class HubAck(BaseModel):
    accepted: bool
    queued: bool
    job_id: str | None = None
    retry_after_seconds: int | None = None
    detail: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    constitution: str
    personas: dict[str, float]
    tool_suggestions: list[str]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    used_external_model: bool
    notes: list[str] = Field(default_factory=list)
    delivery_hints: dict[str, Any] = Field(default_factory=dict)


class ToolSummary(BaseModel):
    name: str
    description: str
    restricted_safe: bool


class ToolInvokeRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolInvokeResponse(BaseModel):
    name: str
    success: bool
    output: Any


class NotepadEntry(BaseModel):
    id: int
    entry_type: str
    constitution: str
    created_at: datetime
    payload: dict[str, Any]


class NotepadSearchResponse(BaseModel):
    query: str
    results: list[NotepadEntry]


class ConstitutionStateResponse(BaseModel):
    mode: str
    reason: str | None = None
    updated_at: datetime


class DashboardMetric(BaseModel):
    metric: str
    value: str
    created_at: datetime


class BotDispatchRequest(BaseModel):
    channel: Literal["direct", "telegram", "slack", "whatsapp"] = "direct"
    user_id: str | None = None
    text: str = Field(min_length=1)
    session_id: str | None = None
    source: str = "api"
    priority: int = Field(default=0, ge=0, le=10)
    max_tokens: int = Field(default=800, ge=64, le=4096)
    deliver: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    reply_to_message_id: int | None = None


class BotDispatchResponse(BaseModel):
    accepted: bool
    delivered: bool
    channel: str
    session_id: str
    job_id: str | None = None
    response: ChatResponse | None = None
    delivery: dict[str, Any] = Field(default_factory=dict)
