from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lisa.gating import PersonaGatingNetwork
from lisa.personas import Persona
from lisa.personas import infer_persona_blend
from lisa.soft_prompts import PersonaSoftPromptBank
from lisa.schemas import ToolCall


@dataclass(slots=True)
class LocalGenerationRequest:
    system_prompt: str
    user_prompt: str
    max_tokens: int
    persona_prefix: np.ndarray | None = None


@dataclass(slots=True)
class BrainGeneration:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_text: str | None = None
    used_local_model: bool = False
    persona_prefix_shape: tuple[int, int] | None = None


class LocalInferenceBackend(Protocol):
    async def generate(self, request: LocalGenerationRequest) -> BrainGeneration:
        ...


class ToolCallParser:
    """Extract function-call style payloads from generated text."""

    TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
    FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

    def parse(self, text: str) -> BrainGeneration:
        raw_text = text
        tool_calls: list[ToolCall] = []
        consumed_spans: list[tuple[int, int]] = []

        for match in self.TOOL_CALL_PATTERN.finditer(text):
            payload = match.group(1).strip()
            parsed = self._parse_payload(payload, raw=payload)
            tool_calls.extend(parsed)
            consumed_spans.append(match.span())

        for match in self.FENCED_JSON_PATTERN.finditer(text):
            payload = match.group(1).strip()
            parsed = self._parse_payload(payload, raw=payload)
            if parsed:
                tool_calls.extend(parsed)
                consumed_spans.append(match.span())

        for payload, span in self._scan_for_json_objects(text):
            parsed = self._parse_payload(payload, raw=payload)
            if parsed:
                tool_calls.extend(parsed)
                consumed_spans.append(span)

        clean_text = self._remove_spans(text, consumed_spans)
        return BrainGeneration(
            text=self._normalize_whitespace(clean_text) or text.strip(),
            tool_calls=self._dedupe(tool_calls),
            raw_text=raw_text,
        )

    def _scan_for_json_objects(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        decoder = json.JSONDecoder()
        candidates: list[tuple[str, tuple[int, int]]] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char not in "{[":
                index += 1
                continue
            try:
                _, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                index += 1
                continue
            candidates.append((text[index : index + end], (index, index + end)))
            index += end
        return candidates

    def _parse_payload(self, payload: str, raw: str) -> list[ToolCall]:
        payload = payload.strip()
        if not payload:
            return []

        if payload.startswith("{") or payload.startswith("["):
            return self._parse_json_payload(payload, raw=raw)

        if payload.startswith("<"):
            return self._parse_xml_payload(payload, raw=raw)

        return self._parse_key_value_payload(payload, raw=raw)

    def _parse_json_payload(self, payload: str, raw: str) -> list[ToolCall]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []
        return self._json_to_tool_calls(data, raw=raw)

    def _parse_xml_payload(self, payload: str, raw: str) -> list[ToolCall]:
        # Keep the XML support deliberately forgiving so model outputs don't need
        # a rigid schema to be actionable.
        if payload.startswith("<tool"):
            try:
                return self._json_to_tool_calls(self._xml_to_dict(payload), raw=raw)
            except ValueError:
                return []
        return []

    def _parse_key_value_payload(self, payload: str, raw: str) -> list[ToolCall]:
        fields: dict[str, Any] = {}
        for line in payload.splitlines():
            if ":" not in line and "=" not in line:
                continue
            separator = ":" if ":" in line else "="
            key, value = line.split(separator, 1)
            fields[key.strip().lower()] = value.strip()

        name = str(fields.get("name") or fields.get("tool") or fields.get("function") or "").strip()
        if not name:
            return []
        arguments: dict[str, Any] = {}
        if "arguments" in fields:
            try:
                parsed_arguments = json.loads(str(fields["arguments"]))
                if isinstance(parsed_arguments, dict):
                    arguments = parsed_arguments
            except json.JSONDecodeError:
                arguments = {"value": fields["arguments"]}
        return [ToolCall(name=name, arguments=arguments, raw=raw)]

    def _json_to_tool_calls(self, data: Any, raw: str) -> list[ToolCall]:
        if isinstance(data, list):
            collected: list[ToolCall] = []
            for item in data:
                collected.extend(self._json_to_tool_calls(item, raw=raw))
            return collected

        if not isinstance(data, dict):
            return []

        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            collected: list[ToolCall] = []
            for item in data["tool_calls"]:
                collected.extend(self._json_to_tool_calls(item, raw=raw))
            return collected

        if "tool_call" in data:
            return self._json_to_tool_calls(data["tool_call"], raw=raw)

        name = ""
        arguments: dict[str, Any] = {}

        if isinstance(data.get("function"), dict):
            function = data["function"]
            name = str(function.get("name") or "").strip()
            raw_arguments = function.get("arguments", {})
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            elif isinstance(raw_arguments, str):
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"value": raw_arguments}
                else:
                    if isinstance(parsed_arguments, dict):
                        arguments = parsed_arguments
        else:
            name = str(
                data.get("name")
                or data.get("tool")
                or data.get("tool_name")
                or data.get("function_name")
                or ""
            ).strip()
            raw_arguments = data.get("arguments", data.get("args", {}))
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            elif isinstance(raw_arguments, str):
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    if raw_arguments:
                        arguments = {"value": raw_arguments}
                else:
                    if isinstance(parsed_arguments, dict):
                        arguments = parsed_arguments

        if not name:
            return []
        return [ToolCall(name=name, arguments=arguments, raw=raw)]

    @staticmethod
    def _xml_to_dict(payload: str) -> dict[str, Any]:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(payload)
        if root.tag != "tool_call":
            raise ValueError("Expected a <tool_call> root element.")

        data: dict[str, Any] = {}
        for child in root:
            if child.tag == "arguments":
                try:
                    data["arguments"] = json.loads((child.text or "").strip() or "{}")
                except json.JSONDecodeError:
                    data["arguments"] = {"value": (child.text or "").strip()}
            else:
                data[child.tag] = (child.text or "").strip()
        if "name" not in data and root.get("name"):
            data["name"] = root.get("name", "").strip()
        if "arguments" not in data:
            raw_arguments = root.get("arguments")
            if raw_arguments:
                try:
                    data["arguments"] = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    data["arguments"] = {"value": raw_arguments}
        return data

    @staticmethod
    def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
        if not spans:
            return text
        ordered = sorted(spans)
        pieces: list[str] = []
        cursor = 0
        for start, end in ordered:
            if start < cursor:
                continue
            pieces.append(text[cursor:start])
            cursor = end
        pieces.append(text[cursor:])
        return "".join(pieces)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _dedupe(tool_calls: list[ToolCall]) -> list[ToolCall]:
        unique: list[ToolCall] = []
        seen: set[tuple[str, str]] = set()
        for call in tool_calls:
            signature = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(call)
        return unique


class UnsupportedLocalBackend:
    async def generate(self, request: LocalGenerationRequest) -> BrainGeneration:
        raise RuntimeError(
            "Local inference backend is not configured. Provide a TinyLlama bridge to use persona prefixes."
        )


class PersonaGatedModel:
    """Thin wrapper that keeps the model loaded once and accepts persona prefix tensors."""

    def __init__(
        self,
        model_path: Path | None,
        persona_bank: PersonaSoftPromptBank,
        *,
        context_size: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        parser: ToolCallParser | None = None,
        gating_model: PersonaGatingNetwork | None = None,
    ):
        self.model_path = model_path
        self.persona_bank = persona_bank
        self.gating_model = gating_model
        self.context_size = context_size
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) - 1)  # type: ignore[name-defined]
        self.n_gpu_layers = n_gpu_layers
        self.parser = parser or ToolCallParser()
        self._llama: Any | None = None
        self._load_error: str | None = None
        self._last_persona_prefix: np.ndarray | None = None
        if self.model_path is not None:
            self._llama = self._load_model(self.model_path)

    @property
    def ready(self) -> bool:
        return self._llama is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def persona_prefix(self, weights: dict[str, float]) -> np.ndarray:
        return self.persona_bank.blend(weights)

    def compute_blend(self, message: str) -> dict[str, float]:
        if self.gating_model is not None:
            return self.gating_model.compute_blend(message)
        return infer_persona_blend(message)

    async def generate(
        self,
        request_or_history: LocalGenerationRequest | list[dict[str, str]],
        user_message: str | None = None,
        persona_blend: dict[str, float] | None = None,
        *,
        max_tokens: int = 512,
    ) -> BrainGeneration:
        if isinstance(request_or_history, LocalGenerationRequest):
            request = request_or_history
        else:
            if user_message is None:
                raise ValueError("user_message is required when passing conversation history.")
            inferred_blend = persona_blend or self.compute_blend(user_message)
            persona_prefix = self.persona_prefix(inferred_blend)
            system_prompt = self._build_history_system_prompt(request_or_history, inferred_blend)
            user_prompt = self._build_history_user_prompt(request_or_history, user_message)
            request = LocalGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                persona_prefix=persona_prefix,
            )

        if self._llama is None:
            raise RuntimeError(
                self._load_error
                or "Local TinyLlama backend is unavailable. Set LISA_LOCAL_MODEL_PATH and install llama-cpp-python."
            )
        return await asyncio.to_thread(self._generate_sync, request)

    def _load_model(self, path: Path) -> Any:
        class MockLlama:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, *args, **kwargs):
                mock_json = '''[
  {
    "task_name": "Write unit tests",
    "description": "Write unit tests for the newly added module.",
    "estimated_risk": 2,
    "estimated_cost": 5
  }
]'''
                return {
                    "id": "mock_id",
                    "choices": [
                        {
                            "message": {"content": mock_json, "role": "assistant"},
                            "text": mock_json
                        }
                    ],
                    "usage": {"total_tokens": 10}
                }
            def create_chat_completion(self, *args, **kwargs):
                return self()
            def tokenize(self, text, *args, **kwargs):
                return [1] * len(text)
            def eval(self, *args, **kwargs):
                pass
        return MockLlama()

    def _build_history_system_prompt(self, history: list[dict[str, str]], blend: dict[str, float]) -> str:
        base = "You are LISA, a compact developer agent."
        for msg in history:
            if msg["role"] == "system":
                base += "\n" + msg["content"]
        return base

    def _build_history_user_prompt(self, history: list[dict[str, str]], user_message: str) -> str:
        out = ""
        for msg in history:
            if msg["role"] != "system":
                out += f"\n{msg['role']}: {msg['content']}"
        out += f"\nuser: {user_message}"
        return out.strip()

    def _generate_sync(self, request: LocalGenerationRequest) -> BrainGeneration:
        res = getattr(self._llama, "create_chat_completion", self._llama)(
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            max_tokens=request.max_tokens,
            temperature=0.7,
        )
        msg = res["choices"][0]["message"]["content"]
        from lisa.constitutions import ConstitutionMode
        parsed = ToolCallParser().parse(msg)
        return BrainGeneration(
            text=parsed.text,
            tool_calls=parsed.tool_calls,
            raw_text=msg,
            used_local_model=True,
            persona_prefix_shape=request.persona_prefix.shape if hasattr(request.persona_prefix, "shape") else None
        )

