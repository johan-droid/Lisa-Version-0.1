from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from lisa.schemas import ToolCall


@dataclass(slots=True)
class ParsedOutput:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_text: str | None = None
    used_local_model: bool = False
    persona_prefix_shape: tuple[int, int] | None = None


class FunctionCallParser:
    """Parse structured tool calls from model output.

    The parser accepts the compact `<tool>{"name":"...", "args":{...}}</tool>`
    form, fenced JSON, plain JSON arrays, or simple key-value fallbacks.
    """

    TOOL_PATTERN = re.compile(
        r"<tool(?:_call)?>(.*?)</tool(?:_call)?>", re.IGNORECASE | re.DOTALL
    )
    FENCED_JSON_PATTERN = re.compile(
        r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL
    )

    def parse(self, text: str) -> ParsedOutput:
        raw_text = text
        tool_calls: list[ToolCall] = []
        consumed_spans: list[tuple[int, int]] = []

        for match in self.TOOL_PATTERN.finditer(text):
            payload = match.group(1).strip()
            tool_calls.extend(self._parse_payload(payload))
            consumed_spans.append(match.span())

        for match in self.FENCED_JSON_PATTERN.finditer(text):
            payload = match.group(1).strip()
            parsed = self._parse_payload(payload)
            if parsed:
                tool_calls.extend(parsed)
                consumed_spans.append(match.span())

        for payload, span in self._scan_for_json_objects(text):
            parsed = self._parse_payload(payload)
            if parsed:
                tool_calls.extend(parsed)
                consumed_spans.append(span)

        clean_text = self._remove_spans(text, consumed_spans)
        return ParsedOutput(
            text=self._normalize_whitespace(clean_text) or text.strip(),
            tool_calls=self._dedupe(tool_calls),
            raw_text=raw_text,
        )

    def _parse_payload(self, payload: str) -> list[ToolCall]:
        payload = payload.strip()
        if not payload:
            return []

        if payload.startswith("{") or payload.startswith("["):
            return self._parse_json_payload(payload)

        if payload.startswith("<"):
            return self._parse_xml_payload(payload)

        return self._parse_key_value_payload(payload)

    def _parse_json_payload(self, payload: str) -> list[ToolCall]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []
        return self._json_to_tool_calls(data)

    def _parse_xml_payload(self, payload: str) -> list[ToolCall]:
        if not payload.startswith("<tool"):
            return []

        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            return []

        if root.tag != "tool":
            return []

        name = root.get("name") or ""
        arguments: dict[str, Any] = {}
        if root.text and root.text.strip():
            try:
                parsed = json.loads(root.text.strip())
            except json.JSONDecodeError:
                arguments = {"value": root.text.strip()}
            else:
                if isinstance(parsed, dict):
                    arguments = parsed

        for child in root:
            if child.tag == "name":
                name = (child.text or "").strip() or name
            elif child.tag in {"args", "arguments"}:
                try:
                    parsed = json.loads((child.text or "").strip() or "{}")
                except json.JSONDecodeError:
                    arguments = {"value": (child.text or "").strip()}
                else:
                    if isinstance(parsed, dict):
                        arguments = parsed
        if not name:
            return []
        return [ToolCall(name=name, arguments=arguments, raw=payload)]

    def _parse_key_value_payload(self, payload: str) -> list[ToolCall]:
        fields: dict[str, Any] = {}
        for line in payload.splitlines():
            if ":" not in line and "=" not in line:
                continue
            separator = ":" if ":" in line else "="
            key, value = line.split(separator, 1)
            fields[key.strip().lower()] = value.strip()

        name = str(
            fields.get("name") or fields.get("tool") or fields.get("function") or ""
        ).strip()
        if not name:
            return []

        arguments: dict[str, Any] = {}
        if "arguments" in fields:
            try:
                parsed_arguments = json.loads(str(fields["arguments"]))
            except json.JSONDecodeError:
                arguments = {"value": fields["arguments"]}
            else:
                if isinstance(parsed_arguments, dict):
                    arguments = parsed_arguments
        return [ToolCall(name=name, arguments=arguments, raw=payload)]

    def _json_to_tool_calls(self, data: Any) -> list[ToolCall]:
        if isinstance(data, list):
            collected: list[ToolCall] = []
            for item in data:
                collected.extend(self._json_to_tool_calls(item))
            return collected

        if not isinstance(data, dict):
            return []

        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            collected: list[ToolCall] = []
            for item in data["tool_calls"]:
                collected.extend(self._json_to_tool_calls(item))
            return collected

        if "tool_call" in data:
            return self._json_to_tool_calls(data["tool_call"])

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
        return [
            ToolCall(
                name=name, arguments=arguments, raw=json.dumps(data, ensure_ascii=False)
            )
        ]

    def _scan_for_json_objects(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        decoder = json.JSONDecoder()
        candidates: list[tuple[str, tuple[int, int]]] = []
        index = 0
        while index < len(text):
            if text[index] not in "{[":
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
            signature = (
                call.name,
                json.dumps(call.arguments, sort_keys=True, default=str),
            )
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(call)
        return unique


ToolCallParser = FunctionCallParser
