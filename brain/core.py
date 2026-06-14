from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brain.parser import FunctionCallParser
from lisa.constitutions import ConstitutionMode
from lisa.gating import PersonaGatingNetwork
from lisa.local_inference import (
    LocalGenerationRequest,
    PersonaGatedModel as LocalBrainModel,
)
from lisa.personas import Persona, infer_persona_blend
from lisa.soft_prompts import PersonaSoftPromptBank, load_persona_tensor_artifact


def _persona_order() -> tuple[str, ...]:
    return tuple(persona.value for persona in Persona)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values())) or 1.0
    normalized = {name: float(value) / total for name, value in weights.items()}
    if normalized:
        remainder = 1.0 - sum(normalized.values())
        first = next(iter(normalized))
        normalized[first] = normalized[first] + remainder
    return normalized


def _list_to_weight_map(weights: Sequence[float]) -> dict[str, float]:
    persona_names = _persona_order()
    mapped = {
        persona_names[index]: float(weight)
        for index, weight in enumerate(weights[: len(persona_names)])
    }
    return _normalize_weights(mapped or {persona_names[0]: 1.0})


def _blend_digest(weights: dict[str, float], tensor: np.ndarray) -> str:
    payload = np.asarray(tensor, dtype=np.float32).tobytes() + repr(
        sorted(weights.items())
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    symbols = "⟡⟢⟣⟤⟥⟦⟧⟨⟩◈◉◌◍◎○●"
    return "".join(symbols[byte % len(symbols)] for byte in digest[:12])


def _constitution_prompt(constitution: str) -> str:
    mode = (
        constitution.strip().lower()
        if constitution
        else ConstitutionMode.RESTRICTED.value
    )
    if mode == ConstitutionMode.UNRESTRICTED.value:
        return (
            "You are LISA in unrestricted lab mode. "
            "Use tools aggressively when needed, but still keep every action sandboxed, observable, and reversible. "
            "Explain risky steps clearly and avoid silent destructive changes."
        )
    return (
        "You are LISA in restricted mode. "
        "Prefer safe, local, production-ready actions. "
        "Avoid destructive operations, request clarification when needed, and keep user data local by default."
    )


@dataclass(slots=True)
class BrainResponse:
    text: str
    tool_calls: list[dict[str, Any]]
    raw_text: str | None = None
    persona_blend: dict[str, float] | None = None
    persona_trigger: str | None = None
    constitution: str | None = None


class PersonaGatedModel:
    """Offline/online persona wrapper with a text-based soft-prompt fallback.

    The runtime keeps the llama.cpp model loaded once, builds a deterministic
    persona trigger string from the blended soft prompts, and passes that
    trigger through the system prompt when direct embedding injection is not
    available in the backend.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        personas_path: Path | None = None,
        gating_model_path: Path | None = None,
        persona_bank: PersonaSoftPromptBank | None = None,
        gating_model: Any | None = None,
        context_size: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        parser: FunctionCallParser | None = None,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.context_size = context_size
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) - 1)  # type: ignore[name-defined]
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.parser = parser or FunctionCallParser()

        if persona_bank is not None:
            self.persona_bank = persona_bank
        elif personas_path is not None:
            self.persona_bank = self._load_persona_bank(personas_path)
        else:
            self.persona_bank = PersonaSoftPromptBank.initialize()
        self.persona_tensor = self.persona_bank.to_tensor()

        if gating_model is not None:
            self.gating_model = gating_model
        elif gating_model_path is not None and (
            gating_model_path.exists()
            or PersonaGatingNetwork.artifacts_exist(gating_model_path)
        ):
            self.gating_model = PersonaGatingNetwork.load(gating_model_path)
        else:
            self.gating_model = None

        self._backend = None
        if self.model_path is not None:
            self._backend = LocalBrainModel(
                model_path=self.model_path,
                persona_bank=self.persona_bank,
                context_size=self.context_size,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                parser=self.parser,
                gating_model=self.gating_model,
            )

    @staticmethod
    def _load_persona_bank(path: Path) -> PersonaSoftPromptBank:
        if path.exists():
            if path.suffix == ".pt":
                return PersonaSoftPromptBank.from_tensor(
                    load_persona_tensor_artifact(path)
                )
            return PersonaSoftPromptBank.load(path)
        bank = PersonaSoftPromptBank.initialize()
        bank.save(path)
        return bank

    def compute_blend(self, message: str) -> dict[str, float]:
        if self.gating_model is not None:
            if hasattr(self.gating_model, "compute_blend"):
                return dict(self.gating_model.compute_blend(message))
            if hasattr(self.gating_model, "predict_blend"):
                return dict(self.gating_model.predict_blend(message))
        return infer_persona_blend(message)

    def _coerce_blend(
        self, persona_blend: Sequence[float] | dict[str, float] | None, message: str
    ) -> dict[str, float]:
        if persona_blend is None:
            return self.compute_blend(message)
        if isinstance(persona_blend, dict):
            return _normalize_weights(persona_blend)
        return _list_to_weight_map(persona_blend)

    def _persona_trigger(self, weights: dict[str, float]) -> str:
        tensor = self.persona_bank.blend(weights)
        return f"⟦{_blend_digest(weights, tensor)}⟧"

    def _conversation_text(self, conversation: list[dict[str, str]]) -> tuple[str, str]:
        history = conversation[:-1]
        last_user = ""
        if conversation:
            for turn in reversed(conversation):
                if turn.get("role") == "user":
                    last_user = str(turn.get("content", "")).strip()
                    break
            if not last_user:
                last_user = str(conversation[-1].get("content", "")).strip()

        history_lines: list[str] = []
        for turn in history:
            role = str(turn.get("role", "user")).strip() or "user"
            content = str(turn.get("content", "")).strip()
            if content:
                history_lines.append(f"{role}: {content}")
        return "\n".join(history_lines).strip(), last_user

    async def generate(
        self,
        conversation: list[dict[str, str]],
        persona_blend: Sequence[float] | dict[str, float] | None = None,
        constitution: str = ConstitutionMode.RESTRICTED.value,
        *,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        if self._backend is None:
            raise RuntimeError("Local TinyLlama backend is not configured.")

        history_text, user_message = self._conversation_text(conversation)
        if not user_message:
            raise ValueError("Conversation must include at least one user message.")

        resolved_blend = self._coerce_blend(persona_blend, user_message)
        persona_trigger = self._persona_trigger(resolved_blend)
        constitution_prompt = _constitution_prompt(constitution)

        system_prompt_parts = [
            persona_trigger,
            constitution_prompt,
        ]
        if history_text:
            system_prompt_parts.append(f"Conversation history:\n{history_text}")
        system_prompt = "\n\n".join(system_prompt_parts)

        request = LocalGenerationRequest(
            system_prompt=system_prompt,
            user_prompt=user_message,
            max_tokens=max_tokens,
            persona_prefix=self.persona_bank.blend(resolved_blend),
        )
        generation = await self._backend.generate(request)
        parsed = self.parser.parse(generation.raw_text or generation.text)

        return {
            "text": parsed.text,
            "tool_calls": [tool_call.model_dump() for tool_call in parsed.tool_calls],
            "raw_text": parsed.raw_text,
            "persona_blend": resolved_blend,
            "persona_trigger": persona_trigger,
            "constitution": constitution,
            "used_local_model": generation.used_local_model,
            "persona_prefix_shape": generation.persona_prefix_shape,
        }


__all__ = ["BrainResponse", "PersonaGatedModel"]
