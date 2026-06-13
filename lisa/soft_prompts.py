from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lisa.personas import Persona

DEFAULT_PERSONA_TOKENS = 200
DEFAULT_EMBEDDING_DIMS = 768
PERSONA_ORDER: tuple[str, ...] = tuple(persona.value for persona in Persona)


@dataclass(slots=True)
class PersonaSoftPrompt:
    name: str
    vectors: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.vectors.shape)  # type: ignore[return-value]


class PersonaSoftPromptBank:
    def __init__(
        self,
        prompts: dict[str, PersonaSoftPrompt],
        tokens: int = DEFAULT_PERSONA_TOKENS,
        dims: int = DEFAULT_EMBEDDING_DIMS,
    ):
        self._prompts = prompts
        self.tokens = tokens
        self.dims = dims

    @classmethod
    def initialize(
        cls,
        tokens: int = DEFAULT_PERSONA_TOKENS,
        dims: int = DEFAULT_EMBEDDING_DIMS,
        seed: int = 42,
    ) -> "PersonaSoftPromptBank":
        rng = np.random.default_rng(seed)
        prompts: dict[str, PersonaSoftPrompt] = {}
        for persona in Persona:
            vectors = rng.normal(loc=0.0, scale=0.02, size=(tokens, dims)).astype(
                np.float32
            )
            prompts[persona.value] = PersonaSoftPrompt(
                name=persona.value, vectors=vectors
            )
        return cls(prompts=prompts, tokens=tokens, dims=dims)

    @classmethod
    def load(cls, path: Path) -> "PersonaSoftPromptBank":
        path = Path(path)
        if path.suffix == ".pt":
            return cls.from_tensor(load_persona_tensor_artifact(path))
        if path.is_dir() or not path.suffix:
            return cls.load_directory(path)

        payload = np.load(path, allow_pickle=False)
        tokens = (
            int(payload["tokens"]) if "tokens" in payload else DEFAULT_PERSONA_TOKENS
        )
        dims = int(payload["dims"]) if "dims" in payload else DEFAULT_EMBEDDING_DIMS
        prompts: dict[str, PersonaSoftPrompt] = {}
        for persona in Persona:
            key = f"{persona.value}_vectors"
            if key not in payload:
                continue
            vectors = payload[key].astype(np.float32, copy=False)
            prompts[persona.value] = PersonaSoftPrompt(
                name=persona.value, vectors=vectors
            )
        return cls(prompts=prompts, tokens=tokens, dims=dims)

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.suffix == ".pt":
            save_persona_tensor_artifact(path, self.to_tensor())
            return
        if path.is_dir() or not path.suffix:
            self.save_directory(path)
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "tokens": np.array(self.tokens, dtype=np.int32),
            "dims": np.array(self.dims, dtype=np.int32),
        }
        for name, prompt in self._prompts.items():
            payload[f"{name}_vectors"] = prompt.vectors.astype(np.float32, copy=False)
        np.savez_compressed(path, **payload)

    def save_directory(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"tokens": self.tokens, "dims": self.dims}
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        for name, prompt in self._prompts.items():
            np.save(
                directory / f"{name}.npy", prompt.vectors.astype(np.float32, copy=False)
            )

    @classmethod
    def load_directory(cls, directory: Path) -> "PersonaSoftPromptBank":
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            tokens = int(metadata.get("tokens", DEFAULT_PERSONA_TOKENS))
            dims = int(metadata.get("dims", DEFAULT_EMBEDDING_DIMS))
        else:
            tokens = DEFAULT_PERSONA_TOKENS
            dims = DEFAULT_EMBEDDING_DIMS

        prompts: dict[str, PersonaSoftPrompt] = {}
        for persona in Persona:
            path = directory / f"{persona.value}.npy"
            if not path.exists():
                continue
            vectors = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            prompts[persona.value] = PersonaSoftPrompt(
                name=persona.value, vectors=vectors
            )
        if prompts:
            first = next(iter(prompts.values()))
            tokens, dims = first.vectors.shape
        return cls(prompts=prompts, tokens=tokens, dims=dims)

    def get(self, persona: str) -> PersonaSoftPrompt:
        try:
            return self._prompts[persona]
        except KeyError as exc:
            raise KeyError(f"Unknown persona soft prompt: {persona}") from exc

    def blend(self, weights: dict[str, float]) -> np.ndarray:
        blended = np.zeros((self.tokens, self.dims), dtype=np.float32)
        total = 0.0
        for persona, weight in weights.items():
            prompt = self._prompts.get(persona)
            if prompt is None:
                continue
            blended += prompt.vectors * float(weight)
            total += float(weight)

        if total <= 0.0:
            return blended
        return blended / total

    def to_tensor(self) -> np.ndarray:
        tensor = np.zeros(
            (len(PERSONA_ORDER), self.tokens, self.dims), dtype=np.float32
        )
        for index, persona in enumerate(PERSONA_ORDER):
            prompt = self._prompts.get(persona)
            if prompt is None:
                continue
            tensor[index] = prompt.vectors.astype(np.float32, copy=False)
        return tensor

    @classmethod
    def from_tensor(
        cls,
        tensor: np.ndarray,
        *,
        persona_order: Sequence[str] = PERSONA_ORDER,
    ) -> "PersonaSoftPromptBank":
        array = np.asarray(tensor, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(
                "Persona tensor artifacts must have shape [personas, tokens, dims]."
            )
        if array.shape[0] != len(persona_order):
            raise ValueError(
                f"Persona tensor has {array.shape[0]} personas, expected {len(persona_order)}."
            )

        prompts: dict[str, PersonaSoftPrompt] = {}
        for index, persona in enumerate(persona_order):
            prompts[persona] = PersonaSoftPrompt(
                name=persona, vectors=array[index].astype(np.float32, copy=False)
            )
        tokens, dims = array.shape[1], array.shape[2]
        return cls(prompts=prompts, tokens=tokens, dims=dims)

    def summary(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "shape": list(prompt.shape),
                "dtype": str(prompt.vectors.dtype),
            }
            for name, prompt in self._prompts.items()
        }


@dataclass(slots=True)
class PersonaInjection:
    weights: dict[str, float]
    prefix_vectors: np.ndarray


def build_persona_injection(
    bank: PersonaSoftPromptBank,
    weights: dict[str, float],
) -> PersonaInjection:
    return PersonaInjection(weights=weights, prefix_vectors=bank.blend(weights))


def load_persona_tensor_artifact(path: Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as handle:
        tensor = np.load(handle, allow_pickle=False)
        return np.asarray(tensor, dtype=np.float32)


def save_persona_tensor_artifact(path: Path, tensor: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.asarray(tensor, dtype=np.float32), allow_pickle=False)
    return path
