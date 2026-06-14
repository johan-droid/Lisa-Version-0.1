from __future__ import annotations

import csv
import io
import json
import os
import pickle
import re
from types import SimpleNamespace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import hmac
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lisa.personas import Persona
from utils.snapshot import get_hmac_key

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _artifact_paths(path: Path) -> tuple[Path, Path, Path]:
    base = Path(path)
    metadata_path = base.with_suffix(".json")
    weights_path = base.with_suffix(".npz")
    signature_path = base.with_suffix(".sig")
    return metadata_path, weights_path, signature_path


def _signing_context(path: Path) -> Any:
    return SimpleNamespace(
        workspace_root=path.parent.parent,
        bot_security_key=os.environ.get("LISA_BOT_SECURITY_KEY"),
    )


def _metadata_bytes(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _weights_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _artifact_signature(
    path: Path, metadata: dict[str, Any], weights_blob: bytes
) -> str:
    key = get_hmac_key(_signing_context(path))
    payload = _metadata_bytes(metadata) + b"\n" + weights_blob
    return hmac.new(key, payload, sha256).hexdigest()


def _write_artifacts(
    path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    metadata_path, weights_path, signature_path = _artifact_paths(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    safe_arrays = {name: np.asarray(array) for name, array in arrays.items()}
    weights_blob = _weights_bytes(safe_arrays)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    weights_path.write_bytes(weights_blob)
    signature_path.write_text(
        _artifact_signature(path, metadata, weights_blob),
        encoding="utf-8",
    )


def _load_artifacts(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path, weights_path, signature_path = _artifact_paths(path)
    if not (
        metadata_path.exists() and weights_path.exists() and signature_path.exists()
    ):
        raise FileNotFoundError(f"Signed gating artifacts are missing for {path}.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    weights_blob = weights_path.read_bytes()
    signature = signature_path.read_text(encoding="utf-8").strip()
    expected_signature = _artifact_signature(path, metadata, weights_blob)
    if not signature or not hmac.compare_digest(signature, expected_signature):
        raise ValueError(f"Gating model signature check failed for {path}.")

    with np.load(io.BytesIO(weights_blob), allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    return metadata, arrays


@dataclass(slots=True)
class PersonaSklearnGatingBundle:
    """Compact TF-IDF + single-hidden-layer MLP gating model stored safely."""

    vocabulary: dict[str, int]
    idf: np.ndarray
    classes: tuple[str, ...]
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
    personas: tuple[str, ...]
    trained_at: str | None = None

    def save(self, path: Path) -> None:
        path = Path(path)
        _write_artifacts(path, self.to_metadata(), self.to_arrays())

    @classmethod
    def load(cls, path: Path) -> "PersonaSklearnGatingBundle":
        metadata, arrays = _load_artifacts(Path(path))
        if metadata.get("format") != "sklearn_bundle_v2":
            raise ValueError(f"Unsupported sklearn gating metadata at {path}.")
        return cls.from_serialized(metadata, arrays)

    def to_state(self) -> dict[str, Any]:
        return {
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "classes": self.classes,
            "hidden_weights": self.hidden_weights,
            "hidden_bias": self.hidden_bias,
            "output_weights": self.output_weights,
            "output_bias": self.output_bias,
            "personas": self.personas,
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "PersonaSklearnGatingBundle":
        return cls(
            vocabulary=dict(state["vocabulary"]),
            idf=np.asarray(state["idf"], dtype=np.float32),
            classes=tuple(state["classes"]),
            hidden_weights=np.asarray(state["hidden_weights"], dtype=np.float32),
            hidden_bias=np.asarray(state["hidden_bias"], dtype=np.float32),
            output_weights=np.asarray(state["output_weights"], dtype=np.float32),
            output_bias=np.asarray(state["output_bias"], dtype=np.float32),
            personas=tuple(
                state.get("personas") or tuple(persona.value for persona in Persona)
            ),
            trained_at=state.get("trained_at"),
        )

    @classmethod
    def from_serialized(
        cls, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> "PersonaSklearnGatingBundle":
        return cls(
            vocabulary={
                str(key): int(value) for key, value in metadata["vocabulary"].items()
            },
            idf=np.asarray(arrays["idf"], dtype=np.float32),
            classes=tuple(str(item) for item in metadata["classes"]),
            hidden_weights=np.asarray(arrays["hidden_weights"], dtype=np.float32),
            hidden_bias=np.asarray(arrays["hidden_bias"], dtype=np.float32),
            output_weights=np.asarray(arrays["output_weights"], dtype=np.float32),
            output_bias=np.asarray(arrays["output_bias"], dtype=np.float32),
            personas=tuple(str(item) for item in metadata["personas"]),
            trained_at=metadata.get("trained_at"),
        )

    def predict_proba(self, text: str) -> dict[str, float]:
        features = self._transform(text)
        hidden = np.tanh(features @ self.hidden_weights + self.hidden_bias)
        logits = hidden @ self.output_weights + self.output_bias
        probabilities = softmax(logits)
        return {
            str(persona): float(probability)
            for persona, probability in zip(self.classes, probabilities, strict=True)
        }

    def predict_blend(self, text: str) -> dict[str, float]:
        probabilities = self.predict_proba(text)
        total = sum(probabilities.values()) or 1.0
        normalized = {
            name: round(value / total, 3) for name, value in probabilities.items()
        }
        remainder = round(1.0 - sum(normalized.values()), 3)
        first_key = self.personas[0]
        normalized[first_key] = round(normalized[first_key] + remainder, 3)
        return normalized

    def compute_blend(self, text: str) -> dict[str, float]:
        return self.predict_blend(text)

    def metadata(self) -> dict[str, Any]:
        return {
            "format": "sklearn",
            "personas": list(self.personas),
            "feature_count": len(self.vocabulary),
            "hidden_size": [int(self.hidden_weights.shape[1])],
            "trained_at": self.trained_at,
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "format": "sklearn_bundle_v2",
            "personas": list(self.personas),
            "classes": list(self.classes),
            "vocabulary": {key: int(value) for key, value in self.vocabulary.items()},
            "trained_at": self.trained_at,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "idf": np.asarray(self.idf, dtype=np.float32),
            "hidden_weights": np.asarray(self.hidden_weights, dtype=np.float32),
            "hidden_bias": np.asarray(self.hidden_bias, dtype=np.float32),
            "output_weights": np.asarray(self.output_weights, dtype=np.float32),
            "output_bias": np.asarray(self.output_bias, dtype=np.float32),
        }

    def _transform(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vector
        counts: dict[int, int] = {}
        for token in tokens:
            index = self.vocabulary.get(token)
            if index is None:
                continue
            counts[index] = counts.get(index, 0) + 1
        if not counts:
            return vector
        total = float(sum(counts.values()))
        for index, count in counts.items():
            vector[index] = (count / total) * self.idf[index]
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    @classmethod
    def from_legacy_objects(
        cls,
        vectorizer: Any,
        classifier: Any,
        personas: tuple[str, ...],
        trained_at: str | None,
    ) -> "PersonaSklearnGatingBundle":
        if (
            len(getattr(classifier, "coefs_", [])) != 2
            or len(getattr(classifier, "intercepts_", [])) != 2
        ):
            raise ValueError(
                "Only single-hidden-layer MLP gating bundles can be migrated safely."
            )
        activation = str(getattr(classifier, "activation", "tanh")).lower()
        if activation != "tanh":
            raise ValueError(f"Unsupported gating activation {activation!r}.")
        vocabulary = dict(getattr(vectorizer, "vocabulary_", {}) or {})
        idf = np.asarray(getattr(vectorizer, "idf_", []), dtype=np.float32)
        classes = tuple(str(item) for item in getattr(classifier, "classes_", personas))
        return cls(
            vocabulary=vocabulary,
            idf=idf,
            classes=classes,
            hidden_weights=np.asarray(classifier.coefs_[0], dtype=np.float32),
            hidden_bias=np.asarray(classifier.intercepts_[0], dtype=np.float32),
            output_weights=np.asarray(classifier.coefs_[1], dtype=np.float32),
            output_bias=np.asarray(classifier.intercepts_[1], dtype=np.float32),
            personas=personas,
            trained_at=trained_at,
        )


@dataclass(slots=True)
class TfIdfEncoder:
    max_features: int = 128
    vocabulary: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def fit(self, texts: Iterable[str]) -> "TfIdfEncoder":
        documents = [tokenize(text) for text in texts]
        doc_frequency: dict[str, int] = {}
        term_frequency: dict[str, int] = {}

        for tokens in documents:
            seen: set[str] = set()
            for token in tokens:
                term_frequency[token] = term_frequency.get(token, 0) + 1
                if token not in seen:
                    doc_frequency[token] = doc_frequency.get(token, 0) + 1
                    seen.add(token)

        ranked = sorted(term_frequency.items(), key=lambda item: (-item[1], item[0]))
        vocab_tokens = [token for token, _ in ranked[: self.max_features]]
        self.vocabulary = {token: index for index, token in enumerate(vocab_tokens)}

        doc_count = max(len(documents), 1)
        idf = np.zeros(len(self.vocabulary), dtype=np.float32)
        for token, index in self.vocabulary.items():
            df = doc_frequency.get(token, 0)
            idf[index] = np.log((1.0 + doc_count) / (1.0 + df)) + 1.0
        self.idf = idf
        return self

    def transform(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        tokens = tokenize(text)
        if not tokens or len(self.vocabulary) == 0:
            return vector

        counts: dict[int, int] = {}
        for token in tokens:
            index = self.vocabulary.get(token)
            if index is None:
                continue
            counts[index] = counts.get(index, 0) + 1

        if not counts:
            return vector

        total = float(sum(counts.values()))
        for index, count in counts.items():
            vector[index] = (count / total) * self.idf[index]

        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def to_state(self) -> dict[str, Any]:
        return {
            "max_features": self.max_features,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "TfIdfEncoder":
        return cls(
            max_features=int(state["max_features"]),
            vocabulary=dict(state["vocabulary"]),
            idf=np.asarray(state["idf"], dtype=np.float32),
        )


@dataclass(slots=True)
class PersonaGatingNetwork:
    encoder: TfIdfEncoder
    hidden_size: int
    personas: tuple[str, ...]
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    trained_at: str | None = None

    @classmethod
    def initialize(
        cls,
        max_features: int = 500,
        hidden_size: int = 64,
        seed: int = 42,
    ) -> "PersonaGatingNetwork":
        return cls.from_examples(
            seed_persona_training_examples(),
            max_features=max_features,
            hidden_size=hidden_size,
            seed=seed,
        )

    @classmethod
    def from_examples(
        cls,
        examples: list["PersonaTrainingExample"],
        *,
        max_features: int = 500,
        hidden_size: int = 64,
        seed: int = 42,
        epochs: int = 120,
        learning_rate: float = 0.1,
    ) -> "PersonaGatingNetwork":
        if not examples:
            raise ValueError("At least one training example is required.")

        rng = np.random.default_rng(seed)
        encoder = TfIdfEncoder(max_features=max_features).fit(
            example.text for example in examples
        )
        feature_count = max(len(encoder.vocabulary), 1)
        personas = tuple(persona.value for persona in Persona)
        W1 = rng.normal(0.0, 0.05, size=(feature_count, hidden_size)).astype(np.float32)
        b1 = np.zeros(hidden_size, dtype=np.float32)
        W2 = rng.normal(0.0, 0.05, size=(hidden_size, len(personas))).astype(np.float32)
        b2 = np.zeros(len(personas), dtype=np.float32)
        network = cls(
            encoder=encoder,
            hidden_size=hidden_size,
            personas=personas,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
        )
        network.fit(examples, epochs=epochs, learning_rate=learning_rate)
        return network

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        max_features: int = 500,
        hidden_size: int = 64,
        seed: int = 42,
        epochs: int = 120,
        learning_rate: float = 0.1,
    ) -> "PersonaGatingNetwork":
        examples = load_persona_training_examples(path)
        return cls.from_examples(
            examples,
            max_features=max_features,
            hidden_size=hidden_size,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
        )

    @classmethod
    def load(cls, path: Path) -> "PersonaGatingNetwork":
        path = Path(path)
        if cls.artifacts_exist(path):
            metadata, arrays = _load_artifacts(path)
            artifact_format = metadata.get("format")
            if artifact_format == "persona_network_v2":
                return cls.from_serialized(metadata, arrays)
            if artifact_format == "sklearn_bundle_v2":
                return PersonaSklearnGatingBundle.from_serialized(metadata, arrays)
            raise ValueError(f"Unsupported gating artifact format {artifact_format!r}.")

        if path.exists():
            migrated = cls._migrate_legacy_pickle(path)
            path.unlink(missing_ok=True)
            return migrated
        raise FileNotFoundError(path)

    @classmethod
    def load_or_initialize(cls, path: Path) -> "PersonaGatingNetwork":
        if cls.artifacts_exist(path) or path.exists():
            return cls.load(path)
        network = cls.initialize()
        network.save(path)
        return network

    def save(self, path: Path) -> None:
        path = Path(path)
        _write_artifacts(path, self.to_metadata(), self.to_arrays())

    def to_state(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder.to_state(),
            "hidden_size": self.hidden_size,
            "personas": self.personas,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "trained_at": self.trained_at,
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "format": "persona_network_v2",
            "encoder_max_features": int(self.encoder.max_features),
            "vocabulary": {
                key: int(value) for key, value in self.encoder.vocabulary.items()
            },
            "hidden_size": int(self.hidden_size),
            "personas": list(self.personas),
            "trained_at": self.trained_at,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "idf": np.asarray(self.encoder.idf, dtype=np.float32),
            "W1": np.asarray(self.W1, dtype=np.float32),
            "b1": np.asarray(self.b1, dtype=np.float32),
            "W2": np.asarray(self.W2, dtype=np.float32),
            "b2": np.asarray(self.b2, dtype=np.float32),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "PersonaGatingNetwork":
        return cls(
            encoder=TfIdfEncoder.from_state(state["encoder"]),
            hidden_size=int(state["hidden_size"]),
            personas=tuple(state["personas"]),
            W1=np.asarray(state["W1"], dtype=np.float32),
            b1=np.asarray(state["b1"], dtype=np.float32),
            W2=np.asarray(state["W2"], dtype=np.float32),
            b2=np.asarray(state["b2"], dtype=np.float32),
            trained_at=state.get("trained_at"),
        )

    @classmethod
    def from_serialized(
        cls, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> "PersonaGatingNetwork":
        encoder = TfIdfEncoder(
            max_features=int(metadata["encoder_max_features"]),
            vocabulary={
                str(key): int(value)
                for key, value in dict(metadata["vocabulary"]).items()
            },
            idf=np.asarray(arrays["idf"], dtype=np.float32),
        )
        return cls(
            encoder=encoder,
            hidden_size=int(metadata["hidden_size"]),
            personas=tuple(str(item) for item in metadata["personas"]),
            W1=np.asarray(arrays["W1"], dtype=np.float32),
            b1=np.asarray(arrays["b1"], dtype=np.float32),
            W2=np.asarray(arrays["W2"], dtype=np.float32),
            b2=np.asarray(arrays["b2"], dtype=np.float32),
            trained_at=metadata.get("trained_at"),
        )

    def fit(
        self,
        examples: list["PersonaTrainingExample"],
        epochs: int = 60,
        learning_rate: float = 0.05,
    ) -> None:
        texts = [example.text for example in examples]
        if texts:
            self.encoder.fit(texts)

        feature_count = max(len(self.encoder.vocabulary), 1)
        if self.W1.shape[0] != feature_count:
            rng = np.random.default_rng(42)
            self.W1 = rng.normal(
                0.0, 0.05, size=(feature_count, self.hidden_size)
            ).astype(np.float32)

        target_matrix = np.vstack(
            [self._example_target(example) for example in examples]
        )
        feature_matrix = np.vstack(
            [self.encoder.transform(example.text) for example in examples]
        )
        if feature_matrix.shape[1] == 0:
            feature_matrix = np.zeros((len(examples), feature_count), dtype=np.float32)

        for _ in range(epochs):
            hidden = np.tanh(feature_matrix @ self.W1 + self.b1)
            logits = hidden @ self.W2 + self.b2
            probabilities = np.vstack([softmax(logit) for logit in logits])
            grad_logits = (probabilities - target_matrix) / float(len(examples))
            grad_W2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = (grad_logits @ self.W2.T) * (1.0 - hidden**2)
            grad_W1 = feature_matrix.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            self.W2 -= learning_rate * grad_W2.astype(np.float32)
            self.b2 -= learning_rate * grad_b2.astype(np.float32)
            self.W1 -= learning_rate * grad_W1.astype(np.float32)
            self.b1 -= learning_rate * grad_b1.astype(np.float32)

        self.trained_at = datetime.now(timezone.utc).isoformat()

    def predict_proba(self, text: str) -> dict[str, float]:
        features = self.encoder.transform(text)
        if features.shape[0] == 0:
            features = np.zeros(self.W1.shape[0], dtype=np.float32)
        hidden = np.tanh(features @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        probabilities = softmax(logits)
        return {
            persona: float(probability)
            for persona, probability in zip(self.personas, probabilities, strict=True)
        }

    def predict_blend(self, text: str) -> dict[str, float]:
        probabilities = self.predict_proba(text)
        total = sum(probabilities.values()) or 1.0
        normalized = {
            name: round(value / total, 3) for name, value in probabilities.items()
        }
        remainder = round(1.0 - sum(normalized.values()), 3)
        first_key = self.personas[0]
        normalized[first_key] = round(normalized[first_key] + remainder, 3)
        return normalized

    def compute_blend(self, text: str) -> dict[str, float]:
        return self.predict_blend(text)

    def metadata(self) -> dict[str, Any]:
        return {
            "personas": list(self.personas),
            "hidden_size": self.hidden_size,
            "feature_count": len(self.encoder.vocabulary),
            "trained_at": self.trained_at,
        }

    def _example_target(self, example: "PersonaTrainingExample") -> np.ndarray:
        if isinstance(example.target, dict):
            target = np.array(
                [float(example.target.get(persona, 0.0)) for persona in self.personas],
                dtype=np.float32,
            )
            total = float(target.sum()) or 1.0
            return target / total

        target = np.zeros(len(self.personas), dtype=np.float32)
        target[self.personas.index(example.target)] = 1.0
        return target

    @classmethod
    def artifacts_exist(cls, path: Path) -> bool:
        metadata_path, weights_path, signature_path = _artifact_paths(Path(path))
        return (
            metadata_path.exists() and weights_path.exists() and signature_path.exists()
        )

    @classmethod
    def _migrate_legacy_pickle(cls, path: Path) -> "PersonaGatingNetwork":
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        if isinstance(state, PersonaSklearnGatingBundle):
            bundle = state
            bundle.save(path)
            return bundle
        if isinstance(state, dict) and "vectorizer" in state and "classifier" in state:
            bundle = PersonaSklearnGatingBundle.from_legacy_objects(
                state["vectorizer"],
                state["classifier"],
                tuple(
                    state.get("personas") or tuple(persona.value for persona in Persona)
                ),
                state.get("trained_at"),
            )
            bundle.save(path)
            return bundle
        if isinstance(state, dict):
            network = cls.from_state(state)
            network.save(path)
            return network
        raise ValueError(f"Unsupported legacy gating pickle at {path}.")


@dataclass(slots=True)
class PersonaTrainingExample:
    text: str
    target: str | dict[str, float]


def generate_synthetic_persona_training_examples(
    count: int = 500,
    seed: int = 42,
) -> list[PersonaTrainingExample]:
    rng = np.random.default_rng(seed)
    examples: list[PersonaTrainingExample] = []
    personas = list(Persona)
    persona_templates: dict[Persona, list[str]] = {
        Persona.ARCHITECT: [
            "build a clean api endpoint",
            "design the module layout",
            "generate a production-ready helper",
            "implement the service with clear boundaries",
        ],
        Persona.ORACLE: [
            "audit the code for bugs",
            "secure the workflow and review the risk",
            "inspect the vulnerability and failure mode",
            "analyze the crash and find the bug",
        ],
        Persona.GUARDIAN: [
            "monitor health and recover safely",
            "backup the workspace and alert on issues",
            "guard uptime and inspect the logs",
            "stabilize the infra and handle failure",
        ],
        Persona.EVOLUTION_ENGINE: [
            "improve the skill and optimize the loop",
            "learn from failures and evolve the helper",
            "adapt the automation and refine the logic",
            "upgrade the workflow and test the change",
        ],
        Persona.DISTRIBUTED_MIND: [
            "coordinate the handoff and sync the team",
            "communicate the plan across agents",
            "orchestrate collaboration and share status",
            "align the execution and build consensus",
        ],
    }
    modifiers = [
        "Please",
        "Quickly",
        "Carefully",
        "Now",
        "Today",
        "Ideally",
    ]

    one_hot_target_count = max(0, count - max(5, count // 10))
    mixed_target_count = count - one_hot_target_count

    for index in range(one_hot_target_count):
        persona = personas[index % len(personas)]
        template = persona_templates[persona][index % len(persona_templates[persona])]
        modifier = modifiers[int(rng.integers(0, len(modifiers)))]
        text = f"{modifier} {template}"
        examples.append(PersonaTrainingExample(text=text, target=persona.value))

    for index in range(mixed_target_count):
        primary = personas[index % len(personas)]
        secondary = personas[(index + 1) % len(personas)]
        primary_template = persona_templates[primary][
            index % len(persona_templates[primary])
        ]
        secondary_template = persona_templates[secondary][
            (index + 2) % len(persona_templates[secondary])
        ]
        weight = 0.6 if index % 2 == 0 else 0.7
        text = f"{primary_template} and also {secondary_template}"
        examples.append(
            PersonaTrainingExample(
                text=text,
                target={
                    primary.value: weight,
                    secondary.value: round(1.0 - weight, 2),
                },
            )
        )

    rng.shuffle(examples)
    return examples[:count]


def write_synthetic_persona_training_csv(
    path: Path,
    count: int = 500,
    seed: int = 42,
) -> Path:
    examples = generate_synthetic_persona_training_examples(count=count, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "target"])
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "text": example.text,
                    "target": json.dumps(example.target, ensure_ascii=True),
                }
            )
    return path


def load_persona_training_examples(path: Path) -> list[PersonaTrainingExample]:
    examples: list[PersonaTrainingExample] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = str(row.get("text") or "").strip()
            target_value = str(row.get("target") or "").strip()
            if not text or not target_value:
                continue
            try:
                target = json.loads(target_value)
            except json.JSONDecodeError:
                target = target_value
            examples.append(PersonaTrainingExample(text=text, target=target))
    return examples


def train_persona_gating_network(
    csv_path: Path,
    *,
    max_features: int = 500,
    hidden_size: int = 64,
    seed: int = 42,
    epochs: int = 120,
    learning_rate: float = 0.1,
) -> PersonaGatingNetwork:
    return PersonaGatingNetwork.from_csv(
        csv_path,
        max_features=max_features,
        hidden_size=hidden_size,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
    )


def seed_persona_training_examples() -> list[PersonaTrainingExample]:
    examples: list[PersonaTrainingExample] = []
    templates = {
        Persona.ARCHITECT.value: [
            "design a new api endpoint and refactor the service",
            "implement the feature with a clean module boundary",
            "plan the system architecture and scaffold the package",
        ],
        Persona.ORACLE.value: [
            "audit the code for bugs and vulnerabilities",
            "review the security posture and analyze failures",
            "profile the hotspot and inspect the risky logic",
        ],
        Persona.GUARDIAN.value: [
            "monitor uptime and recover the service safely",
            "backup the data and inspect health signals",
            "alert on infra issues and harden reliability",
        ],
        Persona.EVOLUTION_ENGINE.value: [
            "improve the skill and optimize the implementation",
            "learn from failures and evolve the workflow",
            "adapt the codebase and refine the automation",
        ],
        Persona.DISTRIBUTED_MIND.value: [
            "coordinate the team handoff and synchronize tasks",
            "communicate status and build consensus across agents",
            "orchestrate the collaboration and sync the execution",
        ],
    }

    for persona, phrases in templates.items():
        for phrase in phrases:
            examples.append(PersonaTrainingExample(text=phrase, target=persona))

    examples.extend(
        [
            PersonaTrainingExample(
                text="build a secure api and audit it for vulnerabilities",
                target={Persona.ARCHITECT.value: 0.45, Persona.ORACLE.value: 0.55},
            ),
            PersonaTrainingExample(
                text="monitor the deployment, recover errors, and improve the workflow",
                target={
                    Persona.GUARDIAN.value: 0.45,
                    Persona.EVOLUTION_ENGINE.value: 0.55,
                },
            ),
            PersonaTrainingExample(
                text="coordinate the handoff while designing the plan",
                target={
                    Persona.DISTRIBUTED_MIND.value: 0.55,
                    Persona.ARCHITECT.value: 0.45,
                },
            ),
        ]
    )
    return examples
