from __future__ import annotations

import csv
import json
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lisa.personas import Persona


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


@dataclass(slots=True)
class PersonaSklearnGatingBundle:
    """Compact TF-IDF + MLP gating model stored in one pickle file.

    The runtime can load this bundle directly or convert it back into the
    lightweight in-repo gating network interface.
    """

    vectorizer: Any
    classifier: Any
    personas: tuple[str, ...]
    trained_at: str | None = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.to_state(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "PersonaSklearnGatingBundle":
        path = Path(path)
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if isinstance(state, cls):
            return state
        return cls.from_state(state)

    def to_state(self) -> dict[str, Any]:
        return {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "personas": self.personas,
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "PersonaSklearnGatingBundle":
        return cls(
            vectorizer=state["vectorizer"],
            classifier=state["classifier"],
            personas=tuple(state.get("personas") or tuple(persona.value for persona in Persona)),
            trained_at=state.get("trained_at"),
        )

    def predict_proba(self, text: str) -> dict[str, float]:
        features = self.vectorizer.transform([text])
        if hasattr(features, "toarray"):
            features = features.toarray()
        probabilities = self.classifier.predict_proba(features)[0]
        class_names = getattr(self.classifier, "classes_", self.personas)
        return {
            str(persona): float(probability)
            for persona, probability in zip(class_names, probabilities, strict=True)
        }

    def predict_blend(self, text: str) -> dict[str, float]:
        probabilities = self.predict_proba(text)
        total = sum(probabilities.values()) or 1.0
        normalized = {name: round(value / total, 3) for name, value in probabilities.items()}
        remainder = round(1.0 - sum(normalized.values()), 3)
        first_key = self.personas[0]
        normalized[first_key] = round(normalized[first_key] + remainder, 3)
        return normalized

    def compute_blend(self, text: str) -> dict[str, float]:
        return self.predict_blend(text)

    def metadata(self) -> dict[str, Any]:
        hidden_sizes = getattr(self.classifier, "hidden_layer_sizes", None)
        if isinstance(hidden_sizes, tuple):
            hidden_size = list(hidden_sizes)
        elif hidden_sizes is None:
            hidden_size = []
        else:
            hidden_size = [int(hidden_sizes)]
        return {
            "format": "sklearn",
            "personas": list(self.personas),
            "feature_count": len(getattr(self.vectorizer, "vocabulary_", {})),
            "hidden_size": hidden_size,
            "trained_at": self.trained_at,
        }


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
        encoder = TfIdfEncoder(max_features=max_features).fit(example.text for example in examples)
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
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if isinstance(state, PersonaSklearnGatingBundle):
            return state
        if isinstance(state, dict) and "vectorizer" in state and "classifier" in state:
            return PersonaSklearnGatingBundle.from_state(state)
        return cls.from_state(state)

    @classmethod
    def load_or_initialize(cls, path: Path) -> "PersonaGatingNetwork":
        if path.exists():
            return cls.load(path)
        network = cls.initialize()
        network.save(path)
        return network

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.to_state(), handle, protocol=pickle.HIGHEST_PROTOCOL)

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
            self.W1 = rng.normal(0.0, 0.05, size=(feature_count, self.hidden_size)).astype(np.float32)

        target_matrix = np.vstack([self._example_target(example) for example in examples])
        feature_matrix = np.vstack([self.encoder.transform(example.text) for example in examples])
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
        normalized = {name: round(value / total, 3) for name, value in probabilities.items()}
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
            target = np.array([float(example.target.get(persona, 0.0)) for persona in self.personas], dtype=np.float32)
            total = float(target.sum()) or 1.0
            return target / total

        target = np.zeros(len(self.personas), dtype=np.float32)
        target[self.personas.index(example.target)] = 1.0
        return target


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
        primary_template = persona_templates[primary][index % len(persona_templates[primary])]
        secondary_template = persona_templates[secondary][(index + 2) % len(persona_templates[secondary])]
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
                target={Persona.GUARDIAN.value: 0.45, Persona.EVOLUTION_ENGINE.value: 0.55},
            ),
            PersonaTrainingExample(
                text="coordinate the handoff while designing the plan",
                target={Persona.DISTRIBUTED_MIND.value: 0.55, Persona.ARCHITECT.value: 0.45},
            ),
        ]
    )
    return examples
