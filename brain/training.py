from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

from lisa.gating import (
    PersonaSklearnGatingBundle,
    generate_synthetic_persona_training_examples,
    load_persona_training_examples,
    tokenize,
)
from lisa.personas import PERSONA_KEYWORDS, Persona
from lisa.soft_prompts import save_persona_tensor_artifact


def write_persona_training_csv(path: Path, count: int = 5000, seed: int = 42) -> Path:
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


def _token_embedding(token: str, dims: int, seed: int) -> np.ndarray:
    digest = sha256(f"{seed}:{token}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vector = rng.normal(0.0, 1.0, size=dims).astype(np.float32)
    norm = float(np.linalg.norm(vector)) or 1.0
    return vector / norm


def _example_vector(text: str, dims: int, seed: int) -> np.ndarray:
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(dims, dtype=np.float32)
    vectors = np.stack(
        [_token_embedding(token, dims, seed) for token in tokens], axis=0
    )
    return vectors.mean(axis=0).astype(np.float32)


def train_persona_tensor(
    examples: list[Any],
    *,
    tokens: int = 200,
    dims: int = 768,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    persona_vectors: dict[str, list[np.ndarray]] = {
        persona.value: [] for persona in Persona
    }
    persona_bias: dict[str, np.ndarray] = {}

    for persona, keywords in PERSONA_KEYWORDS.items():
        anchors = [_token_embedding(keyword, dims, seed) for keyword in keywords]
        persona_bias[persona.value] = np.mean(anchors, axis=0).astype(np.float32)

    for example in examples:
        target = example.target
        vector = _example_vector(example.text, dims, seed)
        if isinstance(target, dict):
            for name, weight in target.items():
                persona_vectors.setdefault(name, []).extend(
                    [vector * float(weight)] * max(1, int(round(weight * 4)))
                )
        else:
            persona_vectors.setdefault(str(target), []).append(vector)

    tensor = np.zeros((len(Persona), tokens, dims), dtype=np.float32)
    positions = np.linspace(0.0, 1.0, tokens, dtype=np.float32)
    harmonic = np.sin(np.linspace(0.0, np.pi * 2.0, dims, dtype=np.float32))

    for persona_index, persona in enumerate(Persona):
        samples = persona_vectors.get(persona.value) or []
        if samples:
            centroid = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
        else:
            centroid = np.zeros(dims, dtype=np.float32)

        base = (0.7 * centroid) + (0.3 * persona_bias[persona.value])
        for token_index, position in enumerate(positions):
            slot_noise = rng.normal(0.0, 0.01, size=dims).astype(np.float32)
            slot_wave = harmonic * (0.02 + 0.01 * position)
            token_vector = base + slot_noise + slot_wave
            tensor[persona_index, token_index] = token_vector.astype(np.float32)

    return tensor


def save_persona_tensor(
    examples: list[Any],
    output_path: Path,
    *,
    tokens: int = 200,
    dims: int = 768,
    seed: int = 42,
) -> Path:
    tensor = train_persona_tensor(examples, tokens=tokens, dims=dims, seed=seed)
    save_persona_tensor_artifact(output_path, tensor)
    return output_path


def load_phase1_examples(csv_path: Path) -> list[Any]:
    return load_persona_training_examples(csv_path)


def _expand_examples_for_classifier(examples: list[Any]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for example in examples:
        target = example.target
        if isinstance(target, dict):
            weights = {
                name: float(value)
                for name, value in target.items()
                if float(value) > 0.0
            }
            total = sum(weights.values()) or 1.0
            for persona, weight in weights.items():
                repeats = max(1, int(round((weight / total) * 6)))
                texts.extend([example.text] * repeats)
                labels.extend([persona] * repeats)
        else:
            texts.append(example.text)
            labels.append(str(target))
    return texts, labels


def train_gating_bundle(
    csv_path: Path,
    output_path: Path,
    *,
    max_features: int = 2000,
    hidden_layer_size: int = 16,
    seed: int = 42,
) -> PersonaSklearnGatingBundle:
    examples = load_phase1_examples(csv_path)
    texts, labels = _expand_examples_for_classifier(examples)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),
        lowercase=True,
        tokenizer=tokenize,
        token_pattern=None,
    )
    features = vectorizer.fit_transform(texts)
    classifier = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        max_iter=300,
        random_state=seed,
        early_stopping=True,
    )
    classifier.fit(features, labels)
    bundle = PersonaSklearnGatingBundle.from_legacy_objects(
        vectorizer,
        classifier,
        tuple(persona.value for persona in Persona),
        datetime.now(timezone.utc).isoformat(),
    )
    bundle.save(output_path)
    return bundle
