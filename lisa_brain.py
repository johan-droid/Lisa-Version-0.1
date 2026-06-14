from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from brain.training import (
    load_phase1_examples,
    save_persona_tensor,
    train_gating_bundle,
    write_persona_training_csv,
)
from lisa.config import Settings
from lisa.gating import (
    PersonaGatingNetwork,
    train_persona_gating_network,
    write_synthetic_persona_training_csv,
)
from lisa.local_inference import PersonaGatedModel
from lisa.soft_prompts import PersonaSoftPromptBank

DEFAULT_CSV_PATH = Path("data") / "persona_training.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LISA brain heartbeat.")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="User prompt for the brain. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to the TinyLlama GGUF model. Defaults to LISA_LOCAL_MODEL_PATH.",
    )
    parser.add_argument(
        "--persona-vectors-path",
        default=None,
        help="Path to the persona vector bank file or directory.",
    )
    parser.add_argument(
        "--gating-model-path",
        default=None,
        help="Path to the serialized gating model pickle.",
    )
    parser.add_argument(
        "--training-csv",
        default=str(DEFAULT_CSV_PATH),
        help="Where synthetic gating training data should be stored.",
    )
    parser.add_argument(
        "--persona-training-csv",
        default=str(DEFAULT_CSV_PATH),
        help="Where the 5,000-example phase-1 persona training corpus should be stored.",
    )
    parser.add_argument(
        "--persona-tensor-path",
        default="data/personas.pt",
        help="Where the phase-1 persona tensor artifact should be written.",
    )
    parser.add_argument(
        "--train-gating",
        action="store_true",
        help="Regenerate synthetic gating data and retrain the MLP before running.",
    )
    parser.add_argument(
        "--train-personas",
        action="store_true",
        help="Generate and save the phase-1 persona tensor artifact before running.",
    )
    parser.add_argument(
        "--train-gating-sklearn",
        action="store_true",
        help="Train and save the compact sklearn TF-IDF + MLP gating bundle before running.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Override the model context window.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override llama.cpp CPU thread count.",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="GPU layers to offload. Default is 0 for CPU-only operation.",
    )
    return parser.parse_args()


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt.strip()
    return input("LISA> ").strip()


def _build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if args.model_path:
        settings.local_model_path = Path(args.model_path).expanduser().resolve()
    elif settings.local_model_path is None:
        env_model_path = os.environ.get("LISA_LOCAL_MODEL_PATH")
        if env_model_path:
            settings.local_model_path = Path(env_model_path).expanduser().resolve()

    if args.persona_vectors_path:
        settings.persona_vectors_path = (
            Path(args.persona_vectors_path).expanduser().resolve()
        )
    if args.gating_model_path:
        settings.gating_model_path = Path(args.gating_model_path).expanduser().resolve()
    if args.context_size is not None:
        settings.local_model_context_size = int(args.context_size)
    if args.threads is not None:
        settings.local_model_n_threads = int(args.threads)
    settings.local_model_n_gpu_layers = int(args.gpu_layers)
    return settings


def _load_persona_bank(path: Path) -> PersonaSoftPromptBank:
    if path.exists():
        return PersonaSoftPromptBank.load(path)
    bank = PersonaSoftPromptBank.initialize()
    bank.save(path)
    return bank


def _load_or_train_gating_model(
    csv_path: Path, model_path: Path, force_train: bool
) -> PersonaGatingNetwork:
    if force_train or not csv_path.exists():
        write_synthetic_persona_training_csv(csv_path, count=500)
    if force_train or not (
        model_path.exists() or PersonaGatingNetwork.artifacts_exist(model_path)
    ):
        gating = train_persona_gating_network(
            csv_path,
            max_features=500,
            hidden_size=64,
        )
        gating.save(model_path)
        return gating
    return PersonaGatingNetwork.load(model_path)


def _train_phase1_persona_tensor(csv_path: Path, output_path: Path) -> Path:
    if not csv_path.exists():
        write_persona_training_csv(csv_path, count=5000)
    examples = load_phase1_examples(csv_path)
    return save_persona_tensor(examples, output_path)


def _train_phase1_gating_bundle(csv_path: Path, output_path: Path) -> Path:
    if not csv_path.exists():
        write_persona_training_csv(csv_path, count=5000)
    train_gating_bundle(csv_path, output_path, max_features=2000, hidden_layer_size=16)
    return output_path


async def run_prompt(
    brain: PersonaGatedModel,
    prompt: str,
    max_tokens: int,
) -> Any:
    blend = brain.compute_blend(prompt)
    return await brain.generate([], prompt, blend, max_tokens=max_tokens)


def build_brain(
    settings: Settings, gating_model: PersonaGatingNetwork
) -> PersonaGatedModel:
    if settings.local_model_path is None:
        raise RuntimeError(
            "No local model path configured. Provide --model-path or LISA_LOCAL_MODEL_PATH."
        )

    persona_bank = _load_persona_bank(settings.persona_vectors_path)
    return PersonaGatedModel(
        model_path=settings.local_model_path,
        persona_bank=persona_bank,
        context_size=settings.local_model_context_size,
        n_threads=settings.local_model_n_threads,
        n_gpu_layers=settings.local_model_n_gpu_layers,
        gating_model=gating_model,
    )


def main() -> None:
    args = parse_args()
    settings = _build_settings(args)
    prompt = _resolve_prompt(args)

    if getattr(args, "train_personas", False):
        persona_csv = (
            Path(getattr(args, "persona_training_csv", DEFAULT_CSV_PATH))
            .expanduser()
            .resolve()
        )
        persona_tensor_path = (
            Path(getattr(args, "persona_tensor_path", "data/personas.pt"))
            .expanduser()
            .resolve()
        )
        _train_phase1_persona_tensor(persona_csv, persona_tensor_path)

    if getattr(args, "train_gating_sklearn", False):
        persona_csv = (
            Path(getattr(args, "persona_training_csv", DEFAULT_CSV_PATH))
            .expanduser()
            .resolve()
        )
        _train_phase1_gating_bundle(persona_csv, settings.gating_model_path)

    gating_model = _load_or_train_gating_model(
        Path(args.training_csv).expanduser().resolve(),
        settings.gating_model_path,
        force_train=args.train_gating,
    )

    brain = build_brain(settings, gating_model)
    generation = asyncio.run(run_prompt(brain, prompt, args.max_tokens))

    print(json.dumps({"persona_blend": brain.compute_blend(prompt)}, indent=2))
    print(generation.text)
    if generation.tool_calls:
        print("\nTool calls:")
        for tool_call in generation.tool_calls:
            print(json.dumps(tool_call.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
