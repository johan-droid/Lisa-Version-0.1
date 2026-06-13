from __future__ import annotations

import asyncio
from pathlib import Path

from lisa.local_inference import PersonaGatedModel
from lisa.soft_prompts import PersonaSoftPromptBank


def test_persona_soft_prompt_bank_saves_and_loads_per_persona_npy(
    tmp_path: Path,
) -> None:
    bank = PersonaSoftPromptBank.initialize(tokens=4, dims=8, seed=7)
    directory = tmp_path / "persona_vectors"

    bank.save(directory)

    assert (directory / "metadata.json").exists()
    assert (directory / "architect.npy").exists()
    assert (directory / "oracle.npy").exists()

    loaded = PersonaSoftPromptBank.load(directory)
    assert loaded.summary()["architect"]["shape"] == [4, 8]
    assert loaded.summary()["guardian"]["dtype"] == "float32"


def test_persona_gated_model_generate_accepts_history_and_blend(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeLlama:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create_chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Here is the helper.\n"
                                '<tool_call>{"name":"search_notepad","arguments":{"query":"overflow"}}</tool_call>\n'
                                "Done."
                            )
                        }
                    }
                ]
            }

    fake_llama = FakeLlama()
    monkeypatch.setattr(
        PersonaGatedModel, "_load_model", lambda self, model_path: fake_llama
    )

    bank = PersonaSoftPromptBank.initialize(tokens=4, dims=8, seed=11)
    model = PersonaGatedModel(model_path=tmp_path / "tinyllama.gguf", persona_bank=bank)

    parsed = asyncio.run(
        model.generate(
            [
                {"role": "system", "content": "You are LISA."},
                {"role": "assistant", "content": "Ready."},
            ],
            "Write a Python function to add two numbers and check for overflows",
            {"architect": 0.8, "oracle": 0.2},
            max_tokens=128,
        )
    )

    assert parsed.persona_prefix_shape == (4, 8)
    assert parsed.tool_calls and parsed.tool_calls[0].name == "search_notepad"
    assert "Here is the helper." in parsed.text
    assert "Done." in parsed.text
    assert fake_llama.calls[0]["messages"][0]["content"].startswith(
        "You are LISA, a compact developer agent."
    )
