from __future__ import annotations

import asyncio
from pathlib import Path

from brain.core import PersonaGatedModel as Phase1Brain
from brain.parser import FunctionCallParser
from brain.training import train_gating_bundle, write_persona_training_csv
from lisa.gating import PersonaGatingNetwork
from lisa.soft_prompts import PersonaSoftPromptBank


def test_persona_tensor_pt_round_trip(tmp_path: Path) -> None:
    bank = PersonaSoftPromptBank.initialize(tokens=6, dims=12, seed=9)
    path = tmp_path / "personas.pt"

    bank.save(path)
    loaded = PersonaSoftPromptBank.load(path)

    assert path.exists()
    assert loaded.to_tensor().shape == (5, 6, 12)
    assert loaded.summary()["architect"]["shape"] == [6, 12]


def test_phase1_parser_handles_tool_tag_and_json_list() -> None:
    parser = FunctionCallParser()
    parsed = parser.parse(
        "Planning.\n"
        '<tool_call>{"name":"file_write","arguments":{"path":"a.txt","content":"ok"}}</tool_call>\n'
        '```json\n[{"name":"dashboard_update","arguments":{"metric":"tokens","value":"12"}}]\n```'
    )

    assert "Planning." in parsed.text
    assert set([call.name for call in parsed.tool_calls]) == {
        "file_write",
        "dashboard_update",
    }
    file_call = next(c for c in parsed.tool_calls if c.name == "file_write")
    assert file_call.arguments["path"] == "a.txt"


def test_phase1_gating_bundle_is_loadable_by_runtime(tmp_path: Path) -> None:
    csv_path = tmp_path / "persona_training.csv"
    pkl_path = tmp_path / "gating_model.pkl"

    write_persona_training_csv(csv_path, count=24, seed=5)
    bundle = train_gating_bundle(
        csv_path, pkl_path, max_features=64, hidden_layer_size=8, seed=5
    )

    loaded = PersonaGatingNetwork.load(pkl_path)
    blend = loaded.compute_blend("build a secure api endpoint and fix the bug")

    assert pkl_path.with_suffix(".json").exists()
    assert pkl_path.with_suffix(".npz").exists()
    assert pkl_path.with_suffix(".sig").exists()
    assert bundle.metadata()["format"] == "sklearn"
    assert abs(sum(blend.values()) - 1.0) < 0.02


def test_phase1_brain_wrapper_uses_trigger_and_parser(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeLlama:
        def create_chat_completion(self, **kwargs):
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

    class FakeBlendModel:
        def compute_blend(self, message: str) -> dict[str, float]:
            return {
                "architect": 0.8,
                "oracle": 0.2,
                "guardian": 0.0,
                "evolution_engine": 0.0,
                "distributed_mind": 0.0,
            }

    monkeypatch.setattr(
        "lisa.local_inference.PersonaGatedModel._load_model",
        lambda self, model_path: FakeLlama(),
    )

    bank = PersonaSoftPromptBank.initialize(tokens=4, dims=8, seed=11)
    model = Phase1Brain(
        model_path=tmp_path / "tinyllama.gguf",
        persona_bank=bank,
        gating_model=FakeBlendModel(),
    )

    result = asyncio.run(
        model.generate(
            [
                {"role": "system", "content": "You are LISA."},
                {"role": "assistant", "content": "Ready."},
                {
                    "role": "user",
                    "content": "Write a Python function to add two numbers and check for overflows",
                },
            ],
            constitution="restricted",
            max_tokens=64,
        )
    )

    assert result["persona_blend"]["architect"] == 0.8
    assert result["persona_trigger"].startswith("⟦")
    assert result["tool_calls"] and result["tool_calls"][0]["name"] == "search_notepad"
    assert "Here is the helper." in result["text"]
