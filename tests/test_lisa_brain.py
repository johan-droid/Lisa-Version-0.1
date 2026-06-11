from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import lisa_brain
from lisa.gating import PersonaGatingNetwork, train_persona_gating_network, write_synthetic_persona_training_csv
from lisa.local_inference import PersonaGatedModel


def test_synthetic_gating_data_and_training_produce_blend(tmp_path: Path) -> None:
    csv_path = tmp_path / "persona_training.csv"
    write_synthetic_persona_training_csv(csv_path, count=500, seed=13)

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 501

    model = train_persona_gating_network(csv_path, max_features=500, hidden_size=64, seed=13)
    blend = model.compute_blend("build a secure api endpoint and fix a bug")

    assert set(blend) == {
        "architect",
        "oracle",
        "guardian",
        "evolution_engine",
        "distributed_mind",
    }
    assert abs(sum(blend.values()) - 1.0) < 0.01


def test_lisa_brain_cli_prints_blend_and_tool_calls(tmp_path: Path, monkeypatch, capsys) -> None:
    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Here is the helper.\n"
                                "<tool_call>{\"name\":\"search_notepad\",\"arguments\":{\"query\":\"overflow\"}}</tool_call>\n"
                                "Done."
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(PersonaGatedModel, "_load_model", lambda self, model_path: FakeLlama())

    settings = SimpleNamespace(
        prompt="Write a Python function to add two numbers and check for overflows",
        model_path=str(tmp_path / "tinyllama.gguf"),
        persona_vectors_path=str(tmp_path / "persona_vectors.npz"),
        gating_model_path=str(tmp_path / "gating.pkl"),
        training_csv=str(tmp_path / "persona_training.csv"),
        train_gating=False,
        max_tokens=64,
        context_size=2048,
        threads=None,
        gpu_layers=0,
    )
    monkeypatch.setattr(lisa_brain, "parse_args", lambda: settings)
    monkeypatch.setattr(
        lisa_brain,
        "_load_or_train_gating_model",
        lambda csv_path, model_path, force_train: PersonaGatingNetwork.initialize(max_features=32, hidden_size=8, seed=4),
    )

    lisa_brain.main()
    output = capsys.readouterr().out

    assert "persona_blend" in output
    assert "search_notepad" in output
    assert "Here is the helper." in output
