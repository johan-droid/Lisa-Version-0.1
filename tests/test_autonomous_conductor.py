from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from lisa.config import Settings
from lisa.conductor import TaskConductor
from lisa.schemas import BrainTask, ChatResponse
from conductor.autonomous import SelfDirectedConductor


def test_self_directed_conductor_generates_and_submits_goal() -> None:
    # 1. Setup mocks
    settings = Settings(
        workspace_root=Path("."),
        db_path=Path("data/test.db"),
        skills_dir=Path("skills"),
        persona_vectors_path=Path("data/persona_vectors.npz"),
        gating_model_path=Path("data/gating.pkl"),
    )
    
    conductor_mock = MagicMock(spec=TaskConductor)
    conductor_mock.is_idle.return_value = True
    
    # Mock LLM Client response
    llm_client_mock = AsyncMock()
    fake_generation = MagicMock()
    fake_generation.text = """
    [
        {"task_name": "Proactive security audit", "estimated_value": 9.0, "estimated_urgency": 8.0, "estimated_risk": 2.0, "estimated_cost": 1.0},
        {"task_name": "Update readme", "estimated_value": 2.0, "estimated_urgency": 1.0, "estimated_risk": 0.0, "estimated_cost": 1.0}
    ]
    """
    llm_client_mock.generate_brain.return_value = fake_generation
    
    conductor_mock.tool_executor = MagicMock()
    conductor_mock.runtime = MagicMock()
    conductor_mock.runtime.llm_client = llm_client_mock
    
    notepad_mock = MagicMock()
    notepad_mock.latest_entries.return_value = []
    conductor_mock.notepad = notepad_mock
    
    writer_mock = AsyncMock()
    conductor_mock.runtime.notepad_writer = writer_mock
    
    # 2. Instantiate and run one-off generation
    sd_conductor = SelfDirectedConductor(settings, conductor_mock)
    
    async def run() -> None:
        await sd_conductor._generate_and_submit_goal()
        
    asyncio.run(run())
    
    # 3. Assertions
    conductor_mock.try_submit_message.assert_called_once()
    inbound_arg = conductor_mock.try_submit_message.call_args[0][0]
    assert inbound_arg.source == "autonomous"
    assert "Proactive security audit" in inbound_arg.text
    assert inbound_arg.metadata["utility_score"] > 5.0
    
    writer_mock.enqueue.assert_called_once()
    assert writer_mock.enqueue.call_args[1]["entry_type"] == "autonomous_goal_selection"
