from __future__ import annotations

from conductor.conductor import BrainTask, ConductorJob


def test_conductor_jobs_sort_higher_priority_first() -> None:
    high = ConductorJob(priority=10, sequence=1, kind="message", payload={})
    low = ConductorJob(priority=1, sequence=2, kind="message", payload={})

    assert high < low


def test_phase2_brain_task_dataclass_exposes_requested_fields() -> None:
    task = BrainTask()

    assert task.id
    assert task.conversation == []
    assert task.persona_blend is None
    assert task.constitution == "restricted"
    assert task.priority == 0
    assert task.callback_future is None
