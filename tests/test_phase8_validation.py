from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lisa.api import create_app
from lisa.config import Settings


def test_memory_stays_under_1gb_during_concurrent_chat_requests(tmp_path: Path) -> None:
    resource = pytest.importorskip("resource")

    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
        message_hub_enabled=False,
        evolution_enabled=False,
    )

    with TestClient(create_app(settings)) as client:
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [
                pool.submit(lambda: client.post("/chat", json={"message": "Hello, LISA?"}).status_code)
                for _ in range(10)
            ]
            statuses = [future.result(timeout=30) for future in futures]
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    rss_kib = after / 1024 if sys.platform == "darwin" else after
    assert all(status == 200 for status in statuses)
    assert rss_kib < 1_000_000
