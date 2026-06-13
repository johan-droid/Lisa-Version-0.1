from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from lisa.config import Settings
from lisa.schemas import InboundMessage
from lisa.conductor import TaskConductor

LOGGER = logging.getLogger("lisa.conductor.autonomous")


class SelfDirectedConductor:
    def __init__(self, settings: Settings, conductor: TaskConductor):
        self.settings = settings
        self.conductor = conductor
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="self-directed-conductor")
            LOGGER.info("SelfDirectedConductor background task started.")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with asyncio.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
            LOGGER.info("SelfDirectedConductor background task stopped.")

    async def _loop(self) -> None:
        await asyncio.sleep(10)
        while not self._stop.is_set():
            try:
                if self.conductor.is_idle():
                    LOGGER.info("System is idle. Generating autonomous candidate goals...")
                    await self._generate_and_submit_goal()
            except Exception as exc:
                LOGGER.exception("Error in SelfDirectedConductor loop: %s", exc)

            await asyncio.sleep(60)

    async def _generate_and_submit_goal(self) -> None:
        git_status = await self._get_git_status()
        system_health = self._get_system_health()
        recent_activity = await self._get_recent_notepad_activity()

        prompt = f"""You are the Distributed Mind persona. Current state:
- Git status: {git_status}
- Open issues: None
- Calendar: No upcoming events
- System health: {system_health}
- Recent user activity: {recent_activity}

Generate 3 candidate tasks I could perform autonomously. For each, provide:
- Task name
- Estimated value (1-10)
- Estimated urgency (1-10)
- Estimated risk (1-10)
- Estimated cost (API calls)
Format as a JSON array of objects.
"""
        try:
            llm_client = getattr(self.conductor.runtime, 'llm_client', getattr(getattr(self.conductor, 'tool_executor', None), 'llm_client', None))
            # We call chat/complete on llm_client
            chat_resp = await llm_client.generate_brain(
                system_prompt="You are a self-directed, autonomous software developer agent.",
                user_prompt=prompt,
                max_tokens=800,
                persona_weights={"Distributed Mind": 1.0},
            )
            response_text = chat_resp.text.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            candidates = json.loads(response_text)
            if not isinstance(candidates, list):
                LOGGER.warning("LLM response was not a JSON list: %s", response_text)
                return

            best_candidate = None
            best_score = -1.0

            for cand in candidates:
                name = cand.get("Task name") or cand.get("task_name") or cand.get("name")
                if not name:
                    continue
                value = float(cand.get("Estimated value") or cand.get("estimated_value") or cand.get("value") or 1.0)
                urgency = float(cand.get("Estimated urgency") or cand.get("estimated_urgency") or cand.get("urgency") or 1.0)
                risk = float(cand.get("Estimated risk") or cand.get("estimated_risk") or cand.get("risk") or 1.0)
                cost = float(cand.get("Estimated cost") or cand.get("estimated_cost") or cand.get("cost") or 1.0)

                # score = (value * urgency) / (cost + risk)
                score = (value * urgency) / (cost + risk + 0.1)
                LOGGER.info("Candidate: %s -> Utility Score: %.2f", name, score)

                if score > best_score:
                    best_score = score
                    best_candidate = cand

            # If top score > threshold, push to TaskConductor queue
            threshold = 1.0
            if best_candidate and best_score >= threshold:
                task_name = best_candidate.get("Task name") or best_candidate.get("task_name") or best_candidate.get("name")
                LOGGER.info("Selected autonomous task: %s with score %.2f", task_name, best_score)

                inbound = InboundMessage(
                    source="autonomous",
                    user_id="system",
                    channel="autonomous",
                    text=f"Execute autonomous goal: {task_name}",
                    priority=2,
                    metadata={"utility_score": best_score, "autonomous": True},
                )
                self.conductor.try_submit_message(inbound)

                # Log autonomous goal choice to Notepad
                await self.conductor.runtime.notepad_writer.enqueue(
                    entry_type="autonomous_goal_selection",
                    payload={"task": best_candidate, "utility_score": best_score},
                    constitution="restricted",
                    personas={"Distributed Mind": 1.0},
                )

        except Exception as exc:
            LOGGER.error("Failed to generate or parse autonomous goals: %s", exc)

    async def _get_git_status(self) -> str:
        try:
            import subprocess
            res = await asyncio.to_thread(
                subprocess.run,
                ["git", "status", "--porcelain"],
                cwd=str(self.settings.workspace_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            return res.stdout.strip() or "clean"
        except Exception:
            return "unknown"

    def _get_system_health(self) -> str:
        try:
            import psutil
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            return f"CPU: {cpu}%, RAM: {mem.percent}%"
        except Exception:
            return "unknown"

    async def _get_recent_notepad_activity(self) -> str:
        try:
            entries = await asyncio.to_thread(self.conductor.notepad.latest_entries, 5)
            if not entries:
                return "No recent entries"
            summary = []
            for entry in entries:
                summary.append(f"[{entry.get('entry_type')}] {str(entry.get('payload'))[:100]}")
            return "\n".join(summary)
        except Exception:
            return "unknown"
