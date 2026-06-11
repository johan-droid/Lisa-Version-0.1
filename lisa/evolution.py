import asyncio
import json
import re
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from lisa.constitutions import ConstitutionMode
from lisa.events import EventBus, LisaEvent


@dataclass(slots=True)
class EvolutionSkillSpec:
    name: str
    description: str
    code: str
    test_command: str
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FailureCluster:
    theme: str
    items: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    score: float = 0.0


@dataclass(slots=True)
class EvolutionCycleResult:
    status: str
    candidate_count: int
    knowledge_sources: list[dict[str, Any]] = field(default_factory=list)
    skill_name: str | None = None
    registered: bool = False
    test_passed: bool = False
    skipped_reason: str | None = None
    details: list[str] = field(default_factory=list)


class NightlyEvolutionScheduler:
    def __init__(
        self,
        runtime: Any,
        conductor: Any,
        event_bus: EventBus,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.runtime = runtime
        self.conductor = conductor
        self.event_bus = event_bus
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._cycle_lock = asyncio.Lock()
        self._last_attempt_at: datetime | None = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.runtime.settings, "evolution_enabled", False))

    async def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="nightly-evolution")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def run_once(self, force: bool = False) -> EvolutionCycleResult:
        async with self._cycle_lock:
            now = self._now()
            if not force and not self.enabled:
                return EvolutionCycleResult(status="skipped", candidate_count=0, skipped_reason="disabled")
            idle_threshold = self._idle_threshold_seconds(now)
            if not force and not self._is_idle(now, idle_threshold):
                return EvolutionCycleResult(
                    status="skipped",
                    candidate_count=0,
                    skipped_reason=f"recent_activity_under_{idle_threshold}s",
                )

            clusters = await self._collect_failure_clusters()
            if not clusters:
                fallback_candidates = await self._collect_candidates()
                clusters = self._clusters_from_candidates(fallback_candidates)

            candidate_count = sum(len(cluster.items) for cluster in clusters)
            await self.event_bus.publish(
                LisaEvent(
                    type="evolution.cycle_started",
                    payload={
                        "candidate_count": candidate_count,
                        "cluster_count": len(clusters),
                        "night_window_start_hour": self.runtime.settings.evolution_window_start_hour,
                    },
                )
            )

            result = EvolutionCycleResult(status="running", candidate_count=candidate_count)
            for cluster in clusters:
                try:
                    knowledge_sources = await self._collect_fresh_knowledge(cluster)
                    result.knowledge_sources = knowledge_sources
                    skill = await self._synthesize_skill(cluster=cluster, knowledge_sources=knowledge_sources)
                    result.skill_name = skill.name
                    result.details.extend(skill.notes)

                    stage_path = self._stage_path_for(skill.name)
                    stage_rel_path = stage_path.relative_to(self.runtime.settings.workspace_root).as_posix()
                    await self._stage_skill(stage_path, skill.code)

                    test_command = self._format_test_command(skill.test_command, stage_rel_path)
                    test_output = await self._test_skill(test_command)
                    result.test_passed = bool(test_output.get("returncode", 1) == 0)
                    result.details.append(f"cluster={cluster.theme}")
                    result.details.append(f"test_returncode={test_output.get('returncode')}")

                    if not result.test_passed:
                        result.status = "failed"
                        result.details.append(test_output.get("stderr") or "skill test failed")
                        await self._update_dashboard("failed", skill.name)
                        await self._write_cycle_record(result, clusters)
                        await self.event_bus.publish(
                            LisaEvent(
                                type="evolution.cycle_finished",
                                payload={
                                    "status": result.status,
                                    "skill_name": skill.name,
                                    "candidate_count": candidate_count,
                                    "cluster": cluster.theme,
                                    "details": result.details,
                                },
                            )
                        )
                        return result

                    await self._register_skill(skill)
                    result.registered = True
                    result.status = "registered"
                    await self._update_dashboard("registered", skill.name)
                    await self.event_bus.publish(
                        LisaEvent(
                            type="evolution.skill_registered",
                            payload={
                                "skill_name": skill.name,
                                "candidate_count": candidate_count,
                                "cluster": cluster.theme,
                                "test_command": test_command,
                            },
                        )
                    )
                    await self._write_cycle_record(result, clusters)
                    await self.event_bus.publish(
                        LisaEvent(
                            type="evolution.cycle_finished",
                            payload={
                                "status": result.status,
                                "skill_name": skill.name,
                                "candidate_count": candidate_count,
                                "cluster": cluster.theme,
                                "details": result.details,
                            },
                        )
                    )
                    self._last_attempt_at = now
                    return result
                except Exception as exc:
                    result.status = "failed"
                    result.details.append(f"{cluster.theme}: {exc}")
                    continue

            await self._update_dashboard("failed", result.skill_name or "unknown")
            await self._write_cycle_record(result, clusters)
            await self.event_bus.publish(
                LisaEvent(
                    type="evolution.cycle_finished",
                    payload={
                        "status": result.status,
                        "skill_name": result.skill_name,
                        "candidate_count": candidate_count,
                        "details": result.details,
                    },
                )
            )
            self._last_attempt_at = now
            return result

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception as exc:  # pragma: no cover - defensive loop guard
                await self.event_bus.publish(
                    LisaEvent(type="evolution.cycle_finished", payload={"status": "error", "error": str(exc)})
                )

            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.runtime.settings.evolution_check_interval_seconds,
                )
            except TimeoutError:
                continue

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            return now.astimezone()
        return now

    def _is_idle(self, now: datetime, idle_for_seconds: int) -> bool:
        if hasattr(self.conductor, "is_idle") and not self.conductor.is_idle():
            return False

        last_activity = self._latest_user_activity()
        if last_activity is None:
            return True

        age_seconds = (now - last_activity).total_seconds()
        return age_seconds >= max(0, idle_for_seconds)

    def _idle_threshold_seconds(self, now: datetime) -> int:
        base_idle = max(60, int(self.runtime.settings.evolution_idle_after_seconds))
        if self._is_nighttime(now):
            return max(60, base_idle // 3)
        return base_idle

    def _is_nighttime(self, now: datetime) -> bool:
        return self._is_within_window(now)

    def _latest_user_activity(self) -> datetime | None:
        source = getattr(self.runtime.notepad, "latest_interaction_timestamp", None)
        if callable(source):
            timestamp = source()
            if isinstance(timestamp, datetime):
                return timestamp if timestamp.tzinfo is not None else timestamp.astimezone()

        latest_entries = self.runtime.notepad.latest_entries(limit=1)
        if not latest_entries:
            return None

        latest = latest_entries[0]
        created_at = datetime.fromisoformat(latest["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.astimezone()
        return created_at

    def _is_within_window(self, now: datetime) -> bool:
        start = int(self.runtime.settings.evolution_window_start_hour) % 24
        duration = max(1, int(self.runtime.settings.evolution_window_duration_hours))
        end = (start + duration) % 24
        if duration >= 24:
            return True
        if start < end:
            return start <= now.hour < end
        return now.hour >= start or now.hour < end

    async def _collect_failure_clusters(self) -> list[FailureCluster]:
        failure_items = await asyncio.to_thread(
            self._failure_items_from_notepad,
        )
        if not failure_items:
            return []
        return self._clusters_from_candidates(failure_items)

    async def _collect_candidates(self) -> list[dict[str, Any]]:
        candidates = await asyncio.to_thread(
            self.runtime.notepad.evolution_candidates,
            self.runtime.settings.evolution_candidate_limit,
            self.runtime.settings.evolution_min_reward,
        )
        if candidates:
            return candidates
        return await asyncio.to_thread(self.runtime.notepad.latest_entries, 5)

    def _failure_items_from_notepad(self) -> list[dict[str, Any]]:
        failure_reader = getattr(self.runtime.notepad, "recent_failure_interactions", None)
        if callable(failure_reader):
            return list(
                failure_reader(
                    limit=max(6, int(self.runtime.settings.evolution_candidate_limit)),
                    min_reward=float(self.runtime.settings.evolution_min_reward),
                )
            )
        return list(self.runtime.notepad.evolution_candidates(
            self.runtime.settings.evolution_candidate_limit,
            self.runtime.settings.evolution_min_reward,
        ))

    def _clusters_from_candidates(self, candidates: list[dict[str, Any]]) -> list[FailureCluster]:
        clusters: dict[str, FailureCluster] = {}
        for item in candidates:
            payload = item.get("payload") if isinstance(item, dict) else {}
            if not isinstance(payload, dict):
                continue
            theme = self._failure_theme_from_payload(payload)
            cluster = clusters.get(theme)
            if cluster is None:
                cluster = FailureCluster(theme=theme, query=self._cluster_query_from_payload(payload))
                clusters[theme] = cluster
            cluster.items.append(item)
            cluster.score = max(cluster.score, float(item.get("score") or 0.0))

        ordered = sorted(
            clusters.values(),
            key=lambda cluster: (cluster.score, len(cluster.items)),
            reverse=True,
        )
        return ordered

    async def _collect_fresh_knowledge(self, cluster: FailureCluster) -> list[dict[str, Any]]:
        if not getattr(self.runtime.settings, "enable_browser_tools", False):
            browser_sources: list[dict[str, Any]] = []
        else:
            browser_sources = []

        queries = self._build_browser_queries(cluster)
        knowledge: list[dict[str, Any]] = []
        for query in queries[: self.runtime.settings.evolution_browser_queries]:
            if getattr(self.runtime.settings, "enable_browser_tools", False):
                try:
                    search_result = await self.runtime.tools.invoke(
                        "browser_search",
                        {"query": query},
                        ConstitutionMode.UNRESTRICTED,
                    )
                except Exception:
                    search_result = None
                if search_result is not None:
                    knowledge.append({"cluster": cluster.theme, "query": query, "search": search_result})
                    results = search_result.get("results") if isinstance(search_result, dict) else None
                    if isinstance(results, list):
                        for item in results[:2]:
                            url = item.get("url") if isinstance(item, dict) else None
                            if not isinstance(url, str) or not url:
                                continue
                            try:
                                fetch_result = await self.runtime.tools.invoke(
                                    "browser_fetch",
                                    {"url": url, "extract_text": True},
                                    ConstitutionMode.UNRESTRICTED,
                                )
                            except Exception:
                                continue
                            knowledge.append({"cluster": cluster.theme, "query": query, "fetch": fetch_result})

            research_prompt = (
                "Given this failure cluster and the surrounding documentation findings, "
                "suggest a reusable Python recovery pattern, but keep the answer compact.\n\n"
                f"Cluster theme: {cluster.theme}\n"
                f"Cluster query: {query}\n"
                f"Cluster examples: {json.dumps(cluster.items[:2], ensure_ascii=True, indent=2)}\n"
                f"Current research: {json.dumps(knowledge[-6:], ensure_ascii=True, indent=2)}"
            )
            try:
                llm_result = await self.runtime.tools.invoke(
                    "call_external_llm",
                    {
                        "provider": self.runtime.settings.freellmapi_default_provider or self.runtime.settings.model_provider or "",
                        "model": self.runtime.settings.model_name or "",
                        "prompt": research_prompt,
                        "max_tokens": 512,
                        "system_prompt": (
                            "You are the Evolution Engine for LISA. Return a short JSON object "
                            "with keys insight, pattern, and keywords."
                        ),
                    },
                    ConstitutionMode.UNRESTRICTED,
                )
                if isinstance(llm_result, dict):
                    knowledge.append({"cluster": cluster.theme, "query": query, "llm": llm_result})
            except Exception:
                continue

        return knowledge

    async def _synthesize_skill(
        self,
        cluster: FailureCluster,
        knowledge_sources: list[dict[str, Any]],
    ) -> EvolutionSkillSpec:
        prompt = self._build_generation_prompt(cluster, knowledge_sources)
        try:
            result = await self.runtime.tools.invoke(
                "call_external_llm",
                {
                    "provider": self.runtime.settings.freellmapi_default_provider or self.runtime.settings.model_provider or "",
                    "model": self.runtime.settings.model_name or "",
                    "prompt": prompt,
                    "max_tokens": 1200,
                    "system_prompt": (
                        "You are the Evolution Engine for LISA. "
                        "Return only strict JSON with keys name, description, code, test_command, notes. "
                        "The code must be a single Python function or small module using only the standard library. "
                        "The test_command must reference the staged file with the literal token {path}."
                    ),
                },
                ConstitutionMode.UNRESTRICTED,
            )
            content = result.get("content") if isinstance(result, dict) else None
        except Exception:
            content = None

        parsed = self._parse_skill_spec(content or "", cluster, knowledge_sources)
        if parsed is not None:
            return parsed
        return self._fallback_skill_spec(cluster, knowledge_sources)

    def _fallback_skill_spec(
        self,
        cluster: FailureCluster,
        knowledge_sources: list[dict[str, Any]],
    ) -> EvolutionSkillSpec:
        slug = self._slugify(cluster.theme or "evolution_helper")
        description = "Summarize a recurring failure theme into a reusable helper."
        lines = [
            f"def {slug}(context):",
            '    """Auto-generated recovery helper."""',
            "    return {",
            f'        "candidate_count": {len(cluster.items)},',
            f'        "knowledge_sources": {len(knowledge_sources)},',
            '        "status": "ready",',
            "    }",
        ]
        return EvolutionSkillSpec(
            name=slug,
            description=description,
            code="\n".join(lines),
            test_command="python -m py_compile {path}",
            notes=["Fallback skill generated because external synthesis was unavailable."],
        )

    def _parse_skill_spec(
        self,
        content: str,
        cluster: FailureCluster,
        knowledge_sources: list[dict[str, Any]],
    ) -> EvolutionSkillSpec | None:
        cleaned = self._strip_code_fences(content).strip()
        if not cleaned:
            return None

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        name = self._slugify(str(data.get("name") or cluster.theme or "evolution_helper"))
        description = str(data.get("description") or "Auto-generated evolution skill.")
        code = str(data.get("code") or "").strip()
        body = str(data.get("body") or "").strip()
        if not code and body:
            code = body
        if not code:
            code = self._fallback_skill_spec(cluster, knowledge_sources).code
        if "def " not in code and "async def " not in code:
            code = "\n".join(
                [
                    f"def {name}(context):",
                    "    \"\"\"Auto-generated evolution skill.\"\"\"",
                    *[f"    {line}" if line else "    pass" for line in (code.splitlines() or ["pass"])],
                ]
            )

        test_command = str(data.get("test_command") or "python -m py_compile {path}").strip()
        notes = data.get("notes")
        note_items = [str(item) for item in notes if isinstance(item, (str, int, float))] if isinstance(notes, list) else []
        return EvolutionSkillSpec(
            name=name,
            description=description,
            code=code,
            test_command=test_command,
            notes=note_items,
        )

    async def _stage_skill(self, path: Path, code: str) -> None:
        await self.runtime.tools.invoke(
            "file_write",
            {"path": str(path), "content": code},
            ConstitutionMode.UNRESTRICTED,
        )

    async def _test_skill(self, command: str) -> dict[str, Any]:
        result = await self.runtime.tools.invoke(
            "terminal_exec",
            {"command": command, "timeout": 60},
            ConstitutionMode.UNRESTRICTED,
        )
        return result if isinstance(result, dict) else {"returncode": 1, "stdout": "", "stderr": str(result)}

    async def _register_skill(self, skill: EvolutionSkillSpec) -> None:
        await self.runtime.tools.invoke(
            "add_skill",
            {"name": skill.name, "code": skill.code},
            ConstitutionMode.UNRESTRICTED,
        )

    async def _update_dashboard(self, status: str, skill_name: str | None) -> None:
        await self.runtime.tools.invoke(
            "dashboard_update",
            {"metric": "evolution_status", "value": status},
            ConstitutionMode.UNRESTRICTED,
        )
        if skill_name:
            await self.runtime.tools.invoke(
                "dashboard_update",
                {"metric": "evolution_last_skill", "value": skill_name},
                ConstitutionMode.UNRESTRICTED,
            )

    async def _write_cycle_record(
        self,
        result: EvolutionCycleResult,
        clusters: list[FailureCluster],
    ) -> None:
        try:
            future = await self.runtime.notepad_writer.enqueue(
                entry_type="evolution_cycle",
                payload={
                    "status": result.status,
                    "candidate_count": result.candidate_count,
                    "skill_name": result.skill_name,
                    "registered": result.registered,
                    "test_passed": result.test_passed,
                    "skipped_reason": result.skipped_reason,
                    "details": result.details,
                    "failure_clusters": [
                        {
                            "theme": cluster.theme,
                            "query": cluster.query,
                            "score": cluster.score,
                            "count": len(cluster.items),
                            "examples": cluster.items[:3],
                        }
                        for cluster in clusters[:5]
                    ],
                },
                constitution=ConstitutionMode.UNRESTRICTED,
                personas={"evolution_engine": 1.0},
                reward=1.0 if result.registered else 0.0,
            )
            await future
        except Exception:
            return

    async def _publish_note(self, message: str) -> None:
        await self.event_bus.publish(
            LisaEvent(
                type="evolution.note",
                payload={"message": message},
            )
        )

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return content.strip()

    def _build_generation_prompt(
        self,
        cluster: FailureCluster,
        knowledge_sources: list[dict[str, Any]],
    ) -> str:
        failure_digest = json.dumps(cluster.items[:5], ensure_ascii=True, indent=2)
        knowledge_digest = json.dumps(knowledge_sources[:5], ensure_ascii=True, indent=2)
        return (
            "Create one new Python skill for LISA.\n"
            "Focus on the most repeated failure theme, keep it small, deterministic, and standard-library only.\n"
            "Return strict JSON with keys: name, description, code, test_command, notes.\n"
            "Use the literal token {path} inside test_command so the scheduler can substitute the staged file path.\n\n"
            f"Cluster theme: {cluster.theme}\n"
            f"Failure candidates:\n{failure_digest}\n\n"
            f"Fresh knowledge:\n{knowledge_digest}\n"
        )

    def _build_browser_queries(self, cluster: FailureCluster) -> list[str]:
        texts: list[str] = []
        for item in cluster.items:
            payload = item.get("payload") if isinstance(item, dict) else {}
            if isinstance(payload, dict):
                texts.extend(
                    [
                        str(payload.get("error") or ""),
                        str(payload.get("self_critique") or ""),
                        str(payload.get("user_input") or ""),
                        str(payload.get("response") or ""),
                    ]
                )
            texts.append(cluster.theme)

        tokens: list[str] = []
        for text in texts:
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", text.lower()):
                if token in {
                    "the",
                    "and",
                    "for",
                    "with",
                    "that",
                    "this",
                    "error",
                    "failed",
                    "task",
                    "lisa",
                    "tool",
                }:
                    continue
                if token not in tokens:
                    tokens.append(token)

        if not tokens:
            return [f"python {cluster.theme.replace('_', ' ')} docs"]

        queries: list[str] = []
        if any(term in tokens for term in {"security", "vulnerability", "cve"}):
            queries.append("python security cve advisory")
        queries.append(f"python {cluster.theme.replace('_', ' ')} docs tutorial")
        if len(tokens) > 1:
            queries.append(f"python {tokens[0]} {tokens[1]} best practices")
        return queries

    @staticmethod
    def _failure_theme_from_payload(payload: dict[str, Any]) -> str:
        for key in ("error", "self_critique", "user_input", "response", "outcome"):
            value = str(payload.get(key) or "").strip()
            if not value:
                continue
            token = re.sub(r"[^a-z0-9_]+", "_", value.split()[0].lower()).strip("_")
            if token:
                return token
        return "evolution_helper"

    def _cluster_query_from_payload(self, payload: dict[str, Any]) -> str:
        parts = [
            str(payload.get("error") or ""),
            str(payload.get("self_critique") or ""),
            str(payload.get("user_input") or ""),
            str(payload.get("response") or ""),
        ]
        tokens: list[str] = []
        for part in parts:
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", part.lower()):
                if token not in tokens:
                    tokens.append(token)
        if not tokens:
            return "python async debugging docs"
        return "python " + " ".join(tokens[:3])

    def _stage_path_for(self, skill_name: str):
        filename = f"{self._slugify(skill_name)}.py"
        return self.runtime.settings.evolution_staging_dir / filename

    def _format_test_command(self, template: str, path: str) -> str:
        command = template.strip() or "python -m py_compile {path}"
        return command.replace("{path}", path)

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
        return slug or "evolution_skill"
