from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import time
from typing import Any
from pathlib import Path

import httpx

from lisa.config import Settings
from lisa.embeddings import EMBEDDING_DIMS, deterministic_embedding
from lisa.local_inference import (
    BrainGeneration,
    LocalGenerationRequest,
    PersonaGatedModel,
    ToolCallParser,
    UnsupportedLocalBackend,
)
from lisa.soft_prompts import PersonaSoftPromptBank, build_persona_injection


@dataclass(slots=True)
class ExternalLLMResult:
    provider: str
    model: str | None
    content: str
    usage: dict[str, int]


@dataclass(slots=True)
class TokenBucket:
    capacity: int
    refill_per_second: float
    tokens: float | None = None
    updated_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = float(self.capacity)

    async def acquire(self, amount: int = 1) -> None:
        amount = max(1, int(amount))
        while True:
            now = time.monotonic()
            elapsed = now - self.updated_at
            self.tokens = min(
                self.capacity, float(self.tokens) + elapsed * self.refill_per_second
            )
            self.updated_at = now
            if self.tokens >= amount:
                self.tokens -= amount
                return
            sleep_for = max(
                0.05, (amount - self.tokens) / max(self.refill_per_second, 1e-6)
            )
            await asyncio.sleep(min(sleep_for, 1.0))


@dataclass(slots=True)
class ProviderRateLimiter:
    request_bucket: TokenBucket
    token_bucket: TokenBucket

    async def acquire(self, estimated_tokens: int) -> None:
        await self.request_bucket.acquire(1)
        await self.token_bucket.acquire(max(1, estimated_tokens))


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        persona_bank: PersonaSoftPromptBank | None = None,
        local_backend: Any | None = None,
    ):
        self.settings = settings
        if persona_bank is not None:
            self.persona_bank = persona_bank
        else:
            self.persona_bank = self._load_or_initialize_persona_bank(
                settings.persona_vectors_path
            )
        self.tool_call_parser = ToolCallParser()
        self._rate_limiters: dict[str, ProviderRateLimiter] = {}
        if local_backend is not None:
            self.local_backend = local_backend
        elif self.settings.local_model_path is not None:
            self.local_backend = PersonaGatedModel(
                model_path=self.settings.local_model_path,
                persona_bank=self.persona_bank,
                context_size=self.settings.local_model_context_size,
                n_threads=self.settings.local_model_n_threads,
                n_gpu_layers=self.settings.local_model_n_gpu_layers,
            )
        else:
            self.local_backend = UnsupportedLocalBackend()

    @staticmethod
    def _load_or_initialize_persona_bank(path: Path) -> PersonaSoftPromptBank:
        if path.exists():
            return PersonaSoftPromptBank.load(path)
        bank = PersonaSoftPromptBank.initialize()
        bank.save(path)
        return bank

    @property
    def configured(self) -> bool:
        return self.local_backend_ready or self.external_backend_configured

    def persona_prefix(self, persona_weights: dict[str, float]) -> Any:
        return build_persona_injection(
            self.persona_bank, persona_weights
        ).prefix_vectors

    def persona_summary(self) -> dict[str, dict[str, Any]]:
        return self.persona_bank.summary()

    @property
    def external_backend_configured(self) -> bool:
        if self.settings.freellmapi_api_key and (
            self.settings.freellmapi_chat_url or self.settings.freellmapi_base_url
        ):
            return True
        return bool(
            self.settings.model_provider
            and self.settings.model_name
            and self.settings.model_base_url
            and self.settings.model_api_key
        )

    def _use_local_backend(self) -> bool:
        if self.settings.model_provider == "local":
            return True
        if self.settings.model_provider:
            return False
        return self.settings.local_model_path is not None

    @property
    def supports_local_generation(self) -> bool:
        return not isinstance(self.local_backend, UnsupportedLocalBackend)

    @property
    def local_backend_ready(self) -> bool:
        if isinstance(self.local_backend, UnsupportedLocalBackend):
            return False
        ready = getattr(self.local_backend, "ready", None)
        if isinstance(ready, bool):
            return ready
        return True

    def local_backend_status(self) -> dict[str, Any]:
        return {
            "configured_path": (
                str(self.settings.local_model_path)
                if self.settings.local_model_path is not None
                else None
            ),
            "supports_local_generation": self.supports_local_generation,
            "ready": self.local_backend_ready,
            "load_error": getattr(self.local_backend, "load_error", None),
        }

    def _external_provider_name(self, override: str | None = None) -> str:
        if override:
            return override.strip()
        configured_model_provider = (self.settings.model_provider or "").strip()
        if configured_model_provider and configured_model_provider != "local":
            return configured_model_provider
        freellmapi_provider = (self.settings.freellmapi_default_provider or "").strip()
        if freellmapi_provider:
            return freellmapi_provider
        return "openai"

    async def generate_brain(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        persona_weights: dict[str, float] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        stress_level: int = 0,
    ) -> BrainGeneration:
        persona_weights = persona_weights or {}
        persona_prefix = self.persona_prefix(persona_weights)
        local_request = LocalGenerationRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            persona_prefix=persona_prefix,
        )
        if self._should_use_dual_brain(
            stress_level=stress_level, user_prompt=user_prompt, max_tokens=max_tokens
        ):
            return await self._generate_hybrid_brain(
                local_request=local_request,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                persona_weights=persona_weights,
                conversation_history=conversation_history,
            )

        if self._use_local_backend():
            if not self.local_backend_ready and self.external_backend_configured:
                return await self._generate_external_brain(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    persona_weights=persona_weights,
                    conversation_history=conversation_history,
                    persona_prefix=persona_prefix,
                )
            return await self._generate_local_brain(local_request)
        if self.external_backend_configured:
            return await self._generate_external_brain(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                persona_weights=persona_weights,
                conversation_history=conversation_history,
                persona_prefix=persona_prefix,
            )
        raise RuntimeError("No configured LLM backend is available.")

    async def generate_with_backend(
        self,
        backend: str,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        persona_weights: dict[str, float] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        stress_level: int = 0,
    ) -> BrainGeneration:
        normalized = backend.strip().lower()
        persona_weights = persona_weights or {}
        persona_prefix = self.persona_prefix(persona_weights)
        if normalized in {"tinyllama", "local", "local_brain"}:
            request = LocalGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                persona_prefix=persona_prefix,
            )
            if self.local_backend_ready:
                return await self._generate_local_brain(request)
            if self.external_backend_configured:
                return await self._generate_external_brain(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    persona_weights=persona_weights,
                    conversation_history=conversation_history,
                    persona_prefix=persona_prefix,
                )
            raise RuntimeError("TinyLlama/local backend is unavailable.")
        if normalized in {
            "freellm_external",
            "external",
            "reflection",
            "code_gen",
            "evolution",
            "tool_use",
        }:
            return await self._generate_external_brain(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                persona_weights=persona_weights,
                conversation_history=conversation_history,
                persona_prefix=persona_prefix,
            )
        if normalized in {"hybrid", "dual"}:
            return await self.generate_brain(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                persona_weights=persona_weights,
                conversation_history=conversation_history,
                stress_level=stress_level,
            )
        return await self.generate_brain(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            persona_weights=persona_weights,
            conversation_history=conversation_history,
            stress_level=stress_level,
        )

    async def classify_with_local(
        self,
        *,
        labels: list[str],
        task: str,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.local_backend_ready:
            return None
        prompt = (
            "Classify the developer task into exactly one label from this list: "
            f"{', '.join(labels)}.\n"
            f"Context: {context or {}}\n"
            f"Task: {task}\n"
            "Return only the label."
        )
        try:
            generation = await self.generate_with_backend(
                "tinyllama",
                system_prompt="You are a routing classifier.",
                user_prompt=prompt,
                max_tokens=16,
                persona_weights={"architect": 1.0},
            )
        except Exception:
            return None
        answer = generation.text.strip().lower()
        for label in labels:
            if label.lower() in answer:
                return label
        return None

    async def embed_text(self, text: str) -> list[float]:
        if self.settings.freellmapi_embeddings_url and self.settings.freellmapi_api_key:
            try:
                async with httpx.AsyncClient(
                    timeout=self.settings.freellmapi_timeout_seconds
                ) as client:
                    response = await client.post(
                        self.settings.freellmapi_embeddings_url,
                        json={"input": text, "model": "auto"},
                        headers={
                            "Authorization": f"Bearer {self.settings.freellmapi_api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    items = data.get("data") or []
                    if (
                        items
                        and isinstance(items[0], dict)
                        and isinstance(items[0].get("embedding"), list)
                    ):
                        vector = [float(value) for value in items[0]["embedding"]]
                        if len(vector) == EMBEDDING_DIMS:
                            return vector
            except Exception:
                pass
        return deterministic_embedding(text, EMBEDDING_DIMS)

    def _should_use_dual_brain(
        self, *, stress_level: int, user_prompt: str, max_tokens: int
    ) -> bool:
        if not self.settings.hybrid_brain_enabled:
            return False
        if not self.local_backend_ready or not self.external_backend_configured:
            return False
        if stress_level >= max(1, int(self.settings.hybrid_brain_stress_threshold)):
            return True
        if len(user_prompt) >= max(
            120, int(self.settings.hybrid_brain_prompt_chars_threshold)
        ):
            return True
        return max_tokens >= 1400

    async def _generate_local_brain(
        self, request: LocalGenerationRequest
    ) -> BrainGeneration:
        if not self.local_backend_ready:
            status = self.local_backend_status()
            raise RuntimeError(
                status.get("load_error")
                or "Local inference backend is configured but not ready."
            )
        return await self.local_backend.generate(request)

    async def _generate_external_brain(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        persona_weights: dict[str, float],
        conversation_history: list[dict[str, str]] | None,
        persona_prefix: Any,
    ) -> BrainGeneration:
        if not self.external_backend_configured:
            raise RuntimeError("External model is not configured.")

        provider_name = self._external_provider_name()
        model_name = self.settings.model_name or "auto"

        await self._throttle_provider(
            provider_name, max_tokens=max_tokens, prompt=user_prompt
        )
        history_messages = conversation_history or []
        api_messages = (
            [{"role": "system", "content": system_prompt}]
            + history_messages
            + [{"role": "user", "content": user_prompt}]
        )

        result = await self._chat_completion(
            provider=provider_name,
            model=model_name,
            messages=api_messages,
            max_tokens=max_tokens,
            extra_payload={
                "lisa_persona_weights": persona_weights if persona_weights else None,
                "lisa_persona_prefix_shape": list(persona_prefix.shape),
            },
        )
        parsed = self.tool_call_parser.parse(result.content)
        parsed.used_local_model = False
        parsed.persona_prefix_shape = tuple(
            int(value) for value in persona_prefix.shape
        )
        return parsed

    async def _generate_hybrid_brain(
        self,
        *,
        local_request: LocalGenerationRequest,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        persona_weights: dict[str, float],
        conversation_history: list[dict[str, str]] | None,
    ) -> BrainGeneration:
        tasks: dict[str, asyncio.Task[BrainGeneration]] = {
            "local": asyncio.create_task(
                self._generate_local_brain(local_request), name="lisa-local-brain"
            ),
            "external": asyncio.create_task(
                self._generate_external_brain(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    persona_weights=persona_weights,
                    conversation_history=conversation_history,
                    persona_prefix=local_request.persona_prefix,
                ),
                name="lisa-external-brain",
            ),
        }
        started_at = time.monotonic()
        candidates: list[tuple[str, BrainGeneration, float]] = []
        errors: list[Exception] = []
        pending = set(tasks.values())

        while pending and not candidates:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for finished in done:
                name = next(key for key, value in tasks.items() if value is finished)
                try:
                    candidates.append(
                        (name, finished.result(), time.monotonic() - started_at)
                    )
                except Exception as exc:
                    errors.append(exc)

        if candidates and pending:
            grace_seconds = max(
                0.05, int(self.settings.hybrid_brain_race_window_ms) / 1000.0
            )
            extra_done, pending = await asyncio.wait(pending, timeout=grace_seconds)
            for finished in extra_done:
                name = next(key for key, value in tasks.items() if value is finished)
                try:
                    candidates.append(
                        (name, finished.result(), time.monotonic() - started_at)
                    )
                except Exception as exc:
                    errors.append(exc)

        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        if not candidates:
            if errors:
                raise errors[0]
            raise RuntimeError("Hybrid brain execution returned no result.")

        chosen_name, chosen_result, _ = self._select_hybrid_candidate(candidates)
        chosen_result.raw_text = (
            chosen_result.raw_text or f"[hybrid:{chosen_name}] {chosen_result.text}"
        )
        return chosen_result

    @staticmethod
    def _select_hybrid_candidate(
        candidates: list[tuple[str, BrainGeneration, float]],
    ) -> tuple[str, BrainGeneration, float]:
        def score(candidate: tuple[str, BrainGeneration, float]) -> tuple[float, float]:
            _, generation, latency = candidate
            tool_bonus = 4.0 if generation.tool_calls else 0.0
            text_bonus = min(2.0, len(generation.text.strip()) / 240.0)
            locality_bonus = 0.1 if generation.used_local_model else 0.2
            latency_penalty = latency
            return (
                tool_bonus + text_bonus + locality_bonus - latency_penalty,
                -latency,
            )

        return max(candidates, key=score)

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        persona_weights: dict[str, float] | None = None,
    ) -> str:
        return (
            await self.generate_brain(
                system_prompt, user_prompt, max_tokens, persona_weights
            )
        ).text

    async def call_external_llm(
        self,
        provider: str | None,
        prompt: str,
        max_tokens: int = 800,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> ExternalLLMResult:
        provider_name = self._external_provider_name(provider)
        await self._throttle_provider(
            provider_name, max_tokens=max_tokens, prompt=prompt
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._chat_completion(
            provider=provider_name,
            model=model or self.settings.model_name,
            messages=messages,
            max_tokens=max_tokens,
        )

    async def _chat_completion(
        self,
        provider: str,
        model: str | None,
        messages: list[dict[str, str]],
        max_tokens: int,
        extra_payload: dict[str, Any] | None = None,
    ) -> ExternalLLMResult:
        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if extra_payload:
            payload.update(
                {
                    key: value
                    for key, value in extra_payload.items()
                    if value is not None
                }
            )

        if self.settings.freellmapi_chat_url:
            url = self.settings.freellmapi_chat_url
            headers = {
                "Authorization": f"Bearer {self.settings.freellmapi_api_key}",
                "Content-Type": "application/json",
            }
            timeout = self.settings.freellmapi_timeout_seconds
        elif self.settings.freellmapi_base_url and self.settings.freellmapi_api_key:
            base_url_str = str(self.settings.freellmapi_base_url).rstrip("/")
            if not base_url_str.endswith("/v1"):
                url = f"{base_url_str}/v1/chat/completions"
            else:
                url = f"{base_url_str}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.settings.freellmapi_api_key}",
                "Content-Type": "application/json",
            }
            timeout = self.settings.freellmapi_timeout_seconds
        else:
            if not (self.settings.model_base_url and self.settings.model_api_key):
                raise RuntimeError("External model is not configured.")
            url = f"{self.settings.model_base_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.settings.model_api_key}",
                "Content-Type": "application/json",
            }
            timeout = self.settings.external_timeout_seconds

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        content, usage = self._extract_completion_content_and_usage(data, messages)
        return ExternalLLMResult(
            provider=provider, model=model, content=content, usage=usage
        )

    @staticmethod
    def _extract_completion_content_and_usage(
        data: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, int]]:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("External model response did not include any choices.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("External model returned an empty message.")

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(
            usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        )
        if total_tokens <= 0:
            prompt_tokens = max(
                1, sum(len(item.get("content", "").split()) for item in messages)
            )
            completion_tokens = max(1, len(content.split()))
            total_tokens = prompt_tokens + completion_tokens

        return content.strip(), {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _rate_limiter(self, provider: str) -> ProviderRateLimiter:
        limiter = self._rate_limiters.get(provider)
        if limiter is not None:
            return limiter

        request_capacity = max(1, int(self.settings.freellmapi_requests_per_minute))
        token_capacity = max(1, int(self.settings.freellmapi_tokens_per_minute))
        limiter = ProviderRateLimiter(
            request_bucket=TokenBucket(
                capacity=request_capacity,
                refill_per_second=request_capacity / 60.0,
            ),
            token_bucket=TokenBucket(
                capacity=token_capacity,
                refill_per_second=token_capacity / 60.0,
            ),
        )
        self._rate_limiters[provider] = limiter
        return limiter

    async def _throttle_provider(
        self, provider: str, *, max_tokens: int, prompt: str
    ) -> None:
        limiter = self._rate_limiter(provider)
        estimated_tokens = max_tokens + max(1, len(prompt.split()))
        await limiter.acquire(estimated_tokens)
