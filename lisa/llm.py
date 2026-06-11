from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import time
from typing import Any
from pathlib import Path

import httpx

from lisa.config import Settings
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
            self.tokens = min(self.capacity, float(self.tokens) + elapsed * self.refill_per_second)
            self.updated_at = now
            if self.tokens >= amount:
                self.tokens -= amount
                return
            sleep_for = max(0.05, (amount - self.tokens) / max(self.refill_per_second, 1e-6))
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
            self.persona_bank = self._load_or_initialize_persona_bank(settings.persona_vectors_path)
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
        if self.settings.model_provider == "local":
            return self.settings.local_model_path is not None
        return bool(
            self.settings.model_provider
            and self.settings.model_name
            and self.settings.model_base_url
            and self.settings.model_api_key
        )

    def persona_prefix(self, persona_weights: dict[str, float]) -> Any:
        return build_persona_injection(self.persona_bank, persona_weights).prefix_vectors

    def persona_summary(self) -> dict[str, dict[str, Any]]:
        return self.persona_bank.summary()

    def _use_local_backend(self) -> bool:
        return self.settings.model_provider == "local" or self.settings.local_model_path is not None

    @property
    def supports_local_generation(self) -> bool:
        return not isinstance(self.local_backend, UnsupportedLocalBackend)

    async def generate_brain(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        persona_weights: dict[str, float] | None = None,
    ) -> BrainGeneration:
        persona_weights = persona_weights or {}
        persona_prefix = self.persona_prefix(persona_weights)
        if self._use_local_backend():
            request = LocalGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                persona_prefix=persona_prefix,
            )
            return await self.local_backend.generate(request)

        if not self.configured:
            raise RuntimeError("External model is not configured.")

        await self._throttle_provider(self.settings.model_provider or "external", max_tokens=max_tokens, prompt=user_prompt)
        result = await self._chat_completion(
            provider=self.settings.model_provider or "external",
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            extra_payload={
                "lisa_persona_weights": persona_weights if persona_weights else None,
                "lisa_persona_prefix_shape": list(persona_prefix.shape),
            },
        )
        parsed = self.tool_call_parser.parse(result.content)
        parsed.used_local_model = False
        parsed.persona_prefix_shape = tuple(int(value) for value in persona_prefix.shape)
        return parsed

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        persona_weights: dict[str, float] | None = None,
    ) -> str:
        return (await self.generate_brain(system_prompt, user_prompt, max_tokens, persona_weights)).text

    async def call_external_llm(
        self,
        provider: str | None,
        prompt: str,
        max_tokens: int = 800,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> ExternalLLMResult:
        provider_name = (provider or self.settings.freellmapi_default_provider or self.settings.model_provider or "openai").strip()
        await self._throttle_provider(provider_name, max_tokens=max_tokens, prompt=prompt)
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
            payload.update({key: value for key, value in extra_payload.items() if value is not None})

        if self.settings.freellmapi_base_url and self.settings.freellmapi_api_key:
            url = f"{self.settings.freellmapi_base_url.rstrip('/')}/v1/chat/completions"
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
        return ExternalLLMResult(provider=provider, model=model, content=content, usage=usage)

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
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        if total_tokens <= 0:
            prompt_tokens = max(1, sum(len(item.get("content", "").split()) for item in messages))
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

    async def _throttle_provider(self, provider: str, *, max_tokens: int, prompt: str) -> None:
        limiter = self._rate_limiter(provider)
        estimated_tokens = max_tokens + max(1, len(prompt.split()))
        await limiter.acquire(estimated_tokens)
