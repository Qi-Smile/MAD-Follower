from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from openai import AsyncOpenAI
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

from .config import LLMSettings


class LLMClientError(RuntimeError):
    pass


class MissingCredentialsError(LLMClientError):
    pass


class LLMTimeoutError(LLMClientError):
    pass


@dataclass(slots=True)
class LLMCompletion:
    text: str
    latency_ms: float
    usage: Dict[str, Any]


class LLMClient:
    """Thin async wrapper around OpenAI-compatible chat completions."""

    def __init__(self, settings: LLMSettings):
        base_url = os.environ.get("BASE_URL")
        api_key = os.environ.get("API_KEY")
        if not base_url or not api_key:
            raise MissingCredentialsError(
                "BASE_URL and API_KEY must be defined in the environment before running the experiment."
            )
        self._settings = settings
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def complete(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMCompletion:
        temperature = temperature if temperature is not None else self._settings.temperature
        max_tokens = max_tokens if max_tokens is not None else self._settings.max_tokens

        attempts = self._settings.max_retries + 1
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                async with self._semaphore:
                    start = time.perf_counter()
                    response = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=self._settings.model,
                            messages=list(messages),
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        timeout=self._settings.timeout_seconds,
                    )
                choice = response.choices[0].message
                latency_ms = (time.perf_counter() - start) * 1000.0
                text = choice.content or ""
                usage_payload = self._normalize_usage(response.usage)
                return LLMCompletion(text=text, latency_ms=latency_ms, usage=usage_payload)
            except asyncio.TimeoutError as exc:  # pragma: no cover - network
                last_error = LLMTimeoutError("Completion timed out")
            except (APITimeoutError, APIConnectionError) as exc:  # pragma: no cover - network
                last_error = LLMTimeoutError(f"LLM request timed out: {exc}")
            except (APIError, APIStatusError) as exc:  # pragma: no cover - network
                last_error = LLMClientError(f"OpenAI API error: {exc}")
            except Exception as exc:  # pragma: no cover - network
                last_error = LLMClientError(f"Unexpected LLM error: {exc}")

            if attempt < attempts - 1:
                await asyncio.sleep(min(2 ** attempt, 5))
                continue
            if last_error:
                raise last_error
        # Should not reach here
        raise LLMClientError("Unknown LLM failure.")

    @property
    def model(self) -> str:
        return self._settings.model

    @property
    def max_tokens(self) -> int:
        return self._settings.max_tokens

    @staticmethod
    def _normalize_usage(usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return dict(usage)
        extractor_order = ("model_dump", "to_dict")
        for attr in extractor_order:
            method = getattr(usage, attr, None)
            if callable(method):
                try:
                    data = method()
                    if isinstance(data, dict):
                        return data
                except Exception:
                    continue
        if hasattr(usage, "__dict__"):
            return {
                key: value
                for key, value in usage.__dict__.items()
                if not key.startswith("_")
            }
        return {"value": repr(usage)}
