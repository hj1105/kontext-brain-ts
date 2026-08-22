"""Shared OpenAI embedding transport for the isolated RAG framework adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping
import urllib.error
import urllib.request


_PROVIDER_API_KEY_ENVIRONMENT_VARIABLES = (
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "NVIDIA_API_KEY",
)

_CODEX_DISABLED_EVALUATOR_FEATURES = (
    "plugins",
    "remote_plugin",
    "plugin_sharing",
    "apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "skill_mcp_dependency_install",
)


def codex_cli_environment(
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that can only use Codex's saved ChatGPT login."""
    environment = dict(os.environ if base_environment is None else base_environment)
    for name in _PROVIDER_API_KEY_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    return environment


def codex_evaluator_isolation_args() -> list[str]:
    """Disable Codex features that can inject tools or remote catalog calls."""
    return [argument for feature in _CODEX_DISABLED_EVALUATOR_FEATURES
            for argument in ("--disable", feature)]


@dataclass(frozen=True)
class EmbeddingUsage:
    prompt_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def plus(self, other: "EmbeddingUsage") -> "EmbeddingUsage":
        return EmbeddingUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            requests=self.requests + other.requests,
        )

    def as_openai(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
        }


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        api_key: str | None = None,
        usage_path: Path | None = None,
        batch_size: int = 32,
        max_retries: int = 3,
    ) -> None:
        resolved_api_key = os.getenv("OPENAI_API_KEY") if api_key is None else api_key
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        if model != "text-embedding-3-small":
            raise ValueError("benchmark protocol requires text-embedding-3-small")
        if dimensions != 1536:
            raise ValueError("benchmark protocol requires 1536 embedding dimensions")
        if batch_size < 1 or batch_size > 2048:
            raise ValueError("batch_size must be between 1 and 2048")
        self.api_key = resolved_api_key
        self.model = model
        self.dimensions = dimensions
        self.usage_path = usage_path
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._usage_lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors, _usage = self.embed_with_usage(texts)
        return vectors

    def embed_with_usage(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingUsage]:
        output: list[list[float]] = []
        total_usage = EmbeddingUsage()
        for offset in range(0, len(texts), self.batch_size):
            vectors, usage = self._embed_batch(texts[offset:offset + self.batch_size])
            output.extend(vectors)
            total_usage = total_usage.plus(usage)
        return output, total_usage

    def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingUsage]:
        payload = json.dumps({
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self.dimensions,
        }).encode()
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                "https://api.openai.com/v1/embeddings",
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.load(response)
                vectors = self._validate_vectors(body, len(texts))
                raw_usage = body.get("usage") or {}
                usage = EmbeddingUsage(
                    prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
                    total_tokens=int(raw_usage.get("total_tokens") or 0),
                    requests=1,
                )
                self._record_usage(usage)
                return vectors, usage
            except urllib.error.HTTPError as error:
                retryable = error.code in (408, 409, 429) or error.code >= 500
                if not retryable or attempt == self.max_retries:
                    message = _http_error_message(error)
                    raise RuntimeError(f"OpenAI embedding request failed ({error.code}): {message}") from error
                time.sleep(_retry_delay_seconds(error, attempt))
        raise RuntimeError("OpenAI embedding retries exhausted")

    def _validate_vectors(self, body: dict[str, Any], expected: int) -> list[list[float]]:
        data = sorted(body.get("data") or [], key=lambda item: int(item.get("index") or 0))
        vectors = [normalize(item.get("embedding") or []) for item in data]
        if len(vectors) != expected or any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError(
                f"OpenAI returned an invalid embedding batch: {len(vectors)} vectors, expected {expected}x{self.dimensions}"
            )
        return vectors

    def _record_usage(self, usage: EmbeddingUsage) -> None:
        if self.usage_path is None:
            return
        event = {
            "provider": "openai",
            "model": self.model,
            "dimensions": self.dimensions,
            "inputTokens": usage.prompt_tokens,
            "totalTokens": usage.total_tokens,
            "estimatedCostUsd": usage.prompt_tokens * 0.02 / 1_000_000,
            "recordedAt": datetime.now(timezone.utc).isoformat(),
        }
        with self._usage_lock:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def normalize(vector: Iterable[float]) -> list[float]:
    values = [float(value) for value in vector]
    magnitude = math.sqrt(sum(value * value for value in values))
    return values if magnitude == 0 else [value / magnitude for value in values]


def _http_error_message(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read().decode())
        return str((body.get("error") or {}).get("message") or error.reason)
    except Exception:
        return str(error.reason)


def _retry_delay_seconds(error: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after)) + 0.25
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                return max(0.0, retry_at.timestamp() - time.time()) + 0.25
            except (TypeError, ValueError):
                pass
    return 0.5 * (2 ** attempt)
