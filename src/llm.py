"""Provider-independent, opt-in LLM support for research-assistance tasks.

This module intentionally has no database or research-policy authority.  It
only sends requests, applies bounded retries, parses responses, and validates
structured JSON.  Callers remain responsible for provenance and review.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.
    def load_dotenv() -> bool:
        return False


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
REMOTE_PROVIDER_NAMES = {"openai", "openai-compatible", "deepseek"}
TModel = TypeVar("TModel", bound=BaseModel)


class LLMError(RuntimeError):
    """A safe, non-secret-bearing LLM request or response error."""


@dataclass(frozen=True)
class LLMMessage:
    """One chat message sent to a provider."""

    role: str
    content: str


@dataclass(frozen=True)
class LLMRequest:
    """A provider-neutral completion request."""

    messages: tuple[LLMMessage, ...]
    temperature: float = 0.0
    json_mode: bool = False


@dataclass(frozen=True)
class LLMResponse:
    """Normalized provider response with usage metadata when supplied."""

    content: str
    provider: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Minimal interface implemented by any remote text-generation provider."""

    provider_name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one completion or raise :class:`LLMError`."""


@dataclass(frozen=True)
class LLMConfiguration:
    """Non-secret configuration loaded from the supported environment variables."""

    provider: str = "placeholder"
    model: str = "placeholder-model"
    api_key: str = ""
    base_url: str = DEFAULT_OPENAI_BASE_URL

    @classmethod
    def from_environment(cls) -> "LLMConfiguration":
        load_dotenv()
        base_url = os.getenv("LLM_BASE_URL", DEFAULT_OPENAI_BASE_URL).strip()
        return cls(
            provider=os.getenv("LLM_PROVIDER", "placeholder").strip().lower() or "placeholder",
            model=os.getenv("LLM_MODEL", "placeholder-model").strip() or "placeholder-model",
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            base_url=(base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/"),
        )

    @property
    def remote_enabled(self) -> bool:
        return self.provider in REMOTE_PROVIDER_NAMES and bool(self.api_key)


class OpenAICompatibleProvider:
    """Provider for OpenAI and services implementing its chat-completions API."""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        api_key: str,
        base_url: str,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.25,
    ) -> None:
        if not api_key:
            raise LLMError("LLM_API_KEY is required for a remote LLM provider")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.provider_name = provider_name
        self.model = model
        self._api_key = api_key
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": request.temperature,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        request_data = json.dumps(payload).encode("utf-8")
        last_error: LLMError | None = None
        for attempt in range(self._max_attempts):
            try:
                http_request = urllib.request.Request(
                    self._endpoint,
                    data=request_data,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(http_request, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return self._parse_response(data)
            except ValueError:
                last_error = LLMError("LLM provider configuration is invalid")
                retryable = False
            except urllib.error.HTTPError as exc:
                last_error = LLMError(f"LLM provider returned HTTP {exc.code}")
                retryable = exc.code == 429 or exc.code >= 500
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = LLMError(f"LLM provider request failed: {type(exc).__name__}")
                retryable = True
            except OSError as exc:
                last_error = LLMError(f"LLM provider request failed: {type(exc).__name__}")
                retryable = True

            if not retryable or attempt == self._max_attempts - 1:
                break
            time.sleep(self._retry_delay_seconds * (2**attempt))

        raise last_error or LLMError("LLM provider request failed")

    def _parse_response(self, data: Any) -> LLMResponse:
        if not isinstance(data, dict):
            raise LLMError("LLM provider response was not a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMError("LLM provider response did not contain a completion choice")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM provider response did not contain message content")
        raw_usage = data.get("usage", {})
        usage = {
            str(key): int(value)
            for key, value in raw_usage.items()
            if isinstance(value, int)
        } if isinstance(raw_usage, dict) else {}
        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=str(data.get("model") or self.model),
            usage=usage,
        )


class LLMClient:
    """Configured provider client with centralized structured-response handling."""

    def __init__(self, provider: LLMProvider | None = None, configuration: LLMConfiguration | None = None) -> None:
        self.configuration = configuration or LLMConfiguration.from_environment()
        self.provider = provider
        if self.provider is None and self.configuration.remote_enabled:
            self.provider = OpenAICompatibleProvider(
                provider_name=self.configuration.provider,
                model=self.configuration.model,
                api_key=self.configuration.api_key,
                base_url=self.configuration.base_url,
            )
        self.last_response: LLMResponse | None = None

    @property
    def available(self) -> bool:
        return self.provider is not None

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name if self.provider else self.configuration.provider

    @property
    def model(self) -> str:
        return self.provider.model if self.provider else self.configuration.model

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.provider is None:
            raise LLMError("No remote LLM provider is configured")
        response = self.provider.complete(request)
        self.last_response = response
        return response

    def complete_json(self, request: LLMRequest, response_model: type[TModel]) -> TModel:
        """Request JSON and validate it against the caller's Pydantic model."""

        if not request.json_mode:
            request = LLMRequest(
                messages=request.messages,
                temperature=request.temperature,
                json_mode=True,
            )
        response = self.complete(request)
        try:
            payload = json.loads(_strip_json_fence(response.content))
        except json.JSONDecodeError as exc:
            raise LLMError("LLM provider returned invalid JSON") from exc
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            raise LLMError(f"LLM provider JSON failed validation: {exc}") from exc

    def metadata(self) -> dict[str, Any]:
        """Return safe response metadata suitable for CLI diagnostics."""

        response = self.last_response
        return {
            "provider": response.provider if response else self.provider_name,
            "model": response.model if response else self.model,
            "usage": response.usage if response else {},
            "remote_available": self.available,
        }


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped
