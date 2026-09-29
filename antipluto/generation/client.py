"""
generation/client.py
====================
Provider-agnostic LLM client for Anti-PLUTO's generation pipeline.

Provides a ``BaseGenerationClient`` ABC with pluggable provider backends:

  - ``OpenAIClient``:     OpenAI-compatible APIs (Azure AI Foundry, OpenRouter, etc.)
  - ``AnthropicClient``:  Anthropic Messages API (Claude via Azure AI Foundry)

Each provider implements ``_call_api()`` for a single API call; the base class
provides shared retry logic with exponential backoff, JSON response parsing,
and resource management.

Extensibility
-------------
Users can add custom provider backends by subclassing ``BaseGenerationClient``
and registering with ``register_provider()``:

    >>> from antipluto.generation.client import BaseGenerationClient, register_provider
    >>> class MyCustomClient(BaseGenerationClient):
    ...     def __init__(self, *, api_key, endpoint, **kwargs):
    ...         super().__init__(**kwargs)
    ...         # setup your SDK client here
    ...     def _call_api(self, model, system_prompt, user_prompt, max_tokens, temperature):
    ...         # make the API call, return the text response
    ...         return "..."
    >>> register_provider("my_custom", MyCustomClient)

Then reference ``provider: "my_custom"`` in the YAML config.

JSON Response Parsing
---------------------
To maintain cross-model compatibility, Anti-PLUTO uses a prompt-based structured
output approach: the LLM is instructed to return a JSON array in the user
prompt, and the response is parsed with a robust extractor that handles:

  1. Raw JSON arrays (ideal case).
  2. JSON wrapped in markdown code fences (``\\`\\`\\`json ... \\`\\`\\`).
  3. JSON preceded or followed by model commentary (stripped with regex).

Dependencies
------------
``openai`` is required for ``OpenAIClient``. Install via: ``pip install openai``.
``anthropic`` is required for ``AnthropicClient``. Install via: ``pip install anthropic``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON response extractor (shared by all providers)
# ---------------------------------------------------------------------------

# Matches a JSON array containing at least one object (i.e. starts with [{)
# inside an optional markdown code fence. The (?s) flag makes . match newlines.
# Group 1: fenced array content. Group 2: bare array content.
_JSON_ARRAY_RE = re.compile(
    r"```(?:json|JSON)?\s*(\[\s*\{.*?\])\s*```|(\[\s*\{.*?\])",
    re.DOTALL,
)


def extract_json_array(text: str) -> list[dict] | None:
    """
    Extract the first JSON array from an LLM response string.

    Handles:
    - Raw arrays: ``[{"subject": ..., "body": ...}]``
    - Fenced arrays: `` ```json\\\\n[...]\\\\n``` ``
    - Arrays preceded by model preamble text

    Parameters
    ----------
    text : str
        Raw LLM response content.

    Returns
    -------
    list[dict] or None
        Parsed list of dicts, or ``None`` if no valid array was found.
    """
    if not text:
        return None

    # Try regex extraction first (handles fences + surrounding text).
    for match in _JSON_ARRAY_RE.finditer(text):
        candidate = match.group(1) or match.group(2)
        if candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue

    # Last resort: try the entire response body as JSON.
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and "subject" in parsed and "body" in parsed:
            return [parsed]
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Custom error type for retry control
# ---------------------------------------------------------------------------


class RetryableError(Exception):
    """Transient API error that should be retried (rate limit, server error, timeout)."""

    def __init__(
        self, message: str, *, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# BaseGenerationClient
# ---------------------------------------------------------------------------


class BaseGenerationClient(ABC):
    """
    Abstract base class for LLM generation clients.

    Subclasses must implement ``_call_api()`` which makes a single API call
    and returns the raw text content. The base class handles retry logic,
    JSON parsing, and resource management.

    Parameters
    ----------
    timeout : float
        HTTP request timeout in seconds.
    max_retries : int
        Maximum number of retry attempts per API call.
    retry_delay : float
        Base delay in seconds between retries (doubled on each retry).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = "",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/") if endpoint else ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def close(self) -> None:
        """Release underlying resources. Override in subclasses if needed."""

    def __enter__(self) -> "BaseGenerationClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.9,
    ) -> list[dict] | None:
        """
        Call the LLM API and return parsed email records.

        Retries on transient errors (rate-limits, server errors) using
        exponential backoff. Returns ``None`` if all retries are exhausted
        or the response cannot be parsed as a JSON array.

        Parameters
        ----------
        model : str
            Model deployment name (e.g. ``"gpt-5-mini"``).
        system_prompt : str
            System prompt text.
        user_prompt : str
            User prompt text containing the seed emails.
        max_tokens : int
            Maximum tokens in the completion response.
        temperature : float
            Sampling temperature.

        Returns
        -------
        list[dict] or None
            Parsed list of ``{"subject": str, "body": str}`` dicts,
            or ``None`` on failure.
        """
        delay = self.retry_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                content = self._call_api(
                    model, system_prompt, user_prompt, max_tokens, temperature,
                )

                if not content:
                    logger.warning(
                        "client › [%s] Empty content in response. Retry %d/%d.",
                        model, attempt, self.max_retries,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                parsed = extract_json_array(content)
                if parsed is None:
                    logger.warning(
                        "client › [%s] Could not parse JSON array from response "
                        "(attempt %d/%d). Content preview: %.120s",
                        model, attempt, self.max_retries, content,
                    )
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    return None

                logger.debug(
                    "client › [%s] Batch OK — %d records parsed.",
                    model, len(parsed),
                )
                return parsed

            except RetryableError as exc:
                retry_after = exc.retry_after or delay
                logger.warning(
                    "client › [%s] %s. Waiting %.1fs before retry %d/%d.",
                    model, exc, retry_after, attempt, self.max_retries,
                )
                time.sleep(retry_after)
                delay *= 2

            except Exception as exc:
                logger.warning(
                    "client › [%s] API error (attempt %d/%d): %s",
                    model, attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2

        logger.error(
            "client › [%s] All %d retry attempts exhausted. Skipping batch.",
            model, self.max_retries,
        )
        return None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _call_api(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Make a single API call and return the raw text content.

        Subclasses must translate provider-specific transient errors
        (rate limits, server errors, timeouts) into ``RetryableError``.
        All other exceptions propagate to the base class retry handler.

        Parameters
        ----------
        model : str
            Model deployment name.
        system_prompt : str
            System-level instruction.
        user_prompt : str
            User-level prompt with seed emails.
        max_tokens : int
            Maximum response tokens.
        temperature : float
            Sampling temperature.

        Returns
        -------
        str
            Raw text content from the LLM response.

        Raises
        ------
        RetryableError
            For transient failures that should be retried.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


class OpenAIClient(BaseGenerationClient):
    """
    Client for OpenAI-compatible chat completions APIs.

    Works with any provider that exposes an OpenAI-compatible surface:
    Azure AI Foundry, OpenRouter, direct OpenAI API, etc.

    Parameters
    ----------
    api_key : str
        API key for the provider.
    endpoint : str
        Base URL for the API endpoint.
    timeout : float
        HTTP request timeout in seconds.
    max_retries : int
        Maximum number of retry attempts per API call.
    retry_delay : float
        Base delay in seconds between retries.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            endpoint=endpoint,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAIClient requires the 'openai' package. "
                "Install via: pip install openai"
            )
        self._openai = openai
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=endpoint.rstrip("/"),
            timeout=timeout,
            max_retries=0,  # We handle retries manually for finer control.
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def _call_api(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            extra: dict = {}
            m_lower = model.lower()
            is_reasoning_model = any(k in m_lower for k in ("gpt-5", "o1", "o3", "o4"))

            # Only include temperature when it differs from 1.0 and is supported.
            # GPT-5 and o-series reasoning models reject any custom temperature.
            if temperature != 1.0 and not is_reasoning_model:
                extra["temperature"] = temperature

            # Disable internal reasoning tokens for OpenAI reasoning models
            # (GPT-5, o1, o3) to save credits and speed up responses.
            if is_reasoning_model:
                extra["reasoning_effort"] = "minimal"

            # Mistral's Azure endpoint and OpenRouter models reject or ignore
            # `max_completion_tokens` (OpenAI-style) and require `max_tokens`.
            # All other Azure OpenAI models use `max_completion_tokens`.
            if "mistral" in model.lower() or "openrouter" in self.endpoint.lower() or "/" in model:
                extra["max_tokens"] = max_tokens
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **extra,
                )
            else:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=max_tokens,
                    **extra,
                )

            return response.choices[0].message.content or ""

        except self._openai.RateLimitError as exc:
            retry_after = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after = float(
                    exc.response.headers.get("Retry-After", self.retry_delay)
                )
            raise RetryableError(
                "Rate limited (429)", retry_after=retry_after,
            ) from exc

        except self._openai.InternalServerError as exc:
            raise RetryableError("Server error (5xx)") from exc

        except self._openai.APITimeoutError as exc:
            raise RetryableError("Request timed out") from exc


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------


class AnthropicClient(BaseGenerationClient):
    """
    Client for the Anthropic Messages API (Claude via Azure AI Foundry).

    Uses the ``anthropic.AnthropicFoundry`` SDK which wraps the Anthropic
    Messages API surface exposed by Azure AI Foundry. Authenticates via
    an API key from the Foundry portal.

    Parameters
    ----------
    api_key : str
        API key from the Azure AI Foundry portal.
    endpoint : str
        Anthropic API endpoint (e.g. the Azure AI Foundry Anthropic endpoint).
    timeout : float
        HTTP request timeout in seconds.
    max_retries : int
        Maximum number of retry attempts per API call.
    retry_delay : float
        Base delay in seconds between retries.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            endpoint=endpoint,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package. "
                "Install via: pip install anthropic"
            )
        self._anthropic = anthropic
        self._client = anthropic.AnthropicFoundry(
            api_key=api_key,
            base_url=endpoint.rstrip("/"),
            timeout=timeout,
            max_retries=0,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def _call_api(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            extra: dict = {}
            if temperature != 1.0:
                extra["temperature"] = temperature

            message = self._client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                thinking={"type": "disabled"},
                **extra,
            )

            # Anthropic returns a list of content blocks (e.g. ThinkingBlock, TextBlock).
            # Extract and join all text blocks.
            text_chunks: list[str] = []
            for block in message.content:
                if getattr(block, "type", "") == "text" or hasattr(block, "text"):
                    text_chunks.append(getattr(block, "text", ""))
            return "".join(text_chunks)

        except self._anthropic.RateLimitError as exc:
            retry_after = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after = float(
                    exc.response.headers.get("Retry-After", self.retry_delay)
                )
            raise RetryableError(
                "Rate limited (429)", retry_after=retry_after,
            ) from exc

        except self._anthropic.InternalServerError as exc:
            raise RetryableError("Server error (5xx)") from exc

        except self._anthropic.APITimeoutError as exc:
            raise RetryableError("Request timed out") from exc


# ---------------------------------------------------------------------------
# Provider registry & factory
# ---------------------------------------------------------------------------


_PROVIDERS: dict[str, type[BaseGenerationClient]] = {
    "openai": OpenAIClient,
    "openrouter": OpenAIClient,
    "anthropic": AnthropicClient,
}


def register_provider(name: str, cls: type[BaseGenerationClient]) -> None:
    """
    Register a custom provider implementation.

    Parameters
    ----------
    name : str
        Provider identifier to use in the YAML config.
    cls : type[BaseGenerationClient]
        A subclass of ``BaseGenerationClient``.

    Raises
    ------
    TypeError
        If ``cls`` is not a subclass of ``BaseGenerationClient``.
    """
    if not issubclass(cls, BaseGenerationClient):
        raise TypeError(
            f"{cls.__name__} must be a subclass of BaseGenerationClient."
        )
    _PROVIDERS[name] = cls


def create_client(
    provider: str,
    provider_config: dict,
    *,
    timeout: float = 120.0,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    api_key_override: str | None = None,
) -> BaseGenerationClient:
    """
    Factory: create a generation client for the given provider.

    API key resolution order:
      1. ``api_key_override`` (from CLI ``--api-key`` flag)
      2. ``provider_config["api_key"]`` (direct value in YAML)
      3. Environment variable named in ``provider_config["api_key_env"]``

    Parameters
    ----------
    provider : str
        Provider name (e.g. ``"openai"``, ``"anthropic"``).
    provider_config : dict
        Provider configuration dict from YAML (must contain ``endpoint``).
    timeout : float
        HTTP request timeout in seconds.
    max_retries : int
        Maximum retry attempts.
    retry_delay : float
        Base delay between retries.
    api_key_override : str or None
        CLI-level API key override applied to all providers.

    Returns
    -------
    BaseGenerationClient
        Configured client instance.

    Raises
    ------
    ValueError
        If the provider is unknown or no endpoint is configured.
    """
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available providers: {sorted(_PROVIDERS.keys())}. "
            f"To add a custom provider, register it via register_provider()."
        )

    # Resolve API key: CLI override > config api_key > env var from api_key_env
    api_key = (
        api_key_override
        or provider_config.get("api_key")
        or os.environ.get(provider_config.get("api_key_env", ""), "")
        or None
    )

    endpoint = provider_config.get("endpoint", "")
    if not endpoint:
        raise ValueError(
            f"No endpoint configured for provider '{provider}'. "
            f"Set endpoint in the providers section of your config YAML."
        )

    if not api_key:
        env_name = provider_config.get("api_key_env", "")
        hint = f" Set the {env_name} environment variable," if env_name else ""
        raise ValueError(
            f"No API key found for provider '{provider}'.{hint} "
            f"add 'api_key' to provider config, or pass --api-key."
        )

    return cls(
        api_key=api_key,
        endpoint=endpoint,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
