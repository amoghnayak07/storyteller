"""The only module that talks to the Anthropic API.

Every node in the pipeline goes through one of two functions:

- ``generate_text``: free-form prose (storyteller, reviser).
- ``generate_structured``: a validated Pydantic object (intake, planner, judge),
  using native structured outputs (``output_config.format``).

Design notes
------------
- **Model**: ``claude-haiku-5-5`` for every call.
- **No sampling parameters.** Haiku 5.5 does not accept ``temperature`` /
  ``top_p`` / ``top_k``. Nodes are differentiated by their prompt and by
  ``effort`` (e.g. the judge runs at ``high``), not by temperature.
- **Prompt caching.** Each node's system prompt is fixed, so it is sent as a
  block with an ephemeral (5-minute) cache breakpoint. Prompts under the
  model's minimum cacheable length are simply processed uncached.
- **Why ``messages.create`` and not ``messages.parse``.** ``parse()`` validates
  the JSON inside its response hook, so a refusal or a truncated response
  surfaces as a bare ``ValidationError`` and we lose the ``stop_reason``. We
  send the same native structured-output format ourselves, check the stop
  reason first, then validate.
- **Thinking counts toward ``max_tokens``.** Adaptive thinking is on by default,
  so limits are generous enough that thinking can't crowd out the answer.
- **Retries.** The SDK already retries rate limits, overloads and 5xx errors.
  On top of that we retry once when the output is malformed or truncated, then
  fail with a friendly ``LLMError`` that the CLI prints without a traceback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Generic, Literal, TypeVar

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

import user_constants as ui
from constants import DEFAULT_MAX_TOKENS, MAX_ATTEMPTS, MODEL

Effort = Literal["low", "medium", "high"]

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Errors and results
# --------------------------------------------------------------------------- #


class LLMError(Exception):
    """A failure with a message that is safe to show the user as-is."""


class LLMRefusalError(LLMError):
    """The model declined to answer. The guardrails treat this as a block."""


@dataclass(frozen=True)
class Usage:
    """Token accounting for one or more calls (shown in ``--verbose`` and evals)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            calls=self.calls + other.calls,
        )

    @classmethod
    def from_response(cls, usage: Any) -> Usage:
        return cls(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
            calls=1,
        )


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    value: T
    usage: Usage = field(default_factory=Usage)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

_client: Any = None


def get_client() -> Any:
    """Return the shared Anthropic client, creating it on first use."""
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise LLMError(ui.MISSING_KEY_ERROR)
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def set_client(client: Any) -> None:
    """Swap the client (tests pass a fake; ``None`` resets to lazy creation)."""
    global _client
    _client = client


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def generate_text(
    system: str,
    user: str,
    *,
    effort: Effort = "medium",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> LLMResult[str]:
    """Free-form generation. Retries once on an empty or truncated answer."""
    usage = Usage()
    for _ in range(MAX_ATTEMPTS):
        response = _call(system, user, effort=effort, max_tokens=max_tokens, output_format=None)
        usage += Usage.from_response(response.usage)
        text = _response_text(response).strip()
        if response.stop_reason != "max_tokens" and text:
            return LLMResult(value=text, usage=usage)
    raise LLMError(ui.CUT_OFF_ERROR)


def generate_structured(
    system: str,
    user: str,
    schema: type[ModelT],
    *,
    effort: Effort = "medium",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> LLMResult[ModelT]:
    """Generation constrained to ``schema``. Retries once on malformed output.

    The API guarantees the JSON matches the schema's *shape*; Pydantic then
    enforces the rules the grammar can't express (e.g. "exactly 5 beats").
    """
    usage = Usage()
    for _ in range(MAX_ATTEMPTS):
        response = _call(system, user, effort=effort, max_tokens=max_tokens, output_format=schema)
        usage += Usage.from_response(response.usage)
        if response.stop_reason == "max_tokens":
            continue
        try:
            value = schema.model_validate_json(_response_text(response))
        except ValidationError:
            continue
        return LLMResult(value=value, usage=usage)
    raise LLMError(ui.UNREADABLE_ERROR.format(schema=schema.__name__))


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _output_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {"type": "json_schema", "schema": anthropic.transform_schema(schema)}


def _call(
    system: str,
    user: str,
    *,
    effort: Effort,
    max_tokens: int,
    output_format: type[BaseModel] | None,
) -> Any:
    output_config: dict[str, Any] = {"effort": effort}
    if output_format is not None:
        output_config["format"] = _output_format(output_format)

    try:
        response = get_client().messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
    except anthropic.AuthenticationError as exc:
        raise LLMError(ui.BAD_KEY_ERROR) from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError(ui.CONNECTION_ERROR) from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(ui.API_STATUS_ERROR.format(status=exc.status_code)) from exc

    if response.stop_reason == "refusal":
        raise LLMRefusalError(ui.REFUSAL_ERROR)
    return response


def _response_text(response: Any) -> str:
    """Join the text blocks, skipping thinking blocks (which may come first)."""
    return "".join(block.text for block in response.content if block.type == "text")
