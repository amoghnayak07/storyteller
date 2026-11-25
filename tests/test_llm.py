"""Tests for the API wrapper: request shape, retries, and error mapping."""

from __future__ import annotations

import anthropic
import httpx2
import pytest
from conftest import make_message
from pydantic import BaseModel

import llm


class Pet(BaseModel):
    name: str
    legs: int


# --------------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------------- #


def test_structured_call_sends_model_effort_cache_and_schema(fake_client):
    client = fake_client(make_message('{"name": "Bob", "legs": 4}'))

    llm.generate_structured("SYSTEM", "USER", Pet, effort="high")

    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["system"] == [
        {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["messages"] == [{"role": "user", "content": "USER"}]
    assert call["output_config"]["effort"] == "high"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"]["properties"].keys() == {"name", "legs"}


def test_no_sampling_parameters_are_sent(fake_client):
    """Haiku 5.5 rejects temperature/top_p/top_k; we must never send them."""
    client = fake_client(make_message("Once..."), make_message('{"name": "Bob", "legs": 4}'))

    llm.generate_text("S", "U")
    llm.generate_structured("S", "U", Pet)

    for call in client.messages.calls:
        assert not {"temperature", "top_p", "top_k"} & call.keys()


def test_text_call_has_no_output_format(fake_client):
    client = fake_client(make_message("Once upon a teapot..."))

    llm.generate_text("S", "U", effort="medium")

    assert client.messages.calls[0]["output_config"] == {"effort": "medium"}


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


def test_structured_returns_validated_model(fake_client):
    fake_client(make_message('{"name": "Bob", "legs": 4}'))

    result = llm.generate_structured("S", "U", Pet)

    assert result.value == Pet(name="Bob", legs=4)


def test_thinking_blocks_are_skipped(fake_client):
    fake_client(make_message("Pip woke up.", thinking="Let me plan the story..."))

    assert llm.generate_text("S", "U").value == "Pip woke up."


def test_usage_includes_cache_tokens(fake_client):
    fake_client(make_message("Story", cache_read=900, cache_write=0))

    usage = llm.generate_text("S", "U").usage

    assert usage == llm.Usage(input_tokens=10, output_tokens=20, cache_read_tokens=900, calls=1)


# --------------------------------------------------------------------------- #
# Retries
# --------------------------------------------------------------------------- #


def test_structured_retries_once_on_invalid_output(fake_client):
    client = fake_client(make_message('{"name": "Bob"}'), make_message('{"name": "Bob", "legs": 4}'))

    result = llm.generate_structured("S", "U", Pet)

    assert result.value.legs == 4
    assert len(client.messages.calls) == 2
    assert result.usage.calls == 2


def test_structured_gives_up_after_two_invalid_outputs(fake_client):
    fake_client(make_message("not json"), make_message("still not json"))

    with pytest.raises(llm.LLMError, match="Pet"):
        llm.generate_structured("S", "U", Pet)


def test_truncated_output_is_retried(fake_client):
    client = fake_client(
        make_message('{"name": "Bo', stop_reason="max_tokens"),
        make_message('{"name": "Bob", "legs": 4}'),
    )

    assert llm.generate_structured("S", "U", Pet).value.name == "Bob"
    assert len(client.messages.calls) == 2


def test_text_retries_once_when_empty_then_fails(fake_client):
    fake_client(make_message("", stop_reason="max_tokens"), make_message(""))

    with pytest.raises(llm.LLMError):
        llm.generate_text("S", "U")


def test_refusal_raises_without_retrying(fake_client):
    client = fake_client(make_message("", stop_reason="refusal"))

    with pytest.raises(llm.LLMRefusalError):
        llm.generate_structured("S", "U", Pet)
    assert len(client.messages.calls) == 1


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    return cls("boom", response=httpx2.Response(status, request=_REQUEST), body=None)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (anthropic.APIConnectionError(request=_REQUEST), "internet connection"),
        (_status_error(anthropic.AuthenticationError, 401), "API key was rejected"),
        (_status_error(anthropic.InternalServerError, 500), r"error \(500\)"),
    ],
)
def test_api_errors_become_friendly_llm_errors(fake_client, error, message):
    fake_client(error)

    with pytest.raises(llm.LLMError, match=message):
        llm.generate_text("S", "U")


def test_missing_api_key_is_a_friendly_error(monkeypatch):
    llm.set_client(None)
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(llm.LLMError, match="ANTHROPIC_API_KEY is not set"):
        llm.get_client()
