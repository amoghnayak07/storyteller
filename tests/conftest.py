"""Shared test fixtures. Nothing here touches the network or needs an API key."""

from __future__ import annotations

from typing import Any

import pytest
from anthropic.types import Message

import constants
import llm


def make_message(
    text: str = "",
    *,
    stop_reason: str = "end_turn",
    thinking: str | None = None,
    cache_read: int = 0,
    cache_write: int = 0,
) -> Message:
    """Build a real SDK ``Message`` so tests exercise the same attribute access as production."""
    content: list[dict[str, Any]] = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking, "signature": "sig"})
    if text:
        content.append({"type": "text", "text": text})
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": constants.MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
            },
        }
    )


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    """Stands in for ``anthropic.Anthropic``; replays queued responses or exceptions."""

    def __init__(self, *responses: Any) -> None:
        self.messages = FakeMessages(list(responses))


@pytest.fixture
def fake_client():
    """Install a FakeClient: ``client = fake_client(msg1, msg2, ...)``."""

    def install(*responses: Any) -> FakeClient:
        client = FakeClient(*responses)
        llm.set_client(client)
        return client

    yield install
    llm.set_client(None)
