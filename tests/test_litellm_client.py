"""Contract tests for the provider boundary and parsing logic."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from runtime.contracts import Message, ModelClientError, ToolDefinition
from runtime.litellm_client import LiteLLMClient, parse_litellm_response


def test_parser_extracts_reasoning_calls_content_and_usage() -> None:
    response = parse_litellm_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "working",
                        "reasoning_content": "private provider reasoning",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Shanghai"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
    )

    assert response.content == "working"
    assert response.reasoning == "private provider reasoning"
    assert response.tool_calls[0].arguments == {"city": "Shanghai"}
    assert response.usage.total_tokens == 14


def test_parser_preserves_malformed_arguments_as_recoverable_call() -> None:
    response = parse_litellm_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "calculator", "arguments": "not-json"}}
                        ]
                    }
                }
            ]
        }
    )

    call = response.tool_calls[0]
    assert call.arguments is None
    assert call.parse_error
    assert call.id == "tool_call_0"


def test_parser_rejects_missing_choices() -> None:
    with pytest.raises(ModelClientError, match="no choices"):
        parse_litellm_response({"choices": []})


async def test_client_serializes_provider_neutral_messages_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured: dict[str, object] = {}

    async def completion(**kwargs: object) -> object:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    client = LiteLLMClient("openai/test", completion=completion)
    result = await client.complete(
        [Message(role="user", content="hello")],
        [ToolDefinition("search", "Search", {"type": "object"})],
    )

    assert result.content == "ok"
    assert captured["model"] == "openrouter/openai/test"
    messages = captured["messages"]
    tools = captured["tools"]
    assert isinstance(messages, Sequence)
    assert isinstance(tools, Sequence)
    assert messages[0] == {"role": "user", "content": "hello"}
    assert tools[0]["function"]["name"] == "search"


async def test_client_classifies_provider_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class ConnectionFailure(Exception):
        pass

    async def completion(**kwargs: object) -> object:
        del kwargs
        raise ConnectionFailure

    client = LiteLLMClient("openai/test", completion=completion)
    with pytest.raises(ModelClientError) as raised:
        await client.complete([], [])
    assert raised.value.retryable
    assert raised.value.code == "model_temporarily_unavailable"
