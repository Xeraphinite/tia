"""Narrow LiteLLM/OpenRouter adapter and response parser."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import cast

import litellm
from dotenv import load_dotenv

from runtime.contracts import (
    JSONObject,
    Message,
    ModelClientError,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)

type CompletionFunction = Callable[..., Awaitable[object]]


class LiteLLMClient:
    """Call OpenRouter through LiteLLM and immediately normalize its response."""

    def __init__(
        self,
        model: str,
        *,
        request_timeout_seconds: float = 45.0,
        completion: CompletionFunction | None = None,
    ) -> None:
        self._model = model if model.startswith("openrouter/") else f"openrouter/{model}"
        self._request_timeout_seconds = request_timeout_seconds
        self._completion = completion or cast(CompletionFunction, litellm.acompletion)

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """Make one model request and parse all provider-specific objects at this boundary."""
        configure_openrouter_credentials()
        try:
            raw = await self._completion(
                model=self._model,
                messages=[_message_payload(message) for message in messages],
                tools=[_tool_payload(tool) for tool in tools],
                tool_choice="auto",
                timeout=self._request_timeout_seconds,
            )
            return parse_litellm_response(raw)
        except ModelClientError:
            raise
        except Exception as exc:
            name = type(exc).__name__.casefold()
            detail = str(exc).casefold()
            if any(
                marker in detail
                for marker in (
                    "context length",
                    "context window",
                    "maximum context",
                    "too many tokens",
                )
            ):
                raise ModelClientError(
                    "context_overflow",
                    "The model context exceeded the provider limit.",
                    retryable=False,
                ) from None
            retryable = any(
                marker in name for marker in ("timeout", "rate", "connection", "serviceunavailable")
            )
            code = "model_temporarily_unavailable" if retryable else "model_provider_error"
            raise ModelClientError(
                code,
                "The model provider is temporarily unavailable."
                if retryable
                else "The model provider rejected the request.",
                retryable=retryable,
            ) from None


def configure_openrouter_credentials() -> None:
    """Copy the documented local key into OpenRouter's process-local variable."""
    load_dotenv()
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    source = os.environ.get("OPENAI_API_KEY")
    if not source:
        raise ModelClientError(
            "missing_credentials",
            "OPENAI_API_KEY is not configured for OpenRouter.",
            retryable=False,
        )
    os.environ["OPENROUTER_API_KEY"] = source


def parse_litellm_response(raw: object) -> ModelResponse:
    """Extract final text, reasoning, tool calls, and usage from a LiteLLM response."""
    choices = _sequence(_field(raw, "choices"))
    if not choices:
        raise ModelClientError(
            "invalid_model_response", "The model returned no choices.", retryable=False
        )
    message = _field(choices[0], "message")
    if message is None:
        raise ModelClientError(
            "invalid_model_response", "The model returned no message.", retryable=False
        )

    content_value = _field(message, "content")
    content = content_value if isinstance(content_value, str) else None
    reasoning = _reasoning(message)
    calls = tuple(_parse_tool_call(value, index) for index, value in enumerate(
        _sequence(_field(message, "tool_calls"))
    ))
    return ModelResponse(
        content=content,
        tool_calls=calls,
        reasoning=reasoning,
        usage=_usage(_field(raw, "usage")),
    )


def _parse_tool_call(value: object, index: int) -> ToolCall:
    function = _field(value, "function")
    call_id_value = _field(value, "id")
    name_value = _field(function, "name")
    arguments_value = _field(function, "arguments")
    call_id = call_id_value if isinstance(call_id_value, str) else f"tool_call_{index}"
    name = name_value if isinstance(name_value, str) else ""
    if isinstance(arguments_value, str):
        raw_arguments = arguments_value
    elif isinstance(arguments_value, Mapping):
        raw_arguments = json.dumps(dict(arguments_value), ensure_ascii=False)
    else:
        raw_arguments = ""
    try:
        decoded = json.loads(raw_arguments)
        if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
            raise ValueError
        arguments = cast(JSONObject, decoded)
        parse_error = None
    except (json.JSONDecodeError, ValueError, TypeError):
        arguments = None
        parse_error = "arguments must be a JSON object"
    return ToolCall(
        id=call_id,
        name=name,
        raw_arguments=raw_arguments,
        arguments=arguments,
        parse_error=parse_error,
    )


def _reasoning(message: object) -> str | None:
    for name in ("reasoning_content", "reasoning"):
        value = _field(message, name)
        if isinstance(value, str) and value:
            return value
    provider_fields = _field(message, "provider_specific_fields")
    for name in ("reasoning_content", "reasoning"):
        value = _field(provider_fields, name)
        if isinstance(value, str) and value:
            return value
    return None


def _usage(value: object) -> Usage:
    return Usage(
        prompt_tokens=_integer(_field(value, "prompt_tokens")),
        completion_tokens=_integer(_field(value, "completion_tokens")),
        total_tokens=_integer(_field(value, "total_tokens")),
    )


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    result: object = getattr(value, name, None)
    return result


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _tool_payload(tool: ToolDefinition) -> JSONObject:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _message_payload(message: Message) -> JSONObject:
    payload: JSONObject = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        payload["name"] = message.name
    return payload
