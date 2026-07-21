"""Provider-neutral model contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]
type Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A normalized tool call produced by a model."""

    id: str
    name: str
    raw_arguments: str
    arguments: JSONObject | None
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class Message:
    """One provider-neutral context message."""

    role: Role
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """The model-visible portion of a registered tool."""

    name: str
    description: str
    input_schema: JSONObject


@dataclass(frozen=True, slots=True)
class Usage:
    """Normalized token usage when supplied by the provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A parsed model decision: final text, tool calls, or both."""

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: str | None = None
    usage: Usage = Usage()


class ModelClient(Protocol):
    """The sole model boundary consumed by the agent loop."""

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """Return one normalized model decision."""
        ...


class ModelClientError(Exception):
    """A safe, classified provider failure."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.retryable = retryable
