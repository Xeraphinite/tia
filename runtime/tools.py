"""Tool registration, validation, redaction, and bounded execution."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import BaseModel, ValidationError

from runtime.contracts import JSONObject, JSONValue, ToolDefinition
from runtime.errors import TinyAgentError


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Explicit authority and ownership passed to a tool invocation."""

    user_id: str
    session_id: str


class ToolHandler(Protocol):
    """A JSON-compatible async tool implementation."""

    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        """Execute a validated invocation."""
        ...


class ToolDomainError(Exception):
    """A safe domain failure that the model may correct."""

    def __init__(self, code: str, message: str, *, details: JSONValue | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.details = details


@dataclass(frozen=True, slots=True)
class Tool:
    """A registered tool and its validation contract."""

    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: ToolHandler
    timeout_seconds: float = 10.0
    read_only: bool = True
    idempotent: bool = True
    sensitive_fields: frozenset[str] = frozenset()
    sensitive_result_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("tool timeout must be positive")

    def definition(self) -> ToolDefinition:
        """Return the JSON Schema shown to the model."""
        schema = self.arguments_model.model_json_schema()
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=_as_json_object(schema),
        )


class ToolRegistry:
    """A small explicit registry with no global state."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a uniquely named tool."""
        if not tool.name or not tool.name.replace("_", "a").isalnum():
            raise ValueError("tool names must contain only letters, numbers, and underscores")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by its stable name."""
        return self._tools.get(name)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return model-visible definitions in registration order."""
        return tuple(tool.definition() for tool in self._tools.values())


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """A structured tool result safe to feed back to the model."""

    ok: bool
    content: str
    error_code: str | None = None


class ToolExecutor:
    """Validate and execute registered tools under explicit policy limits."""

    def __init__(self, registry: ToolRegistry, *, timeout_seconds: float, max_output_chars: int):
        self._registry = registry
        self._timeout_seconds = timeout_seconds
        self._max_output_chars = max_output_chars

    async def execute(
        self,
        name: str,
        arguments: JSONObject | None,
        context: ToolContext,
        *,
        parse_error: str | None = None,
    ) -> ToolExecution:
        """Return tool failures as model-actionable structured results."""
        tool = self._registry.get(name)
        if tool is None:
            return self._error("unknown_tool", f"No tool named '{name}' is registered.")
        if parse_error is not None or arguments is None:
            return self._error("invalid_arguments", "Tool arguments were not valid JSON.")

        try:
            validated = tool.arguments_model.model_validate(arguments)
        except ValidationError as exc:
            details: list[JSONValue] = [
                cast(
                    JSONObject,
                    {
                    "location": ".".join(str(part) for part in item["loc"]),
                    "message": item["msg"],
                    },
                )
                for item in exc.errors(include_url=False, include_input=False)
            ]
            return self._error(
                "invalid_arguments",
                "Tool arguments failed Schema validation.",
                details=details,
            )

        normalized = _as_json_object(validated.model_dump(mode="json"))
        try:
            async with asyncio.timeout(min(self._timeout_seconds, tool.timeout_seconds)):
                result = await tool.handler(context, normalized)
            content = json.dumps(
                {"ok": True, "result": result}, ensure_ascii=False, separators=(",", ":")
            )
        except TimeoutError:
            return self._error("tool_timeout", f"Tool '{name}' exceeded its time limit.")
        except asyncio.CancelledError:
            raise
        except TinyAgentError:
            raise
        except ToolDomainError as exc:
            return self._error(exc.code, exc.safe_message, details=exc.details)
        except Exception:
            return self._error("tool_failed", f"Tool '{name}' failed safely.")

        if len(content) > self._max_output_chars:
            return self._error(
                "tool_output_too_large", f"Tool '{name}' returned more data than allowed."
            )
        return ToolExecution(ok=True, content=content)

    @staticmethod
    def _error(code: str, message: str, *, details: JSONValue | None = None) -> ToolExecution:
        payload: JSONObject = {"ok": False, "error": {"code": code, "message": message}}
        error = payload["error"]
        if details is not None and isinstance(error, dict):
            error["details"] = details
        return ToolExecution(
            ok=False,
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            error_code=code,
        )


def redact_arguments(tool: Tool | None, arguments: JSONObject | None) -> JSONObject:
    """Create a trace-safe copy of tool arguments."""
    if arguments is None:
        return {}
    sensitive = tool.sensitive_fields if tool else frozenset(arguments)
    return {
        key: "[REDACTED]" if key in sensitive else value for key, value in arguments.items()
    }


def redact_result_content(tool: Tool | None, content: str) -> str:
    """Redact configured top-level result fields before persistence or model reuse."""
    if tool is None or not tool.sensitive_result_fields:
        return content
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return "[REDACTED]"
    if not isinstance(value, dict):
        return content
    result = value.get("result")
    if isinstance(result, dict):
        for field in tool.sensitive_result_fields:
            if field in result:
                result[field] = "[REDACTED]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _as_json_object(value: object) -> JSONObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("expected a JSON object")
    return cast(JSONObject, value)
