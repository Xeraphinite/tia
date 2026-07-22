"""Unit tests for tool contracts and built-in implementations."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel, ConfigDict

from runtime.builtin_tools import InMemoryTodoStore, create_builtin_registry
from runtime.contracts import JSONObject, JSONValue
from runtime.tools import Tool, ToolContext, ToolExecutor, ToolRegistry

pytestmark = pytest.mark.unit


async def test_registry_exposes_name_description_and_json_schema() -> None:
    registry = create_builtin_registry()

    definitions = {definition.name: definition for definition in registry.definitions()}

    assert set(definitions) == {"calculator", "search", "todo", "get_weather"}
    calculator = definitions["calculator"]
    assert calculator.description
    assert calculator.input_schema["type"] == "object"
    assert calculator.input_schema["required"] == ["expression"]
    todo = registry.get("todo")
    assert todo is not None
    assert not todo.read_only and not todo.idempotent
    assert todo.timeout_seconds == 10


def test_registry_rejects_duplicates_and_invalid_names() -> None:
    registry = create_builtin_registry()
    calculator = registry.get("calculator")
    assert calculator is not None
    with pytest.raises(ValueError, match="already registered"):
        registry.register(calculator)

    invalid = Tool(
        name="bad name",
        description="invalid",
        arguments_model=calculator.arguments_model,
        handler=calculator.handler,
    )
    with pytest.raises(ValueError, match="tool names"):
        ToolRegistry().register(invalid)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').getcwd()", "1 / 0", "-" * 60 + "1", "10**101"],
)
async def test_calculator_executes_arithmetic_and_classifies_invalid_expressions(
    expression: str,
) -> None:
    executor = ToolExecutor(create_builtin_registry(), timeout_seconds=1, max_output_chars=1000)
    context = ToolContext(user_id="u", session_id="s")

    success = await executor.execute("calculator", {"expression": "(17 * 23) + 5"}, context)
    unsafe = await executor.execute("calculator", {"expression": expression}, context)

    assert success.ok
    assert json.loads(success.content)["result"]["value"] == 396
    assert not unsafe.ok
    assert unsafe.error_code == "invalid_expression"


async def test_schema_validation_and_unknown_tool_are_structured() -> None:
    executor = ToolExecutor(create_builtin_registry(), timeout_seconds=1, max_output_chars=1000)
    context = ToolContext(user_id="u", session_id="s")

    invalid = await executor.execute(
        "get_weather", {"location": "Paris", "extra": 1}, context
    )
    unknown = await executor.execute("missing", {}, context)
    malformed = await executor.execute("get_weather", None, context, parse_error="bad json")

    assert invalid.error_code == "invalid_arguments"
    assert json.loads(invalid.content)["error"]["details"]
    assert unknown.error_code == "unknown_tool"
    assert malformed.error_code == "invalid_arguments"


async def test_search_and_weather_are_deterministic() -> None:
    executor = ToolExecutor(create_builtin_registry(), timeout_seconds=1, max_output_chars=5000)
    context = ToolContext(user_id="u", session_id="s")

    first = await executor.execute("search", {"query": "weather", "limit": 1}, context)
    second = await executor.execute("get_weather", {"location": "Shanghai"}, context)
    third = await executor.execute("get_weather", {"location": "Shanghai"}, context)

    assert json.loads(first.content)["result"][0]["title"] == "Weather operations guide"
    assert second.content == third.content


async def test_weather_accepts_and_returns_an_iso_date() -> None:
    executor = ToolExecutor(create_builtin_registry(), timeout_seconds=1, max_output_chars=5000)
    context = ToolContext(user_id="u", session_id="s")

    dated = await executor.execute(
        "get_weather", {"location": "Shanghai", "date": "2026-07-23"}, context
    )
    invalid = await executor.execute(
        "get_weather", {"location": "Shanghai", "date": "tomorrow"}, context
    )

    result = json.loads(dated.content)["result"]
    assert result["date"] == "2026-07-23"
    assert result["source_time"].startswith("2026-07-23T")
    assert result["location"] == "Shanghai"
    assert invalid.error_code == "invalid_arguments"


async def test_todos_are_isolated_between_sessions() -> None:
    todo_store = InMemoryTodoStore()
    executor = ToolExecutor(
        create_builtin_registry(todo_store=todo_store), timeout_seconds=1, max_output_chars=5000
    )
    user_a_window_1 = ToolContext(user_id="a", session_id="one")
    same_session = ToolContext(user_id="a", session_id="one")
    other_window = ToolContext(user_id="a", session_id="two")

    await executor.execute(
        "todo", {"action": "add", "title": "check weather"}, user_a_window_1
    )
    current = await executor.execute("todo", {"action": "list"}, same_session)
    other = await executor.execute("todo", {"action": "list"}, other_window)

    assert json.loads(current.content)["result"][0] == {
        "id": json.loads(current.content)["result"][0]["id"],
        "title": "check weather",
        "status": "pending",
    }
    assert json.loads(other.content)["result"] == []


async def test_todo_validates_actions_completes_items_and_reports_missing_records() -> None:
    executor = ToolExecutor(create_builtin_registry(), timeout_seconds=1, max_output_chars=5000)
    context = ToolContext(user_id="u", session_id="s")

    missing_title = await executor.execute("todo", {"action": "add"}, context)
    missing_id = await executor.execute("todo", {"action": "complete"}, context)
    added = await executor.execute(
        "todo", {"action": "add", "title": "ship release"}, context
    )
    todo_id = json.loads(added.content)["result"]["id"]
    completed = await executor.execute(
        "todo", {"action": "complete", "todo_id": todo_id}, context
    )
    absent = await executor.execute(
        "todo", {"action": "complete", "todo_id": "missing"}, context
    )

    assert missing_title.error_code == "invalid_arguments"
    assert missing_id.error_code == "invalid_arguments"
    assert json.loads(completed.content)["result"]["status"] == "completed"
    assert absent.error_code == "record_missing"


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlowHandler:
    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        del context, arguments
        await asyncio.sleep(0.05)
        return "late"


class LargeHandler:
    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        del context, arguments
        return "x" * 100


@pytest.mark.parametrize(
    ("handler", "max_output", "expected_code"),
    [(SlowHandler(), 1000, "tool_timeout"), (LargeHandler(), 20, "tool_output_too_large")],
)
async def test_tool_execution_limits(
    handler: SlowHandler | LargeHandler, max_output: int, expected_code: str
) -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="limited",
            description="limited",
            arguments_model=EmptyArguments,
            handler=handler,
            timeout_seconds=0.005 if isinstance(handler, SlowHandler) else 1,
        )
    )
    result = await ToolExecutor(
        registry, timeout_seconds=1, max_output_chars=max_output
    ).execute("limited", {}, ToolContext("u", "s"))

    assert not result.ok
    assert result.error_code == expected_code
