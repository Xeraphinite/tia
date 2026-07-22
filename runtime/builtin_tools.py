"""The built-in calculator, search, todo, and weather tools."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
from datetime import date as Date
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from runtime.contracts import JSONObject, JSONValue
from runtime.tools import Tool, ToolContext, ToolDomainError, ToolRegistry


class CalculatorArguments(BaseModel):
    """Validated calculator input."""

    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=500)


class CalculatorHandler:
    """Evaluate a small arithmetic grammar without executing Python code."""

    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        del context
        expression = arguments["expression"]
        if not isinstance(expression, str):
            raise ValueError("expression must be a string")
        try:
            value = _safe_calculate(expression)
        except (ArithmeticError, SyntaxError, ValueError) as exc:
            raise ToolDomainError(
                "invalid_expression", "The calculator expression is invalid or unsupported."
            ) from exc
        return {"expression": expression, "value": value}


def _safe_calculate(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("expression is too complex")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent is too large")
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.FloorDiv):
                result = left // right
            elif isinstance(node.op, ast.Mod):
                result = left % right
            elif isinstance(node.op, ast.Pow):
                result = left**right
            else:
                raise ValueError("unsupported calculator operator")
            if isinstance(result, complex) or abs(result) > 1e100:
                raise ValueError("result is outside the supported range")
            return result
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
        raise ValueError("unsupported calculator expression")

    return evaluate(tree)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One provider-neutral search hit."""

    title: str
    snippet: str
    url: str


class SearchBackend(Protocol):
    """A replaceable search integration."""

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """Return deterministic, ranked search results."""
        ...


class InMemorySearchBackend:
    """A network-free search backend suitable for local use and tests."""

    def __init__(self, documents: list[SearchResult] | None = None) -> None:
        self._documents = documents or [
            SearchResult(
                title="Tiny Agent documentation",
                snippet="Tiny Agent supports tools, isolated sessions, and bounded context.",
                url="https://example.invalid/tiny-agent",
            ),
            SearchResult(
                title="Weather operations guide",
                snippet="Weather lookups accept a location, date, and temperature unit.",
                url="https://example.invalid/weather",
            ),
        ]

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        terms = {term.casefold() for term in query.split()}
        ranked = sorted(
            self._documents,
            key=lambda item: sum(
                term in f"{item.title} {item.snippet}".casefold() for term in terms
            ),
            reverse=True,
        )
        return ranked[:limit]


class SearchArguments(BaseModel):
    """Validated search input."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)


class SearchHandler:
    def __init__(self, backend: SearchBackend) -> None:
        self._backend = backend

    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        del context
        query = arguments["query"]
        limit = arguments["limit"]
        if not isinstance(query, str) or not isinstance(limit, int):
            raise ValueError("validated search arguments changed type")
        results = await self._backend.search(query, limit)
        return [
            {"title": item.title, "snippet": item.snippet, "url": item.url}
            for item in results
        ]


class TodoArguments(BaseModel):
    """Validated operations for one session's todo list."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["add", "list", "complete"]
    title: str | None = Field(default=None, min_length=1, max_length=500)
    todo_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def operation_has_required_fields(self) -> TodoArguments:
        if self.action == "add" and not self.title:
            raise ValueError("title is required when action is add")
        if self.action == "complete" and not self.todo_id:
            raise ValueError("todo_id is required when action is complete")
        return self


@dataclass(slots=True)
class TodoItem:
    id: str
    title: str
    status: Literal["pending", "completed"] = "pending"


class TodoStore(Protocol):
    """Persistence contract for session-scoped todo items."""

    async def add(self, session_id: str, title: str) -> TodoItem:
        """Add an item to one session."""
        ...

    async def list(self, session_id: str) -> list[TodoItem]:
        """List items belonging to one session."""
        ...

    async def complete(self, session_id: str, todo_id: str) -> TodoItem | None:
        """Complete an item only when it belongs to the requested session."""
        ...


class InMemoryTodoStore:
    """A session-scoped todo store suitable for deterministic tests."""

    def __init__(self) -> None:
        self._items: dict[str, list[TodoItem]] = {}
        self._lock = asyncio.Lock()

    async def add(self, session_id: str, title: str) -> TodoItem:
        async with self._lock:
            item = TodoItem(id=uuid4().hex[:12], title=title)
            self._items.setdefault(session_id, []).append(item)
            return item

    async def list(self, session_id: str) -> list[TodoItem]:
        async with self._lock:
            return [
                TodoItem(item.id, item.title, item.status)
                for item in self._items.get(session_id, [])
            ]

    async def complete(self, session_id: str, todo_id: str) -> TodoItem | None:
        async with self._lock:
            for item in self._items.get(session_id, []):
                if item.id == todo_id:
                    item.status = "completed"
                    return TodoItem(item.id, item.title, item.status)
        return None


class TodoHandler:
    def __init__(self, store: TodoStore) -> None:
        self._store = store

    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        action = arguments["action"]
        if action == "add":
            title = arguments["title"]
            if not isinstance(title, str):
                raise ValueError("validated todo title changed type")
            added = await self._store.add(context.session_id, title)
            return _todo_json(added)
        if action == "complete":
            todo_id = arguments["todo_id"]
            if not isinstance(todo_id, str):
                raise ValueError("validated todo id changed type")
            completed = await self._store.complete(context.session_id, todo_id)
            if completed is None:
                raise ToolDomainError(
                    "record_missing", "The requested todo item does not exist."
                )
            return _todo_json(completed)
        items = await self._store.list(context.session_id)
        return [_todo_json(item) for item in items]


def _todo_json(item: TodoItem) -> JSONObject:
    return {"id": item.id, "title": item.title, "status": item.status}


class WeatherArguments(BaseModel):
    """Validated mocked weather lookup."""

    model_config = ConfigDict(extra="forbid")
    location: str = Field(min_length=1, max_length=100)
    date: Date | None = None
    unit: Literal["celsius", "fahrenheit"] = "celsius"


class WeatherHandler:
    """Deterministic weather data; replace this handler for a real provider."""

    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        del context
        location = arguments["location"]
        requested_date = arguments["date"]
        unit = arguments["unit"]
        if (
            not isinstance(location, str)
            or (requested_date is not None and not isinstance(requested_date, str))
            or not isinstance(unit, str)
        ):
            raise ValueError("validated weather arguments changed type")
        resolved_date = requested_date or Date.today().isoformat()
        weather_key = f"{location.casefold()}:{resolved_date}"
        celsius = 12 + sum(weather_key.encode("utf-8")) % 19
        value = celsius if unit == "celsius" else round(celsius * 9 / 5 + 32, 1)
        return {
            "location": location,
            "date": resolved_date,
            "temperature": value,
            "unit": unit,
            "condition": "clear",
            "source_time": f"{resolved_date}T08:00:00+00:00",
        }


def create_builtin_registry(
    *,
    search_backend: SearchBackend | None = None,
    todo_store: TodoStore | None = None,
) -> ToolRegistry:
    """Create a registry containing all supported built-ins."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Safely evaluate an arithmetic expression.",
            arguments_model=CalculatorArguments,
            handler=CalculatorHandler(),
        )
    )
    registry.register(
        Tool(
            name="search",
            description="Search a deterministic local document index.",
            arguments_model=SearchArguments,
            handler=SearchHandler(search_backend or InMemorySearchBackend()),
        )
    )
    registry.register(
        Tool(
            name="todo",
            description="Add, list, or complete todo items in the current session.",
            arguments_model=TodoArguments,
            handler=TodoHandler(todo_store or InMemoryTodoStore()),
            read_only=False,
            idempotent=False,
        )
    )
    registry.register(
        Tool(
            name="get_weather",
            description="Look up deterministic weather for a location and optional ISO date.",
            arguments_model=WeatherArguments,
            handler=WeatherHandler(),
        )
    )
    return registry
