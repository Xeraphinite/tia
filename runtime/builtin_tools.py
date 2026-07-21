"""The built-in calculator, search, todo, and weather tools."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from runtime.contracts import JSONObject, JSONValue
from runtime.tools import Tool, ToolContext, ToolRegistry


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
        return {"expression": expression, "value": _safe_calculate(expression)}


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
                snippet="Weather lookups accept a city and Celsius or Fahrenheit units.",
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
    """Validated operations for a user's shared todo domain."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["add", "list", "complete"]
    text: str | None = Field(default=None, min_length=1, max_length=500)
    todo_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def operation_has_required_fields(self) -> TodoArguments:
        if self.action == "add" and not self.text:
            raise ValueError("text is required when action is add")
        if self.action == "complete" and not self.todo_id:
            raise ValueError("todo_id is required when action is complete")
        return self


@dataclass(slots=True)
class TodoItem:
    id: str
    text: str
    completed: bool = False


class InMemoryTodoStore:
    """A user-owned domain store independent from conversation history."""

    def __init__(self) -> None:
        self._items: dict[str, list[TodoItem]] = {}
        self._lock = asyncio.Lock()

    async def add(self, user_id: str, text: str) -> TodoItem:
        async with self._lock:
            item = TodoItem(id=uuid4().hex[:12], text=text)
            self._items.setdefault(user_id, []).append(item)
            return item

    async def list(self, user_id: str) -> list[TodoItem]:
        async with self._lock:
            return [
                TodoItem(item.id, item.text, item.completed)
                for item in self._items.get(user_id, [])
            ]

    async def complete(self, user_id: str, todo_id: str) -> TodoItem | None:
        async with self._lock:
            for item in self._items.get(user_id, []):
                if item.id == todo_id:
                    item.completed = True
                    return TodoItem(item.id, item.text, item.completed)
        return None


class TodoHandler:
    def __init__(self, store: InMemoryTodoStore) -> None:
        self._store = store

    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        action = arguments["action"]
        if action == "add":
            text = arguments["text"]
            if not isinstance(text, str):
                raise ValueError("validated todo text changed type")
            added = await self._store.add(context.user_id, text)
            return _todo_json(added)
        if action == "complete":
            todo_id = arguments["todo_id"]
            if not isinstance(todo_id, str):
                raise ValueError("validated todo id changed type")
            completed = await self._store.complete(context.user_id, todo_id)
            return _todo_json(completed) if completed else {"found": False, "todo_id": todo_id}
        items = await self._store.list(context.user_id)
        return [_todo_json(item) for item in items]


def _todo_json(item: TodoItem) -> JSONObject:
    return {"id": item.id, "text": item.text, "completed": item.completed}


class WeatherArguments(BaseModel):
    """Validated mocked weather lookup."""

    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=1, max_length=100)
    unit: Literal["celsius", "fahrenheit"] = "celsius"


class WeatherHandler:
    """Deterministic weather data; replace this handler for a real provider."""

    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        del context
        city = arguments["city"]
        unit = arguments["unit"]
        if not isinstance(city, str) or not isinstance(unit, str):
            raise ValueError("validated weather arguments changed type")
        celsius = 12 + sum(city.casefold().encode("utf-8")) % 19
        value = celsius if unit == "celsius" else round(celsius * 9 / 5 + 32, 1)
        return {"city": city, "temperature": value, "unit": unit, "condition": "clear"}


def create_builtin_registry(
    *,
    search_backend: SearchBackend | None = None,
    todo_store: InMemoryTodoStore | None = None,
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
            description="Add, list, or complete todo items owned by the current user.",
            arguments_model=TodoArguments,
            handler=TodoHandler(todo_store or InMemoryTodoStore()),
            side_effecting=True,
        )
    )
    registry.register(
        Tool(
            name="weather",
            description="Look up deterministic weather for a city.",
            arguments_model=WeatherArguments,
            handler=WeatherHandler(),
        )
    )
    return registry
