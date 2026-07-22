"""Integration tests for SQLite durability and isolation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from agent.runner import AgentRunner
from memory.sqlite import SQLiteSessionStore, SQLiteTodoStore
from runtime.builtin_tools import create_builtin_registry
from runtime.contracts import JSONObject, ModelResponse, ToolCall, Usage
from runtime.errors import SessionExistsError, SessionOwnershipError
from runtime.events import ModelMessageEvent, ToolResultEvent, UserMessageEvent
from runtime.tools import ToolContext, ToolExecutor
from tests.helpers import ScriptedModel, SleepingModel

pytestmark = pytest.mark.integration


def _call(name: str, arguments: JSONObject, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        raw_arguments=json.dumps(arguments),
        arguments=arguments,
    )


async def test_sessions_and_complete_tool_exchanges_recover_after_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    sessions = SQLiteSessionStore(database_path)
    session_id = await sessions.create_session("user-a", "window-one")
    first_model = ScriptedModel(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    _call("todo", {"action": "add", "title": "bring an umbrella"}, "add-1"),
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            ModelResponse(content="Added it."),
        ]
    )
    first_runner = AgentRunner(
        model=first_model,
        tools=create_builtin_registry(todo_store=SQLiteTodoStore(database_path)),
        sessions=sessions,
    )

    first = await first_runner.run(
        user_id="user-a", session_id=session_id, user_input="Add an umbrella reminder"
    )
    assert first.status == "completed"

    restored_sessions = SQLiteSessionStore(database_path)
    restored_events = await restored_sessions.get_events("user-a", session_id)
    assert [event.sequence for event in restored_events] == list(
        range(1, len(restored_events) + 1)
    )
    assert any(isinstance(event, UserMessageEvent) for event in restored_events)
    assert any(isinstance(event, ToolResultEvent) for event in restored_events)
    stored_model = next(
        event
        for event in restored_events
        if isinstance(event, ModelMessageEvent) and event.response.tool_calls
    )
    assert stored_model.response.tool_calls[0].id == "add-1"
    assert stored_model.response.usage.total_tokens == 15
    assert stored_model.turn_id == first.turn_id
    assert stored_model.trace_id == first.trace_id
    assert all(event.event_id for event in restored_events)
    metadata = await restored_sessions.get_session("user-a", session_id)
    assert metadata.event_count == len(restored_events)

    second_model = ScriptedModel(
        [
            ModelResponse(
                content=None,
                tool_calls=(_call("todo", {"action": "list"}, "list-1"),),
            ),
            ModelResponse(content="Your reminder is still here."),
        ]
    )
    second_runner = AgentRunner(
        model=second_model,
        tools=create_builtin_registry(todo_store=SQLiteTodoStore(database_path)),
        sessions=restored_sessions,
    )

    second = await second_runner.run(
        user_id="user-a", session_id=session_id, user_input="What is on my list?"
    )

    assert second.status == "completed"
    restored_context = second_model.calls[0][0]
    assert any(message.content == "Added it." for message in restored_context)
    tool_result = next(
        message for message in second_model.calls[1][0] if message.tool_call_id == "list-1"
    )
    assert json.loads(tool_result.content or "")["result"][0]["title"] == "bring an umbrella"


async def test_sqlite_sessions_preserve_ownership_and_uniqueness_after_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    sessions = SQLiteSessionStore(database_path)
    await sessions.create_session("owner", "fixed")

    restored = SQLiteSessionStore(database_path)
    with pytest.raises(SessionExistsError):
        await restored.create_session("owner", "fixed")
    with pytest.raises(SessionOwnershipError):
        await restored.get_events("intruder", "fixed")


async def test_sqlite_todos_persist_and_cannot_cross_session_boundaries(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"
    sessions = SQLiteSessionStore(database_path)
    await sessions.create_session("user-a", "one")
    await sessions.create_session("user-a", "two")
    store = SQLiteTodoStore(database_path)

    added = await store.add("one", "check tomorrow's weather")
    restored = SQLiteTodoStore(database_path)

    assert [item.title for item in await restored.list("one")] == [
        "check tomorrow's weather"
    ]
    assert await restored.list("two") == []
    assert await restored.complete("two", added.id) is None
    completed = await restored.complete("one", added.id)
    assert completed is not None and completed.status == "completed"


async def test_sqlite_todo_tool_uses_session_scope(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"
    sessions = SQLiteSessionStore(database_path)
    await sessions.create_session("user-a", "one")
    await sessions.create_session("user-a", "two")
    executor = ToolExecutor(
        create_builtin_registry(todo_store=SQLiteTodoStore(database_path)),
        timeout_seconds=1,
        max_output_chars=5000,
    )

    await executor.execute(
        "todo",
        {"action": "add", "title": "only in one"},
        ToolContext(user_id="user-a", session_id="one"),
    )
    other = await executor.execute(
        "todo", {"action": "list"}, ToolContext(user_id="user-a", session_id="two")
    )

    assert json.loads(other.content)["result"] == []


async def test_sqlite_serializes_same_session_and_allows_cross_session_concurrency(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.sqlite3"
    sessions = SQLiteSessionStore(database_path)
    first = await sessions.create_session("user-a", "first")
    second = await sessions.create_session("user-a", "second")
    same_model = SleepingModel()
    same_runner = AgentRunner(
        model=same_model, tools=create_builtin_registry(), sessions=sessions
    )

    await asyncio.gather(
        same_runner.run(user_id="user-a", session_id=first, user_input="one"),
        same_runner.run(user_id="user-a", session_id=first, user_input="two"),
    )
    assert same_model.max_active == 1

    parallel_model = SleepingModel()
    parallel_runner = AgentRunner(
        model=parallel_model, tools=create_builtin_registry(), sessions=sessions
    )
    await asyncio.gather(
        parallel_runner.run(user_id="user-a", session_id=first, user_input="three"),
        parallel_runner.run(user_id="user-a", session_id=second, user_input="four"),
    )
    assert parallel_model.max_active == 2


async def test_sqlite_migrates_legacy_todos_without_data_loss(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    timestamp = "2026-07-22T00:00:00+00:00"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, summary TEXT,
                summary_through_seq INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE todos (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, text TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO sessions VALUES ('s', 'u', NULL, 0, ?, ?)",
            (timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO todos VALUES ('t', 's', 'legacy item', 1, ?, ?)",
            (timestamp, timestamp),
        )

    store = SQLiteTodoStore(database_path)
    items = await store.list("s")

    assert [(item.id, item.title, item.status) for item in items] == [
        ("t", "legacy item", "completed")
    ]


async def test_corrupt_sqlite_event_returns_a_safe_storage_failure(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"
    sessions = SQLiteSessionStore(database_path)
    session_id = await sessions.create_session("user-a", "broken")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO events(session_id, seq, type, payload_json, created_at)
            VALUES (?, 1, 'user_message', '{not-json', '2026-07-22T00:00:00+00:00')
            """,
            (session_id,),
        )
    runner = AgentRunner(
        model=ScriptedModel([]), tools=create_builtin_registry(), sessions=sessions
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Hello")

    assert result.status == "failed"
    assert result.error is not None and result.error.code == "storage_failure"


async def test_todo_storage_failure_ends_the_turn_safely(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"
    sessions = SQLiteSessionStore(database_path)
    session_id = await sessions.create_session("user-a", "broken-todos")
    todo_store = SQLiteTodoStore(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE todos")
    model = ScriptedModel(
        [
            ModelResponse(
                content=None,
                tool_calls=(_call("todo", {"action": "list"}, "list-broken"),),
            )
        ]
    )
    runner = AgentRunner(
        model=model,
        tools=create_builtin_registry(todo_store=todo_store),
        sessions=sessions,
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="List todos")
    events = await sessions.get_events("user-a", session_id)
    calls = [
        call.id
        for event in events
        if isinstance(event, ModelMessageEvent)
        for call in event.response.tool_calls
    ]
    results = [
        event.tool_call_id for event in events if isinstance(event, ToolResultEvent)
    ]

    assert result.status == "failed"
    assert result.error is not None and result.error.code == "storage_failure"
    assert result.trace[-1].data["status"] == "failed"
    assert calls == results == ["list-broken"]
