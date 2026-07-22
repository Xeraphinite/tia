"""Paid provider tests. Enable with TIA_RUN_REAL_TESTS=1."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.runner import AgentLimits, AgentRunner
from memory.sqlite import SQLiteSessionStore, SQLiteTodoStore
from runtime.builtin_tools import create_builtin_registry
from runtime.events import TraceKind
from runtime.litellm_client import LiteLLMClient

pytestmark = [
    pytest.mark.provider,
    pytest.mark.skipif(
        os.getenv("TIA_RUN_REAL_TESTS") != "1",
        reason="real OpenRouter tests are opt-in",
    ),
]


async def test_luna_runs_multi_tool_scenario_and_session_followup(tmp_path: Path) -> None:
    database_path = tmp_path / "real.sqlite3"
    sessions = SQLiteSessionStore(database_path)
    session_id = await sessions.create_session("real-user")
    runner = AgentRunner(
        model=LiteLLMClient("openai/gpt-5.6-luna", request_timeout_seconds=90),
        tools=create_builtin_registry(todo_store=SQLiteTodoStore(database_path)),
        sessions=sessions,
        limits=AgentLimits(total_turn_seconds=180, tool_timeout_seconds=10),
    )

    first = await runner.run(
        user_id="real-user",
        session_id=session_id,
        user_input=(
            "You must use the get_weather tool to check Shanghai, then use the todo tool to add "
            "the item 'bring an umbrella'. Report both tool results."
        ),
    )
    first_tools = {
        str(event.data["name"])
        for event in first.trace
        if event.kind is TraceKind.TOOL_STARTED
    }

    assert first.status == "completed", first.error
    assert {"get_weather", "todo"} <= first_tools

    restored_sessions = SQLiteSessionStore(database_path)
    restored_runner = AgentRunner(
        model=LiteLLMClient("openai/gpt-5.6-luna", request_timeout_seconds=90),
        tools=create_builtin_registry(todo_store=SQLiteTodoStore(database_path)),
        sessions=restored_sessions,
        limits=AgentLimits(total_turn_seconds=180, tool_timeout_seconds=10),
    )
    followup = await restored_runner.run(
        user_id="real-user",
        session_id=session_id,
        user_input="Use the todo tool to list my items and tell me what you added previously.",
    )
    followup_tools = [
        event.data["name"]
        for event in followup.trace
        if event.kind is TraceKind.TOOL_STARTED
    ]

    assert followup.status == "completed", followup.error
    assert "todo" in followup_tools
    assert followup.answer is not None
    assert "umbrella" in followup.answer.casefold()
