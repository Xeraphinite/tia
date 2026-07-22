"""Real-provider API composition used by the browser end-to-end test."""

from __future__ import annotations

import os
from pathlib import Path

from agent.runner import AgentLimits, AgentRunner
from api.app import create_app
from memory.sqlite import SQLiteSessionStore, SQLiteTodoStore
from runtime.builtin_tools import create_builtin_registry
from runtime.litellm_client import LiteLLMClient

database_path = Path(os.environ["TIA_E2E_DATABASE_PATH"])
sessions = SQLiteSessionStore(database_path)
tools = create_builtin_registry(todo_store=SQLiteTodoStore(database_path))
model = LiteLLMClient(
    os.getenv("TIA_E2E_MODEL", "openai/gpt-5.6-luna"),
    request_timeout_seconds=60,
)
runner = AgentRunner(
    model=model,
    tools=tools,
    sessions=sessions,
    limits=AgentLimits(total_turn_seconds=65, tool_timeout_seconds=10),
)
app = create_app(runner=runner, sessions=sessions)
