"""Tiny Agent local composition root."""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from agent.runner import AgentRunner
from api.app import create_app
from memory.sqlite import SQLiteSessionStore, SQLiteTodoStore
from runtime.builtin_tools import create_builtin_registry
from runtime.litellm_client import LiteLLMClient


def build_app() -> FastAPI:
    """Compose the default local runtime without hiding an event loop."""
    load_dotenv()
    database_path = os.getenv("TIA_DATABASE_PATH", "tia.sqlite3")
    sessions = SQLiteSessionStore(database_path)
    tools = create_builtin_registry(todo_store=SQLiteTodoStore(database_path))
    model = LiteLLMClient(os.getenv("TIA_MODEL", "openai/gpt-5.4-nano"))
    runner = AgentRunner(model=model, tools=tools, sessions=sessions)
    return create_app(runner=runner, sessions=sessions)


app = build_app()


def main() -> None:
    """Run the development API server."""
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
