# Tiny Agent

Tiny Agent (`tia`) is a small, provider-neutral agent framework and runtime. It focuses on the
essential agent loop: model decisions, validated tool execution, isolated sessions, bounded context,
explicit limits, and observable events.

The project keeps orchestration visible and composable. LiteLLM sits behind a narrow model adapter,
FastAPI provides the HTTP boundary, and application state remains independent of both.

```text
user input -> context -> model -> final response
                            |
                            +-> tool calls -> validate -> execute -> model
```

## Design

The runtime is async-first and uses provider-neutral contracts. Tools expose a stable name,
description, JSON-compatible input Schema, and one execution function. Sessions isolate conversation
windows and preserve an append-only event history. Context selection combines a compact summary with
recent complete turns and structured tool state.

## Setup

Python is pinned by `.python-version`, and `uv` manages all dependencies.

```bash
uv sync --dev
uv run python main.py
```

## Model provider

OpenRouter is the initial provider. Store its key as `OPENAI_API_KEY` in a local `.env` file. Local
environment files and credentials are excluded from version control.
