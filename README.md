# Tiny Agent

Tiny Agent (`tia`) is an experimental, minimal agent framework and runtime. It aims to provide the
small set of primitives needed for model calls, tool execution, agent loops, memory, limits, and
observable events without depending on another agent framework.

The runtime uses LiteLLM for provider-neutral model access and FastAPI with Uvicorn for its HTTP
interface.

## Setup

Install Python 3.12 and all locked runtime and development dependencies:

```bash
uv sync --dev
```

Useful checks:

```bash
uv run python main.py
uv run ruff check .
uv run mypy .
uv run pytest
```

Provider credentials should be supplied through environment variables or a local `.env` file.
Local environment files are ignored by Git.

Tiny Agent initially targets OpenRouter. Put the OpenRouter key in `OPENAI_API_KEY` in `.env`. A
paid live smoke test is available but deliberately excluded from normal test runs:

```bash
RUN_LIVE_LLM_TESTS=1 uv run pytest -m live
```

The smoke test calls `openai/gpt-5.4-nano` through LiteLLM's OpenRouter adapter.
