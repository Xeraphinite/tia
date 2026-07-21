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

## Streamlit frontend

Tiny Agent includes a chat frontend with multiple conversation sessions, prompt starters, safe error
states, and a collapsible trace for each run. Start the API in one terminal:

```bash
uv run python main.py
```

Then start Streamlit in another:

```bash
uv run streamlit run streamlit_app.py
```

Open `http://localhost:8501`. The sidebar lets you create, switch, rename, and remove conversations.
Each conversation keeps an isolated chat history and its own API session ID. These frontend records
live in the current browser tab; the API remains the owner of runtime conversation state.
`TIA_API_URL` and `TIA_USER_ID` provide optional defaults. Restarting the API invalidates its existing
runtime sessions, so create a new conversation after an API restart.

## Model provider

OpenRouter is the initial provider. Store its key as `OPENAI_API_KEY` in a local `.env` file. Local
environment files and credentials are excluded from version control. The adapter copies that value
into the process-local `OPENROUTER_API_KEY` expected by LiteLLM and never passes credentials through
agent or tool arguments.

`openai/gpt-5.4-nano` is the default local model. Override it without changing code:

```bash
TIA_MODEL=openai/gpt-5.6-luna uv run python main.py
```

## Agent loop

Each run follows one visible bounded loop:

1. Validate and append the user input to its owned session.
2. Build context from the system prompt, a compact summary, and recent complete turns.
3. Ask the model to answer or select tools from their names, descriptions, and JSON Schemas.
4. Parse final text, reasoning metadata, usage, and tool calls at the LiteLLM boundary.
5. Validate each tool call with Pydantic, execute it under a timeout, and pair its structured result
   with the original tool-call ID.
6. Repeat until the model returns a final answer or a typed limit/error ends the turn.

Raw hidden model reasoning is recognized but is neither persisted nor included in later context or
public traces. This preserves decision observability (`final` versus `tool_calls`) without retaining
private reasoning.

The default registry contains four tools:

| Tool | Behavior |
|---|---|
| `calculator` | Safely evaluates a limited arithmetic grammar; it cannot execute Python code. |
| `search` | Searches a deterministic mock document index through a replaceable backend protocol. |
| `todo` | Adds, lists, and completes user-owned items in a domain store shared across that user's windows. |
| `weather` | Returns deterministic mock weather through a replaceable handler. |

Registration, Schema validation, execution policy, and handlers are separate. Tool failures are
structured results the model can correct; cancellation still propagates to the caller.

## Sessions and context

A session ID identifies one conversation window and a user ID owns it. Ownership is checked on every
read and turn. Complete turns in the same session are serialized, while independent sessions can run
concurrently. Two windows for one user therefore retain separate conversation histories, while shared
user data such as Todo items lives in its own store rather than being copied into chat history.

Context includes user messages, assistant answers and tool calls, and paired tool results. It excludes
execution logs and hidden reasoning. Once the number of completed turns exceeds the configured recent
window, old complete turns are replaced in model context by a deterministic bounded summary. The full
append-only event history remains available to the session store for recovery and auditing.

`AgentLimits` explicitly bounds model steps, tool calls, each tool's time, total turn time, model
retries, output size, repeated identical calls, recent context turns, and summary size. Expected
provider, validation, tool, ownership, and limit failures use stable error codes. Sanitized trace events
record model decisions, retries, tool starts/results, and the terminal outcome.

## HTTP API

Start the local API with `uv run python main.py`, then create and continue a session:

```bash
curl -X POST http://127.0.0.1:8000/v1/sessions \
  -H 'content-type: application/json' \
  -d '{"user_id":"user-a","session_id":"window-1"}'

curl -X POST http://127.0.0.1:8000/v1/sessions/window-1/messages \
  -H 'content-type: application/json' \
  -d '{"user_id":"user-a","message":"Check Shanghai weather and add a todo"}'
```

The message response contains a typed terminal status, answer or safe error, and the sanitized trace
for that turn.

## Verification

The default suite is network-free and exercises successful and failing model/tool loops, follow-ups,
session isolation and concurrency, compression, limits, built-in tools, provider parsing, and the HTTP
boundary:

```bash
uv run ruff check .
uv run mypy .
uv run pytest
```

The paid real-provider scenario uses `openai/gpt-5.6-luna` to run weather and Todo tools, then verifies
a tool-using follow-up in the same session:

```bash
TIA_RUN_REAL_TESTS=1 uv run pytest tests/real/test_openrouter.py -vv
```
