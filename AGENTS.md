# Tiny Agent contributor guide

## Project intent

Tiny Agent (`tia`) is a small, provider-neutral agent framework and runtime. Its purpose is to make
the essential reasoning and tool-use loop easy to understand, test, and extend. Favor a compact
implementation with explicit control flow over feature breadth.

Keep the core runtime self-owned. External agent frameworks such as OpenHands, LangChain, LangGraph,
and AutoGen are outside the project scope. Avoid recreating their complexity through large dependency
sets or speculative abstractions.

## Python and dependencies

Python is pinned by `.python-version`, and `uv` exclusively manages dependencies and commands.

```text
uv sync --dev
uv run python main.py
uv add <runtime-package>
uv add --dev <development-package>
```

Commit `pyproject.toml` and `uv.lock` together after dependency changes. Use the standard library when
it meets the requirement. Add a package only when it removes meaningful complexity or provides a
required integration.

Write code for Python 3.12. Public APIs require complete type annotations and useful docstrings.
Mypy runs in strict mode, so keep `Any` at narrow third-party boundaries and convert external values
into internal types immediately. Prefer small protocols and frozen dataclasses for internal contracts.
Use Pydantic for validated configuration, HTTP contracts, and other untrusted public input. Simple
internal values rarely need Pydantic models.

Use descriptive domain names. The normalized model result is `ModelResponse`; provider-specific
response types stay inside their adapter. Keep functions focused, use early returns for failure paths,
and avoid inheritance hierarchies where composition or a protocol is sufficient.

## Architecture boundaries

The source tree follows these responsibilities:

| Area | Responsibility |
|---|---|
| `agent/` | Agent policy, model/tool decision flow, and the bounded loop |
| `runtime/` | Model and tool adapters, execution, events, limits, timeouts, and cancellation |
| `memory/` | Sessions, conversation events, context selection, summaries, and storage adapters |
| `api/` | Thin FastAPI request and response translation |
| `tests/` | Tests that mirror the production boundaries |

Dependencies point inward toward provider-neutral contracts. Agent and runtime modules cannot depend
on FastAPI request, response, or dependency-injection types. The agent loop cannot know whether
memory uses SQLite or an in-memory store. Tool handlers cannot receive implicit filesystem, shell, or
network authority.

Keep LiteLLM behind one narrow `ModelClient` adapter. Convert LiteLLM objects into internal messages,
`ModelResponse`, tool calls, usage, and safe errors at that boundary. Internal contracts cannot expose
LiteLLM response objects.

## Runtime design

All model calls, tool execution, persistence, and runner APIs are async-first. Library code cannot
create or hide an event loop. Callers own loop lifecycle and cancellation.

Every run has explicit limits for model steps, tool calls, individual tool timeouts, total turn time,
retries, output size, and repeated calls. Terminal outcomes are typed and observable. Treat model
output and tool arguments as untrusted input, validate them before execution, and return structured
errors that the agent can act on when recovery is safe.

Tools expose a stable name, precise description, JSON-compatible input Schema, and one execution
function. Registration, validation, and execution policy remain separate. Side-effecting tools require
ordered execution and conservative retries. Read-only tools may gain concurrency after ordering and
cancellation behavior have tests.

Preserve execution state through typed append-only events. Logs support diagnostics; events support
conversation recovery and auditing. Keep assistant tool calls paired with their tool results when
building or compressing model context.

Session IDs represent conversation windows, and user IDs represent ownership. Verify ownership on
every session operation. Serialize complete turns within one session and allow independent sessions to
run concurrently. Shared user data belongs in a separate domain store rather than shared conversation
history.

## Errors, security, and credentials

Classify expected failures with stable error codes. Convert recoverable model and tool failures into
typed results. Authorization failures, cancellation, storage failures, and broken runtime invariants
end the turn safely. Public responses cannot contain internal stack traces or provider request objects.

Never commit `.env`, API keys, tokens, or provider credentials. Keep placeholders in `.env.example`.
OpenRouter is the initial provider. Its key is stored as `OPENAI_API_KEY`; copy that value into the
process-local `OPENROUTER_API_KEY` environment variable before calling LiteLLM. Avoid passing secrets
as function arguments because LiteLLM exception details may contain request values.

Redact tool arguments and results where fields may contain sensitive data. Hidden model reasoning,
authorization headers, credentials, and oversized raw tool results stay outside logs and model
context.

## Tests and delivery

Tests run without network access or credentials by default. Use scripted fake model clients,
in-memory stores, deterministic tool backends, and controlled clocks to cover orchestration. New
behavior needs a success test and a relevant failure test.

`tests/env/` contains dependency checks. Default test output must remain credential-free and
deterministic.

Before handing off a change, run:

```text
uv run ruff check .
uv run mypy .
uv run pytest
```

Keep changes focused. Preserve unrelated work in a dirty tree. Avoid compatibility layers, provider
features, registries, plugins, and persistence abstractions until a concrete requirement needs them.

## Current dependency constraint

LiteLLM remains constrained to the compatible `1.75.x` series. Version `1.93.0` attempts to build its
Rust bridge from source in this environment and stalls during package metadata generation. Re-test a
new release in a focused dependency update before widening the constraint.
