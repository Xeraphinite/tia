# Tiny Agent contributor guide

## Goal

Tiny Agent (`tia`) is a small agent framework and runtime. Prefer a compact, understandable
implementation of the essential agent loop over feature breadth. Do not add an external agent
framework (OpenHands, LangChain, AutoGen, and similar) or recreate one indirectly through a large
dependency.

## Environment

- Python is pinned by `.python-version` and dependencies are managed exclusively with `uv`.
- Bootstrap or refresh the environment with `uv sync --dev`.
- Run Python commands through `uv run`, for example `uv run python main.py`.
- Add runtime dependencies with `uv add <package>` and development-only tools with
  `uv add --dev <package>`. Commit both `pyproject.toml` and `uv.lock` after dependency changes.
- Never commit `.env`, API keys, tokens, or provider credentials. Keep shareable placeholders in
  `.env.example`.

## Architecture boundaries

- `agent/` owns agent-facing policy and the reasoning/tool-use loop.
- `runtime/` owns execution, model and tool adapters, events, limits, and cancellation.
- `memory/` owns conversation state and context selection; storage details must not leak into the
  agent loop.
- Keep FastAPI in a thin HTTP boundary that translates requests and runtime events. Agent and
  runtime modules must not depend on FastAPI request, response, or dependency-injection types.
- `tests/env/` contains dependency and external-service smoke checks. Other tests should mirror
  the `agent/`, `runtime/`, `memory/`, and API boundaries as those features are implemented.
- Keep LiteLLM behind a narrow model-client adapter. Internal message, tool, and result contracts
  must remain provider-neutral and should not expose LiteLLM response objects.
- OpenRouter is the initial model provider. Its key is stored as `OPENAI_API_KEY`; pass that value
  to LiteLLM through a process-local `OPENROUTER_API_KEY` environment variable. Do not pass secrets
  as function arguments in tests because LiteLLM exception tracebacks may include request values.
- Use the standard library when it is sufficient. Pydantic is appropriate for validated public
  contracts and configuration, but avoid turning simple internal values into models without need.

## Runtime design rules

- Design async-first APIs for model calls and tool execution. Do not hide event-loop creation in
  library code.
- Make every loop bound explicit: maximum steps, timeouts, cancellation, and terminal outcomes.
- Treat model output and tool arguments as untrusted input. Validate before execution and return
  structured errors that an agent can reason about.
- Keep tools explicit: stable name, description, JSON-compatible input schema, and one execution
  function. Do not grant filesystem, shell, or network access implicitly.
- Separate orchestration from side effects so the loop can be tested with fake model clients and
  tools without real API calls.
- Prefer small composable protocols over inheritance hierarchies, registries, or plugin machinery
  until a concrete use case requires them.
- Preserve observable execution state with typed events rather than relying only on logs or printed
  output.

## Quality bar

- Before handing off a change, run `uv run ruff check .`, `uv run mypy .`, and `uv run pytest`.
- New behavior needs tests, including the failure path. Tests must not require network access or
  live model credentials by default.
- Mark live provider checks with `@pytest.mark.live` and require `RUN_LIVE_LLM_TESTS=1`. The current
  smoke-test model is `openrouter/openai/gpt-5.4-nano`: the first prefix selects LiteLLM's adapter,
  while `openai/gpt-5.4-nano` is the OpenRouter model slug.
- Catch live-call exceptions at the test boundary and report only the exception type. Provider
  exceptions and request objects may contain authorization headers and must not reach test output.
- Keep public APIs typed and documented. Avoid `Any` at internal boundaries; isolate it when a
  third-party response forces its use.
- Keep commits focused. Do not add abstractions, dependencies, compatibility layers, or provider
  features speculatively.

## Current dependency note

LiteLLM is constrained to the compatible `1.75.x` series. In this environment, `1.93.0` attempts
to build its Rust bridge from source and stalls during package metadata generation. Re-test the
latest release in a separate dependency update before widening the constraint.
