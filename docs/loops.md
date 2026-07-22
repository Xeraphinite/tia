# Agent Loop

## Purpose

The agent loop turns one user message into either a direct answer or a sequence of tool calls followed
by an answer. Tiny Agent owns this loop so its control flow, limits, events, and failure behavior remain
visible and testable. An external agent framework would hide much of the behavior that this project is
designed to provide.

The loop is asynchronous because model requests, tool execution, and persistence are I/O operations.
Async APIs also support cancellation and timeouts without creating an event loop inside library code.

## Runtime flow

One turn follows this sequence:

```text
receive user message
        |
        v
verify and lock session
        |
        v
append user event
        |
        v
build context -> call model -> parse response
                                |          |
                                |          +-> final answer -> persist -> return
                                |
                                +-> tool calls -> validate -> execute -> persist results
                                                             |
                                                             +-> rebuild context and continue
```

The public runner should expose an interface similar to:

```python
class AgentRunner:
    async def run_turn(
        self,
        user_id: str,
        session_id: str,
        text: str,
    ) -> RunOutcome: ...
```

`AgentRunner` coordinates protocols for the model client, tool executor, session store, context
builder, clock, and event sink. It does not depend on FastAPI, SQLite, or LiteLLM response classes.
This separation allows the complete loop to run in tests with scripted model responses and in-memory
storage.

## Model decisions

Tiny Agent should use the provider's native function-calling representation through LiteLLM. Native
tool calls separate assistant text, tool names, call IDs, and arguments. This is more reliable than
parsing JSON or XML embedded in ordinary assistant prose.

The LiteLLM adapter converts provider responses into a provider-neutral structure:

```text
ModelResponse
  text: optional string
  tool_calls: ordered ToolCall collection
  reasoning_summary: optional short string
  usage: normalized token usage
```

The parser applies deterministic rules. Valid tool calls cause tool execution and another loop step.
Non-empty text without tool calls is the final answer. Text accompanying tool calls is progress text;
the loop continues after the calls complete. A response with neither text nor tool calls receives one
format-repair attempt and then ends with a model-format error.

Full hidden chain-of-thought stays outside session storage, logs, and user responses. Providers expose
reasoning in different forms, and intermediate reasoning can contain sensitive or unreliable content.
An optional short `reasoning_summary`, such as "weather data is required before creating the
reminder," provides operational explainability. The durable trace records the selected action,
validation result, execution result, and timing.

## Steps and terminal outcomes

Each model request consumes one step. A tool batch belongs to the step that requested it. After the
batch finishes, the next model request consumes the next step. This definition makes the step limit
predictable and easy to test.

The runner returns a typed outcome rather than raising routine control-flow exceptions:

| Status | Meaning |
|---|---|
| `completed` | The model produced a final answer. |
| `step_limit` | The maximum number of model steps was reached. |
| `timeout` | The total turn deadline expired. |
| `cancelled` | The caller cancelled the turn. |
| `failed` | A terminal model, storage, or runtime error occurred. |

When one step remains, the system policy should instruct the model to answer from available results.
If the model requests another tool anyway, the runner returns a clear partial-result response with the
`step_limit` status. This gives the model a chance to conclude while preserving a hard upper bound.

## Initial limits

The first implementation should use eight model steps, twelve tool calls per turn, a ten-second tool
timeout, a sixty-second turn deadline, and a repeated-call threshold of three. These limits cover
different failure modes: loop length, excessive fan-out, blocked dependencies, total latency, and
stuck behavior.

A repeated call is identified by the normalized tool name and arguments, together with the normalized
result when available. Three equivalent repetitions indicate that another iteration is unlikely to
make progress. The runner then records a loop-detected event and returns a partial outcome.

Independent read-only calls may gain parallel execution later. The first implementation should execute
calls in model-provided order. Ordered execution gives predictable semantics for stateful tools such as
todo and keeps the initial runtime small.

## Events

The loop emits typed events for every observable state transition:

```text
turn_started
user_message
context_built
model_started
model_completed
assistant_tool_calls
tool_started
tool_completed or tool_failed
assistant_message
turn_completed or turn_failed
```

Events preserve what happened and in what order. Structured logs can project latency, retry counts,
token usage, and error codes from these events. Session recovery uses the durable event history rather
than log output.

## Component boundary

The proposed implementation belongs in `agent/loop.py`, with decision policy in
`agent/decisions.py` and stopping rules in `agent/policy.py`. Provider-neutral contracts belong in
`runtime/contracts.py`; the LiteLLM conversion layer belongs in `runtime/litellm_client.py`.

Tool behavior is described in [tools.md](tools.md). Session locking and persistence are described in
[session.md](session.md). Prompt construction for every step is described in
[context.md](context.md). Failure behavior and verification are described in
[testing-and-error-handling.md](testing-and-error-handling.md).
