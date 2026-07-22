# Testing and Error Handling

## Error model

Tiny Agent should classify failures at the boundary where they occur and expose stable error codes to
the loop. Routine failures should produce typed results. Unexpected internal failures should preserve
detailed diagnostics in protected logs while returning a safe public message.

| Error class | Expected handling |
|---|---|
| Invalid tool arguments | Return a structured tool result so the model can correct the call. |
| Unknown tool | Return `unknown_tool` and allow a corrected call within loop limits. |
| Tool timeout | Cancel the handler, store `tool_timeout`, and let the model explain or choose another action. |
| Tool dependency failure | Return a bounded failure result when recovery is possible. |
| Model transient failure | Retry up to two times with bounded backoff. |
| Empty or malformed model response | Perform one format-repair request, then end with `model_format_error`. |
| Context overflow | Compress and retry once, then end with `context_overflow`. |
| Repeated equivalent calls | End after the configured threshold with a partial outcome. |
| Turn deadline | Cancel outstanding work and return `timeout`. |
| Session ownership failure | Reject before the loop and use a response that does not reveal ownership. |
| Storage or invariant failure | End the turn and return a stable internal error code. |

Recoverable errors are sent to the model because a changed action may succeed. An invalid calculator
expression, missing todo title, missing record, or timed-out query can lead to a corrected call or a
useful explanation. Authorization, cancellation, storage corruption, and broken runtime invariants end
the turn because another model action cannot safely resolve them.

Side-effecting tools require conservative retry behavior. Automatic retries are appropriate only when
the operation is read-only or proven idempotent. Todo mutations should execute once in the first
release. A later execution record keyed by tool call ID can provide stronger replay protection.

## Safe diagnostics

Each user turn has a trace ID and turn ID. Events and structured logs may contain the session ID,
event sequence, model name, step, tool name, tool call ID, status, timestamps, latency, token counts,
retry count, and stable error code.

Tool arguments and results require field-level redaction and size bounds. Credentials, authorization
headers, provider request objects, hidden reasoning, and public stack traces are excluded. LiteLLM
exceptions may contain request details, so provider exceptions should be caught at the adapter boundary
and converted by exception type and safe error code.

## Test architecture

Default tests must run without network access or live credentials. A scripted fake model is the core
test double because it controls each model decision exactly. For example, it can return a calculator
call on the first invocation and a final answer on the second. Tests can then assert tool execution,
context updates, event order, and terminal outcome without variation from a live model.

An in-memory session store gives unit and loop tests fast deterministic state. SQLite integration tests
exercise transactions, sequence constraints, restart recovery, and session isolation. Mock search and
weather backends return stable fixtures.

Tests should mirror implementation boundaries:

```text
tests/unit/         isolated contracts, parsing, tools, and frontend state
tests/integration/  agent loop, API, SQLite, and Streamlit component integration
tests/environment/  installed dependency checks
tests/provider/     opt-in real-provider scenarios
tests/e2e/          opt-in browser-to-provider application flows
```

Pytest markers mirror these directories, and collection fails when a test has no primary category or
more than one. Shared deterministic doubles remain in `tests/helpers.py` rather than a test suite.

## Required test cases

The tool suite should verify successful registration, Schema export, duplicate rejection, unknown-tool
lookup, argument validation, and handler timeouts. Calculator tests should cover arithmetic plus unsafe
syntax, division by zero, excessive depth, and excessive result size. Todo tests should cover add, list,
complete, required cross-fields, missing IDs, and session scope.

The model adapter suite should cover a direct text answer, one tool call, multiple ordered tool calls,
text accompanying calls, invalid JSON arguments, empty output, usage normalization, and safe provider
exceptions. These tests should also verify that LiteLLM response objects do not cross the adapter
boundary.

The loop suite should cover direct responses, calculator followed by a final answer, search followed by
a grounded answer, weather followed by todo creation, a corrected invalid call, a tool timeout,
repeated calls, maximum steps, total turn timeout, cancellation, and a model-format failure. Every case
should assert both `RunOutcome` and the ordered event stream.

The session and context suite should cover two sessions owned by the same user, ownership rejection,
plain conversational follow-ups, tool-based follow-ups, SQLite reopen and recovery, same-session
serialization, cross-session concurrency, compression thresholds, summary coverage, structured IDs,
and intact tool-call/result groups.

The key end-to-end scenarios are:

```text
"What is 19 * 23?"
  model -> calculator -> model -> final answer

Window 1: "Check tomorrow's Shanghai weather and add a reminder."
  model -> weather -> model -> todo -> model -> final answer

Window 2: "Draft my weekly report and remind me to send it."
  independent history and independent session-scoped todo state

Resume Window 1 after reopening the SQLite store
  previous weather and todo context remains available
```

## Quality gate

Every implementation phase ends with:

```text
uv run ruff check .
uv run mypy .
uv run pytest
```

New behavior requires a success test and a relevant failure test. The full default suite must remain
deterministic, credential-free, and network-free.

Tool contracts are described in [tools.md](tools.md), loop limits in [loops.md](loops.md), session
recovery in [session.md](session.md), and context reduction in [context.md](context.md).
