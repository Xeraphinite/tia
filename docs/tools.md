# Tools

## Purpose

Tools let the model request controlled capabilities without receiving implicit filesystem, shell, or
network access. The model proposes an action; the runtime validates and executes it. This division lets
the model choose useful actions while keeping authority and safety in deterministic code.

Tiny Agent's first release should include four tools. Together they exercise local computation,
query-style results, contextual follow-ups, and stateful side effects.

| Tool | Capability exercised |
|---|---|
| `calculator` | Deterministic local execution and untrusted input handling |
| `search` | Query execution and bounded result content |
| `get_weather` | Follow-up context involving location and date |
| `todo` | Persistent state, stable identifiers, and side effects |

## Tool definition

Every registered tool has a stable name, a precise description, a JSON-compatible input Schema, one
asynchronous handler, and execution metadata.

```python
@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, JsonValue]
    handler: ToolHandler
    timeout_seconds: float = 10.0
    read_only: bool = True
    idempotent: bool = True
```

The name gives the model and runtime a stable reference. The description explains when the capability
is appropriate; a Schema describes structure but cannot communicate intent. The Schema constrains
generated arguments before they reach application code. Execution metadata controls timeout and retry
policy without coupling these concerns to each handler.

## Registry and executor

`ToolRegistry` owns registration, duplicate-name detection, lookup, and conversion to the
OpenAI-compatible function Schema passed to the model. Registration should reject invalid Schemas and
unknown JSON value types early, during application startup.

`ToolExecutor` owns runtime concerns. It resolves the requested name, decodes JSON arguments, validates
them, applies cross-field rules, emits execution events, enforces the timeout, and converts the result
or exception into a provider-neutral `ToolResult`.

Keeping registry and execution separate produces one validation and error policy for every tool. Tool
handlers can focus on their domain operation and return structured values.

```text
ToolResult
  call_id
  tool_name
  ok
  content for the next model call
  optional structured data
  optional error code
```

Tool results sent back to the model should be concise and JSON-compatible. Large raw payloads can be
stored with the execution record while the prompt receives a bounded representation.

## Calculator

The calculator accepts an arithmetic expression and returns a numeric result. Its input Schema should
require a non-empty expression and reject additional properties.

The implementation should parse the expression with `ast.parse(..., mode="eval")` and recursively
evaluate a small allowlist of numeric constants and arithmetic operators. Names, calls, attributes,
subscripts, comprehensions, and other Python syntax must fail validation. Length, nesting depth, and
result magnitude should have explicit bounds.

This design treats the expression as data. Direct Python evaluation would turn the calculator into a
general code-execution surface.

## Search

Search accepts a query and an optional result limit between one and ten. It returns structured entries
containing a title, short snippet, and stable mock source identifier.

The first backend should be deterministic and injectable. Live search introduces credentials,
network failures, cost, rate limits, and changing results, which would obscure tests of the agent
runtime. A small `SearchBackend` protocol allows a real provider adapter later while preserving the
registry, Schema, loop, and tests.

Search results require a prompt-size bound. The full structured result can stay in the tool execution
record, while the context builder includes a fixed number of entries and bounded snippets.

## Weather

Weather accepts a location and an optional date. It returns location, resolved date, condition,
temperature, and source time as structured fields.

A deterministic backend supports the required follow-up scenario:

```text
User: What is the weather in Shanghai today?
User: What about tomorrow?
```

The second request depends on the earlier location and date. Structured weather output gives the
context builder stable values to preserve. A backend protocol allows a live weather provider later.

## Todo

Todo supports `add`, `list`, and `complete` actions. Its Schema should allow `title` and `todo_id`,
then apply action-specific validation: `add` requires a title, and `complete` requires a todo ID.

Todo records need stable IDs because phrases such as "complete the second item" should resolve to a
specific stored record before execution. The tool returns the resulting item or list with IDs and
statuses so later turns can refer to them reliably.

The first implementation uses session-scoped todos. This gives the two-window acceptance test a clear
isolation rule. User-wide todos represent a different product scope and can later use a separate
user-owned domain store while session conversation history remains isolated.

Todo is a side-effecting tool, so automatic retries require care. A handler failure after the database
commit could otherwise create duplicates. The initial executor should avoid retrying side-effecting
calls automatically. A later idempotency key or unique tool-call record can support safe recovery.

## Tool errors

Recoverable failures become structured tool results so the model can correct its action. Relevant
codes include `unknown_tool`, `invalid_arguments`, `invalid_expression`, `tool_timeout`,
`record_missing`, and `tool_failed`. Internal stack traces remain in protected diagnostics.

The model receives a concise description of what it can change. For example, an `add` request without
a title should explain that `title` is required. Repeated equivalent failures are bounded by the loop
detector described in [loops.md](loops.md).

## Component boundary

Definitions, registry, executor, and built-in tools belong under `runtime/tools/`. The todo handler may
depend on a small repository protocol supplied during application composition. Tool code remains free
of FastAPI and LiteLLM types.
