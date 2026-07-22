# Session Management

## Identity and isolation

A user ID represents ownership, while a session UUID represents one conversation window. This
distinction is required because one user can hold several independent conversations at the same time.

```text
User A
  Session 1: weather, reminder, later weather follow-up
  Session 2: weekly report, reminder, later report follow-up
```

Using the user ID as the conversation key would merge these histories and make short follow-ups such as
"What about tomorrow?" ambiguous. Every request should carry or resolve both identities, and the
session service should verify ownership before reading history or running the agent. A missing session
and a session owned by another user should produce the same public response to avoid revealing valid
identifiers.

Session IDs should be random UUIDs created when a chat window opens. Random identifiers work well in
URLs and prevent simple enumeration.

## Persistence model

SQLite is the initial durable store. The requirement to resume either window after time or process
restart requires persistence beyond process memory. SQLite supplies transactions and constraints
through the standard library and keeps local setup small. A `SessionStore` protocol allows tests to use
an in-memory implementation and leaves room for a production database adapter later.

The initial schema needs three records:

| Record | Key fields |
|---|---|
| `sessions` | `id`, `user_id`, `summary`, `summary_through_seq`, timestamps |
| `events` | `session_id`, `seq`, `type`, `payload_json`, timestamp |
| `todos` | `id`, `session_id`, `title`, `status`, timestamps |

`events` should enforce uniqueness for `(session_id, seq)`. Each append happens within a transaction
that allocates the next sequence number. The ordered stream becomes the source for conversation
recovery and trace inspection.

Session rows contain mutable metadata used for quick loading. Conversation content remains in
append-only events. Updating a summary changes the prompt view while leaving the covered raw events
available for audit and future reprocessing.

## Event history

The event stream should distinguish user messages, assistant messages, requested tool calls, tool
results, summaries, and terminal outcomes. This detail shows whether a side effect was requested,
started, or completed and gives the context builder enough structure to preserve valid message pairs.

A minimal stored event contains:

```text
event_id
session_id
sequence number
event type
JSON payload
creation time
turn_id
trace_id
```

JSON payloads should contain provider-neutral contracts. Provider response objects, request headers,
and credentials stay outside persistence.

## Concurrency

Each session needs its own asynchronous lock around a complete turn. The lock covers history loading,
user event append, all loop steps, final event append, and outcome creation.

Consider two messages arriving close together:

```text
Message A: Check the weather in Shanghai.
Message B: What about tomorrow?
```

Parallel execution could let message B build context before message A stores the location and weather
result. Serial execution guarantees that B observes the completed preceding turn. Different sessions
use different locks and can run concurrently.

An in-process lock matches the first deployment model: one application process using SQLite. Multiple
workers or hosts would require database coordination or a distributed lock. That extension belongs to
a later deployment milestone.

## Todo scope

The first release stores todos by session. This makes the assignment's isolation guarantee observable:
weather-related reminders in window 1 do not appear in the weekly-report window.

A personal assistant product may later choose user-wide todos. That change should introduce a
user-owned todo repository and explicit authorization rules. Conversation context should continue to
use session scope, because shared domain data and shared conversation history have different meanings.

## API boundary

The thin FastAPI layer should expose operations equivalent to creating a session, sending a message,
reading session metadata, and reading execution events for diagnostics.

```text
POST /sessions
POST /sessions/{session_id}/messages
GET  /sessions/{session_id}
GET  /sessions/{session_id}/events
```

Request and response models belong at the HTTP boundary. The API resolves the authenticated user,
checks ownership through the session service, invokes `AgentRunner`, and translates `RunOutcome` into
an HTTP response. Agent and runtime modules should remain independent of FastAPI types.

## Component boundary

Session contracts belong in `memory/store.py`, with implementations in `memory/in_memory.py` and
`memory/sqlite.py`. Context selection belongs in `memory/context.py`. Application composition creates
the session store, lock manager, tools, model client, and runner.

Loop behavior is described in [loops.md](loops.md), and history selection is described in
[context.md](context.md).
