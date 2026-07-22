# Context Management

## Purpose

The durable event stream contains the complete session history. The model context is a bounded working
view assembled for one model request. Separating these concepts allows long conversations to remain
recoverable while controlling token cost, latency, and irrelevant history.

The context builder belongs to the memory layer. It reads provider-neutral events and structured state,
then returns provider-neutral messages. It has no knowledge of SQLite rows or LiteLLM response classes.

## Context contents

Each model request should receive these sections in order:

| Section | Reason |
|---|---|
| System policy | Defines the agent role, safety rules, stopping behavior, and enabled capabilities. |
| Tool Schemas | Allows the model to choose valid registered tools. |
| Session summary | Preserves older goals, constraints, facts, and unresolved work. |
| Structured state | Preserves stable values such as todo IDs and statuses. |
| Recent complete turns | Supports conversational language and short follow-ups. |
| Current user message | Defines the immediate task. |

User messages, final assistant answers, assistant tool calls, corresponding tool results, important
entities, stable IDs, dates, and tool side effects belong in context when relevant.

Trace timings, retry counters, internal stack traces, credentials, hidden reasoning, unrelated
sessions, and obsolete raw payloads stay outside the prompt. These values add cost or risk without
helping the model choose its next action.

## Follow-ups

Plain conversational follow-ups rely mainly on the summary and recent messages. For example, a request
to "make it shorter" needs the preceding draft and style constraints.

Tool-based follow-ups also need structured results. "What about tomorrow?" needs the previous weather
location and resolved date. "Complete the second item" needs the recently displayed list together with
stable todo IDs and statuses. Structured state is more reliable than reconstructing identifiers from
prose.

The context builder should preserve the semantic order of messages. An assistant tool call and every
corresponding tool result form one indivisible group. Keeping only a call may cause a repeated action;
keeping only a result removes its origin and can violate provider message ordering requirements.

## Budgeting

Before each model call, the builder should estimate input size and reserve output capacity. A simple
character-based estimate is adequate for the first version if it is conservative and covered by tests.
The reduction threshold should begin around 75 percent of the configured input budget.

The initial allocation can reserve space for system policy and Schemas, session summary, recent
messages and tool results, and the next model output. Exact percentages should remain configuration
values because tool Schema size and model limits vary.

## Basic compression

Compression runs in two stages. The first stage prunes old bulky tool results from the prompt view and
replaces them with bounded representations containing important facts, source references, side effects,
and stable IDs. The full results remain in the event store.

If the context still exceeds the threshold, the second stage rolls the oldest complete turns into the
session summary. The summary records the highest covered event sequence, and the context then combines
that summary with a recent uncompressed tail.

The summary should preserve user goals, explicit constraints, decisions, confirmed facts, tool side
effects, important IDs, unresolved errors, and the current next step. Greetings, repeated explanations,
superseded drafts, and verbose raw tool payloads have little future value and can be omitted.

The first compressor should be deterministic. Deterministic reduction keeps tests repeatable and avoids
an extra model request that can fail or omit identifiers. A future `Summarizer` protocol can add semantic
LLM summaries after the storage and pairing rules are stable.

## Compression events

Every persisted summary should record its covered sequence range and emit a `context_compressed` event.
This supports audit, prevents the same range from being summarized repeatedly, and allows context to be
rebuilt after restart.

After compression, the builder should estimate size again. If the prompt still exceeds the hard model
limit, it should return a typed context-overflow error. The loop may perform one compression-and-retry
cycle before ending the turn.

## Component boundary

Context assembly belongs in `memory/context.py`; size estimation and reduction belong in
`memory/compression.py`. The session store provides ordered events and summary metadata. The agent loop
asks for a context before each model request and remains unaware of storage details.

Session persistence is described in [session.md](session.md). Model steps and stopping behavior are
described in [loops.md](loops.md).
