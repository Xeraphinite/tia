"""End-to-end deterministic tests for the complete agent loop."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from agent.runner import AgentLimits, AgentRunner
from memory.store import InMemorySessionStore
from runtime.builtin_tools import create_builtin_registry
from runtime.contracts import (
    JSONObject,
    JSONValue,
    Message,
    ModelClientError,
    ModelResponse,
    ToolCall,
    Usage,
)
from runtime.events import ModelMessageEvent, SummaryEvent, ToolResultEvent, TraceEvent, TraceKind
from runtime.tools import Tool, ToolContext, ToolRegistry
from tests.helpers import ScriptedModel, SleepingModel


def call(name: str, arguments: dict[str, object], call_id: str = "call-1") -> ToolCall:
    raw = json.dumps(arguments)
    return ToolCall(
        id=call_id,
        name=name,
        raw_arguments=raw,
        arguments=arguments,  # type: ignore[arg-type]
    )


async def setup_runner(
    responses: Sequence[ModelResponse | ModelClientError],
    *,
    limits: AgentLimits | None = None,
) -> tuple[AgentRunner, ScriptedModel, InMemorySessionStore, str]:
    model = ScriptedModel(responses)
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    runner = AgentRunner(
        model=model,
        tools=create_builtin_registry(),
        sessions=sessions,
        limits=limits,
    )
    return runner, model, sessions, session_id


async def test_direct_answer_completes_one_step_with_trace() -> None:
    runner, model, _, session_id = await setup_runner([ModelResponse(content="Hello")])

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Hi")

    assert result.status == "completed"
    assert result.answer == "Hello"
    assert len(model.calls) == 1
    assert model.calls[0][0][-1] == Message(role="user", content="Hi")
    assert [event.kind for event in result.trace] == [
        TraceKind.TURN_STARTED,
        TraceKind.CONTEXT_BUILT,
        TraceKind.MODEL_STARTED,
        TraceKind.MODEL_COMPLETED,
        TraceKind.MODEL_DECISION,
        TraceKind.TURN_FINISHED,
    ]
    assert result.turn_id and result.trace_id
    assert all(event.turn_id == result.turn_id for event in result.trace)
    assert all(event.trace_id == result.trace_id for event in result.trace)


async def test_tool_result_is_paired_and_sent_back_before_final_answer() -> None:
    runner, model, sessions, session_id = await setup_runner(
        [
            ModelResponse(content=None, tool_calls=(call("calculator", {"expression": "6*7"}),)),
            ModelResponse(content="42"),
        ]
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Compute 6*7")

    assert result.answer == "42"
    second_context = model.calls[1][0]
    assistant = next(message for message in second_context if message.tool_calls)
    tool = next(message for message in second_context if message.role == "tool")
    assert assistant.tool_calls[0].id == tool.tool_call_id
    assert json.loads(tool.content or "")["result"]["value"] == 42
    events = await sessions.get_events("user-a", session_id)
    assert any(isinstance(event, ToolResultEvent) for event in events)


async def test_invalid_tool_arguments_are_recoverable_by_model() -> None:
    invalid = ToolCall(
        id="bad",
        name="get_weather",
        raw_arguments="not-json",
        arguments=None,
        parse_error="bad",
    )
    runner, model, _, session_id = await setup_runner(
        [ModelResponse(content=None, tool_calls=(invalid,)), ModelResponse(content="Please retry")]
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Weather")

    assert result.status == "completed"
    tool_message = next(message for message in model.calls[1][0] if message.role == "tool")
    assert json.loads(tool_message.content or "")["error"]["code"] == "invalid_arguments"


async def test_unknown_tool_is_returned_to_model_as_structured_failure() -> None:
    runner, model, _, session_id = await setup_runner(
        [
            ModelResponse(content=None, tool_calls=(call("unregistered", {}),)),
            ModelResponse(content="I cannot do that"),
        ]
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Do it")

    assert result.status == "completed"
    tool_message = next(message for message in model.calls[1][0] if message.role == "tool")
    assert json.loads(tool_message.content or "")["error"]["code"] == "unknown_tool"


async def test_search_and_weather_to_todo_chains_are_grounded_and_ordered() -> None:
    model = ScriptedModel(
        [
            ModelResponse(None, (call("search", {"query": "weather", "limit": 1}, "s"),)),
            ModelResponse(content="The guide confirms the result."),
            ModelResponse(
                None,
                (
                    call(
                        "get_weather",
                        {"location": "Shanghai", "date": "2026-07-23"},
                        "w",
                    ),
                ),
            ),
            ModelResponse(
                None,
                (
                    call(
                        "todo",
                        {"action": "add", "title": "bring an umbrella"},
                        "t",
                    ),
                ),
            ),
            ModelResponse(content="Weather checked and reminder added."),
        ]
    )
    sessions = InMemorySessionStore()
    search_session = await sessions.create_session("user-a", "search")
    chain_session = await sessions.create_session("user-a", "chain")
    runner = AgentRunner(model=model, tools=create_builtin_registry(), sessions=sessions)

    search = await runner.run(
        user_id="user-a", session_id=search_session, user_input="Find the weather guide"
    )
    chain = await runner.run(
        user_id="user-a", session_id=chain_session, user_input="Check weather and remind me"
    )

    search_tool = next(
        message for message in model.calls[1][0] if message.tool_call_id == "s"
    )
    assert "Weather operations guide" in (search_tool.content or "")
    assert search.status == chain.status == "completed"
    tool_order = [
        event.data["name"]
        for event in chain.trace
        if event.kind is TraceKind.TOOL_STARTED
    ]
    assert tool_order == ["get_weather", "todo"]


async def test_retryable_model_failure_retries_then_succeeds() -> None:
    runner, _, _, session_id = await setup_runner(
        [
            ModelClientError("temporary", "try again", retryable=True),
            ModelResponse(content="recovered"),
        ]
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Hi")

    assert result.answer == "recovered"
    retry = next(event for event in result.trace if event.kind is TraceKind.MODEL_RETRY)
    assert retry.data["will_retry"] is True
    assert retry.data["retry_after_ms"] == 100.0


async def test_non_retryable_model_failure_is_typed_terminal_outcome() -> None:
    runner, _, _, session_id = await setup_runner(
        [ModelClientError("bad_request", "request rejected", retryable=False)]
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Hi")

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "bad_request"
    assert result.error.message == "request rejected"


async def test_model_step_tool_call_and_repeat_limits_are_enforced() -> None:
    repeated_call = call("get_weather", {"location": "Paris"})
    repeated_runner, _, _, repeated_session = await setup_runner(
        [ModelResponse(None, (repeated_call,))] * 3,
        limits=AgentLimits(max_repeated_tool_calls=2),
    )
    repeated_result = await repeated_runner.run(
        user_id="user-a", session_id=repeated_session, user_input="Loop"
    )
    assert repeated_result.error is not None
    assert repeated_result.error.code == "repeated_tool_call"

    calls = tuple(
        call("get_weather", {"location": str(index)}, f"call-{index}")
        for index in range(3)
    )
    count_runner, _, _, count_session = await setup_runner(
        [ModelResponse(None, calls)], limits=AgentLimits(max_tool_calls=2)
    )
    count_result = await count_runner.run(
        user_id="user-a", session_id=count_session, user_input="Many"
    )
    assert count_result.error is not None
    assert count_result.error.code == "tool_call_limit"

    step_runner, step_model, _, step_session = await setup_runner(
        [
            ModelResponse(None, (call("get_weather", {"location": "A"}, "a"),)),
            ModelResponse(None, (call("get_weather", {"location": "B"}, "b"),)),
        ],
        limits=AgentLimits(max_model_steps=2),
    )
    step_result = await step_runner.run(
        user_id="user-a", session_id=step_session, user_input="Never final"
    )
    assert step_result.status == "step_limit"
    assert step_result.answer
    assert step_result.error is not None
    assert step_result.error.code == "model_step_limit"
    assert "final model step" in (step_model.calls[-1][0][-1].content or "")


async def test_tool_call_and_repetition_limits_leave_no_unpaired_calls() -> None:
    calls = tuple(
        call("get_weather", {"location": str(index)}, f"batch-{index}")
        for index in range(3)
    )
    count_runner, _, count_sessions, count_session = await setup_runner(
        [ModelResponse(None, calls)], limits=AgentLimits(max_tool_calls=2)
    )
    count_result = await count_runner.run(
        user_id="user-a", session_id=count_session, user_input="Many"
    )
    count_events = await count_sessions.get_events("user-a", count_session)

    repeated = call("get_weather", {"location": "Paris"}, "repeat")
    repeat_runner, _, repeat_sessions, repeat_session = await setup_runner(
        [ModelResponse(None, (repeated,)), ModelResponse(None, (repeated,))],
        limits=AgentLimits(max_repeated_tool_calls=2),
    )
    repeat_result = await repeat_runner.run(
        user_id="user-a", session_id=repeat_session, user_input="Repeat"
    )
    repeat_events = await repeat_sessions.get_events("user-a", repeat_session)

    assert count_result.status == repeat_result.status == "step_limit"
    assert _tool_call_and_result_ids(count_events) == (
        ["batch-0", "batch-1", "batch-2"],
        ["batch-0", "batch-1", "batch-2"],
    )
    assert _tool_call_and_result_ids(repeat_events) == (
        ["repeat", "repeat"],
        ["repeat", "repeat"],
    )


async def test_empty_and_oversized_model_answers_fail_safely() -> None:
    empty_runner, _, _, empty_session = await setup_runner(
        [ModelResponse(content=None), ModelResponse(content=None)]
    )
    empty = await empty_runner.run(user_id="user-a", session_id=empty_session, user_input="Hi")
    assert empty.status == "failed"
    assert empty.error is not None
    assert empty.error.code == "model_format_error"
    assert any(event.kind is TraceKind.FORMAT_REPAIR for event in empty.trace)

    large_runner, _, _, large_session = await setup_runner(
        [ModelResponse(content="12345")], limits=AgentLimits(max_output_chars=4)
    )
    large = await large_runner.run(user_id="user-a", session_id=large_session, user_input="Hi")
    assert large.error is not None
    assert large.error.code == "output_too_large"


async def test_empty_model_response_gets_one_format_repair_attempt() -> None:
    runner, model, _, session_id = await setup_runner(
        [ModelResponse(content=None), ModelResponse(content="Repaired answer")]
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Hi")

    assert result.status == "completed"
    assert result.answer == "Repaired answer"
    assert len(model.calls) == 2
    assert "previous response" in (model.calls[1][0][-1].content or "")
    assert sum(event.kind is TraceKind.FORMAT_REPAIR for event in result.trace) == 1


async def test_malformed_provider_response_and_final_step_empty_receive_repair() -> None:
    malformed_runner, malformed_model, _, malformed_session = await setup_runner(
        [
            ModelClientError(
                "invalid_model_response", "The model returned no choices.", retryable=False
            ),
            ModelResponse(content="Recovered format"),
        ]
    )
    malformed = await malformed_runner.run(
        user_id="user-a", session_id=malformed_session, user_input="Hi"
    )

    final_runner, final_model, _, final_session = await setup_runner(
        [ModelResponse(content=None), ModelResponse(content="Final repair")],
        limits=AgentLimits(max_model_steps=1),
    )
    final = await final_runner.run(
        user_id="user-a", session_id=final_session, user_input="Hi"
    )

    assert malformed.status == "completed" and len(malformed_model.calls) == 2
    assert final.status == "completed" and len(final_model.calls) == 2
    assert final.answer == "Final repair"
    assert any(event.kind is TraceKind.FORMAT_REPAIR for event in final.trace)


async def test_total_turn_timeout_is_terminal_and_observable() -> None:
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    runner = AgentRunner(
        model=SleepingModel(delay=0.05),
        tools=create_builtin_registry(),
        sessions=sessions,
        limits=AgentLimits(total_turn_seconds=0.005),
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Wait")

    assert result.status == "timeout"
    assert result.error is not None
    assert result.error.code == "turn_timeout"
    assert result.trace[-1].kind is TraceKind.TURN_FINISHED


async def test_caller_cancellation_is_typed_and_observable() -> None:
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    model = SleepingModel(delay=1)
    runner = AgentRunner(model=model, tools=create_builtin_registry(), sessions=sessions)
    task = asyncio.create_task(
        runner.run(user_id="user-a", session_id=session_id, user_input="Wait")
    )
    while model.active == 0:
        await asyncio.sleep(0)

    task.cancel()
    result = await task

    assert result.status == "cancelled"
    assert result.error is not None and result.error.code == "turn_cancelled"
    assert any(event.kind is TraceKind.TURN_CANCELLED for event in result.trace)
    assert result.trace[-1].data["status"] == "cancelled"


async def test_usage_timings_and_identifiers_are_returned() -> None:
    runner, _, sessions, session_id = await setup_runner(
        [
            ModelResponse(
                content="done",
                usage=Usage(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            )
        ]
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Hi")

    assert result.usage == Usage(prompt_tokens=11, completion_tokens=4, total_tokens=15)
    completed = next(event for event in result.trace if event.kind is TraceKind.MODEL_COMPLETED)
    assert isinstance(completed.data["duration_ms"], float)
    assert completed.data["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }
    events = await sessions.get_events("user-a", session_id)
    turn_events = [event for event in events if event.turn_id == result.turn_id]
    assert turn_events
    assert all(event.trace_id == result.trace_id for event in turn_events)


class BlockingHandler:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        del context, arguments
        self.started.set()
        await asyncio.Event().wait()
        return None


def _blocking_registry(handler: BlockingHandler) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="blocking",
            description="Wait until the turn is interrupted.",
            arguments_model=NoArguments,
            handler=handler,
            timeout_seconds=10,
        )
    )
    return registry


def _tool_call_and_result_ids(events: tuple[object, ...]) -> tuple[list[str], list[str]]:
    call_ids: list[str] = []
    result_ids: list[str] = []
    for event in events:
        if isinstance(event, ModelMessageEvent):
            call_ids.extend(call.id for call in event.response.tool_calls)
        elif isinstance(event, ToolResultEvent):
            result_ids.append(event.tool_call_id)
    return call_ids, result_ids


async def test_cancellation_and_timeout_preserve_complete_tool_result_pairs() -> None:
    async def run_interrupted(*, cancel: bool) -> tuple[str, list[str], list[str]]:
        sessions = InMemorySessionStore()
        session_id = await sessions.create_session("user-a")
        handler = BlockingHandler()
        model = ScriptedModel(
            [
                ModelResponse(
                    None,
                    (
                        call("blocking", {}, "slow-1"),
                        call("blocking", {}, "slow-2"),
                    ),
                )
            ]
        )
        runner = AgentRunner(
            model=model,
            tools=_blocking_registry(handler),
            sessions=sessions,
            limits=AgentLimits(total_turn_seconds=0.02 if not cancel else 5),
        )
        task = asyncio.create_task(
            runner.run(user_id="user-a", session_id=session_id, user_input="Wait")
        )
        await handler.started.wait()
        if cancel:
            task.cancel()
        result = await task
        events = await sessions.get_events("user-a", session_id)
        calls, results = _tool_call_and_result_ids(events)
        return result.status, calls, results

    cancelled = await run_interrupted(cancel=True)
    timed_out = await run_interrupted(cancel=False)

    assert cancelled[0] == "cancelled"
    assert timed_out[0] == "timeout"
    assert cancelled[1] == cancelled[2] == ["slow-1", "slow-2"]
    assert timed_out[1] == timed_out[2] == ["slow-1", "slow-2"]


async def test_session_ownership_and_missing_sessions_are_safe() -> None:
    runner, _, _, session_id = await setup_runner([ModelResponse(content="unused")])

    forbidden = await runner.run(user_id="user-b", session_id=session_id, user_input="Hi")
    missing = await runner.run(user_id="user-a", session_id="missing", user_input="Hi")

    assert forbidden.error is not None and forbidden.error.code == "session_forbidden"
    assert missing.error is not None and missing.error.code == "session_not_found"


async def test_two_windows_keep_conversation_context_isolated_and_support_followups() -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="window one answer"),
            ModelResponse(content="window two answer"),
            ModelResponse(content="window one followup"),
        ]
    )
    sessions = InMemorySessionStore()
    first = await sessions.create_session("user-a", "one")
    second = await sessions.create_session("user-a", "two")
    runner = AgentRunner(model=model, tools=create_builtin_registry(), sessions=sessions)

    await runner.run(user_id="user-a", session_id=first, user_input="Weather topic")
    await runner.run(user_id="user-a", session_id=second, user_input="Weekly report topic")
    await runner.run(user_id="user-a", session_id=first, user_input="What about tomorrow?")

    first_followup = model.calls[2][0]
    contents = [message.content for message in first_followup]
    assert "Weather topic" in contents
    assert "window one answer" in contents
    assert "What about tomorrow?" in contents
    assert "Weekly report topic" not in contents
    assert "window two answer" not in contents


async def test_tool_using_followup_retains_prior_complete_tool_exchange() -> None:
    model = ScriptedModel(
        [
            ModelResponse(None, (call("get_weather", {"location": "Shanghai"}),)),
            ModelResponse(content="It is clear."),
            ModelResponse(content="Same city, Celsius."),
        ]
    )
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    runner = AgentRunner(model=model, tools=create_builtin_registry(), sessions=sessions)

    await runner.run(user_id="user-a", session_id=session_id, user_input="Shanghai weather?")
    await runner.run(user_id="user-a", session_id=session_id, user_input="Which unit was that?")

    followup = model.calls[2][0]
    assert any(message.tool_calls for message in followup)
    assert any(
        message.role == "tool" and "Shanghai" in (message.content or "")
        for message in followup
    )


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LargeResultHandler:
    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        del context, arguments
        return {"stable_id": "item-123", "payload": "x" * 1400}


class SecretArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    secret: str
    public: str


class SecretHandler:
    async def __call__(self, context: ToolContext, arguments: JSONObject) -> JSONValue:
        del context
        return {
            "stable_id": "safe-123",
            "secret": arguments["secret"],
            "public": arguments["public"],
        }


async def test_sensitive_tool_fields_are_redacted_from_events_and_context() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="secret_tool",
            description="Exercise durable field redaction.",
            arguments_model=SecretArguments,
            handler=SecretHandler(),
            sensitive_fields=frozenset({"secret"}),
            sensitive_result_fields=frozenset({"secret"}),
        )
    )
    model = ScriptedModel(
        [
            ModelResponse(
                None,
                (
                    call(
                        "secret_tool",
                        {"secret": "top-secret", "public": "visible"},
                        "secret-1",
                    ),
                ),
            ),
            ModelResponse(content="done"),
        ]
    )
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    runner = AgentRunner(model=model, tools=registry, sessions=sessions)

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Use it")
    events = await sessions.get_events("user-a", session_id)
    stored_call = next(
        event.response.tool_calls[0]
        for event in events
        if isinstance(event, ModelMessageEvent) and event.response.tool_calls
    )
    stored_result = next(
        event.content for event in events if isinstance(event, ToolResultEvent)
    )
    second_context = model.calls[1][0]

    assert result.status == "completed"
    assert stored_call.arguments == {"secret": "[REDACTED]", "public": "visible"}
    assert "top-secret" not in stored_call.raw_arguments
    assert json.loads(stored_result)["result"]["secret"] == "[REDACTED]"
    assert all("top-secret" not in (message.content or "") for message in second_context)
    assert all(
        "top-secret" not in call.raw_arguments
        for message in second_context
        for call in message.tool_calls
    )


async def test_size_budget_prunes_tool_results_without_breaking_pairs() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="large_result",
            description="Return a deliberately large deterministic result.",
            arguments_model=NoArguments,
            handler=LargeResultHandler(),
        )
    )
    model = ScriptedModel(
        [
            ModelResponse(None, (call("large_result", {}, "large-1"),)),
            ModelResponse(content="Grounded answer"),
        ]
    )
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    runner = AgentRunner(
        model=model,
        tools=registry,
        sessions=sessions,
        limits=AgentLimits(
            max_context_chars=2500,
            context_output_reserve_chars=300,
            context_reduction_ratio=0.75,
            tool_result_context_chars=180,
        ),
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Use it")

    assert result.status == "completed"
    second_context = model.calls[1][0]
    assistant = next(message for message in second_context if message.tool_calls)
    tool_result = next(message for message in second_context if message.role == "tool")
    assert assistant.tool_calls[0].id == tool_result.tool_call_id == "large-1"
    assert tool_result.content is not None and len(tool_result.content) <= 180
    pruned = json.loads(tool_result.content)
    assert pruned["result"]["stable_id"] == "item-123"
    context_events = [
        event for event in result.trace if event.kind is TraceKind.CONTEXT_BUILT
    ]
    assert context_events[-1].data["pruned_tool_results"] == 1


async def test_irreducible_context_overflow_fails_before_model_call() -> None:
    runner, model, _, session_id = await setup_runner(
        [],
        limits=AgentLimits(max_context_chars=600, context_output_reserve_chars=100),
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Hello")

    assert result.status == "failed"
    assert result.error is not None and result.error.code == "context_overflow"
    assert model.calls == []


async def test_provider_context_overflow_gets_one_aggressive_retry() -> None:
    runner, model, _, session_id = await setup_runner(
        [
            ModelClientError(
                "context_overflow", "The provider context was too large.", retryable=False
            ),
            ModelResponse(content="Recovered after compression"),
        ]
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Hello")

    assert result.status == "completed"
    assert len(model.calls) == 2
    built = [event for event in result.trace if event.kind is TraceKind.CONTEXT_BUILT]
    assert [event.data["aggressive"] for event in built] == [False, True]
    overflow = next(event for event in result.trace if event.kind is TraceKind.MODEL_RETRY)
    assert overflow.data["code"] == "context_overflow"
    assert overflow.data["will_retry"] is True


async def test_context_compression_uses_summary_and_recent_complete_turn() -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="answer one"),
            ModelResponse(content="answer two"),
            ModelResponse(content="answer three"),
            ModelResponse(content="answer four"),
        ]
    )
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    runner = AgentRunner(
        model=model,
        tools=create_builtin_registry(),
        sessions=sessions,
        limits=AgentLimits(
            context_recent_turns=1,
            summary_max_chars=1000,
            max_context_chars=3500,
            context_output_reserve_chars=500,
            context_reduction_ratio=0.7,
        ),
    )

    for text in ("question one", "question two", "question three", "question four"):
        result = await runner.run(user_id="user-a", session_id=session_id, user_input=text)
        assert result.status == "completed"

    fourth_context = model.calls[3][0]
    summary = next(
        message
        for message in fourth_context
        if message.role == "system"
        and message.content
        and "Conversation summary" in message.content
    )
    assert "question one" in (summary.content or "")
    assert "question two" in (summary.content or "")
    assert "question three" in (summary.content or "")
    ordinary = [message.content for message in fourth_context if message.role != "system"]
    assert ordinary == ["question four"]
    events = await sessions.get_events("user-a", session_id)
    assert any(isinstance(event, SummaryEvent) for event in events)
    compressed = [
        event
        for event in events
        if isinstance(event, TraceEvent) and event.kind is TraceKind.CONTEXT_COMPRESSED
    ]
    assert compressed
    for event in compressed:
        through_sequence = event.data["through_sequence"]
        assert isinstance(through_sequence, int) and through_sequence > 0


async def test_compressed_summary_obeys_its_exact_size_limit() -> None:
    model = ScriptedModel(
        [ModelResponse(content="a" * 100), ModelResponse(content="b"), ModelResponse(content="c")]
    )
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    runner = AgentRunner(
        model=model,
        tools=create_builtin_registry(),
        sessions=sessions,
        limits=AgentLimits(
            context_recent_turns=1,
            summary_max_chars=30,
            max_context_chars=3500,
            context_output_reserve_chars=500,
            context_reduction_ratio=0.7,
        ),
    )

    for text in ("first long question", "second", "third"):
        await runner.run(user_id="user-a", session_id=session_id, user_input=text)

    events = await sessions.get_events("user-a", session_id)
    summaries = [event for event in events if isinstance(event, SummaryEvent)]
    assert summaries
    assert all(len(event.summary) <= 30 for event in summaries)


async def test_structured_ids_survive_summary_compression() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                None,
                (
                    call(
                        "todo",
                        {"action": "add", "title": "retain this item"},
                        "todo-add",
                    ),
                ),
            ),
            ModelResponse(content="added"),
            ModelResponse(content="filler answer"),
            ModelResponse(content="found it"),
        ]
    )
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    runner = AgentRunner(
        model=model,
        tools=create_builtin_registry(),
        sessions=sessions,
        limits=AgentLimits(
            max_context_chars=3500,
            context_output_reserve_chars=500,
            context_reduction_ratio=0.7,
            summary_max_chars=80,
        ),
    )

    await runner.run(user_id="user-a", session_id=session_id, user_input="Add it")
    await runner.run(user_id="user-a", session_id=session_id, user_input="Filler")
    await runner.run(user_id="user-a", session_id=session_id, user_input="Find my item")

    events = await sessions.get_events("user-a", session_id)
    todo_result = next(
        json.loads(event.content)["result"]
        for event in events
        if isinstance(event, ToolResultEvent) and event.name == "todo"
    )
    final_context = model.calls[3][0]
    state = next(
        message.content
        for message in final_context
        if message.role == "system"
        and message.content
        and message.content.startswith("Structured state:")
    )

    assert todo_result["id"] in state
    assert "retain this item" in state
    assert '"status":"pending"' in state


async def test_hidden_reasoning_is_parsed_but_never_persisted_or_reused() -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="answer", reasoning="secret chain"),
            ModelResponse(content="followup"),
        ]
    )
    sessions = InMemorySessionStore()
    session_id = await sessions.create_session("user-a")
    runner = AgentRunner(model=model, tools=create_builtin_registry(), sessions=sessions)

    first = await runner.run(user_id="user-a", session_id=session_id, user_input="First")
    await runner.run(user_id="user-a", session_id=session_id, user_input="Second")

    decision = next(event for event in first.trace if event.kind is TraceKind.MODEL_DECISION)
    assert decision.data["reasoning_parsed"] is True
    assert all("secret chain" not in (message.content or "") for message in model.calls[1][0])
    events = await sessions.get_events("user-a", session_id)
    stored = [event for event in events if isinstance(event, ModelMessageEvent)]
    assert all(event.response.reasoning is None for event in stored)


async def test_same_session_turns_serialize_while_different_sessions_can_run_concurrently() -> None:
    sessions = InMemorySessionStore()
    first = await sessions.create_session("user-a", "first")
    second = await sessions.create_session("user-a", "second")
    same_model = SleepingModel()
    runner = AgentRunner(model=same_model, tools=create_builtin_registry(), sessions=sessions)

    await asyncio.gather(
        runner.run(user_id="user-a", session_id=first, user_input="one"),
        runner.run(user_id="user-a", session_id=first, user_input="two"),
    )
    assert same_model.max_active == 1

    parallel_model = SleepingModel()
    parallel_runner = AgentRunner(
        model=parallel_model, tools=create_builtin_registry(), sessions=sessions
    )
    await asyncio.gather(
        parallel_runner.run(user_id="user-a", session_id=first, user_input="three"),
        parallel_runner.run(user_id="user-a", session_id=second, user_input="four"),
    )
    assert parallel_model.max_active == 2
