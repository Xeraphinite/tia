"""End-to-end deterministic tests for the complete agent loop."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from agent.runner import AgentLimits, AgentRunner
from memory.store import InMemorySessionStore
from runtime.builtin_tools import create_builtin_registry
from runtime.contracts import Message, ModelClientError, ModelResponse, ToolCall
from runtime.events import ModelMessageEvent, SummaryEvent, ToolResultEvent, TraceKind
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
        TraceKind.MODEL_STARTED,
        TraceKind.MODEL_DECISION,
        TraceKind.TURN_FINISHED,
    ]


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
        name="weather",
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
    assert json.loads(tool_message.content or "")["error"]["code"] == "invalid_tool_arguments"


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


async def test_non_retryable_model_failure_is_typed_terminal_outcome() -> None:
    runner, _, _, session_id = await setup_runner(
        [ModelClientError("bad_request", "request rejected", retryable=False)]
    )

    result = await runner.run(user_id="user-a", session_id=session_id, user_input="Hi")

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "bad_request"
    assert result.error.message == "request rejected"


async def test_model_step_tool_call_and_repeat_limits_are_enforced() -> None:
    repeated_call = call("weather", {"city": "Paris"})
    repeated_runner, _, _, repeated_session = await setup_runner(
        [ModelResponse(None, (repeated_call,))] * 3,
        limits=AgentLimits(max_repeated_tool_calls=2),
    )
    repeated_result = await repeated_runner.run(
        user_id="user-a", session_id=repeated_session, user_input="Loop"
    )
    assert repeated_result.error is not None
    assert repeated_result.error.code == "repeated_tool_call"

    calls = tuple(call("weather", {"city": str(index)}, f"call-{index}") for index in range(3))
    count_runner, _, _, count_session = await setup_runner(
        [ModelResponse(None, calls)], limits=AgentLimits(max_tool_calls=2)
    )
    count_result = await count_runner.run(
        user_id="user-a", session_id=count_session, user_input="Many"
    )
    assert count_result.error is not None
    assert count_result.error.code == "tool_call_limit"

    step_runner, _, _, step_session = await setup_runner(
        [
            ModelResponse(None, (call("weather", {"city": "A"}, "a"),)),
            ModelResponse(None, (call("weather", {"city": "B"}, "b"),)),
        ],
        limits=AgentLimits(max_model_steps=2),
    )
    step_result = await step_runner.run(
        user_id="user-a", session_id=step_session, user_input="Never final"
    )
    assert step_result.error is not None
    assert step_result.error.code == "model_step_limit"


async def test_empty_and_oversized_model_answers_fail_safely() -> None:
    empty_runner, _, _, empty_session = await setup_runner([ModelResponse(content=None)])
    empty = await empty_runner.run(user_id="user-a", session_id=empty_session, user_input="Hi")
    assert empty.error is not None
    assert empty.error.code == "invalid_model_response"

    large_runner, _, _, large_session = await setup_runner(
        [ModelResponse(content="12345")], limits=AgentLimits(max_output_chars=4)
    )
    large = await large_runner.run(user_id="user-a", session_id=large_session, user_input="Hi")
    assert large.error is not None
    assert large.error.code == "output_too_large"


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

    assert result.error is not None
    assert result.error.code == "turn_timeout"
    assert result.trace[-1].kind is TraceKind.TURN_FINISHED


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
            ModelResponse(None, (call("weather", {"city": "Shanghai"}),)),
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
        limits=AgentLimits(context_recent_turns=1, summary_max_chars=1000),
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
    ordinary = [message.content for message in fourth_context if message.role != "system"]
    assert ordinary == ["question three", "answer three", "question four"]
    events = await sessions.get_events("user-a", session_id)
    assert any(isinstance(event, SummaryEvent) for event in events)


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
        limits=AgentLimits(context_recent_turns=1, summary_max_chars=30),
    )

    for text in ("first long question", "second", "third"):
        await runner.run(user_id="user-a", session_id=session_id, user_input=text)

    events = await sessions.get_events("user-a", session_id)
    summaries = [event for event in events if isinstance(event, SummaryEvent)]
    assert summaries
    assert all(len(event.summary) <= 30 for event in summaries)


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
