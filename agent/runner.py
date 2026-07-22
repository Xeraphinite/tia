"""The explicit, bounded model/tool loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Literal
from uuid import uuid4

from memory.context import ContextManager, ContextWindow, without_reasoning
from memory.store import SessionHandle, SessionStore
from runtime.contracts import (
    JSONObject,
    Message,
    ModelClient,
    ModelClientError,
    ModelResponse,
    ToolCall,
    Usage,
)
from runtime.errors import (
    ContextOverflowError,
    ErrorInfo,
    InvalidInputError,
    TinyAgentError,
)
from runtime.events import (
    ModelMessageEvent,
    ToolResultEvent,
    TraceEvent,
    TraceKind,
    UserMessageEvent,
)
from runtime.tools import (
    ToolContext,
    ToolExecution,
    ToolExecutor,
    ToolRegistry,
    redact_arguments,
    redact_result_content,
)

type RunStatus = Literal["completed", "step_limit", "timeout", "cancelled", "failed"]


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """All resource boundaries for one run and its retained context."""

    max_model_steps: int = 8
    max_tool_calls: int = 12
    tool_timeout_seconds: float = 10.0
    total_turn_seconds: float = 60.0
    model_retries: int = 2
    model_retry_backoff_seconds: float = 0.1
    max_output_chars: int = 20_000
    max_repeated_tool_calls: int = 3
    context_recent_turns: int = 6
    summary_max_chars: int = 4_000
    max_context_chars: int = 100_000
    context_output_reserve_chars: int = 20_000
    context_reduction_ratio: float = 0.75
    tool_result_context_chars: int = 2_000

    def __post_init__(self) -> None:
        positive = (
            self.max_model_steps,
            self.max_tool_calls,
            self.tool_timeout_seconds,
            self.total_turn_seconds,
            self.max_output_chars,
            self.max_repeated_tool_calls,
            self.context_recent_turns,
            self.summary_max_chars,
            self.max_context_chars,
            self.context_output_reserve_chars,
            self.tool_result_context_chars,
        )
        if (
            any(value <= 0 for value in positive)
            or self.model_retries < 0
            or self.model_retry_backoff_seconds < 0
        ):
            raise ValueError("agent limits must be positive and retries cannot be negative")
        if self.context_output_reserve_chars >= self.max_context_chars:
            raise ValueError("context output reserve must be smaller than the context limit")
        if not 0 < self.context_reduction_ratio < 1:
            raise ValueError("context reduction ratio must be between zero and one")


@dataclass(frozen=True, slots=True)
class RunResult:
    """A typed terminal outcome from the agent loop."""

    session_id: str
    status: RunStatus
    answer: str | None
    error: ErrorInfo | None
    trace: tuple[TraceEvent, ...]
    turn_id: str
    trace_id: str
    usage: Usage = field(default_factory=Usage)


DEFAULT_SYSTEM_PROMPT = """You are Tiny Agent. Decide whether to answer directly or call one or more
available tools. Use tools when external data, arithmetic, weather, search, or todo state is needed.
After tool results, either call another tool if necessary or give a concise final answer. Do not
invent tool results. Treat tool errors as recoverable when a corrected call is possible."""

FORMAT_REPAIR_PROMPT = """Your previous response contained neither an answer nor a tool call. Reply
now with either non-empty final text or valid native tool calls."""

FINAL_STEP_PROMPT = """This is the final model step for this turn. Answer now from the information
already available. Do not request another tool unless it is essential; no later model step is
available to interpret another result."""

PARTIAL_LIMIT_ANSWER = "The run reached its configured limit before a final answer was produced."


class AgentRunner:
    """Coordinate model decisions, validated tools, memory, limits, and traces."""

    def __init__(
        self,
        *,
        model: ModelClient,
        tools: ToolRegistry,
        sessions: SessionStore,
        limits: AgentLimits | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._model = model
        self._tools = tools
        self._sessions = sessions
        self._limits = limits or AgentLimits()
        self._executor = ToolExecutor(
            tools,
            timeout_seconds=self._limits.tool_timeout_seconds,
            max_output_chars=self._limits.max_output_chars,
        )
        self._context = ContextManager(
            recent_turns=self._limits.context_recent_turns,
            summary_max_chars=self._limits.summary_max_chars,
            system_prompt=system_prompt,
            max_context_chars=self._limits.max_context_chars,
            output_reserve_chars=self._limits.context_output_reserve_chars,
            reduction_ratio=self._limits.context_reduction_ratio,
            tool_result_max_chars=self._limits.tool_result_context_chars,
        )

    async def run(self, *, user_id: str, session_id: str, user_input: str) -> RunResult:
        """Run one serialized turn until a final answer or typed terminal outcome."""
        turn_id = uuid4().hex
        trace_id = uuid4().hex
        if not user_id.strip() or not session_id.strip():
            return self._preflight_result(
                session_id,
                status="failed",
                error=InvalidInputError("IDs cannot be empty.").as_info(),
                turn_id=turn_id,
                trace_id=trace_id,
            )
        cleaned_input = user_input.strip()
        if not cleaned_input:
            return self._preflight_result(
                session_id,
                status="failed",
                error=InvalidInputError("The user message cannot be empty.").as_info(),
                turn_id=turn_id,
                trace_id=trace_id,
            )

        try:
            async with self._sessions.session_turn(user_id, session_id) as session:
                start_sequence = session.events[-1].sequence if session.events else 0
                started_at = perf_counter()
                try:
                    async with asyncio.timeout(self._limits.total_turn_seconds):
                        return await self._run_loop(
                            session=session,
                            user_id=user_id,
                            session_id=session_id,
                            user_input=cleaned_input,
                            start_sequence=start_sequence,
                            started_at=started_at,
                            turn_id=turn_id,
                            trace_id=trace_id,
                        )
                except TimeoutError:
                    return await self._finish(
                        session,
                        session_id,
                        start_sequence,
                        status="timeout",
                        answer=None,
                        error=ErrorInfo(
                            "turn_timeout", "The turn exceeded its total time limit."
                        ),
                        usage=self._usage_since(session, start_sequence),
                        started_at=started_at,
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                except asyncio.CancelledError:
                    await session.append(
                        self._trace_event(
                            TraceKind.TURN_CANCELLED,
                            {"reason": "caller_cancelled"},
                            turn_id=turn_id,
                            trace_id=trace_id,
                        )
                    )
                    return await self._finish(
                        session,
                        session_id,
                        start_sequence,
                        status="cancelled",
                        answer=None,
                        error=ErrorInfo("turn_cancelled", "The turn was cancelled."),
                        usage=self._usage_since(session, start_sequence),
                        started_at=started_at,
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                except TinyAgentError as exc:
                    return await self._finish(
                        session,
                        session_id,
                        start_sequence,
                        status="failed",
                        answer=None,
                        error=exc.as_info(),
                        usage=self._usage_since(session, start_sequence),
                        started_at=started_at,
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
        except asyncio.CancelledError:
            return self._preflight_result(
                session_id,
                status="cancelled",
                error=ErrorInfo("turn_cancelled", "The turn was cancelled."),
                turn_id=turn_id,
                trace_id=trace_id,
            )
        except TinyAgentError as exc:
            return self._preflight_result(
                session_id,
                status="failed",
                error=exc.as_info(),
                turn_id=turn_id,
                trace_id=trace_id,
            )
        except Exception:
            return self._preflight_result(
                session_id,
                status="failed",
                error=ErrorInfo("runtime_failure", "The agent runtime failed safely."),
                turn_id=turn_id,
                trace_id=trace_id,
            )

    async def _run_loop(
        self,
        *,
        session: SessionHandle,
        user_id: str,
        session_id: str,
        user_input: str,
        start_sequence: int,
        started_at: float,
        turn_id: str,
        trace_id: str,
    ) -> RunResult:
        await session.append(
            self._trace_event(
                TraceKind.TURN_STARTED,
                {},
                turn_id=turn_id,
                trace_id=trace_id,
            )
        )
        await session.append(
            UserMessageEvent(content=user_input, turn_id=turn_id, trace_id=trace_id)
        )
        tool_count = 0
        repeated: dict[str, int] = {}
        usage = Usage()
        format_repair_used = False
        pending_format_repair = False
        provider_overflow_retried = False
        progress_text: str | None = None

        step = 0
        while step < self._limits.max_model_steps or pending_format_repair:
            step += 1
            extra_messages: list[Message] = []
            if pending_format_repair:
                extra_messages.append(Message(role="system", content=FORMAT_REPAIR_PROMPT))
                pending_format_repair = False
            if step >= self._limits.max_model_steps:
                extra_messages.append(Message(role="system", content=FINAL_STEP_PROMPT))

            prepared = await self._prepare_context(
                session,
                step=step,
                extra_messages=tuple(extra_messages),
                turn_id=turn_id,
                trace_id=trace_id,
            )
            if isinstance(prepared, ErrorInfo):
                return await self._finish(
                    session,
                    session_id,
                    start_sequence,
                    status="failed",
                    answer=None,
                    error=prepared,
                    usage=usage,
                    started_at=started_at,
                    turn_id=turn_id,
                    trace_id=trace_id,
                )

            response_or_error = await self._complete_with_retries(
                session,
                prepared.messages,
                step=step,
                turn_id=turn_id,
                trace_id=trace_id,
                context_recovery_available=not provider_overflow_retried,
            )
            if (
                isinstance(response_or_error, ErrorInfo)
                and response_or_error.code == "context_overflow"
                and not provider_overflow_retried
            ):
                provider_overflow_retried = True
                prepared = await self._prepare_context(
                    session,
                    step=step,
                    extra_messages=tuple(extra_messages),
                    turn_id=turn_id,
                    trace_id=trace_id,
                    aggressive=True,
                )
                if not isinstance(prepared, ErrorInfo):
                    response_or_error = await self._complete_with_retries(
                        session,
                        prepared.messages,
                        step=step,
                        turn_id=turn_id,
                        trace_id=trace_id,
                        context_recovery_available=False,
                    )
            if isinstance(response_or_error, ErrorInfo):
                if (
                    response_or_error.code == "invalid_model_response"
                    and not format_repair_used
                ):
                    format_repair_used = True
                    pending_format_repair = True
                    await session.append(
                        self._trace_event(
                            TraceKind.FORMAT_REPAIR,
                            {"step": step, "attempt": 1, "reason": response_or_error.code},
                            turn_id=turn_id,
                            trace_id=trace_id,
                        )
                    )
                    continue
                return await self._finish(
                    session,
                    session_id,
                    start_sequence,
                    status="failed",
                    answer=None,
                    error=response_or_error,
                    usage=usage,
                    started_at=started_at,
                    turn_id=turn_id,
                    trace_id=trace_id,
                )

            response = response_or_error
            usage = self._add_usage(usage, response.usage)
            persisted = self._redacted_response(without_reasoning(response))
            await session.append(
                ModelMessageEvent(response=persisted, turn_id=turn_id, trace_id=trace_id)
            )
            decision = "tool_calls" if response.tool_calls else "final"
            await session.append(
                self._trace_event(
                    TraceKind.MODEL_DECISION,
                    {
                        "step": step,
                        "decision": decision,
                        "tool_names": [call.name for call in response.tool_calls],
                        "reasoning_parsed": response.reasoning is not None,
                    },
                    turn_id=turn_id,
                    trace_id=trace_id,
                )
            )

            if not response.tool_calls:
                if not response.content or not response.content.strip():
                    if not format_repair_used:
                        format_repair_used = True
                        pending_format_repair = True
                        await session.append(
                            self._trace_event(
                                TraceKind.FORMAT_REPAIR,
                                {"step": step, "attempt": 1, "reason": "empty_response"},
                                turn_id=turn_id,
                                trace_id=trace_id,
                            )
                        )
                        continue
                    return await self._finish(
                        session,
                        session_id,
                        start_sequence,
                        status="failed",
                        answer=None,
                        error=ErrorInfo(
                            "model_format_error",
                            "The model returned no answer or tool call after format repair.",
                        ),
                        usage=usage,
                        started_at=started_at,
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                if len(response.content) > self._limits.max_output_chars:
                    return await self._finish(
                        session,
                        session_id,
                        start_sequence,
                        status="failed",
                        answer=None,
                        error=ErrorInfo(
                            "output_too_large", "The model answer exceeded the output limit."
                        ),
                        usage=usage,
                        started_at=started_at,
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                return await self._finish(
                    session,
                    session_id,
                    start_sequence,
                    status="completed",
                    answer=response.content,
                    error=None,
                    usage=usage,
                    started_at=started_at,
                    turn_id=turn_id,
                    trace_id=trace_id,
                )

            if response.content and response.content.strip():
                progress_text = response.content
            if step > self._limits.max_model_steps:
                await self._append_unexecuted_results(
                    session,
                    response.tool_calls,
                    code="model_step_limit",
                    message="Tool execution was skipped because only format repair was allowed.",
                    turn_id=turn_id,
                    trace_id=trace_id,
                )
                return await self._limit_result(
                    session,
                    session_id,
                    start_sequence,
                    error=ErrorInfo(
                        "model_step_limit", "The turn exceeded its model-step limit."
                    ),
                    answer=progress_text,
                    usage=usage,
                    started_at=started_at,
                    turn_id=turn_id,
                    trace_id=trace_id,
                )
            for index, call in enumerate(response.tool_calls):
                tool_count += 1
                if tool_count > self._limits.max_tool_calls:
                    await self._append_unexecuted_results(
                        session,
                        response.tool_calls[index:],
                        code="tool_call_limit",
                        message="Tool execution was skipped because the turn limit was reached.",
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                    return await self._limit_result(
                        session,
                        session_id,
                        start_sequence,
                        error=ErrorInfo(
                            "tool_call_limit", "The turn exceeded its tool-call limit."
                        ),
                        answer=progress_text,
                        usage=usage,
                        started_at=started_at,
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                try:
                    execution = await self._execute_call(
                        session,
                        call,
                        user_id,
                        session_id,
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                except (asyncio.CancelledError, TinyAgentError):
                    await self._append_unexecuted_results(
                        session,
                        response.tool_calls[index + 1 :],
                        code="turn_interrupted",
                        message="Tool execution was skipped because the turn ended early.",
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                    raise
                signature = self._call_signature(call, execution.content)
                repeated[signature] = repeated.get(signature, 0) + 1
                if repeated[signature] >= self._limits.max_repeated_tool_calls:
                    await self._append_unexecuted_results(
                        session,
                        response.tool_calls[index + 1 :],
                        code="repeated_tool_call",
                        message="Tool execution was skipped after repeated equivalent results.",
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                    await session.append(
                        self._trace_event(
                            TraceKind.LOOP_DETECTED,
                            {
                                "step": step,
                                "tool_name": call.name,
                                "repetitions": repeated[signature],
                            },
                            turn_id=turn_id,
                            trace_id=trace_id,
                        )
                    )
                    return await self._limit_result(
                        session,
                        session_id,
                        start_sequence,
                        error=ErrorInfo(
                            "repeated_tool_call",
                            "The model repeated the same tool call too often.",
                        ),
                        answer=progress_text,
                        usage=usage,
                        started_at=started_at,
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )

            if step >= self._limits.max_model_steps:
                return await self._limit_result(
                    session,
                    session_id,
                    start_sequence,
                    error=ErrorInfo(
                        "model_step_limit", "The turn exceeded its model-step limit."
                    ),
                    answer=progress_text,
                    usage=usage,
                    started_at=started_at,
                    turn_id=turn_id,
                    trace_id=trace_id,
                )

        raise RuntimeError("model loop exited without a terminal outcome")

    async def _prepare_context(
        self,
        session: SessionHandle,
        *,
        step: int,
        extra_messages: tuple[Message, ...],
        turn_id: str,
        trace_id: str,
        aggressive: bool = False,
    ) -> ContextWindow | ErrorInfo:
        try:
            prepared = await self._context.prepare(
                session,
                self._tools.definitions(),
                extra_messages=extra_messages,
                turn_id=turn_id,
                trace_id=trace_id,
                aggressive=aggressive,
            )
        except ContextOverflowError as exc:
            return exc.as_info()
        await session.append(
            self._trace_event(
                TraceKind.CONTEXT_BUILT,
                {
                    "step": step,
                    "estimated_chars": prepared.estimated_chars,
                    "input_capacity_chars": prepared.input_capacity_chars,
                    "pruned_tool_results": prepared.pruned_tool_results,
                    "aggressive": aggressive,
                },
                turn_id=turn_id,
                trace_id=trace_id,
            )
        )
        return prepared

    async def _complete_with_retries(
        self,
        session: SessionHandle,
        messages: tuple[Message, ...],
        *,
        step: int,
        turn_id: str,
        trace_id: str,
        context_recovery_available: bool,
    ) -> ModelResponse | ErrorInfo:
        for attempt in range(self._limits.model_retries + 1):
            await session.append(
                self._trace_event(
                    TraceKind.MODEL_STARTED,
                    {"step": step, "attempt": attempt + 1},
                    turn_id=turn_id,
                    trace_id=trace_id,
                )
            )
            started = perf_counter()
            try:
                response = await self._model.complete(messages, self._tools.definitions())
                await session.append(
                    self._trace_event(
                        TraceKind.MODEL_COMPLETED,
                        {
                            "step": step,
                            "attempt": attempt + 1,
                            "duration_ms": self._duration_ms(started),
                            "usage": self._usage_json(response.usage),
                        },
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                )
                return response
            except ModelClientError as exc:
                can_retry_here = (
                    exc.retryable
                    and exc.code != "context_overflow"
                    and attempt < self._limits.model_retries
                )
                will_retry = can_retry_here or (
                    exc.code == "context_overflow" and context_recovery_available
                )
                retry_delay = (
                    self._limits.model_retry_backoff_seconds * (2**attempt)
                    if can_retry_here
                    else 0.0
                )
                await session.append(
                    self._trace_event(
                        TraceKind.MODEL_RETRY,
                        {
                            "code": exc.code,
                            "attempt": attempt + 1,
                            "will_retry": will_retry,
                            "retry_after_ms": round(retry_delay * 1000, 3),
                            "duration_ms": self._duration_ms(started),
                        },
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                )
                if not can_retry_here:
                    return ErrorInfo(exc.code, exc.safe_message)
                if retry_delay:
                    await asyncio.sleep(retry_delay)
            except asyncio.CancelledError:
                raise
            except Exception:
                await session.append(
                    self._trace_event(
                        TraceKind.MODEL_RETRY,
                        {
                            "code": "model_failure",
                            "attempt": attempt + 1,
                            "will_retry": False,
                            "duration_ms": self._duration_ms(started),
                        },
                        turn_id=turn_id,
                        trace_id=trace_id,
                    )
                )
                return ErrorInfo("model_failure", "The model provider failed safely.")
        return ErrorInfo("model_failure", "The model provider failed safely.")

    async def _execute_call(
        self,
        session: SessionHandle,
        call: ToolCall,
        user_id: str,
        session_id: str,
        *,
        turn_id: str,
        trace_id: str,
    ) -> ToolExecution:
        tool = self._tools.get(call.name)
        await session.append(
            self._trace_event(
                TraceKind.TOOL_STARTED,
                {
                    "tool_call_id": call.id,
                    "name": call.name,
                    "arguments": redact_arguments(tool, call.arguments),
                },
                turn_id=turn_id,
                trace_id=trace_id,
            )
        )
        started = perf_counter()
        try:
            execution = await self._executor.execute(
                call.name,
                call.arguments,
                ToolContext(user_id=user_id, session_id=session_id),
                parse_error=call.parse_error,
            )
        except asyncio.CancelledError:
            interrupted = self._tool_error(
                "turn_interrupted", "Tool execution was interrupted before completion."
            )
            await self._persist_tool_execution(
                session,
                call,
                interrupted,
                duration_ms=self._duration_ms(started),
                turn_id=turn_id,
                trace_id=trace_id,
            )
            raise
        except TinyAgentError as exc:
            failed = self._tool_error(exc.code, exc.safe_message)
            await self._persist_tool_execution(
                session,
                call,
                failed,
                duration_ms=self._duration_ms(started),
                turn_id=turn_id,
                trace_id=trace_id,
            )
            raise
        await self._persist_tool_execution(
            session,
            call,
            execution,
            duration_ms=self._duration_ms(started),
            turn_id=turn_id,
            trace_id=trace_id,
        )
        return execution

    async def _persist_tool_execution(
        self,
        session: SessionHandle,
        call: ToolCall,
        execution: ToolExecution,
        *,
        duration_ms: float,
        turn_id: str,
        trace_id: str,
    ) -> None:
        tool = self._tools.get(call.name)
        await session.append(
            ToolResultEvent(
                tool_call_id=call.id,
                name=call.name,
                content=redact_result_content(tool, execution.content),
                turn_id=turn_id,
                trace_id=trace_id,
            )
        )
        await session.append(
            self._trace_event(
                TraceKind.TOOL_FINISHED,
                {
                    "tool_call_id": call.id,
                    "name": call.name,
                    "ok": execution.ok,
                    "error_code": execution.error_code,
                    "duration_ms": duration_ms,
                },
                turn_id=turn_id,
                trace_id=trace_id,
            )
        )

    @staticmethod
    def _call_signature(call: ToolCall, result_content: str) -> str:
        arguments = call.arguments if call.arguments is not None else call.raw_arguments
        serialized = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        try:
            result = json.loads(result_content)
            normalized_result = json.dumps(result, sort_keys=True, ensure_ascii=False)
        except json.JSONDecodeError:
            normalized_result = result_content
        value = f"{call.name}:{serialized}:{normalized_result}"
        return hashlib.sha256(value.encode()).hexdigest()

    async def _append_unexecuted_results(
        self,
        session: SessionHandle,
        calls: tuple[ToolCall, ...],
        *,
        code: str,
        message: str,
        turn_id: str,
        trace_id: str,
    ) -> None:
        execution = self._tool_error(code, message)
        for call in calls:
            await self._persist_tool_execution(
                session,
                call,
                execution,
                duration_ms=0.0,
                turn_id=turn_id,
                trace_id=trace_id,
            )

    def _redacted_response(self, response: ModelResponse) -> ModelResponse:
        calls: list[ToolCall] = []
        for call in response.tool_calls:
            tool = self._tools.get(call.name)
            arguments = redact_arguments(tool, call.arguments)
            if call.arguments is None:
                raw_arguments = (
                    "[REDACTED]"
                    if tool is None or tool.sensitive_fields
                    else call.raw_arguments
                )
                persisted_arguments = None
            else:
                raw_arguments = json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                )
                persisted_arguments = arguments
            calls.append(
                replace(
                    call,
                    raw_arguments=raw_arguments,
                    arguments=persisted_arguments,
                )
            )
        return replace(response, tool_calls=tuple(calls))

    @staticmethod
    def _tool_error(code: str, message: str) -> ToolExecution:
        content = json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return ToolExecution(ok=False, content=content, error_code=code)

    @staticmethod
    def _trace_since(session: SessionHandle, sequence: int) -> tuple[TraceEvent, ...]:
        return tuple(
            event
            for event in session.events
            if event.sequence > sequence and isinstance(event, TraceEvent)
        )

    async def _limit_result(
        self,
        session: SessionHandle,
        session_id: str,
        start_sequence: int,
        *,
        error: ErrorInfo,
        answer: str | None,
        usage: Usage,
        started_at: float,
        turn_id: str,
        trace_id: str,
    ) -> RunResult:
        return await self._finish(
            session,
            session_id,
            start_sequence,
            status="step_limit",
            answer=answer or PARTIAL_LIMIT_ANSWER,
            error=error,
            usage=usage,
            started_at=started_at,
            turn_id=turn_id,
            trace_id=trace_id,
        )

    async def _finish(
        self,
        session: SessionHandle,
        session_id: str,
        start_sequence: int,
        *,
        status: RunStatus,
        answer: str | None,
        error: ErrorInfo | None,
        usage: Usage,
        started_at: float,
        turn_id: str,
        trace_id: str,
    ) -> RunResult:
        data: JSONObject = {
            "status": status,
            "duration_ms": self._duration_ms(started_at),
            "usage": self._usage_json(usage),
        }
        if error is not None:
            data["error_code"] = error.code
        await session.append(
            self._trace_event(
                TraceKind.TURN_FINISHED,
                data,
                turn_id=turn_id,
                trace_id=trace_id,
            )
        )
        return RunResult(
            session_id=session_id,
            status=status,
            answer=answer,
            error=error,
            trace=self._trace_since(session, start_sequence),
            turn_id=turn_id,
            trace_id=trace_id,
            usage=usage,
        )

    @staticmethod
    def _preflight_result(
        session_id: str,
        *,
        status: RunStatus,
        error: ErrorInfo,
        turn_id: str,
        trace_id: str,
    ) -> RunResult:
        return RunResult(
            session_id=session_id,
            status=status,
            answer=None,
            error=error,
            trace=(),
            turn_id=turn_id,
            trace_id=trace_id,
        )

    @staticmethod
    def _trace_event(
        kind: TraceKind,
        data: JSONObject,
        *,
        turn_id: str,
        trace_id: str,
    ) -> TraceEvent:
        return TraceEvent(
            kind=kind,
            data=data,
            turn_id=turn_id,
            trace_id=trace_id,
        )

    @staticmethod
    def _add_usage(total: Usage, addition: Usage) -> Usage:
        return Usage(
            prompt_tokens=total.prompt_tokens + addition.prompt_tokens,
            completion_tokens=total.completion_tokens + addition.completion_tokens,
            total_tokens=total.total_tokens + addition.total_tokens,
        )

    @staticmethod
    def _usage_since(session: SessionHandle, sequence: int) -> Usage:
        total = Usage()
        for event in session.events:
            if event.sequence > sequence and isinstance(event, ModelMessageEvent):
                total = AgentRunner._add_usage(total, event.response.usage)
        return total

    @staticmethod
    def _usage_json(usage: Usage) -> JSONObject:
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    @staticmethod
    def _duration_ms(started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000, 3)
