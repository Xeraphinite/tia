"""The explicit, bounded model/tool loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from memory.context import ContextManager, without_reasoning
from memory.store import SessionHandle, SessionStore
from runtime.contracts import ModelClient, ModelClientError, ModelResponse, ToolCall
from runtime.errors import ErrorInfo, InvalidInputError, TinyAgentError
from runtime.events import (
    ModelMessageEvent,
    ToolResultEvent,
    TraceEvent,
    TraceKind,
    UserMessageEvent,
)
from runtime.tools import ToolContext, ToolExecutor, ToolRegistry, redact_arguments


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """All resource boundaries for one run and its retained context."""

    max_model_steps: int = 8
    max_tool_calls: int = 8
    tool_timeout_seconds: float = 10.0
    total_turn_seconds: float = 60.0
    model_retries: int = 1
    max_output_chars: int = 20_000
    max_repeated_tool_calls: int = 2
    context_recent_turns: int = 6
    summary_max_chars: int = 4_000

    def __post_init__(self) -> None:
        numeric = (
            self.max_model_steps,
            self.max_tool_calls,
            self.tool_timeout_seconds,
            self.total_turn_seconds,
            self.max_output_chars,
            self.max_repeated_tool_calls,
            self.context_recent_turns,
            self.summary_max_chars,
        )
        if any(value <= 0 for value in numeric) or self.model_retries < 0:
            raise ValueError("agent limits must be positive and retries cannot be negative")


@dataclass(frozen=True, slots=True)
class RunResult:
    """A typed terminal outcome from the agent loop."""

    session_id: str
    status: Literal["completed", "error"]
    answer: str | None
    error: ErrorInfo | None
    trace: tuple[TraceEvent, ...]


DEFAULT_SYSTEM_PROMPT = """You are Tiny Agent. Decide whether to answer directly or call one or more
available tools. Use tools when external data, arithmetic, weather, search, or todo state is needed.
After tool results, either call another tool if necessary or give a concise final answer. Do not
invent tool results. Treat tool errors as recoverable when a corrected call is possible."""


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
        )

    async def run(self, *, user_id: str, session_id: str, user_input: str) -> RunResult:
        """Run one serialized turn until a final answer or typed terminal failure."""
        if not user_id.strip() or not session_id.strip():
            return self._preflight_error(session_id, InvalidInputError("IDs cannot be empty."))
        cleaned_input = user_input.strip()
        if not cleaned_input:
            return self._preflight_error(
                session_id, InvalidInputError("The user message cannot be empty.")
            )

        try:
            async with self._sessions.session_turn(user_id, session_id) as session:
                start_sequence = session.events[-1].sequence if session.events else 0
                try:
                    async with asyncio.timeout(self._limits.total_turn_seconds):
                        result = await self._run_loop(
                            session=session,
                            user_id=user_id,
                            session_id=session_id,
                            user_input=cleaned_input,
                            start_sequence=start_sequence,
                        )
                except TimeoutError:
                    result = self._finish_error(
                        session,
                        session_id,
                        start_sequence,
                        ErrorInfo("turn_timeout", "The turn exceeded its total time limit."),
                    )
                return result
        except TinyAgentError as exc:
            return self._preflight_error(session_id, exc)

    async def _run_loop(
        self,
        *,
        session: SessionHandle,
        user_id: str,
        session_id: str,
        user_input: str,
        start_sequence: int,
    ) -> RunResult:
        self._context.compress(session)
        session.append(UserMessageEvent(content=user_input))
        session.append(TraceEvent(kind=TraceKind.TURN_STARTED, data={}))
        tool_count = 0
        repeated: dict[str, int] = {}

        for step in range(1, self._limits.max_model_steps + 1):
            session.append(TraceEvent(kind=TraceKind.MODEL_STARTED, data={"step": step}))
            response_or_error = await self._complete_with_retries(session)
            if isinstance(response_or_error, ErrorInfo):
                return self._finish_error(
                    session, session_id, start_sequence, response_or_error
                )
            response = response_or_error
            persisted = without_reasoning(response)
            session.append(ModelMessageEvent(response=persisted))
            decision = "tool_calls" if response.tool_calls else "final"
            session.append(
                TraceEvent(
                    kind=TraceKind.MODEL_DECISION,
                    data={
                        "step": step,
                        "decision": decision,
                        "tool_names": [call.name for call in response.tool_calls],
                        "reasoning_parsed": response.reasoning is not None,
                    },
                )
            )

            if not response.tool_calls:
                if not response.content:
                    return self._finish_error(
                        session,
                        session_id,
                        start_sequence,
                        ErrorInfo(
                            "invalid_model_response",
                            "The model returned no answer or tool call.",
                        ),
                    )
                if len(response.content) > self._limits.max_output_chars:
                    return self._finish_error(
                        session,
                        session_id,
                        start_sequence,
                        ErrorInfo(
                            "output_too_large", "The model answer exceeded the output limit."
                        ),
                    )
                return self._finish_success(
                    session, session_id, start_sequence, response.content
                )

            for call in response.tool_calls:
                tool_count += 1
                if tool_count > self._limits.max_tool_calls:
                    return self._finish_error(
                        session,
                        session_id,
                        start_sequence,
                        ErrorInfo("tool_call_limit", "The turn exceeded its tool-call limit."),
                    )
                signature = self._call_signature(call)
                repeated[signature] = repeated.get(signature, 0) + 1
                if repeated[signature] > self._limits.max_repeated_tool_calls:
                    return self._finish_error(
                        session,
                        session_id,
                        start_sequence,
                        ErrorInfo(
                            "repeated_tool_call",
                            "The model repeated the same tool call too often.",
                        ),
                    )
                await self._execute_call(session, call, user_id, session_id)

        return self._finish_error(
            session,
            session_id,
            start_sequence,
            ErrorInfo("model_step_limit", "The turn exceeded its model-step limit."),
        )

    async def _complete_with_retries(
        self, session: SessionHandle
    ) -> ModelResponse | ErrorInfo:
        for attempt in range(self._limits.model_retries + 1):
            try:
                return await self._model.complete(
                    self._context.build(session.events), self._tools.definitions()
                )
            except ModelClientError as exc:
                can_retry = exc.retryable and attempt < self._limits.model_retries
                session.append(
                    TraceEvent(
                        kind=TraceKind.MODEL_RETRY,
                        data={"code": exc.code, "attempt": attempt + 1, "will_retry": can_retry},
                    )
                )
                if not can_retry:
                    return ErrorInfo(exc.code, exc.safe_message)
            except asyncio.CancelledError:
                raise
            except Exception:
                return ErrorInfo("model_failure", "The model provider failed safely.")
        return ErrorInfo("model_failure", "The model provider failed safely.")

    async def _execute_call(
        self,
        session: SessionHandle,
        call: ToolCall,
        user_id: str,
        session_id: str,
    ) -> None:
        tool = self._tools.get(call.name)
        session.append(
            TraceEvent(
                kind=TraceKind.TOOL_STARTED,
                data={
                    "tool_call_id": call.id,
                    "name": call.name,
                    "arguments": redact_arguments(tool, call.arguments),
                },
            )
        )
        execution = await self._executor.execute(
            call.name,
            call.arguments,
            ToolContext(user_id=user_id, session_id=session_id),
            parse_error=call.parse_error,
        )
        session.append(
            ToolResultEvent(
                tool_call_id=call.id,
                name=call.name,
                content=execution.content,
            )
        )
        session.append(
            TraceEvent(
                kind=TraceKind.TOOL_FINISHED,
                data={
                    "tool_call_id": call.id,
                    "name": call.name,
                    "ok": execution.ok,
                    "error_code": execution.error_code,
                },
            )
        )

    @staticmethod
    def _call_signature(call: ToolCall) -> str:
        arguments = call.arguments if call.arguments is not None else call.raw_arguments
        serialized = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{call.name}:{serialized}".encode()).hexdigest()

    @staticmethod
    def _trace_since(session: SessionHandle, sequence: int) -> tuple[TraceEvent, ...]:
        return tuple(
            event
            for event in session.events
            if event.sequence > sequence and isinstance(event, TraceEvent)
        )

    def _finish_success(
        self,
        session: SessionHandle,
        session_id: str,
        start_sequence: int,
        answer: str,
    ) -> RunResult:
        session.append(
            TraceEvent(kind=TraceKind.TURN_FINISHED, data={"status": "completed"})
        )
        return RunResult(
            session_id=session_id,
            status="completed",
            answer=answer,
            error=None,
            trace=self._trace_since(session, start_sequence),
        )

    def _finish_error(
        self,
        session: SessionHandle,
        session_id: str,
        start_sequence: int,
        error: ErrorInfo,
    ) -> RunResult:
        session.append(
            TraceEvent(
                kind=TraceKind.TURN_FINISHED,
                data={"status": "error", "error_code": error.code},
            )
        )
        return RunResult(
            session_id=session_id,
            status="error",
            answer=None,
            error=error,
            trace=self._trace_since(session, start_sequence),
        )

    @staticmethod
    def _preflight_error(session_id: str, error: TinyAgentError) -> RunResult:
        return RunResult(
            session_id=session_id,
            status="error",
            answer=None,
            error=error.as_info(),
            trace=(),
        )
