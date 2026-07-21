"""Context selection that retains summaries and recent complete turns."""

from __future__ import annotations

from dataclasses import replace

from memory.store import SessionHandle
from runtime.contracts import Message, ModelResponse
from runtime.events import (
    ModelMessageEvent,
    SessionEvent,
    SummaryEvent,
    ToolResultEvent,
    TraceEvent,
    TraceKind,
    UserMessageEvent,
)


class ContextManager:
    """Compress old complete turns and construct model context."""

    def __init__(self, *, recent_turns: int, summary_max_chars: int, system_prompt: str) -> None:
        if recent_turns < 1:
            raise ValueError("recent_turns must be positive")
        self._recent_turns = recent_turns
        self._summary_max_chars = summary_max_chars
        self._system_prompt = system_prompt

    def compress(self, session: SessionHandle) -> None:
        """Append a summary event when more than the configured complete turns remain."""
        events = session.events
        current_summary = self._latest_summary(events)
        cutoff = current_summary.through_sequence if current_summary else 0
        completed = self._completed_turns(events, after_sequence=cutoff)
        if len(completed) <= self._recent_turns:
            return

        omitted = completed[: -self._recent_turns]
        additions = [self._summarize_turn(turn) for turn in omitted]
        prior = current_summary.summary if current_summary else ""
        combined = "\n".join(part for part in (prior, *additions) if part)
        if len(combined) > self._summary_max_chars:
            prefix = "[Earlier summary truncated]\n"
            tail_size = max(0, self._summary_max_chars - len(prefix))
            combined = (prefix + combined[-tail_size:])[: self._summary_max_chars]
        through = omitted[-1][-1].sequence
        session.append(SummaryEvent(summary=combined, through_sequence=through))

    def build(self, events: tuple[SessionEvent, ...]) -> tuple[Message, ...]:
        """Build provider-neutral context without hidden model reasoning or trace records."""
        summary = self._latest_summary(events)
        cutoff = summary.through_sequence if summary else 0
        messages: list[Message] = [Message(role="system", content=self._system_prompt)]
        if summary is not None:
            messages.append(
                Message(role="system", content=f"Conversation summary:\n{summary.summary}")
            )

        for event in events:
            if event.sequence <= cutoff:
                continue
            if isinstance(event, UserMessageEvent):
                messages.append(Message(role="user", content=event.content))
            elif isinstance(event, ModelMessageEvent):
                messages.append(
                    Message(
                        role="assistant",
                        content=event.response.content,
                        tool_calls=event.response.tool_calls,
                    )
                )
            elif isinstance(event, ToolResultEvent):
                messages.append(
                    Message(
                        role="tool",
                        content=event.content,
                        tool_call_id=event.tool_call_id,
                        name=event.name,
                    )
                )
        return tuple(messages)

    @staticmethod
    def _latest_summary(events: tuple[SessionEvent, ...]) -> SummaryEvent | None:
        return next(
            (event for event in reversed(events) if isinstance(event, SummaryEvent)),
            None,
        )

    @staticmethod
    def _completed_turns(
        events: tuple[SessionEvent, ...], *, after_sequence: int
    ) -> list[list[SessionEvent]]:
        turns: list[list[SessionEvent]] = []
        current: list[SessionEvent] | None = None
        for event in events:
            if event.sequence <= after_sequence:
                continue
            if isinstance(event, UserMessageEvent):
                if current:
                    current = None
                current = [event]
            elif current is not None:
                current.append(event)
                if isinstance(event, TraceEvent) and event.kind is TraceKind.TURN_FINISHED:
                    turns.append(current)
                    current = None
        return turns

    @staticmethod
    def _summarize_turn(events: list[SessionEvent]) -> str:
        parts: list[str] = []
        for event in events:
            if isinstance(event, UserMessageEvent):
                parts.append(f"User: {event.content}")
            elif isinstance(event, ModelMessageEvent) and event.response.content:
                parts.append(f"Assistant: {event.response.content}")
            elif isinstance(event, ToolResultEvent):
                parts.append(f"Tool {event.name}: {event.content}")
        return " | ".join(parts)


def without_reasoning(response: ModelResponse) -> ModelResponse:
    """Remove provider reasoning before persistence or future model context."""
    return replace(response, reasoning=None)
