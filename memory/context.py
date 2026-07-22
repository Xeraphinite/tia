"""Size-aware context selection over the durable session event stream."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from memory.store import SessionHandle
from runtime.contracts import Message, ModelResponse, ToolDefinition
from runtime.errors import ContextOverflowError
from runtime.events import (
    ModelMessageEvent,
    SessionEvent,
    SummaryEvent,
    ToolResultEvent,
    TraceEvent,
    TraceKind,
    UserMessageEvent,
)


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """One bounded model input and its observable size metadata."""

    messages: tuple[Message, ...]
    estimated_chars: int
    input_capacity_chars: int
    pruned_tool_results: int


class ContextManager:
    """Compress old turns, prune bulky results, and construct bounded model context."""

    def __init__(
        self,
        *,
        recent_turns: int,
        summary_max_chars: int,
        system_prompt: str,
        max_context_chars: int,
        output_reserve_chars: int,
        reduction_ratio: float,
        tool_result_max_chars: int,
    ) -> None:
        if recent_turns < 1:
            raise ValueError("recent_turns must be positive")
        if summary_max_chars < 1 or max_context_chars < 1 or tool_result_max_chars < 1:
            raise ValueError("context character limits must be positive")
        if output_reserve_chars < 1 or output_reserve_chars >= max_context_chars:
            raise ValueError("output reserve must be positive and smaller than context size")
        if not 0 < reduction_ratio < 1:
            raise ValueError("context reduction ratio must be between zero and one")
        self._recent_turns = recent_turns
        self._summary_max_chars = summary_max_chars
        self._system_prompt = system_prompt
        self._max_context_chars = max_context_chars
        self._output_reserve_chars = output_reserve_chars
        self._reduction_ratio = reduction_ratio
        self._tool_result_max_chars = tool_result_max_chars

    async def prepare(
        self,
        session: SessionHandle,
        tools: tuple[ToolDefinition, ...],
        *,
        extra_messages: tuple[Message, ...] = (),
        turn_id: str,
        trace_id: str,
        aggressive: bool = False,
    ) -> ContextWindow:
        """Return a bounded prompt, persisting deterministic summaries when necessary."""
        input_capacity = self._max_context_chars - self._output_reserve_chars
        reduction_threshold = max(1, int(input_capacity * self._reduction_ratio))
        tool_limit = self._tool_result_max_chars // 2 if aggressive else None

        if aggressive:
            while await self._compress_oldest_turn(
                session, turn_id=turn_id, trace_id=trace_id
            ):
                pass

        messages, pruned = self._messages(
            session.events,
            tool_result_char_limit=tool_limit,
        )
        messages += extra_messages
        estimated = self._estimate(messages, tools)

        if estimated > reduction_threshold and tool_limit is None:
            tool_limit = self._tool_result_max_chars
            messages, pruned = self._messages(
                session.events,
                tool_result_char_limit=tool_limit,
            )
            messages += extra_messages
            estimated = self._estimate(messages, tools)

        while estimated > reduction_threshold:
            compressed = await self._compress_oldest_turn(
                session, turn_id=turn_id, trace_id=trace_id
            )
            if not compressed:
                break
            messages, pruned = self._messages(
                session.events,
                tool_result_char_limit=tool_limit,
            )
            messages += extra_messages
            estimated = self._estimate(messages, tools)

        if estimated > input_capacity:
            raise ContextOverflowError(
                "The required conversation context exceeds the configured model limit."
            )
        return ContextWindow(
            messages=messages,
            estimated_chars=estimated,
            input_capacity_chars=input_capacity,
            pruned_tool_results=pruned,
        )

    def build(
        self,
        events: tuple[SessionEvent, ...],
        *,
        tool_result_char_limit: int | None = None,
    ) -> tuple[Message, ...]:
        """Build provider-neutral context without trace records or hidden reasoning."""
        messages, _ = self._messages(
            events, tool_result_char_limit=tool_result_char_limit
        )
        return messages

    def _messages(
        self,
        events: tuple[SessionEvent, ...],
        *,
        tool_result_char_limit: int | None,
    ) -> tuple[tuple[Message, ...], int]:
        summary = self._latest_summary(events)
        cutoff = summary.through_sequence if summary else 0
        messages: list[Message] = [Message(role="system", content=self._system_prompt)]
        pruned = 0
        if summary is not None:
            messages.append(
                Message(role="system", content=f"Conversation summary:\n{summary.summary}")
            )
        structured_state = self._structured_state(events)
        if structured_state:
            messages.append(
                Message(role="system", content=f"Structured state:\n{structured_state}")
            )

        for event in events:
            if event.sequence <= cutoff:
                continue
            if isinstance(event, UserMessageEvent):
                messages.append(Message(role="user", content=event.content))
            elif isinstance(event, ModelMessageEvent):
                if not event.response.content and not event.response.tool_calls:
                    continue
                messages.append(
                    Message(
                        role="assistant",
                        content=event.response.content,
                        tool_calls=event.response.tool_calls,
                    )
                )
            elif isinstance(event, ToolResultEvent):
                content = event.content
                if tool_result_char_limit is not None and len(content) > tool_result_char_limit:
                    content = self._prune_tool_result(content, tool_result_char_limit)
                    pruned += 1
                messages.append(
                    Message(
                        role="tool",
                        content=content,
                        tool_call_id=event.tool_call_id,
                        name=event.name,
                    )
                )
        return tuple(messages), pruned

    async def _compress_oldest_turn(
        self, session: SessionHandle, *, turn_id: str, trace_id: str
    ) -> bool:
        current_summary = self._latest_summary(session.events)
        cutoff = current_summary.through_sequence if current_summary else 0
        completed = self._completed_turns(session.events, after_sequence=cutoff)
        if not completed:
            return False
        count = max(1, len(completed) - self._recent_turns)
        await self._append_summary(
            session,
            completed[:count],
            turn_id=turn_id,
            trace_id=trace_id,
        )
        return True

    async def _append_summary(
        self,
        session: SessionHandle,
        turns: list[list[SessionEvent]],
        *,
        turn_id: str,
        trace_id: str,
    ) -> None:
        current_summary = self._latest_summary(session.events)
        additions = [self._summarize_turn(turn) for turn in turns]
        prior = current_summary.summary if current_summary else ""
        combined = "\n".join(part for part in (prior, *additions) if part)
        if len(combined) > self._summary_max_chars:
            marker = "\n[Middle summary truncated]\n"
            available = max(0, self._summary_max_chars - len(marker))
            head_size = available // 2
            tail_size = available - head_size
            combined = combined[:head_size] + marker + combined[-tail_size:]
        prior_through = current_summary.through_sequence if current_summary else 0
        through = turns[-1][-1].sequence
        await session.append(
            SummaryEvent(
                summary=combined,
                through_sequence=through,
                turn_id=turn_id,
                trace_id=trace_id,
            )
        )
        await session.append(
            TraceEvent(
                kind=TraceKind.CONTEXT_COMPRESSED,
                data={
                    "from_sequence": prior_through + 1,
                    "through_sequence": through,
                    "summary_chars": len(combined),
                },
                turn_id=turn_id,
                trace_id=trace_id,
            )
        )

    @staticmethod
    def _estimate(messages: tuple[Message, ...], tools: tuple[ToolDefinition, ...]) -> int:
        message_chars = 0
        for message in messages:
            message_chars += len(message.role) + len(message.content or "")
            message_chars += len(message.tool_call_id or "") + len(message.name or "")
            for call in message.tool_calls:
                message_chars += len(call.id) + len(call.name) + len(call.raw_arguments)
        tool_chars = sum(
            len(tool.name)
            + len(tool.description)
            + len(json.dumps(tool.input_schema, ensure_ascii=False, separators=(",", ":")))
            for tool in tools
        )
        return message_chars + tool_chars

    @staticmethod
    def _prune_tool_result(content: str, limit: int) -> str:
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            value = {"ok": False, "error": {"code": "result_pruned"}}
        projected = ContextManager._project_value(value, string_limit=160, list_limit=8)
        serialized = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) <= limit:
            return serialized
        important = ContextManager._important_values(value)
        reduced = {"_pruned": True, "important": important}
        serialized = json.dumps(reduced, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) <= limit:
            return serialized
        marker = '{"_pruned":true}'
        return marker if len(marker) <= limit else "0"

    @staticmethod
    def _project_value(value: object, *, string_limit: int, list_limit: int) -> object:
        if isinstance(value, dict):
            projected: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    continue
                if isinstance(item, str) and len(item) > string_limit:
                    projected[key] = f"[pruned {len(item)} chars]"
                else:
                    projected[key] = ContextManager._project_value(
                        item, string_limit=string_limit, list_limit=list_limit
                    )
            return projected
        if isinstance(value, list):
            items = [
                ContextManager._project_value(
                    item, string_limit=string_limit, list_limit=list_limit
                )
                for item in value[:list_limit]
            ]
            if len(value) > list_limit:
                items.append({"_pruned_items": len(value) - list_limit})
            return items
        if isinstance(value, str) and len(value) > string_limit:
            return f"[pruned {len(value)} chars]"
        return value

    @staticmethod
    def _important_values(value: object) -> object:
        important_keys = {
            "id",
            "todo_id",
            "stable_id",
            "title",
            "status",
            "location",
            "date",
            "condition",
            "temperature",
            "unit",
            "source_time",
            "found",
            "code",
            "message",
        }
        if isinstance(value, dict):
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    continue
                if key in important_keys:
                    result[key] = ContextManager._project_value(
                        item, string_limit=80, list_limit=5
                    )
                elif isinstance(item, (dict, list)):
                    nested = ContextManager._important_values(item)
                    if nested not in ({}, []):
                        result[key] = nested
            return result
        if isinstance(value, list):
            return [ContextManager._important_values(item) for item in value[:5]]
        return value if isinstance(value, (str, int, float, bool)) or value is None else {}

    @staticmethod
    def _structured_state(events: tuple[SessionEvent, ...]) -> str:
        todos: dict[str, object] = {}
        weather: object | None = None
        for event in events:
            if not isinstance(event, ToolResultEvent):
                continue
            try:
                payload = json.loads(event.content)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                continue
            result = payload.get("result")
            if event.name == "todo":
                items = result if isinstance(result, list) else [result]
                for item in items:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        todos[item["id"]] = ContextManager._important_values(item)
            elif event.name == "get_weather":
                weather = ContextManager._important_values(result)
        state: dict[str, object] = {}
        if todos:
            state["todos"] = list(todos.values())
        if weather is not None:
            state["weather"] = weather
        if not state:
            return ""
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"))

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
