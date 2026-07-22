"""Typed append-only events used to recover conversations and audit runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from runtime.contracts import JSONObject, ModelResponse


class TraceKind(StrEnum):
    """Observable execution milestones."""

    TURN_STARTED = "turn_started"
    CONTEXT_BUILT = "context_built"
    CONTEXT_COMPRESSED = "context_compressed"
    MODEL_STARTED = "model_started"
    MODEL_COMPLETED = "model_completed"
    MODEL_RETRY = "model_retry"
    FORMAT_REPAIR = "format_repair"
    MODEL_DECISION = "model_decision"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    LOOP_DETECTED = "loop_detected"
    TURN_CANCELLED = "turn_cancelled"
    TURN_FINISHED = "turn_finished"


@dataclass(frozen=True, slots=True)
class UserMessageEvent:
    """A user message persisted in a session."""

    content: str
    sequence: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=UTC))
    turn_id: str = ""
    trace_id: str = ""
    event_id: str = ""


@dataclass(frozen=True, slots=True)
class ModelMessageEvent:
    """A normalized assistant message; hidden reasoning is deliberately omitted."""

    response: ModelResponse
    sequence: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=UTC))
    turn_id: str = ""
    trace_id: str = ""
    event_id: str = ""


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """A tool response paired to its assistant tool-call identifier."""

    tool_call_id: str
    name: str
    content: str
    sequence: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=UTC))
    turn_id: str = ""
    trace_id: str = ""
    event_id: str = ""


@dataclass(frozen=True, slots=True)
class SummaryEvent:
    """A compact replacement for completed events through a sequence number."""

    summary: str
    through_sequence: int
    sequence: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=UTC))
    turn_id: str = ""
    trace_id: str = ""
    event_id: str = ""


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """A sanitized execution event suitable for logs and API traces."""

    kind: TraceKind
    data: JSONObject
    sequence: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=UTC))
    turn_id: str = ""
    trace_id: str = ""
    event_id: str = ""


type SessionEvent = (
    UserMessageEvent | ModelMessageEvent | ToolResultEvent | SummaryEvent | TraceEvent
)
