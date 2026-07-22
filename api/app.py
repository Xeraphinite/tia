"""Thin HTTP translation around the provider-neutral runtime."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.runner import AgentRunner
from memory.store import SessionMetadata, SessionStore
from runtime.contracts import JSONObject, JSONValue, Usage
from runtime.errors import (
    SessionNotFoundError,
    SessionOwnershipError,
    TinyAgentError,
)
from runtime.events import (
    ModelMessageEvent,
    SessionEvent,
    SummaryEvent,
    ToolResultEvent,
    TraceEvent,
    UserMessageEvent,
)


class CreateSessionRequest(BaseModel):
    """Create an owned, isolated conversation window."""

    model_config = ConfigDict(extra="forbid")
    session_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("session_id")
    @classmethod
    def reject_blank_ids(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("identifier cannot be blank")
        return value


class CreateSessionResponse(BaseModel):
    """Created session identifier."""

    session_id: str


class MessageRequest(BaseModel):
    """One user turn."""

    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=100_000)

    @field_validator("message")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value


class ErrorResponse(BaseModel):
    """Stable public error shape."""

    code: str
    message: str


class TraceResponse(BaseModel):
    """Sanitized execution event."""

    sequence: int
    kind: str
    data: JSONObject
    turn_id: str
    trace_id: str
    event_id: str


class UsageResponse(BaseModel):
    """Normalized aggregate model usage for one turn."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MessageResponse(BaseModel):
    """Terminal run result."""

    session_id: str
    status: Literal["completed", "step_limit", "timeout", "cancelled", "failed"]
    answer: str | None
    error: ErrorResponse | None
    trace: list[TraceResponse]
    turn_id: str
    trace_id: str
    usage: UsageResponse


class SessionResponse(BaseModel):
    """Owned session metadata."""

    session_id: str
    created_at: datetime
    updated_at: datetime
    event_count: int


class SessionEventResponse(BaseModel):
    """One sanitized durable session event."""

    event_id: str
    sequence: int
    type: str
    created_at: datetime
    turn_id: str
    trace_id: str
    data: JSONObject


UserIdentity = Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=200)]


def create_app(*, runner: AgentRunner, sessions: SessionStore) -> FastAPI:
    """Create an application with dependencies supplied by the composition root."""
    app = FastAPI(title="Tiny Agent", version="0.1.0")

    @app.post("/v1/sessions", response_model=CreateSessionResponse, status_code=201)
    async def create_session(
        request: CreateSessionRequest, user_id: UserIdentity
    ) -> CreateSessionResponse:
        user_id = _clean_identity(user_id)
        try:
            session_id = await sessions.create_session(user_id, request.session_id)
        except TinyAgentError as exc:
            info = exc.as_info()
            raise HTTPException(
                status_code=422 if info.code == "invalid_input" else 409,
                detail={"code": info.code, "message": info.message},
            ) from exc
        return CreateSessionResponse(session_id=session_id)

    @app.post("/v1/sessions/{session_id}/messages", response_model=MessageResponse)
    async def send_message(
        session_id: str, request: MessageRequest, user_id: UserIdentity
    ) -> MessageResponse:
        user_id = _clean_identity(user_id)
        result = await runner.run(
            user_id=user_id,
            session_id=session_id,
            user_input=request.message,
        )
        if result.error and result.error.code in {"session_forbidden", "session_not_found"}:
            raise _session_not_found()
        error = (
            ErrorResponse(code=result.error.code, message=result.error.message)
            if result.error
            else None
        )
        return MessageResponse(
            session_id=result.session_id,
            status=result.status,
            answer=result.answer,
            error=error,
            trace=[_trace_response(event) for event in result.trace],
            turn_id=result.turn_id,
            trace_id=result.trace_id,
            usage=_usage_response(result.usage),
        )

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str, user_id: UserIdentity) -> SessionResponse:
        user_id = _clean_identity(user_id)
        try:
            metadata = await sessions.get_session(user_id, session_id)
        except (SessionNotFoundError, SessionOwnershipError) as exc:
            raise _session_not_found() from exc
        return _session_response(metadata)

    @app.get(
        "/v1/sessions/{session_id}/events",
        response_model=list[SessionEventResponse],
    )
    async def get_events(
        session_id: str, user_id: UserIdentity
    ) -> list[SessionEventResponse]:
        user_id = _clean_identity(user_id)
        try:
            events = await sessions.get_events(user_id, session_id)
        except (SessionNotFoundError, SessionOwnershipError) as exc:
            raise _session_not_found() from exc
        return [_event_response(event) for event in events]

    return app


def _trace_response(event: TraceEvent) -> TraceResponse:
    return TraceResponse(
        sequence=event.sequence,
        kind=event.kind,
        data=event.data,
        turn_id=event.turn_id,
        trace_id=event.trace_id,
        event_id=event.event_id,
    )


def _usage_response(usage: Usage) -> UsageResponse:
    return UsageResponse(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def _clean_identity(user_id: str) -> str:
    cleaned = user_id.strip()
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_input", "message": "User identity cannot be blank."},
        )
    return cleaned


def _session_not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "code": "session_not_found",
            "message": "The requested session does not exist.",
        },
    )


def _session_response(metadata: SessionMetadata) -> SessionResponse:
    return SessionResponse(
        session_id=metadata.session_id,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        event_count=metadata.event_count,
    )


def _event_response(event: SessionEvent) -> SessionEventResponse:
    event_type: str
    data: JSONObject
    if isinstance(event, UserMessageEvent):
        event_type, data = "user_message", {"content": event.content}
    elif isinstance(event, ModelMessageEvent):
        calls: list[JSONValue] = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "parse_error": call.parse_error,
            }
            for call in event.response.tool_calls
        ]
        event_type, data = "model_message", {
            "content": event.response.content,
            "tool_calls": calls,
            "usage": {
                "prompt_tokens": event.response.usage.prompt_tokens,
                "completion_tokens": event.response.usage.completion_tokens,
                "total_tokens": event.response.usage.total_tokens,
            },
        }
    elif isinstance(event, ToolResultEvent):
        event_type, data = "tool_result", {
            "tool_call_id": event.tool_call_id,
            "name": event.name,
            "content": event.content,
        }
    elif isinstance(event, SummaryEvent):
        event_type, data = "summary", {
            "summary": event.summary,
            "through_sequence": event.through_sequence,
        }
    elif isinstance(event, TraceEvent):
        event_type, data = "trace", {"kind": event.kind.value, "data": event.data}
    else:
        raise TypeError(f"unsupported event type: {type(event).__name__}")
    return SessionEventResponse(
        event_id=event.event_id,
        sequence=event.sequence,
        type=event_type,
        created_at=event.created_at,
        turn_id=event.turn_id,
        trace_id=event.trace_id,
        data=data,
    )
