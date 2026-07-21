"""Thin HTTP translation around the provider-neutral runtime."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.runner import AgentRunner
from memory.store import SessionStore
from runtime.contracts import JSONObject
from runtime.errors import TinyAgentError
from runtime.events import TraceEvent


class CreateSessionRequest(BaseModel):
    """Create an owned, isolated conversation window."""

    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=200)
    session_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("user_id", "session_id")
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
    user_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=100_000)

    @field_validator("user_id", "message")
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


class MessageResponse(BaseModel):
    """Terminal run result."""

    session_id: str
    status: Literal["completed", "error"]
    answer: str | None
    error: ErrorResponse | None
    trace: list[TraceResponse]


def create_app(*, runner: AgentRunner, sessions: SessionStore) -> FastAPI:
    """Create an application with dependencies supplied by the composition root."""
    app = FastAPI(title="Tiny Agent", version="0.1.0")

    @app.post("/v1/sessions", response_model=CreateSessionResponse, status_code=201)
    async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
        try:
            session_id = await sessions.create_session(request.user_id, request.session_id)
        except TinyAgentError as exc:
            info = exc.as_info()
            raise HTTPException(
                status_code=422 if info.code == "invalid_input" else 409,
                detail={"code": info.code, "message": info.message},
            ) from exc
        return CreateSessionResponse(session_id=session_id)

    @app.post("/v1/sessions/{session_id}/messages", response_model=MessageResponse)
    async def send_message(session_id: str, request: MessageRequest) -> MessageResponse:
        result = await runner.run(
            user_id=request.user_id,
            session_id=session_id,
            user_input=request.message,
        )
        if result.error and result.error.code in {"session_forbidden", "session_not_found"}:
            status_code = 403 if result.error.code == "session_forbidden" else 404
            raise HTTPException(
                status_code=status_code,
                detail={"code": result.error.code, "message": result.error.message},
            )
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
        )

    return app


def _trace_response(event: TraceEvent) -> TraceResponse:
    return TraceResponse(sequence=event.sequence, kind=event.kind, data=event.data)
