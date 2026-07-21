"""Tests for the Streamlit frontend's narrow API client."""

from __future__ import annotations

import json

import pytest

from frontend.client import (
    FrontendAPIError,
    TinyAgentClient,
    TransportResponse,
)


class StubTransport:
    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, object], float]] = []

    def post(self, url: str, payload: bytes, timeout_seconds: float) -> TransportResponse:
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)
        self.requests.append((url, parsed, timeout_seconds))
        return self.responses.pop(0)


def _response(status: int, payload: object) -> TransportResponse:
    return TransportResponse(status=status, body=json.dumps(payload).encode())


def test_client_creates_session_and_sends_message() -> None:
    transport = StubTransport(
        [
            _response(201, {"session_id": "session-1"}),
            _response(
                200,
                {
                    "session_id": "session-1",
                    "status": "completed",
                    "answer": "It is 22°C.",
                    "error": None,
                    "trace": [
                        {"sequence": 2, "kind": "tool_started", "data": {"name": "weather"}}
                    ],
                },
            ),
        ]
    )
    client = TinyAgentClient("http://localhost:8000/", transport=transport)

    session_id = client.create_session("user-a")
    result = client.send_message("user-a", session_id, "Weather?")

    assert session_id == "session-1"
    assert result.answer == "It is 22°C."
    assert result.trace[0].data == {"name": "weather"}
    assert transport.requests[0][0] == "http://localhost:8000/v1/sessions"
    assert transport.requests[1][1] == {"user_id": "user-a", "message": "Weather?"}


def test_client_surfaces_safe_api_error() -> None:
    transport = StubTransport(
        [
            _response(
                403,
                {"detail": {"code": "session_forbidden", "message": "Wrong owner."}},
            )
        ]
    )
    client = TinyAgentClient("http://localhost:8000", transport=transport)

    with pytest.raises(FrontendAPIError) as caught:
        client.send_message("user-b", "session-1", "Hello")

    assert caught.value.code == "session_forbidden"
    assert caught.value.safe_message == "Wrong owner."


def test_client_rejects_invalid_endpoint_and_response() -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        TinyAgentClient("localhost:8000")

    client = TinyAgentClient(
        "http://localhost:8000", transport=StubTransport([_response(200, {"unexpected": True})])
    )
    with pytest.raises(FrontendAPIError) as caught:
        client.create_session("user-a")

    assert caught.value.code == "invalid_api_response"

