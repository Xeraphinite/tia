"""Integration tests for the HTTP boundary and application stack."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from agent.runner import AgentRunner
from api.app import create_app
from memory.store import InMemorySessionStore
from runtime.builtin_tools import create_builtin_registry
from runtime.contracts import ModelResponse
from tests.helpers import ScriptedModel

pytestmark = pytest.mark.integration


async def test_api_creates_session_and_continues_conversation() -> None:
    sessions = InMemorySessionStore()
    model = ScriptedModel([ModelResponse(content="hello"), ModelResponse(content="again")])
    runner = AgentRunner(model=model, tools=create_builtin_registry(), sessions=sessions)
    app = create_app(runner=runner, sessions=sessions)
    headers = {"X-User-ID": "user-a"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/v1/sessions", json={}, headers=headers)
        session_id = created.json()["session_id"]
        first = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"message": "Hi"},
            headers=headers,
        )
        second = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"message": "Again"},
            headers=headers,
        )
        metadata = await client.get(f"/v1/sessions/{session_id}", headers=headers)
        events = await client.get(f"/v1/sessions/{session_id}/events", headers=headers)

    assert created.status_code == 201
    assert first.json()["answer"] == "hello"
    assert second.json()["answer"] == "again"
    assert second.json()["trace"][-1]["kind"] == "turn_finished"
    assert second.json()["turn_id"]
    assert second.json()["trace_id"]
    assert second.json()["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert all(
        event["trace_id"] == second.json()["trace_id"]
        for event in second.json()["trace"]
    )
    assert metadata.status_code == 200
    assert metadata.json()["event_count"] == len(events.json())
    assert events.json()
    assert all(event["event_id"] for event in events.json())


async def test_api_rejects_duplicate_session_id_and_forbids_wrong_owner() -> None:
    sessions = InMemorySessionStore()
    runner = AgentRunner(
        model=ScriptedModel([]), tools=create_builtin_registry(), sessions=sessions
    )
    app = create_app(runner=runner, sessions=sessions)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/v1/sessions", json={"session_id": "fixed"}, headers={"X-User-ID": "user-a"}
        )
        duplicate = await client.post(
            "/v1/sessions", json={"session_id": "fixed"}, headers={"X-User-ID": "user-a"}
        )
        forbidden = await client.post(
            "/v1/sessions/fixed/messages",
            json={"message": "intrude"},
            headers={"X-User-ID": "user-b"},
        )
        missing = await client.post(
            "/v1/sessions/missing/messages",
            json={"message": "intrude"},
            headers={"X-User-ID": "user-b"},
        )
        forbidden_read = await client.get(
            "/v1/sessions/fixed/events", headers={"X-User-ID": "user-b"}
        )
        missing_read = await client.get(
            "/v1/sessions/missing/events", headers={"X-User-ID": "user-b"}
        )

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "session_exists"
    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json()
    assert forbidden_read.status_code == missing_read.status_code == 404
    assert forbidden_read.json() == missing_read.json()


async def test_api_rejects_blank_public_input() -> None:
    sessions = InMemorySessionStore()
    runner = AgentRunner(
        model=ScriptedModel([]), tools=create_builtin_registry(), sessions=sessions
    )
    app = create_app(runner=runner, sessions=sessions)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/sessions", json={}, headers={"X-User-ID": "   "}
        )
        missing = await client.post("/v1/sessions", json={})

    assert response.status_code == 422
    assert missing.status_code == 422
