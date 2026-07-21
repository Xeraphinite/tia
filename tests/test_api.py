"""HTTP boundary tests using the complete application stack."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from agent.runner import AgentRunner
from api.app import create_app
from memory.store import InMemorySessionStore
from runtime.builtin_tools import create_builtin_registry
from runtime.contracts import ModelResponse
from tests.helpers import ScriptedModel


async def test_api_creates_session_and_continues_conversation() -> None:
    sessions = InMemorySessionStore()
    model = ScriptedModel([ModelResponse(content="hello"), ModelResponse(content="again")])
    runner = AgentRunner(model=model, tools=create_builtin_registry(), sessions=sessions)
    app = create_app(runner=runner, sessions=sessions)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/v1/sessions", json={"user_id": "user-a"})
        session_id = created.json()["session_id"]
        first = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"user_id": "user-a", "message": "Hi"},
        )
        second = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"user_id": "user-a", "message": "Again"},
        )

    assert created.status_code == 201
    assert first.json()["answer"] == "hello"
    assert second.json()["answer"] == "again"
    assert second.json()["trace"][-1]["kind"] == "turn_finished"


async def test_api_rejects_duplicate_session_id_and_forbids_wrong_owner() -> None:
    sessions = InMemorySessionStore()
    runner = AgentRunner(
        model=ScriptedModel([]), tools=create_builtin_registry(), sessions=sessions
    )
    app = create_app(runner=runner, sessions=sessions)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/v1/sessions", json={"user_id": "user-a", "session_id": "fixed"}
        )
        duplicate = await client.post(
            "/v1/sessions", json={"user_id": "user-a", "session_id": "fixed"}
        )
        forbidden = await client.post(
            "/v1/sessions/fixed/messages",
            json={"user_id": "user-b", "message": "intrude"},
        )

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "session_exists"
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"]["code"] == "session_forbidden"


async def test_api_rejects_blank_public_input() -> None:
    sessions = InMemorySessionStore()
    runner = AgentRunner(
        model=ScriptedModel([]), tools=create_builtin_registry(), sessions=sessions
    )
    app = create_app(runner=runner, sessions=sessions)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/sessions", json={"user_id": "   "})

    assert response.status_code == 422
