"""Session storage contracts and the deterministic in-memory adapter."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import TracebackType
from typing import Protocol
from uuid import uuid4

from runtime.errors import (
    InvalidInputError,
    SessionExistsError,
    SessionNotFoundError,
    SessionOwnershipError,
)
from runtime.events import SessionEvent


class SessionHandle(Protocol):
    """A session transaction held for one complete turn."""

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        """Return the current append-only history."""
        ...

    def append(self, event: SessionEvent) -> SessionEvent:
        """Append and sequence one event."""
        ...


class SessionStore(Protocol):
    """Ownership-aware session persistence."""

    async def create_session(self, user_id: str, session_id: str | None = None) -> str:
        """Create an isolated conversation window."""
        ...

    def session_turn(
        self, user_id: str, session_id: str
    ) -> AbstractAsyncContextManager[SessionHandle]:
        """Serialize one complete turn for a session."""
        ...

    async def get_events(self, user_id: str, session_id: str) -> tuple[SessionEvent, ...]:
        """Read a session after verifying ownership."""
        ...


@dataclass(slots=True)
class _SessionData:
    user_id: str
    events: list[SessionEvent] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _InMemoryHandle:
    def __init__(self, data: _SessionData) -> None:
        self._data = data

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        return tuple(self._data.events)

    def append(self, event: SessionEvent) -> SessionEvent:
        stored = replace(
            event,
            sequence=len(self._data.events) + 1,
            created_at=datetime.now(tz=UTC),
        )
        self._data.events.append(stored)
        return stored


class _SessionTurn:
    def __init__(self, store: InMemorySessionStore, user_id: str, session_id: str) -> None:
        self._store = store
        self._user_id = user_id
        self._session_id = session_id
        self._data: _SessionData | None = None

    async def __aenter__(self) -> SessionHandle:
        data = await self._store._owned_session(self._user_id, self._session_id)
        await data.lock.acquire()
        self._data = data
        return _InMemoryHandle(data)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._data is not None:
            self._data.lock.release()


class InMemorySessionStore:
    """An ownership-aware store that serializes turns per session."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionData] = {}
        self._catalog_lock = asyncio.Lock()

    async def create_session(self, user_id: str, session_id: str | None = None) -> str:
        if not user_id.strip() or (session_id is not None and not session_id.strip()):
            raise InvalidInputError("User and session identifiers cannot be empty.")
        chosen_id = session_id or uuid4().hex
        async with self._catalog_lock:
            if chosen_id in self._sessions:
                raise SessionExistsError("That session identifier is already in use.")
            self._sessions[chosen_id] = _SessionData(user_id=user_id)
        return chosen_id

    def session_turn(self, user_id: str, session_id: str) -> _SessionTurn:
        return _SessionTurn(self, user_id, session_id)

    async def get_events(self, user_id: str, session_id: str) -> tuple[SessionEvent, ...]:
        data = await self._owned_session(user_id, session_id)
        async with data.lock:
            return tuple(data.events)

    async def _owned_session(self, user_id: str, session_id: str) -> _SessionData:
        async with self._catalog_lock:
            data = self._sessions.get(session_id)
        if data is None:
            raise SessionNotFoundError("The requested session does not exist.")
        if data.user_id != user_id:
            raise SessionOwnershipError("The requested session belongs to another user.")
        return data
