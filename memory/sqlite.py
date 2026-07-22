"""Durable SQLite adapters for sessions, events, and session-scoped todos."""

from __future__ import annotations

import asyncio
import builtins
import json
import sqlite3
from contextlib import AbstractAsyncContextManager, closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, cast
from uuid import uuid4

from memory.store import SessionHandle, SessionMetadata
from runtime.builtin_tools import TodoItem
from runtime.contracts import JSONObject, JSONValue, ModelResponse, ToolCall, Usage
from runtime.errors import (
    InvalidInputError,
    SessionExistsError,
    SessionNotFoundError,
    SessionOwnershipError,
    StorageError,
)
from runtime.events import (
    ModelMessageEvent,
    SessionEvent,
    SummaryEvent,
    ToolResultEvent,
    TraceEvent,
    TraceKind,
    UserMessageEvent,
)


class _SQLiteHandle:
    def __init__(self, database_path: Path, session_id: str, events: list[SessionEvent]) -> None:
        self._database_path = database_path
        self._session_id = session_id
        self._events = events

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        return tuple(self._events)

    async def append(self, event: SessionEvent) -> SessionEvent:
        """Persist one event without blocking the caller's event loop."""
        return await asyncio.to_thread(self._append_sync, event)

    def _append_sync(self, event: SessionEvent) -> SessionEvent:
        preliminary = replace(
            event,
            sequence=len(self._events) + 1,
            created_at=datetime.now(tz=UTC),
        )
        event_type, payload = _encode_event(preliminary)
        timestamp = preliminary.created_at.isoformat()
        try:
            with closing(_connect(self._database_path)) as connection, connection:
                latest_row = connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session_id = ?",
                    (self._session_id,),
                ).fetchone()
                latest = int(latest_row[0]) if latest_row is not None else 0
                if latest + 1 != preliminary.sequence:
                    raise StorageError("Session event sequencing failed safely.")
                cursor = connection.execute(
                    """
                    INSERT INTO events(
                        session_id, seq, type, payload_json, created_at, turn_id, trace_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._session_id,
                        preliminary.sequence,
                        event_type,
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        timestamp,
                        preliminary.turn_id,
                        preliminary.trace_id,
                    ),
                )
                event_id = cursor.lastrowid
                if event_id is None:
                    raise StorageError("Session event identifier allocation failed safely.")
                stored = replace(preliminary, event_id=str(event_id))
                if isinstance(stored, SummaryEvent):
                    connection.execute(
                        """
                        UPDATE sessions
                        SET summary = ?, summary_through_seq = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (stored.summary, stored.through_sequence, timestamp, self._session_id),
                    )
                else:
                    connection.execute(
                        "UPDATE sessions SET updated_at = ? WHERE id = ?",
                        (timestamp, self._session_id),
                    )
        except sqlite3.DatabaseError as exc:
            raise StorageError("Session events could not be persisted safely.") from exc
        self._events.append(stored)
        return stored


class _SQLiteSessionTurn(AbstractAsyncContextManager[SessionHandle]):
    def __init__(self, store: SQLiteSessionStore, user_id: str, session_id: str) -> None:
        self._store = store
        self._user_id = user_id
        self._session_id = session_id
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> SessionHandle:
        lock = await self._store._session_lock(self._session_id)
        await lock.acquire()
        self._lock = lock
        try:
            events = await asyncio.to_thread(
                self._store._load_owned_events, self._user_id, self._session_id
            )
        except BaseException:
            lock.release()
            self._lock = None
            raise
        return _SQLiteHandle(self._store.database_path, self._session_id, list(events))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._lock is not None:
            self._lock.release()


class SQLiteSessionStore:
    """Ownership-aware durable sessions with an append-only JSON event stream."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._locks: dict[str, asyncio.Lock] = {}
        self._catalog_lock = asyncio.Lock()
        try:
            _initialize_database(self.database_path)
        except sqlite3.DatabaseError as exc:
            raise StorageError("Session storage could not be initialized safely.") from exc

    async def create_session(self, user_id: str, session_id: str | None = None) -> str:
        """Create an isolated durable conversation window."""
        if not user_id.strip() or (session_id is not None and not session_id.strip()):
            raise InvalidInputError("User and session identifiers cannot be empty.")
        chosen_id = session_id or uuid4().hex
        await asyncio.to_thread(self._create_session_sync, user_id, chosen_id)
        return chosen_id

    def _create_session_sync(self, user_id: str, chosen_id: str) -> None:
        timestamp = datetime.now(tz=UTC).isoformat()
        try:
            with closing(_connect(self.database_path)) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO sessions(id, user_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chosen_id, user_id, timestamp, timestamp),
                )
        except sqlite3.IntegrityError as exc:
            raise SessionExistsError("That session identifier is already in use.") from exc
        except sqlite3.DatabaseError as exc:
            raise StorageError("The session could not be created safely.") from exc

    def session_turn(
        self, user_id: str, session_id: str
    ) -> AbstractAsyncContextManager[SessionHandle]:
        """Serialize and persist one complete session turn."""
        return _SQLiteSessionTurn(self, user_id, session_id)

    async def get_events(self, user_id: str, session_id: str) -> tuple[SessionEvent, ...]:
        """Load ordered events after verifying ownership."""
        lock = await self._session_lock(session_id)
        async with lock:
            return await asyncio.to_thread(self._load_owned_events, user_id, session_id)

    async def get_session(self, user_id: str, session_id: str) -> SessionMetadata:
        """Load session metadata after verifying ownership."""
        lock = await self._session_lock(session_id)
        async with lock:
            return await asyncio.to_thread(
                self._load_owned_metadata, user_id, session_id
            )

    async def _session_lock(self, session_id: str) -> asyncio.Lock:
        async with self._catalog_lock:
            return self._locks.setdefault(session_id, asyncio.Lock())

    def _load_owned_events(self, user_id: str, session_id: str) -> tuple[SessionEvent, ...]:
        try:
            with closing(_connect(self.database_path)) as connection:
                owner_row = connection.execute(
                    "SELECT user_id FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if owner_row is None:
                    raise SessionNotFoundError("The requested session does not exist.")
                if str(owner_row[0]) != user_id:
                    raise SessionOwnershipError("The requested session belongs to another user.")
                rows = connection.execute(
                    """
                    SELECT event_id, seq, type, payload_json, created_at, turn_id, trace_id
                    FROM events
                    WHERE session_id = ?
                    ORDER BY seq
                    """,
                    (session_id,),
                ).fetchall()
            return tuple(
                _decode_event(
                    event_id=str(row[0]),
                    event_type=str(row[2]),
                    payload_json=str(row[3]),
                    sequence=int(row[1]),
                    created_at=datetime.fromisoformat(str(row[4])),
                    turn_id=str(row[5]),
                    trace_id=str(row[6]),
                )
                for row in rows
            )
        except (sqlite3.DatabaseError, ValueError, TypeError) as exc:
            raise StorageError("Session storage could not be read safely.") from exc

    def _load_owned_metadata(self, user_id: str, session_id: str) -> SessionMetadata:
        try:
            with closing(_connect(self.database_path)) as connection:
                row = connection.execute(
                    """
                    SELECT user_id, created_at, updated_at,
                           (SELECT COUNT(*) FROM events WHERE session_id = sessions.id)
                    FROM sessions WHERE id = ?
                    """,
                    (session_id,),
                ).fetchone()
            if row is None:
                raise SessionNotFoundError("The requested session does not exist.")
            if str(row[0]) != user_id:
                raise SessionOwnershipError("The requested session belongs to another user.")
            return SessionMetadata(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.fromisoformat(str(row[1])),
                updated_at=datetime.fromisoformat(str(row[2])),
                event_count=int(row[3]),
            )
        except (sqlite3.DatabaseError, ValueError, TypeError) as exc:
            raise StorageError("Session metadata could not be read safely.") from exc


class SQLiteTodoStore:
    """Durable todo storage isolated by session identifier."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._lock = asyncio.Lock()
        try:
            _initialize_database(self._database_path)
        except sqlite3.DatabaseError as exc:
            raise StorageError("Todo storage could not be initialized safely.") from exc

    async def add(self, session_id: str, title: str) -> TodoItem:
        """Persist a new item for one session."""
        item = TodoItem(id=uuid4().hex[:12], title=title)
        async with self._lock:
            await asyncio.to_thread(self._add_sync, session_id, item)
        return item

    def _add_sync(self, session_id: str, item: TodoItem) -> None:
        timestamp = datetime.now(tz=UTC).isoformat()
        try:
            with closing(_connect(self._database_path)) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO todos(id, session_id, title, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (item.id, session_id, item.title, timestamp, timestamp),
                )
        except sqlite3.DatabaseError as exc:
            raise StorageError("Todo storage could not persist an item safely.") from exc

    async def list(self, session_id: str) -> list[TodoItem]:
        """Load todos from one session in creation order."""
        async with self._lock:
            return await asyncio.to_thread(self._list_sync, session_id)

    def _list_sync(self, session_id: str) -> builtins.list[TodoItem]:
        try:
            with closing(_connect(self._database_path)) as connection:
                rows = connection.execute(
                    """
                    SELECT id, title, status
                    FROM todos
                    WHERE session_id = ?
                    ORDER BY created_at, id
                    """,
                    (session_id,),
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise StorageError("Todo storage could not be read safely.") from exc
        return [
            TodoItem(id=str(row[0]), title=str(row[1]), status=_todo_status(row[2]))
            for row in rows
        ]

    async def complete(self, session_id: str, todo_id: str) -> TodoItem | None:
        """Complete an item without crossing its session boundary."""
        async with self._lock:
            return await asyncio.to_thread(self._complete_sync, session_id, todo_id)

    def _complete_sync(self, session_id: str, todo_id: str) -> TodoItem | None:
        timestamp = datetime.now(tz=UTC).isoformat()
        try:
            with closing(_connect(self._database_path)) as connection, connection:
                row = connection.execute(
                    "SELECT title FROM todos WHERE id = ? AND session_id = ?",
                    (todo_id, session_id),
                ).fetchone()
                if row is None:
                    return None
                connection.execute(
                    """
                    UPDATE todos SET status = 'completed', updated_at = ?
                    WHERE id = ? AND session_id = ?
                    """,
                    (timestamp, todo_id, session_id),
                )
        except sqlite3.DatabaseError as exc:
            raise StorageError("Todo storage could not update an item safely.") from exc
        return TodoItem(id=todo_id, title=str(row[0]), status="completed")


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=5.0)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _initialize_database(database_path: Path) -> None:
    with closing(_connect(database_path)) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                summary TEXT,
                summary_through_seq INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                turn_id TEXT NOT NULL DEFAULT '',
                trace_id TEXT NOT NULL DEFAULT '',
                UNIQUE(session_id, seq)
            );

            CREATE TABLE IF NOT EXISTS todos (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'completed')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS events_session_sequence
            ON events(session_id, seq);

            CREATE INDEX IF NOT EXISTS todos_session_created
            ON todos(session_id, created_at);
            """
        )
        _ensure_column(connection, "events", "turn_id", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "events", "trace_id", "TEXT NOT NULL DEFAULT ''")
        _migrate_todos_schema(connection)
        connection.commit()


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, declaration: str
) -> None:
    columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _migrate_todos_schema(connection: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(todos)")}
    if "text" not in columns:
        return
    connection.executescript(
        """
        ALTER TABLE todos RENAME TO todos_legacy;

        CREATE TABLE todos (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'completed')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        INSERT INTO todos(id, session_id, title, status, created_at, updated_at)
        SELECT id, session_id, text,
               CASE WHEN completed = 1 THEN 'completed' ELSE 'pending' END,
               created_at, updated_at
        FROM todos_legacy;

        DROP TABLE todos_legacy;

        CREATE INDEX IF NOT EXISTS todos_session_created
        ON todos(session_id, created_at);
        """
    )


def _todo_status(value: object) -> Literal["pending", "completed"]:
    text = str(value)
    if text not in {"pending", "completed"}:
        raise StorageError("Stored todo status is invalid.")
    return cast(Literal["pending", "completed"], text)


def _encode_event(event: SessionEvent) -> tuple[str, JSONObject]:
    if isinstance(event, UserMessageEvent):
        return "user_message", {"content": event.content}
    if isinstance(event, ModelMessageEvent):
        response = event.response
        tool_calls: list[JSONValue] = [
            {
                "id": call.id,
                "name": call.name,
                "raw_arguments": call.raw_arguments,
                "arguments": call.arguments,
                "parse_error": call.parse_error,
            }
            for call in response.tool_calls
        ]
        return "model_message", {
            "content": response.content,
            "tool_calls": tool_calls,
            "reasoning": response.reasoning,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
    if isinstance(event, ToolResultEvent):
        return "tool_result", {
            "tool_call_id": event.tool_call_id,
            "name": event.name,
            "content": event.content,
        }
    if isinstance(event, SummaryEvent):
        return "summary", {
            "summary": event.summary,
            "through_sequence": event.through_sequence,
        }
    if isinstance(event, TraceEvent):
        return "trace", {"kind": event.kind.value, "data": event.data}
    raise TypeError(f"unsupported session event: {type(event).__name__}")


def _decode_event(
    *,
    event_id: str,
    event_type: str,
    payload_json: str,
    sequence: int,
    created_at: datetime,
    turn_id: str,
    trace_id: str,
) -> SessionEvent:
    payload = _json_object(payload_json)
    if event_type == "user_message":
        return UserMessageEvent(
            content=_required_str(payload, "content"),
            sequence=sequence,
            created_at=created_at,
            turn_id=turn_id,
            trace_id=trace_id,
            event_id=event_id,
        )
    if event_type == "model_message":
        return ModelMessageEvent(
            response=_decode_model_response(payload),
            sequence=sequence,
            created_at=created_at,
            turn_id=turn_id,
            trace_id=trace_id,
            event_id=event_id,
        )
    if event_type == "tool_result":
        return ToolResultEvent(
            tool_call_id=_required_str(payload, "tool_call_id"),
            name=_required_str(payload, "name"),
            content=_required_str(payload, "content"),
            sequence=sequence,
            created_at=created_at,
            turn_id=turn_id,
            trace_id=trace_id,
            event_id=event_id,
        )
    if event_type == "summary":
        return SummaryEvent(
            summary=_required_str(payload, "summary"),
            through_sequence=_required_int(payload, "through_sequence"),
            sequence=sequence,
            created_at=created_at,
            turn_id=turn_id,
            trace_id=trace_id,
            event_id=event_id,
        )
    if event_type == "trace":
        return TraceEvent(
            kind=TraceKind(_required_str(payload, "kind")),
            data=_required_object(payload, "data"),
            sequence=sequence,
            created_at=created_at,
            turn_id=turn_id,
            trace_id=trace_id,
            event_id=event_id,
        )
    raise ValueError(f"unknown stored session event type: {event_type}")


def _decode_model_response(payload: JSONObject) -> ModelResponse:
    calls: list[ToolCall] = []
    for value in _required_array(payload, "tool_calls"):
        if not isinstance(value, dict):
            raise ValueError("stored tool call must be an object")
        item = value
        arguments_value = item.get("arguments")
        if arguments_value is not None and not isinstance(arguments_value, dict):
            raise ValueError("stored tool arguments must be an object or null")
        calls.append(
            ToolCall(
                id=_required_str(item, "id"),
                name=_required_str(item, "name"),
                raw_arguments=_required_str(item, "raw_arguments"),
                arguments=arguments_value,
                parse_error=_optional_str(item, "parse_error"),
            )
        )
    usage = _required_object(payload, "usage")
    return ModelResponse(
        content=_optional_str(payload, "content"),
        tool_calls=tuple(calls),
        reasoning=_optional_str(payload, "reasoning"),
        usage=Usage(
            prompt_tokens=_required_int(usage, "prompt_tokens"),
            completion_tokens=_required_int(usage, "completion_tokens"),
            total_tokens=_required_int(usage, "total_tokens"),
        ),
    )


def _json_object(payload_json: str) -> JSONObject:
    value = cast(object, json.loads(payload_json))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("stored event payload must be a JSON object")
    return value


def _required_str(payload: JSONObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"stored field '{key}' must be a string")
    return value


def _optional_str(payload: JSONObject, key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"stored field '{key}' must be a string or null")
    return value


def _required_int(payload: JSONObject, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"stored field '{key}' must be an integer")
    return value


def _required_object(payload: JSONObject, key: str) -> JSONObject:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"stored field '{key}' must be an object")
    return value


def _required_array(payload: JSONObject, key: str) -> list[JSONValue]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"stored field '{key}' must be an array")
    return value
