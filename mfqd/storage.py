"""SQLite persistence for MFQd conversations and immutable message branches."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import TypeAdapter

from mfqd.models import (
    ContentPart,
    CreateSessionRequest,
    ErrorDetail,
    ForkSessionRequest,
    Message,
    MessageRole,
    ResponseResource,
    ResponseStatus,
    SessionMode,
    SessionResource,
    SessionState,
    TokenUsage,
)

SCHEMA_VERSION = 1
_CONTENT_PARTS = TypeAdapter(list[ContentPart])


class StorageError(RuntimeError):
    """Base class for persistent conversation errors."""


class SessionNotFoundError(StorageError):
    pass


class MessageNotFoundError(StorageError):
    pass


class RevisionConflictError(StorageError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"session revision mismatch: expected {expected}, actual {actual}")
        self.expected = expected
        self.actual = actual


class IdempotencyConflictError(StorageError):
    pass


class ResponseInProgressError(StorageError):
    pass


class ResponseNotFoundError(StorageError):
    pass


class InvalidResponseStateError(StorageError):
    pass


@dataclass(frozen=True)
class BeginResponseResult:
    session: SessionResource
    response: ResponseResource
    started: bool


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


class SessionStore:
    """Own short SQLite transactions; never hold a transaction during inference."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise StorageError(f"SQLite refused WAL mode: {journal_mode}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    state TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 0),
                    title TEXT,
                    runtime_instance_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    parts_json TEXT NOT NULL,
                    parent_id TEXT REFERENCES messages(id),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_messages (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
                    message_id TEXT NOT NULL REFERENCES messages(id),
                    PRIMARY KEY (session_id, ordinal),
                    UNIQUE (session_id, message_id)
                );

                CREATE INDEX IF NOT EXISTS session_messages_message_id
                    ON session_messages(message_id);

                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    request_id TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_message_id TEXT NOT NULL REFERENCES messages(id),
                    output_json TEXT NOT NULL,
                    finish_reason TEXT,
                    usage_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE (session_id, request_id)
                );

                CREATE INDEX IF NOT EXISTS responses_session_status
                    ON responses(session_id, status);
                """
            )
            existing = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(existing["value"]) != SCHEMA_VERSION:
                raise StorageError(
                    f"unsupported database schema {existing['value']}; expected {SCHEMA_VERSION}"
                )

    def journal_mode(self) -> str:
        with self._connection() as connection:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def create_session(
        self,
        request: CreateSessionRequest,
        *,
        session_id: UUID | None = None,
        now: datetime | None = None,
    ) -> SessionResource:
        created_at = now or _utcnow()
        identifier = session_id or uuid4()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions(
                    id, model, mode, state, revision, title, runtime_instance_id,
                    created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, 0, ?, NULL, ?, ?, ?)
                """,
                (
                    str(identifier),
                    request.model,
                    request.mode.value,
                    SessionState.IDLE.value,
                    request.title,
                    _timestamp(created_at),
                    _timestamp(created_at),
                    json.dumps(request.metadata, separators=(",", ":"), sort_keys=True),
                ),
            )
        return self.get_session(identifier)

    def get_session(self, session_id: UUID) -> SessionResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (str(session_id),),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(str(session_id))
        return self._session_from_row(row)

    def list_sessions(self, *, limit: int = 50, offset: int = 0) -> list[SessionResource]:
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM sessions
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def list_messages(self, session_id: UUID) -> list[Message]:
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?",
                (str(session_id),),
            ).fetchone()
            if exists is None:
                raise SessionNotFoundError(str(session_id))
            rows = connection.execute(
                """
                SELECT m.*
                FROM session_messages AS sm
                JOIN messages AS m ON m.id = sm.message_id
                WHERE sm.session_id = ?
                ORDER BY sm.ordinal
                """,
                (str(session_id),),
            ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def append_message(
        self,
        session_id: UUID,
        expected_revision: int,
        role: MessageRole,
        parts: Sequence[ContentPart | dict[str, Any]],
        *,
        message_id: UUID | None = None,
        now: datetime | None = None,
    ) -> tuple[SessionResource, Message]:
        validated_parts = _CONTENT_PARTS.validate_python(list(parts))
        if not validated_parts:
            raise ValueError("a message requires at least one content part")
        created_at = now or _utcnow()
        identifier = message_id or uuid4()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = connection.execute(
                "SELECT revision, state FROM sessions WHERE id = ?",
                (str(session_id),),
            ).fetchone()
            if session is None:
                raise SessionNotFoundError(str(session_id))
            actual_revision = int(session["revision"])
            if actual_revision != expected_revision:
                raise RevisionConflictError(expected_revision, actual_revision)
            if session["state"] not in {SessionState.IDLE.value, SessionState.ERROR.value}:
                raise StorageError(f"cannot append while session is {session['state']}")
            previous = connection.execute(
                """
                SELECT message_id FROM session_messages
                WHERE session_id = ? ORDER BY ordinal DESC LIMIT 1
                """,
                (str(session_id),),
            ).fetchone()
            parent_id = previous["message_id"] if previous is not None else None
            connection.execute(
                """
                INSERT INTO messages(id, role, parts_json, parent_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(identifier),
                    role.value,
                    _CONTENT_PARTS.dump_json(validated_parts).decode("utf-8"),
                    parent_id,
                    _timestamp(created_at),
                ),
            )
            next_revision = actual_revision + 1
            connection.execute(
                """
                INSERT INTO session_messages(session_id, ordinal, message_id)
                VALUES (?, ?, ?)
                """,
                (str(session_id), next_revision, str(identifier)),
            )
            connection.execute(
                "UPDATE sessions SET revision = ?, updated_at = ? WHERE id = ?",
                (next_revision, _timestamp(created_at), str(session_id)),
            )
        return self.get_session(session_id), Message(
            id=identifier,
            role=role,
            parts=validated_parts,
            parent_id=UUID(parent_id) if parent_id else None,
            created_at=created_at,
        )

    def fork_session(
        self,
        source_session_id: UUID,
        request: ForkSessionRequest,
        *,
        target_session_id: UUID | None = None,
        now: datetime | None = None,
    ) -> SessionResource:
        created_at = now or _utcnow()
        identifier = target_session_id or uuid4()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (str(source_session_id),),
            ).fetchone()
            if source is None:
                raise SessionNotFoundError(str(source_session_id))
            limit = int(source["revision"])
            if request.at_message_id is not None:
                target = connection.execute(
                    """
                    SELECT ordinal FROM session_messages
                    WHERE session_id = ? AND message_id = ?
                    """,
                    (str(source_session_id), str(request.at_message_id)),
                ).fetchone()
                if target is None:
                    raise MessageNotFoundError(str(request.at_message_id))
                limit = int(target["ordinal"])
            connection.execute(
                """
                INSERT INTO sessions(
                    id, model, mode, state, revision, title, runtime_instance_id,
                    created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    str(identifier),
                    source["model"],
                    source["mode"],
                    SessionState.IDLE.value,
                    limit,
                    request.title if request.title is not None else source["title"],
                    _timestamp(created_at),
                    _timestamp(created_at),
                    source["metadata_json"],
                ),
            )
            connection.execute(
                """
                INSERT INTO session_messages(session_id, ordinal, message_id)
                SELECT ?, ordinal, message_id
                FROM session_messages
                WHERE session_id = ? AND ordinal <= ?
                ORDER BY ordinal
                """,
                (str(identifier), str(source_session_id), limit),
            )
        return self.get_session(identifier)

    def begin_response(
        self,
        session_id: UUID,
        request_id: UUID,
        response_id: UUID,
        request_fingerprint: str,
        expected_revision: int,
        input_parts: Sequence[ContentPart | dict[str, Any]],
        *,
        input_message_id: UUID | None = None,
        now: datetime | None = None,
    ) -> BeginResponseResult:
        validated_parts = _CONTENT_PARTS.validate_python(list(input_parts))
        if not validated_parts:
            raise ValueError("a response request requires at least one input part")
        created_at = now or _utcnow()
        message_id = input_message_id or uuid4()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM responses WHERE session_id = ? AND request_id = ?",
                (str(session_id), str(request_id)),
            ).fetchone()
            if existing is not None:
                if existing["request_fingerprint"] != request_fingerprint:
                    raise IdempotencyConflictError(
                        f"request_id {request_id} was reused with a different request"
                    )
                existing_response = self._response_from_row(existing)
                session = connection.execute(
                    "SELECT * FROM sessions WHERE id = ?",
                    (str(session_id),),
                ).fetchone()
                if session is None:
                    raise SessionNotFoundError(str(session_id))
                return BeginResponseResult(
                    session=self._session_from_row(session),
                    response=existing_response,
                    started=False,
                )

            session = connection.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (str(session_id),),
            ).fetchone()
            if session is None:
                raise SessionNotFoundError(str(session_id))
            actual_revision = int(session["revision"])
            if actual_revision != expected_revision:
                raise RevisionConflictError(expected_revision, actual_revision)
            if session["state"] not in {
                SessionState.IDLE.value,
                SessionState.ERROR.value,
                SessionState.INTERRUPTED.value,
            }:
                raise ResponseInProgressError(f"session is {session['state']}")
            previous = connection.execute(
                """
                SELECT message_id FROM session_messages
                WHERE session_id = ? ORDER BY ordinal DESC LIMIT 1
                """,
                (str(session_id),),
            ).fetchone()
            parent_id = previous["message_id"] if previous is not None else None
            connection.execute(
                """
                INSERT INTO messages(id, role, parts_json, parent_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(message_id),
                    MessageRole.USER.value,
                    _CONTENT_PARTS.dump_json(validated_parts).decode("utf-8"),
                    parent_id,
                    _timestamp(created_at),
                ),
            )
            next_revision = actual_revision + 1
            connection.execute(
                """
                INSERT INTO session_messages(session_id, ordinal, message_id)
                VALUES (?, ?, ?)
                """,
                (str(session_id), next_revision, str(message_id)),
            )
            connection.execute(
                """
                UPDATE sessions
                SET revision = ?, state = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_revision,
                    SessionState.PROCESSING.value,
                    _timestamp(created_at),
                    str(session_id),
                ),
            )
            connection.execute(
                """
                INSERT INTO responses(
                    id, session_id, request_id, request_fingerprint, status,
                    input_message_id, output_json, finish_reason, usage_json,
                    error_json, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, '[]', NULL, NULL, NULL, ?, NULL)
                """,
                (
                    str(response_id),
                    str(session_id),
                    str(request_id),
                    request_fingerprint,
                    ResponseStatus.RUNNING.value,
                    str(message_id),
                    _timestamp(created_at),
                ),
            )
        return BeginResponseResult(
            session=self.get_session(session_id),
            response=self.get_response(response_id),
            started=True,
        )

    def get_response(self, response_id: UUID) -> ResponseResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM responses WHERE id = ?",
                (str(response_id),),
            ).fetchone()
        if row is None:
            raise ResponseNotFoundError(str(response_id))
        return self._response_from_row(row)

    def complete_response(
        self,
        response_id: UUID,
        output: Sequence[ContentPart | dict[str, Any]],
        finish_reason: str,
        usage: TokenUsage | None,
        *,
        output_message_id: UUID | None = None,
        now: datetime | None = None,
    ) -> ResponseResource:
        validated_output = _CONTENT_PARTS.validate_python(list(output))
        if not validated_output:
            raise ValueError("a completed response requires at least one output part")
        completed_at = now or _utcnow()
        message_id = output_message_id or uuid4()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            response = connection.execute(
                "SELECT * FROM responses WHERE id = ?",
                (str(response_id),),
            ).fetchone()
            if response is None:
                raise ResponseNotFoundError(str(response_id))
            if response["status"] == ResponseStatus.COMPLETED.value:
                return self._response_from_row(response)
            if response["status"] != ResponseStatus.RUNNING.value:
                raise InvalidResponseStateError(
                    f"cannot complete response in state {response['status']}"
                )
            session_id = response["session_id"]
            session = connection.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise SessionNotFoundError(session_id)
            previous = connection.execute(
                """
                SELECT message_id FROM session_messages
                WHERE session_id = ? ORDER BY ordinal DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            parent_id = previous["message_id"] if previous is not None else None
            connection.execute(
                """
                INSERT INTO messages(id, role, parts_json, parent_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(message_id),
                    MessageRole.ASSISTANT.value,
                    _CONTENT_PARTS.dump_json(validated_output).decode("utf-8"),
                    parent_id,
                    _timestamp(completed_at),
                ),
            )
            next_revision = int(session["revision"]) + 1
            connection.execute(
                """
                INSERT INTO session_messages(session_id, ordinal, message_id)
                VALUES (?, ?, ?)
                """,
                (session_id, next_revision, str(message_id)),
            )
            connection.execute(
                """
                UPDATE sessions
                SET revision = ?, state = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_revision,
                    SessionState.IDLE.value,
                    _timestamp(completed_at),
                    session_id,
                ),
            )
            connection.execute(
                """
                UPDATE responses
                SET status = ?, output_json = ?, finish_reason = ?, usage_json = ?,
                    error_json = NULL, completed_at = ?
                WHERE id = ?
                """,
                (
                    ResponseStatus.COMPLETED.value,
                    _CONTENT_PARTS.dump_json(validated_output).decode("utf-8"),
                    finish_reason,
                    usage.model_dump_json() if usage is not None else None,
                    _timestamp(completed_at),
                    str(response_id),
                ),
            )
        return self.get_response(response_id)

    def terminate_response(
        self,
        response_id: UUID,
        error: ErrorDetail,
        *,
        cancelled: bool = False,
        now: datetime | None = None,
    ) -> ResponseResource:
        completed_at = now or _utcnow()
        response_status = ResponseStatus.CANCELLED if cancelled else ResponseStatus.FAILED
        session_state = SessionState.INTERRUPTED if cancelled else SessionState.ERROR
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            response = connection.execute(
                "SELECT * FROM responses WHERE id = ?",
                (str(response_id),),
            ).fetchone()
            if response is None:
                raise ResponseNotFoundError(str(response_id))
            if response["status"] != ResponseStatus.RUNNING.value:
                return self._response_from_row(response)
            connection.execute(
                """
                UPDATE responses
                SET status = ?, error_json = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    response_status.value,
                    error.model_dump_json(),
                    _timestamp(completed_at),
                    str(response_id),
                ),
            )
            connection.execute(
                """
                UPDATE sessions SET state = ?, updated_at = ? WHERE id = ?
                """,
                (session_state.value, _timestamp(completed_at), response["session_id"]),
            )
        return self.get_response(response_id)

    def delete_session(self, session_id: UUID) -> None:
        with self._connection() as connection:
            session = connection.execute(
                "SELECT state FROM sessions WHERE id = ?",
                (str(session_id),),
            ).fetchone()
            if session is None:
                raise SessionNotFoundError(str(session_id))
            if session["state"] in {
                SessionState.LISTENING.value,
                SessionState.PROCESSING.value,
                SessionState.SPEAKING.value,
            }:
                raise ResponseInProgressError(f"session is {session['state']}")
            cursor = connection.execute(
                "DELETE FROM sessions WHERE id = ?",
                (str(session_id),),
            )
            if cursor.rowcount != 1:
                raise StorageError(f"failed to delete session {session_id}")
            connection.execute(
                "DELETE FROM messages WHERE id NOT IN (SELECT message_id FROM session_messages)"
            )

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> SessionResource:
        runtime_instance_id = row["runtime_instance_id"]
        return SessionResource(
            id=UUID(row["id"]),
            model=row["model"],
            mode=SessionMode(row["mode"]),
            state=SessionState(row["state"]),
            revision=int(row["revision"]),
            title=row["title"],
            runtime_instance_id=UUID(runtime_instance_id) if runtime_instance_id else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Message:
        parent_id = row["parent_id"]
        return Message(
            id=UUID(row["id"]),
            role=MessageRole(row["role"]),
            parts=_CONTENT_PARTS.validate_json(row["parts_json"]),
            parent_id=UUID(parent_id) if parent_id else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _response_from_row(row: sqlite3.Row) -> ResponseResource:
        usage = TokenUsage.model_validate_json(row["usage_json"]) if row["usage_json"] else None
        error = ErrorDetail.model_validate_json(row["error_json"]) if row["error_json"] else None
        completed_at = row["completed_at"]
        return ResponseResource(
            id=UUID(row["id"]),
            request_id=UUID(row["request_id"]),
            session_id=UUID(row["session_id"]),
            status=ResponseStatus(row["status"]),
            output=_CONTENT_PARTS.validate_json(row["output_json"]),
            finish_reason=row["finish_reason"],
            usage=usage,
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
            error=error,
        )
