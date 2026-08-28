"""SQLite persistence for MFQ Server conversations and immutable message branches."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import TypeAdapter

from mfq.server.models import (
    ApiKeyResource,
    ArtifactLineageResource,
    ContentPart,
    CreateApiKeyRequest,
    CreateDatasetRequest,
    CreateGenerationPresetRequest,
    CreateMcpServerRequest,
    CreateRemoteNodeRequest,
    CreateRuntimeProfileRequest,
    CreateSessionRequest,
    DatasetResource,
    DocumentResource,
    ErrorDetail,
    EvaluationKind,
    EvaluationResultResource,
    ForkSessionRequest,
    GenerationPresetResource,
    JobEventLevel,
    JobEventResource,
    JobEventType,
    JobResource,
    JobStatus,
    McpServerResource,
    McpTransport,
    MediaRef,
    MediaResource,
    Message,
    MessageRole,
    ModelLoadRequest,
    RemoteNodeResource,
    ResponsePerformance,
    ResponseRequestSettings,
    ResponseResource,
    ResponseStatus,
    RewindSessionRequest,
    RuntimeLogEntry,
    RuntimeLogLevel,
    RuntimeMetricSnapshot,
    RuntimeProfileResource,
    SamplingParams,
    SessionMode,
    SessionResource,
    SessionState,
    TokenUsage,
    UpdateGenerationPresetRequest,
    UpdateRemoteNodeRequest,
    UpdateRuntimeProfileRequest,
    UpdateSessionRequest,
)

SCHEMA_VERSION = 17
_CONTENT_PARTS = TypeAdapter(list[ContentPart])
_MAX_RUNTIME_METRICS = 20_000


class StorageError(RuntimeError):
    """Base class for persistent conversation errors."""


class DatasetNotFoundError(StorageError):
    pass


class EvaluationNotFoundError(StorageError):
    pass


class RemoteNodeNotFoundError(StorageError):
    pass


class SessionNotFoundError(StorageError):
    pass


class MessageNotFoundError(StorageError):
    pass


class MediaNotFoundError(StorageError):
    pass


class MediaIntegrityError(StorageError):
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


class JobNotFoundError(StorageError):
    pass


class InvalidJobStateError(StorageError):
    pass


class GenerationPresetNotFoundError(StorageError):
    pass


class RuntimeProfileNotFoundError(StorageError):
    pass


class ApiKeyNotFoundError(StorageError):
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

    def __init__(self, path: str | Path, *, media_root: str | Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.media_root = (
            Path(media_root)
            if media_root is not None
            else self.path.with_name(f"{self.path.name}.media")
        )
        self.media_root.mkdir(parents=True, exist_ok=True)
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
                    output_message_id TEXT REFERENCES messages(id),
                    output_json TEXT NOT NULL,
                    finish_reason TEXT,
                    usage_json TEXT,
                    performance_json TEXT,
                    settings_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE (session_id, request_id)
                );

                CREATE INDEX IF NOT EXISTS responses_session_status
                    ON responses(session_id, status);

                CREATE TABLE IF NOT EXISTS media (
                    id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL UNIQUE,
                    mime_type TEXT NOT NULL,
                    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
                    storage_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    media_id TEXT PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_count INTEGER,
                    extractor TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mcp_servers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    transport TEXT NOT NULL,
                    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS generation_presets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    model TEXT,
                    mode TEXT,
                    settings_json TEXT NOT NULL,
                    context_size INTEGER NOT NULL CHECK (context_size >= 512),
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    load_json TEXT NOT NULL,
                    sampling_json TEXT,
                    artifact_id TEXT NOT NULL,
                    artifact_modified_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifact_lineage (
                    id TEXT PRIMARY KEY,
                    artifact_uri TEXT NOT NULL UNIQUE,
                    artifact_name TEXT NOT NULL,
                    producer_job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    producer_kind TEXT NOT NULL,
                    source_uris_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifact_validations (
                    artifact_uri TEXT NOT NULL REFERENCES artifact_lineage(artifact_uri)
                        ON DELETE CASCADE,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (artifact_uri, job_id)
                );

                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    key_hash TEXT NOT NULL UNIQUE,
                    prefix TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    role TEXT,
                    expires_at TEXT,
                    revoked_at TEXT,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    artifact_uri TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
                    source_uri TEXT,
                    revision TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
                    dataset_manifest_json TEXT NOT NULL,
                    hardware_identity_json TEXT NOT NULL,
                    runtime_identity_json TEXT NOT NULL,
                    comparison_key TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS evaluation_results_comparison
                    ON evaluation_results(comparison_key, created_at DESC);

                CREATE TABLE IF NOT EXISTS remote_nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    url TEXT NOT NULL UNIQUE,
                    api_key_env TEXT,
                    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    progress REAL NOT NULL CHECK (progress >= 0 AND progress <= 1),
                    cancel_requested INTEGER NOT NULL CHECK (cancel_requested IN (0, 1)),
                    result_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    archived_at TEXT
                );

                CREATE INDEX IF NOT EXISTS jobs_status_created
                    ON jobs(status, created_at);

                CREATE TABLE IF NOT EXISTS job_events (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL CHECK (sequence >= 1),
                    type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT,
                    progress REAL CHECK (progress IS NULL OR (progress >= 0 AND progress <= 1)),
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS runtime_metrics (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT,
                    model TEXT,
                    values_json TEXT NOT NULL,
                    captured_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS runtime_metrics_captured
                    ON runtime_metrics(captured_at DESC);

                CREATE TABLE IF NOT EXISTS runtime_logs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS runtime_logs_created
                    ON runtime_logs(created_at DESC);
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
            elif int(existing["value"]) in {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            }:
                columns = {
                    row["name"] for row in connection.execute("PRAGMA table_info(responses)")
                }
                if "performance_json" not in columns:
                    connection.execute("ALTER TABLE responses ADD COLUMN performance_json TEXT")
                if "output_message_id" not in columns:
                    connection.execute("ALTER TABLE responses ADD COLUMN output_message_id TEXT")
                if "settings_json" not in columns:
                    connection.execute("ALTER TABLE responses ADD COLUMN settings_json TEXT")
                key_columns = {
                    row["name"] for row in connection.execute("PRAGMA table_info(api_keys)")
                }
                if "role" not in key_columns:
                    connection.execute("ALTER TABLE api_keys ADD COLUMN role TEXT")
                preset_columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(generation_presets)")
                }
                if "metadata_json" not in preset_columns:
                    connection.execute(
                        "ALTER TABLE generation_presets ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                    )
                job_columns = {
                    row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
                }
                if "archived_at" not in job_columns:
                    connection.execute("ALTER TABLE jobs ADD COLUMN archived_at TEXT")
                connection.execute(
                    "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                    (str(SCHEMA_VERSION),),
                )
            elif int(existing["value"]) != SCHEMA_VERSION:
                raise StorageError(
                    f"unsupported database schema {existing['value']}; expected {SCHEMA_VERSION}"
                )

    def put_media(
        self,
        data: bytes,
        mime_type: str,
        expected_sha256: str,
        *,
        media_id: UUID | None = None,
        now: datetime | None = None,
    ) -> MediaResource:
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256:
            raise MediaIntegrityError(
                f"media digest mismatch: expected {expected_sha256}, computed {digest}"
            )
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT * FROM media WHERE sha256 = ?",
                (digest,),
            ).fetchone()
        if existing is not None:
            return self._media_from_row(existing)

        storage_key = f"{digest[:2]}/{digest[2:4]}/{digest}"
        destination = self.media_root / storage_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=destination.parent,
                    prefix=f".{digest}.",
                    suffix=".upload",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                assert temporary is not None
                os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

        identifier = media_id or uuid4()
        created_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO media(
                    id, sha256, mime_type, byte_size, storage_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(identifier),
                    digest,
                    mime_type,
                    len(data),
                    storage_key,
                    _timestamp(created_at),
                ),
            )
            row = connection.execute(
                "SELECT * FROM media WHERE sha256 = ?",
                (digest,),
            ).fetchone()
        if row is None:
            raise StorageError("media record was not persisted")
        return self._media_from_row(row)

    def get_media(self, media_id: UUID) -> MediaResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM media WHERE id = ?",
                (str(media_id),),
            ).fetchone()
        if row is None:
            raise MediaNotFoundError(str(media_id))
        return self._media_from_row(row)

    def get_media_path(self, media_id: UUID) -> tuple[MediaResource, Path]:
        resource = self.get_media(media_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT storage_key FROM media WHERE id = ?",
                (str(media_id),),
            ).fetchone()
        if row is None:
            raise MediaNotFoundError(str(media_id))
        path = self.media_root / str(row["storage_key"])
        if not path.is_file():
            raise MediaNotFoundError(f"media blob is missing: {media_id}")
        return resource, path

    def put_document(
        self,
        media_id: UUID,
        name: str,
        text: str,
        extractor: str,
        *,
        page_count: int | None = None,
        now: datetime | None = None,
    ) -> DocumentResource:
        media = self.get_media(media_id)
        created_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents(media_id, name, text, page_count, extractor, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(media_id) DO UPDATE SET
                    name = excluded.name,
                    text = excluded.text,
                    page_count = excluded.page_count,
                    extractor = excluded.extractor
                """,
                (
                    str(media_id),
                    name,
                    text,
                    page_count,
                    extractor,
                    _timestamp(created_at),
                ),
            )
            row = connection.execute(
                "SELECT * FROM documents WHERE media_id = ?", (str(media_id),)
            ).fetchone()
        if row is None:
            raise StorageError("document record was not persisted")
        return DocumentResource(
            media=media.media,
            name=row["name"],
            text=row["text"],
            page_count=int(row["page_count"]) if row["page_count"] is not None else None,
            extractor=row["extractor"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_document(self, media_id: UUID) -> DocumentResource:
        media = self.get_media(media_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE media_id = ?", (str(media_id),)
            ).fetchone()
        if row is None:
            raise MediaNotFoundError(f"document extraction is missing: {media_id}")
        return DocumentResource(
            media=media.media,
            name=row["name"],
            text=row["text"],
            page_count=int(row["page_count"]) if row["page_count"] is not None else None,
            extractor=row["extractor"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def create_mcp_server(
        self,
        request: CreateMcpServerRequest,
        *,
        server_id: UUID | None = None,
        now: datetime | None = None,
    ) -> McpServerResource:
        identifier = server_id or uuid4()
        created_at = now or _utcnow()
        config = request.model_dump(mode="json", exclude={"name", "transport", "enabled"})
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO mcp_servers(
                        id, name, transport, enabled, config_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(identifier),
                        request.name,
                        request.transport.value,
                        int(request.enabled),
                        json.dumps(config, separators=(",", ":"), sort_keys=True),
                        _timestamp(created_at),
                        _timestamp(created_at),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError(f"MCP server already exists: {request.name}") from error
        return self.get_mcp_server(identifier)

    def get_mcp_server(self, server_id: UUID) -> McpServerResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM mcp_servers WHERE id = ?", (str(server_id),)
            ).fetchone()
        if row is None:
            raise MediaNotFoundError(f"MCP server was not found: {server_id}")
        return self._mcp_server_from_row(row)

    def list_mcp_servers(self, *, enabled_only: bool = False) -> list[McpServerResource]:
        query = "SELECT * FROM mcp_servers"
        values: tuple[object, ...] = ()
        if enabled_only:
            query += " WHERE enabled = ?"
            values = (1,)
        query += " ORDER BY name"
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._mcp_server_from_row(row) for row in rows]

    def set_mcp_server_enabled(
        self,
        server_id: UUID,
        enabled: bool,
        *,
        now: datetime | None = None,
    ) -> McpServerResource:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE mcp_servers SET enabled = ?, updated_at = ? WHERE id = ?",
                (int(enabled), _timestamp(now or _utcnow()), str(server_id)),
            )
        if cursor.rowcount != 1:
            raise MediaNotFoundError(f"MCP server was not found: {server_id}")
        return self.get_mcp_server(server_id)

    def delete_mcp_server(self, server_id: UUID) -> None:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM mcp_servers WHERE id = ?", (str(server_id),))
        if cursor.rowcount != 1:
            raise MediaNotFoundError(f"MCP server was not found: {server_id}")

    def create_generation_preset(
        self,
        request: CreateGenerationPresetRequest,
        *,
        preset_id: UUID | None = None,
        now: datetime | None = None,
    ) -> GenerationPresetResource:
        identifier = preset_id or uuid4()
        created_at = now or _utcnow()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO generation_presets(
                        id, name, model, mode, settings_json, context_size, metadata_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(identifier),
                        request.name,
                        request.model,
                        request.mode.value if request.mode is not None else None,
                        request.settings.model_dump_json(by_alias=True),
                        request.context_size,
                        json.dumps(request.metadata, separators=(",", ":"), sort_keys=True),
                        _timestamp(created_at),
                        _timestamp(created_at),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError(f"generation preset already exists: {request.name}") from error
        return self.get_generation_preset(identifier)

    def get_generation_preset(self, preset_id: UUID) -> GenerationPresetResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM generation_presets WHERE id = ?", (str(preset_id),)
            ).fetchone()
        if row is None:
            raise GenerationPresetNotFoundError(str(preset_id))
        return self._generation_preset_from_row(row)

    def list_generation_presets(self) -> list[GenerationPresetResource]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM generation_presets ORDER BY name COLLATE NOCASE, id"
            ).fetchall()
        return [self._generation_preset_from_row(row) for row in rows]

    def update_generation_preset(
        self,
        preset_id: UUID,
        request: UpdateGenerationPresetRequest,
        *,
        now: datetime | None = None,
    ) -> GenerationPresetResource:
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE generation_presets
                    SET name = ?, model = ?, mode = ?, settings_json = ?,
                        context_size = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        request.name,
                        request.model,
                        request.mode.value if request.mode is not None else None,
                        request.settings.model_dump_json(by_alias=True),
                        request.context_size,
                        json.dumps(request.metadata, separators=(",", ":"), sort_keys=True),
                        _timestamp(now or _utcnow()),
                        str(preset_id),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError(f"generation preset already exists: {request.name}") from error
        if cursor.rowcount != 1:
            raise GenerationPresetNotFoundError(str(preset_id))
        return self.get_generation_preset(preset_id)

    def delete_generation_preset(self, preset_id: UUID) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM generation_presets WHERE id = ?", (str(preset_id),)
            )
        if cursor.rowcount != 1:
            raise GenerationPresetNotFoundError(str(preset_id))

    def create_runtime_profile(
        self,
        request: CreateRuntimeProfileRequest,
        artifact_id: str,
        artifact_modified_at: datetime,
        *,
        profile_id: UUID | None = None,
        now: datetime | None = None,
    ) -> RuntimeProfileResource:
        identifier = profile_id or uuid4()
        created_at = now or _utcnow()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO runtime_profiles(
                        id, name, load_json, sampling_json, artifact_id,
                        artifact_modified_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(identifier),
                        request.name,
                        request.load.model_dump_json(),
                        request.load.sampling_defaults.model_dump_json()
                        if request.load.sampling_defaults is not None
                        else None,
                        artifact_id,
                        _timestamp(artifact_modified_at),
                        _timestamp(created_at),
                        _timestamp(created_at),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError(f"runtime profile already exists: {request.name}") from error
        return self.get_runtime_profile(identifier)

    def get_runtime_profile(self, profile_id: UUID) -> RuntimeProfileResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runtime_profiles WHERE id = ?", (str(profile_id),)
            ).fetchone()
        if row is None:
            raise RuntimeProfileNotFoundError(str(profile_id))
        return self._runtime_profile_from_row(row)

    def list_runtime_profiles(self) -> list[RuntimeProfileResource]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM runtime_profiles ORDER BY name COLLATE NOCASE, id"
            ).fetchall()
        return [self._runtime_profile_from_row(row) for row in rows]

    def update_runtime_profile(
        self,
        profile_id: UUID,
        request: UpdateRuntimeProfileRequest,
        artifact_id: str,
        artifact_modified_at: datetime,
        *,
        now: datetime | None = None,
    ) -> RuntimeProfileResource:
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE runtime_profiles
                    SET name = ?, load_json = ?, sampling_json = ?,
                        artifact_id = ?, artifact_modified_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        request.name,
                        request.load.model_dump_json(),
                        request.load.sampling_defaults.model_dump_json()
                        if request.load.sampling_defaults is not None
                        else None,
                        artifact_id,
                        _timestamp(artifact_modified_at),
                        _timestamp(now or _utcnow()),
                        str(profile_id),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError(f"runtime profile already exists: {request.name}") from error
        if cursor.rowcount != 1:
            raise RuntimeProfileNotFoundError(str(profile_id))
        return self.get_runtime_profile(profile_id)

    def delete_runtime_profile(self, profile_id: UUID) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM runtime_profiles WHERE id = ?", (str(profile_id),)
            )
        if cursor.rowcount != 1:
            raise RuntimeProfileNotFoundError(str(profile_id))

    def record_artifact_lineage(
        self,
        *,
        artifact_uri: str,
        artifact_name: str,
        producer_job_id: UUID,
        source_uris: Sequence[str] = (),
        parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> ArtifactLineageResource:
        job = self.get_job(producer_job_id)
        identifier = uuid4()
        created_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO artifact_lineage(
                    id, artifact_uri, artifact_name, producer_job_id,
                    producer_kind, source_uris_json, parameters_json,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_uri) DO UPDATE SET
                    artifact_name = excluded.artifact_name,
                    producer_job_id = excluded.producer_job_id,
                    producer_kind = excluded.producer_kind,
                    source_uris_json = excluded.source_uris_json,
                    parameters_json = excluded.parameters_json,
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    str(identifier),
                    artifact_uri,
                    artifact_name,
                    str(producer_job_id),
                    job.kind,
                    json.dumps(list(source_uris), separators=(",", ":")),
                    json.dumps(parameters or job.payload, separators=(",", ":")),
                    json.dumps(metadata or {}, separators=(",", ":")),
                    _timestamp(created_at),
                ),
            )
            row = connection.execute(
                "SELECT * FROM artifact_lineage WHERE artifact_uri = ?",
                (artifact_uri,),
            ).fetchone()
        if row is None:
            raise StorageError("artifact lineage was not persisted")
        return self._artifact_lineage_from_row(row)

    def record_artifact_validation(
        self,
        artifact_uri: str,
        job_id: UUID,
        *,
        now: datetime | None = None,
    ) -> None:
        self.get_job(job_id)
        with self._connection() as connection:
            lineage = connection.execute(
                "SELECT 1 FROM artifact_lineage WHERE artifact_uri = ?",
                (artifact_uri,),
            ).fetchone()
            if lineage is None:
                raise StorageError(f"artifact lineage was not found: {artifact_uri}")
            connection.execute(
                """
                INSERT OR IGNORE INTO artifact_validations(
                    artifact_uri, job_id, created_at
                ) VALUES (?, ?, ?)
                """,
                (artifact_uri, str(job_id), _timestamp(now or _utcnow())),
            )

    def list_artifact_lineage(
        self, *, artifact_uri: str | None = None, limit: int = 200
    ) -> list[ArtifactLineageResource]:
        query = "SELECT * FROM artifact_lineage"
        values: list[Any] = []
        if artifact_uri is not None:
            query += " WHERE artifact_uri = ?"
            values.append(artifact_uri)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._artifact_lineage_from_row(row) for row in rows]

    def create_dataset(
        self,
        request: CreateDatasetRequest,
        *,
        sha256: str,
        byte_size: int,
        dataset_id: UUID | None = None,
        now: datetime | None = None,
    ) -> DatasetResource:
        identifier = dataset_id or uuid4()
        created_at = now or _utcnow()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO datasets(
                        id, name, kind, artifact_uri, sha256, byte_size,
                        source_uri, revision, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(identifier),
                        request.name,
                        request.kind,
                        request.artifact_uri,
                        sha256,
                        byte_size,
                        request.source_uri,
                        request.revision,
                        json.dumps(request.metadata, separators=(",", ":")),
                        _timestamp(created_at),
                        _timestamp(created_at),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError(f"dataset already exists: {request.name}") from error
        return self.get_dataset(identifier)

    def get_dataset(self, dataset_id: UUID) -> DatasetResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE id = ?", (str(dataset_id),)
            ).fetchone()
        if row is None:
            raise DatasetNotFoundError(str(dataset_id))
        return self._dataset_from_row(row)

    def list_datasets(self) -> list[DatasetResource]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets ORDER BY name COLLATE NOCASE, id"
            ).fetchall()
        return [self._dataset_from_row(row) for row in rows]

    def delete_dataset(self, dataset_id: UUID) -> None:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM datasets WHERE id = ?", (str(dataset_id),))
        if cursor.rowcount != 1:
            raise DatasetNotFoundError(str(dataset_id))

    def record_evaluation(
        self,
        *,
        job_id: UUID,
        kind: EvaluationKind,
        model_id: str,
        metrics: dict[str, Any],
        parameters: dict[str, Any],
        dataset_id: UUID | None,
        dataset_manifest: dict[str, Any],
        hardware_identity: dict[str, Any],
        runtime_identity: dict[str, Any],
        comparison_key: str,
        evaluation_id: UUID | None = None,
        now: datetime | None = None,
    ) -> EvaluationResultResource:
        self.get_job(job_id)
        if dataset_id is not None:
            self.get_dataset(dataset_id)
        identifier = evaluation_id or uuid4()
        created_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_results(
                    id, job_id, kind, model_id, metrics_json, parameters_json,
                    dataset_id, dataset_manifest_json, hardware_identity_json,
                    runtime_identity_json, comparison_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    kind = excluded.kind,
                    model_id = excluded.model_id,
                    metrics_json = excluded.metrics_json,
                    parameters_json = excluded.parameters_json,
                    dataset_id = excluded.dataset_id,
                    dataset_manifest_json = excluded.dataset_manifest_json,
                    hardware_identity_json = excluded.hardware_identity_json,
                    runtime_identity_json = excluded.runtime_identity_json,
                    comparison_key = excluded.comparison_key,
                    created_at = excluded.created_at
                """,
                (
                    str(identifier),
                    str(job_id),
                    kind,
                    model_id,
                    json.dumps(metrics, separators=(",", ":")),
                    json.dumps(parameters, separators=(",", ":")),
                    str(dataset_id) if dataset_id is not None else None,
                    json.dumps(dataset_manifest, separators=(",", ":")),
                    json.dumps(hardware_identity, separators=(",", ":")),
                    json.dumps(runtime_identity, separators=(",", ":")),
                    comparison_key,
                    _timestamp(created_at),
                ),
            )
            row = connection.execute(
                "SELECT * FROM evaluation_results WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        if row is None:
            raise StorageError("evaluation result was not persisted")
        return self._evaluation_from_row(row)

    def get_evaluation(self, evaluation_id: UUID) -> EvaluationResultResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_results WHERE id = ?", (str(evaluation_id),)
            ).fetchone()
        if row is None:
            raise EvaluationNotFoundError(str(evaluation_id))
        return self._evaluation_from_row(row)

    def list_evaluations(
        self,
        *,
        kind: EvaluationKind | None = None,
        model_id: str | None = None,
        limit: int = 200,
    ) -> list[EvaluationResultResource]:
        clauses: list[str] = []
        values: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        if model_id is not None:
            clauses.append("model_id = ?")
            values.append(model_id)
        query = "SELECT * FROM evaluation_results"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id LIMIT ?"
        values.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._evaluation_from_row(row) for row in rows]

    def create_remote_node(
        self,
        request: CreateRemoteNodeRequest,
        *,
        node_id: UUID | None = None,
        now: datetime | None = None,
    ) -> RemoteNodeResource:
        identifier = node_id or uuid4()
        created_at = now or _utcnow()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO remote_nodes(
                        id, name, url, api_key_env, enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(identifier),
                        request.name,
                        request.url.rstrip("/"),
                        request.api_key_env,
                        int(request.enabled),
                        _timestamp(created_at),
                        _timestamp(created_at),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError(f"remote node already exists: {request.name}") from error
        return self.get_remote_node(identifier)

    def get_remote_node(self, node_id: UUID) -> RemoteNodeResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM remote_nodes WHERE id = ?", (str(node_id),)
            ).fetchone()
        if row is None:
            raise RemoteNodeNotFoundError(str(node_id))
        return self._remote_node_from_row(row)

    def list_remote_nodes(self) -> list[RemoteNodeResource]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM remote_nodes ORDER BY name COLLATE NOCASE, id"
            ).fetchall()
        return [self._remote_node_from_row(row) for row in rows]

    def update_remote_node(
        self,
        node_id: UUID,
        request: UpdateRemoteNodeRequest,
        *,
        now: datetime | None = None,
    ) -> RemoteNodeResource:
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE remote_nodes
                    SET name = ?, url = ?, api_key_env = ?, enabled = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        request.name,
                        request.url.rstrip("/"),
                        request.api_key_env,
                        int(request.enabled),
                        _timestamp(now or _utcnow()),
                        str(node_id),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError(f"remote node already exists: {request.name}") from error
        if cursor.rowcount != 1:
            raise RemoteNodeNotFoundError(str(node_id))
        return self.get_remote_node(node_id)

    def delete_remote_node(self, node_id: UUID) -> None:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM remote_nodes WHERE id = ?", (str(node_id),))
        if cursor.rowcount != 1:
            raise RemoteNodeNotFoundError(str(node_id))

    def create_api_key(
        self,
        request: CreateApiKeyRequest,
        key_hash: str,
        prefix: str,
        *,
        key_id: UUID | None = None,
        now: datetime | None = None,
    ) -> ApiKeyResource:
        identifier = key_id or uuid4()
        created_at = now or _utcnow()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO api_keys(
                        id, name, key_hash, prefix, scopes_json, role, expires_at,
                        revoked_at, created_at, last_used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL)
                    """,
                    (
                        str(identifier),
                        request.name,
                        key_hash,
                        prefix,
                        json.dumps(request.scopes, separators=(",", ":")),
                        request.role,
                        _timestamp(request.expires_at) if request.expires_at else None,
                        _timestamp(created_at),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError(f"API key already exists: {request.name}") from error
        return self.get_api_key(identifier)

    def get_api_key(self, key_id: UUID) -> ApiKeyResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM api_keys WHERE id = ?", (str(key_id),)
            ).fetchone()
        if row is None:
            raise ApiKeyNotFoundError(str(key_id))
        return self._api_key_from_row(row)

    def list_api_keys(self) -> list[ApiKeyResource]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM api_keys ORDER BY created_at DESC, id"
            ).fetchall()
        return [self._api_key_from_row(row) for row in rows]

    def authenticate_api_key(
        self, key_hash: str, *, now: datetime | None = None
    ) -> ApiKeyResource | None:
        used_at = now or _utcnow()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
            ).fetchone()
            if row is None or row["revoked_at"] is not None:
                return None
            expires_at = datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
            if expires_at is not None and expires_at <= used_at:
                return None
            connection.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (_timestamp(used_at), row["id"]),
            )
            row = connection.execute("SELECT * FROM api_keys WHERE id = ?", (row["id"],)).fetchone()
        return self._api_key_from_row(row) if row is not None else None

    def revoke_api_key(self, key_id: UUID, *, now: datetime | None = None) -> ApiKeyResource:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ?",
                (_timestamp(now or _utcnow()), str(key_id)),
            )
        if cursor.rowcount != 1:
            raise ApiKeyNotFoundError(str(key_id))
        return self.get_api_key(key_id)

    def rotate_api_key(
        self,
        key_id: UUID,
        key_hash: str,
        prefix: str,
    ) -> ApiKeyResource:
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE api_keys
                    SET key_hash = ?, prefix = ?, revoked_at = NULL,
                        last_used_at = NULL
                    WHERE id = ?
                    """,
                    (key_hash, prefix, str(key_id)),
                )
        except sqlite3.IntegrityError as error:
            raise StorageError("generated API key collided with an existing key") from error
        if cursor.rowcount != 1:
            raise ApiKeyNotFoundError(str(key_id))
        return self.get_api_key(key_id)

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

    def update_session(
        self,
        session_id: UUID,
        request: UpdateSessionRequest,
        *,
        now: datetime | None = None,
    ) -> SessionResource:
        updated_at = now or _utcnow()
        assignments: list[str] = []
        values: list[object] = []
        if "title" in request.model_fields_set:
            assignments.append("title = ?")
            values.append(request.title)
        if "mode" in request.model_fields_set:
            assignments.append("mode = ?")
            values.append(request.mode.value)
        if "metadata" in request.model_fields_set:
            assignments.append("metadata_json = ?")
            values.append(json.dumps(request.metadata, separators=(",", ":"), sort_keys=True))
        if not assignments:
            return self.get_session(session_id)
        assignments.append("updated_at = ?")
        values.extend((_timestamp(updated_at), str(session_id)))
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            if cursor.rowcount != 1:
                raise SessionNotFoundError(str(session_id))
        return self.get_session(session_id)

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
                if not request.include_message:
                    limit -= 1
            elif not request.include_message:
                raise StorageError("include_message=false requires at_message_id")
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

    def rewind_session(
        self,
        session_id: UUID,
        request: RewindSessionRequest,
        *,
        now: datetime | None = None,
    ) -> SessionResource:
        updated_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = connection.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (str(session_id),),
            ).fetchone()
            if session is None:
                raise SessionNotFoundError(str(session_id))
            actual_revision = int(session["revision"])
            if actual_revision != request.expected_revision:
                raise RevisionConflictError(request.expected_revision, actual_revision)
            if session["state"] == SessionState.PROCESSING.value:
                raise ResponseInProgressError(f"session is {session['state']}")
            target = connection.execute(
                """
                SELECT ordinal FROM session_messages
                WHERE session_id = ? AND message_id = ?
                """,
                (str(session_id), str(request.at_message_id)),
            ).fetchone()
            if target is None:
                raise MessageNotFoundError(str(request.at_message_id))
            limit = int(target["ordinal"])
            if not request.include_message:
                limit -= 1
            connection.execute(
                """
                DELETE FROM responses
                WHERE session_id = ? AND (
                    input_message_id IN (
                        SELECT message_id FROM session_messages
                        WHERE session_id = ? AND ordinal > ?
                    ) OR output_message_id IN (
                        SELECT message_id FROM session_messages
                        WHERE session_id = ? AND ordinal > ?
                    )
                )
                """,
                (str(session_id), str(session_id), limit, str(session_id), limit),
            )
            connection.execute(
                "DELETE FROM session_messages WHERE session_id = ? AND ordinal > ?",
                (str(session_id), limit),
            )
            connection.execute(
                """
                UPDATE sessions
                SET revision = ?, state = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    limit,
                    SessionState.IDLE.value,
                    _timestamp(updated_at),
                    str(session_id),
                ),
            )
        return self.get_session(session_id)

    def begin_response(
        self,
        session_id: UUID,
        request_id: UUID,
        response_id: UUID,
        request_fingerprint: str,
        expected_revision: int,
        input_parts: Sequence[ContentPart | dict[str, Any]],
        settings: ResponseRequestSettings | None = None,
        input_role: MessageRole = MessageRole.USER,
        *,
        input_message_id: UUID | None = None,
        now: datetime | None = None,
    ) -> BeginResponseResult:
        validated_parts = _CONTENT_PARTS.validate_python(list(input_parts))
        if not validated_parts:
            raise ValueError("a response request requires at least one input part")
        if input_role not in {MessageRole.USER, MessageRole.TOOL}:
            raise ValueError("a response input must use the user or tool role")
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
                    input_role.value,
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
                    input_message_id, output_message_id, output_json, finish_reason,
                    usage_json, performance_json, settings_json, error_json,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, '[]', NULL, NULL, NULL, ?, NULL, ?, NULL)
                """,
                (
                    str(response_id),
                    str(session_id),
                    str(request_id),
                    request_fingerprint,
                    ResponseStatus.RUNNING.value,
                    str(message_id),
                    settings.model_dump_json() if settings is not None else None,
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
        performance: ResponsePerformance | None = None,
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
                SET status = ?, output_message_id = ?, output_json = ?, finish_reason = ?, usage_json = ?,
                    performance_json = ?, error_json = NULL, completed_at = ?
                WHERE id = ?
                """,
                (
                    ResponseStatus.COMPLETED.value,
                    str(message_id),
                    _CONTENT_PARTS.dump_json(validated_output).decode("utf-8"),
                    finish_reason,
                    usage.model_dump_json() if usage is not None else None,
                    performance.model_dump_json() if performance is not None else None,
                    _timestamp(completed_at),
                    str(response_id),
                ),
            )
        return self.get_response(response_id)

    def list_responses(self, session_id: UUID, *, limit: int = 200) -> list[ResponseResource]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM responses WHERE session_id = ?
                ORDER BY created_at ASC LIMIT ?
                """,
                (str(session_id), limit),
            ).fetchall()
        return [self._response_from_row(row) for row in rows]

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

    @staticmethod
    def _append_job_event(
        connection: sqlite3.Connection,
        job_id: UUID,
        event_type: JobEventType,
        level: JobEventLevel,
        *,
        message: str | None = None,
        progress: float | None = None,
        data: dict[str, Any] | None = None,
        now: datetime,
    ) -> None:
        next_sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM job_events WHERE job_id = ?",
            (str(job_id),),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO job_events(
                job_id, sequence, type, level, message, progress, data_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(job_id),
                int(next_sequence),
                event_type.value,
                level.value,
                message,
                progress,
                json.dumps(data or {}, separators=(",", ":"), sort_keys=True),
                _timestamp(now),
            ),
        )

    def create_job(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        job_id: UUID | None = None,
        now: datetime | None = None,
    ) -> JobResource:
        identifier = job_id or uuid4()
        created_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO jobs(
                    id, kind, status, payload_json, progress, cancel_requested,
                    result_json, error_json, created_at, updated_at,
                    started_at, completed_at, archived_at
                ) VALUES (?, ?, ?, ?, 0, 0, NULL, NULL, ?, ?, NULL, NULL, NULL)
                """,
                (
                    str(identifier),
                    kind,
                    JobStatus.QUEUED.value,
                    json.dumps(payload, separators=(",", ":"), sort_keys=True),
                    _timestamp(created_at),
                    _timestamp(created_at),
                ),
            )
            self._append_job_event(
                connection,
                identifier,
                JobEventType.STATE,
                JobEventLevel.INFO,
                message=JobStatus.QUEUED.value,
                data={"status": JobStatus.QUEUED.value},
                now=created_at,
            )
        return self.get_job(identifier)

    def get_job(self, job_id: UUID) -> JobResource:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ? AND archived_at IS NULL",
                (str(job_id),),
            ).fetchone()
        if row is None:
            raise JobNotFoundError(str(job_id))
        return self._job_from_row(row)

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobResource]:
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        clauses: list[str] = ["archived_at IS NULL"]
        values: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            values.append(status.value)
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.extend((limit, offset))
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM jobs {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                values,
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def archive_job(self, job_id: UUID) -> None:
        terminal_statuses = (
            JobStatus.SUCCEEDED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.INTERRUPTED.value,
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM jobs WHERE id = ? AND archived_at IS NULL",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(str(job_id))
            if row["status"] not in terminal_statuses:
                raise InvalidJobStateError(
                    f"cannot delete job in state {row['status']}"
                )
            connection.execute(
                "UPDATE jobs SET archived_at = ? WHERE id = ?",
                (_timestamp(_utcnow()), str(job_id)),
            )

    def archive_completed_jobs(self) -> int:
        terminal_statuses = (
            JobStatus.SUCCEEDED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.INTERRUPTED.value,
        )
        placeholders = ", ".join("?" for _ in terminal_statuses)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM jobs "
                    f"WHERE archived_at IS NULL AND status IN ({placeholders})",
                    terminal_statuses,
                ).fetchone()[0]
            )
            connection.execute(
                f"UPDATE jobs SET archived_at = ? "
                f"WHERE archived_at IS NULL AND status IN ({placeholders})",
                (_timestamp(_utcnow()), *terminal_statuses),
            )
        return count

    def list_queued_job_ids(self) -> list[UUID]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at, id",
                (JobStatus.QUEUED.value,),
            ).fetchall()
        return [UUID(row["id"]) for row in rows]

    def claim_job(self, job_id: UUID, *, now: datetime | None = None) -> JobResource:
        started_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(str(job_id))
            if row["status"] != JobStatus.QUEUED.value:
                return self._job_from_row(row)
            connection.execute(
                """
                UPDATE jobs SET status = ?, started_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    JobStatus.RUNNING.value,
                    _timestamp(started_at),
                    _timestamp(started_at),
                    str(job_id),
                ),
            )
            self._append_job_event(
                connection,
                job_id,
                JobEventType.STATE,
                JobEventLevel.INFO,
                message=JobStatus.RUNNING.value,
                data={"status": JobStatus.RUNNING.value},
                now=started_at,
            )
        return self.get_job(job_id)

    def update_job_progress(
        self,
        job_id: UUID,
        progress: float,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> JobResource:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("job progress must be between 0 and 1")
        updated_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, progress FROM jobs WHERE id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(str(job_id))
            if row["status"] not in {
                JobStatus.RUNNING.value,
                JobStatus.CANCELLING.value,
            }:
                raise InvalidJobStateError(f"cannot update progress while job is {row['status']}")
            next_progress = max(float(row["progress"]), progress)
            connection.execute(
                "UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?",
                (next_progress, _timestamp(updated_at), str(job_id)),
            )
            self._append_job_event(
                connection,
                job_id,
                JobEventType.PROGRESS,
                JobEventLevel.INFO,
                message=message,
                progress=next_progress,
                data=data,
                now=updated_at,
            )
        return self.get_job(job_id)

    def append_job_event(
        self,
        job_id: UUID,
        event_type: JobEventType,
        level: JobEventLevel = JobEventLevel.INFO,
        *,
        message: str | None = None,
        progress: float | None = None,
        data: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> JobEventResource:
        created_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute("SELECT 1 FROM jobs WHERE id = ?", (str(job_id),)).fetchone()
                is None
            ):
                raise JobNotFoundError(str(job_id))
            self._append_job_event(
                connection,
                job_id,
                event_type,
                level,
                message=message,
                progress=progress,
                data=data,
                now=created_at,
            )
            row = connection.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY sequence DESC LIMIT 1",
                (str(job_id),),
            ).fetchone()
        if row is None:
            raise StorageError("job event was not persisted")
        return self._job_event_from_row(row)

    def list_job_events(
        self,
        job_id: UUID,
        *,
        after: int = 0,
        limit: int = 200,
    ) -> list[JobEventResource]:
        if after < 0:
            raise ValueError("after must be non-negative")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connection() as connection:
            if (
                connection.execute("SELECT 1 FROM jobs WHERE id = ?", (str(job_id),)).fetchone()
                is None
            ):
                raise JobNotFoundError(str(job_id))
            rows = connection.execute(
                """
                SELECT * FROM job_events
                WHERE job_id = ? AND sequence > ?
                ORDER BY sequence LIMIT ?
                """,
                (str(job_id), after, limit),
            ).fetchall()
        return [self._job_event_from_row(row) for row in rows]

    def request_job_cancel(
        self,
        job_id: UUID,
        *,
        now: datetime | None = None,
    ) -> JobResource:
        requested_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(str(job_id))
            status = JobStatus(row["status"])
            if status in {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.INTERRUPTED,
            }:
                return self._job_from_row(row)
            if status == JobStatus.QUEUED:
                next_status = JobStatus.CANCELLED
                completed_at = _timestamp(requested_at)
            else:
                next_status = JobStatus.CANCELLING
                completed_at = None
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, cancel_requested = 1, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    next_status.value,
                    _timestamp(requested_at),
                    completed_at,
                    str(job_id),
                ),
            )
            self._append_job_event(
                connection,
                job_id,
                JobEventType.STATE,
                JobEventLevel.INFO,
                message=next_status.value,
                data={"status": next_status.value},
                now=requested_at,
            )
        return self.get_job(job_id)

    def complete_job(
        self,
        job_id: UUID,
        result: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> JobResource:
        completed_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM jobs WHERE id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(str(job_id))
            if row["status"] == JobStatus.SUCCEEDED.value:
                return self.get_job(job_id)
            if row["status"] != JobStatus.RUNNING.value:
                raise InvalidJobStateError(f"cannot complete job in state {row['status']}")
            connection.execute(
                """
                UPDATE jobs SET status = ?, progress = 1, result_json = ?,
                    error_json = NULL, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    JobStatus.SUCCEEDED.value,
                    json.dumps(result, separators=(",", ":"), sort_keys=True),
                    _timestamp(completed_at),
                    _timestamp(completed_at),
                    str(job_id),
                ),
            )
            self._append_job_event(
                connection,
                job_id,
                JobEventType.STATE,
                JobEventLevel.INFO,
                message=JobStatus.SUCCEEDED.value,
                progress=1.0,
                data={"status": JobStatus.SUCCEEDED.value},
                now=completed_at,
            )
        return self.get_job(job_id)

    def fail_job(
        self,
        job_id: UUID,
        error: ErrorDetail,
        *,
        now: datetime | None = None,
    ) -> JobResource:
        return self._finish_job(
            job_id,
            JobStatus.FAILED,
            error=error,
            now=now,
        )

    def cancel_job(self, job_id: UUID, *, now: datetime | None = None) -> JobResource:
        return self._finish_job(job_id, JobStatus.CANCELLED, now=now)

    def interrupt_job(self, job_id: UUID, *, now: datetime | None = None) -> JobResource:
        return self._finish_job(
            job_id,
            JobStatus.INTERRUPTED,
            error=ErrorDetail(
                code="service_interrupted",
                message="job execution was interrupted by service shutdown",
                retryable=True,
            ),
            now=now,
        )

    def _finish_job(
        self,
        job_id: UUID,
        status: JobStatus,
        *,
        error: ErrorDetail | None = None,
        now: datetime | None = None,
    ) -> JobResource:
        if status not in {JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED}:
            raise ValueError(f"unsupported terminal job status {status}")
        completed_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM jobs WHERE id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(str(job_id))
            current = JobStatus(row["status"])
            if current in {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.INTERRUPTED,
            }:
                return self.get_job(job_id)
            connection.execute(
                """
                UPDATE jobs SET status = ?, error_json = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    error.model_dump_json() if error is not None else None,
                    _timestamp(completed_at),
                    _timestamp(completed_at),
                    str(job_id),
                ),
            )
            self._append_job_event(
                connection,
                job_id,
                JobEventType.STATE,
                JobEventLevel.ERROR if status == JobStatus.FAILED else JobEventLevel.INFO,
                message=status.value,
                data={"status": status.value},
                now=completed_at,
            )
        return self.get_job(job_id)

    def recover_interrupted_jobs(self, *, now: datetime | None = None) -> list[UUID]:
        recovered_at = now or _utcnow()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id FROM jobs WHERE status IN (?, ?)",
                (JobStatus.RUNNING.value, JobStatus.CANCELLING.value),
            ).fetchall()
            for row in rows:
                job_id = UUID(row["id"])
                connection.execute(
                    """
                    UPDATE jobs SET status = ?, error_json = ?, updated_at = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (
                        JobStatus.INTERRUPTED.value,
                        ErrorDetail(
                            code="service_restarted",
                            message="job execution was interrupted by service restart",
                            retryable=True,
                        ).model_dump_json(),
                        _timestamp(recovered_at),
                        _timestamp(recovered_at),
                        str(job_id),
                    ),
                )
                self._append_job_event(
                    connection,
                    job_id,
                    JobEventType.STATE,
                    JobEventLevel.WARNING,
                    message=JobStatus.INTERRUPTED.value,
                    data={"status": JobStatus.INTERRUPTED.value, "reason": "service_restarted"},
                    now=recovered_at,
                )
        return [UUID(row["id"]) for row in rows]

    def append_runtime_metric(
        self,
        values: dict[str, Any],
        *,
        instance_id: UUID | None = None,
        model: str | None = None,
        now: datetime | None = None,
    ) -> RuntimeMetricSnapshot:
        captured_at = now or _utcnow()
        encoded = json.dumps(values, separators=(",", ":"), sort_keys=True)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runtime_metrics(instance_id, model, values_json, captured_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(instance_id) if instance_id is not None else None,
                    model,
                    encoded,
                    _timestamp(captured_at),
                ),
            )
            sequence = int(cursor.lastrowid)
            if sequence > _MAX_RUNTIME_METRICS:
                connection.execute(
                    "DELETE FROM runtime_metrics WHERE sequence <= ?",
                    (sequence - _MAX_RUNTIME_METRICS,),
                )
        return RuntimeMetricSnapshot(
            sequence=sequence,
            instance_id=instance_id,
            model=model,
            values=values,
            captured_at=captured_at,
        )

    def list_runtime_metrics(
        self,
        *,
        instance_id: UUID | None = None,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[RuntimeMetricSnapshot]:
        if limit < 1 or limit > 2000:
            raise ValueError("limit must be between 1 and 2000")
        clauses: list[str] = []
        values: list[object] = []
        if instance_id is not None:
            clauses.append("instance_id = ?")
            values.append(str(instance_id))
        if since is not None:
            clauses.append("captured_at >= ?")
            values.append(_timestamp(since))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM runtime_metrics {where}
                ORDER BY sequence DESC LIMIT ?
                """,
                (*values, limit),
            ).fetchall()
        return [self._runtime_metric_from_row(row) for row in reversed(rows)]

    def append_runtime_log(
        self,
        level: RuntimeLogLevel,
        message: str,
        *,
        instance_id: UUID | None = None,
        fields: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> RuntimeLogEntry:
        created_at = now or _utcnow()
        payload = fields or {}
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runtime_logs(instance_id, level, message, fields_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(instance_id) if instance_id is not None else None,
                    level.value,
                    message,
                    json.dumps(payload, separators=(",", ":"), sort_keys=True),
                    _timestamp(created_at),
                ),
            )
            sequence = int(cursor.lastrowid)
        return RuntimeLogEntry(
            sequence=sequence,
            instance_id=instance_id,
            level=level,
            message=message,
            fields=payload,
            created_at=created_at,
        )

    def list_runtime_logs(
        self,
        *,
        instance_id: UUID | None = None,
        level: RuntimeLogLevel | None = None,
        after: int = 0,
        limit: int = 200,
    ) -> list[RuntimeLogEntry]:
        if after < 0:
            raise ValueError("after must be non-negative")
        if limit < 1 or limit > 2000:
            raise ValueError("limit must be between 1 and 2000")
        clauses = ["sequence > ?"]
        values: list[object] = [after]
        if instance_id is not None:
            clauses.append("instance_id = ?")
            values.append(str(instance_id))
        if level is not None:
            clauses.append("level = ?")
            values.append(level.value)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM runtime_logs
                WHERE {" AND ".join(clauses)}
                ORDER BY sequence ASC LIMIT ?
                """,
                (*values, limit),
            ).fetchall()
        return [self._runtime_log_from_row(row) for row in rows]

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
    def _media_from_row(row: sqlite3.Row) -> MediaResource:
        return MediaResource(
            media=MediaRef(
                id=UUID(row["id"]),
                sha256=row["sha256"],
                mime_type=row["mime_type"],
                byte_size=int(row["byte_size"]),
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _mcp_server_from_row(row: sqlite3.Row) -> McpServerResource:
        config = json.loads(row["config_json"])
        return McpServerResource(
            id=UUID(row["id"]),
            name=row["name"],
            transport=McpTransport(row["transport"]),
            enabled=bool(row["enabled"]),
            url=config.get("url"),
            command=config.get("command"),
            args=config.get("args") or [],
            header_env=config.get("header_env") or {},
            timeout_seconds=float(config.get("timeout_seconds", 30.0)),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _generation_preset_from_row(row: sqlite3.Row) -> GenerationPresetResource:
        return GenerationPresetResource(
            id=UUID(row["id"]),
            name=row["name"],
            model=row["model"],
            mode=SessionMode(row["mode"]) if row["mode"] else None,
            settings=ResponseRequestSettings.model_validate_json(row["settings_json"]),
            context_size=int(row["context_size"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _runtime_profile_from_row(row: sqlite3.Row) -> RuntimeProfileResource:
        load = ModelLoadRequest.model_validate_json(row["load_json"])
        stored_sampling = (
            SamplingParams.model_validate_json(row["sampling_json"])
            if row["sampling_json"]
            else None
        )
        return RuntimeProfileResource(
            id=UUID(row["id"]),
            name=row["name"],
            load=load.model_copy(update={"sampling_defaults": stored_sampling}),
            artifact_id=row["artifact_id"],
            artifact_modified_at=datetime.fromisoformat(row["artifact_modified_at"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _artifact_lineage_from_row(self, row: sqlite3.Row) -> ArtifactLineageResource:
        with self._connection() as connection:
            validations = connection.execute(
                """
                SELECT job_id FROM artifact_validations
                WHERE artifact_uri = ? ORDER BY created_at, job_id
                """,
                (row["artifact_uri"],),
            ).fetchall()
        return ArtifactLineageResource(
            id=UUID(row["id"]),
            artifact_uri=row["artifact_uri"],
            artifact_name=row["artifact_name"],
            producer_job_id=UUID(row["producer_job_id"]),
            producer_kind=row["producer_kind"],
            source_uris=json.loads(row["source_uris_json"]),
            parameters=json.loads(row["parameters_json"]),
            metadata=json.loads(row["metadata_json"]),
            validation_job_ids=[UUID(item["job_id"]) for item in validations],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _dataset_from_row(row: sqlite3.Row) -> DatasetResource:
        return DatasetResource(
            id=UUID(row["id"]),
            name=row["name"],
            kind=row["kind"],
            artifact_uri=row["artifact_uri"],
            sha256=row["sha256"],
            byte_size=int(row["byte_size"]),
            source_uri=row["source_uri"],
            revision=row["revision"],
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _evaluation_from_row(row: sqlite3.Row) -> EvaluationResultResource:
        return EvaluationResultResource(
            id=UUID(row["id"]),
            job_id=UUID(row["job_id"]),
            kind=row["kind"],
            model_id=row["model_id"],
            metrics=json.loads(row["metrics_json"]),
            parameters=json.loads(row["parameters_json"]),
            dataset_id=UUID(row["dataset_id"]) if row["dataset_id"] else None,
            dataset_manifest=json.loads(row["dataset_manifest_json"]),
            hardware_identity=json.loads(row["hardware_identity_json"]),
            runtime_identity=json.loads(row["runtime_identity_json"]),
            comparison_key=row["comparison_key"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _remote_node_from_row(row: sqlite3.Row) -> RemoteNodeResource:
        return RemoteNodeResource(
            id=UUID(row["id"]),
            name=row["name"],
            url=row["url"],
            api_key_env=row["api_key_env"],
            enabled=bool(row["enabled"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _api_key_from_row(row: sqlite3.Row) -> ApiKeyResource:
        return ApiKeyResource(
            id=UUID(row["id"]),
            name=row["name"],
            prefix=row["prefix"],
            scopes=json.loads(row["scopes_json"]),
            role=row["role"],
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            revoked_at=datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            last_used_at=datetime.fromisoformat(row["last_used_at"])
            if row["last_used_at"]
            else None,
        )

    @staticmethod
    def _response_from_row(row: sqlite3.Row) -> ResponseResource:
        row_keys = set(row.keys())
        usage = TokenUsage.model_validate_json(row["usage_json"]) if row["usage_json"] else None
        performance = (
            ResponsePerformance.model_validate_json(row["performance_json"])
            if "performance_json" in row_keys and row["performance_json"]
            else None
        )
        settings = (
            ResponseRequestSettings.model_validate_json(row["settings_json"])
            if "settings_json" in row_keys and row["settings_json"]
            else None
        )
        error = ErrorDetail.model_validate_json(row["error_json"]) if row["error_json"] else None
        completed_at = row["completed_at"]
        return ResponseResource(
            id=UUID(row["id"]),
            request_id=UUID(row["request_id"]),
            session_id=UUID(row["session_id"]),
            status=ResponseStatus(row["status"]),
            output_message_id=UUID(row["output_message_id"])
            if "output_message_id" in row_keys and row["output_message_id"]
            else None,
            output=_CONTENT_PARTS.validate_json(row["output_json"]),
            finish_reason=row["finish_reason"],
            usage=usage,
            performance=performance,
            settings=settings,
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
            error=error,
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobResource:
        return JobResource(
            id=UUID(row["id"]),
            kind=row["kind"],
            status=JobStatus(row["status"]),
            payload=json.loads(row["payload_json"]),
            progress=float(row["progress"]),
            cancel_requested=bool(row["cancel_requested"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=ErrorDetail.model_validate_json(row["error_json"]) if row["error_json"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
        )

    @staticmethod
    def _job_event_from_row(row: sqlite3.Row) -> JobEventResource:
        return JobEventResource(
            job_id=UUID(row["job_id"]),
            sequence=int(row["sequence"]),
            type=JobEventType(row["type"]),
            level=JobEventLevel(row["level"]),
            message=row["message"],
            progress=float(row["progress"]) if row["progress"] is not None else None,
            data=json.loads(row["data_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _runtime_metric_from_row(row: sqlite3.Row) -> RuntimeMetricSnapshot:
        return RuntimeMetricSnapshot(
            sequence=int(row["sequence"]),
            instance_id=UUID(row["instance_id"]) if row["instance_id"] else None,
            model=row["model"],
            values=json.loads(row["values_json"]),
            captured_at=datetime.fromisoformat(row["captured_at"]),
        )

    @staticmethod
    def _runtime_log_from_row(row: sqlite3.Row) -> RuntimeLogEntry:
        return RuntimeLogEntry(
            sequence=int(row["sequence"]),
            instance_id=UUID(row["instance_id"]) if row["instance_id"] else None,
            level=RuntimeLogLevel(row["level"]),
            message=row["message"],
            fields=json.loads(row["fields_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
