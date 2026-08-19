"""MFQ Server session orchestration over a persistent store and streaming backend."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from mfq.server.backend import BackendDelta, BackendError, BackendToolCallDelta, ChatBackend
from mfq.server.catalog import ModelArtifactNotFoundError, ModelCatalog
from mfq.server.documents import DocumentExtractionError, extract_document
from mfq.server.hub import HubCatalog, HubError, HubProvider
from mfq.server.jobs import (
    JobExecutionError,
    JobKindNotRegisteredError,
    JobManager,
    TypedJobHandler,
)
from mfq.server.mcp import McpClient, McpError
from mfq.server.models import (
    AppendMessageRequest,
    AppendMessageResult,
    ArtifactLineageList,
    AudioPart,
    CompareEvaluationsRequest,
    ContentPart,
    CreateDatasetRequest,
    CreateDocumentRequest,
    CreateGenerationPresetRequest,
    CreateJobRequest,
    CreateMcpServerRequest,
    CreateRemoteNodeRequest,
    CreateResponseRequest,
    CreateRuntimeProfileRequest,
    CreateSessionRequest,
    DatasetList,
    DatasetResource,
    DeleteWorkspaceArtifactRequest,
    DocumentPart,
    DocumentResource,
    ErrorDetail,
    ErrorEvent,
    EvaluationComparisonResource,
    EvaluationComparisonRow,
    EvaluationKind,
    EvaluationResultList,
    ForkSessionRequest,
    GeneratedAudioPart,
    GenerationPresetList,
    GenerationPresetResource,
    HubModelInfo,
    HubModelSearchResult,
    ImagePart,
    JobEventList,
    JobKindList,
    JobList,
    JobResource,
    JobStatus,
    McpServerList,
    McpServerResource,
    McpToolCallRequest,
    McpToolCallResult,
    McpToolList,
    McpToolResource,
    MediaResource,
    Message,
    MessageList,
    MessageRole,
    ModelArtifactList,
    ModelLoadRequest,
    ModelUnloadRequest,
    OperationAccepted,
    PortableMedia,
    PortableMessage,
    RealtimeFrame,
    RealtimePayload,
    ReasoningPart,
    RemoteNodeList,
    RemoteNodeResource,
    ResponseCompleted,
    ResponseList,
    ResponsePerformance,
    ResponseReasoningDelta,
    ResponseRequestSettings,
    ResponseResource,
    ResponseStatus,
    ResponseTextDelta,
    ResponseToolCallDelta,
    RewindSessionRequest,
    RuntimeCapabilitiesResource,
    RuntimeInstanceList,
    RuntimeLogLevel,
    RuntimeLogList,
    RuntimeMetricList,
    RuntimeProfileList,
    RuntimeProfileLoadRequest,
    RuntimeProfileResource,
    SessionArchive,
    SessionImportResult,
    SessionList,
    SessionResource,
    SessionState,
    SessionStateChanged,
    TextPart,
    TokenUsage,
    ToolCallPart,
    ToolResultPart,
    TranscriptPart,
    UpdateGenerationPresetRequest,
    UpdateMcpServerRequest,
    UpdateRemoteNodeRequest,
    UpdateRuntimeProfileRequest,
    UpdateSessionRequest,
    VideoPart,
)
from mfq.server.storage import (
    BeginResponseResult,
    DatasetNotFoundError,
    EvaluationNotFoundError,
    GenerationPresetNotFoundError,
    IdempotencyConflictError,
    InvalidJobStateError,
    JobNotFoundError,
    MediaIntegrityError,
    MediaNotFoundError,
    MessageNotFoundError,
    RemoteNodeNotFoundError,
    ResponseInProgressError,
    RevisionConflictError,
    RuntimeProfileNotFoundError,
    SessionNotFoundError,
    SessionStore,
    StorageError,
)

MAX_MEDIA_BYTES = 512 * 1024 * 1024


class ServiceError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = ErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        )


@dataclass(frozen=True)
class PreparedResponse:
    request: CreateResponseRequest
    begin: BeginResponseResult
    backend_messages: tuple[dict[str, Any], ...]


@dataclass
class _ToolCall:
    call_id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)


@dataclass
class _OutputAccumulator:
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: dict[int, _ToolCall] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    performance: ResponsePerformance | None = None

    def apply(self, delta: BackendDelta) -> None:
        if delta.content_delta:
            self.text.append(delta.content_delta)
        if delta.reasoning_delta:
            self.reasoning.append(delta.reasoning_delta)
        for tool_delta in delta.tool_calls:
            self._apply_tool_delta(tool_delta)
        if delta.finish_reason is not None:
            self.finish_reason = delta.finish_reason
        if delta.usage is not None:
            self.usage = delta.usage
        if delta.performance is not None:
            self.performance = delta.performance

    def output_parts(self) -> list[ContentPart]:
        parts: list[ContentPart] = []
        reasoning = "".join(self.reasoning)
        text = "".join(self.text)
        if reasoning:
            parts.append(ReasoningPart(text=reasoning))
        if text:
            parts.append(TextPart(text=text))
        for index in sorted(self.tool_calls):
            tool = self.tool_calls[index]
            if not tool.call_id or not tool.name:
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {index} is missing its id or name",
                )
            raw_arguments = "".join(tool.arguments) or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {index} returned invalid JSON arguments: {error}",
                ) from error
            if not isinstance(arguments, dict):
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {index} arguments must decode to an object",
                )
            parts.append(ToolCallPart(call_id=tool.call_id, name=tool.name, arguments=arguments))
        if not parts:
            parts.append(TextPart(text=""))
        return parts

    def _apply_tool_delta(self, delta: BackendToolCallDelta) -> None:
        tool = self.tool_calls.setdefault(delta.index, _ToolCall())
        if delta.call_id is not None:
            if tool.call_id is not None and tool.call_id != delta.call_id:
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {delta.index} changed id during streaming",
                )
            tool.call_id = delta.call_id
        if delta.name is not None:
            if tool.name is not None and tool.name != delta.name:
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {delta.index} changed name during streaming",
                )
            tool.name = delta.name
        if delta.arguments_delta:
            tool.arguments.append(delta.arguments_delta)


class ServerService:
    def __init__(
        self,
        store: SessionStore,
        backend: ChatBackend,
        *,
        jobs: JobManager | None = None,
        catalog: ModelCatalog | None = None,
        runtime_manager: Any | None = None,
        hub_catalog: HubCatalog | None = None,
        tool_handlers: Any | None = None,
        cluster: Any | None = None,
    ) -> None:
        self.store = store
        self.backend = backend
        self.jobs = jobs or JobManager(store)
        self.catalog = catalog
        self.runtime_manager = runtime_manager
        self.hub_catalog = hub_catalog or HubCatalog()
        self.tool_handlers = tool_handlers
        self.cluster = cluster
        if runtime_manager is not None:
            runtime_manager.store = store
            self.jobs.register(
                "model.load", TypedJobHandler(runtime_manager.load, ModelLoadRequest)
            )
            self.jobs.register(
                "model.unload", TypedJobHandler(runtime_manager.unload, ModelUnloadRequest)
            )

    async def start(self) -> None:
        if self.runtime_manager is not None:
            await self.runtime_manager.start()
        await self.jobs.start()

    async def aclose(self) -> None:
        await self.jobs.close()
        await self.backend.aclose()

    async def create_job(self, request: CreateJobRequest) -> JobResource:
        try:
            return await self.jobs.submit(request)
        except JobKindNotRegisteredError as error:
            raise ServiceError(
                422,
                "job_kind_unavailable",
                f"job kind is not registered: {error}",
            ) from error

    async def job_kinds(self) -> JobKindList:
        return self.jobs.kinds()

    async def list_jobs(
        self,
        *,
        status: JobStatus | None,
        kind: str | None,
        limit: int,
        offset: int,
    ) -> JobList:
        return await self.jobs.list_jobs(
            status=status,
            kind=kind,
            limit=limit,
            offset=offset,
        )

    async def get_job(self, job_id: UUID) -> JobResource:
        try:
            return await asyncio.to_thread(self.store.get_job, job_id)
        except JobNotFoundError as error:
            raise ServiceError(404, "job_not_found", str(error)) from error

    async def cancel_job(self, job_id: UUID) -> JobResource:
        try:
            return await self.jobs.cancel(job_id)
        except JobNotFoundError as error:
            raise ServiceError(404, "job_not_found", str(error)) from error

    async def retry_job(self, job_id: UUID) -> JobResource:
        try:
            return await self.jobs.retry(job_id)
        except JobNotFoundError as error:
            raise ServiceError(404, "job_not_found", str(error)) from error
        except InvalidJobStateError as error:
            raise ServiceError(409, "job_state_conflict", str(error)) from error

    async def search_hub_models(
        self, provider: HubProvider, query: str, limit: int
    ) -> HubModelSearchResult:
        try:
            return await self.hub_catalog.search(provider, query, limit=limit)
        except HubError as error:
            raise ServiceError(502, "model_hub_error", str(error), retryable=True) from error

    async def hub_model_info(
        self, provider: HubProvider, repo_id: str, revision: str | None
    ) -> HubModelInfo:
        try:
            return await self.hub_catalog.info(provider, repo_id, revision)
        except HubError as error:
            raise ServiceError(502, "model_hub_error", str(error), retryable=True) from error

    async def delete_workspace_artifact(
        self, request: DeleteWorkspaceArtifactRequest
    ) -> dict[str, Any]:
        if self.tool_handlers is None:
            raise ServiceError(
                501,
                "workspace_management_unavailable",
                "workspace artifact removal is not configured",
            )
        try:
            result = await asyncio.to_thread(
                self.tool_handlers.remove_workspace_artifact,
                request.artifact_uri,
            )
        except JobExecutionError as error:
            status = 404 if error.detail.code == "artifact_not_found" else 422
            raise ServiceError(status, error.detail.code, error.detail.message) from error
        return {"status": "ok", "artifact_uri": request.artifact_uri, **result}

    async def list_job_events(
        self,
        job_id: UUID,
        *,
        after: int,
        limit: int,
    ) -> JobEventList:
        try:
            events = await asyncio.to_thread(
                self.store.list_job_events,
                job_id,
                after=after,
                limit=limit,
            )
        except JobNotFoundError as error:
            raise ServiceError(404, "job_not_found", str(error)) from error
        return JobEventList(data=events)

    async def create_session(self, request: CreateSessionRequest) -> SessionResource:
        return await asyncio.to_thread(self.store.create_session, request)

    async def create_generation_preset(
        self, request: CreateGenerationPresetRequest
    ) -> GenerationPresetResource:
        try:
            return await asyncio.to_thread(self.store.create_generation_preset, request)
        except StorageError as error:
            raise ServiceError(409, "generation_preset_conflict", str(error)) from error

    async def list_generation_presets(self) -> GenerationPresetList:
        presets = await asyncio.to_thread(self.store.list_generation_presets)
        return GenerationPresetList(data=presets)

    async def update_generation_preset(
        self, preset_id: UUID, request: UpdateGenerationPresetRequest
    ) -> GenerationPresetResource:
        try:
            return await asyncio.to_thread(self.store.update_generation_preset, preset_id, request)
        except GenerationPresetNotFoundError as error:
            raise ServiceError(404, "generation_preset_not_found", str(error)) from error
        except StorageError as error:
            raise ServiceError(409, "generation_preset_conflict", str(error)) from error

    async def delete_generation_preset(self, preset_id: UUID) -> None:
        try:
            await asyncio.to_thread(self.store.delete_generation_preset, preset_id)
        except GenerationPresetNotFoundError as error:
            raise ServiceError(404, "generation_preset_not_found", str(error)) from error

    async def create_runtime_profile(
        self, request: CreateRuntimeProfileRequest
    ) -> RuntimeProfileResource:
        artifact = await self._resolve_profile_artifact(request.load)
        try:
            return await asyncio.to_thread(
                self.store.create_runtime_profile,
                request,
                artifact.resource.id,
                artifact.resource.modified_at,
            )
        except StorageError as error:
            raise ServiceError(409, "runtime_profile_conflict", str(error)) from error

    async def list_runtime_profiles(self) -> RuntimeProfileList:
        profiles = await asyncio.to_thread(self.store.list_runtime_profiles)
        checked = [await self._runtime_profile_drift(profile) for profile in profiles]
        return RuntimeProfileList(data=checked)

    async def get_runtime_profile(self, profile_id: UUID) -> RuntimeProfileResource:
        try:
            profile = await asyncio.to_thread(self.store.get_runtime_profile, profile_id)
        except RuntimeProfileNotFoundError as error:
            raise ServiceError(404, "runtime_profile_not_found", str(error)) from error
        return await self._runtime_profile_drift(profile)

    async def update_runtime_profile(
        self, profile_id: UUID, request: UpdateRuntimeProfileRequest
    ) -> RuntimeProfileResource:
        artifact = await self._resolve_profile_artifact(request.load)
        try:
            return await asyncio.to_thread(
                self.store.update_runtime_profile,
                profile_id,
                request,
                artifact.resource.id,
                artifact.resource.modified_at,
            )
        except RuntimeProfileNotFoundError as error:
            raise ServiceError(404, "runtime_profile_not_found", str(error)) from error
        except StorageError as error:
            raise ServiceError(409, "runtime_profile_conflict", str(error)) from error

    async def delete_runtime_profile(self, profile_id: UUID) -> None:
        try:
            await asyncio.to_thread(self.store.delete_runtime_profile, profile_id)
        except RuntimeProfileNotFoundError as error:
            raise ServiceError(404, "runtime_profile_not_found", str(error)) from error

    async def load_runtime_profile(
        self, profile_id: UUID, request: RuntimeProfileLoadRequest
    ) -> OperationAccepted:
        profile = await self.get_runtime_profile(profile_id)
        if profile.drifted and not request.allow_drift:
            raise ServiceError(
                409,
                "runtime_profile_drift",
                profile.drift_reason or "the model artifact changed after this profile was saved",
            )
        return await self.load_model(profile.load)

    async def _resolve_profile_artifact(self, request: ModelLoadRequest):
        if self.catalog is None:
            raise ServiceError(
                501,
                "model_catalog_unavailable",
                "runtime profiles require a configured model catalog",
            )
        try:
            return await self.catalog.resolve(request.model, request.artifact_uri)
        except ModelArtifactNotFoundError as error:
            raise ServiceError(404, "model_artifact_not_found", str(error)) from error

    async def _runtime_profile_drift(
        self, profile: RuntimeProfileResource
    ) -> RuntimeProfileResource:
        try:
            artifact = await self._resolve_profile_artifact(profile.load)
        except ServiceError as error:
            return profile.model_copy(
                update={
                    "drifted": True,
                    "drift_reason": str(error),
                }
            )
        changed = (
            artifact.resource.id != profile.artifact_id
            or artifact.resource.modified_at != profile.artifact_modified_at
        )
        return profile.model_copy(
            update={
                "drifted": changed,
                "drift_reason": "the selected model artifact changed after this profile was saved"
                if changed
                else None,
            }
        )

    async def list_sessions(self, *, limit: int, offset: int) -> SessionList:
        sessions = await asyncio.to_thread(self.store.list_sessions, limit=limit, offset=offset)
        return SessionList(data=sessions)

    async def get_session(self, session_id: UUID) -> SessionResource:
        try:
            return await asyncio.to_thread(self.store.get_session, session_id)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error

    async def update_session(
        self,
        session_id: UUID,
        request: UpdateSessionRequest,
    ) -> SessionResource:
        try:
            return await asyncio.to_thread(self.store.update_session, session_id, request)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error

    async def list_messages(self, session_id: UUID) -> MessageList:
        try:
            messages = await asyncio.to_thread(self.store.list_messages, session_id)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        return MessageList(data=messages)

    async def export_session(self, session_id: UUID) -> SessionArchive:
        session = await self.get_session(session_id)
        messages = (await self.list_messages(session_id)).data
        media_by_sha: dict[str, PortableMedia] = {}
        for message in messages:
            for part in message.parts:
                media_ref = getattr(part, "media", None)
                if media_ref is None or media_ref.sha256 in media_by_sha:
                    continue
                resource, path = await self.get_media(media_ref.id)
                document: dict[str, Any] | None = None
                if isinstance(part, DocumentPart):
                    extracted = await self.get_document(media_ref.id)
                    document = {
                        "name": extracted.name,
                        "text": extracted.text,
                        "page_count": extracted.page_count,
                        "extractor": extracted.extractor,
                    }
                media_by_sha[media_ref.sha256] = PortableMedia(
                    sha256=resource.media.sha256,
                    mime_type=resource.media.mime_type,
                    data_base64=base64.b64encode(await asyncio.to_thread(path.read_bytes)),
                    document=document,
                )
        return SessionArchive(
            session=session,
            messages=[
                PortableMessage(role=item.role, parts=item.parts, created_at=item.created_at)
                for item in messages
            ],
            media=list(media_by_sha.values()),
        )

    async def import_session(self, archive: SessionArchive) -> SessionImportResult:
        media_map: dict[str, MediaResource] = {}
        for portable in archive.media:
            raw = bytes(portable.data_base64)
            resource = await self.upload_media(raw, portable.mime_type, portable.sha256)
            media_map[portable.sha256] = resource
            if portable.document is not None:
                document = portable.document
                await asyncio.to_thread(
                    self.store.put_document,
                    resource.media.id,
                    str(document.get("name") or "document"),
                    str(document.get("text") or ""),
                    str(document.get("extractor") or "archive"),
                    page_count=document.get("page_count"),
                )
        session = await self.create_session(
            CreateSessionRequest(
                model=archive.session.model,
                mode=archive.session.mode,
                title=archive.session.title,
                metadata={**archive.session.metadata, "imported_from": str(archive.session.id)},
            )
        )
        for message in archive.messages:
            parts: list[ContentPart] = []
            for part in message.parts:
                media_ref = getattr(part, "media", None)
                if media_ref is None:
                    parts.append(part)
                    continue
                replacement = media_map.get(media_ref.sha256)
                if replacement is None:
                    raise ServiceError(
                        422,
                        "archive_media_missing",
                        f"archive does not contain media {media_ref.sha256}",
                    )
                parts.append(part.model_copy(update={"media": replacement.media}))
            session, _ = await asyncio.to_thread(
                self.store.append_message,
                session.id,
                session.revision,
                message.role,
                parts,
                now=message.created_at,
            )
        return SessionImportResult(
            session=session,
            messages_imported=len(archive.messages),
            media_imported=len(media_map),
        )

    async def append_message(
        self,
        session_id: UUID,
        request: AppendMessageRequest,
    ) -> AppendMessageResult:
        try:
            await asyncio.to_thread(self._validate_media_parts, request.parts)
            session, message = await asyncio.to_thread(
                self.store.append_message,
                session_id,
                request.expected_revision,
                request.role,
                request.parts,
            )
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except RevisionConflictError as error:
            raise ServiceError(
                409,
                "revision_conflict",
                str(error),
                details={"expected_revision": error.expected, "actual_revision": error.actual},
            ) from error
        except StorageError as error:
            raise ServiceError(409, "session_state_conflict", str(error)) from error
        return AppendMessageResult(session=session, message=message)

    async def upload_media(
        self,
        data: bytes,
        mime_type: str,
        expected_sha256: str,
    ) -> MediaResource:
        normalized_mime = mime_type.split(";", 1)[0].strip().lower()
        if not normalized_mime.startswith(
            ("image/", "audio/", "video/", "text/")
        ) and normalized_mime not in {
            "application/json",
            "application/pdf",
            "application/xml",
            "application/yaml",
            "application/octet-stream",
            "application/x-mfq-imatrix",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }:
            raise ServiceError(
                415,
                "unsupported_media_type",
                "unsupported media or document MIME type",
            )
        if not data:
            raise ServiceError(400, "empty_media", "media upload must not be empty")
        if len(data) > MAX_MEDIA_BYTES:
            raise ServiceError(
                413,
                "media_too_large",
                f"media upload exceeds the {MAX_MEDIA_BYTES} byte limit",
            )
        try:
            return await asyncio.to_thread(
                self.store.put_media,
                data,
                normalized_mime,
                expected_sha256,
            )
        except MediaIntegrityError as error:
            raise ServiceError(422, "media_digest_mismatch", str(error)) from error

    async def get_media(self, media_id: UUID) -> tuple[MediaResource, Any]:
        try:
            return await asyncio.to_thread(self.store.get_media_path, media_id)
        except MediaNotFoundError as error:
            raise ServiceError(404, "media_not_found", str(error)) from error

    async def create_document(self, request: CreateDocumentRequest) -> DocumentResource:
        try:
            media, path = await asyncio.to_thread(self.store.get_media_path, request.media_id)
            extracted = await asyncio.to_thread(
                extract_document, path, media.media.mime_type, request.name
            )
            return await asyncio.to_thread(
                self.store.put_document,
                request.media_id,
                request.name,
                extracted.text,
                extracted.extractor,
                page_count=extracted.page_count,
            )
        except MediaNotFoundError as error:
            raise ServiceError(404, "media_not_found", str(error)) from error
        except DocumentExtractionError as error:
            raise ServiceError(422, "document_extraction_failed", str(error)) from error

    async def get_document(self, media_id: UUID) -> DocumentResource:
        try:
            return await asyncio.to_thread(self.store.get_document, media_id)
        except MediaNotFoundError as error:
            raise ServiceError(404, "document_not_found", str(error)) from error

    async def create_mcp_server(self, request: CreateMcpServerRequest) -> McpServerResource:
        try:
            return await asyncio.to_thread(self.store.create_mcp_server, request)
        except StorageError as error:
            raise ServiceError(409, "mcp_server_conflict", str(error)) from error

    async def list_mcp_servers(self) -> McpServerList:
        return McpServerList(data=await asyncio.to_thread(self.store.list_mcp_servers))

    async def update_mcp_server(
        self, server_id: UUID, request: UpdateMcpServerRequest
    ) -> McpServerResource:
        try:
            return await asyncio.to_thread(
                self.store.set_mcp_server_enabled, server_id, request.enabled
            )
        except MediaNotFoundError as error:
            raise ServiceError(404, "mcp_server_not_found", str(error)) from error

    async def delete_mcp_server(self, server_id: UUID) -> None:
        try:
            await asyncio.to_thread(self.store.delete_mcp_server, server_id)
        except MediaNotFoundError as error:
            raise ServiceError(404, "mcp_server_not_found", str(error)) from error

    async def list_mcp_tools(self) -> McpToolList:
        servers = await asyncio.to_thread(self.store.list_mcp_servers, enabled_only=True)

        async def discover(
            server: McpServerResource,
        ) -> tuple[list[McpToolResource], str | None]:
            try:
                return await McpClient(server).list_tools(), None
            except (McpError, OSError) as error:
                return [], str(error)

        results = await asyncio.gather(*(discover(server) for server in servers))
        tools: list[McpToolResource] = []
        errors: dict[str, str] = {}
        for server, (items, error) in zip(servers, results, strict=True):
            tools.extend(items)
            if error:
                errors[server.name] = error
        return McpToolList(data=tools, errors=errors)

    async def call_mcp_tool(self, request: McpToolCallRequest) -> McpToolCallResult:
        if not request.confirm:
            raise ServiceError(
                409,
                "tool_confirmation_required",
                "tool execution requires explicit confirmation",
            )
        servers = await asyncio.to_thread(self.store.list_mcp_servers, enabled_only=True)
        server_name, separator, tool_name = request.name.partition(".")
        if not separator or not server_name or not tool_name:
            raise ServiceError(
                422,
                "invalid_tool_name",
                "MCP tools must use the server.tool qualified name",
            )
        server = next((item for item in servers if item.name == server_name), None)
        if server is None:
            raise ServiceError(404, "mcp_server_not_found", server_name)
        started = time.perf_counter()
        try:
            client = McpClient(server)
            available = await client.list_tools()
            if not any(item.name == tool_name for item in available):
                raise ServiceError(404, "mcp_tool_not_found", request.name)
            result = await client.call_tool(tool_name, request.arguments)
        except ServiceError:
            raise
        except (McpError, OSError) as error:
            await asyncio.to_thread(
                self.store.append_runtime_log,
                RuntimeLogLevel.ERROR,
                "MCP tool call failed",
                fields={"server": server.name, "tool": tool_name},
            )
            raise ServiceError(502, "mcp_tool_failed", str(error)) from error
        await asyncio.to_thread(
            self.store.append_runtime_log,
            RuntimeLogLevel.INFO,
            "MCP tool call completed",
            fields={
                "server": server.name,
                "tool": tool_name,
                "is_error": result.is_error,
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            },
        )
        return result

    async def fork_session(
        self,
        session_id: UUID,
        request: ForkSessionRequest,
    ) -> SessionResource:
        try:
            forked = await asyncio.to_thread(self.store.fork_session, session_id, request)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except MessageNotFoundError as error:
            raise ServiceError(404, "message_not_found", str(error)) from error
        except StorageError as error:
            raise ServiceError(409, "session_state_conflict", str(error)) from error
        if request.at_message_id is None:
            await self.backend.fork_session(session_id, forked.id)
        return forked

    async def rewind_session(
        self,
        session_id: UUID,
        request: RewindSessionRequest,
    ) -> SessionResource:
        try:
            return await asyncio.to_thread(
                self.store.rewind_session,
                session_id,
                request,
            )
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except MessageNotFoundError as error:
            raise ServiceError(404, "message_not_found", str(error)) from error
        except RevisionConflictError as error:
            raise ServiceError(
                409,
                "revision_conflict",
                str(error),
                details={"expected_revision": error.expected, "actual_revision": error.actual},
            ) from error
        except ResponseInProgressError as error:
            raise ServiceError(409, "response_in_progress", str(error), retryable=True) from error
        except StorageError as error:
            raise ServiceError(409, "session_state_conflict", str(error)) from error

    async def delete_session(self, session_id: UUID) -> None:
        try:
            await asyncio.to_thread(self.store.delete_session, session_id)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except ResponseInProgressError as error:
            raise ServiceError(409, "response_in_progress", str(error), retryable=True) from error
        await self.backend.close_session(session_id)

    async def runtime_instances(self) -> RuntimeInstanceList:
        if self.runtime_manager is not None:
            return await self.runtime_manager.instances()
        return RuntimeInstanceList(data=[])

    async def model_artifacts(self, *, refresh: bool = False) -> ModelArtifactList:
        if self.catalog is None:
            return ModelArtifactList(data=[])
        return await self.catalog.list(refresh=refresh)

    async def artifact_lineage(
        self, *, artifact_uri: str | None, limit: int
    ) -> ArtifactLineageList:
        data = await asyncio.to_thread(
            self.store.list_artifact_lineage,
            artifact_uri=artifact_uri,
            limit=limit,
        )
        return ArtifactLineageList(data=data)

    async def create_dataset(self, request: CreateDatasetRequest) -> DatasetResource:
        if self.tool_handlers is None:
            raise ServiceError(501, "workspace_unavailable", "workspace tools are unavailable")
        try:
            manifest = await asyncio.to_thread(
                self.tool_handlers.workspace_file_manifest,
                request.artifact_uri,
            )
            return await asyncio.to_thread(
                self.store.create_dataset,
                request,
                sha256=manifest["sha256"],
                byte_size=manifest["byte_size"],
            )
        except JobExecutionError as error:
            raise ServiceError(
                422, error.detail.code, error.detail.message, retryable=error.detail.retryable
            ) from error
        except StorageError as error:
            raise ServiceError(409, "dataset_conflict", str(error)) from error

    async def list_datasets(self) -> DatasetList:
        return DatasetList(data=await asyncio.to_thread(self.store.list_datasets))

    async def delete_dataset(self, dataset_id: UUID) -> None:
        try:
            await asyncio.to_thread(self.store.delete_dataset, dataset_id)
        except DatasetNotFoundError as error:
            raise ServiceError(404, "dataset_not_found", str(error)) from error

    async def evaluations(
        self,
        *,
        kind: EvaluationKind | None,
        model_id: str | None,
        limit: int,
    ) -> EvaluationResultList:
        return EvaluationResultList(
            data=await asyncio.to_thread(
                self.store.list_evaluations,
                kind=kind,
                model_id=model_id,
                limit=limit,
            )
        )

    async def compare_evaluations(
        self, request: CompareEvaluationsRequest
    ) -> EvaluationComparisonResource:
        try:
            evaluations = [
                await asyncio.to_thread(self.store.get_evaluation, identifier)
                for identifier in request.evaluation_ids
            ]
        except EvaluationNotFoundError as error:
            raise ServiceError(404, "evaluation_not_found", str(error)) from error
        keys = {item.comparison_key for item in evaluations}
        kinds = {item.kind for item in evaluations}
        if len(keys) != 1 or len(kinds) != 1:
            raise ServiceError(
                409,
                "evaluations_not_comparable",
                "evaluations must have matching kind, dataset, and execution parameters",
            )
        numeric_names = sorted(
            set.intersection(
                *[
                    {
                        key
                        for key, value in item.metrics.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    }
                    for item in evaluations
                ]
            )
        )
        baseline = evaluations[0]
        rows: list[EvaluationComparisonRow] = []
        for item in evaluations:
            deltas: dict[str, float | None] = {}
            ratios: dict[str, float | None] = {}
            for name in numeric_names:
                value = float(item.metrics[name])
                base = float(baseline.metrics[name])
                deltas[name] = value - base
                ratios[name] = value / base if base != 0.0 else None
            rows.append(EvaluationComparisonRow(evaluation=item, deltas=deltas, ratios=ratios))
        return EvaluationComparisonResource(
            comparison_key=baseline.comparison_key,
            baseline_id=baseline.id,
            metrics=numeric_names,
            rows=rows,
        )

    async def create_remote_node(self, request: CreateRemoteNodeRequest) -> RemoteNodeResource:
        try:
            resource = await asyncio.to_thread(self.store.create_remote_node, request)
        except StorageError as error:
            raise ServiceError(409, "remote_node_conflict", str(error)) from error
        if self.cluster is None:
            return resource
        nodes = await self.cluster.nodes(force=True)
        return next((item for item in nodes if item.id == resource.id), resource)

    async def remote_nodes(self, *, refresh: bool) -> RemoteNodeList:
        if self.cluster is None:
            return RemoteNodeList(data=await asyncio.to_thread(self.store.list_remote_nodes))
        return RemoteNodeList(data=await self.cluster.nodes(force=refresh))

    async def update_remote_node(
        self, node_id: UUID, request: UpdateRemoteNodeRequest
    ) -> RemoteNodeResource:
        try:
            resource = await asyncio.to_thread(self.store.update_remote_node, node_id, request)
        except RemoteNodeNotFoundError as error:
            raise ServiceError(404, "remote_node_not_found", str(error)) from error
        except StorageError as error:
            raise ServiceError(409, "remote_node_conflict", str(error)) from error
        if self.cluster is None:
            return resource
        nodes = await self.cluster.nodes(force=True)
        return next((item for item in nodes if item.id == resource.id), resource)

    async def delete_remote_node(self, node_id: UUID) -> None:
        try:
            await asyncio.to_thread(self.store.delete_remote_node, node_id)
        except RemoteNodeNotFoundError as error:
            raise ServiceError(404, "remote_node_not_found", str(error)) from error
        if self.cluster is not None:
            await self.cluster.nodes(force=True)

    async def runtime_capabilities(self) -> RuntimeCapabilitiesResource:
        try:
            return await self.backend.capabilities()
        except BackendError as error:
            raise ServiceError(
                error.status_code
                if error.status_code in {400, 404, 409, 413, 415, 422, 501, 502, 503}
                else 503,
                error.code,
                str(error),
                retryable=error.retryable,
            ) from error

    async def runtime_status(self) -> dict[str, Any]:
        status = await self._runtime_request("runtime_status")
        instance_id = status.get("instance_id")
        parsed_instance_id = None
        if isinstance(instance_id, str):
            try:
                parsed_instance_id = UUID(instance_id)
            except ValueError:
                parsed_instance_id = None
        await asyncio.to_thread(
            self.store.append_runtime_metric,
            status,
            instance_id=parsed_instance_id,
            model=str(status.get("model")) if status.get("model") is not None else None,
        )
        return status

    async def runtime_metrics(
        self,
        *,
        instance_id: UUID | None,
        since: datetime | None,
        limit: int,
    ) -> RuntimeMetricList:
        data = await asyncio.to_thread(
            self.store.list_runtime_metrics,
            instance_id=instance_id,
            since=since,
            limit=limit,
        )
        return RuntimeMetricList(data=data)

    async def runtime_logs(
        self,
        *,
        instance_id: UUID | None,
        level: RuntimeLogLevel | None,
        after: int,
        limit: int,
    ) -> RuntimeLogList:
        data = await asyncio.to_thread(
            self.store.list_runtime_logs,
            instance_id=instance_id,
            level=level,
            after=after,
            limit=limit,
        )
        return RuntimeLogList(data=data)

    async def runtime_models(self) -> dict[str, Any]:
        return await self._runtime_request("runtime_models")

    async def realtime_capabilities(self) -> dict[str, Any]:
        return await self._runtime_request("realtime_capabilities")

    async def reload_runtime(self, context_size: int) -> dict[str, Any]:
        return await self._runtime_request("reload_runtime", context_size)

    async def clear_runtime_cache(self) -> dict[str, Any]:
        return await self._runtime_request("clear_runtime_cache")

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        connector = getattr(self.backend, "realtime_connect", None)
        if connector is None:
            raise ServiceError(
                501,
                "realtime_unavailable",
                "the configured backend does not expose realtime transport",
            )
        return connector(mode=mode)

    async def load_model(self, request: ModelLoadRequest) -> OperationAccepted:
        if self.runtime_manager is None:
            raise ServiceError(
                501, "model_management_unavailable", "model loading is not available"
            )
        job = await self.create_job(
            CreateJobRequest(kind="model.load", payload=request.model_dump(mode="json"))
        )
        return OperationAccepted(operation_id=job.id)

    async def unload_model(self, request: ModelUnloadRequest) -> OperationAccepted:
        if self.runtime_manager is None:
            raise ServiceError(
                501, "model_management_unavailable", "model unloading is not available"
            )
        job = await self.create_job(
            CreateJobRequest(kind="model.unload", payload=request.model_dump(mode="json"))
        )
        return OperationAccepted(operation_id=job.id)

    async def prepare_response(
        self,
        session_id: UUID,
        request: CreateResponseRequest,
    ) -> PreparedResponse:
        try:
            history = await asyncio.to_thread(self.store.list_messages, session_id)
            await asyncio.to_thread(
                self._validate_media_parts,
                [part for message in history for part in message.parts],
            )
            await asyncio.to_thread(self._validate_media_parts, request.input)
            backend_messages = tuple(
                self._message_to_backend(
                    message,
                    include_reasoning=request.include_reasoning_history,
                )
                for message in history
            )
            if request.system_prompt and request.system_prompt.strip():
                backend_messages = (
                    {"role": "system", "content": request.system_prompt.strip()},
                    *backend_messages,
                )
            input_role = MessageRole(request.input_role)
            backend_input = self._parts_to_backend(input_role, request.input)
            fingerprint = self._request_fingerprint(request)
            begin = await asyncio.to_thread(
                self.store.begin_response,
                session_id,
                request.request_id,
                uuid4(),
                fingerprint,
                request.expected_revision,
                request.input,
                ResponseRequestSettings(
                    sampling=request.sampling,
                    system_prompt=request.system_prompt,
                    include_reasoning_history=request.include_reasoning_history,
                    input_role=request.input_role,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    response_format=request.response_format,
                ),
                input_role,
            )
            if begin.started and begin.session.title is None:
                title = self._title_from_parts(request.input)
                if title:
                    session = await asyncio.to_thread(
                        self.store.update_session,
                        session_id,
                        UpdateSessionRequest(title=title),
                    )
                    begin = BeginResponseResult(
                        session=session,
                        response=begin.response,
                        started=True,
                    )
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except RevisionConflictError as error:
            raise ServiceError(
                409,
                "revision_conflict",
                str(error),
                details={"expected_revision": error.expected, "actual_revision": error.actual},
            ) from error
        except IdempotencyConflictError as error:
            raise ServiceError(409, "idempotency_conflict", str(error)) from error
        except ResponseInProgressError as error:
            raise ServiceError(409, "response_in_progress", str(error), retryable=True) from error
        except StorageError as error:
            raise ServiceError(409, "session_state_conflict", str(error)) from error

        if not begin.started and begin.response.status == ResponseStatus.RUNNING:
            raise ServiceError(
                409,
                "response_in_progress",
                f"request {request.request_id} is already running",
                retryable=True,
            )
        if begin.started:
            backend_messages = (*backend_messages, backend_input)
        return PreparedResponse(
            request=request,
            begin=begin,
            backend_messages=tuple(backend_messages),
        )

    async def collect_response(self, prepared: PreparedResponse) -> ResponseResource:
        if not prepared.begin.started:
            return prepared.begin.response
        accumulator = _OutputAccumulator()
        try:
            async for delta in self.backend.stream(
                model=prepared.begin.session.model,
                messages=prepared.backend_messages,
                sampling=prepared.request.sampling,
                session_id=prepared.begin.session.id,
                tools=prepared.request.tools,
                tool_choice=prepared.request.tool_choice,
                response_format=prepared.request.response_format,
            ):
                accumulator.apply(delta)
            finish_reason = self._require_finish_reason(accumulator)
            return await asyncio.to_thread(
                self.store.complete_response,
                prepared.begin.response.id,
                accumulator.output_parts(),
                finish_reason,
                accumulator.usage,
                accumulator.performance,
            )
        except asyncio.CancelledError:
            detail = ErrorDetail(
                code="client_cancelled",
                message="client disconnected before the response completed",
            )
            await asyncio.to_thread(
                self.store.terminate_response,
                prepared.begin.response.id,
                detail,
                cancelled=True,
            )
            raise
        except BackendError as error:
            await self._terminate_backend_failure(prepared.begin.response.id, error)
            raise ServiceError(
                502,
                error.code,
                str(error),
                retryable=error.retryable,
            ) from error

    async def list_responses(self, session_id: UUID, *, limit: int = 200) -> ResponseList:
        try:
            data = await asyncio.to_thread(self.store.list_responses, session_id, limit=limit)
        except ValueError as error:
            raise ServiceError(422, "invalid_request", str(error)) from error
        return ResponseList(data=data)

    async def stream_response(self, prepared: PreparedResponse) -> AsyncIterator[str]:
        sequence = 0
        if not prepared.begin.started:
            async for event in self._replay_response(prepared.begin.response):
                yield self._encode_sse(
                    event,
                    sequence,
                    session_id=prepared.begin.response.session_id,
                )
                sequence += 1
            session = await asyncio.to_thread(
                self.store.get_session,
                prepared.begin.response.session_id,
            )
            yield self._encode_sse(
                SessionStateChanged(state=session.state, revision=session.revision),
                sequence,
                session_id=session.id,
            )
            return

        yield self._encode_sse(
            SessionStateChanged(
                state=SessionState.PROCESSING,
                revision=prepared.begin.session.revision,
            ),
            sequence,
            session_id=prepared.begin.session.id,
        )
        sequence += 1
        accumulator = _OutputAccumulator()
        try:
            async for delta in self.backend.stream(
                model=prepared.begin.session.model,
                messages=prepared.backend_messages,
                sampling=prepared.request.sampling,
                session_id=prepared.begin.session.id,
                tools=prepared.request.tools,
                tool_choice=prepared.request.tool_choice,
                response_format=prepared.request.response_format,
            ):
                accumulator.apply(delta)
                payloads = self._delta_payloads(prepared.begin.response.id, delta)
                for payload in payloads:
                    yield self._encode_sse(
                        payload,
                        sequence,
                        session_id=prepared.begin.session.id,
                    )
                    sequence += 1
            finish_reason = self._require_finish_reason(accumulator)
            completed = await asyncio.to_thread(
                self.store.complete_response,
                prepared.begin.response.id,
                accumulator.output_parts(),
                finish_reason,
                accumulator.usage,
                accumulator.performance,
            )
            yield self._encode_sse(
                ResponseCompleted(
                    response_id=completed.id,
                    finish_reason=finish_reason,
                    usage=completed.usage,
                    performance=completed.performance,
                ),
                sequence,
                session_id=completed.session_id,
            )
            sequence += 1
            session = await asyncio.to_thread(self.store.get_session, completed.session_id)
            yield self._encode_sse(
                SessionStateChanged(state=session.state, revision=session.revision),
                sequence,
                session_id=session.id,
            )
        except asyncio.CancelledError:
            detail = ErrorDetail(
                code="client_cancelled",
                message="client disconnected before the response completed",
            )
            await asyncio.to_thread(
                self.store.terminate_response,
                prepared.begin.response.id,
                detail,
                cancelled=True,
            )
            raise
        except BackendError as error:
            failed = await self._terminate_backend_failure(prepared.begin.response.id, error)
            yield self._encode_sse(
                ErrorEvent(error=failed.error or ErrorDetail(code=error.code, message=str(error))),
                sequence,
                session_id=failed.session_id,
            )
            sequence += 1
            session = await asyncio.to_thread(self.store.get_session, failed.session_id)
            yield self._encode_sse(
                SessionStateChanged(state=session.state, revision=session.revision),
                sequence,
                session_id=session.id,
            )

    async def _terminate_backend_failure(
        self,
        response_id: UUID,
        error: BackendError,
    ) -> ResponseResource:
        detail = ErrorDetail(
            code=error.code,
            message=str(error),
            retryable=error.retryable,
        )
        return await asyncio.to_thread(self.store.terminate_response, response_id, detail)

    async def _replay_response(self, response: ResponseResource) -> AsyncIterator[RealtimePayload]:
        if response.status == ResponseStatus.COMPLETED:
            tool_index = 0
            for part in response.output:
                if isinstance(part, ReasoningPart):
                    yield ResponseReasoningDelta(response_id=response.id, delta=part.text)
                elif isinstance(part, TextPart):
                    yield ResponseTextDelta(response_id=response.id, delta=part.text)
                elif isinstance(part, ToolCallPart):
                    yield ResponseToolCallDelta(
                        response_id=response.id,
                        index=tool_index,
                        call_id=part.call_id,
                        name=part.name,
                        arguments_delta=json.dumps(part.arguments, separators=(",", ":")),
                    )
                    tool_index += 1
            yield ResponseCompleted(
                response_id=response.id,
                finish_reason=response.finish_reason or "stop",
                usage=response.usage,
                performance=response.performance,
            )
        elif response.error is not None:
            yield ErrorEvent(error=response.error)

    @staticmethod
    def _delta_payloads(response_id: UUID, delta: BackendDelta) -> list[RealtimePayload]:
        payloads: list[RealtimePayload] = []
        if delta.reasoning_delta:
            payloads.append(
                ResponseReasoningDelta(response_id=response_id, delta=delta.reasoning_delta)
            )
        if delta.content_delta:
            payloads.append(ResponseTextDelta(response_id=response_id, delta=delta.content_delta))
        for tool in delta.tool_calls:
            payloads.append(
                ResponseToolCallDelta(
                    response_id=response_id,
                    index=tool.index,
                    call_id=tool.call_id,
                    name=tool.name,
                    arguments_delta=tool.arguments_delta,
                )
            )
        return payloads

    @staticmethod
    def _request_fingerprint(request: CreateResponseRequest) -> str:
        value = request.model_dump(mode="json", exclude={"stream"})
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_media_parts(self, parts: Sequence[ContentPart]) -> None:
        for part in parts:
            if isinstance(part, DocumentPart):
                try:
                    document = self.store.get_document(part.media.id)
                except MediaNotFoundError as error:
                    raise ServiceError(422, "document_not_found", str(error)) from error
                if document.media != part.media or document.name != part.name:
                    raise ServiceError(422, "document_reference_mismatch", part.name)
                continue
            if not isinstance(part, (ImagePart, VideoPart, AudioPart, GeneratedAudioPart)):
                continue
            try:
                stored = self.store.get_media(part.media.id)
            except MediaNotFoundError as error:
                raise ServiceError(422, "media_not_found", str(error)) from error
            if stored.media != part.media:
                raise ServiceError(
                    422,
                    "media_reference_mismatch",
                    f"media reference does not match stored object {part.media.id}",
                )
            expected_prefix = {
                "image": "image/",
                "video": "video/",
                "audio": "audio/",
                "generated_audio": "audio/",
            }[part.type]
            if not part.media.mime_type.startswith(expected_prefix):
                raise ServiceError(
                    422,
                    "media_type_mismatch",
                    f"{part.type} requires a {expected_prefix[:-1]} MIME type",
                )

    def _message_to_backend(
        self,
        message: Message,
        *,
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        return self._parts_to_backend(
            message.role,
            message.parts,
            include_reasoning=include_reasoning,
        )

    def _parts_to_backend(
        self,
        role: MessageRole,
        parts: Sequence[ContentPart],
        *,
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        if role == MessageRole.TOOL:
            if len(parts) != 1 or not isinstance(parts[0], ToolResultPart):
                raise ServiceError(
                    422,
                    "unsupported_tool_message",
                    "a tool message must contain exactly one tool_result part",
                )
            result = parts[0]
            content = (
                result.result
                if isinstance(result.result, str)
                else json.dumps(result.result, separators=(",", ":"), ensure_ascii=False)
            )
            return {"role": "tool", "tool_call_id": result.call_id, "content": content}

        content_fragments: list[str] = []
        content_items: list[dict[str, Any]] = []
        reasoning_fragments: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, (TextPart, TranscriptPart)):
                content_fragments.append(part.text)
                content_items.append({"type": "text", "text": part.text})
            elif isinstance(part, ReasoningPart):
                if role != MessageRole.ASSISTANT:
                    raise ServiceError(
                        422,
                        "unsupported_reasoning_part",
                        "reasoning parts are valid only on assistant messages",
                    )
                if include_reasoning:
                    reasoning_fragments.append(part.text)
            elif isinstance(part, ToolCallPart):
                if role != MessageRole.ASSISTANT:
                    raise ServiceError(
                        422,
                        "unsupported_tool_call_part",
                        "tool_call parts are valid only on assistant messages",
                    )
                tool_calls.append(
                    {
                        "id": part.call_id,
                        "type": "function",
                        "function": {
                            "name": part.name,
                            "arguments": json.dumps(
                                part.arguments,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            elif isinstance(part, (ImagePart, VideoPart, AudioPart, GeneratedAudioPart)):
                content_items.append(self._media_to_backend(part))
            elif isinstance(part, DocumentPart):
                try:
                    document = self.store.get_document(part.media.id)
                except MediaNotFoundError as error:
                    raise ServiceError(422, "document_not_found", str(error)) from error
                content_fragments.append(
                    f'<document name="{document.name}">\n{document.text}\n</document>\n\n'
                )
                content_items.append(
                    {
                        "type": "text",
                        "text": f'<document name="{document.name}">\n{document.text}\n</document>\n\n',
                    }
                )
            elif isinstance(part, ToolResultPart):
                raise ServiceError(
                    422,
                    "unsupported_tool_result_part",
                    "tool_result parts require a tool-role message",
                )
        has_media = any(
            isinstance(part, (ImagePart, VideoPart, AudioPart, GeneratedAudioPart))
            for part in parts
        )
        message: dict[str, Any] = {
            "role": role.value,
            "content": content_items if has_media else "".join(content_fragments),
        }
        if reasoning_fragments:
            message["reasoning_content"] = "".join(reasoning_fragments)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def _media_to_backend(
        self,
        part: ImagePart | VideoPart | AudioPart | GeneratedAudioPart,
    ) -> dict[str, Any]:
        try:
            resource, path = self.store.get_media_path(part.media.id)
        except MediaNotFoundError as error:
            raise ServiceError(422, "media_not_found", str(error)) from error
        if resource.media != part.media:
            raise ServiceError(
                422,
                "media_reference_mismatch",
                f"media reference does not match stored object {part.media.id}",
            )
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        if isinstance(part, ImagePart):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{part.media.mime_type};base64,{encoded}",
                },
            }
        if isinstance(part, VideoPart):
            return {
                "type": "video_url",
                "video_url": {
                    "url": f"data:{part.media.mime_type};base64,{encoded}",
                },
            }
        subtype = part.media.mime_type.split("/", 1)[-1]
        audio_format = {
            "mpeg": "mp3",
            "x-m4a": "m4a",
            "mp4": "m4a",
            "x-wav": "wav",
        }.get(subtype, subtype)
        return {
            "type": "input_audio",
            "input_audio": {"data": encoded, "format": audio_format},
        }

    async def _runtime_request(self, method: str, *args: Any) -> dict[str, Any]:
        operation = getattr(self.backend, method, None)
        if operation is None:
            raise ServiceError(
                501,
                "runtime_control_unavailable",
                f"the configured backend does not implement {method}",
            )
        try:
            return await operation(*args)
        except BackendError as error:
            raise ServiceError(
                503,
                error.code,
                str(error),
                retryable=error.retryable,
            ) from error

    @staticmethod
    def _title_from_parts(parts: Sequence[ContentPart]) -> str | None:
        text = " ".join(
            part.text.strip()
            for part in parts
            if isinstance(part, (TextPart, TranscriptPart)) and part.text.strip()
        )
        normalized = " ".join(text.split())
        if not normalized:
            return None
        return normalized if len(normalized) <= 60 else f"{normalized[:59]}…"

    @staticmethod
    def _require_finish_reason(accumulator: _OutputAccumulator) -> str:
        if accumulator.finish_reason is None:
            raise BackendError(
                "backend_protocol_error",
                "backend stream completed without a finish reason",
            )
        return accumulator.finish_reason

    @staticmethod
    def _encode_sse(
        payload: RealtimePayload,
        sequence: int,
        *,
        session_id: UUID | None = None,
    ) -> str:
        if session_id is None:
            raise ValueError("session_id is required for SSE payloads")
        frame = RealtimeFrame(
            session_id=session_id,
            sequence=sequence,
            timestamp=datetime.now(timezone.utc),
            payload=payload,
        )
        event_type = frame.payload.type
        return f"event: {event_type}\nid: {sequence}\ndata: {frame.model_dump_json()}\n\n"
