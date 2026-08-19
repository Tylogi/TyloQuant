"""FastAPI application for the public ``mfq serve`` API."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import Body, FastAPI, Header, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mfq.server.auth import ApiKeyManager, required_scope
from mfq.server.models import (
    SHA256_PATTERN,
    ApiKeyList,
    ApiKeyResource,
    ApiKeySecretResource,
    AppendMessageRequest,
    AppendMessageResult,
    ArtifactLineageList,
    CompareEvaluationsRequest,
    CreateApiKeyRequest,
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
    DocumentResource,
    ErrorDetail,
    ErrorResponse,
    EvaluationComparisonResource,
    EvaluationKind,
    EvaluationResultList,
    ForkSessionRequest,
    GenerationPresetList,
    GenerationPresetResource,
    HubModelInfo,
    HubModelSearchResult,
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
    MediaResource,
    MessageList,
    ModelArtifactList,
    ModelLoadRequest,
    ModelUnloadRequest,
    OperationAccepted,
    RemoteNodeList,
    RemoteNodeResource,
    ResponseList,
    ResponseResource,
    RewindSessionRequest,
    RuntimeCapabilitiesResource,
    RuntimeInstanceList,
    RuntimeLogLevel,
    RuntimeLogList,
    RuntimeMetricList,
    RuntimeProfileList,
    RuntimeProfileLoadRequest,
    RuntimeProfileResource,
    RuntimeReloadRequest,
    SessionArchive,
    SessionImportResult,
    SessionList,
    SessionResource,
    UpdateGenerationPresetRequest,
    UpdateMcpServerRequest,
    UpdateRemoteNodeRequest,
    UpdateRuntimeProfileRequest,
    UpdateSessionRequest,
)
from mfq.server.service import ServerService, ServiceError
from mfq.server.storage import ApiKeyNotFoundError, StorageError

ERROR_RESPONSES = {
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    501: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def create_app(
    service: ServerService | None = None,
    *,
    web_root: str | Path | None = None,
    api_key: str = "",
    api_keys: ApiKeyManager | None = None,
) -> FastAPI:
    """Create an executable app, or a route-only contract app when service is omitted."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            if service is not None:
                await service.start()
            yield
        finally:
            if service is not None:
                await service.aclose()

    app = FastAPI(
        title="MFQ Server API",
        summary="Persistent sessions, media, model workers, and realtime inference",
        version="1.0.0",
        openapi_version="3.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type", "X-Content-SHA256"],
    )

    @app.middleware("http")
    async def protect_and_harden(request: Request, call_next: Any) -> Response:
        if (api_key or api_keys is not None) and request.url.path.startswith("/api/"):
            authorization = request.headers.get("authorization", "")
            supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
            authenticated = api_keys.authenticate(supplied) if api_keys is not None else None
            legacy = api_key and secrets.compare_digest(supplied, api_key)
            scope = required_scope(request.method, request.url.path)
            if not legacy and authenticated is None:
                detail = ErrorDetail(code="unauthorized", message="invalid API credential")
                return JSONResponse(
                    status_code=401,
                    content=ErrorResponse(error=detail).model_dump(mode="json"),
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if (
                not legacy
                and authenticated is not None
                and api_keys is not None
                and not api_keys.permits(authenticated, scope)
            ):
                detail = ErrorDetail(
                    code="insufficient_scope",
                    message=f"API credential requires the {scope} scope",
                )
                return JSONResponse(
                    status_code=403,
                    content=ErrorResponse(error=detail).model_dump(mode="json"),
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; connect-src 'self' ws: wss:; img-src 'self' data: blob:; "
            "media-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; font-src 'self' data:"
        )
        return response

    async def authorize_websocket(websocket: WebSocket) -> bool:
        if not api_key and api_keys is None:
            return True
        authorization = websocket.headers.get("authorization", "")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        if not supplied:
            supplied = websocket.query_params.get("access_token", "")
        authenticated = api_keys.authenticate(supplied) if api_keys is not None else None
        if (api_key and secrets.compare_digest(supplied, api_key)) or (
            authenticated is not None and api_keys and api_keys.permits(authenticated, "inference")
        ):
            return True
        await websocket.close(code=1008, reason="invalid API credential")
        return False

    def require_service() -> ServerService:
        if service is None:
            raise ServiceError(
                501,
                "contract_only",
                "the protocol contract application does not execute requests",
            )
        return service

    def require_api_keys() -> ApiKeyManager:
        if api_keys is None:
            raise ServiceError(
                501,
                "api_key_management_unavailable",
                "scoped API key management requires a configured root credential",
            )
        return api_keys

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "mfq-server",
            "protocol_version": "1.0",
        }

    @app.post(
        "/api/v1/auth/keys",
        response_model=ApiKeySecretResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["auth"],
    )
    async def create_api_key(body: CreateApiKeyRequest) -> ApiKeySecretResource:
        try:
            return await asyncio.to_thread(require_api_keys().create, body)
        except StorageError as error:
            raise ServiceError(409, "api_key_conflict", str(error)) from error

    @app.get(
        "/api/v1/auth/keys",
        response_model=ApiKeyList,
        responses=ERROR_RESPONSES,
        tags=["auth"],
    )
    async def list_api_keys() -> ApiKeyList:
        data = await asyncio.to_thread(require_api_keys().store.list_api_keys)
        return ApiKeyList(data=data)

    @app.post(
        "/api/v1/auth/keys/{key_id}/revoke",
        response_model=ApiKeyResource,
        responses=ERROR_RESPONSES,
        tags=["auth"],
    )
    async def revoke_api_key(key_id: UUID) -> ApiKeyResource:
        try:
            return await asyncio.to_thread(require_api_keys().store.revoke_api_key, key_id)
        except ApiKeyNotFoundError as error:
            raise ServiceError(404, "api_key_not_found", str(error)) from error

    @app.post(
        "/api/v1/auth/keys/{key_id}/rotate",
        response_model=ApiKeySecretResource,
        responses=ERROR_RESPONSES,
        tags=["auth"],
    )
    async def rotate_api_key(key_id: UUID) -> ApiKeySecretResource:
        try:
            return await asyncio.to_thread(require_api_keys().rotate, key_id)
        except ApiKeyNotFoundError as error:
            raise ServiceError(404, "api_key_not_found", str(error)) from error

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Any, error: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorResponse(error=error.detail).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Any, error: RequestValidationError) -> JSONResponse:
        detail = ErrorDetail(
            code="invalid_request",
            message="request validation failed",
            details={"errors": jsonable_encoder(error.errors())},
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(error=detail).model_dump(mode="json"),
        )

    @app.post(
        "/api/v1/sessions",
        response_model=SessionResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def create_session(body: CreateSessionRequest) -> SessionResource:
        return await require_service().create_session(body)

    @app.post(
        "/api/v1/sessions/import",
        response_model=SessionImportResult,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def import_session(body: SessionArchive) -> SessionImportResult:
        return await require_service().import_session(body)

    @app.post(
        "/api/v1/presets",
        response_model=GenerationPresetResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["presets"],
    )
    async def create_generation_preset(
        body: CreateGenerationPresetRequest,
    ) -> GenerationPresetResource:
        return await require_service().create_generation_preset(body)

    @app.get(
        "/api/v1/presets",
        response_model=GenerationPresetList,
        responses=ERROR_RESPONSES,
        tags=["presets"],
    )
    async def list_generation_presets() -> GenerationPresetList:
        return await require_service().list_generation_presets()

    @app.put(
        "/api/v1/presets/{preset_id}",
        response_model=GenerationPresetResource,
        responses=ERROR_RESPONSES,
        tags=["presets"],
    )
    async def update_generation_preset(
        preset_id: UUID, body: UpdateGenerationPresetRequest
    ) -> GenerationPresetResource:
        return await require_service().update_generation_preset(preset_id, body)

    @app.delete(
        "/api/v1/presets/{preset_id}",
        status_code=204,
        responses=ERROR_RESPONSES,
        tags=["presets"],
    )
    async def delete_generation_preset(preset_id: UUID) -> Response:
        await require_service().delete_generation_preset(preset_id)
        return Response(status_code=204)

    @app.post(
        "/api/v1/runtime/profiles",
        response_model=RuntimeProfileResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def create_runtime_profile(
        body: CreateRuntimeProfileRequest,
    ) -> RuntimeProfileResource:
        return await require_service().create_runtime_profile(body)

    @app.get(
        "/api/v1/runtime/profiles",
        response_model=RuntimeProfileList,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def list_runtime_profiles() -> RuntimeProfileList:
        return await require_service().list_runtime_profiles()

    @app.get(
        "/api/v1/runtime/profiles/{profile_id}",
        response_model=RuntimeProfileResource,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def get_runtime_profile(profile_id: UUID) -> RuntimeProfileResource:
        return await require_service().get_runtime_profile(profile_id)

    @app.put(
        "/api/v1/runtime/profiles/{profile_id}",
        response_model=RuntimeProfileResource,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def update_runtime_profile(
        profile_id: UUID, body: UpdateRuntimeProfileRequest
    ) -> RuntimeProfileResource:
        return await require_service().update_runtime_profile(profile_id, body)

    @app.delete(
        "/api/v1/runtime/profiles/{profile_id}",
        status_code=204,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def delete_runtime_profile(profile_id: UUID) -> Response:
        await require_service().delete_runtime_profile(profile_id)
        return Response(status_code=204)

    @app.post(
        "/api/v1/runtime/profiles/{profile_id}/load",
        response_model=OperationAccepted,
        status_code=202,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def load_runtime_profile(
        profile_id: UUID, body: RuntimeProfileLoadRequest
    ) -> OperationAccepted:
        return await require_service().load_runtime_profile(profile_id, body)

    @app.post(
        "/api/v1/jobs",
        response_model=JobResource,
        status_code=202,
        responses=ERROR_RESPONSES,
        tags=["jobs"],
    )
    async def create_job(body: CreateJobRequest) -> JobResource:
        return await require_service().create_job(body)

    @app.get(
        "/api/v1/jobs/kinds",
        response_model=JobKindList,
        responses=ERROR_RESPONSES,
        tags=["jobs"],
    )
    async def job_kinds() -> JobKindList:
        return await require_service().job_kinds()

    @app.get(
        "/api/v1/jobs",
        response_model=JobList,
        responses=ERROR_RESPONSES,
        tags=["jobs"],
    )
    async def list_jobs(
        status: JobStatus | None = None,
        kind: Annotated[str | None, Query(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> JobList:
        return await require_service().list_jobs(
            status=status,
            kind=kind,
            limit=limit,
            offset=offset,
        )

    @app.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobResource,
        responses=ERROR_RESPONSES,
        tags=["jobs"],
    )
    async def get_job(job_id: UUID) -> JobResource:
        return await require_service().get_job(job_id)

    @app.post(
        "/api/v1/jobs/{job_id}/cancel",
        response_model=JobResource,
        responses=ERROR_RESPONSES,
        tags=["jobs"],
    )
    async def cancel_job(job_id: UUID) -> JobResource:
        return await require_service().cancel_job(job_id)

    @app.post(
        "/api/v1/jobs/{job_id}/retry",
        response_model=JobResource,
        status_code=202,
        responses=ERROR_RESPONSES,
        tags=["jobs"],
    )
    async def retry_job(job_id: UUID) -> JobResource:
        return await require_service().retry_job(job_id)

    @app.get(
        "/api/v1/hub/models",
        response_model=HubModelSearchResult,
        responses=ERROR_RESPONSES,
        tags=["models"],
    )
    async def search_hub_models(
        provider: Annotated[str, Query(pattern="^(huggingface|modelscope)$")],
        query: Annotated[str, Query(min_length=1, max_length=255)],
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> HubModelSearchResult:
        return await require_service().search_hub_models(provider, query, limit)

    @app.get(
        "/api/v1/hub/models/{provider}/{owner}/{name}",
        response_model=HubModelInfo,
        responses=ERROR_RESPONSES,
        tags=["models"],
    )
    async def hub_model_info(
        provider: str,
        owner: str,
        name: str,
        revision: Annotated[str | None, Query(max_length=255)] = None,
    ) -> HubModelInfo:
        if provider not in {"huggingface", "modelscope"}:
            raise ServiceError(404, "model_hub_not_found", provider)
        return await require_service().hub_model_info(provider, f"{owner}/{name}", revision)

    @app.get(
        "/api/v1/jobs/{job_id}/events",
        response_model=JobEventList,
        responses=ERROR_RESPONSES,
        tags=["jobs"],
    )
    async def list_job_events(
        job_id: UUID,
        after: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> JobEventList:
        return await require_service().list_job_events(job_id, after=after, limit=limit)

    @app.get(
        "/api/v1/jobs/{job_id}/events/stream",
        responses=ERROR_RESPONSES,
        tags=["jobs"],
    )
    async def stream_job_events(
        job_id: UUID,
        after: Annotated[int, Query(ge=0)] = 0,
        last_event_id: Annotated[int | None, Header(alias="Last-Event-ID", ge=0)] = None,
    ) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            cursor = max(after, last_event_id or 0)
            while True:
                result = await require_service().list_job_events(
                    job_id,
                    after=cursor,
                    limit=200,
                )
                for event in result.data:
                    cursor = event.sequence
                    payload = event.model_dump_json()
                    yield f"id: {event.sequence}\nevent: {event.type.value}\ndata: {payload}\n\n"
                job = await require_service().get_job(job_id)
                if job.status in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                    JobStatus.INTERRUPTED,
                }:
                    return
                if not result.data:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)

        await require_service().get_job(job_id)
        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(
        "/api/v1/sessions",
        response_model=SessionList,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def list_sessions(
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> SessionList:
        return await require_service().list_sessions(limit=limit, offset=offset)

    @app.get(
        "/api/v1/sessions/{session_id}",
        response_model=SessionResource,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def get_session(session_id: UUID) -> SessionResource:
        return await require_service().get_session(session_id)

    @app.patch(
        "/api/v1/sessions/{session_id}",
        response_model=SessionResource,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def update_session(
        session_id: UUID,
        body: UpdateSessionRequest,
    ) -> SessionResource:
        return await require_service().update_session(session_id, body)

    @app.get(
        "/api/v1/sessions/{session_id}/messages",
        response_model=MessageList,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def list_messages(session_id: UUID) -> MessageList:
        return await require_service().list_messages(session_id)

    @app.get(
        "/api/v1/sessions/{session_id}/export",
        response_model=SessionArchive,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def export_session(session_id: UUID) -> SessionArchive:
        return await require_service().export_session(session_id)

    @app.post(
        "/api/v1/sessions/{session_id}/messages",
        response_model=AppendMessageResult,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def append_message(
        session_id: UUID,
        body: AppendMessageRequest,
    ) -> AppendMessageResult:
        return await require_service().append_message(session_id, body)

    @app.get(
        "/api/v1/sessions/{session_id}/responses",
        response_model=ResponseList,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def list_responses(
        session_id: UUID,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> ResponseList:
        return await require_service().list_responses(session_id, limit=limit)

    @app.post(
        "/api/v1/sessions/{session_id}/responses",
        response_model=ResponseResource,
        responses={
            **ERROR_RESPONSES,
            200: {
                "description": "Completed response or an SSE event stream",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ResponseResource"}
                    },
                    "text/event-stream": {"schema": {"type": "string"}},
                },
            },
        },
        tags=["sessions"],
    )
    async def create_response(session_id: UUID, body: CreateResponseRequest) -> Any:
        daemon = require_service()
        prepared = await daemon.prepare_response(session_id, body)
        if body.stream:
            return StreamingResponse(
                daemon.stream_response(prepared),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return await daemon.collect_response(prepared)

    @app.post(
        "/api/v1/sessions/{session_id}/fork",
        response_model=SessionResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def fork_session(session_id: UUID, body: ForkSessionRequest) -> SessionResource:
        return await require_service().fork_session(session_id, body)

    @app.post(
        "/api/v1/sessions/{session_id}/rewind",
        response_model=SessionResource,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def rewind_session(
        session_id: UUID,
        body: RewindSessionRequest,
    ) -> SessionResource:
        return await require_service().rewind_session(session_id, body)

    @app.delete(
        "/api/v1/sessions/{session_id}",
        status_code=204,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def delete_session(session_id: UUID) -> Response:
        await require_service().delete_session(session_id)
        return Response(status_code=204)

    @app.post(
        "/api/v1/media",
        response_model=MediaResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["media"],
    )
    async def upload_media(
        body: Annotated[bytes, Body(media_type="application/octet-stream")],
        content_type: Annotated[str, Header(alias="Content-Type")],
        content_sha256: Annotated[
            str,
            Header(alias="X-Content-SHA256", pattern=SHA256_PATTERN),
        ],
    ) -> MediaResource:
        return await require_service().upload_media(body, content_type, content_sha256)

    @app.get(
        "/api/v1/media/{media_id}",
        responses=ERROR_RESPONSES,
        tags=["media"],
    )
    async def get_media(media_id: UUID) -> FileResponse:
        resource, path = await require_service().get_media(media_id)
        return FileResponse(
            path,
            media_type=resource.media.mime_type,
            headers={
                "Cache-Control": "private, immutable, max-age=31536000",
                "ETag": f'"{resource.media.sha256}"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post(
        "/api/v1/documents",
        response_model=DocumentResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["media"],
    )
    async def create_document(body: CreateDocumentRequest) -> DocumentResource:
        return await require_service().create_document(body)

    @app.get(
        "/api/v1/documents/{media_id}",
        response_model=DocumentResource,
        responses=ERROR_RESPONSES,
        tags=["media"],
    )
    async def get_document(media_id: UUID) -> DocumentResource:
        return await require_service().get_document(media_id)

    @app.post(
        "/api/v1/mcp/servers",
        response_model=McpServerResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["mcp"],
    )
    async def create_mcp_server(body: CreateMcpServerRequest) -> McpServerResource:
        return await require_service().create_mcp_server(body)

    @app.get(
        "/api/v1/mcp/servers",
        response_model=McpServerList,
        responses=ERROR_RESPONSES,
        tags=["mcp"],
    )
    async def list_mcp_servers() -> McpServerList:
        return await require_service().list_mcp_servers()

    @app.patch(
        "/api/v1/mcp/servers/{server_id}",
        response_model=McpServerResource,
        responses=ERROR_RESPONSES,
        tags=["mcp"],
    )
    async def update_mcp_server(server_id: UUID, body: UpdateMcpServerRequest) -> McpServerResource:
        return await require_service().update_mcp_server(server_id, body)

    @app.delete(
        "/api/v1/mcp/servers/{server_id}",
        status_code=204,
        responses=ERROR_RESPONSES,
        tags=["mcp"],
    )
    async def delete_mcp_server(server_id: UUID) -> Response:
        await require_service().delete_mcp_server(server_id)
        return Response(status_code=204)

    @app.get(
        "/api/v1/mcp/tools",
        response_model=McpToolList,
        responses=ERROR_RESPONSES,
        tags=["mcp"],
    )
    async def list_mcp_tools() -> McpToolList:
        return await require_service().list_mcp_tools()

    @app.post(
        "/api/v1/mcp/tools/call",
        response_model=McpToolCallResult,
        responses=ERROR_RESPONSES,
        tags=["mcp"],
    )
    async def call_mcp_tool(body: McpToolCallRequest) -> McpToolCallResult:
        return await require_service().call_mcp_tool(body)

    @app.post(
        "/api/v1/models/load",
        response_model=OperationAccepted,
        status_code=202,
        responses=ERROR_RESPONSES,
        tags=["models"],
    )
    async def load_model(body: ModelLoadRequest) -> OperationAccepted:
        return await require_service().load_model(body)

    @app.get(
        "/api/v1/models",
        response_model=ModelArtifactList,
        responses=ERROR_RESPONSES,
        tags=["models"],
    )
    async def model_artifacts(refresh: bool = False) -> ModelArtifactList:
        return await require_service().model_artifacts(refresh=refresh)

    @app.get(
        "/api/v1/artifacts/lineage",
        response_model=ArtifactLineageList,
        responses=ERROR_RESPONSES,
        tags=["models"],
    )
    async def artifact_lineage(
        artifact_uri: Annotated[str | None, Query(max_length=2048)] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> ArtifactLineageList:
        return await require_service().artifact_lineage(
            artifact_uri=artifact_uri,
            limit=limit,
        )

    @app.post(
        "/api/v1/artifacts/remove",
        response_model=dict[str, Any],
        responses=ERROR_RESPONSES,
        tags=["models"],
    )
    async def delete_workspace_artifact(
        body: DeleteWorkspaceArtifactRequest,
    ) -> dict[str, Any]:
        return await require_service().delete_workspace_artifact(body)

    @app.post(
        "/api/v1/datasets",
        response_model=DatasetResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["evaluation"],
    )
    async def create_dataset(body: CreateDatasetRequest) -> DatasetResource:
        return await require_service().create_dataset(body)

    @app.get(
        "/api/v1/datasets",
        response_model=DatasetList,
        responses=ERROR_RESPONSES,
        tags=["evaluation"],
    )
    async def list_datasets() -> DatasetList:
        return await require_service().list_datasets()

    @app.delete(
        "/api/v1/datasets/{dataset_id}",
        status_code=204,
        responses=ERROR_RESPONSES,
        tags=["evaluation"],
    )
    async def delete_dataset(dataset_id: UUID) -> Response:
        await require_service().delete_dataset(dataset_id)
        return Response(status_code=204)

    @app.get(
        "/api/v1/evaluations",
        response_model=EvaluationResultList,
        responses=ERROR_RESPONSES,
        tags=["evaluation"],
    )
    async def evaluations(
        kind: EvaluationKind | None = None,
        model_id: Annotated[str | None, Query(max_length=255)] = None,
        limit: Annotated[int, Query(ge=1, le=2000)] = 200,
    ) -> EvaluationResultList:
        return await require_service().evaluations(kind=kind, model_id=model_id, limit=limit)

    @app.post(
        "/api/v1/evaluations/compare",
        response_model=EvaluationComparisonResource,
        responses=ERROR_RESPONSES,
        tags=["evaluation"],
    )
    async def compare_evaluations(
        body: CompareEvaluationsRequest,
    ) -> EvaluationComparisonResource:
        return await require_service().compare_evaluations(body)

    @app.post(
        "/api/v1/cluster/nodes",
        response_model=RemoteNodeResource,
        status_code=201,
        responses=ERROR_RESPONSES,
        tags=["cluster"],
    )
    async def create_remote_node(body: CreateRemoteNodeRequest) -> RemoteNodeResource:
        return await require_service().create_remote_node(body)

    @app.get(
        "/api/v1/cluster/nodes",
        response_model=RemoteNodeList,
        responses=ERROR_RESPONSES,
        tags=["cluster"],
    )
    async def remote_nodes(refresh: bool = False) -> RemoteNodeList:
        return await require_service().remote_nodes(refresh=refresh)

    @app.put(
        "/api/v1/cluster/nodes/{node_id}",
        response_model=RemoteNodeResource,
        responses=ERROR_RESPONSES,
        tags=["cluster"],
    )
    async def update_remote_node(
        node_id: UUID, body: UpdateRemoteNodeRequest
    ) -> RemoteNodeResource:
        return await require_service().update_remote_node(node_id, body)

    @app.delete(
        "/api/v1/cluster/nodes/{node_id}",
        status_code=204,
        responses=ERROR_RESPONSES,
        tags=["cluster"],
    )
    async def delete_remote_node(node_id: UUID) -> Response:
        await require_service().delete_remote_node(node_id)
        return Response(status_code=204)

    @app.post(
        "/api/v1/models/unload",
        response_model=OperationAccepted,
        status_code=202,
        responses=ERROR_RESPONSES,
        tags=["models"],
    )
    async def unload_model(body: ModelUnloadRequest) -> OperationAccepted:
        return await require_service().unload_model(body)

    @app.get(
        "/api/v1/runtime/instances",
        response_model=RuntimeInstanceList,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def runtime_instances() -> RuntimeInstanceList:
        return await require_service().runtime_instances()

    @app.get(
        "/api/v1/runtime/capabilities",
        response_model=RuntimeCapabilitiesResource,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def runtime_capabilities() -> RuntimeCapabilitiesResource:
        return await require_service().runtime_capabilities()

    @app.get(
        "/api/v1/runtime/status",
        response_model=dict[str, Any],
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def runtime_status() -> dict[str, Any]:
        return await require_service().runtime_status()

    @app.get(
        "/api/v1/runtime/metrics",
        response_model=RuntimeMetricList,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def runtime_metrics(
        instance_id: UUID | None = None,
        since: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=2000)] = 200,
    ) -> RuntimeMetricList:
        return await require_service().runtime_metrics(
            instance_id=instance_id, since=since, limit=limit
        )

    @app.get(
        "/api/v1/runtime/logs",
        response_model=RuntimeLogList,
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def runtime_logs(
        instance_id: UUID | None = None,
        level: RuntimeLogLevel | None = None,
        after: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=2000)] = 200,
    ) -> RuntimeLogList:
        return await require_service().runtime_logs(
            instance_id=instance_id, level=level, after=after, limit=limit
        )

    @app.get(
        "/api/v1/runtime/models",
        response_model=dict[str, Any],
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def runtime_models() -> dict[str, Any]:
        return await require_service().runtime_models()

    @app.get(
        "/api/v1/runtime/realtime/capabilities",
        response_model=dict[str, Any],
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def realtime_capabilities() -> dict[str, Any]:
        return await require_service().realtime_capabilities()

    @app.post(
        "/api/v1/runtime/reload",
        response_model=dict[str, Any],
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def reload_runtime(body: RuntimeReloadRequest) -> dict[str, Any]:
        return await require_service().reload_runtime(body.context_size)

    @app.post(
        "/api/v1/runtime/cache/clear",
        response_model=dict[str, Any],
        responses=ERROR_RESPONSES,
        tags=["runtime"],
    )
    async def clear_runtime_cache() -> dict[str, Any]:
        return await require_service().clear_runtime_cache()

    @app.websocket("/api/v1/runtime/realtime")
    async def runtime_realtime(websocket: WebSocket) -> None:
        if not await authorize_websocket(websocket):
            return
        mode = websocket.query_params.get("mode", "audio")
        if mode != "audio":
            await websocket.close(code=1008, reason="audio mode is required")
            return
        try:
            connector = require_service().realtime_connect(mode=mode)
            async with connector as upstream:
                await websocket.accept()

                async def send_upstream() -> None:
                    while True:
                        message = await websocket.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        if message.get("text") is not None:
                            await upstream.send(message["text"])
                        elif message.get("bytes") is not None:
                            await upstream.send(message["bytes"])

                async def send_client() -> None:
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)

                tasks = {
                    asyncio.create_task(send_upstream()),
                    asyncio.create_task(send_client()),
                }
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    task.result()
                with suppress(Exception):
                    await websocket.close(code=1000)
        except WebSocketDisconnect:
            return
        except Exception as error:
            with suppress(Exception):
                await websocket.send_json({"type": "error", "error": {"message": str(error)}})
            with suppress(Exception):
                await websocket.close(code=1011, reason="realtime proxy failed")

    @app.websocket("/api/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        if not await authorize_websocket(websocket):
            return
        await websocket.close(code=1013, reason="Realtime audio transport is not available")

    if web_root is not None:
        root = Path(web_root)
        if not root.is_dir():
            raise ValueError(f"web root is not a directory: {root}")
        app.mount("/", StaticFiles(directory=root, html=True), name="web")

    return app


def create_contract_app() -> FastAPI:
    """Create the route-only application used to generate the public OpenAPI contract."""

    return create_app()
