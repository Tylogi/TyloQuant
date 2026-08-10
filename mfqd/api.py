"""FastAPI application for the MFQ daemon native API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import Body, FastAPI, Header, Query, Response, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mfqd.models import (
    SHA256_PATTERN,
    CreateResponseRequest,
    CreateSessionRequest,
    ErrorDetail,
    ErrorResponse,
    ForkSessionRequest,
    MediaResource,
    MessageList,
    ModelLoadRequest,
    ModelUnloadRequest,
    OperationAccepted,
    ResponseResource,
    RuntimeCapabilitiesResource,
    RuntimeInstanceList,
    SessionList,
    SessionResource,
)
from mfqd.service import MfqdService, ServiceError

ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    501: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def create_app(
    service: MfqdService | None = None,
    *,
    web_root: str | Path | None = None,
) -> FastAPI:
    """Create an executable app, or a route-only contract app when service is omitted."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if service is not None:
                await service.aclose()

    app = FastAPI(
        title="MFQd Native API",
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
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )

    def require_service() -> MfqdService:
        if service is None:
            raise ServiceError(
                501,
                "contract_only",
                "the protocol contract application does not execute requests",
            )
        return service

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "mfqd",
            "protocol_version": "1.0",
        }

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

    @app.get(
        "/api/v1/sessions/{session_id}/messages",
        response_model=MessageList,
        responses=ERROR_RESPONSES,
        tags=["sessions"],
    )
    async def list_messages(session_id: UUID) -> MessageList:
        return await require_service().list_messages(session_id)

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
        del body, content_type, content_sha256
        require_service()
        raise ServiceError(501, "media_unavailable", "media upload is not available")

    @app.post(
        "/api/v1/models/load",
        response_model=OperationAccepted,
        status_code=202,
        responses=ERROR_RESPONSES,
        tags=["models"],
    )
    async def load_model(body: ModelLoadRequest) -> OperationAccepted:
        return await require_service().load_model(body)

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

    @app.websocket("/api/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
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
