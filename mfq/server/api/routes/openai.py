"""OpenAI-compatible health, model, and chat-completion routes."""

from __future__ import annotations

from typing import Any
from dataclasses import replace

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.anthropic_compat import error_body as anthropic_error_body, is_anthropic_request
from mfq.server.api.openai_compat import (
    OpenAIRequestError,
    backend_error_status,
    collect_chat_completion,
    parse_chat_request,
    require_json_content_type,
    stream_chat_completion,
)
from mfq.server.api.openai_compat import (
    error_body as openai_error_body,
)
from mfq.server.runtime.backend import BackendError, preflight_backend_request

router = APIRouter()


@router.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "mfq-server",
        "protocol_version": "1.0",
    }


@router.get("/v1", include_in_schema=False)
async def openai_root(request: Request) -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "mfq-server",
        "endpoints": ["/v1/models", "/v1/messages"] if is_anthropic_request(request) else ["/v1/models", "/v1/chat/completions"],
    }


@router.get("/health/identity", include_in_schema=False)
async def server_identity(request: Request) -> dict[str, Any]:
    return request.app.state.server_identity


@router.get("/v1/models", include_in_schema=False)
async def openai_models(
    service: ServiceDependency, request: Request,
) -> Any:
    models = await service.advertised_models()
    data = models.get("data") if isinstance(models, dict) else None
    if is_anthropic_request(request):
        items = [{"id": item["id"], "type": "model", "display_name": item["id"],
                  "created_at": "1970-01-01T00:00:00Z"}
                 for item in (data if isinstance(data, list) else [])
                 if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]]
        try:
            limit = int(request.query_params.get("limit", "20"))
            if not 1 <= limit <= 1000:
                raise ValueError("limit must be between 1 and 1000")
            for key in ("after_id", "before_id"):
                cursor = request.query_params.get(key)
                if cursor:
                    offset = next((index for index, item in enumerate(items) if item["id"] == cursor), None)
                    if offset is None:
                        raise ValueError(f"unknown {key}")
                    items = items[offset + 1:] if key == "after_id" else items[:offset]
        except ValueError as error:
            return JSONResponse(status_code=400, content=anthropic_error_body(str(error)))
        page = items[:limit]
        return {"data": page, "has_more": len(items) > limit,
                "first_id": page[0]["id"] if page else None, "last_id": page[-1]["id"] if page else None}
    return {
        "object": "list",
        "data": [
            {
                "id": item["id"],
                "object": "model",
                "created": 0,
                "owned_by": "mfq",
            }
            for item in (data if isinstance(data, list) else [])
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        ],
    }


@router.post("/v1/chat/completions", include_in_schema=False)
async def openai_chat_completions(service: ServiceDependency, request: Request) -> Response:
    try:
        require_json_content_type(request.headers.get("content-type", ""))
        body = await request.json()
    except OpenAIRequestError as error:
        return JSONResponse(
            status_code=error.status_code,
            content=openai_error_body(str(error), param=error.param),
        )
    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content=openai_error_body(str(error)),
        )
    try:
        parsed = parse_chat_request(body)
    except OpenAIRequestError as error:
        return JSONResponse(
            status_code=error.status_code,
            content=openai_error_body(str(error), param=error.param),
        )
    backend = service.backend
    parsed = replace(parsed, routing_model=service.resolve_model_alias(parsed.model))
    if parsed.stream:
        try:
            await preflight_backend_request(
                backend,
                model=parsed.routing_model,
                messages=parsed.messages,
                sampling=parsed.sampling,
                session_id=parsed.session_id,
                tools=parsed.tools,
                tool_choice=parsed.tool_choice,
                response_format=parsed.response_format,
            )
        except BackendError as error:
            return JSONResponse(
                status_code=backend_error_status(error),
                headers={"Retry-After": "1"} if error.retryable and backend_error_status(error) in {429, 503} else None,
                content=openai_error_body(
                    str(error),
                    error_type=error.code,
                    code=error.status_code,
                ),
            )
        return StreamingResponse(
            stream_chat_completion(
                backend,
                parsed,
                disconnected=request.is_disconnected,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    try:
        result = await collect_chat_completion(backend, parsed)
    except BackendError as error:
        return JSONResponse(
            status_code=backend_error_status(error),
            headers={"Retry-After": "1"} if error.retryable and backend_error_status(error) in {429, 503} else None,
            content=openai_error_body(
                str(error),
                error_type=error.code,
                code=error.status_code,
            ),
        )
    return JSONResponse(content=result)


@router.post("/v1/completions", include_in_schema=False)
async def unsupported_completions() -> Response:
    return JSONResponse(
        status_code=404,
        content=openai_error_body(
            "Legacy /v1/completions is not supported. Use /v1/chat/completions or /v1/messages.",
            code="unsupported_endpoint",
        ),
    )
