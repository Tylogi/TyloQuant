from dataclasses import replace

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mfq.server.api.anthropic_compat import (
    collect_message,
    error_body,
    error_type,
    parse_messages_request,
    stream_message,
)
from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.openai_compat import OpenAIRequestError, backend_error_status, require_json_content_type
from mfq.server.runtime.backend import BackendError, preflight_backend_request

router = APIRouter()


@router.post("/v1/messages/count_tokens", include_in_schema=False)
async def count_tokens() -> Response:
    return JSONResponse(status_code=501, content=error_body(
        "This runtime does not expose exact prompt tokenization; token counting is not supported.", "api_error"))


@router.get("/v1/models/{model_id}", include_in_schema=False)
async def model_info(service: ServiceDependency, model_id: str) -> Response:
    models = await service.advertised_models()
    if any(item.get("id") == model_id for item in models.get("data", [])):
        return JSONResponse(content={"id": model_id, "type": "model", "display_name": model_id,
                                     "created_at": "1970-01-01T00:00:00Z"})
    return JSONResponse(status_code=404, content=error_body("model not found", "not_found_error"))


@router.post("/v1/messages", include_in_schema=False)
async def messages(service: ServiceDependency, request: Request) -> Response:
    try:
        require_json_content_type(request.headers.get("content-type", ""))
        parsed = parse_messages_request(await request.json())
    except (ValueError, OpenAIRequestError) as error:
        status = error.status_code if isinstance(error, OpenAIRequestError) else 400
        return JSONResponse(status_code=status, content=error_body(str(error), error_type(status)))
    parsed = replace(parsed, chat=replace(parsed.chat, routing_model=service.resolve_model_alias(parsed.chat.model)))
    try:
        if parsed.chat.stream:
            chat = parsed.chat
            await preflight_backend_request(service.backend, model=chat.routing_model, messages=chat.messages,
                sampling=chat.sampling, session_id=chat.session_id, tools=chat.tools,
                tool_choice=chat.tool_choice, response_format=chat.response_format)
            return StreamingResponse(stream_message(service.backend, parsed, disconnected=request.is_disconnected),
                media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return JSONResponse(content=await collect_message(service.backend, parsed))
    except BackendError as error:
        status = backend_error_status(error)
        return JSONResponse(status_code=status, content=error_body(str(error), error_type(status)),
            headers={"Retry-After": "1"} if error.retryable and status in {429, 503, 529} else None)
