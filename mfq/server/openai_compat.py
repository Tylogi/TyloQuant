"""OpenAI chat-completions facade for the managed MFQ runtime pool."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from mfq.server.backend import (
    BackendDelta,
    BackendError,
    BackendToolCallDelta,
    ChatBackend,
)
from mfq.server.models import (
    NamedToolChoice,
    ResponseFormat,
    ResponsePerformance,
    SamplingParams,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
)


class OpenAIRequestError(ValueError):
    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


@dataclass(frozen=True)
class OpenAIChatRequest:
    model: str
    messages: tuple[dict[str, Any], ...]
    sampling: SamplingParams
    stream: bool
    include_usage: bool
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice = "auto"
    response_format: ResponseFormat | None = None
    session_id: UUID | None = None


_TOOL_CHOICE = TypeAdapter(ToolChoice)
_RESPONSE_FORMAT = TypeAdapter(ResponseFormat)


def _value(body: dict[str, Any], name: str, default: Any) -> Any:
    value = body.get(name, default)
    return default if value is None else value


def _boolean(body: dict[str, Any], name: str, default: bool) -> bool:
    value = body.get(name, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise OpenAIRequestError(f"{name} must be a boolean", param=name)
    return value


def parse_chat_request(body: Any) -> OpenAIChatRequest:
    """Validate the OpenAI fields consumed by MFQ while tolerating extensions."""

    if not isinstance(body, dict):
        raise OpenAIRequestError("request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise OpenAIRequestError("model must be a non-empty string", param="model")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise OpenAIRequestError("messages must be a non-empty array", param="messages")
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise OpenAIRequestError(
                f"messages[{index}] must be an object",
                param=f"messages[{index}]",
            )
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise OpenAIRequestError(
                f"messages[{index}].role must be a non-empty string",
                param=f"messages[{index}].role",
            )
        messages.append(dict(message))

    if _value(body, "n", 1) != 1:
        raise OpenAIRequestError("MFQ currently supports n=1", param="n")

    template_kwargs = body.get("chat_template_kwargs")
    if template_kwargs is not None and not isinstance(template_kwargs, dict):
        raise OpenAIRequestError(
            "chat_template_kwargs must be an object",
            param="chat_template_kwargs",
        )
    template_kwargs = template_kwargs or {}
    enable_thinking = body.get(
        "enable_thinking",
        template_kwargs.get("enable_thinking", True),
    )
    if not isinstance(enable_thinking, bool):
        raise OpenAIRequestError(
            "enable_thinking must be a boolean",
            param="enable_thinking",
        )
    reasoning_effort = body.get(
        "reasoning_effort",
        template_kwargs.get("reasoning_effort"),
    )

    max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = _value(body, "max_tokens", 4096)
    sampling_values = {
        "max_tokens": max_tokens,
        "temperature": _value(body, "temperature", 1.0),
        "top_k": _value(body, "top_k", 20),
        "top_p": _value(body, "top_p", 0.95),
        "presence_penalty": _value(body, "presence_penalty", 0.0),
        "frequency_penalty": _value(body, "frequency_penalty", 0.0),
        "repetition_penalty": _value(body, "repetition_penalty", 1.0),
        "seed": body.get("seed"),
        "enable_thinking": enable_thinking,
        "enable_vision": _boolean(body, "enable_vision", True),
        "enable_mtp": _boolean(body, "enable_mtp", True),
        "mtp_max_draft_tokens": _value(body, "mtp_max_draft_tokens", 5),
        "reasoning_effort": reasoning_effort,
    }
    try:
        sampling = SamplingParams.model_validate(sampling_values)
    except ValidationError as error:
        raise OpenAIRequestError(str(error)) from error

    raw_tools = body.get("tools") or []
    if not isinstance(raw_tools, list):
        raise OpenAIRequestError("tools must be an array", param="tools")
    try:
        tools = tuple(ToolDefinition.model_validate(tool) for tool in raw_tools)
        tool_choice = _TOOL_CHOICE.validate_python(_value(body, "tool_choice", "auto"))
        response_format = (
            _RESPONSE_FORMAT.validate_python(body["response_format"])
            if body.get("response_format") is not None
            else None
        )
    except ValidationError as error:
        raise OpenAIRequestError(str(error)) from error
    if isinstance(tool_choice, NamedToolChoice):
        names = {tool.function.name for tool in tools}
        if tool_choice.function.name not in names:
            raise OpenAIRequestError(
                "named tool_choice must match a supplied tool",
                param="tool_choice",
            )

    stream = _boolean(body, "stream", False)
    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise OpenAIRequestError("stream_options must be an object", param="stream_options")
    include_usage = stream_options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise OpenAIRequestError(
            "stream_options.include_usage must be a boolean",
            param="stream_options.include_usage",
        )

    session_id = None
    if body.get("mfq_session_id") is not None:
        try:
            session_id = UUID(str(body["mfq_session_id"]))
        except ValueError as error:
            raise OpenAIRequestError(
                "mfq_session_id must be a UUID",
                param="mfq_session_id",
            ) from error

    return OpenAIChatRequest(
        model=model,
        messages=tuple(messages),
        sampling=sampling,
        stream=stream,
        include_usage=include_usage,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        session_id=session_id,
    )


def error_body(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | int | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def backend_error_status(error: BackendError) -> int:
    return error.status_code or (503 if error.retryable else 502)


@dataclass
class _ToolCall:
    call_id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)


@dataclass
class _Accumulator:
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tools: dict[int, _ToolCall] = field(default_factory=dict)
    finish_reason: str = "stop"
    usage: TokenUsage | None = None
    performance: ResponsePerformance | None = None

    def apply(self, delta: BackendDelta) -> None:
        if delta.content_delta:
            self.content.append(delta.content_delta)
        if delta.reasoning_delta:
            self.reasoning.append(delta.reasoning_delta)
        for tool_delta in delta.tool_calls:
            tool = self.tools.setdefault(tool_delta.index, _ToolCall())
            if tool_delta.call_id is not None:
                tool.call_id = tool_delta.call_id
            if tool_delta.name is not None:
                tool.name = tool_delta.name
            if tool_delta.arguments_delta:
                tool.arguments.append(tool_delta.arguments_delta)
        if delta.finish_reason is not None:
            self.finish_reason = delta.finish_reason
        if delta.usage is not None:
            self.usage = delta.usage
        if delta.performance is not None:
            self.performance = delta.performance

    def tool_calls(self) -> list[dict[str, Any]]:
        return [
            {
                "id": tool.call_id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": tool.name or "unknown",
                    "arguments": "".join(tool.arguments) or "{}",
                },
            }
            for index, tool in sorted(self.tools.items())
        ]


def _backend_stream(
    backend: ChatBackend,
    request: OpenAIChatRequest,
) -> AsyncIterator[BackendDelta]:
    return backend.stream(
        model=request.model,
        messages=request.messages,
        sampling=request.sampling,
        session_id=request.session_id,
        tools=request.tools,
        tool_choice=request.tool_choice,
        response_format=request.response_format,
    )


async def collect_chat_completion(
    backend: ChatBackend,
    request: OpenAIChatRequest,
) -> dict[str, Any]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    accumulator = _Accumulator()
    async for delta in _backend_stream(backend, request):
        accumulator.apply(delta)
    tool_calls = accumulator.tool_calls()
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(accumulator.content) or None,
    }
    reasoning = "".join(accumulator.reasoning)
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    result: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": "tool_calls" if tool_calls else accumulator.finish_reason,
            }
        ],
    }
    if accumulator.usage is not None:
        result["usage"] = accumulator.usage.model_dump(mode="json")
    if accumulator.performance is not None:
        result["mfq_metrics"] = accumulator.performance.model_dump(mode="json")
    return result


def _sse(payload: dict[str, Any] | str) -> str:
    value = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return f"data: {value}\n\n"


def _chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def _tool_deltas(values: Sequence[BackendToolCallDelta]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value in values:
        item: dict[str, Any] = {"index": value.index}
        if value.call_id is not None:
            item.update({"id": value.call_id, "type": "function"})
        function: dict[str, str] = {}
        if value.name is not None:
            function["name"] = value.name
        if value.arguments_delta:
            function["arguments"] = value.arguments_delta
        if function:
            item["function"] = function
        output.append(item)
    return output


async def stream_chat_completion(
    backend: ChatBackend,
    request: OpenAIChatRequest,
    *,
    disconnected: Callable[[], Awaitable[bool]] | None = None,
    keepalive_seconds: float = 15.0,
) -> AsyncIterator[str]:
    """Serialize one backend stream with stable ids, keepalives, and cleanup."""

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    yield _sse(
        _chunk(
            response_id=response_id,
            created=created,
            model=request.model,
            delta={"role": "assistant", "content": ""},
        )
    )
    iterator = _backend_stream(backend, request).__aiter__()
    pending: asyncio.Task[BackendDelta] | None = None
    terminal_emitted = False
    final_usage: TokenUsage | None = None
    final_performance: ResponsePerformance | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(iterator))
            done, _ = await asyncio.wait({pending}, timeout=keepalive_seconds)
            if not done:
                if disconnected is not None and await disconnected():
                    return
                yield ": keep-alive\n\n"
                continue
            task, pending = pending, None
            try:
                delta = task.result()
            except StopAsyncIteration:
                break

            visible: dict[str, Any] = {}
            if delta.reasoning_delta:
                visible["reasoning_content"] = delta.reasoning_delta
            if delta.content_delta:
                visible["content"] = delta.content_delta
            if delta.tool_calls:
                visible["tool_calls"] = _tool_deltas(delta.tool_calls)
            if visible:
                yield _sse(
                    _chunk(
                        response_id=response_id,
                        created=created,
                        model=request.model,
                        delta=visible,
                    )
                )
            if delta.finish_reason is not None:
                terminal_emitted = True
                yield _sse(
                    _chunk(
                        response_id=response_id,
                        created=created,
                        model=request.model,
                        delta={},
                        finish_reason=delta.finish_reason,
                    )
                )
            if delta.usage is not None:
                final_usage = delta.usage
            if delta.performance is not None:
                final_performance = delta.performance
        if not terminal_emitted:
            yield _sse(
                _chunk(
                    response_id=response_id,
                    created=created,
                    model=request.model,
                    delta={},
                    finish_reason="stop",
                )
            )
        if request.include_usage and (
            final_usage is not None or final_performance is not None
        ):
            usage_payload: dict[str, Any] = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [],
            }
            if final_usage is not None:
                usage_payload["usage"] = final_usage.model_dump(mode="json")
            if final_performance is not None:
                usage_payload["mfq_metrics"] = final_performance.model_dump(mode="json")
            yield _sse(usage_payload)
    except BackendError as error:
        yield _sse(
            error_body(
                str(error),
                error_type=error.code,
                code=error.status_code,
            )
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        yield _sse(error_body(str(error), error_type="server_error"))
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()
    yield _sse("[DONE]")
