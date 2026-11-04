from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any

from mfq.server.api.openai_compat import (
    OpenAIChatRequest,
    OpenAIRequestError,
    parse_chat_request,
    stream_chat_completion,
)
from mfq.server.runtime.backend import BackendError, ChatBackend


def error_body(message: str, error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def is_anthropic_request(request: Any) -> bool:
    if request.url.path.startswith("/v1/messages"):
        return True
    if request.url.path == "/v1" or request.url.path.startswith("/v1/models"):
        listener = getattr(request.app.state, "listener", None)
        server = request.scope.get("server")
        return bool(request.headers.get("anthropic-version") or request.headers.get("x-api-key") or (
            listener is not None and server and server[1] == listener.anthropic_port))
    return False


def error_type(status: int) -> str:
    return {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
            404: "not_found_error", 413: "request_too_large",
            415: "invalid_request_error", 422: "invalid_request_error",
            429: "rate_limit_error", 503: "overloaded_error", 529: "overloaded_error"}.get(status, "api_error")


def _string(value: Any, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise OpenAIRequestError(f"{field} must be a {'string' if empty else 'non-empty string'}")
    return value


def _content(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list) or any(not isinstance(block, dict) for block in value):
        raise OpenAIRequestError("content must be a string or an array of content blocks")
    return value


def _media_or_text(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "text":
        return {"type": "text", "text": _string(block.get("text"), "text", empty=True)}
    if block.get("type") == "image":
        source = block.get("source")
        if isinstance(source, dict):
            if source.get("type") == "url":
                url = _string(source.get("url"), "image.source.url")
                if not url.startswith(("http://", "https://")):
                    raise OpenAIRequestError("image URLs must use HTTP or HTTPS")
                return {"type": "image_url", "image_url": {"url": url}}
            if source.get("type") == "base64":
                media_type = source.get("media_type")
                if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
                    raise OpenAIRequestError("unsupported image media_type")
                data = _string(source.get("data"), "image.source.data")
                return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
        raise OpenAIRequestError("image.source must be a URL or base64 source")
    raise OpenAIRequestError(f"unsupported content block type: {block.get('type')}")


@dataclass(frozen=True)
class MessagesRequest:
    chat: OpenAIChatRequest
    stop_sequences: tuple[str, ...] = ()


def parse_messages_request(body: Any) -> MessagesRequest:
    if not isinstance(body, dict):
        raise OpenAIRequestError("request body must be a JSON object")
    maximum = body.get("max_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise OpenAIRequestError("max_tokens must be a positive integer; cache-only generation is not supported")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise OpenAIRequestError("messages must be a non-empty array")
    messages: list[dict[str, Any]] = []
    if "system" in body:
        system = _content(body["system"])
        if any(block.get("type") != "text" for block in system):
            raise OpenAIRequestError("system accepts only text blocks")
        messages.append({"role": "system", "content": "\n".join(
            _string(block.get("text"), "system.text", empty=True) for block in system)})
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise OpenAIRequestError(f"messages[{index}] must be an object", param=f"messages[{index}]")
        if message.get("role") not in ("user", "assistant"):
            raise OpenAIRequestError(
                f"messages[{index}].role must be one of: user, assistant; use top-level system",
                param=f"messages[{index}].role",
            )
        role = message["role"]
        content: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        reasoning: list[str] = []

        def flush(content: list[dict[str, Any]] = content, calls: list[dict[str, Any]] = calls,
                  reasoning: list[str] = reasoning, role: str = role) -> None:
            if content or calls or reasoning:
                converted: dict[str, Any] = {"role": role, "content": list(content)}
                if calls:
                    converted["tool_calls"] = list(calls)
                if reasoning:
                    converted["reasoning_content"] = "".join(reasoning)
                messages.append(converted)
                content.clear()
                calls.clear()
                reasoning.clear()

        blocks = _content(message.get("content"))
        if not blocks:
            raise OpenAIRequestError("messages must contain at least one content block")
        for block in blocks:
            kind = block.get("type")
            if kind == "tool_use" and role == "assistant":
                arguments = block.get("input")
                if not isinstance(arguments, dict):
                    raise OpenAIRequestError("tool_use.input must be an object")
                calls.append({"id": _string(block.get("id"), "tool_use.id"), "type": "function",
                    "function": {"name": _string(block.get("name"), "tool_use.name"),
                                 "arguments": json.dumps(arguments, ensure_ascii=False)}})
            elif kind == "tool_result" and role == "user":
                flush()
                result = [_media_or_text(item) for item in _content(block.get("content", ""))]
                messages.append({"role": "tool", "tool_call_id": _string(block.get("tool_use_id"), "tool_result.tool_use_id"),
                                 "content": result})
            elif kind == "thinking" and role == "assistant":
                reasoning.append(_string(block.get("thinking"), "thinking", empty=True))
            else:
                content.append(_media_or_text(block))
        flush()
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        raise OpenAIRequestError("tools must be an array")
    converted_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") not in {None, "custom"}:
            raise OpenAIRequestError("only client-defined tools are supported")
        schema = tool.get("input_schema")
        if not isinstance(schema, dict):
            raise OpenAIRequestError("tools.input_schema must be an object")
        converted_tools.append({"type": "function", "function": {
            "name": _string(tool.get("name"), "tools.name"),
            "description": tool.get("description", ""), "parameters": schema}})
    choice = body.get("tool_choice", {"type": "auto"})
    if not isinstance(choice, dict) or choice.get("type") not in {"auto", "any", "none", "tool"}:
        raise OpenAIRequestError("tool_choice.type must be auto, any, none or tool")
    if choice.get("disable_parallel_tool_use"):
        raise OpenAIRequestError("disable_parallel_tool_use is not supported")
    converted_choice: Any = {"auto": "auto", "any": "required", "none": "none"}.get(choice["type"])
    if choice["type"] == "tool":
        converted_choice = {"type": "function", "function": {"name": _string(choice.get("name"), "tool_choice.name")}}
    if choice["type"] in {"tool", "any"} and not tools:
        raise OpenAIRequestError("tool_choice requires tools")
    if choice["type"] == "tool" and choice["name"] not in {tool["function"]["name"] for tool in converted_tools}:
        raise OpenAIRequestError("tool_choice names an undefined tool")
    # Scheduling extensions follow the same top-level convention as the other
    # MFQ Message fields; ``metadata`` stays reserved for application data.
    converted = {key: body[key] for key in (
        "model", "max_tokens", "temperature", "top_p", "top_k", "stream",
        "mfq_session_id", "enable_mtp", "enable_thinking", "seed",
        "priority", "timeout_seconds",
    ) if key in body}
    if "temperature" in body and (isinstance(body["temperature"], bool) or not isinstance(body["temperature"], (int, float)) or not 0 <= body["temperature"] <= 1):
        raise OpenAIRequestError("temperature must be between 0 and 1")
    thinking = body.get("thinking")
    if thinking is not None:
        if not isinstance(thinking, dict) or thinking.get("type") not in {"enabled", "disabled", "adaptive"}:
            raise OpenAIRequestError("thinking.type must be enabled, disabled or adaptive")
        if thinking["type"] == "enabled":
            budget = thinking.get("budget_tokens")
            if isinstance(budget, bool) or not isinstance(budget, int) or not 1024 <= budget < maximum:
                raise OpenAIRequestError("thinking.budget_tokens must be at least 1024 and below max_tokens")
            raise OpenAIRequestError("thinking.budget_tokens is not supported by MFQ; use adaptive or disabled thinking")
        converted["enable_thinking"] = thinking["type"] != "disabled"
    for field in ("output_config", "context_management", "container"):
        if body.get(field) is not None:
            raise OpenAIRequestError(f"{field} is not supported")
    stops = body.get("stop_sequences", [])
    if not isinstance(stops, list) or len(stops) > 16 or any(not isinstance(stop, str) or not stop for stop in stops):
        raise OpenAIRequestError("stop_sequences must contain at most 16 non-empty strings")
    converted.update(messages=messages, tools=converted_tools, tool_choice=converted_choice,
                     stream_options={"include_usage": True})
    return MessagesRequest(parse_chat_request(converted), tuple(stops))


def _usage(usage: dict[str, Any]) -> dict[str, int]:
    return {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}


def _stop(text: str, sequences: tuple[str, ...]) -> tuple[str, str | None]:
    matches = [(text.find(stop), stop) for stop in sequences if stop in text]
    if not matches:
        return text, None
    offset, sequence = min(matches, key=lambda pair: pair[0])
    return text[:offset], sequence


def _finish(reason: str | None, stopped: str | None, tools: bool) -> str:
    return "stop_sequence" if stopped else "tool_use" if tools else "max_tokens" if reason == "length" else "end_turn"


async def collect_message(backend: ChatBackend, request: MessagesRequest) -> dict[str, Any]:
    result: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    arguments: dict[int, str] = {}
    async for frame in stream_message(backend, request):
        event = json.loads(frame.split("data: ", 1)[1])
        kind = event["type"]
        if kind == "message_start":
            result = event["message"]
        elif kind == "content_block_start":
            block = event["content_block"]
            if block["type"] == "thinking":
                block["signature"] = ""
            blocks[event["index"]] = block
        elif kind == "content_block_delta":
            index = event["index"]
            delta = event["delta"]
            if delta["type"] == "input_json_delta":
                arguments[index] = arguments.get(index, "") + delta["partial_json"]
            else:
                field = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}[delta["type"]]
                blocks[index][field] += delta[field]
        elif kind == "content_block_stop" and event["index"] in arguments:
            blocks[event["index"]]["input"] = json.loads(arguments[event["index"]])
        elif kind == "message_delta":
            result.update(event["delta"])
            result["usage"].update(event["usage"])
        elif kind == "error":
            raise BackendError(event["error"]["type"], event["error"]["message"],
                               status_code=event.get("status", 502))
    result["content"] = [block for _, block in sorted(blocks.items())]
    return result


def _event(kind: str, **payload: Any) -> str:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **payload}, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def stream_message(backend: ChatBackend, request: MessagesRequest, *,
                         disconnected: Callable[[], Awaitable[bool]] | None = None,
                         keepalive_seconds: float = 15.0) -> AsyncIterator[str]:
    yield _event("message_start", message={"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
        "content": [], "model": request.chat.model, "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}})
    own_session = bool(request.stop_sequences and request.chat.session_id is None)
    chat = replace(request.chat, session_id=uuid.uuid4()) if own_session else request.chat
    iterator = stream_chat_completion(backend, chat, disconnected=disconnected, keepalive_seconds=keepalive_seconds)
    count = 0
    active: tuple[int, str] | None = None
    tool_blocks: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    reason = None
    stopped = None
    buffered = ""
    completed = False
    try:
        async for frame in iterator:
            if disconnected is not None and await disconnected():
                return
            if frame.startswith(":"):
                yield _event("ping")
                continue
            data = frame.removeprefix("data: ").strip()
            if data == "[DONE]":
                completed = True
                break
            payload = json.loads(data)
            if "error" in payload:
                status = payload["error"].get("code")
                yield _event("error", error={"type": error_type(status) if isinstance(status, int) else "api_error", "message": payload["error"]["message"]},
                             **({"status": status} if isinstance(status, int) else {}))
                return
            usage = payload.get("usage", usage)
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            reason = choice.get("finish_reason") or reason
            delta = choice.get("delta", {})
            values = [("thinking", delta.get("reasoning_content", "")), ("text", delta.get("content", ""))]
            for kind, value in values:
                if not value or stopped:
                    continue
                if kind == "text" and request.stop_sequences:
                    buffered += value
                    value, stopped = _stop(buffered, request.stop_sequences)
                    retain = 0 if stopped else max((size for stop in request.stop_sequences
                        for size in range(1, len(stop)) if buffered.endswith(stop[:size])), default=0)
                    buffered = value[-retain:] if retain else ""
                    value = value[:-retain] if retain else value
                    if stopped and chat.session_id is not None:
                        cancel = getattr(backend, "cancel_response", None)
                        if callable(cancel):
                            await cancel(chat.session_id)
                if not value:
                    continue
                if active is None or active[1] != kind:
                    if active is not None:
                        if active[1] == "thinking":
                            yield _event("content_block_delta", index=active[0], delta={"type": "signature_delta", "signature": ""})
                        yield _event("content_block_stop", index=active[0])
                    active = (count, kind)
                    count += 1
                    yield _event("content_block_start", index=active[0], content_block={"type": kind, kind if kind == "thinking" else "text": ""})
                yield _event("content_block_delta", index=active[0], delta={"type": f"{kind}_delta", kind if kind == "thinking" else "text": value})
            for call in delta.get("tool_calls", []) if not stopped else []:
                if active is not None:
                    if active[1] == "thinking":
                        yield _event("content_block_delta", index=active[0], delta={"type": "signature_delta", "signature": ""})
                    yield _event("content_block_stop", index=active[0])
                    active = None
                tool = tool_blocks.setdefault(call["index"], {"index": None, "arguments": "", "pending": "", "started": False})
                tool["id"] = call.get("id") or tool.get("id")
                function = call.get("function", {})
                tool["name"] = function.get("name") or tool.get("name")
                fragment = function.get("arguments", "")
                tool["arguments"] += fragment
                tool["pending"] += fragment
                if tool["id"] and tool["name"]:
                    if not tool["started"]:
                        tool["started"] = True
                        tool["index"] = count
                        count += 1
                        yield _event("content_block_start", index=tool["index"], content_block={"type": "tool_use", "id": tool["id"], "name": tool["name"], "input": {}})
                    if tool["pending"]:
                        yield _event("content_block_delta", index=tool["index"], delta={"type": "input_json_delta", "partial_json": tool["pending"]})
                        tool["pending"] = ""
        if not completed:
            return
        if buffered and not stopped:
            if active is None or active[1] != "text":
                if active is not None:
                    yield _event("content_block_stop", index=active[0])
                active = (count, "text")
                yield _event("content_block_start", index=count, content_block={"type": "text", "text": ""})
            yield _event("content_block_delta", index=active[0], delta={"type": "text_delta", "text": buffered})
        if active is not None:
            if active[1] == "thinking":
                yield _event("content_block_delta", index=active[0], delta={"type": "signature_delta", "signature": ""})
            yield _event("content_block_stop", index=active[0])
        for tool in tool_blocks.values():
            try:
                if not tool["started"] or not isinstance(json.loads(tool["arguments"] or "{}"), dict):
                    raise ValueError("incomplete tool call")
            except ValueError:
                yield _event("error", error={"type": "api_error", "message": "backend returned invalid tool input"})
                return
            yield _event("content_block_stop", index=tool["index"])
        yield _event("message_delta", delta={"stop_reason": _finish(reason, stopped, bool(tool_blocks)), "stop_sequence": stopped}, usage=_usage(usage))
        yield _event("message_stop")
    finally:
        await iterator.aclose()
        close = getattr(backend, "close_session", None)
        if own_session and callable(close):
            with suppress(BackendError):
                await close(chat.session_id)
