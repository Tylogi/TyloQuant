"""Streaming adapter for the existing MFQ OpenAI-compatible text server."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

import httpx

from mfqd.capabilities import capabilities_for_architecture
from mfqd.models import (
    ModelCapabilities,
    RuntimeCapabilitiesResource,
    SamplingParams,
    TokenUsage,
)


class BackendError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class BackendProtocolError(BackendError):
    def __init__(self, message: str) -> None:
        super().__init__("backend_protocol_error", message)


@dataclass(frozen=True)
class BackendToolCallDelta:
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True)
class BackendDelta:
    content_delta: str = ""
    reasoning_delta: str = ""
    tool_calls: tuple[BackendToolCallDelta, ...] = ()
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class ChatBackend(Protocol):
    def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
    ) -> AsyncIterator[BackendDelta]: ...

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool: ...

    async def close_session(self, session_id: UUID) -> bool: ...

    async def capabilities(self) -> RuntimeCapabilitiesResource: ...

    async def aclose(self) -> None: ...


class OpenAIChatBackend:
    """Forward text generation to MFQ's existing C++ streaming endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
        )

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
    ) -> AsyncIterator[BackendDelta]:
        payload = {
            "model": model,
            "messages": list(messages),
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "top_k": sampling.top_k,
            "top_p": sampling.top_p,
            "presence_penalty": sampling.presence_penalty,
            "frequency_penalty": sampling.frequency_penalty,
            "repetition_penalty": sampling.repetition_penalty,
            "stream": True,
            "stream_options": {"include_usage": True},
            "reasoning_format": "auto",
        }
        if sampling.seed is not None:
            payload["seed"] = sampling.seed
        if session_id is not None:
            payload["mfq_session_id"] = str(session_id)
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise self._http_error(response.status_code, body)
                content_type = response.headers.get("content-type", "").lower()
                if "text/event-stream" not in content_type:
                    raise BackendProtocolError(
                        f"backend returned unexpected content type {content_type!r}"
                    )
                saw_done = False
                async for data in self._iter_sse_data(response):
                    if data == "[DONE]":
                        saw_done = True
                        break
                    yield self._parse_event(data)
                if not saw_done:
                    raise BackendProtocolError("backend stream ended without [DONE]")
        except BackendError:
            raise
        except httpx.TimeoutException as error:
            raise BackendError("backend_timeout", str(error), retryable=True) from error
        except httpx.HTTPError as error:
            raise BackendError("backend_connection_error", str(error), retryable=True) from error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool:
        return await self._session_control_request(
            "POST",
            "/api/runtime/sessions/fork",
            json_body={
                "source_session_id": str(source_session_id),
                "target_session_id": str(target_session_id),
            },
        )

    async def close_session(self, session_id: UUID) -> bool:
        return await self._session_control_request(
            "DELETE",
            f"/api/runtime/sessions/{session_id}",
        )

    async def capabilities(self) -> RuntimeCapabilitiesResource:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = await self._client.get(
                f"{self.base_url}/health",
                headers=headers,
            )
            if response.status_code >= 400:
                raise self._http_error(response.status_code, response.content)
            payload = response.json()
        except BackendError:
            raise
        except (httpx.HTTPError, ValueError) as error:
            raise BackendError(
                "backend_capabilities_unavailable",
                str(error),
                retryable=True,
            ) from error
        if not isinstance(payload, dict):
            raise BackendProtocolError("backend health must be a JSON object")
        model = str(payload.get("model") or "mfq-model")
        model_type = str(payload.get("model_type") or "unknown")
        raw_capabilities = payload.get("model_capabilities")
        try:
            capabilities = (
                ModelCapabilities.model_validate(raw_capabilities)
                if isinstance(raw_capabilities, dict)
                else capabilities_for_architecture(model_type)
            )
        except ValueError as error:
            raise BackendProtocolError(
                f"backend returned invalid model capabilities: {error}"
            ) from error
        return RuntimeCapabilitiesResource(
            model=model,
            model_type=model_type,
            model_capabilities=capabilities,
            duplex_available=payload.get("duplex_available") is True,
        )

    async def _session_control_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, str] | None = None,
    ) -> bool:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = await self._client.request(
                method,
                f"{self.base_url}{path}",
                json=json_body,
                headers=headers,
            )
        except httpx.HTTPError:
            return False
        return bool(200 <= response.status_code < 300)

    @staticmethod
    async def _iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if line == "":
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)
        if data_lines:
            yield "\n".join(data_lines)

    @staticmethod
    def _parse_event(data: str) -> BackendDelta:
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise BackendProtocolError(f"backend returned invalid SSE JSON: {error}") from error
        if not isinstance(event, dict):
            raise BackendProtocolError("backend SSE data must be a JSON object")
        if "error" in event:
            detail = event["error"]
            if isinstance(detail, dict):
                raise BackendError(
                    str(detail.get("code") or detail.get("type") or "backend_error"),
                    str(detail.get("message") or "backend request failed"),
                )
            raise BackendError("backend_error", str(detail))

        usage = None
        raw_usage = event.get("usage")
        if raw_usage is not None:
            try:
                usage = TokenUsage.model_validate(raw_usage)
            except ValueError as error:
                raise BackendProtocolError(f"invalid backend usage object: {error}") from error

        choices = event.get("choices")
        if choices == []:
            return BackendDelta(usage=usage)
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise BackendProtocolError("backend SSE event must contain one choice or usage")
        choice = choices[0]
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise BackendProtocolError("backend choice delta must be an object")
        tool_calls = OpenAIChatBackend._parse_tool_call_deltas(delta.get("tool_calls"))
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise BackendProtocolError("backend text deltas must be strings")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise BackendProtocolError("backend finish_reason must be a string or null")
        return BackendDelta(
            content_delta=content,
            reasoning_delta=reasoning,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _parse_tool_call_deltas(value: Any) -> tuple[BackendToolCallDelta, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise BackendProtocolError("backend tool_calls delta must be an array")
        parsed: list[BackendToolCallDelta] = []
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                raise BackendProtocolError("backend tool call delta requires an integer index")
            function = item.get("function") or {}
            if not isinstance(function, dict):
                raise BackendProtocolError("backend tool call function delta must be an object")
            call_id = item.get("id")
            name = function.get("name")
            arguments = function.get("arguments") or ""
            if call_id is not None and not isinstance(call_id, str):
                raise BackendProtocolError("backend tool call id must be a string")
            if name is not None and not isinstance(name, str):
                raise BackendProtocolError("backend tool call name must be a string")
            if not isinstance(arguments, str):
                raise BackendProtocolError("backend tool call arguments delta must be a string")
            parsed.append(
                BackendToolCallDelta(
                    index=item["index"],
                    call_id=call_id,
                    name=name,
                    arguments_delta=arguments,
                )
            )
        return tuple(parsed)

    @staticmethod
    def _http_error(status_code: int, body: bytes) -> BackendError:
        code = f"backend_http_{status_code}"
        message = body.decode("utf-8", errors="replace") or f"backend returned HTTP {status_code}"
        try:
            parsed = json.loads(body)
            detail = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(detail, dict):
                code = str(detail.get("code") or detail.get("type") or code)
                message = str(detail.get("message") or message)
        except json.JSONDecodeError:
            pass
        return BackendError(code, message, retryable=status_code in {429, 502, 503, 504})
