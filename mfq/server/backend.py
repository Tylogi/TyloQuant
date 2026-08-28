"""Streaming adapter for the existing MFQ OpenAI-compatible text server."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import httpx

from mfq.server.capabilities import capabilities_for_architecture
from mfq.server.models import (
    ModelCapabilities,
    ResponseFormat,
    ResponsePerformance,
    RuntimeCapabilitiesResource,
    SamplingParams,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
)
from mfq.server.vision import MiniCPMO45VisionProcessor, VisionProcessingError


class BackendError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


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
    performance: ResponsePerformance | None = None
    backend_request_id: str | None = None


class ChatBackend(Protocol):
    def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> AsyncIterator[BackendDelta]: ...

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool: ...

    async def close_session(self, session_id: UUID) -> bool: ...

    async def cancel_response(self, session_id: UUID) -> bool: ...

    async def capabilities(self) -> RuntimeCapabilitiesResource: ...

    async def runtime_status(self) -> dict[str, Any]: ...

    async def runtime_models(self) -> dict[str, Any]: ...

    async def realtime_capabilities(self) -> dict[str, Any]: ...

    async def reload_runtime(self, context_size: int) -> dict[str, Any]: ...

    async def clear_runtime_cache(self) -> dict[str, Any]: ...

    def realtime_connect(self, *, mode: str = "audio") -> Any: ...

    async def aclose(self) -> None: ...


class OpenAIChatBackend:
    """Forward text generation to MFQ's existing C++ streaming endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        client: httpx.AsyncClient | None = None,
        avfoundation_video_library: str | Path | None = None,
        local_tensor_files: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._owns_client = client is None
        hostname = urlsplit(self.base_url).hostname
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
            # Native runtimes are private loopback workers.  Routing those
            # requests through a desktop/system HTTP proxy makes readiness
            # checks hang and could expose local inference traffic.
            trust_env=hostname not in {"127.0.0.1", "localhost", "::1"},
        )
        self._model_type: str | None = None
        self._vision_processor = MiniCPMO45VisionProcessor(
            avfoundation_library=avfoundation_video_library
        )
        self._local_tensor_files = local_tensor_files and os.name == "posix"
        self._runtime_metric_overrides: dict[str, dict[str, float]] = {}

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> AsyncIterator[BackendDelta]:
        backend_messages = list(messages)
        multimodal: dict[str, Any] | None = None
        cleanup_paths: tuple[Path, ...] = ()
        processor_ms = 0.0
        if self._contains_media(backend_messages):
            if self._model_type is None:
                await self.capabilities()
            if self._model_type == "minicpmo":
                processor_started = time.perf_counter()
                try:
                    processed = await asyncio.to_thread(
                        self._vision_processor.prepare_openai_messages,
                        backend_messages,
                        use_binary_file=self._local_tensor_files,
                    )
                except VisionProcessingError as error:
                    raise BackendError("media_processing_error", str(error)) from error
                processor_ms = (time.perf_counter() - processor_started) * 1000.0
                if processed is not None:
                    backend_messages = processed.messages
                    multimodal = processed.tensors
                    cleanup_paths = processed.cleanup_paths
            else:
                raise BackendError(
                    "media_processing_unsupported",
                    f"the loaded {self._model_type or 'unknown'} model has no registered media processor",
                )
        payload = {
            "model": model,
            "messages": backend_messages,
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
            "chat_template_kwargs": {
                "enable_thinking": sampling.enable_thinking,
            },
        }
        if sampling.reasoning_effort:
            payload["chat_template_kwargs"]["reasoning_effort"] = sampling.reasoning_effort
        if sampling.seed is not None:
            payload["seed"] = sampling.seed
        if tools:
            payload["tools"] = [tool.model_dump(mode="json", by_alias=True) for tool in tools]
            payload["tool_choice"] = (
                tool_choice if isinstance(tool_choice, str) else tool_choice.model_dump(mode="json")
            )
        if response_format is not None and response_format.type != "text":
            payload["response_format"] = response_format.model_dump(mode="json", by_alias=True)
        if session_id is not None:
            payload["mfq_session_id"] = str(session_id)
        if multimodal is not None:
            payload["mfq_multimodal"] = multimodal
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
                    delta = self._parse_event(data)
                    if delta.performance is not None and processor_ms > 0.0:
                        performance = self._with_processor_timing(
                            delta.performance,
                            processor_ms,
                        )
                        delta = replace(delta, performance=performance)
                        if delta.backend_request_id:
                            self._remember_runtime_metric_override(
                                delta.backend_request_id,
                                performance,
                            )
                    yield delta
                if not saw_done:
                    raise BackendProtocolError("backend stream ended without [DONE]")
        except BackendError:
            raise
        except httpx.TimeoutException as error:
            raise BackendError("backend_timeout", str(error), retryable=True) from error
        except httpx.HTTPError as error:
            raise BackendError("backend_connection_error", str(error), retryable=True) from error
        finally:
            for path in cleanup_paths:
                path.unlink(missing_ok=True)

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

    async def cancel_response(self, session_id: UUID) -> bool:
        # A stop can race the native request becoming visible after Python-side
        # media preprocessing. Briefly retry so cancellation remains reliable
        # at that boundary without delaying an already-active decode.
        for attempt in range(20):
            payload = await self._json_request(
                "POST",
                f"/api/runtime/sessions/{session_id}/cancel",
            )
            if payload.get("cancelled") is True:
                return True
            if attempt < 19:
                await asyncio.sleep(0.025)
        return False

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
        self._model_type = model_type
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

    @staticmethod
    def _contains_media(messages: Sequence[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if any(
                isinstance(item, dict)
                and item.get("type") in {"image_url", "video_url", "input_audio"}
                for item in content
            ):
                return True
        return False

    async def runtime_status(self) -> dict[str, Any]:
        try:
            status = await self._json_request("GET", "/api/status")
        except BackendError as error:
            if error.code not in {"backend_http_404", "not_found"}:
                raise
            health = await self._json_request("GET", "/health")
            return {**health, "limited": True}
        last_request = status.get("last_request")
        if isinstance(last_request, dict):
            request_id = last_request.get("id")
            override = (
                self._runtime_metric_overrides.get(request_id)
                if isinstance(request_id, str)
                else None
            )
            if override is not None:
                status = dict(status)
                status["last_request"] = {**last_request, **override}
        return status

    @staticmethod
    def _with_processor_timing(
        performance: ResponsePerformance,
        processor_ms: float,
    ) -> ResponsePerformance:
        # The established product-level multimodal Prefill boundary ends at
        # first-token availability.  Keep native model timings intact and add
        # the Python media preparation that happened before the native request.
        complete_prefill_ms = processor_ms + performance.ttft_ms
        complete_prefill_tps = (
            1000.0 * performance.prefill_tokens / complete_prefill_ms
            if performance.prefill_tokens > 0 and complete_prefill_ms > 0.0
            else 0.0
        )
        return performance.model_copy(
            update={
                "processor_ms": processor_ms,
                "complete_prefill_ms": complete_prefill_ms,
                "complete_prefill_tps": complete_prefill_tps,
                "complete_generation_ms": processor_ms + performance.generation_ms,
            }
        )

    def _remember_runtime_metric_override(
        self,
        request_id: str,
        performance: ResponsePerformance,
    ) -> None:
        self._runtime_metric_overrides[request_id] = {
            "processor_ms": performance.processor_ms,
            "complete_prefill_ms": performance.complete_prefill_ms,
            "complete_prefill_tps": performance.complete_prefill_tps,
            "complete_generation_ms": performance.complete_generation_ms,
        }
        while len(self._runtime_metric_overrides) > 128:
            self._runtime_metric_overrides.pop(next(iter(self._runtime_metric_overrides)))

    async def runtime_models(self) -> dict[str, Any]:
        return await self._json_request("GET", "/v1/models")

    async def realtime_capabilities(self) -> dict[str, Any]:
        return await self._json_request("GET", "/realtime/capabilities")

    async def reload_runtime(self, context_size: int) -> dict[str, Any]:
        return await self._json_request(
            "POST",
            "/api/reload",
            json_body={"context_size": context_size},
        )

    async def clear_runtime_cache(self) -> dict[str, Any]:
        return await self._json_request("POST", "/api/runtime/cache/clear")

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        import websockets

        parsed = urlsplit(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        url = urlunsplit((scheme, parsed.netloc, "/v1/realtime", f"mode={mode}", ""))
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        parameters = inspect.signature(websockets.connect).parameters
        options: dict[str, Any] = {"max_size": 128 * 1024 * 1024}
        if "proxy" in parameters:
            options["proxy"] = None
        if headers:
            header_name = (
                "additional_headers" if "additional_headers" in parameters else "extra_headers"
            )
            options[header_name] = headers
        return websockets.connect(url, **options)

    async def _json_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            if response.status_code >= 400:
                raise self._http_error(response.status_code, response.content)
            payload = response.json()
        except BackendError:
            raise
        except httpx.TimeoutException as error:
            raise BackendError("backend_timeout", str(error), retryable=True) from error
        except (httpx.HTTPError, ValueError) as error:
            raise BackendError("backend_protocol_error", str(error), retryable=True) from error
        if not isinstance(payload, dict):
            raise BackendProtocolError(f"backend {path} response must be a JSON object")
        return payload

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
        raw_request_id = event.get("id")
        backend_request_id = raw_request_id if isinstance(raw_request_id, str) else None
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
        performance = None
        raw_performance = event.get("mfq_metrics")
        if raw_performance is not None:
            try:
                performance = ResponsePerformance.model_validate(raw_performance)
            except ValueError as error:
                raise BackendProtocolError(
                    f"invalid backend performance object: {error}"
                ) from error

        choices = event.get("choices")
        if choices == []:
            return BackendDelta(
                usage=usage,
                performance=performance,
                backend_request_id=backend_request_id,
            )
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
            performance=performance,
            backend_request_id=backend_request_id,
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
        return BackendError(
            code,
            message,
            retryable=status_code in {429, 502, 503, 504},
            status_code=status_code,
        )
