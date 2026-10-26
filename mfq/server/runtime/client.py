"""Transport-neutral client for an MFQ native runtime."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from itertools import count
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

_logger = logging.getLogger(__name__)


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


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield complete SSE data fields while ignoring comments and metadata."""

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


_SAMPLING_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "enable_vision",
    "enable_mtp",
    "mtp_max_draft_tokens",
    "seed",
)


def _runtime_generate_params(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the Python API request into the private runtime contract."""

    input_: dict[str, Any] = {"messages": payload.get("messages", [])}
    if "mfq_preformatted_prompt" in payload:
        input_["preformatted_prompt"] = payload["mfq_preformatted_prompt"]
    if "mfq_benchmark_prompt_tokens" in payload:
        input_["benchmark_prompt_tokens"] = payload["mfq_benchmark_prompt_tokens"]
    sampling = {
        field: payload[field]
        for field in _SAMPLING_FIELDS
        if field in payload
    }
    if "max_tokens" in payload:
        sampling["max_new_tokens"] = payload["max_tokens"]
    stream_options = payload.get("stream_options")
    params: dict[str, Any] = {
        "model": payload.get("model"),
        "input": input_,
        "sampling": sampling,
        "stream": bool(payload.get("stream", False)),
        "include_usage": bool(
            isinstance(stream_options, dict)
            and stream_options.get("include_usage") is True
        ),
    }
    if "reasoning_format" in payload:
        params["output"] = {"reasoning_format": payload["reasoning_format"]}
    if "chat_template_kwargs" in payload:
        params["template"] = payload["chat_template_kwargs"]
    for field in ("tools", "tool_choice", "response_format"):
        if field in payload:
            params[field] = payload[field]
    if "mfq_session_id" in payload:
        params["session_id"] = payload["mfq_session_id"]
    if "mfq_multimodal" in payload:
        params["media"] = payload["mfq_multimodal"]
    if "mfq_prefix_cache_enabled" in payload:
        params["cache"] = {"enabled": payload["mfq_prefix_cache_enabled"]}
    if "priority" in payload:
        params["priority"] = payload["priority"]
    if "mfq_timeout_seconds" in payload:
        params["timeout_seconds"] = payload["mfq_timeout_seconds"]
    return params


def _openai_generation_event(event: dict[str, Any]) -> dict[str, Any]:
    """Adapt a runtime generation event at the Python-owned API boundary."""

    if "choices" in event:  # Compatibility with older private runtimes.
        return event
    if "error" in event:
        # Stream errors keep the native code, status, and retryability so a
        # worker-side deadline is not flattened into a protocol failure.
        return event
    request_id = event.get("request_id")
    base: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": event.get("created", 0),
        "model": event.get("model"),
    }
    kind = event.get("event")
    if kind == "delta":
        delta = event.get("delta")
        if not isinstance(delta, dict):
            raise BackendProtocolError("runtime delta event requires an object delta")
        base["choices"] = [
            {"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}
        ]
    elif kind == "complete":
        finish_reason = event.get("finish_reason")
        base["choices"] = [
            {
                "index": 0,
                "delta": {},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ]
        if "metrics" in event:
            base["mfq_metrics"] = event["metrics"]
    elif kind == "usage":
        base["choices"] = []
        base["usage"] = event.get("usage")
    elif isinstance(event.get("output"), dict):
        output = event["output"]
        delta: dict[str, Any] = {"content": output.get("text", "")}
        if "reasoning" in output:
            delta["reasoning_content"] = output["reasoning"]
        if "tool_calls" in output:
            delta["tool_calls"] = output["tool_calls"]
        base["choices"] = [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": event.get("finish_reason"),
            }
        ]
        if "usage" in event:
            base["usage"] = event["usage"]
        if "metrics" in event:
            base["mfq_metrics"] = event["metrics"]
    else:
        raise BackendProtocolError(
            f"unknown runtime generation event {kind!r}"
        )
    return base


def _openai_models(payload: dict[str, Any]) -> dict[str, Any]:
    if "data" in payload:  # Compatibility with older private runtimes.
        return payload
    models = payload.get("models")
    if not isinstance(models, list):
        raise BackendProtocolError("runtime models response requires a models array")
    return {
        "object": "list",
        "data": [
            {
                "id": model.get("name"),
                "object": "model",
                "created": 0,
                "owned_by": "mfq",
            }
            for model in models
            if isinstance(model, dict)
        ],
    }


class RuntimeClient(Protocol):
    """Raw runtime operations independent of HTTP, IPC, or TCP framing."""

    def generate(
        self,
        payload: dict[str, Any],
    ) -> AbstractAsyncContextManager[AsyncIterator[dict[str, Any] | None]]: ...

    async def health(self) -> dict[str, Any]: ...
    async def count_tokens(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def score(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def status(self) -> dict[str, Any]: ...

    async def models(self) -> dict[str, Any]: ...

    async def fork_session(self, source_session_id: str, target_session_id: str) -> bool: ...

    async def close_session(self, session_id: str) -> bool: ...

    async def cancel_response(self, session_id: str) -> bool: ...

    async def realtime_capabilities(self) -> dict[str, Any]: ...

    async def reload(self, context_size: int) -> dict[str, Any]: ...

    async def clear_cache(self) -> dict[str, Any]: ...

    async def trim_cache(self, target_bytes: int) -> dict[str, Any]: ...

    async def set_cache_budget(self, target_bytes: int, disk_target_bytes: int | None = None, ram_target_bytes: int | None = None) -> dict[str, Any]: ...
    async def refresh_cache_index(self) -> dict[str, Any]: ...
    async def set_resident_memory_budget(self, target_bytes: int, weight_target_bytes: int | None = None) -> dict[str, Any]: ...

    def realtime_connect(self, *, mode: str = "audio") -> Any: ...

    async def aclose(self) -> None: ...


class HttpRuntimeClient:
    """HTTP/SSE/WebSocket implementation of ``RuntimeClient``."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        client: httpx.AsyncClient | None = None,
        control_timeout_seconds: float = 30.0,
        long_control_timeout_seconds: float = 1800.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.control_timeout_seconds = max(0.25, control_timeout_seconds)
        self.long_control_timeout_seconds = max(
            self.control_timeout_seconds,
            long_control_timeout_seconds,
        )
        self._owns_client = client is None
        hostname = urlsplit(self.base_url).hostname
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
            # Native runtimes are private loopback workers.  Routing those
            # requests through a desktop/system HTTP proxy makes readiness
            # checks hang and could expose local inference traffic.
            trust_env=hostname not in {"127.0.0.1", "localhost", "::1"},
        )

    @asynccontextmanager
    async def generate(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[AsyncIterator[dict[str, Any] | None]]:
        headers = self._headers()
        headers["Accept"] = "text/event-stream"
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/runtime/generate",
                json=_runtime_generate_params(payload),
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    raise self._http_error(response.status_code, await response.aread())
                content_type = response.headers.get("content-type", "").lower()
                if "text/event-stream" not in content_type:
                    raise BackendProtocolError(
                        f"backend returned unexpected content type {content_type!r}"
                    )
                events = self._stream_events(response)
                try:
                    yield events
                finally:
                    await events.aclose()
        except BackendError:
            raise
        except httpx.TimeoutException as error:
            raise BackendError("backend_timeout", str(error), retryable=True) from error
        except httpx.HTTPError as error:
            raise BackendError(
                "backend_connection_error",
                str(error),
                retryable=True,
            ) from error

    async def health(self) -> dict[str, Any]:
        return await self._json_request("GET", "/runtime/health")

    async def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._json_request("POST", "/runtime/score", json_body=payload, timeout_seconds=self.long_control_timeout_seconds)

    async def count_tokens(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._json_request("POST", "/runtime/count_tokens", json_body=payload)

    async def status(self) -> dict[str, Any]:
        return await self._json_request("GET", "/runtime/status")

    async def models(self) -> dict[str, Any]:
        return _openai_models(await self._json_request("GET", "/runtime/models"))

    async def fork_session(self, source_session_id: str, target_session_id: str) -> bool:
        return await self._session_control_request(
            "POST",
            "/runtime/sessions/fork",
            json_body={
                "source_session_id": source_session_id,
                "target_session_id": target_session_id,
            },
        )

    async def close_session(self, session_id: str) -> bool:
        return await self._session_control_request(
            "DELETE",
            f"/runtime/sessions/{session_id}",
        )

    async def cancel_response(self, session_id: str) -> bool:
        payload = await self._json_request(
            "POST",
            f"/runtime/sessions/{session_id}/cancel",
            timeout_seconds=min(1.0, self.control_timeout_seconds),
        )
        return payload.get("cancelled") is True

    async def realtime_capabilities(self) -> dict[str, Any]:
        return await self._json_request("GET", "/runtime/realtime/capabilities")

    async def reload(self, context_size: int) -> dict[str, Any]:
        return await self._json_request(
            "POST",
            "/runtime/reload",
            json_body={"context_size": context_size},
            timeout_seconds=self.long_control_timeout_seconds,
        )

    async def clear_cache(self) -> dict[str, Any]:
        return await self._json_request(
            "POST",
            "/runtime/cache/clear",
            timeout_seconds=self.long_control_timeout_seconds,
        )

    async def trim_cache(self, target_bytes: int) -> dict[str, Any]:
        return await self._json_request(
            "POST",
            "/runtime/cache/trim",
            json_body={"target_bytes": target_bytes},
        )

    async def set_cache_budget(self, target_bytes: int, disk_target_bytes: int | None = None, ram_target_bytes: int | None = None) -> dict[str, Any]:
        body = {"target_bytes": target_bytes}
        if disk_target_bytes is not None:
            body["disk_target_bytes"] = disk_target_bytes
        if ram_target_bytes is not None:
            body["ram_target_bytes"] = ram_target_bytes
        return await self._json_request("POST", "/runtime/cache/budget", json_body=body)

    async def refresh_cache_index(self) -> dict[str, Any]:
        return await self._json_request("POST", "/runtime/cache/refresh", json_body={})

    async def set_resident_memory_budget(self, target_bytes: int, weight_target_bytes: int | None = None) -> dict[str, Any]:
        body = {"target_bytes": target_bytes}
        if weight_target_bytes is not None:
            body["weight_target_bytes"] = weight_target_bytes
        return await self._json_request("POST", "/runtime/memory/budget", json_body=body)

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        import websockets

        parsed = urlsplit(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        url = urlunsplit(
            (scheme, parsed.netloc, "/runtime/realtime", f"mode={mode}", "")
        )
        headers = self._headers() or None
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

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _json_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(
                method,
                f"{self.base_url}{path}",
                json=json_body,
                headers=self._headers(),
                timeout=self._control_timeout(
                    self.control_timeout_seconds
                    if timeout_seconds is None
                    else timeout_seconds
                ),
            )
            if response.status_code >= 400:
                raise self._http_error(response.status_code, response.content)
            payload = response.json()
        except BackendError:
            raise
        except httpx.TimeoutException as error:
            raise BackendError("backend_timeout", str(error), retryable=True) from error
        except httpx.HTTPError as error:
            raise BackendError(
                "backend_connection_error",
                str(error),
                retryable=True,
            ) from error
        except ValueError as error:
            raise BackendProtocolError(str(error)) from error
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
        try:
            response = await self._client.request(
                method,
                f"{self.base_url}{path}",
                json=json_body,
                headers=self._headers(),
                timeout=self._control_timeout(self.control_timeout_seconds),
            )
        except httpx.HTTPError:
            return False
        return bool(200 <= response.status_code < 300)

    async def _stream_events(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[dict[str, Any] | None]:
        async for data in iter_sse_data(response):
            if data == "[DONE]":
                yield None
                return
            try:
                event = json.loads(data)
            except json.JSONDecodeError as error:
                raise BackendProtocolError(
                    f"backend returned invalid SSE JSON: {error}"
                ) from error
            if not isinstance(event, dict):
                raise BackendProtocolError("backend SSE data must be a JSON object")
            yield _openai_generation_event(event)
        raise BackendProtocolError("backend stream ended without [DONE]")

    @staticmethod
    def _control_timeout(seconds: float) -> httpx.Timeout:
        return httpx.Timeout(
            seconds,
            connect=min(5.0, seconds),
            write=min(30.0, seconds),
            pool=min(5.0, seconds),
        )

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


class StdioRuntimeClient:
    """NDJSON implementation of ``RuntimeClient`` for one managed process."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        control_timeout_seconds: float = 30.0,
        long_control_timeout_seconds: float = 1800.0,
        max_frame_bytes: int = 64 * 1024 * 1024,
        pending_event_limit: int = 64,
        capture_stderr: bool = True,
    ) -> None:
        if process.stdin is None or process.stdout is None:
            raise ValueError("stdio runtime process requires stdin and stdout pipes")
        self._process = process
        self.control_timeout_seconds = max(0.25, control_timeout_seconds)
        self.long_control_timeout_seconds = max(
            self.control_timeout_seconds,
            long_control_timeout_seconds,
        )
        self.max_frame_bytes = max_frame_bytes
        self.pending_event_limit = max(1, pending_event_limit)
        self._ids = count(1)
        self._write_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Queue[dict[str, Any] | BaseException]] = {}
        self._ready = asyncio.Event()
        self._reader_task: asyncio.Task[None] | None = None
        self._terminal_error: BackendError | None = None
        self._closed = False
        self._capture_stderr = capture_stderr
        self._stderr_tail = bytearray()
        self._stderr_done = asyncio.Event()
        self._stderr_task: asyncio.Task[None] | None = None
        if getattr(process, "stderr", None) is None:
            self._stderr_done.set()

    def record_stderr(self, data: bytes) -> None:
        """Retain a bounded tail; the pool may own the stderr reader/log sink."""
        self._stderr_tail.extend(data)
        del self._stderr_tail[:-32768]

    def finish_stderr(self) -> None:
        self._stderr_done.set()

    async def _read_stderr(self) -> None:
        try:
            while data := await self._process.stderr.read(4096):
                self.record_stderr(data)
        finally:
            self.finish_stderr()

    async def _closed_stdout_error(self) -> BackendError:
        status = getattr(self._process, "returncode", None)
        if status is None and callable(getattr(self._process, "wait", None)):
            with suppress(asyncio.TimeoutError):
                status = await asyncio.wait_for(self._process.wait(), timeout=1.0)
        # stdout/stderr EOF can race. Allow the independent stderr reader to
        # retain the final fatal report before releasing all pending requests.
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stderr_done.wait(), timeout=1.0)
        description = "stdio runtime closed stdout"
        if status is not None:
            description = f"runtime process exited with status {status} (0x{status & 0xFFFFFFFF:08X})"
        tail = self._stderr_tail.decode("utf-8", errors="replace").strip()
        message = description + (f"; last stderr:\n{tail}" if tail else "; no stderr was captured")
        _logger.error("%s", message)
        cause = next((line for line in tail.splitlines() if line.startswith("runtime_fatal error=")), "")
        if cause:
            description += f"; {cause[:512]}"
        return BackendError("backend_connection_error", description, retryable=True, status_code=503)

    @asynccontextmanager
    async def generate(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[AsyncIterator[dict[str, Any] | None]]:
        request_id, queue = await self._open(
            "generate", _runtime_generate_params(payload)
        )
        finished = False

        async def events() -> AsyncIterator[dict[str, Any] | None]:
            nonlocal finished
            while True:
                frame = await queue.get()
                if isinstance(frame, BaseException):
                    raise frame
                frame_type = frame.get("type")
                if frame_type == "event":
                    data = frame.get("data")
                    if not isinstance(data, dict):
                        raise BackendProtocolError("stdio event data must be a JSON object")
                    yield _openai_generation_event(data)
                elif frame_type == "done":
                    finished = True
                    yield None
                    return
                elif frame_type == "error":
                    finished = True
                    raise self._frame_error(frame)
                else:
                    raise BackendProtocolError(
                        f"unexpected stdio generation frame type {frame_type!r}"
                    )

        stream = events()
        try:
            yield stream
        finally:
            await stream.aclose()
            self._pending.pop(request_id, None)
            if not finished and not self._closed:
                with suppress(Exception):
                    await self._notify(
                        "request.cancel",
                        {"target_id": request_id},
                    )

    async def health(self) -> dict[str, Any]:
        return await self._unary("health")

    async def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._unary("score", payload, timeout_seconds=self.long_control_timeout_seconds)

    async def count_tokens(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._unary("count_tokens", payload)

    async def status(self) -> dict[str, Any]:
        return await self._unary("status")

    async def models(self) -> dict[str, Any]:
        return _openai_models(await self._unary("models"))

    async def fork_session(self, source_session_id: str, target_session_id: str) -> bool:
        result = await self._unary(
            "session.fork",
            {
                "source_session_id": source_session_id,
                "target_session_id": target_session_id,
            },
        )
        return result.get("status") == "ok"

    async def close_session(self, session_id: str) -> bool:
        try:
            result = await self._unary("session.close", {"session_id": session_id})
        except BackendError:
            return False
        return result.get("status") == "ok"

    async def cancel_response(self, session_id: str) -> bool:
        result = await self._unary(
            "session.cancel",
            {"session_id": session_id},
            timeout_seconds=min(1.0, self.control_timeout_seconds),
        )
        return result.get("cancelled") is True

    async def realtime_capabilities(self) -> dict[str, Any]:
        return await self._unary("realtime.capabilities")

    async def reload(self, context_size: int) -> dict[str, Any]:
        return await self._unary(
            "reload",
            {"context_size": context_size},
            timeout_seconds=self.long_control_timeout_seconds,
        )

    async def clear_cache(self) -> dict[str, Any]:
        return await self._unary(
            "cache.clear",
            timeout_seconds=self.long_control_timeout_seconds,
        )

    async def trim_cache(self, target_bytes: int) -> dict[str, Any]:
        return await self._unary("cache.trim", {"target_bytes": target_bytes})

    async def set_cache_budget(self, target_bytes: int, disk_target_bytes: int | None = None, ram_target_bytes: int | None = None) -> dict[str, Any]:
        body = {"target_bytes": target_bytes}
        if disk_target_bytes is not None:
            body["disk_target_bytes"] = disk_target_bytes
        if ram_target_bytes is not None:
            body["ram_target_bytes"] = ram_target_bytes
        return await self._unary("cache.budget", body)

    async def refresh_cache_index(self) -> dict[str, Any]:
        return await self._unary("cache.refresh", {})

    async def set_resident_memory_budget(self, target_bytes: int, weight_target_bytes: int | None = None) -> dict[str, Any]:
        body = {"target_bytes": target_bytes}
        if weight_target_bytes is not None:
            body["weight_target_bytes"] = weight_target_bytes
        return await self._unary("memory.budget", body)

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        return _StdioRealtimeConnection(self, mode)

    async def aclose(self) -> None:
        if self._closed:
            return
        with suppress(Exception):
            await self._notify("shutdown", {})
        self._closed = True
        stdin = self._process.stdin
        if stdin is not None:
            stdin.close()
            with suppress(Exception):
                await stdin.wait_closed()
        task = self._reader_task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._fail_pending(
            BackendError(
                "backend_connection_error",
                "stdio runtime client closed",
                retryable=True,
            )
        )

    async def _open(
        self,
        op: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[str, asyncio.Queue[dict[str, Any] | BaseException]]:
        await self._ensure_ready()
        request_id = str(next(self._ids))
        queue: asyncio.Queue[dict[str, Any] | BaseException] = asyncio.Queue(
            self.pending_event_limit
        )
        self._pending[request_id] = queue
        try:
            await self._write(
                {"v": 1, "id": request_id, "op": op, "params": params or {}}
            )
        except BaseException:
            self._pending.pop(request_id, None)
            raise
        return request_id, queue

    async def _unary(
        self,
        op: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        request_id, queue = await self._open(op, params)
        timeout = self.control_timeout_seconds if timeout_seconds is None else timeout_seconds
        try:
            frame = await asyncio.wait_for(queue.get(), timeout=timeout)
            if isinstance(frame, BaseException):
                raise frame
            frame_type = frame.get("type")
            if frame_type == "error":
                raise self._frame_error(frame)
            if frame_type != "result" or not isinstance(frame.get("data"), dict):
                raise BackendProtocolError(
                    f"unexpected stdio response for {op}: {frame_type!r}"
                )
            return frame["data"]
        except asyncio.TimeoutError as error:
            with suppress(Exception):
                await self._notify("request.cancel", {"target_id": request_id})
            raise BackendError("backend_timeout", f"stdio {op} timed out", retryable=True) from error
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, op: str, params: dict[str, Any]) -> None:
        await self._ensure_ready()
        await self._write({"v": 1, "op": op, "params": params})

    async def _ensure_ready(self) -> None:
        if self._capture_stderr and self._stderr_task is None and getattr(self._process, "stderr", None) is not None:
            self._stderr_task = asyncio.create_task(self._read_stderr(), name=f"mfq-stderr-{self._process.pid}")
        if self._reader_task is None:
            self._reader_task = asyncio.create_task(
                self._read_frames(),
                name=f"mfq-stdio-runtime-{self._process.pid}",
            )
        await self._ready.wait()
        if self._terminal_error is not None:
            raise self._terminal_error
        if self._closed:
            raise BackendError(
                "backend_connection_error",
                "stdio runtime client is closed",
                retryable=True,
            )

    async def _write(self, frame: dict[str, Any]) -> None:
        if self._terminal_error is not None:
            raise self._terminal_error
        encoded = (json.dumps(frame, separators=(",", ":")) + "\n").encode()
        if len(encoded) > self.max_frame_bytes:
            raise BackendProtocolError("stdio request exceeds the frame size limit")
        stdin = self._process.stdin
        if stdin is None or stdin.is_closing():
            raise BackendError(
                "backend_connection_error",
                "stdio runtime stdin is closed",
                retryable=True,
            )
        try:
            async with self._write_lock:
                stdin.write(encoded)
                await stdin.drain()
        except (BrokenPipeError, ConnectionError) as error:
            raise BackendError(
                "backend_connection_error",
                str(error),
                retryable=True,
            ) from error

    async def _read_frames(self) -> None:
        stdout = self._process.stdout
        assert stdout is not None
        try:
            while True:
                raw = await stdout.readline()
                if not raw:
                    raise await self._closed_stdout_error()
                if len(raw) > self.max_frame_bytes:
                    raise BackendProtocolError("stdio response exceeds the frame size limit")
                try:
                    frame = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise BackendProtocolError(f"invalid stdio NDJSON frame: {error}") from error
                if not isinstance(frame, dict) or frame.get("v") != 1:
                    raise BackendProtocolError("stdio response has an unsupported protocol version")
                if frame.get("type") == "ready" and "id" not in frame:
                    self._ready.set()
                    continue
                request_id = frame.get("id")
                if not isinstance(request_id, str):
                    raise BackendProtocolError("stdio response requires a string id")
                queue = self._pending.get(request_id)
                if queue is not None:
                    try:
                        queue.put_nowait(frame)
                    except asyncio.QueueFull:
                        # One stalled stream must not block every control reply.
                        self._pending.pop(request_id, None)
                        while not queue.empty():
                            queue.get_nowait()
                        queue.put_nowait(
                            BackendError(
                                "backend_backpressure",
                                "stdio request response queue exceeded its limit",
                                retryable=True,
                            )
                        )
        except asyncio.CancelledError:
            raise
        except BackendError as error:
            self._terminal_error = error
        except Exception as error:
            self._terminal_error = BackendProtocolError(str(error))
        finally:
            self._ready.set()
            if self._terminal_error is None and not self._closed:
                self._terminal_error = BackendError(
                    "backend_connection_error",
                    "stdio runtime reader stopped",
                    retryable=True,
                )
            if self._terminal_error is not None:
                self._fail_pending(self._terminal_error)

    def _fail_pending(self, error: BackendError) -> None:
        for queue in tuple(self._pending.values()):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with suppress(asyncio.QueueFull):
                queue.put_nowait(error)

    @staticmethod
    def _frame_error(frame: dict[str, Any]) -> BackendError:
        detail = frame.get("error")
        if not isinstance(detail, dict):
            return BackendProtocolError("stdio error frame requires an error object")
        raw_status = detail.get("status_code")
        status_code = raw_status if isinstance(raw_status, int) else None
        return BackendError(
            str(detail.get("code") or "backend_error"),
            str(detail.get("message") or "backend request failed"),
            retryable=detail.get("retryable") is True,
            status_code=status_code,
        )


class _StdioRealtimeConnection:
    def __init__(self, client: StdioRuntimeClient, mode: str) -> None:
        self._client = client
        self._mode = mode
        self._request_id: str | None = None
        self._queue: asyncio.Queue[dict[str, Any] | BaseException] | None = None

    async def __aenter__(self) -> _StdioRealtimeConnection:
        request_id, queue = await self._client._open(
            "realtime.open",
            {"mode": self._mode},
        )
        frame = await queue.get()
        if isinstance(frame, BaseException):
            raise frame
        if frame.get("type") == "error":
            raise self._client._frame_error(frame)
        if frame.get("type") != "result":
            raise BackendProtocolError("realtime.open did not return a result")
        self._request_id = request_id
        self._queue = queue
        return self

    async def __aexit__(self, *_error: object) -> None:
        request_id = self._request_id
        self._request_id = None
        if request_id is None:
            return
        with suppress(Exception):
            await self._client._unary(
                "realtime.close",
                {"target_id": request_id},
            )
        self._client._pending.pop(request_id, None)

    async def send(self, message: str | bytes) -> None:
        if not isinstance(message, str):
            raise BackendProtocolError("stdio realtime accepts JSON text messages only")
        if self._request_id is None:
            raise BackendProtocolError("stdio realtime channel is not open")
        await self._client._unary(
            "realtime.send",
            {"target_id": self._request_id, "data": message},
            timeout_seconds=self._client.long_control_timeout_seconds,
        )

    def __aiter__(self) -> _StdioRealtimeConnection:
        return self

    async def __anext__(self) -> str:
        queue = self._queue
        if queue is None:
            raise StopAsyncIteration
        frame = await queue.get()
        if isinstance(frame, BaseException):
            raise frame
        frame_type = frame.get("type")
        if frame_type == "event" and isinstance(frame.get("data"), str):
            return frame["data"]
        if frame_type == "error":
            raise self._client._frame_error(frame)
        if frame_type == "done":
            raise StopAsyncIteration
        raise BackendProtocolError(
            f"unexpected stdio realtime frame type {frame_type!r}"
        )
