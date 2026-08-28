"""Health-aware routing across local and remote MFQ Server nodes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx

from mfq.server.backend import BackendDelta, BackendError, BackendToolCallDelta, ChatBackend
from mfq.server.models import (
    RemoteNodeResource,
    ResponseFormat,
    ResponsePerformance,
    RuntimeCapabilitiesResource,
    SamplingParams,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
)
from mfq.server.storage import SessionStore


@dataclass
class _NodeState:
    resource: RemoteNodeResource
    models: list[str] = field(default_factory=list)
    healthy: bool = False
    active_requests: int = 0
    checked_at: float = 0.0
    checked_at_wall: datetime | None = None
    error: str | None = None
    status: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RemoteSession:
    remote_id: UUID
    revision: int
    synchronized_messages: int


class ClusterBackend:
    """Route matching models to healthy remote MFQ Server nodes and aggregate inventory."""

    def __init__(
        self,
        local: ChatBackend,
        store: SessionStore,
        *,
        health_ttl_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.local = local
        self.store = store
        self.health_ttl_seconds = max(0.25, health_ttl_seconds)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=None, write=30.0, pool=3.0),
            trust_env=False,
        )
        self._states: dict[UUID, _NodeState] = {}
        self._sessions: dict[tuple[UUID, UUID], _RemoteSession] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _headers(node: RemoteNodeResource) -> dict[str, str]:
        if not node.api_key_env:
            return {}
        token = os.environ.get(node.api_key_env, "")
        if not token:
            raise BackendError(
                "remote_node_credential_missing",
                f"environment variable is not set: {node.api_key_env}",
            )
        return {"Authorization": f"Bearer {token}"}

    async def refresh(self, *, force: bool = False) -> list[RemoteNodeResource]:
        resources = await asyncio.to_thread(self.store.list_remote_nodes)
        configured = {item.id for item in resources}
        async with self._lock:
            for stale in set(self._states) - configured:
                self._states.pop(stale, None)
            for resource in resources:
                state = self._states.get(resource.id)
                if state is None:
                    self._states[resource.id] = _NodeState(resource=resource)
                else:
                    state.resource = resource
            states = list(self._states.values())
        await asyncio.gather(
            *(self._probe(state, force=force) for state in states if state.resource.enabled)
        )
        return [self._public(state) for state in states]

    async def _probe(self, state: _NodeState, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - state.checked_at < self.health_ttl_seconds:
            return
        state.checked_at = now
        state.checked_at_wall = datetime.now(timezone.utc)
        try:
            headers = self._headers(state.resource)
            health, models, status = await asyncio.gather(
                self._client.get(f"{state.resource.url}/health", headers=headers),
                self._client.get(f"{state.resource.url}/api/v1/runtime/models", headers=headers),
                self._client.get(f"{state.resource.url}/api/v1/runtime/status", headers=headers),
            )
            health.raise_for_status()
            models.raise_for_status()
            status.raise_for_status()
            payload = models.json()
            data = payload.get("data", []) if isinstance(payload, dict) else []
            state.models = sorted(
                {str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")}
            )
            state.healthy = True
            status_payload = status.json()
            state.status = status_payload if isinstance(status_payload, dict) else {}
            state.error = None
        except Exception as error:
            state.healthy = False
            state.models = []
            state.status = {}
            state.error = str(error)[:512]

    @staticmethod
    def _public(state: _NodeState) -> RemoteNodeResource:
        return state.resource.model_copy(
            update={
                "healthy": state.healthy,
                "models": state.models,
                "active_requests": state.active_requests,
                "metrics": state.status,
                "last_checked_at": state.checked_at_wall,
                "error": state.error,
            }
        )

    async def _select(self, model: str) -> _NodeState | None:
        await self.refresh()
        async with self._lock:
            matches = [
                state
                for state in self._states.values()
                if state.resource.enabled and state.healthy and model in state.models
            ]
        return (
            min(matches, key=lambda item: (item.active_requests, item.resource.name))
            if matches
            else None
        )

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
        node = await self._select(model)
        if node is None:
            async for delta in self.local.stream(
                model=model,
                messages=messages,
                sampling=sampling,
                session_id=session_id,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            ):
                yield delta
            return
        node.active_requests += 1
        try:
            async for delta in self._remote_stream(
                node.resource,
                model=model,
                messages=messages,
                sampling=sampling,
                session_id=session_id,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            ):
                yield delta
        finally:
            node.active_requests = max(0, node.active_requests - 1)

    async def _remote_stream(
        self,
        node: RemoteNodeResource,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None,
        tools: Sequence[ToolDefinition],
        tool_choice: ToolChoice,
        response_format: ResponseFormat | None,
    ) -> AsyncIterator[BackendDelta]:
        headers = self._headers(node)
        key = (node.id, session_id or uuid4())
        remote = self._sessions.get(key)
        if remote is None or remote.synchronized_messages > len(messages):
            created = await self._request_json(
                node,
                "POST",
                "/api/v1/sessions",
                {"model": model, "mode": "text"},
                headers,
            )
            remote = _RemoteSession(
                remote_id=UUID(created["id"]),
                revision=int(created["revision"]),
                synchronized_messages=0,
            )
            self._sessions[key] = remote
        prior = messages[remote.synchronized_messages : -1]
        for message in prior:
            role = str(message.get("role", "user"))
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            parts = await self._message_parts(node, message, headers)
            appended = await self._request_json(
                node,
                "POST",
                f"/api/v1/sessions/{remote.remote_id}/messages",
                {
                    "expected_revision": remote.revision,
                    "role": role,
                    "parts": parts,
                },
                headers,
            )
            remote.revision = int(appended["session"]["revision"])
        last = messages[-1] if messages else {"role": "user", "content": ""}
        input_parts = await self._message_parts(node, last, headers)
        request = {
            "request_id": str(uuid4()),
            "expected_revision": remote.revision,
            "input": input_parts,
            "input_role": "tool" if last.get("role") == "tool" else "user",
            "sampling": sampling.model_dump(mode="json"),
            "include_reasoning_history": True,
            "tools": [item.model_dump(mode="json") for item in tools],
            "tool_choice": tool_choice
            if isinstance(tool_choice, str)
            else tool_choice.model_dump(mode="json"),
            "response_format": (
                response_format.model_dump(mode="json", by_alias=True)
                if response_format is not None
                else {"type": "text"}
            ),
            "stream": True,
        }
        async with self._client.stream(
            "POST",
            f"{node.url}/api/v1/sessions/{remote.remote_id}/responses",
            json=request,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise BackendError(
                    "remote_node_error",
                    body.decode("utf-8", errors="replace")[:1024],
                    retryable=response.status_code >= 500,
                    status_code=response.status_code,
                )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                frame = json.loads(line[6:])
                payload = frame.get("payload", {})
                kind = payload.get("type")
                if kind == "response.text.delta":
                    yield BackendDelta(content_delta=str(payload.get("delta", "")))
                elif kind == "response.reasoning.delta":
                    yield BackendDelta(reasoning_delta=str(payload.get("delta", "")))
                elif kind == "response.tool_call.delta":
                    yield BackendDelta(
                        tool_calls=(
                            BackendToolCallDelta(
                                index=int(payload.get("index", 0)),
                                call_id=payload.get("call_id"),
                                name=payload.get("name"),
                                arguments_delta=str(payload.get("arguments_delta", "")),
                            ),
                        )
                    )
                elif kind == "response.completed":
                    usage = (
                        TokenUsage.model_validate(payload["usage"])
                        if payload.get("usage")
                        else None
                    )
                    performance = (
                        ResponsePerformance.model_validate(payload["performance"])
                        if payload.get("performance")
                        else None
                    )
                    yield BackendDelta(
                        finish_reason=str(payload.get("finish_reason", "stop")),
                        usage=usage,
                        performance=performance,
                    )
                elif kind == "session.state":
                    remote.revision = int(payload.get("revision", remote.revision))
                elif kind == "error":
                    detail = payload.get("error", {})
                    raise BackendError(
                        str(detail.get("code", "remote_node_error")),
                        str(detail.get("message", "remote node failed")),
                        retryable=bool(detail.get("retryable", False)),
                    )
        remote.synchronized_messages = len(messages) + 1

    async def _request_json(
        self,
        node: RemoteNodeResource,
        method: str,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        response = await self._client.request(
            method, f"{node.url}{path}", json=body, headers=headers
        )
        if response.status_code >= 400:
            raise BackendError(
                "remote_node_error",
                response.text[:1024],
                retryable=response.status_code >= 500,
                status_code=response.status_code,
            )
        value = response.json()
        if not isinstance(value, dict):
            raise BackendError("remote_node_protocol_error", "remote response must be an object")
        return value

    async def _parts(
        self,
        node: RemoteNodeResource,
        content: Any,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": str(item.get("text", ""))})
                elif kind == "image_url":
                    url = item.get("image_url", {}).get("url")
                    if isinstance(url, str):
                        media = await self._upload_data_url(node, url, headers)
                        parts.append({"type": "image", "media": media})
                elif kind == "video_url":
                    url = item.get("video_url", {}).get("url")
                    if isinstance(url, str):
                        media = await self._upload_data_url(node, url, headers)
                        parts.append({"type": "video", "media": media})
                elif kind == "input_audio":
                    audio = item.get("input_audio", {})
                    encoded = audio.get("data")
                    if isinstance(encoded, str):
                        media = await self._upload_bytes(
                            node,
                            self._decode_base64(encoded),
                            self._audio_mime(str(audio.get("format", "wav"))),
                            headers,
                        )
                        parts.append(
                            {
                                "type": "audio",
                                "media": media,
                                "sample_rate_hz": int(audio.get("sample_rate_hz") or 16000),
                                "channels": int(audio.get("channels") or 1),
                            }
                        )
            return parts or [{"type": "text", "text": ""}]
        return [{"type": "text", "text": str(content or "")}]

    async def _message_parts(
        self,
        node: RemoteNodeResource,
        message: dict[str, Any],
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        role = str(message.get("role", "user"))
        if role == "tool":
            return [
                {
                    "type": "tool_result",
                    "call_id": str(message.get("tool_call_id") or "remote-tool"),
                    "result": message.get("content", ""),
                    "is_error": False,
                }
            ]
        parts = await self._parts(node, message.get("content"), headers)
        if role != "assistant":
            return parts
        extras: list[dict[str, Any]] = []
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            extras.append({"type": "reasoning", "text": reasoning})
        for call in message.get("tool_calls", []):
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            if not isinstance(function, dict) or not function.get("name"):
                continue
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            extras.append(
                {
                    "type": "tool_call",
                    "call_id": str(call.get("id") or uuid4()),
                    "name": str(function["name"]),
                    "arguments": arguments,
                }
            )
        if extras and parts == [{"type": "text", "text": ""}]:
            parts = []
        return [*parts, *extras]

    async def _upload_data_url(
        self,
        node: RemoteNodeResource,
        url: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        if not url.startswith("data:") or ";base64," not in url:
            raise BackendError(
                "remote_media_unsupported",
                "remote routing accepts embedded base64 media only",
            )
        descriptor, encoded = url.split(",", 1)
        mime_type = descriptor[5:].split(";", 1)[0].strip().lower()
        return await self._upload_bytes(
            node,
            self._decode_base64(encoded),
            mime_type,
            headers,
        )

    async def _upload_bytes(
        self,
        node: RemoteNodeResource,
        data: bytes,
        mime_type: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        response = await self._client.post(
            f"{node.url}/api/v1/media",
            content=data,
            headers={
                **headers,
                "Content-Type": mime_type,
                "X-Content-SHA256": digest,
            },
        )
        if response.status_code >= 400:
            raise BackendError(
                "remote_media_upload_failed",
                response.text[:1024],
                retryable=response.status_code >= 500,
                status_code=response.status_code,
            )
        value = response.json()
        if not isinstance(value, dict) or not isinstance(value.get("media"), dict):
            raise BackendError(
                "remote_node_protocol_error",
                "remote media response is invalid",
            )
        return value["media"]

    @staticmethod
    def _decode_base64(value: str) -> bytes:
        try:
            return base64.b64decode(value, validate=True)
        except ValueError as error:
            raise BackendError("remote_media_invalid", "media base64 is invalid") from error

    @staticmethod
    def _audio_mime(format_name: str) -> str:
        return {
            "mp3": "audio/mpeg",
            "m4a": "audio/x-m4a",
            "wav": "audio/wav",
        }.get(format_name.lower(), f"audio/{format_name.lower()}")

    async def nodes(self, *, force: bool = False) -> list[RemoteNodeResource]:
        return await self.refresh(force=force)

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool:
        for node_id, state in list(self._states.items()):
            source = self._sessions.get((node_id, source_session_id))
            if source is not None:
                created = await self._request_json(
                    state.resource,
                    "POST",
                    f"/api/v1/sessions/{source.remote_id}/fork",
                    {"include_message": True},
                    self._headers(state.resource),
                )
                self._sessions[(node_id, target_session_id)] = _RemoteSession(
                    remote_id=UUID(created["id"]),
                    revision=int(created["revision"]),
                    synchronized_messages=source.synchronized_messages,
                )
                return True
        return await self.local.fork_session(source_session_id, target_session_id)

    async def close_session(self, session_id: UUID) -> bool:
        removed: list[tuple[_NodeState, _RemoteSession]] = []
        for key in [item for item in self._sessions if item[1] == session_id]:
            remote = self._sessions.pop(key, None)
            state = self._states.get(key[0])
            if remote is not None and state is not None:
                removed.append((state, remote))
        for state, remote in removed:
            response = await self._client.delete(
                f"{state.resource.url}/api/v1/sessions/{remote.remote_id}",
                headers=self._headers(state.resource),
            )
            if response.status_code not in {204, 404}:
                raise BackendError(
                    "remote_node_error",
                    response.text[:1024],
                    retryable=response.status_code >= 500,
                    status_code=response.status_code,
                )
        return bool(removed) or await self.local.close_session(session_id)

    async def cancel_response(self, session_id: UUID) -> bool:
        for (node_id, local_session_id), remote in list(self._sessions.items()):
            if local_session_id != session_id:
                continue
            state = self._states.get(node_id)
            if state is None:
                continue
            response = await self._client.post(
                f"{state.resource.url}/api/v1/sessions/{remote.remote_id}/responses/cancel",
                headers=self._headers(state.resource),
            )
            if response.status_code == 200:
                return True
            raise BackendError(
                "remote_node_cancel_failed",
                response.text[:1024],
                retryable=response.status_code >= 500,
                status_code=response.status_code,
            )
        cancel = getattr(self.local, "cancel_response", None)
        return bool(await cancel(session_id)) if callable(cancel) else False

    async def capabilities(self) -> RuntimeCapabilitiesResource:
        return await self.local.capabilities()

    async def runtime_status(self) -> dict[str, Any]:
        status = dict(await self.local.runtime_status())
        nodes = await self.refresh()
        status["cluster_nodes"] = len(nodes)
        status["cluster_healthy_nodes"] = sum(item.healthy for item in nodes)
        status["cluster_active_requests"] = sum(item.active_requests for item in nodes)
        async with self._lock:
            states = list(self._states.values())
        status["cluster_process_resident_bytes"] = sum(
            int(item.status.get("process_resident_bytes") or 0) for item in states
        )
        status["cluster_device_active_bytes"] = sum(
            int(item.status.get("mlx_active_bytes") or item.status.get("cuda_allocated_bytes") or 0)
            for item in states
        )
        status["cluster_total_requests"] = sum(
            int(item.status.get("total_requests") or 0) for item in states
        )
        return status

    async def runtime_models(self) -> dict[str, Any]:
        local = await self.local.runtime_models()
        data = list(local.get("data", [])) if isinstance(local, dict) else []
        for node in await self.refresh():
            data.extend(
                {"id": model, "object": "model", "node_id": str(node.id), "node": node.name}
                for model in node.models
            )
        return {"object": "list", "data": data}

    async def realtime_capabilities(self) -> dict[str, Any]:
        return await self.local.realtime_capabilities()

    async def voice_output_status(self) -> dict[str, Any]:
        return await self.local.voice_output_status()

    async def enable_realtime(self) -> dict[str, Any]:
        return await self.local.enable_realtime()

    async def realtime_serve(self, client: Any, *, mode: str = "audio") -> bool:
        return await self.local.realtime_serve(client, mode=mode)

    async def reload_runtime(self, context_size: int) -> dict[str, Any]:
        return await self.local.reload_runtime(context_size)

    async def clear_runtime_cache(self) -> dict[str, Any]:
        return await self.local.clear_runtime_cache()

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        return self.local.realtime_connect(mode=mode)

    async def aclose(self) -> None:
        await self.local.aclose()
        if self._owns_client:
            await self._client.aclose()
