"""MFQd session orchestration over a persistent store and streaming backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from mfqd.backend import BackendDelta, BackendError, BackendToolCallDelta, ChatBackend
from mfqd.models import (
    AppendMessageRequest,
    AppendMessageResult,
    AudioPart,
    ContentPart,
    CreateResponseRequest,
    CreateSessionRequest,
    ErrorDetail,
    ErrorEvent,
    ForkSessionRequest,
    GeneratedAudioPart,
    ImagePart,
    Message,
    MessageList,
    MessageRole,
    OperationAccepted,
    RealtimeFrame,
    RealtimePayload,
    ReasoningPart,
    RewindSessionRequest,
    ResponseCompleted,
    ResponseReasoningDelta,
    ResponseList,
    ResponseResource,
    ResponseStatus,
    ResponseTextDelta,
    ResponseToolCallDelta,
    RuntimeCapabilitiesResource,
    RuntimeInstanceList,
    SessionList,
    SessionResource,
    SessionState,
    SessionStateChanged,
    TextPart,
    TokenUsage,
    ToolCallPart,
    ToolResultPart,
    TranscriptPart,
    UpdateSessionRequest,
)
from mfqd.storage import (
    BeginResponseResult,
    IdempotencyConflictError,
    MessageNotFoundError,
    ResponseInProgressError,
    RevisionConflictError,
    SessionNotFoundError,
    SessionStore,
    StorageError,
)


class ServiceError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = ErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        )


@dataclass(frozen=True)
class PreparedResponse:
    request: CreateResponseRequest
    begin: BeginResponseResult
    backend_messages: tuple[dict[str, Any], ...]


@dataclass
class _ToolCall:
    call_id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)


@dataclass
class _OutputAccumulator:
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: dict[int, _ToolCall] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    def apply(self, delta: BackendDelta) -> None:
        if delta.content_delta:
            self.text.append(delta.content_delta)
        if delta.reasoning_delta:
            self.reasoning.append(delta.reasoning_delta)
        for tool_delta in delta.tool_calls:
            self._apply_tool_delta(tool_delta)
        if delta.finish_reason is not None:
            self.finish_reason = delta.finish_reason
        if delta.usage is not None:
            self.usage = delta.usage

    def output_parts(self) -> list[ContentPart]:
        parts: list[ContentPart] = []
        reasoning = "".join(self.reasoning)
        text = "".join(self.text)
        if reasoning:
            parts.append(ReasoningPart(text=reasoning))
        if text:
            parts.append(TextPart(text=text))
        for index in sorted(self.tool_calls):
            tool = self.tool_calls[index]
            if not tool.call_id or not tool.name:
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {index} is missing its id or name",
                )
            raw_arguments = "".join(tool.arguments) or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {index} returned invalid JSON arguments: {error}",
                ) from error
            if not isinstance(arguments, dict):
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {index} arguments must decode to an object",
                )
            parts.append(ToolCallPart(call_id=tool.call_id, name=tool.name, arguments=arguments))
        if not parts:
            parts.append(TextPart(text=""))
        return parts

    def _apply_tool_delta(self, delta: BackendToolCallDelta) -> None:
        tool = self.tool_calls.setdefault(delta.index, _ToolCall())
        if delta.call_id is not None:
            if tool.call_id is not None and tool.call_id != delta.call_id:
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {delta.index} changed id during streaming",
                )
            tool.call_id = delta.call_id
        if delta.name is not None:
            if tool.name is not None and tool.name != delta.name:
                raise BackendError(
                    "backend_protocol_error",
                    f"tool call {delta.index} changed name during streaming",
                )
            tool.name = delta.name
        if delta.arguments_delta:
            tool.arguments.append(delta.arguments_delta)


class MfqdService:
    def __init__(self, store: SessionStore, backend: ChatBackend) -> None:
        self.store = store
        self.backend = backend

    async def aclose(self) -> None:
        await self.backend.aclose()

    async def create_session(self, request: CreateSessionRequest) -> SessionResource:
        return await asyncio.to_thread(self.store.create_session, request)

    async def list_sessions(self, *, limit: int, offset: int) -> SessionList:
        sessions = await asyncio.to_thread(self.store.list_sessions, limit=limit, offset=offset)
        return SessionList(data=sessions)

    async def get_session(self, session_id: UUID) -> SessionResource:
        try:
            return await asyncio.to_thread(self.store.get_session, session_id)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error

    async def update_session(
        self,
        session_id: UUID,
        request: UpdateSessionRequest,
    ) -> SessionResource:
        try:
            return await asyncio.to_thread(self.store.update_session, session_id, request)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error

    async def list_messages(self, session_id: UUID) -> MessageList:
        try:
            messages = await asyncio.to_thread(self.store.list_messages, session_id)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        return MessageList(data=messages)

    async def list_responses(self, session_id: UUID, *, limit: int = 200) -> ResponseList:
        try:
            data = await asyncio.to_thread(
                self.store.list_responses, session_id, limit=limit
            )
        except ValueError as error:
            raise ServiceError(422, "invalid_request", str(error)) from error
        return ResponseList(data=data)

    async def append_message(
        self,
        session_id: UUID,
        request: AppendMessageRequest,
    ) -> AppendMessageResult:
        try:
            session, message = await asyncio.to_thread(
                self.store.append_message,
                session_id,
                request.expected_revision,
                request.role,
                request.parts,
            )
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except RevisionConflictError as error:
            raise ServiceError(
                409,
                "revision_conflict",
                str(error),
                details={"expected_revision": error.expected, "actual_revision": error.actual},
            ) from error
        except StorageError as error:
            raise ServiceError(409, "session_state_conflict", str(error)) from error
        return AppendMessageResult(session=session, message=message)

    async def fork_session(
        self,
        session_id: UUID,
        request: ForkSessionRequest,
    ) -> SessionResource:
        try:
            forked = await asyncio.to_thread(self.store.fork_session, session_id, request)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except MessageNotFoundError as error:
            raise ServiceError(404, "message_not_found", str(error)) from error
        except StorageError as error:
            raise ServiceError(409, "session_state_conflict", str(error)) from error
        if request.at_message_id is None:
            await self.backend.fork_session(session_id, forked.id)
        return forked

    async def rewind_session(
        self,
        session_id: UUID,
        request: RewindSessionRequest,
    ) -> SessionResource:
        try:
            return await asyncio.to_thread(
                self.store.rewind_session,
                session_id,
                request,
            )
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except MessageNotFoundError as error:
            raise ServiceError(404, "message_not_found", str(error)) from error
        except RevisionConflictError as error:
            raise ServiceError(
                409,
                "revision_conflict",
                str(error),
                details={"expected_revision": error.expected, "actual_revision": error.actual},
            ) from error
        except ResponseInProgressError as error:
            raise ServiceError(409, "response_in_progress", str(error), retryable=True) from error
        except StorageError as error:
            raise ServiceError(409, "session_state_conflict", str(error)) from error

    async def delete_session(self, session_id: UUID) -> None:
        try:
            await asyncio.to_thread(self.store.delete_session, session_id)
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except ResponseInProgressError as error:
            raise ServiceError(409, "response_in_progress", str(error), retryable=True) from error
        await self.backend.close_session(session_id)

    async def runtime_instances(self) -> RuntimeInstanceList:
        return RuntimeInstanceList(data=[])

    async def runtime_capabilities(self) -> RuntimeCapabilitiesResource:
        try:
            return await self.backend.capabilities()
        except BackendError as error:
            raise ServiceError(
                503,
                error.code,
                str(error),
                retryable=error.retryable,
            ) from error

    async def runtime_status(self) -> dict[str, Any]:
        return await self._runtime_request("runtime_status")

    async def runtime_models(self) -> dict[str, Any]:
        return await self._runtime_request("runtime_models")

    async def realtime_capabilities(self) -> dict[str, Any]:
        return await self._runtime_request("realtime_capabilities")

    async def reload_runtime(self, context_size: int) -> dict[str, Any]:
        return await self._runtime_request("reload_runtime", context_size)

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        connector = getattr(self.backend, "realtime_connect", None)
        if connector is None:
            raise ServiceError(
                501,
                "realtime_unavailable",
                "the configured backend does not expose realtime transport",
            )
        return connector(mode=mode)

    async def load_model(self, _: object) -> OperationAccepted:
        raise ServiceError(501, "model_management_unavailable", "model loading is not available")

    async def unload_model(self, _: object) -> OperationAccepted:
        raise ServiceError(501, "model_management_unavailable", "model unloading is not available")

    async def prepare_response(
        self,
        session_id: UUID,
        request: CreateResponseRequest,
    ) -> PreparedResponse:
        try:
            history = await asyncio.to_thread(self.store.list_messages, session_id)
            backend_messages = tuple(
                self._message_to_backend(
                    message,
                    include_reasoning=request.include_reasoning_history,
                )
                for message in history
            )
            if request.system_prompt and request.system_prompt.strip():
                backend_messages = (
                    {"role": "system", "content": request.system_prompt.strip()},
                    *backend_messages,
                )
            backend_input = self._parts_to_backend(MessageRole.USER, request.input)
            fingerprint = self._request_fingerprint(request)
            begin = await asyncio.to_thread(
                self.store.begin_response,
                session_id,
                request.request_id,
                uuid4(),
                fingerprint,
                request.expected_revision,
                request.input,
            )
            if begin.started and begin.session.title is None:
                title = self._title_from_parts(request.input)
                if title:
                    session = await asyncio.to_thread(
                        self.store.update_session,
                        session_id,
                        UpdateSessionRequest(title=title),
                    )
                    begin = BeginResponseResult(
                        session=session,
                        response=begin.response,
                        started=True,
                    )
        except SessionNotFoundError as error:
            raise ServiceError(404, "session_not_found", str(error)) from error
        except RevisionConflictError as error:
            raise ServiceError(
                409,
                "revision_conflict",
                str(error),
                details={"expected_revision": error.expected, "actual_revision": error.actual},
            ) from error
        except IdempotencyConflictError as error:
            raise ServiceError(409, "idempotency_conflict", str(error)) from error
        except ResponseInProgressError as error:
            raise ServiceError(409, "response_in_progress", str(error), retryable=True) from error
        except StorageError as error:
            raise ServiceError(409, "session_state_conflict", str(error)) from error

        if not begin.started and begin.response.status == ResponseStatus.RUNNING:
            raise ServiceError(
                409,
                "response_in_progress",
                f"request {request.request_id} is already running",
                retryable=True,
            )
        if begin.started:
            backend_messages = (*backend_messages, backend_input)
        return PreparedResponse(
            request=request,
            begin=begin,
            backend_messages=tuple(backend_messages),
        )

    async def collect_response(self, prepared: PreparedResponse) -> ResponseResource:
        if not prepared.begin.started:
            return prepared.begin.response
        accumulator = _OutputAccumulator()
        try:
            async for delta in self.backend.stream(
                model=prepared.begin.session.model,
                messages=prepared.backend_messages,
                sampling=prepared.request.sampling,
                session_id=prepared.begin.session.id,
            ):
                accumulator.apply(delta)
            finish_reason = self._require_finish_reason(accumulator)
            return await asyncio.to_thread(
                self.store.complete_response,
                prepared.begin.response.id,
                accumulator.output_parts(),
                finish_reason,
                accumulator.usage,
            )
        except asyncio.CancelledError:
            detail = ErrorDetail(
                code="client_cancelled",
                message="client disconnected before the response completed",
            )
            await asyncio.to_thread(
                self.store.terminate_response,
                prepared.begin.response.id,
                detail,
                cancelled=True,
            )
            raise
        except BackendError as error:
            await self._terminate_backend_failure(prepared.begin.response.id, error)
            raise ServiceError(
                502,
                error.code,
                str(error),
                retryable=error.retryable,
            ) from error

    async def stream_response(self, prepared: PreparedResponse) -> AsyncIterator[str]:
        sequence = 0
        if not prepared.begin.started:
            async for event in self._replay_response(prepared.begin.response):
                yield self._encode_sse(
                    event,
                    sequence,
                    session_id=prepared.begin.response.session_id,
                )
                sequence += 1
            session = await asyncio.to_thread(
                self.store.get_session,
                prepared.begin.response.session_id,
            )
            yield self._encode_sse(
                SessionStateChanged(state=session.state, revision=session.revision),
                sequence,
                session_id=session.id,
            )
            return

        yield self._encode_sse(
            SessionStateChanged(
                state=SessionState.PROCESSING,
                revision=prepared.begin.session.revision,
            ),
            sequence,
            session_id=prepared.begin.session.id,
        )
        sequence += 1
        accumulator = _OutputAccumulator()
        try:
            async for delta in self.backend.stream(
                model=prepared.begin.session.model,
                messages=prepared.backend_messages,
                sampling=prepared.request.sampling,
                session_id=prepared.begin.session.id,
            ):
                accumulator.apply(delta)
                payloads = self._delta_payloads(prepared.begin.response.id, delta)
                for payload in payloads:
                    yield self._encode_sse(
                        payload,
                        sequence,
                        session_id=prepared.begin.session.id,
                    )
                    sequence += 1
            finish_reason = self._require_finish_reason(accumulator)
            completed = await asyncio.to_thread(
                self.store.complete_response,
                prepared.begin.response.id,
                accumulator.output_parts(),
                finish_reason,
                accumulator.usage,
            )
            yield self._encode_sse(
                ResponseCompleted(
                    response_id=completed.id,
                    finish_reason=finish_reason,
                    usage=completed.usage,
                ),
                sequence,
                session_id=completed.session_id,
            )
            sequence += 1
            session = await asyncio.to_thread(self.store.get_session, completed.session_id)
            yield self._encode_sse(
                SessionStateChanged(state=session.state, revision=session.revision),
                sequence,
                session_id=session.id,
            )
        except asyncio.CancelledError:
            detail = ErrorDetail(
                code="client_cancelled",
                message="client disconnected before the response completed",
            )
            await asyncio.to_thread(
                self.store.terminate_response,
                prepared.begin.response.id,
                detail,
                cancelled=True,
            )
            raise
        except BackendError as error:
            failed = await self._terminate_backend_failure(prepared.begin.response.id, error)
            yield self._encode_sse(
                ErrorEvent(error=failed.error or ErrorDetail(code=error.code, message=str(error))),
                sequence,
                session_id=failed.session_id,
            )
            sequence += 1
            session = await asyncio.to_thread(self.store.get_session, failed.session_id)
            yield self._encode_sse(
                SessionStateChanged(state=session.state, revision=session.revision),
                sequence,
                session_id=session.id,
            )

    async def _terminate_backend_failure(
        self,
        response_id: UUID,
        error: BackendError,
    ) -> ResponseResource:
        detail = ErrorDetail(
            code=error.code,
            message=str(error),
            retryable=error.retryable,
        )
        return await asyncio.to_thread(self.store.terminate_response, response_id, detail)

    async def _replay_response(self, response: ResponseResource) -> AsyncIterator[RealtimePayload]:
        if response.status == ResponseStatus.COMPLETED:
            tool_index = 0
            for part in response.output:
                if isinstance(part, ReasoningPart):
                    yield ResponseReasoningDelta(response_id=response.id, delta=part.text)
                elif isinstance(part, TextPart):
                    yield ResponseTextDelta(response_id=response.id, delta=part.text)
                elif isinstance(part, ToolCallPart):
                    yield ResponseToolCallDelta(
                        response_id=response.id,
                        index=tool_index,
                        call_id=part.call_id,
                        name=part.name,
                        arguments_delta=json.dumps(part.arguments, separators=(",", ":")),
                    )
                    tool_index += 1
            yield ResponseCompleted(
                response_id=response.id,
                finish_reason=response.finish_reason or "stop",
                usage=response.usage,
            )
        elif response.error is not None:
            yield ErrorEvent(error=response.error)

    @staticmethod
    def _delta_payloads(response_id: UUID, delta: BackendDelta) -> list[RealtimePayload]:
        payloads: list[RealtimePayload] = []
        if delta.reasoning_delta:
            payloads.append(
                ResponseReasoningDelta(response_id=response_id, delta=delta.reasoning_delta)
            )
        if delta.content_delta:
            payloads.append(ResponseTextDelta(response_id=response_id, delta=delta.content_delta))
        for tool in delta.tool_calls:
            payloads.append(
                ResponseToolCallDelta(
                    response_id=response_id,
                    index=tool.index,
                    call_id=tool.call_id,
                    name=tool.name,
                    arguments_delta=tool.arguments_delta,
                )
            )
        return payloads

    @staticmethod
    def _request_fingerprint(request: CreateResponseRequest) -> str:
        value = request.model_dump(mode="json", exclude={"stream"})
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _message_to_backend(
        cls,
        message: Message,
        *,
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        return cls._parts_to_backend(
            message.role,
            message.parts,
            include_reasoning=include_reasoning,
        )

    @staticmethod
    def _parts_to_backend(
        role: MessageRole,
        parts: Sequence[ContentPart],
        *,
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        if role == MessageRole.TOOL:
            if len(parts) != 1 or not isinstance(parts[0], ToolResultPart):
                raise ServiceError(
                    422,
                    "unsupported_tool_message",
                    "a tool message must contain exactly one tool_result part",
                )
            result = parts[0]
            content = (
                result.result
                if isinstance(result.result, str)
                else json.dumps(result.result, separators=(",", ":"), ensure_ascii=False)
            )
            return {"role": "tool", "tool_call_id": result.call_id, "content": content}

        content_fragments: list[str] = []
        reasoning_fragments: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, (TextPart, TranscriptPart)):
                content_fragments.append(part.text)
            elif isinstance(part, ReasoningPart):
                if role != MessageRole.ASSISTANT:
                    raise ServiceError(
                        422,
                        "unsupported_reasoning_part",
                        "reasoning parts are valid only on assistant messages",
                    )
                if include_reasoning:
                    reasoning_fragments.append(part.text)
            elif isinstance(part, ToolCallPart):
                if role != MessageRole.ASSISTANT:
                    raise ServiceError(
                        422,
                        "unsupported_tool_call_part",
                        "tool_call parts are valid only on assistant messages",
                    )
                tool_calls.append(
                    {
                        "id": part.call_id,
                        "type": "function",
                        "function": {
                            "name": part.name,
                            "arguments": json.dumps(
                                part.arguments,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            elif isinstance(part, (ImagePart, AudioPart, GeneratedAudioPart)):
                raise ServiceError(
                    422,
                    "unsupported_content_part",
                    f"{part.type} input is not available on the text backend",
                )
            elif isinstance(part, ToolResultPart):
                raise ServiceError(
                    422,
                    "unsupported_tool_result_part",
                    "tool_result parts require a tool-role message",
                )
        message: dict[str, Any] = {"role": role.value, "content": "".join(content_fragments)}
        if reasoning_fragments:
            message["reasoning_content"] = "".join(reasoning_fragments)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    async def _runtime_request(self, method: str, *args: Any) -> dict[str, Any]:
        operation = getattr(self.backend, method, None)
        if operation is None:
            raise ServiceError(
                501,
                "runtime_control_unavailable",
                f"the configured backend does not implement {method}",
            )
        try:
            return await operation(*args)
        except BackendError as error:
            raise ServiceError(
                503,
                error.code,
                str(error),
                retryable=error.retryable,
            ) from error

    @staticmethod
    def _title_from_parts(parts: Sequence[ContentPart]) -> str | None:
        text = " ".join(
            part.text.strip()
            for part in parts
            if isinstance(part, (TextPart, TranscriptPart)) and part.text.strip()
        )
        normalized = " ".join(text.split())
        if not normalized:
            return None
        return normalized if len(normalized) <= 60 else f"{normalized[:59]}…"

    @staticmethod
    def _require_finish_reason(accumulator: _OutputAccumulator) -> str:
        if accumulator.finish_reason is None:
            raise BackendError(
                "backend_protocol_error",
                "backend stream completed without a finish reason",
            )
        return accumulator.finish_reason

    @staticmethod
    def _encode_sse(
        payload: RealtimePayload,
        sequence: int,
        *,
        session_id: UUID | None = None,
    ) -> str:
        if session_id is None:
            raise ValueError("session_id is required for SSE payloads")
        frame = RealtimeFrame(
            session_id=session_id,
            sequence=sequence,
            timestamp=datetime.now(timezone.utc),
            payload=payload,
        )
        event_type = frame.payload.type
        return f"event: {event_type}\nid: {sequence}\ndata: {frame.model_dump_json()}\n\n"
