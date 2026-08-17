"""Versioned public API models shared by MFQd clients and services."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Final, Literal
from uuid import UUID

from pydantic import AwareDatetime, Base64Bytes, BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION: Final[Literal["1.0"]] = "1.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ProtocolModel(BaseModel):
    """Base model for protocol objects with strict forward-compatibility rules."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SessionMode(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    FULL_DUPLEX = "full_duplex"


class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ResponseStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RuntimeInstanceState(str, Enum):
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    UNLOADING = "unloading"
    FAILED = "failed"


class MediaRef(ProtocolModel):
    id: UUID
    sha256: str = Field(pattern=SHA256_PATTERN)
    mime_type: str = Field(min_length=1, max_length=255)
    byte_size: int = Field(ge=0)


class TextPart(ProtocolModel):
    type: Literal["text"] = "text"
    text: str


class ReasoningPart(ProtocolModel):
    type: Literal["reasoning"] = "reasoning"
    text: str


class ImagePart(ProtocolModel):
    type: Literal["image"] = "image"
    media: MediaRef
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class AudioPart(ProtocolModel):
    type: Literal["audio"] = "audio"
    media: MediaRef
    sample_rate_hz: int = Field(ge=1)
    channels: int = Field(ge=1, le=8)
    duration_ms: int | None = Field(default=None, ge=0)


class TranscriptPart(ProtocolModel):
    type: Literal["transcript"] = "transcript"
    text: str
    language: str | None = Field(default=None, min_length=2, max_length=35)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)


class ToolCallPart(ProtocolModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultPart(ProtocolModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str = Field(min_length=1, max_length=255)
    result: Any
    is_error: bool = False


class GeneratedAudioPart(ProtocolModel):
    type: Literal["generated_audio"] = "generated_audio"
    media: MediaRef
    sample_rate_hz: int = Field(ge=1)
    channels: int = Field(ge=1, le=8)
    duration_ms: int | None = Field(default=None, ge=0)


ContentPart = Annotated[
    TextPart
    | ReasoningPart
    | ImagePart
    | AudioPart
    | TranscriptPart
    | ToolCallPart
    | ToolResultPart
    | GeneratedAudioPart,
    Field(discriminator="type"),
]


class Message(ProtocolModel):
    id: UUID
    role: MessageRole
    parts: list[ContentPart] = Field(min_length=1)
    parent_id: UUID | None = None
    created_at: AwareDatetime


class SamplingParams(ProtocolModel):
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_k: int = Field(default=20, ge=0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    seed: int | None = Field(default=None, ge=0)
    enable_thinking: bool = True
    reasoning_effort: str | None = Field(default=None, min_length=1, max_length=32)


class RuntimeIdentity(ProtocolModel):
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    quantization: str = Field(min_length=1, max_length=64)
    runtime_build: str = Field(min_length=1, max_length=255)
    tokenizer_sha256: str = Field(pattern=SHA256_PATTERN)
    chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
    processor_version: str = Field(min_length=1, max_length=255)
    rope_parameters_sha256: str = Field(pattern=SHA256_PATTERN)
    kv_dtype: str = Field(min_length=1, max_length=32)


class CreateSessionRequest(ProtocolModel):
    model: str = Field(min_length=1, max_length=255)
    mode: SessionMode = SessionMode.TEXT
    title: str | None = Field(default=None, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResource(ProtocolModel):
    id: UUID
    model: str
    mode: SessionMode
    state: SessionState
    revision: int = Field(ge=0)
    title: str | None = None
    runtime_instance_id: UUID | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionList(ProtocolModel):
    data: list[SessionResource]


class MessageList(ProtocolModel):
    data: list[Message]


class ForkSessionRequest(ProtocolModel):
    at_message_id: UUID | None = None
    include_message: bool = True
    title: str | None = Field(default=None, max_length=512)


class RewindSessionRequest(ProtocolModel):
    expected_revision: int = Field(ge=0)
    at_message_id: UUID
    include_message: bool = True


class UpdateSessionRequest(ProtocolModel):
    title: str | None = Field(default=None, max_length=512)
    mode: SessionMode | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> "UpdateSessionRequest":
        if not self.model_fields_set:
            raise ValueError("at least one session field must be provided")
        if "mode" in self.model_fields_set and self.mode is None:
            raise ValueError("mode cannot be null")
        return self


class AppendMessageRequest(ProtocolModel):
    expected_revision: int = Field(ge=0)
    role: MessageRole
    parts: list[ContentPart] = Field(min_length=1)


class AppendMessageResult(ProtocolModel):
    session: SessionResource
    message: Message


class ErrorDetail(ProtocolModel):
    code: str = Field(min_length=1, max_length=128)
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(ProtocolModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class CreateResponseRequest(ProtocolModel):
    request_id: UUID
    expected_revision: int = Field(ge=0)
    input: list[ContentPart] = Field(min_length=1)
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    system_prompt: str | None = Field(default=None, max_length=32768)
    include_reasoning_history: bool = True
    stream: bool = True


class ResponseResource(ProtocolModel):
    id: UUID
    request_id: UUID
    session_id: UUID
    status: ResponseStatus
    output_message_id: UUID | None = None
    output: list[ContentPart] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    error: ErrorDetail | None = None


class ResponseList(ProtocolModel):
    data: list[ResponseResource]


class MediaResource(ProtocolModel):
    media: MediaRef
    created_at: AwareDatetime


class ModelLoadRequest(ProtocolModel):
    model: str = Field(min_length=1, max_length=255)
    artifact_uri: str = Field(min_length=1)
    device_ids: list[str] = Field(default_factory=list)
    idle_ttl_seconds: int | None = Field(default=None, ge=0)
    pin: bool = False


class ModelUnloadRequest(ProtocolModel):
    instance_id: UUID
    force: bool = False


class RuntimeInstanceResource(ProtocolModel):
    id: UUID
    model: str
    state: RuntimeInstanceState
    devices: list[str]
    active_sessions: int = Field(ge=0)
    queued_requests: int = Field(ge=0)
    resident_bytes: int = Field(ge=0)
    kv_bytes: int = Field(ge=0)
    last_used_at: AwareDatetime | None = None
    identity: RuntimeIdentity | None = None


class RuntimeInstanceList(ProtocolModel):
    data: list[RuntimeInstanceResource]


class ModelFeatureSet(ProtocolModel):
    text: bool = True
    image_input: bool = False
    video_input: bool = False
    audio_input: bool = False
    audio_output: bool = False
    full_duplex: bool = False


class ModelCapabilities(ProtocolModel):
    architecture_family: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=255)
    features: ModelFeatureSet = Field(default_factory=ModelFeatureSet)


class RuntimeCapabilitiesResource(ProtocolModel):
    model: str = Field(min_length=1, max_length=255)
    model_type: str = Field(min_length=1, max_length=128)
    model_capabilities: ModelCapabilities
    duplex_available: bool = False


class OperationAccepted(ProtocolModel):
    operation_id: UUID
    status: Literal["accepted"] = "accepted"


class RuntimeReloadRequest(ProtocolModel):
    context_size: int = Field(ge=512)


class ErrorResponse(ProtocolModel):
    error: ErrorDetail


class InputAudioDelta(ProtocolModel):
    type: Literal["input_audio.delta"] = "input_audio.delta"
    audio_sequence: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate_hz: int = Field(default=16000, ge=1)
    channels: int = Field(default=1, ge=1, le=8)
    data_base64: Base64Bytes


class InputAudioCommit(ProtocolModel):
    type: Literal["input_audio.commit"] = "input_audio.commit"
    last_audio_sequence: int = Field(ge=0)


class ResponseTextDelta(ProtocolModel):
    type: Literal["response.text.delta"] = "response.text.delta"
    response_id: UUID
    delta: str


class ResponseReasoningDelta(ProtocolModel):
    type: Literal["response.reasoning.delta"] = "response.reasoning.delta"
    response_id: UUID
    delta: str


class ResponseToolCallDelta(ProtocolModel):
    type: Literal["response.tool_call.delta"] = "response.tool_call.delta"
    response_id: UUID
    index: int = Field(ge=0)
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


class ResponseAudioDelta(ProtocolModel):
    type: Literal["response.audio.delta"] = "response.audio.delta"
    response_id: UUID
    audio_sequence: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate_hz: int = Field(ge=1)
    channels: int = Field(ge=1, le=8)
    data_base64: Base64Bytes


class ResponseInterrupted(ProtocolModel):
    type: Literal["response.interrupted"] = "response.interrupted"
    response_id: UUID
    reason: Literal["client_cancelled", "new_input", "session_closed", "runtime_error"]


class ResponseCompleted(ProtocolModel):
    type: Literal["response.completed"] = "response.completed"
    response_id: UUID
    finish_reason: str
    usage: TokenUsage | None = None


class SessionStateChanged(ProtocolModel):
    type: Literal["session.state"] = "session.state"
    state: SessionState
    revision: int = Field(ge=0)


class RuntimeMetrics(ProtocolModel):
    type: Literal["runtime.metrics"] = "runtime.metrics"
    instance_id: UUID
    queue_depth: int = Field(ge=0)
    resident_bytes: int = Field(ge=0)
    kv_bytes: int = Field(ge=0)
    prefill_tokens_per_second: float | None = Field(default=None, ge=0.0)
    decode_tokens_per_second: float | None = Field(default=None, ge=0.0)


class ErrorEvent(ProtocolModel):
    type: Literal["error"] = "error"
    error: ErrorDetail


RealtimePayload = Annotated[
    InputAudioDelta
    | InputAudioCommit
    | ResponseTextDelta
    | ResponseReasoningDelta
    | ResponseToolCallDelta
    | ResponseAudioDelta
    | ResponseInterrupted
    | ResponseCompleted
    | SessionStateChanged
    | RuntimeMetrics
    | ErrorEvent,
    Field(discriminator="type"),
]


class RealtimeFrame(ProtocolModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    session_id: UUID
    sequence: int = Field(ge=0)
    timestamp: AwareDatetime
    payload: RealtimePayload
