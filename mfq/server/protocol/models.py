"""Versioned public API models shared by MFQ Server clients and services."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Final, Literal
from uuid import UUID

from pydantic import AwareDatetime, Base64Bytes, BaseModel, ConfigDict, Field, SecretStr, model_validator

PROTOCOL_VERSION: Final[Literal["1.0"]] = "1.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
# The engine stores the request priority in a C++ ``int`` and compares larger
# values first, so the public contract uses the same type, default, and order.
REQUEST_PRIORITY_MIN: Final[int] = -(2**31)
REQUEST_PRIORITY_MAX: Final[int] = 2**31 - 1
# A longer client timeout is indistinguishable from "no deadline" and only
# hides a request mistake.
MAX_REQUEST_TIMEOUT_SECONDS: Final[float] = 86400.0


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


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class JobEventType(str, Enum):
    STATE = "state"
    PROGRESS = "progress"
    LOG = "log"
    ARTIFACT = "artifact"


class JobEventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


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


class VideoPart(ProtocolModel):
    type: Literal["video"] = "video"
    media: MediaRef
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    duration_ms: int | None = Field(default=None, ge=0)


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


class DocumentPart(ProtocolModel):
    type: Literal["document"] = "document"
    media: MediaRef
    name: str = Field(min_length=1, max_length=512)


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
    | VideoPart
    | AudioPart
    | TranscriptPart
    | DocumentPart
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
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    seed: int | None = Field(default=None, ge=0)
    enable_thinking: bool = True
    enable_vision: bool = True
    enable_mtp: bool = True
    mtp_max_draft_tokens: int = Field(default=3, ge=1, le=5)
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
    model: str = Field(default="", max_length=255, json_schema_extra={"deprecated": True})
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


class PortableMedia(ProtocolModel):
    sha256: str = Field(pattern=SHA256_PATTERN)
    mime_type: str = Field(min_length=1, max_length=255)
    data_base64: Base64Bytes
    document: dict[str, Any] | None = None


class PortableMessage(ProtocolModel):
    role: MessageRole
    parts: list[ContentPart] = Field(min_length=1)
    created_at: AwareDatetime


class SessionArchive(ProtocolModel):
    format: Literal["mfq-session-v1"] = "mfq-session-v1"
    session: SessionResource
    messages: list[PortableMessage]
    media: list[PortableMedia] = Field(default_factory=list)


class SessionImportResult(ProtocolModel):
    session: SessionResource
    messages_imported: int = Field(ge=0)
    media_imported: int = Field(ge=0)


class MessageList(ProtocolModel):
    data: list[Message]


class ForkSessionRequest(ProtocolModel):
    at_message_id: UUID | None = None
    include_message: bool = True
    title: str | None = Field(default=None, max_length=512)
    model: str | None = Field(default=None, min_length=1, max_length=255)


class RewindSessionRequest(ProtocolModel):
    expected_revision: int = Field(ge=0)
    at_message_id: UUID
    include_message: bool = True


class UpdateSessionRequest(ProtocolModel):
    title: str | None = Field(default=None, max_length=512)
    mode: SessionMode | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> UpdateSessionRequest:
        if not self.model_fields_set:
            raise ValueError("at least one session field must be provided")
        if "mode" in self.model_fields_set and self.mode is None:
            raise ValueError("mode cannot be null")
        if "metadata" in self.model_fields_set and self.metadata is None:
            raise ValueError("metadata cannot be null")
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


class ToolFunctionDefinition(ProtocolModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
    description: str | None = Field(default=None, max_length=8192)
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    strict: bool | None = False


class ToolDefinition(ProtocolModel):
    type: Literal["function"] = "function"
    function: ToolFunctionDefinition


class NamedToolChoiceFunction(ProtocolModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


class NamedToolChoice(ProtocolModel):
    type: Literal["function"] = "function"
    function: NamedToolChoiceFunction


ToolChoice = Literal["auto", "none", "required"] | NamedToolChoice


class TextResponseFormat(ProtocolModel):
    type: Literal["text"] = "text"


class JsonObjectResponseFormat(ProtocolModel):
    type: Literal["json_object"] = "json_object"


class JsonSchemaDefinition(ProtocolModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
    description: str | None = Field(default=None, max_length=8192)
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class JsonSchemaResponseFormat(ProtocolModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: JsonSchemaDefinition


ResponseFormat = Annotated[
    TextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat,
    Field(discriminator="type"),
]


class ResponsePerformance(ProtocolModel):
    # Runtime telemetry can gain fields independently of the public request
    # schema. Preserve those metrics instead of rejecting a valid response.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    prefill_tokens: int = Field(ge=0)
    ttft_ms: float = Field(ge=0.0)
    prefill_ms: float = Field(ge=0.0)
    prefill_tps: float = Field(ge=0.0)
    multimodal_ms: float = Field(default=0.0, ge=0.0)
    model_prefill_ms: float = Field(default=0.0, ge=0.0)
    processor_ms: float = Field(default=0.0, ge=0.0)
    complete_prefill_ms: float = Field(default=0.0, ge=0.0)
    complete_prefill_tps: float = Field(default=0.0, ge=0.0)
    decode_ms: float = Field(ge=0.0)
    decode_tps: float = Field(ge=0.0)
    generation_ms: float = Field(ge=0.0)
    complete_generation_ms: float = Field(default=0.0, ge=0.0)
    generation_tps: float = Field(ge=0.0)
    # Time the request spent waiting for a runtime slot inside the server. It is
    # separate from ttft_ms so queueing never hides native prefill latency.
    server_queue_wait_ms: float = Field(default=0.0, ge=0.0)
    mtp_available: bool = False
    mtp_used: bool = False
    mtp_cycles: int = Field(default=0, ge=0)
    mtp_drafted_tokens: int = Field(default=0, ge=0)
    mtp_accepted_tokens: int = Field(default=0, ge=0)
    mtp_acceptance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mtp_selected_depth: int = Field(default=0, ge=0, le=5)
    mtp_depth_0_cycles: int = Field(default=0, ge=0)
    mtp_depth_1_cycles: int = Field(default=0, ge=0)
    mtp_depth_2_cycles: int = Field(default=0, ge=0)
    mtp_depth_3_cycles: int = Field(default=0, ge=0)
    mtp_depth_4_cycles: int = Field(default=0, ge=0)
    mtp_depth_5_cycles: int = Field(default=0, ge=0)
    mtp_position_1_acceptance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mtp_position_2_acceptance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mtp_position_3_acceptance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mtp_position_4_acceptance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mtp_position_5_acceptance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mtp_depth_0_cycle_ms: float = Field(default=0.0, ge=0.0)
    mtp_depth_1_cycle_ms: float = Field(default=0.0, ge=0.0)
    mtp_depth_2_cycle_ms: float = Field(default=0.0, ge=0.0)
    mtp_depth_3_cycle_ms: float = Field(default=0.0, ge=0.0)
    mtp_depth_4_cycle_ms: float = Field(default=0.0, ge=0.0)
    mtp_depth_5_cycle_ms: float = Field(default=0.0, ge=0.0)
    mtp_target_ms: float = Field(default=0.0, ge=0.0)
    mtp_head_ms: float = Field(default=0.0, ge=0.0)
    mtp_rollback_ms: float = Field(default=0.0, ge=0.0)
    sampling: SamplingParams


class CreateResponseRequest(ProtocolModel):
    request_id: UUID
    model: str | None = Field(default=None, min_length=1, max_length=255)
    expected_revision: int = Field(ge=0)
    input: list[ContentPart] = Field(min_length=1)
    input_role: Literal["user", "tool"] = "user"
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    # Scheduling controls. A higher priority leaves the server queue and the
    # native scheduler first; the timeout bounds the whole request, queue wait
    # included, and expires with ``runtime_deadline_exceeded``.
    priority: int = Field(default=0, ge=REQUEST_PRIORITY_MIN, le=REQUEST_PRIORITY_MAX)
    timeout_seconds: float | None = Field(default=None, gt=0.0, le=MAX_REQUEST_TIMEOUT_SECONDS)
    system_prompt: str | None = Field(default=None, max_length=32768)
    include_reasoning_history: bool = True
    tools: list[ToolDefinition] = Field(default_factory=list, max_length=128)
    tool_choice: ToolChoice = "auto"
    response_format: ResponseFormat = Field(default_factory=TextResponseFormat)
    stream: bool = True

    @model_validator(mode="after")
    def validate_tools(self) -> CreateResponseRequest:
        if self.input_role == "tool":
            if len(self.input) != 1 or not isinstance(self.input[0], ToolResultPart):
                raise ValueError("tool input requires exactly one tool_result part")
        elif any(isinstance(part, ToolResultPart) for part in self.input):
            raise ValueError("tool_result parts require input_role=tool")
        names = [tool.function.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        if isinstance(self.tool_choice, NamedToolChoice):
            if self.tool_choice.function.name not in names:
                raise ValueError("named tool_choice must match a supplied tool")
        elif self.tool_choice == "required" and not names:
            raise ValueError("required tool_choice needs at least one tool")
        return self


class ResponseRequestSettings(ProtocolModel):
    sampling: SamplingParams
    model: str | None = Field(default=None, min_length=1, max_length=255)
    system_prompt: str | None = None
    include_reasoning_history: bool = True
    input_role: Literal["user", "tool"] = "user"
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: ToolChoice = "auto"
    response_format: ResponseFormat = Field(default_factory=TextResponseFormat)


class CreateGenerationPresetRequest(ProtocolModel):
    name: str = Field(min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1, max_length=255)
    mode: SessionMode | None = None
    settings: ResponseRequestSettings
    context_size: int | None = Field(default=None, ge=512)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateGenerationPresetRequest(CreateGenerationPresetRequest):
    pass


class GenerationPresetResource(ProtocolModel):
    id: UUID
    name: str
    model: str | None = None
    mode: SessionMode | None = None
    settings: ResponseRequestSettings
    context_size: int | None = Field(default=None, ge=512)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime
    updated_at: AwareDatetime


class GenerationPresetList(ProtocolModel):
    data: list[GenerationPresetResource]


class ResponseResource(ProtocolModel):
    id: UUID
    request_id: UUID
    session_id: UUID
    status: ResponseStatus
    output_message_id: UUID | None = None
    output: list[ContentPart] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    performance: ResponsePerformance | None = None
    settings: ResponseRequestSettings | None = None
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    error: ErrorDetail | None = None


class ResponseList(ProtocolModel):
    data: list[ResponseResource]


class MediaResource(ProtocolModel):
    media: MediaRef
    created_at: AwareDatetime


class CreateDocumentRequest(ProtocolModel):
    media_id: UUID
    name: str = Field(min_length=1, max_length=512)


class DocumentResource(ProtocolModel):
    media: MediaRef
    name: str
    text: str
    page_count: int | None = Field(default=None, ge=1)
    extractor: str
    created_at: AwareDatetime


class RuntimeInferencePolicy(ProtocolModel):
    mtp_enabled: bool = Field(default=True, strict=True)


class QsaKvStorageSettings(ProtocolModel):
    enabled: bool = False
    budget_bytes: int = Field(default=2 << 30, ge=1, le=1 << 50)
    ram_budget_bytes: int | None = Field(default=None, ge=0, le=1 << 50)


class KvMemorySettings(ProtocolModel):
    placement: Literal["auto", "auto_ssd", "vram", "ram"] = "auto"
    allocation: Literal["preallocated", "dynamic"] = "preallocated"
    gpu_budget_bytes: int | None = Field(default=None, ge=1, le=1 << 50)
    ram_preallocated: bool = Field(default=True, strict=True)
    ram_budget_bytes: int | None = Field(default=None, ge=0, le=1 << 50)

    @model_validator(mode="after")
    def validate_ram_storage(self) -> "KvMemorySettings":
        if not self.ram_preallocated and self.placement != "auto_ssd":
            raise ValueError("kv_ssd_required: disabling RAM preallocation requires SSD KV")
        return self


class KvQuantizationSettings(ProtocolModel):
    enabled: bool = Field(default=False, strict=True)
    bits: Literal[2, 2.5, 3, 3.5, 4, 6, 8] = 4
    algorithm: Literal["turboquant"] = "turboquant"


class RuntimeContextPolicy(ProtocolModel):
    fallback_context_size: int = Field(default=32768, ge=512, le=2147483647)
    max_context_size: int | None = Field(default=None, ge=512, le=2147483647)
    model_overrides: dict[str, Annotated[int, Field(ge=512, le=2147483647)]] = Field(default_factory=dict)
    model_yarn_enabled: dict[str, bool] = Field(default_factory=dict)
    model_qsa_kv_offload: dict[str, QsaKvStorageSettings] = Field(default_factory=dict)
    model_kv_quantization: dict[str, KvQuantizationSettings] = Field(default_factory=dict)
    model_kv_memory: dict[str, KvMemorySettings] = Field(default_factory=dict)


class RuntimeMemoryPolicy(ProtocolModel):
    total_limit_bytes: int | None = Field(default=None, ge=1)
    model_limit_bytes: int | None = Field(default=None, ge=1)
    prefix_limit_bytes: int | None = Field(default=None, ge=0)
    prefix_disk_limit_bytes: int | None = Field(default=None, ge=0)
    prefix_ram_limit_bytes: int | None = Field(default=None, ge=0)
    prefix_directory: str | None = Field(default=None, min_length=1, max_length=4096)


class RuntimeModelAliases(ProtocolModel):
    aliases: dict[str, str] = Field(default_factory=dict, max_length=128)


class QsaKvOffloadPolicy(ProtocolModel):
    enabled: bool = False
    budget_bytes: int = Field(default=2 << 30, ge=1, le=1 << 50)
    ram_budget_bytes: int | None = Field(default=None, ge=0, le=1 << 50)
    target_context: int = Field(default=262144, ge=512, le=2147483647)


class ConfigureQsaKvOffloadRequest(ProtocolModel):
    instance_id: UUID
    enabled: bool = False
    budget_bytes: int = Field(default=2 << 30, ge=1, le=1 << 50)
    ram_budget_bytes: int | None = Field(default=None, ge=0, le=1 << 50)


class ModelLoadContextSettings(ProtocolModel):
    automatic: bool = Field(default=True, strict=True)
    ssd_kv: bool = Field(default=False, strict=True)
    ram_preallocated: bool = Field(default=True, strict=True)
    vram_preallocated: bool = Field(default=True, strict=True)
    kv_quantization: KvQuantizationSettings = Field(default_factory=KvQuantizationSettings)


class RuntimeConcurrencySettings(ProtocolModel):
    """Per-model admission settings; zero timeout means an unlimited wait."""

    max_requests: int = Field(default=1, ge=1, le=32, strict=True)
    max_queued_requests: int = Field(default=32, ge=0, strict=True)
    queue_timeout_seconds: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class RuntimeConcurrencyLimits(ProtocolModel):
    """Server ceilings; a null timeout permits unlimited waiting."""

    max_requests: int = Field(ge=1, le=32)
    max_queued_requests: int = Field(ge=0)
    queue_timeout_seconds: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)


class RuntimeConcurrencyPolicy(ProtocolModel):
    """Defaults and ceilings available before any model is loaded."""

    defaults: RuntimeConcurrencySettings
    limits: RuntimeConcurrencyLimits


class RuntimeConcurrencyResource(ProtocolModel):
    """Keep requested rows distinct from negotiated worker admission."""

    requested: RuntimeConcurrencySettings
    limits: RuntimeConcurrencyLimits
    effective: RuntimeConcurrencySettings


class UpdateRuntimeConcurrencyRequest(ProtocolModel):
    """Partial edits; row changes are reported as requiring a reload."""

    max_requests: int | None = Field(default=None, ge=1, le=32, strict=True)
    max_queued_requests: int | None = Field(default=None, ge=0, strict=True)
    queue_timeout_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_settings(self) -> UpdateRuntimeConcurrencyRequest:
        fields = self.model_fields_set & {"max_requests", "max_queued_requests", "queue_timeout_seconds"}
        if not fields or any(getattr(self, name) is None for name in fields):
            raise ValueError("provide at least one non-null concurrency setting")
        return self


class RuntimeConcurrencyUpdate(ProtocolModel):
    """Live queue update result without an implicit worker restart."""

    concurrency: RuntimeConcurrencyResource
    reload_required: bool


class RuntimeConcurrencyReloadRequest(UpdateRuntimeConcurrencyRequest):
    """An explicit reload of one idle runtime with the supplied settings."""

    instance_id: UUID


class ModelLoadRequest(ProtocolModel):
    max_requests: int | None = Field(default=None, ge=1, le=32, strict=True)
    max_queued_requests: int | None = Field(default=None, ge=0, strict=True)
    queue_timeout_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=255)
    artifact_uri: str | None = Field(default=None, min_length=1)
    device_ids: list[str] = Field(default_factory=list)
    idle_ttl_seconds: int | None = Field(default=None, ge=0)
    pin: bool = False
    context_size: int | None = Field(default=None, ge=512, le=2147483647)
    yarn_enabled: bool | None = Field(default=None, strict=True)
    prefill_chunk_size: Literal["auto"] | Annotated[int, Field(ge=1, le=8192)] = "auto"
    moe_gpu_cache_gb: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    prefix_cache_max_sessions: int | None = Field(default=None, ge=0)
    prefix_cache_max_snapshots_per_session: int | None = Field(default=None, ge=0)
    prefix_cache_max_bytes: int | None = Field(default=None, ge=0)
    prefix_cache_enabled: bool = True
    prefix_cache_disk_bytes: int | None = Field(default=None, ge=0)
    prefix_cache_hot_bytes: int | None = Field(default=None, ge=0)
    prefix_cache_ram_bytes: int | None = Field(default=None, ge=0)
    prefix_cache_block_tokens: int | None = Field(default=None, ge=1, le=65536)
    prefix_cache_pending_bytes: int | None = Field(default=None, ge=0)
    sampling_defaults: SamplingParams | None = None

    qsa_kv_offload: QsaKvOffloadPolicy = Field(default_factory=QsaKvOffloadPolicy)
    kv_quantization: KvQuantizationSettings = Field(default_factory=KvQuantizationSettings)
    kv_memory: KvMemorySettings = Field(default_factory=KvMemorySettings)
    context_management: ModelLoadContextSettings | None = None

class CreateRuntimeProfileRequest(ProtocolModel):
    name: str = Field(min_length=1, max_length=64)
    load: ModelLoadRequest


class UpdateRuntimeProfileRequest(CreateRuntimeProfileRequest):
    pass


class RuntimeProfileResource(ProtocolModel):
    id: UUID
    name: str
    load: ModelLoadRequest
    artifact_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    artifact_modified_at: AwareDatetime
    drifted: bool = False
    drift_reason: str | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class RuntimeProfileList(ProtocolModel):
    data: list[RuntimeProfileResource]


class RuntimeProfileLoadRequest(ProtocolModel):
    allow_drift: bool = False


class DeleteWorkspaceArtifactRequest(ProtocolModel):
    artifact_uri: str = Field(pattern=r"^workspace://[^\x00]+$")


ApiKeyScope = Literal["inference", "models", "jobs", "admin"]
ApiKeyRole = Literal["viewer", "operator", "administrator"]


class CreateApiKeyRequest(ProtocolModel):
    name: str = Field(min_length=1, max_length=64)
    scopes: list[ApiKeyScope] = Field(default_factory=list, max_length=4)
    role: ApiKeyRole | None = None
    expires_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_scopes(self) -> CreateApiKeyRequest:
        if len(set(self.scopes)) != len(self.scopes):
            raise ValueError("API key scopes must be unique")
        if not self.scopes and self.role is None:
            raise ValueError("API key requires scopes or a role")
        return self


class ApiKeyResource(ProtocolModel):
    id: UUID
    name: str
    prefix: str
    scopes: list[ApiKeyScope]
    role: ApiKeyRole | None = None
    expires_at: AwareDatetime | None = None
    revoked_at: AwareDatetime | None = None
    created_at: AwareDatetime
    last_used_at: AwareDatetime | None = None


class ApiKeySecretResource(ProtocolModel):
    key: ApiKeyResource
    token: str


class ApiKeyList(ProtocolModel):
    data: list[ApiKeyResource]


class CreateRemoteNodeRequest(ProtocolModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
    url: str = Field(pattern=r"^https?://[^\s/@]+(?::[0-9]+)?(?:/[^\s]*)?$")
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    enabled: bool = True


class UpdateRemoteNodeRequest(CreateRemoteNodeRequest):
    pass


class RemoteNodeResource(ProtocolModel):
    id: UUID
    name: str
    url: str
    api_key_env: str | None = None
    enabled: bool
    healthy: bool = False
    models: list[str] = Field(default_factory=list)
    active_requests: int = Field(default=0, ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    last_checked_at: AwareDatetime | None = None
    error: str | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class RemoteNodeList(ProtocolModel):
    data: list[RemoteNodeResource]


class ModelUnloadRequest(ProtocolModel):
    instance_id: UUID
    force: bool = False


class RuntimeGpuUtilization(ProtocolModel):
    id: str | None = None
    name: str
    core_count: int | None = Field(default=None, ge=1)
    utilization_percent: float | None = Field(default=None, ge=0, le=100)
    memory_total_bytes: int | None = Field(default=None, ge=0)
    memory_used_bytes: int | None = Field(default=None, ge=0)
    memory_available_bytes: int | None = Field(default=None, ge=0)


class RuntimeDiskTraffic(ProtocolModel):
    name: str
    read_bytes_per_second: float | None = Field(default=None, ge=0)
    write_bytes_per_second: float | None = Field(default=None, ge=0)
    busy_percent: float | None = Field(default=None, ge=0, le=100)
    bandwidth_utilization_percent: float | None = Field(default=None, ge=0, le=100)


class RuntimeWeightTraffic(ProtocolModel):
    instance_id: str
    model: str
    expert_read_bytes_per_second: float | None = Field(default=None, ge=0)
    ple_read_bytes_per_second: float | None = Field(default=None, ge=0)
    engram_read_bytes_per_second: float | None = Field(default=None, ge=0)
    kv_read_bytes_per_second: float | None = Field(default=None, ge=0)
    kv_write_bytes_per_second: float | None = Field(default=None, ge=0)


class RuntimeResourceSnapshot(ProtocolModel):
    sampled_at: float = Field(ge=0)
    interval_seconds: float | None = Field(default=None, ge=0)
    cpu_name: str | None = None
    cpu_cores: int | None = Field(default=None, ge=1)
    cpu_utilization_percent: float | None = Field(default=None, ge=0, le=100)
    gpus: list[RuntimeGpuUtilization]
    disks: list[RuntimeDiskTraffic]
    memory_total_bytes: int | None = Field(default=None, ge=0)
    memory_used_bytes: int | None = Field(default=None, ge=0)
    memory_available_bytes: int | None = Field(default=None, ge=0)
    memory_bandwidth_bytes_per_second: float | None = Field(default=None, ge=0)
    memory_bandwidth_limit_bytes_per_second: int | None = Field(default=None, gt=0)
    memory_bandwidth_utilization_percent: float | None = Field(default=None, ge=0, le=100)
    weights: list[RuntimeWeightTraffic]


class RuntimeDeviceMemory(ProtocolModel):
    device_id: str
    resident_bytes: int | None = Field(default=None, ge=0)
    resident_weight_bytes: int | None = Field(default=None, ge=0)
    resident_buffer_bytes: int | None = Field(default=None, ge=0)
    kv_bytes: int | None = Field(default=None, ge=0)
    prefix_cache_bytes: int | None = Field(default=None, ge=0)
    memory_total_bytes: int | None = Field(default=None, ge=0)
    memory_available_bytes: int | None = Field(default=None, ge=0)


class RuntimeMemoryResources(ProtocolModel):
    """Physical residency and file payloads are separate, never added together.

    Null means the worker does not report this measurement, not zero usage.
    Contexts include live KV groups and resident prefix-cache sessions; blocks
    count resident prefixes only (disk-only prefix blocks are excluded).
    """

    resident_weight_bytes: int | None = Field(default=None, ge=0)
    devices: list[RuntimeDeviceMemory] = Field(default_factory=list)
    resident_buffer_bytes: int | None = Field(default=None, ge=0)
    wired_bytes: int | None = Field(default=None, ge=0)
    wired_limit_bytes: int | None = Field(default=None, ge=0)
    wired_available: bool | None = None
    kv_bytes: int | None = Field(default=None, ge=0)
    context_count: int | None = Field(default=None, ge=0)
    prefix_cache_blocks: int | None = Field(default=None, ge=0)
    prefix_cache_bytes: int | None = Field(default=None, ge=0)
    prefix_cache_limit_bytes: int | None = Field(default=None, ge=0)
    prefix_cache_ram_bytes: int | None = Field(default=None, ge=0)
    prefix_cache_ram_limit_bytes: int | None = Field(default=None, ge=0)
    prefix_cache_ram_blocks: int | None = Field(default=None, ge=0)
    ram_experts: bool | None = None
    ram_expert_bytes: int | None = Field(default=None, ge=0)
    ram_kv: bool | None = None
    ram_kv_bytes: int | None = Field(default=None, ge=0)
    ram_kv_limit_bytes: int | None = Field(default=None, ge=0)
    ssd_experts: bool | None = None
    ssd_expert_bytes: int | None = Field(default=None, ge=0)
    ssd_ple: bool | None = None
    ssd_ple_bytes: int | None = Field(default=None, ge=0)

    ssd_kv: bool | None = None
    ssd_kv_bytes: int | None = Field(default=None, ge=0)
    streaming_kv_resident_bytes: int | None = Field(default=None, ge=0)
    streaming_kv_budget_bytes: int | None = Field(default=None, ge=0)
    streaming_kv_pending_bytes: int | None = Field(default=None, ge=0)
    ssd_kv_read_bytes: int | None = Field(default=None, ge=0)
    ssd_kv_written_bytes: int | None = Field(default=None, ge=0)
    ssd_kv_reads: int | None = Field(default=None, ge=0)
    ssd_kv_hits: int | None = Field(default=None, ge=0)


class RuntimeInstanceResource(ProtocolModel):
    concurrency: RuntimeConcurrencyResource | None = None
    id: UUID
    model: str
    state: RuntimeInstanceState
    devices: list[str]
    active_sessions: int = Field(ge=0)
    queued_requests: int = Field(ge=0)
    request_capacity: int = Field(default=1, ge=1)
    batch_capacity: int = Field(default=1, ge=1)
    queue_wait_ms_avg: float = Field(default=0.0, ge=0.0)
    queue_wait_ms_max: float = Field(default=0.0, ge=0.0)
    rejected_requests: int = Field(default=0, ge=0)
    resident_bytes: int | None = Field(default=None, ge=0)
    kv_bytes: int | None = Field(default=None, ge=0)
    memory: RuntimeMemoryResources | None = None
    context_size: int | None = Field(default=None, ge=1)
    context_capacity: int | None = Field(default=None, ge=1)
    started_at: AwareDatetime | None = None
    last_used_at: AwareDatetime | None = None
    idle_ttl_seconds: int | None = Field(default=None, ge=0)
    pinned: bool = False
    mtp_supported: bool = False
    mtp_available: bool = False
    qsa_kv_offload_supported: bool = False
    kv_quantization_supported: bool = False
    kv_memory_supported: bool = False
    kv_memory: KvMemorySettings = Field(default_factory=KvMemorySettings)
    kv_quantization: KvQuantizationSettings = Field(default_factory=KvQuantizationSettings)
    qsa_kv_offload: QsaKvOffloadPolicy = Field(default_factory=QsaKvOffloadPolicy)
    identity: RuntimeIdentity | None = None
    error: ErrorDetail | None = None


class UpdateRuntimeInstanceRequest(ProtocolModel):
    pinned: bool | None = None
    idle_ttl_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_update(self) -> UpdateRuntimeInstanceRequest:
        if not self.model_fields_set:
            raise ValueError("at least one runtime policy field is required")
        if "pinned" in self.model_fields_set and self.pinned is None:
            raise ValueError("pinned must be boolean")
        return self


class RuntimeInstanceList(ProtocolModel):
    data: list[RuntimeInstanceResource]


class RuntimeMetricSnapshot(ProtocolModel):
    sequence: int = Field(ge=1)
    instance_id: UUID | None = None
    model: str | None = Field(default=None, max_length=255)
    values: dict[str, Any]
    captured_at: AwareDatetime


class RuntimeMetricList(ProtocolModel):
    data: list[RuntimeMetricSnapshot]


class RuntimeLogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RuntimeLogEntry(ProtocolModel):
    sequence: int = Field(ge=1)
    instance_id: UUID | None = None
    level: RuntimeLogLevel
    message: str
    fields: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime


class RuntimeLogList(ProtocolModel):
    data: list[RuntimeLogEntry]


class ModelArtifactResource(ProtocolModel):
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=255)
    architecture: str = Field(min_length=1, max_length=128)
    format: Literal["mfq", "hf"] = "mfq"
    shard_count: int = Field(ge=1)
    missing_shards: int = Field(default=0, ge=0)
    total_bytes: int = Field(ge=0)
    estimated_resident_weight_bytes: int | None = Field(default=None, ge=0)
    estimated_weight_bytes_by_role: dict[Literal['dense', 'experts', 'embedding'], Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    ssd_ple_bytes: int | None = Field(default=None, ge=0)
    context_capacity: int | None = Field(default=None, ge=1)
    yarn_context_capacity: int | None = Field(default=None, ge=1)
    yarn_max_factor: float | None = Field(default=None, gt=1)
    tensor_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    dtypes: list[str] = Field(default_factory=list)
    complete: bool
    loadable: bool
    modified_at: AwareDatetime
    error: str | None = None


class ModelArtifactList(ProtocolModel):
    data: list[ModelArtifactResource]


class ModelDeletionPlan(ProtocolModel):
    model_id: str
    name: str
    files: list[str]
    total_bytes: int = Field(ge=0)
    token: str = Field(pattern=SHA256_PATTERN)


class DeleteModelRequest(ProtocolModel):
    token: str = Field(pattern=SHA256_PATTERN)


class ModelDirectoryEntry(ProtocolModel):
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=512)
    model_file_count: int = Field(default=0, ge=0)


class ModelDirectoryFile(ProtocolModel):
    name: str = Field(min_length=1, max_length=512)
    byte_size: int = Field(ge=0)


class OpenModelDirectoryRequest(ProtocolModel):
    directory_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class OpenModelDirectoryResult(ProtocolModel):
    opened: Literal[True] = True


class ModelDirectoryList(ProtocolModel):
    current_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    current_name: str | None = Field(default=None, max_length=512)
    current_path: str | None = Field(default=None, min_length=1, max_length=4096)
    parent_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    model_file_count: int = Field(default=0, ge=0)
    files: list[ModelDirectoryFile] = Field(default_factory=list)
    can_open_directory: bool = False
    can_open_in_finder: bool = False
    data: list[ModelDirectoryEntry]


class RegisterModelDirectoryRequest(ProtocolModel):
    directory_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    path: str | None = Field(default=None, min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_source(self) -> RegisterModelDirectoryRequest:
        if (self.directory_id is None) == (self.path is None):
            raise ValueError("exactly one of directory_id or path is required")
        return self


class ConfigureDownloadDirectoryRequest(ProtocolModel):
    directory_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    evict_prefix_cache: bool | None = None


class HubDownloadSource(ProtocolModel):
    provider: Literal['huggingface', 'modelscope']
    repo_id: str = Field(pattern=r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
    revision: str | None = Field(default=None, min_length=1, max_length=255)


class HubDownloadProbeRequest(ProtocolModel):
    sources: list[HubDownloadSource] = Field(min_length=1, max_length=2)


class HubDownloadReuseRequest(HubDownloadSource):
    destination: str = Field(min_length=1, max_length=1024)
    download_directory: str | None = Field(default=None, min_length=1, max_length=4096)


class HubDownloadReuseResponse(ProtocolModel):
    reusable_files: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)


class HubModelSummary(ProtocolModel):
    provider: Literal["huggingface", "modelscope"]
    repo_id: str = Field(min_length=3, max_length=255)
    source_url: str | None = Field(default=None, max_length=2048)
    author: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=2048)
    downloads: int = Field(default=0, ge=0)
    likes: int = Field(default=0, ge=0)
    total_bytes: int = Field(default=0, ge=0)
    updated_at: AwareDatetime | None = None


class HubModelSearchResult(ProtocolModel):
    data: list[HubModelSummary]


class HubModelFile(ProtocolModel):
    name: str = Field(min_length=1, max_length=1024)
    byte_size: int = Field(default=0, ge=0)
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    weight_bytes: int | None = Field(default=None, ge=0)
    weight_bytes_by_dtype: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    estimated_weight_bytes_by_role: dict[Literal['dense', 'experts', 'embedding'], Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    ssd_ple_bytes: int | None = Field(default=None, ge=0)


class ModelConfigurationStatus(ProtocolModel):
    status: Literal["recommended", "warning", "unknown"]
    recommendation: Literal[
        "three_stars",
        "two_stars",
        "one_star",
        "caution",
        "not_recommended",
        "unknown",
    ]
    required_memory_bytes: int | None = Field(default=None, ge=0)
    available_memory_bytes: int | None = Field(default=None, ge=0)
    reasons: list[str] = Field(default_factory=list)


class HubModelVariant(ProtocolModel):
    id: str = Field(min_length=1, max_length=1024)
    label: str = Field(min_length=1, max_length=255)
    format: Literal["mfq", "hf", "gguf", "unknown"]
    precision: str | None = Field(default=None, max_length=128)
    files: list[str] = Field(default_factory=list, max_length=256)
    byte_size: int = Field(default=0, ge=0)
    resident_weight_bytes: int | None = Field(default=None, ge=0)
    estimated_resident_weight_bytes: int | None = Field(default=None, ge=0)
    estimated_weight_bytes_by_role: dict[Literal['dense', 'experts', 'embedding'], Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    ssd_ple_bytes: int | None = Field(default=None, ge=0)
    configuration: ModelConfigurationStatus


class HubMemoryPool(ProtocolModel):
    kind: Literal["uma", "vram", "ram"]
    device: str | None = Field(default=None, max_length=255)
    capacity_bytes: int | None = Field(default=None, ge=0)
    bandwidth_bytes_per_second: int | None = Field(default=None, gt=0)


class HubSystemProfile(ProtocolModel):
    platform: str = Field(min_length=1, max_length=64)
    machine: str = Field(min_length=1, max_length=64)
    backend: Literal["metal", "cuda", "rocm", "cpu", "unknown"]
    cpu_name: str | None = Field(default=None, max_length=255)
    cpu_cores: int | None = Field(default=None, ge=1)
    gpu_names: list[str] = Field(default_factory=list)
    gpu_cores: int | None = Field(default=None, ge=1)
    physical_memory_bytes: int | None = Field(default=None, ge=0)
    available_memory_bytes: int | None = Field(default=None, ge=0)
    runtime_memory_budget_bytes: int | None = Field(default=None, ge=0)
    memory_pools: list[HubMemoryPool] = Field(default_factory=list)


class ModelParameterBreakdown(ProtocolModel):
    total: int = Field(ge=0)
    dense: int = Field(ge=0)
    routed_experts: int = Field(default=0, ge=0)
    ple: int = Field(default=0, ge=0)
    active: int | None = Field(default=None, ge=0)


class ModelCacheComponent(ProtocolModel):
    group: str | None = None
    name: str | None = None
    subgroup: str | None = None
    layers: int | None = Field(default=None, ge=1)
    bytes_per_row: int = Field(gt=0)
    tokens_per_row: int = Field(default=1, ge=1)
    allocation: Literal['exact', 'power_of_two'] = 'exact'
    minimum_rows: int = Field(default=0, ge=0)
    row_rounding: Literal['ceil', 'floor'] = 'ceil'
    active_after: int = Field(default=0, ge=0)
    max_read_rows_per_token: int | None = Field(default=None, ge=1)
    head_dimension: int | None = Field(default=None, ge=1)
    kv_heads: int | None = Field(default=None, ge=1)


class ModelCacheFixedComponent(ProtocolModel):
    group: str
    name: str
    layers: int | None = Field(default=None, ge=1)
    bytes: int = Field(ge=0)


class ModelCacheProfile(ProtocolModel):
    max_context: int = Field(ge=1)
    fixed_bytes: int = Field(default=0, ge=0)
    fixed_components: list[ModelCacheFixedComponent] = Field(default_factory=list)
    components: list[ModelCacheComponent] = Field(default_factory=list)


class ModelAttentionProfile(ProtocolModel):
    sparse: bool
    compressed: bool
    hybrid: bool


class HubModelInfo(HubModelSummary):
    revision: str
    files: list[HubModelFile]
    tags: list[str] = Field(default_factory=list)
    license: str | None = Field(default=None, max_length=255)
    library: str | None = Field(default=None, max_length=255)
    pipeline_tag: str | None = Field(default=None, max_length=255)
    architectures: list[str] = Field(default_factory=list)
    attention_profile: ModelAttentionProfile | None = None
    config_inspected: bool = False
    modalities: list[str] = Field(default_factory=list)
    parameter_count: int | None = Field(default=None, ge=0)
    parameter_breakdown: ModelParameterBreakdown | None = None
    mtp_supported: bool | None = None
    cache_profile: ModelCacheProfile | None = None
    ple_parameter_count: int | None = Field(default=None, ge=0)
    published_at: AwareDatetime | None = None
    gated: bool = False
    runtime_compatible: bool | None = None
    variants: list[HubModelVariant] = Field(default_factory=list)


class HubReferenceRequest(ProtocolModel):
    reference: str = Field(min_length=1, max_length=2048)
    fallback_provider: Literal["huggingface", "modelscope"] = "huggingface"


class OfficialModelSource(ProtocolModel):
    provider: Literal["huggingface", "modelscope"]
    repo_id: str = Field(min_length=3, max_length=255)
    revision: str | None = Field(default=None, max_length=255)
    url: str = Field(min_length=1, max_length=2048)
    available: bool = True


class OfficialModelInfo(ProtocolModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    family: str = Field(min_length=1, max_length=255)
    architecture: str = Field(min_length=1, max_length=128)
    attention_profile: ModelAttentionProfile | None = None
    description: str = Field(min_length=1, max_length=2048)
    description_zh: str = Field(min_length=1, max_length=2048)
    parameter_label: str | None = Field(default=None, max_length=128)
    active_parameter_label: str | None = Field(default=None, max_length=128)
    parameter_breakdown: ModelParameterBreakdown | None = None
    mtp_supported: bool | None = None
    cache_profile: ModelCacheProfile | None = None
    modalities: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    precision_options: list[str] = Field(default_factory=list)
    license: str | None = Field(default=None, max_length=255)
    supports_ssd_streaming: bool = False
    sources: list[OfficialModelSource]
    selected_source: OfficialModelSource
    revision: str
    downloads: int = Field(default=0, ge=0)
    likes: int = Field(default=0, ge=0)
    updated_at: AwareDatetime | None = None
    published_at: AwareDatetime | None = None
    variants: list[HubModelVariant] = Field(default_factory=list)
    configuration: ModelConfigurationStatus


class OfficialModelList(ProtocolModel):
    system: HubSystemProfile
    data: list[OfficialModelInfo]
    refreshing: bool = False


class ArtifactLineageResource(ProtocolModel):
    id: UUID
    artifact_uri: str = Field(min_length=1, max_length=2048)
    artifact_name: str = Field(min_length=1, max_length=255)
    producer_job_id: UUID
    producer_kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    source_uris: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    validation_job_ids: list[UUID] = Field(default_factory=list)
    created_at: AwareDatetime


class ArtifactLineageList(ProtocolModel):
    data: list[ArtifactLineageResource]


DatasetKind = Literal["wikitext2", "custom"]
EvaluationKind = Literal["perplexity", "kernel_benchmark", "inference_benchmark", "accuracy_benchmark"]


class EvaluationToolsResource(ProtocolModel):
    workspace_root: str | None = None
    api_base: str
    quality_available: bool
    reference_available: bool = False
    benchmark_available: bool
    accuracy_available: bool = False
    task_benchmarks: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ProbabilityScoreRequest(ProtocolModel):
    instance_id: UUID
    prompt: str = Field(min_length=1, max_length=65536)
    continuations: list[Annotated[str, Field(min_length=1, max_length=16384)]] = Field(min_length=1, max_length=64)
    mode: Literal["continuation", "next_token"] = "continuation"


class ProbabilityScore(ProtocolModel):
    token_ids: list[int]
    token_logprobs: list[Annotated[float, Field(allow_inf_nan=False, le=0.00001)]]
    log_likelihood: float = Field(allow_inf_nan=False)


class ProbabilityScoreResult(ProtocolModel):
    prompt_tokens: int = Field(ge=1)
    scores: list[ProbabilityScore]
    log_base: Literal["e"]
    tokenization: Literal["raw-no-special-tokens"]


class OfficialDatasetResource(ProtocolModel):
    id: str
    name: str
    kind: DatasetKind
    repository: str
    revision: str
    filename: str
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int
    rows: int
    license: str
    format: str = "parquet"
    split: str = "test"
    task: str | None = None
    origin: Literal["huggingface", "github"] = "huggingface"


class OfficialDatasetList(ProtocolModel):
    data: list[OfficialDatasetResource]


class CreateDatasetRequest(ProtocolModel):
    name: str = Field(min_length=1, max_length=128)
    kind: DatasetKind = "custom"
    artifact_uri: str = Field(pattern=r"^workspace://[^\x00]+$")
    source_uri: str | None = Field(default=None, max_length=2048)
    revision: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetResource(ProtocolModel):
    id: UUID
    name: str
    kind: DatasetKind
    artifact_uri: str
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)
    source_uri: str | None = None
    revision: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime
    updated_at: AwareDatetime


class DatasetList(ProtocolModel):
    data: list[DatasetResource]


class EvaluationResultResource(ProtocolModel):
    id: UUID
    job_id: UUID
    kind: EvaluationKind
    model_id: str = Field(min_length=1, max_length=255)
    metrics: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    dataset_id: UUID | None = None
    dataset_manifest: dict[str, Any] = Field(default_factory=dict)
    hardware_identity: dict[str, Any] = Field(default_factory=dict)
    runtime_identity: dict[str, Any] = Field(default_factory=dict)
    comparison_key: str = Field(pattern=SHA256_PATTERN)
    created_at: AwareDatetime


class EvaluationResultList(ProtocolModel):
    data: list[EvaluationResultResource]


class CompareEvaluationsRequest(ProtocolModel):
    evaluation_ids: list[UUID] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def validate_evaluations(self) -> CompareEvaluationsRequest:
        if len(set(self.evaluation_ids)) != len(self.evaluation_ids):
            raise ValueError("evaluation IDs must be unique")
        return self


class EvaluationComparisonRow(ProtocolModel):
    evaluation: EvaluationResultResource
    deltas: dict[str, float | None] = Field(default_factory=dict)
    ratios: dict[str, float | None] = Field(default_factory=dict)


class EvaluationComparisonResource(ProtocolModel):
    comparison_key: str = Field(pattern=SHA256_PATTERN)
    baseline_id: UUID
    metrics: list[str]
    rows: list[EvaluationComparisonRow]


class ModelFeatureSet(ProtocolModel):
    text: bool = True
    image_input: bool = False
    video_input: bool = False
    audio_input: bool = False
    audio_output: bool = False
    full_duplex: bool = False
    mtp: bool = False


class ModelCapabilities(ProtocolModel):
    architecture_family: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=255)
    features: ModelFeatureSet = Field(default_factory=ModelFeatureSet)


class RuntimeCapabilitiesResource(ProtocolModel):
    model: str = Field(min_length=1, max_length=255)
    model_type: str = Field(min_length=1, max_length=128)
    model_capabilities: ModelCapabilities
    vision_available: bool = False
    mtp_available: bool = False
    duplex_available: bool = False
    # Rows the worker can batch concurrently. The control plane adopts this
    # report after load instead of assuming the requested concurrency.
    batch_capacity: int = Field(default=1, ge=1)


class OperationAccepted(ProtocolModel):
    operation_id: UUID
    status: Literal["accepted"] = "accepted"


class CreateJobRequest(ProtocolModel):
    kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    payload: dict[str, Any] = Field(default_factory=dict)


class JobKindResource(ProtocolModel):
    kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    payload_schema: dict[str, Any]


class JobKindList(ProtocolModel):
    data: list[JobKindResource]


class JobResource(ProtocolModel):
    id: UUID
    kind: str
    status: JobStatus
    payload: dict[str, Any]
    progress: float = Field(ge=0.0, le=1.0)
    progress_data: dict[str, Any] = Field(default_factory=dict)
    cancel_requested: bool = False
    result: dict[str, Any] | None = None
    error: ErrorDetail | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None


class JobList(ProtocolModel):
    data: list[JobResource]


class JobEventResource(ProtocolModel):
    job_id: UUID
    sequence: int = Field(ge=1)
    type: JobEventType
    level: JobEventLevel
    message: str | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime


class JobEventList(ProtocolModel):
    data: list[JobEventResource]


class AuthenticationPolicyRequest(ProtocolModel):
    enabled: bool = Field(strict=True)
    api_key: SecretStr | None = None


class RuntimeListenerRequest(ProtocolModel):
    port: int = Field(strict=True, ge=1, le=65535)
    protocol: Literal["openai", "anthropic"] = "openai"


class RuntimeReloadRequest(ProtocolModel):
    apply_global_cap: bool = Field(default=False, strict=True)
    context_size: int | None = Field(default=None, ge=512, le=2147483647)
    instance_id: UUID | None = None
    yarn_enabled: bool | None = Field(default=None, strict=True)
    qsa_kv_offload: QsaKvStorageSettings | None = None
    kv_quantization: KvQuantizationSettings | None = None
    kv_memory: KvMemorySettings | None = None


class RuntimeCacheClearRequest(ProtocolModel):
    instance_id: UUID | None = None


class RuntimePrefixCachePurgeRequest(ProtocolModel):
    namespace: str | None = Field(default=None, pattern=SHA256_PATTERN)


class RuntimeCacheTrimRequest(ProtocolModel):
    target_bytes: int = Field(default=0, ge=0)
    instance_id: UUID | None = None


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
    performance: ResponsePerformance | None = None


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
