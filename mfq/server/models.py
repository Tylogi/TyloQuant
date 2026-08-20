"""Versioned public API models shared by MFQ Server clients and services."""

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


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class McpTransport(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


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
    sampling: SamplingParams


class CreateResponseRequest(ProtocolModel):
    request_id: UUID
    expected_revision: int = Field(ge=0)
    input: list[ContentPart] = Field(min_length=1)
    input_role: Literal["user", "tool"] = "user"
    sampling: SamplingParams = Field(default_factory=SamplingParams)
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
    context_size: int = Field(default=32768, ge=512)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateGenerationPresetRequest(CreateGenerationPresetRequest):
    pass


class GenerationPresetResource(ProtocolModel):
    id: UUID
    name: str
    model: str | None = None
    mode: SessionMode | None = None
    settings: ResponseRequestSettings
    context_size: int = Field(ge=512)
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


class CreateMcpServerRequest(ProtocolModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
    transport: McpTransport
    enabled: bool = False
    url: str | None = Field(default=None, max_length=2048)
    command: str | None = Field(default=None, max_length=1024)
    args: list[str] = Field(default_factory=list, max_length=128)
    header_env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)

    @model_validator(mode="after")
    def validate_transport(self) -> CreateMcpServerRequest:
        if self.transport == McpTransport.STREAMABLE_HTTP:
            if not self.url or self.command is not None or self.args:
                raise ValueError("streamable_http needs a URL and no command")
            if not self.url.startswith(("http://", "https://")):
                raise ValueError("MCP URL must use HTTP or HTTPS")
        elif not self.command or self.url is not None or self.header_env:
            raise ValueError("stdio needs a command and cannot use URL headers")
        for header, variable in self.header_env.items():
            if not header.strip() or not variable.isidentifier():
                raise ValueError("MCP headers must map to environment variable names")
        return self


class UpdateMcpServerRequest(ProtocolModel):
    enabled: bool


class McpServerResource(ProtocolModel):
    id: UUID
    name: str
    transport: McpTransport
    enabled: bool
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    header_env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float
    created_at: AwareDatetime
    updated_at: AwareDatetime


class McpServerList(ProtocolModel):
    data: list[McpServerResource]


class McpToolResource(ProtocolModel):
    server_id: UUID
    server: str
    name: str
    qualified_name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})


class McpToolList(ProtocolModel):
    data: list[McpToolResource]
    errors: dict[str, str] = Field(default_factory=dict)


class McpToolCallRequest(ProtocolModel):
    name: str = Field(min_length=1, max_length=255)
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirm: bool = False


class McpToolCallResult(ProtocolModel):
    server: str
    name: str
    content: list[dict[str, Any]] = Field(default_factory=list)
    structured_content: dict[str, Any] | None = None
    is_error: bool = False


class ModelLoadRequest(ProtocolModel):
    model: str = Field(min_length=1, max_length=255)
    artifact_uri: str | None = Field(default=None, min_length=1)
    device_ids: list[str] = Field(default_factory=list)
    idle_ttl_seconds: int | None = Field(default=None, ge=0)
    pin: bool = False
    context_size: int = Field(default=32768, ge=512)
    prefill_chunk_size: int = Field(default=2048, ge=1)
    moe_gpu_cache_gb: float | None = Field(default=None, ge=0.0)
    prefix_cache_max_sessions: int | None = Field(default=None, ge=0)
    prefix_cache_max_snapshots_per_session: int | None = Field(default=None, ge=0)
    prefix_cache_max_bytes: int | None = Field(default=None, ge=0)
    sampling_defaults: SamplingParams | None = None

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


class RuntimeInstanceResource(ProtocolModel):
    id: UUID
    model: str
    state: RuntimeInstanceState
    devices: list[str]
    active_sessions: int = Field(ge=0)
    queued_requests: int = Field(ge=0)
    resident_bytes: int | None = Field(default=None, ge=0)
    kv_bytes: int | None = Field(default=None, ge=0)
    context_size: int | None = Field(default=None, ge=1)
    started_at: AwareDatetime | None = None
    last_used_at: AwareDatetime | None = None
    identity: RuntimeIdentity | None = None
    error: ErrorDetail | None = None


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
    format: Literal["mfq"] = "mfq"
    shard_count: int = Field(ge=1)
    total_bytes: int = Field(ge=0)
    tensor_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    dtypes: list[str] = Field(default_factory=list)
    complete: bool
    loadable: bool
    modified_at: AwareDatetime
    error: str | None = None


class ModelArtifactList(ProtocolModel):
    data: list[ModelArtifactResource]


class ModelDirectoryEntry(ProtocolModel):
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=512)
    model_file_count: int = Field(default=0, ge=0)


class ModelDirectoryList(ProtocolModel):
    current_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    current_name: str | None = Field(default=None, max_length=512)
    current_path: str | None = Field(default=None, min_length=1, max_length=4096)
    parent_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    model_file_count: int = Field(default=0, ge=0)
    data: list[ModelDirectoryEntry]


class RegisterModelDirectoryRequest(ProtocolModel):
    directory_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    path: str | None = Field(default=None, min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_source(self) -> RegisterModelDirectoryRequest:
        if (self.directory_id is None) == (self.path is None):
            raise ValueError("exactly one of directory_id or path is required")
        return self


class HubModelSummary(ProtocolModel):
    provider: Literal["huggingface", "modelscope"]
    repo_id: str = Field(min_length=3, max_length=255)
    downloads: int = Field(default=0, ge=0)
    likes: int = Field(default=0, ge=0)
    total_bytes: int = Field(default=0, ge=0)
    updated_at: AwareDatetime | None = None


class HubModelSearchResult(ProtocolModel):
    data: list[HubModelSummary]


class HubModelFile(ProtocolModel):
    name: str = Field(min_length=1, max_length=1024)
    byte_size: int = Field(default=0, ge=0)


class HubModelInfo(HubModelSummary):
    revision: str
    files: list[HubModelFile]
    tags: list[str] = Field(default_factory=list)


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
EvaluationKind = Literal["perplexity", "kernel_benchmark"]


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
