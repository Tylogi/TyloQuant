export type SessionMode = "text" | "voice" | "full_duplex";
export type SessionState =
  | "idle"
  | "listening"
  | "processing"
  | "speaking"
  | "interrupted"
  | "reconnecting"
  | "error"
  | "closed";
export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface MediaRef {
  id: string;
  sha256: string;
  mime_type: string;
  byte_size: number;
}

export interface MediaResource {
  media: MediaRef;
  created_at: string;
}

export interface DocumentResource {
  media: MediaRef;
  name: string;
  text: string;
  page_count?: number | null;
  extractor: string;
  created_at: string;
}

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "transcript"; text: string; language?: string | null }
  | { type: "document"; media: MediaRef; name: string }
  | { type: "image"; media: MediaRef; width?: number | null; height?: number | null }
  | {
      type: "video";
      media: MediaRef;
      width?: number | null;
      height?: number | null;
      duration_ms?: number | null;
    }
  | {
      type: "audio" | "generated_audio";
      media: MediaRef;
      sample_rate_hz: number;
      channels: number;
      duration_ms?: number | null;
    }
  | { type: "tool_call"; call_id: string; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; call_id: string; result: unknown; is_error: boolean };

export interface Session {
  id: string;
  model: string;
  mode: SessionMode;
  state: SessionState;
  revision: number;
  title: string | null;
  runtime_instance_id: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface SessionArchive {
  format: "mfq-session-v1";
  session: Session;
  messages: Array<{ role: MessageRole; parts: ContentPart[]; created_at: string }>;
  media: Array<{
    sha256: string;
    mime_type: string;
    data_base64: string;
    document?: Record<string, unknown> | null;
  }>;
}

export interface Message {
  id: string;
  role: MessageRole;
  parts: ContentPart[];
  parent_id: string | null;
  created_at: string;
}

export interface SamplingParams {
  max_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  presence_penalty: number;
  frequency_penalty: number;
  repetition_penalty: number;
  seed?: number | null;
  enable_thinking: boolean;
  reasoning_effort?: string | null;
}

export interface ResponsePerformance {
  prefill_tokens: number;
  ttft_ms: number;
  prefill_ms: number;
  prefill_tps: number;
  multimodal_ms: number;
  model_prefill_ms: number;
  processor_ms: number;
  complete_prefill_ms: number;
  complete_prefill_tps: number;
  decode_ms: number;
  decode_tps: number;
  generation_ms: number;
  complete_generation_ms: number;
  generation_tps: number;
  sampling: SamplingParams;
}

export interface ResponseResource {
  id: string;
  request_id: string;
  session_id: string;
  status: "running" | "completed" | "failed" | "cancelled";
  output_message_id?: string | null;
  output: ContentPart[];
  finish_reason?: string | null;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  } | null;
  performance?: ResponsePerformance | null;
  settings?: {
    sampling: SamplingParams;
    system_prompt?: string | null;
    include_reasoning_history: boolean;
  } | null;
  created_at: string;
  completed_at?: string | null;
}

export interface GenerationPresetResource {
  id: string;
  name: string;
  model?: string | null;
  mode?: SessionMode | null;
  settings: {
    sampling: SamplingParams;
    system_prompt?: string | null;
    include_reasoning_history: boolean;
    input_role: "user" | "tool";
    tools: Array<Record<string, unknown>>;
    tool_choice: "auto" | "none" | "required" | Record<string, unknown>;
    response_format: { type: "text" | "json_object" | "json_schema"; [key: string]: unknown };
  };
  context_size: number;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ModelFeatureSet {
  text: boolean;
  image_input: boolean;
  video_input: boolean;
  audio_input: boolean;
  audio_output: boolean;
  full_duplex: boolean;
}

export interface ModelCapabilities {
  architecture_family: string;
  source: string;
  features: ModelFeatureSet;
}

export interface RuntimeCapabilities {
  model: string;
  model_type: string;
  model_capabilities: ModelCapabilities;
  duplex_available: boolean;
}

export interface RuntimeRequestMetrics {
  id?: string;
  endpoint?: string;
  stream?: boolean;
  prompt_tokens?: number;
  prefill_tokens?: number;
  completion_tokens?: number;
  prefill_tps?: number;
  prefill_ms?: number;
  multimodal_ms?: number;
  model_prefill_ms?: number;
  processor_ms?: number;
  complete_prefill_ms?: number;
  complete_prefill_tps?: number;
  decode_tps?: number;
  decode_ms?: number;
  ttft_ms?: number;
  generation_ms?: number;
  complete_generation_ms?: number;
  generation_tps?: number;
  finish_reason?: string;
  client_connected?: boolean;
  completed_at?: number;
}

export interface RuntimeStatus {
  instance_id?: string;
  runtime_state?: string;
  model?: string;
  model_type?: string;
  model_capabilities?: ModelCapabilities;
  duplex_available?: boolean;
  active_requests?: number;
  total_requests?: number;
  failed_requests?: number;
  total_prompt_tokens?: number;
  total_completion_tokens?: number;
  uptime_seconds?: number;
  max_context?: number;
  context_capacity?: number;
  reloading?: boolean;
  process_resident_bytes?: number | null;
  mlx_active_bytes?: number;
  mlx_cache_bytes?: number;
  mlx_peak_bytes?: number;
  cuda_allocated_bytes?: number;
  cuda_reserved_bytes?: number;
  device_free_bytes?: number;
  device_total_bytes?: number;
  prefix_cache_queries?: number;
  prefix_cache_hits?: number;
  prefix_cache_hit_tokens?: number;
  prefix_cache_sessions?: number;
  prefix_cache_snapshots?: number;
  prefix_cache_tokens?: number;
  prefix_cache_bytes?: number;
  prefix_cache_max_sessions?: number;
  prefix_cache_max_snapshots_per_session?: number;
  prefix_cache_max_bytes?: number;
  sampling_defaults?: Partial<SamplingParams>;
  duplex_sampling_defaults?: {
    system_prompt?: string;
    temperature?: number;
    top_k?: number;
    top_p?: number;
    text_repetition_penalty?: number;
    [key: string]: unknown;
  };
  tts_sampling_defaults?: {
    temperature?: number;
    repetition_penalty?: number;
    [key: string]: unknown;
  };
  chat_template_capabilities?: {
    thinking?: { supported?: boolean };
    reasoning_effort?: { supported?: boolean; values?: string[] };
  };
  last_request?: RuntimeRequestMetrics | null;
  [key: string]: unknown;
}

export interface RuntimeModel {
  id: string;
  object?: string;
  owned_by?: string;
}

export interface ModelArtifact {
  id: string;
  name: string;
  architecture: string;
  format: "mfq";
  shard_count: number;
  total_bytes: number;
  tensor_count: number;
  record_count: number;
  dtypes: string[];
  complete: boolean;
  loadable: boolean;
  modified_at: string;
  error?: string | null;
}

export interface RuntimeInstance {
  id: string;
  model: string;
  state: "loading" | "ready" | "busy" | "unloading" | "failed";
  devices: string[];
  active_sessions: number;
  queued_requests: number;
  resident_bytes?: number | null;
  kv_bytes?: number | null;
  context_size?: number | null;
  started_at?: string | null;
  last_used_at?: string | null;
  error?: ApiErrorBody["error"] | null;
}

export interface RuntimeProfile {
  id: string;
  name: string;
  load: {
    model: string;
    artifact_uri?: string | null;
    device_ids: string[];
    idle_ttl_seconds?: number | null;
    pin: boolean;
    context_size: number;
    prefill_chunk_size: number;
    moe_gpu_cache_gb?: number | null;
    prefix_cache_max_sessions?: number | null;
    prefix_cache_max_snapshots_per_session?: number | null;
    prefix_cache_max_bytes?: number | null;
    sampling_defaults?: SamplingParams | null;
  };
  artifact_id: string;
  artifact_modified_at: string;
  drifted: boolean;
  drift_reason?: string | null;
  created_at: string;
  updated_at: string;
}

export interface HubModelSummary {
  provider: "huggingface" | "modelscope";
  repo_id: string;
  downloads: number;
  likes: number;
  total_bytes: number;
  updated_at?: string | null;
}

export interface HubModelInfo extends HubModelSummary {
  revision: string;
  files: Array<{ name: string; byte_size: number }>;
  tags: string[];
}

export interface ArtifactLineage {
  id: string;
  artifact_uri: string;
  artifact_name: string;
  producer_job_id: string;
  producer_kind: string;
  source_uris: string[];
  parameters: Record<string, unknown>;
  metadata: Record<string, unknown>;
  validation_job_ids: string[];
  created_at: string;
}

export interface DatasetResource {
  id: string;
  name: string;
  kind: "wikitext2" | "custom";
  artifact_uri: string;
  sha256: string;
  byte_size: number;
  source_uri?: string | null;
  revision?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface EvaluationResult {
  id: string;
  job_id: string;
  kind: "perplexity" | "kernel_benchmark";
  model_id: string;
  metrics: Record<string, unknown>;
  parameters: Record<string, unknown>;
  dataset_id?: string | null;
  dataset_manifest: Record<string, unknown>;
  hardware_identity: Record<string, unknown>;
  runtime_identity: Record<string, unknown>;
  comparison_key: string;
  created_at: string;
}

export interface EvaluationComparison {
  comparison_key: string;
  baseline_id: string;
  metrics: string[];
  rows: Array<{
    evaluation: EvaluationResult;
    deltas: Record<string, number | null>;
    ratios: Record<string, number | null>;
  }>;
}

export interface RemoteNode {
  id: string;
  name: string;
  url: string;
  api_key_env?: string | null;
  enabled: boolean;
  healthy: boolean;
  models: string[];
  active_requests: number;
  metrics: Record<string, unknown>;
  last_checked_at?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface RuntimeMetricSnapshot {
  sequence: number;
  instance_id?: string | null;
  model?: string | null;
  values: RuntimeStatus;
  captured_at: string;
}

export interface RuntimeLogEntry {
  sequence: number;
  instance_id?: string | null;
  level: "debug" | "info" | "warning" | "error";
  message: string;
  fields: Record<string, unknown>;
  created_at: string;
}

export interface JobResource {
  id: string;
  kind: string;
  status: "queued" | "running" | "cancelling" | "succeeded" | "failed" | "cancelled" | "interrupted";
  payload: Record<string, unknown>;
  progress: number;
  cancel_requested: boolean;
  result?: Record<string, unknown> | null;
  error?: ApiErrorBody["error"] | null;
  created_at: string;
  updated_at: string;
}

export interface JsonSchemaProperty {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  anyOf?: JsonSchemaProperty[];
  items?: JsonSchemaProperty;
}

export interface JobKindResource {
  kind: string;
  payload_schema: {
    type?: string;
    properties?: Record<string, JsonSchemaProperty>;
    required?: string[];
    [key: string]: unknown;
  };
}

export interface McpServerResource {
  id: string;
  name: string;
  transport: "stdio" | "streamable_http";
  enabled: boolean;
  url?: string | null;
  command?: string | null;
  args: string[];
  header_env: Record<string, string>;
  timeout_seconds: number;
  created_at: string;
  updated_at: string;
}

export interface McpToolResource {
  server_id: string;
  server: string;
  name: string;
  qualified_name: string;
  description?: string | null;
  input_schema: Record<string, unknown>;
}

export interface McpToolCallResult {
  server: string;
  name: string;
  content: Array<Record<string, unknown>>;
  structured_content?: Record<string, unknown> | null;
  is_error: boolean;
}

export interface RealtimeCapabilities {
  available: boolean;
  input?: string[];
  output?: string[];
  input_sample_rate?: number;
  output_sample_rate?: number;
  defaults?: Record<string, unknown>;
  model_capabilities?: ModelCapabilities;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    retryable: boolean;
    details: Record<string, unknown>;
  };
}

export interface RealtimeFrame {
  protocol_version: "1.0";
  session_id: string;
  sequence: number;
  timestamp: string;
  payload: Record<string, unknown> & { type: string };
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;

  constructor(status: number, body: ApiErrorBody) {
    super(body.error.message);
    this.name = "ApiError";
    this.status = status;
    this.code = body.error.code;
    this.retryable = body.error.retryable;
  }
}

let apiBaseUrl = "";
let apiToken = "";

export function setApiBaseUrl(value: string): void {
  apiBaseUrl = value.trim().replace(/\/+$/, "");
}

export function setApiToken(value: string): void {
  apiToken = value.trim();
}

export function getApiBaseUrl(): string {
  return apiBaseUrl;
}

function apiUrl(path: string): string {
  return `${apiBaseUrl}${path}`;
}

export function runtimeRealtimeUrl(): string {
  const base = apiBaseUrl || window.location.origin;
  const url = new URL("/api/v1/runtime/realtime?mode=audio", base);
  if (apiToken) url.searchParams.set("access_token", apiToken);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  try {
    return new ApiError(response.status, (await response.json()) as ApiErrorBody);
  } catch {
    return new ApiError(response.status, {
      error: {
        code: `http_${response.status}`,
        message: response.statusText || "Request failed",
        retryable: false,
        details: {},
      },
    });
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) throw await errorFromResponse(response);
  return (await response.json()) as T;
}

function authorizedHeaders(headers?: HeadersInit): Headers {
  const result = new Headers(headers);
  if (apiToken) result.set("Authorization", `Bearer ${apiToken}`);
  return result;
}

export const api = {
  async listSessions(): Promise<Session[]> {
    return (await request<{ data: Session[] }>("/api/v1/sessions?limit=200")).data;
  },

  async generationPresets(): Promise<GenerationPresetResource[]> {
    return (
      await request<{ data: GenerationPresetResource[] }>("/api/v1/presets")
    ).data;
  },

  createGenerationPreset(
    body: Omit<GenerationPresetResource, "id" | "created_at" | "updated_at">,
  ): Promise<GenerationPresetResource> {
    return request("/api/v1/presets", { method: "POST", body: JSON.stringify(body) });
  },

  updateGenerationPreset(
    id: string,
    body: Omit<GenerationPresetResource, "id" | "created_at" | "updated_at">,
  ): Promise<GenerationPresetResource> {
    return request(`/api/v1/presets/${id}`, {
      method: "PUT",
      body: JSON.stringify(body),
    });
  },

  async deleteGenerationPreset(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/presets/${id}`), {
      method: "DELETE",
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  createSession(
    model: string,
    mode: SessionMode,
    title?: string,
    metadata?: Record<string, unknown>,
  ): Promise<Session> {
    return request("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ model, mode, title, metadata: metadata ?? {} }),
    });
  },

  getSession(id: string): Promise<Session> {
    return request(`/api/v1/sessions/${id}`);
  },

  updateSession(
    id: string,
    update: { title?: string | null; mode?: SessionMode; metadata?: Record<string, unknown> },
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}`, {
      method: "PATCH",
      body: JSON.stringify(update),
    });
  },

  async listMessages(id: string): Promise<Message[]> {
    return (await request<{ data: Message[] }>(`/api/v1/sessions/${id}/messages`)).data;
  },

  async listResponses(id: string): Promise<ResponseResource[]> {
    return (
      await request<{ data: ResponseResource[] }>(
        `/api/v1/sessions/${id}/responses?limit=1000`,
      )
    ).data;
  },

  appendMessage(
    id: string,
    expectedRevision: number,
    role: MessageRole,
    parts: ContentPart[],
  ): Promise<{ session: Session; message: Message }> {
    return request(`/api/v1/sessions/${id}/messages`, {
      method: "POST",
      body: JSON.stringify({ expected_revision: expectedRevision, role, parts }),
    });
  },

  forkSession(
    id: string,
    atMessageId: string | null,
    includeMessage = true,
    title?: string | null,
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}/fork`, {
      method: "POST",
      body: JSON.stringify({
        at_message_id: atMessageId,
        include_message: includeMessage,
        title,
      }),
    });
  },

  rewindSession(
    id: string,
    expectedRevision: number,
    atMessageId: string,
    includeMessage = true,
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}/rewind`, {
      method: "POST",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        at_message_id: atMessageId,
        include_message: includeMessage,
      }),
    });
  },

  async deleteSession(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/sessions/${id}`), {
      method: "DELETE",
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  exportSession(id: string): Promise<SessionArchive> {
    return request(`/api/v1/sessions/${id}/export`);
  },

  importSession(archive: SessionArchive): Promise<{
    session: Session;
    messages_imported: number;
    media_imported: number;
  }> {
    return request("/api/v1/sessions/import", {
      method: "POST",
      body: JSON.stringify(archive),
    });
  },

  async uploadMedia(file: File, mimeType?: string): Promise<MediaResource> {
    const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
    const sha256 = Array.from(new Uint8Array(digest), (value) =>
      value.toString(16).padStart(2, "0"),
    ).join("");
    const response = await fetch(apiUrl("/api/v1/media"), {
      method: "POST",
      headers: {
        "Content-Type": mimeType || file.type || "application/octet-stream",
        "X-Content-SHA256": sha256,
        ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
      },
      body: file,
    });
    if (!response.ok) throw await errorFromResponse(response);
    return (await response.json()) as MediaResource;
  },

  mediaUrl(id: string): string {
    return apiUrl(`/api/v1/media/${id}`);
  },

  createDocument(mediaId: string, name: string): Promise<DocumentResource> {
    return request("/api/v1/documents", {
      method: "POST",
      body: JSON.stringify({ media_id: mediaId, name }),
    });
  },

  runtimeCapabilities(): Promise<RuntimeCapabilities> {
    return request("/api/v1/runtime/capabilities");
  },

  runtimeStatus(): Promise<RuntimeStatus> {
    return request("/api/v1/runtime/status");
  },

  async runtimeModels(): Promise<RuntimeModel[]> {
    return (await request<{ data: RuntimeModel[] }>("/api/v1/runtime/models")).data;
  },

  async modelArtifacts(refresh = false): Promise<ModelArtifact[]> {
    return (
      await request<{ data: ModelArtifact[] }>(
        `/api/v1/models${refresh ? "?refresh=true" : ""}`,
      )
    ).data;
  },

  async runtimeInstances(): Promise<RuntimeInstance[]> {
    return (await request<{ data: RuntimeInstance[] }>("/api/v1/runtime/instances")).data;
  },

  async runtimeProfiles(): Promise<RuntimeProfile[]> {
    return (await request<{ data: RuntimeProfile[] }>("/api/v1/runtime/profiles")).data;
  },

  createRuntimeProfile(body: {
    name: string;
    load: RuntimeProfile["load"];
  }): Promise<RuntimeProfile> {
    return request("/api/v1/runtime/profiles", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  async deleteRuntimeProfile(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/runtime/profiles/${id}`), {
      method: "DELETE",
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  loadRuntimeProfile(
    id: string,
    allowDrift = false,
  ): Promise<{ operation_id: string; status: "accepted" }> {
    return request(`/api/v1/runtime/profiles/${id}/load`, {
      method: "POST",
      body: JSON.stringify({ allow_drift: allowDrift }),
    });
  },

  async searchHubModels(
    provider: HubModelSummary["provider"],
    query: string,
  ): Promise<HubModelSummary[]> {
    const params = new URLSearchParams({ provider, query, limit: "20" });
    return (await request<{ data: HubModelSummary[] }>(`/api/v1/hub/models?${params}`)).data;
  },

  hubModelInfo(
    provider: HubModelSummary["provider"],
    repoId: string,
    revision?: string,
  ): Promise<HubModelInfo> {
    const [owner, name] = repoId.split("/", 2);
    const suffix = revision ? `?revision=${encodeURIComponent(revision)}` : "";
    return request(`/api/v1/hub/models/${provider}/${encodeURIComponent(owner)}/${encodeURIComponent(name)}${suffix}`);
  },

  retryJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}/retry`, { method: "POST" });
  },

  removeWorkspaceArtifact(uri: string): Promise<Record<string, unknown>> {
    return request("/api/v1/artifacts/remove", {
      method: "POST",
      body: JSON.stringify({ artifact_uri: uri }),
    });
  },

  async artifactLineage(limit = 200): Promise<ArtifactLineage[]> {
    return (
      await request<{ data: ArtifactLineage[] }>(`/api/v1/artifacts/lineage?limit=${limit}`)
    ).data;
  },

  async datasets(): Promise<DatasetResource[]> {
    return (await request<{ data: DatasetResource[] }>("/api/v1/datasets")).data;
  },

  createDataset(body: {
    name: string;
    kind: DatasetResource["kind"];
    artifact_uri: string;
    source_uri?: string | null;
    revision?: string | null;
    metadata?: Record<string, unknown>;
  }): Promise<DatasetResource> {
    return request("/api/v1/datasets", { method: "POST", body: JSON.stringify(body) });
  },

  async deleteDataset(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/datasets/${id}`), {
      method: "DELETE",
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  async evaluations(limit = 200): Promise<EvaluationResult[]> {
    return (
      await request<{ data: EvaluationResult[] }>(`/api/v1/evaluations?limit=${limit}`)
    ).data;
  },

  compareEvaluations(evaluationIds: string[]): Promise<EvaluationComparison> {
    return request("/api/v1/evaluations/compare", {
      method: "POST",
      body: JSON.stringify({ evaluation_ids: evaluationIds }),
    });
  },

  async remoteNodes(refresh = false): Promise<RemoteNode[]> {
    return (
      await request<{ data: RemoteNode[] }>(
        `/api/v1/cluster/nodes${refresh ? "?refresh=true" : ""}`,
      )
    ).data;
  },

  createRemoteNode(body: {
    name: string;
    url: string;
    api_key_env?: string | null;
    enabled: boolean;
  }): Promise<RemoteNode> {
    return request("/api/v1/cluster/nodes", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  async deleteRemoteNode(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/cluster/nodes/${id}`), {
      method: "DELETE",
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  loadModel(
    model: string,
    contextSize: number,
    prefillChunkSize = 2048,
  ): Promise<{ operation_id: string; status: "accepted" }> {
    return request("/api/v1/models/load", {
      method: "POST",
      body: JSON.stringify({
        model,
        context_size: contextSize,
        prefill_chunk_size: prefillChunkSize,
      }),
    });
  },

  unloadModel(
    instanceId: string,
    force = false,
  ): Promise<{ operation_id: string; status: "accepted" }> {
    return request("/api/v1/models/unload", {
      method: "POST",
      body: JSON.stringify({ instance_id: instanceId, force }),
    });
  },

  async runtimeMetrics(limit = 200): Promise<RuntimeMetricSnapshot[]> {
    return (
      await request<{ data: RuntimeMetricSnapshot[] }>(
        `/api/v1/runtime/metrics?limit=${limit}`,
      )
    ).data;
  },

  async runtimeLogs(limit = 100): Promise<RuntimeLogEntry[]> {
    return (
      await request<{ data: RuntimeLogEntry[] }>(`/api/v1/runtime/logs?limit=${limit}`)
    ).data;
  },

  async jobs(limit = 100): Promise<JobResource[]> {
    return (await request<{ data: JobResource[] }>(`/api/v1/jobs?limit=${limit}`)).data;
  },

  async jobKinds(): Promise<JobKindResource[]> {
    return (await request<{ data: JobKindResource[] }>("/api/v1/jobs/kinds")).data;
  },

  async mcpServers(): Promise<McpServerResource[]> {
    return (await request<{ data: McpServerResource[] }>("/api/v1/mcp/servers")).data;
  },

  async mcpTools(): Promise<{ data: McpToolResource[]; errors: Record<string, string> }> {
    return request("/api/v1/mcp/tools");
  },

  createMcpServer(body: {
    name: string;
    transport: "stdio" | "streamable_http";
    enabled: boolean;
    url?: string | null;
    command?: string | null;
    args?: string[];
    header_env?: Record<string, string>;
  }): Promise<McpServerResource> {
    return request("/api/v1/mcp/servers", { method: "POST", body: JSON.stringify(body) });
  },

  updateMcpServer(id: string, enabled: boolean): Promise<McpServerResource> {
    return request(`/api/v1/mcp/servers/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ enabled }),
    });
  },

  async deleteMcpServer(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/mcp/servers/${id}`), {
      method: "DELETE",
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  callMcpTool(name: string, arguments_: Record<string, unknown>): Promise<McpToolCallResult> {
    return request("/api/v1/mcp/tools/call", {
      method: "POST",
      body: JSON.stringify({ name, arguments: arguments_, confirm: true }),
    });
  },

  createJob(kind: string, payload: Record<string, unknown>): Promise<JobResource> {
    return request("/api/v1/jobs", {
      method: "POST",
      body: JSON.stringify({ kind, payload }),
    });
  },

  cancelJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}/cancel`, { method: "POST" });
  },

  async jobEvents(id: string): Promise<RuntimeLogEntry[]> {
    const response = await request<{
      data: Array<{
        sequence: number;
        level: RuntimeLogEntry["level"];
        message?: string | null;
        data: Record<string, unknown>;
        created_at: string;
      }>;
    }>(`/api/v1/jobs/${id}/events?limit=1000`);
    return response.data
      .filter((event) => event.message)
      .map((event) => ({
        sequence: event.sequence,
        level: event.level,
        message: event.message || "",
        fields: event.data,
        created_at: event.created_at,
      }));
  },

  realtimeCapabilities(): Promise<RealtimeCapabilities> {
    return request("/api/v1/runtime/realtime/capabilities");
  },

  reloadRuntime(contextSize: number): Promise<RuntimeStatus> {
    return request("/api/v1/runtime/reload", {
      method: "POST",
      body: JSON.stringify({ context_size: contextSize }),
    });
  },

  clearRuntimeCache(): Promise<RuntimeStatus & { released_snapshots: number }> {
    return request("/api/v1/runtime/cache/clear", { method: "POST" });
  },
};

export interface StreamRequest {
  request_id: string;
  expected_revision: number;
  input: ContentPart[];
  input_role?: "user" | "tool";
  sampling: SamplingParams;
  system_prompt?: string | null;
  include_reasoning_history: boolean;
  tools?: Array<{
    type: "function";
    function: {
      name: string;
      description?: string | null;
      parameters: Record<string, unknown>;
    };
  }>;
  tool_choice?: "auto" | "none" | "required";
  stream: true;
}

export async function streamResponse(
  sessionId: string,
  body: StreamRequest,
  onFrame: (frame: RealtimeFrame) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(apiUrl(`/api/v1/sessions/${sessionId}/responses`), {
    method: "POST",
    headers: authorizedHeaders({
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    }),
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await errorFromResponse(response);
  if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error("MFQ Server returned an invalid streaming response");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let streamError: ApiError | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    buffer = (buffer + decoder.decode(value, { stream: !done })).replaceAll("\r\n", "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) {
        const frame = JSON.parse(data) as RealtimeFrame;
        onFrame(frame);
        if (frame.payload.type === "error") {
          streamError = new ApiError(502, { error: frame.payload.error as ApiErrorBody["error"] });
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
  if (buffer.trim()) throw new Error("MFQ Server stream ended with an incomplete event");
  if (streamError) throw streamError;
}
