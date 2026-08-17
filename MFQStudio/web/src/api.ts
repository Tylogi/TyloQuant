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

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "transcript"; text: string; language?: string | null }
  | { type: "image"; media: MediaRef; width?: number | null; height?: number | null }
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

export interface Message {
  id: string;
  role: MessageRole;
  parts: ContentPart[];
  parent_id: string | null;
  created_at: string;
}

export interface ResponseResource {
  id: string;
  request_id: string;
  session_id: string;
  status: "queued" | "running" | "completed" | "cancelled" | "failed";
  output_message_id?: string | null;
  output: ContentPart[];
  finish_reason?: string | null;
  created_at: string;
  completed_at?: string | null;
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
  prompt_tokens?: number;
  completion_tokens?: number;
  prefill_tps?: number;
  prefill_ms?: number;
  decode_tps?: number;
  ttft_ms?: number;
  generation_ms?: number;
  finish_reason?: string;
}

export interface RuntimeStatus {
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

export function setApiBaseUrl(value: string): void {
  apiBaseUrl = value.trim().replace(/\/+$/, "");
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
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) throw await errorFromResponse(response);
  return (await response.json()) as T;
}

export const api = {
  async listSessions(): Promise<Session[]> {
    return (await request<{ data: Session[] }>("/api/v1/sessions?limit=200")).data;
  },

  createSession(model: string, mode: SessionMode, title?: string): Promise<Session> {
    return request("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ model, mode, title }),
    });
  },

  getSession(id: string): Promise<Session> {
    return request(`/api/v1/sessions/${id}`);
  },

  updateSession(
    id: string,
    update: { title?: string | null; mode?: SessionMode },
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
    const response = await fetch(apiUrl(`/api/v1/sessions/${id}`), { method: "DELETE" });
    if (!response.ok) throw await errorFromResponse(response);
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

  realtimeCapabilities(): Promise<RealtimeCapabilities> {
    return request("/api/v1/runtime/realtime/capabilities");
  },

  reloadRuntime(contextSize: number): Promise<RuntimeStatus> {
    return request("/api/v1/runtime/reload", {
      method: "POST",
      body: JSON.stringify({ context_size: contextSize }),
    });
  },
};

export interface StreamRequest {
  request_id: string;
  expected_revision: number;
  input: ContentPart[];
  sampling: SamplingParams;
  system_prompt?: string | null;
  include_reasoning_history: boolean;
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
    headers: { Accept: "text/event-stream", "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await errorFromResponse(response);
  if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error("MFQd returned an invalid streaming response");
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
  if (buffer.trim()) throw new Error("MFQd stream ended with an incomplete event");
  if (streamError) throw streamError;
}
