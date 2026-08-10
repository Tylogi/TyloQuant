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

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "transcript"; text: string; language?: string | null }
  | { type: "tool_call"; call_id: string; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; call_id: string; result: unknown; is_error: boolean };

export interface Message {
  id: string;
  role: "system" | "user" | "assistant" | "tool";
  parts: ContentPart[];
  parent_id: string | null;
  created_at: string;
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  return (await response.json()) as T;
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

export const api = {
  async listSessions(): Promise<Session[]> {
    return (await request<{ data: Session[] }>("/api/v1/sessions?limit=200")).data;
  },

  createSession(model: string, mode: SessionMode): Promise<Session> {
    return request("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ model, mode }),
    });
  },

  getSession(id: string): Promise<Session> {
    return request(`/api/v1/sessions/${id}`);
  },

  async listMessages(id: string): Promise<Message[]> {
    return (await request<{ data: Message[] }>(`/api/v1/sessions/${id}/messages`)).data;
  },

  async deleteSession(id: string): Promise<void> {
    const response = await fetch(`/api/v1/sessions/${id}`, { method: "DELETE" });
    if (!response.ok) throw await errorFromResponse(response);
  },

  runtimeCapabilities(): Promise<RuntimeCapabilities> {
    return request("/api/v1/runtime/capabilities");
  },
};

export interface StreamRequest {
  request_id: string;
  expected_revision: number;
  input: Array<{ type: "text"; text: string }>;
  stream: true;
}

export async function streamResponse(
  sessionId: string,
  body: StreamRequest,
  onFrame: (frame: RealtimeFrame) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/v1/sessions/${sessionId}/responses`, {
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
