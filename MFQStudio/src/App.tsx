import { Children, FormEvent, isValidElement, lazy, ReactNode, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  ArtifactLineage,
  ContentPart,
  DatasetResource,
  EvaluationComparison,
  EvaluationResult,
  GenerationPresetResource,
  HubModelInfo,
  HubModelSummary,
  JobResource,
  JobKindResource,
  JsonSchemaProperty,
  McpServerResource,
  McpToolResource,
  Message,
  ModelArtifact,
  ModelDirectoryList,
  RealtimeCapabilities,
  RealtimeFrame,
  ResponseResource,
  RuntimeCapabilities,
  RuntimeInstance,
  RuntimeLogEntry,
  RuntimeModel,
  RuntimeProfile,
  RemoteNode,
  RuntimeRequestMetrics,
  RuntimeStatus,
  SamplingParams,
  SessionArchive,
  Session,
  SessionMode,
  api,
  setApiBaseUrl,
  setApiToken,
  streamResponse,
} from "./api";
import {
  RealtimeAudioController,
  VoiceState,
  loadVoiceClip,
  saveVoiceClip,
} from "./realtimeAudio";
import {
  StudioConfig,
  StudioStatus,
  configureStudio,
  isStudio,
  saveStudioCredential,
  selectLocalModelDirectory,
  startLocalStudio,
  studioConfirm,
  studioCredential,
  studioStatus,
} from "./studio";

const Markdown = lazy(() => import("./Markdown").then((module) => ({ default: module.Markdown })));

interface LiveOutput {
  reasoning: string;
  text: string;
  tools: string[];
}


function schemaDefault(property: JsonSchemaProperty): unknown {
  if (property.default !== undefined) return property.default;
  const option = property.anyOf?.find((item) => item.type && item.type !== "null");
  if (option) return schemaDefault(option);
  if (property.type === "boolean") return false;
  if (property.type === "array") return [];
  return "";
}

function schemaType(property: JsonSchemaProperty): string | undefined {
  if (typeof property.type === "string") return property.type;
  return property.anyOf?.find((item) => item.type && item.type !== "null")?.type as
    | string
    | undefined;
}

type ViewName = "chat" | "dashboard" | "lab";
type DashboardPage = "overview" | "cache" | "models" | "connections";
type LabPage = "models" | "evaluations" | "quantization";
type UiLanguage = "system" | "zh-CN" | "en";
type UiTheme = "system" | "light" | "dark";
type PresetName = "precise" | "balanced" | "creative" | "custom";

interface GenerationSettings {
  language: UiLanguage;
  theme: UiTheme;
  inheritModelDefaults: boolean;
  systemPrompt: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  frequencyPenalty: number;
  enableThinking: boolean;
  reasoningEffort: string;
  excludeReasoning: boolean;
  playbackEnabled: boolean;
  fullDuplex: boolean;
  preset: PresetName;
  seed: number | null;
}

type StoredPresetSettings = Pick<
  GenerationSettings,
  | "systemPrompt"
  | "maxTokens"
  | "temperature"
  | "topP"
  | "topK"
  | "repetitionPenalty"
  | "presencePenalty"
  | "frequencyPenalty"
  | "enableThinking"
  | "reasoningEffort"
  | "excludeReasoning"
  | "seed"
>;

interface StoredPreset {
  id?: string;
  name: string;
  settings: StoredPresetSettings;
  inheritGlobalSettings: boolean;
  contextSize: number;
  model?: string | null;
  mode?: SessionMode | null;
  icon?: string;
  updatedAt: string;
}

interface AssistantRole {
  id: string;
  name: string;
  preset?: StoredPreset;
}

interface RoleEditorDraft {
  roleId: string;
  name: string;
  icon: string;
  model: string;
  mode: SessionMode;
  contextSize: number;
  inheritGlobalSettings: boolean;
  settings: StoredPresetSettings;
}

interface VoiceMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  text: string;
  audioId?: string;
  pending?: boolean;
  created_at: string;
}

interface LiveVoiceOutput {
  sessionId: string;
  text: string;
}

interface EditDraft {
  messageId: string;
  text: string;
  reasoning: string;
}

interface PendingAttachment {
  id: string;
  file: File;
  previewUrl: string;
  kind: "image" | "video" | "audio" | "document";
}

const SETTINGS_KEY = "mfq.studio.generation.v1";
const STORED_PRESETS_KEY = "mfq.studio.presets.v1";
const VOICE_HISTORY_KEY = "mfq.studio.voice-history.v1";
const DEFAULT_ASSISTANT_ID = "assistant:default";

function assistantIdForPreset(preset: StoredPreset): string {
  return preset.id
    ? `assistant:preset:${preset.id}`
    : `assistant:preset:${preset.name.trim().toLocaleLowerCase()}`;
}

function legacyAssistantIdForPreset(preset: StoredPreset): string {
  return `assistant:preset:${preset.name.trim().toLocaleLowerCase()}`;
}

function sessionAssistantId(session: Session): string {
  const identifier = session.metadata?.assistant_id;
  return typeof identifier === "string" && identifier ? identifier : DEFAULT_ASSISTANT_ID;
}

function sessionAssistantName(session: Session): string | null {
  const name = session.metadata?.assistant_name;
  return typeof name === "string" && name.trim() ? name.trim() : null;
}

function canonicalSessionAssistantId(session: Session, presets: StoredPreset[]): string {
  const identifier = sessionAssistantId(session);
  if (identifier === DEFAULT_ASSISTANT_ID) return identifier;
  const name = sessionAssistantName(session);
  const preset = presets.find((candidate) =>
    identifier === assistantIdForPreset(candidate)
      || identifier === legacyAssistantIdForPreset(candidate)
      || name === candidate.name
  );
  return preset ? assistantIdForPreset(preset) : identifier;
}
const DOCUMENT_ACCEPT = [
  ".txt",
  ".md",
  ".markdown",
  ".json",
  ".jsonl",
  ".csv",
  ".tsv",
  ".yaml",
  ".yml",
  ".xml",
  ".html",
  ".css",
  ".py",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".c",
  ".cc",
  ".cpp",
  ".h",
  ".hpp",
  ".rs",
  ".go",
  ".java",
  ".sh",
  ".toml",
  ".ini",
  ".log",
  ".pdf",
  ".docx",
].join(",");
const MAX_DOCUMENT_BYTES = 64 * 1024 * 1024;
const MODE_LABELS: Record<SessionMode, [string, string]> = {
  text: ["文本", "Text"],
  voice: ["语音", "Voice"],
  full_duplex: ["全双工", "Full duplex"],
};
const DEFAULT_SETTINGS: GenerationSettings = {
  language: "system",
  theme: "system",
  inheritModelDefaults: true,
  systemPrompt: "",
  maxTokens: 4096,
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  repetitionPenalty: 1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  enableThinking: true,
  reasoningEffort: "",
  excludeReasoning: false,
  playbackEnabled: true,
  fullDuplex: true,
  preset: "balanced",
  seed: null,
};
const PRESETS: Record<Exclude<PresetName, "custom">, Partial<GenerationSettings>> = {
  precise: { temperature: 0.2, topP: 0.75, topK: 20, repetitionPenalty: 1.05 },
  balanced: { temperature: 0.7, topP: 0.8, topK: 20, repetitionPenalty: 1 },
  creative: { temperature: 1, topP: 0.95, topK: 50, repetitionPenalty: 1 },
};

function modeTemplateSettings(
  current: GenerationSettings,
  mode: SessionMode,
  runtime: RuntimeStatus | null,
  realtime: RealtimeCapabilities | null,
): GenerationSettings {
  const voice = mode !== "text";
  const defaults = (voice
    ? (runtime?.duplex_sampling_defaults ?? realtime?.defaults ?? {})
    : (runtime?.sampling_defaults ?? {})) as Record<string, unknown>;
  const value = (key: string, fallback: number): number => {
    const candidate = Number(defaults[key]);
    return Number.isFinite(candidate) ? candidate : fallback;
  };
  return {
    ...current,
    systemPrompt: voice ? String(defaults.system_prompt ?? "") : "",
    maxTokens: voice ? current.maxTokens : value("max_tokens", current.maxTokens),
    temperature: value("temperature", current.temperature),
    topP: value("top_p", current.topP),
    topK: value("top_k", current.topK),
    repetitionPenalty: value(
      voice ? "text_repetition_penalty" : "repetition_penalty",
      current.repetitionPenalty,
    ),
    presencePenalty: voice ? 0 : value("presence_penalty", current.presencePenalty),
    frequencyPenalty: voice ? 0 : value("frequency_penalty", current.frequencyPenalty),
    enableThinking: voice
      ? false
      : typeof defaults.enable_thinking === "boolean"
        ? defaults.enable_thinking
        : current.enableThinking,
    fullDuplex: mode === "full_duplex",
    preset: "custom",
    seed: null,
  };
}

function roleGenerationSettings(
  globalSettings: GenerationSettings,
  role: StoredPreset | undefined,
): GenerationSettings {
  if (!role) return globalSettings;
  if (role.inheritGlobalSettings) {
    return {
      ...globalSettings,
      systemPrompt: role.settings.systemPrompt.trim()
        ? role.settings.systemPrompt
        : globalSettings.systemPrompt,
    };
  }
  return {
    ...globalSettings,
    ...role.settings,
    preset: "custom",
  };
}
const CAPABILITY_LABELS: Array<[
  keyof RuntimeCapabilities["model_capabilities"]["features"],
  [string, string],
]> = [
  ["text", ["文本", "Text"]],
  ["image_input", ["图片", "Image"]],
  ["video_input", ["视频", "Video"]],
  ["audio_input", ["音频输入", "Audio in"]],
  ["audio_output", ["音频输出", "Audio out"]],
  ["full_duplex", ["全双工", "Full duplex"]],
];

function loadSettings(): GenerationSettings {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function presetSnapshot(settings: GenerationSettings): StoredPresetSettings {
  return {
    systemPrompt: settings.systemPrompt,
    maxTokens: settings.maxTokens,
    temperature: settings.temperature,
    topP: settings.topP,
    topK: settings.topK,
    repetitionPenalty: settings.repetitionPenalty,
    presencePenalty: settings.presencePenalty,
    frequencyPenalty: settings.frequencyPenalty,
    enableThinking: settings.enableThinking,
    reasoningEffort: settings.reasoningEffort,
    excludeReasoning: settings.excludeReasoning,
    seed: settings.seed,
  };
}

function loadStoredPresets(): StoredPreset[] {
  try {
    const decoded = JSON.parse(localStorage.getItem(STORED_PRESETS_KEY) || "[]");
    if (!Array.isArray(decoded)) return [];
    const fallback = presetSnapshot(DEFAULT_SETTINGS);
    return decoded.flatMap((candidate): StoredPreset[] => {
      if (!candidate || typeof candidate !== "object") return [];
      const raw = candidate as Record<string, unknown>;
      const name = typeof raw.name === "string" ? raw.name.replace(/\s+/g, " ").trim() : "";
      const source =
        raw.settings && typeof raw.settings === "object"
          ? (raw.settings as Record<string, unknown>)
          : {};
      if (!name) return [];
      const number = (key: keyof StoredPresetSettings, defaultValue: number) => {
        const value = Number(source[key]);
        return Number.isFinite(value) ? value : defaultValue;
      };
      const contextSize = Number(raw.contextSize);
      return [{
        id: typeof raw.id === "string" ? raw.id : undefined,
        name: name.slice(0, 64),
        settings: {
          systemPrompt:
            typeof source.systemPrompt === "string" ? source.systemPrompt : fallback.systemPrompt,
          maxTokens: number("maxTokens", fallback.maxTokens),
          temperature: number("temperature", fallback.temperature),
          topP: number("topP", fallback.topP),
          topK: number("topK", fallback.topK),
          repetitionPenalty: number("repetitionPenalty", fallback.repetitionPenalty),
          presencePenalty: number("presencePenalty", fallback.presencePenalty),
          frequencyPenalty: number("frequencyPenalty", fallback.frequencyPenalty),
          enableThinking:
            typeof source.enableThinking === "boolean"
              ? source.enableThinking
              : fallback.enableThinking,
          reasoningEffort:
            typeof source.reasoningEffort === "string"
              ? source.reasoningEffort
              : fallback.reasoningEffort,
          excludeReasoning:
            typeof source.excludeReasoning === "boolean"
              ? source.excludeReasoning
              : fallback.excludeReasoning,
          seed:
            source.seed == null || source.seed === ""
              ? null
              : number("seed", fallback.seed ?? 0),
        },
        inheritGlobalSettings:
          typeof raw.inheritGlobalSettings === "boolean" ? raw.inheritGlobalSettings : true,
        contextSize:
          Number.isFinite(contextSize) && contextSize >= 512
            ? Math.floor(contextSize)
            : 32768,
        model: typeof raw.model === "string" ? raw.model : null,
        mode:
          raw.mode === "text" || raw.mode === "voice" || raw.mode === "full_duplex"
            ? raw.mode
            : null,
        icon: typeof raw.icon === "string" ? raw.icon.slice(0, 8) : undefined,
        updatedAt:
          typeof raw.updatedAt === "string" ? raw.updatedAt : new Date(0).toISOString(),
      }];
    }).slice(0, 50);
  } catch {
    return [];
  }
}

function storedPresetFromResource(preset: GenerationPresetResource): StoredPreset {
  const sampling = preset.settings.sampling;
  return {
    id: preset.id,
    name: preset.name,
    settings: {
      systemPrompt: preset.settings.system_prompt ?? "",
      maxTokens: sampling.max_tokens,
      temperature: sampling.temperature,
      topP: sampling.top_p,
      topK: sampling.top_k,
      repetitionPenalty: sampling.repetition_penalty,
      presencePenalty: sampling.presence_penalty,
      frequencyPenalty: sampling.frequency_penalty,
      enableThinking: sampling.enable_thinking,
      reasoningEffort: sampling.reasoning_effort ?? "",
      excludeReasoning: !preset.settings.include_reasoning_history,
      seed: sampling.seed ?? null,
    },
    inheritGlobalSettings:
      typeof preset.metadata?.inherit_global_settings === "boolean"
        ? preset.metadata.inherit_global_settings
        : true,
    contextSize: preset.context_size,
    model: preset.model,
    mode: preset.mode,
    icon: typeof preset.metadata?.icon === "string" ? preset.metadata.icon : undefined,
    updatedAt: preset.updated_at,
  };
}

function presetResourceBody(
  preset: StoredPreset,
  fallbackModel: string,
  fallbackMode: SessionMode,
): Omit<GenerationPresetResource, "id" | "created_at" | "updated_at"> {
  return {
    name: preset.name,
    model: preset.model ?? fallbackModel,
    mode: preset.mode ?? fallbackMode,
    settings: {
      sampling: {
        max_tokens: preset.settings.maxTokens,
        temperature: preset.settings.temperature,
        top_k: preset.settings.topK,
        top_p: preset.settings.topP,
        presence_penalty: preset.settings.presencePenalty,
        frequency_penalty: preset.settings.frequencyPenalty,
        repetition_penalty: preset.settings.repetitionPenalty,
        seed: preset.settings.seed,
        enable_thinking: preset.settings.enableThinking,
        reasoning_effort: preset.settings.reasoningEffort || null,
      },
      system_prompt: preset.settings.systemPrompt || null,
      include_reasoning_history: !preset.settings.excludeReasoning,
      input_role: "user",
      tools: [],
      tool_choice: "auto",
      response_format: { type: "text" },
    },
    context_size: preset.contextSize,
    metadata: {
      icon: preset.icon || preset.name.slice(0, 1).toLocaleUpperCase(),
      inherit_global_settings: preset.inheritGlobalSettings,
    },
  };
}

function loadVoiceHistory(): VoiceMessage[] {
  try {
    const value = JSON.parse(localStorage.getItem(VOICE_HISTORY_KEY) || "[]");
    return Array.isArray(value) ? value.slice(-200) : [];
  } catch {
    return [];
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return `${error.code}: ${error.message}`;
  if (error instanceof Error) return error.message;
  return "Unknown error";
}

function textParts(message: Message): { text: string; reasoning: string } {
  const text = message.parts
    .filter((part) => part.type === "text" || part.type === "transcript")
    .map((part) => part.text)
    .join("");
  const reasoning = message.parts
    .filter((part) => part.type === "reasoning")
    .map((part) => part.text)
    .join("");
  return { text, reasoning };
}

function isMediaPart(
  part: ContentPart,
): part is Extract<ContentPart, { type: "image" | "video" | "audio" | "generated_audio" }> {
  return (
    part.type === "image" ||
    part.type === "video" ||
    part.type === "audio" ||
    part.type === "generated_audio"
  );
}

function isTextDocument(file: File): boolean {
  const extension = file.name.toLowerCase().match(/\.[^.]+$/)?.[0] ?? "";
  return (
    file.type.startsWith("text/") ||
    ["application/json", "application/xml", "application/yaml"].includes(file.type) ||
    DOCUMENT_ACCEPT.split(",").includes(extension)
  );
}

async function mediaMetadata(
  file: File,
  kind: Exclude<PendingAttachment["kind"], "document">,
) {
  if (kind === "image") {
    const bitmap = await createImageBitmap(file);
    const result = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return result;
  }
  if (kind === "audio") {
    const context = new AudioContext();
    try {
      const buffer = await context.decodeAudioData(await file.arrayBuffer());
      return {
        sample_rate_hz: buffer.sampleRate,
        channels: buffer.numberOfChannels,
        duration_ms: Math.round(buffer.duration * 1000),
      };
    } finally {
      await context.close();
    }
  }
  const url = URL.createObjectURL(file);
  try {
    return await new Promise<{ width: number; height: number; duration_ms: number }>(
      (resolve, reject) => {
        const video = document.createElement("video");
        video.preload = "metadata";
        video.onloadedmetadata = () =>
          resolve({
            width: video.videoWidth,
            height: video.videoHeight,
            duration_ms: Math.round(video.duration * 1000),
          });
        video.onerror = () => reject(new Error("Unable to read video metadata"));
        video.src = url;
      },
    );
  } finally {
    URL.revokeObjectURL(url);
  }
}

function MediaPartView({ part }: { part: Extract<ContentPart, { media: unknown }> }) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | null = null;
    setSrc(null);
    void api.fetchMedia(part.media.id, controller.signal).then((blob) => {
      if (controller.signal.aborted) return;
      objectUrl = URL.createObjectURL(blob);
      setSrc(objectUrl);
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) console.error("Unable to load message media", error);
    });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [part.media.id]);

  if (!src) return <span className="message-media-loading" aria-label="Loading media" />;
  if (part.type === "image") {
    return <img alt="Attached image" className="message-media media-image" loading="lazy" src={src} />;
  }
  if (part.type === "video") {
    return <video className="message-media media-video" controls preload="metadata" src={src} />;
  }
  return <audio className="message-audio" controls preload="metadata" src={src} />;
}

function DocumentPartView({ part }: { part: Extract<ContentPart, { type: "document" }> }) {
  return <a className="message-document" download={part.name} href={api.mediaUrl(part.media.id)}><span>DOC</span><div><strong>{part.name}</strong><small>{formatNumber(part.media.byte_size)} B</small></div></a>;
}

function documentMimeType(file: File): string {
  if (file.type) return file.type;
  const extension = file.name.toLowerCase().match(/\.[^.]+$/)?.[0] ?? "";
  if (extension === ".pdf") return "application/pdf";
  if (extension === ".docx") {
    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
  }
  if (extension === ".json" || extension === ".jsonl") return "application/json";
  if (extension === ".xml") return "application/xml";
  return "text/plain";
}

function formatNumber(value: unknown, digits = 0): string {
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(number)
    : "--";
}

interface PrefillMetricLike {
  prompt_tokens?: number;
  prefill_tokens?: number;
  ttft_ms?: number;
  prefill_ms?: number;
  prefill_tps?: number;
  model_prefill_ms?: number;
  complete_prefill_ms?: number;
  complete_prefill_tps?: number;
}

function displayPrefillMetric(metrics?: PrefillMetricLike | null): {
  milliseconds: number | undefined;
  tokensPerSecond: number | undefined;
} {
  if (!metrics) return { milliseconds: undefined, tokensPerSecond: undefined };
  const nativeMilliseconds = Number(metrics.ttft_ms);
  if (Number.isFinite(nativeMilliseconds) && nativeMilliseconds > 0) {
    const tokens = Number(metrics.prefill_tokens ?? metrics.prompt_tokens);
    return {
      milliseconds: nativeMilliseconds,
      tokensPerSecond:
        Number.isFinite(tokens) && tokens > 0
          ? (tokens * 1000) / nativeMilliseconds
          : undefined,
    };
  }
  const modelMilliseconds = Number(metrics.model_prefill_ms);
  const languageMilliseconds = Number(metrics.prefill_ms);
  const milliseconds =
    Number.isFinite(modelMilliseconds) && modelMilliseconds > 0
      ? modelMilliseconds
      : Number.isFinite(languageMilliseconds) && languageMilliseconds > 0
        ? languageMilliseconds
        : undefined;
  const tokens = Number(metrics.prefill_tokens ?? metrics.prompt_tokens);
  if (milliseconds !== undefined && Number.isFinite(tokens) && tokens > 0) {
    return { milliseconds, tokensPerSecond: (tokens * 1000) / milliseconds };
  }
  const reported = Number(metrics.prefill_tps);
  return {
    milliseconds,
    tokensPerSecond: Number.isFinite(reported) ? reported : undefined,
  };
}

const TERMINAL_JOB_STATUSES = new Set<JobResource["status"]>([
  "succeeded",
  "failed",
  "cancelled",
  "interrupted",
]);

function isTerminalJob(job: JobResource): boolean {
  return TERMINAL_JOB_STATUSES.has(job.status);
}

function preferPositiveMetric(primary: unknown, fallback: unknown): number | undefined {
  const preferred = Number(primary);
  if (Number.isFinite(preferred) && preferred > 0) return preferred;
  const fallbackNumber = Number(fallback);
  return Number.isFinite(fallbackNumber) ? fallbackNumber : undefined;
}

function formatDuration(value: unknown): string {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m ${seconds % 60}s`;
}

function AudioClip({ audioId }: { audioId: string }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    let current = true;
    let objectUrl = "";
    loadVoiceClip(audioId)
      .then((blob) => {
        if (!blob || !current) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => undefined);
    return () => {
      current = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [audioId]);
  return url ? <audio className="message-audio" controls preload="metadata" src={url} /> : null;
}

function RuntimeChart({ values }: { values: number[] }) {
  const points = useMemo(() => {
    const source = values.length ? values : [0];
    const maximum = Math.max(...source, 1);
    return source
      .map((value, index) => {
        const x = source.length === 1 ? 50 : (index / (source.length - 1)) * 100;
        const y = 94 - (value / maximum) * 82;
        return `${x},${y}`;
      })
      .join(" ");
  }, [values]);
  return (
    <svg className="runtime-chart" preserveAspectRatio="none" viewBox="0 0 100 100">
      <line x1="0" x2="100" y1="94" y2="94" />
      <line x1="0" x2="100" y1="53" y2="53" />
      <line x1="0" x2="100" y1="12" y2="12" />
      <polyline points={points} />
    </svg>
  );
}

type IconName =
  | "activity"
  | "chat"
  | "flask"
  | "download"
  | "edit"
  | "folder"
  | "lightbulb"
  | "moon"
  | "menu"
  | "paperclip"
  | "play"
  | "plus"
  | "refresh"
  | "send"
  | "settings"
  | "stop"
  | "sun"
  | "sun-moon"
  | "trash"
  | "upload"
  | "volume"
  | "volume-off";

function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  return (
    <svg
      aria-hidden="true"
      className="ui-icon"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      {name === "activity" && <><path d="M3 12h4l2.5-7 5 14 2.5-7h4" /></>}
      {name === "chat" && <><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z" /></>}
      {name === "flask" && <><path d="M9 3h6M10 3v6l-5.7 9.2A1.8 1.8 0 0 0 5.8 21h12.4a1.8 1.8 0 0 0 1.5-2.8L14 9V3" /><path d="M7.5 15h9" /></>}
      {name === "download" && <><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M5 21h14" /></>}
      {name === "edit" && <><path d="M4 20h4l11-11a2.8 2.8 0 0 0-4-4L4 16z" /><path d="m13.5 6.5 4 4" /></>}
      {name === "folder" && <path d="M3 6.5A2.5 2.5 0 0 1 5.5 4H10l2 2h6.5A2.5 2.5 0 0 1 21 8.5v8A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5z" />}
      {name === "lightbulb" && <><path d="M9 18h6M10 22h4" /><path d="M8.4 14.7A6 6 0 1 1 15.6 14.7 4.1 4.1 0 0 0 14 18h-4a4.1 4.1 0 0 0-1.6-3.3z" /></>}
      {name === "moon" && <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 8.5 8.5 0 1 0 20.5 14.2z" />}
      {name === "menu" && <><path d="M4 7h16M4 12h16M4 17h16" /></>}
      {name === "paperclip" && <><path d="m20.5 11.5-8.8 8.8a6 6 0 0 1-8.5-8.5l9.5-9.5a4 4 0 0 1 5.7 5.7l-9.6 9.5a2 2 0 0 1-2.8-2.8l8.8-8.8" /></>}
      {name === "play" && <path d="m8 5 11 7-11 7z" fill="currentColor" stroke="none" />}
      {name === "plus" && <><path d="M12 5v14M5 12h14" /></>}
      {name === "refresh" && <><path d="M20 6v5h-5" /><path d="M4 18v-5h5" /><path d="M18.5 9A7 7 0 0 0 6.1 6.1L4 8M5.5 15A7 7 0 0 0 17.9 17.9L20 16" /></>}
      {name === "send" && <><path d="m5 12 7-7 7 7M12 5v14" /></>}
      {name === "settings" && <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6 1.7 1.7 0 0 0 10 3v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1z" /></>}
      {name === "stop" && <><rect height="9" rx="1" width="9" x="7.5" y="7.5" /></>}
      {name === "sun" && <><circle cx="12" cy="12" r="3.5" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>}
      {name === "sun-moon" && <><path d="M8.5 3.5A6.5 6.5 0 1 0 15 10a5 5 0 0 1-6.5-6.5z" /><path d="M17 3v2M17 9v2M13 7h2M19 7h2" /></>}
      {name === "trash" && <><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5" /></>}
      {name === "upload" && <><path d="M12 21V9" /><path d="m7 14 5-5 5 5" /><path d="M5 3h14" /></>}
      {name === "volume" && <><path d="M11 5 6.5 9H3v6h3.5L11 19z" /><path d="M15 9a4 4 0 0 1 0 6M18 6a8 8 0 0 1 0 12" /></>}
      {name === "volume-off" && <><path d="M11 5 6.5 9H3v6h3.5L11 19zM16 10l5 5M21 10l-5 5" /></>}
    </svg>
  );
}

const PANEL_LAYOUT_KEY = "mfq.studio.panel-offsets.v2";
const PANEL_COLLAPSED_KEY = "mfq.studio.panel-collapsed.v2";

interface PanelPlacement {
  x: number;
  y: number;
  z: number;
  width?: number;
  height?: number;
}

type PanelLayouts = Record<string, Record<string, PanelPlacement>>;

interface PanelOverlap {
  id: string;
  left: number;
  top: number;
  width: number;
  height: number;
  drawTop: boolean;
  drawRight: boolean;
  drawBottom: boolean;
  drawLeft: boolean;
}

interface PanelResizeState {
  id: string;
  edges: string;
  startClientX: number;
  startClientY: number;
  startX: number;
  startY: number;
  startWidth: number;
  startHeight: number;
  baseLeft: number;
  baseTop: number;
  deckWidth: number;
}

const PANEL_RESIZE_INSET = 7;
const PANEL_MIN_WIDTH = 180;
const PANEL_MIN_HEIGHT = 72;

function panelResizeEdges(node: HTMLElement, clientX: number, clientY: number): string {
  const rect = node.getBoundingClientRect();
  const horizontal = clientX - rect.left <= PANEL_RESIZE_INSET
    ? "w"
    : rect.right - clientX <= PANEL_RESIZE_INSET ? "e" : "";
  const vertical = clientY - rect.top <= PANEL_RESIZE_INSET
    ? "n"
    : rect.bottom - clientY <= PANEL_RESIZE_INSET ? "s" : "";
  return `${vertical}${horizontal}`;
}

function panelResizeCursor(edges: string): string {
  if (edges === "n" || edges === "s") return "ns-resize";
  if (edges === "e" || edges === "w") return "ew-resize";
  if (edges === "ne" || edges === "sw") return "nesw-resize";
  if (edges === "nw" || edges === "se") return "nwse-resize";
  return "";
}

function loadPanelLayout(): PanelLayouts {
  try {
    const value = JSON.parse(localStorage.getItem(PANEL_LAYOUT_KEY) || "{}");
    return value && typeof value === "object" ? value as PanelLayouts : {};
  } catch {
    return {};
  }
}

function loadCollapsedPanels(): Record<string, boolean> {
  try {
    const value = JSON.parse(localStorage.getItem(PANEL_COLLAPSED_KEY) || "{}");
    return value && typeof value === "object" ? value as Record<string, boolean> : {};
  } catch {
    return {};
  }
}

interface PanelDeckProps {
  page: string;
  children: ReactNode;
  labels: { collapse: string; drag: string; expand: string };
  resetVersion: number;
}

function panelKey(panel: React.ReactElement, index: number): string {
  const value = String(panel.key ?? `panel-${index}`);
  return value.startsWith(".$") ? value.slice(2) : value.startsWith(".") ? value.slice(1) : value;
}

function PanelDeck({ page, children, labels, resetVersion }: PanelDeckProps) {
  const panels = Children.toArray(children).filter(isValidElement);
  const ids = panels.map(panelKey);
  const panelSignature = ids.join("\u0000");
  const [layouts, setLayouts] = useState<PanelLayouts>(loadPanelLayout);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>(loadCollapsedPanels);
  const [dragged, setDragged] = useState<string | null>(null);
  const [resized, setResized] = useState<string | null>(null);
  const [overlaps, setOverlaps] = useState<PanelOverlap[]>([]);
  const deckRef = useRef<HTMLDivElement | null>(null);
  const panelRefs = useRef(new Map<string, HTMLDivElement>());
  const layoutsRef = useRef(layouts);
  const resetVersionRef = useRef(resetVersion);
  const overlapFrameRef = useRef<number | null>(null);
  const overlapResetTimerRef = useRef<number | null>(null);
  const overlapResettingRef = useRef(false);
  const dragRef = useRef<{ id: string; startClientX: number; startClientY: number; startX: number; startY: number } | null>(null);
  const resizeRef = useRef<PanelResizeState | null>(null);
  const pageLayout = layouts[page] ?? {};
  const byId = new Map(panels.map((panel, index) => [panelKey(panel, index), panel]));
  const panelClasses = `panel-deck page-${page}${panels.length === 1 ? " single" : ""}`;
  const defaultFullWidth: Record<string, string[]> = {
    "dashboard-overview": ["metrics"],
    "dashboard-cache": ["prefix-cache", "profiles"],
    "dashboard-connections": ["mcp", "nodes"],
    "lab-models": ["hubs"],
    "lab-quantization": ["imatrix", "detail"],
  };

  useEffect(() => { layoutsRef.current = layouts; }, [layouts]);

  useEffect(() => {
    if (resetVersionRef.current === resetVersion) return;
    resetVersionRef.current = resetVersion;
    overlapResettingRef.current = true;
    if (overlapFrameRef.current !== null) {
      cancelAnimationFrame(overlapFrameRef.current);
      overlapFrameRef.current = null;
    }
    if (overlapResetTimerRef.current !== null) clearTimeout(overlapResetTimerRef.current);
    setLayouts((current) => {
      const updated = { ...current };
      delete updated[page];
      layoutsRef.current = updated;
      localStorage.setItem(PANEL_LAYOUT_KEY, JSON.stringify(updated));
      return updated;
    });
    setCollapsed((current) => {
      const updated = Object.fromEntries(Object.entries(current).filter(([key]) => !key.startsWith(`${page}:`)));
      localStorage.setItem(PANEL_COLLAPSED_KEY, JSON.stringify(updated));
      return updated;
    });
    panelRefs.current.forEach((node) => {
      node.style.cursor = "";
      node.querySelectorAll<HTMLElement>(".panel-item-clipped").forEach((item) => item.classList.remove("panel-item-clipped"));
    });
    resizeRef.current = null;
    dragRef.current = null;
    setResized(null);
    setDragged(null);
    setOverlaps([]);
    document.body.style.cursor = "";
    document.body.classList.remove("panel-resizing", "panel-reordering");
    overlapResetTimerRef.current = window.setTimeout(() => {
      overlapResetTimerRef.current = null;
      overlapResettingRef.current = false;
      requestAnimationFrame(() => {
        fitPanelItems();
        measureOverlaps();
      });
    }, 180);
  }, [page, resetVersion]);

  const fitPanelItems = useCallback(() => {
    const groupSelector = [
      ".metric-grid", ".request-stats", ".cache-stats", ".model-list", ".request-table",
      ".runtime-log-list", ".profile-list", ".node-list", ".mcp-server-list", ".hub-results",
      ".evaluation-list", ".dataset-list", ".imatrix-list", ".lineage-list", ".job-list",
      ".comparison-table",
    ].join(",");
    panelRefs.current.forEach((panelNode) => {
      panelNode.querySelectorAll<HTMLElement>(".panel-item-clipped").forEach((item) => item.classList.remove("panel-item-clipped"));
      if (!panelNode.classList.contains("custom-sized")) return;
      const panelRect = panelNode.getBoundingClientRect();
      const bottom = panelRect.bottom - 8;
      const right = panelRect.right - 8;
      panelNode.querySelectorAll<HTMLElement>(groupSelector).forEach((group) => {
        [...group.children].forEach((child) => {
          if (!(child instanceof HTMLElement)) return;
          const rect = child.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0 && (rect.bottom > bottom + 0.5 || rect.right > right + 0.5)) {
            child.classList.add("panel-item-clipped");
          }
        });
      });
      panelNode.querySelectorAll<HTMLElement>(".dashboard-panel > :not(.panel-heading)").forEach((item) => {
        if (item.matches(groupSelector)) return;
        const rect = item.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0 && (rect.bottom > bottom + 0.5 || rect.right > right + 0.5)) {
          item.classList.add("panel-item-clipped");
        }
      });
    });
  }, []);

  const measureOverlaps = useCallback(() => {
    const deck = deckRef.current;
    if (!deck) return;
    const deckRect = deck.getBoundingClientRect();
    const entries = [...panelRefs.current.entries()].filter(([, node]) => node.isConnected);
    const next: PanelOverlap[] = [];
    for (let index = 0; index < entries.length; index += 1) {
      const [firstId, firstNode] = entries[index];
      const first = firstNode.getBoundingClientRect();
      for (let otherIndex = index + 1; otherIndex < entries.length; otherIndex += 1) {
        const [secondId, secondNode] = entries[otherIndex];
        const second = secondNode.getBoundingClientRect();
        const left = Math.max(first.left, second.left);
        const right = Math.min(first.right, second.right);
        const top = Math.max(first.top, second.top);
        const bottom = Math.min(first.bottom, second.bottom);
        if (left >= right || top >= bottom) continue;
        const firstZ = Number.parseInt(firstNode.style.zIndex || "1", 10);
        const secondZ = Number.parseInt(secondNode.style.zIndex || "1", 10);
        const covering = firstZ > secondZ ? first : second;
        const edgeTolerance = 0.5;
        next.push({
          id: `${firstId}:${secondId}`,
          left: left - deckRect.left,
          top: top - deckRect.top,
          width: right - left,
          height: bottom - top,
          drawTop: Math.abs(top - covering.top) < edgeTolerance,
          drawRight: Math.abs(right - covering.right) < edgeTolerance,
          drawBottom: Math.abs(bottom - covering.bottom) < edgeTolerance,
          drawLeft: Math.abs(left - covering.left) < edgeTolerance,
        });
      }
    }
    setOverlaps((current) => {
      if (current.length === next.length && current.every((overlap, index) => {
        const candidate = next[index];
        return overlap.id === candidate.id
          && Math.abs(overlap.left - candidate.left) < 0.25
          && Math.abs(overlap.top - candidate.top) < 0.25
          && Math.abs(overlap.width - candidate.width) < 0.25
          && Math.abs(overlap.height - candidate.height) < 0.25
          && overlap.drawTop === candidate.drawTop
          && overlap.drawRight === candidate.drawRight
          && overlap.drawBottom === candidate.drawBottom
          && overlap.drawLeft === candidate.drawLeft;
      })) return current;
      return next;
    });
  }, []);

  const scheduleOverlapCheck = useCallback(() => {
    if (overlapFrameRef.current !== null) cancelAnimationFrame(overlapFrameRef.current);
    overlapFrameRef.current = requestAnimationFrame(() => {
      overlapFrameRef.current = null;
      if (overlapResettingRef.current) {
        setOverlaps([]);
        return;
      }
      fitPanelItems();
      measureOverlaps();
    });
  }, [fitPanelItems, measureOverlaps]);

  useEffect(() => { scheduleOverlapCheck(); }, [collapsed, layouts, page, panelSignature, scheduleOverlapCheck]);

  useEffect(() => {
    const observer = new ResizeObserver(scheduleOverlapCheck);
    if (deckRef.current) observer.observe(deckRef.current);
    panelRefs.current.forEach((node) => observer.observe(node));
    window.addEventListener("resize", scheduleOverlapCheck);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", scheduleOverlapCheck);
      if (overlapFrameRef.current !== null) cancelAnimationFrame(overlapFrameRef.current);
      if (overlapResetTimerRef.current !== null) clearTimeout(overlapResetTimerRef.current);
    };
  }, [page, panelSignature, scheduleOverlapCheck]);

  useEffect(() => () => {
    overlapResettingRef.current = false;
    document.body.style.cursor = "";
    document.body.classList.remove("panel-resizing", "panel-reordering");
  }, []);

  function bringPanelToFront(id: string) {
    setLayouts((current) => {
      const existing = current[page] ?? {};
      const placement = existing[id] ?? { x: 0, y: 0, z: 1 };
      const maximumZ = Math.max(1, ...ids.map((panelId) => existing[panelId]?.z ?? 1));
      if (placement.z > maximumZ) return current;
      const updated = { ...current, [page]: { ...existing, [id]: { ...placement, z: maximumZ + 1 } } };
      layoutsRef.current = updated;
      localStorage.setItem(PANEL_LAYOUT_KEY, JSON.stringify(updated));
      return updated;
    });
  }

  function updatePlacement(id: string, x: number, y: number) {
    setLayouts((current) => {
      const existing = current[page] ?? {};
      const placement = existing[id] ?? { x: 0, y: 0, z: 1 };
      const updated = { ...current, [page]: { ...existing, [id]: { ...placement, x, y } } };
      layoutsRef.current = updated;
      return updated;
    });
  }

  function updatePanelSize(id: string, placement: PanelPlacement) {
    setLayouts((current) => {
      const existing = current[page] ?? {};
      const updated = { ...current, [page]: { ...existing, [id]: placement } };
      layoutsRef.current = updated;
      return updated;
    });
  }

  function beginPanelResize(event: React.PointerEvent<HTMLDivElement>, id: string, isCollapsed: boolean): boolean {
    if (isCollapsed || window.matchMedia("(max-width: 860px)").matches || !event.isPrimary || event.button !== 0 || dragRef.current) return false;
    const node = event.currentTarget;
    const edges = panelResizeEdges(node, event.clientX, event.clientY);
    if (!edges) return false;
    event.preventDefault();
    event.stopPropagation();
    node.setPointerCapture(event.pointerId);
    const deck = deckRef.current;
    const current = layoutsRef.current[page]?.[id] ?? { x: 0, y: 0, z: 1 };
    bringPanelToFront(id);
    resizeRef.current = {
      id,
      edges,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startX: current.x,
      startY: current.y,
      startWidth: node.offsetWidth,
      startHeight: node.offsetHeight,
      baseLeft: node.offsetLeft,
      baseTop: node.offsetTop,
      deckWidth: deck?.clientWidth ?? node.offsetWidth,
    };
    const cursor = panelResizeCursor(edges);
    node.style.cursor = cursor;
    document.body.style.cursor = cursor;
    document.body.classList.add("panel-resizing");
    setResized(id);
    return true;
  }

  function movePanelResize(event: React.PointerEvent<HTMLDivElement>, id: string, isCollapsed: boolean) {
    const activeResize = resizeRef.current;
    if (!activeResize || activeResize.id !== id) {
      const narrowViewport = window.matchMedia("(max-width: 860px)").matches;
      event.currentTarget.style.cursor = isCollapsed || narrowViewport ? "" : panelResizeCursor(panelResizeEdges(event.currentTarget, event.clientX, event.clientY));
      return;
    }
    event.preventDefault();
    const deltaX = event.clientX - activeResize.startClientX;
    const deltaY = event.clientY - activeResize.startClientY;
    const current = layoutsRef.current[page]?.[id] ?? { x: activeResize.startX, y: activeResize.startY, z: 1 };
    let x = activeResize.startX;
    let y = activeResize.startY;
    let width = activeResize.startWidth;
    let height = activeResize.startHeight;
    if (activeResize.edges.includes("w")) {
      const startingRight = activeResize.baseLeft + activeResize.startX + activeResize.startWidth;
      const left = Math.min(startingRight - PANEL_MIN_WIDTH, Math.max(0, activeResize.baseLeft + activeResize.startX + deltaX));
      x = left - activeResize.baseLeft;
      width = startingRight - left;
    } else if (activeResize.edges.includes("e")) {
      const startingLeft = activeResize.baseLeft + activeResize.startX;
      const right = Math.min(Math.max(activeResize.deckWidth, startingLeft + PANEL_MIN_WIDTH), Math.max(startingLeft + PANEL_MIN_WIDTH, startingLeft + activeResize.startWidth + deltaX));
      width = right - startingLeft;
    }
    if (activeResize.edges.includes("n")) {
      const startingBottom = activeResize.baseTop + activeResize.startY + activeResize.startHeight;
      const top = Math.min(startingBottom - PANEL_MIN_HEIGHT, Math.max(0, activeResize.baseTop + activeResize.startY + deltaY));
      y = top - activeResize.baseTop;
      height = startingBottom - top;
    } else if (activeResize.edges.includes("s")) {
      height = Math.max(PANEL_MIN_HEIGHT, activeResize.startHeight + deltaY);
    }
    updatePanelSize(id, { ...current, x, y, width, height });
  }

  function finishPanelResize(event: React.PointerEvent<HTMLDivElement>) {
    const activeResize = resizeRef.current;
    if (!activeResize) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    resizeRef.current = null;
    setResized(null);
    event.currentTarget.style.cursor = "";
    document.body.style.cursor = "";
    document.body.classList.remove("panel-resizing");
    localStorage.setItem(PANEL_LAYOUT_KEY, JSON.stringify(layoutsRef.current));
    scheduleOverlapCheck();
  }

  function finishPanelDrag(event: React.PointerEvent<HTMLButtonElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    setDragged(null);
    document.body.classList.remove("panel-reordering");
    localStorage.setItem(PANEL_LAYOUT_KEY, JSON.stringify(layoutsRef.current));
  }

  return <div className={panelClasses} ref={deckRef}>{ids.map((id) => {
    const panel = byId.get(id);
    if (!panel) return null;
    const isCollapsed = Boolean(collapsed[`${page}:${id}`]);
    const placement = pageLayout[id];
    const wide = panels.length === 1 || defaultFullWidth[page]?.includes(id);
    const positionStyle = {
      "--panel-x": `${placement?.x ?? 0}px`,
      "--panel-y": `${placement?.y ?? 0}px`,
      width: placement?.width ? `${placement.width}px` : undefined,
      height: isCollapsed ? "47px" : placement?.height ? `${placement.height}px` : undefined,
      zIndex: placement?.z ?? 1,
    } as React.CSSProperties;
    return <div className={`panel-shell ${wide ? "wide" : ""} ${placement?.width || placement?.height ? "custom-sized" : ""} ${isCollapsed ? "collapsed" : ""} ${resized === id ? "resizing" : ""} ${dragged === id ? "dragging" : ""}`} data-panel-id={id} key={id} onDoubleClick={() => bringPanelToFront(id)} onPointerCancel={finishPanelResize} onPointerDown={(event) => { void beginPanelResize(event, id, isCollapsed); }} onPointerLeave={(event) => { if (!resizeRef.current) event.currentTarget.style.cursor = ""; }} onPointerMove={(event) => movePanelResize(event, id, isCollapsed)} onPointerUp={finishPanelResize} ref={(node) => { if (node) panelRefs.current.set(id, node); else panelRefs.current.delete(id); }} style={positionStyle}>
      <button aria-label={labels.drag} className="panel-drag" onPointerCancel={finishPanelDrag} onPointerDown={(event) => {
        if (!event.isPrimary || event.button !== 0) return;
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        const current = layoutsRef.current[page]?.[id] ?? { x: 0, y: 0, z: 1 };
        bringPanelToFront(id);
        dragRef.current = { id, startClientX: event.clientX, startClientY: event.clientY, startX: current.x, startY: current.y };
        setDragged(id);
        document.body.classList.add("panel-reordering");
      }} onPointerMove={(event) => {
        const activeDrag = dragRef.current;
        if (!activeDrag) return;
        event.preventDefault();
        const deck = deckRef.current;
        const node = panelRefs.current.get(activeDrag.id);
        const baseLeft = node?.offsetLeft ?? 0;
        const baseTop = node?.offsetTop ?? 0;
        const minimumX = -baseLeft;
        const maximumX = Math.max(minimumX, (deck?.clientWidth ?? 0) - baseLeft - (node?.offsetWidth ?? 0));
        const minimumY = -baseTop;
        const proposedX = activeDrag.startX + event.clientX - activeDrag.startClientX;
        const proposedY = activeDrag.startY + event.clientY - activeDrag.startClientY;
        const x = Math.min(maximumX, Math.max(minimumX, proposedX));
        const y = Math.max(minimumY, proposedY);
        updatePlacement(activeDrag.id, x, y);
      }} onPointerUp={finishPanelDrag} tabIndex={-1} type="button"><span /><span /><span /><span /><span /><span /></button>
      <button aria-expanded={!isCollapsed} aria-label={isCollapsed ? labels.expand : labels.collapse} className="panel-collapse" onClick={() => setCollapsed((current) => {
        const updated = { ...current, [`${page}:${id}`]: !isCollapsed };
        localStorage.setItem(PANEL_COLLAPSED_KEY, JSON.stringify(updated));
        return updated;
      })} type="button">⌄</button>
      {panel}
    </div>;
  })}<div aria-hidden="true" className="panel-overlap-layer">{overlaps.map((overlap) => <span className={`panel-overlap-boundary${overlap.drawTop ? " edge-top" : ""}${overlap.drawRight ? " edge-right" : ""}${overlap.drawBottom ? " edge-bottom" : ""}${overlap.drawLeft ? " edge-left" : ""}`} key={overlap.id} style={{ height: overlap.height, left: overlap.left, top: overlap.top, width: overlap.width }} />)}</div></div>;
}

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [responses, setResponses] = useState<Record<string, ResponseResource>>({});
  const [voiceMessages, setVoiceMessages] = useState<VoiceMessage[]>(loadVoiceHistory);
  const [models, setModels] = useState<RuntimeModel[]>([]);
  const [artifacts, setArtifacts] = useState<ModelArtifact[]>([]);
  const [instances, setInstances] = useState<RuntimeInstance[]>([]);
  const [runtimeProfiles, setRuntimeProfiles] = useState<RuntimeProfile[]>([]);
  const [remoteNodes, setRemoteNodes] = useState<RemoteNode[]>([]);
  const [nodeDraft, setNodeDraft] = useState({ name: "", url: "", api_key_env: "" });
  const [lineage, setLineage] = useState<ArtifactLineage[]>([]);
  const [datasets, setDatasets] = useState<DatasetResource[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationResult[]>([]);
  const [selectedEvaluations, setSelectedEvaluations] = useState<string[]>([]);
  const [evaluationComparison, setEvaluationComparison] = useState<EvaluationComparison | null>(null);
  const [datasetDraft, setDatasetDraft] = useState({ name: "", artifact_uri: "", kind: "custom" as DatasetResource["kind"] });
  const [jobs, setJobs] = useState<JobResource[]>([]);
  const [jobCleanupBusy, setJobCleanupBusy] = useState(false);
  const [jobKinds, setJobKinds] = useState<JobKindResource[]>([]);
  const [mcpServers, setMcpServers] = useState<McpServerResource[]>([]);
  const [mcpTools, setMcpTools] = useState<McpToolResource[]>([]);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [mcpDraft, setMcpDraft] = useState({
    name: "",
    transport: "streamable_http" as "stdio" | "streamable_http",
    endpoint: "",
  });
  const [selectedJobKind, setSelectedJobKind] = useState("");
  const [jobPayload, setJobPayload] = useState<Record<string, unknown>>({});
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [imatrixImporting, setImatrixImporting] = useState(false);
  const [pendingImatrix, setPendingImatrix] = useState("");
  const imatrixInputRef = useRef<HTMLInputElement | null>(null);
  const [jobLogs, setJobLogs] = useState<RuntimeLogEntry[]>([]);
  const [hubProvider, setHubProvider] = useState<HubModelSummary["provider"]>("modelscope");
  const [hubQuery, setHubQuery] = useState("");
  const [hubResults, setHubResults] = useState<HubModelSummary[]>([]);
  const [hubModel, setHubModel] = useState<HubModelInfo | null>(null);
  const [runtimeLogs, setRuntimeLogs] = useState<RuntimeLogEntry[]>([]);
  const [model, setModel] = useState("");
  const [mode, setMode] = useState<SessionMode>("text");
  const [view, setView] = useState<ViewName>("chat");
  const [dashboardPage, setDashboardPage] = useState<DashboardPage>("overview");
  const [labPage, setLabPage] = useState<LabPage>("models");
  const [dashboardLayoutReset, setDashboardLayoutReset] = useState(0);
  const [labLayoutReset, setLabLayoutReset] = useState(0);
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const [live, setLive] = useState<LiveOutput | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [realtime, setRealtime] = useState<RealtimeCapabilities | null>(null);
  const [realtimeAvailable, setRealtimeAvailable] = useState(false);
  const [metricSeries, setMetricSeries] = useState<number[]>([]);
  const [requestHistory, setRequestHistory] = useState<RuntimeRequestMetrics[]>([]);
  const [settings, setSettings] = useState<GenerationSettings>(loadSettings);
  const [settingsDraft, setSettingsDraft] = useState<GenerationSettings>(settings);
  const [storedPresets, setStoredPresets] = useState<StoredPreset[]>(loadStoredPresets);
  const [selectedStoredPreset, setSelectedStoredPreset] = useState("");
  const [storedPresetName, setStoredPresetName] = useState("");
  const [presetStatus, setPresetStatus] = useState<{ error: boolean; text: string } | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedAssistantId, setSelectedAssistantId] = useState(DEFAULT_ASSISTANT_ID);
  const [roleEditor, setRoleEditor] = useState<RoleEditorDraft | null>(null);
  const [contextSize, setContextSize] = useState(32768);
  const [profileName, setProfileName] = useState("");
  const [studio, setStudio] = useState<StudioStatus | null>(null);
  const [studioDraft, setStudioDraft] = useState<StudioConfig | null>(null);
  const [studioOpen, setStudioOpen] = useState(false);
  const [modelBrowser, setModelBrowser] = useState<ModelDirectoryList | null>(null);
  const [modelBrowserOpen, setModelBrowserOpen] = useState(false);
  const [modelDirectoryPath, setModelDirectoryPath] = useState("");
  const [studioToken, setStudioToken] = useState("");
  const [editDraft, setEditDraft] = useState<EditDraft | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [voiceLevel, setVoiceLevel] = useState(0);
  const [liveVoice, setLiveVoice] = useState<LiveVoiceOutput | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const attachmentInputRef = useRef<HTMLInputElement | null>(null);
  const messageScrollerRef = useRef<HTMLDivElement | null>(null);
  const autoFollowOutputRef = useRef(true);
  const voiceRef = useRef<RealtimeAudioController | null>(null);
  const voiceClipWrites = useRef(new Map<string, Promise<void>>());
  const lastMetricId = useRef("");
  const appliedModeTemplate = useRef("");
  const settingsCloseRef = useRef<HTMLButtonElement | null>(null);

  const english =
    settings.language === "en" ||
    (settings.language === "system" && !navigator.language.toLowerCase().startsWith("zh"));
  const tr = useCallback((zh: string, en: string) => (english ? en : zh), [english]);
  const canUseNativeModelPicker = isStudio() && studio?.config.mode !== "remote";
  const panelLabels = useMemo(() => ({
    collapse: tr("收起面板", "Collapse panel"),
    drag: tr("拖动面板", "Drag panel"),
    expand: tr("展开面板", "Expand panel"),
  }), [tr]);
  const active = useMemo(
    () => sessions.find((session) => session.id === activeId) ?? null,
    [activeId, sessions],
  );
  const assistantRoles = useMemo<AssistantRole[]>(() => {
    const roles = new Map<string, AssistantRole>();
    for (const preset of storedPresets) {
      const id = assistantIdForPreset(preset);
      roles.set(id, { id, name: preset.name, preset });
    }
    for (const session of sessions) {
      const id = canonicalSessionAssistantId(session, storedPresets);
      if (!roles.has(id)) {
        roles.set(id, { id, name: sessionAssistantName(session) ?? tr("已删除的角色", "Removed role") });
      }
    }
    if (!roles.size) {
      roles.set(DEFAULT_ASSISTANT_ID, { id: DEFAULT_ASSISTANT_ID, name: tr("默认助手", "Default assistant") });
    }
    return [...roles.values()];
  }, [sessions, storedPresets, tr]);
  const assistantSessions = useMemo(
    () => sessions.filter((session) => canonicalSessionAssistantId(session, storedPresets) === selectedAssistantId),
    [selectedAssistantId, sessions, storedPresets],
  );
  const currentVoiceMessages = useMemo(
    () => voiceMessages.filter((message) => message.sessionId === activeId),
    [activeId, voiceMessages],
  );
  const activeRolePreset = useMemo(
    () => active
      ? storedPresets.find(
          (preset) =>
            assistantIdForPreset(preset) === canonicalSessionAssistantId(active, storedPresets),
        )
      : undefined,
    [active, storedPresets],
  );
  const resolvedGlobalSettings = useMemo(
    () => settings.inheritModelDefaults
      ? modeTemplateSettings(settings, active?.mode ?? mode, runtime, realtime)
      : settings,
    [active?.mode, mode, realtime, runtime, settings],
  );
  const effectiveSettings = useMemo(
    () => roleGenerationSettings(resolvedGlobalSettings, activeRolePreset),
    [activeRolePreset, resolvedGlobalSettings],
  );
  const roleOverridesInference = Boolean(
    activeRolePreset && !activeRolePreset.inheritGlobalSettings,
  );
  useEffect(() => {
    if (active) setSelectedAssistantId(canonicalSessionAssistantId(active, storedPresets));
  }, [active, storedPresets]);
  const reasoningValues = useMemo(() => {
    const values = runtime?.chat_template_capabilities?.reasoning_effort?.values;
    return Array.isArray(values) ? values : [];
  }, [runtime]);
  const thinkingSupported =
    runtime?.chat_template_capabilities?.thinking?.supported === true;
  const attachmentAccept = useMemo(() => {
    const features = capabilities?.model_capabilities.features;
    return [
      features?.image_input ? "image/*" : "",
      features?.video_input ? "video/*" : "",
      features?.audio_input ? "audio/*" : "",
      DOCUMENT_ACCEPT,
    ]
      .filter(Boolean)
      .join(",");
  }, [capabilities]);

  const refreshSessions = useCallback(async (preferredId?: string) => {
    const next = await api.listSessions();
    setSessions(next);
    setActiveId((current) => {
      const wanted = preferredId ?? current;
      if (wanted && next.some((session) => session.id === wanted)) return wanted;
      return next[0]?.id ?? null;
    });
  }, []);

  const refreshMcp = useCallback(async () => {
    const [servers, tools] = await Promise.all([api.mcpServers(), api.mcpTools()]);
    setMcpServers(servers);
    setMcpTools(tools.data);
  }, []);

  const refreshRuntime = useCallback(async (quiet = true) => {
    try {
      const [runtimeResults, management] = await Promise.all([
        Promise.allSettled([
          api.runtimeCapabilities(),
          api.runtimeModels(),
          api.runtimeStatus(),
          api.realtimeCapabilities(),
        ]),
        Promise.all([
          api.runtimeMetrics(200),
          api.modelArtifacts(),
          api.runtimeInstances(),
          api.runtimeProfiles(),
          api.jobs(100),
          api.runtimeLogs(100),
          api.jobKinds(),
          api.artifactLineage(),
          api.datasets(),
          api.evaluations(),
          api.remoteNodes(),
        ]),
      ]);
      const [capabilityResult, modelResult, statusResult, realtimeResult] = runtimeResults;
      const [metricHistory, nextArtifacts, nextInstances, nextProfiles, nextJobs, nextLogs, nextKinds, nextLineage, nextDatasets, nextEvaluations, nextNodes] = management;
      if (capabilityResult.status === "fulfilled") {
        setCapabilities(capabilityResult.value);
        setModel(capabilityResult.value.model);
      } else {
        setCapabilities(null);
      }
      if (modelResult.status === "fulfilled") {
        setModels(modelResult.value);
        if (capabilityResult.status !== "fulfilled") {
          setModel((current) => current || modelResult.value[0]?.id || "");
        }
      }
      const status = statusResult.status === "fulfilled" ? statusResult.value : null;
      setRuntime(status);
      if (realtimeResult.status === "fulfilled") {
        setRealtime(realtimeResult.value);
        setRealtimeAvailable(realtimeResult.value.available === true);
      } else {
        setRealtime(null);
        setRealtimeAvailable(false);
      }
      setArtifacts(nextArtifacts);
      setInstances(nextInstances);
      setRuntimeProfiles(nextProfiles);
      setJobs(nextJobs);
      setRuntimeLogs(nextLogs);
      setJobKinds(nextKinds);
      setLineage(nextLineage);
      setDatasets(nextDatasets);
      setEvaluations(nextEvaluations);
      setRemoteNodes(nextNodes);
      setSelectedJobKind((current) => current || nextKinds[0]?.kind || "");
      const historicRequests = metricHistory
        .map((snapshot) => snapshot.values.last_request)
        .filter((request): request is RuntimeRequestMetrics => Boolean(request?.id))
        .filter(
          (request, index, values) =>
            values.findIndex((candidate) => candidate.id === request.id) === index,
        );
      setRequestHistory(historicRequests.slice(-24).reverse());
      setMetricSeries(
        historicRequests
          .map((request) => Number(request.decode_tps))
          .filter(Number.isFinite)
          .slice(-32),
      );
      const request = status?.last_request;
      const requestId = String(request?.id ?? "");
      const decode = Number(request?.decode_tps);
      if (request && requestId && requestId !== lastMetricId.current && Number.isFinite(decode)) {
        lastMetricId.current = requestId;
        setMetricSeries((current) => [...current, decode].slice(-32));
      }
      const currentContext = Number(status?.max_context);
      if (Number.isFinite(currentContext) && currentContext > 0) {
        setContextSize(Math.floor(currentContext));
      }
    } catch (cause) {
      if (!quiet) setError(errorMessage(cause));
    }
  }, []);

  useEffect(() => {
    if (!active || !runtime || voiceRef.current?.active) return;
    const key = `${runtime.model ?? model}:${active.mode}`;
    if (appliedModeTemplate.current === key) return;
    appliedModeTemplate.current = key;
    void voiceRef.current?.setFullDuplex(active.mode === "full_duplex");
  }, [active, model, runtime]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    voiceRef.current?.setPlayback(settings.playbackEnabled);
    document.documentElement.dataset.theme = settings.theme;
  }, [settings]);

  useEffect(() => {
    if (settingsOpen) settingsCloseRef.current?.focus();
  }, [settingsOpen]);

  useEffect(() => {
    localStorage.setItem(STORED_PRESETS_KEY, JSON.stringify(storedPresets));
  }, [storedPresets]);

  useEffect(() => {
    const selected = jobKinds.find((item) => item.kind === selectedJobKind);
    const properties = selected?.payload_schema.properties ?? {};
    setJobPayload(
      Object.fromEntries(
        Object.entries(properties).map(([name, property]) => [
          name,
          selectedJobKind === "model.quantize" && name === "imatrix" && pendingImatrix
            ? pendingImatrix
            : schemaDefault(property),
        ]),
      ),
    );
  }, [jobKinds, selectedJobKind, pendingImatrix]);

  useEffect(() => {
    if (!selectedJobId) {
      setJobLogs([]);
      return;
    }
    void api.jobEvents(selectedJobId).then(setJobLogs).catch(() => undefined);
  }, [jobs, selectedJobId]);

  useEffect(() => {
    const stable = voiceMessages.filter(
      (message) => !message.pending && (message.text.trim() || message.audioId),
    );
    localStorage.setItem(VOICE_HISTORY_KEY, JSON.stringify(stable.slice(-200)));
  }, [voiceMessages]);

  useEffect(() => {
    voiceRef.current = new RealtimeAudioController(
      {
        onState: setVoiceState,
        onLevel: setVoiceLevel,
        onText: (sessionId, text) =>
          setLiveVoice((current) =>
            text ? { sessionId, text } : current?.sessionId === sessionId ? null : current,
          ),
        onError: (message) => setError(message),
        onInputStart: ({ id, sessionId }) => {
          setVoiceMessages((current) => [
            ...current,
            {
              id,
              sessionId,
              role: "user",
              text: "",
              pending: true,
              created_at: new Date().toISOString(),
            },
          ]);
        },
        onInputEnd: ({ id, sessionId, audio }) => {
          const persist = async () => {
            if (!audio) {
              setVoiceMessages((current) => current.filter((message) => message.id !== id));
              return;
            }
            const audioId = `voice-${id}`;
            await saveVoiceClip(audioId, audio);
            setVoiceMessages((current) =>
              current.map((message) =>
                message.id === id && message.sessionId === sessionId
                  ? { ...message, audioId, pending: false }
                  : message,
              ),
            );
          };
          void persist().catch((cause) => setError(errorMessage(cause)));
        },
        onTurn: ({ id, sessionId, text, audio }) => {
          setVoiceMessages((current) => {
            const existing = current.find((message) => message.id === id);
            if (existing) {
              return current.map((message) =>
                message.id === id ? { ...message, text } : message,
              );
            }
            return [
              ...current,
              {
                id,
                sessionId,
                role: "assistant",
                text,
                created_at: new Date().toISOString(),
              },
            ];
          });
          if (!audio) return;
          const audioId = `voice-${id}`;
          const previous = voiceClipWrites.current.get(audioId) ?? Promise.resolve();
          const persist = previous
            .catch(() => undefined)
            .then(async () => {
              await saveVoiceClip(audioId, audio);
              setVoiceMessages((current) =>
                current.map((message) =>
                  message.id === id ? { ...message, audioId } : message,
                ),
              );
            });
          voiceClipWrites.current.set(audioId, persist);
          void persist
            .catch((cause) => setError(errorMessage(cause)))
            .finally(() => {
              if (voiceClipWrites.current.get(audioId) === persist) {
                voiceClipWrites.current.delete(audioId);
              }
            });
        },
      },
      settings.playbackEnabled,
      settings.fullDuplex,
    );
    return () => {
      void voiceRef.current?.stop();
      voiceRef.current = null;
    };
    // The controller reads current request settings when capture starts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let current = true;
    async function initialize() {
      try {
        let status = await studioStatus();
        if (status?.config.mode === "local" && !status.reachable) {
          status = await startLocalStudio();
        }
        if (status) {
          setApiBaseUrl(status.service_url);
          const token = await studioCredential();
          setApiToken(token);
          if (current) setStudioToken(token);
          if (current) setStudio(status);
        }
        const results = await Promise.allSettled([
          api.runtimeCapabilities(),
          api.runtimeModels(),
          api.runtimeStatus(),
          api.realtimeCapabilities(),
          api.listSessions(),
          api.modelArtifacts(),
          api.runtimeInstances(),
          api.jobs(100),
          api.runtimeMetrics(200),
          api.runtimeLogs(100),
          api.jobKinds(),
          api.mcpServers(),
          api.mcpTools(),
          api.generationPresets(),
          api.runtimeProfiles(),
          api.artifactLineage(),
          api.datasets(),
          api.evaluations(),
          api.remoteNodes(),
        ]);
        if (!current) return;
        if (results[0].status === "fulfilled") {
          setCapabilities(results[0].value);
          setModel(results[0].value.model);
        }
        if (results[1].status === "fulfilled") {
          const nextModels = results[1].value;
          setModels(nextModels);
          if (results[0].status !== "fulfilled") {
            setModel((current) => current || nextModels[0]?.id || "");
          }
        }
        if (results[2].status === "fulfilled") setRuntime(results[2].value);
        if (results[5].status === "fulfilled") setArtifacts(results[5].value);
        if (results[6].status === "fulfilled") setInstances(results[6].value);
        if (results[7].status === "fulfilled") setJobs(results[7].value);
        if (results[8].status === "fulfilled") {
          const history = results[8].value
            .map((snapshot) => snapshot.values.last_request)
            .filter((request): request is RuntimeRequestMetrics => Boolean(request?.id))
            .filter(
              (request, index, values) =>
                values.findIndex((candidate) => candidate.id === request.id) === index,
            );
          setRequestHistory(history.slice(-24).reverse());
          setMetricSeries(
            history
              .map((request) => Number(request.decode_tps))
              .filter(Number.isFinite)
              .slice(-32),
          );
        }
        if (results[9].status === "fulfilled") setRuntimeLogs(results[9].value);
        if (results[10].status === "fulfilled") {
          const kinds = results[10].value;
          setJobKinds(kinds);
          setSelectedJobKind((current) => current || kinds[0]?.kind || "");
        }
        if (results[11].status === "fulfilled") setMcpServers(results[11].value);
        if (results[12].status === "fulfilled") setMcpTools(results[12].value.data);
        if (results[13].status === "fulfilled") {
          if (results[13].value.length > 0) {
            setStoredPresets(results[13].value.map(storedPresetFromResource));
          }
        }
        if (results[14].status === "fulfilled") setRuntimeProfiles(results[14].value);
        if (results[15].status === "fulfilled") setLineage(results[15].value);
        if (results[16].status === "fulfilled") setDatasets(results[16].value);
        if (results[17].status === "fulfilled") setEvaluations(results[17].value);
        if (results[18].status === "fulfilled") setRemoteNodes(results[18].value);
        if (results[3].status === "fulfilled") {
          setRealtime(results[3].value);
          setRealtimeAvailable(results[3].value.available === true);
        }
        if (results[4].status === "fulfilled") {
          setSessions(results[4].value);
          setActiveId(results[4].value[0]?.id ?? null);
        } else {
          throw results[4].reason;
        }
      } catch (cause) {
        if (current) setError(errorMessage(cause));
      } finally {
        if (current) setLoading(false);
      }
    }
    void initialize();
    return () => {
      current = false;
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => void refreshRuntime(true), 2500);
    return () => window.clearInterval(timer);
  }, [refreshRuntime]);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      setResponses({});
      return;
    }
    let current = true;
    Promise.all([api.listMessages(activeId), api.listResponses(activeId)])
      .then(([nextMessages, nextResponses]) => {
        if (!current) return;
        setMessages(nextMessages);
        setResponses(
          Object.fromEntries(
            nextResponses
              .filter((response) => response.output_message_id)
              .map((response) => [response.output_message_id as string, response]),
          ),
        );
      })
      .catch((cause) => current && setError(errorMessage(cause)));
    return () => {
      current = false;
      if (voiceRef.current?.active) void voiceRef.current.stop();
    };
  }, [activeId]);

  useEffect(() => {
    setAttachments((current) => {
      current.forEach((attachment) => {
        if (attachment.previewUrl) URL.revokeObjectURL(attachment.previewUrl);
      });
      return [];
    });
  }, [activeId]);

  const handleMessageScroll = useCallback(() => {
    const scroller = messageScrollerRef.current;
    if (!scroller) return;
    const distanceFromBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    autoFollowOutputRef.current = distanceFromBottom <= 8;
  }, []);

  useEffect(() => {
    autoFollowOutputRef.current = true;
  }, [activeId]);

  useEffect(() => {
    const scroller = messageScrollerRef.current;
    if (!scroller || !autoFollowOutputRef.current) return;
    const frame = window.requestAnimationFrame(() => {
      scroller.scrollTop = scroller.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [messages, currentVoiceMessages, liveVoice, live, busy]);

  useEffect(() => {
    if (capabilities && !capabilities.model_capabilities.features.full_duplex) {
      setMode("text");
    }
  }, [capabilities]);

  function samplingParams(): SamplingParams {
    return {
      max_tokens: effectiveSettings.maxTokens,
      temperature: effectiveSettings.temperature,
      top_k: effectiveSettings.topK,
      top_p: effectiveSettings.topP,
      presence_penalty: effectiveSettings.presencePenalty,
      frequency_penalty: effectiveSettings.frequencyPenalty,
      repetition_penalty: effectiveSettings.repetitionPenalty,
      seed: effectiveSettings.seed,
      enable_thinking: thinkingSupported && effectiveSettings.enableThinking,
      reasoning_effort: effectiveSettings.reasoningEffort || null,
    };
  }

  function realtimeSessionConfig(sessionId: string) {
    return {
      sessionId,
      systemPrompt: effectiveSettings.systemPrompt,
      temperature: effectiveSettings.temperature,
      topP: effectiveSettings.topP,
      topK: effectiveSettings.topK,
      repetitionPenalty: effectiveSettings.repetitionPenalty,
    };
  }

  function selectAssistant(role: AssistantRole) {
    setSelectedAssistantId(role.id);
    const nextSession = sessions.find((session) => canonicalSessionAssistantId(session, storedPresets) === role.id);
    setActiveId(nextSession?.id ?? null);
    if (role.preset) {
      if (role.preset.model) setModel(role.preset.model);
      if (role.preset.mode) setMode(role.preset.mode);
    }
    setView("chat");
    setSidebarOpen(false);
  }

  function editRole(role: AssistantRole) {
    const preset = role.preset;
    const inheritGlobalSettings = preset?.inheritGlobalSettings ?? true;
    const roleMode = preset?.mode || mode;
    const roleGlobalSettings = settings.inheritModelDefaults
      ? modeTemplateSettings(settings, roleMode, runtime, realtime)
      : settings;
    setRoleEditor({
      roleId: role.id,
      name: role.name,
      icon: preset?.icon || role.name.slice(0, 1).toLocaleUpperCase(),
      model: preset?.model || model,
      mode: roleMode,
      contextSize: preset?.contextSize || contextSize,
      inheritGlobalSettings,
      settings: inheritGlobalSettings
        ? {
            ...presetSnapshot(roleGlobalSettings),
            systemPrompt: preset?.settings.systemPrompt ?? "",
          }
        : preset?.settings || presetSnapshot(roleGlobalSettings),
    });
  }

  function createRole() {
    const base = tr("新角色", "New role");
    const used = new Set(assistantRoles.map((role) => role.name.toLocaleLowerCase()));
    let name = base;
    for (let suffix = 2; used.has(name.toLocaleLowerCase()); suffix += 1) name = `${base} ${suffix}`;
    setRoleEditor({
      roleId: "new",
      name,
      icon: name.slice(0, 1).toLocaleUpperCase(),
      model,
      mode,
      contextSize,
      inheritGlobalSettings: true,
      settings: presetSnapshot(
        settings.inheritModelDefaults
          ? modeTemplateSettings(settings, mode, runtime, realtime)
          : settings,
      ),
    });
  }

  async function saveRole(event: FormEvent) {
    event.preventDefault();
    if (!roleEditor) return;
    const name = roleEditor.name.replace(/\s+/g, " ").trim().slice(0, 64);
    if (!name) return;
    if (assistantRoles.some((candidate) => candidate.id !== roleEditor.roleId && candidate.name.toLocaleLowerCase() === name.toLocaleLowerCase())) {
      setError(tr("角色名已存在。", "A role with this name already exists."));
      return;
    }
    const currentRole = assistantRoles.find((role) => role.id === roleEditor.roleId);
    const preset: StoredPreset = {
      id: currentRole?.preset?.id,
      name,
      icon: roleEditor.icon.trim().slice(0, 8) || name.slice(0, 1).toLocaleUpperCase(),
      model: roleEditor.model || null,
      mode: roleEditor.mode,
      contextSize: Math.max(512, Math.floor(roleEditor.contextSize)),
      inheritGlobalSettings: roleEditor.inheritGlobalSettings,
      settings: roleEditor.settings,
      updatedAt: new Date().toISOString(),
    };
    try {
      const saved = storedPresetFromResource(currentRole?.preset?.id
        ? await api.updateGenerationPreset(currentRole.preset.id, presetResourceBody(preset, model, mode))
        : await api.createGenerationPreset(presetResourceBody(preset, model, mode)));
      const nextId = assistantIdForPreset(saved);
      const affected = currentRole
        ? sessions.filter((session) => canonicalSessionAssistantId(session, storedPresets) === currentRole.id)
        : [];
      const updatedSessions = await Promise.all(affected.map((session) => api.updateSession(session.id, {
        metadata: { ...session.metadata, assistant_id: nextId, assistant_name: name },
      })));
      const updates = new Map(updatedSessions.map((session) => [session.id, session]));
      setSessions((current) => current.map((session) => updates.get(session.id) ?? session));
      setStoredPresets((current) => currentRole?.preset
        ? current.map((item) => item.id === currentRole.preset?.id ? saved : item)
        : [...current, saved]);
      setSelectedAssistantId(nextId);
      setModel(saved.model || model);
      setMode(saved.mode || mode);
      setRoleEditor(null);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  function openStudioPage(nextView: "dashboard" | "lab", page: DashboardPage | LabPage) {
    setView(nextView);
    if (nextView === "dashboard") setDashboardPage(page as DashboardPage);
    else setLabPage(page as LabPage);
    setSidebarOpen(false);
  }

  async function createSession() {
    const role = assistantRoles.find((candidate) => candidate.id === selectedAssistantId);
    const selectedModel = model.trim();
    if (!selectedModel) return;
    setError(null);
    try {
      const created = await api.createSession(selectedModel, mode, undefined, {
        assistant_id: role?.id ?? DEFAULT_ASSISTANT_ID,
        assistant_name: role?.name ?? tr("默认助手", "Default assistant"),
      });
      setSessions((current) => [created, ...current]);
      setActiveId(created.id);
      setMessages([]);
      setView("chat");
      setSidebarOpen(false);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function deleteSession(session: Session) {
    if (busy || !await studioConfirm(tr("删除这个会话？", "Delete this session?"))) return;
    try {
      await api.deleteSession(session.id);
      setVoiceMessages((current) => current.filter((item) => item.sessionId !== session.id));
      const next = await api.listSessions();
      setSessions(next);
      setActiveId(
        next.find((candidate) => sessionAssistantId(candidate) === selectedAssistantId)?.id ?? null,
      );
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function clearSessions() {
    if (busy || !assistantSessions.length || !await studioConfirm(tr("清空当前角色的全部会话？", "Clear all sessions for this role?"))) {
      return;
    }
    try {
      const removed = new Set(assistantSessions.map((session) => session.id));
      for (const session of assistantSessions) await api.deleteSession(session.id);
      setSessions((current) => current.filter((session) => !removed.has(session.id)));
      setMessages([]);
      setVoiceMessages((current) => current.filter((message) => !removed.has(message.sessionId)));
      setActiveId(null);
    } catch (cause) {
      setError(errorMessage(cause));
      await refreshSessions();
    }
  }

  async function saveRename(session: Session) {
    const title = renameValue.replace(/\s+/g, " ").trim();
    try {
      const updated = await api.updateSession(session.id, { title: title || null });
      setSessions((current) => current.map((item) => (item.id === updated.id ? updated : item)));
      setRenamingId(null);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  function applyFrame(frame: RealtimeFrame) {
    const payload = frame.payload;
    if (payload.type === "session.state") {
      setSessions((current) =>
        current.map((session) =>
          session.id === frame.session_id
            ? {
                ...session,
                state: payload.state as Session["state"],
                revision: payload.revision as number,
              }
            : session,
        ),
      );
      return;
    }
    if (payload.type === "response.reasoning.delta") {
      setLive((current) => ({
        reasoning: (current?.reasoning ?? "") + String(payload.delta ?? ""),
        text: current?.text ?? "",
        tools: current?.tools ?? [],
      }));
    } else if (payload.type === "response.text.delta") {
      setLive((current) => ({
        reasoning: current?.reasoning ?? "",
        text: (current?.text ?? "") + String(payload.delta ?? ""),
        tools: current?.tools ?? [],
      }));
    } else if (payload.type === "response.tool_call.delta") {
      setLive((current) => {
        const tools = [...(current?.tools ?? [])];
        const index = Number(payload.index);
        tools[index] = (tools[index] ?? "") + String(payload.arguments_delta ?? "");
        return {
          reasoning: current?.reasoning ?? "",
          text: current?.text ?? "",
          tools,
        };
      });
    }
  }

  async function generate(
    session: Session,
    input: ContentPart[],
    optimistic = true,
    inputRole: "user" | "tool" = "user",
  ) {
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setBusy(true);
    setLive({ reasoning: "", text: "", tools: [] });
    if (optimistic) {
      if (inputRole !== "user") throw new Error("Only user input can be optimistic");
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "user",
          parts: input,
          parent_id: current.at(-1)?.id ?? null,
          created_at: new Date().toISOString(),
        },
      ]);
    }
    try {
      await streamResponse(
        session.id,
        {
          request_id: crypto.randomUUID(),
          expected_revision: session.revision,
          input,
          input_role: inputRole,
          sampling: samplingParams(),
          system_prompt: effectiveSettings.systemPrompt || null,
          include_reasoning_history: !effectiveSettings.excludeReasoning,
          tools: mcpTools
            .filter((tool) => selectedTools.includes(tool.qualified_name))
            .map((tool) => ({
              type: "function" as const,
              function: {
                name: tool.qualified_name,
                description: tool.description,
                parameters: tool.input_schema,
              },
            })),
          tool_choice: selectedTools.length ? "auto" : "none",
          stream: true,
        },
        applyFrame,
        controller.signal,
      );
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(errorMessage(cause));
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
      setLive(null);
      try {
        const [persisted, persistedResponses, updated] = await Promise.all([
          api.listMessages(session.id),
          api.listResponses(session.id),
          api.getSession(session.id),
        ]);
        setMessages(persisted);
        setResponses(
          Object.fromEntries(
            persistedResponses
              .filter((response) => response.output_message_id)
              .map((response) => [response.output_message_id as string, response]),
          ),
        );
        await refreshSessions(updated.id);
      } catch (cause) {
        setError(errorMessage(cause));
      }
      void refreshRuntime(true);
    }
  }

  async function send(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!active || (!text && !attachments.length) || busy) return;
    if (
      active.mode !== "text" &&
      realtimeAvailable &&
      voiceRef.current &&
      attachments.length === 0
    ) {
      if (!text) return;
      setDraft("");
      await voiceRef.current.submitText(text, realtimeSessionConfig(active.id));
      setVoiceMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          sessionId: active.id,
          role: "user",
          text,
          created_at: new Date().toISOString(),
        },
      ]);
      return;
    }
    const realtimeController = voiceRef.current;
    const resumeCapture = Boolean(realtimeController?.capturing);
    if (active.mode !== "text" && realtimeController?.active) {
      await realtimeController.stop();
    }
    setBusy(true);
    setError(null);
    try {
      const uploaded = await Promise.all(
        attachments.map(async (attachment): Promise<ContentPart> => {
          if (attachment.kind === "document") {
            const uploaded = await api.uploadMedia(
              attachment.file,
              documentMimeType(attachment.file),
            );
            const document = await api.createDocument(uploaded.media.id, attachment.file.name);
            return { type: "document", media: document.media, name: document.name };
          }
          const [resource, metadata] = await Promise.all([
            api.uploadMedia(attachment.file),
            mediaMetadata(attachment.file, attachment.kind),
          ]);
          if (attachment.kind === "image") {
            const image = metadata as { width: number; height: number };
            return { type: "image", media: resource.media, ...image };
          }
          if (attachment.kind === "video") {
            const video = metadata as { width: number; height: number; duration_ms: number };
            return { type: "video", media: resource.media, ...video };
          }
          const audio = metadata as {
            sample_rate_hz: number;
            channels: number;
            duration_ms: number;
          };
          return { type: "audio", media: resource.media, ...audio };
        }),
      );
      const input: ContentPart[] = [...uploaded];
      if (text) input.push({ type: "text", text });
      setDraft("");
      setAttachments((current) => {
        current.forEach((attachment) => {
          if (attachment.previewUrl) URL.revokeObjectURL(attachment.previewUrl);
        });
        return [];
      });
      await generate(active, input);
    } catch (cause) {
      setBusy(false);
      setError(errorMessage(cause));
    } finally {
      if (resumeCapture && realtimeController && realtimeAvailable) {
        try {
          await realtimeController.start(realtimeSessionConfig(active.id));
        } catch (cause) {
          setError(errorMessage(cause));
        }
      }
    }
  }

  function selectAttachments(files: FileList | null) {
    if (!files) return;
    const next: PendingAttachment[] = [];
    for (const file of Array.from(files).slice(0, 8)) {
      const kind = file.type.startsWith("image/")
        ? "image"
        : file.type.startsWith("video/")
          ? "video"
          : file.type.startsWith("audio/")
            ? "audio"
            : isTextDocument(file)
              ? "document"
              : null;
      if (!kind) continue;
      if (kind === "document") {
        if (file.size > MAX_DOCUMENT_BYTES) {
          setError(tr("文档不能超过 64 MiB。", "Documents are limited to 64 MiB."));
          continue;
        }
      } else if (!attachmentAccept.includes(`${kind}/*`)) {
        continue;
      }
      next.push({
        id: crypto.randomUUID(),
        file,
        kind,
        previewUrl: kind === "document" ? "" : URL.createObjectURL(file),
      });
    }
    setAttachments((current) => {
      const combined = [...current, ...next];
      combined.slice(8).forEach((attachment) => {
        if (attachment.previewUrl) URL.revokeObjectURL(attachment.previewUrl);
      });
      return combined.slice(0, 8);
    });
    if (attachmentInputRef.current) attachmentInputRef.current.value = "";
  }

  function removeAttachment(id: string) {
    setAttachments((current) => {
      const removed = current.find((attachment) => attachment.id === id);
      if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
      return current.filter((attachment) => attachment.id !== id);
    });
  }

  async function saveEdit(message: Message) {
    if (!active || !editDraft || busy) return;
    const messageIndex = messages.findIndex((item) => item.id === message.id);
    if (messageIndex < 0) return;
    const text = editDraft.text.trim();
    const reasoning = editDraft.reasoning.trim();
    if (message.role === "user" && !text && !message.parts.some(isMediaPart)) return;
    if (message.role === "assistant" && !text && !reasoning) return;
    setBusy(true);
    try {
      const rewound = await api.rewindSession(
        active.id,
        active.revision,
        message.id,
        false,
      );
      const parts: ContentPart[] = [];
      if (reasoning) parts.push({ type: "reasoning", text: reasoning });
      if (text) parts.push({ type: "text", text });
      parts.push(...message.parts.filter(isMediaPart));
      const rewoundMessages = messages.slice(0, messageIndex);
      const visibleMessageIds = new Set(rewoundMessages.map((item) => item.id));
      setSessions((current) =>
        current.map((session) => session.id === rewound.id ? rewound : session),
      );
      setMessages(rewoundMessages);
      setResponses((current) =>
        Object.fromEntries(
          Object.entries(current).filter(([messageId]) => visibleMessageIds.has(messageId)),
        ),
      );
      setEditDraft(null);
      if (message.role === "user") {
        setMessages([...rewoundMessages, { ...message, parts }]);
        await generate(rewound, parts, false);
      } else {
        const appended = await api.appendMessage(
          rewound.id,
          rewound.revision,
          message.role,
          parts,
        );
        setSessions((current) =>
          current.map((session) => session.id === appended.session.id ? appended.session : session),
        );
        setMessages([...rewoundMessages, appended.message]);
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function regenerate(message: Message) {
    if (!active || busy || message.role !== "assistant") return;
    const index = messages.findIndex((item) => item.id === message.id);
    let user: Message | undefined;
    let userIndex = -1;
    for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
      if (messages[cursor].role === "user") {
        user = messages[cursor];
        userIndex = cursor;
        break;
      }
    }
    if (!user) return;
    if (!user.parts.length) return;
    setBusy(true);
    try {
      const rewound = await api.rewindSession(
        active.id,
        active.revision,
        user.id,
        false,
      );
      const rewoundMessages = messages.slice(0, userIndex + 1);
      const visibleMessageIds = new Set(rewoundMessages.map((item) => item.id));
      setSessions((current) =>
        current.map((session) => session.id === rewound.id ? rewound : session),
      );
      setMessages(rewoundMessages);
      setResponses((current) =>
        Object.fromEntries(
          Object.entries(current).filter(([messageId]) => visibleMessageIds.has(messageId)),
        ),
      );
      await generate(rewound, user.parts, false);
    } catch (cause) {
      setBusy(false);
      setError(errorMessage(cause));
    }
  }

  async function copyMessage(message: Message) {
    const parts = textParts(message);
    await navigator.clipboard.writeText([parts.reasoning, parts.text].filter(Boolean).join("\n\n"));
  }

  async function executeToolCalls(message: Message) {
    if (!active || busy) return;
    const calls = message.parts.filter(
      (part): part is Extract<ContentPart, { type: "tool_call" }> => part.type === "tool_call",
    );
    if (!calls.length) return;
    setBusy(true);
    setError(null);
    try {
      const results = await Promise.all(
        calls.map(async (call) => ({
          call,
          result: await api.callMcpTool(call.name, call.arguments),
        })),
      );
      let updated = await api.getSession(active.id);
      for (const item of results.slice(0, -1)) {
        const appended = await api.appendMessage(updated.id, updated.revision, "tool", [{
          type: "tool_result",
          call_id: item.call.call_id,
          result: item.result.structured_content ?? item.result.content,
          is_error: item.result.is_error,
        }]);
        updated = appended.session;
      }
      const last = results.at(-1)!;
      setBusy(false);
      await generate(
        updated,
        [{
          type: "tool_result",
          call_id: last.call.call_id,
          result: last.result.structured_content ?? last.result.content,
          is_error: last.result.is_error,
        }],
        false,
        "tool",
      );
    } catch (cause) {
      setBusy(false);
      setError(errorMessage(cause));
    }
  }

  async function createMcpServer(event: FormEvent) {
    event.preventDefault();
    if (!mcpDraft.name.trim() || !mcpDraft.endpoint.trim() || busy) return;
    setBusy(true);
    try {
      await api.createMcpServer({
        name: mcpDraft.name.trim(),
        transport: mcpDraft.transport,
        enabled: true,
        url: mcpDraft.transport === "streamable_http" ? mcpDraft.endpoint.trim() : null,
        command: mcpDraft.transport === "stdio" ? mcpDraft.endpoint.trim() : null,
      });
      setMcpDraft({ name: "", transport: "streamable_http", endpoint: "" });
      await Promise.all([refreshRuntime(false), refreshMcp()]);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function toggleMcpServer(server: McpServerResource) {
    try {
      await api.updateMcpServer(server.id, !server.enabled);
      await refreshMcp();
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function deleteMcpServer(id: string) {
    try {
      await api.deleteMcpServer(id);
      await refreshMcp();
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  function exportConversation() {
    if (!active) return;
    const lines = [`# ${active.title || tr("未命名会话", "Untitled session")}`, ""];
    for (const message of messages) {
      const parts = textParts(message);
      lines.push(`## ${message.role === "assistant" ? "MFQ" : message.role}`, "");
      if (parts.reasoning) {
        lines.push("<details>", `<summary>${tr("思考过程", "Reasoning")}</summary>`, "", parts.reasoning, "", "</details>", "");
      }
      if (parts.text) lines.push(parts.text, "");
    }
    for (const message of currentVoiceMessages) {
      lines.push(`## ${message.role === "assistant" ? "MFQ" : tr("用户", "User")}`, "", message.text, "");
    }
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${(active.title || "MFQ").replace(/[\\/:*?"<>|]/g, "_")}.md`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function toggleVoice() {
    if (
      !active ||
      active.mode === "text" ||
      !realtimeAvailable ||
      busy ||
      !voiceRef.current
    ) return;
    await voiceRef.current.toggleCapture(realtimeSessionConfig(active.id));
  }

  async function selectInteractionMode(nextMode: SessionMode) {
    setMode(nextMode);
    if (!active || active.mode === nextMode || busy) return;
    if (voiceRef.current?.active) await voiceRef.current.stop();
    try {
      const updated = await api.updateSession(active.id, { mode: nextMode });
      appliedModeTemplate.current = "";
      setSessions((current) =>
        current.map((session) => (session.id === updated.id ? updated : session)),
      );
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  function openSettings() {
    setSettingsDraft(resolvedGlobalSettings);
    const current = Number(runtime?.max_context);
    setContextSize(Number.isFinite(current) && current > 0 ? current : contextSize);
    setSettingsOpen(true);
  }

  useEffect(() => {
    function handleKeydown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setSettingsOpen(false);
        setStudioOpen(false);
        setSidebarOpen(false);
      } else if ((event.metaKey || event.ctrlKey) && event.key === ",") {
        event.preventDefault();
        openSettings();
      } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "n") {
        event.preventDefault();
        void createSession();
      }
    }
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  });

  function applyPreset(name: Exclude<PresetName, "custom">) {
    setSettingsDraft((current) => ({
      ...current,
      ...PRESETS[name],
      inheritModelDefaults: false,
      preset: name,
    }));
    setSelectedStoredPreset("");
    setStoredPresetName("");
    setPresetStatus(null);
  }

  function loadStoredPreset(name: string) {
    setSelectedStoredPreset(name);
    setPresetStatus(null);
    if (!name) {
      setStoredPresetName("");
      return;
    }
    const stored = storedPresets.find((preset) => preset.name === name);
    if (!stored) return;
    setStoredPresetName(stored.name);
    setSettingsDraft((current) => ({
      ...current,
      ...stored.settings,
      inheritModelDefaults: false,
      preset: "custom",
    }));
    setContextSize(stored.contextSize);
    setPresetStatus({ error: false, text: tr("预设已载入。", "Preset loaded.") });
  }

  async function saveStoredPreset() {
    const name = storedPresetName.replace(/\s+/g, " ").trim().slice(0, 64);
    if (!name) {
      setPresetStatus({ error: true, text: tr("请输入预设名称。", "Enter a preset name.") });
      return;
    }
    const next: StoredPreset = {
      name,
      settings: presetSnapshot(settingsDraft),
      inheritGlobalSettings: false,
      contextSize,
      model: active?.model ?? model,
      mode: active?.mode ?? mode,
      updatedAt: new Date().toISOString(),
    };
    const selected = storedPresets.find((preset) => preset.name === selectedStoredPreset);
    const body = {
      name,
      model: next.model,
      mode: next.mode,
      settings: {
        sampling: {
          max_tokens: next.settings.maxTokens,
          temperature: next.settings.temperature,
          top_k: next.settings.topK,
          top_p: next.settings.topP,
          presence_penalty: next.settings.presencePenalty,
          frequency_penalty: next.settings.frequencyPenalty,
          repetition_penalty: next.settings.repetitionPenalty,
          seed: next.settings.seed,
          enable_thinking: thinkingSupported && next.settings.enableThinking,
          reasoning_effort: next.settings.reasoningEffort || null,
        },
        system_prompt: next.settings.systemPrompt || null,
        include_reasoning_history: !next.settings.excludeReasoning,
        input_role: "user" as const,
        tools: [],
        tool_choice: "auto" as const,
        response_format: { type: "text" as const },
      },
      context_size: next.contextSize,
      metadata: {
        ...(selected?.icon ? { icon: selected.icon } : {}),
        inherit_global_settings: false,
      },
    };
    try {
      const saved = selected?.id
        ? await api.updateGenerationPreset(selected.id, body)
        : await api.createGenerationPreset(body);
      const stored = storedPresetFromResource(saved);
      setStoredPresets((current) => {
        const index = current.findIndex(
          (preset) => preset.id === stored.id || preset.name === selectedStoredPreset,
        );
        if (index < 0) return [...current, stored].slice(-50);
        return current.map((preset, cursor) => cursor === index ? stored : preset);
      });
      setSelectedStoredPreset(stored.name);
      setStoredPresetName(stored.name);
      setPresetStatus({ error: false, text: tr("预设已保存。", "Preset saved.") });
    } catch (cause) {
      setPresetStatus({ error: true, text: errorMessage(cause) });
    }
  }

  async function deleteStoredPreset() {
    if (!selectedStoredPreset) return;
    if (!await studioConfirm(tr(`删除预设“${selectedStoredPreset}”？`, `Delete preset “${selectedStoredPreset}”?`))) return;
    const selected = storedPresets.find((preset) => preset.name === selectedStoredPreset);
    try {
      if (selected?.id) await api.deleteGenerationPreset(selected.id);
      setStoredPresets((current) =>
        current.filter((preset) => preset.name !== selectedStoredPreset),
      );
      setSelectedStoredPreset("");
      setStoredPresetName("");
      setPresetStatus({ error: false, text: tr("预设已删除。", "Preset deleted.") });
    } catch (cause) {
      setPresetStatus({ error: true, text: errorMessage(cause) });
    }
  }

  function renderPresetManager(disabled = false) {
    return (
      <div className="saved-presets">
        <label>
          <span>{tr("已保存预设", "Saved presets")}</span>
          <select
            disabled={disabled}
            onChange={(event) => loadStoredPreset(event.target.value)}
            value={selectedStoredPreset}
          >
            <option value="">
              {storedPresets.length
                ? tr("选择预设…", "Select a preset…")
                : tr("还没有保存的预设", "No saved presets")}
            </option>
            {storedPresets.map((preset) => (
              <option key={preset.name} value={preset.name}>{preset.name}</option>
            ))}
          </select>
        </label>
        <div className="preset-save-row">
          <input
            aria-label={tr("预设名称", "Preset name")}
            disabled={disabled}
            maxLength={64}
            onChange={(event) => {
              setStoredPresetName(event.target.value);
              setPresetStatus(null);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void saveStoredPreset();
              }
            }}
            placeholder={tr("预设名称", "Preset name")}
            value={storedPresetName}
          />
          <button disabled={disabled} onClick={() => void saveStoredPreset()} type="button">
            {selectedStoredPreset && storedPresetName.trim() === selectedStoredPreset
              ? tr("覆盖", "Update")
              : tr("保存", "Save")}
          </button>
          <button
            aria-label={tr("删除预设", "Delete preset")}
            className="preset-delete"
            disabled={disabled || !selectedStoredPreset}
            onClick={() => void deleteStoredPreset()}
            title={tr("删除预设", "Delete preset")}
            type="button"
          >
            <Icon name="trash" size={14} />
          </button>
        </div>
        {presetStatus && (
          <p className={presetStatus.error ? "preset-status error" : "preset-status"}>
            {presetStatus.text}
          </p>
        )}
        <small>
          {tr(
            "保存系统提示词、上下文和生成参数；不包含界面语言与播放开关。",
            "Stores the system prompt, context, and generation parameters; interface and playback preferences stay separate.",
          )}
        </small>
      </div>
    );
  }

  const presetManager = renderPresetManager(settingsDraft.inheritModelDefaults);

  function setModelDefaultInheritance(enabled: boolean) {
    setSettingsDraft((current) => ({
      ...(enabled
        ? modeTemplateSettings(current, active?.mode ?? mode, runtime, realtime)
        : current),
      inheritModelDefaults: enabled,
    }));
    if (enabled) {
      setSelectedStoredPreset("");
      setStoredPresetName("");
      setPresetStatus(null);
    }
  }

  function saveSettings() {
    setSettings(settingsDraft);
    setSettingsOpen(false);
  }

  function setUiTheme(theme: UiTheme) {
    setSettings((current) => ({ ...current, theme }));
    setSettingsDraft((current) => ({ ...current, theme }));
  }

  function updateGlobalInference(patch: Partial<GenerationSettings>) {
    setSettings((current) => ({
      ...(current.inheritModelDefaults
        ? modeTemplateSettings(current, active?.mode ?? mode, runtime, realtime)
        : current),
      ...patch,
      inheritModelDefaults: false,
      preset: "custom",
    }));
  }

  async function reloadRuntime() {
    if (busy || voiceRef.current?.active) return;
    if (!await studioConfirm(tr(`以 ${formatNumber(contextSize)} token 上下文重载模型？`, `Reload the model with a ${formatNumber(contextSize)} token context?`))) return;
    setBusy(true);
    try {
      const status = await api.reloadRuntime(contextSize);
      setRuntime(status);
      setSettingsOpen(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function clearRuntimeCache() {
    const snapshots = Number(runtime?.prefix_cache_snapshots || 0);
    if (busy || snapshots <= 0 || Number(runtime?.active_requests || 0) > 0) return;
    if (!await studioConfirm(tr(
      `清除 ${formatNumber(snapshots)} 个 prefix cache 快照？`,
      `Clear ${formatNumber(snapshots)} prefix-cache snapshots?`,
    ))) return;
    setBusy(true);
    try {
      const status = await api.clearRuntimeCache();
      setRuntime((current) => current ? { ...current, ...status } : status);
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  function openStudioSettings() {
    if (!studio) return;
    setStudioDraft({ ...studio.config });
    setStudioOpen(true);
  }

  async function saveStudioSettings(event: FormEvent) {
    event.preventDefault();
    if (!studioDraft) return;
    setBusy(true);
    try {
      const status = await configureStudio(studioDraft);
      await saveStudioCredential(studioToken);
      setApiBaseUrl(status.service_url);
      setApiToken(studioToken);
      setStudio(status);
      setStudioOpen(false);
      setMessages([]);
      setActiveId(null);
      await refreshSessions();
      appliedModeTemplate.current = "";
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  function updateJobPayload(name: string, property: JsonSchemaProperty, value: string | boolean) {
    const type = schemaType(property);
    let parsed: unknown = value;
    if (type === "integer" || type === "number") {
      parsed = value === "" ? null : Number(value);
    } else if (type === "array") {
      parsed = String(value)
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => (property.items?.type === "integer" ? Number(item) : item));
    }
    setJobPayload((current) => ({ ...current, [name]: parsed }));
  }

  async function submitJob(event: FormEvent) {
    event.preventDefault();
    if (!selectedJobKind) return;
    setBusy(true);
    try {
      const clean = Object.fromEntries(
        Object.entries(jobPayload).filter(([, value]) => value !== "" && value !== null),
      );
      const created = await api.createJob(selectedJobKind, clean);
      setSelectedJobId(created.id);
      setJobs((current) => [created, ...current]);
      setView("lab");
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function importImatrix(files: FileList | null) {
    const file = files?.[0];
    if (!file || imatrixImporting || busy) return;
    setImatrixImporting(true);
    try {
      const uploaded = await api.uploadMedia(file, "application/x-mfq-imatrix");
      const destination = `artifacts/imatrix/${file.name.replace(/[^A-Za-z0-9_.-]+/g, "-")}`;
      const created = await api.createJob("artifact.import", {
        media_id: uploaded.media.id,
        destination,
        kind: "imatrix",
      });
      setSelectedJobId(created.id);
      setJobs((current) => [created, ...current]);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setImatrixImporting(false);
      if (imatrixInputRef.current) imatrixInputRef.current.value = "";
    }
  }


  async function cancelSelectedJob() {
    if (!selectedJobId) return;
    try {
      const updated = await api.cancelJob(selectedJobId);
      setJobs((current) => current.map((item) => (item.id === updated.id ? updated : item)));
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function retrySelectedJob() {
    if (!selectedJob || busy) return;
    setBusy(true);
    try {
      const retried = await api.retryJob(selectedJob.id);
      setJobs((current) => [retried, ...current]);
      setSelectedJobId(retried.id);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function deleteJobRecord(id: string) {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      await api.deleteJob(id);
      setJobs((current) => current.filter((item) => item.id !== id));
      if (selectedJobId === id) {
        setSelectedJobId(null);
        setJobLogs([]);
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setJobCleanupBusy(false);
    }
  }

  async function clearCompletedJobRecords() {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      await api.clearCompletedJobs();
      setJobs((current) => current.filter((item) => !isTerminalJob(item)));
      if (selectedJob && isTerminalJob(selectedJob)) {
        setSelectedJobId(null);
        setJobLogs([]);
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setJobCleanupBusy(false);
    }
  }

  async function searchHub(event: FormEvent) {
    event.preventDefault();
    const query = hubQuery.trim();
    if (!query || busy) return;
    setBusy(true);
    try {
      const results = await api.searchHubModels(hubProvider, query);
      setHubResults(results);
      setHubModel(null);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function inspectHubModel(item: HubModelSummary) {
    if (busy) return;
    setBusy(true);
    try {
      setHubModel(await api.hubModelInfo(item.provider, item.repo_id));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function downloadHubModel() {
    if (!hubModel || busy) return;
    const name = hubModel.repo_id.split("/").pop()?.replace(/[^A-Za-z0-9_.-]/g, "-") || "model";
    setBusy(true);
    try {
      const created = await api.createJob(
        `download.${hubModel.provider}`,
        {
          repo_id: hubModel.repo_id,
          destination: `downloads/${name}`,
          revision: hubModel.revision,
          expected_bytes: hubModel.total_bytes || null,
        },
      );
      setJobs((current) => [created, ...current]);
      setSelectedJobId(created.id);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function removeSelectedArtifact() {
    const uri = String(selectedJob?.result?.artifact || "");
    if (!uri.startsWith("workspace://") || busy) return;
    if (!await studioConfirm(tr("删除这个本地产物？", "Delete this local artifact?"))) return;
    setBusy(true);
    try {
      await api.removeWorkspaceArtifact(uri);
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function registerDataset(event: FormEvent) {
    event.preventDefault();
    if (busy || !datasetDraft.name.trim() || !datasetDraft.artifact_uri.trim()) return;
    setBusy(true);
    try {
      await api.createDataset({
        name: datasetDraft.name.trim(),
        kind: datasetDraft.kind,
        artifact_uri: datasetDraft.artifact_uri.trim(),
      });
      setDatasetDraft({ name: "", artifact_uri: "", kind: "custom" });
      setDatasets(await api.datasets());
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function registerRemoteNode(event: FormEvent) {
    event.preventDefault();
    if (busy || !nodeDraft.name.trim() || !nodeDraft.url.trim()) return;
    setBusy(true);
    try {
      await api.createRemoteNode({
        name: nodeDraft.name.trim(),
        url: nodeDraft.url.trim(),
        api_key_env: nodeDraft.api_key_env.trim() || null,
        enabled: true,
      });
      setNodeDraft({ name: "", url: "", api_key_env: "" });
      setRemoteNodes(await api.remoteNodes(true));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function compareSelectedEvaluations() {
    if (selectedEvaluations.length < 2 || busy) return;
    setBusy(true);
    try {
      setEvaluationComparison(await api.compareEvaluations(selectedEvaluations));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function exportStudioData() {
    try {
      const sessionArchives = await Promise.all(sessions.map((session) => api.exportSession(session.id)));
      const payload = {
        format: "mfq-studio-export-v2",
        exported_at: new Date().toISOString(),
        sessions: sessionArchives,
        presets: storedPresets,
      };
      const anchor = document.createElement("a");
      anchor.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
      anchor.download = `mfq-studio-${new Date().toISOString().slice(0, 10)}.json`;
      anchor.click();
      URL.revokeObjectURL(anchor.href);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function importStudioData(file: File) {
    try {
      const payload = JSON.parse(await file.text()) as { format?: string; presets?: StoredPreset[]; sessions?: SessionArchive[] };
      if (payload.format !== "mfq-studio-export-v2" || !Array.isArray(payload.presets) || !Array.isArray(payload.sessions)) {
        throw new Error(tr("不是有效的 MFQ Studio 导出文件。", "Not a valid MFQ Studio export."));
      }
      for (const preset of payload.presets) {
        if (!preset.name || !preset.settings || !Number.isFinite(preset.contextSize)) continue;
        const existing = storedPresets.find((item) => item.name === preset.name);
        const resource = presetResourceBody({
          ...preset,
          id: existing?.id,
          inheritGlobalSettings: preset.inheritGlobalSettings !== false,
        }, model, mode);
        if (existing?.id) await api.updateGenerationPreset(existing.id, resource);
        else await api.createGenerationPreset(resource);
      }
      for (const archive of payload.sessions) await api.importSession(archive);
      const next = (await api.generationPresets()).map(storedPresetFromResource);
      setStoredPresets(next);
      await refreshSessions();
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function loadArtifact(name: string) {
    if (busy) return;
    setBusy(true);
    try {
      const accepted = await api.loadModel(name, contextSize);
      setSelectedJobId(accepted.operation_id);
      setView("lab");
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function finishModelRegistration(names: string[]) {
    const nextArtifacts = await api.modelArtifacts(true);
    setArtifacts(nextArtifacts);
    const registered = nextArtifacts.filter((item) => names.includes(item.name));
    if (!registered.length) {
      throw new Error(tr("所选目录中的模型没有出现在模型目录中。", "Models from the selected folder were not registered in the catalog."));
    }
    if (registered.length === 1) {
      const artifact = registered[0];
      if (!artifact.loadable) {
        throw new Error(artifact.error || tr("所选模型不完整或无法加载。", "The selected model is incomplete or cannot be loaded."));
      }
      const loaded = instances.some((item) => item.model === artifact.name && item.state !== "failed")
        || runtime?.model === artifact.name;
      if (!loaded) {
        const accepted = await api.loadModel(artifact.name, contextSize);
        setSelectedJobId(accepted.operation_id);
      }
    }
    setModelBrowserOpen(false);
    setDashboardPage("models");
    setView("dashboard");
    await refreshRuntime(false);
  }

  async function openModelDirectory(directoryId?: string | null, path?: string | null) {
    if (busy) return;
    setBusy(true);
    try {
      const listing = await api.modelDirectories(directoryId, path);
      setModelBrowser(listing);
      setModelDirectoryPath(listing.current_path ?? "");
      setModelBrowserOpen(true);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function jumpToModelDirectory(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const path = modelDirectoryPath.trim();
    if (!path) return;
    await openModelDirectory(null, path);
  }

  async function chooseModelDirectory() {
    if (busy) return;
    if (!canUseNativeModelPicker) {
      await openModelDirectory();
      return;
    }
    setBusy(true);
    try {
      const names = await selectLocalModelDirectory();
      if (names) await finishModelRegistration(names);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function registerCurrentModelDirectory() {
    if (busy || !modelBrowser?.current_id) return;
    setBusy(true);
    try {
      const registered = await api.registerModelDirectory(modelBrowser.current_id);
      await finishModelRegistration(registered.map((item) => item.name));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function saveRuntimeProfile() {
    const name = profileName.replace(/\s+/g, " ").trim();
    const artifact = artifacts.find((item) => item.name === runtime?.model);
    if (busy || !name || !artifact) return;
    setBusy(true);
    try {
      await api.createRuntimeProfile({
        name,
        load: {
          model: artifact.name,
          device_ids: [],
          pin: false,
          context_size: contextSize,
          prefill_chunk_size: 2048,
          sampling_defaults: {
            max_tokens: resolvedGlobalSettings.maxTokens,
            temperature: resolvedGlobalSettings.temperature,
            top_k: resolvedGlobalSettings.topK,
            top_p: resolvedGlobalSettings.topP,
            presence_penalty: resolvedGlobalSettings.presencePenalty,
            frequency_penalty: resolvedGlobalSettings.frequencyPenalty,
            repetition_penalty: resolvedGlobalSettings.repetitionPenalty,
            seed: resolvedGlobalSettings.seed,
            enable_thinking: thinkingSupported && resolvedGlobalSettings.enableThinking,
            reasoning_effort: resolvedGlobalSettings.reasoningEffort || null,
          },
        },
      });
      setProfileName("");
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function loadRuntimeProfile(profile: RuntimeProfile) {
    if (busy) return;
    if (profile.drifted && !await studioConfirm(tr(
      "模型产物已变化。仍使用这个配置档案加载？",
      "The model artifact changed. Load this profile anyway?",
    ))) return;
    setBusy(true);
    try {
      const accepted = await api.loadRuntimeProfile(profile.id, profile.drifted);
      setSelectedJobId(accepted.operation_id);
      setView("lab");
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function deleteRuntimeProfile(id: string) {
    if (busy) return;
    setBusy(true);
    try {
      await api.deleteRuntimeProfile(id);
      setRuntimeProfiles((current) => current.filter((item) => item.id !== id));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function unloadInstance(id: string) {
    if (busy) return;
    setBusy(true);
    try {
      const accepted = await api.unloadModel(id);
      setSelectedJobId(accepted.operation_id);
      setView("lab");
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  const last = runtime?.last_request;
  const lastPrefill = displayPrefillMetric(last);
  const lastTtftMs = preferPositiveMetric(last?.ttft_ms, last?.complete_prefill_ms);
  const lastGenerationMs = preferPositiveMetric(
    last?.generation_ms,
    last?.complete_generation_ms,
  );
  const contextTokens = Number(last?.prompt_tokens || 0) + Number(last?.completion_tokens || 0);
  const activeJobs = jobs.filter((job) =>
    ["queued", "running", "cancelling"].includes(job.status),
  );
  const completedJobs = jobs.filter(isTerminalJob);
  const runtimeMemory = Number(
    runtime?.mlx_active_bytes ??
      runtime?.cuda_allocated_bytes ??
      runtime?.process_resident_bytes ??
      0,
  );
  const runtimeCache = Number(runtime?.mlx_cache_bytes ?? runtime?.cuda_reserved_bytes ?? 0);
  const prefixCacheQueries = Number(runtime?.prefix_cache_queries || 0);
  const prefixCacheHits = Number(runtime?.prefix_cache_hits || 0);
  const prefixCacheSnapshots = Number(runtime?.prefix_cache_snapshots || 0);
  const prefixCacheBytes = Number(runtime?.prefix_cache_bytes || 0);
  const prefixCacheHitRate = prefixCacheQueries > 0
    ? (prefixCacheHits / prefixCacheQueries) * 100
    : 0;
  const prefixCacheSupported = typeof runtime?.prefix_cache_max_bytes === "number";
  const genericJobKinds = jobKinds;
  const selectedKind = genericJobKinds.find((item) => item.kind === selectedJobKind);
  const imatrixArtifacts = lineage.filter((item) =>
    item.producer_kind === "calibrate.imatrix"
      || item.producer_kind === "artifact.import"
      || item.metadata?.media_type === "application/x-mfq-imatrix"
      || item.artifact_name.endsWith(".imatrix")
  );
  const selectedJob = jobs.find((item) => item.id === selectedJobId) ?? null;
  const clusterPanel = (
    <section className="dashboard-panel cluster-panel">
      <div className="panel-heading">
        <div><h2>{tr("远程节点", "Remote nodes")}</h2><p>{tr("按模型和负载路由到健康的 MFQ Server", "Route by model and load across healthy MFQ Server nodes")}</p></div>
        <b>{remoteNodes.filter((node) => node.healthy).length} / {remoteNodes.length}</b>
      </div>
      <form className="node-form" onSubmit={registerRemoteNode}>
        <input onChange={(event) => setNodeDraft((current) => ({ ...current, name: event.target.value }))} placeholder={tr("节点名称", "Node name")} value={nodeDraft.name} />
        <input onChange={(event) => setNodeDraft((current) => ({ ...current, url: event.target.value }))} placeholder="https://worker.example" type="url" value={nodeDraft.url} />
        <input onChange={(event) => setNodeDraft((current) => ({ ...current, api_key_env: event.target.value }))} placeholder={tr("密钥环境变量（可选）", "Credential environment variable (optional)")} value={nodeDraft.api_key_env} />
        <button disabled={busy} type="submit">{tr("添加", "Add")}</button>
      </form>
      <div className="node-list">{remoteNodes.map((node) => <div key={node.id}><span className={node.healthy ? "model-state active" : "model-state failed"} /><div><strong>{node.name}</strong><small>{node.url} · {node.models.length} models · {node.active_requests} active{typeof node.metrics.total_requests === "number" ? ` · ${formatNumber(node.metrics.total_requests)} requests` : ""}{node.error ? ` · ${node.error}` : ""}</small></div><button onClick={() => void api.deleteRemoteNode(node.id).then(() => setRemoteNodes((current) => current.filter((item) => item.id !== node.id))).catch((cause) => setError(errorMessage(cause)))} type="button">{tr("删除", "Delete")}</button></div>)}</div>
    </section>
  );

  const renderMarkdown = (text: string, live = false, normalizeEscapedLineBreaks = false) => (
    <Suspense fallback={<p className="rich-text">{text}</p>}>
      <Markdown live={live} normalizeEscapedLineBreaks={normalizeEscapedLineBreaks} text={text} />
    </Suspense>
  );

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`} id="studio-sidebar">
        <div className="brand">
          <img src="/mfq-mark.svg" alt="" />
          <div><strong>MFQ</strong><span>Studio</span></div>
        </div>
        <div className="sidebar-divider" />
        <nav className="primary-nav">
          <button className={view === "chat" ? "active" : ""} onClick={() => { setView("chat"); setSidebarOpen(false); }} type="button"><Icon name="chat" />{tr("对话", "Chat")}</button>
          <button className={view === "dashboard" ? "active" : ""} onClick={() => { setView("dashboard"); setSidebarOpen(false); }} type="button"><Icon name="activity" />{tr("仪表盘", "Dashboard")}<span>{formatNumber(runtime?.active_requests || 0)}</span></button>
          <button className={view === "lab" ? "active" : ""} onClick={() => { setView("lab"); setSidebarOpen(false); }} type="button"><Icon name="flask" />{tr("实验室", "Lab")}</button>
        </nav>
        <div className="sidebar-divider" />
        <div className={`chat-sidebar-section ${view === "chat" ? "visible" : ""}`}>
        <div className="assistant-heading"><span>{tr("角色", "Roles")}</span><small>{assistantRoles.length}</small></div>
        <nav className="assistant-list" aria-label={tr("角色", "Roles")}>
          {assistantRoles.map((role) => {
            const count = sessions.filter((session) => canonicalSessionAssistantId(session, storedPresets) === role.id).length;
            return <div className={`assistant-row ${selectedAssistantId === role.id ? "active" : ""}`} key={role.id}><button className="assistant-main" onClick={() => selectAssistant(role)} type="button"><span>{role.preset?.icon || role.name.slice(0, 1).toLocaleUpperCase()}</span><div><strong>{role.name}</strong><small>{count} {tr("个会话", "sessions")}</small></div></button><button aria-label={tr("编辑角色", "Edit role")} className="assistant-edit" onClick={() => editRole(role)} title={tr("编辑角色", "Edit role")} type="button"><Icon name="edit" size={12} /></button></div>;
          })}
        </nav>
        <button className="new-session" onClick={createRole} type="button"><Icon name="plus" size={14} />{tr("新角色", "New role")}</button>
        <details className="session-create" open={!model}>
          <summary>{tr("新会话默认设置", "New session defaults")}</summary>
          <button className="open-local-model" disabled={busy} onClick={() => void chooseModelDirectory()} type="button"><Icon name="folder" size={14} />{tr("打开模型文件夹", "Open model folder")}</button>
          <label htmlFor="new-model">{tr("模型", "Model")}</label>
          <select id="new-model" value={model} onChange={(event) => setModel(event.target.value)}>
            {!models.length && <option value="">{tr("尚未加载模型", "No model loaded")}</option>}
            {models.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
          </select>
          <div className="mode-picker">
            {(["text", "voice", "full_duplex"] as SessionMode[]).map((item) => {
              const feature = capabilities?.model_capabilities.features;
              const disabled = item === "voice" ? !feature?.audio_input : item === "full_duplex" ? !feature?.full_duplex : false;
              return <button aria-pressed={mode === item} disabled={disabled} key={item} onClick={() => setMode(item)} type="button">{MODE_LABELS[item][english ? 1 : 0]}</button>;
            })}
          </div>
        </details>
        <div className="history-heading"><span>{tr("会话", "Sessions")}</span><div><button aria-label={tr("新会话", "New session")} disabled={!model} onClick={() => void createSession()} title={model ? tr("新会话", "New session") : tr("请先加载模型", "Load a model first")} type="button"><Icon name="plus" size={14} /></button><button aria-label={tr("清空当前角色会话", "Clear role sessions")} onClick={clearSessions} title={tr("清空当前角色会话", "Clear role sessions")} type="button"><Icon name="trash" size={14} /></button></div></div>
        <nav className="session-list" aria-label="Sessions">
          {loading && <span className="empty-note">{tr("加载中…", "Loading…")}</span>}
          {!loading && assistantSessions.length === 0 && <span className="empty-note">{tr("这个角色还没有会话", "No sessions for this role")}</span>}
          {assistantSessions.map((session) => (
            <div className={`session-row ${activeId === session.id ? "active" : ""}`} key={session.id}>
              {renamingId === session.id ? (
                <input autoFocus onBlur={() => void saveRename(session)} onChange={(event) => setRenameValue(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void saveRename(session); if (event.key === "Escape") setRenamingId(null); }} value={renameValue} />
              ) : (
                <button className="session-main" onClick={() => { setActiveId(session.id); setView("chat"); setSidebarOpen(false); }} type="button"><span>{session.title || tr("未命名会话", "Untitled session")}</span><small>{MODE_LABELS[session.mode][english ? 1 : 0]} · r{session.revision}</small></button>
              )}
              <div className="session-actions">
                <button aria-label={tr("重命名", "Rename")} onClick={() => { setRenamingId(session.id); setRenameValue(session.title || ""); }} title={tr("重命名", "Rename")} type="button"><Icon name="edit" size={13} /></button>
                <button aria-label={tr("删除", "Delete")} onClick={() => void deleteSession(session)} title={tr("删除", "Delete")} type="button"><Icon name="trash" size={13} /></button>
              </div>
            </div>
          ))}
        </nav>
        </div>
        <div className={`context-sidebar-section ${view === "dashboard" ? "visible" : ""}`}>
          <div className="context-sidebar-heading"><strong>{tr("仪表盘", "Dashboard")}</strong><small>{tr("运行与服务", "Runtime and service")}</small></div>
          <nav className="context-nav">
            <button className={dashboardPage === "overview" ? "active" : ""} onClick={() => openStudioPage("dashboard", "overview")} type="button"><Icon name="activity" size={14} />{tr("概览", "Overview")}</button>
            <button className={dashboardPage === "cache" ? "active" : ""} onClick={() => openStudioPage("dashboard", "cache")} type="button">{tr("缓存与配置", "Cache and profiles")}</button>
            <button className={dashboardPage === "models" ? "active" : ""} onClick={() => openStudioPage("dashboard", "models")} type="button">{tr("模型与任务", "Models and jobs")}</button>
            <button className={dashboardPage === "connections" ? "active" : ""} onClick={() => openStudioPage("dashboard", "connections")} type="button">{tr("工具与连接", "Tools and connections")}</button>
          </nav>
        </div>
        <div className={`context-sidebar-section ${view === "lab" ? "visible" : ""}`}>
          <div className="context-sidebar-heading"><strong>{tr("实验室", "Lab")}</strong><small>{tr("数据与量化工作流", "Data and quantization")}</small></div>
          <nav className="context-nav">
            <button className={labPage === "models" ? "active" : ""} onClick={() => openStudioPage("lab", "models")} type="button">{tr("模型仓库", "Model hubs")}</button>
            <button className={labPage === "evaluations" ? "active" : ""} onClick={() => openStudioPage("lab", "evaluations")} type="button">{tr("评测与数据集", "Evaluation and datasets")}</button>
            <button className={labPage === "quantization" ? "active" : ""} onClick={() => openStudioPage("lab", "quantization")} type="button">{tr("量化工作台", "Quantization workspace")}</button>
          </nav>
        </div>
      </aside>
      <button aria-label={tr("关闭侧栏", "Close sidebar")} className={`mobile-scrim ${sidebarOpen ? "open" : ""}`} onClick={() => setSidebarOpen(false)} type="button" />

      <main className="workspace">
        <header className="topbar">
          <div className="topbar-identity">
            <button aria-controls="studio-sidebar" aria-expanded={sidebarOpen} aria-label={tr("打开侧栏", "Open sidebar")} className="sidebar-toggle" onClick={() => setSidebarOpen(true)} type="button"><Icon name="menu" /></button>
            <div className="topbar-model"><span>{active?.model || model || tr("尚未加载模型", "No model loaded")}</span><small>{capabilities?.model_type || runtime?.model_type || "runtime"}</small></div>
          </div>
          <div className="topbar-actions">
            {capabilities && <div className="capabilities">{CAPABILITY_LABELS.filter(([feature]) => capabilities.model_capabilities.features[feature]).map(([feature, label]) => <span className={feature === "full_duplex" && !realtimeAvailable ? "muted" : ""} key={feature}>{label[english ? 1 : 0]}</span>)}</div>}
            <div className="quick-metrics"><span><b>{lastTtftMs == null ? "--" : `${formatNumber(lastTtftMs, 1)} ms`}</b> TTFT</span><span><b>{last ? formatNumber(contextTokens) : "--"}</b> context</span><span><b>{last?.decode_tps == null ? "--" : formatNumber(last.decode_tps, 1)}</b> tok/s</span></div>
            <div aria-label={tr("外观", "Appearance")} className="theme-switcher" role="group">
              <button aria-label={tr("自动主题", "Auto theme")} aria-pressed={settings.theme === "system"} onClick={() => setUiTheme("system")} title={tr("跟随系统", "Auto")} type="button"><Icon name="sun-moon" size={13} /><span>Auto</span></button>
              <button aria-label={tr("浅色主题", "Light theme")} aria-pressed={settings.theme === "light"} onClick={() => setUiTheme("light")} title={tr("浅色", "Light")} type="button"><Icon name="sun" size={13} /><span>Light</span></button>
              <button aria-label={tr("深色主题", "Dark theme")} aria-pressed={settings.theme === "dark"} onClick={() => setUiTheme("dark")} title={tr("深色", "Dark")} type="button"><Icon name="moon" size={13} /><span>Dark</span></button>
            </div>
            <button aria-label={tr("打开模型文件夹", "Open model folder")} disabled={busy} onClick={() => void chooseModelDirectory()} title={tr("选择包含 MFQ 模型的文件夹", "Choose a folder containing MFQ models")} type="button"><Icon name="folder" /></button>
            <button aria-label={tr("导出会话", "Export chat")} disabled={!active} onClick={exportConversation} title={tr("导出会话", "Export chat")} type="button"><Icon name="download" /></button>
            <button aria-label={tr("推理设置", "Inference settings")} onClick={openSettings} title={tr("推理设置", "Inference settings")} type="button"><Icon name="settings" /></button>
          </div>
        </header>

        {view === "chat" ? (
          <section className="chat-view">
            <div className="message-scroller" onScroll={handleMessageScroll} ref={messageScrollerRef}>
              <div className="message-list" aria-live="polite">
                {!active && <div className="welcome"><img src="/mfq-mark.svg" alt="" />{!model ? <><h1>{tr("加载模型", "Load a model")}</h1><p>{tr("从服务器文件夹加载模型，或连接已有模型服务。", "Load a model from a server folder or connect to an existing model server.")}</p><button className="open-model-primary" disabled={busy} onClick={() => void chooseModelDirectory()} type="button"><Icon name="folder" />{tr("选择模型文件夹", "Choose model folder")}</button></> : <><h1>MFQ Studio</h1><p>{tr("创建会话后即可开始本地推理。", "Create a session to start local inference.")}</p><div className="prompt-grid"><button onClick={() => setDraft(tr("介绍一下这个模型。", "Introduce this model."))} type="button">{tr("介绍模型", "Introduce the model")}</button><button onClick={() => setDraft(tr("写一段 Python 示例。", "Write a Python example."))} type="button">{tr("代码示例", "Code example")}</button></div></>}</div>}
                {messages.map((message) => {
                  const parts = textParts(message);
                  const editing = editDraft?.messageId === message.id;
                  const response = responses[message.id];
                  const responsePrefill = displayPrefillMetric(response?.performance);
                  const responseTtftMs = preferPositiveMetric(
                    response?.performance?.ttft_ms,
                    response?.performance?.complete_prefill_ms,
                  );
                  return <article className={`message message-${message.role}`} key={message.id}>
                    <div className="message-avatar">{message.role === "assistant" ? <img src="/mfq-mark.svg" alt="MFQ" /> : <span>{message.role === "user" ? tr("你", "You") : message.role}</span>}</div>
                    <div className="message-body">
                      <div className="message-content">
                        <div className="message-meta"><strong>{message.role === "assistant" ? "MFQ" : message.role === "user" ? tr("你", "You") : message.role}</strong><span>{new Date(message.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span></div>
                        {editing ? <div className="message-editor">{message.role === "assistant" && <textarea aria-label={tr("思考过程", "Reasoning")} onChange={(event) => setEditDraft((current) => current && ({ ...current, reasoning: event.target.value }))} placeholder={tr("思考过程", "Reasoning")} value={editDraft.reasoning} />}<textarea aria-label={tr("消息", "Message")} onChange={(event) => setEditDraft((current) => current && ({ ...current, text: event.target.value }))} value={editDraft.text} /><div><button onClick={() => setEditDraft(null)} type="button">{tr("取消", "Cancel")}</button><button className="primary" onClick={() => void saveEdit(message)} type="button">{tr("保存", "Save")}</button></div></div> : <>{parts.reasoning && <details className="reasoning"><summary>{tr("思考过程", "Reasoning")}</summary>{renderMarkdown(parts.reasoning, false, message.role === "assistant")}</details>}{message.parts.filter(isMediaPart).map((part, index) => <MediaPartView key={`${part.media.id}-${index}`} part={part} />)}{message.parts.filter((part) => part.type === "document").map((part, index) => <DocumentPartView key={`${part.media.id}-${index}`} part={part} />)}{parts.text && renderMarkdown(parts.text, false, message.role === "assistant")}{message.parts.filter((part) => part.type === "tool_call" || part.type === "tool_result").map((part, index) => <div className="tool-call" key={index}><pre>{part.type === "tool_call" ? `${part.name}(${JSON.stringify(part.arguments, null, 2)})` : JSON.stringify(part.result, null, 2)}</pre></div>)}{message.parts.some((part) => part.type === "tool_call" && mcpTools.some((tool) => tool.qualified_name === part.name)) && <div className="tool-confirm"><button disabled={busy} onClick={() => void executeToolCalls(message)} type="button">{tr("确认并执行所有工具", "Confirm and run all tools")}</button></div>}</>}
                      </div>
                      {response?.performance && <details className="response-metrics"><summary><span>{formatNumber(response.performance.decode_tps, 1)} tok/s</span><span>{formatNumber(responsePrefill.tokensPerSecond, 1)} pp</span><span>{formatNumber(responseTtftMs, 1)} ms TTFT</span></summary><div><span>{response.performance.prefill_tokens} prompt tokens</span><span>{response.usage?.completion_tokens ?? 0} output tokens</span>{response.performance.processor_ms > 0 && <span>{tr("媒体准备", "Media preparation")} {formatNumber(response.performance.processor_ms, 1)} ms</span>}{response.performance.multimodal_ms > 0 && <span>{tr("多模态编码", "Multimodal encoding")} {formatNumber(response.performance.multimodal_ms, 1)} ms</span>}{response.performance.model_prefill_ms > response.performance.multimodal_ms && <span>LLM {formatNumber(response.performance.prefill_ms, 1)} ms</span>}<span>{response.finish_reason || "stop"}</span><span>T {formatNumber(response.performance.sampling.temperature, 2)}</span><span>top-p {formatNumber(response.performance.sampling.top_p, 2)}</span><span>repeat {formatNumber(response.performance.sampling.repetition_penalty, 2)}</span>{response.performance.sampling.reasoning_effort && <span>{response.performance.sampling.reasoning_effort}</span>}</div></details>}
                      {!editing && <div className="message-actions"><button onClick={() => void copyMessage(message)} type="button">{tr("复制", "Copy")}</button>{(message.role === "user" || message.role === "assistant") && <button onClick={() => setEditDraft({ messageId: message.id, ...parts })} type="button">{tr("编辑", "Edit")}</button>}{message.role === "assistant" && <button onClick={() => void regenerate(message)} type="button">{tr("重新生成", "Regenerate")}</button>}</div>}
                    </div>
                  </article>;
                })}
                {currentVoiceMessages.map((message) => <article className={`message message-${message.role}`} key={message.id}><div className="message-avatar">{message.role === "assistant" ? <img src="/mfq-mark.svg" alt="MFQ" /> : <span>{tr("你", "You")}</span>}</div><div className="message-body"><div className="message-meta"><strong>{message.role === "assistant" ? "MFQ" : tr("你", "You")}</strong><span>{tr("语音", "Voice")}</span></div>{message.text && renderMarkdown(message.text, false, message.role === "assistant")}{message.audioId && <AudioClip audioId={message.audioId} />}</div></article>)}
                {liveVoice?.sessionId === activeId && liveVoice.text && <article className="message message-assistant live-message"><div className="message-avatar"><img src="/mfq-mark.svg" alt="MFQ" /></div><div className="message-body"><div className="message-meta"><strong>MFQ</strong><span>{tr("生成中", "Generating")}</span></div>{renderMarkdown(liveVoice.text, true, true)}</div></article>}
                {live && <article className="message message-assistant live-message"><div className="message-avatar"><img src="/mfq-mark.svg" alt="MFQ" /></div><div className="message-body"><div className="message-meta"><strong>MFQ</strong><span>{tr("生成中", "Generating")}</span></div>{live.reasoning && <details className="reasoning" open><summary>{tr("正在思考", "Thinking")}</summary>{renderMarkdown(live.reasoning, true, true)}</details>}{live.text && renderMarkdown(live.text, true, true)}{live.tools.map((tool, index) => <pre className="tool-call" key={index}>{tool}</pre>)}{!live.reasoning && !live.text && live.tools.length === 0 && <span className="thinking"><i /><i /><i /></span>}</div></article>}
              </div>
            </div>
            {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError(null)} type="button">×</button></div>}
            <div className="composer-region">
              <form className="composer" onSubmit={send}>
                {attachments.length > 0 && <div className="attachment-tray">{attachments.map((attachment) => <div className="attachment-chip" key={attachment.id}>{attachment.kind === "image" ? <img alt="" src={attachment.previewUrl} /> : attachment.kind === "video" ? <video muted src={attachment.previewUrl} /> : <span>{attachment.kind === "document" ? "TXT" : "♫"}</span>}<div><strong>{attachment.file.name}</strong><small>{attachment.kind} · {formatNumber(attachment.file.size)} B</small></div><button aria-label={tr("移除附件", "Remove attachment")} onClick={() => removeAttachment(attachment.id)} type="button">×</button></div>)}</div>}
                <textarea aria-label={tr("消息", "Message")} disabled={!active || busy} maxLength={32768} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} placeholder={active ? tr("向模型发送消息", "Message MFQ") : tr("请先创建会话", "Create a session first")} rows={1} value={draft} />
                <div className="composer-toolbar">
                  <input accept={attachmentAccept} hidden multiple onChange={(event) => selectAttachments(event.target.files)} ref={attachmentInputRef} type="file" />
                  <button aria-label={tr("添加附件", "Add attachment")} disabled={!active || busy} onClick={() => attachmentInputRef.current?.click()} title={tr("添加文档或媒体", "Add document or media")} type="button"><Icon name="paperclip" /></button>
                  {mcpTools.length > 0 && <select aria-label={tr("可用工具", "Available tools")} onChange={(event) => { const name = event.target.value; if (name && !selectedTools.includes(name)) setSelectedTools((current) => [...current, name]); event.target.value = ""; }} defaultValue=""><option value="">{selectedTools.length ? `${selectedTools.length} ${tr("个工具", "tools")}` : tr("工具", "Tools")}</option>{mcpTools.filter((tool) => !selectedTools.includes(tool.qualified_name)).map((tool) => <option key={tool.qualified_name} value={tool.qualified_name}>{tool.qualified_name}</option>)}</select>}
                  {selectedTools.length > 0 && <button aria-label={tr("清空工具", "Clear tools")} onClick={() => setSelectedTools([])} type="button">× {selectedTools.length}</button>}
                  {realtimeAvailable && <select aria-label={tr("交互模式", "Interaction mode")} disabled={!active || busy || voiceState !== "idle"} onChange={(event) => void selectInteractionMode(event.target.value as SessionMode)} value={active?.mode ?? mode}>{(["text", "voice", "full_duplex"] as SessionMode[]).map((item) => { const feature = capabilities?.model_capabilities.features; const disabled = item === "voice" ? !feature?.audio_input : item === "full_duplex" ? !feature?.full_duplex : false; return <option disabled={disabled} key={item} value={item}>{MODE_LABELS[item][english ? 1 : 0]}</option>; })}</select>}
                  {realtimeAvailable && <button aria-label={tr("语音输入", "Voice input")} aria-pressed={voiceState !== "idle" && voiceState !== "error"} className="voice-button" disabled={!active || active.mode === "text" || busy} onClick={() => void toggleVoice()} style={{ "--voice-level": voiceLevel } as React.CSSProperties} title={active?.mode === "text" ? tr("请先选择语音或全双工模式", "Select voice or full duplex mode first") : voiceState === "processing" ? tr("语音处理中", "Processing voice") : tr("语音输入", "Voice input")} type="button"><span /></button>}
                  {realtimeAvailable && active?.mode !== "text" && <button aria-label={tr("语音播放", "Voice playback")} aria-pressed={settings.playbackEnabled} onClick={() => setSettings((current) => ({ ...current, playbackEnabled: !current.playbackEnabled }))} title={tr("语音播放", "Voice playback")} type="button"><Icon name={settings.playbackEnabled ? "volume" : "volume-off"} /></button>}
                  {active?.mode === "text" && <button aria-pressed={thinkingSupported && effectiveSettings.enableThinking} disabled={!thinkingSupported || roleOverridesInference} onClick={() => updateGlobalInference({ enableThinking: !effectiveSettings.enableThinking })} title={roleOverridesInference ? tr("该参数由当前角色覆盖", "This setting is overridden by the current role") : undefined} type="button"><Icon name="lightbulb" />{tr("思考", "Thinking")}</button>}
                  {active?.mode === "text" && thinkingSupported && effectiveSettings.enableThinking && reasoningValues.length > 0 && <select aria-label={tr("思考档位", "Reasoning effort")} disabled={roleOverridesInference} onChange={(event) => updateGlobalInference({ reasoningEffort: event.target.value })} value={effectiveSettings.reasoningEffort}><option value="">{tr("标准", "Standard")}</option>{reasoningValues.map((value) => <option key={value} value={value}>{value}</option>)}</select>}
                  <span className="composer-hint">{voiceState !== "idle" ? voiceState : tr("Enter 发送 · Shift+Enter 换行", "Enter to send · Shift+Enter for newline")}</span>
                  {busy ? <button aria-label={tr("停止生成", "Stop generation")} className="send-button stop" onClick={() => abortRef.current?.abort()} type="button"><Icon name="stop" size={14} /></button> : <button aria-label={tr("发送", "Send")} className="send-button" disabled={!active || (!draft.trim() && !attachments.length)} type="submit"><Icon name="send" size={15} /></button>}
                </div>
              </form>
              <p>{tr("模型输出可能存在错误，请核对重要信息。", "Model output may be inaccurate. Verify important information.")}</p>
            </div>
          </section>
        ) : view === "dashboard" ? (
          <section className="dashboard-view" id="dashboard-overview">
            <div className="page-heading"><div><h1>{dashboardPage === "overview" ? tr("概览", "Overview") : dashboardPage === "cache" ? tr("缓存与配置", "Cache and profiles") : dashboardPage === "models" ? tr("模型与任务", "Models and jobs") : tr("工具与连接", "Tools and connections")}</h1></div><button aria-label={tr("重置当前布局", "Reset current layout")} onClick={() => setDashboardLayoutReset((current) => current + 1)} title={tr("重置当前布局", "Reset current layout")} type="button"><Icon name="refresh" /></button></div>
            {dashboardPage === "overview" && <PanelDeck labels={panelLabels} page="dashboard-overview" resetVersion={dashboardLayoutReset}>
              <div className="metric-grid" key="metrics"><article><span>Prefill</span><strong>{formatNumber(lastPrefill.tokensPerSecond, 1)}</strong><small>tokens / second</small></article><article><span>Decode</span><strong>{formatNumber(last?.decode_tps, 1)}</strong><small>tokens / second</small></article><article><span>TTFT</span><strong>{formatNumber(lastTtftMs, 1)}</strong><small>milliseconds</small></article><article><span>{tr("内存", "Memory")}</span><strong>{runtimeMemory ? `${formatNumber(runtimeMemory / 2 ** 30, 1)} GB` : "--"}</strong><small>{runtimeCache ? `${formatNumber(runtimeCache / 2 ** 30, 1)} GB cache` : tr("未上报缓存", "cache unavailable")}</small></article><article><span>{tr("任务", "Jobs")}</span><strong>{activeJobs.length}</strong><small>{formatNumber(runtime?.active_requests || 0)} active requests</small></article></div>
              <section className="dashboard-panel chart-panel" key="throughput"><div className="panel-heading"><div><h2>{tr("生成吞吐", "Decode throughput")}</h2><p>{tr("MFQ Server 保存的已完成请求", "Completed requests retained by MFQ Server")}</p></div><b>{formatNumber(last?.decode_tps, 1)} tok/s</b></div><RuntimeChart values={metricSeries} /></section>
              <section className="dashboard-panel" key="runtime"><div className="panel-heading"><div><h2>Runtime</h2><p>{instances.length ? `${instances.length} ${tr("个托管实例", "managed instances")}` : runtime?.model ? tr("外部 Runtime", "External runtime") : tr("尚未加载模型", "No model loaded")}</p></div></div><dl><div><dt>{tr("模型", "Model")}</dt><dd>{runtime?.model || model || "--"}</dd></div><div><dt>{tr("架构", "Architecture")}</dt><dd>{runtime?.model_type || "--"}</dd></div><div><dt>{tr("状态", "State")}</dt><dd>{runtime?.runtime_state || (runtime?.reloading ? "loading" : "unavailable")}</dd></div><div><dt>{tr("上下文", "Context")}</dt><dd>{formatNumber(runtime?.max_context)} / {formatNumber(runtime?.context_capacity)}</dd></div><div><dt>{tr("运行时间", "Uptime")}</dt><dd>{formatDuration(runtime?.uptime_seconds)}</dd></div><div><dt>{tr("失败请求", "Failed")}</dt><dd>{formatNumber(runtime?.failed_requests || 0)}</dd></div></dl><button className="panel-action" disabled={!runtime?.model} onClick={openSettings} type="button">{tr("调整上下文与推理设置", "Context and inference settings")}</button></section>
              <section className="dashboard-panel request-panel" key="request"><div className="panel-heading"><div><h2>{tr("最近请求性能", "Last request performance")}</h2><p>{last?.id || tr("还没有完成的请求", "No completed requests")}</p></div>{(last?.finish_reason || Number(runtime?.active_requests || 0) > 0) && <b>{last?.finish_reason || "Running"}</b>}</div><div className="request-stats"><div><span>{tr("输入", "Input")}</span><strong>{formatNumber(last?.prompt_tokens)}</strong><small>tokens</small></div><div><span>{tr("输出", "Output")}</span><strong>{formatNumber(last?.completion_tokens)}</strong><small>tokens</small></div><div><span>TTFT</span><strong>{formatNumber(lastTtftMs, 1)}</strong><small>ms</small></div><div><span>Prefill</span><strong>{formatNumber(lastPrefill.tokensPerSecond, 1)}</strong><small>{formatNumber(lastPrefill.milliseconds, 1)} ms</small></div><div><span>Decode</span><strong>{formatNumber(last?.decode_tps, 1)}</strong><small>{formatNumber(last?.decode_ms, 1)} ms</small></div><div><span>{tr("总耗时", "Total")}</span><strong>{formatNumber(lastGenerationMs, 1)}</strong><small>ms</small></div></div></section>
            </PanelDeck>}
            {dashboardPage === "cache" && <PanelDeck labels={panelLabels} page="dashboard-cache" resetVersion={dashboardLayoutReset}>
              {prefixCacheSupported && <section className="dashboard-panel cache-panel" key="prefix-cache"><div className="panel-heading"><div><h2>Prefix cache</h2><p>{tr("跨轮次复用稳定前缀的 KV 快照", "Reusable KV snapshots for stable conversation prefixes")}</p></div><b>{prefixCacheQueries > 0 ? `${formatNumber(prefixCacheHitRate, 1)}% hit` : tr("暂无查询", "No queries")}</b></div><div className="cache-stats"><div><span>{tr("会话", "Sessions")}</span><strong>{formatNumber(runtime?.prefix_cache_sessions)}</strong></div><div><span>{tr("快照", "Snapshots")}</span><strong>{formatNumber(prefixCacheSnapshots)}</strong></div><div><span>{tr("复用 tokens", "Reused tokens")}</span><strong>{formatNumber(runtime?.prefix_cache_hit_tokens)}</strong></div><div><span>{tr("占用", "Memory")}</span><strong>{formatNumber(prefixCacheBytes / 2 ** 20, 1)} MB</strong></div><div><span>{tr("预算", "Budget")}</span><strong>{formatNumber(Number(runtime?.prefix_cache_max_bytes || 0) / 2 ** 30, 1)} GB</strong></div></div>{prefixCacheSnapshots > 0 && <button className="panel-action danger" disabled={busy || Number(runtime?.active_requests || 0) > 0} onClick={() => void clearRuntimeCache()} type="button">{tr("清除 prefix cache", "Clear prefix cache")}</button>}</section>}
              <section className="dashboard-panel profile-panel" key="profiles"><div className="panel-heading"><div><h2>{tr("运行配置档案", "Runtime profiles")}</h2><p>{tr("将加载参数和采样默认值绑定到模型产物", "Bind load and sampling defaults to a model artifact")}</p></div><b>{runtimeProfiles.length}</b></div><div className="profile-create"><input maxLength={64} onChange={(event) => setProfileName(event.target.value)} placeholder={tr("当前配置名称", "Current configuration name")} value={profileName} /><button disabled={busy || !profileName.trim() || !artifacts.some((item) => item.name === runtime?.model)} onClick={() => void saveRuntimeProfile()} type="button">{tr("保存当前配置", "Save current")}</button></div>{runtimeProfiles.length > 0 && <div className="profile-list">{runtimeProfiles.map((profile) => <div className={`profile-row ${profile.drifted ? "drifted" : ""}`} key={profile.id}><div><strong>{profile.name}</strong><small>{profile.load.context_size.toLocaleString()} ctx · {profile.load.prefill_chunk_size.toLocaleString()} chunk{profile.drifted ? ` · ${tr("模型已变化", "artifact changed")}` : ""}</small></div><button disabled={busy} onClick={() => void loadRuntimeProfile(profile)} type="button">{tr("加载", "Load")}</button><button aria-label={tr("删除配置档案", "Delete profile")} disabled={busy} onClick={() => void deleteRuntimeProfile(profile.id)} type="button"><Icon name="trash" size={14} /></button></div>)}</div>}</section>
            </PanelDeck>}
            {dashboardPage === "models" && <PanelDeck labels={panelLabels} page="dashboard-models" resetVersion={dashboardLayoutReset}>
              <section className="dashboard-panel" key="catalog"><div className="panel-heading"><div><h2>{tr("模型目录", "Model catalog")}</h2></div><div className="panel-heading-actions"><button disabled={busy} onClick={() => void chooseModelDirectory()} type="button">{tr("添加模型文件夹", "Add model folder")}</button><b>{artifacts.length}</b></div></div>{artifacts.length > 0 && <div className="model-list">{artifacts.slice(0, 8).map((item) => { const instance = instances.find((candidate) => candidate.model === item.name && candidate.state !== "failed"); const loaded = Boolean(instance) || item.name === runtime?.model; return <div className="model-row" key={item.id}><span className={loaded ? "model-state active" : item.loadable ? "model-state" : "model-state failed"} /><div><strong>{item.name}</strong><small>{item.architecture} · {item.shard_count} shards · {formatNumber(item.total_bytes / 2 ** 30, 1)} GB</small></div>{instance ? <button disabled={busy || instance.state === "busy"} onClick={() => void unloadInstance(instance.id)} type="button">{tr("卸载", "Unload")}</button> : loaded ? <em>{tr("已加载", "Loaded")}</em> : !item.loadable ? <em className="failed">{tr("不可用", "Invalid")}</em> : <button disabled={busy} onClick={() => void loadArtifact(item.name)} type="button">{tr("加载", "Load")}</button>}</div>; })}</div>}</section>
              <section className="dashboard-panel" key="requests"><div className="panel-heading"><div><h2>{tr("最近请求", "Recent requests")}</h2></div></div>{requestHistory.length > 0 && <div className="request-table">{requestHistory.slice(0, 8).map((request) => <div className="request-row" key={request.id}><div><strong>{request.id}</strong><small>{request.completed_at ? new Date(request.completed_at * 1000).toLocaleTimeString() : request.endpoint || "completion"}</small></div><span>{formatNumber(request.prompt_tokens)} → {formatNumber(request.completion_tokens)}</span><b>{formatNumber(request.decode_tps, 1)} tok/s</b></div>)}</div>}</section>
              <section className="dashboard-panel" key="jobs">
                <div className="panel-heading"><div><h2>{tr("后台任务", "Background jobs")}</h2></div><b>{activeJobs.length}</b></div>
                {activeJobs.length > 0 && <div className="request-table">{activeJobs.slice(0, 8).map((job) => <div className="job-row" key={job.id}><div><strong>{job.kind}</strong><small>{new Date(job.updated_at).toLocaleTimeString()} · {job.status}</small></div><progress max={1} value={job.progress} /><b>{formatNumber(job.progress * 100)}%</b></div>)}</div>}
                {completedJobs.length > 0 && <details className="completed-jobs">
                  <summary><span>{tr("已完成", "Completed")} <b>{completedJobs.length}</b></span><button disabled={jobCleanupBusy} onClick={(event) => { event.preventDefault(); void clearCompletedJobRecords(); }} type="button">{tr("清理已完成", "Clear completed")}</button></summary>
                  <div className="request-table">{completedJobs.slice(0, 8).map((job) => <div className="job-row completed" key={job.id}><div><strong>{job.kind}</strong><small>{new Date(job.updated_at).toLocaleTimeString()} · {job.status}</small></div><progress max={1} value={job.progress} /><b>{formatNumber(job.progress * 100)}%</b><button aria-label={tr("移出任务历史", "Remove from job history")} disabled={jobCleanupBusy} onClick={() => void deleteJobRecord(job.id)} type="button"><Icon name="trash" size={12} /></button></div>)}</div>
                </details>}
              </section>
              <section className="dashboard-panel" key="logs"><div className="panel-heading"><div><h2>{tr("Runtime 日志", "Runtime logs")}</h2></div><b>{runtimeLogs.length}</b></div>{runtimeLogs.length > 0 && <div className="runtime-log-list">{runtimeLogs.slice(-8).reverse().map((entry) => <div className={`runtime-log ${entry.level}`} key={entry.sequence}><span>{new Date(entry.created_at).toLocaleTimeString()}</span><p>{entry.message}</p></div>)}</div>}</section>
            </PanelDeck>}
            {dashboardPage === "connections" && <PanelDeck labels={panelLabels} page="dashboard-connections" resetVersion={dashboardLayoutReset}>
              <section className="dashboard-panel mcp-panel" key="mcp"><div className="panel-heading"><div><h2>MCP</h2><p>{tr("工具服务器与模型可见工具", "Tool servers and model-visible tools")}</p></div><b>{mcpTools.length} tools</b></div><form className="mcp-form" onSubmit={createMcpServer}><input aria-label={tr("服务器名称", "Server name")} onChange={(event) => setMcpDraft((current) => ({ ...current, name: event.target.value }))} placeholder={tr("名称", "Name")} value={mcpDraft.name} /><select aria-label={tr("传输方式", "Transport")} onChange={(event) => setMcpDraft((current) => ({ ...current, transport: event.target.value as "stdio" | "streamable_http" }))} value={mcpDraft.transport}><option value="streamable_http">HTTP</option><option value="stdio">stdio</option></select><input aria-label={mcpDraft.transport === "streamable_http" ? "Streamable HTTP URL" : tr("可执行文件", "Executable")} onChange={(event) => setMcpDraft((current) => ({ ...current, endpoint: event.target.value }))} placeholder={mcpDraft.transport === "streamable_http" ? "https://host/mcp" : tr("可执行文件路径", "Executable path")} type={mcpDraft.transport === "streamable_http" ? "url" : "text"} value={mcpDraft.endpoint} /><button className="primary" disabled={busy || !mcpDraft.name.trim() || !mcpDraft.endpoint.trim()} type="submit">{tr("添加", "Add")}</button></form><div className="mcp-server-list">{mcpServers.map((server) => <div className="mcp-server" key={server.id}><span className={server.enabled ? "model-state active" : "model-state"} /><div><strong>{server.name}</strong><small>{server.transport} · {server.url || server.command}</small></div><button onClick={() => void toggleMcpServer(server)} type="button">{server.enabled ? tr("停用", "Disable") : tr("启用", "Enable")}</button><button onClick={() => void deleteMcpServer(server.id)} type="button">{tr("删除", "Delete")}</button></div>)}</div></section>
              <div key="nodes">{clusterPanel}</div>
            </PanelDeck>}
          </section>
        ) : (
          <section aria-label="Lab" className="lab-view">
            <div className="page-heading"><div><h1>{labPage === "models" ? tr("模型仓库", "Model hubs") : labPage === "evaluations" ? tr("评测与数据集", "Evaluation and datasets") : tr("量化工作台", "Quantization workspace")}</h1></div><button aria-label={tr("重置当前布局", "Reset current layout")} onClick={() => setLabLayoutReset((current) => current + 1)} title={tr("重置当前布局", "Reset current layout")} type="button"><Icon name="refresh" /></button></div>
            {labPage === "models" && <PanelDeck labels={panelLabels} page="lab-models" resetVersion={labLayoutReset}><div key="hubs"><section className="dashboard-panel hub-panel"><div className="panel-heading"><div><h2>{tr("模型仓库", "Model hubs")}</h2><p>{tr("搜索、检查大小并启动可续传下载", "Search, inspect size, and start resumable downloads")}</p></div></div><form className="hub-search" onSubmit={searchHub}><select onChange={(event) => setHubProvider(event.target.value as HubModelSummary["provider"])} value={hubProvider}><option value="modelscope">ModelScope</option><option value="huggingface">Hugging Face</option></select><input onChange={(event) => setHubQuery(event.target.value)} placeholder={tr("模型名称或仓库", "Model or repository")} value={hubQuery} /><button disabled={busy || !hubQuery.trim()} type="submit">{tr("搜索", "Search")}</button></form>{hubResults.length > 0 && <div className="hub-results">{hubResults.map((item) => <button className={hubModel?.repo_id === item.repo_id ? "active" : ""} key={`${item.provider}:${item.repo_id}`} onClick={() => void inspectHubModel(item)} type="button"><div><strong>{item.repo_id}</strong><small>{formatNumber(item.downloads)} downloads · {formatNumber(item.likes)} likes</small></div><span>{item.total_bytes ? `${formatNumber(item.total_bytes / 2 **30, 1)} GB` : "--"}</span></button>)}</div>}{hubModel && <div className="hub-detail"><div><strong>{hubModel.repo_id}</strong><small>{hubModel.revision} · {hubModel.files.length} files · {formatNumber(hubModel.total_bytes / 2 ** 30, 2)} GB</small></div><button disabled={busy || !jobKinds.some((item) => item.kind === `download.${hubModel.provider}`)} onClick={() => void downloadHubModel()} type="button"><Icon name="download" size={14} />{tr("下载", "Download")}</button></div>}</section></div></PanelDeck>}
            {labPage === "evaluations" && <PanelDeck labels={panelLabels} page="lab-evaluations" resetVersion={labLayoutReset}>
              <section className="dashboard-panel evaluation-panel" key="results">
                <div className="panel-heading"><div><h2>{tr("评测结果", "Evaluation results")}</h2><p>{tr("只允许数据集与运行参数一致的结果对比", "Comparison requires matching datasets and execution parameters")}</p></div><b>{evaluations.length}</b></div>
                <div className="evaluation-list">{evaluations.map((item) => <label key={item.id}><input checked={selectedEvaluations.includes(item.id)} onChange={(event) => setSelectedEvaluations((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id))} type="checkbox" /><div><strong>{item.model_id}</strong><small>{item.kind} · {new Date(item.created_at).toLocaleString()}</small></div><span>{Object.entries(item.metrics).filter(([, value]) => typeof value === "number").slice(0, 2).map(([name, value]) => `${name} ${formatNumber(Number(value), 3)}`).join(" · ")}</span></label>)}</div>
                <button className="panel-action" disabled={busy || selectedEvaluations.length < 2} onClick={() => void compareSelectedEvaluations()} type="button">{tr("对比所选结果", "Compare selected")}</button>
                {evaluationComparison && <div className="comparison-table"><header><span>{tr("模型", "Model")}</span>{evaluationComparison.metrics.map((metric) => <b key={metric}>{metric}</b>)}</header>{evaluationComparison.rows.map((row) => <div key={row.evaluation.id}><strong>{row.evaluation.model_id}</strong>{evaluationComparison.metrics.map((metric) => <span key={metric}>{formatNumber(Number(row.evaluation.metrics[metric]), 4)}<small>{row.deltas[metric] == null ? "" : ` ${Number(row.deltas[metric]) >= 0 ? "+" : ""}${formatNumber(Number(row.deltas[metric]), 4)}`}</small></span>)}</div>)}</div>}
              </section>
              <section className="dashboard-panel dataset-panel" key="datasets">
                <div className="panel-heading"><div><h2>{tr("数据集", "Datasets")}</h2><p>{tr("可复现的文件哈希与来源清单", "Reproducible file hashes and source manifests")}</p></div><b>{datasets.length}</b></div>
                <form className="dataset-form" onSubmit={registerDataset}><input onChange={(event) => setDatasetDraft((current) => ({ ...current, name: event.target.value }))} placeholder={tr("名称", "Name")} value={datasetDraft.name} /><select onChange={(event) => setDatasetDraft((current) => ({ ...current, kind: event.target.value as DatasetResource["kind"] }))} value={datasetDraft.kind}><option value="custom">Custom</option><option value="wikitext2">WikiText-2</option></select><input onChange={(event) => setDatasetDraft((current) => ({ ...current, artifact_uri: event.target.value }))} placeholder="workspace://datasets/corpus.txt" value={datasetDraft.artifact_uri} /><button disabled={busy} type="submit">{tr("注册", "Register")}</button></form>
                <div className="dataset-list">{datasets.map((item) => <div key={item.id}><div><strong>{item.name}</strong><small>{item.kind} · {formatNumber(item.byte_size / 2 ** 20, 2)} MiB · {item.sha256.slice(0, 12)}</small></div><button aria-label={tr("删除数据集", "Delete dataset")} onClick={() => void api.deleteDataset(item.id).then(() => setDatasets((current) => current.filter((entry) => entry.id !== item.id))).catch((cause) => setError(errorMessage(cause)))} type="button"><Icon name="trash" size={13} /></button></div>)}</div>
              </section>
            </PanelDeck>}
            {labPage === "quantization" && <PanelDeck labels={panelLabels} page="lab-quantization" resetVersion={labLayoutReset}>
            <section className="dashboard-panel imatrix-panel" key="imatrix">
              <div className="panel-heading"><div><h2>Imatrix</h2><p>{tr("单独校准、导入，或在量化任务中先校准再使用", "Calibrate separately, import one, or collect it before quantization")}</p></div><b>{imatrixArtifacts.length}</b></div>
              <div className="imatrix-actions">
                <button disabled={busy || !jobKinds.some((item) => item.kind === "calibrate.imatrix")} onClick={() => setSelectedJobKind("calibrate.imatrix")} type="button"><Icon name="plus" size={11} />{tr("新建校准", "New calibration")}</button>
                <button disabled={busy || imatrixImporting || !jobKinds.some((item) => item.kind === "artifact.import")} onClick={() => imatrixInputRef.current?.click()} type="button"><Icon name="upload" size={11} />{imatrixImporting ? tr("正在导入", "Importing") : tr("导入文件", "Import file")}</button>
                <input accept=".imatrix,.gguf,.dat,application/x-mfq-imatrix,application/octet-stream" hidden onChange={(event) => void importImatrix(event.target.files)} ref={imatrixInputRef} type="file" />
              </div>
              {imatrixArtifacts.length > 0 && <div className="imatrix-list">{imatrixArtifacts.slice(0, 12).map((item) => <button key={item.id} onClick={() => { setPendingImatrix(item.artifact_uri.replace(/^workspace:\/\//, "")); setSelectedJobKind("model.quantize"); }} type="button"><div><strong>{item.artifact_name}</strong><small>{item.artifact_uri}</small></div><span>{tr("用于量化", "Use")}</span></button>)}</div>}
            </section>
            <section className="dashboard-panel lineage-panel" key="lineage"><div className="panel-heading"><div><h2>{tr("产物谱系", "Artifact lineage")}</h2><p>{tr("源产物、生成任务、默认后参数和验证记录", "Sources, producing jobs, resolved parameters, and validations")}</p></div><b>{lineage.length}</b></div><div className="lineage-list">{lineage.slice(0, 20).map((item) => <details key={item.id}><summary><div><strong>{item.artifact_name}</strong><small>{item.producer_kind} · {new Date(item.created_at).toLocaleString()}</small></div><span>{item.validation_job_ids.length} checks</span></summary><dl><div><dt>URI</dt><dd>{item.artifact_uri}</dd></div><div><dt>{tr("源", "Sources")}</dt><dd>{item.source_uris.join(", ") || "--"}</dd></div></dl><pre>{JSON.stringify(item.parameters, null, 2)}</pre></details>)}</div></section>
              <form className="dashboard-panel job-builder" key="builder" onSubmit={submitJob}>
                <div className="panel-heading"><div><h2>{tr("新任务", "New job")}</h2></div></div>
                <label><span>{tr("任务类型", "Job type")}</span><select onChange={(event) => setSelectedJobKind(event.target.value)} value={selectedJobKind}>{genericJobKinds.map((item) => <option key={item.kind} value={item.kind}>{item.kind}</option>)}</select></label>
                {Object.entries(selectedKind?.payload_schema.properties ?? {}).map(([name, property]) => {
                  if (
                    selectedJobKind === "model.quantize"
                    && name.startsWith("imatrix_")
                    && name !== "imatrix"
                    && !Boolean(jobPayload.calibrate_imatrix)
                  ) return null;
                  if (
                    selectedJobKind === "model.quantize"
                    && name === "imatrix"
                    && Boolean(jobPayload.calibrate_imatrix)
                  ) return null;
                  const required = selectedKind?.payload_schema.required?.includes(name);
                  const type = schemaType(property);
                  const value = jobPayload[name];
                  const isImatrix = selectedJobKind === "model.quantize" && name === "imatrix";
                  return <label key={name}><span>{property.title || name}{required ? " *" : ""}</span>{isImatrix && imatrixArtifacts.length > 0 ? <select onChange={(event) => updateJobPayload(name, property, event.target.value)} value={String(value ?? "")}><option value="">{tr("不使用", "None")}</option>{imatrixArtifacts.map((item) => <option key={item.id} value={item.artifact_uri.replace(/^workspace:\/\//, "")}>{item.artifact_name}</option>)}</select> : property.enum ? <select onChange={(event) => updateJobPayload(name, property, event.target.value)} required={required} value={String(value ?? "")}>{!required && <option value="" />}{property.enum.map((item) => <option key={String(item)} value={String(item)}>{String(item)}</option>)}</select> : type === "boolean" ? <input checked={Boolean(value)} onChange={(event) => updateJobPayload(name, property, event.target.checked)} type="checkbox" /> : <input max={property.maximum} min={property.minimum} onChange={(event) => updateJobPayload(name, property, event.target.value)} required={required} type={type === "integer" || type === "number" ? "number" : "text"} value={Array.isArray(value) ? value.join(", ") : String(value ?? "")} />}{property.description && <small>{property.description}</small>}</label>;
                })}
                <div className="job-submit">
                  <button className="job-run" disabled={busy || !selectedJobKind} type="submit"><Icon name="play" size={12} />{tr("运行", "Run")}</button>
                </div>
              </form>
              <section className="dashboard-panel job-history" key="history">
                <div className="panel-heading"><div><h2>{tr("任务历史", "Job history")}</h2></div><b>{activeJobs.length}</b></div>
                {activeJobs.length > 0 && <div className="job-list">{activeJobs.map((job) => <button className={job.id === selectedJobId ? "active" : ""} key={job.id} onClick={() => setSelectedJobId(job.id)} type="button"><span className={`job-status ${job.status}`} /><div><strong>{job.kind}</strong><small>{job.status} · {new Date(job.updated_at).toLocaleString()}</small></div><b>{formatNumber(job.progress * 100)}%</b></button>)}</div>}
                {completedJobs.length > 0 && <details className="completed-jobs">
                  <summary><span>{tr("已完成", "Completed")} <b>{completedJobs.length}</b></span><button disabled={jobCleanupBusy} onClick={(event) => { event.preventDefault(); void clearCompletedJobRecords(); }} type="button">{tr("清理已完成", "Clear completed")}</button></summary>
                  <div className="job-list">{completedJobs.map((job) => <button className={job.id === selectedJobId ? "active" : ""} key={job.id} onClick={() => setSelectedJobId(job.id)} type="button"><span className={`job-status ${job.status}`} /><div><strong>{job.kind}</strong><small>{job.status} · {new Date(job.updated_at).toLocaleString()}</small></div><b>{formatNumber(job.progress * 100)}%</b></button>)}</div>
                </details>}
              </section>
              {selectedJob && <section className="dashboard-panel job-detail" key="detail"><div className="panel-heading"><div><h2>{selectedJob.kind}</h2><p>{selectedJob.id}</p></div><b>{selectedJob.status}</b></div><progress max={1} value={selectedJob.progress} /><div className="job-result-grid"><div><span>{tr("进度", "Progress")}</span><strong>{formatNumber(selectedJob.progress * 100)}%</strong></div><div><span>{tr("更新时间", "Updated")}</span><strong>{new Date(selectedJob.updated_at).toLocaleTimeString()}</strong></div></div><div className="job-actions">{["queued", "running", "cancelling"].includes(selectedJob.status) && <button className="secondary" disabled={selectedJob.status === "cancelling"} onClick={() => void cancelSelectedJob()} type="button">{tr("取消任务", "Cancel job")}</button>}{["failed", "cancelled", "interrupted"].includes(selectedJob.status) && <button className="secondary" disabled={busy} onClick={() => void retrySelectedJob()} type="button">{tr("重试", "Retry")}</button>}{isTerminalJob(selectedJob) && <button className="secondary" disabled={jobCleanupBusy} onClick={() => void deleteJobRecord(selectedJob.id)} type="button">{tr("移出任务历史", "Remove from history")}</button>}{String(selectedJob.result?.artifact || "").startsWith("workspace://") && <button className="secondary danger" disabled={busy} onClick={() => void removeSelectedArtifact()} type="button">{tr("删除本地产物", "Delete local artifact")}</button>}</div>{selectedJob.error && <div className="job-error">{selectedJob.error.message}</div>}{selectedJob.result && <pre>{JSON.stringify(selectedJob.result, null, 2)}</pre>}<div className="job-log"><header><span>{tr("事件与日志", "Events and logs")}</span></header>{jobLogs.map((entry) => <div className={entry.level} key={entry.sequence}><time>{new Date(entry.created_at).toLocaleTimeString()}</time><code>{entry.message}</code></div>)}</div></section>}
            </PanelDeck>}
          </section>
        )}
      </main>

      {settingsOpen && <>
        <div className="drawer-scrim" onClick={() => setSettingsOpen(false)} />
        <aside className="settings-panel">
          <header>
            <div><p>Generation</p><h2>{tr("推理设置", "Inference settings")}</h2></div>
            <button onClick={() => setSettingsOpen(false)} ref={settingsCloseRef} type="button">×</button>
          </header>
          <div className="settings-scroll">
            <section className="settings-inheritance">
              <label className="check-field inheritance-toggle">
                <span>
                  <strong>{tr("随模型 / 架构默认值", "Use model / architecture defaults")}</strong>
                  <small>{tr(
                    "优先读取模型元数据，缺失字段由模型型号或架构默认值补齐。",
                    "Read model metadata first, then fill missing fields from model or architecture defaults.",
                  )}</small>
                </span>
                <input
                  checked={settingsDraft.inheritModelDefaults}
                  onChange={(event) => setModelDefaultInheritance(event.target.checked)}
                  type="checkbox"
                />
              </label>
            </section>
            <fieldset className="settings-inherited-fields" disabled={settingsDraft.inheritModelDefaults}>
              <section>
                <h3>{tr("预设", "Presets")}</h3>
                <div className="segmented">
                  {(["precise", "balanced", "creative"] as const).map((name) => (
                    <button
                      aria-pressed={settingsDraft.preset === name}
                      key={name}
                      onClick={() => applyPreset(name)}
                      type="button"
                    >
                      {name === "precise"
                        ? tr("精确", "Precise")
                        : name === "balanced"
                          ? tr("均衡", "Balanced")
                          : tr("创意", "Creative")}
                    </button>
                  ))}
                </div>
                {presetManager}
              </section>
              <section>
                <h3>{tr("提示与输出", "Prompt and output")}</h3>
                <label>
                  <span>{tr("系统提示词", "System prompt")}</span>
                  <textarea
                    onChange={(event) => setSettingsDraft((current) => ({ ...current, systemPrompt: event.target.value }))}
                    rows={4}
                    value={settingsDraft.systemPrompt}
                  />
                </label>
                <label className="check-field">
                  <span>
                    <strong>{tr("排除历史思考", "Exclude reasoning history")}</strong>
                    <small>{tr("后续请求不再发送已保存的思考内容。", "Do not send saved reasoning in later requests.")}</small>
                  </span>
                  <input
                    checked={settingsDraft.excludeReasoning}
                    onChange={(event) => setSettingsDraft((current) => ({ ...current, excludeReasoning: event.target.checked }))}
                    type="checkbox"
                  />
                </label>
                <label>
                  <span>{tr("最大生成 tokens", "Maximum output tokens")}</span>
                  <input
                    max={65536}
                    min={1}
                    onChange={(event) => setSettingsDraft((current) => ({ ...current, maxTokens: Number(event.target.value) }))}
                    type="number"
                    value={settingsDraft.maxTokens}
                  />
                </label>
              </section>
              <section>
                <h3>{tr("采样", "Sampling")}</h3>
                <label>
                  <span>Temperature <output>{settingsDraft.temperature.toFixed(2)}</output></span>
                  <input max={2} min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, temperature: Number(event.target.value), preset: "custom" }))} step={0.05} type="range" value={settingsDraft.temperature} />
                </label>
                <label>
                  <span>Top P <output>{settingsDraft.topP.toFixed(2)}</output></span>
                  <input max={1} min={0.05} onChange={(event) => setSettingsDraft((current) => ({ ...current, topP: Number(event.target.value), preset: "custom" }))} step={0.05} type="range" value={settingsDraft.topP} />
                </label>
                <label><span>Top K</span><input max={1024} min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, topK: Number(event.target.value), preset: "custom" }))} type="number" value={settingsDraft.topK} /></label>
                <label><span>Seed</span><input min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, seed: event.target.value ? Number(event.target.value) : null }))} placeholder={tr("随机", "Random")} type="number" value={settingsDraft.seed ?? ""} /></label>
              </section>
              <section>
                <h3>{tr("惩罚", "Penalties")}</h3>
                {([["Repetition", "repetitionPenalty", 0.5, 2, 0.01], ["Presence", "presencePenalty", -2, 2, 0.05], ["Frequency", "frequencyPenalty", -2, 2, 0.05]] as const).map(([label, key, min, max, step]) => (
                  <label key={key}>
                    <span>{label} <output>{settingsDraft[key].toFixed(2)}</output></span>
                    <input max={max} min={min} onChange={(event) => setSettingsDraft((current) => ({ ...current, [key]: Number(event.target.value), preset: "custom" }))} step={step} type="range" value={settingsDraft[key]} />
                  </label>
                ))}
              </section>
            </fieldset>
            <section>
              <h3>{tr("上下文", "Context")}</h3>
              <label><span>{tr("上下文窗口 tokens", "Context window tokens")}</span><input max={Number(runtime?.context_capacity) || 1048576} min={512} onChange={(event) => setContextSize(Number(event.target.value))} step={512} type="number" value={contextSize} /></label>
              <button className="secondary wide" disabled={busy} onClick={() => void reloadRuntime()} type="button">{tr("按此上下文重载模型", "Reload model with this context")}</button>
            </section>
            <section>
              <h3>{tr("界面与连接", "Interface and connection")}</h3>
              <label><span>{tr("界面语言", "Interface language")}</span><select onChange={(event) => setSettingsDraft((current) => ({ ...current, language: event.target.value as UiLanguage }))} value={settingsDraft.language}><option value="system">{tr("跟随系统", "System")}</option><option value="zh-CN">简体中文</option><option value="en">English</option></select></label>
              <label>
                <span>{tr("主题", "Theme")}</span>
                <select onChange={(event) => setSettingsDraft((current) => ({ ...current, theme: event.target.value as UiTheme }))} value={settingsDraft.theme}>
                  <option value="system">{tr("跟随系统", "System")}</option>
                  <option value="light">{tr("浅色", "Light")}</option>
                  <option value="dark">{tr("深色", "Dark")}</option>
                </select>
              </label>
              <div className="portable-actions">
                <button onClick={exportStudioData} type="button">{tr("导出", "Export")}</button>
                <label>
                  {tr("导入", "Import")}
                  <input
                    accept="application/json,.json"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) void importStudioData(file);
                      event.target.value = "";
                    }}
                    type="file"
                  />
                </label>
              </div>
              {studio && <button className="secondary wide" onClick={openStudioSettings} type="button">{tr("配置 MFQ Server 连接", "Configure MFQ Server connection")}</button>}
            </section>
          </div>
          <footer>
            <button onClick={() => setSettingsDraft({
              ...modeTemplateSettings({
                ...DEFAULT_SETTINGS,
                language: settingsDraft.language,
                theme: settingsDraft.theme,
                playbackEnabled: settingsDraft.playbackEnabled,
              }, active?.mode ?? mode, runtime, realtime),
              inheritModelDefaults: true,
            })} type="button">{tr("恢复默认", "Reset")}</button>
            <button className="primary" onClick={saveSettings} type="button">{tr("应用", "Apply")}</button>
          </footer>
        </aside>
      </>}

      {modelBrowserOpen && modelBrowser && <div className="dialog-backdrop"><section className="studio-dialog model-browser-dialog"><header><div><h2>{tr("选择模型文件夹", "Choose model folder")}</h2><p>{tr("浏览 MFQ Server 所在设备上的文件夹。", "Browse folders on the MFQ Server host.")}</p></div><button onClick={() => setModelBrowserOpen(false)} type="button">×</button></header><form className="model-browser-location" onSubmit={jumpToModelDirectory}><button disabled={busy || !modelBrowser.current_id} onClick={() => void openModelDirectory(modelBrowser.parent_id)} type="button">{tr("上一级", "Up")}</button><input aria-label={tr("当前目录", "Current directory")} onChange={(event) => setModelDirectoryPath(event.target.value)} placeholder={tr("输入服务器上的完整目录", "Enter a full directory on the server")} spellCheck={false} value={modelDirectoryPath} /><button disabled={busy || !modelDirectoryPath.trim()} type="submit">{tr("前往", "Go")}</button>{modelBrowser.current_id && <span>{modelBrowser.model_file_count} MFQ</span>}</form><div className="model-browser-list">{modelBrowser.data.map((directory) => <button disabled={busy} key={directory.id} onClick={() => void openModelDirectory(directory.id)} type="button"><Icon name="folder" /><span>{directory.name}</span>{directory.model_file_count > 0 && <b>{directory.model_file_count} MFQ</b>}</button>)}{modelBrowser.data.length === 0 && <p>{tr("这个文件夹中没有子文件夹。", "This folder has no subfolders.")}</p>}</div><footer><button onClick={() => setModelBrowserOpen(false)} type="button">{tr("取消", "Cancel")}</button><button className="primary" disabled={busy || !modelBrowser.current_id} onClick={() => void registerCurrentModelDirectory()} type="button">{tr("使用此文件夹", "Use this folder")}</button></footer></section></div>}
      {studioOpen && studioDraft && <div className="dialog-backdrop"><form className="studio-dialog" onSubmit={saveStudioSettings}><header><div><h2>{tr("Runtime 连接", "Runtime connection")}</h2><p>{tr("MFQ Studio 关闭后，MFQ Server 会继续运行。", "MFQ Server keeps running after MFQ Studio closes.")}</p></div><button onClick={() => setStudioOpen(false)} type="button">×</button></header><div className="segmented">{(["local", "remote"] as const).map((item) => <button aria-pressed={studioDraft.mode === item} key={item} onClick={() => setStudioDraft((current) => current && ({ ...current, mode: item }))} type="button">{item === "local" ? "Local MFQ Server" : "Remote MFQ Server"}</button>)}</div>{studioDraft.mode === "local" ? <><label><span>MFQ Server port</span><input max={65535} min={1} onChange={(event) => setStudioDraft((current) => current && ({ ...current, local_service_port: Number(event.target.value) }))} required type="number" value={studioDraft.local_service_port} /></label></> : <><label><span>Remote MFQ Server URL</span><input onChange={(event) => setStudioDraft((current) => current && ({ ...current, remote_url: event.target.value }))} required type="url" value={studioDraft.remote_url} /></label><label><span>Remote MFQ Server API key</span><input autoComplete="off" onChange={(event) => setStudioToken(event.target.value)} placeholder={tr("保存在系统凭据库", "Stored in the system credential vault")} type="password" value={studioToken} /></label></>}<div className="dialog-status"><span className={studio?.reachable ? "online" : "offline"} />{studio?.reachable ? `${tr("已连接", "Connected")}: ${studio.service_url}` : tr("MFQ Server 离线", "MFQ Server is offline")}</div><footer><button onClick={() => setStudioOpen(false)} type="button">{tr("取消", "Cancel")}</button><button className="primary" disabled={busy} type="submit">{tr("应用", "Apply")}</button></footer></form></div>}

      {roleEditor && <div className="dialog-backdrop role-dialog-backdrop" onMouseDown={(event) => {
        if (event.currentTarget === event.target) setRoleEditor(null);
      }}>
        <form className="role-dialog" onSubmit={(event) => void saveRole(event)}>
          <header>
            <div>
              <p>{tr("角色", "Assistant")}</p>
              <h2>{roleEditor.roleId === "new" ? tr("新建角色", "New role") : tr("编辑角色", "Edit role")}</h2>
            </div>
            <button aria-label={tr("关闭", "Close")} onClick={() => setRoleEditor(null)} type="button">×</button>
          </header>
          <div className="role-dialog-scroll">
            <section className="role-identity-grid">
              <label className="role-icon-field">
                <span>{tr("图标", "Icon")}</span>
                <div><output>{roleEditor.icon || roleEditor.name.slice(0, 1).toLocaleUpperCase()}</output><input maxLength={8} onChange={(event) => setRoleEditor((current) => current && ({ ...current, icon: event.target.value }))} placeholder="MFQ" value={roleEditor.icon} /></div>
              </label>
              <label>
                <span>{tr("名称", "Name")}</span>
                <input autoFocus maxLength={64} onChange={(event) => setRoleEditor((current) => current && ({ ...current, name: event.target.value }))} required value={roleEditor.name} />
              </label>
            </section>
            <section>
              <h3>{tr("对话默认值", "Conversation defaults")}</h3>
              <div className="role-form-grid">
                <label><span>{tr("模型", "Model")}</span><input onChange={(event) => setRoleEditor((current) => current && ({ ...current, model: event.target.value }))} value={roleEditor.model} /></label>
                <label><span>{tr("交互模式", "Interaction mode")}</span><select onChange={(event) => {
                  const nextMode = event.target.value as SessionMode;
                  setRoleEditor((current) => current && ({
                    ...current,
                    mode: nextMode,
                    settings: current.inheritGlobalSettings
                      ? {
                          ...presetSnapshot(settings.inheritModelDefaults
                            ? modeTemplateSettings(settings, nextMode, runtime, realtime)
                            : settings),
                          systemPrompt: current.settings.systemPrompt,
                        }
                      : current.settings,
                  }));
                }} value={roleEditor.mode}>{(["text", "voice", "full_duplex"] as SessionMode[]).map((item) => <option key={item} value={item}>{MODE_LABELS[item][english ? 1 : 0]}</option>)}</select></label>
                <label><span>{tr("上下文 tokens", "Context tokens")}</span><input min={512} onChange={(event) => setRoleEditor((current) => current && ({ ...current, contextSize: Number(event.target.value) }))} step={512} type="number" value={roleEditor.contextSize} /></label>
              </div>
            </section>
            <section>
              <h3>{tr("系统提示词", "System prompt")}</h3>
              <label><textarea onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, systemPrompt: event.target.value } }))} placeholder={tr("该角色开始新会话时使用的默认提示词", "Default instructions for new sessions with this role")} rows={6} value={roleEditor.settings.systemPrompt} /></label>
            </section>
            <section>
              <div className="role-section-heading">
                <h3>{tr("推理参数", "Inference parameters")}</h3>
                <label className="role-inheritance-toggle">
                  <input
                    checked={roleEditor.inheritGlobalSettings}
                    onChange={(event) => {
                      const enabled = event.target.checked;
                      setRoleEditor((current) => current && ({
                        ...current,
                        inheritGlobalSettings: enabled,
                        settings: enabled
                          ? {
                              ...presetSnapshot(settings.inheritModelDefaults
                                ? modeTemplateSettings(settings, current.mode, runtime, realtime)
                                : settings),
                              systemPrompt: current.settings.systemPrompt,
                            }
                          : current.settings,
                      }));
                    }}
                    type="checkbox"
                  />
                  <span>{tr("随全局设置", "Use global settings")}</span>
                </label>
              </div>
              <fieldset className="role-inherited-fields" disabled={roleEditor.inheritGlobalSettings}>
                <div className="role-form-grid">
                  <label><span>{tr("最大生成 tokens", "Maximum output tokens")}</span><input min={1} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, maxTokens: Number(event.target.value) } }))} type="number" value={roleEditor.settings.maxTokens} /></label>
                  <label><span>Temperature</span><input max={2} min={0} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, temperature: Number(event.target.value) } }))} step={0.01} type="number" value={roleEditor.settings.temperature} /></label>
                  <label><span>Top P</span><input max={1} min={0} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, topP: Number(event.target.value) } }))} step={0.01} type="number" value={roleEditor.settings.topP} /></label>
                  <label><span>Top K</span><input min={0} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, topK: Number(event.target.value) } }))} type="number" value={roleEditor.settings.topK} /></label>
                  <label><span>Repetition penalty</span><input min={0} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, repetitionPenalty: Number(event.target.value) } }))} step={0.01} type="number" value={roleEditor.settings.repetitionPenalty} /></label>
                  <label><span>Presence penalty</span><input max={2} min={-2} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, presencePenalty: Number(event.target.value) } }))} step={0.01} type="number" value={roleEditor.settings.presencePenalty} /></label>
                  <label><span>Frequency penalty</span><input max={2} min={-2} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, frequencyPenalty: Number(event.target.value) } }))} step={0.01} type="number" value={roleEditor.settings.frequencyPenalty} /></label>
                  <label><span>Seed</span><input min={0} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, seed: event.target.value ? Number(event.target.value) : null } }))} placeholder={tr("随机", "Random")} type="number" value={roleEditor.settings.seed ?? ""} /></label>
                  <label><span>{tr("思考档位", "Reasoning effort")}</span><select onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, reasoningEffort: event.target.value } }))} value={roleEditor.settings.reasoningEffort}><option value="">{tr("自动", "Auto")}</option>{reasoningValues.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                </div>
                <div className="role-checks">
                  <label><input checked={thinkingSupported && roleEditor.settings.enableThinking} disabled={!thinkingSupported} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, enableThinking: event.target.checked } }))} type="checkbox" /><span>{tr("默认开启思考", "Enable thinking by default")}</span></label>
                  <label><input checked={roleEditor.settings.excludeReasoning} onChange={(event) => setRoleEditor((current) => current && ({ ...current, settings: { ...current.settings, excludeReasoning: event.target.checked } }))} type="checkbox" /><span>{tr("排除历史思考", "Exclude reasoning history")}</span></label>
                </div>
              </fieldset>
            </section>
          </div>
          <footer><button onClick={() => setRoleEditor(null)} type="button">{tr("取消", "Cancel")}</button><button className="primary" disabled={busy || !roleEditor.name.trim()} type="submit">{tr("保存角色", "Save role")}</button></footer>
        </form>
      </div>}
    </div>
  );
}
