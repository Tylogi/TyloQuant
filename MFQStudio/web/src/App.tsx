import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  ContentPart,
  Message,
  RealtimeCapabilities,
  RealtimeFrame,
  RuntimeCapabilities,
  RuntimeModel,
  RuntimeStatus,
  SamplingParams,
  Session,
  SessionMode,
  api,
  setApiBaseUrl,
  streamResponse,
} from "./api";
import { Markdown } from "./Markdown";
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
  startLocalStudio,
  studioStatus,
} from "./studio";

interface LiveOutput {
  reasoning: string;
  text: string;
  tools: string[];
}

type ViewName = "chat" | "monitor";
type UiLanguage = "system" | "zh-CN" | "en";
type PresetName = "precise" | "balanced" | "creative" | "custom";

interface GenerationSettings {
  language: UiLanguage;
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

interface VoiceMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  text: string;
  audioId?: string;
  created_at: string;
}

interface EditDraft {
  messageId: string;
  text: string;
  reasoning: string;
}

const SETTINGS_KEY = "mfq.studio.generation.v1";
const VOICE_HISTORY_KEY = "mfq.studio.voice-history.v1";
const MODE_LABELS: Record<SessionMode, [string, string]> = {
  text: ["文本", "Text"],
  voice: ["语音", "Voice"],
  full_duplex: ["全双工", "Full duplex"],
};
const DEFAULT_SETTINGS: GenerationSettings = {
  language: "system",
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
    fullDuplex: mode === "full_duplex",
    preset: "custom",
    seed: null,
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
): part is Extract<ContentPart, { type: "image" | "audio" | "generated_audio" }> {
  return part.type === "image" || part.type === "audio" || part.type === "generated_audio";
}

function formatNumber(value: unknown, digits = 0): string {
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(number)
    : "--";
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

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [voiceMessages, setVoiceMessages] = useState<VoiceMessage[]>(loadVoiceHistory);
  const [models, setModels] = useState<RuntimeModel[]>([]);
  const [model, setModel] = useState("default");
  const [mode, setMode] = useState<SessionMode>("text");
  const [view, setView] = useState<ViewName>("chat");
  const [draft, setDraft] = useState("");
  const [live, setLive] = useState<LiveOutput | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [realtime, setRealtime] = useState<RealtimeCapabilities | null>(null);
  const [realtimeAvailable, setRealtimeAvailable] = useState(false);
  const [metricSeries, setMetricSeries] = useState<number[]>([]);
  const [settings, setSettings] = useState<GenerationSettings>(loadSettings);
  const [settingsDraft, setSettingsDraft] = useState<GenerationSettings>(settings);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [contextSize, setContextSize] = useState(32768);
  const [studio, setStudio] = useState<StudioStatus | null>(null);
  const [studioDraft, setStudioDraft] = useState<StudioConfig | null>(null);
  const [studioOpen, setStudioOpen] = useState(false);
  const [editDraft, setEditDraft] = useState<EditDraft | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [voiceLevel, setVoiceLevel] = useState(0);
  const [liveVoiceText, setLiveVoiceText] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const activeIdRef = useRef<string | null>(null);
  const voiceRef = useRef<RealtimeAudioController | null>(null);
  const lastMetricId = useRef("");
  const hadStoredSettings = useRef(Boolean(localStorage.getItem(SETTINGS_KEY)));
  const appliedModeTemplate = useRef("");

  const english =
    settings.language === "en" ||
    (settings.language === "system" && !navigator.language.toLowerCase().startsWith("zh"));
  const tr = useCallback((zh: string, en: string) => (english ? en : zh), [english]);
  const active = useMemo(
    () => sessions.find((session) => session.id === activeId) ?? null,
    [activeId, sessions],
  );
  const currentVoiceMessages = useMemo(
    () => voiceMessages.filter((message) => message.sessionId === activeId),
    [activeId, voiceMessages],
  );
  const reasoningValues = useMemo(() => {
    const values = runtime?.chat_template_capabilities?.reasoning_effort?.values;
    return Array.isArray(values) ? values : [];
  }, [runtime]);

  const refreshSessions = useCallback(async (preferredId?: string) => {
    const next = await api.listSessions();
    setSessions(next);
    setActiveId((current) => {
      const wanted = preferredId ?? current;
      if (wanted && next.some((session) => session.id === wanted)) return wanted;
      return next[0]?.id ?? null;
    });
  }, []);

  const refreshRuntime = useCallback(async (quiet = true) => {
    try {
      const status = await api.runtimeStatus();
      setRuntime(status);
      const request = status.last_request;
      const requestId = String(request?.id ?? "");
      const decode = Number(request?.decode_tps);
      if (requestId && requestId !== lastMetricId.current && Number.isFinite(decode)) {
        lastMetricId.current = requestId;
        setMetricSeries((current) => [...current, decode].slice(-32));
      }
      const currentContext = Number(status.max_context);
      if (Number.isFinite(currentContext) && currentContext > 0) {
        setContextSize(Math.floor(currentContext));
      }
    } catch (cause) {
      if (!quiet) setError(errorMessage(cause));
    }
  }, []);

  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  useEffect(() => {
    if (!active || !runtime || voiceRef.current?.active) return;
    const key = `${runtime.model ?? model}:${active.mode}`;
    if (appliedModeTemplate.current === key) return;
    appliedModeTemplate.current = key;
    void voiceRef.current?.setFullDuplex(active.mode === "full_duplex");
    setSettings((current) => modeTemplateSettings(current, active.mode, runtime, realtime));
  }, [active, model, realtime, runtime]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    voiceRef.current?.setPlayback(settings.playbackEnabled);
  }, [settings]);

  useEffect(() => {
    localStorage.setItem(VOICE_HISTORY_KEY, JSON.stringify(voiceMessages.slice(-200)));
  }, [voiceMessages]);

  useEffect(() => {
    voiceRef.current = new RealtimeAudioController(
      {
        onState: setVoiceState,
        onLevel: setVoiceLevel,
        onText: setLiveVoiceText,
        onError: (message) => setError(message),
        onTurn: ({ text, audio }) => {
          const sessionId = activeIdRef.current;
          if (!sessionId) return;
          const id = crypto.randomUUID();
          const audioId = audio ? `voice-${id}` : undefined;
          const persist = async () => {
            if (audio && audioId) await saveVoiceClip(audioId, audio);
            setVoiceMessages((current) => [
              ...current,
              {
                id,
                sessionId,
                role: "assistant",
                text,
                audioId,
                created_at: new Date().toISOString(),
              },
            ]);
          };
          void persist().catch((cause) => setError(errorMessage(cause)));
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
          if (current) setStudio(status);
        }
        const results = await Promise.allSettled([
          api.runtimeCapabilities(),
          api.runtimeModels(),
          api.runtimeStatus(),
          api.realtimeCapabilities(),
          api.listSessions(),
        ]);
        if (!current) return;
        if (results[0].status === "fulfilled") {
          setCapabilities(results[0].value);
          setModel(results[0].value.model);
        }
        if (results[1].status === "fulfilled") setModels(results[1].value);
        if (results[2].status === "fulfilled") {
          setRuntime(results[2].value);
          if (!hadStoredSettings.current && results[2].value.sampling_defaults) {
            const defaults = results[2].value.sampling_defaults;
            setSettings((current) => ({
              ...current,
              maxTokens: Number(defaults.max_tokens ?? current.maxTokens),
              temperature: Number(defaults.temperature ?? current.temperature),
              topP: Number(defaults.top_p ?? current.topP),
              topK: Number(defaults.top_k ?? current.topK),
              repetitionPenalty: Number(
                defaults.repetition_penalty ?? current.repetitionPenalty,
              ),
              presencePenalty: Number(defaults.presence_penalty ?? current.presencePenalty),
              frequencyPenalty: Number(
                defaults.frequency_penalty ?? current.frequencyPenalty,
              ),
              preset: "custom",
            }));
            hadStoredSettings.current = true;
          }
        }
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
      return;
    }
    let current = true;
    api
      .listMessages(activeId)
      .then((next) => current && setMessages(next))
      .catch((cause) => current && setError(errorMessage(cause)));
    return () => {
      current = false;
      if (voiceRef.current?.active) void voiceRef.current.stop();
    };
  }, [activeId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: busy ? "smooth" : "auto" });
  }, [messages, currentVoiceMessages, live, busy]);

  useEffect(() => {
    if (capabilities && !capabilities.model_capabilities.features.full_duplex) {
      setMode("text");
    }
  }, [capabilities]);

  function samplingParams(): SamplingParams {
    return {
      max_tokens: settings.maxTokens,
      temperature: settings.temperature,
      top_k: settings.topK,
      top_p: settings.topP,
      presence_penalty: settings.presencePenalty,
      frequency_penalty: settings.frequencyPenalty,
      repetition_penalty: settings.repetitionPenalty,
      seed: settings.seed,
      enable_thinking: settings.enableThinking,
      reasoning_effort: settings.reasoningEffort || null,
    };
  }

  function realtimeSessionConfig() {
    return {
      systemPrompt: settings.systemPrompt,
      temperature: settings.temperature,
      topP: settings.topP,
      topK: settings.topK,
      repetitionPenalty: settings.repetitionPenalty,
    };
  }

  async function createSession() {
    const selectedModel = model.trim();
    if (!selectedModel) return;
    setError(null);
    try {
      const created = await api.createSession(selectedModel, mode);
      setSessions((current) => [created, ...current]);
      setActiveId(created.id);
      setMessages([]);
      setView("chat");
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function deleteSession(session: Session) {
    if (busy || !window.confirm(tr("删除这个会话？", "Delete this session?"))) return;
    try {
      await api.deleteSession(session.id);
      setVoiceMessages((current) => current.filter((item) => item.sessionId !== session.id));
      await refreshSessions();
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function clearSessions() {
    if (busy || !sessions.length || !window.confirm(tr("清空全部会话？", "Clear all sessions?"))) {
      return;
    }
    try {
      for (const session of sessions) await api.deleteSession(session.id);
      setSessions([]);
      setMessages([]);
      setVoiceMessages([]);
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

  async function generate(session: Session, text: string, optimistic = true) {
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setBusy(true);
    setLive({ reasoning: "", text: "", tools: [] });
    if (optimistic) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "user",
          parts: [{ type: "text", text }],
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
          input: [{ type: "text", text }],
          sampling: samplingParams(),
          system_prompt: settings.systemPrompt || null,
          include_reasoning_history: !settings.excludeReasoning,
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
        const [persisted, updated] = await Promise.all([
          api.listMessages(session.id),
          api.getSession(session.id),
        ]);
        setMessages(persisted);
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
    if (!active || !text || busy) return;
    setDraft("");
    if (active.mode !== "text" && voiceRef.current) {
      await voiceRef.current.submitText(text, realtimeSessionConfig());
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
    await generate(active, text);
  }

  async function saveEdit(message: Message) {
    if (!active || !editDraft || busy) return;
    const text = editDraft.text.trim();
    const reasoning = editDraft.reasoning.trim();
    if (message.role === "user" && !text) return;
    if (message.role === "assistant" && !text && !reasoning) return;
    setBusy(true);
    try {
      const branch = await api.forkSession(
        active.id,
        message.id,
        false,
        active.title ? `${active.title} · edit` : null,
      );
      const parts: ContentPart[] = [];
      if (reasoning) parts.push({ type: "reasoning", text: reasoning });
      if (text) parts.push({ type: "text", text });
      const appended = await api.appendMessage(branch.id, branch.revision, message.role, parts);
      setSessions((current) => [appended.session, ...current]);
      setActiveId(appended.session.id);
      setMessages([...(await api.listMessages(appended.session.id))]);
      setEditDraft(null);
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
    for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
      if (messages[cursor].role === "user") {
        user = messages[cursor];
        break;
      }
    }
    if (!user) return;
    const text = textParts(user).text.trim();
    if (!text) return;
    setBusy(true);
    try {
      const branch = await api.forkSession(
        active.id,
        user.id,
        false,
        active.title ? `${active.title} · retry` : null,
      );
      setSessions((current) => [branch, ...current]);
      setActiveId(branch.id);
      setMessages(await api.listMessages(branch.id));
      setBusy(false);
      await generate(branch, text);
    } catch (cause) {
      setBusy(false);
      setError(errorMessage(cause));
    }
  }

  async function copyMessage(message: Message) {
    const parts = textParts(message);
    await navigator.clipboard.writeText([parts.reasoning, parts.text].filter(Boolean).join("\n\n"));
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
    await voiceRef.current.toggleCapture(realtimeSessionConfig());
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
    setSettingsDraft(settings);
    const current = Number(runtime?.max_context);
    setContextSize(Number.isFinite(current) && current > 0 ? current : contextSize);
    setSettingsOpen(true);
  }

  function applyPreset(name: Exclude<PresetName, "custom">) {
    setSettingsDraft((current) => ({ ...current, ...PRESETS[name], preset: name }));
  }

  function saveSettings() {
    setSettings(settingsDraft);
    setSettingsOpen(false);
  }

  async function reloadRuntime() {
    if (busy || voiceRef.current?.active) return;
    if (!window.confirm(tr(`以 ${formatNumber(contextSize)} token 上下文重载模型？`, `Reload the model with a ${formatNumber(contextSize)} token context?`))) return;
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
      setApiBaseUrl(status.service_url);
      setStudio(status);
      setStudioOpen(false);
      setMessages([]);
      setActiveId(null);
      await refreshSessions();
      const [nextCapabilities, nextModels] = await Promise.all([
        api.runtimeCapabilities(),
        api.runtimeModels(),
      ]);
      setCapabilities(nextCapabilities);
      setModels(nextModels);
      const nextRealtime = await api.realtimeCapabilities();
      setRealtime(nextRealtime);
      setRealtimeAvailable(nextRealtime.available === true);
      appliedModeTemplate.current = "";
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  const last = runtime?.last_request;
  const contextTokens = Number(last?.prompt_tokens || 0) + Number(last?.completion_tokens || 0);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <img src="/mfq-mark.svg" alt="" />
          <div><strong>MFQ</strong><span>Studio</span></div>
        </div>
        <button className="new-session" onClick={createSession} type="button">＋ {tr("新会话", "New session")}</button>
        <nav className="primary-nav">
          <button className={view === "chat" ? "active" : ""} onClick={() => setView("chat")} type="button">◫ {tr("对话", "Chat")}</button>
          <button className={view === "monitor" ? "active" : ""} onClick={() => setView("monitor")} type="button">⌁ {tr("监控", "Monitor")}<span>{formatNumber(runtime?.active_requests || 0)}</span></button>
        </nav>
        <div className="session-create">
          <label htmlFor="new-model">{tr("新会话模型", "Model for new sessions")}</label>
          <select id="new-model" value={model} onChange={(event) => setModel(event.target.value)}>
            {(models.length ? models : [{ id: model }]).map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
          </select>
          <div className="mode-picker">
            {(["text", "voice", "full_duplex"] as SessionMode[]).map((item) => {
              const feature = capabilities?.model_capabilities.features;
              const disabled = item === "voice" ? !feature?.audio_input : item === "full_duplex" ? !feature?.full_duplex : false;
              return <button aria-pressed={mode === item} disabled={disabled} key={item} onClick={() => setMode(item)} type="button">{MODE_LABELS[item][english ? 1 : 0]}</button>;
            })}
          </div>
        </div>
        <div className="history-heading"><span>{tr("最近", "Recent")}</span><button onClick={clearSessions} title={tr("清空历史", "Clear history")} type="button">⌫</button></div>
        <nav className="session-list" aria-label="Sessions">
          {loading && <span className="empty-note">{tr("加载中…", "Loading…")}</span>}
          {!loading && sessions.length === 0 && <span className="empty-note">{tr("还没有会话", "No saved sessions")}</span>}
          {sessions.map((session) => (
            <div className={`session-row ${activeId === session.id ? "active" : ""}`} key={session.id}>
              {renamingId === session.id ? (
                <input autoFocus onBlur={() => void saveRename(session)} onChange={(event) => setRenameValue(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void saveRename(session); if (event.key === "Escape") setRenamingId(null); }} value={renameValue} />
              ) : (
                <button className="session-main" onClick={() => { setActiveId(session.id); setView("chat"); }} type="button"><span>{session.title || tr("未命名会话", "Untitled session")}</span><small>{MODE_LABELS[session.mode][english ? 1 : 0]} · r{session.revision}</small></button>
              )}
              <div className="session-actions">
                <button onClick={() => { setRenamingId(session.id); setRenameValue(session.title || ""); }} title={tr("重命名", "Rename")} type="button">✎</button>
                <button onClick={() => void deleteSession(session)} title={tr("删除", "Delete")} type="button">×</button>
              </div>
            </div>
          ))}
        </nav>
        <div className={`connection-card ${studio?.reachable === false ? "offline" : ""}`}><span /><div><strong>{studio?.reachable === false ? tr("离线", "Offline") : tr("在线", "Online")}</strong><small>{studio?.service_url || window.location.origin}</small></div></div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="topbar-model"><span>{active?.model || model}</span><small>{capabilities?.model_type || runtime?.model_type || "runtime"}</small></div>
          <div className="topbar-actions">
            {capabilities && <div className="capabilities">{CAPABILITY_LABELS.filter(([feature]) => capabilities.model_capabilities.features[feature]).map(([feature, label]) => <span className={feature === "full_duplex" && !realtimeAvailable ? "muted" : ""} key={feature}>{label[english ? 1 : 0]}</span>)}</div>}
            <div className="quick-metrics"><span><b>{last?.ttft_ms == null ? "--" : `${formatNumber(last.ttft_ms, 1)} ms`}</b> TTFT</span><span><b>{last ? formatNumber(contextTokens) : "--"}</b> context</span><span><b>{last?.decode_tps == null ? "--" : formatNumber(last.decode_tps, 1)}</b> tok/s</span></div>
            <button disabled={!active} onClick={exportConversation} title={tr("导出会话", "Export chat")} type="button">⇩</button>
            <button onClick={openSettings} title={tr("推理设置", "Inference settings")} type="button">⚙</button>
          </div>
        </header>

        {view === "chat" ? (
          <section className="chat-view">
            <div className="message-scroller">
              <div className="message-list" aria-live="polite">
                {!active && <div className="welcome"><img src="/mfq-mark.svg" alt="" /><h1>MFQ Studio</h1><p>{tr("创建会话后即可开始本地推理。", "Create a session to start local inference.")}</p><div className="prompt-grid"><button onClick={() => setDraft(tr("介绍一下这个模型。", "Introduce this model."))} type="button">{tr("介绍模型", "Introduce the model")}</button><button onClick={() => setDraft(tr("写一段 Python 示例。", "Write a Python example."))} type="button">{tr("代码示例", "Code example")}</button></div></div>}
                {messages.map((message) => {
                  const parts = textParts(message);
                  const editing = editDraft?.messageId === message.id;
                  return <article className={`message message-${message.role}`} key={message.id}>
                    <div className="message-avatar">{message.role === "assistant" ? <img src="/mfq-mark.svg" alt="MFQ" /> : <span>{message.role === "user" ? tr("你", "You") : message.role}</span>}</div>
                    <div className="message-body">
                      <div className="message-meta"><strong>{message.role === "assistant" ? "MFQ" : message.role === "user" ? tr("你", "You") : message.role}</strong><span>{new Date(message.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span></div>
                      {editing ? <div className="message-editor">{message.role === "assistant" && <textarea aria-label={tr("思考过程", "Reasoning")} onChange={(event) => setEditDraft((current) => current && ({ ...current, reasoning: event.target.value }))} placeholder={tr("思考过程", "Reasoning")} value={editDraft.reasoning} />}<textarea aria-label={tr("消息", "Message")} onChange={(event) => setEditDraft((current) => current && ({ ...current, text: event.target.value }))} value={editDraft.text} /><div><button onClick={() => setEditDraft(null)} type="button">{tr("取消", "Cancel")}</button><button className="primary" onClick={() => void saveEdit(message)} type="button">{tr("保存到新分支", "Save as branch")}</button></div></div> : <>{parts.reasoning && <details className="reasoning"><summary>{tr("思考过程", "Reasoning")}</summary><Markdown text={parts.reasoning} /></details>}{parts.text && <Markdown text={parts.text} />}{message.parts.filter(isMediaPart).map((part, index) => <div className="media-part" key={index}>{part.type === "image" ? "Image" : part.type === "audio" ? "Audio" : "Generated audio"} · {part.media.mime_type} · {formatNumber(part.media.byte_size)} bytes</div>)}{message.parts.filter((part) => part.type === "tool_call" || part.type === "tool_result").map((part, index) => <pre className="tool-call" key={index}>{part.type === "tool_call" ? `${part.name}(${JSON.stringify(part.arguments, null, 2)})` : JSON.stringify(part.result, null, 2)}</pre>)}</>}
                      {!editing && <div className="message-actions"><button onClick={() => void copyMessage(message)} type="button">{tr("复制", "Copy")}</button>{(message.role === "user" || message.role === "assistant") && <button onClick={() => setEditDraft({ messageId: message.id, ...parts })} type="button">{tr("编辑", "Edit")}</button>}{message.role === "assistant" && <button onClick={() => void regenerate(message)} type="button">{tr("重新生成", "Regenerate")}</button>}</div>}
                    </div>
                  </article>;
                })}
                {currentVoiceMessages.map((message) => <article className={`message message-${message.role}`} key={message.id}><div className="message-avatar">{message.role === "assistant" ? <img src="/mfq-mark.svg" alt="MFQ" /> : <span>{tr("你", "You")}</span>}</div><div className="message-body"><div className="message-meta"><strong>{message.role === "assistant" ? "MFQ" : tr("你", "You")}</strong><span>{tr("语音", "Voice")}</span></div>{message.text && <Markdown text={message.text} />}{message.audioId && <AudioClip audioId={message.audioId} />}</div></article>)}
                {liveVoiceText && <article className="message message-assistant live-message"><div className="message-avatar"><img src="/mfq-mark.svg" alt="MFQ" /></div><div className="message-body"><div className="message-meta"><strong>MFQ</strong><span>{tr("生成中", "Generating")}</span></div><Markdown live text={liveVoiceText} /></div></article>}
                {live && <article className="message message-assistant live-message"><div className="message-avatar"><img src="/mfq-mark.svg" alt="MFQ" /></div><div className="message-body"><div className="message-meta"><strong>MFQ</strong><span>{tr("生成中", "Generating")}</span></div>{live.reasoning && <details className="reasoning" open><summary>{tr("正在思考", "Thinking")}</summary><Markdown live text={live.reasoning} /></details>}{live.text && <Markdown live text={live.text} />}{live.tools.map((tool, index) => <pre className="tool-call" key={index}>{tool}</pre>)}{!live.reasoning && !live.text && live.tools.length === 0 && <span className="thinking"><i /><i /><i /></span>}</div></article>}
                <div ref={endRef} />
              </div>
            </div>
            {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError(null)} type="button">×</button></div>}
            <div className="composer-region">
              <form className="composer" onSubmit={send}>
                <textarea aria-label={tr("消息", "Message")} disabled={!active || busy} maxLength={32768} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} placeholder={active ? tr("向模型发送消息", "Message MFQ") : tr("请先创建会话", "Create a session first")} rows={1} value={draft} />
                <div className="composer-toolbar">
                  {realtimeAvailable && <select aria-label={tr("交互模式", "Interaction mode")} disabled={!active || busy || voiceState !== "idle"} onChange={(event) => void selectInteractionMode(event.target.value as SessionMode)} value={active?.mode ?? mode}>{(["text", "voice", "full_duplex"] as SessionMode[]).map((item) => { const feature = capabilities?.model_capabilities.features; const disabled = item === "voice" ? !feature?.audio_input : item === "full_duplex" ? !feature?.full_duplex : false; return <option disabled={disabled} key={item} value={item}>{MODE_LABELS[item][english ? 1 : 0]}</option>; })}</select>}
                  {realtimeAvailable && <button aria-label={tr("语音输入", "Voice input")} aria-pressed={voiceState !== "idle" && voiceState !== "error"} className="voice-button" disabled={!active || active.mode === "text" || busy} onClick={() => void toggleVoice()} style={{ "--voice-level": voiceLevel } as React.CSSProperties} title={active?.mode === "text" ? tr("请先选择语音或全双工模式", "Select voice or full duplex mode first") : voiceState === "processing" ? tr("语音处理中", "Processing voice") : tr("语音输入", "Voice input")} type="button"><span /></button>}
                  {realtimeAvailable && active?.mode !== "text" && <button aria-pressed={settings.playbackEnabled} onClick={() => setSettings((current) => ({ ...current, playbackEnabled: !current.playbackEnabled }))} title={tr("语音播放", "Voice playback")} type="button">{settings.playbackEnabled ? "🔊" : "🔇"}</button>}
                  {active?.mode === "text" && <button aria-pressed={settings.enableThinking} onClick={() => setSettings((current) => ({ ...current, enableThinking: !current.enableThinking }))} type="button">◉ {tr("思考", "Thinking")}</button>}
                  {active?.mode === "text" && settings.enableThinking && reasoningValues.length > 0 && <select aria-label={tr("思考档位", "Reasoning effort")} onChange={(event) => setSettings((current) => ({ ...current, reasoningEffort: event.target.value }))} value={settings.reasoningEffort}><option value="">{tr("标准", "Standard")}</option>{reasoningValues.map((value) => <option key={value} value={value}>{value}</option>)}</select>}
                  <span className="composer-hint">{voiceState !== "idle" ? voiceState : tr("Enter 发送 · Shift+Enter 换行", "Enter to send · Shift+Enter for newline")}</span>
                  {busy ? <button className="send-button stop" onClick={() => abortRef.current?.abort()} type="button">■</button> : <button className="send-button" disabled={!active || !draft.trim()} type="submit">↑</button>}
                </div>
              </form>
              <p>{tr("模型输出可能存在错误，请核对重要信息。", "Model output may be inaccurate. Verify important information.")}</p>
            </div>
          </section>
        ) : (
          <section className="monitor-view">
            <div className="monitor-heading"><div><p>Runtime</p><h1>{tr("服务监控", "Runtime monitor")}</h1><span>{runtime ? tr("状态已连接", "Status connected") : tr("等待状态数据", "Waiting for status")}</span></div><button onClick={() => void refreshRuntime(false)} type="button">↻</button></div>
            <div className="metric-grid"><article><span>Prefill</span><strong>{formatNumber(last?.prefill_tps, 1)}</strong><small>tokens / second</small></article><article><span>Decode</span><strong>{formatNumber(last?.decode_tps, 1)}</strong><small>tokens / second</small></article><article><span>TTFT</span><strong>{formatNumber(last?.ttft_ms, 1)}</strong><small>milliseconds</small></article><article><span>{tr("请求", "Requests")}</span><strong>{formatNumber(runtime?.total_requests || 0)}</strong><small>{formatNumber(runtime?.active_requests || 0)} active</small></article><article><span>Tokens</span><strong>{formatNumber(Number(runtime?.total_prompt_tokens || 0) + Number(runtime?.total_completion_tokens || 0))}</strong><small>prompt + completion</small></article></div>
            <div className="monitor-grid"><section className="monitor-panel chart-panel"><div className="panel-heading"><div><h2>{tr("生成吞吐", "Decode throughput")}</h2><p>{tr("最近请求的 decode tokens/s", "Decode tokens/s for recent requests")}</p></div><b>{formatNumber(last?.decode_tps, 1)} tok/s</b></div><RuntimeChart values={metricSeries} /></section><section className="monitor-panel"><div className="panel-heading"><div><h2>Runtime</h2><p>{tr("当前服务实例", "Current runtime instance")}</p></div></div><dl><div><dt>{tr("模型", "Model")}</dt><dd>{runtime?.model || model}</dd></div><div><dt>{tr("架构", "Architecture")}</dt><dd>{runtime?.model_type || "--"}</dd></div><div><dt>{tr("上下文", "Context")}</dt><dd>{formatNumber(runtime?.max_context)}</dd></div><div><dt>{tr("运行时间", "Uptime")}</dt><dd>{formatDuration(runtime?.uptime_seconds)}</dd></div><div><dt>{tr("失败请求", "Failed")}</dt><dd>{formatNumber(runtime?.failed_requests || 0)}</dd></div></dl></section></div>
            <section className="monitor-panel request-panel"><div className="panel-heading"><div><h2>{tr("最近请求", "Last request")}</h2><p>{last?.id || tr("还没有完成的请求", "No completed requests")}</p></div><b>{last?.finish_reason || (runtime?.active_requests ? "Running" : "Idle")}</b></div><div className="request-stats"><div><span>{tr("输入", "Input")}</span><strong>{formatNumber(last?.prompt_tokens)}</strong><small>tokens</small></div><div><span>{tr("输出", "Output")}</span><strong>{formatNumber(last?.completion_tokens)}</strong><small>tokens</small></div><div><span>Prefill</span><strong>{formatNumber(last?.prefill_tps, 1)}</strong><small>{formatNumber(last?.prefill_ms, 1)} ms</small></div><div><span>{tr("总耗时", "Total")}</span><strong>{formatNumber(last?.generation_ms, 1)}</strong><small>ms</small></div><div><span>{tr("结束原因", "Finish")}</span><strong>{last?.finish_reason || "--"}</strong><small>finish reason</small></div></div></section>
          </section>
        )}
      </main>

      {settingsOpen && <><div className="drawer-scrim" onClick={() => setSettingsOpen(false)} /><aside className="settings-panel"><header><div><p>Generation</p><h2>{tr("推理设置", "Inference settings")}</h2></div><button onClick={() => setSettingsOpen(false)} type="button">×</button></header><div className="settings-scroll"><section><h3>{tr("预设", "Presets")}</h3><div className="segmented">{(["precise", "balanced", "creative"] as const).map((name) => <button aria-pressed={settingsDraft.preset === name} key={name} onClick={() => applyPreset(name)} type="button">{name === "precise" ? tr("精确", "Precise") : name === "balanced" ? tr("均衡", "Balanced") : tr("创意", "Creative")}</button>)}</div></section><section><h3>{tr("上下文", "Context")}</h3><label><span>{tr("系统提示词", "System prompt")}</span><textarea onChange={(event) => setSettingsDraft((current) => ({ ...current, systemPrompt: event.target.value }))} rows={4} value={settingsDraft.systemPrompt} /></label><label className="check-field"><span><strong>{tr("排除历史思考", "Exclude reasoning history")}</strong><small>{tr("后续请求不再发送已保存的思考内容。", "Do not send saved reasoning in later requests.")}</small></span><input checked={settingsDraft.excludeReasoning} onChange={(event) => setSettingsDraft((current) => ({ ...current, excludeReasoning: event.target.checked }))} type="checkbox" /></label><label><span>{tr("上下文窗口 tokens", "Context window tokens")}</span><input max={Number(runtime?.context_capacity) || 1048576} min={512} onChange={(event) => setContextSize(Number(event.target.value))} step={512} type="number" value={contextSize} /></label><button className="secondary wide" disabled={busy} onClick={() => void reloadRuntime()} type="button">{tr("按此上下文重载模型", "Reload model with this context")}</button><label><span>{tr("最大生成 tokens", "Maximum output tokens")}</span><input max={65536} min={1} onChange={(event) => setSettingsDraft((current) => ({ ...current, maxTokens: Number(event.target.value) }))} type="number" value={settingsDraft.maxTokens} /></label></section><section><h3>{tr("采样", "Sampling")}</h3><label><span>Temperature <output>{settingsDraft.temperature.toFixed(2)}</output></span><input max={2} min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, temperature: Number(event.target.value), preset: "custom" }))} step={0.05} type="range" value={settingsDraft.temperature} /></label><label><span>Top P <output>{settingsDraft.topP.toFixed(2)}</output></span><input max={1} min={0.05} onChange={(event) => setSettingsDraft((current) => ({ ...current, topP: Number(event.target.value), preset: "custom" }))} step={0.05} type="range" value={settingsDraft.topP} /></label><label><span>Top K</span><input max={1024} min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, topK: Number(event.target.value), preset: "custom" }))} type="number" value={settingsDraft.topK} /></label><label><span>Seed</span><input min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, seed: event.target.value ? Number(event.target.value) : null }))} placeholder={tr("随机", "Random")} type="number" value={settingsDraft.seed ?? ""} /></label></section><section><h3>{tr("惩罚", "Penalties")}</h3>{([["Repetition", "repetitionPenalty", 0.5, 2, 0.01], ["Presence", "presencePenalty", -2, 2, 0.05], ["Frequency", "frequencyPenalty", -2, 2, 0.05]] as const).map(([label, key, min, max, step]) => <label key={key}><span>{label} <output>{settingsDraft[key].toFixed(2)}</output></span><input max={max} min={min} onChange={(event) => setSettingsDraft((current) => ({ ...current, [key]: Number(event.target.value), preset: "custom" }))} step={step} type="range" value={settingsDraft[key]} /></label>)}</section><section><h3>{tr("界面与连接", "Interface and connection")}</h3><label><span>{tr("界面语言", "Interface language")}</span><select onChange={(event) => setSettingsDraft((current) => ({ ...current, language: event.target.value as UiLanguage }))} value={settingsDraft.language}><option value="system">{tr("跟随系统", "System")}</option><option value="zh-CN">简体中文</option><option value="en">English</option></select></label>{studio && <button className="secondary wide" onClick={openStudioSettings} type="button">{tr("配置 MFQd 连接", "Configure MFQd connection")}</button>}</section></div><footer><button onClick={() => setSettingsDraft(modeTemplateSettings({ ...DEFAULT_SETTINGS, language: settingsDraft.language }, active?.mode ?? mode, runtime, realtime))} type="button">{tr("恢复默认", "Reset")}</button><button className="primary" onClick={saveSettings} type="button">{tr("应用", "Apply")}</button></footer></aside></>}

      {studioOpen && studioDraft && <div className="dialog-backdrop"><form className="studio-dialog" onSubmit={saveStudioSettings}><header><div><h2>{tr("Runtime 连接", "Runtime connection")}</h2><p>{tr("MFQ Studio 关闭后，MFQd 会继续运行。", "MFQd keeps running after MFQ Studio closes.")}</p></div><button onClick={() => setStudioOpen(false)} type="button">×</button></header><div className="segmented">{(["local", "remote"] as const).map((item) => <button aria-pressed={studioDraft.mode === item} key={item} onClick={() => setStudioDraft((current) => current && ({ ...current, mode: item }))} type="button">{item === "local" ? "Local MFQd" : "Remote MFQd"}</button>)}</div>{studioDraft.mode === "local" ? <><label><span>{tr("Runtime / realtime gateway URL", "Runtime / realtime gateway URL")}</span><input onChange={(event) => setStudioDraft((current) => current && ({ ...current, local_backend_url: event.target.value }))} required type="url" value={studioDraft.local_backend_url} /></label><label><span>MFQd port</span><input max={65535} min={1} onChange={(event) => setStudioDraft((current) => current && ({ ...current, local_service_port: Number(event.target.value) }))} required type="number" value={studioDraft.local_service_port} /></label><label><span>MFQd executable</span><input onChange={(event) => setStudioDraft((current) => current && ({ ...current, mfqd_executable: event.target.value || null }))} placeholder="mfqd from PATH" value={studioDraft.mfqd_executable ?? ""} /></label></> : <label><span>Remote MFQd URL</span><input onChange={(event) => setStudioDraft((current) => current && ({ ...current, remote_url: event.target.value }))} required type="url" value={studioDraft.remote_url} /></label>}<div className="dialog-status"><span className={studio?.reachable ? "online" : "offline"} />{studio?.reachable ? `${tr("已连接", "Connected")}: ${studio.service_url}` : tr("MFQd 离线", "MFQd is offline")}</div><footer><button onClick={() => setStudioOpen(false)} type="button">{tr("取消", "Cancel")}</button><button className="primary" disabled={busy} type="submit">{tr("应用", "Apply")}</button></footer></form></div>}
    </div>
  );
}
