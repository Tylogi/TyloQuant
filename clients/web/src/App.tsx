import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  ContentPart,
  Message,
  RealtimeFrame,
  RuntimeCapabilities,
  Session,
  SessionMode,
  api,
  setApiBaseUrl,
  streamResponse,
} from "./api";
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

const MODE_LABELS: Record<SessionMode, string> = {
  text: "Text",
  voice: "Voice",
  full_duplex: "Full duplex",
};

const CAPABILITY_LABELS: Array<[
  keyof RuntimeCapabilities["model_capabilities"]["features"],
  string,
]> = [
  ["text", "Text"],
  ["image_input", "Image"],
  ["video_input", "Video"],
  ["audio_input", "Audio in"],
  ["audio_output", "Audio out"],
  ["full_duplex", "Full duplex"],
];

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return `${error.code}: ${error.message}`;
  if (error instanceof Error) return error.message;
  return "Unknown error";
}

function partView(part: ContentPart, index: number) {
  if (part.type === "text" || part.type === "transcript") {
    return <p key={index}>{part.text}</p>;
  }
  if (part.type === "reasoning") {
    return (
      <details className="reasoning" key={index}>
        <summary>Reasoning</summary>
        <p>{part.text}</p>
      </details>
    );
  }
  if (part.type === "tool_call") {
    return (
      <pre className="tool-call" key={index}>
        {part.name}({JSON.stringify(part.arguments, null, 2)})
      </pre>
    );
  }
  return (
    <pre className="tool-call" key={index}>
      Tool result: {JSON.stringify(part.result, null, 2)}
    </pre>
  );
}

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [model, setModel] = useState("default");
  const [mode, setMode] = useState<SessionMode>("text");
  const [draft, setDraft] = useState("");
  const [live, setLive] = useState<LiveOutput | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities | null>(null);
  const [studio, setStudio] = useState<StudioStatus | null>(null);
  const [studioDraft, setStudioDraft] = useState<StudioConfig | null>(null);
  const [studioOpen, setStudioOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  const active = useMemo(
    () => sessions.find((session) => session.id === activeId) ?? null,
    [activeId, sessions],
  );

  const refreshSessions = useCallback(async (preferredId?: string) => {
    const next = await api.listSessions();
    setSessions(next);
    setActiveId((current) => {
      const wanted = preferredId ?? current;
      if (wanted && next.some((session) => session.id === wanted)) return wanted;
      return next[0]?.id ?? null;
    });
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
        const [nextCapabilities] = await Promise.all([
          api.runtimeCapabilities(),
          refreshSessions(),
        ]);
        if (current) setCapabilities(nextCapabilities);
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
  }, [refreshSessions]);

  useEffect(() => {
    if (!capabilities?.model_capabilities.features.full_duplex ||
        !capabilities.duplex_available) {
      setMode("text");
    }
  }, [capabilities]);

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
    };
  }, [activeId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: busy ? "smooth" : "auto" });
  }, [messages, live, busy]);

  async function createSession() {
    const selectedModel = model.trim();
    if (!selectedModel) return;
    setError(null);
    try {
      const created = await api.createSession(selectedModel, mode);
      setSessions((current) => [created, ...current]);
      setActiveId(created.id);
      setMessages([]);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function deleteActive() {
    if (!active || busy) return;
    setError(null);
    try {
      await api.deleteSession(active.id);
      setMessages([]);
      await refreshSessions();
    } catch (cause) {
      setError(errorMessage(cause));
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
    setError(null);
    try {
      const status = await configureStudio(studioDraft);
      setApiBaseUrl(status.service_url);
      setStudio(status);
      setStudioOpen(false);
      setMessages([]);
      setActiveId(null);
      const [nextCapabilities] = await Promise.all([
        api.runtimeCapabilities(),
        refreshSessions(),
      ]);
      setCapabilities(nextCapabilities);
      if (!status.reachable) {
        setError(`MFQd is not reachable at ${status.service_url}`);
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
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

  async function send(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!active || !text || busy) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setDraft("");
    setError(null);
    setBusy(true);
    setLive({ reasoning: "", text: "", tools: [] });
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
    try {
      await streamResponse(
        active.id,
        {
          request_id: crypto.randomUUID(),
          expected_revision: active.revision,
          input: [{ type: "text", text }],
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
        const [persisted, session] = await Promise.all([
          api.listMessages(active.id),
          api.getSession(active.id),
        ]);
        setMessages(persisted);
        setSessions((current) =>
          current.map((item) => (item.id === session.id ? session : item)),
        );
      } catch (cause) {
        setError(errorMessage(cause));
      }
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <img src="/mfq-mark.svg" alt="" />
          <div>
            <strong>MFQ</strong>
            <span>Local inference</span>
          </div>
        </div>

        <div className="session-create">
          <label htmlFor="model">Model</label>
          <input
            id="model"
            value={model}
            onChange={(event) => setModel(event.target.value)}
            placeholder="Model name"
          />
          {capabilities?.model_capabilities.features.full_duplex && (
            <button
              aria-pressed={mode === "full_duplex"}
              className={`duplex-session-toggle ${mode === "full_duplex" ? "selected" : ""}`}
              disabled={!capabilities.duplex_available}
              onClick={() => setMode((current) => current === "full_duplex" ? "text" : "full_duplex")}
              title={capabilities.duplex_available ? "Use full-duplex voice for new sessions" : "The runtime does not have a duplex worker"}
              type="button"
            >
              Full duplex {mode === "full_duplex" ? "on" : "off"}
            </button>
          )}
          <button className="new-session" onClick={createSession} type="button">
            New session
          </button>
        </div>

        <nav className="session-list" aria-label="Sessions">
          <div className="section-label">Sessions</div>
          {loading && <span className="empty-note">Loading…</span>}
          {!loading && sessions.length === 0 && (
            <span className="empty-note">No saved sessions</span>
          )}
          {sessions.map((session) => (
            <button
              className={`session-row ${activeId === session.id ? "active" : ""}`}
              key={session.id}
              onClick={() => setActiveId(session.id)}
              type="button"
            >
              <span>{session.title || "Untitled session"}</span>
              <small>
                {MODE_LABELS[session.mode]} · r{session.revision}
              </small>
            </button>
          ))}
        </nav>
      </aside>

      <main className="conversation">
        <header className="conversation-header">
          <div>
            <h1>{active?.title || (active ? "Untitled session" : "MFQ workspace")}</h1>
            <p>{active ? active.model : "Create a session to start local inference"}</p>
          </div>
          <div className="header-actions">
            {capabilities && (
              <div
                className="header-capabilities"
                title={capabilities.model_capabilities.architecture_family}
              >
                {CAPABILITY_LABELS.filter(
                  ([feature]) => capabilities.model_capabilities.features[feature],
                ).map(([feature, label]) => (
                  <span
                    className={
                      feature === "full_duplex" && !capabilities.duplex_available
                        ? "unavailable"
                        : ""
                    }
                    key={feature}
                  >
                    {label}
                  </span>
                ))}
              </div>
            )}
            {studio && (
              <button
                className={`studio-runtime ${studio.reachable ? "reachable" : "unreachable"}`}
                onClick={openStudioSettings}
                title={studio.service_url}
                type="button"
              >
                <span aria-hidden="true" />
                {studio.config.mode === "local" ? "Local" : "Remote"}
              </button>
            )}
            {active && (
              <>
              <span className={`state state-${active.state}`}>{active.state}</span>
              <button onClick={deleteActive} disabled={busy} type="button">
                Delete
              </button>
              </>
            )}
          </div>
        </header>

        <section className="message-list" aria-live="polite">
          {!active && (
            <div className="welcome">
              <img src="/mfq-mark.svg" alt="" />
              <h2>Persistent local sessions</h2>
              <p>Text conversations now live in MFQd and remain available after a page reload.</p>
            </div>
          )}
          {active && messages.length === 0 && !busy && (
            <div className="welcome compact">
              <h2>Ready</h2>
              <p>This session uses {MODE_LABELS[active.mode].toLowerCase()} mode.</p>
            </div>
          )}
          {messages.map((message) => (
            <article className={`message message-${message.role}`} key={message.id}>
              <div className="message-role">{message.role}</div>
              <div className="message-body">
                {message.parts.map((part, index) => partView(part, index))}
              </div>
            </article>
          ))}
          {live && (
            <article className="message message-assistant live-message">
              <div className="message-role">assistant</div>
              <div className="message-body">
                {live.reasoning && (
                  <details className="reasoning" open>
                    <summary>Reasoning</summary>
                    <p>{live.reasoning}</p>
                  </details>
                )}
                {live.text && <p>{live.text}</p>}
                {live.tools.map((tool, index) => (
                  <pre className="tool-call" key={index}>
                    {tool}
                  </pre>
                ))}
                {!live.reasoning && !live.text && live.tools.length === 0 && (
                  <span className="thinking">Generating…</span>
                )}
              </div>
            </article>
          )}
          <div ref={endRef} />
        </section>

        {error && (
          <div className="error-banner" role="alert">
            {error}
            <button onClick={() => setError(null)} type="button" aria-label="Dismiss error">
              ×
            </button>
          </div>
        )}

        <form className="composer" onSubmit={send}>
          <textarea
            aria-label="Message"
            disabled={!active || busy}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder={active ? "Message MFQ…" : "Create a session first"}
            rows={1}
            value={draft}
          />
          {busy ? (
            <button
              className="stop-button"
              onClick={() => abortRef.current?.abort()}
              type="button"
            >
              Stop
            </button>
          ) : (
            <button className="send-button" disabled={!active || !draft.trim()} type="submit">
              Send
            </button>
          )}
        </form>
      </main>

      <aside className="inspector">
        <div className="section-label">Session</div>
        {active ? (
          <dl>
            <div>
              <dt>Mode</dt>
              <dd>{MODE_LABELS[active.mode]}</dd>
            </div>
            <div>
              <dt>Revision</dt>
              <dd>{active.revision}</dd>
            </div>
            <div>
              <dt>Runtime</dt>
              <dd>{active.runtime_instance_id ? "Attached" : "Text gateway"}</dd>
            </div>
          </dl>
        ) : (
          <p className="empty-note">No active session</p>
        )}
        <div className="capability-card">
          <strong>Current transport</strong>
          <span>REST + typed SSE</span>
          <p>Media and duplex state stay attached to the persisted session.</p>
        </div>
      </aside>

      {studioOpen && studioDraft && (
        <div className="studio-dialog-backdrop" role="presentation">
          <form
            aria-labelledby="studio-dialog-title"
            className="studio-dialog"
            onSubmit={saveStudioSettings}
            role="dialog"
          >
            <div className="studio-dialog-header">
              <div>
                <h2 id="studio-dialog-title">Runtime connection</h2>
                <p>MFQ Studio keeps MFQd running when this window closes.</p>
              </div>
              <button
                aria-label="Close runtime settings"
                onClick={() => setStudioOpen(false)}
                type="button"
              >
                ×
              </button>
            </div>
            <div className="studio-mode-picker" aria-label="Runtime location">
              {(["local", "remote"] as const).map((item) => (
                <button
                  aria-pressed={studioDraft.mode === item}
                  className={studioDraft.mode === item ? "selected" : ""}
                  key={item}
                  onClick={() => setStudioDraft((current) => current && ({ ...current, mode: item }))}
                  type="button"
                >
                  {item === "local" ? "Local MFQd" : "Remote MFQd"}
                </button>
              ))}
            </div>
            {studioDraft.mode === "local" ? (
              <>
                <label htmlFor="studio-backend-url">Local runtime URL</label>
                <input
                  id="studio-backend-url"
                  onChange={(event) => setStudioDraft((current) => current && ({ ...current, local_backend_url: event.target.value }))}
                  required
                  type="url"
                  value={studioDraft.local_backend_url}
                />
                <label htmlFor="studio-port">MFQd port</label>
                <input
                  id="studio-port"
                  max={65535}
                  min={1}
                  onChange={(event) => setStudioDraft((current) => current && ({ ...current, local_service_port: Number(event.target.value) }))}
                  required
                  type="number"
                  value={studioDraft.local_service_port}
                />
                <label htmlFor="studio-executable">MFQd executable</label>
                <input
                  id="studio-executable"
                  onChange={(event) => setStudioDraft((current) => current && ({ ...current, mfqd_executable: event.target.value || null }))}
                  placeholder="mfqd from PATH"
                  value={studioDraft.mfqd_executable ?? ""}
                />
              </>
            ) : (
              <>
                <label htmlFor="studio-remote-url">Remote MFQd URL</label>
                <input
                  id="studio-remote-url"
                  onChange={(event) => setStudioDraft((current) => current && ({ ...current, remote_url: event.target.value }))}
                  required
                  type="url"
                  value={studioDraft.remote_url}
                />
              </>
            )}
            <div className="studio-dialog-status">
              <span className={studio?.reachable ? "reachable" : "unreachable"} />
              {studio?.reachable ? `Connected to ${studio.service_url}` : "MFQd is offline"}
            </div>
            <div className="studio-dialog-actions">
              <button onClick={() => setStudioOpen(false)} type="button">
                Cancel
              </button>
              <button className="primary" disabled={busy} type="submit">
                Apply
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
