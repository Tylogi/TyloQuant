(() => {
  "use strict";

  const STORAGE_KEY = "mfq.console.v1";
  const API_KEY_STORAGE = "mfq.console.api-key";
  const MAX_CONVERSATIONS = 40;
  const MAX_METRIC_POINTS = 32;
  const LEGACY_LOCAL_ENDPOINT = "http://127.0.0.1:8080";

  const defaultEndpoint = (() => {
    if (location.protocol === "http:" || location.protocol === "https:") {
      return location.origin;
    }
    return LEGACY_LOCAL_ENDPOINT;
  })();

  const defaultSettings = {
    endpoint: defaultEndpoint,
    model: "",
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
    excludeReasoningFromContext: false,
    preset: "balanced",
    samplingCustomized: false,
  };

  const legacySamplingDefaults = {
    temperature: 0.7,
    topP: 0.8,
    topK: 20,
    repetitionPenalty: 1,
    presencePenalty: 0,
    frequencyPenalty: 0,
  };

  const deepSeekV4SamplingDefaults = {
    temperature: 1,
    topP: 0.8,
    topK: 20,
    repetitionPenalty: 1.05,
    presencePenalty: 1.35,
    frequencyPenalty: 0,
  };

  const presets = {
    precise: {
      temperature: 0.2,
      topP: 0.75,
      topK: 20,
      repetitionPenalty: 1.05,
    },
    balanced: {
      temperature: 0.7,
      topP: 0.8,
      topK: 20,
      repetitionPenalty: 1,
    },
    creative: {
      temperature: 1,
      topP: 0.95,
      topK: 50,
      repetitionPenalty: 1,
    },
  };

  const state = {
    conversations: [],
    activeId: "",
    settings: { ...defaultSettings },
    apiKey: "",
    view: "chat",
    status: null,
    models: [],
    connected: false,
    connectionMessage: "正在连接",
    statusApiAvailable: null,
    generating: false,
    generatingMessage: null,
    editingMessage: null,
    renamingConversationId: "",
    controller: null,
    metricSeries: [],
    lastMetricRequestId: "",
    renderPending: false,
    followOutput: true,
    reasoningOpenState: new WeakMap(),
    pollTimer: 0,
    samplingDefaults: null,
  };

  const refs = {};

  function queryRefs() {
    const ids = [
      "sidebar", "sidebar-scrim", "open-sidebar", "close-sidebar",
      "new-chat", "clear-history", "conversation-list",
      "sidebar-status-dot", "sidebar-status-label", "sidebar-endpoint",
      "connection-pill", "active-request-count", "model-select",
      "top-ttft", "top-prefill-tps", "top-tps", "export-chat", "open-settings",
      "chat-view", "monitor-view", "message-scroller", "message-list",
      "composer-form", "message-input", "thinking-toggle",
      "reasoning-effort-control", "reasoning-effort-select", "composer-hint",
      "send-button", "refresh-status", "monitor-updated",
      "metric-prefill-tps", "metric-decode-tps", "metric-ttft",
      "metric-requests", "metric-active",
      "metric-tokens", "throughput-chart", "chart-current",
      "runtime-model", "runtime-type", "runtime-context", "runtime-uptime",
      "runtime-failed", "last-request-id", "last-request-state",
      "last-prompt-tokens", "last-completion-tokens", "last-prefill-tps",
      "last-prefill-ms", "last-generation-ms", "last-finish-reason",
      "settings-panel", "settings-scrim",
      "close-settings", "setting-endpoint", "setting-api-key",
      "preset-control", "setting-system-prompt",
      "setting-exclude-reasoning", "setting-context-window",
      "setting-context-limit", "reload-model", "setting-max-tokens",
      "setting-temperature", "temperature-value", "setting-top-p",
      "top-p-value", "setting-top-k", "setting-repetition",
      "repetition-value", "setting-presence", "presence-value",
      "setting-frequency", "frequency-value", "reset-settings",
      "save-settings", "toast-region",
    ];
    for (const id of ids) refs[id] = document.getElementById(id);
    refs.navItems = [...document.querySelectorAll(".nav-item[data-view]")];
    refs.presetButtons = [...document.querySelectorAll("[data-preset]")];
  }

  function randomId(prefix) {
    if (globalThis.crypto?.randomUUID) {
      return `${prefix}-${crypto.randomUUID()}`;
    }
    return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function newConversation() {
    return {
      id: randomId("chat"),
      title: "新对话",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
  }

  function sanitizeConversation(value) {
    if (!value || typeof value !== "object" || !Array.isArray(value.messages)) {
      return null;
    }
    const messages = value.messages
      .filter((message) =>
        message &&
        (message.role === "user" || message.role === "assistant") &&
        typeof message.content === "string")
      .map((message) => ({
        role: message.role,
        content: message.content,
        reasoning: typeof message.reasoning === "string" ? message.reasoning : "",
        createdAt: Number(message.createdAt) || Date.now(),
        error: Boolean(message.error),
      }));
    return {
      id: typeof value.id === "string" ? value.id : randomId("chat"),
      title: typeof value.title === "string" && value.title.trim()
        ? value.title.slice(0, 80)
        : "新对话",
      createdAt: Number(value.createdAt) || Date.now(),
      updatedAt: Number(value.updatedAt) || Date.now(),
      messages,
    };
  }

  function loadState() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      if (saved.settings && typeof saved.settings === "object") {
        state.settings = { ...defaultSettings, ...saved.settings };
        if (!["", "high", "max"].includes(state.settings.reasoningEffort)) {
          state.settings.reasoningEffort = "";
        }
        if (typeof saved.settings.samplingCustomized !== "boolean") {
          state.settings.samplingCustomized = !samplingMatches(
            state.settings,
            legacySamplingDefaults
          );
        }
        if (
          location.pathname.startsWith("/admin") &&
          defaultEndpoint !== LEGACY_LOCAL_ENDPOINT &&
          state.settings.endpoint === LEGACY_LOCAL_ENDPOINT
        ) {
          state.settings.endpoint = defaultEndpoint;
        }
      }
      if (Array.isArray(saved.conversations)) {
        state.conversations = saved.conversations
          .map(sanitizeConversation)
          .filter(Boolean)
          .slice(0, MAX_CONVERSATIONS);
      }
      if (typeof saved.activeId === "string") state.activeId = saved.activeId;
    } catch {
      state.settings = { ...defaultSettings };
      state.conversations = [];
    }
    state.apiKey = sessionStorage.getItem(API_KEY_STORAGE) || "";
    if (!state.conversations.length) {
      const conversation = newConversation();
      state.conversations.push(conversation);
      state.activeId = conversation.id;
    }
    if (!state.conversations.some((item) => item.id === state.activeId)) {
      state.activeId = state.conversations[0].id;
    }
  }

  function persistState() {
    const conversations = state.conversations
      .slice()
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_CONVERSATIONS);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        settings: state.settings,
        conversations,
        activeId: state.activeId,
      }));
      if (state.apiKey) {
        sessionStorage.setItem(API_KEY_STORAGE, state.apiKey);
      } else {
        sessionStorage.removeItem(API_KEY_STORAGE);
      }
    } catch {
      showToast("浏览器存储空间不足，历史记录未保存。", true);
    }
  }

  function samplingMatches(settings, defaults) {
    return settings.temperature === defaults.temperature &&
      settings.topP === defaults.topP &&
      settings.topK === defaults.topK &&
      settings.repetitionPenalty === defaults.repetitionPenalty &&
      settings.presencePenalty === defaults.presencePenalty &&
      settings.frequencyPenalty === defaults.frequencyPenalty;
  }

  function normalizeSamplingDefaults(value) {
    if (!value || typeof value !== "object") return null;
    const defaults = {
      temperature: Number(value.temperature),
      topP: Number(value.top_p),
      topK: Number(value.top_k),
      repetitionPenalty: Number(value.repetition_penalty),
      presencePenalty: Number(value.presence_penalty),
      frequencyPenalty: Number(value.frequency_penalty),
    };
    return Object.values(defaults).every(Number.isFinite) ? defaults : null;
  }

  function applyServerSamplingDefaults(status) {
    const defaults = normalizeSamplingDefaults(status?.sampling_defaults) ||
      (status?.model_type === "deepseek_v4"
        ? deepSeekV4SamplingDefaults
        : null);
    if (!defaults) return;
    state.samplingDefaults = defaults;
    if (state.settings.samplingCustomized ||
        samplingMatches(state.settings, defaults)) return;
    Object.assign(state.settings, defaults);
    state.settings.preset = "custom";
    persistState();
  }

  function activeConversation() {
    return state.conversations.find((item) => item.id === state.activeId);
  }

  function normalizeEndpoint(value) {
    const trimmed = String(value || "").trim().replace(/\/+$/, "");
    return trimmed || defaultEndpoint;
  }

  function apiUrl(path) {
    return `${normalizeEndpoint(state.settings.endpoint)}${path}`;
  }

  function requestHeaders(jsonBody = false) {
    const headers = {};
    if (jsonBody) headers["Content-Type"] = "application/json";
    if (state.apiKey) headers.Authorization = `Bearer ${state.apiKey}`;
    return headers;
  }

  async function responseError(response) {
    let message = `${response.status} ${response.statusText}`.trim();
    const fallback = response.clone();
    try {
      const body = await response.json();
      message = body?.error?.message || body?.message || message;
    } catch {
      const text = await fallback.text().catch(() => "");
      if (text.trim()) message = text.trim().slice(0, 300);
    }
    return message;
  }

  async function fetchJson(path, options = {}) {
    const response = await fetch(apiUrl(path), {
      ...options,
      headers: {
        ...requestHeaders(Boolean(options.body)),
        ...(options.headers || {}),
      },
    });
    if (!response.ok) {
      const error = new Error(await responseError(response));
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  function setConnection(connected, message) {
    state.connected = connected;
    state.connectionMessage = message;
    refs["connection-pill"].classList.toggle("is-connected", connected);
    refs["connection-pill"].classList.toggle("is-error", !connected);
    refs["connection-pill"].querySelector("span:last-child").textContent = message;
    refs["sidebar-status-dot"].classList.toggle("is-connected", connected);
    refs["sidebar-status-dot"].classList.toggle("is-error", !connected);
    refs["sidebar-status-label"].textContent = message;
    try {
      const endpoint = new URL(normalizeEndpoint(state.settings.endpoint));
      refs["sidebar-endpoint"].textContent = endpoint.host;
    } catch {
      refs["sidebar-endpoint"].textContent = normalizeEndpoint(state.settings.endpoint);
    }
  }

  function showToast(message, isError = false) {
    const toast = document.createElement("div");
    toast.className = `toast${isError ? " is-error" : ""}`;
    toast.textContent = message;
    refs["toast-region"].append(toast);
    window.setTimeout(() => toast.remove(), 3500);
  }

  function formatNumber(value, maximumFractionDigits = 0) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "--";
    return new Intl.NumberFormat("zh-CN", {
      maximumFractionDigits,
    }).format(number);
  }

  function formatDuration(totalSeconds) {
    const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (days) return `${days}天 ${hours}小时`;
    if (hours) return `${hours}小时 ${minutes}分`;
    if (minutes) return `${minutes}分 ${seconds % 60}秒`;
    return `${seconds}秒`;
  }

  function formatTime(timestamp) {
    const time = Number(timestamp);
    if (!Number.isFinite(time)) return "";
    return new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(time));
  }

  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#icon-${name}`);
    svg.append(use);
    return svg;
  }

  function finishConversationRename(conversation, value) {
    const title = String(value || "").replace(/\s+/g, " ").trim();
    state.renamingConversationId = "";
    if (!title) {
      renderConversationList();
      return;
    }
    conversation.title = title.slice(0, 80);
    conversation.updatedAt = Date.now();
    persistState();
    renderConversationList();
  }

  function cancelConversationRename() {
    state.renamingConversationId = "";
    renderConversationList();
  }

  function renderConversationList() {
    refs["conversation-list"].replaceChildren();
    const ordered = state.conversations
      .slice()
      .sort((a, b) => b.updatedAt - a.updatedAt);
    for (const conversation of ordered) {
      const row = document.createElement("div");
      row.className = `conversation-item${conversation.id === state.activeId ? " is-active" : ""}`;
      row.setAttribute("role", "button");
      row.tabIndex = 0;
      row.title = conversation.title;

      if (state.renamingConversationId === conversation.id) {
        const input = document.createElement("input");
        input.className = "conversation-rename-input";
        input.value = conversation.title;
        input.maxLength = 80;
        input.setAttribute("aria-label", "对话名称");
        input.addEventListener("click", (event) => event.stopPropagation());
        input.addEventListener("keydown", (event) => {
          event.stopPropagation();
          if (event.key === "Enter") {
            event.preventDefault();
            finishConversationRename(conversation, input.value);
          } else if (event.key === "Escape") {
            event.preventDefault();
            cancelConversationRename();
          }
        });
        input.addEventListener("blur", () => {
          if (state.renamingConversationId === conversation.id) {
            finishConversationRename(conversation, input.value);
          }
        });
        row.append(input);
        requestAnimationFrame(() => {
          input.focus();
          input.select();
        });
      } else {
        const title = document.createElement("span");
        title.textContent = conversation.title;
        row.append(title);
      }

      const rename = document.createElement("button");
      rename.type = "button";
      rename.className = "conversation-action conversation-rename";
      rename.title = "重命名对话";
      rename.setAttribute("aria-label", `重命名 ${conversation.title}`);
      rename.append(icon("edit"));
      rename.addEventListener("click", (event) => {
        event.stopPropagation();
        if (state.generating) return;
        state.renamingConversationId = conversation.id;
        renderConversationList();
      });
      row.append(rename);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "conversation-action conversation-delete";
      remove.title = "删除对话";
      remove.setAttribute("aria-label", `删除 ${conversation.title}`);
      remove.append(icon("trash"));
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteConversation(conversation.id);
      });
      row.append(remove);

      const activate = () => {
        if (state.generating) return;
        if (state.renamingConversationId === conversation.id) return;
        state.activeId = conversation.id;
        state.editingMessage = null;
        state.renamingConversationId = "";
        state.followOutput = true;
        conversation.updatedAt = Date.now();
        persistState();
        renderConversationList();
        renderMessages();
        closeSidebar();
      };
      row.addEventListener("click", activate);
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
      refs["conversation-list"].append(row);
    }
  }

  function createConversation() {
    if (state.generating) stopGeneration();
    const conversation = newConversation();
    state.conversations.unshift(conversation);
    state.activeId = conversation.id;
    state.editingMessage = null;
    state.renamingConversationId = "";
    state.followOutput = true;
    persistState();
    renderConversationList();
    renderMessages();
    switchView("chat");
    closeSidebar();
    refs["message-input"].focus();
  }

  function deleteConversation(id) {
    if (state.generating && id === state.activeId) return;
    state.conversations = state.conversations.filter((item) => item.id !== id);
    if (!state.conversations.length) {
      state.conversations.push(newConversation());
    }
    if (!state.conversations.some((item) => item.id === state.activeId)) {
      state.activeId = state.conversations[0].id;
    }
    state.editingMessage = null;
    state.renamingConversationId = "";
    state.followOutput = true;
    persistState();
    renderConversationList();
    renderMessages();
  }

  function emptyState() {
    const root = document.createElement("div");
    root.className = "empty-state";

    const mark = document.createElement("div");
    mark.className = "empty-state-mark";
    const image = document.createElement("img");
    image.src = "mfq-mark.svg";
    image.alt = "";
    mark.append(image);

    const heading = document.createElement("h1");
    heading.textContent = "本地模型，直接对话";
    const copy = document.createElement("p");
    copy.textContent = "连接 MFQ C++ runtime，支持流式输出、思考过程与完整采样控制。";

    const prompts = [
      "解释 NINT 的 per-neuron super block 设计",
      "写一个 CUDA kernel 性能分析清单",
      "比较稠密模型和 MoE 的量化策略",
      "把这段技术结论改写得更简洁",
    ];
    const grid = document.createElement("div");
    grid.className = "prompt-grid";
    for (const prompt of prompts) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "prompt-suggestion";
      button.textContent = prompt;
      button.addEventListener("click", () => {
        refs["message-input"].value = prompt;
        resizeComposer();
        refs["message-input"].focus();
      });
      grid.append(button);
    }
    root.append(mark, heading, copy, grid);
    return root;
  }

  function appendInline(parent, text) {
    const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^)\s]+\)|https?:\/\/[^\s<]+)/g;
    let cursor = 0;
    for (const match of text.matchAll(pattern)) {
      if (match.index > cursor) {
        parent.append(document.createTextNode(text.slice(cursor, match.index)));
      }
      const token = match[0];
      if (token.startsWith("`") && token.endsWith("`")) {
        const code = document.createElement("code");
        code.textContent = token.slice(1, -1);
        parent.append(code);
      } else if (token.startsWith("**") && token.endsWith("**")) {
        const strong = document.createElement("strong");
        strong.textContent = token.slice(2, -2);
        parent.append(strong);
      } else {
        const markdownLink = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/);
        const link = document.createElement("a");
        link.href = markdownLink ? markdownLink[2] : token;
        link.textContent = markdownLink ? markdownLink[1] : token;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        parent.append(link);
      }
      cursor = match.index + token.length;
    }
    if (cursor < text.length) {
      parent.append(document.createTextNode(text.slice(cursor)));
    }
  }

  function codeBlock(language, content) {
    const root = document.createElement("div");
    root.className = "code-block";
    const header = document.createElement("div");
    header.className = "code-header";
    const label = document.createElement("span");
    label.textContent = language || "text";
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "code-copy";
    copy.title = "复制代码";
    copy.setAttribute("aria-label", "复制代码");
    copy.append(icon("copy"));
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(content);
        copy.replaceChildren(icon("check"));
        window.setTimeout(() => copy.replaceChildren(icon("copy")), 1200);
      } catch {
        showToast("浏览器拒绝了剪贴板访问。", true);
      }
    });
    header.append(label, copy);
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = content;
    pre.append(code);
    root.append(header, pre);
    return root;
  }

  function renderBasicRichText(text) {
    const root = document.createDocumentFragment();
    const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
    let paragraph = [];
    let list = null;
    let listType = "";

    const flushParagraph = () => {
      if (!paragraph.length) return;
      const p = document.createElement("p");
      appendInline(p, paragraph.join("\n"));
      root.append(p);
      paragraph = [];
    };

    const flushList = () => {
      if (!list) return;
      root.append(list);
      list = null;
      listType = "";
    };

    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      const fence = line.match(/^```([\w.+-]*)\s*$/);
      if (fence) {
        flushParagraph();
        flushList();
        const codeLines = [];
        index += 1;
        while (index < lines.length && !/^```\s*$/.test(lines[index])) {
          codeLines.push(lines[index]);
          index += 1;
        }
        root.append(codeBlock(fence[1], codeLines.join("\n")));
        continue;
      }

      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        flushParagraph();
        flushList();
        const node = document.createElement(`h${heading[1].length}`);
        appendInline(node, heading[2]);
        root.append(node);
        continue;
      }

      const unordered = line.match(/^\s*[-*]\s+(.+)$/);
      const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
      if (unordered || ordered) {
        flushParagraph();
        const type = ordered ? "ol" : "ul";
        if (!list || listType !== type) {
          flushList();
          list = document.createElement(type);
          listType = type;
        }
        const item = document.createElement("li");
        appendInline(item, (unordered || ordered)[1]);
        list.append(item);
        continue;
      }

      const quote = line.match(/^>\s?(.*)$/);
      if (quote) {
        flushParagraph();
        flushList();
        const blockquote = document.createElement("blockquote");
        appendInline(blockquote, quote[1]);
        root.append(blockquote);
        continue;
      }

      if (!line.trim()) {
        flushParagraph();
        flushList();
        continue;
      }

      flushList();
      paragraph.push(line);
    }
    flushParagraph();
    flushList();
    return root;
  }

  function decorateCodeBlocks(container) {
    for (const code of [...container.querySelectorAll("pre > code")]) {
      const pre = code.parentElement;
      if (!pre) continue;
      const languageClass = [...code.classList]
        .find((name) => name.startsWith("language-"));
      const language = languageClass
        ? languageClass.slice("language-".length)
        : "text";
      pre.replaceWith(codeBlock(language, code.textContent || ""));
    }
  }

  function renderRichText(text, options = {}) {
    const source = String(text || "");
    if (
      typeof globalThis.marked?.parse !== "function" ||
      typeof globalThis.DOMPurify?.sanitize !== "function"
    ) {
      return renderBasicRichText(source);
    }

    try {
      const html = globalThis.marked.parse(source, {
        async: false,
        breaks: true,
        gfm: true,
      });
      const container = document.createElement("div");
      container.innerHTML = globalThis.DOMPurify.sanitize(html, {
        USE_PROFILES: { html: true },
        FORBID_TAGS: ["script", "style", "template"],
        FORBID_ATTR: ["style"],
      });
      for (const link of container.querySelectorAll("a[href]")) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
      decorateCodeBlocks(container);
      if (
        options.renderMath !== false &&
        typeof globalThis.renderMathInElement === "function"
      ) {
        globalThis.renderMathInElement(container, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "\\[", right: "\\]", display: true },
            { left: "\\(", right: "\\)", display: false },
            { left: "$", right: "$", display: false },
          ],
          ignoredTags: [
            "script", "noscript", "style", "textarea", "pre", "code",
          ],
          throwOnError: false,
          strict: "ignore",
          trust: false,
        });
      }
      const root = document.createDocumentFragment();
      root.append(...container.childNodes);
      return root;
    } catch {
      return renderBasicRichText(source);
    }
  }

  function assistantParts(message) {
    let content = message.content || "";
    let reasoning = message.reasoning || "";

    const thinkStart = content.indexOf("<think>");
    const thinkEnd = content.indexOf("</think>");
    if (thinkStart >= 0 && thinkEnd > thinkStart) {
      reasoning = `${reasoning}${reasoning ? "\n" : ""}${content.slice(thinkStart + 7, thinkEnd)}`;
      content = `${content.slice(0, thinkStart)}${content.slice(thinkEnd + 8)}`;
    }

    const channelStart = content.indexOf("<|channel>thought");
    const channelEnd = content.indexOf("<channel|>", channelStart + 1);
    if (channelStart >= 0 && channelEnd > channelStart) {
      const start = channelStart + "<|channel>thought".length;
      reasoning = `${reasoning}${reasoning ? "\n" : ""}${content.slice(start, channelEnd)}`;
      content = `${content.slice(0, channelStart)}${content.slice(channelEnd + "<channel|>".length)}`;
    }

    content = content
      .replaceAll("<|channel>final", "")
      .replaceAll("<|channel>analysis", "")
      .replaceAll("<channel|>", "")
      .trimStart();
    return { content, reasoning: reasoning.trim() };
  }

  function resizeEditor(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }

  function startMessageEdit(message) {
    if (state.generating) return;
    state.editingMessage = message;
    state.followOutput = false;
    renderMessages({ scroll: false });
  }

  function messageEditorElement(conversation, message) {
    const form = document.createElement("form");
    form.className = "message-editor";

    const fields = [];
    const addField = (labelText, value, className) => {
      const label = document.createElement("label");
      label.className = "message-editor-field";
      const caption = document.createElement("span");
      caption.textContent = labelText;
      const textarea = document.createElement("textarea");
      textarea.className = className;
      textarea.value = value;
      textarea.rows = 1;
      textarea.addEventListener("input", () => resizeEditor(textarea));
      textarea.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          state.editingMessage = null;
          renderMessages({ scroll: false });
        } else if (event.key === "Enter" && event.ctrlKey) {
          event.preventDefault();
          form.requestSubmit();
        }
      });
      label.append(caption, textarea);
      form.append(label);
      fields.push(textarea);
      return textarea;
    };

    let reasoningInput = null;
    let contentInput = null;
    if (message.role === "assistant") {
      const parts = assistantParts(message);
      reasoningInput = addField("思考过程", parts.reasoning, "message-editor-reasoning");
      contentInput = addField("回答", parts.content, "message-editor-content");
    } else {
      contentInput = addField("消息", message.content, "message-editor-content");
    }

    const buttons = document.createElement("div");
    buttons.className = "message-editor-buttons";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "secondary-button";
    cancel.textContent = "取消";
    cancel.addEventListener("click", () => {
      state.editingMessage = null;
      renderMessages({ scroll: false });
    });
    const save = document.createElement("button");
    save.type = "submit";
    save.className = "primary-button";
    save.textContent = "保存";
    buttons.append(cancel, save);
    form.append(buttons);

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const content = contentInput.value.trim();
      const reasoning = reasoningInput?.value.trim() || "";
      if (message.role === "user" && !content) {
        showToast("用户消息不能为空。", true);
        contentInput.focus();
        return;
      }
      if (message.role === "assistant" && !content && !reasoning) {
        showToast("回答和思考过程不能同时为空。", true);
        contentInput.focus();
        return;
      }
      message.content = content;
      message.reasoning = reasoning;
      message.error = false;
      conversation.updatedAt = Date.now();
      state.editingMessage = null;
      persistState();
      renderConversationList();
      renderMessages({ scroll: false });
    });

    requestAnimationFrame(() => {
      for (const field of fields) resizeEditor(field);
      (contentInput || fields[0])?.focus();
    });
    return form;
  }

  function messageActionButton(iconName, label, handler) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "message-action";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.append(icon(iconName), document.createTextNode(label));
    button.addEventListener("click", handler);
    return button;
  }

  function messageElement(conversation, message) {
    const article = document.createElement("article");
    article.className = `message ${message.role}${message.error ? " is-error" : ""}`;

    if (message.role === "assistant") {
      const avatar = document.createElement("div");
      avatar.className = "message-avatar";
      const image = document.createElement("img");
      image.src = "mfq-mark.svg";
      image.alt = "MFQ";
      avatar.append(image);
      article.append(avatar);
    }

    const body = document.createElement("div");
    body.className = "message-body";

    const meta = document.createElement("div");
    meta.className = "message-meta";
    const name = document.createElement("strong");
    name.textContent = message.role === "assistant" ? "MFQ" : "你";
    const time = document.createElement("span");
    time.textContent = formatTime(message.createdAt);
    meta.append(name, time);
    body.append(meta);

    if (state.editingMessage === message) {
      body.append(messageEditorElement(conversation, message));
      article.append(body);
      return article;
    }

    const parts = message.role === "assistant"
      ? assistantParts(message)
      : { content: message.content, reasoning: "" };
    const isGeneratingMessage =
      state.generating && state.generatingMessage === message;

    if (parts.reasoning) {
      const reasoning = document.createElement("details");
      reasoning.className = "reasoning";
      const rememberedOpen = state.reasoningOpenState.get(message);
      reasoning.open = rememberedOpen ?? isGeneratingMessage;
      const summary = document.createElement("summary");
      summary.append(icon("chevron"), document.createTextNode(
        isGeneratingMessage ? "正在思考" : "思考过程"));
      const toggleReasoning = () => {
        const nextOpen = !reasoning.open;
        state.reasoningOpenState.set(message, nextOpen);
        reasoning.open = nextOpen;
      };
      summary.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        toggleReasoning();
      });
      summary.addEventListener("click", (event) => {
        event.preventDefault();
        if (event.detail === 0) toggleReasoning();
      });
      reasoning.addEventListener("toggle", () => {
        state.reasoningOpenState.set(message, reasoning.open);
      });
      const content = document.createElement("div");
      content.className = "reasoning-content";
      content.append(renderRichText(parts.reasoning, {
        renderMath: !isGeneratingMessage,
      }));
      reasoning.append(summary, content);
      body.append(reasoning);
    }

    const content = document.createElement("div");
    content.className = "message-content";
    if (parts.content) {
      content.append(renderRichText(parts.content, {
        renderMath: !isGeneratingMessage,
      }));
    } else if (isGeneratingMessage && message.role === "assistant") {
      const typing = document.createElement("div");
      typing.className = "typing-indicator";
      typing.setAttribute("aria-label", "正在生成");
      typing.append(document.createElement("span"), document.createElement("span"), document.createElement("span"));
      content.append(typing);
    }
    body.append(content);

    let actions = null;
    if (!state.generating) {
      actions = document.createElement("div");
      actions.className = "message-actions";
      actions.append(messageActionButton("edit", "编辑", () => {
        startMessageEdit(message);
      }));
      if (message.role === "assistant") {
        actions.append(messageActionButton("refresh", "重新生成", () => {
          void rerollAssistant(conversation, message);
        }));
      }
      if (message.role === "assistant") body.append(actions);
    }

    article.append(body);
    if (actions && message.role === "user") article.append(actions);
    return article;
  }

  function renderMessages(options = {}) {
    const scroller = refs["message-scroller"];
    const previousScrollTop = scroller.scrollTop;
    const shouldFollow = options.scroll !== false && state.followOutput;
    const conversation = activeConversation();
    refs["message-list"].replaceChildren();
    if (!conversation || !conversation.messages.length) {
      refs["message-list"].append(emptyState());
    } else {
      conversation.messages.forEach((message) => {
        refs["message-list"].append(messageElement(conversation, message));
      });
    }
    requestAnimationFrame(() => {
      scroller.scrollTop = shouldFollow ? scroller.scrollHeight : previousScrollTop;
    });
  }

  function scheduleMessageRender() {
    if (state.renderPending) return;
    state.renderPending = true;
    requestAnimationFrame(() => {
      state.renderPending = false;
      renderMessages();
    });
  }

  function resizeComposer() {
    const input = refs["message-input"];
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 190)}px`;
    refs["send-button"].disabled = !state.generating && !input.value.trim();
  }

  function setGenerating(generating) {
    state.generating = generating;
    refs["send-button"].classList.toggle("is-generating", generating);
    refs["send-button"].disabled = !generating && !refs["message-input"].value.trim();
    refs["send-button"].title = generating ? "停止" : "发送";
    refs["send-button"].setAttribute("aria-label", generating ? "停止" : "发送");
    refs["message-input"].disabled = generating;
    refs["new-chat"].disabled = generating;
    refs["model-select"].disabled = generating;
    refs["thinking-toggle"].disabled = generating;
    updateThinkingControls();
  }

  function stopGeneration() {
    state.controller?.abort();
  }

  function conversationRequestMessages(
    conversation,
    endExclusive = conversation.messages.length
  ) {
    const messages = [];
    if (state.settings.systemPrompt.trim()) {
      messages.push({ role: "system", content: state.settings.systemPrompt.trim() });
    }
    for (const message of conversation.messages.slice(0, endExclusive)) {
      if (message.role === "assistant" && !message.content && !message.reasoning) continue;
      const requestMessage = {
        role: message.role,
        content: message.content,
      };
      if (
        message.role === "assistant" &&
        message.reasoning &&
        !state.settings.excludeReasoningFromContext
      ) {
        requestMessage.reasoning_content = message.reasoning;
      }
      messages.push(requestMessage);
    }
    return messages;
  }

  function generationRequestBody(messages) {
    const chatTemplateKwargs = {
      enable_thinking: state.settings.enableThinking,
    };
    const reasoningCapability =
      state.status?.chat_template_capabilities?.reasoning_effort;
    const supportedReasoningEfforts = Array.isArray(reasoningCapability?.values)
      ? reasoningCapability.values
      : [];
    if (
      state.settings.enableThinking &&
      supportedReasoningEfforts.includes(state.settings.reasoningEffort)
    ) {
      chatTemplateKwargs.reasoning_effort = state.settings.reasoningEffort;
    }
    return {
      model: state.settings.model || state.models[0] || "mfq",
      messages,
      max_tokens: state.settings.maxTokens,
      temperature: state.settings.temperature,
      top_p: state.settings.topP,
      top_k: state.settings.topK,
      repetition_penalty: state.settings.repetitionPenalty,
      presence_penalty: state.settings.presencePenalty,
      frequency_penalty: state.settings.frequencyPenalty,
      stream: true,
      stream_options: { include_usage: true },
      reasoning_format: "auto",
      chat_template_kwargs: chatTemplateKwargs,
    };
  }

  function updateConversationTitle(conversation, content) {
    if (conversation.messages.filter((item) => item.role === "user").length !== 1) return;
    const normalized = content.replace(/\s+/g, " ").trim();
    conversation.title = normalized.length > 28
      ? `${normalized.slice(0, 28)}…`
      : normalized || "新对话";
  }

  function appendSsePayload(payload, assistant, streamState) {
    if (!payload || payload === "[DONE]") return;
    let event;
    try {
      event = JSON.parse(payload);
    } catch {
      return;
    }
    if (event.error) {
      throw new Error(event.error.message || "服务返回了错误");
    }
    const choice = Array.isArray(event.choices) ? event.choices[0] : null;
    if (choice) {
      const delta = choice.delta || {};
      const content = typeof delta.content === "string"
        ? delta.content
        : typeof choice.text === "string"
          ? choice.text
          : "";
      const reasoning = typeof delta.reasoning_content === "string"
        ? delta.reasoning_content
        : typeof delta.reasoning === "string"
          ? delta.reasoning
          : "";
      if ((content || reasoning) && !streamState.firstTokenAt) {
        streamState.firstTokenAt = performance.now();
      }
      assistant.content += content;
      assistant.reasoning += reasoning;
      if (choice.finish_reason) streamState.finishReason = choice.finish_reason;
    }
    if (event.usage && typeof event.usage === "object") {
      streamState.usage = event.usage;
    }
  }

  async function consumeSse(response, assistant, streamState) {
    if (!response.body) throw new Error("浏览器未收到流式响应体");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      buffer = buffer.replace(/\r\n/g, "\n");

      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const eventBlock = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = eventBlock
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (data === "[DONE]") return;
        if (data) {
          appendSsePayload(data, assistant, streamState);
          scheduleMessageRender();
        }
        boundary = buffer.indexOf("\n\n");
      }
      if (done) break;
    }

    const trailing = buffer.trim();
    if (trailing) {
      const data = trailing
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data && data !== "[DONE]") appendSsePayload(data, assistant, streamState);
    }
  }

  async function generateAssistant(
    conversation,
    assistant,
    requestMessages,
    options = {}
  ) {
    state.editingMessage = null;
    state.generatingMessage = assistant;
    setGenerating(true);
    renderConversationList();
    renderMessages({ scroll: options.followOutput !== false });

    const controller = new AbortController();
    state.controller = controller;
    const streamState = {
      startedAt: performance.now(),
      firstTokenAt: 0,
      finishReason: "",
      usage: null,
    };

    try {
      const response = await fetch(apiUrl("/v1/chat/completions"), {
        method: "POST",
        headers: requestHeaders(true),
        body: JSON.stringify(generationRequestBody(requestMessages)),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await responseError(response));
      setConnection(true, "在线");
      await consumeSse(response, assistant, streamState);
      if (
        !assistant.content &&
        assistant.reasoning &&
        streamState.finishReason === "length"
      ) {
        showToast(
          "已达到生成上限，模型仍处于思考阶段；可提高最大 Token 或关闭深度思考。",
          true
        );
      }
      if (!assistant.content && !assistant.reasoning && options.restoreOnFailure) {
        Object.assign(assistant, options.restoreOnFailure);
        showToast("未生成新内容，已恢复原回答。", true);
      }
      conversation.updatedAt = Date.now();
    } catch (error) {
      if (options.restoreOnFailure) {
        Object.assign(assistant, options.restoreOnFailure);
      }
      if (error?.name === "AbortError") {
        if (
          !options.restoreOnFailure &&
          options.removeIfEmptyOnAbort &&
          !assistant.content &&
          !assistant.reasoning
        ) {
          const index = conversation.messages.indexOf(assistant);
          if (index >= 0) conversation.messages.splice(index, 1);
        }
        showToast("已停止生成。");
      } else {
        if (!options.restoreOnFailure) {
          assistant.error = true;
          assistant.content =
            assistant.content || `请求失败：${error?.message || "未知错误"}`;
        }
        setConnection(false, "连接失败");
        showToast(error?.message || "请求失败", true);
      }
    } finally {
      state.controller = null;
      state.generatingMessage = null;
      setGenerating(false);
      conversation.updatedAt = Date.now();
      persistState();
      renderConversationList();
      renderMessages();
      refs["message-input"].focus();
      window.setTimeout(() => refreshStatus({ quiet: true }), 120);
    }
  }

  async function rerollAssistant(conversation, assistant) {
    if (state.generating || assistant.role !== "assistant") return;
    const assistantIndex = conversation.messages.indexOf(assistant);
    if (assistantIndex < 0) return;

    const snapshot = {
      content: assistant.content,
      reasoning: assistant.reasoning,
      createdAt: assistant.createdAt,
      error: assistant.error,
    };
    const requestMessages =
      conversationRequestMessages(conversation, assistantIndex);
    assistant.content = "";
    assistant.reasoning = "";
    assistant.createdAt = Date.now();
    assistant.error = false;
    state.reasoningOpenState.delete(assistant);
    state.followOutput =
      assistantIndex === conversation.messages.length - 1;
    await generateAssistant(conversation, assistant, requestMessages, {
      followOutput: state.followOutput,
      restoreOnFailure: snapshot,
    });
  }

  async function sendMessage() {
    if (state.generating) {
      stopGeneration();
      return;
    }
    const content = refs["message-input"].value.trim();
    if (!content) return;

    const conversation = activeConversation();
    if (!conversation) return;
    const userMessage = {
      role: "user",
      content,
      reasoning: "",
      createdAt: Date.now(),
      error: false,
    };
    conversation.messages.push(userMessage);
    updateConversationTitle(conversation, content);
    conversation.updatedAt = Date.now();

    const assistant = {
      role: "assistant",
      content: "",
      reasoning: "",
      createdAt: Date.now(),
      error: false,
    };
    conversation.messages.push(assistant);
    const assistantIndex = conversation.messages.length - 1;
    const requestMessages =
      conversationRequestMessages(conversation, assistantIndex);
    state.followOutput = true;
    refs["message-input"].value = "";
    resizeComposer();
    await generateAssistant(conversation, assistant, requestMessages, {
      followOutput: true,
      removeIfEmptyOnAbort: true,
    });
  }

  function renderModels() {
    const current = state.settings.model;
    refs["model-select"].replaceChildren();
    const models = state.models.length ? state.models : [current || "mfq"];
    for (const model of [...new Set(models.filter(Boolean))]) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      refs["model-select"].append(option);
    }
    if (!state.settings.model || !models.includes(state.settings.model)) {
      state.settings.model = models[0] || "mfq";
    }
    refs["model-select"].value = state.settings.model;
  }

  async function loadModels() {
    try {
      const payload = await fetchJson("/v1/models");
      state.models = Array.isArray(payload.data)
        ? payload.data.map((item) => item?.id).filter((id) => typeof id === "string")
        : [];
      renderModels();
      persistState();
      return true;
    } catch {
      renderModels();
      return false;
    }
  }

  function mergeFallbackStatus(health) {
    return {
      status: health?.status || "ok",
      model: health?.model || state.settings.model || state.models[0] || "mfq",
      model_type: health?.model_type || "--",
      sampling_defaults: health?.sampling_defaults || null,
      chat_template_capabilities:
        health?.chat_template_capabilities || null,
      max_context: null,
      uptime_seconds: null,
      active_requests: state.generating ? 1 : 0,
      total_requests: state.status?.total_requests || 0,
      failed_requests: state.status?.failed_requests || 0,
      total_prompt_tokens: state.status?.total_prompt_tokens || 0,
      total_completion_tokens: state.status?.total_completion_tokens || 0,
      last_request: state.status?.last_request || null,
      limited: true,
    };
  }

  async function refreshStatus(options = {}) {
    try {
      let payload;
      if (state.statusApiAvailable == null) {
        try {
          const root = await fetchJson("/");
          state.statusApiAvailable = Array.isArray(root.endpoints)
            ? root.endpoints.includes("/api/status")
            : true;
        } catch {
          state.statusApiAvailable = true;
        }
      }
      if (state.statusApiAvailable) {
        try {
          payload = await fetchJson("/api/status");
        } catch (statusError) {
          if (statusError.status === 404) state.statusApiAvailable = false;
          const health = await fetchJson("/health");
          payload = mergeFallbackStatus(health);
          if (!options.quiet && statusError.status === 401) {
            throw statusError;
          }
        }
      } else {
        const health = await fetchJson("/health");
        payload = mergeFallbackStatus(health);
      }
      state.status = payload;
      applyServerSamplingDefaults(payload);
      setConnection(
        true,
        payload.reloading
          ? "重载中"
          : payload.active_requests > 0
            ? "生成中"
            : "在线"
      );
      if (!state.models.length) await loadModels();
      updateMonitor();
    } catch (error) {
      setConnection(false, error?.message?.includes("401") ? "需要 API Key" : "离线");
      if (!options.quiet) showToast(error?.message || "无法连接服务", true);
      updateMonitor();
    }
  }

  function updateMonitor() {
    const status = state.status || {};
    const last = status.last_request || null;
    const prefillTps = Number(last?.prefill_tps);
    const prefillMs = Number(last?.prefill_ms);
    const decodeTps = Number(last?.decode_tps);
    const ttft = Number(last?.ttft_ms);
    const active = Number(status.active_requests) || 0;
    const promptTokens = Number(status.total_prompt_tokens) || 0;
    const completionTokens = Number(status.total_completion_tokens) || 0;

    refs["active-request-count"].textContent = String(active);
    refs["top-ttft"].textContent = Number.isFinite(ttft) ? `${formatNumber(ttft, 1)} ms` : "--";
    refs["top-prefill-tps"].textContent = Number.isFinite(prefillTps)
      ? formatNumber(prefillTps, 1)
      : "--";
    refs["top-tps"].textContent = Number.isFinite(decodeTps) ? formatNumber(decodeTps, 1) : "--";
    refs["metric-prefill-tps"].textContent = Number.isFinite(prefillTps)
      ? formatNumber(prefillTps, 1)
      : "--";
    refs["metric-decode-tps"].textContent = Number.isFinite(decodeTps) ? formatNumber(decodeTps, 1) : "--";
    refs["metric-ttft"].textContent = Number.isFinite(ttft) ? formatNumber(ttft, 1) : "--";
    refs["metric-requests"].textContent = formatNumber(status.total_requests || 0);
    refs["metric-active"].textContent = `${formatNumber(active)} active`;
    refs["metric-tokens"].textContent = formatNumber(promptTokens + completionTokens);
    updateThinkingControls();
    refs["runtime-model"].textContent = status.model || state.settings.model || "--";
    refs["runtime-type"].textContent = status.model_type || "--";
    refs["runtime-context"].textContent = status.max_context
      ? formatNumber(status.max_context)
      : "--";
    refs["runtime-uptime"].textContent = status.uptime_seconds == null
      ? "--"
      : formatDuration(status.uptime_seconds);
    refs["runtime-failed"].textContent = formatNumber(status.failed_requests || 0);
    refs["monitor-updated"].textContent = state.connected
      ? `最后更新 ${new Intl.DateTimeFormat("zh-CN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }).format(new Date())}${status.limited ? " · 旧版状态接口" : ""}`
      : "服务不可用";

    if (last) {
      refs["last-request-id"].textContent = last.id || "最近请求";
      refs["last-request-state"].textContent = last.finish_reason || "Complete";
      refs["last-prompt-tokens"].textContent = formatNumber(last.prompt_tokens);
      refs["last-completion-tokens"].textContent = formatNumber(last.completion_tokens);
      refs["last-prefill-tps"].textContent = Number.isFinite(prefillTps)
        ? formatNumber(prefillTps, 1)
        : "--";
      refs["last-prefill-ms"].textContent = Number.isFinite(prefillMs)
        ? `${formatNumber(prefillMs, 1)} ms`
        : "-- ms";
      refs["last-generation-ms"].textContent = formatNumber(last.generation_ms, 1);
      refs["last-finish-reason"].textContent = last.finish_reason || "--";
      if (last.id && last.id !== state.lastMetricRequestId && Number.isFinite(decodeTps)) {
        state.lastMetricRequestId = last.id;
        state.metricSeries.push(decodeTps);
        state.metricSeries = state.metricSeries.slice(-MAX_METRIC_POINTS);
      }
    } else {
      refs["last-request-id"].textContent = "还没有完成的请求";
      refs["last-request-state"].textContent = active ? "Running" : "Idle";
      refs["last-prompt-tokens"].textContent = "--";
      refs["last-completion-tokens"].textContent = "--";
      refs["last-prefill-tps"].textContent = "--";
      refs["last-prefill-ms"].textContent = "-- ms";
      refs["last-generation-ms"].textContent = "--";
      refs["last-finish-reason"].textContent = "--";
    }

    refs["chart-current"].textContent = Number.isFinite(decodeTps)
      ? `${formatNumber(decodeTps, 1)} tok/s`
      : "-- tok/s";
    drawChart();
  }

  function updateThinkingControls() {
    const capability =
      state.status?.chat_template_capabilities?.reasoning_effort;
    const capabilityKnown = state.status !== null;
    const advertisedValues = Array.isArray(capability?.values)
      ? capability.values.filter((value) => value === "high" || value === "max")
      : [];
    const supported = Boolean(capability?.supported) && advertisedValues.length > 0;
    const select = refs["reasoning-effort-select"];
    const control = refs["reasoning-effort-control"];
    if (!select || !control) return;

    for (const option of select.options) {
      if (!option.value) continue;
      const available = advertisedValues.includes(option.value);
      option.hidden = !available;
      option.disabled = !available;
    }
    if (
      capabilityKnown &&
      state.settings.reasoningEffort &&
      !advertisedValues.includes(state.settings.reasoningEffort)
    ) {
      state.settings.reasoningEffort = "";
      persistState();
    }
    select.value = state.settings.reasoningEffort;
    control.hidden = !supported || !state.settings.enableThinking;
    select.disabled =
      state.generating || !state.settings.enableThinking || !supported;
  }

  function drawChart() {
    const canvas = refs["throughput-chart"];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);

    const width = rect.width;
    const height = rect.height;
    const pad = { top: 12, right: 12, bottom: 24, left: 42 };
    const plotWidth = Math.max(1, width - pad.left - pad.right);
    const plotHeight = Math.max(1, height - pad.top - pad.bottom);
    const values = state.metricSeries.length ? state.metricSeries : [0];
    const rawMax = Math.max(...values, 1);
    const maxValue = Math.ceil(rawMax / 25) * 25;

    ctx.clearRect(0, 0, width, height);
    ctx.font = "10px Segoe UI, sans-serif";
    ctx.fillStyle = "#7a8883";
    ctx.strokeStyle = "#e4eae7";
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let line = 0; line <= 4; line += 1) {
      const y = pad.top + (plotHeight * line) / 4;
      const label = maxValue - (maxValue * line) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, Math.round(y) + 0.5);
      ctx.lineTo(width - pad.right, Math.round(y) + 0.5);
      ctx.stroke();
      ctx.fillText(formatNumber(label), pad.left - 8, y);
    }

    if (state.metricSeries.length < 2) {
      ctx.fillStyle = "#95a19d";
      ctx.textAlign = "center";
      ctx.fillText("完成请求后显示吞吐趋势", pad.left + plotWidth / 2, pad.top + plotHeight / 2);
      return;
    }

    const points = state.metricSeries.map((value, index) => ({
      x: pad.left + (plotWidth * index) / Math.max(1, state.metricSeries.length - 1),
      y: pad.top + plotHeight - (plotHeight * value) / maxValue,
    }));

    ctx.beginPath();
    ctx.moveTo(points[0].x, pad.top + plotHeight);
    for (const point of points) ctx.lineTo(point.x, point.y);
    ctx.lineTo(points.at(-1).x, pad.top + plotHeight);
    ctx.closePath();
    ctx.fillStyle = "rgb(59 114 185 / 9%)";
    ctx.fill();

    ctx.beginPath();
    points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.strokeStyle = "#3b72b9";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();

    const last = points.at(-1);
    ctx.beginPath();
    ctx.arc(last.x, last.y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#3b72b9";
    ctx.stroke();
  }

  function switchView(view) {
    state.view = view === "monitor" ? "monitor" : "chat";
    refs["chat-view"].classList.toggle("is-visible", state.view === "chat");
    refs["monitor-view"].classList.toggle("is-visible", state.view === "monitor");
    for (const item of refs.navItems) {
      item.classList.toggle("is-active", item.dataset.view === state.view);
    }
    if (state.view === "monitor") {
      refreshStatus({ quiet: true });
      requestAnimationFrame(drawChart);
    }
  }

  function openSettings() {
    populateSettings();
    refs["settings-panel"].classList.add("is-visible");
    refs["settings-panel"].setAttribute("aria-hidden", "false");
    refs["settings-scrim"].classList.add("is-visible");
    window.setTimeout(() => refs["setting-endpoint"].focus(), 100);
  }

  function closeSettings() {
    refs["settings-panel"].classList.remove("is-visible");
    refs["settings-panel"].setAttribute("aria-hidden", "true");
    refs["settings-scrim"].classList.remove("is-visible");
  }

  function openSidebar() {
    refs.sidebar.classList.add("is-visible");
    refs["sidebar-scrim"].classList.add("is-visible");
  }

  function closeSidebar() {
    refs.sidebar.classList.remove("is-visible");
    refs["sidebar-scrim"].classList.remove("is-visible");
  }

  function syncRangeOutputs() {
    refs["temperature-value"].value = Number(refs["setting-temperature"].value).toFixed(2);
    refs["top-p-value"].value = Number(refs["setting-top-p"].value).toFixed(2);
    refs["repetition-value"].value = Number(refs["setting-repetition"].value).toFixed(2);
    refs["presence-value"].value = Number(refs["setting-presence"].value).toFixed(2);
    refs["frequency-value"].value = Number(refs["setting-frequency"].value).toFixed(2);
  }

  function setPresetActive(name) {
    refs.presetButtons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.preset === name);
    });
  }

  function populateSettings() {
    const settings = state.settings;
    refs["setting-endpoint"].value = settings.endpoint;
    refs["setting-api-key"].value = state.apiKey;
    refs["setting-system-prompt"].value = settings.systemPrompt;
    refs["setting-exclude-reasoning"].checked =
      settings.excludeReasoningFromContext;
    const serverContext = Number(state.status?.max_context);
    const contextCapacity = Number(state.status?.context_capacity);
    const contextLimit = Number.isFinite(contextCapacity) && contextCapacity > 0
      ? Math.floor(contextCapacity)
      : 32768;
    refs["setting-context-window"].max = String(contextLimit);
    refs["setting-context-window"].value = String(
      Number.isFinite(serverContext) && serverContext > 0
        ? Math.min(Math.floor(serverContext), contextLimit)
        : contextLimit
    );
    refs["setting-context-limit"].textContent =
      `当前已加载 ${formatNumber(serverContext || contextLimit)}，模型上限 ${formatNumber(contextLimit)} tokens。重载会卸载当前 runtime 并重新分配 KV cache。`;
    refs["setting-max-tokens"].value = String(settings.maxTokens);
    refs["setting-temperature"].value = String(settings.temperature);
    refs["setting-top-p"].value = String(settings.topP);
    refs["setting-top-k"].value = String(settings.topK);
    refs["setting-repetition"].value = String(settings.repetitionPenalty);
    refs["setting-presence"].value = String(settings.presencePenalty);
    refs["setting-frequency"].value = String(settings.frequencyPenalty);
    setPresetActive(settings.preset);
    syncRangeOutputs();
  }

  function applyPreset(name) {
    const preset = presets[name];
    if (!preset) return;
    refs["setting-temperature"].value = String(preset.temperature);
    refs["setting-top-p"].value = String(preset.topP);
    refs["setting-top-k"].value = String(preset.topK);
    refs["setting-repetition"].value = String(preset.repetitionPenalty);
    setPresetActive(name);
    syncRangeOutputs();
  }

  function numberInput(ref, fallback, min, max) {
    const value = Number(ref.value);
    if (!Number.isFinite(value)) return fallback;
    return Math.min(max, Math.max(min, value));
  }

  function saveSettings() {
    const oldEndpoint = state.settings.endpoint;
    const activePreset = refs.presetButtons.find((button) => button.classList.contains("is-active"));
    const nextSettings = {
      ...state.settings,
      endpoint: normalizeEndpoint(refs["setting-endpoint"].value),
      model: refs["model-select"].value || state.settings.model,
      systemPrompt: refs["setting-system-prompt"].value,
      excludeReasoningFromContext:
        refs["setting-exclude-reasoning"].checked,
      maxTokens: Math.round(numberInput(refs["setting-max-tokens"], 4096, 1, 8192)),
      temperature: numberInput(refs["setting-temperature"], 0.7, 0, 2),
      topP: numberInput(refs["setting-top-p"], 0.8, 0.05, 1),
      topK: Math.round(numberInput(refs["setting-top-k"], 20, 0, 1024)),
      repetitionPenalty: numberInput(refs["setting-repetition"], 1, 0.5, 2),
      presencePenalty: numberInput(refs["setting-presence"], 0, -2, 2),
      frequencyPenalty: numberInput(refs["setting-frequency"], 0, -2, 2),
      preset: activePreset?.dataset.preset || "custom",
    };
    nextSettings.samplingCustomized = state.samplingDefaults
      ? !samplingMatches(nextSettings, state.samplingDefaults)
      : true;
    state.settings = nextSettings;
    state.apiKey = refs["setting-api-key"].value.trim();
    persistState();
    closeSettings();
    setConnection(false, "正在连接");
    if (oldEndpoint !== state.settings.endpoint) {
      state.models = [];
      state.status = null;
      state.statusApiAvailable = null;
      state.metricSeries = [];
      state.lastMetricRequestId = "";
    }
    renderModels();
    refreshStatus();
    showToast("推理设置已应用。");
  }

  async function reloadModel() {
    if (state.generating) {
      showToast("请先停止当前生成再重载模型。", true);
      return;
    }
    const capacityValue = Number(state.status?.context_capacity);
    const capacity = Number.isFinite(capacityValue) && capacityValue > 0
      ? Math.floor(capacityValue)
      : 32768;
    const contextSize = Math.round(numberInput(
      refs["setting-context-window"],
      Number(state.status?.max_context) || capacity,
      512,
      capacity
    ));
    const currentContext = Number(state.status?.max_context) || 0;
    const action = currentContext === contextSize
      ? `以当前 ${formatNumber(contextSize)} token 上下文重新加载模型？`
      : `将模型从 ${formatNumber(currentContext)} token 上下文重载为 ${formatNumber(contextSize)}？`;
    if (!window.confirm(`${action}\n\n重载期间不能生成，通常需要约 1–2 分钟。`)) {
      return;
    }

    const button = refs["reload-model"];
    const previousLabel = button.textContent;
    button.disabled = true;
    button.textContent = "正在重载模型…";
    refs["setting-context-window"].disabled = true;
    setConnection(true, "重载中");
    try {
      const payload = await fetchJson("/api/reload", {
        method: "POST",
        body: JSON.stringify({ context_size: contextSize }),
      });
      state.status = {
        ...(state.status || {}),
        ...payload,
        reloading: false,
      };
      updateMonitor();
      populateSettings();
      setConnection(true, "在线");
      showToast(`模型已按 ${formatNumber(payload.max_context)} token 上下文重载。`);
    } catch (error) {
      setConnection(false, "重载失败");
      showToast(error?.message || "模型重载失败", true);
      await refreshStatus({ quiet: true });
    } finally {
      button.disabled = false;
      button.textContent = previousLabel;
      refs["setting-context-window"].disabled = false;
    }
  }

  function resetSettingsForm() {
    const endpoint = state.settings.endpoint || defaultEndpoint;
    state.settings = { ...defaultSettings, endpoint };
    if (state.samplingDefaults) {
      Object.assign(state.settings, state.samplingDefaults);
      state.settings.preset = "custom";
    }
    populateSettings();
  }

  function exportConversation() {
    const conversation = activeConversation();
    if (!conversation || !conversation.messages.length) {
      showToast("当前对话没有可导出的内容。", true);
      return;
    }
    const lines = [`# ${conversation.title}`, ""];
    for (const message of conversation.messages) {
      lines.push(message.role === "user" ? "## 用户" : "## MFQ", "");
      if (message.reasoning) {
        lines.push("<details>", "<summary>思考过程</summary>", "", message.reasoning, "", "</details>", "");
      }
      lines.push(message.content, "");
    }
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${conversation.title.replace(/[\\/:*?"<>|]/g, "_")}.md`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function bindEvents() {
    refs["new-chat"].addEventListener("click", createConversation);
    refs["clear-history"].addEventListener("click", () => {
      if (state.generating) return;
      state.conversations = [newConversation()];
      state.activeId = state.conversations[0].id;
      state.editingMessage = null;
      state.renamingConversationId = "";
      state.followOutput = true;
      persistState();
      renderConversationList();
      renderMessages();
      showToast("对话历史已清空。");
    });
    refs.navItems.forEach((item) => {
      item.addEventListener("click", () => {
        switchView(item.dataset.view);
        closeSidebar();
      });
    });
    refs["open-sidebar"].addEventListener("click", openSidebar);
    refs["close-sidebar"].addEventListener("click", closeSidebar);
    refs["sidebar-scrim"].addEventListener("click", closeSidebar);
    refs["open-settings"].addEventListener("click", openSettings);
    refs["close-settings"].addEventListener("click", closeSettings);
    refs["settings-scrim"].addEventListener("click", closeSettings);
    refs["save-settings"].addEventListener("click", saveSettings);
    refs["reload-model"].addEventListener("click", reloadModel);
    refs["reset-settings"].addEventListener("click", resetSettingsForm);
    refs["export-chat"].addEventListener("click", exportConversation);
    refs["refresh-status"].addEventListener("click", () => refreshStatus());
    refs["model-select"].addEventListener("change", () => {
      state.settings.model = refs["model-select"].value;
      persistState();
    });
    refs["thinking-toggle"].addEventListener("click", () => {
      state.settings.enableThinking = !state.settings.enableThinking;
      refs["thinking-toggle"].setAttribute("aria-pressed", String(state.settings.enableThinking));
      updateThinkingControls();
      persistState();
    });
    refs["reasoning-effort-select"].addEventListener("change", () => {
      state.settings.reasoningEffort =
        refs["reasoning-effort-select"].value;
      persistState();
    });
    refs["message-input"].addEventListener("input", resizeComposer);
    refs["message-scroller"].addEventListener("scroll", () => {
      const scroller = refs["message-scroller"];
      const distanceFromBottom =
        scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop;
      state.followOutput = distanceFromBottom <= 48;
    }, { passive: true });
    refs["message-input"].addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        sendMessage();
      }
    });
    refs["composer-form"].addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage();
    });
    refs.presetButtons.forEach((button) => {
      button.addEventListener("click", () => applyPreset(button.dataset.preset));
    });
    [
      "setting-temperature", "setting-top-p", "setting-repetition",
      "setting-presence", "setting-frequency",
    ].forEach((id) => refs[id].addEventListener("input", syncRangeOutputs));
    window.addEventListener("resize", () => requestAnimationFrame(drawChart));
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      if (refs["settings-panel"].classList.contains("is-visible")) closeSettings();
      else if (refs.sidebar.classList.contains("is-visible")) closeSidebar();
      else if (state.generating) stopGeneration();
    });
  }

  function initialize() {
    queryRefs();
    loadState();
    bindEvents();
    renderModels();
    renderConversationList();
    renderMessages({ scroll: false });
    refs["thinking-toggle"].setAttribute("aria-pressed", String(state.settings.enableThinking));
    updateThinkingControls();
    refs["setting-endpoint"].value = state.settings.endpoint;
    resizeComposer();
    updateMonitor();
    setConnection(false, "正在连接");
    refreshStatus({ quiet: true });
    state.pollTimer = window.setInterval(() => refreshStatus({ quiet: true }), 2000);
  }

  document.addEventListener("DOMContentLoaded", initialize, { once: true });
})();
