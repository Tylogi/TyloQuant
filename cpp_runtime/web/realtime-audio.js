(() => {
  "use strict";

  const INPUT_RATE = 16000;
  const OUTPUT_RATE = 24000;
  const CHUNK_SAMPLES = INPUT_RATE;
  const SPEAK_TOKENS = 20;
  const PLAYBACK_DELAY_SECONDS = 0.2;
  const PLAYBACK_STORAGE_KEY = "mfq.console.voice-playback";
  const DUPLEX_STORAGE_KEY = "mfq.console.full-duplex";
  const element = (id) => document.getElementById(id);

  const labels = {
    input: ["语音输入", "Voice input"],
    stopInput: ["停止语音输入", "Stop voice input"],
    connecting: ["语音连接中", "Connecting voice"],
    processing: ["语音处理中", "Processing voice"],
    playbackOn: ["关闭实时语音播放", "Mute voice responses"],
    playbackOff: ["开启实时语音播放", "Play voice responses"],
    fullDuplex: ["全双工模式", "Full-duplex mode"],
    halfDuplex: ["半双工模式", "Half-duplex mode"],
    fullDuplexShort: ["全双工", "Full duplex"],
    halfDuplexShort: ["半双工", "Half duplex"],
    failed: ["语音连接失败", "Voice connection failed"],
  };

  const state = {
    socket: null,
    stream: null,
    inputContext: null,
    outputContext: null,
    source: null,
    capture: null,
    pending: [],
    pendingLength: 0,
    outboundAudio: [],
    pendingText: [],
    heldHalfDuplexChunk: null,
    halfDuplexPendingSteps: 0,
    awaitingHalfDuplexResponse: false,
    sessionReady: false,
    playAt: 0,
    playing: new Set(),
    playbackEnabled: localStorage.getItem(PLAYBACK_STORAGE_KEY) !== "false",
    fullDuplexEnabled: localStorage.getItem(DUPLEX_STORAGE_KEY) !== "false",
    duplexSupported: false,
    stopping: false,
  };

  function bridge() {
    return globalThis.MFQRealtimeBridge;
  }

  function label(name) {
    return labels[name][document.documentElement.lang === "en" ? 1 : 0];
  }

  function setInputState(kind) {
    const button = element("voice-input-toggle");
    const active = kind === "active";
    button.disabled = kind === "connecting" || kind === "processing";
    button.setAttribute("aria-pressed", String(active));
    button.title = label(
      kind === "active"
        ? "stopInput"
        : kind === "connecting"
          ? "connecting"
          : kind === "processing"
            ? "processing"
            : "input"
    );
    button.setAttribute("aria-label", button.title);
  }

  function syncPlaybackButton() {
    const button = element("voice-playback-toggle");
    button.setAttribute("aria-pressed", String(state.playbackEnabled));
    button.title = label(state.playbackEnabled ? "playbackOn" : "playbackOff");
    button.setAttribute("aria-label", button.title);
  }

  function syncDuplexButton() {
    const button = element("duplex-mode-toggle");
    button.hidden = !state.duplexSupported;
    button.disabled = Boolean(state.inputContext || state.awaitingHalfDuplexResponse);
    button.setAttribute("aria-pressed", String(state.fullDuplexEnabled));
    button.title = label(state.fullDuplexEnabled ? "fullDuplex" : "halfDuplex");
    button.setAttribute("aria-label", button.title);
    const text = button.querySelector(".duplex-mode-label");
    if (text) {
      text.textContent = label(
        state.fullDuplexEnabled ? "fullDuplexShort" : "halfDuplexShort"
      );
    }
  }

  function float32ToBase64(values) {
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
    }
    return btoa(binary);
  }

  function base64ToFloat32(value) {
    const binary = atob(value);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return new Float32Array(bytes.buffer);
  }

  function resample(input, sourceRate, targetRate) {
    if (sourceRate === targetRate) return new Float32Array(input);
    const length = Math.round(input.length * targetRate / sourceRate);
    const output = new Float32Array(length);
    const scale = sourceRate / targetRate;
    for (let index = 0; index < length; index += 1) {
      const position = index * scale;
      const left = Math.floor(position);
      const right = Math.min(left + 1, input.length - 1);
      const mix = position - left;
      output[index] = input[left] * (1 - mix) + input[right] * mix;
    }
    return output;
  }

  function sendInput(samples, options = {}) {
    if (state.socket?.readyState !== WebSocket.OPEN || !state.sessionReady) {
      state.outboundAudio.push({samples, options});
      if (state.outboundAudio.length > 64) state.outboundAudio.shift();
      return;
    }
    const input = {
      audio: float32ToBase64(samples),
      max_new_speak_tokens: SPEAK_TOKENS,
    };
    if (options.forceListen) input.force_listen = true;
    if (options.forceSpeak) input.force_speak = true;
    state.socket.send(JSON.stringify({
      type: "input.append",
      input,
    }));
  }

  function sendText(value) {
    if (typeof value !== "string" || !value.trim()) return;
    if (state.socket?.readyState !== WebSocket.OPEN || !state.sessionReady) {
      state.pendingText.push(value);
      return;
    }
    state.socket.send(JSON.stringify({
      type: "input.append",
      input: {
        text: value,
        max_new_speak_tokens: SPEAK_TOKENS,
      },
    }));
  }

  function updateInputLevel(samples) {
    let energy = 0;
    for (let index = 0; index < samples.length; index += 1) {
      energy += samples[index] * samples[index];
    }
    const level = Math.min(1, Math.sqrt(energy / Math.max(1, samples.length)) * 10);
    const button = element("voice-input-toggle");
    button.style.setProperty("--voice-scale", (0.78 + level * 0.42).toFixed(3));
    button.style.setProperty("--voice-glow", `${(7 + level * 9).toFixed(1)}px`);
  }

  function queueInput(samples) {
    if (!state.inputContext) return;
    updateInputLevel(samples);
    const converted = resample(samples, state.inputContext.sampleRate, INPUT_RATE);
    state.pending.push(converted);
    state.pendingLength += converted.length;
    while (state.pendingLength >= CHUNK_SAMPLES) {
      const chunk = new Float32Array(CHUNK_SAMPLES);
      let offset = 0;
      while (offset < CHUNK_SAMPLES) {
        const head = state.pending[0];
        const count = Math.min(head.length, CHUNK_SAMPLES - offset);
        chunk.set(head.subarray(0, count), offset);
        offset += count;
        if (count === head.length) state.pending.shift();
        else state.pending[0] = head.subarray(count);
        state.pendingLength -= count;
      }
      if (state.fullDuplexEnabled) {
        sendInput(chunk);
      } else {
        if (state.heldHalfDuplexChunk) {
          sendInput(state.heldHalfDuplexChunk, {forceListen: true});
          state.halfDuplexPendingSteps += 1;
        }
        state.heldHalfDuplexChunk = chunk;
      }
    }
  }

  function drainPendingAudio() {
    if (!state.pendingLength) return null;
    const chunk = new Float32Array(CHUNK_SAMPLES);
    let offset = 0;
    while (state.pending.length) {
      const head = state.pending.shift();
      const count = Math.min(head.length, CHUNK_SAMPLES - offset);
      chunk.set(head.subarray(0, count), offset);
      offset += count;
      if (count < head.length) {
        state.pending.unshift(head.subarray(count));
        break;
      }
    }
    state.pending = [];
    state.pendingLength = 0;
    return chunk;
  }

  function stopPlayback() {
    for (const source of state.playing) {
      try {
        source.stop();
      } catch {
      }
    }
    state.playing.clear();
    if (state.outputContext) state.playAt = state.outputContext.currentTime;
    element("voice-playback-toggle").classList.remove("is-speaking");
  }

  function playAudio(samples) {
    if (!state.playbackEnabled || !samples.length || !state.outputContext) return;
    const context = state.outputContext;
    const buffer = context.createBuffer(1, samples.length, OUTPUT_RATE);
    buffer.copyToChannel(samples, 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    state.playAt = Math.max(
      state.playAt,
      context.currentTime + PLAYBACK_DELAY_SECONDS
    );
    source.start(state.playAt);
    state.playAt += buffer.duration;
    state.playing.add(source);
    element("voice-playback-toggle").classList.add("is-speaking");
    source.onended = () => {
      state.playing.delete(source);
      if (!state.playing.size) {
        element("voice-playback-toggle").classList.remove("is-speaking");
      }
    };
  }

  async function startAudio() {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    state.inputContext = new AudioContext();
    if (!state.outputContext) {
      state.outputContext = new AudioContext({sampleRate: OUTPUT_RATE});
    }
    await Promise.all([state.inputContext.resume(), state.outputContext.resume()]);
    await state.inputContext.audioWorklet.addModule("pcm-capture-worklet.js");
    state.source = state.inputContext.createMediaStreamSource(state.stream);
    state.capture = new AudioWorkletNode(state.inputContext, "pcm-capture");
    const silent = state.inputContext.createGain();
    silent.gain.value = 0;
    state.source.connect(state.capture).connect(silent).connect(state.inputContext.destination);
    state.capture.port.onmessage = (event) => queueInput(event.data);
    syncDuplexButton();
  }

  async function stopInputCapture() {
    state.stream?.getTracks().forEach((track) => track.stop());
    state.stream = null;
    state.source?.disconnect();
    state.capture?.disconnect();
    await state.inputContext?.close().catch(() => {});
    state.inputContext = null;
    state.source = null;
    state.capture = null;
    element("voice-input-toggle").style.removeProperty("--voice-scale");
    element("voice-input-toggle").style.removeProperty("--voice-glow");
    syncDuplexButton();
  }

  async function finishHalfDuplexInput() {
    if (!state.inputContext || state.fullDuplexEnabled) return;
    await stopInputCapture();
    const tail = drainPendingAudio();
    if (tail && state.heldHalfDuplexChunk) {
      sendInput(state.heldHalfDuplexChunk, {forceListen: true});
      state.halfDuplexPendingSteps += 1;
      state.heldHalfDuplexChunk = null;
    }
    const finalChunk = tail || state.heldHalfDuplexChunk;
    state.heldHalfDuplexChunk = null;
    if (!finalChunk) {
      setInputState("idle");
      return;
    }
    sendInput(finalChunk, {forceSpeak: true});
    state.halfDuplexPendingSteps += 1;
    state.awaitingHalfDuplexResponse = true;
    setInputState("processing");
    syncDuplexButton();
  }

  function handleEvent(event, config) {
    if (event.type === "session.queue_done") {
      const payload = {};
      if (config.systemPrompt) payload.system_prompt = config.systemPrompt;
      state.socket.send(JSON.stringify({
        type: "session.init",
        payload,
      }));
    } else if (event.type === "session.created") {
      state.sessionReady = true;
      setInputState(state.awaitingHalfDuplexResponse ? "processing" : "active");
      for (const item of state.outboundAudio.splice(0)) {
        sendInput(item.samples, item.options);
      }
      for (const text of state.pendingText.splice(0)) sendText(text);
    } else if (event.kind === "text") {
      bridge()?.appendText(event.text || "");
    } else if (event.kind === "audio") {
      const samples = base64ToFloat32(event.audio);
      bridge()?.appendAudio(samples, Number(event.sample_rate) || OUTPUT_RATE);
      playAudio(samples);
    } else if (event.kind === "listen") {
      bridge()?.finishTurn();
    } else if (event.type === "response.step.done" &&
        !state.fullDuplexEnabled) {
      if (state.halfDuplexPendingSteps > 0) {
        state.halfDuplexPendingSteps -= 1;
      }
      if (state.awaitingHalfDuplexResponse &&
          state.halfDuplexPendingSteps === 0) {
        state.awaitingHalfDuplexResponse = false;
        bridge()?.finishTurn();
        setInputState("idle");
        syncDuplexButton();
      }
    } else if (event.type === "error") {
      throw new Error(event.error?.message || label("failed"));
    }
  }

  async function start() {
    if (state.inputContext || state.awaitingHalfDuplexResponse) return;
    if (state.socket?.readyState === WebSocket.OPEN && state.sessionReady) {
      await startAudio();
      setInputState("active");
      return;
    }
    if (state.socket) return;
    const config = bridge()?.begin();
    if (!config) return;
    setInputState("connecting");
    try {
      await startAudio();
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      state.socket = new WebSocket(`${protocol}://${location.host}/v1/realtime?mode=audio`);
      state.socket.onmessage = (message) => {
        try {
          handleEvent(JSON.parse(message.data), config);
        } catch (error) {
          console.error(error);
          bridge()?.notify(error.message || label("failed"), true);
          stop(false);
        }
      };
      state.socket.onerror = () => {
        bridge()?.notify(label("failed"), true);
      };
      state.socket.onclose = () => {
        if (!state.stopping) stop(false);
      };
    } catch (error) {
      console.error(error);
      bridge()?.notify(error.message || label("failed"), true);
      await stop(false);
    }
  }

  async function stop(sendClose = true) {
    if (state.stopping) return;
    state.stopping = true;
    if (sendClose && state.socket?.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify({type: "session.close", reason: "user_stop"}));
    }
    const socket = state.socket;
    state.socket = null;
    socket?.close();
    await stopInputCapture();
    stopPlayback();
    await state.outputContext?.close().catch(() => {});
    state.outputContext = null;
    state.pending = [];
    state.pendingLength = 0;
    state.outboundAudio = [];
    state.pendingText = [];
    state.heldHalfDuplexChunk = null;
    state.halfDuplexPendingSteps = 0;
    state.awaitingHalfDuplexResponse = false;
    state.sessionReady = false;
    state.playAt = 0;
    bridge()?.setActive(false);
    setInputState("idle");
    syncDuplexButton();
    state.stopping = false;
  }

  async function detectRealtimeCapability() {
    try {
      const response = await fetch("/realtime/capabilities", {cache: "no-store"});
      const payload = response.ok ? await response.json() : null;
      const available = payload?.available === true;
      state.duplexSupported = available &&
        payload?.model_capabilities?.features?.full_duplex === true;
      element("voice-input-toggle").hidden = !available;
      element("voice-playback-toggle").hidden = !available;
      syncDuplexButton();
      if (!available && (state.socket || state.inputContext)) stop(true);
    } catch {
      state.duplexSupported = false;
      element("voice-input-toggle").hidden = true;
      element("voice-playback-toggle").hidden = true;
      syncDuplexButton();
    }
  }

  element("voice-input-toggle").addEventListener("click", () => {
    if (state.inputContext && !state.fullDuplexEnabled) {
      finishHalfDuplexInput().catch((error) => {
        console.error(error);
        bridge()?.notify(error.message || label("failed"), true);
        stop(false);
      });
    } else if (state.inputContext) {
      stop(true);
    } else if (!state.awaitingHalfDuplexResponse) {
      start();
    }
  });
  element("voice-playback-toggle").addEventListener("click", () => {
    state.playbackEnabled = !state.playbackEnabled;
    localStorage.setItem(PLAYBACK_STORAGE_KEY, String(state.playbackEnabled));
    if (!state.playbackEnabled) stopPlayback();
    syncPlaybackButton();
  });
  element("duplex-mode-toggle").addEventListener("click", async () => {
    if (!state.duplexSupported || state.inputContext ||
        state.awaitingHalfDuplexResponse) return;
    if (state.socket) await stop(true);
    state.fullDuplexEnabled = !state.fullDuplexEnabled;
    localStorage.setItem(DUPLEX_STORAGE_KEY, String(state.fullDuplexEnabled));
    syncDuplexButton();
  });
  window.addEventListener("beforeunload", () => stop(true));
  document.addEventListener("mfq:realtime-stop", () => stop(true));
  document.addEventListener("mfq:realtime-text-input", (event) => {
    sendText(event.detail?.text || "");
  });
  document.addEventListener("mfq:model-status-changed", detectRealtimeCapability);
  syncPlaybackButton();
  syncDuplexButton();
  detectRealtimeCapability();
})();
