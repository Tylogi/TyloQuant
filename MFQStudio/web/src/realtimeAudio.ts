import { runtimeRealtimeUrl } from "./api";

const INPUT_RATE = 16_000;
const OUTPUT_RATE = 24_000;
const CHUNK_SAMPLES = INPUT_RATE;
const SPEAK_TOKENS = 20;
const MAX_RESPONSE_DRAIN_STEPS = 120;
const PLAYBACK_DELAY_SECONDS = 0.2;
const SPEECH_RMS_THRESHOLD = 0.015;
const SPEECH_START_SAMPLES = Math.round(INPUT_RATE * 0.08);
const SPEECH_END_SAMPLES = Math.round(INPUT_RATE * 0.7);
const SPEECH_PREROLL_SAMPLES = Math.round(INPUT_RATE * 0.25);
const MIN_USER_TURN_SAMPLES = Math.round(INPUT_RATE * 0.12);
const AUDIO_DATABASE = "mfq.studio.audio.v1";
const AUDIO_STORE = "clips";

export type VoiceState = "idle" | "connecting" | "listening" | "processing" | "error";

export interface RealtimeSessionConfig {
  sessionId: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
}

export interface VoiceTurn {
  id: string;
  sessionId: string;
  text: string;
  audio: Blob | null;
}

export interface VoiceInputTurn {
  id: string;
  sessionId: string;
  audio?: Blob | null;
}

interface BufferedVoiceTurn {
  id: string;
  sessionId: string;
  inputTurnId: string | null;
  text: string;
  audio: Float32Array[];
  audioRate: number;
}

export interface RealtimeCallbacks {
  onState(state: VoiceState): void;
  onLevel(level: number): void;
  onText(sessionId: string, text: string): void;
  onInputStart(turn: VoiceInputTurn): void;
  onInputEnd(turn: VoiceInputTurn): void;
  onTurn(turn: VoiceTurn): void;
  onError(message: string): void;
}

function float32ToBase64(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

function base64ToFloat32(value: string): Float32Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Float32Array(bytes.buffer);
}

function resample(input: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) return new Float32Array(input);
  const length = Math.round((input.length * targetRate) / sourceRate);
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

function wavBlob(chunks: Float32Array[], sampleRate: number): Blob {
  const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);
  const writeText = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + sampleCount * 2, true);
  writeText(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, sampleCount * 2, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (const raw of chunk) {
      const sample = Math.max(-1, Math.min(1, raw));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function audioDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(AUDIO_DATABASE, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(AUDIO_STORE)) {
        request.result.createObjectStore(AUDIO_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveVoiceClip(id: string, blob: Blob): Promise<void> {
  const database = await audioDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(AUDIO_STORE, "readwrite");
    transaction.objectStore(AUDIO_STORE).put(blob, id);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

export async function loadVoiceClip(id: string): Promise<Blob | null> {
  const database = await audioDatabase();
  return new Promise((resolve, reject) => {
    const request = database.transaction(AUDIO_STORE).objectStore(AUDIO_STORE).get(id);
    request.onsuccess = () => resolve((request.result as Blob | undefined) ?? null);
    request.onerror = () => reject(request.error);
  });
}

export class RealtimeAudioController {
  private socket: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private inputContext: AudioContext | null = null;
  private outputContext: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private capture: AudioWorkletNode | null = null;
  private pending: Float32Array[] = [];
  private pendingLength = 0;
  private outbound: Array<{
    samples: Float32Array;
    forceListen?: boolean;
    forceSpeak?: boolean;
    turnId: string | null;
  }> = [];
  private pendingText: Array<{ text: string; turnId: string }> = [];
  private pendingResponseTurns: Array<string | null> = [];
  private responseTurnIds = new Map<string, string | null>();
  private responseTurnOrder: string[] = [];
  private responseMessageIds = new Map<string, string>();
  private currentInputTurnId: string | null = null;
  private heldHalfDuplexChunk: Float32Array | null = null;
  private halfDuplexPendingSteps = 0;
  private awaitingHalfDuplexResponse = false;
  private responseDrainSteps = 0;
  private currentStepWasListen = false;
  private sessionReady = false;
  private clientSessionId: string | null = null;
  private stopping = false;
  private stopPromise: Promise<void> | null = null;
  private playAt = 0;
  private playing = new Set<AudioBufferSourceNode>();
  private activeTurn: BufferedVoiceTurn | null = null;
  private completedTurns = new Map<string, BufferedVoiceTurn>();
  private completedTurnOrder: string[] = [];
  private lastCompletedByInputTurn = new Map<string | null, string>();
  private inputTurnId: string | null = null;
  private inputTurnChunks: Float32Array[] = [];
  private inputTurnSamples = 0;
  private speechPreroll: Float32Array[] = [];
  private speechPrerollSamples = 0;
  private speechAboveSamples = 0;
  private speechSilenceSamples = 0;

  constructor(
    private readonly callbacks: RealtimeCallbacks,
    private playbackEnabled: boolean,
    private fullDuplexEnabled: boolean,
  ) {}

  get active(): boolean {
    return Boolean(this.inputContext || this.awaitingHalfDuplexResponse || this.socket);
  }

  get fullDuplex(): boolean {
    return this.fullDuplexEnabled;
  }

  setPlayback(enabled: boolean): void {
    this.playbackEnabled = enabled;
    if (!enabled) this.stopPlayback();
  }

  async setFullDuplex(enabled: boolean): Promise<void> {
    if (this.inputContext || this.awaitingHalfDuplexResponse) return;
    if (this.socket) await this.stop();
    this.fullDuplexEnabled = enabled;
  }

  async start(config: RealtimeSessionConfig): Promise<void> {
    await this.connect(config, true);
  }

  async submitText(value: string, config: RealtimeSessionConfig): Promise<void> {
    const text = value.trim();
    if (!text || this.awaitingHalfDuplexResponse) return;
    await this.connect(config, false);
    this.sendText(text);
  }

  private async connect(config: RealtimeSessionConfig, capture: boolean): Promise<void> {
    if (this.clientSessionId && this.clientSessionId !== config.sessionId) {
      await this.stop();
    }
    if (this.inputContext || this.awaitingHalfDuplexResponse) return;
    this.clientSessionId = config.sessionId;
    if (this.socket?.readyState === WebSocket.OPEN && this.sessionReady) {
      if (capture) {
        await this.startAudio();
        this.callbacks.onState("listening");
      }
      return;
    }
    if (this.socket) return;
    this.callbacks.onState("connecting");
    try {
      if (capture) {
        await this.startAudio();
      } else {
        this.outputContext ??= new AudioContext({ sampleRate: OUTPUT_RATE });
        await this.outputContext.resume();
      }
      const socket = new WebSocket(runtimeRealtimeUrl());
      this.socket = socket;
      socket.onmessage = (message) => {
        try {
          this.handleEvent(JSON.parse(String(message.data)) as Record<string, unknown>, config);
        } catch (error) {
          this.fail(error);
        }
      };
      socket.onerror = () => this.callbacks.onError("Voice connection failed");
      socket.onclose = () => {
        if (!this.stopping) void this.stop(false);
      };
    } catch (error) {
      this.fail(error);
    }
  }

  async toggleCapture(config: RealtimeSessionConfig): Promise<void> {
    if (this.inputContext && !this.fullDuplexEnabled) {
      await this.finishHalfDuplexInput();
    } else if (this.inputContext) {
      await this.stop();
    } else if (!this.awaitingHalfDuplexResponse) {
      await this.start(config);
    }
  }

  sendText(value: string, turnId: string = crypto.randomUUID()): void {
    const text = value.trim();
    if (!text) return;
    this.finishTurn();
    this.currentInputTurnId = turnId;
    if (!this.inputContext) {
      this.halfDuplexPendingSteps += 1;
      this.awaitingHalfDuplexResponse = true;
      this.responseDrainSteps = 0;
      this.currentStepWasListen = false;
      this.callbacks.onState("processing");
    }
    if (this.socket?.readyState !== WebSocket.OPEN || !this.sessionReady) {
      this.pendingText.push({ text, turnId });
      return;
    }
    this.pendingResponseTurns.push(turnId);
    this.socket.send(
      JSON.stringify({
        type: "input.append",
        input: { text, max_new_speak_tokens: SPEAK_TOKENS },
      }),
    );
  }

  async stop(sendClose = true): Promise<void> {
    if (this.stopPromise) return this.stopPromise;
    const pending = this.performStop(sendClose);
    this.stopPromise = pending;
    try {
      await pending;
    } finally {
      if (this.stopPromise === pending) this.stopPromise = null;
    }
  }

  private async performStop(sendClose: boolean): Promise<void> {
    this.stopping = true;
    if (sendClose && this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "session.close", reason: "user_stop" }));
    }
    const socket = this.socket;
    this.socket = null;
    socket?.close();
    await this.stopInputCapture();
    this.finishInputTurn();
    this.stopPlayback();
    await this.outputContext?.close().catch(() => undefined);
    this.outputContext = null;
    this.pending = [];
    this.pendingLength = 0;
    this.outbound = [];
    this.pendingText = [];
    this.pendingResponseTurns = [];
    this.responseTurnIds.clear();
    this.responseTurnOrder = [];
    this.currentInputTurnId = null;
    this.heldHalfDuplexChunk = null;
    this.halfDuplexPendingSteps = 0;
    this.awaitingHalfDuplexResponse = false;
    this.responseDrainSteps = 0;
    this.currentStepWasListen = false;
    this.sessionReady = false;
    this.playAt = 0;
    this.finishTurn();
    this.responseMessageIds.clear();
    this.completedTurns.clear();
    this.completedTurnOrder = [];
    this.lastCompletedByInputTurn.clear();
    this.clientSessionId = null;
    this.resetSpeechDetector();
    this.callbacks.onLevel(0);
    this.callbacks.onState("idle");
    this.stopping = false;
  }

  private async startAudio(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.inputContext = new AudioContext();
    this.outputContext ??= new AudioContext({ sampleRate: OUTPUT_RATE });
    await Promise.all([this.inputContext.resume(), this.outputContext.resume()]);
    await this.inputContext.audioWorklet.addModule("/pcm-capture-worklet.js");
    this.source = this.inputContext.createMediaStreamSource(this.stream);
    this.capture = new AudioWorkletNode(this.inputContext, "pcm-capture");
    const silent = this.inputContext.createGain();
    silent.gain.value = 0;
    this.source.connect(this.capture).connect(silent).connect(this.inputContext.destination);
    this.capture.port.onmessage = (event: MessageEvent<Float32Array>) => this.queueInput(event.data);
  }

  private async stopInputCapture(): Promise<void> {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.source?.disconnect();
    this.capture?.disconnect();
    await this.inputContext?.close().catch(() => undefined);
    this.inputContext = null;
    this.source = null;
    this.capture = null;
  }

  private queueInput(samples: Float32Array): void {
    if (!this.inputContext) return;
    let energy = 0;
    for (const sample of samples) energy += sample * sample;
    const rms = Math.sqrt(energy / Math.max(1, samples.length));
    this.callbacks.onLevel(Math.min(1, rms * 10));
    const converted = resample(samples, this.inputContext.sampleRate, INPUT_RATE);
    this.trackUserInput(converted, rms);
    this.pending.push(converted);
    this.pendingLength += converted.length;
    while (this.pendingLength >= CHUNK_SAMPLES) {
      const chunk = this.takeChunk();
      if (this.fullDuplexEnabled) {
        this.sendInput(chunk);
      } else {
        if (this.heldHalfDuplexChunk) {
          this.sendInput(this.heldHalfDuplexChunk, { forceListen: true });
          this.halfDuplexPendingSteps += 1;
        }
        this.heldHalfDuplexChunk = chunk;
      }
    }
  }

  private beginInputTurn(chunks: Float32Array[] = []): void {
    if (this.inputTurnId || !this.clientSessionId) return;
    this.finishTurn();
    this.inputTurnId = crypto.randomUUID();
    this.currentInputTurnId = this.inputTurnId;
    this.inputTurnChunks = chunks.map((chunk) => new Float32Array(chunk));
    this.inputTurnSamples = this.inputTurnChunks.reduce(
      (total, chunk) => total + chunk.length,
      0,
    );
    this.callbacks.onInputStart({
      id: this.inputTurnId,
      sessionId: this.clientSessionId,
    });
  }

  private appendInputTurn(samples: Float32Array): void {
    if (!this.inputTurnId) this.beginInputTurn();
    if (!this.inputTurnId) return;
    this.inputTurnChunks.push(new Float32Array(samples));
    this.inputTurnSamples += samples.length;
  }

  private finishInputTurn(): void {
    const id = this.inputTurnId;
    const sessionId = this.clientSessionId;
    if (!id || !sessionId) return;
    const audio =
      this.inputTurnSamples >= MIN_USER_TURN_SAMPLES
        ? wavBlob(this.inputTurnChunks, INPUT_RATE)
        : null;
    this.inputTurnId = null;
    this.inputTurnChunks = [];
    this.inputTurnSamples = 0;
    this.callbacks.onInputEnd({ id, sessionId, audio });
  }

  private trackUserInput(samples: Float32Array, rms: number): void {
    if (!this.fullDuplexEnabled) {
      this.appendInputTurn(samples);
      return;
    }
    if (this.inputTurnId) {
      this.appendInputTurn(samples);
      this.speechSilenceSamples =
        rms < SPEECH_RMS_THRESHOLD ? this.speechSilenceSamples + samples.length : 0;
      if (this.speechSilenceSamples >= SPEECH_END_SAMPLES) {
        this.finishInputTurn();
        this.speechSilenceSamples = 0;
        this.speechAboveSamples = 0;
      }
      return;
    }
    this.speechPreroll.push(new Float32Array(samples));
    this.speechPrerollSamples += samples.length;
    while (
      this.speechPreroll.length > 1 &&
      this.speechPrerollSamples > SPEECH_PREROLL_SAMPLES
    ) {
      this.speechPrerollSamples -= this.speechPreroll.shift()!.length;
    }
    this.speechAboveSamples =
      rms >= SPEECH_RMS_THRESHOLD ? this.speechAboveSamples + samples.length : 0;
    if (this.speechAboveSamples >= SPEECH_START_SAMPLES) {
      const preroll = this.speechPreroll;
      this.speechPreroll = [];
      this.speechPrerollSamples = 0;
      this.speechAboveSamples = 0;
      this.speechSilenceSamples = 0;
      this.beginInputTurn(preroll);
    }
  }

  private resetSpeechDetector(): void {
    this.speechPreroll = [];
    this.speechPrerollSamples = 0;
    this.speechAboveSamples = 0;
    this.speechSilenceSamples = 0;
  }

  private takeChunk(): Float32Array {
    const chunk = new Float32Array(CHUNK_SAMPLES);
    let offset = 0;
    while (offset < CHUNK_SAMPLES) {
      const head = this.pending[0];
      const count = Math.min(head.length, CHUNK_SAMPLES - offset);
      chunk.set(head.subarray(0, count), offset);
      offset += count;
      if (count === head.length) this.pending.shift();
      else this.pending[0] = head.subarray(count);
      this.pendingLength -= count;
    }
    return chunk;
  }

  private drainPending(): Float32Array | null {
    if (!this.pendingLength) return null;
    const chunk = new Float32Array(CHUNK_SAMPLES);
    let offset = 0;
    while (this.pending.length) {
      const head = this.pending.shift()!;
      const count = Math.min(head.length, CHUNK_SAMPLES - offset);
      chunk.set(head.subarray(0, count), offset);
      offset += count;
      if (count < head.length) break;
    }
    this.pending = [];
    this.pendingLength = 0;
    return chunk;
  }

  private async finishHalfDuplexInput(): Promise<void> {
    if (!this.inputContext || this.fullDuplexEnabled) return;
    await this.stopInputCapture();
    this.finishInputTurn();
    const tail = this.drainPending();
    if (tail && this.heldHalfDuplexChunk) {
      this.sendInput(this.heldHalfDuplexChunk, { forceListen: true });
      this.halfDuplexPendingSteps += 1;
      this.heldHalfDuplexChunk = null;
    }
    const finalChunk = tail || this.heldHalfDuplexChunk;
    this.heldHalfDuplexChunk = null;
    if (!finalChunk) {
      this.callbacks.onState("idle");
      return;
    }
    this.sendInput(finalChunk, { forceSpeak: true });
    this.halfDuplexPendingSteps += 1;
    this.awaitingHalfDuplexResponse = true;
    this.responseDrainSteps = 0;
    this.currentStepWasListen = false;
    this.callbacks.onState("processing");
  }

  private sendInput(
    samples: Float32Array,
    options: { forceListen?: boolean; forceSpeak?: boolean } = {},
    turnId = this.currentInputTurnId,
  ): void {
    if (this.socket?.readyState !== WebSocket.OPEN || !this.sessionReady) {
      this.outbound.push({ samples, ...options, turnId });
      if (this.outbound.length > 64) this.outbound.shift();
      return;
    }
    const input: Record<string, unknown> = {
      audio: float32ToBase64(samples),
      max_new_speak_tokens: SPEAK_TOKENS,
    };
    if (options.forceListen) input.force_listen = true;
    if (options.forceSpeak) input.force_speak = true;
    this.pendingResponseTurns.push(turnId);
    this.socket.send(JSON.stringify({ type: "input.append", input }));
  }

  private handleEvent(event: Record<string, unknown>, config: RealtimeSessionConfig): void {
    if (event.type === "session.queue_done") {
      this.socket?.send(
        JSON.stringify({
          type: "session.init",
          payload: {
            system_prompt: config.systemPrompt,
            config: {
              temperature: config.temperature,
              top_p: config.topP,
              top_k: config.topK,
              text_repetition_penalty: config.repetitionPenalty,
            },
          },
        }),
      );
    } else if (event.type === "session.created") {
      this.sessionReady = true;
      this.callbacks.onState(this.awaitingHalfDuplexResponse ? "processing" : "listening");
      for (const item of this.outbound.splice(0)) {
        this.sendInput(item.samples, item, item.turnId);
      }
      for (const item of this.pendingText.splice(0)) this.sendText(item.text, item.turnId);
    } else if (event.kind === "text") {
      const target = this.bufferForResponse(event);
      if (!target) return;
      const text = String(event.text ?? "");
      target.buffer.text += text;
      if (target.late) {
        this.publishTurn(target.buffer);
      } else {
        this.callbacks.onText(target.buffer.sessionId, target.buffer.text);
      }
    } else if (event.kind === "audio") {
      const target = this.bufferForResponse(event);
      if (!target) return;
      const samples = base64ToFloat32(String(event.audio ?? ""));
      const sampleRate = Number(event.sample_rate) || OUTPUT_RATE;
      target.buffer.audio.push(samples);
      target.buffer.audioRate = sampleRate;
      if (target.late) {
        this.publishTurn(target.buffer);
      } else {
        this.playAudio(samples, sampleRate);
      }
    } else if (event.kind === "listen") {
      const target = this.bufferForResponse(event);
      if (!target || target.late) return;
      this.currentStepWasListen = true;
      this.finishTurn();
    } else if (event.type === "response.step.done") {
      const target = this.bufferForResponse(event);
      if (this.pendingResponseTurns.length) this.pendingResponseTurns.shift();
      if (!target || target.late) {
        this.currentStepWasListen = false;
        return;
      }
      if (this.halfDuplexPendingSteps > 0) this.halfDuplexPendingSteps -= 1;
      if (this.awaitingHalfDuplexResponse && this.halfDuplexPendingSteps === 0) {
        const ended = event.end_of_turn === true || this.currentStepWasListen;
        if (!ended && this.responseDrainSteps < MAX_RESPONSE_DRAIN_STEPS) {
          this.currentStepWasListen = false;
          this.responseDrainSteps += 1;
          this.halfDuplexPendingSteps = 1;
          this.sendInput(new Float32Array(CHUNK_SAMPLES));
        } else {
          this.awaitingHalfDuplexResponse = false;
          this.responseDrainSteps = 0;
          this.finishTurn();
          this.callbacks.onState("idle");
        }
      } else if (!this.awaitingHalfDuplexResponse && event.end_of_turn === true) {
        this.finishTurn();
      }
      this.currentStepWasListen = false;
    } else if (event.type === "error") {
      const detail = event.error as { message?: string } | undefined;
      throw new Error(detail?.message || "Voice connection failed");
    }
  }

  private responseId(event: Record<string, unknown>): string {
    return typeof event.response_id === "string" ? event.response_id : "";
  }

  private responseTurnId(event: Record<string, unknown>): string | null {
    const responseId = this.responseId(event);
    if (responseId && this.responseTurnIds.has(responseId)) {
      return this.responseTurnIds.get(responseId) ?? null;
    }
    const turnId = this.pendingResponseTurns[0] ?? null;
    if (responseId) {
      this.responseTurnIds.set(responseId, turnId);
      this.responseTurnOrder.push(responseId);
      if (this.responseTurnOrder.length > 512) {
        const expired = this.responseTurnOrder.shift();
        if (expired) {
          this.responseTurnIds.delete(expired);
          this.responseMessageIds.delete(expired);
        }
      }
    }
    return turnId;
  }

  private createTurn(inputTurnId: string | null): BufferedVoiceTurn | null {
    if (!this.clientSessionId) return null;
    return {
      id: crypto.randomUUID(),
      sessionId: this.clientSessionId,
      inputTurnId,
      text: "",
      audio: [],
      audioRate: OUTPUT_RATE,
    };
  }

  private rememberCompletedTurn(buffer: BufferedVoiceTurn): void {
    if (!this.completedTurns.has(buffer.id)) this.completedTurnOrder.push(buffer.id);
    this.completedTurns.set(buffer.id, buffer);
    this.lastCompletedByInputTurn.set(buffer.inputTurnId, buffer.id);
    while (this.completedTurnOrder.length > 64) {
      const expired = this.completedTurnOrder.shift();
      if (expired) this.completedTurns.delete(expired);
    }
  }

  private bufferForResponse(
    event: Record<string, unknown>,
  ): { buffer: BufferedVoiceTurn; late: boolean } | null {
    const responseId = this.responseId(event);
    const knownMessageId = responseId ? this.responseMessageIds.get(responseId) : undefined;
    if (knownMessageId) {
      if (this.activeTurn?.id === knownMessageId) {
        return { buffer: this.activeTurn, late: false };
      }
      const completed = this.completedTurns.get(knownMessageId);
      if (completed) return { buffer: completed, late: true };
    }

    const inputTurnId = this.responseTurnId(event);
    const stale =
      this.currentInputTurnId !== null && inputTurnId !== this.currentInputTurnId;
    let buffer: BufferedVoiceTurn | null = null;
    if (stale) {
      const completedId = this.lastCompletedByInputTurn.get(inputTurnId);
      buffer = completedId ? this.completedTurns.get(completedId) ?? null : null;
      if (!buffer) {
        buffer = this.createTurn(inputTurnId);
        if (buffer) this.rememberCompletedTurn(buffer);
      }
    } else {
      if (this.activeTurn && this.activeTurn.inputTurnId !== inputTurnId) {
        this.finishTurn();
      }
      this.activeTurn ??= this.createTurn(inputTurnId);
      buffer = this.activeTurn;
    }
    if (buffer && responseId) this.responseMessageIds.set(responseId, buffer.id);
    return buffer ? { buffer, late: stale } : null;
  }

  private publishTurn(buffer: BufferedVoiceTurn): void {
    if (!buffer.text.trim() && !buffer.audio.length) return;
    const audio = buffer.audio.length ? wavBlob(buffer.audio, buffer.audioRate) : null;
    this.callbacks.onTurn({
      id: buffer.id,
      sessionId: buffer.sessionId,
      text: buffer.text,
      audio,
    });
  }

  private finishTurn(): void {
    const buffer = this.activeTurn;
    if (!buffer) return;
    this.publishTurn(buffer);
    this.rememberCompletedTurn(buffer);
    this.activeTurn = null;
    this.callbacks.onText(buffer.sessionId, "");
  }

  private playAudio(samples: Float32Array, sampleRate: number): void {
    if (!this.playbackEnabled || !samples.length || !this.outputContext) return;
    const context = this.outputContext;
    const buffer = context.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(new Float32Array(samples), 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    this.playAt = Math.max(this.playAt, context.currentTime + PLAYBACK_DELAY_SECONDS);
    source.start(this.playAt);
    this.playAt += buffer.duration;
    this.playing.add(source);
    source.onended = () => this.playing.delete(source);
  }

  private stopPlayback(): void {
    for (const source of this.playing) {
      try {
        source.stop();
      } catch {
        // The source may already have ended.
      }
    }
    this.playing.clear();
    if (this.outputContext) this.playAt = this.outputContext.currentTime;
  }

  private fail(error: unknown): void {
    const message = error instanceof Error ? error.message : "Voice connection failed";
    this.callbacks.onError(message);
    this.callbacks.onState("error");
    void this.stop(false);
  }
}
