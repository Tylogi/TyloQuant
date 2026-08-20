import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "MFQStudio" / "desktop"
TAURI = STUDIO / "src-tauri"
RUST = (TAURI / "src" / "main.rs").read_text(encoding="utf-8")
BUILD = (TAURI / "build.rs").read_text(encoding="utf-8")
APP = (ROOT / "MFQStudio" / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
MARKDOWN = (ROOT / "MFQStudio" / "web" / "src" / "Markdown.tsx").read_text(encoding="utf-8")
MARKDOWN_TEXT = (ROOT / "MFQStudio" / "web" / "src" / "markdownText.ts").read_text(
    encoding="utf-8"
)
STYLES = (ROOT / "MFQStudio" / "web" / "src" / "styles.css").read_text(encoding="utf-8")
REALTIME_AUDIO = (ROOT / "MFQStudio" / "web" / "src" / "realtimeAudio.ts").read_text(
    encoding="utf-8"
)


def test_studio_embeds_the_shared_web_client():
    config = json.loads((TAURI / "tauri.conf.json").read_text(encoding="utf-8"))
    assert config["build"]["frontendDist"] == "../../web/dist"
    assert config["build"]["beforeBuildCommand"] == "npm --prefix ../web run build"
    assert config["identifier"] == "com.tylogi.mfq-studio"
    assert "icons/icon.ico" in config["bundle"]["icon"]
    assert "icons/icon.icns" in config["bundle"]["icon"]
    assert "IconDir::new" in BUILD
    assert "IconFamily::new" in BUILD


def test_assistant_markdown_recovers_fully_escaped_structural_line_breaks():
    assert "normalizeEscapedMarkdownLineBreaks" in MARKDOWN
    assert "normalizeEscapedLineBreaks={normalizeEscapedLineBreaks}" in APP
    assert 'message.role === "assistant"' in APP
    assert "isJsonDocument" in MARKDOWN_TEXT
    assert 'text.includes("```")' in MARKDOWN_TEXT
    assert "hasEscapedMarkdownStructure" in MARKDOWN_TEXT


def test_studio_starts_the_unified_local_server_and_bundled_runtime():
    assert 'Some("mfq-server")' in RUST
    assert "Command::new" in RUST
    assert "studio_start_local" in RUST
    assert '.arg("serve")' in RUST
    assert 'command.arg("--running-executable")' in RUST
    assert "MFQ_MLX_METALLIB" in RUST
    assert "MFQ_AVFOUNDATION_VIDEO_LIBRARY" in RUST


def test_studio_supports_local_and_remote_server_connections_with_voice_controls():
    assert "RuntimeMode::Local" in RUST
    assert "RuntimeMode::Remote" in RUST
    assert "studio_configure" in RUST
    assert "studio_start_local" in RUST
    assert "studio_select_model_file" in RUST
    assert "selectLocalModelFile" in APP
    assert "const canChooseLocalModel = isStudio()" in APP
    assert 'tr("选择本地模型文件", "Choose a local model file")' in APP
    assert 'tr("加载本地模型或连接模型服务后即可开始。", "Load a local model or connect to a model server to get started.")' in APP
    assert 'tr("选择本地模型", "Choose local model")' in APP
    assert "访达" not in APP
    assert "Finder" not in APP
    assert 'className="open-local-model"' in APP
    assert APP.count("chooseLocalModelFile()") >= 4
    assert "Object.keys(MODE_LABELS)" not in APP
    assert "RealtimeAudioController" in APP
    assert '(["text", "voice", "full_duplex"] as SessionMode[])' in APP
    assert "selectInteractionMode" in APP


def test_studio_drains_duplex_output_after_microphone_capture_stops():
    assert "MAX_RESPONSE_DRAIN_STEPS" in REALTIME_AUDIO
    assert "event.end_of_turn === true" in REALTIME_AUDIO
    assert "this.sendInput(new Float32Array(CHUNK_SAMPLES))" in REALTIME_AUDIO
    assert "this.callbacks.onText(target.buffer.sessionId, target.buffer.text)" in REALTIME_AUDIO


def test_studio_preserves_resampling_phase_across_audio_worklet_blocks():
    assert "class StreamingLinearResampler" in REALTIME_AUDIO
    assert "private position = 0" in REALTIME_AUDIO
    assert "this.position += this.step" in REALTIME_AUDIO
    assert "new AudioContext({ sampleRate: INPUT_RATE })" in REALTIME_AUDIO
    assert "Math.round((input.length * targetRate) / sourceRate)" not in REALTIME_AUDIO


def test_studio_uses_the_model_bound_duplex_system_prompt():
    assert "modeTemplateSettings" in APP
    assert 'runtime?.duplex_sampling_defaults ?? realtime?.defaults' in APP
    assert "system_prompt: config.systemPrompt" in REALTIME_AUDIO
    assert "text_repetition_penalty: config.repetitionPenalty" in REALTIME_AUDIO
    assert "submitText(text, realtimeSessionConfig(active.id))" in APP


def test_studio_resolves_model_global_and_role_inference_settings_in_order():
    assert "inheritModelDefaults: true" in APP
    assert "function roleGenerationSettings" in APP
    assert "if (role.inheritGlobalSettings)" in APP
    assert "const resolvedGlobalSettings = useMemo" in APP
    assert "const effectiveSettings = useMemo" in APP
    assert "roleGenerationSettings(resolvedGlobalSettings, activeRolePreset)" in APP
    assert "sampling: samplingParams()" in APP
    assert "max_tokens: effectiveSettings.maxTokens" in APP
    assert "system_prompt: effectiveSettings.systemPrompt || null" in APP
    assert "setSettings((current) => ({ ...current, ...rolePreset.settings" not in APP


def test_studio_defaults_global_and_role_editors_to_inherited_parameters():
    assert 'className="settings-inherited-fields" disabled={settingsDraft.inheritModelDefaults}' in APP
    assert 'checked={settingsDraft.inheritModelDefaults}' in APP
    assert 'inheritGlobalSettings: preset?.inheritGlobalSettings ?? true' not in APP
    assert "const inheritGlobalSettings = preset?.inheritGlobalSettings ?? true" in APP
    assert 'className="role-inherited-fields" disabled={roleEditor.inheritGlobalSettings}' in APP
    assert 'checked={roleEditor.inheritGlobalSettings}' in APP
    assert "inherit_global_settings: preset.inheritGlobalSettings" in APP
    assert 'typeof raw.inheritGlobalSettings === "boolean" ? raw.inheritGlobalSettings : true' in APP
    assert 'typeof preset.metadata?.inherit_global_settings === "boolean"' in APP
    assert ".settings-inherited-fields:disabled section" in STYLES
    assert ".role-inherited-fields:disabled" in STYLES


def test_realtime_turns_remain_bound_to_the_session_that_created_them():
    assert "sessionId: string;" in REALTIME_AUDIO
    assert "private clientSessionId: string | null = null" in REALTIME_AUDIO
    assert "sessionId: buffer.sessionId" in REALTIME_AUDIO
    assert "onTurn: ({ id, sessionId, text, audio })" in APP
    assert "activeIdRef" not in APP


def test_full_duplex_user_speech_visually_splits_assistant_turns():
    assert "SPEECH_RMS_THRESHOLD" in REALTIME_AUDIO
    assert "this.finishTurn();\n    this.inputTurnId = crypto.randomUUID()" in REALTIME_AUDIO
    assert "this.callbacks.onInputStart" in REALTIME_AUDIO
    assert "this.callbacks.onInputEnd" in REALTIME_AUDIO
    assert 'role: "user"' in APP
    assert "message.pending" in APP


def test_full_duplex_routes_pre_interrupt_response_tails_back_to_the_old_turn():
    assert "private pendingResponseTurns" in REALTIME_AUDIO
    assert "private responseTurnIds" in REALTIME_AUDIO
    assert "private currentInputTurnId" in REALTIME_AUDIO
    assert "private responseMessageIds" in REALTIME_AUDIO
    assert "private completedTurns" in REALTIME_AUDIO
    assert "this.lastCompletedByInputTurn.get(inputTurnId)" in REALTIME_AUDIO
    assert "this.publishTurn(target.buffer)" in REALTIME_AUDIO
    assert "message.id === id ? { ...message, text }" in APP


def test_closing_a_full_duplex_microphone_stops_instead_of_forcing_speech():
    assert "finishFullDuplexInput" not in REALTIME_AUDIO
    assert "} else if (this.inputContext) {\n      await this.stop();" in REALTIME_AUDIO
    assert "this.stopPlayback();" in REALTIME_AUDIO


def test_dashboard_and_lab_use_subpages_and_reorderable_collapsible_panels():
    assert 'type DashboardPage = "overview" | "cache" | "models" | "connections"' in APP
    assert 'type LabPage = "models" | "evaluations" | "quantization"' in APP
    assert "function PanelDeck" in APP
    assert "PANEL_LAYOUT_KEY" in APP
    assert "PANEL_COLLAPSED_KEY" in APP
    assert "function panelKey" in APP
    assert "const minimumX = -baseLeft" in APP
    assert "const minimumY = -baseTop" in APP
    assert "onDoubleClick={() => bringPanelToFront(id)}" in APP
    assert "getBoundingClientRect()" in APP
    assert "interface PanelOverlap" in APP
    assert "panel-overlap-boundary" in APP
    assert "left: left - deckRect.left" in APP
    assert "const covering = firstZ > secondZ ? first : second" in APP
    assert 'overlap.drawTop ? " edge-top"' in APP
    assert ".panel-overlap-boundary.edge-top.edge-left" in STYLES
    assert "border-top-left-radius: 11px" in STYLES
    assert ".panel-drag { right: 38px;" in STYLES
    assert "place-content: center" in STYLES
    assert "function panelResizeEdges" in APP
    assert "function beginPanelResize" in APP
    assert "width: placement?.width" in APP
    assert ".panel-item-clipped" in STYLES
    assert "@container (max-width: 620px)" in STYLES
    assert 'aria-label={tr("重置当前布局", "Reset current layout")}' in APP
    assert "delete updated[page]" in APP
    assert "setDashboardLayoutReset((current) => current + 1)" in APP
    assert "setLabLayoutReset((current) => current + 1)" in APP
    assert "overlapResettingRef.current = true" in APP
    assert "if (overlapResettingRef.current)" in APP
    assert "}, 180);" in APP
    assert 'aria-label={tr("刷新状态", "Refresh status")}' not in APP
    assert 'dashboardPage === "overview"' in APP
    assert 'page="lab-quantization"' in APP
    assert 'tr("量化工作台", "Quantization workspace")' in APP
    assert "Run, track, and reproduce MFQ workloads" not in APP
    assert "Runtime health and request performance" not in APP
    assert "<p>Dashboard</p>" not in APP
    assert "<p>Lab</p>" not in APP
    assert 'page="lab-imatrix"' not in APP
    assert 'page="lab-jobs"' not in APP
