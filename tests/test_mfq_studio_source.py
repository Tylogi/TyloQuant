import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "MFQStudio"
TAURI = STUDIO / "src-tauri"
RUST = (TAURI / "src" / "main.rs").read_text(encoding="utf-8")
BUILD = (TAURI / "build.rs").read_text(encoding="utf-8")
APP = (STUDIO / "src" / "App.tsx").read_text(encoding="utf-8")
API = (STUDIO / "src" / "api.ts").read_text(encoding="utf-8")
MAIN = (STUDIO / "src" / "main.tsx").read_text(encoding="utf-8")
MARKDOWN = (STUDIO / "src" / "Markdown.tsx").read_text(encoding="utf-8")
MARKDOWN_TEXT = (STUDIO / "src" / "markdownText.ts").read_text(encoding="utf-8")
STUDIO_BRIDGE = (STUDIO / "src" / "studio.ts").read_text(encoding="utf-8")
STYLES = (STUDIO / "src" / "styles.css").read_text(encoding="utf-8")
REALTIME_AUDIO = (STUDIO / "src" / "realtimeAudio.ts").read_text(encoding="utf-8")
RELEASE_SCRIPT = (ROOT / "release" / "build_release_mac.sh").read_text(encoding="utf-8")


def test_studio_uses_one_package_for_web_and_desktop_clients():
    config = json.loads((TAURI / "tauri.conf.json").read_text(encoding="utf-8"))
    release_config = json.loads(
        (TAURI / "tauri.release-macos.conf.json").read_text(encoding="utf-8")
    )
    package = json.loads((STUDIO / "package.json").read_text(encoding="utf-8"))
    assert not (STUDIO / "web").exists()
    assert not (STUDIO / "desktop").exists()
    assert package["name"] == "@mfq/studio"
    assert package["scripts"]["tauri"] == "tauri"
    assert config["build"]["frontendDist"] == "../dist"
    assert config["build"]["beforeBuildCommand"] == "npm run build"
    assert config["identifier"] == "com.tylogi.mfq-studio"
    assert "icons/icon.ico" in config["bundle"]["icon"]
    assert "icons/icon.icns" in config["bundle"]["icon"]
    assert "IconDir::new" in BUILD
    assert "IconFamily::new" in BUILD
    assert "media-src 'self' asset: data: blob:" in config["app"]["security"]["csp"]
    assert release_config["bundle"]["macOS"]["hardenedRuntime"] is False
    assert 'if [[ "${mfq_signing_identity}" != "-" ]]' in RELEASE_SCRIPT
    assert '"hardenedRuntime":true' in RELEASE_SCRIPT
    assert "--remap-path-prefix=${HOME}=/mfq-build/home" in RELEASE_SCRIPT
    assert "--remap-path-prefix=${mfq_project_dir}=/mfq-src" in RELEASE_SCRIPT
    assert 'RUSTFLAGS="${mfq_release_rustflags}"' in RELEASE_SCRIPT
    assert "packaged Studio contains a private build path" in RELEASE_SCRIPT
    assert 'mfq-decode-metal" --self-test-metal' in RELEASE_SCRIPT
    assert 'ln -s "mlx/lib/mlx.metallib"' in RELEASE_SCRIPT
    assert "_mlx-runtime-check" in RELEASE_SCRIPT


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
    assert "studio_select_model_directory" in RUST
    assert "selectLocalModelDirectory" in APP
    assert "/api/v1/models/directories/register" in RUST
    assert ".mfq-files.json" not in RUST
    assert "canUseNativeModelPicker" in APP
    assert 'tr("添加模型", "Add model")' in APP
    assert 'tr("浏览 MFQ Server 所在设备上的文件夹。", "Browse folders on the MFQ Server host.")' in APP
    assert 'tr("选择模型文件夹", "Choose model folder")' in APP
    assert "访达" not in APP
    assert "Finder" not in APP
    assert 'className="open-model-primary"' in APP
    assert APP.count("chooseModelDirectory()") >= 3
    assert "registerCurrentModelDirectory" in APP
    assert "modelDirectoryPath" in APP
    assert "jumpToModelDirectory" in APP
    assert "listing.current_path" in APP
    assert 'tr("前往", "Go")' in APP
    assert "Object.keys(MODE_LABELS)" not in APP
    assert "RealtimeAudioController" in APP
    assert '(["text", "voice", "full_duplex"] as SessionMode[])' in APP
    assert "selectInteractionMode" in APP


def test_voice_component_prompt_requires_an_explicit_full_duplex_selection():
    assert 'const interactionMode = active?.mode ?? mode;' in APP
    assert 'interactionMode === "full_duplex"' in APP
    assert "const needsVoiceOutputComponent =" in APP
    assert "{needsVoiceOutputComponent && voiceComponent &&" in APP
    assert "capabilities?.model_capabilities.features.audio_output && !realtimeAvailable" not in APP
    assert "capabilities.model_capabilities.features.full_duplex) && <select" in APP


def test_studio_handles_a_running_server_without_a_loaded_model():
    assert 'useState("")' in APP
    assert 'tr("尚未加载模型", "No model loaded")' in APP
    assert "if (!selectedModel) return;" in APP
    assert 'runtime?.model || "Empty"' in APP
    assert 'runtime?.model || "MFQ Server"' not in APP
    assert 'statusResult.status === "fulfilled" ? statusResult.value : null' in APP
    assert "setRuntime(status)" in APP
    assert "Promise.allSettled([" in APP


def test_studio_can_select_and_load_an_external_mfq_directory_in_local_mode():
    assert "rfd::AsyncFileDialog::new()" in RUST
    assert ".pick_folder()" in RUST
    assert "studio_select_model_directory" in RUST
    assert "/api/v1/models/directories/register" in RUST
    assert 'tauri.invoke<string[] | null>("studio_select_model_directory")' in STUDIO_BRIDGE
    assert "selectLocalModelDirectory" in APP
    assert "api.modelArtifacts(true)" in APP
    assert "api.loadModel(artifact.name, contextSize, 2048, {" in APP
    assert "canUseNativeModelPicker" in APP
    assert 'tr("选择模型文件夹", "Choose model folder")' in APP


def test_studio_uses_native_confirmation_dialogs_for_destructive_actions():
    assert "fn studio_confirm(message: String) -> bool" in RUST
    assert "rfd::MessageButtons::YesNo" in RUST
    assert "studio_confirm," in RUST
    assert 'tauri.invoke<boolean>("studio_confirm", { message })' in STUDIO_BRIDGE
    assert "window.confirm" not in APP
    assert APP.count("await studioConfirm(") >= 6


def test_studio_has_a_render_error_boundary_instead_of_a_blank_window():
    assert "class AppErrorBoundary" in MAIN
    assert "static getDerivedStateFromError" in MAIN
    assert '<main className="fatal-error" role="alert">' in MAIN
    assert "<AppErrorBoundary>" in MAIN


def test_studio_loads_message_media_through_authenticated_blob_urls():
    assert "async fetchMedia(id: string, signal?: AbortSignal): Promise<Blob>" in API
    assert "headers: authorizedHeaders()" in API
    assert "api.fetchMedia(part.media.id, controller.signal)" in APP
    assert "URL.createObjectURL(blob)" in APP
    assert "URL.revokeObjectURL(objectUrl)" in APP
    assert 'alt="Attached image"' in APP
    assert "const src = api.mediaUrl(part.media.id)" not in APP


def test_studio_renders_video_first_frame_posters():
    assert "function VideoWithFirstFrame" in APP
    assert 'video.onloadeddata = () =>' in APP
    assert 'drawImage(video, 0, 0, canvas.width, canvas.height)' in APP
    assert 'canvas.toBlob((blob) =>' in APP
    assert 'poster={poster ?? undefined}' in APP
    assert '<VideoWithFirstFrame className="message-media media-video" controls src={src} />' in APP
    assert '<VideoWithFirstFrame muted src={attachment.previewUrl} />' in APP


def test_studio_validates_video_metadata_before_uploading_media():
    assert "const timeout = window.setTimeout(" in APP
    assert "video.load();" in APP
    metadata = APP.index("const metadata = await mediaMetadata(attachment.file, attachment.kind);")
    upload = APP.index("const resource = await api.uploadMedia(attachment.file);", metadata)
    assert metadata < upload


def test_studio_fetches_protected_message_media_and_documents():
    assert "const [loadFailed, setLoadFailed] = useState(false);" in APP
    assert "Unable to load attachment" in APP
    document = APP.index("async function downloadDocument()")
    assert "await api.fetchMedia(part.media.id)" in APP[document:]
    assert "anchor.download = part.name" in APP[document:]
    assert "document.body.appendChild(anchor)" in APP[document:]
    assert "anchor.remove()" in APP[document:]
    assert "href={api.mediaUrl(part.media.id)}" not in APP


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


def test_studio_resolves_model_and_global_inference_settings_without_roles():
    assert "inheritModelDefaults: true" in APP
    assert "const resolvedGlobalSettings = useMemo" in APP
    assert "const effectiveSettings = resolvedGlobalSettings" in APP
    assert "function roleGenerationSettings" not in APP
    assert "roleEditor" not in APP
    assert "assistant_id" not in APP
    assert "api.createSession(selectedModel, mode)" in APP
    assert "sampling: samplingParams()" in APP
    assert "max_tokens: effectiveSettings.maxTokens" in APP
    assert "const effectiveSystemPrompt = useMemo" in APP
    assert "system_prompt: effectiveSystemPrompt" in APP
    assert "systemPrompt: effectiveSystemPrompt" in APP
    assert "Never insert Chinese words into an English answer" in APP
    assert "setSettings((current) => ({ ...current, ...rolePreset.settings" not in APP


def test_studio_defaults_global_settings_to_inherited_model_parameters():
    assert 'className="settings-page-inherited" disabled={settingsDraft.inheritModelDefaults}' in APP
    assert 'checked={settingsDraft.inheritModelDefaults}' in APP
    assert "inherit_global_settings: preset.inheritGlobalSettings" in APP
    assert 'typeof raw.inheritGlobalSettings === "boolean" ? raw.inheritGlobalSettings : true' in APP
    assert 'typeof preset.metadata?.inherit_global_settings === "boolean"' in APP
    assert ".settings-page-inherited:disabled" in STYLES
    assert 'className="role-inherited-fields"' not in APP


def test_studio_exposes_theme_selection_without_using_sidebar_status_space():
    assert 'settingsDraft.theme' in APP
    assert '<option value="system">{tr("跟随系统", "System")}</option>' in APP
    assert '<option value="light">{tr("浅色", "Light")}</option>' in APP
    assert '<option value="dark">{tr("深色", "Dark")}</option>' in APP
    assert 'className="theme-switcher"' not in APP
    assert "connection-card" not in APP
    assert ".connection-card" not in STYLES


def test_studio_exposes_omlx_style_runtime_lifecycle_controls():
    assert 'className="runtime-hero"' in APP
    assert 'tr("推理", "Inference")' in APP
    assert "MFQ Runtime" in APP
    assert 'tr("固定到内存", "Pin in memory")' in APP
    assert 'tr("空闲卸载", "Idle unload")' in APP
    assert "const [loadPinned, setLoadPinned] = useState(false)" in APP
    assert "const [loadIdleTtl, setLoadIdleTtl] = useState<number | null>(null)" in APP
    assert "pin: loadPinned" in APP
    assert "idle_ttl_seconds: loadIdleTtl" in APP
    assert "idle_ttl_seconds?: number | null" in API
    assert "pin?: boolean" in API
    assert ".runtime-hero {" in STYLES
    assert "--accent: #0a84ff" in STYLES


def test_studio_runtime_monogram_tracks_the_real_model_lifecycle():
    assert "function ModelMonogram" in APP
    assert 'const runtimeModelName = runtime?.model || "Empty";' in APP
    assert 'job.kind === "model.load"' in APP
    assert 'instance.state === "loading"' in APP
    assert 'instance.state === "failed"' in APP
    assert '<ModelMonogram name={modelHero.name} state={modelHero.state} />' in APP
    assert 'className={`runtime-status-pill ${modelHero.state}`}' in APP
    assert ".model-monogram.loading { color: var(--warning); }" in STYLES
    assert ".model-monogram.ready { color: var(--success); }" in STYLES
    assert ".model-monogram.failed { color: var(--danger); }" in STYLES


def test_studio_adapts_prefix_cache_panel_to_flash_next_hot_cache():
    assert 'prefix_cache_mode?: string;' in API
    assert 'prefix_cache_pending_bytes?: number;' in API
    assert 'prefix_cache_pending_max_bytes?: number;' in API
    assert 'runtime?.prefix_cache_mode === "single_device_hot_prefix"' in APP
    assert "const prefixCachePersistent" in APP
    assert "const prefixCacheSupported = prefixCachePersistent || prefixCacheHotOnly" in APP
    assert 'tr("设备热前缀", "Device-hot prefix")' in APP
    assert 'tr("进程生命周期", "Process lifetime")' in APP
    assert "device-hot prefix? Chat history will be kept." in APP


def test_studio_uses_theme_aware_model_actions_and_readable_errors():
    assert ".panel-heading-actions button {" in STYLES
    assert "border: 1px solid var(--accent-border)" in STYLES
    assert ".mcp-form button { min-width: 64px;" in STYLES
    assert ".job-actions .secondary { border: 1px solid var(--panel-line);" in STYLES
    assert ".runtime-log p { min-width: 0; overflow-wrap: anywhere;" in STYLES
    assert ".error-banner span { min-width: 0; overflow-wrap: anywhere;" in STYLES


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


def test_dashboard_uses_hivellm_style_static_backend_console_components():
    assert 'type DashboardPage = "overview" | "models" | "connections" | "cache" | "logs" | "settings"' in APP
    assert 'type LabPage = "models" | "evaluations" | "quantization"' in APP
    assert "function ScreenHeader" in APP
    assert "function SectionLabel" in APP
    assert "function TMPanel" in APP
    assert "function MetricTile" in APP
    assert "function SettingRow" in APP
    assert "function UsageBar" in APP
    assert "function EmptyPanel" in APP
    assert '<details className="sidebar-more"' not in APP
    assert 'className="overview-memory-panel"' in APP
    assert 'className="overview-footer-grid"' in APP
    assert 'label={tr("解码", "Decode")}' in APP and 'icon="waveform"' in APP
    assert 'label={tr("首字延迟", "TTFT")}' in APP and 'icon="clock"' in APP
    assert 'dashboardPage === "settings" && settingsPage' in APP
    assert 'className="settings-panel"' not in APP
    assert 'className="drawer-scrim"' not in APP
    assert '<Icon name="gauge" />{tr("概览", "Overview")}' in APP
    assert '<Icon name="server-rack" />{tr("服务器", "Server")}' in APP
    assert '<Icon name="memory" />{tr("资源", "Resources")}' in APP
    assert "const STATIC_PANEL_LAYOUT = true" in APP
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
    assert 'aria-label={tr("重置当前布局", "Reset current layout")}' not in APP
    assert "setDashboardLayoutReset" not in APP
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
    assert 'tr("已完成", "Completed")' in APP
    assert 'tr("清理已完成", "Clear completed")' in APP
    assert "api.clearCompletedJobs()" in APP
    assert "api.deleteJob(id)" in APP
    assert ".completed-jobs" in STYLES


def test_server_page_matches_hivellm_information_architecture():
    assert "const serverPage = (" in APP
    assert 'title={tr("运行服务", "Runtime")}' in APP
    assert 'title={tr("内存规划", "Memory plan")}' in APP
    assert 'title={tr("持久化前缀缓存", "Persistent Prefix cache")}' in APP
    assert 'title={tr("对话", "Chat")}' in APP
    assert 'title={tr("自动化", "Automation")}' in APP
    assert 'title={tr("Runtime 可执行文件", "Runtime executable")}' in APP
    assert 'title={tr("模型 ID", "Model ID")}' in APP
    assert 'title={tr("绑定地址", "Bind address")}' in APP
    assert 'title={tr("模型总驻留", "Total model residency")}' in APP
    assert 'title={tr("启用 SSD 层", "Enable SSD tier")}' in APP
    assert 'title={tr("最大输出", "Maximum output")}' in APP
    assert 'dashboardPage === "connections" && serverPage' in APP
    assert "const toolsRoutingPanel = <>" in APP
    assert "{toolsRoutingPanel}" in APP
    assert "studioOpen" not in APP
    assert "setStudioOpen" not in APP
    assert 'className="server-active-notice"' in APP
    assert ".server-active-notice" in STYLES
