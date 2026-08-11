import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "MFQStudio" / "desktop"
TAURI = STUDIO / "src-tauri"
RUST = (TAURI / "src" / "main.rs").read_text(encoding="utf-8")
BUILD = (TAURI / "build.rs").read_text(encoding="utf-8")
APP = (ROOT / "MFQStudio" / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
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


def test_studio_has_detached_windows_and_unix_mfqd_launchers():
    assert "#[cfg(windows)]" in RUST
    assert "DETACHED_PROCESS" in RUST
    assert "CREATE_NEW_PROCESS_GROUP" in RUST
    assert "#[cfg(unix)]" in RUST
    assert "command.process_group(0)" in RUST
    assert '.arg("--backend-url")' in RUST
    assert '.arg("--db")' in RUST
    assert 'command.arg("-m").arg("mfqd.cli")' in RUST
    assert "--backend-api-key" not in RUST


def test_studio_supports_local_and_remote_mfqd_with_voice_controls():
    assert "RuntimeMode::Local" in RUST
    assert "RuntimeMode::Remote" in RUST
    assert "studio_configure" in RUST
    assert "studio_start_local" in RUST
    assert "Object.keys(MODE_LABELS)" not in APP
    assert "RealtimeAudioController" in APP
    assert '(["text", "voice", "full_duplex"] as SessionMode[])' in APP
    assert "selectInteractionMode" in APP


def test_studio_drains_duplex_output_after_microphone_capture_stops():
    assert "MAX_RESPONSE_DRAIN_STEPS" in REALTIME_AUDIO
    assert "event.end_of_turn === true" in REALTIME_AUDIO
    assert "this.sendInput(new Float32Array(CHUNK_SAMPLES))" in REALTIME_AUDIO
    assert "this.callbacks.onText(target.buffer.sessionId, target.buffer.text)" in REALTIME_AUDIO


def test_studio_uses_the_model_bound_duplex_system_prompt():
    assert "modeTemplateSettings" in APP
    assert 'runtime?.duplex_sampling_defaults ?? realtime?.defaults' in APP
    assert "system_prompt: config.systemPrompt" in REALTIME_AUDIO
    assert "text_repetition_penalty: config.repetitionPenalty" in REALTIME_AUDIO
    assert "submitText(text, realtimeSessionConfig(active.id))" in APP


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
