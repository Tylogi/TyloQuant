import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "MFQStudio" / "desktop"
TAURI = STUDIO / "src-tauri"
RUST = (TAURI / "src" / "main.rs").read_text(encoding="utf-8")
BUILD = (TAURI / "build.rs").read_text(encoding="utf-8")
APP = (ROOT / "MFQStudio" / "web" / "src" / "App.tsx").read_text(encoding="utf-8")


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
    assert "toggleDuplex" in APP
