from pathlib import Path

from mfqd.capabilities import capabilities_for_architecture


ROOT = Path(__file__).resolve().parents[1]
SERVER = (ROOT / "cpp_runtime" / "mfq_server.cpp").read_text(encoding="utf-8")
WEB_APP = (ROOT / "cpp_runtime" / "web" / "app.js").read_text(encoding="utf-8")
WEB_AUDIO = (ROOT / "cpp_runtime" / "web" / "realtime-audio.js").read_text(
    encoding="utf-8"
)
WEB_HTML = (ROOT / "cpp_runtime" / "web" / "index.html").read_text(
    encoding="utf-8"
)


def test_minicpmo_family_registers_every_supported_modality() -> None:
    profile = capabilities_for_architecture("minicpmo")
    assert profile.architecture_family == "minicpmo"
    assert profile.features.model_dump() == {
        "text": True,
        "image_input": True,
        "video_input": True,
        "audio_input": True,
        "audio_output": True,
        "full_duplex": True,
    }


def test_text_architecture_families_do_not_advertise_media() -> None:
    for model_type, family in (
        ("deepseek_v4", "deepseek_v4"),
        ("glm_moe_dsa", "glm_dsa"),
        ("gemma4_text", "gemma4"),
        ("qwen3_5_text", "qwen3_5"),
    ):
        profile = capabilities_for_architecture(model_type)
        assert profile.architecture_family == family
        assert profile.features.text is True
        assert not any(
            (
                profile.features.image_input,
                profile.features.video_input,
                profile.features.audio_input,
                profile.features.audio_output,
                profile.features.full_duplex,
            )
        )


def test_unknown_architecture_keeps_text_and_a_stable_family_key() -> None:
    profile = capabilities_for_architecture("Future Model/2")
    assert profile.architecture_family == "future_model_2"
    assert profile.features.text is True
    assert profile.source == "architecture-registry:future_model_2"


def test_cpp_server_publishes_the_same_architecture_capability_contract() -> None:
    assert "architecture_capability_profile(" in SERVER
    assert 'identity == "minicpmo"' in SERVER
    assert 'identity == "deepseek_v4"' in SERVER
    assert 'identity == "glm_moe_dsa"' in SERVER
    for feature in (
        "text",
        "image_input",
        "video_input",
        "audio_input",
        "audio_output",
        "full_duplex",
    ):
        assert f'{{"{feature}", profile.{feature}}}' in SERVER
    assert '{"model_capabilities", model_capabilities}' in SERVER


def test_webui_displays_capabilities_and_exposes_only_the_duplex_mode_switch() -> None:
    assert 'id="model-capabilities"' in WEB_HTML
    assert 'id="duplex-mode-toggle"' in WEB_HTML
    assert "renderModelCapabilities()" in WEB_APP
    for feature in (
        "text",
        "image_input",
        "video_input",
        "audio_input",
        "audio_output",
        "full_duplex",
    ):
        assert f'["{feature}",' in WEB_APP
    assert "heldHalfDuplexChunk" in WEB_AUDIO
    assert "forceListen: true" in WEB_AUDIO
    assert "forceSpeak: true" in WEB_AUDIO
    assert 'event.type === "response.step.done"' in WEB_AUDIO
