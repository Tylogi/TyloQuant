from pathlib import Path

from mfq.server.capabilities import capabilities_for_architecture

ROOT = Path(__file__).resolve().parents[1]
SERVER = (ROOT / "cpp_runtime" / "server" / "src" / "server.cpp").read_text(encoding="utf-8")
CUDA_PLAN = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "include" / "mfq_cuda_model_plan.h"
).read_text(encoding="utf-8")
CUDA_DECODE = (ROOT / "cpp_runtime" / "backends" / "cuda" / "apps" / "mfq_decode.cpp").read_text(
    encoding="utf-8"
)
CUDA_COMPONENTS = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "runtime" / "server_components.h"
).read_text(encoding="utf-8")
STUDIO_APP = (ROOT / "MFQStudio" / "src" / "App.tsx").read_text(
    encoding="utf-8"
)
STUDIO_AUDIO = (ROOT / "MFQStudio" / "src" / "realtimeAudio.ts").read_text(
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
        "mtp": False,
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


def test_deepseek_v4_vision_alias_advertises_only_image_input() -> None:
    profile = capabilities_for_architecture("deepseek_v4_vision")
    assert profile.architecture_family == "deepseek_v4"
    assert profile.features.model_dump() == {
        "text": True,
        "image_input": True,
        "video_input": False,
        "audio_input": False,
        "audio_output": False,
        "full_duplex": False,
        "mtp": True,
    }


def test_deepseek_v41_advertises_vision_and_mtp() -> None:
    profile = capabilities_for_architecture("deepseek_v41")
    assert profile.architecture_family == "deepseek_v41"
    assert profile.features.text
    assert profile.features.image_input
    assert profile.features.mtp


def test_every_mtp_runtime_family_registers_architecture_support() -> None:
    for model_type in (
        "deepseek_v4",
        "deepseek_v4_vision",
        "qwen3_5",
        "qwen3_5_text",
        "qwen4_exp",
        "glm5_next",
    ):
        assert capabilities_for_architecture(model_type).features.mtp


def test_unknown_architecture_keeps_text_and_a_stable_family_key() -> None:
    profile = capabilities_for_architecture("Future Model/2")
    assert profile.architecture_family == "future_model_2"
    assert profile.features.text is True
    assert profile.source == "architecture-registry:future_model_2"


def test_cpp_server_publishes_the_same_architecture_capability_contract() -> None:
    assert "architecture_capability_profile(" in SERVER
    assert "kModelCapabilityRegistry" in SERVER
    for model_type in (
        "minicpmo",
        "deepseek_v4",
        "deepseek_v4_vision",
        "glm_moe_dsa",
        "qwen3_5",
        "qwen4_exp",
        "glm5_next",
    ):
        assert f'"{model_type}"' in SERVER
    for feature in (
        "text",
        "image_input",
        "video_input",
        "audio_input",
        "audio_output",
        "full_duplex",
        "mtp",
    ):
        assert f'{{"{feature}", profile.{feature}}}' in SERVER
    assert '{"model_capabilities", model_capabilities}' in SERVER
    for state in (
        "vision_supported",
        "vision_available",
        "video_available",
        "mtp_supported",
        "mtp_available",
        "vision_enabled_default",
        "mtp_enabled_default",
    ):
        assert f'"{state}"' in SERVER


def test_cpp_server_keeps_health_metrics_out_of_response_performance() -> None:
    assert SERVER.count("add_request_runtime_metrics(performance)") == 2
    assert "add_runtime_metrics(performance)" not in SERVER
    for metric in (
        "mtp_available",
        "mtp_used",
        "mtp_cycles",
        "mtp_drafted_tokens",
        "mtp_accepted_tokens",
        "mtp_acceptance_rate",
        "mtp_target_ms",
        "mtp_head_ms",
        "mtp_rollback_ms",
    ):
        assert f'"{metric}"' in SERVER


def test_cuda_uses_one_architecture_and_optional_component_registry() -> None:
    assert "MfqCudaModelPlan" in CUDA_PLAN
    assert "MfqCudaComponentState" in CUDA_PLAN
    assert "mfq_cuda_model_plan(" in CUDA_PLAN
    assert "mfq_cuda_component_state(" in CUDA_PLAN
    for implementation in (
        "deepseek_v4",
        "qwen3_5",
        "gemma4",
        "glm_dsa",
        "minicpmo45",
    ):
        assert f'"{implementation}"' in CUDA_PLAN

    assert "load_cuda_runtime_components(" in CUDA_DECODE
    assert "CudaRuntimeComponents server_components" in CUDA_DECODE
    assert "switch (result.plan.vision)" in CUDA_COMPONENTS
    assert "server_minicpmo_runtime" not in CUDA_DECODE


def test_studio_displays_capabilities_and_gates_voice_modes() -> None:
    assert '(["text", "voice", "full_duplex"] as SessionMode[])' in STUDIO_APP
    assert "!feature.audio_input" in STUDIO_APP
    assert "!feature.full_duplex" in STUDIO_APP
    assert "features.mtp === true" in STUDIO_APP
    assert "enableVision: !effectiveSettings.enableVision" in STUDIO_APP
    assert "enableMtp: !effectiveSettings.enableMtp" in STUDIO_APP
    assert "heldHalfDuplexChunk" in STUDIO_AUDIO
    assert "forceListen: true" in STUDIO_AUDIO
    assert "forceSpeak: true" in STUDIO_AUDIO
    assert 'event.type === "response.step.done"' in STUDIO_AUDIO
