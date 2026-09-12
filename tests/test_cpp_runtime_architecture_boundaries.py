"""Executable ownership rules for shared native-runtime behavior."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / "cpp_runtime" / "backends" / "metal"
MODELS = METAL / "models"
MTP_HEADER = (METAL / "runtime" / "mlx_mtp.h").read_text(encoding="utf-8")
MTP_SOURCE = (METAL / "runtime" / "mlx_mtp.cpp").read_text(encoding="utf-8")
SAMPLING_HEADER = (METAL / "runtime" / "mlx_sampling.h").read_text(
    encoding="utf-8"
)
TRANSFORMER_HEADER = (METAL / "runtime" / "mlx_transformer.h").read_text(
    encoding="utf-8"
)
QWEN = (MODELS / "qwen35" / "mlx_qwen35_causal_lm.cpp").read_text(
    encoding="utf-8"
)
DSV = (
    MODELS / "deepseek_family" / "mlx_deepseek_family_causal_lm.cpp"
).read_text(
    encoding="utf-8"
)
CONTRIBUTING = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
METAL_DECODE = (METAL / "apps" / "mfq_decode_mlx.cpp").read_text(
    encoding="utf-8"
)


def model_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in MODELS.rglob("*")
        if path.suffix in {".h", ".cpp"}
    )


def test_development_rules_forbid_architecture_bound_reuse() -> None:
    normalized = " ".join(CONTRIBUTING.split())
    assert "reusable code must not be architecture-bound" in normalized
    assert "mandatory extraction point" in normalized
    runtime_readme = " ".join(
        (ROOT / "cpp_runtime" / "README.md")
        .read_text(encoding="utf-8")
        .split()
    )
    assert "must never live in a model-architecture directory" in runtime_readme


def test_deepseek_v4_tree_has_no_v41_implementation() -> None:
    v4_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (MODELS / "deepseek_v4").rglob("*")
        if path.suffix in {".h", ".cpp", ".inc"}
    )
    assert re.search(r"deepseek[_ -]?v?4[.]?1|dsv41", v4_sources, re.I) is None

    v41 = MODELS / "deepseek_v41"
    assert (v41 / "mlx_deepseek_v41_deepselect.cpp").is_file()
    assert (v41 / "mlx_deepseek_v41_hf_engram.cpp").is_file()


def test_dsv41_raw_hf_tuning_policy_is_architecture_owned() -> None:
    family_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (MODELS / "deepseek_family").rglob("*")
        if path.suffix in {".h", ".cpp", ".inc"}
    )
    policy = (
        MODELS / "deepseek_v41" / "mlx_deepseek_v41_hf_policy.h"
    ).read_text(encoding="utf-8")
    for setting in (
        "MFQ_METAL_DSV41_FAST_INDEXER",
        "MFQ_METAL_DSV41_CIRCULAR_PREFILL",
        "MFQ_METAL_DSV41_FUSED_KV_PREP",
        "MFQ_METAL_DSV41_EXACT_HC_POST",
        "MFQ_METAL_DSV41_PREFILL_LAYER_GROUP",
        "MFQ_DEEPSEEK_V41_ENGRAM_CACHE_MIB",
    ):
        assert setting in policy
        assert setting not in family_sources

    deepselect = (
        MODELS / "deepseek_v41" / "mlx_deepseek_v41_deepselect.cpp"
    ).read_text(encoding="utf-8")
    assert "MFQ_METAL_DSV41_DEEPSELECT" in deepselect
    assert "MFQ_METAL_DSV41_DEEPSELECT" not in family_sources


def test_raw_hf_dispatch_uses_architecture_owned_facades() -> None:
    v41_dispatch = METAL_DECODE.index("run_native_hf_v41_server(arguments)")
    v4_dispatch = METAL_DECODE.index("run_native_hf_v4_server(arguments)")
    assert v41_dispatch < v4_dispatch
    assert "load_deepseek_v41_hf" in METAL_DECODE
    facade = (
        MODELS / "deepseek_v41" / "mlx_deepseek_v41_hf_causal_lm.h"
    ).read_text(encoding="utf-8")
    assert "MlxDeepseekFamilyCausalLm load_deepseek_v41_hf(" in facade


def test_mtp_policy_and_lifecycle_are_runtime_owned() -> None:
    for symbol in (
        "struct MlxMtpGenerationStats",
        "class MlxMtpDepthController",
        "struct MlxMtpEngineCallbacks",
        "run_mlx_mtp_generation",
        "verify_stochastic_mtp_top_k_chain_device",
    ):
        assert symbol in MTP_HEADER or symbol in MTP_SOURCE

    sources = model_sources()
    assert "struct MlxMtpGenerationStats" not in sources
    assert "class MlxMtpDepthController" not in sources
    assert "verify_greedy_mtp(" not in sources
    assert "verify_stochastic_mtp(" not in sources
    assert "host_sampling_distribution(" not in sources


def test_qwen_and_dspark_are_thin_clients_of_one_mtp_engine() -> None:
    assert QWEN.count("run_mlx_mtp_generation(") == 1
    assert DSV.count("run_mlx_mtp_generation(") == 1
    assert "draft_greedy(" not in DSV
    assert "MlxMtpGenerationStats last_mtp_stats_" in (
        MODELS / "deepseek_family" / "mlx_deepseek_family_causal_lm.h"
    ).read_text(encoding="utf-8")
    assert "begin_speculative_target(1, draft_count + 1)" in DSV
    assert "rollback_speculative_target(" in DSV
    assert "forward_chunk(\n                    committed_ids" not in DSV


def test_generic_generation_and_sequence_cache_helpers_are_not_redeclared() -> None:
    assert "mlx_last_token_logits" in SAMPLING_HEADER
    assert "class MlxSequenceCache" in TRANSFORMER_HEADER
    sources = model_sources()
    forbidden_definitions = (
        r"\b(?:array|mlx::core::array)\s+last_(?:token_)?logits\s*\(",
        r"\bclass\s+SequenceCache\b",
        r"\bstd::optional<[^>]*array[^>]*>\s+\w*generation_token_counts\s*\(",
    )
    for pattern in forbidden_definitions:
        assert re.search(pattern, sources) is None, pattern
