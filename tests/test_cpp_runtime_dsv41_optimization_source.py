"""Source contracts for the raw-HF DeepSeek-V4 family Metal fast paths."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / "cpp_runtime" / "backends" / "metal"
FAMILY = METAL / "models" / "deepseek_family"
V41 = METAL / "models" / "deepseek_v41"
OPS = METAL / "ops"
ATTENTION = (FAMILY / "mlx_deepseek_family_attention.cpp").read_text(
    encoding="utf-8"
)
CAUSAL_LM = (FAMILY / "mlx_deepseek_family_causal_lm.cpp").read_text(
    encoding="utf-8"
)
CAUSAL_LM_HEADER = (FAMILY / "mlx_deepseek_family_causal_lm.h").read_text(
    encoding="utf-8"
)
SPARSE = (OPS / "mlx_deepseek_sparse.cpp").read_text(encoding="utf-8")
DSPARK = (FAMILY / "mlx_deepseek_family_dspark.cpp").read_text(
    encoding="utf-8"
)
MOE = (OPS / "mlx_moe.cpp").read_text(encoding="utf-8")
V41_HF_POLICY = (V41 / "mlx_deepseek_v41_hf_policy.h").read_text(
    encoding="utf-8"
)
METAL_SERVER = (
    ROOT
    / "cpp_runtime"
    / "backends"
    / "metal"
    / "apps"
    / "mfq_decode_mlx.cpp"
).read_text(encoding="utf-8")


def test_raw_hf_v41_prefill_uses_the_direct_circular_sparse_kernel() -> None:
    assert '"MFQ_METAL_DSV41_CIRCULAR_PREFILL"' in V41_HF_POLICY
    assert "deepseek_v41_hf_circular_prefill_enabled()" in ATTENTION
    assert "tokens > 1 &&" in ATTENTION
    assert "visibility == nullptr &&" in ATTENTION
    assert "pool_len > 0 &&" in ATTENTION
    assert "direct_prefill = attention_dsv4_sparse_prefill(" in ATTENTION
    assert "return mlx_sparse_circular_mla_attention(" in SPARSE


def test_v41_mtp_verifier_uses_fast_circular_attention() -> None:
    assert "acceptance-only depth policy" in ATTENTION
    assert "tokens > 1 &&" in ATTENTION


def test_chunked_prefill_skips_discarded_vocabulary_projections() -> None:
    assert "bool skip_lm_head" in CAUSAL_LM_HEADER
    assert "if (skip_lm_head)" in CAUSAL_LM
    assert "!full_logits && end < tokens && end - start > 1" in CAUSAL_LM
    assert "else if (end == tokens)" in CAUSAL_LM
    assert "stop < end && stop - start > 1" in CAUSAL_LM


def test_v41_dspark_only_evaluates_the_requested_draft_width() -> None:
    assert "const int physical_width = impl_->config.is_v41()" in DSPARK
    assert "? std::min(requested, available_width)" in DSPARK
    assert ": available_width;" in DSPARK


def test_dspark_serving_skips_unused_diagnostic_graphs() -> None:
    assert "dspark_->propose(" in CAUSAL_LM
    assert "if (!collect_diagnostics)" in DSPARK
    assert 'std::getenv("MFQ_MLX_MTP_TRACE")' in DSPARK


def test_raw_hf_v41_dspark_uses_fused_sparse_attention() -> None:
    assert "bool sparse_fast_path" in DSPARK
    assert "sparse_fast_path && heads == 64 && dimension == 512" in DSPARK
    assert "return attention_dsv4_sparse(" in DSPARK
    assert "config.is_v41());" in DSPARK


def test_v41_mtp_depth_is_acceptance_deterministic() -> None:
    assert "MlxMtpDepthPolicy::AcceptanceOnly" in CAUSAL_LM


def test_v41_new_sessions_reuse_cache_storage() -> None:
    assert "reuse_v41_storage" in CAUSAL_LM
    assert "state.reset_v41()" in CAUSAL_LM


def test_m3_ultra_nax_prefill_covers_both_flash_expert_geometries() -> None:
    assert "const bool dsv4f_geometry = experts == 256" in MOE
    assert "input_width == 4096" in MOE
    assert "output_width == 2048 || output_width == 4096" in MOE
    assert "const bool dsv41_geometry = experts == 384" in MOE
    assert "dsv4f_geometry || dsv41_geometry" in MOE
    assert "apple_m3_ultra() && optimized_geometry" in MOE


def test_m3_ultra_v4_flash_prefill_uses_balanced_4096_chunks() -> None:
    assert '"MFQ_METAL_DSV4_PREFILL_AUTOTUNE"' in METAL_SERVER
    assert "const bool dsv4_flash_geometry =" in METAL_SERVER
    assert "config.n_experts == 256" in METAL_SERVER
    assert "config.moe_inter == 2048" in METAL_SERVER
    assert (
        "effective_arguments.prefill_chunk_size = config.is_v41()"
        in METAL_SERVER
    )
    assert "? 5440" in METAL_SERVER
    assert ": 4096;" in METAL_SERVER
