from pathlib import Path


ROOT = Path(__file__).parents[1]
DECODE = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(encoding="utf-8")
GRAPH = (ROOT / "cpp_runtime" / "minicpmo45_runtime.inc").read_text(
    encoding="utf-8"
)


def test_minicpmo45_uses_native_composite_graph_and_hf_names():
    assert '#include "minicpmo45_runtime.inc"' in DECODE
    assert 'c.hf_model_prefix = "llm.model."' in DECODE
    assert 'c.hf_output_name = "llm.lm_head.weight"' in DECODE
    assert "c.is_minicpmo45()) ? 0.0 : 1.0" in DECODE


def test_minicpmo45_graph_binds_all_checkpoint_components():
    required_names = (
        "vpm.embeddings.patch_embedding.weight",
        "vpm.encoder.layers.",
        "resampler.attn.in_proj_weight",
        "apm.conv1.weight",
        "apm.layers.",
        "audio_projection_layer.linear1",
        'result.hf_model_prefix = "tts.model."',
        "tts.emb_text.weight",
        "tts.emb_code.0.weight",
        "tts.projector_semantic.linear1",
        "tts.head_code.0.parametrizations.weight.original0",
        "tts.head_code.0.parametrizations.weight.original1",
    )
    for name in required_names:
        assert name in GRAPH


def test_minicpmo45_audio_and_tts_follow_official_attention_contracts():
    assert "query_positions / 50 + 1" in GRAPH
    assert "raw_lengths.to(" in GRAPH
    assert "at::scaled_dot_product_attention(" in GRAPH
    assert "torch::baddbmm(" in GRAPH
    assert "at::linear(" in GRAPH
    assert 'result.hf_model_prefix = "tts.model."' in GRAPH
    assert "result.norm_weight_offset = 0.0" in GRAPH
    assert "cache_position += tokens" in GRAPH


def test_minicpmo45_resampler_requires_exact_numpy_position_asset():
    assert "minicpmo45-resampler-pos-embed-v1.bf16" in GRAPH
    assert "MFQRSPB1" in GRAPH
    assert ".narrow(0, 0, height)" in GRAPH
    assert "sincos_position" not in GRAPH


def test_minicpmo45_supports_python_tensor_files_and_bfloat16_tts():
    assert "torch::pickle_load(bytes)" in GRAPH
    assert "torch::pickle_save(" in GRAPH
    assert "rr.scalar_type() == torch::kBFloat16" in DECODE
    assert "ff2.scalar_type() == torch::kBFloat16" in DECODE
    assert "down.is_dense() || down.is_mxfp8()" in DECODE


def test_minicpmo45_qwen_runtime_follows_official_bfloat16_sdpa_path():
    assert "qwen_rms_norm_bf16" in DECODE
    assert "active_rope.apply_bf16" in DECODE
    assert "c.is_minicpmo45() ? torch::kBFloat16" in DECODE
    assert "k.scalar_type() != torch::kFloat16" in DECODE
    assert "const bool aten_decode_enabled = c.is_minicpmo45()" in DECODE
    assert "logits = logits.to(torch::kBFloat16)" in DECODE
    assert "repeated_k = kh.repeat_interleave(repeat, 1)" in DECODE
    assert '"full.minicpmo45_ffn_swiglu"' in DECODE
    assert "torch::silu(gate) * up" in DECODE


def test_minicpmo45_cli_exposes_tensor_fixture_contract():
    assert 'a == "--minicpmo-input-prefix"' in DECODE
    assert 'a == "--minicpmo-output-prefix"' in DECODE
    assert 'a == "--minicpmo-tts-steps"' in DECODE
    assert 'input_prefix + ".input_ids.pt"' in GRAPH
    assert 'output_prefix + ".image_embeddings.pt"' in GRAPH
    assert 'output_prefix + ".audio_embeddings.pt"' in GRAPH
    assert 'output_prefix + ".tts_codes.pt"' in GRAPH
