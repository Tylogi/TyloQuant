from pathlib import Path

ROOT = Path(__file__).parents[1]
DECODE = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(encoding="utf-8")
GRAPH = (ROOT / "cpp_runtime" / "minicpmo45_runtime.inc").read_text(
    encoding="utf-8"
)
METAL_GRAPH = (ROOT / "cpp_runtime" / "metal" / "mlx_minicpmo45.cpp").read_text(
    encoding="utf-8"
)
METAL_HEADER = (ROOT / "cpp_runtime" / "metal" / "mlx_minicpmo45.h").read_text(
    encoding="utf-8"
)
METAL_DECODE = (ROOT / "cpp_runtime" / "metal" / "mfq_decode_mlx.cpp").read_text(
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
    assert 'result.model_type = "minicpmtts"' in GRAPH
    assert 'model_type == "minicpmo" || model_type == "minicpmtts"' in DECODE
    assert "result.norm_weight_offset = 0.0" in GRAPH
    assert "cache_position += tokens" in GRAPH
    assert "generate_official(" in GRAPH
    assert "torch::multinomial(" in GRAPH
    assert "repetition_penalty = 1.05" in GRAPH
    assert "if (!generated.empty())" in GRAPH
    assert "sampled.size(1) - 1" in GRAPH
    assert "logits_trace->push_back(raw_step_logits.clone())" in GRAPH


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
    assert "official_bf16 ? torch::kBFloat16" in DECODE


def test_minicpmo45_qwen_runtime_follows_official_bfloat16_sdpa_path():
    assert "qwen_rms_norm_bf16" in DECODE
    assert "active_rope.apply_bf16" in DECODE
    assert "official_bf16 ? torch::kBFloat16" in DECODE
    assert "k.scalar_type() != torch::kFloat16" in DECODE
    assert "const bool aten_decode_enabled = official_bf16" in DECODE
    assert "logits = logits.to(torch::kBFloat16)" in DECODE
    assert "repeated_k = kh.repeat_interleave(repeat, 1)" in DECODE
    assert '"full.minicpmo45_ffn_swiglu"' in DECODE
    assert "torch::silu(gate) * up" in DECODE
    assert "return logits_from_hidden(" in DECODE
    assert "last.to(torch::kBFloat16)" in DECODE
    assert "cache_pos > 0 && T > 1" in DECODE
    assert "minicpmo45_attention_mask" in DECODE
    assert "std::numeric_limits<c10::BFloat16>::lowest()" in DECODE
    assert "!c.is_minicpmo45() && cache_pos > 0" in DECODE
    assert "attention_mask.value().eq(1).all().item<bool>()" in DECODE


def test_minicpmo45_preserves_qkv_projection_boundaries():
    assert "bool preserve_projection_boundaries = false" in DECODE
    assert "std::move(layers), preserve_projection_boundaries" in DECODE
    assert (
        'ap + "q_proj.weight", ap + "k_proj.weight", ap + "v_proj.weight"},\n'
        "            2, nullptr, c.is_minicpmo45())"
    ) in DECODE
    assert "f.down, 2, c.is_minicpmo45())" in DECODE
    assert "MFQ_DIAGNOSTIC_DISABLE_NINT_GROUP" in DECODE


def test_minicpmo45_matches_official_rope_frequency_construction():
    assert "bool official_reciprocal_frequencies = false" in DECODE
    assert "torch::reciprocal(" in DECODE
    assert "freq.copy_(official_freq)" in DECODE
    assert "0, -1, device, c.is_minicpmo45())" in DECODE


def test_minicpmo45_cli_exposes_tensor_fixture_contract():
    assert 'a == "--minicpmo-input-prefix"' in DECODE
    assert 'a == "--minicpmo-output-prefix"' in DECODE
    assert 'a == "--minicpmo-tts-steps"' in DECODE
    assert 'input_prefix + ".input_ids.pt"' in GRAPH
    assert 'input_prefix + ".position_ids.pt"' in GRAPH
    assert 'input_prefix + ".attention_mask.pt"' in GRAPH
    assert 'output_prefix + ".image_embeddings.pt"' in GRAPH
    assert 'output_prefix + ".audio_embeddings.pt"' in GRAPH
    assert 'output_prefix + ".tts_codes.pt"' in GRAPH


def test_minicpmo45_native_duplex_preserves_streaming_caches():
    assert "forward_streaming(" in GRAPH
    assert "cache_length() + conv_tokens_before_crop" in GRAPH
    assert "prefix_extra_frames + 1" in GRAPH
    assert "suffix_extra_frames + 1" in GRAPH
    assert "MiniCPMO45DuplexSession" in GRAPH
    assert "runtime.language.reset(1)" in GRAPH
    assert "runtime.audio.reset()" in GRAPH
    assert "runtime.tts.reset(1)" in GRAPH
    assert "runtime.language.cache_pos" in GRAPH
    assert "runtime.audio.cache_length()" in GRAPH
    assert "runtime.tts.cache_position" in GRAPH
    assert "session.audio_chunk_index" in GRAPH


def test_minicpmo45_native_duplex_follows_official_unit_state_machine():
    assert "feed_id(ids.unit_start)" in GRAPH
    assert "feed_id(ids.unit_end)" in GRAPH
    assert "ids.is_chunk_terminator(token)" in GRAPH
    assert "!forced_decision && token == ids.listen" in GRAPH
    assert "!current_turn_ended" in GRAPH
    assert "token = ids.tts_bos" in GRAPH
    assert "index == max_new_speak_tokens - 1" in GRAPH
    assert "feed_id(ids.chunk_eos)" in GRAPH
    assert "if (index != 0)" in GRAPH
    assert "result.end_of_turn = token == ids.turn_eos" in GRAPH
    assert "generation_logits = pending.first" in GRAPH


def test_minicpmo45_native_duplex_uses_official_sampling_contracts():
    assert "first_id == ids.chunk_eos" in GRAPH
    assert "selected / repetition_penalty" in GRAPH
    assert "logits.index_fill_" in GRAPH
    assert "logits.topk(top_k" in GRAPH
    assert "cumulative > top_p" in GRAPH
    assert "generate_duplex_chunk(" in GRAPH
    assert "sampled.size() > 16" in GRAPH
    assert "condition.size(1) + result.tts_codes.size(1)" in GRAPH


def test_minicpmo45_cli_exposes_native_duplex_tensor_contract():
    assert 'a == "--minicpmo-duplex-input-prefix"' in DECODE
    assert 'a == "--minicpmo-duplex-output-prefix"' in DECODE
    assert 'a == "--minicpmo-duplex-steps"' in DECODE
    assert 'a == "--minicpmo-duplex-max-speak-tokens"' in DECODE
    assert 'a == "--minicpmo-duplex-seed"' in DECODE
    assert 'a == "--minicpmo-duplex-greedy"' in DECODE
    assert 'input_prefix + ".special_ids.pt"' in GRAPH
    assert 'input + ".audio_features.pt"' in GRAPH
    assert 'input + ".force_listen.pt"' in GRAPH
    assert 'input + ".reset_session.pt"' in GRAPH
    assert 'output + ".generated_ids.pt"' in GRAPH
    assert 'output + ".tts_codes.pt"' in GRAPH
    assert 'output + ".state.pt"' in GRAPH


def test_minicpmo45_metal_uses_an_independent_qwen3_backbone():
    assert "class MiniQwen3Block" in METAL_GRAPH
    assert "class MiniQwen3Language" in METAL_GRAPH
    assert '"llm.model.layers."' in METAL_GRAPH
    assert "result.query_heads != 32" in METAL_GRAPH
    assert "result.kv_heads != 8" in METAL_GRAPH
    assert "mlx_qwen35" not in METAL_GRAPH
    assert "mlx_qwen35" not in METAL_HEADER
    assert 'architecture.rfind("minicpmo", 0)' in METAL_DECODE


def test_minicpmo45_metal_binds_the_complete_composite_graph():
    for symbol in (
        "class VisionEncoder",
        "class Resampler",
        "class AudioEncoder",
        "class TtsDecoder",
        "vpm.embeddings.patch_embedding.weight",
        "resampler.attn.in_proj_weight",
        "apm.conv1.weight",
        "audio_projection_layer.linear1",
        "tts.emb_text.weight",
        "tts.head_code.0.parametrizations.weight.original0",
    ):
        assert symbol in METAL_GRAPH
    assert "minicpmo45-resampler-pos-embed-v1.bf16" in METAL_GRAPH
    assert "MFQRSPB1" in METAL_GRAPH


def test_minicpmo45_metal_duplex_tracks_all_cache_lifetimes():
    assert "prepare_duplex(" in METAL_GRAPH
    assert "duplex_step(" in METAL_GRAPH
    assert "tts_text_start_position" in METAL_GRAPH
    assert "TTS cache did not advance exactly" in METAL_GRAPH
    assert "audio_chunk_index" in METAL_GRAPH
    assert "implementation_->duplex.reset()" in METAL_GRAPH
    assert "language_cache_position" in METAL_HEADER
    assert "audio_cache_position" in METAL_HEADER
    assert "tts_cache_position" in METAL_HEADER
