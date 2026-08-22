from pathlib import Path

ROOT = Path(__file__).parents[1]
DECODE = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(encoding="utf-8")
ROPE = (ROOT / "mfq" / "kernels" / "cuda" / "rope.cu").read_text(
    encoding="utf-8"
)
ATTENTION = (ROOT / "mfq" / "kernels" / "cuda" / "attention.cu").read_text(
    encoding="utf-8"
)
NORM = (ROOT / "mfq" / "kernels" / "cuda" / "norm.cu").read_text(
    encoding="utf-8"
)
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
SERVER_HEADER = (ROOT / "cpp_runtime" / "mfq_server.h").read_text(
    encoding="utf-8"
)
SERVER_SOURCE = (ROOT / "cpp_runtime" / "mfq_server.cpp").read_text(
    encoding="utf-8"
)
REALTIME_GATEWAY = (
    ROOT / "mfq" / "runtime" / "minicpmo45_realtime.py"
).read_text(encoding="utf-8")
STUDIO_APP = (ROOT / "MFQStudio" / "src" / "App.tsx").read_text(
    encoding="utf-8"
)
STUDIO_REALTIME = (
    ROOT / "MFQStudio" / "src" / "realtimeAudio.ts"
).read_text(encoding="utf-8")


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
    assert "mfq_scaled_dot_product_attention(" in GRAPH
    assert "mfq_tensor_backend::baddbmm(" in GRAPH
    assert "mfq_linear(" in GRAPH
    assert 'result.hf_model_prefix = "tts.model."' in GRAPH
    assert 'result.model_type = "minicpmtts"' in GRAPH
    assert 'model_type == "minicpmo" || model_type == "minicpmtts"' in DECODE
    assert "result.norm_weight_offset = 0.0" in GRAPH
    assert "cache_position += tokens" in GRAPH
    assert "generate_official(" in GRAPH
    assert "mfq_tensor_backend::multinomial(" in GRAPH
    assert "repetition_penalty = 1.05" in GRAPH
    assert "if (!generated.empty())" in GRAPH
    assert "sampled.size(1) - (hit_eos ? 1 : 0)" in GRAPH
    assert "logits_trace->push_back(raw_step_logits.clone())" in GRAPH


def test_minicpmo45_resampler_requires_exact_numpy_position_asset():
    assert "minicpmo45-resampler-pos-embed-v1.bf16" in GRAPH
    assert "MFQRSPB1" in GRAPH
    assert ".narrow(0, 0, height)" in GRAPH
    assert "sincos_position" not in GRAPH


def test_minicpmo45_supports_native_tensor_files_and_bfloat16_tts():
    assert "mfq_tensor_backend::pickle_load(bytes)" in GRAPH
    assert "mfq_tensor_backend::pickle_save(" in GRAPH
    assert "MFQTNSR1" in (ROOT / "cpp_runtime" / "cuda" / "mfq_native_tensor.cpp").read_text(
        encoding="utf-8"
    )
    assert "rr.scalar_type() == mfq_tensor_backend::kBFloat16" in DECODE
    assert "ff2.scalar_type() == mfq_tensor_backend::kBFloat16" in DECODE
    assert "down.is_dense() || down.is_mxfp8()" in DECODE
    assert "official_bf16 ? mfq_tensor_backend::kBFloat16" in DECODE


def test_minicpmo45_qwen_runtime_follows_official_bfloat16_boundaries():
    assert "qwen_rms_norm_bf16" in DECODE
    assert 'std::getenv("MFQ_MINICPM_FUSED_BF16_RMSNORM")' in DECODE
    assert "qwen_rms_norm_bf16_cuda(" in DECODE
    assert "qwen_rms_norm_pair_bf16_cuda(" in DECODE
    assert "qwen_rms_norm_bf16_finalize_kernel" in NORM
    assert "qwen_rms_norm_pair_bf16_finalize_kernel" in NORM
    assert "attention_cache_decode_split_gqa4_d128_part_kernel" in ATTENTION
    assert "mfq_dispatch_bfloat16" in ATTENTION
    assert "minicpm_bf16_rope_cache_write_cuda" in DECODE
    assert 'std::getenv("MFQ_MINICPM_FUSED_ROPE_KV")' in DECODE
    assert "minicpm_bf16_rope_cache_write_kernel" in ROPE
    assert "minicpm_qk_norm_rope_cache_write_bf16_kernel" in NORM
    assert 'std::getenv("MFQ_MINICPM_FUSED_QK_NORM_ROPE_KV")' in DECODE
    assert "active_rope.apply_bf16" in DECODE
    assert "official_bf16 ? mfq_tensor_backend::kBFloat16" in DECODE
    assert "k.scalar_type() != mfq_tensor_backend::kFloat16" in DECODE
    assert 'std::getenv("MFQ_MINICPM_BF16_GQA_DECODE")' in DECODE
    assert "const bool bf16_gqa_decode = official_bf16 && T == 1" in DECODE
    assert "official_bf16 && !bf16_gqa_decode" in DECODE
    assert "logits = logits.to(mfq_tensor_backend::kBFloat16)" in DECODE
    assert "repeated_k = kh.repeat_interleave(repeat, 1)" in DECODE
    assert '"full.minicpmo45_ffn_swiglu"' in DECODE
    assert "mfq_tensor_backend::silu(gate) * up" in DECODE
    assert "return logits_from_hidden(" in DECODE
    assert "last.to(mfq_tensor_backend::kBFloat16)" in DECODE
    assert "cache_pos > 0 && T > 1" in DECODE
    assert "minicpmo45_attention_mask" in DECODE
    assert "std::numeric_limits<mfq_bfloat16>::lowest()" in DECODE
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
    assert "mfq_tensor_backend::reciprocal(" in DECODE
    assert "freq.copy_(official_freq)" in DECODE
    assert "0, -1, device, c.is_minicpmo45())" in DECODE
    assert "rope_table_bf16_cuda" in DECODE
    assert "rope_table_bf16_kernel" in ROPE


def test_minicpmo45_serializes_independent_decode_projection_branches():
    assert "bool decode_branch_parallel = true" in DECODE
    assert "result.decode_branch_parallel = !preserve_projection_boundaries" in DECODE
    assert "decode_branch_parallel &&" in DECODE


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
    assert "mfq_tensor_backend::topk(" in GRAPH
    assert "logits, top_k, -1, true, true" in GRAPH
    assert "cumulative > top_p" in GRAPH
    assert "generate_duplex_chunk(" in GRAPH
    assert "sampled.size() > 16" in GRAPH
    assert "condition.size(1) + result.tts_codes.size(1)" in GRAPH
    assert "int64_t top_k = 100" in GRAPH
    assert "double length_penalty = 1.0" in GRAPH
    assert "selected * length_penalty" in GRAPH


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


def test_minicpmo45_cuda_server_binds_the_realtime_backend():
    assert 'a == "--minicpmo-duplex"' not in DECODE
    assert "make_cuda_minicpmo45_duplex_backend(" in DECODE
    assert "if (server_minicpmo_runtime)" in DECODE
    assert 'backend.name = "cuda"' in DECODE
    assert "MiniCPMO45Runtime::load_with_language(" in DECODE
    assert "session->prepare(" in DECODE
    assert "parameters.reference_audio_features" in DECODE
    assert "input.force_speak" in DECODE
    assert "result.tts_force_flush" in DECODE


def test_minicpmo45_native_servers_share_mfqd_vision_tensors():
    assert "struct MfqVisionInput" in SERVER_HEADER
    assert "MfqMultimodalGenerateFn" in SERVER_HEADER
    assert "parse_mfq_vision(" in SERVER_SOURCE
    assert "class TensorFileReader final" in SERVER_SOURCE
    assert 'value.contains("binary_file")' in SERVER_SOURCE
    assert "file_reader->read(" in SERVER_SOURCE
    assert 'single_special_token(tokenizer, "<image>")' in SERVER_SOURCE
    assert "MiniCPM-o image placeholder must contain 64 query tokens" in SERVER_SOURCE
    assert "generate_server_multimodal_tokens(" in DECODE
    assert "runtime.forward(" in DECODE
    assert "sample_server_logits(" in DECODE
    assert "generate_multimodal(" in METAL_HEADER
    assert "MlxMiniCPMO45Runtime::generate_multimodal(" in METAL_GRAPH
    assert "MfqMultimodalGenerateFn multimodal_generate" in METAL_DECODE
    assert "arguments.minicpmo_duplex" not in METAL_DECODE
    assert "Runtime, mfq::metal::MlxMiniCPMO45Runtime" in METAL_DECODE


def test_minicpmo45_metal_dispatches_m3_family_tuning_by_device():
    assert 'sysctlbyname(\n            "machdep.cpu.brand_string"' in METAL_GRAPH
    assert '"MFQ_MINICPM_METAL_PROFILE"' in METAL_GRAPH
    assert 'std::strcmp(requested, "m3") == 0' in METAL_GRAPH
    assert 'std::strcmp(requested, "baseline") == 0' in METAL_GRAPH
    assert 'chip_name.rfind("Apple M3", 0) == 0' in METAL_GRAPH
    assert '"Apple M3 Pro"' in METAL_GRAPH
    assert '"Apple M3 Ultra"' in METAL_GRAPH
    assert '"Apple M4 Max"' in METAL_GRAPH
    assert '"Apple M5 Max"' in METAL_GRAPH
    assert "std::clamp((sequence + 255) / 256, 8, 16)" in METAL_GRAPH
    assert "std::min(6, (sequence + 1'023) / 1'024)" in METAL_GRAPH


def test_minicpmo45_metal_tracks_both_in_place_kv_cache_outputs():
    assert (
        "encoder.register_output_array(inputs[5]);\n"
        "        encoder.register_output_array(inputs[6]);"
    ) in METAL_GRAPH
    assert (
        "encoder.register_output_array(inputs[3]);\n"
        "        encoder.register_output_array(inputs[4]);"
    ) in METAL_GRAPH


def test_minicpmo45_cuda_duplex_uses_runtime_profile_tts_sampling():
    assert "double tts_temperature = 0.8" in GRAPH
    assert "double tts_repetition_penalty = 1.05" in GRAPH
    assert "tts_temperature, tts_repetition_penalty" in GRAPH
    assert 'number_field(\n                            generation, "tts_temperature"' in SERVER_SOURCE
    assert 'number_field(\n                            generation, "tts_repetition_penalty"' in SERVER_SOURCE


def test_minicpmo45_eval_batch_matches_pr_tts_sampler_and_optional_media():
    assert "std::mt19937 * evaluator_rng = nullptr" in GRAPH
    assert "std::uniform_real_distribution<float> distribution" in GRAPH
    assert 'request.value("tts_temperature", 0.8)' in GRAPH
    assert 'request.value("tts_top_p", 0.85)' in GRAPH
    assert 'request.value("tts_top_k", int64_t{25})' in GRAPH
    assert 'request.value("tts_min_tokens_to_keep", int64_t{3})' in GRAPH
    assert 'input_prefix + ".pixel_values.pt", false' in GRAPH
    assert '(reuse_prefix_cache &&\n                     prefix_length >= input_ids.size(1))' in GRAPH


def test_minicpmo45_eval_batch_maps_each_audio_bound_to_its_source():
    assert "valid_lengths[bound.source]" in GRAPH
    assert "bound.source, Slice(0, available), Slice()" in GRAPH
    assert "used_audio[static_cast<size_t>(bound.source)] = true" in GRAPH
    assert "MiniCPM-o eval batch has unused Whisper segments" in GRAPH


def test_minicpmo45_eval_batch_preserves_pr_teacher_forcing_segments():
    assert 'current_prefix + ".prefill_splits.pt",' in GRAPH
    assert "tts_teacher_forcing || require_segmented_prefill" in GRAPH
    assert "segment_begin == text_begin && segment_end == text_end" in GRAPH
    assert "teacher_text_hidden = segment_hidden" in GRAPH
    assert "teacher-forcing splits omit the text span" in GRAPH


def test_minicpmo45_eval_batch_can_match_pr_prefill_call_boundaries():
    assert 'request.value("require_segmented_prefill", false)' in GRAPH
    assert 'request.value("segmented_prefill_chunk_tokens", int64_t{0})' in GRAPH
    assert 'first_prefix + ".prefill_splits.pt"' in GRAPH
    assert "minicpmo45_prefill_segments(" in GRAPH
    assert "minicpmo45_teacher_prefill_segments(" in GRAPH
    assert "boundary - segment_begin >= chunk_tokens" in GRAPH
    assert "segments.emplace_back(text)" in GRAPH
    assert "teacher-forcing splits omit the text span" in GRAPH
    assert "MiniCPM-o segmented prefill omits a prompt span" in GRAPH
    assert "image_embedding_parts.push_back(runtime.resampler.forward(" in GRAPH
    assert "MiniCPM-o per-segment Whisper length mismatch" in GRAPH
    assert "Slice(0, raw_lengths[index])" in GRAPH


def test_minicpmo45_realtime_renderer_prefers_cuda_when_available():
    cuda_probe = "if torch.cuda.is_available():"
    mps_probe = "elif torch.backends.mps.is_available():"
    assert cuda_probe in REALTIME_GATEWAY
    assert mps_probe in REALTIME_GATEWAY
    assert REALTIME_GATEWAY.index(cuda_probe) < REALTIME_GATEWAY.index(mps_probe)


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


def test_minicpmo45_metal_fuses_streaming_audio_projections_safely():
    assert '"MFQ_METAL_AUDIO_FUSED_QKV"' in METAL_GRAPH
    assert "mlx::core::concatenate(" in METAL_GRAPH
    assert "auto pieces = mlx::core::split(" in METAL_GRAPH
    assert '"MFQ_METAL_AUDIO_STREAMING_NO_MASK"' in METAL_GRAPH
    assert "? std::nullopt" in METAL_GRAPH
    assert '"MFQ_METAL_AUDIO_BITWISE_HASH"' in METAL_GRAPH


def test_minicpmo45_metal_batches_only_official_vision_geometry():
    assert "minicpmo_vision_batchable_length" in METAL_GRAPH
    assert "return tokens >= 900 && tokens <= 1100" in METAL_GRAPH
    assert METAL_GRAPH.count("!minicpmo_vision_batchable_length(") == 2


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
    assert "kMinicpmoDuplexCacheLimitBytes" in METAL_DECODE
    assert "set_cache_limit(kMinicpmoDuplexCacheLimitBytes)" in METAL_DECODE
    assert "runtime_holder->value().reset();\n                    mlx::core::synchronize(runtime_stream);\n                    mlx::core::clear_cache();" in METAL_DECODE


def test_minicpmo45_realtime_uses_official_demo_defaults():
    assert "int32_t top_k = 100" in SERVER_HEADER
    assert "double length_penalty = 1.0" in SERVER_HEADER
    assert "duplex_defaults.top_k.value_or(100)" in SERVER_SOURCE
    assert "duplex_defaults.length_penalty.value_or(1.0)" in SERVER_SOURCE
    assert "std::random_device random" in SERVER_SOURCE
    assert "std::int32_t top_k = 100" in METAL_HEADER
    assert "double length_penalty = 1.0" in METAL_HEADER
    assert "config.length_penalty" in METAL_GRAPH
    assert '"force_listen_count": 0' in REALTIME_GATEWAY
    assert '"length_penalty": 1.0' in REALTIME_GATEWAY
    assert "await self.backend_runtime_defaults()" in REALTIME_GATEWAY
    assert "token2wav_steps: int = 10" in REALTIME_GATEWAY
    assert "DEFAULT_DUPLEX_SYSTEM_PROMPT" in REALTIME_GATEWAY
    assert "const SPEAK_TOKENS = 20" in STUDIO_REALTIME
    assert "const PLAYBACK_DELAY_SECONDS = 0.2" in STUDIO_REALTIME
    assert "REALTIME_SYSTEM_PROMPTS" not in STUDIO_APP
    assert "systemPrompt: effectiveSettings.systemPrompt" in STUDIO_APP


def test_minicpmo45_realtime_preserves_official_first_tts_flush():
    assert "tts_force_flush" in METAL_HEADER
    assert "first_tts_chunk" in METAL_GRAPH
    assert "end_of_turn || first_tts_chunk ? 0 : 26" in METAL_GRAPH
    assert '{"force_flush", result.tts_force_flush}' in SERVER_SOURCE
    assert "result.end_of_turn && !result.is_listen" in SERVER_SOURCE
    assert "force_flush=bool(event.get(\"force_flush\", False))" in REALTIME_GATEWAY
