from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "cpp_runtime" / "metal" / "mfq_perplexity_mlx.cpp"
).read_text(encoding="utf-8")
SERVER = (ROOT / "cpp_runtime" / "mfq_server.cpp").read_text(
    encoding="utf-8"
)
HEADER = (ROOT / "cpp_runtime" / "mfq_server.h").read_text(
    encoding="utf-8"
)
CMAKE = (
    ROOT / "cpp_runtime" / "metal" / "CMakeLists.txt"
).read_text(encoding="utf-8")


def test_mfq_perplexity_is_a_native_metal_target() -> None:
    assert "add_executable(mfq-perplexity" in CMAKE
    assert "mfq_perplexity_mlx.cpp" in CMAKE
    assert "mfq-metal-runtime" in CMAKE
    assert "mfq-server" in CMAKE
    assert "mfq::metal::MlxQwen35CausalLm::load" in SOURCE
    assert "mfq::metal::MlxDeepseekV4CausalLm::load" in SOURCE
    assert "Python" not in SOURCE


def test_mfq_perplexity_matches_llamacpp_window_geometry() -> None:
    assert "const int first = context / 2;" in SOURCE
    assert "context - first - 1" in SOURCE
    assert "position == 0 && tokenizer.add_bos" in SOURCE
    assert "position > first" in SOURCE
    assert "common_tokenize(..., add_special=true)" in SOURCE
    assert "tokenizer.add_eos" in SOURCE
    assert "tokenizer.tokens.size() /" in SOURCE


def test_mfq_perplexity_scores_on_gpu_and_can_export_trace_v3() -> None:
    assert "mlx::core::logsumexp(scored_logits, -1)" in SOURCE
    assert "mlx::core::take_along_axis" in SOURCE
    assert "negative_log_likelihood.eval()" in SOURCE
    assert "logits.data<float>" not in SOURCE
    assert 'value == "--logits-file"' in SOURCE
    assert "class TraceV3Writer" in SOURCE
    assert 'header.write("_logit3_", 8)' in SOURCE
    assert '"exact_float32"' in SOURCE
    assert '"linear_uint16_log_probability"' in SOURCE
    assert "TraceV3Writer::logit_range" in SOURCE


def test_logits_export_writes_a_complete_versioned_contract() -> None:
    assert '"mfq.perplexity-logits-manifest.v1"' in SOURCE
    assert '"dataset"' in SOURCE
    assert '"input_token_ids_sha256"' in SOURCE
    assert '"n_ctx"' in SOURCE
    assert '"n_batch"' in SOURCE
    assert '"n_ubatch"' in SOURCE
    assert '"n_seq"' in SOURCE
    assert '"score_count_per_chunk"' in SOURCE
    assert '"scored_tokens"' in SOURCE
    assert '"attention"' in SOURCE
    assert '"kv_cache_dtype"' in SOURCE
    assert '"record_dtype_counts"' in SOURCE
    assert "sha256_file(arguments.logits_file)" in SOURCE
    assert "sha256_file(arguments.input)" in SOURCE
    assert "model.source_paths()" in SOURCE


def test_kld_requires_and_strictly_matches_the_logits_contract() -> None:
    assert "KLD requires the logits generation contract" in SOURCE
    assert "--ctx-size differs from the logits generation contract" in SOURCE
    assert "--batch-size differs from the logits generation contract" in SOURCE
    assert "--ubatch-size differs from the logits generation contract" in SOURCE
    assert "--parallel differs from the logits generation contract" in SOURCE
    assert "--chunks differs from the logits generation contract" in SOURCE
    assert "--kl-score-count differs from the logits generation contract" in SOURCE
    assert "--dataset differs from the logits generation contract" in SOURCE
    assert "logits manifest/reference SHA-256 mismatch" in SOURCE
    assert "logits manifest/reference token SHA-256 mismatch" in SOURCE


def test_ubatch_is_a_real_physical_graph_split() -> None:
    assert "forward_with_ubatch(" in SOURCE
    assert "runtime.reset_cache(batch)" in SOURCE
    assert "tokens_per_ubatch" in SOURCE
    assert "output.eval()" in SOURCE
    assert "mlx::core::concatenate(std::move(outputs), 1)" in SOURCE


def test_mfq_perplexity_consumes_existing_kld_references_in_process() -> None:
    assert 'value == "--kl-base"' in SOURCE
    assert 'result.format == "_logits_"' in SOURCE
    assert 'result.format == "_logit2_"' in SOURCE
    assert 'result.format == "_logit3_"' in SOURCE
    assert "read_kl_rows(" in SOURCE
    assert "reference_probability" in SOURCE
    assert "normalized_reference_logp" in SOURCE
    assert '"cpp_kl_result chunks="' in SOURCE
    assert '" same_top_count="' in SOURCE
    assert '" reference_ce="' in SOURCE
    assert '" candidate_ce="' in SOURCE
    assert '" reference_precision="' in SOURCE
    assert '"bf16_ce="' not in SOURCE


def test_tokenizer_probe_exposes_llamacpp_special_token_policy() -> None:
    assert "bool add_bos = false;" in HEADER
    assert "bool add_eos = false;" in HEADER
    assert "bool add_special = false" in HEADER
    assert "llama_vocab_get_add_bos(vocab_)" in SERVER
    assert "llama_vocab_get_add_eos(vocab_)" in SERVER
    assert "add_special, parse_special" in SERVER
