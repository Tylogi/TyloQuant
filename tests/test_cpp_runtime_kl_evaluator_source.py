from pathlib import Path


SOURCE = (
    Path(__file__).parents[1] / "cpp_runtime" / "mfq_decode.cpp"
).read_text(encoding="utf-8")
NINT_MATMUL_CUDA_SOURCE = (
    Path(__file__).parents[1] / "mfq" / "kernels" / "cuda" / "nint_matmul.cu"
).read_text(encoding="utf-8")


def test_kl_evaluator_cli_defaults_to_full_optimized_and_records_execution():
    assert "enum class KlEvaluator" in SOURCE
    assert 'std::string kl_evaluator_arg = "optimized";' in SOURCE
    assert "int kl_chunks = -1;" in SOURCE
    assert 'a == "--kl-evaluator"' in SOURCE
    assert 'a == "--kl-n-batch"' in SOURCE
    assert 'a == "--kl-score-count"' in SOURCE
    assert "parse_kl_evaluator(kl_evaluator_arg)" in SOURCE
    assert '" execution="' in SOURCE
    assert "same_top_count=" in SOURCE


def test_optimized_kl_preserves_transformer_shape_and_selects_lm_head_rows():
    assert "model.hidden_forward(ids)" in SOURCE
    assert "selected_hidden" in SOURCE
    assert "model.logits_from_hidden(selected_hidden)" in SOURCE
    assert "optimized_kld_sum" in SOURCE
    assert "optimized_same_top" in SOURCE


def test_optimized_kl_derives_sequence_count_from_explicit_batch_geometry():
    assert "run_kl_eval_batched" in SOURCE
    assert "constexpr int LLAMA_KL_N_SEQ = 4;" not in SOURCE
    assert "constexpr int LLAMA_KL_N_BATCH = 2048;" not in SOURCE
    assert "requested_n_batch == 0" in SOURCE
    assert "n_batch / n_ctx" in SOURCE
    assert "streamed_kl_ids(" in SOURCE
    assert "input, begin, n_seq_batch, n_ctx" in SOURCE
    assert "model.hidden_forward(ids)" in SOURCE
    assert "model.logits_from_hidden(selected_hidden)" in SOURCE
    assert '" n_batch=" << n_batch' in SOURCE


def test_optimized_kl_does_not_silently_force_nint8_mmq():
    assert "g_force_nint8_prefill_mmq" not in SOURCE
    assert 'std::getenv("MFQ_NINT8_PREFILL_MMQ")' in SOURCE
    assert (
        'TORCH_CHECK(M >= 1 && M <= 4096, "NINT8 MMQ supports M in [1, 4096]")'
        in NINT_MATMUL_CUDA_SOURCE
    )


def test_kld_only_mmq_selector_is_scoped_and_defaults_off():
    assert 'std::string kl_mmq_arg = "default";' in SOURCE
    assert 'a == "--kl-mmq"' in SOURCE
    assert "parse_kl_mmq_mode(kl_mmq_arg)" in SOURCE
    assert "struct KlMmqScope" in SOURCE
    assert "KlMmqMode::Nint8One" in SOURCE
    assert "KlMmqMode::Fp16" in SOURCE
    assert "g_kl_mmq_mode = previous_mode;" in SOURCE
    assert '" mmq="' in SOURCE
    assert '" fallback_calls="' in SOURCE


def test_nint6_int8_mmq_is_explicit_opt_in():
    assert 'std::string nint6_mmq_arg = "fp16";' in SOURCE
    assert 'a == "--nint6-mmq"' in SOURCE
    assert "parse_nint6_mmq_mode(nint6_mmq_arg)" in SOURCE
    assert "g_nint6_mmq_mode == Nint6MmqMode::Int8" in SOURCE
    assert '<< " nint6_mmq="' in SOURCE


def test_optimized_kl_uses_one_graph_independent_of_mmq_selection():
    dispatch_start = SOURCE.index(
        "if (!kl_base.empty()) {",
        SOURCE.index(
            "Model model = load_model(mfq_path, config_path, context_size);"
        ),
    )
    dispatch_end = SOURCE.index(
        "if (!prefill_sweep_sizes.empty())", dispatch_start
    )
    dispatch = SOURCE[dispatch_start:dispatch_end]
    assert "run_selected_kl_eval(" in dispatch
    assert (
        "evaluator == KlEvaluator::Optimized\n"
        "        ? run_kl_eval_batched"
    ) in SOURCE


def test_kld_contract_rejects_hidden_overrides_and_records_reference_geometry():
    assert "MFQ_KL_WINDOW_M is disabled" in SOURCE
    assert 'a == "--kl-allow-overlays"' in SOURCE
    assert "explicit --kl-allow-overlays" in SOURCE
    assert 'a == "--kl-reference-n-batch"' in SOURCE
    assert 'a == "--kl-reference-n-ubatch"' in SOURCE
    assert '" reference_n_batch="' in SOURCE
    assert '" reference_n_ubatch="' in SOURCE


def test_dual_kld_mmq_sequence_reuses_one_loaded_model():
    assert 'a == "--kl-mmq-sequence"' in SOURCE
    assert "parse_kl_mmq_sequence(kl_mmq_sequence_arg)" in SOURCE
    assert "--kl-mmq and --kl-mmq-sequence are mutually exclusive" in SOURCE
    assert "cpp_kl_mmq_sequence_begin" in SOURCE
    assert "KlMmqScope run_scope(kl_mmq_sequence[index]);" in SOURCE
    load_position = SOURCE.index(
        "Model model = load_model(mfq_path, config_path, context_size);"
    )
    sequence_position = SOURCE.index("cpp_kl_mmq_sequence_begin")
    assert load_position < sequence_position


def test_kld_progress_is_flushed_for_unattended_logs():
    assert "if (!kl_base.empty()) std::cout << std::unitbuf;" in SOURCE
