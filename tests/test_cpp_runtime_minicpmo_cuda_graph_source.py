from pathlib import Path
import re


SOURCE = (Path(__file__).parents[1] / "cpp_runtime" / "mfq_decode.cpp").read_text(
    encoding="utf-8"
)
ATTENTION_SOURCE = (
    Path(__file__).parents[1] / "mfq" / "kernels" / "cuda" / "attention.cu"
).read_text(encoding="utf-8")
BACKEND_SOURCE = (
    Path(__file__).parents[1] / "cpp_runtime" / "cuda" / "mfq_tensor_backend.h"
).read_text(encoding="utf-8")
CONTEXT_SOURCE = (
    Path(__file__).parents[1] / "cpp_runtime" / "cuda" / "mfq_cuda_context.cu"
).read_text(encoding="utf-8")
NATIVE_OPS_SOURCE = (
    Path(__file__).parents[1] / "cpp_runtime" / "cuda" / "mfq_native_tensor_ops.cu"
).read_text(encoding="utf-8")
NATIVE_TENSOR_SOURCE = (
    Path(__file__).parents[1] / "cpp_runtime" / "cuda" / "mfq_native_tensor.cu"
).read_text(encoding="utf-8")


def test_minicpmo_native_server_keeps_cuda_graph_enabled() -> None:
    assert "graph_architecture_supported" not in SOURCE
    graph_gate = SOURCE.split(
        'const char * graph_env = std::getenv("MFQ_SERVER_CUDA_GRAPH");', 1
    )[1].split(
        'const char * graph_min_env = std::getenv(', 1
    )[0]
    assert "is_minicpmo45" not in graph_gate


def test_static_decode_uses_dynamic_position_for_kv_writes() -> None:
    assert SOURCE.count("hidden_forward(ids, pos, seq_len, nullptr, pos)") == 2
    assert "cache_positions_override.value(), primary" in SOURCE
    assert '"cache_positions must have shape [tokens]"' in SOURCE


def test_minicpmo_persistent_decode_workspaces_are_warmed_before_capture() -> None:
    warmup_gates = re.findall(
        r"model\.c\.is_glm_dsa\(\) \|\|\s+model\.c\.is_minicpmo45\(\)\) \{",
        SOURCE,
    )
    assert len(warmup_gates) == 2


def test_graph_stage_events_start_after_decode_workspace_warmup() -> None:
    graph_path = SOURCE.rsplit(
        "MfqCudaGraph graph;", 1
    )[1].split(
        "graph.capture_end();", 1
    )[0]
    warmup = graph_path.index("model.next_token_static(static_input, static_pos, static_len)")
    profiler_reset = graph_path.index("g_profiler.reset();")
    external_events = graph_path.index(
        "g_profiler.graph_events = profile_cuda_graph;"
    )
    capture = graph_path.index("graph.capture_begin();")
    assert warmup < profiler_reset < external_events < capture


def test_graph_profile_covers_model_and_commit_boundaries() -> None:
    graph_path = SOURCE.rsplit(
        "MfqCudaGraph graph;", 1
    )[1].split(
        "graph.capture_end();", 1
    )[0]
    assert 'g_profiler.measure("decode.model_total"' in graph_path
    assert 'g_profiler.measure("decode.commit"' in graph_path
    assert graph_path.index('g_profiler.measure("decode.model_total"') < graph_path.index(
        'g_profiler.measure("decode.commit"'
    )


def test_torch_reference_graph_can_emit_a_debug_dump() -> None:
    assert 'std::getenv("MFQ_TORCH_CUDA_GRAPH_DUMP")' in BACKEND_SOURCE
    assert "graph.enable_debug_mode();" in BACKEND_SOURCE
    assert "graph.debug_dump(debug_path);" in BACKEND_SOURCE
    assert "mfq_debug_dump_cuda_graph(graph);" in SOURCE


def test_backend_bf16_add_check_covers_eager_and_graph_paths() -> None:
    assert "run_backend_bf16_add_check" in SOURCE
    assert 'a == "--check-backend-bf16-add"' in SOURCE
    check = SOURCE.split("static int run_backend_bf16_add_check", 1)[1].split(
        "static int64_t g_decode_graph_attention_parts", 1
    )[0]
    assert "eager = (left + right).contiguous();" in check
    assert "graph.capture_begin();" in check
    assert "graph.replay();" in check
    assert '<< " max_abs=" << maximum' in check
    assert "constexpr int chain_nodes = 512;" in check
    assert "chain_graph.capture_begin();" in check
    assert '<< " chain_node_us="' in check
    assert "shared_result = silu_mul_cuda(left, right);" in check
    assert "shared_graph.capture_begin();" in check
    assert '<< " shared_chain_node_us="' in check


def test_native_bf16_argmax_uses_bounded_contiguous_last_dimension_path() -> None:
    assert "argmax_last_contiguous_bf16_kernel" in NATIVE_OPS_SOURCE
    selection = NATIVE_OPS_SOURCE.split("if (operation == 3 &&", 1)[1].split(
        "auto launch =", 1
    )[0]
    assert "source.scalar_type() == kBFloat16" in selection
    assert "source.is_contiguous()" in selection
    assert "outer <= std::numeric_limits<unsigned int>::max()" in selection
    assert "selected + 1 == static_cast<std::size_t>(source.dim())" in selection


def test_native_prefill_batched_matmul_preserves_per_matrix_gemm() -> None:
    selection = NATIVE_OPS_SOURCE.split(
        'std::getenv("MFQ_DISABLE_NATIVE_PARALLEL_BATCH_MATMUL")', 1
    )[1].split("\n    for (std::int64_t batch = 0; batch < batches; ++batch)", 1)[0]
    assert "rows >= 32" in NATIVE_OPS_SOURCE
    assert "cudaStreamCaptureStatusNone" in NATIVE_OPS_SOURCE
    assert "parallel.ready.record(stream);" in selection
    assert "worker.wait(parallel.ready);" in selection
    assert "cublasGemmEx(" in selection
    assert "cublasGemmStridedBatchedEx(" not in selection
    assert "cudaStreamWaitEvent(" in selection


def test_native_bf16_causal_scale_fusion_is_exactly_bounded() -> None:
    selection = NATIVE_OPS_SOURCE.split(
        'std::getenv("MFQ_DISABLE_NATIVE_FUSED_CAUSAL_SCALE")', 1
    )[1].split("if (fused_causal_scale)", 1)[0]
    assert "causal && !mask.has_value()" in selection
    assert "scores.scalar_type() == kBFloat16" in selection
    assert "scores.is_contiguous()" in selection
    assert "scores.dim() >= 2" in selection
    assert "scale_causal_bf16_kernel" in NATIVE_OPS_SOURCE
    assert "load_number(source, linear) * factor" in NATIVE_OPS_SOURCE


def test_native_bf16_softmax_preserves_serial_reduction_order() -> None:
    selection = NATIVE_OPS_SOURCE.split(
        'std::getenv("MFQ_DISABLE_NATIVE_EXACT_BF16_SOFTMAX")', 1
    )[1].split("auto working =", 1)[0]
    assert "input.scalar_type() == kBFloat16" in selection
    assert "input.is_contiguous()" in selection
    assert "selected + 1 == static_cast<std::size_t>(input.dim())" in selection
    assert "exact_bf16_softmax_max_kernel" in NATIVE_OPS_SOURCE
    assert "exact_bf16_softmax_numerator_kernel" in NATIVE_OPS_SOURCE
    assert "exact_bf16_softmax_sum_kernel" in NATIVE_OPS_SOURCE
    assert "exact_bf16_softmax_normalize_element_kernel" in NATIVE_OPS_SOURCE


def test_cuda_profiler_filter_supports_low_perturbation_eager_attribution() -> None:
    assert 'std::getenv("MFQ_PROFILE_CUDA_FILTER")' in SOURCE
    assert "if (!enabled || !selected(name)) return fn();" in SOURCE
    eager_path = SOURCE.rsplit("} else {", 1)[1].split(
        "mfq_cuda_synchronize();", 1
    )[0]
    assert 'g_profiler.measure("decode.eager_model"' in eager_path
    assert 'g_profiler.measure("decode.eager_commit"' in eager_path


def test_bf16_head_to_token_candidate_is_exactly_stride_bounded() -> None:
    assert "materialize_bf16_head_to_token_d128_kernel" in NATIVE_TENSOR_SOURCE
    assert "MFQ_NATIVE_BF16_HEAD_TO_TOKEN_CONTIGUOUS" not in NATIVE_TENSOR_SOURCE
    selection = NATIVE_TENSOR_SOURCE.split(
        "const bool head_to_token_layout =", 1
    )[1].split("if (head_to_token_layout)", 1)[0]
    assert "source.dim() == 4" in selection
    assert "destination.dim() == 4" in selection
    assert "destination.size(1) == tokens" in selection
    assert "destination.is_contiguous()" in selection
    assert "batch > 0 && tokens > 0" in selection
    assert "depth == 128" in selection
    assert "source.stride(1) == depth" in selection
    assert "source.stride(2) == tokens * depth" in selection
    assert "source.stride(0) == heads * tokens * depth" in selection
    assert "alignof(uint4)" in selection


def test_graph_attention_tracks_eager_split_count_from_device_length() -> None:
    assert "attention_cache_decode_dynamic_cuda" in SOURCE
    assert "g_decode_graph_attention_parts > 1" in SOURCE
    active_parts = ATTENTION_SOURCE.split(
        "__device__ __forceinline__ int attention_decode_active_parts", 1
    )[1].split("template <int BD, typename scalar_t>", 1)[0]
    assert "cache_position < 192" in active_parts
    assert "return 1;" in active_parts
    assert "attention_cache_decode_select_kernel" not in ATTENTION_SOURCE


def test_minicpmo_bf16_residual_norm_is_a_fused_cuda_operation() -> None:
    assert "acc_rms_norm_bf16_cuda" in SOURCE
    official_branch = SOURCE.split(
        "if (rr.scalar_type() == mfq_tensor_backend::kBFloat16", 1
    )[1].split("return acc_rms_norm_cuda", 1)[0]
    assert "return acc_rms_norm_bf16_cuda" in official_branch
