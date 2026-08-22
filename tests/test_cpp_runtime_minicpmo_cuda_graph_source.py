from pathlib import Path
import re


SOURCE = (Path(__file__).parents[1] / "cpp_runtime" / "mfq_decode.cpp").read_text(
    encoding="utf-8"
)
ATTENTION_SOURCE = (
    Path(__file__).parents[1] / "mfq" / "kernels" / "cuda" / "attention.cu"
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
