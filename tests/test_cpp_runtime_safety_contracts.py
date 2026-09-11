"""Source contracts for runtime safety and boundary checks."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECODE = (ROOT / "cpp_runtime" / "backends" / "cuda" / "apps" / "mfq_decode.cpp").read_text(
    encoding="utf-8"
)
SERVER = (ROOT / "cpp_runtime" / "server" / "src" / "server.cpp").read_text(
    encoding="utf-8"
)
METAL_VQ = (ROOT / "cpp_runtime" / "backends" / "metal" / "ops" / "mlx_vq.cpp").read_text(
    encoding="utf-8"
)
NVQ2J_CUDA = (ROOT / "mfq" / "quantize" / "cuda" / "nvq2j_assign.cu").read_text(
    encoding="utf-8"
)
NVQ3J_CUDA = (ROOT / "mfq" / "quantize" / "cuda" / "nvq3j_assign.cu").read_text(
    encoding="utf-8"
)
UNIFIED_CUDA_EXT = (ROOT / "mfq" / "kernels" / "cuda" / "_ext.py").read_text(
    encoding="utf-8"
)


def test_single_source_moe_cache_holds_full_demand_set() -> None:
    start = DECODE.index("bool MoeExpertCache::prepare(")
    stop = DECODE.index("bool MoeExpertCache::prepare_bundle(", start)
    prepare = DECODE[start:stop]
    assert "arena_demands" in prepare
    assert "item.first->book->capacity()" in prepare
    assert "book->mark_inflight(slot)" in prepare
    assert "&held_slots" in prepare


def test_moe_cache_capacity_failure_uses_full_projection_path() -> None:
    assert "if (!cache_->prepare(" in DECODE
    assert "count_full_projection_fallback" in DECODE
    assert "stage_cpu_nint_moe(cpu_)" in DECODE
    assert "stage_cpu_mixed_moe(cpu_)" in DECODE


def test_optional_predictor_experts_join_the_shared_moe_cache() -> None:
    load_model = DECODE[
        DECODE.index("static Model load_model(") : DECODE.index(
            '#include "deepseek_v41/deepseek_v41_dspark.inc"'
        )
    ]
    main = DECODE[DECODE.index("int main(int argc, char ** argv)") :]
    assert "bool defer_moe_cache_finalize = false" in load_model
    assert "!defer_moe_cache_finalize" in load_model
    assert "const bool load_optional_components" in main
    assert main.index("load_cuda_runtime_components(") < main.index(
        "g_moe_expert_cache->finalize();"
    )


def test_reload_and_request_registration_share_one_gate() -> None:
    assert "std::mutex reload_gate;" in SERVER
    assert SERVER.count("std::lock_guard<std::mutex> gate(reload_gate);") >= 2
    assert "std::make_shared<ActiveRequest>(server_metrics)" in SERVER
    assert "active_request->complete(" in SERVER


def test_metal_jsc_rejects_partial_code_vectors() -> None:
    assert "jsc && header.input_size % profile.vector_size != 0" in METAL_VQ


def test_cuda_jsc_direct_entry_validates_balanced_bank_mapping() -> None:
    assert "bank_counts[bank] == kStatesPerBank" in NVQ2J_CUDA
    assert "bank_counts[bank] == kStatesPerBank" in NVQ3J_CUDA


def test_cuda_kl_rejects_context_larger_than_model_capacity() -> None:
    assert DECODE.count("KL reference exceeds model context capacity") >= 2


def test_cuda_nint_loader_expands_v2_metadata_before_kernel_dispatch() -> None:
    start = DECODE.index("static NintCpu unpack_nint(")
    stop = DECODE.index("struct Nint8ZeroCpu", start)
    loader = DECODE[start:stop]
    assert "const bool is_nint_v2 = (raw_bits & 0x80) != 0;" in loader
    assert "t.sub_bits = blob[off++];" in loader
    assert "if (is_nint_v2)" in loader
    assert "const auto selectors = unpack_bits(" in loader
    assert "t.sub_scale.resize(sub_count);" in loader
    assert "t.sub_min.resize(sub_count);" in loader
    assert 'throw std::runtime_error("invalid NINT trailing bytes")' in loader


def test_unified_cuda_extension_can_include_runtime_headers() -> None:
    for include in (
        "_REPOSITORY_ROOT",
        "_CUDA_RUNTIME_INCLUDE",
        "_GGML_INCLUDE",
        "_GGML_SOURCE_INCLUDE",
        "_GGML_CUDA_INCLUDE",
    ):
        assert include in UNIFIED_CUDA_EXT
    assert "extra_include_paths=[" in UNIFIED_CUDA_EXT
    assert '"--extended-lambda"' in UNIFIED_CUDA_EXT
    assert '"-U__CUDA_NO_HALF_CONVERSIONS__"' in UNIFIED_CUDA_EXT
