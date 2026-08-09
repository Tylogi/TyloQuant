"""Source contracts for runtime safety and boundary checks."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECODE = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(
    encoding="utf-8"
)
SERVER = (ROOT / "cpp_runtime" / "mfq_server.cpp").read_text(
    encoding="utf-8"
)
METAL_VQ = (ROOT / "cpp_runtime" / "metal" / "mlx_vq.cpp").read_text(
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


def test_unified_cuda_extension_can_include_runtime_headers() -> None:
    assert "_REPOSITORY_ROOT" in UNIFIED_CUDA_EXT
    assert "extra_include_paths=[_REPOSITORY_ROOT]" in UNIFIED_CUDA_EXT
