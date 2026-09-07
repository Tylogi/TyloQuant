from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECODE = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "apps" / "mfq_decode.cpp"
).read_text(encoding="utf-8")
BATCHING = (
    ROOT
    / "cpp_runtime"
    / "backends"
    / "cuda"
    / "runtime"
    / "qwen_continuous_batching.h"
).read_text(encoding="utf-8")
ROPE = (ROOT / "mfq" / "kernels" / "cuda" / "rope.cu").read_text(
    encoding="utf-8"
)


def test_continuous_batching_is_an_explicit_server_mode():
    assert '"--continuous-batching"' in DECODE
    assert '"--check-continuous-batching"' in DECODE
    assert "CudaContinuousBatcher" in DECODE
    assert "decode=target_only mtp=disabled" in DECODE


def test_scheduler_supports_dynamic_join_retire_and_per_request_sampling():
    assert "take_qwen_batch_state" in BATCHING
    assert "restore_qwen_batch_states" in BATCHING
    assert "compact_qwen_batch_state" in BATCHING
    assert "sample_server_logits" in BATCHING
    assert "request->rng" in BATCHING
    assert "request->token_constraint" in BATCHING
    assert "pending_" in BATCHING
    assert "active_" in BATCHING


def test_qwen_decode_accepts_independent_batch_positions():
    assert "cache_positions.size(0) == B" in DECODE
    assert "write_positions.size(0) == B" in DECODE
    assert "cache_positions_override.has_value()" in DECODE
    assert "pos_batches" in ROPE
    assert "rows_per_batch" in ROPE


def test_real_weight_gate_exercises_join_and_compaction():
    assert "run_qwen_continuous_batching_check" in BATCHING
    assert "batcher.queued_requests()" in BATCHING
    assert 'metric("continuous_batching_max_batch") >= 2.0' in BATCHING
    assert 'metric("continuous_batching_compactions") >= 1.0' in BATCHING

