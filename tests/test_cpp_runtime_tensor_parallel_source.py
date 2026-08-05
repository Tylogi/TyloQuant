from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(
    encoding="utf-8"
)
CMAKE = (ROOT / "cpp_runtime" / "CMakeLists.txt").read_text(
    encoding="utf-8"
)


def test_tensor_parallel_cli_and_weighted_split_are_wired():
    assert '"--tensor-parallel"' in SOURCE
    assert '"--tensor-split"' in SOURCE
    assert "plan_tensor_parallel_slices" in SOURCE
    assert "cudaDeviceEnablePeerAccess" in SOURCE


def test_dense_ffn_keeps_intermediate_shards_local():
    assert "forward_tensor_parallel_dense" in SOURCE
    assert '"ffn.tensor_parallel"' in SOURCE
    assert "reduce_tensor_parallel_outputs" in SOURCE


def test_routed_moe_all_compact_families_are_row_sharded():
    assert "to_cuda_device_moe_output_slice" in SOURCE
    assert "select_nint_cpu_rows" in SOURCE
    assert "select_nint8_zero_cpu_rows" in SOURCE
    assert "select_nvq_cpu_rows" in SOURCE
    assert "select_nepq_cpu_rows" in SOURCE
    assert "tensor_parallel_paired_output" in SOURCE
    assert '"--check-tp-moe"' in SOURCE
    assert "run_tensor_parallel_moe_check" in SOURCE


def test_tensor_parallel_has_a_native_partition_test_target():
    assert "mfq-tensor-parallel-test" in CMAKE
    assert "tensor_parallel_test.cpp" in CMAKE


def test_tensor_parallel_rejects_silent_moe_cache_bypass():
    assert (
        '"--moe-gpu-cache-gb cannot be combined with "'
        in SOURCE
    )
    assert '"--tensor-parallel"' in SOURCE


def test_tensor_parallel_disables_single_device_cuda_graphs():
    assert SOURCE.count("!g_tensor_parallel.enabled()") >= 2
    assert "const bool graph_enabled" in SOURCE
    assert "bool use_cuda_graph" in SOURCE


def test_deepseek_v4_split_gate_up_uses_existing_moe_runtime_and_cache():
    assert 'p + "ffn_gate_exps.weight"' in SOURCE
    assert 'p + "ffn_up_exps.weight"' in SOURCE
    assert "has_split_gate != has_split_up" in SOURCE
    assert "moe_split_gate_up" in SOURCE
    assert 'true, i, "gate"' in SOURCE
    assert 'true, i, "up"' in SOURCE
    assert '"moe.gate_up_split"' in SOURCE
    assert "torch::cat({gate, up}, -1).contiguous()" in SOURCE


def test_native_float_linears_are_supported_without_forcing_tp_shards():
    assert "QuantLinearKind::Dense" in SOURCE
    assert 'dtype == "BF16" || dtype == "F16" || dtype == "F32"' in SOURCE
    assert 'std::getenv("MFQ_TP_SHARD_NATIVE_FLOAT")' in SOURCE
    assert "result.dense = cpu.to(torch::kCUDA).contiguous()" in SOURCE
