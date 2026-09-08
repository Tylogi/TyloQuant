from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "cpp_runtime" / "backends" / "cuda" / "apps" / "mfq_decode.cpp").read_text(
    encoding="utf-8"
)
CORE = (ROOT / "cpp_runtime" / "core" / "tensor_parallel.h").read_text(
    encoding="utf-8"
)
CMAKE = (ROOT / "cpp_runtime" / "tests" / "CMakeLists.txt").read_text(
    encoding="utf-8"
)
COMPONENTS = (
    ROOT
    / "cpp_runtime"
    / "backends"
    / "cuda"
    / "runtime"
    / "server_components.h"
).read_text(encoding="utf-8")


def test_tensor_parallel_cli_and_weighted_split_are_wired():
    assert '"--tensor-parallel"' in SOURCE
    assert '"--tensor-split"' in SOURCE
    assert "plan_tensor_parallel_slices" in SOURCE
    assert "cudaDeviceEnablePeerAccess" in SOURCE


def test_dense_ffn_keeps_intermediate_shards_local():
    assert "forward_tensor_parallel_dense" in SOURCE
    assert '"ffn.tensor_parallel"' in SOURCE
    assert "reduce_model_parallel_outputs" in SOURCE


def test_routed_moe_all_compact_families_are_expert_sharded():
    assert "to_cuda_device_moe_expert_slice" in SOURCE
    assert "select_nint_cpu_rows" in SOURCE
    assert "select_nint8_zero_cpu_rows" in SOURCE
    assert "select_nvq_cpu_rows" in SOURCE
    assert "select_nepq_cpu_rows" in SOURCE
    assert "partial_experts = true" in SOURCE
    assert "expert_parallel_shards" in SOURCE
    assert '"--check-ep-moe"' in SOURCE
    assert "run_expert_parallel_moe_check" in SOURCE


def test_expert_parallel_cli_and_hybrid_device_contract_are_wired():
    assert '"--expert-parallel"' in SOURCE
    assert '"--expert-split"' in SOURCE
    assert "g_expert_parallel" in SOURCE
    assert "plan_moe_expert_parallel_slices" in SOURCE
    assert "g_tensor_parallel.devices != g_expert_parallel.devices" in SOURCE


def test_tensor_parallel_has_a_native_partition_test_target():
    assert "mfq-tensor-parallel-test" in CMAKE
    assert "tensor_parallel_test.cpp" in CMAKE


def test_model_parallel_rejects_silent_moe_cache_bypass():
    assert (
        '"--moe-gpu-cache-gb cannot be combined with "'
        in SOURCE
    )
    assert "model_parallel_enabled()" in SOURCE
    assert '"tensor/expert parallelism"' in SOURCE


def test_tensor_parallel_graph_capture_registers_all_participant_streams():
    assert '"MFQ_MODEL_PARALLEL_CUDA_GRAPH"' in SOURCE
    assert '"MFQ_TP_CUDA_GRAPH"' in SOURCE
    assert '"MFQ_EP_CUDA_GRAPH"' in SOURCE
    assert "model_parallel_cuda_graph_enabled()" in SOURCE
    assert "environment == nullptr || environment[0] != '0'" in SOURCE
    assert "graph_participant_streams" in SOURCE
    assert "g_model_parallel_collectives.streams.begin()" in SOURCE
    assert "graph_cache.compute_streams" in SOURCE
    assert "participant_streams" in SOURCE
    assert "const bool graph_enabled" in SOURCE
    assert "bool use_cuda_graph" in SOURCE


def test_tensor_parallel_graph_primes_and_captures_nccl_peer_transfers():
    assert "p2p_warmup_buffers" in SOURCE
    assert "ncclSend(" in SOURCE
    assert "ncclRecv(" in SOURCE
    assert "runtime.streams[destination_rank].stream()" in SOURCE
    assert "runtime.ready[source_rank]" in SOURCE
    assert "cudaStreamSynchronize(\n                participant.stream())" in SOURCE
    assert "for (int pass = 0; pass < 2; ++pass)" in SOURCE


def test_expert_parallel_graph_uses_capture_safe_peer_transfers():
    assert "tensor_to_cuda_device(std::move(value), device)" in SOURCE
    assert (
        "destination = tensor_to_cuda_device(\n"
        "            source, device, std::move(destination));"
        in SOURCE
    )
    assert "auto destination = destination_tensor();" in SOURCE
    assert ".clone().to(pool.weight.q_packed.device()).contiguous()" in SOURCE


def test_two_rank_fp16_reduce_avoids_round_trip_casts():
    assert '"MFQ_MODEL_PARALLEL_FP16_REDUCE"' in SOURCE
    assert '"MFQ_TP_FP16_REDUCE"' in SOURCE
    assert "model_parallel_fp16_reduce_enabled()" in SOURCE
    assert "environment == nullptr || std::atoi(environment) != 0" in SOURCE
    assert "outputs.size() == 2" in SOURCE
    assert "output_dtype == mfq_tensor_backend::kFloat16" in SOURCE
    assert "fp16_reduce ? ncclFloat16 : ncclFloat32" in SOURCE
    assert "result = outputs[index]" in SOURCE


def test_tensor_parallel_projection_groups_share_each_rank_input_transfer():
    assert 'std::getenv(\n        "MFQ_TP_GROUPED_PROJECTIONS")' in SOURCE
    assert "environment == nullptr || std::atoi(environment) != 0" in SOURCE
    assert "tensor_parallel_output_compatible()" in SOURCE
    assert "forward_tensor_parallel_output_group(x)" in SOURCE
    assert "auto local_x = tensor_to_cuda_device(flat, device);" in SOURCE


def test_batched_decode_graph_orders_tp_groups_by_projection():
    assert "g_decode_graph_tp_projection_major" in SOURCE
    assert "DecodeGraphTpProjectionScope" in SOURCE
    assert "std::vector<mfq_tensor_backend::Tensor> local_inputs" in SOURCE
    assert "local_inputs[shard] = tensor_to_cuda_device(flat, device)" in SOURCE


def test_tensor_parallel_peer_first_launch_preserves_rank_indexing():
    assert '"MFQ_MODEL_PARALLEL_PEER_FIRST_LAUNCH"' in SOURCE
    assert '"MFQ_TP_PEER_FIRST_LAUNCH"' in SOURCE
    assert "model_parallel_launch_index(" in SOURCE
    assert "environment == nullptr || std::atoi(environment) != 0" in SOURCE
    assert "peer_first_parallel_launch_index" in SOURCE
    assert "launch_position < primary_rank" in CORE
    assert "local_outputs[index] =" in SOURCE
    assert "partials[index] = run_quant_linear_shard(" in SOURCE


def test_qwen35_mtp_accepts_dense_tensor_parallel_placement():
    assert "const bool supported_placement" in COMPONENTS
    assert "!g_layer_placement.enabled()" in COMPONENTS
    assert "!g_tensor_parallel.enabled()" not in COMPONENTS
    assert "dense GPU-resident Qwen blocks" in COMPONENTS


def test_quantized_tp_weights_and_workspaces_follow_the_shard_device():
    assert SOURCE.count("mfq_current_cuda_device()") >= 3
    assert "const auto workspace_options = q_packed.options();" in SOURCE
    assert "const auto workspace_options = indices_packed.options();" in SOURCE


def test_deepseek_v4_split_gate_up_uses_existing_moe_runtime_and_cache():
    assert 'p + "mlp.experts.gate.weight"' in SOURCE
    assert 'p + "mlp.experts.up.weight"' in SOURCE
    assert "has_split_gate != has_split_up" in SOURCE
    assert "moe_split_gate_up" in SOURCE
    assert 'true, i, "gate"' in SOURCE
    assert 'true, i, "up"' in SOURCE
    assert '"moe.gate_up_split"' in SOURCE
    assert "mfq_tensor_backend::cat({gate, up}, -1).contiguous()" in SOURCE


def test_native_float_linears_are_supported_without_forcing_tp_shards():
    assert "QuantLinearKind::Dense" in SOURCE
    assert 'dtype == "BF16" || dtype == "F16" || dtype == "F32"' in SOURCE
    assert 'std::getenv("MFQ_TP_SHARD_NATIVE_FLOAT")' in SOURCE
    assert (
        "result.dense = cpu.to(mfq_tensor_backend::kCUDA).contiguous()" in SOURCE
    )
