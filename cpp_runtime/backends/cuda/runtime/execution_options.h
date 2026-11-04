#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace mfq::cuda::runtime_options {
struct QsaKvOffloadOptions {
    std::size_t gpu_budget_bytes = 0;
    std::size_t ram_budget_bytes = 4ULL << 30;
    std::string directory;
};
QsaKvOffloadOptions qsa_kv_offload();
// Keep dynamic diagnostic/ablation options at the execution boundary. Values
// are read on access so changing an option before graph capture remains valid.
bool resident_plan_overlap();
std::optional<bool> rotary_fusion();
bool mhc_timings();
bool layer_timings();
int route_poll_interval_us();
bool single_window_flush();
bool early_no_cpu_ready();
bool skip_gpu_only_cpu_wait();
bool skip_fixed_gpu_timing();
bool window_input_views();
bool decode_host_pin();
bool kv_vram_preallocated();
bool vision_resident();
bool dma_graph_batch();
bool ram_field_packing();
bool warp_multi_sum();
float residency_heat_retention();
int residency_interval();
bool router_lookahead();
std::optional<bool> router_lookahead_override();
bool batch_attention_projections();
bool gdn_independent_batch();
bool gdn_independent_batch_trace();
bool qsa_independent_batch_trace();
bool batch_packed_prefill();
int batch_packed_prefill_token_budget();
bool nint_compressed_decode_batch();
bool nint_compressed_large_batch(bool vocabulary=false);
bool nint_large_output_tiling();
int continuous_batch_kv_initial_capacity();
std::size_t workspace_reserve_bytes();
std::size_t resident_budget_bytes();
bool phased_transfer();
bool mapped_copy_overlap();
bool background_calibration();
bool early_gate_up();
bool parallel_gate_up();
bool cpu_transfer_budget();
bool transfer_cache();
bool mapped_copy();
struct MoeDiagnostics {
    bool cpu_rows = false, cpu_rows_fail_alloc = false;
    bool dma = false, dma_fail_alloc = false, shared_cpu_cost = false;
    std::string dispatch_record, miss_record;
    std::optional<std::string> dispatch_replay;
};
MoeDiagnostics moe_diagnostics();
// Operator and storage controls are parsed only at this execution boundary.
std::optional<int> embedding_mapped();
std::optional<bool> gdn_inplace_state();
std::optional<bool> gdn_prefill_fused();
std::optional<bool> gr_dense_vector();
std::optional<bool> mfe_group_dot();
std::optional<bool> mfe_nint_late_scale();
std::optional<bool> mfe_nvq_word_signs();
std::optional<bool> mfe_shared_dense_math();
std::optional<bool> moe_assert_resident();
std::optional<bool> moe_nvq_dense();
std::optional<bool> moe_plan_reuse();
std::optional<bool> moe_plan_reuse_check();
std::optional<bool> moe_prefill_compact_mma();
std::optional<bool> moe_prefill_direct_ram();
std::optional<bool> moe_prefill_expert_batch();
std::optional<std::int64_t> moe_prefill_expert_batch_tokens();
std::optional<bool> moe_prefill_graph();
std::optional<bool> moe_prefill_groups();
std::optional<bool> moe_prefill_layer();
std::optional<bool> moe_prefill_layer_async();
std::optional<bool> moe_prefill_layer_flush();
std::optional<bool> moe_prefill_layer_major();
std::optional<bool> moe_prefill_layer_overlap();
std::optional<bool> moe_prefill_layer_phased();
std::optional<bool> moe_prefill_layer_reuse_async();
std::optional<bool> moe_prefill_pool_retain();
std::optional<bool> moe_preload_all();
bool nint_dense_reference();
std::optional<bool> nint_direct_pool();
std::optional<bool> nint_group_dot();
std::optional<bool> nint_route_hint();
std::optional<bool> nint_route_row_workspace();
std::optional<int> nint_route_warps();
std::optional<bool> nvq1_group_records();
std::optional<bool> nvq1_integer_delta();
bool nvq_dense_reference();
std::optional<bool> ple_file_map();
std::optional<bool> prefill_expert_batch_audit();
std::optional<bool> prefill_layer_major_trace();
std::optional<bool> qsa_prefill_fused();
bool qsa_pool_store_fused();
bool qsa_pool_incremental();
std::optional<bool> qsa_select_fused();
std::optional<bool> qsa_sparse_gate_fused();
std::optional<bool> shared_gate_fused();
std::optional<bool> trace_mfe_dispatch();
std::optional<bool> trace_mfe_dispatch_requested();
std::optional<bool> trace_moe_serve();
std::optional<bool> trace_ple_timings();
int moe_prefill_group_size();

// Decode and short-batch ablation controls; preserve unset defaults at call sites.
std::optional<bool> mfe_token_groups();
std::optional<bool> mfe_expert_plan();
std::optional<bool> mfe_token_compact_plan();
std::optional<bool> mfe_token_aligned_gu();
std::optional<bool> mfe_shared_small_whole();
std::optional<bool> mfe_nvq_bank_map();
std::optional<bool> nint_fused_fixed_bits();
std::optional<bool> nint_group_quantize();
std::optional<bool> ple_native_fusion();
std::optional<bool> gdn_strided_projections();
std::optional<bool> gdn_short_inplace();
std::optional<bool> gdn_short_fusion();
std::optional<bool> gdn_batch_projection();
std::optional<bool> nint_prefill_decode_reuse();
std::optional<bool> moe_prefill_append_tail();
std::optional<bool> gdn_prefill_layer_batch();
std::optional<bool> gr_prefill_layer_batch();
std::optional<bool> gr_chain_ffn_post();
std::optional<bool> mtp_predictor_graph();
std::optional<std::int64_t> continuous_ffn_tile();
std::optional<bool> gdn_packed_qkv();
std::optional<int> moe_prefill_shared_chain_streams();
std::optional<int> moe_prefill_shared_streams();
std::optional<int> gdn_prefill_projection_streams();
std::optional<bool> trace_gr_dispatch();
std::optional<bool> gr_batch_two_stage();
std::optional<bool> gr_continuous_two_stage();
std::optional<std::int64_t> gr_prefill_work_chunk();
std::optional<bool> position_buffer(const char* name);
std::optional<bool> single_output_view();
std::optional<bool> moe_direct_cache_dma();
std::optional<bool> moe_compact_host_stage();
std::optional<bool> moe_prefill_host_maps();
std::optional<char> gr_up_group_coefficients();
std::optional<bool> gr_dense_bf16_direct();
std::optional<int> mfe_token_group_capacity();
std::optional<std::int64_t> moe_ram_pin_bytes();
std::optional<std::int64_t> moe_ram_pin_reserve_bytes();
} // namespace mfq::cuda::runtime_options
