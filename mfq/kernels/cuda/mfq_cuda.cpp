// Binding for the unified mfq_cuda extension (mirrors llama.cpp: all ops compiled into one
// libggml-cuda, the dispatcher includes each op header). This file holds the forward
// declarations of every host launcher plus a single PYBIND module.
//
// Kernel bodies live in the matching .cu; this file is glue only.

#include <torch/extension.h>
#include <vector>
#include <optional>

std::vector<torch::Tensor> nint8_one_quantize_reconstruct_cuda(
    torch::Tensor x);

// norm.cu
torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor rms_norm_offset_cuda(torch::Tensor x, torch::Tensor weight, double eps, double weight_offset);
torch::Tensor rms_norm_f16_cuda(torch::Tensor x, torch::Tensor weight, double eps,
                                double weight_offset);
torch::Tensor l2_norm_cuda(torch::Tensor x, double eps);
// acc.cu
torch::Tensor acc_cuda(torch::Tensor a, torch::Tensor b);
std::vector<torch::Tensor> acc_rms_norm_cuda(torch::Tensor a, torch::Tensor b,
                                             torch::Tensor weight, double eps,
                                             double weight_offset);
std::vector<torch::Tensor> acc_rms_norm_f16_cuda(torch::Tensor a, torch::Tensor b,
                                                 torch::Tensor weight, double eps,
                                                 double weight_offset);
std::vector<torch::Tensor> gemma4_attn_residual_pre_norms_f16_cuda(
    torch::Tensor residual, torch::Tensor attn,
    torch::Tensor attn_post_weight, torch::Tensor dense_pre_weight,
    torch::Tensor router_weight, torch::Tensor moe_pre_weight, double eps);
torch::Tensor gemma4_ffn_merge_f16_cuda(
    torch::Tensor dense, torch::Tensor moe, torch::Tensor residual,
    torch::Tensor dense_post_weight, torch::Tensor moe_post_weight,
    torch::Tensor final_post_weight, torch::Tensor layer_scale, double eps);
void decode_graph_commit_cuda(torch::Tensor next, torch::Tensor generated, torch::Tensor step,
                              torch::Tensor input, torch::Tensor pos, torch::Tensor len);
// rope.cu
torch::Tensor rope_cuda(torch::Tensor x, torch::Tensor pos, double base);
torch::Tensor rope_ext_cuda(torch::Tensor x, torch::Tensor pos, double base, int64_t rotary_dim, torch::Tensor sections);
torch::Tensor rope_table_cuda(torch::Tensor x, torch::Tensor pos, torch::Tensor cos, torch::Tensor sin,
                              int64_t rotary_dim, torch::Tensor sections);
// attention.cu
torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal);
torch::Tensor attention_swa_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                 double scale, int64_t window);
torch::Tensor attention_cache_decode_cuda(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                          torch::Tensor seq_len, double scale);
torch::Tensor attention_cache_swa_cuda(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                       torch::Tensor seq_len, double scale, int64_t window);
// gated_delta_net.cu
std::vector<torch::Tensor> gdn_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> state);
std::vector<torch::Tensor> gdn_inplace_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state);
std::vector<torch::Tensor> gdn_inplace_transposed_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state);
// kv_cache.cu
std::vector<torch::Tensor> kv_cache_write_cuda(
    torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k, torch::Tensor v, torch::Tensor positions);
std::vector<torch::Tensor> kv_cache_write_ring_cuda(
    torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k, torch::Tensor v, int64_t position_start);
std::vector<torch::Tensor> kv_cache_write_ring_positions_cuda(
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k, torch::Tensor v, torch::Tensor positions);
// activation.cu
torch::Tensor silu_mul_cuda(torch::Tensor gate, torch::Tensor up);
torch::Tensor gelu_mul_cuda(torch::Tensor gate, torch::Tensor up);
// moe.cu
std::vector<torch::Tensor> moe_topk_cuda(
    torch::Tensor logits, int64_t top_k, bool use_sigmoid, bool use_sqrt_softplus, bool normalize,
    bool delayed_softmax, c10::optional<torch::Tensor> bias,
    double norm_floor, double scale);
torch::Tensor moe_sqrtsoftplus_weights_cuda(
    torch::Tensor logits, torch::Tensor ids, double norm_floor, double scale);
std::vector<torch::Tensor> moe_build_expert_map_cuda(
    torch::Tensor ids, int64_t n_experts, int64_t tile_m);
void nint_moe_quantize_input_ws_cuda(
    torch::Tensor x, int64_t gs, torch::Tensor qx, torch::Tensor xscale);
void nint_moe_quantize_24_28_ws_cuda(
    torch::Tensor x, torch::Tensor qx24, torch::Tensor xscale24,
    torch::Tensor qx28, torch::Tensor xscale28);
void nint_moe_quantize_swiglu_input_ws_cuda(
    torch::Tensor gate_up, int64_t gs, torch::Tensor qx, torch::Tensor xscale);
void nint_moe_quantize_geglu_input_ws_cuda(
    torch::Tensor gate_up, int64_t gs, torch::Tensor qx, torch::Tensor xscale);
void nint_moe_quantize_swiglu_24_28_ws_cuda(
    torch::Tensor gate_up, torch::Tensor qx24, torch::Tensor xscale24,
    torch::Tensor qx28, torch::Tensor xscale28);
torch::Tensor nint_moe_grouped_matmul_hetero_qx_cuda(
    torch::Tensor weight_ptrs, torch::Tensor pool_params, torch::Tensor activation_ptrs,
    torch::Tensor expert_pool, torch::Tensor expert_local, torch::Tensor ids,
    int64_t profile_mask, int64_t n_experts, int64_t out_per_expert,
    int64_t input_width, bool routed_input, torch::Tensor out,
    torch::Tensor ids_dst, torch::Tensor expert_bounds, torch::Tensor tile_bounds,
    torch::Tensor tile_experts);
torch::Tensor nint_moe_grouped_matmul_hetero_f16_cuda(
    torch::Tensor weight_ptrs, torch::Tensor pool_params,
    torch::Tensor expert_pool, torch::Tensor expert_local,
    torch::Tensor x, torch::Tensor ids,
    int64_t n_experts, int64_t out_per_expert, int64_t input_width,
    bool routed_input, torch::Tensor out,
    torch::Tensor ids_dst, torch::Tensor expert_bounds, torch::Tensor tile_bounds,
    torch::Tensor tile_experts);
torch::Tensor nint_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min,
    torch::Tensor x, torch::Tensor ids, torch::Tensor expert_local,
    int64_t n_experts, int64_t n_local_experts, int64_t out_per_expert,
    int64_t gs, int64_t bits, bool route_map_ready, bool input_quantized,
    torch::Tensor out, torch::Tensor qx, torch::Tensor xscale,
    torch::Tensor counts, torch::Tensor cursors, torch::Tensor ids_dst,
    torch::Tensor expert_bounds, torch::Tensor tile_bounds, torch::Tensor tile_experts);
torch::Tensor nint8_zero_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor x,
    torch::Tensor ids, torch::Tensor expert_local, int64_t n_experts,
    int64_t n_local_experts, int64_t out_per_expert, bool route_map_ready,
    bool input_quantized, bool use_f16_mma, torch::Tensor out,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor counts,
    torch::Tensor cursors, torch::Tensor ids_dst,
    torch::Tensor expert_bounds, torch::Tensor tile_bounds,
    torch::Tensor tile_experts);
torch::Tensor moe_weighted_reduce_cuda(torch::Tensor pair_output, torch::Tensor weights);
torch::Tensor moe_swiglu_split_cuda(torch::Tensor gate_up);
torch::Tensor moe_geglu_split_cuda(torch::Tensor gate_up);
torch::Tensor moe_apply_expert_scale_cuda(
    torch::Tensor weights, torch::Tensor ids, torch::Tensor scales);
torch::Tensor moe_add_shared_gate_cuda(
    torch::Tensor routed, torch::Tensor shared, torch::Tensor gate_logits);
torch::Tensor moe_weighted_reduce_shared_gate_cuda(
    torch::Tensor pair_output, torch::Tensor weights,
    torch::Tensor shared, torch::Tensor gate_logits);
// embedding.cu
torch::Tensor embedding_lookup_cuda(torch::Tensor weight, torch::Tensor token_ids);
torch::Tensor nint_embedding_lookup_cuda(
    torch::Tensor q, torch::Tensor d_eff, torch::Tensor m_eff,
    torch::Tensor token_ids, int64_t neuron_len, int64_t gs);
torch::Tensor nint_embedding_lookup_packed_eff_cuda(
    torch::Tensor q_packed, torch::Tensor eff_pair,
    torch::Tensor token_ids, int64_t neuron_len, int64_t gs);
torch::Tensor nint_embedding_lookup_packed_compact_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min,
    torch::Tensor token_ids, int64_t neuron_len, int64_t gs);
torch::Tensor nint_embedding_lookup_packed_compact_bits_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min,
    torch::Tensor token_ids, int64_t neuron_len, int64_t gs, int64_t bits);
// sampling.cu
torch::Tensor sample_greedy_cuda(torch::Tensor logits);
torch::Tensor sample_softmax_cuda(torch::Tensor logits, torch::Tensor random, double temperature);
torch::Tensor sample_top_k_top_p_cuda(
    torch::Tensor logits, torch::Tensor random, double temperature, int64_t top_k, double top_p);
void sample_token_counts_add_cuda(torch::Tensor counts, torch::Tensor tokens);
torch::Tensor sample_apply_penalties_cuda(
    torch::Tensor logits, torch::Tensor counts,
    double presence_penalty, double frequency_penalty, double repetition_penalty);
// ssm_conv.cu
torch::Tensor ssm_conv_silu_cuda(torch::Tensor conv_input, torch::Tensor weight, torch::Tensor bias, int64_t n_tokens);
torch::Tensor ssm_conv_silu_decode_cuda(torch::Tensor state, torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
std::vector<torch::Tensor> linear_conv_qkv_decode_cuda(
    torch::Tensor state, torch::Tensor qk, torch::Tensor v, torch::Tensor weight, torch::Tensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
// nint_matmul.cu
// nint_matmul.cu (decode / small-batch path)
torch::Tensor nint_gemv_cuda(
    torch::Tensor q, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs);
torch::Tensor nint_gemv_packed_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_int6_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_qx_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_gate_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, torch::Tensor gate,
    int64_t gs, int64_t mode, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_swiglu_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_geglu_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_batch_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_bits_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs, int64_t bits,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_bits_qx_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, int64_t gs, int64_t bits,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_bits_gate_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, torch::Tensor gate,
    int64_t gs, int64_t bits, int64_t mode,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_bits_swiglu_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs, int64_t bits,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_bits_geglu_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs, int64_t bits,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
void nint_ffn_gate_up_swiglu_quant_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    int64_t gu_gs, int64_t gu_bits, int64_t down_gs,
    torch::Tensor gu_qx, torch::Tensor gu_xscale, torch::Tensor gu_xsum,
    torch::Tensor down_qx, torch::Tensor down_xscale, torch::Tensor down_xsum);
void nint_ffn_gate_up_geglu_quant_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    int64_t gu_gs, int64_t gu_bits, int64_t down_gs,
    torch::Tensor gu_qx, torch::Tensor gu_xscale, torch::Tensor gu_xsum,
    torch::Tensor down_qx, torch::Tensor down_xscale, torch::Tensor down_xsum);
torch::Tensor nint_gemv_packed_bits_argmax_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs, int64_t bits,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum,
    torch::Tensor block_vals, torch::Tensor block_idxs);
torch::Tensor nint5_gs28_q5_repack_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min);
torch::Tensor nint5_gs28_q5_dequant_cuda(
    torch::Tensor q_packed, torch::Tensor neuron_scale, torch::Tensor neuron_min,
    int64_t neuron_len);
torch::Tensor nint5_gs28_q5_gemv_ws_cuda(
    torch::Tensor q_packed, torch::Tensor neuron_scale, torch::Tensor neuron_min,
    torch::Tensor x,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint5_gs28_q5_argmax_ws_cuda(
    torch::Tensor q_packed, torch::Tensor neuron_scale, torch::Tensor neuron_min,
    torch::Tensor x,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum,
    torch::Tensor block_vals, torch::Tensor block_idxs);
torch::Tensor nint_gemv_packed_bits_m1_out_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs, int64_t bits,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum,
    torch::Tensor out);
torch::Tensor nint_gemv_packed_bits_linear_out_norm_gate_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min,
    torch::Tensor y, torch::Tensor z, torch::Tensor norm_weight,
    int64_t gs, int64_t bits, int64_t dv, double eps,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum, torch::Tensor rinv);
torch::Tensor nint_gemv_packed_u8_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_batch_eff_ws_cuda(
    torch::Tensor q_packed, torch::Tensor d_eff, torch::Tensor m_eff, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_batch_eff2_ws_cuda(
    torch::Tensor q_packed, torch::Tensor eff_pair, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_batch_eff2_gate_ws_cuda(
    torch::Tensor q_packed, torch::Tensor eff_pair, torch::Tensor x, torch::Tensor gate,
    int64_t gs, int64_t mode, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_dequant_wq_packed_cuda(
    torch::Tensor q_packed, torch::Tensor d_eff, int64_t neuron_len, int64_t gs);
torch::Tensor nint_cublas_gemm_nt_f16acc_cuda(torch::Tensor x, torch::Tensor w);
torch::Tensor nint_dequant_full_packed_cuda(
    torch::Tensor q_packed, torch::Tensor d_eff, torch::Tensor m_eff, int64_t neuron_len, int64_t gs);
torch::Tensor nint_dequant_full_packed_compact_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, int64_t neuron_len, int64_t gs);
torch::Tensor nint_dequant_full_packed_compact_bits_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, int64_t neuron_len, int64_t gs, int64_t bits);
torch::Tensor nint_dequant_full_packed_gs24_x2_cuda(
    torch::Tensor q_packed, torch::Tensor d_eff, torch::Tensor m_eff, int64_t neuron_len);
torch::Tensor nint_dequant_full_packed_gs24_x2h2_cuda(
    torch::Tensor q_packed, torch::Tensor eff_pair, int64_t neuron_len);
torch::Tensor nint_dequant_full_packed_h2_cuda(
    torch::Tensor q_packed, torch::Tensor eff_pair, int64_t neuron_len, int64_t gs);
// nint_matmul.cu (batch path, tiled dp4a, M=2..64)
torch::Tensor nint_mmq_cuda(
    torch::Tensor q, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs);
torch::Tensor nint_mmq_packed_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_mmq_gs24_group32_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    torch::Tensor qx_mmq, torch::Tensor xscale, torch::Tensor xsum,
    int64_t split_k, torch::Tensor partial);
torch::Tensor nint_mmq_gs24_f16_nint3_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x);
torch::Tensor nint_mmq_gs24_f16_nint4_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x);
torch::Tensor nint_mmq_gs24_f16_nint6_split4_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    torch::Tensor partial);
torch::Tensor nint_mmq_f16_packed_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    int64_t gs, int64_t bits);
torch::Tensor nint_mmq_f32_packed_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    int64_t gs, int64_t bits);
torch::Tensor nint8_zero_mmq_f16_packed_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor x,
    int64_t neuron_len);
torch::Tensor nint8_zero_mmq_f32_packed_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor x,
    int64_t neuron_len);
torch::Tensor nint_mmq_packed_u8_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_mmq_packed_exec_ws_cuda(
    torch::Tensor q_mmq_packed, torch::Tensor sub_scale_mmq, torch::Tensor sub_min_mmq,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    int64_t ng, int64_t gs, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint8_zero_gemv_ws_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor x,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nint8_zero_mmq_ws_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor x,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nint8_zero_dequant_cuda(
    torch::Tensor q, torch::Tensor scale, int64_t neuron_len);
torch::Tensor nint8_zero_embedding_lookup_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor token_ids,
    int64_t neuron_len);
// mx_matmul.cu
torch::Tensor mxfp8_dequant_cuda(
    torch::Tensor values, torch::Tensor scales);
torch::Tensor mxfp8_embedding_lookup_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor token_ids);
torch::Tensor mxfp8_small_m_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor x);
torch::Tensor mxfp8_matmul_f16_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor input);
torch::Tensor mxfp8_gemm_f32_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor input);
torch::Tensor mxfp4_dequant_cuda(
    torch::Tensor values, torch::Tensor scales);
torch::Tensor mxfp4_embedding_lookup_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor token_ids);
torch::Tensor mxfp4_matmul_f16_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor input);
torch::Tensor mxfp4_moe_grouped_matmul_pool_f16_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor input,
    torch::Tensor ids, torch::Tensor expert_local,
    int64_t global_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len,
    torch::Tensor output, torch::Tensor ids_dst,
    torch::Tensor expert_bounds, torch::Tensor tile_bounds,
    torch::Tensor tile_experts);
// tpq_matmul.cu
torch::Tensor tpq_int4_matmul_f16_cuda(
    torch::Tensor packed, torch::Tensor scales,
    torch::Tensor input, int64_t group_size);
torch::Tensor tpq_int4_dequant_cuda(
    torch::Tensor packed, torch::Tensor scales, int64_t group_size);
torch::Tensor tpq_int4_embedding_lookup_cuda(
    torch::Tensor packed, torch::Tensor scales,
    torch::Tensor token_ids, int64_t group_size);
torch::Tensor tpq_pq_matmul_f16_cuda(
    torch::Tensor indices, torch::Tensor codebook, torch::Tensor input,
    int64_t outputs, int64_t width,
    int64_t vector_size, int64_t index_bits);
torch::Tensor tpq_pq_dequant_cuda(
    torch::Tensor indices, torch::Tensor codebook,
    int64_t outputs, int64_t width,
    int64_t vector_size, int64_t index_bits);
torch::Tensor tpq_pq_embedding_lookup_cuda(
    torch::Tensor indices, torch::Tensor codebook, torch::Tensor token_ids,
    int64_t outputs, int64_t width,
    int64_t vector_size, int64_t index_bits);
torch::Tensor tpq_pq_moe_grouped_matmul_pool_f16_cuda(
    torch::Tensor indices, torch::Tensor codebook,
    torch::Tensor input, torch::Tensor ids,
    torch::Tensor expert_local, int64_t global_experts,
    int64_t pool_experts, int64_t out_per_expert,
    int64_t width, int64_t vector_size, int64_t index_bits,
    torch::Tensor output, torch::Tensor ids_dst,
    torch::Tensor expert_bounds, torch::Tensor tile_bounds,
    torch::Tensor tile_experts);
// nvq_matmul.cu
// nepq.cu
torch::Tensor nepq_hadamard_input_cuda(
    torch::Tensor input, torch::Tensor signs, int64_t block_size);
torch::Tensor nepq_sparse_residual_matmul_cuda(
    torch::Tensor dictionary, torch::Tensor first, torch::Tensor second,
    torch::Tensor input, int64_t position_bits, int64_t block_vectors,
    torch::Tensor output);
torch::Tensor nepq_sparse_residual_dequant_cuda(
    torch::Tensor dictionary, torch::Tensor first, torch::Tensor second,
    int64_t position_bits, int64_t block_vectors, torch::Tensor weight);
torch::Tensor nepq_sparse_residual_grouped_cuda(
    torch::Tensor dictionary, torch::Tensor first, torch::Tensor second,
    torch::Tensor input, torch::Tensor route_ids, torch::Tensor expert_local,
    int64_t out_per_expert, int64_t position_bits, int64_t block_vectors,
    torch::Tensor output);
torch::Tensor nepq_dequant_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool, torch::Tensor bank_ids,
    int64_t neuron_len, int64_t sub_bits, int64_t format);
torch::Tensor nepq_gemv_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool, torch::Tensor bank_ids,
    torch::Tensor x, int64_t neuron_len, int64_t sub_bits, int64_t format,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nepq_mmq_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool, torch::Tensor bank_ids,
    torch::Tensor x, int64_t neuron_len, int64_t sub_bits, int64_t format,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nepq_gemm_f16_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool, torch::Tensor bank_ids,
    torch::Tensor x, int64_t neuron_len, int64_t sub_bits, int64_t format);
torch::Tensor nepq_moe_grouped_matmul_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool, torch::Tensor bank_ids,
    torch::Tensor grouped_table_pool, torch::Tensor x, torch::Tensor ids,
    int64_t n_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t sub_bits, int64_t format,
    torch::Tensor out, torch::Tensor qx, torch::Tensor xscale,
    torch::Tensor ids_dst, torch::Tensor expert_bounds, torch::Tensor tile_bounds,
    torch::Tensor tile_experts);
torch::Tensor nepq_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool, torch::Tensor bank_ids,
    torch::Tensor grouped_table_pool, torch::Tensor x, torch::Tensor ids,
    torch::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t sub_bits, int64_t format,
    bool input_quantized, torch::Tensor out, torch::Tensor qx,
    torch::Tensor xscale, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts);
torch::Tensor nvq_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x,
    torch::Tensor ids, torch::Tensor expert_local, int64_t n_experts,
    int64_t pool_experts, int64_t out_per_expert, int64_t neuron_len,
    int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    bool input_quantized, torch::Tensor out, torch::Tensor qx,
    torch::Tensor xscale, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts);
torch::Tensor nvq_dequant_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
torch::Tensor nvq_gemm_f16_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
torch::Tensor nvq_gemv_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nvq_gemv_qx_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nvq_gemv_multi2_ws_cuda(
    torch::Tensor first_indices, torch::Tensor first_aux, torch::Tensor first_sub_scale,
    torch::Tensor first_neuron_scale, torch::Tensor first_codebook,
    torch::Tensor second_indices, torch::Tensor second_aux, torch::Tensor second_sub_scale,
    torch::Tensor second_neuron_scale, torch::Tensor second_codebook,
    torch::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t first_sub_bits, int64_t first_format, int64_t first_sign_mode,
    int64_t second_sub_bits, int64_t second_format, int64_t second_sign_mode,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nvq_gemv_swiglu_ws_cuda(
    torch::Tensor gate_indices, torch::Tensor gate_aux, torch::Tensor gate_sub_scale,
    torch::Tensor gate_neuron_scale, torch::Tensor gate_codebook,
    torch::Tensor up_indices, torch::Tensor up_aux, torch::Tensor up_sub_scale,
    torch::Tensor up_neuron_scale, torch::Tensor up_codebook,
    torch::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nvq2_gemv_swiglu_vec4_ordered_ws_cuda(
    torch::Tensor gate_indices, torch::Tensor gate_aux, torch::Tensor gate_sub_scale,
    torch::Tensor gate_neuron_scale, torch::Tensor gate_codebook,
    torch::Tensor up_indices, torch::Tensor up_aux, torch::Tensor up_sub_scale,
    torch::Tensor up_neuron_scale, torch::Tensor up_codebook,
    torch::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode,
    torch::Tensor qx, torch::Tensor xscale);
void nvq_ffn_swiglu_quant_ws_cuda(
    torch::Tensor gate_indices, torch::Tensor gate_aux, torch::Tensor gate_sub_scale,
    torch::Tensor gate_neuron_scale, torch::Tensor gate_codebook,
    torch::Tensor up_indices, torch::Tensor up_aux, torch::Tensor up_sub_scale,
    torch::Tensor up_neuron_scale, torch::Tensor up_codebook,
    torch::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode, int64_t down_gs,
    torch::Tensor input_qx, torch::Tensor input_xscale,
    torch::Tensor output_qx, torch::Tensor output_xscale, torch::Tensor swiglu_scratch);
torch::Tensor nvq_gemv_m1_vec8_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t nwarps, torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nvq_gemv_batch_vec8_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t nwarps, torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nvq_gemv_gate_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x, torch::Tensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nvq_mmq_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nvq_mmq_gate_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x, torch::Tensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nvq_embedding_lookup_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor token_ids,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nint8_one_quantize_reconstruct_cuda",
          &nint8_one_quantize_reconstruct_cuda,
          "Q8_1-compatible activation quantize/reconstruct (CUDA)");
    m.def("rms_norm_cuda", &rms_norm_cuda, "RMSNorm (CUDA)");
    m.def("rms_norm_offset_cuda", &rms_norm_offset_cuda, "RMSNorm with weight offset (CUDA)");
    m.def("rms_norm_f16_cuda", &rms_norm_f16_cuda,
          "F16 RMSNorm with FP32 accumulation (CUDA)");
    m.def("l2_norm_cuda", &l2_norm_cuda, "L2 Norm (CUDA)");
    m.def("acc_cuda", &acc_cuda, "residual add (CUDA)");
    m.def("acc_rms_norm_cuda", &acc_rms_norm_cuda, "residual add + RMSNorm (CUDA)");
    m.def("acc_rms_norm_f16_cuda", &acc_rms_norm_f16_cuda, "residual add + f16 RMSNorm (CUDA)");
    m.def("gemma4_attn_residual_pre_norms_f16_cuda",
          &gemma4_attn_residual_pre_norms_f16_cuda,
          "Gemma4 attention residual and shared-input pre norms (CUDA)");
    m.def("gemma4_ffn_merge_f16_cuda", &gemma4_ffn_merge_f16_cuda,
          "Gemma4 branch post norms, merge, residual, and layer scale (CUDA)");
    m.def("decode_graph_commit_cuda", &decode_graph_commit_cuda, "CUDA graph decode state commit (CUDA)");
    m.def("rope_cuda", &rope_cuda, "RoPE rotate-half (CUDA)");
    m.def("rope_ext_cuda", &rope_ext_cuda, "RoPE rotate-half partial/MRoPE (CUDA)");
    m.def("rope_table_cuda", &rope_table_cuda, "RoPE rotate-half with precomputed cos/sin table (CUDA)");
    m.def("attention_cuda", &attention_cuda, "full/GQA attention, online softmax (CUDA)");
    m.def("attention_swa_cuda", &attention_swa_cuda, "causal sliding-window GQA attention (CUDA)");
    m.def("attention_cache_decode_cuda", &attention_cache_decode_cuda, "full/GQA decode attention over fixed KV cache (CUDA)");
    m.def("attention_cache_swa_cuda", &attention_cache_swa_cuda, "causal sliding-window attention over circular KV cache (CUDA)");
    m.def("gdn_cuda", &gdn_cuda, "Gated DeltaNet forward (CUDA)");
    m.def("gdn_inplace_cuda", &gdn_inplace_cuda, "Gated DeltaNet forward with in-place state update (CUDA)");
    m.def("gdn_inplace_transposed_cuda", &gdn_inplace_transposed_cuda,
          "Gated DeltaNet forward with transposed in-place state (CUDA)");
    m.def("kv_cache_write_cuda", &kv_cache_write_cuda, "KV cache write [B,H,T,D] -> [B,H,max_seq,D] (CUDA)");
    m.def("kv_cache_write_ring_cuda", &kv_cache_write_ring_cuda, "Contiguous append into circular KV cache (CUDA)");
    m.def("kv_cache_write_ring_positions_cuda", &kv_cache_write_ring_positions_cuda,
          "Position-indexed append into circular KV cache (CUDA)");
    m.def("silu_mul_cuda", &silu_mul_cuda, "SwiGLU silu(gate)*up (CUDA)");
    m.def("gelu_mul_cuda", &gelu_mul_cuda, "GeGLU gelu_tanh(gate)*up (CUDA)");
    m.def("moe_topk_cuda", &moe_topk_cuda, "Fused MoE routing transform + top-k (CUDA)");
    m.def("moe_sqrtsoftplus_weights_cuda", &moe_sqrtsoftplus_weights_cuda,
          "Gather and normalize sqrt-softplus hash-route weights (CUDA)");
    m.def("moe_build_expert_map_cuda", &moe_build_expert_map_cuda, "Build compact expert route map (CUDA)");
    m.def("nint_moe_quantize_input_ws_cuda", &nint_moe_quantize_input_ws_cuda,
          "Quantize one activation layout for heterogeneous expert NINT (CUDA)");
    m.def("nint_moe_quantize_24_28_ws_cuda", &nint_moe_quantize_24_28_ws_cuda,
          "Simultaneous gs24/gs28 activation quantization (CUDA)");
    m.def("nint_moe_quantize_swiglu_input_ws_cuda", &nint_moe_quantize_swiglu_input_ws_cuda,
          "Fused SwiGLU + quantize for heterogeneous expert NINT (CUDA)");
    m.def("nint_moe_quantize_geglu_input_ws_cuda", &nint_moe_quantize_geglu_input_ws_cuda,
          "Fused GeGLU + quantize for heterogeneous expert NINT (CUDA)");
    m.def("nint_moe_quantize_swiglu_24_28_ws_cuda", &nint_moe_quantize_swiglu_24_28_ws_cuda,
          "Fused SwiGLU + simultaneous gs24/gs28 quantization (CUDA)");
    m.def("nint_moe_grouped_matmul_hetero_qx_cuda", &nint_moe_grouped_matmul_hetero_qx_cuda,
          "Single-launch heterogeneous expert-wise NINT mul_mat_id (CUDA)");
    m.def("nint_moe_grouped_matmul_hetero_f16_cuda", &nint_moe_grouped_matmul_hetero_f16_cuda,
          "Heterogeneous expert-wise NINT grouped MMA prefill (CUDA)");
    m.def("nint_moe_grouped_matmul_pool_ws_cuda", &nint_moe_grouped_matmul_pool_ws_cuda,
          "Expert-wise NINT cohort mul_mat_id with caller workspace (CUDA)");
    m.def("nint8_zero_moe_grouped_matmul_pool_ws_cuda",
          &nint8_zero_moe_grouped_matmul_pool_ws_cuda,
          "Expert-wise NINT8-0 cohort mul_mat_id with caller workspace (CUDA)");
    m.def("moe_weighted_reduce_cuda", &moe_weighted_reduce_cuda, "MoE route-weighted reduction (CUDA)");
    m.def("moe_swiglu_split_cuda", &moe_swiglu_split_cuda, "MoE fused gate/up SwiGLU split (CUDA)");
    m.def("moe_geglu_split_cuda", &moe_geglu_split_cuda, "Fused gate/up GeGLU split (CUDA)");
    m.def("moe_apply_expert_scale_cuda", &moe_apply_expert_scale_cuda,
          "Apply per-expert scales to selected MoE route weights (CUDA)");
    m.def("moe_add_shared_gate_cuda", &moe_add_shared_gate_cuda,
          "MoE routed + sigmoid(shared gate) * shared output (CUDA)");
    m.def("moe_weighted_reduce_shared_gate_cuda", &moe_weighted_reduce_shared_gate_cuda,
          "Fused MoE route reduction + shared expert gate (CUDA)");
    m.def("embedding_lookup_cuda", &embedding_lookup_cuda, "Embedding row lookup (CUDA)");
    m.def("nint_embedding_lookup_cuda", &nint_embedding_lookup_cuda, "NINT selected-row embedding dequant (CUDA)");
    m.def("nint_embedding_lookup_packed_eff_cuda", &nint_embedding_lookup_packed_eff_cuda, "NINT selected-row embedding dequant, packed q + eff_pair (CUDA)");
    m.def("nint_embedding_lookup_packed_compact_cuda", &nint_embedding_lookup_packed_compact_cuda, "NINT selected-row embedding dequant, packed q + compact metadata (CUDA)");
    m.def("nint_embedding_lookup_packed_compact_bits_cuda", &nint_embedding_lookup_packed_compact_bits_cuda, "NINT selected-row embedding dequant, generic packed bits + compact metadata (CUDA)");
    m.def("sample_greedy_cuda", &sample_greedy_cuda, "Greedy logits sampling (CUDA)");
    m.def("sample_softmax_cuda", &sample_softmax_cuda, "Temperature softmax logits sampling (CUDA)");
    m.def("sample_top_k_top_p_cuda", &sample_top_k_top_p_cuda, "Top-k/top-p logits sampling (CUDA)");
    m.def("sample_token_counts_add_cuda", &sample_token_counts_add_cuda, "Accumulate token counts (CUDA)");
    m.def("sample_apply_penalties_cuda", &sample_apply_penalties_cuda, "Apply logits penalties in-place (CUDA)");
    m.def("ssm_conv_silu_cuda", &ssm_conv_silu_cuda, "Fused SSM depthwise conv + SiLU (CUDA)");
    m.def("ssm_conv_silu_decode_cuda", &ssm_conv_silu_decode_cuda, "Fused decode SSM depthwise conv + SiLU (CUDA)");
    m.def("linear_conv_qkv_decode_cuda", &linear_conv_qkv_decode_cuda, "Linear-attn decode conv + q/k L2 + repeat (CUDA)");
    m.def("nint_gemv_cuda", &nint_gemv_cuda, "NINT INT-GEMV, decode/small batch (CUDA)");
    m.def("nint_gemv_packed_ws_cuda", &nint_gemv_packed_ws_cuda, "NINT INT4-packed GEMV with caller workspace (CUDA)");
    m.def("nint_gemv_packed_int6_ws_cuda", &nint_gemv_packed_int6_ws_cuda, "NINT6 (6-bit) packed GEMV with caller workspace, requires 4|gs (CUDA)");
    m.def("nint_gemv_packed_qx_ws_cuda", &nint_gemv_packed_qx_ws_cuda, "NINT INT4-packed GEMV from prequantized activation workspace (CUDA)");
    m.def("nint_gemv_packed_gate_ws_cuda", &nint_gemv_packed_gate_ws_cuda, "NINT INT4-packed GEMV with fused input gate activation (CUDA)");
    m.def("nint_gemv_packed_swiglu_ws_cuda", &nint_gemv_packed_swiglu_ws_cuda, "NINT INT4-packed gate/up GEMV with fused SwiGLU output (CUDA)");
    m.def("nint_gemv_packed_geglu_ws_cuda", &nint_gemv_packed_geglu_ws_cuda, "NINT INT4-packed gate/up GEMV with fused GeGLU output (CUDA)");
    m.def("nint_gemv_packed_batch_ws_cuda", &nint_gemv_packed_batch_ws_cuda, "NINT INT4-packed batched GEMV/MMVQ with caller workspace (CUDA)");
    m.def("nint_gemv_packed_bits_ws_cuda", &nint_gemv_packed_bits_ws_cuda, "NINT generic packed-bits GEMV/MMVQ with caller workspace (CUDA)");
    m.def("nint_gemv_packed_bits_qx_ws_cuda", &nint_gemv_packed_bits_qx_ws_cuda, "NINT generic packed-bits GEMV from prequantized activation workspace (CUDA)");
    m.def("nint_gemv_packed_bits_gate_ws_cuda", &nint_gemv_packed_bits_gate_ws_cuda, "NINT generic packed-bits GEMV with fused input gate activation (CUDA)");
    m.def("nint_gemv_packed_bits_swiglu_ws_cuda", &nint_gemv_packed_bits_swiglu_ws_cuda, "NINT packed-bits gate/up GEMV with fused SwiGLU output (CUDA)");
    m.def("nint_gemv_packed_bits_geglu_ws_cuda", &nint_gemv_packed_bits_geglu_ws_cuda, "NINT packed-bits gate/up GEMV with fused GeGLU output (CUDA)");
    m.def("nint_ffn_gate_up_swiglu_quant_ws_cuda", &nint_ffn_gate_up_swiglu_quant_ws_cuda, "NINT gate/up GEMV + SwiGLU directly quantized for Wdown (CUDA)");
    m.def("nint_ffn_gate_up_geglu_quant_ws_cuda", &nint_ffn_gate_up_geglu_quant_ws_cuda, "NINT gate/up GEMV + GeGLU directly quantized for Wdown (CUDA)");
    m.def("nint_gemv_packed_bits_argmax_ws_cuda", &nint_gemv_packed_bits_argmax_ws_cuda, "NINT5/6 GEMV + greedy argmax (CUDA)");
    m.def("nint5_gs28_q5_repack_cuda", &nint5_gs28_q5_repack_cuda, "Repack NINT5 gs28 into low4/high1 execution layout (CUDA)");
    m.def("nint5_gs28_q5_dequant_cuda", &nint5_gs28_q5_dequant_cuda, "Dequantize NINT5 gs28 low4/high1 execution layout (CUDA)");
    m.def("nint5_gs28_q5_gemv_ws_cuda", &nint5_gs28_q5_gemv_ws_cuda, "NINT5 gs28 low4/high1 GEMV (CUDA)");
    m.def("nint5_gs28_q5_argmax_ws_cuda", &nint5_gs28_q5_argmax_ws_cuda, "NINT5 gs28 low4/high1 GEMV + argmax (CUDA)");
    m.def("nint_gemv_packed_bits_m1_out_ws_cuda", &nint_gemv_packed_bits_m1_out_ws_cuda, "NINT6 gs26 M=1 GEMV into caller output (CUDA)");
    m.def("nint_gemv_packed_bits_linear_out_norm_gate_ws_cuda", &nint_gemv_packed_bits_linear_out_norm_gate_ws_cuda, "NINT5 gs28 linear-attn out RMSNorm+gate+GEMV (CUDA)");
    m.def("nint_gemv_packed_u8_ws_cuda", &nint_gemv_packed_u8_ws_cuda, "NINT8 byte-packed dp4a GEMV/MMVQ with caller workspace (CUDA)");
    m.def("nint_gemv_packed_batch_eff_ws_cuda", &nint_gemv_packed_batch_eff_ws_cuda, "NINT INT4-packed batched GEMV/MMVQ with fused execution metadata (CUDA)");
    m.def("nint_gemv_packed_batch_eff2_ws_cuda", &nint_gemv_packed_batch_eff2_ws_cuda, "NINT INT4-packed batched GEMV/MMVQ with half2 fused execution metadata (CUDA)");
    m.def("nint_gemv_packed_batch_eff2_gate_ws_cuda", &nint_gemv_packed_batch_eff2_gate_ws_cuda, "NINT INT4-packed batched GEMV/MMVQ with fused input gate activation (CUDA)");
    m.def("nint_dequant_wq_packed_cuda", &nint_dequant_wq_packed_cuda, "NINT INT4-packed dequantize Wq for prefill GEMM (CUDA)");
    m.def("nint_cublas_gemm_nt_f16acc_cuda", &nint_cublas_gemm_nt_f16acc_cuda, "llama.cpp-style cuBLAS fp16-accumulate GEMM y=x*w.T (CUDA)");
    m.def("nint_dequant_full_packed_cuda", &nint_dequant_full_packed_cuda, "NINT INT4-packed full dequantize W for prefill GEMM (CUDA)");
    m.def("nint_dequant_full_packed_compact_cuda", &nint_dequant_full_packed_compact_cuda, "NINT INT4-packed full dequantize W from compact metadata (CUDA)");
    m.def("nint_dequant_full_packed_compact_bits_cuda", &nint_dequant_full_packed_compact_bits_cuda, "NINT generic packed-bits full dequantize W from compact metadata (CUDA)");
    m.def("nint_dequant_full_packed_gs24_x2_cuda", &nint_dequant_full_packed_gs24_x2_cuda, "NINT gs24 full dequantize W with 2 qbytes per thread (CUDA)");
    m.def("nint_dequant_full_packed_gs24_x2h2_cuda", &nint_dequant_full_packed_gs24_x2h2_cuda, "NINT gs24 half2 metadata full dequantize W with 2 qbytes per thread (CUDA)");
    m.def("nint_dequant_full_packed_h2_cuda", &nint_dequant_full_packed_h2_cuda, "NINT INT4-packed half2 full dequantize W for prefill GEMM (CUDA)");
    m.def("nint_mmq_cuda", &nint_mmq_cuda, "NINT tiled dp4a MMQ, batch (CUDA)");
    m.def("nint_mmq_packed_ws_cuda", &nint_mmq_packed_ws_cuda, "NINT INT4-packed tiled dp4a MMQ with caller workspace (CUDA)");
    m.def("nint_mmq_gs24_group32_ws_cuda", &nint_mmq_gs24_group32_ws_cuda, "NINT2 gs16 or NINT3/NINT4 gs24 group32 int8-MMA MMQ for M>=9 (CUDA)");
    m.def("nint_mmq_gs24_f16_nint3_cuda", &nint_mmq_gs24_f16_nint3_cuda, "Packed NINT2 gs16 / NINT3 gs24 fp16 Tensor-Core MMQ for M>=9 (CUDA)");
    m.def("nint_mmq_gs24_f16_nint4_cuda", &nint_mmq_gs24_f16_nint4_cuda, "Packed NINT4 gs24 fp16 Tensor-Core MMQ for M16..32 (CUDA)");
    m.def("nint_mmq_gs24_f16_nint6_split4_ws_cuda", &nint_mmq_gs24_f16_nint6_split4_ws_cuda, "Packed NINT6 gs24 fp16 Tensor-Core split-K=4 MMQ for M16..32 (CUDA)");
    m.def("nint_mmq_f16_packed_cuda", &nint_mmq_f16_packed_cuda,
          "Common packed NINT FP16 Tensor-Core MMQ with FP32 accumulation (CUDA)");
    m.def("nint_mmq_f32_packed_cuda", &nint_mmq_f32_packed_cuda,
          "Common packed NINT FP16 Tensor-Core MMQ with FP32 output (CUDA)");
    m.def("nint8_zero_mmq_f16_packed_cuda",
          &nint8_zero_mmq_f16_packed_cuda,
          "Common packed NINT8-0 FP16 Tensor-Core MMQ with FP32 accumulation (CUDA)");
    m.def("nint8_zero_mmq_f32_packed_cuda",
          &nint8_zero_mmq_f32_packed_cuda,
          "Common packed NINT8-0 FP16 Tensor-Core MMQ with FP32 output (CUDA)");
    m.def("nint_mmq_packed_u8_ws_cuda", &nint_mmq_packed_u8_ws_cuda, "NINT8 byte-packed tiled dp4a MMQ with caller workspace (CUDA)");
    m.def("nint_mmq_packed_exec_ws_cuda", &nint_mmq_packed_exec_ws_cuda, "NINT INT4-packed MMQ with MMQ execution-format weights (CUDA)");
    m.def("nint8_zero_gemv_ws_cuda", &nint8_zero_gemv_ws_cuda,
          "NINT8-0 packed GEMV with caller workspace (CUDA)");
    m.def("nint8_zero_mmq_ws_cuda", &nint8_zero_mmq_ws_cuda,
          "NINT8-0 packed MMQ with caller workspace (CUDA)");
    m.def("nint8_zero_dequant_cuda", &nint8_zero_dequant_cuda,
          "NINT8-0 full dequantization (CUDA)");
    m.def("nint8_zero_embedding_lookup_cuda",
          &nint8_zero_embedding_lookup_cuda,
          "NINT8-0 selected-row embedding decode (CUDA)");
    m.def("mxfp8_dequant_cuda", &mxfp8_dequant_cuda,
          "MXFP8 full dequantization (CUDA)");
    m.def("mxfp8_embedding_lookup_cuda", &mxfp8_embedding_lookup_cuda,
          "MXFP8 selected-row embedding decode (CUDA)");
    m.def("mxfp8_small_m_cuda", &mxfp8_small_m_cuda,
          "MXFP8 packed small-M matmul (CUDA)");
    m.def("mxfp8_matmul_f16_cuda", &mxfp8_matmul_f16_cuda,
          "MXFP8 packed matmul (CUDA)");
    m.def("mxfp8_gemm_f32_cuda", &mxfp8_gemm_f32_cuda,
          "MXFP8 packed FP32-output matmul (CUDA)");
    m.def("mxfp4_dequant_cuda", &mxfp4_dequant_cuda,
          "MXFP4 full dequantization (CUDA)");
    m.def("mxfp4_embedding_lookup_cuda", &mxfp4_embedding_lookup_cuda,
          "MXFP4 selected-row embedding decode (CUDA)");
    m.def("mxfp4_matmul_f16_cuda", &mxfp4_matmul_f16_cuda,
          "MXFP4 packed matmul (CUDA)");
    m.def("mxfp4_moe_grouped_matmul_pool_f16_cuda",
          &mxfp4_moe_grouped_matmul_pool_f16_cuda,
          "MXFP4 routed cohort matmul (CUDA)");
    m.def("tpq_int4_matmul_f16_cuda", &tpq_int4_matmul_f16_cuda,
          "TPQ symmetric int4 packed matmul (CUDA)");
    m.def("tpq_int4_dequant_cuda", &tpq_int4_dequant_cuda,
          "TPQ symmetric int4 dequantization (CUDA)");
    m.def("tpq_int4_embedding_lookup_cuda", &tpq_int4_embedding_lookup_cuda,
          "TPQ symmetric int4 selected-row embedding decode (CUDA)");
    m.def("tpq_pq_matmul_f16_cuda", &tpq_pq_matmul_f16_cuda,
          "TPQ learned product-VQ packed matmul (CUDA)");
    m.def("tpq_pq_dequant_cuda", &tpq_pq_dequant_cuda,
          "TPQ learned product-VQ dequantization (CUDA)");
    m.def("tpq_pq_embedding_lookup_cuda", &tpq_pq_embedding_lookup_cuda,
          "TPQ learned product-VQ selected-row embedding decode (CUDA)");
    m.def("tpq_pq_moe_grouped_matmul_pool_f16_cuda",
          &tpq_pq_moe_grouped_matmul_pool_f16_cuda,
          "TPQ-PQ routed cohort matmul (CUDA)");
    m.def("nepq_hadamard_input_cuda", &nepq_hadamard_input_cuda, "NEPQ signed block-Hadamard activation transform (CUDA)");
    m.def("nepq_sparse_residual_matmul_cuda", &nepq_sparse_residual_matmul_cuda, "NEPQ-A sparse residual matmul (CUDA)");
    m.def("nepq_sparse_residual_dequant_cuda", &nepq_sparse_residual_dequant_cuda, "NEPQ-A sparse residual dequantization (CUDA)");
    m.def("nepq_sparse_residual_grouped_cuda", &nepq_sparse_residual_grouped_cuda, "NEPQ-A sparse residual routed matmul (CUDA)");
    m.def("nepq_dequant_cuda", &nepq_dequant_cuda, "NEPQ shared-bank full dequant (CUDA)");
    m.def("nepq_gemv_ws_cuda", &nepq_gemv_ws_cuda, "NEPQ shared-bank q8 GEMV with caller workspace (CUDA)");
    m.def("nepq_mmq_ws_cuda", &nepq_mmq_ws_cuda, "NEPQ shared-bank gs24 int8 Tensor Core MMQ (CUDA)");
    m.def("nepq_gemm_f16_cuda", &nepq_gemm_f16_cuda, "NEPQ shared-bank online-dequant FP16 Tensor Core GEMM (CUDA)");
    m.def("nepq_moe_grouped_matmul_ws_cuda", &nepq_moe_grouped_matmul_ws_cuda, "NEPQ routed multi-warp/grouped expert matmul (CUDA)");
    m.def("nepq_moe_grouped_matmul_pool_ws_cuda", &nepq_moe_grouped_matmul_pool_ws_cuda, "NEPQ routed cohort matmul with global expert mapping (CUDA)");
    m.def("nvq_moe_grouped_matmul_pool_ws_cuda", &nvq_moe_grouped_matmul_pool_ws_cuda, "NVQ/NPQ routed cohort matmul with global expert mapping (CUDA)");
    m.def("nvq_dequant_cuda", &nvq_dequant_cuda, "Compact NPQ/NVQ full dequant (CUDA)");
    m.def("nvq_gemm_f16_cuda", &nvq_gemm_f16_cuda, "Compact NPQ/NVQ gs24 online-dequant FP16 Tensor Core GEMM for M16-M256 (CUDA)");
    m.def("nvq_gemv_ws_cuda", &nvq_gemv_ws_cuda, "Compact NPQ/NVQ q8 GEMV with caller workspace (CUDA)");
    m.def("nvq_gemv_qx_ws_cuda", &nvq_gemv_qx_ws_cuda, "Compact NPQ/NVQ GEMV from a prequantized activation (CUDA)");
    m.def("nvq_gemv_multi2_ws_cuda", &nvq_gemv_multi2_ws_cuda, "NVQ two-projection GEMV with shared activation quantization (CUDA)");
    m.def("nvq_gemv_swiglu_ws_cuda", &nvq_gemv_swiglu_ws_cuda, "NVQ paired gate/up GEMV with SwiGLU output (CUDA)");
    m.def("nvq2_gemv_swiglu_vec4_ordered_ws_cuda", &nvq2_gemv_swiglu_vec4_ordered_ws_cuda, "NVQ2 packed four-vector SwiGLU with ordered reduction (CUDA)");
    m.def("nvq_ffn_swiglu_quant_ws_cuda", &nvq_ffn_swiglu_quant_ws_cuda, "NVQ paired gate/up SwiGLU directly quantized for Wdown (CUDA)");
    m.def("nvq_gemv_m1_vec8_ws_cuda", &nvq_gemv_m1_vec8_ws_cuda, "Compact NPQ/NVQ llama-style vec8 M1 GEMV candidate (CUDA)");
    m.def("nvq_gemv_batch_vec8_ws_cuda", &nvq_gemv_batch_vec8_ws_cuda, "Compact NPQ/NVQ llama-style vec8 M2-M16 GEMV candidate (CUDA)");
    m.def("nvq_gemv_gate_ws_cuda", &nvq_gemv_gate_ws_cuda, "Compact NPQ/NVQ q8 GEMV with fused input gate (CUDA)");
    m.def("nvq_mmq_ws_cuda", &nvq_mmq_ws_cuda, "Compact NPQ/NVQ gs24 int8 Tensor Core MMQ with caller workspace (CUDA)");
    m.def("nvq_mmq_gate_ws_cuda", &nvq_mmq_gate_ws_cuda, "Compact NPQ/NVQ gs24 int8 Tensor Core MMQ with fused input gate (CUDA)");
    m.def("nvq_embedding_lookup_cuda", &nvq_embedding_lookup_cuda, "Compact NPQ/NVQ selected-row embedding decode (CUDA)");

    m.attr("niq_dequant_cuda") = m.attr("nvq_dequant_cuda");
    m.attr("niq_gemm_f16_cuda") = m.attr("nvq_gemm_f16_cuda");
    m.attr("niq_gemv_ws_cuda") = m.attr("nvq_gemv_ws_cuda");
    m.attr("niq_gemv_qx_ws_cuda") = m.attr("nvq_gemv_qx_ws_cuda");
    m.attr("niq_gemv_multi2_ws_cuda") = m.attr("nvq_gemv_multi2_ws_cuda");
    m.attr("niq_gemv_swiglu_ws_cuda") = m.attr("nvq_gemv_swiglu_ws_cuda");
    m.attr("niq2_gemv_swiglu_vec4_ordered_ws_cuda") = m.attr("nvq2_gemv_swiglu_vec4_ordered_ws_cuda");
    m.attr("niq_ffn_swiglu_quant_ws_cuda") = m.attr("nvq_ffn_swiglu_quant_ws_cuda");
    m.attr("niq_gemv_m1_vec8_ws_cuda") = m.attr("nvq_gemv_m1_vec8_ws_cuda");
    m.attr("niq_gemv_batch_vec8_ws_cuda") = m.attr("nvq_gemv_batch_vec8_ws_cuda");
    m.attr("niq_gemv_gate_ws_cuda") = m.attr("nvq_gemv_gate_ws_cuda");
    m.attr("niq_mmq_ws_cuda") = m.attr("nvq_mmq_ws_cuda");
    m.attr("niq_mmq_gate_ws_cuda") = m.attr("nvq_mmq_gate_ws_cuda");
    m.attr("niq_embedding_lookup_cuda") = m.attr("nvq_embedding_lookup_cuda");
}
