#include <torch/torch.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#ifdef MFQ_HAVE_NCCL
#include <nccl.h>
#endif

#include "mfq_server.h"
#include "moe_cache_policy.h"
#include "moe_cache_profile.h"
#include "tensor_parallel.h"
#include "nvq_codebooks.generated.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#define MFQ_CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
    } \
} while (0)

#ifdef MFQ_HAVE_NCCL
#define MFQ_NCCL_CHECK(expr) do { \
    ncclResult_t err__ = (expr); \
    if (err__ != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(err__)); \
    } \
} while (0)
#endif

static void mfq_set_env(const char * name, const char * value) {
#ifdef _WIN32
    if (_putenv_s(name, value ? value : "") != 0) {
        throw std::runtime_error(std::string("failed to update environment variable ") + name);
    }
#else
    const int status = value && value[0] != '\0'
        ? setenv(name, value, 1)
        : unsetenv(name);
    if (status != 0) {
        throw std::runtime_error(
            std::string("failed to update environment variable ") + name +
            ": " + std::strerror(errno));
    }
#endif
}

torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor rms_norm_offset_cuda(torch::Tensor x, torch::Tensor weight, double eps, double weight_offset);
torch::Tensor rms_norm_f16_cuda(torch::Tensor x, torch::Tensor weight, double eps,
                                double weight_offset);
torch::Tensor l2_norm_cuda(torch::Tensor x, double eps);
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
std::vector<torch::Tensor> linear_gate_beta_cuda(
    torch::Tensor alpha, torch::Tensor beta, torch::Tensor dt_bias, torch::Tensor a_log);
torch::Tensor rope_table_cuda(torch::Tensor x, torch::Tensor pos, torch::Tensor cos, torch::Tensor sin,
                              int64_t rotary_dim, torch::Tensor sections);
torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal);
torch::Tensor attention_swa_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                 double scale, int64_t window);
torch::Tensor attention_cache_swa_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t window);
torch::Tensor attention_cache_swa_planned_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t window, int64_t planned_length);
torch::Tensor attention_llama_flash256_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);
torch::Tensor attention_llama_flash512_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);
torch::Tensor attention_llama_flash256_swa_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, int64_t window);
torch::Tensor attention_llama_flash256_decode_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t planned_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta);
torch::Tensor attention_llama_flash512_decode_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t planned_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta);
torch::Tensor attention_glm_mla576_cuda(
    torch::Tensor q, torch::Tensor kv, double scale);
torch::Tensor attention_glm_mla576_cached_cuda(
    torch::Tensor q, torch::Tensor kv_cache, int64_t logical_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta,
    double scale);
torch::Tensor attention_glm_mla576_decode_cuda(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor seq_len,
    double scale, int64_t planned_len, torch::Tensor mask,
    torch::Tensor kv_max, torch::Tensor meta);
torch::Tensor glm_interleaved_rope_cuda(
    torch::Tensor x, torch::Tensor positions,
    torch::Tensor cos, torch::Tensor sin, int64_t rotary_dim);
torch::Tensor glm_dsa_indexer_layer_norm_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
torch::Tensor glm_dsa_cache_write_cuda(
    torch::Tensor cache, torch::Tensor values, torch::Tensor positions);
torch::Tensor glm_dsa_indexer_scores_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor weights,
    int64_t query_offset, int64_t logical_k);
torch::Tensor glm_dsa_indexer_scores_decode_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor weights,
    torch::Tensor seq_len, int64_t planned_k);
torch::Tensor attention_glm_mla_sparse_cuda(
    torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
    torch::Tensor meta, double scale);
torch::Tensor dsv4_compress_cuda(
    torch::Tensor kv, torch::Tensor gate, torch::Tensor ape,
    torch::Tensor norm, torch::Tensor prev_kv, torch::Tensor prev_gate,
    torch::Tensor positions, torch::Tensor cos, torch::Tensor sin,
    int64_t ratio, bool overlap, int64_t quant_mode, double eps);
torch::Tensor dsv4_fp4_sim_cuda(torch::Tensor input);
std::vector<torch::Tensor> dsv4_hc_pre_cuda(
    torch::Tensor x, torch::Tensor mixes, torch::Tensor scale,
    torch::Tensor base, int64_t iterations, double eps);
torch::Tensor dsv4_hc_post_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor post,
    torch::Tensor combination);
torch::Tensor dsv4_decode_pool_update_cuda(
    torch::Tensor kv_token, torch::Tensor gate_token,
    torch::Tensor ape, torch::Tensor norm,
    torch::Tensor state_kv, torch::Tensor state_gate,
    torch::Tensor prev_kv, torch::Tensor prev_gate,
    torch::Tensor pool, torch::Tensor seq_len,
    torch::Tensor cos, torch::Tensor sin,
    int64_t ratio, bool overlap, int64_t quant_mode, double eps);
torch::Tensor dsv4_indexer_scores_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor weights,
    int64_t query_offset, int64_t ratio);
torch::Tensor dsv4_topk512_cuda(torch::Tensor scores);
std::vector<torch::Tensor> dsv4_build_prefill_plan_cuda(
    torch::Tensor topk, int64_t query_offset, int64_t local_history,
    int64_t pool_len, int64_t ratio, int64_t window);
std::vector<torch::Tensor> dsv4_build_decode_plan_cuda(
    torch::Tensor topk, torch::Tensor seq_len, int64_t pool_len,
    int64_t ratio, int64_t window);
torch::Tensor attention_dsv4_sparse_cuda(
    torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
    torch::Tensor mask, torch::Tensor sinks, torch::Tensor meta,
    double scale);
torch::Tensor attention_llama_flash256_swa_decode_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t planned_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta);
torch::Tensor attention_cache_decode_cuda(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                          torch::Tensor seq_len, double scale);
torch::Tensor attention_cache_decode_split_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale,
    torch::Tensor partial_o, torch::Tensor partial_m, torch::Tensor partial_l,
    int64_t parts);
torch::Tensor silu_mul_cuda(torch::Tensor gate, torch::Tensor up);
torch::Tensor gelu_mul_cuda(torch::Tensor gate, torch::Tensor up);
std::vector<torch::Tensor> moe_topk_cuda(
    torch::Tensor logits, int64_t top_k, bool use_sigmoid, bool use_sqrt_softplus,
    bool normalize,
    bool delayed_softmax, c10::optional<torch::Tensor> bias, double norm_floor,
    double scale);
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
void nint_moe_quantize_swiglu_clamped_input_ws_cuda(
    torch::Tensor gate_up, int64_t gs, double limit,
    torch::Tensor qx, torch::Tensor xscale);
void nint_moe_quantize_swiglu_24_28_ws_cuda(
    torch::Tensor gate_up, torch::Tensor qx24, torch::Tensor xscale24,
    torch::Tensor qx28, torch::Tensor xscale28);
void nint_moe_quantize_geglu_input_ws_cuda(
    torch::Tensor gate_up, int64_t gs, torch::Tensor qx, torch::Tensor xscale);
void nint_moe_quantize_geglu_24_28_ws_cuda(
    torch::Tensor gate_up, torch::Tensor qx24, torch::Tensor xscale24,
    torch::Tensor qx28, torch::Tensor xscale28);
torch::Tensor nint_moe_grouped_matmul_hetero_qx_cuda(
    torch::Tensor weight_ptrs, torch::Tensor pool_params, torch::Tensor activation_ptrs,
    torch::Tensor expert_pool, torch::Tensor expert_local, torch::Tensor ids,
    int64_t profile_mask, int64_t n_experts, int64_t out_per_expert,
    int64_t input_width, bool routed_input, torch::Tensor out,
    torch::Tensor ids_dst, torch::Tensor expert_bounds, torch::Tensor tile_bounds,
    torch::Tensor tile_experts);
torch::Tensor nint_moe_grouped_matmul_hetero_glu_qx_cuda(
    torch::Tensor weight_ptrs, torch::Tensor pool_params, torch::Tensor activation_ptrs,
    torch::Tensor expert_pool, torch::Tensor expert_local, torch::Tensor ids,
    int64_t profile_mask, int64_t n_experts, int64_t hidden_width,
    bool gelu, torch::Tensor out);
torch::Tensor nint_moe_grouped_matmul_hetero_f16_cuda(
    torch::Tensor weight_ptrs, torch::Tensor pool_params,
    torch::Tensor expert_pool, torch::Tensor expert_local,
    torch::Tensor x, torch::Tensor ids,
    int64_t n_experts, int64_t out_per_expert, int64_t input_width,
    bool routed_input, torch::Tensor out,
    torch::Tensor ids_dst, torch::Tensor expert_bounds, torch::Tensor tile_bounds,
    torch::Tensor tile_experts);
torch::Tensor nint_moe_grouped_matmul_hetero_f16_slice_cuda(
    torch::Tensor weight_ptrs, torch::Tensor pool_params,
    torch::Tensor expert_pool, torch::Tensor expert_local,
    torch::Tensor x, torch::Tensor ids,
    int64_t n_experts, int64_t out_per_expert, int64_t input_width,
    bool routed_input, torch::Tensor out,
    torch::Tensor ids_dst, torch::Tensor expert_bounds, torch::Tensor tile_bounds,
    torch::Tensor tile_experts, int64_t weight_out_stride,
    int64_t weight_row_offset);
torch::Tensor nint_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    torch::Tensor ids, torch::Tensor expert_local, int64_t n_experts,
    int64_t n_local_experts, int64_t out_per_expert, int64_t gs, int64_t bits,
    bool route_map_ready, bool input_quantized, torch::Tensor out,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor counts,
    torch::Tensor cursors, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts);
torch::Tensor nint8_zero_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor x,
    torch::Tensor ids, torch::Tensor expert_local, int64_t n_experts,
    int64_t n_local_experts, int64_t out_per_expert, bool route_map_ready,
    bool input_quantized, bool use_f16_mma, torch::Tensor out, torch::Tensor qx,
    torch::Tensor xscale, torch::Tensor counts, torch::Tensor cursors,
    torch::Tensor ids_dst, torch::Tensor expert_bounds,
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
torch::Tensor nvq_moe_grouped_matmul_pool_f16_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x,
    torch::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t gs,
    int64_t sub_bits, int64_t format, int64_t sign_mode,
    torch::Tensor out, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts);
torch::Tensor nepq_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool, torch::Tensor bank_ids,
    torch::Tensor grouped_table_pool, torch::Tensor x, torch::Tensor ids,
    torch::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t sub_bits, int64_t format,
    bool input_quantized, torch::Tensor out, torch::Tensor qx,
    torch::Tensor xscale, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts);
torch::Tensor nepq_moe_grouped_matmul_pool_f16_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool,
    torch::Tensor bank_ids, torch::Tensor x, torch::Tensor expert_local,
    int64_t n_experts, int64_t pool_experts, int64_t out_per_expert,
    int64_t neuron_len, int64_t sub_bits, int64_t format,
    torch::Tensor out, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts);
torch::Tensor nepq_hadamard_input_cuda(
    torch::Tensor input, torch::Tensor signs, int64_t block_size);
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
void nint_moe_set_small_mmq_cuda(int64_t mode);
std::vector<torch::Tensor> gdn_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> state);
std::vector<torch::Tensor> gdn_inplace_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                             torch::Tensor g, torch::Tensor beta, torch::Tensor state);
std::vector<torch::Tensor> gdn_inplace_transposed_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state);
std::vector<torch::Tensor> gdn_inplace_tiled_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state);
std::vector<torch::Tensor> gdn_inplace_transposed_tiled_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state);
std::vector<torch::Tensor> kv_cache_write_cuda(torch::Tensor k_cache, torch::Tensor v_cache,
                                               torch::Tensor k, torch::Tensor v, torch::Tensor positions);
std::vector<torch::Tensor> kv_cache_write_ring_cuda(
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k, torch::Tensor v, int64_t position_start);
std::vector<torch::Tensor> kv_cache_write_ring_positions_cuda(
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k, torch::Tensor v, torch::Tensor positions);
torch::Tensor ssm_conv_silu_cuda(torch::Tensor conv_input, torch::Tensor weight, torch::Tensor bias, int64_t n_tokens);
torch::Tensor ssm_conv_silu_decode_cuda(torch::Tensor state, torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
std::vector<torch::Tensor> linear_conv_qkv_decode_cuda(
    torch::Tensor state, torch::Tensor qk, torch::Tensor v, torch::Tensor weight, torch::Tensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
std::vector<torch::Tensor> linear_conv_qkv_prefill_cuda(
    torch::Tensor state, torch::Tensor qk, torch::Tensor v, torch::Tensor weight, torch::Tensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
torch::Tensor nint_embedding_lookup_packed_compact_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor token_ids,
    int64_t neuron_len, int64_t gs);
torch::Tensor nint_embedding_lookup_packed_compact_bits_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor token_ids,
    int64_t neuron_len, int64_t gs, int64_t bits);
torch::Tensor nint8_zero_embedding_lookup_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor token_ids,
    int64_t neuron_len);
torch::Tensor sample_greedy_cuda(torch::Tensor logits);
torch::Tensor sample_softmax_cuda(torch::Tensor logits, torch::Tensor random, double temperature);
torch::Tensor sample_top_k_top_p_cuda(
    torch::Tensor logits, torch::Tensor random, double temperature, int64_t top_k, double top_p);
void sample_token_counts_add_cuda(torch::Tensor counts, torch::Tensor tokens);
torch::Tensor sample_apply_penalties_cuda(
    torch::Tensor logits, torch::Tensor counts,
    double presence_penalty, double frequency_penalty, double repetition_penalty);
torch::Tensor nint_gemv_packed_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_qx_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_batch_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_gate_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, torch::Tensor gate,
    int64_t gs, int64_t mode, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_swiglu_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    int64_t gs, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_geglu_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    int64_t gs, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
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
std::vector<torch::Tensor> nint8_one_quantize_reconstruct_cuda(
    torch::Tensor x);
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
torch::Tensor nint_gemv_packed_bits_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    int64_t bits, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_int6_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_bits_qx_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, int64_t gs, int64_t bits,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_bits_swiglu_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    int64_t bits, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_bits_geglu_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    int64_t bits, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
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
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    int64_t bits, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum,
    torch::Tensor block_vals, torch::Tensor block_idxs);
torch::Tensor nint5_gs28_q5_repack_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min);
torch::Tensor nint5_gs28_q5_dequant_cuda(
    torch::Tensor q_packed, torch::Tensor neuron_scale, torch::Tensor neuron_min,
    int64_t neuron_len);
torch::Tensor nint5_gs28_q5_gemv_ws_cuda(
    torch::Tensor q_packed, torch::Tensor neuron_scale, torch::Tensor neuron_min,
    torch::Tensor x, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_bits_m1_out_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    int64_t bits, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum,
    torch::Tensor out);
torch::Tensor nint_gemv_packed_bits_linear_out_norm_gate_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min,
    torch::Tensor y, torch::Tensor z, torch::Tensor norm_weight,
    int64_t gs, int64_t bits, int64_t dv, double eps,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum, torch::Tensor rinv);
torch::Tensor nint_gemv_packed_bits_gate_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, torch::Tensor gate,
    int64_t gs, int64_t bits, int64_t mode, torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_u8_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint_gemv_packed_u8_groupwise_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x,
    int64_t gs, int64_t groups, torch::Tensor qx, torch::Tensor xscale,
    torch::Tensor xsum);
torch::Tensor nint_mmq_packed_u8_ws_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, torch::Tensor x, int64_t gs,
    torch::Tensor qx, torch::Tensor xscale, torch::Tensor xsum);
torch::Tensor nint8_zero_gemv_ws_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor x,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nint8_zero_mmq_ws_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor x,
    torch::Tensor qx, torch::Tensor xscale);
torch::Tensor nint8_zero_dequant_cuda(
    torch::Tensor q, torch::Tensor scale, int64_t neuron_len);
torch::Tensor nint_dequant_full_packed_compact_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, int64_t neuron_len, int64_t gs);
torch::Tensor nint_dequant_full_packed_compact_bits_cuda(
    torch::Tensor q_packed, torch::Tensor sub_scale, torch::Tensor sub_min,
    torch::Tensor neuron_scale, torch::Tensor neuron_min, int64_t neuron_len, int64_t gs, int64_t bits);
torch::Tensor nint_cublas_gemm_nt_f32acc_cuda(torch::Tensor x, torch::Tensor w);
torch::Tensor mxfp8_dequant_cuda(
    torch::Tensor values, torch::Tensor scales);
torch::Tensor mxfp8_small_m_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor x);
torch::Tensor mxfp8_small_m_f32_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor x);
torch::Tensor mxfp8_gemm_f32_cuda(
    torch::Tensor values, torch::Tensor scales, torch::Tensor x);
torch::Tensor mxfp8_groupwise_small_m_cuda(
    torch::Tensor values, torch::Tensor scales,
    torch::Tensor x, int64_t groups);
torch::Tensor mxfp8_groupwise_small_m_f32_cuda(
    torch::Tensor values, torch::Tensor scales,
    torch::Tensor x, int64_t groups);
torch::Tensor nvq_dequant_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
torch::Tensor nepq_dequant_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool,
    torch::Tensor bank_ids, int64_t neuron_len,
    int64_t sub_bits, int64_t format);
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
using torch::indexing::Slice;

struct ProfileStat {
    double ms = 0.0;
    double wall_ms = 0.0;
    int64_t calls = 0;
};

struct CudaProfiler {
    struct PendingEvent {
        std::string name;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
    };

    bool enabled = false;
    bool graph_events = false;
    std::unordered_map<std::string, ProfileStat> stats;
    std::vector<std::string> order;
    std::vector<PendingEvent> pending;

    void reset() {
        for (auto & p : pending) {
            cudaEventDestroy(p.start);
            cudaEventDestroy(p.stop);
        }
        pending.clear();
        stats.clear();
        order.clear();
    }

    template <typename Fn>
    auto measure(const std::string & name, Fn && fn) -> decltype(fn()) {
        if (!enabled) return fn();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        if (graph_events) {
            cudaEventRecordWithFlags(start, stream, cudaEventRecordExternal);
        } else {
            cudaEventRecord(start, stream);
        }
        auto t0 = std::chrono::steady_clock::now();
        auto out = fn();
        if (graph_events) {
            cudaEventRecordWithFlags(stop, stream, cudaEventRecordExternal);
        } else {
            cudaEventRecord(stop, stream);
        }
        auto t1 = std::chrono::steady_clock::now();
        auto it = stats.find(name);
        if (it == stats.end()) {
            order.push_back(name);
            it = stats.emplace(name, ProfileStat{}).first;
        }
        it->second.wall_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        it->second.calls += 1;
        pending.push_back(PendingEvent{name, start, stop});
        return out;
    }

    void report(const std::string & title) {
        if (!enabled) return;
        if (!pending.empty()) cudaEventSynchronize(pending.back().stop);
        for (auto & p : pending) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, p.start, p.stop);
            stats.at(p.name).ms += (double)ms;
            cudaEventDestroy(p.start);
            cudaEventDestroy(p.stop);
        }
        pending.clear();
        std::cerr << "profile " << title << "\n";
        for (const auto & name : order) {
            const auto & s = stats.at(name);
            std::cerr << "profile_item"
                      << " name=" << name
                      << " calls=" << s.calls
                      << " cuda_ms=" << s.ms
                      << " cuda_avg_ms=" << (s.calls ? s.ms / (double)s.calls : 0.0)
                      << " wall_ms=" << s.wall_ms
                      << " wall_avg_ms=" << (s.calls ? s.wall_ms / (double)s.calls : 0.0)
                      << "\n";
        }
    }
};

static CudaProfiler g_profiler;
static int64_t g_decode_graph_attention_parts = 0;
static int64_t g_decode_graph_attention_kv_len = 0;
static bool g_force_moe_pool_path = false;
static bool g_force_moe_unfused_reduce = false;
static bool g_force_moe_materialized_swiglu = false;
static bool g_force_moe_prefill_mma_off = false;
enum class KlMmqMode {
    Default,
    Nint8One,
    Fp16,
};
enum class Nint6MmqMode {
    Fp16,
    Int8,
};
static KlMmqMode g_kl_mmq_mode = KlMmqMode::Default;
static Nint6MmqMode g_nint6_mmq_mode = Nint6MmqMode::Fp16;
static int64_t g_kl_mmq_activation_quantize_calls = 0;
static int64_t g_kl_mmq_dense_calls = 0;
static int64_t g_kl_mmq_moe_calls = 0;
static int64_t g_kl_mmq_fallback_calls = 0;
static int64_t g_kl_kv_cache_capacity = 0;
static std::unordered_set<int> g_dsv4_cpu_offload_layers;
static int64_t g_dsv4_cpu_offload_host_bytes = 0;
class MoeExpertCache;
static std::shared_ptr<MoeExpertCache> g_moe_expert_cache;
static int g_moe_cache_registration_min_slots = 8;
static int g_gemma_trace_layer = -1;
static std::vector<std::pair<std::string, torch::Tensor>> * g_gemma_stage_trace = nullptr;

enum class TensorParallelAxis {
    Mirrored,
    Output,
    Input,
};

struct TensorParallelConfig {
    std::vector<int> devices;
    std::vector<double> split;
    bool allow_duplicate_devices = false;

    bool enabled() const {
        return devices.size() > 1;
    }

    int primary_device() const {
        return devices.empty() ? 0 : devices.front();
    }
};

static TensorParallelConfig g_tensor_parallel;

struct TensorParallelCollectiveRuntime {
    using Stream = decltype(at::cuda::getStreamFromPool(false));

    std::vector<int> devices;
    std::vector<Stream> streams;
    std::vector<cudaEvent_t> ready;
    std::vector<cudaEvent_t> completed;
    std::vector<torch::Tensor> reduction_buffers;
#ifdef MFQ_HAVE_NCCL
    std::vector<ncclComm_t> communicators;
#endif
    bool collectives_enabled = false;

    ~TensorParallelCollectiveRuntime() {
        reset();
    }

    void reset() noexcept {
        // Release CUDA-owned state while every device context is still alive.
        // Relying on static destruction is too late: libtorch's CUDA allocator
        // may already be shutting down when tensors on secondary TP devices are
        // destroyed.
        for (int device : devices) {
            (void)cudaSetDevice(device);
            (void)cudaDeviceSynchronize();
        }
        reduction_buffers.clear();
#ifdef MFQ_HAVE_NCCL
        for (auto communicator : communicators) {
            if (communicator != nullptr) {
                (void)ncclCommDestroy(communicator);
            }
        }
        communicators.clear();
#endif
        for (size_t index = 0; index < devices.size(); ++index) {
            (void)cudaSetDevice(devices[index]);
            if (index < ready.size() && ready[index] != nullptr) {
                (void)cudaEventDestroy(ready[index]);
            }
            if (index < completed.size() && completed[index] != nullptr) {
                (void)cudaEventDestroy(completed[index]);
            }
        }
        devices.clear();
        streams.clear();
        ready.clear();
        completed.clear();
        collectives_enabled = false;
    }

    void configure(
            const std::vector<int> & requested_devices,
            bool allow_duplicate_devices) {
        reset();
        if (requested_devices.size() < 2 || allow_duplicate_devices) {
            return;
        }
        devices = requested_devices;
        streams.reserve(devices.size());
        ready.resize(devices.size(), nullptr);
        completed.resize(devices.size(), nullptr);
        reduction_buffers.resize(devices.size());
        for (size_t index = 0; index < devices.size(); ++index) {
            c10::cuda::CUDAGuard guard(devices[index]);
            streams.push_back(
                at::cuda::getStreamFromPool(false, devices[index]));
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &ready[index], cudaEventDisableTiming));
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &completed[index], cudaEventDisableTiming));
        }
#ifdef MFQ_HAVE_NCCL
        communicators.resize(devices.size(), nullptr);
        MFQ_NCCL_CHECK(ncclCommInitAll(
            communicators.data(),
            static_cast<int>(devices.size()),
            devices.data()));
        collectives_enabled = true;
#endif
    }
};

static TensorParallelCollectiveRuntime g_tensor_parallel_collectives;

struct LayerPlacementConfig {
    std::vector<int> devices;
    std::vector<double> split;
    std::vector<int> layer_devices;
    int load_device = -1;

    bool enabled() const {
        return devices.size() > 1;
    }

    int primary_device() const {
        return devices.empty()
            ? g_tensor_parallel.primary_device()
            : devices.front();
    }

    void prepare(int64_t layers) {
        layer_devices.assign(
            static_cast<size_t>(layers), primary_device());
        if (!enabled()) return;
        const auto slices = mfq::plan_tensor_parallel_slices(
            layers, 1, devices, split);
        for (const auto & slice : slices) {
            for (int64_t layer = slice.begin; layer < slice.end; ++layer) {
                layer_devices.at(static_cast<size_t>(layer)) = slice.device;
            }
        }
    }

    int device_for_layer(int64_t layer) const {
        if (layer < 0 || layer >= static_cast<int64_t>(layer_devices.size())) {
            throw std::runtime_error("layer-placement index is outside the model");
        }
        return layer_devices.at(static_cast<size_t>(layer));
    }
};

static LayerPlacementConfig g_layer_placement;

static int active_weight_load_device() {
    return g_layer_placement.load_device >= 0
        ? g_layer_placement.load_device
        : g_tensor_parallel.primary_device();
}

static const char * kl_mmq_mode_name(KlMmqMode mode) {
    switch (mode) {
        case KlMmqMode::Default: return "default";
        case KlMmqMode::Nint8One: return "nint8_1";
        case KlMmqMode::Fp16: return "fp16";
    }
    return "invalid";
}

struct KlMmqScope {
    KlMmqMode previous_mode;
    int64_t previous_activation_quantize_calls;
    int64_t previous_dense_calls;
    int64_t previous_moe_calls;
    int64_t previous_fallback_calls;

    explicit KlMmqScope(KlMmqMode mode)
        : previous_mode(g_kl_mmq_mode),
          previous_activation_quantize_calls(
              g_kl_mmq_activation_quantize_calls),
          previous_dense_calls(g_kl_mmq_dense_calls),
          previous_moe_calls(g_kl_mmq_moe_calls),
          previous_fallback_calls(g_kl_mmq_fallback_calls) {
        g_kl_mmq_mode = mode;
        g_kl_mmq_activation_quantize_calls = 0;
        g_kl_mmq_dense_calls = 0;
        g_kl_mmq_moe_calls = 0;
        g_kl_mmq_fallback_calls = 0;
    }

    ~KlMmqScope() {
        g_kl_mmq_mode = previous_mode;
        g_kl_mmq_activation_quantize_calls =
            previous_activation_quantize_calls;
        g_kl_mmq_dense_calls = previous_dense_calls;
        g_kl_mmq_moe_calls = previous_moe_calls;
        g_kl_mmq_fallback_calls = previous_fallback_calls;
    }
};

struct KlKvCacheCapacityScope {
    int64_t previous_capacity;

    explicit KlKvCacheCapacityScope(int64_t capacity)
        : previous_capacity(g_kl_kv_cache_capacity) {
        g_kl_kv_cache_capacity = capacity;
    }

    ~KlKvCacheCapacityScope() {
        g_kl_kv_cache_capacity = previous_capacity;
    }
};

static torch::Tensor kl_mmq_prepare_activation(torch::Tensor x) {
    if (g_kl_mmq_mode != KlMmqMode::Nint8One) {
        return x;
    }
    const auto original_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)}).contiguous();
    auto quantized = nint8_one_quantize_reconstruct_cuda(flat);
    ++g_kl_mmq_activation_quantize_calls;
    return quantized.at(3).reshape(original_shape).contiguous();
}

struct MoeRouteLayerStats {
    torch::Tensor counts;
    torch::Tensor weight_sum;
    torch::Tensor weight_sq_sum;
    torch::Tensor output_energy;
    torch::Tensor weighted_output_energy;
};

static std::unordered_map<int, MoeRouteLayerStats> g_moe_route_stats;

static const char * moe_route_stats_path() {
    const char * value = std::getenv("MFQ_MOE_ROUTE_STATS");
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

static bool moe_route_output_energy_enabled() {
    const char * value = std::getenv("MFQ_MOE_ROUTE_OUTPUT_ENERGY");
    return value != nullptr && std::atoi(value) != 0;
}

static void record_moe_route_stats(
        int layer,
        const torch::Tensor & ids,
        const torch::Tensor & weights,
        const torch::Tensor & output,
        int n_experts) {
    if (layer < 0 || moe_route_stats_path() == nullptr) return;
    auto found = g_moe_route_stats.find(layer);
    if (found == g_moe_route_stats.end()) {
        auto options = torch::TensorOptions()
            .device(ids.device()).dtype(torch::kFloat64);
        MoeRouteLayerStats value{
            torch::zeros({n_experts}, options),
            torch::zeros({n_experts}, options),
            torch::zeros({n_experts}, options),
            torch::zeros({n_experts}, options),
            torch::zeros({n_experts}, options),
        };
        found = g_moe_route_stats.emplace(layer, std::move(value)).first;
    }
    auto flat_ids = ids.reshape({-1}).to(torch::kInt64);
    auto flat_weights = weights.reshape({-1}).to(torch::kFloat64);
    auto ones = torch::ones_like(flat_weights);
    found->second.counts.scatter_add_(0, flat_ids, ones);
    found->second.weight_sum.scatter_add_(0, flat_ids, flat_weights);
    found->second.weight_sq_sum.scatter_add_(
        0, flat_ids, flat_weights.square());
    if (moe_route_output_energy_enabled()) {
        auto energy = output.reshape({flat_ids.numel(), output.size(-1)})
            .to(torch::kFloat32).square().sum(1).to(torch::kFloat64);
        found->second.output_energy.scatter_add_(0, flat_ids, energy);
        found->second.weighted_output_energy.scatter_add_(
            0, flat_ids, energy * flat_weights.square());
    }
}

static void write_moe_route_stats() {
    const char * path_value = moe_route_stats_path();
    if (path_value == nullptr || g_moe_route_stats.empty()) return;
    torch::cuda::synchronize();
    std::filesystem::path path(path_value);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "cannot write MoE route statistics: " + path.string());
    }
    out << "layer,expert,count,weight_sum,weight_sq_sum,"
           "output_energy,weighted_output_energy\n";
    std::vector<int> layers;
    layers.reserve(g_moe_route_stats.size());
    for (const auto & item : g_moe_route_stats) layers.push_back(item.first);
    std::sort(layers.begin(), layers.end());
    out << std::setprecision(17);
    for (int layer : layers) {
        const auto & value = g_moe_route_stats.at(layer);
        auto counts = value.counts.to(torch::kCPU).contiguous();
        auto weight_sum = value.weight_sum.to(torch::kCPU).contiguous();
        auto weight_sq_sum = value.weight_sq_sum.to(torch::kCPU).contiguous();
        auto output_energy = value.output_energy.to(torch::kCPU).contiguous();
        auto weighted_output_energy =
            value.weighted_output_energy.to(torch::kCPU).contiguous();
        const auto n_experts = counts.numel();
        const double * count_data = counts.data_ptr<double>();
        const double * weight_data = weight_sum.data_ptr<double>();
        const double * weight_sq_data = weight_sq_sum.data_ptr<double>();
        const double * energy_data = output_energy.data_ptr<double>();
        const double * weighted_energy_data =
            weighted_output_energy.data_ptr<double>();
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            out << layer << ',' << expert << ','
                << count_data[expert] << ','
                << weight_data[expert] << ','
                << weight_sq_data[expert] << ','
                << energy_data[expert] << ','
                << weighted_energy_data[expert] << '\n';
        }
    }
    std::cout << "moe_route_stats_path=" << path.string()
              << " layers=" << layers.size()
              << " output_energy="
              << (moe_route_output_energy_enabled() ? 1 : 0) << "\n";
}

static void trace_gemma_stage(int layer, const char * name, const torch::Tensor & value) {
    if (g_gemma_stage_trace != nullptr && layer == g_gemma_trace_layer) {
        g_gemma_stage_trace->emplace_back(
            name, value.to(torch::kFloat32).contiguous().clone());
    }
}

static bool gemma4_fused_norms_enabled() {
    const char * value = std::getenv("MFQ_GEMMA4_FUSED_NORMS");
    return value == nullptr || std::atoi(value) != 0;
}

static void report_cuda_memory(const char * stage) {
    const char * enabled = std::getenv("MFQ_REPORT_CUDA_MEMORY");
    if (enabled == nullptr || std::atoi(enabled) == 0) return;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    MFQ_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
        c10::cuda::current_device());
    constexpr size_t aggregate = static_cast<size_t>(
        c10::CachingDeviceAllocator::StatType::AGGREGATE);
    constexpr double mib = 1024.0 * 1024.0;
    std::cout << "cuda_memory_stage=" << stage
              << " free_mib=" << free_bytes / mib
              << " used_mib=" << (total_bytes - free_bytes) / mib
              << " total_mib=" << total_bytes / mib
              << " allocated_mib=" << stats.allocated_bytes[aggregate].current / mib
              << " active_mib=" << stats.active_bytes[aggregate].current / mib
              << " reserved_mib=" << stats.reserved_bytes[aggregate].current / mib
              << " inactive_split_mib="
              << stats.inactive_split_bytes[aggregate].current / mib
              << " requested_mib=" << stats.requested_bytes[aggregate].current / mib
              << " allocations=" << stats.allocation[aggregate].current
              << " segments=" << stats.segment[aggregate].current
              << " retries=" << stats.num_alloc_retries
              << " ooms=" << stats.num_ooms << "\n";
}

static bool moe_small_hetero_enabled(int tokens) {
    static const bool disabled = [] {
        const char * value = std::getenv("MFQ_DISABLE_MOE_SMALL_HETERO");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return tokens == 1 || (tokens <= 4 && !disabled);
}

static uint32_t read_u32(std::istream & is) {
    uint32_t v = 0;
    is.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!is) throw std::runtime_error("unexpected EOF reading u32");
    return v;
}

static uint64_t read_u64(std::istream & is) {
    uint64_t v = 0;
    is.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!is) throw std::runtime_error("unexpected EOF reading u64");
    return v;
}

static int32_t read_i32_from(const std::vector<uint8_t> & b, size_t & off) {
    int32_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

static uint16_t read_u16_from(const std::vector<uint8_t> & b, size_t & off) {
    uint16_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

static int64_t read_i64_from(const std::vector<uint8_t> & b, size_t & off) {
    int64_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

static uint32_t read_u32_from(const std::vector<uint8_t> & b, size_t & off) {
    uint32_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

static uint64_t read_u64_from(const std::vector<uint8_t> & b, size_t & off) {
    uint64_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

static std::string read_str(std::istream & is) {
    uint32_t n = read_u32(is);
    std::string s(n, '\0');
    is.read(s.data(), n);
    if (!is) throw std::runtime_error("unexpected EOF reading string");
    return s;
}

static std::vector<uint8_t> unpack_bits(const std::vector<uint8_t> & blob, size_t & off, size_t count, int bits) {
    std::vector<uint8_t> out(count);
    if (bits == 8) {
        std::copy(blob.begin() + (ptrdiff_t)off, blob.begin() + (ptrdiff_t)(off + count), out.begin());
        off += count;
        return out;
    }
    if (bits == 4) {
        size_t nbytes = (count + 1) / 2;
        for (size_t i = 0; i < count; ++i) {
            uint8_t p = blob[off + i / 2];
            out[i] = (i & 1) ? (p >> 4) : (p & 0x0f);
        }
        off += nbytes;
        return out;
    }
    size_t nbytes = (count * (size_t)bits + 7) / 8;
    for (size_t i = 0; i < count; ++i) {
        uint32_t v = 0;
        size_t bit0 = i * (size_t)bits;
        for (int j = 0; j < bits; ++j) {
            size_t bit = bit0 + (size_t)j;
            uint8_t by = blob[off + bit / 8];
            v |= ((by >> (bit & 7)) & 1u) << j;
        }
        out[i] = (uint8_t)v;
    }
    off += nbytes;
    return out;
}

struct Record {
    std::string name;
    std::string dtype;
    std::string source_path;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

struct MfqRecordHeader {
    uint32_t version = 0;
    std::string architecture;
    std::unordered_map<std::string, std::string> extra_json;
    uint32_t record_count = 0;
};

static uint64_t mfq_json_uint(
        const std::unordered_map<std::string, std::string> & extra,
        const std::string & key,
        uint64_t default_value) {
    auto it = extra.find(key);
    if (it == extra.end()) return default_value;
    const std::string & text = it->second;
    size_t parsed = 0;
    uint64_t value = 0;
    try {
        value = std::stoull(text, &parsed, 10);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid MFQ integer metadata " + key + ": " + text);
    }
    while (parsed < text.size() &&
           std::isspace(static_cast<unsigned char>(text[parsed]))) {
        ++parsed;
    }
    if (parsed != text.size()) {
        throw std::runtime_error("invalid MFQ integer metadata " + key + ": " + text);
    }
    return value;
}

static std::vector<std::string> mfq_shard_paths(
        const std::string & path_value,
        uint64_t split_no,
        uint64_t split_count) {
    const std::filesystem::path input(path_value);
    const std::string filename = input.filename().string();
    static const std::regex pattern(
        R"(^(.*)-([0-9]{5})-of-([0-9]{5})\.mfq$)");
    std::smatch match;
    if (!std::regex_match(filename, match, pattern)) {
        throw std::runtime_error(
            "sharded MFQ path lacks -00001-of-00000 suffix: " + path_value);
    }
    const uint64_t file_no = std::stoull(match[2].str());
    const uint64_t file_count = std::stoull(match[3].str());
    if (file_no != split_no + 1 || file_count != split_count) {
        throw std::runtime_error(
            "MFQ shard filename/metadata mismatch: " + path_value);
    }
    std::vector<std::string> result;
    result.reserve(static_cast<size_t>(split_count));
    for (uint64_t i = 1; i <= split_count; ++i) {
        std::ostringstream name;
        name << match[1].str() << "-" << std::setfill('0') << std::setw(5) << i
             << "-of-" << std::setw(5) << split_count << ".mfq";
        const auto shard = input.parent_path() / name.str();
        if (!std::filesystem::is_regular_file(shard)) {
            throw std::runtime_error("missing MFQ shard: " + shard.string());
        }
        result.push_back(shard.string());
    }
    return result;
}

static bool g_mfq_drop_file_cache = false;

static void mfq_drop_file_cache(
    const std::string & path,
    uint64_t offset,
    uint64_t nbytes) {
#ifndef _WIN32
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    (void)::posix_fadvise(
        fd, (off_t)offset, (off_t)nbytes, POSIX_FADV_DONTNEED);
    (void)::close(fd);
#else
    (void)path;
    (void)offset;
    (void)nbytes;
#endif
}

struct MfqDropFileCacheGuard {
    bool previous = false;

    explicit MfqDropFileCacheGuard(bool enabled)
        : previous(g_mfq_drop_file_cache) {
        g_mfq_drop_file_cache = enabled;
    }

    ~MfqDropFileCacheGuard() {
        g_mfq_drop_file_cache = previous;
    }
};

struct MfqFile {
    std::string path;
    std::vector<std::string> source_paths;
    std::unordered_map<std::string, Record> records;
    std::unordered_map<std::string, Record> expert_overlays;
    std::unordered_map<std::string, Record> tensor_overlays;

    explicit MfqFile(std::string p) : path(std::move(p)) {
        source_paths = {path};
        auto load_records = [](
                const std::string & source,
                std::unordered_map<std::string, Record> & destination) {
            std::ifstream is(source, std::ios::binary);
            if (!is) {
                throw std::runtime_error("cannot open MFQ file: " + source);
            }
            char magic[4];
            is.read(magic, 4);
            if (std::string(magic, magic + 4) != "MFQ1") {
                throw std::runtime_error("bad MFQ magic: " + source);
            }
            MfqRecordHeader header;
            header.version = read_u32(is);
            header.architecture = read_str(is);
            if (header.version >= 2) {
                uint32_t n_extra = read_u32(is);
                for (uint32_t i = 0; i < n_extra; ++i) {
                    const std::string key = read_str(is);
                    const std::string value = read_str(is);
                    if (!header.extra_json.emplace(key, value).second) {
                        throw std::runtime_error(
                            "duplicate MFQ metadata key: " + key + ": " + source);
                    }
                }
            }
            uint32_t nrec = read_u32(is);
            header.record_count = nrec;
            std::vector<Record> raw;
            raw.reserve(nrec);
            for (uint32_t i = 0; i < nrec; ++i) {
                Record r;
                r.name = read_str(is);
                r.dtype = read_str(is);
                r.source_path = source;
                r.nbytes = read_u64(is);
                raw.push_back(std::move(r));
            }
            uint64_t off = (uint64_t)is.tellg();
            for (auto & r : raw) {
                r.offset = off;
                off += r.nbytes;
                if (!destination.emplace(r.name, std::move(r)).second) {
                    throw std::runtime_error(
                        "duplicate MFQ tensor record: " + source);
                }
            }
            is.seekg(0, std::ios::end);
            const uint64_t file_size = static_cast<uint64_t>(is.tellg());
            if (off != file_size) {
                throw std::runtime_error(
                    "MFQ file length does not match record table: " + source);
            }
            return header;
        };
        MfqRecordHeader probe = load_records(path, records);
        const uint64_t split_no = mfq_json_uint(probe.extra_json, "split.no", 0);
        const uint64_t split_count =
            mfq_json_uint(probe.extra_json, "split.count", 1);
        if (split_count < 1 || split_count > 99999 || split_no >= split_count) {
            throw std::runtime_error("invalid MFQ split metadata: " + path);
        }
        if (split_count > 1) {
            const auto paths = mfq_shard_paths(path, split_no, split_count);
            records.clear();
            uint64_t expected_records = std::numeric_limits<uint64_t>::max();
            uint64_t expected_tensors = std::numeric_limits<uint64_t>::max();
            uint64_t actual_records = 0;
            uint64_t actual_tensors = 0;
            for (uint64_t i = 0; i < split_count; ++i) {
                MfqRecordHeader current = load_records(paths[(size_t)i], records);
                const uint64_t current_no =
                    mfq_json_uint(current.extra_json, "split.no", split_count);
                const uint64_t current_count =
                    mfq_json_uint(current.extra_json, "split.count", 0);
                if (current_no != i || current_count != split_count) {
                    throw std::runtime_error(
                        "MFQ shard metadata mismatch: " + paths[(size_t)i]);
                }
                if (current.version != probe.version ||
                    current.architecture != probe.architecture) {
                    throw std::runtime_error(
                        "MFQ shard architecture/version mismatch: " +
                        paths[(size_t)i]);
                }
                const uint64_t current_expected_records = mfq_json_uint(
                    current.extra_json, "split.records.count",
                    std::numeric_limits<uint64_t>::max());
                const uint64_t current_expected_tensors = mfq_json_uint(
                    current.extra_json, "split.tensors.count",
                    std::numeric_limits<uint64_t>::max());
                if (expected_records == std::numeric_limits<uint64_t>::max()) {
                    expected_records = current_expected_records;
                } else if (
                    current_expected_records != std::numeric_limits<uint64_t>::max() &&
                    current_expected_records != expected_records) {
                    throw std::runtime_error(
                        "MFQ shard record count mismatch: " + paths[(size_t)i]);
                }
                if (expected_tensors == std::numeric_limits<uint64_t>::max()) {
                    expected_tensors = current_expected_tensors;
                } else if (
                    current_expected_tensors != std::numeric_limits<uint64_t>::max() &&
                    current_expected_tensors != expected_tensors) {
                    throw std::runtime_error(
                        "MFQ shard tensor count mismatch: " + paths[(size_t)i]);
                }
                actual_records += current.record_count;
            }
            for (const auto & item : records) {
                if (item.first.rfind("__mfq_asset__/", 0) != 0) {
                    ++actual_tensors;
                }
            }
            if (expected_records != std::numeric_limits<uint64_t>::max() &&
                actual_records != expected_records) {
                throw std::runtime_error("MFQ shard record total mismatch");
            }
            if (expected_tensors != std::numeric_limits<uint64_t>::max() &&
                actual_tensors != expected_tensors) {
                throw std::runtime_error("MFQ shard tensor total mismatch");
            }
            path = paths.front();
            source_paths = paths;
        }
        if (const char * overlay = std::getenv("MFQ_TENSOR_OVERLAY")) {
            if (overlay[0] != '\0') {
                const MfqRecordHeader overlay_header =
                    load_records(overlay, tensor_overlays);
                if (mfq_json_uint(overlay_header.extra_json, "split.count", 1) != 1) {
                    throw std::runtime_error(
                        "sharded MFQ tensor overlays are not supported");
                }
                for (const auto & item : tensor_overlays) {
                    auto base = records.find(item.first);
                    if (base == records.end()) {
                        throw std::runtime_error(
                            "tensor overlay replaces an unknown record: " + item.first);
                    }
                    base->second = item.second;
                }
            }
        }
        if (const char * overlay = std::getenv("MFQ_EXPERT_OVERLAY")) {
            if (overlay[0] != '\0') {
                const MfqRecordHeader overlay_header =
                    load_records(overlay, expert_overlays);
                if (mfq_json_uint(overlay_header.extra_json, "split.count", 1) != 1) {
                    throw std::runtime_error(
                        "sharded MFQ expert overlays are not supported");
                }
                for (const auto & item : expert_overlays) {
                    if (item.second.dtype != "NINTMD" ||
                        records.find(item.first) == records.end() ||
                        records.at(item.first).dtype != "NINTM") {
                        throw std::runtime_error(
                            "invalid expert overlay record: " + item.first);
                    }
                }
            }
        }
    }

    static std::vector<uint8_t> read_record_blob(
            const Record & r, const std::string & name) {
        std::vector<uint8_t> blob((size_t)r.nbytes);
        std::ifstream is(r.source_path, std::ios::binary);
        is.seekg((std::streamoff)r.offset);
        is.read(reinterpret_cast<char *>(blob.data()), (std::streamsize)blob.size());
        if (!is) throw std::runtime_error("failed reading tensor blob: " + name);
        is.close();
        if (g_mfq_drop_file_cache) {
            mfq_drop_file_cache(r.source_path, r.offset, r.nbytes);
        }
        return blob;
    }

    std::vector<uint8_t> read_blob(const std::string & name) const {
        auto it = records.find(name);
        if (it == records.end()) throw std::runtime_error("missing tensor: " + name);
        return read_record_blob(it->second, name);
    }

    bool has_record(const std::string & name) const {
        return records.find(name) != records.end();
    }

    std::string read_asset_text(const std::string & name) const {
        const Record & value = record(name);
        if (value.dtype != "BLOB") {
            throw std::runtime_error(
                "MFQ runtime asset has non-BLOB dtype: " + name);
        }
        const auto blob = read_record_blob(value, name);
        return std::string(blob.begin(), blob.end());
    }

    std::vector<uint8_t> read_asset(const std::string & name) const {
        const Record & value = record(name);
        if (value.dtype != "BLOB") {
            throw std::runtime_error(
                "MFQ runtime asset has non-BLOB dtype: " + name);
        }
        return read_record_blob(value, name);
    }

    bool has_expert_overlay(const std::string & name) const {
        return expert_overlays.find(name) != expert_overlays.end();
    }

    std::vector<uint8_t> read_expert_overlay(
            const std::string & name) const {
        auto it = expert_overlays.find(name);
        if (it == expert_overlays.end()) {
            throw std::runtime_error("missing expert overlay: " + name);
        }
        return read_record_blob(it->second, name);
    }

    const Record & record(const std::string & name) const {
        auto it = records.find(name);
        if (it == records.end()) throw std::runtime_error("missing tensor: " + name);
        return it->second;
    }
};

struct NintCpu {
    int bits = 0;
    int sub_bits = 0;
    int gs = 0;
    int axis = 0;
    int neuron_len = 0;
    std::vector<int64_t> shape;
    int out = 0;
    int ng = 0;
    int qbytes = 0;
    std::vector<uint8_t> q_packed;
    std::vector<uint8_t> sub_scale;
    std::vector<uint8_t> sub_min;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<uint16_t> neuron_min_h;
};

static NintCpu unpack_nint(const std::vector<uint8_t> & blob) {
    NintCpu t;
    size_t off = 0;
    t.bits = blob[off++];
    t.sub_bits = blob[off++];
    t.gs = read_i32_from(blob, off);
    t.axis = read_i32_from(blob, off);
    t.neuron_len = read_i32_from(blob, off);
    uint32_t ndim = read_u32_from(blob, off);
    t.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) t.shape[i] = read_i64_from(blob, off);
    t.out = (int)read_u32_from(blob, off);
    t.ng = (int)read_u32_from(blob, off);
    t.neuron_scale_h.resize(t.out);
    std::memcpy(t.neuron_scale_h.data(), blob.data() + off, (size_t)t.out * 2);
    off += (size_t)t.out * 2;
    t.neuron_min_h.resize(t.out);
    std::memcpy(t.neuron_min_h.data(), blob.data() + off, (size_t)t.out * 2);
    off += (size_t)t.out * 2;
    size_t sub_count = (size_t)t.out * t.ng;
    size_t q_count = sub_count * (size_t)t.gs;
    t.sub_scale = unpack_bits(blob, off, sub_count, t.sub_bits);
    t.sub_min = unpack_bits(blob, off, sub_count, t.sub_bits);
    t.qbytes = (t.gs * t.bits + 7) / 8;
    size_t compact_q_nbytes = (q_count * (size_t)t.bits + 7) / 8;
    size_t row_bits = (size_t)t.gs * t.bits;
    size_t nrows = (size_t)t.out * t.ng;
    if ((row_bits & 7u) == 0) {
        t.q_packed.resize(compact_q_nbytes);
        std::memcpy(t.q_packed.data(), blob.data() + off, compact_q_nbytes);
    } else {
        t.q_packed.assign(nrows * (size_t)t.qbytes, 0);
        for (size_t row = 0; row < nrows; ++row) {
            size_t src_bit0 = row * row_bits;
            size_t dst_byte0 = row * (size_t)t.qbytes;
            for (size_t bit = 0; bit < row_bits; ++bit) {
                uint8_t v = (blob[off + (src_bit0 + bit) / 8] >> ((src_bit0 + bit) & 7)) & 1u;
                if (v) t.q_packed[dst_byte0 + bit / 8] |= (uint8_t)(1u << (bit & 7));
            }
        }
    }
    off += compact_q_nbytes;
    return t;
}

struct Nint8ZeroCpu {
    int axis = 0;
    int neuron_len = 0;
    std::vector<int64_t> shape;
    int out = 0;
    int ng = 0;
    std::vector<uint8_t> q;
    std::vector<uint16_t> scale_h;
};

static Nint8ZeroCpu unpack_nint8_zero(const std::vector<uint8_t> & blob) {
    constexpr size_t block_bytes = 34;
    if (blob.size() < 24 || std::memcmp(blob.data(), "NI80", 4) != 0) {
        throw std::runtime_error("invalid NINT8-0 header");
    }
    Nint8ZeroCpu t;
    size_t off = 4;
    t.axis = read_i32_from(blob, off);
    t.neuron_len = read_i32_from(blob, off);
    uint32_t ndim = read_u32_from(blob, off);
    if (ndim == 0 || t.axis < 0 || t.axis >= static_cast<int>(ndim) ||
        t.neuron_len <= 0 || t.neuron_len % 32 != 0) {
        throw std::runtime_error("invalid NINT8-0 dimensions");
    }
    t.shape.resize(ndim);
    for (uint32_t index = 0; index < ndim; ++index) {
        t.shape[index] = read_i64_from(blob, off);
    }
    t.out = static_cast<int>(read_u32_from(blob, off));
    t.ng = static_cast<int>(read_u32_from(blob, off));
    if (t.out <= 0 || t.ng != t.neuron_len / 32 ||
        t.shape[static_cast<size_t>(t.axis)] != t.out) {
        throw std::runtime_error("NINT8-0 shape/header mismatch");
    }
    int64_t elements = 1;
    for (int64_t value : t.shape) {
        if (value <= 0 || elements > INT64_MAX / value) {
            throw std::runtime_error("invalid NINT8-0 logical shape");
        }
        elements *= value;
    }
    if (elements != static_cast<int64_t>(t.out) * t.neuron_len) {
        throw std::runtime_error("NINT8-0 logical element count mismatch");
    }
    const size_t blocks = static_cast<size_t>(t.out) * t.ng;
    if (off > blob.size() || blob.size() - off != blocks * block_bytes) {
        throw std::runtime_error("invalid NINT8-0 block payload length");
    }
    t.q.resize(blocks * 32);
    t.scale_h.resize(blocks);
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t * source = blob.data() + off + block * block_bytes;
        std::memcpy(&t.scale_h[block], source, sizeof(uint16_t));
        std::memcpy(t.q.data() + block * 32, source + 2, 32);
    }
    return t;
}

static void require_tp_row_major_weight(
        const std::vector<int64_t> & shape,
        int axis,
        int out,
        int neuron_len,
        const char * format) {
    if (shape.size() != 2 || axis != 0 ||
        shape[0] != out || shape[1] != neuron_len) {
        throw std::runtime_error(
            std::string(format) +
            " tensor parallelism requires a row-major rank-2 weight");
    }
}

static NintCpu slice_nint_cpu_output(
        const NintCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT output shard");
    }
    NintCpu result = source;
    result.out = static_cast<int>(end - begin);
    result.shape[0] = result.out;
    const size_t first_group =
        static_cast<size_t>(begin) * source.ng;
    const size_t group_count =
        static_cast<size_t>(result.out) * source.ng;
    result.q_packed.assign(
        source.q_packed.begin() +
            static_cast<ptrdiff_t>(first_group * source.qbytes),
        source.q_packed.begin() +
            static_cast<ptrdiff_t>(
                (first_group + group_count) * source.qbytes));
    result.sub_scale.assign(
        source.sub_scale.begin() + static_cast<ptrdiff_t>(first_group),
        source.sub_scale.begin() +
            static_cast<ptrdiff_t>(first_group + group_count));
    result.sub_min.assign(
        source.sub_min.begin() + static_cast<ptrdiff_t>(first_group),
        source.sub_min.begin() +
            static_cast<ptrdiff_t>(first_group + group_count));
    result.neuron_scale_h.assign(
        source.neuron_scale_h.begin() + static_cast<ptrdiff_t>(begin),
        source.neuron_scale_h.begin() + static_cast<ptrdiff_t>(end));
    result.neuron_min_h.assign(
        source.neuron_min_h.begin() + static_cast<ptrdiff_t>(begin),
        source.neuron_min_h.begin() + static_cast<ptrdiff_t>(end));
    return result;
}

static NintCpu slice_nint_cpu_input_groups(
        const NintCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT input shard");
    }
    NintCpu result = source;
    result.ng = static_cast<int>(end - begin);
    const int64_t element_begin = begin * source.gs;
    const int64_t element_end =
        std::min<int64_t>(end * source.gs, source.neuron_len);
    result.neuron_len = static_cast<int>(element_end - element_begin);
    result.shape[1] = result.neuron_len;
    result.q_packed.resize(
        static_cast<size_t>(source.out) * result.ng * source.qbytes);
    result.sub_scale.resize(
        static_cast<size_t>(source.out) * result.ng);
    result.sub_min.resize(
        static_cast<size_t>(source.out) * result.ng);
    for (int row = 0; row < source.out; ++row) {
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.q_packed.data() +
                destination_group * source.qbytes,
            source.q_packed.data() +
                source_group * source.qbytes,
            static_cast<size_t>(result.ng) * source.qbytes);
        std::memcpy(
            result.sub_scale.data() + destination_group,
            source.sub_scale.data() + source_group,
            static_cast<size_t>(result.ng));
        std::memcpy(
            result.sub_min.data() + destination_group,
            source.sub_min.data() + source_group,
            static_cast<size_t>(result.ng));
    }
    return result;
}

static Nint8ZeroCpu slice_nint8_zero_cpu_output(
        const Nint8ZeroCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT8-0 output shard");
    }
    Nint8ZeroCpu result = source;
    result.out = static_cast<int>(end - begin);
    result.shape[0] = result.out;
    const size_t first_group =
        static_cast<size_t>(begin) * source.ng;
    const size_t group_count =
        static_cast<size_t>(result.out) * source.ng;
    result.q.assign(
        source.q.begin() +
            static_cast<ptrdiff_t>(first_group * 32),
        source.q.begin() +
            static_cast<ptrdiff_t>((first_group + group_count) * 32));
    result.scale_h.assign(
        source.scale_h.begin() + static_cast<ptrdiff_t>(first_group),
        source.scale_h.begin() +
            static_cast<ptrdiff_t>(first_group + group_count));
    return result;
}

static Nint8ZeroCpu slice_nint8_zero_cpu_input_groups(
        const Nint8ZeroCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT8-0 input shard");
    }
    Nint8ZeroCpu result = source;
    result.ng = static_cast<int>(end - begin);
    result.neuron_len = result.ng * 32;
    result.shape[1] = result.neuron_len;
    result.q.resize(
        static_cast<size_t>(source.out) * result.ng * 32);
    result.scale_h.resize(
        static_cast<size_t>(source.out) * result.ng);
    for (int row = 0; row < source.out; ++row) {
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.q.data() + destination_group * 32,
            source.q.data() + source_group * 32,
            static_cast<size_t>(result.ng) * 32);
        std::memcpy(
            result.scale_h.data() + destination_group,
            source.scale_h.data() + source_group,
            static_cast<size_t>(result.ng) * sizeof(uint16_t));
    }
    return result;
}

struct Mxfp8Cpu {
    int64_t out = 0;
    int64_t neuron_len = 0;
    std::vector<uint8_t> values;
    std::vector<uint8_t> scales;
};

static size_t checked_mxfp8_size(
        uint64_t left, uint64_t right,
        const char * label) {
    if (left == 0 || right == 0 ||
            left > std::numeric_limits<size_t>::max() / right) {
        throw std::runtime_error(
            std::string("invalid MXFP8 ") + label);
    }
    return static_cast<size_t>(left * right);
}

static Mxfp8Cpu unpack_mxfp8(
        const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 56;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "MXT1", 4) != 0 ||
            blob[4] != 1 || blob[5] != 8 ||
            blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid MXFP8 payload header");
    }
    size_t offset = 8;
    const uint64_t rows = read_u64_from(blob, offset);
    const uint64_t columns = read_u64_from(blob, offset);
    const uint64_t storage_rows = read_u64_from(blob, offset);
    const uint64_t storage_columns = read_u64_from(blob, offset);
    const uint64_t scale_rows = read_u64_from(blob, offset);
    const uint64_t scale_columns = read_u64_from(blob, offset);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns % 128 != 0 ||
            storage_rows != rows || storage_columns != columns ||
            scale_rows != (rows + 127) / 128 ||
            scale_columns != columns / 128) {
        throw std::runtime_error("invalid MXFP8 payload geometry");
    }
    const size_t value_bytes = checked_mxfp8_size(
        storage_rows, storage_columns, "value size");
    const size_t scale_bytes = checked_mxfp8_size(
        scale_rows, scale_columns, "scale size");
    if (value_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - value_bytes) {
        throw std::runtime_error("invalid MXFP8 payload length");
    }
    Mxfp8Cpu result;
    result.out = static_cast<int64_t>(rows);
    result.neuron_len = static_cast<int64_t>(columns);
    result.values.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + value_bytes));
    offset += value_bytes;
    result.scales.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.end());
    return result;
}

static Mxfp8Cpu slice_mxfp8_cpu(
        const Mxfp8Cpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis != TensorParallelAxis::Output &&
            axis != TensorParallelAxis::Input) {
        throw std::runtime_error("MXFP8 slicing requires output or input axis");
    }
    const int64_t extent = axis == TensorParallelAxis::Output
        ? source.out : source.neuron_len;
    if (begin < 0 || begin >= end || end > extent ||
            begin % 128 != 0 ||
            (end != extent && end % 128 != 0)) {
        throw std::runtime_error(
            "MXFP8 tensor-parallel shards must preserve 128-element blocks");
    }
    Mxfp8Cpu result;
    if (axis == TensorParallelAxis::Output) {
        result.out = end - begin;
        result.neuron_len = source.neuron_len;
        const size_t value_begin =
            static_cast<size_t>(begin * source.neuron_len);
        const size_t value_end =
            static_cast<size_t>(end * source.neuron_len);
        result.values.assign(
            source.values.begin() + static_cast<ptrdiff_t>(value_begin),
            source.values.begin() + static_cast<ptrdiff_t>(value_end));
        const int64_t scale_columns = source.neuron_len / 128;
        const size_t scale_begin =
            static_cast<size_t>((begin / 128) * scale_columns);
        const size_t scale_end = static_cast<size_t>(
            ((end + 127) / 128) * scale_columns);
        result.scales.assign(
            source.scales.begin() + static_cast<ptrdiff_t>(scale_begin),
            source.scales.begin() + static_cast<ptrdiff_t>(scale_end));
        return result;
    }

    result.out = source.out;
    result.neuron_len = end - begin;
    result.values.resize(
        static_cast<size_t>(result.out * result.neuron_len));
    for (int64_t row = 0; row < source.out; ++row) {
        std::memcpy(
            result.values.data() +
                static_cast<size_t>(row * result.neuron_len),
            source.values.data() +
                static_cast<size_t>(row * source.neuron_len + begin),
            static_cast<size_t>(result.neuron_len));
    }
    const int64_t source_scale_columns = source.neuron_len / 128;
    const int64_t result_scale_columns = result.neuron_len / 128;
    const int64_t scale_rows = (source.out + 127) / 128;
    result.scales.resize(
        static_cast<size_t>(scale_rows * result_scale_columns));
    for (int64_t row = 0; row < scale_rows; ++row) {
        std::memcpy(
            result.scales.data() +
                static_cast<size_t>(row * result_scale_columns),
            source.scales.data() + static_cast<size_t>(
                row * source_scale_columns + begin / 128),
            static_cast<size_t>(result_scale_columns));
    }
    return result;
}

static torch::Tensor cpu_u8_tensor(const std::vector<uint8_t> & v, std::initializer_list<int64_t> shape) {
    return torch::from_blob((void *)v.data(), shape, torch::TensorOptions().dtype(torch::kUInt8)).clone();
}

static torch::Tensor cpu_f16_tensor(
        const std::vector<uint16_t> & v,
        std::initializer_list<int64_t> shape) {
    return torch::from_blob(
        (void *)v.data(), shape,
        torch::TensorOptions().dtype(torch::kFloat16)).clone();
}

static torch::Tensor cpu_f16_to_f32_tensor(const std::vector<uint16_t> & v, int64_t n) {
    auto h = torch::from_blob((void *)v.data(), {n}, torch::TensorOptions().dtype(torch::kFloat16)).clone();
    return h.to(torch::kFloat32).contiguous();
}

struct Mxfp8Weight {
    torch::Tensor values;
    torch::Tensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;
};

static Mxfp8Weight to_cuda_device_mxfp8(
        const Mxfp8Cpu & source,
        int device) {
    c10::cuda::CUDAGuard guard(device);
    Mxfp8Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values,
        {source.out, source.neuron_len})
        .to(torch::Device(torch::kCUDA, device), false, false)
        .contiguous();
    result.scales = cpu_u8_tensor(
        source.scales,
        {(source.out + 127) / 128, source.neuron_len / 128})
        .to(torch::Device(torch::kCUDA, device), false, false)
        .contiguous();
    return result;
}

struct Workspace {
    int M = 0;
    int K_pad = 0;
    int argmax_blocks = 0;
    torch::Tensor qx;
    torch::Tensor xscale;
    torch::Tensor xsum;
    torch::Tensor argmax_vals;
    torch::Tensor argmax_idxs;
    torch::Tensor out_buf;
    torch::Tensor rinv;
    torch::Tensor mmq_partial;
    torch::Tensor mmq_qx;
};

struct NintWeight {
    torch::Tensor q_packed;
    torch::Tensor q8_zero_scale;
    torch::Tensor sub_scale;
    torch::Tensor sub_min;
    torch::Tensor neuron_scale;
    torch::Tensor neuron_min;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t gs = 0;
    int64_t bits = 0;
    int64_t qbytes = 0;
    int64_t neuron_len = 0;
    bool q5_exec = false;
    bool q8_zero = false;
    std::vector<int64_t> shape;
    mutable std::unordered_map<int, Workspace> workspaces;

    Workspace & workspace(int M) const {
        int K_pad = (int)(ng * gs);
        auto it = workspaces.find(M);
        if (it != workspaces.end() && it->second.K_pad == K_pad) return it->second;
        Workspace ws;
        ws.M = M;
        ws.K_pad = K_pad;
        ws.qx = torch::empty({M, K_pad}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt8));
        ws.xscale = torch::empty({M, ng}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        ws.xsum = torch::empty({M, ng}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
        auto res = workspaces.emplace(M, std::move(ws));
        return res.first->second;
    }

    void ensure_argmax_workspace(Workspace & ws) const {
        int nb = (int)((out + 3) / 4);
        if (ws.argmax_vals.defined() && ws.argmax_blocks >= nb) return;
        ws.argmax_blocks = nb;
        ws.argmax_vals = torch::empty({nb}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        ws.argmax_idxs = torch::empty({nb}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    }

    void ensure_output_workspace(Workspace & ws) const {
        if (ws.out_buf.defined() && ws.out_buf.size(0) == 1 && ws.out_buf.size(1) == out) return;
        ws.out_buf = torch::empty({1, out}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
    }

    void ensure_rinv_workspace(Workspace & ws, int64_t n) const {
        if (ws.rinv.defined() && ws.rinv.numel() >= n) return;
        ws.rinv = torch::empty({n}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    }

    void ensure_mmq_partial_workspace(Workspace & ws, int64_t n) const {
        if (ws.mmq_partial.defined() && ws.mmq_partial.numel() >= n) return;
        ws.mmq_partial = torch::empty({n}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    }

    void ensure_mmq_qx_workspace(Workspace & ws, int M) const {
        const int kStride = bits == 2 && gs == 16 ? 36 : 68;
        int m_pad = ((M + 15) / 16) * 16;
        int nchunks = ((int)ng + 7) / 8;
        int64_t n = (int64_t)nchunks * m_pad * kStride;
        if (ws.mmq_qx.defined() && ws.mmq_qx.numel() >= n) return;
        ws.mmq_qx = torch::empty({n}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    }
};

struct NintPrefillWorkspaceSlot {
    int64_t ng = 0;
    int64_t kstride = 0;
    uint64_t last_use = 0;
    Workspace workspace;
};

struct NintPrefillWorkspaceRing {
    uint64_t clock = 0;
    std::vector<NintPrefillWorkspaceSlot> slots;
};

struct NintPrefillStreamKey {
    int device = 0;
    std::uintptr_t stream = 0;

    bool operator==(const NintPrefillStreamKey & other) const {
        return device == other.device && stream == other.stream;
    }
};

struct NintPrefillStreamKeyHash {
    size_t operator()(const NintPrefillStreamKey & key) const {
        const size_t device_hash = std::hash<int>{}(key.device);
        const size_t stream_hash =
            std::hash<std::uintptr_t>{}(key.stream);
        return device_hash ^
            (stream_hash + 0x9e3779b9U +
             (device_hash << 6) + (device_hash >> 2));
    }
};

static Workspace & nint_prefill_mmq_workspace(
        const NintWeight & weight,
        const torch::Tensor & x,
        int M) {
    constexpr size_t kWorkspaceSlots = 3;
    thread_local std::unordered_map<
        NintPrefillStreamKey,
        NintPrefillWorkspaceRing,
        NintPrefillStreamKeyHash> rings;

    const auto cuda_stream =
        at::cuda::getCurrentCUDAStream().stream();
    const NintPrefillStreamKey key{
        x.get_device(),
        reinterpret_cast<std::uintptr_t>(cuda_stream)};
    auto & ring = rings[key];
    const uint64_t use = ++ring.clock;
    const int64_t kstride =
        weight.bits == 2 && weight.gs == 16 ? 36 : 68;

    NintPrefillWorkspaceSlot * selected = nullptr;
    for (auto & slot : ring.slots) {
        if (slot.ng == weight.ng && slot.kstride == kstride) {
            selected = &slot;
            break;
        }
    }
    if (selected == nullptr && ring.slots.size() < kWorkspaceSlots) {
        ring.slots.emplace_back();
        selected = &ring.slots.back();
    }
    if (selected == nullptr) {
        selected = &*std::min_element(
            ring.slots.begin(), ring.slots.end(),
            [](const auto & lhs, const auto & rhs) {
                return lhs.last_use < rhs.last_use;
            });
        selected->workspace = Workspace{};
    }

    selected->ng = weight.ng;
    selected->kstride = kstride;
    selected->last_use = use;
    Workspace & workspace = selected->workspace;
    workspace.M = std::max(workspace.M, M);
    workspace.K_pad = static_cast<int>(weight.ng * weight.gs);
    const int64_t rows = std::max<int64_t>(
        M,
        workspace.xscale.defined()
            ? workspace.xscale.size(0)
            : 0);
    if (!workspace.xscale.defined() ||
            workspace.xscale.size(0) < M ||
            workspace.xscale.size(1) != weight.ng) {
        const auto float_options =
            x.options().dtype(torch::kFloat32);
        workspace.xscale = torch::empty(
            {rows, weight.ng}, float_options);
        workspace.xsum = torch::empty(
            {rows, weight.ng},
            x.options().dtype(torch::kInt32));
    }
    return workspace;
}

static const char * nint6_mmq_mode_name(Nint6MmqMode mode) {
    return mode == Nint6MmqMode::Int8 ? "int8" : "fp16";
}

static NintWeight to_device_nint(const NintCpu & c, bool cuda) {
    if (c.bits < 1 || c.bits > 8) throw std::runtime_error("unsupported NINT bits");
    int qbytes = (c.gs * c.bits + 7) / 8;
    if (qbytes != c.qbytes) throw std::runtime_error("NINT qbytes mismatch");
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = c.gs;
    w.bits = c.bits;
    w.qbytes = qbytes;
    w.neuron_len = c.neuron_len;
    w.shape = c.shape;
    w.q_packed = cpu_u8_tensor(c.q_packed, {c.out, c.ng, qbytes});
    w.sub_scale = cpu_u8_tensor(c.sub_scale, {c.out, c.ng});
    w.sub_min = cpu_u8_tensor(c.sub_min, {c.out, c.ng});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    w.neuron_min = cpu_f16_to_f32_tensor(c.neuron_min_h, c.out);
    if (cuda) {
        w.q_packed = w.q_packed.to(torch::kCUDA).contiguous();
        w.sub_scale = w.sub_scale.to(torch::kCUDA).contiguous();
        w.sub_min = w.sub_min.to(torch::kCUDA).contiguous();
        w.neuron_scale = w.neuron_scale.to(torch::kCUDA).contiguous();
        w.neuron_min = w.neuron_min.to(torch::kCUDA).contiguous();
    }
    return w;
}

static NintWeight to_gpu_nint(const NintCpu & c) {
    return to_device_nint(c, true);
}

static NintWeight to_cuda_device_nint(
        const NintCpu & c, int device) {
    c10::cuda::CUDAGuard guard(device);
    return to_device_nint(c, true);
}

static NintWeight to_cpu_nint(const NintCpu & c) {
    return to_device_nint(c, false);
}

static NintWeight to_device_nint8_zero(
        const Nint8ZeroCpu & c, bool cuda) {
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = 32;
    w.bits = 8;
    w.qbytes = 32;
    w.neuron_len = c.neuron_len;
    w.q8_zero = true;
    w.shape = c.shape;
    w.q_packed = cpu_u8_tensor(c.q, {c.out, c.ng, 32});
    w.q8_zero_scale = cpu_f16_tensor(c.scale_h, {c.out, c.ng});
    if (cuda) {
        w.q_packed = w.q_packed.to(torch::kCUDA).contiguous();
        w.q8_zero_scale = w.q8_zero_scale.to(torch::kCUDA).contiguous();
    }
    return w;
}

static NintWeight to_gpu_nint8_zero(const Nint8ZeroCpu & c) {
    return to_device_nint8_zero(c, true);
}

static NintWeight to_cuda_device_nint8_zero(
        const Nint8ZeroCpu & c, int device) {
    c10::cuda::CUDAGuard guard(device);
    return to_device_nint8_zero(c, true);
}

static NintWeight to_cpu_nint8_zero(const Nint8ZeroCpu & c) {
    return to_device_nint8_zero(c, false);
}

static NintWeight load_nint_gpu(const MfqFile & mfq, const std::string & name) {
    if (mfq.record(name).dtype == "NINT8-0") {
        return to_gpu_nint8_zero(unpack_nint8_zero(mfq.read_blob(name)));
    }
    return to_gpu_nint(unpack_nint(mfq.read_blob(name)));
}

struct NintMoeCpuPool {
    std::vector<int32_t> expert_ids;
    std::string dtype;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> runtime_payload;
    NintCpu weight;
    Nint8ZeroCpu q8_zero;
};

struct NintMoeCpu {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<NintMoeCpuPool> pools;
};

static NintMoeCpu unpack_nint_moe_impl(
        const std::vector<uint8_t> & blob,
        bool allow_partial) {
    if (blob.size() < 20) {
        throw std::runtime_error("invalid NINTM header");
    }
    const bool version1 = std::memcmp(blob.data(), "NIM1", 4) == 0;
    const bool delta = std::memcmp(blob.data(), "NID2", 4) == 0;
    const bool version2 =
        std::memcmp(blob.data(), "NIM2", 4) == 0 || delta;
    if (!version1 && !version2) throw std::runtime_error("invalid NINTM header");
    if (delta && !allow_partial) {
        throw std::runtime_error("NINTM delta cannot be loaded as a full tensor");
    }
    size_t off = 4;
    NintMoeCpu result;
    result.n_experts = (int)read_u32_from(blob, off);
    result.out_per_expert = (int)read_u32_from(blob, off);
    result.neuron_len = (int)read_u32_from(blob, off);
    int pool_count = (int)read_u32_from(blob, off);
    if (result.n_experts <= 0 || result.out_per_expert <= 0 || result.neuron_len <= 0 ||
        pool_count <= 0 || pool_count > result.n_experts) {
        throw std::runtime_error("invalid NINTM dimensions");
    }
    std::vector<int> owners((size_t)result.n_experts, -1);
    result.pools.reserve((size_t)pool_count);
    for (int pool_index = 0; pool_index < pool_count; ++pool_index) {
        const size_t pool_header_bytes = version1 ? 12 : 24;
        if (off + pool_header_bytes > blob.size()) {
            throw std::runtime_error("truncated NINTM pool header");
        }
        uint32_t expert_count = read_u32_from(blob, off);
        uint32_t dtype_nbytes = version1 ? 0 : read_u32_from(blob, off);
        uint64_t payload_nbytes = read_u64_from(blob, off);
        uint64_t runtime_nbytes = version1 ? 0 : read_u64_from(blob, off);
        size_t ids_nbytes = (size_t)expert_count * sizeof(int32_t);
        if (expert_count == 0 || off + ids_nbytes > blob.size() ||
            dtype_nbytes > 32 ||
            dtype_nbytes + runtime_nbytes > blob.size() - off - ids_nbytes ||
            payload_nbytes > blob.size() - off - ids_nbytes -
                dtype_nbytes - runtime_nbytes) {
            throw std::runtime_error("truncated NINTM pool payload");
        }
        NintMoeCpuPool pool;
        pool.expert_ids.resize(expert_count);
        std::memcpy(pool.expert_ids.data(), blob.data() + off, ids_nbytes);
        off += ids_nbytes;
        for (int32_t expert : pool.expert_ids) {
            if (expert < 0 || expert >= result.n_experts || owners[(size_t)expert] >= 0) {
                throw std::runtime_error("invalid or duplicate NINTM expert id");
            }
            owners[(size_t)expert] = pool_index;
        }
        if (version2) {
            if (dtype_nbytes == 0) throw std::runtime_error("empty NINTM pool dtype");
            pool.dtype.assign(
                reinterpret_cast<const char *>(blob.data() + off), dtype_nbytes);
            off += dtype_nbytes;
            pool.runtime_payload.assign(
                blob.begin() + (ptrdiff_t)off,
                blob.begin() + (ptrdiff_t)(off + (size_t)runtime_nbytes));
            off += (size_t)runtime_nbytes;
        }
        size_t payload_end = off + (size_t)payload_nbytes;
        pool.payload.assign(
            blob.begin() + (ptrdiff_t)off, blob.begin() + (ptrdiff_t)payload_end);
        off = payload_end;
        if (!version1 && pool.dtype == "NINT8-0") {
            pool.q8_zero = unpack_nint8_zero(pool.payload);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.q8_zero.axis != 0 ||
                pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != result.neuron_len ||
                pool.q8_zero.shape.size() != 2 ||
                pool.q8_zero.shape[0] != expected_rows ||
                pool.q8_zero.shape[1] != result.neuron_len) {
                throw std::runtime_error(
                    "NINTM NINT8-0 pool weight shape mismatch");
            }
        } else if (version1 ||
                   (pool.dtype != "NINTM" &&
                    pool.dtype.rfind("NINT", 0) == 0)) {
            pool.weight = unpack_nint(pool.payload);
            if (version1) pool.dtype = "NINT" + std::to_string(pool.weight.bits);
            int expected_rows = (int)expert_count * result.out_per_expert;
            if (pool.weight.axis != 0 || pool.weight.out != expected_rows ||
                pool.weight.neuron_len != result.neuron_len ||
                pool.weight.shape.size() != 2 ||
                pool.weight.shape[0] != expected_rows ||
                pool.weight.shape[1] != result.neuron_len) {
                throw std::runtime_error("NINTM pool weight shape mismatch");
            }
        }
        result.pools.push_back(std::move(pool));
    }
    if (off != blob.size() ||
        (!allow_partial &&
         std::find(owners.begin(), owners.end(), -1) != owners.end())) {
        throw std::runtime_error("NINTM expert coverage or tail mismatch");
    }
    return result;
}

static NintMoeCpu unpack_nint_moe(const std::vector<uint8_t> & blob) {
    return unpack_nint_moe_impl(blob, false);
}

static NintMoeCpu unpack_nint_moe_delta(
        const std::vector<uint8_t> & blob) {
    return unpack_nint_moe_impl(blob, true);
}

struct NintMoePoolWeight {
    NintWeight weight;
    torch::Tensor expert_local;
    int local_experts = 0;
    int profile_code = -1;
};

static int nint_moe_profile_code(int bits, int gs) {
    if (bits == 2 && gs == 16) return 6;
    if (bits == 3 && gs == 24) return 5;
    if (bits == 4 && gs == 24) return 0;
    if (bits == 5 && gs == 28) return 1;
    if (bits == 6 && gs == 24) return 2;
    if (bits == 8 && gs == 48) return 3;
    if (bits == 8 && gs == 24) return 4;
    return -1;
}

struct MoeActivationKey {
    int input_rows = 0;
    int groups = 0;
    int gs = 0;
    int device = 0;

    bool operator==(const MoeActivationKey & other) const {
        return input_rows == other.input_rows && groups == other.groups &&
               gs == other.gs && device == other.device;
    }
};

struct MoeActivationKeyHash {
    size_t operator()(const MoeActivationKey & key) const {
        size_t value = (size_t)key.input_rows;
        value = value * 1315423911u + (size_t)key.groups;
        value = value * 1315423911u + (size_t)key.gs;
        return value * 1315423911u + (size_t)key.device;
    }
};

struct MoeActivationWorkspace {
    torch::Tensor qx;
    torch::Tensor xscale;
};

struct MoeHeteroWorkspace {
    std::vector<torch::Tensor> qx;
    std::vector<torch::Tensor> xscale;
    torch::Tensor activation_ptrs;
};

struct MoeRoutePlan {
    torch::Tensor ids;
    torch::Tensor ids_dst;
    torch::Tensor expert_bounds;
    torch::Tensor tile_bounds;
    torch::Tensor tile_experts;
    torch::Tensor counts;
    torch::Tensor cursors;
    int n_experts = 0;
    bool map_ready = false;
    uint64_t generation = 0;
    mutable std::shared_ptr<std::vector<int32_t>>
        host_unique_experts;
};

static std::atomic<uint64_t> g_moe_route_generation{1};

static torch::Tensor moe_tensor_to_device(
        torch::Tensor value, int device) {
    if (!value.defined()) return value;
    if (value.is_cuda() && value.get_device() == device) {
        return value.contiguous();
    }
    c10::cuda::CUDAGuard guard(device);
    return value.to(
        value.options().device(
            torch::Device(torch::kCUDA, device)),
        true, false).contiguous();
}

struct MoeRouteReplicaEntry {
    uint64_t generation = 0;
    MoeRoutePlan plan;
};

class MoeRouteReplicaCache {
public:
    const MoeRoutePlan & get(
            const MoeRoutePlan & source, int device) {
        auto & entry = entries_[device];
        if (entry.generation == source.generation &&
                entry.plan.generation == source.generation) {
            return entry.plan;
        }
        copy_plan(entry.plan, source, device);
        entry.generation = source.generation;
        return entry.plan;
    }

private:
    static void copy_tensor(
            torch::Tensor & destination,
            const torch::Tensor & source,
            int device) {
        if (!source.defined()) {
            destination = torch::Tensor();
            return;
        }
        if (source.is_cuda() && source.get_device() == device) {
            destination = source.contiguous();
            return;
        }
        c10::cuda::CUDAGuard guard(device);
        auto options = source.options().device(
            torch::Device(torch::kCUDA, device));
        if (!destination.defined() ||
                destination.sizes() != source.sizes() ||
                destination.scalar_type() != source.scalar_type() ||
                !destination.is_cuda() ||
                destination.get_device() != device) {
            destination = torch::empty(source.sizes(), options);
        }
        destination.copy_(source, true);
    }

    static void copy_plan(
            MoeRoutePlan & destination,
            const MoeRoutePlan & source,
            int device) {
        copy_tensor(destination.ids, source.ids, device);
        copy_tensor(destination.ids_dst, source.ids_dst, device);
        copy_tensor(destination.expert_bounds, source.expert_bounds, device);
        copy_tensor(destination.tile_bounds, source.tile_bounds, device);
        copy_tensor(destination.tile_experts, source.tile_experts, device);
        copy_tensor(destination.counts, source.counts, device);
        copy_tensor(destination.cursors, source.cursors, device);
        destination.n_experts = source.n_experts;
        destination.map_ready = source.map_ready;
        destination.generation = source.generation;
        destination.host_unique_experts = source.host_unique_experts;
    }

    std::unordered_map<int, MoeRouteReplicaEntry> entries_;
};

static thread_local MoeRouteReplicaCache g_moe_route_replica_cache;

static MoeRoutePlan moe_route_to_device(
        const MoeRoutePlan & source, int device) {
    return g_moe_route_replica_cache.get(source, device);
}

static MoeRoutePlan build_moe_route_plan(torch::Tensor ids, int n_experts) {
    if (!ids.is_cuda() || ids.scalar_type() != torch::kInt32 || ids.dim() != 2) {
        throw std::runtime_error("MoE ids must be CUDA int32 [tokens, routes]");
    }
    MoeRoutePlan result;
    result.generation = g_moe_route_generation.fetch_add(
        1, std::memory_order_relaxed);
    result.ids = ids.contiguous();
    result.n_experts = n_experts;
    auto empty = torch::empty({0}, result.ids.options());
    result.ids_dst = empty;
    result.expert_bounds = empty;
    result.tile_bounds = empty;
    result.tile_experts = empty;
    result.counts = empty;
    result.cursors = empty;
    if (result.ids.size(0) > 8) {
        auto mapped = moe_build_expert_map_cuda(result.ids, n_experts, 8);
        result.ids_dst = mapped.at(0);
        result.expert_bounds = mapped.at(1);
        result.tile_bounds = mapped.at(2);
        result.tile_experts = mapped.at(3);
        result.counts = mapped.at(4);
    }
    result.map_ready = result.ids.size(0) <= 8 ||
        result.ids_dst.numel() == result.ids.numel();
    return result;
}

static torch::Tensor reduce_tensor_parallel_outputs(
    std::vector<torch::Tensor> outputs);

struct NintMoeWeight {
    struct TensorParallelShard {
        int device = 0;
        int64_t output_begin = 0;
        int64_t output_end = 0;
        std::shared_ptr<NintMoeWeight> weight;
    };

    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<NintMoePoolWeight> pools;
    bool hetero_supported = false;
    torch::Tensor weight_ptrs;
    torch::Tensor pool_params;
    torch::Tensor expert_pool;
    torch::Tensor expert_local;
    std::vector<int> quantize_pool_indices;
    int gs24_quantize_index = -1;
    int gs28_quantize_index = -1;
    int profile_mask = 0;
    int64_t mixed_weight_bytes = 0;
    bool tensor_parallel_paired_output = false;
    bool tensor_parallel_experts = false;
    bool partial_experts = false;
    std::vector<TensorParallelShard>
        tensor_parallel_shards;
    std::function<void(const MoeRoutePlan &)> cache_prefetch;
    std::function<torch::Tensor(torch::Tensor, const MoeRoutePlan &)> mixed_forward;
    std::function<torch::Tensor(
        torch::Tensor, const MoeRoutePlan &, bool)> mixed_glu_output_forward;
    std::function<torch::Tensor(
        torch::Tensor, const MoeRoutePlan &, bool)> mixed_glu_forward;
    std::function<torch::Tensor(
        torch::Tensor, const MoeRoutePlan &, double)> mixed_clamped_swiglu_forward;
    mutable std::unordered_map<MoeActivationKey, MoeActivationWorkspace, MoeActivationKeyHash>
        activation_workspaces;
    mutable std::unordered_map<int, MoeHeteroWorkspace> hetero_workspaces;

    MoeActivationWorkspace & activation_workspace(
            torch::Tensor x, int input_rows, int groups, int gs) const {
        MoeActivationKey key{input_rows, groups, gs, x.get_device()};
        auto it = activation_workspaces.find(key);
        if (it != activation_workspaces.end()) return it->second;
        MoeActivationWorkspace workspace;
        workspace.qx = torch::empty(
            {input_rows, groups * gs}, x.options().dtype(torch::kInt8));
        workspace.xscale = torch::empty(
            {input_rows, groups}, x.options().dtype(torch::kFloat32));
        return activation_workspaces.emplace(key, std::move(workspace)).first->second;
    }

    MoeHeteroWorkspace & hetero_workspace(torch::Tensor x, int input_rows) const {
        auto found = hetero_workspaces.find(input_rows);
        if (found != hetero_workspaces.end()) return found->second;
        MoeHeteroWorkspace workspace;
        workspace.qx.reserve(pools.size());
        workspace.xscale.reserve(pools.size());
        std::vector<int64_t> pointers;
        pointers.reserve(pools.size() * 2);
        for (const auto & pool : pools) {
            const int groups = static_cast<int>(pool.weight.ng);
            const int gs = static_cast<int>(pool.weight.gs);
            auto & activation = activation_workspace(x, input_rows, groups, gs);
            workspace.qx.push_back(activation.qx);
            workspace.xscale.push_back(activation.xscale);
            pointers.push_back(static_cast<int64_t>(
                reinterpret_cast<uintptr_t>(activation.qx.data_ptr<int8_t>())));
            pointers.push_back(static_cast<int64_t>(
                reinterpret_cast<uintptr_t>(activation.xscale.data_ptr<float>())));
        }
        workspace.activation_ptrs = torch::from_blob(
            pointers.data(), {static_cast<int64_t>(pools.size()), 2},
            torch::TensorOptions().dtype(torch::kInt64)).clone().to(torch::kCUDA).contiguous();
        return hetero_workspaces.emplace(input_rows, std::move(workspace)).first->second;
    }

    bool supports_dual_quant() const {
        return quantize_pool_indices.size() == 2 &&
            gs24_quantize_index >= 0 && gs28_quantize_index >= 0;
    }

    bool tensor_parallel() const {
        return !tensor_parallel_shards.empty();
    }

    template <typename Forward>
    torch::Tensor forward_tensor_parallel(
            torch::Tensor x,
            const MoeRoutePlan & route,
            bool paired_output,
            Forward && forward) const {
        std::vector<torch::Tensor> outputs;
        outputs.reserve(tensor_parallel_shards.size());
        for (const auto & shard : tensor_parallel_shards) {
            if (!shard.weight) {
                throw std::runtime_error(
                    "tensor-parallel MoE shard is missing");
            }
            c10::cuda::CUDAGuard guard(shard.device);
            auto local_x =
                moe_tensor_to_device(x, shard.device);
            auto local_route =
                moe_route_to_device(route, shard.device);
            outputs.push_back(
                forward(
                    *shard.weight,
                    local_x,
                    local_route));
        }
        const int primary =
            g_tensor_parallel.primary_device();
        c10::cuda::CUDAGuard primary_guard(primary);
        if (!paired_output) {
            for (auto & output : outputs) {
                output =
                    moe_tensor_to_device(output, primary);
            }
            return torch::cat(outputs, -1).contiguous();
        }
        std::vector<torch::Tensor> first;
        std::vector<torch::Tensor> second;
        first.reserve(outputs.size());
        second.reserve(outputs.size());
        for (auto & output : outputs) {
            output =
                moe_tensor_to_device(output, primary);
            if (output.size(-1) % 2 != 0) {
                throw std::runtime_error(
                    "paired tensor-parallel MoE shard "
                    "has an odd output width");
            }
            const int64_t half =
                output.size(-1) / 2;
            first.push_back(
                output.narrow(-1, 0, half));
            second.push_back(
                output.narrow(-1, half, half));
        }
        return torch::cat(
            {torch::cat(first, -1),
             torch::cat(second, -1)},
            -1).contiguous();
    }

    void prefetch(const MoeRoutePlan & route) const {
        if (cache_prefetch) cache_prefetch(route);
    }

    torch::Tensor forward(
            torch::Tensor x,
            const MoeRoutePlan & route) const {
        if (tensor_parallel()) {
            if (tensor_parallel_experts) {
                std::vector<torch::Tensor> partials;
                partials.reserve(tensor_parallel_shards.size());
                for (const auto & shard : tensor_parallel_shards) {
                    c10::cuda::CUDAGuard guard(shard.device);
                    auto local_x = moe_tensor_to_device(x, shard.device);
                    auto local_route = moe_route_to_device(route, shard.device);
                    partials.push_back(
                        shard.weight->forward(local_x, local_route));
                }
                return reduce_tensor_parallel_outputs(
                    std::move(partials));
            }
            return forward_tensor_parallel(
                x, route,
                tensor_parallel_paired_output,
                [](const NintMoeWeight & shard,
                   torch::Tensor local_x,
                   const MoeRoutePlan & local_route) {
                    return shard.forward(
                        local_x, local_route);
                });
        }
        if (mixed_forward) return mixed_forward(x, route);
        if (!x.is_cuda() || !x.is_contiguous() || x.scalar_type() != torch::kFloat16 ||
            (x.dim() != 2 && x.dim() != 3) || x.size(-1) != neuron_len) {
            throw std::runtime_error("NINTM input must be contiguous CUDA f16 with exact K");
        }
        int tokens = (int)route.ids.size(0);
        int routes = (int)route.ids.size(1);
        if (route.n_experts != n_experts || x.size(0) != tokens ||
            (x.dim() == 3 && x.size(1) != routes)) {
            throw std::runtime_error("NINTM input and route shape mismatch");
        }
        int input_rows = x.dim() == 3 ? tokens * routes : tokens;
        auto output = partial_experts
            ? torch::zeros(
                {tokens, routes, out_per_expert},
                x.options().dtype(torch::kFloat16))
            : torch::empty(
                {tokens, routes, out_per_expert},
                x.options().dtype(torch::kFloat16));
        if (g_kl_mmq_mode != KlMmqMode::Default) {
            TORCH_CHECK(
                hetero_supported && route.map_ready &&
                route.ids_dst.numel() == route.ids.numel(),
                "KLD common MMQ requires the routed packed-FP16 path");
            auto prepared = kl_mmq_prepare_activation(x);
            ++g_kl_mmq_moe_calls;
            return nint_moe_grouped_matmul_hetero_f16_cuda(
                weight_ptrs, pool_params, expert_pool, expert_local,
                prepared, route.ids, n_experts, out_per_expert,
                neuron_len, x.dim() == 3, output, route.ids_dst,
                route.expert_bounds, route.tile_bounds,
                route.tile_experts);
        }
        static const bool disable_prefill_mma = [] {
            const char * value = std::getenv("MFQ_DISABLE_MOE_PREFILL_MMA");
            return value != nullptr && std::atoi(value) != 0;
        }();
        static const int prefill_mma_min_tokens = [] {
            const char * value = std::getenv("MFQ_MOE_PREFILL_MMA_MIN_TOKENS");
            return value == nullptr ? 256 : std::max(9, std::atoi(value));
        }();
        if (!disable_prefill_mma && !g_force_moe_prefill_mma_off && hetero_supported &&
                tokens >= prefill_mma_min_tokens &&
                route.map_ready && route.ids_dst.numel() == route.ids.numel()) {
            return nint_moe_grouped_matmul_hetero_f16_cuda(
                weight_ptrs, pool_params, expert_pool, expert_local, x, route.ids,
                n_experts, out_per_expert, neuron_len, x.dim() == 3, output,
                route.ids_dst, route.expert_bounds, route.tile_bounds, route.tile_experts);
        }
        if (hetero_supported && moe_small_hetero_enabled(tokens) &&
                hetero_workspaces.find(input_rows) == hetero_workspaces.end()) {
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            MFQ_CUDA_CHECK(cudaStreamIsCapturing(
                at::cuda::getCurrentCUDAStream().stream(), &capture_status));
            if (capture_status == cudaStreamCaptureStatusNone) {
                hetero_workspace(x, input_rows);
            }
        }
        static const int hetero_min_k = [] {
            const char * value = std::getenv("MFQ_MOE_HETERO_MIN_K");
            return value == nullptr ? 0 : std::max(0, std::atoi(value));
        }();
        bool use_hetero = !g_force_moe_pool_path && hetero_supported &&
            moe_small_hetero_enabled(tokens) && routes <= 8 &&
            neuron_len >= hetero_min_k;
        if (use_hetero) {
            auto & workspace = hetero_workspace(x, input_rows);
            static const bool disable_dual_quant = [] {
                const char * value = std::getenv("MFQ_DISABLE_MOE_DUAL_QUANT");
                return value != nullptr && std::atoi(value) != 0;
            }();
            if (!disable_dual_quant && supports_dual_quant()) {
                nint_moe_quantize_24_28_ws_cuda(
                    x,
                    workspace.qx.at(static_cast<size_t>(gs24_quantize_index)),
                    workspace.xscale.at(static_cast<size_t>(gs24_quantize_index)),
                    workspace.qx.at(static_cast<size_t>(gs28_quantize_index)),
                    workspace.xscale.at(static_cast<size_t>(gs28_quantize_index)));
            } else {
                for (int pool_index : quantize_pool_indices) {
                    const auto & pool = pools.at(static_cast<size_t>(pool_index));
                    nint_moe_quantize_input_ws_cuda(
                        x, pool.weight.gs, workspace.qx.at(static_cast<size_t>(pool_index)),
                        workspace.xscale.at(static_cast<size_t>(pool_index)));
                }
            }
            return nint_moe_grouped_matmul_hetero_qx_cuda(
                weight_ptrs, pool_params, workspace.activation_ptrs, expert_pool, expert_local,
                route.ids, profile_mask, n_experts, out_per_expert, neuron_len, x.dim() == 3, output,
                route.ids_dst, route.expert_bounds, route.tile_bounds, route.tile_experts);
        }
        std::unordered_set<MoeActivationKey, MoeActivationKeyHash> quantized;
        for (const auto & pool : pools) {
            int groups = (int)pool.weight.ng;
            int gs = (int)pool.weight.gs;
            MoeActivationKey key{input_rows, groups, gs, x.get_device()};
            auto & workspace = activation_workspace(x, input_rows, groups, gs);
            bool input_quantized = quantized.find(key) != quantized.end();
            nint_moe_grouped_matmul_pool_ws_cuda(
                pool.weight.q_packed, pool.weight.sub_scale, pool.weight.sub_min,
                pool.weight.neuron_scale, pool.weight.neuron_min, x, route.ids,
                pool.expert_local, n_experts, pool.local_experts, out_per_expert,
                gs, pool.weight.bits, route.map_ready, input_quantized, output,
                workspace.qx, workspace.xscale, route.counts, route.cursors,
                route.ids_dst, route.expert_bounds, route.tile_bounds, route.tile_experts);
            quantized.insert(key);
        }
        return output;
    }

    torch::Tensor forward_glu_output(
            torch::Tensor x, const MoeRoutePlan & route, bool gelu) const {
        if (tensor_parallel()) {
            return forward_tensor_parallel(
                x, route, false,
                [gelu](
                    const NintMoeWeight & shard,
                    torch::Tensor local_x,
                    const MoeRoutePlan & local_route) {
                    return shard.forward_glu_output(
                        local_x, local_route, gelu);
                });
        }
        if (mixed_glu_output_forward) {
            return mixed_glu_output_forward(x, route, gelu);
        }
        if (!hetero_supported || !x.is_cuda() || !x.is_contiguous() ||
            x.scalar_type() != torch::kFloat16 || x.dim() != 2 ||
            x.size(0) < 1 || x.size(0) > 4 ||
            x.size(1) != neuron_len || out_per_expert <= 0 || out_per_expert % 2 != 0 ||
            route.ids.size(0) != x.size(0) || route.n_experts != n_experts) {
            throw std::runtime_error(
                "fused NINTM gate/up GLU requires 1-4 contiguous CUDA f16 tokens");
        }
        const int tokens = static_cast<int>(x.size(0));
        const int hidden_width = out_per_expert / 2;
        const int routes = static_cast<int>(route.ids.size(1));
        auto output = partial_experts
            ? torch::zeros(
                {tokens, routes, hidden_width},
                x.options().dtype(torch::kFloat16))
            : torch::empty(
                {tokens, routes, hidden_width},
                x.options().dtype(torch::kFloat16));
        auto & workspace = hetero_workspace(x, tokens);
        if (supports_dual_quant()) {
            nint_moe_quantize_24_28_ws_cuda(
                x,
                workspace.qx.at(static_cast<size_t>(gs24_quantize_index)),
                workspace.xscale.at(static_cast<size_t>(gs24_quantize_index)),
                workspace.qx.at(static_cast<size_t>(gs28_quantize_index)),
                workspace.xscale.at(static_cast<size_t>(gs28_quantize_index)));
        } else {
            for (int pool_index : quantize_pool_indices) {
                const auto & pool = pools.at(static_cast<size_t>(pool_index));
                nint_moe_quantize_input_ws_cuda(
                    x, pool.weight.gs,
                    workspace.qx.at(static_cast<size_t>(pool_index)),
                    workspace.xscale.at(static_cast<size_t>(pool_index)));
            }
        }
        return nint_moe_grouped_matmul_hetero_glu_qx_cuda(
            weight_ptrs, pool_params, workspace.activation_ptrs,
            expert_pool, expert_local, route.ids, profile_mask, n_experts,
            hidden_width, gelu, output);
    }

    torch::Tensor forward_glu(
            torch::Tensor gate_up, const MoeRoutePlan & route, bool gelu) const {
        if (tensor_parallel()) {
            return forward_tensor_parallel(
                gate_up, route, false,
                [gelu](
                    const NintMoeWeight & shard,
                    torch::Tensor local_gate_up,
                    const MoeRoutePlan & local_route) {
                    return shard.forward_glu(
                        local_gate_up,
                        local_route, gelu);
                });
        }
        if (mixed_glu_forward) {
            return mixed_glu_forward(gate_up, route, gelu);
        }
        if (!hetero_supported || !gate_up.is_cuda() || !gate_up.is_contiguous() ||
            gate_up.scalar_type() != torch::kFloat16 || gate_up.dim() != 3 ||
            gate_up.size(2) != 2 * neuron_len) {
            throw std::runtime_error("fused NINTM GLU input has an unsupported layout");
        }
        const int tokens = static_cast<int>(route.ids.size(0));
        const int routes = static_cast<int>(route.ids.size(1));
        if (tokens > 8 || routes > 8 || route.n_experts != n_experts ||
            gate_up.size(0) != tokens || gate_up.size(1) != routes) {
            throw std::runtime_error("fused NINTM GLU only supports M<=8 and up to eight routes");
        }
        const int input_rows = tokens * routes;
        auto output = partial_experts
            ? torch::zeros(
                {tokens, routes, out_per_expert},
                gate_up.options().dtype(torch::kFloat16))
            : torch::empty(
                {tokens, routes, out_per_expert},
                gate_up.options().dtype(torch::kFloat16));
        auto & workspace = hetero_workspace(gate_up, input_rows);
        if (supports_dual_quant() && neuron_len <= 4096) {
            if (gelu) {
                nint_moe_quantize_geglu_24_28_ws_cuda(
                    gate_up,
                    workspace.qx.at(static_cast<size_t>(gs24_quantize_index)),
                    workspace.xscale.at(static_cast<size_t>(gs24_quantize_index)),
                    workspace.qx.at(static_cast<size_t>(gs28_quantize_index)),
                    workspace.xscale.at(static_cast<size_t>(gs28_quantize_index)));
            } else {
                nint_moe_quantize_swiglu_24_28_ws_cuda(
                    gate_up,
                    workspace.qx.at(static_cast<size_t>(gs24_quantize_index)),
                    workspace.xscale.at(static_cast<size_t>(gs24_quantize_index)),
                    workspace.qx.at(static_cast<size_t>(gs28_quantize_index)),
                    workspace.xscale.at(static_cast<size_t>(gs28_quantize_index)));
            }
        } else {
            for (int pool_index : quantize_pool_indices) {
                const auto & pool = pools.at(static_cast<size_t>(pool_index));
                if (gelu) {
                    nint_moe_quantize_geglu_input_ws_cuda(
                        gate_up, pool.weight.gs, workspace.qx.at(static_cast<size_t>(pool_index)),
                        workspace.xscale.at(static_cast<size_t>(pool_index)));
                } else {
                    nint_moe_quantize_swiglu_input_ws_cuda(
                        gate_up, pool.weight.gs, workspace.qx.at(static_cast<size_t>(pool_index)),
                        workspace.xscale.at(static_cast<size_t>(pool_index)));
                }
            }
        }
        return nint_moe_grouped_matmul_hetero_qx_cuda(
            weight_ptrs, pool_params, workspace.activation_ptrs, expert_pool, expert_local,
            route.ids, profile_mask, n_experts, out_per_expert, neuron_len, true, output,
            route.ids_dst, route.expert_bounds, route.tile_bounds, route.tile_experts);
    }

    torch::Tensor forward_swiglu(torch::Tensor gate_up, const MoeRoutePlan & route) const {
        return forward_glu(gate_up, route, false);
    }

    bool supports_clamped_swiglu() const {
        if (tensor_parallel()) {
            return std::all_of(
                tensor_parallel_shards.begin(),
                tensor_parallel_shards.end(),
                [](const TensorParallelShard & shard) {
                    return shard.weight &&
                        shard.weight
                            ->supports_clamped_swiglu();
                });
        }
        return static_cast<bool>(mixed_clamped_swiglu_forward);
    }

    torch::Tensor forward_clamped_swiglu(
            torch::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) const {
        if (tensor_parallel()) {
            return forward_tensor_parallel(
                gate_up, route, false,
                [limit](
                    const NintMoeWeight & shard,
                    torch::Tensor local_gate_up,
                    const MoeRoutePlan & local_route) {
                    return shard.forward_clamped_swiglu(
                        local_gate_up,
                        local_route, limit);
                });
        }
        if (!mixed_clamped_swiglu_forward) {
            throw std::runtime_error(
                "clamped SwiGLU is unavailable for this NINTM tensor");
        }
        return mixed_clamped_swiglu_forward(gate_up, route, limit);
    }

    torch::Tensor forward_geglu(torch::Tensor gate_up, const MoeRoutePlan & route) const {
        return forward_glu(gate_up, route, true);
    }
};

static void initialize_nint_moe_dispatch(
        NintMoeWeight & result,
        const std::vector<int32_t> & expert_pool,
        const std::vector<int32_t> & expert_local) {
    result.quantize_pool_indices.clear();
    result.gs24_quantize_index = -1;
    result.gs28_quantize_index = -1;
    result.profile_mask = 0;
    bool hetero_supported = true;
    std::unordered_set<int64_t> quantized_shapes;
    for (int pool_index = 0;
         pool_index < static_cast<int>(result.pools.size());
         ++pool_index) {
        auto & pool = result.pools.at(static_cast<size_t>(pool_index));
        pool.profile_code = nint_moe_profile_code(
            static_cast<int>(pool.weight.bits),
            static_cast<int>(pool.weight.gs));
        hetero_supported =
            hetero_supported && pool.profile_code >= 0;
        if (pool.profile_code >= 0) {
            result.profile_mask |= 1 << pool.profile_code;
        }
        const int64_t activation_key =
            (pool.weight.gs << 32) ^ pool.weight.ng;
        if (quantized_shapes.insert(activation_key).second) {
            result.quantize_pool_indices.push_back(pool_index);
        }
    }
    for (int pool_index : result.quantize_pool_indices) {
        const int gs = static_cast<int>(
            result.pools.at(
                static_cast<size_t>(pool_index)).weight.gs);
        if (gs == 24) result.gs24_quantize_index = pool_index;
        if (gs == 28) result.gs28_quantize_index = pool_index;
    }
    const char * disable_hetero =
        std::getenv("MFQ_DISABLE_MOE_HETERO");
    result.hetero_supported = hetero_supported &&
        !(disable_hetero != nullptr &&
          std::atoi(disable_hetero) != 0);
    if (!result.hetero_supported) return;

    std::vector<int64_t> weight_ptrs;
    std::vector<int32_t> pool_params;
    weight_ptrs.reserve(result.pools.size() * 5);
    pool_params.reserve(result.pools.size() * 2);
    for (const auto & pool : result.pools) {
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                pool.weight.q_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                pool.weight.sub_scale.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                pool.weight.sub_min.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                pool.weight.neuron_scale.data_ptr<float>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                pool.weight.neuron_min.data_ptr<float>())));
        pool_params.push_back(pool.profile_code);
        pool_params.push_back(
            static_cast<int32_t>(pool.weight.ng));
    }
    result.weight_ptrs = torch::from_blob(
        weight_ptrs.data(),
        {static_cast<int64_t>(result.pools.size()), 5},
        torch::TensorOptions().dtype(torch::kInt64))
        .clone().to(torch::kCUDA).contiguous();
    result.pool_params = torch::from_blob(
        pool_params.data(),
        {static_cast<int64_t>(result.pools.size()), 2},
        torch::TensorOptions().dtype(torch::kInt32))
        .clone().to(torch::kCUDA).contiguous();
    result.expert_pool = torch::from_blob(
        const_cast<int32_t *>(expert_pool.data()),
        {static_cast<int64_t>(expert_pool.size())},
        torch::TensorOptions().dtype(torch::kInt32))
        .clone().to(torch::kCUDA).contiguous();
    result.expert_local = torch::from_blob(
        const_cast<int32_t *>(expert_local.data()),
        {static_cast<int64_t>(expert_local.size())},
        torch::TensorOptions().dtype(torch::kInt32))
        .clone().to(torch::kCUDA).contiguous();
}

static NintMoeWeight to_gpu_nint_moe(const NintMoeCpu & cpu) {
    NintMoeWeight result;
    result.n_experts = cpu.n_experts;
    result.out_per_expert = cpu.out_per_expert;
    result.neuron_len = cpu.neuron_len;
    result.pools.reserve(cpu.pools.size());
    std::vector<int32_t> expert_pool(
        static_cast<size_t>(cpu.n_experts), -1);
    std::vector<int32_t> expert_local(
        static_cast<size_t>(cpu.n_experts), -1);
    for (int pool_index = 0;
         pool_index < static_cast<int>(cpu.pools.size());
         ++pool_index) {
        const auto & source_pool = cpu.pools.at(static_cast<size_t>(pool_index));
        if (source_pool.dtype == "NINTM" ||
            source_pool.dtype.rfind("NINT", 0) != 0) {
            throw std::runtime_error("non-NINT cohort reached the NINT-only loader");
        }
        NintMoePoolWeight pool;
        pool.weight = to_gpu_nint(source_pool.weight);
        pool.local_experts = (int)source_pool.expert_ids.size();
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            const int expert = source_pool.expert_ids.at(static_cast<size_t>(index));
            local[static_cast<size_t>(expert)] = index;
            expert_pool[static_cast<size_t>(expert)] = pool_index;
            expert_local[static_cast<size_t>(expert)] = index;
        }
        pool.expert_local = torch::from_blob(
            local.data(), {(int64_t)local.size()}, torch::TensorOptions().dtype(torch::kInt32))
            .clone().to(torch::kCUDA).contiguous();
        result.pools.push_back(std::move(pool));
    }
    initialize_nint_moe_dispatch(
        result, expert_pool, expert_local);
    return result;
}

static std::vector<mfq::TensorParallelSlice>
plan_moe_tensor_parallel_slices(
    int64_t extent,
    const std::string & name);

static NintMoeWeight load_nint_moe_gpu(
    const MfqFile & mfq, const std::string & name,
    bool cacheable = false,
    int layer_id = -1,
    const std::string & projection_role = {});

static std::vector<torch::Tensor> nint_moe_ffn_forward(
        const NintMoeWeight & gate_up, const NintMoeWeight & down,
        torch::Tensor x, torch::Tensor router_logits,
        int top_k, bool use_sigmoid, bool use_sqrt_softplus,
        bool normalize, bool delayed_softmax,
        c10::optional<torch::Tensor> router_bias, double router_scale) {
    if (gate_up.n_experts != down.n_experts ||
        gate_up.out_per_expert != 2 * down.neuron_len ||
        down.out_per_expert != gate_up.neuron_len) {
        throw std::runtime_error("incompatible fused gate_up/down NINTM tensors");
    }
    auto selected = moe_topk_cuda(
        router_logits.contiguous(), top_k, use_sigmoid, use_sqrt_softplus,
        normalize, delayed_softmax,
        router_bias, 1e-20, router_scale);
    MoeRoutePlan route = build_moe_route_plan(selected.at(0), gate_up.n_experts);
    auto gate_up_pair = gate_up.forward(x, route);
    down.prefetch(route);
    auto hidden = moe_swiglu_split_cuda(gate_up_pair);
    auto down_pair = down.forward(hidden, route);
    auto output = moe_weighted_reduce_cuda(down_pair, selected.at(1));
    return {output, selected.at(0), selected.at(1)};
}

struct NvqCpu {
    int format = 0;
    int sign_mode = 0;
    int sub_bits = 0;
    int gs = 0;
    int axis = 0;
    int neuron_len = 0;
    int out = 0;
    int ng = 0;
    int nvec = 0;
    int nsign = 0;
    std::vector<int64_t> shape;
    std::vector<uint8_t> indices_packed;
    std::vector<uint8_t> aux_packed;
    std::vector<uint8_t> sub_scale_packed;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<int8_t> codebook;
};

static int nvq_vector_size(int format) {
    return (format == 3 || format == 10 || format == 12 || format == 15) ? 4 : 8;
}

static int nvq_index_bits(int format) {
    return format == 1 ? 11 :
        (format == 8 ? 9 :
         (format == 7 ? 7 :
          (format == 9 ? 6 :
           (format == 12 ? 9 :
            ((format == 13 || format == 15) ? 10 :
             (format == 14 ? 12 : 8))))));
}

static bool nvq_delta_format(int format) {
    return format == 1 || format == 8;
}

static bool nvq_no_aux_format(int format) {
    return format == 7 || format == 9;
}

static void copy_packed_bits(
        const std::vector<uint8_t> & source,
        size_t source_bit,
        std::vector<uint8_t> & destination,
        size_t destination_bit,
        size_t bit_count) {
    if (bit_count == 0) return;
    if (source_bit + bit_count >
            source.size() * 8 ||
        destination_bit + bit_count >
            destination.size() * 8) {
        throw std::runtime_error(
            "packed tensor-parallel bit copy is out of bounds");
    }
    if ((source_bit & 7u) == 0 &&
        (destination_bit & 7u) == 0) {
        const size_t full_bytes =
            bit_count / 8;
        if (full_bytes != 0) {
            std::memcpy(
                destination.data() +
                    destination_bit / 8,
                source.data() +
                    source_bit / 8,
                full_bytes);
            source_bit += full_bytes * 8;
            destination_bit += full_bytes * 8;
            bit_count -= full_bytes * 8;
        }
    }
    while (bit_count >= 8) {
        const size_t source_byte =
            source_bit >> 3;
        const int source_shift =
            static_cast<int>(source_bit & 7u);
        uint16_t source_word =
            source[source_byte];
        if (source_shift != 0 &&
            source_byte + 1 < source.size()) {
            source_word |=
                static_cast<uint16_t>(
                    source[source_byte + 1])
                << 8;
        }
        const uint8_t value =
            static_cast<uint8_t>(
                source_word >> source_shift);
        const size_t destination_byte =
            destination_bit >> 3;
        const int destination_shift =
            static_cast<int>(
                destination_bit & 7u);
        destination[destination_byte] |=
            static_cast<uint8_t>(
                value << destination_shift);
        if (destination_shift != 0 &&
            destination_byte + 1 <
                destination.size()) {
            destination[destination_byte + 1] |=
                static_cast<uint8_t>(
                    value >>
                    (8 - destination_shift));
        }
        source_bit += 8;
        destination_bit += 8;
        bit_count -= 8;
    }
    for (size_t bit = 0; bit < bit_count; ++bit) {
        if ((source[source_bit >> 3] >>
             (source_bit & 7u)) & 1u) {
            destination[destination_bit >> 3] |=
                static_cast<uint8_t>(
                    1u << (destination_bit & 7u));
        }
        ++source_bit;
        ++destination_bit;
    }
}

static std::vector<uint8_t> slice_packed_rows(
        const std::vector<uint8_t> & source,
        int64_t source_rows,
        int64_t source_items_per_row,
        int64_t row_begin,
        int64_t row_end,
        int64_t item_begin,
        int64_t item_count,
        int bits) {
    if (bits == 0) return {};
    if (source_rows <= 0 || source_items_per_row <= 0 ||
        row_begin < 0 || row_begin >= row_end ||
        row_end > source_rows || item_begin < 0 ||
        item_count <= 0 ||
        item_begin + item_count > source_items_per_row) {
        throw std::runtime_error("invalid packed tensor-parallel slice");
    }
    const int64_t destination_rows = row_end - row_begin;
    const size_t destination_items =
        static_cast<size_t>(destination_rows) * item_count;
    std::vector<uint8_t> destination(
        (destination_items * static_cast<size_t>(bits) + 7) / 8,
        0);
    for (int64_t row = 0; row < destination_rows; ++row) {
        const size_t source_item =
            static_cast<size_t>(row + row_begin) *
                source_items_per_row +
            static_cast<size_t>(item_begin);
        const size_t destination_item =
            static_cast<size_t>(row) * item_count;
        copy_packed_bits(
            source,
            source_item * static_cast<size_t>(bits),
            destination,
            destination_item * static_cast<size_t>(bits),
            static_cast<size_t>(item_count) *
                static_cast<size_t>(bits));
    }
    return destination;
}

static NvqCpu slice_nvq_cpu(
        const NvqCpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NVQ");
    if (axis != TensorParallelAxis::Output &&
        axis != TensorParallelAxis::Input) {
        throw std::runtime_error("NVQ shard requires an output or input axis");
    }
    NvqCpu result = source;
    int64_t row_begin = 0;
    int64_t row_end = source.out;
    int64_t group_begin = 0;
    int64_t group_count = source.ng;
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid NVQ output shard");
        }
        row_begin = begin;
        row_end = end;
        result.out = static_cast<int>(end - begin);
        result.shape[0] = result.out;
    } else {
        if (begin < 0 || begin >= end || end > source.ng) {
            throw std::runtime_error("invalid NVQ input shard");
        }
        group_begin = begin;
        group_count = end - begin;
        result.ng = static_cast<int>(group_count);
        const int64_t element_begin = begin * source.gs;
        const int64_t element_end =
            std::min<int64_t>(end * source.gs, source.neuron_len);
        result.neuron_len =
            static_cast<int>(element_end - element_begin);
        result.shape[1] = result.neuron_len;
        const int vector_size = nvq_vector_size(source.format);
        if (element_begin % vector_size != 0 ||
            element_begin % 8 != 0) {
            throw std::runtime_error(
                "NVQ input shard does not preserve vector boundaries");
        }
        result.nvec =
            (result.neuron_len + vector_size - 1) / vector_size;
        result.nsign = (result.neuron_len + 7) / 8;
    }

    result.neuron_scale_h.assign(
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_begin),
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_end));
    result.sub_scale_packed = slice_packed_rows(
        source.sub_scale_packed,
        source.out, source.ng,
        row_begin, row_end,
        group_begin, group_count,
        source.sub_bits);

    const int vector_size = nvq_vector_size(source.format);
    const int64_t element_begin = group_begin * source.gs;
    const int64_t vector_begin = element_begin / vector_size;
    const int64_t vector_count =
        axis == TensorParallelAxis::Output
        ? source.nvec
        : result.nvec;
    result.indices_packed = slice_packed_rows(
        source.indices_packed,
        source.out, source.nvec,
        row_begin, row_end,
        vector_begin, vector_count,
        nvq_index_bits(source.format));

    if (nvq_delta_format(source.format)) {
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.ng,
            row_begin, row_end,
            group_begin, group_count,
            1);
    } else if (nvq_no_aux_format(source.format)) {
        result.aux_packed.clear();
    } else {
        const int64_t sign_begin = element_begin / 8;
        const int64_t sign_count =
            axis == TensorParallelAxis::Output
            ? source.nsign
            : result.nsign;
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.nsign,
            row_begin, row_end,
            sign_begin, sign_count,
            7);
    }
    return result;
}

static std::vector<int8_t> expand_nvq_codebook(
    int format, const uint16_t * packed, size_t count) {
    const int dims = format == 3 ? 4 : 8;
    const int expected = format == 1 ? 2048 : (format == 8 ? 1024 : 256);
    if ((int)count != expected) throw std::runtime_error("NVQ codebook entry count mismatch");
    const int digit_bits = format == 3 ? 3 : 2;
    std::vector<int8_t> codebook(count * (size_t)dims);
    for (size_t row = 0; row < count; ++row) {
        uint16_t word = packed[row];
        for (int i = 0; i < dims; ++i) {
            int digit = (word >> (digit_bits * i)) & ((1 << digit_bits) - 1);
            const bool ternary = format == 1 || format == 8;
            int value = ternary ? digit - 1 : 2 * digit + 1;
            if ((ternary && (value < -1 || value > 1)) ||
                (!ternary && (value < 1 || value > (format == 2 ? 7 : 15)))) {
                throw std::runtime_error("invalid NVQ codebook digit");
            }
            codebook[row * (size_t)dims + i] = (int8_t)value;
        }
    }
    return codebook;
}

static std::vector<int8_t> expand_npq0_s_runtime_lut(
    const std::vector<uint8_t> & packed) {
    constexpr size_t kMetadataBytes = 64;
    constexpr size_t kStates = 4;
    constexpr size_t kEntries = 8;
    constexpr size_t kSubvector = 4;
    constexpr size_t kPackedBytes = kMetadataBytes + 2 * kStates * kEntries * kSubvector;
    constexpr size_t kRuntimeBytes = kMetadataBytes + kStates * 64 * 8;
    if (packed.size() != kPackedBytes) {
        throw std::runtime_error("NPQ0-S packed table size mismatch");
    }
    std::vector<int8_t> runtime(kRuntimeBytes);
    std::memcpy(runtime.data(), packed.data(), kMetadataBytes);
    const int8_t * first = reinterpret_cast<const int8_t *>(packed.data() + kMetadataBytes);
    const int8_t * second = first + kStates * kEntries * kSubvector;
    for (size_t state = 0; state < kStates; ++state) {
        for (size_t second_index = 0; second_index < kEntries; ++second_index) {
            for (size_t first_index = 0; first_index < kEntries; ++first_index) {
                const size_t index = first_index | (second_index << 3);
                int8_t * destination = runtime.data() + kMetadataBytes
                    + (state * 64 + index) * 8;
                std::memcpy(
                    destination,
                    first + (state * kEntries + first_index) * kSubvector,
                    kSubvector);
                std::memcpy(
                    destination + kSubvector,
                    second + (state * kEntries + second_index) * kSubvector,
                    kSubvector);
            }
        }
    }
    return runtime;
}

static std::vector<uint16_t> read_codebook_words(
    const std::vector<uint8_t> & blob, size_t & off, size_t count) {
    size_t nbytes = count * sizeof(uint16_t);
    if (off + nbytes > blob.size()) throw std::runtime_error("truncated NVQ custom codebook");
    std::vector<uint16_t> words(count);
    std::memcpy(words.data(), blob.data() + off, nbytes);
    off += nbytes;
    return words;
}

static std::vector<uint8_t> take_bytes(
    const std::vector<uint8_t> & blob, size_t & off, size_t count, const char * label) {
    if (off + count > blob.size()) throw std::runtime_error(std::string("truncated NVQ ") + label);
    std::vector<uint8_t> result(count);
    std::memcpy(result.data(), blob.data() + off, count);
    off += count;
    return result;
}

static NvqCpu unpack_nvq(const std::vector<uint8_t> & blob, const std::string & dtype) {
    if (blob.size() < 20) throw std::runtime_error("truncated NVQ header");
    NvqCpu t;
    if (dtype == "NVQ1-L") t.format = 1;
    else if (dtype == "NVQ2" || dtype == "NIQ2") t.format = 2;
    else if (dtype == "NVQ3" || dtype == "NIQ3") t.format = 3;
    else if (dtype == "NVQ2J" || dtype == "NIQ2J") t.format = 5;
    else if (dtype == "NVQ2J-L") t.format = 13;
    else if (dtype == "NVQ2J-XL") t.format = 14;
    else if (dtype == "NVQ3J") t.format = 10;
    else if (dtype == "NVQ3J-512") t.format = 12;
    else if (dtype == "NVQ3J-L") t.format = 15;
    else if (dtype == "NPQ0-L") t.format = 7;
    else if (dtype == "NVQ1-S") t.format = 8;
    else if (dtype == "NPQ0-S") t.format = 9;
    else throw std::runtime_error("unsupported compact VQ dtype: " + dtype);

    const uint8_t expected4 = (t.format == 1 || t.format == 7)
        ? (uint8_t)'L' : ((t.format == 8 || t.format == 9) ? (uint8_t)'S' : (uint8_t)'1');
    const bool nvq_magic = blob[0] == 'N' && blob[1] == 'V' && blob[2] == 'Q';
    const bool niq_magic = blob[0] == 'N' && blob[1] == 'I' && blob[2] == 'Q';
    const bool npq_magic = blob[0] == 'N' && blob[1] == 'P' && blob[2] == 'Q';
    const bool nvq1_l_magic = blob[0] == 'N' && blob[1] == 'Q' && blob[2] == '1';
    const bool valid_magic = (t.format == 7 || t.format == 9) ? npq_magic :
        ((t.format == 1 || t.format == 8) ? nvq1_l_magic : (nvq_magic || niq_magic));
    if (!valid_magic || blob[3] != expected4) {
        throw std::runtime_error("bad compact VQ blob magic for " + dtype);
    }
    size_t off = 4;
    const uint8_t profile = blob[off++];
    t.sub_bits = blob[off++];
    t.gs = (int)read_u16_from(blob, off);
    t.axis = read_i32_from(blob, off);
    t.neuron_len = read_i32_from(blob, off);
    uint32_t ndim = read_u32_from(blob, off);
    if (ndim == 0 || ndim > 8) throw std::runtime_error("invalid NVQ ndim");
    t.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) t.shape[i] = read_i64_from(blob, off);
    t.out = (int)read_u32_from(blob, off);
    if (t.gs != 24 || t.sub_bits < 1 || t.sub_bits > 8) {
        throw std::runtime_error("C++ NVQ runtime requires gs24 and sub_bits in [1,8]");
    }
    if (t.axis != 0 || t.shape.size() != 2 || t.shape[0] != t.out || t.shape[1] != t.neuron_len) {
        throw std::runtime_error("C++ NVQ runtime requires row-major rank-2 axis=0 weights");
    }

    bool custom = false;
    if (t.format == 7) {
        if (profile != 1 || t.sub_bits != 3) {
            throw std::runtime_error("unsupported NPQ0-L profile");
        }
    } else if (t.format == 1) {
        if (profile != 1 && profile != 2) throw std::runtime_error("unsupported NVQ1-L profile");
        custom = profile == 2;
    } else if (t.format == 8) {
        if (profile != 1 || t.sub_bits != 4) {
            throw std::runtime_error("unsupported NVQ1-S profile");
        }
        custom = true;
    } else if (t.format == 9) {
        if (profile != 2 || t.sub_bits != 2) {
            throw std::runtime_error("unsupported NPQ0-S profile");
        }
    } else {
        constexpr uint8_t kIndexParity = 0x80;
        constexpr uint8_t kCustom = 0x40;
        constexpr uint8_t kJsc = 0x20;
        const bool jsc = (profile & kJsc) != 0;
        int codebook_id = profile & ~(kIndexParity | kCustom | kJsc);
        int expected_id =
            t.format == 13 ? 4 :
            (t.format == 14 ? 5 :
             (t.format == 15 ? 6 :
              ((t.format == 2 || t.format == 5)
               ? 1
               : (t.format == 12 ? 3 : 2))));
        if (codebook_id != expected_id) throw std::runtime_error("NVQ dtype/codebook mismatch");
        t.sign_mode = (profile & kIndexParity) ? 1 : 0;
        if (t.sign_mode && t.format != 2) throw std::runtime_error("NVQ index parity requires NVQ2");
        custom = (profile & kCustom) != 0;
        if (t.format == 5 || t.format == 10 || t.format == 12 ||
            t.format == 13 || t.format == 14 || t.format == 15) {
            if (!jsc || custom || t.sign_mode || t.sub_bits != 4) {
                throw std::runtime_error("invalid NVQ-JSC profile flags");
            }
        } else if (jsc) {
            throw std::runtime_error("NVQ-JSC blob requires an NVQ2J/NVQ3J dtype");
        }
    }

    const size_t codebook_count = t.format == 1 ? 2048 : (t.format == 8 ? 1024 : 256);
    if (t.format == 7) {
        constexpr size_t kMetadataBytes = 64;
        constexpr size_t kTableBytes = 832;
        if (off + kTableBytes > blob.size()) {
            throw std::runtime_error("truncated NPQ0-L product tables");
        }
        const uint8_t * header = blob.data() + off;
        const uint8_t expected[6] = {1, 8, 3, 4, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (header[i] != expected[i]) {
                throw std::runtime_error("unsupported NPQ0-L table profile");
            }
        }
        if (header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid NPQ0-L reserved table bytes");
        }
        for (int state = 0; state < 8; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 8 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NPQ0-L scale LUT must be finite and non-negative");
            }
        }
        for (size_t i = 24; i < kMetadataBytes; ++i) {
            if (header[i] != 0) {
                throw std::runtime_error("invalid NPQ0-L reserved table bytes");
            }
        }
        auto metadata = take_bytes(blob, off, kTableBytes, "NPQ0-L product tables");
        t.codebook.resize(metadata.size());
        std::memcpy(t.codebook.data(), metadata.data(), metadata.size());
    } else if (t.format == 9) {
        constexpr size_t kMetadataBytes = 64;
        constexpr size_t kTableBytes = 320;
        if (off + kTableBytes > blob.size()) {
            throw std::runtime_error("truncated NPQ0-S product tables");
        }
        const uint8_t * header = blob.data() + off;
        const uint8_t expected[6] = {2, 4, 3, 3, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (header[i] != expected[i]) {
                throw std::runtime_error("unsupported NPQ0-S table profile");
            }
        }
        if (header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid NPQ0-S reserved table bytes");
        }
        for (int state = 0; state < 4; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 8 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NPQ0-S scale LUT must be finite and non-negative");
            }
        }
        for (size_t i = 16; i < kMetadataBytes; ++i) {
            if (header[i] != 0) {
                throw std::runtime_error("invalid NPQ0-S reserved table bytes");
            }
        }
        auto metadata = take_bytes(blob, off, kTableBytes, "NPQ0-S product tables");
        t.codebook = expand_npq0_s_runtime_lut(metadata);
    } else if (t.format == 5 || t.format == 10 || t.format == 12 ||
               t.format == 13 || t.format == 14 || t.format == 15) {
        constexpr size_t kHeaderBytes = 64;
        const int vector_size = nvq_vector_size(t.format);
        const size_t entries =
            t.format == 12 ? 512 :
            ((t.format == 13 || t.format == 15) ? 1024 :
             (t.format == 14 ? 4096 : 256));
        const size_t kBankBytes = entries * (size_t)vector_size;
        if (off + kHeaderBytes > blob.size()) {
            throw std::runtime_error("truncated NVQ2J metadata header");
        }
        const uint8_t * header = blob.data() + off;
        const int banks = header[1];
        if (header[0] != 1 || (banks != 1 && banks != 2 && banks != 4) || header[2] != 16) {
            throw std::runtime_error("unsupported NVQ2J metadata dimensions");
        }
        if (header[3] != 0 && header[3] != 1) {
            throw std::runtime_error("invalid NVQ-JSC state mode");
        }
        for (int state = 0; state < 16; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 4 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NVQ-JSC scale LUT must be finite and non-negative");
            }
            if (header[36 + state] >= banks) {
                throw std::runtime_error("NVQ-JSC state references a missing bank");
            }
            if (header[3] == 1) {
                const uint8_t expected_bank = (uint8_t)(state % banks);
                const int rank = state / banks;
                const float expected_scale = banks == 1
                    ? (float)state
                    : (vector_size == 4
                        ? (float)(rank + 1)
                        : 15.0f * (float)(rank + 1) / (float)(16 / banks));
                const c10::Half expected_half(expected_scale);
                uint16_t expected_alpha_h;
                std::memcpy(&expected_alpha_h, &expected_half, sizeof(expected_alpha_h));
                if (header[36 + state] != expected_bank || alpha_h != expected_alpha_h) {
                    throw std::runtime_error("invalid analytic NVQ-JSC state tables");
                }
            }
        }
        for (size_t i = 52; i < kHeaderBytes; ++i) {
            if (header[i] != 0) throw std::runtime_error("invalid NVQ-JSC reserved metadata bytes");
        }
        const size_t metadata_bytes = kHeaderBytes + (size_t)banks * kBankBytes;
        auto metadata = take_bytes(blob, off, metadata_bytes, "JSC metadata");
        for (int bank = 0; bank < banks; ++bank) {
            const size_t base = kHeaderBytes + (size_t)bank * kBankBytes;
            for (size_t entry = 0; entry < entries; ++entry) {
                bool nonzero = false;
                for (int dim = 0; dim < vector_size; ++dim) {
                    const uint8_t value =
                        metadata[base + (size_t)entry * vector_size + dim];
                    if (value > 127) throw std::runtime_error("NVQ-JSC codebook value exceeds int8");
                    nonzero = nonzero || value != 0;
                }
                if (!nonzero) throw std::runtime_error("NVQ-JSC codeword must not be all zero");
            }
        }
        t.codebook.resize(metadata.size());
        std::memcpy(t.codebook.data(), metadata.data(), metadata.size());
    } else if (custom) {
        auto words = read_codebook_words(blob, off, codebook_count);
        t.codebook = expand_nvq_codebook(t.format, words.data(), words.size());
    } else if (t.format == 1) {
        t.codebook = expand_nvq_codebook(
            1, mfq::nvq_codebooks::kNvq1LCodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq1LCodebookPacked) / sizeof(uint16_t));
    } else if (t.format == 2) {
        t.codebook = expand_nvq_codebook(
            2, mfq::nvq_codebooks::kNvq2CodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq2CodebookPacked) / sizeof(uint16_t));
    } else {
        t.codebook = expand_nvq_codebook(
            3, mfq::nvq_codebooks::kNvq3CodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq3CodebookPacked) / sizeof(uint16_t));
    }

    t.ng = (t.neuron_len + t.gs - 1) / t.gs;
    const int vector_size = nvq_vector_size(t.format);
    t.nvec = (t.neuron_len + vector_size - 1) / vector_size;
    t.nsign = (t.neuron_len + 7) / 8;
    size_t anchor_bytes = (size_t)t.out * 2;
    if (off + anchor_bytes > blob.size()) throw std::runtime_error("truncated NVQ neuron anchors");
    t.neuron_scale_h.resize(t.out);
    std::memcpy(t.neuron_scale_h.data(), blob.data() + off, anchor_bytes);
    off += anchor_bytes;
    size_t sub_bytes = ((size_t)t.out * t.ng * t.sub_bits + 7) / 8;
    const size_t index_bits = (size_t)nvq_index_bits(t.format);
    size_t index_bytes = ((size_t)t.out * t.nvec * index_bits + 7) / 8;
    const bool delta_format = t.format == 1 || t.format == 8;
    const bool no_aux_format = t.format == 7 || t.format == 9;
    size_t aux_count = delta_format ? (size_t)t.out * t.ng :
        (no_aux_format ? 0 : (size_t)t.out * t.nsign);
    size_t aux_bits = delta_format ? 1 : (no_aux_format ? 0 : 7);
    size_t aux_bytes = (aux_count * aux_bits + 7) / 8;
    t.sub_scale_packed = take_bytes(blob, off, sub_bytes, "sub-scale stream");
    t.indices_packed = take_bytes(blob, off, index_bytes, "index stream");
    t.aux_packed = take_bytes(blob, off, aux_bytes, "aux stream");
    if (off != blob.size()) throw std::runtime_error("invalid NVQ blob tail");
    return t;
}

static torch::Tensor cpu_i8_tensor(const std::vector<int8_t> & v, std::initializer_list<int64_t> shape) {
    return torch::from_blob((void *)v.data(), shape, torch::TensorOptions().dtype(torch::kInt8)).clone();
}

struct NvqWorkspace {
    int M = 0;
    int K_pad = 0;
    torch::Tensor qx;
    torch::Tensor xscale;
    torch::Tensor swiglu_scratch;
};

struct NvqWeight {
    torch::Tensor indices_packed;
    torch::Tensor aux_packed;
    torch::Tensor sub_scale_packed;
    torch::Tensor neuron_scale;
    torch::Tensor codebook;
    int64_t format = 0;
    int64_t kernel_format = 0;
    int64_t sign_mode = 0;
    int64_t sub_bits = 0;
    int64_t gs = 0;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t neuron_len = 0;
    std::vector<int64_t> shape;
    mutable std::unordered_map<int, NvqWorkspace> workspaces;

    NvqWorkspace & workspace(int M) const {
        int K_pad = (int)(ng * gs);
        auto it = workspaces.find(M);
        if (it != workspaces.end() && it->second.K_pad == K_pad) return it->second;
        NvqWorkspace ws;
        ws.M = M;
        ws.K_pad = K_pad;
        ws.qx = torch::empty({M, K_pad}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt8));
        ws.xscale = torch::empty({M, ng}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        return workspaces.emplace(M, std::move(ws)).first->second;
    }
};

static std::vector<uint8_t> repack_nvq2_exec_metadata(const NvqCpu & c) {
    if ((c.format != 2 && c.format != 5) || c.nvec != c.nsign) {
        throw std::runtime_error("NVQ2 execution metadata requires one sign record per vector");
    }
    const size_t count = (size_t)c.out * c.nvec;
    if (c.indices_packed.size() != count) {
        throw std::runtime_error("NVQ2 index stream length mismatch during execution repack");
    }
    std::vector<uint8_t> metadata(count * 2);
    for (size_t linear = 0; linear < count; ++linear) {
        const size_t bit = linear * 7;
        const size_t byte = bit >> 3;
        const int shift = (int)(bit & 7);
        uint16_t word = c.aux_packed[byte];
        if (byte + 1 < c.aux_packed.size()) {
            word |= (uint16_t)c.aux_packed[byte + 1] << 8;
        }
        const uint8_t index = c.indices_packed[linear];
        const uint8_t mask7 = (uint8_t)((word >> shift) & 0x7f);
        uint8_t parity = mask7;
        parity ^= parity >> 4;
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        const uint8_t last = (parity & 1u) ^
            (c.sign_mode ? ((index >> 7) & 1u) : 0u);
        metadata[2 * linear] = c.indices_packed[linear];
        metadata[2 * linear + 1] = mask7 | (last << 7);
    }
    return metadata;
}

static bool nvq2_exec_layout_enabled() {
    const char * disable = std::getenv("MFQ_DISABLE_NVQ2_EXEC");
    if (disable == nullptr) disable = std::getenv("MFQ_DISABLE_NIQ2_EXEC");
    return disable == nullptr || disable[0] != '1';
}

static NvqWeight to_device_nvq(const NvqCpu & c, bool cuda) {
    NvqWeight w;
    w.format = c.format;
    w.kernel_format = c.format;
    w.sign_mode = c.sign_mode;
    w.sub_bits = c.sub_bits;
    w.gs = c.gs;
    w.out = c.out;
    w.ng = c.ng;
    w.neuron_len = c.neuron_len;
    w.shape = c.shape;
    if ((c.format == 2 || c.format == 5) && nvq2_exec_layout_enabled()) {
        auto metadata = repack_nvq2_exec_metadata(c);
        w.indices_packed = cpu_u8_tensor(
            metadata, {(int64_t)metadata.size()});
        w.aux_packed = torch::empty(
            {0}, torch::TensorOptions().dtype(torch::kUInt8));
        w.kernel_format = c.format == 5 ? 6 : 4;
    } else if (
        c.format == 10 && c.codebook.size() >= 64 &&
        c.codebook[3] == 1 && c.codebook[1] == 2) {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
        w.kernel_format = 11;
    } else {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
    }
    w.sub_scale_packed = cpu_u8_tensor(
        c.sub_scale_packed, {(int64_t)c.sub_scale_packed.size()});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    if (
        c.format == 5 || c.format == 7 || c.format == 9 ||
        c.format == 10 || c.format == 12 ||
        c.format == 13 || c.format == 14 || c.format == 15) {
        w.codebook = cpu_i8_tensor(
            c.codebook, {(int64_t)c.codebook.size()});
    } else {
        const int dims = nvq_vector_size(c.format);
        const int entries = c.format == 1 ? 2048 : (c.format == 8 ? 1024 : 256);
        w.codebook = cpu_i8_tensor(c.codebook, {entries, dims});
    }
    if (cuda) {
        w.indices_packed = w.indices_packed.to(torch::kCUDA).contiguous();
        w.aux_packed = w.aux_packed.to(torch::kCUDA).contiguous();
        w.sub_scale_packed =
            w.sub_scale_packed.to(torch::kCUDA).contiguous();
        w.neuron_scale = w.neuron_scale.to(torch::kCUDA).contiguous();
        w.codebook = w.codebook.to(torch::kCUDA).contiguous();
        (void)w.workspace(1);
    }
    return w;
}

static NvqWeight to_gpu_nvq(const NvqCpu & c) {
    return to_device_nvq(c, true);
}

static NvqWeight to_cuda_device_nvq(
        const NvqCpu & c, int device) {
    c10::cuda::CUDAGuard guard(device);
    return to_device_nvq(c, true);
}

static NvqWeight to_cpu_nvq(const NvqCpu & c) {
    return to_device_nvq(c, false);
}

struct NepqCpu {
    int profile = -1;
    int format = 0;
    int state_bits = 0;
    int index_bits = 0;
    int aux_bits = 0;
    int table_bytes = 0;
    int runtime_table_bytes = 0;
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int bank_count = 0;
    int rotation_block = 0;
    uint64_t rotation_seed = 0;
    int ng = 0;
    int nvec = 0;
    int nsuper = 0;
    std::vector<uint8_t> indices_packed;
    std::vector<uint8_t> aux_packed;
    std::vector<uint8_t> state_packed;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<int8_t> table_pool;
    std::vector<int8_t> grouped_table_pool;
    std::vector<uint8_t> bank_ids;
    std::vector<int8_t> rotation_signs;
};

static size_t packed_nbytes(size_t count, int bits) {
    return bits == 0 ? 0 : (count * (size_t)bits + 7) / 8;
}

static void configure_nepq_profile(NepqCpu & value, const std::string & dtype) {
    if (dtype == "NEPQ0-S") {
        value.profile = 0;
        value.format = 9;
        value.state_bits = 2;
        value.index_bits = 6;
        value.aux_bits = 0;
        value.table_bytes = 320;
        value.runtime_table_bytes = 320;
    } else if (dtype == "NEPQ0-L") {
        value.profile = 1;
        value.format = 7;
        value.state_bits = 3;
        value.index_bits = 7;
        value.aux_bits = 0;
        value.table_bytes = 832;
        value.runtime_table_bytes = 832;
    } else if (dtype == "NEPQ1-S") {
        value.profile = 2;
        value.format = 8;
        value.state_bits = 4;
        value.index_bits = 9;
        value.aux_bits = 1;
        value.table_bytes = 2048;
        value.runtime_table_bytes = 1024 * 8;
    } else if (dtype == "NEPQ1-L") {
        value.profile = 3;
        value.format = 1;
        value.state_bits = 3;
        value.index_bits = 11;
        value.aux_bits = 1;
        value.table_bytes = 4096;
        value.runtime_table_bytes = 2048 * 8;
    } else {
        throw std::runtime_error("unsupported NEPQ cohort dtype: " + dtype);
    }
}

static std::vector<int8_t> expand_nepq_table(
        const uint8_t * source, const NepqCpu & value) {
    std::vector<uint8_t> packed(
        source, source + static_cast<ptrdiff_t>(value.table_bytes));
    if (value.profile == 0) {
        const uint8_t expected[6] = {2, 4, 3, 3, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (packed[(size_t)i] != expected[i]) {
                throw std::runtime_error("unsupported NEPQ0-S table profile");
            }
        }
        std::vector<int8_t> result(value.table_bytes);
        std::memcpy(result.data(), packed.data(), packed.size());
        return result;
    }
    if (value.profile == 1) {
        const uint8_t expected[6] = {1, 8, 3, 4, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (packed[(size_t)i] != expected[i]) {
                throw std::runtime_error("unsupported NEPQ0-L table profile");
            }
        }
        std::vector<int8_t> result(value.table_bytes);
        std::memcpy(result.data(), packed.data(), packed.size());
        return result;
    }
    std::vector<uint16_t> words((size_t)value.table_bytes / 2);
    std::memcpy(words.data(), packed.data(), packed.size());
    return expand_nvq_codebook(
        value.profile == 2 ? 8 : 1, words.data(), words.size());
}

static NepqCpu unpack_nepq(
        const std::vector<uint8_t> & blob,
        const std::string & dtype,
        const std::vector<uint8_t> & runtime_payload) {
    if (blob.size() < 36 || std::memcmp(blob.data(), "NEP1", 4) != 0) {
        throw std::runtime_error("invalid NEPQ cohort header");
    }
    NepqCpu value;
    configure_nepq_profile(value, dtype);
    size_t off = 4;
    const uint8_t version = blob[off++];
    const uint8_t profile = blob[off++];
    const uint8_t groups_per_supergroup = blob[off++];
    const uint8_t flags = blob[off++];
    value.n_experts = (int)read_u32_from(blob, off);
    value.out_per_expert = (int)read_u32_from(blob, off);
    value.neuron_len = (int)read_u32_from(blob, off);
    value.bank_count = (int)read_u32_from(blob, off);
    value.rotation_block = (int)read_u32_from(blob, off);
    value.rotation_seed = read_u64_from(blob, off);
    if (version != 1 || profile != value.profile ||
        groups_per_supergroup != 4 || (flags & ~1u) != 0 ||
        ((flags & 1u) != 0) != (value.rotation_block != 0)) {
        throw std::runtime_error("unsupported NEPQ cohort profile");
    }
    if (value.n_experts <= 0 || value.out_per_expert <= 0 ||
        value.neuron_len <= 0 || value.neuron_len % 8 != 0 ||
        value.bank_count <= 0 || value.bank_count > 256) {
        throw std::runtime_error("invalid NEPQ cohort dimensions");
    }
    if (value.rotation_block != 0 &&
        ((value.rotation_block & (value.rotation_block - 1)) != 0 ||
         value.neuron_len % value.rotation_block != 0)) {
        throw std::runtime_error("invalid NEPQ Hadamard block");
    }
    value.ng = (value.neuron_len + 23) / 24;
    value.nvec = value.neuron_len / 8;
    value.nsuper = (value.ng + 3) / 4;
    const int rows = value.n_experts * value.out_per_expert;

    const size_t all_table_bytes =
        (size_t)value.bank_count * value.table_bytes;
    if (off + all_table_bytes > blob.size()) {
        throw std::runtime_error("truncated NEPQ table pool");
    }
    value.table_pool.reserve(
        (size_t)value.bank_count * value.runtime_table_bytes);
    value.grouped_table_pool.reserve(
        (size_t)value.bank_count *
        (value.profile == 0 ? 2112 : value.runtime_table_bytes));
    for (int bank = 0; bank < value.bank_count; ++bank) {
        const uint8_t * table = blob.data() + off + (size_t)bank * value.table_bytes;
        auto runtime = expand_nepq_table(table, value);
        value.table_pool.insert(
            value.table_pool.end(), runtime.begin(), runtime.end());
        if (value.profile == 0) {
            std::vector<uint8_t> compact(
                table, table + static_cast<ptrdiff_t>(value.table_bytes));
            auto grouped = expand_npq0_s_runtime_lut(compact);
            value.grouped_table_pool.insert(
                value.grouped_table_pool.end(), grouped.begin(), grouped.end());
        } else {
            value.grouped_table_pool.insert(
                value.grouped_table_pool.end(), runtime.begin(), runtime.end());
        }
    }
    off += all_table_bytes;
    const size_t anchor_bytes = (size_t)rows * 2;
    if (off + anchor_bytes > blob.size()) {
        throw std::runtime_error("truncated NEPQ neuron anchors");
    }
    value.neuron_scale_h.resize((size_t)rows);
    std::memcpy(value.neuron_scale_h.data(), blob.data() + off, anchor_bytes);
    off += anchor_bytes;
    value.state_packed = take_bytes(
        blob, off, packed_nbytes((size_t)rows * value.ng, value.state_bits),
        "NEPQ state stream");
    value.indices_packed = take_bytes(
        blob, off, packed_nbytes((size_t)rows * value.nvec, value.index_bits),
        "NEPQ index stream");
    value.aux_packed = take_bytes(
        blob, off, packed_nbytes((size_t)rows * value.ng, value.aux_bits),
        "NEPQ aux stream");
    value.bank_ids = take_bytes(
        blob, off, (size_t)rows * value.nsuper, "NEPQ bank selectors");
    if (off != blob.size()) throw std::runtime_error("invalid NEPQ cohort tail");
    for (uint8_t bank : value.bank_ids) {
        if ((int)bank >= value.bank_count) {
            throw std::runtime_error("NEPQ selector references a missing bank");
        }
    }

    if (value.rotation_block == 0) {
        if (!runtime_payload.empty()) {
            throw std::runtime_error("unexpected NEPQ rotation metadata");
        }
    } else {
        if (runtime_payload.size() != 20 + (size_t)value.neuron_len ||
            std::memcmp(runtime_payload.data(), "HSG1", 4) != 0) {
            throw std::runtime_error("missing NEPQ rotation sign vector");
        }
        size_t runtime_off = 4;
        uint32_t width = read_u32_from(runtime_payload, runtime_off);
        uint32_t block = read_u32_from(runtime_payload, runtime_off);
        uint64_t seed = read_u64_from(runtime_payload, runtime_off);
        if ((int)width != value.neuron_len ||
            (int)block != value.rotation_block || seed != value.rotation_seed) {
            throw std::runtime_error("NEPQ rotation metadata mismatch");
        }
        value.rotation_signs.resize((size_t)value.neuron_len);
        std::memcpy(
            value.rotation_signs.data(), runtime_payload.data() + runtime_off,
            (size_t)value.neuron_len);
        for (int8_t sign : value.rotation_signs) {
            if (sign != -1 && sign != 1) {
                throw std::runtime_error("invalid NEPQ rotation sign");
            }
        }
    }
    return value;
}

struct NepqWeight {
    torch::Tensor indices_packed;
    torch::Tensor aux_packed;
    torch::Tensor state_packed;
    torch::Tensor neuron_scale;
    torch::Tensor table_pool;
    torch::Tensor grouped_table_pool;
    torch::Tensor bank_ids;
    torch::Tensor rotation_signs;
    int format = 0;
    int state_bits = 0;
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int ng = 0;
    int rotation_block = 0;
    uint64_t rotation_seed = 0;
};

static NepqWeight to_device_nepq(const NepqCpu & cpu, bool cuda) {
    NepqWeight value;
    value.format = cpu.format;
    value.state_bits = cpu.state_bits;
    value.n_experts = cpu.n_experts;
    value.out_per_expert = cpu.out_per_expert;
    value.neuron_len = cpu.neuron_len;
    value.ng = cpu.ng;
    value.rotation_block = cpu.rotation_block;
    value.rotation_seed = cpu.rotation_seed;
    value.indices_packed = cpu_u8_tensor(
        cpu.indices_packed, {(int64_t)cpu.indices_packed.size()});
    value.aux_packed = cpu_u8_tensor(
        cpu.aux_packed, {(int64_t)cpu.aux_packed.size()});
    value.state_packed = cpu_u8_tensor(
        cpu.state_packed, {(int64_t)cpu.state_packed.size()});
    value.neuron_scale = cpu_f16_to_f32_tensor(
        cpu.neuron_scale_h, cpu.n_experts * cpu.out_per_expert);
    value.table_pool = cpu_i8_tensor(
        cpu.table_pool, {cpu.bank_count, cpu.runtime_table_bytes});
    const int grouped_stride = cpu.profile == 0 ? 2112 : cpu.runtime_table_bytes;
    value.grouped_table_pool = cpu_i8_tensor(
        cpu.grouped_table_pool, {cpu.bank_count, grouped_stride});
    value.bank_ids = cpu_u8_tensor(
        cpu.bank_ids, {
            (int64_t)cpu.n_experts * cpu.out_per_expert, cpu.nsuper});
    value.rotation_signs = cpu_i8_tensor(
        cpu.rotation_signs, {(int64_t)cpu.rotation_signs.size()});
    if (cuda) {
        value.indices_packed =
            value.indices_packed.to(torch::kCUDA).contiguous();
        value.aux_packed =
            value.aux_packed.to(torch::kCUDA).contiguous();
        value.state_packed =
            value.state_packed.to(torch::kCUDA).contiguous();
        value.neuron_scale =
            value.neuron_scale.to(torch::kCUDA).contiguous();
        value.table_pool =
            value.table_pool.to(torch::kCUDA).contiguous();
        value.grouped_table_pool =
            value.grouped_table_pool.to(torch::kCUDA).contiguous();
        value.bank_ids =
            value.bank_ids.to(torch::kCUDA).contiguous();
        value.rotation_signs =
            value.rotation_signs.to(torch::kCUDA).contiguous();
    }
    return value;
}

static NepqWeight to_gpu_nepq(const NepqCpu & cpu) {
    return to_device_nepq(cpu, true);
}

static NepqWeight to_cpu_nepq(const NepqCpu & cpu) {
    return to_device_nepq(cpu, false);
}

static std::vector<int64_t> moe_output_row_indices(
        int local_experts,
        int source_out_per_expert,
        int64_t begin,
        int64_t end,
        bool paired) {
    if (local_experts <= 0 || begin < 0 ||
        begin >= end) {
        throw std::runtime_error(
            "invalid tensor-parallel MoE output slice");
    }
    const int64_t logical_extent = paired
        ? source_out_per_expert / 2
        : source_out_per_expert;
    if ((paired && source_out_per_expert % 2 != 0) ||
        end > logical_extent) {
        throw std::runtime_error(
            "tensor-parallel MoE output slice "
            "exceeds its logical width");
    }
    std::vector<int64_t> rows;
    rows.reserve(
        static_cast<size_t>(local_experts) *
        static_cast<size_t>(end - begin) *
        (paired ? 2u : 1u));
    for (int expert = 0;
         expert < local_experts; ++expert) {
        const int64_t base =
            static_cast<int64_t>(expert) *
            source_out_per_expert;
        for (int64_t row = begin; row < end; ++row) {
            rows.push_back(base + row);
        }
        if (paired) {
            const int64_t second =
                base + logical_extent;
            for (int64_t row = begin;
                 row < end; ++row) {
                rows.push_back(second + row);
            }
        }
    }
    return rows;
}

static NintCpu select_nint_cpu_rows(
        const NintCpu & source,
        const std::vector<int64_t> & rows) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (rows.empty()) {
        throw std::runtime_error(
            "cannot create an empty NINT MoE shard");
    }
    NintCpu result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    const size_t q_row_bytes =
        static_cast<size_t>(source.ng) *
        source.qbytes;
    result.q_packed.resize(
        rows.size() * q_row_bytes);
    result.sub_scale.resize(
        rows.size() * source.ng);
    result.sub_min.resize(
        rows.size() * source.ng);
    result.neuron_scale_h.resize(rows.size());
    result.neuron_min_h.resize(rows.size());
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NINT MoE shard row is out of range");
        }
        std::memcpy(
            result.q_packed.data() +
                destination * q_row_bytes,
            source.q_packed.data() +
                static_cast<size_t>(source_row) *
                    q_row_bytes,
            q_row_bytes);
        std::memcpy(
            result.sub_scale.data() +
                destination * source.ng,
            source.sub_scale.data() +
                static_cast<size_t>(source_row) *
                    source.ng,
            static_cast<size_t>(source.ng));
        std::memcpy(
            result.sub_min.data() +
                destination * source.ng,
            source.sub_min.data() +
                static_cast<size_t>(source_row) *
                    source.ng,
            static_cast<size_t>(source.ng));
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
        result.neuron_min_h[destination] =
            source.neuron_min_h[
                static_cast<size_t>(source_row)];
    }
    return result;
}

static Nint8ZeroCpu select_nint8_zero_cpu_rows(
        const Nint8ZeroCpu & source,
        const std::vector<int64_t> & rows) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (rows.empty()) {
        throw std::runtime_error(
            "cannot create an empty NINT8-0 MoE shard");
    }
    Nint8ZeroCpu result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    const size_t q_row_bytes =
        static_cast<size_t>(source.ng) * 32;
    result.q.resize(rows.size() * q_row_bytes);
    result.scale_h.resize(
        rows.size() * source.ng);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NINT8-0 MoE shard row is out of range");
        }
        std::memcpy(
            result.q.data() +
                destination * q_row_bytes,
            source.q.data() +
                static_cast<size_t>(source_row) *
                    q_row_bytes,
            q_row_bytes);
        std::memcpy(
            result.scale_h.data() +
                destination * source.ng,
            source.scale_h.data() +
                static_cast<size_t>(source_row) *
                    source.ng,
            static_cast<size_t>(source.ng) *
                sizeof(uint16_t));
    }
    return result;
}

static NvqCpu select_nvq_cpu_rows(
        const NvqCpu & source,
        const std::vector<int64_t> & rows) {
    if (source.shape.size() != 2 ||
        source.axis != 0 ||
        source.shape[0] != source.out ||
        rows.empty()) {
        throw std::runtime_error(
            "invalid NVQ MoE output shard source");
    }
    NvqCpu result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    result.neuron_scale_h.resize(rows.size());
    const int index_bits =
        nvq_index_bits(source.format);
    const bool delta =
        nvq_delta_format(source.format);
    const bool no_aux =
        nvq_no_aux_format(source.format);
    const int aux_items =
        delta ? source.ng :
        (no_aux ? 0 : source.nsign);
    const int aux_bits =
        delta ? 1 : (no_aux ? 0 : 7);
    result.sub_scale_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.sub_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(aux_items),
            aux_bits),
        0);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NVQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.sub_scale_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.sub_bits,
            result.sub_scale_packed,
            destination * source.ng *
                source.sub_bits,
            static_cast<size_t>(source.ng) *
                source.sub_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * index_bits,
            result.indices_packed,
            destination * source.nvec *
                index_bits,
            static_cast<size_t>(source.nvec) *
                index_bits);
        if (aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    aux_items * aux_bits,
                result.aux_packed,
                destination * aux_items *
                    aux_bits,
                static_cast<size_t>(aux_items) *
                    aux_bits);
        }
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
    }
    return result;
}

static NepqCpu select_nepq_cpu_rows(
        const NepqCpu & source,
        const std::vector<int64_t> & rows,
        int output_per_expert,
        int selected_experts = -1) {
    const int destination_experts = selected_experts < 0
        ? source.n_experts
        : selected_experts;
    const int source_rows =
        source.n_experts *
        source.out_per_expert;
    if (rows.empty() ||
        output_per_expert <= 0 ||
        destination_experts <= 0 ||
        static_cast<int>(rows.size()) !=
            destination_experts *
                output_per_expert) {
        throw std::runtime_error(
            "invalid NEPQ MoE output shard");
    }
    NepqCpu result = source;
    result.n_experts = destination_experts;
    result.out_per_expert = output_per_expert;
    result.neuron_scale_h.resize(rows.size());
    result.state_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.state_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            source.index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.aux_bits),
        0);
    result.bank_ids.resize(
        rows.size() * source.nsuper);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source_rows) {
            throw std::runtime_error(
                "NEPQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.state_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.state_bits,
            result.state_packed,
            destination * source.ng *
                source.state_bits,
            static_cast<size_t>(source.ng) *
                source.state_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * source.index_bits,
            result.indices_packed,
            destination * source.nvec *
                source.index_bits,
            static_cast<size_t>(source.nvec) *
                source.index_bits);
        if (source.aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    source.ng * source.aux_bits,
                result.aux_packed,
                destination * source.ng *
                    source.aux_bits,
                static_cast<size_t>(source.ng) *
                    source.aux_bits);
        }
        std::memcpy(
            result.bank_ids.data() +
                destination * source.nsuper,
            source.bank_ids.data() +
                static_cast<size_t>(source_row) *
                    source.nsuper,
            static_cast<size_t>(source.nsuper));
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
    }
    return result;
}

enum class MixedMoeFamily {
    Nint,
    Nint8Zero,
    Nvq,
    Nepq,
};

struct MixedMoePool {
    MixedMoeFamily family = MixedMoeFamily::Nint;
    NintWeight nint;
    NintWeight q8_zero;
    NvqWeight nvq;
    NepqWeight nepq;
    torch::Tensor expert_local;
    int local_experts = 0;
};

struct MixedMoeTransformKey {
    int block = 0;
    uint64_t seed = 0;

    bool operator==(const MixedMoeTransformKey & other) const {
        return block == other.block && seed == other.seed;
    }
};

struct MixedMoeTransformKeyHash {
    size_t operator()(const MixedMoeTransformKey & key) const {
        return ((size_t)key.block * 1315423911u) ^
            (size_t)(key.seed ^ (key.seed >> 32));
    }
};

struct MixedMoeActivationKey {
    int input_rows = 0;
    int groups = 0;
    int gs = 0;
    int device = 0;
    MixedMoeTransformKey transform;

    bool operator==(const MixedMoeActivationKey & other) const {
        return input_rows == other.input_rows && groups == other.groups &&
            gs == other.gs && device == other.device &&
            transform == other.transform;
    }
};

struct MixedMoeActivationKeyHash {
    size_t operator()(const MixedMoeActivationKey & key) const {
        size_t value = (size_t)key.input_rows;
        value = value * 1315423911u + (size_t)key.groups;
        value = value * 1315423911u + (size_t)key.gs;
        value = value * 1315423911u + (size_t)key.device;
        return value * 1315423911u +
            MixedMoeTransformKeyHash{}(key.transform);
    }
};

struct MixedMoeRuntime {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    bool partial_experts = false;
    std::vector<MixedMoePool> pools;
    mutable std::unordered_map<
        MixedMoeActivationKey, MoeActivationWorkspace,
        MixedMoeActivationKeyHash> activation_workspaces;

    MoeActivationWorkspace & activation_workspace(
            torch::Tensor x, int input_rows, int groups, int gs,
            MixedMoeTransformKey transform) const {
        MixedMoeActivationKey key{
            input_rows, groups, gs, x.get_device(), transform};
        auto found = activation_workspaces.find(key);
        if (found != activation_workspaces.end()) return found->second;
        MoeActivationWorkspace value;
        value.qx = torch::empty(
            {input_rows, groups * gs}, x.options().dtype(torch::kInt8));
        value.xscale = torch::empty(
            {input_rows, groups}, x.options().dtype(torch::kFloat32));
        return activation_workspaces.emplace(key, std::move(value)).first->second;
    }

    torch::Tensor forward(torch::Tensor x, const MoeRoutePlan & route) const {
        if (!x.is_cuda() || !x.is_contiguous() ||
            x.scalar_type() != torch::kFloat16 ||
            (x.dim() != 2 && x.dim() != 3) || x.size(-1) != neuron_len) {
            throw std::runtime_error(
                "mixed NINTM input must be contiguous CUDA f16 with exact K");
        }
        const int tokens = (int)route.ids.size(0);
        const int routes = (int)route.ids.size(1);
        if (route.n_experts != n_experts || x.size(0) != tokens ||
            (x.dim() == 3 && x.size(1) != routes)) {
            throw std::runtime_error("mixed NINTM input and route shape mismatch");
        }
        const int input_rows = x.dim() == 3 ? tokens * routes : tokens;
        auto output = partial_experts
            ? torch::zeros(
                {tokens, routes, out_per_expert},
                x.options().dtype(torch::kFloat16))
            : torch::empty(
                {tokens, routes, out_per_expert},
                x.options().dtype(torch::kFloat16));
        std::unordered_map<
            MixedMoeTransformKey, torch::Tensor,
            MixedMoeTransformKeyHash> prepared_inputs;
        const MixedMoeTransformKey identity{};
        prepared_inputs.emplace(identity, x);
        std::unordered_set<
            MixedMoeActivationKey, MixedMoeActivationKeyHash> quantized;
        static const bool disable_prefill_mma = [] {
            const char * value = std::getenv("MFQ_DISABLE_MOE_PREFILL_MMA");
            return value != nullptr && std::atoi(value) != 0;
        }();
        static const int prefill_mma_min_tokens = [] {
            const char * value = std::getenv("MFQ_MOE_PREFILL_MMA_MIN_TOKENS");
            return value == nullptr ? 256 : std::max(9, std::atoi(value));
        }();
        const bool use_f16_mma =
            !disable_prefill_mma && !g_force_moe_prefill_mma_off &&
            tokens >= prefill_mma_min_tokens && route.map_ready &&
            route.ids_dst.numel() == route.ids.numel();
        const bool use_kl_mmq = g_kl_mmq_mode != KlMmqMode::Default;
        if (use_kl_mmq) {
            TORCH_CHECK(
                route.map_ready && route.ids_dst.numel() == route.ids.numel(),
                "KLD mixed routed FP16 requires the compact route map");
        }

        for (const auto & pool : pools) {
            int gs = 24;
            int groups = 0;
            MixedMoeTransformKey transform{};
            torch::Tensor value = x;
            if (pool.family == MixedMoeFamily::Nint) {
                gs = (int)pool.nint.gs;
                groups = (int)pool.nint.ng;
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                gs = 32;
                groups = (int)pool.q8_zero.ng;
            } else if (pool.family == MixedMoeFamily::Nvq) {
                gs = (int)pool.nvq.gs;
                groups = (int)pool.nvq.ng;
            } else {
                groups = pool.nepq.ng;
                transform = {
                    pool.nepq.rotation_block,
                    pool.nepq.rotation_seed,
                };
                if (transform.block != 0) {
                    auto found = prepared_inputs.find(transform);
                    if (found == prepared_inputs.end()) {
                        auto flat = x.reshape({input_rows, neuron_len}).contiguous();
                        auto rotated = nepq_hadamard_input_cuda(
                            flat, pool.nepq.rotation_signs, transform.block);
                        value = x.dim() == 3
                            ? rotated.reshape({tokens, routes, neuron_len})
                            : rotated;
                        prepared_inputs.emplace(transform, value);
                    } else {
                        value = found->second;
                    }
                }
            }
            if (use_kl_mmq) {
                value = kl_mmq_prepare_activation(value);
                ++g_kl_mmq_moe_calls;
                if (pool.family == MixedMoeFamily::Nvq) {
                    nvq_moe_grouped_matmul_pool_f16_cuda(
                        pool.nvq.indices_packed, pool.nvq.aux_packed,
                        pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                        pool.nvq.codebook, value, pool.expert_local,
                        n_experts, pool.local_experts, out_per_expert,
                        neuron_len, pool.nvq.gs, pool.nvq.sub_bits,
                        pool.nvq.kernel_format, pool.nvq.sign_mode, output,
                        route.ids_dst, route.expert_bounds,
                        route.tile_bounds, route.tile_experts);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Nepq) {
                    nepq_moe_grouped_matmul_pool_f16_cuda(
                        pool.nepq.indices_packed, pool.nepq.aux_packed,
                        pool.nepq.state_packed, pool.nepq.neuron_scale,
                        pool.nepq.table_pool, pool.nepq.bank_ids, value,
                        pool.expert_local, n_experts, pool.local_experts,
                        out_per_expert, neuron_len, pool.nepq.state_bits,
                        pool.nepq.format, output, route.ids_dst,
                        route.expert_bounds, route.tile_bounds,
                        route.tile_experts);
                    continue;
                }
                ++g_kl_mmq_fallback_calls;
                throw std::runtime_error(
                    "KLD mixed routed FP16 encountered a non-VQ pool");
            }
            MixedMoeActivationKey activation_key{
                input_rows, groups, gs, x.get_device(), transform};
            auto & workspace = activation_workspace(
                value, input_rows, groups, gs, transform);
            const bool input_quantized =
                quantized.find(activation_key) != quantized.end();
            if (pool.family == MixedMoeFamily::Nint) {
                nint_moe_grouped_matmul_pool_ws_cuda(
                    pool.nint.q_packed, pool.nint.sub_scale, pool.nint.sub_min,
                    pool.nint.neuron_scale, pool.nint.neuron_min, value,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, gs, pool.nint.bits,
                    route.map_ready, input_quantized, output,
                    workspace.qx, workspace.xscale, route.counts, route.cursors,
                    route.ids_dst, route.expert_bounds, route.tile_bounds,
                    route.tile_experts);
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                nint8_zero_moe_grouped_matmul_pool_ws_cuda(
                    pool.q8_zero.q_packed, pool.q8_zero.q8_zero_scale, value,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, route.map_ready,
                    input_quantized, use_f16_mma, output,
                    workspace.qx, workspace.xscale,
                    route.counts, route.cursors, route.ids_dst,
                    route.expert_bounds, route.tile_bounds, route.tile_experts);
            } else if (pool.family == MixedMoeFamily::Nvq) {
                nvq_moe_grouped_matmul_pool_ws_cuda(
                    pool.nvq.indices_packed, pool.nvq.aux_packed,
                    pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                    pool.nvq.codebook, value, route.ids, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert, neuron_len,
                    pool.nvq.gs, pool.nvq.sub_bits, pool.nvq.kernel_format,
                    pool.nvq.sign_mode, input_quantized, output,
                    workspace.qx, workspace.xscale, route.ids_dst,
                    route.expert_bounds, route.tile_bounds, route.tile_experts);
            } else {
                nepq_moe_grouped_matmul_pool_ws_cuda(
                    pool.nepq.indices_packed, pool.nepq.aux_packed,
                    pool.nepq.state_packed, pool.nepq.neuron_scale,
                    pool.nepq.table_pool, pool.nepq.bank_ids,
                    pool.nepq.grouped_table_pool, value, route.ids,
                    pool.expert_local, n_experts, pool.local_experts,
                    out_per_expert, neuron_len, pool.nepq.state_bits,
                    pool.nepq.format, input_quantized, output,
                    workspace.qx, workspace.xscale, route.ids_dst,
                    route.expert_bounds, route.tile_bounds, route.tile_experts);
            }
            quantized.insert(activation_key);
        }
        return output;
    }

    bool supports_clamped_swiglu() const {
        return std::all_of(
            pools.begin(), pools.end(), [](const MixedMoePool & pool) {
                if (pool.family == MixedMoeFamily::Nepq ||
                    pool.family == MixedMoeFamily::Nint8Zero) return false;
                const int gs = pool.family == MixedMoeFamily::Nint
                    ? static_cast<int>(pool.nint.gs)
                    : static_cast<int>(pool.nvq.gs);
                return gs == 24 || gs == 28 || gs == 48;
            });
    }

    torch::Tensor forward_clamped_swiglu(
            torch::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) const {
        if (!supports_clamped_swiglu() ||
            !gate_up.is_cuda() || !gate_up.is_contiguous() ||
            gate_up.scalar_type() != torch::kFloat16 ||
            gate_up.dim() != 3 || gate_up.size(2) != 2 * neuron_len) {
            throw std::runtime_error(
                "mixed NINTM clamped SwiGLU input is unsupported");
        }
        const int tokens = static_cast<int>(route.ids.size(0));
        const int routes = static_cast<int>(route.ids.size(1));
        if (route.n_experts != n_experts ||
            gate_up.size(0) != tokens || gate_up.size(1) != routes) {
            throw std::runtime_error(
                "mixed NINTM clamped SwiGLU route shape mismatch");
        }
        const int input_rows = tokens * routes;
        auto output = partial_experts
            ? torch::zeros(
                {tokens, routes, out_per_expert},
                gate_up.options().dtype(torch::kFloat16))
            : torch::empty(
                {tokens, routes, out_per_expert},
                gate_up.options().dtype(torch::kFloat16));
        const MixedMoeTransformKey identity{};
        std::unordered_set<
            MixedMoeActivationKey, MixedMoeActivationKeyHash> quantized;

        for (const auto & pool : pools) {
            const int gs = pool.family == MixedMoeFamily::Nint
                ? static_cast<int>(pool.nint.gs)
                : pool.family == MixedMoeFamily::Nint8Zero
                    ? 32
                    : static_cast<int>(pool.nvq.gs);
            const int groups = pool.family == MixedMoeFamily::Nint
                ? static_cast<int>(pool.nint.ng)
                : pool.family == MixedMoeFamily::Nint8Zero
                    ? static_cast<int>(pool.q8_zero.ng)
                    : static_cast<int>(pool.nvq.ng);
            const MixedMoeActivationKey activation_key{
                input_rows, groups, gs, gate_up.get_device(), identity};
            auto & workspace = activation_workspace(
                gate_up, input_rows, groups, gs, identity);
            if (quantized.insert(activation_key).second) {
                nint_moe_quantize_swiglu_clamped_input_ws_cuda(
                    gate_up, gs, limit, workspace.qx, workspace.xscale);
            }
            if (pool.family == MixedMoeFamily::Nint) {
                nint_moe_grouped_matmul_pool_ws_cuda(
                    pool.nint.q_packed, pool.nint.sub_scale, pool.nint.sub_min,
                    pool.nint.neuron_scale, pool.nint.neuron_min, gate_up,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, gs, pool.nint.bits,
                    route.map_ready, true, output,
                    workspace.qx, workspace.xscale, route.counts, route.cursors,
                    route.ids_dst, route.expert_bounds, route.tile_bounds,
                    route.tile_experts);
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                nint8_zero_moe_grouped_matmul_pool_ws_cuda(
                    pool.q8_zero.q_packed, pool.q8_zero.q8_zero_scale, gate_up,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, route.map_ready,
                    true, false, output, workspace.qx, workspace.xscale,
                    route.counts, route.cursors, route.ids_dst,
                    route.expert_bounds, route.tile_bounds, route.tile_experts);
            } else {
                nvq_moe_grouped_matmul_pool_ws_cuda(
                    pool.nvq.indices_packed, pool.nvq.aux_packed,
                    pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                    pool.nvq.codebook, gate_up, route.ids, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert, neuron_len,
                    pool.nvq.gs, pool.nvq.sub_bits, pool.nvq.kernel_format,
                    pool.nvq.sign_mode, true, output,
                    workspace.qx, workspace.xscale, route.ids_dst,
                    route.expert_bounds, route.tile_bounds, route.tile_experts);
            }
        }
        return output;
    }
};

static int64_t tensor_storage_bytes(const torch::Tensor & value) {
    return value.defined()
        ? value.numel() * (int64_t)value.element_size()
        : 0;
}

static int64_t mixed_moe_storage_bytes(const MixedMoeRuntime & runtime) {
    int64_t bytes = 0;
    for (const auto & pool : runtime.pools) {
        bytes += tensor_storage_bytes(pool.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            bytes += tensor_storage_bytes(pool.nint.q_packed);
            bytes += tensor_storage_bytes(pool.nint.sub_scale);
            bytes += tensor_storage_bytes(pool.nint.sub_min);
            bytes += tensor_storage_bytes(pool.nint.neuron_scale);
            bytes += tensor_storage_bytes(pool.nint.neuron_min);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            bytes += tensor_storage_bytes(pool.q8_zero.q_packed);
            bytes += tensor_storage_bytes(pool.q8_zero.q8_zero_scale);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            bytes += tensor_storage_bytes(pool.nvq.indices_packed);
            bytes += tensor_storage_bytes(pool.nvq.aux_packed);
            bytes += tensor_storage_bytes(pool.nvq.sub_scale_packed);
            bytes += tensor_storage_bytes(pool.nvq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nvq.codebook);
        } else {
            bytes += tensor_storage_bytes(pool.nepq.indices_packed);
            bytes += tensor_storage_bytes(pool.nepq.aux_packed);
            bytes += tensor_storage_bytes(pool.nepq.state_packed);
            bytes += tensor_storage_bytes(pool.nepq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nepq.table_pool);
            bytes += tensor_storage_bytes(pool.nepq.grouped_table_pool);
            bytes += tensor_storage_bytes(pool.nepq.bank_ids);
            bytes += tensor_storage_bytes(pool.nepq.rotation_signs);
        }
    }
    return bytes;
}

static std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
        const NintMoeCpu & cpu, bool cuda) {
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->pools.reserve(cpu.pools.size());
    for (const auto & source : cpu.pools) {
        MixedMoePool pool;
        pool.local_experts = (int)source.expert_ids.size();
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            local[(size_t)source.expert_ids[(size_t)index]] = index;
        }
        pool.expert_local = torch::from_blob(
            local.data(), {(int64_t)local.size()},
            torch::TensorOptions().dtype(torch::kInt32))
            .clone();
        if (cuda) {
            pool.expert_local =
                pool.expert_local.to(torch::kCUDA).contiguous();
        }
        const int expected_rows = pool.local_experts * cpu.out_per_expert;
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = cuda
                ? to_gpu_nint8_zero(source.q8_zero)
                : to_cpu_nint8_zero(source.q8_zero);
            if (pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed NINT8-0 cohort shape mismatch");
            }
        } else if (source.dtype != "NINTM" &&
                   source.dtype.rfind("NINT", 0) == 0) {
            pool.family = MixedMoeFamily::Nint;
            pool.nint = cuda
                ? to_gpu_nint(source.weight)
                : to_cpu_nint(source.weight);
            if (pool.nint.out != expected_rows ||
                pool.nint.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NINT cohort shape mismatch");
            }
        } else if (source.dtype.rfind("NEPQ", 0) == 0) {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = unpack_nepq(
                source.payload, source.dtype, source.runtime_payload);
            if (parsed.n_experts != pool.local_experts ||
                parsed.out_per_expert != cpu.out_per_expert ||
                parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NEPQ cohort shape mismatch");
            }
            pool.nepq = cuda
                ? to_gpu_nepq(parsed)
                : to_cpu_nepq(parsed);
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = unpack_nvq(source.payload, source.dtype);
            if (parsed.out != expected_rows || parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NVQ/NPQ cohort shape mismatch");
            }
            pool.nvq = cuda
                ? to_gpu_nvq(parsed)
                : to_cpu_nvq(parsed);
        }
        runtime->pools.push_back(std::move(pool));
    }
    return runtime;
}

static NintMoeWeight wrap_mixed_moe_runtime(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    NintMoeWeight result;
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.partial_experts = runtime->partial_experts;
    result.hetero_supported = false;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    result.mixed_forward = [runtime](
            torch::Tensor x, const MoeRoutePlan & route) {
        return runtime->forward(x, route);
    };
    if (runtime->supports_clamped_swiglu()) {
        result.mixed_clamped_swiglu_forward = [runtime](
                torch::Tensor gate_up,
                const MoeRoutePlan & route,
                double limit) {
            return runtime->forward_clamped_swiglu(gate_up, route, limit);
        };
    }
    return result;
}

static NintMoeWeight to_gpu_mixed_moe(const NintMoeCpu & cpu) {
    return wrap_mixed_moe_runtime(make_mixed_moe_runtime(cpu, true));
}

static NintMoeWeight to_cuda_device_moe_output_slice(
        const NintMoeCpu & cpu,
        int64_t begin,
        int64_t end,
        bool paired,
        int device) {
    const int output_per_expert =
        static_cast<int>(
            (end - begin) *
            (paired ? 2 : 1));
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(),
        [](const NintMoeCpuPool & pool) {
            return pool.dtype != "NINTM" &&
                pool.dtype != "NINT8-0" &&
                pool.dtype.rfind("NINT", 0) == 0;
        });
    c10::cuda::CUDAGuard guard(device);
    if (all_nint) {
        NintMoeCpu sliced = cpu;
        sliced.out_per_expert =
            output_per_expert;
        for (size_t pool_index = 0;
             pool_index < cpu.pools.size();
             ++pool_index) {
            const auto & source =
                cpu.pools[pool_index];
            auto rows = moe_output_row_indices(
                static_cast<int>(
                    source.expert_ids.size()),
                cpu.out_per_expert,
                begin, end, paired);
            sliced.pools[pool_index].weight =
                select_nint_cpu_rows(
                    source.weight, rows);
        }
        return to_gpu_nint_moe(sliced);
    }

    auto runtime =
        std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert =
        output_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->pools.reserve(cpu.pools.size());
    for (const auto & source : cpu.pools) {
        MixedMoePool pool;
        pool.local_experts =
            static_cast<int>(
                source.expert_ids.size());
        std::vector<int32_t> local(
            static_cast<size_t>(cpu.n_experts),
            -1);
        for (int index = 0;
             index < pool.local_experts;
             ++index) {
            local[static_cast<size_t>(
                source.expert_ids[
                    static_cast<size_t>(index)])] =
                index;
        }
        pool.expert_local = torch::from_blob(
            local.data(),
            {static_cast<int64_t>(local.size())},
            torch::TensorOptions().dtype(
                torch::kInt32))
            .clone().to(torch::kCUDA).contiguous();
        auto rows = moe_output_row_indices(
            pool.local_experts,
            cpu.out_per_expert,
            begin, end, paired);
        if (source.dtype == "NINT8-0") {
            pool.family =
                MixedMoeFamily::Nint8Zero;
            pool.q8_zero =
                to_gpu_nint8_zero(
                    select_nint8_zero_cpu_rows(
                        source.q8_zero, rows));
        } else if (
                source.dtype != "NINTM" &&
                source.dtype.rfind("NINT", 0) == 0) {
            pool.family =
                MixedMoeFamily::Nint;
            pool.nint =
                to_gpu_nint(
                    select_nint_cpu_rows(
                        source.weight, rows));
        } else if (
                source.dtype.rfind("NEPQ", 0) == 0) {
            pool.family =
                MixedMoeFamily::Nepq;
            auto parsed = unpack_nepq(
                source.payload,
                source.dtype,
                source.runtime_payload);
            pool.nepq = to_gpu_nepq(
                select_nepq_cpu_rows(
                    parsed, rows,
                    output_per_expert));
        } else {
            pool.family =
                MixedMoeFamily::Nvq;
            auto parsed = unpack_nvq(
                source.payload,
                source.dtype);
            pool.nvq = to_gpu_nvq(
                select_nvq_cpu_rows(
                    parsed, rows));
        }
        runtime->pools.push_back(
            std::move(pool));
    }
    return wrap_mixed_moe_runtime(runtime);
}

static NintMoeWeight to_cuda_device_moe_expert_slice(
        const NintMoeCpu & cpu,
        int64_t expert_begin,
        int64_t expert_end,
        int device) {
    if (expert_begin < 0 || expert_begin >= expert_end ||
            expert_end > cpu.n_experts) {
        throw std::runtime_error(
            "invalid tensor-parallel MoE expert shard");
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(),
        [](const NintMoeCpuPool & pool) {
            return pool.dtype != "NINTM" &&
                pool.dtype != "NINT8-0" &&
                pool.dtype.rfind("NINT", 0) == 0;
        });
    c10::cuda::CUDAGuard guard(device);
    if (all_nint) {
        NintMoeCpu sliced = cpu;
        sliced.pools.clear();
        for (const auto & source : cpu.pools) {
            NintMoeCpuPool destination = source;
            destination.expert_ids.clear();
            std::vector<int64_t> rows;
            for (size_t local = 0;
                 local < source.expert_ids.size(); ++local) {
                const int expert = source.expert_ids[local];
                if (expert < expert_begin || expert >= expert_end) continue;
                destination.expert_ids.push_back(expert);
                for (int row = 0; row < cpu.out_per_expert; ++row) {
                    rows.push_back(
                        static_cast<int64_t>(local) * cpu.out_per_expert + row);
                }
            }
            if (rows.empty()) continue;
            destination.weight = select_nint_cpu_rows(
                source.weight, rows);
            sliced.pools.push_back(std::move(destination));
        }
        auto result = to_gpu_nint_moe(sliced);
        result.partial_experts = true;
        return result;
    }

    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->partial_experts = true;
    for (const auto & source : cpu.pools) {
        std::vector<int> expert_ids;
        std::vector<int64_t> rows;
        for (size_t local = 0;
             local < source.expert_ids.size(); ++local) {
            const int expert = source.expert_ids[local];
            if (expert < expert_begin || expert >= expert_end) continue;
            expert_ids.push_back(expert);
            for (int row = 0; row < cpu.out_per_expert; ++row) {
                rows.push_back(
                    static_cast<int64_t>(local) * cpu.out_per_expert + row);
            }
        }
        if (rows.empty()) continue;

        MixedMoePool pool;
        pool.local_experts = static_cast<int>(expert_ids.size());
        std::vector<int32_t> local_map(
            static_cast<size_t>(cpu.n_experts), -1);
        for (size_t local = 0; local < expert_ids.size(); ++local) {
            local_map[static_cast<size_t>(expert_ids[local])] =
                static_cast<int32_t>(local);
        }
        pool.expert_local = torch::from_blob(
            local_map.data(),
            {static_cast<int64_t>(local_map.size())},
            torch::TensorOptions().dtype(torch::kInt32))
            .clone().to(torch::kCUDA).contiguous();
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = to_gpu_nint8_zero(
                select_nint8_zero_cpu_rows(source.q8_zero, rows));
        } else if (source.dtype != "NINTM" &&
                source.dtype.rfind("NINT", 0) == 0) {
            pool.family = MixedMoeFamily::Nint;
            pool.nint = to_gpu_nint(
                select_nint_cpu_rows(source.weight, rows));
        } else if (source.dtype.rfind("NEPQ", 0) == 0) {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = unpack_nepq(
                source.payload, source.dtype, source.runtime_payload);
            pool.nepq = to_gpu_nepq(select_nepq_cpu_rows(
                parsed, rows, cpu.out_per_expert,
                pool.local_experts));
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = unpack_nvq(source.payload, source.dtype);
            pool.nvq = to_gpu_nvq(
                select_nvq_cpu_rows(parsed, rows));
        }
        runtime->pools.push_back(std::move(pool));
    }
    if (runtime->pools.empty()) {
        throw std::runtime_error(
            "tensor-parallel MoE expert shard has no owned experts");
    }
    return wrap_mixed_moe_runtime(runtime);
}

static torch::Tensor copy_cpu_weight_to_cuda(
        const torch::Tensor & source) {
    return source.defined()
        ? source.to(torch::kCUDA).contiguous()
        : torch::Tensor();
}

static NintWeight copy_cpu_nint_to_cuda(const NintWeight & source) {
    NintWeight result = source;
    result.workspaces.clear();
    result.q_packed = copy_cpu_weight_to_cuda(source.q_packed);
    result.q8_zero_scale =
        copy_cpu_weight_to_cuda(source.q8_zero_scale);
    result.sub_scale = copy_cpu_weight_to_cuda(source.sub_scale);
    result.sub_min = copy_cpu_weight_to_cuda(source.sub_min);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.neuron_min = copy_cpu_weight_to_cuda(source.neuron_min);
    return result;
}

static NintMoeWeight stage_cpu_nint_moe(
        const std::shared_ptr<MixedMoeRuntime> & cpu) {
    if (!cpu) throw std::runtime_error("missing CPU-offloaded MoE state");
    NintMoeWeight result;
    result.n_experts = cpu->n_experts;
    result.out_per_expert = cpu->out_per_expert;
    result.neuron_len = cpu->neuron_len;
    result.pools.reserve(cpu->pools.size());
    std::vector<int32_t> expert_pool(
        static_cast<size_t>(cpu->n_experts), -1);
    std::vector<int32_t> expert_local(
        static_cast<size_t>(cpu->n_experts), -1);
    for (int pool_index = 0;
         pool_index < static_cast<int>(cpu->pools.size());
         ++pool_index) {
        const auto & source =
            cpu->pools.at(static_cast<size_t>(pool_index));
        if (source.family != MixedMoeFamily::Nint ||
                !source.expert_local.is_cpu() ||
                !source.expert_local.is_contiguous() ||
                source.expert_local.scalar_type() != torch::kInt32 ||
                source.expert_local.numel() != cpu->n_experts) {
            throw std::runtime_error(
                "pure NINT MoE staging received an incompatible pool");
        }
        NintMoePoolWeight pool;
        pool.weight = copy_cpu_nint_to_cuda(source.nint);
        pool.expert_local =
            copy_cpu_weight_to_cuda(source.expert_local);
        pool.local_experts = source.local_experts;
        const auto * local =
            source.expert_local.data_ptr<int32_t>();
        for (int expert = 0; expert < cpu->n_experts; ++expert) {
            const int local_index = local[expert];
            if (local_index < 0) continue;
            if (local_index >= source.local_experts ||
                    expert_pool.at(static_cast<size_t>(expert)) >= 0) {
                throw std::runtime_error(
                    "pure NINT MoE staging has invalid expert ownership");
            }
            expert_pool.at(static_cast<size_t>(expert)) =
                pool_index;
            expert_local.at(static_cast<size_t>(expert)) =
                local_index;
        }
        result.pools.push_back(std::move(pool));
    }
    if (std::any_of(
            expert_pool.begin(), expert_pool.end(),
            [](int value) { return value < 0; })) {
        throw std::runtime_error(
            "pure NINT MoE staging does not cover every expert");
    }
    initialize_nint_moe_dispatch(
        result, expert_pool, expert_local);
    return result;
}

static NvqWeight copy_cpu_nvq_to_cuda(const NvqWeight & source) {
    NvqWeight result = source;
    result.workspaces.clear();
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.sub_scale_packed =
        copy_cpu_weight_to_cuda(source.sub_scale_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.codebook = copy_cpu_weight_to_cuda(source.codebook);
    return result;
}

static NepqWeight copy_cpu_nepq_to_cuda(const NepqWeight & source) {
    NepqWeight result = source;
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.state_packed =
        copy_cpu_weight_to_cuda(source.state_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.table_pool = copy_cpu_weight_to_cuda(source.table_pool);
    result.grouped_table_pool =
        copy_cpu_weight_to_cuda(source.grouped_table_pool);
    result.bank_ids = copy_cpu_weight_to_cuda(source.bank_ids);
    result.rotation_signs =
        copy_cpu_weight_to_cuda(source.rotation_signs);
    return result;
}

static NintMoeWeight stage_cpu_mixed_moe(
        const std::shared_ptr<MixedMoeRuntime> & cpu) {
    if (!cpu) throw std::runtime_error("missing CPU-offloaded MoE state");
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu->n_experts;
    runtime->out_per_expert = cpu->out_per_expert;
    runtime->neuron_len = cpu->neuron_len;
    runtime->pools.reserve(cpu->pools.size());
    for (const auto & source : cpu->pools) {
        MixedMoePool pool;
        pool.family = source.family;
        pool.local_experts = source.local_experts;
        pool.expert_local =
            copy_cpu_weight_to_cuda(source.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            pool.nint = copy_cpu_nint_to_cuda(source.nint);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            pool.q8_zero = copy_cpu_nint_to_cuda(source.q8_zero);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            pool.nvq = copy_cpu_nvq_to_cuda(source.nvq);
        } else {
            pool.nepq = copy_cpu_nepq_to_cuda(source.nepq);
        }
        runtime->pools.push_back(std::move(pool));
    }
    return wrap_mixed_moe_runtime(runtime);
}

static NintMoeWeight cpu_mixed_moe_metadata(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    NintMoeWeight result;
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.hetero_supported = false;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    return result;
}

static NintMoeCpu load_nint_moe_cpu(
        const MfqFile & mfq, const std::string & name) {
    if (mfq.record(name).dtype != "NINTM") {
        throw std::runtime_error("expert tensor must use NINTM: " + name);
    }
    auto cpu = unpack_nint_moe(mfq.read_blob(name));
    if (mfq.has_expert_overlay(name)) {
        auto delta = unpack_nint_moe_delta(
            mfq.read_expert_overlay(name));
        if (delta.n_experts != cpu.n_experts ||
            delta.out_per_expert != cpu.out_per_expert ||
            delta.neuron_len != cpu.neuron_len) {
            throw std::runtime_error(
                "expert overlay shape mismatch: " + name);
        }
        cpu.pools.insert(
            cpu.pools.end(),
            std::make_move_iterator(delta.pools.begin()),
            std::make_move_iterator(delta.pools.end()));
    }
    return cpu;
}

struct MoeCacheTransfer {
    const uint8_t * source = nullptr;
    uint8_t * destination = nullptr;
    int64_t nbytes = 0;
    bool packed_weight = false;
};

struct MoeCachedCohort;
class MoeCachedSource;

static int64_t tensor_nbytes(const torch::Tensor & value) {
    return value.defined()
        ? value.numel() * static_cast<int64_t>(value.element_size())
        : 0;
}

static std::vector<torch::Tensor> moe_cache_fields(
        const MixedMoePool & pool) {
    if (pool.family == MixedMoeFamily::Nint) {
        return {
            pool.nint.q_packed,
            pool.nint.sub_scale,
            pool.nint.sub_min,
            pool.nint.neuron_scale,
            pool.nint.neuron_min,
        };
    }
    if (pool.family == MixedMoeFamily::Nint8Zero) {
        return {
            pool.q8_zero.q_packed,
            pool.q8_zero.q8_zero_scale,
        };
    }
    if (pool.family == MixedMoeFamily::Nvq) {
        return {
            pool.nvq.indices_packed,
            pool.nvq.aux_packed,
            pool.nvq.sub_scale_packed,
            pool.nvq.neuron_scale,
        };
    }
    return {
        pool.nepq.indices_packed,
        pool.nepq.aux_packed,
        pool.nepq.state_packed,
        pool.nepq.neuron_scale,
        pool.nepq.bank_ids,
    };
}

static void validate_nepq_expert_boundaries(
        const MixedMoePool & pool,
        int out_per_expert,
        int neuron_len) {
    if (pool.family != MixedMoeFamily::Nepq) return;
    const int index_bits =
        pool.nepq.format == 9 ? 6 :
        pool.nepq.format == 7 ? 7 :
        pool.nepq.format == 8 ? 9 :
        pool.nepq.format == 1 ? 11 : 0;
    const int aux_bits =
        pool.nepq.format == 8 || pool.nepq.format == 1 ? 1 : 0;
    const int nvec = neuron_len / 8;
    const int ng = (neuron_len + 23) / 24;
    const int nsuper = (ng + 3) / 4;
    const std::vector<int64_t> bits{
        static_cast<int64_t>(out_per_expert) * nvec * index_bits,
        static_cast<int64_t>(out_per_expert) * ng * aux_bits,
        static_cast<int64_t>(out_per_expert) * ng *
            pool.nepq.state_bits,
        static_cast<int64_t>(out_per_expert) * 32,
        static_cast<int64_t>(out_per_expert) * nsuper * 8,
    };
    if (index_bits == 0 ||
        std::any_of(bits.begin(), bits.end(), [](int64_t value) {
            return value % 8 != 0;
        })) {
        throw std::runtime_error(
            "NEPQ expert payload is not byte aligned for GPU caching");
    }
}

static std::string moe_cache_signature(
        const MixedMoePool & pool,
        int out_per_expert,
        int neuron_len) {
    std::ostringstream stream;
    stream << static_cast<int>(pool.family)
           << ":o" << out_per_expert
           << ":k" << neuron_len;
    if (pool.family == MixedMoeFamily::Nint) {
        stream << ":b" << pool.nint.bits
               << ":g" << pool.nint.gs
               << ":n" << pool.nint.ng
               << ":q" << pool.nint.qbytes;
    } else if (pool.family == MixedMoeFamily::Nint8Zero) {
        stream << ":g32:n" << pool.q8_zero.ng;
    } else if (pool.family == MixedMoeFamily::Nvq) {
        stream << ":f" << pool.nvq.format
               << ":kf" << pool.nvq.kernel_format
               << ":s" << pool.nvq.sub_bits
               << ":g" << pool.nvq.gs
               << ":n" << pool.nvq.ng
               << ":sm" << pool.nvq.sign_mode;
    } else {
        stream << ":f" << pool.nepq.format
               << ":s" << pool.nepq.state_bits
               << ":g" << pool.nepq.ng
               << ":r" << pool.nepq.rotation_block;
    }
    const auto fields = moe_cache_fields(pool);
    for (const auto & field : fields) {
        if (!field.defined() || !field.is_cpu() || !field.is_contiguous()) {
            throw std::runtime_error(
                "MoE cache source fields must be contiguous CPU tensors");
        }
        if (field.numel() % pool.local_experts != 0) {
            throw std::runtime_error(
                "MoE cache source field cannot be split by expert");
        }
        stream << ":" << static_cast<int>(field.scalar_type())
               << "x" << field.numel() / pool.local_experts;
    }
    return stream.str();
}

struct MoeGpuArena {
    std::string signature;
    int64_t slot_bytes = 0;
    int minimum_slots = 0;
    int registered_experts = 0;
    int prototype_experts = 0;
    int slots = 0;
    std::vector<torch::Tensor> prototypes;
    std::vector<int64_t> elements_per_expert;
    std::vector<torch::Tensor> fields;
    std::unique_ptr<mfq::MoeCacheSlotBook> book;
};

struct MoePinnedStage {
    torch::Tensor host;
    cudaEvent_t done = nullptr;
    bool pending = false;
};

struct MoeCacheStats {
    int64_t demand_hits = 0;
    int64_t demand_misses = 0;
    int64_t prefetch_hits = 0;
    int64_t prefetch_misses = 0;
    int64_t evictions = 0;
    int64_t h2d_bytes = 0;
    int64_t route_d2h_bytes = 0;
    int64_t hetero_dispatches = 0;
    int64_t full_projection_fallbacks = 0;
};

class MoeExpertCache : public std::enable_shared_from_this<MoeExpertCache> {
public:
    explicit MoeExpertCache(int64_t budget_bytes)
        : budget_bytes_(budget_bytes) {
        if (budget_bytes_ <= 0) {
            throw std::invalid_argument(
                "MoE GPU cache budget must be positive");
        }
        MFQ_CUDA_CHECK(cudaStreamCreateWithFlags(
            &weight_stream_, cudaStreamNonBlocking));
        MFQ_CUDA_CHECK(cudaStreamCreateWithFlags(
            &route_stream_, cudaStreamNonBlocking));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &compute_done_, cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &transfer_ready_, cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &route_input_ready_, cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &route_done_, cudaEventDisableTiming));
        stages_.resize(4);
        for (auto & stage : stages_) {
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &stage.done, cudaEventDisableTiming));
        }
    }

    ~MoeExpertCache() {
        for (auto & stage : stages_) {
            if (stage.done != nullptr) cudaEventDestroy(stage.done);
        }
        if (compute_done_ != nullptr) cudaEventDestroy(compute_done_);
        if (transfer_ready_ != nullptr) {
            cudaEventDestroy(transfer_ready_);
        }
        if (route_input_ready_ != nullptr) {
            cudaEventDestroy(route_input_ready_);
        }
        if (route_done_ != nullptr) {
            cudaEventDestroy(route_done_);
        }
        if (route_stream_ != nullptr) cudaStreamDestroy(route_stream_);
        if (weight_stream_ != nullptr) cudaStreamDestroy(weight_stream_);
    }

    std::shared_ptr<MoeCachedSource> register_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> cpu,
        int minimum_slots,
        int layer_id,
        std::string projection_role);

    MoeGpuArena * register_cohort(
            const MixedMoePool & pool,
            int out_per_expert,
            int neuron_len,
            int minimum_slots) {
        if (finalized_) {
            throw std::runtime_error(
                "cannot register a MoE source after cache finalization");
        }
        validate_nepq_expert_boundaries(
            pool, out_per_expert, neuron_len);
        const std::string signature =
            moe_cache_signature(pool, out_per_expert, neuron_len);
        const auto fields = moe_cache_fields(pool);
        auto found = arenas_.find(signature);
        if (found == arenas_.end()) {
            auto arena = std::make_unique<MoeGpuArena>();
            arena->signature = signature;
            arena->minimum_slots =
                std::min(minimum_slots, pool.local_experts);
            arena->registered_experts = pool.local_experts;
            arena->prototype_experts = pool.local_experts;
            for (const auto & field : fields) {
                const int64_t elements =
                    field.numel() / pool.local_experts;
                arena->prototypes.push_back(field);
                arena->elements_per_expert.push_back(elements);
                arena->slot_bytes +=
                    elements * static_cast<int64_t>(field.element_size());
            }
            MoeGpuArena * result = arena.get();
            arenas_.emplace(signature, std::move(arena));
            return result;
        }
        MoeGpuArena * arena = found->second.get();
        arena->minimum_slots = std::max(
            arena->minimum_slots,
            std::min(minimum_slots, pool.local_experts));
        arena->registered_experts += pool.local_experts;
        if (arena->prototypes.size() != fields.size()) {
            throw std::runtime_error(
                "MoE cache signature merged incompatible field counts");
        }
        return arena;
    }

    void finalize();

    void set_profile(mfq::MoeCacheProfile profile) {
        if (finalized_ || !sources_.empty()) {
            throw std::runtime_error(
                "MoE cache profile must be set before source registration");
        }
        profile_ = std::move(profile);
    }

    bool finalized() const noexcept {
        return finalized_;
    }

    bool has_sources() const noexcept {
        return !sources_.empty();
    }

    int64_t budget_bytes() const noexcept {
        return budget_bytes_;
    }

    int64_t allocated_bytes() const noexcept {
        return allocated_bytes_;
    }

    const MoeCacheStats & stats() const noexcept {
        return stats_;
    }

    void prepare(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch);

    void prewarm();

    std::vector<int32_t> read_route_experts(
            const torch::Tensor & ids,
            int n_experts) {
        if (!ids.is_cuda() || !ids.is_contiguous() ||
                ids.scalar_type() != torch::kInt32 ||
                ids.dim() != 2) {
            throw std::runtime_error(
                "cached MoE routes must be contiguous CUDA int32");
        }
        const int64_t count = ids.numel();
        if (count <= 0) {
            throw std::runtime_error(
                "cached MoE route list is empty");
        }
        if (!route_host_.defined() ||
                route_host_.numel() < count) {
            route_host_ = torch::empty(
                {count},
                torch::TensorOptions()
                    .device(torch::kCPU)
                    .dtype(torch::kInt32)
                    .pinned_memory(true));
        }
        auto current =
            at::cuda::getCurrentCUDAStream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(
            route_input_ready_, current));
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(
            route_stream_, route_input_ready_, 0));
        const int64_t nbytes =
            count * static_cast<int64_t>(sizeof(int32_t));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            route_host_.data_ptr<int32_t>(),
            ids.data_ptr<int32_t>(),
            static_cast<size_t>(nbytes),
            cudaMemcpyDeviceToHost,
            route_stream_));
        MFQ_CUDA_CHECK(cudaEventRecord(
            route_done_, route_stream_));
        MFQ_CUDA_CHECK(cudaEventSynchronize(route_done_));
        stats_.route_d2h_bytes += nbytes;
        const auto * values =
            route_host_.data_ptr<int32_t>();
        std::vector<int32_t> result(
            values, values + count);
        std::sort(result.begin(), result.end());
        result.erase(
            std::unique(result.begin(), result.end()),
            result.end());
        for (int expert : result) {
            if (expert < 0 || expert >= n_experts) {
                throw std::runtime_error(
                    "MoE route selected an out-of-range expert");
            }
        }
        return result;
    }

    void record_compute_use() {
        MFQ_CUDA_CHECK(cudaEventRecord(
            compute_done_,
            at::cuda::getCurrentCUDAStream().stream()));
        compute_done_recorded_ = true;
    }

    void count_full_projection_fallback() {
        ++stats_.full_projection_fallbacks;
    }

    void count_hetero_dispatch() {
        ++stats_.hetero_dispatches;
    }

    void print_stats(std::ostream & stream) const {
        stream << "moe_cache_stats"
               << " budget_bytes=" << budget_bytes_
               << " allocated_bytes=" << allocated_bytes_
               << " host_bytes=" << host_bytes_
               << " demand_hits=" << stats_.demand_hits
               << " demand_misses=" << stats_.demand_misses
               << " prefetch_hits=" << stats_.prefetch_hits
               << " prefetch_misses=" << stats_.prefetch_misses
               << " evictions=" << stats_.evictions
               << " h2d_bytes=" << stats_.h2d_bytes
               << " route_d2h_bytes="
               << stats_.route_d2h_bytes
               << " hetero_dispatches="
               << stats_.hetero_dispatches
               << " full_projection_fallbacks="
               << stats_.full_projection_fallbacks
               << "\n";
    }

private:
    friend class MoeCachedSource;

    MoePinnedStage & acquire_stage(int64_t required_bytes) {
        for (size_t offset = 0; offset < stages_.size(); ++offset) {
            const size_t index =
                (next_stage_ + offset) % stages_.size();
            auto & stage = stages_[index];
            if (!stage.pending ||
                cudaEventQuery(stage.done) == cudaSuccess) {
                stage.pending = false;
                next_stage_ = (index + 1) % stages_.size();
                if (!stage.host.defined() ||
                    stage.host.numel() < required_bytes) {
                    int64_t capacity = 1;
                    while (capacity < required_bytes) capacity *= 2;
                    stage.host = torch::empty(
                        {capacity},
                        torch::TensorOptions()
                            .device(torch::kCPU)
                            .dtype(torch::kUInt8)
                            .pinned_memory(true));
                }
                return stage;
            }
            (void)cudaGetLastError();
        }
        auto & stage = stages_[next_stage_];
        MFQ_CUDA_CHECK(cudaEventSynchronize(stage.done));
        stage.pending = false;
        next_stage_ = (next_stage_ + 1) % stages_.size();
        if (!stage.host.defined() ||
            stage.host.numel() < required_bytes) {
            int64_t capacity = 1;
            while (capacity < required_bytes) capacity *= 2;
            stage.host = torch::empty(
                {capacity},
                torch::TensorOptions()
                    .device(torch::kCPU)
                    .dtype(torch::kUInt8)
                    .pinned_memory(true));
        }
        return stage;
    }

    void submit_transfers(
            const std::vector<MoeCacheTransfer> & transfers,
            bool waits_for_compute,
            bool wait_on_compute_stream) {
        int64_t total = 0;
        for (const auto & transfer : transfers) {
            if (transfer.nbytes < 0 ||
                (transfer.nbytes > 0 &&
                 (transfer.source == nullptr ||
                  transfer.destination == nullptr))) {
                throw std::runtime_error(
                    "invalid MoE cache transfer");
            }
            total += transfer.nbytes;
        }
        if (total == 0) {
            if (wait_on_compute_stream &&
                    transfer_ready_recorded_) {
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                    at::cuda::getCurrentCUDAStream().stream(),
                    transfer_ready_, 0));
            }
            return;
        }
        auto & stage = acquire_stage(total);
        auto * staging = stage.host.data_ptr<uint8_t>();
        int64_t offset = 0;
        for (const auto & transfer : transfers) {
            if (transfer.nbytes == 0) continue;
            std::memcpy(
                staging + offset,
                transfer.source,
                static_cast<size_t>(transfer.nbytes));
            offset += transfer.nbytes;
        }
        if (waits_for_compute && compute_done_recorded_) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                weight_stream_, compute_done_, 0));
        }
        offset = 0;
        for (const auto & transfer : transfers) {
            if (transfer.nbytes == 0) continue;
            MFQ_CUDA_CHECK(cudaMemcpyAsync(
                transfer.destination,
                staging + offset,
                static_cast<size_t>(transfer.nbytes),
                cudaMemcpyHostToDevice,
                weight_stream_));
            if (transfer.packed_weight) {
                stats_.h2d_bytes += transfer.nbytes;
            }
            offset += transfer.nbytes;
        }
        MFQ_CUDA_CHECK(cudaEventRecord(stage.done, weight_stream_));
        MFQ_CUDA_CHECK(cudaEventRecord(
            transfer_ready_, weight_stream_));
        transfer_ready_recorded_ = true;
        stage.pending = true;
        if (wait_on_compute_stream) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                at::cuda::getCurrentCUDAStream().stream(),
                transfer_ready_, 0));
        }
    }

    void invalidate(const mfq::MoeCacheKey & key, int slot);

    int64_t budget_bytes_ = 0;
    int64_t allocated_bytes_ = 0;
    int64_t host_bytes_ = 0;
    bool finalized_ = false;
    cudaStream_t weight_stream_ = nullptr;
    cudaStream_t route_stream_ = nullptr;
    cudaEvent_t compute_done_ = nullptr;
    bool compute_done_recorded_ = false;
    cudaEvent_t transfer_ready_ = nullptr;
    bool transfer_ready_recorded_ = false;
    cudaEvent_t route_input_ready_ = nullptr;
    cudaEvent_t route_done_ = nullptr;
    torch::Tensor route_host_;
    std::vector<MoePinnedStage> stages_;
    size_t next_stage_ = 0;
    std::unordered_map<std::string, std::unique_ptr<MoeGpuArena>> arenas_;
    std::vector<std::shared_ptr<MoeCachedSource>> sources_;
    std::optional<mfq::MoeCacheProfile> profile_;
    std::vector<mfq::MoeProfileCandidate> prewarm_selected_;
    MoeCacheStats stats_;
};

struct MoeCachedCohort {
    int index = -1;
    const MixedMoePool * cpu = nullptr;
    MoeGpuArena * arena = nullptr;
    std::vector<torch::Tensor> cpu_fields;
    std::vector<int64_t> bytes_per_expert;
    std::vector<int32_t> expert_to_local;
    std::vector<int32_t> host_map;
    bool map_dirty = false;
    MixedMoePool active;
};

class MoeCachedSource : public std::enable_shared_from_this<MoeCachedSource> {
public:
    MoeCachedSource(
            MoeExpertCache * cache,
            int id,
            std::string name,
            std::shared_ptr<MixedMoeRuntime> cpu,
            int minimum_slots,
            int layer_id,
            std::string projection_role)
        : cache_(cache),
          id_(id),
          name_(std::move(name)),
          layer_id_(layer_id),
          projection_role_(std::move(projection_role)),
          cpu_(std::move(cpu)),
          expert_to_cohort_(
              static_cast<size_t>(cpu_->n_experts), -1),
          expert_to_local_(
              static_cast<size_t>(cpu_->n_experts), -1) {
        if (minimum_slots <= 0) {
            throw std::runtime_error(
                "MoE cache source minimum slots must be positive");
        }
        cohorts_.reserve(cpu_->pools.size());
        for (int cohort_index = 0;
             cohort_index < static_cast<int>(cpu_->pools.size());
             ++cohort_index) {
            const auto & pool =
                cpu_->pools.at(static_cast<size_t>(cohort_index));
            MoeCachedCohort cohort;
            cohort.index = cohort_index;
            cohort.cpu = &pool;
            cohort.arena = cache_->register_cohort(
                pool, cpu_->out_per_expert, cpu_->neuron_len,
                minimum_slots);
            cohort.cpu_fields = moe_cache_fields(pool);
            cohort.expert_to_local.assign(
                static_cast<size_t>(cpu_->n_experts), -1);
            cohort.host_map.assign(
                static_cast<size_t>(cpu_->n_experts), -1);
            const auto * local =
                pool.expert_local.data_ptr<int32_t>();
            for (int expert = 0; expert < cpu_->n_experts; ++expert) {
                const int local_index = local[expert];
                cohort.expert_to_local[
                    static_cast<size_t>(expert)] = local_index;
                if (local_index < 0) continue;
                if (expert_to_cohort_[
                        static_cast<size_t>(expert)] >= 0) {
                    throw std::runtime_error(
                        "MoE cache source has duplicate expert ownership");
                }
                expert_to_cohort_[
                    static_cast<size_t>(expert)] = cohort_index;
                expert_to_local_[
                    static_cast<size_t>(expert)] = local_index;
            }
            for (const auto & field : cohort.cpu_fields) {
                const int64_t nbytes = tensor_nbytes(field);
                if (nbytes % pool.local_experts != 0) {
                    throw std::runtime_error(
                        "MoE cache field byte count is not expert aligned");
                }
                cohort.bytes_per_expert.push_back(
                    nbytes / pool.local_experts);
            }
            cohorts_.push_back(std::move(cohort));
        }
        if (std::any_of(
                expert_to_cohort_.begin(),
                expert_to_cohort_.end(),
                [](int value) { return value < 0; })) {
            throw std::runtime_error(
                "MoE cache source does not cover every expert");
        }
        const char * disable_hetero =
            std::getenv("MFQ_DISABLE_MOE_HETERO");
        pure_nint_candidate_ =
            (disable_hetero == nullptr ||
             std::atoi(disable_hetero) == 0) &&
            std::all_of(
                cpu_->pools.begin(),
                cpu_->pools.end(),
                [](const MixedMoePool & pool) {
                    return pool.family == MixedMoeFamily::Nint &&
                        nint_moe_profile_code(
                            static_cast<int>(pool.nint.bits),
                            static_cast<int>(pool.nint.gs)) >= 0;
                });
        if (pure_nint_candidate_) {
            hetero_host_map_.assign(
                static_cast<size_t>(cpu_->n_experts), -1);
        }
    }

    int id() const noexcept {
        return id_;
    }

    const std::string & name() const noexcept {
        return name_;
    }

    int layer_id() const noexcept {
        return layer_id_;
    }

    const std::string & projection_role() const noexcept {
        return projection_role_;
    }

    int n_experts() const noexcept {
        return cpu_->n_experts;
    }

    MoeGpuArena * arena_for_expert(int expert) const {
        if (expert < 0 || expert >= cpu_->n_experts) {
            throw std::out_of_range(
                "MoE cache profile expert is out of range");
        }
        const int cohort = expert_to_cohort_.at(
            static_cast<size_t>(expert));
        return cohorts_.at(static_cast<size_t>(cohort)).arena;
    }

    int64_t bytes_for_expert(int expert) const {
        return arena_for_expert(expert)->slot_bytes;
    }

    void touch_expert(int expert) {
        const int cohort = expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & arena = *cohorts_.at(
            static_cast<size_t>(cohort)).arena;
        const int slot = arena.book->slot_for(
            {id_, cohort, expert});
        if (slot < 0) {
            throw std::runtime_error(
                "prewarmed MoE expert is absent from its arena");
        }
        arena.book->touch(slot);
    }

    int64_t host_bytes() const {
        return mixed_moe_storage_bytes(*cpu_);
    }

    void finalize() {
        if (active_) return;
        active_ = std::make_shared<MixedMoeRuntime>();
        active_->n_experts = cpu_->n_experts;
        active_->out_per_expert = cpu_->out_per_expert;
        active_->neuron_len = cpu_->neuron_len;
        active_->pools.reserve(cohorts_.size());
        for (auto & cohort : cohorts_) {
            auto & source = *cohort.cpu;
            auto & arena = *cohort.arena;
            MixedMoePool pool = source;
            pool.local_experts = arena.slots;
            pool.expert_local = torch::full(
                {cpu_->n_experts}, -1,
                torch::TensorOptions()
                    .device(torch::kCUDA)
                    .dtype(torch::kInt32));
            size_t field = 0;
            if (pool.family == MixedMoeFamily::Nint) {
                pool.nint.workspaces.clear();
                pool.nint.q_packed = arena.fields.at(field++);
                pool.nint.sub_scale = arena.fields.at(field++);
                pool.nint.sub_min = arena.fields.at(field++);
                pool.nint.neuron_scale = arena.fields.at(field++);
                pool.nint.neuron_min = arena.fields.at(field++);
                pool.nint.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.nint.shape.empty()) {
                    pool.nint.shape[0] = pool.nint.out;
                }
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                pool.q8_zero.workspaces.clear();
                pool.q8_zero.q_packed = arena.fields.at(field++);
                pool.q8_zero.q8_zero_scale = arena.fields.at(field++);
                pool.q8_zero.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.q8_zero.shape.empty()) {
                    pool.q8_zero.shape[0] = pool.q8_zero.out;
                }
            } else if (pool.family == MixedMoeFamily::Nvq) {
                pool.nvq.workspaces.clear();
                pool.nvq.indices_packed = arena.fields.at(field++);
                pool.nvq.aux_packed = arena.fields.at(field++);
                pool.nvq.sub_scale_packed = arena.fields.at(field++);
                pool.nvq.neuron_scale = arena.fields.at(field++);
                pool.nvq.codebook =
                    copy_cpu_weight_to_cuda(source.nvq.codebook);
                pool.nvq.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.nvq.shape.empty()) {
                    pool.nvq.shape[0] = pool.nvq.out;
                }
            } else {
                pool.nepq.indices_packed = arena.fields.at(field++);
                pool.nepq.aux_packed = arena.fields.at(field++);
                pool.nepq.state_packed = arena.fields.at(field++);
                pool.nepq.neuron_scale = arena.fields.at(field++);
                pool.nepq.bank_ids = arena.fields.at(field++);
                pool.nepq.table_pool =
                    copy_cpu_weight_to_cuda(source.nepq.table_pool);
                pool.nepq.grouped_table_pool =
                    copy_cpu_weight_to_cuda(
                        source.nepq.grouped_table_pool);
                pool.nepq.rotation_signs =
                    copy_cpu_weight_to_cuda(
                        source.nepq.rotation_signs);
                pool.nepq.n_experts = arena.slots;
            }
            cohort.active = std::move(pool);
            active_->pools.push_back(cohort.active);
        }
        if (pure_nint_candidate_) {
            NintMoeWeight value;
            value.n_experts = cpu_->n_experts;
            value.out_per_expert = cpu_->out_per_expert;
            value.neuron_len = cpu_->neuron_len;
            value.pools.reserve(active_->pools.size());
            for (const auto & active_pool : active_->pools) {
                if (active_pool.family != MixedMoeFamily::Nint) {
                    throw std::runtime_error(
                        "cached NINT heterogeneous dispatch received "
                        "a non-NINT cohort");
                }
                NintMoePoolWeight pool;
                pool.weight = active_pool.nint;
                pool.expert_local = active_pool.expert_local;
                pool.local_experts = active_pool.local_experts;
                value.pools.push_back(std::move(pool));
            }
            std::vector<int32_t> expert_pool(
                expert_to_cohort_.begin(),
                expert_to_cohort_.end());
            initialize_nint_moe_dispatch(
                value, expert_pool, hetero_host_map_);
            if (!value.hetero_supported) {
                throw std::runtime_error(
                    "cached NINT heterogeneous dispatch initialization failed");
            }
            pure_nint_ =
                std::make_shared<NintMoeWeight>(std::move(value));
        }
    }

    void invalidate(int cohort_index, int expert, int slot) {
        if (cohort_index < 0 ||
            cohort_index >= static_cast<int>(cohorts_.size()) ||
            expert < 0 || expert >= cpu_->n_experts) {
            throw std::runtime_error(
                "invalid MoE cache eviction key");
        }
        auto & cohort =
            cohorts_.at(static_cast<size_t>(cohort_index));
        if (cohort.host_map[static_cast<size_t>(expert)] == slot) {
            cohort.host_map[static_cast<size_t>(expert)] = -1;
            cohort.map_dirty = true;
        }
        if (pure_nint_candidate_ &&
                hetero_host_map_.at(
                    static_cast<size_t>(expert)) == slot) {
            hetero_host_map_.at(
                static_cast<size_t>(expert)) = -1;
            hetero_map_dirty_ = true;
        }
    }

    std::vector<int32_t> route_experts(
            const MoeRoutePlan & route) const {
        if (!route.host_unique_experts) {
            route.host_unique_experts =
                std::make_shared<std::vector<int32_t>>(
                    cache_->read_route_experts(
                        route.ids, cpu_->n_experts));
        }
        return *route.host_unique_experts;
    }

    bool use_full_projection(const MoeRoutePlan & route) const {
        return route.ids.size(0) > 8;
    }

    torch::Tensor forward(
            torch::Tensor x,
            const MoeRoutePlan & route) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = pure_nint_candidate_
                ? stage_cpu_nint_moe(cpu_)
                : stage_cpu_mixed_moe(cpu_);
            return staged.forward(x, route);
        }
        cache_->prepare(*this, route_experts(route), false);
        torch::Tensor output;
        if (pure_nint_) {
            cache_->count_hetero_dispatch();
            output = pure_nint_->forward(x, route);
        } else {
            output = active_->forward(x, route);
        }
        cache_->record_compute_use();
        return output;
    }

    void prefetch(const MoeRoutePlan & route) {
        if (use_full_projection(route)) return;
        cache_->prepare(
            *this, route_experts(route), true);
    }

    torch::Tensor forward_glu_output(
            torch::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        if (!pure_nint_) {
            throw std::runtime_error(
                "cached heterogeneous GLU output requires pure NINT cohorts");
        }
        cache_->prepare(*this, route_experts(route), false);
        cache_->count_hetero_dispatch();
        auto output =
            pure_nint_->forward_glu_output(x, route, gelu);
        cache_->record_compute_use();
        return output;
    }

    torch::Tensor forward_glu(
            torch::Tensor gate_up,
            const MoeRoutePlan & route,
            bool gelu) {
        if (!pure_nint_) {
            throw std::runtime_error(
                "cached heterogeneous GLU input requires pure NINT cohorts");
        }
        cache_->prepare(*this, route_experts(route), false);
        cache_->count_hetero_dispatch();
        auto output =
            pure_nint_->forward_glu(gate_up, route, gelu);
        cache_->record_compute_use();
        return output;
    }

    torch::Tensor forward_clamped_swiglu(
            torch::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_cpu_mixed_moe(cpu_);
            return staged.forward_clamped_swiglu(
                gate_up, route, limit);
        }
        cache_->prepare(*this, route_experts(route), false);
        auto output =
            active_->forward_clamped_swiglu(
                gate_up, route, limit);
        cache_->record_compute_use();
        return output;
    }

    bool supports_clamped_swiglu() const {
        return cpu_->supports_clamped_swiglu();
    }

    bool supports_nint_hetero() const noexcept {
        return pure_nint_candidate_;
    }

private:
    friend class MoeExpertCache;

    MoeExpertCache * cache_ = nullptr;
    int id_ = -1;
    std::string name_;
    int layer_id_ = -1;
    std::string projection_role_;
    std::shared_ptr<MixedMoeRuntime> cpu_;
    std::vector<MoeCachedCohort> cohorts_;
    std::vector<int> expert_to_cohort_;
    std::vector<int> expert_to_local_;
    std::shared_ptr<MixedMoeRuntime> active_;
    bool pure_nint_candidate_ = false;
    std::vector<int32_t> hetero_host_map_;
    bool hetero_map_dirty_ = false;
    std::shared_ptr<NintMoeWeight> pure_nint_;
};

std::shared_ptr<MoeCachedSource> MoeExpertCache::register_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> cpu,
        int minimum_slots,
        int layer_id,
        std::string projection_role) {
    const int id = static_cast<int>(sources_.size());
    auto source = std::make_shared<MoeCachedSource>(
        this, id, name, std::move(cpu), minimum_slots,
        layer_id, std::move(projection_role));
    host_bytes_ += source->host_bytes();
    sources_.push_back(source);
    return source;
}

void MoeExpertCache::finalize() {
    if (finalized_) return;
    if (sources_.empty()) {
        throw std::runtime_error(
            "MoE cache has no registered expert sources");
    }
    std::vector<mfq::MoeArenaDemand> demands;
    demands.reserve(arenas_.size());
    for (const auto & item : arenas_) {
        const auto & arena = *item.second;
        demands.push_back({
            arena.signature,
            arena.slot_bytes,
            arena.minimum_slots,
            arena.registered_experts,
        });
    }

    if (profile_.has_value()) {
        std::unordered_map<
            int, std::vector<std::shared_ptr<MoeCachedSource>>>
            layer_sources;
        std::unordered_map<int, int> layer_experts;
        std::unordered_map<int, std::unordered_set<std::string>>
            layer_roles;
        for (const auto & source : sources_) {
            if (source->layer_id() < 0) continue;
            if (source->projection_role().empty()) {
                throw std::runtime_error(
                    "profiled MoE cache source has no projection role");
            }
            if (!layer_roles[source->layer_id()]
                     .insert(source->projection_role()).second) {
                throw std::runtime_error(
                    "profiled MoE cache layer has a duplicate projection role");
            }
            const auto inserted = layer_experts.emplace(
                source->layer_id(), source->n_experts());
            if (!inserted.second &&
                    inserted.first->second != source->n_experts()) {
                throw std::runtime_error(
                    "profiled MoE cache layer sources disagree on expert count");
            }
            layer_sources[source->layer_id()].push_back(source);
        }
        mfq::validate_moe_cache_profile(
            *profile_, layer_experts);

        std::unordered_map<std::string, int> required_slots;
        std::unordered_map<std::string, int> selected_slots;
        int64_t required_bytes = 0;
        for (const auto & demand : demands) {
            required_slots[demand.signature] =
                demand.minimum_slots;
            required_bytes +=
                static_cast<int64_t>(demand.minimum_slots) *
                demand.slot_bytes;
        }
        if (required_bytes > budget_bytes_) {
            throw std::runtime_error(
                "MoE cache budget is below the minimum decode working set");
        }

        auto bundle_bytes = [&](int layer, int expert) {
            int64_t bytes = 0;
            for (const auto & source : layer_sources.at(layer)) {
                const int64_t source_bytes =
                    source->bytes_for_expert(expert);
                if (source_bytes >
                        std::numeric_limits<int64_t>::max() - bytes) {
                    throw std::overflow_error(
                        "MoE cache prewarm bundle byte count overflows int64");
                }
                bytes += source_bytes;
            }
            return bytes;
        };

        auto select_candidate = [&](
                const mfq::MoeProfileCandidate & candidate,
                int64_t layer_limit,
                std::unordered_map<int, int64_t> * layer_used) {
            const auto source_it =
                layer_sources.find(candidate.layer);
            if (source_it == layer_sources.end() ||
                    source_it->second.empty()) {
                return false;
            }
            std::unordered_map<std::string, int> needed;
            for (const auto & source : source_it->second) {
                ++needed[
                    source->arena_for_expert(
                        candidate.expert)->signature];
            }
            int64_t incremental_bytes = 0;
            for (const auto & item : needed) {
                const auto arena_it = arenas_.find(item.first);
                if (arena_it == arenas_.end()) {
                    throw std::runtime_error(
                        "MoE cache profile references an unknown arena");
                }
                const auto & arena = *arena_it->second;
                const int selected =
                    selected_slots[item.first] + item.second;
                if (selected > arena.registered_experts) {
                    return false;
                }
                const int new_required = std::max(
                    required_slots[item.first], selected);
                incremental_bytes +=
                    static_cast<int64_t>(
                        new_required -
                        required_slots[item.first]) *
                    arena.slot_bytes;
            }
            const int64_t bytes =
                bundle_bytes(candidate.layer, candidate.expert);
            if (layer_used != nullptr &&
                    bytes >
                        layer_limit -
                        (*layer_used)[candidate.layer]) {
                return false;
            }
            if (incremental_bytes >
                    budget_bytes_ - required_bytes) {
                return false;
            }
            for (const auto & item : needed) {
                selected_slots[item.first] += item.second;
                required_slots[item.first] = std::max(
                    required_slots[item.first],
                    selected_slots[item.first]);
            }
            required_bytes += incremental_bytes;
            if (layer_used != nullptr) {
                (*layer_used)[candidate.layer] += bytes;
            }
            prewarm_selected_.push_back(candidate);
            return true;
        };

        const auto candidates =
            mfq::order_moe_profile_candidates(
                *profile_, false);
        int64_t frequency_bytes = 0;
        std::unordered_set<int> ranking_layers;
        for (const auto & candidate : candidates) {
            if (candidate.has_frequency) {
                if (select_candidate(
                        candidate,
                        std::numeric_limits<int64_t>::max(),
                        nullptr)) {
                    frequency_bytes +=
                        bundle_bytes(
                            candidate.layer,
                            candidate.expert);
                }
            } else {
                ranking_layers.insert(candidate.layer);
            }
        }

        std::unordered_map<int, int64_t> rank_layer_total;
        int64_t rank_model_total = 0;
        for (int layer : ranking_layers) {
            int64_t total = 0;
            for (int expert = 0;
                 expert < layer_experts.at(layer);
                 ++expert) {
                total += bundle_bytes(layer, expert);
            }
            rank_layer_total[layer] = total;
            rank_model_total += total;
        }
        const int64_t rank_budget =
            std::max<int64_t>(
                0, budget_bytes_ - frequency_bytes);
        std::unordered_map<int, int64_t> rank_layer_limit;
        for (const auto & item : rank_layer_total) {
            const long double fraction =
                rank_model_total > 0
                ? static_cast<long double>(item.second) /
                    static_cast<long double>(rank_model_total)
                : 0.0L;
            rank_layer_limit[item.first] =
                static_cast<int64_t>(
                    static_cast<long double>(rank_budget) *
                    fraction);
        }
        std::unordered_map<int, int64_t> rank_layer_used;
        for (const auto & candidate : candidates) {
            if (candidate.has_frequency) continue;
            (void)select_candidate(
                candidate,
                rank_layer_limit.at(candidate.layer),
                &rank_layer_used);
        }

        for (auto & demand : demands) {
            demand.minimum_slots = std::max(
                demand.minimum_slots,
                required_slots.at(demand.signature));
        }
    }
    const auto plan =
        mfq::plan_moe_arena_slots(budget_bytes_, demands);
    for (auto & item : arenas_) {
        auto & arena = *item.second;
        arena.slots = plan.at(arena.signature);
        arena.fields.reserve(arena.prototypes.size());
        for (size_t index = 0;
             index < arena.prototypes.size();
             ++index) {
            const auto & prototype = arena.prototypes[index];
            if (prototype.dim() < 1 ||
                    prototype.size(0) %
                        arena.prototype_experts != 0) {
                throw std::runtime_error(
                    "MoE cache field has no expert-major leading dimension");
            }
            auto shape = prototype.sizes().vec();
            shape[0] =
                prototype.size(0) /
                arena.prototype_experts *
                arena.slots;
            arena.fields.push_back(torch::empty(
                shape,
                torch::TensorOptions()
                    .device(torch::kCUDA)
                    .dtype(prototype.scalar_type())));
        }
        arena.book =
            std::make_unique<mfq::MoeCacheSlotBook>(
                arena.slots);
        allocated_bytes_ +=
            static_cast<int64_t>(arena.slots) *
            arena.slot_bytes;
        std::cerr
            << "moe_cache_arena"
            << " signature=" << arena.signature
            << " slots=" << arena.slots
            << " slot_bytes=" << arena.slot_bytes
            << " bytes="
            << static_cast<int64_t>(arena.slots) *
                arena.slot_bytes
            << std::endl;
    }
    if (allocated_bytes_ > budget_bytes_) {
        throw std::runtime_error(
            "MoE cache arena allocation exceeded its budget");
    }
    for (auto & source : sources_) source->finalize();
    finalized_ = true;
    std::cerr
        << "moe_cache_ready"
        << " sources=" << sources_.size()
        << " host_bytes=" << host_bytes_
        << " budget_bytes=" << budget_bytes_
        << " allocated_bytes=" << allocated_bytes_
        << std::endl;
    if (profile_.has_value()) prewarm();
}

void MoeExpertCache::prewarm() {
    if (!finalized_ || !profile_.has_value()) return;

    std::unordered_map<
        int, std::vector<std::shared_ptr<MoeCachedSource>>>
        layer_sources;
    for (const auto & source : sources_) {
        if (source->layer_id() < 0) continue;
        layer_sources[source->layer_id()].push_back(source);
    }

    const auto started = std::chrono::steady_clock::now();
    std::unordered_map<
        MoeCachedSource *, std::vector<int32_t>>
        source_experts;
    for (auto selected_it = prewarm_selected_.rbegin();
         selected_it != prewarm_selected_.rend();
         ++selected_it) {
        for (const auto & source :
             layer_sources.at(selected_it->layer)) {
            source_experts[source.get()].push_back(
                selected_it->expert);
        }
    }
    for (const auto & source : sources_) {
        const auto found = source_experts.find(source.get());
        if (found == source_experts.end()) continue;
        prepare(*source, found->second, true);
    }
    if (transfer_ready_recorded_) {
        MFQ_CUDA_CHECK(
            cudaEventSynchronize(transfer_ready_));
    }
    for (auto selected_it = prewarm_selected_.rbegin();
         selected_it != prewarm_selected_.rend();
         ++selected_it) {
        for (const auto & source :
             layer_sources.at(selected_it->layer)) {
            source->touch_expert(
                selected_it->expert);
        }
    }
    const auto stopped = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            stopped - started).count();
    const int64_t prewarm_h2d_bytes = stats_.h2d_bytes;
    const int64_t projection_entries =
        stats_.prefetch_misses;
    const int64_t unfilled_bytes =
        std::max<int64_t>(
            0, allocated_bytes_ - prewarm_h2d_bytes);
    std::cout
        << "moe_cache_prewarm"
        << " profile=" << profile_->path
        << " expert_bundles=" << prewarm_selected_.size()
        << " projection_entries=" << projection_entries
        << " h2d_bytes=" << prewarm_h2d_bytes
        << " time_ms=" << elapsed_ms
        << " unfilled_bytes=" << unfilled_bytes
        << "\n";
    stats_ = {};
}

void MoeExpertCache::invalidate(
        const mfq::MoeCacheKey & key,
        int slot) {
    if (key.source < 0 ||
        key.source >= static_cast<int>(sources_.size())) {
        throw std::runtime_error(
            "MoE cache eviction references an invalid source");
    }
    sources_.at(static_cast<size_t>(key.source))
        ->invalidate(key.cohort, key.expert, slot);
}

void MoeExpertCache::prepare(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch) {
    if (!finalized_) {
        throw std::runtime_error(
            "MoE cache must be finalized before inference");
    }
    std::vector<MoeCacheTransfer> transfers;
    bool replaced_occupied = false;
    for (int expert : experts) {
        const int cohort_index =
            source.expert_to_cohort_.at(
                static_cast<size_t>(expert));
        const int local_index =
            source.expert_to_local_.at(
                static_cast<size_t>(expert));
        auto & cohort =
            source.cohorts_.at(
                static_cast<size_t>(cohort_index));
        auto & arena = *cohort.arena;
        const mfq::MoeCacheKey key{
            source.id_, cohort_index, expert};
        auto lease = arena.book->acquire(key);
        if (lease.hit) {
            if (prefetch) {
                ++stats_.prefetch_hits;
            } else {
                ++stats_.demand_hits;
            }
        } else {
            if (prefetch) {
                ++stats_.prefetch_misses;
            } else {
                ++stats_.demand_misses;
            }
            if (lease.replaced.has_value()) {
                ++stats_.evictions;
                replaced_occupied = true;
                invalidate(*lease.replaced, lease.slot);
            }
            for (size_t field = 0;
                 field < cohort.cpu_fields.size();
                 ++field) {
                const int64_t nbytes =
                    cohort.bytes_per_expert[field];
                if (nbytes == 0) continue;
                const auto & cpu_field =
                    cohort.cpu_fields[field];
                auto & gpu_field =
                    arena.fields[field];
                transfers.push_back({
                    reinterpret_cast<const uint8_t *>(
                        cpu_field.data_ptr()) +
                        static_cast<int64_t>(local_index) *
                        nbytes,
                    reinterpret_cast<uint8_t *>(
                        gpu_field.data_ptr()) +
                        static_cast<int64_t>(lease.slot) *
                        nbytes,
                    nbytes,
                    true,
                });
            }
        }
        if (cohort.host_map[
                static_cast<size_t>(expert)] != lease.slot) {
            cohort.host_map[
                static_cast<size_t>(expert)] = lease.slot;
            cohort.map_dirty = true;
        }
        if (source.pure_nint_candidate_ &&
                source.hetero_host_map_.at(
                    static_cast<size_t>(expert)) != lease.slot) {
            source.hetero_host_map_.at(
                static_cast<size_t>(expert)) = lease.slot;
            source.hetero_map_dirty_ = true;
        }
    }

    for (auto & cohort : source.cohorts_) {
        if (!cohort.map_dirty) continue;
        auto & active =
            source.active_->pools.at(
                static_cast<size_t>(cohort.index));
        transfers.push_back({
            reinterpret_cast<const uint8_t *>(
                cohort.host_map.data()),
            reinterpret_cast<uint8_t *>(
                active.expert_local.data_ptr<int32_t>()),
            static_cast<int64_t>(
                cohort.host_map.size() *
                sizeof(int32_t)),
            false,
        });
        cohort.map_dirty = false;
    }
    if (source.hetero_map_dirty_) {
        if (!source.pure_nint_ ||
                !source.pure_nint_->expert_local.defined()) {
            throw std::runtime_error(
                "cached NINT heterogeneous map is unavailable");
        }
        transfers.push_back({
            reinterpret_cast<const uint8_t *>(
                source.hetero_host_map_.data()),
            reinterpret_cast<uint8_t *>(
                source.pure_nint_->expert_local.data_ptr<int32_t>()),
            static_cast<int64_t>(
                source.hetero_host_map_.size() *
                sizeof(int32_t)),
            false,
        });
        source.hetero_map_dirty_ = false;
    }
    submit_transfers(
        transfers,
        replaced_occupied,
        !prefetch);
}

static NintMoeWeight wrap_cached_moe_source(
        const std::shared_ptr<MoeCachedSource> & source,
        const std::shared_ptr<MixedMoeRuntime> & cpu) {
    NintMoeWeight result =
        cpu_mixed_moe_metadata(cpu);
    result.hetero_supported =
        source->supports_nint_hetero();
    result.cache_prefetch = [source](
            const MoeRoutePlan & route) {
        source->prefetch(route);
    };
    result.mixed_forward = [source](
            torch::Tensor x,
            const MoeRoutePlan & route) {
        return source->forward(x, route);
    };
    if (source->supports_nint_hetero()) {
        result.mixed_glu_output_forward = [source](
                torch::Tensor x,
                const MoeRoutePlan & route,
                bool gelu) {
            return source->forward_glu_output(
                x, route, gelu);
        };
        result.mixed_glu_forward = [source](
                torch::Tensor gate_up,
                const MoeRoutePlan & route,
                bool gelu) {
            return source->forward_glu(
                gate_up, route, gelu);
        };
    }
    if (source->supports_clamped_swiglu()) {
        result.mixed_clamped_swiglu_forward = [source](
                torch::Tensor gate_up,
                const MoeRoutePlan & route,
                double limit) {
            return source->forward_clamped_swiglu(
                gate_up, route, limit);
        };
    }
    return result;
}

static NintMoeWeight load_nint_moe_gpu(
        const MfqFile & mfq, const std::string & name,
        bool cacheable,
        int layer_id,
        const std::string & projection_role) {
    auto cpu = load_nint_moe_cpu(mfq, name);
    if (g_tensor_parallel.enabled()) {
        const bool paired =
            projection_role == "gate_up";
        if (paired &&
            cpu.out_per_expert % 2 != 0) {
            throw std::runtime_error(
                "tensor-parallel gate/up MoE "
                "width must be even: " + name);
        }
        auto slices = plan_moe_tensor_parallel_slices(
            cpu.n_experts, name);
        NintMoeWeight result;
        result.n_experts = cpu.n_experts;
        result.out_per_expert =
            cpu.out_per_expert;
        result.neuron_len =
            cpu.neuron_len;
        result.tensor_parallel_paired_output =
            paired;
        result.tensor_parallel_experts = true;
        result.hetero_supported = true;
        for (const auto & slice : slices) {
            auto shard =
                std::make_shared<NintMoeWeight>(
                    to_cuda_device_moe_expert_slice(
                        cpu, slice.begin,
                        slice.end,
                        slice.device));
            result.hetero_supported =
                result.hetero_supported &&
                shard->hetero_supported;
            result.tensor_parallel_shards.push_back({
                slice.device,
                slice.begin,
                slice.end,
                std::move(shard),
            });
        }
        return result;
    }
    if (g_moe_expert_cache && cacheable) {
        auto runtime =
            make_mixed_moe_runtime(cpu, false);
        auto source = g_moe_expert_cache->register_source(
            name, runtime,
            std::min(
                g_moe_cache_registration_min_slots,
                runtime->n_experts),
            layer_id,
            projection_role);
        return wrap_cached_moe_source(source, runtime);
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(), [](const NintMoeCpuPool & pool) {
            return pool.dtype != "NINTM" && pool.dtype != "NINT8-0" &&
                   pool.dtype.rfind("NINT", 0) == 0;
        });
    return all_nint ? to_gpu_nint_moe(cpu) : to_gpu_mixed_moe(cpu);
}

static std::shared_ptr<MixedMoeRuntime> load_nint_moe_cpu_offloaded(
        const MfqFile & mfq, const std::string & name) {
    return make_mixed_moe_runtime(load_nint_moe_cpu(mfq, name), false);
}

static void enable_nint5_q5_exec(NintWeight & w) {
    if (w.bits != 5 || w.gs != 28 || w.qbytes != 18 || w.q5_exec) {
        throw std::runtime_error("NINT5 Q5 execution layout requires NINT5 gs28 source weights");
    }
    auto q_exec = nint5_gs28_q5_repack_cuda(w.q_packed, w.sub_scale, w.sub_min);
    torch::cuda::synchronize();
    w.q_packed = std::move(q_exec);
    w.sub_scale = torch::Tensor();
    w.sub_min = torch::Tensor();
    w.qbytes = 20;
    w.q5_exec = true;
    c10::cuda::CUDACachingAllocator::emptyCache();
}

static bool nint5_q5_exec_enabled() {
    const char * disable = std::getenv("MFQ_DISABLE_NINT5_Q5_EXEC");
    return disable == nullptr || disable[0] != '1';
}

static NvqWeight load_nvq_gpu(const MfqFile & mfq, const std::string & name) {
    const auto & rec = mfq.record(name);
    return to_gpu_nvq(unpack_nvq(mfq.read_blob(name), rec.dtype));
}

static torch::Tensor nvq_dequant(const NvqWeight & w) {
    return nvq_dequant_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}

static torch::Tensor load_dense_gpu(const MfqFile & mfq, const std::string & name) {
    c10::cuda::CUDAGuard guard(
        active_weight_load_device());
    const auto & rec = mfq.record(name);
    auto blob = mfq.read_blob(name);
    if (rec.dtype == "NINT8-0") {
        const auto packed = to_gpu_nint8_zero(
            unpack_nint8_zero(blob));
        auto dense = nint8_zero_dequant_cuda(
            packed.q_packed,
            packed.q8_zero_scale,
            packed.neuron_len)
            .to(torch::kFloat32)
            .contiguous();
        if (packed.shape.size() != 2 ||
            dense.size(0) != packed.shape[0] ||
            dense.size(1) != packed.shape[1]) {
            throw std::runtime_error(
                "NINT8-0 dense tensor shape mismatch: " + name);
        }
        return dense;
    }
    size_t off = 0;
    uint32_t ndim = read_u32_from(blob, off);
    std::vector<int64_t> shape(ndim);
    int64_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        shape[i] = read_i64_from(blob, off);
        numel *= shape[i];
    }
    torch::Tensor t;
    if (rec.dtype == "F32") {
        t = torch::from_blob(blob.data() + off, shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    } else if (rec.dtype == "BF16") {
        t = torch::from_blob(blob.data() + off, shape, torch::TensorOptions().dtype(torch::kBFloat16)).clone().to(torch::kFloat32);
    } else if (rec.dtype == "F16") {
        t = torch::from_blob(blob.data() + off, shape, torch::TensorOptions().dtype(torch::kFloat16)).clone().to(torch::kFloat32);
    } else if (rec.dtype == "I64") {
        t = torch::from_blob(blob.data() + off, shape, torch::TensorOptions().dtype(torch::kInt64)).clone();
    } else if (rec.dtype == "I32") {
        t = torch::from_blob(blob.data() + off, shape, torch::TensorOptions().dtype(torch::kInt32)).clone();
    } else {
        throw std::runtime_error("unsupported dense dtype for C++ runtime: " + rec.dtype + " tensor " + name);
    }
    (void)numel;
    return t.to(torch::kCUDA).contiguous();
}

static torch::Tensor load_dense_linear_cpu(
        const MfqFile & mfq,
        const std::string & name) {
    const auto & rec = mfq.record(name);
    auto blob = mfq.read_blob(name);
    size_t off = 0;
    const uint32_t ndim = read_u32_from(blob, off);
    std::vector<int64_t> shape(ndim);
    for (uint32_t index = 0; index < ndim; ++index) {
        shape[index] = read_i64_from(blob, off);
    }
    if (shape.size() != 2) {
        throw std::runtime_error(
            "dense linear tensor must be rank 2: " + name);
    }
    torch::Tensor value;
    if (rec.dtype == "BF16") {
        value = torch::from_blob(
            blob.data() + off, shape,
            torch::TensorOptions().dtype(torch::kBFloat16)).clone();
    } else if (rec.dtype == "F16") {
        value = torch::from_blob(
            blob.data() + off, shape,
            torch::TensorOptions().dtype(torch::kFloat16)).clone();
    } else if (rec.dtype == "F32") {
        value = torch::from_blob(
            blob.data() + off, shape,
            torch::TensorOptions().dtype(torch::kFloat32)).clone();
    } else {
        throw std::runtime_error(
            "unsupported dense linear dtype: " + rec.dtype +
            " tensor " + name);
    }
    return value.contiguous();
}

static torch::Tensor load_dense_linear_gpu(
        const MfqFile & mfq,
        const std::string & name) {
    c10::cuda::CUDAGuard guard(active_weight_load_device());
    return load_dense_linear_cpu(mfq, name)
        .to(torch::kCUDA).contiguous();
}

static NintWeight cat_weights(const std::vector<NintWeight> & ws) {
    if (ws.empty()) throw std::runtime_error("empty NINT group");
    const auto & a = ws[0];
    if (a.q5_exec) throw std::runtime_error("NINT5 Q5 execution weights cannot be concatenated");
    std::vector<torch::Tensor> qp, q8s, ss, sm, ns, nm;
    int64_t out = 0;
    for (const auto & w : ws) {
        if (w.q5_exec) throw std::runtime_error("NINT5 Q5 execution weights cannot be concatenated");
        if (w.ng != a.ng || w.gs != a.gs || w.bits != a.bits ||
            w.qbytes != a.qbytes || w.neuron_len != a.neuron_len ||
            w.q8_zero != a.q8_zero) {
            throw std::runtime_error("cannot group NINT tensors with different input layout");
        }
        qp.push_back(w.q_packed);
        if (w.q8_zero) {
            q8s.push_back(w.q8_zero_scale);
            out += w.out;
            continue;
        }
        ss.push_back(w.sub_scale);
        sm.push_back(w.sub_min);
        ns.push_back(w.neuron_scale);
        nm.push_back(w.neuron_min);
        out += w.out;
    }
    NintWeight g;
    g.out = out;
    g.ng = a.ng;
    g.gs = a.gs;
    g.bits = a.bits;
    g.qbytes = a.qbytes;
    g.neuron_len = a.neuron_len;
    g.q8_zero = a.q8_zero;
    g.shape = a.shape;
    g.shape[0] = out;
    g.q_packed = torch::cat(qp, 0).contiguous();
    if (a.q8_zero) {
        g.q8_zero_scale = torch::cat(q8s, 0).contiguous();
        return g;
    }
    g.sub_scale = torch::cat(ss, 0).contiguous();
    g.sub_min = torch::cat(sm, 0).contiguous();
    g.neuron_scale = torch::cat(ns, 0).contiguous();
    g.neuron_min = torch::cat(nm, 0).contiguous();
    return g;
}

static torch::Tensor pad_last(torch::Tensor x, int64_t target) {
    if (x.size(1) == target) return x;
    if (x.size(1) > target) throw std::runtime_error("activation width exceeds neuron_len");
    return torch::constant_pad_nd(x, {0, target - x.size(1)}, 0);
}

enum class NvqMatmulPath {
    Gemv,
    Mmq,
    OnlineF16,
    DequantGemm,
};

static NvqMatmulPath select_nvq_matmul_path(const NvqWeight & w, int M) {
    const bool e8_family =
        w.format == 2 || w.format == 5 || w.format == 7 ||
        w.format == 8 || w.format == 9 ||
        w.format == 13 || w.format == 14;

    if (w.format == 8 && M >= 14 && M <= 16 && w.neuron_len >= 2 * w.out) {
        return NvqMatmulPath::DequantGemm;
    }

    if (w.format == 9 && M >= 13) {
        const bool wide_output = w.out * 8 >= w.neuron_len * 21;
        const bool wide_mmq =
            w.out >= 4096 && (w.out >= 2 * w.neuron_len || w.neuron_len >= 2 * w.out);
        if (M <= 15) {
            if (wide_mmq) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M == 16) {
            if (w.out >= 4096) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M <= 31) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            if (w.out >= 4096 && w.out >= w.neuron_len) return NvqMatmulPath::Mmq;
            return NvqMatmulPath::DequantGemm;
        }
        if (M == 32) {
            return w.out >= 4096 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 47) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 48) {
            return wide_output ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 63) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 64) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            return w.out >= 8192 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        return NvqMatmulPath::DequantGemm;
    }

    if (M <= 13) return NvqMatmulPath::Gemv;
    if (M == 14) {
        return w.out >= 2048 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
    }
    if (M == 15) {
        if (e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        return NvqMatmulPath::DequantGemm;
    }
    if (M == 16) {
        if (e8_family && w.out >= 6144) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        if ((w.format == 3 || w.format == 10 ||
             w.format == 12 || w.format == 15) &&
            w.out >= 4096 && w.neuron_len >= 8192) {
            return NvqMatmulPath::Mmq;
        }
        return NvqMatmulPath::DequantGemm;
    }

    const bool wide_expansion = e8_family && w.out >= 3 * w.neuron_len;
    if (wide_expansion && ((M >= 17 && M <= 31) || (M >= 33 && M <= 47))) {
        return NvqMatmulPath::OnlineF16;
    }
    if (M == 32 && e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
    if (M == 48 && e8_family && w.out >= 12288) return NvqMatmulPath::Mmq;
    if (M == 64 && (w.format == 7 || w.format == 9) && w.out >= 8192) {
        return NvqMatmulPath::Mmq;
    }
    return NvqMatmulPath::DequantGemm;
}

static torch::Tensor nvq_matmul(const NvqWeight & w, torch::Tensor x) {
    x = x.contiguous().to(torch::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    if (g_kl_mmq_mode != KlMmqMode::Default) {
        TORCH_CHECK(
            M >= 16,
            "KLD common NVQ MMQ requires at least 16 activation rows");
        x = kl_mmq_prepare_activation(x);
        ++g_kl_mmq_dense_calls;
        return g_profiler.measure("kld_mmq.nvq.fp16", [&]() {
            return nvq_gemm_f16_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode);
        });
    }
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.gemv", [&]() {
            return nvq_gemv_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.mma24", [&]() {
            return nvq_mmq_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::OnlineF16) {
        return g_profiler.measure("nvq.gemm_online_f16", [&]() {
            return nvq_gemm_f16_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode);
        });
    }
    auto weight = g_profiler.measure("nvq.dequant", [&]() { return nvq_dequant(w); });
    return g_profiler.measure("nvq.gemm", [&]() {
        return nint_cublas_gemm_nt_f32acc_cuda(x, weight);
    });
}

static torch::Tensor nvq_matmul_input_mul(
    const NvqWeight & w,
    torch::Tensor x,
    torch::Tensor gate,
    int mode) {
    TORCH_CHECK(mode == 1 || mode == 2, "NVQ input gate mode must be 1(sigmoid) or 2(silu)");
    x = x.contiguous().to(torch::kFloat16);
    gate = gate.contiguous().to(torch::kFloat16);
    TORCH_CHECK(x.sizes() == gate.sizes(), "NVQ x and gate shapes must match");
    x = pad_last(x, w.neuron_len);
    gate = pad_last(gate, w.neuron_len);
    int M = (int)x.size(0);
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.gemv_gate", [&]() {
            return nvq_gemv_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.mma24_gate", [&]() {
            return nvq_mmq_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    torch::Tensor value = mode == 1 ? x * torch::sigmoid(gate) : x * torch::silu(gate);
    return nvq_matmul(w, value);
}

static bool nvq_pair_compatible(const NvqWeight & first, const NvqWeight & second) {
    return first.format == second.format && first.kernel_format == second.kernel_format &&
           first.gs == second.gs &&
           first.neuron_len == second.neuron_len && first.ng == second.ng;
}

static bool nvq_fusion_enabled() {
    const char * disable = std::getenv("MFQ_DISABLE_NVQ_FUSION");
    if (disable == nullptr) disable = std::getenv("MFQ_DISABLE_NIQ_FUSION");
    return disable == nullptr || disable[0] != '1';
}

static torch::Tensor nvq_matmul_multi2(
    const NvqWeight & first,
    const NvqWeight & second,
    torch::Tensor x) {
    if (!nvq_pair_compatible(first, second)) {
        throw std::runtime_error("NVQ multi-projection requires compatible formats and input layouts");
    }
    x = pad_last(x.contiguous().to(torch::kFloat16), first.neuron_len);
    const int M = (int)x.size(0);
    if (M > 8) return torch::cat({nvq_matmul(first, x), nvq_matmul(second, x)}, -1);
    NvqWorkspace & ws = first.workspace(M);
    return g_profiler.measure("nvq.gemv_multi2", [&]() {
        return nvq_gemv_multi2_ws_cuda(
            first.indices_packed, first.aux_packed, first.sub_scale_packed,
            first.neuron_scale, first.codebook,
            second.indices_packed, second.aux_packed, second.sub_scale_packed,
            second.neuron_scale, second.codebook,
            x, first.neuron_len, first.gs,
            first.sub_bits, first.kernel_format, first.sign_mode,
            second.sub_bits, second.kernel_format, second.sign_mode,
            ws.qx, ws.xscale);
    });
}

static torch::Tensor nvq_matmul_swiglu(
    const NvqWeight & gate,
    const NvqWeight & up,
    torch::Tensor x) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out) {
        throw std::runtime_error("NVQ SwiGLU requires compatible equal-width gate/up weights");
    }
    x = pad_last(x.contiguous().to(torch::kFloat16), gate.neuron_len);
    if (x.size(0) != 1) {
        auto pair = nvq_matmul_multi2(gate, up, x);
        auto parts = pair.split_with_sizes({gate.out, up.out}, -1);
        return torch::silu(parts[0]) * parts[1];
    }
    NvqWorkspace & ws = gate.workspace(1);
    return g_profiler.measure("nvq.gemv_swiglu", [&]() {
        return nvq_gemv_swiglu_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode,
            ws.qx, ws.xscale);
    });
}

static torch::Tensor nvq_ffn_swiglu_down(
    const NvqWeight & gate,
    const NvqWeight & up,
    const NvqWeight & down,
    torch::Tensor x) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out || gate.out != down.neuron_len) {
        throw std::runtime_error("NVQ fused FFN weight layouts are incompatible");
    }
    x = pad_last(x.contiguous().to(torch::kFloat16), gate.neuron_len);
    if (x.size(0) != 1 || (down.gs != 24 && down.gs != 28 && down.gs != 32)) {
        return nvq_matmul(down, nvq_matmul_swiglu(gate, up, x));
    }
    NvqWorkspace & input_ws = gate.workspace(1);
    NvqWorkspace & output_ws = down.workspace(1);
    if (!input_ws.swiglu_scratch.defined() || input_ws.swiglu_scratch.numel() < gate.out) {
        input_ws.swiglu_scratch = torch::empty(
            {gate.out}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    }
    g_profiler.measure("nvq.ffn_swiglu_quant", [&]() {
        nvq_ffn_swiglu_quant_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode, down.gs,
            input_ws.qx, input_ws.xscale,
            output_ws.qx, output_ws.xscale, input_ws.swiglu_scratch);
        return 0;
    });
    return g_profiler.measure("nvq.gemv_qx", [&]() {
        return nvq_gemv_qx_ws_cuda(
            down.indices_packed, down.aux_packed, down.sub_scale_packed,
            down.neuron_scale, down.codebook,
            down.neuron_len, down.gs, down.sub_bits, down.kernel_format, down.sign_mode,
            output_ws.qx, output_ws.xscale);
    });
}

static torch::Tensor nvq_embedding(const NvqWeight & w, torch::Tensor token_ids) {
    return nvq_embedding_lookup_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, token_ids, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}

static bool nint_gs24_group32_use_m(int M) {
    return M == 16 || M == 32;
}

static bool nint2_group32_use_m(int64_t out, int M) {
    if (out < 2048 || M < 9) return false;
    return M <= (out < 8192 ? 64 : 128);
}

static int nint_gs24_group32_split_k(int64_t out) {
    return out <= 8192 ? 2 : 1;
}

static int nint4_batch_gemv_max_m() {
    const char * env = std::getenv("MFQ_NINT4_BATCH_GEMV_MAX_M");
    if (env == nullptr) return 16;
    return std::max(1, std::atoi(env));
}

static int nint_hi_gemv_max_m() {
    const char * env = std::getenv("MFQ_NINT_HI_GEMV_MAX_M");
    if (env == nullptr) return 8;
    return std::max(1, std::atoi(env));
}

static bool nint6_batch_gemv() {
    const char * env = std::getenv("MFQ_NINT6_BATCH_GEMV");
    return env != nullptr && env[0] == '1';
}

enum class Nint3PrefillPath {
    Group32,
    F16Packed,
    DequantCublas,
};

static Nint3PrefillPath nint3_prefill_path() {
    const char * env = std::getenv("MFQ_NINT3_PREFILL_PATH");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "group32") == 0) {
        return Nint3PrefillPath::Group32;
    }
    if (std::strcmp(env, "f16") == 0) {
        return Nint3PrefillPath::F16Packed;
    }
    if (std::strcmp(env, "dequant") == 0) {
        return Nint3PrefillPath::DequantCublas;
    }
    throw std::runtime_error(
        "MFQ_NINT3_PREFILL_PATH must be group32, f16, or dequant");
}

static torch::Tensor nint_matmul(const NintWeight & w, torch::Tensor x) {
    x = x.contiguous().to(torch::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    if (g_kl_mmq_mode != KlMmqMode::Default) {
        const int original_m = M;
        TORCH_CHECK(original_m > 0, "KLD common MMQ requires activation rows");
        if (M < 16) {
            auto padding = torch::zeros(
                {16 - M, x.size(1)}, x.options());
            x = torch::cat({x, padding}, 0).contiguous();
            M = 16;
        }
        TORCH_CHECK(
            !w.q5_exec,
            "KLD common MMQ requires original NINT5 packed weights");
        x = kl_mmq_prepare_activation(x);
        ++g_kl_mmq_dense_calls;
        torch::Tensor result;
        if (w.q8_zero) {
            result = g_profiler.measure("kld_mmq.nint8_zero.fp16", [&]() {
                return nint8_zero_mmq_f16_packed_cuda(
                    w.q_packed, w.q8_zero_scale, x, w.neuron_len);
            });
        } else {
            result = g_profiler.measure("kld_mmq.nint.fp16", [&]() {
                return nint_mmq_f16_packed_cuda(
                    w.q_packed, w.sub_scale, w.sub_min,
                    w.neuron_scale, w.neuron_min, x, w.gs, w.bits);
            });
        }
        return original_m < 16
            ? result.index({Slice(0, original_m)}).contiguous()
            : result;
    }
    if (w.q8_zero) {
        Workspace & ws = w.workspace(M);
        if (M <= 8) {
            return g_profiler.measure("nint8_zero.gemv", [&]() {
                return nint8_zero_gemv_ws_cuda(
                    w.q_packed, w.q8_zero_scale, x, ws.qx, ws.xscale);
            });
        }
        if (M <= 64) {
            return g_profiler.measure("nint8_zero.mmq", [&]() {
                return nint8_zero_mmq_ws_cuda(
                    w.q_packed, w.q8_zero_scale, x, ws.qx, ws.xscale);
            });
        }
        auto ww = g_profiler.measure("nint8_zero.dequant", [&]() {
            return nint8_zero_dequant_cuda(
                w.q_packed, w.q8_zero_scale, w.neuron_len);
        });
        return g_profiler.measure("nint8_zero.gemm", [&]() {
            return torch::matmul(x, ww.transpose(0, 1));
        });
    }
    if (w.q5_exec) {
        if (M <= 8) {
            Workspace & ws = w.workspace(M);
            return g_profiler.measure("nint.q5_exec_gemv", [&]() {
                return nint5_gs28_q5_gemv_ws_cuda(
                    w.q_packed, w.neuron_scale, w.neuron_min,
                    x, ws.qx, ws.xscale, ws.xsum);
            });
        }
        auto ww = g_profiler.measure("nint.q5_exec_dequant", [&]() {
            return nint5_gs28_q5_dequant_cuda(
                w.q_packed, w.neuron_scale, w.neuron_min, w.neuron_len);
        });
        return g_profiler.measure("nint.q5_exec_gemm", [&]() {
            return torch::matmul(x, ww.transpose(0, 1));
        });
    }
    const Nint3PrefillPath nint3_path =
        w.bits == 3 ? nint3_prefill_path() : Nint3PrefillPath::Group32;
    if (w.out >= 1024 &&
            ((w.bits == 2 && w.gs == 16 && nint2_group32_use_m(w.out, M)) ||
             (w.bits == 3 && w.gs == 24 && M >= 9 &&
              nint3_path == Nint3PrefillPath::Group32) ||
             (w.bits == 4 && w.gs == 24 && nint_gs24_group32_use_m(M)) ||
             (w.bits == 6 && w.gs == 24 && M >= 9 &&
              g_nint6_mmq_mode == Nint6MmqMode::Int8))) {
        Workspace & ws =
            w.bits == 6 && M >= 128
            ? nint_prefill_mmq_workspace(w, x, M)
            : w.workspace(M);
        w.ensure_mmq_qx_workspace(ws, M);
        const int split_k =
            w.bits == 6 && M >= 128
            ? 1
            : nint_gs24_group32_split_k(w.out);
        if (split_k > 1) {
            w.ensure_mmq_partial_workspace(ws, (int64_t)split_k * M * w.out);
        }
        return g_profiler.measure("nint.mma_group32", [&]() {
            return nint_mmq_gs24_group32_ws_cuda(
                w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                x, ws.mmq_qx, ws.xscale, ws.xsum, split_k, ws.mmq_partial);
        });
    }
    if (w.gs == 24 && w.bits == 6 && w.out <= 8192 && M >= 16 && M <= 32) {
        constexpr int split_k = 4;
        Workspace & ws = w.workspace(M);
        w.ensure_mmq_partial_workspace(ws, (int64_t)split_k * M * w.out);
        return g_profiler.measure("nint.f16mmq24_nint6_split4", [&]() {
            return nint_mmq_gs24_f16_nint6_split4_ws_cuda(
                w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                x, ws.mmq_partial);
        });
    }
    if (w.gs == 24 && w.bits == 6 && M >= 16) {
        return g_profiler.measure("nint.f16mmq24_nint6", [&]() {
            return nint_mmq_f16_packed_cuda(
                w.q_packed, w.sub_scale, w.sub_min,
                w.neuron_scale, w.neuron_min, x, w.gs, w.bits);
        });
    }
    if (w.gs == 24 && w.bits == 4 && w.out >= 8192 && M >= 16 && M <= 32) {
        return g_profiler.measure("nint.f16mmq24_nint4", [&]() {
            return nint_mmq_gs24_f16_nint4_cuda(
                w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min, x);
        });
    }
    if (w.bits != 4) {
        if (w.bits == 3 && w.gs == 24 && M > 8 &&
                nint3_path != Nint3PrefillPath::DequantCublas) {
            return g_profiler.measure("nint.f16mmq24_nint3", [&]() {
                return nint_mmq_gs24_f16_nint3_cuda(
                    w.q_packed, w.sub_scale, w.sub_min,
                    w.neuron_scale, w.neuron_min, x);
            });
        }
        if (w.bits == 6 && w.gs == 24 && M <= 16) {
            Workspace & ws = w.workspace(M);
            return nint_gemv_packed_int6_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                                 x, w.gs, ws.qx, ws.xscale, ws.xsum);
        }
        if (M <= nint_hi_gemv_max_m()) {
            Workspace & ws = w.workspace(M);
            if (w.bits == 8) {
                return nint_gemv_packed_u8_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                                   x, w.gs, ws.qx, ws.xscale, ws.xsum);
            }
            if (w.bits == 6 && (w.gs % 4) == 0 && (M == 1 || !nint6_batch_gemv())) {
                return nint_gemv_packed_int6_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                                     x, w.gs, ws.qx, ws.xscale, ws.xsum);
            }
            return nint_gemv_packed_bits_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                                 x, w.gs, w.bits, ws.qx, ws.xscale, ws.xsum);
        }
        const char * nint8_prefill_mmq_env =
            std::getenv("MFQ_NINT8_PREFILL_MMQ");
        if (w.bits == 8 &&
                (M <= 64 ||
                 (nint8_prefill_mmq_env != nullptr &&
                  nint8_prefill_mmq_env[0] == '1'))) {
            Workspace & ws = w.workspace(M);
            return nint_mmq_packed_u8_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                              x, w.gs, ws.qx, ws.xscale, ws.xsum);
        }
        auto ww = g_profiler.measure("nint.dequant_hi", [&]() {
            return nint_dequant_full_packed_compact_bits_cuda(w.q_packed, w.sub_scale, w.sub_min,
                                                               w.neuron_scale, w.neuron_min,
                                                               w.neuron_len, w.gs, w.bits);
        });
        return g_profiler.measure("nint.gemm_hi", [&]() { return torch::matmul(x, ww.transpose(0, 1)); });
    }
    if (M == 1) {
        Workspace & ws = w.workspace(M);
        return nint_gemv_packed_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                        x, w.gs, ws.qx, ws.xscale, ws.xsum);
    }
    const int batch_gemv_max_m = nint4_batch_gemv_max_m();
    if (M <= batch_gemv_max_m && (M <= 8 || w.gs == 24)) {
        Workspace & ws = w.workspace(M);
        return nint_gemv_packed_batch_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                              x, w.gs, ws.qx, ws.xscale, ws.xsum);
    }
    auto ww = g_profiler.measure("nint.dequant4", [&]() {
        return nint_dequant_full_packed_compact_cuda(w.q_packed, w.sub_scale, w.sub_min,
                                                     w.neuron_scale, w.neuron_min,
                                                     w.neuron_len, w.gs);
    });
    return g_profiler.measure("nint.gemm4", [&]() { return nint_cublas_gemm_nt_f32acc_cuda(x, ww); });
}

static torch::Tensor nint_matmul_f32_kld(
        const NintWeight & w, torch::Tensor x) {
    TORCH_CHECK(
        g_kl_mmq_mode == KlMmqMode::Fp16,
        "FP32-output NINT MMQ is restricted to the FP16 KLD path");
    x = pad_last(
        x.contiguous().to(torch::kFloat16),
        w.neuron_len);
    TORCH_CHECK(
        x.size(0) >= 16,
        "FP32-output NINT MMQ requires at least 16 activation rows");
    TORCH_CHECK(
        !w.q5_exec,
        "FP32-output NINT MMQ requires original NINT5 packed weights");
    ++g_kl_mmq_dense_calls;
    if (w.q8_zero) {
        return g_profiler.measure(
            "kld_mmq.nint8_zero.fp32_output", [&]() {
                return nint8_zero_mmq_f32_packed_cuda(
                    w.q_packed, w.q8_zero_scale,
                    x, w.neuron_len);
            });
    }
    return g_profiler.measure(
        "kld_mmq.nint.fp32_output", [&]() {
            return nint_mmq_f32_packed_cuda(
                w.q_packed, w.sub_scale, w.sub_min,
                w.neuron_scale, w.neuron_min,
                x, w.gs, w.bits);
        });
}

static torch::Tensor nint_matmul_input_mul_f32_kld(
        const NintWeight & w,
        torch::Tensor x,
        torch::Tensor gate,
        int mode) {
    TORCH_CHECK(
        mode == 1 || mode == 2,
        "FP32-output NINT input gate mode must be sigmoid or SiLU");
    x = pad_last(
        x.contiguous().to(torch::kFloat16),
        w.neuron_len);
    gate = pad_last(
        gate.contiguous().to(torch::kFloat16),
        w.neuron_len);
    TORCH_CHECK(
        x.sizes() == gate.sizes(),
        "FP32-output NINT x and gate shapes must match");
    auto activation = mode == 1
        ? x * torch::sigmoid(gate)
        : x * torch::silu(gate);
    return nint_matmul_f32_kld(
        w, activation.contiguous());
}

static torch::Tensor nint_matmul_groupwise_u8(
        const NintWeight & w, torch::Tensor x, int64_t groups) {
    TORCH_CHECK(
        w.bits == 8 && w.gs == 48,
        "groupwise NINT projection requires NINT8 gs48");
    TORCH_CHECK(
        x.dim() == 3 && x.size(1) == groups && x.size(2) == w.neuron_len,
        "groupwise NINT projection expects [B, groups, K]");
    TORCH_CHECK(
        w.out % groups == 0,
        "groupwise NINT projection output rows must divide groups");
    x = x.contiguous().to(torch::kFloat16);
    const int input_rows = (int)(x.size(0) * groups);
    Workspace & ws = w.workspace(input_rows);
    return g_profiler.measure("nint.groupwise_u8", [&]() {
        return nint_gemv_packed_u8_groupwise_ws_cuda(
            w.q_packed, w.sub_scale, w.sub_min,
            w.neuron_scale, w.neuron_min, x, w.gs, groups,
            ws.qx, ws.xscale, ws.xsum);
    });
}

static torch::Tensor nint_matmul_input_mul(const NintWeight & w, torch::Tensor x, torch::Tensor gate, int mode) {
    x = x.contiguous().to(torch::kFloat16);
    gate = gate.contiguous().to(torch::kFloat16);
    x = pad_last(x, w.neuron_len);
    gate = pad_last(gate, w.neuron_len);
    int M = (int)x.size(0);
    if (w.q8_zero) {
        if (mode == 1) return nint_matmul(w, x * torch::sigmoid(gate));
        return nint_matmul(w, x * torch::silu(gate));
    }
    if (w.bits == 4 && M == 1) {
        Workspace & ws = w.workspace(M);
        return nint_gemv_packed_gate_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                             x, gate, w.gs, mode, ws.qx, ws.xscale, ws.xsum);
    }
    if ((w.bits == 2 || w.bits == 3 || w.bits == 5 || w.bits == 6 || w.bits == 8) && M <= 8) {
        Workspace & ws = w.workspace(M);
        return nint_gemv_packed_bits_gate_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                                  x, gate, w.gs, w.bits, mode, ws.qx, ws.xscale, ws.xsum);
    }
    if (mode == 1) return nint_matmul(w, x * torch::sigmoid(gate));
    return nint_matmul(w, x * torch::silu(gate));
}

static torch::Tensor nint_matmul_qx(const NintWeight & w, Workspace & ws) {
    if (w.q8_zero) {
        throw std::runtime_error(
            "NINT8-0 prequantized activation path is not enabled");
    }
    int M = (int)ws.qx.size(0);
    if (w.bits == 4) {
        return nint_gemv_packed_qx_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                           w.gs, ws.qx, ws.xscale, ws.xsum);
    }
    if (w.bits == 2 || w.bits == 3 || w.bits == 5 || w.bits == 6 || w.bits == 8) {
        if (M > 8) throw std::runtime_error("NINT packed-bits prequantized GEMV supports only M<=8");
        return nint_gemv_packed_bits_qx_ws_cuda(w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                                                w.gs, w.bits, ws.qx, ws.xscale, ws.xsum);
    }
    throw std::runtime_error("NINT prequantized GEMV unsupported bit width");
}

static torch::Tensor nint_matmul_swiglu(const NintWeight & w, torch::Tensor x) {
    x = x.contiguous().to(torch::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    if (w.q8_zero) {
        auto parts = nint_matmul(w, x).chunk(2, -1);
        return torch::silu(parts[0]) * parts[1];
    }
    if (M != 1) {
        throw std::runtime_error("NINT SwiGLU fusion supports only decode M=1");
    }
    Workspace & ws = w.workspace(M);
    if (w.bits == 4) {
        return nint_gemv_packed_swiglu_ws_cuda(
            w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
            x, w.gs, ws.qx, ws.xscale, ws.xsum);
    }
    if (w.bits == 2 || w.bits == 3 || w.bits == 5 || w.bits == 6 || w.bits == 8) {
        return nint_gemv_packed_bits_swiglu_ws_cuda(
            w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
            x, w.gs, w.bits, ws.qx, ws.xscale, ws.xsum);
    }
    throw std::runtime_error("NINT SwiGLU fusion unsupported bit width");
}

static torch::Tensor nint_matmul_geglu(const NintWeight & w, torch::Tensor x) {
    x = x.contiguous().to(torch::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    if (w.q8_zero) {
        auto parts = nint_matmul(w, x).chunk(2, -1);
        return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
    }
    if (M != 1) {
        throw std::runtime_error("NINT GeGLU fusion supports only decode M=1");
    }
    Workspace & ws = w.workspace(M);
    if (w.bits == 4) {
        return nint_gemv_packed_geglu_ws_cuda(
            w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
            x, w.gs, ws.qx, ws.xscale, ws.xsum);
    }
    if (w.bits == 2 || w.bits == 3 || w.bits == 5 || w.bits == 6 || w.bits == 8) {
        return nint_gemv_packed_bits_geglu_ws_cuda(
            w.q_packed, w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
            x, w.gs, w.bits, ws.qx, ws.xscale, ws.xsum);
    }
    throw std::runtime_error("NINT GeGLU fusion unsupported bit width");
}

struct NintLinear {
    NintWeight w;
    torch::Tensor forward(torch::Tensor x) const {
        auto shape = x.sizes().vec();
        int64_t last = shape.back();
        (void)last;
        auto y = nint_matmul(w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    torch::Tensor forward_input_mul(torch::Tensor x, torch::Tensor gate, int mode) const {
        auto shape = x.sizes().vec();
        auto y = nint_matmul_input_mul(w, x.reshape({-1, x.size(-1)}), gate.reshape({-1, gate.size(-1)}), mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    torch::Tensor forward_input_mul_f32_kld(
            torch::Tensor x,
            torch::Tensor gate,
            int mode) const {
        auto shape = x.sizes().vec();
        auto y = nint_matmul_input_mul_f32_kld(
            w,
            x.reshape({-1, x.size(-1)}),
            gate.reshape({-1, gate.size(-1)}),
            mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

static bool decode_branch_parallel_enabled(int64_t rows) {
    const char * disabled =
        std::getenv("MFQ_DISABLE_DECODE_BRANCH_PARALLEL");
    return rows == 1 &&
        (disabled == nullptr || disabled[0] != '1');
}

struct CudaIndependentBranchExecutor {
    using Stream =
        decltype(at::cuda::getStreamFromPool(false));

    int device = -1;
    cudaEvent_t ready = nullptr;
    std::vector<Stream> streams;
    std::vector<cudaEvent_t> completed;

    ~CudaIndependentBranchExecutor() {
        for (cudaEvent_t event : completed) {
            if (event != nullptr) {
                (void)cudaEventDestroy(event);
            }
        }
        if (ready != nullptr) {
            (void)cudaEventDestroy(ready);
        }
    }

    bool ensure(size_t branches, const Stream & parent) {
        const int parent_device = parent.device_index();
        if (device >= 0 && device != parent_device) {
            return false;
        }
        if (streams.size() >= branches) {
            return true;
        }
        cudaStreamCaptureStatus capture_status =
            cudaStreamCaptureStatusNone;
        MFQ_CUDA_CHECK(cudaStreamIsCapturing(
            parent.stream(), &capture_status));
        if (capture_status != cudaStreamCaptureStatusNone) {
            return false;
        }
        device = parent_device;
        if (ready == nullptr) {
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &ready, cudaEventDisableTiming));
        }
        while (streams.size() < branches) {
            streams.push_back(
                at::cuda::getStreamFromPool(false, device));
            cudaEvent_t event = nullptr;
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &event, cudaEventDisableTiming));
            completed.push_back(event);
        }
        return true;
    }

    template <typename Fn>
    bool run(
            size_t branches,
            Fn && fn,
            std::vector<torch::Tensor> & outputs) {
        if (branches < 2) {
            return false;
        }
        const Stream parent =
            at::cuda::getCurrentCUDAStream();
        if (!ensure(branches, parent)) {
            return false;
        }

        outputs.resize(branches);
        MFQ_CUDA_CHECK(cudaEventRecord(
            ready, parent.stream()));
        for (size_t index = 0; index < branches; ++index) {
            const Stream branch_stream = streams[index];
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                branch_stream.stream(), ready, 0));
            {
                c10::cuda::CUDAStreamGuard guard(
                    branch_stream);
                outputs[index] = fn(index);
            }
            MFQ_CUDA_CHECK(cudaEventRecord(
                completed[index], branch_stream.stream()));
        }
        for (size_t index = 0; index < branches; ++index) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                parent.stream(), completed[index], 0));
            if (outputs[index].defined()) {
                c10::cuda::CUDACachingAllocator::recordStream(
                    outputs[index].storage().data_ptr(),
                    parent);
            }
        }
        return true;
    }
};

struct NintLinearGroup {
    NintWeight w;
    std::vector<NintWeight> projection_w;
    std::vector<NintWeight> split_w;
    std::vector<std::vector<int64_t>> split_outs;
    std::vector<int64_t> outs;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        branch_executor =
            std::make_shared<CudaIndependentBranchExecutor>();
    std::vector<torch::Tensor> forward(torch::Tensor x) const {
        auto shape = x.sizes().vec();
        std::vector<torch::Tensor> parts;
        auto xf = x.reshape({-1, x.size(-1)});
        if (w.q8_zero && xf.size(0) > 64 &&
                projection_w.size() == outs.size()) {
            parts.reserve(projection_w.size());
            for (const auto & projection : projection_w) {
                parts.push_back(nint_matmul(projection, xf));
            }
        } else if (!split_w.empty()) {
            parts.reserve(outs.size());
            std::vector<torch::Tensor> grouped_outputs;
            const bool parallel =
                decode_branch_parallel_enabled(xf.size(0)) &&
                branch_executor->run(
                    split_w.size(),
                    [&](size_t index) {
                        return nint_matmul(
                            split_w[index], xf);
                    },
                    grouped_outputs);
            for (size_t i = 0; i < split_w.size(); ++i) {
                auto y = parallel
                    ? grouped_outputs[i]
                    : nint_matmul(split_w[i], xf);
                auto ys = y.split_with_sizes(split_outs[i], -1);
                for (auto & p : ys) parts.push_back(p);
            }
        } else {
            auto y = nint_matmul(w, x.reshape({-1, x.size(-1)}));
            parts = y.split_with_sizes(outs, -1);
        }
        for (auto & p : parts) {
            auto s = shape;
            s.back() = p.size(-1);
            p = p.reshape(s);
        }
        return parts;
    }
    torch::Tensor forward_swiglu(torch::Tensor x) const {
        if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("NINT SwiGLU fusion requires one packed [gate, up] group");
        }
        auto shape = x.sizes().vec();
        auto y = nint_matmul_swiglu(w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    torch::Tensor forward_geglu(torch::Tensor x) const {
        if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("NINT GeGLU fusion requires one packed [gate, up] group");
        }
        auto shape = x.sizes().vec();
        auto y = nint_matmul_geglu(w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

static NintLinearGroup make_linear_group(const std::vector<NintWeight> & ws) {
    NintLinearGroup g;
    for (const auto & w : ws) g.outs.push_back(w.out);
    bool same = !ws.empty();
    for (size_t i = 1; i < ws.size(); ++i) {
        if (ws[i].ng != ws[0].ng || ws[i].gs != ws[0].gs ||
            ws[i].bits != ws[0].bits || ws[i].qbytes != ws[0].qbytes ||
            ws[i].neuron_len != ws[0].neuron_len ||
            ws[i].q8_zero != ws[0].q8_zero) {
            same = false;
            break;
        }
    }
    if (same) {
        g.w = cat_weights(ws);
        if (g.w.q8_zero) {
            g.projection_w.reserve(ws.size());
            int64_t offset = 0;
            for (const auto & source : ws) {
                NintWeight projection = g.w;
                projection.out = source.out;
                projection.shape = source.shape;
                projection.q_packed =
                    g.w.q_packed.narrow(0, offset, source.out);
                projection.q8_zero_scale =
                    g.w.q8_zero_scale.narrow(0, offset, source.out);
                projection.workspaces.clear();
                g.projection_w.push_back(std::move(projection));
                offset += source.out;
            }
            TORCH_CHECK(offset == g.w.out,
                        "Q8 projection views do not cover grouped output");
        }
    } else {
        std::vector<NintWeight> cur;
        std::vector<int64_t> cur_outs;
        auto flush = [&]() {
            if (cur.empty()) return;
            g.split_w.push_back(cur.size() == 1 ? cur[0] : cat_weights(cur));
            g.split_outs.push_back(cur_outs);
            cur.clear();
            cur_outs.clear();
        };
        for (const auto & w : ws) {
            bool append = !cur.empty() &&
                w.ng == cur[0].ng && w.gs == cur[0].gs && w.bits == cur[0].bits &&
                w.qbytes == cur[0].qbytes &&
                w.neuron_len == cur[0].neuron_len &&
                w.q8_zero == cur[0].q8_zero;
            if (!append) flush();
            cur.push_back(w);
            cur_outs.push_back(w.out);
        }
        flush();
    }
    return g;
}

struct NvqLinear {
    NvqWeight w;
    torch::Tensor forward(torch::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = nvq_matmul(w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    torch::Tensor forward_input_mul(torch::Tensor x, torch::Tensor gate, int mode) const {
        auto shape = x.sizes().vec();
        auto y = nvq_matmul_input_mul(
            w, x.reshape({-1, x.size(-1)}), gate.reshape({-1, gate.size(-1)}), mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

static torch::Tensor mxfp8_matmul(
        const Mxfp8Weight & weight,
        torch::Tensor x) {
    x = x.contiguous().to(torch::kFloat16);
    TORCH_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 activation width mismatch");
    if (x.size(0) <= 8) {
        return g_profiler.measure("mxfp8.small_m", [&]() {
            return mxfp8_small_m_cuda(
                weight.values, weight.scales, x);
        });
    }
    auto dense = g_profiler.measure("mxfp8.dequant", [&]() {
        return mxfp8_dequant_cuda(
            weight.values, weight.scales);
    });
    return g_profiler.measure("mxfp8.gemm", [&]() {
        return nint_cublas_gemm_nt_f32acc_cuda(x, dense);
    });
}

static torch::Tensor mxfp8_matmul_f32(
        const Mxfp8Weight & weight,
        torch::Tensor x) {
    x = x.contiguous().to(torch::kFloat16);
    TORCH_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 FP32-output activation width mismatch");
    if (x.size(0) <= 8) {
        return g_profiler.measure("mxfp8.small_m_f32", [&]() {
            return mxfp8_small_m_f32_cuda(
                weight.values, weight.scales, x);
        });
    }
    return g_profiler.measure("mxfp8.gemm_f32", [&]() {
        return mxfp8_gemm_f32_cuda(
            weight.values, weight.scales, x);
    });
}

static torch::Tensor mxfp8_cpu_reference(
        const Mxfp8Weight & weight) {
    auto values = weight.values.to(torch::kCPU).contiguous();
    auto scales = weight.scales.to(torch::kCPU).contiguous();
    const auto * value_bytes = values.data_ptr<uint8_t>();
    const auto * scale_bytes = scales.data_ptr<uint8_t>();
    const int64_t scale_columns = weight.neuron_len / 128;
    std::vector<float> dense(
        static_cast<size_t>(weight.out * weight.neuron_len));
    for (int64_t row = 0; row < weight.out; ++row) {
        for (int64_t column = 0;
             column < weight.neuron_len; ++column) {
            const uint8_t raw = value_bytes[
                static_cast<size_t>(row * weight.neuron_len + column)];
            const unsigned exponent =
                (static_cast<unsigned>(raw) >> 3u) & 15u;
            const unsigned mantissa =
                static_cast<unsigned>(raw) & 7u;
            float value;
            if (exponent == 15u && mantissa == 7u) {
                value = std::numeric_limits<float>::quiet_NaN();
            } else {
                value = exponent == 0u
                    ? std::ldexp(float(mantissa) * 0.125f, -6)
                    : std::ldexp(
                        1.0f + float(mantissa) * 0.125f,
                        static_cast<int>(exponent) - 7);
                if ((raw & 128u) != 0u) value = -value;
            }
            const uint8_t raw_scale = scale_bytes[
                static_cast<size_t>(
                    (row / 128) * scale_columns + column / 128)];
            const float scale = raw_scale == 255u
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, int(raw_scale) - 127);
            dense[static_cast<size_t>(
                row * weight.neuron_len + column)] = value * scale;
        }
    }
    return torch::from_blob(
        dense.data(), {weight.out, weight.neuron_len},
        torch::TensorOptions().dtype(torch::kFloat32))
        .clone().to(weight.values.device())
        .to(torch::kFloat16).contiguous();
}

static torch::Tensor mxfp8_groupwise_matmul(
        const Mxfp8Weight & weight,
        torch::Tensor grouped,
        int64_t groups) {
    grouped = grouped.contiguous().to(torch::kFloat16);
    TORCH_CHECK(
        grouped.dim() == 3 && grouped.size(1) == groups &&
            grouped.size(2) == weight.neuron_len &&
            weight.out % groups == 0,
        "MXFP8 groupwise projection geometry mismatch");
    const int64_t outputs_per_group = weight.out / groups;
    TORCH_CHECK(
        outputs_per_group % 128 == 0,
        "MXFP8 groupwise output width must preserve scale blocks");
    if (grouped.size(0) <= 8) {
        return g_profiler.measure("mxfp8.groupwise_small_m", [&]() {
            return mxfp8_groupwise_small_m_cuda(
                weight.values, weight.scales, grouped, groups);
        });
    }
    std::vector<torch::Tensor> outputs;
    outputs.reserve(static_cast<size_t>(groups));
    const int64_t scale_rows_per_group = outputs_per_group / 128;
    for (int64_t group = 0; group < groups; ++group) {
        Mxfp8Weight shard;
        shard.out = outputs_per_group;
        shard.neuron_len = weight.neuron_len;
        shard.values = weight.values.narrow(
            0, group * outputs_per_group,
            outputs_per_group).contiguous();
        shard.scales = weight.scales.narrow(
            0, group * scale_rows_per_group,
            scale_rows_per_group).contiguous();
        outputs.push_back(mxfp8_matmul(
            shard, grouped.select(1, group).contiguous()));
    }
    return torch::cat(outputs, -1).contiguous();
}

static torch::Tensor mxfp8_groupwise_matmul_f32(
        const Mxfp8Weight & weight,
        torch::Tensor grouped,
        int64_t groups) {
    grouped = grouped.contiguous().to(torch::kFloat16);
    TORCH_CHECK(
        grouped.dim() == 3 && grouped.size(1) == groups &&
            grouped.size(2) == weight.neuron_len &&
            weight.out % groups == 0,
        "MXFP8 groupwise FP32-output projection geometry mismatch");
    const int64_t outputs_per_group = weight.out / groups;
    TORCH_CHECK(
        outputs_per_group % 128 == 0,
        "MXFP8 groupwise output width must preserve scale blocks");
    if (grouped.size(0) <= 8) {
        return g_profiler.measure(
            "mxfp8.groupwise_small_m_f32", [&]() {
                return mxfp8_groupwise_small_m_f32_cuda(
                    weight.values, weight.scales, grouped, groups);
            });
    }
    std::vector<torch::Tensor> outputs;
    outputs.reserve(static_cast<size_t>(groups));
    const int64_t scale_rows_per_group = outputs_per_group / 128;
    for (int64_t group = 0; group < groups; ++group) {
        Mxfp8Weight shard;
        shard.out = outputs_per_group;
        shard.neuron_len = weight.neuron_len;
        shard.values = weight.values.narrow(
            0, group * outputs_per_group,
            outputs_per_group).contiguous();
        shard.scales = weight.scales.narrow(
            0, group * scale_rows_per_group,
            scale_rows_per_group).contiguous();
        outputs.push_back(mxfp8_matmul_f32(
            shard, grouped.select(1, group).contiguous()));
    }
    return torch::cat(outputs, -1).contiguous();
}

struct Mxfp8Linear {
    Mxfp8Weight weight;

    torch::Tensor forward(torch::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = mxfp8_matmul(
            weight, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

enum class QuantLinearKind {
    Nint,
    Nvq,
    Mxfp8,
    Dense,
};

struct QuantLinearShard {
    int device = 0;
    int64_t input_begin = 0;
    int64_t input_end = 0;
    int64_t output_begin = 0;
    int64_t output_end = 0;
    QuantLinearKind kind = QuantLinearKind::Nint;
    NintWeight nint;
    NvqWeight nvq;
    Mxfp8Weight mxfp8;
    torch::Tensor dense;
};

static torch::Tensor run_quant_linear_shard(
        const QuantLinearShard & shard,
        torch::Tensor x,
        c10::optional<torch::Tensor> gate =
            c10::nullopt,
        int gate_mode = 0) {
    c10::cuda::CUDAGuard guard(shard.device);
    if (shard.kind == QuantLinearKind::Nint) {
        return gate.has_value()
            ? nint_matmul_input_mul(
                shard.nint, x, gate.value(), gate_mode)
            : nint_matmul(shard.nint, x);
    }
    if (shard.kind == QuantLinearKind::Nvq) {
        return gate.has_value()
            ? nvq_matmul_input_mul(
                shard.nvq, x, gate.value(), gate_mode)
            : nvq_matmul(shard.nvq, x);
    }
    if (shard.kind == QuantLinearKind::Dense) {
        auto local = x.to(shard.dense.scalar_type());
        if (gate.has_value()) {
            TORCH_CHECK(
                gate_mode == 1 || gate_mode == 2,
                "dense input gate mode must be sigmoid or SiLU");
            auto local_gate = gate.value().to(local.scalar_type());
            local = gate_mode == 1
                ? local * torch::sigmoid(local_gate)
                : local * torch::silu(local_gate);
        }
        return torch::matmul(local, shard.dense.transpose(0, 1));
    }
    TORCH_CHECK(
        !gate.has_value(),
        "MXFP8 tensor-parallel linear does not support input gating");
    return mxfp8_matmul(shard.mxfp8, x);
}

static torch::Tensor tensor_to_cuda_device(
        torch::Tensor value, int device) {
    if (value.is_cuda() && value.get_device() == device) {
        return value.contiguous();
    }
    c10::cuda::CUDAGuard guard(device);
    return value.to(
        value.options().device(torch::Device(torch::kCUDA, device)),
        true, false).contiguous();
}

static torch::Tensor reduce_tensor_parallel_outputs(
        std::vector<torch::Tensor> outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "cannot reduce an empty tensor-parallel output");
    }
#ifdef MFQ_HAVE_NCCL
    if (g_tensor_parallel_collectives.collectives_enabled &&
            outputs.size() ==
                g_tensor_parallel_collectives.devices.size()) {
        auto & runtime = g_tensor_parallel_collectives;
        const auto shape = outputs.front().sizes().vec();
        const auto output_dtype = outputs.front().scalar_type();
        const int64_t elements = outputs.front().numel();
        for (size_t index = 0; index < outputs.size(); ++index) {
            const int device = runtime.devices[index];
            if (!outputs[index].defined() || !outputs[index].is_cuda() ||
                    outputs[index].get_device() != device ||
                    outputs[index].sizes().vec() != shape) {
                throw std::runtime_error(
                    "NCCL tensor-parallel reduction received mismatched shards");
            }
            c10::cuda::CUDAGuard guard(device);
            const auto producer =
                at::cuda::getCurrentCUDAStream(device);
            MFQ_CUDA_CHECK(cudaEventRecord(
                runtime.ready[index], producer.stream()));
            const auto communication = runtime.streams[index];
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                communication.stream(), runtime.ready[index], 0));
            {
                c10::cuda::CUDAStreamGuard stream_guard(communication);
                auto & buffer = runtime.reduction_buffers[index];
                if (!buffer.defined() || buffer.sizes().vec() != shape ||
                        buffer.get_device() != device) {
                    buffer = torch::empty(
                        shape,
                        outputs[index].options().dtype(torch::kFloat32));
                }
                buffer.copy_(outputs[index], true);
                c10::cuda::CUDACachingAllocator::recordStream(
                    outputs[index].storage().data_ptr(), communication);
            }
        }

        MFQ_NCCL_CHECK(ncclGroupStart());
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto & buffer = runtime.reduction_buffers[index];
            MFQ_NCCL_CHECK(ncclAllReduce(
                buffer.data_ptr<float>(),
                buffer.data_ptr<float>(),
                static_cast<size_t>(elements),
                ncclFloat32,
                ncclSum,
                runtime.communicators[index],
                runtime.streams[index].stream()));
        }
        MFQ_NCCL_CHECK(ncclGroupEnd());

        torch::Tensor result;
        const int primary = g_tensor_parallel.primary_device();
        for (size_t index = 0; index < outputs.size(); ++index) {
            c10::cuda::CUDAGuard guard(runtime.devices[index]);
            const auto communication = runtime.streams[index];
            if (runtime.devices[index] == primary) {
                c10::cuda::CUDAStreamGuard stream_guard(communication);
                result = runtime.reduction_buffers[index]
                    .to(output_dtype).contiguous();
            }
            MFQ_CUDA_CHECK(cudaEventRecord(
                runtime.completed[index], communication.stream()));
        }
        if (!result.defined()) {
            throw std::runtime_error(
                "tensor-parallel primary device is absent from NCCL ranks");
        }
        c10::cuda::CUDAGuard primary_guard(primary);
        const auto parent = at::cuda::getCurrentCUDAStream(primary);
        const auto primary_rank = static_cast<size_t>(
            std::find(runtime.devices.begin(), runtime.devices.end(), primary) -
            runtime.devices.begin());
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(
            parent.stream(), runtime.completed[primary_rank], 0));
        c10::cuda::CUDACachingAllocator::recordStream(
            result.storage().data_ptr(), parent);
        return result;
    }
#endif
    const int primary =
        g_tensor_parallel.primary_device();
    c10::cuda::CUDAGuard primary_guard(primary);
    const auto output_dtype =
        outputs.front().scalar_type();
    torch::Tensor reduced;
    for (auto & output : outputs) {
        auto partial =
            tensor_to_cuda_device(output, primary)
                .to(torch::kFloat32);
        if (!reduced.defined()) {
            reduced = std::move(partial);
        } else {
            reduced.add_(partial);
        }
    }
    return reduced.to(output_dtype).contiguous();
}

struct QuantLinear {
    QuantLinearKind kind = QuantLinearKind::Nint;
    NintLinear nint;
    NvqLinear nvq;
    Mxfp8Linear mxfp8;
    torch::Tensor dense;
    TensorParallelAxis tensor_parallel_axis =
        TensorParallelAxis::Mirrored;
    std::vector<QuantLinearShard> tensor_parallel_shards;
    int64_t logical_out = 0;
    int64_t logical_neuron_len = 0;

    bool tensor_parallel() const {
        return !tensor_parallel_shards.empty();
    }

    bool is_nint() const { return kind == QuantLinearKind::Nint; }
    bool is_nvq() const { return kind == QuantLinearKind::Nvq; }
    bool is_mxfp8() const { return kind == QuantLinearKind::Mxfp8; }
    bool is_dense() const { return kind == QuantLinearKind::Dense; }

    torch::Tensor forward_tensor_parallel_flat(
            torch::Tensor x,
            c10::optional<torch::Tensor> gate,
            int gate_mode) const {
        TORCH_CHECK(
            tensor_parallel(),
            "tensor-parallel linear has no shards");
        TORCH_CHECK(
            tensor_parallel_axis == TensorParallelAxis::Output ||
            tensor_parallel_axis == TensorParallelAxis::Input,
            "tensor-parallel linear has an invalid axis");
        std::vector<torch::Tensor> local_outputs;
        local_outputs.reserve(tensor_parallel_shards.size());
        for (const auto & shard : tensor_parallel_shards) {
            c10::cuda::CUDAGuard guard(shard.device);
            torch::Tensor local_x = x;
            torch::Tensor local_gate;
            if (tensor_parallel_axis == TensorParallelAxis::Input) {
                local_x = x.narrow(
                    -1, shard.input_begin,
                    shard.input_end - shard.input_begin);
                if (gate.has_value()) {
                    local_gate = gate.value().narrow(
                        -1, shard.input_begin,
                        shard.input_end - shard.input_begin);
                }
            } else if (gate.has_value()) {
                local_gate = gate.value();
            }
            local_x = tensor_to_cuda_device(local_x, shard.device);
            if (gate.has_value()) {
                local_gate =
                    tensor_to_cuda_device(local_gate, shard.device);
            }
            if (is_mxfp8() &&
                    tensor_parallel_axis == TensorParallelAxis::Input) {
                TORCH_CHECK(
                    !gate.has_value(),
                    "MXFP8 input-axis tensor parallelism does not support gating");
                local_outputs.push_back(
                    mxfp8_matmul_f32(shard.mxfp8, local_x));
            } else {
                local_outputs.push_back(
                    run_quant_linear_shard(
                        shard, local_x,
                        gate.has_value()
                            ? c10::optional<torch::Tensor>(
                                local_gate)
                            : c10::nullopt,
                        gate_mode));
            }
        }

        const int primary = g_tensor_parallel.primary_device();
        c10::cuda::CUDAGuard primary_guard(primary);
        if (tensor_parallel_axis == TensorParallelAxis::Output) {
            std::vector<torch::Tensor> gathered;
            gathered.reserve(local_outputs.size());
            for (auto & output : local_outputs) {
                gathered.push_back(
                    tensor_to_cuda_device(output, primary));
            }
            return torch::cat(gathered, -1).contiguous();
        }

        auto reduced = reduce_tensor_parallel_outputs(
            std::move(local_outputs));
        return is_mxfp8()
            ? reduced.to(x.scalar_type()).contiguous()
            : reduced;
    }

    torch::Tensor forward(torch::Tensor x) const {
        if (tensor_parallel()) {
            auto shape = x.sizes().vec();
            auto y = forward_tensor_parallel_flat(
                x.reshape({-1, x.size(-1)}),
                c10::nullopt, 0);
            shape.back() = y.size(-1);
            return y.reshape(shape);
        }
        if (is_nint()) return nint.forward(x);
        if (is_nvq()) return nvq.forward(x);
        if (is_dense()) {
            return torch::matmul(
                x.to(dense.scalar_type()),
                dense.transpose(0, 1));
        }
        return mxfp8.forward(x);
    }
    torch::Tensor forward_mxfp8_groupwise(
            torch::Tensor grouped,
            int64_t groups) const {
        TORCH_CHECK(
            is_mxfp8(),
            "groupwise MXFP8 projection requires an MXFP8 tensor");
        if (!tensor_parallel()) {
            return mxfp8_groupwise_matmul(
                mxfp8.weight, grouped, groups);
        }
        TORCH_CHECK(
            tensor_parallel_axis == TensorParallelAxis::Input,
            "groupwise MXFP8 tensor parallelism requires input-axis shards");
        std::vector<torch::Tensor> partials;
        partials.reserve(tensor_parallel_shards.size());
        for (const auto & shard : tensor_parallel_shards) {
            TORCH_CHECK(
                shard.kind == QuantLinearKind::Mxfp8,
                "groupwise MXFP8 tensor-parallel shard kind mismatch");
            c10::cuda::CUDAGuard guard(shard.device);
            auto local = grouped.narrow(
                -1, shard.input_begin,
                shard.input_end - shard.input_begin);
            local = tensor_to_cuda_device(
                local, shard.device);
            partials.push_back(mxfp8_groupwise_matmul_f32(
                shard.mxfp8, local, groups));
        }
        return reduce_tensor_parallel_outputs(
            std::move(partials))
            .to(grouped.scalar_type()).contiguous();
    }
    torch::Tensor forward_input_mul(torch::Tensor x, torch::Tensor gate, int mode) const {
        if (tensor_parallel()) {
            auto shape = x.sizes().vec();
            auto y = forward_tensor_parallel_flat(
                x.reshape({-1, x.size(-1)}),
                gate.reshape({-1, gate.size(-1)}),
                mode);
            shape.back() = y.size(-1);
            return y.reshape(shape);
        }
        if (is_nint()) return nint.forward_input_mul(x, gate, mode);
        if (is_nvq()) return nvq.forward_input_mul(x, gate, mode);
        throw std::runtime_error(
            "MXFP8/dense linear does not support input gating");
    }
    torch::Tensor forward_input_mul_f32_kld(
            torch::Tensor x,
            torch::Tensor gate,
            int mode) const {
        TORCH_CHECK(
            !tensor_parallel() && is_nint(),
            "FP32-output KLD down projection requires a local NINT tensor");
        return nint.forward_input_mul_f32_kld(
            x, gate, mode);
    }
    int64_t out() const {
        return tensor_parallel()
            ? logical_out
            : (is_nint() ? nint.w.out
               : (is_nvq() ? nvq.w.out
                  : (is_dense() ? dense.size(0) : mxfp8.weight.out)));
    }
    int64_t neuron_len() const {
        return tensor_parallel()
            ? logical_neuron_len
            : (is_nint() ? nint.w.neuron_len
               : (is_nvq() ? nvq.w.neuron_len
                  : (is_dense() ? dense.size(1)
                     : mxfp8.weight.neuron_len)));
    }
};

struct QuantLinearGroup {
    bool nint_grouped = false;
    bool nvq_prefix2 = false;
    NintLinearGroup nint;
    std::vector<QuantLinear> layers;
    std::vector<int64_t> outs;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        branch_executor =
            std::make_shared<CudaIndependentBranchExecutor>();

    std::vector<torch::Tensor> forward(torch::Tensor x) const {
        if (nint_grouped) return nint.forward(x);
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                nvq_prefix2 && nvq_fusion_enabled()) {
            auto shape = x.sizes().vec();
            const auto flat =
                x.reshape({-1, x.size(-1)});
            std::vector<torch::Tensor> branch_outputs;
            const bool parallel =
                decode_branch_parallel_enabled(flat.size(0)) &&
                layers.size() > 2 &&
                branch_executor->run(
                    layers.size() - 1,
                    [&](size_t branch) {
                        if (branch == 0) {
                            return nvq_matmul_multi2(
                                layers[0].nvq.w,
                                layers[1].nvq.w,
                                flat);
                        }
                        return layers[branch + 1].forward(x);
                    },
                    branch_outputs);
            auto combined = parallel
                ? branch_outputs[0]
                : nvq_matmul_multi2(
                    layers[0].nvq.w, layers[1].nvq.w,
                    flat);
            auto pair = combined.split_with_sizes({outs[0], outs[1]}, -1);
            std::vector<torch::Tensor> result;
            result.reserve(layers.size());
            for (size_t i = 0; i < 2; ++i) {
                auto part_shape = shape;
                part_shape.back() = outs[i];
                result.push_back(pair[i].reshape(part_shape));
            }
            for (size_t i = 2; i < layers.size(); ++i) {
                result.push_back(
                    parallel
                        ? branch_outputs[i - 1]
                        : layers[i].forward(x));
            }
            return result;
        }
        std::vector<torch::Tensor> result;
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                decode_branch_parallel_enabled(
                    x.numel() / x.size(-1)) &&
                branch_executor->run(
                    layers.size(),
                    [&](size_t index) {
                        return layers[index].forward(x);
                    },
                    result)) {
            return result;
        }
        result.reserve(layers.size());
        for (const auto & layer : layers) {
            result.push_back(layer.forward(x));
        }
        return result;
    }
    torch::Tensor forward_swiglu(torch::Tensor x) const {
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                nint_grouped && nint.split_w.empty() &&
                x.numel() / x.size(-1) == 1) {
            return nint.forward_swiglu(x);
        }
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                nvq_prefix2 && layers.size() == 2 &&
                nvq_fusion_enabled()) {
            auto shape = x.sizes().vec();
            auto y = nvq_matmul_swiglu(
                layers[0].nvq.w, layers[1].nvq.w,
                x.reshape({-1, x.size(-1)}));
            shape.back() = y.size(-1);
            return y.reshape(shape);
        }
        if (outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("SwiGLU requires equal gate/up output widths");
        }
        auto parts = forward(x);
        return torch::silu(parts[0]) * parts[1];
    }
    torch::Tensor forward_geglu(torch::Tensor x) const {
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                nint_grouped && nint.split_w.empty() &&
                x.numel() / x.size(-1) == 1) {
            return nint.forward_geglu(x);
        }
        if (outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("GeGLU requires equal gate/up output widths");
        }
        auto parts = forward(x);
        return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
    }
};

static bool is_nint_linear_dtype(const std::string & dtype) {
    return dtype != "NINTM" && dtype.rfind("NINT", 0) == 0;
}

static bool is_nvq_linear_dtype(const std::string & dtype) {
    return dtype == "NPQ0-L" || dtype == "NPQ0-S" ||
           dtype == "NVQ1-L" || dtype == "NVQ1-S" || dtype == "NVQ2" ||
           dtype == "NVQ2J" || dtype == "NVQ2J-L" || dtype == "NVQ2J-XL" ||
           dtype == "NVQ3" || dtype == "NVQ3J" ||
           dtype == "NVQ3J-512" || dtype == "NVQ3J-L" ||
           dtype == "NIQ2" || dtype == "NIQ2J" || dtype == "NIQ3";
}

static TensorParallelAxis infer_tensor_parallel_axis(
        const std::string & name) {
    if (name.find("embed_tokens") != std::string::npos ||
        name.find("token_embd") != std::string::npos ||
        name.find("tok_embeddings") != std::string::npos) {
        return TensorParallelAxis::Mirrored;
    }
    if (name == "output.weight" ||
        name.find("lm_head.weight") != std::string::npos) {
        return TensorParallelAxis::Output;
    }
    if (name.find("ffn_down") != std::string::npos ||
        name.find("down_proj.weight") != std::string::npos ||
        name.find("attn_output.weight") != std::string::npos ||
        name.find(".o_proj.weight") != std::string::npos ||
        name.find(".out_proj.weight") != std::string::npos ||
        name.find("output_a.weight") != std::string::npos ||
        name.find("output_b.weight") != std::string::npos) {
        return TensorParallelAxis::Input;
    }
    return TensorParallelAxis::Output;
}

static int64_t tensor_parallel_granularity(
        int64_t extent, int64_t preferred) {
    int64_t granularity = std::max<int64_t>(
        1, std::min<int64_t>(
            preferred,
            extent /
                static_cast<int64_t>(
                    g_tensor_parallel.devices.size())));
    while (granularity > 1 &&
           extent <
               static_cast<int64_t>(
                   g_tensor_parallel.devices.size()) *
                   granularity) {
        granularity /= 2;
    }
    return std::max<int64_t>(1, granularity);
}

static std::vector<mfq::TensorParallelSlice>
plan_quant_tensor_parallel_slices(
        int64_t extent,
        int64_t preferred_granularity,
        const std::string & name) {
    (void)name;
    const int64_t granularity =
        tensor_parallel_granularity(
            extent, preferred_granularity);
    auto slices = mfq::plan_tensor_parallel_slices(
        extent, granularity,
        g_tensor_parallel.devices,
        g_tensor_parallel.split);
    mfq::validate_tensor_parallel_slices(
        slices, extent, granularity);
    return slices;
}

static std::vector<mfq::TensorParallelSlice>
plan_moe_tensor_parallel_slices(
        int64_t extent,
        const std::string & name) {
    return plan_quant_tensor_parallel_slices(
        extent, 128, name);
}

static QuantLinear load_quant_linear(
        const MfqFile & mfq,
        const std::string & name,
        std::optional<TensorParallelAxis> axis_override =
            std::nullopt,
        const std::vector<mfq::TensorParallelSlice> *
            slices_override = nullptr) {
    const auto & dtype = mfq.record(name).dtype;
    QuantLinear result;
    const TensorParallelAxis axis =
        axis_override.value_or(
            infer_tensor_parallel_axis(name));
    if (slices_override != nullptr &&
        axis != TensorParallelAxis::Output) {
        throw std::runtime_error(
            "explicit tensor-parallel slices require an output-axis weight");
    }
    auto select_slices = [&](
            int64_t extent,
            int64_t preferred) {
        if (slices_override == nullptr) {
            return plan_quant_tensor_parallel_slices(
                extent, preferred, name);
        }
        auto slices = *slices_override;
        mfq::validate_tensor_parallel_slices(
            slices, extent, 1);
        if (slices.size() !=
            g_tensor_parallel.devices.size()) {
            throw std::runtime_error(
                "explicit tensor-parallel slice count mismatch");
        }
        for (size_t index = 0;
             index < slices.size(); ++index) {
            if (slices[index].device !=
                g_tensor_parallel.devices[index]) {
                throw std::runtime_error(
                    "explicit tensor-parallel device order mismatch");
            }
        }
        return slices;
    };
    result.tensor_parallel_axis = axis;
    if (is_nint_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nint;
        const auto blob = mfq.read_blob(name);
        if (dtype == "NINT8-0") {
            const auto cpu = unpack_nint8_zero(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : 4;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint8_zero_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint8_zero_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * 32 : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * 32,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint =
                        to_cuda_device_nint8_zero(
                            shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                c10::cuda::CUDAGuard guard(
                    active_weight_load_device());
                result.nint.w =
                    to_device_nint8_zero(cpu, true);
            }
        } else {
            const auto cpu = unpack_nint(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * cpu.gs : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * cpu.gs,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint = to_cuda_device_nint(
                        shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                c10::cuda::CUDAGuard guard(
                    active_weight_load_device());
                result.nint.w =
                    to_device_nint(cpu, true);
            }
        }
    } else if (is_nvq_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nvq;
        const auto cpu =
            unpack_nvq(mfq.read_blob(name), dtype);
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (g_tensor_parallel.enabled() &&
            axis != TensorParallelAxis::Mirrored) {
            const int64_t extent =
                axis == TensorParallelAxis::Output
                ? cpu.out : cpu.ng;
            const int64_t preferred =
                axis == TensorParallelAxis::Output
                ? 128
                : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
            for (const auto & slice :
                 select_slices(
                     extent, preferred)) {
                auto shard_cpu = slice_nvq_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Nvq;
                shard.output_begin =
                    axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end =
                    axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin =
                    axis == TensorParallelAxis::Input
                    ? slice.begin * cpu.gs : 0;
                shard.input_end =
                    axis == TensorParallelAxis::Input
                    ? std::min<int64_t>(
                        slice.end * cpu.gs,
                        cpu.neuron_len)
                    : cpu.neuron_len;
                shard.nvq = to_cuda_device_nvq(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else {
            c10::cuda::CUDAGuard guard(
                active_weight_load_device());
            result.nvq.w = to_device_nvq(cpu, true);
        }
    } else if (dtype == "MXFP8") {
        result.kind = QuantLinearKind::Mxfp8;
        const auto cpu = unpack_mxfp8(mfq.read_blob(name));
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            for (const auto & slice : select_slices(extent, 128)) {
                auto shard_cpu = slice_mxfp8_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Mxfp8;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.mxfp8 = to_cuda_device_mxfp8(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else {
            result.mxfp8.weight = to_cuda_device_mxfp8(
                cpu, active_weight_load_device());
        }
    } else if (dtype == "BF16" || dtype == "F16" || dtype == "F32") {
        result.kind = QuantLinearKind::Dense;
        auto cpu = load_dense_linear_cpu(mfq, name);
        result.logical_out = cpu.size(0);
        result.logical_neuron_len = cpu.size(1);
        // Keep native floating-point linears whole on the primary TP rank.
        // Splitting these matrices changes the cuBLAS GEMM geometry and causes
        // materially larger drift than the weight formats under test. They
        // are a small fraction of the model; routed experts and MXFP8/NINT/NVQ
        // weights remain sharded.
        const char * shard_native_float_env =
            std::getenv("MFQ_TP_SHARD_NATIVE_FLOAT");
        const bool shard_native_float =
            shard_native_float_env != nullptr &&
            std::atoi(shard_native_float_env) != 0;
        if (shard_native_float && g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.size(0) : cpu.size(1);
            for (const auto & slice : select_slices(extent, 128)) {
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Dense;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.size(0);
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.size(1);
                const int64_t dimension =
                    axis == TensorParallelAxis::Output ? 0 : 1;
                c10::cuda::CUDAGuard guard(slice.device);
                shard.dense = cpu.narrow(
                        dimension, slice.begin,
                        slice.end - slice.begin)
                    .to(torch::Device(torch::kCUDA, slice.device))
                    .contiguous();
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else {
            c10::cuda::CUDAGuard guard(active_weight_load_device());
            result.dense = cpu.to(torch::kCUDA).contiguous();
        }
    } else {
        throw std::runtime_error(
            "linear tensor must be NINT/NVQ/MXFP8/BF16/F16/F32: " +
            name + " dtype=" + dtype);
    }
    return result;
}

static bool is_quant_dtype(const std::string & dtype) {
    return is_nint_linear_dtype(dtype) ||
        is_nvq_linear_dtype(dtype) || dtype == "MXFP8";
}

static QuantLinearGroup make_quant_group(std::vector<QuantLinear> layers) {
    if (layers.empty()) throw std::runtime_error("empty quantized linear group");
    QuantLinearGroup result;
    result.outs.reserve(layers.size());
    bool all_nint = true;
    std::vector<NintWeight> nint_weights;
    nint_weights.reserve(layers.size());
    for (const auto & layer : layers) {
        result.outs.push_back(layer.out());
        all_nint = all_nint && layer.is_nint();
        if (layer.is_nint() && !layer.tensor_parallel()) {
            nint_weights.push_back(layer.nint.w);
        }
    }
    const char * disable_nint_group =
        std::getenv("MFQ_DIAGNOSTIC_DISABLE_NINT_GROUP");
    const bool keep_nint_separate =
        disable_nint_group != nullptr && disable_nint_group[0] == '1';
    const bool any_tensor_parallel = std::any_of(
        layers.begin(), layers.end(),
        [](const QuantLinear & layer) {
            return layer.tensor_parallel();
        });
    if (all_nint && !keep_nint_separate &&
        !any_tensor_parallel) {
        result.nint_grouped = true;
        result.nint = make_linear_group(nint_weights);
    } else {
        result.layers = std::move(layers);
        result.nvq_prefix2 = result.layers.size() >= 2 &&
            !any_tensor_parallel &&
            result.layers[0].is_nvq() && result.layers[1].is_nvq() &&
            nvq_pair_compatible(result.layers[0].nvq.w, result.layers[1].nvq.w);
    }
    return result;
}

static bool quant_linear_pair_compatible(const QuantLinear & a, const QuantLinear & b) {
    if (a.tensor_parallel() || b.tensor_parallel()) {
        return a.tensor_parallel() && b.tensor_parallel() &&
            a.kind == b.kind &&
            a.tensor_parallel_axis == b.tensor_parallel_axis &&
            a.tensor_parallel_shards.size() ==
                b.tensor_parallel_shards.size();
    }
    if (a.kind != b.kind) return false;
    if (a.is_nvq()) return nvq_pair_compatible(a.nvq.w, b.nvq.w);
    if (a.is_mxfp8()) {
        return a.mxfp8.weight.neuron_len == b.mxfp8.weight.neuron_len;
    }
    const auto & x = a.nint.w;
    const auto & y = b.nint.w;
    return x.ng == y.ng && x.gs == y.gs && x.bits == y.bits &&
        x.qbytes == y.qbytes && x.neuron_len == y.neuron_len && x.q5_exec == y.q5_exec;
}

static QuantLinearGroup load_quant_group(
    const MfqFile & mfq, const std::vector<std::string> & names,
    size_t required_compatible_prefix = 0,
    const std::vector<mfq::TensorParallelSlice> *
        slices_override = nullptr) {
    std::vector<QuantLinear> layers;
    layers.reserve(names.size());
    for (const auto & name : names) {
        layers.push_back(
            load_quant_linear(
                mfq, name,
                slices_override != nullptr
                    ? std::optional<TensorParallelAxis>(
                        TensorParallelAxis::Output)
                    : std::nullopt,
                slices_override));
    }
    if (required_compatible_prefix > layers.size()) {
        throw std::runtime_error("invalid required quantized-group prefix length");
    }
    // A compatible prefix is fused opportunistically by make_quant_group().
    // Mixed-precision Q/K and gate/up pairs remain separate QuantLinear
    // branches and preserve the recipe-selected layouts.
    return make_quant_group(std::move(layers));
}

static std::vector<mfq::TensorParallelSlice>
tensor_parallel_output_slices_for_input(
        const QuantLinear & input_parallel) {
    if (!input_parallel.tensor_parallel() ||
        input_parallel.tensor_parallel_axis !=
            TensorParallelAxis::Input) {
        return {};
    }
    std::vector<mfq::TensorParallelSlice> result;
    result.reserve(
        input_parallel.tensor_parallel_shards.size());
    for (const auto & shard :
         input_parallel.tensor_parallel_shards) {
        result.push_back({
            shard.device,
            shard.input_begin,
            shard.input_end,
        });
    }
    mfq::validate_tensor_parallel_slices(
        result,
        input_parallel.neuron_len(),
        1);
    return result;
}

static QuantLinearGroup load_paired_gate_up(
        const MfqFile & mfq,
        const std::vector<std::string> & names,
        const QuantLinear & down,
        size_t required_compatible_prefix = 2) {
    auto slices =
        tensor_parallel_output_slices_for_input(
            down);
    return slices.empty()
        ? load_quant_group(
            mfq, names,
            required_compatible_prefix)
        : load_quant_group(
            mfq, names,
            required_compatible_prefix,
            &slices);
}

struct DenseLinearGroup {
    torch::Tensor w;
    std::vector<int64_t> outs;

    std::vector<torch::Tensor> forward(torch::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = torch::matmul(x.reshape({-1, x.size(-1)}).to(torch::kFloat32), w.transpose(0, 1));
        auto parts = y.split_with_sizes(outs, -1);
        for (auto & p : parts) {
            auto s = shape;
            s.back() = p.size(-1);
            p = p.reshape(s);
        }
        return parts;
    }
};

static DenseLinearGroup make_dense_group(const std::vector<torch::Tensor> & ws) {
    if (ws.empty()) throw std::runtime_error("empty dense group");
    DenseLinearGroup g;
    std::vector<torch::Tensor> parts;
    parts.reserve(ws.size());
    for (const auto & w : ws) {
        if (w.dim() != 2) throw std::runtime_error("dense linear group expects 2D weights");
        if (!parts.empty() && w.size(1) != parts[0].size(1)) {
            throw std::runtime_error("cannot group dense tensors with different input width");
        }
        g.outs.push_back(w.size(0));
        parts.push_back(w.to(torch::kFloat32));
    }
    g.w = torch::cat(parts, 0).contiguous();
    return g;
}

static torch::Tensor dequant_nint_dense_f32(const NintWeight & w) {
    torch::Tensor dense;
    if (w.q8_zero) {
        dense = nint8_zero_dequant_cuda(
            w.q_packed, w.q8_zero_scale, w.neuron_len);
    } else {
        if (w.q5_exec) {
            throw std::runtime_error(
                "FP32 compressor projection does not accept Q5 execution-only weights");
        }
        dense = w.bits == 4
            ? nint_dequant_full_packed_compact_cuda(
                w.q_packed, w.sub_scale, w.sub_min,
                w.neuron_scale, w.neuron_min, w.neuron_len, w.gs)
            : nint_dequant_full_packed_compact_bits_cuda(
                w.q_packed, w.sub_scale, w.sub_min,
                w.neuron_scale, w.neuron_min,
                w.neuron_len, w.gs, w.bits);
    }
    return dense.to(torch::kFloat32).contiguous();
}

static torch::Tensor dequant_quant_linear_f32(const QuantLinear & linear) {
    if (!linear.tensor_parallel()) {
        if (linear.is_nint()) {
            return dequant_nint_dense_f32(linear.nint.w);
        }
        if (linear.is_nvq()) {
            return nvq_dequant(linear.nvq.w)
                .to(torch::kFloat32).contiguous();
        }
        if (linear.is_mxfp8()) {
            return mxfp8_dequant_cuda(
                linear.mxfp8.weight.values,
                linear.mxfp8.weight.scales)
                .to(torch::kFloat32).contiguous();
        }
        if (linear.is_dense()) {
            return linear.dense.to(torch::kFloat32).contiguous();
        }
        throw std::runtime_error(
            "unsupported linear kind for FP32 reconstruction");
    }
    if (linear.tensor_parallel_axis != TensorParallelAxis::Output &&
        linear.tensor_parallel_axis != TensorParallelAxis::Input) {
        throw std::runtime_error(
            "cannot reconstruct a mirrored tensor-parallel linear");
    }
    const int primary = g_tensor_parallel.primary_device();
    std::vector<torch::Tensor> parts;
    parts.reserve(linear.tensor_parallel_shards.size());
    for (const auto & shard : linear.tensor_parallel_shards) {
        c10::cuda::CUDAGuard shard_guard(shard.device);
        torch::Tensor part;
        if (shard.kind == QuantLinearKind::Nint) {
            part = dequant_nint_dense_f32(shard.nint);
        } else if (shard.kind == QuantLinearKind::Nvq) {
            part = nvq_dequant(shard.nvq)
                .to(torch::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp8) {
            part = mxfp8_dequant_cuda(
                shard.mxfp8.values,
                shard.mxfp8.scales)
                .to(torch::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Dense) {
            part = shard.dense.to(torch::kFloat32).contiguous();
        } else {
            throw std::runtime_error(
                "unsupported tensor-parallel shard kind for reconstruction");
        }
        parts.push_back(
            tensor_to_cuda_device(part, primary)
                .to(torch::kFloat32).contiguous());
    }
    c10::cuda::CUDAGuard primary_guard(primary);
    auto dense = torch::cat(
        parts,
        linear.tensor_parallel_axis == TensorParallelAxis::Output
            ? 0 : 1).contiguous();
    if (dense.size(0) != linear.out() ||
        dense.size(1) != linear.neuron_len()) {
        throw std::runtime_error(
            "reconstructed tensor-parallel linear shape mismatch");
    }
    return dense;
}

static DenseLinearGroup make_fp32_quant_group(QuantLinearGroup group) {
    DenseLinearGroup dense;
    dense.outs = group.outs;
    if (group.nint_grouped) {
        if (group.nint.split_w.empty()) {
            dense.w = dequant_nint_dense_f32(group.nint.w);
        } else {
            std::vector<torch::Tensor> parts;
            parts.reserve(group.nint.split_w.size());
            for (const auto & weight : group.nint.split_w) {
                parts.push_back(dequant_nint_dense_f32(weight));
            }
            dense.w = torch::cat(parts, 0).contiguous();
        }
    } else {
        std::vector<torch::Tensor> parts;
        parts.reserve(group.layers.size());
        for (const auto & linear : group.layers) {
            parts.push_back(dequant_quant_linear_f32(linear));
        }
        dense.w = torch::cat(parts, 0).contiguous();
    }
    TORCH_CHECK(
        dense.w.dim() == 2 &&
            dense.w.size(0) ==
                std::accumulate(
                    dense.outs.begin(), dense.outs.end(), int64_t{0}),
        "FP32 compressor projection shape mismatch");
    return dense;
}

struct Config {
    std::string model_type;
    int64_t vocab_size = 0, hidden_size = 0, intermediate_size = 0, num_hidden_layers = 0;
    int64_t num_attention_heads = 0, num_key_value_heads = 0, max_position_embeddings = 0;
    int64_t head_dim = 0;
    double rope_base = 1000000.0;
    double swa_rope_base = 10000.0;
    int64_t rotary_dim = 0;
    int64_t global_head_dim = 0, num_global_key_value_heads = 0;
    int64_t sliding_window = 0;
    double full_rotary_factor = 1.0;
    bool attention_k_eq_v = false;
    double final_logit_softcapping = 0.0;
    double embed_scale = 1.0;
    double rms_norm_eps = 1e-6;
    bool tie_word_embeddings = false;
    bool qwen35_attn_q_gate = false;
    int64_t linear_conv_kernel_dim = 4, linear_key_head_dim = 128, linear_value_head_dim = 128;
    int64_t linear_num_key_heads = 0, linear_num_value_heads = 0;
    int64_t num_experts = 0, num_experts_per_tok = 0;
    int64_t moe_intermediate_size = 0, shared_expert_intermediate_size = 0;
    int64_t n_shared_experts = 0, first_k_dense_replace = 0, moe_layer_freq = 1;
    int64_t q_lora_rank = 0, kv_lora_rank = 0;
    int64_t qk_nope_head_dim = 0, qk_rope_head_dim = 0, v_head_dim = 0;
    int64_t index_head_dim = 0, index_n_heads = 0, index_topk = 0;
    int64_t index_topk_freq = 1, index_skip_topk_offset = 0;
    int64_t hc_mult = 1, hc_sinkhorn_iters = 0;
    int64_t o_groups = 1, o_lora_rank = 0;
    int64_t rope_original_positions = 0;
    int64_t n_group = 1, topk_group = 1;
    double routed_scaling_factor = 1.0;
    double hc_eps = 1e-6, swiglu_limit = 0.0;
    double rope_factor = 1.0, rope_beta_fast = 32.0, rope_beta_slow = 1.0;
    double compress_rope_base = 0.0;
    bool norm_topk_prob = false;
    std::string expert_gating_func = "softmax";
    int64_t dsv4_hash_layer_count = 0;
    double norm_weight_offset = 1.0;
    std::vector<std::string> layer_types;
    std::vector<std::string> indexer_types;
    std::vector<std::string> mlp_layer_types;
    std::vector<int64_t> compress_ratios;
    bool is_gemma4() const { return model_type == "gemma4" || model_type == "gemma4_text"; }
    bool is_glm_dsa() const { return model_type == "glm_moe_dsa"; }
    bool is_dsv4() const { return model_type == "deepseek_v4"; }
    int64_t attention_size() const { return num_attention_heads * head_dim; }
    int64_t kv_size() const { return num_key_value_heads * head_dim; }
    int64_t linear_k_size() const { return linear_num_key_heads * linear_key_head_dim; }
    int64_t linear_v_size() const { return linear_num_value_heads * linear_value_head_dim; }
};

static std::string slurp(const std::string & path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("cannot open config: " + path);
    return std::string(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>());
}

static int64_t json_int(const std::string & s, const std::string & key, int64_t def = -1) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch m;
    if (std::regex_search(s, m, re)) return std::stoll(m[1].str());
    if (def >= 0) return def;
    throw std::runtime_error("missing config int: " + key);
}

static double json_float(const std::string & s, const std::string & key, double def) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)");
    std::smatch m;
    if (std::regex_search(s, m, re)) return std::stod(m[1].str());
    return def;
}

static bool json_bool(const std::string & s, const std::string & key, bool def) {
    std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch m;
    if (std::regex_search(s, m, re)) return m[1].str() == "true";
    return def;
}

static std::string json_string(const std::string & s, const std::string & key,
                               const std::string & def = "") {
    std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch m;
    if (std::regex_search(s, m, re)) return m[1].str();
    return def;
}

static double json_object_float(const std::string & s, const std::string & object,
                                const std::string & key, double def) {
    std::regex re("\"" + object + "\"\\s*:\\s*\\{[^}]*\"" + key +
                  "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)");
    std::smatch m;
    if (std::regex_search(s, m, re)) return std::stod(m[1].str());
    return def;
}

static std::vector<std::string> json_layer_types(const std::string & s, int64_t n) {
    std::vector<std::string> out;
    std::regex re("\"layer_types\"\\s*:\\s*\\[([^\\]]+)\\]");
    std::smatch m;
    if (std::regex_search(s, m, re)) {
        std::string body = m[1].str();
        std::regex item("\"([^\"]+)\"");
        for (auto it = std::sregex_iterator(body.begin(), body.end(), item); it != std::sregex_iterator(); ++it) {
            out.push_back((*it)[1].str());
        }
    }
    if (out.empty()) out.assign((size_t)n, "full_attention");
    return out;
}

static std::vector<std::string> json_string_array(
    const std::string & s, const std::string & key) {
    std::vector<std::string> out;
    std::regex re("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch m;
    if (!std::regex_search(s, m, re)) return out;
    std::string body = m[1].str();
    std::regex item("\"([^\"]+)\"");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), item);
         it != std::sregex_iterator(); ++it) {
        out.push_back((*it)[1].str());
    }
    return out;
}

static std::vector<int64_t> json_int_array(
    const std::string & s, const std::string & key) {
    std::vector<int64_t> out;
    std::regex re("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch m;
    if (!std::regex_search(s, m, re)) return out;
    std::string body = m[1].str();
    std::regex item("-?\\d+");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), item);
         it != std::sregex_iterator(); ++it) {
        out.push_back(std::stoll((*it)[0].str()));
    }
    return out;
}

static Config parse_config_json(const std::string & s) {
    Config c;
    c.model_type = json_string(s, "model_type");
    c.vocab_size = json_int(s, "vocab_size");
    c.hidden_size = json_int(s, "hidden_size");
    c.intermediate_size = json_int(s, "intermediate_size", 0);
    c.num_hidden_layers = json_int(s, "num_hidden_layers");
    c.num_attention_heads = json_int(s, "num_attention_heads");
    c.num_key_value_heads = json_int(s, "num_key_value_heads");
    c.max_position_embeddings = json_int(s, "max_position_embeddings");
    c.head_dim = json_int(s, "head_dim", c.hidden_size / c.num_attention_heads);
    c.global_head_dim = json_int(s, "global_head_dim", c.head_dim);
    c.num_global_key_value_heads = json_int(
        s, "num_global_key_value_heads", c.num_key_value_heads);
    c.sliding_window = json_int(s, "sliding_window", 0);
    c.rope_base = json_object_float(
        s, "full_attention", "rope_theta", json_float(s, "rope_theta", 1000000.0));
    c.swa_rope_base = json_object_float(
        s, "sliding_attention", "rope_theta", 10000.0);
    double partial = json_object_float(
        s, "full_attention", "partial_rotary_factor",
        json_float(s, "partial_rotary_factor", 1.0));
    c.full_rotary_factor = partial;
    c.rotary_dim = (int64_t)std::llround(partial * (double)c.head_dim);
    c.rms_norm_eps = json_float(s, "rms_norm_eps", 1e-6);
    c.tie_word_embeddings = json_bool(s, "tie_word_embeddings", false);
    c.attention_k_eq_v = json_bool(s, "attention_k_eq_v", false);
    c.final_logit_softcapping = json_float(s, "final_logit_softcapping", 0.0);
    c.qwen35_attn_q_gate = json_bool(s, "attn_output_gate", false);
    c.linear_conv_kernel_dim = json_int(s, "linear_conv_kernel_dim", 4);
    c.linear_key_head_dim = json_int(s, "linear_key_head_dim", 128);
    c.linear_value_head_dim = json_int(s, "linear_value_head_dim", 128);
    c.linear_num_key_heads = json_int(s, "linear_num_key_heads", c.num_key_value_heads);
    c.linear_num_value_heads = json_int(s, "linear_num_value_heads", c.num_attention_heads);
    c.num_experts = json_int(s, "num_experts", 0);
    if (c.num_experts <= 0) c.num_experts = json_int(s, "n_routed_experts", 0);
    c.num_experts_per_tok = json_int(s, "num_experts_per_tok", 0);
    if (c.num_experts_per_tok <= 0) {
        c.num_experts_per_tok = json_int(s, "top_k_experts", 0);
    }
    c.moe_intermediate_size = json_int(s, "moe_intermediate_size", 0);
    c.shared_expert_intermediate_size = json_int(s, "shared_expert_intermediate_size", 0);
    c.n_shared_experts = json_int(s, "n_shared_experts", 0);
    c.first_k_dense_replace = json_int(s, "first_k_dense_replace", 0);
    c.moe_layer_freq = json_int(s, "moe_layer_freq", 1);
    c.q_lora_rank = json_int(s, "q_lora_rank", 0);
    c.kv_lora_rank = json_int(s, "kv_lora_rank", 0);
    c.qk_nope_head_dim = json_int(s, "qk_nope_head_dim", 0);
    c.qk_rope_head_dim = json_int(s, "qk_rope_head_dim", 0);
    c.v_head_dim = json_int(s, "v_head_dim", 0);
    c.index_head_dim = json_int(s, "index_head_dim", 0);
    c.index_n_heads = json_int(s, "index_n_heads", 0);
    c.index_topk = json_int(s, "index_topk", 0);
    c.index_topk_freq = json_int(s, "index_topk_freq", 1);
    c.index_skip_topk_offset = json_int(s, "index_skip_topk_offset", 0);
    c.hc_mult = json_int(s, "hc_mult", 1);
    c.hc_sinkhorn_iters = json_int(s, "hc_sinkhorn_iters", 0);
    c.hc_eps = json_float(s, "hc_eps", 1e-6);
    c.o_groups = json_int(s, "o_groups", 1);
    c.o_lora_rank = json_int(s, "o_lora_rank", 0);
    c.swiglu_limit = json_float(s, "swiglu_limit", 0.0);
    c.compress_rope_base = json_float(s, "compress_rope_theta", c.rope_base);
    c.rope_original_positions = static_cast<int64_t>(json_object_float(
        s, "rope_scaling", "original_max_position_embeddings", 0.0));
    c.rope_factor = json_object_float(s, "rope_scaling", "factor", 1.0);
    c.rope_beta_fast = json_object_float(s, "rope_scaling", "beta_fast", 32.0);
    c.rope_beta_slow = json_object_float(s, "rope_scaling", "beta_slow", 1.0);
    c.compress_ratios = json_int_array(s, "compress_ratios");
    c.n_group = json_int(s, "n_group", 1);
    c.topk_group = json_int(s, "topk_group", 1);
    c.routed_scaling_factor = json_float(s, "routed_scaling_factor", 1.0);
    c.norm_topk_prob = json_bool(s, "norm_topk_prob", false);
    c.expert_gating_func = json_string(s, "scoring_func", "softmax");
    c.dsv4_hash_layer_count = json_int(
        s, "num_hash_layers", json_int(s, "n_hash_layers", 0));
    c.layer_types = json_layer_types(s, c.num_hidden_layers);
    c.indexer_types = json_string_array(s, "indexer_types");
    c.mlp_layer_types = json_string_array(s, "mlp_layer_types");
    if (c.is_glm_dsa()) {
        c.layer_types.assign(static_cast<size_t>(c.num_hidden_layers), "glm_dsa");
        c.rotary_dim = c.qk_rope_head_dim;
        if (c.shared_expert_intermediate_size <= 0) {
            c.shared_expert_intermediate_size =
                c.n_shared_experts * c.moe_intermediate_size;
        }
        if (c.indexer_types.empty()) {
            c.indexer_types.reserve(static_cast<size_t>(c.num_hidden_layers));
            const int64_t freq = std::max<int64_t>(1, c.index_topk_freq);
            for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
                const int64_t phase = std::max<int64_t>(
                    i - c.index_skip_topk_offset + 1, 0);
                c.indexer_types.push_back(
                    phase % freq == 0 ? "full" : "shared");
            }
        }
        if (c.mlp_layer_types.empty()) {
            c.mlp_layer_types.reserve(static_cast<size_t>(c.num_hidden_layers));
            for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
                const bool sparse = i >= c.first_k_dense_replace &&
                    i % std::max<int64_t>(1, c.moe_layer_freq) == 0;
                c.mlp_layer_types.push_back(sparse ? "sparse" : "dense");
            }
        }
        if (c.indexer_types.size() != static_cast<size_t>(c.num_hidden_layers) ||
            c.mlp_layer_types.size() != static_cast<size_t>(c.num_hidden_layers)) {
            throw std::runtime_error(
                "GLM DSA layer schedules do not match num_hidden_layers");
        }
        bool have_full_indexer = false;
        for (size_t i = 0; i < c.indexer_types.size(); ++i) {
            const auto & indexer = c.indexer_types[i];
            if (indexer == "full") {
                have_full_indexer = true;
            } else if (indexer != "shared" || !have_full_indexer) {
                throw std::runtime_error(
                    "invalid GLM DSA indexer schedule at layer " +
                    std::to_string(i));
            }
            const auto & mlp = c.mlp_layer_types[i];
            if (mlp != "dense" && mlp != "sparse") {
                throw std::runtime_error(
                    "invalid GLM DSA MLP schedule at layer " +
                    std::to_string(i));
            }
        }
        if (c.q_lora_rank <= 0 || c.kv_lora_rank <= 0 ||
            c.qk_nope_head_dim <= 0 || c.qk_rope_head_dim <= 0 ||
            c.v_head_dim <= 0 || c.index_head_dim <= 0 ||
            c.index_n_heads <= 0 || c.index_topk <= 0 ||
            c.num_experts <= 0 || c.num_experts_per_tok <= 0 ||
            c.num_attention_heads != 64 || c.num_key_value_heads != 64 ||
            c.kv_lora_rank != 512 ||
            c.qk_nope_head_dim != 192 || c.qk_rope_head_dim != 64 ||
            c.v_head_dim != 256 || c.index_head_dim != 128 ||
            c.index_n_heads != 32 || c.index_topk != 2048 ||
            json_int(s, "qk_head_dim", 0) !=
                c.qk_nope_head_dim + c.qk_rope_head_dim ||
            json_bool(s, "attention_bias", false) ||
            !json_bool(s, "rope_interleave", true) ||
            !json_bool(s, "indexer_rope_interleave", true) ||
            json_string(s, "hidden_act") != "silu" ||
            c.n_group != 1 || c.topk_group != 1 ||
            c.n_shared_experts != 1 ||
            json_string(s, "scoring_func") != "sigmoid" ||
            json_string(s, "topk_method") != "noaux_tc") {
            throw std::runtime_error("unsupported GLM DSA configuration");
        }
    }
    if (c.is_gemma4()) {
        c.norm_weight_offset = 0.0;
        c.embed_scale = static_cast<float>(
            at::BFloat16(static_cast<float>(std::sqrt((double)c.hidden_size))));
    }
    if (c.is_dsv4()) {
        c.norm_weight_offset = 0.0;
        c.layer_types.assign(
            static_cast<size_t>(c.num_hidden_layers), "deepseek_v4");
        if (c.shared_expert_intermediate_size <= 0) {
            c.shared_expert_intermediate_size =
                c.n_shared_experts * c.moe_intermediate_size;
        }
        if (c.compress_ratios.size() < static_cast<size_t>(c.num_hidden_layers)) {
            throw std::runtime_error(
                "DeepSeek V4 compress_ratios is shorter than num_hidden_layers");
        }
        c.compress_ratios.resize(static_cast<size_t>(c.num_hidden_layers));
        if (c.hidden_size != 4096 || c.num_attention_heads != 64 ||
            c.head_dim != 512 || c.q_lora_rank != 1024 ||
            c.qk_rope_head_dim != 64 || c.index_head_dim != 128 ||
            c.index_n_heads != 64 || c.index_topk != 512 ||
            c.o_groups != 8 || c.o_lora_rank != 1024 ||
            c.hc_mult != 4 || c.hc_sinkhorn_iters != 20 ||
            c.num_experts != 256 || c.num_experts_per_tok != 6 ||
            c.moe_intermediate_size != 2048 ||
            c.n_shared_experts != 1 ||
            c.expert_gating_func != "sqrtsoftplus") {
            throw std::runtime_error("unsupported DeepSeek V4 configuration");
        }
    }
    return c;
}

static constexpr const char * MFQ_MODEL_CONFIG_ASSET =
    "__mfq_asset__/model_config.json";
static constexpr const char * MFQ_TOKENIZER_GGUF_ASSET =
    "__mfq_asset__/tokenizer.gguf";

static Config load_config(
        const MfqFile & mfq,
        const std::string & external_path) {
    if (!external_path.empty()) {
        return parse_config_json(slurp(external_path));
    }
    if (!mfq.has_record(MFQ_MODEL_CONFIG_ASSET)) {
        throw std::runtime_error(
            "MFQ has no embedded model config; legacy files require --config");
    }
    return parse_config_json(mfq.read_asset_text(MFQ_MODEL_CONFIG_ASSET));
}

static std::string layer_name(const std::string & templ, int i) {
    std::string s = templ;
    auto p = s.find("{i}");
    if (p != std::string::npos) s.replace(p, 3, std::to_string(i));
    return s;
}

static torch::Tensor qwen_rms_norm(torch::Tensor x, torch::Tensor weight, const Config & c) {
    return rms_norm_offset_cuda(x, weight, c.rms_norm_eps, c.norm_weight_offset);
}

static torch::Tensor gemma_rms_norm_f16(
    torch::Tensor x, torch::Tensor weight, const Config & c) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat16,
                "gemma_rms_norm_f16: activation must remain f16");
    return rms_norm_f16_cuda(
        x.contiguous(), weight, c.rms_norm_eps, c.norm_weight_offset);
}

struct RopeCache {
    torch::Tensor cos;
    torch::Tensor sin;
    torch::Tensor sections;
    int64_t rotary_dim = 0;

    RopeCache() = default;
    RopeCache(
        int64_t max_positions,
        int64_t dim,
        double base,
        int64_t frequency_dim = 0,
        int64_t active_pairs = -1) : rotary_dim(dim) {
        int64_t half = rotary_dim / 2;
        const int64_t denominator = frequency_dim > 0 ? frequency_dim : rotary_dim;
        if (active_pairs < 0) active_pairs = half;
        if (active_pairs > half) {
            throw std::runtime_error("RoPE active pair count exceeds rotary dimension");
        }
        auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        auto pos = torch::arange(max_positions, opts);
        auto ar = torch::arange(0, rotary_dim, 2, opts);
        auto freq = torch::pow(torch::full({half}, base, opts), -ar / (double)denominator);
        if (active_pairs < half) {
            auto pair = torch::arange(half, opts);
            freq = torch::where(pair < active_pairs, freq, torch::zeros_like(freq));
        }
        auto ang = pos.unsqueeze(1) * freq.unsqueeze(0);
        cos = torch::cos(ang).contiguous();
        sin = torch::sin(ang).contiguous();
        sections = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    }
    explicit RopeCache(const Config & c)
        : RopeCache(c.max_position_embeddings, c.rotary_dim, c.rope_base) {}
    torch::Tensor apply(torch::Tensor x, torch::Tensor pos, const Config & c) const {
        return rope_table_cuda(x.contiguous().to(torch::kFloat32), pos.contiguous().to(torch::kCUDA, torch::kInt64),
                               cos, sin, rotary_dim, sections);
    }
};

struct FFN {
    QuantLinearGroup gate_up;
    QuantLinear down;
    std::unique_ptr<FFN> important_neurons;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        important_neuron_executor =
            std::make_shared<CudaIndependentBranchExecutor>();
    bool geglu = false;
    bool is_moe = false;
    bool moe_split_gate_up = false;
    NintMoeWeight moe_gate_up;
    NintMoeWeight moe_gate;
    NintMoeWeight moe_up;
    NintMoeWeight moe_down;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_down;
    torch::Tensor moe_router;
    torch::Tensor moe_shared_gate;
    torch::Tensor moe_router_bias;
    torch::Tensor moe_hash_ids;
    std::unique_ptr<FFN> shared;
    int moe_top_k = 0;
    bool moe_use_sigmoid = false;
    bool moe_use_sqrt_softplus = false;
    bool moe_normalize = false;
    bool moe_delayed_softmax = true;
    bool moe_shared_ungated = false;
    double moe_router_scale = 1.0;
    double swiglu_limit = 0.0;
    int moe_layer = -1;

    bool tensor_parallel_dense_compatible() const {
        if (is_moe ||
            gate_up.layers.size() != 2 ||
            !down.tensor_parallel() ||
            down.tensor_parallel_axis !=
                TensorParallelAxis::Input) {
            return false;
        }
        const auto & gate = gate_up.layers[0];
        const auto & up = gate_up.layers[1];
        if (!gate.tensor_parallel() ||
            !up.tensor_parallel() ||
            gate.tensor_parallel_axis !=
                TensorParallelAxis::Output ||
            up.tensor_parallel_axis !=
                TensorParallelAxis::Output ||
            gate.tensor_parallel_shards.size() !=
                up.tensor_parallel_shards.size() ||
            gate.tensor_parallel_shards.size() !=
                down.tensor_parallel_shards.size()) {
            return false;
        }
        for (size_t index = 0;
             index < gate.tensor_parallel_shards.size();
             ++index) {
            const auto & gate_shard =
                gate.tensor_parallel_shards[index];
            const auto & up_shard =
                up.tensor_parallel_shards[index];
            const auto & down_shard =
                down.tensor_parallel_shards[index];
            if (gate_shard.device != up_shard.device ||
                gate_shard.device != down_shard.device ||
                gate_shard.output_begin !=
                    up_shard.output_begin ||
                gate_shard.output_end !=
                    up_shard.output_end ||
                gate_shard.output_begin !=
                    down_shard.input_begin ||
                gate_shard.output_end !=
                    down_shard.input_end) {
                return false;
            }
        }
        return true;
    }

    torch::Tensor forward_tensor_parallel_dense(
            torch::Tensor xh) const {
        auto shape = xh.sizes().vec();
        auto flat = xh.reshape(
            {-1, xh.size(-1)});
        std::vector<torch::Tensor> partials;
        partials.reserve(
            down.tensor_parallel_shards.size());
        for (size_t index = 0;
             index <
                 down.tensor_parallel_shards.size();
             ++index) {
            const auto & gate_shard =
                gate_up.layers[0]
                    .tensor_parallel_shards[index];
            const auto & up_shard =
                gate_up.layers[1]
                    .tensor_parallel_shards[index];
            const auto & down_shard =
                down.tensor_parallel_shards[index];
            c10::cuda::CUDAGuard guard(
                gate_shard.device);
            auto local_x =
                tensor_to_cuda_device(
                    flat, gate_shard.device);
            auto gate_output =
                run_quant_linear_shard(
                    gate_shard, local_x);
            auto up_output =
                run_quant_linear_shard(
                    up_shard, local_x);
            torch::Tensor activation;
            if (geglu) {
                activation = gelu_mul_cuda(
                    gate_output.contiguous(),
                    up_output.contiguous());
            } else if (swiglu_limit > 0.0) {
                auto clipped_gate =
                    torch::clamp_max(
                        gate_output.to(
                            torch::kFloat32),
                        swiglu_limit);
                auto clipped_up = torch::clamp(
                    up_output.to(
                            torch::kFloat32),
                        -swiglu_limit,
                        swiglu_limit);
                activation =
                    (torch::silu(clipped_gate) *
                     clipped_up)
                        .to(torch::kFloat16)
                        .contiguous();
            } else {
                activation =
                    (torch::silu(gate_output) *
                     up_output).contiguous();
            }
            partials.push_back(
                run_quant_linear_shard(
                    down_shard, activation));
        }
        auto output =
            reduce_tensor_parallel_outputs(
                std::move(partials));
        shape.back() = output.size(-1);
        return output.reshape(shape);
    }

    bool tensor_parallel_moe_compatible() const {
        if (!is_moe || moe_split_gate_up ||
                !moe_gate_up.tensor_parallel_experts ||
                !moe_down.tensor_parallel_experts ||
                moe_gate_up.tensor_parallel_shards.size() !=
                    moe_down.tensor_parallel_shards.size() ||
                moe_gate_up.tensor_parallel_shards.empty()) {
            return false;
        }
        for (size_t index = 0;
             index < moe_gate_up.tensor_parallel_shards.size(); ++index) {
            const auto & gate = moe_gate_up.tensor_parallel_shards[index];
            const auto & down = moe_down.tensor_parallel_shards[index];
            if (!gate.weight || !down.weight ||
                    gate.device != down.device ||
                    gate.output_begin != down.output_begin ||
                    gate.output_end != down.output_end) {
                return false;
            }
        }
        return true;
    }

    torch::Tensor forward_tensor_parallel_moe(
            torch::Tensor x,
            const MoeRoutePlan & route,
            torch::Tensor route_weights) const {
        std::vector<torch::Tensor> routed_partials;
        std::vector<torch::Tensor> down_partials;
        routed_partials.reserve(
            moe_gate_up.tensor_parallel_shards.size());
        const bool collect_output_energy =
            moe_route_stats_path() != nullptr &&
            moe_route_output_energy_enabled();
        if (collect_output_energy) {
            down_partials.reserve(
                moe_gate_up.tensor_parallel_shards.size());
        }
        static const bool disable_swiglu_quant_fusion = [] {
            const char * value = std::getenv(
                "MFQ_DISABLE_MOE_SWIGLU_QUANT_FUSION");
            return value != nullptr && std::atoi(value) != 0;
        }();
        for (size_t index = 0;
             index < moe_gate_up.tensor_parallel_shards.size(); ++index) {
            const auto & gate_shard =
                moe_gate_up.tensor_parallel_shards[index];
            const auto & down_shard =
                moe_down.tensor_parallel_shards[index];
            c10::cuda::CUDAGuard guard(gate_shard.device);
            auto local_x = tensor_to_cuda_device(x, gate_shard.device);
            auto local_weights = tensor_to_cuda_device(
                route_weights, gate_shard.device);
            auto local_route = moe_route_to_device(
                route, gate_shard.device);
            auto gate_up_pair = gate_shard.weight->forward(
                local_x, local_route);
            torch::Tensor down_pair;
            const bool allow_fusion =
                !g_force_moe_materialized_swiglu &&
                !disable_swiglu_quant_fusion &&
                moe_small_hetero_enabled(
                    static_cast<int>(gate_up_pair.size(0)));
            if (swiglu_limit <= 0.0 && allow_fusion &&
                    down_shard.weight->hetero_supported) {
                down_pair = down_shard.weight->forward_swiglu(
                    gate_up_pair, local_route);
            } else if (swiglu_limit > 0.0 && allow_fusion &&
                    down_shard.weight->supports_clamped_swiglu()) {
                down_pair = down_shard.weight->forward_clamped_swiglu(
                    gate_up_pair, local_route, swiglu_limit);
            } else {
                torch::Tensor hidden;
                if (swiglu_limit <= 0.0) {
                    hidden = moe_swiglu_split_cuda(gate_up_pair);
                } else {
                    const int64_t width = gate_up_pair.size(-1) / 2;
                    auto gate = torch::clamp_max(
                        gate_up_pair.slice(-1, 0, width)
                            .to(torch::kFloat32),
                        swiglu_limit);
                    auto up = torch::clamp(
                        gate_up_pair.slice(-1, width, 2 * width)
                            .to(torch::kFloat32),
                        -swiglu_limit, swiglu_limit);
                    hidden = (torch::silu(gate) * up)
                        .to(torch::kFloat16).contiguous();
                }
                down_pair = down_shard.weight->forward(
                    hidden, local_route);
            }
            if (collect_output_energy) {
                down_partials.push_back(down_pair);
            }
            routed_partials.push_back(
                moe_weighted_reduce_cuda(
                    down_pair, local_weights));
        }
        if (moe_route_stats_path() != nullptr) {
            torch::Tensor complete_down;
            if (collect_output_energy) {
                complete_down = reduce_tensor_parallel_outputs(
                    std::move(down_partials));
            }
            record_moe_route_stats(
                moe_layer, route.ids, route_weights,
                complete_down, moe_gate_up.n_experts);
        }
        return reduce_tensor_parallel_outputs(
            std::move(routed_partials));
    }

    torch::Tensor forward_dense_f32_down_kld(
            torch::Tensor xh) const {
        TORCH_CHECK(
            !is_moe && !tensor_parallel_dense_compatible() &&
            !geglu && swiglu_limit <= 0.0,
            "FP32-output IN diagnostic requires a local dense SiLU FFN");
        auto parts = g_profiler.measure(
            "ffn.gate_up", [&]() {
                return gate_up.forward(xh);
            });
        return g_profiler.measure(
            "ffn.down.fp32_output", [&]() {
                return down.forward_input_mul_f32_kld(
                    parts[1], parts[0], 2);
            });
    }

    torch::Tensor forward_impl(
        torch::Tensor x,
        c10::optional<torch::Tensor> input_ids,
        bool allow_important_neurons) const {
        torch::Tensor xh;
        if (x.scalar_type() == torch::kFloat16) {
            xh = x;
        } else {
            xh = g_profiler.measure("ffn.input_cast", [&]() { return x.to(torch::kFloat16); });
        }
        if (allow_important_neurons && important_neurons) {
            if (is_moe) {
                throw std::runtime_error(
                    "important-neuron branches are only supported for dense FFNs");
            }
            const int64_t rows =
                xh.numel() / xh.size(-1);
            const char * f32_down =
                std::getenv(
                    "MFQ_DIAGNOSTIC_IN_F32_DOWN");
            const bool use_f32_down =
                rows >= 16 &&
                g_kl_mmq_mode == KlMmqMode::Fp16 &&
                f32_down != nullptr &&
                f32_down[0] == '1';
            if (use_f32_down) {
                auto low =
                    forward_dense_f32_down_kld(xh);
                auto high =
                    important_neurons
                        ->forward_dense_f32_down_kld(xh);
                return g_profiler.measure(
                    "ffn.in.combine.fp32", [&]() {
                        return (low + high)
                            .to(torch::kFloat16)
                            .contiguous();
                    });
            }
            auto run_branch = [&](size_t index) {
                return index == 0
                    ? forward_impl(xh, input_ids, false)
                    : important_neurons->forward_impl(
                        xh, input_ids, false);
            };
            std::vector<torch::Tensor> outputs;
            const char * disable_parallel =
                std::getenv("MFQ_DISABLE_IN_BRANCH_PARALLEL");
            const bool parallel =
                decode_branch_parallel_enabled(rows) &&
                (disable_parallel == nullptr ||
                 disable_parallel[0] != '1') &&
                important_neuron_executor->run(
                    2, run_branch, outputs);
            auto low = parallel ? outputs[0] : run_branch(0);
            auto high = parallel ? outputs[1] : run_branch(1);
            return g_profiler.measure(
                "ffn.in.combine", [&]() {
                    return acc_cuda(
                        low.contiguous(), high.contiguous());
                });
        }
        if (tensor_parallel_dense_compatible()) {
            return g_profiler.measure(
                "ffn.tensor_parallel", [&]() {
                    return forward_tensor_parallel_dense(
                        xh);
                });
        }
        if (is_moe) {
            if (!shared || moe_top_k <= 0 || !moe_router.defined() ||
                (!moe_shared_ungated && !moe_shared_gate.defined())) {
                throw std::runtime_error("incomplete MoE FFN state");
            }
            auto xf = xh.reshape({-1, xh.size(-1)}).contiguous();
            auto xf32 = g_profiler.measure("moe.input_f32", [&]() {
                return xf.to(torch::kFloat32);
            });
            auto router_logits = g_profiler.measure("moe.router", [&]() {
                return torch::matmul(xf32, moe_router.transpose(0, 1));
            });
            torch::Tensor shared_gate_logits;
            std::vector<torch::Tensor> selected;
            if (moe_hash_ids.defined()) {
                if (!input_ids.has_value()) {
                    throw std::runtime_error(
                        "hash-routed MoE requires the current token ids");
                }
                selected = g_profiler.measure("moe.hash_route", [&]() {
                    auto ids = moe_hash_ids.index_select(
                        0, input_ids.value().reshape({-1})
                               .to(torch::kCUDA, torch::kInt64))
                        .to(torch::kInt32).contiguous();
                    auto weights = moe_sqrtsoftplus_weights_cuda(
                        router_logits.contiguous(), ids, 1e-20,
                        moe_router_scale);
                    return std::vector<torch::Tensor>{ids, weights};
                });
            } else {
                selected = g_profiler.measure("moe.topk", [&]() {
                    return moe_topk_cuda(
                        router_logits.contiguous(), moe_top_k,
                        moe_use_sigmoid, moe_use_sqrt_softplus,
                        moe_normalize, moe_delayed_softmax,
                        moe_router_bias.defined()
                            ? c10::optional<torch::Tensor>(moe_router_bias)
                            : c10::nullopt,
                        1e-20, moe_router_scale);
                });
            }
            auto route = g_profiler.measure("moe.route_map", [&]() {
                return build_moe_route_plan(
                    selected.at(0),
                    moe_split_gate_up ? moe_gate.n_experts : moe_gate_up.n_experts);
            });
            if (tensor_parallel_moe_compatible()) {
                auto routed = g_profiler.measure(
                    "moe.tensor_parallel", [&]() {
                        return forward_tensor_parallel_moe(
                            xf, route, selected.at(1));
                    });
                auto shared_output = shared->forward(xf);
                auto shared_half = shared_output
                    .reshape({xf.size(0), xf.size(1)})
                    .contiguous().to(torch::kFloat16);
                if (moe_shared_ungated) {
                    return g_profiler.measure("moe.combine", [&]() {
                        return acc_cuda(
                            routed.contiguous(), shared_half);
                    });
                }
                shared_gate_logits = g_profiler.measure(
                    "moe.shared_gate", [&]() {
                        return torch::matmul(
                            xf32,
                            moe_shared_gate.transpose(0, 1))
                            .contiguous();
                    });
                return g_profiler.measure("moe.combine", [&]() {
                    return moe_add_shared_gate_cuda(
                        routed.contiguous(), shared_half,
                        shared_gate_logits);
                });
            }
            std::optional<NintMoeWeight> staged_gate_up;
            std::optional<NintMoeWeight> staged_gate;
            std::optional<NintMoeWeight> staged_up;
            const NintMoeWeight * active_gate_up = &moe_gate_up;
            const NintMoeWeight * active_gate = &moe_gate;
            const NintMoeWeight * active_up = &moe_up;
            torch::Tensor gate_up_pair;
            if (moe_split_gate_up) {
                if (cpu_moe_gate) {
                    staged_gate.emplace(g_profiler.measure(
                        "moe.cpu_offload_gate_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_gate);
                        }));
                    active_gate = &staged_gate.value();
                }
                if (cpu_moe_up) {
                    staged_up.emplace(g_profiler.measure(
                        "moe.cpu_offload_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_up);
                        }));
                    active_up = &staged_up.value();
                }
                gate_up_pair = g_profiler.measure("moe.gate_up_split", [&]() {
                    auto gate = active_gate->forward(xf, route);
                    auto up = active_up->forward(xf, route);
                    return torch::cat({gate, up}, -1).contiguous();
                });
            } else {
                if (cpu_moe_gate_up) {
                    staged_gate_up.emplace(g_profiler.measure(
                        "moe.cpu_offload_gate_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_gate_up);
                        }));
                    active_gate_up = &staged_gate_up.value();
                }
                gate_up_pair = g_profiler.measure("moe.gate_up", [&]() {
                    return active_gate_up->forward(xf, route);
                });
            }
            staged_gate_up.reset();
            staged_gate.reset();
            staged_up.reset();
            moe_down.prefetch(route);
            std::optional<NintMoeWeight> staged_down;
            const NintMoeWeight * active_down = &moe_down;
            if (cpu_moe_down) {
                staged_down.emplace(g_profiler.measure(
                    "moe.cpu_offload_down_h2d", [&]() {
                        return stage_cpu_mixed_moe(cpu_moe_down);
                    }));
                active_down = &staged_down.value();
            }
            static const bool disable_swiglu_quant_fusion = [] {
                const char * value = std::getenv("MFQ_DISABLE_MOE_SWIGLU_QUANT_FUSION");
                return value != nullptr && std::atoi(value) != 0;
            }();
            torch::Tensor down_pair;
            const bool allow_swiglu_quant_fusion =
                !g_force_moe_materialized_swiglu &&
                !disable_swiglu_quant_fusion &&
                moe_small_hetero_enabled(
                    static_cast<int>(gate_up_pair.size(0)));
            if (swiglu_limit <= 0.0 && allow_swiglu_quant_fusion &&
                    active_down->hetero_supported) {
                down_pair = g_profiler.measure("moe.swiglu_down", [&]() {
                    return active_down->forward_swiglu(gate_up_pair, route);
                });
            } else if (swiglu_limit > 0.0 && allow_swiglu_quant_fusion &&
                    active_down->supports_clamped_swiglu()) {
                down_pair = g_profiler.measure("moe.swiglu_down", [&]() {
                    return active_down->forward_clamped_swiglu(
                        gate_up_pair, route, swiglu_limit);
                });
            } else {
                auto hidden = g_profiler.measure("moe.swiglu", [&]() {
                    if (swiglu_limit <= 0.0) {
                        return moe_swiglu_split_cuda(gate_up_pair);
                    }
                    const int64_t width = gate_up_pair.size(-1) / 2;
                    auto gate = torch::clamp_max(
                        gate_up_pair.slice(-1, 0, width).to(torch::kFloat32),
                        swiglu_limit);
                    auto up = torch::clamp(
                        gate_up_pair.slice(-1, width, 2 * width)
                            .to(torch::kFloat32),
                        -swiglu_limit, swiglu_limit);
                    return (torch::silu(gate) * up)
                        .to(torch::kFloat16).contiguous();
                });
                down_pair = g_profiler.measure("moe.down", [&]() {
                    return active_down->forward(hidden, route);
                });
            }
            record_moe_route_stats(
                moe_layer, selected.at(0), selected.at(1), down_pair,
                moe_split_gate_up ? moe_gate.n_experts : moe_gate_up.n_experts);
            staged_down.reset();
            static const bool disable_reduce_gate_fusion = [] {
                const char * value = std::getenv("MFQ_DISABLE_MOE_REDUCE_GATE_FUSION");
                return value != nullptr && std::atoi(value) != 0;
            }();
            const bool fuse_reduce_gate = !g_force_moe_unfused_reduce &&
                !disable_reduce_gate_fusion && !moe_shared_ungated &&
                down_pair.size(0) <= 8;
            torch::Tensor routed;
            if (!fuse_reduce_gate) {
                routed = g_profiler.measure("moe.reduce", [&]() {
                    return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                });
            }
            auto shared_output = shared->forward(xf);
            if (!moe_shared_ungated && !shared_gate_logits.defined()) {
                shared_gate_logits = g_profiler.measure("moe.shared_gate", [&]() {
                    return torch::matmul(xf32, moe_shared_gate.transpose(0, 1)).contiguous();
                });
            }
            auto shared_half = shared_output.reshape({xf.size(0), xf.size(1)}).contiguous().to(torch::kFloat16);
            if (moe_shared_ungated) {
                return g_profiler.measure("moe.combine", [&]() {
                    return acc_cuda(routed.contiguous(), shared_half);
                });
            }
            if (fuse_reduce_gate) {
                return g_profiler.measure("moe.reduce_combine", [&]() {
                    return moe_weighted_reduce_shared_gate_cuda(
                        down_pair, selected.at(1), shared_half, shared_gate_logits);
                });
            }
            return g_profiler.measure("moe.combine", [&]() {
                return moe_add_shared_gate_cuda(
                    routed.contiguous(), shared_half,
                    shared_gate_logits);
            });
        }
        if (geglu) {
            const char * disable_geglu = std::getenv("MFQ_DISABLE_FFN_GEGLU_FUSION");
            const bool geglu_fusion_enabled =
                disable_geglu == nullptr || disable_geglu[0] != '1';
            const char * enable_quant = std::getenv("MFQ_ENABLE_FFN_GATEUP_QUANT_FUSION");
            const bool quant_fusion_enabled = geglu_fusion_enabled &&
                (enable_quant == nullptr || enable_quant[0] == '1');
            const bool down_quant_layout_supported =
                down.is_nint() && !down.tensor_parallel() &&
                !down.nint.w.q8_zero &&
                down.nint.w.gs <= 32;
            if (gate_up.nint_grouped && down.is_nint() &&
                quant_fusion_enabled && down_quant_layout_supported &&
                !gate_up.nint.w.q8_zero &&
                xh.numel() / xh.size(-1) == 1 && gate_up.nint.split_w.empty() &&
                gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1] &&
                (gate_up.nint.w.bits == 2 || gate_up.nint.w.bits == 3 || gate_up.nint.w.bits == 4 || gate_up.nint.w.bits == 5 ||
                 gate_up.nint.w.bits == 6 || gate_up.nint.w.bits == 8)) {
                auto shape = xh.sizes().vec();
                shape.back() = down.nint.w.out;
                Workspace & gate_ws = gate_up.nint.w.workspace(1);
                Workspace & down_ws = down.nint.w.workspace(1);
                auto xf = pad_last(
                    xh.reshape({-1, xh.size(-1)}).contiguous(), gate_up.nint.w.neuron_len);
                g_profiler.measure("ffn.gate_up_geglu_quant", [&]() {
                    nint_ffn_gate_up_geglu_quant_ws_cuda(
                        gate_up.nint.w.q_packed, gate_up.nint.w.sub_scale,
                        gate_up.nint.w.sub_min, gate_up.nint.w.neuron_scale,
                        gate_up.nint.w.neuron_min, xf, gate_up.nint.w.gs,
                        gate_up.nint.w.bits, down.nint.w.gs,
                        gate_ws.qx, gate_ws.xscale, gate_ws.xsum,
                        down_ws.qx, down_ws.xscale, down_ws.xsum);
                    return down_ws.qx;
                });
                return g_profiler.measure("ffn.down_qx", [&]() {
                    return nint_matmul_qx(down.nint.w, down_ws).reshape(shape);
                });
            }
            if (geglu_fusion_enabled && xh.numel() / xh.size(-1) == 1 && gate_up.nint_grouped &&
                gate_up.nint.split_w.empty()) {
                auto act = g_profiler.measure("ffn.gate_up_geglu", [&]() {
                    return gate_up.forward_geglu(xh);
                });
                return g_profiler.measure("ffn.down", [&]() { return down.forward(act); });
            }
            auto parts = g_profiler.measure("ffn.gate_up", [&]() { return gate_up.forward(xh); });
            auto act = g_profiler.measure("ffn.geglu", [&]() {
                return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
            });
            return g_profiler.measure("ffn.down", [&]() { return down.forward(act); });
        }
        if (swiglu_limit > 0.0) {
            auto parts = g_profiler.measure("ffn.gate_up", [&]() {
                return gate_up.forward(xh);
            });
            auto act = g_profiler.measure("ffn.swiglu_clamped", [&]() {
                auto gate = torch::clamp_max(
                    parts[0].to(torch::kFloat32), swiglu_limit);
                auto up = torch::clamp(
                    parts[1].to(torch::kFloat32),
                    -swiglu_limit, swiglu_limit);
                return (torch::silu(gate) * up)
                    .to(torch::kFloat16).contiguous();
            });
            return g_profiler.measure("ffn.down", [&]() {
                return down.forward(act);
            });
        }
        if (nvq_fusion_enabled() && xh.numel() / xh.size(-1) == 1 &&
            gate_up.nvq_prefix2 && gate_up.layers.size() == 2 &&
            gate_up.layers[0].is_nvq() && gate_up.layers[1].is_nvq() && down.is_nvq() &&
            gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1] &&
            gate_up.outs[0] == down.nvq.w.neuron_len &&
            (down.nvq.w.gs == 24 || down.nvq.w.gs == 28 || down.nvq.w.gs == 32)) {
            auto shape = xh.sizes().vec();
            shape.back() = down.nvq.w.out;
            auto y = nvq_ffn_swiglu_down(
                gate_up.layers[0].nvq.w, gate_up.layers[1].nvq.w, down.nvq.w,
                xh.reshape({-1, xh.size(-1)}));
            return y.reshape(shape);
        }
        const char* enable_gateup_quant = std::getenv("MFQ_ENABLE_FFN_GATEUP_QUANT_FUSION");
            if (gate_up.nint_grouped && down.is_nint() &&
                !gate_up.nint.w.q8_zero && !down.nint.w.q8_zero &&
                (enable_gateup_quant != nullptr && enable_gateup_quant[0] == '1') &&
            xh.numel() / xh.size(-1) == 1 &&
            gate_up.nint.split_w.empty() && gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1] &&
            down.nint.w.gs <= 32 &&
            (gate_up.nint.w.bits == 2 || gate_up.nint.w.bits == 3 || gate_up.nint.w.bits == 4 || gate_up.nint.w.bits == 5 || gate_up.nint.w.bits == 6 || gate_up.nint.w.bits == 8) &&
            (down.nint.w.bits == 2 || down.nint.w.bits == 3 || down.nint.w.bits == 4 || down.nint.w.bits == 5 || down.nint.w.bits == 6 || down.nint.w.bits == 8)) {
            auto shape = xh.sizes().vec();
            shape.back() = down.nint.w.out;
            Workspace & gate_ws = gate_up.nint.w.workspace(1);
            Workspace & down_ws = down.nint.w.workspace(1);
            auto xf = xh.reshape({-1, xh.size(-1)}).contiguous();
            xf = pad_last(xf, gate_up.nint.w.neuron_len);
            g_profiler.measure("ffn.gate_up_quant", [&]() {
                nint_ffn_gate_up_swiglu_quant_ws_cuda(
                    gate_up.nint.w.q_packed, gate_up.nint.w.sub_scale, gate_up.nint.w.sub_min,
                    gate_up.nint.w.neuron_scale, gate_up.nint.w.neuron_min, xf,
                    gate_up.nint.w.gs, gate_up.nint.w.bits, down.nint.w.gs,
                    gate_ws.qx, gate_ws.xscale, gate_ws.xsum,
                    down_ws.qx, down_ws.xscale, down_ws.xsum);
                return down_ws.qx;
            });
            return g_profiler.measure("ffn.down_qx", [&]() {
                return nint_matmul_qx(down.nint.w, down_ws).reshape(shape);
            });
        }
        const char* disable_swiglu = std::getenv("MFQ_DISABLE_FFN_SWIGLU_FUSION");
        if ((disable_swiglu == nullptr || disable_swiglu[0] != '1') &&
            xh.numel() / xh.size(-1) == 1 &&
            gate_up.nint_grouped && gate_up.nint.split_w.empty() &&
            gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1]) {
            auto act = g_profiler.measure("ffn.gate_up_swiglu", [&]() { return gate_up.forward_swiglu(xh); });
            return g_profiler.measure("ffn.down", [&]() { return down.forward(act); });
        }
        auto parts = g_profiler.measure("ffn.gate_up", [&]() { return gate_up.forward(xh); });
        return g_profiler.measure("ffn.down", [&]() { return down.forward_input_mul(parts[1], parts[0], 2); });
    }

    torch::Tensor forward(
        torch::Tensor x,
        c10::optional<torch::Tensor> input_ids =
            c10::nullopt) const {
        return forward_impl(
            std::move(x), input_ids, true);
    }
};

struct KVCache {
    torch::Tensor k;
    torch::Tensor v;
    bool ring = false;
    KVCache() = default;
    KVCache(int64_t B, int64_t H, int64_t max_seq, int64_t D, bool use_ring = false)
        : ring(use_ring) {
        auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);
        k = torch::zeros({B, H, max_seq, D}, opts);
        v = torch::zeros({B, H, max_seq, D}, opts);
    }
    std::pair<torch::Tensor, torch::Tensor> append(
            torch::Tensor kk, torch::Tensor vv, torch::Tensor pos,
            int64_t start_pos, int64_t end_pos) {
        auto kh = kk.to(torch::kFloat16).contiguous();
        auto vh = vv.to(torch::kFloat16).contiguous();
        const char * aten_write_env = std::getenv("MFQ_KV_CACHE_WRITE_ATEN");
        const bool aten_write = aten_write_env != nullptr && aten_write_env[0] == '1';
        if (aten_write) {
            auto slots = ring ? torch::remainder(pos, k.size(2)) : pos;
            slots = slots.to(torch::kInt64).contiguous();
            k.index_copy_(2, slots, kh);
            v.index_copy_(2, slots, vh);
        } else {
            auto out = ring
                ? kv_cache_write_ring_positions_cuda(k, v, kh, vh, pos)
                : kv_cache_write_cuda(k, v, kh, vh, pos);
            (void)out;
        }
        if (ring) return {k, v};
        return {k.index({Slice(), Slice(), Slice(0, end_pos), Slice()}),
                v.index({Slice(), Slice(), Slice(0, end_pos), Slice()})};
    }
};

struct Dsv4RopeTable {
    torch::Tensor cos;
    torch::Tensor sin;
    torch::Tensor negative_sin;

    Dsv4RopeTable() = default;

    Dsv4RopeTable(const Config & c, int64_t compress_ratio) {
        constexpr int64_t rotary_dim = 64;
        const int64_t half = rotary_dim / 2;
        const int64_t positions = std::max<int64_t>(
            1, c.max_position_embeddings);
        const double base = compress_ratio > 0
            ? c.compress_rope_base : c.rope_base;
        auto opts = torch::TensorOptions()
            .device(torch::kCUDA).dtype(torch::kFloat32);
        auto dims = torch::arange(0, rotary_dim, 2, opts);
        auto freqs = torch::pow(
            torch::full({half}, base, opts),
            -dims / static_cast<double>(rotary_dim));
        if (compress_ratio > 0 && c.rope_original_positions > 0) {
            auto correction = [&](double rotations) {
                return static_cast<double>(rotary_dim) *
                    std::log(
                        static_cast<double>(c.rope_original_positions) /
                        (rotations * 6.28318530717958647692)) /
                    (2.0 * std::log(base));
            };
            const double low = std::max(
                0.0, std::floor(correction(c.rope_beta_fast)));
            const double high = std::min(
                static_cast<double>(rotary_dim - 1),
                std::ceil(correction(c.rope_beta_slow)));
            const double denominator =
                high == low ? 0.001 : high - low;
            auto ramp = torch::clamp(
                (torch::arange(half, opts) - low) / denominator,
                0.0, 1.0);
            auto smooth = 1.0 - ramp;
            freqs = freqs / c.rope_factor * (1.0 - smooth) +
                freqs * smooth;
        }
        auto angles = torch::arange(positions, opts).unsqueeze(1) *
            freqs.unsqueeze(0);
        cos = torch::cos(angles).contiguous();
        sin = torch::sin(angles).contiguous();
        negative_sin = (-sin).contiguous();
    }
};

static torch::Tensor dsv4_rotate_rope_tail(
    torch::Tensor x,
    torch::Tensor positions,
    const Dsv4RopeTable & table,
    bool inverse = false) {
    if (x.dim() != 4 || x.size(-1) < 64) {
        throw std::runtime_error(
            "DeepSeek V4 RoPE expects contiguous [B,H,T,D>=64]");
    }
    const int64_t width = x.size(-1);
    auto head = x.slice(-1, 0, width - 64).contiguous();
    auto tail = x.slice(-1, width - 64, width).contiguous();
    auto rotated = glm_interleaved_rope_cuda(
        tail, positions.contiguous().to(torch::kCUDA, torch::kInt64),
        table.cos, inverse ? table.negative_sin : table.sin, 64);
    return torch::cat({head, rotated}, -1).contiguous();
}

static std::vector<torch::Tensor> dsv4_hc_split_sinkhorn(
    torch::Tensor mixes,
    torch::Tensor scale,
    torch::Tensor base,
    int64_t hc_mult,
    int64_t iterations,
    double eps) {
    const int64_t prefix = 2 * hc_mult;
    auto pre = torch::sigmoid(
        mixes.slice(-1, 0, hc_mult) * scale.index({0}) +
        base.slice(0, 0, hc_mult)) + eps;
    auto post = 2.0 * torch::sigmoid(
        mixes.slice(-1, hc_mult, prefix) * scale.index({1}) +
        base.slice(0, hc_mult, prefix));
    auto comb = (
        mixes.slice(-1, prefix, mixes.size(-1)) * scale.index({2}) +
        base.slice(0, prefix, base.size(0)))
        .reshape({mixes.size(0), mixes.size(1), hc_mult, hc_mult});
    comb = torch::softmax(comb, -1) + eps;
    comb = comb / (comb.sum(-2, true) + eps);
    for (int64_t i = 1; i < iterations; ++i) {
        comb = comb / (comb.sum(-1, true) + eps);
        comb = comb / (comb.sum(-2, true) + eps);
    }
    return {pre, post, comb};
}

static bool g_dsv4_fused_hc = true;
static bool g_dsv4_compare_hc_ops = false;

static void dsv4_report_hc_difference(
    int layer,
    const char * stage,
    const char * tensor_name,
    const torch::Tensor & reference,
    const torch::Tensor & candidate)
{
    auto reference_f32 = reference.to(torch::kFloat32);
    auto candidate_f32 = candidate.to(torch::kFloat32);
    auto difference = candidate_f32 - reference_f32;
    const double reference_norm = std::max(
        reference_f32.norm().item<double>(), 1.0e-30);
    std::cout << std::scientific << std::setprecision(9)
              << "dsv4_hc_op_ab layer=" << layer
              << " stage=" << stage
              << " tensor=" << tensor_name
              << " differing="
              << candidate.ne(reference).sum().item<int64_t>()
              << " rel_l2="
              << difference.norm().item<double>() / reference_norm
              << " mean_abs=" << difference.abs().mean().item<double>()
              << " max_abs=" << difference.abs().max().item<double>()
              << "\n";
}

struct Dsv4SharedState {
    torch::Tensor attention_meta;
    torch::Tensor hadamard_signs;

    void ensure() {
        auto cuda = torch::TensorOptions().device(torch::kCUDA);
        if (!attention_meta.defined()) {
            attention_meta = torch::empty(
                {8 * 1024 * 1024}, cuda.dtype(torch::kFloat32));
        }
        if (!hadamard_signs.defined()) {
            hadamard_signs = torch::ones(
                {128}, cuda.dtype(torch::kInt8));
        }
    }
};

struct Block {
    int cuda_device = 0;
    virtual ~Block() = default;
    virtual void reset(int64_t B) = 0;
    virtual void set_token_ids(const torch::Tensor &) {}
    virtual torch::Tensor forward(torch::Tensor x, torch::Tensor pos, int64_t cache_pos,
                                  const c10::optional<torch::Tensor> & seq_len,
                                  const Config & c, const RopeCache & rope) = 0;
};

struct Dsv4PoolState {
    int64_t ratio = 0;
    int64_t head_dim = 0;
    bool overlap = false;
    int64_t cache_quant_mode = 0;
    int64_t capacity = 0;
    DenseLinearGroup projection;
    torch::Tensor ape;
    torch::Tensor norm;
    torch::Tensor state_kv;
    torch::Tensor state_gate;
    torch::Tensor previous_kv;
    torch::Tensor previous_gate;
    torch::Tensor pool;

    void reset(int64_t batch, int64_t max_positions) {
        if (ratio <= 0 || head_dim <= 0) return;
        capacity = std::max<int64_t>(
            1, (max_positions + ratio - 1) / ratio);
        const int64_t output_dim = overlap ? 2 * head_dim : head_dim;
        auto fp32 = torch::TensorOptions()
            .device(torch::kCUDA).dtype(torch::kFloat32);
        auto half = fp32.dtype(torch::kFloat16);
        if (state_kv.defined() && state_kv.size(0) == batch &&
            pool.size(1) == capacity) {
            state_kv.zero_();
            state_gate.fill_(-std::numeric_limits<float>::infinity());
            if (overlap) {
                previous_kv.zero_();
                previous_gate.fill_(
                    -std::numeric_limits<float>::infinity());
            }
            pool.zero_();
            return;
        }
        state_kv = torch::zeros({batch, ratio, output_dim}, fp32);
        state_gate = torch::full(
            {batch, ratio, output_dim},
            -std::numeric_limits<float>::infinity(), fp32);
        if (overlap) {
            previous_kv = torch::zeros(
                {batch, ratio, head_dim}, fp32);
            previous_gate = torch::full(
                {batch, ratio, head_dim},
                -std::numeric_limits<float>::infinity(), fp32);
        } else {
            previous_kv = torch::empty({0}, fp32);
            previous_gate = torch::empty({0}, fp32);
        }
        pool = torch::zeros({batch, capacity, head_dim}, half);
    }

    std::vector<torch::Tensor> project(
        torch::Tensor x, int64_t batch, int64_t tokens) const {
        auto parts = projection.forward(x.to(torch::kFloat32));
        const int64_t output_dim = overlap ? 2 * head_dim : head_dim;
        return {
            parts.at(0).reshape({batch, tokens, output_dim})
                .to(torch::kFloat32).contiguous(),
            parts.at(1).reshape({batch, tokens, output_dim})
                .to(torch::kFloat32).contiguous(),
        };
    }

    void update(
        torch::Tensor kv,
        torch::Tensor gate,
        torch::Tensor length,
        const Dsv4RopeTable & rope) {
        dsv4_decode_pool_update_cuda(
            kv, gate, ape, norm, state_kv, state_gate,
            previous_kv, previous_gate, pool, length,
            rope.cos, rope.sin, ratio, overlap,
            cache_quant_mode, 1e-6);
    }

    int64_t prefill(
        const torch::Tensor & kv,
        const torch::Tensor & gate,
        const Dsv4RopeTable & rope) {
        if (ratio <= 0) return 0;
        if (kv.dim() != 3 || gate.sizes() != kv.sizes()) {
            throw std::runtime_error(
                "DeepSeek V4 compressor prefill expects matching [B,T,D] tensors");
        }
        const int64_t batch = kv.size(0);
        const int64_t tokens = kv.size(1);
        const int64_t output_dim = overlap ? 2 * head_dim : head_dim;
        if (kv.size(2) != output_dim) {
            throw std::runtime_error(
                "DeepSeek V4 compressor prefill projection width mismatch");
        }
        const int64_t windows = tokens / ratio;
        const int64_t cutoff = windows * ratio;
        if (windows > 0) {
            auto grouped_kv = kv.narrow(1, 0, cutoff)
                .reshape({batch, windows, ratio, output_dim}).contiguous();
            auto grouped_gate = gate.narrow(1, 0, cutoff)
                .reshape({batch, windows, ratio, output_dim}).contiguous();
            auto positions = torch::arange(
                windows,
                torch::TensorOptions()
                    .device(kv.device()).dtype(torch::kInt64))
                .reshape({1, windows}).expand({batch, windows}).contiguous();
            auto empty = torch::empty({0}, kv.options());
            auto compressed = dsv4_compress_cuda(
                grouped_kv, grouped_gate, ape, norm, empty, empty,
                positions, rope.cos, rope.sin, ratio, overlap,
                cache_quant_mode, 1e-6);
            pool.narrow(1, 0, windows).copy_(compressed);

            state_kv.copy_(
                kv.narrow(1, cutoff - ratio, ratio).contiguous());
            state_gate.copy_(
                gate.narrow(1, cutoff - ratio, ratio).contiguous());
            if (overlap) {
                previous_kv.copy_(state_kv.narrow(2, 0, head_dim));
                previous_gate.copy_(state_gate.narrow(2, 0, head_dim));
            }
        }
        const int64_t remainder = tokens - cutoff;
        if (remainder > 0) {
            state_kv.narrow(1, 0, remainder).copy_(
                kv.narrow(1, cutoff, remainder));
            state_gate.narrow(1, 0, remainder).copy_(
                gate.narrow(1, cutoff, remainder));
        }
        return windows;
    }
};

struct Dsv4Block : Block {
    int layer = -1;
    int64_t max_positions = 0;
    int64_t compress_ratio = 0;
    int64_t hidden_size = 4096;
    int64_t heads = 64;
    int64_t head_dim = 512;
    int64_t groups = 8;
    int64_t o_rank = 1024;
    int64_t hc_mult = 4;
    int64_t hc_iterations = 20;
    double eps = 1e-6;
    double hc_eps = 1e-6;
    std::shared_ptr<Dsv4SharedState> shared_state;
    torch::Tensor current_ids;

    torch::Tensor attn_norm;
    torch::Tensor ffn_norm;
    torch::Tensor q_a_norm;
    torch::Tensor kv_norm;
    torch::Tensor sinks;
    torch::Tensor hc_attn_fn;
    torch::Tensor hc_attn_scale;
    torch::Tensor hc_attn_base;
    torch::Tensor hc_ffn_fn;
    torch::Tensor hc_ffn_scale;
    torch::Tensor hc_ffn_base;
    QuantLinear q_a;
    QuantLinear q_b;
    QuantLinear kv;
    QuantLinear output_a;
    QuantLinear output_b;
    Dsv4PoolState compressor;
    Dsv4PoolState indexer_compressor;
    QuantLinear indexer_q;
    torch::Tensor indexer_weight;
    FFN ffn;
    Dsv4RopeTable attention_rope;

    torch::Tensor local_cache;

    void reset(int64_t batch) override {
        auto half = torch::TensorOptions()
            .device(torch::kCUDA).dtype(torch::kFloat16);
        if (!local_cache.defined() || local_cache.size(0) != batch) {
            local_cache = torch::zeros(
                {batch, 128, head_dim}, half);
        } else {
            local_cache.zero_();
        }
        compressor.reset(batch, max_positions);
        indexer_compressor.reset(batch, max_positions);
        shared_state->ensure();
    }

    void set_token_ids(const torch::Tensor & ids) override {
        current_ids = ids;
    }

    std::vector<torch::Tensor> hc_pre(
        torch::Tensor x,
        torch::Tensor function,
        torch::Tensor scale,
        torch::Tensor base,
        const char * stage) const {
        auto flat = x.flatten(2).to(torch::kFloat32);
        auto inverse_rms = torch::rsqrt(
            flat.square().mean(-1, true) + eps);
        auto mixes = torch::matmul(
            flat, function.transpose(0, 1)) * inverse_rms;
        std::vector<torch::Tensor> candidate;
        if (g_dsv4_fused_hc || g_dsv4_compare_hc_ops) {
            candidate = dsv4_hc_pre_cuda(
                x, mixes.contiguous(), scale, base,
                hc_iterations, hc_eps);
        }
        std::vector<torch::Tensor> reference;
        if (!g_dsv4_fused_hc || g_dsv4_compare_hc_ops) {
            auto split = dsv4_hc_split_sinkhorn(
                mixes, scale, base, hc_mult, hc_iterations, hc_eps);
            auto reduced = (
                split.at(0).unsqueeze(-1) *
                flat.reshape(x.sizes()))
                .sum(2).to(torch::kFloat16).contiguous();
            reference = {reduced, split.at(1), split.at(2)};
        }
        if (g_dsv4_compare_hc_ops && layer == 0) {
            dsv4_report_hc_difference(
                layer, stage, "reduced", reference.at(0), candidate.at(0));
            dsv4_report_hc_difference(
                layer, stage, "post", reference.at(1), candidate.at(1));
            dsv4_report_hc_difference(
                layer, stage, "combination", reference.at(2), candidate.at(2));
        }
        return g_dsv4_fused_hc ? candidate : reference;
    }

    torch::Tensor hc_post(
        torch::Tensor x,
        torch::Tensor residual,
        torch::Tensor post,
        torch::Tensor combination,
        const char * stage) const {
        torch::Tensor candidate;
        if (g_dsv4_fused_hc || g_dsv4_compare_hc_ops) {
            candidate = dsv4_hc_post_cuda(
                x.contiguous(), residual.contiguous(),
                post.contiguous(), combination.contiguous());
        }
        torch::Tensor reference;
        if (!g_dsv4_fused_hc || g_dsv4_compare_hc_ops) {
            reference = (
                post.unsqueeze(-1) * x.unsqueeze(-2) +
                (combination.unsqueeze(-1) *
                 residual.to(torch::kFloat32).unsqueeze(-2)).sum(2))
                .to(torch::kFloat16).contiguous();
        }
        if (g_dsv4_compare_hc_ops && layer == 0) {
            dsv4_report_hc_difference(
                layer, stage, "expanded", reference, candidate);
        }
        return g_dsv4_fused_hc ? candidate : reference;
    }

    torch::Tensor output_projection(torch::Tensor attention) const {
        const int64_t batch = attention.size(0);
        const int64_t tokens = attention.size(1);
        const int64_t rows = batch * tokens;
        auto grouped = attention.contiguous()
            .reshape({rows, groups, heads / groups * head_dim})
            .to(torch::kFloat16);
        static const bool groupwise_enabled = [] {
            const char * value = std::getenv("MFQ_DSV4_GROUPWISE_OUTPUT_A");
            return value == nullptr || value[0] != '0';
        }();
        if (groupwise_enabled && output_a.is_nint() &&
                output_a.nint.w.bits == 8 && output_a.nint.w.gs == 48 &&
                output_a.nint.w.out == groups * o_rank) {
            auto low_rank = g_profiler.measure("dsv4.output_a", [&]() {
                return nint_matmul_groupwise_u8(
                    output_a.nint.w, grouped, groups);
            });
            return g_profiler.measure("dsv4.output_b", [&]() {
                return output_b.forward(low_rank)
                    .reshape({batch, tokens, hidden_size});
            });
        }
        if (groupwise_enabled && output_a.is_mxfp8() &&
                output_a.out() == groups * o_rank) {
            auto low_rank = g_profiler.measure(
                "dsv4.output_a", [&]() {
                    return output_a.forward_mxfp8_groupwise(
                        grouped, groups);
                });
            return g_profiler.measure("dsv4.output_b", [&]() {
                return output_b.forward(low_rank)
                    .reshape({batch, tokens, hidden_size});
            });
        }
        auto expanded = g_profiler.measure("dsv4.output_a", [&]() {
            return output_a.forward(
                grouped.reshape({rows * groups, grouped.size(-1)}))
                .reshape({rows, groups, groups, o_rank});
        });
        std::vector<torch::Tensor> diagonal;
        diagonal.reserve(static_cast<size_t>(groups));
        for (int64_t group = 0; group < groups; ++group) {
            diagonal.push_back(
                expanded.index({Slice(), group, group, Slice()}));
        }
        auto low_rank = torch::stack(diagonal, 1)
            .reshape({rows, groups * o_rank})
            .to(torch::kFloat16).contiguous();
        return g_profiler.measure("dsv4.output_b", [&]() {
            return output_b.forward(low_rank)
                .reshape({batch, tokens, hidden_size});
        });
    }

    torch::Tensor attention_forward(
        torch::Tensor x,
        torch::Tensor positions,
        int64_t cache_pos,
        const c10::optional<torch::Tensor> & seq_len) {
        const int64_t batch = x.size(0);
        const int64_t tokens = x.size(1);
        auto flat = x.reshape({batch * tokens, hidden_size})
            .to(torch::kFloat16).contiguous();

        auto qr = g_profiler.measure("dsv4.q_a", [&]() {
            return q_a.forward(flat);
        });
        qr = g_profiler.measure("dsv4.q_a_norm", [&]() {
            return rms_norm_cuda(
                qr.reshape({-1, qr.size(-1)}).to(torch::kFloat32),
                q_a_norm, eps).to(torch::kFloat16).contiguous();
        });
        auto queries = g_profiler.measure("dsv4.q_b", [&]() {
            return q_b.forward(qr)
                .reshape({batch, tokens, heads, head_dim})
                .transpose(1, 2).contiguous()
                .to(torch::kFloat32);
        });
        queries = g_profiler.measure("dsv4.q_norm_rope", [&]() {
            auto normalized = queries * torch::rsqrt(
                queries.square().mean(-1, true) + eps);
            return dsv4_rotate_rope_tail(
                normalized, positions, attention_rope, false);
        });

        auto values = g_profiler.measure("dsv4.kv", [&]() {
            return kv.forward(flat)
                .reshape({batch, tokens, head_dim});
        });
        values = g_profiler.measure("dsv4.kv_norm_rope", [&]() {
            auto normalized = rms_norm_cuda(
                values.reshape({-1, head_dim}).to(torch::kFloat32),
                kv_norm, eps).reshape({batch, tokens, head_dim})
                .to(torch::kFloat16);
            auto values_heads = normalized.unsqueeze(1).contiguous();
            values_heads = dsv4_rotate_rope_tail(
                values_heads, positions, attention_rope, false);
            return values_heads.squeeze(1).contiguous();
        });

        std::vector<torch::Tensor> compressor_parts;
        if (compress_ratio > 0) {
            compressor_parts = g_profiler.measure(
                "dsv4.compressor_proj", [&]() {
                    return compressor.project(flat, batch, tokens);
                });
        }
        std::vector<torch::Tensor> indexer_parts;
        if (compress_ratio == 4) {
            indexer_parts = g_profiler.measure(
                "dsv4.indexer_compressor_proj", [&]() {
                    return indexer_compressor.project(flat, batch, tokens);
                });
        }

        if (tokens > 1 && cache_pos == 0 && !seq_len.has_value()) {
            const int64_t local_tokens = std::min<int64_t>(tokens, 128);
            auto local_positions = positions.narrow(
                0, tokens - local_tokens, local_tokens)
                .remainder(128).to(torch::kInt64).contiguous();
            g_profiler.measure("dsv4.local_cache_prefill", [&]() {
                local_cache.index_copy_(
                    1, local_positions,
                    values.narrow(1, tokens - local_tokens, local_tokens));
                return 0;
            });

            int64_t visible = 0;
            if (compress_ratio > 0) {
                visible = g_profiler.measure(
                    "dsv4.compressor_prefill", [&]() {
                        return compressor.prefill(
                            compressor_parts.at(0), compressor_parts.at(1),
                            attention_rope);
                    });
            }
            if (compress_ratio == 4) {
                const int64_t index_visible = g_profiler.measure(
                    "dsv4.indexer_compressor_prefill", [&]() {
                        return indexer_compressor.prefill(
                            indexer_parts.at(0), indexer_parts.at(1),
                            attention_rope);
                    });
                if (index_visible != visible) {
                    throw std::runtime_error(
                        "DeepSeek V4 compressor/indexer prefill length mismatch");
                }
            }

            torch::Tensor selected;
            auto int_options = torch::TensorOptions()
                .device(x.device()).dtype(torch::kInt32);
            if (compress_ratio == 4 && visible > 512) {
                auto index_query = g_profiler.measure(
                    "dsv4.indexer_q_prefill", [&]() {
                        return indexer_q.forward(qr)
                            .reshape({batch, tokens, heads, 128})
                            .transpose(1, 2).contiguous();
                    });
                index_query = dsv4_rotate_rope_tail(
                    index_query, positions, attention_rope, false)
                    .transpose(1, 2).contiguous();
                index_query = nepq_hadamard_input_cuda(
                    index_query.reshape({batch * tokens * heads, 128})
                        .to(torch::kFloat16).contiguous(),
                    shared_state->hadamard_signs, 128)
                    .reshape({batch, tokens, heads, 128});
                index_query = dsv4_fp4_sim_cuda(index_query.contiguous());
                auto weights = g_profiler.measure(
                    "dsv4.indexer_weight_prefill", [&]() {
                        return torch::matmul(
                            x.reshape({batch * tokens, hidden_size})
                                .to(torch::kFloat32),
                            indexer_weight.transpose(0, 1))
                            .reshape({batch, tokens, heads})
                            .to(torch::kFloat16).contiguous();
                    });
                auto scores = g_profiler.measure(
                    "dsv4.indexer_scores_prefill", [&]() {
                        return dsv4_indexer_scores_cuda(
                            index_query,
                            indexer_compressor.pool
                                .narrow(1, 0, visible).contiguous(),
                            weights, 0, 4);
                    });
                selected = g_profiler.measure(
                    "dsv4.indexer_topk_prefill", [&]() {
                        return dsv4_topk512_cuda(scores);
                    });
            } else if (visible > 0) {
                selected = torch::arange(visible, int_options)
                    .reshape({1, 1, visible})
                    .expand({batch, tokens, visible}).contiguous();
            } else {
                selected = torch::zeros({batch, tokens, 1}, int_options);
            }

            const int64_t plan_ratio =
                compress_ratio > 0 ? compress_ratio : 1;
            auto plan = g_profiler.measure(
                "dsv4.attention_plan_prefill", [&]() {
                    return dsv4_build_prefill_plan_cuda(
                        selected, 0, 0, visible, plan_ratio, 128);
                });
            auto cache = g_profiler.measure(
                "dsv4.attention_cache_prefill", [&]() {
                    return visible > 0
                        ? torch::cat({
                            values,
                            compressor.pool.narrow(1, 0, visible)}, 1)
                            .contiguous()
                        : values.contiguous();
                });
            auto attention = g_profiler.measure(
                "dsv4.sparse_attention_prefill", [&]() {
                    return attention_dsv4_sparse_cuda(
                        queries, cache, plan.at(0), plan.at(1),
                        sinks, shared_state->attention_meta,
                        1.0 / std::sqrt(static_cast<double>(head_dim)));
                });
            attention = g_profiler.measure(
                "dsv4.attention_inverse_rope_prefill", [&]() {
                    auto transposed = attention.transpose(1, 2).contiguous();
                    transposed = dsv4_rotate_rope_tail(
                        transposed, positions, attention_rope, true);
                    return transposed.transpose(1, 2).contiguous();
                });
            return output_projection(attention);
        }

        std::vector<torch::Tensor> outputs;
        outputs.reserve(static_cast<size_t>(tokens));
        for (int64_t token = 0; token < tokens; ++token) {
            const int64_t absolute_length = cache_pos + token + 1;
            auto position = positions.narrow(0, token, 1);
            auto slot = torch::remainder(position, 128)
                .to(torch::kInt64).contiguous();
            g_profiler.measure("dsv4.local_cache_write", [&]() {
                glm_dsa_cache_write_cuda(
                    local_cache,
                    values.narrow(1, token, 1).contiguous(),
                    slot);
                return 0;
            });

            torch::Tensor length;
            if (tokens == 1 && seq_len.has_value()) {
                length = seq_len.value().contiguous();
            } else {
                length = torch::full(
                    {batch}, absolute_length,
                    torch::TensorOptions()
                        .device(torch::kCUDA).dtype(torch::kInt64));
            }
            if (compress_ratio > 0) {
                g_profiler.measure("dsv4.compressor_update", [&]() {
                    compressor.update(
                        compressor_parts.at(0).narrow(1, token, 1)
                            .contiguous(),
                        compressor_parts.at(1).narrow(1, token, 1)
                            .contiguous(),
                        length, attention_rope);
                    return 0;
                });
            }
            if (compress_ratio == 4) {
                g_profiler.measure(
                    "dsv4.indexer_compressor_update", [&]() {
                        indexer_compressor.update(
                            indexer_parts.at(0).narrow(1, token, 1)
                                .contiguous(),
                            indexer_parts.at(1).narrow(1, token, 1)
                                .contiguous(),
                            length, attention_rope);
                        return 0;
                    });
            }

            const int64_t visible = compress_ratio > 0
                ? absolute_length / compress_ratio : 0;
            torch::Tensor selected;
            auto int_options = torch::TensorOptions()
                .device(torch::kCUDA).dtype(torch::kInt32);
            if (compress_ratio == 4 && visible > 512) {
                auto qr_token = qr.reshape(
                    {batch, tokens, qr.size(-1)})
                    .narrow(1, token, 1)
                    .reshape({batch, qr.size(-1)})
                    .contiguous();
                auto index_query = g_profiler.measure(
                    "dsv4.indexer_q", [&]() {
                        return indexer_q.forward(qr_token)
                            .reshape({batch, 1, heads, 128})
                            .transpose(1, 2).contiguous();
                    });
                index_query = dsv4_rotate_rope_tail(
                    index_query, position, attention_rope, false)
                    .transpose(1, 2).contiguous();
                index_query = nepq_hadamard_input_cuda(
                    index_query.reshape({batch * heads, 128})
                        .to(torch::kFloat16).contiguous(),
                    shared_state->hadamard_signs, 128)
                    .reshape({batch, 1, heads, 128});
                index_query = dsv4_fp4_sim_cuda(
                    index_query.contiguous());
                auto weights = g_profiler.measure(
                    "dsv4.indexer_weight", [&]() {
                        return torch::matmul(
                            x.narrow(1, token, 1)
                                .reshape({batch, hidden_size})
                                .to(torch::kFloat32),
                            indexer_weight.transpose(0, 1))
                            .reshape({batch, 1, heads})
                            .to(torch::kFloat16).contiguous();
                    });
                auto scores = g_profiler.measure(
                    "dsv4.indexer_scores", [&]() {
                        return dsv4_indexer_scores_cuda(
                            index_query,
                            indexer_compressor.pool
                                .narrow(1, 0, visible).contiguous(),
                            weights, cache_pos + token, 4);
                    });
                selected = g_profiler.measure(
                    "dsv4.indexer_topk", [&]() {
                        return dsv4_topk512_cuda(scores);
                    });
            } else {
                selected = torch::arange(visible, int_options)
                    .reshape({1, 1, visible})
                    .expand({batch, 1, visible}).contiguous();
            }

            const int64_t plan_ratio =
                compress_ratio > 0 ? compress_ratio : 1;
            auto plan = g_profiler.measure("dsv4.attention_plan", [&]() {
                return dsv4_build_decode_plan_cuda(
                    selected, length, visible, plan_ratio, 128);
            });
            auto cache = g_profiler.measure("dsv4.attention_cache", [&]() {
                return visible > 0
                    ? torch::cat({
                        local_cache,
                        compressor.pool.narrow(1, 0, visible)}, 1)
                        .contiguous()
                    : local_cache;
            });
            auto query = queries.narrow(2, token, 1).contiguous();
            auto attention = g_profiler.measure(
                "dsv4.sparse_attention", [&]() {
                    return attention_dsv4_sparse_cuda(
                        query, cache, plan.at(0), plan.at(1),
                        sinks, shared_state->attention_meta,
                        1.0 / std::sqrt(static_cast<double>(head_dim)));
                });
            attention = g_profiler.measure(
                "dsv4.attention_inverse_rope", [&]() {
                    auto transposed = attention.transpose(1, 2).contiguous();
                    transposed = dsv4_rotate_rope_tail(
                        transposed, position, attention_rope, true);
                    return transposed.transpose(1, 2).contiguous();
                });
            outputs.push_back(output_projection(attention));
        }
        return torch::cat(outputs, 1);
    }

    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor pos,
        int64_t cache_pos,
        const c10::optional<torch::Tensor> & seq_len,
        const Config &,
        const RopeCache &) override {
        if (!current_ids.defined()) {
            throw std::runtime_error(
                "DeepSeek V4 block did not receive token ids");
        }
        const int64_t batch = x.size(0);
        const int64_t tokens = x.size(1);
        auto residual = x;
        auto pre = g_profiler.measure("dsv4.hc_attn_pre", [&]() {
            return hc_pre(
                x, hc_attn_fn, hc_attn_scale, hc_attn_base, "attn_pre");
        });
        auto normalized = g_profiler.measure("dsv4.attn_norm", [&]() {
            return rms_norm_cuda(
                pre.at(0).reshape({batch * tokens, hidden_size})
                    .to(torch::kFloat32),
                attn_norm, eps)
                .reshape({batch, tokens, hidden_size})
                .to(torch::kFloat16);
        });
        auto attention = attention_forward(
            normalized, pos, cache_pos, seq_len);
        x = g_profiler.measure("dsv4.hc_attn_post", [&]() {
            return hc_post(
                attention, residual, pre.at(1), pre.at(2), "attn_post");
        });

        residual = x;
        pre = g_profiler.measure("dsv4.hc_ffn_pre", [&]() {
            return hc_pre(
                x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, "ffn_pre");
        });
        normalized = g_profiler.measure("dsv4.ffn_norm", [&]() {
            return rms_norm_cuda(
                pre.at(0).reshape({batch * tokens, hidden_size})
                    .to(torch::kFloat32),
                ffn_norm, eps)
                .reshape({batch, tokens, hidden_size})
                .to(torch::kFloat16);
        });
        auto feed_forward = ffn.forward(
            normalized.reshape({batch * tokens, hidden_size}),
            current_ids);
        feed_forward = feed_forward.reshape(
            {batch, tokens, hidden_size});
        return g_profiler.measure("dsv4.hc_ffn_post", [&]() {
            return hc_post(
                feed_forward, residual, pre.at(1), pre.at(2), "ffn_post");
        });
    }
};

struct GlmDsaSharedState {
    torch::Tensor topk_indices;
    int64_t dense_prefix_rows = 0;
    std::unordered_map<int64_t, MoeRoutePlan> head_routes;
    MoeRoutePlan transient_head_route;
    int64_t transient_head_route_key = -1;
    torch::Tensor dense_mask;
    torch::Tensor decode_mask;
    torch::Tensor kv_max;
    torch::Tensor attention_meta;

    void reset() {
        topk_indices = torch::Tensor();
        dense_prefix_rows = 0;
    }

    const MoeRoutePlan & head_route(int64_t rows, int heads) {
        const int64_t key = rows * 4096 + heads;
        if (rows != heads) {
            if (transient_head_route_key != key) {
                auto options = torch::TensorOptions()
                    .device(torch::kCUDA).dtype(torch::kInt32);
                auto ids = torch::remainder(torch::arange(rows, options), heads)
                    .reshape({rows, 1}).contiguous();
                transient_head_route = build_moe_route_plan(ids, heads);
                transient_head_route_key = key;
            }
            return transient_head_route;
        }
        auto found = head_routes.find(key);
        if (found != head_routes.end()) return found->second;
        auto options = torch::TensorOptions()
            .device(torch::kCUDA).dtype(torch::kInt32);
        auto ids = torch::remainder(torch::arange(rows, options), heads)
            .reshape({rows, 1}).contiguous();
        return head_routes.emplace(
            key, build_moe_route_plan(ids, heads)).first->second;
    }

    void ensure_meta() {
        constexpr int64_t kMetaFloats = 8 * 1024 * 1024;
        auto cuda = torch::TensorOptions().device(torch::kCUDA);
        if (!attention_meta.defined() || attention_meta.numel() < kMetaFloats) {
            attention_meta = torch::empty(
                {kMetaFloats}, cuda.dtype(torch::kFloat32));
        }
    }

    void ensure_dense_workspace(int64_t B, int64_t M, int64_t logical_len) {
        const int64_t stride = (logical_len + 63) / 64 * 64;
        auto cuda = torch::TensorOptions().device(torch::kCUDA);
        if (!dense_mask.defined() || dense_mask.size(0) < M ||
            dense_mask.size(1) < stride) {
            dense_mask = torch::empty(
                {M, stride}, cuda.dtype(torch::kFloat16));
        }
        if (!kv_max.defined() || kv_max.numel() < B * M) {
            kv_max = torch::empty({B * M}, cuda.dtype(torch::kInt32));
        }
        ensure_meta();
    }

    void ensure_decode_workspace(int64_t B, int64_t planned_len) {
        const int64_t stride = (planned_len + 63) / 64 * 64;
        auto cuda = torch::TensorOptions().device(torch::kCUDA);
        if (!decode_mask.defined() || decode_mask.size(0) != B ||
            decode_mask.size(1) < stride) {
            decode_mask = torch::empty(
                {B, stride}, cuda.dtype(torch::kFloat16));
        }
        if (!kv_max.defined() || kv_max.numel() < B) {
            kv_max = torch::empty({B}, cuda.dtype(torch::kInt32));
        }
        ensure_meta();
    }
};

struct FullBlock : Block {
    int layer = -1;
    bool gemma4 = false;
    bool gemma4_moe = false;
    bool sliding = false;
    bool value_equals_key = false;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t attention_head_dim = 0;
    int64_t attention_rotary_dim = 0;
    int64_t attention_window = 0;
    double attention_scale = 0.0;
    RopeCache attention_rope;
    torch::Tensor attn_norm, ffn_norm, q_norm, k_norm;
    torch::Tensor v_norm, attn_post_norm;
    torch::Tensor ffn_post_norm, ffn_post_norm_1, ffn_pre_norm_2, ffn_post_norm_2;
    torch::Tensor layer_scale;
    QuantLinearGroup qkv;
    QuantLinear o;
    FFN ffn;
    NintMoeWeight gemma_moe_gate_up;
    NintMoeWeight gemma_moe_down;
    torch::Tensor gemma_router;
    torch::Tensor gemma_router_norm_scale;
    torch::Tensor gemma_expert_scale;
    int gemma_top_k = 0;
    KVCache cache;
    torch::Tensor decode_partial_o, decode_partial_m, decode_partial_l;
    torch::Tensor decode_llama_mask, decode_llama_kv_max, decode_llama_meta;

    static constexpr int64_t kDecodeAttentionMaxParts = 16;

    void reset(int64_t B) override {
        if (cache.k.defined() && cache.k.size(0) == B) return;
        cache = KVCache();
        decode_partial_o = torch::Tensor();
        decode_partial_m = torch::Tensor();
        decode_partial_l = torch::Tensor();
        decode_llama_mask = torch::Tensor();
        decode_llama_kv_max = torch::Tensor();
        decode_llama_meta = torch::Tensor();
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor pos, int64_t cache_pos,
                          const c10::optional<torch::Tensor> & seq_len,
                          const Config & c, const RopeCache & rope) override {
        int64_t B = x.size(0), T = x.size(1), H = x.size(2);
        const int64_t nh = attention_heads > 0 ? attention_heads : c.num_attention_heads;
        const int64_t nkh = kv_heads > 0 ? kv_heads : c.num_key_value_heads;
        const int64_t hd = attention_head_dim > 0 ? attention_head_dim : c.head_dim;
        const int64_t attn_width = nh * hd;
        const int64_t cache_capacity = sliding
            ? attention_window
            : (g_kl_kv_cache_capacity > 0
                   ? std::max<int64_t>(
                         cache_pos + T, g_kl_kv_cache_capacity)
                   : c.max_position_embeddings);
        const RopeCache & active_rope = attention_rope.cos.defined() ? attention_rope : rope;
        if (!cache.k.defined() || cache.k.numel() == 0) {
            cache = KVCache(B, nkh, cache_capacity, hd, sliding);
            const int64_t total = B * nh;
            auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
            decode_partial_o = torch::empty({total, kDecodeAttentionMaxParts, hd}, opts);
            decode_partial_m = torch::empty({total, kDecodeAttentionMaxParts}, opts);
            decode_partial_l = torch::empty({total, kDecodeAttentionMaxParts}, opts);
        }
        auto residual = x;
        auto xn = g_profiler.measure("full.attn_norm", [&]() {
            return qwen_rms_norm(x.reshape({B * T, H}).to(torch::kFloat32), attn_norm, c).reshape({B, T, H});
        });
        auto parts = g_profiler.measure("full.qkv", [&]() { return qkv.forward(xn); });
        if (parts.size() != (value_equals_key ? 2u : 3u)) {
            throw std::runtime_error("attention projection group has the wrong output count");
        }
        auto q_full = parts[0], k_full = parts[1];
        auto v_full = value_equals_key ? k_full : parts[2];
        torch::Tensor q_raw, q_gate;
        g_profiler.measure("full.qkv_view", [&]() {
            if (c.qwen35_attn_q_gate) {
            auto qp = q_full.reshape({B, T, nh, hd * 2});
            auto chunks = qp.chunk(2, -1);
            q_raw = chunks[0];
            q_gate = chunks[1];
            } else {
            q_raw = q_full.reshape({B, T, nh, hd});
            }
            return q_raw;
        });
        auto q = g_profiler.measure("full.q_view", [&]() { return q_raw.transpose(1, 2).contiguous(); });
        auto k = g_profiler.measure("full.k_view", [&]() { return k_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous(); });
        auto v = g_profiler.measure("full.v_view", [&]() { return v_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous(); });
        if (q_norm.defined()) q = g_profiler.measure("full.q_norm", [&]() { return qwen_rms_norm(q.reshape({-1, hd}).to(torch::kFloat32), q_norm, c).reshape_as(q); });
        if (k_norm.defined()) k = g_profiler.measure("full.k_norm", [&]() { return qwen_rms_norm(k.reshape({-1, hd}).to(torch::kFloat32), k_norm, c).reshape_as(k); });
        if (v_norm.defined()) v = g_profiler.measure("full.v_norm", [&]() { return qwen_rms_norm(v.reshape({-1, hd}).to(torch::kFloat32), v_norm, c).reshape_as(v); });
        q = g_profiler.measure("full.q_rope", [&]() { return active_rope.apply(q, pos, c); });
        k = g_profiler.measure("full.k_rope", [&]() { return active_rope.apply(k, pos, c); });
        auto kv = g_profiler.measure("full.kv_write", [&]() { return cache.append(k, v, pos, cache_pos, cache_pos + T); });
        double attn_scale = attention_scale > 0.0 ? attention_scale : 1.0 / std::sqrt((double)hd);
        torch::Tensor a;
        bool attention_token_major = false;
        a = g_profiler.measure("full.attention", [&]() {
            auto qh = q.to(torch::kFloat16).contiguous();
            auto kh = k.to(torch::kFloat16).contiguous();
            auto vh = v.to(torch::kFloat16).contiguous();
            if (cache_pos == 0 && T > 1) {
                const char * llama_flash_env = std::getenv("MFQ_LLAMA_FLASH256");
                const bool llama_flash_enabled =
                    llama_flash_env == nullptr || llama_flash_env[0] != '0';
                if (!sliding && T % 256 == 0 && hd == 512 && nh == 8 * nkh &&
                    llama_flash_enabled) {
                    a = attention_llama_flash512_cuda(
                        q.to(torch::kFloat32).contiguous(), kh, vh, attn_scale);
                    attention_token_major = true;
                } else if (sliding && T >= 32 && hd == 256 && nh == 2 * nkh &&
                    llama_flash_enabled) {
                    a = attention_llama_flash256_swa_cuda(
                        q.to(torch::kFloat32).contiguous(), kh, vh,
                        attn_scale, attention_window);
                    attention_token_major = true;
                } else if (!sliding && T >= 32 && hd == 256 && nh == 4 * nkh &&
                           llama_flash_enabled) {
                    a = attention_llama_flash256_cuda(
                        q.to(torch::kFloat32).contiguous(), kh, vh, attn_scale);
                    attention_token_major = true;
                } else if (sliding) {
                    a = attention_swa_cuda(qh, kh, vh, attn_scale, attention_window);
                } else {
                    a = at::scaled_dot_product_attention(
                        qh, kh, vh, std::nullopt, 0.0, true, attn_scale, true);
                }
            } else if (seq_len.has_value()) {
                const int64_t planned_len = g_decode_graph_attention_kv_len > 0
                    ? g_decode_graph_attention_kv_len : cache_pos + T;
                const char * aten_decode_env = std::getenv("MFQ_ATTENTION_DECODE_ATEN");
                const bool aten_decode_enabled =
                    aten_decode_env != nullptr && aten_decode_env[0] == '1';
                const char * llama_decode_env = std::getenv("MFQ_LLAMA_FLASH_DECODE");
                const bool llama_decode_enabled =
                    llama_decode_env == nullptr || llama_decode_env[0] != '0';
                auto prepare_llama_decode_workspace = [&](int64_t visible_len, int64_t kv_tile) {
                    const int64_t mask_stride = (visible_len + kv_tile - 1) / kv_tile * kv_tile;
                    const int64_t ntiles_kv = (visible_len + kv_tile - 1) / kv_tile;
                    const int64_t max_blocks = B * nkh * ntiles_kv;
                    const int64_t meta_float2 = max_blocks * 8 * (2 + hd / 2);
                    auto cuda = torch::TensorOptions().device(torch::kCUDA);
                    if (!decode_llama_mask.defined() || decode_llama_mask.size(0) != B ||
                        decode_llama_mask.size(1) < mask_stride) {
                        decode_llama_mask = torch::empty(
                            {B, mask_stride}, cuda.dtype(torch::kFloat16));
                    }
                    if (!decode_llama_kv_max.defined() || decode_llama_kv_max.numel() < B) {
                        decode_llama_kv_max = torch::empty({B}, cuda.dtype(torch::kInt32));
                    }
                    if (!decode_llama_meta.defined() || decode_llama_meta.numel() < 2 * meta_float2) {
                        decode_llama_meta = torch::empty(
                            {2 * meta_float2}, cuda.dtype(torch::kFloat32));
                    }
                };
                if (aten_decode_enabled) {
                    const int64_t visible_len = sliding
                        ? std::min<int64_t>(attention_window, cache_pos + T)
                        : cache_pos + T;
                    auto cached_k = cache.k.index({
                        Slice(), Slice(), Slice(0, visible_len), Slice()}).contiguous();
                    auto cached_v = cache.v.index({
                        Slice(), Slice(), Slice(0, visible_len), Slice()}).contiguous();
                    a = at::scaled_dot_product_attention(
                        qh, cached_k, cached_v, std::nullopt,
                        0.0, false, attn_scale, true);
                } else if (sliding && T == 1 && llama_decode_enabled && hd == 256 && nh == 2 * nkh) {
                    const int64_t visible_len = std::min<int64_t>(attention_window, planned_len);
                    prepare_llama_decode_workspace(visible_len, 64);
                    a = attention_llama_flash256_swa_decode_cuda(
                        q.to(torch::kFloat32).contiguous(), cache.k, cache.v,
                        seq_len.value(), attn_scale, visible_len,
                        decode_llama_mask, decode_llama_kv_max, decode_llama_meta);
                    attention_token_major = true;
                } else if (!sliding && T == 1 && llama_decode_enabled &&
                           nh == 8 * nkh && (hd == 256 || hd == 512)) {
                    const int64_t kv_tile = hd == 512 ? 32 : 64;
                    prepare_llama_decode_workspace(planned_len, kv_tile);
                    a = hd == 512
                        ? attention_llama_flash512_decode_cuda(
                            q.to(torch::kFloat32).contiguous(), cache.k, cache.v,
                            seq_len.value(), attn_scale, planned_len,
                            decode_llama_mask, decode_llama_kv_max, decode_llama_meta)
                        : attention_llama_flash256_decode_cuda(
                            q.to(torch::kFloat32).contiguous(), cache.k, cache.v,
                            seq_len.value(), attn_scale, planned_len,
                            decode_llama_mask, decode_llama_kv_max, decode_llama_meta);
                    attention_token_major = true;
                } else if (sliding) {
                    a = attention_cache_swa_planned_cuda(
                        qh, cache.k, cache.v, seq_len.value(),
                        attn_scale, attention_window, planned_len);
                } else {
                        const char * split_env = std::getenv("MFQ_ATTENTION_DECODE_SPLITK");
                        const bool split_enabled =
                            split_env == nullptr || split_env[0] != '0';
                        int64_t parts = split_enabled && cache_pos >= 192
                            ? (cache_pos + 127) / 128 : 1;
                        if (split_enabled && g_decode_graph_attention_parts > 0) {
                            parts = g_decode_graph_attention_parts;
                        }
                        parts = std::min<int64_t>(parts, kDecodeAttentionMaxParts);
                        a = parts > 1
                            ? attention_cache_decode_split_cuda(
                                qh, cache.k, cache.v, seq_len.value(), attn_scale,
                                decode_partial_o, decode_partial_m, decode_partial_l, parts)
                            : attention_cache_decode_cuda(
                                qh, cache.k, cache.v, seq_len.value(), attn_scale);
                }
            } else {
                a = sliding
                    ? attention_swa_cuda(qh, kv.first.contiguous(), kv.second.contiguous(),
                                         attn_scale, attention_window)
                    : attention_cuda(qh, kv.first.contiguous(), kv.second.contiguous(),
                                     attn_scale, true);
            }
            return a;
        });
        torch::Tensor oo;
        if (q_gate.defined()) {
            auto af = g_profiler.measure("full.attn_out_view", [&]() {
                return attention_token_major ? a.reshape({B, T, attn_width}) :
                    a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
            });
            auto gf = g_profiler.measure("full.q_gate_view", [&]() { return q_gate.contiguous().reshape({B, T, attn_width}); });
            oo = g_profiler.measure("full.o_proj_gate", [&]() { return o.forward_input_mul(af, gf, 1); });
        } else {
            auto af = g_profiler.measure("full.attn_out_view", [&]() {
                return attention_token_major ? a.reshape({B, T, attn_width}) :
                    a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
            });
            oo = g_profiler.measure("full.o_proj", [&]() { return o.forward(af); });
        }
        if (gemma4) {
            trace_gemma_stage(layer, "attention_output", oo);
            const bool fused_norms = gemma4_moe &&
                gemma4_fused_norms_enabled() &&
                g_gemma_stage_trace == nullptr && layer_scale.defined();
            torch::Tensor dense_input;
            torch::Tensor router_input;
            torch::Tensor moe_input;
            if (fused_norms) {
                auto prepared = g_profiler.measure("gemma.attn_residual_pre_norms", [&]() {
                    return gemma4_attn_residual_pre_norms_f16_cuda(
                        residual.reshape({B * T, H}), oo.reshape({B * T, H}),
                        attn_post_norm, ffn_norm, gemma_router_norm_scale,
                        ffn_pre_norm_2, c.rms_norm_eps);
                });
                x = prepared[0].reshape({B, T, H});
                residual = x;
                dense_input = prepared[1];
                router_input = prepared[2];
                moe_input = prepared[3];
            } else {
                auto attn_post = g_profiler.measure("gemma.attn_post_norm", [&]() {
                    return gemma_rms_norm_f16(
                        oo.reshape({B * T, H}), attn_post_norm, c);
                });
                x = g_profiler.measure("gemma.attn_residual", [&]() {
                    return acc_cuda(residual.reshape({B * T, H}), attn_post).reshape({B, T, H});
                });
                trace_gemma_stage(layer, "attention_residual", x);
                residual = x;
                dense_input = g_profiler.measure("gemma.ffn_pre_norm", [&]() {
                    return gemma_rms_norm_f16(
                        x.reshape({B * T, H}), ffn_norm, c);
                });
                if (gemma4_moe) {
                    router_input = g_profiler.measure("gemma.router_norm", [&]() {
                        return qwen_rms_norm(
                            x.reshape({B * T, H}).to(torch::kFloat32),
                            gemma_router_norm_scale, c);
                    });
                    moe_input = g_profiler.measure("gemma.ffn_pre_norm_2", [&]() {
                        return gemma_rms_norm_f16(
                            x.reshape({B * T, H}), ffn_pre_norm_2, c);
                    });
                }
            }
            auto dense_output = g_profiler.measure("gemma.ffn_dense", [&]() {
                return ffn.forward(dense_input).reshape({B * T, H});
            });
            if (!gemma4_moe) {
                auto dense_post = g_profiler.measure("gemma.ffn_post_norm", [&]() {
                    return dense_output.scalar_type() == torch::kFloat16
                        ? gemma_rms_norm_f16(dense_output, ffn_post_norm, c)
                        : qwen_rms_norm(
                            dense_output.to(torch::kFloat32),
                            ffn_post_norm, c)
                            .to(torch::kFloat16)
                            .contiguous();
                });
                auto result = g_profiler.measure("gemma.ffn_residual", [&]() {
                    return acc_cuda(
                        residual.reshape({B * T, H}), dense_post)
                        .reshape({B, T, H});
                });
                if (layer_scale.defined()) {
                    result = g_profiler.measure("gemma.layer_scale", [&]() {
                        return result * layer_scale;
                    });
                }
                trace_gemma_stage(layer, "layer_output", result);
                return result;
            }
            if (!fused_norms) {
                dense_output = g_profiler.measure("gemma.ffn_post_norm_1", [&]() {
                    return gemma_rms_norm_f16(
                        dense_output, ffn_post_norm_1, c);
                });
                trace_gemma_stage(layer, "dense_output", dense_output);
            }
            auto router_logits = g_profiler.measure("gemma.router", [&]() {
                return torch::matmul(router_input, gemma_router.transpose(0, 1));
            });
            auto selected = g_profiler.measure("gemma.topk", [&]() {
                return moe_topk_cuda(
                    router_logits.contiguous(), gemma_top_k,
                    false, false, false, true, c10::nullopt, 1e-20, 1.0);
            });
            trace_gemma_stage(layer, "route_ids", selected.at(0));
            trace_gemma_stage(layer, "route_weights_before_scale", selected.at(1));
            g_profiler.measure("gemma.route_scale", [&]() {
                return moe_apply_expert_scale_cuda(
                    selected.at(1), selected.at(0), gemma_expert_scale);
            });
            trace_gemma_stage(layer, "route_weights", selected.at(1));
            auto route = g_profiler.measure("gemma.route_map", [&]() {
                return build_moe_route_plan(selected.at(0), gemma_moe_gate_up.n_experts);
            });
            torch::Tensor down_pair;
            const bool tracing_layer =
                g_gemma_stage_trace != nullptr && layer == g_gemma_trace_layer;
            if (!tracing_layer && moe_input.dim() == 2 && moe_input.size(0) <= 4 &&
                    gemma_moe_gate_up.hetero_supported) {
                auto moe_hidden = g_profiler.measure("gemma.moe_gate_up_geglu", [&]() {
                    return gemma_moe_gate_up.forward_glu_output(moe_input, route, true);
                });
                gemma_moe_down.prefetch(route);
                down_pair = g_profiler.measure("gemma.moe_down", [&]() {
                    return gemma_moe_down.forward(moe_hidden, route);
                });
            } else {
                auto gate_up_pair = g_profiler.measure("gemma.moe_gate_up", [&]() {
                    return gemma_moe_gate_up.forward(moe_input, route);
                });
                gemma_moe_down.prefetch(route);
                trace_gemma_stage(layer, "moe_gate_up", gate_up_pair);
                if (tracing_layer || gate_up_pair.size(0) > 4 ||
                    !gemma_moe_down.hetero_supported) {
                    auto moe_hidden = g_profiler.measure("gemma.moe_geglu", [&]() {
                        return moe_geglu_split_cuda(gate_up_pair);
                    });
                    if (tracing_layer) trace_gemma_stage(layer, "moe_hidden", moe_hidden);
                    down_pair = g_profiler.measure("gemma.moe_down", [&]() {
                        return gemma_moe_down.forward(moe_hidden, route);
                    });
                } else {
                    down_pair = g_profiler.measure("gemma.moe_geglu_down", [&]() {
                        return gemma_moe_down.forward_geglu(gate_up_pair, route);
                    });
                }
            }
            trace_gemma_stage(layer, "moe_down", down_pair);
            auto moe_output = g_profiler.measure("gemma.moe_reduce", [&]() {
                return moe_weighted_reduce_cuda(down_pair, selected.at(1));
            });
            trace_gemma_stage(layer, "moe_reduce", moe_output);
            if (fused_norms) {
                auto result = g_profiler.measure("gemma.ffn_merge", [&]() {
                    return gemma4_ffn_merge_f16_cuda(
                        dense_output, moe_output, residual.reshape({B * T, H}),
                        ffn_post_norm_1, ffn_post_norm_2, ffn_post_norm,
                        layer_scale, c.rms_norm_eps).reshape({B, T, H});
                });
                return result;
            }
            moe_output = g_profiler.measure("gemma.ffn_post_norm_2", [&]() {
                return gemma_rms_norm_f16(
                    moe_output, ffn_post_norm_2, c);
            });
            auto combined = g_profiler.measure("gemma.ffn_combine", [&]() {
                return dense_output + moe_output;
            });
            auto post = g_profiler.measure("gemma.ffn_post_norm", [&]() {
                return gemma_rms_norm_f16(combined, ffn_post_norm, c);
            });
            auto result = g_profiler.measure("gemma.ffn_residual", [&]() {
                return acc_cuda(residual.reshape({B * T, H}), post).reshape({B, T, H});
            });
            if (layer_scale.defined()) {
                result = g_profiler.measure("gemma.layer_scale", [&]() {
                    return result * layer_scale;
                });
            }
            trace_gemma_stage(layer, "layer_output", result);
            return result;
        }
        auto attn_pair = g_profiler.measure("full.attn_residual_ffn_norm", [&]() {
            auto rr = residual.reshape({-1, H});
            auto oo2 = oo.reshape({-1, H});
            const char * fp32_residual_env =
                std::getenv("MFQ_DIAGNOSTIC_FP32_RESIDUAL");
            if (fp32_residual_env != nullptr &&
                    fp32_residual_env[0] == '1') {
                return acc_rms_norm_cuda(
                    rr.to(torch::kFloat32),
                    oo2.to(torch::kFloat32),
                    ffn_norm, c.rms_norm_eps,
                    c.norm_weight_offset);
            }
            if (rr.scalar_type() == torch::kFloat16 && oo2.scalar_type() == torch::kFloat16) {
                return acc_rms_norm_f16_cuda(rr, oo2, ffn_norm, c.rms_norm_eps, c.norm_weight_offset);
            }
            return acc_rms_norm_cuda(rr, oo2, ffn_norm, c.rms_norm_eps, c.norm_weight_offset);
        });
        x = attn_pair[0].reshape({B, T, H});
        residual = x;
        xn = attn_pair[1].reshape({B, T, H});
        auto ff = ffn.forward(xn.reshape({B * T, H})).reshape({B, T, H});
        return g_profiler.measure("full.ffn_residual", [&]() {
            auto rr = residual.reshape({-1, H});
            auto ff2 = ff.reshape({-1, H});
            const char * fp32_residual_env =
                std::getenv("MFQ_DIAGNOSTIC_FP32_RESIDUAL");
            if (fp32_residual_env != nullptr &&
                    fp32_residual_env[0] == '1') {
                rr = rr.to(torch::kFloat32);
                ff2 = ff2.to(torch::kFloat32);
            }
            return acc_cuda(rr, ff2).reshape({B, T, H});
        });
    }
};

struct GlmDsaBlock : Block {
    int layer = -1;
    bool full_indexer = false;
    std::shared_ptr<GlmDsaSharedState> shared_state;
    torch::Tensor attn_norm, ffn_norm;
    torch::Tensor q_a_norm, kv_a_norm;
    torch::Tensor index_k_norm, index_k_bias;
    QuantLinearGroup input_proj;
    QuantLinearGroup q_proj;
    NintMoeWeight embed_q;
    NintMoeWeight unembed_out;
    QuantLinear o_proj;
    FFN ffn;
    torch::Tensor kv_cache;
    torch::Tensor index_cache;

    void reset(int64_t B) override {
        shared_state->reset();
        if (kv_cache.defined() && kv_cache.size(0) == B) return;
        kv_cache = torch::Tensor();
        index_cache = torch::Tensor();
    }

    torch::Tensor headwise_project(
        const NintMoeWeight & weight, torch::Tensor x,
        int64_t B, int64_t T, int64_t heads) const {
        if (x.dim() != 4 || x.size(0) != B || x.size(1) != T ||
            x.size(2) != heads || x.size(3) != weight.neuron_len ||
            weight.n_experts != heads) {
            throw std::runtime_error("GLM head-wise NINTM projection shape mismatch");
        }
        const int64_t rows = B * T * heads;
        const auto & route = shared_state->head_route(rows, static_cast<int>(heads));
        auto y = weight.forward(x.contiguous().reshape({rows, weight.neuron_len}), route);
        return y.reshape({B, T, heads, weight.out_per_expert});
    }

    torch::Tensor dense_attention(
        torch::Tensor q, int64_t logical_len, int64_t B,
        const c10::optional<torch::Tensor> & seq_len,
        double scale) const {
        if (seq_len.has_value() && q.size(2) == 1 && B == 1) {
            const int64_t planned_len = g_decode_graph_attention_kv_len > 0
                ? g_decode_graph_attention_kv_len : logical_len;
            shared_state->ensure_decode_workspace(B, planned_len);
            return attention_glm_mla576_decode_cuda(
                q, kv_cache, seq_len.value(), scale, planned_len,
                shared_state->decode_mask, shared_state->kv_max,
                shared_state->attention_meta);
        }
        shared_state->ensure_dense_workspace(B, q.size(2), logical_len);
        return attention_glm_mla576_cached_cuda(
            q, kv_cache, logical_len, shared_state->dense_mask,
            shared_state->kv_max, shared_state->attention_meta, scale);
    }

    void update_indexer(
        torch::Tensor index_q, torch::Tensor index_weights,
        int64_t B, int64_t T, int64_t cache_pos,
        const c10::optional<torch::Tensor> & seq_len,
        const Config & c) const {
        const int64_t logical_len = cache_pos + T;
        if (logical_len <= c.index_topk) {
            shared_state->topk_indices = torch::Tensor();
            shared_state->dense_prefix_rows = T;
            return;
        }

        if (seq_len.has_value() && T == 1 && B == 1) {
            const int64_t planned_len = g_decode_graph_attention_kv_len > 0
                ? g_decode_graph_attention_kv_len : logical_len;
            auto scores = g_profiler.measure("glm.indexer_scores", [&]() {
                return glm_dsa_indexer_scores_decode_cuda(
                    index_q, index_cache, index_weights,
                    seq_len.value(), planned_len);
            });
            auto selected = g_profiler.measure("glm.indexer_topk", [&]() {
                return torch::topk(scores, c.index_topk, -1, true, false);
            });
            shared_state->topk_indices =
                std::get<1>(selected).to(torch::kInt32).contiguous();
            shared_state->dense_prefix_rows = 0;
            return;
        }

        const int64_t prefix_rows = std::max<int64_t>(
            0, std::min<int64_t>(T, c.index_topk - cache_pos));
        const int64_t sparse_rows = T - prefix_rows;
        if (sparse_rows <= 0) {
            shared_state->topk_indices = torch::Tensor();
            shared_state->dense_prefix_rows = T;
            return;
        }
        auto indices = torch::empty(
            {B, sparse_rows, c.index_topk},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
        constexpr int64_t kMaxScoreElements = 32 * 1024 * 1024;
        int64_t rows_per_chunk = std::max<int64_t>(
            1, kMaxScoreElements / std::max<int64_t>(1, B * logical_len));
        rows_per_chunk = std::min<int64_t>(rows_per_chunk, 256);
        if (rows_per_chunk >= 64) rows_per_chunk = rows_per_chunk / 64 * 64;
        for (int64_t start = 0; start < sparse_rows; start += rows_per_chunk) {
            const int64_t count = std::min<int64_t>(rows_per_chunk, sparse_rows - start);
            auto q_chunk = index_q.narrow(1, prefix_rows + start, count).contiguous();
            auto w_chunk = index_weights.narrow(1, prefix_rows + start, count).contiguous();
            auto scores = g_profiler.measure("glm.indexer_scores", [&]() {
                return glm_dsa_indexer_scores_cuda(
                    q_chunk, index_cache, w_chunk,
                    cache_pos + prefix_rows + start, logical_len);
            });
            auto selected = g_profiler.measure("glm.indexer_topk", [&]() {
                return torch::topk(scores, c.index_topk, -1, true, false);
            });
            indices.narrow(1, start, count).copy_(
                std::get<1>(selected).to(torch::kInt32));
        }
        shared_state->topk_indices = indices;
        shared_state->dense_prefix_rows = prefix_rows;
    }

    torch::Tensor forward(
        torch::Tensor x, torch::Tensor pos, int64_t cache_pos,
        const c10::optional<torch::Tensor> & seq_len,
        const Config & c, const RopeCache & rope) override {
        const int64_t B = x.size(0);
        const int64_t T = x.size(1);
        const int64_t H = x.size(2);
        constexpr int64_t kHeads = 64;
        constexpr int64_t kNope = 192;
        constexpr int64_t kRope = 64;
        constexpr int64_t kLatent = 512;
        constexpr int64_t kMlaWidth = kLatent + kRope;
        constexpr int64_t kValue = 256;
        const int64_t logical_len = cache_pos + T;
        if (!kv_cache.defined()) {
            auto options = torch::TensorOptions()
                .device(torch::kCUDA).dtype(torch::kFloat16);
            kv_cache = torch::empty(
                {B, 1, c.max_position_embeddings, kMlaWidth}, options);
            if (full_indexer) {
                index_cache = torch::empty(
                    {B, c.max_position_embeddings, c.index_head_dim}, options);
            }
        }

        auto residual = x.scalar_type() == torch::kFloat16
            ? x.contiguous() : x.to(torch::kFloat16).contiguous();
        auto xn = g_profiler.measure("glm.attn_norm", [&]() {
            return rms_norm_f16_cuda(
                residual.reshape({B * T, H}), attn_norm,
                c.rms_norm_eps, 0.0).reshape({B, T, H});
        });
        auto first = g_profiler.measure("glm.input_proj", [&]() {
            return input_proj.forward(xn);
        });
        const size_t expected_first = full_indexer ? 4u : 2u;
        if (first.size() != expected_first) {
            throw std::runtime_error("GLM DSA input projection count mismatch");
        }
        auto qr = g_profiler.measure("glm.q_a_norm", [&]() {
            return rms_norm_f16_cuda(
                first[0].reshape({B * T, c.q_lora_rank}).to(torch::kFloat16).contiguous(),
                q_a_norm, 1e-6, 0.0).reshape({B, T, c.q_lora_rank});
        });
        auto second = g_profiler.measure("glm.q_proj", [&]() {
            return q_proj.forward(qr);
        });
        const size_t expected_second = full_indexer ? 2u : 1u;
        if (second.size() != expected_second) {
            throw std::runtime_error("GLM DSA q projection count mismatch");
        }

        auto q_main = second[0].to(torch::kFloat16)
            .reshape({B, T, kHeads, kNope + kRope});
        auto q_nope = q_main.index({Slice(), Slice(), Slice(), Slice(0, kNope)})
            .contiguous();
        auto q_pe = q_main.index({Slice(), Slice(), Slice(), Slice(kNope, kNope + kRope)})
            .permute({0, 2, 1, 3}).contiguous();
        q_pe = g_profiler.measure("glm.q_rope", [&]() {
            return glm_interleaved_rope_cuda(
                q_pe, pos.contiguous(), rope.cos, rope.sin, kRope);
        });

        auto compressed = first[1].to(torch::kFloat16)
            .reshape({B, T, kLatent + kRope});
        auto kv_latent = g_profiler.measure("glm.kv_a_norm", [&]() {
            auto raw = compressed.index({Slice(), Slice(), Slice(0, kLatent)})
                .contiguous();
            return rms_norm_f16_cuda(
                raw.reshape({B * T, kLatent}), kv_a_norm,
                1e-6, 0.0).reshape({B, T, kLatent});
        });
        auto k_pe = compressed.index({Slice(), Slice(), Slice(kLatent, kLatent + kRope)})
            .reshape({B, T, 1, kRope}).permute({0, 2, 1, 3}).contiguous();
        k_pe = g_profiler.measure("glm.k_rope", [&]() {
            return glm_interleaved_rope_cuda(
                k_pe, pos.contiguous(), rope.cos, rope.sin, kRope);
        });
        auto kv_rows = torch::cat({
            kv_latent,
            k_pe.permute({0, 2, 1, 3}).reshape({B, T, kRope})}, -1)
            .contiguous();
        g_profiler.measure("glm.kv_write", [&]() {
            return glm_dsa_cache_write_cuda(
                kv_cache.view({B, c.max_position_embeddings, kMlaWidth}),
                kv_rows, pos.contiguous());
        });

        if (full_indexer) {
            auto index_k = g_profiler.measure("glm.indexer_k_norm", [&]() {
                return glm_dsa_indexer_layer_norm_cuda(
                    first[2].to(torch::kFloat16).reshape({B, T, c.index_head_dim}).contiguous(),
                    index_k_norm, index_k_bias, 1e-5);
            });
            index_k = glm_interleaved_rope_cuda(
                index_k.reshape({B, T, 1, c.index_head_dim})
                    .permute({0, 2, 1, 3}).contiguous(),
                pos.contiguous(), rope.cos, rope.sin, kRope)
                .permute({0, 2, 1, 3}).reshape({B, T, c.index_head_dim}).contiguous();
            g_profiler.measure("glm.indexer_k_write", [&]() {
                return glm_dsa_cache_write_cuda(
                    index_cache, index_k, pos.contiguous());
            });
            auto index_q = second[1].to(torch::kFloat16)
                .reshape({B, T, c.index_n_heads, c.index_head_dim})
                .permute({0, 2, 1, 3}).contiguous();
            index_q = glm_interleaved_rope_cuda(
                index_q, pos.contiguous(), rope.cos, rope.sin, kRope)
                .permute({0, 2, 1, 3}).contiguous();
            auto index_weights = first[3].reshape({B, T, c.index_n_heads})
                .to(torch::kFloat32).contiguous();
            update_indexer(
                index_q, index_weights, B, T, cache_pos, seq_len, c);
        }

        auto q_absorbed = g_profiler.measure("glm.embed_q", [&]() {
            return headwise_project(embed_q, q_nope, B, T, kHeads);
        }).permute({0, 2, 1, 3}).contiguous();
        auto q_mla = torch::cat({q_absorbed, q_pe}, -1)
            .to(torch::kFloat32).contiguous();
        const double scale = 1.0 / std::sqrt(
            static_cast<double>(kNope + kRope));
        torch::Tensor attended;
        if (!shared_state->topk_indices.defined()) {
            attended = g_profiler.measure("glm.attention_dense", [&]() {
                return dense_attention(q_mla, logical_len, B, seq_len, scale);
            });
        } else {
            const int64_t prefix = shared_state->dense_prefix_rows;
            const int64_t sparse_rows = shared_state->topk_indices.size(1);
            if (prefix + sparse_rows != T) {
                throw std::runtime_error("GLM DSA shared index state has the wrong row count");
            }
            torch::Tensor dense_out;
            if (prefix > 0) {
                dense_out = g_profiler.measure("glm.attention_dense_prefix", [&]() {
                    return dense_attention(
                        q_mla.narrow(2, 0, prefix).contiguous(),
                        cache_pos + prefix, B, c10::nullopt, scale);
                });
            }
            shared_state->ensure_meta();
            auto sparse_out = g_profiler.measure("glm.attention_sparse", [&]() {
                return attention_glm_mla_sparse_cuda(
                    q_mla.narrow(2, prefix, sparse_rows).contiguous(),
                    kv_cache.view({B, c.max_position_embeddings, kMlaWidth}),
                    shared_state->topk_indices,
                    shared_state->attention_meta, scale);
            });
            attended = prefix > 0
                ? torch::cat({dense_out, sparse_out}, 1) : sparse_out;
        }
        auto value_heads = g_profiler.measure("glm.unembed_out", [&]() {
            return headwise_project(
                unembed_out, attended.to(torch::kFloat16).contiguous(),
                B, T, kHeads);
        });
        auto attn_out = g_profiler.measure("glm.o_proj", [&]() {
            return o_proj.forward(
                value_heads.reshape({B, T, kHeads * kValue}).contiguous());
        });
        auto attn_pair = g_profiler.measure("glm.attn_residual_ffn_norm", [&]() {
            return acc_rms_norm_f16_cuda(
                residual.reshape({B * T, H}),
                attn_out.reshape({B * T, H}).to(torch::kFloat16).contiguous(),
                ffn_norm, c.rms_norm_eps, 0.0);
        });
        auto hidden = attn_pair[0].reshape({B, T, H});
        auto ffn_input = attn_pair[1].reshape({B * T, H});
        auto ffn_out = ffn.forward(ffn_input).reshape({B * T, H});
        return g_profiler.measure("glm.ffn_residual", [&]() {
            return acc_cuda(hidden.reshape({B * T, H}), ffn_out)
                .reshape({B, T, H});
        });
    }
};

struct LinearBlock : Block {
    torch::Tensor attn_norm, ffn_norm, conv_weight, conv_bias, dt_bias, a_log, linear_norm;
    bool split_in_proj = false;
    bool split_dense_zab = false;
    bool dense_ab_tail = false;
    bool ab_is_nint = false;
    bool dense_out_proj = false;
    bool tiled_v_heads = false;
    QuantLinearGroup in_proj;
    QuantLinearGroup qkv_proj;
    QuantLinearGroup qkvz_proj;
    QuantLinear z_proj;
    QuantLinearGroup ab_nint_proj;
    DenseLinearGroup ab_proj;
    DenseLinearGroup zab_proj;
    QuantLinear out_proj;
    torch::Tensor out_proj_dense;
    FFN ffn;
    torch::Tensor conv_state, gdn_state;

    void reset(int64_t B) override {
        if (conv_state.defined() && gdn_state.defined() &&
            conv_state.size(0) == B && gdn_state.size(0) == B) {
            conv_state.zero_();
            gdn_state.zero_();
            return;
        }
        conv_state = torch::Tensor();
        gdn_state = torch::Tensor();
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor pos, int64_t cache_pos,
                          const c10::optional<torch::Tensor> & seq_len,
                          const Config & c, const RopeCache & rope) override {
        (void)seq_len;
        (void)pos; (void)cache_pos; (void)rope;
        int64_t B = x.size(0), T = x.size(1), H = x.size(2);
        int64_t nk = c.linear_num_key_heads, nv = c.linear_num_value_heads;
        int64_t dk = c.linear_key_head_dim, dv = c.linear_value_head_dim;
        int64_t ksz = c.linear_k_size(), vsz = c.linear_v_size();
        int64_t conv_dim = 2 * ksz + vsz;
        if (!conv_state.defined()) {
            conv_state = torch::zeros({B, c.linear_conv_kernel_dim - 1, conv_dim}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            gdn_state = torch::zeros({B, nv, dv, dv}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        }
        auto residual = x;
        auto xn = g_profiler.measure("linear.attn_norm", [&]() {
            return qwen_rms_norm(x.reshape({B * T, H}).to(torch::kFloat32), attn_norm, c).reshape({B, T, H});
        });
        torch::Tensor qkv, qk_part, v_part, z, alpha_raw, beta_raw;
        if (dense_ab_tail) {
            auto parts = g_profiler.measure("linear.in_proj", [&]() { return in_proj.forward(xn); });
            qkv = g_profiler.measure("linear.qkv_cast", [&]() { return parts[0].to(torch::kFloat32); });
            z = parts[1];
            auto ab = g_profiler.measure("linear.ab_proj", [&]() { return ab_proj.forward(xn); });
            alpha_raw = ab[0];
            beta_raw = ab[1];
        } else if (split_in_proj) {
            if (split_dense_zab) {
                auto qkv_parts = g_profiler.measure("linear.qkv_proj", [&]() { return qkv_proj.forward(xn); });
                qk_part = qkv_parts[0];
                v_part = qkv_parts[1];
                auto zab = g_profiler.measure("linear.zab_proj", [&]() { return zab_proj.forward(xn); });
                z = zab[0];
                alpha_raw = zab[1];
                beta_raw = zab[2];
            } else {
                auto qkv_parts = g_profiler.measure("linear.qkv_proj", [&]() { return qkv_proj.forward(xn); });
                qk_part = qkv_parts[0];
                v_part = qkv_parts[1];
                z = g_profiler.measure("linear.z_proj", [&]() { return z_proj.forward(xn); });
                auto ab = g_profiler.measure("linear.ab_proj", [&]() {
                    return ab_is_nint ? ab_nint_proj.forward(xn) : ab_proj.forward(xn);
                });
                alpha_raw = ab[0];
                beta_raw = ab[1];
            }
        } else {
            auto parts = g_profiler.measure("linear.in_proj", [&]() { return in_proj.forward(xn); });
            qkv = g_profiler.measure("linear.qkv_cast", [&]() { return parts[0].to(torch::kFloat32); });
            z = parts[1];
            alpha_raw = parts[2];
            beta_raw = parts[3];
        }
        auto gates = g_profiler.measure("linear.gates_fused", [&]() {
            return linear_gate_beta_cuda(alpha_raw.reshape({B, T, nv}), beta_raw.reshape({B, T, nv}), dt_bias, a_log);
        });
        auto gate_t = gates[0];
        auto beta_t = gates[1];
        auto bias = conv_bias.defined() ? conv_bias : torch::empty({0}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        torch::Tensor q, k, v;
        const char * fused_prefill_env = std::getenv("MFQ_LINEAR_CONV_PREFILL_FUSED");
        bool fused_prefill = T >= 256 && (fused_prefill_env == nullptr || fused_prefill_env[0] != '0');
        if (T > 1 && split_in_proj && fused_prefill) {
            auto qkv_fast = g_profiler.measure("linear.conv_qkv_prefill", [&]() {
                return linear_conv_qkv_prefill_cuda(
                    conv_state,
                    qk_part.contiguous(),
                    v_part.contiguous(),
                    conv_weight,
                    bias,
                    nk,
                    nv,
                    dk,
                    dv,
                    c.rms_norm_eps);
            });
            q = qkv_fast[0];
            k = qkv_fast[1];
            v = qkv_fast[2];
            // Server CUDA graphs retain this state storage address across requests.
            conv_state.copy_(qkv_fast[3]);
        } else if (T == 1 && split_in_proj) {
            auto qkv_fast = g_profiler.measure("linear.conv_qkv_decode", [&]() {
                return linear_conv_qkv_decode_cuda(
                    conv_state,
                    qk_part.contiguous(),
                    v_part.contiguous(),
                    conv_weight,
                    bias,
                    nk,
                    nv,
                    dk,
                    dv,
                    c.rms_norm_eps);
            });
            q = qkv_fast[0];
            k = qkv_fast[1];
            v = qkv_fast[2];
        } else {
            if (split_in_proj) {
                qkv = g_profiler.measure("linear.qkv_cat", [&]() {
                    return torch::cat({qk_part.to(torch::kFloat32), v_part.to(torch::kFloat32)}, -1);
                });
            }
            torch::Tensor conv;
            if (T == 1) {
            conv = g_profiler.measure("linear.conv_silu_decode", [&]() {
                return ssm_conv_silu_decode_cuda(conv_state, qkv, conv_weight, bias);
            });
            } else {
            auto conv_in = g_profiler.measure("linear.conv_input", [&]() { return torch::cat({conv_state, qkv}, 1); });
            auto next_conv_state = g_profiler.measure("linear.conv_state_update", [&]() { return conv_in.index({Slice(), Slice(-(c.linear_conv_kernel_dim - 1), torch::indexing::None), Slice()}).contiguous(); });
            conv_state.copy_(next_conv_state);
            conv = g_profiler.measure("linear.conv_silu", [&]() { return ssm_conv_silu_cuda(conv_in, conv_weight, bias, T); });
            }
            q = g_profiler.measure("linear.q_view", [&]() { return conv.index({Slice(), Slice(), Slice(0, ksz)}).reshape({B, T, nk, dk}).transpose(1, 2); });
            k = g_profiler.measure("linear.k_view", [&]() { return conv.index({Slice(), Slice(), Slice(ksz, 2 * ksz)}).reshape({B, T, nk, dk}).transpose(1, 2); });
            v = g_profiler.measure("linear.v_view", [&]() { return conv.index({Slice(), Slice(), Slice(2 * ksz, 2 * ksz + vsz)}).reshape({B, T, nv, dv}).transpose(1, 2); });
            q = g_profiler.measure("linear.q_l2", [&]() { return l2_norm_cuda(q.contiguous().reshape({-1, dk}), c.rms_norm_eps).reshape_as(q); });
            k = g_profiler.measure("linear.k_l2", [&]() { return l2_norm_cuda(k.contiguous().reshape({-1, dk}), c.rms_norm_eps).reshape_as(k); });
        }
        auto gd = g_profiler.measure("linear.gdn", [&]() {
            const char* transposed_env = std::getenv("MFQ_GDN_TRANSPOSED_STATE");
            bool transposed = transposed_env == nullptr || transposed_env[0] != '0';
            if (transposed) {
                if (tiled_v_heads) {
                    return gdn_inplace_transposed_tiled_cuda(
                        q.contiguous(), k.contiguous(), v.contiguous(),
                        gate_t, beta_t, gdn_state);
                }
                return gdn_inplace_transposed_cuda(q.contiguous(), k.contiguous(), v.contiguous(),
                                                   gate_t, beta_t, gdn_state);
            }
            if (tiled_v_heads) {
                return gdn_inplace_tiled_cuda(q.contiguous(), k.contiguous(), v.contiguous(),
                                              gate_t, beta_t, gdn_state);
            }
            return gdn_inplace_cuda(q.contiguous(), k.contiguous(), v.contiguous(),
                                    gate_t, beta_t, gdn_state);
        });
        auto y = gd[0];
        gdn_state = gd[1];
        torch::Tensor oo;
        const NintWeight * out_nint =
            !dense_out_proj && out_proj.is_nint() &&
            !out_proj.tensor_parallel()
                ? &out_proj.nint.w : nullptr;
        if (out_nint != nullptr && B == 1 && T == 1 && out_nint->bits == 5 && out_nint->gs == 28 &&
            y.scalar_type() == torch::kFloat32 && z.scalar_type() == torch::kFloat16) {
            oo = g_profiler.measure("linear.out_norm_gate_proj", [&]() {
                Workspace & ws = out_nint->workspace(1);
                out_nint->ensure_rinv_workspace(ws, nv);
                return nint_gemv_packed_bits_linear_out_norm_gate_ws_cuda(
                    out_nint->q_packed, out_nint->sub_scale, out_nint->sub_min,
                    out_nint->neuron_scale, out_nint->neuron_min,
                    y.contiguous().reshape({vsz}), z.contiguous().reshape({vsz}), linear_norm,
                    out_nint->gs, out_nint->bits, dv, c.rms_norm_eps,
                    ws.qx, ws.xscale, ws.xsum, ws.rinv).reshape({B, T, H});
            });
        } else {
            z = g_profiler.measure("linear.z_view", [&]() { return z.reshape({B, T, nv, dv}).transpose(1, 2).contiguous(); });
            auto y_norm = g_profiler.measure("linear.out_norm", [&]() { return rms_norm_cuda(y.reshape({-1, dv}).to(torch::kFloat32), linear_norm, c.rms_norm_eps).reshape_as(y); });
            auto yf = g_profiler.measure("linear.y_view", [&]() { return y_norm.transpose(1, 2).contiguous().reshape({B, T, vsz}); });
            auto zf = g_profiler.measure("linear.zf_view", [&]() { return z.transpose(1, 2).contiguous().reshape({B, T, vsz}); });
            if (dense_out_proj) {
                oo = g_profiler.measure("linear.out_proj_gate_dense", [&]() {
                    auto dtype = out_proj_dense.scalar_type();
                    auto gated = yf.to(dtype) * torch::silu(zf.to(dtype));
                    return torch::matmul(
                        gated.reshape({B * T, vsz}),
                        out_proj_dense.transpose(0, 1)).reshape({B, T, H});
                });
            } else {
                oo = g_profiler.measure("linear.out_proj_gate", [&]() {
                    return out_proj.forward_input_mul(yf, zf, 2);
                });
            }
        }
        auto attn_pair = g_profiler.measure("linear.attn_residual_ffn_norm", [&]() {
            auto rr = residual.reshape({-1, H});
            auto oo2 = oo.reshape({-1, H});
            const char * fp32_residual_env =
                std::getenv("MFQ_DIAGNOSTIC_FP32_RESIDUAL");
            if (fp32_residual_env != nullptr &&
                    fp32_residual_env[0] == '1') {
                return acc_rms_norm_cuda(
                    rr.to(torch::kFloat32),
                    oo2.to(torch::kFloat32),
                    ffn_norm, c.rms_norm_eps,
                    c.norm_weight_offset);
            }
            if (rr.scalar_type() == torch::kFloat16 && oo2.scalar_type() == torch::kFloat16) {
                return acc_rms_norm_f16_cuda(rr, oo2, ffn_norm, c.rms_norm_eps, c.norm_weight_offset);
            }
            return acc_rms_norm_cuda(rr, oo2, ffn_norm, c.rms_norm_eps, c.norm_weight_offset);
        });
        x = attn_pair[0].reshape({B, T, H});
        residual = x;
        xn = attn_pair[1].reshape({B, T, H});
        auto ff = ffn.forward(xn.reshape({B * T, H})).reshape({B, T, H});
        return g_profiler.measure("linear.ffn_residual", [&]() {
            auto rr = residual.reshape({-1, H});
            auto ff2 = ff.reshape({-1, H});
            const char * fp32_residual_env =
                std::getenv("MFQ_DIAGNOSTIC_FP32_RESIDUAL");
            if (fp32_residual_env != nullptr &&
                    fp32_residual_env[0] == '1') {
                rr = rr.to(torch::kFloat32);
                ff2 = ff2.to(torch::kFloat32);
            }
            return acc_cuda(rr, ff2).reshape({B, T, H});
        });
    }
};

static void prepare_ffn_workspaces(FFN & f) {
    if (f.down.tensor_parallel()) return;
    if (f.gate_up.nvq_prefix2 && f.gate_up.layers.size() == 2 && f.down.is_nvq() &&
        f.gate_up.outs.size() == 2 && f.gate_up.outs[0] == f.gate_up.outs[1] &&
        f.gate_up.outs[0] == f.down.nvq.w.neuron_len) {
        NvqWorkspace & ws = f.gate_up.layers[0].nvq.w.workspace(1);
        ws.swiglu_scratch = torch::empty(
            {f.gate_up.outs[0]}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        (void)f.down.nvq.w.workspace(1);
    }
    if (f.important_neurons) {
        prepare_ffn_workspaces(*f.important_neurons);
    }
}

static void load_important_neuron_branch(
        const MfqFile & mfq,
        const Config & c,
        FFN & f,
        const std::string & down_name,
        const std::string & gate_name,
        const std::string & up_name) {
    const std::string down_high = down_name + ".in_high";
    const std::string gate_high = gate_name + ".in_high";
    const std::string up_high = up_name + ".in_high";
    const bool has_down = mfq.has_record(down_high);
    const bool has_gate = mfq.has_record(gate_high);
    const bool has_up = mfq.has_record(up_high);
    if (!has_down && !has_gate && !has_up) {
        return;
    }
    if (!has_down || !has_gate || !has_up) {
        throw std::runtime_error(
            "important-neuron FFN requires matching gate/up/down .in_high records");
    }
    if (f.is_moe) {
        throw std::runtime_error(
            "important-neuron records are unsupported on routed MoE FFNs");
    }

    auto high = std::make_unique<FFN>();
    high->down = load_quant_linear(
        mfq, down_high, TensorParallelAxis::Input);
    high->gate_up = load_paired_gate_up(
        mfq, {gate_high, up_high}, high->down);
    high->geglu = f.geglu;
    high->swiglu_limit = f.swiglu_limit;

    if (f.gate_up.outs.size() != 2 ||
        high->gate_up.outs.size() != 2 ||
        f.gate_up.outs[0] != f.gate_up.outs[1] ||
        high->gate_up.outs[0] != high->gate_up.outs[1] ||
        f.down.out() != c.hidden_size ||
        high->down.out() != c.hidden_size ||
        f.down.neuron_len() != f.gate_up.outs[0] ||
        high->down.neuron_len() != high->gate_up.outs[0] ||
        f.down.neuron_len() + high->down.neuron_len() !=
            c.intermediate_size) {
        throw std::runtime_error(
            "important-neuron FFN tensor shapes disagree with model config");
    }
    f.important_neurons = std::move(high);
}

static FFN load_ffn(const MfqFile & mfq, const Config & c, int i, bool gguf_names) {
    FFN f;
    if (c.is_glm_dsa()) {
        if (gguf_names) {
            throw std::runtime_error("GLM DSA runtime expects HF tensor names in MFQ");
        }
        const std::string p =
            "model.layers." + std::to_string(i) + ".mlp.";
        if (c.mlp_layer_types.at(static_cast<size_t>(i)) == "sparse") {
            const std::string expert_gate_up = p + "experts.gate_up_proj";
            const std::string expert_down = p + "experts.down_proj";
            f.is_moe = true;
            f.moe_gate_up = load_nint_moe_gpu(
                mfq, expert_gate_up, true, i, "gate_up");
            f.moe_down = load_nint_moe_gpu(
                mfq, expert_down, true, i, "down");
            f.moe_router = load_dense_gpu(
                mfq, p + "gate.weight").to(torch::kFloat32).contiguous();
            f.moe_router_bias = load_dense_gpu(
                mfq, p + "gate.e_score_correction_bias")
                .to(torch::kFloat32).contiguous();
            f.moe_top_k = static_cast<int>(c.num_experts_per_tok);
            f.moe_use_sigmoid = true;
            f.moe_use_sqrt_softplus = false;
            f.moe_normalize = c.norm_topk_prob;
            f.moe_delayed_softmax = false;
            f.moe_shared_ungated = true;
            f.moe_router_scale = c.routed_scaling_factor;
            f.shared = std::make_unique<FFN>();
            f.shared->down = load_quant_linear(
                mfq, p + "shared_experts.down_proj.weight");
            f.shared->gate_up = load_paired_gate_up(mfq, {
                p + "shared_experts.gate_proj.weight",
                p + "shared_experts.up_proj.weight"},
                f.shared->down);
            prepare_ffn_workspaces(*f.shared);
            if (f.moe_gate_up.n_experts != c.num_experts ||
                f.moe_down.n_experts != c.num_experts ||
                f.moe_gate_up.neuron_len != c.hidden_size ||
                f.moe_gate_up.out_per_expert != 2 * c.moe_intermediate_size ||
                f.moe_down.neuron_len != c.moe_intermediate_size ||
                f.moe_down.out_per_expert != c.hidden_size ||
                f.moe_router.dim() != 2 ||
                f.moe_router.size(0) != c.num_experts ||
                f.moe_router.size(1) != c.hidden_size ||
                f.moe_router_bias.numel() != c.num_experts) {
                throw std::runtime_error(
                    "GLM DSA MoE tensor shapes disagree with config at layer " +
                    std::to_string(i));
            }
            return f;
        }
        const std::string down_name = p + "down_proj.weight";
        const std::string gate_name = p + "gate_proj.weight";
        const std::string up_name = p + "up_proj.weight";
        f.down = load_quant_linear(mfq, down_name);
        f.gate_up = load_paired_gate_up(mfq, {
            gate_name, up_name},
            f.down);
        load_important_neuron_branch(
            mfq, c, f, down_name, gate_name, up_name);
        prepare_ffn_workspaces(f);
        return f;
    }
    if (gguf_names) {
        std::string p = "blk." + std::to_string(i) + ".ffn_";
        const std::string down_name = p + "down.weight";
        const std::string gate_name = p + "gate.weight";
        const std::string up_name = p + "up.weight";
        f.down = load_quant_linear(mfq, down_name);
        f.gate_up = load_paired_gate_up(
            mfq, {gate_name, up_name},
            f.down);
        load_important_neuron_branch(
            mfq, c, f, down_name, gate_name, up_name);
    } else {
        std::string p = "model.language_model.layers." + std::to_string(i) + ".mlp.";
        const std::string expert_gate_up = p + "experts.gate_up_proj";
        const std::string expert_down = p + "experts.down_proj";
        if (mfq.records.count(expert_gate_up) || mfq.records.count(expert_down)) {
            if (!mfq.records.count(expert_gate_up) || !mfq.records.count(expert_down)) {
                throw std::runtime_error("MoE layer has only one expert tensor: " + p);
            }
            if (c.num_experts <= 0 || c.num_experts_per_tok <= 0 ||
                c.moe_intermediate_size <= 0 || c.shared_expert_intermediate_size <= 0) {
                throw std::runtime_error("MoE config fields are missing");
            }
            f.is_moe = true;
            f.moe_gate_up = load_nint_moe_gpu(
                mfq, expert_gate_up, true, i, "gate_up");
            f.moe_down = load_nint_moe_gpu(
                mfq, expert_down, true, i, "down");
            f.moe_router = load_dense_gpu(mfq, p + "gate.weight").to(torch::kFloat32).contiguous();
            f.moe_shared_gate = load_dense_gpu(
                mfq, p + "shared_expert_gate.weight").to(torch::kFloat32).contiguous();
            f.moe_top_k = static_cast<int>(c.num_experts_per_tok);
            f.moe_use_sqrt_softplus = c.expert_gating_func == "sqrtsoftplus";
            if (f.moe_use_sqrt_softplus) {
                f.moe_normalize = c.norm_topk_prob;
                f.moe_delayed_softmax = false;
                f.moe_router_scale = c.routed_scaling_factor;
            }
            f.shared = std::make_unique<FFN>();
            f.shared->down = load_quant_linear(mfq, p + "shared_expert.down_proj.weight");
            f.shared->gate_up = load_paired_gate_up(mfq, {
                p + "shared_expert.gate_proj.weight",
                p + "shared_expert.up_proj.weight"},
                f.shared->down);
            prepare_ffn_workspaces(*f.shared);
            if (f.moe_gate_up.n_experts != c.num_experts ||
                f.moe_down.n_experts != c.num_experts ||
                f.moe_gate_up.neuron_len != c.hidden_size ||
                f.moe_gate_up.out_per_expert != 2 * c.moe_intermediate_size ||
                f.moe_down.neuron_len != c.moe_intermediate_size ||
                f.moe_down.out_per_expert != c.hidden_size ||
                f.moe_router.dim() != 2 || f.moe_router.size(0) != c.num_experts ||
                f.moe_router.size(1) != c.hidden_size ||
                f.moe_shared_gate.dim() != 2 || f.moe_shared_gate.size(0) != 1 ||
                f.moe_shared_gate.size(1) != c.hidden_size) {
                throw std::runtime_error("MoE tensor shapes disagree with config at layer " + std::to_string(i));
            }
            return f;
        }
        const std::string down_name = p + "down_proj.weight";
        const std::string gate_name = p + "gate_proj.weight";
        const std::string up_name = p + "up_proj.weight";
        f.down = load_quant_linear(mfq, down_name);
        f.gate_up = load_paired_gate_up(mfq, {
            gate_name, up_name},
            f.down);
        load_important_neuron_branch(
            mfq, c, f, down_name, gate_name, up_name);
    }
    prepare_ffn_workspaces(f);
    return f;
}

static std::unique_ptr<Block> load_block(
    const MfqFile & mfq,
    const Config & c,
    int i,
    const std::string & type,
    bool gguf_names,
    const std::shared_ptr<GlmDsaSharedState> & glm_state = nullptr,
    const std::shared_ptr<Dsv4SharedState> & dsv4_state = nullptr) {
    if (c.is_dsv4()) {
        if (!gguf_names || type != "deepseek_v4" || !dsv4_state) {
            throw std::runtime_error(
                "invalid DeepSeek V4 block loader state");
        }
        const std::string p =
            "blk." + std::to_string(i) + ".";
        auto b = std::make_unique<Dsv4Block>();
        b->layer = i;
        b->max_positions = c.max_position_embeddings;
        b->compress_ratio =
            c.compress_ratios.at(static_cast<size_t>(i));
        b->hidden_size = c.hidden_size;
        b->heads = c.num_attention_heads;
        b->head_dim = c.head_dim;
        b->groups = c.o_groups;
        b->o_rank = c.o_lora_rank;
        b->hc_mult = c.hc_mult;
        b->hc_iterations = c.hc_sinkhorn_iters;
        b->eps = c.rms_norm_eps;
        b->hc_eps = c.hc_eps;
        b->shared_state = dsv4_state;

        b->attn_norm = load_dense_gpu(
            mfq, p + "attn_norm.weight");
        b->ffn_norm = load_dense_gpu(
            mfq, p + "ffn_norm.weight");
        b->q_a_norm = load_dense_gpu(
            mfq, p + "attn_q_a_norm.weight");
        b->kv_norm = load_dense_gpu(
            mfq, p + "attn_kv_a_norm.weight");
        b->sinks = load_dense_gpu(
            mfq, p + "attn_sinks.weight")
            .to(torch::kFloat32).contiguous();
        b->hc_attn_fn = load_dense_gpu(
            mfq, p + "hc_attn_fn.weight")
            .to(torch::kFloat32).contiguous();
        b->hc_attn_scale = load_dense_gpu(
            mfq, p + "hc_attn_scale.weight")
            .to(torch::kFloat32).contiguous();
        b->hc_attn_base = load_dense_gpu(
            mfq, p + "hc_attn_base.weight")
            .to(torch::kFloat32).contiguous();
        b->hc_ffn_fn = load_dense_gpu(
            mfq, p + "hc_ffn_fn.weight")
            .to(torch::kFloat32).contiguous();
        b->hc_ffn_scale = load_dense_gpu(
            mfq, p + "hc_ffn_scale.weight")
            .to(torch::kFloat32).contiguous();
        b->hc_ffn_base = load_dense_gpu(
            mfq, p + "hc_ffn_base.weight")
            .to(torch::kFloat32).contiguous();
        b->q_a = load_quant_linear(
            mfq, p + "attn_q_a.weight");
        b->q_b = load_quant_linear(
            mfq, p + "attn_q_b.weight");
        b->kv = load_quant_linear(
            mfq, p + "attn_kv.weight");
        b->output_a = load_quant_linear(
            mfq, p + "attn_output_a.weight");
        b->output_b = load_quant_linear(
            mfq, p + "attn_output_b.weight");
        b->attention_rope = Dsv4RopeTable(
            c, b->compress_ratio);

        if (b->compress_ratio > 0) {
            b->compressor.ratio = b->compress_ratio;
            b->compressor.head_dim = c.head_dim;
            b->compressor.overlap =
                b->compress_ratio == 4;
            b->compressor.cache_quant_mode = 1;
            b->compressor.projection = make_fp32_quant_group(
                load_quant_group(mfq, {
                    p + "attn_compressor_kv.weight",
                    p + "attn_compressor_gate.weight"}));
            b->compressor.ape = load_dense_gpu(
                mfq, p + "attn_compressor_ape.weight")
                .to(torch::kFloat32).contiguous();
            b->compressor.norm = load_dense_gpu(
                mfq, p + "attn_compressor_norm.weight")
                .to(torch::kFloat32).contiguous();
        }
        if (b->compress_ratio == 4) {
            b->indexer_compressor.ratio = 4;
            b->indexer_compressor.head_dim = c.index_head_dim;
            b->indexer_compressor.overlap = true;
            b->indexer_compressor.cache_quant_mode = 2;
            b->indexer_compressor.projection =
                make_fp32_quant_group(load_quant_group(mfq, {
                    p + "indexer_compressor_kv.weight",
                    p + "indexer_compressor_gate.weight"}));
            b->indexer_compressor.ape = load_dense_gpu(
                mfq, p + "indexer_compressor_ape.weight")
                .to(torch::kFloat32).contiguous();
            b->indexer_compressor.norm = load_dense_gpu(
                mfq, p + "indexer_compressor_norm.weight")
                .to(torch::kFloat32).contiguous();
            b->indexer_q = load_quant_linear(
                mfq, p + "indexer.attn_q_b.weight");
            b->indexer_weight = load_dense_gpu(
                mfq, p + "indexer.proj.weight")
                .to(torch::kFloat32).contiguous();
        }

        b->ffn.is_moe = true;
        const bool has_split_gate =
            mfq.records.count(p + "ffn_gate_exps.weight") != 0;
        const bool has_split_up =
            mfq.records.count(p + "ffn_up_exps.weight") != 0;
        if (has_split_gate != has_split_up) {
            throw std::runtime_error(
                "DeepSeek V4 split routed Gate/Up records are incomplete at layer " +
                std::to_string(i));
        }
        b->ffn.moe_split_gate_up = has_split_gate;
        const bool cpu_offload =
            g_dsv4_cpu_offload_layers.count(i) != 0;
        if (cpu_offload) {
            if (b->ffn.moe_split_gate_up) {
                b->ffn.cpu_moe_gate = load_nint_moe_cpu_offloaded(
                    mfq, p + "ffn_gate_exps.weight");
                b->ffn.cpu_moe_up = load_nint_moe_cpu_offloaded(
                    mfq, p + "ffn_up_exps.weight");
                b->ffn.moe_gate =
                    cpu_mixed_moe_metadata(b->ffn.cpu_moe_gate);
                b->ffn.moe_up =
                    cpu_mixed_moe_metadata(b->ffn.cpu_moe_up);
            } else {
                b->ffn.cpu_moe_gate_up = load_nint_moe_cpu_offloaded(
                    mfq, p + "ffn_gate_up_exps.weight");
                b->ffn.moe_gate_up =
                    cpu_mixed_moe_metadata(b->ffn.cpu_moe_gate_up);
            }
            b->ffn.cpu_moe_down = load_nint_moe_cpu_offloaded(
                mfq, p + "ffn_down_exps.weight");
            b->ffn.moe_down =
                cpu_mixed_moe_metadata(b->ffn.cpu_moe_down);
            const int64_t gate_up_bytes = b->ffn.moe_split_gate_up
                ? b->ffn.moe_gate.mixed_weight_bytes +
                    b->ffn.moe_up.mixed_weight_bytes
                : b->ffn.moe_gate_up.mixed_weight_bytes;
            const int64_t down_bytes =
                b->ffn.moe_down.mixed_weight_bytes;
            g_dsv4_cpu_offload_host_bytes +=
                gate_up_bytes + down_bytes;
            std::cerr
                << "cpu_offload layer=" << i
                << " gate_up_bytes=" << gate_up_bytes
                << " down_bytes=" << down_bytes
                << " total_host_bytes="
                << g_dsv4_cpu_offload_host_bytes
                << std::endl;
        } else {
            if (b->ffn.moe_split_gate_up) {
                b->ffn.moe_gate = load_nint_moe_gpu(
                    mfq, p + "ffn_gate_exps.weight",
                    true, i, "gate");
                b->ffn.moe_up = load_nint_moe_gpu(
                    mfq, p + "ffn_up_exps.weight",
                    true, i, "up");
            } else {
                b->ffn.moe_gate_up = load_nint_moe_gpu(
                    mfq, p + "ffn_gate_up_exps.weight",
                    true, i, "gate_up");
            }
            b->ffn.moe_down = load_nint_moe_gpu(
                mfq, p + "ffn_down_exps.weight",
                true, i, "down");
        }
        b->ffn.moe_router = load_dense_gpu(
            mfq, p + "ffn_gate_inp.weight")
            .to(torch::kFloat32).contiguous();
        if (mfq.records.count(p + "exp_probs_b.bias")) {
            b->ffn.moe_router_bias = load_dense_gpu(
                mfq, p + "exp_probs_b.bias")
                .to(torch::kFloat32).contiguous();
        }
        if (i < c.dsv4_hash_layer_count) {
            b->ffn.moe_hash_ids = load_dense_gpu(
                mfq, p + "ffn_gate_tid2eid.weight")
                .to(torch::kInt32).contiguous();
        }
        b->ffn.moe_top_k =
            static_cast<int>(c.num_experts_per_tok);
        b->ffn.moe_use_sqrt_softplus = true;
        b->ffn.moe_normalize = c.norm_topk_prob;
        b->ffn.moe_delayed_softmax = false;
        b->ffn.moe_shared_ungated = true;
        b->ffn.moe_router_scale =
            c.routed_scaling_factor;
        b->ffn.swiglu_limit = c.swiglu_limit;
        b->ffn.moe_layer = i;
        b->ffn.shared = std::make_unique<FFN>();
        b->ffn.shared->down = load_quant_linear(
            mfq, p + "ffn_down_shexp.weight");
        b->ffn.shared->gate_up = load_paired_gate_up(mfq, {
            p + "ffn_gate_shexp.weight",
            p + "ffn_up_shexp.weight"},
            b->ffn.shared->down, 0);
        b->ffn.shared->swiglu_limit = c.swiglu_limit;
        prepare_ffn_workspaces(*b->ffn.shared);

        const bool base_shapes =
            b->attn_norm.numel() == c.hidden_size &&
            b->ffn_norm.numel() == c.hidden_size &&
            b->q_a_norm.numel() == c.q_lora_rank &&
            b->kv_norm.numel() == c.head_dim &&
            b->sinks.numel() == c.num_attention_heads &&
            b->q_a.neuron_len() == c.hidden_size &&
            b->q_a.out() == c.q_lora_rank &&
            b->q_b.neuron_len() == c.q_lora_rank &&
            b->q_b.out() == c.num_attention_heads * c.head_dim &&
            b->kv.neuron_len() == c.hidden_size &&
            b->kv.out() == c.head_dim &&
            b->output_a.neuron_len() ==
                c.num_attention_heads * c.head_dim / c.o_groups &&
            b->output_a.out() == c.o_groups * c.o_lora_rank &&
            b->output_b.neuron_len() == c.o_groups * c.o_lora_rank &&
            b->output_b.out() == c.hidden_size;
        const bool gate_up_shapes = b->ffn.moe_split_gate_up
            ? b->ffn.moe_gate.n_experts == c.num_experts &&
                b->ffn.moe_up.n_experts == c.num_experts &&
                b->ffn.moe_gate.neuron_len == c.hidden_size &&
                b->ffn.moe_up.neuron_len == c.hidden_size &&
                b->ffn.moe_gate.out_per_expert == c.moe_intermediate_size &&
                b->ffn.moe_up.out_per_expert == c.moe_intermediate_size
            : b->ffn.moe_gate_up.n_experts == c.num_experts &&
                b->ffn.moe_gate_up.neuron_len == c.hidden_size &&
                b->ffn.moe_gate_up.out_per_expert ==
                    2 * c.moe_intermediate_size;
        const bool moe_shapes =
            gate_up_shapes &&
            b->ffn.moe_down.n_experts == c.num_experts &&
            b->ffn.moe_down.neuron_len == c.moe_intermediate_size &&
            b->ffn.moe_down.out_per_expert == c.hidden_size &&
            b->ffn.moe_router.size(0) == c.num_experts &&
            b->ffn.moe_router.size(1) == c.hidden_size;
        if (!base_shapes || !moe_shapes) {
            throw std::runtime_error(
                "DeepSeek V4 tensor shapes disagree with config at layer " +
                std::to_string(i));
        }
        return b;
    }
    if (c.is_glm_dsa()) {
        if (gguf_names || type != "glm_dsa" || !glm_state) {
            throw std::runtime_error("invalid GLM DSA block loader state");
        }
        const std::string lp =
            "model.layers." + std::to_string(i) + ".";
        const std::string ap = lp + "self_attn.";
        auto b = std::make_unique<GlmDsaBlock>();
        b->layer = i;
        b->full_indexer =
            c.indexer_types.at(static_cast<size_t>(i)) == "full";
        b->shared_state = glm_state;
        b->attn_norm = load_dense_gpu(mfq, lp + "input_layernorm.weight");
        b->ffn_norm = load_dense_gpu(
            mfq, lp + "post_attention_layernorm.weight");
        b->q_a_norm = load_dense_gpu(
            mfq, ap + "q_a_layernorm.weight");
        b->kv_a_norm = load_dense_gpu(
            mfq, ap + "kv_a_layernorm.weight");
        std::vector<std::string> first_names = {
            ap + "q_a_proj.weight",
            ap + "kv_a_proj_with_mqa.weight",
        };
        std::vector<std::string> second_names = {
            ap + "q_b_proj.weight",
        };
        if (b->full_indexer) {
            first_names.push_back(ap + "indexer.wk.weight");
            first_names.push_back(ap + "indexer.weights_proj.weight");
            second_names.push_back(ap + "indexer.wq_b.weight");
            b->index_k_norm = load_dense_gpu(
                mfq, ap + "indexer.k_norm.weight");
            b->index_k_bias = load_dense_gpu(
                mfq, ap + "indexer.k_norm.bias");
        }
        b->input_proj = load_quant_group(mfq, first_names);
        b->q_proj = load_quant_group(mfq, second_names);
        b->embed_q = load_nint_moe_gpu(mfq, ap + "embed_q");
        b->unembed_out = load_nint_moe_gpu(mfq, ap + "unembed_out");
        b->o_proj = load_quant_linear(mfq, ap + "o_proj.weight");
        b->ffn = load_ffn(mfq, c, i, false);

        const bool input_shape_ok =
            b->input_proj.outs.size() == (b->full_indexer ? 4u : 2u) &&
            b->input_proj.outs[0] == c.q_lora_rank &&
            b->input_proj.outs[1] == c.kv_lora_rank + c.qk_rope_head_dim &&
            (!b->full_indexer ||
             (b->input_proj.outs[2] == c.index_head_dim &&
              b->input_proj.outs[3] == c.index_n_heads));
        const bool q_shape_ok =
            b->q_proj.outs.size() == (b->full_indexer ? 2u : 1u) &&
            b->q_proj.outs[0] ==
                c.num_attention_heads *
                    (c.qk_nope_head_dim + c.qk_rope_head_dim) &&
            (!b->full_indexer ||
             b->q_proj.outs[1] == c.index_n_heads * c.index_head_dim);
        const bool head_shape_ok =
            b->embed_q.n_experts == c.num_attention_heads &&
            b->embed_q.neuron_len == c.qk_nope_head_dim &&
            b->embed_q.out_per_expert == c.kv_lora_rank &&
            b->unembed_out.n_experts == c.num_attention_heads &&
            b->unembed_out.neuron_len == c.kv_lora_rank &&
            b->unembed_out.out_per_expert == c.v_head_dim &&
            b->o_proj.neuron_len() ==
                c.num_attention_heads * c.v_head_dim &&
            b->o_proj.out() == c.hidden_size;
        if (!input_shape_ok || !q_shape_ok || !head_shape_ok ||
            b->attn_norm.numel() != c.hidden_size ||
            b->ffn_norm.numel() != c.hidden_size ||
            b->q_a_norm.numel() != c.q_lora_rank ||
            b->kv_a_norm.numel() != c.kv_lora_rank ||
            (b->full_indexer &&
             (b->index_k_norm.numel() != c.index_head_dim ||
              b->index_k_bias.numel() != c.index_head_dim))) {
            throw std::runtime_error(
                "GLM DSA tensor shapes disagree with config at layer " +
                std::to_string(i));
        }
        return b;
    }
    if (c.is_gemma4()) {
        if (gguf_names) {
            throw std::runtime_error("Gemma4 runtime expects HF tensor names in MFQ");
        }
        if (type != "full_attention" && type != "sliding_attention") {
            throw std::runtime_error("unsupported Gemma4 layer type: " + type);
        }
        const std::string lp =
            "model.language_model.layers." + std::to_string(i) + ".";
        const std::string ap = lp + "self_attn.";
        auto b = std::make_unique<FullBlock>();
        b->layer = i;
        b->gemma4 = true;
        b->gemma4_moe = c.num_experts > 0;
        b->sliding = type == "sliding_attention";
        b->value_equals_key = !b->sliding && c.attention_k_eq_v;
        b->attention_heads = c.num_attention_heads;
        b->kv_heads = b->sliding ? c.num_key_value_heads : c.num_global_key_value_heads;
        b->attention_head_dim = b->sliding ? c.head_dim : c.global_head_dim;
        b->attention_rotary_dim = b->attention_head_dim;
        b->attention_window = b->sliding ? c.sliding_window : 0;
        b->attention_scale = 1.0;
        b->attention_rope = RopeCache(
            c.max_position_embeddings, b->attention_rotary_dim,
            b->sliding ? c.swa_rope_base : c.rope_base,
            b->attention_head_dim,
            b->sliding ? -1 : (int64_t)std::llround(
                c.full_rotary_factor * (double)b->attention_head_dim / 2.0));

        b->attn_norm = load_dense_gpu(mfq, lp + "input_layernorm.weight");
        b->attn_post_norm = load_dense_gpu(
            mfq, lp + "post_attention_layernorm.weight");
        b->q_norm = load_dense_gpu(mfq, ap + "q_norm.weight");
        b->k_norm = load_dense_gpu(mfq, ap + "k_norm.weight");
        b->v_norm = torch::ones(
            {b->attention_head_dim},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        std::vector<std::string> projections = {
            ap + "q_proj.weight", ap + "k_proj.weight"};
        if (!b->value_equals_key) projections.push_back(ap + "v_proj.weight");
        b->qkv = load_quant_group(mfq, projections, 2);
        b->o = load_quant_linear(mfq, ap + "o_proj.weight");

        b->ffn_norm = load_dense_gpu(
            mfq, lp + "pre_feedforward_layernorm.weight");
        b->ffn_post_norm = load_dense_gpu(
            mfq, lp + "post_feedforward_layernorm.weight");
        b->layer_scale = load_dense_gpu(mfq, lp + "layer_scalar")
            .to(torch::kFloat16).contiguous();
        if (b->gemma4_moe) {
            b->ffn_post_norm_1 = load_dense_gpu(
                mfq, lp + "post_feedforward_layernorm_1.weight");
            b->ffn_pre_norm_2 = load_dense_gpu(
                mfq, lp + "pre_feedforward_layernorm_2.weight");
            b->ffn_post_norm_2 = load_dense_gpu(
                mfq, lp + "post_feedforward_layernorm_2.weight");
        }

        const std::string mp = lp + "mlp.";
        b->ffn.geglu = true;
        b->ffn.down = load_quant_linear(mfq, mp + "down_proj.weight");
        b->ffn.gate_up = load_paired_gate_up(mfq, {
            mp + "gate_proj.weight", mp + "up_proj.weight"},
            b->ffn.down);
        prepare_ffn_workspaces(b->ffn);

        if (!b->gemma4_moe) return b;

        b->gemma_moe_gate_up = load_nint_moe_gpu(
            mfq, lp + "experts.gate_up_proj",
            true, i, "gate_up");
        b->gemma_moe_down = load_nint_moe_gpu(
            mfq, lp + "experts.down_proj",
            true, i, "down");
        b->gemma_router = load_dense_gpu(
            mfq, lp + "router.proj.weight").to(torch::kFloat32).contiguous();
        b->gemma_router_norm_scale = (
            load_dense_gpu(mfq, lp + "router.scale").to(torch::kFloat32) /
            std::sqrt((double)c.hidden_size)).contiguous();
        b->gemma_expert_scale = load_dense_gpu(
            mfq, lp + "router.per_expert_scale").to(torch::kFloat32).contiguous();
        b->gemma_top_k = static_cast<int>(c.num_experts_per_tok);

        if (b->gemma_moe_gate_up.n_experts != c.num_experts ||
            b->gemma_moe_down.n_experts != c.num_experts ||
            b->gemma_moe_gate_up.neuron_len != c.hidden_size ||
            b->gemma_moe_gate_up.out_per_expert != 2 * c.moe_intermediate_size ||
            b->gemma_moe_down.neuron_len != c.moe_intermediate_size ||
            b->gemma_moe_down.out_per_expert != c.hidden_size ||
            b->gemma_router.dim() != 2 ||
            b->gemma_router.size(0) != c.num_experts ||
            b->gemma_router.size(1) != c.hidden_size ||
            b->gemma_router_norm_scale.numel() != c.hidden_size ||
            b->gemma_expert_scale.numel() != c.num_experts) {
            throw std::runtime_error(
                "Gemma4 MoE tensor shapes disagree with config at layer " +
                std::to_string(i));
        }
        return b;
    }
    if (gguf_names) {
        const std::string p = "blk." + std::to_string(i) + ".";
        if (type == "full_attention") {
            auto b = std::make_unique<FullBlock>();
            b->layer = i;
            b->attn_norm = load_dense_gpu(mfq, p + "attn_norm.weight");
            b->ffn_norm = load_dense_gpu(mfq, p + "post_attention_norm.weight");
            b->qkv = load_quant_group(mfq, {
                p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, 2);
            b->o = load_quant_linear(mfq, p + "attn_output.weight");
            if (mfq.records.count(p + "attn_q_norm.weight")) {
                b->q_norm = load_dense_gpu(mfq, p + "attn_q_norm.weight");
            }
            if (mfq.records.count(p + "attn_k_norm.weight")) {
                b->k_norm = load_dense_gpu(mfq, p + "attn_k_norm.weight");
            }
            b->ffn = load_ffn(mfq, c, i, true);
            return b;
        }
        if (type == "linear_attention") {
            auto b = std::make_unique<LinearBlock>();
            b->tiled_v_heads = true;
            b->attn_norm = load_dense_gpu(mfq, p + "attn_norm.weight");
            b->ffn_norm = load_dense_gpu(mfq, p + "post_attention_norm.weight");
            const std::string alpha_name = p + "ssm_alpha.weight";
            const std::string beta_name = p + "ssm_beta.weight";
            const bool alpha_quant = is_quant_dtype(mfq.record(alpha_name).dtype);
            const bool beta_quant = is_quant_dtype(mfq.record(beta_name).dtype);
            if (alpha_quant != beta_quant) {
                throw std::runtime_error(
                    "linear_attention alpha/beta must use the same storage kind");
            }
            if (alpha_quant) {
                b->in_proj = load_quant_group(mfq, {
                    p + "attn_qkv.weight", p + "attn_gate.weight",
                    alpha_name, beta_name});
            } else {
                b->dense_ab_tail = true;
                b->in_proj = load_quant_group(
                    mfq, {p + "attn_qkv.weight", p + "attn_gate.weight"});
                b->ab_proj = make_dense_group({
                    load_dense_gpu(mfq, alpha_name),
                    load_dense_gpu(mfq, beta_name),
                });
            }
            b->conv_weight = load_dense_gpu(mfq, p + "ssm_conv1d.weight");
            if (mfq.records.count(p + "ssm_conv1d.bias")) {
                b->conv_bias = load_dense_gpu(mfq, p + "ssm_conv1d.bias");
            }
            b->dt_bias = load_dense_gpu(mfq, p + "ssm_dt.bias");
            b->a_log = torch::log(-load_dense_gpu(mfq, p + "ssm_a"));
            b->linear_norm = load_dense_gpu(mfq, p + "ssm_norm.weight");
            const std::string out_name = p + "ssm_out.weight";
            if (is_quant_dtype(mfq.record(out_name).dtype)) {
                b->out_proj = load_quant_linear(mfq, out_name);
            } else {
                b->dense_out_proj = true;
                b->out_proj_dense = load_dense_gpu(mfq, out_name);
                if (mfq.record(out_name).dtype == "F16") {
                    b->out_proj_dense =
                        b->out_proj_dense.to(torch::kFloat16).contiguous();
                }
                if (b->out_proj_dense.dim() != 2) {
                    throw std::runtime_error(
                        "dense linear_attention output projection must be 2D");
                }
            }
            b->ffn = load_ffn(mfq, c, i, true);
            return b;
        }
        throw std::runtime_error("unsupported layer type: " + type);
    }
    std::string lp = "model.language_model.layers." + std::to_string(i) + ".";
    if (type == "full_attention") {
        auto b = std::make_unique<FullBlock>();
        b->layer = i;
        b->attn_norm = load_dense_gpu(mfq, lp + "input_layernorm.weight");
        b->ffn_norm = load_dense_gpu(mfq, lp + "post_attention_layernorm.weight");
        const std::string ap = lp + "self_attn.";
        b->qkv = load_quant_group(mfq, {
            ap + "q_proj.weight", ap + "k_proj.weight", ap + "v_proj.weight"}, 2);
        b->o = load_quant_linear(mfq, ap + "o_proj.weight");
        if (mfq.records.count(ap + "q_norm.weight")) b->q_norm = load_dense_gpu(mfq, ap + "q_norm.weight");
        if (mfq.records.count(ap + "k_norm.weight")) b->k_norm = load_dense_gpu(mfq, ap + "k_norm.weight");
        b->ffn = load_ffn(mfq, c, i, false);
        return b;
    }
    if (type == "linear_attention") {
        auto b = std::make_unique<LinearBlock>();
        b->attn_norm = load_dense_gpu(mfq, lp + "input_layernorm.weight");
        b->ffn_norm = load_dense_gpu(mfq, lp + "post_attention_layernorm.weight");
        const std::string sp = lp + "linear_attn.";
        if (mfq.records.count(sp + "in_proj_qk.weight") && mfq.records.count(sp + "in_proj_v.weight")) {
            b->split_in_proj = true;
            b->qkv_proj = load_quant_group(mfq, {sp + "in_proj_qk.weight", sp + "in_proj_v.weight"});
            if (is_quant_dtype(mfq.record(sp + "in_proj_z.weight").dtype)) {
                b->z_proj = load_quant_linear(mfq, sp + "in_proj_z.weight");
                bool a_nint = is_quant_dtype(mfq.record(sp + "in_proj_a.weight").dtype);
                bool b_nint = is_quant_dtype(mfq.record(sp + "in_proj_b.weight").dtype);
                if (a_nint != b_nint) throw std::runtime_error("linear_attn a/b must use the same storage kind");
                b->ab_is_nint = a_nint;
                if (b->ab_is_nint) {
                    b->ab_nint_proj = load_quant_group(
                        mfq, {sp + "in_proj_a.weight", sp + "in_proj_b.weight"});
                } else {
                    b->ab_proj = make_dense_group({
                        load_dense_gpu(mfq, sp + "in_proj_a.weight"),
                        load_dense_gpu(mfq, sp + "in_proj_b.weight"),
                    });
                }
            } else {
                b->split_dense_zab = true;
                b->zab_proj = make_dense_group({
                    load_dense_gpu(mfq, sp + "in_proj_z.weight"),
                    load_dense_gpu(mfq, sp + "in_proj_a.weight"),
                    load_dense_gpu(mfq, sp + "in_proj_b.weight"),
                });
            }
        } else {
            b->in_proj = load_quant_group(mfq, {
                sp + "in_proj_qkv.weight", sp + "in_proj_z.weight",
                sp + "in_proj_a.weight", sp + "in_proj_b.weight"});
        }
        b->conv_weight = load_dense_gpu(mfq, sp + "conv1d.weight");
        if (mfq.records.count(sp + "conv1d.bias")) b->conv_bias = load_dense_gpu(mfq, sp + "conv1d.bias");
        b->dt_bias = load_dense_gpu(mfq, sp + "dt_bias");
        b->a_log = load_dense_gpu(mfq, sp + "A_log");
        b->linear_norm = load_dense_gpu(mfq, sp + "norm.weight");
        b->out_proj = load_quant_linear(mfq, sp + "out_proj.weight");
        b->ffn = load_ffn(mfq, c, i, false);
        return b;
    }
    throw std::runtime_error("unsupported layer type: " + type);
}

struct Model {
    Config c;
    RopeCache rope;
    std::unordered_map<int, RopeCache> device_ropes;
    QuantLinear embed;
    std::vector<std::unique_ptr<Block>> blocks;
    torch::Tensor output_norm;
    QuantLinear lm_head;
    torch::Tensor dsv4_hc_head_fn;
    torch::Tensor dsv4_hc_head_scale;
    torch::Tensor dsv4_hc_head_base;
    int64_t cache_pos = 0;

    torch::Tensor embed_forward(torch::Tensor ids) const {
        auto token_ids = ids.contiguous().to(torch::kCUDA, torch::kInt64);
        if (embed.is_dense()) {
            auto output_shape = token_ids.sizes().vec();
            output_shape.push_back(embed.dense.size(1));
            return embed.dense.index_select(
                0, token_ids.reshape({-1})).reshape(output_shape);
        }
        if (embed.is_nvq()) {
            return nvq_embedding(embed.nvq.w, token_ids);
        }
        TORCH_CHECK(
            embed.is_nint(),
            "MXFP8 token embeddings are not supported by the CUDA runtime");
        if (embed.nint.w.q8_zero) {
            return nint8_zero_embedding_lookup_cuda(
                embed.nint.w.q_packed, embed.nint.w.q8_zero_scale,
                token_ids, embed.nint.w.neuron_len);
        }
        if (embed.nint.w.bits == 4) {
            return nint_embedding_lookup_packed_compact_cuda(
                embed.nint.w.q_packed, embed.nint.w.sub_scale, embed.nint.w.sub_min,
                embed.nint.w.neuron_scale, embed.nint.w.neuron_min,
                token_ids, embed.nint.w.neuron_len, embed.nint.w.gs);
        }
        return nint_embedding_lookup_packed_compact_bits_cuda(
            embed.nint.w.q_packed, embed.nint.w.sub_scale, embed.nint.w.sub_min,
            embed.nint.w.neuron_scale, embed.nint.w.neuron_min,
            token_ids, embed.nint.w.neuron_len, embed.nint.w.gs, embed.nint.w.bits);
    }

    void reset(int64_t B) {
        cache_pos = 0;
        for (auto & b : blocks) {
            c10::cuda::CUDAGuard guard(b->cuda_device);
            b->reset(B);
        }
    }

    torch::Tensor finalize_hidden(torch::Tensor x, int64_t B, int64_t T) {
        if (c.is_dsv4()) {
            x = g_profiler.measure("model.dsv4_hc_head", [&]() {
                auto flat = x.flatten(2).to(torch::kFloat32);
                auto inverse_rms = torch::rsqrt(
                    flat.square().mean(-1, true) + c.rms_norm_eps);
                auto mixes = torch::matmul(
                    flat, dsv4_hc_head_fn.transpose(0, 1)) *
                    inverse_rms;
                auto pre = torch::sigmoid(
                    mixes * dsv4_hc_head_scale +
                    dsv4_hc_head_base) + c.hc_eps;
                return (
                    pre.unsqueeze(-1) *
                    flat.reshape({B, T, c.hc_mult, c.hidden_size}))
                    .sum(2).to(torch::kFloat16).contiguous();
            });
        }
        return g_profiler.measure("model.output_norm", [&]() {
            return qwen_rms_norm(
                x.reshape({B * T, c.hidden_size}).to(torch::kFloat32),
                output_norm, c).reshape({B, T, c.hidden_size});
        });
    }

    torch::Tensor hidden_forward(torch::Tensor ids,
                                 c10::optional<torch::Tensor> pos_override = c10::nullopt,
                                 c10::optional<torch::Tensor> seq_len = c10::nullopt,
                                 std::vector<torch::Tensor> * block_trace = nullptr) {
        const int primary = g_layer_placement.primary_device();
        c10::cuda::CUDAGuard primary_guard(primary);
        ids = tensor_to_cuda_device(
            ids.to(torch::kInt64), primary);
        if (ids.dim() == 1) ids = ids.unsqueeze(0);
        int64_t B = ids.size(0), T = ids.size(1);
        if (cache_pos == 0) reset(B);
        auto pos = pos_override.has_value()
            ? pos_override.value()
            : torch::arange(cache_pos, cache_pos + T, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
        auto x = g_profiler.measure("model.embed", [&]() { return embed_forward(ids); });
        if (c.is_gemma4()) {
            x = g_profiler.measure("model.embed_scale", [&]() {
                return x * c.embed_scale;
            });
        }
        if (c.is_dsv4()) {
            x = x.to(torch::kFloat16)
                .unsqueeze(2)
                .expand({B, T, c.hc_mult, c.hidden_size})
                .contiguous();
        }
        if (block_trace != nullptr) block_trace->push_back(x.to(torch::kFloat32).clone());
        for (auto & b : blocks) {
            c10::cuda::CUDAGuard block_guard(b->cuda_device);
            auto local_ids = tensor_to_cuda_device(ids, b->cuda_device);
            auto local_pos = tensor_to_cuda_device(pos, b->cuda_device);
            c10::optional<torch::Tensor> local_seq_len = c10::nullopt;
            if (seq_len.has_value()) {
                local_seq_len = tensor_to_cuda_device(
                    seq_len.value(), b->cuda_device);
            }
            x = tensor_to_cuda_device(x, b->cuda_device);
            b->set_token_ids(local_ids);
            const RopeCache & active_rope = device_ropes.empty()
                ? rope : device_ropes.at(b->cuda_device);
            x = b->forward(
                x, local_pos, cache_pos, local_seq_len, c, active_rope);
            if (block_trace != nullptr) {
                block_trace->push_back(
                    tensor_to_cuda_device(x, primary)
                        .to(torch::kFloat32).clone());
            }
        }
        if (!pos_override.has_value()) cache_pos += T;
        x = tensor_to_cuda_device(x, primary);
        return finalize_hidden(x, B, T);
    }

    torch::Tensor logits_from_hidden(torch::Tensor y) {
        auto logits = g_profiler.measure("model.lm_head", [&]() { return lm_head.forward(y); });
        if (c.final_logit_softcapping > 0.0) {
            logits = torch::tanh(logits / c.final_logit_softcapping) *
                c.final_logit_softcapping;
        }
        return logits;
    }

    torch::Tensor forward(torch::Tensor ids) {
        return logits_from_hidden(hidden_forward(ids));
    }

    torch::Tensor last_logits(torch::Tensor ids) {
        c10::optional<torch::Tensor> seq_len = c10::nullopt;
        const int64_t token_count = ids.dim() == 1 ? ids.size(0) : ids.size(1);
        const int64_t batch_size = ids.dim() == 1 ? 1 : ids.size(0);
        if (cache_pos > 0 && token_count == 1) {
            seq_len = torch::full(
                {batch_size}, cache_pos + 1,
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
        }
        auto y = hidden_forward(ids, c10::nullopt, seq_len);
        auto last = y.index({Slice(), -1, Slice()}).to(torch::kFloat16).contiguous();
        auto logits = lm_head.forward(last);
        if (c.final_logit_softcapping > 0.0) {
            logits = torch::tanh(logits / c.final_logit_softcapping) *
                c.final_logit_softcapping;
        }
        return logits;
    }

    torch::Tensor last_logits_static(torch::Tensor ids, torch::Tensor pos, torch::Tensor seq_len) {
        auto y = hidden_forward(ids, pos, seq_len);
        auto last = y.index({Slice(), -1, Slice()}).to(torch::kFloat16).contiguous();
        auto logits = lm_head.forward(last);
        if (c.final_logit_softcapping > 0.0) {
            logits = torch::tanh(logits / c.final_logit_softcapping) *
                c.final_logit_softcapping;
        }
        return logits;
    }

    torch::Tensor next_token(torch::Tensor ids) {
        c10::optional<torch::Tensor> seq_len = c10::nullopt;
        const int64_t token_count = ids.dim() == 1 ? ids.size(0) : ids.size(1);
        const int64_t batch_size = ids.dim() == 1 ? 1 : ids.size(0);
        if (cache_pos > 0 && token_count == 1) {
            seq_len = torch::full(
                {batch_size}, cache_pos + 1,
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
        }
        auto y = hidden_forward(ids, c10::nullopt, seq_len);
        return next_token_from_hidden(y);
    }

    torch::Tensor next_token_static(torch::Tensor ids, torch::Tensor pos, torch::Tensor seq_len) {
        auto y = hidden_forward(ids, pos, seq_len);
        return next_token_from_hidden(y);
    }

    torch::Tensor next_token_from_hidden(torch::Tensor y) {
        auto last = y.index({Slice(), -1, Slice()}).to(torch::kFloat16).contiguous();
        const char* use_argmax = std::getenv("MFQ_LM_HEAD_ARGMAX");
        const char* disable_argmax = std::getenv("MFQ_DISABLE_LM_HEAD_ARGMAX");
        bool lm_head_argmax = (disable_argmax == nullptr || disable_argmax[0] != '1') ||
                              (use_argmax != nullptr && use_argmax[0] == '1');
        NintWeight * lm_nint =
            lm_head.is_nint() && !lm_head.tensor_parallel()
                ? &lm_head.nint.w : nullptr;
        if (lm_head_argmax && lm_nint != nullptr &&
            !lm_nint->q5_exec &&
            last.size(0) == 1 &&
            ((lm_nint->bits == 5 && lm_nint->gs == 28) ||
             (lm_nint->bits == 6 && (lm_nint->gs == 24 || lm_nint->gs == 26)))) {
            return g_profiler.measure("model.lm_head_argmax", [&]() {
                Workspace & ws = lm_nint->workspace(1);
                lm_nint->ensure_argmax_workspace(ws);
                return nint_gemv_packed_bits_argmax_ws_cuda(
                    lm_nint->q_packed, lm_nint->sub_scale, lm_nint->sub_min,
                    lm_nint->neuron_scale, lm_nint->neuron_min, pad_last(last, lm_nint->neuron_len),
                    lm_nint->gs, lm_nint->bits, ws.qx, ws.xscale, ws.xsum,
                    ws.argmax_vals, ws.argmax_idxs);
            });
        }
        const char* use_outbuf = std::getenv("MFQ_LM_HEAD_OUTBUF");
        if (use_outbuf != nullptr && use_outbuf[0] == '1' && lm_nint != nullptr &&
            last.size(0) == 1 && lm_nint->bits == 6 && lm_nint->gs == 26) {
            auto logits = g_profiler.measure("model.lm_head_outbuf", [&]() {
                Workspace & ws = lm_nint->workspace(1);
                lm_nint->ensure_output_workspace(ws);
                return nint_gemv_packed_bits_m1_out_ws_cuda(
                    lm_nint->q_packed, lm_nint->sub_scale, lm_nint->sub_min,
                    lm_nint->neuron_scale, lm_nint->neuron_min, pad_last(last, lm_nint->neuron_len),
                    lm_nint->gs, lm_nint->bits, ws.qx, ws.xscale, ws.xsum, ws.out_buf);
            });
            return sample_greedy_cuda(logits);
        }
        auto logits = g_profiler.measure("model.lm_head", [&]() { return lm_head.forward(last); });
        return sample_greedy_cuda(logits.contiguous().view({last.size(0), -1}));
    }
};

static Model load_model(const std::string & mfq_path, const std::string & config_path,
                        int64_t context_size_override = 0,
                        bool load_blocks = true) {
    Model m;
    MfqFile mfq(mfq_path);
    m.c = load_config(mfq, config_path);
    g_layer_placement.prepare(m.c.num_hidden_layers);
    g_layer_placement.load_device =
        g_layer_placement.primary_device();
    c10::cuda::CUDAGuard model_guard(
        g_layer_placement.primary_device());
    if (!g_dsv4_cpu_offload_layers.empty()) {
        if (!m.c.is_dsv4()) {
            throw std::runtime_error(
                "--cpu-offload-layers currently supports DeepSeek V4 only");
        }
        for (int layer : g_dsv4_cpu_offload_layers) {
            if (layer < 0 || layer >= m.c.num_hidden_layers) {
                throw std::runtime_error(
                    "CPU-offload layer is outside the model: " +
                    std::to_string(layer));
            }
        }
        g_dsv4_cpu_offload_host_bytes = 0;
        g_mfq_drop_file_cache = true;
    }
    if (context_size_override > 0) {
        if (context_size_override > m.c.max_position_embeddings) {
            throw std::runtime_error("--ctx-size exceeds max_position_embeddings");
        }
        m.c.max_position_embeddings = context_size_override;
    }
    if (!m.c.is_gemma4() && !m.c.is_dsv4()) {
        m.rope = RopeCache(m.c);
        if (g_layer_placement.enabled()) {
            for (int device : g_layer_placement.devices) {
                c10::cuda::CUDAGuard rope_guard(device);
                m.device_ropes.emplace(device, RopeCache(m.c));
            }
        }
    }
    const bool gguf_names = mfq.records.count("token_embd.weight") != 0;
    m.c.norm_weight_offset =
        (gguf_names || m.c.is_gemma4() || m.c.is_glm_dsa()) ? 0.0 : 1.0;
    const std::string hf_prefix = m.c.is_glm_dsa()
        ? "model." : "model.language_model.";
    const std::string embed_name = gguf_names
        ? "token_embd.weight" : hf_prefix + "embed_tokens.weight";
    const std::string norm_name = gguf_names
        ? "output_norm.weight" : hf_prefix + "norm.weight";
    const std::string output_name = gguf_names ? "output.weight" : "lm_head.weight";
    m.embed = load_quant_linear(mfq, embed_name);
    m.output_norm = load_dense_gpu(mfq, norm_name);
    if (m.c.is_dsv4()) {
        m.dsv4_hc_head_fn = load_dense_gpu(
            mfq, "output_hc_fn.weight")
            .to(torch::kFloat32).contiguous();
        m.dsv4_hc_head_scale = load_dense_gpu(
            mfq, "output_hc_scale.weight")
            .to(torch::kFloat32).contiguous();
        m.dsv4_hc_head_base = load_dense_gpu(
            mfq, "output_hc_base.weight")
            .to(torch::kFloat32).contiguous();
    }
    if (m.c.tie_word_embeddings || !mfq.records.count(output_name)) {
        m.lm_head = g_tensor_parallel.enabled()
            ? load_quant_linear(
                mfq, embed_name,
                TensorParallelAxis::Output)
            : m.embed;
    } else {
        m.lm_head = load_quant_linear(
            mfq, output_name,
            TensorParallelAxis::Output);
    }
    if (g_kl_mmq_mode == KlMmqMode::Default &&
        nint5_q5_exec_enabled() &&
        m.lm_head.is_nint() &&
        ((m.lm_head.tensor_parallel() &&
          !m.lm_head.tensor_parallel_shards.empty() &&
          m.lm_head.tensor_parallel_shards.front().nint.bits == 5 &&
          m.lm_head.tensor_parallel_shards.front().nint.gs == 28) ||
         (!m.lm_head.tensor_parallel() &&
          m.lm_head.nint.w.bits == 5 &&
          m.lm_head.nint.w.gs == 28))) {
        std::cerr << "repacking lm_head NINT5 gs28 execution layout" << std::endl;
        if (m.lm_head.tensor_parallel()) {
            for (auto & shard : m.lm_head.tensor_parallel_shards) {
                c10::cuda::CUDAGuard guard(shard.device);
                enable_nint5_q5_exec(shard.nint);
            }
        } else {
            enable_nint5_q5_exec(m.lm_head.nint.w);
        }
    }
    if (load_blocks) {
        m.blocks.reserve((size_t)m.c.num_hidden_layers);
        std::unordered_map<int, std::shared_ptr<GlmDsaSharedState>>
            glm_states;
        std::unordered_map<int, std::shared_ptr<Dsv4SharedState>>
            dsv4_states;
        for (int i = 0; i < m.c.num_hidden_layers; ++i) {
            const int device = g_layer_placement.device_for_layer(i);
            g_layer_placement.load_device = device;
            c10::cuda::CUDAGuard layer_guard(device);
            std::shared_ptr<GlmDsaSharedState> glm_state;
            std::shared_ptr<Dsv4SharedState> dsv4_state;
            if (m.c.is_glm_dsa()) {
                auto & state = glm_states[device];
                if (!state) state = std::make_shared<GlmDsaSharedState>();
                glm_state = state;
            }
            if (m.c.is_dsv4()) {
                auto & state = dsv4_states[device];
                if (!state) state = std::make_shared<Dsv4SharedState>();
                dsv4_state = state;
            }
            std::cerr << "loading layer " << i << " " << m.c.layer_types[(size_t)i] << std::endl;
            auto block = load_block(
                mfq, m.c, i, m.c.layer_types[(size_t)i], gguf_names,
                glm_state, dsv4_state);
            block->cuda_device = device;
            m.blocks.push_back(std::move(block));
        }
    }
    g_layer_placement.load_device =
        g_layer_placement.primary_device();
    if (g_moe_expert_cache &&
            g_moe_expert_cache->has_sources() &&
            !g_moe_expert_cache->finalized()) {
        g_moe_expert_cache->finalize();
    }
    return m;
}

static std::vector<int64_t> parse_ids(const std::string & s) {
    std::vector<int64_t> ids;
    std::regex re("-?\\d+");
    for (auto it = std::sregex_iterator(s.begin(), s.end(), re); it != std::sregex_iterator(); ++it) {
        ids.push_back(std::stoll((*it)[0].str()));
    }
    if (ids.empty()) throw std::runtime_error("--ids must contain at least one token id");
    return ids;
}

static std::vector<std::string> split_csv_values(
        const std::string & value,
        const char * option) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(
            std::remove_if(
                item.begin(), item.end(),
                [](unsigned char ch) {
                    return std::isspace(ch) != 0;
                }),
            item.end());
        if (item.empty()) {
            throw std::runtime_error(
                std::string(option) +
                " contains an empty item");
        }
        result.push_back(std::move(item));
    }
    if (result.empty()) {
        throw std::runtime_error(
            std::string(option) +
            " requires at least one value");
    }
    return result;
}

static void configure_tensor_parallel(
        const std::string & devices_arg,
        const std::string & split_arg,
        bool allow_duplicate_devices = false) {
    g_tensor_parallel_collectives.reset();
    g_tensor_parallel = {};
    if (devices_arg.empty()) {
        if (!split_arg.empty()) {
            throw std::runtime_error(
                "--tensor-split requires --tensor-parallel");
        }
        MFQ_CUDA_CHECK(cudaSetDevice(0));
        return;
    }

    const auto device_values =
        split_csv_values(
            devices_arg, "--tensor-parallel");
    if (device_values.size() == 1 &&
        devices_arg.find(',') == std::string::npos) {
        const int count = std::stoi(device_values.front());
        if (count < 2) {
            throw std::runtime_error(
                "--tensor-parallel device count must be at least 2");
        }
        g_tensor_parallel.devices.resize(
            static_cast<size_t>(count));
        std::iota(
            g_tensor_parallel.devices.begin(),
            g_tensor_parallel.devices.end(), 0);
    } else {
        for (const auto & item : device_values) {
            g_tensor_parallel.devices.push_back(
                std::stoi(item));
        }
        if (g_tensor_parallel.devices.size() < 2) {
            throw std::runtime_error(
                "--tensor-parallel requires at least two devices");
        }
    }
    g_tensor_parallel.allow_duplicate_devices =
        allow_duplicate_devices;

    int available = 0;
    MFQ_CUDA_CHECK(cudaGetDeviceCount(&available));
    std::unordered_set<int> unique_devices;
    for (int device : g_tensor_parallel.devices) {
        if (device < 0 || device >= available) {
            throw std::runtime_error(
                "tensor-parallel CUDA device is unavailable: " +
                std::to_string(device));
        }
        if (!allow_duplicate_devices &&
            !unique_devices.insert(device).second) {
            throw std::runtime_error(
                "tensor-parallel CUDA devices must be unique");
        }
    }

    if (!split_arg.empty()) {
        for (const auto & item :
             split_csv_values(
                 split_arg, "--tensor-split")) {
            g_tensor_parallel.split.push_back(
                std::stod(item));
        }
        if (g_tensor_parallel.split.size() !=
            g_tensor_parallel.devices.size()) {
            throw std::runtime_error(
                "--tensor-split count must match "
                "--tensor-parallel devices");
        }
    }

    for (int source : unique_devices) {
        for (int destination : unique_devices) {
            if (source == destination) continue;
            int can_access = 0;
            MFQ_CUDA_CHECK(cudaDeviceCanAccessPeer(
                &can_access, source, destination));
            if (!can_access) continue;
            MFQ_CUDA_CHECK(cudaSetDevice(source));
            const cudaError_t status =
                cudaDeviceEnablePeerAccess(
                    destination, 0);
            if (status != cudaSuccess &&
                status != cudaErrorPeerAccessAlreadyEnabled) {
                throw std::runtime_error(
                    "failed to enable tensor-parallel peer access from CUDA " +
                    std::to_string(source) + " to CUDA " +
                    std::to_string(destination) + ": " +
                    cudaGetErrorString(status));
            }
            if (status == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
            }
        }
    }
    g_tensor_parallel_collectives.configure(
        g_tensor_parallel.devices,
        allow_duplicate_devices);
    MFQ_CUDA_CHECK(
        cudaSetDevice(
            g_tensor_parallel.primary_device()));
    std::cerr << "tensor_parallel devices=";
    for (size_t index = 0;
         index < g_tensor_parallel.devices.size();
         ++index) {
        if (index) std::cerr << ',';
        std::cerr << g_tensor_parallel.devices[index];
    }
    if (!g_tensor_parallel.split.empty()) {
        std::cerr << " split=";
        for (size_t index = 0;
             index < g_tensor_parallel.split.size();
             ++index) {
            if (index) std::cerr << ',';
            std::cerr << g_tensor_parallel.split[index];
        }
    }
    std::cerr << " collective_backend="
              << (g_tensor_parallel_collectives.collectives_enabled
                  ? "nccl" : "serial")
              << '\n';
}

static void configure_layer_placement(
        const std::string & devices_arg,
        const std::string & split_arg) {
    g_layer_placement = {};
    if (devices_arg.empty()) {
        if (!split_arg.empty()) {
            throw std::runtime_error(
                "--layer-split requires --layer-parallel");
        }
        return;
    }
    if (g_tensor_parallel.enabled()) {
        throw std::runtime_error(
            "--layer-parallel cannot be combined with --tensor-parallel");
    }

    const auto device_values =
        split_csv_values(devices_arg, "--layer-parallel");
    if (device_values.size() == 1 &&
            devices_arg.find(',') == std::string::npos) {
        const int count = std::stoi(device_values.front());
        if (count < 2) {
            throw std::runtime_error(
                "--layer-parallel device count must be at least 2");
        }
        g_layer_placement.devices.resize(static_cast<size_t>(count));
        std::iota(
            g_layer_placement.devices.begin(),
            g_layer_placement.devices.end(), 0);
    } else {
        for (const auto & item : device_values) {
            g_layer_placement.devices.push_back(std::stoi(item));
        }
        if (g_layer_placement.devices.size() < 2) {
            throw std::runtime_error(
                "--layer-parallel requires at least two devices");
        }
    }

    int available = 0;
    MFQ_CUDA_CHECK(cudaGetDeviceCount(&available));
    std::unordered_set<int> unique_devices;
    for (int device : g_layer_placement.devices) {
        if (device < 0 || device >= available) {
            throw std::runtime_error(
                "layer-placement CUDA device is unavailable: " +
                std::to_string(device));
        }
        if (!unique_devices.insert(device).second) {
            throw std::runtime_error(
                "layer-placement CUDA devices must be unique");
        }
    }
    if (!split_arg.empty()) {
        for (const auto & item :
             split_csv_values(split_arg, "--layer-split")) {
            g_layer_placement.split.push_back(std::stod(item));
        }
        if (g_layer_placement.split.size() !=
                g_layer_placement.devices.size()) {
            throw std::runtime_error(
                "--layer-split count must match --layer-parallel devices");
        }
    }
    MFQ_CUDA_CHECK(cudaSetDevice(g_layer_placement.primary_device()));
    std::cerr << "layer_parallel devices=";
    for (size_t index = 0;
         index < g_layer_placement.devices.size(); ++index) {
        if (index) std::cerr << ',';
        std::cerr << g_layer_placement.devices[index];
    }
    if (!g_layer_placement.split.empty()) {
        std::cerr << " split=";
        for (size_t index = 0;
             index < g_layer_placement.split.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << g_layer_placement.split[index];
        }
    }
    std::cerr << '\n';
}

static std::vector<int64_t> load_ids_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open --ids-file: " + path);
    const std::streamsize bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(int32_t)) != 0) {
        throw std::runtime_error("--ids-file must contain raw int32 token ids");
    }
    input.seekg(0);
    std::vector<int32_t> stored(
        static_cast<size_t>(bytes / static_cast<std::streamsize>(sizeof(int32_t))));
    input.read(reinterpret_cast<char *>(stored.data()), bytes);
    if (!input) throw std::runtime_error("truncated --ids-file: " + path);
    return std::vector<int64_t>(stored.begin(), stored.end());
}

static std::unordered_set<int> parse_layer_ranges(
        const std::string & value) {
    std::unordered_set<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(
            std::remove_if(
                item.begin(), item.end(),
                [](unsigned char ch) { return std::isspace(ch) != 0; }),
            item.end());
        if (item.empty()) {
            throw std::runtime_error(
                "--cpu-offload-layers contains an empty item");
        }
        const size_t dash = item.find('-');
        int first = 0;
        int last = 0;
        if (dash == std::string::npos) {
            first = last = std::stoi(item);
        } else {
            if (dash == 0 || dash + 1 >= item.size() ||
                item.find('-', dash + 1) != std::string::npos) {
                throw std::runtime_error(
                    "invalid --cpu-offload-layers range: " + item);
            }
            first = std::stoi(item.substr(0, dash));
            last = std::stoi(item.substr(dash + 1));
        }
        if (first < 0 || last < first) {
            throw std::runtime_error(
                "invalid --cpu-offload-layers range: " + item);
        }
        for (int layer = first; layer <= last; ++layer) {
            result.insert(layer);
        }
    }
    if (result.empty()) {
        throw std::runtime_error(
            "--cpu-offload-layers requires at least one layer");
    }
    return result;
}

static int run_prefill_sweep(
    Model & model,
    const std::vector<int64_t> & sizes,
    int repeats) {
    if (repeats < 1) throw std::runtime_error("--prefill-sweep-reps must be positive");
    const int64_t max_m = *std::max_element(sizes.begin(), sizes.end());
    if (max_m > model.c.max_position_embeddings) {
        throw std::runtime_error("--prefill-sweep exceeds the configured context size");
    }

    std::vector<int64_t> token_ids((size_t)max_m);
    const int64_t token_span = std::max<int64_t>(1, std::min<int64_t>(1024, model.c.vocab_size - 2));
    for (int64_t i = 0; i < max_m; ++i) token_ids[(size_t)i] = 1 + i % token_span;
    auto all_ids = torch::tensor(
        token_ids, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)).unsqueeze(0);

    for (int64_t m : sizes) {
        auto ids = all_ids.narrow(1, 0, m);
        for (int warmup = 0; warmup < 2; ++warmup) {
            model.reset(1);
            (void)model.last_logits(ids);
            torch::cuda::synchronize();
        }

        std::vector<double> elapsed_ms;
        elapsed_ms.reserve((size_t)repeats);
        int64_t top = -1;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            model.reset(1);
            auto started = std::chrono::steady_clock::now();
            auto logits = model.last_logits(ids);
            torch::cuda::synchronize();
            auto ended = std::chrono::steady_clock::now();
            elapsed_ms.push_back(std::chrono::duration<double, std::milli>(ended - started).count());
            if (repeat == 0) top = logits.argmax(-1).item<int64_t>();
        }
        std::sort(elapsed_ms.begin(), elapsed_ms.end());
        const double median_ms = elapsed_ms[elapsed_ms.size() / 2];
        std::cout << "prefill_sweep_m=" << m
                  << " median_ms=" << median_ms
                  << " min_ms=" << elapsed_ms.front()
                  << " max_ms=" << elapsed_ms.back()
                  << " tok_per_s=" << (1000.0 * (double)m / median_ms)
                  << " top=" << top << "\n";
    }
    return 0;
}

static int run_block_trace_compare(
    Model & test,
    const std::string & reference_mfq,
    const std::string & config_path,
    int64_t context_size,
    torch::Tensor ids)
{
    Model reference = load_model(reference_mfq, config_path, context_size);
    std::vector<torch::Tensor> test_trace;
    std::vector<torch::Tensor> reference_trace;

    test.reset(ids.size(0));
    auto test_hidden = test.hidden_forward(ids, c10::nullopt, c10::nullopt, &test_trace);
    reference.reset(ids.size(0));
    auto reference_hidden = reference.hidden_forward(
        ids, c10::nullopt, c10::nullopt, &reference_trace);
    torch::cuda::synchronize();

    if (test_trace.size() != reference_trace.size()) {
        throw std::runtime_error("block trace stage count mismatch");
    }
    for (size_t i = 0; i < test_trace.size(); ++i) {
        auto ref = reference_trace[i].reshape({-1}).to(torch::kFloat64);
        auto got = test_trace[i].reshape({-1}).to(torch::kFloat64);
        if (ref.numel() != got.numel()) {
            throw std::runtime_error("block trace tensor size mismatch");
        }
        auto ref_norm = ref.norm();
        auto got_norm = got.norm();
        const double ref_norm_value = ref_norm.item<double>();
        const double denominator = std::max(ref_norm_value, 1.0e-30);
        const double relative_l2 = (got - ref).norm().item<double>() / denominator;
        const double cosine = torch::dot(ref, got).item<double>() /
            std::max(ref_norm_value * got_norm.item<double>(), 1.0e-30);
        const double norm_ratio = got_norm.item<double>() / denominator;
        const double reference_rms = ref.square().mean().sqrt().item<double>();
        const double test_rms = got.square().mean().sqrt().item<double>();
        const std::string stage = i == 0
            ? "embedding"
            : "block_" + std::to_string(i - 1);
        std::cout << "block_trace stage=" << stage
                  << " relative_l2=" << relative_l2
                  << " cosine=" << cosine
                  << " norm_ratio=" << norm_ratio
                  << " reference_rms=" << reference_rms
                  << " test_rms=" << test_rms << "\n";
    }

    auto reference_logits = reference.lm_head.forward(reference_hidden).to(torch::kFloat32);
    auto test_logits = test.lm_head.forward(test_hidden).to(torch::kFloat32);
    double kl_sum = 0.0;
    int64_t same_top = 0;
    int64_t rows = 0;
    const int64_t tokens = reference_logits.numel() / reference_logits.size(-1);
    auto ref2 = reference_logits.reshape({tokens, -1});
    auto got2 = test_logits.reshape({tokens, -1});
    for (int64_t start = 0; start < tokens; start += 8) {
        const int64_t end = std::min(start + 8, tokens);
        auto ref_chunk = ref2.index({Slice(start, end)});
        auto got_chunk = got2.index({Slice(start, end)});
        auto ref_logp = torch::log_softmax(ref_chunk, -1);
        auto got_logp = torch::log_softmax(got_chunk, -1);
        kl_sum += (ref_logp.exp() * (ref_logp - got_logp)).sum(-1)
            .to(torch::kFloat64).sum().item<double>();
        same_top += ref_chunk.argmax(-1).eq(got_chunk.argmax(-1)).sum().item<int64_t>();
        rows += end - start;
    }
    const double logits_relative_l2 =
        (test_logits.to(torch::kFloat64) - reference_logits.to(torch::kFloat64)).norm().item<double>() /
        std::max(reference_logits.to(torch::kFloat64).norm().item<double>(), 1.0e-30);
    std::cout << "block_trace_logits kld=" << (kl_sum / std::max<int64_t>(rows, 1))
              << " same_top=" << ((double)same_top / std::max<int64_t>(rows, 1))
              << " relative_l2=" << logits_relative_l2 << "\n";
    return 0;
}

static int run_block_trace_dump(
    Model & model,
    const std::string & output_dir,
    torch::Tensor ids,
    int64_t token_start,
    int64_t token_count)
{
    const int64_t total_tokens = ids.size(1);
    if (token_start < 0 || token_start >= total_tokens) {
        throw std::runtime_error("--dump-block-trace-start is outside the token range");
    }
    if (token_count <= 0) token_count = total_tokens - token_start;
    if (token_count > total_tokens - token_start) {
        throw std::runtime_error("--dump-block-trace-count exceeds the token range");
    }
    const std::filesystem::path root(output_dir);
    std::error_code error;
    if (!std::filesystem::create_directories(root, error) || error) {
        throw std::runtime_error(
            "block trace output directory must be new: " + output_dir);
    }

    model.reset(ids.size(0));
    std::vector<torch::Tensor> trace;
    auto final_hidden = model.hidden_forward(
        ids, c10::nullopt, c10::nullopt, &trace);
    torch::cuda::synchronize();

    auto ids_cpu = ids.to(torch::kCPU, torch::kInt32).contiguous();
    {
        std::ofstream output(root / "tokens.i32", std::ios::binary);
        if (!output) throw std::runtime_error("cannot create block trace token file");
        output.write(
            reinterpret_cast<const char *>(ids_cpu.data_ptr<int32_t>()),
            static_cast<std::streamsize>(ids_cpu.nbytes()));
        if (!output) throw std::runtime_error("failed to write block trace tokens");
    }

    std::ofstream metadata(root / "trace_meta.jsonl");
    if (!metadata) throw std::runtime_error("cannot create block trace metadata");
    auto token_slice = [&](torch::Tensor value) {
        if (value.dim() >= 3 && value.size(1) == total_tokens) {
            return value.narrow(1, token_start, token_count);
        }
        return value;
    };
    for (size_t index = 0; index < trace.size(); ++index) {
        auto value = token_slice(trace[index])
            .to(torch::kCPU, torch::kFloat32).contiguous();
        const std::string stage = index == 0
            ? "embedding"
            : "block_" + std::to_string(index - 1);
        const std::filesystem::path file = root / (stage + ".f32");
        std::ofstream output(file, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create block trace tensor: " + file.string());
        }
        output.write(
            reinterpret_cast<const char *>(value.data_ptr<float>()),
            static_cast<std::streamsize>(value.nbytes()));
        if (!output) {
            throw std::runtime_error("failed to write block trace tensor: " + file.string());
        }
        metadata << "{\"stage\":\"" << stage << "\",\"file\":\""
                 << file.filename().string() << "\",\"shape\":[";
        for (int64_t dim = 0; dim < value.dim(); ++dim) {
            if (dim) metadata << ',';
            metadata << value.size(dim);
        }
        metadata << "],\"dtype\":\"float32\"}\n";
        std::cout << "block_trace_dump stage=" << stage
                  << " values=" << value.numel() << "\n";
    }
    auto dump_terminal = [&](const std::string & stage, torch::Tensor value) {
        value = value.to(torch::kCPU, torch::kFloat32).contiguous();
        const std::filesystem::path file = root / (stage + ".f32");
        std::ofstream output(file, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create block trace tensor: " + file.string());
        }
        output.write(
            reinterpret_cast<const char *>(value.data_ptr<float>()),
            static_cast<std::streamsize>(value.nbytes()));
        if (!output) {
            throw std::runtime_error("failed to write block trace tensor: " + file.string());
        }
        metadata << "{\"stage\":\"" << stage << "\",\"file\":\""
                 << file.filename().string() << "\",\"shape\":[";
        for (int64_t dim = 0; dim < value.dim(); ++dim) {
            if (dim) metadata << ',';
            metadata << value.size(dim);
        }
        metadata << "],\"dtype\":\"float32\"}\n";
        metadata.flush();
        std::cout << "block_trace_dump stage=" << stage
                  << " values=" << value.numel() << "\n";
    };
    final_hidden = token_slice(final_hidden);
    dump_terminal("final_norm", final_hidden);
    auto logits = model.lm_head.forward(final_hidden);
    if (model.c.final_logit_softcapping > 0.0) {
        logits = torch::tanh(logits / model.c.final_logit_softcapping) *
            model.c.final_logit_softcapping;
    }
    dump_terminal("logits", logits);
    metadata.flush();
    if (!metadata) throw std::runtime_error("failed to write block trace metadata");
    return 0;
}

static int run_dsv4_hc_model_compare(
    Model & model,
    torch::Tensor ids)
{
    std::vector<torch::Tensor> reference_trace;
    std::vector<torch::Tensor> candidate_trace;
    std::vector<torch::Tensor> repeat_trace;

    g_dsv4_fused_hc = false;
    model.reset(ids.size(0));
    auto reference_hidden = model.hidden_forward(
        ids, c10::nullopt, c10::nullopt, &reference_trace);
    auto reference_logits =
        model.lm_head.forward(reference_hidden).to(torch::kFloat32);

    g_dsv4_fused_hc = true;
    model.reset(ids.size(0));
    auto candidate_hidden = model.hidden_forward(
        ids, c10::nullopt, c10::nullopt, &candidate_trace);
    auto candidate_logits =
        model.lm_head.forward(candidate_hidden).to(torch::kFloat32);

    g_dsv4_fused_hc = false;
    model.reset(ids.size(0));
    auto repeat_hidden = model.hidden_forward(
        ids, c10::nullopt, c10::nullopt, &repeat_trace);
    auto repeat_logits =
        model.lm_head.forward(repeat_hidden).to(torch::kFloat32);
    g_dsv4_fused_hc = true;
    torch::cuda::synchronize();

    if (reference_trace.size() != candidate_trace.size() ||
            reference_trace.size() != repeat_trace.size()) {
        throw std::runtime_error(
            "DeepSeek V4 HC trace stage count mismatch");
    }
    for (size_t index = 0; index < reference_trace.size(); ++index) {
        auto reference = reference_trace[index].reshape({-1});
        auto candidate = candidate_trace[index].reshape({-1});
        auto repeat = repeat_trace[index].reshape({-1});
        auto reference_f64 = reference.to(torch::kFloat64);
        auto candidate_f64 = candidate.to(torch::kFloat64);
        const double denominator = std::max(
            reference_f64.norm().item<double>(), 1.0e-30);
        const std::string stage = index == 0
            ? "embedding"
            : "block_" + std::to_string(index - 1);
        std::cout << std::scientific << std::setprecision(9)
                  << "dsv4_hc_model_trace stage=" << stage
                  << " differing="
                  << candidate.ne(reference).sum().item<int64_t>()
                  << " rel_l2="
                  << (candidate_f64 - reference_f64)
                         .norm().item<double>() / denominator
                  << " mean_abs="
                  << (candidate_f64 - reference_f64)
                         .abs().mean().item<double>()
                  << " max_abs="
                  << (candidate_f64 - reference_f64)
                         .abs().max().item<double>()
                  << " repeat_differing="
                  << repeat.ne(reference).sum().item<int64_t>()
                  << "\n";
    }

    auto reference_logp = torch::log_softmax(reference_logits, -1);
    auto candidate_logp = torch::log_softmax(candidate_logits, -1);
    const double kld_candidate_reference = (
        candidate_logp.exp() *
        (candidate_logp - reference_logp))
        .sum(-1).mean().item<double>();
    const double kld_reference_candidate = (
        reference_logp.exp() *
        (reference_logp - candidate_logp))
        .sum(-1).mean().item<double>();
    auto logit_diff = (
        candidate_logits.to(torch::kFloat64) -
        reference_logits.to(torch::kFloat64));
    std::cout << std::scientific << std::setprecision(9)
              << "dsv4_hc_model_logits"
              << " mean_kld_candidate_reference="
              << kld_candidate_reference
              << " mean_kld_reference_candidate="
              << kld_reference_candidate
              << " relative_l2="
              << logit_diff.norm().item<double>() /
                    std::max(
                        reference_logits.to(torch::kFloat64)
                            .norm().item<double>(),
                        1.0e-30)
              << " mean_abs="
              << logit_diff.abs().mean().item<double>()
              << " max_abs="
              << logit_diff.abs().max().item<double>()
              << " same_top="
              << candidate_logits.argmax(-1)
                     .eq(reference_logits.argmax(-1))
                     .to(torch::kFloat32).mean().item<double>()
              << " repeat_logits_equal="
              << (repeat_logits.equal(reference_logits) ? 1 : 0)
              << "\n";
    return 0;
}

static bool sampling_has_penalties(const MfqSamplingParams & sampling) {
    return sampling.presence_penalty != 0.0 ||
           sampling.frequency_penalty != 0.0 ||
           sampling.repetition_penalty != 1.0;
}

struct ServerDecodeGraphCache {
    decltype(at::cuda::getStreamFromPool(false)) stream;
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    torch::Tensor static_input;
    torch::Tensor static_pos;
    torch::Tensor static_len;
    torch::Tensor static_step;
    torch::Tensor generated;
    torch::Tensor random;
    torch::Tensor counts;
    torch::Tensor static_next;
    int64_t generated_capacity = 0;
    int64_t planned_len = 0;
    bool greedy = false;
    double temperature = 0.0;
    int32_t top_k = 0;
    double top_p = 1.0;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    uint64_t captures = 0;
    uint64_t reuses = 0;
    bool valid = false;

    explicit ServerDecodeGraphCache(int64_t context_capacity)
        : stream(at::cuda::getStreamFromPool(false)),
          generated_capacity(std::max<int64_t>(context_capacity, 2048)) {}

    bool matches(int64_t candidate_len, const MfqSamplingParams & sampling,
                 bool candidate_greedy) const {
        return valid && planned_len == candidate_len && greedy == candidate_greedy &&
               (candidate_greedy ||
                (temperature == sampling.temperature && top_k == sampling.top_k &&
                 top_p == sampling.top_p)) &&
               presence_penalty == sampling.presence_penalty &&
               frequency_penalty == sampling.frequency_penalty &&
               repetition_penalty == sampling.repetition_penalty;
    }

    void ensure_storage(int64_t vocab_size) {
        auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        if (!static_input.defined()) static_input = torch::empty({1, 1}, i64);
        if (!static_pos.defined()) static_pos = torch::empty({1}, i64);
        if (!static_len.defined()) static_len = torch::empty({1}, i64);
        if (!static_step.defined()) static_step = torch::empty({1}, i64);
        if (!generated.defined()) generated = torch::empty({generated_capacity}, i64);
        if (!random.defined()) {
            random = torch::empty(
                {1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        }
        if (!counts.defined()) {
            counts = torch::empty(
                {vocab_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        }
    }

    void invalidate() {
        if (graph) graph->reset();
        graph.reset();
        static_next = torch::Tensor();
        valid = false;
    }

    void set_key(int64_t candidate_len, const MfqSamplingParams & sampling,
                 bool candidate_greedy) {
        planned_len = candidate_len;
        greedy = candidate_greedy;
        temperature = sampling.temperature;
        top_k = sampling.top_k;
        top_p = sampling.top_p;
        presence_penalty = sampling.presence_penalty;
        frequency_penalty = sampling.frequency_penalty;
        repetition_penalty = sampling.repetition_penalty;
        valid = true;
    }
};

static int64_t server_decode_graph_bucket(int64_t planned_len, int64_t context_capacity) {
    const int64_t quantum = planned_len <= 4096 ? 512 :
        (planned_len <= 16384 ? 1024 : 2048);
    const int64_t rounded = ((planned_len + quantum - 1) / quantum) * quantum;
    return std::min<int64_t>(rounded, context_capacity);
}

static bool trace_server_cuda_graph() {
    const char * value = std::getenv("MFQ_SERVER_TRACE_CUDA_GRAPH");
    return value != nullptr && std::atoi(value) != 0;
}

static torch::Tensor sample_server_token(
    Model & model,
    torch::Tensor ids,
    const MfqSamplingParams & sampling,
    torch::Tensor counts,
    torch::Tensor random_host,
    torch::Tensor random_cuda,
    std::mt19937_64 & rng,
    cudaEvent_t prefill_finished = nullptr)
{
    const bool greedy = sampling.temperature <= 0.0 || sampling.top_k == 1;
    const bool has_penalties = sampling_has_penalties(sampling);
    if (greedy && !has_penalties) {
        auto next = model.next_token(ids);
        if (prefill_finished != nullptr) {
            MFQ_CUDA_CHECK(cudaEventRecord(
                prefill_finished, at::cuda::getCurrentCUDAStream()));
        }
        return next;
    }

    auto logits = model.last_logits(ids).contiguous().view({1, -1});
    if (prefill_finished != nullptr) {
        MFQ_CUDA_CHECK(cudaEventRecord(
            prefill_finished, at::cuda::getCurrentCUDAStream()));
    }
    if (has_penalties) {
        sample_apply_penalties_cuda(
            logits, counts, sampling.presence_penalty,
            sampling.frequency_penalty, sampling.repetition_penalty);
    }
    if (greedy) return sample_greedy_cuda(logits);

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    *random_host.data_ptr<float>() = uniform(rng);
    random_cuda.copy_(random_host, true);
    if (sampling.top_k > 0) {
        return sample_top_k_top_p_cuda(
            logits, random_cuda, sampling.temperature, sampling.top_k, sampling.top_p);
    }
    return sample_softmax_cuda(logits, random_cuda, sampling.temperature);
}

class ServerPrefillCudaTimer {
public:
    ServerPrefillCudaTimer()
        : stream_(at::cuda::getCurrentCUDAStream()) {
        MFQ_CUDA_CHECK(cudaEventCreate(&started_));
        try {
            MFQ_CUDA_CHECK(cudaEventCreate(&finished_));
            MFQ_CUDA_CHECK(cudaEventRecord(started_, stream_));
        } catch (...) {
            if (finished_ != nullptr) cudaEventDestroy(finished_);
            cudaEventDestroy(started_);
            finished_ = nullptr;
            started_ = nullptr;
            throw;
        }
    }

    ~ServerPrefillCudaTimer() {
        if (finished_ != nullptr) cudaEventDestroy(finished_);
        if (started_ != nullptr) cudaEventDestroy(started_);
    }

    cudaEvent_t finished_event() const {
        return finished_;
    }

    double elapsed_ms() const {
        MFQ_CUDA_CHECK(cudaEventSynchronize(finished_));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, started_, finished_));
        return static_cast<double>(elapsed);
    }

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t started_ = nullptr;
    cudaEvent_t finished_ = nullptr;
};

static int32_t generate_server_tokens(
    Model & model,
    std::mutex & model_mutex,
    ServerDecodeGraphCache & graph_cache,
    const std::vector<int64_t> & prompt,
    const MfqSamplingParams & sampling,
    const MfqTokenCallback & on_token,
    const MfqPrefillCallback & on_prefill)
{
    std::lock_guard<std::mutex> lock(model_mutex);
    model.reset(1);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto ids = torch::tensor(prompt, options).reshape({1, -1}).contiguous();
    const bool has_penalties = sampling_has_penalties(sampling);
    graph_cache.ensure_storage(model.c.vocab_size);
    auto counts = has_penalties ? graph_cache.counts : torch::Tensor();
    if (has_penalties) {
        counts.zero_();
        sample_token_counts_add_cuda(counts, ids);
    }

    auto random_host = torch::empty(
        {1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    auto random_cuda = torch::empty(
        {1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    std::mt19937_64 rng(sampling.seed);
    auto sample_first_token = [&]() {
        ServerPrefillCudaTimer prefill_timer;
        auto next = sample_server_token(
            model, ids, sampling, counts, random_host, random_cuda, rng,
            prefill_timer.finished_event());
        const int64_t token = next.item<int64_t>();
        const double prefill_ms = prefill_timer.elapsed_ms();
        if (on_prefill) on_prefill(prompt.size(), prefill_ms);
        return std::make_pair(std::move(next), token);
    };
    const char * reprefill_env = std::getenv("MFQ_SERVER_REPREFILL");
    const bool reprefill = reprefill_env != nullptr && reprefill_env[0] == '1';
    std::vector<int64_t> history = prompt;
    const char * trace_incremental_env = std::getenv("MFQ_SERVER_TRACE_INCREMENTAL");
    const bool trace_incremental =
        trace_incremental_env != nullptr && trace_incremental_env[0] == '1';
    if (trace_incremental && sampling.max_tokens > 0) {
        auto [first, first_token] = sample_first_token();
        if (!on_token(first_token)) return 1;

        std::vector<torch::Tensor> incremental_trace;
        std::vector<torch::Tensor> full_trace;
        std::vector<std::pair<std::string, torch::Tensor>> incremental_gemma_trace;
        std::vector<std::pair<std::string, torch::Tensor>> full_gemma_trace;
        const int64_t decode_len = model.cache_pos + 1;
        auto seq_len = torch::tensor({decode_len}, options);
        g_gemma_trace_layer = 0;
        g_gemma_stage_trace = &incremental_gemma_trace;
        auto incremental_hidden = model.hidden_forward(
            first.reshape({1, 1}), c10::nullopt, seq_len, &incremental_trace);
        g_gemma_stage_trace = nullptr;

        history.push_back(first_token);
        model.reset(1);
        auto full_ids = torch::tensor(history, options).reshape({1, -1}).contiguous();
        g_gemma_stage_trace = &full_gemma_trace;
        auto full_hidden = model.hidden_forward(
            full_ids, c10::nullopt, c10::nullopt, &full_trace);
        g_gemma_stage_trace = nullptr;
        g_gemma_trace_layer = -1;
        torch::cuda::synchronize();

        if (incremental_trace.size() != full_trace.size()) {
            throw std::runtime_error("incremental trace stage count mismatch");
        }
        for (size_t i = 0; i < incremental_trace.size(); ++i) {
            auto got = incremental_trace[i].reshape({-1}).to(torch::kFloat64);
            auto ref = full_trace[i].index({Slice(), -1, Slice()}).reshape({-1}).to(torch::kFloat64);
            const double denominator = std::max(ref.norm().item<double>(), 1.0e-30);
            const double relative_l2 = (got - ref).norm().item<double>() / denominator;
            const double cosine = torch::dot(got, ref).item<double>() /
                std::max(got.norm().item<double>() * denominator, 1.0e-30);
            std::cerr << "incremental_trace stage="
                      << (i == 0 ? "embedding" : "block_" + std::to_string(i - 1))
                      << " relative_l2=" << relative_l2
                      << " cosine=" << cosine << std::endl;
        }
        if (incremental_gemma_trace.size() != full_gemma_trace.size()) {
            throw std::runtime_error("incremental Gemma stage count mismatch");
        }
        for (size_t i = 0; i < incremental_gemma_trace.size(); ++i) {
            const auto & got_tensor = incremental_gemma_trace[i].second;
            const auto & full_tensor = full_gemma_trace[i].second;
            if (incremental_gemma_trace[i].first != full_gemma_trace[i].first ||
                full_tensor.numel() < got_tensor.numel()) {
                throw std::runtime_error("incremental Gemma stage layout mismatch");
            }
            auto got = got_tensor.reshape({-1}).to(torch::kFloat64);
            auto full_flat = full_tensor.reshape({-1});
            auto ref = full_flat.narrow(
                0, full_flat.numel() - got_tensor.numel(), got_tensor.numel()).to(torch::kFloat64);
            const double denominator = std::max(ref.norm().item<double>(), 1.0e-30);
            const double relative_l2 = (got - ref).norm().item<double>() / denominator;
            const double cosine = torch::dot(got, ref).item<double>() /
                std::max(got.norm().item<double>() * denominator, 1.0e-30);
            std::cerr << "incremental_gemma_trace stage="
                      << incremental_gemma_trace[i].first
                      << " relative_l2=" << relative_l2
                      << " cosine=" << cosine << std::endl;
        }
        auto incremental_logits = model.lm_head.forward(
            incremental_hidden.index({Slice(), -1, Slice()}).to(torch::kFloat16).contiguous());
        auto full_logits = model.lm_head.forward(
            full_hidden.index({Slice(), -1, Slice()}).to(torch::kFloat16).contiguous());
        const double logits_relative_l2 =
            (incremental_logits.to(torch::kFloat64) - full_logits.to(torch::kFloat64)).norm().item<double>() /
            std::max(full_logits.to(torch::kFloat64).norm().item<double>(), 1.0e-30);
        std::cerr << "incremental_trace logits_relative_l2=" << logits_relative_l2
                  << " incremental_top=" << incremental_logits.argmax(-1).item<int64_t>()
                  << " full_top=" << full_logits.argmax(-1).item<int64_t>() << std::endl;
        return 1;
    }

    const char * graph_env = std::getenv("MFQ_SERVER_CUDA_GRAPH");
    const bool graph_enabled =
        (graph_env == nullptr || graph_env[0] != '0') &&
        g_dsv4_cpu_offload_layers.empty() &&
        !g_moe_expert_cache &&
        !g_tensor_parallel.enabled();
    const char * graph_min_env = std::getenv("MFQ_SERVER_CUDA_GRAPH_MIN_TOKENS");
    const int32_t graph_min_tokens = graph_min_env != nullptr
        ? std::max<int32_t>(2, std::atoi(graph_min_env))
        : 16;
    const bool graph_eligible = graph_enabled && !reprefill &&
        sampling.max_tokens >= graph_min_tokens &&
        sampling.max_tokens <= graph_cache.generated_capacity;
    if (graph_eligible) {
        const bool greedy = sampling.temperature <= 0.0 || sampling.top_k == 1;
        auto [first, first_token] = sample_first_token();
        int32_t generated = 1;
        if (!on_token(first_token) || generated >= sampling.max_tokens) return generated;

        c10::cuda::CUDAStreamGuard graph_guard(graph_cache.stream);
        cudaStream_t graph_raw_stream = graph_cache.stream.stream();
        if (has_penalties) {
            sample_token_counts_add_cuda(graph_cache.counts, first.contiguous());
        }

        int64_t pos_h = model.cache_pos;
        int64_t len_h = pos_h + 1;
        int64_t step_h = 1;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_input.data_ptr<int64_t>(), first.data_ptr<int64_t>(),
            sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.generated.data_ptr<int64_t>(), first.data_ptr<int64_t>(),
            sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_pos.data_ptr<int64_t>(), &pos_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_len.data_ptr<int64_t>(), &len_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_step.data_ptr<int64_t>(), &step_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        *random_host.data_ptr<float>() = 0.5f;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.random.data_ptr<float>(), random_host.data_ptr<float>(),
            sizeof(float), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));

        auto sample_static = [&]() {
            if (greedy && !has_penalties) {
                return model.next_token_static(
                    graph_cache.static_input, graph_cache.static_pos, graph_cache.static_len);
            }
            auto logits = model.last_logits_static(
                    graph_cache.static_input, graph_cache.static_pos, graph_cache.static_len)
                .contiguous().view({1, -1});
            if (has_penalties) {
                sample_apply_penalties_cuda(
                    logits, graph_cache.counts, sampling.presence_penalty,
                    sampling.frequency_penalty, sampling.repetition_penalty);
            }
            if (greedy) return sample_greedy_cuda(logits);
            if (sampling.top_k > 0) {
                return sample_top_k_top_p_cuda(
                    logits, graph_cache.random, sampling.temperature,
                    sampling.top_k, sampling.top_p);
            }
            return sample_softmax_cuda(logits, graph_cache.random, sampling.temperature);
        };

        const int64_t requested_len = model.cache_pos + sampling.max_tokens;
        const int64_t planned_len = server_decode_graph_bucket(
            requested_len, model.c.max_position_embeddings);
        const bool cache_hit = graph_cache.matches(planned_len, sampling, greedy);
        if (!cache_hit) {
            graph_cache.invalidate();
            c10::cuda::CUDACachingAllocator::emptyCache();
            g_decode_graph_attention_kv_len = planned_len;
            g_decode_graph_attention_parts = planned_len >= 192 ? (planned_len + 127) / 128 : 1;
            g_decode_graph_attention_parts = std::min<int64_t>(
                g_decode_graph_attention_parts, FullBlock::kDecodeAttentionMaxParts);
            try {
                if (model.c.is_gemma4() || model.c.is_glm_dsa()) {
                    (void)sample_static();
                    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));
                }

                graph_cache.graph = std::make_unique<at::cuda::CUDAGraph>();
                graph_cache.graph->capture_begin();
                graph_cache.static_next = sample_static();
                if (has_penalties) {
                    sample_token_counts_add_cuda(
                        graph_cache.counts, graph_cache.static_next.contiguous());
                }
                decode_graph_commit_cuda(
                    graph_cache.static_next, graph_cache.generated,
                    graph_cache.static_step, graph_cache.static_input,
                    graph_cache.static_pos, graph_cache.static_len);
                graph_cache.graph->capture_end();
                graph_cache.set_key(planned_len, sampling, greedy);
                ++graph_cache.captures;
            } catch (...) {
                graph_cache.invalidate();
                g_decode_graph_attention_kv_len = 0;
                g_decode_graph_attention_parts = 0;
                throw;
            }
            g_decode_graph_attention_kv_len = 0;
            g_decode_graph_attention_parts = 0;
            report_cuda_memory("server_graph_capture");
        } else {
            ++graph_cache.reuses;
        }
        if (trace_server_cuda_graph()) {
            std::cerr << "server_cuda_graph action=" << (cache_hit ? "reuse" : "capture")
                      << " requested_len=" << requested_len
                      << " planned_len=" << planned_len
                      << " captures=" << graph_cache.captures
                      << " reuses=" << graph_cache.reuses << std::endl;
        }

        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        while (generated < sampling.max_tokens) {
            if (!greedy) {
                *random_host.data_ptr<float>() = uniform(rng);
                MFQ_CUDA_CHECK(cudaMemcpyAsync(
                    graph_cache.random.data_ptr<float>(), random_host.data_ptr<float>(),
                    sizeof(float), cudaMemcpyHostToDevice, graph_raw_stream));
            }
            graph_cache.graph->replay();
            const int64_t token = graph_cache.static_next.item<int64_t>();
            ++generated;
            if (!on_token(token)) break;
        }
        model.cache_pos += generated - 1;
        return generated;
    }

    int32_t generated = 0;
    while (generated < sampling.max_tokens) {
        if (reprefill && generated > 0) {
            model.reset(1);
            ids = torch::tensor(history, options).reshape({1, -1}).contiguous();
        }
        torch::Tensor next;
        int64_t token = 0;
        if (generated == 0) {
            auto first = sample_first_token();
            next = std::move(first.first);
            token = first.second;
        } else {
            next = sample_server_token(
                model, ids, sampling, counts, random_host, random_cuda, rng);
            token = next.item<int64_t>();
        }
        ++generated;
        if (!on_token(token)) break;
        history.push_back(token);
        if (has_penalties) sample_token_counts_add_cuda(counts, next.contiguous());
        ids = next.reshape({1, 1});
    }
    return generated;
}

struct KlEvalChunk {
    std::vector<int32_t> tokens;
    std::vector<float> target_log_probs;
    int target_start = 0;
    int score_count = 0;
    std::streamoff row_offset = 0;
};

struct KlReferenceContract {
    int64_t n_batch = 0;
    int64_t n_ubatch = 0;
};

enum class KlEvaluator {
    Legacy,
    Optimized,
};

static KlEvaluator parse_kl_evaluator(const std::string & value) {
    if (value == "legacy") return KlEvaluator::Legacy;
    if (value == "optimized") return KlEvaluator::Optimized;
    throw std::runtime_error(
        "--kl-evaluator must be legacy or optimized");
}

static KlMmqMode parse_kl_mmq_mode(const std::string & value) {
    if (value == "default") return KlMmqMode::Default;
    if (value == "nint8_1") return KlMmqMode::Nint8One;
    if (value == "fp16") return KlMmqMode::Fp16;
    throw std::runtime_error(
        "--kl-mmq must be default, nint8_1, or fp16");
}

static Nint6MmqMode parse_nint6_mmq_mode(
        const std::string & value) {
    if (value == "fp16") return Nint6MmqMode::Fp16;
    if (value == "int8") return Nint6MmqMode::Int8;
    throw std::runtime_error(
        "--nint6-mmq must be fp16 or int8");
}

static std::vector<KlMmqMode> parse_kl_mmq_sequence(
        const std::string & value) {
    std::vector<KlMmqMode> modes;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(
            std::remove_if(
                item.begin(), item.end(),
                [](unsigned char ch) { return std::isspace(ch) != 0; }),
            item.end());
        if (item.empty()) {
            throw std::runtime_error(
                "--kl-mmq-sequence contains an empty item");
        }
        const KlMmqMode mode = parse_kl_mmq_mode(item);
        if (mode == KlMmqMode::Default) {
            throw std::runtime_error(
                "--kl-mmq-sequence accepts only nint8_1 and fp16");
        }
        if (std::find(modes.begin(), modes.end(), mode) != modes.end()) {
            throw std::runtime_error(
                "--kl-mmq-sequence contains a duplicate mode");
        }
        modes.push_back(mode);
    }
    if (modes.empty()) {
        throw std::runtime_error("--kl-mmq-sequence cannot be empty");
    }
    return modes;
}

static const char * kl_evaluator_name(KlEvaluator evaluator) {
    return evaluator == KlEvaluator::Legacy ? "legacy" : "optimized";
}

static int run_kl_eval(
        Model & model,
        const std::string & path,
        int max_chunks,
        KlEvaluator evaluator,
        int score_override,
        const KlReferenceContract & reference_contract) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open KL logits file: " + path);
    char magic[8];
    int32_t n_vocab = 0;
    f.read(magic, sizeof(magic));
    if (!f) {
        throw std::runtime_error("invalid KL logits header");
    }
    const std::string reference_format(magic, sizeof(magic));
    std::vector<KlEvalChunk> eval_chunks;
    if (reference_format == "_logits_") {
        uint32_t n_ctx = 0;
        int32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&n_ctx), sizeof(n_ctx));
        f.read(reinterpret_cast<char*>(&n_vocab), sizeof(n_vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || n_ctx < 2 || n_vocab <= 0 || n_chunks <= 0) {
            throw std::runtime_error("invalid legacy KL logits header");
        }
        std::vector<int32_t> tokens((size_t)n_ctx * n_chunks);
        f.read(
            reinterpret_cast<char*>(tokens.data()),
            (std::streamsize)(tokens.size() * sizeof(int32_t)));
        if (!f) throw std::runtime_error("truncated legacy KL token header");
        const int first = (int)n_ctx / 2;
        const int score_count = (int)n_ctx - 1 - first;
        eval_chunks.resize((size_t)n_chunks);
        const int32_t legacy_bos = tokens[0];
        int legacy_bos_replacements = 0;
        for (int ci = 0; ci < n_chunks; ++ci) {
            auto begin = tokens.begin() + (size_t)ci * n_ctx;
            eval_chunks[(size_t)ci].tokens.assign(begin, begin + n_ctx);
            // llama-perplexity temporarily replaces every Gemma context
            // chunk's first corpus token with BOS before evaluating it, then
            // serializes the original corpus tokens in this legacy header.
            // Qwen legacy references use the serialized token directly.
            if (model.c.is_gemma4() &&
                    eval_chunks[(size_t)ci].tokens[0] != legacy_bos) {
                eval_chunks[(size_t)ci].tokens[0] = legacy_bos;
                ++legacy_bos_replacements;
            }
            eval_chunks[(size_t)ci].target_start = first + 1;
            eval_chunks[(size_t)ci].score_count = score_count;
        }
        std::cout << "cpp_kl_legacy_chunk_bos="
                  << (model.c.is_gemma4() ? legacy_bos : -1)
                  << " replacements=" << legacy_bos_replacements
                  << " model_type=" << model.c.model_type << "\n";
    } else if (reference_format == "_logit2_" || reference_format == "_logit3_") {
        const bool has_exact_target_log_probs = reference_format == "_logit3_";
        uint32_t vocab = 0;
        uint32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&vocab), sizeof(vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || vocab == 0 || vocab > (uint32_t)std::numeric_limits<int32_t>::max() ||
            n_chunks == 0 || n_chunks > (1u << 20)) {
            throw std::runtime_error("invalid trace KL logits header");
        }
        n_vocab = (int32_t)vocab;
        eval_chunks.resize((size_t)n_chunks);
        std::vector<uint32_t> token_counts(n_chunks);
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            uint32_t target_start = 0;
            uint32_t score_count = 0;
            f.read(reinterpret_cast<char*>(&token_counts[ci]), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&target_start), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&score_count), sizeof(uint32_t));
            if (!f || token_counts[ci] < 2 || target_start < 1 ||
                target_start >= token_counts[ci] || score_count < 1 ||
                score_count > token_counts[ci] - target_start) {
                throw std::runtime_error("invalid trace KL chunk descriptor");
            }
            eval_chunks[ci].target_start = (int)target_start;
            eval_chunks[ci].score_count = (int)score_count;
        }
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            auto & chunk = eval_chunks[ci];
            chunk.tokens.resize(token_counts[ci]);
            f.read(
                reinterpret_cast<char*>(chunk.tokens.data()),
                (std::streamsize)(chunk.tokens.size() * sizeof(int32_t)));
            if (!f) throw std::runtime_error("truncated trace KL token header");
        }
        if (has_exact_target_log_probs) {
            for (auto & chunk : eval_chunks) {
                chunk.target_log_probs.resize((size_t)chunk.score_count);
                f.read(
                    reinterpret_cast<char*>(chunk.target_log_probs.data()),
                    (std::streamsize)(chunk.target_log_probs.size() * sizeof(float)));
                if (!f) throw std::runtime_error("truncated trace KL target log probabilities");
                for (float value : chunk.target_log_probs) {
                    if (!std::isfinite(value) || value > 0.0f) {
                        throw std::runtime_error("invalid trace KL target log probability");
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("invalid KL logits magic");
    }
    const std::streamoff data_offset = f.tellg();
    const int nv = 2 * ((n_vocab + 1) / 2) + 4;
    std::streamoff row_offset = data_offset;
    for (auto & chunk : eval_chunks) {
        chunk.row_offset = row_offset;
        row_offset += (std::streamoff)chunk.score_count * nv * (std::streamoff)sizeof(uint16_t);
    }
    f.seekg(0, std::ios::end);
    if (!f || f.tellg() < row_offset) {
        throw std::runtime_error("truncated KL logits rows");
    }
    const int available_chunks = (int)eval_chunks.size();
    const int chunks = max_chunks < 0
        ? available_chunks
        : std::min(max_chunks, available_chunks);
    if (chunks <= 0) throw std::runtime_error("KL evaluation requires at least one chunk");
    constexpr int KL_BATCH = 8;
    double kld_sum = 0.0;
    double reverse_kld_sum = 0.0;
    double bf16_ce_sum = 0.0;
    double mfq_ce_sum = 0.0;
    int64_t same_top = 0;
    int64_t count = 0;
    const bool optimized = evaluator == KlEvaluator::Optimized;
    auto started = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_execution evaluator="
              << kl_evaluator_name(evaluator)
              << " graph=single_sequence"
              << " available_chunks=" << available_chunks
              << " selected_chunks=" << chunks
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << "\n";

    for (int ci = 0; ci < chunks; ++ci) {
        const auto & chunk = eval_chunks[(size_t)ci];
        const int stored_score_count = chunk.score_count;
        const int score_count = score_override < 0 ? stored_score_count : score_override;
        if (score_count > stored_score_count) {
            throw std::runtime_error(
                "--kl-score-count exceeds the stored chunk score count");
        }
        const int first = chunk.target_start - 1;
        model.reset(1);
        std::vector<int64_t> chunk_tokens(chunk.tokens.size());
        for (size_t j = 0; j < chunk.tokens.size(); ++j) {
            chunk_tokens[j] = chunk.tokens[j];
        }
        auto ids = torch::from_blob(chunk_tokens.data(), {1, (int64_t)chunk_tokens.size()},
                                    torch::TensorOptions().dtype(torch::kInt64)).clone().to(torch::kCUDA);
        torch::Tensor pred;
        if (optimized) {
            auto hidden = model.hidden_forward(ids);
            auto selected_hidden = hidden.index({
                0, Slice(first, first + score_count), Slice()
            }).contiguous();
            pred = model.logits_from_hidden(selected_hidden);
        } else if (score_count < stored_score_count) {
            (void)model.hidden_forward(ids.index({Slice(), Slice(0, first)}));
            auto logits = model.forward(ids.index({Slice(), Slice(first, first + score_count)}));
            pred = logits.index({0, Slice(), Slice()});
        } else {
            auto logits = model.forward(ids);
            pred = logits.index({0, Slice(first, first + score_count), Slice()});
        }
        if (pred.size(1) != n_vocab) throw std::runtime_error("KL vocab size mismatch");

        torch::Tensor optimized_kld_sum;
        torch::Tensor optimized_reverse_kld_sum;
        torch::Tensor optimized_bf16_ce_sum;
        torch::Tensor optimized_mfq_ce_sum;
        torch::Tensor optimized_same_top;
        if (optimized) {
            const auto cuda = torch::TensorOptions().device(torch::kCUDA);
            optimized_kld_sum = torch::zeros(
                {}, cuda.dtype(torch::kFloat64));
            optimized_reverse_kld_sum = torch::zeros(
                {}, cuda.dtype(torch::kFloat64));
            optimized_bf16_ce_sum = torch::zeros(
                {}, cuda.dtype(torch::kFloat64));
            optimized_mfq_ce_sum = torch::zeros(
                {}, cuda.dtype(torch::kFloat64));
            optimized_same_top = torch::zeros(
                {}, cuda.dtype(torch::kInt64));
        }
        f.seekg(chunk.row_offset);
        for (int s = 0; s < score_count; s += KL_BATCH) {
            const int b = std::min(KL_BATCH, score_count - s);
            std::vector<uint16_t> rows((size_t)b * nv);
            f.read(reinterpret_cast<char*>(rows.data()), (std::streamsize)(rows.size() * sizeof(uint16_t)));
            if (!f) throw std::runtime_error("truncated KL logits data");

            std::vector<float> scales(b), mins(b);
            std::vector<int32_t> codes((size_t)b * n_vocab);
            for (int r = 0; r < b; ++r) {
                const uint16_t* row = rows.data() + (size_t)r * nv;
                std::memcpy(&scales[r], row + 0, sizeof(float));
                std::memcpy(&mins[r], row + 2, sizeof(float));
                for (int v = 0; v < n_vocab; ++v) codes[(size_t)r * n_vocab + v] = row[4 + v];
            }
            auto scale = torch::from_blob(scales.data(), {b}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(torch::kCUDA);
            auto min_lp = torch::from_blob(mins.data(), {b}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(torch::kCUDA);
            auto base_codes = torch::from_blob(
                codes.data(), {b, n_vocab}, torch::TensorOptions().dtype(torch::kInt32))
                .clone().to(torch::kCUDA);
            auto base_logp = base_codes.to(torch::kFloat32) * scale.unsqueeze(1) +
                min_lp.unsqueeze(1);
            auto q = pred.index({Slice(s, s + b), Slice()}).to(torch::kFloat32);
            auto lse = torch::logsumexp(q, -1);
            auto quant_logp = q - lse.unsqueeze(1);
            auto normalized_base_logp =
                base_logp - torch::logsumexp(base_logp, -1, true);
            auto p_base = torch::exp(base_logp).masked_fill(base_codes.eq(0), 0.0f);
            auto kld = (p_base * (base_logp - q + lse.unsqueeze(1))).sum(-1);
            auto reverse_kld =
                (torch::exp(quant_logp) * (quant_logp - normalized_base_logp)).sum(-1);
            if (optimized) {
                optimized_kld_sum.add_(kld.to(torch::kFloat64).sum());
                optimized_reverse_kld_sum.add_(
                    reverse_kld.to(torch::kFloat64).sum());
            } else {
                kld_sum += kld.to(torch::kFloat64).sum().item<double>();
                reverse_kld_sum +=
                    reverse_kld.to(torch::kFloat64).sum().item<double>();
            }
            std::vector<int64_t> target_ids(b);
            for (int r = 0; r < b; ++r) {
                target_ids[r] = chunk_tokens[(size_t)chunk.target_start + s + r];
            }
            auto target = torch::from_blob(target_ids.data(), {b, 1},
                                            torch::TensorOptions().dtype(torch::kInt64))
                              .clone().to(torch::kCUDA);
            if (chunk.target_log_probs.empty()) {
                auto batch_bf16_ce =
                    -base_logp.gather(1, target).to(torch::kFloat64).sum();
                if (optimized) {
                    optimized_bf16_ce_sum.add_(batch_bf16_ce);
                } else {
                    bf16_ce_sum += batch_bf16_ce.item<double>();
                }
            } else {
                for (int r = 0; r < b; ++r) {
                    bf16_ce_sum -= chunk.target_log_probs[(size_t)s + r];
                }
            }
            auto batch_mfq_ce = (lse.unsqueeze(1) - q.gather(1, target))
                                     .to(torch::kFloat64).sum();
            auto batch_same_top =
                q.argmax(-1).eq(base_logp.argmax(-1)).sum();
            if (optimized) {
                optimized_mfq_ce_sum.add_(batch_mfq_ce);
                optimized_same_top.add_(batch_same_top);
            } else {
                mfq_ce_sum += batch_mfq_ce.item<double>();
                same_top += batch_same_top.item<int64_t>();
            }
            count += b;
        }
        if (optimized) {
            kld_sum += optimized_kld_sum.item<double>();
            reverse_kld_sum += optimized_reverse_kld_sum.item<double>();
            if (chunk.target_log_probs.empty()) {
                bf16_ce_sum += optimized_bf16_ce_sum.item<double>();
            }
            mfq_ce_sum += optimized_mfq_ce_sum.item<double>();
            same_top += optimized_same_top.item<int64_t>();
        }
        torch::cuda::synchronize();
        std::cout << "cpp_kl_chunk=" << (ci + 1)
                  << " mean=" << (kld_sum / (double)count)
                  << " mean_kld_q_ref=" << (reverse_kld_sum / (double)count)
                  << " same_top=" << ((double)same_top / (double)count) << "\n";
    }
    auto ended = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_result chunks=" << chunks
              << " scored_tokens=" << count
              << " sec=" << std::chrono::duration<double>(ended - started).count()
              << " kld=" << (kld_sum / (double)count)
              << " mean_kld_q_ref=" << (reverse_kld_sum / (double)count)
              << " bf16_ce=" << (bf16_ce_sum / (double)count)
              << " mfq_ce=" << (mfq_ce_sum / (double)count)
              << " bf16_ppl=" << std::exp(bf16_ce_sum / (double)count)
              << " mfq_ppl=" << std::exp(mfq_ce_sum / (double)count)
              << " kld_pct_bf16=" << (100.0 * kld_sum / bf16_ce_sum)
              << " same_top=" << ((double)same_top / (double)count)
              << " same_top_count=" << same_top
              << " reference_format="
              << (reference_format == "_logit3_" ? "trace_v3" :
                  (reference_format == "_logit2_" ? "trace_v2" : "legacy"))
              << " execution=" << kl_evaluator_name(evaluator)
              << " graph=single_sequence"
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << "\n";
    return 0;
}

struct StreamedKlInput {
    std::string reference_format;
    int32_t n_vocab = 0;
    int nv = 0;
    std::vector<KlEvalChunk> chunks;
};

static StreamedKlInput load_streamed_kl_input(
    const std::string & path,
    int max_chunks) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open KL logits file: " + path);
    char magic[8];
    f.read(magic, sizeof(magic));
    if (!f) throw std::runtime_error("invalid KL logits header");

    StreamedKlInput input;
    input.reference_format.assign(magic, sizeof(magic));
    if (input.reference_format == "_logits_") {
        uint32_t n_ctx = 0;
        int32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&n_ctx), sizeof(n_ctx));
        f.read(reinterpret_cast<char*>(&input.n_vocab), sizeof(input.n_vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || n_ctx < 2 || input.n_vocab <= 0 || n_chunks <= 0) {
            throw std::runtime_error("invalid legacy KL logits header");
        }
        std::vector<int32_t> tokens((size_t)n_ctx * n_chunks);
        f.read(
            reinterpret_cast<char*>(tokens.data()),
            (std::streamsize)(tokens.size() * sizeof(int32_t)));
        if (!f) throw std::runtime_error("truncated legacy KL token header");
        const int first = (int)n_ctx / 2;
        const int score_count = (int)n_ctx - 1 - first;
        input.chunks.resize((size_t)n_chunks);
        for (int ci = 0; ci < n_chunks; ++ci) {
            auto begin = tokens.begin() + (size_t)ci * n_ctx;
            input.chunks[(size_t)ci].tokens.assign(begin, begin + n_ctx);
            input.chunks[(size_t)ci].target_start = first + 1;
            input.chunks[(size_t)ci].score_count = score_count;
        }
    } else if (
        input.reference_format == "_logit2_" ||
        input.reference_format == "_logit3_") {
        const bool has_exact_target_log_probs =
            input.reference_format == "_logit3_";
        uint32_t vocab = 0;
        uint32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&vocab), sizeof(vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || vocab == 0 ||
            vocab > (uint32_t)std::numeric_limits<int32_t>::max() ||
            n_chunks == 0 || n_chunks > (1u << 20)) {
            throw std::runtime_error("invalid trace KL logits header");
        }
        input.n_vocab = (int32_t)vocab;
        input.chunks.resize((size_t)n_chunks);
        std::vector<uint32_t> token_counts(n_chunks);
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            uint32_t target_start = 0;
            uint32_t score_count = 0;
            f.read(reinterpret_cast<char*>(&token_counts[ci]), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&target_start), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&score_count), sizeof(uint32_t));
            if (!f || token_counts[ci] < 2 || target_start < 1 ||
                target_start >= token_counts[ci] || score_count < 1 ||
                score_count > token_counts[ci] - target_start) {
                throw std::runtime_error("invalid trace KL chunk descriptor");
            }
            input.chunks[ci].target_start = (int)target_start;
            input.chunks[ci].score_count = (int)score_count;
        }
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            auto & chunk = input.chunks[ci];
            chunk.tokens.resize(token_counts[ci]);
            f.read(
                reinterpret_cast<char*>(chunk.tokens.data()),
                (std::streamsize)(chunk.tokens.size() * sizeof(int32_t)));
            if (!f) throw std::runtime_error("truncated trace KL token header");
        }
        if (has_exact_target_log_probs) {
            for (auto & chunk : input.chunks) {
                chunk.target_log_probs.resize((size_t)chunk.score_count);
                f.read(
                    reinterpret_cast<char*>(chunk.target_log_probs.data()),
                    (std::streamsize)(
                        chunk.target_log_probs.size() * sizeof(float)));
                if (!f) {
                    throw std::runtime_error(
                        "truncated trace KL target log probabilities");
                }
                for (float value : chunk.target_log_probs) {
                    if (!std::isfinite(value) || value > 0.0f) {
                        throw std::runtime_error(
                            "invalid trace KL target log probability");
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("invalid KL logits magic");
    }

    const std::streamoff data_offset = f.tellg();
    input.nv = 2 * ((input.n_vocab + 1) / 2) + 4;
    std::streamoff row_offset = data_offset;
    for (auto & chunk : input.chunks) {
        chunk.row_offset = row_offset;
        row_offset += (std::streamoff)chunk.score_count * input.nv *
            (std::streamoff)sizeof(uint16_t);
    }
    f.seekg(0, std::ios::end);
    if (!f || f.tellg() < row_offset) {
        throw std::runtime_error("truncated KL logits rows");
    }
    const int available_chunks = (int)input.chunks.size();
    const int chunks = max_chunks < 0
        ? available_chunks
        : std::min(max_chunks, available_chunks);
    if (chunks <= 0) {
        throw std::runtime_error("KL evaluation requires at least one chunk");
    }
    input.chunks.resize((size_t)chunks);
    return input;
}

static torch::Tensor streamed_kl_ids(
    const StreamedKlInput & input,
    int begin,
    int count,
    int64_t n_ctx) {
    std::vector<int64_t> token_ids((size_t)count * n_ctx);
    for (int bi = 0; bi < count; ++bi) {
        const auto & source = input.chunks[(size_t)(begin + bi)].tokens;
        for (int64_t ti = 0; ti < n_ctx; ++ti) {
            token_ids[(size_t)bi * n_ctx + (size_t)ti] =
                source[(size_t)ti];
        }
    }
    return torch::from_blob(
        token_ids.data(), {count, n_ctx},
        torch::TensorOptions().dtype(torch::kInt64))
        .clone().to(torch::kCUDA);
}

struct KlEvalMetrics {
    double kld_sum = 0.0;
    double reverse_kld_sum = 0.0;
    double reference_ce_sum = 0.0;
    double quant_ce_sum = 0.0;
    int64_t same_top = 0;
    int64_t count = 0;
};

static void accumulate_streamed_kl_chunk(
    std::ifstream & f,
    const StreamedKlInput & input,
    int chunk_index,
    torch::Tensor pred,
    int score_count,
    KlEvalMetrics & metrics) {
    constexpr int KL_BATCH = 8;
    const auto & chunk = input.chunks[(size_t)chunk_index];
    f.seekg(chunk.row_offset);
    for (int s = 0; s < score_count; s += KL_BATCH) {
        const int b = std::min(KL_BATCH, score_count - s);
        std::vector<uint16_t> rows((size_t)b * input.nv);
        f.read(
            reinterpret_cast<char*>(rows.data()),
            (std::streamsize)(rows.size() * sizeof(uint16_t)));
        if (!f) throw std::runtime_error("truncated KL logits data");

        std::vector<float> scales(b), mins(b);
        std::vector<int32_t> codes((size_t)b * input.n_vocab);
        for (int r = 0; r < b; ++r) {
            const uint16_t * row = rows.data() + (size_t)r * input.nv;
            std::memcpy(&scales[r], row + 0, sizeof(float));
            std::memcpy(&mins[r], row + 2, sizeof(float));
            for (int v = 0; v < input.n_vocab; ++v) {
                codes[(size_t)r * input.n_vocab + v] = row[4 + v];
            }
        }
        auto scale = torch::from_blob(
            scales.data(), {b},
            torch::TensorOptions().dtype(torch::kFloat32))
            .clone().to(torch::kCUDA);
        auto min_lp = torch::from_blob(
            mins.data(), {b},
            torch::TensorOptions().dtype(torch::kFloat32))
            .clone().to(torch::kCUDA);
        auto base_codes = torch::from_blob(
            codes.data(), {b, input.n_vocab},
            torch::TensorOptions().dtype(torch::kInt32))
            .clone().to(torch::kCUDA);
        auto base_logp = base_codes.to(torch::kFloat32) *
            scale.unsqueeze(1) + min_lp.unsqueeze(1);
        auto q = pred.index({Slice(s, s + b), Slice()}).to(torch::kFloat32);
        auto lse = torch::logsumexp(q, -1);
        auto quant_logp = q - lse.unsqueeze(1);
        auto normalized_base_logp =
            base_logp - torch::logsumexp(base_logp, -1, true);
        auto p_base =
            torch::exp(base_logp).masked_fill(base_codes.eq(0), 0.0f);
        auto kld =
            (p_base * (base_logp - q + lse.unsqueeze(1))).sum(-1);
        auto reverse_kld =
            (torch::exp(quant_logp) *
             (quant_logp - normalized_base_logp)).sum(-1);
        metrics.kld_sum +=
            kld.to(torch::kFloat64).sum().item<double>();
        metrics.reverse_kld_sum +=
            reverse_kld.to(torch::kFloat64).sum().item<double>();

        std::vector<int64_t> target_ids(b);
        for (int r = 0; r < b; ++r) {
            target_ids[r] =
                chunk.tokens[(size_t)chunk.target_start + s + r];
        }
        auto target = torch::from_blob(
            target_ids.data(), {b, 1},
            torch::TensorOptions().dtype(torch::kInt64))
            .clone().to(torch::kCUDA);
        if (chunk.target_log_probs.empty()) {
            metrics.reference_ce_sum +=
                -base_logp.gather(1, target)
                    .to(torch::kFloat64).sum().item<double>();
        } else {
            for (int r = 0; r < b; ++r) {
                metrics.reference_ce_sum -=
                    chunk.target_log_probs[(size_t)s + r];
            }
        }
        metrics.quant_ce_sum +=
            (lse.unsqueeze(1) - q.gather(1, target))
                .to(torch::kFloat64).sum().item<double>();
        metrics.same_top +=
            q.argmax(-1).eq(base_logp.argmax(-1)).sum().item<int64_t>();
        metrics.count += b;
    }
}

static int run_kl_eval_batched(
    Model & model,
    const std::string & reference_path,
    int max_chunks,
    int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract & reference_contract) {
    auto input = load_streamed_kl_input(reference_path, max_chunks);
    const int chunks = (int)input.chunks.size();
    const int64_t n_ctx = (int64_t)input.chunks[0].tokens.size();
    const int64_t n_batch = requested_n_batch == 0
        ? n_ctx : requested_n_batch;
    if (n_ctx <= 0 || n_batch < n_ctx || n_batch % n_ctx != 0) {
        throw std::runtime_error(
            "optimized KL requires --kl-n-batch to be at least n_ctx "
            "and exactly divisible by n_ctx");
    }
    const int llama_kl_n_seq = std::max<int64_t>(
        1, n_batch / n_ctx);
    KlKvCacheCapacityScope kv_cache_capacity_scope(n_ctx);
    const int target_start = input.chunks[0].target_start;
    int score_count = input.chunks[0].score_count;
    for (const auto & chunk : input.chunks) {
        if ((int64_t)chunk.tokens.size() != n_ctx ||
            chunk.target_start != target_start ||
            chunk.score_count != score_count) {
            throw std::runtime_error(
                "optimized KL requires uniform chunk geometry");
        }
    }
    if (score_override >= 0) {
        score_count = score_override;
        if (score_count < 1 ||
            score_count > input.chunks[0].score_count) {
            throw std::runtime_error(
                "--kl-score-count exceeds the stored chunk score count");
        }
    }
    if (input.n_vocab != model.c.vocab_size) {
        throw std::runtime_error("KL vocab size mismatch");
    }
    int bos_replacements = 0;
    int32_t legacy_bos = -1;
    if (input.reference_format == "_logits_" &&
        model.c.is_gemma4()) {
        legacy_bos = input.chunks[0].tokens[0];
        for (auto & chunk : input.chunks) {
            if (chunk.tokens[0] != legacy_bos) {
                chunk.tokens[0] = legacy_bos;
                ++bos_replacements;
            }
        }
    }

    std::ifstream reference(reference_path, std::ios::binary);
    if (!reference) {
        throw std::runtime_error(
            "cannot reopen KL logits file: " + reference_path);
    }
    const int first = target_start - 1;
    KlEvalMetrics metrics;
    auto started = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_execution evaluator=optimized"
              << " graph=batched_contexts"
              << " n_batch=" << n_batch
              << " n_ctx=" << n_ctx
              << " n_seq=" << llama_kl_n_seq
              << " score_count=" << score_count
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << " logits_start=" << first
              << " bos=" << legacy_bos
              << " bos_replacements=" << bos_replacements
              << "\n";

    for (int begin = 0; begin < chunks; begin += llama_kl_n_seq) {
        const int n_seq_batch =
            std::min(llama_kl_n_seq, chunks - begin);
        auto ids = streamed_kl_ids(
            input, begin, n_seq_batch, n_ctx);
        model.reset(n_seq_batch);
        auto hidden = model.hidden_forward(ids);
        for (int bi = 0; bi < n_seq_batch; ++bi) {
            auto selected_hidden = hidden.index({
                bi, Slice(first, first + score_count), Slice()
            }).contiguous();
            auto pred =
                model.logits_from_hidden(selected_hidden);
            accumulate_streamed_kl_chunk(
                reference, input, begin + bi, pred,
                score_count, metrics);
            std::cout << "cpp_kl_chunk=" << (begin + bi + 1)
                      << " mean="
                      << (metrics.kld_sum / (double)metrics.count)
                      << " mean_kld_q_ref="
                      << (metrics.reverse_kld_sum /
                          (double)metrics.count)
                      << " same_top="
                      << ((double)metrics.same_top /
                          (double)metrics.count)
                      << "\n";
        }
        torch::cuda::synchronize();
    }
    auto ended = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_result chunks=" << chunks
              << " scored_tokens=" << metrics.count
              << " sec="
              << std::chrono::duration<double>(
                     ended - started).count()
              << " kld="
              << (metrics.kld_sum / (double)metrics.count)
              << " mean_kld_q_ref="
              << (metrics.reverse_kld_sum / (double)metrics.count)
              << " bf16_ce="
              << (metrics.reference_ce_sum / (double)metrics.count)
              << " mfq_ce="
              << (metrics.quant_ce_sum / (double)metrics.count)
              << " bf16_ppl="
              << std::exp(
                     metrics.reference_ce_sum /
                     (double)metrics.count)
              << " mfq_ppl="
              << std::exp(
                     metrics.quant_ce_sum /
                     (double)metrics.count)
              << " kld_pct_bf16="
              << (100.0 * metrics.kld_sum /
                  metrics.reference_ce_sum)
              << " same_top="
              << ((double)metrics.same_top /
                  (double)metrics.count)
              << " same_top_count=" << metrics.same_top
              << " reference_format="
              << (input.reference_format == "_logit3_"
                      ? "trace_v3"
                      : (input.reference_format == "_logit2_"
                             ? "trace_v2" : "legacy"))
               << " execution=optimized"
               << " graph=batched_contexts"
               << " n_ctx=" << n_ctx
               << " n_batch=" << n_batch
               << " n_seq=" << llama_kl_n_seq
               << " score_count=" << score_count
               << " score_count_override=" << score_override
               << " reference_n_batch=" << reference_contract.n_batch
               << " reference_n_ubatch=" << reference_contract.n_ubatch
               << "\n";
    std::cout << "cpp_kl_mmq"
              << " mmq=" << kl_mmq_mode_name(g_kl_mmq_mode)
              << " activation_quantize_calls="
              << g_kl_mmq_activation_quantize_calls
              << " dense_calls=" << g_kl_mmq_dense_calls
              << " moe_calls=" << g_kl_mmq_moe_calls
              << " fallback_calls=" << g_kl_mmq_fallback_calls
              << "\n";
    return 0;
}

static int run_selected_kl_eval(
    Model & model,
    const std::string & reference_path,
    int max_chunks,
    KlEvaluator evaluator,
    int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract & reference_contract) {
    return evaluator == KlEvaluator::Optimized
        ? run_kl_eval_batched(
              model, reference_path, max_chunks,
              requested_n_batch, score_override,
              reference_contract)
        : run_kl_eval(
              model, reference_path, max_chunks, evaluator,
              score_override, reference_contract);
}

static int run_kl_eval_streamed(
    const std::string & mfq_path,
    const std::string & config_path,
    const std::string & reference_path,
    const std::string & logits_output_path,
    int max_chunks,
    int layer_group,
    int chunk_batch,
    int score_override,
    const KlReferenceContract & reference_contract) {
    if (layer_group < 1 || chunk_batch < 1) {
        throw std::runtime_error(
            "streamed KL layer group and chunk batch must be positive");
    }
    g_moe_route_stats.clear();
    MfqDropFileCacheGuard drop_cache_guard(true);
    auto input = load_streamed_kl_input(reference_path, max_chunks);
    const int chunks = (int)input.chunks.size();
    const int64_t n_ctx = (int64_t)input.chunks[0].tokens.size();
    const int target_start = input.chunks[0].target_start;
    int score_count = input.chunks[0].score_count;
    for (const auto & chunk : input.chunks) {
        if ((int64_t)chunk.tokens.size() != n_ctx ||
            chunk.target_start != target_start ||
            chunk.score_count != score_count) {
            throw std::runtime_error(
                "streamed KL currently requires uniform chunk geometry");
        }
    }
    if (score_override >= 0) {
        score_count = score_override;
        if (score_count < 1 ||
            score_count > input.chunks[0].score_count) {
            throw std::runtime_error(
                "--kl-score-count exceeds the stored chunk score count");
        }
    }
    const int first = target_start - 1;

    std::ofstream logits_output;
    std::filesystem::path logits_final;
    std::filesystem::path logits_partial;
    if (!logits_output_path.empty()) {
        logits_final = std::filesystem::path(logits_output_path);
        logits_partial = logits_final;
        logits_partial += ".partial";
        if (std::filesystem::exists(logits_final) ||
                std::filesystem::exists(logits_partial)) {
            throw std::runtime_error(
                "refusing to overwrite saved KL logits: " +
                logits_output_path);
        }
        if (!logits_final.parent_path().empty()) {
            std::filesystem::create_directories(
                logits_final.parent_path());
        }
        logits_output.open(logits_partial, std::ios::binary);
        if (!logits_output) {
            throw std::runtime_error(
                "cannot create saved KL logits: " +
                logits_partial.string());
        }
        const char magic[8] = {'_', 'm', 'f', 'q', 'f', '1', '6', '_'};
        const int32_t header[5] = {
            chunks,
            static_cast<int32_t>(n_ctx),
            target_start,
            score_count,
            input.n_vocab,
        };
        logits_output.write(magic, sizeof(magic));
        logits_output.write(
            reinterpret_cast<const char *>(header), sizeof(header));
    }

    auto started = std::chrono::steady_clock::now();
    Model model = load_model(mfq_path, config_path, n_ctx, false);
    if (!model.c.is_dsv4()) {
        throw std::runtime_error(
            "streamed KL is currently implemented for DeepSeek V4");
    }
    if (chunk_batch > 16) {
        throw std::runtime_error(
            "DeepSeek V4 streamed KL chunk batch must not exceed 16");
    }
    if (model.c.vocab_size != input.n_vocab) {
        throw std::runtime_error("KL vocab size mismatch");
    }
    MfqFile mfq(mfq_path);
    const bool gguf_names = mfq.records.count("token_embd.weight") != 0;

    const int64_t hidden_bytes =
        (int64_t)chunks * n_ctx * model.c.hc_mult *
        model.c.hidden_size * (int64_t)sizeof(c10::Half);
    auto hidden_cpu = torch::empty(
        {chunks, n_ctx, model.c.hc_mult, model.c.hidden_size},
        torch::TensorOptions()
            .device(torch::kCPU)
            .dtype(torch::kFloat16)
            .pinned_memory(true));
    std::cout << "cpp_kl_stream_begin chunks=" << chunks
              << " ctx=" << n_ctx
              << " score_count=" << score_count
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << " layer_group=" << layer_group
              << " chunk_batch=" << chunk_batch
              << " hidden_bytes=" << hidden_bytes << "\n";

    for (int begin = 0; begin < chunks; begin += chunk_batch) {
        const int count = std::min(chunk_batch, chunks - begin);
        auto ids = streamed_kl_ids(input, begin, count, n_ctx);
        auto x = model.embed_forward(ids);
        x = x.unsqueeze(2)
            .expand({
                count, n_ctx, model.c.hc_mult, model.c.hidden_size})
            .contiguous();
        hidden_cpu.narrow(0, begin, count).copy_(x, true);
        torch::cuda::synchronize();
    }
    std::cout << "cpp_kl_stream_phase=embedding completed_chunks="
              << chunks << "\n";

    for (int layer_begin = 0;
        layer_begin < model.c.num_hidden_layers;
         layer_begin += layer_group) {
        const int layer_end = std::min(
            layer_begin + layer_group,
            (int)model.c.num_hidden_layers);
        auto group_started = std::chrono::steady_clock::now();
        auto dsv4_state = std::make_shared<Dsv4SharedState>();
        std::vector<std::unique_ptr<Block>> blocks;
        blocks.reserve((size_t)(layer_end - layer_begin));
        for (int layer = layer_begin; layer < layer_end; ++layer) {
            std::cerr << "stream loading layer " << layer << " "
                      << model.c.layer_types[(size_t)layer] << std::endl;
            blocks.push_back(load_block(
                mfq, model.c, layer,
                model.c.layer_types[(size_t)layer], gguf_names,
                nullptr, dsv4_state));
        }
        for (int begin = 0; begin < chunks; begin += chunk_batch) {
            const int count = std::min(chunk_batch, chunks - begin);
            auto ids = streamed_kl_ids(input, begin, count, n_ctx);
            auto pos = torch::arange(
                0, n_ctx,
                torch::TensorOptions()
                    .device(torch::kCUDA).dtype(torch::kInt64));
            auto x = hidden_cpu.narrow(0, begin, count).to(torch::kCUDA);
            for (auto & block : blocks) {
                block->reset(count);
                block->set_token_ids(ids);
                x = block->forward(
                    x, pos, 0, c10::nullopt, model.c, model.rope);
            }
            hidden_cpu.narrow(0, begin, count).copy_(x, true);
            torch::cuda::synchronize();
        }
        blocks.clear();
        dsv4_state.reset();
        c10::cuda::CUDACachingAllocator::emptyCache();
        auto group_ended = std::chrono::steady_clock::now();
        std::cout << "cpp_kl_stream_layers begin=" << layer_begin
                  << " end=" << layer_end
                  << " sec="
                  << std::chrono::duration<double>(
                         group_ended - group_started).count()
                  << "\n";
    }

    std::ifstream reference(reference_path, std::ios::binary);
    if (!reference) {
        throw std::runtime_error(
            "cannot reopen KL logits file: " + reference_path);
    }
    KlEvalMetrics metrics;
    for (int begin = 0; begin < chunks; begin += chunk_batch) {
        const int count = std::min(chunk_batch, chunks - begin);
        auto x = hidden_cpu.narrow(0, begin, count).to(torch::kCUDA);
        auto y = model.finalize_hidden(x, count, n_ctx);
        auto selected =
            y.index({Slice(), Slice(first, first + score_count), Slice()});
        auto logits = model.lm_head.forward(selected);
        if (model.c.final_logit_softcapping > 0.0) {
            logits = torch::tanh(
                logits / model.c.final_logit_softcapping) *
                model.c.final_logit_softcapping;
        }
        if (logits_output.is_open()) {
            auto saved = logits.to(torch::kCPU, torch::kFloat16)
                             .contiguous();
            logits_output.write(
                reinterpret_cast<const char *>(
                    saved.data_ptr<c10::Half>()),
                static_cast<std::streamsize>(
                    saved.numel() * sizeof(c10::Half)));
            if (!logits_output) {
                throw std::runtime_error(
                    "failed while writing saved KL logits: " +
                    logits_partial.string());
            }
        }
        for (int bi = 0; bi < count; ++bi) {
            accumulate_streamed_kl_chunk(
                reference, input, begin + bi,
                logits.index({bi, Slice(), Slice()}),
                score_count, metrics);
            std::cout << "cpp_kl_chunk=" << (begin + bi + 1)
                      << " mean="
                      << (metrics.kld_sum / (double)metrics.count)
                      << " mean_kld_q_ref="
                      << (metrics.reverse_kld_sum / (double)metrics.count)
                      << " same_top="
                      << ((double)metrics.same_top /
                          (double)metrics.count)
                      << "\n";
        }
        torch::cuda::synchronize();
    }
    auto ended = std::chrono::steady_clock::now();
    if (logits_output.is_open()) {
        logits_output.close();
        if (!logits_output) {
            throw std::runtime_error(
                "failed to finalize saved KL logits: " +
                logits_partial.string());
        }
        std::filesystem::rename(logits_partial, logits_final);
    }
    std::cout << "cpp_kl_result chunks=" << chunks
              << " scored_tokens=" << metrics.count
              << " sec="
              << std::chrono::duration<double>(ended - started).count()
              << " kld="
              << (metrics.kld_sum / (double)metrics.count)
              << " mean_kld_q_ref="
              << (metrics.reverse_kld_sum / (double)metrics.count)
              << " bf16_ce="
              << (metrics.reference_ce_sum / (double)metrics.count)
              << " mfq_ce="
              << (metrics.quant_ce_sum / (double)metrics.count)
              << " bf16_ppl="
              << std::exp(
                  metrics.reference_ce_sum / (double)metrics.count)
              << " mfq_ppl="
              << std::exp(metrics.quant_ce_sum / (double)metrics.count)
              << " kld_pct_bf16="
              << (100.0 * metrics.kld_sum / metrics.reference_ce_sum)
              << " same_top="
              << ((double)metrics.same_top / (double)metrics.count)
              << " reference_format="
              << (input.reference_format == "_logit3_"
                  ? "trace_v3"
                  : (input.reference_format == "_logit2_"
                     ? "trace_v2" : "legacy"))
              << " execution=streamed_layer_groups"
              << " graph=streamed_layer_groups"
              << " n_ctx=" << n_ctx
              << " score_count=" << score_count
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << "\n";
    write_moe_route_stats();
    return 0;
}

static int run_linear_check(
    const std::string & mfq_path,
    const std::string & name,
    int M,
    int gate_mode,
    int reps) {
    MfqFile mfq(mfq_path);
    auto linear = load_quant_linear(mfq, name);
    if (nint5_q5_exec_enabled() && linear.is_nint() &&
        linear.nint.w.bits == 5 && linear.nint.w.gs == 28) {
        enable_nint5_q5_exec(linear.nint.w);
    }
    TORCH_CHECK(M >= 1 && M <= 4096, "--check-linear-m must be in [1, 4096]");
    TORCH_CHECK(reps >= 1, "--check-linear-reps must be positive");
    int64_t neuron_len = linear.neuron_len();
    auto x = torch::arange((int64_t)M * neuron_len,
                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32))
                 .reshape({M, neuron_len});
    x = (x.remainder(97) - 48) / 512.0;
    auto xh = x.to(torch::kFloat16).contiguous();
    torch::Tensor gateh;
    if (gate_mode != 0) {
        TORCH_CHECK(gate_mode == 1 || gate_mode == 2, "linear check gate mode must be 0, 1, or 2");
        gateh = ((torch::arange((int64_t)M * neuron_len, x.options()).reshape({M, neuron_len})
                    .remainder(53) - 26) / 16.0)
                    .to(torch::kFloat16).contiguous();
    }
    auto run = [&]() {
        return gate_mode == 0 ? linear.forward(xh) : linear.forward_input_mul(xh, gateh, gate_mode);
    };
    torch::Tensor y_test;
    const int warmups = std::min(30, std::max(1, reps));
    for (int i = 0; i < warmups; ++i) y_test = run();
    torch::cuda::synchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaEventRecord(start, stream);
    for (int i = 0; i < reps; ++i) y_test = run();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    y_test = y_test.to(torch::kFloat32);

    const NintWeight * nint = linear.is_nint() ? &linear.nint.w : nullptr;
    const NvqWeight * nvq = linear.is_nvq() ? &linear.nvq.w : nullptr;
    const Mxfp8Weight * mxfp8 = linear.is_mxfp8()
        ? &linear.mxfp8.weight : nullptr;
    const char * dq_env_name = nint == nullptr ? nullptr :
        (nint->bits == 4 ? "MFQ_NINT4_GS24_DQ_VEC2" :
         (nint->bits == 6 ? "MFQ_NINT6_GS24_DQ_VEC4" : nullptr));
    std::string dq_env_saved;
    bool had_dq_env = false;
    if (dq_env_name != nullptr) {
        const char * value = std::getenv(dq_env_name);
        if (value != nullptr) {
            had_dq_env = true;
            dq_env_saved = value;
        }
        mfq_set_env(dq_env_name, "0");
    }
    torch::Tensor ww;
    if (nint != nullptr) {
        ww = nint->q8_zero
            ? nint8_zero_dequant_cuda(
                  nint->q_packed, nint->q8_zero_scale, nint->neuron_len)
            : nint->q5_exec
            ? nint5_gs28_q5_dequant_cuda(
                  nint->q_packed, nint->neuron_scale, nint->neuron_min, nint->neuron_len)
            : nint->bits == 4
            ? nint_dequant_full_packed_compact_cuda(
                  nint->q_packed, nint->sub_scale, nint->sub_min,
                  nint->neuron_scale, nint->neuron_min, nint->neuron_len, nint->gs)
            : nint_dequant_full_packed_compact_bits_cuda(
                  nint->q_packed, nint->sub_scale, nint->sub_min,
                  nint->neuron_scale, nint->neuron_min, nint->neuron_len, nint->gs, nint->bits);
    } else if (nvq != nullptr) {
        ww = nvq_dequant(*nvq);
    } else {
        ww = mxfp8_cpu_reference(*mxfp8);
    }
    torch::Tensor ref_input = xh;
    if (gate_mode == 1) ref_input = xh * torch::sigmoid(gateh);
    else if (gate_mode == 2) ref_input = xh * torch::silu(gateh);
    auto y_ref = torch::matmul(ref_input, ww.transpose(0, 1)).to(torch::kFloat32);
    torch::cuda::synchronize();
    if (dq_env_name != nullptr) {
        mfq_set_env(dq_env_name, had_dq_env ? dq_env_saved.c_str() : "");
    }
    auto diff = (y_test - y_ref).abs();
    double per_ms = (double)elapsed_ms / (double)reps;
    double weight_bytes = nint != nullptr
        ? (double)nint->q_packed.numel() +
          (nint->q8_zero
              ? (double)nint->q8_zero_scale.numel() * sizeof(c10::Half)
              : (nint->q5_exec
                    ? 0.0
                    : (double)(nint->sub_scale.numel() +
                               nint->sub_min.numel())) +
                (double)(nint->neuron_scale.numel() +
                         nint->neuron_min.numel()) * sizeof(float))
        : nvq != nullptr
        ? (double)nvq->indices_packed.numel() + (double)nvq->aux_packed.numel() +
          (double)nvq->sub_scale_packed.numel() +
          (double)nvq->neuron_scale.numel() * sizeof(float) + (double)nvq->codebook.numel()
        : (double)mxfp8->values.numel() + (double)mxfp8->scales.numel();
    std::cout << "shape=" << y_ref.sizes() << "\n";
    if (nint != nullptr) {
        std::cout << "dtype="
                  << (nint->q8_zero ? "NINT8-0" : "NINT" + std::to_string(nint->bits))
                  << " gs=" << nint->gs << " m=" << M << "\n";
    } else if (nvq != nullptr) {
        std::cout << "dtype=NVQ" << nvq->format << " gs=" << nvq->gs
                  << " sub_bits=" << nvq->sub_bits << " m=" << M << "\n";
    } else {
        std::cout << "dtype=MXFP8 block=128x128 m=" << M << "\n";
    }
    if (gate_mode != 0) std::cout << "gate=" << (gate_mode == 1 ? "sigmoid" : "silu") << "\n";
    std::cout << "production_ms=" << per_ms << "\n";
    std::cout << "production_weight_gbps=" << weight_bytes / (per_ms * 1.0e6) << "\n";
    std::cout << "production_rel=" << ((y_test - y_ref).norm() / y_ref.norm()).item<float>() << "\n";
    std::cout << "production_mean_abs=" << diff.mean().item<float>() << "\n";
    std::cout << "production_max_abs=" << diff.max().item<float>() << "\n";
    return 0;
}

static int run_tensor_parallel_linear_check(
        const std::string & mfq_path,
        const std::string & name,
        TensorParallelAxis axis,
        int M) {
    TORCH_CHECK(
        g_tensor_parallel.enabled(),
        "--check-tp-linear requires --tensor-parallel");
    TORCH_CHECK(
        axis == TensorParallelAxis::Output ||
        axis == TensorParallelAxis::Input,
        "--check-tp-axis must be output or input");
    TORCH_CHECK(
        M >= 1 && M <= 4096,
        "--check-tp-m must be in [1, 4096]");
    MfqFile mfq(mfq_path);
    const TensorParallelConfig saved =
        g_tensor_parallel;
    g_tensor_parallel = {};
    g_tensor_parallel.devices = {
        saved.primary_device()};
    auto full = load_quant_linear(
        mfq, name, TensorParallelAxis::Mirrored);
    g_tensor_parallel = saved;
    auto sharded = load_quant_linear(
        mfq, name, axis);
    TORCH_CHECK(
        sharded.tensor_parallel(),
        "tensor-parallel diagnostic did not create shards");

    const int64_t width = full.neuron_len();
    auto x = torch::arange(
        static_cast<int64_t>(M) * width,
        torch::TensorOptions()
            .device(torch::Device(
                torch::kCUDA,
                saved.primary_device()))
            .dtype(torch::kFloat32))
        .reshape({M, width});
    x = ((x.remainder(127) - 63) / 384.0)
        .to(torch::kFloat16).contiguous();
    auto reference = full.forward(x).to(torch::kFloat32);
    auto test = sharded.forward(x).to(torch::kFloat32);
    torch::cuda::synchronize();
    const auto difference = (test - reference).abs();
    const double denominator =
        std::max(
            reference.norm().item<double>(),
            1.0e-30);
    const double relative =
        (test - reference).norm().item<double>() /
        denominator;
    const double mean_abs =
        difference.mean().item<double>();
    const double max_abs =
        difference.max().item<double>();
    std::cout
        << "tensor_parallel_check=1"
        << " tensor=" << name
        << " axis="
        << (axis == TensorParallelAxis::Output
            ? "output" : "input")
        << " logical_shape=[" << full.out()
        << ',' << full.neuron_len() << ']'
        << " shards="
        << sharded.tensor_parallel_shards.size()
        << " m=" << M
        << " relative=" << relative
        << " mean_abs=" << mean_abs
        << " max_abs=" << max_abs
        << '\n';
    const double tolerance = full.is_mxfp8()
        ? 5.0e-4
        : axis == TensorParallelAxis::Output
        ? 1.0e-6 : 5.0e-3;
    if (!torch::isfinite(test).all().item<bool>() ||
        relative > tolerance) {
        throw std::runtime_error(
            "tensor-parallel linear numerical check failed");
    }
    return 0;
}

static std::vector<std::string> parse_tensor_names(const std::string & value) {
    std::vector<std::string> names;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find(',', begin);
        const size_t count =
            end == std::string::npos ? value.size() - begin : end - begin;
        const std::string name = value.substr(begin, count);
        if (name.empty()) {
            throw std::runtime_error("empty tensor name in comma-separated list");
        }
        names.push_back(name);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return names;
}

static int run_linear_group_check(
        const std::string & mfq_path,
        const std::string & names_arg,
        int M) {
    TORCH_CHECK(M >= 1 && M <= 4096, "--check-linear-m must be in [1, 4096]");
    const auto names = parse_tensor_names(names_arg);
    TORCH_CHECK(names.size() >= 2, "--check-linear-group requires at least two tensors");
    MfqFile mfq(mfq_path);
    auto group = load_quant_group(mfq, names, names.size());
    const int64_t width = group.nint_grouped
        ? (group.nint.split_w.empty()
            ? group.nint.w.neuron_len
            : group.nint.split_w.front().neuron_len)
        : group.layers.front().neuron_len();
    auto sequence = torch::arange(
        (int64_t)M * width,
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    auto x = (((sequence.remainder(257) - 128.0) / 127.0) +
              0.03125 * torch::sin(sequence * 0.015625))
                 .to(torch::kFloat16)
                 .reshape({M, width})
                 .contiguous();
    auto actual = group.forward(x);
    TORCH_CHECK(actual.size() == names.size(), "linear group output count mismatch");
    std::vector<torch::Tensor> graph_actual;
    if (M == 1 && decode_branch_parallel_enabled(M)) {
        torch::cuda::synchronize();
        const auto graph_stream =
            at::cuda::getStreamFromPool(false);
        c10::cuda::CUDAStreamGuard graph_guard(
            graph_stream);
        at::cuda::CUDAGraph graph;
        graph.capture_begin();
        graph_actual = group.forward(x);
        graph.capture_end();
        graph.replay();
        MFQ_CUDA_CHECK(cudaStreamSynchronize(
            graph_stream.stream()));
        TORCH_CHECK(
            graph_actual.size() == actual.size(),
            "linear group CUDA Graph output count mismatch");
        for (size_t index = 0;
             index < graph_actual.size(); ++index) {
            const auto difference =
                (graph_actual[index].to(torch::kFloat32) -
                 actual[index].to(torch::kFloat32)).abs();
            const double maximum =
                difference.max().item<double>();
            std::cout
                << "linear_group_graph_check"
                << " tensor=" << names[index]
                << " max_abs=" << maximum << "\n";
            TORCH_CHECK(
                maximum == 0.0,
                "linear group CUDA Graph replay differs from eager output");
        }
    }
    at::globalContext().setAllowTF32CuBLAS(false);
    std::vector<torch::Tensor> dense_weights;
    std::vector<torch::Tensor> separate_dense_references;
    std::vector<torch::Tensor> fp32_references;
    std::vector<torch::Tensor> separate_production;
    dense_weights.reserve(names.size());
    separate_dense_references.reserve(names.size());
    fp32_references.reserve(names.size());
    separate_production.reserve(names.size());
    for (size_t index = 0; index < names.size(); ++index) {
        auto linear = load_quant_linear(mfq, names[index]);
        TORCH_CHECK(
            linear.is_nint(),
            "--check-linear-group currently requires NINT tensors");
        const auto & weight = linear.nint.w;
        auto dense = weight.q8_zero
            ? nint8_zero_dequant_cuda(
                  weight.q_packed, weight.q8_zero_scale, weight.neuron_len)
            : weight.q5_exec
            ? nint5_gs28_q5_dequant_cuda(
                  weight.q_packed, weight.neuron_scale, weight.neuron_min,
                  weight.neuron_len)
            : weight.bits == 4
            ? nint_dequant_full_packed_compact_cuda(
                  weight.q_packed, weight.sub_scale, weight.sub_min,
                  weight.neuron_scale, weight.neuron_min,
                  weight.neuron_len, weight.gs)
            : nint_dequant_full_packed_compact_bits_cuda(
                  weight.q_packed, weight.sub_scale, weight.sub_min,
                  weight.neuron_scale, weight.neuron_min,
                  weight.neuron_len, weight.gs, weight.bits);
        separate_production.push_back(linear.forward(x).to(torch::kFloat32));
        separate_dense_references.push_back(
            torch::matmul(x, dense.transpose(0, 1)).to(torch::kFloat32));
        fp32_references.push_back(torch::matmul(
            x.to(torch::kFloat32),
            dense.to(torch::kFloat32).transpose(0, 1)));
        dense_weights.push_back(std::move(dense));
    }
    auto combined_dense = torch::cat(dense_weights, 0).contiguous();
    auto combined_output =
        torch::matmul(x, combined_dense.transpose(0, 1)).to(torch::kFloat32);
    auto combined_references =
        combined_output.split_with_sizes(group.outs, -1);
    for (size_t index = 0; index < names.size(); ++index) {
        auto combined_reference = combined_references[index];
        auto separate_dense_reference = separate_dense_references[index];
        auto fp32_reference = fp32_references[index];
        auto separate_candidate = separate_production[index];
        auto candidate = actual[index].to(torch::kFloat32);
        auto combined_dense_difference = candidate - combined_reference;
        auto grouped_vs_separate = candidate - separate_candidate;
        auto separate_dense_difference =
            separate_candidate - separate_dense_reference;
        auto grouped_fp32_difference = candidate - fp32_reference;
        auto separate_fp32_difference = separate_candidate - fp32_reference;
        const double grouped_vs_separate_rel =
            (grouped_vs_separate.norm() /
             separate_candidate.norm().clamp_min(1.0e-30)).item<double>();
        const double grouped_vs_separate_snr =
            grouped_vs_separate_rel == 0.0
                ? std::numeric_limits<double>::infinity()
                : -20.0 * std::log10(grouped_vs_separate_rel);
        const double grouped_fp32_rel =
            (grouped_fp32_difference.norm() /
             fp32_reference.norm().clamp_min(1.0e-30)).item<double>();
        const double separate_fp32_rel =
            (separate_fp32_difference.norm() /
             fp32_reference.norm().clamp_min(1.0e-30)).item<double>();
        std::cout << std::fixed << std::setprecision(9)
                  << "linear_group_check"
                  << " tensor=" << names[index]
                  << " m=" << M
                  << " n=" << combined_reference.size(1)
                  << " k=" << width
                  << " grouped_vs_combined_dense_rel="
                  << (combined_dense_difference.norm() /
                      combined_reference.norm().clamp_min(1.0e-30)).item<double>()
                  << " grouped_vs_separate_rel=" << grouped_vs_separate_rel
                  << " grouped_vs_separate_snr_db=" << grouped_vs_separate_snr
                  << " grouped_vs_separate_mean_abs="
                  << grouped_vs_separate.abs().mean().item<double>()
                  << " grouped_vs_separate_max_abs="
                  << grouped_vs_separate.abs().max().item<double>()
                  << " separate_vs_separate_dense_rel="
                  << (separate_dense_difference.norm() /
                      separate_dense_reference.norm().clamp_min(1.0e-30)).item<double>()
                  << " grouped_vs_fp32_rel=" << grouped_fp32_rel
                  << " grouped_vs_fp32_snr_db="
                  << (grouped_fp32_rel == 0.0
                          ? std::numeric_limits<double>::infinity()
                          : -20.0 * std::log10(grouped_fp32_rel))
                  << " separate_vs_fp32_rel=" << separate_fp32_rel
                  << " separate_vs_fp32_snr_db="
                  << (separate_fp32_rel == 0.0
                          ? std::numeric_limits<double>::infinity()
                          : -20.0 * std::log10(separate_fp32_rel))
                  << "\n";
    }
    return 0;
}

static torch::Tensor read_f32_tensor(
        const std::filesystem::path & path,
        const std::vector<int64_t> & shape) {
    const int64_t count = std::accumulate(
        shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    std::vector<float> values(static_cast<size_t>(count));
    std::ifstream input(path, std::ios::binary);
    TORCH_CHECK(input, "failed to open ", path.string());
    input.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    TORCH_CHECK(
        input.gcount() ==
            static_cast<std::streamsize>(values.size() * sizeof(float)),
        "short f32 tensor read from ", path.string());
    return torch::from_blob(
               values.data(), shape,
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(torch::kCUDA)
        .contiguous();
}

static void write_f32_tensor(
        const std::filesystem::path & path,
        const torch::Tensor & tensor) {
    auto host = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
    std::ofstream output(path, std::ios::binary);
    TORCH_CHECK(output, "failed to create ", path.string());
    output.write(
        reinterpret_cast<const char *>(host.data_ptr<float>()),
        static_cast<std::streamsize>(host.numel() * sizeof(float)));
    TORCH_CHECK(output, "failed to write ", path.string());
}

static int run_gdn_operator_check(
        const std::string & input_dir,
        const std::string & output_path,
        const std::string & state_path,
        int64_t tokens,
        int64_t q_heads,
        int64_t v_heads,
        int64_t head_dim) {
    TORCH_CHECK(
        tokens >= 1 && q_heads >= 1 && v_heads >= q_heads && head_dim >= 1,
        "invalid GDN diagnostic shape");
    TORCH_CHECK(
        v_heads % q_heads == 0,
        "GDN diagnostic value heads must be divisible by query heads");
    const std::filesystem::path root(input_dir);
    auto q = read_f32_tensor(
        root / "q.bin", {1, tokens, q_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
    auto k = read_f32_tensor(
        root / "k.bin", {1, tokens, q_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
    auto v = read_f32_tensor(
        root / "v.bin", {1, tokens, v_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
    auto g = read_f32_tensor(
        root / "g.bin", {1, tokens, v_heads})
                 .permute({0, 2, 1})
                 .contiguous();
    auto beta = read_f32_tensor(
        root / "beta.bin", {1, tokens, v_heads})
                    .permute({0, 2, 1})
                    .contiguous();
    auto state = read_f32_tensor(
        root / "state.bin", {1, v_heads, head_dim, head_dim});
    auto result = gdn_inplace_transposed_tiled_cuda(
        q, k, v, g, beta, state);
    write_f32_tensor(output_path, result[0]);
    write_f32_tensor(state_path, result[1]);
    std::cout << "mfq_gdn_operator"
              << " t=" << tokens
              << " hq=" << q_heads
              << " hv=" << v_heads
              << " d=" << head_dim
              << " output=" << output_path
              << " state=" << state_path << "\n";
    return 0;
}

static int run_linear_conv_operator_check(
        const std::string & input_dir,
        const std::string & output_dir,
        int64_t tokens,
        int64_t q_heads,
        int64_t v_heads,
        int64_t key_dim,
        int64_t value_dim,
        int64_t kernel_size,
        double eps) {
    TORCH_CHECK(
        tokens >= 2 && q_heads >= 1 && v_heads >= 1 &&
            key_dim >= 1 && value_dim >= 1 &&
            kernel_size >= 2 && kernel_size <= 8,
        "invalid linear-conv diagnostic shape");
    const int64_t qk_width = 2 * q_heads * key_dim;
    const int64_t v_width = v_heads * value_dim;
    const int64_t channels = qk_width + v_width;
    const std::filesystem::path input_root(input_dir);
    const std::filesystem::path output_root(output_dir);
    std::filesystem::create_directories(output_root);
    auto state = read_f32_tensor(
        input_root / "state.bin",
        {1, kernel_size - 1, channels});
    auto qk = read_f32_tensor(
                  input_root / "qk.bin",
                  {1, tokens, qk_width})
                  .to(torch::kHalf)
                  .contiguous();
    auto v = read_f32_tensor(
                 input_root / "v.bin",
                 {1, tokens, v_width})
                 .to(torch::kHalf)
                 .contiguous();
    auto weight = read_f32_tensor(
        input_root / "weight.bin",
        {channels, 1, kernel_size});
    torch::Tensor bias;
    if (std::filesystem::exists(input_root / "bias.bin")) {
        bias = read_f32_tensor(
            input_root / "bias.bin", {channels});
    } else {
        bias = torch::empty(
            {0},
            torch::TensorOptions()
                .device(torch::kCUDA)
                .dtype(torch::kFloat32));
    }
    auto result = linear_conv_qkv_prefill_cuda(
        state, qk, v, weight, bias,
        q_heads, v_heads, key_dim, value_dim, eps);
    write_f32_tensor(output_root / "q.bin", result[0]);
    write_f32_tensor(output_root / "k.bin", result[1]);
    write_f32_tensor(output_root / "v.bin", result[2]);
    write_f32_tensor(output_root / "state.bin", result[3]);
    std::cout << "mfq_linear_conv_operator"
              << " t=" << tokens
              << " hq=" << q_heads
              << " hv=" << v_heads
              << " dk=" << key_dim
              << " dv=" << value_dim
              << " kernel=" << kernel_size
              << " output=" << output_dir << "\n";
    return 0;
}

static int run_q8_embedding_check(
        const std::string & mfq_path,
        const std::string & name) {
    MfqFile mfq(mfq_path);
    auto linear = load_quant_linear(mfq, name);
    TORCH_CHECK(
        linear.is_nint() && linear.nint.w.q8_zero,
        "--check-q8-embedding requires an NINT8-0 tensor");
    const int64_t vocab = linear.nint.w.out;
    std::vector<int64_t> host_ids = {
        0,
        std::min<int64_t>(1, vocab - 1),
        std::min<int64_t>(106, vocab - 1),
        std::min<int64_t>(12345, vocab - 1),
        std::min<int64_t>(255999, vocab - 1),
        vocab - 1,
    };
    auto ids = torch::from_blob(
        host_ids.data(), {(int64_t)host_ids.size()},
        torch::TensorOptions().dtype(torch::kInt64))
                   .clone()
                   .to(torch::kCUDA)
                   .contiguous();
    auto candidate = nint8_zero_embedding_lookup_cuda(
        linear.nint.w.q_packed, linear.nint.w.q8_zero_scale,
        ids, linear.nint.w.neuron_len);
    auto dense = nint8_zero_dequant_cuda(
        linear.nint.w.q_packed, linear.nint.w.q8_zero_scale,
        linear.nint.w.neuron_len);
    auto reference = dense.index_select(0, ids);
    auto difference =
        candidate.to(torch::kFloat32) - reference.to(torch::kFloat32);
    std::cout << std::fixed << std::setprecision(9)
              << "q8_embedding_check"
              << " tensor=" << name
              << " ids=" << host_ids.size()
              << " vocab=" << vocab
              << " width=" << linear.nint.w.neuron_len
              << " equal=" << (candidate.equal(reference) ? 1 : 0)
              << " rel="
              << (difference.norm() /
                  reference.to(torch::kFloat32).norm()).item<double>()
              << " mean_abs=" << difference.abs().mean().item<double>()
              << " max_abs=" << difference.abs().max().item<double>()
              << "\n";
    return 0;
}

static int run_dsv4_output_a_check(
        const std::string & mfq_path,
        const std::string & name,
        int batch,
        int reps) {
    constexpr int64_t kGroups = 8;
    TORCH_CHECK(batch > 0 && reps > 0, "DSV4 output_a check requires positive batch and reps");
    MfqFile mfq(mfq_path);
    auto linear = load_quant_linear(
        mfq, name, TensorParallelAxis::Input);
    const bool supported_nint =
        linear.is_nint() &&
        linear.nint.w.bits == 8 &&
        linear.nint.w.gs == 48;
    TORCH_CHECK(
        supported_nint || linear.is_mxfp8(),
        "DSV4 output_a check requires NINT8 gs48 or MXFP8");
    TORCH_CHECK(
        linear.out() % kGroups == 0,
        "DSV4 output_a rows must divide eight groups");

    const int64_t width = linear.neuron_len();
    const int64_t rows_per_group = linear.out() / kGroups;
    auto sequence = torch::arange(
        (int64_t)batch * kGroups * width,
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    auto grouped = (
        (sequence.remainder(257) - 128.0) / 127.0 +
        0.03125 * torch::sin(sequence * 0.015625))
        .to(torch::kFloat16)
        .reshape({batch, kGroups, width})
        .contiguous();

    auto legacy = [&]() {
        auto expanded = linear.forward(
            grouped.reshape({batch * kGroups, width}))
            .reshape({batch, kGroups, kGroups, rows_per_group});
        std::vector<torch::Tensor> diagonal;
        diagonal.reserve(kGroups);
        for (int64_t group = 0; group < kGroups; ++group) {
            diagonal.push_back(
                expanded.index({Slice(), group, group, Slice()}));
        }
        return torch::stack(diagonal, 1).reshape({batch, linear.out()});
    };
    auto groupwise = [&]() {
        return linear.is_mxfp8()
            ? linear.forward_mxfp8_groupwise(grouped, kGroups)
            : nint_matmul_groupwise_u8(
                linear.nint.w, grouped, kGroups);
    };
    auto time_ms = [&](auto && fn) {
        torch::Tensor output;
        for (int warmup = 0; warmup < 10; ++warmup) output = fn();
        torch::cuda::synchronize();
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int iteration = 0; iteration < reps; ++iteration) output = fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return std::pair<float, torch::Tensor>(elapsed / reps, output);
    };

    auto legacy_result = time_ms(legacy);
    auto groupwise_result = time_ms(groupwise);
    auto reference = legacy_result.second.to(torch::kFloat32);
    auto candidate = groupwise_result.second.to(torch::kFloat32);
    auto diff = (candidate - reference).abs();
    const float relative =
        ((candidate - reference).norm() / reference.norm()).item<float>();
    std::cout << std::fixed << std::setprecision(9)
              << "dsv4_output_a_check"
              << " tensor=" << name
              << " format=" << (linear.is_mxfp8() ? "MXFP8" : "NINT8")
              << " batch=" << batch
              << " groups=" << kGroups
              << " rows_per_group=" << rows_per_group
              << " k=" << width
              << " equal=" << (candidate.equal(reference) ? 1 : 0)
              << " rel=" << relative
              << " mean_abs=" << diff.mean().item<float>()
              << " max_abs=" << diff.max().item<float>()
              << " legacy_ms=" << legacy_result.first
              << " groupwise_ms=" << groupwise_result.first
              << " speedup=" << legacy_result.first / groupwise_result.first
              << " checksum=" << candidate.sum().item<double>()
              << "\n";
    if (linear.is_mxfp8()) {
        TORCH_CHECK(
            torch::isfinite(candidate).all().item<bool>() &&
                relative <= 5.0e-4f,
            "DSV4 MXFP8 groupwise output_a exceeded the FP16 GEMM "
            "reduction-order tolerance");
    } else {
        TORCH_CHECK(
            candidate.equal(reference),
            "DSV4 NINT groupwise output_a must be bit-exact with the "
            "legacy path");
    }
    return 0;
}

static int run_gemma_geglu_check(
    const std::string & mfq_path,
    int layer,
    int reps) {
    if (layer < 0 || reps < 1) {
        throw std::runtime_error("Gemma GeGLU check requires a nonnegative layer and positive reps");
    }
    MfqFile mfq(mfq_path);
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer) + ".mlp.";
    auto gate_up = load_quant_group(
        mfq, {prefix + "gate_proj.weight", prefix + "up_proj.weight"}, 2);
    auto down = load_quant_linear(mfq, prefix + "down_proj.weight");
    if (!gate_up.nint_grouped || !gate_up.nint.split_w.empty() || !down.is_nint() ||
        gate_up.outs.size() != 2 || gate_up.outs[0] != gate_up.outs[1]) {
        throw std::runtime_error("Gemma GeGLU check requires packed NINT gate/up and NINT down tensors");
    }

    const int64_t hidden = gate_up.nint.w.neuron_len;
    auto xf = torch::arange(
        hidden, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    auto x = (((xf.remainder(257) - 128.0) / 64.0) +
              0.125 * torch::sin(xf * 0.03125)).to(torch::kFloat16).reshape({1, hidden}).contiguous();

    auto materialized_activation = [&]() {
        auto parts = gate_up.forward(x);
        return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
    };
    auto materialized = [&]() { return down.forward(materialized_activation()); };
    auto fused_activation = [&]() { return gate_up.forward_geglu(x); };
    auto fused = [&]() { return down.forward(fused_activation()); };
    auto fused_quant = [&]() {
        Workspace & gate_ws = gate_up.nint.w.workspace(1);
        Workspace & down_ws = down.nint.w.workspace(1);
        nint_ffn_gate_up_geglu_quant_ws_cuda(
            gate_up.nint.w.q_packed, gate_up.nint.w.sub_scale,
            gate_up.nint.w.sub_min, gate_up.nint.w.neuron_scale,
            gate_up.nint.w.neuron_min, x, gate_up.nint.w.gs,
            gate_up.nint.w.bits, down.nint.w.gs,
            gate_ws.qx, gate_ws.xscale, gate_ws.xsum,
            down_ws.qx, down_ws.xscale, down_ws.xsum);
        return nint_matmul_qx(down.nint.w, down_ws);
    };

    auto reference_activation = materialized_activation();
    auto reference_output = down.forward(reference_activation);
    const bool quant_supported = down.nint.w.gs <= 32;
    std::cout << "gemma_geglu_check layer=" << layer
              << " gate_up_bits=" << gate_up.nint.w.bits
              << " gate_up_gs=" << gate_up.nint.w.gs
              << " gate_out=" << gate_up.outs[0]
              << " up_out=" << gate_up.outs[1]
              << " down_bits=" << down.nint.w.bits
              << " down_gs=" << down.nint.w.gs << "\n";
    auto report = [&](const char * name, torch::Tensor value, torch::Tensor reference) {
        auto got = value.to(torch::kFloat64);
        auto ref = reference.to(torch::kFloat64);
        const double ref_norm = std::max(ref.norm().item<double>(), 1.0e-30);
        const double got_norm = std::max(got.norm().item<double>(), 1.0e-30);
        std::cout << "gemma_geglu_check path=" << name
                  << " relative_l2=" << (got - ref).norm().item<double>() / ref_norm
                  << " cosine=" << torch::dot(got.reshape({-1}), ref.reshape({-1})).item<double>() /
                         (got_norm * ref_norm)
                  << " max_abs=" << (got - ref).abs().max().item<double>() << "\n";
    };

    mfq_set_env("MFQ_NINT_GLU_COMBINED", "1");
    report("activation_combined", fused_activation(), reference_activation);
    report("output_combined", fused(), reference_output);
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "0");
    report("activation_pair", fused_activation(), reference_activation);
    report("output_pair", fused(), reference_output);
    if (quant_supported) {
        report("output_quant", fused_quant(), reference_output);
    }
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "");

    auto time_ms = [&](auto && fn) {
        for (int i = 0; i < 10; ++i) (void)fn();
        torch::cuda::synchronize();
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) (void)fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    const float materialized_ms = time_ms(materialized);
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "1");
    const float combined_ms = time_ms(fused);
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "0");
    const float pair_ms = time_ms(fused);
    const float quant_ms = quant_supported ? time_ms(fused_quant) : 0.0f;
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "");
    std::cout << "gemma_geglu_check layer=" << layer
              << " gate_bits=" << gate_up.nint.w.bits
              << " gate_gs=" << gate_up.nint.w.gs
              << " down_bits=" << down.nint.w.bits
              << " down_gs=" << down.nint.w.gs
              << " materialized_ms=" << materialized_ms
              << " combined_ms=" << combined_ms
              << " pair_ms=" << pair_ms
              << " quant_ms=" << quant_ms << "\n";
    return 0;
}

static double nint_moe_weight_bytes(const NintMoeWeight & weight) {
    double bytes = static_cast<double>(weight.mixed_weight_bytes);
    for (const auto & shard :
         weight.tensor_parallel_shards) {
        if (shard.weight) {
            bytes += nint_moe_weight_bytes(
                *shard.weight);
        }
    }
    for (const auto & pool : weight.pools) {
        bytes += static_cast<double>(pool.weight.q_packed.numel()) * pool.weight.q_packed.element_size();
        bytes += static_cast<double>(pool.weight.sub_scale.numel()) * pool.weight.sub_scale.element_size();
        bytes += static_cast<double>(pool.weight.sub_min.numel()) * pool.weight.sub_min.element_size();
        bytes += static_cast<double>(pool.weight.neuron_scale.numel()) * pool.weight.neuron_scale.element_size();
        bytes += static_cast<double>(pool.weight.neuron_min.numel()) * pool.weight.neuron_min.element_size();
    }
    return bytes;
}

static int run_tensor_parallel_moe_check(
        const std::string & mfq_path,
        const std::string & tensor_name,
        int tokens,
        int routes) {
    TORCH_CHECK(
        g_tensor_parallel.enabled(),
        "--check-tp-moe requires --tensor-parallel");
    TORCH_CHECK(
        tokens >= 1 && tokens <= 4096,
        "--check-tp-moe-tokens must be in [1, 4096]");
    TORCH_CHECK(
        routes >= 1,
        "--check-tp-moe-routes must be positive");
    MfqFile mfq(mfq_path);
    const std::string role =
        tensor_name.find("gate_up") != std::string::npos
        ? "gate_up" : "diagnostic";
    const TensorParallelConfig saved =
        g_tensor_parallel;
    g_tensor_parallel = {};
    g_tensor_parallel.devices = {
        saved.primary_device()};
    auto full = load_nint_moe_gpu(
        mfq, tensor_name, false, 0, role);
    g_tensor_parallel = saved;
    auto sharded = load_nint_moe_gpu(
        mfq, tensor_name, false, 0, role);
    TORCH_CHECK(
        sharded.tensor_parallel(),
        "tensor-parallel MoE diagnostic did not create shards");
    TORCH_CHECK(
        full.n_experts == sharded.n_experts &&
        full.out_per_expert == sharded.out_per_expert &&
        full.neuron_len == sharded.neuron_len,
        "tensor-parallel MoE metadata differs");
    TORCH_CHECK(
        routes <= full.n_experts,
        "--check-tp-moe-routes exceeds the expert count");

    c10::cuda::CUDAGuard primary_guard(
        saved.primary_device());
    const int64_t count =
        static_cast<int64_t>(tokens) *
        full.neuron_len;
    auto sequence = torch::arange(
        count,
        torch::TensorOptions()
            .device(torch::Device(
                torch::kCUDA,
                saved.primary_device()))
            .dtype(torch::kFloat32));
    auto x = (
        (sequence.remainder(257) - 128.0) / 127.0 +
        0.03125 * torch::sin(sequence * 0.015625))
        .to(torch::kFloat16)
        .reshape({tokens, full.neuron_len})
        .contiguous();
    std::vector<int32_t> host_ids(
        static_cast<size_t>(tokens) *
        static_cast<size_t>(routes));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            host_ids[
                static_cast<size_t>(token) * routes +
                static_cast<size_t>(route)] =
                (token * routes + route * 3) %
                full.n_experts;
        }
    }
    auto ids = torch::from_blob(
        host_ids.data(), {tokens, routes},
        torch::TensorOptions().dtype(torch::kInt32))
        .clone()
        .to(torch::Device(
            torch::kCUDA,
            saved.primary_device()))
        .contiguous();
    auto route =
        build_moe_route_plan(
            ids, full.n_experts);
    auto reference =
        full.forward(x, route)
            .to(torch::kFloat32);
    auto test =
        sharded.forward(x, route)
            .to(torch::kFloat32);
    torch::cuda::synchronize();
    auto difference = test - reference;
    const double denominator =
        std::max(
            reference.norm().item<double>(),
            1.0e-30);
    const double relative =
        difference.norm().item<double>() /
        denominator;
    const double mean_abs =
        difference.abs().mean().item<double>();
    const double max_abs =
        difference.abs().max().item<double>();
    const int64_t differing =
        test.ne(reference).sum().item<int64_t>();
    std::cout
        << "tensor_parallel_moe_check=1"
        << " tensor=" << tensor_name
        << " tokens=" << tokens
        << " routes=" << routes
        << " experts=" << full.n_experts
        << " logical_shape=["
        << full.out_per_expert << ','
        << full.neuron_len << ']'
        << " shards="
        << sharded.tensor_parallel_shards.size()
        << " differing=" << differing
        << " relative=" << relative
        << " mean_abs=" << mean_abs
        << " max_abs=" << max_abs
        << '\n';
    if (!torch::isfinite(test).all().item<bool>() ||
        relative > 1.0e-6) {
        throw std::runtime_error(
            "tensor-parallel MoE numerical check failed");
    }
    return 0;
}

static int run_nintm_tensor_check(
        const std::string & mfq_path,
        const std::string & tensor_name,
        int tokens,
        int routes,
        int reps,
        int split_width,
        bool routed_input) {
    if (tokens < 1 || tokens > 4096 || routes < 1 || reps < 1 ||
            split_width < 0) {
        throw std::runtime_error("NINTM tensor check dimensions are invalid");
    }
    MfqFile mfq(mfq_path);
    auto weight = load_nint_moe_gpu(
        mfq, tensor_name, true, 0, "diagnostic");
    if (g_moe_expert_cache &&
            !g_moe_expert_cache->finalized()) {
        g_moe_expert_cache->finalize();
    }
    if (routes > weight.n_experts) {
        throw std::runtime_error("NINTM tensor check routes exceed expert count");
    }
    if (split_width > 0 &&
            (split_width >= weight.out_per_expert ||
             !weight.hetero_supported || weight.mixed_forward)) {
        throw std::runtime_error(
            "NINTM merged/split check requires a pure supported NINT tensor "
            "and an interior split width");
    }
    const int64_t count =
        (int64_t)tokens * (routed_input ? routes : 1) * weight.neuron_len;
    auto sequence = torch::arange(
        count,
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    auto x = (
        (sequence.remainder(257) - 128.0) / 127.0 +
        0.03125 * torch::sin(sequence * 0.015625))
        .to(torch::kFloat16)
        .reshape(
            routed_input
                ? std::vector<int64_t>{tokens, routes, weight.neuron_len}
                : std::vector<int64_t>{tokens, weight.neuron_len})
        .contiguous();
    std::vector<int32_t> host_ids((size_t)tokens * routes);
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            host_ids[(size_t)token * routes + route] =
                (token * routes + route * 3) % weight.n_experts;
        }
    }
    auto ids = torch::from_blob(
        host_ids.data(), {tokens, routes},
        torch::TensorOptions().dtype(torch::kInt32))
        .clone().to(torch::kCUDA).contiguous();
    auto route = build_moe_route_plan(ids, weight.n_experts);
    torch::Tensor output;
    for (int warmup = 0; warmup < 5; ++warmup) {
        output = weight.forward(x, route);
        if (warmup == 0) weight.prefetch(route);
    }
    torch::cuda::synchronize();
    cudaEvent_t start, stop;
    MFQ_CUDA_CHECK(cudaEventCreate(&start));
    MFQ_CUDA_CHECK(cudaEventCreate(&stop));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int index = 0; index < reps; ++index) output = weight.forward(x, route);
    MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(start));
    MFQ_CUDA_CHECK(cudaEventDestroy(stop));
    double dense_reference_rel = -1.0;
    double dense_reference_mean_abs = -1.0;
    double dense_reference_max_abs = -1.0;
    if (split_width == 0) {
        auto cpu_reference = unpack_nint_moe(mfq.read_blob(tensor_name));
        auto reference = torch::empty(
            {tokens * routes, weight.out_per_expert},
            output.options().dtype(torch::kFloat16));
        for (const auto & pool : cpu_reference.pools) {
            torch::Tensor dense_flat;
            int rotation_block = 0;
            torch::Tensor rotation_signs;
            if (pool.dtype == "NINT8-0") {
                auto packed = to_gpu_nint8_zero(pool.q8_zero);
                dense_flat = nint8_zero_dequant_cuda(
                    packed.q_packed, packed.q8_zero_scale,
                    packed.neuron_len);
            } else if (pool.dtype != "NINTM" &&
                       pool.dtype.rfind("NINT", 0) == 0) {
                auto packed = to_gpu_nint(pool.weight);
                dense_flat = packed.bits == 4
                    ? nint_dequant_full_packed_compact_cuda(
                          packed.q_packed, packed.sub_scale,
                          packed.sub_min, packed.neuron_scale,
                          packed.neuron_min, packed.neuron_len,
                          packed.gs)
                    : nint_dequant_full_packed_compact_bits_cuda(
                          packed.q_packed, packed.sub_scale,
                          packed.sub_min, packed.neuron_scale,
                          packed.neuron_min, packed.neuron_len,
                          packed.gs, packed.bits);
            } else if (pool.dtype.rfind("NEPQ", 0) == 0) {
                auto parsed = unpack_nepq(
                    pool.payload, pool.dtype, pool.runtime_payload);
                auto packed = to_gpu_nepq(parsed);
                dense_flat = nepq_dequant_cuda(
                    packed.indices_packed, packed.aux_packed,
                    packed.state_packed, packed.neuron_scale,
                    packed.table_pool, packed.bank_ids,
                    packed.neuron_len, packed.state_bits,
                    packed.format);
                rotation_block = packed.rotation_block;
                rotation_signs = packed.rotation_signs;
            } else {
                auto packed = to_gpu_nvq(
                    unpack_nvq(pool.payload, pool.dtype));
                dense_flat = nvq_dequant(packed);
            }
            auto dense = dense_flat
                .reshape({
                    static_cast<int64_t>(pool.expert_ids.size()),
                    weight.out_per_expert,
                    weight.neuron_len});
            for (size_t local = 0; local < pool.expert_ids.size(); ++local) {
                const int expert = pool.expert_ids[local];
                std::vector<int64_t> pair_indices;
                std::vector<int64_t> token_indices;
                for (int pair = 0; pair < tokens * routes; ++pair) {
                    if (host_ids[static_cast<size_t>(pair)] == expert) {
                        pair_indices.push_back(pair);
                        token_indices.push_back(pair / routes);
                    }
                }
                if (pair_indices.empty()) continue;
                auto pair_index = torch::from_blob(
                    pair_indices.data(),
                    {static_cast<int64_t>(pair_indices.size())},
                    torch::TensorOptions().dtype(torch::kInt64))
                    .clone().to(torch::kCUDA);
                auto token_index = torch::from_blob(
                    token_indices.data(),
                    {static_cast<int64_t>(token_indices.size())},
                    torch::TensorOptions().dtype(torch::kInt64))
                    .clone().to(torch::kCUDA);
                auto selected = routed_input
                    ? x.reshape({tokens * routes, weight.neuron_len})
                          .index_select(0, pair_index)
                    : x.index_select(0, token_index);
                if (rotation_block != 0) {
                    selected = nepq_hadamard_input_cuda(
                        selected.contiguous(), rotation_signs,
                        rotation_block);
                }
                auto expected = torch::matmul(
                    selected, dense.index({static_cast<int64_t>(local)})
                                  .transpose(0, 1));
                reference.index_copy_(0, pair_index, expected);
            }
        }
        torch::cuda::synchronize();
        auto actual_f32 =
            output.reshape({tokens * routes, weight.out_per_expert})
                .to(torch::kFloat32);
        auto reference_f32 = reference.to(torch::kFloat32);
        auto difference = actual_f32 - reference_f32;
        dense_reference_rel =
            (difference.norm() / reference_f32.norm()).item<double>();
        dense_reference_mean_abs =
            difference.abs().mean().item<double>();
        dense_reference_max_abs =
            difference.abs().max().item<double>();
    }
    auto flat = output.to(torch::kFloat32).cpu().reshape({-1});
    if (!torch::isfinite(flat).all().item<bool>()) {
        throw std::runtime_error("NINTM tensor check produced a non-finite value");
    }
    std::cout << std::fixed << std::setprecision(9)
              << "nintm_tensor_check"
              << " tensor=" << tensor_name
              << " tokens=" << tokens
              << " routes=" << routes
              << " experts=" << weight.n_experts
              << " out=" << weight.out_per_expert
              << " k=" << weight.neuron_len
              << " routed_input=" << (routed_input ? 1 : 0)
              << " mixed=" << (weight.mixed_forward ? 1 : 0)
              << " hetero=" << (weight.hetero_supported ? 1 : 0)
              << " weight_bytes=" << nint_moe_weight_bytes(weight)
              << " cuda_ms=" << elapsed / reps
              << " dense_reference_rel=" << dense_reference_rel
              << " dense_reference_mean_abs=" << dense_reference_mean_abs
              << " dense_reference_max_abs=" << dense_reference_max_abs
              << " checksum=" << flat.sum().item<double>()
              << " sqsum=" << flat.square().sum().item<double>()
              << " values=";
    const int64_t shown = std::min<int64_t>(flat.numel(), 128);
    const float * values = flat.data_ptr<float>();
    for (int64_t index = 0; index < shown; ++index) {
        if (index) std::cout << ",";
        std::cout << values[index];
    }
    std::cout << "\n";

    if (split_width > 0) {
        if (tokens <= 8 || !route.map_ready ||
                route.ids_dst.numel() != route.ids.numel()) {
            throw std::runtime_error(
                "NINTM merged/split check requires the grouped MMA route map");
        }
        auto run_segment = [&](int width, int row_offset) {
            auto segment = torch::empty(
                {tokens, routes, width},
                x.options().dtype(torch::kFloat16));
            return nint_moe_grouped_matmul_hetero_f16_slice_cuda(
                weight.weight_ptrs, weight.pool_params,
                weight.expert_pool, weight.expert_local,
                x, route.ids, weight.n_experts, width,
                weight.neuron_len, false, segment,
                route.ids_dst, route.expert_bounds,
                route.tile_bounds, route.tile_experts,
                weight.out_per_expert, row_offset);
        };
        torch::Tensor merged;
        torch::Tensor left;
        torch::Tensor right;
        for (int warmup = 0; warmup < 3; ++warmup) {
            merged = run_segment(weight.out_per_expert, 0);
            left = run_segment(split_width, 0);
            right = run_segment(
                weight.out_per_expert - split_width, split_width);
        }
        torch::cuda::synchronize();
        auto split = torch::cat({left, right}, 2).contiguous();
        auto difference =
            merged.to(torch::kFloat32) - split.to(torch::kFloat32);
        const int64_t left_differing =
            merged.slice(2, 0, split_width).ne(left).sum().item<int64_t>();
        const int64_t right_differing =
            merged.slice(2, split_width, weight.out_per_expert)
                .ne(right).sum().item<int64_t>();
        const int64_t differing = left_differing + right_differing;
        const double merged_norm = merged.to(torch::kFloat32).norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "nintm_merged_split_check"
                  << " tensor=" << tensor_name
                  << " path=nint_moe_hetero_mma_kernel"
                  << " tokens=" << tokens
                  << " routes=" << routes
                  << " experts=" << weight.n_experts
                  << " bm=" << (tokens <= 128 ? 64 : (tokens <= 512 ? 32 : 64))
                  << " bn=64"
                  << " k=" << weight.neuron_len
                  << " full_width=" << weight.out_per_expert
                  << " split_width=" << split_width
                  << " right_width=" << weight.out_per_expert - split_width
                  << " weight_out_stride=" << weight.out_per_expert
                  << " left_row_offset=0"
                  << " right_row_offset=" << split_width
                  << " equal=" << (merged.equal(split) ? 1 : 0)
                  << " differing=" << differing
                  << " values=" << merged.numel()
                  << " left_differing=" << left_differing
                  << " right_differing=" << right_differing
                  << " rel_l2="
                  << (merged_norm == 0.0
                          ? 0.0
                          : difference.norm().item<double>() / merged_norm)
                  << " mean_abs=" << difference.abs().mean().item<double>()
                  << " max_abs=" << difference.abs().max().item<double>()
                  << "\n";
    }

    const char * warp_ab_env = std::getenv("MFQ_CHECK_NVQ_MOE_WARP_AB");
    if (warp_ab_env != nullptr && std::atoi(warp_ab_env) != 0) {
        const char * original_env = std::getenv("MFQ_NVQ_MOE_WARPS");
        const bool had_original_env = original_env != nullptr;
        const std::string original_value = had_original_env ? original_env : "";
        const char * original_exact_env =
            std::getenv("MFQ_NVQ_MOE_EXACT_REDUCTION");
        const bool had_original_exact_env = original_exact_env != nullptr;
        const std::string original_exact_value =
            had_original_exact_env ? original_exact_env : "";
        auto set_env = [](const char * name, const char * value) {
#ifdef _WIN32
            _putenv_s(name, value);
#else
            setenv(name, value, 1);
#endif
        };
        auto restore_env = [&](const char * name, bool existed, const std::string & value) {
#ifdef _WIN32
            _putenv_s(name, existed ? value.c_str() : "");
#else
            if (existed) {
                setenv(name, value.c_str(), 1);
            } else {
                unsetenv(name);
            }
#endif
        };
        set_env("MFQ_NVQ_MOE_WARPS", "0");
        set_env("MFQ_NVQ_MOE_EXACT_REDUCTION", "1");
        auto candidate = weight.forward(x, route);
        set_env("MFQ_NVQ_MOE_EXACT_REDUCTION", "0");
        auto baseline = weight.forward(x, route);
        torch::cuda::synchronize();
        restore_env(
            "MFQ_NVQ_MOE_WARPS", had_original_env, original_value);
        restore_env(
            "MFQ_NVQ_MOE_EXACT_REDUCTION",
            had_original_exact_env, original_exact_value);
        auto candidate_f32 = candidate.to(torch::kFloat32);
        auto baseline_f32 = baseline.to(torch::kFloat32);
        auto diff = candidate_f32 - baseline_f32;
        const double baseline_norm = baseline_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "nvq_moe_warp_ab"
                  << " candidate_physical_warps=2"
                  << " baseline_warps=" << (weight.neuron_len >= 4096 ? 8 : 4)
                  << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                  << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                  << " rel_l2=" << diff.norm().item<double>() / baseline_norm
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    }

    const char * clamped_ab_env =
        std::getenv("MFQ_CHECK_NINTM_CLAMPED_SWIGLU_AB");
    if (clamped_ab_env != nullptr && weight.supports_clamped_swiglu()) {
        const double limit = std::atof(clamped_ab_env);
        if (!std::isfinite(limit) || limit <= 0.0) {
            throw std::runtime_error(
                "MFQ_CHECK_NINTM_CLAMPED_SWIGLU_AB must be positive");
        }
        const int64_t gate_up_count =
            static_cast<int64_t>(tokens) * routes * 2 * weight.neuron_len;
        auto gate_up_sequence = torch::arange(
            gate_up_count,
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        auto gate_up = (
            12.0 * torch::sin(gate_up_sequence * 0.013671875) +
            0.5 * torch::cos(gate_up_sequence * 0.00390625))
            .to(torch::kFloat16)
            .reshape({tokens, routes, 2 * weight.neuron_len})
            .contiguous();
        auto run_candidate = [&]() {
            return weight.forward_clamped_swiglu(gate_up, route, limit);
        };
        auto run_baseline = [&]() {
            const int64_t width = weight.neuron_len;
            auto gate = torch::clamp_max(
                gate_up.slice(-1, 0, width).to(torch::kFloat32), limit);
            auto up = torch::clamp(
                gate_up.slice(-1, width, 2 * width).to(torch::kFloat32),
                -limit, limit);
            auto hidden = (torch::silu(gate) * up)
                .to(torch::kFloat16).contiguous();
            return weight.forward(hidden, route);
        };
        torch::Tensor candidate;
        torch::Tensor baseline;
        for (int warmup = 0; warmup < 5; ++warmup) {
            candidate = run_candidate();
            baseline = run_baseline();
        }
        torch::cuda::synchronize();
        auto time_ms = [&](auto && fn) {
            cudaEvent_t begin, end;
            MFQ_CUDA_CHECK(cudaEventCreate(&begin));
            MFQ_CUDA_CHECK(cudaEventCreate(&end));
            auto cuda_stream = at::cuda::getCurrentCUDAStream().stream();
            MFQ_CUDA_CHECK(cudaEventRecord(begin, cuda_stream));
            torch::Tensor value;
            for (int iteration = 0; iteration < reps; ++iteration) {
                value = fn();
            }
            MFQ_CUDA_CHECK(cudaEventRecord(end, cuda_stream));
            MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float elapsed_ms = 0.0f;
            MFQ_CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_ms, begin, end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));
            MFQ_CUDA_CHECK(cudaEventDestroy(end));
            return std::pair<double, torch::Tensor>(
                elapsed_ms / reps, std::move(value));
        };
        auto candidate_timing = time_ms(run_candidate);
        auto baseline_timing = time_ms(run_baseline);
        candidate = std::move(candidate_timing.second);
        baseline = std::move(baseline_timing.second);
        auto candidate_f32 = candidate.to(torch::kFloat32);
        auto baseline_f32 = baseline.to(torch::kFloat32);
        auto diff = candidate_f32 - baseline_f32;
        const double baseline_norm = baseline_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "nintm_clamped_swiglu_ab"
                  << " limit=" << limit
                  << " candidate_ms=" << candidate_timing.first
                  << " baseline_ms=" << baseline_timing.first
                  << " speedup=" << baseline_timing.first / candidate_timing.first
                  << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                  << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                  << " rel_l2=" << diff.norm().item<double>() / baseline_norm
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    }
    if (g_moe_expert_cache) {
        g_moe_expert_cache->print_stats(std::cout);
    }
    return 0;
}

static int run_gemma_moe_check(
        const MfqFile & mfq,
        const Config & config,
        int layer,
        const std::vector<int64_t> & token_sizes,
        int reps) {
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer) + ".experts.";
    auto gate_up = load_nint_moe_gpu(
        mfq, prefix + "gate_up_proj",
        true, layer, "gate_up");
    auto down = load_nint_moe_gpu(
        mfq, prefix + "down_proj",
        true, layer, "down");
    if (g_moe_expert_cache &&
            !g_moe_expert_cache->finalized()) {
        g_moe_expert_cache->finalize();
    }
    const int routes = static_cast<int>(config.num_experts_per_tok);
    const int experts = static_cast<int>(config.num_experts);
    TORCH_CHECK(routes > 0 && routes <= 8 && experts > routes,
        "Gemma MoE benchmark requires 1..8 routes and more experts than routes");
    const char * dense_reference_env =
        std::getenv("MFQ_CHECK_GEMMA_MOE_DENSE_REFERENCE");
    const bool dense_reference_enabled =
        dense_reference_env != nullptr && std::atoi(dense_reference_env) != 0;
    auto materialize_dense_moe = [&](const std::string & name) {
        auto cpu = unpack_nint_moe(mfq.read_blob(name));
        auto dense = torch::empty(
            {cpu.n_experts, cpu.out_per_expert, cpu.neuron_len},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
        for (const auto & pool : cpu.pools) {
            torch::Tensor local_flat;
            if (pool.dtype == "NINT8-0") {
                auto packed = to_gpu_nint8_zero(pool.q8_zero);
                local_flat = nint8_zero_dequant_cuda(
                    packed.q_packed, packed.q8_zero_scale,
                    packed.neuron_len);
            } else if (pool.dtype.rfind("NINT", 0) == 0) {
                auto packed = to_gpu_nint(pool.weight);
                local_flat = packed.bits == 4
                    ? nint_dequant_full_packed_compact_cuda(
                          packed.q_packed, packed.sub_scale,
                          packed.sub_min, packed.neuron_scale,
                          packed.neuron_min, packed.neuron_len,
                          packed.gs)
                    : nint_dequant_full_packed_compact_bits_cuda(
                          packed.q_packed, packed.sub_scale,
                          packed.sub_min, packed.neuron_scale,
                          packed.neuron_min, packed.neuron_len,
                          packed.gs, packed.bits);
            } else {
                throw std::runtime_error(
                    "Gemma dense MoE reference requires NINT cohorts");
            }
            auto local = local_flat.reshape({
                static_cast<int64_t>(pool.expert_ids.size()),
                cpu.out_per_expert, cpu.neuron_len});
            auto expert_index = torch::from_blob(
                const_cast<int32_t *>(pool.expert_ids.data()),
                {static_cast<int64_t>(pool.expert_ids.size())},
                torch::TensorOptions().dtype(torch::kInt32))
                .clone().to(torch::kCUDA).to(torch::kInt64);
            dense.index_copy_(0, expert_index, local);
        }
        return dense;
    };
    torch::Tensor dense_gate_up;
    torch::Tensor dense_down;
    if (dense_reference_enabled) {
        dense_gate_up = materialize_dense_moe(prefix + "gate_up_proj");
        dense_down = materialize_dense_moe(prefix + "down_proj");
    }
    std::cout << "gemma_moe_bench_config"
              << " layer=" << layer
              << " experts=" << experts
              << " top_k=" << routes
              << " hidden=" << gate_up.neuron_len
              << " intermediate=" << down.neuron_len
              << " gate_up_pools=" << gate_up.pools.size()
              << " down_pools=" << down.pools.size()
              << " routed_weight_bytes=" << std::fixed << std::setprecision(0)
              << nint_moe_weight_bytes(gate_up) + nint_moe_weight_bytes(down)
              << "\n";

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto time_ms = [&](auto && fn, int iterations) {
        torch::Tensor output;
        for (int warmup = 0; warmup < 5; ++warmup) output = fn();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < iterations; ++iteration) output = fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return std::pair<double, torch::Tensor>(elapsed / iterations, output);
    };

    auto topk_logits = torch::randn(
        {1, experts}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    auto selected = moe_topk_cuda(
        topk_logits, routes, false, false, false, true, c10::nullopt, 1e-20, 1.0);
    auto reference = torch::topk(topk_logits, routes, 1, true, true);
    auto reference_weights = torch::softmax(std::get<0>(reference), 1);
    torch::cuda::synchronize();
    auto topk_timing = time_ms([&]() {
        return moe_topk_cuda(
            topk_logits, routes, false, false, false, true,
            c10::nullopt, 1e-20, 1.0).at(1);
    }, reps);
    std::cout << std::fixed << std::setprecision(6)
              << "gemma_topk_check"
              << " ids_equal="
              << (selected.at(0).equal(std::get<1>(reference).to(torch::kInt32)) ? 1 : 0)
              << " weights_max_abs="
              << (selected.at(1) - reference_weights).abs().max().item<float>()
              << " cuda_ms=" << topk_timing.first << "\n";

    torch::manual_seed(20260721 + layer);
    for (int64_t tokens : token_sizes) {
        auto x = torch::randn(
            {tokens, gate_up.neuron_len},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
        std::vector<int32_t> host_ids(static_cast<size_t>(tokens) * routes);
        for (int64_t token = 0; token < tokens; ++token) {
            for (int route_index = 0; route_index < routes; ++route_index) {
                host_ids[static_cast<size_t>(token) * routes + route_index] =
                    static_cast<int32_t>(
                        (token * routes + route_index) % experts);
            }
        }
        auto ids = torch::from_blob(
            host_ids.data(), {tokens, routes}, torch::TensorOptions().dtype(torch::kInt32))
            .clone().to(torch::kCUDA).contiguous();
        auto weights = torch::full(
            {tokens, routes}, 1.0 / routes,
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        auto forward_materialized = [&]() {
            auto route = build_moe_route_plan(ids, experts);
            auto gate_pair = gate_up.forward(x, route);
            auto hidden = moe_geglu_split_cuda(gate_pair);
            auto down_pair = down.forward(hidden, route);
            return moe_weighted_reduce_cuda(down_pair, weights);
        };
        auto forward_gate_glu = [&]() {
            auto route = build_moe_route_plan(ids, experts);
            auto hidden = gate_up.forward_glu_output(x, route, true);
            auto down_pair = down.forward(hidden, route);
            return moe_weighted_reduce_cuda(down_pair, weights);
        };
        auto forward = [&]() {
            return tokens <= 4 ? forward_gate_glu() : forward_materialized();
        };
        auto fused_check = forward();
        auto materialized_check = forward_materialized();
        auto gate_glu_check = tokens <= 4 ? forward_gate_glu() : fused_check;
        torch::cuda::synchronize();
        auto fused_diff = (fused_check - materialized_check).abs().to(torch::kFloat32);
        auto fused_time = time_ms(forward, reps);
        auto materialized_time = time_ms(forward_materialized, reps);
        auto gate_glu_time = tokens <= 4 ? time_ms(forward_gate_glu, reps) : fused_time;
        auto gate_glu_diff = (gate_glu_check - materialized_check).abs().to(torch::kFloat32);
        std::cout << std::fixed << std::setprecision(6)
                  << "gemma_moe_geglu_quant_fusion"
                  << " tokens=" << tokens
                  << " equal=" << (fused_check.equal(materialized_check) ? 1 : 0)
                  << " max_abs=" << fused_diff.max().item<float>()
                  << " fused_ms=" << fused_time.first
                  << " materialized_ms=" << materialized_time.first
                  << " speedup=" << materialized_time.first / fused_time.first
                  << " gate_glu_equal=" << (gate_glu_check.equal(materialized_check) ? 1 : 0)
                  << " gate_glu_max_abs=" << gate_glu_diff.max().item<float>()
                  << " gate_glu_ms=" << gate_glu_time.first << "\n";

        if (tokens == 1) {
            auto stage_route = build_moe_route_plan(ids, experts);
            auto stage_hidden = gate_up.forward_glu_output(x, stage_route, true);
            auto stage_down = down.forward(stage_hidden, stage_route);
            torch::cuda::synchronize();
            auto gate_stage = time_ms(
                [&]() { return gate_up.forward_glu_output(x, stage_route, true); }, reps);
            auto down_stage = time_ms(
                [&]() { return down.forward(stage_hidden, stage_route); }, reps);
            auto reduce_stage = time_ms(
                [&]() { return moe_weighted_reduce_cuda(stage_down, weights); }, reps);
            std::cout << "gemma_moe_stage"
                      << " gate_up_geglu_ms=" << gate_stage.first
                      << " down_ms=" << down_stage.first
                      << " reduce_ms=" << reduce_stage.first << "\n";
        }

        if (tokens != 1) {
            g_force_moe_prefill_mma_off = false;
            nint_moe_set_small_mmq_cuda(1);
            auto mma = time_ms(forward, reps);
            auto mma_first = mma.second.clone();
            auto mma_repeat = forward().clone();
            torch::cuda::synchronize();
            g_force_moe_prefill_mma_off = true;
            nint_moe_set_small_mmq_cuda(0);
            auto baseline = time_ms(forward, reps);
            torch::cuda::synchronize();
            g_force_moe_prefill_mma_off = false;
            nint_moe_set_small_mmq_cuda(-1);
            auto repeat_diff = (mma_repeat - mma_first).abs().to(torch::kFloat32);
            auto baseline_diff = (mma_first - baseline.second).abs().to(torch::kFloat32);
            std::cout << std::setprecision(6)
                      << "gemma_moe_prefill_ab"
                      << " tokens=" << tokens
                      << " mma_ms=" << mma.first
                      << " baseline_ms=" << baseline.first
                      << " speedup=" << baseline.first / mma.first
                      << " repeat_equal=" << (mma_repeat.equal(mma_first) ? 1 : 0)
                      << " repeat_max_abs=" << repeat_diff.max().item<float>()
                      << " baseline_rel="
                      << ((mma_first - baseline.second).to(torch::kFloat32).norm() /
                          baseline.second.to(torch::kFloat32).norm()).item<float>()
                      << " baseline_max_abs=" << baseline_diff.max().item<float>()
                      << "\n";
            if (dense_reference_enabled) {
                auto dense_route_forward = [&](const torch::Tensor & dense,
                                               const torch::Tensor & input) {
                    auto result = torch::empty(
                        {tokens, routes, dense.size(1)},
                        input.options().dtype(torch::kFloat16));
                    for (int route_index = 0; route_index < routes; ++route_index) {
                        const int expert = (route_index * 17) % experts;
                        auto selected = input.dim() == 3
                            ? input.select(1, route_index)
                            : input;
                        result.select(1, route_index).copy_(torch::matmul(
                            selected,
                            dense.index({expert}).transpose(0, 1)));
                    }
                    return result;
                };
                auto dense_gate_pair =
                    dense_route_forward(dense_gate_up, x);
                auto dense_hidden =
                    moe_geglu_split_cuda(dense_gate_pair);
                auto dense_down_pair =
                    dense_route_forward(dense_down, dense_hidden);
                auto dense_output =
                    moe_weighted_reduce_cuda(dense_down_pair, weights);
                auto dense_difference =
                    mma_first.to(torch::kFloat32) -
                    dense_output.to(torch::kFloat32);
                const double dense_norm =
                    dense_output.to(torch::kFloat32).norm().item<double>();
                std::cout << std::scientific << std::setprecision(9)
                          << "gemma_moe_dense_reference"
                          << " tokens=" << tokens
                          << " rel_l2="
                          << dense_difference.norm().item<double>() / dense_norm
                          << " mean_abs="
                          << dense_difference.abs().mean().item<double>()
                          << " max_abs="
                          << dense_difference.abs().max().item<double>()
                          << "\n";
            }
        }
    }
    if (g_moe_expert_cache) {
        g_moe_expert_cache->print_stats(std::cout);
    }
    return 0;
}

static int run_moe_check(
        const std::string & mfq_path,
        const std::string & config_path,
        int layer,
        const std::vector<int64_t> & token_sizes,
        int reps) {
    if (layer < 0) throw std::runtime_error("--check-moe-layer must be nonnegative");
    if (reps < 1) throw std::runtime_error("--check-moe-reps must be positive");
    if (token_sizes.empty() || std::any_of(token_sizes.begin(), token_sizes.end(),
            [](int64_t value) { return value < 1 || value > 4096; })) {
        throw std::runtime_error("--check-moe-tokens values must be in [1, 4096]");
    }
    MfqFile mfq(mfq_path);
    Config config = load_config(mfq, config_path);
    if (layer >= config.num_hidden_layers) throw std::runtime_error("MoE benchmark layer is out of range");
    if (config.is_gemma4()) {
        return run_gemma_moe_check(mfq, config, layer, token_sizes, reps);
    }
    FFN ffn = load_ffn(mfq, config, layer, false);
    if (!ffn.is_moe) throw std::runtime_error("selected layer does not contain NINTM MoE weights");
    if (g_moe_expert_cache &&
            !g_moe_expert_cache->finalized()) {
        g_moe_expert_cache->finalize();
    }
    const double routed_weight_bytes =
        nint_moe_weight_bytes(ffn.moe_gate_up) + nint_moe_weight_bytes(ffn.moe_down);
    std::cout << "moe_bench_config"
              << " layer=" << layer
              << " experts=" << ffn.moe_gate_up.n_experts
              << " top_k=" << ffn.moe_top_k
              << " hidden=" << ffn.moe_gate_up.neuron_len
              << " intermediate=" << ffn.moe_down.neuron_len
              << " gate_up_pools=" << ffn.moe_gate_up.pools.size()
              << " down_pools=" << ffn.moe_down.pools.size()
              << " routed_weight_bytes=" << std::fixed << std::setprecision(0)
              << routed_weight_bytes << "\n";

    torch::manual_seed(20260720 + layer);
    torch::Tensor output;
    for (int64_t tokens : token_sizes) {
        auto x = torch::randn(
            {tokens, config.hidden_size},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
        for (int warmup = 0; warmup < 10; ++warmup) output = ffn.forward(x);
        torch::cuda::synchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        auto wall_start = std::chrono::steady_clock::now();
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < reps; ++iteration) output = ffn.forward(x);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        auto wall_stop = std::chrono::steady_clock::now();
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        const double cuda_ms = static_cast<double>(elapsed_ms) / reps;
        const double wall_ms =
            std::chrono::duration<double, std::milli>(wall_stop - wall_start).count() / reps;
        const double checksum = output.to(torch::kFloat32).sum().item<double>();
        if (!std::isfinite(checksum)) throw std::runtime_error("non-finite MoE benchmark output");
        std::cout << std::setprecision(6)
                  << "moe_bench_result"
                  << " layer=" << layer
                  << " tokens=" << tokens
                  << " reps=" << reps
                  << " cuda_ms=" << cuda_ms
                  << " wall_ms=" << wall_ms
                  << " tokens_per_second=" << (1000.0 * tokens / cuda_ms)
                  << " checksum=" << checksum << "\n";

        const char * prefill_ab_env = std::getenv("MFQ_CHECK_MOE_PREFILL_MMA_AB");
        if (prefill_ab_env != nullptr && std::atoi(prefill_ab_env) != 0 && tokens >= 9) {
            auto candidate = output.clone();
            g_force_moe_prefill_mma_off = true;
            auto baseline = ffn.forward(x);
            torch::cuda::synchronize();
            g_force_moe_prefill_mma_off = false;
            auto diff = (candidate - baseline).to(torch::kFloat32);
            const double baseline_norm = baseline.to(torch::kFloat32).norm().item<double>();
            std::cout << "moe_prefill_mma_ab"
                      << " tokens=" << tokens
                      << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                      << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                      << " rel_l2=" << (diff.norm().item<double>() / baseline_norm)
                      << " mean_abs=" << diff.abs().mean().item<float>()
                      << " max_abs=" << diff.abs().max().item<float>()
                      << "\n";
        }

        const char * small_mmq_ab_env = std::getenv("MFQ_CHECK_MOE_SMALL_MMQ_AB");
        if (small_mmq_ab_env != nullptr && std::atoi(small_mmq_ab_env) != 0 &&
                tokens >= 16 && tokens <= 128) {
            nint_moe_set_small_mmq_cuda(1);
            auto candidate = ffn.forward(x);
            nint_moe_set_small_mmq_cuda(0);
            auto baseline = ffn.forward(x);
            torch::cuda::synchronize();
            nint_moe_set_small_mmq_cuda(-1);
            auto candidate_f32 = candidate.to(torch::kFloat32);
            auto baseline_f32 = baseline.to(torch::kFloat32);
            auto diff = candidate_f32 - baseline_f32;
            const double baseline_norm = baseline_f32.norm().item<double>();
            std::cout << "moe_small_mmq_ab"
                      << " tokens=" << tokens
                      << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                      << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                      << " rel_l2=" << (diff.norm().item<double>() / baseline_norm)
                      << " mean_abs=" << diff.abs().mean().item<float>()
                      << " max_abs=" << diff.abs().max().item<float>()
                      << " candidate_checksum=" << candidate_f32.sum().item<double>()
                      << " baseline_checksum=" << baseline_f32.sum().item<double>()
                      << "\n";
        }

        const char * exact_env = std::getenv("MFQ_CHECK_MOE_POOL_EXACT");
        if (exact_env != nullptr && std::atoi(exact_env) != 0 && tokens <= 8) {
            auto candidate = output;
            g_force_moe_pool_path = true;
            g_force_moe_unfused_reduce = true;
            g_force_moe_materialized_swiglu = true;
            auto baseline = ffn.forward(x);
            torch::cuda::synchronize();
            g_force_moe_pool_path = false;
            g_force_moe_unfused_reduce = false;
            g_force_moe_materialized_swiglu = false;
            auto diff = (candidate - baseline).abs().to(torch::kFloat32);
            std::cout << "moe_exact_result"
                      << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                      << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                      << " max_abs=" << diff.max().item<float>()
                      << "\n";
        }

        g_profiler.reset();
        g_profiler.enabled = true;
        const int profile_reps = std::min(reps, 10);
        for (int iteration = 0; iteration < profile_reps; ++iteration) output = ffn.forward(x);
        torch::cuda::synchronize();
        g_profiler.report("moe_layer" + std::to_string(layer) + "_m" + std::to_string(tokens));
        g_profiler.enabled = false;
        g_profiler.reset();
    }
    if (g_moe_expert_cache) {
        g_moe_expert_cache->print_stats(std::cout);
    }
    return 0;
}

static int run_attention_decode_check(int length, int reps, int D, bool sliding, int window) {
    if (length < 1 || length > 32768) {
        throw std::runtime_error("--check-attention-decode must be in [1, 32768]");
    }
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    constexpr int B = 1;
    constexpr int Hq = 16;
    constexpr int max_parts = 256;
    if (D != 256 && D != 512) {
        throw std::runtime_error("--check-attention-head-dim must be 256 or 512");
    }
    if (sliding && D != 256) {
        throw std::runtime_error("SWA decode check currently requires head_dim 256");
    }
    const int Hk = sliding ? 8 : 2;
    const int kv_tile = D == 512 ? 32 : 64;
    const int visible_len = sliding ? std::min(length, window) : length;
    const int max_seq = sliding ? window : (length + kv_tile - 1) / kv_tile * kv_tile;
    const int parts = std::min(max_parts, std::max(1, (length + 127) / 128));
    auto cuda = torch::TensorOptions().device(torch::kCUDA);
    torch::manual_seed(20260720);
    auto q = torch::randn({B, Hq, 1, D}, cuda.dtype(torch::kFloat32));
    auto k = torch::randn({B, Hk, max_seq, D}, cuda.dtype(torch::kFloat16));
    auto v = torch::randn({B, Hk, max_seq, D}, cuda.dtype(torch::kFloat16));
    auto seq_len = torch::tensor({length}, cuda.dtype(torch::kInt64));
    auto partial_o = torch::empty({B * Hq, max_parts, D}, cuda.dtype(torch::kFloat32));
    auto partial_m = torch::empty({B * Hq, max_parts}, cuda.dtype(torch::kFloat32));
    auto partial_l = torch::empty({B * Hq, max_parts}, cuda.dtype(torch::kFloat32));
    auto qh = q.to(torch::kFloat16);
    const int mask_stride = (visible_len + kv_tile - 1) / kv_tile * kv_tile;
    const int ntiles_kv = (visible_len + kv_tile - 1) / kv_tile;
    const int64_t max_blocks = B * Hk * ntiles_kv;
    const int64_t meta_float2 = max_blocks * 8 * (2 + D / 2);
    auto mask = torch::empty({B, mask_stride}, cuda.dtype(torch::kFloat16));
    auto kv_max = torch::empty({B}, cuda.dtype(torch::kInt32));
    auto meta = torch::empty({2 * meta_float2}, cuda.dtype(torch::kFloat32));
    const double scale = 1.0 / std::sqrt((double)D);
    auto run_ref = [&]() {
        if (sliding) {
            return attention_cache_swa_planned_cuda(
                qh, k, v, seq_len, scale, window, visible_len);
        }
        return parts > 1
            ? attention_cache_decode_split_cuda(
                  qh, k, v, seq_len, scale, partial_o, partial_m, partial_l, parts)
            : attention_cache_decode_cuda(qh, k, v, seq_len, scale);
    };
    auto run_test = [&]() {
        if (sliding) {
            return attention_llama_flash256_swa_decode_cuda(
                q, k, v, seq_len, scale, visible_len, mask, kv_max, meta);
        }
        return D == 512
            ? attention_llama_flash512_decode_cuda(
                q, k, v, seq_len, scale, length, mask, kv_max, meta)
            : attention_llama_flash256_decode_cuda(
                q, k, v, seq_len, scale, length, mask, kv_max, meta);
    };
    torch::Tensor ref, test;
    for (int i = 0; i < 10; ++i) {
        ref = run_ref();
        test = run_test();
    }
    torch::cuda::synchronize();
    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cudaEventRecord(start, stream);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return elapsed / reps;
    };
    const float ref_ms = time_ms(run_ref);
    const float test_ms = time_ms(run_test);
    ref = run_ref().to(torch::kFloat32);
    test = run_test().permute({0, 2, 1, 3}).contiguous();
    torch::cuda::synchronize();
    auto diff = (test - ref).abs();
    std::cout << "attention_decode_check mode=" << (sliding ? "swa" : "full")
              << " head_dim=" << D << " length=" << length
              << " parts=" << parts << " ref_ms=" << ref_ms
              << " test_ms=" << test_ms
              << " speedup=" << ref_ms / test_ms << "\n";
    std::cout << "attention_decode_rel="
              << ((test - ref).norm() / ref.norm()).item<float>() << "\n";
    std::cout << "attention_decode_mean_abs=" << diff.mean().item<float>() << "\n";
    std::cout << "attention_decode_max_abs=" << diff.max().item<float>() << "\n";
    std::cout << "attention_decode_test_finite="
              << (torch::isfinite(test).all().item<bool>() ? 1 : 0) << "\n";
    return 0;
}

static int run_gemma4_swa_check(int reps) {
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    constexpr int B = 1;
    constexpr int Hq = 32;
    constexpr int Hk = 16;
    constexpr int D = 256;
    const double scale = 1.0 / std::sqrt((double)D);
    auto cuda = torch::TensorOptions().device(torch::kCUDA);
    torch::manual_seed(20260721);

    struct Shape {
        int tokens;
        int window;
    };
    const Shape shapes[] = {
        {1, 1}, {17, 5}, {33, 17}, {73, 32}, {256, 128}, {1024, 1024},
    };
    double worst_rel = 0.0;
    double worst_mean_abs = 0.0;
    double worst_max_abs = 0.0;
    for (const auto shape : shapes) {
        auto q = torch::randn({B, Hq, shape.tokens, D}, cuda.dtype(torch::kFloat32));
        auto k = torch::randn({B, Hk, shape.tokens, D}, cuda.dtype(torch::kFloat16));
        auto v = torch::randn({B, Hk, shape.tokens, D}, cuda.dtype(torch::kFloat16));
        auto ref = attention_swa_cuda(
            q, k.to(torch::kFloat32), v.to(torch::kFloat32), scale, shape.window);
        auto test = attention_llama_flash256_swa_cuda(q, k, v, scale, shape.window)
            .permute({0, 2, 1, 3}).contiguous();
        torch::cuda::synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const double mean_abs = diff.mean().item<double>();
        const double max_abs = diff.max().item<double>();
        const bool finite = torch::isfinite(test).all().item<bool>();
        worst_rel = std::max(worst_rel, rel);
        worst_mean_abs = std::max(worst_mean_abs, mean_abs);
        worst_max_abs = std::max(worst_max_abs, max_abs);
        std::cout << "gemma4_swa_check tokens=" << shape.tokens
                  << " window=" << shape.window
                  << " rel=" << rel
                  << " mean_abs=" << mean_abs
                  << " max_abs=" << max_abs
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("Gemma4 SWA FlashAttention numerical check failed");
        }
    }

    for (const int tokens : {33, 256}) {
        constexpr int full_hq = 16;
        constexpr int full_hk = 4;
        auto q = torch::randn({B, full_hq, tokens, D}, cuda.dtype(torch::kFloat32));
        auto k = torch::randn({B, full_hk, tokens, D}, cuda.dtype(torch::kFloat16));
        auto v = torch::randn({B, full_hk, tokens, D}, cuda.dtype(torch::kFloat16));
        auto ref = attention_cuda(
            q, k.to(torch::kFloat32), v.to(torch::kFloat32), scale, true);
        auto test = attention_llama_flash256_cuda(q, k, v, scale)
            .permute({0, 2, 1, 3}).contiguous();
        torch::cuda::synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = torch::isfinite(test).all().item<bool>();
        std::cout << "flash256_full_check tokens=" << tokens
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("full FlashAttention numerical regression failed");
        }
    }

    for (const int tokens : {256}) {
        constexpr int full_hq = 16;
        constexpr int full_hk = 2;
        constexpr int full_d = 512;
        auto q = torch::randn({B, full_hq, tokens, full_d}, cuda.dtype(torch::kFloat32));
        auto k = torch::randn({B, full_hk, tokens, full_d}, cuda.dtype(torch::kFloat16));
        auto v = torch::randn({B, full_hk, tokens, full_d}, cuda.dtype(torch::kFloat16));
        auto ref = attention_cuda(
            q, k.to(torch::kFloat32), v.to(torch::kFloat32), 1.0, true);
        auto test = attention_llama_flash512_cuda(q, k, v, 1.0)
            .permute({0, 2, 1, 3}).contiguous();
        torch::cuda::synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = torch::isfinite(test).all().item<bool>();
        std::cout << "gemma4_flash512_check tokens=" << tokens
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("Gemma4 D512 FlashAttention numerical check failed");
        }
    }

    constexpr int bench_tokens = 256;
    constexpr int bench_window = 128;
    auto q = torch::randn({B, Hq, bench_tokens, D}, cuda.dtype(torch::kFloat32));
    auto k = torch::randn({B, Hk, bench_tokens, D}, cuda.dtype(torch::kFloat16));
    auto v = torch::randn({B, Hk, bench_tokens, D}, cuda.dtype(torch::kFloat16));
    auto kf = k.to(torch::kFloat32);
    auto vf = v.to(torch::kFloat32);
    auto run_ref = [&]() { return attention_swa_cuda(q, kf, vf, scale, bench_window); };
    auto run_test = [&]() {
        return attention_llama_flash256_swa_cuda(q, k, v, scale, bench_window);
    };
    for (int i = 0; i < 10; ++i) {
        run_ref();
        run_test();
    }
    torch::cuda::synchronize();
    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    const float ref_ms = time_ms(run_ref);
    const float test_ms = time_ms(run_test);
    std::cout << "gemma4_swa_bench tokens=" << bench_tokens
              << " window=" << bench_window
              << " generic_ms=" << ref_ms
              << " llama_mma_ms=" << test_ms
              << " speedup=" << ref_ms / test_ms << "\n";

    std::cout << "gemma4_swa_worst_rel=" << worst_rel
              << " worst_mean_abs=" << worst_mean_abs
              << " worst_max_abs=" << worst_max_abs << "\n";
    return 0;
}

static int run_glm_dsa_check(int reps) {
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    torch::NoGradGuard no_grad;
    auto cuda = torch::TensorOptions().device(torch::kCUDA);
    torch::manual_seed(20260722);

    {
        constexpr int B = 2, H = 3, T = 5, D = 128, RD = 64;
        auto x = torch::randn({B, H, T, D}, cuda.dtype(torch::kFloat32));
        auto pos = torch::arange(7, 7 + T, cuda.dtype(torch::kInt64));
        auto freq = torch::pow(
            torch::full({RD / 2}, 8000000.0, cuda.dtype(torch::kFloat32)),
            -torch::arange(0, RD, 2, cuda.dtype(torch::kFloat32)) / (double)RD);
        auto angle = torch::arange(32, cuda.dtype(torch::kFloat32)).unsqueeze(1) * freq.unsqueeze(0);
        auto cos = torch::cos(angle).contiguous();
        auto sin = torch::sin(angle).contiguous();
        auto test = glm_interleaved_rope_cuda(x, pos, cos, sin, RD);
        auto paired = x.index({Slice(), Slice(), Slice(), Slice(0, RD)})
            .reshape({B, H, T, RD / 2, 2});
        auto c = cos.index_select(0, pos).reshape({1, 1, T, RD / 2});
        auto s = sin.index_select(0, pos).reshape({1, 1, T, RD / 2});
        auto rotated = torch::stack({
            paired.select(-1, 0) * c - paired.select(-1, 1) * s,
            paired.select(-1, 1) * c + paired.select(-1, 0) * s,
        }, -1).reshape({B, H, T, RD});
        auto ref = torch::cat({
            rotated, x.index({Slice(), Slice(), Slice(), Slice(RD, D)})}, -1);
        const double max_abs = (test - ref).abs().max().item<double>();
        std::cout << "glm_rope_max_abs=" << max_abs << "\n";
        if (max_abs > 2e-6) throw std::runtime_error("GLM interleaved RoPE numerical check failed");
    }

    {
        constexpr int ROWS = 17, D = 128;
        auto x = torch::randn({ROWS, D}, cuda.dtype(torch::kFloat16));
        auto weight = torch::randn({D}, cuda.dtype(torch::kFloat32));
        auto bias = torch::randn({D}, cuda.dtype(torch::kFloat32));
        auto test = glm_dsa_indexer_layer_norm_cuda(x, weight, bias, 1e-5);
        auto ref = torch::layer_norm(
            x.to(torch::kFloat32), {D}, weight, bias, 1e-5).to(torch::kFloat16);
        const double max_abs = (test.to(torch::kFloat32) - ref.to(torch::kFloat32))
            .abs().max().item<double>();
        std::cout << "glm_indexer_layer_norm_max_abs=" << max_abs << "\n";
        if (max_abs > 0.004) {
            throw std::runtime_error("GLM indexer LayerNorm numerical check failed");
        }
    }

    {
        constexpr int ROWS = 3, EXPERTS = 256, TOPK = 8;
        auto logits = torch::randn({ROWS, EXPERTS}, cuda.dtype(torch::kFloat32));
        auto bias = torch::randn({EXPERTS}, cuda.dtype(torch::kFloat32));
        auto selected = moe_topk_cuda(
            logits, TOPK, true, false, true, false, bias, 1e-20, 2.5);
        auto sigmoid = torch::sigmoid(logits);
        auto expected_topk = torch::topk(
            sigmoid + bias.unsqueeze(0), TOPK, 1, true, true);
        auto expected_ids = std::get<1>(expected_topk);
        auto expected_weights = sigmoid.gather(1, expected_ids);
        expected_weights = expected_weights /
            expected_weights.sum(1, true).clamp_min(1e-20) * 2.5;
        const bool ids_equal = selected[0].to(torch::kInt64).equal(expected_ids);
        const double max_abs = (selected[1] - expected_weights)
            .abs().max().item<double>();
        std::cout << "glm_moe_route_ids_equal=" << (ids_equal ? 1 : 0)
                  << " max_abs=" << max_abs << "\n";
        if (!ids_equal || max_abs > 2e-6) {
            throw std::runtime_error("GLM MoE routing numerical check failed");
        }
    }

    {
        constexpr int B = 1, M = 3, K = 2112, H = 32, D = 128;
        auto q = torch::randn({B, M, H, D}, cuda.dtype(torch::kFloat16));
        auto k = torch::randn({B, K, D}, cuda.dtype(torch::kFloat16));
        auto weights = torch::randn({B, M, H}, cuda.dtype(torch::kFloat32));
        const int offset = K - M;
        auto test = glm_dsa_indexer_scores_cuda(q, k, weights, offset, K);
        auto heads = torch::einsum(
            "bmhd,bkd->bmhk", {q.to(torch::kFloat32), k.to(torch::kFloat32)}) /
            std::sqrt(128.0);
        auto ref = (torch::relu(heads) * weights.unsqueeze(-1)).sum(2) / std::sqrt(32.0);
        auto key_pos = torch::arange(K, cuda.dtype(torch::kInt64)).reshape({1, 1, K});
        auto query_pos = torch::arange(offset, offset + M, cuda.dtype(torch::kInt64)).reshape({1, M, 1});
        ref = ref.masked_fill(key_pos > query_pos, -std::numeric_limits<float>::infinity());
        auto finite = torch::isfinite(ref);
        auto diff = torch::where(finite, (test - ref).abs(), torch::zeros_like(ref));
        const double rel = torch::where(finite, test - ref, torch::zeros_like(ref)).norm().item<double>() /
            std::max(torch::where(finite, ref, torch::zeros_like(ref)).norm().item<double>(), 1e-30);
        const double max_abs = diff.max().item<double>();
        std::cout << "glm_indexer_rel=" << rel << " max_abs=" << max_abs << "\n";
        if (!torch::isneginf(test.masked_select(~finite)).all().item<bool>() || rel > 0.003) {
            throw std::runtime_error("GLM indexer score numerical check failed");
        }
    }

    {
        constexpr int B = 1, M = 1, VISIBLE = 2112, PLANNED = 2176;
        constexpr int H = 32, D = 128;
        auto q = torch::randn({B, M, H, D}, cuda.dtype(torch::kFloat16));
        auto k = torch::randn({B, PLANNED, D}, cuda.dtype(torch::kFloat16));
        auto weights = torch::randn({B, M, H}, cuda.dtype(torch::kFloat32));
        auto seq_len = torch::tensor({VISIBLE}, cuda.dtype(torch::kInt64));
        auto test = glm_dsa_indexer_scores_decode_cuda(
            q, k, weights, seq_len, PLANNED);
        auto ref = (torch::relu(torch::einsum(
            "bmhd,bkd->bmhk",
            {q.to(torch::kFloat32), k.to(torch::kFloat32)}) / std::sqrt(128.0)) *
            weights.unsqueeze(-1)).sum(2) / std::sqrt(32.0);
        ref.index({Slice(), Slice(), Slice(VISIBLE, PLANNED)})
            .fill_(-std::numeric_limits<float>::infinity());
        auto finite = torch::isfinite(ref);
        const double rel = torch::where(
            finite, test - ref, torch::zeros_like(ref)).norm().item<double>() /
            std::max(torch::where(
                finite, ref, torch::zeros_like(ref)).norm().item<double>(), 1e-30);
        const bool future_inf = torch::isneginf(
            test.index({Slice(), Slice(), Slice(VISIBLE, PLANNED)})).all().item<bool>();
        std::cout << "glm_indexer_decode_rel=" << rel
                  << " future_inf=" << (future_inf ? 1 : 0) << "\n";
        if (rel > 0.003 || !future_inf) {
            throw std::runtime_error("GLM decode indexer numerical check failed");
        }
    }

    {
        constexpr int B = 1, H = 64, T = 33, DQ = 576, DV = 512;
        const double scale = 1.0 / std::sqrt(256.0);
        auto q = torch::randn({B, H, T, DQ}, cuda.dtype(torch::kFloat32));
        auto kv = torch::randn({B, 1, T, DQ}, cuda.dtype(torch::kFloat16));
        auto kv_cache = torch::zeros({B, 1, 64, DQ}, cuda.dtype(torch::kFloat16));
        kv_cache.index({Slice(), Slice(), Slice(0, T), Slice()}).copy_(kv);
        auto mask = torch::empty({T, 64}, cuda.dtype(torch::kFloat16));
        auto kv_max = torch::empty({B * ((T + 3) / 4)}, cuda.dtype(torch::kInt32));
        auto meta = torch::empty({8 * 1024 * 1024}, cuda.dtype(torch::kFloat32));
        auto test = attention_glm_mla576_cached_cuda(
            q, kv_cache, T, mask, kv_max, meta, scale);
        auto q_ref = q.to(torch::kFloat16).to(torch::kFloat32);
        auto k_ref = kv.to(torch::kFloat32).expand({B, H, T, DQ});
        auto scores = torch::matmul(q_ref, k_ref.transpose(-1, -2)) * scale;
        auto causal = torch::ones({T, T}, cuda.dtype(torch::kBool)).tril();
        scores = scores.masked_fill(~causal, -std::numeric_limits<float>::infinity());
        auto ref = torch::matmul(
            torch::softmax(scores, -1),
            k_ref.index({Slice(), Slice(), Slice(), Slice(0, DV)}))
            .permute({0, 2, 1, 3}).contiguous();
        const double rel = (test - ref).norm().item<double>() / ref.norm().item<double>();
        const double max_abs = (test - ref).abs().max().item<double>();
        std::cout << "glm_dense_mla_rel=" << rel << " max_abs=" << max_abs << "\n";
        if (!torch::isfinite(test).all().item<bool>() || rel > 0.02) {
            throw std::runtime_error("GLM dense MLA numerical check failed");
        }
    }

    {
        constexpr int B = 1, H = 64, VISIBLE = 33, PLANNED = 64;
        constexpr int DQ = 576, DV = 512;
        const double scale = 1.0 / std::sqrt(256.0);
        auto q = torch::randn({B, H, 1, DQ}, cuda.dtype(torch::kFloat32));
        auto kv = torch::randn({B, 1, PLANNED, DQ}, cuda.dtype(torch::kFloat16));
        auto seq_len = torch::tensor({VISIBLE}, cuda.dtype(torch::kInt64));
        auto mask = torch::empty({B, PLANNED}, cuda.dtype(torch::kFloat16));
        auto kv_max = torch::empty({B}, cuda.dtype(torch::kInt32));
        auto meta = torch::empty({8 * 1024 * 1024}, cuda.dtype(torch::kFloat32));
        auto test = attention_glm_mla576_decode_cuda(
            q, kv, seq_len, scale, PLANNED, mask, kv_max, meta);
        auto selected = kv.index({Slice(), Slice(), Slice(0, VISIBLE), Slice()})
            .to(torch::kFloat32).expand({B, H, VISIBLE, DQ});
        auto ref = torch::matmul(
            torch::softmax(torch::matmul(q, selected.transpose(-1, -2)) * scale, -1),
            selected.index({Slice(), Slice(), Slice(), Slice(0, DV)}))
            .permute({0, 2, 1, 3}).contiguous();
        const double rel = (test - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "glm_dense_decode_rel=" << rel << "\n";
        if (!torch::isfinite(test).all().item<bool>() || rel > 0.02) {
            throw std::runtime_error("GLM dense decode numerical check failed");
        }
    }

    constexpr int B = 1, H = 64, M = 3, K = 2112, TOPK = 2048, DQ = 576, DV = 512;
    const double scale = 1.0 / std::sqrt(256.0);
    auto q = torch::randn({B, H, M, DQ}, cuda.dtype(torch::kFloat32));
    auto kv = torch::randn({B, K, DQ}, cuda.dtype(torch::kFloat16));
    std::vector<torch::Tensor> rows;
    rows.reserve(M);
    for (int row = 0; row < M; ++row) {
        rows.push_back(torch::randperm(K - M + row + 1, cuda.dtype(torch::kInt64))
            .narrow(0, 0, TOPK).to(torch::kInt32));
    }
    auto indices = torch::stack(rows, 0).unsqueeze(0).contiguous();
    auto meta = torch::empty({8 * 1024 * 1024}, cuda.dtype(torch::kFloat32));
    auto run_sparse = [&]() {
        return attention_glm_mla_sparse_cuda(q, kv, indices, meta, scale);
    };
    auto test = run_sparse();
    std::vector<torch::Tensor> refs;
    refs.reserve(M);
    auto q_ref = q.to(torch::kFloat16).to(torch::kFloat32);
    for (int row = 0; row < M; ++row) {
        auto idx = indices.index({0, row}).to(torch::kInt64);
        auto selected = kv.index_select(1, idx).index({0}).to(torch::kFloat32);
        auto score = torch::matmul(q_ref.index({0, Slice(), row}), selected.transpose(0, 1)) * scale;
        refs.push_back(torch::matmul(
            torch::softmax(score, -1), selected.index({Slice(), Slice(0, DV)})));
    }
    auto ref = torch::stack(refs, 0).unsqueeze(0);
    torch::cuda::synchronize();
    const double rel = (test - ref).norm().item<double>() / ref.norm().item<double>();
    const double max_abs = (test - ref).abs().max().item<double>();
    std::cout << "glm_sparse_mla_rel=" << rel << " max_abs=" << max_abs << "\n";
    if (!torch::isfinite(test).all().item<bool>() || rel > 0.02) {
        throw std::runtime_error("GLM sparse MLA numerical check failed");
    }

    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    for (int i = 0; i < 5; ++i) run_sparse();
    const float sparse_ms = time_ms(run_sparse);
    std::cout << "glm_sparse_mla_m=" << M << " k=" << K
              << " topk=" << TOPK << " cuda_ms=" << sparse_ms << "\n";
    return 0;
}

static int run_dsv4_hc_check(int reps) {
    if (reps < 1) {
        throw std::runtime_error("--check-attention-reps must be positive");
    }
    torch::NoGradGuard no_grad;
    auto cuda = torch::TensorOptions().device(torch::kCUDA);
    constexpr int64_t hc = 4;
    constexpr int64_t hidden = 4096;
    constexpr int64_t mix_width = 24;
    constexpr double eps = 1e-6;

    auto x_sequence = torch::arange(
        hc * hidden, cuda.dtype(torch::kFloat32));
    auto x = (
        (x_sequence.remainder(251) - 125.0) / 64.0 +
        0.125 * torch::sin(x_sequence * 0.015625))
        .to(torch::kFloat16)
        .reshape({1, 1, hc, hidden})
        .contiguous();
    auto function_sequence = torch::arange(
        mix_width * hc * hidden, cuda.dtype(torch::kFloat32));
    auto function = (
        0.003 * torch::sin(function_sequence * 0.001953125) +
        0.001 * torch::cos(function_sequence * 0.0078125))
        .reshape({mix_width, hc * hidden})
        .contiguous();
    auto scale = (
        0.7 + 0.2 * torch::arange(3, cuda.dtype(torch::kFloat32)))
        .contiguous();
    auto base = (
        0.1 * torch::sin(
            torch::arange(
                mix_width, cuda.dtype(torch::kFloat32)) * 0.3125))
        .contiguous();
    auto flat = x.flatten(2).to(torch::kFloat32);
    auto inverse_rms = torch::rsqrt(
        flat.square().mean(-1, true) + eps);
    auto mixes = (
        torch::matmul(flat, function.transpose(0, 1)) * inverse_rms)
        .contiguous();

    auto reference_split = dsv4_hc_split_sinkhorn(
        mixes, scale, base, hc, 20, eps);
    auto reference_reduced = (
        reference_split.at(0).unsqueeze(-1) *
        flat.reshape({1, 1, hc, hidden}))
        .sum(2).to(torch::kFloat16).contiguous();
    auto candidate = dsv4_hc_pre_cuda(
        x, mixes, scale, base, 20, eps);

    auto direct = x.index(
        {Slice(), Slice(), 0, Slice()}).contiguous();
    auto reference_post = (
        reference_split.at(1).unsqueeze(-1) *
            direct.unsqueeze(-2) +
        (reference_split.at(2).unsqueeze(-1) *
            x.to(torch::kFloat32).unsqueeze(-2)).sum(2))
        .to(torch::kFloat16).contiguous();
    auto candidate_post = dsv4_hc_post_cuda(
        direct, x, candidate.at(1), candidate.at(2));
    torch::cuda::synchronize();

    auto compare = [](const char * name,
                      torch::Tensor reference,
                      torch::Tensor value) {
        auto reference_f32 = reference.to(torch::kFloat32);
        auto value_f32 = value.to(torch::kFloat32);
        auto diff = value_f32 - reference_f32;
        const double norm = reference_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "dsv4_hc_ab tensor=" << name
                  << " equal=" << (value.equal(reference) ? 1 : 0)
                  << " differing="
                  << value.ne(reference).sum().item<int64_t>()
                  << " rel_l2="
                  << (norm == 0.0
                      ? 0.0
                      : diff.norm().item<double>() / norm)
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    };
    compare("reduced", reference_reduced, candidate.at(0));
    compare("post", reference_split.at(1), candidate.at(1));
    compare("combination", reference_split.at(2), candidate.at(2));
    compare("hc_post", reference_post, candidate_post);

    auto time_ms = [&](auto && fn) {
        for (int warmup = 0; warmup < 10; ++warmup) fn();
        torch::cuda::synchronize();
        cudaEvent_t begin, end;
        MFQ_CUDA_CHECK(cudaEventCreate(&begin));
        MFQ_CUDA_CHECK(cudaEventCreate(&end));
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(begin, stream));
        for (int iteration = 0; iteration < reps; ++iteration) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(end, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
        MFQ_CUDA_CHECK(cudaEventDestroy(begin));
        MFQ_CUDA_CHECK(cudaEventDestroy(end));
        return elapsed / reps;
    };
    const float reference_pre_ms = time_ms([&]() {
        auto split = dsv4_hc_split_sinkhorn(
            mixes, scale, base, hc, 20, eps);
        return (
            split.at(0).unsqueeze(-1) *
            flat.reshape({1, 1, hc, hidden}))
            .sum(2).to(torch::kFloat16).contiguous();
    });
    const float candidate_pre_ms = time_ms([&]() {
        return dsv4_hc_pre_cuda(
            x, mixes, scale, base, 20, eps);
    });
    const float reference_post_ms = time_ms([&]() {
        return (
            reference_split.at(1).unsqueeze(-1) *
                direct.unsqueeze(-2) +
            (reference_split.at(2).unsqueeze(-1) *
                x.to(torch::kFloat32).unsqueeze(-2)).sum(2))
            .to(torch::kFloat16).contiguous();
    });
    const float candidate_post_ms = time_ms([&]() {
        return dsv4_hc_post_cuda(
            direct, x, candidate.at(1), candidate.at(2));
    });
    std::cout << std::fixed << std::setprecision(9)
              << "dsv4_hc_timing"
              << " reference_pre_ms=" << reference_pre_ms
              << " candidate_pre_ms=" << candidate_pre_ms
              << " pre_speedup=" << reference_pre_ms / candidate_pre_ms
              << " reference_post_ms=" << reference_post_ms
              << " candidate_post_ms=" << candidate_post_ms
              << " post_speedup=" << reference_post_ms / candidate_post_ms
              << "\n";
    return 0;
}

static int run_dsv4_attention_check(int reps) {
    if (reps < 1) {
        throw std::runtime_error("--check-attention-reps must be positive");
    }
    torch::NoGradGuard no_grad;
    auto cuda = torch::TensorOptions().device(torch::kCUDA);
    torch::manual_seed(20260723);

    {
        constexpr int EXPERTS = 256, TOPK = 6;
        auto logits = torch::randn({1, EXPERTS}, cuda.dtype(torch::kFloat32)) * 3.0;
        auto bias = torch::randn({EXPERTS}, cuda.dtype(torch::kFloat32)) * 0.05;
        auto selected = moe_topk_cuda(
            logits, TOPK, false, true, true, false, bias, 1e-20, 1.5);
        auto transformed = torch::sqrt(torch::where(
            logits > 20.0, logits, torch::log1p(torch::exp(logits))));
        auto expected_topk = torch::topk(
            transformed + bias.unsqueeze(0), TOPK, 1, true, true);
        auto expected_ids = std::get<1>(expected_topk);
        auto expected_weights = transformed.gather(1, expected_ids);
        expected_weights = expected_weights /
            expected_weights.sum(1, true).clamp_min(1e-20) * 1.5;
        const bool ids_equal =
            selected.at(0).to(torch::kInt64).equal(expected_ids);
        const double max_abs = (selected.at(1) - expected_weights)
            .abs().max().item<double>();

        auto hash_ids = torch::randint(
            0, EXPERTS, {7, TOPK}, cuda.dtype(torch::kInt32));
        auto hash_logits =
            torch::randn({7, EXPERTS}, cuda.dtype(torch::kFloat32)) * 3.0;
        auto hash_weights = moe_sqrtsoftplus_weights_cuda(
            hash_logits, hash_ids, 1e-20, 1.5);
        auto hash_transformed = torch::sqrt(torch::where(
            hash_logits > 20.0, hash_logits,
            torch::log1p(torch::exp(hash_logits))));
        auto expected_hash = hash_transformed.gather(
            1, hash_ids.to(torch::kInt64));
        expected_hash = expected_hash /
            expected_hash.sum(1, true).clamp_min(1e-20) * 1.5;
        const double hash_max_abs = (hash_weights - expected_hash)
            .abs().max().item<double>();
        std::cout << "dsv4_moe_route_ids_equal=" << (ids_equal ? 1 : 0)
                  << " max_abs=" << max_abs
                  << " hash_max_abs=" << hash_max_abs << "\n";
        if (!ids_equal || max_abs > 2e-6 || hash_max_abs > 2e-6) {
            throw std::runtime_error(
                "DeepSeek V4 MoE routing numerical check failed");
        }
    }

    constexpr int B = 1;
    constexpr int D = 512;
    constexpr int RD = 64;
    constexpr double eps = 1e-6;
    auto norm = torch::randn({D}, cuda.dtype(torch::kFloat32));

    {
        constexpr int W = 2, R = 128;
        auto kv = torch::randn({B, W, R, D}, cuda.dtype(torch::kFloat16));
        auto gate = torch::randn({B, W, R, D}, cuda.dtype(torch::kFloat16));
        auto ape = torch::randn({R, D}, cuda.dtype(torch::kFloat32));
        auto empty = torch::empty({0}, cuda.dtype(torch::kFloat16));
        auto positions = torch::arange(W, cuda.dtype(torch::kInt64)).reshape({B, W});
        auto cos = torch::ones({W + 1, RD / 2}, cuda.dtype(torch::kFloat32));
        auto sin = torch::zeros_like(cos);
        auto test = dsv4_compress_cuda(
            kv, gate, ape, norm, empty, empty, positions, cos, sin,
            R, false, 1, eps);
        auto score = gate.to(torch::kFloat32) +
            ape.reshape({1, 1, R, D});
        auto pooled = (kv.to(torch::kFloat32) *
            torch::softmax(score, 2)).sum(2);
        auto ref = pooled * torch::rsqrt(
            pooled.square().mean(-1, true) + eps) * norm;
        ref = ref.to(torch::kBFloat16).to(torch::kFloat32);
        auto ref_nope = ref.slice(-1, 0, D - RD)
            .reshape({-1, (D - RD) / 64, 64});
        auto fp8_scale = ref_nope.abs().amax(-1, true)
            .clamp_min(1e-4) / 448.0;
        ref_nope = (ref_nope / fp8_scale)
            .clamp(-448.0, 448.0)
            .to(c10::kFloat8_e4m3fn)
            .to(torch::kFloat32) * fp8_scale;
        ref.slice(-1, 0, D - RD).copy_(
            ref_nope.reshape({B, W, D - RD}));
        ref = ref.to(torch::kBFloat16)
            .to(torch::kFloat16).to(torch::kFloat32);
        const double rel =
            (test.to(torch::kFloat32) - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "dsv4_hca_compressor_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 ratio-128 compressor numerical check failed");
        }
    }

    {
        constexpr int W = 3, R = 4, OD = 2 * D;
        auto kv = torch::randn({B, W, R, OD}, cuda.dtype(torch::kFloat16));
        auto gate = torch::randn({B, W, R, OD}, cuda.dtype(torch::kFloat16));
        auto prev_kv = torch::randn({B, R, D}, cuda.dtype(torch::kFloat16));
        auto prev_gate = torch::randn({B, R, D}, cuda.dtype(torch::kFloat16));
        auto ape = torch::randn({R, OD}, cuda.dtype(torch::kFloat32));
        auto positions = torch::arange(W, cuda.dtype(torch::kInt64)).reshape({B, W});
        auto cos = torch::ones({W + 1, RD / 2}, cuda.dtype(torch::kFloat32));
        auto sin = torch::zeros_like(cos);
        auto test = dsv4_compress_cuda(
            kv, gate, ape, norm, prev_kv, prev_gate, positions,
            cos, sin, R, true, 0, eps);
        std::vector<torch::Tensor> ref_rows;
        for (int w = 0; w < W; ++w) {
            auto left_kv = w == 0
                ? prev_kv
                : kv.index({Slice(), w - 1, Slice(), Slice(0, D)});
            auto left_gate = w == 0
                ? prev_gate
                : gate.index({Slice(), w - 1, Slice(), Slice(0, D)});
            auto right_kv =
                kv.index({Slice(), w, Slice(), Slice(D, OD)});
            auto right_gate =
                gate.index({Slice(), w, Slice(), Slice(D, OD)});
            auto values = torch::cat({left_kv, right_kv}, 1)
                .to(torch::kFloat32);
            auto score = torch::cat({
                left_gate.to(torch::kFloat32) +
                    ape.index({Slice(), Slice(0, D)}).unsqueeze(0),
                right_gate.to(torch::kFloat32) +
                    ape.index({Slice(), Slice(D, OD)}).unsqueeze(0)}, 1);
            auto pooled = (values * torch::softmax(score, 1)).sum(1);
            ref_rows.push_back(
                pooled * torch::rsqrt(
                    pooled.square().mean(-1, true) + eps) * norm);
        }
        auto ref = torch::stack(ref_rows, 1)
            .to(torch::kFloat16).to(torch::kFloat32);
        const double rel =
            (test.to(torch::kFloat32) - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "dsv4_csa_overlap_compressor_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 ratio-4 overlap compressor numerical check failed");
        }
    }

    {
        constexpr int T = 12, R = 4, W = T / R, OD = 2 * D;
        auto kv = torch::randn({B, T, OD}, cuda.dtype(torch::kFloat32));
        auto gate = torch::randn({B, T, OD}, cuda.dtype(torch::kFloat32));
        auto ape = torch::randn({R, OD}, cuda.dtype(torch::kFloat32));
        auto empty = torch::empty({0}, cuda.dtype(torch::kFloat16));
        auto positions = torch::arange(W, cuda.dtype(torch::kInt64)).reshape({B, W});
        auto cos = torch::ones({W + 1, RD / 2}, cuda.dtype(torch::kFloat32));
        auto sin = torch::zeros_like(cos);
        auto batch = dsv4_compress_cuda(
            kv.reshape({B, W, R, OD}).contiguous(),
            gate.reshape({B, W, R, OD}).contiguous(),
            ape, norm, empty, empty, positions, cos, sin,
            R, true, 1, eps);
        auto state_kv = torch::zeros(
            {B, R, OD}, cuda.dtype(torch::kFloat32));
        auto state_gate = torch::zeros_like(state_kv);
        auto prev_kv = torch::zeros(
            {B, R, D}, cuda.dtype(torch::kFloat32));
        auto prev_gate = torch::zeros_like(prev_kv);
        auto pool = torch::zeros(
            {B, W, D}, cuda.dtype(torch::kFloat16));
        auto seq_len = torch::zeros({B}, cuda.dtype(torch::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            dsv4_decode_pool_update_cuda(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                ape, norm, state_kv, state_gate, prev_kv, prev_gate,
                pool, seq_len, cos, sin, R, true, 1, eps);
        }
        const double rel =
            (pool.to(torch::kFloat32) - batch.to(torch::kFloat32))
                .norm().item<double>() /
            std::max(batch.to(torch::kFloat32).norm().item<double>(), 1e-30);
        std::cout << "dsv4_decode_pool_state_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 decode compressor state check failed");
        }
    }

    {
        constexpr int ID = 128, T = 12, R = 4, W = T / R;
        constexpr int OD = 2 * ID;
        auto index_norm = torch::randn(
            {ID}, cuda.dtype(torch::kFloat32));
        auto kv = torch::randn(
            {B, T, OD}, cuda.dtype(torch::kFloat32));
        auto gate = torch::randn(
            {B, T, OD}, cuda.dtype(torch::kFloat32));
        auto ape = torch::randn(
            {R, OD}, cuda.dtype(torch::kFloat32));
        auto empty = torch::empty({0}, cuda.dtype(torch::kFloat16));
        auto positions = torch::arange(
            W, cuda.dtype(torch::kInt64)).reshape({B, W});
        auto cos = torch::ones(
            {W + 1, RD / 2}, cuda.dtype(torch::kFloat32));
        auto sin = torch::zeros_like(cos);
        auto batch = dsv4_compress_cuda(
            kv.reshape({B, W, R, OD}).contiguous(),
            gate.reshape({B, W, R, OD}).contiguous(),
            ape, index_norm, empty, empty, positions, cos, sin,
            R, true, 2, eps);
        auto state_kv = torch::zeros(
            {B, R, OD}, cuda.dtype(torch::kFloat32));
        auto state_gate = torch::zeros_like(state_kv);
        auto prev_kv = torch::zeros(
            {B, R, ID}, cuda.dtype(torch::kFloat32));
        auto prev_gate = torch::zeros_like(prev_kv);
        auto pool = torch::zeros(
            {B, W, ID}, cuda.dtype(torch::kFloat16));
        auto seq_len = torch::zeros(
            {B}, cuda.dtype(torch::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            dsv4_decode_pool_update_cuda(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                ape, index_norm, state_kv, state_gate,
                prev_kv, prev_gate, pool, seq_len, cos, sin,
                R, true, 2, eps);
        }
        const double rel =
            (pool.to(torch::kFloat32) - batch.to(torch::kFloat32))
                .norm().item<double>() /
            std::max(
                batch.to(torch::kFloat32).norm().item<double>(),
                1e-30);
        std::cout << "dsv4_indexer_pool_state_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 indexer compressor state check failed");
        }
    }

    {
        constexpr int ID = 128, T = 14, R = 4, OD = 2 * ID;
        auto index_norm = torch::randn(
            {ID}, cuda.dtype(torch::kFloat32));
        auto kv = torch::randn(
            {B, T, OD}, cuda.dtype(torch::kFloat32));
        auto gate = torch::randn(
            {B, T, OD}, cuda.dtype(torch::kFloat32));
        Dsv4RopeTable rope;
        rope.cos = torch::ones(
            {T + 1, RD / 2}, cuda.dtype(torch::kFloat32));
        rope.sin = torch::zeros_like(rope.cos);
        rope.negative_sin = -rope.sin;

        Dsv4PoolState batched;
        batched.ratio = R;
        batched.head_dim = ID;
        batched.overlap = true;
        batched.cache_quant_mode = 2;
        batched.ape = torch::randn(
            {R, OD}, cuda.dtype(torch::kFloat32));
        batched.norm = index_norm;
        batched.reset(B, T);

        Dsv4PoolState decoded;
        decoded.ratio = batched.ratio;
        decoded.head_dim = batched.head_dim;
        decoded.overlap = batched.overlap;
        decoded.cache_quant_mode = batched.cache_quant_mode;
        decoded.ape = batched.ape;
        decoded.norm = batched.norm;
        decoded.reset(B, T);

        const int64_t windows = batched.prefill(kv, gate, rope);
        auto seq_len = torch::zeros(
            {B}, cuda.dtype(torch::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            decoded.update(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                seq_len, rope);
        }
        const auto pool_width = T / R;
        auto reference_pool = decoded.pool.narrow(1, 0, pool_width);
        auto candidate_pool = batched.pool.narrow(1, 0, pool_width);
        const double pool_rel =
            (candidate_pool.to(torch::kFloat32) -
             reference_pool.to(torch::kFloat32)).norm().item<double>() /
            std::max(
                reference_pool.to(torch::kFloat32).norm().item<double>(),
                1e-30);
        const double state_rel =
            (batched.state_kv - decoded.state_kv)
                .norm().item<double>() /
            std::max(decoded.state_kv.norm().item<double>(), 1e-30);
        const double gate_rel =
            (batched.state_gate - decoded.state_gate)
                .norm().item<double>() /
            std::max(decoded.state_gate.norm().item<double>(), 1e-30);
        const double previous_rel =
            (batched.previous_kv - decoded.previous_kv)
                .norm().item<double>() /
            std::max(decoded.previous_kv.norm().item<double>(), 1e-30);
        const double previous_gate_rel =
            (batched.previous_gate - decoded.previous_gate)
                .norm().item<double>() /
            std::max(decoded.previous_gate.norm().item<double>(), 1e-30);
        std::cout << "dsv4_pool_prefill_windows=" << windows
                  << " pool_rel=" << pool_rel
                  << " state_rel=" << state_rel
                  << " gate_rel=" << gate_rel
                  << " previous_rel=" << previous_rel
                  << " previous_gate_rel=" << previous_gate_rel << "\n";
        if (windows != pool_width || pool_rel > 0.002 ||
            state_rel > 1e-7 || gate_rel > 1e-7 ||
            previous_rel > 1e-7 || previous_gate_rel > 1e-7) {
            throw std::runtime_error(
                "DeepSeek V4 batched compressor prefill state check failed");
        }
    }

    {
        constexpr int ROWS = 9, WIDTH = 128;
        auto input = torch::randn(
            {ROWS, WIDTH}, cuda.dtype(torch::kFloat16)) * 2.0;
        auto test = dsv4_fp4_sim_cuda(input.contiguous());
        auto grouped = input.to(torch::kFloat32)
            .reshape({ROWS, WIDTH / 32, 32});
        auto scale = torch::exp2(torch::ceil(torch::log2(
            grouped.abs().amax(-1, true)
                .clamp_min(6.0 * std::ldexp(1.0, -126)) / 6.0)));
        auto normalized = (grouped / scale).clamp(-6.0, 6.0);
        auto magnitude = normalized.abs();
        auto quantized = torch::where(
            magnitude <= 0.25, torch::zeros_like(magnitude),
            torch::where(
                magnitude < 0.75, torch::full_like(magnitude, 0.5),
                torch::where(
                    magnitude <= 1.25, torch::ones_like(magnitude),
                    torch::where(
                        magnitude < 1.75,
                        torch::full_like(magnitude, 1.5),
                        torch::where(
                            magnitude <= 2.5,
                            torch::full_like(magnitude, 2.0),
                            torch::where(
                                magnitude < 3.5,
                                torch::full_like(magnitude, 3.0),
                                torch::where(
                                    magnitude <= 5.0,
                                    torch::full_like(magnitude, 4.0),
                                    torch::full_like(
                                        magnitude, 6.0))))))));
        quantized = torch::where(
            normalized < 0, -quantized, quantized);
        auto reference = (quantized * scale)
            .reshape({ROWS, WIDTH}).to(torch::kFloat16);
        const double max_abs = (test - reference)
            .abs().max().item<double>();
        std::cout << "dsv4_fp4_sim_max_abs=" << max_abs << "\n";
        if (max_abs != 0.0) {
            throw std::runtime_error(
                "DeepSeek V4 FP4 activation simulation check failed");
        }
    }

    {
        constexpr int M = 3, K = 768, H = 64, ID = 128;
        auto q = torch::randn({B, M, H, ID}, cuda.dtype(torch::kFloat16));
        auto k = torch::randn({B, K, ID}, cuda.dtype(torch::kFloat16));
        auto weights = torch::randn({B, M, H}, cuda.dtype(torch::kFloat16));
        auto test = dsv4_indexer_scores_cuda(q, k, weights, 4096, 4);
        auto dot = torch::einsum(
            "bmhd,bkd->bmhk",
            {q.to(torch::kFloat32), k.to(torch::kFloat32)});
        auto ref = (torch::relu(dot) *
            weights.to(torch::kFloat32).unsqueeze(-1)).sum(2) /
            std::sqrt(static_cast<double>(H * ID));
        ref = ref.to(torch::kFloat16);
        const double rel =
            (test.to(torch::kFloat32) - ref.to(torch::kFloat32))
                .norm().item<double>() /
            std::max(ref.to(torch::kFloat32).norm().item<double>(), 1e-30);
        auto selected = dsv4_topk512_cuda(test);
        auto expected_topk = torch::topk(test, 512, -1, true, false);
        auto expected_scores = std::get<0>(expected_topk);
        auto expected = std::get<1>(expected_topk);
        auto selected_i64 = selected.to(torch::kInt64);
        auto selected_sorted = std::get<0>(
            torch::sort(selected_i64, -1));
        auto expected_sorted = std::get<0>(
            torch::sort(expected.to(torch::kInt64), -1));
        const bool topk_id_equal = selected_sorted.equal(expected_sorted);
        const bool topk_ids_valid =
            (selected_i64 >= 0).all().item<bool>() &&
            (selected_i64 < test.size(-1)).all().item<bool>();
        const bool topk_ids_unique =
            selected_sorted.slice(-1, 1, selected_sorted.size(-1))
                .ne(selected_sorted.slice(-1, 0, selected_sorted.size(-1) - 1))
                .all().item<bool>();
        bool topk_scores_equal = false;
        if (topk_ids_valid) {
            auto selected_scores = test.gather(-1, selected_i64);
            auto selected_score_sorted = std::get<0>(
                torch::sort(selected_scores, -1));
            auto expected_score_sorted = std::get<0>(
                torch::sort(expected_scores, -1));
            topk_scores_equal =
                selected_score_sorted.equal(expected_score_sorted);
        }
        std::cout << "dsv4_indexer_rel=" << rel
                  << " topk_id_set_equal=" << (topk_id_equal ? 1 : 0)
                  << " topk_score_multiset_equal="
                  << (topk_scores_equal ? 1 : 0)
                  << " topk_ids_valid=" << (topk_ids_valid ? 1 : 0)
                  << " topk_ids_unique=" << (topk_ids_unique ? 1 : 0)
                  << "\n";
        if (rel > 0.004 || !topk_scores_equal ||
            !topk_ids_valid || !topk_ids_unique) {
            throw std::runtime_error(
                "DeepSeek V4 indexer/top-k numerical check failed");
        }
    }

    constexpr int H = 64;
    constexpr int M = 3;
    constexpr int HISTORY = 127;
    constexpr int POOL = 800;
    constexpr int TOPK = 512;
    constexpr int WINDOW = 128;
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    auto q = torch::randn({B, H, M, D}, cuda.dtype(torch::kFloat32));
    auto raw = torch::randn({B, HISTORY + M, D}, cuda.dtype(torch::kFloat16));
    auto pooled = torch::randn({B, POOL, D}, cuda.dtype(torch::kFloat16));
    auto kv = torch::cat({raw, pooled}, 1).contiguous();
    std::vector<torch::Tensor> topk_rows;
    for (int row = 0; row < M; ++row) {
        topk_rows.push_back(
            torch::randperm(POOL, cuda.dtype(torch::kInt64))
                .narrow(0, 0, TOPK).to(torch::kInt32));
    }
    auto topk = torch::stack(topk_rows, 0).unsqueeze(0).contiguous();
    auto plan = dsv4_build_prefill_plan_cuda(
        topk, 4096, HISTORY, POOL, 4, WINDOW);
    {
        auto seq_len = torch::tensor({4097}, cuda.dtype(torch::kInt64));
        auto decode_plan = dsv4_build_decode_plan_cuda(
            topk.narrow(1, 0, 1).contiguous(),
            seq_len, POOL, 4, WINDOW);
        auto expected_local =
            torch::arange(4097 - WINDOW, 4097, cuda.dtype(torch::kInt64))
                .remainder(WINDOW);
        const bool local_equal = decode_plan[0]
            .index({0, 0, Slice(0, WINDOW)})
            .to(torch::kInt64).equal(expected_local);
        const bool pooled_equal = decode_plan[0]
            .index({0, 0, Slice(WINDOW, WINDOW + TOPK)})
            .equal(topk.index({0, 0}) + WINDOW);
        const bool mask_clear =
            (decode_plan[1] == 0).all().item<bool>();
        std::cout << "dsv4_decode_plan="
                  << (local_equal && pooled_equal && mask_clear ? 1 : 0)
                  << "\n";
        if (!local_equal || !pooled_equal || !mask_clear) {
            throw std::runtime_error(
                "DeepSeek V4 decode cache plan check failed");
        }
    }
    auto sinks = torch::randn({H}, cuda.dtype(torch::kFloat32));
    auto meta = torch::empty(
        {8 * 1024 * 1024}, cuda.dtype(torch::kFloat32));
    auto run_sparse = [&]() {
        return attention_dsv4_sparse_cuda(
            q, kv, plan[0], plan[1], sinks, meta, scale);
    };
    auto test = run_sparse();
    std::vector<torch::Tensor> ref_rows;
    auto q_ref = q.to(torch::kFloat16).to(torch::kFloat32);
    for (int row = 0; row < M; ++row) {
        auto idx = plan[0].index({0, row}).to(torch::kInt64);
        auto selected = kv.index_select(1, idx).index({0})
            .to(torch::kFloat32);
        auto score = torch::matmul(
            q_ref.index({0, Slice(), row}),
            selected.transpose(0, 1)) * scale;
        score = score + plan[1].index({0, row})
            .to(torch::kFloat32).unsqueeze(0);
        auto logits = torch::cat({score, sinks.unsqueeze(1)}, 1);
        auto probabilities = torch::softmax(logits, -1)
            .index({Slice(), Slice(0, score.size(1))});
        ref_rows.push_back(torch::matmul(probabilities, selected));
    }
    auto ref = torch::stack(ref_rows, 0).unsqueeze(0);
    torch::cuda::synchronize();
    const double rel = (test - ref).norm().item<double>() /
        std::max(ref.norm().item<double>(), 1e-30);
    const double max_abs = (test - ref).abs().max().item<double>();
    std::cout << "dsv4_sparse_attention_rel=" << rel
              << " max_abs=" << max_abs << "\n";
    if (!torch::isfinite(test).all().item<bool>() || rel > 0.02) {
        throw std::runtime_error(
            "DeepSeek V4 sparse attention numerical check failed");
    }

    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        for (int i = 0; i < 5; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    std::cout << "dsv4_sparse_attention_m=" << M
              << " selected=" << plan[0].size(2)
              << " cuda_ms=" << time_ms(run_sparse) << "\n";
    return 0;
}

int main(int argc, char ** argv) {
    struct TensorParallelCollectiveCleanup {
        ~TensorParallelCollectiveCleanup() {
            g_tensor_parallel_collectives.reset();
        }
    } tensor_parallel_collective_cleanup;
    try {
        std::string mfq_path, config_path, ids_arg, ids_file;
        std::string check_linear, check_linear_gate, kl_base;
        std::string kl_save_logits_f16;
        std::string check_tp_linear;
        std::string check_tp_moe;
        std::string check_tp_axis_arg = "output";
        std::string check_linear_group, check_q8_embedding;
        std::string check_gdn_input, check_gdn_output, check_gdn_state;
        std::string check_linear_conv_input, check_linear_conv_output;
        std::string check_nintm_tensor, check_dsv4_output_a;
        std::string kl_chunks_sequence_arg, kl_mmq_sequence_arg;
        std::string kl_evaluator_arg = "optimized";
        std::string kl_mmq_arg = "default";
        std::string nint6_mmq_arg = "fp16";
        std::string prefill_sweep_arg, check_moe_tokens = "1,2,4,8,16,32,64,128,256";
        std::string block_trace_reference, block_trace_output;
        std::string server_host = "127.0.0.1";
        std::string tokenizer_model, server_model_name = "mfq-model", server_api_key;
        std::string check_tokenizer_text =
            "MFQ tokenizer check: hello, world! <think>";
        std::string server_web_root;
        std::string cpu_offload_layers_arg;
        std::string moe_cache_profile_path;
        std::string tensor_parallel_arg;
        std::string tensor_split_arg;
        std::string layer_parallel_arg;
        std::string layer_split_arg;
        double moe_gpu_cache_gb = 0.0;
        int gen = 16;
        int server_port = 8080;
        int64_t context_size = 0;
        int kl_chunks = -1;
        int kl_score_count = -1;
        int64_t kl_n_batch = 0;
        KlReferenceContract kl_reference_contract;
        int kl_stream_layers = 0;
        int kl_stream_batch = 4;
        int64_t block_trace_start = 0;
        int64_t block_trace_count = 0;
        int prefill_repeat = 0;
        int prefill_sweep_reps = 5;
        int check_linear_m = 1;
        int check_linear_reps = 200;
        int check_tp_m = 1;
        int check_tp_moe_tokens = 1;
        int check_tp_moe_routes = 2;
        int check_gdn_tokens = 512;
        int check_gdn_q_heads = 16;
        int check_gdn_v_heads = 32;
        int check_gdn_head_dim = 128;
        int check_linear_conv_tokens = 512;
        int check_linear_conv_q_heads = 16;
        int check_linear_conv_v_heads = 32;
        int check_linear_conv_key_dim = 128;
        int check_linear_conv_value_dim = 128;
        int check_linear_conv_kernel = 4;
        int check_gemma_geglu_layer = -1;
        int check_gemma_geglu_reps = 100;
        int check_moe_layer = -1;
        int check_moe_reps = 100;
        int check_nintm_tokens = 1;
        int check_nintm_routes = 2;
        int check_nintm_reps = 100;
        int check_nintm_split_width = 0;
        int check_dsv4_output_a_batch = 1;
        int check_dsv4_output_a_reps = 200;
        int check_attention_decode = 0;
        int check_attention_reps = 200;
        int check_attention_head_dim = 256;
        int check_attention_window = 4096;
        int compare_llama_decode_steps = 1;
        int compare_llama_decode_planned_len = 0;
        bool profile = false;
        bool compare_llama_flash = false;
        bool compare_decode_splitk = false;
        bool compare_llama_decode = false;
        bool compare_nvq_vec4 = false;
        bool check_gemma4_swa = false;
        bool check_glm_dsa = false;
        bool check_dsv4_attention = false;
        bool check_dsv4_hc = false;
        bool compare_dsv4_hc_ops = false;
        bool compare_dsv4_hc_model = false;
        bool check_attention_swa_decode = false;
        bool check_nintm_routed_input = false;
        bool tensor_parallel_test_duplicates = false;
        bool server_mode = false;
        bool check_runtime_assets = false;
        bool check_mfq_container = false;
        bool kl_allow_overlays = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--mfq" && i + 1 < argc) mfq_path = argv[++i];
            else if (a == "--config" && i + 1 < argc) config_path = argv[++i];
            else if (a == "--ids" && i + 1 < argc) ids_arg = argv[++i];
            else if (a == "--ids-file" && i + 1 < argc) ids_file = argv[++i];
            else if (a == "--check-linear" && i + 1 < argc) check_linear = argv[++i];
            else if (a == "--check-tp-linear" && i + 1 < argc) {
                check_tp_linear = argv[++i];
            }
            else if (a == "--check-tp-axis" && i + 1 < argc) {
                check_tp_axis_arg = argv[++i];
            }
            else if (a == "--check-tp-m" && i + 1 < argc) {
                check_tp_m = std::stoi(argv[++i]);
            }
            else if (a == "--check-tp-moe" && i + 1 < argc) {
                check_tp_moe = argv[++i];
            }
            else if (a == "--check-tp-moe-tokens" && i + 1 < argc) {
                check_tp_moe_tokens = std::stoi(argv[++i]);
            }
            else if (a == "--check-tp-moe-routes" && i + 1 < argc) {
                check_tp_moe_routes = std::stoi(argv[++i]);
            }
            else if (a == "--check-linear-group" && i + 1 < argc) {
                check_linear_group = argv[++i];
            }
            else if (a == "--check-q8-embedding" && i + 1 < argc) {
                check_q8_embedding = argv[++i];
            }
            else if (a == "--check-gdn-input" && i + 1 < argc) check_gdn_input = argv[++i];
            else if (a == "--check-gdn-output" && i + 1 < argc) check_gdn_output = argv[++i];
            else if (a == "--check-gdn-state" && i + 1 < argc) check_gdn_state = argv[++i];
            else if (a == "--check-gdn-tokens" && i + 1 < argc) check_gdn_tokens = std::stoi(argv[++i]);
            else if (a == "--check-gdn-q-heads" && i + 1 < argc) check_gdn_q_heads = std::stoi(argv[++i]);
            else if (a == "--check-gdn-v-heads" && i + 1 < argc) check_gdn_v_heads = std::stoi(argv[++i]);
            else if (a == "--check-gdn-head-dim" && i + 1 < argc) check_gdn_head_dim = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-input" && i + 1 < argc) check_linear_conv_input = argv[++i];
            else if (a == "--check-linear-conv-output" && i + 1 < argc) check_linear_conv_output = argv[++i];
            else if (a == "--check-linear-conv-tokens" && i + 1 < argc) check_linear_conv_tokens = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-q-heads" && i + 1 < argc) check_linear_conv_q_heads = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-v-heads" && i + 1 < argc) check_linear_conv_v_heads = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-key-dim" && i + 1 < argc) check_linear_conv_key_dim = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-value-dim" && i + 1 < argc) check_linear_conv_value_dim = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-kernel" && i + 1 < argc) check_linear_conv_kernel = std::stoi(argv[++i]);
            else if (a == "--check-linear-m" && i + 1 < argc) check_linear_m = std::stoi(argv[++i]);
            else if (a == "--check-linear-reps" && i + 1 < argc) {
                check_linear_reps = std::stoi(argv[++i]);
            }
            else if (a == "--check-linear-gate" && i + 1 < argc) check_linear_gate = argv[++i];
            else if (a == "--check-gemma-geglu-layer" && i + 1 < argc) check_gemma_geglu_layer = std::stoi(argv[++i]);
            else if (a == "--check-gemma-geglu-reps" && i + 1 < argc) check_gemma_geglu_reps = std::stoi(argv[++i]);
            else if (a == "--check-moe-layer" && i + 1 < argc) check_moe_layer = std::stoi(argv[++i]);
            else if (a == "--check-moe-tokens" && i + 1 < argc) check_moe_tokens = argv[++i];
            else if (a == "--check-moe-reps" && i + 1 < argc) check_moe_reps = std::stoi(argv[++i]);
            else if (a == "--check-nintm-tensor" && i + 1 < argc) check_nintm_tensor = argv[++i];
            else if (a == "--check-nintm-tokens" && i + 1 < argc) check_nintm_tokens = std::stoi(argv[++i]);
            else if (a == "--check-nintm-routes" && i + 1 < argc) check_nintm_routes = std::stoi(argv[++i]);
            else if (a == "--check-nintm-reps" && i + 1 < argc) check_nintm_reps = std::stoi(argv[++i]);
            else if (a == "--check-nintm-split-width" && i + 1 < argc) check_nintm_split_width = std::stoi(argv[++i]);
            else if (a == "--check-nintm-routed-input") check_nintm_routed_input = true;
            else if (a == "--check-dsv4-output-a" && i + 1 < argc) check_dsv4_output_a = argv[++i];
            else if (a == "--check-dsv4-output-a-batch" && i + 1 < argc) check_dsv4_output_a_batch = std::stoi(argv[++i]);
            else if (a == "--check-dsv4-output-a-reps" && i + 1 < argc) check_dsv4_output_a_reps = std::stoi(argv[++i]);
            else if (a == "--check-attention-decode" && i + 1 < argc) check_attention_decode = std::stoi(argv[++i]);
            else if (a == "--check-attention-reps" && i + 1 < argc) check_attention_reps = std::stoi(argv[++i]);
            else if (a == "--check-attention-head-dim" && i + 1 < argc) check_attention_head_dim = std::stoi(argv[++i]);
            else if (a == "--check-attention-window" && i + 1 < argc) check_attention_window = std::stoi(argv[++i]);
            else if (a == "--check-attention-swa-decode") check_attention_swa_decode = true;
            else if (a == "--check-gemma4-swa") check_gemma4_swa = true;
            else if (a == "--check-glm-dsa") check_glm_dsa = true;
            else if (a == "--check-dsv4-attention") check_dsv4_attention = true;
            else if (a == "--check-dsv4-hc") check_dsv4_hc = true;
            else if (a == "--compare-dsv4-hc-ops") compare_dsv4_hc_ops = true;
            else if (a == "--compare-dsv4-hc-model") compare_dsv4_hc_model = true;
            else if (a == "--kl-base" && i + 1 < argc) kl_base = argv[++i];
            else if (a == "--kl-save-logits-f16" && i + 1 < argc) {
                kl_save_logits_f16 = argv[++i];
            }
            else if (a == "--kl-chunks" && i + 1 < argc) kl_chunks = std::stoi(argv[++i]);
            else if (a == "--kl-score-count" && i + 1 < argc) {
                kl_score_count = std::stoi(argv[++i]);
            }
            else if (a == "--kl-n-batch" && i + 1 < argc) {
                kl_n_batch = std::stoll(argv[++i]);
            }
            else if (a == "--kl-reference-n-batch" && i + 1 < argc) {
                kl_reference_contract.n_batch = std::stoll(argv[++i]);
            }
            else if (a == "--kl-reference-n-ubatch" && i + 1 < argc) {
                kl_reference_contract.n_ubatch = std::stoll(argv[++i]);
            }
            else if (a == "--kl-allow-overlays") {
                kl_allow_overlays = true;
            }
            else if (a == "--kl-chunks-sequence" && i + 1 < argc) {
                kl_chunks_sequence_arg = argv[++i];
            }
            else if (a == "--kl-evaluator" && i + 1 < argc) {
                kl_evaluator_arg = argv[++i];
            }
            else if (a == "--kl-mmq" && i + 1 < argc) {
                kl_mmq_arg = argv[++i];
            }
            else if (a == "--nint6-mmq" && i + 1 < argc) {
                nint6_mmq_arg = argv[++i];
            }
            else if (a == "--kl-mmq-sequence" && i + 1 < argc) {
                kl_mmq_sequence_arg = argv[++i];
            }
            else if (a == "--kl-stream-layers" && i + 1 < argc) {
                kl_stream_layers = std::stoi(argv[++i]);
            }
            else if (a == "--kl-stream-batch" && i + 1 < argc) {
                kl_stream_batch = std::stoi(argv[++i]);
            }
            else if (a == "--compare-block-trace" && i + 1 < argc) block_trace_reference = argv[++i];
            else if (a == "--dump-block-trace" && i + 1 < argc) block_trace_output = argv[++i];
            else if (a == "--dump-block-trace-start" && i + 1 < argc) {
                block_trace_start = std::stoll(argv[++i]);
            }
            else if (a == "--dump-block-trace-count" && i + 1 < argc) {
                block_trace_count = std::stoll(argv[++i]);
            }
            else if (a == "--prefill-repeat" && i + 1 < argc) prefill_repeat = std::stoi(argv[++i]);
            else if (a == "--prefill-sweep" && i + 1 < argc) prefill_sweep_arg = argv[++i];
            else if (a == "--prefill-sweep-reps" && i + 1 < argc) prefill_sweep_reps = std::stoi(argv[++i]);
            else if (a == "--gen" && i + 1 < argc) gen = std::stoi(argv[++i]);
            else if (a == "--server") server_mode = true;
            else if (a == "--host" && i + 1 < argc) server_host = argv[++i];
            else if (a == "--port" && i + 1 < argc) server_port = std::stoi(argv[++i]);
            else if (a == "--ctx-size" && i + 1 < argc) context_size = std::stoll(argv[++i]);
            else if (a == "--cpu-offload-layers" && i + 1 < argc) {
                cpu_offload_layers_arg = argv[++i];
            }
            else if (a == "--moe-gpu-cache-gb" && i + 1 < argc) {
                moe_gpu_cache_gb = std::stod(argv[++i]);
            }
            else if (a == "--moe-cache-profile" && i + 1 < argc) {
                moe_cache_profile_path = argv[++i];
            }
            else if (a == "--tensor-parallel" && i + 1 < argc) {
                tensor_parallel_arg = argv[++i];
            }
            else if (a == "--tensor-split" && i + 1 < argc) {
                tensor_split_arg = argv[++i];
            }
            else if (a == "--layer-parallel" && i + 1 < argc) {
                layer_parallel_arg = argv[++i];
            }
            else if (a == "--layer-split" && i + 1 < argc) {
                layer_split_arg = argv[++i];
            }
            else if (a == "--tensor-parallel-test-duplicates") {
                tensor_parallel_test_duplicates = true;
            }
            else if (a == "--tokenizer-model" && i + 1 < argc) tokenizer_model = argv[++i];
            else if (a == "--check-runtime-assets") check_runtime_assets = true;
            else if (a == "--check-mfq-container") check_mfq_container = true;
            else if (a == "--check-tokenizer-text" && i + 1 < argc) {
                check_tokenizer_text = argv[++i];
            }
            else if (a == "--model-name" && i + 1 < argc) server_model_name = argv[++i];
            else if (a == "--api-key" && i + 1 < argc) server_api_key = argv[++i];
            else if (a == "--web-root" && i + 1 < argc) server_web_root = argv[++i];
            else if (a == "--profile") profile = true;
            else if (a == "--compare-llama-flash") compare_llama_flash = true;
            else if (a == "--compare-decode-splitk") compare_decode_splitk = true;
            else if (a == "--compare-llama-decode") compare_llama_decode = true;
            else if (a == "--compare-llama-decode-steps" && i + 1 < argc) compare_llama_decode_steps = std::stoi(argv[++i]);
            else if (a == "--compare-llama-decode-planned-len" && i + 1 < argc) compare_llama_decode_planned_len = std::stoi(argv[++i]);
            else if (a == "--compare-nvq-vec4" || a == "--compare-niq-vec4") compare_nvq_vec4 = true;
            else {
                std::cerr << "usage: mfq-decode --mfq model.mfq [--config config.json] "
                             "(--ids 1,2,3 --gen 128 | --server "
                             "[--host 127.0.0.1 --port 8080 --ctx-size 32768 --model-name name "
                             "--tensor-parallel 0,1 --tensor-split 1,1 "
                             "--layer-parallel 0,1 --layer-split 1,1 "
                             "--cpu-offload-layers 0-7,12 --moe-gpu-cache-gb 8 "
                             "--moe-cache-profile profile.json "
                             "--api-key key --web-root path] | --kl-base reference.bin "
                             "[--kl-evaluator optimized|legacy --kl-chunks -1 "
                             "--kl-score-count N --kl-n-batch N "
                             "--kl-reference-n-batch N "
                             "--kl-reference-n-ubatch N --kl-mmq default|fp16|nint8_1] "
                             "[--kl-save-logits-f16 PATH] "
                             "[--nint6-mmq fp16|int8])\n";
                return 2;
            }
        }
        g_nint6_mmq_mode =
            parse_nint6_mmq_mode(nint6_mmq_arg);
        configure_tensor_parallel(
            tensor_parallel_arg,
            tensor_split_arg,
            tensor_parallel_test_duplicates);
        configure_layer_placement(
            layer_parallel_arg, layer_split_arg);
        if (moe_gpu_cache_gb < 0.0 ||
                !std::isfinite(moe_gpu_cache_gb)) {
            throw std::runtime_error(
                "--moe-gpu-cache-gb must be finite and non-negative");
        }
        if (moe_gpu_cache_gb > 0.0 &&
                !cpu_offload_layers_arg.empty()) {
            throw std::runtime_error(
                "--moe-gpu-cache-gb cannot be combined with "
                "--cpu-offload-layers");
        }
        if (moe_gpu_cache_gb > 0.0 &&
                g_tensor_parallel.enabled()) {
            throw std::runtime_error(
                "--moe-gpu-cache-gb cannot be combined with "
                "--tensor-parallel");
        }
        if (!moe_cache_profile_path.empty() &&
                moe_gpu_cache_gb <= 0.0) {
            throw std::runtime_error(
                "--moe-cache-profile requires --moe-gpu-cache-gb");
        }
        if (moe_gpu_cache_gb > 0.0) {
            constexpr double gib =
                1024.0 * 1024.0 * 1024.0;
            const double bytes = moe_gpu_cache_gb * gib;
            if (bytes < 1.0 ||
                    bytes >
                        static_cast<double>(
                            std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error(
                    "--moe-gpu-cache-gb is outside the supported range");
            }
            g_moe_expert_cache =
                std::make_shared<MoeExpertCache>(
                    static_cast<int64_t>(bytes));
            if (!moe_cache_profile_path.empty()) {
                g_moe_expert_cache->set_profile(
                    mfq::load_moe_cache_profile(
                        moe_cache_profile_path));
            }
            g_mfq_drop_file_cache = true;
        }
        if (!check_linear.empty()) {
            if (mfq_path.empty()) throw std::runtime_error("--check-linear requires --mfq");
            int gate_mode = 0;
            if (check_linear_gate == "sigmoid") gate_mode = 1;
            else if (check_linear_gate == "silu") gate_mode = 2;
            else if (!check_linear_gate.empty()) {
                throw std::runtime_error("--check-linear-gate must be sigmoid or silu");
            }
            return run_linear_check(
                mfq_path, check_linear, check_linear_m, gate_mode,
                check_linear_reps);
        }
        if (!check_tp_linear.empty()) {
            if (mfq_path.empty()) {
                throw std::runtime_error(
                    "--check-tp-linear requires --mfq");
            }
            const TensorParallelAxis axis =
                check_tp_axis_arg == "output"
                ? TensorParallelAxis::Output
                : check_tp_axis_arg == "input"
                ? TensorParallelAxis::Input
                : throw std::runtime_error(
                    "--check-tp-axis must be output or input");
            return run_tensor_parallel_linear_check(
                mfq_path, check_tp_linear,
                axis, check_tp_m);
        }
        if (!check_tp_moe.empty()) {
            if (mfq_path.empty()) {
                throw std::runtime_error(
                    "--check-tp-moe requires --mfq");
            }
            return run_tensor_parallel_moe_check(
                mfq_path, check_tp_moe,
                check_tp_moe_tokens,
                check_tp_moe_routes);
        }
        if (!check_linear_group.empty()) {
            if (mfq_path.empty()) {
                throw std::runtime_error("--check-linear-group requires --mfq");
            }
            return run_linear_group_check(
                mfq_path, check_linear_group, check_linear_m);
        }
        if (!check_q8_embedding.empty()) {
            if (mfq_path.empty()) {
                throw std::runtime_error("--check-q8-embedding requires --mfq");
            }
            return run_q8_embedding_check(mfq_path, check_q8_embedding);
        }
        if (!check_gdn_input.empty()) {
            if (check_gdn_output.empty() || check_gdn_state.empty()) {
                throw std::runtime_error(
                    "--check-gdn-input requires --check-gdn-output and --check-gdn-state");
            }
            return run_gdn_operator_check(
                check_gdn_input, check_gdn_output, check_gdn_state,
                check_gdn_tokens, check_gdn_q_heads,
                check_gdn_v_heads, check_gdn_head_dim);
        }
        if (!check_linear_conv_input.empty()) {
            if (check_linear_conv_output.empty()) {
                throw std::runtime_error(
                    "--check-linear-conv-input requires --check-linear-conv-output");
            }
            return run_linear_conv_operator_check(
                check_linear_conv_input, check_linear_conv_output,
                check_linear_conv_tokens, check_linear_conv_q_heads,
                check_linear_conv_v_heads, check_linear_conv_key_dim,
                check_linear_conv_value_dim, check_linear_conv_kernel,
                1.0e-6);
        }
        if (check_gemma_geglu_layer >= 0) {
            if (mfq_path.empty()) {
                throw std::runtime_error("--check-gemma-geglu-layer requires --mfq");
            }
            return run_gemma_geglu_check(
                mfq_path, check_gemma_geglu_layer, check_gemma_geglu_reps);
        }
        if (check_mfq_container) {
            if (mfq_path.empty()) {
                throw std::runtime_error(
                    "--check-mfq-container requires --mfq");
            }
            MfqFile mfq(mfq_path);
            std::cout << "mfq_container_check=ok"
                      << " shards=" << mfq.source_paths.size()
                      << " records=" << mfq.records.size()
                      << "\n";
            return 0;
        }
        if (check_runtime_assets) {
            if (mfq_path.empty()) {
                throw std::runtime_error(
                    "--check-runtime-assets requires --mfq");
            }
            MfqFile mfq(mfq_path);
            const Config config = load_config(mfq, config_path);
            if (!mfq.has_record(MFQ_TOKENIZER_GGUF_ASSET)) {
                throw std::runtime_error(
                    "MFQ has no embedded tokenizer GGUF");
            }
            const auto tokenizer_blob =
                mfq.read_asset(MFQ_TOKENIZER_GGUF_ASSET);
            const auto embedded =
                probe_mfq_tokenizer(tokenizer_blob, check_tokenizer_text);
            if (embedded.vocab_size != config.vocab_size) {
                throw std::runtime_error(
                    "embedded tokenizer/model vocabulary mismatch: tokenizer=" +
                    std::to_string(embedded.vocab_size) + " model=" +
                    std::to_string(config.vocab_size));
            }
            std::cout << "runtime_assets_check=ok"
                      << " config=embedded"
                      << " tokenizer=embedded"
                      << " vocab_size=" << embedded.vocab_size
                      << " chat_template="
                      << (!embedded.chat_template.empty() ? 1 : 0)
                      << " bos=" << embedded.bos_token
                      << " eos=" << embedded.eos_token
                      << " eot=" << embedded.eot_token
                      << " pad=" << embedded.pad_token
                      << " token_count=" << embedded.tokens.size()
                      << "\n";
            if (!tokenizer_model.empty()) {
                const auto external =
                    probe_mfq_tokenizer(tokenizer_model, check_tokenizer_text);
                if (external.vocab_size != embedded.vocab_size ||
                    external.bos_token != embedded.bos_token ||
                    external.eos_token != embedded.eos_token ||
                    external.eot_token != embedded.eot_token ||
                    external.pad_token != embedded.pad_token ||
                    external.chat_template != embedded.chat_template ||
                    external.tokens != embedded.tokens) {
                    throw std::runtime_error(
                        "embedded tokenizer differs from external GGUF");
                }
                std::cout << "runtime_assets_external_match=ok\n";
            }
            return 0;
        }
        if (check_moe_layer >= 0) {
            if (mfq_path.empty()) {
                throw std::runtime_error("--check-moe-layer requires --mfq");
            }
            return run_moe_check(
                mfq_path, config_path, check_moe_layer, parse_ids(check_moe_tokens), check_moe_reps);
        }
        if (!check_nintm_tensor.empty()) {
            if (mfq_path.empty()) {
                throw std::runtime_error("--check-nintm-tensor requires --mfq");
            }
            KlMmqScope check_kl_mmq_scope(
                parse_kl_mmq_mode(kl_mmq_arg));
            const auto tensor_names =
                parse_tensor_names(check_nintm_tensor);
            if (g_moe_expert_cache &&
                    tensor_names.size() != 1) {
                throw std::runtime_error(
                    "cached --check-nintm-tensor accepts one tensor "
                    "per invocation");
            }
            for (const auto & tensor_name : tensor_names) {
                const int result = run_nintm_tensor_check(
                    mfq_path, tensor_name, check_nintm_tokens,
                    check_nintm_routes, check_nintm_reps,
                    check_nintm_split_width, check_nintm_routed_input);
                if (result != 0) return result;
            }
            return 0;
        }
        if (!check_dsv4_output_a.empty()) {
            if (mfq_path.empty()) {
                throw std::runtime_error("--check-dsv4-output-a requires --mfq");
            }
            return run_dsv4_output_a_check(
                mfq_path, check_dsv4_output_a,
                check_dsv4_output_a_batch, check_dsv4_output_a_reps);
        }
        if (check_attention_decode > 0) {
            return run_attention_decode_check(
                check_attention_decode, check_attention_reps,
                check_attention_head_dim, check_attention_swa_decode,
                check_attention_window);
        }
        if (check_gemma4_swa) {
            return run_gemma4_swa_check(check_attention_reps);
        }
        if (check_glm_dsa) {
            return run_glm_dsa_check(check_attention_reps);
        }
        if (check_dsv4_attention) {
            return run_dsv4_attention_check(check_attention_reps);
        }
        if (check_dsv4_hc) {
            return run_dsv4_hc_check(check_attention_reps);
        }
        if (mfq_path.empty() ||
            (!server_mode && ids_arg.empty() && ids_file.empty() &&
                kl_base.empty() && prefill_sweep_arg.empty())) {
            std::cerr << "usage: mfq-decode --mfq model.mfq [--config config.json] "
                         "(--ids 1,2,3 --gen 128 | --server "
                         "[--host 127.0.0.1 --port 8080 --ctx-size 32768 --model-name name "
                         "--api-key key --web-root path] | --kl-base reference.bin "
                         "[--kl-evaluator optimized|legacy --kl-chunks -1 "
                         "--kl-score-count N --kl-n-batch N "
                         "--kl-reference-n-batch N "
                         "--kl-reference-n-ubatch N --kl-mmq default|fp16|nint8_1] "
                         "[--nint6-mmq fp16|int8])\n";
            return 2;
        }
        if (context_size < 0) throw std::runtime_error("--ctx-size must be positive");
        if (server_mode && context_size == 0) context_size = 32768;
        if (!cpu_offload_layers_arg.empty()) {
            g_dsv4_cpu_offload_layers =
                parse_layer_ranges(cpu_offload_layers_arg);
            std::vector<int> ordered(
                g_dsv4_cpu_offload_layers.begin(),
                g_dsv4_cpu_offload_layers.end());
            std::sort(ordered.begin(), ordered.end());
            std::cerr << "cpu_offload_layers=";
            for (size_t index = 0; index < ordered.size(); ++index) {
                if (index) std::cerr << ',';
                std::cerr << ordered[index];
            }
            std::cerr << std::endl;
        }
        std::vector<int64_t> prefill_sweep_sizes;
        if (!prefill_sweep_arg.empty()) {
            prefill_sweep_sizes = parse_ids(prefill_sweep_arg);
            if (std::any_of(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end(),
                            [](int64_t value) { return value <= 0; })) {
                throw std::runtime_error("--prefill-sweep values must be positive");
            }
            if (context_size == 0) {
                context_size = *std::max_element(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end());
            }
        }
        if ((!block_trace_reference.empty() || !block_trace_output.empty()) &&
                context_size == 0) {
            context_size = (int64_t)(
                ids_file.empty() ? parse_ids(ids_arg) : load_ids_file(ids_file)).size();
        }
        std::vector<int64_t> kl_chunks_sequence;
        const KlEvaluator kl_evaluator =
            parse_kl_evaluator(kl_evaluator_arg);
        const KlMmqMode kl_mmq_mode =
            parse_kl_mmq_mode(kl_mmq_arg);
        if (kl_chunks == 0 || kl_chunks < -1) {
            throw std::runtime_error(
                "--kl-chunks must be positive or -1 for all chunks");
        }
        if (kl_score_count == 0 || kl_score_count < -1) {
            throw std::runtime_error(
                "--kl-score-count must be positive or -1 for the stored count");
        }
        if (kl_n_batch < 0) {
            throw std::runtime_error(
                "--kl-n-batch must be non-negative; 0 uses n_ctx");
        }
        if (kl_reference_contract.n_batch < 0 ||
                kl_reference_contract.n_ubatch < 0) {
            throw std::runtime_error(
                "KL reference n_batch and n_ubatch must be non-negative");
        }
        if ((kl_reference_contract.n_batch == 0) !=
                (kl_reference_contract.n_ubatch == 0)) {
            throw std::runtime_error(
                "--kl-reference-n-batch and --kl-reference-n-ubatch must "
                "be supplied together");
        }
        if (kl_reference_contract.n_ubatch >
                kl_reference_contract.n_batch) {
            throw std::runtime_error(
                "--kl-reference-n-ubatch must not exceed "
                "--kl-reference-n-batch");
        }
        const auto active_env = [](const char * name) {
            const char * value = std::getenv(name);
            return value != nullptr && value[0] != '\0';
        };
        const bool tensor_overlay_active =
            active_env("MFQ_TENSOR_OVERLAY");
        const bool expert_overlay_active =
            active_env("MFQ_EXPERT_OVERLAY");
        if (!kl_base.empty() && active_env("MFQ_KL_WINDOW_M")) {
            throw std::runtime_error(
                "MFQ_KL_WINDOW_M is disabled because it silently changes "
                "the metric; use --kl-score-count");
        }
        if (!kl_base.empty() &&
                (tensor_overlay_active || expert_overlay_active) &&
                !kl_allow_overlays) {
            throw std::runtime_error(
                "KLD with MFQ_TENSOR_OVERLAY or MFQ_EXPERT_OVERLAY requires "
                "explicit --kl-allow-overlays");
        }
        std::vector<KlMmqMode> kl_mmq_sequence;
        if (!kl_mmq_sequence_arg.empty()) {
            if (kl_mmq_arg != "default") {
                throw std::runtime_error(
                    "--kl-mmq and --kl-mmq-sequence are mutually exclusive");
            }
            kl_mmq_sequence =
                parse_kl_mmq_sequence(kl_mmq_sequence_arg);
        }
        if (kl_mmq_mode != KlMmqMode::Default &&
                (kl_base.empty() ||
                 kl_evaluator != KlEvaluator::Optimized)) {
            throw std::runtime_error(
                "--kl-mmq nint8_1/fp16 requires "
                "--kl-base and --kl-evaluator optimized");
        }
        if (!kl_mmq_sequence.empty() &&
                (kl_base.empty() ||
                 kl_evaluator != KlEvaluator::Optimized)) {
            throw std::runtime_error(
                "--kl-mmq-sequence requires "
                "--kl-base and --kl-evaluator optimized");
        }
        if (kl_n_batch != 0 &&
                (kl_base.empty() ||
                 kl_evaluator != KlEvaluator::Optimized ||
                 kl_stream_layers > 0)) {
            throw std::runtime_error(
                "--kl-n-batch requires non-streamed --kl-base with "
                "--kl-evaluator optimized");
        }
        std::unique_ptr<KlMmqScope> kl_mmq_scope;
        const KlMmqMode load_mmq_mode =
            kl_mmq_sequence.empty()
            ? kl_mmq_mode : kl_mmq_sequence.front();
        if (load_mmq_mode != KlMmqMode::Default) {
            kl_mmq_scope =
                std::make_unique<KlMmqScope>(load_mmq_mode);
        }
        if (!kl_chunks_sequence_arg.empty()) {
            kl_chunks_sequence = parse_ids(kl_chunks_sequence_arg);
            if (kl_base.empty() ||
                std::any_of(
                    kl_chunks_sequence.begin(), kl_chunks_sequence.end(),
                    [](int64_t value) { return value <= 0; })) {
                throw std::runtime_error(
                    "--kl-chunks-sequence requires --kl-base and "
                    "positive comma-separated chunk counts");
            }
        }
        g_profiler.enabled = false;
        torch::NoGradGuard no_grad;
        if (!kl_base.empty()) std::cout << std::unitbuf;
        if (!kl_base.empty()) {
            std::cout << "cpp_kl_contract"
                      << " evaluator=" << kl_evaluator_name(kl_evaluator)
                      << " mmq=" << kl_mmq_mode_name(kl_mmq_mode)
                      << " nint6_mmq="
                      << nint6_mmq_mode_name(g_nint6_mmq_mode)
                      << " chunk_limit=" << kl_chunks
                      << " score_count_override=" << kl_score_count
                      << " requested_n_batch=" << kl_n_batch
                      << " reference_n_batch="
                      << kl_reference_contract.n_batch
                      << " reference_n_ubatch="
                      << kl_reference_contract.n_ubatch
                      << " tensor_overlay="
                      << (tensor_overlay_active ? 1 : 0)
                      << " expert_overlay="
                      << (expert_overlay_active ? 1 : 0)
                      << " overlays_explicitly_allowed="
                      << (kl_allow_overlays ? 1 : 0)
                      << "\n";
        }
        if (!kl_base.empty() && kl_stream_layers > 0) {
            if (kl_evaluator != KlEvaluator::Legacy) {
                throw std::runtime_error(
                    "--kl-evaluator optimized is unavailable for streamed KL");
            }
            if (!kl_chunks_sequence.empty()) {
                throw std::runtime_error(
                    "--kl-chunks-sequence is unavailable for streamed KL");
            }
            if (g_moe_expert_cache) {
                throw std::runtime_error(
                    "--moe-gpu-cache-gb is unavailable for streamed KL");
            }
            return run_kl_eval_streamed(
                mfq_path, config_path, kl_base,
                kl_save_logits_f16, kl_chunks,
                kl_stream_layers, kl_stream_batch,
                kl_score_count, kl_reference_contract);
        }
        if (server_mode) {
            if (!config_path.empty()) {
                throw std::runtime_error(
                    "MFQ server does not accept an external model config");
            }
            MfqFile runtime_assets(mfq_path);
            if (!runtime_assets.has_record(MFQ_MODEL_CONFIG_ASSET) ||
                !runtime_assets.has_record(MFQ_TOKENIZER_GGUF_ASSET)) {
                throw std::runtime_error(
                    "MFQ server requires embedded model config, tokenizer, "
                    "and chat template");
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        Model model = load_model(mfq_path, config_path, context_size);
        torch::cuda::synchronize();
        auto t1 = std::chrono::steady_clock::now();
        report_cuda_memory("loaded");
        if (server_mode) {
            if (server_api_key.empty()) {
                const char * env_key = std::getenv("MFQ_API_KEY");
                if (env_key != nullptr) server_api_key = env_key;
            }
            if (server_web_root.empty()) {
                std::error_code error;
                auto candidate = std::filesystem::absolute(
                    std::filesystem::path(argv[0]), error).parent_path() / "web";
                if (!error && std::filesystem::is_regular_file(
                        candidate / "index.html", error)) {
                    server_web_root = candidate.string();
                } else {
                    error.clear();
                    candidate = std::filesystem::current_path(error) /
                        "cpp_runtime" / "web";
                    if (!error && std::filesystem::is_regular_file(
                            candidate / "index.html", error)) {
                        server_web_root = candidate.string();
                    }
                }
            }
            MfqServerConfig server_config;
            server_config.host = server_host;
            server_config.port = server_port;
            server_config.model_name = server_model_name;
            server_config.model_type = model.c.model_type;
            MfqFile runtime_assets(mfq_path);
            if (!runtime_assets.has_record(MFQ_TOKENIZER_GGUF_ASSET)) {
                throw std::runtime_error(
                    "MFQ server requires embedded tokenizer and chat template");
            }
            server_config.tokenizer_gguf =
                runtime_assets.read_asset(MFQ_TOKENIZER_GGUF_ASSET);
            server_config.api_key = server_api_key;
            server_config.web_root = server_web_root;
            server_config.max_context = model.c.max_position_embeddings;
            server_config.vocab_size = model.c.vocab_size;
            std::mutex model_mutex;
            ServerDecodeGraphCache decode_graph_cache(model.c.max_position_embeddings);
            const int status = run_mfq_server(
                server_config, [&](const std::vector<int64_t> & prompt,
                                   const MfqSamplingParams & sampling,
                                   const MfqTokenCallback & on_token,
                                   const MfqPrefillCallback & on_prefill,
                                   const MfqPromptCachePlan &) {
                return generate_server_tokens(
                    model, model_mutex, decode_graph_cache, prompt, sampling,
                    on_token, on_prefill);
            });
            if (g_moe_expert_cache) {
                g_moe_expert_cache->print_stats(std::cout);
            }
            return status;
        }
        if (!kl_base.empty()) {
            if (!kl_mmq_sequence.empty()) {
                if (!kl_chunks_sequence.empty()) {
                    throw std::runtime_error(
                        "--kl-mmq-sequence cannot be combined with "
                        "--kl-chunks-sequence");
                }
                for (size_t index = 0;
                     index < kl_mmq_sequence.size(); ++index) {
                    std::cout << "cpp_kl_mmq_sequence_begin index=" << index
                              << " mmq="
                              << kl_mmq_mode_name(kl_mmq_sequence[index])
                              << " chunks=" << kl_chunks << "\n";
                    KlMmqScope run_scope(kl_mmq_sequence[index]);
                    const int status = run_kl_eval_batched(
                        model, kl_base, kl_chunks,
                        kl_n_batch, kl_score_count,
                        kl_reference_contract);
                    if (status != 0) return status;
                    std::cout << "cpp_kl_mmq_sequence_end index=" << index
                              << " mmq="
                              << kl_mmq_mode_name(kl_mmq_sequence[index])
                              << " chunks=" << kl_chunks << "\n";
                }
                return 0;
            }
            if (kl_chunks_sequence.empty()) {
                const int status = run_selected_kl_eval(
                    model, kl_base, kl_chunks,
                    kl_evaluator,
                    kl_n_batch, kl_score_count,
                    kl_reference_contract);
                if (g_moe_expert_cache) {
                    g_moe_expert_cache->print_stats(std::cout);
                }
                return status;
            }
            for (size_t index = 0; index < kl_chunks_sequence.size(); ++index) {
                const int chunks =
                    static_cast<int>(kl_chunks_sequence[index]);
                std::cout << "cpp_kl_sequence_begin index=" << index
                          << " chunks=" << chunks << "\n";
                const int status = run_selected_kl_eval(
                    model, kl_base, chunks,
                    kl_evaluator,
                    kl_n_batch, kl_score_count,
                    kl_reference_contract);
                if (status != 0) return status;
                std::cout << "cpp_kl_sequence_end index=" << index
                          << " chunks=" << chunks << "\n";
            }
            if (g_moe_expert_cache) {
                g_moe_expert_cache->print_stats(std::cout);
            }
            return 0;
        }
        if (!prefill_sweep_sizes.empty()) {
            const int status = run_prefill_sweep(
                model, prefill_sweep_sizes, prefill_sweep_reps);
            if (g_moe_expert_cache) {
                g_moe_expert_cache->print_stats(std::cout);
            }
            return status;
        }
        auto ids_vec = ids_file.empty() ? parse_ids(ids_arg) : load_ids_file(ids_file);
        auto ids = torch::tensor(ids_vec, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)).unsqueeze(0);
        if (compare_dsv4_hc_ops) {
            g_dsv4_compare_hc_ops = true;
            g_dsv4_fused_hc = false;
            model.reset(1);
            (void)model.forward(ids);
            torch::cuda::synchronize();
            return 0;
        }
        if (compare_dsv4_hc_model) {
            return run_dsv4_hc_model_compare(model, ids);
        }
        if (!block_trace_reference.empty()) {
            return run_block_trace_compare(
                model, block_trace_reference, config_path, context_size, ids);
        }
        if (!block_trace_output.empty()) {
            return run_block_trace_dump(
                model, block_trace_output, ids,
                block_trace_start, block_trace_count);
        }
        if (profile) {
            model.reset(1);
            (void)model.next_token(ids);
            torch::cuda::synchronize();
            model.reset(1);
            g_profiler.reset();
            g_profiler.enabled = true;
        }
        if (compare_decode_splitk) {
            auto run = [&](const char * split) {
                mfq_set_env("MFQ_ATTENTION_DECODE_SPLITK", split);
                model.reset(1);
                (void)model.hidden_forward(ids);
                const int64_t decode_len = model.cache_pos + 1;
                auto seq_len = torch::tensor({decode_len}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
                auto decode_id = ids.index({Slice(), -1}).reshape({1, 1});
                auto hidden = model.hidden_forward(decode_id, c10::nullopt, seq_len);
                auto last = hidden.index({Slice(), -1, Slice()}).to(torch::kFloat16).contiguous();
                auto logits = model.lm_head.forward(last).to(torch::kFloat32);
                torch::cuda::synchronize();
                return logits;
            };
            auto ref = run("0");
            auto test = run("1");
            auto ref_logp = torch::log_softmax(ref, -1);
            auto test_logp = torch::log_softmax(test, -1);
            auto kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1);
            auto diff = (ref - test).abs();
            std::cout << "decode_splitk_compare_kl=" << kl.item<float>() << "\n";
            std::cout << "decode_splitk_compare_rel=" << ((test - ref).norm() / ref.norm()).item<float>() << "\n";
            std::cout << "decode_splitk_compare_mean_abs=" << diff.mean().item<float>() << "\n";
            std::cout << "decode_splitk_compare_max_abs=" << diff.max().item<float>() << "\n";
            std::cout << "decode_splitk_compare_same_top="
                      << (ref.argmax(-1).eq(test.argmax(-1)).item<bool>() ? 1 : 0) << "\n";
            return 0;
        }
        if (compare_llama_decode) {
            if (compare_llama_decode_steps < 1) {
                throw std::runtime_error("--compare-llama-decode-steps must be positive");
            }
            auto cuda_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
            std::vector<torch::Tensor> reference_logits;
            std::vector<int64_t> teacher_tokens;
            reference_logits.reserve(compare_llama_decode_steps);
            teacher_tokens.reserve(compare_llama_decode_steps);
            mfq_set_env("MFQ_LLAMA_FLASH256_DECODE", "0");
            model.reset(1);
            auto input = model.next_token(ids).reshape({1, 1});
            const int64_t initial_teacher_token = input.item<int64_t>();
            for (int step = 0; step < compare_llama_decode_steps; ++step) {
                const int64_t decode_len = model.cache_pos + 1;
                auto seq_len = torch::tensor({decode_len}, cuda_i64);
                auto hidden = model.hidden_forward(input, c10::nullopt, seq_len);
                auto last = hidden.index({Slice(), -1, Slice()}).to(torch::kFloat16).contiguous();
                auto logits = model.lm_head.forward(last).to(torch::kFloat32);
                reference_logits.push_back(logits.clone());
                const int64_t next = logits.argmax(-1).item<int64_t>();
                teacher_tokens.push_back(next);
                input = torch::tensor({next}, cuda_i64).reshape({1, 1});
            }
            mfq_set_env("MFQ_LLAMA_FLASH256_DECODE", "1");
            g_decode_graph_attention_kv_len = compare_llama_decode_planned_len > 0
                ? compare_llama_decode_planned_len
                : ids.size(1) + compare_llama_decode_steps;
            if (g_decode_graph_attention_kv_len > model.c.max_position_embeddings) {
                throw std::runtime_error("decode comparison planned length exceeds context capacity");
            }
            model.reset(1);
            (void)model.next_token(ids);
            input = torch::tensor({initial_teacher_token}, cuda_i64).reshape({1, 1});
            double kl_sum = 0.0;
            double kl_max = 0.0;
            double abs_sum = 0.0;
            double delta_sq_sum = 0.0;
            double reference_sq_sum = 0.0;
            double max_abs = 0.0;
            int same_top = 0;
            int first_top_difference = -1;
            int64_t values = 0;
            for (int step = 0; step < compare_llama_decode_steps; ++step) {
                const int64_t decode_len = model.cache_pos + 1;
                auto seq_len = torch::tensor({decode_len}, cuda_i64);
                auto hidden = model.hidden_forward(input, c10::nullopt, seq_len);
                auto last = hidden.index({Slice(), -1, Slice()}).to(torch::kFloat16).contiguous();
                auto test = model.lm_head.forward(last).to(torch::kFloat32);
                const auto & ref = reference_logits[(size_t)step];
                auto ref_logp = torch::log_softmax(ref, -1);
                auto test_logp = torch::log_softmax(test, -1);
                const double kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1).item<double>();
                auto delta = test - ref;
                kl_sum += kl;
                kl_max = std::max(kl_max, kl);
                abs_sum += delta.abs().sum().item<double>();
                delta_sq_sum += delta.square().sum().item<double>();
                reference_sq_sum += ref.square().sum().item<double>();
                max_abs = std::max(max_abs, delta.abs().max().item<double>());
                values += delta.numel();
                const bool top_equal = test.argmax(-1).eq(ref.argmax(-1)).item<bool>();
                same_top += top_equal ? 1 : 0;
                if (!top_equal && first_top_difference < 0) first_top_difference = step;
                input = torch::tensor({teacher_tokens[(size_t)step]}, cuda_i64).reshape({1, 1});
            }
            torch::cuda::synchronize();
            g_decode_graph_attention_kv_len = 0;
            std::cout << std::setprecision(10)
                      << "llama_decode_compare_steps=" << compare_llama_decode_steps << "\n"
                      << "llama_decode_compare_mean_kl=" << kl_sum / compare_llama_decode_steps << "\n"
                      << "llama_decode_compare_max_kl=" << kl_max << "\n"
                      << "llama_decode_compare_rel=" << std::sqrt(delta_sq_sum / reference_sq_sum) << "\n"
                      << "llama_decode_compare_mean_abs=" << abs_sum / values << "\n"
                      << "llama_decode_compare_max_abs=" << max_abs << "\n"
                      << "llama_decode_compare_same_top=" << same_top << "\n"
                      << "llama_decode_compare_first_top_difference=" << first_top_difference << "\n";
            return 0;
        }
        if (compare_nvq_vec4) {
            auto run = [&](const char * enabled) {
                mfq_set_env("MFQ_NVQ_SWIGLU_VEC4", enabled);
                model.reset(1);
                auto logits = model.last_logits(ids).to(torch::kFloat32);
                torch::cuda::synchronize();
                return logits;
            };
            auto ref = run("0");
            auto repeat = run("0");
            auto test = run("1");
            auto ref_logp = torch::log_softmax(ref, -1);
            auto repeat_logp = torch::log_softmax(repeat, -1);
            auto test_logp = torch::log_softmax(test, -1);
            auto repeat_kl = (ref_logp.exp() * (ref_logp - repeat_logp)).sum(-1);
            auto kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1);
            auto diff = (ref - test).abs();
            auto repeat_diff = (ref - repeat).abs();
            std::cout << "nvq_vec4_repeat_kl=" << repeat_kl.item<float>() << "\n";
            std::cout << "nvq_vec4_repeat_max_abs=" << repeat_diff.max().item<float>() << "\n";
            std::cout << "nvq_vec4_compare_kl=" << kl.item<float>() << "\n";
            std::cout << "nvq_vec4_compare_rel="
                      << ((test - ref).norm() / ref.norm()).item<float>() << "\n";
            std::cout << "nvq_vec4_compare_mean_abs=" << diff.mean().item<float>() << "\n";
            std::cout << "nvq_vec4_compare_max_abs=" << diff.max().item<float>() << "\n";
            std::cout << "nvq_vec4_compare_same_top="
                      << (ref.argmax(-1).eq(test.argmax(-1)).item<bool>() ? 1 : 0) << "\n";
            return 0;
        }
        if (prefill_repeat > 0) {
            const char * trace_env = std::getenv("MFQ_CHECK_PREFILL_REPEAT_TRACE");
            const bool trace_repeat = trace_env != nullptr && std::atoi(trace_env) != 0;
            std::vector<torch::Tensor> reference_trace;
            std::vector<std::pair<std::string, torch::Tensor>> reference_gemma_trace;
            for (int i = 0; i < prefill_repeat; ++i) {
                model.reset(1);
                auto run_t0 = std::chrono::steady_clock::now();
                std::vector<torch::Tensor> trace;
                std::vector<std::pair<std::string, torch::Tensor>> gemma_trace;
                torch::Tensor logits;
                if (trace_repeat) {
                    g_gemma_trace_layer = 0;
                    g_gemma_stage_trace = &gemma_trace;
                    auto hidden = model.hidden_forward(
                        ids, c10::nullopt, c10::nullopt, &trace);
                    g_gemma_stage_trace = nullptr;
                    g_gemma_trace_layer = -1;
                    auto last = hidden.index({Slice(), -1, Slice()})
                        .to(torch::kFloat16).contiguous();
                    logits = model.lm_head.forward(last);
                } else {
                    logits = model.last_logits(ids);
                }
                torch::cuda::synchronize();
                auto run_t1 = std::chrono::steady_clock::now();
                std::cout << "prefill_repeat=" << (i + 1)
                          << " sec=" << std::chrono::duration<double>(run_t1 - run_t0).count()
                          << " top=" << logits.argmax(-1).item<int64_t>() << "\n";
                if (trace_repeat) {
                    if (reference_trace.empty()) {
                        reference_trace = std::move(trace);
                        reference_gemma_trace = std::move(gemma_trace);
                    } else {
                        for (size_t stage = 0; stage < gemma_trace.size(); ++stage) {
                            auto diff = (gemma_trace[stage].second -
                                         reference_gemma_trace[stage].second).abs();
                            const float max_abs = diff.max().item<float>();
                            if (max_abs != 0.0f) {
                                std::cout << "prefill_repeat_first_gemma_stage="
                                          << gemma_trace[stage].first
                                          << " max_abs=" << max_abs
                                          << " mean_abs=" << diff.mean().item<float>() << "\n";
                                break;
                            }
                        }
                        for (size_t stage = 0; stage < trace.size(); ++stage) {
                            auto diff = (trace[stage] - reference_trace[stage]).abs();
                            const float max_abs = diff.max().item<float>();
                            if (max_abs != 0.0f) {
                                std::cout << "prefill_repeat_first_difference=" << stage
                                          << " layer=" << static_cast<int64_t>(stage) - 1
                                          << " max_abs=" << max_abs
                                          << " mean_abs=" << diff.mean().item<float>() << "\n";
                                break;
                            }
                        }
                    }
                }
            }
            return 0;
        }
        if (compare_llama_flash) {
            mfq_set_env("MFQ_LLAMA_FLASH256", "0");
            auto ref = model.last_logits(ids).to(torch::kFloat32);
            torch::cuda::synchronize();
            model.reset(1);
            mfq_set_env("MFQ_LLAMA_FLASH256", "1");
            auto test = model.last_logits(ids).to(torch::kFloat32);
            torch::cuda::synchronize();
            auto ref_logp = torch::log_softmax(ref, -1);
            auto test_logp = torch::log_softmax(test, -1);
            auto kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1);
            auto diff = (ref - test).abs();
            auto same_top = ref.argmax(-1).eq(test.argmax(-1));
            std::cout << "attention_compare_kl=" << kl.item<float>() << "\n";
            std::cout << "attention_compare_max_logit_abs=" << diff.max().item<float>() << "\n";
            std::cout << "attention_compare_mean_logit_abs=" << diff.mean().item<float>() << "\n";
            std::cout << "attention_compare_same_top=" << (same_top.item<bool>() ? 1 : 0) << "\n";
            return 0;
        }
        g_profiler.reset();
        auto next = model.next_token(ids);
        torch::cuda::synchronize();
        auto t2 = std::chrono::steady_clock::now();
        report_cuda_memory("prefill");
        const char * empty_cache_env = std::getenv("MFQ_EMPTY_CACHE_BEFORE_GRAPH");
        if (empty_cache_env != nullptr && std::atoi(empty_cache_env) != 0) {
            c10::cuda::CUDACachingAllocator::emptyCache();
            report_cuda_memory("prefill_empty_cache");
        }
        g_profiler.report("prefill");
        g_profiler.reset();
        auto generated_cuda = torch::empty({gen}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        MFQ_CUDA_CHECK(cudaMemcpyAsync(generated_cuda.data_ptr<int64_t>(), next.data_ptr<int64_t>(),
                                   sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
        const char* graph_env = std::getenv("MFQ_CUDA_GRAPH");
        const char * profile_graph_env = std::getenv("MFQ_PROFILE_CUDA_GRAPH");
        const bool profile_cuda_graph = profile && profile_graph_env != nullptr &&
            std::atoi(profile_graph_env) != 0;
        g_profiler.graph_events = profile_cuda_graph;
        bool use_cuda_graph =
            (graph_env == nullptr || graph_env[0] != '0') &&
            g_dsv4_cpu_offload_layers.empty() &&
            !g_moe_expert_cache &&
            !g_tensor_parallel.enabled() &&
            (!profile || profile_cuda_graph) && gen > 1;
        const char * cuda_profiler_env = std::getenv("MFQ_CUDA_PROFILER_RANGE");
        const bool cuda_profiler_range = cuda_profiler_env != nullptr &&
            std::atoi(cuda_profiler_env) != 0;
        auto decode_replay_t0 = t2;
        if (cuda_profiler_range) MFQ_CUDA_CHECK(cudaProfilerStart());
        if (use_cuda_graph) {
            auto static_input = torch::empty({1, 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            auto static_pos = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            auto static_len = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            auto static_step = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            auto graph_stream = at::cuda::getStreamFromPool(false);
            c10::cuda::CUDAStreamGuard graph_guard(graph_stream);
            cudaStream_t graph_raw_stream = graph_stream.stream();

            int64_t pos_h = model.cache_pos;
            int64_t len_h = pos_h + 1;
            int64_t step_h = 1;
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_input.data_ptr<int64_t>(), next.data_ptr<int64_t>(),
                                       sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_pos.data_ptr<int64_t>(), &pos_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_len.data_ptr<int64_t>(), &len_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_step.data_ptr<int64_t>(), &step_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));

            at::cuda::CUDAGraph graph;
            torch::Tensor static_next;
            const int64_t planned_len = model.cache_pos + gen;
            g_decode_graph_attention_kv_len = planned_len;
            g_decode_graph_attention_parts = planned_len >= 192 ? (planned_len + 127) / 128 : 1;
            g_decode_graph_attention_parts = std::min<int64_t>(
                g_decode_graph_attention_parts, FullBlock::kDecodeAttentionMaxParts);
            if (model.c.is_gemma4() || model.c.is_glm_dsa()) {
                // These graphs use M=1 expert and attention workspaces that differ
                // from prefill. Initialize their pointer tables before capture; the
                // captured pass overwrites the same KV position.
                (void)model.next_token_static(static_input, static_pos, static_len);
                MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));
            }
            graph.capture_begin();
            static_next = model.next_token_static(static_input, static_pos, static_len);
            decode_graph_commit_cuda(static_next, generated_cuda, static_step, static_input, static_pos, static_len);
            graph.capture_end();
            report_cuda_memory("graph_captured");
            g_decode_graph_attention_kv_len = 0;
            g_decode_graph_attention_parts = 0;

            decode_replay_t0 = std::chrono::steady_clock::now();
            for (int i = 1; i < gen; ++i) {
                graph.replay();
            }
            MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));
        } else {
            for (int i = 1; i < gen; ++i) {
                next = model.next_token(next.view({1, 1}));
                MFQ_CUDA_CHECK(cudaMemcpyAsync(generated_cuda.data_ptr<int64_t>() + i, next.data_ptr<int64_t>(),
                                           sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
            }
        }
        torch::cuda::synchronize();
        if (cuda_profiler_range) MFQ_CUDA_CHECK(cudaProfilerStop());
        auto t3 = std::chrono::steady_clock::now();
        g_profiler.report("decode");
        auto generated_tensor = generated_cuda.to(torch::kCPU).contiguous();
        auto generated_ptr = generated_tensor.data_ptr<int64_t>();
        double load_s = std::chrono::duration<double>(t1 - t0).count();
        double prefill_s = std::chrono::duration<double>(t2 - t1).count();
        double decode_s = std::chrono::duration<double>(t3 - t2).count();
        double decode_replay_s = std::chrono::duration<double>(t3 - decode_replay_t0).count();
        std::cout << "load_sec=" << load_s << "\n";
        std::cout << "prefill_sec=" << prefill_s << "\n";
        std::cout << "decode_tokens=" << std::max(0, gen - 1) << "\n";
        std::cout << "decode_setup_sec=" << (decode_s - decode_replay_s) << "\n";
        std::cout << "decode_replay_sec=" << decode_replay_s << "\n";
        std::cout << "decode_sec=" << decode_s << "\n";
        if (gen > 1) std::cout << "decode_tok_per_s=" << (double)(gen - 1) / decode_s << "\n";
        if (gen > 1) std::cout << "decode_steady_tok_per_s=" << (double)(gen - 1) / decode_replay_s << "\n";
        std::cout << "generated_ids=";
        for (int64_t i = 0; i < generated_tensor.numel(); ++i) {
            if (i) std::cout << ",";
            std::cout << generated_ptr[i];
        }
        std::cout << "\n";
        if (g_moe_expert_cache) {
            g_moe_expert_cache->print_stats(std::cout);
        }
        return 0;
    } catch (const c10::Error & e) {
        std::cerr << "c10_error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
