#pragma once

#include "runtime/batch_decode_graph.h"
#include "batch_selected_attention.h"

namespace mfq::cuda::qwen4_exp {

inline void Qsa::bind_decode_state(BatchDecodeInputs& inputs, int64_t field, int64_t row) const {
    inputs.state(row, field, graph_keys_, 2);
    inputs.state(row, field + 1, graph_values_, 2);
    inputs.state(row, field + 2, index_.storage(), 1);
    inputs.state(row, field + 3, graph_pooled_, 1);
    if (managed_memory() && memory_->hot.exclusive())
        memory_->hot.bind_decode_state(inputs, field + 4, row);
}

inline int64_t Qsa::decode_graph_capacity() const {
    return graph_keys_.defined() ? graph_keys_.size(2) : 0;
}

inline Tensor Qsa::forward_independent_graph(CudaExecutionContext& execution, const Tensor& hidden,
        const Tensor& positions, const BatchDecodeInputs& inputs, int64_t field, int64_t capacity,
        std::vector<Tensor> projections) {
    auto p = project(execution, hidden, positions, capacity > c_.budget, {}, {}, {}, std::move(projections));
    const auto transform = [&](const Tensor& pooled, const Tensor& block_positions) {
        if (rotary_->fused())
            return rotary_->forward_normalized(pooled.unsqueeze(2), w_.index_key_norm,
                                               block_positions, c_.eps).squeeze(2);
        auto normalized = rms_norm(pooled, w_.index_key_norm + 1, c_.eps);
        return rotary_->forward(normalized.unsqueeze(1), block_positions).squeeze(1);
    };
    auto output = batch_selected_attention(inputs, field, p.query, encode_key(p.key), encode_value(p.value),
        p.gate, p.raw, p.iq, positions, {c_.pool, c_.budget, capacity, quantization_}, transform);
    return output.to(hidden.scalar_type()).reshape({inputs.rows(), 1, c_.heads * c_.width});
}

inline Tensor Qsa::forward_indirect_graph(CudaExecutionContext& execution, const Tensor& hidden,
        const Tensor& positions, const Tensor& cache_positions, const BatchDecodeInputs& inputs,
        int64_t field, int64_t row, int64_t capacity, std::vector<Tensor> projections, bool output_projection) {
    auto p = project(execution, hidden, positions, capacity > c_.budget, cache_positions,
                     {}, {}, std::move(projections));
    const bool exclusive = managed_memory() && memory_->hot.exclusive();
    auto keys_to_append = encode_key(p.key).permute({0, 2, 1, 3});
    auto values_to_append = encode_value(p.value).permute({0, 2, 1, 3});
    if (exclusive) {
        MFQ_RUNTIME_CHECK(memory_->hot.append_decode(keys_to_append, values_to_append, cache_positions,
            {}, {}, false, &inputs, field + 4, row), "unsupported indirect KV append geometry");
    } else {
        batch_state_scatter(inputs, field, row, cache_positions, keys_to_append);
        batch_state_scatter(inputs, field + 1, row, cache_positions, values_to_append);
    }
    batch_state_scatter(inputs, field + 2, row, cache_positions,
                        p.raw.to(index_.storage().scalar_type()).reshape({1, 1, 1, c_.index_width}));
    Tensor selected;
    if (capacity <= c_.budget) {
        auto columns = tb::arange(capacity, cache_positions.options()).reshape({1, 1, capacity});
        selected = tb::where(columns <= cache_positions.reshape({1, 1, 1}), columns, tb::full_like(columns, -1));
    } else {
        const auto pools = (capacity + c_.pool - 1) / c_.pool;
        auto block = cache_positions / c_.pool;
        auto indices = (block * c_.pool + tb::arange(c_.pool, cache_positions.options())).clamp(0, c_.maximum - 1);
        auto raw = batch_state_gather(inputs, field + 2, row, indices, 1, c_.index_width,
                                      index_.storage().options()).reshape({1, c_.pool, c_.index_width});
        // Use the existing ordered pool reduction on the gathered local block.
        auto local_position = cache_positions.remainder(c_.pool);
        auto pooled = mfq_qwen4_exp::live_index_pool(raw, positions, local_position, c_.pool, c_.pool)[0];
        auto block_positions = positions - cache_positions + block * c_.pool;
        if (rotary_->fused()) {
            pooled = rotary_->forward_normalized(pooled.unsqueeze(2), w_.index_key_norm,
                                                 block_positions, c_.eps).squeeze(1);
        } else {
            pooled = rms_norm(pooled, w_.index_key_norm.to(tb::kFloat32) + 1, c_.eps);
            pooled = rotary_->forward(pooled.unsqueeze(1), block_positions).squeeze(1);
        }
        batch_state_scatter(inputs, field + 3, row, block, pooled.reshape({1, 1, 1, c_.index_width}));
        auto pool_indices = tb::arange(pools, cache_positions.options());
        auto history = batch_state_gather(inputs, field + 3, row, pool_indices, 1, c_.index_width,
                                          graph_pooled_.options()).reshape({1, pools, c_.index_width});
        auto scores = mfq_qwen4_exp::live_block_scores(p.iq, history, cache_positions, c_.pool);
        auto blocks = mfq_qwen4_exp::live_topk_blocks(scores, cache_positions, c_.pool,
                                                    std::min(c_.budget / c_.pool, pools));
        selected = mfq_qwen4_exp::qsa_selected_tokens(blocks, cache_positions, c_.pool, c_.budget).to(tb::kInt64);
    }
    auto flat = selected.reshape({-1}).contiguous();
    const auto hot_field = exclusive ? field + 4 : -1;
    auto keys = batch_state_gather(inputs, field, row, flat, c_.kv_heads, key_width(),
                                   graph_keys_.options(), hot_field);
    auto values = batch_state_gather(inputs, field + 1, row, flat, c_.kv_heads, value_width(),
                                     graph_values_.options(), hot_field, true);
    auto local = tb::arange(flat.numel(), cache_positions.options()).reshape(selected.sizes());
    local = tb::where(selected >= 0, local, tb::full_like(local, -1)).to(tb::kInt32).contiguous();
    Tensor attended;
    if (quantization_.enabled())
        attended = cuda_kv_attention(p.query, keys, values, quantization_, local, cache_positions);
    else
        attended = mfq_qwen4_exp::sparse_gqa_attention(p.query, keys, values, local);
    auto gated = (attended.to(tb::kFloat32) * tb::sigmoid(p.gate.to(tb::kFloat32))).to(hidden.scalar_type());
    auto output = gated.reshape({1, 1, c_.heads * c_.width});
    return output_projection ? w_.output(execution, output) : output;
}

} // namespace mfq::cuda::qwen4_exp
