#include "runtime/execution_options.h"
#include "batch_selected_attention.h"
#include "selected_attention.h"
#include "runtime/decode_window.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;

bool batch_selected_attention_enabled() {
    const auto* option = std::getenv("MFQ_QSA_INDEPENDENT_BATCH");
    return !option || option[0] != '0';
}

tb::Tensor batch_selected_attention(const BatchDecodeInputs& inputs, int64_t field,
        const tb::Tensor& query, const tb::Tensor& key, const tb::Tensor& value, const tb::Tensor& gate,
        const tb::Tensor& raw_index, const tb::Tensor& index_query, const tb::Tensor& positions,
        const BatchSelectedAttentionConfig& config, const BatchPoolTransform& transform) {
    MFQ_RUNTIME_CHECK(query.dim() == 4 && query.size(0) == inputs.rows() && query.size(2) == 1 &&
        config.pool > 0 && config.budget > 0 && config.budget % config.pool == 0 && config.capacity > 0,
        "invalid independent selected-attention geometry");
    if (runtime_options::qsa_independent_batch_trace())
        std::fprintf(stderr, "qsa_independent_batch rows=%lld\n", static_cast<long long>(inputs.rows()));
    const auto mark = [](const char* phase) {
        if (auto* window = DecodeWindow::recording())
            window->mark_current(phase);
    };
    batch_attention_store(inputs, field, key, value);
    auto pooled = batch_index_pool(inputs, field + 2, raw_index, config.pool);
    mark("qsa_cache_end");
    tb::Tensor blocks;
    if (config.capacity > config.budget) {
        auto pool_positions = positions - inputs.positions().remainder(config.pool).reshape({1, inputs.rows(), 1});
        pooled = transform(pooled, pool_positions);
        batch_pool_store(inputs, field + 3, pooled, config.pool);
        mark("qsa_pool_end");
        const auto pools = (config.capacity + config.pool - 1) / config.pool;
        auto scores = batch_index_scores(inputs, field + 3, index_query, config.pool, pools);
        mark("qsa_scores_end");
        blocks = mfq_selected_attention::live_topk_blocks(scores, inputs.positions(), config.pool,
            std::min(config.budget / config.pool, pools));
        mark("qsa_topk_end");
    }
    const auto columns = config.capacity <= config.budget ? config.capacity : config.budget + config.pool - 1;
    auto selected = batch_attention_selection(inputs, field, blocks, config.pool, config.budget, columns);
    mark("qsa_select_end");
    const IndirectKvView indirect{inputs.states().data_ptr<int64_t>(), inputs.enabled().data_ptr<int64_t>(),
                                  int(inputs.rows()), int(field), int(field + 4)};
    // Inactive MMA tiles may load index zero before applying their mask. These
    // tiny zero rows are scratch, independent of any request's allocation lifetime.
    auto zero_keys = tb::zeros({1, key.size(2), 1, key.size(3)}, key.options());
    auto zero_values = tb::zeros({1, value.size(2), 1, value.size(3)}, value.options());
    tb::Tensor attended;
    if (config.quantization.enabled()) {
        attended = cuda_kv_attention(query, zero_keys, zero_values, config.quantization, selected,
            inputs.positions().reshape({inputs.rows(), 1}), 0, {}, config.capacity,
            inputs.lengths(), {}, 0, {}, indirect);
    } else {
        attended = mfq_selected_attention::sparse(query, zero_keys, zero_values, selected,
            1.0 / std::sqrt(double(query.size(3))), false, {}, indirect);
    }
    mark("qsa_attention_end");
    return attended.to(tb::kFloat32) * tb::sigmoid(gate.to(tb::kFloat32));
}

} // namespace mfq::cuda
