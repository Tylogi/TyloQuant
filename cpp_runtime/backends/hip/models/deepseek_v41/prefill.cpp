#include "ops.h"
#include "storage/moe_expert_cache.h"
#include <cstring>

namespace mfq::cuda {
namespace {
using Tensor = mfq_tensor_backend::Tensor;
using HashBatch = mfq::models::deepseek_v41::EngramHashBatch;

Tensor append_tail(const Tensor& previous, const Tensor& current, int64_t window) {
    if (current.size(1) >= window)
        return current.narrow(1, current.size(1) - window, window).clone();
    auto combined = previous.defined() ? mfq_tensor_backend::cat({previous, current}, 1) : current;
    const auto count = std::min(window, combined.size(1));
    return combined.narrow(1, combined.size(1) - count, count).clone();
}

HashBatch append_hash_tail(const HashBatch& previous, const HashBatch& current, int window) {
    HashBatch result;
    result.batch = current.batch;
    result.layers = current.layers;
    result.columns = current.columns;
    result.tokens = std::min(window, previous.tokens + current.tokens);
    result.values.resize(std::size_t(result.layers) * result.batch * result.tokens * result.columns);
    const int start = previous.tokens + current.tokens - result.tokens;
    for (int layer = 0; layer < result.layers; ++layer) for (int batch = 0; batch < result.batch; ++batch) {
        auto* output = result.values.data() +
            ((std::size_t(layer) * result.batch + batch) * result.tokens) * result.columns;
        for (int token = 0; token < result.tokens; ++token) {
            const int position = start + token;
            const auto* input = position < previous.tokens
                ? previous.layer_data(layer) + (std::size_t(batch) * previous.tokens + position) * result.columns
                : current.layer_data(layer) + (std::size_t(batch) * current.tokens + position - previous.tokens) * result.columns;
            std::memcpy(output + std::size_t(token) * result.columns, input,
                        std::size_t(result.columns) * sizeof(std::int64_t));
        }
    }
    return result;
}
}

Tensor deepseek_v41_prefill_hidden(DeepseekV41CausalLm& model, Tensor ids,
                                   int64_t compute_chunk, const std::function<bool()>& stopped, Tensor* raw_hidden) {
    auto scope = model.execution_scope();
    ids = model.device_ids(std::move(ids));
    const auto batch = ids.size(0), tokens = ids.size(1);
    const auto* profile=std::getenv("MFQ_PROFILE_SSD_EXPERTS");
    const bool trace=profile && profile[0]=='1';
    if(trace)std::cerr<<"runtime_deepseek_v41_prefill phase=prepare tokens="<<tokens<<'\n';
    MFQ_RUNTIME_CHECK(model.shared && batch > 0 && tokens > 0 && compute_chunk > 0 &&
                          model.speculative_start < 0 &&
                          model.cache_pos + tokens <= model.max_position_embeddings(),
                      "invalid DeepSeek-V4.1 encoder prefill request");
    if (model.adapter_requires_batch_reset(batch) || model.cache_pos == 0) model.reset(batch);
    const auto begin = model.cache_pos, end = begin + tokens;
    const auto& config = model.config;
    const auto split = config.causal_encoder_layers;
    MFQ_RUNTIME_CHECK(split > 0 && split < model.num_hidden_layers(), "invalid CED encoder boundary");
    compute_chunk = std::min<int64_t>(compute_chunk, 8192);
    Tensor tail_hidden, tail_pre, tail_ids;
    HashBatch tail_hashes;
    const auto cancelled = [&] {
        if (!stopped || !stopped()) return false;
        model.reset(batch);
        return true;
    };
    const auto* profile_layer=std::getenv("MFQ_PROFILE_DSV41_LAYER");
    const auto apply_layer=[&](auto& ops,int64_t layer,Tensor hidden) {
        auto& profiler=model.execution->profiler;
        const bool selected=profile_layer && layer==std::atoi(profile_layer);
        const bool previous=profiler.enabled;
        if(selected) {profiler.reset();profiler.enabled=true;}
        try {
            hidden=ops.layer(model.blocks.at(layer),std::move(hidden));
            if(selected)profiler.report("deepseek_v41.layer."+std::to_string(layer));
        } catch(...) {profiler.enabled=previous;throw;}
        profiler.enabled=previous;
        return hidden;
    };
    for (int64_t offset = 0; offset < tokens; offset += compute_chunk) {
        if (cancelled()) return {};
        const auto count = std::min(compute_chunk, tokens - offset);
        typename DeepseekV41CausalLm::Inputs input;
        input.ids = ids.narrow(1, offset, count).contiguous();
        input.input_embeddings = model.embed_forward(input.ids);
        auto ops = model.forward_ops(model, std::move(input));
        ops.prepare_inputs();
        ops.cache_positions = ops.position_range(model.cache_pos, count);
        ops.pos = ops.full_positions = ops.cache_positions;
        auto hidden = ops.prepare_hidden(batch, count);
        model.adapter_begin_forward(raw_hidden != nullptr);
        for (int64_t layer = 0; layer < split; ++layer) {
            if (cancelled()) return {};
            if(trace)std::cerr<<"runtime_deepseek_v41_prefill phase=encoder layer="<<layer<<" tokens="<<count<<'\n';
            hidden = apply_layer(ops,layer,std::move(hidden));
        }
        // Decoder global KV comes from the encoder boundary for every prompt token.
        // Decoder attention and FFNs run only over the bounded local replay below.
        for (int64_t layer = split; layer < model.num_hidden_layers(); ++layer) {
            auto& block = dynamic_cast<deepseek_v41_runtime::Block&>(*model.blocks.at(layer));
            if (!block.kv_source()) continue;
            auto source = deepseek_v41_hc_input_cuda(hidden, model.shared->previous_pre,
                                                     block.attention_norm, config.rms_eps);
            model.execution->profiler.measure("deepseek_v41.ced.global_kv", [&] {
                block.publish_compressed_source(*model.execution, source, model.cache_pos);
                return source;
            });
            block.prefilled_global_end = model.cache_pos + count;
        }
        tail_hidden = append_tail(tail_hidden, hidden, config.sliding_window);
        tail_pre = append_tail(tail_pre, model.shared->previous_pre, config.sliding_window);
        tail_ids = append_tail(tail_ids, ops.ids, config.sliding_window);
        tail_hashes = append_hash_tail(tail_hashes, model.shared->engram_hashes,
                                       static_cast<int>(config.sliding_window));
        model.cache_pos += count;
    }
    const auto replay_tokens = tail_ids.size(1), replay_begin = end - replay_tokens;
    const bool truncated_local = replay_begin > begin;
    model.cache_pos = replay_begin;
    model.shared->batch = batch;
    model.shared->tokens = replay_tokens;
    model.shared->cache_position = replay_begin;
    model.shared->previous_pre = std::move(tail_pre);
    model.shared->engram_hashes = std::move(tail_hashes);
    typename DeepseekV41CausalLm::Inputs input;
    input.ids = std::move(tail_ids);
    auto ops = model.forward_ops(model, std::move(input));
    ops.prepare_inputs();
    ops.cache_positions = ops.position_range(replay_begin, replay_tokens);
    ops.pos = ops.full_positions = ops.cache_positions;
    for (int64_t layer = split; layer < model.num_hidden_layers(); ++layer) {
        if (cancelled()) return {};
        if(trace)std::cerr<<"runtime_deepseek_v41_prefill phase=decoder layer="<<layer<<" tokens="<<replay_tokens<<'\n';
        auto& block = dynamic_cast<deepseek_v41_runtime::Block&>(*model.blocks.at(layer));
        block.state.position = replay_begin;
        block.replay_local_window = truncated_local;
        tail_hidden = apply_layer(ops,layer,std::move(tail_hidden));
    }
    model.adapter_finish_forward(ops.full_positions, batch, replay_tokens);
    model.cache_pos = end;
    std::cerr << "runtime_deepseek_v41_ced_prefill tokens=" << tokens << " encoder_layers=" << split
              << " decoder_tokens=" << replay_tokens << " compute_chunk=" << compute_chunk << '\n';
    ops.raw_hidden = raw_hidden;
    return ops.finish(std::move(tail_hidden), batch, replay_tokens);
}
} // namespace mfq::cuda
