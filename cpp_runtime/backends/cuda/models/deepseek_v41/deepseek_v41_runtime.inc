// Included by mfq_decode.cpp after the common linear/MoE loaders.  Keeping the
// architecture adapter here prevents the legacy decode driver from becoming
// the implementation of DeepSeek-V4.1's CED/CSA2 state machine.

namespace mfq::cuda::deepseek_v41_runtime {

using Tensor = mfq_tensor_backend::Tensor;
using CommonConfig = mfq::models::deepseek_v41::Config;

struct SharedState {
    Tensor compressed_kv;
    Tensor index_k;
    Tensor topk;
    Tensor candidate_blocks;
    Tensor previous_pre;
    Tensor attention_meta;
    std::unique_ptr<EngramHashState> engram_hash;
    EngramHashBatch engram_hashes;
    std::int64_t compressed_length = 0;
    std::int64_t ratio = 0;
    std::int64_t cache_position = -1;
    std::int64_t batch = 0;
    std::int64_t tokens = 0;
    std::int64_t next_layer = 0;

    void reset_session(std::int64_t current_batch) {
        compressed_kv = {};
        index_k = {};
        topk = {};
        candidate_blocks = {};
        previous_pre = {};
        engram_hashes = {};
        compressed_length = 0;
        ratio = 0;
        cache_position = -1;
        batch = 0;
        tokens = 0;
        next_layer = 0;
        MFQ_RUNTIME_CHECK(
            engram_hash && current_batch > 0 &&
                current_batch <= std::numeric_limits<int>::max(),
            "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->reset(static_cast<int>(current_batch));
    }

    void enter_layer(
        std::int64_t layer,
        std::int64_t layer_count,
        std::int64_t current_batch,
        std::int64_t current_tokens,
        std::int64_t current_position,
        const mfq_tensor_backend::Device& device,
        const Tensor& token_ids) {
        if (layer == 0) {
            compressed_kv = {};
            index_k = {};
            topk = {};
            candidate_blocks = {};
            compressed_length = 0;
            ratio = 0;
            cache_position = current_position;
            batch = current_batch;
            tokens = current_tokens;
            next_layer = 0;
            previous_pre = mfq_tensor_backend::zeros(
                {current_batch, current_tokens, 4},
                mfq_tensor_backend::TensorOptions()
                    .device(device)
                    .dtype(mfq_tensor_backend::kFloat32));
            previous_pre.narrow(2, 0, 1).fill_(1.0);
            MFQ_RUNTIME_CHECK(
                engram_hash && token_ids.dim() == 2 &&
                    token_ids.size(0) == current_batch &&
                    token_ids.size(1) == current_tokens,
                "DeepSeek-V4.1 Engram token state is unavailable");
            engram_hashes = engram_hash->forward(
                token_ids, current_position);
        }
        MFQ_RUNTIME_CHECK(
            layer == next_layer && layer >= 0 && layer < layer_count &&
                current_batch == batch && current_tokens == tokens &&
                current_position == cache_position &&
                previous_pre.defined(),
            "DeepSeek-V4.1 layer execution order changed");
        ++next_layer;
        if (!attention_meta.defined() || attention_meta.device() != device) {
            attention_meta = mfq_tensor_backend::empty(
                {8 * 1024 * 1024},
                mfq_tensor_backend::TensorOptions()
                    .device(device)
                    .dtype(mfq_tensor_backend::kFloat32));
        }
    }

    Tensor final_collapse(
        const Tensor& hidden,
        std::int64_t layer_count) const {
        MFQ_RUNTIME_CHECK(
            hidden.dim() == 4 && hidden.size(2) == 4 &&
                previous_pre.defined() &&
                previous_pre.sizes() == hidden.sizes().slice(0, 3) &&
                next_layer == layer_count,
            "DeepSeek-V4.1 final Mega-mHC state is incomplete");
        return (previous_pre.to(mfq_tensor_backend::kFloat32).unsqueeze(-1) *
                hidden.to(mfq_tensor_backend::kFloat32))
            .sum(2)
            .to(mfq_tensor_backend::kFloat16)
            .contiguous();
    }
};

struct AttentionState {
    Tensor local_kv;
    Tensor compressed_kv;
    Tensor index_k;
    Tensor partial_kv;
    Tensor partial_score;
    std::int64_t position = 0;
    std::int64_t compressed_length = 0;
    std::int64_t partial_length = 0;
};

static Tensor weighted_rms(
    const Tensor& input,
    const Tensor& weight,
    double epsilon) {
    const auto shape = input.sizes().vec();
    MFQ_RUNTIME_CHECK(
        !shape.empty() && weight.numel() == shape.back(),
        "DeepSeek-V4.1 weighted RMS geometry mismatch");
    return rms_norm_cuda(
               input.reshape({-1, shape.back()})
                   .to(mfq_tensor_backend::kFloat32),
               weight,
               epsilon)
        .reshape(shape)
        .to(mfq_tensor_backend::kFloat16)
        .contiguous();
}

static Tensor rotate_token_major_tail(
    Tensor input,
    const Tensor& positions,
    const Dsv4RopeTable& rope,
    bool inverse = false) {
    MFQ_RUNTIME_CHECK(
        input.dim() == 3,
        "DeepSeek-V4.1 RoPE input must be [B,T,D]");
    return dsv4_rotate_rope_tail(
               input.unsqueeze(1).contiguous(),
               positions,
               rope,
               inverse)
        .squeeze(1)
        .contiguous();
}

static Tensor empty_topk(
    std::int64_t batch,
    std::int64_t tokens,
    const mfq_tensor_backend::Device& device) {
    return mfq_tensor_backend::empty(
        {batch, tokens, 0},
        mfq_tensor_backend::TensorOptions()
            .device(device)
            .dtype(mfq_tensor_backend::kInt32));
}

static Tensor normalize_topk(
    Tensor selected,
    const Tensor& values,
    std::int64_t pool_length) {
    selected = selected.to(mfq_tensor_backend::kInt64).contiguous();
    auto valid = (selected >= 0.0) & (selected < static_cast<double>(pool_length));
    valid = valid & mfq_tensor_backend::isneginf(values).logical_not();
    auto masked = mfq_tensor_backend::where(
        valid, selected, static_cast<double>(pool_length));
    auto ordered = std::get<0>(
        mfq_tensor_backend::sort(masked, -1, false));
    return mfq_tensor_backend::where(
               ordered < static_cast<double>(pool_length), ordered, -1.0)
        .to(mfq_tensor_backend::kInt32)
        .contiguous();
}

static Tensor select_topk(
    const Tensor& scores,
    std::int64_t requested,
    std::int64_t pool_length,
    const std::optional<Tensor>& source_indices = std::nullopt) {
    const auto width = scores.size(-1);
    const auto count = std::min(requested, width);
    if (count <= 0) {
        return empty_topk(scores.size(0), scores.size(1), scores.device());
    }
    Tensor relative;
    Tensor values;
    if (count == width) {
        relative = mfq_tensor_backend::arange(
            width,
            scores.options().dtype(mfq_tensor_backend::kInt64))
            .reshape({1, 1, width})
            .expand({scores.size(0), scores.size(1), width})
            .contiguous();
        values = scores;
    } else if (count == 512 &&
               scores.scalar_type() == mfq_tensor_backend::kFloat16) {
        relative = dsv4_topk512_cuda(scores.contiguous())
            .to(mfq_tensor_backend::kInt64)
            .contiguous();
        values = scores.gather(-1, relative);
    } else {
        auto result = mfq_tensor_backend::topk(
            scores, count, -1, true, false);
        values = std::get<0>(result);
        relative = std::get<1>(result);
    }
    auto selected = source_indices
        ? source_indices->to(mfq_tensor_backend::kInt64)
              .gather(-1, relative)
        : relative;
    return normalize_topk(selected, values, pool_length);
}

static Tensor candidate_blocks_from_scores(
    const Tensor& scores,
    std::int64_t ratio,
    std::int64_t position,
    std::int64_t requested_blocks,
    std::int64_t block_size) {
    const auto batch = scores.size(0);
    const auto tokens = scores.size(1);
    const auto width = scores.size(2);
    const auto blocks = (width + block_size - 1) / block_size;
    const auto padded = blocks * block_size;
    Tensor padded_scores = scores;
    if (padded != width) {
        padded_scores = mfq_tensor_backend::cat(
            {scores,
             mfq_tensor_backend::full(
                 {batch, tokens, padded - width},
                 -std::numeric_limits<float>::infinity(),
                 scores.options())},
            -1);
    }
    auto block_scores = padded_scores
        .reshape({batch, tokens, blocks, block_size})
        .amax(-1);
    auto int_options = scores.options().dtype(mfq_tensor_backend::kInt64);
    auto visible = (mfq_tensor_backend::arange(
                        position + 1,
                        position + tokens + 1,
                        int_options)
                        .reshape({1, tokens, 1}) /
                    static_cast<double>(ratio));
    auto last = mfq_tensor_backend::where(
        visible > 0.0,
        (visible - 1.0) / static_cast<double>(block_size),
        -1.0);
    auto block_ids = mfq_tensor_backend::arange(blocks, int_options)
        .reshape({1, 1, blocks});
    block_scores = mfq_tensor_backend::where(
        block_ids == last,
        std::numeric_limits<float>::infinity(),
        block_scores);
    const auto count = std::min(requested_blocks, blocks);
    auto result = mfq_tensor_backend::topk(
        block_scores, count, -1, true, false);
    auto values = std::get<0>(result);
    auto selected = std::get<1>(result);
    return mfq_tensor_backend::where(
               mfq_tensor_backend::isneginf(values).logical_not(),
               selected,
               -1.0)
        .to(mfq_tensor_backend::kInt32)
        .contiguous();
}

static Tensor candidate_positions(
    const Tensor& blocks,
    std::int64_t block_size,
    std::int64_t pool_length) {
    const auto count = blocks.size(-1);
    auto block64 = blocks.to(mfq_tensor_backend::kInt64);
    auto within = mfq_tensor_backend::arange(
        block_size,
        blocks.options().dtype(mfq_tensor_backend::kInt64))
        .reshape({1, 1, 1, block_size});
    auto expanded = (block64.unsqueeze(-1) *
                     static_cast<double>(block_size) + within)
        .reshape({blocks.size(0), blocks.size(1), count * block_size});
    auto valid = (expanded >= 0.0) &
        (expanded < static_cast<double>(pool_length));
    return mfq_tensor_backend::where(valid, expanded, -1.0)
        .to(mfq_tensor_backend::kInt64)
        .contiguous();
}

struct MhcResult {
    Tensor branch;
    Tensor next_pre;
    Tensor post;
    Tensor combination;
};

struct Block final : ::Block {
    CommonConfig config;
    std::int64_t layer = -1;
    std::int64_t ratio = 0;
    std::int64_t max_context = 0;
    std::shared_ptr<SharedState> shared;
    AttentionState state;
    Tensor current_ids;

    Tensor attention_norm;
    Tensor mlp_norm;
    Tensor query_a_norm;
    Tensor key_value_norm;
    Tensor sinks;
    Tensor attention_mhc_function;
    Tensor attention_mhc_scale;
    Tensor attention_mhc_base;
    Tensor mlp_mhc_function;
    Tensor mlp_mhc_scale;
    Tensor mlp_mhc_base;
    Tensor compressor_norm;
    Tensor index_key_norm;

    QuantLinear query_a;
    QuantLinear query_b;
    QuantLinear key_value;
    QuantLinear output_a;
    QuantLinear output_b;
    QuantLinear compressor_key_value;
    QuantLinear compressor_gate;
    QuantLinear index_query;
    QuantLinear index_key;
    QuantLinear index_score;
    FFN mlp;
    Dsv4RopeTable rope;
    std::unique_ptr<Engram> engram;

    bool kv_source() const noexcept {
        return config.is_kv_source(layer);
    }

    bool index_source() const noexcept {
        return config.is_index_source(layer);
    }

    void reset(std::int64_t batch) override {
        const auto half = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA)
            .dtype(mfq_tensor_backend::kFloat16);
        state.local_kv = mfq_tensor_backend::zeros(
            {batch, config.sliding_window, config.head_dim}, half);
        state.compressed_kv = {};
        state.index_k = {};
        state.partial_kv = {};
        state.partial_score = {};
        if (kv_source()) {
            const auto capacity = std::max<std::int64_t>(
                1, max_context / ratio);
            state.compressed_kv = mfq_tensor_backend::zeros(
                {batch, capacity, config.head_dim}, half);
            state.index_k = mfq_tensor_backend::zeros(
                {batch, capacity, config.index_head_dim}, half);
            if (ratio > 1) {
                const auto fp32 = half.dtype(mfq_tensor_backend::kFloat32);
                state.partial_kv = mfq_tensor_backend::zeros(
                    {batch, ratio, config.head_dim}, fp32);
                state.partial_score = mfq_tensor_backend::full(
                    {batch, ratio, config.head_dim},
                    -std::numeric_limits<float>::infinity(), fp32);
            }
        }
        state.position = 0;
        state.compressed_length = 0;
        state.partial_length = 0;
        if (layer == 0) shared->reset_session(batch);
    }

    void set_token_ids(const Tensor& ids) override {
        current_ids = ids;
    }

    MhcResult collapse(
        const Tensor& residual,
        const Tensor& previous_pre,
        const Tensor& function,
        const Tensor& scale,
        const Tensor& base,
        const Tensor& norm,
        const char* profile) const {
        return g_profiler.measure(profile, [&]() {
            auto flat = residual.flatten(2).to(mfq_tensor_backend::kFloat32);
            auto inverse = mfq_tensor_backend::rsqrt(
                flat.square().mean(-1, true) + config.rms_eps);
            auto mixes = mfq_tensor_backend::matmul(
                flat * inverse, function.transpose(0, 1));
            auto expansion = dsv4_hc_pre_cuda(
                residual,
                mixes.contiguous(),
                scale,
                base,
                config.hc_sinkhorn_iters,
                config.hc_eps);
            auto next_pre = mfq_tensor_backend::sigmoid(
                mixes.slice(-1, 0, config.hc_mult) * scale.index({0}) +
                base.slice(0, 0, config.hc_mult)) + config.hc_eps;
            auto branch = (previous_pre.to(mfq_tensor_backend::kFloat32)
                                   .unsqueeze(-1) *
                           residual.to(mfq_tensor_backend::kFloat32))
                              .sum(2);
            branch = weighted_rms(branch, norm, config.rms_eps);
            return MhcResult{
                std::move(branch),
                next_pre.contiguous(),
                expansion.at(1),
                expansion.at(2)};
        });
    }

    Tensor expand(
        const Tensor& branch,
        const Tensor& residual,
        const MhcResult& mix,
        const char* profile) const {
        return g_profiler.measure(profile, [&]() {
            return dsv4_hc_post_cuda(
                branch.contiguous(),
                residual.contiguous(),
                mix.post.contiguous(),
                mix.combination.contiguous());
        });
    }

    Tensor output_projection(Tensor attention) const {
        const auto batch = attention.size(0);
        const auto tokens = attention.size(1);
        const auto rows = batch * tokens;
        const auto groups = config.o_groups;
        const auto group_width =
            config.n_heads * config.head_dim / groups;
        auto grouped = attention.contiguous()
            .reshape({rows, groups, group_width})
            .to(mfq_tensor_backend::kFloat16);
        if (output_a.is_nint() && output_a.nint.w.bits == 8 &&
            output_a.nint.w.gs == 48 &&
            output_a.nint.w.out == groups * config.o_lora_rank) {
            auto low_rank = nint_matmul_groupwise_u8(
                output_a.nint.w, grouped, groups);
            return output_b.forward(low_rank)
                .reshape({batch, tokens, config.hidden});
        }
        if (output_a.is_mxfp8() &&
            output_a.out() == groups * config.o_lora_rank) {
            auto low_rank = output_a.forward_mxfp8_groupwise(
                grouped, groups);
            return output_b.forward(low_rank)
                .reshape({batch, tokens, config.hidden});
        }
        auto expanded = output_a.forward(
            grouped.reshape({rows * groups, group_width}))
            .reshape({rows, groups, groups, config.o_lora_rank});
        std::vector<Tensor> diagonal;
        diagonal.reserve(static_cast<std::size_t>(groups));
        for (std::int64_t group = 0; group < groups; ++group) {
            diagonal.push_back(
                expanded.index({Slice(), group, group, Slice()}));
        }
        auto low_rank = mfq_tensor_backend::stack(diagonal, 1)
            .reshape({rows, groups * config.o_lora_rank})
            .to(mfq_tensor_backend::kFloat16)
            .contiguous();
        return output_b.forward(low_rank)
            .reshape({batch, tokens, config.hidden});
    }

    std::optional<Tensor> update_compressed_source(
        const Tensor& input,
        std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        if (ratio == 1) {
            return weighted_rms(
                compressor_key_value.forward(input),
                compressor_norm,
                config.rms_eps);
        }
        auto projected_kv = compressor_key_value.forward(input)
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
        auto projected_score = compressor_gate.forward(input)
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
        std::vector<Tensor> emitted;
        if (state.partial_length == 0 && position % ratio == 0) {
            const auto complete = tokens / ratio;
            const auto cutoff = complete * ratio;
            if (complete > 0) {
                auto kv_groups = projected_kv.narrow(1, 0, cutoff)
                    .reshape({batch, complete, ratio, config.head_dim});
                auto score_groups = projected_score.narrow(1, 0, cutoff)
                    .reshape({batch, complete, ratio, config.head_dim});
                auto pooled = (kv_groups *
                               mfq_tensor_backend::softmax(score_groups, 2))
                                  .sum(2);
                emitted.push_back(weighted_rms(
                    pooled.to(mfq_tensor_backend::kFloat16),
                    compressor_norm,
                    config.rms_eps));
            }
            const auto remainder = tokens - cutoff;
            if (remainder > 0) {
                state.partial_kv.narrow(1, 0, remainder).copy_(
                    projected_kv.narrow(1, cutoff, remainder));
                state.partial_score.narrow(1, 0, remainder).copy_(
                    projected_score.narrow(1, cutoff, remainder));
            }
            state.partial_length = remainder;
        } else {
            for (std::int64_t token = 0; token < tokens; ++token) {
                const auto slot = (position + token) % ratio;
                state.partial_kv.narrow(1, slot, 1).copy_(
                    projected_kv.narrow(1, token, 1));
                state.partial_score.narrow(1, slot, 1).copy_(
                    projected_score.narrow(1, token, 1));
                if (slot + 1 == ratio) {
                    auto pooled = (state.partial_kv *
                                   mfq_tensor_backend::softmax(
                                       state.partial_score, 1))
                                      .sum(1, true);
                    emitted.push_back(weighted_rms(
                        pooled.to(mfq_tensor_backend::kFloat16),
                        compressor_norm,
                        config.rms_eps));
                }
            }
            state.partial_length = (position + tokens) % ratio;
        }
        if (emitted.empty()) return std::nullopt;
        return emitted.size() == 1
            ? std::optional<Tensor>(std::move(emitted.front()))
            : std::optional<Tensor>(
                  mfq_tensor_backend::cat(emitted, 1).contiguous());
    }

    void publish_compressed_source(
        const Tensor& input,
        std::int64_t position) {
        auto latent = update_compressed_source(input, position);
        if (latent) {
            const auto count = latent->size(1);
            const auto begin = state.compressed_length;
            MFQ_RUNTIME_CHECK(
                begin + count <= state.compressed_kv.size(1),
                "DeepSeek-V4.1 compressed cache capacity exceeded");
            auto group_positions = mfq_tensor_backend::arange(
                begin * ratio,
                (begin + count) * ratio,
                ratio,
                latent->options().dtype(mfq_tensor_backend::kInt64));
            auto index_values = weighted_rms(
                index_key.forward(*latent),
                index_key_norm,
                config.rms_eps);
            index_values = rotate_token_major_tail(
                index_values, group_positions, rope);
            index_values = dsv4_fp4_sim_cuda(index_values.contiguous());
            auto compressed = rotate_token_major_tail(
                *latent, group_positions, rope);
            compressed = deepseek_v41_mxfp4_e4m3_scale_sim_cuda(
                compressed.contiguous());
            auto rows = mfq_tensor_backend::arange(
                begin,
                begin + count,
                latent->options().dtype(mfq_tensor_backend::kInt64));
            state.index_k.index_copy_(1, rows, index_values);
            state.compressed_kv.index_copy_(1, rows, compressed);
            state.compressed_length += count;
        }
        shared->compressed_kv = state.compressed_kv;
        shared->index_k = state.index_k;
        shared->compressed_length = state.compressed_length;
        shared->ratio = ratio;
    }

    Tensor compute_topk(
        const Tensor& input,
        const Tensor& q_rank,
        const Tensor& positions,
        std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        if (ratio == 0) {
            return empty_topk(batch, tokens, input.device());
        }
        if (!index_source()) {
            MFQ_RUNTIME_CHECK(
                shared->topk.defined(),
                "DeepSeek-V4.1 CSA2 consumer has no published index plan");
            return shared->topk;
        }
        auto topk = empty_topk(batch, tokens, input.device());
        const auto pool_length = shared->compressed_length;
        if (pool_length > 0) {
            auto query = index_query.forward(q_rank)
                .reshape({batch, tokens,
                          config.index_n_heads,
                          config.index_head_dim})
                .transpose(1, 2)
                .contiguous();
            query = dsv4_rotate_rope_tail(
                query, positions, rope, false)
                .transpose(1, 2)
                .contiguous();
            query = dsv4_fp4_sim_cuda(
                query.to(mfq_tensor_backend::kFloat16).contiguous());
            auto weights = index_score.forward(input)
                .reshape({batch, tokens, config.index_n_heads})
                .to(mfq_tensor_backend::kFloat16)
                .contiguous();
            auto scores = dsv4_indexer_scores_cuda(
                query,
                shared->index_k.narrow(1, 0, pool_length).contiguous(),
                weights,
                position,
                ratio);
            if (layer == config.candidate_source_layer) {
                shared->candidate_blocks = candidate_blocks_from_scores(
                    scores,
                    ratio,
                    position,
                    config.candidate_topk_blocks,
                    config.candidate_block_size);
            }
            if (layer > config.candidate_source_layer) {
                MFQ_RUNTIME_CHECK(
                    shared->candidate_blocks.defined(),
                    "DeepSeek-V4.1 hierarchical indexer has no candidates");
                auto candidates = candidate_positions(
                    shared->candidate_blocks,
                    config.candidate_block_size,
                    pool_length);
                auto safe = candidates.clamp(
                    0.0, static_cast<double>(pool_length - 1));
                auto candidate_scores = scores.gather(-1, safe);
                candidate_scores = mfq_tensor_backend::where(
                    candidates >= 0.0,
                    candidate_scores,
                    -std::numeric_limits<float>::infinity());
                topk = select_topk(
                    candidate_scores,
                    config.index_topk,
                    pool_length,
                    candidates);
            } else {
                topk = select_topk(
                    scores,
                    config.index_topk,
                    pool_length);
            }
        }
        shared->topk = topk;
        return topk;
    }

    Tensor attention_forward(
        const Tensor& input,
        const Tensor& positions,
        std::int64_t position,
        const MfqOptional<Tensor>& sequence_lengths) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        auto q_rank = weighted_rms(
            query_a.forward(input), query_a_norm, config.rms_eps);
        auto query = query_b.forward(q_rank)
            .reshape({batch, tokens, config.n_heads, config.head_dim})
            .transpose(1, 2)
            .contiguous()
            .to(mfq_tensor_backend::kFloat32);
        query = dsv4_rotate_rope_tail(query, positions, rope, false);

        auto local = weighted_rms(
            key_value.forward(input), key_value_norm, config.rms_eps)
            .reshape({batch, tokens, config.head_dim});
        local = rotate_token_major_tail(local, positions, rope);
        local = deepseek_v41_mxfp8_e4m3_sim_cuda(local.contiguous());

        if (kv_source()) {
            publish_compressed_source(input, position);
        } else if (ratio > 0) {
            MFQ_RUNTIME_CHECK(
                shared->compressed_kv.defined() &&
                    shared->index_k.defined() &&
                    shared->ratio == ratio,
                "DeepSeek-V4.1 CSA2 consumer ran before its source layer");
        }
        auto topk = compute_topk(input, q_rank, positions, position);

        Tensor local_for_attention;
        if (tokens == 1) {
            auto slot = mfq_tensor_backend::full(
                {1}, position % config.sliding_window,
                positions.options().dtype(mfq_tensor_backend::kInt64));
            state.local_kv.index_copy_(1, slot, local);
            local_for_attention = state.local_kv;
        } else {
            const auto history = std::min(
                position, config.sliding_window);
            std::vector<Tensor> local_parts;
            if (history > 0) {
                auto slots = mfq_tensor_backend::arange(
                                 position - history,
                                 position,
                                 positions.options().dtype(
                                     mfq_tensor_backend::kInt64))
                                 .remainder(
                                     static_cast<double>(config.sliding_window))
                                 .to(mfq_tensor_backend::kInt64)
                                 .contiguous();
                local_parts.push_back(
                    state.local_kv.index_select(1, slots));
            }
            local_parts.push_back(local);
            local_for_attention = local_parts.size() == 1
                ? local_parts.front()
                : mfq_tensor_backend::cat(local_parts, 1).contiguous();
            const auto recent = std::min(
                tokens, config.sliding_window);
            auto recent_slots = mfq_tensor_backend::arange(
                                    position + tokens - recent,
                                    position + tokens,
                                    positions.options().dtype(
                                        mfq_tensor_backend::kInt64))
                                    .remainder(
                                        static_cast<double>(config.sliding_window))
                                    .to(mfq_tensor_backend::kInt64)
                                    .contiguous();
            state.local_kv.index_copy_(
                1,
                recent_slots,
                local.narrow(1, tokens - recent, recent));
        }

        const auto pool_length = ratio > 0
            ? shared->compressed_length
            : 0;
        auto cache = pool_length > 0
            ? mfq_tensor_backend::cat(
                  {local_for_attention,
                   shared->compressed_kv.narrow(
                       1, 0, pool_length)},
                  1)
                  .contiguous()
            : local_for_attention.contiguous();
        std::vector<Tensor> plan;
        if (tokens == 1) {
            auto lengths = sequence_lengths.has_value()
                ? sequence_lengths.value()
                      .to(input.device(), mfq_tensor_backend::kInt64)
                      .contiguous()
                : mfq_tensor_backend::full(
                      {batch},
                      position + 1,
                      positions.options().dtype(
                          mfq_tensor_backend::kInt64));
            plan = dsv4_build_decode_plan_cuda(
                topk,
                lengths,
                pool_length,
                ratio > 0 ? ratio : 1,
                config.sliding_window);
        } else {
            MFQ_RUNTIME_CHECK(
                !sequence_lengths.has_value(),
                "DeepSeek-V4.1 chunked prefill does not accept per-row lengths");
            const auto history = std::min(
                position, config.sliding_window);
            plan = dsv4_build_prefill_plan_cuda(
                topk,
                position,
                history,
                pool_length,
                ratio > 0 ? ratio : 1,
                config.sliding_window);
        }
        auto attended = attention_dsv4_sparse_cuda(
            query,
            cache,
            plan.at(0),
            plan.at(1),
            sinks,
            shared->attention_meta,
            1.0 / std::sqrt(static_cast<double>(config.head_dim)));
        attended = dsv4_rotate_rope_tail(
                       attended.transpose(1, 2).contiguous(),
                       positions,
                       rope,
                       true)
                       .transpose(1, 2)
                       .contiguous();
        state.position += tokens;
        return output_projection(attended);
    }

    Tensor forward(
        Tensor hidden,
        Tensor positions,
        std::int64_t cache_position,
        const MfqOptional<Tensor>& sequence_lengths,
        const ::Config&,
        const RopeCache&,
        const MfqOptional<Tensor>& cache_positions = mfq_nullopt,
        const MfqOptional<Tensor>& attention_mask = mfq_nullopt) override {
        MFQ_RUNTIME_CHECK(
            current_ids.defined() && hidden.dim() == 4 &&
                hidden.size(2) == config.hc_mult &&
                hidden.size(3) == config.hidden &&
                positions.dim() == 1 &&
                positions.numel() == hidden.size(1) &&
                !cache_positions.has_value() &&
                !attention_mask.has_value() &&
                state.position == cache_position,
            "DeepSeek-V4.1 block input/cache mismatch");
        shared->enter_layer(
            layer,
            config.n_layers,
            hidden.size(0),
            hidden.size(1),
            cache_position,
            hidden.device(),
            current_ids);

        if (engram) {
            hidden = g_profiler.measure("deepseek_v41.engram", [&]() {
                return engram->forward(hidden, shared->engram_hashes);
            });
        }

        auto attention_residual = hidden;
        auto attention_mix = collapse(
            attention_residual,
            shared->previous_pre,
            attention_mhc_function,
            attention_mhc_scale,
            attention_mhc_base,
            attention_norm,
            "deepseek_v41.mhc.attention.collapse");
        auto attention = attention_forward(
            attention_mix.branch,
            positions,
            cache_position,
            sequence_lengths);
        hidden = expand(
            attention,
            attention_residual,
            attention_mix,
            "deepseek_v41.mhc.attention.expand");

        auto mlp_residual = hidden;
        auto mlp_mix = collapse(
            mlp_residual,
            attention_mix.next_pre,
            mlp_mhc_function,
            mlp_mhc_scale,
            mlp_mhc_base,
            mlp_norm,
            "deepseek_v41.mhc.mlp.collapse");
        auto feed_forward = mlp.forward(
            mlp_mix.branch.reshape(
                {-1, config.hidden}),
            current_ids)
                                .reshape(
                                    {hidden.size(0),
                                     hidden.size(1),
                                     config.hidden});
        hidden = expand(
            feed_forward,
            mlp_residual,
            mlp_mix,
            "deepseek_v41.mhc.mlp.expand");
        shared->previous_pre = std::move(mlp_mix.next_pre);
        return hidden;
    }
};

static FFN load_moe(
    const MfqFile& model,
    const ::Config& runtime,
    std::int64_t layer) {
    const auto prefix = "model.block." + std::to_string(layer) + ".mlp.";
    FFN result;
    result.is_moe = true;
    const bool split_gate = model.has_record(prefix + "experts.gate.weight");
    const bool split_up = model.has_record(prefix + "experts.up.weight");
    if (split_gate != split_up) {
        throw std::runtime_error(
            "DeepSeek-V4.1 split routed Gate/Up records are incomplete at layer " +
            std::to_string(layer));
    }
    result.moe_split_gate_up = split_gate;
    if (split_gate) {
        result.moe_gate = load_nint_moe_gpu(
            model, prefix + "experts.gate.weight", true,
            static_cast<int>(layer), "gate");
        result.moe_up = load_nint_moe_gpu(
            model, prefix + "experts.up.weight", true,
            static_cast<int>(layer), "up");
    } else {
        result.moe_gate_up = load_nint_moe_gpu(
            model, prefix + "experts.gate_up.weight", true,
            static_cast<int>(layer), "gate_up");
    }
    result.moe_down = load_nint_moe_gpu(
        model, prefix + "experts.down.weight", true,
        static_cast<int>(layer), "down");
    result.moe_router = load_dense_gpu(model, prefix + "router.weight")
        .to(mfq_tensor_backend::kFloat32)
        .contiguous();
    result.moe_router_bias = load_dense_gpu(model, prefix + "router.bias")
        .to(mfq_tensor_backend::kFloat32)
        .contiguous();
    result.moe_top_k = static_cast<int>(runtime.num_experts_per_tok);
    result.moe_use_sqrt_softplus = true;
    result.moe_normalize = runtime.norm_topk_prob;
    result.moe_delayed_softmax = false;
    result.moe_shared_ungated = true;
    result.moe_router_scale = runtime.routed_scaling_factor;
    result.swiglu_limit = runtime.swiglu_limit;
    result.moe_layer = static_cast<int>(layer);
    result.shared = std::make_unique<FFN>();
    result.shared->down = load_quant_linear(
        model, prefix + "shared_expert.down.weight");
    result.shared->gate_up = load_paired_gate_up(
        model,
        {prefix + "shared_expert.gate.weight",
         prefix + "shared_expert.up.weight"},
        result.shared->down,
        0);
    result.shared->swiglu_limit = runtime.swiglu_limit;
    prepare_ffn_workspaces(*result.shared);
    return result;
}

static std::unique_ptr<::Block> load_block(
    const MfqFile& model,
    const ::Config& runtime,
    std::int64_t layer,
    const std::shared_ptr<SharedState>& shared) {
    MFQ_RUNTIME_CHECK(
        runtime.deepseek_v41.has_value() && shared,
        "missing DeepSeek-V4.1 CUDA configuration/state");
    const auto& config = *runtime.deepseek_v41;
    const auto prefix = "model.block." + std::to_string(layer) + ".";
    auto result = std::make_unique<Block>();
    result->config = config;
    result->layer = layer;
    result->ratio = config.compress_ratios.at(
        static_cast<std::size_t>(layer));
    result->max_context = runtime.max_position_embeddings;
    result->shared = shared;
    if (config.has_engram(layer)) {
        result->engram = Engram::load(
            model, config, static_cast<int>(layer));
    }
    result->attention_norm = load_dense_gpu(
        model, prefix + "attention.norm.weight");
    result->mlp_norm = load_dense_gpu(
        model, prefix + "mlp.norm.weight");
    result->query_a_norm = load_dense_gpu(
        model, prefix + "attention.query_a_norm.weight");
    result->key_value_norm = load_dense_gpu(
        model, prefix + "attention.key_value_norm.weight");
    result->sinks = load_dense_gpu(
        model, prefix + "attention.sink")
                            .to(mfq_tensor_backend::kFloat32)
                            .contiguous();
    result->attention_mhc_function = load_dense_gpu(
        model, prefix + "attention.mhc.pre.function")
                                         .to(mfq_tensor_backend::kFloat32)
                                         .contiguous();
    result->attention_mhc_scale = load_dense_gpu(
        model, prefix + "attention.mhc.pre.scale")
                                      .to(mfq_tensor_backend::kFloat32)
                                      .contiguous();
    result->attention_mhc_base = load_dense_gpu(
        model, prefix + "attention.mhc.pre.base")
                                     .to(mfq_tensor_backend::kFloat32)
                                     .contiguous();
    result->mlp_mhc_function = load_dense_gpu(
        model, prefix + "mlp.mhc.pre.function")
                                   .to(mfq_tensor_backend::kFloat32)
                                   .contiguous();
    result->mlp_mhc_scale = load_dense_gpu(
        model, prefix + "mlp.mhc.pre.scale")
                                .to(mfq_tensor_backend::kFloat32)
                                .contiguous();
    result->mlp_mhc_base = load_dense_gpu(
        model, prefix + "mlp.mhc.pre.base")
                               .to(mfq_tensor_backend::kFloat32)
                               .contiguous();
    result->query_a = load_quant_linear(
        model, prefix + "attention.query_a.weight");
    result->query_b = load_quant_linear(
        model, prefix + "attention.query_b.weight");
    result->key_value = load_quant_linear(
        model, prefix + "attention.key_value.weight");
    result->output_a = load_quant_linear(
        model, prefix + "attention.output_a.weight");
    result->output_b = load_quant_linear(
        model, prefix + "attention.output_b.weight");
    if (result->kv_source()) {
        result->compressor_key_value = load_quant_linear(
            model, prefix + "attention.compressor.key_value.weight");
        result->compressor_norm = load_dense_gpu(
            model, prefix + "attention.compressor.norm.weight");
        result->index_key = load_quant_linear(
            model, prefix + "attention.indexer.key.weight");
        result->index_key_norm = load_dense_gpu(
            model, prefix + "attention.indexer.key_norm.weight");
        if (result->ratio > 1) {
            result->compressor_gate = load_quant_linear(
                model, prefix + "attention.compressor.gate.weight");
        }
    }
    if (result->index_source()) {
        result->index_query = load_quant_linear(
            model, prefix + "attention.indexer.query.weight");
        result->index_score = load_quant_linear(
            model, prefix + "attention.indexer.score.weight");
    }
    result->mlp = load_moe(model, runtime, layer);
    result->rope = Dsv4RopeTable(runtime, result->ratio);

    const auto function_width = config.hc_mult * config.hidden;
    const auto valid_mhc = [&](const Tensor& function,
                               const Tensor& scale,
                               const Tensor& base,
                               const Tensor& norm) {
        return function.dim() == 2 && function.size(0) == 24 &&
            function.size(1) == function_width &&
            scale.numel() == 3 && base.numel() == 24 &&
            norm.numel() == config.hidden;
    };
    const bool base_shapes =
        valid_mhc(
            result->attention_mhc_function,
            result->attention_mhc_scale,
            result->attention_mhc_base,
            result->attention_norm) &&
        valid_mhc(
            result->mlp_mhc_function,
            result->mlp_mhc_scale,
            result->mlp_mhc_base,
            result->mlp_norm) &&
        result->query_a_norm.numel() == config.q_lora_rank &&
        result->key_value_norm.numel() == config.head_dim &&
        result->sinks.numel() == config.n_heads &&
        result->query_a.neuron_len() == config.hidden &&
        result->query_a.out() == config.q_lora_rank &&
        result->query_b.neuron_len() == config.q_lora_rank &&
        result->query_b.out() == config.n_heads * config.head_dim &&
        result->key_value.neuron_len() == config.hidden &&
        result->key_value.out() == config.head_dim &&
        result->output_a.neuron_len() ==
            config.n_heads * config.head_dim / config.o_groups &&
        result->output_a.out() == config.o_groups * config.o_lora_rank &&
        result->output_b.neuron_len() ==
            config.o_groups * config.o_lora_rank &&
        result->output_b.out() == config.hidden;
    const bool source_shapes = !result->kv_source() ||
        (result->compressor_key_value.neuron_len() == config.hidden &&
         result->compressor_key_value.out() == config.head_dim &&
         result->compressor_norm.numel() == config.head_dim &&
         result->index_key.neuron_len() == config.head_dim &&
         result->index_key.out() == config.index_head_dim &&
         result->index_key_norm.numel() == config.index_head_dim &&
         (result->ratio == 1 ||
          (result->compressor_gate.neuron_len() == config.hidden &&
           result->compressor_gate.out() == config.head_dim)));
    const bool index_shapes = !result->index_source() ||
        (result->index_query.neuron_len() == config.q_lora_rank &&
         result->index_query.out() ==
             config.index_n_heads * config.index_head_dim &&
         result->index_score.neuron_len() == config.hidden &&
         result->index_score.out() == config.index_n_heads);
    MFQ_RUNTIME_CHECK(
        config.hc_mult == 4 && config.n_heads == 64 &&
            config.head_dim == 512 && config.rope_head_dim == 64 &&
            config.index_n_heads == 32 && config.index_head_dim == 128 &&
            base_shapes && source_shapes && index_shapes,
        "DeepSeek-V4.1 CUDA tensor geometry disagrees at layer ",
        layer);
    return result;
}

static int run_self_check() {
    EngramHashState::self_check();
    auto options = mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCUDA)
        .dtype(mfq_tensor_backend::kInt32);
    auto blocks = mfq_tensor_backend::tensor(
        std::vector<std::int32_t>{0, 2, -1}, options)
                      .reshape({1, 1, 3});
    auto positions = candidate_positions(blocks, 2, 5)
        .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
        .contiguous();
    const std::vector<std::int64_t> expected{
        0, 1, 4, -1, -1, -1};
    const auto* values = positions.data_ptr<std::int64_t>();
    MFQ_RUNTIME_CHECK(
        positions.numel() == static_cast<std::int64_t>(expected.size()) &&
            std::equal(expected.begin(), expected.end(), values),
        "DeepSeek-V4.1 candidate hierarchy contract failed");
    std::cout << "deepseek_v41_runtime_check=ok\n";
    return 0;
}

} // namespace mfq::cuda::deepseek_v41_runtime
