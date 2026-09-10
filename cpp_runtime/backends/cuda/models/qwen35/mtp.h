#pragma once

// Qwen3.5's predictor shares the main embedding/output head and owns only
// its fusion/norm/attention/FFN weights and an independent attention history.
struct CudaQwen35Mtp final : CudaMtpModule {
    Config c;
    QuantLinear fusion;
    mfq_tensor_backend::Tensor hidden_norm, embedding_norm, output_norm;
    std::vector<std::unique_ptr<Block>> blocks;
    int64_t cache_pos = 0;

    static std::optional<CudaQwen35Mtp> load_if_present(
            const MfqFile& file,
            const Config& main) {
        const bool fusion_present = file.has_record("predictor.fusion.weight");
        const bool any = fusion_present ||
            file.has_record("predictor.hidden_norm.weight") ||
            file.has_record("predictor.embedding_norm.weight") ||
            file.has_record("predictor.output_norm.weight") ||
            file.has_record("predictor.block.0.attention.query.weight");
        if (!fusion_present) {
            MFQ_RUNTIME_CHECK(!any, "Qwen MFQ contains an incomplete MTP head");
            return std::nullopt;
        }
        MFQ_RUNTIME_CHECK(
            main.mtp_num_hidden_layers > 0 &&
                !main.mtp_use_dedicated_embeddings,
            "Qwen MTP requires declared predictor layers and shared embeddings");
        CudaQwen35Mtp result;
        result.c = main;
        result.c.tensor_root = "predictor";
        result.c.num_hidden_layers = main.mtp_num_hidden_layers;
        result.c.layer_types.assign(
            static_cast<size_t>(main.mtp_num_hidden_layers),
            "full_attention");
        result.hidden_norm = load_dense_gpu(
            file, "predictor.hidden_norm.weight");
        result.embedding_norm = load_dense_gpu(
            file, "predictor.embedding_norm.weight");
        result.output_norm = load_dense_gpu(
            file, "predictor.output_norm.weight");
        result.fusion = load_quant_linear(file, "predictor.fusion.weight");
        MFQ_RUNTIME_CHECK(
            result.fusion.neuron_len() == 2 * main.hidden_size &&
                result.fusion.out() == main.hidden_size &&
                result.hidden_norm.numel() == main.hidden_size &&
                result.embedding_norm.numel() == main.hidden_size &&
                result.output_norm.numel() == main.hidden_size,
            "Qwen MTP component dimensions disagree with the backbone");
        for (int layer = 0; layer < main.mtp_num_hidden_layers; ++layer) {
            auto block = load_block(
                file, result.c, layer, "full_attention");
            block->cuda_device = g_layer_placement.primary_device();
            result.blocks.push_back(std::move(block));
        }
        return result;
    }

    void reset(int64_t batch = 1) override {
        cache_pos = 0;
        for (auto& block : blocks) block->reset(batch);
    }

    mfq_tensor_backend::Tensor forward(
            Model& main,
            mfq_tensor_backend::Tensor hidden,
            mfq_tensor_backend::Tensor next_ids) override {
        MFQ_RUNTIME_CHECK(
            hidden.dim() == 3 && next_ids.dim() == 2 &&
                hidden.size(0) == next_ids.size(0) &&
                hidden.size(1) == next_ids.size(1) &&
                hidden.size(2) == c.hidden_size && hidden.size(1) > 0,
            "Qwen MTP inputs must be matching [B,T,H] hidden states and "
            "[B,T] next-token IDs");
        const auto batch = hidden.size(0);
        const auto tokens = hidden.size(1);
        MFQ_RUNTIME_CHECK(
            cache_pos + tokens <= c.max_position_embeddings,
            "Qwen MTP history exceeds context capacity");
        auto embedded = main.embed_forward(next_ids).to(hidden.scalar_type());
        auto e = qwen_rms_norm(
            embedded.reshape({batch * tokens, c.hidden_size})
                .to(mfq_tensor_backend::kFloat32),
            embedding_norm, c).reshape_as(hidden);
        auto h = qwen_rms_norm(
            hidden.reshape({batch * tokens, c.hidden_size})
                .to(mfq_tensor_backend::kFloat32),
            hidden_norm, c).reshape_as(hidden);
        trace_gemma_stage(0, "mtp.embedding_norm", e);
        trace_gemma_stage(0, "mtp.hidden_norm", h);
        auto x = fusion.forward(mfq_tensor_backend::cat({e, h}, -1));
        trace_gemma_stage(0, "mtp.fusion", x);
        auto pos = mfq_tensor_backend::arange(
            cache_pos, cache_pos + tokens,
            mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kInt64));
        MfqOptional<mfq_tensor_backend::Tensor> length = mfq_nullopt;
        if (tokens == 1 && cache_pos > 0) {
            length = mfq_tensor_backend::full(
                {batch}, cache_pos + 1, pos.options());
        }
        for (auto& block : blocks) {
            x = block->forward(x, pos, cache_pos, length, c, main.rope);
        }
        cache_pos += tokens;
        auto output = qwen_rms_norm(
            x.reshape({batch * tokens, c.hidden_size})
                .to(mfq_tensor_backend::kFloat32),
            output_norm, c).reshape({batch, tokens, c.hidden_size});
        trace_gemma_stage(0, "mtp.output_norm", output);
        return output;
    }

    int64_t cache_position() const noexcept override { return cache_pos; }

    void trim_cache_to(int64_t position) override {
        MFQ_RUNTIME_CHECK(
            position >= 0 && position <= cache_pos,
            "Qwen MTP cache trim position is invalid");
        // Full-attention cache storage is position addressed. Lowering the
        // logical boundary makes the next predictor pass overwrite the
        // discarded draft suffix without copying history-sized tensors.
        cache_pos = position;
    }

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
};
