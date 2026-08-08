// MiniCPM-o 4.5 native CUDA composite graph.
//
// This file is included by mfq_decode.cpp after the generic causal-LM loader.
// It deliberately reuses QuantLinear and FullBlock so compact matrix weights
// stay on the same production CUDA kernels as the text-only runtime.

static constexpr const char * MINICPMO45_RESAMPLER_POS_EMBED_ASSET =
    "__mfq_asset__/minicpmo45-resampler-pos-embed-v1.bf16";

struct MiniCPMO45Linear {
    QuantLinear weight;
    torch::Tensor bias;

    static MiniCPMO45Linear load(
            const MfqFile & mfq,
            const std::string & prefix,
            bool with_bias = true) {
        MiniCPMO45Linear result;
        result.weight = load_quant_linear(mfq, prefix + ".weight");
        const std::string bias_name = prefix + ".bias";
        if (with_bias && mfq.has_record(bias_name)) {
            result.bias = load_dense_gpu(mfq, bias_name);
        }
        return result;
    }

    torch::Tensor forward(torch::Tensor input) const {
        const auto output_dtype = input.scalar_type();
        if (weight.is_dense()) {
            const auto dense_dtype = weight.dense.scalar_type();
            std::optional<torch::Tensor> dense_bias = std::nullopt;
            if (bias.defined()) {
                dense_bias = bias.to(dense_dtype);
            }
            return at::linear(
                input.to(dense_dtype), weight.dense, dense_bias)
                .to(output_dtype)
                .contiguous();
        }
        auto output = weight.forward(input);
        if (bias.defined()) {
            output = output + bias.to(output.scalar_type());
        }
        return output.to(output_dtype).contiguous();
    }
};

static torch::Tensor load_dense_native_gpu(
        const MfqFile & mfq,
        const std::string & name) {
    c10::cuda::CUDAGuard guard(active_weight_load_device());
    const auto & record = mfq.record(name);
    const auto blob = mfq.read_blob(name);
    size_t offset = 0;
    const uint32_t dimensions = read_u32_from(blob, offset);
    std::vector<int64_t> shape(dimensions);
    for (uint32_t index = 0; index < dimensions; ++index) {
        shape[index] = read_i64_from(blob, offset);
    }
    torch::Tensor value;
    if (record.dtype == "BF16") {
        value = torch::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            torch::TensorOptions().dtype(torch::kBFloat16)).clone();
    } else if (record.dtype == "F16") {
        value = torch::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            torch::TensorOptions().dtype(torch::kFloat16)).clone();
    } else if (record.dtype == "F32") {
        value = torch::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            torch::TensorOptions().dtype(torch::kFloat32)).clone();
    } else if (record.dtype == "I64") {
        value = torch::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            torch::TensorOptions().dtype(torch::kInt64)).clone();
    } else if (record.dtype == "I32") {
        value = torch::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            torch::TensorOptions().dtype(torch::kInt32)).clone();
    } else {
        throw std::runtime_error(
            "MiniCPM-o dense tensor has unsupported dtype: " +
            name + " dtype=" + record.dtype);
    }
    return value.to(torch::kCUDA).contiguous();
}

static void minicpmo45_write_pickle_tensor(
        const torch::Tensor & value,
        const std::string & path) {
    auto bytes = torch::pickle_save(
        value.detach().to(torch::kCPU).contiguous());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "failed to create MiniCPM-o tensor file: " + path);
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error(
            "failed to write MiniCPM-o tensor file: " + path);
    }
}

static torch::Tensor minicpmo45_embedding(
        const QuantLinear & embedding,
        torch::Tensor ids,
        torch::ScalarType output_dtype = torch::kBFloat16) {
    ids = ids.contiguous().to(torch::kCUDA, torch::kInt64);
    torch::Tensor output;
    if (embedding.is_dense()) {
        auto shape = ids.sizes().vec();
        shape.push_back(embedding.dense.size(1));
        output = embedding.dense.index_select(
            0, ids.reshape({-1})).reshape(shape);
    } else if (embedding.is_nvq()) {
        output = nvq_embedding(embedding.nvq.w, ids);
    } else if (embedding.is_nint()) {
        if (embedding.nint.w.q8_zero) {
            output = nint8_zero_embedding_lookup_cuda(
                embedding.nint.w.q_packed,
                embedding.nint.w.q8_zero_scale,
                ids, embedding.nint.w.neuron_len);
        } else if (embedding.nint.w.bits == 4) {
            output = nint_embedding_lookup_packed_compact_cuda(
                embedding.nint.w.q_packed,
                embedding.nint.w.sub_scale,
                embedding.nint.w.sub_min,
                embedding.nint.w.neuron_scale,
                embedding.nint.w.neuron_min,
                ids, embedding.nint.w.neuron_len,
                embedding.nint.w.gs);
        } else {
            output = nint_embedding_lookup_packed_compact_bits_cuda(
                embedding.nint.w.q_packed,
                embedding.nint.w.sub_scale,
                embedding.nint.w.sub_min,
                embedding.nint.w.neuron_scale,
                embedding.nint.w.neuron_min,
                ids, embedding.nint.w.neuron_len,
                embedding.nint.w.gs,
                embedding.nint.w.bits);
        }
    } else {
        throw std::runtime_error(
            "MiniCPM-o embedding does not support MXFP8");
    }
    return output.to(output_dtype).contiguous();
}

static torch::Tensor minicpmo45_layer_norm(
        torch::Tensor input,
        const torch::Tensor & weight,
        const torch::Tensor & bias,
        double eps) {
    const auto dtype = input.scalar_type();
    auto output = torch::layer_norm(
        input, {input.size(-1)},
        weight.to(dtype), bias.to(dtype), eps);
    return output.to(dtype).contiguous();
}

static torch::Tensor minicpmo45_attention(
        torch::Tensor query,
        torch::Tensor key,
        torch::Tensor value,
    c10::optional<torch::Tensor> additive_mask,
        double scale) {
    auto scores = torch::matmul(
        query,
        key.transpose(-2, -1)) * scale;
    if (additive_mask.has_value()) {
        scores = scores + additive_mask.value().to(scores.scalar_type());
    }
    auto probabilities = torch::softmax(
        scores, -1, torch::kFloat32).to(value.scalar_type());
    return torch::matmul(probabilities, value).contiguous();
}

struct MiniCPMO45VisionAttention {
    MiniCPMO45Linear q;
    MiniCPMO45Linear k;
    MiniCPMO45Linear v;
    MiniCPMO45Linear output;
    int64_t heads = 16;
    int64_t head_dim = 72;

    static MiniCPMO45VisionAttention load(
            const MfqFile & mfq,
            const std::string & prefix) {
        MiniCPMO45VisionAttention result;
        result.q = MiniCPMO45Linear::load(mfq, prefix + ".q_proj");
        result.k = MiniCPMO45Linear::load(mfq, prefix + ".k_proj");
        result.v = MiniCPMO45Linear::load(mfq, prefix + ".v_proj");
        result.output = MiniCPMO45Linear::load(
            mfq, prefix + ".out_proj");
        return result;
    }

    torch::Tensor forward(
            torch::Tensor input,
            c10::optional<torch::Tensor> mask) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](torch::Tensor value) {
            return value.reshape({batch, tokens, heads, head_dim})
                .transpose(1, 2).contiguous();
        };
        auto attended = minicpmo45_attention(
            reshape(q.forward(input)),
            reshape(k.forward(input)),
            reshape(v.forward(input)),
            mask, 1.0 / std::sqrt(static_cast<double>(head_dim)));
        attended = attended.transpose(1, 2).reshape(
            {batch, tokens, heads * head_dim});
        return output.forward(attended);
    }
};

struct MiniCPMO45VisionLayer {
    MiniCPMO45VisionAttention attention;
    torch::Tensor norm1_weight;
    torch::Tensor norm1_bias;
    torch::Tensor norm2_weight;
    torch::Tensor norm2_bias;
    MiniCPMO45Linear fc1;
    MiniCPMO45Linear fc2;

    static MiniCPMO45VisionLayer load(
            const MfqFile & mfq,
            int index) {
        const std::string prefix =
            "vpm.encoder.layers." + std::to_string(index);
        MiniCPMO45VisionLayer result;
        result.attention = MiniCPMO45VisionAttention::load(
            mfq, prefix + ".self_attn");
        result.norm1_weight = load_dense_native_gpu(
            mfq, prefix + ".layer_norm1.weight");
        result.norm1_bias = load_dense_native_gpu(
            mfq, prefix + ".layer_norm1.bias");
        result.norm2_weight = load_dense_native_gpu(
            mfq, prefix + ".layer_norm2.weight");
        result.norm2_bias = load_dense_native_gpu(
            mfq, prefix + ".layer_norm2.bias");
        result.fc1 = MiniCPMO45Linear::load(mfq, prefix + ".mlp.fc1");
        result.fc2 = MiniCPMO45Linear::load(mfq, prefix + ".mlp.fc2");
        return result;
    }

    torch::Tensor forward(
            torch::Tensor input,
            c10::optional<torch::Tensor> mask) const {
        auto normalized = minicpmo45_layer_norm(
            input, norm1_weight, norm1_bias, 1e-6);
        auto hidden = input + attention.forward(normalized, mask);
        normalized = minicpmo45_layer_norm(
            hidden, norm2_weight, norm2_bias, 1e-6);
        auto mlp = fc2.forward(torch::gelu(fc1.forward(normalized), "tanh"));
        return (hidden + mlp).contiguous();
    }
};

struct MiniCPMO45VisionEncoder {
    torch::Tensor patch_weight;
    torch::Tensor patch_bias;
    torch::Tensor position_embedding;
    std::vector<MiniCPMO45VisionLayer> layers;
    torch::Tensor post_norm_weight;
    torch::Tensor post_norm_bias;
    int64_t patch_size = 14;
    int64_t position_side = 70;

    static MiniCPMO45VisionEncoder load(const MfqFile & mfq) {
        MiniCPMO45VisionEncoder result;
        result.patch_weight = load_dense_native_gpu(
            mfq, "vpm.embeddings.patch_embedding.weight");
        result.patch_bias = load_dense_native_gpu(
            mfq, "vpm.embeddings.patch_embedding.bias");
        result.position_embedding = load_dense_native_gpu(
            mfq, "vpm.embeddings.position_embedding.weight");
        result.post_norm_weight = load_dense_native_gpu(
            mfq, "vpm.post_layernorm.weight");
        result.post_norm_bias = load_dense_native_gpu(
            mfq, "vpm.post_layernorm.bias");
        result.layers.reserve(27);
        for (int index = 0; index < 27; ++index) {
            result.layers.push_back(
                MiniCPMO45VisionLayer::load(mfq, index));
        }
        if (result.patch_weight.sizes().vec() !=
                std::vector<int64_t>({1152, 3, 14, 14}) ||
            result.position_embedding.sizes().vec() !=
                std::vector<int64_t>({4900, 1152})) {
            throw std::runtime_error(
                "MiniCPM-o SigLIP tensor shapes disagree with version 4.5");
        }
        return result;
    }

    torch::Tensor forward(
            torch::Tensor pixels,
            torch::Tensor patch_mask,
            torch::Tensor target_sizes) const {
        if (patch_mask.dim() == 3) {
            patch_mask = patch_mask.flatten(1);
        }
        if (pixels.dim() != 4 || pixels.size(1) != 3 ||
                patch_mask.dim() != 2 || target_sizes.dim() != 2 ||
                target_sizes.size(1) != 2 ||
                pixels.size(0) != patch_mask.size(0) ||
                pixels.size(0) != target_sizes.size(0) ||
                target_sizes.device().is_cuda()) {
            throw std::runtime_error(
                "MiniCPM-o vision input geometry is invalid");
        }
        auto embedded = torch::conv2d(
            pixels.to(patch_weight.scalar_type()),
            patch_weight, patch_bias,
            std::vector<int64_t>{patch_size, patch_size},
            std::vector<int64_t>{0, 0},
            std::vector<int64_t>{1, 1}, 1)
            .flatten(2).transpose(1, 2).contiguous();
        if (embedded.size(1) != patch_mask.size(1)) {
            throw std::runtime_error(
                "MiniCPM-o patch mask length does not match patch convolution");
        }
        std::vector<int64_t> position_ids(
            static_cast<size_t>(embedded.size(0) * embedded.size(1)), 0);
        auto sizes = target_sizes.to(torch::kCPU, torch::kInt64).contiguous();
        auto mask_cpu = patch_mask.to(torch::kCPU, torch::kBool).contiguous();
        const auto * size_data = sizes.data_ptr<int64_t>();
        const auto * mask_data = mask_cpu.data_ptr<bool>();
        bool all_patches_active = true;
        for (int64_t batch = 0; batch < embedded.size(0); ++batch) {
            const int64_t height = size_data[2 * batch];
            const int64_t width = size_data[2 * batch + 1];
            if (height <= 0 || width <= 0 ||
                    height * width > embedded.size(1)) {
                throw std::runtime_error(
                    "MiniCPM-o target patch size is invalid");
            }
            int64_t active = 0;
            for (int64_t patch = 0; patch < embedded.size(1); ++patch) {
                if (!mask_data[batch * embedded.size(1) + patch]) {
                    all_patches_active = false;
                    continue;
                }
                if (active >= height * width) {
                    throw std::runtime_error(
                        "MiniCPM-o patch mask has too many active entries");
                }
                const int64_t row = active / width;
                const int64_t column = active % width;
                const int64_t bucket_column = std::min<int64_t>(
                    position_side - 1, column * position_side / width);
                const int64_t bucket_row = std::min<int64_t>(
                    position_side - 1, row * position_side / height);
                position_ids[static_cast<size_t>(
                    batch * embedded.size(1) + patch)] =
                    bucket_row * position_side + bucket_column;
                ++active;
            }
            if (active != height * width) {
                throw std::runtime_error(
                    "MiniCPM-o patch mask active count disagrees with target size");
            }
        }
        auto ids = torch::from_blob(
            position_ids.data(),
            std::vector<int64_t>{embedded.size(0), embedded.size(1)},
            torch::TensorOptions().dtype(torch::kInt64)).clone().to(torch::kCUDA);
        embedded = embedded + position_embedding.index_select(
            0, ids.reshape({-1})).reshape(embedded.sizes());
        c10::optional<torch::Tensor> attention_mask = c10::nullopt;
        if (!all_patches_active) {
            auto invalid = patch_mask.to(torch::kCUDA, torch::kBool)
                .logical_not().unsqueeze(1).unsqueeze(2);
            attention_mask = torch::zeros(
                {embedded.size(0), 1, embedded.size(1), embedded.size(1)},
                embedded.options().dtype(torch::kFloat32))
                .masked_fill(invalid, -std::numeric_limits<float>::infinity());
        }
        for (const auto & layer : layers) {
            embedded = layer.forward(embedded, attention_mask);
        }
        embedded = minicpmo45_layer_norm(
            embedded, post_norm_weight, post_norm_bias, 1e-6);
        return embedded;
    }
};

struct MiniCPMO45Resampler {
    torch::Tensor query;
    torch::Tensor position_embedding;
    MiniCPMO45Linear kv_projection;
    torch::Tensor q_norm_weight;
    torch::Tensor q_norm_bias;
    torch::Tensor kv_norm_weight;
    torch::Tensor kv_norm_bias;
    torch::Tensor post_norm_weight;
    torch::Tensor post_norm_bias;
    torch::Tensor in_projection_weight;
    torch::Tensor in_projection_bias;
    torch::Tensor out_projection_weight;
    torch::Tensor out_projection_bias;
    torch::Tensor final_projection;
    int64_t heads = 32;
    int64_t head_dim = 128;

    static MiniCPMO45Resampler load(const MfqFile & mfq) {
        MiniCPMO45Resampler result;
        if (!mfq.has_record(MINICPMO45_RESAMPLER_POS_EMBED_ASSET)) {
            throw std::runtime_error(
                "MiniCPM-o MFQ is missing the exact Resampler position asset; "
                "repack it with the current runtime assets tool");
        }
        auto position_blob = mfq.read_asset(
            MINICPMO45_RESAMPLER_POS_EMBED_ASSET);
        constexpr size_t position_header_bytes = 20;
        constexpr size_t position_value_bytes =
            static_cast<size_t>(70) * 70 * 4096 * sizeof(uint16_t);
        if (position_blob.size() !=
                position_header_bytes + position_value_bytes ||
            std::memcmp(position_blob.data(), "MFQRSPB1", 8) != 0) {
            throw std::runtime_error(
                "MiniCPM-o Resampler position asset has an invalid format");
        }
        size_t position_offset = 8;
        const uint32_t position_height = read_u32_from(
            position_blob, position_offset);
        const uint32_t position_width = read_u32_from(
            position_blob, position_offset);
        const uint32_t position_dim = read_u32_from(
            position_blob, position_offset);
        if (position_height != 70 || position_width != 70 ||
                position_dim != 4096 ||
                position_offset != position_header_bytes) {
            throw std::runtime_error(
                "MiniCPM-o Resampler position asset shape is not 70x70x4096");
        }
        {
            c10::cuda::CUDAGuard guard(active_weight_load_device());
            result.position_embedding = torch::from_blob(
                position_blob.data() + position_offset,
                std::vector<int64_t>{70, 70, 4096},
                torch::TensorOptions().dtype(torch::kBFloat16))
                .clone().to(torch::kCUDA).contiguous();
        }
        result.query = load_dense_native_gpu(mfq, "resampler.query");
        result.kv_projection = MiniCPMO45Linear::load(
            mfq, "resampler.kv_proj", false);
        result.q_norm_weight = load_dense_native_gpu(
            mfq, "resampler.ln_q.weight");
        result.q_norm_bias = load_dense_native_gpu(
            mfq, "resampler.ln_q.bias");
        result.kv_norm_weight = load_dense_native_gpu(
            mfq, "resampler.ln_kv.weight");
        result.kv_norm_bias = load_dense_native_gpu(
            mfq, "resampler.ln_kv.bias");
        result.post_norm_weight = load_dense_native_gpu(
            mfq, "resampler.ln_post.weight");
        result.post_norm_bias = load_dense_native_gpu(
            mfq, "resampler.ln_post.bias");
        result.in_projection_weight = load_dense_native_gpu(
            mfq, "resampler.attn.in_proj_weight");
        result.in_projection_bias = load_dense_native_gpu(
            mfq, "resampler.attn.in_proj_bias");
        result.out_projection_weight = load_dense_native_gpu(
            mfq, "resampler.attn.out_proj.weight");
        result.out_projection_bias = load_dense_native_gpu(
            mfq, "resampler.attn.out_proj.bias");
        result.final_projection = load_dense_native_gpu(
            mfq, "resampler.proj");
        if (result.query.sizes().vec() !=
                std::vector<int64_t>({64, 4096}) ||
            result.in_projection_weight.sizes().vec() !=
                std::vector<int64_t>({12288, 4096})) {
            throw std::runtime_error(
                "MiniCPM-o Resampler tensor shapes disagree with version 4.5");
        }
        return result;
    }

    torch::Tensor forward(
            torch::Tensor input,
            torch::Tensor target_sizes) const {
        if (input.dim() != 3 || input.size(2) != 1152 ||
                target_sizes.dim() != 2 || target_sizes.size(1) != 2 ||
                target_sizes.size(0) != input.size(0) ||
                target_sizes.device().is_cuda() ||
                input.scalar_type() != torch::kBFloat16) {
            throw std::runtime_error(
                "MiniCPM-o Resampler input geometry is invalid");
        }
        const int64_t batch = input.size(0);
        const int64_t length = input.size(1);
        auto sizes = target_sizes.to(torch::kCPU, torch::kInt64).contiguous();
        const auto * size_data = sizes.data_ptr<int64_t>();
        std::vector<torch::Tensor> positions;
        positions.reserve(static_cast<size_t>(batch));
        std::vector<uint8_t> key_padding(
            static_cast<size_t>(batch * length), uint8_t{1});
        for (int64_t index = 0; index < batch; ++index) {
            const int64_t height = size_data[2 * index];
            const int64_t width = size_data[2 * index + 1];
            const int64_t patches = height * width;
            if (height <= 0 || width <= 0 ||
                    height > position_embedding.size(0) ||
                    width > position_embedding.size(1) ||
                    patches > length) {
                throw std::runtime_error(
                    "MiniCPM-o Resampler target size is invalid");
            }
            auto position = position_embedding
                .narrow(0, 0, height)
                .narrow(1, 0, width)
                .reshape({patches, 4096});
            if (patches < length) {
                position = torch::cat({
                    position,
                    torch::zeros(
                        {length - patches, 4096}, position.options())}, 0);
            }
            positions.push_back(position);
            for (int64_t patch = 0; patch < patches; ++patch) {
                key_padding[static_cast<size_t>(index * length + patch)] = 0;
            }
        }
        auto position = torch::stack(positions, 0);
        auto key_padding_mask = torch::from_blob(
            key_padding.data(), std::vector<int64_t>{batch, length},
            torch::TensorOptions().dtype(torch::kUInt8)).clone()
            .to(torch::kCUDA, torch::kBool);
        auto kv = kv_projection.forward(input);
        kv = minicpmo45_layer_norm(
            kv, kv_norm_weight, kv_norm_bias, 1e-6);
        auto normalized_query = minicpmo45_layer_norm(
            query, q_norm_weight, q_norm_bias, 1e-6);
        auto repeated_query = normalized_query.unsqueeze(0)
            .expand({batch, normalized_query.size(0), normalized_query.size(1)})
            .contiguous();

        const auto dtype = input.scalar_type();
        auto project = [&](torch::Tensor value, int64_t offset) {
            auto weight = in_projection_weight.narrow(0, offset, 4096);
            auto bias = in_projection_bias.narrow(0, offset, 4096);
            return at::linear(
                value.to(weight.scalar_type()), weight, bias)
                .to(dtype).contiguous();
        };
        auto q = project(repeated_query, 0)
            .reshape({batch, 64, heads, head_dim})
            .transpose(1, 2).contiguous();
        auto k = project(kv + position, 4096)
            .reshape({batch, length, heads, head_dim})
            .transpose(1, 2).contiguous();
        auto v = project(kv, 8192)
            .reshape({batch, length, heads, head_dim})
            .transpose(1, 2).contiguous();
        auto mask = torch::zeros(
            {batch, 1, 1, length},
            torch::TensorOptions().device(torch::kCUDA).dtype(dtype))
            .masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -std::numeric_limits<float>::infinity())
            .expand({batch, heads, 1, length})
            .reshape({batch * heads, 1, length});
        auto q_flat = q.reshape({batch * heads, 64, head_dim});
        auto k_flat = k.reshape({batch * heads, length, head_dim});
        auto v_flat = v.reshape({batch * heads, length, head_dim});
        auto attention_weights = torch::baddbmm(
            mask,
            q_flat * std::sqrt(1.0 / static_cast<double>(head_dim)),
            k_flat.transpose(1, 2));
        attention_weights = torch::softmax(attention_weights, -1);
        auto attended = torch::bmm(attention_weights, v_flat)
            .reshape({batch, heads, 64, head_dim})
            .transpose(1, 2).reshape({batch, 64, 4096});
        attended = at::linear(
            attended.to(out_projection_weight.scalar_type()),
            out_projection_weight, out_projection_bias)
            .to(dtype).contiguous();
        attended = minicpmo45_layer_norm(
            attended, post_norm_weight, post_norm_bias, 1e-6);
        return torch::matmul(
            attended.to(final_projection.scalar_type()),
            final_projection).to(dtype).contiguous();
    }
};

struct MiniCPMO45WhisperAttention {
    MiniCPMO45Linear q;
    MiniCPMO45Linear k;
    MiniCPMO45Linear v;
    MiniCPMO45Linear output;
    int64_t heads = 16;
    int64_t head_dim = 64;

    static MiniCPMO45WhisperAttention load(
            const MfqFile & mfq,
            const std::string & prefix) {
        MiniCPMO45WhisperAttention result;
        result.q = MiniCPMO45Linear::load(mfq, prefix + ".q_proj");
        result.k = MiniCPMO45Linear::load(
            mfq, prefix + ".k_proj", false);
        result.v = MiniCPMO45Linear::load(mfq, prefix + ".v_proj");
        result.output = MiniCPMO45Linear::load(
            mfq, prefix + ".out_proj");
        return result;
    }

    torch::Tensor forward(
            torch::Tensor input,
            c10::optional<torch::Tensor> mask,
            torch::Tensor * key_cache = nullptr,
            torch::Tensor * value_cache = nullptr) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](torch::Tensor value) {
            return value.reshape({batch, tokens, heads, head_dim})
                .transpose(1, 2).contiguous();
        };
        auto query_projection = q.forward(input);
        auto key_projection = k.forward(input);
        auto value_projection = v.forward(input);
        auto query = reshape(query_projection);
        auto key = reshape(key_projection);
        auto value = reshape(value_projection);
        if (key_cache != nullptr && value_cache != nullptr) {
            if (key_cache->defined()) {
                key = torch::cat({*key_cache, key}, 2).contiguous();
                value = torch::cat({*value_cache, value}, 2).contiguous();
            }
            *key_cache = key;
            *value_cache = value;
        }
        std::optional<torch::Tensor> attention_mask = std::nullopt;
        if (mask.has_value()) {
            attention_mask = mask.value().to(query.scalar_type());
        }
        auto attended = at::scaled_dot_product_attention(
            query, key, value, attention_mask,
            0.0, false, std::nullopt, false).contiguous();
        attended = attended.transpose(1, 2).reshape(
            {batch, tokens, heads * head_dim});
        return output.forward(attended);
    }
};

struct MiniCPMO45WhisperLayer {
    MiniCPMO45WhisperAttention attention;
    torch::Tensor attention_norm_weight;
    torch::Tensor attention_norm_bias;
    torch::Tensor final_norm_weight;
    torch::Tensor final_norm_bias;
    MiniCPMO45Linear fc1;
    MiniCPMO45Linear fc2;
    torch::Tensor key_cache;
    torch::Tensor value_cache;

    static MiniCPMO45WhisperLayer load(
            const MfqFile & mfq,
            int index) {
        const std::string prefix =
            "apm.layers." + std::to_string(index);
        MiniCPMO45WhisperLayer result;
        result.attention = MiniCPMO45WhisperAttention::load(
            mfq, prefix + ".self_attn");
        result.attention_norm_weight = load_dense_native_gpu(
            mfq, prefix + ".self_attn_layer_norm.weight");
        result.attention_norm_bias = load_dense_native_gpu(
            mfq, prefix + ".self_attn_layer_norm.bias");
        result.final_norm_weight = load_dense_native_gpu(
            mfq, prefix + ".final_layer_norm.weight");
        result.final_norm_bias = load_dense_native_gpu(
            mfq, prefix + ".final_layer_norm.bias");
        result.fc1 = MiniCPMO45Linear::load(mfq, prefix + ".fc1");
        result.fc2 = MiniCPMO45Linear::load(mfq, prefix + ".fc2");
        return result;
    }

    void reset() {
        key_cache = torch::Tensor();
        value_cache = torch::Tensor();
    }

    torch::Tensor forward(
            torch::Tensor input,
            c10::optional<torch::Tensor> mask,
            bool use_cache) {
        auto normalized = minicpmo45_layer_norm(
            input, attention_norm_weight, attention_norm_bias, 1e-5);
        auto attended = use_cache
            ? attention.forward(
                normalized, mask, &key_cache, &value_cache)
            : attention.forward(normalized, mask);
        auto hidden = (input + attended).contiguous();
        normalized = minicpmo45_layer_norm(
            hidden, final_norm_weight, final_norm_bias, 1e-5);
        auto fc1_output = fc1.forward(normalized);
        auto activation = torch::gelu(fc1_output);
        auto feed_forward = fc2.forward(activation);
        return (hidden + feed_forward).contiguous();
    }
};

struct MiniCPMO45AudioEncoder {
    torch::Tensor conv1_weight;
    torch::Tensor conv1_bias;
    torch::Tensor conv2_weight;
    torch::Tensor conv2_bias;
    torch::Tensor position_embedding;
    std::vector<MiniCPMO45WhisperLayer> layers;
    torch::Tensor final_norm_weight;
    torch::Tensor final_norm_bias;
    MiniCPMO45Linear projector1;
    MiniCPMO45Linear projector2;

    static MiniCPMO45AudioEncoder load(const MfqFile & mfq) {
        MiniCPMO45AudioEncoder result;
        result.conv1_weight = load_dense_native_gpu(
            mfq, "apm.conv1.weight");
        result.conv1_bias = load_dense_native_gpu(
            mfq, "apm.conv1.bias");
        result.conv2_weight = load_dense_native_gpu(
            mfq, "apm.conv2.weight");
        result.conv2_bias = load_dense_native_gpu(
            mfq, "apm.conv2.bias");
        result.position_embedding = load_dense_native_gpu(
            mfq, "apm.embed_positions.weight");
        result.final_norm_weight = load_dense_native_gpu(
            mfq, "apm.layer_norm.weight");
        result.final_norm_bias = load_dense_native_gpu(
            mfq, "apm.layer_norm.bias");
        result.projector1 = MiniCPMO45Linear::load(
            mfq, "audio_projection_layer.linear1");
        result.projector2 = MiniCPMO45Linear::load(
            mfq, "audio_projection_layer.linear2");
        result.layers.reserve(24);
        for (int index = 0; index < 24; ++index) {
            result.layers.push_back(
                MiniCPMO45WhisperLayer::load(mfq, index));
        }
        if (result.conv1_weight.sizes().vec() !=
                std::vector<int64_t>({1024, 80, 3}) ||
            result.conv2_weight.sizes().vec() !=
                std::vector<int64_t>({1024, 1024, 3}) ||
            result.position_embedding.sizes().vec() !=
                std::vector<int64_t>({1500, 1024})) {
            throw std::runtime_error(
                "MiniCPM-o Whisper tensor shapes disagree with version 4.5");
        }
        return result;
    }

    void reset() {
        for (auto & layer : layers) layer.reset();
    }

    static std::vector<int64_t> pooled_lengths(
            torch::Tensor raw_lengths) {
        auto lengths = raw_lengths.to(torch::kCPU, torch::kInt64).contiguous();
        const auto * values = lengths.data_ptr<int64_t>();
        std::vector<int64_t> result(static_cast<size_t>(lengths.numel()));
        for (int64_t index = 0; index < lengths.numel(); ++index) {
            const int64_t after_conv = (values[index] - 1) / 2 + 1;
            result[static_cast<size_t>(index)] =
                (after_conv - 5) / 5 + 1;
            if (values[index] <= 0 || result[static_cast<size_t>(index)] <= 0) {
                throw std::runtime_error(
                    "MiniCPM-o audio length is too short");
            }
        }
        return result;
    }

    torch::Tensor forward(
            torch::Tensor features,
            torch::Tensor raw_lengths,
            bool use_cache = false) {
        if (features.dim() != 3 || features.size(1) != 80 ||
                raw_lengths.dim() != 1 ||
                raw_lengths.size(0) != features.size(0)) {
            throw std::runtime_error(
                "MiniCPM-o audio input geometry is invalid");
        }
        auto conv1 = torch::conv1d(
            features.to(conv1_weight.scalar_type()),
            conv1_weight, conv1_bias,
            std::vector<int64_t>{1},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        auto hidden = torch::gelu(conv1);
        auto conv2 = torch::conv1d(
            hidden, conv2_weight, conv2_bias,
            std::vector<int64_t>{2},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        hidden = torch::gelu(conv2);
        hidden = hidden.transpose(1, 2).contiguous();
        const int64_t tokens = hidden.size(1);
        int64_t past = 0;
        if (use_cache && !layers.empty() && layers.front().key_cache.defined()) {
            past = layers.front().key_cache.size(2);
        }
        if (past + tokens > position_embedding.size(0)) {
            throw std::runtime_error(
                "MiniCPM-o Whisper position cache exceeds 1500 frames");
        }
        hidden = hidden + position_embedding.narrow(0, past, tokens);

        // Preserve the official graph: its padding comparison uses the raw
        // Whisper feature lengths against the post-convolution sequence.
        auto length_tensor = raw_lengths.to(
            torch::kCUDA, torch::kInt64).contiguous();
        auto key_positions = torch::arange(
            past + tokens,
            torch::TensorOptions().device(torch::kCUDA)
                .dtype(torch::kInt64));
        auto valid_keys = key_positions.unsqueeze(0) <
            (length_tensor + past).unsqueeze(1);
        auto query_positions = torch::arange(
            past, past + tokens,
            torch::TensorOptions().device(torch::kCUDA)
                .dtype(torch::kInt64));
        // Official MiniCPM-o 4.5 uses audio_chunk_length=1.0. Whisper emits
        // 50 encoder frames per second, and every query can see its current
        // 50-frame chunk plus all preceding chunks.
        auto chunk_visible = key_positions.unsqueeze(0) <
            ((query_positions / 50 + 1) * 50).unsqueeze(1);
        auto visible = valid_keys.unsqueeze(1) & chunk_visible.unsqueeze(0);
        auto attention_mask = torch::zeros(
            {features.size(0), 1, tokens, past + tokens},
            torch::TensorOptions().device(torch::kCUDA)
                .dtype(torch::kFloat32))
            .masked_fill(
                visible.logical_not().unsqueeze(1),
                -std::numeric_limits<float>::infinity());
        for (auto & layer : layers) {
            hidden = layer.forward(hidden, attention_mask, use_cache);
        }
        hidden = minicpmo45_layer_norm(
            hidden, final_norm_weight, final_norm_bias, 1e-5);
        hidden = projector1.forward(hidden);
        hidden = torch::relu(hidden);
        hidden = projector2.forward(hidden);
        hidden = torch::avg_pool1d(
            hidden.transpose(1, 2),
            std::vector<int64_t>{5},
            std::vector<int64_t>{5})
            .transpose(1, 2).contiguous();
        return hidden;
    }
};

struct MiniCPMO45TtsDecoder {
    Config config;
    RopeCache rope;
    QuantLinear text_embedding;
    QuantLinear code_embedding;
    MiniCPMO45Linear semantic_projector1;
    MiniCPMO45Linear semantic_projector2;
    MiniCPMO45Linear speaker_projector1;
    MiniCPMO45Linear speaker_projector2;
    std::vector<std::unique_ptr<Block>> blocks;
    torch::Tensor output_norm;
    torch::Tensor code_head;
    int64_t cache_position = 0;

    static Config make_config() {
        Config result;
        result.model_type = "minicpmtts";
        result.hf_model_prefix = "tts.model.";
        result.hidden_size = 768;
        result.intermediate_size = 3072;
        result.num_hidden_layers = 20;
        result.num_attention_heads = 12;
        result.num_key_value_heads = 12;
        result.head_dim = 64;
        result.rotary_dim = 64;
        result.max_position_embeddings = 4096;
        result.rope_base = 10000.0;
        result.rms_norm_eps = 1e-6;
        result.norm_weight_offset = 0.0;
        result.layer_types.assign(20, "full_attention");
        return result;
    }

    static MiniCPMO45TtsDecoder load(const MfqFile & mfq) {
        MiniCPMO45TtsDecoder result;
        result.config = make_config();
        result.rope = RopeCache(result.config);
        result.text_embedding = load_quant_linear(
            mfq, "tts.emb_text.weight");
        result.code_embedding = load_quant_linear(
            mfq, "tts.emb_code.0.weight");
        result.semantic_projector1 = MiniCPMO45Linear::load(
            mfq, "tts.projector_semantic.linear1");
        result.semantic_projector2 = MiniCPMO45Linear::load(
            mfq, "tts.projector_semantic.linear2");
        result.speaker_projector1 = MiniCPMO45Linear::load(
            mfq, "tts.projector_spk.linear1");
        result.speaker_projector2 = MiniCPMO45Linear::load(
            mfq, "tts.projector_spk.linear2");
        result.output_norm = load_dense_native_gpu(
            mfq, "tts.model.norm.weight");
        auto head_g = load_dense_native_gpu(
            mfq, "tts.head_code.0.parametrizations.weight.original0");
        auto head_v = load_dense_native_gpu(
            mfq, "tts.head_code.0.parametrizations.weight.original1");
        if (head_g.sizes().vec() != std::vector<int64_t>({6562, 1}) ||
                head_v.sizes().vec() != std::vector<int64_t>({6562, 768}) ||
                result.text_embedding.out() != 152064 ||
                result.text_embedding.neuron_len() != 768 ||
                result.code_embedding.out() != 6562 ||
                result.code_embedding.neuron_len() != 768) {
            throw std::runtime_error(
                "MiniCPM-o TTS tensor shapes disagree with version 4.5");
        }
        auto head_v_f32 = head_v.to(torch::kFloat32);
        auto row_norm = torch::sqrt(
            torch::sum(head_v_f32.square(), -1, true))
            .clamp_min(1e-12);
        result.code_head = (
            head_g.to(torch::kFloat32) * head_v_f32 /
            row_norm).to(head_v.scalar_type()).contiguous();
        result.blocks.reserve(20);
        for (int index = 0; index < 20; ++index) {
            auto block = load_block(
                mfq, result.config, index, "full_attention", false);
            block->cuda_device = g_layer_placement.primary_device();
            result.blocks.push_back(std::move(block));
        }
        return result;
    }

    void reset(int64_t batch) {
        cache_position = 0;
        for (auto & block : blocks) block->reset(batch);
    }

    torch::Tensor semantic_projection(torch::Tensor hidden) const {
        auto projected = semantic_projector1.forward(hidden);
        projected = torch::relu(projected);
        projected = semantic_projector2.forward(projected);
        auto norm = torch::sqrt(
            torch::sum(projected.to(torch::kFloat32).square(), -1, true))
            .clamp_min(1e-12);
        return (projected / norm.to(projected.scalar_type())).contiguous();
    }

    torch::Tensor speaker_projection(torch::Tensor hidden) const {
        return speaker_projector2.forward(
            torch::relu(speaker_projector1.forward(hidden)));
    }

    torch::Tensor condition(
            torch::Tensor text_ids,
            torch::Tensor language_hidden) const {
        if (text_ids.dim() == 1) text_ids = text_ids.unsqueeze(0);
        if (language_hidden.dim() == 2) {
            language_hidden = language_hidden.unsqueeze(0);
        }
        if (text_ids.dim() != 2 || language_hidden.dim() != 3 ||
                text_ids.size(0) != 1 || language_hidden.size(0) != 1 ||
                text_ids.size(1) != language_hidden.size(1) ||
                language_hidden.size(2) != 4096) {
            throw std::runtime_error(
                "MiniCPM-o TTS condition expects one aligned text/hidden span");
        }
        auto text = minicpmo45_embedding(text_embedding, text_ids);
        auto semantic = semantic_projection(language_hidden)
            .to(text.scalar_type());
        auto merged = (text + semantic).contiguous();
        auto suffix_ids = torch::tensor(
            std::vector<int64_t>{151692, 151687},
            torch::TensorOptions().dtype(torch::kInt64)
                .device(torch::kCUDA)).reshape({1, 2});
        auto suffix = minicpmo45_embedding(
            text_embedding, suffix_ids, text.scalar_type());
        return torch::cat({merged, suffix}, 1).contiguous();
    }

    torch::Tensor hidden_forward(torch::Tensor input_embeddings) {
        if (input_embeddings.dim() != 3 ||
                input_embeddings.size(2) != config.hidden_size) {
            throw std::runtime_error(
                "MiniCPM-o TTS embeddings must be [batch,tokens,768]");
        }
        const int64_t batch = input_embeddings.size(0);
        const int64_t tokens = input_embeddings.size(1);
        if (cache_position == 0) reset(batch);
        if (cache_position + tokens > config.max_position_embeddings) {
            throw std::runtime_error(
                "MiniCPM-o TTS context exceeds 4096 tokens");
        }
        auto positions = torch::arange(
            cache_position, cache_position + tokens,
            torch::TensorOptions().device(torch::kCUDA)
                .dtype(torch::kInt64));
        c10::optional<torch::Tensor> sequence_length = c10::nullopt;
        if (cache_position > 0) {
            sequence_length = torch::full(
                {batch}, cache_position + tokens,
                torch::TensorOptions().device(torch::kCUDA)
                    .dtype(torch::kInt64));
        }
        auto hidden = input_embeddings.contiguous();
        for (auto & block : blocks) {
            hidden = block->forward(
                hidden, positions, cache_position,
                sequence_length, config, rope);
        }
        cache_position += tokens;
        auto flat = hidden.reshape(
            {batch * tokens, config.hidden_size});
        auto normalized_f32 = flat.to(torch::kFloat32);
        normalized_f32 = normalized_f32 * torch::rsqrt(
            torch::mean(normalized_f32.square(), -1, true) +
            config.rms_norm_eps);
        return (
            normalized_f32.to(flat.scalar_type()) *
            output_norm.to(flat.scalar_type()))
            .reshape({batch, tokens, config.hidden_size})
            .contiguous();
    }

    torch::Tensor logits(torch::Tensor hidden) const {
        return torch::matmul(
            hidden.to(code_head.scalar_type()),
            code_head.transpose(0, 1));
    }

    torch::Tensor generate_greedy(
            torch::Tensor condition_embeddings,
            int64_t steps,
            int64_t eos_token = 6561,
            int64_t minimum_steps = 0,
            std::vector<torch::Tensor> * logits_trace = nullptr) {
        if (steps <= 0 || minimum_steps < 0 || minimum_steps > steps ||
                eos_token < 0 || eos_token >= 6562) {
            throw std::runtime_error(
                "MiniCPM-o TTS generation limits are invalid");
        }
        reset(condition_embeddings.size(0));
        std::vector<torch::Tensor> generated;
        generated.reserve(static_cast<size_t>(steps));
        auto current = condition_embeddings;
        for (int64_t step = 0; step < steps; ++step) {
            auto hidden = hidden_forward(current);
            auto step_logits = logits(
                hidden.index({Slice(), -1, Slice()}))
                .to(torch::kFloat32);
            if (step < minimum_steps) {
                step_logits.index_put_({Slice(), eos_token},
                    -std::numeric_limits<float>::infinity());
            }
            if (logits_trace != nullptr) {
                logits_trace->push_back(step_logits.clone());
            }
            auto token = torch::argmax(step_logits, -1)
                .to(torch::kInt64);
            generated.push_back(token);
            if (token.eq(eos_token).all().item<bool>()) break;
            current = minicpmo45_embedding(
                code_embedding, token.unsqueeze(1));
        }
        return torch::stack(generated, 1).unsqueeze(-1).contiguous();
    }
};

struct MiniCPMO45Bounds {
    int64_t batch = 0;
    int64_t source = 0;
    int64_t begin = 0;
    int64_t end = 0;
};

static std::vector<MiniCPMO45Bounds> minicpmo45_parse_bounds(
        torch::Tensor bounds,
        const char * label) {
    if (!bounds.defined() || bounds.numel() == 0) return {};
    bounds = bounds.to(torch::kCPU, torch::kInt64).contiguous();
    if (bounds.dim() != 2 || bounds.size(1) != 4) {
        throw std::runtime_error(
            std::string("MiniCPM-o ") + label +
            " bounds must have shape [count,4]");
    }
    const auto * values = bounds.data_ptr<int64_t>();
    std::vector<MiniCPMO45Bounds> result;
    result.reserve(static_cast<size_t>(bounds.size(0)));
    for (int64_t row = 0; row < bounds.size(0); ++row) {
        MiniCPMO45Bounds bound;
        bound.batch = values[row * 4];
        bound.source = values[row * 4 + 1];
        bound.begin = values[row * 4 + 2];
        bound.end = values[row * 4 + 3];
        if (bound.batch < 0 || bound.source < 0 ||
                bound.begin < 0 || bound.end <= bound.begin) {
            throw std::runtime_error(
                std::string("MiniCPM-o ") + label +
                " bounds contain an invalid row");
        }
        result.push_back(bound);
    }
    return result;
}

struct MiniCPMO45ForwardResult {
    torch::Tensor vision_states;
    torch::Tensor image_embeddings;
    torch::Tensor audio_embeddings;
    torch::Tensor input_embeddings;
    torch::Tensor hidden_states;
    torch::Tensor logits;
};

struct MiniCPMO45Runtime {
    Model language;
    MiniCPMO45VisionEncoder vision;
    MiniCPMO45Resampler resampler;
    MiniCPMO45AudioEncoder audio;
    MiniCPMO45TtsDecoder tts;

    static MiniCPMO45Runtime load(
            const std::string & mfq_path,
            const std::string & config_path,
            int64_t context_size) {
        MiniCPMO45Runtime result;
        result.language = load_model(
            mfq_path, config_path, context_size);
        if (!result.language.c.is_minicpmo45()) {
            throw std::runtime_error(
                "native composite graph requires MiniCPM-o 4.5");
        }
        g_layer_placement.load_device =
            g_layer_placement.primary_device();
        c10::cuda::CUDAGuard guard(
            g_layer_placement.primary_device());
        MfqFile mfq(mfq_path);
        result.vision = MiniCPMO45VisionEncoder::load(mfq);
        result.resampler = MiniCPMO45Resampler::load(mfq);
        result.audio = MiniCPMO45AudioEncoder::load(mfq);
        result.tts = MiniCPMO45TtsDecoder::load(mfq);
        return result;
    }

    MiniCPMO45ForwardResult forward(
            torch::Tensor input_ids,
            torch::Tensor pixels,
            torch::Tensor patch_mask,
            torch::Tensor target_sizes,
            torch::Tensor image_bounds,
            torch::Tensor audio_features,
            torch::Tensor audio_lengths,
            torch::Tensor audio_bounds) {
        if (input_ids.dim() == 1) input_ids = input_ids.unsqueeze(0);
        if (input_ids.dim() != 2) {
            throw std::runtime_error(
                "MiniCPM-o input_ids must have shape [batch,tokens]");
        }
        input_ids = input_ids.to(torch::kCUDA, torch::kInt64).contiguous();
        MiniCPMO45ForwardResult result;
        result.input_embeddings = language.embed_forward(input_ids);
        const auto images = minicpmo45_parse_bounds(
            image_bounds, "image");
        if (!images.empty()) {
            if (!pixels.defined() || !patch_mask.defined() ||
                    !target_sizes.defined()) {
                throw std::runtime_error(
                    "MiniCPM-o image bounds require image tensors");
            }
            result.vision_states = vision.forward(
                pixels.to(torch::kCUDA), patch_mask, target_sizes);
            result.image_embeddings = resampler.forward(
                result.vision_states, target_sizes);
            for (const auto & bound : images) {
                if (bound.batch >= input_ids.size(0) ||
                        bound.source >= result.image_embeddings.size(0) ||
                        bound.end > input_ids.size(1) ||
                        bound.end - bound.begin !=
                            result.image_embeddings.size(1)) {
                    throw std::runtime_error(
                        "MiniCPM-o image bound does not match 64 resampler queries");
                }
                result.input_embeddings.index({
                    bound.batch, Slice(bound.begin, bound.end), Slice()})
                    .copy_(result.image_embeddings.index({bound.source})
                        .to(result.input_embeddings.scalar_type()));
            }
        }
        const auto audios = minicpmo45_parse_bounds(
            audio_bounds, "audio");
        if (!audios.empty()) {
            if (!audio_features.defined() || !audio_lengths.defined()) {
                throw std::runtime_error(
                    "MiniCPM-o audio bounds require audio tensors");
            }
            audio.reset();
            result.audio_embeddings = audio.forward(
                audio_features.to(torch::kCUDA), audio_lengths, false);
            const auto valid_lengths =
                MiniCPMO45AudioEncoder::pooled_lengths(audio_lengths);
            for (const auto & bound : audios) {
                if (bound.batch >= input_ids.size(0) ||
                        bound.source >= result.audio_embeddings.size(0) ||
                        bound.source >=
                            static_cast<int64_t>(valid_lengths.size()) ||
                        bound.end > input_ids.size(1) ||
                        bound.end - bound.begin !=
                            valid_lengths[static_cast<size_t>(bound.source)]) {
                    throw std::runtime_error(
                        "MiniCPM-o audio bound does not match pooled audio length");
                }
                result.input_embeddings.index({
                    bound.batch, Slice(bound.begin, bound.end), Slice()})
                    .copy_(result.audio_embeddings.index({
                        bound.source,
                        Slice(0, bound.end - bound.begin), Slice()})
                        .to(result.input_embeddings.scalar_type()));
            }
        }
        language.reset(input_ids.size(0));
        result.hidden_states = language.hidden_forward_inputs(
            input_ids, result.input_embeddings,
            c10::nullopt, c10::nullopt);
        result.logits = language.logits_from_hidden(result.hidden_states);
        return result;
    }
};

static torch::Tensor minicpmo45_load_tensor(
        const std::string & path,
        bool required) {
    if (!std::filesystem::is_regular_file(path)) {
        if (required) {
            throw std::runtime_error(
                "MiniCPM-o tensor file is missing: " + path);
        }
        return torch::Tensor();
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "failed to open MiniCPM-o tensor file: " + path);
    }
    const auto end = input.tellg();
    if (end <= 0) {
        throw std::runtime_error(
            "MiniCPM-o tensor file is empty: " + path);
    }
    std::vector<char> bytes(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw std::runtime_error(
            "failed to read MiniCPM-o tensor file: " + path);
    }
    auto value = torch::pickle_load(bytes);
    if (!value.isTensor()) {
        throw std::runtime_error(
            "MiniCPM-o input is not a tensor: " + path);
    }
    return value.toTensor();
}

static void minicpmo45_save_tensor(
        const torch::Tensor & value,
        const std::string & path) {
    if (!value.defined()) return;
    minicpmo45_write_pickle_tensor(value, path);
}

static int run_minicpmo45_composite(
        const std::string & mfq_path,
        const std::string & config_path,
        const std::string & input_prefix,
        const std::string & output_prefix,
        int64_t context_size,
        int64_t tts_steps) {
    torch::NoGradGuard no_grad;
    g_profiler.enabled = false;
    if (input_prefix.empty() || output_prefix.empty()) {
        throw std::runtime_error(
            "MiniCPM-o composite graph requires input and output prefixes");
    }
    auto runtime = MiniCPMO45Runtime::load(
        mfq_path, config_path, context_size);
    const auto input_ids = minicpmo45_load_tensor(
        input_prefix + ".input_ids.pt", true);
    const auto pixels = minicpmo45_load_tensor(
        input_prefix + ".pixel_values.pt", false);
    const auto patch_mask = minicpmo45_load_tensor(
        input_prefix + ".patch_mask.pt", false);
    const auto target_sizes = minicpmo45_load_tensor(
        input_prefix + ".target_sizes.pt", false);
    const auto image_bounds = minicpmo45_load_tensor(
        input_prefix + ".image_bounds.pt", false);
    const auto audio_features = minicpmo45_load_tensor(
        input_prefix + ".audio_features.pt", false);
    const auto audio_lengths = minicpmo45_load_tensor(
        input_prefix + ".audio_lengths.pt", false);
    const auto audio_bounds = minicpmo45_load_tensor(
        input_prefix + ".audio_bounds.pt", false);
    auto result = runtime.forward(
        input_ids, pixels, patch_mask, target_sizes, image_bounds,
        audio_features, audio_lengths, audio_bounds);
    minicpmo45_save_tensor(
        result.vision_states, output_prefix + ".vision_states.pt");
    minicpmo45_save_tensor(
        result.image_embeddings, output_prefix + ".image_embeddings.pt");
    minicpmo45_save_tensor(
        result.audio_embeddings, output_prefix + ".audio_embeddings.pt");
    minicpmo45_save_tensor(
        result.input_embeddings, output_prefix + ".input_embeddings.pt");
    minicpmo45_save_tensor(
        result.hidden_states, output_prefix + ".hidden_states.pt");
    minicpmo45_save_tensor(
        result.logits, output_prefix + ".logits.pt");
    if (tts_steps > 0) {
        torch::Tensor condition = minicpmo45_load_tensor(
            input_prefix + ".tts_inputs_embeds.pt", false);
        if (!condition.defined()) {
            auto bound = minicpmo45_load_tensor(
                input_prefix + ".tts_bound.pt", true)
                .to(torch::kCPU, torch::kInt64).contiguous();
            if (bound.dim() != 1 || bound.numel() != 2) {
                throw std::runtime_error(
                    "MiniCPM-o tts_bound must contain [begin,end]");
            }
            const int64_t begin = bound[0].item<int64_t>();
            const int64_t end = bound[1].item<int64_t>();
            if (begin < 0 || end <= begin ||
                    end > input_ids.size(-1) || input_ids.size(0) != 1) {
                throw std::runtime_error(
                    "MiniCPM-o TTS text span is invalid");
            }
            condition = runtime.tts.condition(
                input_ids.index({0, Slice(begin, end)}),
                result.hidden_states.index({0, Slice(begin, end), Slice()}));
        } else {
            condition = condition.to(torch::kCUDA).contiguous();
        }
        std::vector<torch::Tensor> tts_logits;
        const auto codes = runtime.tts.generate_greedy(
            condition, tts_steps, 6561, 0, &tts_logits);
        minicpmo45_save_tensor(
            condition, output_prefix + ".tts_inputs_embeds.pt");
        minicpmo45_save_tensor(
            codes, output_prefix + ".tts_codes.pt");
        if (!tts_logits.empty()) {
            minicpmo45_save_tensor(
                torch::stack(tts_logits, 1),
                output_prefix + ".tts_logits.pt");
        }
    }
    std::cout << "minicpmo45_composite=ok"
              << " batch=" << input_ids.size(0)
              << " tokens=" << input_ids.size(1)
              << " images=" << (image_bounds.defined() ? image_bounds.size(0) : 0)
              << " audios=" << (audio_bounds.defined() ? audio_bounds.size(0) : 0)
              << " tts_steps=" << tts_steps << "\n";
    return 0;
}
