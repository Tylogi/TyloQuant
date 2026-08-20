// MiniCPM-o 4.5 native CUDA composite graph.
//
// This file is included by mfq_decode.cpp after the generic causal-LM loader.
// It deliberately reuses QuantLinear and FullBlock so compact matrix weights
// stay on the same production CUDA kernels as the text-only runtime.

static constexpr const char * MINICPMO45_RESAMPLER_POS_EMBED_ASSET =
    "__mfq_asset__/minicpmo45-resampler-pos-embed-v1.bf16";

struct MiniCPMO45Linear {
    QuantLinear weight;
    mfq_tensor_backend::Tensor bias;

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

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor input) const {
        const auto output_dtype = input.scalar_type();
        if (weight.is_dense()) {
            const auto dense_dtype = weight.dense.scalar_type();
            std::optional<mfq_tensor_backend::Tensor> dense_bias = std::nullopt;
            if (bias.defined()) {
                dense_bias = bias.to(dense_dtype);
            }
            return mfq_linear(
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

static mfq_tensor_backend::Tensor load_dense_native_gpu(
        const MfqFile & mfq,
        const std::string & name) {
    MfqCudaGuard guard(active_weight_load_device());
    const auto & record = mfq.record(name);
    const auto blob = mfq.read_blob(name);
    size_t offset = 0;
    const uint32_t dimensions = read_u32_from(blob, offset);
    std::vector<int64_t> shape(dimensions);
    for (uint32_t index = 0; index < dimensions; ++index) {
        shape[index] = read_i64_from(blob, offset);
    }
    mfq_tensor_backend::Tensor value;
    if (record.dtype == "BF16") {
        value = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16)).clone();
    } else if (record.dtype == "F16") {
        value = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone();
    } else if (record.dtype == "F32") {
        value = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone();
    } else if (record.dtype == "I64") {
        value = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone();
    } else if (record.dtype == "I32") {
        value = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32)).clone();
    } else {
        throw std::runtime_error(
            "MiniCPM-o dense tensor has unsupported dtype: " +
            name + " dtype=" + record.dtype);
    }
    return value.to(mfq_tensor_backend::kCUDA).contiguous();
}

static void minicpmo45_write_pickle_tensor(
        const mfq_tensor_backend::Tensor & value,
        const std::string & path) {
    auto bytes = mfq_tensor_backend::pickle_save(
        value.detach().to(mfq_tensor_backend::kCPU).contiguous());
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

static mfq_tensor_backend::Tensor minicpmo45_embedding(
        const QuantLinear & embedding,
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::ScalarType output_dtype = mfq_tensor_backend::kBFloat16) {
    ids = ids.contiguous().to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64);
    auto output = quant_embedding_lookup(embedding, ids);
    return output.to(output_dtype).contiguous();
}

static mfq_tensor_backend::Tensor minicpmo45_layer_norm(
        mfq_tensor_backend::Tensor input,
        const mfq_tensor_backend::Tensor & weight,
        const mfq_tensor_backend::Tensor & bias,
        double eps) {
    const auto dtype = input.scalar_type();
    auto output = mfq_tensor_backend::layer_norm(
        input, {input.size(-1)},
        weight.to(dtype), bias.to(dtype), eps);
    return output.to(dtype).contiguous();
}

static mfq_tensor_backend::Tensor minicpmo45_attention(
        mfq_tensor_backend::Tensor query,
        mfq_tensor_backend::Tensor key,
        mfq_tensor_backend::Tensor value,
    MfqOptional<mfq_tensor_backend::Tensor> additive_mask,
        double scale) {
    auto scores = mfq_tensor_backend::matmul(
        query,
        key.transpose(-2, -1)) * scale;
    if (additive_mask.has_value()) {
        scores = scores + additive_mask.value().to(scores.scalar_type());
    }
    auto probabilities = mfq_tensor_backend::softmax(
        scores, -1, mfq_tensor_backend::kFloat32).to(value.scalar_type());
    return mfq_tensor_backend::matmul(probabilities, value).contiguous();
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

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            MfqOptional<mfq_tensor_backend::Tensor> mask) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](mfq_tensor_backend::Tensor value) {
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
    mfq_tensor_backend::Tensor norm1_weight;
    mfq_tensor_backend::Tensor norm1_bias;
    mfq_tensor_backend::Tensor norm2_weight;
    mfq_tensor_backend::Tensor norm2_bias;
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

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            MfqOptional<mfq_tensor_backend::Tensor> mask) const {
        auto normalized = minicpmo45_layer_norm(
            input, norm1_weight, norm1_bias, 1e-6);
        auto hidden = input + attention.forward(normalized, mask);
        normalized = minicpmo45_layer_norm(
            hidden, norm2_weight, norm2_bias, 1e-6);
        auto mlp = fc2.forward(mfq_tensor_backend::gelu(fc1.forward(normalized), "tanh"));
        return (hidden + mlp).contiguous();
    }
};

struct MiniCPMO45VisionEncoder {
    mfq_tensor_backend::Tensor patch_weight;
    mfq_tensor_backend::Tensor patch_bias;
    mfq_tensor_backend::Tensor position_embedding;
    std::vector<MiniCPMO45VisionLayer> layers;
    mfq_tensor_backend::Tensor post_norm_weight;
    mfq_tensor_backend::Tensor post_norm_bias;
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

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor pixels,
            mfq_tensor_backend::Tensor patch_mask,
            mfq_tensor_backend::Tensor target_sizes) const {
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
        auto embedded = mfq_tensor_backend::conv2d(
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
        auto sizes = target_sizes.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        auto mask_cpu = patch_mask.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kBool).contiguous();
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
        auto ids = mfq_tensor_backend::from_blob(
            position_ids.data(),
            std::vector<int64_t>{embedded.size(0), embedded.size(1)},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone().to(mfq_tensor_backend::kCUDA);
        embedded = embedded + position_embedding.index_select(
            0, ids.reshape({-1})).reshape(embedded.sizes());
        MfqOptional<mfq_tensor_backend::Tensor> attention_mask = mfq_nullopt;
        if (!all_patches_active) {
            auto invalid = patch_mask.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kBool)
                .logical_not().unsqueeze(1).unsqueeze(2);
            attention_mask = mfq_tensor_backend::zeros(
                {embedded.size(0), 1, embedded.size(1), embedded.size(1)},
                embedded.options().dtype(mfq_tensor_backend::kFloat32))
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
    mfq_tensor_backend::Tensor query;
    mfq_tensor_backend::Tensor position_embedding;
    MiniCPMO45Linear kv_projection;
    mfq_tensor_backend::Tensor q_norm_weight;
    mfq_tensor_backend::Tensor q_norm_bias;
    mfq_tensor_backend::Tensor kv_norm_weight;
    mfq_tensor_backend::Tensor kv_norm_bias;
    mfq_tensor_backend::Tensor post_norm_weight;
    mfq_tensor_backend::Tensor post_norm_bias;
    mfq_tensor_backend::Tensor in_projection_weight;
    mfq_tensor_backend::Tensor in_projection_bias;
    mfq_tensor_backend::Tensor out_projection_weight;
    mfq_tensor_backend::Tensor out_projection_bias;
    mfq_tensor_backend::Tensor final_projection;
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
            MfqCudaGuard guard(active_weight_load_device());
            result.position_embedding = mfq_tensor_backend::from_blob(
                position_blob.data() + position_offset,
                std::vector<int64_t>{70, 70, 4096},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16))
                .clone().to(mfq_tensor_backend::kCUDA).contiguous();
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

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            mfq_tensor_backend::Tensor target_sizes) const {
        if (input.dim() != 3 || input.size(2) != 1152 ||
                target_sizes.dim() != 2 || target_sizes.size(1) != 2 ||
                target_sizes.size(0) != input.size(0) ||
                target_sizes.device().is_cuda() ||
                input.scalar_type() != mfq_tensor_backend::kBFloat16) {
            throw std::runtime_error(
                "MiniCPM-o Resampler input geometry is invalid");
        }
        const int64_t batch = input.size(0);
        const int64_t length = input.size(1);
        auto sizes = target_sizes.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        const auto * size_data = sizes.data_ptr<int64_t>();
        std::vector<mfq_tensor_backend::Tensor> positions;
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
                position = mfq_tensor_backend::cat({
                    position,
                    mfq_tensor_backend::zeros(
                        {length - patches, 4096}, position.options())}, 0);
            }
            positions.push_back(position);
            for (int64_t patch = 0; patch < patches; ++patch) {
                key_padding[static_cast<size_t>(index * length + patch)] = 0;
            }
        }
        auto position = mfq_tensor_backend::stack(positions, 0);
        auto key_padding_mask = mfq_tensor_backend::from_blob(
            key_padding.data(), std::vector<int64_t>{batch, length},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8)).clone()
            .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kBool);
        auto kv = kv_projection.forward(input);
        kv = minicpmo45_layer_norm(
            kv, kv_norm_weight, kv_norm_bias, 1e-6);
        auto normalized_query = minicpmo45_layer_norm(
            query, q_norm_weight, q_norm_bias, 1e-6);
        auto repeated_query = normalized_query.unsqueeze(0)
            .expand({batch, normalized_query.size(0), normalized_query.size(1)})
            .contiguous();

        const auto dtype = input.scalar_type();
        auto project = [&](mfq_tensor_backend::Tensor value, int64_t offset) {
            auto weight = in_projection_weight.narrow(0, offset, 4096);
            auto bias = in_projection_bias.narrow(0, offset, 4096);
            return mfq_linear(
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
        auto mask = mfq_tensor_backend::zeros(
            {batch, 1, 1, length},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(dtype))
            .masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -std::numeric_limits<float>::infinity())
            .expand({batch, heads, 1, length})
            .reshape({batch * heads, 1, length});
        auto q_flat = q.reshape({batch * heads, 64, head_dim});
        auto k_flat = k.reshape({batch * heads, length, head_dim});
        auto v_flat = v.reshape({batch * heads, length, head_dim});
        auto attention_weights = mfq_tensor_backend::baddbmm(
            mask,
            q_flat * std::sqrt(1.0 / static_cast<double>(head_dim)),
            k_flat.transpose(1, 2));
        attention_weights = mfq_tensor_backend::softmax(attention_weights, -1);
        auto attended = mfq_tensor_backend::bmm(attention_weights, v_flat)
            .reshape({batch, heads, 64, head_dim})
            .transpose(1, 2).reshape({batch, 64, 4096});
        attended = mfq_linear(
            attended.to(out_projection_weight.scalar_type()),
            out_projection_weight, out_projection_bias)
            .to(dtype).contiguous();
        attended = minicpmo45_layer_norm(
            attended, post_norm_weight, post_norm_bias, 1e-6);
        return mfq_tensor_backend::matmul(
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

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            MfqOptional<mfq_tensor_backend::Tensor> mask,
            mfq_tensor_backend::Tensor * key_cache = nullptr,
            mfq_tensor_backend::Tensor * value_cache = nullptr) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](mfq_tensor_backend::Tensor value) {
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
                key = mfq_tensor_backend::cat({*key_cache, key}, 2).contiguous();
                value = mfq_tensor_backend::cat({*value_cache, value}, 2).contiguous();
            }
            *key_cache = key;
            *value_cache = value;
        }
        std::optional<mfq_tensor_backend::Tensor> attention_mask = std::nullopt;
        if (mask.has_value()) {
            attention_mask = mask.value().to(query.scalar_type());
        }
        auto attended = mfq_scaled_dot_product_attention(
            query, key, value, attention_mask,
            0.0, false, std::nullopt, false).contiguous();
        attended = attended.transpose(1, 2).reshape(
            {batch, tokens, heads * head_dim});
        return output.forward(attended);
    }
};

struct MiniCPMO45WhisperLayer {
    MiniCPMO45WhisperAttention attention;
    mfq_tensor_backend::Tensor attention_norm_weight;
    mfq_tensor_backend::Tensor attention_norm_bias;
    mfq_tensor_backend::Tensor final_norm_weight;
    mfq_tensor_backend::Tensor final_norm_bias;
    MiniCPMO45Linear fc1;
    MiniCPMO45Linear fc2;
    mfq_tensor_backend::Tensor key_cache;
    mfq_tensor_backend::Tensor value_cache;

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
        key_cache = mfq_tensor_backend::Tensor();
        value_cache = mfq_tensor_backend::Tensor();
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            MfqOptional<mfq_tensor_backend::Tensor> mask,
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
        auto activation = mfq_tensor_backend::gelu(fc1_output);
        auto feed_forward = fc2.forward(activation);
        return (hidden + feed_forward).contiguous();
    }
};

struct MiniCPMO45AudioEncoder {
    mfq_tensor_backend::Tensor conv1_weight;
    mfq_tensor_backend::Tensor conv1_bias;
    mfq_tensor_backend::Tensor conv2_weight;
    mfq_tensor_backend::Tensor conv2_bias;
    mfq_tensor_backend::Tensor position_embedding;
    std::vector<MiniCPMO45WhisperLayer> layers;
    mfq_tensor_backend::Tensor final_norm_weight;
    mfq_tensor_backend::Tensor final_norm_bias;
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

    int64_t cache_length() const {
        if (layers.empty() || !layers.front().key_cache.defined()) return 0;
        return layers.front().key_cache.size(2);
    }

    mfq_tensor_backend::Tensor forward_streaming(
            mfq_tensor_backend::Tensor features,
            int64_t prefix_extra_frames,
            int64_t suffix_extra_frames) {
        if (features.dim() != 3 || features.size(0) != 1 ||
                features.size(1) != 80 || prefix_extra_frames < 0 ||
                suffix_extra_frames < 0) {
            throw std::runtime_error(
                "MiniCPM-o streaming audio expects [1,80,frames] and "
                "non-negative extra-frame counts");
        }
        const int64_t conv_tokens_before_crop =
            (features.size(2) - 1) / 2 + 1;
        if (cache_length() + conv_tokens_before_crop >=
                position_embedding.size(0)) {
            reset();
        }

        auto conv1 = mfq_tensor_backend::conv1d(
            features.to(conv1_weight.scalar_type()),
            conv1_weight, conv1_bias,
            std::vector<int64_t>{1},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        auto hidden = mfq_tensor_backend::gelu(conv1);
        auto conv2 = mfq_tensor_backend::conv1d(
            hidden, conv2_weight, conv2_bias,
            std::vector<int64_t>{2},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        hidden = mfq_tensor_backend::gelu(conv2);

        const int64_t prefix_to_remove =
            prefix_extra_frames > 0 ? (prefix_extra_frames + 1) / 2 : 0;
        const int64_t suffix_to_remove =
            suffix_extra_frames > 0 ? (suffix_extra_frames + 1) / 2 : 0;
        if (prefix_to_remove + suffix_to_remove >= hidden.size(2)) {
            throw std::runtime_error(
                "MiniCPM-o streaming audio extra context removes every frame");
        }
        if (prefix_to_remove > 0) {
            hidden = hidden.narrow(
                2, prefix_to_remove, hidden.size(2) - prefix_to_remove);
        }
        if (suffix_to_remove > 0) {
            hidden = hidden.narrow(
                2, 0, hidden.size(2) - suffix_to_remove);
        }
        hidden = hidden.transpose(1, 2).contiguous();

        const int64_t past = cache_length();
        const int64_t tokens = hidden.size(1);
        if (past + tokens > position_embedding.size(0)) {
            throw std::runtime_error(
                "MiniCPM-o streaming Whisper position cache exceeds 1500 frames");
        }
        hidden = hidden + position_embedding.narrow(0, past, tokens);
        auto attention_mask = mfq_tensor_backend::zeros(
            {1, 1, tokens, past + tokens},
            hidden.options());
        for (auto & layer : layers) {
            hidden = layer.forward(hidden, attention_mask, true);
        }
        hidden = minicpmo45_layer_norm(
            hidden, final_norm_weight, final_norm_bias, 1e-5);
        hidden = projector1.forward(hidden);
        hidden = mfq_tensor_backend::relu(hidden);
        hidden = projector2.forward(hidden);
        if (hidden.size(1) < 5) {
            throw std::runtime_error(
                "MiniCPM-o streaming audio chunk is too short for stride-5 pooling");
        }
        return mfq_tensor_backend::avg_pool1d(
            hidden.transpose(1, 2),
            std::vector<int64_t>{5},
            std::vector<int64_t>{5})
            .transpose(1, 2).contiguous();
    }

    static std::vector<int64_t> pooled_lengths(
            mfq_tensor_backend::Tensor raw_lengths) {
        auto lengths = raw_lengths.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
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

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor features,
            mfq_tensor_backend::Tensor raw_lengths,
            bool use_cache = false) {
        if (features.dim() != 3 || features.size(1) != 80 ||
                raw_lengths.dim() != 1 ||
                raw_lengths.size(0) != features.size(0)) {
            throw std::runtime_error(
                "MiniCPM-o audio input geometry is invalid");
        }
        auto conv1 = mfq_tensor_backend::conv1d(
            features.to(conv1_weight.scalar_type()),
            conv1_weight, conv1_bias,
            std::vector<int64_t>{1},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        auto hidden = mfq_tensor_backend::gelu(conv1);
        auto conv2 = mfq_tensor_backend::conv1d(
            hidden, conv2_weight, conv2_bias,
            std::vector<int64_t>{2},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        hidden = mfq_tensor_backend::gelu(conv2);
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
            mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        auto key_positions = mfq_tensor_backend::arange(
            past + tokens,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kInt64));
        auto valid_keys = key_positions.unsqueeze(0) <
            (length_tensor + past).unsqueeze(1);
        auto query_positions = mfq_tensor_backend::arange(
            past, past + tokens,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kInt64));
        // Official MiniCPM-o 4.5 uses audio_chunk_length=1.0. Whisper emits
        // 50 encoder frames per second, and every query can see its current
        // 50-frame chunk plus all preceding chunks.
        auto chunk_visible = key_positions.unsqueeze(0) <
            ((query_positions / 50 + 1) * 50).unsqueeze(1);
        auto visible = valid_keys.unsqueeze(1) & chunk_visible.unsqueeze(0);
        auto attention_mask = mfq_tensor_backend::zeros(
            {features.size(0), 1, tokens, past + tokens},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kFloat32))
            .masked_fill(
                visible.logical_not().unsqueeze(1),
                -std::numeric_limits<float>::infinity());
        for (auto & layer : layers) {
            hidden = layer.forward(hidden, attention_mask, use_cache);
        }
        hidden = minicpmo45_layer_norm(
            hidden, final_norm_weight, final_norm_bias, 1e-5);
        hidden = projector1.forward(hidden);
        hidden = mfq_tensor_backend::relu(hidden);
        hidden = projector2.forward(hidden);
        hidden = mfq_tensor_backend::avg_pool1d(
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
    mfq_tensor_backend::Tensor output_norm;
    mfq_tensor_backend::Tensor code_head;
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
        auto head_v_f32 = head_v.to(mfq_tensor_backend::kFloat32);
        auto row_norm = mfq_tensor_backend::sqrt(
            mfq_tensor_backend::sum(head_v_f32.square(), -1, true))
            .clamp_min(1e-12);
        result.code_head = (
            head_g.to(mfq_tensor_backend::kFloat32) * head_v_f32 /
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

    mfq_tensor_backend::Tensor semantic_projection(mfq_tensor_backend::Tensor hidden) const {
        auto projected = semantic_projector1.forward(hidden);
        projected = mfq_tensor_backend::relu(projected);
        projected = semantic_projector2.forward(projected);
        auto norm = mfq_tensor_backend::sqrt(
            mfq_tensor_backend::sum(projected.to(mfq_tensor_backend::kFloat32).square(), -1, true))
            .clamp_min(1e-12);
        return (projected / norm.to(projected.scalar_type())).contiguous();
    }

    mfq_tensor_backend::Tensor speaker_projection(mfq_tensor_backend::Tensor hidden) const {
        return speaker_projector2.forward(
            mfq_tensor_backend::relu(speaker_projector1.forward(hidden)));
    }

    mfq_tensor_backend::Tensor condition(
            mfq_tensor_backend::Tensor text_ids,
            mfq_tensor_backend::Tensor language_hidden) const {
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
        auto suffix_ids = mfq_tensor_backend::tensor(
            std::vector<int64_t>{151692, 151687},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)).reshape({1, 2});
        auto suffix = minicpmo45_embedding(
            text_embedding, suffix_ids, text.scalar_type());
        return mfq_tensor_backend::cat({merged, suffix}, 1).contiguous();
    }

    mfq_tensor_backend::Tensor duplex_condition(
            mfq_tensor_backend::Tensor text_ids,
            mfq_tensor_backend::Tensor language_hidden,
            int64_t audio_bos_token) const {
        if (text_ids.dim() == 1) text_ids = text_ids.unsqueeze(0);
        if (language_hidden.dim() == 2) {
            language_hidden = language_hidden.unsqueeze(0);
        }
        if (text_ids.dim() != 2 || language_hidden.dim() != 3 ||
                text_ids.size(0) != 1 || language_hidden.size(0) != 1 ||
                text_ids.size(1) != language_hidden.size(1) ||
                language_hidden.size(2) != 4096 ||
                audio_bos_token < 0 || audio_bos_token >= text_embedding.out()) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS condition has invalid geometry or audio BOS");
        }
        mfq_tensor_backend::Tensor condition;
        if (text_ids.size(1) > 0) {
            auto text = minicpmo45_embedding(text_embedding, text_ids);
            auto semantic = semantic_projection(language_hidden)
                .to(text.scalar_type());
            condition = (text + semantic).contiguous();
        }
        auto bos_ids = mfq_tensor_backend::tensor(
            std::vector<int64_t>{audio_bos_token},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)).reshape({1, 1});
        auto bos = minicpmo45_embedding(
            text_embedding, bos_ids, mfq_tensor_backend::kBFloat16);
        return condition.defined()
            ? mfq_tensor_backend::cat({condition, bos}, 1).contiguous()
            : bos.contiguous();
    }

    mfq_tensor_backend::Tensor hidden_forward(mfq_tensor_backend::Tensor input_embeddings) {
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
        auto positions = mfq_tensor_backend::arange(
            cache_position, cache_position + tokens,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kInt64));
        MfqOptional<mfq_tensor_backend::Tensor> sequence_length = mfq_nullopt;
        if (cache_position > 0) {
            sequence_length = mfq_tensor_backend::full(
                {batch}, cache_position + tokens,
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt64));
        }
        auto hidden = input_embeddings.to(mfq_tensor_backend::kBFloat16).contiguous();
        for (auto & block : blocks) {
            hidden = block->forward(
                hidden, positions, cache_position,
                sequence_length, config, rope);
        }
        cache_position += tokens;
        auto flat = hidden.reshape(
            {batch * tokens, config.hidden_size});
        auto normalized_f32 = flat.to(mfq_tensor_backend::kFloat32);
        normalized_f32 = normalized_f32 * mfq_tensor_backend::rsqrt(
            mfq_tensor_backend::mean(normalized_f32.square(), -1, true) +
            config.rms_norm_eps);
        return (
            normalized_f32.to(flat.scalar_type()) *
            output_norm.to(flat.scalar_type()))
            .reshape({batch, tokens, config.hidden_size})
            .contiguous();
    }

    mfq_tensor_backend::Tensor logits(mfq_tensor_backend::Tensor hidden) const {
        return mfq_tensor_backend::matmul(
            hidden.to(code_head.scalar_type()),
            code_head.transpose(0, 1));
    }

    struct ChunkResult {
        mfq_tensor_backend::Tensor codes;
        bool finished = false;
    };

    ChunkResult generate_duplex_chunk(
            mfq_tensor_backend::Tensor condition_embeddings,
            int64_t max_new_tokens,
            int64_t minimum_new_tokens,
            int64_t eos_token,
            double temperature,
            double repetition_penalty) {
        if (condition_embeddings.dim() != 3 ||
                condition_embeddings.size(0) != 1 ||
                condition_embeddings.size(2) != config.hidden_size ||
                max_new_tokens <= 0 || minimum_new_tokens < 0 ||
                eos_token < 0 || eos_token >= code_embedding.out() ||
                temperature <= 0.0 || repetition_penalty <= 0.0) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS generation limits are invalid");
        }
        std::vector<mfq_tensor_backend::Tensor> sampled;
        sampled.reserve(static_cast<size_t>(max_new_tokens));
        auto current = condition_embeddings.to(mfq_tensor_backend::kBFloat16).contiguous();
        bool finished = false;
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            auto hidden = hidden_forward(current);
            auto step_logits = logits(
                hidden.index({Slice(), -1, Slice()}))
                .to(mfq_tensor_backend::kFloat32) / temperature;
            if (!sampled.empty() && repetition_penalty != 1.0) {
                const size_t begin = sampled.size() > 16
                    ? sampled.size() - 16 : 0;
                auto counts = mfq_tensor_backend::zeros_like(step_logits);
                for (size_t index = begin; index < sampled.size(); ++index) {
                    counts.scatter_add_(
                        1, sampled[index].reshape({1, 1}),
                        mfq_tensor_backend::ones({1, 1}, counts.options()));
                }
                auto alpha = mfq_tensor_backend::pow(
                    mfq_tensor_backend::full_like(counts, repetition_penalty), counts);
                step_logits = mfq_tensor_backend::where(
                    step_logits < 0,
                    step_logits * alpha,
                    step_logits / alpha);
            }
            if (step < minimum_new_tokens) {
                step_logits.index_put_(
                    {Slice(), eos_token},
                    -std::numeric_limits<float>::infinity());
            }
            auto token = mfq_tensor_backend::multinomial(
                mfq_tensor_backend::softmax(step_logits, -1), 1)
                .reshape({1}).to(mfq_tensor_backend::kInt64);
            sampled.push_back(token);
            finished = token.eq(eos_token).all().item<bool>();
            if (finished) break;
            current = minicpmo45_embedding(
                code_embedding, token.unsqueeze(1));
        }
        const int64_t returned = std::max<int64_t>(
            0, static_cast<int64_t>(sampled.size()) - 1);
        mfq_tensor_backend::Tensor codes;
        if (returned > 0) {
            codes = mfq_tensor_backend::stack(sampled, 1)
                .narrow(1, 0, returned)
                .unsqueeze(-1).contiguous();
        } else {
            codes = mfq_tensor_backend::empty(
                {1, 0, 1},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt64));
        }
        return {codes, finished};
    }

    mfq_tensor_backend::Tensor generate_official(
            mfq_tensor_backend::Tensor condition_embeddings,
            int64_t steps,
            int64_t eos_token = 6561,
            int64_t minimum_steps = 50,
            double temperature = 0.8,
            double top_p = 0.85,
            int64_t top_k = 25,
            double repetition_penalty = 1.05,
            std::vector<mfq_tensor_backend::Tensor> * logits_trace = nullptr,
            double min_p = 0.0,
            std::mt19937 * evaluator_rng = nullptr,
            int64_t evaluator_min_keep = 0) {
        if (steps <= 0 || minimum_steps < 0 ||
                eos_token < 0 || eos_token >= 6562 ||
                temperature <= 0.0 || top_p <= 0.0 || top_p > 1.0 ||
                min_p < 0.0 || min_p > 1.0 ||
                top_k < 3 || top_k > 6562 || repetition_penalty <= 0.0 ||
                evaluator_min_keep < 0 || evaluator_min_keep > top_k) {
            throw std::runtime_error(
                "MiniCPM-o TTS generation limits are invalid");
        }
        if (condition_embeddings.size(0) != 1) {
            throw std::runtime_error(
                "official MiniCPM-o TTS generation requires batch size one");
        }
        reset(condition_embeddings.size(0));
        std::vector<mfq_tensor_backend::Tensor> generated;
        bool hit_eos = false;
        generated.reserve(static_cast<size_t>(steps));
        auto current = condition_embeddings;
        for (int64_t step = 0; step < steps; ++step) {
            auto hidden = hidden_forward(current);
            auto raw_step_logits = logits(
                hidden.index({Slice(), -1, Slice()}))
                .to(mfq_tensor_backend::kFloat32);
            if (logits_trace != nullptr) {
                logits_trace->push_back(raw_step_logits.clone());
            }
            auto step_logits = evaluator_rng != nullptr
                ? raw_step_logits / temperature
                : raw_step_logits;
            if (!generated.empty()) {
                if (repetition_penalty != 1.0) {
                    auto counts = mfq_tensor_backend::zeros_like(step_logits);
                    const size_t begin = generated.size() > 16
                        ? generated.size() - 16 : 0;
                    for (size_t index = begin; index < generated.size(); ++index) {
                        counts.scatter_add_(
                            1, generated[index].reshape({1, 1}),
                            mfq_tensor_backend::ones(
                                {1, 1}, counts.options()));
                    }
                    auto alpha = mfq_tensor_backend::pow(
                        mfq_tensor_backend::full_like(counts, repetition_penalty), counts);
                    step_logits = mfq_tensor_backend::where(
                        step_logits < 0,
                        step_logits * alpha,
                        step_logits / alpha);
                }
            }
            if (step < minimum_steps) {
                step_logits.index_put_({Slice(), eos_token},
                    -std::numeric_limits<float>::infinity());
            }
            if (evaluator_rng != nullptr) {
                auto cpu_logits = step_logits
                    .reshape({-1}).to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
                    .contiguous();
                const auto * values = cpu_logits.data_ptr<float>();
                const int64_t vocabulary = cpu_logits.numel();
                float maximum = values[0];
                for (int64_t index = 1; index < vocabulary; ++index) {
                    maximum = std::max(maximum, values[index]);
                }
                std::vector<std::pair<float, int64_t>> probabilities;
                probabilities.reserve(static_cast<size_t>(vocabulary));
                float sum = 0.0F;
                for (int64_t index = 0; index < vocabulary; ++index) {
                    const float probability = std::exp(values[index] - maximum);
                    probabilities.emplace_back(probability, index);
                    sum += probability;
                }
                for (auto & probability : probabilities) {
                    probability.first /= sum;
                }
                std::sort(probabilities.begin(), probabilities.end(),
                    [](const auto & left, const auto & right) {
                        return left.first > right.first;
                    });
                std::vector<float> kept_probabilities;
                std::vector<int64_t> kept_indices;
                kept_probabilities.reserve(static_cast<size_t>(top_k));
                kept_indices.reserve(static_cast<size_t>(top_k));
                float cumulative = 0.0F;
                for (const auto & probability : probabilities) {
                    if (static_cast<int64_t>(kept_probabilities.size()) <
                            evaluator_min_keep ||
                            (cumulative < static_cast<float>(top_p) &&
                             static_cast<int64_t>(kept_probabilities.size()) <
                                 top_k)) {
                        cumulative += probability.first;
                        kept_probabilities.push_back(probability.first);
                        kept_indices.push_back(probability.second);
                    } else {
                        break;
                    }
                }
                float kept_sum = std::accumulate(
                    kept_probabilities.begin(), kept_probabilities.end(), 0.0F);
                for (auto & probability : kept_probabilities) {
                    probability /= kept_sum;
                }
                std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
                const float sample = distribution(*evaluator_rng);
                float selected_sum = 0.0F;
                int64_t selected = kept_indices.back();
                for (size_t index = 0; index < kept_probabilities.size(); ++index) {
                    selected_sum += kept_probabilities[index];
                    if (sample <= selected_sum) {
                        selected = kept_indices[index];
                        break;
                    }
                }
                auto token = mfq_tensor_backend::tensor(
                    std::vector<int64_t>{selected},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                        .dtype(mfq_tensor_backend::kInt64));
                generated.push_back(token);
                if (token.eq(eos_token).all().item<bool>()) {
                    hit_eos = true;
                    break;
                }
                current = minicpmo45_embedding(
                    code_embedding, token.unsqueeze(1));
                continue;
            }
            if (top_k < step_logits.size(1)) {
                auto top_values = std::get<0>(
                    mfq_tensor_backend::topk(
                        step_logits, top_k, -1, true, true));
                auto threshold = top_values.select(1, top_k - 1)
                    .unsqueeze(1);
                step_logits = step_logits.masked_fill(
                    step_logits < threshold,
                    -std::numeric_limits<float>::infinity());
            }
            if (top_p < 1.0) {
                auto sorted = mfq_tensor_backend::sort(
                    step_logits, -1, true);
                auto sorted_logits = std::get<0>(sorted);
                auto sorted_indices = std::get<1>(sorted);
                auto cumulative = mfq_tensor_backend::cumsum(
                    mfq_tensor_backend::softmax(sorted_logits, -1), -1);
                auto keep = cumulative <= top_p;
                keep.narrow(1, 1, keep.size(1) - 1).copy_(
                    keep.narrow(1, 0, keep.size(1) - 1).clone());
                keep.index_put_({Slice(), 0}, true);
                sorted_logits = sorted_logits.masked_fill(
                    keep.logical_not(),
                    -std::numeric_limits<float>::infinity());
                step_logits = mfq_tensor_backend::full_like(
                    step_logits,
                    -std::numeric_limits<float>::infinity());
                step_logits.scatter_(1, sorted_indices, sorted_logits);
            }
            if (min_p > 0.0) {
                auto max_logits = std::get<0>(
                    mfq_tensor_backend::max(step_logits, -1, true));
                auto remove =
                    step_logits < max_logits + std::log(min_p);
                step_logits = step_logits.masked_fill(
                    remove,
                    -std::numeric_limits<float>::infinity());
            }
            step_logits = step_logits / temperature;
            auto token = mfq_tensor_backend::multinomial(
                mfq_tensor_backend::softmax(step_logits, -1), 1)
                .reshape({1}).to(mfq_tensor_backend::kInt64);
            generated.push_back(token);
            if (token.eq(eos_token).all().item<bool>()) {
                hit_eos = true;
                break;
            }
            current = minicpmo45_embedding(
                code_embedding, token.unsqueeze(1));
        }
        auto sampled = mfq_tensor_backend::stack(generated, 1);
        const int64_t returned = sampled.size(1) - (hit_eos ? 1 : 0);
        return sampled.narrow(1, 0, returned)
            .unsqueeze(-1).contiguous();
    }
};

struct MiniCPMO45Bounds {
    int64_t batch = 0;
    int64_t source = 0;
    int64_t begin = 0;
    int64_t end = 0;
};

static std::vector<MiniCPMO45Bounds> minicpmo45_parse_bounds(
        mfq_tensor_backend::Tensor bounds,
        const char * label) {
    if (!bounds.defined() || bounds.numel() == 0) return {};
    bounds = bounds.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
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
    mfq_tensor_backend::Tensor vision_states;
    mfq_tensor_backend::Tensor image_embeddings;
    mfq_tensor_backend::Tensor audio_embeddings;
    mfq_tensor_backend::Tensor input_embeddings;
    mfq_tensor_backend::Tensor hidden_states;
    mfq_tensor_backend::Tensor logits;
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
        return load_with_language(
            load_model(mfq_path, config_path, context_size), mfq_path);
    }

    static MiniCPMO45Runtime load_with_language(
            Model language,
            const std::string & mfq_path) {
        MiniCPMO45Runtime result;
        result.language = std::move(language);
        if (!result.language.c.is_minicpmo45()) {
            throw std::runtime_error(
                "native composite graph requires MiniCPM-o 4.5");
        }
        g_layer_placement.load_device =
            g_layer_placement.primary_device();
        MfqCudaGuard guard(
            g_layer_placement.primary_device());
        MfqFile mfq(mfq_path);
        result.vision = MiniCPMO45VisionEncoder::load(mfq);
        result.resampler = MiniCPMO45Resampler::load(mfq);
        result.audio = MiniCPMO45AudioEncoder::load(mfq);
        result.tts = MiniCPMO45TtsDecoder::load(mfq);
        return result;
    }

    MiniCPMO45ForwardResult forward(
            mfq_tensor_backend::Tensor input_ids,
            mfq_tensor_backend::Tensor position_ids,
            mfq_tensor_backend::Tensor attention_mask,
            mfq_tensor_backend::Tensor pixels,
            mfq_tensor_backend::Tensor patch_mask,
            mfq_tensor_backend::Tensor target_sizes,
            mfq_tensor_backend::Tensor image_bounds,
            mfq_tensor_backend::Tensor audio_features,
            mfq_tensor_backend::Tensor audio_lengths,
            mfq_tensor_backend::Tensor audio_bounds) {
        if (input_ids.dim() == 1) input_ids = input_ids.unsqueeze(0);
        if (input_ids.dim() != 2) {
            throw std::runtime_error(
                "MiniCPM-o input_ids must have shape [batch,tokens]");
        }
        input_ids = input_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
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
                pixels.to(mfq_tensor_backend::kCUDA), patch_mask, target_sizes);
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
                audio_features.to(mfq_tensor_backend::kCUDA), audio_lengths, false);
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
        MfqOptional<mfq_tensor_backend::Tensor> positions = mfq_nullopt;
        if (position_ids.defined()) positions = position_ids;
        MfqOptional<mfq_tensor_backend::Tensor> mask = mfq_nullopt;
        if (attention_mask.defined()) mask = attention_mask;
        result.hidden_states = language.hidden_forward_inputs(
            input_ids, result.input_embeddings,
            positions, mfq_nullopt, nullptr, mask,
            position_ids.defined());
        result.logits = language.logits_from_hidden(result.hidden_states);
        return result;
    }
};

struct MiniCPMO45DuplexSpecialIds {
    int64_t unit_start = -1;
    int64_t unit_end = -1;
    int64_t image_start = -1;
    int64_t image_end = -1;
    int64_t slice_start = -1;
    int64_t slice_end = -1;
    int64_t listen = -1;
    int64_t speak = -1;
    int64_t tts_bos = -1;
    int64_t tts_eos = -1;
    int64_t chunk_eos = -1;
    int64_t chunk_tts_eos = -1;
    int64_t turn_eos = -1;
    int64_t tts_pad = -1;
    int64_t audio_bos = -1;

    static MiniCPMO45DuplexSpecialIds from_tensor(mfq_tensor_backend::Tensor value) {
        value = value.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
        if (value.numel() != 15) {
            throw std::runtime_error(
                "MiniCPM-o duplex special_ids must contain 15 token IDs");
        }
        const auto * ids = value.data_ptr<int64_t>();
        MiniCPMO45DuplexSpecialIds result;
        result.unit_start = ids[0];
        result.unit_end = ids[1];
        result.image_start = ids[2];
        result.image_end = ids[3];
        result.slice_start = ids[4];
        result.slice_end = ids[5];
        result.listen = ids[6];
        result.speak = ids[7];
        result.tts_bos = ids[8];
        result.tts_eos = ids[9];
        result.chunk_eos = ids[10];
        result.chunk_tts_eos = ids[11];
        result.turn_eos = ids[12];
        result.tts_pad = ids[13];
        result.audio_bos = ids[14];
        const int64_t minimum = *std::min_element(ids, ids + 15);
        if (minimum < 0) {
            throw std::runtime_error(
                "MiniCPM-o duplex special token IDs must be non-negative");
        }
        return result;
    }

    bool is_chunk_terminator(int64_t token) const {
        return token == listen || token == chunk_eos ||
            token == chunk_tts_eos;
    }

    bool is_special(int64_t token) const {
        const std::array<int64_t, 15> values = {
            unit_start, unit_end, image_start, image_end,
            slice_start, slice_end, listen, speak, tts_bos,
            tts_eos, chunk_eos, chunk_tts_eos, turn_eos,
            tts_pad, audio_bos};
        return std::find(values.begin(), values.end(), token) != values.end();
    }
};

struct MiniCPMO45DuplexStepResult {
    mfq_tensor_backend::Tensor decision_logits;
    mfq_tensor_backend::Tensor audio_embeddings;
    mfq_tensor_backend::Tensor generated_ids;
    mfq_tensor_backend::Tensor tts_codes;
    bool is_listen = false;
    bool end_of_turn = false;
    bool tts_force_flush = false;
};

struct MiniCPMO45DuplexSession {
    MiniCPMO45Runtime & runtime;
    MiniCPMO45DuplexSpecialIds ids;
    std::vector<int64_t> forbidden_ids;
    std::vector<int64_t> generated_text_ids;
    int64_t audio_chunk_index = 0;
    int64_t tts_text_start_position = 0;
    bool current_turn_ended = true;
    bool greedy = false;
    double temperature = 0.7;
    int64_t top_k = 100;
    double top_p = 0.8;
    double listen_probability_scale = 1.0;
    double repetition_penalty = 1.05;
    int64_t repetition_window = 512;
    double length_penalty = 1.0;
    double tts_temperature = 0.8;
    double tts_repetition_penalty = 1.05;

    MiniCPMO45DuplexSession(
            MiniCPMO45Runtime & runtime_,
            MiniCPMO45DuplexSpecialIds ids_,
            std::vector<int64_t> forbidden_ids_,
            bool greedy_)
        : runtime(runtime_), ids(ids_),
          forbidden_ids(std::move(forbidden_ids_)), greedy(greedy_) {
        if (std::find(forbidden_ids.begin(), forbidden_ids.end(),
                ids.chunk_eos) == forbidden_ids.end()) {
            forbidden_ids.push_back(ids.chunk_eos);
        }
        if (std::find(forbidden_ids.begin(), forbidden_ids.end(),
                ids.tts_pad) == forbidden_ids.end()) {
            forbidden_ids.push_back(ids.tts_pad);
        }
    }

    void prepare(
            mfq_tensor_backend::Tensor system_prefix_ids,
            mfq_tensor_backend::Tensor reference_audio_features = mfq_tensor_backend::Tensor(),
            mfq_tensor_backend::Tensor system_suffix_ids = mfq_tensor_backend::Tensor()) {
        runtime.language.reset(1);
        runtime.audio.reset();
        runtime.tts.reset(1);
        generated_text_ids.clear();
        audio_chunk_index = 0;
        tts_text_start_position = 0;
        current_turn_ended = true;
        if (system_prefix_ids.defined() &&
                system_prefix_ids.numel() > 0) {
            feed_ids(system_prefix_ids);
        }
        if (reference_audio_features.defined()) {
            if (reference_audio_features.dim() != 3 ||
                    reference_audio_features.size(0) != 1 ||
                    reference_audio_features.size(1) != 80 ||
                    reference_audio_features.size(2) < 3) {
                throw std::runtime_error(
                    "MiniCPM-o duplex reference audio expects [1,80,frames]");
            }
            auto raw_lengths = mfq_tensor_backend::tensor(
                std::vector<int64_t>{reference_audio_features.size(2)},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
            auto embeddings = runtime.audio.forward(
                reference_audio_features.to(mfq_tensor_backend::kCUDA),
                raw_lengths, false);
            feed_embeddings(embeddings);
            runtime.audio.reset();
        }
        if (system_suffix_ids.defined() &&
                system_suffix_ids.numel() > 0) {
            feed_ids(system_suffix_ids);
        }
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> feed_embeddings(
            mfq_tensor_backend::Tensor embeddings,
            mfq_tensor_backend::Tensor token_ids = mfq_tensor_backend::Tensor()) {
        if (embeddings.dim() == 2) embeddings = embeddings.unsqueeze(0);
        if (embeddings.dim() != 3 || embeddings.size(0) != 1 ||
                embeddings.size(2) != runtime.language.c.hidden_size) {
            throw std::runtime_error(
                "MiniCPM-o duplex embeddings must have shape [1,tokens,4096]");
        }
        if (!token_ids.defined()) {
            token_ids = mfq_tensor_backend::zeros(
                {1, embeddings.size(1)},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA));
        } else {
            if (token_ids.dim() == 1) token_ids = token_ids.unsqueeze(0);
            if (token_ids.dim() != 2 || token_ids.size(0) != 1 ||
                    token_ids.size(1) != embeddings.size(1)) {
                throw std::runtime_error(
                    "MiniCPM-o duplex token IDs do not match embeddings");
            }
            token_ids = token_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        }
        auto hidden = runtime.language.hidden_forward_inputs(
            token_ids, embeddings.to(mfq_tensor_backend::kBFloat16).contiguous());
        auto logits = runtime.language.logits_from_hidden(
            hidden.index({Slice(), -1, Slice()}));
        return {logits, hidden};
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> feed_ids(mfq_tensor_backend::Tensor token_ids) {
        if (token_ids.dim() == 0) token_ids = token_ids.reshape({1});
        if (token_ids.dim() == 1) token_ids = token_ids.unsqueeze(0);
        token_ids = token_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        return feed_embeddings(
            runtime.language.embed_forward(token_ids), token_ids);
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> feed_id(int64_t token) {
        return feed_ids(mfq_tensor_backend::tensor(
            std::vector<int64_t>{token},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)));
    }

    mfq_tensor_backend::Tensor apply_text_sampling_filters(mfq_tensor_backend::Tensor logits) const {
        logits = logits.clone().to(mfq_tensor_backend::kFloat32);
        if (!generated_text_ids.empty() && repetition_penalty != 1.0) {
            const size_t begin = generated_text_ids.size() >
                    static_cast<size_t>(repetition_window)
                ? generated_text_ids.size() -
                    static_cast<size_t>(repetition_window)
                : 0;
            std::vector<int64_t> recent(
                generated_text_ids.begin() +
                    static_cast<std::ptrdiff_t>(begin),
                generated_text_ids.end());
            std::sort(recent.begin(), recent.end());
            recent.erase(std::unique(recent.begin(), recent.end()), recent.end());
            auto recent_tensor = mfq_tensor_backend::tensor(
                recent, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(logits.device()));
            auto selected = logits.index_select(1, recent_tensor);
            selected = repetition_penalty > 1.0
                ? selected / repetition_penalty
                : selected * (1.0 / repetition_penalty);
            logits.index_copy_(1, recent_tensor, selected);
        }
        if (length_penalty != 1.0) {
            auto selected = logits.index({Slice(), ids.turn_eos});
            selected = mfq_tensor_backend::where(
                selected > 0,
                selected / length_penalty,
                selected * length_penalty);
            logits.index_put_({Slice(), ids.turn_eos}, selected);
        }
        if (listen_probability_scale != 1.0) {
            logits.index_put_(
                {Slice(), ids.listen},
                logits.index({Slice(), ids.listen}) *
                    listen_probability_scale);
        }
        if (!forbidden_ids.empty()) {
            auto forbidden = mfq_tensor_backend::tensor(
                forbidden_ids,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(logits.device()));
            logits.index_fill_(
                1, forbidden,
                -std::numeric_limits<float>::infinity());
        }
        if (greedy) return logits;
        logits = logits / temperature;
        if (top_k > 0 && top_k < logits.size(1)) {
            auto values = std::get<0>(
                mfq_tensor_backend::topk(
                    logits, top_k, -1, true, true));
            auto threshold = values.select(1, top_k - 1).unsqueeze(1);
            logits = logits.masked_fill(
                logits < threshold,
                -std::numeric_limits<float>::infinity());
        }
        if (top_p < 1.0) {
            auto sorted = mfq_tensor_backend::sort(logits, -1, true);
            auto sorted_logits = std::get<0>(sorted);
            auto sorted_indices = std::get<1>(sorted);
            auto cumulative = mfq_tensor_backend::cumsum(
                mfq_tensor_backend::softmax(sorted_logits, -1), -1);
            auto remove = cumulative > top_p;
            auto shifted = mfq_tensor_backend::zeros_like(remove);
            shifted.narrow(1, 1, remove.size(1) - 1)
                .copy_(remove.narrow(1, 0, remove.size(1) - 1));
            sorted_logits = sorted_logits.masked_fill(
                shifted, -std::numeric_limits<float>::infinity());
            logits = mfq_tensor_backend::full_like(
                logits, -std::numeric_limits<float>::infinity());
            logits.scatter_(1, sorted_indices, sorted_logits);
        }
        return logits;
    }

    int64_t sample_text_token(mfq_tensor_backend::Tensor logits, bool force_listen) const {
        if (force_listen) return ids.listen;
        auto original = logits.to(mfq_tensor_backend::kFloat32);
        auto first = greedy
            ? mfq_tensor_backend::argmax(original, -1)
            : mfq_tensor_backend::multinomial(mfq_tensor_backend::softmax(original, -1), 1)
                .reshape({1});
        const int64_t first_id = first.item<int64_t>();
        if (first_id == ids.chunk_eos) return first_id;
        auto filtered = apply_text_sampling_filters(original);
        auto sampled = greedy
            ? mfq_tensor_backend::argmax(filtered, -1)
            : mfq_tensor_backend::multinomial(mfq_tensor_backend::softmax(filtered, -1), 1)
                .reshape({1});
        return sampled.item<int64_t>();
    }

    MiniCPMO45DuplexStepResult run_step(
            mfq_tensor_backend::Tensor pixels,
            mfq_tensor_backend::Tensor patch_mask,
            mfq_tensor_backend::Tensor target_sizes,
            mfq_tensor_backend::Tensor image_slice_counts,
            mfq_tensor_backend::Tensor audio_features,
            int64_t audio_prefix_extra_frames,
            int64_t audio_suffix_extra_frames,
            mfq_tensor_backend::Tensor text_ids,
            int64_t max_new_speak_tokens,
            bool force_listen,
            bool force_speak = false) {
        if (max_new_speak_tokens < 2) {
            throw std::runtime_error(
                "MiniCPM-o duplex generation requires at least two token slots");
        }
        auto pending = feed_id(ids.unit_start);
        mfq_tensor_backend::Tensor generation_logits;
        bool has_content = false;
        mfq_tensor_backend::Tensor audio_embeddings;

        if (pixels.defined()) {
            if (!patch_mask.defined() || !target_sizes.defined()) {
                throw std::runtime_error(
                    "MiniCPM-o duplex image pixels require patch mask and target sizes");
            }
            auto vision_states = runtime.vision.forward(
                pixels.to(mfq_tensor_backend::kCUDA), patch_mask, target_sizes);
            auto image_embeddings = runtime.resampler.forward(
                vision_states, target_sizes);
            std::vector<int64_t> counts;
            if (image_slice_counts.defined()) {
                auto count_tensor = image_slice_counts
                    .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
                const auto * values = count_tensor.data_ptr<int64_t>();
                counts.assign(values, values + count_tensor.numel());
            } else {
                counts.assign(
                    static_cast<size_t>(image_embeddings.size(0)), 1);
            }
            int64_t offset = 0;
            for (const int64_t count : counts) {
                if (count <= 0 || offset + count > image_embeddings.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o duplex image slice counts are invalid");
                }
                pending = feed_id(ids.image_start);
                pending = feed_embeddings(image_embeddings.index({offset}));
                pending = feed_id(ids.image_end);
                ++offset;
                for (int64_t slice = 1; slice < count; ++slice) {
                    pending = feed_id(ids.slice_start);
                    pending = feed_embeddings(image_embeddings.index({offset}));
                    pending = feed_id(ids.slice_end);
                    ++offset;
                }
            }
            if (offset != image_embeddings.size(0)) {
                throw std::runtime_error(
                    "MiniCPM-o duplex image slice counts do not cover embeddings");
            }
            generation_logits = pending.first;
            has_content = true;
        }

        if (audio_features.defined()) {
            audio_embeddings = runtime.audio.forward_streaming(
                audio_features.to(mfq_tensor_backend::kCUDA),
                audio_prefix_extra_frames,
                audio_suffix_extra_frames);
            pending = feed_embeddings(audio_embeddings);
            generation_logits = pending.first;
            ++audio_chunk_index;
            has_content = true;
        }

        if (text_ids.defined() && text_ids.numel() > 0) {
            pending = feed_ids(text_ids);
            if (!generation_logits.defined()) {
                generation_logits = pending.first;
            }
            has_content = true;
        }
        if (!has_content) {
            throw std::runtime_error(
                "MiniCPM-o duplex step contains no image, audio, or text input");
        }
        if (pixels.defined() && !audio_features.defined()) {
            ++audio_chunk_index;
        }

        MiniCPMO45DuplexStepResult result;
        result.decision_logits = generation_logits;
        result.audio_embeddings = audio_embeddings;
        std::vector<int64_t> generated;
        std::vector<int64_t> spoken_ids;
        std::vector<mfq_tensor_backend::Tensor> spoken_hidden;
        auto logits = generation_logits;
        bool force_current = force_listen;
        for (int64_t index = 0; index < max_new_speak_tokens; ++index) {
            if (index == max_new_speak_tokens - 1) {
                feed_id(ids.chunk_eos);
                generated.push_back(ids.chunk_eos);
                break;
            }
            const bool forced_decision = force_current;
            int64_t token = sample_text_token(logits, force_current);
            force_current = false;
            if (!forced_decision && token != ids.chunk_eos) {
                generated_text_ids.push_back(token);
            }
            if (!forced_decision && token == ids.listen &&
                    (!current_turn_ended || force_speak)) {
                token = ids.tts_bos;
            }
            generated.push_back(token);
            result.is_listen = token == ids.listen;
            if (ids.is_chunk_terminator(token)) {
                pending = feed_id(token);
                break;
            }

            current_turn_ended = false;
            pending = feed_id(token);
            logits = pending.first;
            result.end_of_turn = token == ids.turn_eos;
            if (result.end_of_turn) current_turn_ended = true;
            if (index != 0) {
                spoken_ids.push_back(token);
                spoken_hidden.push_back(
                    pending.second.index({Slice(), -1, Slice()}));
            }
        }
        feed_id(ids.unit_end);

        result.generated_ids = mfq_tensor_backend::tensor(
            generated,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA));
        if (!result.is_listen) {
            mfq_tensor_backend::Tensor text_id_tensor;
            mfq_tensor_backend::Tensor hidden_tensor;
            if (spoken_ids.empty()) {
                text_id_tensor = mfq_tensor_backend::empty(
                    {1, 0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                        .device(mfq_tensor_backend::kCUDA));
                hidden_tensor = mfq_tensor_backend::empty(
                    {1, 0, runtime.language.c.hidden_size},
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16)
                        .device(mfq_tensor_backend::kCUDA));
            } else {
                text_id_tensor = mfq_tensor_backend::tensor(
                    spoken_ids,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                        .device(mfq_tensor_backend::kCUDA)).unsqueeze(0);
                hidden_tensor = mfq_tensor_backend::stack(spoken_hidden, 1)
                    .to(mfq_tensor_backend::kBFloat16).contiguous();
            }
            auto condition = runtime.tts.duplex_condition(
                text_id_tensor, hidden_tensor, ids.audio_bos);
            const bool first_tts_chunk = tts_text_start_position == 0;
            if (first_tts_chunk) {
                runtime.tts.reset(1);
                result.tts_force_flush = true;
            }
            if (runtime.tts.cache_position != tts_text_start_position) {
                throw std::runtime_error(
                    "MiniCPM-o duplex TTS cache position is inconsistent");
            }
            const int64_t minimum_codes =
                result.end_of_turn || first_tts_chunk ? 0 : 26;
            auto tts_result = runtime.tts.generate_duplex_chunk(
                condition, 26, minimum_codes, 6561,
                tts_temperature, tts_repetition_penalty);
            result.tts_codes = tts_result.codes;
            if (result.end_of_turn) {
                runtime.tts.reset(1);
                tts_text_start_position = 0;
            } else {
                tts_text_start_position +=
                    condition.size(1) + result.tts_codes.size(1);
                if (runtime.tts.cache_position != tts_text_start_position) {
                    throw std::runtime_error(
                        "MiniCPM-o duplex TTS cache did not advance exactly");
                }
            }
        } else {
            result.tts_codes = mfq_tensor_backend::empty(
                {1, 0, 1},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA));
        }
        return result;
    }
};

static mfq_tensor_backend::Tensor minicpmo45_load_tensor(
        const std::string & path,
        bool required) {
    if (!std::filesystem::is_regular_file(path)) {
        if (required) {
            throw std::runtime_error(
                "MiniCPM-o tensor file is missing: " + path);
        }
        return mfq_tensor_backend::Tensor();
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
    auto value = mfq_tensor_backend::pickle_load(bytes);
    if (!value.isTensor()) {
        throw std::runtime_error(
            "MiniCPM-o input is not a tensor: " + path);
    }
    return value.toTensor();
}

static void minicpmo45_save_tensor(
        const mfq_tensor_backend::Tensor & value,
        const std::string & path) {
    if (!value.defined()) return;
    const std::filesystem::path output(path);
    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }
    minicpmo45_write_pickle_tensor(value, path);
}

static int64_t minicpmo45_load_optional_scalar(
        const std::string & path,
        int64_t fallback) {
    auto value = minicpmo45_load_tensor(path, false);
    if (!value.defined()) return fallback;
    if (value.numel() != 1) {
        throw std::runtime_error(
            "MiniCPM-o scalar tensor must contain one value: " + path);
    }
    return value.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).item<int64_t>();
}

static std::string minicpmo45_duplex_step_prefix(
        const std::string & prefix,
        int64_t step) {
    std::ostringstream stream;
    stream << prefix << ".step" << std::setfill('0')
           << std::setw(4) << step;
    return stream.str();
}

static int run_minicpmo45_duplex(
        const std::string & mfq_path,
        const std::string & config_path,
        const std::string & input_prefix,
        const std::string & output_prefix,
        int64_t context_size,
        int64_t steps,
        int64_t max_speak_tokens,
        bool greedy,
        int64_t seed) {
    mfq_tensor_backend::NoGradGuard no_grad;
    g_profiler.enabled = false;
    if (input_prefix.empty() || output_prefix.empty() || steps <= 0 ||
            max_speak_tokens < 2 || seed < 0) {
        throw std::runtime_error(
            "MiniCPM-o duplex mode requires prefixes, positive steps, "
            "at least two generation slots, and a non-negative seed");
    }
    mfq_tensor_backend::manual_seed(seed);
    mfq_cuda_manual_seed_all(seed);
    auto runtime = MiniCPMO45Runtime::load(
        mfq_path, config_path, context_size);
    auto special_ids = MiniCPMO45DuplexSpecialIds::from_tensor(
        minicpmo45_load_tensor(input_prefix + ".special_ids.pt", true));
    std::vector<int64_t> forbidden_ids;
    auto forbidden = minicpmo45_load_tensor(
        input_prefix + ".forbidden_ids.pt", false);
    if (forbidden.defined()) {
        forbidden = forbidden.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
            .contiguous().reshape({-1});
        const auto * values = forbidden.data_ptr<int64_t>();
        forbidden_ids.assign(values, values + forbidden.numel());
    }
    MiniCPMO45DuplexSession session(
        runtime, special_ids, std::move(forbidden_ids), greedy);
    const auto system_ids = minicpmo45_load_tensor(
        input_prefix + ".system_ids.pt", false);
    session.prepare(system_ids);

    int64_t listen_steps = 0;
    int64_t speak_steps = 0;
    for (int64_t step = 0; step < steps; ++step) {
        const std::string input = minicpmo45_duplex_step_prefix(
            input_prefix, step);
        const std::string output = minicpmo45_duplex_step_prefix(
            output_prefix, step);
        const bool reset_session = minicpmo45_load_optional_scalar(
            input + ".reset_session.pt", 0) != 0;
        if (reset_session) session.prepare(system_ids);
        const auto pixels = minicpmo45_load_tensor(
            input + ".pixel_values.pt", false);
        const auto patch_mask = minicpmo45_load_tensor(
            input + ".patch_mask.pt", false);
        const auto target_sizes = minicpmo45_load_tensor(
            input + ".target_sizes.pt", false);
        const auto image_slice_counts = minicpmo45_load_tensor(
            input + ".image_slice_counts.pt", false);
        const auto audio_features = minicpmo45_load_tensor(
            input + ".audio_features.pt", false);
        const auto text_ids = minicpmo45_load_tensor(
            input + ".text_ids.pt", false);
        const int64_t prefix_extra = minicpmo45_load_optional_scalar(
            input + ".audio_prefix_extra_frames.pt",
            session.audio_chunk_index == 0 ? 0 : 2);
        const int64_t suffix_extra = minicpmo45_load_optional_scalar(
            input + ".audio_suffix_extra_frames.pt", 2);
        const bool force_listen = minicpmo45_load_optional_scalar(
            input + ".force_listen.pt", 0) != 0;

        auto result = session.run_step(
            pixels, patch_mask, target_sizes, image_slice_counts,
            audio_features, prefix_extra, suffix_extra, text_ids,
            max_speak_tokens, force_listen);
        minicpmo45_save_tensor(
            result.decision_logits, output + ".decision_logits.pt");
        minicpmo45_save_tensor(
            result.audio_embeddings, output + ".audio_embeddings.pt");
        minicpmo45_save_tensor(
            result.generated_ids, output + ".generated_ids.pt");
        minicpmo45_save_tensor(
            result.tts_codes, output + ".tts_codes.pt");
        minicpmo45_save_tensor(
            mfq_tensor_backend::tensor(
                std::vector<int64_t>{
                    result.is_listen ? 1 : 0,
                    result.end_of_turn ? 1 : 0,
                    runtime.language.cache_pos,
                    runtime.audio.cache_length(),
                    runtime.tts.cache_position,
                    session.audio_chunk_index},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)),
            output + ".state.pt");
        if (result.is_listen) {
            ++listen_steps;
        } else {
            ++speak_steps;
        }
    }
    std::cout << "minicpmo45_duplex=ok"
              << " steps=" << steps
              << " listens=" << listen_steps
              << " speaks=" << speak_steps
              << " language_cache=" << runtime.language.cache_pos
              << " audio_cache=" << runtime.audio.cache_length()
              << " tts_cache=" << runtime.tts.cache_position
              << " seed=" << seed << "\n";
    return 0;
}


static mfq_tensor_backend::Tensor minicpmo45_eval_embeddings(
        MiniCPMO45Runtime & runtime,
        mfq_tensor_backend::Tensor input_ids,
        const mfq_tensor_backend::Tensor & image_embeddings,
        mfq_tensor_backend::Tensor image_bounds,
        const mfq_tensor_backend::Tensor & audio_embeddings = mfq_tensor_backend::Tensor(),
        mfq_tensor_backend::Tensor audio_bounds = mfq_tensor_backend::Tensor(),
        mfq_tensor_backend::Tensor audio_valid_lengths = mfq_tensor_backend::Tensor()) {
    if (input_ids.dim() == 1) input_ids = input_ids.unsqueeze(0);
    if (input_ids.dim() != 2 || input_ids.size(0) != 1) {
        throw std::runtime_error(
            "MiniCPM-o eval batch input_ids must have shape [1,tokens]");
    }
    input_ids = input_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
    auto embeddings = runtime.language.embed_forward(input_ids);
    const auto bounds = minicpmo45_parse_bounds(image_bounds, "image");
    const int64_t image_count = image_embeddings.defined()
        ? image_embeddings.size(0) : 0;
    if (bounds.size() != static_cast<size_t>(image_count)) {
        throw std::runtime_error(
            "MiniCPM-o eval batch requires one image bound per frame");
    }
    for (const auto & bound : bounds) {
        if (bound.batch != 0 || bound.source >= image_embeddings.size(0) ||
                bound.end > input_ids.size(1) ||
                bound.end - bound.begin != image_embeddings.size(1)) {
            throw std::runtime_error(
                "MiniCPM-o eval image bound does not match resampler output");
        }
        embeddings.index({
            0, Slice(bound.begin, bound.end), Slice()})
            .copy_(image_embeddings.index({bound.source})
                .to(embeddings.scalar_type()));
    }
    const auto audios = minicpmo45_parse_bounds(audio_bounds, "audio");
    if (!audios.empty()) {
        if (!audio_embeddings.defined() || !audio_valid_lengths.defined()) {
            throw std::runtime_error(
                "MiniCPM-o eval batch audio bounds require audio embeddings");
        }
        audio_valid_lengths = audio_valid_lengths.to(
            mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
        if (audio_valid_lengths.numel() != audio_embeddings.size(0)) {
            throw std::runtime_error(
                "MiniCPM-o eval batch audio length count is invalid");
        }
        const auto * valid_lengths = audio_valid_lengths.data_ptr<int64_t>();
        std::vector<bool> used_audio(
            static_cast<size_t>(audio_embeddings.size(0)), false);
        for (const auto & bound : audios) {
            const int64_t required = bound.end - bound.begin;
            if (bound.batch != 0 || bound.source < 0 ||
                    bound.source >= audio_embeddings.size(0) ||
                    bound.end > input_ids.size(1)) {
                throw std::runtime_error(
                    "MiniCPM-o eval audio bound does not match Whisper output");
            }
            const int64_t available = valid_lengths[bound.source];
            if (available < 0 || available > audio_embeddings.size(1) ||
                    required != available) {
                throw std::runtime_error(
                    "MiniCPM-o eval audio bound length does not match its source");
            }
            embeddings.index({
                0, Slice(bound.begin, bound.end), Slice()})
                .copy_(audio_embeddings.index({
                    bound.source, Slice(0, available), Slice()})
                    .to(embeddings.scalar_type()));
            used_audio[static_cast<size_t>(bound.source)] = true;
        }
        if (std::find(used_audio.begin(), used_audio.end(), false) !=
                used_audio.end()) {
            throw std::runtime_error(
                "MiniCPM-o eval batch has unused Whisper segments");
        }
    }
    return embeddings;
}

static bool minicpmo45_contains_token(
        const std::vector<int64_t> & values,
        int64_t token) {
    return std::find(values.begin(), values.end(), token) != values.end();
}

static std::vector<std::pair<int64_t, int64_t>>
minicpmo45_prefill_segments(
        mfq_tensor_backend::Tensor boundaries,
        int64_t range_begin,
        int64_t range_end,
        int64_t chunk_tokens) {
    if (range_begin < 0 || range_end <= range_begin || chunk_tokens < 0) {
        throw std::runtime_error(
            "MiniCPM-o prefill segment range is invalid");
    }
    boundaries = boundaries.to(
        mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
    if (boundaries.numel() == 0) {
        throw std::runtime_error(
            "MiniCPM-o segmented prefill requires boundaries");
    }
    const auto * values = boundaries.data_ptr<int64_t>();
    std::vector<int64_t> eligible;
    eligible.reserve(static_cast<size_t>(boundaries.numel()));
    bool found_begin = range_begin == 0;
    bool found_end = false;
    int64_t previous = 0;
    for (int64_t index = 0; index < boundaries.numel(); ++index) {
        const int64_t boundary = values[index];
        if (boundary <= previous) {
            throw std::runtime_error(
                "MiniCPM-o prefill boundaries must be increasing");
        }
        previous = boundary;
        if (boundary == range_begin) found_begin = true;
        if (boundary > range_begin && boundary <= range_end) {
            eligible.push_back(boundary);
        }
        if (boundary == range_end) found_end = true;
    }
    if (!found_begin || !found_end) {
        throw std::runtime_error(
            "MiniCPM-o segmented prefill omits a prompt span");
    }

    std::vector<std::pair<int64_t, int64_t>> segments;
    int64_t segment_begin = range_begin;
    for (const int64_t boundary : eligible) {
        if (chunk_tokens == 0 ||
                boundary == range_end ||
                boundary - segment_begin >= chunk_tokens) {
            segments.emplace_back(segment_begin, boundary);
            segment_begin = boundary;
        }
    }
    if (segments.empty() || segments.back().second != range_end) {
        throw std::runtime_error(
            "MiniCPM-o segmented prefill omits a prompt span");
    }
    return segments;
}

static std::vector<std::pair<int64_t, int64_t>>
minicpmo45_teacher_prefill_segments(
        mfq_tensor_backend::Tensor boundaries,
        int64_t range_end,
        int64_t text_begin,
        int64_t text_end,
        int64_t chunk_tokens) {
    if (range_end <= 0 || text_begin < 0 || text_end <= text_begin ||
            text_end > range_end || chunk_tokens < 0) {
        throw std::runtime_error(
            "MiniCPM-o teacher-forcing prefill range is invalid");
    }
    const auto exact = minicpmo45_prefill_segments(
        boundaries, 0, range_end, 0);
    const auto text = std::make_pair(text_begin, text_end);
    if (std::find(exact.begin(), exact.end(), text) == exact.end()) {
        throw std::runtime_error(
            "MiniCPM-o teacher-forcing splits omit the text span");
    }
    if (chunk_tokens == 0) return exact;

    std::vector<std::pair<int64_t, int64_t>> segments;
    const auto append = [&](const auto & parts) {
        segments.insert(segments.end(), parts.begin(), parts.end());
    };
    if (text_begin > 0) {
        append(minicpmo45_prefill_segments(
            boundaries, 0, text_begin, chunk_tokens));
    }
    segments.emplace_back(text);
    if (text_end < range_end) {
        append(minicpmo45_prefill_segments(
            boundaries, text_end, range_end, chunk_tokens));
    }
    return segments;
}

static int run_minicpmo45_eval_batch(
        const std::string & mfq_path,
        const std::string & config_path,
        int64_t context_size,
        int64_t vision_batch_size) {
    mfq_tensor_backend::NoGradGuard no_grad;
    g_profiler.enabled = false;
    if (context_size <= 0) context_size = 8192;
    if (vision_batch_size <= 0) vision_batch_size = 16;
    auto runtime = MiniCPMO45Runtime::load(
        mfq_path, config_path, context_size);
    std::cout << "minicpmo45_eval_batch=ready"
              << " context_size=" << context_size
              << " vision_batch_size=" << vision_batch_size << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto request = nlohmann::json::parse(line, nullptr, false);
        if (!request.is_object()) {
            throw std::runtime_error(
                "MiniCPM-o eval request must be a JSON object");
        }
        if (request.value("shutdown", false)) break;
        const std::string request_id =
            request.at("request_id").get<std::string>();
        const std::string input_prefix =
            request.at("input_prefix").get<std::string>();
        const std::string output_prefix =
            request.value("output_prefix", std::string{});
        const int64_t question_count =
            request.at("question_count").get<int64_t>();
        const int64_t prefix_length =
            request.at("common_prefix_length").get<int64_t>();
        const bool reuse_prefix_cache =
            request.value("reuse_prefix_cache", true);
        const bool require_segmented_prefill =
            request.value("require_segmented_prefill", false);
        const int64_t segmented_prefill_chunk_tokens =
            request.value("segmented_prefill_chunk_tokens", int64_t{0});
        const int64_t max_new_tokens =
            request.value("max_new_tokens", int64_t{128});
        const double repetition_penalty =
            request.value("repetition_penalty", 1.02);
        const bool capture_logits =
            request.value("capture_logits", false);
        const bool ignore_eos =
            request.value("ignore_eos", false);
        const bool generate_tts =
            request.value("generate_tts", false);
        const bool tts_teacher_forcing =
            request.value("tts_teacher_forcing", false);
        const int64_t tts_max_steps =
            request.value("tts_max_steps", int64_t{2048});
        const double tts_temperature =
            request.value("tts_temperature", 0.8);
        const double tts_top_p =
            request.value("tts_top_p", 0.85);
        const int64_t tts_top_k =
            request.value("tts_top_k", int64_t{25});
        const double tts_min_p =
            request.value("tts_min_p", 0.01);
        const double tts_repetition_penalty =
            request.value("tts_repetition_penalty", 1.05);
        const int64_t tts_minimum_steps =
            request.value("tts_minimum_steps", int64_t{0});
        const int64_t tts_min_tokens_to_keep =
            request.value("tts_min_tokens_to_keep", int64_t{3});
        const int64_t seed =
            request.value("seed", int64_t{0});
        const auto eos_token_ids = request.at("eos_token_ids")
            .get<std::vector<int64_t>>();
        if (question_count <= 0 || prefix_length < 0 ||
                (reuse_prefix_cache && prefix_length <= 0) ||
                max_new_tokens <= 0 || repetition_penalty <= 0.0 ||
                segmented_prefill_chunk_tokens < 0 ||
                (segmented_prefill_chunk_tokens > 0 &&
                 !require_segmented_prefill) ||
                eos_token_ids.empty()) {
            throw std::runtime_error(
                "MiniCPM-o eval request has invalid generation settings");
        }
        if (generate_tts && (question_count != 1 || tts_max_steps <= 0 ||
                tts_temperature <= 0.0 ||
                tts_top_p <= 0.0 || tts_top_p > 1.0 ||
                tts_top_k < 3 || tts_top_k > 6562 ||
                tts_min_p < 0.0 || tts_min_p > 1.0 ||
                tts_repetition_penalty <= 0.0 ||
                tts_minimum_steps < 0 || tts_min_tokens_to_keep < 0 ||
                tts_min_tokens_to_keep > tts_top_k)) {
            throw std::runtime_error(
                "MiniCPM-o TTS batch requires one question and positive steps");
        }
        if (tts_teacher_forcing && !generate_tts) {
            throw std::runtime_error(
                "MiniCPM-o teacher forcing requires TTS generation");
        }
        const auto wall_started = std::chrono::steady_clock::now();
        auto pixels = minicpmo45_load_tensor(
            input_prefix + ".pixel_values.pt", false);
        auto patch_mask = minicpmo45_load_tensor(
            input_prefix + ".patch_mask.pt", false);
        auto target_sizes = minicpmo45_load_tensor(
            input_prefix + ".target_sizes.pt", false);
        if (target_sizes.defined()) {
            target_sizes = target_sizes.to(
                mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        }
        auto audio_features = minicpmo45_load_tensor(
            input_prefix + ".audio_features.pt", false);
        auto audio_lengths = minicpmo45_load_tensor(
            input_prefix + ".audio_lengths.pt", false);
        if (pixels.defined() != patch_mask.defined() ||
                pixels.defined() != target_sizes.defined() ||
                (pixels.defined() &&
                 (pixels.dim() != 4 || pixels.size(1) != 3 ||
                  patch_mask.dim() != 2 || target_sizes.dim() != 2 ||
                  pixels.size(0) != patch_mask.size(0) ||
                  pixels.size(0) != target_sizes.size(0)))) {
            throw std::runtime_error(
                "MiniCPM-o eval tensors have incompatible shapes");
        }
        if (audio_features.defined() != audio_lengths.defined() ||
                (audio_features.defined() &&
                 (audio_features.dim() != 3 ||
                  audio_features.size(1) != 80 ||
                  audio_lengths.dim() != 1 ||
                  audio_lengths.size(0) != audio_features.size(0)))) {
            throw std::runtime_error(
                "MiniCPM-o eval batch audio tensors have incompatible shapes");
        }

        MFQ_CUDA_CHECK(cudaDeviceSynchronize());
        const auto vision_started = std::chrono::steady_clock::now();
        std::vector<mfq_tensor_backend::Tensor> vision_parts;
        std::vector<mfq_tensor_backend::Tensor> image_embedding_parts;
        mfq_tensor_backend::Tensor image_embeddings;
        if (pixels.defined() && pixels.size(0) > 0) {
            for (int64_t begin = 0; begin < pixels.size(0);
                    begin += vision_batch_size) {
                const int64_t count = std::min<int64_t>(
                    vision_batch_size, pixels.size(0) - begin);
                auto vision_part = runtime.vision.forward(
                    pixels.narrow(0, begin, count).to(mfq_tensor_backend::kCUDA),
                    patch_mask.narrow(0, begin, count),
                    target_sizes.narrow(0, begin, count));
                vision_parts.push_back(vision_part);
                image_embedding_parts.push_back(runtime.resampler.forward(
                    vision_part, target_sizes.narrow(0, begin, count)));
            }
            image_embeddings = mfq_tensor_backend::cat(
                image_embedding_parts, 0).contiguous();
        }
        mfq_tensor_backend::Tensor audio_embeddings;
        mfq_tensor_backend::Tensor audio_valid_lengths;
        double audio_seconds = 0.0;
        if (audio_features.defined()) {
            const auto audio_started = std::chrono::steady_clock::now();
            auto host_audio_lengths = audio_lengths.to(
                mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
            const auto * raw_lengths =
                host_audio_lengths.data_ptr<int64_t>();
            const auto valid_length_values =
                MiniCPMO45AudioEncoder::pooled_lengths(audio_lengths);
            audio_valid_lengths = mfq_tensor_backend::tensor(
                valid_length_values,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
            const int64_t max_valid_length = *std::max_element(
                valid_length_values.begin(), valid_length_values.end());
            std::vector<mfq_tensor_backend::Tensor> audio_parts;
            audio_parts.reserve(static_cast<size_t>(audio_features.size(0)));
            for (int64_t index = 0; index < audio_features.size(0); ++index) {
                runtime.audio.reset();
                auto part_features = audio_features.index({
                    Slice(index, index + 1), Slice(),
                    Slice(0, raw_lengths[index])}).to(mfq_tensor_backend::kCUDA);
                auto part_length = mfq_tensor_backend::tensor(
                    std::vector<int64_t>{raw_lengths[index]},
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
                auto part = runtime.audio.forward(
                    part_features, part_length, false);
                if (part.size(1) != valid_length_values[
                        static_cast<size_t>(index)]) {
                    throw std::runtime_error(
                        "MiniCPM-o per-segment Whisper length mismatch");
                }
                if (part.size(1) < max_valid_length) {
                    part = mfq_tensor_backend::constant_pad_nd(
                        part, {0, 0, 0, max_valid_length - part.size(1)}, 0.0);
                }
                audio_parts.push_back(part);
            }
            runtime.audio.reset();
            audio_embeddings = mfq_tensor_backend::cat(audio_parts, 0).contiguous();
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            audio_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - audio_started).count();
        }
        MFQ_CUDA_CHECK(cudaDeviceSynchronize());
        const double vision_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - vision_started).count();
        pixels = mfq_tensor_backend::Tensor();
        patch_mask = mfq_tensor_backend::Tensor();
        vision_parts.clear();
        image_embedding_parts.clear();

        const auto question_prefix = [&](int64_t index) {
            std::ostringstream value;
            value << input_prefix << ".q" << std::setw(3)
                  << std::setfill('0') << index;
            return value.str();
        };
        const auto first_prefix = question_prefix(0);
        auto first_ids = minicpmo45_load_tensor(
            first_prefix + ".input_ids.pt", true)
            .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        if (first_ids.dim() == 1) first_ids = first_ids.unsqueeze(0);
        auto first_bounds = minicpmo45_load_tensor(
            first_prefix + ".image_bounds.pt", false);
        auto first_audio_bounds = minicpmo45_load_tensor(
            first_prefix + ".audio_bounds.pt", false);
        auto first_prefill_splits = minicpmo45_load_tensor(
            first_prefix + ".prefill_splits.pt",
            require_segmented_prefill);
        auto first_embeddings = minicpmo45_eval_embeddings(
            runtime, first_ids, image_embeddings, first_bounds,
            audio_embeddings, first_audio_bounds, audio_valid_lengths);
        if (prefix_length > first_ids.size(1)) {
            throw std::runtime_error(
                "MiniCPM-o common prefix exceeds the first prompt");
        }
        double prefix_seconds = 0.0;
        int64_t prefix_prefill_calls = 0;
        if (reuse_prefix_cache) {
            runtime.language.reset(1);
            auto prefix_ids = first_ids.narrow(1, 0, prefix_length);
            auto prefix_embeddings = first_embeddings.narrow(
                1, 0, prefix_length);
            auto prefix_mask = mfq_tensor_backend::ones(
                {1, prefix_length},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kBool));
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            const auto prefix_started = std::chrono::steady_clock::now();
            mfq_tensor_backend::Tensor prefix_hidden;
            if (require_segmented_prefill) {
                const auto segments = minicpmo45_prefill_segments(
                    first_prefill_splits, 0, prefix_length,
                    segmented_prefill_chunk_tokens);
                for (const auto & segment : segments) {
                    const int64_t segment_begin = segment.first;
                    const int64_t segment_end = segment.second;
                    prefix_hidden = runtime.language.hidden_forward_inputs(
                        prefix_ids.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        prefix_embeddings.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt);
                    ++prefix_prefill_calls;
                }
            } else {
                prefix_hidden = runtime.language.hidden_forward_inputs(
                    prefix_ids, prefix_embeddings,
                    mfq_nullopt, mfq_nullopt, nullptr, prefix_mask);
                prefix_prefill_calls = 1;
            }
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            prefix_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - prefix_started).count();
            if (runtime.language.cache_pos != prefix_length) {
                throw std::runtime_error(
                    "MiniCPM-o prefix cache length mismatch");
            }
        }

        nlohmann::json results = nlohmann::json::array();
        for (int64_t question = 0; question < question_count; ++question) {
            const auto current_prefix = question_prefix(question);
            auto input_ids = question == 0
                ? first_ids
                : minicpmo45_load_tensor(
                    current_prefix + ".input_ids.pt", true)
                    .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
            auto image_bounds = question == 0
                ? first_bounds
                : minicpmo45_load_tensor(
                    current_prefix + ".image_bounds.pt", false);
            auto audio_bounds = question == 0
                ? first_audio_bounds
                : minicpmo45_load_tensor(
                    current_prefix + ".audio_bounds.pt", false);
            auto attention_mask = minicpmo45_load_tensor(
                current_prefix + ".attention_mask.pt", true)
                .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kBool).contiguous();
            auto tts_bound = minicpmo45_load_tensor(
                current_prefix + ".tts_bound.pt", tts_teacher_forcing);
            auto prefill_splits = minicpmo45_load_tensor(
                current_prefix + ".prefill_splits.pt",
                tts_teacher_forcing || require_segmented_prefill);
            if (input_ids.dim() == 1) input_ids = input_ids.unsqueeze(0);
            if (attention_mask.dim() == 1) {
                attention_mask = attention_mask.unsqueeze(0);
            }
            if (input_ids.size(0) != 1 ||
                    attention_mask.sizes() != input_ids.sizes() ||
                    (reuse_prefix_cache &&
                     prefix_length >= input_ids.size(1)) ||
                    input_ids.size(1) + max_new_tokens > context_size) {
                throw std::runtime_error(
                    "MiniCPM-o eval prompt exceeds the configured context");
            }
            auto input_embeddings = question == 0
                ? first_embeddings
                : minicpmo45_eval_embeddings(
                    runtime, input_ids, image_embeddings, image_bounds,
                    audio_embeddings, audio_bounds, audio_valid_lengths);
            if (reuse_prefix_cache) {
                runtime.language.cache_pos = prefix_length;
            } else {
                runtime.language.reset(1);
            }
            const int64_t evaluated_prefix_length =
                reuse_prefix_cache ? prefix_length : 0;
            const int64_t suffix_length =
                input_ids.size(1) - evaluated_prefix_length;
            auto suffix_ids = input_ids.narrow(
                1, evaluated_prefix_length, suffix_length);
            auto suffix_embeddings = input_embeddings.narrow(
                1, evaluated_prefix_length, suffix_length);
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            const auto generation_started = std::chrono::steady_clock::now();
            mfq_tensor_backend::Tensor hidden;
            mfq_tensor_backend::Tensor teacher_text_hidden;
            mfq_tensor_backend::Tensor logits;
            int64_t suffix_prefill_calls = 0;
            if (tts_teacher_forcing) {
                if (reuse_prefix_cache || evaluated_prefix_length != 0) {
                    throw std::runtime_error(
                        "MiniCPM-o teacher forcing cannot reuse a prefix cache");
                }
                tts_bound = tts_bound.to(
                    mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
                prefill_splits = prefill_splits.to(
                    mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
                if (tts_bound.numel() != 2 || prefill_splits.numel() < 2) {
                    throw std::runtime_error(
                        "MiniCPM-o teacher-forcing boundaries are invalid");
                }
                const int64_t text_begin = tts_bound[0].item<int64_t>();
                const int64_t text_end = tts_bound[1].item<int64_t>();
                const auto segments = minicpmo45_teacher_prefill_segments(
                    prefill_splits, input_ids.size(1), text_begin, text_end,
                    segmented_prefill_chunk_tokens);
                bool found_text = false;
                for (const auto & segment : segments) {
                    const int64_t segment_begin = segment.first;
                    const int64_t segment_end = segment.second;
                    auto segment_hidden = runtime.language.hidden_forward_inputs(
                        input_ids.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        input_embeddings.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt);
                    if (segment_begin == text_begin && segment_end == text_end) {
                        teacher_text_hidden = segment_hidden;
                        found_text = true;
                    }
                    hidden = segment_hidden;
                    ++suffix_prefill_calls;
                }
                if (!found_text) {
                    throw std::runtime_error(
                        "MiniCPM-o teacher-forcing splits omit the text span");
                }
            } else if (require_segmented_prefill) {
                const auto segments = minicpmo45_prefill_segments(
                    prefill_splits, evaluated_prefix_length,
                    input_ids.size(1), segmented_prefill_chunk_tokens);
                for (const auto & segment : segments) {
                    const int64_t segment_begin = segment.first;
                    const int64_t segment_end = segment.second;
                    hidden = runtime.language.hidden_forward_inputs(
                        input_ids.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        input_embeddings.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt);
                    ++suffix_prefill_calls;
                }
                logits = runtime.language.logits_from_hidden(
                    hidden.index({Slice(), -1, Slice()}));
            } else {
                hidden = runtime.language.hidden_forward_inputs(
                    suffix_ids, suffix_embeddings,
                    mfq_nullopt, mfq_nullopt, nullptr, attention_mask);
                suffix_prefill_calls = 1;
                logits = runtime.language.logits_from_hidden(
                    hidden.index({Slice(), -1, Slice()}));
            }
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            const double suffix_prefill_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - generation_started).count();
            const auto decode_started = std::chrono::steady_clock::now();
            double first_token_seconds = 0.0;
            auto counts = mfq_tensor_backend::zeros(
                {runtime.language.c.vocab_size},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt32));
            std::vector<int64_t> generated;
            std::vector<mfq_tensor_backend::Tensor> captured_logits;
            std::vector<mfq_tensor_backend::Tensor> generated_hidden;
            generated.reserve(static_cast<size_t>(max_new_tokens));
            if (capture_logits) {
                captured_logits.reserve(
                    static_cast<size_t>(max_new_tokens));
            }
            for (int64_t step = 0;
                    !tts_teacher_forcing && step < max_new_tokens; ++step) {
                if (capture_logits) {
                    captured_logits.push_back(
                        logits.detach().to(mfq_tensor_backend::kCPU));
                }
                auto scores = generated.empty() || repetition_penalty == 1.0
                    ? logits
                    : sample_apply_penalties_cuda(
                        logits, counts, 0.0, 0.0,
                        repetition_penalty);
                auto next = sample_greedy_cuda(
                    scores.contiguous().view({1, -1}));
                const int64_t token = next.item<int64_t>();
                if (step == 0) {
                    first_token_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - decode_started).count();
                }
                generated.push_back(token);
                if ((!ignore_eos &&
                        minicpmo45_contains_token(eos_token_ids, token)) ||
                        step + 1 == max_new_tokens) {
                    break;
                }
                sample_token_counts_add_cuda(counts, next.contiguous());
                auto next_hidden = runtime.language.hidden_forward(
                    next.reshape({1, 1}));
                generated_hidden.push_back(next_hidden);
                logits = runtime.language.logits_from_hidden(
                    next_hidden.index({Slice(), -1, Slice()}));
            }
            mfq_tensor_backend::Tensor tts_codes;
            if (generate_tts) {
                mfq_tensor_backend::Tensor text_ids;
                mfq_tensor_backend::Tensor text_hidden;
                if (tts_teacher_forcing) {
                    const int64_t begin =
                        tts_bound[0].item<int64_t>();
                    const int64_t end =
                        tts_bound[1].item<int64_t>();
                    if (begin < evaluated_prefix_length || end <= begin ||
                            end > input_ids.size(1)) {
                        throw std::runtime_error(
                            "MiniCPM-o teacher-forcing span is invalid");
                    }
                    text_ids = input_ids.narrow(1, begin, end - begin);
                    text_hidden = teacher_text_hidden;
                } else {
                    if (generated_hidden.empty()) {
                        throw std::runtime_error(
                            "MiniCPM-o TTS generation produced no text condition");
                    }
                    constexpr int64_t kTtsTextEosToken = 151704;
                    const auto text_eos = std::find(
                        generated.begin(), generated.end(), kTtsTextEosToken);
                    const int64_t text_length = std::min<int64_t>(
                        static_cast<int64_t>(generated_hidden.size()),
                        static_cast<int64_t>(
                            std::distance(generated.begin(), text_eos)));
                    if (text_length <= 0) {
                        throw std::runtime_error(
                            "MiniCPM-o TTS generation produced no text tokens");
                    }
                    text_ids = mfq_tensor_backend::tensor(
                        std::vector<int64_t>(
                            generated.begin(), generated.begin() + text_length),
                        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                            .dtype(mfq_tensor_backend::kInt64)).reshape({1, text_length});
                    text_hidden = mfq_tensor_backend::cat(generated_hidden, 1)
                        .narrow(1, 0, text_length).contiguous();
                }
                auto condition = runtime.tts.condition(
                    text_ids, text_hidden);
                std::mt19937 evaluator_rng(
                    static_cast<std::mt19937::result_type>(seed));
                tts_codes = runtime.tts.generate_official(
                    condition, tts_max_steps, 6561,
                    tts_minimum_steps,
                    tts_temperature, tts_top_p, tts_top_k,
                    tts_repetition_penalty, nullptr, tts_min_p,
                    &evaluator_rng, tts_min_tokens_to_keep)
                    .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).reshape({-1});
                if (!output_prefix.empty()) {
                    std::ostringstream path;
                    path << output_prefix << ".q" << std::setw(3)
                         << std::setfill('0') << question << ".tts_codes.pt";
                    minicpmo45_save_tensor(tts_codes, path.str());
                }
            }
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            const double generation_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - generation_started).count();
            const double decode_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - decode_started).count();
            if (capture_logits && !output_prefix.empty()) {
                std::ostringstream path;
                path << output_prefix << ".q" << std::setw(3)
                     << std::setfill('0') << question << ".logits.pt";
                minicpmo45_save_tensor(
                    mfq_tensor_backend::cat(captured_logits, 0), path.str());
            }
            results.push_back({
                {"question_index", question},
                {"context_tokens", input_ids.size(1)},
                {"prefix_tokens", evaluated_prefix_length},
                {"suffix_tokens", suffix_length},
                {"suffix_prefill_calls", suffix_prefill_calls},
                {"generated_token_ids", generated},
                {"generation_seconds", generation_seconds},
                {"suffix_prefill_seconds", suffix_prefill_seconds},
                {"first_token_seconds", first_token_seconds},
                {"decode_seconds", decode_seconds},
                {"tts_code_count", tts_codes.defined()
                    ? tts_codes.numel() : 0},
            });
            if (question == 0) {
                first_ids = mfq_tensor_backend::Tensor();
                first_bounds = mfq_tensor_backend::Tensor();
                first_audio_bounds = mfq_tensor_backend::Tensor();
                first_embeddings = mfq_tensor_backend::Tensor();
            }
        }
        MFQ_CUDA_CHECK(cudaDeviceSynchronize());
        const auto stats = mfq_cuda_memory_stats(mfq_current_cuda_device());
        nlohmann::json response = {
            {"request_id", request_id},
            {"status", "ok"},
            {"frame_count", image_embeddings.defined()
                ? image_embeddings.size(0) : 0},
            {"audio_count", audio_embeddings.defined()
                ? audio_embeddings.size(0) : 0},
            {"vision_seconds", vision_seconds},
            {"audio_seconds", audio_seconds},
            {"prefix_seconds", prefix_seconds},
            {"prefix_prefill_calls", prefix_prefill_calls},
            {"segmented_prefill_chunk_tokens",
                segmented_prefill_chunk_tokens},
            {"wall_seconds", std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wall_started).count()},
            {"allocated_bytes", stats.allocated_bytes},
            {"reserved_bytes", stats.reserved_bytes},
            {"peak_allocated_bytes", stats.peak_allocated_bytes},
            {"peak_reserved_bytes", stats.peak_reserved_bytes},
            {"questions", results},
        };
        std::cout << "minicpmo45_eval_result="
                  << response.dump() << std::endl;
    }
    return 0;
}

static int run_minicpmo45_composite(
        const std::string & mfq_path,
        const std::string & config_path,
        const std::string & input_prefix,
        const std::string & output_prefix,
        int64_t context_size,
        int64_t tts_steps) {
    mfq_tensor_backend::NoGradGuard no_grad;
    g_profiler.enabled = false;
    if (input_prefix.empty() || output_prefix.empty()) {
        throw std::runtime_error(
            "MiniCPM-o composite graph requires input and output prefixes");
    }
    auto runtime = MiniCPMO45Runtime::load(
        mfq_path, config_path, context_size);
    const auto input_ids = minicpmo45_load_tensor(
        input_prefix + ".input_ids.pt", true);
    const auto position_ids = minicpmo45_load_tensor(
        input_prefix + ".position_ids.pt", false);
    const auto attention_mask = minicpmo45_load_tensor(
        input_prefix + ".attention_mask.pt", false);
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
        input_ids, position_ids, attention_mask,
        pixels, patch_mask, target_sizes, image_bounds,
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
        mfq_tensor_backend::Tensor condition = minicpmo45_load_tensor(
            input_prefix + ".tts_inputs_embeds.pt", false);
        if (!condition.defined()) {
            auto bound = minicpmo45_load_tensor(
                input_prefix + ".tts_bound.pt", true)
                .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
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
            condition = condition.to(mfq_tensor_backend::kCUDA).contiguous();
        }
        std::vector<mfq_tensor_backend::Tensor> tts_logits;
        const auto codes = runtime.tts.generate_official(
            condition, tts_steps, 6561,
            50,
            0.8, 0.85, 25, 1.05, &tts_logits);
        minicpmo45_save_tensor(
            condition, output_prefix + ".tts_inputs_embeds.pt");
        minicpmo45_save_tensor(
            codes, output_prefix + ".tts_codes.pt");
        if (!tts_logits.empty()) {
            minicpmo45_save_tensor(
                mfq_tensor_backend::stack(tts_logits, 1),
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
