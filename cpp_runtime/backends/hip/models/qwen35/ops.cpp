#include "hip/hip_runtime.h"
#include "storage/weight_loader.h"
#include "ops.h"
#include "storage/session_codec.h"
#include "linear_attention.h"

#include "storage/transformer_loader.h"

namespace mfq::cuda::qwen35 {
struct BlockLoader : TransformerWeightLoader {
    using Ffn = FFN;
    auto full_block() const {
        auto block=std::make_unique<FullBlock>();block->kv_quantization=cuda_kv_quantization();return block;
    }
    auto linear_block() const { return std::make_unique<LinearAttentionBlock>(); }
    template <class T> static std::unique_ptr<::Block> finish(std::unique_ptr<T> block) {
        return block;
    }
    static auto log_negative(mfq_tensor_backend::Tensor value) { return mfq_tensor_backend::log(-value); }
    void linear_projections(LinearAttentionBlock& b,
        const mfq::models::qwen35::LinearWeightNames& names, bool split_qkv) const {
        if (split_qkv) {
            b.split_in_proj = true;
            b.qkv_proj =
                load_quant_group(execution, source, {names.qk, names.value});
            if (is_quant_dtype(require_tensor(source, names.gate).dtype)) {
                b.z_proj = load_quant_linear(execution, source, names.gate);
                const bool a_nint = is_quant_dtype(require_tensor(source, names.alpha).dtype);
                const bool b_nint = is_quant_dtype(require_tensor(source, names.beta).dtype);
                if (a_nint != b_nint)
                    throw std::runtime_error("linear_attn a/b must use the same storage kind");
                b.ab_is_nint = a_nint;
                if (b.ab_is_nint) {
                    const auto scalar_axis =
                        execution.config.tensor_parallel_mirror_linear_attention_scalars
                            ? std::optional<TensorParallelAxis>(TensorParallelAxis::Mirrored)
                            : std::nullopt;
                    b.ab_nint_proj = make_quant_group(
                        execution,
                        {
                            load_quant_linear(execution, source, names.alpha, scalar_axis),
                            load_quant_linear(execution, source, names.beta, scalar_axis),
                        });
                } else {
                    b.ab_proj = make_dense_group({
                        load_dense_gpu(execution, source, names.alpha),
                        load_dense_gpu(execution, source, names.beta),
                    });
                }
            } else {
                b.split_dense_zab = true;
                b.zab_proj = make_dense_group({
                    load_dense_gpu(execution, source, names.gate),
                    load_dense_gpu(execution, source, names.alpha),
                    load_dense_gpu(execution, source, names.beta),
                });
            }
        } else {
            const bool alpha_quant = is_quant_dtype(require_tensor(source, names.alpha).dtype);
            const bool beta_quant = is_quant_dtype(require_tensor(source, names.beta).dtype);
            if (alpha_quant != beta_quant) {
                throw std::runtime_error(
                    "linear_attention alpha/beta must use the same storage kind");
            }
            if (alpha_quant) {
                b.in_proj = load_quant_group(
                    execution, source,
                    {names.qkv, names.gate, names.alpha, names.beta});
            } else {
                b.dense_ab_tail = true;
                b.in_proj =
                    load_quant_group(execution, source, {names.qkv, names.gate});
                b.ab_proj = make_dense_group({
                    load_dense_gpu(execution, source, names.alpha),
                    load_dense_gpu(execution, source, names.beta),
                });
            }
        }
    }
    void linear_output(LinearAttentionBlock& b, const std::string& name) const {
        if (is_quant_dtype(require_tensor(source, name).dtype)) {
            b.out_proj = load_quant_linear(execution, source, name);
        } else {
            b.dense_out_proj = true;
            b.out_proj_dense = load_dense_gpu(execution, source, name);
            if (require_tensor(source, name).dtype == "F16") {
                b.out_proj_dense = b.out_proj_dense.to(mfq_tensor_backend::kFloat16).contiguous();
            }
            if (b.out_proj_dense.dim() != 2) {
                throw std::runtime_error("dense linear_attention output projection must be 2D");
            }
        }
    }
};

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &source,
                                    const Config &config, int layer, const std::string &type,
                                    std::string_view tensor_root) {
    BlockLoader loader{{execution, source}};
    return mfq::models::qwen35::load_block(loader, config, layer, type, tensor_root);
}

void LinearAttentionBlock::clear_speculative() noexcept {
    speculative_recurrent = LinearRecurrentInputs();
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_tokens = 0;
    speculative_pending = false;
}

mfq_tensor_backend::Tensor LinearAttentionBlock::forward_context(CudaExecutionContext &execution,
                                                                 mfq_tensor_backend::Tensor input,
                                                                 const Block::Context &context,
                                                                 const RopeCache &rope) {
    if (context.confirmed_prefix == 0) {
        return Block::forward_context(execution, std::move(input), context, rope);
    }
    MFQ_RUNTIME_CHECK(input.is_cuda() && !speculative_pending && context.confirmed_prefix > 0 &&
                          context.confirmed_prefix < input.size(1) && conv_state.defined() &&
                          gdn_state.defined(),
                      "invalid Qwen3.5 speculative linear-attention transaction");

    if (!speculative_conv.defined() || speculative_conv.sizes() != conv_state.sizes() ||
        !speculative_gdn.defined() || speculative_gdn.sizes() != gdn_state.sizes()) {
        speculative_conv = conv_state.clone();
        speculative_gdn = gdn_state.clone();
    } else {
        speculative_conv.copy_(conv_state);
        speculative_gdn.copy_(gdn_state);
    }
    speculative_start = context.cache_position;
    speculative_confirmed = context.confirmed_prefix;
    speculative_tokens = input.size(1);
    speculative_pending = true;

    try {
        // Match Metal: evaluate the complete [confirmed, drafts...] window
        // once so projections and FFN stay batched. Rollback replays only the
        // recurrent state prefix from this layer's retained projections.
        auto attention =
            forward_attention_cuda(execution, std::move(input), &speculative_recurrent);
        auto result = forward_ffn_cuda(execution, std::move(attention[0]), std::move(attention[1]));
        ++speculative_projection_batches;
        ++speculative_ffn_batches;
        return result;
    } catch (...) {
        conv_state.copy_(speculative_conv);
        gdn_state.copy_(speculative_gdn);
        clear_speculative();
        throw;
    }
}

void LinearAttentionBlock::commit_speculative() noexcept { clear_speculative(); }

void LinearAttentionBlock::rollback_speculative(int64_t keep_position) {
    MFQ_RUNTIME_CHECK(speculative_pending &&
                          keep_position >= speculative_start + speculative_confirmed &&
                          keep_position < speculative_start + speculative_tokens,
                      "invalid Qwen3.5 speculative rollback position");
    const int64_t retained = keep_position - speculative_start;
    conv_state.copy_(speculative_conv);
    gdn_state.copy_(speculative_gdn);
    try {
        // Restore only convolution/GDN state from the retained projected
        // rows; target projections, output projection and FFN are not rerun.
        replay_recurrent_cuda(speculative_recurrent, retained);
        clear_speculative();
    } catch (...) {
        clear_speculative();
        throw;
    }
}

bool supports_text_session_state(const std::vector<std::unique_ptr<::Block>> &blocks) {
    return !blocks.empty() &&
           std::all_of(blocks.begin(), blocks.end(), [](const std::unique_ptr<::Block> &block) {
               return dynamic_cast<const FullBlock *>(block.get()) != nullptr ||
                      dynamic_cast<const LinearAttentionBlock *>(block.get()) != nullptr;
           });
}

TextSessionState capture_text_session_state(const std::vector<std::unique_ptr<::Block>> &blocks,
                                            const std::vector<std::int64_t> &tokens,
                                            std::int64_t cache_position) {
    if (!supports_text_session_state(blocks) || cache_position <= 0 ||
        static_cast<std::size_t>(cache_position) != tokens.size()) {
        throw std::runtime_error("Qwen hybrid session token count does not match the cache");
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_position;
    state.payload = std::vector<HybridBlockSessionState>{};
    auto &layers = std::get<std::vector<HybridBlockSessionState>>(state.payload);
    layers.reserve(blocks.size());
    for (const auto &block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        HybridBlockSessionState saved;
        if (const auto *full = dynamic_cast<const FullBlock *>(block.get())) {
            saved.kind = HybridBlockSessionStateKind::FullAttention;
            saved.full_attention =
                capture_full_attention_session_state(*full, cache_position, state.bytes);
        } else if (const auto *linear = dynamic_cast<const LinearAttentionBlock *>(block.get())) {
            if (linear->speculative_pending || !linear->conv_state.defined() ||
                !linear->gdn_state.defined()) {
                throw std::runtime_error("Qwen recurrent session state is unavailable");
            }
            saved.kind = HybridBlockSessionStateKind::Recurrent;
            saved.convolution_state = linear->conv_state.clone();
            saved.recurrent_state = linear->gdn_state.clone();
            state.bytes += session_tensor_bytes(saved.convolution_state);
            state.bytes += session_tensor_bytes(saved.recurrent_state);
        } else {
            throw std::runtime_error("Qwen hybrid session layer type changed");
        }
        layers.push_back(std::move(saved));
    }
    return state;
}

void restore_text_session_state(std::vector<std::unique_ptr<::Block>> &blocks,
                                const TextSessionState &state) {
    const auto *layers = std::get_if<std::vector<HybridBlockSessionState>>(&state.payload);
    if (state.kind() != TextSessionStateKind::HybridAttention || state.cache_pos <= 0 ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_pos) || layers == nullptr ||
        layers->size() != blocks.size()) {
        throw CudaSessionStateError("Qwen hybrid session state is incompatible");
    }
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        auto &block = blocks[index];
        const auto &saved = (*layers)[index];
        MfqCudaGuard guard(block->cuda_device);
        if (saved.kind == HybridBlockSessionStateKind::FullAttention) {
            auto *full = dynamic_cast<FullBlock *>(block.get());
            if (full == nullptr) {
                throw CudaSessionStateError("Qwen hybrid full-attention layer changed");
            }
            restore_full_attention_session_state(*full, saved.full_attention);
            continue;
        }
        auto *linear = dynamic_cast<LinearAttentionBlock *>(block.get());
        const auto convolution_width = linear != nullptr ? 2 * linear->qwen_config.linear_k_size() +
                                                               linear->qwen_config.linear_v_size()
                                                         : 0;
        if (linear == nullptr || !saved.convolution_state.defined() ||
            !saved.recurrent_state.defined() ||
            saved.convolution_state.scalar_type() != mfq_tensor_backend::kFloat32 ||
            saved.recurrent_state.scalar_type() != mfq_tensor_backend::kFloat32 ||
            saved.convolution_state.dim() != 3 || saved.convolution_state.size(0) != 1 ||
            saved.convolution_state.size(1) != linear->qwen_config.linear_conv_kernel_dim - 1 ||
            saved.convolution_state.size(2) != convolution_width ||
            saved.recurrent_state.dim() != 4 || saved.recurrent_state.size(0) != 1 ||
            saved.recurrent_state.size(1) != linear->qwen_config.linear_num_value_heads ||
            saved.recurrent_state.size(2) != linear->qwen_config.linear_value_head_dim ||
            saved.recurrent_state.size(3) != linear->qwen_config.linear_value_head_dim ||
            !saved.convolution_state.is_cuda() || !saved.recurrent_state.is_cuda() ||
            saved.convolution_state.get_device() != block->cuda_device ||
            saved.recurrent_state.get_device() != block->cuda_device) {
            throw CudaSessionStateError("Qwen recurrent session topology changed");
        }
        restore_session_tensor(linear->conv_state, saved.convolution_state);
        restore_session_tensor(linear->gdn_state, saved.recurrent_state);
        linear->speculative_conv = mfq_tensor_backend::Tensor();
        linear->speculative_gdn = mfq_tensor_backend::Tensor();
        linear->clear_speculative();
    }
}

} // namespace mfq::cuda::qwen35

namespace mfq::cuda {

void Qwen35Model::adapter_validate_components(const mfq::ModelGraph &graph) const {
    if (graph.component("vision") != nullptr &&
        cuda_model_plan(graph).vision != CudaVisionAdapter::grid_vit) {
        throw std::runtime_error("Qwen CUDA vision requires grid_vit/grid_vision.v1/grid_mrope");
    }
}

void Qwen35Model::adapter_configure_rope(RopeCache &rope, mfq_tensor_backend::Device device) const {
    rope.configure_mrope(config.mrope_sections, config.mrope_interleaved, config.rotary_dim,
                         device);
}

bool Qwen35Model::adapter_uses_common_rope() const noexcept { return true; }

bool Qwen35Model::adapter_supports_dense_cpu_offload() const noexcept { return true; }

std::unique_ptr<Block> Qwen35Model::adapter_load_block(const mfq::ModelSource &source, int layer,
                                                       int, const std::string &type) {
    return qwen35::load_block(*execution, source, config, layer, type);
}

TextSessionStateKind CudaSessionCodec<Qwen35Model>::kind(const Model &model) {
    const auto full = FullAttentionSessionCodec<Qwen35Model>::kind(model);
    if (full != TextSessionStateKind::Unsupported)
        return full;
    return qwen35::supports_text_session_state(model.blocks) ? TextSessionStateKind::HybridAttention
                                                             : TextSessionStateKind::Unsupported;
}

bool CudaSessionCodec<Qwen35Model>::supports_paged(const Model &model) {
    return FullAttentionSessionCodec<Qwen35Model>::supports_paged(model);
}

TextSessionState CudaSessionCodec<Qwen35Model>::capture(const Model &model,
                                                        const std::vector<int64_t> &tokens) {
    if (kind(model) == TextSessionStateKind::HybridAttention) {
        if (model.cache_pos <= 0 || static_cast<size_t>(model.cache_pos) != tokens.size()) {
            throw std::runtime_error("text session token count does not match the model cache");
        }
        return qwen35::capture_text_session_state(model.blocks, tokens, model.cache_pos);
    }
    return FullAttentionSessionCodec<Qwen35Model>::capture(model, tokens);
}

void CudaSessionCodec<Qwen35Model>::restore(Model &model, const TextSessionState &state) {
    if (kind(model) == TextSessionStateKind::HybridAttention &&
        state.kind() == TextSessionStateKind::HybridAttention) {
        qwen35::restore_text_session_state(model.blocks, state);
        model.cache_pos = state.cache_pos;
        return;
    }
    FullAttentionSessionCodec<Qwen35Model>::restore(model, state);
}

mfq_tensor_backend::Tensor Qwen35Model::adapter_embed(mfq_tensor_backend::Tensor output) const {
    return output.to(mfq_tensor_backend::kFloat16).contiguous();
}

bool Qwen35Model::adapter_supports_prepared_prompt() const noexcept { return true; }

} // namespace mfq::cuda

namespace mfq::cuda {

template struct FullAttentionSessionCodec<Qwen35Model>;

} // namespace mfq::cuda

namespace mfq::models {
template struct qwen35::CausalLm<cuda::CudaCausalOps<cuda::Qwen35Model>>;
} // namespace mfq::models
