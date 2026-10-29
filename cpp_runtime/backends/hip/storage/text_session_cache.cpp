#include "mfq_hip_kv_quantization.h"
#include "models/nemotron_h/ops.h"
#include "hip/hip_runtime.h"
#include "storage/text_session_cache.h"
#include "core/full_block.h"
#include <array>
#include <cstring>

#include "hip_runtime_config.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"
#include "storage/session_state.h"
#include "core/mtp.h"
#include "mfq_paged_prefix_cache.h"
#include "paged_session_bindings.h"
#include "session_cache.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mfq::cuda::internal {

static std::string cuda_prefix_cache_compatibility_key(
    const mfq::ModelSource &source, int64_t max_position_embeddings) {
    std::ostringstream key;
    key << "mfq-cuda-prefix-v1\n"
        << "codec=" << (source.architecture()=="qwen4_exp" ? "cuda-qwen4-session-v1" :
            source.architecture()=="deepseek_v41" ? "cuda-deepseek-v41-session-v1" : "cuda-full-attention-kv-v1") << '\n'
        << "context=" << max_position_embeddings << '\n'
        << "architecture=" << source.architecture() << '\n';
    const auto quantization=mfq::cuda::cuda_kv_quantization_tag();
    if(!quantization.empty())key<<"kv="<<quantization<<'\n';
    for (std::size_t index = 0; index < source.source_paths().size(); ++index) {
        const auto &path = source.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source for prefix cache: " + error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error("cannot identify model source timestamp: " + error.message());
        }
        const auto modified_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(modified.time_since_epoch())
                .count();
        key << "source=" << index << ':' << path.filename().string() << ':' << size << ':'
            << modified_ns << '\n';
    }
    for (const auto &tensor : source.tensors()) {
        key << "tensor=" << tensor.name << ':' << tensor.dtype << ':' << tensor.nbytes << '\n';
    }
    return key.str();
}

std::shared_ptr<mfq::cache::PagedPrefixCache> make_cuda_paged_prefix_cache(
    const mfq::ModelSource &source, int64_t max_position_embeddings,
    bool supports_paged_text_session_state, const mfq::engine::PrefixCacheConfig &config) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if (!config.enabled ||
        (format != source.metadata().end() && format->second == "hf-safetensors") ||
        !supports_paged_text_session_state) {
        return {};
    }
    mfq::cache::PagedPrefixCacheConfig cache;
    cache.cache_dir = config.directory;
    if (!source.source_paths().empty()) cache.model_path = source.source_paths().front();
    cache.compatibility_key = cuda_prefix_cache_compatibility_key(source, max_position_embeddings);
    cache.block_size_tokens = static_cast<size_t>(config.block_tokens);
    cache.retain_tail_versions = 0;
    cache.max_disk_bytes = config.disk_bytes;
    cache.max_hot_bytes = config.hot_bytes;
    cache.max_pending_writes = config.pending_writes;
    cache.max_pending_bytes = config.pending_bytes;
    return std::make_shared<mfq::cache::PagedPrefixCache>(std::move(cache));
}

struct CudaSessionOps {
    using Snapshot = TextSessionState;
    using Restore = TextSessionRestore;
    using Predictor = MtpModule;
    using StateError = CudaSessionStateError;
    template <class Payloads>
    static auto decode(
        const Payloads &payloads, const std::vector<int64_t> &tokens, size_t block_size) {
        if(payloads.empty())throw CudaSessionStateError("empty HIP prefix chain");
        if(payloads.front() && payloads.front()->size()>6 && (*payloads.front())[6]=='2')
            return decode_cuda_qwen4_paged_session(payloads,tokens,block_size);
        if(payloads.front() && payloads.front()->size()>6 && (*payloads.front())[6]=='3')
            return decode_cuda_deepseek_v41_paged_session(payloads,tokens,block_size);
        return decode_cuda_paged_session(payloads, tokens, block_size);
    }
    static auto encode(const Snapshot &state, size_t block_size, size_t index) {
        if(state.kind()==TextSessionStateKind::Qwen4)return encode_cuda_qwen4_paged_block(state,block_size,index);
        if(state.kind()==TextSessionStateKind::DeepseekV41)return encode_cuda_deepseek_v41_paged_block(state,block_size,index);
        return encode_cuda_paged_block(state, block_size, index);
    }
    static bool paged_mtp(const Snapshot& state) {
        return state.kind()==TextSessionStateKind::Qwen4 || state.kind()==TextSessionStateKind::DeepseekV41;
    }
    static bool partial_blocks(const Snapshot& state) { return paged_mtp(state); }
    static bool paged_media(const Snapshot& state) { return state.kind()==TextSessionStateKind::DeepseekV41; }
    static bool checkpoint_needs_update(const CudaPagedPayload& payload,const Snapshot& state) {
        return cuda_paged_checkpoint_needs_update(payload,state);
    }
    static size_t decodable_blocks(const std::vector<CudaPagedPayload>& payloads) { return cuda_paged_decodable_blocks(payloads); }
    static size_t token_count(const CudaPagedPayload& payload) { return cuda_paged_payload_tokens(payload); }
    static void release_host_cache() { mfq_release_host_allocator_cache(); }
};

struct TextSessionCache::Impl : mfq::engine::SessionCache<CudaSessionOps> {
    using SessionCache::SessionCache;
};

TextSessionCache::TextSessionCache(const mfq::engine::SessionCacheConfig &session_config,
    const mfq::engine::PrefixCacheConfig &prefix_config,
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache, bool supported, int disabled_reason)
    : impl_(std::make_unique<Impl>(
          session_config, prefix_config, std::move(paged_cache), supported, disabled_reason)) {}

TextSessionCache::~TextSessionCache() = default;

bool TextSessionCache::persistent_prefix_enabled() const noexcept {
    return impl_->persistent_prefix_enabled();
}

template <typename Model>
TextSessionRestore TextSessionCache::restore_best(Model &model, MtpModule *mtp,
    const std::string &requested_session, const std::vector<int64_t> &prompt,
    size_t maximum_prefix_tokens, const std::string &input_key) {
    return impl_->restore_best(
        model, mtp, requested_session, prompt, maximum_prefix_tokens, input_key);
}

void TextSessionCache::store(const std::string &session_id, TextSessionState state) {
    impl_->store(session_id, std::move(state));
}

size_t TextSessionCache::fork_session(
    const std::string &source_session, const std::string &target_session) {
    return impl_->fork_session(source_session, target_session);
}

size_t TextSessionCache::close_session(const std::string &session_id) {
    return impl_->close_session(session_id);
}

std::vector<std::pair<std::string, double>> TextSessionCache::metrics() const {
    return impl_->metrics();
}

size_t TextSessionCache::clear_live_sessions() noexcept { return impl_->clear_live_sessions(); }

size_t TextSessionCache::clear() { return impl_->clear(); }

uint64_t TextSessionCache::trim_hot(uint64_t target_bytes) { return impl_->trim_hot(target_bytes); }

#define MFQ_INSTANTIATE_SESSION_CACHE(MODEL)                                                       \
    template TextSessionRestore TextSessionCache::restore_best(MODEL &,                            \
        MtpModule *,                                                                               \
        const std::string &,                                                                       \
        const std::vector<int64_t> &,                                                              \
        size_t,                                                                                    \
        const std::string &);

MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Qwen35CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::MiniCPMO45CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::MiniCPMOTtsCausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Gemma4CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::GlmDsaCausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Glm5CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Qwen4CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::DeepseekV4CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::DeepseekV41CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::NemotronHCausalLm)

#undef MFQ_INSTANTIATE_SESSION_CACHE

} // namespace mfq::cuda::internal

void CudaPagedWriter::raw(const void *data, size_t size) {
    const auto *begin = static_cast<const uint8_t *>(data);
    bytes_.insert(bytes_.end(), begin, begin + size);
}

void CudaPagedWriter::tensor(const mfq_tensor_backend::Tensor &source) {
    if (!source.defined() || !source.is_cuda() || source.dim() <= 0 || source.dim() > 8) {
        throw std::runtime_error("invalid CUDA paged cache tensor");
    }
    const auto dtype = source.scalar_type();
    if (dtype != mfq_tensor_backend::kFloat16 && dtype != mfq_tensor_backend::kBFloat16 &&
        dtype != mfq_tensor_backend::kFloat32 && dtype != mfq_tensor_backend::kInt32 && dtype != mfq_tensor_backend::kInt64) {
        throw std::runtime_error("unsupported CUDA paged cache tensor dtype");
    }
    const auto cpu = source.contiguous().to(mfq_tensor_backend::kCPU);
    scalar<int32_t>(static_cast<int32_t>(dtype));
    scalar<int32_t>(source.get_device());
    scalar<uint32_t>(static_cast<uint32_t>(cpu.dim()));
    scalar<uint32_t>(0);
    for (const auto dimension : cpu.sizes())
        scalar<int64_t>(dimension);
    const auto bytes = static_cast<uint64_t>(cpu.numel() * cpu.element_size());
    scalar<uint64_t>(bytes);
    raw(cpu.data_ptr(), static_cast<size_t>(bytes));
}

CudaPagedPayload CudaPagedWriter::finish() && {
    return std::make_shared<const std::vector<uint8_t>>(std::move(bytes_));
}

void CudaPagedReader::raw(void *destination, size_t size, const char *name) {
    if (remaining() < size) {
        throw CudaSessionStateError(std::string("truncated CUDA paged cache ") + name);
    }
    std::memcpy(destination, cursor_, size);
    cursor_ += size;
}

std::pair<mfq_tensor_backend::Tensor, int> CudaPagedReader::tensor() {
    const auto dtype_value = scalar<int32_t>("dtype");
    const auto device = scalar<int32_t>("device");
    const auto rank = scalar<uint32_t>("rank");
    (void)scalar<uint32_t>("tensor flags");
    if (rank == 0 || rank > 8 || device < 0) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor header");
    }
    const auto dtype = static_cast<mfq_tensor_backend::ScalarType>(dtype_value);
    if (dtype != mfq_tensor_backend::kFloat16 && dtype != mfq_tensor_backend::kBFloat16 &&
        dtype != mfq_tensor_backend::kFloat32 && dtype != mfq_tensor_backend::kInt32 && dtype != mfq_tensor_backend::kInt64) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor dtype");
    }
    std::vector<int64_t> shape;
    shape.reserve(rank);
    uint64_t elements = 1;
    for (uint32_t index = 0; index < rank; ++index) {
        const auto dimension = scalar<int64_t>("shape");
        if (dimension <= 0 ||
            elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension)) {
            throw CudaSessionStateError("invalid CUDA paged cache tensor shape");
        }
        shape.push_back(dimension);
        elements *= static_cast<uint64_t>(dimension);
    }
    const auto bytes = scalar<uint64_t>("tensor size");
    const uint64_t element_size = dtype == mfq_tensor_backend::kInt64 ? 8 :
        (dtype == mfq_tensor_backend::kFloat32 || dtype == mfq_tensor_backend::kInt32) ? 4 : 2;
    if (elements > std::numeric_limits<uint64_t>::max() / element_size ||
        bytes != elements * element_size || bytes > remaining()) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor size");
    }
    auto cpu = mfq_tensor_backend::empty(
        shape, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(dtype));
    raw(cpu.data_ptr(), static_cast<size_t>(bytes), "tensor");
    return {std::move(cpu), device};
}

void CudaPagedReader::skip(size_t bytes,const char* name) {
    if(bytes>remaining())throw CudaSessionStateError(std::string("truncated HIP paged cache ")+name);
    cursor_+=bytes;
}

void CudaPagedReader::expect_end() const {
    if (cursor_ != end_) {
        throw CudaSessionStateError("trailing CUDA paged cache payload bytes");
    }
}

size_t CudaPagedReader::remaining() const { return static_cast<size_t>(end_ - cursor_); }



std::vector<CudaPagedPayload> encode_cuda_paged_session(const TextSessionState &state,
                                                        size_t block_size, size_t first_block,
                                                        size_t maximum_blocks) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    const auto *layers = std::get_if<std::vector<FullBlockSessionState>>(&state.payload);
    // Paged prefixes are text-only; media positions are not block-sliceable.
    if (state.kind() != TextSessionStateKind::FullAttention || state.decode_position_delta != 0 ||
        !state.input_key.empty() || state.cache_pos <= 0 || layers == nullptr || layers->empty() ||
        state.tokens.size() != static_cast<size_t>(state.cache_pos)) {
        throw std::runtime_error("CUDA session state is not block-sliceable");
    }
    const size_t full_blocks = state.tokens.size() / block_size;
    if (first_block > full_blocks) {
        throw std::runtime_error("invalid CUDA cache first block");
    }
    const size_t end_block =
        maximum_blocks > full_blocks - first_block ? full_blocks : first_block + maximum_blocks;
    std::vector<CudaPagedPayload> result;
    result.reserve(end_block - first_block);
    for (size_t block_index = first_block; block_index < end_block; ++block_index) {
        const int64_t start = static_cast<int64_t>(block_index * block_size);
        CudaPagedWriter writer;
        writer.raw(magic.data(), magic.size());
        writer.scalar<uint32_t>(1);
        writer.scalar<uint32_t>(static_cast<uint32_t>(start));
        writer.scalar<uint32_t>(static_cast<uint32_t>(block_size));
        writer.scalar<uint32_t>(static_cast<uint32_t>(layers->size()));
        for (const auto &layer : *layers) {
            if (layer.ring || layer.capacity <= 0 || !layer.k.defined() || !layer.v.defined() ||
                !session_kv_geometry_matches(layer.k,layer.v) ||
                layer.k.size(2) < start + static_cast<int64_t>(block_size)) {
                throw std::runtime_error("CUDA session layer cannot be block-sliced");
            }
            writer.scalar<int64_t>(layer.capacity);
            writer.tensor(layer.k.narrow(2, start, static_cast<int64_t>(block_size)));
            writer.tensor(layer.v.narrow(2, start, static_cast<int64_t>(block_size)));
        }
        result.push_back(std::move(writer).finish());
    }
    return result;
}

CudaPagedPayload encode_cuda_paged_block(const TextSessionState &state, size_t block_size,
                                         size_t block_index) {
    auto result = encode_cuda_paged_session(state, block_size, block_index, 1);
    if (result.size() != 1) {
        throw std::runtime_error("invalid CUDA cache block index");
    }
    return std::move(result.front());
}

CudaDecodedPagedBlock decode_cuda_paged_block(const std::vector<uint8_t> &payload) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    CudaPagedReader reader(payload);
    std::array<uint8_t, 8> found_magic{};
    reader.raw(found_magic.data(), found_magic.size(), "magic");
    if (found_magic != magic || reader.scalar<uint32_t>("version") != 1) {
        throw CudaSessionStateError("unsupported CUDA paged cache payload");
    }
    CudaDecodedPagedBlock block;
    block.start = reader.scalar<uint32_t>("start");
    block.count = reader.scalar<uint32_t>("count");
    const auto layer_count = reader.scalar<uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw CudaSessionStateError("invalid CUDA paged cache geometry");
    }
    block.layers.reserve(layer_count);
    for (uint32_t index = 0; index < layer_count; ++index) {
        CudaDecodedPagedLayer layer;
        layer.capacity = reader.scalar<int64_t>("capacity");
        auto key = reader.tensor();
        auto value = reader.tensor();
        if (key.second != value.second || !session_kv_geometry_matches(key.first,value.first) ||
            key.first.size(2) != static_cast<int64_t>(block.count)) {
            throw CudaSessionStateError("invalid CUDA paged cache layer");
        }
        layer.device = key.second;
        layer.k = std::move(key.first);
        layer.v = std::move(value.first);
        block.layers.push_back(std::move(layer));
    }
    reader.expect_end();
    return block;
}

TextSessionState decode_cuda_paged_session(const std::vector<CudaPagedPayload> &payloads,
                                           const std::vector<int64_t> &tokens, size_t block_size) {
    if (payloads.empty() || tokens.size() != payloads.size() * block_size) {
        throw CudaSessionStateError("CUDA paged cache chain length mismatch");
    }
    std::vector<CudaDecodedPagedBlock> blocks;
    blocks.reserve(payloads.size());
    size_t expected_start = 0;
    size_t layer_count = 0;
    for (const auto &payload : payloads) {
        if (!payload) {
            throw CudaSessionStateError("CUDA paged cache block payload is null");
        }
        auto block = decode_cuda_paged_block(*payload);
        if (block.start != expected_start || block.count != block_size ||
            (layer_count != 0 && block.layers.size() != layer_count)) {
            throw CudaSessionStateError("incompatible CUDA paged cache chain");
        }
        expected_start += block.count;
        layer_count = block.layers.size();
        blocks.push_back(std::move(block));
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = static_cast<int64_t>(tokens.size());
    state.payload = std::vector<FullBlockSessionState>{};
    auto &layers = std::get<std::vector<FullBlockSessionState>>(state.payload);
    layers.reserve(layer_count);
    for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        const auto &final = blocks.back().layers[layer_index];
        std::vector<mfq_tensor_backend::Tensor> keys;
        std::vector<mfq_tensor_backend::Tensor> values;
        keys.reserve(blocks.size());
        values.reserve(blocks.size());
        for (const auto &block : blocks) {
            const auto &layer = block.layers[layer_index];
            if (layer.capacity != final.capacity || layer.device != final.device ||
                layer.k.scalar_type() != final.k.scalar_type() ||
                layer.k.size(0) != final.k.size(0) || layer.k.size(1) != final.k.size(1) ||
                layer.k.size(3) != final.k.size(3)) {
                throw CudaSessionStateError("inconsistent CUDA paged cache topology");
            }
            keys.push_back(layer.k);
            values.push_back(layer.v);
        }
        auto key = mfq_tensor_backend::cat(keys, 2).contiguous();
        auto value = mfq_tensor_backend::cat(values, 2).contiguous();
        MfqCudaGuard guard(final.device);
        const auto options = key.options().device(
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, final.device));
        key = key.to(options, true, false).contiguous();
        value = value.to(options, true, false).contiguous();
        state.bytes += static_cast<size_t>(key.numel() * key.element_size() +
                                           value.numel() * value.element_size());
        layers.push_back(FullBlockSessionState{
            std::move(key),
            std::move(value),
            final.capacity,
            false,
        });
    }
    return state;
}

size_t session_tensor_bytes(const mfq_tensor_backend::Tensor &tensor) {
    if (!tensor.defined())
        return 0;
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

bool session_tensor_layout_matches(const mfq_tensor_backend::Tensor &target,
                                   const mfq_tensor_backend::Tensor &source) {
    return target.defined() && source.defined() && target.sizes() == source.sizes() &&
           target.scalar_type() == source.scalar_type() && target.device() == source.device();
}

void restore_session_tensor(mfq_tensor_backend::Tensor &target,
                            const mfq_tensor_backend::Tensor &source) {
    if (!source.defined()) {
        target = mfq_tensor_backend::Tensor();
        return;
    }
    if (!session_tensor_layout_matches(target, source)) {
        target = mfq_tensor_backend::empty(source.sizes(), source.options());
    }
    target.copy_(source);
}

void restore_session_prefix_tensor(mfq_tensor_backend::Tensor &target,
                                   const mfq_tensor_backend::Tensor &source, int64_t dimension,
                                   int64_t capacity) {
    if (!source.defined() || dimension < 0 || dimension >= source.dim() || capacity <= 0 ||
        source.size(dimension) > capacity) {
        throw CudaSessionStateError("session prefix tensor layout is invalid");
    }
    auto target_shape = source.sizes().vec();
    target_shape.at(static_cast<size_t>(dimension)) = capacity;
    const bool compatible =
        target.defined() && target.sizes() == mfq_tensor_backend::IntArrayRef(target_shape) &&
        target.scalar_type() == source.scalar_type() && target.device() == source.device();
    if (!compatible) {
        target = mfq_tensor_backend::empty(target_shape, source.options());
    }
    if (source.size(dimension) > 0) {
        target.narrow(dimension, 0, source.size(dimension)).copy_(source);
    }
}

FullBlockSessionState capture_full_attention_session_state(const FullBlock &block,
                                                           int64_t cache_pos, size_t &bytes) {
    if (!block.cache.k.defined() || !block.cache.v.defined() || !session_kv_geometry_matches(block.cache.k,block.cache.v) || block.cache.k.size(0) != 1) {
        throw std::runtime_error("full-attention KV cache is unavailable");
    }
    FullBlockSessionState state;
    state.capacity = block.cache.k.size(2);
    state.ring = block.cache.ring;
    const int64_t saved_tokens =
        state.ring ? state.capacity : std::min<int64_t>(cache_pos, state.capacity);
    state.k = block.cache.k.narrow(2, 0, saved_tokens).clone();
    state.v = block.cache.v.narrow(2, 0, saved_tokens).clone();
    bytes += session_tensor_bytes(state.k);
    bytes += session_tensor_bytes(state.v);
    return state;
}

void restore_full_attention_session_state(FullBlock &block, const FullBlockSessionState &state) {
    if (!state.k.defined() || !state.v.defined() || state.capacity <= 0 || state.k.dim() != 4 ||
        !session_kv_geometry_matches(state.k,state.v) || state.k.size(0) != 1 ||
        state.k.size(2) > state.capacity) {
        throw CudaSessionStateError("full-attention session KV layout is invalid");
    }
    restore_session_prefix_tensor(block.cache.k, state.k, 2, state.capacity);
    restore_session_prefix_tensor(block.cache.v, state.v, 2, state.capacity);
    block.cache.ring = state.ring;
    block.cache.quantization=block.kv_quantization;
}
