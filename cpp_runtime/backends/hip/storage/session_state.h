#pragma once

#include "mfq_tensor_backend.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

struct Dsv4PoolState;
struct FullBlock;
namespace mfq::models::deepseek_v41 { struct EngramHashSnapshot; }

struct FullBlockSessionState {
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
    int64_t capacity = 0;
    bool ring = false;
};

inline bool session_kv_geometry_matches(const mfq_tensor_backend::Tensor& k,
        const mfq_tensor_backend::Tensor& v) {
    if(!k.defined() || !v.defined() || k.dim()!=4 || v.dim()!=4 || k.scalar_type()!=v.scalar_type())return false;
    for(int axis=0;axis<3;++axis)if(k.size(axis)!=v.size(axis))return false;
    if(k.scalar_type()==mfq_tensor_backend::kInt32)return k.size(3)>1 && v.size(3)>1;
    return k.size(3)>0 && k.size(3)==v.size(3);
}

class CudaSessionStateError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class TextSessionStateKind : int {
    Unsupported,
    FullAttention,
    HybridAttention,
    DeepseekV4,
    GlmDsa,
    Qwen4,
    DeepseekV41,
};

struct Dsv4PoolSessionState {
    int64_t ratio = 0;
    int64_t head_dim = 0;
    int64_t cache_quant_mode = 0;
    int64_t capacity = 0;
    bool overlap = false;
    mfq_tensor_backend::Tensor state_kv;
    mfq_tensor_backend::Tensor state_gate;
    mfq_tensor_backend::Tensor previous_kv;
    mfq_tensor_backend::Tensor previous_gate;
    mfq_tensor_backend::Tensor pool;
};

struct Dsv4BlockSessionState {
    mfq_tensor_backend::Tensor local_cache;
    Dsv4PoolSessionState compressor;
    Dsv4PoolSessionState indexer_compressor;
};

struct DeepseekV41BlockSessionState {
    mfq_tensor_backend::Tensor local_kv, compressed_kv, index_k, partial_kv, partial_score;
    int64_t compressed_length = 0, partial_length = 0;
};

struct GlmDsaBlockSessionState {
    mfq_tensor_backend::Tensor kv_cache;
    mfq_tensor_backend::Tensor index_cache;
    int64_t kv_capacity = 0;
    int64_t index_capacity = 0;
    bool full_indexer = false;
};

enum class HybridBlockSessionStateKind {
    FullAttention,
    Recurrent,
    // The block keeps no cache state, so a checkpoint only records position.
    Stateless,
};

struct HybridBlockSessionState {
    HybridBlockSessionStateKind kind =
        HybridBlockSessionStateKind::FullAttention;
    FullBlockSessionState full_attention;
    mfq_tensor_backend::Tensor convolution_state;
    mfq_tensor_backend::Tensor recurrent_state;
};

struct Qwen4BlockSessionState {
    mfq_tensor_backend::Tensor keys, values, index;
    mfq_tensor_backend::Tensor convolution, recurrent, ple_convolution;
    int64_t ple_batch = 0;
    std::vector<int64_t> ple_tokens;
};

struct MtpSessionState {
    std::vector<FullBlockSessionState> blocks;
    mfq_tensor_backend::Tensor last_target_hidden;
    std::vector<Qwen4BlockSessionState> qwen4_blocks;
    std::vector<mfq_tensor_backend::Tensor> qwen4_positions;
    std::vector<int64_t> qwen4_lengths;
    std::vector<mfq_tensor_backend::Tensor> deepseek_v41_rings;
    int64_t cache_pos = 0;
    size_t bytes = 0;
};

using TextSessionPayload = std::variant<
    std::monostate,
    std::vector<FullBlockSessionState>,
    std::vector<HybridBlockSessionState>,
    std::vector<Dsv4BlockSessionState>,
    std::vector<GlmDsaBlockSessionState>,
    std::vector<Qwen4BlockSessionState>,
    std::vector<DeepseekV41BlockSessionState>>;

struct TextSessionState {
    std::vector<int64_t> tokens;
    std::string input_key;
    TextSessionPayload payload;
    mfq_tensor_backend::Tensor qwen4_positions;
    std::shared_ptr<const mfq::models::deepseek_v41::EngramHashSnapshot> deepseek_v41_engram;
    std::optional<MtpSessionState> mtp;
    int64_t cache_pos = 0;
    int64_t decode_position_delta = 0;
    size_t bytes = 0;

    TextSessionStateKind kind() const noexcept {
        return static_cast<TextSessionStateKind>(payload.index());
    }
};

namespace mfq::cuda {
template <typename Model> struct CudaCausalOps;
template <typename Model> struct FullAttentionSessionCodec {
    using CausalModel = typename Model::template CausalModel<CudaCausalOps<Model>>;
    static TextSessionStateKind kind(const CausalModel &model);
    static bool supports_paged(const CausalModel &model);
    static TextSessionState capture(const CausalModel &model, const std::vector<int64_t> &tokens);
    static void restore(CausalModel &model, const TextSessionState &state);
};
} // namespace mfq::cuda

using CudaPagedPayload =
    std::shared_ptr<const std::vector<uint8_t>>;

class CudaPagedWriter {
public:
    template <typename T>
    void scalar(T value) {
        using Unsigned = std::make_unsigned_t<T>;
        const auto converted = static_cast<Unsigned>(value);
        for (size_t index = 0; index < sizeof(T); ++index) {
            bytes_.push_back(static_cast<uint8_t>(
                converted >> (index * 8)));
        }
    }

    void raw(const void* data, size_t size);
    void tensor(const mfq_tensor_backend::Tensor& source);
    CudaPagedPayload finish() &&;

private:
    std::vector<uint8_t> bytes_;
};

class CudaPagedReader {
public:
    explicit CudaPagedReader(const std::vector<uint8_t>& bytes)
        : cursor_(bytes.data()), end_(bytes.data() + bytes.size()) {}

    template <typename T>
    T scalar(const char* name) {
        if (remaining() < sizeof(T)) {
            throw CudaSessionStateError(
                std::string("truncated CUDA paged cache ") + name);
        }
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned value = 0;
        for (size_t index = 0; index < sizeof(T); ++index) {
            value |= static_cast<Unsigned>(cursor_[index]) << (index * 8);
        }
        cursor_ += sizeof(T);
        return static_cast<T>(value);
    }

    void raw(void* destination, size_t size, const char* name);
    std::pair<mfq_tensor_backend::Tensor, int> tensor();
    void skip(size_t bytes,const char* name);
    void expect_end() const;

private:
    size_t remaining() const;

    const uint8_t* cursor_;
    const uint8_t* end_;
};

struct CudaDecodedPagedLayer {
    int64_t capacity = 0;
    int device = -1;
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
};

struct CudaDecodedPagedBlock {
    uint32_t start = 0;
    uint32_t count = 0;
    std::vector<CudaDecodedPagedLayer> layers;
};

std::vector<CudaPagedPayload> encode_cuda_paged_session(
    const TextSessionState& state,
    size_t block_size,
    size_t first_block,
    size_t maximum_blocks = std::numeric_limits<size_t>::max());

CudaPagedPayload encode_cuda_paged_block(
    const TextSessionState& state,
    size_t block_size,
    size_t block_index);

CudaDecodedPagedBlock decode_cuda_paged_block(
    const std::vector<uint8_t>& payload);

TextSessionState decode_cuda_paged_session(
    const std::vector<CudaPagedPayload>& payloads,
    const std::vector<int64_t>& tokens,
    size_t block_size);

size_t session_tensor_bytes(const mfq_tensor_backend::Tensor& tensor);

CudaPagedPayload encode_cuda_qwen4_paged_block(const TextSessionState&,size_t,size_t);
TextSessionState decode_cuda_qwen4_paged_session(const std::vector<CudaPagedPayload>&,const std::vector<int64_t>&,size_t);
size_t cuda_paged_decodable_blocks(const std::vector<CudaPagedPayload>&);
size_t cuda_paged_payload_tokens(const CudaPagedPayload&);
CudaPagedPayload encode_cuda_deepseek_v41_paged_block(const TextSessionState&,size_t,size_t);
TextSessionState decode_cuda_deepseek_v41_paged_session(
    const std::vector<CudaPagedPayload>&,const std::vector<int64_t>&,size_t);
bool cuda_paged_checkpoint_needs_update(const CudaPagedPayload&,const TextSessionState&);
bool cuda_deepseek_v41_paged_exact(const CudaPagedPayload&);

bool cuda_qwen4_paged_checkpoint_needs_update(const CudaPagedPayload&,const TextSessionState&);

bool session_tensor_layout_matches(
    const mfq_tensor_backend::Tensor& target,
    const mfq_tensor_backend::Tensor& source);

void restore_session_tensor(
    mfq_tensor_backend::Tensor& target,
    const mfq_tensor_backend::Tensor& source);

void restore_session_prefix_tensor(
    mfq_tensor_backend::Tensor& target,
    const mfq_tensor_backend::Tensor& source,
    int64_t dimension,
    int64_t capacity);

FullBlockSessionState capture_full_attention_session_state(
    const FullBlock& block,
    int64_t cache_pos,
    size_t& bytes);

void restore_full_attention_session_state(
    FullBlock& block,
    const FullBlockSessionState& state);

Dsv4PoolSessionState capture_dsv4_pool_session_state(
    const Dsv4PoolState& source,
    int64_t cache_pos,
    size_t& bytes);

void restore_dsv4_pool_session_state(
    Dsv4PoolState& target,
    const Dsv4PoolSessionState& state);
