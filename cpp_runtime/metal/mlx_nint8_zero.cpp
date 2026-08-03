#include "mlx_nint8_zero.h"

#include "mlx_staging_allocator.h"

#include <mlx/allocator.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr std::size_t kBlockElements = 32;
constexpr std::size_t kBlockBytes = 34;

constexpr const char* kNint8ZeroMatmul = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    float accumulators[TILE_M];
    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        accumulators[local_row] = 0.0f;
    }
    for (uint column = lane; column < uint(K); column += 32u) {
        uint group = column >> 5;
        float weight =
            float(scales[output * uint(NG) + group]) *
            float(q[output * uint(K) + column]);
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                accumulators[local_row] +=
                    float(x[row * uint(K) + column]) * weight;
            }
        }
    }

    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        float total = simd_sum(accumulators[local_row]);
        uint row = first_row + local_row;
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

constexpr const char* kNint8ZeroGemv = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup;
    if (output >= uint(OUT)) {
        return;
    }

    float accumulator = 0.0f;
    // Four adjacent Q8 values stay within one 32-value scale block.  Loading
    // them as one vector cuts loop/address overhead by 4x while preserving the
    // coalesced one-SIMD-per-output memory layout.
    for (uint column = lane * 4u;
         column < uint(K);
         column += 128u) {
        uint group = column >> 5;
        float scale = float(scales[output * uint(NG) + group]);
        uint offset = output * uint(K) + column;
        const device char4* packed =
            (const device char4*)(q + offset);
        char4 codes = *packed;
        float4 activations = float4(
            float(x[column]),
            float(x[column + 1u]),
            float(x[column + 2u]),
            float(x[column + 3u]));
        float4 weights = scale * float4(codes);
        accumulator +=
            activations.x * weights.x
            + activations.y * weights.y
            + activations.z * weights.z
            + activations.w * weights.w;
    }

    float total = simd_sum(accumulator);
    if (lane == 0u) {
        y[output] = T(total);
    }
)METAL";

constexpr const char* kNint8ZeroGroupedRow = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    uint input_group = output / uint(OUT_PER_GROUP);
    float accumulators[TILE_M];
    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        accumulators[local_row] = 0.0f;
    }

    // One SIMD owns one output row. Unlike the generic grouped fallback, it
    // evaluates only the diagonal input/output group pair. Four adjacent Q8
    // values share a scale block and are consumed together.
    for (uint column = lane * 4u;
         column < uint(K);
         column += 128u) {
        uint quant_group = column >> 5;
        float scale =
            float(scales[output * uint(NG) + quant_group]);
        uint offset = output * uint(K) + column;
        const device char4* packed =
            (const device char4*)(q + offset);
        float4 weights = scale * float4(*packed);
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                uint input_offset =
                    (row * uint(GROUP_COUNT) + input_group) * uint(K)
                    + column;
                float4 activations = float4(
                    float(x[input_offset]),
                    float(x[input_offset + 1u]),
                    float(x[input_offset + 2u]),
                    float(x[input_offset + 3u]));
                accumulators[local_row] +=
                    activations.x * weights.x
                    + activations.y * weights.y
                    + activations.z * weights.z
                    + activations.w * weights.w;
            }
        }
    }

    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        float total = simd_sum(accumulators[local_row]);
        uint row = first_row + local_row;
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

constexpr const char* kNint8ZeroGroupedRowInverseRope = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    uint input_group = output / uint(OUT_PER_GROUP);
    float accumulators[TILE_M];
    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        accumulators[local_row] = 0.0f;
    }

    constexpr uint PREFIX = uint(HEAD_DIM - ROTARY);
    constexpr uint PAIRS = uint(ROTARY / 2);
    for (uint column = lane * 4u;
         column < uint(K);
         column += 128u) {
        uint quant_group = column >> 5;
        float scale =
            float(scales[output * uint(NG) + quant_group]);
        uint weight_offset = output * uint(K) + column;
        const device char4* packed =
            (const device char4*)(q + weight_offset);
        float4 weights = scale * float4(*packed);
        uint head_column = column % uint(HEAD_DIM);

        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                uint input_offset =
                    (row * uint(GROUP_COUNT) + input_group) * uint(K)
                    + column;
                float4 activations = float4(
                    float(x[input_offset]),
                    float(x[input_offset + 1u]),
                    float(x[input_offset + 2u]),
                    float(x[input_offset + 3u]));
                if (head_column >= PREFIX) {
                    uint token = row % uint(TOKENS);
                    uint pair = (head_column - PREFIX) >> 1u;
                    uint rope_offset = token * PAIRS + pair;
                    float cosine0 = float(cos_values[rope_offset]);
                    float sine0 = float(sin_values[rope_offset]);
                    float cosine1 = float(cos_values[rope_offset + 1u]);
                    float sine1 = float(sin_values[rope_offset + 1u]);

                    // Match the original graph exactly: inverse RoPE writes
                    // one T-typed intermediate before the packed GEMV reads
                    // it back and promotes it to float for accumulation.
                    T rotated0 = T(
                        activations.x * cosine0
                        + activations.y * sine0);
                    T rotated1 = T(
                        activations.y * cosine0
                        - activations.x * sine0);
                    T rotated2 = T(
                        activations.z * cosine1
                        + activations.w * sine1);
                    T rotated3 = T(
                        activations.w * cosine1
                        - activations.z * sine1);
                    activations = float4(
                        float(rotated0),
                        float(rotated1),
                        float(rotated2),
                        float(rotated3));
                }
                accumulators[local_row] +=
                    activations.x * weights.x
                    + activations.y * weights.y
                    + activations.z * weights.z
                    + activations.w * weights.w;
            }
        }
    }

    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        float total = simd_sum(accumulators[local_row]);
        uint row = first_row + local_row;
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

constexpr const char* kNint8ZeroEmbedding = R"METAL(
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(COUNT * K)) {
        return;
    }
    uint token_position = linear / uint(K);
    uint column = linear - token_position * uint(K);
    uint output = uint(token_ids[token_position]);
    if (output >= uint(OUT)) {
        y[linear] = T(0.0f);
        return;
    }
    uint group = column >> 5;
    y[linear] = T(
        float(scales[output * uint(NG) + group]) *
        float(q[output * uint(K) + column]));
)METAL";

class BlobCursor {
public:
    explicit BlobCursor(std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        require(sizeof(T), name);
        T value{};
        std::memcpy(&value, blob_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    const std::uint8_t* bytes(
        std::size_t count,
        const char* name) {
        require(count, name);
        const auto* result = blob_.data() + offset_;
        offset_ += count;
        return result;
    }

    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    void require(std::size_t count, const char* name) const {
        if (count > blob_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated NINT8-0 ") + name);
        }
    }

    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::int32_t checked_shape(std::int64_t value, const char* name) {
    if (value <= 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(
            std::string("invalid NINT8-0 ") + name);
    }
    return static_cast<std::int32_t>(value);
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (right != 0 &&
        left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::runtime_error(
            std::string("NINT8-0 ") + name + " overflows");
    }
    return left * right;
}

template <typename Allocator>
array make_int8_array(
    std::vector<std::int8_t, Allocator> values,
    Shape shape) {
    auto result = array(
        mlx::core::allocator::malloc(values.size()),
        std::move(shape),
        mlx::core::int8);
    std::memcpy(
        result.data<std::int8_t>(),
        values.data(),
        values.size());
    return result;
}

template <typename Allocator>
array make_float16_array(
    std::vector<std::uint16_t, Allocator> values,
    Shape shape) {
    const auto bytes = values.size() * sizeof(std::uint16_t);
    auto result = array(
        mlx::core::allocator::malloc(bytes),
        std::move(shape),
        mlx::core::float16);
    std::memcpy(
        result.data<std::uint16_t>(),
        values.data(),
        bytes);
    return result;
}

mlx::core::fast::CustomKernelFunction make_matmul_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint8_zero_packed_matmul",
        {"q", "scales", "x"},
        {"y"},
        kNint8ZeroMatmul,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& matmul_kernel() {
    static const auto kernel = make_matmul_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_gemv_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint8_zero_packed_gemv",
        {"q", "scales", "x"},
        {"y"},
        kNint8ZeroGemv,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& gemv_kernel() {
    static const auto kernel = make_gemv_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_grouped_row_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint8_zero_grouped_row",
        {"q", "scales", "x"},
        {"y"},
        kNint8ZeroGroupedRow,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& grouped_row_kernel() {
    static const auto kernel = make_grouped_row_kernel();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
grouped_row_inverse_rope_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_nint8_zero_grouped_row_inverse_rope",
            {"q", "scales", "x", "cos_values", "sin_values"},
            {"y"},
            kNint8ZeroGroupedRowInverseRope,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_embedding_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint8_zero_packed_embedding",
        {"q", "scales", "token_ids"},
        {"y"},
        kNint8ZeroEmbedding,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& embedding_kernel() {
    static const auto kernel = make_embedding_kernel();
    return kernel;
}

} // namespace

bool is_nint8_zero_dtype(std::string_view dtype) noexcept {
    return dtype == "NINT8-0";
}

MlxNint8ZeroWeight::MlxNint8ZeroWeight(
    array q,
    array scales,
    int input_size,
    int output_size,
    int groups)
    : q_(std::move(q)),
      scales_(std::move(scales)),
      input_size_(input_size),
      output_size_(output_size),
      groups_(groups) {}

MlxNint8ZeroWeight MlxNint8ZeroWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto* magic = cursor.bytes(4, "magic");
    if (std::memcmp(magic, "NI80", 4) != 0) {
        throw std::runtime_error("invalid NINT8-0 magic");
    }

    const auto axis = cursor.scalar<std::int32_t>("axis");
    const auto neuron_len =
        cursor.scalar<std::int32_t>("neuron length");
    const auto dimensions =
        cursor.scalar<std::uint32_t>("dimension count");
    if (dimensions != 2 || axis != 0 || neuron_len <= 0 ||
        neuron_len % static_cast<int>(kBlockElements) != 0) {
        throw std::runtime_error(
            "NINT8-0 Metal requires a row-major rank-two weight");
    }

    const auto output_shape =
        cursor.scalar<std::int64_t>("output shape");
    const auto input_shape =
        cursor.scalar<std::int64_t>("input shape");
    const auto output_size =
        cursor.scalar<std::uint32_t>("output size");
    const auto groups =
        cursor.scalar<std::uint32_t>("group count");
    if (output_shape != output_size ||
        input_shape != neuron_len ||
        output_size == 0 ||
        groups != static_cast<std::uint32_t>(
            neuron_len / static_cast<int>(kBlockElements))) {
        throw std::runtime_error(
            "inconsistent NINT8-0 Metal dimensions");
    }

    const auto blocks = checked_product(
        static_cast<std::size_t>(output_size),
        static_cast<std::size_t>(groups),
        "block count");
    const auto expected_bytes =
        checked_product(blocks, kBlockBytes, "payload length");
    if (cursor.remaining() != expected_bytes) {
        throw std::runtime_error(
            "invalid NINT8-0 block payload length");
    }

    detail::StagingVector<std::int8_t> q(
        checked_product(blocks, kBlockElements, "quantized value count"));
    detail::StagingVector<std::uint16_t> scales(blocks);
    for (std::size_t block = 0; block < blocks; ++block) {
        scales[block] =
            cursor.scalar<std::uint16_t>("block scale");
        const auto* values =
            cursor.bytes(kBlockElements, "block values");
        std::memcpy(
            q.data() + block * kBlockElements,
            values,
            kBlockElements);
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "trailing bytes in NINT8-0 tensor");
    }

    const auto checked_output =
        checked_shape(output_size, "output size");
    const auto checked_input =
        checked_shape(neuron_len, "input size");
    const auto checked_groups =
        checked_shape(groups, "group count");
    return MlxNint8ZeroWeight(
        make_int8_array(
            std::move(q),
            Shape{checked_output, checked_input}),
        make_float16_array(
            std::move(scales),
            Shape{checked_output, checked_groups}),
        checked_input,
        checked_output,
        checked_groups);
}

array MlxNint8ZeroWeight::matmul(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error(
            "NINT8-0 input width does not match packed weight");
    }

    std::int64_t rows = 1;
    Shape output_shape = input.shape();
    for (std::size_t index = 0; index + 1 < input.ndim(); ++index) {
        if (input.shape(static_cast<int>(index)) < 0 ||
            rows > std::numeric_limits<std::int32_t>::max() /
                std::max(1, input.shape(static_cast<int>(index)))) {
            throw std::runtime_error(
                "unsupported NINT8-0 input row count");
        }
        rows *= input.shape(static_cast<int>(index));
    }
    if (rows <= 0 ||
        rows > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(
            "unsupported NINT8-0 input row count");
    }
    output_shape.back() = output_size_;

    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source,
        Shape{
            static_cast<std::int32_t>(rows),
            input_size_,
        });
    if (rows >= 64 &&
        source.dtype() == mlx::core::float16) {
        const auto token_ids = mlx::core::arange(
            output_size_,
            mlx::core::int32);
        const auto dense = embedding(
            token_ids,
            mlx::core::float16);
        auto result = mlx::core::matmul(
            source,
            mlx::core::transpose(dense));
        return mlx::core::reshape(
            std::move(result),
            std::move(output_shape));
    }

    const int tile_rows =
        rows == 1 ? 1 : (rows <= 16 ? static_cast<int>(rows) : 8);
    const auto row_tiles =
        (rows + tile_rows - 1) / tile_rows;
    const auto grid_x =
        row_tiles * static_cast<std::int64_t>(output_size_) * 32;
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "NINT8-0 Metal grid exceeds MLX limits");
    }

    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
        templates{
            {"T", source.dtype()},
            {"M", static_cast<int>(rows)},
            {"TILE_M", tile_rows},
            {"OUT", output_size_},
            {"K", input_size_},
            {"NG", groups_},
        };
    const auto* kernel =
        rows == 1 ? &gemv_kernel() : &matmul_kernel();
    const int threadgroup = static_cast<int>(
        std::min<std::int64_t>(256, grid_x));
    auto outputs = (*kernel)(
        {q_, scales_, source},
        {
            Shape{
                static_cast<std::int32_t>(rows),
                output_size_,
            },
        },
        {source.dtype()},
        {static_cast<int>(grid_x), 1, 1},
        {threadgroup, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        outputs.front(),
        std::move(output_shape));
}

array MlxNint8ZeroWeight::grouped_row_matmul(
    const array& input,
    int group_count) const {
    if (group_count <= 0 ||
        input.ndim() < 2 ||
        input.shape(-2) != group_count ||
        input.shape(-1) != input_size_ ||
        output_size_ % group_count != 0) {
        throw std::runtime_error(
            "NINT8-0 grouped-row input or weight shape is incompatible");
    }

    std::int64_t rows = 1;
    Shape output_shape(
        input.shape().begin(),
        input.shape().end() - 2);
    for (std::size_t index = 0; index + 2 < input.ndim(); ++index) {
        const int extent = input.shape(static_cast<int>(index));
        if (extent < 0 ||
            rows > std::numeric_limits<std::int32_t>::max() /
                std::max(1, extent)) {
            throw std::runtime_error(
                "unsupported NINT8-0 grouped-row count");
        }
        rows *= extent;
    }
    if (rows <= 0 ||
        rows > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(
            "unsupported NINT8-0 grouped-row count");
    }

    const int out_per_group = output_size_ / group_count;
    output_shape.push_back(group_count);
    output_shape.push_back(out_per_group);

    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(
            source,
            mlx::core::float16);
    }
    source = mlx::core::contiguous(
        mlx::core::reshape(
            source,
            Shape{
                static_cast<std::int32_t>(rows),
                group_count,
                input_size_,
            }));

    const int tile_rows =
        rows == 1 ? 1 : (rows <= 16 ? static_cast<int>(rows) : 8);
    const auto row_tiles =
        (rows + tile_rows - 1) / tile_rows;
    const auto grid_x =
        row_tiles * static_cast<std::int64_t>(output_size_) * 32;
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "NINT8-0 grouped-row Metal grid exceeds MLX limits");
    }

    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
        templates{
            {"T", source.dtype()},
            {"M", static_cast<int>(rows)},
            {"TILE_M", tile_rows},
            {"GROUP_COUNT", group_count},
            {"OUT_PER_GROUP", out_per_group},
            {"OUT", output_size_},
            {"K", input_size_},
            {"NG", groups_},
        };
    const int threadgroup = static_cast<int>(
        std::min<std::int64_t>(256, grid_x));
    auto outputs = grouped_row_kernel()(
        {q_, scales_, source},
        {
            Shape{
                static_cast<std::int32_t>(rows),
                output_size_,
            },
        },
        {source.dtype()},
        {static_cast<int>(grid_x), 1, 1},
        {threadgroup, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        outputs.front(),
        std::move(output_shape));
}

array MlxNint8ZeroWeight::grouped_row_matmul_inverse_rope(
    const array& input,
    int group_count,
    const array& cosine,
    const array& sine,
    int head_dimension,
    int rotary_dimension) const {
    if (group_count <= 0 ||
        input.ndim() < 3 ||
        input.shape(-2) != group_count ||
        input.shape(-1) != input_size_ ||
        output_size_ % group_count != 0 ||
        head_dimension <= 0 ||
        rotary_dimension <= 0 ||
        rotary_dimension > head_dimension ||
        head_dimension % 4 != 0 ||
        rotary_dimension % 4 != 0 ||
        input_size_ % head_dimension != 0 ||
        cosine.shape() != sine.shape() ||
        cosine.ndim() != 2 ||
        cosine.shape(0) <= 0 ||
        cosine.shape(1) != rotary_dimension / 2) {
        throw std::runtime_error(
            "NINT8-0 inverse-RoPE grouped-row shape is incompatible");
    }

    std::int64_t rows = 1;
    Shape output_shape(
        input.shape().begin(),
        input.shape().end() - 2);
    for (std::size_t index = 0; index + 2 < input.ndim(); ++index) {
        const int extent = input.shape(static_cast<int>(index));
        if (extent <= 0 ||
            rows > std::numeric_limits<std::int32_t>::max() / extent) {
            throw std::runtime_error(
                "unsupported NINT8-0 inverse-RoPE grouped-row count");
        }
        rows *= extent;
    }
    const int tokens = cosine.shape(0);
    if (rows <= 0 ||
        rows > std::numeric_limits<std::int32_t>::max() ||
        rows % tokens != 0) {
        throw std::runtime_error(
            "NINT8-0 inverse-RoPE token count is incompatible");
    }

    const int out_per_group = output_size_ / group_count;
    output_shape.push_back(group_count);
    output_shape.push_back(out_per_group);

    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(
        mlx::core::reshape(
            source,
            Shape{
                static_cast<std::int32_t>(rows),
                group_count,
                input_size_,
            }));
    auto cos_values = cosine;
    auto sin_values = sine;
    if (cos_values.dtype() != mlx::core::float32) {
        cos_values = mlx::core::astype(
            cos_values,
            mlx::core::float32);
    }
    if (sin_values.dtype() != mlx::core::float32) {
        sin_values = mlx::core::astype(
            sin_values,
            mlx::core::float32);
    }
    cos_values = mlx::core::contiguous(cos_values);
    sin_values = mlx::core::contiguous(sin_values);

    const int tile_rows =
        rows == 1 ? 1 : (rows <= 16 ? static_cast<int>(rows) : 8);
    const auto row_tiles =
        (rows + tile_rows - 1) / tile_rows;
    const auto grid_x =
        row_tiles * static_cast<std::int64_t>(output_size_) * 32;
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "NINT8-0 inverse-RoPE grouped-row Metal grid exceeds MLX limits");
    }

    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
        templates{
            {"T", source.dtype()},
            {"M", static_cast<int>(rows)},
            {"TILE_M", tile_rows},
            {"TOKENS", tokens},
            {"GROUP_COUNT", group_count},
            {"OUT_PER_GROUP", out_per_group},
            {"OUT", output_size_},
            {"K", input_size_},
            {"NG", groups_},
            {"HEAD_DIM", head_dimension},
            {"ROTARY", rotary_dimension},
        };
    const int threadgroup = static_cast<int>(
        std::min<std::int64_t>(256, grid_x));
    auto outputs = grouped_row_inverse_rope_kernel()(
        {q_, scales_, source, cos_values, sin_values},
        {
            Shape{
                static_cast<std::int32_t>(rows),
                output_size_,
            },
        },
        {source.dtype()},
        {static_cast<int>(grid_x), 1, 1},
        {threadgroup, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        outputs.front(),
        std::move(output_shape));
}

array MlxNint8ZeroWeight::embedding(
    const array& token_ids,
    Dtype dtype) const {
    if (dtype != mlx::core::float16 &&
        dtype != mlx::core::float32) {
        throw std::runtime_error(
            "NINT8-0 embedding output must be float16 or float32");
    }
    auto ids = token_ids;
    if (ids.dtype() != mlx::core::int32 &&
        ids.dtype() != mlx::core::uint32) {
        ids = mlx::core::astype(ids, mlx::core::int32);
    }
    const auto count = ids.size();
    Shape output_shape = ids.shape();
    output_shape.push_back(input_size_);
    if (count == 0) {
        return mlx::core::zeros(output_shape, dtype);
    }
    if (count >
            static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max()) ||
        count > static_cast<std::size_t>(
                    std::numeric_limits<int>::max() / input_size_)) {
        throw std::runtime_error(
            "NINT8-0 embedding input is too large");
    }

    ids = mlx::core::reshape(
        ids,
        Shape{static_cast<std::int32_t>(count)});
    const int grid = static_cast<int>(count) * input_size_;
    const int threadgroup = std::min(256, std::max(1, grid));
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
        templates{
            {"T", dtype},
            {"COUNT", static_cast<int>(count)},
            {"OUT", output_size_},
            {"K", input_size_},
            {"NG", groups_},
        };
    auto outputs = embedding_kernel()(
        {q_, scales_, ids},
        {
            Shape{
                static_cast<std::int32_t>(count),
                input_size_,
            },
        },
        {dtype},
        {grid, 1, 1},
        {threadgroup, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        outputs.front(),
        std::move(output_shape));
}

std::size_t MlxNint8ZeroWeight::packed_nbytes() const noexcept {
    return q_.nbytes() + scales_.nbytes();
}

} // namespace mfq::metal
