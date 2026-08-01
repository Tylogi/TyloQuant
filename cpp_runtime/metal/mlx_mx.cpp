#include "mlx_mx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr std::array<std::uint8_t, 4> kMagic{'M', 'X', 'T', '1'};
constexpr std::uint8_t kVersion = 1;

class Cursor {
public:
    explicit Cursor(const std::vector<std::uint8_t>& blob) : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        if (sizeof(T) > remaining()) {
            throw std::runtime_error(std::string("truncated MX ") + name);
        }
        T value{};
        std::memcpy(&value, blob_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::vector<std::uint8_t> bytes(std::size_t count, const char* name) {
        if (count > remaining()) {
            throw std::runtime_error(std::string("truncated MX ") + name);
        }
        std::vector<std::uint8_t> value(
            blob_.begin() + static_cast<std::ptrdiff_t>(offset_),
            blob_.begin() + static_cast<std::ptrdiff_t>(offset_ + count));
        offset_ += count;
        return value;
    }

    std::size_t remaining() const noexcept { return blob_.size() - offset_; }

private:
    const std::vector<std::uint8_t>& blob_;
    std::size_t offset_ = 0;
};

std::size_t checked_product(
    std::uint64_t left,
    std::uint64_t right,
    const char* name) {
    if (left == 0 || right == 0 ||
        left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::runtime_error(std::string("invalid MX ") + name);
    }
    return static_cast<std::size_t>(left * right);
}

int checked_dimension(std::uint64_t value, const char* name) {
    if (value == 0 || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid MX ") + name);
    }
    return static_cast<int>(value);
}

template <typename T>
array make_array(const std::vector<T>& values, Shape shape) {
    return array(values.begin(), std::move(shape));
}

constexpr const char* kMxHeader = R"METAL(
inline float mfq_mx_e8m0(uchar raw) {
    return raw == 255u ? NAN : exp2(float(int(raw) - 127));
}

inline float mfq_mx_fp4(uchar raw) {
    uchar magnitude = raw & 7u;
    float value = magnitude == 0u ? 0.0f
        : (magnitude == 1u ? 0.5f
        : (magnitude == 2u ? 1.0f
        : (magnitude == 3u ? 1.5f
        : (magnitude == 4u ? 2.0f
        : (magnitude == 5u ? 3.0f
        : (magnitude == 6u ? 4.0f : 6.0f))))));
    return (raw & 8u) == 0u ? value : -value;
}

inline float mfq_mx_fp8(uchar raw) {
    uint exponent = (uint(raw) >> 3u) & 15u;
    uint mantissa = uint(raw) & 7u;
    if (exponent == 15u && mantissa == 7u) {
        return NAN;
    }
    float value = exponent == 0u
        ? ldexp(float(mantissa) * 0.125f, -6)
        : ldexp(1.0f + float(mantissa) * 0.125f, int(exponent) - 7);
    return (raw & 128u) == 0u ? value : -value;
}

template <typename ValueStream, typename ScaleStream>
inline float mfq_mx_weight(
    ValueStream values,
    ScaleStream scales,
    uint output,
    uint column,
    uint mx_bits,
    uint width
) {
    if (mx_bits == 4u) {
        uchar packed = values[output * (width / 2u) + (column >> 1u)];
        uchar code = (column & 1u) == 0u ? packed & 15u : packed >> 4u;
        uchar scale = scales[output * (width / 32u) + column / 32u];
        return mfq_mx_fp4(code) * mfq_mx_e8m0(scale);
    } else {
        uchar code = values[output * width + column];
        uint scale_row = output / 128u;
        uint scale_column = column / 128u;
        uchar scale = scales[scale_row * (width / 128u) + scale_column];
        return mfq_mx_fp8(code) * mfq_mx_e8m0(scale);
    }
}
)METAL";

constexpr const char* kMxMatmul = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5u;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }
    float accum[TILE_M];
    for (uint row = 0u; row < uint(TILE_M); ++row) {
        accum[row] = 0.0f;
    }
    for (uint column = lane; column < uint(K); column += 32u) {
        float weight = mfq_mx_weight(
            values, scales, output, column, uint(MX_BITS), uint(K));
        for (uint local = 0u; local < uint(TILE_M); ++local) {
            uint row = first_row + local;
            if (row < uint(M)) {
                accum[local] += float(x[row * uint(K) + column]) * weight;
            }
        }
    }
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        uint row = first_row + local;
        float total = simd_sum(accum[local]);
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

constexpr const char* kMxGemv = R"METAL(
    constexpr uint OUTPUTS_PER_SIMD = 4u;
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint OUTPUTS_PER_TG = OUTPUTS_PER_SIMD * SIMD_GROUPS;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint tg_index = thread_position_in_grid.x / 64u;
    uint first_output = tg_index * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    float accum[OUTPUTS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation = float(x[column]);
        for (uint local = 0u; local < OUTPUTS_PER_SIMD; ++local) {
            uint output = first_output + local;
            if (output < uint(OUT)) {
                accum[local] += activation
                    * mfq_mx_weight(
                        values,
                        scales,
                        output,
                        column,
                        uint(MX_BITS),
                        uint(K));
            }
        }
    }
    for (uint local = 0u; local < OUTPUTS_PER_SIMD; ++local) {
        uint output = first_output + local;
        float total = simd_sum(accum[local]);
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T(total);
        }
    }
)METAL";

constexpr const char* kMxDequantize = R"METAL(
    uint index = thread_position_in_grid.x;
    uint count = uint(OUT) * uint(K);
    if (index < count) {
        uint output = index / uint(K);
        uint column = index - output * uint(K);
        y[index] = T(mfq_mx_weight(
            values,
            scales,
            output,
            column,
            uint(MX_BITS),
            uint(K)));
    }
)METAL";

constexpr const char* kMxEmbedding = R"METAL(
    uint index = thread_position_in_grid.x;
    uint count = uint(M) * uint(K);
    if (index < count) {
        uint token = index / uint(K);
        uint column = index - token * uint(K);
        uint output = uint(x[token]);
        y[index] = output < uint(OUT)
            ? T(mfq_mx_weight(
                values, scales, output, column, uint(MX_BITS), uint(K)))
            : T(NAN);
    }
)METAL";

mlx::core::fast::CustomKernelFunction make_kernel(
    std::string name,
    const char* source) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        std::move(name),
        {"values", "scales", "x"},
        {"y"},
        source,
        kMxHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& mx_matmul_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mx_packed_matmul", kMxMatmul);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mx_gemv_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mx_packed_gemv", kMxGemv);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mx_dequantize_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mx_dequantize", kMxDequantize);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mx_embedding_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mx_embedding", kMxEmbedding);
    return kernel;
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
templates(Dtype dtype, int bits, int input_size, int output_size, int rows = 1,
          int tile_rows = 1) {
    return {
        {"T", dtype},
        {"MX_BITS", bits},
        {"K", input_size},
        {"OUT", output_size},
        {"M", rows},
        {"TILE_M", tile_rows},
    };
}

} // namespace

bool is_mx_dtype(std::string_view dtype) noexcept {
    return dtype == "MXFP4" || dtype == "MXFP8";
}

MlxMxWeight::MlxMxWeight(
    array values,
    array scales,
    int bits,
    int input_size,
    int output_size)
    : values_(std::move(values)),
      scales_(std::move(scales)),
      bits_(bits),
      input_size_(input_size),
      output_size_(output_size) {}

MlxMxWeight MlxMxWeight::from_blob(
    std::string_view dtype,
    const std::vector<std::uint8_t>& blob) {
    if (!is_mx_dtype(dtype)) {
        throw std::runtime_error("unsupported MX MFQ dtype: " + std::string(dtype));
    }
    Cursor cursor(blob);
    for (const auto expected : kMagic) {
        if (cursor.scalar<std::uint8_t>("magic") != expected) {
            throw std::runtime_error("invalid MX MFQ magic");
        }
    }
    const auto version = cursor.scalar<std::uint8_t>("version");
    const auto kind = cursor.scalar<std::uint8_t>("kind");
    const auto reserved = cursor.scalar<std::uint16_t>("reserved");
    const auto rows = cursor.scalar<std::uint64_t>("logical rows");
    const auto columns = cursor.scalar<std::uint64_t>("logical columns");
    const auto storage_rows = cursor.scalar<std::uint64_t>("storage rows");
    const auto storage_columns = cursor.scalar<std::uint64_t>("storage columns");
    const auto scale_rows = cursor.scalar<std::uint64_t>("scale rows");
    const auto scale_columns = cursor.scalar<std::uint64_t>("scale columns");
    const int bits = dtype == "MXFP4" ? 4 : 8;
    if (version != kVersion || kind != bits || reserved != 0) {
        throw std::runtime_error("invalid MX MFQ header version/kind");
    }
    const auto output_size = checked_dimension(rows, "output size");
    const auto input_size = checked_dimension(columns, "input size");
    const auto expected_storage_columns = bits == 4 ? columns / 2 : columns;
    const auto expected_scale_rows = bits == 4 ? rows : (rows + 127) / 128;
    const auto expected_scale_columns = bits == 4 ? columns / 32 : columns / 128;
    if ((bits == 4 && columns % 32 != 0) ||
        (bits == 8 && columns % 128 != 0) ||
        storage_rows != rows ||
        storage_columns != expected_storage_columns ||
        scale_rows != expected_scale_rows ||
        scale_columns != expected_scale_columns) {
        throw std::runtime_error("invalid MX MFQ block geometry");
    }
    const auto value_count = checked_product(
        storage_rows, storage_columns, "value byte count");
    const auto scale_count = checked_product(
        scale_rows, scale_columns, "scale byte count");
    auto values = cursor.bytes(value_count, "values");
    auto scales = cursor.bytes(scale_count, "scales");
    if (cursor.remaining() != 0) {
        throw std::runtime_error("trailing bytes in MX MFQ tensor");
    }
    return MlxMxWeight(
        make_array(
            values,
            Shape{
                checked_dimension(storage_rows, "storage rows"),
                checked_dimension(storage_columns, "storage columns"),
            }),
        make_array(
            scales,
            Shape{
                checked_dimension(scale_rows, "scale rows"),
                checked_dimension(scale_columns, "scale columns"),
            }),
        bits,
        input_size,
        output_size);
}

array MlxMxWeight::dequantize(Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error("MX dequantization requires float16 or float32");
    }
    const auto elements = checked_product(
        static_cast<std::uint64_t>(output_size_),
        static_cast<std::uint64_t>(input_size_),
        "dequantization grid");
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MX dequantization grid exceeds MLX limits");
    }
    auto outputs = mx_dequantize_kernel()(
        {values_, scales_, values_},
        {Shape{output_size_, input_size_}},
        {dtype},
        {static_cast<int>(elements), 1, 1},
        {std::min(256, static_cast<int>(elements)), 1, 1},
        templates(dtype, bits_, input_size_, output_size_),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxMxWeight::matmul(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error("MX input width does not match packed weight");
    }
    const auto rows = input.size() / static_cast<std::size_t>(input_size_);
    if (rows == 0 || rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("unsupported MX input row count");
    }
    Shape output_shape = input.shape();
    output_shape.back() = output_size_;
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source,
        Shape{static_cast<int>(rows), input_size_});
    if (rows >= 64) {
        auto dense = dequantize(source.dtype());
        auto result = mlx::core::matmul(source, mlx::core::transpose(dense));
        return mlx::core::reshape(std::move(result), std::move(output_shape));
    }

    const bool gemv = rows == 1;
    const int tile_rows = gemv ? 1 : (rows <= 16 ? static_cast<int>(rows) : 8);
    const auto row_tiles = (rows + static_cast<std::size_t>(tile_rows) - 1) /
        static_cast<std::size_t>(tile_rows);
    const auto grid = gemv
        ? static_cast<std::size_t>((output_size_ + 7) / 8) * 64
        : row_tiles * static_cast<std::size_t>(output_size_) * 32;
    if (grid > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MX Metal grid exceeds MLX limits");
    }
    auto outputs = (gemv ? mx_gemv_kernel() : mx_matmul_kernel())(
        {values_, scales_, source},
        {Shape{static_cast<int>(rows), output_size_}},
        {source.dtype()},
        {static_cast<int>(grid), 1, 1},
        {gemv ? 64 : 32, 1, 1},
        templates(
            source.dtype(),
            bits_,
            input_size_,
            output_size_,
            static_cast<int>(rows),
            tile_rows),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(std::move(outputs.front()), std::move(output_shape));
}

array MlxMxWeight::embedding(const array& token_ids, Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error("MX embedding requires float16 or float32");
    }
    auto ids = token_ids;
    if (ids.dtype() != mlx::core::int32 && ids.dtype() != mlx::core::uint32) {
        ids = mlx::core::astype(ids, mlx::core::int32);
    }
    const auto tokens = ids.size();
    Shape output_shape = ids.shape();
    output_shape.push_back(input_size_);
    if (tokens == 0) {
        return mlx::core::zeros(output_shape, dtype);
    }
    const auto elements = checked_product(
        static_cast<std::uint64_t>(tokens),
        static_cast<std::uint64_t>(input_size_),
        "embedding grid");
    if (tokens > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MX embedding grid exceeds MLX limits");
    }
    auto outputs = mx_embedding_kernel()(
        {values_, scales_, mlx::core::reshape(
            ids, Shape{static_cast<int>(tokens)})},
        {output_shape},
        {dtype},
        {static_cast<int>(elements), 1, 1},
        {std::min(256, static_cast<int>(elements)), 1, 1},
        templates(
            dtype,
            bits_,
            input_size_,
            output_size_,
            static_cast<int>(tokens)),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

std::size_t MlxMxWeight::packed_nbytes() const noexcept {
    return values_.nbytes() + scales_.nbytes();
}

} // namespace mfq::metal
