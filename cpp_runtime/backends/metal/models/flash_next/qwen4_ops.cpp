#include "qwen4_ops.h"
#include "mlx_nint_metadata.h"
#include "mlx_kernel_prepare.h"

#include "mlx_transformer.h"
#include "mlx_sparse_attention.h"

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;
using mlx::core::CompileOptions;
using mlx::core::MathMode;

constexpr int kGatedHcProjectionThreads = 256;
constexpr int kGatedHcUpThreads = 320;

class QsaPrefillOutputGatePrimitive final : public mlx::core::Primitive, public MlxPreparableKernel {
public:
    QsaPrefillOutputGatePrimitive(mlx::core::Stream stream, mlx::core::Dtype dtype,
        int heads, int dimension)
        : Primitive(stream), dtype_(dtype), heads_(heads), dimension_(dimension) {}

    std::string preparation_key() const override { return "mfq_qsa_prefill_output_gate_" + mlx::core::type_to_name(dtype_) + "_" + std::to_string(heads_) + "_" + std::to_string(dimension_); }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = kernel_device(stream().device);
        const std::string type = dtype_ == mlx::core::float16 ? "half"
            : dtype_ == mlx::core::bfloat16 ? "bfloat" : "float";
        const std::string name = "mfq_qsa_prefill_output_gate_" + type + "_" +
            std::to_string(heads_) + "_" + std::to_string(dimension_);
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        auto* library = device.get_library(name, options, [&] {
            return "#include <metal_stdlib>\nusing namespace metal;\nusing T = " + type +
                ";\n#define HEADS " + std::to_string(heads_) +
                "\n#define DIM " + std::to_string(dimension_) +
                "\nkernel void " + name + R"METAL((
                    device const T* attended [[buffer(0)]],
                    device const T* gate [[buffer(1)]],
                    device T* output [[buffer(2)]],
                    constant long* strides [[buffer(3)]],
                    constant uint2& shape [[buffer(4)]],
                    uint tid [[thread_position_in_grid]]) {
                    uint begin = tid * 4u;
                    #pragma clang loop unroll(full)
                    for (uint lane = 0u; lane < 4u; ++lane) {
                        uint index = begin + lane;
                        if (index >= shape.y) return;
                        uint row = index / uint(HEADS * DIM);
                        uint feature = index % uint(HEADS * DIM);
                        uint batch = row / shape.x, token = row % shape.x;
                        uint head = feature / uint(DIM), dim = feature % uint(DIM);
                        long a = long(batch) * strides[0] + long(token) * strides[1] +
                            long(head) * strides[2] + long(dim) * strides[3];
                        long g = long(batch) * strides[4] + long(token) * strides[5] +
                            long(head) * strides[6] + long(dim) * strides[7];
                        float value = float(gate[g]);
                        float tail = 1.0f / (1.0f + metal::precise::exp(metal::abs(value)));
                        float probability = value < 0.0f ? tail : 1.0f - tail;
                        output[index] = T(float(attended[a]) * probability);
                    }
                })METAL";
        });
        return device.get_kernel(name, library);
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("QSA prefill output gate requires Metal");
    }

    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& output = outputs.front();
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(output, 2);
        std::array<std::int64_t, 8> strides;
        for (int input = 0; input < 2; ++input)
            for (int axis = 0; axis < 4; ++axis)
                strides[input * 4 + axis] = inputs[input].strides(axis);
        const std::array<std::uint32_t, 2> shape{
            static_cast<std::uint32_t>(output.shape(1)),
            static_cast<std::uint32_t>(output.size())};
        encoder.set_bytes(strides, 3);
        encoder.set_bytes(shape, 4);
        encoder.dispatch_threads(MTL::Size((output.size() + 3) / 4, 1, 1),
            MTL::Size(256, 1, 1));
    }

    const char* name() const override { return "QsaPrefillOutputGate"; }

private:
    mlx::core::Dtype dtype_;
    int heads_;
    int dimension_;
};

const std::string kGatedHcWeightHeader = std::string(detail::kNintMetadataSource) + R"METAL(
#ifndef DOWN_RPS
#define DOWN_RPS 2
#endif
inline uint mfq_hc_load_u32(device const uchar* stream, uint byte) {
    const packed_uchar4 values =
        *reinterpret_cast<device const packed_uchar4*>(stream + byte);
    return as_type<uint>(values);
}

inline uint mfq_hc_load_u32(constant const uchar* stream, uint byte) {
    return uint(stream[byte]) | (uint(stream[byte + 1u]) << 8u) |
        (uint(stream[byte + 2u]) << 16u) | (uint(stream[byte + 3u]) << 24u);
}

template <typename Stream>
inline uint4 mfq_hc_decode4(Stream stream, uint byte, uint shift, uint bits) {
    uint word = mfq_hc_load_u32(stream, byte);
    if (shift != 0u)
        word = (word >> shift) | (shift + 4u * bits > 32u
            ? uint(stream[byte + 4u]) << (32u - shift) : 0u);
    const uint mask = (1u << bits) - 1u;
    return uint4(word & mask, (word >> bits) & mask,
        (word >> (2u * bits)) & mask, (word >> (3u * bits)) & mask);
}

struct MfqHcCodes8 { uint4 low; uint4 high; };

template <int GS, typename Stream>
inline MfqHcCodes8 mfq_hc_decode8(Stream stream, uint byte, uint shift, uint bits) {
    const uint word0 = mfq_hc_load_u32(stream, byte);
    uint packed0, packed1;
    if constexpr (GS % 4 == 0) {
        if constexpr (GS % 8 == 0) shift = 0u;
        const uint word1 = bits > 4u ? mfq_hc_load_u32(stream, byte + 4u) : 0u;
        packed0 = shift == 0u ? word0 : (word0 >> 4u) | (word1 << 28u);
        const uint cursor = 4u * bits + shift;
        packed1 = cursor == 32u ? word1
            : (word0 >> cursor) | (word1 << (32u - cursor));
    } else {
        const uint word1 = shift + 8u * bits > 32u
            ? mfq_hc_load_u32(stream, byte + 4u) : 0u;
        const uint word2 = shift + 8u * bits > 64u ? uint(stream[byte + 8u]) : 0u;
        packed0 = shift == 0u ? word0
            : (word0 >> shift) | (word1 << (32u - shift));
        const uint cursor = shift + 4u * bits;
        const uint low = cursor >= 32u ? word1 : word0;
        const uint high = cursor >= 32u ? word2 : word1;
        const uint bit = cursor & 31u;
        packed1 = bit == 0u ? low
            : (low >> bit) | (high << (32u - bit));
    }
    const uint mask = (1u << bits) - 1u;
    return {
        uint4(packed0 & mask, (packed0 >> bits) & mask,
            (packed0 >> (2u * bits)) & mask, (packed0 >> (3u * bits)) & mask),
        uint4(packed1 & mask, (packed1 >> bits) & mask,
            (packed1 >> (2u * bits)) & mask, (packed1 >> (3u * bits)) & mask)
    };
}

template <int GS, int NG, int NR, typename Weight, typename Rows, typename Sub, typename Input>
inline void mfq_hc_dot_groups(
    Weight weight, Rows rows, Sub scales, Sub minima,
    uint row_base, uint begin, uint end, uint first_group, uint group_stride,
    Input input, uint input_origin, thread float* accumulators) {
    uint bits[NR], offsets[NR], shifts[NR];
    uint sub_offsets[NR], sub_shifts[NR], sub_bits[NR];
    float neuron_scales[NR], neuron_minima[NR];
    for (uint row = 0u; row < uint(NR); ++row) {
        const MfqNintRow metadata = mfq_nint_row(rows, row_base + row);
        bits[row] = metadata.q_bits;
        shifts[row] = metadata.q_shift;
        offsets[row] = metadata.q_offset;
        sub_offsets[row] = metadata.sub_offset;
        sub_shifts[row] = metadata.sub_shift;
        sub_bits[row] = metadata.sub_bits;
        neuron_scales[row] = metadata.scale;
        neuron_minima[row] = metadata.minimum;
    }
    for (uint group = first_group; group * uint(GS) < end; group += group_stride) {
        uint column = max(begin, group * uint(GS));
        const uint limit = min(end, (group + 1u) * uint(GS));
        float activation_sum = 0.0f;
        float quantized_dots[NR];
        for (uint row = 0u; row < uint(NR); ++row) quantized_dots[row] = 0.0f;
        for (; column + 7u < limit; column += 8u) {
            const uint local = column - input_origin;
            const float4 activation0 = float4(input[local], input[local + 1u],
                input[local + 2u], input[local + 3u]);
            const float4 activation1 = float4(input[local + 4u], input[local + 5u],
                input[local + 6u], input[local + 7u]);
            activation_sum += activation0.x + activation0.y + activation0.z + activation0.w
                + activation1.x + activation1.y + activation1.z + activation1.w;
            for (uint row = 0u; row < uint(NR); ++row) {
                const uint bit = shifts[row] + column * bits[row];
                const auto codes = mfq_hc_decode8<GS>(
                    weight, offsets[row] + (bit >> 3u), bit & 7u, bits[row]);
                quantized_dots[row] += dot(activation0, float4(codes.low))
                    + dot(activation1, float4(codes.high));
            }
        }
        for (; column + 3u < limit; column += 4u) {
            const uint local = column - input_origin;
            const float4 activation = float4(input[local], input[local + 1u],
                input[local + 2u], input[local + 3u]);
            activation_sum += activation.x + activation.y + activation.z + activation.w;
            for (uint row = 0u; row < uint(NR); ++row) {
                const uint bit = shifts[row] + column * bits[row];
                const auto codes = mfq_hc_decode4(
                    weight, offsets[row] + (bit >> 3u), bit & 7u, bits[row]);
                quantized_dots[row] += dot(activation, float4(codes));
            }
        }
        for (; column < limit; ++column) {
            const float activation = float(input[column - input_origin]);
            activation_sum += activation;
            for (uint row = 0u; row < uint(NR); ++row) {
                const uint bit = shifts[row] + column * bits[row];
                const uint byte = offsets[row] + (bit >> 3u);
                const uint shift = bit & 7u;
                uint packed = uint(weight[byte]);
                if (shift + bits[row] > 8u) packed |= uint(weight[byte + 1u]) << 8u;
                const uint code = (packed >> shift) & ((1u << bits[row]) - 1u);
                quantized_dots[row] += activation * float(code);
            }
        }
        for (uint row = 0u; row < uint(NR); ++row) {
            const float scale = neuron_scales[row] * float(mfq_nint_sub_value(
                scales, sub_offsets[row], sub_shifts[row], sub_bits[row], group));
            const float minimum = neuron_minima[row] * float(mfq_nint_sub_value(
                minima, sub_offsets[row], sub_shifts[row], sub_bits[row], group));
            accumulators[row] = fma(scale, quantized_dots[row],
                fma(-minimum, activation_sum, accumulators[row]));
        }
    }
}
)METAL";

mlx::core::Dtype hc_weight_dtype(const MlxLinear& weight) {
    if (weight.nint_weight_ref()) return mlx::core::float16;
    if (const auto* dense = weight.dense_weight_ref()) return dense->dtype();
    throw std::invalid_argument("Qwen4 MHC requires dense or NINT weights");
}

struct HcWeightInputs {
    array values;
    array rows;
    array scales;
    array minima;
    int group_size;
    int groups;
};

HcWeightInputs hc_weight_inputs(const MlxLinear& weight) {
    if (const auto* nint = weight.nint_weight_ref())
        return {nint->packed_values(), nint->row_metadata(),
            nint->sub_scales(), nint->sub_mins(), nint->group_size(), nint->groups()};
    static const auto empty = mlx::core::zeros(Shape{1}, mlx::core::uint32);
    if (const auto* dense = weight.dense_weight_ref())
        return {*dense, empty, empty, empty, 0, 0};
    throw std::invalid_argument("Qwen4 MHC requires dense or NINT weights");
}

array hc_project(const MlxLinear& weight, const array& input) {
    if (const auto* nint = weight.nint_weight_ref()) {
        const auto dtype = mlx::core::promote_types(input.dtype(), mlx::core::float16);
        return mlx::core::astype(nint->matmul(
            mlx::core::astype(input, dtype)), dtype);
    }
    return weight(input);
}


constexpr const char* kQsaDecodePrologueSource = R"METAL(
    constexpr uint VALUES_PER_THREAD = 4u;
    constexpr uint THREADS = 64u;
    constexpr uint SIMD_GROUPS = THREADS / 32u;
    constexpr uint TOTAL_HEADS =
        uint(QUERY_HEADS + KEY_HEADS + INDEX_HEADS);

    uint head_group = threadgroup_position_in_grid.x;
    uint token = threadgroup_position_in_grid.y;
    uint rows = uint(token_count[0]);
    uint local_thread = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    bool is_query = head_group < uint(QUERY_HEADS);
    bool is_key = !is_query &&
        head_group < uint(QUERY_HEADS + KEY_HEADS);
    uint head = is_query
        ? head_group
        : (is_key
            ? head_group - uint(QUERY_HEADS)
            : head_group - uint(QUERY_HEADS + KEY_HEADS));
    uint width = is_query || is_key
        ? uint(HEAD_DIM)
        : uint(INDEX_DIM);
    uint source_base = is_query
        ? (token * uint(QUERY_HEADS) + head) * uint(2 * HEAD_DIM)
        : (is_key
            ? (token * uint(KEY_HEADS) + head) * uint(HEAD_DIM)
            : (token * uint(INDEX_HEADS + 1) + head) * uint(INDEX_DIM));
    device const T* source = is_query
        ? query_gate_input
        : (is_key ? key_input : index_query_key_input);
    device const float* norm_weight = is_query
        ? query_weight
        : (is_key ? key_weight : index_weight);

    threadgroup float inverse_rms[1];
    threadgroup float local_sums[32];
    uint column_base = local_thread * VALUES_PER_THREAD;
    float sum_squares = 0.0f;
    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint column = column_base + item;
        if (column < width) {
            float value = float(source[source_base + column]);
            sum_squares += value * value;
        }
    }
    sum_squares = simd_sum(sum_squares);
    if (simd_group == 0u) local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) local_sums[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float total = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            inverse_rms[0] = metal::precise::rsqrt(
                total / float(width) + params[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint ROTARY_HALF = uint(ROTARY_DIM / 2);
    int position = positions[token];
    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint column = column_base + item;
        if (column >= width) continue;
        float normalized;
        if (column < uint(ROTARY_DIM)) {
            uint pair = column < ROTARY_HALF
                ? column
                : column - ROTARY_HALF;
            T first_rounded = T(
                float(source[source_base + pair])
                * inverse_rms[0] * norm_weight[pair]);
            T second_rounded = T(
                float(source[source_base + pair + ROTARY_HALF])
                * inverse_rms[0]
                * norm_weight[pair + ROTARY_HALF]);
            float exponent =
                -2.0f * float(pair) / float(ROTARY_DIM);
            float angle = float(position) * pow(params[1], exponent);
            float cosine = cos(angle);
            float sine = sin(angle);
            normalized = column < ROTARY_HALF
                ? float(first_rounded) * cosine
                    - float(second_rounded) * sine
                : float(second_rounded) * cosine
                    + float(first_rounded) * sine;
        } else {
            normalized = float(T(
                float(source[source_base + column])
                * inverse_rms[0] * norm_weight[column]));
        }
        uint output_index = (head * rows + token) * width + column;
        if (is_query) {
            query_output[output_index] = T(normalized);
            output_gate[(token * uint(QUERY_HEADS) + head) * width + column] = query_gate_input[
                source_base + uint(HEAD_DIM) + column];
        } else if (is_key) {
            key_output[output_index] = T(normalized);
        } else {
            index_query_output[(token * uint(INDEX_HEADS) + head) * width + column] = T(normalized);
        }
    }
)METAL";

constexpr const char* kGroupedRmsAffineSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    constexpr uint VALUES_PER_THREAD = 4u;
    constexpr uint SIMD_SIZE = 32u;
    threadgroup float inverse_rms[1];
    threadgroup float local_sums[SIMD_SIZE];

    uint input_offset = row * uint(GROUP_SIZE);
    uint weight_offset =
        (row % uint(GROUP_COUNT)) * uint(GROUP_SIZE);
    uint column = local_thread * VALUES_PER_THREAD;
    float values[VALUES_PER_THREAD] = {
        0.0f, 0.0f, 0.0f, 0.0f};
    float sum_squares = 0.0f;
    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            float value = float(input[input_offset + current]);
            values[item] = value;
            sum_squares += value * value;
        }
    }

    // Mirror MLX's non-looped FP32 RMS kernel exactly: four adjacent reads
    // per thread, SIMD reduction, then a 32-slot cross-SIMD reduction.
    sum_squares = simd_sum(sum_squares);
    if (simd_group == 0u)
        local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u)
        local_sums[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float total = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            inverse_rms[0] = metal::precise::rsqrt(
                total / float(GROUP_SIZE) + epsilon[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            float scaled = (values[item] * inverse_rms[0]) *
                (1.0f + float(weight[weight_offset + current]));
            output[input_offset + current] = T(scaled);
        }
    }
)METAL";

constexpr const char* kGroupedRmsAffineWriteSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    constexpr uint VALUES_PER_THREAD = 4u;
    constexpr uint SIMD_SIZE = 32u;
    threadgroup float inverse_rms[1];
    threadgroup float local_sums[SIMD_SIZE];

    uint token = row / uint(HC_COUNT);
    uint stream = row % uint(HC_COUNT);
    uint input_offset = row * uint(GROUP_SIZE);
    uint branch_offset = token * uint(GROUP_SIZE);
    uint weight_offset = stream * uint(GROUP_SIZE);
    G gate = injection[token * uint(HC_COUNT) + stream];
    uint column = local_thread * VALUES_PER_THREAD;
    float values[VALUES_PER_THREAD] = {
        0.0f, 0.0f, 0.0f, 0.0f};
    float sum_squares = 0.0f;
    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            // Preserve the standalone residual kernel's promoted multiply,
            // explicit rounding, and add order before normalizing the result.
            U update = U(U(branch[branch_offset + current]) * U(gate));
            O value = O(O(residual[input_offset + current]) + O(update));
            updated[input_offset + current] = value;
            values[item] = float(value);
            sum_squares += values[item] * values[item];
        }
    }

    sum_squares = simd_sum(sum_squares);
    if (simd_group == 0u)
        local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u)
        local_sums[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float total = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            inverse_rms[0] = metal::precise::rsqrt(
                total / float(GROUP_SIZE) + epsilon[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            float scaled = (values[item] * inverse_rms[0]) *
                (1.0f + float(weight[weight_offset + current]));
            normalized[input_offset + current] = O(scaled);
        }
    }
)METAL";

constexpr const char* kGatedHcProjectionSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    constexpr uint ROWS_PER_WORKGROUP = 4u;
    constexpr uint DOWN_WORKGROUPS = uint(LOW_RANK) / ROWS_PER_WORKGROUP;
    constexpr uint SIMD_GROUPS = 8u;
    threadgroup float partials[SIMD_GROUPS * ROWS_PER_WORKGROUP];

    bool is_injection = workgroup >= DOWN_WORKGROUPS;
    uint matrix_workgroup = is_injection
        ? workgroup - DOWN_WORKGROUPS
        : workgroup;
    uint row_base = matrix_workgroup * ROWS_PER_WORKGROUP;

    // This is the same 8-way K split and four-adjacent-value traversal used
    // by MLX's BF16/F16 GEMV for the decode geometry. Both projections share
    // normalized, so one launch can produce the 320 low-rank values and the
    // four residual-injection values without a second tiny GEMV dispatch.
    float accumulators[ROWS_PER_WORKGROUP] = {
        0.0f, 0.0f, 0.0f, 0.0f};
    uint column = simd_group * 128u + lane * 4u;
    for (; column < uint(WIDTH); column += 1024u) {
        for (uint item = 0u; item < 4u; ++item) {
            float value = float(normalized[column + item]);
            for (uint row = 0u; row < ROWS_PER_WORKGROUP; ++row) {
                uint offset = (row_base + row) * uint(WIDTH) + column + item;
                float weight = is_injection
                    ? float(injection_weight[offset])
                    : float(down_weight[offset]);
                accumulators[row] += weight * value;
            }
        }
    }

    for (uint row = 0u; row < ROWS_PER_WORKGROUP; ++row) {
        float value = accumulators[row];
        for (ushort delta = 16; delta >= 1; delta >>= 1)
            value += simd_shuffle_down(value, delta);
        if (lane == 0u)
            partials[simd_group * ROWS_PER_WORKGROUP + row] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u && lane < ROWS_PER_WORKGROUP) {
        float value = partials[lane];
        for (uint group = 1u; group < SIMD_GROUPS; ++group)
            value += partials[group * ROWS_PER_WORKGROUP + lane];
        uint output_row = row_base + lane;
        if (is_injection) {
            I projected = I(value);
            I scaled = I(projected / I(HC_COUNT));
            I tail = I(1) /
                (I(1) + metal::exp(metal::abs(scaled)));
            I sigmoid = scaled < I(0) ? tail : I(1) - tail;
            injection[output_row] = I(I(2) * sigmoid);
        } else {
            down[output_row] = D(value);
        }
    }
)METAL";

constexpr const char* kGatedHcUpCollapseSource = R"METAL(
    constexpr uint FEATURES_PER_TG = 8u;
    constexpr uint CHUNK = 32u;
    constexpr uint CHUNKS = (uint(LOW_RANK) + CHUNK - 1u) / CHUNK;
    constexpr uint OUTPUT_ROWS = FEATURES_PER_TG * uint(HC_COUNT);
    constexpr uint PART_STRIDE = CHUNKS + 1u;
    static_assert(HC_COUNT == 4, "coalesced HC up expects four streams");
    static_assert(LOW_RANK == 320, "coalesced HC up is tuned for Qwen4");
    static_assert(CHUNKS * OUTPUT_ROWS == 320,
                  "one thread per up-projection chunk");

    const uint tid = thread_index_in_threadgroup;
    const uint chunk = tid % CHUNKS;
    const uint output_row = tid / CHUNKS;
    const uint stream = output_row & 3u;
    const uint local_feature = output_row >> 2u;
    const uint feature =
        threadgroup_position_in_grid.x * FEATURES_PER_TG + local_feature;

    // All 320 threads first produce one activated low-rank value.  The old
    // kernel recomputed these values independently in every SIMD group.
    threadgroup L activated[LOW_RANK];
    threadgroup float partials[OUTPUT_ROWS * PART_STRIDE];
    if (tid < uint(LOW_RANK)) {
        L value = L(down_projection[tid] / L(HC_COUNT));
        L tail = L(1) /
            (L(1) + metal::exp(metal::abs(value)));
        L sigmoid = value < L(0) ? tail : L(1) - tail;
        activated[tid] = L(value * sigmoid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulator = 0.0f;
    const uint low_base = chunk * CHUNK;
    if (feature < uint(HIDDEN)) {
        const uint row = stream * uint(HIDDEN) + feature;
        const uint weight_base = row * uint(LOW_RANK) + low_base;
        for (uint item = 0u; item < CHUNK; ++item) {
            const uint low = low_base + item;
            if (low < uint(LOW_RANK)) {
                accumulator += float(up_weight[weight_base + item]) *
                    float(activated[low]);
            }
        }
    }
    partials[output_row * PART_STRIDE + chunk] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The first SIMD group reduces ten contiguous chunks for 32 projection
    // rows (four streams by eight features), then performs the stream mix.
    if (tid < OUTPUT_ROWS) {
        const uint result_stream = tid & 3u;
        const uint result_feature =
            threadgroup_position_in_grid.x * FEATURES_PER_TG + (tid >> 2u);
        float projected = 0.0f;
        const uint base = tid * PART_STRIDE;
        for (uint current = 0u; current < CHUNKS; ++current)
            projected += partials[base + current];
        T rounded = T(projected);
        T tail = T(1) /
            (T(1) + metal::exp(metal::abs(rounded)));
        T gate = rounded < T(0) ? tail : T(1) - tail;
        T product = T(gate * T(normalized[
            result_stream * uint(HIDDEN) + result_feature]));
        float mixed = float(product);
        mixed += simd_shuffle_down(mixed, 1);
        mixed += simd_shuffle_down(mixed, 2);
        if (result_stream == 0u && result_feature < uint(HIDDEN))
            branch[result_feature] = T(mixed / float(HC_COUNT));
    }
)METAL";

// Decode-only two-stage hyper-connection path.  The first launch repeats the
// tiny per-stream norm in each projection row group so the 10240x320 down
// projection can consume threadgroup-resident values without materializing a
// separate norm launch.  Four stream-slice sums are joined by the up launch.
constexpr const char* kGatedHcNormDownSource = R"METAL(
    constexpr uint STREAMS = 4u;
    constexpr uint ROWS_PER_SIMD = uint(DOWN_RPS);
    constexpr uint ROWS_PER_TG = 8u * ROWS_PER_SIMD;
    constexpr uint DOWN_GROUPS = uint(LOW_RANK) / ROWS_PER_TG;
    constexpr uint GROUPS = DOWN_GROUPS + uint(HAS_INJECTION);
    constexpr uint PER = (uint(HIDDEN) + 255u) / 256u;
    constexpr uint PART_STRIDE = uint(LOW_RANK + HC_COUNT);
    static_assert(HC_COUNT == 4 && HIDDEN == 2560 && LOW_RANK == 320,
                  "Qwen4 HC decode geometry changed");

    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint workgroup = threadgroup_position_in_grid.x;
    const uint stream = workgroup & 3u;
    const uint group = workgroup >> 2u;
    const uint input_base = stream * uint(HIDDEN);

    threadgroup N normalized_stream[HIDDEN];
    threadgroup float norm_partials[8];
    float values[PER];
    float scales[PER];
    float sum_squares = 0.0f;
    for (uint item = 0u; item < PER; ++item) {
        const uint feature = tid + item * 256u;
        float value = 0.0f;
        float scale = 0.0f;
        if (feature < uint(HIDDEN)) {
            value = float(input[input_base + feature]);
            scale = 1.0f + float(norm_weight[input_base + feature]);
        }
        values[item] = value;
        scales[item] = scale;
        sum_squares += value * value;
    }
    sum_squares = simd_sum(sum_squares);
    if (lane == 0u) norm_partials[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint part = 0u; part < 8u; ++part)
        total += norm_partials[part];
    const float inverse_rms = metal::rsqrt(
        total / float(HIDDEN) + epsilon[0]);
    N stored = N(0);
    for (uint item = 0u; item < PER; ++item) {
        const uint feature = tid + item * 256u;
        if (feature < uint(HIDDEN)) {
            const N value = N(values[item] * inverse_rms * scales[item]);
            normalized_stream[feature] = value;
            if (item == group) stored = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* output_parts =
        parts + stream * PART_STRIDE;
    if (group < DOWN_GROUPS) {
        const uint first_row =
            group * ROWS_PER_TG + simd_group * ROWS_PER_SIMD;
        float accumulators[ROWS_PER_SIMD] = {0.0f};
        if constexpr (DOWN_GS != 0) {
            mfq_hc_dot_groups<DOWN_GS, DOWN_NG, ROWS_PER_SIMD>(
                down_weight, down_rows, down_scales, down_minima,
                first_row, input_base, input_base + uint(HIDDEN),
                input_base / uint(DOWN_GS) + lane, 32u,
                normalized_stream, input_base, accumulators);
        } else {
            for (uint feature = lane * 4u;
                 feature < uint(HIDDEN);
                 feature += 128u) {
                float inputs[4];
                for (uint item = 0u; item < 4u; ++item)
                    inputs[item] = float(normalized_stream[feature + item]);
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    const uint weight_base =
                        (first_row + row) * uint(HIDDEN * HC_COUNT) +
                        input_base + feature;
                    for (uint item = 0u; item < 4u; ++item)
                        accumulators[row] +=
                            float(down_weight[weight_base + item]) * inputs[item];
                }
            }
        }
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            const float value = simd_sum(accumulators[row]);
            if (lane == 0u) output_parts[first_row + row] = value;
        }
    } else if (HAS_INJECTION != 0 && simd_group < uint(HC_COUNT)) {
        float accumulator = 0.0f;
        const uint weight_row = simd_group;
        if constexpr (INJECT_GS != 0) {
            mfq_hc_dot_groups<INJECT_GS, INJECT_NG, 1>(
                injection_weight, injection_rows, injection_scales, injection_minima,
                weight_row, input_base, input_base + uint(HIDDEN),
                input_base / uint(INJECT_GS) + lane, 32u,
                normalized_stream, input_base, &accumulator);
        } else {
            for (uint feature = lane * 4u;
                 feature < uint(HIDDEN);
                 feature += 128u) {
                const uint weight_base =
                    weight_row * uint(HIDDEN * HC_COUNT) + input_base + feature;
                for (uint item = 0u; item < 4u; ++item) {
                    accumulator += float(injection_weight[weight_base + item]) *
                        float(normalized_stream[feature + item]);
                }
            }
        }
        accumulator = simd_sum(accumulator);
        if (lane == 0u)
            output_parts[uint(LOW_RANK) + weight_row] = accumulator;
    }

    // Store after all device weight reads.  This avoids a conservative device
    // read-after-write fence in Apple's Metal compiler.
    if (group < PER) {
        const uint feature = tid + group * 256u;
        if (feature < uint(HIDDEN))
            normalized[input_base + feature] = stored;
    }
)METAL";

constexpr const char* kGatedHcWriteNormDownSource = R"METAL(
    constexpr uint STREAMS = 4u;
    constexpr uint ROWS_PER_SIMD = uint(DOWN_RPS);
    constexpr uint ROWS_PER_TG = 8u * ROWS_PER_SIMD;
    constexpr uint DOWN_GROUPS = uint(LOW_RANK) / ROWS_PER_TG;
    constexpr uint GROUPS = DOWN_GROUPS + uint(HAS_INJECTION);
    constexpr uint PER = (uint(HIDDEN) + 255u) / 256u;
    constexpr uint PART_STRIDE = uint(LOW_RANK + HC_COUNT);
    static_assert(HC_COUNT == 4 && HIDDEN == 2560 && LOW_RANK == 320,
                  "Qwen4 HC decode geometry changed");

    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint workgroup = threadgroup_position_in_grid.x;
    const uint stream = workgroup & 3u;
    const uint group = workgroup >> 2u;
    const uint input_base = stream * uint(HIDDEN);

    threadgroup O normalized_stream[HIDDEN];
    threadgroup float norm_partials[8];
    float values[PER];
    float scales[PER];
    const G gate_value = previous_injection[stream];
    float sum_squares = 0.0f;
    O written = O(0);
    for (uint item = 0u; item < PER; ++item) {
        const uint feature = tid + item * 256u;
        float value = 0.0f;
        float scale = 0.0f;
        if (feature < uint(HIDDEN)) {
            #pragma clang fp contract(off)
            U update = U(U(previous_branch[feature]) * U(gate_value));
            written = O(O(previous_residual[input_base + feature]) + O(update));
            value = float(written);
            scale = 1.0f + float(norm_weight[input_base + feature]);
        }
        values[item] = value;
        scales[item] = scale;
        sum_squares += value * value;
    }
    sum_squares = simd_sum(sum_squares);
    if (lane == 0u) norm_partials[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint part = 0u; part < 8u; ++part)
        total += norm_partials[part];
    const float inverse_rms = metal::rsqrt(
        total / float(HIDDEN) + epsilon[0]);
    O stored_norm = O(0);
    O stored_residual = O(0);
    for (uint item = 0u; item < PER; ++item) {
        const uint feature = tid + item * 256u;
        if (feature < uint(HIDDEN)) {
            const O value = O(values[item]);
            const O normed = O(float(value) * inverse_rms * scales[item]);
            normalized_stream[feature] = normed;
            if (item == group) {
                stored_norm = normed;
                stored_residual = value;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* output_parts = parts + stream * PART_STRIDE;
    if (group < DOWN_GROUPS) {
        const uint first_row =
            group * ROWS_PER_TG + simd_group * ROWS_PER_SIMD;
        float accumulators[ROWS_PER_SIMD] = {0.0f};
        if constexpr (DOWN_GS != 0) {
            mfq_hc_dot_groups<DOWN_GS, DOWN_NG, ROWS_PER_SIMD>(
                down_weight, down_rows, down_scales, down_minima,
                first_row, input_base, input_base + uint(HIDDEN),
                input_base / uint(DOWN_GS) + lane, 32u,
                normalized_stream, input_base, accumulators);
        } else {
            for (uint feature = lane * 4u;
                 feature < uint(HIDDEN);
                 feature += 128u) {
                float inputs[4];
                for (uint item = 0u; item < 4u; ++item)
                    inputs[item] = float(normalized_stream[feature + item]);
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    const uint weight_base =
                        (first_row + row) * uint(HIDDEN * HC_COUNT) +
                        input_base + feature;
                    for (uint item = 0u; item < 4u; ++item)
                        accumulators[row] +=
                            float(down_weight[weight_base + item]) * inputs[item];
                }
            }
        }
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            const float value = simd_sum(accumulators[row]);
            if (lane == 0u) output_parts[first_row + row] = value;
        }
    } else if (HAS_INJECTION != 0 && simd_group < uint(HC_COUNT)) {
        float accumulator = 0.0f;
        const uint weight_row = simd_group;
        if constexpr (INJECT_GS != 0) {
            mfq_hc_dot_groups<INJECT_GS, INJECT_NG, 1>(
                injection_weight, injection_rows, injection_scales, injection_minima,
                weight_row, input_base, input_base + uint(HIDDEN),
                input_base / uint(INJECT_GS) + lane, 32u,
                normalized_stream, input_base, &accumulator);
        } else {
            for (uint feature = lane * 4u;
                 feature < uint(HIDDEN);
                 feature += 128u) {
                const uint weight_base =
                    weight_row * uint(HIDDEN * HC_COUNT) + input_base + feature;
                for (uint item = 0u; item < 4u; ++item) {
                    accumulator += float(injection_weight[weight_base + item]) *
                        float(normalized_stream[feature + item]);
                }
            }
        }
        accumulator = simd_sum(accumulator);
        if (lane == 0u)
            output_parts[uint(LOW_RANK) + weight_row] = accumulator;
    }

    if (group < PER) {
        const uint feature = tid + group * 256u;
        if (feature < uint(HIDDEN)) {
            const uint index = input_base + feature;
            normalized[index] = stored_norm;
            residual[index] = stored_residual;
        }
    }
)METAL";


constexpr const char* kGatedHcPartsUpSource = R"METAL(
    constexpr uint FEATURES_PER_TG = 8u;
    constexpr uint CHUNK = UP_GS == 0 ? 32u
        : uint(UP_GS) * ((32u + uint(UP_GS) - 1u) / uint(UP_GS));
    constexpr uint CHUNKS = (uint(LOW_RANK) + CHUNK - 1u) / CHUNK;
    constexpr uint OUTPUT_ROWS = FEATURES_PER_TG * uint(HC_COUNT);
    constexpr uint THREADS = OUTPUT_ROWS * CHUNKS;
    constexpr uint PART_STRIDE = uint(LOW_RANK + HC_COUNT);
    constexpr uint REDUCE_STRIDE = CHUNKS + 1u;
    static_assert(HC_COUNT == 4 && LOW_RANK == 320,
                  "Qwen4 HC up geometry changed");

    const uint tid = thread_index_in_threadgroup;
    const uint chunk = tid % CHUNKS;
    const uint output_row = tid / CHUNKS;
    const uint stream = output_row & 3u;
    const uint local_feature = output_row >> 2u;
    const uint feature =
        threadgroup_position_in_grid.x * FEATURES_PER_TG + local_feature;

    threadgroup L activated[LOW_RANK];
    threadgroup float projection_partials[OUTPUT_ROWS * REDUCE_STRIDE];
    for (uint low = tid; low < uint(LOW_RANK); low += THREADS) {
        float value = parts[low] + parts[PART_STRIDE + low] +
            parts[2u * PART_STRIDE + low] +
            parts[3u * PART_STRIDE + low];
        L rounded = L(value / float(HC_COUNT));
        L tail = L(1) /
            (L(1) + metal::exp(metal::abs(rounded)));
        L sigmoid = rounded < L(0) ? tail : L(1) - tail;
        activated[low] = L(rounded * sigmoid);
    }
    if (HAS_INJECTION != 0 && tid < uint(HC_COUNT)) {
        const uint index = uint(LOW_RANK) + tid;
        float value = parts[index] + parts[PART_STRIDE + index] +
            parts[2u * PART_STRIDE + index] +
            parts[3u * PART_STRIDE + index];
        I rounded = I(value / float(HC_COUNT));
        I tail = I(1) /
            (I(1) + metal::exp(metal::abs(rounded)));
        I sigmoid = rounded < I(0) ? tail : I(1) - tail;
        injection[tid] = I(I(2) * sigmoid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulator = 0.0f;
    const uint low_base = chunk * CHUNK;
    if (feature < uint(HIDDEN)) {
        const uint row = stream * uint(HIDDEN) + feature;
        if constexpr (UP_GS != 0) {
            mfq_hc_dot_groups<UP_GS, UP_NG, 1>(
                up_weight, up_rows, up_scales, up_minima,
                row, low_base, min(low_base + CHUNK, uint(LOW_RANK)),
                low_base / uint(UP_GS), 1u, activated, 0u, &accumulator);
        } else {
            const uint weight_base = row * uint(LOW_RANK) + low_base;
            for (uint item = 0u; item < CHUNK; ++item) {
                accumulator += float(up_weight[weight_base + item]) *
                    float(activated[low_base + item]);
            }
        }
    }
    projection_partials[output_row * REDUCE_STRIDE + chunk] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < OUTPUT_ROWS) {
        const uint result_stream = tid & 3u;
        const uint result_feature =
            threadgroup_position_in_grid.x * FEATURES_PER_TG + (tid >> 2u);
        float projected = 0.0f;
        const uint base = tid * REDUCE_STRIDE;
        for (uint current = 0u; current < CHUNKS; ++current)
            projected += projection_partials[base + current];
        T rounded = T(projected);
        T tail = T(1) /
            (T(1) + metal::exp(metal::abs(rounded)));
        T gate = rounded < T(0) ? tail : T(1) - tail;
        T product = T(gate * T(normalized[
            result_stream * uint(HIDDEN) + result_feature]));
        float mixed = float(product);
        mixed += simd_shuffle_down(mixed, 1);
        mixed += simd_shuffle_down(mixed, 2);
        if (result_stream == 0u && result_feature < uint(HIDDEN))
            branch[result_feature] = T(mixed / float(HC_COUNT));
    }
)METAL";

struct PackedHcConfig {
    mlx::core::Dtype residual_type;
    mlx::core::Dtype branch_type;
    mlx::core::Dtype gate_type;
    mlx::core::Dtype norm_type;
    mlx::core::Dtype update_type;
    mlx::core::Dtype output_type;
    mlx::core::Dtype low_type;
    mlx::core::Dtype result_type;
    mlx::core::Dtype injection_type;
    int down_gs, down_ng, up_gs, up_ng, injection_gs, injection_ng;
    int down_rows_per_simd;
    bool after, has_injection;
    float eps;
};

std::string packed_hc_source(const PackedHcConfig& config, const std::string& key) {
    const auto type = [](mlx::core::Dtype dtype) {
        if (dtype == mlx::core::float16) return "half";
        if (dtype == mlx::core::bfloat16) return "bfloat16_t";
        if (dtype == mlx::core::float32) return "float";
        throw std::invalid_argument("packed MHC requires floating input");
    };
    std::string source = "#include <metal_stdlib>\nusing namespace metal;\nusing bfloat16_t = bfloat;\n"
        "namespace metal { inline bfloat abs(bfloat x) { return bfloat(abs(float(x))); } "
        "inline bfloat exp(bfloat x) { return bfloat(exp(float(x))); } }\n";
    const std::pair<const char*, mlx::core::Dtype> aliases[] = {
        {"R", config.residual_type}, {"B", config.branch_type}, {"G", config.gate_type},
        {"W", config.norm_type}, {"U", config.update_type}, {"O", config.output_type},
        {"N", config.output_type}, {"L", config.low_type}, {"T", config.result_type},
        {"I", config.injection_type},
    };
    for (const auto& [name, dtype] : aliases)
        source += "using " + std::string(name) + " = " + type(dtype) + ";\n";
    for (const auto& [name, value] : {
            std::pair{"HIDDEN", 2560}, {"LOW_RANK", 320}, {"HC_COUNT", 4},
            {"DOWN_GS", config.down_gs}, {"DOWN_NG", config.down_ng},
            {"UP_GS", config.up_gs}, {"UP_NG", config.up_ng},
            {"INJECT_GS", config.injection_gs}, {"INJECT_NG", config.injection_ng},
            {"DOWN_RPS", config.down_rows_per_simd},
            {"HAS_INJECTION", int(config.has_injection)}})
        source += "#define " + std::string(name) + " " + std::to_string(value) + "\n";
    source += kGatedHcWeightHeader;
    const std::string builtins =
        ", uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]"
        ", uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]"
        ", uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]"
        ", uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n";
    source += "kernel void " + key + "_down(";
    source += config.after
        ? "device const R* residual_storage [[buffer(0)]], device const B* branch_storage [[buffer(1)]], device const G* gate_storage [[buffer(2)]], "
        : "device const R* input_storage [[buffer(0)]], ";
    source +=
        "device const W* norm_weight [[buffer(3)]], "
        "device const uchar* down_weight [[buffer(4)]], device const uint* down_rows [[buffer(5)]], "
        "device const uchar* down_scales [[buffer(6)]], device const uchar* down_minima [[buffer(7)]], "
        "device const uchar* injection_weight [[buffer(12)]], device const uint* injection_rows [[buffer(13)]], "
        "device const uchar* injection_scales [[buffer(14)]], device const uchar* injection_minima [[buffer(15)]], "
        "device O* normalized_storage [[buffer(16)]], device float* parts_storage [[buffer(17)]], "
        "device O* output_storage [[buffer(18)]], constant float* epsilon [[buffer(19)]]";
    source += builtins;
    source += "const uint token = threadgroup_position_in_grid.y;\n"
        "device O* normalized = normalized_storage + token * 10240u;\n"
        "device float* parts = parts_storage + token * 1296u;\n";
    source += config.after
        ? "device const R* previous_residual = residual_storage + token * 10240u;\n"
          "device const B* previous_branch = branch_storage + token * 2560u;\n"
          "device const G* previous_injection = gate_storage + token * 4u;\n"
          "device O* residual = output_storage + token * 10240u;\n"
        : "device const R* input = input_storage + token * 10240u;\n";
    source += config.after ? kGatedHcWriteNormDownSource : kGatedHcNormDownSource;
    source += "}\nkernel void " + key + "_up("
        "device const uchar* up_weight [[buffer(8)]], device const uint* up_rows [[buffer(9)]], "
        "device const uchar* up_scales [[buffer(10)]], device const uchar* up_minima [[buffer(11)]], "
        "device const O* normalized_storage [[buffer(16)]], device const float* parts_storage [[buffer(17)]], "
        "device T* branch_storage [[buffer(20)]], device I* injection_storage [[buffer(21)]]";
    source += builtins;
    source += "const uint token = threadgroup_position_in_grid.y;\n"
        "device const O* normalized = normalized_storage + token * 10240u;\n"
        "device const float* parts = parts_storage + token * 1296u;\n"
        "device T* branch = branch_storage + token * 2560u;\n"
        "device I* injection = injection_storage + token * 4u;\n";
    source += kGatedHcPartsUpSource;
    source += "}\n";
    return source;
}

class PackedHcPrimitive final : public mlx::core::Primitive, public MlxPreparableKernel {
public:
    PackedHcPrimitive(mlx::core::Stream stream, PackedHcConfig config)
        : Primitive(stream), config_(config) {
        key_ = "mfq_packed_hc_v3";
        for (auto dtype : {config.residual_type, config.branch_type, config.gate_type,
                config.norm_type, config.update_type, config.output_type, config.low_type,
                config.result_type, config.injection_type})
            key_ += "_" + mlx::core::type_to_name(dtype);
        for (int value : {config.down_gs, config.down_ng, config.up_gs, config.up_ng,
                config.injection_gs, config.injection_ng, config.down_rows_per_simd,
                int(config.after), int(config.has_injection)})
            key_ += "_" + std::to_string(value);
    }

    std::string preparation_key() const override { return key_; }
    void prepare_gpu() override { (void)prepared_kernels(); }

    std::pair<MTL::ComputePipelineState*, MTL::ComputePipelineState*> prepared_kernels() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(key_, options, [this] { return packed_hc_source(config_, key_); });
        return {device.get_kernel(key_ + "_down", library), device.get_kernel(key_ + "_up", library)};
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("packed MHC requires Metal");
    }

    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        for (auto& output : outputs)
            output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        const int rows = static_cast<int>(inputs[0].size() / 10240);
        array normalized(Shape{rows, 10240}, config_.output_type, nullptr, {});
        array parts(Shape{rows, 4, 324}, mlx::core::float32, nullptr, {});
        normalized.set_data(mlx::core::allocator::malloc(normalized.nbytes()));
        parts.set_data(mlx::core::allocator::malloc(parts.nbytes()));
        const auto kernels = prepared_kernels();
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        for (int index = 0; index < int(inputs.size()); ++index)
            encoder.set_input_array(inputs[index], index);
        encoder.set_output_array(normalized, 16);
        encoder.set_output_array(parts, 17);
        if (config_.after) encoder.set_output_array(outputs[2], 18);
        encoder.set_bytes(config_.eps, 19);
        encoder.set_compute_pipeline_state(kernels.first);
        const int down_groups = 320 / (8 * config_.down_rows_per_simd);
        encoder.dispatch_threadgroups(MTL::Size((down_groups + int(config_.has_injection)) * 4, rows, 1),
                                      MTL::Size(256, 1, 1));
        encoder.set_input_array(normalized, 16);
        encoder.set_input_array(parts, 17);
        encoder.set_output_array(outputs[0], 20);
        encoder.set_output_array(outputs[1], 21);
        const int chunk = config_.up_gs * ((32 + config_.up_gs - 1) / config_.up_gs);
        const int threads = 32 * ((320 + chunk - 1) / chunk);
        encoder.set_compute_pipeline_state(kernels.second);
        encoder.dispatch_threadgroups(MTL::Size(320, rows, 1), MTL::Size(threads, 1, 1));
        encoder.add_temporary(std::move(normalized));
        encoder.add_temporary(std::move(parts));
    }

    const char* name() const override { return "PackedHcPrimitive"; }

private:
    PackedHcConfig config_;
    std::string key_;
};

bool can_use_packed_hc(
    const array& input, const array& norm,
    const MlxLinear& down, const MlxLinear& up,
    const std::optional<MlxLinear>& injection) {
    return input.flags().row_contiguous && norm.flags().row_contiguous
        && down.nint_weight_ref() && up.nint_weight_ref()
        && (!injection || injection->nint_weight_ref());
}

MlxQwen4GatedResidualPre packed_gated_hc(
    const array& input, const std::optional<array>& previous_branch,
    const std::optional<array>& previous_injection, const array& norm,
    const MlxLinear& down_weight, const MlxLinear& up_weight,
    const std::optional<MlxLinear>& injection_weight, float eps) {
    const bool after = previous_branch.has_value();
    const auto branch = after ? *previous_branch : input;
    const auto gate = after ? *previous_injection : input;
    const auto update_type = after
        ? mlx::core::promote_types(branch.dtype(), gate.dtype()) : input.dtype();
    const auto output_type = mlx::core::promote_types(input.dtype(), update_type);
    const auto low_type = mlx::core::promote_types(output_type, mlx::core::float16);
    const auto result_type = mlx::core::promote_types(low_type, mlx::core::float16);
    const auto injection_type = injection_weight
        ? mlx::core::promote_types(output_type, mlx::core::float16) : output_type;
    const auto down = hc_weight_inputs(down_weight);
    const auto up = hc_weight_inputs(up_weight);
    const auto injection = hc_weight_inputs(injection_weight ? *injection_weight : down_weight);
    PackedHcConfig config{
        input.dtype(), branch.dtype(), gate.dtype(), norm.dtype(), update_type, output_type,
        low_type, result_type, injection_type,
        down.group_size, down.groups, up.group_size, up.groups, injection.group_size, injection.groups,
        input.size() / 10240 == 1 ? 1 : 2,
        after, injection_weight.has_value(), eps};
    auto branch_shape = input.shape();
    branch_shape.back() = 2560;
    auto injection_shape = input.shape();
    injection_shape.back() = 4;
    auto outputs = array::make_arrays(
        after ? std::vector<Shape>{branch_shape, injection_shape, input.shape()}
              : std::vector<Shape>{branch_shape, injection_shape},
        after ? std::vector<mlx::core::Dtype>{result_type, injection_type, output_type}
              : std::vector<mlx::core::Dtype>{result_type, injection_type},
        std::make_shared<PackedHcPrimitive>(
            mlx::core::default_stream(mlx::core::default_device()), config),
        {input, branch, gate, norm, down.values, down.rows, down.scales, down.minima,
         up.values, up.rows, up.scales, up.minima,
         injection.values, injection.rows, injection.scales, injection.minima});
    return {std::move(outputs[0]), after ? std::move(outputs[2]) : input,
        injection_weight ? std::optional<array>(std::move(outputs[1])) : std::nullopt};
}

const mlx::core::fast::CustomKernelFunction& grouped_rms_affine_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_grouped_rms_affine",
            {"input", "weight", "epsilon"},
            {"output"},
            kGroupedRmsAffineSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
grouped_rms_affine_write_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_grouped_rms_affine_write",
            {
                "residual",
                "branch",
                "injection",
                "weight",
                "epsilon",
            },
            {"updated", "normalized"},
            kGroupedRmsAffineWriteSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

struct QsaPrologueConfig {
    mlx::core::Dtype dtype;
    std::array<int, 6> geometry;
    std::array<float, 2> params;
};

class QsaProloguePrimitive final : public mlx::core::Primitive, public MlxPreparableKernel {
public:
    QsaProloguePrimitive(mlx::core::Stream stream, QsaPrologueConfig config)
        : Primitive(stream), config_(config),
          kernel_name_("mfq_qsa_decode_prologue_v2") {
        kernel_name_ += "_" + mlx::core::type_to_name(config_.dtype);
        for (int value : config_.geometry) {
            kernel_name_ += "_" + std::to_string(value);
        }
    }

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_, options, [this] { return source(); });
        return device.get_kernel(kernel_name_, library);
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("QSA decode prologue requires Metal");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        for (auto& output : outputs) {
            output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        }
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(
            kernel);
        for (int index = 0; index < 7; ++index) {
            encoder.set_input_array(inputs[index], index);
        }
        encoder.set_bytes(config_.params, 7);
        const int rows = inputs[0].shape(1);
        encoder.set_bytes(rows, 12);
        for (int index = 0; index < 4; ++index) {
            encoder.set_output_array(outputs[index], index + 8);
        }
        encoder.dispatch_threadgroups(
            MTL::Size(config_.geometry[0] + config_.geometry[1] +
                config_.geometry[2], rows, 1),
            MTL::Size(64, 1, 1));
    }

    const char* name() const override { return "QsaDecodePrologue"; }

private:
    std::string source() const {
        std::string code =
            "#include <metal_stdlib>\nusing namespace metal;\n";
        const char* type = config_.dtype == mlx::core::float16
            ? "half" : config_.dtype == mlx::core::bfloat16 ? "bfloat" : "float";
        code += "using T = " + std::string(type) + ";\n";
        const std::array<const char*, 6> names{
            "QUERY_HEADS", "KEY_HEADS", "INDEX_HEADS",
            "HEAD_DIM", "INDEX_DIM", "ROTARY_DIM"};
        for (int index = 0; index < 6; ++index) {
            code += "#define " + std::string(names[index]) + " " +
                std::to_string(config_.geometry[index]) + "\n";
        }
        code += "kernel void " + kernel_name_ + "("
            "device const T* query_gate_input [[buffer(0)]], "
            "device const T* key_input [[buffer(1)]], "
            "device const T* index_query_key_input [[buffer(2)]], "
            "device const float* query_weight [[buffer(3)]], "
            "device const float* key_weight [[buffer(4)]], "
            "device const float* index_weight [[buffer(5)]], "
            "device const int* positions [[buffer(6)]], "
            "constant float* params [[buffer(7)]], "
            "device T* query_output [[buffer(8)]], "
            "device T* output_gate [[buffer(9)]], "
            "device T* key_output [[buffer(10)]], "
            "device T* index_query_output [[buffer(11)]], "
            "constant int* token_count [[buffer(12)]], "
            "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], "
            "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], "
            "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], "
            "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {\n";
        code += kQsaDecodePrologueSource;
        code += "}\n";
        return code;
    }

    QsaPrologueConfig config_;
    std::string kernel_name_;
};

const mlx::core::fast::CustomKernelFunction& gated_hc_post_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_post",
            {"branch", "residual", "injection"},
            {"output"},
            R"METAL(
                uint index = thread_position_in_grid.x;
                if (index >= uint(residual_shape[0])) return;
                uint row = index / uint(HIDDEN * HC_COUNT);
                uint stream = (index / uint(HIDDEN)) % uint(HC_COUNT);
                uint feature = index % uint(HIDDEN);
                // Preserve the multiply's promoted rounding before the add;
                // do not replace this with a fused multiply-add.
                U update = U(U(branch[row * uint(HIDDEN) + feature]) *
                    U(injection[row * uint(HC_COUNT) + stream]));
                output[index] = T(T(residual[index]) + T(update));
            )METAL",
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_projection_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_projection",
            {"normalized", "down_weight", "injection_weight"},
            {"down", "injection"},
            kGatedHcProjectionSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_up_collapse_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_up_collapse",
            {"down_projection", "up_weight", "normalized"},
            {"branch"},
            kGatedHcUpCollapseSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_mix_kernel() {
    static const auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_qwen_gated_hc_mix",
        {"logits", "normalized"},
        {"branch"},
        R"METAL(
            uint index = thread_position_in_grid.x;
            if (index >= uint(normalized_shape[0]) * uint(HIDDEN)) return;
            uint row = index / uint(HIDDEN), feature = index % uint(HIDDEN);
            float total = 0.0f;
#pragma clang loop unroll(full)
            for (uint group = 0u; group < uint(HC_COUNT); ++group) {
                uint location = (row * uint(HC_COUNT) + group) * uint(HIDDEN) + feature;
                P logit = logits[location];
                auto inverse = 1 / (1 + metal::precise::exp(metal::abs(logit)));
                P probability = logit < 0 ? inverse : 1 - inverse;
                T product = T(probability) * T(normalized[location]);
                total += float(product);
            }
            branch[index] = T(total / float(HC_COUNT));
        )METAL");
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_norm_down_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_norm_down",
            {
                "input",
                "norm_weight",
                "down_weight",
                "down_rows",
                "down_scales",
                "down_minima",
                "injection_weight",
                "injection_rows",
                "injection_scales",
                "injection_minima",
                "epsilon",
            },
            {"normalized", "parts"},
            kGatedHcNormDownSource,
            kGatedHcWeightHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
gated_hc_write_norm_down_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_write_norm_down",
            {
                "previous_residual",
                "previous_branch",
                "previous_injection",
                "norm_weight",
                "down_weight",
                "down_rows",
                "down_scales",
                "down_minima",
                "injection_weight",
                "injection_rows",
                "injection_scales",
                "injection_minima",
                "epsilon",
            },
            {"residual", "normalized", "parts"},
            kGatedHcWriteNormDownSource,
            kGatedHcWeightHeader,
            true,
            false,
            options);
    }();
    return kernel;
}


const mlx::core::fast::CustomKernelFunction& gated_hc_parts_up_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_parts_up",
            {"normalized", "parts", "up_weight", "up_rows", "up_scales", "up_minima"},
            {"branch", "injection"},
            kGatedHcPartsUpSource,
            kGatedHcWeightHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

bool gated_hc_fast_path_enabled() noexcept {
    static const bool enabled = [] {
        const char* setting =
            std::getenv("MFQ_METAL_QWEN_GATED_HC_FAST");
        return setting == nullptr || std::atoi(setting) != 0;
    }();
    return enabled;
}

bool gated_hc_float_dtype(mlx::core::Dtype dtype) noexcept {
    return dtype == mlx::core::float16 ||
        dtype == mlx::core::bfloat16 || dtype == mlx::core::float32;
}

bool can_fuse_grouped_rms_affine(
    const array& value,
    const array& weight,
    int group_size) {
    return gated_hc_fast_path_enabled() &&
        gated_hc_float_dtype(value.dtype()) &&
        gated_hc_float_dtype(weight.dtype()) &&
        value.ndim() >= 1 &&
        group_size > 0 && group_size <= 4096 &&
        value.shape(-1) % group_size == 0 &&
        weight.ndim() == 1 &&
        weight.shape(0) == value.shape(-1);
}

array fused_grouped_rms_affine(
    const array& value,
    const array& weight,
    int group_size,
    float eps) {
    constexpr int values_per_thread = 4;
    constexpr int simd_size = 32;
    const int threads_needed =
        (group_size + values_per_thread - 1) / values_per_thread;
    const int simdgroups =
        (threads_needed + simd_size - 1) / simd_size;
    const int threads = simdgroups * simd_size;
    const int rows = static_cast<int>(value.size() / group_size);
    const int group_count = value.shape(-1) / group_size;
    const array epsilon({eps}, Shape{1});
    auto outputs = grouped_rms_affine_kernel()(
        {value, weight, epsilon},
        {value.shape()},
        {value.dtype()},
        {rows * threads, 1, 1},
        {threads, 1, 1},
        {
            {"T", value.dtype()},
            {"GROUP_SIZE", group_size},
            {"GROUP_COUNT", group_count},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

struct GatedHcWriteNorm {
    array residual;
    array normalized;
};

bool can_fuse_grouped_rms_affine_write(
    const array& branch,
    const array& residual,
    const array& injection,
    const array& weight,
    int group_size,
    int hc_count) {
    if (!gated_hc_fast_path_enabled() ||
        !gated_hc_float_dtype(branch.dtype()) ||
        !gated_hc_float_dtype(residual.dtype()) ||
        !gated_hc_float_dtype(injection.dtype()) ||
        !gated_hc_float_dtype(weight.dtype()) ||
        branch.ndim() < 2 || residual.ndim() != branch.ndim() ||
        injection.ndim() != branch.ndim() ||
        group_size <= 0 || group_size > 4096 || hc_count <= 1 ||
        residual.shape(-1) != group_size * hc_count ||
        weight.shape() != Shape{group_size * hc_count}) {
        return false;
    }
    auto branch_shape = residual.shape();
    branch_shape.back() = group_size;
    auto injection_shape = residual.shape();
    injection_shape.back() = hc_count;
    return branch.shape() == branch_shape &&
        injection.shape() == injection_shape &&
        residual.size() > 0 &&
        residual.size() <= static_cast<std::size_t>(
            std::numeric_limits<int>::max());
}

GatedHcWriteNorm fused_grouped_rms_affine_write(
    const array& branch,
    const array& residual,
    const array& injection,
    const array& weight,
    int group_size,
    int hc_count,
    float eps) {
    constexpr int values_per_thread = 4;
    constexpr int simd_size = 32;
    const int threads_needed =
        (group_size + values_per_thread - 1) / values_per_thread;
    const int simdgroups =
        (threads_needed + simd_size - 1) / simd_size;
    const int threads = simdgroups * simd_size;
    const int rows = static_cast<int>(residual.size() / group_size);
    const auto update_dtype = mlx::core::promote_types(
        branch.dtype(), injection.dtype());
    const auto output_dtype = mlx::core::promote_types(
        residual.dtype(), update_dtype);
    const array epsilon({eps}, Shape{1});
    auto outputs = grouped_rms_affine_write_kernel()(
        {residual, branch, injection, weight, epsilon},
        {residual.shape(), residual.shape()},
        {output_dtype, output_dtype},
        {rows * threads, 1, 1},
        {threads, 1, 1},
        {
            {"R", residual.dtype()},
            {"B", branch.dtype()},
            {"G", injection.dtype()},
            {"W", weight.dtype()},
            {"U", update_dtype},
            {"O", output_dtype},
            {"GROUP_SIZE", group_size},
            {"HC_COUNT", hc_count},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs[0]),
        std::move(outputs[1]),
    };
}

bool can_fuse_gated_hc_projections(
    const array& normalized,
    const array& down_weight,
    const array& injection_weight,
    int hc_count) {
    if (!gated_hc_fast_path_enabled() ||
        !gated_hc_float_dtype(normalized.dtype()) ||
        normalized.ndim() != 3 ||
        normalized.shape(0) != 1 ||
        normalized.shape(1) != 1 ||
        down_weight.ndim() != 2 ||
        injection_weight.ndim() != 2 ||
        !gated_hc_float_dtype(down_weight.dtype()) ||
        !gated_hc_float_dtype(injection_weight.dtype()) ||
        hc_count != 4) {
        return false;
    }
    const int width = normalized.shape(2);
    const int low_rank = down_weight.shape(0);
    return width > 0 && width % 1024 == 0 &&
        low_rank > 0 && low_rank % 4 == 0 &&
        down_weight.shape(1) == width &&
        injection_weight.shape() == Shape{hc_count, width} &&
        width >= 16 * low_rank;
}

struct GatedHcProjections {
    array down;
    array injection;
};

GatedHcProjections fused_gated_hc_projections(
    const array& normalized,
    const array& down_weight,
    const array& injection_weight,
    int hc_count) {
    const int width = normalized.shape(2);
    const int low_rank = down_weight.shape(0);
    const int workgroups = low_rank / 4 + hc_count / 4;
    // Keep storage types independent and match matmul's promotion without
    // materializing full FP32 copies of the BF16/FP16 projection matrices.
    const auto down_dtype = mlx::core::promote_types(
        normalized.dtype(), down_weight.dtype());
    const auto injection_dtype = mlx::core::promote_types(
        normalized.dtype(), injection_weight.dtype());
    auto outputs = gated_hc_projection_kernel()(
        {normalized, down_weight, injection_weight},
        {Shape{1, 1, low_rank}, Shape{1, 1, hc_count}},
        {down_dtype, injection_dtype},
        {workgroups * kGatedHcProjectionThreads, 1, 1},
        {kGatedHcProjectionThreads, 1, 1},
        {
            {"D", down_dtype},
            {"I", injection_dtype},
            {"WIDTH", width},
            {"LOW_RANK", low_rank},
            {"HC_COUNT", hc_count},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs[0]),
        std::move(outputs[1]),
    };
}

bool can_fuse_gated_hc_up_collapse(
    const array& down_projection,
    const array& up_weight,
    const array& normalized,
    int hidden_size,
    int hc_count) {
    return gated_hc_fast_path_enabled() &&
        gated_hc_float_dtype(normalized.dtype()) &&
        gated_hc_float_dtype(down_projection.dtype()) &&
        gated_hc_float_dtype(up_weight.dtype()) &&
        normalized.ndim() == 3 &&
        normalized.shape(0) == 1 &&
        normalized.shape(1) == 1 &&
        normalized.shape(2) == hidden_size * hc_count &&
        down_projection.ndim() == 3 &&
        down_projection.shape(0) == 1 &&
        down_projection.shape(1) == 1 &&
        down_projection.shape(2) == 320 &&
        up_weight.ndim() == 2 &&
        up_weight.shape(0) == hidden_size * hc_count &&
        up_weight.shape(1) == down_projection.shape(2) &&
        hc_count == 4;
}

array fused_gated_hc_up_collapse(
    const array& down_projection,
    const array& up_weight,
    const array& normalized,
    int hidden_size,
    int hc_count) {
    const int workgroups = (hidden_size + 7) / 8;
    const auto dtype = mlx::core::promote_types(
        down_projection.dtype(), up_weight.dtype());
    auto outputs = gated_hc_up_collapse_kernel()(
        {down_projection, up_weight, normalized},
        {Shape{1, 1, hidden_size}},
        {dtype},
        {workgroups * kGatedHcUpThreads, 1, 1},
        {kGatedHcUpThreads, 1, 1},
        {
            {"T", dtype},
            {"L", down_projection.dtype()},
            {"HIDDEN", hidden_size},
            {"LOW_RANK", down_projection.shape(2)},
            {"HC_COUNT", hc_count},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

bool can_fuse_gated_hc_two_stage(
    const array& input,
    const array& norm_weight,
    const MlxLinear& down_weight,
    const MlxLinear& up_weight,
    const std::optional<MlxLinear>& injection_weight,
    int hidden_size,
    int hc_count) {
    if (down_weight.tq_weight_ref() || up_weight.tq_weight_ref()
        || (injection_weight && injection_weight->tq_weight_ref())) return false;
    if (!gated_hc_fast_path_enabled() ||
        hidden_size != 2560 || hc_count != 4 ||
        input.ndim() != 3 || input.shape(0) != 1 ||
        input.shape(1) < 1 || input.shape(1) > 16 || input.shape(2) != 10240 ||
        norm_weight.shape() != Shape{10240} ||
        (down_weight.output_size() != 320 || down_weight.input_size() != 10240) ||
        (up_weight.output_size() != 10240 || up_weight.input_size() != 320) ||
        !gated_hc_float_dtype(input.dtype()) ||
        !gated_hc_float_dtype(norm_weight.dtype()) ||
        !gated_hc_float_dtype(hc_weight_dtype(down_weight)) ||
        !gated_hc_float_dtype(hc_weight_dtype(up_weight))) {
        return false;
    }
    if (input.shape(1) > 1 &&
        !can_use_packed_hc(input, norm_weight, down_weight, up_weight, injection_weight))
        return false;
    return !injection_weight ||
        (injection_weight->output_size() == 4 && injection_weight->input_size() == 10240 &&
         gated_hc_float_dtype(hc_weight_dtype(*injection_weight)));
}

struct GatedHcNormDown {
    array normalized;
    array parts;
};


GatedHcNormDown fused_gated_hc_norm_down(
    const array& input,
    const array& norm_weight,
    const MlxLinear& down_weight,
    const std::optional<MlxLinear>& injection_weight,
    float eps) {
    constexpr int hidden_size = 2560;
    constexpr int hc_count = 4;
    constexpr int low_rank = 320;
    constexpr int rows_per_group = 16;
    const bool has_injection = injection_weight.has_value();
    const int groups = low_rank / rows_per_group +
        static_cast<int>(has_injection);
    const array epsilon({eps}, Shape{1});
    const auto down = hc_weight_inputs(down_weight);
    const auto injection = hc_weight_inputs(has_injection ? *injection_weight : down_weight);
    auto outputs = gated_hc_norm_down_kernel()(
        {input, norm_weight, down.values, down.rows, down.scales, down.minima,
         injection.values, injection.rows, injection.scales, injection.minima, epsilon},
        {input.shape(), Shape{1, hc_count, low_rank + hc_count}},
        {input.dtype(), mlx::core::float32},
        {groups * hc_count * kGatedHcProjectionThreads, 1, 1},
        {kGatedHcProjectionThreads, 1, 1},
        {
            {"N", input.dtype()},
            {"DOWN_GS", down.group_size},
            {"DOWN_NG", down.groups},
            {"INJECT_GS", injection.group_size},
            {"INJECT_NG", injection.groups},
            {"HIDDEN", hidden_size},
            {"LOW_RANK", low_rank},
            {"HC_COUNT", hc_count},
            {"HAS_INJECTION", static_cast<int>(has_injection)},
        },
        std::nullopt,
        false,
        {});
    return {std::move(outputs[0]), std::move(outputs[1])};
}

struct GatedHcWriteNormDown {
    array residual;
    array normalized;
    array parts;
};

GatedHcWriteNormDown fused_gated_hc_write_norm_down(
    const array& previous_branch,
    const array& previous_residual,
    const array& previous_injection,
    const array& norm_weight,
    const MlxLinear& down_weight,
    const std::optional<MlxLinear>& injection_weight,
    float eps) {
    constexpr int hidden_size = 2560;
    constexpr int hc_count = 4;
    constexpr int low_rank = 320;
    constexpr int rows_per_group = 16;
    const bool has_injection = injection_weight.has_value();
    const int groups = low_rank / rows_per_group +
        static_cast<int>(has_injection);
    const auto update_dtype = mlx::core::promote_types(
        previous_branch.dtype(), previous_injection.dtype());
    const auto output_dtype = mlx::core::promote_types(
        previous_residual.dtype(), update_dtype);
    const array epsilon({eps}, Shape{1});
    const auto down = hc_weight_inputs(down_weight);
    const auto injection = hc_weight_inputs(has_injection ? *injection_weight : down_weight);
    auto outputs = gated_hc_write_norm_down_kernel()(
        {
            previous_residual,
            previous_branch,
            previous_injection,
            norm_weight,
            down.values, down.rows, down.scales, down.minima,
            injection.values, injection.rows, injection.scales, injection.minima,
            epsilon,
        },
        {
            previous_residual.shape(),
            previous_residual.shape(),
            Shape{1, hc_count, low_rank + hc_count},
        },
        {output_dtype, output_dtype, mlx::core::float32},
        {groups * hc_count * kGatedHcProjectionThreads, 1, 1},
        {kGatedHcProjectionThreads, 1, 1},
        {
            {"G", previous_injection.dtype()},
            {"U", update_dtype},
            {"O", output_dtype},
            {"DOWN_GS", down.group_size},
            {"DOWN_NG", down.groups},
            {"INJECT_GS", injection.group_size},
            {"INJECT_NG", injection.groups},
            {"HIDDEN", hidden_size},
            {"LOW_RANK", low_rank},
            {"HC_COUNT", hc_count},
            {"HAS_INJECTION", static_cast<int>(has_injection)},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs[0]),
        std::move(outputs[1]),
        std::move(outputs[2]),
    };
}

struct GatedHcPartsUp {
    array branch;
    array injection;
};

GatedHcPartsUp fused_gated_hc_parts_up(
    const array& normalized,
    const array& parts,
    const MlxLinear& down_weight,
    const MlxLinear& up_weight,
    const std::optional<MlxLinear>& injection_weight) {
    constexpr int hidden_size = 2560;
    constexpr int hc_count = 4;
    constexpr int low_rank = 320;
    const bool has_injection = injection_weight.has_value();
    const auto down_dtype = mlx::core::promote_types(
        normalized.dtype(), hc_weight_dtype(down_weight));
    const auto branch_dtype = mlx::core::promote_types(
        down_dtype, hc_weight_dtype(up_weight));
    const auto injection_dtype = has_injection
        ? mlx::core::promote_types(
              normalized.dtype(), hc_weight_dtype(*injection_weight))
        : normalized.dtype();
    const auto up = hc_weight_inputs(up_weight);
    const int workgroups = hidden_size / 8;
    const int chunk = up.group_size == 0 ? 32
        : up.group_size * ((32 + up.group_size - 1) / up.group_size);
    const int threads = 32 * ((low_rank + chunk - 1) / chunk);
    auto outputs = gated_hc_parts_up_kernel()(
        {normalized, parts, up.values, up.rows, up.scales, up.minima},
        {Shape{1, 1, hidden_size}, Shape{1, 1, hc_count}},
        {branch_dtype, injection_dtype},
        {workgroups * threads, 1, 1},
        {threads, 1, 1},
        {
            {"T", branch_dtype},
            {"L", down_dtype},
            {"I", injection_dtype},
            {"UP_GS", up.group_size},
            {"UP_NG", up.groups},
            {"HIDDEN", hidden_size},
            {"LOW_RANK", low_rank},
            {"HC_COUNT", hc_count},
            {"HAS_INJECTION", static_cast<int>(has_injection)},
        },
        std::nullopt,
        false,
        {});
    return {std::move(outputs[0]), std::move(outputs[1])};
}

MlxQwen4GatedResidualPre fused_gated_hc_two_stage(
    const array& input,
    const array& norm_weight,
    const MlxLinear& down_weight,
    const MlxLinear& up_weight,
    const std::optional<MlxLinear>& injection_weight,
    float eps) {
    if (can_use_packed_hc(input, norm_weight, down_weight, up_weight, injection_weight))
        return packed_gated_hc(input, std::nullopt, std::nullopt, norm_weight,
            down_weight, up_weight, injection_weight, eps);
    auto first = fused_gated_hc_norm_down(
        input, norm_weight, down_weight, injection_weight, eps);
    auto second = fused_gated_hc_parts_up(
        first.normalized,
        first.parts,
        down_weight,
        up_weight,
        injection_weight);
    std::optional<array> injection;
    if (injection_weight) injection = std::move(second.injection);
    return {
        std::move(second.branch),
        input,
        std::move(injection),
    };
}

MlxQwen4GatedResidualPre fused_gated_hc_two_stage_after(
    const array& previous_branch,
    const array& previous_residual,
    const array& previous_injection,
    const array& norm_weight,
    const MlxLinear& down_weight,
    const MlxLinear& up_weight,
    const std::optional<MlxLinear>& injection_weight,
    float eps) {
    if (can_use_packed_hc(previous_residual, norm_weight, down_weight, up_weight, injection_weight)
        && previous_branch.flags().row_contiguous && previous_injection.flags().row_contiguous)
        return packed_gated_hc(previous_residual, previous_branch, previous_injection, norm_weight,
            down_weight, up_weight, injection_weight, eps);
    auto first = fused_gated_hc_write_norm_down(
        previous_branch,
        previous_residual,
        previous_injection,
        norm_weight,
        down_weight,
        injection_weight,
        eps);
    auto second = fused_gated_hc_parts_up(
        first.normalized,
        first.parts,
        down_weight,
        up_weight,
        injection_weight);
    std::optional<array> injection;
    if (injection_weight) injection = std::move(second.injection);
    return {
        std::move(second.branch),
        std::move(first.residual),
        std::move(injection),
    };
}

const mlx::core::fast::CustomKernelFunction& gated_hc_activation_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel("mfq_gated_hc_activation",
            {"down", "inject"}, {"low", "injection"}, R"METAL(
uint index = thread_position_in_grid.x;
if (index < uint(down_shape[0])) {
    T scaled = down[index] / T(GROUPS);
    low[index] = scaled * gr_sigmoid(scaled);
}
if (index < uint(inject_shape[0])) {
    T scaled = inject[index] / T(GROUPS);
    injection[index] = T(2.f) * gr_sigmoid(scaled);
}
)METAL", R"METAL(
template <typename T> inline T gr_sigmoid(T value) {
    auto tail = 1 / (1 + metal::precise::exp(metal::abs(value)));
    return value < 0 ? tail : 1 - tail;
}
)METAL", true, false, options);
    }();
    return kernel;
}

MlxQwen4GatedResidualPre gated_hc_from_normalized(
    array normalized,
    array residual,
    const MlxLinear& down_weight,
    const MlxLinear& up_weight,
    const std::optional<MlxLinear>& inject_weight,
    int hidden_size,
    int hc_count) {
    const int width = hidden_size * hc_count;
    const auto* dense_down = down_weight.dense_weight_ref();
    const auto* dense_up = up_weight.dense_weight_ref();
    const auto* dense_inject = inject_weight ? inject_weight->dense_weight_ref() : nullptr;
    std::optional<array> fused_injection;
    std::optional<array> activated_low;
    array down_projection = [&] {
        if (const auto* down = down_weight.tq_weight_ref(); down && inject_weight) {
            if (const auto* injection = inject_weight->tq_weight_ref()) {
                auto projections = MlxTqWeight::matmul_many({*down, *injection}, normalized);
                if (normalized.size() / static_cast<std::size_t>(width) <= 16) {
                    auto activations = gated_hc_activation_kernel()(
                        {mlx::core::reshape(projections[0], {-1}), mlx::core::reshape(projections[1], {-1})},
                        {projections[0].shape(), projections[1].shape()}, {normalized.dtype(), normalized.dtype()},
                        {static_cast<int>(std::max(projections[0].size(), projections[1].size())), 1, 1}, {128, 1, 1},
                        {{"T", normalized.dtype()}, {"GROUPS", hc_count}}, std::nullopt, false, {});
                    activated_low = std::move(activations[0]);
                    fused_injection = std::move(activations[1]);
                } else {
                    fused_injection = array(2.0f, normalized.dtype()) * mlx::core::sigmoid(
                        projections[1] / array(float(hc_count), normalized.dtype()));
                }
                return std::move(projections[0]);
            }
        }
        if (dense_down && dense_inject && can_fuse_gated_hc_projections(
                normalized,
                *dense_down,
                *dense_inject,
                hc_count)) {
            auto projections = fused_gated_hc_projections(
                normalized,
                *dense_down,
                *dense_inject,
                hc_count);
            fused_injection = std::move(projections.injection);
            return std::move(projections.down);
        }
        return hc_project(down_weight, normalized);
    }();
    const array connection_count(
        static_cast<float>(hc_count), normalized.dtype());
    array mixed = [&] {
        if (dense_up && can_fuse_gated_hc_up_collapse(
                down_projection,
                *dense_up,
                normalized,
                hidden_size,
                hc_count)) {
            return fused_gated_hc_up_collapse(
                down_projection,
                *dense_up,
                normalized,
                hidden_size,
                hc_count);
        }
        auto low = activated_low ? *activated_low : down_projection / connection_count;
        if (!activated_low) low = low * mlx::core::sigmoid(low);
        const auto rows = normalized.size() / static_cast<std::size_t>(width);
        if (const auto* tq = up_weight.tq_weight_ref(); tq && tq->can_fuse_gated_mix(low)
            && low.dtype() == normalized.dtype())
            return tq->gated_mix(low, normalized, hc_count);
        auto logits = hc_project(up_weight, low);
        const auto elements = rows * static_cast<std::size_t>(hidden_size);
        if (gated_hc_fast_path_enabled() && rows >= 64 &&
            gated_hc_float_dtype(logits.dtype()) && gated_hc_float_dtype(normalized.dtype()) &&
            elements <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            auto shape = residual.shape();
            shape.back() = hidden_size;
            const auto dtype = mlx::core::promote_types(logits.dtype(), normalized.dtype());
            return gated_hc_mix_kernel()(
                {logits, mlx::core::reshape(normalized,
                    Shape{static_cast<int>(rows), width})}, {shape}, {dtype},
                {static_cast<int>(elements), 1, 1}, {256, 1, 1},
                {{"P", logits.dtype()}, {"T", dtype},
                 {"HIDDEN", hidden_size}, {"HC_COUNT", hc_count}},
                std::nullopt, false, {}).front();
        }
        auto mixing = mlx::core::sigmoid(logits);
        auto stream_shape = residual.shape();
        stream_shape.back() = hc_count;
        stream_shape.push_back(hidden_size);
        return mlx::core::mean(
            mlx::core::reshape(mixing, stream_shape) *
                mlx::core::reshape(normalized, stream_shape),
            -2);
    }();
    std::optional<array> injection;
    if (inject_weight) {
        if (inject_weight->output_size() != hc_count ||
            inject_weight->input_size() != width) {
            throw std::invalid_argument("Qwen4 residual injection mismatch");
        }
        if (fused_injection) {
            injection = std::move(*fused_injection);
        } else {
            auto projected = hc_project(*inject_weight, normalized);
            injection = array(2.0f, normalized.dtype()) *
                mlx::core::sigmoid(projected / connection_count);
        }
    }
    return {
        std::move(mixed),
        std::move(residual),
        std::move(injection),
    };
}

array floating(const array& value, mlx::core::Dtype dtype) {
    auto result = value.dtype() == dtype ? value : mlx::core::astype(value, dtype);
    return mlx::core::contiguous(result);
}

void require_rank(const array& value, int rank, const char* name) {
    if (value.ndim() != rank) {
        throw std::invalid_argument(std::string(name) + " has an invalid rank");
    }
}

array qsa_head_scores(const array& query, const array& pooled_keys) {
    require_rank(query, 4, "Qwen4 index query");
    require_rank(pooled_keys, 3, "Qwen4 pooled key");
    if (query.shape(0) != pooled_keys.shape(0) ||
        query.shape(3) != pooled_keys.shape(2)) {
        throw std::invalid_argument("Qwen4 QSA score dimensions disagree");
    }
    auto keys = mlx::core::transpose(
        mlx::core::astype(pooled_keys, mlx::core::float32), {0, 2, 1});
    return mlx::core::reshape(
        mlx::core::matmul(
            mlx::core::reshape(mlx::core::astype(query, mlx::core::float32),
                Shape{query.shape(0), query.shape(1) * query.shape(2), query.shape(3)}), keys),
        Shape{query.shape(0), query.shape(1), query.shape(2), pooled_keys.shape(1)});
}

} // namespace

array qwen4_grouped_rms_norm(
    const array& value,
    const array& weight,
    int group_size,
    float eps) {
    if (value.ndim() == 0 || group_size <= 0 ||
        value.shape(-1) % group_size != 0 ||
        weight.ndim() != 1 || weight.shape(0) != value.shape(-1) ||
        !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument("Qwen4 grouped RMSNorm dimensions disagree");
    }
    if (can_fuse_grouped_rms_affine(value, weight, group_size)) {
        return fused_grouped_rms_affine(
            value, weight, group_size, eps);
    }
    const auto source_dtype = value.dtype();
    auto shape = value.shape();
    const int width = shape.back();
    shape.pop_back();
    shape.push_back(width / group_size);
    shape.push_back(group_size);
    auto source = mlx::core::reshape(
        mlx::core::astype(value, mlx::core::float32), shape);
    // Treat every hyper-connection stream as an independent RMSNorm row.
    // MLX's fast primitive keeps the reduction and normalization in one
    // dispatch; the per-stream centred weights remain a separate affine so
    // all groups can retain their own parameters.
    auto normalized = mlx::core::fast::rms_norm(
        source, std::nullopt, eps);
    normalized = mlx::core::reshape(normalized, value.shape()) *
        (array(1.0f) + mlx::core::astype(weight, mlx::core::float32));
    return normalized.dtype() == source_dtype
        ? normalized : mlx::core::astype(normalized, source_dtype);
}

MlxQwen4GatedResidualPre qwen4_gated_residual_pre(
    const array& hyper_input,
    const array& norm_weight,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps) {
    std::optional<MlxLinear> injection;
    if (inject_weight) injection.emplace(*inject_weight);
    return qwen4_gated_residual_pre(
        hyper_input, norm_weight, MlxLinear(down_weight), MlxLinear(up_weight),
        injection, hidden_size, hc_count, eps);
}

MlxQwen4GatedResidualPre qwen4_gated_residual_pre(
    const array& hyper_input,
    const array& norm_weight,
    const MlxLinear& down_weight,
    const MlxLinear& up_weight,
    const std::optional<MlxLinear>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps) {
    const int width = hidden_size * hc_count;
    if (hidden_size <= 0 || hc_count <= 1 || hyper_input.ndim() < 2 ||
        hyper_input.shape(-1) != width ||
        down_weight.input_size() != width || up_weight.output_size() != width ||
        up_weight.input_size() != down_weight.output_size()) {
        throw std::invalid_argument("Qwen4 gated-residual projection mismatch");
    }
    if (std::isfinite(eps) && eps > 0.0f &&
        can_fuse_gated_hc_two_stage(
            hyper_input,
            norm_weight,
            down_weight,
            up_weight,
            inject_weight,
            hidden_size,
            hc_count)) {
        return fused_gated_hc_two_stage(
            hyper_input,
            norm_weight,
            down_weight,
            up_weight,
            inject_weight,
            eps);
    }
    auto normalized = qwen4_grouped_rms_norm(
        hyper_input, norm_weight, hidden_size, eps);
    return gated_hc_from_normalized(
        std::move(normalized),
        hyper_input,
        down_weight,
        up_weight,
        inject_weight,
        hidden_size,
        hc_count);
}

MlxQwen4GatedResidualPre qwen4_gated_residual_pre_after(
    const array& previous_branch,
    const array& previous_residual,
    const array& previous_injection,
    const array& norm_weight,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps) {
    std::optional<MlxLinear> injection;
    if (inject_weight) injection.emplace(*inject_weight);
    return qwen4_gated_residual_pre_after(
        previous_branch, previous_residual, previous_injection, norm_weight,
        MlxLinear(down_weight), MlxLinear(up_weight), injection, hidden_size, hc_count, eps);
}

MlxQwen4GatedResidualPre qwen4_gated_residual_pre_after(
    const array& previous_branch,
    const array& previous_residual,
    const array& previous_injection,
    const array& norm_weight,
    const MlxLinear& down_weight,
    const MlxLinear& up_weight,
    const std::optional<MlxLinear>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps) {
    const int width = hidden_size * hc_count;
    if (hidden_size <= 0 || hc_count <= 1 ||
        previous_residual.ndim() < 2 ||
        previous_residual.shape(-1) != width ||
        norm_weight.shape() != Shape{width} ||
        down_weight.input_size() != width || up_weight.output_size() != width ||
        up_weight.input_size() != down_weight.output_size() ||
        !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument(
            "Qwen4 chained gated-residual projection mismatch");
    }

    if (can_fuse_gated_hc_two_stage(
            previous_residual,
            norm_weight,
            down_weight,
            up_weight,
            inject_weight,
            hidden_size,
            hc_count) &&
        previous_branch.shape() == Shape{1, previous_residual.shape(1), hidden_size} &&
        previous_injection.shape() == Shape{1, previous_residual.shape(1), hc_count} &&
        (previous_residual.shape(1) == 1 ||
         (previous_branch.flags().row_contiguous && previous_injection.flags().row_contiguous)) &&
        gated_hc_float_dtype(previous_branch.dtype()) &&
        gated_hc_float_dtype(previous_injection.dtype())) {
        return fused_gated_hc_two_stage_after(
            previous_branch,
            previous_residual,
            previous_injection,
            norm_weight,
            down_weight,
            up_weight,
            inject_weight,
            eps);
    }

    array updated = previous_residual;
    array normalized = previous_residual;
    if (can_fuse_grouped_rms_affine_write(
            previous_branch,
            previous_residual,
            previous_injection,
            norm_weight,
            hidden_size,
            hc_count)) {
        auto result = fused_grouped_rms_affine_write(
            previous_branch,
            previous_residual,
            previous_injection,
            norm_weight,
            hidden_size,
            hc_count,
            eps);
        updated = std::move(result.residual);
        normalized = std::move(result.normalized);
    } else {
        updated = qwen4_gated_residual_post(
            previous_branch,
            previous_residual,
            previous_injection,
            hc_count);
        normalized = qwen4_grouped_rms_norm(
            updated,
            norm_weight,
            hidden_size,
            eps);
    }
    return gated_hc_from_normalized(
        std::move(normalized),
        std::move(updated),
        down_weight,
        up_weight,
        inject_weight,
        hidden_size,
        hc_count);
}

array qwen4_gated_residual_post(
    const array& branch,
    const array& residual,
    const array& injection,
    int hc_count) {
    if (branch.ndim() < 2 || hc_count <= 1 ||
        residual.shape(-1) != hc_count * branch.shape(-1)) {
        throw std::invalid_argument("Qwen4 gated-residual post mismatch");
    }
    auto expected = branch.shape();
    expected.back() = hc_count;
    if (injection.shape() != expected) {
        throw std::invalid_argument("Qwen4 gated-residual gate mismatch");
    }
    if (gated_hc_fast_path_enabled() &&
        gated_hc_float_dtype(branch.dtype()) &&
        gated_hc_float_dtype(residual.dtype()) &&
        gated_hc_float_dtype(injection.dtype()) &&
        residual.size() == branch.size() * static_cast<std::size_t>(hc_count) &&
        residual.size() > 0 &&
        residual.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        const auto update_dtype = mlx::core::promote_types(
            branch.dtype(), injection.dtype());
        const auto dtype = mlx::core::promote_types(residual.dtype(), update_dtype);
        auto outputs = gated_hc_post_kernel()(
            {branch, mlx::core::reshape(residual,
                Shape{static_cast<int>(residual.size())}), injection},
            {residual.shape()},
            {dtype},
            {static_cast<int>(residual.size()), 1, 1},
            {256, 1, 1},
            {
                {"T", dtype},
                {"U", update_dtype},
                {"HIDDEN", branch.shape(-1)},
                {"HC_COUNT", hc_count},
            },
            std::nullopt,
            false,
            {});
        return std::move(outputs.front());
    }
    auto update = mlx::core::expand_dims(branch, -2) *
        mlx::core::expand_dims(injection, -1);
    return residual + mlx::core::reshape(update, residual.shape());
}

MlxQwen4QsaDecodePrologue qwen4_qsa_decode_prologue(
    const array& query_gate,
    const array& key,
    const array& index_query_key,
    const array& query_norm_weight,
    const array& key_norm_weight,
    const array& index_query_norm_weight,
    const array& positions,
    int query_heads,
    int key_heads,
    int index_heads,
    int head_dimension,
    int index_dimension,
    int rotary_dimension,
    float rope_theta,
    float eps) {
    const auto dtype = query_gate.dtype();
    const int rows = query_gate.ndim() == 3 ? query_gate.shape(1) : 0;
    if ((dtype != mlx::core::float16 &&
         dtype != mlx::core::bfloat16 &&
         dtype != mlx::core::float32) ||
        key.dtype() != dtype || index_query_key.dtype() != dtype ||
        query_heads <= 0 || key_heads <= 0 || index_heads <= 0 ||
        head_dimension <= 0 || index_dimension <= 0 ||
        head_dimension > 256 || index_dimension > 256 ||
        rotary_dimension <= 0 || rotary_dimension % 2 != 0 ||
        rotary_dimension > std::min(head_dimension, index_dimension) ||
        !std::isfinite(rope_theta) || rope_theta <= 0.0f ||
        !std::isfinite(eps) || eps <= 0.0f ||
        rows < 1 || rows > 6 ||
        query_gate.shape() != Shape{1, rows, query_heads * 2 * head_dimension} ||
        key.shape() != Shape{1, rows, key_heads * head_dimension} ||
        index_query_key.shape() !=
            Shape{1, rows, (index_heads + 1) * index_dimension} ||
        query_norm_weight.dtype() != mlx::core::float32 ||
        query_norm_weight.shape() != Shape{head_dimension} ||
        key_norm_weight.dtype() != mlx::core::float32 ||
        key_norm_weight.shape() != Shape{head_dimension} ||
        index_query_norm_weight.dtype() != mlx::core::float32 ||
        index_query_norm_weight.shape() != Shape{index_dimension} ||
        positions.dtype() != mlx::core::int32 ||
        positions.shape() != Shape{rows}) {
        throw std::invalid_argument(
            "Qwen4 QSA decode prologue geometry disagrees");
    }
    const QsaPrologueConfig config{
        dtype,
        {query_heads, key_heads, index_heads,
         head_dimension, index_dimension, rotary_dimension},
        {eps, rope_theta}};
    auto outputs = array::make_arrays(
        {
            Shape{1, query_heads, rows, head_dimension},
            Shape{1, rows, query_heads * head_dimension},
            Shape{1, key_heads, rows, head_dimension},
            Shape{1, rows, index_heads, index_dimension},
        },
        {dtype, dtype, dtype, dtype},
        std::make_shared<QsaProloguePrimitive>(
            mlx::core::default_stream(mlx::core::default_device()), config),
        {
            mlx::core::contiguous(query_gate),
            mlx::core::contiguous(key),
            mlx::core::contiguous(index_query_key),
            query_norm_weight,
            key_norm_weight,
            index_query_norm_weight,
            mlx::core::contiguous(positions),
        });
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
        std::move(outputs.at(2)),
        std::move(outputs.at(3)),
    };
}

array qwen4_qsa_prefill_output_gate(const array& attended, const array& gate) {
    if (attended.ndim() != 4 || attended.shape() != gate.shape() ||
        attended.dtype() != gate.dtype() ||
        (attended.dtype() != mlx::core::float16 &&
         attended.dtype() != mlx::core::bfloat16 && attended.dtype() != mlx::core::float32) ||
        attended.size() == 0 || attended.size() > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<std::int64_t>(attended.shape(2)) * attended.shape(3) >
            std::numeric_limits<int>::max()) {
        throw std::invalid_argument("QSA prefill output gate geometry disagrees");
    }
    return array(Shape{attended.shape(0), attended.shape(1),
        attended.shape(2) * attended.shape(3)}, attended.dtype(),
        std::make_shared<QsaPrefillOutputGatePrimitive>(
            mlx::core::default_stream(mlx::core::default_device()), attended.dtype(),
            attended.shape(2), attended.shape(3)), {attended, gate});
}

array qwen4_qsa_block_scores(
    const array& query,
    const array& pooled_keys) {
    auto scores = qsa_head_scores(query, pooled_keys);
    scores = mlx::core::maximum(scores, array(0.0f));
    return mlx::core::sum(scores, -2) /
        std::sqrt(static_cast<float>(query.shape(3)));
}

array qwen4_qsa_select_blocks(
    const array& query,
    const array& pooled_keys,
    int query_offset,
    int block_size) {
    return mlx_sparse_indexer_topk512(
        qsa_head_scores(query, pooled_keys), query_offset, block_size);
}

array qwen4_dense_gqa_attention(
    const array& query,
    const array& key,
    const array& value,
    int query_offset) {
    require_rank(query, 4, "Qwen4 attention query");
    require_rank(key, 4, "Qwen4 attention key");
    if (value.shape() != key.shape() || query.shape(0) != key.shape(0) ||
        query.shape(1) % key.shape(1) != 0 ||
        query.shape(3) != key.shape(3) || query_offset < 0 ||
        query_offset + query.shape(2) > key.shape(2)) {
        throw std::invalid_argument("Qwen4 dense GQA dimensions disagree");
    }
    const int tokens = query.shape(2);
    const int keys = key.shape(2);
    if (query_offset + tokens == keys) {
        auto output = scaled_dot_product_attention(
            query,
            key,
            value,
            tokens > 1,
            1.0f / std::sqrt(static_cast<float>(query.shape(3))));
        return mlx::core::transpose(output, {0, 2, 1, 3});
    }
    auto key_positions = mlx::core::reshape(
        mlx::core::arange(0, keys, 1, mlx::core::int32), Shape{1, keys});
    auto query_positions = mlx::core::reshape(
        mlx::core::arange(
            query_offset, query_offset + tokens, 1, mlx::core::int32),
        Shape{tokens, 1});
    auto mask = mlx::core::expand_dims(
        mlx::core::expand_dims(key_positions <= query_positions, 0), 0);
    auto output = scaled_dot_product_attention(
        query, key, value, false,
        1.0f / std::sqrt(static_cast<float>(query.shape(3))), mask);
    return mlx::core::transpose(output, {0, 2, 1, 3});
}

} // namespace mfq::metal
