// Canonical CUDA execution for NINT.
//
// Every ordinary NINT tensor is normalized at the loader boundary to one
// row-major bitstream plus per-neuron q metadata.  Uniform presets and
// adaptive q/k allocations therefore execute through exactly the same
// kernels.  Small batched decode can stage decoded tiles in shared memory
// for Tensor Core projection; large prefill retains its GEMM dispatch.

// NINT8-0 is a separate symmetric format.  It retains one small-M packed
// kernel; larger batches use its row decoder followed by GEMM.


#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "async_copy.cuh"
#include "glu.cuh"
#include "mfq_tensor_backend.h"
#include "mfq_cuda_nint_route_hint.h"
#include "packed_backward.cuh"
#include "packed_nint.cuh"
#include "nint_route_group.cuh"
#include "nint_short_decode.h"
#include "../../../cpp_runtime/backends/cuda/kernels/kernel_dispatch_trace.h"


#define MFQ_CUBLAS_CHECK(expression) \
    MFQ_RUNTIME_CHECK( \
        (expression) == CUBLAS_STATUS_SUCCESS, \
        "cuBLAS call failed: ", #expression)


namespace {


using namespace mfq::cuda::packed_nint;

thread_local NintInputQuantizationScope* prepared_input_scope=nullptr;
struct NintGroupQuantizeArgs {
    struct Group {int8_t *codes,*interleaved;float* scale;int* sum;int gs,groups,begin,columns;} group[4]{};
    const void* input=nullptr;
    int count=0,rows=0,width=0;
    bool fp32=false;
};

__global__ void nint8_one_quantize_reconstruct_kernel(
        const __half * __restrict__ input,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scale_output,
        __half * __restrict__ sum_output,
        __half * __restrict__ reconstructed,
        int rows,
        int width,
        int groups) {
    const int group_index = static_cast<int>(blockIdx.x);
    if (group_index >= rows * groups) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int row = group_index / groups;
    const int group = group_index - row * groups;
    const int column = group * 32 + lane;
    const float value = column < width
        ? __half2float(input[static_cast<size_t>(row) * width + column])
        : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_xor_sync(0xffffffffu, maximum, offset));
    }
    const float scale = maximum / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    int code = static_cast<int>(roundf(value * inverse));
    code = max(-127, min(127, code));
    quantized[static_cast<size_t>(group_index) * 32 + lane] =
        static_cast<int8_t>(code);

    int sum = code;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    const __half stored_scale = __float2half_rn(scale);
    if (lane == 0) {
        scale_output[group_index] = stored_scale;
        sum_output[group_index] = __float2half_rn(
            static_cast<float>(sum) * scale);
    }
    if (column < width) {
        reconstructed[static_cast<size_t>(row) * width + column] =
            __float2half_rn(
                static_cast<float>(code) * __half2float(stored_scale));
    }
}


template <typename Output>
__device__ __forceinline__ Output nint_decode_store(float value) {
    if constexpr (std::is_same_v<Output, __half>) {
        return __float2half_rn(value);
    } else {
        return value;
    }
}


template <typename Output>
__global__ void nint_decode_rows_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        Output * __restrict__ output,
        int rows,
        int groups,
        int group_size,
        int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int row = static_cast<int>(index / width);
        const int group = column / group_size;
        const size_t metadata = static_cast<size_t>(row) * groups + group;
        const int bits = static_cast<int>(row_q_bits[row]);
        const uint8_t code = unpack_nint_code(
            bitstream,
            static_cast<uint64_t>(row_q_bit_offsets[row]),
            column,
            bits);
        const float scale = neuron_scale[row] *
            static_cast<float>(subgroup_scale[metadata]);
        const float minimum = neuron_minimum[row] *
            static_cast<float>(subgroup_minimum[metadata]);
        output[index] = nint_decode_store<Output>(
            scale * static_cast<float>(code) - minimum);
    }
}


// Runtime group size deliberately avoids a kernel family per quantizer preset.
__device__ __forceinline__ float nint_activation_value(
        const void* input, size_t index, bool fp32) {
    // Match the existing fp16 activation contract without materializing a
    // converted tensor or launching a separate conversion kernel.
    return fp32 ? __half2float(__float2half_rn(static_cast<const float*>(input)[index]))
                : __half2float(static_cast<const __half*>(input)[index]);
}

template<bool CachedSum=false,int InterleavedColumns=0>
__global__ void __launch_bounds__(64) nint_quantize_activation_kernel(
        const void * __restrict__ input,
        const void * __restrict__ gate,
        int8_t * __restrict__ quantized,
        float * __restrict__ scale_output,
        int rows,
        int real_width,
        int padded_width,
        int groups,
        int group_size,
        int activation_mode,
        bool input_fp32,
        bool gate_fp32,
        int* sum_output=nullptr,
        int8_t* interleaved_output=nullptr) {
    const int row = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int column = group * group_size + lane;
    const bool valid = lane < group_size && column < real_width;
    float value = valid
        ? nint_activation_value(input, static_cast<size_t>(row) * real_width + column, input_fp32)
        : 0.0f;
    if (valid && gate != nullptr) {
        const float gate_value = nint_activation_value(
            gate, static_cast<size_t>(row) * real_width + column, gate_fp32);
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        value *= activation_mode == 1 ? sigmoid : gate_value * sigmoid;
    }
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    if (blockDim.x == 32) {
        maximum = __shfl_sync(0xffffffffu, maximum, 0);
        const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        if (lane == 0) scale_output[static_cast<size_t>(row) * groups + group] = scale;
        const int code = valid ? max(-127, min(127, static_cast<int>(roundf(value / scale)))) : 0;
        if (lane < group_size) {
            if constexpr(InterleavedColumns>0) {
                if constexpr(InterleavedColumns==5 || InterleavedColumns==6) {
                    const size_t index=size_t(group)*7+lane/4;
                    const size_t at=row<4?index*4+row:
                        size_t(groups)*7*4+index*(InterleavedColumns-4)+(row-4);
                    interleaved_output[at*4+lane%4]=static_cast<int8_t>(code);
                }else {
                const size_t at=(size_t(group)*7+lane/4)*InterleavedColumns*4+lane%4;
                interleaved_output[at+row*4]=static_cast<int8_t>(code);
                if(row==rows-1) {
                    #pragma unroll
                    for(int padding=0;padding<InterleavedColumns;++padding)
                        if(padding>=rows)interleaved_output[at+padding*4]=0;
                }
                }
            }else {
            quantized[static_cast<size_t>(row) * padded_width + column] = static_cast<int8_t>(code);
            }
        }
        if constexpr(CachedSum) {
            int sum=code;
            #pragma unroll
            for(int offset=16;offset>0;offset>>=1)
                sum+=__shfl_down_sync(0xffffffffu,sum,offset);
            if(lane==0)sum_output[size_t(row)*groups+group]=sum;
        }
        return;
    }
    __shared__ float warp_maxima[2];
    if ((lane & 31) == 0) {
        warp_maxima[lane >> 5] = maximum;
    }
    __syncthreads();
    if (lane < 32) {
        const int active_warps = (blockDim.x + 31) / 32;
        maximum = lane < active_warps ? warp_maxima[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffu, maximum, offset));
        }
    }
    __shared__ float group_scale;
    if (lane == 0) {
        group_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scale_output[static_cast<size_t>(row) * groups + group] = group_scale;
    }
    __syncthreads();
    int code = 0;
    if (lane < group_size) {
        if (valid) {
            code = static_cast<int>(roundf(value / group_scale));
            code = max(-127, min(127, code));
        }
        quantized[static_cast<size_t>(row) * padded_width + column] =
            static_cast<int8_t>(code);
    }
    if constexpr(CachedSum) {
        int sum=code;
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
        __shared__ int warp_sums[2];
        if((lane&31)==0)warp_sums[lane>>5]=sum;
        __syncthreads();
        if(lane==0)sum_output[size_t(row)*groups+group]=warp_sums[0]+warp_sums[1];
    }
}


// One warp per original group, including GS48's two values per lane.
// All formats keep the canonical fp16 rounding and per-group division.
__global__ void __launch_bounds__(32) nint_group_quantize_kernel(NintGroupQuantizeArgs a) {
    int selected=0;
    for(int i=1;i<a.count;++i)if(int(blockIdx.x)>=a.group[i].begin)selected=i;
    const auto g=a.group[selected];
    const int group=int(blockIdx.x)-g.begin,row=blockIdx.y,lane=threadIdx.x;
    const int column=group*g.gs+lane;
    const bool valid0=lane<g.gs && column<a.width;
    const bool valid1=lane+32<g.gs && column+32<a.width;
    const float v0=valid0?nint_activation_value(a.input,size_t(row)*a.width+column,a.fp32):0.f;
    const float v1=valid1?nint_activation_value(a.input,size_t(row)*a.width+column+32,a.fp32):0.f;
    float maximum=fmaxf(fabsf(v0),fabsf(v1));
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1)
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
    maximum=__shfl_sync(0xffffffffu,maximum,0);
    const float scale=maximum>0.f?maximum/127.f:1.f;
    if(lane==0)g.scale[size_t(row)*g.groups+group]=scale;
    const int c0=valid0?max(-127,min(127,int(roundf(v0/scale)))):0;
    const int c1=valid1?max(-127,min(127,int(roundf(v1/scale)))):0;
    if(lane<g.gs)g.codes[size_t(row)*g.groups*g.gs+column]=int8_t(c0);
    if(lane+32<g.gs)g.codes[size_t(row)*g.groups*g.gs+column+32]=int8_t(c1);
    if(g.interleaved && lane<g.gs) {
        const size_t index=size_t(group)*7+lane/4;
        if(g.columns==5 || g.columns==6) {
            const size_t at=row<4?index*4+row:size_t(g.groups)*7*4+index*(g.columns-4)+(row-4);
            g.interleaved[at*4+lane%4]=int8_t(c0);
        } else {
            const size_t at=index*g.columns*4+lane%4;
            g.interleaved[at+row*4]=int8_t(c0);
            if(row==a.rows-1)for(int padding=a.rows;padding<g.columns;++padding)
                g.interleaved[at+padding*4]=0;
        }
    }
    int sum=c0+c1;
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
    if(lane==0)g.sum[size_t(row)*g.groups+group]=sum;
}

constexpr int kNintPrefillTileN = 64;
constexpr int kNintPrefillTileK = 64;
constexpr int kNintPrefillStrideK = 72;
constexpr int kNintPrefillWarps = 8;


template <bool AsyncCopy, int TileM, bool Dense = false>
__device__ __forceinline__ void nint_prefill_load_activation_tile(
        const __half * __restrict__ input,
        const int32_t * __restrict__ ids_dst,
        __half * __restrict__ activation_tile,
        int first,
        int last,
        int routes,
        int input_width,
        int column_base,
        bool routed_input,
        int thread) {
    constexpr int vector_width = 8;
    constexpr int vectors_per_row = kNintPrefillTileK / vector_width;
    constexpr int vectors = TileM * vectors_per_row;
    for (int index = thread; index < vectors;
         index += kNintPrefillWarps * 32) {
        const int matrix_row = index / vectors_per_row;
        const int vector = index - matrix_row * vectors_per_row;
        const int column = column_base + vector * vector_width;
        const int compact = first + matrix_row;
        int source_row = -1;
        if (compact < last) {
            const int pair = Dense ? compact : ids_dst[compact];
            source_row = Dense || routed_input ? pair : pair / routes;
        }
        __half * destination = activation_tile +
            matrix_row * kNintPrefillStrideK + vector * vector_width;
        if (source_row < 0 || column >= input_width) {
            *reinterpret_cast<int4 *>(destination) = make_int4(0, 0, 0, 0);
        } else if ((input_width & (vector_width - 1)) == 0 &&
                column + vector_width <= input_width) {
            const __half * source = input +
                static_cast<size_t>(source_row) * input_width + column;
            if constexpr (AsyncCopy) {
                mfq::cuda_detail::copy_16_async(destination, source);
            } else {
                *reinterpret_cast<int4 *>(destination) =
                    *reinterpret_cast<const int4 *>(source);
            }
        } else {
#pragma unroll
            for (int element = 0; element < vector_width; ++element) {
                destination[element] = column + element < input_width
                    ? input[static_cast<size_t>(source_row) * input_width +
                        column + element]
                    : __float2half_rn(0.0f);
            }
        }
    }
}


__device__ __forceinline__ void nint_prefill_load_weight_tile(
        const uint8_t * __restrict__ expert_stream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        __half * __restrict__ weight_tile,
        int local_expert,
        int output_rows,
        int result_rows,
        int groups,
        int group_size,
        int input_width,
        int result_row_base,
        int column_base,
        int projection,
        int thread) {
    constexpr int values_per_vector = 8;
    constexpr int vectors_per_row =
        kNintPrefillTileK / values_per_vector;
    constexpr int vectors = kNintPrefillTileN * vectors_per_row;
    for (int index = thread; index < vectors;
         index += kNintPrefillWarps * 32) {
        const int tile_row = index / vectors_per_row;
        const int vector = index - tile_row * vectors_per_row;
        const int result_row = result_row_base + tile_row;
        const int output_row = result_row + projection * result_rows;
        const int column = column_base + vector * values_per_vector;
        __half * destination = weight_tile +
            tile_row * kNintPrefillStrideK + vector * values_per_vector;
        const bool valid_row = result_row < result_rows &&
            output_row < output_rows;
        const int weight_row = valid_row
            ? local_expert * output_rows + output_row
            : 0;
        const int row_lane = thread & 7;
        int bits = valid_row && row_lane == 0
            ? static_cast<int>(row_q_bits[weight_row])
            : 0;
        uint64_t row_bit_offset = valid_row && row_lane == 0
            ? static_cast<uint64_t>(row_q_bit_offsets[weight_row])
            : 0u;
        float outer_scale = valid_row && row_lane == 0
            ? neuron_scale[weight_row]
            : 0.0f;
        float outer_minimum = valid_row && row_lane == 0
            ? neuron_minimum[weight_row]
            : 0.0f;
        const uint32_t offset_low = __shfl_sync(
            0xffffffffu, static_cast<uint32_t>(row_bit_offset), 0, 8);
        const uint32_t offset_high = __shfl_sync(
            0xffffffffu, static_cast<uint32_t>(row_bit_offset >> 32), 0, 8);
        bits = __shfl_sync(0xffffffffu, bits, 0, 8);
        outer_scale = __shfl_sync(0xffffffffu, outer_scale, 0, 8);
        outer_minimum = __shfl_sync(0xffffffffu, outer_minimum, 0, 8);
        row_bit_offset = static_cast<uint64_t>(offset_low) |
            (static_cast<uint64_t>(offset_high) << 32);

        const int valid_values = valid_row && column < input_width
            ? min(values_per_vector, input_width - column)
            : 0;
        const uint64_t codes = valid_values > 0
            ? unpack_nint_codes8_packed(
                expert_stream,
                row_bit_offset + static_cast<uint64_t>(column) * bits,
                bits)
            : 0u;
        const uint32_t code_mask = bits > 0 ? (1u << bits) - 1u : 0u;
        const size_t metadata_base = static_cast<size_t>(weight_row) * groups;

        // The eight lanes serving one row cooperatively load every subgroup
        // touched by this 64-wide tile exactly once.  Standard NINT groups
        // are at least eight values, so at most eight groups are present.
        // Smaller experimental groups retain this kernel and load metadata
        // directly while decoding each value.
        const int physical_last_column = groups * group_size - 1;
        const int tile_first_group = column_base / group_size;
        const int tile_last_column = column_base + min(
            kNintPrefillTileK - 1,
            physical_last_column - column_base);
        const int tile_last_group = tile_last_column / group_size;
        const int first_group = min(column, physical_last_column) / group_size;
        const int last_group = valid_values > 0
            ? (column + valid_values - 1) / group_size
            : first_group;
        float first_scale = 0.0f;
        float first_minimum = 0.0f;
        float last_scale = 0.0f;
        float last_minimum = 0.0f;
        if (group_size >= values_per_vector) {
            const int source_group = tile_first_group + row_lane;
            float source_scale = 0.0f;
            float source_minimum = 0.0f;
            if (valid_row && source_group <= tile_last_group) {
                const size_t metadata = metadata_base + source_group;
                source_scale = outer_scale *
                    static_cast<float>(subgroup_scale[metadata]);
                source_minimum = outer_minimum *
                    static_cast<float>(subgroup_minimum[metadata]);
            }
            const int first_source_lane = first_group - tile_first_group;
            const int last_source_lane = last_group - tile_first_group;
            first_scale = __shfl_sync(
                0xffffffffu, source_scale, first_source_lane, 8);
            first_minimum = __shfl_sync(
                0xffffffffu, source_minimum, first_source_lane, 8);
            last_scale = __shfl_sync(
                0xffffffffu, source_scale, last_source_lane, 8);
            last_minimum = __shfl_sync(
                0xffffffffu, source_minimum, last_source_lane, 8);
        }

#pragma unroll
        for (int element = 0; element < values_per_vector; element += 2) {
            float decoded[2] = {0.0f, 0.0f};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int local = element + half;
                if (local < valid_values) {
                    const int value_column = column + local;
                    const int group = value_column / group_size;
                    float scale = first_scale;
                    float minimum = first_minimum;
                    if (group_size < values_per_vector) {
                        const size_t metadata = metadata_base + group;
                        scale = outer_scale *
                            static_cast<float>(subgroup_scale[metadata]);
                        minimum = outer_minimum *
                            static_cast<float>(subgroup_minimum[metadata]);
                    } else if (group != first_group) {
                        scale = last_scale;
                        minimum = last_minimum;
                    }
                    const int code = static_cast<int>(
                        (codes >> (local * bits)) & code_mask);
                    decoded[half] =
                        scale * static_cast<float>(code) - minimum;
                }
            }
            reinterpret_cast<__half2 *>(destination)[element / 2] =
                __floats2half2_rn(decoded[0], decoded[1]);
        }
    }
}


// FusedGlu is an epilogue contract, not a q/k specialization.  Both variants
// consume the same heterogeneous row metadata and execute the same tiled
// decoder.  Keeping the plain projection separate prevents it from reserving
// a second accumulator set that it can never use.
template <int TileM, bool AsyncActivation, bool FusedGlu, bool Dense = false>
__global__ void __launch_bounds__(256, 1) nint_matmul_tiled_prefill_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const __half * __restrict__ input,
        __half * __restrict__ output,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        int routes,
        int experts,
        int output_rows,
        int groups,
        int group_size,
        int input_width,
        int q_expert_stride,
        int pool_phase,
        int epilogue_mode,
        bool routed_input,
        int dense_rows,
        int split_k,
        float* __restrict__ dense_partial) {
    static_assert(
        TileM == 16 || TileM == 32 || TileM == 64 || TileM == 128);
    constexpr int matrix_tiles_m = TileM / 16;
    constexpr int matrix_tiles_n = kNintPrefillTileN / 16;
    constexpr int accumulators_per_warp = (matrix_tiles_m + 1) / 2;
    constexpr int operand_halves =
        (kNintPrefillTileN + TileM) * kNintPrefillStrideK;
    constexpr int output_floats =
        2 * kNintPrefillWarps * 16 * 16;
    constexpr int operand_bytes = operand_halves * sizeof(__half);
    constexpr int output_bytes = output_floats * sizeof(float);
    constexpr int shared_bytes =
        operand_bytes > output_bytes ? operand_bytes : output_bytes;
    __shared__ __align__(16) uint8_t storage[shared_bytes];
    __half * weight_tile = reinterpret_cast<__half *>(storage);
    __half * activation_tile = weight_tile +
        kNintPrefillTileN * kNintPrefillStrideK;

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int thread = warp * 32 + lane;
    const int warp_m0 = warp / matrix_tiles_n;
    const int warp_n = warp - warp_m0 * matrix_tiles_n;
    const int result_rows = FusedGlu ? output_rows / 2 : output_rows;
    constexpr int projections = FusedGlu ? 2 : 1;
    const int output_tiles =
        (result_rows + kNintPrefillTileN - 1) / kNintPrefillTileN;
    const int total_tiles = Dense ? (dense_rows + TileM - 1) / TileM : tile_bounds[experts];
    const int64_t total_tasks =
        static_cast<int64_t>(total_tiles) * output_tiles;

    using FragmentA = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, 16, 16, 16,
        __half, nvcuda::wmma::row_major>;
    using FragmentB = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, 16, 16, 16,
        __half, nvcuda::wmma::col_major>;
    using FragmentC = nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, 16, 16, 16, float>;

    const int64_t task_shift = total_tasks == 0
        ? 0
        : (static_cast<int64_t>(pool_phase) * 17 * output_tiles) %
            total_tasks;
    for (int64_t logical_task = blockIdx.x;
         logical_task < total_tasks;
         logical_task += gridDim.x) {
        const int64_t task = (logical_task + task_shift) % total_tasks;
        const int tile = static_cast<int>(task / output_tiles);
        const int output_tile = static_cast<int>(
            task - static_cast<int64_t>(tile) * output_tiles);
        const int expert = Dense ? 0 : tile_experts[tile];
        const int local_expert = Dense ? 0 : expert_local[expert];
        if (local_expert < 0) {
            continue;
        }
        const int local_tile = Dense ? tile : tile - tile_bounds[expert];
        const int first = Dense ? local_tile * TileM : expert_bounds[expert] + local_tile * TileM;
        const int last = min(first + TileM, Dense ? dense_rows : expert_bounds[expert + 1]);
        const int result_row_base = output_tile * kNintPrefillTileN;
        const uint8_t * expert_stream = bitstream +
            static_cast<size_t>(local_expert) * q_expert_stride;

        FragmentC accumulators[projections][accumulators_per_warp];
#pragma unroll
        for (int projection = 0; projection < projections; ++projection) {
#pragma unroll
            for (int accumulator = 0;
                 accumulator < accumulators_per_warp; ++accumulator) {
                nvcuda::wmma::fill_fragment(
                    accumulators[projection][accumulator], 0.0f);
            }
        }

        const int split_columns = ((input_width + kNintPrefillTileK*split_k - 1) /
            (kNintPrefillTileK*split_k))*kNintPrefillTileK;
        const int column_begin = Dense ? int(blockIdx.y)*split_columns : 0;
        const int column_end = Dense ? min(input_width,column_begin+split_columns) : input_width;
        for (int column_base = column_begin; column_base < column_end;
             column_base += kNintPrefillTileK) {
            nint_prefill_load_activation_tile<AsyncActivation, TileM, Dense>(
                input, ids_dst, activation_tile, first, last, routes,
                input_width, column_base, routed_input, thread);
            if constexpr (AsyncActivation) {
                mfq::cuda_detail::copy_async_commit();
            }
            nint_prefill_load_weight_tile(
                expert_stream, row_q_bits, row_q_bit_offsets,
                subgroup_scale, subgroup_minimum, neuron_scale,
                neuron_minimum, weight_tile, local_expert, output_rows,
                result_rows, groups, group_size, input_width,
                result_row_base, column_base, 0, thread);
            if constexpr (AsyncActivation) {
                mfq::cuda_detail::copy_async_wait();
            }
            __syncthreads();

            const bool active_warp = warp_m0 < matrix_tiles_m;
#pragma unroll
            for (int k_local = 0; k_local < kNintPrefillTileK;
                 k_local += 16) {
                if (active_warp && column_base + k_local < input_width) {
                    FragmentB weight_fragment;
                    nvcuda::wmma::load_matrix_sync(
                        weight_fragment,
                        weight_tile + warp_n * 16 *
                            kNintPrefillStrideK + k_local,
                        kNintPrefillStrideK);
#pragma unroll
                    for (int accumulator = 0;
                         accumulator < accumulators_per_warp;
                         ++accumulator) {
                        const int matrix_tile_m =
                            warp_m0 + accumulator * 2;
                        if (matrix_tile_m < matrix_tiles_m) {
                            FragmentA activation_fragment;
                            nvcuda::wmma::load_matrix_sync(
                                activation_fragment,
                                activation_tile + matrix_tile_m * 16 *
                                    kNintPrefillStrideK + k_local,
                                kNintPrefillStrideK);
                            nvcuda::wmma::mma_sync(
                                accumulators[0][accumulator],
                                activation_fragment, weight_fragment,
                                accumulators[0][accumulator]);
                        }
                    }
                }
            }
            __syncthreads();

            if constexpr (FusedGlu) {
                nint_prefill_load_weight_tile(
                    expert_stream, row_q_bits, row_q_bit_offsets,
                    subgroup_scale, subgroup_minimum, neuron_scale,
                    neuron_minimum, weight_tile, local_expert, output_rows,
                    result_rows, groups, group_size, input_width,
                    result_row_base, column_base, 1, thread);
                __syncthreads();
#pragma unroll
                for (int k_local = 0; k_local < kNintPrefillTileK;
                     k_local += 16) {
                    if (active_warp && column_base + k_local < input_width) {
                        FragmentB weight_fragment;
                        nvcuda::wmma::load_matrix_sync(
                            weight_fragment,
                            weight_tile + warp_n * 16 *
                                kNintPrefillStrideK + k_local,
                            kNintPrefillStrideK);
#pragma unroll
                        for (int accumulator = 0;
                             accumulator < accumulators_per_warp;
                             ++accumulator) {
                            const int matrix_tile_m =
                                warp_m0 + accumulator * 2;
                            if (matrix_tile_m < matrix_tiles_m) {
                                FragmentA activation_fragment;
                                nvcuda::wmma::load_matrix_sync(
                                    activation_fragment,
                                    activation_tile + matrix_tile_m * 16 *
                                        kNintPrefillStrideK + k_local,
                                    kNintPrefillStrideK);
                                nvcuda::wmma::mma_sync(
                                    accumulators[1][accumulator],
                                    activation_fragment, weight_fragment,
                                    accumulators[1][accumulator]);
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }

        float * output_tile_storage = reinterpret_cast<float *>(storage);
#pragma unroll
        for (int accumulator = 0;
             accumulator < accumulators_per_warp; ++accumulator) {
            const int matrix_tile_m = warp_m0 + accumulator * 2;
            if (matrix_tile_m >= matrix_tiles_m) {
                continue;
            }
            float * first_projection = output_tile_storage + warp * 16 * 16;
            float * second_projection = first_projection +
                kNintPrefillWarps * 16 * 16;
            nvcuda::wmma::store_matrix_sync(
                first_projection, accumulators[0][accumulator], 16,
                nvcuda::wmma::mem_row_major);
            if constexpr (FusedGlu) {
                nvcuda::wmma::store_matrix_sync(
                    second_projection, accumulators[1][accumulator], 16,
                    nvcuda::wmma::mem_row_major);
            }
            __syncwarp();
#pragma unroll
            for (int element = lane; element < 16 * 16; element += 32) {
                const int row = element / 16;
                const int column = element - row * 16;
                const int compact = first + matrix_tile_m * 16 + row;
                const int result_row =
                    result_row_base + warp_n * 16 + column;
                if (compact < last && result_row < result_rows) {
                    const int pair = Dense ? compact : ids_dst[compact];
                    float value = first_projection[element];
                    if constexpr (FusedGlu) {
                        const float gate = __half2float(
                            __float2half_rn(value));
                        const float up = __half2float(
                            __float2half_rn(second_projection[element]));
                        value = mfq_glu_runtime(
                            gate, up, epilogue_mode == 2 ? 1 : 0);
                    }
                    if constexpr(Dense) {
                        if(split_k>1)dense_partial[(size_t(blockIdx.y)*dense_rows+pair)*result_rows+result_row]=value;
                        else output[static_cast<size_t>(pair)*result_rows+result_row]=__float2half_rn(value);
                    } else output[static_cast<size_t>(pair)*result_rows+result_row]=__float2half_rn(value);
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }
}


template<int GroupLanes=0,int ResidentBlocks=1,int Rows=2>
__global__ void __launch_bounds__(128,ResidentBlocks) nint_mixed_routes_kernel(
        const int64_t* pointers,const int32_t* params,const int32_t* expert_pool,
        const int32_t* expert_local,const int32_t* ids,__half* output,
        int pairs,int routes,int experts,int output_rows,bool routed_input,bool multi_sum) {
    const int pair=blockIdx.y;
    const int output_row=int(blockIdx.x)*(4*Rows)+int(threadIdx.y)*Rows;
    if(pair>=pairs || output_row>=output_rows)return;
    const int expert=ids[pair];
    if(static_cast<unsigned>(expert)>=static_cast<unsigned>(experts))return;
    const int pool=expert_pool[expert],local=expert_local[expert];
    if(pool<0 || local<0)return;
    const auto* p=pointers+int64_t(pool)*9;
    const auto* geometry=params+int64_t(pool)*3;
    if constexpr(GroupLanes!=0) {
        if(geometry[1]==24 || geometry[1]==28) {
            float values[Rows];
            const int source=routed_input?pair:pair/routes;
#define MFQ_NINT_ROUTED_GROUP(GS) mfq::cuda::packed_nint::nint_grouped_routed_pair<GS,GroupLanes,Rows>( \
                reinterpret_cast<const uint8_t*>(p[0]),reinterpret_cast<const uint8_t*>(p[1]), \
                reinterpret_cast<const int64_t*>(p[2]),reinterpret_cast<const uint8_t*>(p[3]), \
                reinterpret_cast<const uint8_t*>(p[4]),reinterpret_cast<const float*>(p[5]), \
                reinterpret_cast<const float*>(p[6]), \
                reinterpret_cast<const int8_t*>(p[7])+int64_t(source)*geometry[0]*geometry[1], \
                reinterpret_cast<const float*>(p[8])+int64_t(source)*geometry[0], \
                local,output_row,output_rows,geometry[0],geometry[2],values,multi_sum)
            if(geometry[1]==24)MFQ_NINT_ROUTED_GROUP(24);else MFQ_NINT_ROUTED_GROUP(28);
#undef MFQ_NINT_ROUTED_GROUP
            if(threadIdx.x==0) {
#pragma unroll
                for(int r=0;r<Rows;++r)if(output_row+r<output_rows)
                    output[int64_t(pair)*output_rows+output_row+r]=__float2half_rn(values[r]);
            }
            return;
        }
    }
    static_assert(Rows==2 || GroupLanes==32);
    if constexpr(Rows==1) {
        // The one-row variant is selected only for the grouped geometries.
        // Other group sizes use paired rows with the unchanged fallback.
        float values[2];
        nint_matmul_routed_pair<true>(reinterpret_cast<const uint8_t*>(p[0]),
            reinterpret_cast<const uint8_t*>(p[1]),reinterpret_cast<const int64_t*>(p[2]),
            reinterpret_cast<const uint8_t*>(p[3]),reinterpret_cast<const uint8_t*>(p[4]),
            reinterpret_cast<const float*>(p[5]),reinterpret_cast<const float*>(p[6]),
            reinterpret_cast<const int8_t*>(p[7]),reinterpret_cast<const float*>(p[8]),
            output,pair,routed_input?pair:pair/routes,local,output_row,output_rows,
            geometry[0],geometry[0]*geometry[1],geometry[1],geometry[2],0,values);
        if(threadIdx.x==0)output[int64_t(pair)*output_rows+output_row]=__float2half(values[0]);
        return;
    }
    nint_matmul_routed_pair(reinterpret_cast<const uint8_t*>(p[0]),
        reinterpret_cast<const uint8_t*>(p[1]),reinterpret_cast<const int64_t*>(p[2]),
        reinterpret_cast<const uint8_t*>(p[3]),reinterpret_cast<const uint8_t*>(p[4]),
        reinterpret_cast<const float*>(p[5]),reinterpret_cast<const float*>(p[6]),
        reinterpret_cast<const int8_t*>(p[7]),reinterpret_cast<const float*>(p[8]),
        output,pair,routed_input?pair:pair/routes,local,output_row,output_rows,
        geometry[0],geometry[0]*geometry[1],geometry[1],geometry[2],0);
}

template<int GroupSize,int Bits,int Warps=1,bool Single=false,bool LateScale=false,bool PackedRows=false>
__global__ void __launch_bounds__(128) nint_mixed_routes_hinted_kernel(
        const int64_t* pointers,const int32_t* params,const int32_t* expert_pool,
        const int32_t* expert_local,const int32_t* ids,__half* output,
        int pairs,int routes,int experts,int output_rows,bool routed_input,
        const NintSingleRouteWeight single,bool multi_sum) {
    static_assert(Warps==1 || Warps==4);
    static_assert(!PackedRows || Single);
    constexpr int Rows=Warps==4?4:2,rows_per_block=Warps==4?4:8;
    const int pair=blockIdx.y,first=int(blockIdx.x)*rows_per_block+(Warps==1?int(threadIdx.y)*Rows:0);
    if(pair>=pairs || first>=output_rows)return;
    const int expert=ids[pair];
    if(static_cast<unsigned>(expert)>=static_cast<unsigned>(experts))return;
    const int local=expert_local[expert];
    const int64_t* p;const int32_t* geometry;
    if constexpr(Single) {
        if(unsigned(local)>=unsigned(single.local_experts))return;
        p=single.pointers;geometry=single.geometry;
    } else {
        const int pool=expert_pool[expert];if(pool<0 || local<0)return;
        p=pointers+int64_t(pool)*9;geometry=params+int64_t(pool)*3;
    }
    const auto* bits=reinterpret_cast<const uint8_t*>(p[1]);
    const int neuron=local*output_rows+first,source=routed_input?pair:pair/routes;
    // Expert slots and row metadata may change after graph capture. A host
    // hint never substitutes for checking the current rows on the device.
    bool matching=geometry[1]==GroupSize && geometry[0]<INT_MAX/(GroupSize*Bits);
#pragma unroll
    for(int r=0;r<Rows;++r)if(first+r<output_rows)matching=matching && bits[neuron+r]==Bits;
    if(matching) {
        float values[Rows];
        mfq::cuda::packed_nint::nint_fixed_group_rows<GroupSize,Bits,Warps,Rows,LateScale,PackedRows>(
            reinterpret_cast<const uint8_t*>(p[0]),reinterpret_cast<const int64_t*>(p[2]),
            reinterpret_cast<const uint8_t*>(p[3]),reinterpret_cast<const uint8_t*>(p[4]),
            reinterpret_cast<const float*>(p[5]),reinterpret_cast<const float*>(p[6]),
            reinterpret_cast<const int8_t*>(p[7])+int64_t(source)*geometry[0]*GroupSize,
            reinterpret_cast<const float*>(p[8])+int64_t(source)*geometry[0],
            local,first,output_rows,geometry[0],geometry[2],values,multi_sum,
            reinterpret_cast<const uint4*>(single.row_metadata));
        if constexpr(Warps==4) {
            __shared__ float partial[4][Rows];
            if(threadIdx.x==0) {
#pragma unroll
                for(int r=0;r<Rows;++r)partial[threadIdx.y][r]=values[r];
            }
            __syncthreads();
            if(threadIdx.x==0 && threadIdx.y==0) {
#pragma unroll
                for(int r=0;r<Rows;++r)if(first+r<output_rows)
                    output[int64_t(pair)*output_rows+first+r]=__float2half_rn(
                        (partial[0][r]+partial[1][r])+(partial[2][r]+partial[3][r]));
            }
        }else if(threadIdx.x==0) {
            output[int64_t(pair)*output_rows+first]=__float2half_rn(values[0]);
            if(first+1<output_rows)output[int64_t(pair)*output_rows+first+1]=__float2half_rn(values[1]);
        }
    } else if(Warps==1 || threadIdx.y==0) {
#pragma unroll
        for(int r=0;r<Rows;r+=2)nint_matmul_routed_pair(reinterpret_cast<const uint8_t*>(p[0]),bits,
            reinterpret_cast<const int64_t*>(p[2]),reinterpret_cast<const uint8_t*>(p[3]),
            reinterpret_cast<const uint8_t*>(p[4]),reinterpret_cast<const float*>(p[5]),
            reinterpret_cast<const float*>(p[6]),reinterpret_cast<const int8_t*>(p[7]),
            reinterpret_cast<const float*>(p[8]),output,pair,source,local,first+r,output_rows,
            geometry[0],geometry[0]*geometry[1],geometry[1],geometry[2],0);
    }
}

__device__ __forceinline__ int unpack_nint_codes4_words(
        const uint8_t* stream,uint64_t bit_offset,int bits,uint64_t bytes) {
    const uint64_t word_byte=(bit_offset>>3)&~uint64_t(3);
    // The selected stream base is aligned. Small unpadded tensors retain the
    // byte decoder at their end; two word loads never cross the storage bound.
    if(word_byte+8>bytes)return unpack_nint_codes4(stream,bit_offset,bits);
    const auto* words=reinterpret_cast<const uint32_t*>(stream+word_byte);
    const uint32_t packed=__funnelshift_r(words[0],words[1],int(bit_offset&31u));
    const uint32_t pairs=__byte_perm(packed,packed>>(2*bits),0x5410);
    const uint32_t codes=__byte_perm(pairs,pairs>>bits,0x6240);
    return static_cast<int>(codes * 1u & (((1u<<bits)-1u)*0x01010101u));

}

__device__ __forceinline__ int unpack_nint_codes4_unchecked(
        const uint8_t* stream,uint64_t bit_offset,int bits,uint64_t bytes) {
    const uint64_t word_byte=(bit_offset>>3)&~uint64_t(3);
    // The caller proves both aligned word loads fit the packed row storage.
    const auto* words=reinterpret_cast<const uint32_t*>(stream+word_byte);
    const uint32_t packed=__funnelshift_r(words[0],words[1],int(bit_offset&31u));
    const uint32_t pairs=__byte_perm(packed,packed>>(2*bits),0x5410);
    const uint32_t codes=__byte_perm(pairs,pairs>>bits,0x6240);
    return static_cast<int>(codes * 1u & (((1u<<bits)-1u)*0x01010101u));

}

// A lane accumulates one complete affine group with integer dot products.
// Row-specific bit widths remain dynamic, including adaptive NINTv2 and q8.
template<int GroupSize, int Rows, int Warps>
__global__ void __launch_bounds__(128) nint_matmul_dense_kernel(
        const uint8_t* __restrict__ bitstream,
        const uint8_t* __restrict__ row_q_bits,
        const int64_t* __restrict__ row_q_bit_offsets,
        const uint8_t* __restrict__ subgroup_scale,
        const uint8_t* __restrict__ subgroup_minimum,
        const float* __restrict__ neuron_scale,
        const float* __restrict__ neuron_minimum,
        const int8_t* __restrict__ activation,
        const float* __restrict__ activation_scale,
        half* __restrict__ output,
        int activation_rows, int output_rows, int groups, uint64_t packed_bytes) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int row = blockIdx.x * (4 / Warps) + warp / Warps;
    const int local_warp = warp % Warps;
    const bool valid_row = row < output_rows;
    const int bits = valid_row ? row_q_bits[row] : 4;
    const uint64_t row_bit = valid_row ? row_q_bit_offsets[row] : 0;
    const int padded_width = groups * GroupSize;
    const float outer_scale = valid_row ? neuron_scale[row] : 0.0f;
    const float outer_minimum = valid_row ? neuron_minimum[row] : 0.0f;
    const uint64_t last_word = ((row_bit + uint64_t(padded_width) * bits) >> 3) & ~uint64_t(3);
    const bool full_words = last_word <= packed_bytes && packed_bytes - last_word >= 8;
    float acc[Rows] = {};
    for (int group = valid_row ? local_warp * 32 + lane : groups;
         group < groups; group += Warps * 32) {
        int dots[Rows] = {}, sums[Rows] = {};
#pragma unroll
        for (int chunk = 0; chunk < GroupSize / 4; ++chunk) {
            const int column = group * GroupSize + chunk * 4;
            const uint64_t bit = row_bit + uint64_t(column) * bits;
            const int qw = full_words ? unpack_nint_codes4_unchecked(bitstream, bit, bits, packed_bytes)
                : unpack_nint_codes4_words(bitstream, bit, bits, packed_bytes);
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                if (m < activation_rows) {
                    const int qx = *reinterpret_cast<const int*>(activation + size_t(m) * padded_width + column);
                    sums[m] = __dp4a(0x01010101, qx, sums[m]);
                    dots[m] = __dp4a(bits == 8 ? qw ^ int(0x80808080u) : qw, qx, dots[m]);
                }
            }
        }
        const float scale = outer_scale * float(subgroup_scale[size_t(row) * groups + group]);
        const float minimum = outer_minimum * float(subgroup_minimum[size_t(row) * groups + group]);
#pragma unroll
        for (int m = 0; m < Rows; ++m) {
            if (m < activation_rows) {
                const int dot = dots[m] + (bits == 8 ? 128 * sums[m] : 0);
                acc[m] += activation_scale[size_t(m) * groups + group] *
                    (scale * float(dot) - minimum * float(sums[m]));
            }
        }
    }
#pragma unroll
    for (int m = 0; m < Rows; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], offset);
    }
    if constexpr (Warps == 1) {
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (lane == 0 && valid_row && m < activation_rows)
                output[size_t(m) * output_rows + row] = __float2half_rn(acc[m]);
    } else {
        __shared__ float partial[Rows][4];
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (lane == 0) partial[m][warp] = acc[m];
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                float value = lane < Warps ? partial[m][lane] : 0.0f;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_xor_sync(0xffffffffu, value, offset);
                if (lane == 0 && valid_row && m < activation_rows)
                    output[size_t(m) * output_rows + row] = __float2half_rn(value);
            }
        }
    }
}

#include "nint_whole_group.cuh"
#include "nint_whole_warp_rows.cuh"

// Decode four adjacent values once and store one aligned eight-byte vector.
template<int GroupSize>
__global__ void nint_decode_vec4_kernel(
        const uint8_t* bitstream, const uint8_t* row_q_bits,
        const int64_t* row_q_bit_offsets, const uint8_t* subgroup_scale,
        const uint8_t* subgroup_minimum, const float* neuron_scale,
        const float* neuron_minimum, half* output,
        int rows, int groups, int width, uint64_t packed_bytes) {
    const int column = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (column >= width) return;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int bits = row_q_bits[row];
        const int codes = unpack_nint_codes4_words(bitstream,
            uint64_t(row_q_bit_offsets[row]) + uint64_t(column) * bits, bits, packed_bytes);
        const size_t metadata = size_t(row) * groups + column / GroupSize;
        const float scale = neuron_scale[row] * float(subgroup_scale[metadata]);
        const float minimum = neuron_minimum[row] * float(subgroup_minimum[metadata]);
        half values[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            values[i] = __float2half_rn(scale * float((uint32_t(codes) >> (8 * i)) & 255u) - minimum);
        reinterpret_cast<uint2*>(output + size_t(row) * width)[column / 4] = make_uint2(
            (uint32_t(__half_as_ushort(values[0])) | (uint32_t(__half_as_ushort(values[1])) << 16)),
            (uint32_t(__half_as_ushort(values[2])) | (uint32_t(__half_as_ushort(values[3])) << 16)));
    }
}

// One NINT compute kernel covers every q, k, group size, M<=8, and routed MFE
// projection. q is row metadata; k has already been baked into the subgroup
// metadata values. Routed execution changes only the indexing contract, not
// the packed-weight compute kernel.
// Accumulate a complete quantization group in integers before its affine.
// The caller selects this experimental math only for one activation row.
__device__ __forceinline__ float nint_group_partial(
        const uint8_t* bitstream,int bits,uint64_t row_bit_offset,
        const uint8_t* scale_row,const uint8_t* minimum_row,
        float outer_scale,float outer_minimum,const int8_t* activation,
        const float* activation_scale,int groups,int padded_width,int group_size,
        uint64_t packed_bytes,int lane) {
    float partial=0.0f;
    const uint64_t end_word=((row_bit_offset+uint64_t(padded_width)*uint64_t(bits))>>3)&~uint64_t(3);
    const uint32_t row_bit_remainder=uint32_t(row_bit_offset&31u);
    const bool relative_fits=uint64_t(padded_width)*uint64_t(bits)+row_bit_remainder<=0xffffffffu;
    const auto* row_stream=bitstream+((row_bit_offset>>3)&~uint64_t(3));
    const bool complete_words=relative_fits && end_word<=packed_bytes && packed_bytes-end_word>=8;
    for(int group=lane;group<groups;group+=32) {
        const int column0=group*group_size;
        int dot=0,activation_sum=0;
        int element=0;
        for(;element+3<group_size;element+=4) {
            const int column=column0+element;
            const uint32_t relative_bit=row_bit_remainder+uint32_t(column)*uint32_t(bits);
            const int weight_codes=complete_words
                ? unpack_nint_codes4_unchecked(row_stream,relative_bit,bits,packed_bytes)
                : unpack_nint_codes4(bitstream,row_bit_offset+uint64_t(column)*uint64_t(bits),bits);
            const int activation_codes=load_i8x4(activation+column);
            activation_sum=__dp4a(0x01010101,activation_codes,activation_sum);
            dot=__dp4a(bits==8 ? (weight_codes^int(0x80808080u)):weight_codes,activation_codes,dot);
        }
        if(bits==8)dot+=128*activation_sum;
        for(;element<group_size;++element) {
            const int column=column0+element;
            const int qx=int(activation[column]);
            const int qw=int(unpack_nint_code(bitstream,row_bit_offset,column,bits));
            activation_sum+=qx;dot+=qx*qw;
        }
        const float inner_scale=float(scale_row[group]);
        const float inner_minimum=float(minimum_row[group]);
        partial+=activation_scale[group]*(outer_scale*inner_scale*float(dot)-outer_minimum*inner_minimum*float(activation_sum));
    }

    return partial;
}

template<bool Single=false,bool Grouped=false,int RowCapacity=8>
__global__ void __launch_bounds__(128) nint_matmul_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int activation_rows,
        int output_rows,
        int groups,
        int padded_width,
        int group_size,
        const int32_t * __restrict__ route_ids,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        int routes,
        int experts,
        int q_expert_stride,
        int pool_phase,
        int epilogue_mode,
        bool routed_input,
        uint64_t packed_bytes=0) {
    constexpr int warps_per_block = 4;
    constexpr int maximum_activation_rows = Single ? 1 : RowCapacity;
    constexpr int routed_rows_per_warp = 2;
    const int chunks = (group_size + 3) / 4;
    const int groups_per_warp = 32 / chunks;
    const int lane = static_cast<int>(threadIdx.x);

    if constexpr(!Single) {
    if (ids_dst != nullptr) {
        constexpr int route_tile = 8;
        const int rows_per_task = epilogue_mode == 0
            ? routed_rows_per_warp
            : 1;
        const int result_rows = epilogue_mode == 0
            ? output_rows
            : output_rows / 2;
        const int row_blocks =
            (result_rows + rows_per_task - 1) / rows_per_task;
        const int total_tiles = tile_bounds[experts];
        const int64_t total_tasks =
            static_cast<int64_t>(total_tiles) * row_blocks;
        const int64_t task_shift = total_tasks == 0
            ? 0
            : (static_cast<int64_t>(pool_phase) * 17 * row_blocks) %
                total_tasks;
        for (int64_t logical_task = blockIdx.x;
             logical_task < total_tasks;
             logical_task += gridDim.x) {
            const int64_t task =
                (logical_task + task_shift) % total_tasks;
            const int output_row0 =
                static_cast<int>(task % row_blocks) *
                    rows_per_task;
            const int tile = static_cast<int>(task / row_blocks);
            const int expert = tile_experts[tile];
            const int local_expert = expert_local[expert];
            if (local_expert < 0) {
                continue;
            }
            const int local_tile = tile - tile_bounds[expert];
            const int first =
                expert_bounds[expert] + local_tile * route_tile;
            const int last = min(
                first + route_tile, expert_bounds[expert + 1]);
            for (int compact = first + static_cast<int>(threadIdx.y);
                 compact < last;
                 compact += warps_per_block) {
                const int pair = ids_dst[compact];
                const int source_row = routed_input
                    ? pair
                    : pair / routes;
                nint_matmul_routed_pair(
                    bitstream, row_q_bits, row_q_bit_offsets,
                    subgroup_scale, subgroup_minimum, neuron_scale,
                    neuron_minimum, activation, activation_scale, output,
                    pair, source_row, local_expert, output_row0,
                    output_rows, groups, padded_width, group_size,
                    q_expert_stride, epilogue_mode);
            }
        }
        return;
    }

    if (route_ids != nullptr) {
        const int token = static_cast<int>(blockIdx.z) * warps_per_block +
            static_cast<int>(threadIdx.y);
        const int route = static_cast<int>(blockIdx.y);
        const int rows_per_task = epilogue_mode == 0
            ? routed_rows_per_warp
            : 1;
        const int output_row0 =
            static_cast<int>(blockIdx.x) * rows_per_task;
        if (token >= activation_rows || route >= routes) {
            return;
        }
        const int pair = token * routes + route;
        const int expert = route_ids[pair];
        if (static_cast<unsigned int>(expert) >=
                static_cast<unsigned int>(experts)) {
            return;
        }
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            return;
        }
        const int source_row = routed_input ? pair : token;
        nint_matmul_routed_pair(
            bitstream, row_q_bits, row_q_bit_offsets, subgroup_scale,
            subgroup_minimum, neuron_scale, neuron_minimum, activation,
            activation_scale, output, pair, source_row, local_expert,
            output_row0, output_rows, groups, padded_width, group_size,
            q_expert_stride, epilogue_mode);
        return;
    }

    }
    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    if (output_row >= output_rows) {
        return;
    }

    const int bits = static_cast<int>(row_q_bits[output_row]);
    const uint64_t row_bit_offset =
        static_cast<uint64_t>(row_q_bit_offsets[output_row]);
    const uint8_t * scale_row = subgroup_scale +
        static_cast<size_t>(output_row) * groups;
    const uint8_t * minimum_row = subgroup_minimum +
        static_cast<size_t>(output_row) * groups;
    const float outer_scale = neuron_scale[output_row];
    const float outer_minimum = neuron_minimum[output_row];
    float accumulators[maximum_activation_rows];
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
        accumulators[row] = 0.0f;
    }

    if constexpr(Single && Grouped) {
        accumulators[0]=nint_group_partial(bitstream,bits,row_bit_offset,
            scale_row,minimum_row,outer_scale,outer_minimum,activation,
            activation_scale,groups,padded_width,group_size,packed_bytes,lane);
    } else {
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int element = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const bool full_chunk = active_lane && element + 3 < group_size;
    const bool tail_chunk =
        active_lane && element < group_size && !full_chunk;
    const uint64_t end_word=((row_bit_offset+uint64_t(padded_width)*uint64_t(bits))>>3)&~uint64_t(3);
    const uint32_t row_bit_remainder=uint32_t(row_bit_offset&31u);
    const bool relative_fits=uint64_t(padded_width)*uint64_t(bits)+row_bit_remainder<=0xffffffffu;
    const auto* row_stream=bitstream+((row_bit_offset>>3)&~uint64_t(3));
    const bool complete_words=Single && full_chunk && relative_fits && end_word<=packed_bytes && packed_bytes-end_word>=8 && (group_size&3)==0 && (reinterpret_cast<uintptr_t>(activation)&3u)==0 && bits<8;
    if(complete_words) {
    for (int group = active_lane ? relative_group : groups,
             column = relative_group * group_size + element;
         group < groups;
         group += groups_per_warp, column += groups_per_warp * group_size) {
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        if (full_chunk) {
            const uint32_t bit_offset=row_bit_remainder+uint32_t(column)*uint32_t(bits);
            const int weight_codes=unpack_nint_codes4_unchecked(row_stream,bit_offset,bits,packed_bytes);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = *reinterpret_cast<const int*>(activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }
    } else {
    for (int group = active_lane ? relative_group : groups,
             column = relative_group * group_size + element;
         group < groups;
         group += groups_per_warp, column += groups_per_warp * group_size) {
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        if (full_chunk) {
            const uint64_t bit_offset = row_bit_offset +
                static_cast<uint64_t>(column) * static_cast<uint64_t>(bits);
            const int weight_codes = Single
                ? unpack_nint_codes4_words(bitstream,bit_offset,bits,packed_bytes)
                : unpack_nint_codes4(bitstream, bit_offset, bits);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = load_i8x4(
                        activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = bits == 8
                        ? __dp4a(
                              weight_codes ^ static_cast<int>(0x80808080u),
                              activation_codes,
                              0) + 128 * activation_sum
                        : __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        } else if (tail_chunk) {
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    int dot = 0;
                    int activation_sum = 0;
#pragma unroll
                    for (int component = 0; component < 4; ++component) {
                        if (element + component < group_size) {
                            const int activation_code = static_cast<int>(
                                activation[
                                    static_cast<size_t>(row) * padded_width +
                                    column + component]);
                            const int weight_code = static_cast<int>(
                                unpack_nint_code(
                                    bitstream,
                                    row_bit_offset,
                                    column + component,
                                    bits));
                            dot += weight_code * activation_code;
                            activation_sum += activation_code;
                        }
                    }
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }
    }
    }

#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            accumulators[row] += __shfl_xor_sync(
                0xffffffffu, accumulators[row], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (row < activation_rows) {
                output[static_cast<size_t>(row) * output_rows + output_row] =
                    __float2half_rn(accumulators[row]);
            }
        }
    }
}


// Reuse prepared activations across output warps for the one-row path.
// Packed row metadata and the original FP32 accumulation remain unchanged.
template<bool Grouped=false>
__global__ void __launch_bounds__(1024) nint_matmul_activation_cache_kernel(
    const uint8_t* __restrict__ bitstream,
    const uint8_t* __restrict__ row_q_bits,
    const int64_t* __restrict__ row_q_bit_offsets,
    const uint8_t* __restrict__ subgroup_scale,
    const uint8_t* __restrict__ subgroup_minimum,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_minimum,
    const int8_t* __restrict__ prepared_activation,
    const float* __restrict__ prepared_scale,
    half* __restrict__ output,
    int padded_width, int groups, int group_size, int output_rows,
    uint64_t packed_bytes) {
    const int warps_per_block = int(blockDim.y);
    const int lane = int(threadIdx.x), warp = int(threadIdx.y);
    const int thread = warp * 32 + lane;
    extern __shared__ unsigned char scratch[];
    int8_t* activation = reinterpret_cast<int8_t*>(scratch);
    float* activation_scale = reinterpret_cast<float*>(scratch + ((padded_width + 3) & ~3));
    for (int column = thread * 4; column < padded_width; column += warps_per_block * 128) {
        if (column + 4 <= padded_width) {
            *reinterpret_cast<int*>(activation + column) = load_i8x4(prepared_activation + column);
        } else {
            for (int tail = column; tail < padded_width; ++tail)
                activation[tail] = prepared_activation[tail];
        }
    }
    for (int group = thread; group < groups; group += warps_per_block * 32)
        activation_scale[group] = prepared_scale[group];
    __syncthreads();
    constexpr bool Single = true;
    constexpr int maximum_activation_rows = 1;
    constexpr int activation_rows = 1;
    const int chunks = (group_size + 3) / 4;
    const int groups_per_warp = 32 / chunks;
    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    if (output_row >= output_rows) {
        return;
    }

    const int bits = static_cast<int>(row_q_bits[output_row]);
    const uint64_t row_bit_offset =
        static_cast<uint64_t>(row_q_bit_offsets[output_row]);
    const uint8_t * scale_row = subgroup_scale +
        static_cast<size_t>(output_row) * groups;
    const uint8_t * minimum_row = subgroup_minimum +
        static_cast<size_t>(output_row) * groups;
    const float outer_scale = neuron_scale[output_row];
    const float outer_minimum = neuron_minimum[output_row];
    float accumulators[maximum_activation_rows];
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
        accumulators[row] = 0.0f;
    }

    if constexpr(Grouped) {
        accumulators[0]=nint_group_partial(bitstream,bits,row_bit_offset,
            scale_row,minimum_row,outer_scale,outer_minimum,activation,
            activation_scale,groups,padded_width,group_size,packed_bytes,lane);
    } else {
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int element = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const bool full_chunk = active_lane && element + 3 < group_size;
    const bool tail_chunk =
        active_lane && element < group_size && !full_chunk;
    const uint64_t end_word=((row_bit_offset+uint64_t(padded_width)*uint64_t(bits))>>3)&~uint64_t(3);
    const uint32_t row_bit_remainder=uint32_t(row_bit_offset&31u);
    const bool relative_fits=uint64_t(padded_width)*uint64_t(bits)+row_bit_remainder<=0xffffffffu;
    const auto* row_stream=bitstream+((row_bit_offset>>3)&~uint64_t(3));
    const bool complete_words=Single && full_chunk && relative_fits && end_word<=packed_bytes && packed_bytes-end_word>=8 && (group_size&3)==0 && (reinterpret_cast<uintptr_t>(activation)&3u)==0 && bits<8;
    if(complete_words) {
    for (int group_base = 0;
         group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        const int column = group * group_size + element;
        if (full_chunk) {
            const uint32_t bit_offset=row_bit_remainder+uint32_t(column)*uint32_t(bits);
            const int weight_codes=(bits == 8 && (bit_offset & 31u) == 0) ?
                *reinterpret_cast<const int*>(row_stream + (bit_offset >> 3)) :
                unpack_nint_codes4_unchecked(row_stream,bit_offset,bits,packed_bytes);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = *reinterpret_cast<const int*>(activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }
    } else {
    for (int group_base = 0;
         group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        const int column = group * group_size + element;
        if (full_chunk) {
            const uint64_t bit_offset = row_bit_offset +
                static_cast<uint64_t>(column) * static_cast<uint64_t>(bits);
            const int weight_codes = Single
                ? ((bits == 8 && (bit_offset & 31u) == 0 && (bit_offset >> 3) + 4 <= packed_bytes) ?
                    *reinterpret_cast<const int*>(bitstream + (bit_offset >> 3)) :
                    unpack_nint_codes4_words(bitstream,bit_offset,bits,packed_bytes))
                : unpack_nint_codes4(bitstream, bit_offset, bits);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = load_i8x4(
                        activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = bits == 8
                        ? __dp4a(
                              weight_codes ^ static_cast<int>(0x80808080u),
                              activation_codes,
                              0) + 128 * activation_sum
                        : __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        } else if (tail_chunk) {
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    int dot = 0;
                    int activation_sum = 0;
#pragma unroll
                    for (int component = 0; component < 4; ++component) {
                        if (element + component < group_size) {
                            const int activation_code = static_cast<int>(
                                activation[
                                    static_cast<size_t>(row) * padded_width +
                                    column + component]);
                            const int weight_code = static_cast<int>(
                                unpack_nint_code(
                                    bitstream,
                                    row_bit_offset,
                                    column + component,
                                    bits));
                            dot += weight_code * activation_code;
                            activation_sum += activation_code;
                        }
                    }
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }
    }
    }

#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            accumulators[row] += __shfl_xor_sync(
                0xffffffffu, accumulators[row], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (row < activation_rows) {
                output[static_cast<size_t>(row) * output_rows + output_row] =
                    __float2half_rn(accumulators[row]);
            }
        }
    }
}


// The caller has proven q8
// row widths, aligned offsets, four-element groups and aligned storage.
#include "nint_q8_ordered.cuh"

template<int maximum_activation_rows,int FixedGroup=0>
__global__ void __launch_bounds__(128) nint_matmul_aligned_q8_kernel(
        const uint8_t * __restrict__ bitstream,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int activation_rows, int output_rows, int groups,
        int padded_width, int runtime_group_size) {
    const int group_size=FixedGroup?FixedGroup:runtime_group_size;
    const int output_row = static_cast<int>(blockIdx.x) * 4 +
        static_cast<int>(threadIdx.y);
    if (output_row >= output_rows) {
        return;
    }
    const int chunks = group_size / 4;
    const int groups_per_warp = 32 / chunks;
    const int lane = static_cast<int>(threadIdx.x);
    const int relative_group = lane / chunks;
    const int element = (lane - relative_group * chunks) * 4;
    const uint8_t * weight_row = bitstream +
        (static_cast<uint64_t>(row_q_bit_offsets[output_row]) >> 3);
    const uint8_t * scale_row = subgroup_scale +
        static_cast<size_t>(output_row) * groups;
    const uint8_t * minimum_row = subgroup_minimum +
        static_cast<size_t>(output_row) * groups;
    const float outer_scale = neuron_scale[output_row];
    const float outer_minimum = neuron_minimum[output_row];
    float accumulators[maximum_activation_rows];
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
        accumulators[row] = 0.0f;
    }
    for (int group_base = 0; group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (relative_group >= groups_per_warp || group >= groups) {
            continue;
        }
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        const int column = group * group_size + element;
        const int weight_codes = static_cast<int>(
            *reinterpret_cast<const uint32_t *>(weight_row + column));
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (maximum_activation_rows == 1 || row < activation_rows) {
                const int activation_codes = load_i8x4(
                    activation + static_cast<size_t>(row) * padded_width +
                    column);
                const int activation_sum = __dp4a(
                    0x01010101, activation_codes, 0);
                const int dot = __dp4a(
                    weight_codes ^ static_cast<int>(0x80808080u),
                    activation_codes, 0) + 128 * activation_sum;
                const float input_scale = activation_scale[
                    static_cast<size_t>(row) * groups + group];
                accumulators[row] += input_scale * (
                    outer_scale * inner_scale * static_cast<float>(dot) -
                    outer_minimum * inner_minimum *
                        static_cast<float>(activation_sum));
            }
        }
    }
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            accumulators[row] += __shfl_xor_sync(
                0xffffffffu, accumulators[row], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (maximum_activation_rows == 1 || row < activation_rows) {
                output[static_cast<size_t>(row) * output_rows + output_row] =
                    __float2half_rn(accumulators[row]);
            }
        }
    }
}


// One metadata-driven packed input-gradient kernel covers every NINT q/k
// allocation.  Each warp owns four adjacent input columns and reduces over
// output neurons; q is read from row metadata and k is already represented by
// the baked subgroup metadata.
__global__ void __launch_bounds__(128) nint_backward_input_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const __half * __restrict__ output_gradient,
        __half * __restrict__ input_gradient,
        int activation_rows,
        int output_rows,
        int groups,
        int width,
        int group_size) {
    constexpr int warps_per_block = 4;
    constexpr int columns_per_warp = 4;
    constexpr int rows_per_tile = 4;
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int chunks_per_group = (group_size + columns_per_warp - 1) /
        columns_per_warp;
    const int chunk_index = static_cast<int>(blockIdx.x) * warps_per_block +
        warp;
    const int group = chunk_index / chunks_per_group;
    const int group_element =
        (chunk_index - group * chunks_per_group) * columns_per_warp;
    const int column_base = group * group_size + group_element;
    const int first_activation_row =
        static_cast<int>(blockIdx.y) * rows_per_tile;
    if (group >= groups || column_base >= width ||
            first_activation_row >= activation_rows) {
        return;
    }

    float accumulators[rows_per_tile][columns_per_warp] = {};
    for (int output_row = lane; output_row < output_rows; output_row += 32) {
        const int bits = static_cast<int>(row_q_bits[output_row]);
        const uint64_t row_bit_offset = static_cast<uint64_t>(
            row_q_bit_offsets[output_row]);
        const size_t metadata_index =
            static_cast<size_t>(output_row) * groups + group;
        const float scale = neuron_scale[output_row] *
            static_cast<float>(subgroup_scale[metadata_index]);
        const float minimum = neuron_minimum[output_row] *
            static_cast<float>(subgroup_minimum[metadata_index]);
        const int valid_columns = min(
            columns_per_warp,
            min(group_size - group_element, width - column_base));
        uint32_t packed_codes = 0;
        if (valid_columns == columns_per_warp) {
            packed_codes = static_cast<uint32_t>(
                unpack_nint_codes4(
                    bitstream,
                    row_bit_offset +
                        static_cast<uint64_t>(column_base) * bits,
                    bits));
        } else {
#pragma unroll
            for (int component = 0; component < columns_per_warp; ++component) {
                if (component < valid_columns) {
                    packed_codes |= static_cast<uint32_t>(unpack_nint_code(
                        bitstream,
                        row_bit_offset,
                        column_base + component,
                        bits)) << (8 * component);
                }
            }
        }
        float weights[columns_per_warp];
#pragma unroll
        for (int component = 0; component < columns_per_warp; ++component) {
            const int code = (packed_codes >> (8 * component)) & 255;
            weights[component] = scale * static_cast<float>(code) - minimum;
        }
#pragma unroll
        for (int local_row = 0; local_row < rows_per_tile; ++local_row) {
            const int activation_row = first_activation_row + local_row;
            const float gradient = activation_row < activation_rows
                ? __half2float(output_gradient[
                      static_cast<size_t>(activation_row) * output_rows +
                      output_row])
                : 0.0f;
#pragma unroll
            for (int component = 0; component < columns_per_warp; ++component) {
                accumulators[local_row][component] = fmaf(
                    gradient,
                    weights[component],
                    accumulators[local_row][component]);
            }
        }
    }

#pragma unroll
    for (int local_row = 0; local_row < rows_per_tile; ++local_row) {
#pragma unroll
        for (int component = 0; component < columns_per_warp; ++component) {
            float total = accumulators[local_row][component];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xffffffffu, total, offset);
            }
            const int activation_row = first_activation_row + local_row;
            const int column = column_base + component;
            if (lane == 0 && activation_row < activation_rows &&
                    component < group_size - group_element && column < width) {
                input_gradient[
                    static_cast<size_t>(activation_row) * width + column] =
                    __float2half_rn(total);
            }
        }
    }
}


template <typename Output>
__global__ void nint8_zero_decode_rows_kernel(
        const int8_t * __restrict__ quantized,
        const __half * __restrict__ scale,
        Output * __restrict__ output,
        int rows,
        int groups,
        int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / width);
        const int column = static_cast<int>(index % width);
        const int group = column / 32;
        const int lane = column & 31;
        const size_t block = static_cast<size_t>(row) * groups + group;
        output[index] = nint_decode_store<Output>(
            __half2float(scale[block]) * static_cast<float>(
                quantized[block * 32 + lane]));
    }
}


// NINT8-0 has one packed small-M compute kernel; M is a grid dimension rather
// than a template parameter, so it does not create one binary per M.
__global__ void __launch_bounds__(128) nint8_zero_matmul_kernel(
        const int8_t * __restrict__ weight,
        const __half * __restrict__ weight_scale,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int activation_rows,
        int output_rows,
        int groups,
        int padded_width) {
    constexpr int warps_per_block = 4;
    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    const int activation_row = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (output_row >= output_rows || activation_row >= activation_rows) {
        return;
    }
    float accumulator = 0.0f;
    for (int base = lane * 4; base < padded_width; base += 32 * 4) {
        const int group = base / 32;
        const int offset = base & 31;
        const size_t block = static_cast<size_t>(output_row) * groups + group;
        const int weight_codes = *reinterpret_cast<const int *>(
            weight + block * 32 + offset);
        const int activation_codes = *reinterpret_cast<const int *>(
            activation + static_cast<size_t>(activation_row) * padded_width +
            base);
        accumulator += __half2float(weight_scale[block]) *
            activation_scale[
                static_cast<size_t>(activation_row) * groups + group] *
            static_cast<float>(__dp4a(weight_codes, activation_codes, 0));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(
            0xffffffffu, accumulator, offset);
    }
    if (lane == 0) {
        output[static_cast<size_t>(activation_row) * output_rows +
            output_row] = __float2half_rn(accumulator);
    }
}


void validate_nint8_zero(
        const mfq_tensor_backend::Tensor & quantized,
        const mfq_tensor_backend::Tensor & scale) {
    MFQ_RUNTIME_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == mfq_tensor_backend::kUInt8 &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 q must be contiguous CUDA uint8 [N,G,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == mfq_tensor_backend::kFloat16 &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 scale must be contiguous CUDA fp16 [N,G]");
    MFQ_RUNTIME_CHECK(
        quantized.device() == scale.device(),
        "NINT8-0 tensors must share one CUDA device");
}


mfq_tensor_backend::Tensor cublas_gemm_nt_f32_output(
        const mfq_tensor_backend::Tensor & input,
        const mfq_tensor_backend::Tensor & weight) {
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        input.options().dtype(mfq_tensor_backend::kFloat32));
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, mfq_current_cuda_stream()));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputs,
        rows,
        width,
        &alpha,
        weight.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        input.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        &beta,
        result.data_ptr<float>(),
        CUDA_R_32F,
        outputs,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return result;
}


}  // namespace

NintInputQuantizationScope::NintInputQuantizationScope(
        mfq_tensor_backend::Tensor source,const std::vector<int>& sizes):input(std::move(source)) {
    if(sizes.empty())return;
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.dim()==2 &&
        input.size(0)>=1 && input.size(0)<=6 && input.size(1)>0 && input.size(1)<=INT_MAX-64 &&
        (input.scalar_type()==tb::kFloat16 || input.scalar_type()==tb::kFloat32) &&
        sizes.size()<=4,"invalid shared NINT quantization input");
    const MfqCudaGuard guard(input.device());
    NintGroupQuantizeArgs args;
    args.input=input.data_ptr();args.rows=int(input.size(0));args.width=int(input.size(1));
    args.fp32=input.scalar_type()==tb::kFloat32;
    const auto* exact=std::getenv("MFQ_NINT_EXACT_BATCH");
    const int columns=args.rows<=2?2:args.rows<=4?4:(exact && exact[0]=='0')?8:args.rows;
    int total=0;
    for(int gs:sizes) {
        MFQ_RUNTIME_CHECK(gs==24 || gs==28 || gs==48,"unsupported shared NINT group size");
        const int ng=(args.width+gs-1)/gs;
        NintPreparedGroupInput prepared;prepared.gs=gs;
        prepared.codes=tb::empty({args.rows,ng*gs},input.options().dtype(tb::kInt8));
        prepared.scale=tb::empty({args.rows,ng},input.options().dtype(tb::kFloat32));
        prepared.sum=tb::empty({args.rows,ng},input.options().dtype(tb::kInt32));
        if(gs==28 && args.rows>=2)
            prepared.interleaved=tb::empty({ng,7,columns,4},input.options().dtype(tb::kInt8));
        args.group[args.count++]={prepared.codes.data_ptr<int8_t>(),
            prepared.interleaved.defined()?prepared.interleaved.data_ptr<int8_t>():nullptr,
            prepared.scale.data_ptr<float>(),prepared.sum.data_ptr<int32_t>(),gs,ng,total,columns};
        total+=ng;groups.push_back(std::move(prepared));
    }
    nint_group_quantize_kernel<<<dim3(total,args.rows),32,0,mfq_current_cuda_stream()>>>(args);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(
        mfq_current_cuda_stream(),"nint_group_quantize",reinterpret_cast<const void*>(nint_group_quantize_kernel),
        "M="+std::to_string(args.rows)+" width="+std::to_string(args.width)+" formats="+std::to_string(args.count));
    previous=prepared_input_scope;prepared_input_scope=this;
}
NintInputQuantizationScope::~NintInputQuantizationScope() {
    if(!groups.empty())prepared_input_scope=previous;
}

bool nint_late_scale_enabled(int64_t input_width,int bits,int /*device*/) {
    if(const auto* setting=std::getenv("MFQ_NINT_LATE_SCALE"))return setting[0]!='0';
    if(input_width<2048 || bits<5 || bits>6)return false;
    return true;
}

static void nint_moe_grouped_matmul_dispatch_cuda(
        mfq_tensor_backend::Tensor pointers,mfq_tensor_backend::Tensor params,
        mfq_tensor_backend::Tensor expert_pool,mfq_tensor_backend::Tensor expert_local,
        mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor output,
        int64_t input_width,bool routed_input,int group_hint,int bits_hint) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(pointers.is_cuda() && pointers.is_contiguous() && pointers.scalar_type()==tb::kInt64 &&
        pointers.dim()==2 && pointers.size(1)==9 && pointers.size(0)>0 &&
        params.is_cuda() && params.is_contiguous() && params.scalar_type()==tb::kInt32 &&
        params.dim()==2 && params.size(0)==pointers.size(0) && params.size(1)==3,
        "mixed NINT pointer/geometry table disagrees");
    MFQ_RUNTIME_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type()==tb::kInt32 && ids.dim()==2 &&
        ids.size(0)>0 && ids.size(0)<=8 && ids.size(1)>0 && ids.numel()<=65535 && input_width>0 &&
        expert_pool.is_cuda() && expert_pool.is_contiguous() && expert_pool.scalar_type()==tb::kInt32 &&
        expert_pool.dim()==1 && expert_pool.numel()>0 && expert_pool.numel()<=4096 &&
        expert_local.is_cuda() && expert_local.is_contiguous() && expert_local.scalar_type()==tb::kInt32 &&
        expert_local.sizes()==expert_pool.sizes() && output.is_cuda() && output.is_contiguous() &&
        output.scalar_type()==tb::kFloat16 && output.dim()==3 && output.size(0)==ids.size(0) &&
        output.size(1)==ids.size(1) && output.size(2)>0 && output.size(2)<=INT_MAX &&
        pointers.device()==output.device() && params.device()==output.device() && ids.device()==output.device() &&
        expert_pool.device()==output.device() && expert_local.device()==output.device(),
        "mixed NINT route/output geometry or device disagrees");
    MfqCudaGuard guard(output.device());
    const auto launch=[&](auto lanes,auto blocks) {
    nint_mixed_routes_kernel<decltype(lanes)::value,decltype(blocks)::value><<<dim3(unsigned((output.size(2)+7)/8),unsigned(ids.numel())),dim3(32,4),0,
        mfq_current_cuda_stream()>>>(pointers.data_ptr<int64_t>(),params.data_ptr<int32_t>(),
        expert_pool.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr()),int(ids.numel()),int(ids.size(1)),int(expert_pool.numel()),
        int(output.size(2)),routed_input,mfq::cuda::warp_multi_sum::enabled());
    };
    const char* grouped=std::getenv("MFQ_NINT_GROUP_DOT");
    const char* lanes=std::getenv("MFQ_NINT_ROUTE_LANES");
    const char* blocks=std::getenv("MFQ_NINT_ROUTE_BLOCKS");
    const char* rows=std::getenv("MFQ_NINT_ROUTE_ROWS");
    const char* hint=std::getenv("MFQ_NINT_ROUTE_HINT");
    if((!grouped || grouped[0]!='0') && (!hint || hint[0]!='0') &&
            (group_hint==24 || group_hint==28) && bits_hint>=4 && bits_hint<=6) {
        const auto hinted_warps=[&](auto gs,auto bits,auto warps) {
            static constexpr int rows_per_block=decltype(warps)::value==4?4:8;
            const auto launch_scale=[&](auto late) {
            nint_mixed_routes_hinted_kernel<decltype(gs)::value,decltype(bits)::value,decltype(warps)::value,false,decltype(late)::value>
                <<<dim3(unsigned((output.size(2)+rows_per_block-1)/rows_per_block),unsigned(ids.numel())),dim3(32,4),0,mfq_current_cuda_stream()>>>(
                    pointers.data_ptr<int64_t>(),params.data_ptr<int32_t>(),expert_pool.data_ptr<int32_t>(),
                    expert_local.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),reinterpret_cast<__half*>(output.data_ptr()),
                    int(ids.numel()),int(ids.size(1)),int(expert_pool.numel()),int(output.size(2)),routed_input,{},mfq::cuda::warp_multi_sum::enabled());
            };
            if(nint_late_scale_enabled(input_width,bits_hint,output.get_device()))
                launch_scale(std::true_type{});
            else launch_scale(std::false_type{});
        };
        const auto hinted=[&](auto gs,auto bits) {
            const char* warps=std::getenv("MFQ_NINT_ROUTE_WARPS");
            if(warps && std::atoi(warps)==4)hinted_warps(gs,bits,std::integral_constant<int,4>{});
            else hinted_warps(gs,bits,std::integral_constant<int,1>{});
        };
        const auto select_bits=[&](auto gs) {
            if(bits_hint==4)hinted(gs,std::integral_constant<int,4>{});
            else if(bits_hint==5)hinted(gs,std::integral_constant<int,5>{});
            else hinted(gs,std::integral_constant<int,6>{});
        };
        if(group_hint==24)select_bits(std::integral_constant<int,24>{});
        else select_bits(std::integral_constant<int,28>{});
    }
    else if(rows && std::atoi(rows)==1 && (!grouped || grouped[0]!='0')) {
        nint_mixed_routes_kernel<32,1,1><<<dim3(unsigned((output.size(2)+3)/4),unsigned(ids.numel())),dim3(32,4),0,
            mfq_current_cuda_stream()>>>(pointers.data_ptr<int64_t>(),params.data_ptr<int32_t>(),
            expert_pool.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(output.data_ptr()),int(ids.numel()),int(ids.size(1)),int(expert_pool.numel()),
            int(output.size(2)),routed_input,mfq::cuda::warp_multi_sum::enabled());
    }
    else if(grouped && grouped[0]=='0')launch(std::integral_constant<int,0>{},std::integral_constant<int,1>{});
    else if(lanes && std::atoi(lanes)==16)launch(std::integral_constant<int,16>{},std::integral_constant<int,1>{});
    // Short reductions gain from more resident warps. Long reductions keep
    // more registers for independent packed loads and instruction overlap.
    else if((blocks && std::atoi(blocks)==8) || (!blocks && input_width<=768 && output.size(2)>=1024))
        launch(std::integral_constant<int,32>{},std::integral_constant<int,8>{});
    else launch(std::integral_constant<int,32>{},std::integral_constant<int,1>{});
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

void nint_moe_grouped_matmul_hetero_cuda(
        mfq_tensor_backend::Tensor pointers,mfq_tensor_backend::Tensor params,
        mfq_tensor_backend::Tensor expert_pool,mfq_tensor_backend::Tensor expert_local,
        mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor output,
        int64_t input_width,bool routed_input) {
    nint_moe_grouped_matmul_dispatch_cuda(pointers,params,expert_pool,expert_local,
        ids,output,input_width,routed_input,0,0);
}

void nint_moe_grouped_matmul_hinted_cuda(
        mfq_tensor_backend::Tensor pointers,mfq_tensor_backend::Tensor params,
        mfq_tensor_backend::Tensor expert_pool,mfq_tensor_backend::Tensor expert_local,
        mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor output,
        int64_t input_width,bool routed_input,int group_hint,int bits_hint) {
    nint_moe_grouped_matmul_dispatch_cuda(pointers,params,expert_pool,expert_local,
        ids,output,input_width,routed_input,group_hint,bits_hint);
}

bool nint_try_single_route_cuda(
        NintSingleRouteWeight weight,mfq_tensor_backend::Tensor expert_local,
        mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor output,
        bool routed_input,int bits_hint,bool metadata_ready) {
    namespace tb=mfq_tensor_backend;
    const char* setting=std::getenv("MFQ_NINT_DIRECT_POOL");
    const char* grouped=std::getenv("MFQ_NINT_GROUP_DOT");
    const char* hint=std::getenv("MFQ_NINT_ROUTE_HINT");
    const char* warps=std::getenv("MFQ_NINT_ROUTE_WARPS");
    if((setting && setting[0]=='0') || (grouped && grouped[0]=='0') || (hint && hint[0]=='0') ||
            (warps && std::atoi(warps)==4) || bits_hint<4 || bits_hint>6 ||
            (weight.geometry[1]!=24 && weight.geometry[1]!=28))return false;
    MFQ_RUNTIME_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() && expert_local.scalar_type()==tb::kInt32 &&
        expert_local.dim()==1 && expert_local.numel()>0 && expert_local.numel()<=4096 &&
        ids.is_cuda() && ids.is_contiguous() && ids.scalar_type()==tb::kInt32 && ids.dim()==2 &&
        ids.size(0)>0 && ids.size(0)<=8 && ids.size(1)>0 && ids.numel()<=65535 &&
        output.is_cuda() && output.is_contiguous() && output.scalar_type()==tb::kFloat16 && output.dim()==3 &&
        output.size(0)==ids.size(0) && output.size(1)==ids.size(1) && output.size(2)>0 && output.size(2)<=INT_MAX &&
        output.device()==ids.device() && output.device()==expert_local.device() &&
        weight.local_experts>0 && weight.geometry[0]>0 && weight.geometry[2]>0,
        "single-pool NINT routed tensors disagree");
    MfqCudaGuard guard(output.device());
    const auto launch=[&](auto gs,auto bits) {
        const auto launch_scale=[&](auto late) {
        const auto launch_metadata=[&](auto packed) {
        nint_mixed_routes_hinted_kernel<decltype(gs)::value,decltype(bits)::value,1,true,decltype(late)::value,decltype(packed)::value>
            <<<dim3(unsigned((output.size(2)+7)/8),unsigned(ids.numel())),dim3(32,4),0,mfq_current_cuda_stream()>>>(
                nullptr,nullptr,nullptr,expert_local.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),
                reinterpret_cast<__half*>(output.data_ptr()),int(ids.numel()),int(ids.size(1)),
                int(expert_local.numel()),int(output.size(2)),routed_input,weight,mfq::cuda::warp_multi_sum::enabled());
        };
        const char* metadata=std::getenv("MFQ_NINT_ROUTE_METADATA");
        if(weight.row_metadata && (metadata_ready || (metadata && metadata[0]!='0')))
            launch_metadata(std::true_type{});
        else launch_metadata(std::false_type{});
        };
        if(nint_late_scale_enabled(int64_t(weight.geometry[0])*weight.geometry[1],bits_hint,output.get_device()))
            launch_scale(std::true_type{});
        else launch_scale(std::false_type{});
    };
    const auto select_bits=[&](auto gs) {
        if(bits_hint==4)launch(gs,std::integral_constant<int,4>{});
        else if(bits_hint==5)launch(gs,std::integral_constant<int,5>{});
        else launch(gs,std::integral_constant<int,6>{});
    };
    if(weight.geometry[1]==24)select_bits(std::integral_constant<int,24>{});
    else select_bits(std::integral_constant<int,28>{});
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}


void launch_nint_matmul_routed_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor route_ids,
        mfq_tensor_backend::Tensor expert_local,
        bool route_map_ready,
        mfq_tensor_backend::Tensor ids_dst,
        mfq_tensor_backend::Tensor expert_bounds,
        mfq_tensor_backend::Tensor tile_bounds,
        mfq_tensor_backend::Tensor tile_experts,
        mfq_tensor_backend::Tensor output,
        int tokens,
        int routes,
        int experts,
        int output_rows,
        int groups,
        int padded_width,
        int input_width,
        int group_size,
        int q_expert_stride,
        int route_tile_m,
        int pool_phase,
        bool masked_experts,
        int epilogue_mode,
        bool routed_input,
        cudaStream_t stream) {
    constexpr int warps_per_block = 4;
    constexpr int rows_per_warp = 2;
    const int rows_per_task = epilogue_mode == 0 ? rows_per_warp : 1;
    const int result_rows = epilogue_mode == 0
        ? output_rows
        : output_rows / 2;
    const int row_blocks =
        (result_rows + rows_per_task - 1) / rows_per_task;
    const bool use_compact = tokens > 8 && route_map_ready;
    const bool use_tiled_prefill = use_compact &&
        (route_tile_m == 16 || route_tile_m == 32 ||
         route_tile_m == 64 || route_tile_m == 128);
    if (use_tiled_prefill) {
        constexpr int output_tile = kNintPrefillTileN;
        const int output_tiles =
            (result_rows + output_tile - 1) / output_tile;
        const int pairs = tokens * routes;
        const int64_t maximum_tiles =
            (pairs + route_tile_m - 1) / route_tile_m + experts;
        const int64_t maximum_tasks = maximum_tiles * output_tiles;
        int block_cap = pairs >= 32768 ? 8192 : 4096;
        const int scaled_block_cap = static_cast<int>(
            std::min<int64_t>(INT_MAX, (maximum_tasks + 3) / 4));
        block_cap = std::max(block_cap, scaled_block_cap);
        if (masked_experts && maximum_tasks > block_cap) {
            const int routed_rows_per_expert = std::max(
                1, (pairs + experts - 1) / experts);
            const int task_period =
                ((routed_rows_per_expert + route_tile_m - 1) /
                    route_tile_m) * output_tiles;
            if (task_period > 0 && block_cap % task_period == 0) {
                block_cap += task_period;
            }
        }
        const int blocks = static_cast<int>(std::max<int64_t>(
            1, std::min<int64_t>(maximum_tasks, block_cap)));
#define MFQ_LAUNCH_NINT_TILED_PREFILL(TILE_M, FUSED_GLU) \
        nint_matmul_tiled_prefill_kernel<TILE_M, true, FUSED_GLU><<< \
            blocks, dim3(32, kNintPrefillWarps), 0, stream>>>( \
                bitstream.data_ptr<uint8_t>(), \
                row_q_bits.data_ptr<uint8_t>(), \
                row_q_bit_offsets.data_ptr<int64_t>(), \
                subgroup_scale.data_ptr<uint8_t>(), \
                subgroup_minimum.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), \
                neuron_minimum.data_ptr<float>(), \
                reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()), \
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), \
                expert_local.data_ptr<int32_t>(), \
                ids_dst.data_ptr<int32_t>(), \
                expert_bounds.data_ptr<int32_t>(), \
                tile_bounds.data_ptr<int32_t>(), \
                tile_experts.data_ptr<int32_t>(), \
                routes, experts, output_rows, groups, group_size, \
                input_width, q_expert_stride, pool_phase, epilogue_mode, \
                routed_input, 0, 1, nullptr)
        if (route_tile_m == 128 && epilogue_mode == 0) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(128, false);
        } else if (route_tile_m == 128) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(128, true);
        } else if (route_tile_m == 32 && epilogue_mode == 0) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(32, false);
        } else if (route_tile_m == 32) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(32, true);
        } else if (route_tile_m == 16 && epilogue_mode == 0) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(16, false);
        } else if (route_tile_m == 16) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(16, true);
        } else if (epilogue_mode == 0) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(64, false);
        } else {
            MFQ_LAUNCH_NINT_TILED_PREFILL(64, true);
        }
#undef MFQ_LAUNCH_NINT_TILED_PREFILL
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    MFQ_RUNTIME_CHECK(
        route_tile_m == 8,
        "NINT routed tile must be 8, 16, 32, 64, or 128");
    const int token_blocks =
        (tokens + warps_per_block - 1) / warps_per_block;
    const int64_t maximum_tasks =
        static_cast<int64_t>(tokens) * routes * row_blocks;
    constexpr int persistent_blocks = 4096;
    const int compact_blocks = static_cast<int>(
        std::min<int64_t>(maximum_tasks, persistent_blocks));
    const dim3 grid = use_compact
        ? dim3(compact_blocks)
        : dim3(row_blocks, routes, token_blocks);
    nint_matmul_kernel<false><<<
        grid,
        dim3(32, warps_per_block), 0, stream>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            tokens,
            output_rows,
            groups,
            padded_width,
            group_size,
            route_ids.data_ptr<int32_t>(),
            expert_local.data_ptr<int32_t>(),
            use_compact ? ids_dst.data_ptr<int32_t>() : nullptr,
            use_compact ? expert_bounds.data_ptr<int32_t>() : nullptr,
            use_compact ? tile_bounds.data_ptr<int32_t>() : nullptr,
            use_compact ? tile_experts.data_ptr<int32_t>() : nullptr,
            routes,
            experts,
            q_expert_stride,
            pool_phase,
            epilogue_mode,
            routed_input);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}


std::vector<mfq_tensor_backend::Tensor> nint8_one_quantize_reconstruct_cuda(
        mfq_tensor_backend::Tensor input) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "NINT8-1 input must be contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    MFQ_RUNTIME_CHECK(rows > 0 && width > 0, "NINT8-1 input must be non-empty");
    const int groups = (width + 31) / 32;
    auto quantized = mfq_tensor_backend::empty(
        {rows, groups, 32},
        input.options().dtype(mfq_tensor_backend::kInt8));
    auto scale = mfq_tensor_backend::empty({rows, groups}, input.options());
    auto sum = mfq_tensor_backend::empty({rows, groups}, input.options());
    auto reconstructed = mfq_tensor_backend::empty_like(input);
    nint8_one_quantize_reconstruct_kernel<<<
        rows * groups, 32, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            quantized.data_ptr<int8_t>(),
            reinterpret_cast<__half *>(scale.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(sum.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(reconstructed.data_ptr<mfq_half>()),
            rows,
            width,
            groups);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return {quantized, scale, sum, reconstructed};
}


mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f16acc_cuda(
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor weight) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == mfq_tensor_backend::kFloat16 &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    MFQ_RUNTIME_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, mfq_current_cuda_stream()));
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputs,
        rows,
        width,
        &alpha,
        weight.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        input.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        &beta,
        result.data_ptr<mfq_half>(),
        CUDA_R_16F,
        outputs,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return result;
}


mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f32acc_cuda(
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor weight) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == mfq_tensor_backend::kFloat16 &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    MFQ_RUNTIME_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, mfq_current_cuda_stream()));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputs,
        rows,
        width,
        &alpha,
        weight.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        input.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        &beta,
        result.data_ptr<mfq_half>(),
        CUDA_R_16F,
        outputs,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return result;
}


template <typename Output>
static mfq_tensor_backend::Tensor nint_decode_impl(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1,
        "NINT bitstream must be contiguous CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(
        row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1,
        "NINT q metadata must be contiguous CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(
        row_q_bit_offsets.is_cuda() && row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT row offsets must be contiguous CUDA int64 rank-1");
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT neuron metadata is invalid");
    const int rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    MFQ_RUNTIME_CHECK(
        rows > 0 && groups > 0 && group_size >= 4 && group_size <= 64 &&
        width > 0 && width <= static_cast<int64_t>(groups) * group_size,
        "NINT decode dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == rows && row_q_bit_offsets.numel() == rows &&
        neuron_scale.numel() == rows && neuron_minimum.numel() == rows,
        "NINT row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == row_q_bits.device() &&
        bitstream.device() == row_q_bit_offsets.device() &&
        bitstream.device() == subgroup_scale.device() &&
        bitstream.device() == subgroup_minimum.device() &&
        bitstream.device() == neuron_scale.device() &&
        bitstream.device() == neuron_minimum.device(),
        "NINT tensors must share one CUDA device");
    auto output = mfq_tensor_backend::empty(
        {rows, width},
        neuron_scale.options().dtype(
            std::is_same_v<Output, __half>
                ? mfq_tensor_backend::kFloat16
                : mfq_tensor_backend::kFloat32));
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(rows) * width;
    const int blocks = static_cast<int>(std::min<size_t>(
        (total + threads - 1) / threads, 65535));
    if constexpr (std::is_same_v<Output, __half>) {
        if ((group_size == 24 || group_size == 28) && width % 4 == 0 &&
                (reinterpret_cast<std::uintptr_t>(bitstream.data_ptr()) & 3u) == 0 &&
                std::getenv("MFQ_NINT_DENSE_REFERENCE") == nullptr) {
            const auto launch = [&](auto gs) {
                nint_decode_vec4_kernel<decltype(gs)::value><<<
                    dim3((width / 4 + 127) / 128, std::min(rows, 65535)), 128, 0, mfq_current_cuda_stream()>>>(
                    bitstream.data_ptr<uint8_t>(), row_q_bits.data_ptr<uint8_t>(), row_q_bit_offsets.data_ptr<int64_t>(),
                    subgroup_scale.data_ptr<uint8_t>(), subgroup_minimum.data_ptr<uint8_t>(),
                    neuron_scale.data_ptr<float>(), neuron_minimum.data_ptr<float>(),
                    reinterpret_cast<half*>(output.template data_ptr<mfq_half>()), rows, groups, int(width), uint64_t(bitstream.numel()));
            };
            if (group_size == 24) launch(std::integral_constant<int, 24>{});
            else launch(std::integral_constant<int, 28>{});
            MFQ_CUDA_KERNEL_LAUNCH_CHECK();
            return output;
        }
    }
    const auto output_data = [&]() {
        if constexpr (std::is_same_v<Output, __half>) {
            return reinterpret_cast<__half *>(output.template data_ptr<mfq_half>());
        } else {
            return output.template data_ptr<float>();
        }
    }();
    nint_decode_rows_kernel<Output><<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            output_data,
            rows,
            groups,
            static_cast<int>(group_size),
            static_cast<int>(width));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint_decode_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    return nint_decode_impl<__half>(
        std::move(bitstream), std::move(row_q_bits),
        std::move(row_q_bit_offsets), std::move(subgroup_scale),
        std::move(subgroup_minimum), std::move(neuron_scale),
        std::move(neuron_minimum), width, group_size);
}


mfq_tensor_backend::Tensor nint_decode_f32_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    return nint_decode_impl<float>(
        std::move(bitstream), std::move(row_q_bits),
        std::move(row_q_bit_offsets), std::move(subgroup_scale),
        std::move(subgroup_minimum), std::move(neuron_scale),
        std::move(neuron_minimum), width, group_size);
}


static mfq_tensor_backend::Tensor nint_matmul_ws_impl(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor input,
        const mfq_tensor_backend::Tensor * gate,
        int64_t activation_mode,
        int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale,
        bool aligned_q8 = false,int uniform_bits = 0) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        (input.scalar_type() == mfq_tensor_backend::kFloat16 ||
         input.scalar_type() == mfq_tensor_backend::kFloat32) &&
        input.dim() == 2,
        "NINT input must be contiguous CUDA fp16/fp32 rank-2");
    if (gate != nullptr) {
        MFQ_RUNTIME_CHECK(
            activation_mode == 1 || activation_mode == 2,
            "NINT input gate mode must be sigmoid or SiLU");
        MFQ_RUNTIME_CHECK(
            gate->is_cuda() && gate->is_contiguous() &&
            (gate->scalar_type() == mfq_tensor_backend::kFloat16 ||
             gate->scalar_type() == mfq_tensor_backend::kFloat32) &&
            gate->dim() == 2 && gate->sizes() == input.sizes(),
            "NINT gate must be contiguous CUDA fp16/fp32 and match input shape");
    } else {
        MFQ_RUNTIME_CHECK(
            activation_mode == 0,
            "NINT activation mode requires a gate tensor");
    }
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1 && row_q_bits.is_cuda() &&
        row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT bitstream or q metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT neuron metadata is invalid");
    const int output_rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    const int activation_rows = static_cast<int>(input.size(0));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * static_cast<int>(group_size);
    MFQ_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8,
        "NINT packed matmul supports M in [1,8]");
    MFQ_RUNTIME_CHECK(
        group_size >= 4 && group_size <= 64 && real_width <= padded_width,
        "NINT packed matmul dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == output_rows &&
        row_q_bit_offsets.numel() == output_rows &&
        neuron_scale.numel() == output_rows &&
        neuron_minimum.numel() == output_rows,
        "NINT row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == mfq_tensor_backend::kInt8 &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT activation workspace is invalid");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == input.device() &&
        row_q_bits.device() == input.device() &&
        row_q_bit_offsets.device() == input.device() &&
        subgroup_scale.device() == input.device() &&
        subgroup_minimum.device() == input.device() &&
        neuron_scale.device() == input.device() &&
        neuron_minimum.device() == input.device() &&
        quantized_input.device() == input.device() &&
        input_scale.device() == input.device() &&
        (gate == nullptr || gate->device() == input.device()),
        "NINT tensors and workspace must share one CUDA device");
    auto output = mfq_tensor_backend::empty(
        {activation_rows, output_rows}, input.options().dtype(mfq_tensor_backend::kFloat16));
    const cudaStream_t stream = mfq_current_cuda_stream();
    const int quantize_threads = group_size <= 32 ? 32 : 64;
    const char* dense_group_setting = std::getenv("MFQ_NINT_GROUP_DOT");
    const char* whole_setting_for_sum = std::getenv("MFQ_NINT_WHOLE_GROUP");
    const char* batch_setting_for_sum = std::getenv("MFQ_NINT_BATCH_WHOLE_GROUP");
    const char* sum_setting = std::getenv("MFQ_NINT_ACTIVATION_SUM");
    const auto* interleaved_setting=std::getenv("MFQ_NINT_INTERLEAVED");
    const bool interleaved_default=activation_rows>=3 && activation_rows<=6;
    const bool interleaved_requested=interleaved_setting?interleaved_setting[0]=='1':interleaved_default;
    const auto* float_whole_setting=std::getenv("MFQ_NINT_FLOAT_WHOLE_GROUP");
    const bool whole_input=input.scalar_type()==mfq_tensor_backend::kFloat16 ||
        (input.scalar_type()==mfq_tensor_backend::kFloat32 && (!float_whole_setting || float_whole_setting[0]!='0'));
    const bool sum_default=nint_default_activation_sum(int(group_size),groups,activation_rows,uniform_bits,interleaved_requested);
    const char* q8_decode_option=std::getenv("MFQ_NINT_Q8_DECODE");
    const int q8_decode=q8_decode_option?std::atoi(q8_decode_option):2;
    const bool grouped_router=output_rows==512 && groups==54;
    const bool grouped_q8=aligned_q8 && group_size==48 && q8_decode>0 &&
        (reinterpret_cast<uintptr_t>(bitstream.data_ptr())&3u)==0 &&
        (activation_rows>1 || output_rows>2048 || q8_decode==3 || grouped_router);
    const auto* q8_sum_setting=std::getenv("MFQ_NINT_Q8_SUM");
    // Large vocabulary projections amortize this once-per-activation sum.
    // Tiny router/alpha/beta matrices keep their existing launch costs.
    const bool head_sum_default=output_rows>=65536 && groups==54 && activation_rows<=6;
    const bool q8_sums=grouped_q8 && q8_decode>=2 &&
        (q8_sum_setting?q8_sum_setting[0]=='1':head_sum_default);
    const bool cached_packed_sums = (sum_setting?sum_setting[0]=='1':sum_default) && !aligned_q8 &&
        (group_size==24 || group_size==28) && groups>32 &&
        whole_input &&
        (reinterpret_cast<uintptr_t>(bitstream.data_ptr())&3u)==0 &&
        (!dense_group_setting || dense_group_setting[0]!='0') &&
        (!whole_setting_for_sum || whole_setting_for_sum[0]!='0') &&
        (activation_rows==1 || !batch_setting_for_sum || batch_setting_for_sum[0]!='0') &&
        std::getenv("MFQ_NINT_DENSE_REFERENCE")==nullptr;
    const bool cached_sums=cached_packed_sums || q8_sums;
    const auto* fixed_setting=std::getenv("MFQ_NINT_FIXED_BITS");
    const bool interleaved=interleaved_requested &&
        uniform_bits==5 && group_size==28 && groups>32 && groups<=96 &&
        activation_rows>=2 && activation_rows<=6 && !aligned_q8 &&
        whole_input &&
        (reinterpret_cast<uintptr_t>(bitstream.data_ptr())&3u)==0 &&
        (!dense_group_setting || dense_group_setting[0]!='0') &&
        (!whole_setting_for_sum || whole_setting_for_sum[0]!='0') &&
        (!batch_setting_for_sum || batch_setting_for_sum[0]!='0') &&
        (!fixed_setting || fixed_setting[0]!='0') &&
        std::getenv("MFQ_NINT_DENSE_REFERENCE")==nullptr;
    const auto* exact_batch=std::getenv("MFQ_NINT_EXACT_BATCH");
    const int interleaved_columns=activation_rows<=2?2:activation_rows<=4?4:
        (exact_batch && exact_batch[0]=='0')?8:activation_rows;
    const NintPreparedGroupInput* prepared=nullptr;
    if(!gate)for(auto* scope=prepared_input_scope;scope && !prepared;scope=scope->previous) {
        if(scope->input.data_ptr()!=input.data_ptr() || scope->input.sizes()!=input.sizes() ||
                scope->input.scalar_type()!=input.scalar_type() || scope->input.device()!=input.device())continue;
        for(const auto& group:scope->groups)if(group.gs==group_size &&
                group.scale.size(1)==groups && (!interleaved || group.interleaved.defined())) {
            prepared=&group;break;
        }
    }
    if(prepared){quantized_input=prepared->codes;input_scale=prepared->scale;}
    mfq_tensor_backend::Tensor interleaved_activation;
    if(interleaved)interleaved_activation=prepared?prepared->interleaved:mfq_tensor_backend::empty(
        {groups,7,interleaved_columns,4},input.options().dtype(mfq_tensor_backend::kInt8));
    mfq_tensor_backend::Tensor activation_sums;
    if(cached_sums)activation_sums=prepared?prepared->sum:mfq_tensor_backend::empty(
        {activation_rows,groups},input.options().dtype(mfq_tensor_backend::kInt32));
    const auto quantize = [&](auto cached_sum) {
    const auto quantize_columns = [&](auto columns) {
    nint_quantize_activation_kernel<decltype(cached_sum)::value,decltype(columns)::value><<<
        dim3(activation_rows, groups), quantize_threads, 0, stream>>>(
            input.data_ptr(),
            gate == nullptr
                ? nullptr
                : gate->data_ptr(),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            activation_rows,
            real_width,
            padded_width,
            groups,
            static_cast<int>(group_size),
            static_cast<int>(activation_mode),
            input.scalar_type() == mfq_tensor_backend::kFloat32,
            gate != nullptr && gate->scalar_type() == mfq_tensor_backend::kFloat32,
            activation_sums.defined()?activation_sums.data_ptr<int32_t>():nullptr,
            interleaved_activation.defined()?interleaved_activation.data_ptr<int8_t>():nullptr);
    };
    if(interleaved) {
        if(interleaved_columns==2)quantize_columns(std::integral_constant<int,2>{});
        else if(interleaved_columns==4)quantize_columns(std::integral_constant<int,4>{});
        else if(interleaved_columns==5)quantize_columns(std::integral_constant<int,5>{});
        else if(interleaved_columns==6)quantize_columns(std::integral_constant<int,6>{});
        else quantize_columns(std::integral_constant<int,8>{});
    }else quantize_columns(std::integral_constant<int,0>{});
    };
    if(!prepared) {
        if(cached_sums)quantize(std::true_type{});else quantize(std::false_type{});
    }
    if (aligned_q8 && group_size % 4 == 0 &&
        (reinterpret_cast<uintptr_t>(bitstream.data_ptr<uint8_t>()) & 3u) == 0) {
        // The real 512-expert, 2560-wide router benefits from one affine per
        // complete group at M=1 as well. Retain the ordered path for tiny
        // projections such as GDN alpha/beta, which need more resident warps.
        if(grouped_q8) {
            const auto launch_short=[&](auto rows) {
                if(q8_decode>=2) {
                    const NintWholeArgs args{bitstream.data_ptr<uint8_t>(),row_q_bits.data_ptr<uint8_t>(),
                        subgroup_scale.data_ptr<uint8_t>(),subgroup_minimum.data_ptr<uint8_t>(),row_q_bit_offsets.data_ptr<int64_t>(),
                        neuron_scale.data_ptr<float>(),neuron_minimum.data_ptr<float>(),quantized_input.data_ptr<int8_t>(),
                        input_scale.data_ptr<float>(),reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
                        uint64_t(bitstream.numel()),output_rows,groups,activation_rows,false,
                        activation_sums.defined()?activation_sums.data_ptr<int32_t>():nullptr};
                    nint_launch_grouped_q8(args,stream);
                } else {
                    nint_matmul_aligned_q8_kernel<decltype(rows)::value,48><<<
                        (output_rows+3)/4,dim3(32,4),0,stream>>>(
                        bitstream.data_ptr<uint8_t>(),row_q_bit_offsets.data_ptr<int64_t>(),
                        subgroup_scale.data_ptr<uint8_t>(),subgroup_minimum.data_ptr<uint8_t>(),
                        neuron_scale.data_ptr<float>(),neuron_minimum.data_ptr<float>(),
                        quantized_input.data_ptr<int8_t>(),input_scale.data_ptr<float>(),
                        reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
                        activation_rows,output_rows,groups,padded_width,48);
                }
            };
            switch(activation_rows) {
            case 1:launch_short(std::integral_constant<int,1>{});break;
            case 2:launch_short(std::integral_constant<int,2>{});break;
            case 3:launch_short(std::integral_constant<int,3>{});break;
            case 4:launch_short(std::integral_constant<int,4>{});break;
            case 5:launch_short(std::integral_constant<int,5>{});break;
            case 6:launch_short(std::integral_constant<int,6>{});break;
            default:launch_short(std::integral_constant<int,8>{});break;
            }
        } else {
        // Specialize only the proven one-row input, independent of device.
        if (activation_rows == 1) {
        const char* ordered = std::getenv("MFQ_NINT_Q8_ORDERED");
        if (group_size == 48 && groups <= 64 && output_rows <= 2048 &&
                (!ordered || ordered[0] != '0')) {
            nint_q8_ordered_kernel<<<output_rows, dim3(32, 4), 0, stream>>>(
                bitstream.data_ptr<uint8_t>(), row_q_bit_offsets.data_ptr<int64_t>(),
                subgroup_scale.data_ptr<uint8_t>(), subgroup_minimum.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_minimum.data_ptr<float>(),
                quantized_input.data_ptr<int8_t>(), input_scale.data_ptr<float>(),
                reinterpret_cast<__half*>(output.data_ptr<mfq_half>()), output_rows, groups);
        } else {
        nint_matmul_aligned_q8_kernel<1><<<
            dim3((output_rows + 3) / 4), dim3(32, 4), 0, stream>>>(
                bitstream.data_ptr<uint8_t>(),
                row_q_bit_offsets.data_ptr<int64_t>(),
                subgroup_scale.data_ptr<uint8_t>(),
                subgroup_minimum.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(),
                neuron_minimum.data_ptr<float>(),
                quantized_input.data_ptr<int8_t>(),
                input_scale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
                activation_rows, output_rows, groups, padded_width,
                static_cast<int>(group_size));
        }
        } else {
        nint_matmul_aligned_q8_kernel<8><<<
            dim3((output_rows + 3) / 4), dim3(32, 4), 0, stream>>>(
                bitstream.data_ptr<uint8_t>(),
                row_q_bit_offsets.data_ptr<int64_t>(),
                subgroup_scale.data_ptr<uint8_t>(),
                subgroup_minimum.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(),
                neuron_minimum.data_ptr<float>(),
                quantized_input.data_ptr<int8_t>(),
                input_scale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
                activation_rows, output_rows, groups, padded_width,
                static_cast<int>(group_size));
        }
        }
    } else if ((group_size == 24 || group_size == 28) &&
            (reinterpret_cast<std::uintptr_t>(bitstream.data_ptr()) & 3u) == 0 &&
            (!dense_group_setting || dense_group_setting[0] != '0') &&
            std::getenv("MFQ_NINT_DENSE_REFERENCE") == nullptr) {
        const char* whole_setting=std::getenv("MFQ_NINT_WHOLE_GROUP");
        const char* batch_whole_setting=std::getenv("MFQ_NINT_BATCH_WHOLE_GROUP");
        const auto* small_group_setting=std::getenv("MFQ_NINT_FIXED_SMALL_GROUPS");
        const bool fixed_small=(!small_group_setting || small_group_setting[0]!='0') && groups<=32 &&
            activation_rows<=6 && (!fixed_setting || fixed_setting[0]!='0') &&
            ((group_size==24 && uniform_bits==6) || (group_size==28 && uniform_bits==5));
        if((activation_rows==1 || (activation_rows<=8 && (!batch_whole_setting || batch_whole_setting[0]!='0'))) &&
                (groups>32 || fixed_small) && whole_input &&
                (!whole_setting || whole_setting[0]!='0')) {
            const auto* group_load_option=std::getenv("MFQ_NINT_GROUP_LOAD");
            const bool group_load=group_load_option && group_load_option[0]=='1';
            const NintWholeArgs args{bitstream.data_ptr<uint8_t>(),row_q_bits.data_ptr<uint8_t>(),
                subgroup_scale.data_ptr<uint8_t>(),subgroup_minimum.data_ptr<uint8_t>(),
                row_q_bit_offsets.data_ptr<int64_t>(),neuron_scale.data_ptr<float>(),neuron_minimum.data_ptr<float>(),
                quantized_input.data_ptr<int8_t>(),input_scale.data_ptr<float>(),
                reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),uint64_t(bitstream.numel()),output_rows,groups,activation_rows,group_load,activation_sums.defined()?activation_sums.data_ptr<int32_t>():nullptr,interleaved_activation.defined()?reinterpret_cast<const int*>(interleaved_activation.data_ptr<int8_t>()):nullptr};
            const auto launch_whole=[&](auto gs,auto rows) {
                // Preserve the original group-to-lane map and four-term XOR
                // tree. A projection with at most 96 groups has no work for
                // warp 3; its contribution is the original positive zero.
                const auto* short_warps=std::getenv("MFQ_NINT_SHORT_WARPS");
                const auto* staged=std::getenv("MFQ_NINT_PACKED_STAGE");
                const auto* warp_rows=std::getenv("MFQ_NINT_WARP_ROWS");
                // Two-token projections benefit from four rows per CTA. At
                // six tokens the extra live accumulators regress, so the
                // default remains the original three-warps-per-row there.
                const bool tracing=mfq::cuda::dispatch_trace::enabled();
                const auto details=tracing?"M="+std::to_string(activation_rows)+" N="+std::to_string(output_rows)+
                    " gs="+std::to_string(group_size)+" groups="+std::to_string(groups)+" capacity="+std::to_string(decltype(rows)::value):std::string{};
                if(nint_launch_fixed_decode(args,int(group_size),uniform_bits,stream))return;
                const auto launch_cached=[&](auto sum_constant) {
                constexpr bool CachedSum=decltype(sum_constant)::value;
                if(groups<=96 && (warp_rows?warp_rows[0]=='1':activation_rows==2)) {
                    nint_whole_warp_rows_kernel<decltype(gs)::value,decltype(rows)::value,CachedSum><<<(output_rows+3)/4,dim3(32,4),0,stream>>>(args);
                    if(tracing)mfq::cuda::dispatch_trace::record(stream,"nint_whole_warp_rows",
                        reinterpret_cast<const void*>(nint_whole_warp_rows_kernel<decltype(gs)::value,decltype(rows)::value,CachedSum>),details);
                }else if(groups<=96 && staged && staged[0]=='1' &&
                    (reinterpret_cast<uintptr_t>(bitstream.data_ptr())&15u)==0) {
                    nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,3,true,CachedSum><<<output_rows,dim3(32,3),0,stream>>>(args);
                    if(tracing)mfq::cuda::dispatch_trace::record(stream,"nint_packed_stage",
                        reinterpret_cast<const void*>(nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,3,true,CachedSum>),details);
                }else if(groups<=96 && (!short_warps || short_warps[0]!='0')) {
                    nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,3,false,CachedSum><<<output_rows,dim3(32,3),0,stream>>>(args);
                    if(tracing)mfq::cuda::dispatch_trace::record(stream,"nint_whole_group_3warp",
                        reinterpret_cast<const void*>(nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,3,false,CachedSum>),details);
                }else {
                    nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,4,false,CachedSum><<<output_rows,dim3(32,4),0,stream>>>(args);
                    if(tracing)mfq::cuda::dispatch_trace::record(stream,"nint_whole_group_4warp",
                        reinterpret_cast<const void*>(nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,4,false,CachedSum>),details);
                }
                };
                if(cached_sums)launch_cached(std::true_type{});else launch_cached(std::false_type{});
            };
            const auto select_whole_rows=[&](auto gs) {
                if(activation_rows==1)launch_whole(gs,std::integral_constant<int,1>{});
                else if(activation_rows<=2)launch_whole(gs,std::integral_constant<int,2>{});
                else if(activation_rows<=4)launch_whole(gs,std::integral_constant<int,4>{});
                else if(const auto* exact=std::getenv("MFQ_NINT_EXACT_BATCH");!exact || exact[0]!='0') {
                    if(activation_rows==5)launch_whole(gs,std::integral_constant<int,5>{});
                    else if(activation_rows==6)launch_whole(gs,std::integral_constant<int,6>{});
                    else launch_whole(gs,std::integral_constant<int,8>{});
                }
                else launch_whole(gs,std::integral_constant<int,8>{});
            };
            if(group_size==24)select_whole_rows(std::integral_constant<int,24>{});
            else select_whole_rows(std::integral_constant<int,28>{});
        } else {
        const auto launch_dense = [&](auto gs, auto rows, auto warps) {
            constexpr int rows_per_block = 4 / decltype(warps)::value;
            nint_matmul_dense_kernel<decltype(gs)::value, decltype(rows)::value, decltype(warps)::value><<<
                (output_rows + rows_per_block - 1) / rows_per_block, dim3(32, 4), 0, stream>>>(
                bitstream.data_ptr<uint8_t>(), row_q_bits.data_ptr<uint8_t>(),
                row_q_bit_offsets.data_ptr<int64_t>(), subgroup_scale.data_ptr<uint8_t>(),
                subgroup_minimum.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),
                neuron_minimum.data_ptr<float>(), quantized_input.data_ptr<int8_t>(),
                input_scale.data_ptr<float>(), reinterpret_cast<half*>(output.data_ptr<mfq_half>()),
                activation_rows, output_rows, groups, uint64_t(bitstream.numel()));
        };
        const auto select_warps = [&](auto gs, auto rows) {
            if (groups <= 32) launch_dense(gs, rows, std::integral_constant<int, 1>{});
            else launch_dense(gs, rows, std::integral_constant<int, 4>{});
        };
        const auto select_rows = [&](auto gs) {
            if (activation_rows == 1) select_warps(gs, std::integral_constant<int, 1>{});
            else if (activation_rows <= 2) select_warps(gs, std::integral_constant<int, 2>{});
            else if (activation_rows <= 4) select_warps(gs, std::integral_constant<int, 4>{});
            else select_warps(gs, std::integral_constant<int, 8>{});
        };
        if (group_size == 24) select_rows(std::integral_constant<int, 24>{});
        else select_rows(std::integral_constant<int, 28>{});
        }
    } else {
    const auto launch=[&](auto single,auto grouped) {
    const auto launch_rows=[&](auto rows) {
    nint_matmul_kernel<decltype(single)::value,decltype(grouped)::value,decltype(rows)::value><<<
        dim3((output_rows + 3) / 4), dim3(32, 4), 0, stream>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            activation_rows,
            output_rows,
            groups,
            padded_width,
            static_cast<int>(group_size),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            0,
            0,
            0,
            0,
            false,static_cast<uint64_t>(bitstream.numel()));
    };
    const auto* exact=std::getenv("MFQ_NINT_EXACT_BATCH");
    if constexpr(!decltype(single)::value) {
        if(!exact || exact[0]!='0') {
            if(activation_rows==1)launch_rows(std::integral_constant<int,1>{});
            else if(activation_rows<=2)launch_rows(std::integral_constant<int,2>{});
            else if(activation_rows<=4)launch_rows(std::integral_constant<int,4>{});
            else if(activation_rows==5)launch_rows(std::integral_constant<int,5>{});
            else if(activation_rows==6)launch_rows(std::integral_constant<int,6>{});
            else launch_rows(std::integral_constant<int,8>{});
            return;
        }
    }
    launch_rows(std::integral_constant<int,8>{});
    };
    const auto* single=std::getenv("MFQ_NINT_SINGLE_ROW");
    if (activation_rows == 1 && single && std::strcmp(single, "1") == 0 &&
            (reinterpret_cast<std::uintptr_t>(bitstream.data_ptr()) & 3u) == 0) {
        const size_t activation_bytes =
            (static_cast<size_t>(padded_width) + 3u) & ~size_t(3u);
        const size_t shared_bytes = activation_bytes +
            static_cast<size_t>(groups) * sizeof(float);
        bool use_cache = false;
        int cache_warps = 1;
        if (real_width > output_rows) {
            // Query the actual device limit once per host thread/device.
            thread_local int cached_device = -1;
            thread_local int shared_limit = 0;
            thread_local int processor_count = 0;
            const int device = input.get_device();
            if (cached_device != device) {
                MFQ_RUNTIME_CHECK(cudaDeviceGetAttribute(&shared_limit,
                    cudaDevAttrMaxSharedMemoryPerBlock, device) == cudaSuccess,
                    "NINT shared activation device query failed");
                MFQ_RUNTIME_CHECK(cudaDeviceGetAttribute(&processor_count,
                    cudaDevAttrMultiProcessorCount, device) == cudaSuccess && processor_count > 0,
                    "NINT processor count query failed");
                cached_device = device;
            }
            const int rows_per_processor = (output_rows - 1) / processor_count + 1;
            while (cache_warps < rows_per_processor && cache_warps < 32) {
                cache_warps *= 2;
            }
            use_cache = cache_warps > 1 && shared_bytes <= static_cast<size_t>(shared_limit);
        }
        // At least one warp of groups avoids underfilled lanes.
        const auto* group_flag=std::getenv("MFQ_NINT_GROUP_DOT");
        const bool use_group=(!group_flag || std::strcmp(group_flag,"1")==0) &&
            groups>=32 && group_size>=1 && group_size<=64;
        if (use_cache) {
            const auto launch_cache=[&](auto grouped) {
            nint_matmul_activation_cache_kernel<decltype(grouped)::value><<<
                dim3((output_rows + cache_warps - 1) / cache_warps),
                dim3(32, cache_warps), shared_bytes, stream>>>(
                    bitstream.data_ptr<uint8_t>(), row_q_bits.data_ptr<uint8_t>(),
                    row_q_bit_offsets.data_ptr<int64_t>(),
                    subgroup_scale.data_ptr<uint8_t>(),
                    subgroup_minimum.data_ptr<uint8_t>(),
                    neuron_scale.data_ptr<float>(), neuron_minimum.data_ptr<float>(),
                    quantized_input.data_ptr<int8_t>(), input_scale.data_ptr<float>(),
                    reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
                    padded_width, groups, static_cast<int>(group_size), output_rows,
                    static_cast<uint64_t>(bitstream.numel()));
            };
            if(use_group)launch_cache(std::true_type{});
            else launch_cache(std::false_type{});
        } else {
            if(use_group)launch(std::true_type{},std::true_type{});
            else launch(std::true_type{},std::false_type{});
        }
    } else {
        launch(std::false_type{},std::false_type{});
    }
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint_matmul_fixed_bits_ws_cuda(
        mfq_tensor_backend::Tensor bitstream,mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,mfq_tensor_backend::Tensor input,
        int64_t group_size,mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale,int uniform_bits) {
    return nint_matmul_ws_impl(bitstream,row_q_bits,row_q_bit_offsets,subgroup_scale,
        subgroup_minimum,neuron_scale,neuron_minimum,input,nullptr,0,group_size,
        quantized_input,input_scale,false,uniform_bits);
}

mfq_tensor_backend::Tensor nint_matmul_ws_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor input,
        int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, nullptr, 0, group_size, quantized_input, input_scale);
}


mfq_tensor_backend::Tensor nint_matmul_q8_ws_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor input,
        int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, nullptr, 0, group_size, quantized_input, input_scale, true);
}


__global__ void nint_online_reduce_kernel(const float* partial,__half* output,int elements,int splits) {
    for(int index=int(blockIdx.x)*blockDim.x+threadIdx.x;index<elements;index+=blockDim.x*gridDim.x) {
        float sum=0;
        for(int split=0;split<splits;++split)sum+=partial[size_t(split)*elements+index];
        output[index]=__float2half_rn(sum);
    }
}

mfq_tensor_backend::Tensor nint_matmul_online_f16_cuda(
        mfq_tensor_backend::Tensor bitstream,mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,mfq_tensor_backend::Tensor input,int64_t group_size) {
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.dim()==2 &&
        input.scalar_type()==mfq_tensor_backend::kFloat16 && input.size(0)>8 && input.size(0)<=32,
        "NINT online projection requires contiguous CUDA FP16 input with 9..32 rows");
    for(const auto* tensor:{&bitstream,&row_q_bits,&row_q_bit_offsets,&subgroup_scale,
            &subgroup_minimum,&neuron_scale,&neuron_minimum}) {
        MFQ_RUNTIME_CHECK(tensor->is_cuda() && tensor->is_contiguous() && tensor->device()==input.device(),
            "NINT online projection tensors must share the input CUDA device");
    }
    MFQ_RUNTIME_CHECK(bitstream.scalar_type()==mfq_tensor_backend::kUInt8 && bitstream.dim()==1 &&
        row_q_bits.scalar_type()==mfq_tensor_backend::kUInt8 && row_q_bits.dim()==1 &&
        row_q_bit_offsets.scalar_type()==mfq_tensor_backend::kInt64 && row_q_bit_offsets.dim()==1 &&
        subgroup_scale.scalar_type()==mfq_tensor_backend::kUInt8 && subgroup_scale.dim()==2 &&
        subgroup_minimum.scalar_type()==mfq_tensor_backend::kUInt8 && subgroup_minimum.sizes()==subgroup_scale.sizes() &&
        neuron_scale.scalar_type()==mfq_tensor_backend::kFloat32 && neuron_minimum.scalar_type()==mfq_tensor_backend::kFloat32,
        "NINT online projection metadata types are invalid");
    const int rows=int(input.size(0)),outputs=int(subgroup_scale.size(0)),groups=int(subgroup_scale.size(1));
    MFQ_RUNTIME_CHECK(outputs>0 && groups>0 && group_size>=4 && group_size<=64 &&
        input.size(1)>0 && input.size(1)<=int64_t(groups)*group_size && row_q_bits.numel()==outputs &&
        row_q_bit_offsets.numel()==outputs && neuron_scale.numel()==outputs && neuron_minimum.numel()==outputs,
        "NINT online projection metadata dimensions are invalid");
    auto output=mfq_tensor_backend::empty({rows,outputs},input.options());
    const int output_tiles=(outputs+kNintPrefillTileN-1)/kNintPrefillTileN;
    int split=1;
    static thread_local int cached_device=-1,processors=1;
    if(cached_device!=input.get_device()) {
        MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&processors,cudaDevAttrMultiProcessorCount,input.get_device()));
        cached_device=input.get_device();
    }
    while(split<8 && output_tiles*split<processors*2 && input.size(1)>=int64_t(split*2)*kNintPrefillTileK)split*=2;
    auto partial=split>1 ? mfq_tensor_backend::empty({split,rows,outputs},input.options().dtype(mfq_tensor_backend::kFloat32))
        : mfq_tensor_backend::Tensor();
    const auto launch=[&](auto row_tile) {
        nint_matmul_tiled_prefill_kernel<decltype(row_tile)::value,true,false,true><<<
            dim3(output_tiles,split),dim3(32,kNintPrefillWarps),0,mfq_current_cuda_stream()>>>(
                bitstream.data_ptr<uint8_t>(),row_q_bits.data_ptr<uint8_t>(),row_q_bit_offsets.data_ptr<int64_t>(),
                subgroup_scale.data_ptr<uint8_t>(),subgroup_minimum.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(),neuron_minimum.data_ptr<float>(),
                reinterpret_cast<const __half*>(input.data_ptr<mfq_half>()),reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
                nullptr,nullptr,nullptr,nullptr,nullptr,
                1,1,outputs,groups,int(group_size),int(input.size(1)),0,0,0,true,rows,split,
                split>1 ? partial.data_ptr<float>() : nullptr);
    };
    if(rows<=16)launch(std::integral_constant<int,16>{});
    else launch(std::integral_constant<int,32>{});
    if(split>1)nint_online_reduce_kernel<<<(rows*outputs+255)/256,256,0,mfq_current_cuda_stream()>>>(
        partial.data_ptr<float>(),reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),rows*outputs,split);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

mfq_tensor_backend::Tensor nint_matmul_input_mul_fixed_bits_ws_cuda(
        mfq_tensor_backend::Tensor bitstream,mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor gate,int64_t activation_mode,int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,mfq_tensor_backend::Tensor input_scale,int uniform_bits) {
    return nint_matmul_ws_impl(bitstream,row_q_bits,row_q_bit_offsets,subgroup_scale,
        subgroup_minimum,neuron_scale,neuron_minimum,input,&gate,activation_mode,
        group_size,quantized_input,input_scale,false,uniform_bits);
}

mfq_tensor_backend::Tensor nint_matmul_input_mul_ws_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor gate,
        int64_t activation_mode,
        int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, &gate, activation_mode, group_size,
        quantized_input, input_scale);
}


mfq_tensor_backend::Tensor nint_matmul_input_mul_q8_ws_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor gate,
        int64_t activation_mode,
        int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, &gate, activation_mode, group_size,
        quantized_input, input_scale, true);
}


mfq_tensor_backend::Tensor nint_backward_input_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor output_gradient,
        int64_t width,
        int64_t group_size) {
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output_gradient.dim() == 2,
        "NINT output gradient must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1 && row_q_bits.is_cuda() &&
        row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT bitstream or q metadata is invalid");
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT neuron metadata is invalid");
    const int output_rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    const int activation_rows = static_cast<int>(output_gradient.size(0));
    MFQ_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8 &&
        output_gradient.size(1) == output_rows && group_size >= 4 &&
        group_size <= 64 && width > 0 &&
        width <= static_cast<int64_t>(groups) * group_size,
        "NINT packed backward dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == output_rows &&
        row_q_bit_offsets.numel() == output_rows &&
        neuron_scale.numel() == output_rows &&
        neuron_minimum.numel() == output_rows,
        "NINT backward row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == output_gradient.device() &&
        row_q_bits.device() == output_gradient.device() &&
        row_q_bit_offsets.device() == output_gradient.device() &&
        subgroup_scale.device() == output_gradient.device() &&
        subgroup_minimum.device() == output_gradient.device() &&
        neuron_scale.device() == output_gradient.device() &&
        neuron_minimum.device() == output_gradient.device(),
        "NINT backward tensors must share one CUDA device");
    auto result = mfq_tensor_backend::empty(
        {activation_rows, width}, output_gradient.options());
    constexpr int warps_per_block = 4;
    constexpr int rows_per_tile = 4;
    const int chunks_per_group =
        (static_cast<int>(group_size) + 3) / 4;
    const int total_chunks = groups * chunks_per_group;
    nint_backward_input_kernel<<<
        dim3(
            (total_chunks + warps_per_block - 1) / warps_per_block,
            (activation_rows + rows_per_tile - 1) / rows_per_tile),
        dim3(32, warps_per_block),
        0,
        mfq_current_cuda_stream()>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
            activation_rows,
            output_rows,
            groups,
            static_cast<int>(width),
            static_cast<int>(group_size));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}


template <typename Output>
static mfq_tensor_backend::Tensor nint8_zero_dequant_impl(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32,
        "NINT8-0 width is invalid");
    const int rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    auto output = mfq_tensor_backend::empty(
        {rows, width},
        quantized.options().dtype(
            std::is_same_v<Output, __half>
                ? mfq_tensor_backend::kFloat16
                : mfq_tensor_backend::kFloat32));
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(rows) * width;
    const int blocks = static_cast<int>(std::min<size_t>(
        (total + threads - 1) / threads, 65535));
    const auto output_data = [&]() {
        if constexpr (std::is_same_v<Output, __half>) {
            return reinterpret_cast<__half *>(output.template data_ptr<mfq_half>());
        } else {
            return output.template data_ptr<float>();
        }
    }();
    nint8_zero_decode_rows_kernel<Output><<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const int8_t *>(
                quantized.data_ptr<uint8_t>()),
            reinterpret_cast<const __half *>(scale.data_ptr<mfq_half>()),
            output_data,
            rows,
            groups,
            static_cast<int>(width));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint8_zero_dequant_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        int64_t width) {
    return nint8_zero_dequant_impl<__half>(
        std::move(quantized), std::move(scale), width);
}


mfq_tensor_backend::Tensor nint8_zero_dequant_f32_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        int64_t width) {
    return nint8_zero_dequant_impl<float>(
        std::move(quantized), std::move(scale), width);
}


mfq_tensor_backend::Tensor nint8_zero_gemv_ws_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        (input.scalar_type() == mfq_tensor_backend::kFloat16 ||
         input.scalar_type() == mfq_tensor_backend::kFloat32) &&
        input.dim() == 2,
        "NINT8-0 input must be contiguous CUDA fp16/fp32 rank-2");
    const int activation_rows = static_cast<int>(input.size(0));
    const int output_rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * 32;
    MFQ_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8 &&
        real_width <= padded_width,
        "NINT8-0 packed matmul expects M in [1,8] and K <= packed K");
    MFQ_RUNTIME_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == mfq_tensor_backend::kInt8 &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT8-0 activation workspace is invalid");
    MFQ_RUNTIME_CHECK(
        quantized.device() == input.device() && scale.device() == input.device() &&
        quantized_input.device() == input.device() &&
        input_scale.device() == input.device(),
        "NINT8-0 tensors and workspace must share one CUDA device");
    auto output = mfq_tensor_backend::empty(
        {activation_rows, output_rows}, input.options().dtype(mfq_tensor_backend::kFloat16));
    const cudaStream_t stream = mfq_current_cuda_stream();
    nint_quantize_activation_kernel<false><<<
        dim3(activation_rows, groups), 32, 0, stream>>>(
            input.data_ptr(),
            nullptr,
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            activation_rows,
            real_width,
            padded_width,
            groups,
            32,
            0,
            input.scalar_type() == mfq_tensor_backend::kFloat32,
            false);
    nint8_zero_matmul_kernel<<<
        dim3((output_rows + 3) / 4, activation_rows),
        dim3(32, 4),
        0,
        stream>>>(
            reinterpret_cast<const int8_t *>(
                quantized.data_ptr<uint8_t>()),
            reinterpret_cast<const __half *>(scale.data_ptr<mfq_half>()),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            activation_rows,
            output_rows,
            groups,
            padded_width);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint8_zero_mmq_f16_packed_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor input,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return nint_cublas_gemm_nt_f16acc_cuda(input, weight);
}


mfq_tensor_backend::Tensor nint8_zero_mmq_f32_packed_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor input,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 FP32 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return cublas_gemm_nt_f32_output(input, weight);
}


mfq_tensor_backend::Tensor nint8_zero_backward_input_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor output_gradient,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 &&
        output_gradient.size(1) == quantized.size(0),
        "NINT8-0 output-gradient geometry is invalid");
    MFQ_RUNTIME_CHECK(
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 ||
        output_gradient.scalar_type() == mfq_tensor_backend::kBFloat16 ||
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT8-0 output gradient must be fp16, bf16, or fp32");
    MFQ_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32 &&
        quantized.device() == output_gradient.device(),
        "NINT8-0 backward dimensions or device are invalid");
    const int rows = static_cast<int>(output_gradient.size(0));
    const int outputs = static_cast<int>(quantized.size(0));
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    auto result = mfq_tensor_backend::empty(
        {rows, width}, output_gradient.options());
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient,
        weight,
        result,
        rows,
        outputs,
        static_cast<int>(width),
        mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
