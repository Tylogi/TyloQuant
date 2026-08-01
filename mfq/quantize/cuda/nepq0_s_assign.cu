#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kBanks = 256;
constexpr int kStates = 4;
constexpr int kEntries = 8;
constexpr int kSubvector = 4;
constexpr int kVector = 8;
constexpr int kVectorsPerGroup = 3;
constexpr int kGroup = 24;
constexpr int kGroupsPerSuper = 4;
constexpr int kSuper = kGroup * kGroupsPerSuper;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ bool better(
    float error,
    int index,
    float best_error,
    int best_index) {
    return error < best_error || (error == best_error && index < best_index);
}

__device__ __forceinline__ int table_offset(
    int state,
    int entry,
    int coordinate,
    int bank) {
    // [state, entry, coordinate, bank] keeps a bank-wide load coalesced.
    return (((state * kEntries + entry) * kSubvector + coordinate) * kBanks)
        + bank;
}

__device__ __forceinline__ void nearest_half(
    const float* __restrict__ value,
    int valid,
    float scale,
    const int8_t* __restrict__ table,
    int state,
    int bank,
    float& best_error,
    uint8_t& best_index) {
    best_error = FLT_MAX;
    best_index = 0;
    if (valid <= 0) {
        best_error = 0.0f;
        return;
    }
    for (int entry = 0; entry < kEntries; ++entry) {
        float error = 0.0f;
#pragma unroll
        for (int coordinate = 0; coordinate < kSubvector; ++coordinate) {
            if (coordinate < valid) {
                const float code = static_cast<float>(
                    table[table_offset(state, entry, coordinate, bank)]);
                const float residual = value[coordinate] - scale * code;
                error = fmaf(residual, residual, error);
            }
        }
        if (better(
                error,
                entry,
                best_error,
                static_cast<int>(best_index))) {
            best_error = error;
            best_index = static_cast<uint8_t>(entry);
        }
    }
}

__global__ void nepq0_s_assign_kernel(
    const float* __restrict__ value,
    const float* __restrict__ anchor,
    const float* __restrict__ scale_lut,
    const int8_t* __restrict__ first_tables,
    const int8_t* __restrict__ second_tables,
    uint8_t* __restrict__ out_bank,
    uint8_t* __restrict__ out_state,
    uint8_t* __restrict__ out_indices,
    int width,
    int groups_per_row,
    int supers_per_row) {
    const int row = blockIdx.x / supers_per_row;
    const int super_index = blockIdx.x - row * supers_per_row;
    const int bank = threadIdx.x;
    const int first_group = super_index * kGroupsPerSuper;
    const int first_position = first_group * kGroup;

    __shared__ float shared_value[kSuper];
    __shared__ float warp_error[kWarps];
    __shared__ int warp_bank[kWarps];
    __shared__ int winning_bank;

    if (bank < kSuper) {
        const int position = first_position + bank;
        shared_value[bank] = position < width
            ? value[static_cast<int64_t>(row) * width + position]
            : 0.0f;
    }
    __syncthreads();

    float super_error = 0.0f;
    uint8_t selected_state[kGroupsPerSuper] = {0, 0, 0, 0};
    uint8_t selected_indices[kGroupsPerSuper][kVectorsPerGroup] = {};
    const float row_anchor = anchor[row];

#pragma unroll
    for (int local_group = 0; local_group < kGroupsPerSuper; ++local_group) {
        const int group = first_group + local_group;
        if (group >= groups_per_row) {
            continue;
        }
        const int valid_group = min(kGroup, width - group * kGroup);
        const float* group_value = shared_value + local_group * kGroup;
        float best_group_error = FLT_MAX;
        uint8_t best_group_state = 0;
        uint8_t best_group_indices[kVectorsPerGroup] = {0, 0, 0};

#pragma unroll
        for (int state = 0; state < kStates; ++state) {
            const float relative = scale_lut[state * kBanks + bank];
            const float scale = row_anchor * relative;
            float group_error = 0.0f;
            uint8_t group_indices[kVectorsPerGroup] = {0, 0, 0};

#pragma unroll
            for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
                int valid_vector = valid_group - vector * kVector;
                valid_vector = max(0, min(kVector, valid_vector));
                const int valid_first = min(kSubvector, valid_vector);
                const int valid_second = max(0, valid_vector - kSubvector);
                const float* vector_value = group_value + vector * kVector;
                float first_error;
                float second_error;
                uint8_t first_index;
                uint8_t second_index;
                nearest_half(
                    vector_value,
                    valid_first,
                    scale,
                    first_tables,
                    state,
                    bank,
                    first_error,
                    first_index);
                nearest_half(
                    vector_value + kSubvector,
                    valid_second,
                    scale,
                    second_tables,
                    state,
                    bank,
                    second_error,
                    second_index);
                group_error += first_error + second_error;
                group_indices[vector] = static_cast<uint8_t>(
                    first_index | (second_index << 3));
            }
            if (better(
                    group_error,
                    state,
                    best_group_error,
                    static_cast<int>(best_group_state))) {
                best_group_error = group_error;
                best_group_state = static_cast<uint8_t>(state);
#pragma unroll
                for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
                    best_group_indices[vector] = group_indices[vector];
                }
            }
        }
        super_error += best_group_error;
        selected_state[local_group] = best_group_state;
#pragma unroll
        for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
            selected_indices[local_group][vector] = best_group_indices[vector];
        }
    }

    const int lane = bank & 31;
    const int warp = bank >> 5;
    float reduced_error = super_error;
    int reduced_bank = bank;
    for (int shift = 16; shift > 0; shift >>= 1) {
        const float other_error =
            __shfl_down_sync(0xffffffff, reduced_error, shift);
        const int other_bank =
            __shfl_down_sync(0xffffffff, reduced_bank, shift);
        if (lane < shift
                && better(
                    other_error,
                    other_bank,
                    reduced_error,
                    reduced_bank)) {
            reduced_error = other_error;
            reduced_bank = other_bank;
        }
    }
    if (lane == 0) {
        warp_error[warp] = reduced_error;
        warp_bank[warp] = reduced_bank;
    }
    __syncthreads();

    if (warp == 0) {
        reduced_error = lane < kWarps ? warp_error[lane] : FLT_MAX;
        reduced_bank = lane < kWarps ? warp_bank[lane] : kBanks;
        for (int shift = 16; shift > 0; shift >>= 1) {
            const float other_error =
                __shfl_down_sync(0xffffffff, reduced_error, shift);
            const int other_bank =
                __shfl_down_sync(0xffffffff, reduced_bank, shift);
            if (lane < shift
                    && better(
                        other_error,
                        other_bank,
                        reduced_error,
                        reduced_bank)) {
                reduced_error = other_error;
                reduced_bank = other_bank;
            }
        }
        if (lane == 0) {
            winning_bank = reduced_bank;
            out_bank[static_cast<int64_t>(row) * supers_per_row + super_index] =
                static_cast<uint8_t>(reduced_bank);
        }
    }
    __syncthreads();

    if (bank == winning_bank) {
#pragma unroll
        for (int local_group = 0; local_group < kGroupsPerSuper; ++local_group) {
            const int group = first_group + local_group;
            if (group >= groups_per_row) {
                continue;
            }
            out_state[static_cast<int64_t>(row) * groups_per_row + group] =
                selected_state[local_group];
#pragma unroll
            for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
                out_indices[
                    (static_cast<int64_t>(row) * groups_per_row + group)
                        * kVectorsPerGroup
                    + vector] = selected_indices[local_group][vector];
            }
        }
    }
}

__global__ void nepq0_s_refit_anchor_kernel(
    const float* __restrict__ value,
    const float* __restrict__ previous_anchor,
    const float* __restrict__ scale_lut,
    const int8_t* __restrict__ first_tables,
    const int8_t* __restrict__ second_tables,
    const uint8_t* __restrict__ bank_ids,
    const uint8_t* __restrict__ states,
    const uint8_t* __restrict__ indices,
    float* __restrict__ out_anchor,
    int width,
    int groups_per_row,
    int supers_per_row) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float numerator = 0.0f;
    float denominator = 0.0f;

    for (int position = tid; position < width; position += blockDim.x) {
        const int group = position / kGroup;
        const int in_group = position - group * kGroup;
        const int vector = in_group / kVector;
        const int in_vector = in_group - vector * kVector;
        const int bank = bank_ids[
            static_cast<int64_t>(row) * supers_per_row + group / kGroupsPerSuper];
        const int state =
            states[static_cast<int64_t>(row) * groups_per_row + group];
        const int composite =
            indices[
                (static_cast<int64_t>(row) * groups_per_row + group)
                    * kVectorsPerGroup
                + vector];
        const bool second = in_vector >= kSubvector;
        const int entry = second ? composite >> 3 : composite & 7;
        const int coordinate =
            second ? in_vector - kSubvector : in_vector;
        const int8_t* table = second ? second_tables : first_tables;
        const float code = static_cast<float>(
            table[table_offset(state, entry, coordinate, bank)]);
        const float basis = scale_lut[state * kBanks + bank] * code;
        const float x = value[static_cast<int64_t>(row) * width + position];
        numerator = fmaf(x, basis, numerator);
        denominator = fmaf(basis, basis, denominator);
    }

    for (int shift = 16; shift > 0; shift >>= 1) {
        numerator += __shfl_down_sync(0xffffffff, numerator, shift);
        denominator += __shfl_down_sync(0xffffffff, denominator, shift);
    }
    __shared__ float warp_numerator[kWarps];
    __shared__ float warp_denominator[kWarps];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_numerator[warp] = numerator;
        warp_denominator[warp] = denominator;
    }
    __syncthreads();

    if (warp == 0) {
        numerator = lane < kWarps ? warp_numerator[lane] : 0.0f;
        denominator = lane < kWarps ? warp_denominator[lane] : 0.0f;
        for (int shift = 16; shift > 0; shift >>= 1) {
            numerator += __shfl_down_sync(0xffffffff, numerator, shift);
            denominator += __shfl_down_sync(0xffffffff, denominator, shift);
        }
        if (lane == 0) {
            float fitted = denominator > 0.0f
                ? fmaxf(numerator / denominator, 0.0f)
                : previous_anchor[row];
            out_anchor[row] = __half2float(__float2half_rn(fitted));
        }
    }
}

void check_nepq0_s_inputs(
    const torch::Tensor& value,
    const torch::Tensor& initial_anchor,
    const torch::Tensor& scale_lut,
    const torch::Tensor& first_tables,
    const torch::Tensor& second_tables) {
    TORCH_CHECK(
        value.is_cuda() && value.is_contiguous()
            && value.scalar_type() == torch::kFloat32,
        "nepq0_s_assign: value must be CUDA contiguous float32");
    TORCH_CHECK(
        value.dim() == 2 && value.size(0) > 0 && value.size(1) > 0
            && value.size(1) % kVector == 0,
        "nepq0_s_assign: value must have shape [rows,K] with K divisible by 8");
    TORCH_CHECK(
        initial_anchor.is_cuda() && initial_anchor.is_contiguous()
            && initial_anchor.scalar_type() == torch::kFloat32
            && initial_anchor.dim() == 1
            && initial_anchor.size(0) == value.size(0),
        "nepq0_s_assign: initial_anchor must be CUDA contiguous float32 [rows]");
    TORCH_CHECK(
        scale_lut.is_cuda() && scale_lut.is_contiguous()
            && scale_lut.scalar_type() == torch::kFloat32
            && scale_lut.sizes() == torch::IntArrayRef({kStates, kBanks}),
        "nepq0_s_assign: scale_lut must be CUDA float32 [4,256]");
    TORCH_CHECK(
        first_tables.is_cuda() && first_tables.is_contiguous()
            && first_tables.scalar_type() == torch::kInt8
            && first_tables.sizes()
                == torch::IntArrayRef(
                    {kStates, kEntries, kSubvector, kBanks}),
        "nepq0_s_assign: first_tables must be CUDA int8 [4,8,4,256]");
    TORCH_CHECK(
        second_tables.is_cuda() && second_tables.is_contiguous()
            && second_tables.scalar_type() == torch::kInt8
            && second_tables.sizes() == first_tables.sizes(),
        "nepq0_s_assign: second_tables shape mismatch");
    const int device = value.get_device();
    TORCH_CHECK(
        initial_anchor.get_device() == device
            && scale_lut.get_device() == device
            && first_tables.get_device() == device
            && second_tables.get_device() == device,
        "nepq0_s_assign: tensors must share one CUDA device");
}

}  // namespace

std::vector<torch::Tensor> nepq0_s_assign_cuda(
    torch::Tensor value,
    torch::Tensor initial_anchor,
    torch::Tensor scale_lut,
    torch::Tensor first_tables,
    torch::Tensor second_tables) {
    check_nepq0_s_inputs(
        value,
        initial_anchor,
        scale_lut,
        first_tables,
        second_tables);
    const int64_t rows = value.size(0);
    const int width = static_cast<int>(value.size(1));
    const int groups_per_row = (width + kGroup - 1) / kGroup;
    const int supers_per_row =
        (groups_per_row + kGroupsPerSuper - 1) / kGroupsPerSuper;
    auto byte_options = value.options().dtype(torch::kUInt8);
    auto bank_ids = torch::empty({rows, supers_per_row}, byte_options);
    auto states = torch::empty({rows, groups_per_row}, byte_options);
    auto indices = torch::empty(
        {rows, groups_per_row, kVectorsPerGroup},
        byte_options);
    auto fitted_anchor = torch::empty({rows}, value.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t assignment_blocks = rows * supers_per_row;

    nepq0_s_assign_kernel<<<assignment_blocks, kThreads, 0, stream>>>(
        value.data_ptr<float>(),
        initial_anchor.data_ptr<float>(),
        scale_lut.data_ptr<float>(),
        first_tables.data_ptr<int8_t>(),
        second_tables.data_ptr<int8_t>(),
        bank_ids.data_ptr<uint8_t>(),
        states.data_ptr<uint8_t>(),
        indices.data_ptr<uint8_t>(),
        width,
        groups_per_row,
        supers_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    nepq0_s_refit_anchor_kernel<<<rows, kThreads, 0, stream>>>(
        value.data_ptr<float>(),
        initial_anchor.data_ptr<float>(),
        scale_lut.data_ptr<float>(),
        first_tables.data_ptr<int8_t>(),
        second_tables.data_ptr<int8_t>(),
        bank_ids.data_ptr<uint8_t>(),
        states.data_ptr<uint8_t>(),
        indices.data_ptr<uint8_t>(),
        fitted_anchor.data_ptr<float>(),
        width,
        groups_per_row,
        supers_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    nepq0_s_assign_kernel<<<assignment_blocks, kThreads, 0, stream>>>(
        value.data_ptr<float>(),
        fitted_anchor.data_ptr<float>(),
        scale_lut.data_ptr<float>(),
        first_tables.data_ptr<int8_t>(),
        second_tables.data_ptr<int8_t>(),
        bank_ids.data_ptr<uint8_t>(),
        states.data_ptr<uint8_t>(),
        indices.data_ptr<uint8_t>(),
        width,
        groups_per_row,
        supers_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {fitted_anchor, bank_ids, states, indices};
}
