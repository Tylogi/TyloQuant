#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kStates = 16;
constexpr int kBanks = 2;
constexpr int kStatesPerBank = kStates / kBanks;
constexpr int kEntries = 256;
constexpr int kVector = 4;
constexpr int kVectorsPerGroup = 6;
constexpr int kGroup = 24;
constexpr int kAssignThreads = kVectorsPerGroup * 32;
constexpr int kRefitThreads = 256;
constexpr int kRefitWarps = kRefitThreads / 32;

__device__ __forceinline__ bool better(
    float error,
    int index,
    float best_error,
    int best_index) {
    return error < best_error || (error == best_error && index < best_index);
}

__device__ __forceinline__ int code_offset(
    int bank,
    int coordinate,
    int entry) {
    return (bank * kVector + coordinate) * kEntries + entry;
}

__global__ void nvq3j_assign_kernel(
    const float* __restrict__ value,
    const float* __restrict__ objective_weight,
    const float* __restrict__ anchor,
    const float* __restrict__ scale_lut,
    const uint8_t* __restrict__ bank_for_state,
    const int8_t* __restrict__ codebooks,
    uint8_t* __restrict__ out_state,
    uint8_t* __restrict__ out_indices,
    int padded_width,
    int valid_width,
    int groups_per_row) {
    const int flat_group = blockIdx.x;
    const int row = flat_group / groups_per_row;
    const int group = flat_group - row * groups_per_row;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int first_position = group * kGroup;
    const int valid_group = min(kGroup, valid_width - first_position);

    __shared__ float shared_value[kGroup];
    __shared__ float shared_weight[kGroup];
    __shared__ float vector_error[kStates][kVectorsPerGroup];
    __shared__ uint8_t vector_index[kStates][kVectorsPerGroup];

    if (tid < kGroup) {
        if (tid < valid_group) {
            const int64_t offset =
                static_cast<int64_t>(row) * padded_width + first_position + tid;
            shared_value[tid] = value[offset];
            shared_weight[tid] = objective_weight[offset];
        } else {
            shared_value[tid] = 0.0f;
            shared_weight[tid] = 0.0f;
        }
    }
    __syncthreads();

    const int vector_position = warp * kVector;
    const int valid_vector = max(0, min(kVector, valid_group - vector_position));
    for (int bank = 0; bank < kBanks; ++bank) {
        int states_for_bank[kStatesPerBank];
        int state_count = 0;
#pragma unroll
        for (int state = 0; state < kStates; ++state) {
            if (bank_for_state[state] == bank) {
                if (state_count < kStatesPerBank) {
                    states_for_bank[state_count] = state;
                }
                ++state_count;
            }
        }
        float local_error[kStatesPerBank];
        int local_index[kStatesPerBank];
#pragma unroll
        for (int rank = 0; rank < kStatesPerBank; ++rank) {
            local_error[rank] = FLT_MAX;
            local_index[rank] = 0;
        }

        for (int entry = lane; entry < kEntries; entry += 32) {
#pragma unroll
            for (int rank = 0; rank < kStatesPerBank; ++rank) {
                const int state = states_for_bank[rank];
                const float scale = anchor[row] * scale_lut[state];
                float error = 0.0f;
#pragma unroll
                for (int coordinate = 0; coordinate < kVector; ++coordinate) {
                    if (coordinate < valid_vector) {
                        const float code = static_cast<float>(
                            codebooks[code_offset(bank, coordinate, entry)]);
                        const float source =
                            shared_value[vector_position + coordinate];
                        const float weight =
                            shared_weight[vector_position + coordinate];
                        const float residual = source - scale * code;
                        error += weight * residual * residual;
                    }
                }
                if (better(
                        error,
                        entry,
                        local_error[rank],
                        local_index[rank])) {
                    local_error[rank] = error;
                    local_index[rank] = entry;
                }
            }
        }
#pragma unroll
        for (int rank = 0; rank < kStatesPerBank; ++rank) {
            for (int shift = 16; shift > 0; shift >>= 1) {
                const float other_error =
                    __shfl_down_sync(0xffffffff, local_error[rank], shift);
                const int other_index =
                    __shfl_down_sync(0xffffffff, local_index[rank], shift);
                if (lane < shift
                        && better(
                            other_error,
                            other_index,
                            local_error[rank],
                            local_index[rank])) {
                    local_error[rank] = other_error;
                    local_index[rank] = other_index;
                }
            }
            if (lane == 0) {
                const int state = states_for_bank[rank];
                vector_error[state][warp] = local_error[rank];
                vector_index[state][warp] =
                    static_cast<uint8_t>(local_index[rank]);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        float best_error = FLT_MAX;
        int best_state = 0;
        uint8_t best_indices[kVectorsPerGroup] = {0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int state = 0; state < kStates; ++state) {
            float error = 0.0f;
#pragma unroll
            for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
                error += vector_error[state][vector];
            }
            if (better(error, state, best_error, best_state)) {
                best_error = error;
                best_state = state;
#pragma unroll
                for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
                    best_indices[vector] = vector_index[state][vector];
                }
            }
        }
        out_state[flat_group] = static_cast<uint8_t>(best_state);
#pragma unroll
        for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
            out_indices[
                static_cast<int64_t>(flat_group) * kVectorsPerGroup + vector] =
                best_indices[vector];
        }
    }
}

__global__ void nvq3j_refit_anchor_kernel(
    const float* __restrict__ value,
    const float* __restrict__ objective_weight,
    const float* __restrict__ previous_anchor,
    const float* __restrict__ scale_lut,
    const uint8_t* __restrict__ bank_for_state,
    const int8_t* __restrict__ codebooks,
    const uint8_t* __restrict__ states,
    const uint8_t* __restrict__ indices,
    float* __restrict__ out_anchor,
    int padded_width,
    int valid_width,
    int groups_per_row) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float numerator = 0.0f;
    float denominator = 0.0f;

    for (int position = tid; position < valid_width; position += blockDim.x) {
        const int group = position / kGroup;
        const int in_group = position - group * kGroup;
        const int vector = in_group / kVector;
        const int coordinate = in_group - vector * kVector;
        const int flat_group = row * groups_per_row + group;
        const int state = states[flat_group];
        const int bank = bank_for_state[state];
        const int entry = indices[
            static_cast<int64_t>(flat_group) * kVectorsPerGroup + vector];
        const float code = static_cast<float>(
            codebooks[code_offset(bank, coordinate, entry)]);
        const float basis = scale_lut[state] * code;
        const int64_t offset =
            static_cast<int64_t>(row) * padded_width + position;
        const float x = value[offset];
        const float weight = objective_weight[offset];
        numerator = fmaf(weight * x, basis, numerator);
        denominator = fmaf(weight * basis, basis, denominator);
    }

    for (int shift = 16; shift > 0; shift >>= 1) {
        numerator += __shfl_down_sync(0xffffffff, numerator, shift);
        denominator += __shfl_down_sync(0xffffffff, denominator, shift);
    }
    __shared__ float warp_numerator[kRefitWarps];
    __shared__ float warp_denominator[kRefitWarps];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_numerator[warp] = numerator;
        warp_denominator[warp] = denominator;
    }
    __syncthreads();
    if (warp == 0) {
        numerator = lane < kRefitWarps ? warp_numerator[lane] : 0.0f;
        denominator = lane < kRefitWarps ? warp_denominator[lane] : 0.0f;
        for (int shift = 16; shift > 0; shift >>= 1) {
            numerator += __shfl_down_sync(0xffffffff, numerator, shift);
            denominator += __shfl_down_sync(0xffffffff, denominator, shift);
        }
        if (lane == 0) {
            const float fitted = denominator > 0.0f
                ? fmaxf(numerator / denominator, 0.0f)
                : previous_anchor[row];
            out_anchor[row] = __half2float(__float2half_rn(fitted));
        }
    }
}

void check_inputs(
    const torch::Tensor& value,
    const torch::Tensor& objective_weight,
    const torch::Tensor& initial_anchor,
    const torch::Tensor& scale_lut,
    const torch::Tensor& bank_for_state,
    const torch::Tensor& codebooks,
    int64_t valid_width,
    int64_t refine_steps) {
    TORCH_CHECK(
        value.is_cuda() && value.is_contiguous()
            && value.scalar_type() == torch::kFloat32
            && value.dim() == 2 && value.size(0) > 0
            && value.size(1) % kGroup == 0,
        "nvq3j_assign: value must be CUDA contiguous float32 [rows,padded_K]");
    TORCH_CHECK(
        objective_weight.is_cuda() && objective_weight.is_contiguous()
            && objective_weight.scalar_type() == torch::kFloat32
            && objective_weight.sizes() == value.sizes(),
        "nvq3j_assign: objective_weight must match value");
    TORCH_CHECK(
        initial_anchor.is_cuda() && initial_anchor.is_contiguous()
            && initial_anchor.scalar_type() == torch::kFloat32
            && initial_anchor.dim() == 1
            && initial_anchor.size(0) == value.size(0),
        "nvq3j_assign: initial_anchor must be CUDA float32 [rows]");
    TORCH_CHECK(
        scale_lut.is_cuda() && scale_lut.is_contiguous()
            && scale_lut.scalar_type() == torch::kFloat32
            && scale_lut.sizes() == torch::IntArrayRef({kStates}),
        "nvq3j_assign: scale_lut must be CUDA float32 [16]");
    TORCH_CHECK(
        bank_for_state.is_cuda() && bank_for_state.is_contiguous()
            && bank_for_state.scalar_type() == torch::kUInt8
            && bank_for_state.sizes() == torch::IntArrayRef({kStates}),
        "nvq3j_assign: bank_for_state must be CUDA uint8 [16]");
    TORCH_CHECK(
        codebooks.is_cuda() && codebooks.is_contiguous()
            && codebooks.scalar_type() == torch::kInt8
            && codebooks.sizes()
                == torch::IntArrayRef({kBanks, kVector, kEntries}),
        "nvq3j_assign: codebooks must be CUDA int8 [2,4,256]");
    TORCH_CHECK(
        valid_width > 0 && valid_width <= value.size(1)
            && value.size(1) - valid_width < kGroup,
        "nvq3j_assign: invalid unpadded width");
    TORCH_CHECK(
        refine_steps >= 0 && refine_steps <= 4,
        "nvq3j_assign: refine_steps must be in [0,4]");
    const int device = value.get_device();
    TORCH_CHECK(
        objective_weight.get_device() == device
            && initial_anchor.get_device() == device
            && scale_lut.get_device() == device
            && bank_for_state.get_device() == device
            && codebooks.get_device() == device,
        "nvq3j_assign: tensors must share one CUDA device");
}

}  // namespace

std::vector<torch::Tensor> nvq3j_assign_cuda(
    torch::Tensor value,
    torch::Tensor objective_weight,
    torch::Tensor initial_anchor,
    torch::Tensor scale_lut,
    torch::Tensor bank_for_state,
    torch::Tensor codebooks,
    int64_t valid_width,
    int64_t refine_steps) {
    check_inputs(
        value,
        objective_weight,
        initial_anchor,
        scale_lut,
        bank_for_state,
        codebooks,
        valid_width,
        refine_steps);
    const int64_t rows = value.size(0);
    const int padded_width = static_cast<int>(value.size(1));
    const int groups_per_row = padded_width / kGroup;
    const int64_t groups = rows * groups_per_row;
    auto byte_options = value.options().dtype(torch::kUInt8);
    auto states = torch::empty({rows, groups_per_row}, byte_options);
    auto indices = torch::empty(
        {rows, groups_per_row, kVectorsPerGroup}, byte_options);
    auto first_anchor = initial_anchor.contiguous();
    auto second_anchor = torch::empty({rows}, value.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto launch_assign = [&](const torch::Tensor& current_anchor) {
        nvq3j_assign_kernel<<<groups, kAssignThreads, 0, stream>>>(
            value.data_ptr<float>(),
            objective_weight.data_ptr<float>(),
            current_anchor.data_ptr<float>(),
            scale_lut.data_ptr<float>(),
            bank_for_state.data_ptr<uint8_t>(),
            codebooks.data_ptr<int8_t>(),
            states.data_ptr<uint8_t>(),
            indices.data_ptr<uint8_t>(),
            padded_width,
            static_cast<int>(valid_width),
            groups_per_row);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    auto launch_refit = [&](
        const torch::Tensor& current_anchor,
        torch::Tensor& fitted_anchor) {
        nvq3j_refit_anchor_kernel<<<rows, kRefitThreads, 0, stream>>>(
            value.data_ptr<float>(),
            objective_weight.data_ptr<float>(),
            current_anchor.data_ptr<float>(),
            scale_lut.data_ptr<float>(),
            bank_for_state.data_ptr<uint8_t>(),
            codebooks.data_ptr<int8_t>(),
            states.data_ptr<uint8_t>(),
            indices.data_ptr<uint8_t>(),
            fitted_anchor.data_ptr<float>(),
            padded_width,
            static_cast<int>(valid_width),
            groups_per_row);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launch_assign(first_anchor);
    for (int64_t refine = 0; refine < refine_steps; ++refine) {
        launch_refit(first_anchor, second_anchor);
        launch_assign(second_anchor);
        std::swap(first_anchor, second_anchor);
    }
    return {first_anchor, states, indices};
}
