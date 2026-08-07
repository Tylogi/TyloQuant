#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kStates = 4;
constexpr int kEntries = 8;
constexpr int kSubvector = 4;
constexpr int kVector = 8;
constexpr int kVectorsPerGroup = 3;
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
    int state,
    int entry,
    int coordinate) {
    return (state * kEntries + entry) * kSubvector + coordinate;
}

__global__ void npq0_s_assign_kernel(
    const float* __restrict__ value,
    const float* __restrict__ objective_weight,
    const float* __restrict__ anchor,
    const float* __restrict__ scale_lut,
    const int8_t* __restrict__ first_codebooks,
    const int8_t* __restrict__ second_codebooks,
    uint8_t* __restrict__ out_state,
    uint8_t* __restrict__ out_first,
    uint8_t* __restrict__ out_second,
    float* __restrict__ out_group_error,
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
    __shared__ uint8_t vector_first[kStates][kVectorsPerGroup];
    __shared__ uint8_t vector_second[kStates][kVectorsPerGroup];

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
    const bool active_half = lane < 16;
    const bool second_half = lane >= 8;
    const int entry = lane & 7;
    const int valid_half = second_half
        ? max(0, valid_vector - kSubvector)
        : min(kSubvector, valid_vector);
    const int coordinate_base = vector_position + (second_half ? kSubvector : 0);

#pragma unroll
    for (int state = 0; state < kStates; ++state) {
        float error = FLT_MAX;
        if (active_half) {
            float signal = 0.0f;
            float dot = 0.0f;
            float norm = 0.0f;
            const int8_t* codebook = second_half
                ? second_codebooks
                : first_codebooks;
#pragma unroll
            for (int coordinate = 0; coordinate < kSubvector; ++coordinate) {
                if (coordinate < valid_half) {
                    const float x = shared_value[coordinate_base + coordinate];
                    const float w = shared_weight[coordinate_base + coordinate];
                    const float code = static_cast<float>(
                        codebook[code_offset(state, entry, coordinate)]);
                    signal = fmaf(w * x, x, signal);
                    dot = fmaf(w * x, code, dot);
                    norm = fmaf(w * code, code, norm);
                }
            }
            const float scale = anchor[row] * scale_lut[state];
            error = fmaf(
                scale * scale,
                norm,
                fmaf(-2.0f * scale, dot, signal));
        }
        int best_index = entry;
        for (int shift = 4; shift > 0; shift >>= 1) {
            const float other_error =
                __shfl_down_sync(0xffffffff, error, shift, 8);
            const int other_index =
                __shfl_down_sync(0xffffffff, best_index, shift, 8);
            if ((lane & 7) < shift
                    && better(other_error, other_index, error, best_index)) {
                error = other_error;
                best_index = other_index;
            }
        }
        const float second_error = __shfl_sync(0xffffffff, error, 8);
        const int second_index = __shfl_sync(0xffffffff, best_index, 8);
        if (lane == 0) {
            vector_error[state][warp] = error + second_error;
            vector_first[state][warp] = static_cast<uint8_t>(best_index);
            vector_second[state][warp] = static_cast<uint8_t>(second_index);
        }
    }
    __syncthreads();

    if (tid == 0) {
        float best_error = FLT_MAX;
        int best_state = 0;
        uint8_t best_first[kVectorsPerGroup] = {0, 0, 0};
        uint8_t best_second[kVectorsPerGroup] = {0, 0, 0};
#pragma unroll
        for (int state = 0; state < kStates; ++state) {
            const float group_error =
                vector_error[state][0]
                + vector_error[state][1]
                + vector_error[state][2];
            if (better(group_error, state, best_error, best_state)) {
                best_error = group_error;
                best_state = state;
#pragma unroll
                for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
                    best_first[vector] = vector_first[state][vector];
                    best_second[vector] = vector_second[state][vector];
                }
            }
        }
        out_state[flat_group] = static_cast<uint8_t>(best_state);
        out_group_error[flat_group] = best_error;
#pragma unroll
        for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
            const int64_t offset =
                static_cast<int64_t>(flat_group) * kVectorsPerGroup + vector;
            out_first[offset] = best_first[vector];
            out_second[offset] = best_second[vector];
        }
    }
}

__global__ void npq0_s_refit_anchor_kernel(
    const float* __restrict__ value,
    const float* __restrict__ objective_weight,
    const float* __restrict__ previous_anchor,
    const float* __restrict__ scale_lut,
    const int8_t* __restrict__ first_codebooks,
    const int8_t* __restrict__ second_codebooks,
    const uint8_t* __restrict__ states,
    const uint8_t* __restrict__ first_indices,
    const uint8_t* __restrict__ second_indices,
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
        const int in_vector = in_group - vector * kVector;
        const int flat_group = row * groups_per_row + group;
        const int state = states[flat_group];
        const bool second = in_vector >= kSubvector;
        const int coordinate = second ? in_vector - kSubvector : in_vector;
        const int entry = second
            ? second_indices[
                static_cast<int64_t>(flat_group) * kVectorsPerGroup + vector]
            : first_indices[
                static_cast<int64_t>(flat_group) * kVectorsPerGroup + vector];
        const int8_t* codebook = second ? second_codebooks : first_codebooks;
        const float code = static_cast<float>(
            codebook[code_offset(state, entry, coordinate)]);
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
    const torch::Tensor& first_codebooks,
    const torch::Tensor& second_codebooks,
    int64_t valid_width,
    int64_t refine_steps) {
    TORCH_CHECK(
        value.is_cuda() && value.is_contiguous()
            && value.scalar_type() == torch::kFloat32
            && value.dim() == 2 && value.size(0) > 0
            && value.size(1) % kGroup == 0,
        "npq0_s_assign: value must be CUDA contiguous float32 [rows,padded_K]");
    TORCH_CHECK(
        objective_weight.is_cuda() && objective_weight.is_contiguous()
            && objective_weight.scalar_type() == torch::kFloat32
            && objective_weight.sizes() == value.sizes(),
        "npq0_s_assign: objective_weight must match value");
    TORCH_CHECK(
        initial_anchor.is_cuda() && initial_anchor.is_contiguous()
            && initial_anchor.scalar_type() == torch::kFloat32
            && initial_anchor.dim() == 1
            && initial_anchor.size(0) == value.size(0),
        "npq0_s_assign: initial_anchor must be CUDA float32 [rows]");
    TORCH_CHECK(
        scale_lut.is_cuda() && scale_lut.is_contiguous()
            && scale_lut.scalar_type() == torch::kFloat32
            && scale_lut.sizes() == torch::IntArrayRef({kStates}),
        "npq0_s_assign: scale_lut must be CUDA float32 [4]");
    TORCH_CHECK(
        first_codebooks.is_cuda() && first_codebooks.is_contiguous()
            && first_codebooks.scalar_type() == torch::kInt8
            && first_codebooks.sizes()
                == torch::IntArrayRef({kStates, kEntries, kSubvector}),
        "npq0_s_assign: first_codebooks must be CUDA int8 [4,8,4]");
    TORCH_CHECK(
        second_codebooks.is_cuda() && second_codebooks.is_contiguous()
            && second_codebooks.scalar_type() == torch::kInt8
            && second_codebooks.sizes() == first_codebooks.sizes(),
        "npq0_s_assign: second_codebooks shape mismatch");
    TORCH_CHECK(
        valid_width > 0 && valid_width <= value.size(1)
            && value.size(1) - valid_width < kGroup,
        "npq0_s_assign: invalid unpadded width");
    TORCH_CHECK(
        refine_steps >= 0 && refine_steps <= 8,
        "npq0_s_assign: refine_steps must be in [0,8]");
    const int device = value.get_device();
    TORCH_CHECK(
        objective_weight.get_device() == device
            && initial_anchor.get_device() == device
            && scale_lut.get_device() == device
            && first_codebooks.get_device() == device
            && second_codebooks.get_device() == device,
        "npq0_s_assign: tensors must share one CUDA device");
}

}  // namespace

std::vector<torch::Tensor> npq0_s_assign_cuda(
    torch::Tensor value,
    torch::Tensor objective_weight,
    torch::Tensor initial_anchor,
    torch::Tensor scale_lut,
    torch::Tensor first_codebooks,
    torch::Tensor second_codebooks,
    int64_t valid_width,
    int64_t refine_steps) {
    check_inputs(
        value,
        objective_weight,
        initial_anchor,
        scale_lut,
        first_codebooks,
        second_codebooks,
        valid_width,
        refine_steps);
    const int64_t rows = value.size(0);
    const int padded_width = static_cast<int>(value.size(1));
    const int groups_per_row = padded_width / kGroup;
    const int64_t groups = rows * groups_per_row;
    auto byte_options = value.options().dtype(torch::kUInt8);
    auto states = torch::empty({rows, groups_per_row}, byte_options);
    auto first_indices = torch::empty(
        {rows, groups_per_row, kVectorsPerGroup}, byte_options);
    auto second_indices = torch::empty_like(first_indices);
    auto group_error = torch::empty({rows, groups_per_row}, value.options());
    auto first_anchor = initial_anchor.contiguous();
    auto second_anchor = torch::empty({rows}, value.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto launch_assign = [&](const torch::Tensor& current_anchor) {
        npq0_s_assign_kernel<<<groups, kAssignThreads, 0, stream>>>(
            value.data_ptr<float>(),
            objective_weight.data_ptr<float>(),
            current_anchor.data_ptr<float>(),
            scale_lut.data_ptr<float>(),
            first_codebooks.data_ptr<int8_t>(),
            second_codebooks.data_ptr<int8_t>(),
            states.data_ptr<uint8_t>(),
            first_indices.data_ptr<uint8_t>(),
            second_indices.data_ptr<uint8_t>(),
            group_error.data_ptr<float>(),
            padded_width,
            static_cast<int>(valid_width),
            groups_per_row);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    auto launch_refit = [&](
        const torch::Tensor& current_anchor,
        torch::Tensor& fitted_anchor) {
        npq0_s_refit_anchor_kernel<<<rows, kRefitThreads, 0, stream>>>(
            value.data_ptr<float>(),
            objective_weight.data_ptr<float>(),
            current_anchor.data_ptr<float>(),
            scale_lut.data_ptr<float>(),
            first_codebooks.data_ptr<int8_t>(),
            second_codebooks.data_ptr<int8_t>(),
            states.data_ptr<uint8_t>(),
            first_indices.data_ptr<uint8_t>(),
            second_indices.data_ptr<uint8_t>(),
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
    auto row_error = group_error.sum(1);
    return {first_anchor, states, first_indices, second_indices, row_error};
}
