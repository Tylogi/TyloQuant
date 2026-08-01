#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kQkxThreads = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_min(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fminf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float quant_level(float value, float minimum, float iscale, int nmax) {
    const float rounded = nearbyintf(iscale * (value - minimum));
    return fminf(static_cast<float>(nmax), fmaxf(0.0f, rounded));
}

__device__ __forceinline__ float positive_quant_level(float value, float iscale, int nmax) {
    const float rounded = nearbyintf(iscale * value);
    return fminf(static_cast<float>(nmax), fmaxf(0.0f, rounded));
}

__global__ void nint_make_qkx3_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out_scale,
    float* __restrict__ out_minimum,
    int64_t groups,
    int group_size,
    int nmax,
    double rmin,
    double rdelta,
    int nstep) {
    // One warp covers up to 64 values by assigning at most two values to each
    // lane. Eight independent groups share a block.
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int64_t group =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (group >= groups) {
        return;
    }

    const int64_t offset = group * group_size;
    const int second_index = lane + kWarpSize;
    const bool valid_first = lane < group_size;
    const bool valid_second = second_index < group_size;
    const float first_value =
        valid_first ? x[offset + lane] : 0.0f;
    const float second_value =
        valid_second ? x[offset + second_index] : 0.0f;
    const float first_weight =
        valid_first ? weight[offset + lane] : 0.0f;
    const float second_weight =
        valid_second ? weight[offset + second_index] : 0.0f;

    float local_minimum = valid_first ? first_value : FLT_MAX;
    float local_maximum = valid_first ? first_value : -FLT_MAX;
    if (valid_second) {
        local_minimum = fminf(local_minimum, second_value);
        local_maximum = fmaxf(local_maximum, second_value);
    }
    float minimum = warp_min(local_minimum);
    float maximum = warp_max(local_maximum);
    float sum_weight = warp_sum(first_weight + second_weight);
    float sum_x = warp_sum(
        first_weight * first_value + second_weight * second_value);
    minimum = __shfl_sync(0xffffffff, minimum, 0);
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    sum_weight = __shfl_sync(0xffffffff, sum_weight, 0);
    sum_x = __shfl_sync(0xffffffff, sum_x, 0);
    minimum = fminf(minimum, 0.0f);

    const bool degenerate = maximum <= minimum || sum_weight <= 0.0f;
    const float range = degenerate ? 1.0f : maximum - minimum;
    // PyTorch's CUDA scalar/tensor division uses the fast float divide.
    // Matching it keeps half-way rounding decisions identical to the
    // reference search.
    const float reciprocal_range = __frcp_rn(range);
    const float initial_iscale =
        static_cast<float>(nmax) * reciprocal_range;
    const float initial_scale = __fdividef(1.0f, initial_iscale);
    const float first_initial_level = valid_first
        ? quant_level(first_value, minimum, initial_iscale, nmax)
        : 0.0f;
    const float second_initial_level = valid_second
        ? quant_level(second_value, minimum, initial_iscale, nmax)
        : 0.0f;
    const float first_initial_diff =
        initial_scale * first_initial_level + minimum - first_value;
    const float second_initial_diff =
        initial_scale * second_initial_level + minimum - second_value;
    float best_error = warp_sum(
        first_weight * first_initial_diff * first_initial_diff
        + second_weight * second_initial_diff * second_initial_diff);
    float best_scale = initial_scale;
    float best_minimum = minimum;

    for (int step = 0; step <= nstep; ++step) {
        const float numerator = static_cast<float>(
            rmin + rdelta * static_cast<double>(step)
            + static_cast<double>(nmax));
        const float iscale = numerator * reciprocal_range;
        const float first_level = valid_first
            ? quant_level(first_value, minimum, iscale, nmax)
            : 0.0f;
        const float second_level = valid_second
            ? quant_level(second_value, minimum, iscale, nmax)
            : 0.0f;
        const float first_weighted_level = first_weight * first_level;
        const float second_weighted_level = second_weight * second_level;
        const float sum_l = warp_sum(
            first_weighted_level + second_weighted_level);
        const float sum_l2 = warp_sum(
            first_weighted_level * first_level
            + second_weighted_level * second_level);
        const float sum_xl = warp_sum(
            first_weighted_level * first_value
            + second_weighted_level * second_value);

        float candidate_scale = 0.0f;
        float candidate_minimum = 0.0f;
        bool candidate_valid = false;
        if (lane == 0) {
            const float determinant =
                sum_weight * sum_l2 - sum_l * sum_l;
            candidate_valid = determinant > 0.0f;
            if (candidate_valid) {
                candidate_scale =
                    (sum_weight * sum_xl - sum_x * sum_l) / determinant;
                candidate_minimum =
                    (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
                if (candidate_minimum > 0.0f) {
                    candidate_scale =
                        sum_l2 > 0.0f ? sum_xl / sum_l2 : 0.0f;
                    candidate_minimum = 0.0f;
                }
            }
        }
        candidate_scale = __shfl_sync(0xffffffff, candidate_scale, 0);
        candidate_minimum = __shfl_sync(0xffffffff, candidate_minimum, 0);
        candidate_valid = __shfl_sync(
            0xffffffff, static_cast<int>(candidate_valid), 0) != 0;
        const float first_candidate_diff =
            candidate_scale * first_level
            + candidate_minimum
            - first_value;
        const float second_candidate_diff =
            candidate_scale * second_level
            + candidate_minimum
            - second_value;
        const float candidate_error = warp_sum(
            first_weight * first_candidate_diff * first_candidate_diff
            + second_weight * second_candidate_diff * second_candidate_diff);
        if (lane == 0 && candidate_valid && candidate_error < best_error) {
            best_error = candidate_error;
            best_scale = candidate_scale;
            best_minimum = candidate_minimum;
        }
    }

    if (lane == 0) {
        out_scale[group] = degenerate ? 0.0f : best_scale;
        out_minimum[group] = degenerate ? fminf(minimum, 0.0f) : best_minimum;
    }
}

__global__ void nint_make_qp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out_scale,
    int32_t* __restrict__ levels,
    int width,
    int nmax) {
    // Search reductions use the full warp. The five coordinate-refinement
    // passes are order-dependent and therefore remain serial on lane zero.
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * width;

    float local_maximum = -FLT_MAX;
    for (int index = lane; index < width; index += kWarpSize) {
        local_maximum = fmaxf(local_maximum, x[row_offset + index]);
    }
    const float maximum = __shfl_sync(
        0xffffffff, warp_max(local_maximum), 0);
    const bool active = maximum >= 1e-15f;
    const float safe_maximum = active ? maximum : 1.0f;
    const float reciprocal_maximum = __frcp_rn(safe_maximum);
    float iscale =
        static_cast<float>(nmax) * reciprocal_maximum;
    float scale = __fdividef(1.0f, iscale);

    float local_error = 0.0f;
    for (int index = lane; index < width; index += kWarpSize) {
        const float value = x[row_offset + index];
        const float level = positive_quant_level(value, iscale, nmax);
        const float difference = value - scale * level;
        local_error += weight[row_offset + index] * difference * difference;
    }
    float best_error = warp_sum(local_error);

    for (int offset = -4; offset <= 4; ++offset) {
        if (offset == 0) {
            continue;
        }
        const float candidate_numerator = static_cast<float>(
            static_cast<double>(nmax) + 0.1 * static_cast<double>(offset));
        const float candidate_iscale =
            candidate_numerator * reciprocal_maximum;
        const float candidate_scale = __fdividef(1.0f, candidate_iscale);
        float candidate_local_error = 0.0f;
        for (int index = lane; index < width; index += kWarpSize) {
            const float value = x[row_offset + index];
            const float level =
                positive_quant_level(value, candidate_iscale, nmax);
            const float difference = value - candidate_scale * level;
            candidate_local_error +=
                weight[row_offset + index] * difference * difference;
        }
        const float candidate_error = warp_sum(candidate_local_error);
        if (lane == 0 && active && candidate_error < best_error) {
            best_error = candidate_error;
            iscale = candidate_iscale;
        }
        iscale = __shfl_sync(0xffffffff, iscale, 0);
    }

    float local_lx = 0.0f;
    float local_l2 = 0.0f;
    for (int index = lane; index < width; index += kWarpSize) {
        const int64_t position = row_offset + index;
        const float value = x[position];
        const int32_t level = active
            ? static_cast<int32_t>(positive_quant_level(value, iscale, nmax))
            : 0;
        levels[position] = level;
        const float level_f = static_cast<float>(level);
        local_lx += weight[position] * value * level_f;
        local_l2 += weight[position] * level_f * level_f;
    }
    float sum_lx = warp_sum(local_lx);
    float sum_l2 = warp_sum(local_l2);
    __syncwarp();

    if (lane == 0) {
        for (int pass = 0; pass < 5; ++pass) {
            for (int index = 0; index < width; ++index) {
                const int64_t position = row_offset + index;
                const int32_t old_level = levels[position];
                const float old_level_f = static_cast<float>(old_level);
                const float objective_weight = weight[position];
                const float value = x[position];
                const float candidate_lx =
                    sum_lx - objective_weight * value * old_level_f;
                const float candidate_l2 =
                    sum_l2 - objective_weight * old_level_f * old_level_f;
                if (candidate_lx <= 0.0f || candidate_l2 <= 0.0f) {
                    continue;
                }
                const int32_t new_level = static_cast<int32_t>(
                    positive_quant_level(
                        value, candidate_l2 / candidate_lx, nmax));
                if (new_level == old_level) {
                    continue;
                }
                const float new_level_f = static_cast<float>(new_level);
                const float updated_lx =
                    candidate_lx
                    + objective_weight * value * new_level_f;
                const float updated_l2 =
                    candidate_l2
                    + objective_weight * new_level_f * new_level_f;
                if (
                    updated_lx * updated_lx * sum_l2
                    > sum_lx * sum_lx * updated_l2) {
                    levels[position] = new_level;
                    sum_lx = updated_lx;
                    sum_l2 = updated_l2;
                }
            }
        }
        out_scale[row] =
            active && sum_l2 > 0.0f ? sum_lx / sum_l2 : 0.0f;
    }
}

void check_float_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.is_contiguous(),
        name,
        " must be CUDA contiguous");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32,
        name,
        " must be float32");
}

}  // namespace

std::vector<torch::Tensor> nint_make_qkx3_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t nmax,
    double rmin,
    double rdelta,
    int64_t nstep) {
    check_float_cuda_contiguous(x, "nint_make_qkx3: x");
    check_float_cuda_contiguous(weight, "nint_make_qkx3: weight");
    TORCH_CHECK(x.dim() == 3, "nint_make_qkx3: x must have shape [rows, groups, width]");
    TORCH_CHECK(weight.sizes() == x.sizes(), "nint_make_qkx3: weight shape mismatch");
    TORCH_CHECK(x.get_device() == weight.get_device(), "nint_make_qkx3: tensors must share one CUDA device");
    TORCH_CHECK(x.size(2) > 0 && x.size(2) <= 2 * kWarpSize, "nint_make_qkx3: group width must be in [1, 64]");
    TORCH_CHECK(nmax > 0 && nmax <= 255, "nint_make_qkx3: nmax must be in [1, 255]");
    TORCH_CHECK(nstep >= 0, "nint_make_qkx3: nstep must be non-negative");

    const int64_t groups = x.numel() / x.size(2);
    auto scale = torch::empty({x.size(0), x.size(1)}, x.options());
    auto minimum = torch::empty_like(scale);
    const int64_t blocks =
        (groups + kWarpsPerBlock - 1) / kWarpsPerBlock;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nint_make_qkx3_kernel<<<blocks, kQkxThreads, 0, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        scale.data_ptr<float>(),
        minimum.data_ptr<float>(),
        groups,
        static_cast<int>(x.size(2)),
        static_cast<int>(nmax),
        rmin,
        rdelta,
        static_cast<int>(nstep));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scale, minimum};
}

std::vector<torch::Tensor> nint_make_qp_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t nmax) {
    check_float_cuda_contiguous(x, "nint_make_qp: x");
    check_float_cuda_contiguous(weight, "nint_make_qp: weight");
    TORCH_CHECK(x.dim() == 2, "nint_make_qp: x must have shape [rows, width]");
    TORCH_CHECK(weight.sizes() == x.sizes(), "nint_make_qp: weight shape mismatch");
    TORCH_CHECK(x.get_device() == weight.get_device(), "nint_make_qp: tensors must share one CUDA device");
    TORCH_CHECK(x.size(1) > 0, "nint_make_qp: width must be positive");
    TORCH_CHECK(nmax > 0 && nmax <= 255, "nint_make_qp: nmax must be in [1, 255]");

    auto scale = torch::empty({x.size(0)}, x.options());
    auto levels = torch::empty(
        x.sizes(), x.options().dtype(torch::kInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nint_make_qp_kernel<<<x.size(0), kWarpSize, 0, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        scale.data_ptr<float>(),
        levels.data_ptr<int32_t>(),
        static_cast<int>(x.size(1)),
        static_cast<int>(nmax));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scale, levels};
}
