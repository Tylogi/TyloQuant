#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kCodebookEntries = 2048;
constexpr int kVectorSize = 8;
constexpr int kVectorsPerGroup = 3;
constexpr int kGroupSize = kVectorSize * kVectorsPerGroup;

__device__ __forceinline__ bool better(float value, int index, float best, int best_index) {
    return value < best || (value == best && index < best_index);
}

template <int QCount>
__global__ void nvq1_l_assign_kernel(
    const float* __restrict__ xgroup,
    const float* __restrict__ wgroup,
    const float* __restrict__ group_anchor,
    const int8_t* __restrict__ codebook,
    uint8_t* __restrict__ out_scale,
    uint8_t* __restrict__ out_delta,
    int64_t* __restrict__ out_indices,
    int groups_per_row,
    int valid_last,
    float delta_value) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;

    extern __shared__ unsigned char shared_raw[];
    int8_t* shared_codebook = reinterpret_cast<int8_t*>(shared_raw);
    size_t offset = kCodebookEntries * kVectorSize * sizeof(int8_t);
    offset = (offset + alignof(float) - 1) & ~(alignof(float) - 1);
    float* shared_error = reinterpret_cast<float*>(shared_raw + offset);
    offset += QCount * kThreads * sizeof(float);
    uint16_t* shared_index = reinterpret_cast<uint16_t*>(shared_raw + offset);

    __shared__ float vector_best_error[2][kVectorsPerGroup][QCount];
    __shared__ uint16_t vector_best_index[2][kVectorsPerGroup][QCount];

    for (int i = tid; i < kCodebookEntries * kVectorSize; i += kThreads) {
        shared_codebook[i] = codebook[i];
    }
    __syncthreads();

    int valid_group = kGroupSize;
    if (group % groups_per_row == groups_per_row - 1) {
        valid_group = valid_last;
    }
    const float anchor = group_anchor[group];

    for (int delta_bit = 0; delta_bit < 2; ++delta_bit) {
        const float delta = delta_bit == 0 ? delta_value : -delta_value;
        for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
            int valid = valid_group - vector * kVectorSize;
            valid = valid < 0 ? 0 : (valid > kVectorSize ? kVectorSize : valid);
            float x[kVectorSize];
#pragma unroll
            for (int coordinate = 0; coordinate < kVectorSize; ++coordinate) {
                x[coordinate] = coordinate < valid
                    ? xgroup[static_cast<int64_t>(group) * kGroupSize
                             + vector * kVectorSize + coordinate]
                    : 0.0f;
            }

            float local_error[QCount];
            int local_index[QCount];
#pragma unroll
            for (int q = 0; q < QCount; ++q) {
                local_error[q] = FLT_MAX;
                local_index[q] = 0;
            }

            for (int code_index = tid; code_index < kCodebookEntries; code_index += kThreads) {
                float dot = 0.0f;
                float norm = 0.0f;
#pragma unroll
                for (int coordinate = 0; coordinate < kVectorSize; ++coordinate) {
                    if (coordinate < valid) {
                        const float weight = wgroup[
                            static_cast<int64_t>(group) * kGroupSize
                            + vector * kVectorSize + coordinate];
                        const float code = static_cast<float>(
                            shared_codebook[code_index * kVectorSize + coordinate]) + delta;
                        dot = fmaf(weight * x[coordinate], code, dot);
                        norm = fmaf(weight * code, code, norm);
                    }
                }
#pragma unroll
                for (int q = 0; q < QCount; ++q) {
                    const float scale = anchor * static_cast<float>(q);
                    const float error = fmaf(scale * scale, norm, -2.0f * scale * dot);
                    if (better(error, code_index, local_error[q], local_index[q])) {
                        local_error[q] = error;
                        local_index[q] = code_index;
                    }
                }
            }

#pragma unroll
            for (int q = 0; q < QCount; ++q) {
                shared_error[q * kThreads + tid] = local_error[q];
                shared_index[q * kThreads + tid] = static_cast<uint16_t>(local_index[q]);
            }
            __syncthreads();

            for (int q = warp; q < QCount; q += kThreads / kWarpSize) {
                float best_error = FLT_MAX;
                int best_index = 0;
                for (int source = lane; source < kThreads; source += kWarpSize) {
                    const float value = shared_error[q * kThreads + source];
                    const int index = static_cast<int>(shared_index[q * kThreads + source]);
                    if (better(value, index, best_error, best_index)) {
                        best_error = value;
                        best_index = index;
                    }
                }
                for (int shift = kWarpSize / 2; shift > 0; shift >>= 1) {
                    const float other_error = __shfl_down_sync(0xffffffff, best_error, shift);
                    const int other_index = __shfl_down_sync(0xffffffff, best_index, shift);
                    if (lane < shift && better(other_error, other_index, best_error, best_index)) {
                        best_error = other_error;
                        best_index = other_index;
                    }
                }
                if (lane == 0) {
                    vector_best_error[delta_bit][vector][q] = best_error;
                    vector_best_index[delta_bit][vector][q] = static_cast<uint16_t>(best_index);
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        float best_error = FLT_MAX;
        int best_delta = 0;
        int best_q = 0;
        for (int delta_bit = 0; delta_bit < 2; ++delta_bit) {
            for (int q = 0; q < QCount; ++q) {
                float error = 0.0f;
#pragma unroll
                for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
                    error += vector_best_error[delta_bit][vector][q];
                }
                if (error < best_error) {
                    best_error = error;
                    best_delta = delta_bit;
                    best_q = q;
                }
            }
        }
        out_scale[group] = static_cast<uint8_t>(best_q);
        out_delta[group] = static_cast<uint8_t>(best_delta);
#pragma unroll
        for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
            out_indices[static_cast<int64_t>(group) * kVectorsPerGroup + vector] =
                static_cast<int64_t>(vector_best_index[best_delta][vector][best_q]);
        }
    }
}

template <int VectorSize, int VectorsPerGroup, int CodebookEntries>
__device__ __forceinline__ void nvq_assign_vectors(
    const float* __restrict__ group_values,
    const float* __restrict__ group_weights,
    const int8_t* __restrict__ shared_codebook,
    float scale,
    int valid_group,
    int* __restrict__ selected) {
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    if (warp < VectorsPerGroup) {
        int valid = valid_group - warp * VectorSize;
        valid = valid < 0 ? 0 : (valid > VectorSize ? VectorSize : valid);
        float best_error = FLT_MAX;
        int best_index = 0;
        for (
            int code_index = lane;
            code_index < CodebookEntries;
            code_index += kWarpSize) {
            float dot = 0.0f;
            float norm = 0.0f;
#pragma unroll
            for (int coordinate = 0; coordinate < VectorSize; ++coordinate) {
                if (coordinate < valid) {
                    const float code = static_cast<float>(
                        shared_codebook[code_index * VectorSize + coordinate]);
                    const float value = group_values[warp * VectorSize + coordinate];
                    const float weight = group_weights[warp * VectorSize + coordinate];
                    dot = fmaf(weight * value, code, dot);
                    norm = fmaf(weight * code, code, norm);
                }
            }
            const float error = fmaf(scale * scale, norm, -2.0f * scale * dot);
            if (better(error, code_index, best_error, best_index)) {
                best_error = error;
                best_index = code_index;
            }
        }
        for (int shift = kWarpSize / 2; shift > 0; shift >>= 1) {
            const float other_error = __shfl_down_sync(0xffffffff, best_error, shift);
            const int other_index = __shfl_down_sync(0xffffffff, best_index, shift);
            if (lane < shift && better(other_error, other_index, best_error, best_index)) {
                best_error = other_error;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            selected[warp] = best_index;
        }
    }
    __syncthreads();
}

template <int VectorSize, int VectorsPerGroup>
__device__ __forceinline__ float nvq_refit_scale(
    const float* __restrict__ group_values,
    const float* __restrict__ group_weights,
    const int8_t* __restrict__ shared_codebook,
    const int* __restrict__ selected,
    int valid_group) {
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int position = 0; position < valid_group; ++position) {
        const int vector = position / VectorSize;
        const int coordinate = position % VectorSize;
        const float code = static_cast<float>(
            shared_codebook[selected[vector] * VectorSize + coordinate]);
        const float weight = group_weights[position];
        numerator = fmaf(weight * group_values[position], code, numerator);
        denominator = fmaf(weight * code, code, denominator);
    }
    return denominator > 0.0f ? fmaxf(numerator / denominator, 0.0f) : 0.0f;
}

template <int VectorSize, int VectorsPerGroup, int CodebookEntries>
__global__ void nvq_search_kernel(
    const float* __restrict__ xgroup,
    const float* __restrict__ wgroup,
    const int8_t* __restrict__ codebook,
    float* __restrict__ out_scale,
    int64_t* __restrict__ out_indices,
    int groups_per_row,
    int valid_last,
    int search_steps,
    float qmax) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int8_t shared_codebook[CodebookEntries * VectorSize];
    __shared__ int selected[VectorsPerGroup];
    __shared__ int best_indices[VectorsPerGroup];
    __shared__ float current_scale;
    __shared__ float best_scale;
    __shared__ float best_error;

    for (int i = tid; i < CodebookEntries * VectorSize; i += kThreads) {
        shared_codebook[i] = codebook[i];
    }
    int valid_group = kGroupSize;
    if (group % groups_per_row == groups_per_row - 1) {
        valid_group = valid_last;
    }
    const float* group_values = xgroup + static_cast<int64_t>(group) * kGroupSize;
    const float* group_weights = wgroup + static_cast<int64_t>(group) * kGroupSize;
    if (tid == 0) {
        best_error = FLT_MAX;
        best_scale = 0.0f;
    }
    __syncthreads();

    for (int step = 0; step < search_steps; ++step) {
        if (tid == 0) {
            float max_abs = 0.0f;
            for (int position = 0; position < valid_group; ++position) {
                max_abs = fmaxf(max_abs, fabsf(group_values[position]));
            }
            const float offset = search_steps == 1
                ? -0.12f * qmax
                : -0.12f * qmax
                    + static_cast<float>(step) * (0.24f * qmax / (search_steps - 1));
            current_scale = max_abs > 0.0f ? max_abs / (qmax + offset) : 0.0f;
        }
        __syncthreads();
        nvq_assign_vectors<VectorSize, VectorsPerGroup, CodebookEntries>(
            group_values, group_weights, shared_codebook,
            current_scale, valid_group, selected);
        if (tid == 0) {
            current_scale = nvq_refit_scale<VectorSize, VectorsPerGroup>(
                group_values, group_weights, shared_codebook, selected, valid_group);
        }
        __syncthreads();
        nvq_assign_vectors<VectorSize, VectorsPerGroup, CodebookEntries>(
            group_values, group_weights, shared_codebook,
            current_scale, valid_group, selected);
        if (tid == 0) {
            current_scale = nvq_refit_scale<VectorSize, VectorsPerGroup>(
                group_values, group_weights, shared_codebook, selected, valid_group);
            float error = 0.0f;
            for (int position = 0; position < valid_group; ++position) {
                const int vector = position / VectorSize;
                const int coordinate = position % VectorSize;
                const float code = static_cast<float>(
                    shared_codebook[selected[vector] * VectorSize + coordinate]);
                const float residual = current_scale * code - group_values[position];
                error = fmaf(group_weights[position] * residual, residual, error);
            }
            if (error < best_error) {
                best_error = error;
                best_scale = current_scale;
#pragma unroll
                for (int vector = 0; vector < VectorsPerGroup; ++vector) {
                    best_indices[vector] = selected[vector];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_scale[group] = best_scale;
#pragma unroll
        for (int vector = 0; vector < VectorsPerGroup; ++vector) {
            out_indices[static_cast<int64_t>(group) * VectorsPerGroup + vector] =
                static_cast<int64_t>(best_indices[vector]);
        }
    }
}

template <int VectorSize, int VectorsPerGroup, int CodebookEntries>
__global__ void nvq_reassign_kernel(
    const float* __restrict__ xgroup,
    const float* __restrict__ wgroup,
    const float* __restrict__ scale,
    const int8_t* __restrict__ codebook,
    int64_t* __restrict__ out_indices,
    int groups_per_row,
    int valid_last) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int8_t shared_codebook[CodebookEntries * VectorSize];
    __shared__ int selected[VectorsPerGroup];
    for (int i = tid; i < CodebookEntries * VectorSize; i += kThreads) {
        shared_codebook[i] = codebook[i];
    }
    int valid_group = kGroupSize;
    if (group % groups_per_row == groups_per_row - 1) {
        valid_group = valid_last;
    }
    __syncthreads();
    nvq_assign_vectors<VectorSize, VectorsPerGroup, CodebookEntries>(
        xgroup + static_cast<int64_t>(group) * kGroupSize,
        wgroup + static_cast<int64_t>(group) * kGroupSize,
        shared_codebook,
        scale[group],
        valid_group,
        selected);
    if (tid == 0) {
#pragma unroll
        for (int vector = 0; vector < VectorsPerGroup; ++vector) {
            out_indices[static_cast<int64_t>(group) * VectorsPerGroup + vector] =
                static_cast<int64_t>(selected[vector]);
        }
    }
}

}  // namespace

std::vector<torch::Tensor> nvq1_l_assign_cuda(
    torch::Tensor xgroup,
    torch::Tensor wgroup,
    torch::Tensor group_anchor,
    torch::Tensor codebook,
    int64_t groups_per_row,
    int64_t valid_last,
    int64_t sub_bits,
    double delta) {
    TORCH_CHECK(xgroup.is_cuda() && xgroup.is_contiguous(),
                "nvq1_l_assign: xgroup must be CUDA contiguous");
    TORCH_CHECK(wgroup.is_cuda() && wgroup.is_contiguous(),
                "nvq1_l_assign: wgroup must be CUDA contiguous");
    TORCH_CHECK(group_anchor.is_cuda() && group_anchor.is_contiguous(),
                "nvq1_l_assign: group_anchor must be CUDA contiguous");
    TORCH_CHECK(codebook.is_cuda() && codebook.is_contiguous(),
                "nvq1_l_assign: codebook must be CUDA contiguous");
    TORCH_CHECK(xgroup.scalar_type() == torch::kFloat32,
                "nvq1_l_assign: xgroup must be float32");
    TORCH_CHECK(wgroup.scalar_type() == torch::kFloat32,
                "nvq1_l_assign: wgroup must be float32");
    TORCH_CHECK(group_anchor.scalar_type() == torch::kFloat32,
                "nvq1_l_assign: group_anchor must be float32");
    TORCH_CHECK(codebook.scalar_type() == torch::kInt8,
                "nvq1_l_assign: codebook must be int8");
    TORCH_CHECK(xgroup.dim() == 2 && xgroup.size(1) == kGroupSize,
                "nvq1_l_assign: xgroup must have shape [groups, 24]");
    TORCH_CHECK(wgroup.sizes() == xgroup.sizes(),
                "nvq1_l_assign: wgroup shape mismatch");
    TORCH_CHECK(group_anchor.dim() == 1 && group_anchor.size(0) == xgroup.size(0),
                "nvq1_l_assign: group_anchor shape mismatch");
    TORCH_CHECK(codebook.sizes() == torch::IntArrayRef({kCodebookEntries, kVectorSize}),
                "nvq1_l_assign: codebook must have shape [2048, 8]");
    TORCH_CHECK(groups_per_row > 0, "nvq1_l_assign: groups_per_row must be positive");
    TORCH_CHECK(valid_last > 0 && valid_last <= kGroupSize,
                "nvq1_l_assign: valid_last must be in [1, 24]");
    TORCH_CHECK(sub_bits == 3 || sub_bits == 4,
                "nvq1_l_assign: sub_bits must be 3 or 4");
    TORCH_CHECK(xgroup.get_device() == wgroup.get_device()
                    && xgroup.get_device() == group_anchor.get_device()
                    && xgroup.get_device() == codebook.get_device(),
                "nvq1_l_assign: tensors must share one CUDA device");

    const auto groups = xgroup.size(0);
    auto byte_options = xgroup.options().dtype(torch::kUInt8);
    auto long_options = xgroup.options().dtype(torch::kInt64);
    auto out_scale = torch::empty({groups}, byte_options);
    auto out_delta = torch::empty({groups}, byte_options);
    auto out_indices = torch::empty({groups, kVectorsPerGroup}, long_options);
    const int q_count = 1 << sub_bits;
    const size_t shared_bytes =
        kCodebookEntries * kVectorSize * sizeof(int8_t)
        + q_count * kThreads * (sizeof(float) + sizeof(uint16_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (sub_bits == 3) {
        nvq1_l_assign_kernel<8><<<groups, kThreads, shared_bytes, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            group_anchor.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            out_scale.data_ptr<uint8_t>(),
            out_delta.data_ptr<uint8_t>(),
            out_indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last),
            static_cast<float>(delta));
    } else {
        nvq1_l_assign_kernel<16><<<groups, kThreads, shared_bytes, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            group_anchor.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            out_scale.data_ptr<uint8_t>(),
            out_delta.data_ptr<uint8_t>(),
            out_indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last),
            static_cast<float>(delta));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_scale, out_delta, out_indices};
}

namespace {

void check_nvq_inputs(
    const torch::Tensor& xgroup,
    const torch::Tensor& wgroup,
    const torch::Tensor& codebook,
    int64_t groups_per_row,
    int64_t valid_last,
    int64_t vector_size) {
    TORCH_CHECK(xgroup.is_cuda() && xgroup.is_contiguous(),
                "nvq_quant: xgroup must be CUDA contiguous");
    TORCH_CHECK(wgroup.is_cuda() && wgroup.is_contiguous(),
                "nvq_quant: wgroup must be CUDA contiguous");
    TORCH_CHECK(codebook.is_cuda() && codebook.is_contiguous(),
                "nvq_quant: codebook must be CUDA contiguous");
    TORCH_CHECK(xgroup.scalar_type() == torch::kFloat32,
                "nvq_quant: xgroup must be float32");
    TORCH_CHECK(wgroup.scalar_type() == torch::kFloat32,
                "nvq_quant: wgroup must be float32");
    TORCH_CHECK(codebook.scalar_type() == torch::kInt8,
                "nvq_quant: codebook must be int8");
    TORCH_CHECK(xgroup.dim() == 2 && xgroup.size(1) == kGroupSize,
                "nvq_quant: xgroup must have shape [groups, 24]");
    TORCH_CHECK(xgroup.size(0) > 0,
                "nvq_quant: xgroup must contain at least one group");
    TORCH_CHECK(wgroup.sizes() == xgroup.sizes(),
                "nvq_quant: wgroup shape mismatch");
    TORCH_CHECK(vector_size == 4 || vector_size == 8,
                "nvq_quant: vector_size must be 4 or 8");
    TORCH_CHECK(codebook.dim() == 2
                    && (codebook.size(0) == 256
                        || codebook.size(0) == 512
                        || codebook.size(0) == 1024
                        || codebook.size(0) == 4096)
                    && codebook.size(1) == vector_size,
                "nvq_quant: codebook shape mismatch");
    TORCH_CHECK(
        (vector_size == 8
         && (codebook.size(0) == 256
             || codebook.size(0) == 1024
             || codebook.size(0) == 4096))
        || (vector_size == 4
            && (codebook.size(0) == 256
                || codebook.size(0) == 512
                || codebook.size(0) == 1024)),
        "nvq_quant: unsupported vector-size/codebook-size pair");
    TORCH_CHECK(groups_per_row > 0, "nvq_quant: groups_per_row must be positive");
    TORCH_CHECK(valid_last > 0 && valid_last <= kGroupSize,
                "nvq_quant: valid_last must be in [1, 24]");
    TORCH_CHECK(xgroup.get_device() == wgroup.get_device()
                    && xgroup.get_device() == codebook.get_device(),
                "nvq_quant: tensors must share one CUDA device");
}

}  // namespace

std::vector<torch::Tensor> nvq_search_cuda(
    torch::Tensor xgroup,
    torch::Tensor wgroup,
    torch::Tensor codebook,
    int64_t groups_per_row,
    int64_t valid_last,
    int64_t vector_size,
    int64_t search_steps,
    double qmax) {
    check_nvq_inputs(xgroup, wgroup, codebook, groups_per_row, valid_last, vector_size);
    TORCH_CHECK(search_steps > 0, "nvq_search: search_steps must be positive");
    TORCH_CHECK(qmax > 0.0, "nvq_search: qmax must be positive");
    const auto groups = xgroup.size(0);
    const int vectors_per_group = kGroupSize / static_cast<int>(vector_size);
    auto scales = torch::empty({groups}, xgroup.options());
    auto indices = torch::empty(
        {groups, vectors_per_group}, xgroup.options().dtype(torch::kInt64));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (vector_size == 8 && codebook.size(0) == 4096) {
        nvq_search_kernel<8, 3, 4096><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last),
            static_cast<int>(search_steps),
            static_cast<float>(qmax));
    } else if (vector_size == 8 && codebook.size(0) == 1024) {
        nvq_search_kernel<8, 3, 1024><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last),
            static_cast<int>(search_steps),
            static_cast<float>(qmax));
    } else if (vector_size == 8) {
        nvq_search_kernel<8, 3, 256><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last),
            static_cast<int>(search_steps),
            static_cast<float>(qmax));
    } else if (codebook.size(0) == 1024) {
        nvq_search_kernel<4, 6, 1024><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last),
            static_cast<int>(search_steps),
            static_cast<float>(qmax));
    } else if (codebook.size(0) == 512) {
        nvq_search_kernel<4, 6, 512><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last),
            static_cast<int>(search_steps),
            static_cast<float>(qmax));
    } else {
        nvq_search_kernel<4, 6, 256><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            scales.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last),
            static_cast<int>(search_steps),
            static_cast<float>(qmax));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scales, indices};
}

torch::Tensor nvq_reassign_cuda(
    torch::Tensor xgroup,
    torch::Tensor wgroup,
    torch::Tensor scale,
    torch::Tensor codebook,
    int64_t groups_per_row,
    int64_t valid_last,
    int64_t vector_size) {
    check_nvq_inputs(xgroup, wgroup, codebook, groups_per_row, valid_last, vector_size);
    TORCH_CHECK(scale.is_cuda() && scale.is_contiguous()
                    && scale.scalar_type() == torch::kFloat32,
                "nvq_reassign: scale must be CUDA contiguous float32");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == xgroup.size(0),
                "nvq_reassign: scale shape mismatch");
    TORCH_CHECK(scale.get_device() == xgroup.get_device(),
                "nvq_reassign: tensors must share one CUDA device");
    const auto groups = xgroup.size(0);
    const int vectors_per_group = kGroupSize / static_cast<int>(vector_size);
    auto indices = torch::empty(
        {groups, vectors_per_group}, xgroup.options().dtype(torch::kInt64));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (vector_size == 8 && codebook.size(0) == 4096) {
        nvq_reassign_kernel<8, 3, 4096><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last));
    } else if (vector_size == 8 && codebook.size(0) == 1024) {
        nvq_reassign_kernel<8, 3, 1024><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last));
    } else if (vector_size == 8) {
        nvq_reassign_kernel<8, 3, 256><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last));
    } else if (codebook.size(0) == 1024) {
        nvq_reassign_kernel<4, 6, 1024><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last));
    } else if (codebook.size(0) == 512) {
        nvq_reassign_kernel<4, 6, 512><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last));
    } else {
        nvq_reassign_kernel<4, 6, 256><<<groups, kThreads, 0, stream>>>(
            xgroup.data_ptr<float>(),
            wgroup.data_ptr<float>(),
            scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(),
            indices.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            static_cast<int>(valid_last));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return indices;
}
