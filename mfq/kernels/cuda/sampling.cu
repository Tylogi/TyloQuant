// GPU logits sampling helpers.
// Greedy uses a parallel block reduction. Stochastic full-softmax keeps logits on GPU;
// random uniforms are passed as a small GPU tensor.

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cub/block/block_radix_sort.cuh>
#include <float.h>
#include <climits>
#include <cstdlib>
#include <cstdint>

constexpr int SAMPLE_BD = 256;
constexpr int SAMPLE_MAX_TOP_K = 1024;
constexpr int SAMPLE_RADIX_MAX_TOP_K = 64;

__device__ __forceinline__ uint64_t sample_make_sort_key(float value, int index)
{
    if (isnan(value)) value = -FLT_MAX;
    const uint32_t raw = __float_as_uint(value);
    const uint32_t ordered = raw ^ ((raw & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    return ((uint64_t)ordered << 32) | (uint64_t)(0xffffffffu - (uint32_t)index);
}

__device__ __forceinline__ float sample_sort_key_value(uint64_t key)
{
    const uint32_t ordered = (uint32_t)(key >> 32);
    const uint32_t raw = ordered ^ ((ordered & 0x80000000u) ? 0x80000000u : 0xffffffffu);
    return __uint_as_float(raw);
}

template <typename scalar_t>
__global__ void sample_top_k_radix_stage1_kernel(
    const scalar_t* __restrict__ logits,
    float* __restrict__ out_vals,
    int* __restrict__ out_idxs,
    int V,
    int top_k)
{
    using BlockSort = cub::BlockRadixSort<uint64_t, SAMPLE_BD, 1, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    const int i = blockIdx.x * SAMPLE_BD + threadIdx.x;
    const int index = i < V ? i : INT_MAX;
    const float value = i < V ? (float)logits[i] : -FLT_MAX;
    uint64_t keys[1] = {i < V ? sample_make_sort_key(value, index) : 0};
    int sorted_indices[1] = {index};
    BlockSort(sort_storage).SortDescending(keys, sorted_indices);
    if (threadIdx.x < top_k) {
        const int out = blockIdx.x * top_k + threadIdx.x;
        out_vals[out] = sorted_indices[0] == INT_MAX ? -FLT_MAX : sample_sort_key_value(keys[0]);
        out_idxs[out] = sorted_indices[0];
    }
}

__global__ void sample_top_k_radix_reduce_kernel(
    const float* __restrict__ in_vals,
    const int* __restrict__ in_idxs,
    float* __restrict__ out_vals,
    int* __restrict__ out_idxs,
    int count,
    int top_k)
{
    using BlockSort = cub::BlockRadixSort<uint64_t, SAMPLE_BD, 1, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    const int i = blockIdx.x * SAMPLE_BD + threadIdx.x;
    const int index = i < count ? in_idxs[i] : INT_MAX;
    const float value = i < count ? in_vals[i] : -FLT_MAX;
    uint64_t keys[1] = {i < count && index != INT_MAX ? sample_make_sort_key(value, index) : 0};
    int sorted_indices[1] = {index};
    BlockSort(sort_storage).SortDescending(keys, sorted_indices);
    if (threadIdx.x < top_k) {
        const int out = blockIdx.x * top_k + threadIdx.x;
        out_vals[out] = sorted_indices[0] == INT_MAX ? -FLT_MAX : sample_sort_key_value(keys[0]);
        out_idxs[out] = sorted_indices[0];
    }
}

__global__ void sample_top_k_radix_finish_kernel(
    const float* __restrict__ in_vals,
    const int* __restrict__ in_idxs,
    const float* __restrict__ random,
    int64_t* __restrict__ out,
    int count,
    int top_k,
    float temperature,
    float top_p)
{
    using BlockSort = cub::BlockRadixSort<uint64_t, SAMPLE_BD, 1, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    __shared__ float top_vals[SAMPLE_RADIX_MAX_TOP_K];
    __shared__ int top_idxs[SAMPLE_RADIX_MAX_TOP_K];
    const int i = threadIdx.x;
    const int index = i < count ? in_idxs[i] : INT_MAX;
    const float value = i < count ? in_vals[i] : -FLT_MAX;
    uint64_t keys[1] = {i < count && index != INT_MAX ? sample_make_sort_key(value, index) : 0};
    int sorted_indices[1] = {index};
    BlockSort(sort_storage).SortDescending(keys, sorted_indices);
    if (threadIdx.x < top_k) {
        top_vals[threadIdx.x] = sample_sort_key_value(keys[0]) / temperature;
        top_idxs[threadIdx.x] = sorted_indices[0];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const float max_v = top_vals[0];
    float probs[SAMPLE_RADIX_MAX_TOP_K];
    float sum = 0.0f;
    for (int j = 0; j < top_k; ++j) {
        probs[j] = expf(top_vals[j] - max_v);
        sum += probs[j];
    }
    int keep = top_k;
    float keep_sum = sum;
    if (top_p > 0.0f && top_p < 1.0f) {
        const float cutoff = top_p * sum;
        float cumulative = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            cumulative += probs[j];
            if (cumulative >= cutoff) {
                keep = j + 1;
                keep_sum = cumulative;
                break;
            }
        }
    }
    const float u = fminf(fmaxf(random[0], 0.0f), 0.99999994f);
    const float target = u * keep_sum;
    float cumulative = 0.0f;
    int chosen = top_idxs[keep - 1];
    for (int j = 0; j < keep; ++j) {
        cumulative += probs[j];
        if (cumulative >= target) {
            chosen = top_idxs[j];
            break;
        }
    }
    out[0] = chosen;
}

__global__ void sample_token_counts_add_kernel(
    int32_t* __restrict__ counts,
    const int64_t* __restrict__ tokens,
    int64_t n,
    int vocab_size)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t token = tokens[i];
    if (token >= 0 && token < vocab_size) atomicAdd(counts + token, 1);
}

template <typename scalar_t>
__global__ void sample_apply_penalties_kernel(
    scalar_t* __restrict__ logits,
    const int32_t* __restrict__ counts,
    int vocab_size,
    float presence_penalty,
    float frequency_penalty,
    float repetition_penalty)
{
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= vocab_size) return;
    int count = counts[token];
    if (count == 0) return;
    float value = (float)logits[token];
    if (repetition_penalty != 1.0f) {
        value = value < 0.0f ? value * repetition_penalty : value / repetition_penalty;
    }
    value -= presence_penalty + frequency_penalty * count;
    logits[token] = (scalar_t)value;
}

template <typename scalar_t>
__device__ inline float sample_load(const scalar_t* p, size_t i)
{
    return (float)p[i];
}

template <typename scalar_t>
__global__ void sample_greedy_kernel(
    const scalar_t* __restrict__ logits,
    int64_t* __restrict__ out,
    int B,
    int V)
{
    int row = blockIdx.x;
    __shared__ float vals[SAMPLE_BD];
    __shared__ int idxs[SAMPLE_BD];
    float best = -FLT_MAX;
    int best_i = 0;
    for (int i = threadIdx.x; i < V; i += SAMPLE_BD) {
        float v = sample_load(logits, (size_t)row * V + i);
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    vals[threadIdx.x] = best;
    idxs[threadIdx.x] = best_i;
    __syncthreads();
    for (int stride = SAMPLE_BD / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float ov = vals[threadIdx.x + stride];
            int oi = idxs[threadIdx.x + stride];
            if (ov > vals[threadIdx.x] || (ov == vals[threadIdx.x] && oi < idxs[threadIdx.x])) {
                vals[threadIdx.x] = ov;
                idxs[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = idxs[0];
    }
}

template <typename scalar_t>
__global__ void sample_softmax_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ random,
    int64_t* __restrict__ out,
    int B,
    int V,
    float temperature)
{
    int row = blockIdx.x;
    __shared__ float red[SAMPLE_BD];
    __shared__ float row_max;
    __shared__ float row_sum;
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < V; i += SAMPLE_BD) {
        float v = sample_load(logits, (size_t)row * V + i) / temperature;
        local_max = fmaxf(local_max, v);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = SAMPLE_BD / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_max = red[0];
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += SAMPLE_BD) {
        float v = sample_load(logits, (size_t)row * V + i) / temperature;
        local_sum += expf(v - row_max);
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = SAMPLE_BD / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_sum = red[0];
        float u = fminf(fmaxf(random[row], 0.0f), 0.99999994f);
        float target = u * row_sum;
        float cdf = 0.0f;
        int chosen = V - 1;
        for (int i = 0; i < V; ++i) {
            float v = sample_load(logits, (size_t)row * V + i) / temperature;
            cdf += expf(v - row_max);
            if (cdf >= target) {
                chosen = i;
                break;
            }
        }
        out[row] = chosen;
    }
}

template <typename scalar_t>
__global__ void sample_top_k_top_p_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ random,
    int64_t* __restrict__ out,
    int B,
    int V,
    float temperature,
    int top_k,
    float top_p)
{
    int row = blockIdx.x;
    __shared__ float top_vals[SAMPLE_MAX_TOP_K];
    __shared__ int top_idx[SAMPLE_MAX_TOP_K];
    __shared__ float reduce_vals[SAMPLE_BD];
    __shared__ int reduce_idxs[SAMPLE_BD];

    if (top_k <= 64) {
        for (int rank = 0; rank < top_k; ++rank) {
            float best = -FLT_MAX;
            int best_i = V;
            for (int i = threadIdx.x; i < V; i += SAMPLE_BD) {
                bool selected = false;
                for (int j = 0; j < rank; ++j) {
                    selected = selected || top_idx[j] == i;
                }
                if (selected) continue;
                float v = sample_load(logits, (size_t)row * V + i) / temperature;
                if (v > best || (v == best && i < best_i)) {
                    best = v;
                    best_i = i;
                }
            }
            reduce_vals[threadIdx.x] = best;
            reduce_idxs[threadIdx.x] = best_i;
            __syncthreads();
            for (int stride = SAMPLE_BD / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    float other = reduce_vals[threadIdx.x + stride];
                    int other_i = reduce_idxs[threadIdx.x + stride];
                    if (other > reduce_vals[threadIdx.x] ||
                        (other == reduce_vals[threadIdx.x] && other_i < reduce_idxs[threadIdx.x])) {
                        reduce_vals[threadIdx.x] = other;
                        reduce_idxs[threadIdx.x] = other_i;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                top_vals[rank] = reduce_vals[0];
                top_idx[rank] = reduce_idxs[0];
            }
            __syncthreads();
        }
    } else {
        if (threadIdx.x == 0) {
            for (int j = 0; j < top_k; ++j) {
                top_vals[j] = -FLT_MAX;
                top_idx[j] = 0;
            }
            for (int i = 0; i < V; ++i) {
                float v = sample_load(logits, (size_t)row * V + i) / temperature;
                if (v < top_vals[top_k - 1] ||
                    (v == top_vals[top_k - 1] && i > top_idx[top_k - 1])) continue;
                int pos = top_k - 1;
                while (pos > 0 && (v > top_vals[pos - 1] ||
                       (v == top_vals[pos - 1] && i < top_idx[pos - 1]))) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_idx[pos] = top_idx[pos - 1];
                    --pos;
                }
                top_vals[pos] = v;
                top_idx[pos] = i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    float max_v = top_vals[0];
    float probs[SAMPLE_MAX_TOP_K];
    float sum = 0.0f;
    for (int j = 0; j < top_k; ++j) {
        probs[j] = expf(top_vals[j] - max_v);
        sum += probs[j];
    }
    int keep = top_k;
    float keep_sum = sum;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cutoff = top_p * sum;
        float c = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            c += probs[j];
            if (c >= cutoff) {
                keep = j + 1;
                keep_sum = c;
                break;
            }
        }
    }
    float u = fminf(fmaxf(random[row], 0.0f), 0.99999994f);
    float target = u * keep_sum;
    float cdf = 0.0f;
    int chosen = top_idx[keep - 1];
    for (int j = 0; j < keep; ++j) {
        cdf += probs[j];
        if (cdf >= target) {
            chosen = top_idx[j];
            break;
        }
    }
    out[row] = chosen;
}

torch::Tensor sample_greedy_cuda(torch::Tensor logits)
{
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "sample_greedy: logits must be cuda contiguous");
    TORCH_CHECK(logits.dim() == 2, "sample_greedy: logits must be [B,V]");
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat32 ||
        logits.scalar_type() == torch::kFloat16 ||
        logits.scalar_type() == torch::kBFloat16,
        "sample_greedy: logits dtype must be f32, f16, or bf16");
    int B = (int)logits.size(0);
    int V = (int)logits.size(1);
    auto out = torch::empty({B}, logits.options().dtype(torch::kInt64));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        logits.scalar_type(), "sample_greedy_cuda", [&] {
        sample_greedy_kernel<scalar_t><<<B, SAMPLE_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
            logits.data_ptr<scalar_t>(), out.data_ptr<int64_t>(), B, V);
    });
    return out;
}

torch::Tensor sample_softmax_cuda(torch::Tensor logits, torch::Tensor random, double temperature)
{
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "sample_softmax: logits must be cuda contiguous");
    TORCH_CHECK(random.is_cuda() && random.is_contiguous() && random.scalar_type() == torch::kFloat32,
                "sample_softmax: random must be cuda contiguous f32");
    TORCH_CHECK(logits.dim() == 2, "sample_softmax: logits must be [B,V]");
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat32 ||
        logits.scalar_type() == torch::kFloat16 ||
        logits.scalar_type() == torch::kBFloat16,
        "sample_softmax: logits dtype must be f32, f16, or bf16");
    int B = (int)logits.size(0);
    int V = (int)logits.size(1);
    TORCH_CHECK(random.numel() == B, "sample_softmax: random length must match B");
    TORCH_CHECK(temperature > 0.0, "sample_softmax: temperature must be > 0");
    auto out = torch::empty({B}, logits.options().dtype(torch::kInt64));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        logits.scalar_type(), "sample_softmax_cuda", [&] {
        sample_softmax_kernel<scalar_t><<<B, SAMPLE_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
            logits.data_ptr<scalar_t>(), random.data_ptr<float>(), out.data_ptr<int64_t>(),
            B, V, (float)temperature);
    });
    return out;
}

torch::Tensor sample_top_k_top_p_cuda(
    torch::Tensor logits,
    torch::Tensor random,
    double temperature,
    int64_t top_k,
    double top_p)
{
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "sample_top_k_top_p: logits must be cuda contiguous");
    TORCH_CHECK(random.is_cuda() && random.is_contiguous() && random.scalar_type() == torch::kFloat32,
                "sample_top_k_top_p: random must be cuda contiguous f32");
    TORCH_CHECK(logits.dim() == 2, "sample_top_k_top_p: logits must be [B,V]");
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat32 ||
        logits.scalar_type() == torch::kFloat16 ||
        logits.scalar_type() == torch::kBFloat16,
        "sample_top_k_top_p: logits dtype must be f32, f16, or bf16");
    int B = (int)logits.size(0);
    int V = (int)logits.size(1);
    TORCH_CHECK(random.numel() == B, "sample_top_k_top_p: random length must match B");
    TORCH_CHECK(temperature > 0.0, "sample_top_k_top_p: temperature must be > 0");
    TORCH_CHECK(top_k > 0 && top_k <= V && top_k <= SAMPLE_MAX_TOP_K,
                "sample_top_k_top_p: top_k must be in [1, min(V,1024)]");
    auto out = torch::empty({B}, logits.options().dtype(torch::kInt64));

    const char* radix_env = std::getenv("MFQ_SAMPLE_RADIX_TOPK");
    const bool use_radix = B == 1 && top_k <= SAMPLE_RADIX_MAX_TOP_K &&
        (radix_env == nullptr || radix_env[0] != '0');
    if (use_radix) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int first_blocks = (V + SAMPLE_BD - 1) / SAMPLE_BD;
        const int64_t capacity = (int64_t)first_blocks * top_k;
        auto vals_a = torch::empty(
            {capacity}, logits.options().dtype(torch::kFloat32));
        auto idxs_a = torch::empty(
            {capacity}, logits.options().dtype(torch::kInt32));
        auto vals_b = torch::empty(
            {capacity}, logits.options().dtype(torch::kFloat32));
        auto idxs_b = torch::empty(
            {capacity}, logits.options().dtype(torch::kInt32));
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            logits.scalar_type(), "sample_top_k_radix_stage1", [&] {
            sample_top_k_radix_stage1_kernel<scalar_t><<<first_blocks, SAMPLE_BD, 0, stream>>>(
                logits.data_ptr<scalar_t>(), vals_a.data_ptr<float>(), idxs_a.data_ptr<int>(),
                V, (int)top_k);
        });

        int count = first_blocks * (int)top_k;
        bool input_is_a = true;
        while (count > SAMPLE_BD) {
            const int blocks = (count + SAMPLE_BD - 1) / SAMPLE_BD;
            if (input_is_a) {
                sample_top_k_radix_reduce_kernel<<<blocks, SAMPLE_BD, 0, stream>>>(
                    vals_a.data_ptr<float>(), idxs_a.data_ptr<int>(),
                    vals_b.data_ptr<float>(), idxs_b.data_ptr<int>(), count, (int)top_k);
            } else {
                sample_top_k_radix_reduce_kernel<<<blocks, SAMPLE_BD, 0, stream>>>(
                    vals_b.data_ptr<float>(), idxs_b.data_ptr<int>(),
                    vals_a.data_ptr<float>(), idxs_a.data_ptr<int>(), count, (int)top_k);
            }
            count = blocks * (int)top_k;
            input_is_a = !input_is_a;
        }
        if (input_is_a) {
            sample_top_k_radix_finish_kernel<<<1, SAMPLE_BD, 0, stream>>>(
                vals_a.data_ptr<float>(), idxs_a.data_ptr<int>(), random.data_ptr<float>(),
                out.data_ptr<int64_t>(), count, (int)top_k, (float)temperature, (float)top_p);
        } else {
            sample_top_k_radix_finish_kernel<<<1, SAMPLE_BD, 0, stream>>>(
                vals_b.data_ptr<float>(), idxs_b.data_ptr<int>(), random.data_ptr<float>(),
                out.data_ptr<int64_t>(), count, (int)top_k, (float)temperature, (float)top_p);
        }
        return out;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        logits.scalar_type(), "sample_top_k_top_p_cuda", [&] {
        sample_top_k_top_p_kernel<scalar_t><<<B, SAMPLE_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
            logits.data_ptr<scalar_t>(), random.data_ptr<float>(), out.data_ptr<int64_t>(),
            B, V, (float)temperature, (int)top_k, (float)top_p);
    });
    return out;
}

void sample_token_counts_add_cuda(torch::Tensor counts, torch::Tensor tokens)
{
    TORCH_CHECK(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type() == torch::kInt32,
                "sample_token_counts_add: counts must be contiguous CUDA int32");
    TORCH_CHECK(tokens.is_cuda() && tokens.is_contiguous() && tokens.scalar_type() == torch::kInt64,
                "sample_token_counts_add: tokens must be contiguous CUDA int64");
    TORCH_CHECK(counts.dim() == 1, "sample_token_counts_add: counts must be [V]");
    int64_t n = tokens.numel();
    if (n == 0) return;
    constexpr int bd = 256;
    int blocks = (int)((n + bd - 1) / bd);
    sample_token_counts_add_kernel<<<blocks, bd, 0, at::cuda::getCurrentCUDAStream()>>>(
        counts.data_ptr<int32_t>(), tokens.data_ptr<int64_t>(), n, (int)counts.numel());
}

torch::Tensor sample_apply_penalties_cuda(
    torch::Tensor logits,
    torch::Tensor counts,
    double presence_penalty,
    double frequency_penalty,
    double repetition_penalty)
{
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(),
                "sample_apply_penalties: logits must be contiguous CUDA");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == 1,
                "sample_apply_penalties: logits must be [1,V]");
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat32 ||
        logits.scalar_type() == torch::kFloat16 ||
        logits.scalar_type() == torch::kBFloat16,
        "sample_apply_penalties: logits dtype must be f32, f16, or bf16");
    TORCH_CHECK(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type() == torch::kInt32,
                "sample_apply_penalties: counts must be contiguous CUDA int32");
    TORCH_CHECK(counts.dim() == 1 && counts.numel() == logits.size(1),
                "sample_apply_penalties: counts must be [V]");
    int vocab_size = (int)logits.size(1);
    constexpr int bd = 256;
    int blocks = (vocab_size + bd - 1) / bd;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        logits.scalar_type(), "sample_apply_penalties_cuda", [&] {
        sample_apply_penalties_kernel<scalar_t><<<blocks, bd, 0, at::cuda::getCurrentCUDAStream()>>>(
            logits.data_ptr<scalar_t>(), counts.data_ptr<int32_t>(), vocab_size,
            (float)presence_penalty, (float)frequency_penalty, (float)repetition_penalty);
    });
    return logits;
}
