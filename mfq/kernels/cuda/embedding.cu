// Embedding lookup helpers.
// Raw path: weight [vocab, D], token_ids arbitrary shape -> out token_ids.shape + [D].
// NINT path: selected-row dequant from compressed token embedding rows.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

template <int BITS>
__device__ __forceinline__ uint8_t nint_unpack_qbits_one(const uint8_t* p, int lane)
{
    if constexpr (BITS == 8) {
        return p[lane];
    } else {
        constexpr uint32_t MASK = (1u << BITS) - 1u;
        int bit = lane * BITS;
        int byte = bit >> 3;
        int shift = bit & 7;
        uint32_t word = (uint32_t)p[byte];
        if (shift + BITS > 8) {
            word |= ((uint32_t)p[byte + 1] << 8);
        }
        return (uint8_t)((word >> shift) & MASK);
    }
}

template <typename scalar_t>
__global__ void embedding_lookup_kernel(
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ ids,
    scalar_t* __restrict__ out,
    int N,
    int D,
    int vocab)
{
    size_t total = (size_t)N * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % D);
        int n = (int)(idx / D);
        int64_t tok = ids[n];
        if (tok < 0 || tok >= vocab) {
            out[idx] = scalar_t(0);
        } else {
            out[idx] = weight[(size_t)tok * D + d];
        }
    }
}

__global__ void nint_embedding_lookup_kernel(
    const uint8_t* __restrict__ q,
    const float* __restrict__ d_eff,
    const float* __restrict__ m_eff,
    const int64_t* __restrict__ ids,
    half* __restrict__ out,
    int N,
    int vocab,
    int ng,
    int gs,
    int D)
{
    size_t total = (size_t)N * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % D);
        int n = (int)(idx / D);
        int64_t tok = ids[n];
        float val = 0.0f;
        if (tok >= 0 && tok < vocab) {
            int g = d / gs;
            int lane = d - g * gs;
            size_t q_idx = (((size_t)tok * ng + g) * gs + lane);
            size_t m_idx = (size_t)tok * ng + g;
            val = d_eff[m_idx] * (float)q[q_idx] - m_eff[m_idx];
        }
        out[idx] = __float2half(val);
    }
}

__global__ void nint_embedding_lookup_packed_eff_kernel(
    const uint8_t* __restrict__ q_packed,
    const half2* __restrict__ eff_pair,
    const int64_t* __restrict__ ids,
    half* __restrict__ out,
    int N,
    int vocab,
    int ng,
    int gs,
    int qbytes,
    int D)
{
    size_t total = (size_t)N * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % D);
        int n = (int)(idx / D);
        int64_t tok = ids[n];
        float val = 0.0f;
        if (tok >= 0 && tok < vocab) {
            int g = d / gs;
            int lane = d - g * gs;
            uint8_t qb = q_packed[((size_t)tok * ng + g) * qbytes + lane / 2];
            uint8_t qv = (lane & 1) ? (qb >> 4) : (qb & 0x0F);
            half2 dm = eff_pair[(size_t)tok * ng + g];
            float d_eff = __half2float(__low2half(dm));
            float m_eff = __half2float(__high2half(dm));
            val = d_eff * (float)qv - m_eff;
        }
        out[idx] = __float2half(val);
    }
}

__global__ void nint_embedding_lookup_packed_compact_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int64_t* __restrict__ ids,
    half* __restrict__ out,
    int N,
    int vocab,
    int ng,
    int gs,
    int qbytes,
    int D)
{
    size_t total = (size_t)N * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % D);
        int n = (int)(idx / D);
        int64_t tok = ids[n];
        float val = 0.0f;
        if (tok >= 0 && tok < vocab) {
            int g = d / gs;
            int lane = d - g * gs;
            size_t meta_idx = (size_t)tok * ng + g;
            uint8_t qb = q_packed[meta_idx * qbytes + lane / 2];
            uint8_t qv = (lane & 1) ? (qb >> 4) : (qb & 0x0F);
            float d_eff = neuron_scale[tok] * (float)sub_scale[meta_idx];
            float m_eff = neuron_min[tok] * (float)sub_min[meta_idx];
            val = d_eff * (float)qv - m_eff;
        }
        out[idx] = __float2half(val);
    }
}

template <int BITS, int GS>
__global__ void nint_embedding_lookup_packed_compact_bits_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int64_t* __restrict__ ids,
    half* __restrict__ out,
    int N,
    int vocab,
    int ng,
    int D)
{
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    size_t total = (size_t)N * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % D);
        int n = (int)(idx / D);
        int64_t tok = ids[n];
        float val = 0.0f;
        if (tok >= 0 && tok < vocab) {
            int g = d / GS;
            int lane = d - g * GS;
            size_t meta_idx = (size_t)tok * ng + g;
            uint8_t qv = nint_unpack_qbits_one<BITS>(q_packed + meta_idx * QBYTES, lane);
            float d_eff = neuron_scale[tok] * (float)sub_scale[meta_idx];
            float m_eff = neuron_min[tok] * (float)sub_min[meta_idx];
            val = d_eff * (float)qv - m_eff;
        }
        out[idx] = __float2half(val);
    }
}

__global__ void nint8_zero_embedding_lookup_kernel(
    const int8_t* __restrict__ q,
    const half* __restrict__ scale,
    const int64_t* __restrict__ ids,
    half* __restrict__ out,
    int N,
    int vocab,
    int ng,
    int D)
{
    const size_t total = static_cast<size_t>(N) * D;
    for (size_t index =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int d = static_cast<int>(index % D);
        const int n = static_cast<int>(index / D);
        const int64_t token = ids[n];
        float value = 0.0f;
        if (token >= 0 && token < vocab) {
            const int group = d / 32;
            const int lane = d % 32;
            const size_t block =
                static_cast<size_t>(token) * ng + group;
            value = __half2float(scale[block]) *
                static_cast<float>(q[block * 32 + lane]);
        }
        out[index] = __float2half(value);
    }
}

torch::Tensor embedding_lookup_cuda(torch::Tensor weight, torch::Tensor token_ids)
{
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "embedding_lookup: weight must be cuda contiguous");
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() && token_ids.scalar_type() == torch::kInt64,
                "embedding_lookup: token_ids must be cuda contiguous int64");
    TORCH_CHECK(weight.dim() == 2, "embedding_lookup: weight must be [vocab,D]");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32 || weight.scalar_type() == torch::kFloat16,
                "embedding_lookup: weight dtype must be f32 or f16");
    int vocab = (int)weight.size(0);
    int D = (int)weight.size(1);
    int N = (int)token_ids.numel();
    auto out_shape = token_ids.sizes().vec();
    out_shape.push_back(D);
    auto out = torch::empty(out_shape, weight.options());
    constexpr int BD = 256;
    size_t total = (size_t)N * D;
    int grid = (int)((total + BD - 1) / BD);
    grid = grid > 4096 ? 4096 : grid;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "embedding_lookup_cuda", [&] {
        embedding_lookup_kernel<scalar_t><<<grid, BD, 0, at::cuda::getCurrentCUDAStream()>>>(
            weight.data_ptr<scalar_t>(), token_ids.data_ptr<int64_t>(), out.data_ptr<scalar_t>(), N, D, vocab);
    });
    return out;
}

torch::Tensor nint_embedding_lookup_cuda(
    torch::Tensor q,
    torch::Tensor d_eff,
    torch::Tensor m_eff,
    torch::Tensor token_ids,
    int64_t neuron_len,
    int64_t gs)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kUInt8,
                "nint_embedding_lookup: q must be cuda contiguous uint8");
    TORCH_CHECK(d_eff.is_cuda() && d_eff.is_contiguous() && d_eff.scalar_type() == torch::kFloat32,
                "nint_embedding_lookup: d_eff must be cuda contiguous f32");
    TORCH_CHECK(m_eff.is_cuda() && m_eff.is_contiguous() && m_eff.scalar_type() == torch::kFloat32,
                "nint_embedding_lookup: m_eff must be cuda contiguous f32");
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() && token_ids.scalar_type() == torch::kInt64,
                "nint_embedding_lookup: token_ids must be cuda contiguous int64");
    TORCH_CHECK(q.dim() == 3, "nint_embedding_lookup: q must be [vocab,ng,gs]");
    int vocab = (int)q.size(0);
    int ng = (int)q.size(1);
    TORCH_CHECK(q.size(2) == gs, "nint_embedding_lookup: q gs mismatch");
    TORCH_CHECK(d_eff.size(0) == vocab && d_eff.size(1) == ng && m_eff.sizes() == d_eff.sizes(),
                "nint_embedding_lookup: metadata shape mismatch");
    int D = (int)neuron_len;
    int N = (int)token_ids.numel();
    auto out_shape = token_ids.sizes().vec();
    out_shape.push_back(D);
    auto out = torch::empty(out_shape, q.options().dtype(torch::kFloat16));
    constexpr int BD = 256;
    size_t total = (size_t)N * D;
    int grid = (int)((total + BD - 1) / BD);
    grid = grid > 4096 ? 4096 : grid;
    nint_embedding_lookup_kernel<<<grid, BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<uint8_t>(), d_eff.data_ptr<float>(), m_eff.data_ptr<float>(),
        token_ids.data_ptr<int64_t>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        N, vocab, ng, (int)gs, D);
    return out;
}

torch::Tensor nint_embedding_lookup_packed_eff_cuda(
    torch::Tensor q_packed,
    torch::Tensor eff_pair,
    torch::Tensor token_ids,
    int64_t neuron_len,
    int64_t gs)
{
    TORCH_CHECK(q_packed.is_cuda() && q_packed.is_contiguous() && q_packed.scalar_type() == torch::kUInt8,
                "nint_embedding_lookup_packed_eff: q_packed must be cuda contiguous uint8");
    TORCH_CHECK(eff_pair.is_cuda() && eff_pair.is_contiguous() && eff_pair.scalar_type() == torch::kFloat16,
                "nint_embedding_lookup_packed_eff: eff_pair must be cuda contiguous f16");
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() && token_ids.scalar_type() == torch::kInt64,
                "nint_embedding_lookup_packed_eff: token_ids must be cuda contiguous int64");
    TORCH_CHECK(q_packed.dim() == 3, "nint_embedding_lookup_packed_eff: q_packed must be [vocab,ng,gs/2]");
    TORCH_CHECK(eff_pair.dim() == 3 && eff_pair.size(2) == 2,
                "nint_embedding_lookup_packed_eff: eff_pair must be [vocab,ng,2]");
    int vocab = (int)q_packed.size(0);
    int ng = (int)q_packed.size(1);
    int qbytes = (int)q_packed.size(2);
    TORCH_CHECK(qbytes * 2 == gs, "nint_embedding_lookup_packed_eff: q_packed gs mismatch");
    TORCH_CHECK(eff_pair.size(0) == vocab && eff_pair.size(1) == ng,
                "nint_embedding_lookup_packed_eff: metadata shape mismatch");
    int D = (int)neuron_len;
    int N = (int)token_ids.numel();
    auto out_shape = token_ids.sizes().vec();
    out_shape.push_back(D);
    auto out = torch::empty(out_shape, q_packed.options().dtype(torch::kFloat16));
    constexpr int BD = 256;
    size_t total = (size_t)N * D;
    int grid = (int)((total + BD - 1) / BD);
    grid = grid > 4096 ? 4096 : grid;
    nint_embedding_lookup_packed_eff_kernel<<<grid, BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_packed.data_ptr<uint8_t>(), reinterpret_cast<const half2*>(eff_pair.data_ptr<at::Half>()),
        token_ids.data_ptr<int64_t>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        N, vocab, ng, (int)gs, qbytes, D);
    return out;
}

torch::Tensor nint_embedding_lookup_packed_compact_cuda(
    torch::Tensor q_packed,
    torch::Tensor sub_scale,
    torch::Tensor sub_min,
    torch::Tensor neuron_scale,
    torch::Tensor neuron_min,
    torch::Tensor token_ids,
    int64_t neuron_len,
    int64_t gs)
{
    TORCH_CHECK(q_packed.is_cuda() && q_packed.is_contiguous() && q_packed.scalar_type() == torch::kUInt8,
                "nint_embedding_lookup_packed_compact: q_packed must be cuda contiguous uint8");
    TORCH_CHECK(sub_scale.is_cuda() && sub_scale.is_contiguous() && sub_scale.scalar_type() == torch::kUInt8,
                "nint_embedding_lookup_packed_compact: sub_scale must be cuda contiguous uint8");
    TORCH_CHECK(sub_min.is_cuda() && sub_min.is_contiguous() && sub_min.scalar_type() == torch::kUInt8,
                "nint_embedding_lookup_packed_compact: sub_min must be cuda contiguous uint8");
    TORCH_CHECK(neuron_scale.is_cuda() && neuron_scale.is_contiguous() && neuron_scale.scalar_type() == torch::kFloat32,
                "nint_embedding_lookup_packed_compact: neuron_scale must be cuda contiguous f32");
    TORCH_CHECK(neuron_min.is_cuda() && neuron_min.is_contiguous() && neuron_min.scalar_type() == torch::kFloat32,
                "nint_embedding_lookup_packed_compact: neuron_min must be cuda contiguous f32");
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() && token_ids.scalar_type() == torch::kInt64,
                "nint_embedding_lookup_packed_compact: token_ids must be cuda contiguous int64");
    TORCH_CHECK(q_packed.dim() == 3, "nint_embedding_lookup_packed_compact: q_packed must be [vocab,ng,gs/2]");
    int vocab = (int)q_packed.size(0);
    int ng = (int)q_packed.size(1);
    int qbytes = (int)q_packed.size(2);
    TORCH_CHECK(qbytes * 2 == gs, "nint_embedding_lookup_packed_compact: q_packed gs mismatch");
    TORCH_CHECK(sub_scale.size(0) == vocab && sub_scale.size(1) == ng,
                "nint_embedding_lookup_packed_compact: sub_scale shape mismatch");
    TORCH_CHECK(sub_min.sizes() == sub_scale.sizes(), "nint_embedding_lookup_packed_compact: sub_min shape mismatch");
    TORCH_CHECK(neuron_scale.size(0) == vocab && neuron_min.size(0) == vocab,
                "nint_embedding_lookup_packed_compact: neuron metadata shape mismatch");
    int D = (int)neuron_len;
    int N = (int)token_ids.numel();
    auto out_shape = token_ids.sizes().vec();
    out_shape.push_back(D);
    auto out = torch::empty(out_shape, q_packed.options().dtype(torch::kFloat16));
    constexpr int BD = 256;
    size_t total = (size_t)N * D;
    int grid = (int)((total + BD - 1) / BD);
    grid = grid > 4096 ? 4096 : grid;
    nint_embedding_lookup_packed_compact_kernel<<<grid, BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
        neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(),
        token_ids.data_ptr<int64_t>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        N, vocab, ng, (int)gs, qbytes, D);
    return out;
}

torch::Tensor nint_embedding_lookup_packed_compact_bits_cuda(
    torch::Tensor q_packed,
    torch::Tensor sub_scale,
    torch::Tensor sub_min,
    torch::Tensor neuron_scale,
    torch::Tensor neuron_min,
    torch::Tensor token_ids,
    int64_t neuron_len,
    int64_t gs,
    int64_t bits)
{
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 5 || bits == 6 || bits == 8,
                "packed-bits embedding supports bits in {2,3,5,6,8}, got ", bits);
    TORCH_CHECK(q_packed.is_cuda() && q_packed.is_contiguous() && q_packed.scalar_type() == torch::kUInt8,
                "nint_embedding_lookup_packed_compact_bits: q_packed must be cuda contiguous uint8");
    TORCH_CHECK(sub_scale.is_cuda() && sub_scale.is_contiguous() && sub_scale.scalar_type() == torch::kUInt8,
                "nint_embedding_lookup_packed_compact_bits: sub_scale must be cuda contiguous uint8");
    TORCH_CHECK(sub_min.is_cuda() && sub_min.is_contiguous() && sub_min.scalar_type() == torch::kUInt8,
                "nint_embedding_lookup_packed_compact_bits: sub_min must be cuda contiguous uint8");
    TORCH_CHECK(neuron_scale.is_cuda() && neuron_scale.is_contiguous() && neuron_scale.scalar_type() == torch::kFloat32,
                "nint_embedding_lookup_packed_compact_bits: neuron_scale must be cuda contiguous f32");
    TORCH_CHECK(neuron_min.is_cuda() && neuron_min.is_contiguous() && neuron_min.scalar_type() == torch::kFloat32,
                "nint_embedding_lookup_packed_compact_bits: neuron_min must be cuda contiguous f32");
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() && token_ids.scalar_type() == torch::kInt64,
                "nint_embedding_lookup_packed_compact_bits: token_ids must be cuda contiguous int64");
    TORCH_CHECK(q_packed.dim() == 3, "nint_embedding_lookup_packed_compact_bits: q_packed must be [vocab,ng,qbytes]");
    int vocab = (int)q_packed.size(0);
    int ng = (int)q_packed.size(1);
    TORCH_CHECK((int)q_packed.size(2) == (((int)gs * (int)bits + 7) / 8),
                "nint_embedding_lookup_packed_compact_bits: q_packed gs/bits mismatch");
    TORCH_CHECK(sub_scale.size(0) == vocab && sub_scale.size(1) == ng,
                "nint_embedding_lookup_packed_compact_bits: sub_scale shape mismatch");
    TORCH_CHECK(sub_min.sizes() == sub_scale.sizes(), "nint_embedding_lookup_packed_compact_bits: sub_min shape mismatch");
    TORCH_CHECK(neuron_scale.size(0) == vocab && neuron_min.size(0) == vocab,
                "nint_embedding_lookup_packed_compact_bits: neuron metadata shape mismatch");
    int D = (int)neuron_len;
    int N = (int)token_ids.numel();
    auto out_shape = token_ids.sizes().vec();
    out_shape.push_back(D);
    auto out = torch::empty(out_shape, q_packed.options().dtype(torch::kFloat16));
    constexpr int BD = 256;
    size_t total = (size_t)N * D;
    int grid = (int)((total + BD - 1) / BD);
    grid = grid > 4096 ? 4096 : grid;

#define NINT_EMB_BITS_LAUNCH(BITSVAL, GSVAL)                                           \
    nint_embedding_lookup_packed_compact_bits_kernel<BITSVAL, GSVAL><<<grid, BD, 0, at::cuda::getCurrentCUDAStream()>>>( \
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
        neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), token_ids.data_ptr<int64_t>(), \
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), N, vocab, ng, D)

#define NINT_EMB_BITS_GS_SWITCH(BITSVAL)                                                \
    switch ((int)gs) {                                                                  \
        case 16: NINT_EMB_BITS_LAUNCH(BITSVAL, 16); break;                              \
        case 20: NINT_EMB_BITS_LAUNCH(BITSVAL, 20); break;                              \
        case 22: NINT_EMB_BITS_LAUNCH(BITSVAL, 22); break;                              \
        case 24: NINT_EMB_BITS_LAUNCH(BITSVAL, 24); break;                              \
        case 26: NINT_EMB_BITS_LAUNCH(BITSVAL, 26); break;                              \
        case 28: NINT_EMB_BITS_LAUNCH(BITSVAL, 28); break;                              \
        case 30: NINT_EMB_BITS_LAUNCH(BITSVAL, 30); break;                              \
        case 32: NINT_EMB_BITS_LAUNCH(BITSVAL, 32); break;                              \
        case 34: NINT_EMB_BITS_LAUNCH(BITSVAL, 34); break;                              \
        case 36: NINT_EMB_BITS_LAUNCH(BITSVAL, 36); break;                              \
        case 40: NINT_EMB_BITS_LAUNCH(BITSVAL, 40); break;                              \
        case 48: NINT_EMB_BITS_LAUNCH(BITSVAL, 48); break;                              \
        case 64: NINT_EMB_BITS_LAUNCH(BITSVAL, 64); break;                              \
        default: TORCH_CHECK(false, "packed-bits embedding unsupported gs ", gs);        \
    }

    if (bits == 2) {
        NINT_EMB_BITS_GS_SWITCH(2);
    } else if (bits == 3) {
        NINT_EMB_BITS_GS_SWITCH(3);
    } else if (bits == 5) {
        NINT_EMB_BITS_GS_SWITCH(5);
    } else if (bits == 6) {
        NINT_EMB_BITS_GS_SWITCH(6);
    } else {
        NINT_EMB_BITS_GS_SWITCH(8);
    }
#undef NINT_EMB_BITS_GS_SWITCH
#undef NINT_EMB_BITS_LAUNCH
    return out;
}

torch::Tensor nint8_zero_embedding_lookup_cuda(
    torch::Tensor q,
    torch::Tensor scale,
    torch::Tensor token_ids,
    int64_t neuron_len)
{
    TORCH_CHECK(
        q.is_cuda() && q.is_contiguous() &&
            q.scalar_type() == torch::kUInt8 && q.dim() == 3 &&
            q.size(2) == 32,
        "NINT8-0 embedding q must be contiguous CUDA uint8 [vocab,ng,32]");
    TORCH_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
            scale.scalar_type() == torch::kFloat16 && scale.dim() == 2 &&
            scale.size(0) == q.size(0) && scale.size(1) == q.size(1),
        "NINT8-0 embedding scale must be contiguous CUDA f16 [vocab,ng]");
    TORCH_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
            token_ids.scalar_type() == torch::kInt64,
        "NINT8-0 embedding token_ids must be contiguous CUDA int64");
    TORCH_CHECK(
        neuron_len > 0 && neuron_len <= q.size(1) * 32,
        "NINT8-0 embedding neuron_len is invalid");
    const int vocab = static_cast<int>(q.size(0));
    const int ng = static_cast<int>(q.size(1));
    const int count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(neuron_len);
    auto out = torch::empty(shape, q.options().dtype(torch::kFloat16));
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(count) * neuron_len;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    blocks = std::min(blocks, 4096);
    nint8_zero_embedding_lookup_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int8_t *>(q.data_ptr<uint8_t>()),
        reinterpret_cast<const half *>(scale.data_ptr<at::Half>()),
        token_ids.data_ptr<int64_t>(),
        reinterpret_cast<half *>(out.data_ptr<at::Half>()),
        count, vocab, ng, static_cast<int>(neuron_len));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
