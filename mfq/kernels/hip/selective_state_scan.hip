// Selective state scan (SSD recurrence) for linear-attention mixers.
//
// The recurrent state is float32 [B, H, P, N]: one P x N matrix per (batch,
// head). B/C projections are grouped [.., G, N] and head h reads group
// h / (H / G). Every token performs the same update (decode is prefill with
// T = 1), so a prefill of T tokens equals T decode calls up to rounding:
//
//   dt_t  = clamp(softplus(dt_t + dt_bias[h]), dt_limit_lo, dt_limit_hi)
//   A_h   = -exp(A_log[h])
//   state = state * exp(dt_t * A_h) + dt_t * outer(x_t, B_g)
//   y_t   = sum_n(state * C_g) + D[h] * x_t
//
// decode : state [B,H,P,N] updated in place, x [B,H,P], dt [B,H],
//          B/C [B,G,N]; returns y [B,H,P] f32.
// prefill: x [B,T,H,P], dt [B,T,H], B/C [B,T,G,N], optional initial_state
//          [B,H,P,N]; returns y [B,T,H,P] and final_state [B,H,P,N], both f32.
//
// x/dt/B/C may be f32, f16 or bf16 and must share one dtype. The state, the
// parameters (dt_bias/A_log/D) and every output are always f32; accumulation
// is f32 as well.
//
// The prefill kernel runs the recurrence token by token, so it accepts any T
// (1, 257, 1024, ...) and any initial state without chunk alignment. Cost is
// O(B*H*T*P*N) with the time axis serial inside one block, i.e. long prompts
// are latency-bound. A chunked (two-level) SSD factorization can replace the
// body of selective_state_scan_prefill_kernel behind the same interface: the
// intra-chunk blocks then become parallel over T and only the inter-chunk
// state carry stays serial.

#include <hip/hip_runtime.h>
#include "mfq_tensor_backend.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "mfq_hip_compat.h"
#include "reduce.cuh"

// One thread per state column for decode; 4 warps make block_sum cheap.
constexpr int SELECTIVE_SCAN_DECODE_THREADS = 128;
// Prefill: one warp owns one head column p, so the n-axis reduction is a pure
// warp shuffle and no block-wide barrier is needed inside the scan.
constexpr int SELECTIVE_SCAN_WARP = 32;
constexpr int SELECTIVE_SCAN_WARPS = 8;

__device__ __forceinline__ float selective_scan_value_f32(float value)
{
    return value;
}

__device__ __forceinline__ float selective_scan_value_f32(__half value)
{
    return __half2float(value);
}

__device__ __forceinline__ float selective_scan_value_f32(__hip_bfloat16 value)
{
    return __bfloat162float(value);
}

// softplus(dt + dt_bias) followed by the caller's clamp bounds. The max() form
// of softplus stays finite for large positive inputs, matching the reference
// log1p(exp(.)) closed form.
__device__ __forceinline__ float selective_scan_step_dt(
    float raw_dt, float dt_bias, float limit_lo, float limit_hi)
{
    const float shifted = raw_dt + dt_bias;
    const float softplus = fmaxf(shifted, 0.0f) + log1pf(expf(-fabsf(shifted)));
    return fminf(fmaxf(softplus, limit_lo), limit_hi);
}

// Decode: one block per (batch, head, head column p), threads span the N axis.
template <class Value>
__global__ void selective_state_scan_decode_kernel(
    float* __restrict__ state,
    const Value* __restrict__ x,
    const Value* __restrict__ dt,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const Value* __restrict__ b,
    const Value* __restrict__ c,
    const float* __restrict__ d,
    float* __restrict__ output,
    int heads,
    int columns,
    int state_width,
    int groups,
    float limit_lo,
    float limit_hi)
{
    const int column = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int group = head / (heads / groups);

    const int64_t row = ((int64_t)(batch * heads + head) * columns) + column;
    float* state_row = state + row * state_width;
    const Value* b_row = b + ((int64_t)batch * groups + group) * state_width;
    const Value* c_row = c + ((int64_t)batch * groups + group) * state_width;

    const float dt_value = selective_scan_step_dt(
        selective_scan_value_f32(dt[batch * heads + head]), dt_bias[head],
        limit_lo, limit_hi);
    const float decay = expf(dt_value * -expf(a_log[head]));
    const float x_value = selective_scan_value_f32(x[row]);

    float accumulated = 0.0f;
    for (int index = threadIdx.x; index < state_width; index += blockDim.x) {
        const float updated = state_row[index] * decay +
            dt_value * x_value * selective_scan_value_f32(b_row[index]);
        state_row[index] = updated;
        accumulated = __fadd_rn(accumulated,
            __fmul_rn(updated, selective_scan_value_f32(c_row[index])));
    }
    accumulated = block_sum<SELECTIVE_SCAN_DECODE_THREADS / 32>(accumulated);
    if (threadIdx.x == 0) {
        output[row] = __fadd_rn(accumulated, __fmul_rn(d[head], x_value));
    }
}

// Prefill: one block per (batch, head); warp w owns columns p = w, w + warps,
// ... and each lane owns state columns n = lane, lane + 32, ... .
template <class Value, int NPerLane>
__global__ void selective_state_scan_prefill_kernel(
    const Value* __restrict__ x,
    const Value* __restrict__ dt,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const Value* __restrict__ b,
    const Value* __restrict__ c,
    const float* __restrict__ d,
    const float* __restrict__ initial_state,
    float* __restrict__ output,
    float* __restrict__ final_state,
    int tokens,
    int heads,
    int columns,
    int state_width,
    int groups,
    float limit_lo,
    float limit_hi)
{
    const int head = blockIdx.x;
    const int batch = blockIdx.y;
    const int lane = threadIdx.x;
    const int warps = blockDim.y;
    const int group = head / (heads / groups);

    const float dt_bias_head = dt_bias[head];
    const float decay_scale = -expf(a_log[head]);
    const float skip = d[head];
    const Value* dt_head = dt + (int64_t)batch * tokens * heads + head;
    const Value* b_group = b + ((int64_t)batch * tokens * groups + group) * state_width;
    const Value* c_group = c + ((int64_t)batch * tokens * groups + group) * state_width;

    // Independent column groups occupy separate blocks instead of serializing
    // up to ten groups inside one head block on wide Mamba mixers.
    for (int column = blockIdx.z * warps + threadIdx.y;
         column < min(columns, int(blockIdx.z + 1) * warps); column += warps) {
        const int64_t state_row =
            ((int64_t)(batch * heads + head) * columns + column) * state_width;
        float carried[NPerLane];
        #pragma unroll
        for (int slot = 0; slot < NPerLane; ++slot) {
            const int index = lane + slot * SELECTIVE_SCAN_WARP;
            carried[slot] = initial_state == nullptr
                ? 0.0f : initial_state[state_row + index];
        }

        float* output_row = output +
            (((int64_t)batch * tokens * heads + head) * columns + column);
        for (int token = 0; token < tokens; ++token) {
            const float dt_value = selective_scan_step_dt(
                selective_scan_value_f32(dt_head[(int64_t)token * heads]),
                dt_bias_head, limit_lo, limit_hi);
            const float decay = expf(dt_value * decay_scale);
            const float x_value = selective_scan_value_f32(x[
                ((int64_t)batch * tokens + token) * heads * columns +
                (int64_t)head * columns + column]);
            const int64_t token_offset = (int64_t)token * groups * state_width;
            const float drive = dt_value * x_value;

            float accumulated = 0.0f;
            #pragma unroll
            for (int slot = 0; slot < NPerLane; ++slot) {
                const int index = token_offset + lane + slot * SELECTIVE_SCAN_WARP;
                const float updated = carried[slot] * decay +
                    drive * selective_scan_value_f32(b_group[index]);
                carried[slot] = updated;
                accumulated = __fadd_rn(accumulated,
                    __fmul_rn(updated, selective_scan_value_f32(c_group[index])));
            }
            accumulated = warp_sum(accumulated);
            if (lane == 0) {
                output_row[(int64_t)token * heads * columns] =
                    __fadd_rn(accumulated, __fmul_rn(skip, x_value));
            }
        }

        #pragma unroll
        for (int slot = 0; slot < NPerLane; ++slot) {
            const int index = lane + slot * SELECTIVE_SCAN_WARP;
            final_state[state_row + index] = carried[slot];
        }
    }
}

namespace {

namespace tb = mfq_tensor_backend;

bool selective_scan_supported(const tb::Tensor& tensor) {
    return tensor.scalar_type() == tb::kFloat32 ||
        tensor.scalar_type() == tb::kFloat16 ||
        tensor.scalar_type() == tb::kBFloat16;
}

void selective_scan_check_parameters(
    const char* name, const tb::Tensor& dt_bias, const tb::Tensor& a_log,
    const tb::Tensor& d, int64_t heads) {
    for (const tb::Tensor* parameter : {&dt_bias, &a_log, &d}) {
        MFQ_RUNTIME_CHECK(parameter->is_cuda() && parameter->is_contiguous() &&
            parameter->scalar_type() == tb::kFloat32 && parameter->dim() == 1 &&
            parameter->numel() == heads,
            name, ": dt_bias/A_log/D must be contiguous CUDA f32 [H] on the state device");
    }
}

}  // namespace

mfq_tensor_backend::Tensor selective_state_scan_decode_cuda(
    mfq_tensor_backend::Tensor state,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor dt,
    mfq_tensor_backend::Tensor dt_bias,
    mfq_tensor_backend::Tensor a_log,
    mfq_tensor_backend::Tensor b,
    mfq_tensor_backend::Tensor c,
    mfq_tensor_backend::Tensor d,
    double dt_limit_lo,
    double dt_limit_hi)
{
    const char* name = "selective_state_scan_decode";
    MFQ_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() &&
        state.scalar_type() == tb::kFloat32 && state.dim() == 4,
        name, ": state must be contiguous CUDA f32 [B,H,P,N]");
    const int64_t batches = state.size(0);
    const int64_t heads = state.size(1);
    const int64_t columns = state.size(2);
    const int64_t state_width = state.size(3);
    MFQ_RUNTIME_CHECK(batches > 0 && heads > 0 && columns > 0 && state_width > 0,
        name, ": state dimensions must be positive");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 &&
        x.size(0) == batches && x.size(1) == heads && x.size(2) == columns,
        name, ": x must be contiguous CUDA [B,H,P] matching the state");
    MFQ_RUNTIME_CHECK(dt.is_cuda() && dt.is_contiguous() && dt.dim() == 2 &&
        dt.size(0) == batches && dt.size(1) == heads,
        name, ": dt must be contiguous CUDA [B,H] matching the state");
    MFQ_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous() && b.dim() == 3 &&
        b.size(0) == batches && b.size(2) == state_width && b.sizes() == c.sizes(),
        name, ": B/C must be contiguous CUDA [B,G,N] matching the state width");
    MFQ_RUNTIME_CHECK(c.is_cuda() && c.is_contiguous(),
        name, ": C must be contiguous CUDA");
    const int64_t groups = b.size(1);
    MFQ_RUNTIME_CHECK(groups > 0 && heads % groups == 0,
        name, ": head count must be divisible by the B/C group count");
    MFQ_RUNTIME_CHECK(selective_scan_supported(x) &&
        dt.scalar_type() == x.scalar_type() && b.scalar_type() == x.scalar_type() &&
        c.scalar_type() == x.scalar_type(),
        name, ": x/dt/B/C must share one of f32, f16 or bf16");
    for (const tb::Tensor* tensor : {&x, &dt, &b, &c}) {
        MFQ_RUNTIME_CHECK(tensor->device() == state.device(),
            name, ": every input must live on the state device");
    }
    selective_scan_check_parameters(name, dt_bias, a_log, d, heads);
    MFQ_RUNTIME_CHECK(dt_bias.device() == state.device(),
        name, ": parameters must live on the state device");
    MFQ_RUNTIME_CHECK(dt_limit_lo <= dt_limit_hi,
        name, ": dt_limit_lo must not exceed dt_limit_hi");

    auto output = tb::empty({batches, heads, columns},
        state.options().dtype(tb::kFloat32));
    const dim3 grid((unsigned)columns, (unsigned)heads, (unsigned)batches);
    const auto launch = [&]<class Value>() {
        selective_state_scan_decode_kernel<Value><<<
            grid, SELECTIVE_SCAN_DECODE_THREADS, 0, mfq_current_cuda_stream()>>>(
            state.data_ptr<float>(),
            static_cast<const Value*>(x.data_ptr()),
            static_cast<const Value*>(dt.data_ptr()),
            dt_bias.data_ptr<float>(),
            a_log.data_ptr<float>(),
            static_cast<const Value*>(b.data_ptr()),
            static_cast<const Value*>(c.data_ptr()),
            d.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)heads, (int)columns, (int)state_width, (int)groups,
            (float)dt_limit_lo, (float)dt_limit_hi);
    };
    if (x.scalar_type() == tb::kFloat32) {
        launch.template operator()<float>();
    } else if (x.scalar_type() == tb::kFloat16) {
        launch.template operator()<__half>();
    } else {
        launch.template operator()<__hip_bfloat16>();
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<mfq_tensor_backend::Tensor> selective_state_scan_prefill_cuda(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor dt,
    mfq_tensor_backend::Tensor dt_bias,
    mfq_tensor_backend::Tensor a_log,
    mfq_tensor_backend::Tensor b,
    mfq_tensor_backend::Tensor c,
    mfq_tensor_backend::Tensor d,
    mfq_tensor_backend::Tensor initial_state,
    double dt_limit_lo,
    double dt_limit_hi)
{
    const char* name = "selective_state_scan_prefill";
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 4,
        name, ": x must be contiguous CUDA [B,T,H,P]");
    const int64_t batches = x.size(0);
    const int64_t tokens = x.size(1);
    const int64_t heads = x.size(2);
    const int64_t columns = x.size(3);
    MFQ_RUNTIME_CHECK(batches > 0 && tokens > 0 && heads > 0 && columns > 0,
        name, ": x dimensions must be positive");
    MFQ_RUNTIME_CHECK(dt.is_cuda() && dt.is_contiguous() && dt.dim() == 3 &&
        dt.size(0) == batches && dt.size(1) == tokens && dt.size(2) == heads,
        name, ": dt must be contiguous CUDA [B,T,H] matching x");
    MFQ_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous() && c.is_cuda() &&
        c.is_contiguous() && b.dim() == 4 && b.sizes() == c.sizes() &&
        b.size(0) == batches && b.size(1) == tokens,
        name, ": B/C must be contiguous CUDA [B,T,G,N] with matching tokens");
    const int64_t groups = b.size(2);
    const int64_t state_width = b.size(3);
    MFQ_RUNTIME_CHECK(groups > 0 && state_width > 0 && heads % groups == 0,
        name, ": head count must be divisible by the B/C group count");
    MFQ_RUNTIME_CHECK(state_width % SELECTIVE_SCAN_WARP == 0,
        name, ": state width N must be a multiple of 32");
    MFQ_RUNTIME_CHECK(selective_scan_supported(x) &&
        dt.scalar_type() == x.scalar_type() && b.scalar_type() == x.scalar_type() &&
        c.scalar_type() == x.scalar_type(),
        name, ": x/dt/B/C must share one of f32, f16 or bf16");
    selective_scan_check_parameters(name, dt_bias, a_log, d, heads);
    MFQ_RUNTIME_CHECK(dt_limit_lo <= dt_limit_hi,
        name, ": dt_limit_lo must not exceed dt_limit_hi");
    const bool has_initial_state = initial_state.defined();
    MFQ_RUNTIME_CHECK(!has_initial_state ||
        (initial_state.is_cuda() && initial_state.is_contiguous() &&
            initial_state.scalar_type() == tb::kFloat32 &&
            initial_state.sizes().vec() ==
                std::vector<int64_t>({batches, heads, columns, state_width})),
        name, ": initial_state must be contiguous CUDA f32 [B,H,P,N]");

    auto options = tb::TensorOptions().dtype(tb::kFloat32).device(x.device());
    auto output = tb::empty({batches, tokens, heads, columns}, options);
    auto final_state = has_initial_state
        ? initial_state.clone() : tb::zeros({batches, heads, columns, state_width}, options);
    const dim3 grid((unsigned)heads, (unsigned)batches,
        (unsigned)((columns + SELECTIVE_SCAN_WARPS - 1) / SELECTIVE_SCAN_WARPS));
    const dim3 block(SELECTIVE_SCAN_WARP, SELECTIVE_SCAN_WARPS);
    const int n_per_lane = (int)(state_width / SELECTIVE_SCAN_WARP);
    const auto launch_width = [&]<class Value, int Width>() {
        selective_state_scan_prefill_kernel<Value, Width><<<
            grid, block, 0, mfq_current_cuda_stream()>>>(
            static_cast<const Value*>(x.data_ptr()),
            static_cast<const Value*>(dt.data_ptr()),
            dt_bias.data_ptr<float>(),
            a_log.data_ptr<float>(),
            static_cast<const Value*>(b.data_ptr()),
            static_cast<const Value*>(c.data_ptr()),
            d.data_ptr<float>(),
            has_initial_state ? final_state.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            final_state.data_ptr<float>(),
            (int)tokens, (int)heads, (int)columns,
            (int)state_width, (int)groups,
            (float)dt_limit_lo, (float)dt_limit_hi);
    };
    const auto launch = [&]<class Value>() {
        switch (n_per_lane) {
        case 1: launch_width.template operator()<Value, 1>(); break;
        case 2: launch_width.template operator()<Value, 2>(); break;
        case 4: launch_width.template operator()<Value, 4>(); break;
        case 8: launch_width.template operator()<Value, 8>(); break;
        case 16: launch_width.template operator()<Value, 16>(); break;
        case 32: launch_width.template operator()<Value, 32>(); break;
        default:
            MFQ_RUNTIME_CHECK(false,
                name, ": state width N must be 32 * {1,2,4,8,16,32}");
        }
    };
    if (x.scalar_type() == tb::kFloat32) {
        launch.template operator()<float>();
    } else if (x.scalar_type() == tb::kFloat16) {
        launch.template operator()<__half>();
    } else {
        launch.template operator()<__hip_bfloat16>();
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, final_state};
}
