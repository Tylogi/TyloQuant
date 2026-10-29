#include "../runtime/execution_options.h"
#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_quant_ops.h"
#include "mfq_cuda_nint_route_hint.h"
#include "storage/weight_loader.h"
#include "quant_linear.h"
#include "mfq_weight_memory.h"

#include "format.h"
#include "fp8_sq.h"
#include "mfe_weight.h"
#include "mx.h"
#include "mxfp4_sq.h"
#include "nint.h"
#include "vq.h"
#include "mfq_format_compat.h"
#include "mfe_expert_store.h"
#include "nvq_codebooks.generated.h"
#include "mfq/mxfp4_sq_decode.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define MFQ_CPU_X86_GNU 1
#endif

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

int64_t read_i64_from(const std::vector<uint8_t> & b, size_t & off) {
    int64_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

uint32_t read_u32_from(const std::vector<uint8_t> & b, size_t & off) {
    uint32_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

const mfq::TensorMetadata& require_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto* tensor = source.find_tensor(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + std::string(name));
    }
    return *tensor;
}

bool has_tensor(
        const mfq::ModelSource& source,
        std::string_view name) noexcept {
    return source.find_tensor(name) != nullptr;
}

bool has_tensor_prefix(
        const mfq::ModelSource& source,
        std::string_view prefix) {
    for (const auto& tensor : source.tensors()) {
        if (tensor.name == prefix ||
            (tensor.name.size() > prefix.size() &&
             tensor.name.compare(0, prefix.size(), prefix) == 0 &&
             tensor.name[prefix.size()] == '.')) {
            return true;
        }
    }
    return false;
}

std::vector<std::uint8_t> read_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto& tensor = require_tensor(source, name);
    if (tensor.nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("model tensor is too large to read");
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(tensor.nbytes));
    source.read_range_into(
        name, 0, reinterpret_cast<std::byte*>(result.data()), result.size());
    return result;
}

static std::vector<std::uint8_t> read_tensor(
        bool drop_file_cache,
        const mfq::ModelSource& source,
        std::string_view name) {
    auto result = read_tensor(source, name);
    if (drop_file_cache) source.drop_file_cache();
    return result;
}

std::vector<std::uint8_t> read_asset(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    std::vector<std::uint8_t> result(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

std::string read_asset_text(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    return {
        reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
    CudaExecutionContext& execution,
    std::vector<mfq_tensor_backend::Tensor> outputs);

std::vector<mfq::TensorParallelSlice>
plan_moe_expert_parallel_slices(
    const ParallelConfig& parallel,
    int64_t extent,
    const std::string & name);

static mfq_tensor_backend::Tensor dequant_nint_dense_f32(
    const NintWeight & weight);

mfq_tensor_backend::Tensor load_dense_gpu(
        CudaExecutionContext& execution,
        const mfq::ModelSource& mfq,
        const std::string& name) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    const bool cpu_layer = execution.loading_cpu_layer;
    MfqCudaGuard guard(active_weight_load_device(execution));
    const auto & rec = require_tensor(mfq, name);
    if(rec.dtype=="TQ") {
        // This entry is for parameters explicitly consumed as dense tensors
        // (norm/function tables), not ordinary quantized projections.
        const auto weight=load_tq_weight(mfq,name,active_weight_load_device(execution));
        auto ids=mfq_tensor_backend::arange(weight.out,mfq_tensor_backend::TensorOptions()
            .device(weight.image.device()).dtype(mfq_tensor_backend::kInt64));
        auto value=weight.embedding(ids).to(mfq_tensor_backend::kFloat32).contiguous();
        return cpu_layer?value.cpu().contiguous():value;
    }
    if (rec.dtype == "MFE") {
        auto dense = materialize_mfe_dense(mfq, name)
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
        return cpu_layer ? dense.cpu().contiguous() : dense;
    }
    if (rec.dtype != "NINT8-0" && rec.dtype != "NINT") {
        auto value = load_dense_cpu(execution, mfq, name);
        if (rec.dtype == "BF16" || rec.dtype == "F16")
            value = value.to(mfq_tensor_backend::kFloat32);
        return cpu_layer ? value : value.to(mfq_tensor_backend::kCUDA).contiguous();
    }
    auto blob = read_tensor(execution.drop_file_cache, mfq, name);
    if (rec.dtype == "NINT8-0") {
        const auto source = unpack_nint8_zero(blob);
        if (cpu_layer) {
            return dequant_nint8_zero_cpu(source);
        }
        const auto packed = to_gpu_nint8_zero(source);
        auto dense = nint8_zero_dequant_f32_cuda(
            packed.q_packed,
            packed.q8_zero_scale,
            packed.neuron_len)
            .contiguous();
        if (packed.shape.size() != 2 ||
            dense.size(0) != packed.shape[0] ||
            dense.size(1) != packed.shape[1]) {
            throw std::runtime_error(
                "NINT8-0 dense tensor shape mismatch: " + name);
        }
        return dense;
    }
    auto dense = dequant_nint_dense_f32(to_gpu_nint(unpack_nint(blob)));
    return cpu_layer
        ? dense.cpu().contiguous()
        : dense;
}

static NintWeight cat_weights(const std::vector<NintWeight> & ws) {
    if (ws.empty()) throw std::runtime_error("empty NINT group");
    const auto & a = ws[0];
    std::vector<mfq_tensor_backend::Tensor> qp, rqb, rqoff, q8s, ss, sm, ns, nm;
    int64_t out = 0;
    int64_t q_bit_base = 0;
    bool aligned_q8 = !a.q8_zero;
    int uniform_qbits=a.uniform_qbits;
    for (const auto & w : ws) {
        if(w.uniform_qbits!=uniform_qbits)uniform_qbits=0;
        if (w.ng != a.ng || w.gs != a.gs ||
            w.neuron_len != a.neuron_len || w.q8_zero != a.q8_zero) {
            throw std::runtime_error("cannot group NINT tensors with different input layout");
        }
        qp.push_back(w.q_packed);
        if (!w.q8_zero) {
            aligned_q8 = aligned_q8 && w.aligned_q8 && (q_bit_base & 31) == 0;
            rqb.push_back(w.row_q_bits);
            rqoff.push_back(w.row_q_bit_offsets + q_bit_base);
            q_bit_base += w.q_packed.numel() * 8;
        }
        if (w.q8_zero) {
            q8s.push_back(w.q8_zero_scale);
            out += w.out;
            continue;
        }
        ss.push_back(w.sub_scale);
        sm.push_back(w.sub_min);
        ns.push_back(w.neuron_scale);
        nm.push_back(w.neuron_min);
        out += w.out;
    }
    NintWeight g;
    g.out = out;
    g.ng = a.ng;
    g.gs = a.gs;
    g.bits = a.bits;
    g.neuron_len = a.neuron_len;
    g.q8_zero = a.q8_zero;
    g.aligned_q8 = aligned_q8;
    g.uniform_qbits=uniform_qbits;
    g.shape = a.shape;
    g.shape[0] = out;
    g.q_packed = mfq_tensor_backend::cat(qp, 0).contiguous();
    if (!a.q8_zero) {
        g.row_q_bits = mfq_tensor_backend::cat(rqb, 0).contiguous();
        g.row_q_bit_offsets = mfq_tensor_backend::cat(rqoff, 0).contiguous();
    }
    if (a.q8_zero) {
        g.q8_zero_scale = mfq_tensor_backend::cat(q8s, 0).contiguous();
        return g;
    }
    g.sub_scale = mfq_tensor_backend::cat(ss, 0).contiguous();
    g.sub_min = mfq_tensor_backend::cat(sm, 0).contiguous();
    g.neuron_scale = mfq_tensor_backend::cat(ns, 0).contiguous();
    g.neuron_min = mfq_tensor_backend::cat(nm, 0).contiguous();
    return g;
}

static NintLinearGroup make_linear_group(const std::vector<NintWeight> & ws) {
    NintLinearGroup g;
    for (const auto & w : ws) g.outs.push_back(w.out);
    bool same = !ws.empty();
    for (size_t i = 1; i < ws.size(); ++i) {
        if (ws[i].ng != ws[0].ng || ws[i].gs != ws[0].gs ||
            ws[i].neuron_len != ws[0].neuron_len ||
            ws[i].q8_zero != ws[0].q8_zero) {
            same = false;
            break;
        }
    }
    if (same) {
        g.w = cat_weights(ws);
        if (g.w.q8_zero) {
            g.projection_w.reserve(ws.size());
            int64_t offset = 0;
            for (const auto & source : ws) {
                NintWeight projection = g.w;
                projection.out = source.out;
                projection.shape = source.shape;
                projection.q_packed =
                    g.w.q_packed.narrow(0, offset, source.out);
                projection.q8_zero_scale =
                    g.w.q8_zero_scale.narrow(0, offset, source.out);
                projection.workspaces.clear();
                g.projection_w.push_back(std::move(projection));
                offset += source.out;
            }
            MFQ_RUNTIME_CHECK(offset == g.w.out,
                        "Q8 projection views do not cover grouped output");
        }
    } else {
        std::vector<NintWeight> cur;
        std::vector<int64_t> cur_outs;
        auto flush = [&]() {
            if (cur.empty()) return;
            g.split_w.push_back(cur.size() == 1 ? cur[0] : cat_weights(cur));
            g.split_outs.push_back(cur_outs);
            cur.clear();
            cur_outs.clear();
        };
        for (const auto & w : ws) {
            bool append = !cur.empty() &&
                w.ng == cur[0].ng && w.gs == cur[0].gs &&
                w.neuron_len == cur[0].neuron_len &&
                w.q8_zero == cur[0].q8_zero;
            if (!append) flush();
            cur.push_back(w);
            cur_outs.push_back(w.out);
        }
        flush();
    }
    return g;
}

template <typename Decode>
static mfq_tensor_backend::Tensor sq_matmul_cpu_impl(
        mfq_tensor_backend::Tensor input,
        int64_t outputs,
        int64_t width,
        Decode decode) {
    MFQ_RUNTIME_CHECK(
        !input.is_cuda() && input.size(-1) == width,
        "SQ CPU activation device or width mismatch");
    const auto dtype = input.scalar_type();
    auto shape = input.sizes().vec();
    input = input.to(mfq_tensor_backend::kFloat32)
        .reshape({-1, width}).contiguous();
    const auto rows = input.size(0);
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    const auto* source = input.data_ptr<float>();
    auto* destination = result.data_ptr<float>();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> weights(width);
        for (auto output = begin; output < end; ++output) {
            decode(output, weights.data());
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0;
                for (int64_t column = 0; column < width; ++column) {
                    sum = std::fma(
                        source[row * width + column], weights[column], sum);
                }
                destination[row * outputs + output] = sum;
            }
        }
    });
    shape.back() = outputs;
    return result.to(dtype).reshape(shape);
}

mfq_tensor_backend::Tensor sq_matmul_cpu(
        const Mxfp4SqWeight& weight,
        mfq_tensor_backend::Tensor input) {
    MFQ_RUNTIME_CHECK(!weight.blob.is_cuda(), "SQ CPU weights must be on CPU");
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::sq::parse(blob, weight.blob.numel());
    return sq_matmul_cpu_impl(
        std::move(input), weight.out, weight.neuron_len,
        [&](int64_t row, float* output) {
            mfq::sq::decode_cpu_row(
                blob, layout, weight.row_q.data_ptr<uint8_t>()[row],
                weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                weight.row_auxiliary.data_ptr<int32_t>()[row], output);
        });
}

mfq_tensor_backend::Tensor sq_matmul_cpu(
        const Fp8SqWeight& weight,
        mfq_tensor_backend::Tensor input) {
    MFQ_RUNTIME_CHECK(!weight.blob.is_cuda(), "SQ CPU weights must be on CPU");
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::fp8sq::parse(
        weight.dtype, blob, weight.blob.numel());
    return sq_matmul_cpu_impl(
        std::move(input), weight.out, weight.neuron_len,
        [&](int64_t row, float* output) {
            for (int64_t column = 0; column < weight.neuron_len; ++column) {
                output[column] = mfq::fp8sq::decode_cpu(
                    blob, layout, weight.row_q.data_ptr<uint8_t>()[row],
                    weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                    row, column);
            }
        });
}

// Execution policy stays here; the format operators only receive a profiler.
mfq_tensor_backend::Tensor run_nint_linear(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        int mode) {
    if (!x.is_cuda() || !w.q8_zero || kl_mmq.mode == KlMmqMode::Default) {
        return gate.has_value()
            ? nint_matmul_input_mul(profiler, w, x, gate.value(), mode)
            : nint_matmul(profiler, w, x);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    if (gate.has_value()) {
        MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "NINT input gate mode must be sigmoid or SiLU");
        auto g = gate.value().contiguous().to(mfq_tensor_backend::kFloat16);
        MFQ_RUNTIME_CHECK(x.sizes() == g.sizes(), "NINT x and gate shapes must match");
        x = mode == 1 ? x * mfq_tensor_backend::sigmoid(g) : x * mfq_tensor_backend::silu(g);
    }
    x = pad_last(x, w.neuron_len);
    const int64_t rows = x.size(0);
    MFQ_RUNTIME_CHECK(rows > 0, "KLD NINT8-0 MMQ requires activation rows");
    if (rows < 16) {
        x = mfq_tensor_backend::cat({x, mfq_tensor_backend::zeros(
            {16 - rows, x.size(1)}, x.options())}, 0).contiguous();
    }
    x = kl_mmq.prepare_activation(x);
    ++kl_mmq.dense_calls;
    auto result = profiler.measure("kld_mmq.nint8_zero.fp16", [&]() {
        return nint8_zero_mmq_f16_packed_cuda(
            w.q_packed, w.q8_zero_scale, x, w.neuron_len);
    });
    return rows < 16 ? result.narrow(0, 0, rows).contiguous() : result;
}

mfq_tensor_backend::Tensor run_nvq_linear(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        const NvqWeight& w,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        int mode) {
    if (!x.is_cuda() || kl_mmq.mode == KlMmqMode::Default) {
        return gate.has_value()
            ? nvq_matmul_input_mul(profiler, w, x, gate.value(), mode)
            : nvq_matmul(profiler, w, x);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    if (gate.has_value()) {
        MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ input gate mode must be sigmoid or SiLU");
        auto g = gate.value().contiguous().to(mfq_tensor_backend::kFloat16);
        MFQ_RUNTIME_CHECK(x.sizes() == g.sizes(), "NVQ x and gate shapes must match");
        x = mode == 1 ? x * mfq_tensor_backend::sigmoid(g) : x * mfq_tensor_backend::silu(g);
    }
    x = pad_last(x, w.neuron_len);
    MFQ_RUNTIME_CHECK(x.size(0) >= 16, "KLD common NVQ MMQ requires at least 16 activation rows");
    x = kl_mmq.prepare_activation(x);
    ++kl_mmq.dense_calls;
    return profiler.measure("kld_mmq.nvq.fp16", [&]() {
        return nvq_gemm_f16_cuda(
            w.indices_packed, w.aux_packed, w.sub_scale_packed,
            w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
            w.sub_bits, w.kernel_format, w.sign_mode);
    });
}

mfq_tensor_backend::Tensor run_quant_linear_shard(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        const QuantLinearShard & shard,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        int gate_mode) {
    MfqCudaGuard guard(shard.device);
    if (shard.kind == QuantLinearKind::Mxfp4Sq ||
            shard.kind == QuantLinearKind::Fp8Sq) {
        if (gate.has_value()) {
            MFQ_RUNTIME_CHECK(
                gate_mode == 1 || gate_mode == 2,
                "SQ input gate mode must be sigmoid or SiLU");
            x = x * (gate_mode == 1
                ? mfq_tensor_backend::sigmoid(gate.value())
                : mfq_tensor_backend::silu(gate.value()));
        }
        return shard.kind == QuantLinearKind::Mxfp4Sq
            ? shard.mxfp4_sq.forward(x)
            : shard.fp8_sq.forward(x);
    }
    if (shard.kind == QuantLinearKind::Nint) {
        return run_nint_linear(profiler, kl_mmq, shard.nint, x, gate, gate_mode);
    }
    if (shard.kind == QuantLinearKind::Nvq) {
        return run_nvq_linear(profiler, kl_mmq, shard.nvq, x, gate, gate_mode);
    }
    if (shard.kind == QuantLinearKind::Mxfp4) {
        MFQ_RUNTIME_CHECK(
            !gate.has_value(),
            "MXFP4 tensor-parallel linear does not support input gating");
        return mxfp4_matmul(shard.mxfp4, x);
    }
    if (shard.kind == QuantLinearKind::Dense) {
        auto local = x.to(shard.dense.scalar_type());
        if (gate.has_value()) {
            MFQ_RUNTIME_CHECK(
                gate_mode == 1 || gate_mode == 2,
                "dense input gate mode must be sigmoid or SiLU");
            auto local_gate = gate.value().to(local.scalar_type());
            local = gate_mode == 1
                ? local * mfq_tensor_backend::sigmoid(local_gate)
                : local * mfq_tensor_backend::silu(local_gate);
        }
        return dense_projection(std::move(local), shard.dense);
    }
    MFQ_RUNTIME_CHECK(
        !gate.has_value(),
        "MXFP8 tensor-parallel linear does not support input gating");
    return mxfp8_matmul(profiler, shard.mxfp8, x);
}

mfq_tensor_backend::Tensor tensor_to_cuda_device(
        ModelParallelCollectiveRuntime& collectives,
        mfq_tensor_backend::Tensor value,
        int device,
        mfq_tensor_backend::Tensor reusable) {
    if (value.is_cuda() && value.get_device() == device) {
        return value.contiguous();
    }
    const auto reusable_matches = [&]() {
        return reusable.defined() && reusable.is_cuda() &&
            reusable.get_device() == device &&
            reusable.sizes() == value.sizes() &&
            reusable.scalar_type() == value.scalar_type() &&
            reusable.is_contiguous();
    };
    const auto destination_tensor = [&]() {
        if (reusable_matches()) return reusable;
        MfqCudaGuard destination_guard(device);
        return mfq_tensor_backend::empty(
            value.sizes(),
            value.options().device(
                mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, device)));
    };
    if (value.numel() == 0) {
        return destination_tensor();
    }
#if defined(MFQ_NATIVE_CUDA_RUNTIME) && defined(MFQ_HAVE_NCCL)
    auto& runtime = collectives;
    if (value.is_cuda() && runtime.collectives_enabled) {
        const int source_device = value.get_device();
        const auto source_rank_it = std::find(
            runtime.devices.begin(), runtime.devices.end(), source_device);
        const auto destination_rank_it = std::find(
            runtime.devices.begin(), runtime.devices.end(), device);
        if (source_rank_it != runtime.devices.end() &&
                destination_rank_it != runtime.devices.end()) {
            const auto source_stream =
                mfq_get_current_cuda_stream(source_device);
            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusNone;
            {
                MfqCudaGuard source_guard(source_device);
                MFQ_CUDA_CHECK(cudaStreamIsCapturing(
                    source_stream.stream(), &capture_status));
            }
            if (capture_status != cudaStreamCaptureStatusNone) {
                auto source = value.contiguous();
                auto destination = destination_tensor();
                const auto source_rank = static_cast<size_t>(
                    source_rank_it - runtime.devices.begin());
                const auto destination_rank = static_cast<size_t>(
                    destination_rank_it - runtime.devices.begin());
                {
                    MfqCudaGuard source_guard(source_device);
                    MFQ_CUDA_CHECK(cudaEventRecord(
                        runtime.ready[source_rank],
                        source_stream.stream()));
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams[source_rank].stream(),
                        runtime.ready[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    // Cross-device event waits make the receiving NCCL stream
                    // participate in the same capture before ncclRecv runs.
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams[destination_rank].stream(),
                        runtime.ready[source_rank], 0));
                }
                MFQ_NCCL_CHECK(ncclGroupStart());
                MFQ_NCCL_CHECK(ncclSend(
                    source.data_ptr(), source.nbytes(), ncclUint8,
                    static_cast<int>(destination_rank),
                    runtime.communicators[source_rank],
                    runtime.streams[source_rank].stream()));
                MFQ_NCCL_CHECK(ncclRecv(
                    destination.data_ptr(), destination.nbytes(), ncclUint8,
                    static_cast<int>(source_rank),
                    runtime.communicators[destination_rank],
                    runtime.streams[destination_rank].stream()));
                MFQ_NCCL_CHECK(ncclGroupEnd());
                {
                    MfqCudaGuard source_guard(source_device);
                    MFQ_CUDA_CHECK(cudaEventRecord(
                        runtime.completed[source_rank],
                        runtime.streams[source_rank].stream()));
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        source_stream.stream(),
                        runtime.completed[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    const auto destination_stream =
                        mfq_get_current_cuda_stream(device);
                    MFQ_CUDA_CHECK(cudaEventRecord(
                        runtime.completed[destination_rank],
                        runtime.streams[destination_rank].stream()));
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        destination_stream.stream(),
                        runtime.completed[destination_rank], 0));
                }
                return destination;
            }
        }
    }
#endif
    MfqCudaGuard guard(device);
    if (reusable_matches()) {
        reusable.copy_(value, true);
        return reusable;
    }
    return value.to(
        value.options().device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, device)),
        true, false).contiguous();
}

mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
        CudaExecutionContext& execution,
        std::vector<mfq_tensor_backend::Tensor> outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "cannot reduce an empty model-parallel output");
    }
#ifdef MFQ_HAVE_NCCL
    auto& runtime = execution.model_parallel_collectives;
    if (runtime.collectives_enabled &&
            outputs.size() == runtime.devices.size()) {
        const auto shape = outputs.front().sizes().vec();
        const auto output_dtype = outputs.front().scalar_type();
        const int64_t elements = outputs.front().numel();
        const bool reduce_to_primary =
            execution.config.model_parallel_reduce_to_primary;
        // A two-input FP16 sum has the same final FP16 rounding as the former
        // FP32 reduction, while avoiding both conversion passes.
        const bool fp16_reduce =
            execution.config.model_parallel_fp16_reduce &&
            reduce_to_primary &&
            outputs.size() == 2 &&
            output_dtype == mfq_tensor_backend::kFloat16;
        const int primary = model_parallel_primary_device(execution);
        const auto primary_rank_it = std::find(
            runtime.devices.begin(), runtime.devices.end(), primary);
        if (primary_rank_it == runtime.devices.end()) {
            throw std::runtime_error(
                "model-parallel primary device is absent from NCCL ranks");
        }
        const auto primary_rank = static_cast<size_t>(
            primary_rank_it - runtime.devices.begin());
        for (size_t index = 0; index < outputs.size(); ++index) {
            const int device = runtime.devices[index];
            if (!outputs[index].defined() || !outputs[index].is_cuda() ||
                    outputs[index].get_device() != device ||
                    outputs[index].sizes().vec() != shape ||
                    outputs[index].scalar_type() != output_dtype) {
                throw std::runtime_error(
                    "NCCL model-parallel reduction received mismatched shards");
            }
            MfqCudaGuard guard(device);
            if (fp16_reduce) {
                outputs[index] = outputs[index].contiguous();
            }
            const auto producer =
                mfq_get_current_cuda_stream(device);
            MFQ_CUDA_CHECK(cudaEventRecord(
                runtime.ready[index], producer.stream()));
            const auto communication = runtime.streams[index];
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                communication.stream(), runtime.ready[index], 0));
            if (fp16_reduce) {
                mfq_cuda_record_stream(outputs[index], communication);
            } else {
                MfqCudaStreamGuard stream_guard(communication);
                auto & buffer = runtime.reduction_buffers[index];
                if (!buffer.defined() || buffer.sizes().vec() != shape ||
                        buffer.get_device() != device ||
                        buffer.scalar_type() != mfq_tensor_backend::kFloat32) {
                    buffer = mfq_tensor_backend::empty(
                        shape,
                        outputs[index].options().dtype(mfq_tensor_backend::kFloat32));
                }
                buffer.copy_(outputs[index], true);
                mfq_cuda_record_stream(outputs[index], communication);
            }
        }

        MFQ_NCCL_CHECK(ncclGroupStart());
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto & buffer = runtime.reduction_buffers[index];
            if (reduce_to_primary) {
                void * reduction_data = fp16_reduce
                    ? outputs[index].data_ptr()
                    : buffer.data_ptr<float>();
                MFQ_NCCL_CHECK(ncclReduce(
                    reduction_data,
                    reduction_data,
                    static_cast<size_t>(elements),
                    fp16_reduce ? ncclFloat16 : ncclFloat32,
                    ncclSum,
                    static_cast<int>(primary_rank),
                    runtime.communicators[index],
                    runtime.streams[index].stream()));
            } else {
                MFQ_NCCL_CHECK(ncclAllReduce(
                    buffer.data_ptr<float>(),
                    buffer.data_ptr<float>(),
                    static_cast<size_t>(elements),
                    ncclFloat32,
                    ncclSum,
                    runtime.communicators[index],
                    runtime.streams[index].stream()));
            }
        }
        MFQ_NCCL_CHECK(ncclGroupEnd());

        mfq_tensor_backend::Tensor result;
        for (size_t index = 0; index < outputs.size(); ++index) {
            MfqCudaGuard guard(runtime.devices[index]);
            const auto communication = runtime.streams[index];
            if (runtime.devices[index] == primary) {
                if (fp16_reduce) {
                    result = outputs[index];
                } else {
                    MfqCudaStreamGuard stream_guard(communication);
                    result = runtime.reduction_buffers[index]
                        .to(output_dtype).contiguous();
                }
            }
            MFQ_CUDA_CHECK(cudaEventRecord(
                runtime.completed[index], communication.stream()));
        }
        MfqCudaGuard primary_guard(primary);
        const auto parent = mfq_get_current_cuda_stream(primary);
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(
            parent.stream(), runtime.completed[primary_rank], 0));
        mfq_cuda_record_stream(result, parent);
        return result;
    }
#endif
    const int primary = model_parallel_primary_device(execution);
    MfqCudaGuard primary_guard(primary);
    const auto output_dtype =
        outputs.front().scalar_type();
    mfq_tensor_backend::Tensor reduced;
    for (auto & output : outputs) {
        auto partial =
            tensor_to_cuda_device(
                execution.model_parallel_collectives, output, primary)
                .to(mfq_tensor_backend::kFloat32);
        if (!reduced.defined()) {
            reduced = std::move(partial);
        } else {
            reduced.add_(partial);
        }
    }
    return reduced.to(output_dtype).contiguous();
}

mfq_tensor_backend::Tensor quant_embedding_lookup(
        const QuantLinear & embedding,
        mfq_tensor_backend::Tensor token_ids) {
    MFQ_RUNTIME_CHECK(
        !embedding.tensor_parallel(),
        "quantized embedding does not support tensor-parallel shards");
    if (embedding.is_dense()) {
        auto output_shape = token_ids.sizes().vec();
        output_shape.push_back(embedding.dense.size(1));
        return embedding.dense.index_select(
            0, token_ids.reshape({-1})).reshape(output_shape);
    }
    if (embedding.is_tq()) return embedding.tq.embedding(token_ids);
    if (embedding.is_nvq()) {
        return nvq_embedding(embedding.nvq, token_ids);
    }
    if (embedding.is_nint()) {
        if (embedding.nint.q8_zero) {
            return nint8_zero_embedding_lookup_cuda(
                embedding.nint.q_packed,
                embedding.nint.q8_zero_scale,
                token_ids, embedding.nint.neuron_len);
        }
        return nint_embedding_cuda(
            embedding.nint.q_packed,
            embedding.nint.row_q_bits,
            embedding.nint.row_q_bit_offsets,
            embedding.nint.sub_scale,
            embedding.nint.sub_min,
            embedding.nint.neuron_scale,
            embedding.nint.neuron_min,
            token_ids, embedding.nint.neuron_len,
            embedding.nint.gs);
    }
    if (embedding.is_mxfp4()) {
        return mxfp4_embedding_lookup_cuda(
            embedding.mxfp4.weight.values,
            embedding.mxfp4.weight.scales, token_ids);
    }
    if (embedding.is_mxfp8()) {
        return mxfp8_embedding_lookup_cuda(
            embedding.mxfp8.weight.values,
            embedding.mxfp8.weight.scales, token_ids);
    }
    MFQ_RUNTIME_CHECK(
        !embedding.is_mxfp4_sq() && !embedding.is_fp8_sq(),
        "SQ tensors do not support embedding lookup");
    throw std::runtime_error("unsupported quantized embedding kind");
}

bool tensor_parallel_output_projections_compatible(
        const QuantLinearProjectionRefs & projections) {
    if (projections.size() < 2 ||
            !std::all_of(
                projections.begin(), projections.end(),
                [](const QuantLinear * layer) {
                    return layer != nullptr &&
                        layer->tensor_parallel() &&
                        layer->tensor_parallel_axis ==
                            TensorParallelAxis::Output;
                })) {
        return false;
    }
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (shard_count < 2) return false;
    for (const auto * layer : projections) {
        if (layer->tensor_parallel_shards.size() != shard_count) {
            return false;
        }
        for (size_t shard = 0; shard < shard_count; ++shard) {
            const auto & candidate =
                layer->tensor_parallel_shards[shard];
            const auto & reference =
                projections.front()->tensor_parallel_shards[shard];
            if (candidate.device != reference.device ||
                    candidate.input_begin != reference.input_begin ||
                    candidate.input_end != reference.input_end) {
                return false;
            }
        }
    }
    return true;
}

std::vector<mfq_tensor_backend::Tensor>
forward_tensor_parallel_output_projections(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        const QuantLinearProjectionRefs & projections) {
    MFQ_RUNTIME_CHECK(
        tensor_parallel_output_projections_compatible(projections),
        "tensor-parallel output projections are incompatible");
    auto output_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)});
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (execution.decode_graph_tp_projection_major) {
        std::vector<mfq_tensor_backend::Tensor> local_inputs(shard_count);
        for (size_t launch_position = 0;
             launch_position < shard_count; ++launch_position) {
            const size_t shard = model_parallel_launch_index(
                execution.config, launch_position, shard_count);
            const int device =
                projections.front()->tensor_parallel_shards[shard].device;
            MfqCudaGuard guard(device);
            local_inputs[shard] = tensor_to_cuda_device(
                execution.model_parallel_collectives, flat, device);
        }

        const int primary = model_parallel_primary_device(execution);
        std::vector<mfq_tensor_backend::Tensor> result;
        result.reserve(projections.size());
        for (const auto * projection : projections) {
            std::vector<mfq_tensor_backend::Tensor> local_outputs(shard_count);
            for (size_t launch_position = 0;
                 launch_position < shard_count; ++launch_position) {
                const size_t shard = model_parallel_launch_index(
                execution.config, launch_position, shard_count);
                const auto & weight =
                    projection->tensor_parallel_shards[shard];
                MfqCudaGuard guard(weight.device);
                local_outputs[shard] = run_quant_linear_shard(
                    execution.profiler, execution.kl_mmq,
                    weight, local_inputs[shard]);
            }
            MfqCudaGuard primary_guard(primary);
            std::vector<mfq_tensor_backend::Tensor> gathered;
            gathered.reserve(shard_count);
            for (auto & output : local_outputs) {
                gathered.push_back(tensor_to_cuda_device(
                    execution.model_parallel_collectives,
                    output, primary));
            }
            auto combined = mfq_tensor_backend::cat(
                gathered, -1).contiguous();
            auto shape = output_shape;
            shape.back() = combined.size(-1);
            result.push_back(combined.reshape(shape));
        }
        return result;
    }
    std::vector<std::vector<mfq_tensor_backend::Tensor>>
        local_outputs(projections.size());
    for (auto & outputs : local_outputs) {
        outputs.resize(shard_count);
    }
    for (size_t launch_position = 0;
         launch_position < shard_count; ++launch_position) {
        const size_t shard = model_parallel_launch_index(
                execution.config, launch_position, shard_count);
        const int device =
            projections.front()->tensor_parallel_shards[shard].device;
        MfqCudaGuard guard(device);
        // All projections consume the same immutable activation. Transfer it
        // once per rank, then keep the independent local projection work on
        // that rank before gathering each output.
        auto local_x = tensor_to_cuda_device(
            execution.model_parallel_collectives, flat, device);
        for (size_t projection = 0;
                projection < projections.size(); ++projection) {
            local_outputs[projection][shard] =
                run_quant_linear_shard(
                    execution.profiler, execution.kl_mmq,
                    projections[projection]
                        ->tensor_parallel_shards[shard], local_x);
        }
    }

    const int primary = model_parallel_primary_device(execution);
    MfqCudaGuard primary_guard(primary);
    std::vector<mfq_tensor_backend::Tensor> result;
    result.reserve(projections.size());
    for (auto & projection_outputs : local_outputs) {
        std::vector<mfq_tensor_backend::Tensor> gathered;
        gathered.reserve(shard_count);
        for (auto & output : projection_outputs) {
            gathered.push_back(tensor_to_cuda_device(
                execution.model_parallel_collectives,
                output, primary));
        }
        auto combined = mfq_tensor_backend::cat(
            gathered, -1).contiguous();
        auto shape = output_shape;
        shape.back() = combined.size(-1);
        result.push_back(combined.reshape(shape));
    }
    return result;
}

static bool is_nint_linear_dtype(const std::string & dtype) {
    return dtype == "NINT" || dtype == "NINT8-0";
}

static bool is_nvq_linear_dtype(const std::string & dtype) {
    return dtype == "NVQ" || dtype == "NPQ";
}

static TensorParallelAxis infer_tensor_parallel_axis(
        const std::string & name) {
    const auto ends_with = [&name](std::string_view suffix) {
        return name.size() >= suffix.size() &&
            std::string_view(name).substr(name.size() - suffix.size()) == suffix;
    };
    if (name == "model.token_embedding.weight" ||
        ends_with(".token_embedding.weight") ||
        ends_with(".text_embedding.weight") ||
        (name.find(".code_embedding.") != std::string::npos &&
         ends_with(".weight"))) {
        return TensorParallelAxis::Mirrored;
    }
    if (name == "model.output.weight" ||
        name.find(".code_output.") != std::string::npos) {
        return TensorParallelAxis::Output;
    }
    if (ends_with(".mlp.down.weight") ||
        ends_with(".mlp.shared_expert.down.weight") ||
        ends_with(".attention.output.weight") ||
        ends_with(".linear_attention.output.weight") ||
        ends_with(".attention.output_a.weight") ||
        ends_with(".attention.output_b.weight") ||
        ends_with(".projector.output.weight")) {
        return TensorParallelAxis::Input;
    }
    return TensorParallelAxis::Output;
}

static std::vector<mfq::TensorParallelSlice>
plan_parallel_slices(
        int64_t extent,
        int64_t preferred_granularity,
        const ParallelConfig & config,
        const std::string & name) {
    (void)name;
    if (!config.enabled()) {
        throw std::runtime_error(
            "parallel slice planning requires at least two ranks");
    }
    int64_t granularity = std::max<int64_t>(
        1, std::min<int64_t>(
            preferred_granularity,
            extent / static_cast<int64_t>(config.devices.size())));
    while (granularity > 1 &&
           extent < static_cast<int64_t>(config.devices.size()) *
               granularity) {
        granularity /= 2;
    }
    auto slices = mfq::plan_tensor_parallel_slices(
        extent, granularity, config.devices, config.split);
    mfq::validate_tensor_parallel_slices(
        slices, extent, granularity);
    return slices;
}

std::vector<mfq::TensorParallelSlice>
plan_moe_expert_parallel_slices(
        const ParallelConfig& parallel,
        int64_t extent,
        const std::string & name) {
    return plan_parallel_slices(extent, 1, parallel, name);
}

QuantLinear load_quant_linear(
        CudaExecutionContext& execution,
        const mfq::ModelSource & mfq,
        const std::string & name,
        std::optional<TensorParallelAxis> axis_override,
        const std::vector<mfq::TensorParallelSlice> *
            slices_override) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    const bool cpu_layer = execution.loading_cpu_layer;
    const auto & dtype = require_tensor(mfq, name).dtype;
    QuantLinear result;
    const TensorParallelAxis axis =
        axis_override.value_or(
            infer_tensor_parallel_axis(name));
    if (slices_override != nullptr &&
        axis != TensorParallelAxis::Output) {
        throw std::runtime_error(
            "explicit tensor-parallel slices require an output-axis weight");
    }
    auto select_slices = [&](
            int64_t extent,
            int64_t preferred) {
        if (slices_override == nullptr) {
            return plan_parallel_slices(
                extent, preferred, execution.tensor_parallel, name);
        }
        auto slices = *slices_override;
        mfq::validate_tensor_parallel_slices(
            slices, extent, 1);
        if (slices.size() !=
            execution.tensor_parallel.devices.size()) {
            throw std::runtime_error(
                "explicit tensor-parallel slice count mismatch");
        }
        for (size_t index = 0;
             index < slices.size(); ++index) {
            if (slices[index].device !=
                execution.tensor_parallel.devices[index]) {
                throw std::runtime_error(
                    "explicit tensor-parallel device order mismatch");
            }
        }
        return slices;
    };
    result.tensor_parallel_axis = axis;
    if(dtype=="TQ") {
        result.kind=QuantLinearKind::Tq;
        result.tq=load_tq_weight(mfq,name,active_weight_load_device(execution),!cpu_layer);
        result.logical_out=result.tq.out;result.logical_neuron_len=result.tq.neuron_len;
        return result;
    }
    if (is_nint_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nint;
        const auto blob = read_tensor(
            execution.drop_file_cache, mfq, name);
        if (dtype == "NINT8-0") {
            const auto cpu = unpack_nint8_zero(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : 4;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint8_zero_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint8_zero_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * 32 : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * 32,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint =
                        to_cuda_device_nint8_zero(
                            shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                if (cpu_layer) {
                    result.nint = to_device_nint8_zero(cpu, false);
                } else {
                    MfqCudaGuard guard(active_weight_load_device(execution));
                    result.nint =
                        to_device_nint8_zero(cpu, true);
                }
            }
        } else {
            const auto cpu = unpack_nint(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * cpu.gs : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * cpu.gs,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint = to_cuda_device_nint(
                        shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                if (cpu_layer) {
                    result.nint = to_device_nint(cpu, false);
                } else {
                    MfqCudaGuard guard(active_weight_load_device(execution));
                    result.nint =
                        to_device_nint(cpu, true);
                }
            }
        }
    } else if (is_nvq_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nvq;
        const auto cpu = unpack_nvq(
            read_tensor(execution.drop_file_cache, mfq, name), dtype);
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (execution.tensor_parallel.enabled() &&
            axis != TensorParallelAxis::Mirrored) {
            const int64_t extent =
                axis == TensorParallelAxis::Output
                ? cpu.out : cpu.ng;
            const int64_t preferred =
                axis == TensorParallelAxis::Output
                ? 128
                : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
            for (const auto & slice :
                 select_slices(
                     extent, preferred)) {
                auto shard_cpu = slice_nvq_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Nvq;
                shard.output_begin =
                    axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end =
                    axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin =
                    axis == TensorParallelAxis::Input
                    ? slice.begin * cpu.gs : 0;
                shard.input_end =
                    axis == TensorParallelAxis::Input
                    ? std::min<int64_t>(
                        slice.end * cpu.gs,
                        cpu.neuron_len)
                    : cpu.neuron_len;
                shard.nvq = to_cuda_device_nvq(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else {
            if (cpu_layer) {
                result.nvq = to_device_nvq(cpu, false, execution.config);
            } else {
                MfqCudaGuard guard(active_weight_load_device(execution));
                result.nvq = to_device_nvq(cpu, true, execution.config);
            }
        }
    } else if (dtype == "MXFP4-SQ") {
        result.kind = QuantLinearKind::Mxfp4Sq;
        const auto payload = read_tensor(
            execution.drop_file_cache, mfq, name);
        const auto layout = mfq::sq::parse(payload.data(), payload.size());
        result.logical_out = layout.outputs;
        result.logical_neuron_len = layout.width;
        if (!cpu_layer && execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const bool output_axis = axis == TensorParallelAxis::Output;
            for (const auto& slice : select_slices(
                    output_axis ? layout.outputs : layout.width,
                    output_axis ? 128 : 32)) {
                QuantLinearShard shard;
                shard.kind = result.kind;
                shard.device = slice.device;
                shard.output_begin = output_axis ? slice.begin : 0;
                shard.output_end = output_axis ? slice.end : layout.outputs;
                shard.input_begin = output_axis ? 0 : slice.begin;
                shard.input_end = output_axis ? layout.width : slice.end;
                std::vector<int64_t> rows(
                    shard.output_end - shard.output_begin);
                std::iota(rows.begin(), rows.end(), shard.output_begin);
                shard.mxfp4_sq.weight = to_device_mxfp4_sq(
                    mfq::sq::select_rows(
                        payload, rows, shard.input_begin, shard.input_end),
                    true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.mxfp4_sq.weight = to_device_mxfp4_sq(
                payload, !cpu_layer,
                cpu_layer ? -1 : active_weight_load_device(execution));
        }
    } else if (mfq::fp8sq::is_dtype(dtype)) {
        result.kind = QuantLinearKind::Fp8Sq;
        const auto payload = read_tensor(
            execution.drop_file_cache, mfq, name);
        const auto layout = mfq::fp8sq::parse(
            dtype, payload.data(), payload.size());
        result.logical_out = layout.outputs;
        result.logical_neuron_len = layout.width;
        if (!cpu_layer && execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const bool output_axis = axis == TensorParallelAxis::Output;
            const auto alignment = output_axis
                ? layout.block_rows : layout.block_columns;
            for (const auto& slice : select_slices(
                    output_axis ? layout.outputs : layout.width, alignment)) {
                QuantLinearShard shard;
                shard.kind = result.kind;
                shard.device = slice.device;
                shard.output_begin = output_axis ? slice.begin : 0;
                shard.output_end = output_axis ? slice.end : layout.outputs;
                shard.input_begin = output_axis ? 0 : slice.begin;
                shard.input_end = output_axis ? layout.width : slice.end;
                shard.fp8_sq.weight = to_device_fp8_sq(
                    dtype, mfq::fp8sq::slice(
                        dtype, payload,
                        shard.output_begin, shard.output_end,
                        shard.input_begin, shard.input_end),
                    true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.fp8_sq.weight = to_device_fp8_sq(
                dtype, payload, !cpu_layer,
                cpu_layer ? -1 : active_weight_load_device(execution));
        }
    } else if (dtype == "MXFP4") {
        result.kind = QuantLinearKind::Mxfp4;
        const auto cpu = unpack_mxfp4(
            read_tensor(execution.drop_file_cache, mfq, name));
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            const int64_t preferred = axis == TensorParallelAxis::Output
                ? 128 : 32;
            for (const auto & slice : select_slices(extent, preferred)) {
                auto shard_cpu = slice_mxfp4_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Mxfp4;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.mxfp4 = to_device_mxfp4(
                    shard_cpu, true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.mxfp4.weight = to_device_mxfp4(
                cpu, !cpu_layer,
                cpu_layer ? -1 : active_weight_load_device(execution));
        }
    } else if (dtype == "MXFP8") {
        result.kind = QuantLinearKind::Mxfp8;
        const auto cpu = unpack_mxfp8(
            read_tensor(execution.drop_file_cache, mfq, name));
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            for (const auto & slice : select_slices(extent, 128)) {
                auto shard_cpu = slice_mxfp8_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Mxfp8;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.mxfp8 = to_cuda_device_mxfp8(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else if (cpu_layer) {
            result.mxfp8.weight = to_device_mxfp8(cpu, false);
        } else {
            result.mxfp8.weight = to_cuda_device_mxfp8(
                cpu, active_weight_load_device(execution));
        }
    } else if (dtype == "BF16" || dtype == "F16" || dtype == "F32") {
        result.kind = QuantLinearKind::Dense;
        auto cpu = load_dense_cpu(execution, mfq, name);
        MFQ_RUNTIME_CHECK(cpu.dim() == 2, "dense linear tensor must be rank 2: ", name);
        result.logical_out = cpu.size(0);
        result.logical_neuron_len = cpu.size(1);
        // Keep native floating-point linears whole on the primary TP rank.
        // Splitting these matrices changes the cuBLAS GEMM geometry and causes
        // materially larger drift than the weight formats under test. They
        // are a small fraction of the model; routed experts and MXFP8/NINT/NVQ
        // weights remain sharded.
        const bool shard_native_float =
            execution.config.tensor_parallel_shard_native_float;
        if (shard_native_float && execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.size(0) : cpu.size(1);
            for (const auto & slice : select_slices(extent, 128)) {
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Dense;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.size(0);
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.size(1);
                const int64_t dimension =
                    axis == TensorParallelAxis::Output ? 0 : 1;
                MfqCudaGuard guard(slice.device);
                shard.dense = cpu.narrow(
                        dimension, slice.begin,
                        slice.end - slice.begin)
                    .to(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, slice.device))
                    .contiguous();
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else if (cpu_layer) {
            result.dense = cpu.contiguous();
        } else {
            MfqCudaGuard guard(active_weight_load_device(execution));
            result.dense = cpu.to(mfq_tensor_backend::kCUDA).contiguous();
        }
    } else {
        throw std::runtime_error(
            "linear tensor must be NINT/NVQ/MXFP4-SQ/MXFP8-SQ/FP8-128SQ/MXFP4/MXFP8/BF16/F16/F32: " +
            name + " dtype=" + dtype);
    }
    return result;
}

bool is_quant_dtype(const std::string & dtype) {
    return dtype == "TQ" || is_nint_linear_dtype(dtype) ||
        is_nvq_linear_dtype(dtype) || dtype == "MXFP4-SQ" ||
        mfq::fp8sq::is_dtype(dtype) ||
        dtype == "MXFP4" || dtype == "MXFP8";
}

QuantLinearGroup make_quant_group(
        CudaExecutionContext& execution,
        std::vector<QuantLinear> layers,
        bool preserve_projection_boundaries) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    if (layers.empty()) throw std::runtime_error("empty quantized linear group");
    QuantLinearGroup result;
    result.decode_branch_parallel = !preserve_projection_boundaries;
    result.outs.reserve(layers.size());
    bool all_nint = true;
    std::vector<NintWeight> nint_weights;
    nint_weights.reserve(layers.size());
    for (const auto & layer : layers) {
        result.outs.push_back(layer.out());
        all_nint = all_nint && layer.is_nint();
        if (layer.is_nint() && !layer.tensor_parallel()) {
            nint_weights.push_back(layer.nint);
        }
    }
    const bool diagnostic_keep_nint_separate =
        !execution.config.diagnostic_nint_group;
    const bool cpu_layer = execution.loading_cpu_layer;
    const bool any_tensor_parallel = std::any_of(
        layers.begin(), layers.end(),
        [](const QuantLinear & layer) {
            return layer.tensor_parallel();
        });
    const bool all_tq = std::all_of(layers.begin(),layers.end(),
        [](const QuantLinear& layer){return layer.is_tq() && layer.tq.image.is_cuda();});
    if(all_tq && !any_tensor_parallel) {
        std::vector<TqWeight> weights;weights.reserve(layers.size());
        for(const auto& layer:layers)weights.push_back(layer.tq);
        result.tq=TqProjectionGroup(std::move(weights));
    }
    if (all_nint && !preserve_projection_boundaries &&
        !diagnostic_keep_nint_separate && !cpu_layer &&
        !any_tensor_parallel) {
        result.nint_grouped = true;
        result.nint = make_linear_group(nint_weights);
    } else {
        result.layers = std::move(layers);
        result.nvq_prefix2 = !cpu_layer && result.layers.size() >= 2 &&
            !any_tensor_parallel &&
            result.layers[0].is_nvq() && result.layers[1].is_nvq() &&
            nvq_pair_compatible(result.layers[0].nvq, result.layers[1].nvq);
    }
    return result;
}

static bool quant_linear_pair_compatible(const QuantLinear & a, const QuantLinear & b) {
    if (a.tensor_parallel() || b.tensor_parallel()) {
        return a.tensor_parallel() && b.tensor_parallel() &&
            a.kind == b.kind &&
            a.tensor_parallel_axis == b.tensor_parallel_axis &&
            a.tensor_parallel_shards.size() ==
                b.tensor_parallel_shards.size();
    }
    if (a.kind != b.kind) return false;
    if (a.is_tq()) return a.tq.neuron_len == b.tq.neuron_len;
    if (a.is_nvq()) return nvq_pair_compatible(a.nvq, b.nvq);
    if (a.is_mxfp8()) {
        return a.mxfp8.weight.neuron_len == b.mxfp8.weight.neuron_len;
    }
    if (a.is_mxfp4()) {
        return a.mxfp4.weight.neuron_len == b.mxfp4.weight.neuron_len;
    }
    if (a.is_mxfp4_sq()) {
        return a.mxfp4_sq.weight.neuron_len ==
            b.mxfp4_sq.weight.neuron_len;
    }
    if (a.is_fp8_sq()) {
        return a.fp8_sq.weight.dtype == b.fp8_sq.weight.dtype &&
            a.fp8_sq.weight.neuron_len == b.fp8_sq.weight.neuron_len &&
            a.fp8_sq.weight.block_rows == b.fp8_sq.weight.block_rows &&
            a.fp8_sq.weight.block_columns == b.fp8_sq.weight.block_columns &&
            a.fp8_sq.weight.scale_kind == b.fp8_sq.weight.scale_kind;
    }
    if (a.is_dense()) {
        return a.dense.size(1) == b.dense.size(1);
    }
    const auto & x = a.nint;
    const auto & y = b.nint;
    return x.ng == y.ng && x.gs == y.gs &&
        x.neuron_len == y.neuron_len && x.q8_zero == y.q8_zero;
}

QuantLinearGroup load_quant_group(
    CudaExecutionContext& execution,
    const mfq::ModelSource & mfq, const std::vector<std::string> & names,
    size_t required_compatible_prefix,
    const std::vector<mfq::TensorParallelSlice> *
        slices_override,
    bool preserve_projection_boundaries) {
    std::vector<QuantLinear> layers;
    layers.reserve(names.size());
    for (const auto & name : names) {
        layers.push_back(
            load_quant_linear(
                execution, mfq, name,
                slices_override != nullptr
                    ? std::optional<TensorParallelAxis>(
                        TensorParallelAxis::Output)
                    : std::nullopt,
                slices_override));
    }
    if (required_compatible_prefix > layers.size()) {
        throw std::runtime_error("invalid required quantized-group prefix length");
    }
    // A compatible prefix is fused opportunistically by make_quant_group().
    // Mixed-precision Q/K and gate/up pairs remain separate QuantLinear
    // branches and preserve the recipe-selected layouts.
    return make_quant_group(
        execution, std::move(layers), preserve_projection_boundaries);
}

static std::vector<mfq::TensorParallelSlice>
tensor_parallel_output_slices_for_input(
        const QuantLinear & input_parallel) {
    if (!input_parallel.tensor_parallel() ||
        input_parallel.tensor_parallel_axis !=
            TensorParallelAxis::Input) {
        return {};
    }
    std::vector<mfq::TensorParallelSlice> result;
    result.reserve(
        input_parallel.tensor_parallel_shards.size());
    for (const auto & shard :
         input_parallel.tensor_parallel_shards) {
        result.push_back({
            shard.device,
            shard.input_begin,
            shard.input_end,
        });
    }
    mfq::validate_tensor_parallel_slices(
        result,
        input_parallel.neuron_len(),
        1);
    return result;
}

QuantLinearGroup load_paired_gate_up(
        CudaExecutionContext& execution,
        const mfq::ModelSource & mfq,
        const std::vector<std::string> & names,
        const QuantLinear & down,
        size_t required_compatible_prefix,
        bool preserve_projection_boundaries) {
    // An ungated MLP has one Up projection; only the available prefix may fuse.
    required_compatible_prefix = std::min(required_compatible_prefix, names.size());
    auto slices =
        tensor_parallel_output_slices_for_input(
            down);
    return slices.empty()
        ? load_quant_group(
            execution, mfq, names,
            required_compatible_prefix,
            nullptr,
            preserve_projection_boundaries)
        : load_quant_group(
            execution, mfq, names,
            required_compatible_prefix,
            &slices,
            preserve_projection_boundaries);
}

DenseLinearGroup make_dense_group(const std::vector<mfq_tensor_backend::Tensor> & ws) {
    if (ws.empty()) throw std::runtime_error("empty dense group");
    DenseLinearGroup g;
    std::vector<mfq_tensor_backend::Tensor> parts;
    parts.reserve(ws.size());
    for (const auto & w : ws) {
        if (w.dim() != 2) throw std::runtime_error("dense linear group expects 2D weights");
        if (!parts.empty() && w.size(1) != parts[0].size(1)) {
            throw std::runtime_error("cannot group dense tensors with different input width");
        }
        g.outs.push_back(w.size(0));
        parts.push_back(w.to(mfq_tensor_backend::kFloat32));
    }
    g.w = mfq_tensor_backend::cat(parts, 0).contiguous();
    return g;
}

static mfq_tensor_backend::Tensor dequant_nint_dense_f32(const NintWeight & w) {
    if (w.q8_zero) {
        return nint8_zero_dequant_f32_cuda(
            w.q_packed, w.q8_zero_scale, w.neuron_len).contiguous();
    }
    return nint_decode_f32_cuda(
        w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
        w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
        w.neuron_len, w.gs).contiguous();
}

static mfq_tensor_backend::Tensor dequant_mxfp4_sq_cpu(
        const Mxfp4SqWeight& weight) {
    auto output = mfq_tensor_backend::empty(
        {weight.out, weight.neuron_len},
        weight.blob.options().dtype(mfq_tensor_backend::kFloat32));
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::sq::parse(blob, weight.blob.numel());
    mfq_parallel_for(0, weight.out, 1, [&](int64_t begin, int64_t end) {
        for (auto row = begin; row < end; ++row) {
            mfq::sq::decode_cpu_row(
                blob, layout, weight.row_q.data_ptr<uint8_t>()[row],
                weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                weight.row_auxiliary.data_ptr<int32_t>()[row],
                output.data_ptr<float>() + row * weight.neuron_len);
        }
    });
    return output;
}

static mfq_tensor_backend::Tensor dequant_quant_linear_f32(
        CudaExecutionContext& execution,
        const QuantLinear& linear) {
    if (!linear.tensor_parallel()) {
        if (linear.is_nint()) {
            return dequant_nint_dense_f32(linear.nint);
        }
        if (linear.is_nvq()) {
            return nvq_dequant(linear.nvq)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp8()) {
            return mxfp8_dequant_cuda(
                linear.mxfp8.weight.values,
                linear.mxfp8.weight.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp4()) {
            return mxfp4_dequant_cuda(
                linear.mxfp4.weight.values,
                linear.mxfp4.weight.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp4_sq()) {
            const auto & weight = linear.mxfp4_sq.weight;
            if (!weight.blob.is_cuda()) return dequant_mxfp4_sq_cpu(weight);
            return mxfp4_sq_dequant_cuda(
                weight.blob,
                weight.row_q,
                weight.row_symbol_byte_offsets,
                weight.row_auxiliary,
                weight.bits,
                weight.out,
                weight.neuron_len,
                weight.matrix_scale_base,
                weight.q_sum,
                weight.sq4_rows,
                true).contiguous();
        }
        if (linear.is_fp8_sq()) {
            return dequant_fp8_sq(linear.fp8_sq.weight, true).contiguous();
        }
        if (linear.is_dense()) {
            return linear.dense.to(mfq_tensor_backend::kFloat32).contiguous();
        }
        throw std::runtime_error(
            "unsupported linear kind for FP32 reconstruction");
    }
    if (linear.tensor_parallel_axis != TensorParallelAxis::Output &&
        linear.tensor_parallel_axis != TensorParallelAxis::Input) {
        throw std::runtime_error(
            "cannot reconstruct a mirrored tensor-parallel linear");
    }
    const int primary = model_parallel_primary_device(execution);
    std::vector<mfq_tensor_backend::Tensor> parts;
    parts.reserve(linear.tensor_parallel_shards.size());
    for (const auto & shard : linear.tensor_parallel_shards) {
        MfqCudaGuard shard_guard(shard.device);
        mfq_tensor_backend::Tensor part;
        if (shard.kind == QuantLinearKind::Nint) {
            part = dequant_nint_dense_f32(shard.nint);
        } else if (shard.kind == QuantLinearKind::Nvq) {
            part = nvq_dequant(shard.nvq)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp8) {
            part = mxfp8_dequant_cuda(
                shard.mxfp8.values,
                shard.mxfp8.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp4) {
            part = mxfp4_dequant_cuda(
                shard.mxfp4.values, shard.mxfp4.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp4Sq ||
                shard.kind == QuantLinearKind::Fp8Sq) {
            QuantLinear local;
            local.kind = shard.kind;
            local.mxfp4_sq = shard.mxfp4_sq;
            local.fp8_sq = shard.fp8_sq;
            part = dequant_quant_linear_f32(execution, local);
        } else if (shard.kind == QuantLinearKind::Dense) {
            part = shard.dense.to(mfq_tensor_backend::kFloat32).contiguous();
        } else {
            throw std::runtime_error(
                "unsupported tensor-parallel shard kind for reconstruction");
        }
        parts.push_back(
            tensor_to_cuda_device(
                execution.model_parallel_collectives, part, primary)
                .to(mfq_tensor_backend::kFloat32).contiguous());
    }
    MfqCudaGuard primary_guard(primary);
    auto dense = mfq_tensor_backend::cat(
        parts,
        linear.tensor_parallel_axis == TensorParallelAxis::Output
            ? 0 : 1).contiguous();
    if (dense.size(0) != linear.out() ||
        dense.size(1) != linear.neuron_len()) {
        throw std::runtime_error(
            "reconstructed tensor-parallel linear shape mismatch");
    }
    return dense;
}

DenseLinearGroup make_fp32_quant_group(
        CudaExecutionContext& execution,
        QuantLinearGroup group) {
    DenseLinearGroup dense;
    dense.outs = group.outs;
    if (group.nint_grouped) {
        if (group.nint.split_w.empty()) {
            dense.w = dequant_nint_dense_f32(group.nint.w);
        } else {
            std::vector<mfq_tensor_backend::Tensor> parts;
            parts.reserve(group.nint.split_w.size());
            for (const auto & weight : group.nint.split_w) {
                parts.push_back(dequant_nint_dense_f32(weight));
            }
            dense.w = mfq_tensor_backend::cat(parts, 0).contiguous();
        }
    } else {
        std::vector<mfq_tensor_backend::Tensor> parts;
        parts.reserve(group.layers.size());
        for (const auto & linear : group.layers) {
            parts.push_back(dequant_quant_linear_f32(
                execution, linear));
        }
        dense.w = mfq_tensor_backend::cat(parts, 0).contiguous();
    }
    MFQ_RUNTIME_CHECK(
        dense.w.dim() == 2 &&
            dense.w.size(0) ==
                std::accumulate(
                    dense.outs.begin(), dense.outs.end(), int64_t{0}),
        "FP32 compressor projection shape mismatch");
    return dense;
}

mfq_tensor_backend::Tensor quant_linear_reference_weight(
        const QuantLinear& linear) {
    if (linear.is_nint()) {
        const auto& weight = linear.nint;
        return weight.q8_zero
            ? nint8_zero_dequant_cuda(
                  weight.q_packed, weight.q8_zero_scale,
                  weight.neuron_len)
            : nint_decode_cuda(
                  weight.q_packed, weight.row_q_bits,
                  weight.row_q_bit_offsets, weight.sub_scale,
                  weight.sub_min, weight.neuron_scale,
                  weight.neuron_min, weight.neuron_len, weight.gs);
    }
    if (linear.is_nvq()) return nvq_dequant(linear.nvq);
    if (linear.is_mxfp4()) {
        return mxfp4_dequant_cuda(
            linear.mxfp4.weight.values, linear.mxfp4.weight.scales);
    }
    if (linear.is_mxfp4_sq()) {
        const auto& weight = linear.mxfp4_sq.weight;
        if (!weight.blob.is_cuda()) {
            return dequant_mxfp4_sq_cpu(weight)
                .to(mfq_tensor_backend::kFloat16);
        }
        return mxfp4_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.row_auxiliary, weight.bits, weight.out,
            weight.neuron_len, weight.matrix_scale_base,
            weight.q_sum, weight.sq4_rows, false);
    }
    if (linear.is_fp8_sq()) {
        return dequant_fp8_sq(linear.fp8_sq.weight, false);
    }
    if (linear.is_mxfp8()) {
        return mxfp8_cpu_reference(linear.mxfp8.weight);
    }
    if (linear.is_dense()) return linear.dense;
    throw std::runtime_error("unsupported linear reference format");
}

mfq_tensor_backend::Tensor QuantLinear::forward_tensor_parallel_flat(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        int gate_mode) const {
    MFQ_RUNTIME_CHECK(
        tensor_parallel(),
        "tensor-parallel linear has no shards");
    MFQ_RUNTIME_CHECK(
        tensor_parallel_axis == TensorParallelAxis::Output ||
        tensor_parallel_axis == TensorParallelAxis::Input,
        "tensor-parallel linear has an invalid axis");
    std::vector<mfq_tensor_backend::Tensor> local_outputs(
        tensor_parallel_shards.size());
    for (size_t launch_position = 0;
         launch_position < tensor_parallel_shards.size();
         ++launch_position) {
        const size_t index = model_parallel_launch_index(
            execution.config, launch_position, tensor_parallel_shards.size());
        const auto & shard = tensor_parallel_shards[index];
        MfqCudaGuard guard(shard.device);
        mfq_tensor_backend::Tensor local_x = x;
        mfq_tensor_backend::Tensor local_gate;
        if (tensor_parallel_axis == TensorParallelAxis::Input) {
            local_x = x.narrow(
                -1, shard.input_begin,
                shard.input_end - shard.input_begin);
            if (gate.has_value()) {
                local_gate = gate.value().narrow(
                    -1, shard.input_begin,
                    shard.input_end - shard.input_begin);
            }
        } else if (gate.has_value()) {
            local_gate = gate.value();
        }
        local_x = tensor_to_cuda_device(
            execution.model_parallel_collectives,
            local_x, shard.device);
        if (gate.has_value()) {
            local_gate =
                tensor_to_cuda_device(
                    execution.model_parallel_collectives,
                    local_gate, shard.device);
        }
        if (is_mxfp8() &&
                tensor_parallel_axis == TensorParallelAxis::Input) {
            MFQ_RUNTIME_CHECK(
                !gate.has_value(),
                "MXFP8 input-axis tensor parallelism does not support gating");
            local_outputs[index] =
                mxfp8_matmul_f32(
                    execution.profiler, shard.mxfp8, local_x);
        } else {
            local_outputs[index] =
                run_quant_linear_shard(
                    execution.profiler, execution.kl_mmq,
                    shard, local_x,
                    gate.has_value()
                        ? MfqOptional<mfq_tensor_backend::Tensor>(
                            local_gate)
                        : mfq_nullopt,
                    gate_mode);
        }
    }

    const int primary = model_parallel_primary_device(execution);
    MfqCudaGuard primary_guard(primary);
    if (tensor_parallel_axis == TensorParallelAxis::Output) {
        std::vector<mfq_tensor_backend::Tensor> gathered;
        gathered.reserve(local_outputs.size());
        for (auto & output : local_outputs) {
            gathered.push_back(
                tensor_to_cuda_device(
                    execution.model_parallel_collectives,
                    output, primary));
        }
        return mfq_tensor_backend::cat(gathered, -1).contiguous();
    }

    auto reduced = reduce_model_parallel_outputs(
        execution, std::move(local_outputs));
    return is_mxfp8()
        ? reduced.to(x.scalar_type()).contiguous()
        : reduced;
}

mfq_tensor_backend::Tensor dense_projection(
        mfq_tensor_backend::Tensor input, const mfq_tensor_backend::Tensor& weight) {
    input = input.to(weight.scalar_type());
    const int64_t rows = input.numel() / input.size(-1);
    if (rows > 1 && rows <= 6) {
        auto shape = input.sizes().vec();
        shape.back() = weight.size(0);
        auto flat = input.reshape({rows, input.size(-1)});
        std::vector<mfq_tensor_backend::Tensor> outputs;
        outputs.reserve(rows);
        // A strided-batched GEMM may select different accumulation from M=1.
        // Keep short target/predictor windows on the decode projection path.
        // ponytail: at most six launches; fuse only with decode-equivalent accumulation.
        for (int64_t row = 0; row < rows; ++row)
            outputs.push_back(mfq_tensor_backend::matmul(flat.narrow(0, row, 1), weight.transpose(0, 1)));
        return mfq_tensor_backend::cat(outputs, 0).reshape(shape);
    }
    return mfq_tensor_backend::matmul(input, weight.transpose(0, 1));
}

mfq_tensor_backend::Tensor QuantLinear::forward_dense(mfq_tensor_backend::Tensor x) const {
    return dense_projection(std::move(x), dense);
}

mfq_tensor_backend::Tensor QuantLinear::forward(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    if(is_tq())return execution.profiler.measure("tq.linear",[&] {return tq.forward(std::move(x));});
    if (tensor_parallel() || is_nint() || is_nvq()) {
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
        auto y = tensor_parallel()
            ? forward_tensor_parallel_flat(execution, flat, mfq_nullopt, 0)
            : is_nint() ? run_nint_linear(
                  execution.profiler, execution.kl_mmq, nint, flat)
                        : run_nvq_linear(
                  execution.profiler, execution.kl_mmq, nvq, flat);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (is_mxfp4()) return mxfp4.forward(x);
    if (is_mxfp4_sq()) return mxfp4_sq.forward(x);
    if (is_fp8_sq()) return fp8_sq.forward(x);
    if (is_dense()) return forward_dense(x);
    return mxfp8.forward(execution.profiler, x);
}

mfq_tensor_backend::Tensor QuantLinear::forward_bf16_output(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    return forward(execution, x)
        .to(mfq_tensor_backend::kBFloat16).contiguous();
}

mfq_tensor_backend::Tensor QuantLinear::forward_mxfp8_groupwise(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor grouped,
        int64_t groups) const {
    MFQ_RUNTIME_CHECK(
        is_mxfp8(),
        "groupwise MXFP8 projection requires an MXFP8 tensor");
    if (!tensor_parallel()) {
        return mxfp8_groupwise_matmul(
            execution.profiler, mxfp8.weight, grouped, groups);
    }
    MFQ_RUNTIME_CHECK(
        tensor_parallel_axis == TensorParallelAxis::Input,
        "groupwise MXFP8 tensor parallelism requires input-axis shards");
    std::vector<mfq_tensor_backend::Tensor> partials;
    partials.reserve(tensor_parallel_shards.size());
    for (const auto & shard : tensor_parallel_shards) {
        MFQ_RUNTIME_CHECK(
            shard.kind == QuantLinearKind::Mxfp8,
            "groupwise MXFP8 tensor-parallel shard kind mismatch");
        MfqCudaGuard guard(shard.device);
        auto local = grouped.narrow(
            -1, shard.input_begin,
            shard.input_end - shard.input_begin);
        local = tensor_to_cuda_device(
            execution.model_parallel_collectives,
            local, shard.device);
        partials.push_back(mxfp8_groupwise_matmul_f32(
            execution.profiler, shard.mxfp8, local, groups));
    }
    return reduce_model_parallel_outputs(execution, std::move(partials))
        .to(grouped.scalar_type()).contiguous();
}

mfq_tensor_backend::Tensor QuantLinear::forward_input_mul(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        int mode) const {
    if (tensor_parallel() || is_nint() || is_nvq()) {
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
        auto flat_gate = gate.reshape({-1, gate.size(-1)});
        auto y = tensor_parallel()
            ? forward_tensor_parallel_flat(execution, flat, flat_gate, mode)
            : is_nint() ? run_nint_linear(
                  execution.profiler, execution.kl_mmq,
                  nint, flat, flat_gate, mode)
                        : run_nvq_linear(
                  execution.profiler, execution.kl_mmq,
                  nvq, flat, flat_gate, mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (is_dense()) {
        MFQ_RUNTIME_CHECK(mode == 1 || mode == 2,
            "dense input gate mode must be sigmoid or SiLU");
        // Match the existing dense shard path, including dtype rounding.
        auto local = x.to(dense.scalar_type());
        auto local_gate = gate.to(dense.scalar_type());
        auto gated = mode == 1
            ? local * mfq_tensor_backend::sigmoid(local_gate)
            : local * mfq_tensor_backend::silu(local_gate);
        return forward_dense(gated);
    }
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2,
        "input gate mode must be sigmoid or SiLU");
    auto local_gate = gate.to(x.scalar_type());
    auto gated = mode == 1
        ? x * mfq_tensor_backend::sigmoid(local_gate)
        : x * mfq_tensor_backend::silu(local_gate);
    return forward(execution, gated);
}

mfq_tensor_backend::Tensor QuantLinear::forward_input_mul_f32_kld(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        int mode) const {
    MFQ_RUNTIME_CHECK(
        !tensor_parallel() && is_nint(),
        "FP32-output KLD down projection requires a local NINT tensor");
    MFQ_RUNTIME_CHECK(
        execution.kl_mmq.mode == KlMmqMode::Fp16,
        "FP32-output NINT MMQ is restricted to the FP16 KLD path");
    auto shape = x.sizes().vec();
    auto y = nint_matmul_input_mul_f32(
        execution.profiler, nint, x.reshape({-1, x.size(-1)}),
        gate.reshape({-1, gate.size(-1)}), mode);
    if (nint.q8_zero) ++execution.kl_mmq.dense_calls;
    shape.back() = y.size(-1);
    return y.reshape(shape);
}

int64_t QuantLinear::out() const {
    if(is_tq())return tq.out;
    if (tensor_parallel()) return logical_out;
    if (is_nint()) return nint.out;
    if (is_nvq()) return nvq.out;
    if (is_mxfp4()) return mxfp4.weight.out;
    if (is_mxfp4_sq()) return mxfp4_sq.weight.out;
    if (is_fp8_sq()) return fp8_sq.weight.out;
    if (is_dense()) return dense.size(0);
    return mxfp8.weight.out;
}

int64_t QuantLinear::neuron_len() const {
    if(is_tq())return tq.neuron_len;
    if (tensor_parallel()) return logical_neuron_len;
    if (is_nint()) return nint.neuron_len;
    if (is_nvq()) return nvq.neuron_len;
    if (is_mxfp4()) return mxfp4.weight.neuron_len;
    if (is_mxfp4_sq()) return mxfp4_sq.weight.neuron_len;
    if (is_fp8_sq()) return fp8_sq.weight.neuron_len;
    if (is_dense()) return dense.size(1);
    return mxfp8.weight.neuron_len;
}

QuantLinearProjectionRefs QuantLinearGroup::tensor_parallel_output_projections() const {
    QuantLinearProjectionRefs projections;
    projections.reserve(layers.size());
    for (const auto & layer : layers) {
        projections.push_back(&layer);
    }
    return projections;
}

bool QuantLinearGroup::tensor_parallel_output_compatible() const {
    return tensor_parallel_output_projections_compatible(
        tensor_parallel_output_projections());
}

std::vector<mfq_tensor_backend::Tensor>
QuantLinearGroup::forward_tensor_parallel_output_group(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    return forward_tensor_parallel_output_projections(
        execution, x, tensor_parallel_output_projections());
}

std::vector<mfq_tensor_backend::Tensor> QuantLinearGroup::forward(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    if(tq.defined())return execution.profiler.measure("tq.parallel",[&] {return tq.forward(std::move(x));});
    const bool default_mmq =
        execution.kl_mmq.mode == KlMmqMode::Default;
    if (!x.is_cuda()) {
        MFQ_RUNTIME_CHECK(
            !nint_grouped,
            "CPU dense offload requires separate compact linear weights");
        std::vector<mfq_tensor_backend::Tensor> result;
        result.reserve(layers.size());
        for (const auto & layer : layers) {
            result.push_back(layer.forward(execution, x));
        }
        return result;
    }
    if (execution.config.tensor_parallel_grouped_projections &&
            tensor_parallel_output_compatible()) {
        return forward_tensor_parallel_output_group(execution, x);
    }
    if (nint_grouped) {
        return nint.forward(
            execution.profiler, execution.kl_mmq,
            execution.config, execution.decode_graph_serial_branches, x);
    }
    if (default_mmq && nvq_prefix2 && nvq_fusion_enabled(execution.config)) {
        auto shape = x.sizes().vec();
        const auto flat =
            x.reshape({-1, x.size(-1)});
        std::vector<mfq_tensor_backend::Tensor> branch_outputs;
        const bool parallel =
            decode_branch_parallel &&
            decode_branch_parallel_enabled(
                execution.config, execution.decode_graph_serial_branches, flat.size(0)) &&
            layers.size() > 2 &&
            branch_executor->run(
                layers.size() - 1,
                [&](size_t branch) {
                    if (branch == 0) {
                        return nvq_matmul_multi2(
                            execution.profiler, layers[0].nvq,
                            layers[1].nvq, flat);
                    }
                    return layers[branch + 1].forward(execution, x);
                },
                branch_outputs);
        auto combined = parallel
            ? branch_outputs[0]
            : nvq_matmul_multi2(
                execution.profiler, layers[0].nvq,
                layers[1].nvq, flat);
        auto pair = combined.split_with_sizes({outs[0], outs[1]}, -1);
        std::vector<mfq_tensor_backend::Tensor> result;
        result.reserve(layers.size());
        for (size_t i = 0; i < 2; ++i) {
            auto part_shape = shape;
            part_shape.back() = outs[i];
            result.push_back(pair[i].reshape(part_shape));
        }
        for (size_t i = 2; i < layers.size(); ++i) {
            result.push_back(
                parallel
                    ? branch_outputs[i - 1]
                    : layers[i].forward(execution, x));
        }
        return result;
    }
    std::vector<mfq_tensor_backend::Tensor> result;
    if (default_mmq && decode_branch_parallel &&
            decode_branch_parallel_enabled(
                execution.config, execution.decode_graph_serial_branches, x.numel() / x.size(-1)) &&
            branch_executor->run(
                layers.size(),
                [&](size_t index) {
                    return layers[index].forward(execution, x);
                },
                result)) {
        return result;
    }
    result.reserve(layers.size());
    for (const auto & layer : layers) {
        result.push_back(layer.forward(execution, x));
    }
    return result;
}

mfq_tensor_backend::Tensor QuantLinearGroup::forward_swiglu(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    if(tq.defined() && outs.size()==2 && outs[0]==outs[1])
        return execution.profiler.measure("tq.swiglu",[&] {return tq.swiglu(std::move(x));});
    const bool default_mmq =
        execution.kl_mmq.mode == KlMmqMode::Default;
    if (default_mmq && nint_grouped && nint.split_w.empty() &&
            x.numel() / x.size(-1) >= 1 && x.numel() / x.size(-1) <= 6) {
        return nint.forward_swiglu(execution.profiler, x);
    }
    if (default_mmq && nvq_prefix2 && layers.size() == 2 &&
            nvq_fusion_enabled(execution.config)) {
        auto shape = x.sizes().vec();
        auto y = nvq_matmul_swiglu(
            execution.profiler, layers[0].nvq, layers[1].nvq,
            x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (outs.size() != 2 || outs[0] != outs[1]) {
        throw std::runtime_error("SwiGLU requires equal gate/up output widths");
    }
    auto parts = forward(execution, x);
    return mfq_tensor_backend::silu(parts[0]) * parts[1];
}

mfq_tensor_backend::Tensor QuantLinearGroup::forward_geglu(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    if (execution.kl_mmq.mode == KlMmqMode::Default &&
            nint_grouped && nint.split_w.empty() &&
            x.numel() / x.size(-1) == 1) {
        return nint.forward_geglu(execution.profiler, x);
    }
    if (outs.size() != 2 || outs[0] != outs[1]) {
        throw std::runtime_error("GeGLU requires equal gate/up output widths");
    }
    auto parts = forward(execution, x);
    return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
}

std::vector<mfq_tensor_backend::Tensor> DenseLinearGroup::forward(mfq_tensor_backend::Tensor x) const {
    auto shape = x.sizes().vec();
    auto y = dense_projection(x.reshape({-1, x.size(-1)}), w);
    auto parts = y.split_with_sizes(outs, -1);
    for (auto & p : parts) {
        auto s = shape;
        s.back() = p.size(-1);
        p = p.reshape(s);
    }
    return parts;
}

std::vector<mfq_tensor_backend::Tensor> NintLinearGroup::forward(
            CudaProfiler& profiler,
            KlMmqState& kl_mmq,
            const CudaExecutionConfig& config,
            bool serial_branches,
            mfq_tensor_backend::Tensor x) const {
    auto shape = x.sizes().vec();
    std::vector<mfq_tensor_backend::Tensor> parts;
    auto xf = x.reshape({-1, x.size(-1)});
    if (w.q8_zero && xf.size(0) > 64 &&
            projection_w.size() == outs.size()) {
        parts.reserve(projection_w.size());
        for (const auto & projection : projection_w) {
            parts.push_back(run_nint_linear(
                profiler, kl_mmq, projection, xf));
        }
    } else if (!split_w.empty()) {
        parts.reserve(outs.size());
        std::vector<int> shared_groups;
        const auto quant_group_setting=mfq::cuda::runtime_options::nint_group_quantize();
        if((!quant_group_setting || *quant_group_setting) && serial_branches &&
                kl_mmq.mode==KlMmqMode::Default && xf.is_contiguous() &&
                xf.size(0)>=1 && xf.size(0)<=6 &&
                (xf.scalar_type()==mfq_tensor_backend::kFloat16 || xf.scalar_type()==mfq_tensor_backend::kFloat32)) {
            bool compatible=true;
            for(const auto& weight:split_w) {
                compatible=compatible && !weight.q8_zero && weight.neuron_len==xf.size(1) &&
                    (weight.gs==24 || weight.gs==28 || weight.gs==48);
                if(std::find(shared_groups.begin(),shared_groups.end(),weight.gs)==shared_groups.end())
                    shared_groups.push_back(weight.gs);
            }
            if(!compatible || shared_groups.size()<2)shared_groups.clear();
        }
        NintInputQuantizationScope prepared(xf,shared_groups);
        std::vector<mfq_tensor_backend::Tensor> grouped_outputs;
        const bool parallel =
            decode_branch_parallel_enabled(
                config, serial_branches, xf.size(0)) &&
            branch_executor->run(
                split_w.size(),
                [&](size_t index) {
                    return run_nint_linear(
                        profiler, kl_mmq, split_w[index], xf);
                },
                grouped_outputs);
        for (size_t i = 0; i < split_w.size(); ++i) {
            auto y = parallel
                ? grouped_outputs[i]
                : run_nint_linear(
                      profiler, kl_mmq, split_w[i], xf);
            auto ys = y.split_with_sizes(split_outs[i], -1);
            for (auto & p : ys) parts.push_back(p);
        }
    } else {
        auto y = run_nint_linear(
            profiler, kl_mmq,
            w, x.reshape({-1, x.size(-1)}));
        parts = y.split_with_sizes(outs, -1);
    }
    for (auto & p : parts) {
        auto s = shape;
        s.back() = p.size(-1);
        p = p.reshape(s);
    }
    return parts;
}

mfq_tensor_backend::Tensor NintLinearGroup::forward_swiglu(
            CudaProfiler& profiler,
            mfq_tensor_backend::Tensor x) const {
    if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
        throw std::runtime_error("NINT SwiGLU fusion requires one packed [gate, up] group");
    }
    auto shape = x.sizes().vec();
    auto y = nint_matmul_swiglu(
        profiler, w, x.reshape({-1, x.size(-1)}));
    shape.back() = y.size(-1);
    return y.reshape(shape);
}

mfq_tensor_backend::Tensor NintLinearGroup::forward_geglu(
            CudaProfiler& profiler,
            mfq_tensor_backend::Tensor x) const {
    if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
        throw std::runtime_error("NINT GeGLU fusion requires one packed [gate, up] group");
    }
    auto shape = x.sizes().vec();
    auto y = nint_matmul_geglu(
        profiler, w, x.reshape({-1, x.size(-1)}));
    shape.back() = y.size(-1);
    return y.reshape(shape);
}
