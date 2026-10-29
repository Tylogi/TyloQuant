#include "runtime/decode_window.h"
#include "hip/hip_runtime.h"
#include "mfq_hip_activation_ops.h"
#include "mfq_hip_quant_ops.h"
#include "nint.h"
#include "hip_execution.h"
#include "format.h"
#include "packed_nint.h"
#include "cpu_projection_rows.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define MFQ_CPU_X86_GNU 1
#endif

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

mfq_tensor_backend::Tensor nint_row_embedding_lookup(
        const mfq::NintRows& table, const mfq_tensor_backend::Tensor& ids) {
    MFQ_RUNTIME_CHECK(ids.is_cuda() && ids.scalar_type() == mfq_tensor_backend::kInt64,
        "NINT row IDs must be CUDA int64");
    const MfqCudaGuard guard(ids.device());
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    MFQ_CUDA_CHECK(hipStreamIsCapturing(mfq_current_cuda_stream(), &capture));
    MFQ_RUNTIME_CHECK(capture == hipStreamCaptureStatusNone,
        "range-backed NINT row lookup cannot run inside CUDA graph capture");
    auto shape = ids.sizes().vec();
    shape.push_back(table.width());
    if (ids.numel() == 0)
        return mfq_tensor_backend::empty(shape, ids.options().dtype(mfq_tensor_backend::kFloat16));
    MFQ_RUNTIME_CHECK(ids.numel() <= std::numeric_limits<int>::max() / 6 &&
        static_cast<uint64_t>(ids.numel()) * table.width() <= std::numeric_limits<uint32_t>::max(),
        "NINT row lookup batch exceeds bounds");
    auto host = ids.contiguous().cpu();
    mfq::NintRowBatch selected;
    table.append_rows(host.data_ptr<int64_t>(), static_cast<std::size_t>(host.numel()),
        selected, mfq_get_num_threads());
    selected.validate();
    auto packed = cpu_u8_tensor(selected.packed(), {static_cast<int64_t>(selected.packed_nbytes())}).to(ids.device());
    std::vector<int32_t> words(selected.descriptors().size());
    std::memcpy(words.data(), selected.descriptors().data(), words.size() * sizeof(uint32_t));
    auto descriptors = cpu_i32_tensor(words, {static_cast<int64_t>(selected.rows()), 6}).to(ids.device());
    return nint_selected_rows_cuda(packed, descriptors, table.width()).reshape(shape);
}

mfq_tensor_backend::Tensor pad_last(mfq_tensor_backend::Tensor x, int64_t target) {
    if (x.size(1) == target) return x;
    if (x.size(1) > target) throw std::runtime_error("activation width exceeds neuron_len");
    return mfq_tensor_backend::constant_pad_nd(x, {0, target - x.size(1)}, 0);
}
uint32_t cpu_load_packed_bits(
        const uint8_t * data,
        int64_t nbytes,
        int64_t bit,
        int bits) {
    const int64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = data[byte];
    if (byte + 1 < nbytes) word |= static_cast<uint32_t>(data[byte + 1]) << 8;
    if (byte + 2 < nbytes) word |= static_cast<uint32_t>(data[byte + 2]) << 16;
    return (word >> shift) & ((1u << bits) - 1u);
}

#ifdef MFQ_CPU_X86_GNU
__attribute__((target("avx512f,avx512bw,avx512vnni")))
int32_t cpu_dot_s8_s8_vnni(
        const int8_t * left,
        const int8_t * right,
        int32_t right_sum) {
    const __m512i signed_a = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(left));
    const __m512i a = _mm512_xor_si512(
        signed_a, _mm512_set1_epi8(static_cast<char>(-128)));
    const __m512i b = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(right));
    const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a, b);
    return _mm512_reduce_add_epi32(dot) - 128 * right_sum;
}

__attribute__((target("avx512f,avx512bw,avx512vnni")))
int32_t cpu_dot_u8_s8_vnni(
        const uint8_t * left,
        const int8_t * right) {
    const __m512i a = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(left));
    const __m512i b = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(right));
    const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a, b);
    return _mm512_reduce_add_epi32(dot);
}

bool cpu_has_avx512_vnni() {
    static const bool supported = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vnni");
    }();
    return supported;
}
#endif

int32_t cpu_dot_s8_s8_64(
        const int8_t * left,
        const int8_t * right,
        int32_t right_sum) {
#ifdef MFQ_CPU_X86_GNU
    if (cpu_has_avx512_vnni()) {
        return cpu_dot_s8_s8_vnni(left, right, right_sum);
    }
#endif
    int32_t result = 0;
    for (int index = 0; index < 64; ++index) {
        result += static_cast<int32_t>(left[index]) *
            static_cast<int32_t>(right[index]);
    }
    return result;
}

int32_t cpu_dot_u8_s8_64(
        const uint8_t * left,
        const int8_t * right) {
#ifdef MFQ_CPU_X86_GNU
    if (cpu_has_avx512_vnni()) return cpu_dot_u8_s8_vnni(left, right);
#endif
    int32_t result = 0;
    for (int index = 0; index < 64; ++index) {
        result += static_cast<int32_t>(left[index]) *
            static_cast<int32_t>(right[index]);
    }
    return result;
}

void cpu_unpack_nint_group(
        const uint8_t * packed,
        int bits,
        int valid,
        uint8_t * values) {
    std::fill(values, values + 64, static_cast<uint8_t>(0));
    if (bits == 8) {
        std::memcpy(values, packed, static_cast<size_t>(valid));
        return;
    }
    const uint32_t mask = (1u << bits) - 1u;
    uint32_t reservoir = 0;
    int available = 0;
    int source = 0;
    for (int index = 0; index < valid; ++index) {
        while (available < bits) {
            reservoir |= static_cast<uint32_t>(packed[source++]) << available;
            available += 8;
        }
        values[index] = static_cast<uint8_t>(reservoir & mask);
        reservoir >>= bits;
        available -= bits;
    }
}

void cpu_unpack_nint_group_at_bit_offset(
        const uint8_t * packed,
        uint64_t bit_offset,
        int bits,
        int valid,
        uint8_t * values) {
    std::fill(values, values + 64, static_cast<uint8_t>(0));
    const uint32_t mask = bits == 8 ? 255u : ((1u << bits) - 1u);
    for (int index = 0; index < valid; ++index) {
        const uint64_t value_bit =
            bit_offset + static_cast<uint64_t>(index) * static_cast<uint64_t>(bits);
        const size_t byte = static_cast<size_t>(value_bit >> 3);
        const int shift = static_cast<int>(value_bit & 7u);
        const uint32_t pair = static_cast<uint32_t>(packed[byte]) |
            (static_cast<uint32_t>(packed[byte + 1]) << 8);
        values[index] = static_cast<uint8_t>((pair >> shift) & mask);
    }
}

CpuQuantizedActivation cpu_quantize_activation(
        mfq_tensor_backend::Tensor x,
        int64_t width,
        int64_t group_size,
        bool with_sums) {
    MFQ_RUNTIME_CHECK(!x.is_cuda(), "CPU activation quantization requires a CPU tensor");
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), width).contiguous();
    const int64_t rows = x.size(0);
    const int64_t groups = (width + group_size - 1) / group_size;
    MFQ_RUNTIME_CHECK(group_size > 0 && group_size <= 64,
        "CPU compact GEMV group size must be in [1, 64]");
    CpuQuantizedActivation result;
    result.rows = rows;
    result.groups = groups;
    result.group_size = group_size;
    result.values.resize(static_cast<size_t>(rows * groups * 64));
    result.scales.resize(static_cast<size_t>(rows * groups));
    if (with_sums) {
        result.sums.resize(static_cast<size_t>(rows * groups));
    }
    const mfq_half * input = x.data_ptr<mfq_half>();
    mfq_parallel_for(0, rows * groups, 1, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t row = linear / groups;
            const int64_t group = linear - row * groups;
            const int64_t k0 = group * group_size;
            const int64_t valid = std::min(group_size, width - k0);
            float maximum = 0.0f;
            for (int64_t index = 0; index < valid; ++index) {
                maximum = std::max(
                    maximum,
                    std::fabs(static_cast<float>(input[row * width + k0 + index])));
            }
            const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
            result.scales[static_cast<size_t>(linear)] = scale;
            int32_t sum = 0;
            int8_t * quantized = result.values.data() + linear * 64;
            for (int64_t index = 0; index < valid; ++index) {
                int value = static_cast<int>(std::round(
                    static_cast<float>(input[row * width + k0 + index]) / scale));
                value = std::max(-127, std::min(127, value));
                quantized[index] = static_cast<int8_t>(value);
                sum += value;
            }
            std::fill(
                quantized + valid,
                quantized + 64,
                static_cast<int8_t>(0));
            if (with_sums) {
                result.sums[static_cast<size_t>(linear)] = sum;
            }
        }
    });
    return result;
}

float cpu_half_from_bytes(const int8_t * bytes, int64_t offset) {
    uint16_t raw = 0;
    std::memcpy(&raw, bytes + offset, sizeof(raw));
    mfq_half value;
    std::memcpy(&value, &raw, sizeof(raw));
    return static_cast<float>(value);
}
CpuProjectionRows make_cpu_projection_rows(const NintWeight& w) {
    MFQ_RUNTIME_CHECK(!w.q8_zero && w.q_packed.is_cpu(),"original-FP32 rows require CPU NINTv2 storage");
    CpuProjectionRows p;
    p.width=static_cast<int>(w.neuron_len); p.outputs=static_cast<int>(w.out);
    p.nint={w.q_packed.data_ptr<uint8_t>(),static_cast<size_t>(w.q_packed.numel()),
        w.row_q_bits.data_ptr<uint8_t>(),w.row_q_bit_offsets.data_ptr<int64_t>(),
        w.sub_scale.data_ptr<uint8_t>(),w.sub_min.data_ptr<uint8_t>(),
        w.neuron_scale.data_ptr<float>(),w.neuron_min.data_ptr<float>(),w.out,w.neuron_len,w.gs};
    p.nint_rows=mfq::cpu::packed_nint_rows_kernel();
    return p;
}
mfq_tensor_backend::Tensor nint_sharded_embedding_lookup(
    const std::vector<std::shared_ptr<mfq::NintRows>>& tables,const std::vector<int64_t>& ids,
    const std::vector<int64_t>& shape,const mfq_tensor_backend::Device& device) {
    namespace tb=mfq_tensor_backend;
    if(tables.empty() || !device.is_cuda())throw std::invalid_argument("sharded NINT embeddings require CUDA tables");
    const MfqCudaGuard guard(device);const int64_t rows=tables.front()->rows();const int width=tables.front()->width();
    int64_t count=1;for(auto n:shape){if(n<0 || (n && count>std::numeric_limits<int64_t>::max()/n))throw std::overflow_error("embedding shape overflow");count*=n;}
    if(count!=static_cast<int64_t>(ids.size()))throw std::invalid_argument("sharded embedding ID shape mismatch");
    std::vector<std::vector<int64_t>> selected(tables.size()),positions(tables.size());
    for(std::size_t i=0;i<ids.size();++i) {
        const auto id=ids[i];
        if(id<0 || id/rows>=static_cast<int64_t>(tables.size()))throw std::out_of_range("sharded embedding row ID");
        selected[id/rows].push_back(id%rows);positions[id/rows].push_back(static_cast<int64_t>(i));
    }
    std::vector<mfq::NintRowBatch> pieces(tables.size());
    mfq::host_parallel_for(0,static_cast<int64_t>(tables.size()),1,mfq_get_num_threads(),[&](int64_t first,int64_t last) {
        for(auto s=first;s<last;++s) {
            if(tables[s]->rows()!=rows || tables[s]->width()!=width)throw std::invalid_argument("NINT shard geometry mismatch");
            if(!selected[s].empty())tables[s]->append_rows(selected[s].data(),selected[s].size(),pieces[s],1);
        }
    });
    mfq::NintRowBatch batch;std::vector<int64_t> inverse(ids.size());int64_t at=0;
    for(std::size_t s=0;s<tables.size();++s) {
        batch.append_batch(pieces[s]);for(auto p:positions[s])inverse[p]=at++;
    }
    auto out_shape=shape;out_shape.push_back(width);
    if(ids.empty())return tb::empty(out_shape,tb::TensorOptions().device(device).dtype(tb::kFloat16));
    batch.validate();
    auto packed=cpu_u8_tensor(batch.packed(),{static_cast<int64_t>(batch.packed_nbytes())}).to(device);
    std::vector<int32_t> words(batch.descriptors().size());std::memcpy(words.data(),batch.descriptors().data(),words.size()*sizeof(uint32_t));
    auto descriptors=cpu_i32_tensor(words,{static_cast<int64_t>(batch.rows()),6}).to(device);
    return nint_selected_rows_cuda(packed,descriptors,width).index_select(0,tb::tensor(inverse).to(device)).reshape(out_shape);
}
static mfq_tensor_backend::Tensor nint_matmul_cpu(
        const NintWeight & w,
        mfq_tensor_backend::Tensor x) {
    MFQ_RUNTIME_CHECK(!x.is_cuda(), "CPU NINT GEMV requires CPU activations");
    MFQ_RUNTIME_CHECK(!w.q_packed.is_cuda(), "CPU NINT GEMV requires CPU-resident weights");
    if (!w.q8_zero) {
        MFQ_RUNTIME_CHECK(!w.row_q_bits.is_cuda() && !w.row_q_bit_offsets.is_cuda() &&
            !w.sub_scale.is_cuda() && !w.sub_min.is_cuda() &&
            !w.neuron_scale.is_cuda() && !w.neuron_min.is_cuda(),
            "CPU NINT GEMV requires CPU-resident metadata");
        x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat32), w.neuron_len).contiguous();
        auto result = mfq_tensor_backend::empty({x.size(0), w.out},
            x.options().dtype(mfq_tensor_backend::kFloat32));
        mfq::cpu::PackedNintView view;
        view.packed = w.q_packed.data_ptr<uint8_t>();
        view.packed_bytes = static_cast<size_t>(w.q_packed.numel());
        view.row_bits = w.row_q_bits.data_ptr<uint8_t>();
        view.row_bit_offsets = w.row_q_bit_offsets.data_ptr<int64_t>();
        view.group_scale = w.sub_scale.data_ptr<uint8_t>();
        view.group_min = w.sub_min.data_ptr<uint8_t>();
        view.row_scale = w.neuron_scale.data_ptr<float>();
        view.row_min = w.neuron_min.data_ptr<float>();
        view.outputs = w.out;
        view.width = w.neuron_len;
        view.group_size = w.gs;
        mfq::cpu::packed_nint_matmul(view, x.data_ptr<float>(), x.size(0),
            w.neuron_len, result.data_ptr<float>(), w.out, mfq_get_num_threads());
        return result.to(mfq_tensor_backend::kFloat16);
    }
    auto activation = cpu_quantize_activation(
        std::move(x), w.neuron_len, w.gs, true);
    const int64_t rows = activation.rows;
    const int64_t width = w.neuron_len;
    const int64_t outputs = w.out;
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    mfq_half * output = result.data_ptr<mfq_half>();
    const uint8_t * quant = w.q_packed.data_ptr<uint8_t>();
    const uint8_t * row_q_bits = w.q8_zero
        ? nullptr : w.row_q_bits.data_ptr<uint8_t>();
    const int64_t * row_q_bit_offsets = w.q8_zero
        ? nullptr : w.row_q_bit_offsets.data_ptr<int64_t>();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            const int neuron_bits = w.q8_zero
                ? 8 : static_cast<int>(row_q_bits[neuron]);
            for (int64_t group = 0; group < w.ng; ++group) {
                const int64_t meta = neuron * w.ng + group;
                const uint8_t * packed = w.q8_zero
                    ? quant + meta * 32
                    : quant;
                float scale = 0.0f;
                float minimum = 0.0f;
                if (w.q8_zero) {
                    scale = static_cast<float>(
                        w.q8_zero_scale.data_ptr<mfq_half>()[meta]);
                } else {
                    scale = w.neuron_scale.data_ptr<float>()[neuron] *
                        static_cast<float>(w.sub_scale.data_ptr<uint8_t>()[meta]);
                    minimum = w.neuron_min.data_ptr<float>()[neuron] *
                        static_cast<float>(w.sub_min.data_ptr<uint8_t>()[meta]);
                }
                const int64_t valid = std::min<int64_t>(
                    w.gs, width - group * w.gs);
                alignas(64) uint8_t weights[64];
                if (w.q8_zero) {
                    std::fill(weights, weights + 64, static_cast<uint8_t>(0));
                    std::memcpy(weights, packed, static_cast<size_t>(valid));
                } else {
                    const uint64_t bit_offset =
                        static_cast<uint64_t>(row_q_bit_offsets[neuron]) +
                        static_cast<uint64_t>(group * w.gs) *
                            static_cast<uint64_t>(neuron_bits);
                    cpu_unpack_nint_group_at_bit_offset(
                        packed, bit_offset, neuron_bits,
                        static_cast<int>(valid), weights);
                }
                for (int64_t row = 0; row < rows; ++row) {
                    const int64_t activation_group =
                        row * activation.groups + group;
                    const int8_t * quantized = activation.values.data() +
                        activation_group * 64;
                    int32_t dot = 0;
                    if (w.q8_zero) {
                        dot = cpu_dot_s8_s8_64(
                            reinterpret_cast<const int8_t *>(weights), quantized,
                            activation.sums[
                                static_cast<size_t>(activation_group)]);
                        accumulators[static_cast<size_t>(row)] = std::fma(
                            scale * activation.scales[
                                static_cast<size_t>(activation_group)],
                            static_cast<float>(dot),
                            accumulators[static_cast<size_t>(row)]);
                    } else {
                        dot = cpu_dot_u8_s8_64(weights, quantized);
                        const float activation_scale = activation.scales[
                            static_cast<size_t>(activation_group)];
                        const float dot_term = scale * activation_scale *
                            static_cast<float>(dot);
                        const float min_term = minimum * activation_scale *
                            static_cast<float>(activation.sums[
                                static_cast<size_t>(activation_group)]);
                        accumulators[static_cast<size_t>(row)] +=
                            dot_term - min_term;
                    }
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] =
                    mfq_half(accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}
mfq_tensor_backend::Tensor nint_matmul(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x) {
    if (!x.is_cuda()) return nint_matmul_cpu(w, std::move(x));
    x = x.contiguous();
    if (x.size(0) > 8 || (x.scalar_type() != mfq_tensor_backend::kFloat16 &&
            x.scalar_type() != mfq_tensor_backend::kFloat32))
        x = x.to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    if (!w.q8_zero) {
        const char* online=std::getenv("MFQ_HIP_NINT_ONLINE");
        const bool online_size=w.out>=65536 ||
            (w.out>=512 && uint64_t(w.out)*uint64_t(w.neuron_len)>=4ull*1024*1024);
        if(M>8 && M<=32 && online_size && w.gs>=4 && w.gs<=64 && mfq::cuda::DecodeWindow::recording() &&
                (!online || online[0]!='0'))
            return profiler.measure("nint.matmul.online_f16",[&] {
                return nint_matmul_online_f16_cuda(w.q_packed,w.row_q_bits,w.row_q_bit_offsets,
                    w.sub_scale,w.sub_min,w.neuron_scale,w.neuron_min,x,w.gs);
            });
        if (M <= 8) {
            Workspace & ws = w.workspace(M);
            if (w.aligned_q8) {
                return profiler.measure("nint.matmul", [&]() {
                    return nint_matmul_q8_ws_cuda(
                        w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                        w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                        x, w.gs, ws.qx, ws.xscale);
                });
            }
            if(M>1 && ((w.uniform_qbits==5 && w.gs==28) ||
                    (w.uniform_qbits==6 && (w.gs==24 || w.gs==28))))
                return profiler.measure("nint.matmul.fixed_bits",[&] {
                    return nint_matmul_fixed_bits_ws_cuda(w.q_packed,w.row_q_bits,w.row_q_bit_offsets,
                        w.sub_scale,w.sub_min,w.neuron_scale,w.neuron_min,x,w.gs,ws.qx,ws.xscale,w.uniform_qbits);
                });
            if (M == 1 && w.fused_ok) {
                return profiler.measure("nint.matmul.fused", [&]() {
                    return nint_matmul_fused_decode_cuda(
                        w.fused_q, w.fused_row_ids, w.neuron_scale,
                        w.neuron_min, x, w.gs, w.fused_segment_bits,
                        w.fused_segment_row_offsets,
                        w.fused_segment_byte_offsets,
                        w.fused_segment_slots, ws.qx, ws.xscale, ws.xsum);
                });
            }
            return profiler.measure("nint.matmul", [&]() {
                return nint_matmul_ws_cuda(
                    w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                    w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                    x, w.gs, ws.qx, ws.xscale);
            });
        }
        auto dense = profiler.measure("nint.dequant", [&]() {
            return nint_decode_cuda(
                w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                w.neuron_len, w.gs);
        });
        return profiler.measure("nint.gemm", [&]() {
            return mfq_tensor_backend::matmul(x, dense.transpose(0, 1));
        });
    }
    if (M <= 8) {
        Workspace & ws = w.workspace(M);
        return profiler.measure("nint8_zero.gemv", [&]() {
            return nint8_zero_gemv_ws_cuda(
                w.q_packed, w.q8_zero_scale, x, ws.qx, ws.xscale);
        });
    }
    return profiler.measure("nint8_zero.packed_mmq", [&]() {
        return nint8_zero_mmq_f16_packed_cuda(
            w.q_packed, w.q8_zero_scale, x, w.neuron_len);
    });
}

static mfq_tensor_backend::Tensor nint_matmul_f32(
        CudaProfiler& profiler,
        const NintWeight & w,
        mfq_tensor_backend::Tensor x) {
    x = pad_last(
        x.contiguous().to(mfq_tensor_backend::kFloat16),
        w.neuron_len);
    if (!w.q8_zero) {
        return nint_matmul(
            profiler, w, x).to(mfq_tensor_backend::kFloat32);
    }
    MFQ_RUNTIME_CHECK(
        x.size(0) >= 16,
        "FP32-output NINT MMQ requires at least 16 activation rows");
    return profiler.measure(
        "kld_mmq.nint8_zero.fp32_output", [&]() {
            return nint8_zero_mmq_f32_packed_cuda(
                w.q_packed, w.q8_zero_scale,
                x, w.neuron_len);
        });
}

mfq_tensor_backend::Tensor nint_matmul_input_mul_f32(
        CudaProfiler& profiler,
        const NintWeight & w,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        int mode) {
    MFQ_RUNTIME_CHECK(
        mode == 1 || mode == 2,
        "FP32-output NINT input gate mode must be sigmoid or SiLU");
    x = pad_last(
        x.contiguous().to(mfq_tensor_backend::kFloat16),
        w.neuron_len);
    gate = pad_last(
        gate.contiguous().to(mfq_tensor_backend::kFloat16),
        w.neuron_len);
    MFQ_RUNTIME_CHECK(
        x.sizes() == gate.sizes(),
        "FP32-output NINT x and gate shapes must match");
    auto activation = mode == 1
        ? x * mfq_tensor_backend::sigmoid(gate)
        : x * mfq_tensor_backend::silu(gate);
    return nint_matmul_f32(
        profiler, w, activation.contiguous());
}

mfq_tensor_backend::Tensor nint_matmul_groupwise_u8(
        CudaProfiler& profiler,
        const NintWeight & w,
        mfq_tensor_backend::Tensor x,
        int64_t groups) {
    MFQ_RUNTIME_CHECK(
        w.bits == 8 && w.gs == 48,
        "groupwise NINT projection requires NINT8 gs48");
    MFQ_RUNTIME_CHECK(
        x.dim() == 3 && x.size(1) == groups && x.size(2) == w.neuron_len,
        "groupwise NINT projection expects [B, groups, K]");
    MFQ_RUNTIME_CHECK(
        w.out % groups == 0,
        "groupwise NINT projection output rows must divide groups");
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    auto dense = profiler.measure("nint.groupwise_dequant", [&]() {
        return nint_decode_cuda(
            w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
            w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
            w.neuron_len, w.gs);
    });
    const int64_t rows_per_group = w.out / groups;
    return profiler.measure("nint.groupwise_gemm", [&]() {
        return mfq_tensor_backend::bmm(
            x.transpose(0, 1),
            dense.reshape({groups, rows_per_group, w.neuron_len})
                .transpose(1, 2))
            .transpose(0, 1)
            .contiguous()
            .reshape({x.size(0), w.out});
    });
}

static mfq_tensor_backend::Tensor nint_matmul_input_mul_impl(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        int mode) {
    if (!x.is_cuda()) {
        x = x.contiguous().to(mfq_tensor_backend::kFloat16);
        gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
        MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NINT x and gate shapes must match");
        auto value = mode == 1
            ? x * mfq_tensor_backend::sigmoid(gate)
            : x * mfq_tensor_backend::silu(gate);
        return nint_matmul_cpu(w, value);
    }
    x = x.contiguous();
    gate = gate.contiguous();
    const bool fused_input = !w.q8_zero && x.size(0) <= 8;
    if (!fused_input || (x.scalar_type() != mfq_tensor_backend::kFloat16 &&
            x.scalar_type() != mfq_tensor_backend::kFloat32))
        x = x.to(mfq_tensor_backend::kFloat16);
    if (!fused_input || (gate.scalar_type() != mfq_tensor_backend::kFloat16 &&
            gate.scalar_type() != mfq_tensor_backend::kFloat32))
        gate = gate.to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.sizes() == gate.sizes(),
        "NINT x and gate shapes must match");
    x = pad_last(x, w.neuron_len);
    gate = pad_last(gate, w.neuron_len);
    if (!w.q8_zero && x.size(0) <= 8) {
        Workspace & ws = w.workspace(static_cast<int>(x.size(0)));
        if (x.size(0) == 1 && w.fused_ok) {
            return profiler.measure("nint.matmul.input_mul.fused", [&]() {
                return nint_matmul_fused_decode_cuda(
                    w.fused_q, w.fused_row_ids, w.neuron_scale,
                    w.neuron_min, x, w.gs, w.fused_segment_bits,
                    w.fused_segment_row_offsets, w.fused_segment_byte_offsets,
                    w.fused_segment_slots, ws.qx, ws.xscale, ws.xsum, gate, mode);
            });
        }
        return profiler.measure("nint.matmul.input_mul", [&]() {
            if(x.size(0)>1 && ((w.uniform_qbits==5 && w.gs==28) ||
                    (w.uniform_qbits==6 && (w.gs==24 || w.gs==28))))
                return nint_matmul_input_mul_fixed_bits_ws_cuda(w.q_packed,w.row_q_bits,w.row_q_bit_offsets,
                    w.sub_scale,w.sub_min,w.neuron_scale,w.neuron_min,x,gate,mode,w.gs,ws.qx,ws.xscale,w.uniform_qbits);
            if (w.aligned_q8) {
                return nint_matmul_input_mul_q8_ws_cuda(
                    w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                    w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                    x, gate, mode, w.gs, ws.qx, ws.xscale);
            }
            return nint_matmul_input_mul_ws_cuda(
                w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                x, gate, mode, w.gs, ws.qx, ws.xscale);
        });
    }
    if (mode == 1) {
        return nint_matmul(
            profiler, w, x * mfq_tensor_backend::sigmoid(gate));
    }
    return nint_matmul(
        profiler, w, x * mfq_tensor_backend::silu(gate));
}

mfq_tensor_backend::Tensor nint_matmul_input_mul(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        int mode) {
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2,
        "NINT input gate mode must be sigmoid or SiLU");
    return nint_matmul_input_mul_impl(profiler, w, x, gate, mode);
}

mfq_tensor_backend::Tensor nint_matmul_input_swiglu_f16(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor up,
        mfq_tensor_backend::Tensor gate) {
    MFQ_RUNTIME_CHECK(up.is_cuda() && gate.is_cuda() && up.dim() == 2 &&
        up.size(0) == 1 && up.sizes() == gate.sizes() &&
        up.is_contiguous() && gate.is_contiguous() &&
        up.scalar_type() == mfq_tensor_backend::kFloat16 &&
        gate.scalar_type() == mfq_tensor_backend::kFloat16,
        "NINT rounded SwiGLU requires one contiguous CUDA FP16 row");
    return nint_matmul_input_mul_impl(profiler, w, up, gate, 3);
}

mfq_tensor_backend::Tensor nint_matmul_swiglu(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    auto parts = nint_matmul(profiler, w, x).chunk(2, -1);
    return mfq_tensor_backend::silu(parts[0]) * parts[1];
}

mfq_tensor_backend::Tensor nint_matmul_geglu(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    auto parts = nint_matmul(profiler, w, x).chunk(2, -1);
    return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
}

bool decode_branch_parallel_enabled(
        const CudaExecutionConfig& config,
        bool serial_branches,
        int64_t rows) {
    // Branch output storage belongs to its allocating stream. Keep graph
    // warmup/capture on the graph pool's stream until cross-stream allocation
    // lifetime tracking supports a fully rejoined capture. Eager is unchanged.
    return rows == 1 &&
        !serial_branches && config.decode_branch_parallel;
}
static void refresh_nint_descriptor(NintCpu & t) {
    if (t.out <= 0 || t.neuron_len <= 0 || t.ng <= 0 || t.gs <= 0 ||
            t.row_q_bits.size() != static_cast<size_t>(t.out) ||
            t.row_sub_bits.size() != static_cast<size_t>(t.out)) {
        throw std::runtime_error("cannot describe incomplete NINT row metadata");
    }
    constexpr int q_selector_bits = 3;
    constexpr int k_selector_bits = 2;
    double encoded_bits = static_cast<double>(t.out) *
        (32.0 + q_selector_bits + k_selector_bits);
    std::array<size_t, 64> joint_counts{};
    const double padded_values_per_row =
        static_cast<double>(t.ng) * t.gs;
    for (int row = 0; row < t.out; ++row) {
        const int q_bits = t.row_q_bits[static_cast<size_t>(row)];
        const int k_bits = t.row_sub_bits[static_cast<size_t>(row)];
        if (q_bits < 1 || q_bits > 8 || k_bits < 1 || k_bits > 8) {
            throw std::runtime_error("invalid NINT descriptor row metadata");
        }
        encoded_bits += padded_values_per_row * q_bits +
            2.0 * t.ng * k_bits;
        ++joint_counts[static_cast<size_t>((q_bits - 1) * 8 + (k_bits - 1))];
    }
    t.aggregate_bpw = encoded_bits /
        (static_cast<double>(t.out) * t.neuron_len);
    t.distribution_entropy = 0.0;
    for (const size_t count : joint_counts) {
        if (count == 0) continue;
        const double probability = static_cast<double>(count) / t.out;
        t.distribution_entropy -= probability * std::log2(probability);
    }
}

NintCpu unpack_nint(const std::vector<uint8_t> & blob) {
    constexpr size_t fixed_header_nbytes = 2 + 3 * sizeof(int32_t) + sizeof(uint32_t);
    if (blob.size() < fixed_header_nbytes) {
        throw std::runtime_error("truncated NINT header");
    }
    NintCpu t;
    size_t off = 0;
    const int raw_bits = blob[off++];
    const bool is_nint_v2 = (raw_bits & 0x80) != 0;
    t.bits = raw_bits & 0x7f;
    t.sub_bits = blob[off++];
    t.gs = read_i32_from(blob, off);
    t.axis = read_i32_from(blob, off);
    t.neuron_len = read_i32_from(blob, off);
    uint32_t ndim = read_u32_from(blob, off);
    if (t.bits < 1 || t.bits > 8 || t.sub_bits < 1 || t.sub_bits > 8 ||
            t.gs <= 0 || t.axis < 0 || t.neuron_len <= 0 ||
            ndim == 0 || t.axis >= static_cast<int>(ndim) ||
            ndim > (blob.size() - off) / sizeof(int64_t)) {
        throw std::runtime_error("invalid NINT header");
    }
    const size_t shape_and_counts =
        static_cast<size_t>(ndim) * sizeof(int64_t) + 2 * sizeof(uint32_t);
    if (shape_and_counts > blob.size() - off) {
        throw std::runtime_error("truncated NINT dimensions");
    }
    t.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) t.shape[i] = read_i64_from(blob, off);
    t.out = (int)read_u32_from(blob, off);
    t.ng = (int)read_u32_from(blob, off);
    int64_t flattened_neuron_len = 1;
    for (uint32_t index = 0; index < ndim; ++index) {
        if (t.shape[index] <= 0 ||
                (static_cast<int>(index) != t.axis &&
                 flattened_neuron_len >
                    std::numeric_limits<int64_t>::max() / t.shape[index])) {
            throw std::runtime_error("invalid NINT tensor shape");
        }
        if (static_cast<int>(index) != t.axis) {
            flattened_neuron_len *= t.shape[index];
        }
    }
    const int64_t expected_groups =
        (static_cast<int64_t>(t.neuron_len) + t.gs - 1) / t.gs;
    if (t.out <= 0 || t.ng <= 0 ||
            t.shape[static_cast<size_t>(t.axis)] != t.out ||
            flattened_neuron_len != t.neuron_len ||
            expected_groups != t.ng) {
        throw std::runtime_error("inconsistent NINT tensor dimensions");
    }
    const size_t anchor_nbytes = static_cast<size_t>(t.out) * 2;
    if (anchor_nbytes > (blob.size() - off) / 2) {
        throw std::runtime_error("truncated NINT neuron metadata");
    }
    t.neuron_scale_h.resize(t.out);
    std::memcpy(t.neuron_scale_h.data(), blob.data() + off, (size_t)t.out * 2);
    off += (size_t)t.out * 2;
    t.neuron_min_h.resize(t.out);
    std::memcpy(t.neuron_min_h.data(), blob.data() + off, (size_t)t.out * 2);
    off += (size_t)t.out * 2;
    for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
        if ((t.neuron_scale_h[row] & 0x7c00u) == 0x7c00u ||
                (t.neuron_min_h[row] & 0x7c00u) == 0x7c00u) {
            throw std::runtime_error("NINT neuron metadata must be finite");
        }
    }
    if (static_cast<size_t>(t.out) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(t.ng)) {
        throw std::runtime_error("NINT metadata size overflow");
    }
    size_t sub_count = static_cast<size_t>(t.out) * static_cast<size_t>(t.ng);
    if (sub_count >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(t.gs)) {
        throw std::runtime_error("NINT value size overflow");
    }
    size_t q_count = sub_count * static_cast<size_t>(t.gs);
    t.row_sub_bits.assign(
        static_cast<size_t>(t.out),
        static_cast<uint8_t>(t.sub_bits));
    if (is_nint_v2) {
        constexpr int selector_bits = 2;
        const auto selectors = unpack_bits(
            blob, off, static_cast<size_t>(t.out), selector_bits);
        for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
            const int row_bits = t.sub_bits - 1 + selectors[row];
            if (row_bits < 1 || row_bits > 8) {
                throw std::runtime_error("invalid NINT v2 subgroup width");
            }
            t.row_sub_bits[row] = static_cast<uint8_t>(row_bits);
        }
        t.sub_scale.resize(sub_count);
        t.sub_min.resize(sub_count);
        for (int selector = 0; selector < (1 << selector_bits); ++selector) {
            const int row_bits = t.sub_bits - 1 + selector;
            const size_t selected_rows = static_cast<size_t>(std::count(
                selectors.begin(), selectors.end(), static_cast<uint8_t>(selector)));
            if (selected_rows == 0) continue;
            if (row_bits < 1 || row_bits > 8) {
                throw std::runtime_error("invalid NINT v2 subgroup width");
            }
            const size_t selected_values = selected_rows * static_cast<size_t>(t.ng);
            const auto scales = unpack_bits(blob, off, selected_values, row_bits);
            const auto minima = unpack_bits(blob, off, selected_values, row_bits);
            size_t local_row = 0;
            for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
                if (selectors[row] != static_cast<uint8_t>(selector)) continue;
                const size_t source = local_row * static_cast<size_t>(t.ng);
                const size_t destination = row * static_cast<size_t>(t.ng);
                std::copy_n(
                    scales.begin() + static_cast<ptrdiff_t>(source),
                    t.ng,
                    t.sub_scale.begin() + static_cast<ptrdiff_t>(destination));
                std::copy_n(
                    minima.begin() + static_cast<ptrdiff_t>(source),
                    t.ng,
                    t.sub_min.begin() + static_cast<ptrdiff_t>(destination));
                ++local_row;
            }
        }
    } else {
        t.sub_scale = unpack_bits(blob, off, sub_count, t.sub_bits);
        t.sub_min = unpack_bits(blob, off, sub_count, t.sub_bits);
    }
    if (is_nint_v2) {
        constexpr int selector_bits = 3;
        const auto selectors = unpack_bits(
            blob, off, static_cast<size_t>(t.out), selector_bits);
        t.row_q_bits.resize(static_cast<size_t>(t.out));
        t.row_q_bit_offsets.resize(static_cast<size_t>(t.out));
        const size_t values_per_row =
            static_cast<size_t>(t.ng) * static_cast<size_t>(t.gs);
        for (int selector = 0; selector < (1 << selector_bits); ++selector) {
            const int row_bits = selector + 1;
            const size_t selected_rows = static_cast<size_t>(std::count(
                selectors.begin(), selectors.end(), static_cast<uint8_t>(selector)));
            if (selected_rows == 0) continue;
            if (values_per_row > std::numeric_limits<size_t>::max() / selected_rows) {
                throw std::runtime_error("NINT mixed-q value count overflow");
            }
            const size_t selected_values = selected_rows * values_per_row;
            if (selected_values >
                    (std::numeric_limits<size_t>::max() - 7) /
                        static_cast<size_t>(row_bits)) {
                throw std::runtime_error("NINT mixed-q packed size overflow");
            }
            const size_t stream_nbytes =
                (selected_values * static_cast<size_t>(row_bits) + 7) / 8;
            if (off > blob.size() || stream_nbytes > blob.size() - off) {
                throw std::runtime_error("truncated NINT mixed-q value stream");
            }
            const uint64_t cohort_start_bits =
                static_cast<uint64_t>(t.q_packed.size()) * 8u;
            t.q_packed.insert(
                t.q_packed.end(),
                blob.begin() + static_cast<ptrdiff_t>(off),
                blob.begin() + static_cast<ptrdiff_t>(off + stream_nbytes));
            off += stream_nbytes;
            size_t local_row = 0;
            const uint64_t packed_row_bits =
                static_cast<uint64_t>(values_per_row) *
                static_cast<uint64_t>(row_bits);
            for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
                if (selectors[row] != static_cast<uint8_t>(selector)) continue;
                const uint64_t bit_offset = cohort_start_bits +
                    static_cast<uint64_t>(local_row) * packed_row_bits;
                if (bit_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    throw std::runtime_error("NINT mixed-q row offset overflow");
                }
                t.row_q_bits[row] = static_cast<uint8_t>(row_bits);
                t.row_q_bit_offsets[row] = static_cast<int64_t>(bit_offset);
                ++local_row;
            }
        }
        t.q_packed.insert(t.q_packed.end(), 8, 0);
    } else {
        if (q_count >
                (std::numeric_limits<size_t>::max() - 7) /
                    static_cast<size_t>(t.bits)) {
            throw std::runtime_error("NINT packed value size overflow");
        }
        const size_t compact_q_nbytes =
            (q_count * static_cast<size_t>(t.bits) + 7) / 8;
        if (off > blob.size() || compact_q_nbytes > blob.size() - off) {
            throw std::runtime_error("truncated NINT packed values");
        }
        t.q_packed.assign(
            blob.begin() + static_cast<ptrdiff_t>(off),
            blob.begin() + static_cast<ptrdiff_t>(off + compact_q_nbytes));
        off += compact_q_nbytes;
        t.row_q_bits.assign(static_cast<size_t>(t.out),
                            static_cast<uint8_t>(t.bits));
        t.row_q_bit_offsets.resize(static_cast<size_t>(t.out));
        const uint64_t row_bits =
            static_cast<uint64_t>(t.ng) * static_cast<uint64_t>(t.gs) *
            static_cast<uint64_t>(t.bits);
        for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
            const uint64_t bit_offset = static_cast<uint64_t>(row) * row_bits;
            if (bit_offset >
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("NINT row offset overflow");
            }
            t.row_q_bit_offsets[row] = static_cast<int64_t>(bit_offset);
        }
        t.q_packed.insert(t.q_packed.end(), 8, 0);
    }
    if (off != blob.size()) {
        throw std::runtime_error("invalid NINT trailing bytes");
    }
    refresh_nint_descriptor(t);
    return t;
}

Nint8ZeroCpu unpack_nint8_zero(const std::vector<uint8_t> & blob) {
    constexpr size_t block_bytes = 34;
    if (blob.size() < 24 || std::memcmp(blob.data(), "NI80", 4) != 0) {
        throw std::runtime_error("invalid NINT8-0 header");
    }
    Nint8ZeroCpu t;
    size_t off = 4;
    t.axis = read_i32_from(blob, off);
    t.neuron_len = read_i32_from(blob, off);
    uint32_t ndim = read_u32_from(blob, off);
    if (ndim == 0 || t.axis < 0 || t.axis >= static_cast<int>(ndim) ||
        t.neuron_len <= 0 || t.neuron_len % 32 != 0) {
        throw std::runtime_error("invalid NINT8-0 dimensions");
    }
    t.shape.resize(ndim);
    for (uint32_t index = 0; index < ndim; ++index) {
        t.shape[index] = read_i64_from(blob, off);
    }
    t.out = static_cast<int>(read_u32_from(blob, off));
    t.ng = static_cast<int>(read_u32_from(blob, off));
    if (t.out <= 0 || t.ng != t.neuron_len / 32 ||
        t.shape[static_cast<size_t>(t.axis)] != t.out) {
        throw std::runtime_error("NINT8-0 shape/header mismatch");
    }
    int64_t elements = 1;
    for (int64_t value : t.shape) {
        if (value <= 0 || elements > INT64_MAX / value) {
            throw std::runtime_error("invalid NINT8-0 logical shape");
        }
        elements *= value;
    }
    if (elements != static_cast<int64_t>(t.out) * t.neuron_len) {
        throw std::runtime_error("NINT8-0 logical element count mismatch");
    }
    const size_t blocks = static_cast<size_t>(t.out) * t.ng;
    if (off > blob.size() ||
            blocks > (blob.size() - off) / block_bytes ||
            blob.size() - off != blocks * block_bytes) {
        throw std::runtime_error("invalid NINT8-0 block payload length");
    }
    t.q.resize(blocks * 32);
    t.scale_h.resize(blocks);
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t * source = blob.data() + off + block * block_bytes;
        std::memcpy(&t.scale_h[block], source, sizeof(uint16_t));
        const uint16_t scale = t.scale_h[block];
        if ((scale & 0x7c00u) == 0x7c00u || (scale & 0x8000u) != 0u) {
            throw std::runtime_error(
                "NINT8-0 scales must be finite and non-negative");
        }
        std::memcpy(t.q.data() + block * 32, source + 2, 32);
    }
    return t;
}

void require_tp_row_major_weight(
        const std::vector<int64_t> & shape,
        int axis,
        int out,
        int neuron_len,
        const char * format) {
    if (shape.size() != 2 || axis != 0 ||
        shape[0] != out || shape[1] != neuron_len) {
        throw std::runtime_error(
            std::string(format) +
            " tensor parallelism requires a row-major rank-2 weight");
    }
}

static uint64_t nint_row_payload_bits(
        int ng, int gs, int bits) {
    if (ng <= 0 || gs <= 0 || bits < 1 || bits > 8 ||
            static_cast<uint64_t>(ng) >
                std::numeric_limits<uint64_t>::max() /
                    static_cast<uint64_t>(gs) /
                    static_cast<uint64_t>(bits)) {
        throw std::runtime_error("invalid NINT row bit geometry");
    }
    return static_cast<uint64_t>(ng) *
        static_cast<uint64_t>(gs) * static_cast<uint64_t>(bits);
}

static void copy_nint_packed_bits(
        const std::vector<uint8_t> & source,
        uint64_t source_bit,
        std::vector<uint8_t> & destination,
        uint64_t destination_bit,
        uint64_t bit_count) {
    const uint64_t source_capacity =
        static_cast<uint64_t>(source.size()) * 8u;
    const uint64_t destination_capacity =
        static_cast<uint64_t>(destination.size()) * 8u;
    if (source_bit > source_capacity ||
            bit_count > source_capacity - source_bit ||
            destination_bit > destination_capacity ||
            bit_count > destination_capacity - destination_bit) {
        throw std::runtime_error("NINT packed bit range is out of bounds");
    }
    if (((source_bit | destination_bit | bit_count) & 7u) == 0u) {
        std::memcpy(
            destination.data() + static_cast<size_t>(destination_bit >> 3),
            source.data() + static_cast<size_t>(source_bit >> 3),
            static_cast<size_t>(bit_count >> 3));
        return;
    }
    while (bit_count != 0) {
        const int source_shift = static_cast<int>(source_bit & 7u);
        const int destination_shift = static_cast<int>(destination_bit & 7u);
        const int chunk = static_cast<int>(std::min<uint64_t>(
            bit_count,
            static_cast<uint64_t>(std::min(
                8 - source_shift, 8 - destination_shift))));
        const uint8_t mask = static_cast<uint8_t>((1u << chunk) - 1u);
        const uint8_t value = static_cast<uint8_t>(
            (source[static_cast<size_t>(source_bit >> 3)] >> source_shift) &
            mask);
        destination[static_cast<size_t>(destination_bit >> 3)] |=
            static_cast<uint8_t>(value << destination_shift);
        source_bit += static_cast<uint64_t>(chunk);
        destination_bit += static_cast<uint64_t>(chunk);
        bit_count -= static_cast<uint64_t>(chunk);
    }
}

static void require_canonical_nint_cpu(const NintCpu & source) {
    if (source.out <= 0 ||
            source.row_q_bits.size() != static_cast<size_t>(source.out) ||
            source.row_q_bit_offsets.size() != static_cast<size_t>(source.out) ||
            source.row_sub_bits.size() != static_cast<size_t>(source.out) ||
            source.q_packed.size() < 8) {
        throw std::runtime_error("NINT storage is missing canonical row metadata");
    }
    const uint64_t payload_capacity =
        static_cast<uint64_t>(source.q_packed.size() - 8) * 8u;
    for (int row = 0; row < source.out; ++row) {
        const int bits = source.row_q_bits[static_cast<size_t>(row)];
        const int64_t signed_offset =
            source.row_q_bit_offsets[static_cast<size_t>(row)];
        const uint64_t row_bits =
            nint_row_payload_bits(source.ng, source.gs, bits);
        if (signed_offset < 0 ||
                static_cast<uint64_t>(signed_offset) > payload_capacity ||
                row_bits > payload_capacity -
                    static_cast<uint64_t>(signed_offset)) {
            throw std::runtime_error("invalid canonical NINT row metadata");
        }
    }
}

static NintCpu repack_nint_cpu_rows(
        const NintCpu & source,
        const std::vector<int64_t> & rows) {
    require_canonical_nint_cpu(source);
    if (rows.empty()) {
        throw std::runtime_error("cannot create an empty NINT row selection");
    }
    uint64_t total_bits = 0;
    for (int64_t row : rows) {
        if (row < 0 || row >= source.out) {
            throw std::runtime_error("NINT selected row is out of range");
        }
        const uint64_t row_bits = nint_row_payload_bits(
            source.ng, source.gs,
            source.row_q_bits[static_cast<size_t>(row)]);
        if (row_bits > std::numeric_limits<uint64_t>::max() - total_bits) {
            throw std::runtime_error("NINT selected row payload is too large");
        }
        total_bits += row_bits;
    }
    if (total_bits > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            (total_bits + 7u) / 8u >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 8)) {
        throw std::runtime_error("NINT selected row payload is too large");
    }
    NintCpu result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    result.q_packed.assign(static_cast<size_t>((total_bits + 7u) / 8u) + 8, 0);
    result.row_q_bits.resize(rows.size());
    result.row_q_bit_offsets.resize(rows.size());
    result.row_sub_bits.resize(rows.size());
    result.sub_scale.resize(rows.size() * static_cast<size_t>(source.ng));
    result.sub_min.resize(rows.size() * static_cast<size_t>(source.ng));
    result.neuron_scale_h.resize(rows.size());
    result.neuron_min_h.resize(rows.size());
    uint64_t destination_bit = 0;
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const size_t source_row = static_cast<size_t>(rows[destination]);
        const int bits = source.row_q_bits[source_row];
        const uint64_t row_bits =
            nint_row_payload_bits(source.ng, source.gs, bits);
        result.row_q_bits[destination] = static_cast<uint8_t>(bits);
        result.row_sub_bits[destination] = source.row_sub_bits[source_row];
        result.row_q_bit_offsets[destination] =
            static_cast<int64_t>(destination_bit);
        copy_nint_packed_bits(
            source.q_packed,
            static_cast<uint64_t>(source.row_q_bit_offsets[source_row]),
            result.q_packed, destination_bit, row_bits);
        const size_t source_group = source_row * static_cast<size_t>(source.ng);
        const size_t destination_group =
            destination * static_cast<size_t>(source.ng);
        std::copy_n(
            source.sub_scale.begin() + static_cast<ptrdiff_t>(source_group),
            source.ng,
            result.sub_scale.begin() + static_cast<ptrdiff_t>(destination_group));
        std::copy_n(
            source.sub_min.begin() + static_cast<ptrdiff_t>(source_group),
            source.ng,
            result.sub_min.begin() + static_cast<ptrdiff_t>(destination_group));
        result.neuron_scale_h[destination] = source.neuron_scale_h[source_row];
        result.neuron_min_h[destination] = source.neuron_min_h[source_row];
        destination_bit += row_bits;
    }
    refresh_nint_descriptor(result);
    return result;
}

NintCpu slice_nint_cpu_output(
        const NintCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT output shard");
    }
    std::vector<int64_t> rows(static_cast<size_t>(end - begin));
    std::iota(rows.begin(), rows.end(), begin);
    return repack_nint_cpu_rows(source, rows);
}

NintCpu slice_nint_cpu_input_groups(
        const NintCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT input shard");
    }
    require_canonical_nint_cpu(source);
    NintCpu result = source;
    result.ng = static_cast<int>(end - begin);
    const int64_t element_begin = begin * source.gs;
    const int64_t element_end =
        std::min<int64_t>(end * source.gs, source.neuron_len);
    result.neuron_len = static_cast<int>(element_end - element_begin);
    result.shape[1] = result.neuron_len;
    uint64_t total_bits = 0;
    for (int row = 0; row < source.out; ++row) {
        const uint64_t row_bits = nint_row_payload_bits(
            result.ng, source.gs,
            source.row_q_bits[static_cast<size_t>(row)]);
        if (row_bits > std::numeric_limits<uint64_t>::max() - total_bits) {
            throw std::runtime_error("NINT input shard payload is too large");
        }
        total_bits += row_bits;
    }
    if (total_bits > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            (total_bits + 7u) / 8u >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 8)) {
        throw std::runtime_error("NINT input shard payload is too large");
    }
    result.q_packed.assign(static_cast<size_t>((total_bits + 7u) / 8u) + 8, 0);
    result.row_q_bit_offsets.resize(static_cast<size_t>(source.out));
    result.sub_scale.resize(
        static_cast<size_t>(source.out) * result.ng);
    result.sub_min.resize(
        static_cast<size_t>(source.out) * result.ng);
    uint64_t destination_bit = 0;
    for (int row = 0; row < source.out; ++row) {
        const int bits = source.row_q_bits[static_cast<size_t>(row)];
        const uint64_t row_bits =
            nint_row_payload_bits(result.ng, source.gs, bits);
        const uint64_t source_bit = static_cast<uint64_t>(
            source.row_q_bit_offsets[static_cast<size_t>(row)]) +
            static_cast<uint64_t>(begin) *
                static_cast<uint64_t>(source.gs) *
                static_cast<uint64_t>(bits);
        result.row_q_bit_offsets[static_cast<size_t>(row)] =
            static_cast<int64_t>(destination_bit);
        copy_nint_packed_bits(
            source.q_packed, source_bit,
            result.q_packed, destination_bit, row_bits);
        destination_bit += row_bits;
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.sub_scale.data() + destination_group,
            source.sub_scale.data() + source_group,
            static_cast<size_t>(result.ng));
        std::memcpy(
            result.sub_min.data() + destination_group,
            source.sub_min.data() + source_group,
            static_cast<size_t>(result.ng));
    }
    refresh_nint_descriptor(result);
    return result;
}

Nint8ZeroCpu slice_nint8_zero_cpu_output(
        const Nint8ZeroCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT8-0 output shard");
    }
    Nint8ZeroCpu result = source;
    result.out = static_cast<int>(end - begin);
    result.shape[0] = result.out;
    const size_t first_group =
        static_cast<size_t>(begin) * source.ng;
    const size_t group_count =
        static_cast<size_t>(result.out) * source.ng;
    result.q.assign(
        source.q.begin() +
            static_cast<ptrdiff_t>(first_group * 32),
        source.q.begin() +
            static_cast<ptrdiff_t>((first_group + group_count) * 32));
    result.scale_h.assign(
        source.scale_h.begin() + static_cast<ptrdiff_t>(first_group),
        source.scale_h.begin() +
            static_cast<ptrdiff_t>(first_group + group_count));
    return result;
}

Nint8ZeroCpu slice_nint8_zero_cpu_input_groups(
        const Nint8ZeroCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT8-0 input shard");
    }
    Nint8ZeroCpu result = source;
    result.ng = static_cast<int>(end - begin);
    result.neuron_len = result.ng * 32;
    result.shape[1] = result.neuron_len;
    result.q.resize(
        static_cast<size_t>(source.out) * result.ng * 32);
    result.scale_h.resize(
        static_cast<size_t>(source.out) * result.ng);
    for (int row = 0; row < source.out; ++row) {
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.q.data() + destination_group * 32,
            source.q.data() + source_group * 32,
            static_cast<size_t>(result.ng) * 32);
        std::memcpy(
            result.scale_h.data() + destination_group,
            source.scale_h.data() + source_group,
            static_cast<size_t>(result.ng) * sizeof(uint16_t));
    }
    return result;
}
namespace {
int64_t align_up_4(int64_t value) {
    return (value + 3) & ~int64_t(3);
}

// Copy one row's bit run to a byte-aligned staging buffer with 64-bit
// funnel reads; per-group slots then copy from aligned staging bytes.
void copy_row_bits(
        const std::vector<uint8_t>& source,
        uint64_t source_bit,
        int64_t count_bits,
        std::vector<uint8_t>& staging) {
    const int64_t bytes = (count_bits + 7) / 8;
    staging.assign(size_t(bytes) + 8, 0);
    const uint64_t limit = source.size();
    const unsigned shift = source_bit & 7u;
    const uint8_t* base = source.data() + (source_bit >> 3);
    if (shift == 0) {
        std::memcpy(staging.data(), base, size_t(bytes));
    } else {
        for (int64_t index = 0; index < bytes; index += 8) {
            uint64_t low = 0, high = 0;
            const int64_t at = (source_bit >> 3) + index;
            if (uint64_t(at) + 8 <= limit)
                std::memcpy(&low, source.data() + at, 8);
            else if (uint64_t(at) < limit)
                std::memcpy(&low, source.data() + at,
                            size_t(limit - uint64_t(at)));
            if (uint64_t(at) + 16 <= limit)
                std::memcpy(&high, source.data() + at + 8, 8);
            else if (uint64_t(at) + 8 < limit)
                std::memcpy(&high, source.data() + at + 8,
                            size_t(limit - uint64_t(at) - 8));
            const uint64_t window =
                (low >> shift) | (high << (64u - shift));
            const int64_t remain = bytes - index;
            std::memcpy(staging.data() + index, &window,
                        size_t(remain < 8 ? remain : 8));
        }
    }
}

// Build the fused single-row decode layout: rows grouped into uniform-bit
// segments; per group one 4-byte-aligned slot [weight bits | pad | u16
// scale/minimum]. Keeps the original packed tensors for all other paths.
void build_fused_decode_layout(
        NintWeight& w,
        const NintCpu& c,
        const std::function<bool()>& enabled) {
    w.fused_ok = false;
    if (!enabled() || w.aligned_q8 || c.gs < 4 || c.gs > 64 ||
            c.gs % 4 != 0 || c.out <= 0 || c.ng <= 0 ||
            c.row_q_bits.size() != size_t(c.out) ||
            c.sub_scale.size() != size_t(c.out) * size_t(c.ng)) {
        return;
    }
    int histogram[9] = {};
    for (uint8_t bits : c.row_q_bits) {
        if (bits < 2 || bits > 8) return;
        ++histogram[bits];
    }
    const int64_t K = c.ng * c.gs;
    std::vector<int64_t> segment_row_offsets = {0};
    std::vector<int64_t> segment_byte_offsets = {0};
    std::vector<int64_t> segment_bits, segment_slots;
    for (int bits = 2; bits <= 8; ++bits) {
        if (!histogram[bits]) continue;
        const int64_t weight_bytes = (c.gs * bits + 7) / 8;
        const int64_t slot = align_up_4(align_up_4(weight_bytes) + 2);
        segment_bits.push_back(bits);
        segment_slots.push_back(slot);
        segment_row_offsets.push_back(
            segment_row_offsets.back() + histogram[bits]);
        segment_byte_offsets.push_back(
            segment_byte_offsets.back() +
            int64_t(histogram[bits]) * c.ng * slot);
    }
    const size_t total_bytes =
        size_t(segment_byte_offsets.back());
    std::vector<uint8_t> fused(total_bytes, 0);
    std::vector<int32_t> row_ids(size_t(c.out));
    std::vector<int64_t> filled(9, 0);
    std::vector<uint8_t> staging;
    for (int64_t row = 0; row < c.out; ++row) {
        const int bits = c.row_q_bits[row];
        const size_t segment =
            size_t(std::find(segment_bits.begin(), segment_bits.end(), bits) -
                   segment_bits.begin());
        const int64_t slot = segment_slots[segment];
        const int64_t slot_row = filled[bits]++;
        uint8_t* row_base = fused.data() +
            segment_byte_offsets[segment] +
            size_t(slot_row) * size_t(c.ng) * size_t(slot);
        copy_row_bits(
            c.q_packed, uint64_t(c.row_q_bit_offsets[row]),
            int64_t(c.ng) * c.gs * bits, staging);
        const size_t group_bytes_row =
            size_t((uint64_t(c.gs) * uint64_t(bits)) / 8u);
        const bool group_aligned =
            (uint64_t(c.gs) * uint64_t(bits)) % 8u == 0u;
        const uint8_t* staged = staging.data();
        const uint8_t* scale_src = c.sub_scale.data() +
            size_t(row) * size_t(c.ng);
        const uint8_t* min_src =
            c.sub_min.data() + size_t(row) * size_t(c.ng);
        const size_t copy_bytes =
            size_t((uint64_t(c.gs) * uint64_t(bits) + 7u) / 8u);
        for (int64_t group = 0; group < c.ng; ++group) {
            uint8_t* slot_ptr = row_base + size_t(group) * size_t(slot);
            if (group_aligned) {
                std::memcpy(slot_ptr,
                            staged + size_t(group) * group_bytes_row,
                            group_bytes_row);
            } else {
                const uint64_t bit =
                    uint64_t(group) * uint64_t(c.gs) * uint64_t(bits);
                const size_t byte = size_t(bit / 8u);
                const unsigned shift = unsigned(bit % 8u);
                for (size_t index = 0; index < copy_bytes; ++index) {
                    const unsigned value = shift == 0
                        ? unsigned(staged[byte + index])
                        : (unsigned(staged[byte + index]) >> shift) |
                            (unsigned(staged[byte + index + 1])
                             << (8u - shift));
                    slot_ptr[index] = uint8_t(value);
                }
            }
            const uint16_t scales =
                uint16_t(scale_src[size_t(group)]) |
                uint16_t(uint16_t(min_src[size_t(group)]) << 8);
            std::memcpy(slot_ptr + slot - 2, &scales, sizeof(scales));
        }
        row_ids[size_t(segment_row_offsets[segment] + slot_row)] = int32_t(row);
    }

    const auto target = w.q_packed.device();
    w.fused_q = cpu_u8_tensor(
        fused, {int64_t(total_bytes)}).to(target).contiguous();
    w.fused_row_ids = cpu_i32_tensor(
        row_ids, {c.out}).to(target).contiguous();
    w.fused_segment_bits = std::move(segment_bits);
    w.fused_segment_row_offsets = std::move(segment_row_offsets);
    w.fused_segment_byte_offsets = std::move(segment_byte_offsets);
    w.fused_segment_slots = std::move(segment_slots);
    w.fused_ok = true;
}
}  // namespace

NintWeight to_device_nint(const NintCpu & c, bool cuda) {
    if (c.bits < 1 || c.bits > 8) throw std::runtime_error("unsupported NINT bits");
    require_canonical_nint_cpu(c);
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = c.gs;
    w.bits = c.bits;
    w.neuron_len = c.neuron_len;
    w.format_version = c.format_version;
    w.aggregate_bpw = c.aggregate_bpw;
    w.distribution_entropy = c.distribution_entropy;
    w.shape = c.shape;
    w.uniform_qbits=c.row_q_bits.empty()?0:int(c.row_q_bits.front());
    for(auto bit:c.row_q_bits)if(int(bit)!=w.uniform_qbits) {w.uniform_qbits=0;break;}
    w.aligned_q8 = c.gs % 4 == 0;
    for (size_t row = 0; row < c.row_q_bits.size(); ++row) {
        w.aligned_q8 = w.aligned_q8 && c.row_q_bits[row] == 8 &&
            (c.row_q_bit_offsets[row] & 31) == 0;
    }
    w.q_packed = cpu_u8_tensor(
        c.q_packed, {static_cast<int64_t>(c.q_packed.size())});
    w.row_q_bits = cpu_u8_tensor(c.row_q_bits, {c.out});
    w.row_q_bit_offsets = cpu_i64_tensor(c.row_q_bit_offsets, {c.out});
    w.sub_scale = cpu_u8_tensor(c.sub_scale, {c.out, c.ng});
    w.sub_min = cpu_u8_tensor(c.sub_min, {c.out, c.ng});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    w.neuron_min = cpu_f16_to_f32_tensor(c.neuron_min_h, c.out);
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        w.q_packed = w.q_packed.to(target).contiguous();
        w.row_q_bits = w.row_q_bits.to(target).contiguous();
        w.row_q_bit_offsets = w.row_q_bit_offsets.to(target).contiguous();
        w.sub_scale = w.sub_scale.to(target).contiguous();
        w.sub_min = w.sub_min.to(target).contiguous();
        w.neuron_scale = w.neuron_scale.to(target).contiguous();
        w.neuron_min = w.neuron_min.to(target).contiguous();
        build_fused_decode_layout(
            w, c, [] {
                const char* flag = std::getenv("MFQ_NINT_FUSED_DECODE");
                return flag == nullptr || std::strcmp(flag, "0") != 0;
            });
    }
    return w;
}

NintWeight to_gpu_nint(const NintCpu & c) {
    return to_device_nint(c, true);
}

NintWeight to_cuda_device_nint(
        const NintCpu & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nint(c, true);
}

NintWeight to_cpu_nint(const NintCpu & c) {
    return to_device_nint(c, false);
}

NintWeight to_device_mfe_nint(
        const NintCpu & source,
        int local_experts,
        int out_per_expert,
        bool cuda) {
    require_canonical_nint_cpu(source);
    if (local_experts <= 0 || out_per_expert <= 0 ||
            source.out != local_experts * out_per_expert) {
        throw std::runtime_error("invalid MFE NINT expert geometry");
    }
    uint64_t maximum_payload_bits = 0;
    for (int expert = 0; expert < local_experts; ++expert) {
        uint64_t payload_bits = 0;
        for (int local_row = 0; local_row < out_per_expert; ++local_row) {
            const size_t row = static_cast<size_t>(expert) * out_per_expert +
                static_cast<size_t>(local_row);
            const uint64_t row_bits = nint_row_payload_bits(
                source.ng, source.gs, source.row_q_bits[row]);
            if (row_bits > std::numeric_limits<uint64_t>::max() - payload_bits) {
                throw std::runtime_error("MFE NINT expert payload is too large");
            }
            payload_bits += row_bits;
        }
        maximum_payload_bits = std::max(maximum_payload_bits, payload_bits);
    }
    const uint64_t stride_u64 = (maximum_payload_bits + 7u) / 8u + 8u;
    if (stride_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            static_cast<uint64_t>(local_experts) >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                    stride_u64) {
        throw std::runtime_error("MFE NINT expert payload is too large");
    }
    const size_t stride = static_cast<size_t>(stride_u64);
    std::vector<uint8_t> expert_stream(
        static_cast<size_t>(local_experts) * stride, 0);
    std::vector<int64_t> relative_offsets(static_cast<size_t>(source.out));
    for (int expert = 0; expert < local_experts; ++expert) {
        uint64_t destination_bit = 0;
        for (int local_row = 0; local_row < out_per_expert; ++local_row) {
            const size_t row = static_cast<size_t>(expert) * out_per_expert +
                static_cast<size_t>(local_row);
            const int bits = source.row_q_bits[row];
            const uint64_t row_bits =
                nint_row_payload_bits(source.ng, source.gs, bits);
            relative_offsets[row] = static_cast<int64_t>(destination_bit);
            copy_nint_packed_bits(
                source.q_packed,
                static_cast<uint64_t>(source.row_q_bit_offsets[row]),
                expert_stream,
                static_cast<uint64_t>(expert) * stride_u64 * 8u +
                    destination_bit,
                row_bits);
            destination_bit += row_bits;
        }
    }
    NintWeight result = to_device_nint(source, false);
    result.aligned_q8 = false;
    result.q_expert_stride = static_cast<int64_t>(stride);
    result.q_packed = cpu_u8_tensor(
        expert_stream, {local_experts, static_cast<int64_t>(stride)});
    result.row_q_bit_offsets = cpu_i64_tensor(
        relative_offsets, {source.out});
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        result.q_packed = result.q_packed.to(target).contiguous();
        result.row_q_bits = result.row_q_bits.to(target).contiguous();
        result.row_q_bit_offsets =
            result.row_q_bit_offsets.to(target).contiguous();
        result.sub_scale = result.sub_scale.to(target).contiguous();
        result.sub_min = result.sub_min.to(target).contiguous();
        result.neuron_scale = result.neuron_scale.to(target).contiguous();
        result.neuron_min = result.neuron_min.to(target).contiguous();
    }
    return result;
}

NintWeight to_device_nint8_zero(
        const Nint8ZeroCpu & c, bool cuda) {
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = 32;
    w.bits = 8;
    w.neuron_len = c.neuron_len;
    w.q8_zero = true;
    w.shape = c.shape;
    w.q_packed = cpu_u8_tensor(c.q, {c.out, c.ng, 32});
    w.q8_zero_scale = cpu_f16_tensor(c.scale_h, {c.out, c.ng});
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        w.q_packed = w.q_packed.to(target).contiguous();
        w.q8_zero_scale = w.q8_zero_scale.to(target).contiguous();
    }
    return w;
}

NintWeight to_gpu_nint8_zero(const Nint8ZeroCpu & c) {
    return to_device_nint8_zero(c, true);
}

NintWeight to_cuda_device_nint8_zero(
        const Nint8ZeroCpu & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nint8_zero(c, true);
}

NintWeight to_cpu_nint8_zero(const Nint8ZeroCpu & c) {
    return to_device_nint8_zero(c, false);
}

NintCpu select_nint_cpu_rows(
        const NintCpu & source,
        const std::vector<int64_t> & rows) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    return repack_nint_cpu_rows(source, rows);
}

Nint8ZeroCpu select_nint8_zero_cpu_rows(
        const Nint8ZeroCpu & source,
        const std::vector<int64_t> & rows) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (rows.empty()) {
        throw std::runtime_error(
            "cannot create an empty NINT8-0 MoE shard");
    }
    Nint8ZeroCpu result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    const size_t q_row_bytes =
        static_cast<size_t>(source.ng) * 32;
    result.q.resize(rows.size() * q_row_bytes);
    result.scale_h.resize(
        rows.size() * source.ng);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NINT8-0 MoE shard row is out of range");
        }
        std::memcpy(
            result.q.data() +
                destination * q_row_bytes,
            source.q.data() +
                static_cast<size_t>(source_row) *
                    q_row_bytes,
            q_row_bytes);
        std::memcpy(
            result.scale_h.data() +
                destination * source.ng,
            source.scale_h.data() +
                static_cast<size_t>(source_row) *
                    source.ng,
            static_cast<size_t>(source.ng) *
                sizeof(uint16_t));
    }
    return result;
}

mfq_tensor_backend::Tensor dequant_nint8_zero_cpu(const Nint8ZeroCpu & source) {
    MFQ_RUNTIME_CHECK(source.shape.size() == 2,
        "CPU NINT8-0 dense tensor must be rank 2");
    auto packed = to_device_nint8_zero(source, false);
    auto result = mfq_tensor_backend::empty(
        source.shape,
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32));
    const auto * quantized =
        reinterpret_cast<const int8_t *>(packed.q_packed.data_ptr<uint8_t>());
    const mfq_half * scales = packed.q8_zero_scale.data_ptr<mfq_half>();
    float * output = result.data_ptr<float>();
    mfq_parallel_for(0, source.out, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            for (int64_t group = 0; group < source.ng; ++group) {
                const int64_t meta = row * source.ng + group;
                const float scale = static_cast<float>(scales[meta]);
                const int64_t k0 = group * 32;
                const int64_t valid = std::min<int64_t>(
                    32, source.neuron_len - k0);
                for (int64_t index = 0; index < valid; ++index) {
                    output[row * source.neuron_len + k0 + index] =
                        scale * static_cast<float>(quantized[meta * 32 + index]);
                }
            }
        }
    });
    return result.contiguous();
}
