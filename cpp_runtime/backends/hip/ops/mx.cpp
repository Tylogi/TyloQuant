#include "hip/hip_runtime.h"
#include "mfq_hip_quant_ops.h"
#include "mx.h"
#include "format.h"

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

mfq_tensor_backend::Tensor mxfp8_matmul(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 activation width mismatch");
    if (!x.is_cuda()) {
        MFQ_RUNTIME_CHECK(
            !weight.values.is_cuda() && !weight.scales.is_cuda(),
            "CPU MXFP8 matmul requires CPU-resident weights");
        const int64_t rows = x.size(0);
        const int64_t outputs = weight.out;
        const int64_t width = weight.neuron_len;
        const int64_t scale_columns = weight.scales.size(1);
        const int64_t block_columns = width / scale_columns;
        const int64_t block_rows = block_columns == 128 ? 128 : weight.scales.size(0) == outputs ? 1 : 32;
        const auto * input = x.data_ptr<mfq_half>();
        const auto * values = weight.values.data_ptr<uint8_t>();
        const auto * scales = weight.scales.data_ptr<uint8_t>();
        auto result = mfq_tensor_backend::empty(
            {rows, outputs},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
        auto * output = result.data_ptr<mfq_half>();
        mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
            std::vector<float> accumulators(static_cast<size_t>(rows));
            for (int64_t neuron = begin; neuron < end; ++neuron) {
                std::fill(accumulators.begin(), accumulators.end(), 0.0f);
                for (int64_t column = 0; column < width; ++column) {
                    const uint8_t raw = values[neuron * width + column];
                    const unsigned exponent =
                        (static_cast<unsigned>(raw) >> 3u) & 15u;
                    const unsigned mantissa = static_cast<unsigned>(raw) & 7u;
                    float decoded = exponent == 0u
                        ? std::ldexp(float(mantissa) * 0.125f, -6)
                        : std::ldexp(
                            1.0f + float(mantissa) * 0.125f,
                            static_cast<int>(exponent) - 7);
                    if ((raw & 128u) != 0u) decoded = -decoded;
                    const uint8_t raw_scale = scales[
                        (neuron / block_rows) * scale_columns + column / block_columns];
                    const float weight_value = decoded * std::ldexp(
                        1.0f, static_cast<int>(raw_scale) - 127);
                    for (int64_t row = 0; row < rows; ++row) {
                        accumulators[static_cast<size_t>(row)] = std::fma(
                            static_cast<float>(input[row * width + column]),
                            weight_value,
                            accumulators[static_cast<size_t>(row)]);
                    }
                }
                for (int64_t row = 0; row < rows; ++row) {
                    output[row * outputs + neuron] = mfq_half(
                        accumulators[static_cast<size_t>(row)]);
                }
            }
        });
        return result;
    }
    if (x.size(0) <= 8) {
        return profiler.measure("mxfp8.small_m", [&]() {
            return mxfp8_small_m_cuda(
                weight.values, weight.scales, x);
        });
    }
    if (weight.scales.size(1) == weight.neuron_len / 32)
        return profiler.measure("mxfp8.block32_mfma_f16", [&]() {
            return mxfp8_groupwise_gemm_f16_cuda(weight.values, weight.scales, x.unsqueeze(1), 1);
        });
    return profiler.measure("mxfp8.packed_matmul", [&]() {
        return mxfp8_matmul_f16_cuda(
            weight.values, weight.scales, x);
    });
}

mfq_tensor_backend::Tensor mxfp8_matmul_f32(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 FP32-output activation width mismatch");
    if (x.size(0) <= 8) {
        return profiler.measure("mxfp8.small_m_f32", [&]() {
            return mxfp8_small_m_f32_cuda(
                weight.values, weight.scales, x);
        });
    }
    if (weight.scales.size(1) == weight.neuron_len / 32)
        return profiler.measure("mxfp8.block32_mfma_f32", [&]() {
            return mxfp8_groupwise_gemm_f32_cuda(weight.values, weight.scales, x.unsqueeze(1), 1);
        });
    return profiler.measure("mxfp8.gemm_f32", [&]() {
        return mxfp8_gemm_f32_cuda(
            weight.values, weight.scales, x);
    });
}

mfq_tensor_backend::Tensor mxfp8_cpu_reference(
        const Mxfp8Weight & weight) {
    auto values = weight.values.to(mfq_tensor_backend::kCPU).contiguous();
    auto scales = weight.scales.to(mfq_tensor_backend::kCPU).contiguous();
    const auto * value_bytes = values.data_ptr<uint8_t>();
    const auto * scale_bytes = scales.data_ptr<uint8_t>();
    const int64_t scale_columns = weight.scales.size(1);
    const int64_t block_columns = weight.neuron_len / scale_columns;
    const int64_t block_rows = block_columns == 128 ? 128 : weight.scales.size(0) == weight.out ? 1 : 32;
    std::vector<float> dense(
        static_cast<size_t>(weight.out * weight.neuron_len));
    for (int64_t row = 0; row < weight.out; ++row) {
        for (int64_t column = 0;
             column < weight.neuron_len; ++column) {
            const uint8_t raw = value_bytes[
                static_cast<size_t>(row * weight.neuron_len + column)];
            const unsigned exponent =
                (static_cast<unsigned>(raw) >> 3u) & 15u;
            const unsigned mantissa =
                static_cast<unsigned>(raw) & 7u;
            float value;
            if (exponent == 15u && mantissa == 7u) {
                value = std::numeric_limits<float>::quiet_NaN();
            } else {
                value = exponent == 0u
                    ? std::ldexp(float(mantissa) * 0.125f, -6)
                    : std::ldexp(
                        1.0f + float(mantissa) * 0.125f,
                        static_cast<int>(exponent) - 7);
                if ((raw & 128u) != 0u) value = -value;
            }
            const uint8_t raw_scale = scale_bytes[
                static_cast<size_t>(
                    (row / block_rows) * scale_columns + column / block_columns)];
            const float scale = raw_scale == 255u
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, int(raw_scale) - 127);
            dense[static_cast<size_t>(
                row * weight.neuron_len + column)] = value * scale;
        }
    }
    return mfq_tensor_backend::from_blob(
        dense.data(), {weight.out, weight.neuron_len},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
        .clone().to(weight.values.device())
        .to(mfq_tensor_backend::kFloat16).contiguous();
}

mfq_tensor_backend::Tensor mxfp8_groupwise_matmul(CudaProfiler& profiler,const Mxfp8Weight& weight,
        mfq_tensor_backend::Tensor grouped,int64_t groups) {
    namespace tb=mfq_tensor_backend;grouped=grouped.contiguous().to(tb::kFloat16);
    MFQ_RUNTIME_CHECK(groups>0 && grouped.dim()==3 && grouped.size(1)==groups &&
        grouped.size(2)==weight.neuron_len && weight.out%groups==0,"invalid groupwise MXFP8 projection");
    const auto per_group=weight.out/groups;
    const bool block128=weight.neuron_len%128==0 && weight.scales.size(0)==(weight.out+127)/128 &&
        weight.scales.size(1)==weight.neuron_len/128;
    if(grouped.size(0)<=8 && per_group%32==0)
        return profiler.measure("mxfp8.groupwise_small_m_f16",[&] {
            return mxfp8_groupwise_small_m_cuda(weight.values,weight.scales,grouped,groups);
        });
    return profiler.measure("mxfp8.groupwise_mfma_f16",[&] {
        return mxfp8_groupwise_gemm_f16_cuda(weight.values,weight.scales,grouped,groups);
    });
}

mfq_tensor_backend::Tensor mxfp8_groupwise_matmul_f32(CudaProfiler& profiler,const Mxfp8Weight& weight,
        mfq_tensor_backend::Tensor grouped,int64_t groups) {
    namespace tb=mfq_tensor_backend;grouped=grouped.contiguous().to(tb::kFloat16);
    MFQ_RUNTIME_CHECK(groups>0 && grouped.dim()==3 && grouped.size(1)==groups &&
        grouped.size(2)==weight.neuron_len && weight.out%groups==0,"invalid groupwise MXFP8 projection");
    const auto per_group=weight.out/groups;
    const bool block128=weight.neuron_len%128==0 && weight.scales.size(0)==(weight.out+127)/128 &&
        weight.scales.size(1)==weight.neuron_len/128;
    if(grouped.size(0)<=8 && per_group%32==0)
        return profiler.measure("mxfp8.groupwise_small_m_f32",[&] {
            return mxfp8_groupwise_small_m_f32_cuda(weight.values,weight.scales,grouped,groups);
        });
    return profiler.measure("mxfp8.groupwise_mfma_f32",[&] {
        return mxfp8_groupwise_gemm_f32_cuda(weight.values,weight.scales,grouped,groups);
    });
}

mfq_tensor_backend::Tensor mxfp4_matmul(
        const Mxfp4Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP4 activation width mismatch");
    if (x.is_cuda()) {
        MFQ_RUNTIME_CHECK(
            weight.values.is_cuda() && weight.scales.is_cuda(),
            "CUDA MXFP4 matmul requires CUDA-resident weights");
        return mxfp4_matmul_f16_cuda(
            weight.values, weight.scales, x);
    }
    MFQ_RUNTIME_CHECK(
        !weight.values.is_cuda() && !weight.scales.is_cuda(),
        "CPU MXFP4 matmul requires CPU-resident weights");
    const int64_t rows = x.size(0);
    const int64_t outputs = weight.out;
    const int64_t width = weight.neuron_len;
    const int64_t packed_columns = width / 2;
    const int64_t scale_columns = width / 32;
    const auto * input = x.data_ptr<mfq_half>();
    const auto * values = weight.values.data_ptr<uint8_t>();
    const auto * scales = weight.scales.data_ptr<uint8_t>();
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    auto * output = result.data_ptr<mfq_half>();
    static constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            for (int64_t column = 0; column < width; ++column) {
                const uint8_t packed = values[
                    neuron * packed_columns + column / 2];
                const uint8_t code = static_cast<uint8_t>(
                    (packed >> ((column & 1) * 4)) & 15u);
                float decoded = magnitude[code & 7u];
                if ((code & 8u) != 0u) decoded = -decoded;
                const uint8_t raw_scale = scales[
                    neuron * scale_columns + column / 32];
                const float weight_value = decoded * std::ldexp(
                    1.0f, static_cast<int>(raw_scale) - 127);
                for (int64_t row = 0; row < rows; ++row) {
                    accumulators[static_cast<size_t>(row)] = std::fma(
                        static_cast<float>(input[row * width + column]),
                        weight_value,
                        accumulators[static_cast<size_t>(row)]);
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] = mfq_half(
                    accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}
size_t checked_mxfp8_size(
        uint64_t left, uint64_t right,
        const char * label) {
    if (left == 0 || right == 0 ||
            left > std::numeric_limits<size_t>::max() / right) {
        throw std::runtime_error(
            std::string("invalid MXFP8 ") + label);
    }
    return static_cast<size_t>(left * right);
}

Mxfp8Cpu unpack_mxfp8(
        const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 56;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "MXT1", 4) != 0 ||
            blob[4] != 1 || blob[5] != 8 ||
            blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid MXFP8 payload header");
    }
    size_t offset = 8;
    const uint64_t rows = read_u64_from(blob, offset);
    const uint64_t columns = read_u64_from(blob, offset);
    const uint64_t storage_rows = read_u64_from(blob, offset);
    const uint64_t storage_columns = read_u64_from(blob, offset);
    const uint64_t scale_rows = read_u64_from(blob, offset);
    const uint64_t scale_columns = read_u64_from(blob, offset);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns % 32 != 0 || storage_rows != rows || storage_columns != columns ||
            !((columns % 128 == 0 && scale_rows == (rows + 127) / 128 && scale_columns == columns / 128) ||
              (scale_rows == (rows + 31) / 32 && scale_columns == columns / 32) ||
              (scale_rows == rows && scale_columns == columns / 32))) {
        throw std::runtime_error("invalid MXFP8 payload geometry");
    }
    const size_t value_bytes = checked_mxfp8_size(
        storage_rows, storage_columns, "value size");
    const size_t scale_bytes = checked_mxfp8_size(
        scale_rows, scale_columns, "scale size");
    if (value_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - value_bytes) {
        throw std::runtime_error("invalid MXFP8 payload length");
    }
    Mxfp8Cpu result;
    result.block_columns = scale_columns == columns / 32 ? 32 : 128;
    result.block_rows = result.block_columns == 128 ? 128 : scale_rows == rows ? 1 : 32;
    result.out = static_cast<int64_t>(rows);
    result.neuron_len = static_cast<int64_t>(columns);
    result.values.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + value_bytes));
    offset += value_bytes;
    result.scales.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.end());
    if (std::any_of(
            result.values.begin(), result.values.end(),
            [](uint8_t value) { return (value & 0x7fu) == 0x7fu; })) {
        throw std::runtime_error("MXFP8 payload contains an E4M3 NaN code");
    }
    if (std::find(result.scales.begin(), result.scales.end(), 255u) !=
            result.scales.end()) {
        throw std::runtime_error("MXFP8 payload contains an E8M0 NaN scale");
    }
    return result;
}

Mxfp4Cpu unpack_mxfp4(
        const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 56;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "MXT1", 4) != 0 ||
            blob[4] != 1 || blob[5] != 4 ||
            blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid MXFP4 payload header");
    }
    size_t offset = 8;
    const uint64_t rows = read_u64_from(blob, offset);
    const uint64_t columns = read_u64_from(blob, offset);
    const uint64_t storage_rows = read_u64_from(blob, offset);
    const uint64_t storage_columns = read_u64_from(blob, offset);
    const uint64_t scale_rows = read_u64_from(blob, offset);
    const uint64_t scale_columns = read_u64_from(blob, offset);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns % 32 != 0 || storage_rows != rows ||
            storage_columns != columns / 2 || scale_rows != rows ||
            scale_columns != columns / 32) {
        throw std::runtime_error("invalid MXFP4 payload geometry");
    }
    const size_t value_bytes = checked_mxfp8_size(
        storage_rows, storage_columns, "MXFP4 value size");
    const size_t scale_bytes = checked_mxfp8_size(
        scale_rows, scale_columns, "MXFP4 scale size");
    if (value_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - value_bytes) {
        throw std::runtime_error("invalid MXFP4 payload length");
    }
    Mxfp4Cpu result;
    result.out = static_cast<int64_t>(rows);
    result.neuron_len = static_cast<int64_t>(columns);
    result.values.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + value_bytes));
    offset += value_bytes;
    result.scales.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset), blob.end());
    if (std::find(result.scales.begin(), result.scales.end(), 255u) !=
            result.scales.end()) {
        throw std::runtime_error("MXFP4 payload contains an E8M0 NaN scale");
    }
    return result;
}

mfq_tensor_backend::Tensor dequant_mxfp4_cpu(const Mxfp4Cpu & source) {
    auto dense = mfq_tensor_backend::empty(
        {source.out, source.neuron_len},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32));
    float * destination = dense.data_ptr<float>();
    static constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const int64_t packed_columns = source.neuron_len / 2;
    const int64_t scale_columns = source.neuron_len / 32;
    for (int64_t row = 0; row < source.out; ++row) {
        for (int64_t column = 0; column < source.neuron_len; ++column) {
            const uint8_t packed = source.values[
                static_cast<size_t>(row * packed_columns + column / 2)];
            const uint8_t code = static_cast<uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const uint8_t raw_scale = source.scales[
                static_cast<size_t>(row * scale_columns + column / 32)];
            const float scale = raw_scale == 255u
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, static_cast<int>(raw_scale) - 127);
            const float value = magnitude[code & 7u] * scale;
            destination[row * source.neuron_len + column] =
                (code & 8u) == 0u ? value : -value;
        }
    }
    return dense.to(mfq_tensor_backend::kFloat16);
}

Mxfp4Cpu select_mxfp4_cpu_rows(
        const Mxfp4Cpu & source,
        const std::vector<int64_t> & rows) {
    Mxfp4Cpu result;
    result.out = static_cast<int64_t>(rows.size());
    result.neuron_len = source.neuron_len;
    const size_t value_row = static_cast<size_t>(source.neuron_len / 2);
    const size_t scale_row = static_cast<size_t>(source.neuron_len / 32);
    result.values.resize(rows.size() * value_row);
    result.scales.resize(rows.size() * scale_row);
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 || source_row >= source.out) {
            throw std::runtime_error("MXFP4 selected row is out of range");
        }
        std::memcpy(
            result.values.data() + destination * value_row,
            source.values.data() + static_cast<size_t>(source_row) * value_row,
            value_row);
        std::memcpy(
            result.scales.data() + destination * scale_row,
            source.scales.data() + static_cast<size_t>(source_row) * scale_row,
            scale_row);
    }
    return result;
}

Mxfp4Cpu slice_mxfp4_cpu(
        const Mxfp4Cpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid MXFP4 output shard");
        }
        std::vector<int64_t> rows(static_cast<size_t>(end - begin));
        std::iota(rows.begin(), rows.end(), begin);
        return select_mxfp4_cpu_rows(source, rows);
    }
    if (axis != TensorParallelAxis::Input || begin < 0 || begin >= end ||
            end > source.neuron_len || begin % 32 != 0 || end % 32 != 0) {
        throw std::runtime_error(
            "MXFP4 input shards must preserve 32-value scale blocks");
    }
    Mxfp4Cpu result;
    result.out = source.out;
    result.neuron_len = end - begin;
    const size_t source_value_row =
        static_cast<size_t>(source.neuron_len / 2);
    const size_t destination_value_row =
        static_cast<size_t>(result.neuron_len / 2);
    const size_t source_scale_row =
        static_cast<size_t>(source.neuron_len / 32);
    const size_t destination_scale_row =
        static_cast<size_t>(result.neuron_len / 32);
    result.values.resize(static_cast<size_t>(result.out) * destination_value_row);
    result.scales.resize(static_cast<size_t>(result.out) * destination_scale_row);
    for (int64_t row = 0; row < result.out; ++row) {
        std::memcpy(
            result.values.data() + static_cast<size_t>(row) * destination_value_row,
            source.values.data() + static_cast<size_t>(row) * source_value_row +
                static_cast<size_t>(begin / 2),
            destination_value_row);
        std::memcpy(
            result.scales.data() + static_cast<size_t>(row) * destination_scale_row,
            source.scales.data() + static_cast<size_t>(row) * source_scale_row +
                static_cast<size_t>(begin / 32),
            destination_scale_row);
    }
    return result;
}
Mxfp8Cpu slice_mxfp8_cpu(
        const Mxfp8Cpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis != TensorParallelAxis::Output &&
            axis != TensorParallelAxis::Input) {
        throw std::runtime_error("MXFP8 slicing requires output or input axis");
    }
    const int64_t extent = axis == TensorParallelAxis::Output
        ? source.out : source.neuron_len;
    const auto alignment = axis == TensorParallelAxis::Output ? source.block_rows : source.block_columns;
    if (begin < 0 || begin >= end || end > extent ||
            begin % alignment != 0 ||
            (end != extent && end % alignment != 0)) {
        throw std::runtime_error(
            "MXFP8 tensor-parallel shards must preserve native scale blocks");
    }
    Mxfp8Cpu result;
    result.block_rows = source.block_rows;
    result.block_columns = source.block_columns;
    if (axis == TensorParallelAxis::Output) {
        result.out = end - begin;
        result.neuron_len = source.neuron_len;
        const size_t value_begin =
            static_cast<size_t>(begin * source.neuron_len);
        const size_t value_end =
            static_cast<size_t>(end * source.neuron_len);
        result.values.assign(
            source.values.begin() + static_cast<ptrdiff_t>(value_begin),
            source.values.begin() + static_cast<ptrdiff_t>(value_end));
        const int64_t scale_columns = source.neuron_len / source.block_columns;
        const size_t scale_begin =
            static_cast<size_t>((begin / source.block_rows) * scale_columns);
        const size_t scale_end = static_cast<size_t>(
            ((end + source.block_rows - 1) / source.block_rows) * scale_columns);
        result.scales.assign(
            source.scales.begin() + static_cast<ptrdiff_t>(scale_begin),
            source.scales.begin() + static_cast<ptrdiff_t>(scale_end));
        return result;
    }

    result.out = source.out;
    result.neuron_len = end - begin;
    result.values.resize(
        static_cast<size_t>(result.out * result.neuron_len));
    for (int64_t row = 0; row < source.out; ++row) {
        std::memcpy(
            result.values.data() +
                static_cast<size_t>(row * result.neuron_len),
            source.values.data() +
                static_cast<size_t>(row * source.neuron_len + begin),
            static_cast<size_t>(result.neuron_len));
    }
    const int64_t source_scale_columns = source.neuron_len / source.block_columns;
    const int64_t result_scale_columns = result.neuron_len / source.block_columns;
    const int64_t scale_rows = (source.out + source.block_rows - 1) / source.block_rows;
    result.scales.resize(
        static_cast<size_t>(scale_rows * result_scale_columns));
    for (int64_t row = 0; row < scale_rows; ++row) {
        std::memcpy(
            result.scales.data() +
                static_cast<size_t>(row * result_scale_columns),
            source.scales.data() + static_cast<size_t>(
                row * source_scale_columns + begin / source.block_columns),
            static_cast<size_t>(result_scale_columns));
    }
    return result;
}
Mxfp8Weight to_device_mxfp8(
        const Mxfp8Cpu & source,
        bool cuda,
        int device) {
    Mxfp8Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values,
        {source.out, source.neuron_len});
    result.scales = cpu_u8_tensor(
        source.scales,
        {(source.out + source.block_rows - 1) / source.block_rows, source.neuron_len / source.block_columns});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}

Mxfp8Weight to_cuda_device_mxfp8(
        const Mxfp8Cpu & source,
        int device) {
    return to_device_mxfp8(source, true, device);
}
Mxfp4Weight to_device_mxfp4(
        const Mxfp4Cpu & source, bool cuda, int device) {
    Mxfp4Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values, {source.out, source.neuron_len / 2});
    result.scales = cpu_u8_tensor(
        source.scales, {source.out, source.neuron_len / 32});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}
