#include "batch_decode_inputs.h"
#include "mfq_cuda_indirect_kv.h"

#include <algorithm>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
namespace {

__global__ void transfer_state(const int64_t* table, const int64_t* enabled, int rows,
        int field, int row, unsigned char* local, int64_t bytes, bool store) {
    const auto address = table[2 * (int64_t(field) * rows + row)];
    auto* state = reinterpret_cast<unsigned char*>(address);
    const bool active = enabled[row] && state;
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < bytes; index += int64_t(gridDim.x) * blockDim.x) {
        if (store) {
            if (active)
                state[index] = local[index];
        } else {
            local[index] = active ? state[index] : 0;
        }
    }
}

__global__ void transfer_sequence(const int64_t* table, const int64_t* enabled, int rows,
        int field, int row, const int64_t* indices, int64_t columns, int64_t heads,
        int64_t width_bytes, unsigned char* local, bool store, int hot_field = -1, bool values = false) {
    const IndirectKvView cache{table, enabled, rows, field, hot_field};
    const int64_t size = heads * columns * width_bytes;
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < size; index += int64_t(gridDim.x) * blockDim.x) {
        const auto byte = index % width_bytes;
        const auto column = (index / width_bytes) % columns;
        const auto head = index / (width_bytes * columns);
        const auto position = indices[column];
        auto* physical = cache.resolve<unsigned char>(row, position, head, width_bytes, 0, values);
        if (store) {
            if (physical)
                physical[byte] = local[index];
        } else {
            local[index] = physical ? physical[byte] : 0;
        }
    }
}

__global__ void mask_routes(int32_t* routes, float* weights, const int64_t* enabled,
        int64_t offset, int64_t count, int64_t topk) {
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < count; index += int64_t(gridDim.x) * blockDim.x) {
        if (!enabled[offset + index / topk]) {
            routes[index] = -1;
            weights[index] = 0;
        }
    }
}

int blocks(int64_t bytes) { return static_cast<int>(std::min<int64_t>((bytes + 255) / 256, 4096)); }

void transfer(const BatchDecodeInputs& inputs, int64_t field, int64_t row,
        const tb::Tensor& value, bool store) {
    if (!value.numel())
        return;
    transfer_state<<<blocks(value.nbytes()), 256, 0, mfq_current_cuda_stream()>>>(
        inputs.states().data_ptr<int64_t>(), inputs.enabled().data_ptr<int64_t>(),
        inputs.rows(), field, row, static_cast<unsigned char*>(value.data_ptr()), value.nbytes(), store);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

void batch_state_load(const BatchDecodeInputs& inputs, int64_t field, int64_t row, const tb::Tensor& output) {
    transfer(inputs, field, row, output, false);
}

void batch_state_store(const BatchDecodeInputs& inputs, int64_t field, int64_t row, const tb::Tensor& input) {
    transfer(inputs, field, row, input, true);
}

tb::Tensor batch_state_gather(const BatchDecodeInputs& inputs, int64_t field, int64_t row,
        const tb::Tensor& indices, int64_t heads, int64_t width, const tb::TensorOptions& options,
        int64_t hot_field, bool values) {
    auto output = tb::empty({1, heads, indices.numel(), width}, options);
    transfer_sequence<<<blocks(output.nbytes()), 256, 0, mfq_current_cuda_stream()>>>(
        inputs.states().data_ptr<int64_t>(), inputs.enabled().data_ptr<int64_t>(), inputs.rows(),
        field, row, indices.data_ptr<int64_t>(), indices.numel(), heads, width * output.element_size(),
        static_cast<unsigned char*>(output.data_ptr()), false, hot_field, values);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void batch_state_scatter(const BatchDecodeInputs& inputs, int64_t field, int64_t row,
        const tb::Tensor& indices, const tb::Tensor& values) {
    MFQ_RUNTIME_CHECK(values.dim() == 4 && values.size(0) == 1 && values.size(2) == indices.numel(),
                      "invalid indirect decode sequence update");
    auto contiguous = values.contiguous();
    transfer_sequence<<<blocks(values.nbytes()), 256, 0, mfq_current_cuda_stream()>>>(
        inputs.states().data_ptr<int64_t>(), inputs.enabled().data_ptr<int64_t>(), inputs.rows(),
        field, row, indices.data_ptr<int64_t>(), indices.numel(), values.size(1),
        values.size(3) * values.element_size(), static_cast<unsigned char*>(contiguous.data_ptr()), true);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

void mask_decode_routes(const tb::Tensor& routes, const tb::Tensor& weights,
        const tb::Tensor& enabled, int64_t offset) {
    mask_routes<<<blocks(routes.numel()), 256, 0, mfq_current_cuda_stream()>>>(
        routes.data_ptr<int32_t>(), weights.data_ptr<float>(), enabled.data_ptr<int64_t>(),
        offset, routes.numel(), routes.size(1));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace mfq::cuda
