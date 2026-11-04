#include "../ops/include/batch_selected_attention.h"

#include <algorithm>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
namespace {

IndirectKvView view(const BatchDecodeInputs& inputs, int64_t field) {
    return {inputs.states().data_ptr<int64_t>(), inputs.enabled().data_ptr<int64_t>(),
            int(inputs.rows()), int(field)};
}

__global__ void store_index_pool(IndirectKvView cache, const int64_t* positions,
        const mfq_half* raw, mfq_half* pooled, int width, int pool) {
    const int row = blockIdx.x;
    const int64_t position = positions[row];
    const bool active = cache.active(row) && position >= 0 && position < cache.capacity(row);
    auto* index = active ? static_cast<mfq_half*>(cache.address(row)) : nullptr;
    const int64_t start = position / pool * pool;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const mfq_half fresh = raw[int64_t(row) * width + column];
        if (active)
            index[position * width + column] = fresh;
        float sum = 0;
        for (int offset = 0; offset < pool; ++offset) {
            const int64_t token = start + offset;
            // The current row comes from projection input, never another CTA's write.
            const float value = !active || token > position ? 0.f : token == position ? float(fresh) :
                float(index[token * width + column]);
            sum = __fadd_rn(sum, value);
        }
        pooled[int64_t(row) * width + column] = mfq_half(__fdiv_rn(sum, float(pool)));
    }
}

__global__ void store_pool(IndirectKvView cache, const int64_t* positions,
        const mfq_half* pooled, int width, int pool) {
    const int row = blockIdx.x;
    const auto block = positions[row] / pool;
    if (!cache.active(row) || block < 0 || block >= cache.capacity(row))
        return;
    auto* target = static_cast<mfq_half*>(cache.address(row));
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        target[block * width + column] = pooled[int64_t(row) * width + column];
}

template<class Query>
__global__ void index_scores(IndirectKvView cache, const int64_t* positions,
        const Query* query, float* scores, int pools, int heads, int width, int pool) {
    const int lane = threadIdx.x & 31;
    const int block = int(blockIdx.x) * 4 + threadIdx.x / 32;
    const int row = blockIdx.y;
    if (block >= pools)
        return;
    if (!cache.active(row) || block >= cache.capacity(row) ||
            int64_t(block + 1) * pool > positions[row] + 1) {
        if (lane == 0)
            scores[int64_t(row) * pools + block] = -1e30f;
        return;
    }
    const auto* keys = static_cast<const mfq_half*>(cache.address(row));
    float score = 0;
    for (int head = 0; head < heads; ++head) {
        float dot = 0;
        for (int column = lane; column < width; column += 32)
            dot = fmaf(float(query[(int64_t(row) * heads + head) * width + column]),
                       float(keys[int64_t(block) * width + column]), dot);
        for (int delta = 16; delta; delta /= 2)
            dot += __shfl_down_sync(0xffffffff, dot, delta);
        score += fmaxf(0.f, dot);
    }
    if (lane == 0)
        scores[int64_t(row) * pools + block] = score / sqrtf(float(width));
}

__global__ void select_tokens(IndirectKvView cache, const int64_t* positions, const int64_t* blocks,
        int32_t* selected, int count, int pool, int budget, int columns) {
    const int row = blockIdx.y;
    const int64_t position = positions[row];
    for (int slot = int(blockIdx.x) * blockDim.x + threadIdx.x; slot < columns;
            slot += gridDim.x * blockDim.x) {
        int64_t token = -1;
        if (cache.active(row)) {
            if (position + 1 <= budget) {
                if (slot <= position)
                    token = slot;
            } else if (slot < count * pool) {
                const int64_t block = blocks[int64_t(row) * count + slot / pool];
                if (block >= 0 && (block + 1) * pool <= position + 1)
                    token = block * pool + slot % pool;
            } else if (slot >= budget && slot - budget < (position + 1) % pool) {
                token = (position + 1) / pool * pool + slot - budget;
            }
            if (token >= cache.capacity(row))
                token = -1;
        }
        selected[int64_t(row) * columns + slot] = int32_t(token);
    }
}

} // namespace

tb::Tensor batch_index_pool(const BatchDecodeInputs& inputs, int64_t field,
        const tb::Tensor& raw, int64_t pool) {
    auto source = raw.to(tb::kFloat16).contiguous();
    auto pooled = tb::empty({inputs.rows(), 1, raw.size(-1)}, source.options());
    store_index_pool<<<unsigned(inputs.rows()), 128, 0, mfq_current_cuda_stream()>>>(view(inputs, field),
        inputs.positions().data_ptr<int64_t>(), source.data_ptr<mfq_half>(), pooled.data_ptr<mfq_half>(),
        int(raw.size(-1)), int(pool));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return pooled;
}

void batch_pool_store(const BatchDecodeInputs& inputs, int64_t field,
        const tb::Tensor& pooled, int64_t pool) {
    auto source = pooled.to(tb::kFloat16).contiguous();
    store_pool<<<unsigned(inputs.rows()), 128, 0, mfq_current_cuda_stream()>>>(view(inputs, field),
        inputs.positions().data_ptr<int64_t>(), source.data_ptr<mfq_half>(), int(pooled.size(-1)), int(pool));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

tb::Tensor batch_index_scores(const BatchDecodeInputs& inputs, int64_t field,
        const tb::Tensor& query, int64_t pool, int64_t pools) {
    auto source = query.contiguous();
    auto scores = tb::empty({1, inputs.rows(), pools}, query.options().dtype(tb::kFloat32));
    const dim3 grid(unsigned((pools + 3) / 4), unsigned(inputs.rows()));
    const auto launch = [&](auto* q) {
        index_scores<<<grid, 128, 0, mfq_current_cuda_stream()>>>(view(inputs, field),
            inputs.positions().data_ptr<int64_t>(), q, scores.data_ptr<float>(), int(pools),
            int(query.size(2)), int(query.size(3)), int(pool));
    };
    if (source.scalar_type() == tb::kFloat16)
        launch(source.data_ptr<mfq_half>());
    else {
        source = source.to(tb::kFloat32);
        launch(source.data_ptr<float>());
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}

tb::Tensor batch_attention_selection(const BatchDecodeInputs& inputs, int64_t field, const tb::Tensor& blocks,
        int64_t pool, int64_t budget, int64_t columns) {
    auto selected = tb::empty({inputs.rows(), 1, columns}, inputs.positions().options().dtype(tb::kInt32));
    select_tokens<<<dim3(unsigned((columns + 255) / 256), unsigned(inputs.rows())),
        256, 0, mfq_current_cuda_stream()>>>(view(inputs, field), inputs.positions().data_ptr<int64_t>(),
        blocks.defined() ? blocks.data_ptr<int64_t>() : nullptr, selected.data_ptr<int32_t>(),
        blocks.defined() ? int(blocks.size(-1)) : 0, int(pool), int(budget), int(columns));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return selected;
}

} // namespace mfq::cuda
