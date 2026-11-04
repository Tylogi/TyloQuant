#pragma once
#include "mfq_tensor_backend.h"
#include <cstdint>

namespace mfq::cuda {
class BatchDecodeInputs;

// Row-table fields for an exclusive cache's physical tiers and allocation maps.
struct HotKvStateFields {
    enum { Keys, Values, ColdKeys, ColdValues, Mapping, SlotIds, Epochs,
           ColdMapping, FreeCold, FreeGpu, ColdControl, Control, Count };
};
// Addresses stay fixed across CUDA Graph replays. A negative mapping reads
// the canonical mapped-RAM row; resident mappings read the GPU copy.
struct HotKvView {
    const void* keys = nullptr;
    const void* values = nullptr;
    const int32_t* mapping = nullptr;
    int64_t cells = 0;
    const int32_t* cold_mapping = nullptr;
    int64_t cold_cells = 0;
};
struct HotKvCache {
    mfq_tensor_backend::Tensor keys, values, mapping, slot_ids, requested, epochs, queue, control;
    // In exclusive mode a logical row lives in exactly one tier. The cold
    // tensors contain only maximum-cells rows of mapped RAM.
    mfq_tensor_backend::Tensor cold_keys, cold_values, cold_mapping, free_cold, free_gpu, cold_control;
    static HotKvCache allocate(int64_t heads, int64_t maximum, int64_t cells,
        int64_t key_width, int64_t value_width, const mfq_tensor_backend::TensorOptions& options);
    bool defined() const { return mapping.defined(); }
    bool exclusive() const { return cold_mapping.defined(); }
    void set_cold_storage(const mfq_tensor_backend::Tensor& keys, const mfq_tensor_backend::Tensor& values);
    // Physical tiers and allocation tables must cover exactly the logical capacity.
    void validate_exclusive_capacity() const;
    void append(const mfq_tensor_backend::Tensor& keys, const mfq_tensor_backend::Tensor& values,
        const mfq_tensor_backend::Tensor& positions);
    // Short contiguous appends: reserve/evict/write in one CTA, optionally
    // publishing the indexer rows in the same launch. False leaves state alone.
    // begin_prepare initializes the next prepare(..., skip_small_prefix=true,
    // epoch_ready=true); both calls must use the same positions and stream.
    bool append_decode(const mfq_tensor_backend::Tensor& keys, const mfq_tensor_backend::Tensor& values,
        const mfq_tensor_backend::Tensor& positions,
        const mfq_tensor_backend::Tensor& index_rows = {}, const mfq_tensor_backend::Tensor& index_storage = {},
        bool begin_prepare = false, const BatchDecodeInputs* inputs = nullptr, int64_t field = 0, int64_t row = 0);
    void bind_decode_state(BatchDecodeInputs&, int64_t field, int64_t row) const;
    std::pair<mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor> gather(int64_t count) const;
    HotKvView view() const;
    std::size_t bytes() const;
    void reset();
    void invalidate(const mfq_tensor_backend::Tensor& positions);
    // Logical append positions are contiguous and increasing. Publish the
    // newest resident window directly; rewritten positions discard old tags.
    void publish(const mfq_tensor_backend::Tensor& source_keys, const mfq_tensor_backend::Tensor& source_values,
        const mfq_tensor_backend::Tensor& positions);
    void prepare(const mfq_tensor_backend::Tensor& source_keys, const mfq_tensor_backend::Tensor& source_values,
        const mfq_tensor_backend::Tensor& selected, const mfq_tensor_backend::Tensor& positions = {}, int64_t columns = 0,
        bool skip_small_prefix = false, bool epoch_ready = false);
};
}
