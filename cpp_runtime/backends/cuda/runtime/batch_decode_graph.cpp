#include "batch_decode_graph.h"
#include "quant_linear.h"

#include <algorithm>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;

int64_t batch_decode_row_bucket(int64_t rows, int64_t maximum) {
    MFQ_RUNTIME_CHECK(rows > 0 && rows <= maximum, "invalid continuous decode row count");
    int64_t bucket = 1;
    while (bucket < rows && bucket < maximum)
        bucket = bucket > maximum / 2 ? maximum : bucket * 2;
    return bucket;
}

int64_t batch_decode_cache_capacity(int64_t required, int64_t rows, int64_t maximum, int64_t initial) {
    MFQ_RUNTIME_CHECK(required > 0 && required <= maximum, "batch graph exceeds max_context");
    auto capacity = std::min<int64_t>(rows > 8 ? initial : 256, maximum);
    if (rows > 16 && capacity < required) {
        const auto chunks = 1 + (required - 1) / 128;
        capacity = chunks > maximum / 128 ? maximum : chunks * 128;
    }
    while (capacity < required)
        capacity = capacity > maximum / 2 ? maximum : capacity * 2;
    return capacity;
}

BatchDecodeInputs::BatchDecodeInputs(int64_t rows, int64_t fields, int device)
    : rows_(rows), fields_(fields) {
    auto host = tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true);
    auto cuda = tb::TensorOptions().device(tb::Device(tb::kCUDA, device)).dtype(tb::kInt64);
    host_metadata_ = tb::zeros({5, rows}, host);
    metadata_ = tb::empty({5, rows}, cuda);
    host_states_ = tb::zeros({fields, rows, 2}, host);
    states_ = tb::empty({fields, rows, 2}, cuda);
    for (int64_t row = 0; row < rows; ++row)
        this->row(row, 0, 0, -1, false);
}

void BatchDecodeInputs::row(int64_t index, int64_t token, int64_t position, int64_t slot, bool enabled) {
    MFQ_RUNTIME_CHECK(index >= 0 && index < rows_, "decode row is outside its bucket");
    auto* values = host_metadata_.data_ptr<int64_t>();
    values[index] = enabled ? token : 0;
    values[rows_ + index] = enabled ? position : 0;
    values[2 * rows_ + index] = enabled ? position + 1 : 1;
    values[3 * rows_ + index] = enabled;
    values[4 * rows_ + index] = enabled ? slot : -1;
}

void BatchDecodeInputs::update_rows(const std::vector<BatchDecodeRow>& rows) {
    MFQ_RUNTIME_CHECK(rows.size() <= std::size_t(rows_), "decode inputs exceed their row bucket");
    host_states_.zero_();
    for (int64_t index = 0; index < rows_; ++index) {
        const bool enabled = index < int64_t(rows.size());
        const auto value = enabled ? rows[index] : BatchDecodeRow{};
        row(index, value.token, value.position, value.slot, enabled);
    }
}

void BatchDecodeInputs::state(int64_t row, int64_t field, const Tensor& value, int64_t axis) {
    MFQ_RUNTIME_CHECK(row >= 0 && row < rows_ && field >= 0 && field < fields_, "invalid decode state field");
    MFQ_RUNTIME_CHECK(!value.defined() || (value.is_cuda() && value.is_contiguous()),
                      "indirect decode state must be contiguous CUDA storage");
    auto* values = host_states_.data_ptr<int64_t>() + 2 * (field * rows_ + row);
    values[0] = value.defined() ? reinterpret_cast<int64_t>(value.data_ptr()) : 0;
    values[1] = value.defined() ? (axis >= 0 ? value.size(axis) : value.numel()) : 0;
}

void BatchDecodeInputs::upload() {
    metadata_.copy_(host_metadata_);
    states_.copy_(host_states_);
}

void BatchDecodeInputs::validate_position(int64_t row, int64_t maximum) const {
    MFQ_RUNTIME_CHECK(row >= 0 && row < rows_, "invalid decode row");
    const auto* metadata = host_metadata_.data_ptr<int64_t>();
    MFQ_RUNTIME_CHECK(!metadata[3 * rows_ + row] ||
        (metadata[rows_ + row] >= 0 && metadata[rows_ + row] < maximum),
        "exclusive KV decode position exceeds logical capacity");
}

void BatchDecodeInputs::copy_from(const BatchDecodeInputs& source) {
    host_metadata_.copy_(source.host_metadata_);
    host_states_.copy_(source.host_states_);
    upload();
    join_batch_decode_stream(mfq_get_current_cuda_stream(metadata_.get_device()),
                             mfq_get_current_cuda_stream(source.metadata_.get_device()));
}

void join_batch_decode_stream(const MfqCudaStream& worker, const MfqCudaStream& primary) {
    if (worker.device_index() == primary.device_index() && worker.stream() == primary.stream())
        return;
    MfqCudaGuard worker_guard(worker.device_index());
    Event ready;
    ready.record(worker);
    MfqCudaGuard primary_guard(primary.device_index());
    MFQ_CUDA_CHECK(cudaStreamWaitEvent(primary.stream(), ready.get(), 0));
}

BatchDecodeGraphCache::BatchDecodeGraphCache(bool current_stream)
    : stream_(current_stream ? mfq_get_current_cuda_stream() : mfq_get_stream_from_pool(false)) {}
BatchDecodeGraphCache::~BatchDecodeGraphCache() { clear(); }

BatchDecodeGraphCache::Entry& BatchDecodeGraphCache::get(BatchDecodeGraphKey key, int64_t fields, int device) {
    auto& entry = entries_[key];
    if (!entry) {
        entry = std::make_unique<Entry>();
        entry->key = key;
        entry->inputs = std::make_unique<BatchDecodeInputs>(key.rows, fields, device);
    }
    return *entry;
}

void BatchDecodeGraphCache::prepare_pool(const ParallelConfig& parallel,
        const ModelParallelCollectiveRuntime& collectives, bool external) {
    if (compute_streams_.empty())
        compute_streams_ = make_cuda_graph_compute_streams(stream_, parallel);
    if (pool_ || external_pool_)
        return;
    external_pool_ = external;
    if (external)
        return;
    pool_ = std::make_unique<MfqCudaGraph>();
    mfq_prepare_cuda_graph_memory(*pool_, participants(collectives));
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    for (const auto& participant : participants(collectives))
        default_context(participant.device_index())->end_graph_warmup(participant.stream());
#endif
}

std::vector<MfqCudaStream> BatchDecodeGraphCache::participants(
        const ModelParallelCollectiveRuntime& collectives) const {
    std::vector<MfqCudaStream> result;
    for (const auto& candidate : cuda_graph_participant_streams(compute_streams_, collectives)) {
        if (std::none_of(result.begin(), result.end(), [&](const auto& stream) {
                return candidate.device_index() == stream.device_index() && candidate.stream() == stream.stream();
            }))
            result.push_back(candidate);
    }
    return result;
}

void BatchDecodeGraphCache::clear() {
    if (!entries_.empty()) {
        MfqCudaGuard guard(stream_.device_index());
        (void)cudaStreamSynchronize(mfq_get_current_cuda_stream(stream_.device_index()).stream());
        for (const auto& stream : compute_streams_) {
            MfqCudaGuard device_guard(stream.device_index());
            (void)cudaStreamSynchronize(stream.stream());
        }
    }
    entries_.clear();
    prepared_outputs_.clear();
    pool_.reset();
    compute_streams_.clear();
    external_pool_ = false;
}

bool BatchDecodeGraphCache::has_graphs() const {
    return std::any_of(entries_.begin(), entries_.end(), [](const auto& value) {
        return value.second->valid;
    });
}

void BatchDecodeGraphCache::captured(Entry& entry, double milliseconds) {
    ++captures;
    capture_ms += milliseconds;
    entry.fresh_capture = true;
    entry.valid = true;
}

void BatchDecodeGraphCache::replayed(Entry& entry) {
    if (!entry.fresh_capture)
        ++bucket_reuses;
    entry.fresh_capture = false;
    ++replays;
}

void mask_batch_routes(CudaExecutionContext& execution, std::vector<tb::Tensor>& selected) {
    if (!execution.decode_row_mask.defined())
        return;
    selected[0] = selected[0].to(tb::kInt32).contiguous();
    selected[1] = selected[1].to(tb::kFloat32).contiguous();
    MFQ_RUNTIME_CHECK(selected[0].dim() == 2 && selected[0].sizes() == selected[1].sizes(),
                      "invalid masked decode routes");
    auto mask = tensor_to_cuda_device(execution.model_parallel_collectives,
                                      execution.decode_row_mask, selected[0].get_device());
    mask_decode_routes(selected[0], selected[1], mask, execution.decode_row_offset);
}

BatchDecodeCaptureScope::BatchDecodeCaptureScope(MfqCudaGraph& graph,
        const std::vector<MfqCudaStream>& participants) : graph_(graph) {
    const auto primary = mfq_get_current_cuda_stream();
    participants_.reserve(participants.size());
    try {
        for (const auto& stream : participants) {
            if (stream.device_index() == primary.device_index() && stream.stream() == primary.stream())
                continue;
            default_context(stream.device_index())->begin_graph_capture(stream.stream());
            participants_.push_back(stream);
        }
        graph_.capture_begin_shared();
    } catch (...) {
        for (const auto& stream : participants_)
            default_context(stream.device_index())->end_graph_capture(stream.stream());
        throw;
    }
}

BatchDecodeCaptureScope::~BatchDecodeCaptureScope() {
    if (!finished_)
        graph_.reset();
    for (const auto& stream : participants_)
        default_context(stream.device_index())->end_graph_capture(stream.stream());
}

void BatchDecodeCaptureScope::finish() {
    graph_.capture_end();
    finished_ = true;
}

BatchDecodeMaskScope::BatchDecodeMaskScope(CudaExecutionContext& execution, tb::Tensor mask, int64_t offset)
    : execution_(execution), previous_(execution.decode_row_mask), offset_(execution.decode_row_offset) {
    execution_.decode_row_mask = std::move(mask);
    execution_.decode_row_offset = offset;
}

BatchDecodeMaskScope::~BatchDecodeMaskScope() {
    execution_.decode_row_mask = std::move(previous_);
    execution_.decode_row_offset = offset_;
}

} // namespace mfq::cuda
