#include "engram.h"
#include "mfq_cuda_context.h"
#include "mfq/host_parallel.h"
#include "mfq/mapped_prefetch_scope.h"
#include "mfq/nint_row_cache.h"
#include "runtime/nint_row_pipeline.h"

#include <condition_variable>
#include <future>
#include <map>
#include <thread>

namespace mfq::cuda::deepseek_v41_runtime {
namespace tb = mfq_tensor_backend;
namespace {
struct RowPacket {
    std::vector<std::uint8_t> packed;
    std::vector<std::int64_t> inverse;
    std::size_t unique = 0;
};

// The pinned packet and completion event follow the PLE row stage lifetime:
// CPU publication must wait for the preceding DMA to finish reading it.
struct RowStage {
    mfq::cuda::HostBuffer host;
    tb::Tensor packet;
    mfq::cuda::Event uploaded;
    bool pending = false;
    RowStage(std::size_t bytes, const tb::Device& device)
        : host(bytes), packet(tb::empty({static_cast<int64_t>(bytes)},
              tb::TensorOptions().device(device).dtype(tb::kUInt8))) {
        mfq::cuda::charge_tensor_host_bytes(bytes);
    }
    ~RowStage() {
        if (pending) {
            try { uploaded.synchronize(); } catch (...) {}
        }
        mfq::cuda::tensor_host_bytes.fetch_sub(host.size());
    }
};
}

struct EngramTable::Impl {
    const mfq::ModelSource& source;
    std::string name;
    std::uint64_t rows, value_bytes;
    int width;
    std::size_t capacity, row_bytes;
    mfq::ModelSource::TensorBatchReader read_batch;
    struct CacheEntry {
        std::vector<std::uint8_t> packed;
        std::list<std::int64_t>::iterator recency;
    };
    std::list<std::int64_t> recency;
    std::unordered_map<std::int64_t, CacheEntry> cache;
    mfq::HostParallelPool read_pool{{}, false};
    std::vector<std::int64_t> current_ids;
    std::future<RowPacket> pending;
    std::optional<RowPacket> ready;
    std::mutex mutex;
    std::condition_variable available;
    std::packaged_task<RowPacket()> work;
    std::thread worker;
    bool stopping = false;
    std::unique_ptr<RowStage> stage;
    std::shared_ptr<NintRowPipeline> nint_pipeline;
    std::shared_ptr<NintRowStage> nint_stage;
    std::unique_ptr<mfq::cuda::Event> nint_done;
    bool nint_pending=false;
    std::size_t nint_stage_rows=0;
    std::shared_ptr<TqRowTable> tq_table;

    Impl(const mfq::ModelSource& data, std::string tensor, std::uint64_t count,
         int columns, std::uint64_t values, std::size_t cache_rows,std::shared_ptr<mfq::NintRows> nint)
        : source(data), name(std::move(tensor)), rows(count), value_bytes(values),
          width(columns), capacity(cache_rows), row_bytes(columns + columns / 32) {
        if(is_tq_row_table(source,name)) {
            tq_table=std::make_shared<TqRowTable>(source,name);
            if(tq_table->rows()!=rows || tq_table->width()!=width)
                throw std::invalid_argument("TQ Engram table shape differs");
            return;
        }
        if(nint) {
            nint_pipeline=std::make_shared<NintRowPipeline>(std::vector{std::move(nint)},
                std::make_shared<mfq::NintRowCache>(cache_rows));
            return;
        }
        read_batch = source.mapped_tensor_batch_reader(name);
        if (!read_batch) read_batch = source.tensor_batch_reader(name);
    }
    ~Impl() {
        if(nint_pending)try {nint_done->synchronize();}catch(...) {}
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        available.notify_one();
        if (worker.joinable()) worker.join();
    }

    RowPacket gather_rows(const std::vector<std::int64_t>& ids) {
        RowPacket out;
        std::map<std::int64_t, std::size_t> locations;
        for (auto id : ids) locations.emplace(id, 0);
        out.unique = locations.size();
        if (out.unique > std::numeric_limits<std::size_t>::max() / row_bytes)
            throw std::overflow_error("DeepSeek-V4.1 Engram packet is too large");
        out.packed.resize(out.unique * row_bytes);
        std::vector<std::pair<std::int64_t, std::size_t>> misses;
        std::size_t index = 0;
        for (auto& [id, location] : locations) {
            location = index++;
            const auto found = cache.find(id);
            if (found == cache.end()) misses.emplace_back(id, location);
            else {
                std::memcpy(out.packed.data() + location * row_bytes,
                            found->second.packed.data(), row_bytes);
                recency.splice(recency.begin(), recency, found->second.recency);
            }
        }
        // Values and scales occupy separate table streams. Submit all selected
        // pages together, exactly as the PLE batch reader does for NINT streams.
        std::vector<mfq::ReadSpan> spans;
        spans.reserve(misses.size() * 2);
        for (int scale = 0; scale < 2; ++scale) for (auto [id, location] : misses) {
            const auto count = scale ? width / 32 : width;
            const auto offset = engram_detail::kMxHeaderBytes +
                (scale ? value_bytes : 0) + static_cast<std::uint64_t>(id) * count;
            auto* destination = out.packed.data() + location * row_bytes + (scale ? width : 0);
            spans.push_back({offset, reinterpret_cast<std::byte*>(destination), std::size_t(count)});
        }
        if (!spans.empty()) {
            if (read_batch) {
                mfq::MappedPrefetchScope prefetch(misses.size() > 128);
                read_batch(spans);
            }
            else read_pool.run(0, spans.size(), 32,
                source.supports_parallel_tensor_reads() ? 8 : 1, [&](int64_t first, int64_t last) {
                    for (auto i = first; i < last; ++i) {
                        const auto& span = spans[i];
                        source.read_range_into(name, span.offset, span.destination, span.size);
                    }
                });
        }
        // Validate once on insertion, retaining the original malformed-row
        // errors without inspecting or expanding cached values on every token.
        for (auto [id, location] : misses) {
            const auto* packed = out.packed.data() + location * row_bytes;
            for (int col = 0; col < width; ++col) if ((packed[col] & 127) == 127)
                throw std::runtime_error("DeepSeek-V4.1 Engram row contains an E4M3 NaN");
            for (int col = 0; col < width / 32; ++col) if (packed[width + col] == 255)
                throw std::runtime_error("DeepSeek-V4.1 Engram row contains an invalid E8M0 scale");
            if (!capacity) continue;
            if (cache.size() >= capacity) {
                cache.erase(recency.back());
                recency.pop_back();
            }
            recency.push_front(id);
            cache.emplace(id, CacheEntry{std::vector<std::uint8_t>(packed, packed + row_bytes),
                                          recency.begin()});
        }
        out.inverse.reserve(ids.size());
        for (auto id : ids) out.inverse.push_back(locations.at(id));
        return out;
    }

    void issue(const std::int64_t* ids, std::size_t count) {
        if (!ids || !count) throw std::invalid_argument("DeepSeek-V4.1 Engram lookup is empty");
        for (std::size_t i = 0; i < count; ++i)
            if (ids[i] < 0 || static_cast<std::uint64_t>(ids[i]) >= rows)
                throw std::out_of_range("DeepSeek-V4.1 Engram lookup row is outside the table");
        if (current_ids.size() == count && std::equal(current_ids.begin(), current_ids.end(), ids) &&
            (pending.valid() || ready)) return;
        if (pending.valid()) pending.get();
        ready.reset();
        current_ids.assign(ids, ids + count);
        std::packaged_task<RowPacket()> next([this, selected = current_ids] { return gather_rows(selected); });
        pending = next.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (work.valid()) throw std::logic_error("Engram row worker has an unconsumed request");
            if (!worker.joinable()) worker = std::thread([this] {
                for (;;) {
                    std::packaged_task<RowPacket()> next;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        available.wait(lock, [this] { return stopping || work.valid(); });
                        if (!work.valid()) return;
                        next = std::move(work);
                    }
                    next();
                }
            });
            work = std::move(next);
        }
        available.notify_one();
    }
};

EngramTable::EngramTable(const mfq::ModelSource& source, std::string name, std::uint64_t,
    std::int64_t rows, int width, std::uint64_t values, std::size_t capacity,std::shared_ptr<mfq::NintRows> nint)
    : impl_(std::make_unique<Impl>(source, std::move(name), rows, width, values, capacity,std::move(nint))) {}
EngramTable::~EngramTable() = default;

void EngramTable::prefetch(const std::int64_t* rows, std::size_t count) const {
    if(impl_->tq_table)return;
    if(impl_->nint_pipeline) {
        if(!rows || !count)throw std::invalid_argument("DeepSeek-V4.1 Engram lookup is empty");
        for(std::size_t i=0;i<count;++i)if(rows[i]<0 || static_cast<std::uint64_t>(rows[i])>=impl_->rows)
            throw std::out_of_range("DeepSeek-V4.1 Engram lookup row is outside the table");
        impl_->nint_pipeline->issue(std::vector<int64_t>(rows,rows+count),{static_cast<int64_t>(count)});
        return;
    }
    impl_->issue(rows, count);
}

EngramTensor EngramTable::gather(const std::int64_t* rows, int batch, int tokens, int columns,
                                const tb::Device& device) const {
    if (batch <= 0 || tokens <= 0 || columns <= 0 || !device.is_cuda())
        throw std::invalid_argument("DeepSeek-V4.1 Engram lookup geometry disagrees");
    auto& s = *impl_;
    const auto count = std::size_t(batch) * tokens * columns;
    if(s.tq_table)return s.tq_table->lookup(rows,count,device)
        .reshape({batch,tokens,static_cast<int64_t>(columns)*s.width});
    if(s.nint_pipeline) {
        if(!rows)throw std::invalid_argument("DeepSeek-V4.1 Engram row IDs are missing");
        const MfqCudaGuard guard(device);
        if(s.nint_pending) {s.nint_done->synchronize();s.nint_pending=false;}
        const std::vector<int64_t> shape{static_cast<int64_t>(count)};
        if(!s.nint_stage || s.nint_stage_rows!=count || s.nint_stage->output().device()!=device) {
            s.nint_stage=s.nint_pipeline->make_stage(shape,device);s.nint_stage_rows=count;
        }
        if(!s.nint_done)s.nint_done=std::make_unique<mfq::cuda::Event>(cudaEventDisableTiming);
        s.nint_pipeline->upload(*s.nint_stage,std::vector<int64_t>(rows,rows+count),shape);
        s.nint_stage->decode();
        s.nint_done->record(mfq_current_cuda_stream());s.nint_pending=true;
        return s.nint_stage->output().reshape({batch,tokens,static_cast<int64_t>(columns)*s.width});
    }
    s.issue(rows, count);
    if (s.pending.valid()) s.ready.emplace(s.pending.get());
    const auto& selected = *s.ready;
    const auto inverse_offset = (selected.packed.size() + 7) & ~std::size_t(7);
    if (count > (std::numeric_limits<std::size_t>::max() - inverse_offset) / sizeof(int64_t))
        throw std::overflow_error("DeepSeek-V4.1 Engram inverse packet is too large");
    const auto bytes = inverse_offset + count * sizeof(int64_t);
    const MfqCudaGuard guard(device);
    if (s.stage && s.stage->pending) { s.stage->uploaded.synchronize(); s.stage->pending = false; }
    if (!s.stage || s.stage->host.size() < bytes || s.stage->packet.device() != device)
        s.stage = std::make_unique<RowStage>(bytes, device);
    auto& stage = *s.stage;
    auto* host = static_cast<uint8_t*>(stage.host.data());
    std::memcpy(host, selected.packed.data(), selected.packed.size());
    std::memcpy(host + inverse_offset, selected.inverse.data(), count * sizeof(int64_t));
    MFQ_CUDA_CHECK(cudaMemcpyAsync(stage.packet.data_ptr(), host, bytes, cudaMemcpyHostToDevice,
                                   mfq_current_cuda_stream()));
    auto options = stage.packet.options();
    auto* data = stage.packet.data_ptr<std::uint8_t>();
    auto packed = tb::from_blob(data, {static_cast<int64_t>(selected.unique),
                                      static_cast<int64_t>(s.row_bytes)}, options);
    auto inverse = tb::from_blob(data + inverse_offset, {static_cast<int64_t>(count)},
                                 options.dtype(tb::kInt64));
    auto output = deepseek_v41_engram_rows_cuda(packed, inverse, s.width)
        .reshape({batch, tokens, static_cast<int64_t>(columns) * s.width});
    stage.uploaded.record(mfq_current_cuda_stream());
    stage.pending = true;
    return output;
}

std::shared_ptr<EngramTable> EngramTable::load(const mfq::ModelSource &model, const std::string &name,
                                             std::int64_t expected_rows, int expected_width,
                                             std::size_t cache_capacity_rows) {
        if(is_tq_row_table(model,name))return std::shared_ptr<EngramTable>(new EngramTable(model,name,0,
            expected_rows,expected_width,0,cache_capacity_rows,{}));
        const auto record = require_tensor(model, name);
        if(record.dtype=="NINT") {
            auto table=load_nint_row_table(model,name,true);
            if(table->rows()!=expected_rows || table->width()!=expected_width)
                throw std::runtime_error("DeepSeek-V4.1 Engram NINT table shape disagrees: "+name);
            std::fprintf(stderr,"deepseek_v41_engram_dispatch format=NINT row_pipeline=ple rows=%lld width=%d\n",
                static_cast<long long>(expected_rows),expected_width);
            return std::shared_ptr<EngramTable>(new EngramTable(model,name,record.nbytes,expected_rows,
                expected_width,0,cache_capacity_rows,std::move(table)));
        }
        if (record.dtype != "MXFP8") {
            throw std::runtime_error("DeepSeek-V4.1 Engram table must use native MXFP8: " + name);
        }
        auto raw_header = model.read(name, 0, engram_detail::kMxHeaderBytes);
        std::vector<std::uint8_t> header(raw_header.size());
        std::memcpy(header.data(), raw_header.data(), header.size());
        if (header.size() < engram_detail::kMxHeaderBytes ||
            std::memcmp(header.data(), "MXT1", 4) != 0 || header[4] != 1 || header[5] != 8 ||
            header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid DeepSeek-V4.1 Engram MXFP8 header: " + name);
        }
        const auto rows = engram_detail::read_u64_at(header, 8);
        const auto columns = engram_detail::read_u64_at(header, 16);
        const auto storage_rows = engram_detail::read_u64_at(header, 24);
        const auto storage_columns = engram_detail::read_u64_at(header, 32);
        const auto scale_rows = engram_detail::read_u64_at(header, 40);
        const auto scale_columns = engram_detail::read_u64_at(header, 48);
        if (rows != static_cast<std::uint64_t>(expected_rows) ||
            columns != static_cast<std::uint64_t>(expected_width) || storage_rows != rows ||
            storage_columns != columns || scale_rows != rows || scale_columns != columns / 32 ||
            columns == 0 || columns % 32) {
            throw std::runtime_error("DeepSeek-V4.1 Engram table is not row-wise MXFP8: " + name);
        }
        const auto value_bytes = engram_detail::checked_product(rows, columns, "value");
        const auto scale_bytes = engram_detail::checked_product(scale_rows, scale_columns, "scale");
        if (value_bytes >
                std::numeric_limits<std::uint64_t>::max() - engram_detail::kMxHeaderBytes ||
            scale_bytes > std::numeric_limits<std::uint64_t>::max() -
                              engram_detail::kMxHeaderBytes - value_bytes ||
            engram_detail::kMxHeaderBytes + value_bytes + scale_bytes != record.nbytes) {
            throw std::runtime_error("DeepSeek-V4.1 Engram MXFP8 payload size disagrees: " + name);
        }
        return std::shared_ptr<EngramTable>(
            new EngramTable(model, name, record.nbytes, static_cast<std::int64_t>(rows),
                            static_cast<int>(columns), value_bytes, cache_capacity_rows));
    }

} // namespace mfq::cuda::deepseek_v41_runtime
