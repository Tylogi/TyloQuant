#include "nint_row_pipeline.h"
#include "mfq/nint_row_cache.h"
#include "mfq/mapped_prefetch_scope.h"
#include "mfq_hip_quant_ops.h"
#include "mfq/host_parallel.h"
#include "mfq_hip_context.h"
#include <atomic>
#include <future>
#include <map>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <optional>
#include <thread>
namespace tb=mfq_tensor_backend;
namespace {
using RowClock=std::chrono::steady_clock;
uint64_t row_elapsed(RowClock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(RowClock::now()-start).count();
}
struct GatherStats {
    uint64_t queue_ns=0,gather_ns=0,read_ns=0,source_bytes=0,misses=0;
};
struct Gathered : GatherStats {
    mfq::NintRowBatch batch;std::vector<int64_t> inverse;
    uint64_t serial=0;
};
int64_t shape_rows(const std::vector<int64_t>& shape) {
    int64_t n=1;
    for(auto x:shape) {
        if(x<0 || (x && n>INT64_MAX/x))throw std::overflow_error("PLE output shape overflow");
        n*=x;
    }
    return n;
}
// Sources share immutable encoded rows, with source identity in every key.
std::shared_ptr<mfq::NintRowCache> row_cache() {
    static auto cache=std::make_shared<mfq::NintRowCache>(1u<<20);return cache;
}
}
struct NintRowStage::Impl {
    uint64_t source;
    int64_t rows;
    int width;
    std::size_t payload_bytes,host_bytes;
    bool mapped;
    bool profile=false;
    uint64_t uploads=0,gathers=0,last_gather=0,queue_ns=0,gather_ns=0,read_ns=0,source_bytes=0,misses=0;
    uint64_t future_ns=0,pack_ns=0,copy_api_ns=0;
    std::vector<int64_t> shape;
    mfq::cuda::HostBuffer host;
    tb::Tensor packet,packed,descriptors,inverse,flat,shaped;
    Impl(uint64_t identity,int64_t n,int w,std::size_t capacity,
        std::vector<int64_t> dimensions,const tb::Device& device,bool use_mapped)
        :source(identity),rows(n),width(w),payload_bytes(capacity),host_bytes(capacity+n*32),
         mapped(use_mapped),shape(std::move(dimensions)),host(host_bytes,use_mapped) {
        const auto* trace=std::getenv("MFQ_TRACE_PLE_TIMINGS");profile=trace && trace[0]=='1';
        const MfqCudaGuard guard(device);
        auto options=tb::TensorOptions().device(device);
        std::vector<int32_t> initial(static_cast<std::size_t>(n)*6,0);
        for(int64_t row=0;row<n;++row)initial[row*6+4]=w;
        const auto* packed_option=std::getenv("MFQ_PLE_PACKED_UPLOAD");
        const bool packed_upload=!packed_option || packed_option[0]!='0';
        if(mapped) {
            // The stage owns this locked host allocation until the captured
            // window and all of its readers have finished. Host publication
            // precedes the GPU wait/decode; the next token cannot overwrite it
            // until DecodeWindow::run has waited for the previous graph.
            uint8_t* address=nullptr;
            MFQ_CUDA_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&address),host.data(),0));
            std::memset(host.data(),0,host_bytes);
            std::memcpy(static_cast<uint8_t*>(host.data())+capacity,initial.data(),initial.size()*sizeof(int32_t));
            packed=tb::from_blob(address,{static_cast<int64_t>(capacity)},options.dtype(tb::kUInt8));
            descriptors=tb::from_blob(address+capacity,{n,6},options.dtype(tb::kInt32));
            inverse=tb::from_blob(address+capacity+n*24,{n},options.dtype(tb::kInt64));
        }else if(packed_upload) {
            // One stable packet contains the payload, row descriptors and
            // inverse map. All views remain alive through the stage owner.
            std::memset(host.data(),0,host_bytes);
            std::memcpy(static_cast<uint8_t*>(host.data())+capacity,initial.data(),initial.size()*sizeof(int32_t));
            packet=tb::empty({static_cast<int64_t>(host_bytes)},options.dtype(tb::kUInt8));
            auto* address=packet.data_ptr<uint8_t>();
            packed=tb::from_blob(address,{static_cast<int64_t>(capacity)},options.dtype(tb::kUInt8));
            descriptors=tb::from_blob(address+capacity,{n,6},options.dtype(tb::kInt32));
            inverse=tb::from_blob(address+capacity+n*24,{n},options.dtype(tb::kInt64));
            MFQ_CUDA_CHECK(hipMemcpyAsync(address,host.data(),host_bytes,hipMemcpyHostToDevice,mfq_current_cuda_stream()));
        }else {
            packed=tb::zeros({static_cast<int64_t>(capacity)},options.dtype(tb::kUInt8));
            descriptors=tb::tensor(initial).reshape({n,6}).to(device);
            inverse=tb::zeros({n},options.dtype(tb::kInt64));
        }
        flat=tb::zeros({n,w},options.dtype(tb::kFloat16));
        auto out_shape=shape;out_shape.push_back(w);shaped=flat.reshape(out_shape);
        mfq::cuda::charge_tensor_host_bytes(host_bytes);
    }
    ~Impl(){
        if(profile)std::fprintf(stderr,"ple_rows_host mapped_rows=%d uploads=%llu gathers=%llu queue_ns=%llu gather_ns=%llu read_ns=%llu source_bytes=%llu misses=%llu future_ns=%llu pack_ns=%llu copy_api_ns=%llu\n",
            int(mapped),static_cast<unsigned long long>(uploads),static_cast<unsigned long long>(gathers),
            static_cast<unsigned long long>(queue_ns),static_cast<unsigned long long>(gather_ns),static_cast<unsigned long long>(read_ns),
            static_cast<unsigned long long>(source_bytes),static_cast<unsigned long long>(misses),
            static_cast<unsigned long long>(future_ns),static_cast<unsigned long long>(pack_ns),static_cast<unsigned long long>(copy_api_ns));
        mfq::cuda::tensor_host_bytes.fetch_sub(host_bytes);
    }
};
NintRowStage::NintRowStage(std::unique_ptr<Impl> impl):impl_(std::move(impl)){}
NintRowStage::~NintRowStage()=default;
tb::Tensor NintRowStage::output() const {return impl_->shaped;}
bool NintRowStage::mapped_rows() const {return impl_->mapped;}
void NintRowStage::decode() {
    auto& s=*impl_;nint_selected_rows_into_cuda(s.packed,s.descriptors,s.inverse,s.flat);
}
struct NintRowPipeline::Impl {
    std::vector<std::shared_ptr<mfq::NintRows>> tables;
    std::shared_ptr<mfq::NintRowCache> cache=row_cache();
    uint64_t source_id=0;int64_t rows=0;int width=0;
    bool profile=false,serial_small=false;uint64_t serial=0;
    std::vector<int64_t> current_ids,current_shape;
    bool early_prefix=false;unsigned reported_prefix_sizes=0;
    std::future<Gathered> pending;std::optional<Gathered> ready;
    bool reuse_worker=true,active_read_pool=true,stopping=false;
    mfq::HostParallelPool read_pool{{},false};
    std::mutex work_mutex;
    std::condition_variable work_ready;
    std::packaged_task<Gathered()> work;
    std::thread worker;
    explicit Impl(std::vector<std::shared_ptr<mfq::NintRows>> data,std::shared_ptr<mfq::NintRowCache> rows_cache):tables(std::move(data)) {
        if(rows_cache)cache=std::move(rows_cache);
        const auto* trace=std::getenv("MFQ_TRACE_PLE_TIMINGS");profile=trace && trace[0]=='1';
        const auto* serial_setting=std::getenv("MFQ_PLE_SERIAL_SMALL_GATHER");
        serial_small=serial_setting && serial_setting[0]=='1';
        const auto* worker_setting=std::getenv("MFQ_PLE_REUSE_WORKER");
        reuse_worker=!worker_setting || worker_setting[0]!='0';
        const auto* pool_setting=std::getenv("MFQ_PLE_ACTIVE_READ_POOL");
        active_read_pool=!pool_setting || pool_setting[0]!='0';
        static std::atomic<uint64_t> next{1};source_id=next.fetch_add(1);
        if(tables.empty() || !tables.front())throw std::invalid_argument("MFQ PLE needs row sources");
        rows=tables.front()->rows();width=tables.front()->width();
        for(const auto& table:tables)if(!table || table->rows()!=rows || table->width()!=width)
            throw std::invalid_argument("MFQ PLE shard dimensions differ");
        if(rows*static_cast<int64_t>(tables.size())>int64_t(UINT32_MAX))throw std::overflow_error("PLE logical row cache key overflow");
        if(source_id>UINT32_MAX)throw std::overflow_error("PLE source cache identity overflow");
    }
    ~Impl() {
        // A queued read owns references to this source and its row cache.
        // Complete it before destroying either, including on failed requests.
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            stopping=true;
        }
        work_ready.notify_one();
        if(worker.joinable())worker.join();
        if(pending.valid())pending.wait();
    }
    std::future<Gathered> submit(std::vector<int64_t> ids,uint64_t sequence,RowClock::time_point issued,GatherStats carried) {
        if(!reuse_worker)return std::async(std::launch::async,
            [this,ids=std::move(ids),sequence,issued,carried]{return gather(ids,sequence,issued,carried);});
        std::packaged_task<Gathered()> next([this,ids=std::move(ids),sequence,issued,carried]{return gather(ids,sequence,issued,carried);});
        auto result=next.get_future();
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            // issue() consumes the preceding future before submitting again.
            if(work.valid())throw std::logic_error("PLE source has an unconsumed row request");
            if(!worker.joinable())worker=std::thread([this] {
                for(;;) {
                    std::packaged_task<Gathered()> next;
                    {
                        std::unique_lock<std::mutex> lock(work_mutex);
                        work_ready.wait(lock,[this]{return stopping || work.valid();});
                        if(!work.valid())return;
                        next=std::move(work);
                    }
                    next(); // packaged_task propagates read failures to get().
                }
            });
            work=std::move(next);
        }
        work_ready.notify_one();
        return result;
    }
    Gathered gather(const std::vector<int64_t>& ids,uint64_t sequence,RowClock::time_point issued,GatherStats carried) {
        Gathered out;static_cast<GatherStats&>(out)=carried;out.serial=sequence;
        const auto started=profile?RowClock::now():RowClock::time_point{};
        if(profile)out.queue_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(started-issued).count();
        std::map<int64_t,std::shared_ptr<const mfq::NintRowBatch>> selected;
        std::vector<std::vector<int64_t>> misses(tables.size());
        for(const auto id:ids) {
            if(id<0 || id/rows>=static_cast<int64_t>(tables.size()))throw std::out_of_range("PLE logical row ID");
            if(selected.contains(id))continue;
            auto value=cache->find((source_id<<32)|static_cast<uint32_t>(id));selected.emplace(id,std::move(value));
            if(!selected.at(id))misses[id/rows].push_back(id%rows);
        }
        std::vector<mfq::NintRowBatch> fetched(tables.size());
        const auto read_started=profile?RowClock::now():RowClock::time_point{};
        std::size_t missing_rows=0;
        for(const auto& shard:misses)missing_rows+=shard.size();
        const int threads=serial_small && missing_rows<=32?1:mfq_get_num_threads();
        const auto fetch=[&](std::size_t shard) {
            mfq::MappedPrefetchScope prefetch(misses[shard].size()>128,true);
            tables[shard]->append_rows(misses[shard].data(),misses[shard].size(),fetched[shard],1);
        };
        if(active_read_pool) {
            // Decode normally misses only a few of the 128 tables. Give each
            // active shard its own work item instead of partitioning empty
            // shards too. Finish when the reads finish; sleeping workers do
            // not own the callback and need not acknowledge an empty slice.
            std::vector<std::size_t> active;
            for(std::size_t shard=0;shard<misses.size();++shard)
                if(!misses[shard].empty())active.push_back(shard);
            read_pool.run(0,static_cast<int64_t>(active.size()),1,threads,[&](int64_t first,int64_t last) {
                for(auto i=first;i<last;++i) {
                    const auto shard=active[i];
                    fetch(shard);
                }
            });
        }else {
            mfq::host_parallel_for(0,static_cast<int64_t>(tables.size()),1,threads,[&](int64_t first,int64_t last) {
                for(auto shard=first;shard<last;++shard)if(!misses[shard].empty())
                    fetch(shard);
            });
        }
        if(profile) {
            out.read_ns+=row_elapsed(read_started);
            for(const auto& batch:fetched){out.source_bytes+=batch.source_bytes_read();out.misses+=batch.rows();}
        }
        for(std::size_t shard=0;shard<tables.size();++shard)for(std::size_t i=0;i<misses[shard].size();++i) {
            auto value=std::make_shared<mfq::NintRowBatch>();
            fetched[shard].copy_row(i,*value);
            const auto id=int64_t(shard)*rows+misses[shard][i];selected.at(id)=value;
            cache->insert((source_id<<32)|static_cast<uint32_t>(id),std::move(value));
        }
        std::map<int64_t,int64_t> locations;
        for(const auto& [id,value]:selected){locations.emplace(id,out.batch.rows());out.batch.append_batch(*value);}
        for(auto id:ids)out.inverse.push_back(locations.at(id));
        if(profile)out.gather_ns+=row_elapsed(started);
        return out;
    }
};
NintRowPipeline::NintRowPipeline(std::vector<std::shared_ptr<mfq::NintRows>> data,
    std::shared_ptr<mfq::NintRowCache> cache):impl_(std::make_unique<Impl>(std::move(data),std::move(cache))){}
NintRowPipeline::~NintRowPipeline()=default;
void NintRowPipeline::issue(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape,bool early_prefix) {
    auto& s=*impl_;if(s.current_ids==ids && s.current_shape==shape && (s.ready || s.pending.valid()))return;
    if(s.pending.valid())s.ready.emplace(s.pending.get());
    // Attribute the hint's actual I/O to the window that consumes it; otherwise
    // moving reads earlier would incorrectly appear to reduce source bytes.
    const GatherStats carried=s.profile && s.early_prefix && s.ready?
        static_cast<const GatherStats&>(*s.ready):GatherStats{};
    // Only the immutable rows are prefetched. Recompute the complete window
    // from its committed ngram context, then reuse matching cache entries.
    if(s.early_prefix && s.ready && s.current_shape.size()==3 && shape.size()==3 &&
       s.current_shape[0]==1 && s.current_shape[1]>=1 && shape[0]==1 &&
       shape[1]>s.current_shape[1] && shape[1]<=8 && shape[2]==s.current_shape[2] &&
       s.current_ids.size()<=ids.size() &&
       std::equal(s.current_ids.begin(),s.current_ids.end(),ids.begin())) {
        const auto bit=1u<<unsigned(shape[1]);
        if(!(s.reported_prefix_sizes&bit)) {
            s.reported_prefix_sizes|=bit;
            std::fprintf(stderr,"ple_prefix_prefetch_reuse enabled=1 tokens=%lld prefix_tokens=%lld rows=%zu\n",
                static_cast<long long>(shape[1]),static_cast<long long>(s.current_shape[1]),s.current_ids.size());
        }
    }
    s.current_ids=ids;s.current_shape=shape;s.ready.reset();
    s.early_prefix=early_prefix;
    const auto issued=s.profile?RowClock::now():RowClock::time_point{};
    const auto sequence=++s.serial;
    s.pending=s.submit(ids,sequence,issued,carried);
}
tb::Tensor NintRowPipeline::collect(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape,const tb::Device& device) {
    auto& s=*impl_;
    const auto started=s.profile?RowClock::now():RowClock::time_point{};
    issue(ids,shape);
    if(s.pending.valid())s.ready.emplace(s.pending.get());
    s.early_prefix=false;
    const auto& result=*s.ready;auto out_shape=shape;out_shape.push_back(s.width);
    const auto n=shape_rows(shape);
    if(n!=static_cast<int64_t>(ids.size()) || !device.is_cuda())throw std::invalid_argument("PLE pipeline input shape/device mismatch");
    const MfqCudaGuard guard(device);
    if(ids.empty())return tb::empty(out_shape,tb::TensorOptions().device(device).dtype(tb::kFloat16));
    result.batch.validate();auto packed=tb::tensor(result.batch.packed()).to(device);
    std::vector<int32_t> words(result.batch.descriptors().size());std::memcpy(words.data(),result.batch.descriptors().data(),words.size()*4);
    auto descriptors=tb::tensor(words).reshape({static_cast<int64_t>(result.batch.rows()),6}).to(device);
    if(s.profile && ids.size()>128)std::fprintf(stderr,
        "ple_prefill_rows rows=%zu unique_rows=%zu misses=%llu source_bytes=%llu packed_bytes=%zu queue_ms=%.3f gather_ms=%.3f read_ms=%.3f collect_ms=%.3f\n",
        ids.size(),result.batch.rows(),static_cast<unsigned long long>(result.misses),
        static_cast<unsigned long long>(result.source_bytes),result.batch.packed_nbytes(),
        result.queue_ns/1e6,result.gather_ns/1e6,result.read_ns/1e6,row_elapsed(started)/1e6);
    return nint_selected_rows_cuda(packed,descriptors,s.width).index_select(0,tb::tensor(result.inverse).to(device)).reshape(out_shape);
}
std::shared_ptr<NintRowStage> NintRowPipeline::make_stage(const std::vector<int64_t>& shape,const tb::Device& device) {
    const auto& s=*impl_;const auto n=shape_rows(shape);
    if(!device.is_cuda() || n<=0 || n>INT_MAX/6 || uint64_t(n)*s.width>UINT32_MAX)
        throw std::invalid_argument("PLE graph stage shape/device exceeds bounds");
    std::size_t row_bytes=0;
    for(const auto& table:s.tables)row_bytes=std::max(row_bytes,table->selected_row_nbytes_bound());
    if(row_bytes>std::size_t(INT_MAX)/n)throw std::overflow_error("PLE graph payload exceeds bounds");
    const auto capacity=(row_bytes*n+7)&~std::size_t(7);
    const auto* setting=std::getenv("MFQ_PLE_MAPPED_ROWS");
    // Small decode packets need only a few PCIe cache lines. Reading the
    // locked packet in the existing decoder avoids a separate DMA submission
    // and host completion callback. Large windows keep the device upload.
    const bool small_shape=shape.size()==1 ||
        (shape.size()==3 && shape[0]==1 && shape[1]<=6);
    const bool small_decode=small_shape && capacity+std::size_t(n)*32<=16u*1024;
    const bool mapped=setting?setting[0]=='1':small_decode;
    return std::shared_ptr<NintRowStage>(new NintRowStage(std::make_unique<NintRowStage::Impl>(
        s.source_id,n,s.width,capacity,shape,device,mapped)));
}
void NintRowPipeline::upload(NintRowStage& stage,const std::vector<int64_t>& ids,const std::vector<int64_t>& shape) {
    auto& s=*impl_;auto& target=*stage.impl_;
    if(target.source!=s.source_id || target.shape!=shape || int64_t(ids.size())!=target.rows)
        throw std::invalid_argument("PLE graph upload differs from its prepared stage");
    auto point=target.profile?RowClock::now():RowClock::time_point{};
    issue(ids,shape);
    if(s.pending.valid())s.ready.emplace(s.pending.get());
    s.early_prefix=false;
    if(target.profile){target.future_ns+=row_elapsed(point);point=RowClock::now();++target.uploads;}
    const auto& result=*s.ready;result.batch.validate();
    if(target.profile && target.last_gather!=result.serial) {
        target.last_gather=result.serial;++target.gathers;
        target.queue_ns+=result.queue_ns;target.gather_ns+=result.gather_ns;target.read_ns+=result.read_ns;
        target.source_bytes+=result.source_bytes;target.misses+=result.misses;
    }
    if(result.batch.packed_nbytes()>target.payload_bytes || result.batch.rows()>std::size_t(target.rows) ||
       result.inverse.size()!=ids.size())throw std::overflow_error("PLE graph upload exceeds its prepared stage");
    auto* host=static_cast<uint8_t*>(target.host.data());
    const auto descriptor_bytes=result.batch.descriptors().size()*sizeof(uint32_t);
    auto* descriptors=host+target.payload_bytes;
    auto* inverse=descriptors+target.rows*6*sizeof(uint32_t);
    std::memcpy(host,result.batch.packed().data(),result.batch.packed_nbytes());
    std::memcpy(descriptors,result.batch.descriptors().data(),descriptor_bytes);
    std::memcpy(inverse,result.inverse.data(),result.inverse.size()*sizeof(int64_t));
    if(target.profile){target.pack_ns+=row_elapsed(point);point=RowClock::now();}
    if(target.mapped)return;
    const MfqCudaGuard guard(target.packed.device());const auto stream=mfq_current_cuda_stream();
    if(target.packet.defined()) {
        MFQ_CUDA_CHECK(hipMemcpyAsync(target.packet.data_ptr(),host,target.host_bytes,hipMemcpyHostToDevice,stream));
    }else {
        MFQ_CUDA_CHECK(hipMemcpyAsync(target.packed.data_ptr(),host,result.batch.packed_nbytes(),hipMemcpyHostToDevice,stream));
        MFQ_CUDA_CHECK(hipMemcpyAsync(target.descriptors.data_ptr(),descriptors,descriptor_bytes,hipMemcpyHostToDevice,stream));
        MFQ_CUDA_CHECK(hipMemcpyAsync(target.inverse.data_ptr(),inverse,result.inverse.size()*sizeof(int64_t),hipMemcpyHostToDevice,stream));
    }
    if(target.profile)target.copy_api_ns+=row_elapsed(point);
}
