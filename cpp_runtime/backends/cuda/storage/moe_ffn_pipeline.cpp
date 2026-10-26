#include "../runtime/execution_options.h"
#include "moe_ffn_pipeline.h"
#include "mfq/host_copy.h"
#include "moe_cached_source_internal.h"
#include "cpu_projection_rows.h"
#include "mfq/cpu_expert_pool.h"
#include "runtime/graph_registry.h"
#include "runtime/moe_pipeline.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_shared_gate.h"
#include "runtime/decode_window.h"
#include "runtime/dma_copy_batch.h"
#include "runtime/native_dma_batch.h"
#include "mfq_cuda_context.h"
#include "mfq/runtime.h"
#include "mfe_ffn_runtime.h"
#include "quant_linear.h"
#include <cuda.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <atomic>
#include <exception>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <map>
#include <sstream>
#include <stdexcept>

namespace tb=mfq_tensor_backend;
namespace {
using StreamFlagWrite = decltype(&cuStreamWriteValue32);
void* stream_flag_driver_entry(const char* symbol) {
#if CUDA_VERSION >= 12000 && CUDART_VERSION >= 12000
    void* pointer=nullptr;
    cudaDriverEntryPointQueryResult result=cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
    const auto error=cudaGetDriverEntryPointByVersion(symbol,&pointer,12000,cudaEnableDefault,&result);
#else   // the versioned query arrived in CUDA 12.5
    const auto error=cudaGetDriverEntryPoint(symbol,&pointer,cudaEnableDefault,&result);
#endif
    if(error==cudaErrorNotSupported) {cudaGetLastError();return nullptr;}
    MFQ_CUDA_CHECK(error);
    return result==cudaDriverEntryPointSuccess?pointer:nullptr;
#else
    (void)symbol;return nullptr;
#endif
}
StreamFlagWrite stream_flag_writer() {
#if CUDA_VERSION >= 12000 && CUDART_VERSION >= 12000
    static const auto write=reinterpret_cast<StreamFlagWrite>(stream_flag_driver_entry("cuStreamWriteValue32"));
    return write;
#else
    return nullptr;
#endif
}

// Mapped plans/flags have fixed addresses for every replay. Counting their
// allocations supplements the process's physical-memory measurement.
struct Mapped {
    void* host=nullptr;void* device=nullptr;std::size_t bytes=0;
    std::shared_ptr<Mapped> shared_owner;
    explicit Mapped(std::size_t n):bytes(n) {
        const auto allocation=cudaHostAlloc(&host,n,cudaHostAllocMapped);
        if(mfq::cuda::pinned_allocation_rejected(allocation)) {
            (void)cudaGetLastError();
            throw MfqResourceExhausted("CUDA pinned allocation failed: requested="+std::to_string(n)+
                " host_allocated="+std::to_string(mfq::cuda::tensor_host_bytes.load())+
                " error="+cudaGetErrorName(allocation));
        }
        MFQ_CUDA_CHECK(allocation);
        const auto status=cudaHostGetDevicePointer(&device,host,0);
        if(status!=cudaSuccess) {cudaFreeHost(host);host=nullptr;MFQ_CUDA_CHECK(status);}
        mfq::cuda::charge_tensor_host_bytes(n);std::memset(host,0,n);
    }
    Mapped(const Mapped&)=delete;
    Mapped& operator=(const Mapped&)=delete;
    void borrow(std::shared_ptr<Mapped> owner) {
        if(host && !shared_owner){cudaFreeHost(host);mfq::cuda::tensor_host_bytes.fetch_sub(bytes);}
        shared_owner=std::move(owner);host=shared_owner->host;device=shared_owner->device;bytes=shared_owner->bytes;
    }
    ~Mapped(){if(host && !shared_owner){cudaFreeHost(host);mfq::cuda::tensor_host_bytes.fetch_sub(bytes);}}
    template<class T>T* h()const{return static_cast<T*>(host);}
    template<class T>T* d()const{return static_cast<T*>(device);}
};
void publish(uint32_t* flag) {
    std::atomic_thread_fence(std::memory_order_seq_cst);_mm_sfence();*(volatile uint32_t*)flag=1;
}
MixedMoePool fields_pool(MixedMoePool pool,const std::vector<tb::Tensor>& fields,int slots,int output) {
    pool.local_experts=slots;
    if(pool.family==MixedMoeFamily::Nint) {
        auto& w=pool.nint;w.workspaces.clear();w.aligned_q8=false;
        w.q_packed=fields[0];w.row_q_bits=fields[1];w.row_q_bit_offsets=fields[2];
        w.sub_scale=fields[3];w.sub_min=fields[4];w.neuron_scale=fields[5];w.neuron_min=fields[6];
        w.out=int64_t(slots)*output;if(!w.shape.empty())w.shape[0]=w.out;
    } else if(pool.family==MixedMoeFamily::Nvq) {
        auto& w=pool.nvq;w.workspaces.clear();w.indices_packed=fields[0];w.aux_packed=fields[1];
        w.sub_scale_packed=fields[2];w.neuron_scale=fields[3];
        w.out=int64_t(slots)*output;if(!w.shape.empty())w.shape[0]=w.out;
    } else if(pool.family==MixedMoeFamily::Mxfp4) {
        pool.mxfp4.values=fields[0];pool.mxfp4.scales=fields[1];
        pool.mxfp4.out=int64_t(slots)*output;
    } else throw std::runtime_error("MFQ FFN CPU callback lacks this packed format");
    return pool;
}
CpuProjectionRows rows_plan(const MixedMoePool& p,bool up_half=false) {
    if(p.family==MixedMoeFamily::Mxfp4) {
        auto w=p.mxfp4;
        if(up_half) {
            w.out/=2;
            w.values=w.values.narrow(0,w.out,w.out);
            w.scales=w.scales.narrow(0,w.out,w.out);
        }
        return make_cpu_projection_rows(w);
    }
    if(up_half) {
        const auto half=[](const tb::Tensor& field) {
            return field.reshape({-1}).narrow(0,field.numel()/2,field.numel()/2);
        };
        if(p.family==MixedMoeFamily::Nint) {
            auto w=p.nint;w.out/=2;
            w.row_q_bits=half(w.row_q_bits);w.row_q_bit_offsets=half(w.row_q_bit_offsets);
            w.sub_scale=half(w.sub_scale);w.sub_min=half(w.sub_min);
            w.neuron_scale=half(w.neuron_scale);w.neuron_min=half(w.neuron_min);
            return make_cpu_projection_rows(w);
        }
        auto w=p.nvq;w.out/=2;
        w.indices_packed=half(w.indices_packed);w.aux_packed=half(w.aux_packed);
        w.sub_scale_packed=half(w.sub_scale_packed);w.neuron_scale=half(w.neuron_scale);
        return make_cpu_projection_rows(w);
    }
    return p.family==MixedMoeFamily::Nint ? make_cpu_projection_rows(p.nint) : make_cpu_projection_rows(p.nvq);
}

std::function<mfq::MoeCpuCalibration::Observation()> cpu_calibration_work(
        mfq::MoeCpuCostModel::Key key,std::array<MoeHostExpertCache::Lease,3> leases,
        std::vector<float> input,int width,int intermediate,int count,double preparation_ns,bool combined=false,double swiglu_limit=0,bool up_only=false) {
    return [key,leases=std::move(leases),input=std::move(input),width,intermediate,count,preparation_ns,combined,swiglu_limit,up_only]() {
        const auto started=std::chrono::steady_clock::now();
        const std::array<CpuProjectionRows,3> rows{
            rows_plan(leases[0]->weights),rows_plan(leases[1]->weights,combined),rows_plan(leases[2]->weights)};
        std::vector<float> gate(size_t(count)*intermediate),up(gate.size()),hidden(gate.size()),
            output(size_t(count)*width);
        mfq::cpu::expert_pool().rows(intermediate,[&](int64_t begin,int64_t end) {
            rows[0].run(input.data(),count,width,gate.data(),intermediate,int(begin),int(end));
            rows[1].run(input.data(),count,width,up.data(),intermediate,int(begin),int(end));
            for(int row=0;row<count;++row)for(int column=int(begin);column<int(end);++column) {
                const auto i=size_t(row)*intermediate+column;
                const mfq_half g(gate[i]),u(up[i]);
                if(up_only) {
                    const float positive=std::max(0.0f,float(u));
                    hidden[i]=float(mfq_half(positive*positive));continue;
                }
                if(swiglu_limit>0) {
                    const auto clipped=std::min(float(g),float(swiglu_limit));
                    const auto value=std::clamp(float(u),-float(swiglu_limit),float(swiglu_limit));
                    hidden[i]=float(mfq_half((clipped/(1.0f+std::exp(-clipped)))*value));continue;
                }
                const mfq_half sigmoid(1.0/(1.0+std::exp(-double(float(g)))));
                const mfq_half first(float(g)*float(sigmoid));
                hidden[i]=float(mfq_half(float(first)*float(u)));
            }
        });
        mfq::cpu::expert_pool().rows(width,[&](int64_t begin,int64_t end) {
            rows[2].run(hidden.data(),count,intermediate,output.data(),width,int(begin),int(end));
        });
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
        return mfq::MoeCpuCalibration::Observation{key,preparation_ns+double(ns)};
    };
}

}

struct MoeFfnPrefillHostWorkspace {
    std::shared_ptr<Mapped> x,ids,cpu;
    MoeFfnPrefillHostWorkspace(int tokens,int routes,int width)
        :x(std::make_shared<Mapped>(std::size_t(tokens)*width*sizeof(float))),
         ids(std::make_shared<Mapped>(std::size_t(tokens)*routes*sizeof(int32_t))),
         cpu(std::make_shared<Mapped>(std::size_t(tokens)*routes*width*sizeof(mfq_half))) {}
};

class MoeFfnPipeline : public std::enable_shared_from_this<MoeFfnPipeline> {
    struct CopyEvents {
        cudaEvent_t fork=nullptr,done=nullptr;
        CopyEvents() {
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&fork,cudaEventDisableTiming));
            const auto status=cudaEventCreateWithFlags(&done,cudaEventDisableTiming);
            if(status!=cudaSuccess){cudaEventDestroy(fork);fork=nullptr;MFQ_CUDA_CHECK(status);}
        }
        ~CopyEvents(){if(done)cudaEventDestroy(done);if(fork)cudaEventDestroy(fork);}
    };
    struct PhaseTimes {
        std::array<cudaEvent_t,4> points{};
        void initialize() {
            if(points[0])return;
            try {for(auto& event:points)MFQ_CUDA_CHECK(cudaEventCreate(&event));}
            catch(...) {for(auto& event:points)if(event){cudaEventDestroy(event);event=nullptr;}throw;}
        }
        ~PhaseTimes(){for(auto event:points)if(event)cudaEventDestroy(event);}
    };
    struct DmaTimes {
        std::array<cudaEvent_t,3> points{};
        MoeExpertCache::DmaCallSample sample;
        std::chrono::steady_clock::time_point fetch_started{},enqueue_started{};
        bool started=false;
        DmaTimes() {
            try {
                for(auto& event:points) {
                    const auto status=cudaEventCreate(&event);
                    if(status==cudaErrorMemoryAllocation)throw std::bad_alloc();
                    MFQ_CUDA_CHECK(status);
                }
            } catch(...) {for(auto event:points)if(event)cudaEventDestroy(event);throw;}
        }
        ~DmaTimes(){for(auto event:points)if(event)cudaEventDestroy(event);}
    };
    struct Binding {
        MixedMoeRuntime hot,transfer;
        std::vector<MoeGpuArena*> stages;
        std::vector<tb::Tensor> owner;
    };
    struct CpuJob {
        int expert=0,offset=0,count=0;
        std::array<MoeHostExpertCache::Lease,3> leases;
        std::array<CpuProjectionRows,3> rows;
    };
    struct TransferTemplate {
        mfq::cuda::MfePackedProjection view;
        std::array<const uint8_t*,7> bases{};
    };
    struct TransferScratch {
        struct Field {uint8_t* destination;const uint8_t* source;int64_t bytes,wire_offset=0;int projection=0,expert=0,field=0;};
        struct Interval {const uint8_t* source;int64_t bytes,offset;};
        std::vector<Field> fields;
        std::array<std::vector<int>,3> slots;
        std::array<std::vector<Interval>,2> intervals;
        std::array<std::vector<mfq::cuda::DmaCopy>,2> cache_copies,projection_copies;
        std::array<std::vector<const Field*>,2> loan_copies;
        std::vector<bool> direct_to_cache,projection_fields;
        std::vector<mfq::cuda::KvExpertLoan::Key> protected_keys;
        std::vector<std::size_t> direct,staged,wire_fields;
        std::vector<MoeCacheNewLease> added;
        void clear() {
            fields.clear();added.clear();protected_keys.clear();
            direct.clear();staged.clear();wire_fields.clear();
            direct_to_cache.clear();projection_fields.clear();
            for(auto& phase:intervals)phase.clear();
            for(auto& phase:cache_copies)phase.clear();
            for(auto& phase:projection_copies)phase.clear();
            for(auto& phase:loan_copies)phase.clear();
        }
    };
    struct PredictionCell;
    struct Cell {
        std::shared_ptr<PredictionCell> router_hint;
        TransferScratch transfer_scratch;
        int tokens,routes,entries;
        int calibration_expert=-1;
        tb::Tensor input,ids,weights,output,dispatch_pointers,wire,device_abort,device_kind;
        tb::Tensor shared_input_descriptors;
        int shared_input_groups=0;
        bool shared_input=false;
        bool gpu_resident_dispatch=false;
        bool fused_activation=false;
        bool two_stage=false;
        bool batch_decode=false;
        bool graph_replay=true;
        std::unique_ptr<mfq::cuda::Event> eager_done;
        bool skip_cpu_wait=false;
        bool export_cpu_input=true;
        bool skip_gpu_timing=false;
        bool window_input_views=false;
        bool transfer_cache=false;
        bool mapped_copy=false;
        bool mapped_overlap=false;
        bool phased_transfer=false;
        bool fused_transfer_copy=false;
        bool host_cache_packet=false;
        bool ssd_gate_up_submitted=false;
        bool ssd_down_submitted=false;
        bool ssd_gate_up_ready=false;
        int64_t ssd_gate_up_host_tail=0;
        bool direct_packet=false;
        bool wire_fallback=false;
        bool indirect_wire=false;
        int64_t wire_capacity=0;
        bool cached_resident=false;
        bool reuse_plan=false;
        bool check_reused_views=false;
        std::array<MixedMoeRuntime,3> fused_projections;
        std::unique_ptr<MfeFfnRuntime> fused;
        std::unique_ptr<Mapped> transfer_index,transfer_descriptors,ssd_gate_up_descriptors;
        std::unique_ptr<Mapped> transfer_route_plan;
        std::unique_ptr<Mapped> mapped_copy_table;
        std::unique_ptr<Mapped> kv_loan_table;
        tb::Tensor kv_loan_snapshot;
        tb::Tensor mapped_copy_snapshot;
        std::unique_ptr<CopyEvents> copy_events;
        int native_dma_mode=0;
        std::unique_ptr<mfq::cuda::Event> transfer_notification;
        std::array<bool,2> transfer_notification_pending{};
        std::vector<mfq::cuda::MfePackedProjection> transfer_views;
        std::array<std::vector<TransferTemplate>,3> transfer_templates;
        tb::Tensor shared_gate;
        tb::Tensor shared_gate_quantized,shared_gate_scales;
        bool shared_gate_trace_reported=false;
        bool route_fence_trace_reported=false;
        bool direct_packet_trace_reported=false;
        int64_t wire_header_bytes=0,host_stage_bytes=0;
        int wire_blocks=0,wire_threads=0;
        Mapped x,host_ids,flags,kind,local,cpu;
        std::array<Binding,3> binding;
        PhaseTimes timing;
        std::unique_ptr<DmaTimes> dma_timing;
        std::unique_ptr<mfq::cuda::GraphRegistry> graphs;
        std::vector<float> cpu_input,g,u,hidden,y;
        std::vector<CpuJob> jobs;
        std::uint64_t cpu_profile_calls=0;
        mfq::MoeDispatchPlan plan;
        mfq::MoeDispatchWorkspace plan_workspace;
        std::vector<int32_t> plan_ids;
        bool pending=false,in_window=false;
        bool allow_padding=false;
        int active_tokens=0;
        StreamFlagWrite ready_writer=nullptr;
        Cell(int t,int k,int width,int experts,cudaStream_t stream,bool eager):tokens(t),routes(k),entries(t*k),
            x(eager?64:std::size_t(t)*width*sizeof(float)),host_ids(eager?64:std::size_t(t)*k*sizeof(int32_t)),flags(6*64),
            kind(std::size_t(experts)*sizeof(int32_t)),local(std::size_t(experts)*3*sizeof(int32_t)),
            cpu(eager?64:std::size_t(t)*k*width*sizeof(mfq_half)),graphs(std::make_unique<mfq::cuda::GraphRegistry>(stream)) {
            const auto gpu=tb::TensorOptions().device(tb::kCUDA);
            input=tb::zeros({t,width},gpu.dtype(tb::kFloat16));ids=tb::zeros({t,k},gpu.dtype(tb::kInt32));
            weights=tb::zeros({t,k},gpu.dtype(tb::kFloat32));
        }
    };
    std::array<std::shared_ptr<MoeCachedSource>,3> sources_;
    MoeExpertCache* cache_;
    MoeFfnShared shared_;
    double swiglu_limit_=0;
    MoeFfnSharedWeights shared_weights_;
    MoeFfnDispatchObserver observer_;
    std::map<std::tuple<int,int,const void*,bool,int>,std::unique_ptr<Cell>> cells_;
    struct WindowRoutes {std::vector<int32_t> ids;int tokens=0;};
    std::map<const void*,std::shared_ptr<WindowRoutes>> window_routes_;
    struct PredictionCell {
        Mapped flags;
        int routes=0;
        tb::Tensor aborted=tb::zeros({1},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
        explicit PredictionCell(std::size_t candidates=16):flags((64+candidates*sizeof(int32_t)+127)/128*128) {
            publish(flags.h<uint32_t>()+4);publish(flags.h<uint32_t>()+8);
        }
    };
    std::map<const void*,std::shared_ptr<PredictionCell>> prediction_cells_;
    int width_,ff_,experts_;
    bool combined_gate_up_=false;
    // Single Up image shared by the Gate and Up stages (non-gated experts).
    bool up_only_=false;
    bool single_up()const{return combined_gate_up_||up_only_;}
    cudaStream_t stream_=nullptr;
    bool failed_=false;
    std::shared_ptr<Mapped> cached_expert_fields_;
    uint64_t cached_resident_experts_=0;
    bool trace_serve_=false;
    double cpu_ns_=0,cpu_work_=0,gpu_ns_=0,gpu_samples_=0;
    double stage_ns_=0,stage_bytes_=0;
    std::vector<mfq::MoeCpuCostModel::Key> cpu_cost_keys_;
    mfq::MoeCpuCostModel::Key cpu_cost_key(const mfq::MoeDispatchGroup& group)const {
        auto key=cpu_cost_keys_[group.expert];key[3]=group.positions.size();return key;
    }

        // Two compact expert groups keep the copy engine ahead of arithmetic.
    // The group-local route space avoids a full-layer transfer/output workspace.
    static constexpr int prefill_group_capacity=96;
    struct PrefillBuffer {
        std::array<MixedMoeRuntime,3> projections;
        tb::Tensor maps;
        std::array<tb::Tensor,3> nvq_metadata;
        std::array<std::vector<MoeGpuArena*>,3> stages;
        mfq::cuda::Event gu_ready{cudaEventDisableTiming};
        mfq::cuda::Event down_ready{cudaEventDisableTiming};
        mfq::cuda::Event used{cudaEventDisableTiming};
    };
    struct PrefillStream {
        std::array<PrefillBuffer,2> buffers;
        int group_size=prefill_group_capacity;
    };
    std::unique_ptr<PrefillStream> prefill_stream_;

    static int64_t prefill_align(int64_t bytes) {return (bytes+255)&~int64_t(255);}
    static int64_t prefill_cohort_bytes(const MoeCachedCohort& cohort,int capacity) {
        const int slots=2*std::min(capacity,cohort.cpu->local_experts);
        int64_t bytes=0;
        for(const auto size:cohort.bytes_per_expert)bytes+=prefill_align(slots*size);
        return bytes;
    }
    MoeGpuArena& prefill_stage(int projection,MoeCachedCohort& cohort,int64_t& offset,int capacity) {
        const auto key="prefill:"+std::to_string(sources_[projection]->id_)+":"+std::to_string(offset);
        auto& arena=cache_->pipeline_stages_[key];
        if(arena) {offset+=prefill_cohort_bytes(cohort,capacity);return *arena;}
        auto prepared=std::make_unique<MoeGpuArena>();
        prepared->signature=key;prepared->slot_bytes=cohort.arena->slot_bytes;
        prepared->layouts=cohort.arena->layouts;
        prepared->slots=2*std::min(capacity,cohort.cpu->local_experts);
        for(std::size_t f=0;f<prepared->layouts.size();++f) {
            const auto& layout=prepared->layouts[f];
            auto shape=layout.slot_shape;shape[0]*=prepared->slots;
            const auto bytes=prefill_align(prepared->slots*cohort.bytes_per_expert[f]);
            if(offset+bytes>cache_->pipeline_prefill_ring_.numel())
                throw std::logic_error("MFQ prefill ring budget is smaller than its layer layout");
            auto* pointer=cache_->pipeline_prefill_ring_.data_ptr<uint8_t>()+offset;
            prepared->fields.push_back(tb::from_blob(pointer,shape,
                tb::TensorOptions().device(tb::kCUDA).dtype(layout.scalar_type)));
            offset+=bytes;
        }
        arena=std::move(prepared);return *arena;
    }
    void initialize_prefill_stream(CudaExecutionContext& execution,const tb::Tensor& ids) {
        if(prefill_stream_)return;
        const bool predictor=sources_[0]->layer_id_<0;
        if(cache_->ssd_experts() && cache_->pipeline_prefill_ring_.defined() &&
                cache_->pipeline_prefill_predictor_!=predictor)
            release_moe_prefill_buffers(cache_);
        auto prepared=std::make_unique<PrefillStream>();
        int capacity=cache_->pipeline_prefill_group_capacity_;
        if(!cache_->pipeline_prefill_ring_.defined()) {
            capacity=std::clamp(mfq::cuda::runtime_options::moe_prefill_group_size(),1,prefill_group_capacity);
            if(cache_->ssd_experts()) {
                if(!std::getenv("MFQ_MOE_PREFILL_GROUP_SIZE"))capacity=64;
                std::unordered_map<int,std::size_t> layer_expert_bytes;
                for(const auto& source:cache_->sources_) {
                    if((source->layer_id_<0)!=predictor)continue;
                    std::size_t largest=0;
                    for(const auto& cohort:source->cohorts_) {
                        std::size_t bytes=0;for(const auto size:cohort.bytes_per_expert)bytes+=size;
                        largest=std::max(largest,bytes);
                    }
                    layer_expert_bytes[source->layer_id_]+=largest;
                }
                std::size_t largest=1;for(const auto& layer:layer_expert_bytes)largest=std::max(largest,layer.second);
                capacity=std::min<std::size_t>(capacity,std::max<std::size_t>(1,
                    cache_->host_experts_->stats().budget_bytes/2/largest));
            }
            const auto ring_bytes=[&](int slots) {
                if(cache_->ssd_experts()) {
                    struct Layer {int64_t baseline=0,largest=0;};
                    std::unordered_map<int,Layer> layers;
                    for(const auto& source:cache_->sources_) {
                        // Backbone and predictor prefill execute separately.
                        // A large MTP projection must not shrink every
                        // backbone group or occupy its transfer workspace.
                        if((source->layer_id_<0)!=predictor)continue;
                        auto& layer=layers[source->layer_id_];int64_t largest=0;
                        for(const auto& cohort:source->cohorts_) {
                            int64_t bytes=0;
                            for(const auto size:cohort.bytes_per_expert) {
                                bytes+=size;layer.baseline+=prefill_align(size);
                            }
                            largest=std::max(largest,bytes);
                        }
                        layer.largest+=largest;
                    }
                    int64_t bytes=0;
                    for(const auto& [layer,budget]:layers)
                        bytes=std::max(bytes,2*prefill_align(budget.baseline+slots*budget.largest));
                    return bytes;
                }
                std::unordered_map<int,int64_t> layers;
                for(const auto& source:cache_->sources_)for(const auto& cohort:source->cohorts_)
                    layers[source->layer_id_]+=prefill_cohort_bytes(cohort,slots);
                int64_t bytes=0;for(const auto& layer:layers)bytes=std::max(bytes,layer.second);
                return bytes;
            };
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            const auto context=mfq::cuda::default_context(ids.get_device());
            // SSD rings are additional to attention/hidden-state scratch.
            // Consuming half that budget here can prevent the next layer from
            // allocating its FP32 query and attention output.
            const auto reserve=mfq::cuda::runtime_options::workspace_reserve_bytes()/(cache_->ssd_experts()?1:2);
            auto available=context->available_memory()+context->workspace_available();
            const auto requested=std::size_t(ring_bytes(capacity))+reserve;
            if(cache_->ssd_experts() && available<requested && cache_->reserve_workspace_) {
                cache_->reserve_workspace_(requested);
                available=context->available_memory()+context->workspace_available();
            }
            const auto limit=available>reserve?available-reserve:0;
            while(capacity>1 && std::uint64_t(ring_bytes(capacity))>limit)capacity=(capacity+1)/2;
#endif
            const auto bytes=ring_bytes(capacity);
            cache_->pipeline_prefill_ring_=tb::empty({bytes},ids.options().dtype(tb::kUInt8));
            cache_->pipeline_prefill_group_capacity_=capacity;
            cache_->pipeline_prefill_predictor_=predictor;
            if(cache_->ssd_experts())std::cerr<<"moe_ssd_prefill_ring group_size="<<capacity<<" bytes="<<bytes
                <<" predictor="<<int(predictor)<<'\n';
        }
        prepared->group_size=std::min(capacity,experts_);
        const int prefill_group_size=prepared->group_size;
        std::array<std::vector<MoeGpuArena*>,3> stages;int64_t stage_offset=0;
        for(int p=0;p<3;++p) {
            if(single_up() && p==1) { stages[p]=stages[0]; continue; }
            for(auto& cohort:sources_[p]->cohorts_)
                stages[p].push_back(&prefill_stage(p,cohort,stage_offset,cache_->ssd_experts()?1:capacity));
        }
        if(swiglu_limit_<=0 && execution.config.moe_ffn_fused_activation && !cache_->pipeline_sigmoid_table_.defined())
            cache_->pipeline_sigmoid_table_=moe_swiglu_sigmoid_table_cuda();
        int map_count=0;
        for(const auto& source:sources_)map_count+=2*int(source->cohorts_.size())*prefill_group_size;
        for(auto& buffer:prepared->buffers)buffer.maps=tb::full({map_count},-1,ids.options());
        constexpr int64_t nvq_descriptor_bytes=5*8+3*8+7*4;
        if(cache_->ssd_experts() && !cache_->pipeline_prefill_metadata_host_.defined()) {
            std::unordered_map<int,int64_t> layers;
            for(const auto& source:cache_->sources_) {
                int64_t count=0;
                for(const auto& cohort:source->cohorts_)count+=cohort.active.family==MixedMoeFamily::Nvq;
                layers[source->layer_id_]+=prefill_align(2*count*nvq_descriptor_bytes);
            }
            int64_t stride=0;for(const auto& [layer,bytes]:layers)stride=std::max(stride,bytes);
            if(stride) {
                cache_->pipeline_prefill_metadata_host_=tb::empty({2*stride},
                    tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
                cache_->pipeline_prefill_metadata_stride_=std::size_t(stride);
            }
        }
        std::array<int64_t,2> metadata_offsets{};
        for(int b=0;b<2;++b)for(int p=0;p<3;++p) {
            auto& buffer=prepared->buffers[b];auto& runtime=buffer.projections[p];
            auto& source=*sources_[p];
            runtime.n_experts=prefill_group_size;runtime.out_per_expert=source.cpu_->out_per_expert;
            runtime.neuron_len=source.cpu_->neuron_len;
            runtime.prefill_only=true;
            // Each routed row has exactly one owner across the hot/cold pools.
            runtime.partial_experts=false;
            int map_base=0;
            for(int previous=0;previous<p;++previous)map_base+=2*int(sources_[previous]->cohorts_.size())*prefill_group_size;
            int cohort_index=0;
            for(auto& cohort:source.cohorts_) {
                auto& arena=*stages[p][cohort_index];buffer.stages[p].push_back(&arena);
                std::vector<tb::Tensor> fields;
                for(std::size_t f=0;f<arena.fields.size();++f) {
                    const auto rows=arena.layouts[f].slot_shape[0];
                    const int slots=arena.slots/2;
                    fields.push_back(arena.fields[f].narrow(0,b*slots*rows,slots*rows));
                }
                auto hot=cohort.active;
                hot.expert_local=buffer.maps.narrow(0,map_base+2*cohort_index*prefill_group_size,prefill_group_size);
                auto cold=fields_pool(cohort.active,fields,arena.slots/2,runtime.out_per_expert);
                cold.expert_local=buffer.maps.narrow(0,map_base+(2*cohort_index+1)*prefill_group_size,prefill_group_size);
                runtime.pools.push_back(std::move(hot));runtime.pools.push_back(std::move(cold));++cohort_index;
            }
            const auto mma=mfq::cuda::runtime_options::moe_prefill_compact_mma();
            // These maps were just initialized to -1. Supplying their known
            // host values avoids reading each one back from the GPU, which
            // otherwise serializes construction with preceding layer work.
            const std::vector<std::vector<int32_t>> initial_maps(runtime.pools.size(),
                std::vector<int32_t>(prefill_group_size,-1));
            initialize_mixed_nvq_dispatch(runtime,execution.config,!mma || *mma!=0,&initial_maps);
            if(cache_->ssd_experts() && runtime.nvq_dispatch && !(single_up() && p==1)) {
                const auto bytes=runtime.nvq_dispatch->pool_count*nvq_descriptor_bytes;
                const auto stride=cache_->pipeline_prefill_metadata_stride_;
                if(std::uint64_t(metadata_offsets[b]+bytes)>stride)
                    throw std::logic_error("prefill NVQ descriptor packet exceeds its buffer");
                buffer.nvq_metadata[p]=cache_->pipeline_prefill_metadata_host_.narrow(0,
                    int64_t(b*stride)+metadata_offsets[b],bytes);
                metadata_offsets[b]+=prefill_align(bytes);
            }
        }
        prefill_stream_=std::move(prepared);
        cache_->pipeline_prefill_release_.push_back([owner=weak_from_this()] {
            if(auto pipeline=owner.lock())pipeline->prefill_stream_.reset();
        });
    }
    void upload_prefill_nvq_metadata(PrefillBuffer& buffer,int projection) {
        auto& runtime=buffer.projections[projection];
        if(!buffer.nvq_metadata[projection].defined())return;
        auto& dispatch=*runtime.nvq_dispatch;
        auto* base=static_cast<uint8_t*>(buffer.nvq_metadata[projection].data_ptr());
        const auto count=dispatch.pool_count;
        auto* pointers=reinterpret_cast<int64_t*>(base);
        auto* sizes=reinterpret_cast<int64_t*>(base+count*5*8);
        auto* params=reinterpret_cast<int32_t*>(base+count*8*8);
        int index=0;
        for(const auto& pool:runtime.pools)if(pool.family==MixedMoeFamily::Nvq) {
            if(index>=count)throw std::logic_error("prefill NVQ pool count changed");
            const auto& w=pool.nvq;
            const std::array<tb::Tensor,5> fields={w.indices_packed,w.aux_packed,w.sub_scale_packed,w.neuron_scale,w.codebook};
            for(int f=0;f<5;++f)pointers[index*5+f]=int64_t(reinterpret_cast<uintptr_t>(fields[f].data_ptr()));
            sizes[index*3]=w.indices_packed.numel();sizes[index*3+1]=w.aux_packed.numel();sizes[index*3+2]=w.sub_scale_packed.numel();
            const int format=int(w.kernel_format);
            const bool d4=format==3 || format==10 || format==11 || format==12 || format==15 || format==17;
            const std::array<int32_t,7> row={pool.local_experts,int(w.ng),
                (runtime.neuron_len+(d4?3:7))/(d4?4:8),(runtime.neuron_len+7)/8,
                int(w.sub_bits),int(w.sign_mode),format};
            std::copy(row.begin(),row.end(),params+index*7);++index;
        }
        if(index!=count)throw std::logic_error("prefill NVQ descriptor packet lost a pool");
        MFQ_CUDA_CHECK(cudaMemcpyAsync(dispatch.weight_ptrs.data_ptr(),pointers,std::size_t(count)*5*8,cudaMemcpyHostToDevice,cache_->weight_stream_));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(dispatch.weight_sizes.data_ptr(),sizes,std::size_t(count)*3*8,cudaMemcpyHostToDevice,cache_->weight_stream_));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(dispatch.pool_params.data_ptr(),params,std::size_t(count)*7*4,cudaMemcpyHostToDevice,cache_->weight_stream_));
    }
    tb::Tensor streamed_prefill(CudaExecutionContext& execution,const tb::Tensor& input,
            const tb::Tensor& ids,const tb::Tensor& weights) {
        using Clock=std::chrono::steady_clock;
        const auto elapsed=[](Clock::time_point started) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-started).count();
        };
        const auto started=Clock::now();
        initialize_prefill_stream(execution,ids);
        auto& stream=*prefill_stream_;auto& stats=cache_->stats_;
        const auto plan_before=stats.pipeline_plan_ns,fetch_before=stats.pipeline_fetch_ns;
        const auto route_before=stats.pipeline_route_wait_ns;
        std::uint64_t read_join_ns=0,buffer_wait_ns=0,compute_submit_ns=0;
        const int prefill_group_size=stream.group_size;
        const int tokens=int(ids.size(0)),routes=int(ids.size(1)),entries=tokens*routes;
        auto publication=std::static_pointer_cast<Mapped>(cache_->pipeline_prefill_route_publication_);
        const auto publication_bytes=64+std::size_t(entries)*sizeof(int32_t);
        // A caller may change compute streams between eager chunks. The
        // preceding expert fence also retires its reads of the shared packet.
        if(publication && cache_->compute_done_recorded_)
            MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->compute_done_));
        if(!publication || publication->bytes<publication_bytes) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));
            publication=std::make_shared<Mapped>(publication_bytes);
            cache_->pipeline_prefill_route_publication_=publication;
        }
        const auto setup_ns=elapsed(started);
        auto* published=publication->h<uint32_t>();*published=0;
        mfq::cuda::moe_publish_routes(nullptr,ids.data_ptr<int32_t>(),nullptr,
            publication->d<int32_t>()+16,publication->d<uint32_t>(),0,entries,stream_);
        auto shared=shared_?shared_(execution,input):tb::Tensor{};
        const auto route_started=Clock::now();
        mfq::cuda::wait_route_publication(published,stream_);
        stats.pipeline_route_wait_ns+=elapsed(route_started);
        if(cache_->ssd_experts() && !stats.pipeline_prefill_group_serves)
            std::cerr<<"moe_ssd_prefill_dispatch path=grouped tokens="<<tokens<<" routes="<<routes<<'\n';
        ++stats.pipeline_serves;++stats.pipeline_prefill_group_serves;
        // Publication follows every previous consumer of the shared ring and
        // pinned metadata. A completed previous DMA is implied by that consumer.
        if(cache_->transfer_ready_recorded_)MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->transfer_ready_));
        cache_->pipeline_dma_leases_.clear();
        const auto plan_started=Clock::now();
        const auto* host_ids=publication->h<int32_t>()+16;
        std::vector<std::vector<int32_t>> positions(experts_);
        for(int i=0;i<entries;++i) {
            const int e=host_ids[i];
            if(e<0 || e>=experts_)throw std::runtime_error("MFQ prefill expert outside route geometry");
            positions[e].push_back(i);
        }
        std::vector<int> order;
        for(int e=0;e<experts_;++e)if(!positions[e].empty())order.push_back(e);
        if(cache_->ssd_experts())std::sort(order.begin(),order.end(),[&](int a,int b) {
            for(int p=0;p<3;++p) {
                const auto x=sources_[p]->expert_to_cohort_[a],y=sources_[p]->expert_to_cohort_[b];
                if(x!=y)return x<y;
            }
            return a<b;
        });
        const int groups=(int(order.size())+prefill_group_size-1)/prefill_group_size;
        struct Upload {int cohort,slot,expert;std::vector<tb::Tensor> fields;int route=0;};
        struct Group {
            int begin=0,rows=0;
            int route_offset=0,mma_tile_m=8,wide_tile_m=8;
            std::array<std::vector<Upload>,3> uploads;
            std::array<std::future<void>,2> pending;
            std::size_t down_host_offset=0;
        };
        std::vector<Group> work(groups);stats.pipeline_prefill_groups+=groups;
        struct DrainFailedReads {
            cudaStream_t stream;
            std::vector<Group>& work;
            int exceptions=std::uncaught_exceptions();
            ~DrainFailedReads(){
                for(auto& group:work)for(auto& phase:group.pending)if(phase.valid())phase.wait();
                if(std::uncaught_exceptions()>exceptions)(void)cudaStreamSynchronize(stream);
            }
        } drain{cache_->weight_stream_,work};
        std::array<int,3> map_offset{};int map_stride=0;
        for(int p=0;p<3;++p) {
            map_offset[p]=map_stride;
            map_stride+=int(stream.buffers[0].projections[p].pools.size())*prefill_group_size;
        }
        // The following layer's route publication retires every preceding
        // metadata read on this stream. Reuse its pinned packet instead of
        // allocating/freeing locked memory and synchronizing GPU work per layer.
        const std::size_t index_bytes=std::size_t(entries)*2*sizeof(int64_t);
        const std::size_t id_bytes=std::size_t(entries)*sizeof(int32_t);
        const std::size_t map_bytes=std::size_t(groups)*map_stride*sizeof(int32_t);
        const int bounds_size=prefill_group_size+1;
        const int group_fixed=4*bounds_size+prefill_group_size;
        const std::size_t route_words=std::size_t(entries)*4+std::size_t(groups)*group_fixed;
        const auto metadata_bytes=index_bytes+id_bytes+map_bytes+route_words*sizeof(int32_t);
        auto metadata=std::static_pointer_cast<Mapped>(cache_->pipeline_prefill_dma_metadata_);
        if(!metadata || metadata->bytes<metadata_bytes) {
            metadata=std::make_shared<Mapped>(metadata_bytes);
            cache_->pipeline_prefill_dma_metadata_=metadata;
        }
        auto* token_indices=metadata->h<int64_t>();
        auto* pair_indices=token_indices+entries;
        auto* group_ids=reinterpret_cast<int32_t*>(pair_indices+entries);
        auto* maps=group_ids+entries;std::fill(maps,maps+std::size_t(groups)*map_stride,-1);
        auto* route_metadata=maps+std::size_t(groups)*map_stride;
        int row=0;
        for(int g=0;g<groups;++g) {
            auto& group=work[g];group.begin=row;
            std::array<std::vector<int>,3> next;
            for(int p=0;p<3;++p)next[p].resize(sources_[p]->cohorts_.size());
            const int count=std::min(prefill_group_size,int(order.size())-g*prefill_group_size);
            for(int q=0;q<count;++q) {
                const int e=order[g*prefill_group_size+q];
                ++stats.hybrid_gpu_experts;bool cold_expert=false;
                for(int position:positions[e]) {
                    token_indices[row]=position/routes;pair_indices[row]=position;group_ids[row]=q;++row;
                }
                for(int p=0;p<3;++p) {
                    if(single_up() && p==1)continue;
                    auto& source=*sources_[p];const int c=source.expert_to_cohort_[e];
                    const auto& cohort=source.cohorts_[c];
                    const int hot=cohort.arena->book->slot_for({source.id_,c,e});
                    auto* map=maps+g*map_stride+map_offset[p]+2*c*prefill_group_size;
                    if(hot>=0) {map[q]=hot;continue;}
                    cold_expert=true;
                    const int slot=next[p][c]++;map[prefill_group_size+q]=slot;
                    if(cache_->ssd_experts())group.uploads[p].push_back({c,slot,e,{},q});
                    else {
                        auto lease=source.quant_source_->acquire_expert(e);
                        group.uploads[p].push_back({c,slot,e,moe_cache_fields(lease->weights)});
                        cache_->pipeline_dma_leases_.push_back(std::move(lease));
                    }
                }
                stats.ram_pcie_experts+=cold_expert;
            }
            if(cache_->ssd_experts())for(int p=0;p<3;++p) {
                auto& uploads=group.uploads[p];const auto& store=sources_[p]->quant_source_->store();
                // Compute order is shared by all projections, while their
                // physical expert rows differ. Pack each cold projection in
                // its own file order and publish the corresponding map.
                std::sort(uploads.begin(),uploads.end(),[&](const Upload& a,const Upload& b) {
                    return a.cohort!=b.cohort?a.cohort<b.cohort:store.expert_local(a.expert)<store.expert_local(b.expert);
                });
                int cohort=-1,slot=0;
                for(auto& upload:uploads) {
                    if(cohort!=upload.cohort){cohort=upload.cohort;slot=0;}
                    upload.slot=slot++;
                    maps[g*map_stride+map_offset[p]+(2*cohort+1)*prefill_group_size+upload.route]=upload.slot;
                }
            }
            group.rows=row-group.begin;
        }
        if(row!=entries)throw std::logic_error("MFQ prefill grouping lost routed rows");
        // The host has already sorted rows by expert for the DMA plan. Reuse
        // that ordering instead of running histogram/prefix/scatter kernels
        // and allocating three separate maps for every group.
        int route_offset=0;
        for(auto& group:work) {
            group.route_offset=route_offset;
            const int density=std::max(1,(group.rows+prefill_group_size-1)/prefill_group_size);
            group.mma_tile_m=density<4?8:(density<=16?16:(density<=32?32:64));
            group.wide_tile_m=density>64?128:group.mma_tile_m;
            auto* destination=route_metadata+route_offset;
            auto* bounds=destination+group.rows;
            auto* counts=bounds+bounds_size;
            std::fill(counts,counts+prefill_group_size,0);
            for(int i=0;i<group.rows;++i) {destination[i]=i;++counts[group_ids[group.begin+i]];}
            bounds[0]=0;
            for(int q=0;q<prefill_group_size;++q)bounds[q+1]=bounds[q]+counts[q];
            auto* tile_bounds=counts+prefill_group_size;
            auto* tile_experts=tile_bounds+bounds_size;
            for(int tile_m:{8,group.mma_tile_m,group.wide_tile_m}) {
                tile_bounds[0]=0;std::fill(tile_experts,tile_experts+group.rows,0);
                for(int q=0;q<prefill_group_size;++q) {
                    tile_bounds[q+1]=tile_bounds[q]+(counts[q]+tile_m-1)/tile_m;
                    std::fill(tile_experts+tile_bounds[q],tile_experts+tile_bounds[q+1],q);
                }
                tile_bounds=tile_experts+group.rows;tile_experts=tile_bounds+bounds_size;
            }
            route_offset+=4*group.rows+group_fixed;
        }
        if(std::size_t(route_offset)!=route_words)throw std::logic_error("MFQ prefill route metadata geometry differs");
        stats.pipeline_plan_ns+=elapsed(plan_started);
        auto token_device=tb::empty({entries},ids.options().dtype(tb::kInt64));
        auto pair_device=tb::empty({entries},ids.options().dtype(tb::kInt64));
        auto route_device=tb::empty({entries},ids.options());
        auto map_device=tb::empty({int64_t(route_words)},ids.options());
        MFQ_CUDA_CHECK(cudaMemcpyAsync(token_device.data_ptr(),token_indices,std::size_t(entries)*sizeof(int64_t),cudaMemcpyHostToDevice,stream_));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(pair_device.data_ptr(),pair_indices,std::size_t(entries)*sizeof(int64_t),cudaMemcpyHostToDevice,stream_));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(route_device.data_ptr(),group_ids,id_bytes,cudaMemcpyHostToDevice,stream_));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(map_device.data_ptr(),route_metadata,route_words*sizeof(int32_t),cudaMemcpyHostToDevice,stream_));
        auto parts=tb::empty({entries,width_},input.options());
        const auto ram_begin=reinterpret_cast<std::uintptr_t>(cache_->pipeline_ram_complement_->data_ptr());
        const auto ram_end=ram_begin+cache_->pipeline_ram_registered_bytes_;
        if(cache_->ssd_experts()) {
            std::size_t required=0;
            for(const auto& group:work) {
                std::size_t bytes=0;
                for(int p=0;p<3;++p)for(std::size_t c=0;c<sources_[p]->cohorts_.size();++c) {
                    const auto count=std::count_if(group.uploads[p].begin(),group.uploads[p].end(),
                        [&](const auto& upload){return upload.cohort==int(c);});
                    if(!count)continue;
                    for(const auto field:sources_[p]->cohorts_[c].bytes_per_expert)if(field)
                        bytes=prefill_align(bytes)+std::size_t(count)*field;
                }
                required=std::max(required,std::size_t(prefill_align(bytes)));
            }
            if(required>cache_->pipeline_prefill_host_stride_) {
                // The preceding layer's transfer fence was joined above.
                // Keep two pinned packets across layers; neither
                // one is replaced while a DMA can still read it.
                cache_->pipeline_prefill_host_ring_={};
                cache_->pipeline_prefill_host_stride_=0;
                cache_->pipeline_prefill_host_ring_=tb::empty({int64_t(required)*2},
                    tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
                cache_->pipeline_prefill_host_stride_=required;
            }
        }
        const auto queue_group=[&](int g) {
            std::size_t offset=0;
            auto& buffer=stream.buffers[g%2];
            // All formats share this half of the ring. Only the group's
            // actual cold experts need weight storage, with one valid dummy
            // slot for maps that contain no routed expert of that format.
            int64_t device_offset=int64_t(g%2)*cache_->pipeline_prefill_ring_.numel()/2;
            const int64_t device_end=int64_t(g%2+1)*cache_->pipeline_prefill_ring_.numel()/2;
            for(int p=0;p<3;++p) {
                if(single_up() && p==1)continue;
                for(std::size_t c=0;c<sources_[p]->cohorts_.size();++c) {
                    const auto& cohort=sources_[p]->cohorts_[c];
                    const auto count=std::max<std::size_t>(1,std::count_if(work[g].uploads[p].begin(),work[g].uploads[p].end(),
                        [&](const auto& upload){return upload.cohort==int(c);}));
                    std::vector<tb::Tensor> fields;
                    for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                        device_offset=prefill_align(device_offset);
                        const auto bytes=int64_t(count)*cohort.bytes_per_expert[f];
                        if(bytes>device_end-device_offset)throw std::logic_error("compact prefill group exceeded its shared ring");
                        auto shape=cohort.arena->layouts[f].slot_shape;shape[0]*=int64_t(count);
                        fields.push_back(tb::from_blob(cache_->pipeline_prefill_ring_.data_ptr<uint8_t>()+device_offset,shape,
                            tb::TensorOptions().device(tb::kCUDA).dtype(cohort.arena->layouts[f].scalar_type)));
                        device_offset+=bytes;
                    }
                    auto& cold=buffer.projections[p].pools[2*c+1];
                    const auto map=cold.expert_local;
                    cold=fields_pool(cohort.active,fields,int(count),buffer.projections[p].out_per_expert);
                    cold.expert_local=map;
                }
            }
            for(int p=0;p<3;++p)for(std::size_t c=0;c<sources_[p]->cohorts_.size();++c) {
                if(p==2 && c==0)work[g].down_host_offset=offset;
                std::vector<Upload*> uploads;
                for(auto& upload:work[g].uploads[p])if(upload.cohort==int(c))uploads.push_back(&upload);
                if(uploads.empty())continue;
                const auto fields=moe_cache_fields(buffer.projections[p].pools[2*c+1]);
                for(auto* upload:uploads)upload->fields.resize(fields.size());
                for(std::size_t f=0;f<fields.size();++f) {
                    const auto bytes=std::size_t(sources_[p]->cohorts_[c].bytes_per_expert[f]);
                    if(bytes)offset=prefill_align(offset);
                    if(offset>cache_->pipeline_prefill_host_stride_ ||
                            uploads.size()*bytes>cache_->pipeline_prefill_host_stride_-offset)
                        throw std::logic_error("SSD prefill direct target exceeded its packet");
                    auto* target=cache_->pipeline_prefill_host_ring_.data_ptr<uint8_t>()+
                        std::size_t(g%2)*cache_->pipeline_prefill_host_stride_+offset;
                    for(std::size_t i=0;i<uploads.size();++i)
                        uploads[i]->fields[f]=tb::from_blob(target+i*bytes,
                            {int64_t(bytes/fields[f].element_size())},
                            tb::TensorOptions().device(tb::kCPU).dtype(fields[f].scalar_type()));
                    offset+=uploads.size()*bytes;
                }
            }
            std::array<std::vector<MoeExpertCache::SsdExpertCopy>,2> requests;
            for(int p=0;p<3;++p)for(const auto& upload:work[g].uploads[p])
                requests[p==2?1:0].push_back({sources_[p]->quant_source_.get(),upload.expert,upload.fields});
            if(requests[0].empty() && requests[1].empty())return;
            work[g].pending=cache_->queue_ssd_prefill_phases(std::move(requests));
        };
        const auto retire_cpu_group=[&](int g) {
            const auto wait_started=Clock::now();
            stream.buffers[g%2].down_ready.synchronize();
            buffer_wait_ns+=elapsed(wait_started);
            for(auto& uploads:work[g].uploads)for(auto& upload:uploads)upload.fields.clear();
        };
        const auto stage_group=[&](int g,int phase=-1) {
            const auto started=Clock::now();auto& buffer=stream.buffers[g%2];
            std::size_t host_offset=0;
            if(cache_->ssd_experts()) {
                const auto wait_started=Clock::now();
                if(g>=2 && phase!=1)buffer.down_ready.synchronize();
                buffer_wait_ns+=elapsed(wait_started);
                const auto read_started=Clock::now();
                if(work[g].pending.at(phase).valid())work[g].pending.at(phase).get();
                read_join_ns+=elapsed(read_started);
            }
            if(g>=2 && phase!=1)MFQ_CUDA_CHECK(cudaStreamWaitEvent(cache_->weight_stream_,buffer.used.get(),0));
            const int first_projection=phase==1?2:0,last_projection=phase==0?2:3;
            // Down follows the GU portion of the same pinned packet.
            if(phase==1)host_offset=work[g].down_host_offset;
            for(int p=first_projection;p<last_projection;++p) {
                if(cache_->ssd_experts())upload_prefill_nvq_metadata(buffer,p);
                const auto& uploads=work[g].uploads[p];auto& source=*sources_[p];
                for(std::size_t c=0;c<source.cohorts_.size();++c) {
                    std::vector<const Upload*> cohort_uploads;
                    for(const auto& upload:uploads)if(upload.cohort==int(c))cohort_uploads.push_back(&upload);
                    const auto& cohort=source.cohorts_[c];
                    const auto destination_fields=moe_cache_fields(buffer.projections[p].pools[2*c+1]);
                    for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                        const auto bytes=std::size_t(cohort.bytes_per_expert[f]);if(!bytes)continue;
                        auto* destination_base=static_cast<uint8_t*>(destination_fields[f].data_ptr());
                        if(cache_->ssd_experts()) {
                            if(cohort_uploads.empty())continue;
                            host_offset=prefill_align(host_offset);
                            const auto count=cohort_uploads.size(),total=count*bytes;
                            if(host_offset>cache_->pipeline_prefill_host_stride_ ||
                                    total>cache_->pipeline_prefill_host_stride_-host_offset)
                                throw std::logic_error("SSD prefill pinned packet exceeded its bound");
                            auto* packed=cache_->pipeline_prefill_host_ring_.data_ptr<uint8_t>()+
                                std::size_t(g%2)*cache_->pipeline_prefill_host_stride_+host_offset;
                            for(std::size_t i=0;i<count;++i) {
                                if(cohort_uploads[i]->slot!=int(i))
                                    throw std::logic_error("SSD prefill cohort slots are not contiguous");
                                if(cohort_uploads[i]->fields[f].data_ptr()!=packed+i*bytes)
                                    throw std::logic_error("SSD prefill field was not read into its pinned packet");
                            }
                            MFQ_CUDA_CHECK(cudaMemcpyAsync(destination_base,packed,total,
                                cudaMemcpyHostToDevice,cache_->weight_stream_));
                            host_offset+=total;
                            stats.ram_pcie_bytes+=total;stats.pipeline_staged_ram_bytes+=total;
                            if(p<2)stats.pipeline_gate_up_dma_bytes+=total;else stats.pipeline_down_dma_bytes+=total;
                            ++stats.pipeline_dma_copies;continue;
                        }
                        for(std::size_t first=0;first<cohort_uploads.size();) {
                            const auto start=reinterpret_cast<std::uintptr_t>(cohort_uploads[first]->fields[f].data_ptr());
                            if(start<ram_begin || start>ram_end || bytes>ram_end-start) {
                                // Predictor weights can remain pageable to leave
                                // Windows pinned-memory headroom for decode.
                                auto* destination=destination_base+std::size_t(cohort_uploads[first]->slot)*bytes;
                                // CUDA rejects a range spanning registered and
                                // pageable host pages. The registration budget
                                // can end inside an expert field.
                                const auto pinned=start>=ram_begin && start<ram_end?
                                    std::min(bytes,std::size_t(ram_end-start)):std::size_t(0);
                                if(pinned && pinned<bytes) {
                                    MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,reinterpret_cast<const void*>(start),
                                        pinned,cudaMemcpyHostToDevice,cache_->weight_stream_));
                                    MFQ_CUDA_CHECK(cudaMemcpyAsync(destination+pinned,reinterpret_cast<const void*>(start+pinned),
                                        bytes-pinned,cudaMemcpyHostToDevice,cache_->weight_stream_));
                                    ++stats.pipeline_dma_copies;
                                } else MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,reinterpret_cast<const void*>(start),
                                    bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
                                stats.ram_pcie_bytes+=bytes;stats.pipeline_staged_ram_bytes+=bytes;
                                if(p<2)stats.pipeline_gate_up_dma_bytes+=bytes;else stats.pipeline_down_dma_bytes+=bytes;
                                ++stats.pipeline_dma_copies;++first;continue;
                            }
                            std::size_t last=first+1,pitch=0;
                            if(last<cohort_uploads.size()) {
                                const auto next=reinterpret_cast<std::uintptr_t>(cohort_uploads[last]->fields[f].data_ptr());
                                if(next>start && next-start>=bytes && next-start<=std::size_t(INT_MAX))pitch=next-start;
                            }
                            if(pitch)while(last<cohort_uploads.size()) {
                                const auto next=reinterpret_cast<std::uintptr_t>(cohort_uploads[last]->fields[f].data_ptr());
                                if(next!=start+(last-first)*pitch || next>ram_end || bytes>ram_end-next)break;
                                ++last;
                            }
                            const auto rows=last-first,total=bytes*rows;
                            auto* destination=destination_base+std::size_t(cohort_uploads[first]->slot)*bytes;
                            if(rows==1 || pitch==bytes)MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,reinterpret_cast<const void*>(start),total,cudaMemcpyHostToDevice,cache_->weight_stream_));
                            else MFQ_CUDA_CHECK(cudaMemcpy2DAsync(destination,bytes,reinterpret_cast<const void*>(start),pitch,bytes,rows,cudaMemcpyHostToDevice,cache_->weight_stream_));
                            stats.ram_pcie_bytes+=total;stats.pipeline_direct_ram_bytes+=total;
                            if(p<2)stats.pipeline_gate_up_dma_bytes+=total;else stats.pipeline_down_dma_bytes+=total;
                            ++stats.pipeline_direct_ram_copies;++stats.pipeline_dma_copies;first=last;
                        }
                    }
                }
            }
            if(phase!=1)buffer.gu_ready.record(cache_->weight_stream_);
            if(phase!=0)buffer.down_ready.record(cache_->weight_stream_);
            stats.pipeline_fetch_ns+=elapsed(started);
        };
        if(cache_->ssd_experts()) {
            if(groups)queue_group(0);
            if(groups>1)queue_group(1);
        }
        if(groups)stage_group(0,cache_->ssd_experts()?0:-1);
        if(groups>1 && !cache_->ssd_experts())stage_group(1);
        for(int g=0;g<groups;++g) {
            if(cache_->ssd_experts() && g)stage_group(g,0);
            const auto submit_started=Clock::now();
            std::uint64_t transfer_submit_ns=0;
            auto& buffer=stream.buffers[g%2];const auto& group=work[g];
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,buffer.gu_ready.get(),0));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(buffer.maps.data_ptr(),maps+g*map_stride,
                std::size_t(map_stride)*sizeof(int32_t),cudaMemcpyHostToDevice,stream_));
            auto x=input.index_select(0,token_device.narrow(0,group.begin,group.rows));
            MoeRoutePlan route;
            route.ids=route_device.narrow(0,group.begin,group.rows).reshape({group.rows,1});
            route.n_experts=prefill_group_size;route.map_ready=true;
            int offset=group.route_offset;
            const auto take=[&](int count) {auto value=map_device.narrow(0,offset,count);offset+=count;return value;};
            route.ids_dst=take(group.rows);route.expert_bounds=take(bounds_size);route.counts=take(prefill_group_size);
            route.tile_bounds=take(bounds_size);route.tile_experts=take(group.rows);
            route.mma_tile_bounds=take(bounds_size);route.mma_tile_experts=take(group.rows);
            route.wide_tile_bounds=take(bounds_size);route.wide_tile_experts=take(group.rows);
            route.mma_tile_m=group.mma_tile_m;route.wide_tile_m=group.wide_tile_m;
            const auto projection=[&](int p,const tb::Tensor& x) {
                auto& runtime=buffer.projections[p];
                if(!cache_->pipeline_prefill_activations_)
                    cache_->pipeline_prefill_activations_=std::make_shared<MixedMoeRuntime>();
                for(const auto& geometry:runtime.activation_geometry()) {
                    const MixedMoeTransformKey transform{geometry.transform_block,geometry.transform_seed};
                    const MixedMoeActivationKey key{group.rows,geometry.groups,geometry.gs,x.get_device(),transform};
                    runtime.activation_workspaces.insert_or_assign(key,
                        cache_->pipeline_prefill_activations_->prefill_activation_workspace(
                            x,group.rows,geometry.groups,geometry.gs,transform));
                }
                auto output=runtime.forward(execution.config,execution.kl_mmq,execution.force_moe_prefill_mma_off,
                    execution.force_moe_pool_path,x,route);
                // Variable group densities must not retain one q8 scratch
                // allocation per row count across all prompt chunks.
                runtime.nint_input_plans.clear();runtime.nvq_active_plans.clear();
                runtime.activation_workspaces.clear();return output;
            };
            auto gate=projection(0,x);
            auto up=up_only_ ? gate : combined_gate_up_ ? gate.narrow(-1,ff_,ff_).contiguous() : projection(1,x);
            if(combined_gate_up_)gate=gate.narrow(-1,0,ff_).contiguous();
            auto hidden=up_only_ ? relu_squared_cuda(gate.contiguous()) : activation(gate,up,execution.config.moe_ffn_fused_activation);
            if(cache_->ssd_experts()) {
                const auto transfer_started=Clock::now();
                stage_group(g,1);
                transfer_submit_ns=elapsed(transfer_started);
                static std::atomic_flag reported=ATOMIC_FLAG_INIT;
                if(!reported.test_and_set())std::cerr<<"moe_ssd_prefill_read_overlap enabled=1 phases=gate_up,down packets=2\n";
            }
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,buffer.down_ready.get(),0));
            auto down=projection(2,hidden).reshape({group.rows,width_});
            parts.index_copy_(0,pair_device.narrow(0,group.begin,group.rows),down);
            buffer.used.record(stream_);
            compute_submit_ns+=elapsed(submit_started)-transfer_submit_ns;
            if(g+2<groups) {
                if(cache_->ssd_experts()) {
                    // Retire the H2D source while its GPU copy is in use.
                    // At most two groups hold or are reading CPU payloads.
                    retire_cpu_group(g);queue_group(g+2);
                } else stage_group(g+2);
            }
        }
        MFQ_CUDA_CHECK(cudaEventRecord(cache_->transfer_ready_,cache_->weight_stream_));
        cache_->transfer_ready_recorded_=true;
        auto output=parts.reshape({tokens,routes,width_});
        if(shared.defined())output=moe_weighted_reduce_cuda(output,weights)+shared;
        cache_->record_compute_use();
        cache_->moe_residency_->record_prefill_routes(sources_[0]->layer_id_,
            ids.contiguous().data_ptr<int32_t>(),tokens,routes,stream_);
        if(cache_->ssd_experts())if(const auto* profile=std::getenv("MFQ_PROFILE_SSD_EXPERTS");profile && profile[0]=='1')
            std::fprintf(stderr,"moe_ssd_prefill_host layer=%d groups=%d tokens=%d wall_ms=%.3f plan_ms=%.3f read_join_ms=%.3f buffer_wait_ms=%.3f compute_submit_ms=%.3f fetch_ms=%.3f route_wait_ms=%.3f setup_ms=%.3f\n",
                sources_[0]->layer_id_,groups,tokens,elapsed(started)/1e6,(stats.pipeline_plan_ns-plan_before)/1e6,
                read_join_ns/1e6,buffer_wait_ns/1e6,compute_submit_ns/1e6,(stats.pipeline_fetch_ns-fetch_before)/1e6,
                (stats.pipeline_route_wait_ns-route_before)/1e6,setup_ns/1e6);
        MFQ_CUDA_CHECK(cudaGetLastError());return output;
    }
        struct LayerPrefill {
        std::array<MixedMoeRuntime,3> projections;
        int layer_id=0,slot=0;
        int64_t ram_begin=0,bytes=0;
    };
    std::unique_ptr<LayerPrefill> prefill_layer_;
    bool prefill_async_reported_=false;
    bool prefill_layer_flush() const {
        const auto option=mfq::cuda::runtime_options::moe_prefill_layer_flush();
        return !option || *option!=0;
    }
    bool prefill_layer_async() const {
        const auto option=mfq::cuda::runtime_options::moe_prefill_layer_async();
        return !option || *option!=0;
    }
    bool prefill_layer_phased() const {
        const auto option=mfq::cuda::runtime_options::moe_prefill_layer_phased();
        return !option || *option!=0;
    }

    void retain_prefill_pool(const tb::Tensor& ids) {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const auto retain=mfq::cuda::runtime_options::moe_prefill_pool_retain();
        if((retain && *retain==0) || cache_->pipeline_prefill_pool_)return;
        int supported=0;
        MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrMemoryPoolsSupported,ids.get_device()));
        if(!supported)return;
        cudaMemPool_t pool=nullptr;
        uint64_t previous=0;
        MFQ_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,ids.get_device()));
        MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&previous));
        const auto limit=mfq::cuda::default_context(ids.get_device())->memory_stats().limit;
        uint64_t threshold=limit?limit:std::numeric_limits<uint64_t>::max();
        MFQ_CUDA_CHECK(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&threshold));
        cache_->pipeline_prefill_pool_previous_threshold_=previous;
        cache_->pipeline_prefill_pool_=pool;
#endif
    }

    void handle_prefill_failure() {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        // A quota rejection does not invalidate CUDA execution. Retire the
        // partial prefill's scratch; its request state is discarded by the
        // scheduler. Driver and numerical failures still poison the session.
        try { std::rethrow_exception(std::current_exception()); }
        catch(const std::exception& error) {
            const std::string message=error.what();
            if(dynamic_cast<const MfqResourceExhausted*>(&error) ||
                    message.starts_with("CUDA process VRAM limit exceeded:") ||
                    message.starts_with("CUDA VRAM limit exceeded:") ||
                    message.starts_with("CUDA pinned allocation failed:")) {
                try {
                    release_moe_prefill_buffers(cache_);
                    mfq::cuda::default_context(mfq_current_cuda_device())->reclaim_idle_memory();
                    return;
                } catch(...) {}
            }
        } catch(...) {}
#endif
        failed_=true;
    }

    void initialize_prefill_layer(CudaExecutionContext& execution,const tb::Tensor& ids) {
        if(prefill_layer_)return;
        if(!cache_->pipeline_prefill_layer_layout_)
            throw std::logic_error("whole-layer prefill requires field-major preload");
        const auto layout=cache_->pipeline_prefill_layers_.at(sources_[0]->layer_id_);
        // The predictor's combined NINT image can exceed the target's image.
        // Retire all views before replacing the shared staging allocation.
        const auto layer_bytes=[&](int id) {
            auto bytes=cache_->pipeline_prefill_layers_.at(id).second;
            if(cache_->prefill_loan_)if(auto it=cache_->prefill_loan_->layers.find(id);it!=cache_->prefill_loan_->layers.end())
                bytes=((bytes+255)&~int64_t(255))+int64_t(it->second.bytes);
            return bytes;
        };
        if(cache_->pipeline_prefill_layer_buffer_.defined() && cache_->pipeline_prefill_layer_stride_<layer_bytes(sources_[0]->layer_id_))
            release_moe_prefill_buffers(cache_);
        if(!cache_->pipeline_prefill_layer_buffer_.defined()) {
            int64_t maximum=0;
            // Predictor images are separate from backbone prefill. Their
            // packed projections can be much larger than a backbone layer.
            maximum=layer_bytes(sources_[0]->layer_id_);
            int layer_count=0;
            for(const auto& layer:cache_->pipeline_prefill_layers_)
                if((layer.first<0)==(sources_[0]->layer_id_<0)) {
                    ++layer_count;
                    maximum=std::max(maximum,layer_bytes(layer.first));
                }
            const auto overlap=mfq::cuda::runtime_options::moe_prefill_layer_overlap();
            cache_->pipeline_prefill_layer_overlap_=layer_count>1 && (!overlap || *overlap!=0);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            // Strata sizes its transfer ring together with the compute
            // workspace. A second layer image is optional under pressure.
            const auto context=mfq::cuda::default_context(ids.get_device());
            const auto reserve=mfq::cuda::runtime_options::workspace_reserve_bytes()/2;
            auto available=context->available_memory()+context->workspace_available();
            const auto requested=std::size_t(maximum)*2+reserve;
            if(cache_->pipeline_prefill_layer_overlap_ && available<requested && cache_->reserve_workspace_) {
                cache_->reserve_workspace_(requested);
                available=context->available_memory()+context->workspace_available();
            }
            if(cache_->pipeline_prefill_layer_overlap_ &&
                (available<=reserve || std::size_t(maximum)>((available-reserve)/2))) {
                cache_->pipeline_prefill_layer_overlap_=false;
                std::cerr<<"runtime_prefill_ring slots=1 layer_bytes="<<maximum
                    <<" available_bytes="<<available<<" compute_reserve_bytes="<<reserve<<'\n';
            }
#endif
            cache_->pipeline_prefill_layer_stride_=maximum;
            // The single predictor image is larger than any one expert arena.
            // Allocate Gate/Up and Down separately so the loan can actually
            // host its transfer buffers without another multi-GiB allocation.
            const auto cut=cache_->pipeline_prefill_layer_gate_up_bytes_.at(sources_[0]->layer_id_);
            const bool split=cache_->prefill_loan_ && layer_count==1 && cut>0 && cut<maximum;
            // Borrowed expert storage consists of separate field allocations.
            // Keep ring slots separate so their sum need not fit one region.
            const bool separate_slots=cache_->prefill_loan_ && cache_->pipeline_prefill_layer_overlap_;
            cache_->pipeline_prefill_layer_split_=split?cut:0;
            cache_->pipeline_prefill_layer_buffer_=tb::empty(
                {split?cut:maximum*(cache_->pipeline_prefill_layer_overlap_ && !separate_slots?2:1)},ids.options().dtype(tb::kUInt8));
            if(separate_slots) {
                try {
                    cache_->pipeline_prefill_layer_second_buffer_=tb::empty({maximum},ids.options().dtype(tb::kUInt8));
                } catch(const std::exception& error) {
                    const std::string_view message=error.what();
                    if(!message.starts_with("CUDA process VRAM limit exceeded:") &&
                       !message.starts_with("CUDA VRAM limit exceeded:") &&
                       !message.starts_with("CUDA allocation failed:"))throw;
                    cache_->pipeline_prefill_layer_overlap_=false;
                    std::cerr<<"runtime_prefill_ring slots=1 layer_bytes="<<maximum<<" fragmented_workspace=1\n";
                }
            }
            if(split) {
                cache_->pipeline_prefill_layer_down_buffer_=tb::empty({maximum-cut},ids.options().dtype(tb::kUInt8));
                std::cerr<<"runtime_prefill_ring split=1 gate_up_bytes="<<cut<<" down_bytes="<<maximum-cut<<'\n';
            }
            if(cache_->pipeline_prefill_layer_overlap_) {
                for(auto& event:cache_->pipeline_prefill_layer_ready_)
                    if(!event)MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
                for(auto& event:cache_->pipeline_prefill_layer_gate_up_ready_)
                    if(!event)MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
                for(auto& event:cache_->pipeline_prefill_layer_read_done_)
                    if(!event)MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
                // The allocation is stream ordered; copy readers wait for it.
                MFQ_CUDA_CHECK(cudaEventRecord(cache_->compute_done_,stream_));
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(cache_->weight_stream_,cache_->compute_done_,0));
            }
        }
        auto prepared=std::make_unique<LayerPrefill>();
        prepared->layer_id=sources_[0]->layer_id_;
        prepared->slot=cache_->pipeline_prefill_layer_overlap_?
            int(std::distance(cache_->pipeline_prefill_layers_.begin(),
                cache_->pipeline_prefill_layers_.find(prepared->layer_id)))%2:0;
        prepared->ram_begin=layout.first;prepared->bytes=layout.second;
        for(int p=0;p<3;++p) {
            const auto& source=*sources_[p];auto& runtime=prepared->projections[p];
            std::vector<std::vector<int32_t>> host_maps;
            runtime.n_experts=experts_;runtime.neuron_len=source.cpu_->neuron_len;
            runtime.out_per_expert=source.cpu_->out_per_expert;
            runtime.prefill_only=true;runtime.partial_experts=false;
            for(const auto& cohort:source.cohorts_) {
                std::vector<int32_t> hot_map(experts_,-1),cold_map(experts_,-1);
                for(int e=0;e<experts_;++e)if(source.expert_to_cohort_[e]==cohort.index)
                    hot_map[e]=cohort.arena->book->slot_for({source.id_,cohort.index,e});
                if(cache_->prefill_loan_) {
                    const auto borrowed=cache_->prefill_loan_->cohorts.find({source.id_,cohort.index});
                    if(borrowed!=cache_->prefill_loan_->cohorts.end() && !borrowed->second.experts.empty()) {
                        const auto& loan=borrowed->second;
                        const auto count=int(loan.experts.size());
                        std::vector<int32_t> loan_map(experts_,-1);
                        for(int i=0;i<count;++i) {hot_map[loan.experts[i]]=-1;loan_map[loan.experts[i]]=i;}
                        std::vector<tb::Tensor> fields;
                        for(std::size_t f=0;f<loan.fields.size();++f) {
                            auto shape=cohort.arena->layouts[f].slot_shape;shape[0]*=count;
                            fields.push_back(tb::from_blob(prefill_layer_pointer(prepared->slot,
                                ((layout.second+255)&~int64_t(255))+loan.fields[f],count*cohort.bytes_per_expert[f]),
                                shape,tb::TensorOptions().device(tb::kCUDA).dtype(cohort.arena->layouts[f].scalar_type)));
                        }
                        auto cold=fields_pool(cohort.active,fields,count,runtime.out_per_expert);
                        cold.expert_local=tb::tensor(loan_map).to(tb::kCUDA);
                        runtime.pools.push_back(std::move(cold));host_maps.push_back(std::move(loan_map));
                    }
                }
                auto hot=cohort.active;hot.expert_local=tb::tensor(hot_map).to(tb::kCUDA);
                runtime.pools.push_back(std::move(hot));
                host_maps.push_back(std::move(hot_map));
                const int slots=int(cohort.prefill_ram_experts.size());
                if(!slots)continue;
                std::vector<tb::Tensor> fields;
                for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                    const auto& field=cohort.arena->layouts[f];auto shape=field.slot_shape;shape[0]*=slots;
                    const auto offset=cohort.prefill_ram_fields[f]-layout.first;
                    const auto bytes=slots*cohort.bytes_per_expert[f];
                    if(offset<0 || offset>layout.second || bytes>layout.second-offset)
                        throw std::logic_error("prefill field exceeds its layer image");
                    fields.push_back(tb::from_blob(prefill_layer_pointer(prepared->slot,offset,bytes),shape,
                        tb::TensorOptions().device(tb::kCUDA).dtype(field.scalar_type)));
                }
                for(int slot=0;slot<slots;++slot)cold_map[cohort.prefill_ram_experts[slot]]=slot;
                auto cold=fields_pool(cohort.active,fields,slots,runtime.out_per_expert);
                cold.expert_local=tb::tensor(cold_map).to(tb::kCUDA);runtime.pools.push_back(std::move(cold));
                host_maps.push_back(std::move(cold_map));
            }
            const auto compact=mfq::cuda::runtime_options::moe_prefill_compact_mma();
            const auto host_maps_option=mfq::cuda::runtime_options::moe_prefill_host_maps();
            initialize_mixed_nvq_dispatch(runtime,execution.config,!compact || *compact!=0,
                !host_maps_option || *host_maps_option?&host_maps:nullptr);
        }
        if(swiglu_limit_<=0 && execution.config.moe_ffn_fused_activation && !cache_->pipeline_sigmoid_table_.defined())
            cache_->pipeline_sigmoid_table_=moe_swiglu_sigmoid_table_cuda();
        prefill_layer_=std::move(prepared);
        cache_->pipeline_prefill_release_.push_back([owner=weak_from_this()] {
            if(auto pipeline=owner.lock())pipeline->prefill_layer_.reset();
        });
    }

    uint8_t* prefill_layer_pointer(int slot,int64_t offset,int64_t bytes) {
        const auto split=cache_->pipeline_prefill_layer_split_;
        if(split) {
            if(slot || offset<0 || bytes<0 || offset>cache_->pipeline_prefill_layer_stride_-bytes ||
                    (offset<split && bytes>split-offset))
                throw std::logic_error("prefill field crosses split transfer image");
            if(offset>=split)return cache_->pipeline_prefill_layer_down_buffer_.data_ptr<uint8_t>()+offset-split;
        }
        if(slot==1 && cache_->pipeline_prefill_layer_second_buffer_.defined())
            return cache_->pipeline_prefill_layer_second_buffer_.data_ptr<uint8_t>()+offset;
        return cache_->pipeline_prefill_layer_buffer_.data_ptr<uint8_t>()+
            slot*cache_->pipeline_prefill_layer_stride_+offset;
    }
    MoeExpertCache::PrefillCopySubmission copy_prefill_layer_bytes(int layer_id,int slot,CudaProfiler* profiler) {
        const auto layout=cache_->pipeline_prefill_layers_.at(layer_id);
        const auto copy_stream=cache_->pipeline_prefill_layer_overlap_?cache_->weight_stream_:stream_;
        const auto gate_up_bytes=cache_->pipeline_prefill_layer_split_?cache_->pipeline_prefill_layer_split_:
            cache_->pipeline_prefill_layer_overlap_ && prefill_layer_phased()?
                cache_->pipeline_prefill_layer_gate_up_bytes_.at(layer_id):layout.second;
        if(gate_up_bytes<0 || gate_up_bytes>layout.second)
            throw std::logic_error("prefill Gate/Up copy exceeds layer image");
        if(cache_->pipeline_prefill_layer_overlap_ && cache_->pipeline_prefill_layer_read_recorded_[slot])
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(copy_stream,cache_->pipeline_prefill_layer_read_done_[slot],0));
        int copies=0,direct_copies=0;int64_t direct_bytes=0,loan_bytes=0;
        {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        MfqCudaStreamGuard guard(MfqCudaStream(mfq_current_cuda_device(),copy_stream));
#else
        MfqCudaStreamGuard guard(at::cuda::getStreamFromExternal(copy_stream,mfq_current_cuda_device()));
#endif
        const auto copies_body=[&] {
            const auto* source=cache_->pipeline_ram_complement_->data_ptr<uint8_t>()+layout.first;
            const auto copy=[&](int64_t begin,int64_t bytes) {
                auto* destination=prefill_layer_pointer(slot,begin,bytes);
                // CUDA forbids a single host copy spanning registered and
                // pageable ranges. Split at the registration boundary.
                const auto pinned=std::clamp<int64_t>(cache_->pipeline_ram_registered_bytes_-layout.first-begin,0,bytes);
                if(pinned) {MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,source+begin,std::size_t(pinned),
                    cudaMemcpyHostToDevice,copy_stream));++copies;++direct_copies;direct_bytes+=pinned;}
                if(bytes>pinned) {MFQ_CUDA_CHECK(cudaMemcpyAsync(destination+pinned,source+begin+pinned,
                    std::size_t(bytes-pinned),cudaMemcpyHostToDevice,copy_stream));++copies;}
            };
            copy(0,gate_up_bytes);
            // Loaned experts are a temporary third pool. Publish their rows
            // before Gate/Up readiness; the original cold image remains intact.
            if(cache_->prefill_loan_)if(auto it=cache_->prefill_loan_->layers.find(layer_id);it!=cache_->prefill_loan_->layers.end()) {
                auto& host=cache_->prefill_loan_->stages[slot];
                // This host slot must outlive its preceding DMA. Its GPU slot
                // separately waits for read_done before any overwrite.
                if(cache_->pipeline_prefill_layer_overlap_ && cache_->pipeline_prefill_layer_loaded_[slot])
                    MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->pipeline_prefill_layer_ready_[slot]));
                for(const auto& range:it->second.copies)
                    std::memcpy(host.host.data_ptr<uint8_t>()+range.offset,range.source,range.bytes);
                const auto bytes=it->second.bytes;
                loan_bytes=int64_t(bytes);
                MFQ_CUDA_CHECK(cudaMemcpyAsync(prefill_layer_pointer(slot,((layout.second+255)&~int64_t(255)),bytes),
                    host.host.data_ptr(),bytes,cudaMemcpyHostToDevice,copy_stream));
                ++copies;if(host.pinned){++direct_copies;direct_bytes+=bytes;}
            }
            if(cache_->pipeline_prefill_layer_overlap_)
                MFQ_CUDA_CHECK(cudaEventRecord(cache_->pipeline_prefill_layer_gate_up_ready_[slot],copy_stream));
            copy(gate_up_bytes,layout.second-gate_up_bytes);
            return 0;
        };
        if(profiler)profiler->measure("prefill.expert_h2d",copies_body);else copies_body();
        }
        if(cache_->pipeline_prefill_layer_overlap_)
            MFQ_CUDA_CHECK(cudaEventRecord(cache_->pipeline_prefill_layer_ready_[slot],copy_stream));
        if(cache_->pipeline_prefill_layer_overlap_ && prefill_layer_flush())
            mfq::cuda::check_stream_progress(copy_stream);
        MoeExpertCache::PrefillCopySubmission result;
        result.layer=layer_id;result.slot=slot;result.copies=copies;result.direct_copies=direct_copies;
        result.direct_bytes=direct_bytes;result.total_bytes=layout.second+loan_bytes;
        result.phased=gate_up_bytes<layout.second;result.gate_up_bytes=gate_up_bytes+loan_bytes;
        result.down_bytes=layout.second-gate_up_bytes;return result;
    }
    void collect_prefill_layer_copy(CudaExecutionContext& execution,int slot) {
        auto result=cache_->finish_prefill_copy(slot);
        if(!result.profile)return;
        auto& source=*result.profile;auto& destination=execution.profiler;
        for(const auto& name:source.order) {
            auto [entry,inserted]=destination.stats.try_emplace(name);
            if(inserted)destination.order.push_back(name);
            const auto& value=source.stats.at(name);
            entry->second.calls+=value.calls;entry->second.wall_ms+=value.wall_ms;entry->second.ms+=value.ms;
        }
        destination.pending.insert(destination.pending.end(),
            std::make_move_iterator(source.pending.begin()),std::make_move_iterator(source.pending.end()));
        source.pending.clear();
    }
    void submit_prefill_layer_copy(CudaExecutionContext& execution,int layer_id,int slot,bool background=false) {
        // Another expert batch of the current layer does not consume the
        // next layer. Keep its outstanding upload in the background.
        if(background && cache_->pipeline_prefill_copy_tasks_[slot].valid() &&
                cache_->pipeline_prefill_copy_targets_[slot]==layer_id)return;
        collect_prefill_layer_copy(execution,slot);
        if(cache_->pipeline_prefill_layer_loaded_[slot]==layer_id)return;
        const auto* option=std::getenv("MFQ_PREFILL_CPU_PREFETCH");
        if(background && cache_->pipeline_prefill_layer_overlap_ && prefill_layer_async() &&
                (!option || option[0]!='0')) {
            std::shared_ptr<CudaProfiler> timing;
            if(execution.profiler.enabled && execution.profiler.selected("prefill.expert_h2d")) {
                timing=std::shared_ptr<CudaProfiler>(new CudaProfiler,[](auto* value){value->reset();delete value;});
                timing->enabled=true;timing->graph_events=execution.profiler.graph_events;
            }
            const int device=mfq_current_cuda_device();
            try {
                cache_->pipeline_prefill_copy_tasks_[slot]=std::async(std::launch::async,
                    [owner=shared_from_this(),timing,device,layer_id,slot] {
                        const mfq::cuda::DeviceGuard guard(device);
                        auto result=owner->copy_prefill_layer_bytes(layer_id,slot,timing.get());
                        result.profile=timing;return result;
                    });
                cache_->pipeline_prefill_copy_targets_[slot]=layer_id;
                ++cache_->pipeline_prefill_copy_pending_;++cache_->stats_.pipeline_prefill_cpu_copy_jobs;
                if(!cache_->pipeline_prefill_cpu_reported_) {
                    cache_->pipeline_prefill_cpu_reported_=true;
                    std::cerr<<"runtime_prefill_cpu_prefetch enabled=1 slots=2 lookahead=1\n";
                }
                return;
            }catch(const std::system_error&) {
                // Thread creation can fail under process pressure; the same
                // copy operation and its fences remain valid on this thread.
            }
        }
        cache_->record_prefill_copy(copy_prefill_layer_bytes(layer_id,slot,&execution.profiler));
    }

    tb::Tensor whole_layer_prefill(CudaExecutionContext& execution,const tb::Tensor& input,
            const tb::Tensor& ids,const tb::Tensor& weights) {
        // Bound submission to two layers of lookahead. The CPU only waits for
        // the old reader of this slot, while current GR/attention remain queued.
        // Copy-stream fences protect the other slot until its reader retires.
        const auto reuse_option=mfq::cuda::runtime_options::moe_prefill_layer_reuse_async();
        const bool same_layer=cache_->pipeline_prefill_layer_last_served_==sources_[0]->layer_id_ &&
            (!reuse_option || *reuse_option!=0);
        if(!cache_->pipeline_prefill_layer_buffer_.defined() || !cache_->pipeline_prefill_layer_overlap_ ||
                (!prefill_layer_async() && !same_layer))MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));
        else if(prefill_layer_async() && !prefill_async_reported_) {
            std::cerr<<"runtime_prefill_async_dispatch layer="<<sources_[0]->layer_id_
                <<" tokens="<<input.size(0)<<" overlap=1\n";
            prefill_async_reported_=true;
        }
        if(prefill_stream_)return streamed_prefill(execution,input,ids,weights);
        try { initialize_prefill_layer(execution,ids); }
        catch(const MfqResourceExhausted& error) {
            // Grouped/decode dispatch sees the normal resident-slot books.
            // It cannot run while those slots hold a prefill workspace.
            if(cache_->prefill_loan_)throw;
            // Retire an oversized whole-layer image before switching to the
            // bounded expert-group ring. Keep that ring for subsequent chunks.
            release_moe_prefill_buffers(cache_);
            std::cerr<<"runtime_prefill_ring grouped_fallback=1 layer="<<sources_[0]->layer_id_
                <<" reason="<<error.what()<<'\n';
            return streamed_prefill(execution,input,ids,weights);
        }
        auto& layer=*prefill_layer_;
        if(cache_->pipeline_prefill_layer_overlap_ && prefill_layer_async() && !same_layer &&
                cache_->pipeline_prefill_layer_read_recorded_[layer.slot])
            MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->pipeline_prefill_layer_read_done_[layer.slot]));
        submit_prefill_layer_copy(execution,layer.layer_id,layer.slot);
        if(cache_->pipeline_prefill_layer_overlap_) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,cache_->pipeline_prefill_layer_gate_up_ready_[layer.slot],0));
            const auto next=std::next(cache_->pipeline_prefill_layers_.find(layer.layer_id));
            if(next!=cache_->pipeline_prefill_layers_.end() && (next->first<0)==(layer.layer_id<0) &&
                cache_->prefill_layer_active(next->first))
                submit_prefill_layer_copy(execution,next->first,layer.slot^1,true);
        }
        ++cache_->stats_.pipeline_serves;
        auto shared=shared_?shared_(execution,input):tb::Tensor{};
        auto route=build_moe_route_plan(ids,experts_);
        const auto projection=[&](int p,const tb::Tensor& x) {
            auto& runtime=layer.projections[p];
            if(!cache_->pipeline_prefill_activations_)
                cache_->pipeline_prefill_activations_=std::make_shared<MixedMoeRuntime>();
            const int rows=int(x.numel()/x.size(-1));
            // Direct FP16 MMA consumes x; allocate shared Q8 scratch only when
            // the integer path actually needs it.
            if(!execution.config.moe_prefill_mma || execution.force_moe_prefill_mma_off)
            for(const auto& geometry:runtime.activation_geometry()) {
                const MixedMoeTransformKey transform{geometry.transform_block,geometry.transform_seed};
                const MixedMoeActivationKey key{rows,geometry.groups,geometry.gs,x.get_device(),transform};
                runtime.activation_workspaces.insert_or_assign(key,
                    cache_->pipeline_prefill_activations_->prefill_activation_workspace(
                        x,rows,geometry.groups,geometry.gs,transform));
            }
            auto output=execution.profiler.measure(p==0?"prefill.expert_gate":p==1?"prefill.expert_up":"prefill.expert_down",[&] {
                return runtime.forward(execution.config,execution.kl_mmq,execution.force_moe_prefill_mma_off,
                    execution.force_moe_pool_path,x,route);
            });
            runtime.activation_workspaces.clear();return output;
        };
        auto gate=projection(0,input);
        auto up=up_only_ ? gate : combined_gate_up_ ? gate.narrow(-1,ff_,ff_).contiguous() : projection(1,input);
        if(combined_gate_up_)gate=gate.narrow(-1,0,ff_).contiguous();
        auto hidden=up_only_ ? relu_squared_cuda(gate.contiguous()) : activation(gate,up,execution.config.moe_ffn_fused_activation);
        if(cache_->pipeline_prefill_layer_overlap_) {
            // Submit Gate/Up independently of Down's completion on WDDM.
            if(prefill_layer_flush())mfq::cuda::check_stream_progress(stream_);
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,cache_->pipeline_prefill_layer_ready_[layer.slot],0));
        }
        auto output=projection(2,hidden);
        if(shared.defined())output=moe_weighted_reduce_cuda(output,weights)+shared;
        if(cache_->pipeline_prefill_layer_overlap_) {
            MFQ_CUDA_CHECK(cudaEventRecord(cache_->pipeline_prefill_layer_read_done_[layer.slot],stream_));
            cache_->pipeline_prefill_layer_read_recorded_[layer.slot]=true;
        }
        if(cache_->pipeline_prefill_layer_overlap_ && prefill_layer_flush())
            mfq::cuda::check_stream_progress(stream_);
        cache_->pipeline_prefill_layer_last_served_=layer.layer_id;
        cache_->moe_residency_->record_prefill_routes(sources_[0]->layer_id_,
            ids.contiguous().data_ptr<int32_t>(),int(ids.size(0)),int(ids.size(1)),stream_);
        return output;
    }

    bool pipeline_source(const MoeCachedSource& source) const {
        for(const auto& bundle:cache_->pipeline_bundles_)
            if(std::find(bundle.begin(),bundle.end(),source.id_)!=bundle.end())return true;
        return false;
    }
    MoeGpuArena& stage(int projection,MoeCachedCohort& cohort,int requested_slots=moe_decode_transfer_slots(8,10),bool packed=false) {
        if(single_up() && projection==1)projection=0;
        if(packed && moe_packed_transfer_stages_enabled()) {
            const bool predictor=sources_[0]->layer_id_<0 && moe_predictor_transfer_cache_enabled();
            const auto key=moe_packed_transfer_prefix(projection,predictor)+std::to_string(requested_slots);
            auto& arena=cache_->pipeline_stages_[key];
            if(arena)return *arena;
            arena=std::make_unique<MoeGpuArena>();arena->signature=key;
            for(const auto& bundle:cache_->pipeline_bundles_) {
                if(projection==1 && bundle[0]==bundle[1])continue;
                const auto& source=cache_->sources_[bundle[projection]];
                if(moe_predictor_transfer_cache_enabled() && (source->layer_id_<0)!=predictor)continue;
                arena->slots=std::max(arena->slots,int(source->n_experts()));
                for(const auto& c:source->cohorts_)
                    arena->slot_bytes=std::max(arena->slot_bytes,moe_packed_slot_bytes(c.bytes_per_expert));
            }
            arena->slots=std::min(arena->slots,requested_slots);
            cache_->reserve_workspace(std::size_t(arena->slots)*arena->slot_bytes+(std::size_t(64)<<20));
            arena->packed=tb::empty({arena->slots*arena->slot_bytes},
                tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
            arena->book=std::make_unique<mfq::MoeCacheSlotBook>(arena->slots);
            std::cerr<<"moe_packed_transfer_arena projection="<<projection<<" predictor="<<predictor<<" slots="<<arena->slots
                <<" bytes="<<arena->slots*arena->slot_bytes<<'\n';
            return *arena;
        }
        const auto key=std::to_string(projection)+":"+cohort.arena->signature+":"+std::to_string(requested_slots);
        auto& arena=cache_->pipeline_stages_[key];
        if(arena)return *arena;
        arena=std::make_unique<MoeGpuArena>();arena->signature=key;arena->slot_bytes=cohort.arena->slot_bytes;
        arena->layouts=cohort.arena->layouts;
        // Batched decode needs at most 32 rows of routes. Keep reusable
        // slots, without allocating an entire expert cohort for one token.
        // Prefill uses a separate arena so captured addresses stay unchanged.
        for(const auto& source:cache_->sources_)if(pipeline_source(*source))for(const auto& c:source->cohorts_)
            if(c.arena->signature==cohort.arena->signature)arena->slots=std::max(arena->slots,c.cpu->local_experts);
        arena->slots=std::min(arena->slots,requested_slots);
        // First decode can create these arenas after spare VRAM was lent to
        // experts. Return the borrowed pages before allocating their fields.
        cache_->reserve_workspace(std::size_t(arena->slots)*arena->slot_bytes+(std::size_t(64)<<20));
        if(std::getenv("MFQ_TRACE_RESIDENCY_MEMORY"))
            std::cerr<<"moe_transfer_arena key="<<key<<" slots="<<arena->slots
                <<" bytes="<<arena->slots*arena->slot_bytes<<'\n';
        for(const auto& layout:arena->layouts) {
            auto shape=layout.slot_shape;shape[0]*=arena->slots;
            arena->fields.push_back(tb::empty(shape,tb::TensorOptions().device(tb::kCUDA).dtype(layout.scalar_type)));
        }
        return *arena;
    }
    void bind_wire(Cell& cell) {
        if(!cache_->pipeline_wire_gpu_.defined() || cache_->pipeline_wire_gpu_.numel()<cell.wire_capacity) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(cache_->weight_stream_));
            cache_->pipeline_wire_gpu_={};
            cache_->reserve_workspace(std::size_t(cell.wire_capacity)+(std::size_t(64)<<20));
            cache_->pipeline_wire_gpu_=tb::empty({cell.wire_capacity},cell.input.options().dtype(tb::kUInt8));
            MFQ_CUDA_CHECK(cudaMemsetAsync(cache_->pipeline_wire_gpu_.data_ptr(),0,32,stream_));
        }
        cell.wire=cache_->pipeline_wire_gpu_;
    }
    void ensure_host_stage(int64_t bytes) {
        if(cache_->pipeline_host_stage_.defined() && cache_->pipeline_host_stage_.numel()>=bytes)return;
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(cache_->weight_stream_));
        // Drain DMA consumers before replacing their pinned source.
        for(auto& batch:cache_->pipeline_dma_batches_)batch.reset();
        cache_->pipeline_host_stage_={};cache_->pipeline_host_stage_alias_=nullptr;
        cache_->pipeline_host_stage_=tb::empty({bytes},
            tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
    }
    void initialize(Cell& cell,CudaExecutionContext& execution,bool window=false) {
        cell.in_window=window;
        const auto prefill_graph=mfq::cuda::runtime_options::moe_prefill_graph();
        cell.batch_decode=cell.tokens<=8 || (cell.tokens<=32 && execution.continuous_batch_cache_serial);
        // SSD payloads are leased per demand. Eager prefill must bind this
        // call's fields, rather than retain a graph over an evicted payload.
        cell.graph_replay=window || cell.batch_decode ||
            (!cache_->ssd_experts() && (!prefill_graph || *prefill_graph!=0));
        if(!cell.graph_replay) {
            cell.eager_done=std::make_unique<mfq::cuda::Event>(cudaEventDisableTiming);
            const auto key=std::to_string(cell.tokens)+":"+std::to_string(cell.routes)+":"+std::to_string(width_);
            auto& workspace=cache_->pipeline_prefill_host_workspaces_[key];
            if(!workspace)workspace=std::make_shared<MoeFfnPrefillHostWorkspace>(cell.tokens,cell.routes,width_);
            // The next route publication follows the previous merge on the
            // compute stream. Host input reads and CPU writes start only after
            // that publication, so layers cannot use this scratch concurrently.
            cell.x.borrow(workspace->x);cell.host_ids.borrow(workspace->ids);cell.cpu.borrow(workspace->cpu);
        }
        cell.ready_writer=window?stream_flag_writer():nullptr;
        cell.two_stage=execution.config.moe_two_stage_ffn && cell.batch_decode && cell.routes+(shared_?1:0)<=32 &&
            execution.kl_mmq.mode==KlMmqMode::Default;
        for(const auto& source:sources_)for(const auto& cohort:source->cohorts_)
            cell.two_stage=cell.two_stage && (cohort.active.family==MixedMoeFamily::Nint ||
                cohort.active.family==MixedMoeFamily::Nvq || cohort.active.family==MixedMoeFamily::Mxfp4);
        if(shared_) {
            cell.two_stage=cell.two_stage && bool(shared_weights_.gate);
            for(const auto& weight:shared_weights_.projections)
                cell.two_stage=cell.two_stage && weight && weight->is_nint() && !weight->tensor_parallel() && !weight->nint.q8_zero;
        }
        const auto reuse=mfq::cuda::runtime_options::moe_plan_reuse();
        cell.reuse_plan=cell.two_stage && (!reuse || *reuse!=0);
        const auto check=mfq::cuda::runtime_options::moe_plan_reuse_check();
        cell.check_reused_views=cell.reuse_plan && check && *check!=0;
        const auto trace=mfq::cuda::runtime_options::trace_mfe_dispatch();
        if(trace && *trace!=0)
            std::fprintf(stderr,"moe_plan_reuse_runtime tokens=%d routes=%d reuse=%d check=%d\n",
                cell.tokens,cell.routes,int(cell.reuse_plan),int(cell.check_reused_views));
        // A fixed full PCIe quota schedules every cold expert on the GPU.
        // CPU dispatch replays retain their original wait, even with that quota.
        cell.skip_cpu_wait=cell.two_stage && mfq::cuda::runtime_options::skip_gpu_only_cpu_wait() &&
            cache_->config_.moe_ram_pcie_fraction && cache_->ram_pcie_fraction_==1.0 &&
            std::all_of(cache_->pipeline_dispatch_replay_.begin(),cache_->pipeline_dispatch_replay_.end(),
                [](const auto& record){return record.cpu.empty();});
        // The same fixed-quota proof also excludes CPU calibration and CPU
        // expert work. Publish only IDs so the DMA planner can start sooner.
        const auto* input_export=std::getenv("MFQ_MOE_SKIP_GPU_ONLY_INPUT_COPY");
        cell.export_cpu_input=!cell.skip_cpu_wait || (input_export && input_export[0]=='0');
        if(trace && *trace!=0)std::fprintf(stderr,"moe_input_export tokens=%d routes=%d cpu_input=%d\n",
            cell.tokens,cell.routes,int(cell.export_cpu_input));
        // With a fixed GPU quota no dispatch decision consumes the measured
        // arithmetic rate. Keep timing whenever detailed DMA profiling needs
        // these event origins, or a dispatch replay can contain CPU work.
        cell.skip_gpu_timing=cell.two_stage && mfq::cuda::runtime_options::skip_fixed_gpu_timing() &&
            cache_->config_.moe_ram_pcie_fraction && cache_->ram_pcie_fraction_==1.0 &&
            !mfq::cuda::runtime_options::moe_diagnostics().dma &&
            std::all_of(cache_->pipeline_dispatch_replay_.begin(),cache_->pipeline_dispatch_replay_.end(),
                [](const auto& record){return record.cpu.empty();});
        if(!cell.skip_gpu_timing)cell.timing.initialize();
        cell.gpu_resident_dispatch=execution.config.moe_gpu_resident_dispatch && cell.tokens<=8;
        cell.fused_activation=swiglu_limit_<=0 && execution.config.moe_ffn_fused_activation;
        if((cell.fused_activation || cell.two_stage) && !cache_->pipeline_sigmoid_table_.defined())
            cache_->pipeline_sigmoid_table_=moe_swiglu_sigmoid_table_cuda();
        if(cell.two_stage) {
            std::array<std::vector<int32_t>,3> mapping;
            for(int p=0;p<3;++p) {
                auto& runtime=cell.fused_projections[p];const auto& source=*sources_[p];
                runtime.n_experts=experts_;runtime.out_per_expert=source.cpu_->out_per_expert;
                runtime.neuron_len=source.cpu_->neuron_len;runtime.partial_experts=true;
                for(const auto& cohort:source.cohorts_)runtime.pools.push_back(cohort.active);
                mapping[p]=source.expert_to_cohort_;
            }
            std::array<const NintWeight*,3> shared{};
            if(shared_) {
                for(int p=0;p<3;++p)shared[p]=&shared_weights_.projections[p]->nint;
                cell.shared_gate=shared_weights_.gate(execution,cell.input).contiguous();
            }
            cell.fused=std::make_unique<MfeFfnRuntime>(std::array<MixedMoeRuntime*,3>{&cell.fused_projections[0],
                &cell.fused_projections[1],&cell.fused_projections[2]},cell.input,cell.ids,cell.weights,
                cache_->pipeline_sigmoid_table_,shared,cell.shared_gate,width_,ff_,&mapping,!shared_,combined_gate_up_,
                execution.continuous_batch_cache_serial && cell.tokens==8?8:0,swiglu_limit_,up_only_,false);
            const auto* nint_gate=std::getenv("MFQ_SHARED_GATE_NINT");
            const auto* reuse_gate=std::getenv("MFQ_SHARED_GATE_REUSE");
            const auto fused_gate=mfq::cuda::runtime_options::shared_gate_fused();
            if(cell.tokens<=6 && shared_weights_.scalar_gate && shared_weights_.scalar_gate->is_nint() &&
                    !shared_weights_.scalar_gate->tensor_parallel() && (!fused_gate || *fused_gate) &&
                    (!nint_gate || nint_gate[0]!='0') && (!reuse_gate || reuse_gate[0]!='0')) {
                const auto& w=shared_weights_.scalar_gate->nint;
                if(w.out==1 && w.aligned_q8 && !w.q8_zero && w.gs==48 && w.ng<=64 && w.neuron_len==width_)
                    std::tie(cell.shared_gate_quantized,cell.shared_gate_scales)=cell.fused->reserve_input_quantization(int(w.ng),int(w.gs));
            }
            cell.transfer_index=std::make_unique<Mapped>(std::size_t(experts_)*sizeof(int32_t));
            std::memset(cell.transfer_index->host,0xff,cell.transfer_index->bytes);
            cell.transfer_descriptors=std::make_unique<Mapped>(std::size_t(cell.entries)*3*sizeof(mfq::cuda::MfePackedProjection));
            cell.fused->asynchronous(cell.kind.d<int32_t>(),cell.flags.d<uint32_t>()+16,cell.flags.d<uint32_t>()+32,
                cell.flags.d<uint32_t>()+48,cell.flags.d<uint32_t>()+64,cell.transfer_index->d<int32_t>(),cell.cpu.device);
            const auto* route_packet=std::getenv("MFQ_MFE_TRANSFER_ROUTE_PACKET");
            if(cell.fused->batch().resident_plan_overlap && (!route_packet || route_packet[0]!='0')) {
                cell.transfer_route_plan=std::make_unique<Mapped>(std::size_t(cell.entries)*2*sizeof(int32_t));
                cell.fused->bind_transfer_route_plan(cell.transfer_route_plan->d<int32_t>());
            }
            if(cell.reuse_plan)for(int p=0;p<3;++p) {
                const auto& source=*sources_[p];
                auto& templates=cell.transfer_templates[p];templates.reserve(source.cohorts_.size());
                for(std::size_t c=0;c<source.cohorts_.size();++c) {
                    const auto found=std::find(source.expert_to_cohort_.begin(),source.expert_to_cohort_.end(),int(c));
                    if(found==source.expert_to_cohort_.end()){templates.emplace_back();continue;}
                    const int e=int(found-source.expert_to_cohort_.begin());const auto& cohort=source.cohorts_[c];
                    TransferTemplate prepared{cell.fused->expert_view(single_up() && p==0?1:p,e,cohort.bytes_per_expert,-1)};
                    const auto fields=moe_cache_fields(cohort.active);
                    if(fields.size()!=cohort.bytes_per_expert.size())throw std::logic_error("MFE transfer template field count differs");
                    for(std::size_t f=0;f<fields.size();++f)
                        prepared.bases[f]=fields[f].defined()?static_cast<const uint8_t*>(fields[f].data_ptr()):nullptr;
                    templates.push_back(prepared);
                }
            }
        }
        cell.transfer_cache=cell.two_stage && cache_->pipeline_transfer_cache_ &&
            (cache_->pipeline_ram_registered_bytes_>0 || cache_->ssd_experts());
        cell.mapped_copy=cell.transfer_cache && cache_->pipeline_mapped_copy_;
        cell.phased_transfer=cell.transfer_cache && !cell.mapped_copy &&
            execution.config.moe_ffn_transfer_phases.value_or(mfq::cuda::runtime_options::phased_transfer());
        cell.mapped_overlap=cell.mapped_copy && mfq::cuda::runtime_options::mapped_copy_overlap();
        const auto* native_dma=std::getenv("MFQ_MOE_NATIVE_DMA_BATCH");
#if defined(_WIN32)
        const int native_default=1;
#else
        const int native_default=0;
#endif
        const int native_requested=native_dma?std::clamp(std::atoi(native_dma),0,2):native_default;
        if(cell.graph_replay && cell.phased_transfer && native_requested && mfq::cuda::NativeDmaBatch::available())
            cell.native_dma_mode=native_requested;
        const auto* direct_read=std::getenv("MFQ_MOE_KV_DIRECT_READ");
        const bool direct_enabled=direct_read?direct_read[0]!='0':cell.native_dma_mode>0;
        const auto* poll_ready=std::getenv("MFQ_MOE_POLL_TRANSFER_READY");
        const bool poll_enabled=poll_ready?poll_ready[0]=='1':(native_default && cell.native_dma_mode && direct_enabled);
        if(cell.graph_replay && cell.phased_transfer && cell.skip_cpu_wait && poll_enabled) {
            cell.transfer_notification=std::make_unique<mfq::cuda::Event>();
        }
        // Eight-row tiles need eight route lists. Large windows trade some
        // cached history for smaller staging arenas, retaining every route.
        // Explicit independent-request tiles bound the live route union. Keep
        // one sixteen-row arena for their 9–16 row tails instead of reserving
        // thirty-two route lists. Untiled callers retain their shared capacity.
        const int decode_route_rows=cell.tokens>8?
            (execution.continuous_batch_ffn_tiles>0 && cell.tokens<=16?16:32):8;
        const auto* wire_fallback=std::getenv("MFQ_MOE_WIRE_FALLBACK");
        const auto* packet_option=std::getenv("MFQ_MOE_NATIVE_DIRECT_PACKET");
        const auto* host_packet_option=std::getenv("MFQ_MOE_KV_HOST_PACKET");
        const auto* fused_copy_option=std::getenv("MFQ_MOE_FUSED_TRANSFER_COPY");
        cell.wire_fallback=(!wire_fallback || wire_fallback[0]!='0') && cell.tokens<=8 &&
            cell.graph_replay && cell.phased_transfer && cell.skip_cpu_wait && cell.native_dma_mode &&
            direct_enabled && bool(cache_->kv_expert_loan_) && bool(cell.transfer_route_plan) &&
            cell.fused->batch().resident_plan_overlap &&
            (!packet_option || packet_option[0]!='0') &&
            (!host_packet_option || host_packet_option[0]=='1') &&
            (!fused_copy_option || fused_copy_option[0]!='0') &&
            (!mfq::cuda::runtime_options::router_lookahead() || cache_->ssd_experts()) && !cache_->pipeline_prefetch_host_stage_.defined();
        if(cell.wire_fallback)cache_->prepare_direct_transfer_cache();
        if(cell.transfer_cache)for(int p=0;p<3;++p)for(auto& cohort:sources_[p]->cohorts_) {
            // Share one capacity across M=9..32; changing concurrency must not
            // retain a separate weight staging arena for every batch size.
            if(cell.wire_fallback){cell.binding[p].stages.push_back(nullptr);continue;}
            auto& arena=stage(p,cohort,moe_decode_transfer_slots(decode_route_rows,cell.routes),true);
            if(!arena.book)arena.book=std::make_unique<mfq::MoeCacheSlotBook>(arena.slots);
            cell.binding[p].stages.push_back(&arena);
        }
        if(!cell.two_stage)for(int projection=0;projection<3;++projection) {
            auto& source=*sources_[projection];auto& binding=cell.binding[projection];
            for(auto* runtime:{&binding.hot,&binding.transfer}) {
                runtime->n_experts=experts_;runtime->out_per_expert=source.cpu_->out_per_expert;
                runtime->neuron_len=source.cpu_->neuron_len;runtime->partial_experts=true;
            }
            for(auto& cohort:source.cohorts_) {
                const int slots=cell.batch_decode?cell.entries:experts_;
                auto& temporary=stage(projection,cohort,slots);binding.stages.push_back(&temporary);
                auto hot=cohort.active;hot.expert_local=tb::full({experts_},-1,cell.ids.options());
                auto transfer=fields_pool(cohort.active,temporary.fields,temporary.slots,source.cpu_->out_per_expert);
                transfer.expert_local=tb::full({experts_},-1,cell.ids.options());
                auto retained=cohort.active;retained.expert_local=tb::full({experts_},-1,cell.ids.options());
                binding.hot.pools.push_back(std::move(hot));binding.transfer.pools.push_back(std::move(transfer));
                binding.transfer.pools.push_back(std::move(retained));
                binding.owner.push_back(tb::tensor(cohort.expert_to_local).to(tb::kCUDA));
            }
            const auto compact_mma=mfq::cuda::runtime_options::moe_prefill_compact_mma();
            const bool dense_prefill=cell.tokens>8 && compact_mma && *compact_mma!=0;
            initialize_mixed_nvq_dispatch(binding.hot,execution.config,dense_prefill);
            initialize_mixed_nvq_dispatch(binding.transfer,execution.config,dense_prefill);
            if(!cell.graph_replay) {
                if(!cache_->pipeline_prefill_activations_)
                    cache_->pipeline_prefill_activations_=std::make_shared<MixedMoeRuntime>();
                const int rows=projection==2?cell.entries:cell.tokens;
                for(auto* runtime:{&binding.hot,&binding.transfer})
                    for(const auto& g:runtime->activation_geometry()) {
                        const MixedMoeTransformKey transform{g.transform_block,g.transform_seed};
                        const MixedMoeActivationKey key{rows,g.groups,g.gs,cell.input.get_device(),transform};
                        runtime->activation_workspaces.insert_or_assign(key,
                            cache_->pipeline_prefill_activations_->activation_workspace(
                                cell.input,rows,g.groups,g.gs,transform));
                    }
            }
        }
        cell.shared_input=!cell.two_stage && execution.config.split_moe_activation_reuse && cell.tokens<=8 &&
            execution.kl_mmq.mode==KlMmqMode::Default && !(execution.config.moe_prefill_mma &&
                !execution.force_moe_prefill_mma_off && cell.tokens>=execution.config.moe_prefill_mma_min_tokens);
        if(cell.shared_input) {
            std::vector<MoeActivationGeometry> geometries;
            for(int projection=0;projection<2;++projection)for(const auto& geometry:cell.binding[projection].hot.activation_geometry()) {
                if(geometry.transform_block || geometry.groups<=0 || geometry.gs<=0 || geometry.gs>64)
                    throw std::runtime_error("shared FFN input needs canonical unrotated NINT/NVQ groups");
                if(std::find(geometries.begin(),geometries.end(),geometry)==geometries.end())geometries.push_back(geometry);
            }
            std::vector<int64_t> descriptors;
            for(const auto& geometry:geometries) {
                auto workspace=cell.binding[0].hot.activation_workspace(cell.input,cell.tokens,geometry.groups,geometry.gs,{});
                const MixedMoeActivationKey key{cell.tokens,geometry.groups,geometry.gs,cell.input.get_device(),{}};
                for(int projection=0;projection<2;++projection)for(auto* runtime:{&cell.binding[projection].hot,&cell.binding[projection].transfer})
                    runtime->activation_workspaces.insert_or_assign(key,workspace);
                descriptors.insert(descriptors.end(),{reinterpret_cast<int64_t>(workspace.qx.data_ptr()),
                    reinterpret_cast<int64_t>(workspace.xscale.data_ptr()),geometry.groups,geometry.gs});
                cell.shared_input_groups+=geometry.groups;
            }
            cell.shared_input=!descriptors.empty();
            if(cell.shared_input)
                cell.shared_input_descriptors=tb::tensor(descriptors).reshape({int64_t(geometries.size()),4}).to(tb::kCUDA);
        }
        std::vector<int64_t> dispatch;
        const auto pointer=[&](const auto* p){dispatch.push_back(reinterpret_cast<int64_t>(p));};
        if(!cell.two_stage)for(int i=0;i<3;++i)for(std::size_t c=0;c<sources_[i]->cohorts_.size();++c) {
            const auto& binding=cell.binding[i];
            pointer(sources_[i]->cohorts_[c].active.expert_local.data_ptr<int32_t>());
            pointer(binding.owner[c].data_ptr<int32_t>());
            pointer(cell.local.d<int32_t>()+i*experts_);
            pointer(binding.hot.pools[c].expert_local.data_ptr<int32_t>());
            pointer(binding.transfer.pools[2*c].expert_local.data_ptr<int32_t>());
            pointer(binding.transfer.pools[2*c+1].expert_local.data_ptr<int32_t>());
        }
        if(!cell.two_stage)cell.dispatch_pointers=tb::tensor(dispatch).to(tb::kCUDA);
        cell.device_abort=tb::empty({1},cell.input.options().dtype(tb::kInt32));
        cell.device_kind=tb::empty({experts_},cell.input.options().dtype(tb::kInt32));
        // Predictor experts can be much larger than backbone experts. Size
        // this invocation from its own projections before sharing the arenas;
        // a one-row predictor must not multiply its geometry by 32 decode rows.
        int64_t route_bytes=0,route_fields=0;
        for(int p=0;p<3;++p) {
            if(single_up() && p==1)continue;
            int64_t bytes=0,fields=0;
            for(const auto& cohort:sources_[p]->cohorts_) {
                const auto n=static_cast<int64_t>(cohort.bytes_per_expert.size());
                int64_t payload=16*n;
                for(const auto count:cohort.bytes_per_expert)payload+=count;
                bytes=std::max(bytes,payload);fields=std::max(fields,n);
            }
            route_bytes+=bytes;route_fields+=fields;
        }
        cache_->pipeline_wire_route_bytes_=std::max(cache_->pipeline_wire_route_bytes_,route_bytes);
        cache_->pipeline_wire_route_fields_=std::max(cache_->pipeline_wire_route_fields_,route_fields);
        auto& wire_geometry=cache_->pipeline_wire_geometry_[sources_[0]->layer_id_<0];
        if(!wire_geometry.first) {
            std::unordered_map<int,int64_t> layer_bytes,layer_fields;
            for(const auto& source:cache_->sources_)if(source->quant_source_ && pipeline_source(*source) &&
                    (source->layer_id_<0)==(sources_[0]->layer_id_<0)) {
                int64_t bytes=0,fields=0;
                for(const auto& cohort:source->cohorts_) {
                    const auto n=int64_t(cohort.bytes_per_expert.size());int64_t payload=16*n;
                    for(const auto count:cohort.bytes_per_expert)payload+=count;
                    bytes=std::max(bytes,payload);fields=std::max(fields,n);
                }
                layer_bytes[source->layer_id_]+=bytes;layer_fields[source->layer_id_]+=fields;
            }
            for(const auto& [layer,bytes]:layer_bytes)wire_geometry.first=std::max(wire_geometry.first,bytes);
            for(const auto& [layer,fields]:layer_fields)wire_geometry.second=std::max(wire_geometry.second,fields);
        }
        const auto host_routes=std::min(experts_,cell.batch_decode?cell.entries:std::max(cell.entries,8*cell.routes));
        const auto host_header=(32+24*host_routes*route_fields+15)&~int64_t(15);
        int64_t host_payload=host_routes*route_bytes;
        const auto compact_host=mfq::cuda::runtime_options::moe_compact_host_stage();
        if(cell.transfer_cache && cache_->pipeline_prefill_layer_layout_ && compact_host && *compact_host) {
            // RAM expert leases are stable after full preload. Inspect their
            // actual fields once: retained warm copies may live outside the
            // registered cold arena. Only pageable fields and packed metadata
            // need a second pinned copy; direct payloads leave no host holes.
            std::vector<int64_t> staged(experts_);
            const bool pack_small=mfq::cuda::runtime_options::ram_field_packing();
            const auto ram_begin=reinterpret_cast<uintptr_t>(cache_->pipeline_ram_complement_->data_ptr());
            const auto ram_bytes=uint64_t(cache_->pipeline_ram_registered_bytes_);
            for(int p=0;p<3;++p) {
                if(single_up() && p==1)continue;
                auto& source=*sources_[p];
                auto [found,fresh]=cache_->pipeline_host_staged_expert_bytes_.try_emplace(source.id_);
                auto& costs=found->second;
                if(fresh) {
                    costs.resize(experts_);
                    for(int e=0;e<experts_;++e) {
                        if(source.quant_source_->gpu_resident(e))continue;
                        auto lease=source.quant_source_->acquire_expert(e);
                        for(const auto& field:moe_cache_fields(lease->weights)) {
                            const auto bytes=tensor_nbytes(field);if(!bytes)continue;
                            const auto address=reinterpret_cast<uintptr_t>(field.data_ptr());
                            if((pack_small && bytes<=16*1024) || address<ram_begin ||
                                    address-ram_begin>ram_bytes || uint64_t(bytes)>ram_bytes-(address-ram_begin))
                                costs[e]+=bytes+15;
                        }
                    }
                }
                for(int e=0;e<experts_;++e)staged[e]+=costs[e];
            }
            std::sort(staged.begin(),staged.end(),std::greater<int64_t>{});
            host_payload=0;for(int64_t i=0;i<host_routes;++i)host_payload+=staged[i];
        }
        const auto host_bytes=host_header+host_payload+32;
        cell.host_stage_bytes=host_bytes;ensure_host_stage(host_bytes);
        if(cell.batch_decode) {
            const auto routes=std::min(cell.entries,experts_);
            cell.wire_header_bytes=(32+24*routes*route_fields+15)&~int64_t(15);
            if(cell.phased_transfer && cache_->kv_expert_loan_) {
                const auto bytes=2*(32+32*routes*route_fields);
                cell.kv_loan_table=std::make_unique<Mapped>(std::size_t(bytes));
                cell.kv_loan_snapshot=tb::empty({bytes},cell.input.options().dtype(tb::kUInt8));
                const auto* fuse=std::getenv("MFQ_MOE_FUSED_TRANSFER_COPY");
                cell.fused_transfer_copy=(fuse?fuse[0]!='0':cell.tokens<=8) && bool(cell.transfer_route_plan);
                const auto* host_packet=std::getenv("MFQ_MOE_KV_HOST_PACKET");
                cell.host_cache_packet=cell.graph_replay && cell.skip_cpu_wait && cell.fused_transfer_copy &&
                    cell.native_dma_mode && direct_enabled && (!host_packet || host_packet[0]=='1');
                const auto* direct_packet=std::getenv("MFQ_MOE_NATIVE_DIRECT_PACKET");
                // The packet owns the union of routed experts, not one row's
                // routes. Final destinations and both readiness fences remain
                // valid for short MTP verification windows as well.
                cell.direct_packet=cell.host_cache_packet && cell.tokens<=8 &&
                    (!direct_packet || direct_packet[0]!='0');
                if(cache_->ssd_experts() && cell.direct_packet)
                    cell.ssd_gate_up_descriptors=std::make_unique<Mapped>(cell.transfer_descriptors->bytes);
                if(cell.wire_fallback && !cell.direct_packet)
                    throw std::logic_error("wire fallback requires a native direct packet");
                if(std::getenv("MFQ_TRACE_HOST_CACHE_PACKET"))std::fprintf(stderr,
                    "host_cache_packet window=%d skip_cpu=%d fused=%d native=%d direct=%d option=%s enabled=%d\n",
                    int(cell.in_window),int(cell.skip_cpu_wait),int(cell.fused_transfer_copy),cell.native_dma_mode,
                    int(direct_enabled),host_packet?host_packet:"unset",int(cell.host_cache_packet));
            }
            // Direct packets publish final stage/cache pointers; their graph
            // never scatters payload from this image. Retain only its header.
            // The wire fallback still needs space for every routed payload.
            const auto capacity=cell.direct_packet && !cell.wire_fallback ? cell.wire_header_bytes :
                ((32+24*routes*wire_geometry.second+15)&~int64_t(15))+routes*wire_geometry.first;
            cell.wire_capacity=capacity;
            // Direct packets publish payload destinations through the mapped
            // descriptor table. Their captured kernels never embed the wire
            // address; only the serialized host/DMA producer uses this buffer.
            cell.indirect_wire=cell.two_stage && cell.direct_packet && cell.wire_fallback &&
                !window && !mfq::cuda::runtime_options::dma_graph_batch();
            bind_wire(cell);
            if(cell.wire_fallback && !cache_->pipeline_wire_fallback_cells_++)
                std::fprintf(stderr,"moe_wire_fallback_dispatch scope=first_cell tokens=%d cache_bytes=%zu wire_bytes=%lld\n",
                    cell.tokens,std::size_t(tensor_nbytes(cache_->pipeline_direct_cache_storage_)),static_cast<long long>(cell.wire.numel()));
            MFQ_CUDA_CHECK(cudaMemsetAsync(cell.wire.data_ptr(),0,32,stream_));
            MFQ_CUDA_CHECK(mfq::cuda::moe_wire_launch_geometry(&cell.wire_blocks,&cell.wire_threads));
        }
        if(cell.native_dma_mode) {
            if(cache_->pipeline_ram_registered_bytes_ && !cache_->pipeline_ram_alias_)MFQ_CUDA_CHECK(cudaHostGetDevicePointer(
                &cache_->pipeline_ram_alias_,cache_->pipeline_ram_complement_->data_ptr(),0));
            if(!cache_->pipeline_host_stage_alias_)MFQ_CUDA_CHECK(cudaHostGetDevicePointer(
                &cache_->pipeline_host_stage_alias_,cache_->pipeline_host_stage_.data_ptr(),0));
        }
        if(cell.mapped_copy) {
            // The first decode cell can follow an ordinary prefill DMA.
            // Later graph-owned copies are fenced by the next route export.
            if(cache_->transfer_ready_recorded_) {
                MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->transfer_ready_));
                cache_->transfer_ready_recorded_=false;cache_->pipeline_dma_leases_.clear();
            }
            if(cache_->pipeline_ram_registered_bytes_ && !cache_->pipeline_ram_alias_)MFQ_CUDA_CHECK(cudaHostGetDevicePointer(
                &cache_->pipeline_ram_alias_,cache_->pipeline_ram_complement_->data_ptr(),0));
            if(!cache_->pipeline_host_stage_alias_)MFQ_CUDA_CHECK(cudaHostGetDevicePointer(
                &cache_->pipeline_host_stage_alias_,cache_->pipeline_host_stage_.data_ptr(),0));
            const auto fields=std::size_t(cell.entries)*route_fields+1;
            cell.mapped_copy_table=std::make_unique<Mapped>(32+fields*sizeof(mfq::cuda::MoeMappedCopyDescriptor));
            cell.mapped_copy_snapshot=tb::empty({int64_t(cell.mapped_copy_table->bytes)},
                cell.input.options().dtype(tb::kUInt8));
            if(cell.mapped_overlap)cell.copy_events=std::make_unique<CopyEvents>();
            MFQ_CUDA_CHECK(mfq::cuda::moe_mapped_copy_launch_geometry(&cell.wire_blocks,&cell.wire_threads));
        }
        const auto* cached_resident=std::getenv("MFQ_MOE_KV_GPU_RESIDENT");
        const auto* projection_cache=std::getenv("MFQ_MOE_KV_PROJECTION_CACHE");
        const auto* warp_prepare=std::getenv("MFQ_MFE_PREPARE_WARP");
        cell.cached_resident=cell.host_cache_packet && cell.fused->batch().resident_plan_overlap &&
            cell.tokens*cell.routes<=256 && (!projection_cache || projection_cache[0]=='1') &&
            (!warp_prepare || warp_prepare[0]!='0') && (!cached_resident || cached_resident[0]=='1');
        if(cell.cached_resident) {
            if(!cached_expert_fields_) {
                auto table=std::make_shared<Mapped>(std::size_t(experts_)*sizeof(mfq::cuda::MfeCachedExpert));
                for(int p=0;p<3;++p) {
                    const auto& source=*sources_[p];
                    std::vector<std::array<std::size_t,8>> adjustments(experts_);
                    std::vector<std::vector<int64_t>> sizes(experts_);
                    for(int e=0;e<experts_;++e) {
                        sizes[e]=source.cohorts_[source.expert_to_cohort_[e]].bytes_per_expert;
                        if(combined_gate_up_ && p==1) {
                            const auto view=cell.fused->expert_view(p,e,sizes[e],-1);
                            auto& a=adjustments[e];
                            if(view.family==1) {
                                a[1]=ff_;a[2]=std::size_t(ff_)*8;
                                a[3]=a[4]=std::size_t(ff_)*view.groups;a[5]=a[6]=std::size_t(ff_)*4;
                            }else if(view.family==3) {
                                a[0]=std::size_t(sizes[e][0])/2;a[1]=std::size_t(sizes[e][1])/2;
                            }else {
                                for(int f=0;f<3;++f)a[f]=std::size_t(sizes[e][f])/2;
                                a[3]=std::size_t(ff_)*4;
                            }
                        }
                    }
                    cache_->kv_expert_loan_->observe_projections(source.id_,
                        [weak_table=std::weak_ptr(table),p,sizes=std::move(sizes),adjustments=std::move(adjustments)]
                        (int64_t expert,const void* base,const std::array<std::size_t,8>& offsets) {
                            const auto table=weak_table.lock();if(!table)return;
                            auto& state=table->h<mfq::cuda::MfeCachedExpert>()[expert];
                            const auto version=(state.ready_mask+8)&~uint64_t(7);
                            if(!base){state.ready_mask=version|((state.ready_mask&7)&~(uint64_t(1)<<p));std::atomic_thread_fence(std::memory_order_release);return;}
                            for(std::size_t f=0;f<sizes[expert].size();++f)
                                state.fields[p][f]=sizes[expert][f]?reinterpret_cast<uint64_t>(base)+offsets[f]+adjustments[expert][f]:0;
                            std::atomic_thread_fence(std::memory_order_release);_mm_sfence();state.ready_mask=version|(state.ready_mask&7)|(uint64_t(1)<<p);
                        },table);
                }
                cached_expert_fields_=std::move(table);
            }
            std::array<std::vector<std::vector<int64_t>>,3> sizes;
            for(int p=0;p<3;++p)for(const auto& cohort:sources_[p]->cohorts_)sizes[p].push_back(cohort.bytes_per_expert);
            cell.fused->enable_cached_dispatch(cached_expert_fields_->d<mfq::cuda::MfeCachedExpert>(),sizes,cell.flags.d<uint32_t>()+4);
            if(cell.shared_gate_quantized.defined() && shared_weights_.scalar_gate)
                cell.fused->try_fuse_prepare_shared_gate(shared_weights_.scalar_gate->nint,cell.shared_gate);
        }
        if(!cache_->pipeline_pool_context_) {
            cache_->pipeline_pool_context_=mfq::cuda::default_context(cell.input.get_device());
            cache_->pipeline_pool_stream_=stream_;cache_->pipeline_pool_context_->begin_graph_pool(stream_);
        } else if(cache_->pipeline_pool_stream_!=stream_)throw std::runtime_error("MFQ graph buffers cannot cross execution streams");
        for(int flag=0;flag<4;++flag)publish(cell.flags.h<uint32_t>()+16*flag);
        if(cell.phased_transfer)publish(cell.flags.h<uint32_t>()+80);
        if(cell.phased_transfer && mfq::cuda::runtime_options::dma_graph_batch() &&
                !cache_->pipeline_dma_batches_[0]) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(cache_->weight_stream_));
            auto* dummy=cache_->pipeline_host_stage_.data_ptr<uint8_t>();
            *dummy=0;
            for(int phase=0;phase<2;++phase) {
                cache_->pipeline_dma_batches_[phase]=std::make_unique<mfq::cuda::DmaCopyBatch>(cache_->weight_stream_);
                cache_->pipeline_dma_batches_[phase]->prepare(128,cell.wire.data_ptr(),dummy);
            }
        }
        if(cache_->pipeline_dma_profile_enabled_ && !cell.mapped_copy && !cell.phased_transfer) {
            try {
                if(cache_->pipeline_dma_profile_fail_alloc_)throw std::bad_alloc();
                cell.dma_timing=std::make_unique<DmaTimes>();
            } catch(const std::bad_alloc&) {
                cache_->pipeline_dma_profile_enabled_=false;cache_->pipeline_dma_profile_dropped_=true;
            }
        }
        if(window)return;
        if(!cell.graph_replay) {
            // Prime format metadata and activation workspaces with all waits
            // released. Eager prefill temporaries are stream-ordered and must
            // not be retained by the decode graph pool.
            launch_body(cell,execution);
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));cell.output={};
            return;
        }
        // Warm the allocation/workspace shape, then capture exactly the same
        // body. Plans disable every expert during this preparation.
        // Initialize persistent format caches before lending temporaries to
        // the graph pool; first-use caches otherwise consume warmed blocks.
        launch_body(cell,execution);
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));cell.output={};
        {mfq::cuda::GraphWarmupScope warmup(cache_->pipeline_pool_context_,stream_);launch_body(cell,execution);}
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));cell.output={};
        std::string error;
        if(!cell.graphs->record(1,cell.tokens,[&]{launch_body(cell,execution);},error))throw std::runtime_error(error);
        ++cache_->stats_.pipeline_graph_captures;
    }
    tb::Tensor activation(const tb::Tensor& gate,const tb::Tensor& up,bool fused) {
        if(swiglu_limit_>0)return clipped_silu_mul_cuda(gate,up,swiglu_limit_);
        if(fused)return moe_swiglu_rounded_cuda(gate,up,cache_->pipeline_sigmoid_table_);
        return (gate*tb::sigmoid(gate))*up;
    }
    tb::Tensor gpu_ffn(Cell& cell,CudaExecutionContext& execution,bool transfer,const MoeRoutePlan& route) {
        auto projection=[&](int i,const tb::Tensor& x) {
            auto& binding=cell.binding[i];auto& runtime=transfer ? binding.transfer : binding.hot;
            return runtime.forward(execution.config,execution.kl_mmq,execution.force_moe_prefill_mma_off,
                execution.force_moe_pool_path,x,route,cell.shared_input && i<2);
        };
        auto gate_output=projection(0,cell.input);
        // An ungated bank publishes Up in the Gate slot and has no activation
        // partner, so the duplicated projection only feeds relu-squared.
        if(up_only_)return projection(2,relu_squared_cuda(gate_output.contiguous()));
        auto up_output=projection(1,cell.input);
        if(swiglu_limit_>0 || cell.fused_activation)
            return projection(2,activation(gate_output,up_output,cell.fused_activation));
        auto gu=tb::cat({gate_output,up_output},-1);
        auto gate=gu.narrow(-1,0,ff_),up=gu.narrow(-1,ff_,ff_);
        return projection(2,(gate*tb::sigmoid(gate))*up);
    }
    void launch_body(Cell& cell,CudaExecutionContext& execution,const tb::Tensor* input_view=nullptr,
            const tb::Tensor* ids_view=nullptr,const tb::Tensor* weights_view=nullptr) {
        const auto& input=input_view?*input_view:cell.input;
        const auto& ids=ids_view?*ids_view:cell.ids;
        if(cell.two_stage) {
            if(input_view)cell.fused->bind_window_inputs(input,ids,*weights_view);
            else cell.fused->use_staged_inputs();
        }
        // SSD predictions only populate host leases. Demand transfer owns its
        // normal completion flags; no speculative GPU wire is consumed.
        if(auto* window=mfq::cuda::DecodeWindow::recording();window && !cache_->ssd_experts()) {
            const auto found=prediction_cells_.find(window);
            if(found!=prediction_cells_.end()) {
                auto& prediction=*found->second;
                mfq::cuda::wait_mapped_plan(prediction.flags.d<uint32_t>()+4,nullptr,nullptr,0,
                    prediction.flags.d<uint32_t>()+8,reinterpret_cast<uint32_t*>(prediction.aborted.data_ptr<int32_t>()),stream_);
                mfq::cuda::moe_scatter_wire(cache_->pipeline_prefetch_wire_gpu_.data_ptr(),
                    reinterpret_cast<uint32_t*>(prediction.aborted.data_ptr<int32_t>()),cell.wire_blocks,cell.wire_threads,stream_);
            }
        }
        const auto copy_mapped=[&](cudaStream_t copy_stream) {
            auto* snapshot=cell.mapped_copy_snapshot.data_ptr();
            const auto capacity=(cell.mapped_copy_table->bytes-32)/sizeof(mfq::cuda::MoeMappedCopyDescriptor);
            mfq::cuda::moe_snapshot_mapped(cell.mapped_copy_table->device,cell.flags.d<uint32_t>()+64,
                snapshot,capacity,copy_stream);
            mfq::cuda::moe_copy_mapped(snapshot,reinterpret_cast<const uint32_t*>(snapshot)+4,
                cell.wire_blocks,cell.wire_threads,copy_stream);
        };
        const auto mark=[&](const char* phase) {
            if(auto* window=mfq::cuda::DecodeWindow::recording())window->mark(sources_[0]->layer_id_,phase);
        };
        const auto copy_kv_loan=[&](int phase) {
            if(!cell.kv_loan_table)return;
            const auto offset=phase*cell.kv_loan_table->bytes/2;
            auto* snapshot=static_cast<uint8_t*>(cell.kv_loan_snapshot.data_ptr())+offset;
            mfq::cuda::moe_snapshot_mapped(static_cast<const uint8_t*>(cell.kv_loan_table->device)+offset,
                cell.fused->batch().aborted,snapshot,(cell.kv_loan_table->bytes/2-32)/sizeof(mfq::cuda::MoeMappedCopyDescriptor),stream_);
            mfq::cuda::moe_copy_mapped(snapshot,reinterpret_cast<const uint32_t*>(snapshot)+4,
                cell.wire_blocks,cell.wire_threads,stream_);
        };
        const auto scatter_phase=[&](int phase) {
            // This fixed policy submits every field to its final destination
            // or reads it directly from the loan. Both readiness waits remain.
            if(cell.direct_packet) {
                if(!cell.direct_packet_trace_reported) {
                    cell.direct_packet_trace_reported=true;
                    std::fprintf(stderr,"moe_native_direct_packet_dispatch scope=first_skipped_scatter enabled=1 tokens=%d phases=2 native_dma=%d\n",
                        cell.tokens,cell.native_dma_mode);
                }
                return;
            }
            if(cell.fused_transfer_copy) {
                const auto half=cell.kv_loan_table->bytes/2;
                mfq::cuda::moe_scatter_wire_and_loan_phase(cell.wire.data_ptr(),
                    cell.kv_loan_snapshot.data_ptr<uint8_t>()+(phase-1)*half,
                    (half-32)/sizeof(mfq::cuda::MoeMappedCopyDescriptor),cell.fused->batch().aborted,
                    cell.wire_blocks,cell.wire_threads,phase,stream_);
            }else {
                mfq::cuda::moe_scatter_wire_phase(cell.wire.data_ptr(),cell.fused->batch().aborted,
                    cell.wire_blocks,cell.wire_threads,phase,stream_);
                copy_kv_loan(phase-1);
            }
        };
        mark("expert_begin");
        cudaStreamCaptureStatus capture;
        MFQ_CUDA_CHECK(cudaStreamIsCapturing(stream_,&capture));
        const auto event_flags=capture==cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault;
        const auto measure=[&](int point) {
            if(!cell.skip_gpu_timing)
                MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[point],stream_,event_flags));
        };
        const auto* publish_prepare=std::getenv("MFQ_MFE_PREPARE_PUBLISH");
        const bool prepared_early=cell.two_stage && !cell.export_cpu_input &&
            cell.fused->batch().resident_plan_overlap && cell.fused->batch().transfer_ready &&
            (!publish_prepare || publish_prepare[0]!='0');
        if(prepared_early)cell.fused->prepare(cell.host_ids.d<int32_t>(),cell.flags.d<uint32_t>());
        else mfq::cuda::moe_publish_routes(cell.export_cpu_input?input.data_ptr():nullptr,ids.data_ptr<int32_t>(),
            cell.export_cpu_input?cell.x.d<float>():nullptr,cell.host_ids.d<int32_t>(),cell.flags.d<uint32_t>(),
            cell.export_cpu_input?cell.tokens*width_:0,cell.entries,stream_);
        if(cell.two_stage) {
            if(cell.mapped_overlap) {
                MFQ_CUDA_CHECK(cudaEventRecord(cell.copy_events->fork,stream_));
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(cache_->weight_stream_,cell.copy_events->fork,0));
                mfq::cuda::wait_mapped_flag(cell.flags.d<uint32_t>()+32,cache_->weight_stream_);
                copy_mapped(cache_->weight_stream_);
                MFQ_CUDA_CHECK(cudaEventRecord(cell.copy_events->done,cache_->weight_stream_));
            }
            if(shared_ && prepared_early && cell.fused->shared_gate_in_prepare()) {
                cell.fused->bind_shared_gate(cell.shared_gate);
                if(!cell.shared_gate_trace_reported) {
                    cell.shared_gate_trace_reported=true;
                    std::fprintf(stderr,"mfe_shared_gate_prepare_dispatch scope=first_launch enabled=1 tokens=%d window=%d\n",cell.tokens,int(cell.in_window));
                }
            }else if(shared_) {
                // Recording replaces this cell's prior capture. Return its
                // warmup result before allocating the new gate in that pool.
                cell.fused->bind_shared_gate({});
                tb::Tensor gate;
                if(prepared_early && cell.shared_gate_quantized.defined()) {
                    if(auto value=try_nint_shared_gate_sigmoid_cuda(input,shared_weights_.scalar_gate->nint,
                            cell.shared_gate_quantized,cell.shared_gate_scales))gate=*value;
                }
                if(!cell.shared_gate_trace_reported) {
                    cell.shared_gate_trace_reported=true;
                    const auto trace=mfq::cuda::runtime_options::trace_mfe_dispatch_requested();
                    if(trace && *trace)std::fprintf(stderr,
                        "mfe_shared_gate_dispatch cell=%p scope=first_launch tokens=%d prepared=%d window=%d\n",
                        static_cast<void*>(&cell),cell.tokens,int(gate.defined()),int(cell.in_window));
                }
                if(!gate.defined())gate=shared_weights_.gate(execution,input);
                const auto* view=std::getenv("MFQ_MFE_SHARED_GATE_VIEW");
                if(cell.in_window && (!view || view[0]!='0') && gate.is_contiguous() &&
                        gate.scalar_type()==cell.shared_gate.scalar_type())cell.fused->bind_shared_gate(gate);
                else {cell.shared_gate.copy_(gate);cell.fused->bind_shared_gate(cell.shared_gate);}
            }
            mark("shared_end");
            if(!prepared_early)cell.fused->prepare();
            // Resident arithmetic runs while cold weights are in flight.
            // Keep DMA waits outside the two arithmetic event intervals.
            measure(0);mark("hot_begin");
            cell.fused->resident();
            measure(1);mark("hot_end");
            if(cell.mapped_overlap)MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,cell.copy_events->done,0));
            if(cell.fused_transfer_copy)cell.fused->wait_transfer(cell.kv_loan_table->device,
                cell.kv_loan_snapshot.data_ptr(),(cell.kv_loan_table->bytes/2-32)/sizeof(mfq::cuda::MoeMappedCopyDescriptor),
                cell.host_cache_packet?cell.transfer_descriptors->device:nullptr,
                cell.host_cache_packet?cell.transfer_descriptors->bytes/sizeof(uint64_t):0);
            else cell.fused->wait_transfer();
            if(cell.mapped_copy && !cell.mapped_overlap)copy_mapped(stream_);
            if(cell.phased_transfer)scatter_phase(1);
            else if(cell.transfer_cache && !cell.mapped_copy)mfq::cuda::moe_scatter_wire(cell.wire.data_ptr(),cell.fused->batch().aborted,
                cell.wire_blocks,cell.wire_threads,stream_);
            if(!cell.phased_transfer)copy_kv_loan(0);
            measure(2);mark("dma_end");
            cell.fused->transferred();
            mark("transfer_gate_up_end");
            if(cell.phased_transfer) {
                if(cell.fused_transfer_copy) {
                    const auto half=cell.kv_loan_table->bytes/2;
                    mfq::cuda::wait_mapped_plan_snapshot(cell.flags.d<uint32_t>()+80,cell.flags.d<uint32_t>()+64,
                        cell.fused->batch().aborted,static_cast<const uint8_t*>(cell.kv_loan_table->device)+half,
                        cell.kv_loan_snapshot.data_ptr<uint8_t>()+half,(half-32)/sizeof(mfq::cuda::MoeMappedCopyDescriptor),stream_,
                        cell.ssd_gate_up_descriptors?cell.transfer_descriptors->device:nullptr,
                        cell.ssd_gate_up_descriptors?cell.fused->transfer_buffer().data_ptr():nullptr,
                        cell.ssd_gate_up_descriptors?cell.transfer_descriptors->bytes/sizeof(uint64_t):0);
                }else mfq::cuda::wait_mapped_plan(cell.flags.d<uint32_t>()+80,nullptr,nullptr,0,
                        cell.flags.d<uint32_t>()+64,cell.fused->batch().aborted,stream_);
                scatter_phase(2);
            }
            mark("cpu_wait_begin");
            if(!cell.skip_cpu_wait)cell.fused->wait_cpu();
            mark("cpu_wait_end");
            cell.fused->down_reduce_compute();
            measure(3);mark("transfer_end");
            cell.output=cell.fused->output();mark("cpu_end");mark("expert_end");return;
        }
        if(cell.shared_input)moe_quantize_shared_input_cuda(cell.input,cell.shared_input_descriptors,cell.shared_input_groups);
        auto shared=shared_ ? shared_(execution,cell.input) : tb::Tensor{};
        mark("shared_end");
        if(cell.gpu_resident_dispatch) {
            // Residency publication precedes this graph on its compute stream.
            // A complete GPU bundle is never assigned to CPU by the host plan.
            mfq::cuda::moe_classify_resident(reinterpret_cast<const uint64_t*>(cell.dispatch_pointers.data_ptr<int64_t>()),
                static_cast<int>(sources_[0]->cohorts_.size()),static_cast<int>(sources_[1]->cohorts_.size()),
                static_cast<int>(sources_[2]->cohorts_.size()),experts_,cell.device_kind.data_ptr<int32_t>(),
                reinterpret_cast<uint32_t*>(cell.device_abort.data_ptr<int32_t>()),stream_);
        } else {
            mfq::cuda::wait_mapped_plan(cell.flags.d<uint32_t>()+16,cell.kind.d<int32_t>(),cell.device_kind.data_ptr<int32_t>(),
                experts_,cell.flags.d<uint32_t>()+64,reinterpret_cast<uint32_t*>(cell.device_abort.data_ptr<int32_t>()),stream_);
        }
        const auto maps=[&] {
            mfq::cuda::moe_update_dispatch_maps(reinterpret_cast<const uint64_t*>(cell.dispatch_pointers.data_ptr<int64_t>()),
                static_cast<int>(cell.dispatch_pointers.numel()/6),experts_,cell.device_kind.data_ptr<int32_t>(),
                reinterpret_cast<const uint32_t*>(cell.device_abort.data_ptr<int32_t>()),stream_);
        };
        maps();
        const auto route=build_moe_route_plan(cell.ids,experts_);
        MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[0],stream_,event_flags));
        mark("hot_begin");
        auto hot=gpu_ffn(cell,execution,false,route);
        MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[1],stream_,event_flags));
        mark("hot_end");
        // DMA completion also publishes the host CPU/transfer choices. Partial
        // GPU bundles remain on that path, with their resident projections retained.
        mfq::cuda::wait_mapped_plan(cell.flags.d<uint32_t>()+32,
            cell.gpu_resident_dispatch?cell.kind.d<int32_t>():nullptr,
            cell.gpu_resident_dispatch?cell.device_kind.data_ptr<int32_t>():nullptr,
            cell.gpu_resident_dispatch?experts_:0,cell.flags.d<uint32_t>()+64,
            reinterpret_cast<uint32_t*>(cell.device_abort.data_ptr<int32_t>()),stream_);
        // Re-read cancellation after DMA completion, before kernels may use
        // transferred fields. A failed host phase can safely release the graph.
        MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[2],stream_,event_flags));
        mark("dma_end");
        maps();
        if(cell.wire.defined())mfq::cuda::moe_scatter_wire(cell.wire.data_ptr(),reinterpret_cast<const uint32_t*>(cell.device_abort.data_ptr<int32_t>()),
            cell.wire_blocks,cell.wire_threads,stream_);
        auto transfer=gpu_ffn(cell,execution,true,route);
        MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[3],stream_,event_flags));
        mark("transfer_end");
        mfq::cuda::wait_mapped_flag(cell.flags.d<uint32_t>()+48,stream_);
        mark("cpu_end");
        auto parts=tb::empty({cell.tokens,cell.routes,width_},cell.input.options());
        mfq::cuda::moe_merge_results(hot.data_ptr(),transfer.data_ptr(),cell.cpu.device,cell.ids.data_ptr<int32_t>(),
            cell.device_kind.data_ptr<int32_t>(),parts.data_ptr(),cell.entries,width_,stream_);
        cell.output=shared.defined() ? moe_weighted_reduce_cuda(parts,cell.weights)+shared : parts;
        mark("expert_end");
        MFQ_CUDA_CHECK(cudaGetLastError());
    }
    void wait_publication(Cell& cell) {
        const auto* flag=cell.flags.h<uint32_t>();
        auto flush=std::chrono::steady_clock::now();const auto start=flush;uint32_t spins=0;
        const auto poll_interval=std::chrono::microseconds(mfq::cuda::runtime_options::route_poll_interval_us());
        if(*(volatile const uint32_t*)flag<1)mfq::cuda::check_route_submission(stream_);
        while(*(volatile const uint32_t*)flag<1) {
            _mm_pause();if((++spins&1023u)!=0)continue;
            const auto now=std::chrono::steady_clock::now();
            if(now-flush>=poll_interval) {
                flush=now;const auto q=cudaStreamQuery(stream_);
                if(q!=cudaSuccess && q!=cudaErrorNotReady)MFQ_CUDA_CHECK(q);
            }
            if(now-start>std::chrono::seconds(60)) {
                trace_serve(cell,"route_timeout",true);
                throw std::runtime_error("MFQ GPU route publication stalled");
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    void plan(Cell& cell) {
        const bool must_drain=!cell.in_window || !cache_->pipeline_dispatch_replay_.empty();
        const bool calibration_busy=cache_->pipeline_cpu_calibration_.collect(cache_->pipeline_cpu_cost_,must_drain);
        const bool shadow=cell.in_window && cache_->pipeline_shared_cpu_cost_ &&
            cache_->pipeline_cpu_transfer_budget_ && mfq::cuda::runtime_options::background_calibration() && cache_->pipeline_dispatch_replay_.empty();
        cell.calibration_expert=-1;
        std::vector<int32_t> legacy_ids;
        auto& ids=cell.reuse_plan?cell.plan_ids:legacy_ids;
        ids.assign(cell.host_ids.h<int32_t>(),cell.host_ids.h<int32_t>()+cell.entries);
        auto resident=[&](int e) {
            const uint64_t cached=cell.cached_resident?cached_expert_fields_->h<mfq::cuda::MfeCachedExpert>()[e].ready_mask:0;
            for(int p=0;p<3;++p)if(!sources_[p]->quant_source_->gpu_resident(e) && !(cached&(uint64_t(1)<<p)))return false;
            return true;
        };
        auto eligible=[&](int e) {
            return std::all_of(sources_.begin(),sources_.end(),[&](const auto& s) {
                // A completely cold SSD expert can be read for CPU execution.
                // Keep partial GPU bundles on the GPU so their resident
                // projections are not redundantly fetched from storage.
                if(cache_->ssd_experts())return !s->quant_source_->gpu_resident(e);
                return s->quant_source_->host_cache()->contains(s->quant_source_->host_key(e));
            });
        };
        const auto transfer_bytes=[&](int e) {
            uint64_t bytes=0;
            for(int p=0;p<3;++p) {
                if(single_up() && p==1)continue;
                const auto& source=*sources_[p];const int c=source.expert_to_cohort_[e];
                if(source.quant_source_->gpu_resident(e))continue;
                if(cell.transfer_cache && !cell.wire_fallback) {
                    const auto* book=cell.binding[p].stages[c]->book.get();
                    if(book && book->slot_for({source.id_,c,e})>=0)continue;
                }
                bytes+=source.bytes_for_expert(e);
            }
            return bytes;
        };
        const auto measured_cpu=[&](const mfq::MoeDispatchGroup& group) {
            return cache_->pipeline_shared_cpu_cost_
                ? cache_->pipeline_cpu_cost_.estimate(cpu_cost_key(group))
                : cpu_work_ ? cpu_ns_/cpu_work_*group.positions.size() : 0;
        };
        const mfq::MoeDispatchRates rates{cache_->pipeline_pcie_gbps_,
            stage_bytes_ ? stage_ns_/stage_bytes_ : 0,gpu_samples_ ? gpu_ns_/gpu_samples_ : 0,
            cache_->pipeline_cpu_transfer_budget_};
        if(cell.reuse_plan)
            mfq::plan_moe_dispatch_reuse(cell.plan,cell.plan_workspace,ids,experts_,0,resident,
                [](int){return true;},cell.allow_padding);
        else cell.plan=mfq::plan_moe_dispatch(ids,experts_,0,resident,[](int){return true;},cell.allow_padding);
        std::size_t misses=0;
        for(const auto& group:cell.plan.groups) {
            const bool hot=group.kind==mfq::MoeDispatchKind::GpuResident;misses+=!hot;
            if(cell.in_window && (cell.tokens==3 || cell.tokens==4)) {
                const auto count=group.positions.size();
                if(count>=1 && count<=4) {
                    ++cache_->stats_.pipeline_small_window_groups[count];
                    if(hot)++cache_->stats_.pipeline_small_window_hot_groups[count];
                }
            }
        }
        if(!cell.batch_decode || cache_->config_.moe_ram_pcie_fraction) {
            const auto quota=!cell.batch_decode ? misses : static_cast<std::size_t>(misses*cache_->ram_pcie_fraction_);
            if(cell.reuse_plan)mfq::assign_moe_transfer_quota(cell.plan,quota,[](int){return true;});
            else cell.plan=mfq::plan_moe_dispatch(ids,experts_,quota,resident,
                [](int){return true;},cell.allow_padding);
        } else {
            std::vector<std::size_t> cold;
            std::vector<mfq::MoeDispatchCost> costs;
            for(std::size_t i=0;i<cell.plan.groups.size();++i) {
                auto& group=cell.plan.groups[i];if(group.kind==mfq::MoeDispatchKind::GpuResident)continue;
                const auto bytes=transfer_bytes(group.expert);
                cold.push_back(i);
                const double measured=measured_cpu(group);
                costs.push_back({measured,bytes,bytes>0 && eligible(group.expert)});
                group.kind=mfq::MoeDispatchKind::GpuTransfer;
            }
            std::vector<bool> selected(costs.size(),false);
            if(cache_->pipeline_shared_cpu_cost_) {
                // Acquire three observations per compatible FFN geometry so
                // one cold worker wake cannot establish the session's rate.
                std::size_t sample=costs.size();
                for(std::size_t i=0;i<costs.size();++i)if(costs[i].cpu_eligible && !costs[i].cpu_ns &&
                    (sample==costs.size() || costs[i].transfer_bytes>costs[sample].transfer_bytes))sample=i;
                if(sample<costs.size()) {
                    if(shadow) {
                        if(!calibration_busy)cell.calibration_expert=cell.plan.groups[cold[sample]].expert;
                    }else selected[sample]=true;
                }
                else {
                    for(auto& cost:costs)cost.cpu_eligible=cost.cpu_eligible && cost.cpu_ns>0;
                    if(!calibration_busy)selected=mfq::select_moe_cpu_work(costs,rates);
                }
            } else if(cpu_work_) {
                selected=mfq::select_moe_cpu_work(costs,rates);
            } else {
                // One retained production group supplies the first CPU rate;
                // every other cold group uses the calibrated PCIe path.
                std::size_t sample=costs.size();
                for(std::size_t i=0;i<costs.size();++i)if(costs[i].cpu_eligible &&
                    (sample==costs.size() || costs[i].transfer_bytes>costs[sample].transfer_bytes))sample=i;
                if(sample<costs.size())selected[sample]=true;
            }
            for(std::size_t i=0;i<cold.size();++i)if(selected[i])cell.plan.groups[cold[i]].kind=mfq::MoeDispatchKind::Cpu;
        }
        if(!cache_->pipeline_dispatch_replay_.empty()) {
            if(cache_->pipeline_dispatch_replay_cursor_>=cache_->pipeline_dispatch_replay_.size())
                throw std::runtime_error("MFQ dispatch replay ended before inference");
            const auto& replay=cache_->pipeline_dispatch_replay_[cache_->pipeline_dispatch_replay_cursor_++];
            if(replay.layer!=sources_[0]->layer_id_ || replay.tokens!=cell.tokens || replay.ids!=ids) {
                std::ostringstream message;
                message<<"MFQ dispatch replay input/route mismatch record="
                    <<cache_->pipeline_dispatch_replay_cursor_-1
                    <<" layer="<<sources_[0]->layer_id_<<" expected_layer="<<replay.layer
                    <<" tokens="<<cell.tokens<<" expected_tokens="<<replay.tokens
                    <<" routes="<<ids.size()<<" expected_routes="<<replay.ids.size();
                const auto count=std::min(ids.size(),replay.ids.size());
                for(std::size_t position=0;position<count;++position)if(ids[position]!=replay.ids[position]) {
                    message<<" position="<<position<<" expert="<<ids[position]
                        <<" expected_expert="<<replay.ids[position];
                    break;
                }
                throw std::runtime_error(message.str());
            }
            std::size_t matched=0;
            for(auto& group:cell.plan.groups) {
                const bool cpu=std::find(replay.cpu.begin(),replay.cpu.end(),group.expert)!=replay.cpu.end();
                if(cpu && (resident(group.expert) || !eligible(group.expert)))
                    throw std::runtime_error("MFQ dispatch replay CPU expert changed tier ownership");
                matched+=cpu;
                group.kind=cpu?mfq::MoeDispatchKind::Cpu:resident(group.expert)?mfq::MoeDispatchKind::GpuResident:mfq::MoeDispatchKind::GpuTransfer;
            }
            if(matched!=replay.cpu.size())throw std::runtime_error("MFQ dispatch replay CPU expert missing from routes");
        }
        if(cell.batch_decode && cell.tokens>1) {
            const int phase=sources_[0]->layer_id_<0;
            cache_->stats_.pipeline_multi_unique_routes[phase]+=cell.plan.groups.size();
            for(const auto& group:cell.plan.groups)
                cache_->stats_.pipeline_multi_route_positions[phase]+=group.positions.size();
        }
        // Explicit quotas also share the CPU pool. An unfinished calibration
        // cannot be followed by inference work queued behind that calibration.
        if(calibration_busy)for(auto& group:cell.plan.groups)
            if(group.kind==mfq::MoeDispatchKind::Cpu)group.kind=mfq::MoeDispatchKind::GpuTransfer;
        std::memset(cell.kind.host,0,cell.kind.bytes);
        // Only the unfused dispatch map consumes local cohort ordinals.
        if(!cell.reuse_plan)std::memset(cell.local.host,0xff,cell.local.bytes);
        const bool early_gu=cell.two_stage && cell.fused->batch().resident_plan_overlap &&
            mfq::cuda::runtime_options::early_gate_up() && mfq::cuda::runtime_options::parallel_gate_up();
        cache_->stats_.pipeline_early_gate_up_enabled=early_gu;
        for(auto& group:cell.plan.groups) {
            // Mixed formats have projection-granular residency. A partial
            // GPU bundle must use its remaining RAM projections on the GPU.
            if(group.kind==mfq::MoeDispatchKind::Cpu && !eligible(group.expert))group.kind=mfq::MoeDispatchKind::GpuTransfer;
            const int kind=group.kind==mfq::MoeDispatchKind::Cpu ? 0 : group.kind==mfq::MoeDispatchKind::GpuResident ? 1 : 2;
            cell.kind.h<int32_t>()[group.expert]=kind;
            if(cell.cached_resident && kind==2 && early_gu) {
                const auto mask=cached_expert_fields_->h<mfq::cuda::MfeCachedExpert>()[group.expert].ready_mask;
                const bool gate=sources_[0]->quant_source_->gpu_resident(group.expert),up=sources_[1]->quant_source_->gpu_resident(group.expert);
                if((gate || (mask&1)) && (up || (mask&2)) && (!gate || !up) &&
                        !sources_[2]->quant_source_->gpu_resident(group.expert) && !(mask&4)) {
                    if(!cache_->stats_.pipeline_kv_gpu_early_gu_experts++)std::fprintf(stderr,
                        "moe_kv_gpu_early_gu_dispatch scope=first_partial_expert enabled=1 tokens=%d expert=%d\n",cell.tokens,group.expert);
                }
            }
            if(cell.two_stage && kind==2 && sources_[0]->quant_source_->gpu_resident(group.expert) &&
                sources_[1]->quant_source_->gpu_resident(group.expert) && !sources_[2]->quant_source_->gpu_resident(group.expert)) {
                ++cache_->stats_.pipeline_gate_up_primary_down_missing_experts;
                cache_->stats_.pipeline_gate_up_primary_down_missing_positions+=group.positions.size();
                if(early_gu)cache_->stats_.pipeline_early_gate_up_positions+=group.positions.size();
            }
            for(auto pos:group.positions)cell.plan.kinds[pos]=group.kind;
        }
        if(!cell.reuse_plan)for(int projection=0;projection<3;++projection) {
            const auto& source=*sources_[projection];std::vector<int> slots(source.cohorts_.size());
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                const auto e=group.expert,c=source.expert_to_cohort_[e];cell.local.h<int32_t>()[projection*experts_+e]=slots[c]++;
            }
        }
        if(cell.two_stage) {
            std::memset(cell.transfer_index->host,0xff,cell.transfer_index->bytes);cell.transfer_views.clear();
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                cell.transfer_index->h<int32_t>()[group.expert]=int32_t(cell.transfer_views.size()/3);
                for(int p=0;p<3;++p) {
                    const auto& source=*sources_[p];const int c=source.expert_to_cohort_[group.expert];
                    const auto& cohort=source.cohorts_[c];const int slot=cohort.arena->book->slot_for({source.id_,c,group.expert});
                    if(cell.reuse_plan) {
                        const auto& prepared=cell.transfer_templates[p][c];auto view=prepared.view;
                        if(slot>=0)for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f)
                            view.fields[f]=prepared.bases[f]?prepared.bases[f]+int64_t(slot)*cohort.bytes_per_expert[f]:nullptr;
                        if(cell.check_reused_views) {
                            const auto original=cell.fused->expert_view(single_up() && p==0?1:p,group.expert,cohort.bytes_per_expert,slot);
                            if(std::memcmp(&original,&view,sizeof(view)))throw std::logic_error("MFE reused transfer metadata differs from original");
                        }
                        cell.transfer_views.push_back(view);
                    } else cell.transfer_views.push_back(cell.fused->expert_view(single_up() && p==0?1:p,group.expert,cohort.bytes_per_expert,slot));
                }
            }
        }
        if(cache_->pipeline_dispatch_capture_required_ || !cache_->pipeline_dispatch_record_path_.empty()) {
            try {
                MoeExpertCache::DispatchRecord row;row.layer=sources_[0]->layer_id_;row.tokens=cell.tokens;row.ids=ids;
                for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::Cpu)row.cpu.push_back(group.expert);
                if(cell.tokens==1 && !cache_->pipeline_miss_record_path_.empty()) {
                    row.missing_bytes.resize(experts_,0);
                    for(int e=0;e<experts_;++e)for(int p=0;p<3;++p) {
                        const auto& source=*sources_[p];const int c=source.expert_to_cohort_[e];
                        if(source.quant_source_->gpu_resident(e))continue;
                        const auto* book=cell.transfer_cache && !cell.wire_fallback?cell.binding[p].stages[c]->book.get():nullptr;
                        if(book && book->slot_for({source.id_,c,e})>=0)continue;
                        row.missing_bytes[e]+=source.bytes_for_expert(e);
                    }
                }
                cache_->pipeline_dispatch_records_.push_back(std::move(row));
            }catch(const std::bad_alloc&) {
                if(cache_->pipeline_dispatch_capture_required_)throw;
                cache_->pipeline_dispatch_record_path_.clear();cache_->pipeline_dispatch_records_.clear();
            }
        }
        if(cell.skip_cpu_wait) {
            if(std::any_of(cell.plan.groups.begin(),cell.plan.groups.end(),[](const auto& group) {
                return group.kind==mfq::MoeDispatchKind::Cpu;
            }))throw std::logic_error("GPU-only CPU-wait graph received CPU expert work");
            ++cache_->stats_.pipeline_gpu_only_cpu_wait_skips;
        }
        if(cell.skip_gpu_timing) {
            if(std::any_of(cell.plan.groups.begin(),cell.plan.groups.end(),[](const auto& group) {
                return group.kind==mfq::MoeDispatchKind::Cpu;
            }))throw std::logic_error("fixed GPU timing graph received CPU expert work");
            ++cache_->stats_.pipeline_fixed_gpu_timing_skips;
        }
        if(cell.transfer_route_plan) {
            auto* packet=cell.transfer_route_plan->h<int32_t>();
            for(int pair=0;pair<cell.entries;++pair) {
                const int expert=ids[pair];
                packet[pair*2]=expert >= 0 ? cell.kind.h<int32_t>()[expert] : -1;
                packet[pair*2+1]=expert >= 0 ? cell.transfer_index->h<int32_t>()[expert] : -1;
            }
        }
        if(observer_)observer_(cell.plan);
        publish(cell.flags.h<uint32_t>()+16);
        // GPU-only routes consume no CPU pair buffer. Their CPU completion
        // is independent of DMA submission and can be published with the plan.
        // Every graph replay resets this flag before receiving its new routes.
        if(mfq::cuda::runtime_options::early_no_cpu_ready() &&
            std::none_of(cell.plan.groups.begin(),cell.plan.groups.end(),[](const auto& group) {
                return group.kind==mfq::MoeDispatchKind::Cpu;
            })) {
            publish(cell.flags.h<uint32_t>()+48);
            ++cache_->stats_.pipeline_early_no_cpu_ready_serves;
        }
    }
    void copy_transfer(Cell& cell,void* destination,const void* source,std::size_t bytes,
            std::size_t source_pitch=0,std::size_t rows=1) {
        const auto total_bytes=bytes*rows;
        const bool trace=cache_->pipeline_dma_profile_enabled_ && cell.dma_timing;
        const bool first=trace && !cell.dma_timing->started;
        if(trace) {
            auto& timing=*cell.dma_timing;
            if(first) {
                timing.enqueue_started=std::chrono::steady_clock::now();
                timing.sample.prepare_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    timing.enqueue_started-timing.fetch_started).count();
                timing.sample.first_copy_bytes=total_bytes;
            }
            ++timing.sample.copies;timing.sample.copy_bytes+=total_bytes;
        }
        if(rows==1)MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,source,bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
        else MFQ_CUDA_CHECK(cudaMemcpy2DAsync(destination,bytes,source,source_pitch,bytes,rows,
            cudaMemcpyHostToDevice,cache_->weight_stream_));
        if(first) {
            // A timing event ahead of the first H2D stalls the concurrent
            // graph/producer protocol on the verified WDDM native workload.
            // Start after the original first copy; its bytes are reported
            // separately and its duration is excluded from the copy interval.
            MFQ_CUDA_CHECK(cudaEventRecord(cell.dma_timing->points[0],cache_->weight_stream_));
            cell.dma_timing->started=true;
        }
    }
    void queue_transfer_ready(Cell& cell,int offset) {
        const int phase=offset==80?1:0;
        if(cell.transfer_notification) {
            // Record Gate/Up without blocking Down's H2D on a host callback.
            // The final event protects only H2D payloads; direct loan reads
            // do not submit background D2D work on this stream.
            if(!phase)MFQ_CUDA_CHECK(cudaEventRecord(cell.transfer_notification->get(),cache_->weight_stream_));
            cell.transfer_notification_pending[phase]=true;
            return;
        }
        if(cell.ready_writer) {
            const auto result=cell.ready_writer(reinterpret_cast<CUstream>(cache_->weight_stream_),
                reinterpret_cast<CUdeviceptr>(cell.flags.d<uint32_t>()+offset),1u,0u);
            if(result==CUDA_SUCCESS) {++cache_->stats_.pipeline_transfer_stream_writes;return;}
            if(result==CUDA_ERROR_NOT_SUPPORTED)cell.ready_writer=nullptr;
            else throw std::runtime_error("CUDA transfer notification failed: "+std::to_string(result));
        }
        MFQ_CUDA_CHECK(cudaLaunchHostFunc(cache_->weight_stream_,
            [](void* p){publish(static_cast<uint32_t*>(p));},cell.flags.h<uint32_t>()+offset));
        ++cache_->stats_.pipeline_transfer_host_callbacks;
    }
    void poll_transfer_notifications(Cell& cell,int phases=2,bool block=true) {
        for(int phase=0;phase<phases;++phase)if(cell.transfer_notification_pending[phase]) {
            const auto started=std::chrono::steady_clock::now();
            uint32_t polls=0;
            const auto event=phase==0?cell.transfer_notification->get():cache_->transfer_ready_;
            cudaError_t status;
            while((status=cudaEventQuery(event))==cudaErrorNotReady) {
                if(!block)break;
                _mm_pause();
                if((++polls&1023u)==0 && std::chrono::steady_clock::now()-started>std::chrono::seconds(60))
                    throw std::runtime_error("MFQ expert DMA notification stalled");
            }
            if(status==cudaErrorNotReady)continue;
            MFQ_CUDA_CHECK(status);
            publish(cell.flags.h<uint32_t>()+(phase==0?32:80));
            cell.transfer_notification_pending[phase]=false;
            ++cache_->stats_.pipeline_transfer_event_polls;
        }
    }
    void finish_transfer(Cell& cell,bool pending) {
        auto* flag=cell.flags.h<uint32_t>()+(cell.phased_transfer?80:32);
        if(cell.mapped_copy) {
            // The graph owns completion and joins its copy stream before use.
            // A following route publication proves these RAM leases are idle.
            cache_->transfer_ready_recorded_=false;publish(flag);return;
        }
        const bool trace=cache_->pipeline_dma_profile_enabled_ && cell.dma_timing && cell.dma_timing->started;
        if(trace) {
            auto& timing=*cell.dma_timing;
            timing.sample.enqueue_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-timing.enqueue_started).count();
            MFQ_CUDA_CHECK(cudaEventRecord(timing.points[1],cache_->weight_stream_));
        }
        if(!pending)publish(flag);
        else queue_transfer_ready(cell,cell.phased_transfer?80:32);
        if(trace)MFQ_CUDA_CHECK(cudaEventRecord(cell.dma_timing->points[2],cache_->weight_stream_));
        MFQ_CUDA_CHECK(cudaEventRecord(cache_->transfer_ready_,cache_->weight_stream_));
        cache_->transfer_ready_recorded_=true;
        // GU may have DMA and a driver-written ready flag while Down is a
        // cache hit. Driver notifications have no host-poll pending bit. Flush
        // that stream too, or WDDM can defer GU behind its waiting graph.
        mfq::cuda::check_stream_progress(cache_->weight_stream_);
        poll_transfer_notifications(cell);
    }
    int64_t upload_transfer_views(Cell& cell,std::vector<mfq::cuda::DmaCopy>* batch=nullptr,bool force_dma=false,
            Mapped* early_image=nullptr) {
        if(!cell.two_stage || cell.transfer_views.empty())return 0;
        const int64_t bytes=int64_t(cell.transfer_views.size())*sizeof(mfq::cuda::MfePackedProjection);
        if(bytes>int64_t(cell.transfer_descriptors->bytes) || bytes>cell.fused->transfer_buffer().numel())
            throw std::logic_error("MFE transfer descriptor capacity exceeded");
        auto& image=early_image?*early_image:*cell.transfer_descriptors;
        std::memcpy(image.host,cell.transfer_views.data(),std::size_t(bytes));
        if(up_only_) {
            // An ungated bank transfers Up through the Gate slot; the kernel reads
            // it as projection 1, which must publish the same staged image.
            auto* views=image.h<mfq::cuda::MfePackedProjection>();
            for(std::size_t i=0;i<cell.transfer_views.size();i+=3)views[i+1]=cell.transfer_views[i];
        }
        if(combined_gate_up_) {
            auto* views=image.h<mfq::cuda::MfePackedProjection>();
            for(std::size_t i=0;i<cell.transfer_views.size();i+=3) {
                auto up=cell.transfer_views[i+1];
                // Gate and Up share one staged compressed expert image.
                for(int f=0;f<(up.family==1?7:up.family==3?2:4);++f)up.fields[f]=cell.transfer_views[i].fields[f];
                views[i+1]=cell.fused->projection_view(1,up);
            }
        }
        const auto* uploaded=image.h<mfq::cuda::MfePackedProjection>();
        for(std::size_t i=0;i<cell.transfer_views.size();++i) {
            if(early_image && i%3==2)continue;
            const auto& view=uploaded[i];
            for(int f=0;f<(view.family==1?7:view.family==3?2:5);++f)
                if(!view.fields[f] && !(view.family==2 && ((f<3 && !view.sizes[f]) || f==4)))
                    throw std::logic_error("MFE transfer descriptor field not published: family="+
                        std::to_string(view.family)+" format="+std::to_string(view.format)+
                        " field="+std::to_string(f)+" tokens="+std::to_string(cell.tokens));
        }
        if(!cell.mapped_copy && (!cell.host_cache_packet || force_dma)) {
            if(batch)batch->push_back({cell.fused->transfer_buffer().data_ptr(),image.host,std::size_t(bytes)});
            else copy_transfer(cell,cell.fused->transfer_buffer().data_ptr(),image.host,std::size_t(bytes));
        }
        return bytes;
    }
    MoeHostExpertCache::Lease prepared_expert(const MoeExpertCache::SsdExpertReads* reads,int projection,int expert) const {
        auto* source=sources_[projection]->quant_source_.get();
        if(reads)for(std::size_t i=0;i<reads->requests.size();++i)
            if(reads->requests[i].first==source && reads->requests[i].second==expert) {
                if(!reads->leases[i])throw std::logic_error("SSD expert consumer preceded its read completion");
                static std::atomic<bool> reported{false};
                if(!reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))
                    std::fprintf(stderr,"moe_ssd_prepared_image_dispatch scope=first_consumer enabled=1\n");
                return reads->leases[i];
            }
        return source->acquire_expert(expert);
    }
    void copy_host_fields(const std::vector<mfq::HostCopy>& copies) {
        if(copies.empty())return;
        const auto started=std::chrono::steady_clock::now();
        mfq::copy_host_ranges(copies);
        cache_->stats_.pipeline_host_copy_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
    }
    void fetch_registered(Cell& cell,const MoeExpertCache::SsdExpertReads* reads=nullptr) {
        using Field=TransferScratch::Field;
        auto& scratch=cell.transfer_scratch;scratch.clear();
        // Source owners remain in pipeline_dma_leases_ through the existing
        // DMA/reader fence. Scratch records keep no additional Tensor owners.
        auto& fields=scratch.fields;
        const auto host_limit=cell.ssd_gate_up_host_tail;
        const auto* projection_option=std::getenv("MFQ_MOE_KV_PROJECTION_CACHE");
        const auto* lookup_option=std::getenv("MFQ_MOE_KV_LOOKUP_FIRST");
        const auto* direct_loan=std::getenv("MFQ_MOE_KV_DIRECT_READ");
        const bool direct_enabled=direct_loan?direct_loan[0]!='0':cell.native_dma_mode>0;
        const bool projection_cache=cell.phased_transfer && direct_enabled && cell.graph_replay &&
            cell.skip_cpu_wait && cell.two_stage && cell.transfer_cache && cell.kv_loan_table &&
            (projection_option?projection_option[0]=='1':cell.native_dma_mode>0);
        const bool lookup_first=projection_cache && (!lookup_option || lookup_option[0]=='1');
        int64_t early_hit_bytes=0,early_hit_fields=0;
        struct CacheRollback {
            std::vector<MoeCacheNewLease>& added;
            bool published=false;
            ~CacheRollback(){if(!published)for(const auto& item:added)item.book->discard(item.key,item.slot,item.generation);}
        } rollback{scratch.added};
        // Protect the whole routed working set before any miss can evict a
        // cached field belonging to a later expert in this same invocation.
        if(cell.transfer_cache && !cell.wire_fallback)for(int p=0;p<3;++p) {
            if(single_up() && p==1)continue;
            const auto& source=*sources_[p];
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                const int e=group.expert,c=source.expert_to_cohort_[e];
                if(source.cohorts_[c].arena->book->slot_for({source.id_,c,e})>=0)continue;
                auto* book=cell.binding[p].stages[c]->book.get();const mfq::MoeCacheKey key{source.id_,c,e};
                const int slot=book->slot_for(key);
                if(slot>=0) {
                    cache_->pipeline_transfer_slots_.push_back({book,key,slot,0});
                    book->mark_inflight(slot);
                }
            }
        }
        for(int projection=0;projection<3;++projection) {
            if(single_up() && projection==1)continue;
            auto& source=*sources_[projection];
            auto& slots=scratch.slots[projection];slots.assign(source.cohorts_.size(),0);
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                const int e=group.expert,c=source.expert_to_cohort_[e];int slot=slots[c]++;
                const auto& cohort=source.cohorts_[c];
                auto* stage=(!cell.two_stage || cell.transfer_cache) && !cell.wire_fallback?cell.binding[projection].stages[c]:nullptr;
                if(stage && slot>=stage->slots)throw std::logic_error("MFQ direct transfer exceeds cohort geometry");
                if(cohort.arena->book->slot_for({source.id_,c,e})>=0)continue;
                if(lookup_first) {
                    std::array<std::size_t,8> offsets{};std::size_t logical_bytes=0;
                    if(const auto* cached=cache_->kv_expert_loan_->pin_existing_projection(
                            source.id_,e,cohort.bytes_per_expert,offsets,logical_bytes)) {
                        const int index=cell.transfer_index->h<int32_t>()[e];
                        for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                            const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                            cell.transfer_views.at(index*3+projection).fields[f]=static_cast<const uint8_t*>(cached)+offsets[f];
                            ++early_hit_fields;
                            if(bytes<int64_t(mfq::cuda::KvExpertLoan::page))cache_->stats_.pipeline_kv_projection_small_hit_bytes+=bytes;
                        }
                        early_hit_bytes+=int64_t(logical_bytes);
                        (projection==2?cache_->stats_.pipeline_down_dma_bytes:cache_->stats_.pipeline_gate_up_dma_bytes)+=int64_t(logical_bytes);
                        if(++cache_->stats_.pipeline_kv_projection_hits==1)
                            std::fprintf(stderr,"moe_kv_cache_dispatch scope=first_complete_hit projection_cache=1 host_packet=%d native_dma=%d lookup_first=1\n",
                                int(cell.host_cache_packet),cell.native_dma_mode);
                        ++cache_->stats_.pipeline_kv_projection_early_hits;
                        continue;
                    }
                }
                if(cell.transfer_cache && !cell.wire_fallback) {
                    const mfq::MoeCacheKey key{source.id_,c,e};const auto held=stage->book->acquire(key);
                    slot=held.slot;
                    const MoeCacheNewLease item{stage->book.get(),key,slot,held.generation};
                    if(!held.hit)rollback.added.push_back(item);
                    cache_->pipeline_transfer_slots_.push_back(item);stage->book->mark_inflight(slot);
                    const int index=cell.transfer_index->h<int32_t>()[e];
                    for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f)
                        cell.transfer_views[index*3+projection].fields[f]=stage->field_pointer(slot,f,cohort.bytes_per_expert);
                    if(held.hit) {
                        if(std::find(cache_->pipeline_prefetched_keys_.begin(),cache_->pipeline_prefetched_keys_.end(),key)!=cache_->pipeline_prefetched_keys_.end())
                            cache_->stats_.pipeline_prefetch_hit_bytes+=source.bytes_for_expert(e);
                        ++cache_->stats_.pipeline_transfer_cache_hits;
                        cache_->stats_.pipeline_transfer_cache_saved_bytes+=source.bytes_for_expert(e);
                        continue;
                    }
                    ++cache_->stats_.pipeline_transfer_cache_misses;
                }
                // The routed read already owns this immutable image. Looking
                // it up again would mark a freshly loaded expert as reused by
                // 2Q and unnecessarily enter the shared RAM-cache mutex.
                auto lease=prepared_expert(reads,projection,e);
                const auto values=moe_cache_field_refs(lease->weights);
                if(values.size()!=cohort.bytes_per_expert.size())throw std::logic_error("MFQ RAM field count changed");
                for(std::size_t f=0;f<values.size();++f) {
                    const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                    if(bytes!=tensor_nbytes(values[f]))throw std::logic_error("MFQ RAM field geometry changed");
                    fields.push_back({stage?stage->field_pointer(slot,f,cohort.bytes_per_expert):nullptr,
                        static_cast<const uint8_t*>(values[f].data_ptr()),bytes,0,projection,e,int(f)});
                }
                cache_->pipeline_dma_leases_.push_back(std::move(lease));
            }
        }
        if(cell.wire_fallback) {
            int64_t offset=cell.wire_header_bytes;
            for(auto& field:fields) {
                offset=(offset+15)&~int64_t(15);
                if(field.bytes>cell.wire.numel()-offset)
                    throw std::logic_error("direct fallback fields exceeded their routed wire capacity");
                field.destination=cell.wire.data_ptr<uint8_t>()+offset;offset+=field.bytes;
                const int index=cell.transfer_index->h<int32_t>()[field.expert];
                cell.transfer_views.at(index*3+field.projection).fields[field.field]=field.destination;
            }
        }
        auto* packed=cache_->pipeline_host_stage_.data_ptr<uint8_t>();
        const auto begin=reinterpret_cast<uintptr_t>(cache_->pipeline_ram_complement_->data_ptr());
        const auto size=static_cast<uint64_t>(cache_->pipeline_ram_registered_bytes_);
        if(cell.mapped_copy) {
            auto* header=cell.mapped_copy_table->h<uint64_t>();
            auto* descriptors=reinterpret_cast<mfq::cuda::MoeMappedCopyDescriptor*>(header+4);
            const auto capacity=(cell.mapped_copy_table->bytes-32)/sizeof(*descriptors);
            std::size_t count=0;uint64_t total=0;
            const auto append=[&](void* destination,const void* source,uint64_t bytes) {
                if(!bytes)return;
                if(count==capacity)throw std::logic_error("mapped RAM copy descriptor geometry exceeded");
                total=(total+15)&~uint64_t(15);
                if(bytes>UINT64_MAX-total)throw std::overflow_error("mapped RAM copy byte geometry exceeded");
                descriptors[count++]={reinterpret_cast<uint64_t>(destination),reinterpret_cast<uint64_t>(source),total,bytes};
                total+=bytes;
            };
            int64_t direct_bytes=0,staged_bytes=0;
            for(const auto& field:fields) {
                const auto address=reinterpret_cast<uintptr_t>(field.source);
                const uint8_t* alias=nullptr;
                if(address>=begin && address-begin<=size && uint64_t(field.bytes)<=size-(address-begin)) {
                    alias=static_cast<const uint8_t*>(cache_->pipeline_ram_alias_)+(address-begin);
                    direct_bytes+=field.bytes;
                } else {
                    staged_bytes=(staged_bytes+15)&~int64_t(15);
                    if(field.bytes>cache_->pipeline_host_stage_.numel()-staged_bytes)
                        throw std::logic_error("mapped RAM fallback exceeds pinned stage geometry");
                    std::memcpy(packed+staged_bytes,field.source,std::size_t(field.bytes));
                    alias=static_cast<const uint8_t*>(cache_->pipeline_host_stage_alias_)+staged_bytes;
                    staged_bytes+=field.bytes;
                }
                append(field.destination,alias,field.bytes);
            }
            const auto descriptor_bytes=upload_transfer_views(cell);
            if(descriptor_bytes)append(cell.fused->transfer_buffer().data_ptr(),cell.transfer_descriptors->device,descriptor_bytes);
            header[0]=count;header[1]=total;header[2]=header[3]=0;
            int64_t payload=0;for(const auto& field:fields)payload+=field.bytes;
            cache_->stats_.ram_pcie_bytes+=payload;
            cache_->stats_.pipeline_direct_ram_bytes+=direct_bytes;
            cache_->stats_.pipeline_staged_ram_bytes+=staged_bytes;
            ++cache_->stats_.pipeline_mapped_copy_serves;
            cache_->stats_.pipeline_mapped_overlap_serves+=cell.mapped_overlap;
            cache_->stats_.pipeline_mapped_copy_bytes+=payload;
            cache_->stats_.pipeline_mapped_copy_descriptors+=count;
            if(cell.in_window)cache_->stats_.pipeline_window_dma_bytes+=payload+descriptor_bytes+32+count*sizeof(*descriptors);
            finish_transfer(cell,true);rollback.published=true;return;
        }

        // Field-major prefill storage separates small scales and format metadata
        // from each expert's payload. Pack those fields into the existing pinned
        // wire stage rather than submitting a DMA operation for every field.
        // Expert-major storage keeps its contiguous direct-copy intervals.
        const bool pack_small_fields=cache_->pipeline_prefill_layer_layout_ &&
            mfq::cuda::runtime_options::ram_field_packing();
        const auto registered_field=[&](const Field& field) {
            const auto address=reinterpret_cast<uintptr_t>(field.source);
            return address>=begin && address-begin<=size && uint64_t(field.bytes)<=size-(address-begin);
        };
        const auto direct_field=[&](const Field& field) {
            return !(pack_small_fields && field.bytes<=16*1024) && registered_field(field);
        };
        const auto native_alias=[&](const void* source,std::size_t bytes) -> CUdeviceptr {
            const auto address=reinterpret_cast<uintptr_t>(source);
            const auto in_range=[&](const void* host,const void* device,std::size_t capacity) -> CUdeviceptr {
                const auto base=reinterpret_cast<uintptr_t>(host);
                return device && address>=base && address-base<=capacity && bytes<=capacity-(address-base)
                    ?reinterpret_cast<CUdeviceptr>(device)+(address-base):0;
            };
            if(auto alias=in_range(cache_->pipeline_ram_complement_->data_ptr(),cache_->pipeline_ram_alias_,
                    cache_->pipeline_ram_registered_bytes_))return alias;
            if(auto alias=in_range(packed,cache_->pipeline_host_stage_alias_,cache_->pipeline_host_stage_.numel()))return alias;
            return in_range(cell.transfer_descriptors->host,cell.transfer_descriptors->device,cell.transfer_descriptors->bytes);
        };
        if(cell.phased_transfer) {
            auto& intervals=scratch.intervals;
            auto& cache_copies=scratch.cache_copies;
            auto& loan_copies=scratch.loan_copies;
            int64_t loan_hits=early_hit_bytes;
            auto& direct_to_cache=scratch.direct_to_cache;direct_to_cache.assign(fields.size(),false);
            const auto direct_option=mfq::cuda::runtime_options::moe_direct_cache_dma();
            const bool direct_cache=cell.transfer_cache && direct_option && *direct_option;
            std::array<int64_t,2> staged_begin{},staged_end{},staged_host_begin{};
            int64_t offset=cell.wire_header_bytes,direct_bytes=0,staged_bytes=0,gate_up_end=0;
            int64_t host_offset=cell.wire_header_bytes;
            const bool consume_loan=direct_enabled && cell.graph_replay &&
                cell.skip_cpu_wait && cell.two_stage && cell.transfer_cache && cell.kv_loan_table;
            auto& projection_copies=scratch.projection_copies;
            std::array<std::vector<mfq::HostCopy>,2> projection_host_copies;
            std::size_t submitted_projection_copies=0;
            bool projection_gu_submitted=false;
            const auto* overlap_option=std::getenv("MFQ_MOE_GU_BUILD_OVERLAP");
            const bool overlap_gu_build=cell.wire_fallback && cell.direct_packet && cell.native_dma_mode &&
                (!overlap_option || overlap_option[0]!='0');
            auto& projection_fields=scratch.projection_fields;projection_fields.assign(fields.size(),false);
            int64_t projection_field_count=early_hit_fields,projection_field_bytes=early_hit_bytes;
            cache_->stats_.pipeline_kv_direct_read_bytes+=early_hit_bytes;
            cache_->stats_.pipeline_kv_direct_read_fields+=early_hit_fields;
            if(projection_cache) {
                auto& protected_keys=scratch.protected_keys;
                protected_keys.reserve(fields.size()*2);
                for(std::size_t i=0;i<fields.size();++i) {
                    const auto& f=fields[i];const auto source=sources_[f.projection]->id_;
                    if(!i || f.projection!=fields[i-1].projection || f.expert!=fields[i-1].expert)
                        protected_keys.push_back({source,f.expert,-1});
                    if(f.bytes>=int64_t(mfq::cuda::KvExpertLoan::page))protected_keys.push_back({source,f.expert,f.field});
                }
                cache_->kv_expert_loan_->protect_packet_fields(protected_keys);
                for(std::size_t first=0;first<fields.size();) {
                    const auto& initial=fields[first];auto last=first+1;
                    while(last<fields.size() && fields[last].projection==initial.projection &&
                        fields[last].expert==initial.expert)++last;
                    if(overlap_gu_build && initial.projection==2 && !projection_copies[0].empty()) {
                        // Pin/protect of every existing packet member preceded
                        // this allocation pass. GU's destinations and sources
                        // are now final and live until the next reader fence.
                        // Start its upload while the CPU allocates Down's pages.
                        // Readiness still follows the complete descriptor image,
                        // including Down metadata needed by hidden quantization.
                        copy_host_fields(projection_host_copies[0]);
                        projection_host_copies[0].clear();
                        const auto& copies=projection_copies[0];
                        if(mfq::cuda::NativeDmaBatch::submit(copies,cache_->weight_stream_,cell.native_dma_mode,native_alias)) {
                            ++cache_->stats_.pipeline_native_dma_batches;
                            cache_->stats_.pipeline_native_dma_copies+=copies.size();
                        } else {
                            for(const auto& copy:copies)MFQ_CUDA_CHECK(cudaMemcpyAsync(
                                copy.destination,copy.source,copy.bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
                        }
                        mfq::cuda::check_stream_progress(cache_->weight_stream_);
                        submitted_projection_copies=copies.size();projection_gu_submitted=true;
                        cache_->stats_.pipeline_gu_build_overlap_copies+=copies.size();
                        if(++cache_->stats_.pipeline_gu_build_overlap_serves==1)
                            std::fprintf(stderr,"moe_gu_build_overlap_dispatch scope=first_upload enabled=1 tokens=%d copies=%zu\n",
                                cell.tokens,copies.size());
                        projection_copies[0].clear();
                    }
                    const auto& source=*sources_[initial.projection];
                    const auto& cohort=source.cohorts_[source.expert_to_cohort_[initial.expert]];
                    bool hit=false;const int phase=initial.projection==2;
                    const auto* cached=cache_->kv_expert_loan_->pin_projection(source.id_,initial.expert,
                        cohort.bytes_per_expert,cache_->weight_stream_,hit,
                        [&](void* destination,const std::array<std::size_t,8>& offsets) {
                            auto& copies=projection_copies[phase];const auto first_copy=copies.size();
                            for(auto i=first;i<last;++i) {
                                const auto& f=fields[i];const auto address=reinterpret_cast<uintptr_t>(f.source);
                                const void* origin=f.source;
                                if(address>=begin && address-begin<=size && uint64_t(f.bytes)<=size-(address-begin))
                                    direct_bytes+=f.bytes;
                                else {
                                    host_offset=(host_offset+15)&~int64_t(15);
                                    if(f.bytes>host_limit-host_offset)
                                        throw std::logic_error("KV projection upload exceeds pinned stage geometry");
                                    origin=packed+host_offset;
                                    projection_host_copies[phase].push_back({packed+host_offset,f.source,std::size_t(f.bytes)});
                                    host_offset+=f.bytes;staged_bytes+=f.bytes;
                                }
                                auto* target=static_cast<uint8_t*>(destination)+offsets[f.field];
                                if(copies.size()>first_copy) {
                                    auto& previous=copies.back();
                                    const auto old_source=reinterpret_cast<uintptr_t>(previous.source);
                                    const auto source=reinterpret_cast<uintptr_t>(origin);
                                    const auto old_target=reinterpret_cast<uintptr_t>(previous.destination);
                                    const auto next_target=reinterpret_cast<uintptr_t>(target);
                                    // All destinations in this callback share one
                                    // GPU projection allocation. Merge only the
                                    // matching intervals inside the registered
                                    // host image; small alignment gaps are owned.
                                    if(old_source>=begin && old_source-begin<=size &&
                                            source>=old_source+previous.bytes && source-old_source-previous.bytes<=15 &&
                                            next_target>=old_target+previous.bytes &&
                                            next_target-old_target==source-old_source &&
                                            source-old_source+std::size_t(f.bytes)<=size-(old_source-begin)) {
                                        previous.bytes=source-old_source+std::size_t(f.bytes);
                                        ++cache_->stats_.pipeline_projection_dma_merges;
                                        continue;
                                    }
                                }
                                copies.push_back({target,origin,std::size_t(f.bytes)});
                            }
                        });
                    if(cached) {
                        std::size_t field_offset=0;
                        std::array<std::size_t,8> offsets{};
                        for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                            field_offset=(field_offset+15)&~std::size_t(15);offsets[f]=field_offset;
                            field_offset+=std::size_t(cohort.bytes_per_expert[f]);
                        }
                        const int index=cell.transfer_index->h<int32_t>()[initial.expert];
                        for(auto i=first;i<last;++i) {
                            const auto& f=fields[i];projection_fields[i]=direct_to_cache[i]=true;
                            cell.transfer_views.at(index*3+f.projection).fields[f.field]=static_cast<const uint8_t*>(cached)+offsets[f.field];
                            ++projection_field_count;projection_field_bytes+=f.bytes;
                            cache_->stats_.pipeline_kv_direct_read_bytes+=f.bytes;++cache_->stats_.pipeline_kv_direct_read_fields;
                            if(hit) {
                                loan_hits+=f.bytes;
                                if(f.bytes<int64_t(mfq::cuda::KvExpertLoan::page))cache_->stats_.pipeline_kv_projection_small_hit_bytes+=f.bytes;
                            }
                        }
                        for(auto& held:cache_->pipeline_transfer_slots_)
                            if(held.key.source==source.id_ && held.key.expert==initial.expert)held.aliased=true;
                        if(hit) {
                            if(++cache_->stats_.pipeline_kv_projection_hits==1)
                                std::fprintf(stderr,"moe_kv_cache_dispatch scope=first_complete_hit projection_cache=1 host_packet=%d native_dma=%d lookup_first=%d\n",
                                    int(cell.host_cache_packet),cell.native_dma_mode,int(lookup_first));
                        }
                        else ++cache_->stats_.pipeline_kv_projection_misses;
                    } else ++cache_->stats_.pipeline_kv_projection_fallbacks;
                    first=last;
                }
            }
            // Every staged source must be ready before either phase submits
            // its DMA, including the early Gate/Up submission above.
            for(const auto& host_copies:projection_host_copies)copy_host_fields(host_copies);
            // A packet boundary separates Gate/Up from Down. Cache ownership,
            // original RAM leases and the final completion event are shared.
            for(int phase=0;phase<2;++phase) {
                auto& direct=scratch.direct;auto& staged=scratch.staged;direct.clear();staged.clear();
                for(std::size_t i=0;i<fields.size();++i) {
                    if((fields[i].projection==2)!=(phase==1) || projection_fields[i])continue;
                    if(cell.kv_loan_table && fields[i].destination &&
                            fields[i].bytes>=int64_t(mfq::cuda::KvExpertLoan::page)) {
                        loan_copies[phase].push_back(&fields[i]);direct_to_cache[i]=true;
                        if(direct_field(fields[i]))direct_bytes+=fields[i].bytes;
                    } else if(direct_cache && fields[i].bytes>16*1024 && direct_field(fields[i])) {
                        // The protected cache slot is the final consumer's
                        // storage. DMA large registered payloads there once;
                        // only small metadata needs the wire scatter. Both
                        // copies precede the original phase-ready publication.
                        const auto& field=fields[i];
                        cache_copies[phase].push_back({field.destination,field.source,std::size_t(field.bytes)});
                        if(cell.wire_fallback)cache_->pipeline_wire_fallback_bytes_+=field.bytes;
                        direct_to_cache[i]=true;direct_bytes+=field.bytes;
                    }
                    else if(cell.direct_packet) {
                        const auto& field=fields[i];
                        if(!field.destination)throw std::logic_error("direct packet field has no final destination");
                        const void* source=field.source;
                        // Native packets submit these fields separately even
                        // when small-field wire packing is enabled. Registered
                        // metadata already has a stable pinned source.
                        if(registered_field(field))direct_bytes+=field.bytes;
                        else {
                            host_offset=(host_offset+15)&~int64_t(15);
                            if(field.bytes>host_limit-host_offset)
                                throw std::logic_error("direct packet staging exceeds its pinned capacity");
                            auto* staged=packed+host_offset;
                            std::memcpy(staged,source,std::size_t(field.bytes));
                            source=staged;host_offset+=field.bytes;staged_bytes+=field.bytes;
                        }
                        cache_copies[phase].push_back({field.destination,source,std::size_t(field.bytes)});
                        if(cell.wire_fallback)cache_->pipeline_wire_fallback_bytes_+=field.bytes;
                        direct_to_cache[i]=true;
                    }
                    else if(direct_field(fields[i]))direct.push_back(i);
                    else staged.push_back(i);
                }
                std::sort(direct.begin(),direct.end(),[&](auto a,auto b) {
                    return reinterpret_cast<uintptr_t>(fields[a].source)<reinterpret_cast<uintptr_t>(fields[b].source);
                });
                for(std::size_t first=0;first<direct.size();) {
                    const auto address=reinterpret_cast<uintptr_t>(fields[direct[first]].source);
                    auto end=address+fields[direct[first]].bytes;auto last=first+1;
                    while(last<direct.size()) {
                        const auto next=reinterpret_cast<uintptr_t>(fields[direct[last]].source);
                        // The wire scatter visits 16-byte boundaries; every
                        // field descriptor must begin on one of them.
                        if(next<end || next-end>15 || ((next-address)&15))break;
                        end=next+fields[direct[last]].bytes;++last;
                    }
                    offset=(offset+15)&~int64_t(15);
                    const auto bytes=static_cast<int64_t>(end-address);
                    if(bytes>cell.wire.numel()-offset)throw std::logic_error("MFQ phased RAM interval exceeds wire geometry");
                    for(auto i=first;i<last;++i)fields[direct[i]].wire_offset=offset+
                        static_cast<int64_t>(reinterpret_cast<uintptr_t>(fields[direct[i]].source)-address);
                    intervals[phase].push_back({reinterpret_cast<const uint8_t*>(address),bytes,offset});
                    offset+=bytes;direct_bytes+=bytes;first=last;
                }
                offset=(offset+15)&~int64_t(15);staged_begin[phase]=offset;
                host_offset=(host_offset+15)&~int64_t(15);staged_host_begin[phase]=host_offset;
                for(const auto i:staged) {
                    offset=(offset+15)&~int64_t(15);auto& field=fields[i];
                    host_offset=staged_host_begin[phase]+offset-staged_begin[phase];
                    if(field.bytes>cell.wire.numel()-offset || field.bytes>host_limit-host_offset)
                        throw std::logic_error("MFQ phased exchanged RAM field exceeds wire geometry: layer="+
                            std::to_string(sources_[0]->layer_id_)+" projection="+std::to_string(field.projection)+
                            " expert="+std::to_string(field.expert)+" field="+std::to_string(field.field)+
                            " bytes="+std::to_string(field.bytes)+" wire="+std::to_string(cell.wire.numel())+
                            " offset="+std::to_string(offset)+" host="+std::to_string(cache_->pipeline_host_stage_.numel())+
                            " host_offset="+std::to_string(host_offset));
                    std::memcpy(packed+host_offset,field.source,std::size_t(field.bytes));
                    field.wire_offset=offset;offset+=field.bytes;host_offset+=field.bytes;staged_bytes+=field.bytes;
                }
                staged_end[phase]=offset;
                if(!phase)gate_up_end=(offset+15)&~int64_t(15);
            }
            if(32+fields.size()*24>std::size_t(cell.wire_header_bytes))
                throw std::logic_error("MFQ phased RAM descriptors exceed route geometry");
            auto& wire_fields=scratch.wire_fields;
            for(std::size_t i=0;i<fields.size();++i)if(!direct_to_cache[i])wire_fields.push_back(i);
            std::sort(wire_fields.begin(),wire_fields.end(),[&](auto a,auto b){return fields[a].wire_offset<fields[b].wire_offset;});
            auto* words=reinterpret_cast<uint64_t*>(packed);
            words[0]=wire_fields.size();words[1]=cell.wire_header_bytes;words[2]=offset;words[3]=gate_up_end;
            int64_t payload_bytes=early_hit_bytes;
            for(std::size_t i=0;i<fields.size();++i) {
                payload_bytes+=fields[i].bytes;
                (fields[i].projection==2?cache_->stats_.pipeline_down_dma_bytes:cache_->stats_.pipeline_gate_up_dma_bytes)+=fields[i].bytes;
            }
            for(std::size_t i=0;i<wire_fields.size();++i) {
                const auto& field=fields[wire_fields[i]];
                words[4+i*3]=reinterpret_cast<uint64_t>(field.destination);
                words[5+i*3]=field.wire_offset;words[6+i*3]=field.bytes;
            }
            auto* wire=cell.wire.data_ptr<uint8_t>();
            const bool batched=mfq::cuda::runtime_options::dma_graph_batch() || cell.native_dma_mode;
            auto& batches=projection_copies;
            const auto projection_copy_count=submitted_projection_copies+batches[0].size()+batches[1].size();
            int64_t copies=1+projection_field_count,wire_bytes=cell.wire_header_bytes+projection_field_bytes;
            const auto* early_gu_option=std::getenv("MFQ_MOE_KV_EARLY_GU");
            const bool early_gu=consume_loan && early_gu_option && early_gu_option[0]=='1' && (!batched || cell.native_dma_mode);
            bool early_gu_submitted=projection_gu_submitted || cell.ssd_gate_up_submitted;
            if(consume_loan)for(int phase=0;phase<2;++phase) {
            for(const auto* field:loan_copies[phase]) {
                const auto stage_source=[&]() -> const void* {
                    if(direct_field(*field))return field->source;
                    host_offset=(host_offset+15)&~int64_t(15);
                    if(field->bytes>host_limit-host_offset)
                        throw std::logic_error("pageable direct KV expert loan exceeds pinned stage geometry");
                    auto* source=packed+host_offset;
                    std::memcpy(source,field->source,std::size_t(field->bytes));
                    host_offset+=field->bytes;staged_bytes+=field->bytes;
                    return source;
                };
                bool hit=false;
                // Collect both phases before uploading the descriptor image;
                // their payloads are still submitted in Gate/Up, Down order.
                const auto* cached=cache_->kv_expert_loan_->pin(
                    {sources_[field->projection]->id_,field->expert,field->field},
                    field->source,std::size_t(field->bytes),cache_->weight_stream_,hit,
                    &batches[phase],stage_source);
                if(cached) {
                    const int index=cell.transfer_index->h<int32_t>()[field->expert];
                    cell.transfer_views.at(index*3+field->projection).fields[field->field]=cached;
                    // The large field remains in its loan page. Do not claim
                    // that this partially populated staging slot is a hit.
                    for(auto& held:cache_->pipeline_transfer_slots_)
                        if(held.key.source==sources_[field->projection]->id_ && held.key.expert==field->expert)
                            held.aliased=true;
                    cache_->stats_.pipeline_kv_direct_read_bytes+=field->bytes;
                    ++cache_->stats_.pipeline_kv_direct_read_fields;
                } else {
                    batches[phase].push_back({field->destination,stage_source(),std::size_t(field->bytes)});
                    if(cell.wire_fallback)cache_->pipeline_wire_fallback_bytes_+=field->bytes;
                }
                if(hit)loan_hits+=field->bytes;
                ++copies;wire_bytes+=field->bytes;
            }
            if(!phase && early_gu) {
                // Upload the large Gate/Up misses while preparing Down's loan
                // descriptors. The ready event still follows the complete
                // descriptor image and small metadata, as before.
                if(!batches[0].empty()) {
                    if(mfq::cuda::NativeDmaBatch::submit(batches[0],cache_->weight_stream_,cell.native_dma_mode,native_alias)) {
                        ++cache_->stats_.pipeline_native_dma_batches;
                        cache_->stats_.pipeline_native_dma_copies+=batches[0].size();
                    } else {
                        if(cell.native_dma_mode)++cache_->stats_.pipeline_native_dma_fallbacks;
                        for(const auto& copy:batches[0])MFQ_CUDA_CHECK(cudaMemcpyAsync(
                            copy.destination,copy.source,copy.bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
                    }
                    ++cache_->stats_.pipeline_kv_early_gu_serves;
                    early_gu_submitted=true;
                    batches[0].clear();
                }
            }
            }
            const bool skip_wire=cell.host_cache_packet && wire_fields.empty();
            if(cell.direct_packet && (!skip_wire || !consume_loan))
                throw std::logic_error("direct packet unexpectedly requires a GPU scatter");
            if(cell.direct_packet) {
                ++cache_->stats_.pipeline_direct_packet_serves;
                if(cell.tokens>1)++cache_->stats_.pipeline_direct_packet_batch_serves;
            }
            if(!skip_wire) {
                if(batched)batches[0].push_back({wire,packed,std::size_t(cell.wire_header_bytes)});
                else MFQ_CUDA_CHECK(cudaMemcpyAsync(wire,packed,std::size_t(cell.wire_header_bytes),cudaMemcpyHostToDevice,cache_->weight_stream_));
            }
            // Down's immutable format metadata must arrive before hidden-group
            // quantization; its weight fields can remain in flight until use.
            const auto* mapped_views=std::getenv("MFQ_MOE_GU_BUILD_MAPPED_VIEWS");
            const bool early_mapped_views=projection_gu_submitted && cell.direct_packet &&
                (!mapped_views || mapped_views[0]!='0');
            // The wait kernel already snapshots host views after the phase's
            // release/acquire flag. An early GU upload need not add a second
            // DMA submission solely for these views: the same ordered ready
            // notification covers its payload and the completed host image.
            const bool snapshot_descriptors=cell.host_cache_packet && (cell.ssd_gate_up_ready || early_mapped_views ||
                (!early_gu_submitted && batches[0].empty() && cache_copies[0].empty() &&
                 intervals[0].empty() && staged_end[0]==staged_begin[0]));
            const auto descriptor_bytes=upload_transfer_views(cell,batched?&batches[0]:nullptr,!snapshot_descriptors);
            copies+=descriptor_bytes>0;wire_bytes+=descriptor_bytes;
            std::array<bool,2> no_dma{};
            if(cell.host_cache_packet) {
                ++cache_->stats_.pipeline_host_packet_serves;
                cache_->stats_.pipeline_host_packet_descriptor_bytes+=snapshot_descriptors?descriptor_bytes:0;
            }
            for(int phase=0;phase<2;++phase) {
                if(!phase && cell.ssd_gate_up_ready) {
                    if(!batches[0].empty() || !cache_copies[0].empty() || !intervals[0].empty() ||
                            !loan_copies[0].empty() || staged_end[0]!=staged_begin[0] || !skip_wire)
                        throw std::logic_error("SSD early GU packet changed its completed fields");
                    continue;
                }
                auto* loan_header=cell.kv_loan_table?reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(cell.kv_loan_table->host)+phase*cell.kv_loan_table->bytes/2):nullptr;
                if(loan_header){loan_header[0]=loan_header[1]=loan_header[2]=loan_header[3]=0;}
                if(cell.host_cache_packet) {
                    // Empty wire and weight DMA are independent of whether
                    // this route has transfer-classified cached projections.
                    // The existing snapshot guard suppresses stale wire.
                    if(skip_wire)loan_header[0]=~uint64_t(0);
                    if(snapshot_descriptors && ((!phase && !cell.ssd_gate_up_ready) || (phase && cell.ssd_gate_up_ready)))
                        loan_header[2]=uint64_t(descriptor_bytes)/sizeof(uint64_t);
                }
                if(!consume_loan)for(const auto* field:loan_copies[phase]) {
                    bool hit=false;
                    const auto stage_source=[&]() -> const void* {
                        if(direct_field(*field))return field->source;
                        host_offset=(host_offset+15)&~int64_t(15);
                        if(field->bytes>host_limit-host_offset)
                            throw std::logic_error("pageable KV expert loan exceeds pinned stage geometry");
                        auto* source=packed+host_offset;
                        std::memcpy(source,field->source,std::size_t(field->bytes));
                        host_offset+=field->bytes;staged_bytes+=field->bytes;
                        return source;
                    };
                    const auto* cached=cache_->kv_expert_loan_->pin({sources_[field->projection]->id_,field->expert,field->field},
                        field->source,std::size_t(field->bytes),cache_->weight_stream_,hit,batched?&batches[phase]:nullptr,stage_source);
                    if(!cached) {
                        const auto* source=stage_source();
                        if(batched)batches[phase].push_back({field->destination,source,std::size_t(field->bytes)});
                        else MFQ_CUDA_CHECK(cudaMemcpyAsync(field->destination,source,std::size_t(field->bytes),
                            cudaMemcpyHostToDevice,cache_->weight_stream_));
                    }
                    else {
                        auto* descriptors=reinterpret_cast<mfq::cuda::MoeMappedCopyDescriptor*>(loan_header+4);
                        descriptors[loan_header[0]++]={reinterpret_cast<uint64_t>(field->destination),reinterpret_cast<uint64_t>(cached),loan_header[1],uint64_t(field->bytes)};
                        loan_header[1]=(loan_header[1]+field->bytes+15)&~uint64_t(15);
                    }
                    if(hit)loan_hits+=field->bytes;
                    ++copies;wire_bytes+=field->bytes;
                }
                if(loan_header) {
                    // Route publication fences the preceding reader. Publish
                    // this small mapped descriptor packet before the existing
                    // DMA-ready flag instead of adding a transfer per phase.
                    std::atomic_thread_fence(std::memory_order_seq_cst);_mm_sfence();
                }
                if(consume_loan && !batched)for(const auto& copy:batches[phase])
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,
                        cudaMemcpyHostToDevice,cache_->weight_stream_));
                for(const auto& copy:cache_copies[phase]) {
                    if(batched)batches[phase].push_back(copy);
                    else MFQ_CUDA_CHECK(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,
                        cudaMemcpyHostToDevice,cache_->weight_stream_));
                    ++copies;wire_bytes+=copy.bytes;
                }
                for(const auto& interval:intervals[phase]) {
                    if(batched)batches[phase].push_back({wire+interval.offset,interval.source,std::size_t(interval.bytes)});
                    else MFQ_CUDA_CHECK(cudaMemcpyAsync(wire+interval.offset,interval.source,
                        std::size_t(interval.bytes),cudaMemcpyHostToDevice,cache_->weight_stream_));
                    ++copies;wire_bytes+=interval.bytes;
                }
                if(staged_end[phase]>staged_begin[phase]) {
                    const auto bytes=staged_end[phase]-staged_begin[phase];
                    if(batched)batches[phase].push_back({wire+staged_begin[phase],packed+staged_host_begin[phase],std::size_t(bytes)});
                    else MFQ_CUDA_CHECK(cudaMemcpyAsync(wire+staged_begin[phase],packed+staged_host_begin[phase],
                        std::size_t(bytes),cudaMemcpyHostToDevice,cache_->weight_stream_));
                    ++copies;wire_bytes+=bytes;
                }
                // An early Down upload remains in flight even when lookup
                // now finds every projection in the GPU cache. Its existing
                // DMA completion event must precede the Down-ready flag.
                no_dma[phase]=cell.host_cache_packet && batched && batches[phase].empty() &&
                    (phase || !early_gu_submitted) && !(phase && cell.ssd_down_submitted);
                if(no_dma[phase]) {
                    publish(cell.flags.h<uint32_t>()+(phase?80:32));
                    ++cache_->stats_.pipeline_cached_phase_publishes;
                    continue;
                }
                if(batched && !batches[phase].empty()) {
                    if(mfq::cuda::NativeDmaBatch::submit(batches[phase],cache_->weight_stream_,cell.native_dma_mode,native_alias)) {
                        ++cache_->stats_.pipeline_native_dma_batches;
                        cache_->stats_.pipeline_native_dma_copies+=batches[phase].size();
                        if(!phase)queue_transfer_ready(cell,32);
                        continue;
                    }
                    if(cell.native_dma_mode)++cache_->stats_.pipeline_native_dma_fallbacks;
                    auto& batch=cache_->pipeline_dma_batches_[phase];
                    if(batch && batch->submit(batches[phase]))++cache_->stats_.pipeline_dma_graph_launches;
                    else {
                        if(!batch)for(const auto& copy:batches[phase])
                            MFQ_CUDA_CHECK(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,
                                cudaMemcpyHostToDevice,cache_->weight_stream_));
                        ++cache_->stats_.pipeline_dma_graph_fallbacks;
                    }
                }
                if(!phase)queue_transfer_ready(cell,32);
            }
            cache_->stats_.ram_pcie_bytes+=payload_bytes-loan_hits;
            cache_->stats_.pipeline_dma_copies+=copies;
            cache_->stats_.pipeline_direct_ram_bytes+=direct_bytes;
            cache_->stats_.pipeline_direct_ram_copies+=intervals[0].size()+intervals[1].size()+
                cache_copies[0].size()+cache_copies[1].size()+projection_copy_count;
            cache_->stats_.pipeline_staged_ram_bytes+=staged_bytes;
            if(cell.in_window) {
                cache_->stats_.pipeline_window_dma_copies+=copies;cache_->stats_.pipeline_window_dma_bytes+=wire_bytes;
            }
            if(no_dma[0] && no_dma[1])cache_->transfer_ready_recorded_=false;
            else finish_transfer(cell,!no_dma[1]);
            rollback.published=true;return;
        }

        auto& direct=scratch.direct;auto& staged=scratch.staged;direct.clear();staged.clear();
        for(std::size_t i=0;i<fields.size();++i) {
            if(direct_field(fields[i]))direct.push_back(i);
            else staged.push_back(i);
        }
        std::sort(direct.begin(),direct.end(),[&](auto a,auto b) {
            return reinterpret_cast<uintptr_t>(fields[a].source)<reinterpret_cast<uintptr_t>(fields[b].source);
        });
        int64_t offset=cell.wire_header_bytes,direct_bytes=0,staged_bytes=0;
        auto& intervals=scratch.intervals[0];
        for(std::size_t first=0;first<direct.size();) {
            const auto address=reinterpret_cast<uintptr_t>(fields[direct[first]].source);
            auto end=address+fields[direct[first]].bytes;auto last=first+1;
            while(last<direct.size()) {
                const auto next=reinterpret_cast<uintptr_t>(fields[direct[last]].source);
                if(next<end || next-end>15 || ((next-address)&15))break;
                end=next+fields[direct[last]].bytes;++last;
            }
            offset=(offset+15)&~int64_t(15);
            const auto bytes=static_cast<int64_t>(end-address);
            if(bytes>cell.wire.numel()-offset)throw std::logic_error("MFQ direct RAM interval exceeds wire geometry");
            for(auto i=first;i<last;++i)fields[direct[i]].wire_offset=offset+
                static_cast<int64_t>(reinterpret_cast<uintptr_t>(fields[direct[i]].source)-address);
            intervals.push_back({reinterpret_cast<const uint8_t*>(address),bytes,offset});
            offset+=bytes;direct_bytes+=bytes;first=last;
        }
        const auto staged_start=staged.empty()?offset:(offset+15)&~int64_t(15);offset=staged_start;
        for(const auto i:staged) {
            offset=(offset+15)&~int64_t(15);auto& field=fields[i];
            const auto host_offset=cell.wire_header_bytes+offset-staged_start;
            if(field.bytes>cell.wire.numel()-offset || field.bytes>cache_->pipeline_host_stage_.numel()-host_offset)
                throw std::logic_error("MFQ exchanged RAM field exceeds wire geometry");
            std::memcpy(packed+host_offset,field.source,static_cast<std::size_t>(field.bytes));
            field.wire_offset=offset;offset+=field.bytes;staged_bytes+=field.bytes;
        }
        if(!staged.empty() && offset>cell.wire.numel())throw std::logic_error("MFQ staged RAM transfer exceeds wire geometry");
        if(32+fields.size()*24>static_cast<std::size_t>(cell.wire_header_bytes))
            throw std::logic_error("MFQ direct RAM descriptors exceed route geometry");
        std::sort(fields.begin(),fields.end(),[](const auto& a,const auto& b){return a.wire_offset<b.wire_offset;});
        auto* words=reinterpret_cast<uint64_t*>(packed);
        words[0]=fields.size();words[1]=cell.wire_header_bytes;words[2]=offset;words[3]=0;
        int64_t payload_bytes=0;
        for(std::size_t i=0;i<fields.size();++i) {
            words[4+i*3]=reinterpret_cast<uint64_t>(fields[i].destination);
            words[5+i*3]=fields[i].wire_offset;words[6+i*3]=fields[i].bytes;payload_bytes+=fields[i].bytes;
        }
        auto* wire=cell.wire.data_ptr<uint8_t>();
        if(cell.two_stage && !cell.transfer_cache)for(const auto& field:fields) {
            const int index=cell.transfer_index->h<int32_t>()[field.expert];
            cell.transfer_views[index*3+field.projection].fields[field.field]=wire+field.wire_offset;
        }
        copy_transfer(cell,wire,packed,static_cast<std::size_t>(cell.wire_header_bytes));
        for(const auto& interval:intervals)copy_transfer(cell,wire+interval.offset,interval.source,static_cast<std::size_t>(interval.bytes));
        if(!staged.empty())copy_transfer(cell,wire+staged_start,packed+cell.wire_header_bytes,static_cast<std::size_t>(offset-staged_start));
        const auto descriptor_bytes=upload_transfer_views(cell);
        const int64_t copies=1+intervals.size()+!staged.empty()+(descriptor_bytes>0);
        cache_->stats_.ram_pcie_bytes+=payload_bytes;
        cache_->stats_.pipeline_dma_copies+=copies;
        cache_->stats_.pipeline_direct_ram_bytes+=direct_bytes;
        cache_->stats_.pipeline_direct_ram_copies+=intervals.size();
        cache_->stats_.pipeline_staged_ram_bytes+=staged_bytes;
        if(cell.in_window) {
            cache_->stats_.pipeline_window_dma_copies+=copies;
            cache_->stats_.pipeline_window_dma_bytes+=descriptor_bytes+cell.wire_header_bytes+direct_bytes+
                (staged.empty()?0:offset-staged_start);
        }
        finish_transfer(cell,true);
        rollback.published=true;
    }
    void upload_ssd_projections(Cell& cell,const MoeExpertCache::SsdExpertReads& reads,
            const std::vector<std::size_t>& ready,bool during_gate_up_reads=false) {
        auto& loan=*cache_->kv_expert_loan_;
        // Reserve every routed hit before GU allocations can evict a Down
        // projection omitted from the read batch by the GPU's snapshot.
        if(!cell.ssd_gate_up_submitted && !cell.ssd_down_submitted) {
            // Protection lasts through this packet. Subsequent allocations
            // are born pinned to the same packet, so ready batches need not
            // rebuild and rehash every routed field again.
            auto& protected_keys=cell.transfer_scratch.protected_keys;
            protected_keys.clear();
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer)
                for(int p=0;p<3;++p) {
                    const auto& source=*sources_[p];
                    const auto& sizes=source.cohorts_[source.expert_to_cohort_[group.expert]].bytes_per_expert;
                    protected_keys.push_back({source.id_,group.expert,-1});
                    for(std::size_t f=0;f<sizes.size();++f)if(sizes[f]>=int64_t(mfq::cuda::KvExpertLoan::page))
                        protected_keys.push_back({source.id_,group.expert,int64_t(f)});
                }
            loan.protect_packet_fields(protected_keys);
        }
        const auto begin=reinterpret_cast<uintptr_t>(cache_->pipeline_ram_complement_->data_ptr());
        const auto capacity=cache_->pipeline_ram_registered_bytes_;
        const auto ram_alias=[&](const void* pointer,std::size_t bytes)->CUdeviceptr {
            const auto address=reinterpret_cast<uintptr_t>(pointer);
            return address>=begin && address-begin<=capacity && bytes<=capacity-(address-begin)
                ?reinterpret_cast<CUdeviceptr>(cache_->pipeline_ram_alias_)+(address-begin):0;
        };
        auto* host=cache_->pipeline_host_stage_.data_ptr<uint8_t>();
        const auto host_begin=reinterpret_cast<uintptr_t>(host);
        const auto host_bytes=std::size_t(cache_->pipeline_host_stage_.numel());
        const auto alias=[&](const void* pointer,std::size_t bytes)->CUdeviceptr {
            if(const auto address=ram_alias(pointer,bytes))return address;
            const auto address=reinterpret_cast<uintptr_t>(pointer);
            return cache_->pipeline_host_stage_alias_ && address>=host_begin && address-host_begin<=host_bytes &&
                bytes<=host_bytes-(address-host_begin)
                ?reinterpret_cast<CUdeviceptr>(cache_->pipeline_host_stage_alias_)+(address-host_begin):0;
        };
        std::vector<mfq::cuda::DmaCopy> copies;
        std::vector<mfq::HostCopy> host_copies;
        std::size_t bytes=0,staged_bytes=0;
        const bool down=!ready.empty() && ready.front()>=reads.gate_up_requests;
        for(const auto i:ready) {
            if(i>=reads.requests.size() || !reads.leases[i] ||
                    (i>=reads.gate_up_requests)!=down)
                throw std::logic_error("SSD projection batch preceded its read completion");
            const auto& [quant,expert]=reads.requests[i];
            const int p=down?2:quant==sources_[0]->quant_source_.get()?0:1;
            if(quant!=sources_[p]->quant_source_.get())
                throw std::logic_error("SSD projection read lost its source");
            const auto& source=*sources_[p];
            const auto& sizes=source.cohorts_[source.expert_to_cohort_[expert]].bytes_per_expert;
            std::size_t total=0;for(const auto size:sizes)total=((total+15)&~std::size_t(15))+std::size_t(size);
            if(loan.find_for_exchange({source.id_,expert,-1},total))continue;
            const auto& lease=reads.leases[i];
            const auto fields=moe_cache_field_refs(lease->weights);
            if(fields.size()!=sizes.size())throw std::logic_error("SSD GU field count changed");
            std::size_t staging_bytes=0;
            for(std::size_t f=0;f<sizes.size();++f)if(sizes[f]) {
                if(tensor_nbytes(fields[f])!=sizes[f])throw std::logic_error("SSD GU field size changed");
                if(!ram_alias(fields[f].data_ptr(),std::size_t(sizes[f])))
                    staging_bytes+=((std::size_t(sizes[f])+15)&~std::size_t(15));
            }
            // Completed projections own the existing pinned packet's tail.
            // Remaining payload/header preparation uses its prefix until the same
            // completion fence, without another allocation or source lifetime.
            if(staging_bytes>std::size_t(cell.ssd_gate_up_host_tail-cell.wire_header_bytes))continue;
            bool hit=false;
            const auto* cached=loan.pin_projection(source.id_,expert,sizes,cache_->weight_stream_,hit,
                [&](void* destination,const std::array<std::size_t,8>& offsets) {
                    cache_->pipeline_dma_leases_.push_back(lease);
                    const auto first_copy=copies.size();
                    cell.ssd_gate_up_host_tail-=int64_t(staging_bytes);
                    std::size_t staging_offset=0;
                    for(std::size_t f=0;f<sizes.size();++f)if(sizes[f]) {
                        auto* target=static_cast<uint8_t*>(destination)+offsets[f];
                        const void* origin=fields[f].data_ptr();
                        if(!ram_alias(origin,std::size_t(sizes[f]))) {
                            auto* staged=host+cell.ssd_gate_up_host_tail+staging_offset;
                            host_copies.push_back({staged,origin,std::size_t(sizes[f])});origin=staged;
                            staging_offset+=((std::size_t(sizes[f])+15)&~std::size_t(15));
                            staged_bytes+=std::size_t(sizes[f]);
                        }
                        bool merged=false;
                        if(copies.size()>first_copy) {
                            auto& previous=copies.back();
                            const auto gap=reinterpret_cast<uintptr_t>(origin)-reinterpret_cast<uintptr_t>(previous.source);
                            if(gap>=previous.bytes && gap-previous.bytes<=15 &&
                                    reinterpret_cast<uintptr_t>(target)-reinterpret_cast<uintptr_t>(previous.destination)==gap) {
                                previous.bytes=gap+std::size_t(sizes[f]);merged=true;
                                ++cache_->stats_.pipeline_projection_dma_merges;
                            }
                        }
                        if(!merged)copies.push_back({target,origin,std::size_t(sizes[f])});
                        bytes+=std::size_t(sizes[f]);
                    }
                });
            if(cached && !hit)++cache_->stats_.pipeline_kv_projection_misses;
        }
        if(copies.empty())return;
        copy_host_fields(host_copies);
        if(staged_bytes && !cache_->pipeline_host_stage_alias_)
            MFQ_CUDA_CHECK(cudaHostGetDevicePointer(&cache_->pipeline_host_stage_alias_,host,0));
        if(mfq::cuda::NativeDmaBatch::submit(copies,cache_->weight_stream_,cell.native_dma_mode,alias)) {
            ++cache_->stats_.pipeline_native_dma_batches;
            cache_->stats_.pipeline_native_dma_copies+=copies.size();
        } else for(const auto& copy:copies)MFQ_CUDA_CHECK(cudaMemcpyAsync(
            copy.destination,copy.source,copy.bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
        MFQ_CUDA_CHECK(cudaEventRecord(cache_->transfer_ready_,cache_->weight_stream_));
        cache_->transfer_ready_recorded_=true;
        const bool first_upload=!cell.ssd_gate_up_submitted;
        if(down)cell.ssd_down_submitted=true;else cell.ssd_gate_up_submitted=true;
        mfq::cuda::check_stream_progress(cache_->weight_stream_);
        auto& stats=cache_->stats_;
        stats.ram_pcie_bytes+=bytes;stats.pipeline_direct_ram_bytes+=bytes-staged_bytes;
        stats.pipeline_staged_ram_bytes+=staged_bytes;
        (down?stats.pipeline_ssd_down_staged_bytes:stats.pipeline_ssd_gu_staged_bytes)+=staged_bytes;
        // fetch_registered counts these projections once when it builds the
        // phase packet. Keep its payload/copy accounting independent of which
        // subset was submitted early; native DMA tracks actual submissions.
        stats.pipeline_direct_ram_copies+=copies.size();
        if(down) {
            stats.pipeline_ssd_down_overlap_bytes+=bytes;
            if(during_gate_up_reads) {
                stats.pipeline_ssd_down_early_max_batch=std::max(stats.pipeline_ssd_down_early_max_batch,int64_t(ready.size()));
                stats.pipeline_ssd_down_early_max_bytes=std::max(stats.pipeline_ssd_down_early_max_bytes,int64_t(bytes));
                static std::atomic_flag recorded=ATOMIC_FLAG_INIT;
                if(!recorded.test_and_set(std::memory_order_relaxed))
                    std::fprintf(stderr,"moe_ssd_down_early_upload scope=first_upload enabled=1 copies=%zu bytes=%zu projections=%zu queue=idle_byte_budget budget_bytes=%zu\n",
                        copies.size(),bytes,ready.size(),MoeExpertCache::SsdExpertReads::early_down_budget);
            }
            if(!stats.pipeline_ssd_down_upload_batches++)std::fprintf(stderr,
                "moe_ssd_down_io_overlap scope=first_upload enabled=1 copies=%zu bytes=%zu\n",copies.size(),bytes);
        } else {
            stats.pipeline_ssd_gu_overlap_bytes+=bytes;
            ++stats.pipeline_ssd_gu_upload_batches;
        }
        if(!down && first_upload && ++stats.pipeline_ssd_gu_overlap_serves==1)
            std::fprintf(stderr,"moe_ssd_gu_io_overlap scope=first_upload enabled=1 copies=%zu bytes=%zu\n",copies.size(),bytes);
        if(!down && staged_bytes && stats.pipeline_ssd_gu_staged_bytes==int64_t(staged_bytes))
            std::fprintf(stderr,"moe_ssd_gu_pageable_overlap scope=first_upload enabled=1 staged_bytes=%zu extra_pinned_bytes=0\n",staged_bytes);
        publish_ssd_gate_up(cell);
    }
    bool publish_ssd_gate_up(Cell& cell) {
        const auto* compute=std::getenv("MFQ_MOE_SSD_PHASED_COMPUTE");
        if(cell.ssd_gate_up_ready || !cell.ssd_gate_up_descriptors || (compute && compute[0]=='0'))return false;
        auto& loan=*cache_->kv_expert_loan_;
        bool complete=true;
        for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer)
            for(int p=0;p<2;++p) {
                if(single_up() && p==1)continue;
                const auto& source=*sources_[p];
                const auto& sizes=source.cohorts_[source.expert_to_cohort_[group.expert]].bytes_per_expert;
                auto& view=cell.transfer_views.at(cell.transfer_index->h<int32_t>()[group.expert]*3+p);
                if(!source.quant_source_->gpu_resident(group.expert)) {
                    std::array<std::size_t,8> offsets{};std::size_t logical=0;
                    if(const auto* cached=loan.pin_existing_projection(source.id_,group.expert,sizes,offsets,logical)) {
                        for(std::size_t f=0;f<sizes.size();++f)if(sizes[f])view.fields[f]=static_cast<const uint8_t*>(cached)+offsets[f];
                    }else complete=false;
                }
                for(std::size_t f=0;f<sizes.size();++f)if(sizes[f] && !view.fields[f])complete=false;
            }
        if(!complete)return false;
        // The first image contains final GU pointers and immutable Down format
        // metadata. The independent second image gets Down's final pointers;
        // the graph snapshots it after GU arithmetic, at the Down-ready wait.
        const auto descriptor_bytes=upload_transfer_views(cell,nullptr,true,cell.ssd_gate_up_descriptors.get());
        auto* header=cell.kv_loan_table->h<uint64_t>();
        header[0]=~uint64_t(0);header[1]=header[2]=header[3]=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);_mm_sfence();
        queue_transfer_ready(cell,32);
        mfq::cuda::check_stream_progress(cache_->weight_stream_);
        // Down's CPU preparation and completed read batches do not consume GU
        // weights. Let them proceed while its DMA completes; the Down wait
        // loop below continues publishing GU as soon as its event is ready.
        poll_transfer_notifications(cell,1,false);
        cell.ssd_gate_up_ready=true;
        if(cell.transfer_notification) {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "moe_ssd_gu_prepare_overlap scope=first_ready enabled=1 producer_wait=0\n");
        }
        auto& stats=cache_->stats_;
        stats.pipeline_window_dma_bytes+=descriptor_bytes;++stats.pipeline_window_dma_copies;
        if(++stats.pipeline_ssd_gu_compute_serves==1)
            std::fprintf(stderr,"moe_ssd_gu_compute_overlap scope=first_ready enabled=1 tokens=%d descriptor_bytes=%lld\n",
                cell.tokens,static_cast<long long>(descriptor_bytes));
        return true;
    }
    void fetch(Cell& cell,MoeExpertCache::SsdExpertReads* ssd_reads=nullptr) {
        cell.ssd_gate_up_submitted=false;
        cell.ssd_down_submitted=false;
        cell.ssd_gate_up_ready=false;
        cell.ssd_gate_up_host_tail=cache_->pipeline_host_stage_.numel()&~int64_t(15);
        struct ClearPrefetchKeys {
            std::vector<mfq::MoeCacheKey>& keys;
            ~ClearPrefetchKeys(){keys.clear();}
        } clear_prefetch_keys{cache_->pipeline_prefetched_keys_};
        // The planner overlaps GPU cache validation. Allocation, retirement
        // and next_packet publication must wait until its field snapshot is
        // complete; no separate CUDA event or graph node is necessary.
        if(cell.cached_resident)mfq::cuda::wait_route_publication(cell.flags.h<uint32_t>()+4,stream_);
        // The preceding call can finish CPU work before its weight DMA. Its
        // pinned source must remain immutable until that DMA has completed.
        // In a cached-resident window, the current snapshot follows all prior
        // consumers on the same stream. Top-1/2 predictive prefetch can contain
        // unconsumed uploads and retains its explicit completion event.
        const bool previous_consumed=cell.cached_resident && cell.routes>2 && cache_->pipeline_prefetched_keys_.empty();
        if(cache_->transfer_ready_recorded_ && previous_consumed && !cache_->stats_.pipeline_reader_fence_reuses++)
            std::fprintf(stderr,"moe_reader_fence_reuse scope=first_completed_consumer enabled=1 routes=%d tokens=%d\n",cell.routes,cell.tokens);
        if(cache_->transfer_ready_recorded_ && !previous_consumed)MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->transfer_ready_));
        cell.transfer_notification_pending={};
        cache_->pipeline_dma_leases_.clear();
        // Route publication on the same compute stream proves the preceding
        // layer finished reading these slots. DMA completion alone is weaker.
        for(const auto& held:cache_->pipeline_transfer_slots_)
            if(held.book->slot_for(held.key)==held.slot) {
                if(held.aliased)held.book->discard(held.key,held.slot,held.generation);
                else held.book->clear_inflight(held.slot);
            }
        cache_->pipeline_transfer_slots_.clear();
        if(cache_->kv_expert_loan_)cache_->kv_expert_loan_->next_packet(
            {sources_[0]->id_,sources_[1]->id_,sources_[2]->id_},cell.host_ids.h<int32_t>(),cell.tokens,cell.routes,experts_);
        // Forecast hits need no disk read, but this packet's allocations can
        // evict them before the next layer. Retain existing fields for this
        // packet only; the following publication expires the protection.
        if(cache_->kv_expert_loan_ && !cache_->ssd_forecast_gpu_fields_.empty()) {
            const auto fields=cache_->kv_expert_loan_->protect_packet_fields(cache_->ssd_forecast_gpu_fields_,true);
            if(fields && !cache_->ssd_prediction_gpu_protected_fields_)std::fprintf(stderr,
                "moe_ssd_router_gpu_protect scope=first_packet enabled=1 fields=%zu lifetime=one_packet\n",fields);
            cache_->ssd_prediction_gpu_protected_fields_+=fields;
        }
        cache_->ssd_forecast_gpu_fields_.clear();
        if(cell.cached_resident) {
            auto& protected_gu=cell.transfer_scratch.protected_keys;protected_gu.clear();
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                const auto mask=cached_expert_fields_->h<mfq::cuda::MfeCachedExpert>()[group.expert].ready_mask;
                for(int p=0;p<2;++p)if((mask&(uint64_t(1)<<p)) && !sources_[p]->quant_source_->gpu_resident(group.expert))
                    protected_gu.push_back({sources_[p]->id_,group.expert,-1});
            }
            cache_->kv_expert_loan_->protect_packet_fields(protected_gu);
        }
        if(cell.cached_resident)for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuResident) {
            bool borrowed=false;
            for(int p=0;p<3;++p) {
                if((single_up() && p==1) || sources_[p]->quant_source_->gpu_resident(group.expert))continue;
                const auto& source=*sources_[p];const auto& cohort=source.cohorts_[source.expert_to_cohort_[group.expert]];
                std::array<std::size_t,8> offsets{};std::size_t bytes=0;
                if(!cache_->kv_expert_loan_->pin_existing_projection(source.id_,group.expert,cohort.bytes_per_expert,offsets,bytes))
                    throw std::logic_error("GPU-selected cached resident projection lost before protection");
                ++cache_->stats_.pipeline_kv_projection_hits;++cache_->stats_.pipeline_kv_projection_early_hits;
                cache_->stats_.pipeline_kv_gpu_resident_bytes+=bytes;
                for(auto size:cohort.bytes_per_expert)if(size<int64_t(mfq::cuda::KvExpertLoan::page))
                    cache_->stats_.pipeline_kv_projection_small_hit_bytes+=size;
                borrowed=true;
            }
            if(borrowed) {
                ++cache_->stats_.pipeline_kv_gpu_resident_experts;
                if(!cached_resident_experts_++)std::fprintf(stderr,
                    "moe_kv_gpu_resident_dispatch scope=first_complete_expert enabled=1 tokens=%d expert=%d compressed=1\n",cell.tokens,group.expert);
            }
        }
        if(!cell.two_stage)for(auto& item:cache_->pipeline_stages_)
            if(item.second->book)item.second->book=std::make_unique<mfq::MoeCacheSlotBook>(item.second->slots);
        // serve() has already observed this layer's route publication. In a
        // model window it follows the preceding layer's weight consumers on
        // the same compute stream, including the previous window/prefill.
        // The separate compute event adds no ordering at this boundary.
        // Standalone calls still need their original stream dependency.
        const auto* route_fence=std::getenv("MFQ_MOE_WINDOW_ROUTE_FENCE");
        // Cached dispatch observes the graph's published routes and snapshot
        // acknowledgement before fetching. Both follow the prior readers on
        // the compute stream, so the earlier compute event adds no dependency.
        // Explicit overrides retain the original standalone/legacy behavior.
        const bool route_fenced=cell.in_window &&
            (route_fence ? route_fence[0]!='0' : cell.cached_resident);
        const auto* empty_transfer=std::getenv("MFQ_MOE_EMPTY_TRANSFER");
        if(cell.fused_transfer_copy && cell.skip_cpu_wait && cell.transfer_views.empty() &&
                (!empty_transfer || empty_transfer[0]!='0')) {
            // The preceding DMA has completed above, and route publication
            // fences its readers. No weight or projection descriptor is needed
            // for this all-resident plan. Mark both snapshots so their scatter
            // kernels ignore the preceding layer's unchanged wire packet.
            for(int phase=0;phase<2;++phase) {
                auto* header=reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(cell.kv_loan_table->host)+
                    phase*cell.kv_loan_table->bytes/2);
                // Encode emptiness in the count already read by both waits.
                // A separate mapped flag adds a PCIe read to every layer.
                header[0]=~uint64_t(0);header[1]=header[2]=header[3]=0;
            }
            cache_->transfer_ready_recorded_=false;
            publish(cell.flags.h<uint32_t>()+32);publish(cell.flags.h<uint32_t>()+80);
            ++cache_->stats_.pipeline_empty_transfer_serves;
            return;
        }
        // Empty plans perform no transfer-stream work. Only establish this
        // dependency when the stream will actually reuse transfer storage.
        if(!cell.mapped_copy && cache_->compute_done_recorded_) {
            if(route_fenced) {
                if(!cell.route_fence_trace_reported) {
                    cell.route_fence_trace_reported=true;
                    std::fprintf(stderr,"moe_window_route_fence_dispatch scope=first_skipped_event enabled=1 tokens=%d cached=%d automatic=%d\n",
                        cell.tokens,int(cell.cached_resident),int(route_fence==nullptr));
                }
            } else MFQ_CUDA_CHECK(cudaStreamWaitEvent(cache_->weight_stream_,cache_->compute_done_,0));
        }
        if(cell.wire.defined() && (cache_->pipeline_ram_registered_bytes_ || cache_->ssd_experts())) {
            if(ssd_reads) {
                const auto wait=[&](bool gate_up,std::vector<std::size_t>& batch) {
                    const auto started=std::chrono::steady_clock::now();
                    bool ready=false;
                    if(gate_up)ready=ssd_reads->next_projection_batch(batch,false,{},true,[&] {
                        if(!cache_->transfer_ready_recorded_)return true;
                        const auto status=cudaEventQuery(cache_->transfer_ready_);
                        if(status==cudaErrorNotReady) {
                            if(std::chrono::steady_clock::now()-started>std::chrono::seconds(60))
                                throw std::runtime_error("MFQ early Down DMA stalled");
                            _mm_pause();return false;
                        }
                        MFQ_CUDA_CHECK(status);return true;
                    });
                    else ready=ssd_reads->next_projection_batch(batch,true,[&] {
                        poll_transfer_notifications(cell,1,false);
                        if(cell.transfer_notification_pending[0]) {
                            if(std::chrono::steady_clock::now()-started>std::chrono::seconds(60))
                                throw std::runtime_error("MFQ expert DMA notification stalled");
                            _mm_pause();
                        }
                        return cell.transfer_notification_pending[0];
                    });
                    const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-started).count();
                    cache_->stats_.pipeline_ssd_read_ns+=elapsed;
                    (gate_up?cache_->stats_.pipeline_ssd_gate_up_wait_ns:cache_->stats_.pipeline_ssd_down_wait_ns)+=elapsed;
                    return ready;
                };
                std::vector<std::size_t> batch;
                if(ssd_reads->publish_progress) {
                    while(wait(true,batch))upload_ssd_projections(cell,*ssd_reads,batch,true);
                    // No GU read is needed when both projections are already
                    // cached. Publish their descriptors before joining cold
                    // Down reads, just as after an actual GU upload.
                    if(publish_ssd_gate_up(cell) && !ssd_reads->gate_up_requests) {
                        static std::atomic_flag reported=ATOMIC_FLAG_INIT;
                        if(!reported.test_and_set(std::memory_order_relaxed))std::fprintf(stderr,
                            "moe_ssd_cached_gu_compute scope=first_ready enabled=1 down_requests=%zu\n",
                            ssd_reads->requests.size());
                    }
                    // GU retains priority; ready Down images also fill idle
                    // transfer time while GU disk reads are still pending.
                    while(wait(false,batch)) {
                        poll_transfer_notifications(cell,1,false);
                        upload_ssd_projections(cell,*ssd_reads,batch);
                    }
                }
                const auto joined=std::chrono::steady_clock::now();
                ssd_reads->wait();
                const auto join_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-joined).count();
                cache_->stats_.pipeline_ssd_read_ns+=join_ns;
                cache_->stats_.pipeline_ssd_down_wait_ns+=join_ns;
            }
            fetch_registered(cell,ssd_reads);return;
        }
        auto* packed=cache_->pipeline_host_stage_.data_ptr<uint8_t>();int64_t offset=cell.wire_header_bytes;
        struct Copy {void* dst;const void* src;int64_t bytes;std::size_t pitch=0,rows=1;};
        std::vector<Copy> copies;
        const auto direct_option=mfq::cuda::runtime_options::moe_prefill_direct_ram();
        const bool direct_prefill=cell.tokens>8 && !cell.two_stage && !cell.wire.defined() &&
            cache_->pipeline_ram_registered_bytes_>0 && (!direct_option || *direct_option!=0);
        const auto ram_begin=cache_->pipeline_ram_complement_
            ? reinterpret_cast<std::uintptr_t>(cache_->pipeline_ram_complement_->data_ptr()) : std::uintptr_t(0);
        const auto ram_end=ram_begin+cache_->pipeline_ram_registered_bytes_;
        const auto registered=[&](const void* pointer,std::size_t bytes) {
            const auto address=reinterpret_cast<std::uintptr_t>(pointer);
            return direct_prefill && address>=ram_begin && address<=ram_end && bytes<=ram_end-address;
        };
        for(int projection=0;projection<3;++projection) {
            auto& source=*sources_[projection];std::vector<int> slots(source.cohorts_.size());
            struct Staged {int slot,expert;MoeHostExpertCache::Lease lease;std::vector<tb::Tensor> fields;};
            if(single_up() && projection==1)continue;
            std::vector<std::vector<Staged>> staged(source.cohorts_.size());
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                const int e=group.expert,c=source.expert_to_cohort_[e];auto& cohort=source.cohorts_[c];
                const int slot=slots[c]++;
                auto* stage=cell.two_stage?nullptr:cell.binding[projection].stages[c];
                if(stage && slot>=stage->slots)throw std::logic_error("MFQ transfer cohort exceeds source geometry");
                const int hot_slot=cohort.arena->book->slot_for({source.id_,c,e});
                // Split experts use resident GPU fields directly and upload
                // only their missing RAM projections.
                if(hot_slot>=0)continue;
                auto lease=prepared_expert(ssd_reads,projection,e);
                auto fields=moe_cache_fields(lease->weights);
                staged[c].push_back({slot,e,std::move(lease),std::move(fields)});
            }
            for(std::size_t c=0;c<staged.size();++c) {
                auto& experts=staged[c];const auto& cohort=source.cohorts_[c];
                const auto* stage=cell.two_stage?nullptr:cell.binding[projection].stages[c];
                for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                    const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                    for(std::size_t first=0;first<experts.size();) {
                        if(direct_prefill && registered(experts[first].fields[f].data_ptr(),std::size_t(bytes))) {
                            const auto start=reinterpret_cast<std::uintptr_t>(experts[first].fields[f].data_ptr());
                            std::size_t last=first+1,pitch=0;
                            if(last<experts.size() && experts[last].slot==experts[first].slot+1) {
                                const auto next=reinterpret_cast<std::uintptr_t>(experts[last].fields[f].data_ptr());
                                if(next>start && next-start>=std::size_t(bytes) && next-start<=std::size_t(INT_MAX))
                                    pitch=next-start;
                            }
                            if(pitch)while(last<experts.size() && experts[last].slot==experts[last-1].slot+1 &&
                                    registered(experts[last].fields[f].data_ptr(),std::size_t(bytes)) &&
                                    reinterpret_cast<std::uintptr_t>(experts[last].fields[f].data_ptr())==start+(last-first)*pitch)++last;
                            const auto rows=last-first,total=std::size_t(bytes)*rows;
                            auto* destination=static_cast<uint8_t*>(stage->fields[f].data_ptr())+int64_t(experts[first].slot)*bytes;
                            if(rows>1 && pitch==std::size_t(bytes))copies.push_back({destination,experts[first].fields[f].data_ptr(),int64_t(total)});
                            else copies.push_back({destination,experts[first].fields[f].data_ptr(),bytes,pitch,rows});
                            cache_->stats_.ram_pcie_bytes+=total;
                            cache_->stats_.pipeline_direct_ram_bytes+=total;
                            ++cache_->stats_.pipeline_direct_ram_copies;
                            first=last;continue;
                        }
                        auto last=first+1;
                        while(last<experts.size() && experts[last].slot==experts[last-1].slot+1)++last;
                        const auto total=bytes*static_cast<int64_t>(last-first);
                        offset=(offset+15)&~int64_t(15);
                        if(total>static_cast<int64_t>(cache_->pipeline_host_stage_.numel())-offset)
                            throw std::logic_error("MFQ pinned stage geometry exceeded");
                        for(auto i=first;i<last;++i)std::memcpy(packed+offset+bytes*(i-first),
                            experts[i].fields[f].data_ptr(),static_cast<std::size_t>(bytes));
                        if(cell.two_stage)for(auto i=first;i<last;++i) {
                            const int index=cell.transfer_index->h<int32_t>()[experts[i].expert];
                            cell.transfer_views[index*3+projection].fields[f]=cell.wire.data_ptr<uint8_t>()+offset+bytes*(i-first);
                        }
                        auto* dst=stage?static_cast<uint8_t*>(stage->fields[f].data_ptr())+int64_t(experts[first].slot)*bytes:nullptr;
                        copies.push_back({dst,packed+offset,total});offset+=total;
                        if(direct_prefill)cache_->stats_.pipeline_staged_ram_bytes+=total;
                        cache_->stats_.ram_pcie_bytes+=total;first=last;
                    }
                }
                if(direct_prefill)for(auto& expert:experts)
                    cache_->pipeline_dma_leases_.push_back(std::move(expert.lease));
            }
        }
        const auto descriptor_bytes=upload_transfer_views(cell);
        const auto submissions=(cell.wire.defined() ? 1 : static_cast<int64_t>(copies.size()))+(descriptor_bytes>0);
        cache_->stats_.pipeline_dma_copies+=submissions;
        if(cell.in_window) {
            cache_->stats_.pipeline_window_dma_copies+=submissions;
            if(cell.wire.defined())cache_->stats_.pipeline_window_dma_bytes+=offset+descriptor_bytes;
            else for(const auto& copy:copies)cache_->stats_.pipeline_window_dma_bytes+=copy.bytes;
        }
        // The GPU's plan becomes visible before DMA submission and CPU jobs.
        if(cell.wire.defined()) {
            if(offset>cell.wire.numel() || 32+copies.size()*24>static_cast<std::size_t>(cell.wire_header_bytes))
                throw std::logic_error("MFQ packed DMA exceeded its real route geometry");
            auto* words=reinterpret_cast<uint64_t*>(packed);
            words[0]=copies.size();words[1]=cell.wire_header_bytes;words[2]=offset;words[3]=0;
            for(std::size_t i=0;i<copies.size();++i) {
                words[4+i*3]=reinterpret_cast<uint64_t>(copies[i].dst);
                words[5+i*3]=static_cast<const uint8_t*>(copies[i].src)-packed;
                words[6+i*3]=copies[i].bytes;
            }
            copy_transfer(cell,cell.wire.data_ptr(),packed,static_cast<std::size_t>(offset));
        } else for(const auto& copy:copies)copy_transfer(cell,copy.dst,copy.src,static_cast<std::size_t>(copy.bytes),copy.pitch,copy.rows);
        finish_transfer(cell,!copies.empty() || cell.wire.defined());
    }
    void cpu(Cell& cell,const MoeExpertCache::SsdExpertReads* reads=nullptr) {
        using Clock=std::chrono::steady_clock;
        const auto ns=[](Clock::time_point begin,Clock::time_point end) {
            return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        };
        struct ProjectionTiming {
            std::atomic<std::uint64_t> work_ns{0},callbacks{0},rows{0},activation_ns{0};
        };
        MoeExpertCache::CpuCallSample sample;
        std::unique_ptr<ProjectionTiming[]> timing;
        auto started=cache_->pipeline_cpu_profile_enabled_ ? Clock::now() : Clock::time_point{};
        if(cache_->pipeline_cpu_profile_enabled_)++cell.cpu_profile_calls;
        cell.jobs.clear();int offset=0;
        for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::Cpu) {
            CpuJob job;job.expert=group.expert;job.offset=offset;job.count=static_cast<int>(group.positions.size());
            for(int i=0;i<3;++i){job.leases[i]=combined_gate_up_ && i==1?job.leases[0]:prepared_expert(reads,i,group.expert);job.rows[i]=rows_plan(job.leases[i]->weights,combined_gate_up_ && i==1);}
            offset+=job.count;cell.jobs.push_back(std::move(job));
        }
        cache_->record_hybrid(static_cast<int>(cell.jobs.size()),static_cast<int>(cell.plan.groups.size()-cell.jobs.size()));
        for(const auto& source:sources_)source->quant_source_->record_cpu_projections(cell.jobs.size());
        for(const auto& group:cell.plan.groups)cache_->stats_.ram_pcie_experts+=group.kind==mfq::MoeDispatchKind::GpuTransfer;
        if(cell.jobs.empty()){publish(cell.flags.h<uint32_t>()+48);return;}
        cell.cpu_input.resize(std::size_t(offset)*width_);cell.g.resize(std::size_t(offset)*ff_);
        cell.u.resize(std::size_t(offset)*ff_);cell.hidden.resize(std::size_t(offset)*ff_);cell.y.resize(std::size_t(offset)*width_);
        for(const auto& job:cell.jobs) {
            const auto& positions=std::find_if(cell.plan.groups.begin(),cell.plan.groups.end(),[&](const auto& g){return g.expert==job.expert;})->positions;
            for(int row=0;row<job.count;++row)std::memcpy(cell.cpu_input.data()+std::size_t(job.offset+row)*width_,
                cell.x.h<float>()+(positions[row]/cell.routes)*width_,width_*sizeof(float));
        }
        if(cache_->pipeline_cpu_profile_enabled_) {
            // Only auxiliary storage failure disables tracing. Lease, plan,
            // projection and result failures still propagate to the caller.
            try {
                if(cache_->pipeline_cpu_profile_fail_alloc_)throw std::bad_alloc();
                timing=std::make_unique<ProjectionTiming[]>(cell.jobs.size()*3);
                sample.projections.resize(cell.jobs.size()*3);
            } catch(const std::bad_alloc&) {
                timing.reset();cache_->pipeline_cpu_profile_enabled_=false;
                cache_->pipeline_cpu_profile_dropped_=true;
            }
        }
        const auto prepared=timing ? Clock::now() : Clock::time_point{};
        const auto projection=[&](std::size_t index,int p,const float* input,int stride,
                float* output,int out,int first,int last) {
            const auto& job=cell.jobs[index];
            if(!timing) { job.rows[p].run(input,job.count,stride,output,out,first,last);return; }
            const auto before=Clock::now();
            job.rows[p].run(input,job.count,stride,output,out,first,last);
            auto& t=timing[index*3+p];
            t.work_ns.fetch_add(ns(before,Clock::now()),std::memory_order_relaxed);
            t.callbacks.fetch_add(1,std::memory_order_relaxed);
            t.rows.fetch_add(last-first,std::memory_order_relaxed);
        };
        mfq::cpu::expert_pool().rows(int64_t(cell.jobs.size())*ff_,[&](int64_t begin,int64_t end) {
            while(begin<end) {
                const auto index=static_cast<std::size_t>(begin/ff_);
                const auto& job=cell.jobs[index];const int first=static_cast<int>(begin%ff_);
                const int last=static_cast<int>(std::min<int64_t>(ff_,first+end-begin));
                const auto* input=cell.cpu_input.data()+std::size_t(job.offset)*width_;
                projection(index,0,input,width_,cell.g.data()+std::size_t(job.offset)*ff_,ff_,first,last);
                projection(index,1,input,width_,cell.u.data()+std::size_t(job.offset)*ff_,ff_,first,last);
                const auto activation_started=timing ? Clock::now() : Clock::time_point{};
                // Every row chunk owns both projections and its hidden values.
                // Keep each original F16 operation before the Down dispatch.
                for(int row=0;row<job.count;++row)for(int col=first;col<last;++col) {
                    const auto i=std::size_t(job.offset+row)*ff_+col;
                    const mfq_half g(cell.g[i]),u(cell.u[i]);
                    if(up_only_) {
                        const float positive=std::max(0.0f,float(u));
                        cell.hidden[i]=float(mfq_half(positive*positive));continue;
                    }
                    if(swiglu_limit_>0) {
                        const auto clipped=std::min(float(g),float(swiglu_limit_));
                        const auto value=std::clamp(float(u),-float(swiglu_limit_),float(swiglu_limit_));
                        cell.hidden[i]=float(mfq_half((clipped/(1.0f+std::exp(-clipped)))*value));continue;
                    }
                    const mfq_half sigmoid(1.0/(1.0+std::exp(-double(float(g)))));
                    const mfq_half first(float(g)*float(sigmoid));
                    cell.hidden[i]=float(mfq_half(float(first)*float(u)));
                }
                if(timing)timing[index*3].activation_ns.fetch_add(
                    ns(activation_started,Clock::now()),std::memory_order_relaxed);
                begin+=last-first;
            }
        },timing ? &sample.gate_up : nullptr);
        mfq::cpu::expert_pool().rows(int64_t(cell.jobs.size())*width_,[&](int64_t begin,int64_t end) {
            while(begin<end) {
                const auto index=static_cast<std::size_t>(begin/width_);
                const auto& job=cell.jobs[index];const int first=static_cast<int>(begin%width_);
                const int last=static_cast<int>(std::min<int64_t>(width_,first+end-begin));
                projection(index,2,cell.hidden.data()+std::size_t(job.offset)*ff_,ff_,
                    cell.y.data()+std::size_t(job.offset)*width_,width_,first,last);begin+=last-first;
            }
        },timing ? &sample.down : nullptr);
        const auto scatter_started=timing ? Clock::now() : Clock::time_point{};
        for(const auto& job:cell.jobs) {
            const auto& positions=std::find_if(cell.plan.groups.begin(),cell.plan.groups.end(),[&](const auto& g){return g.expert==job.expert;})->positions;
            for(int row=0;row<job.count;++row)for(int col=0;col<width_;++col)
                cell.cpu.h<mfq_half>()[positions[row]*width_+col]=mfq_half(cell.y[std::size_t(job.offset+row)*width_+col]);
        }
        publish(cell.flags.h<uint32_t>()+48);
        if(timing) {
            const auto finished=Clock::now();
            sample.serve=cache_->stats_.pipeline_serves;sample.cell_call=cell.cpu_profile_calls;
            sample.layer=sources_[0]->layer_id_;sample.tokens=cell.tokens;sample.window=cell.in_window;
            sample.experts=static_cast<int>(cell.jobs.size());sample.positions=offset;
            sample.prepare_ns=ns(started,prepared);sample.activation_inside_gate_up=true;
            sample.scatter_ns=ns(scatter_started,finished);sample.total_ns=ns(started,finished);
            for(std::size_t i=0;i<cell.jobs.size();++i)for(int p=0;p<3;++p) {
                const auto& job=cell.jobs[i];const auto& w=job.leases[p]->weights;
                const auto& t=timing[i*3+p];auto& r=sample.projections[i*3+p];
                sample.activation_work_ns+=t.activation_ns.load();
                r.expert=job.expert;r.projection=p;r.family=static_cast<int>(w.family);
                r.format=static_cast<int>(w.family==MixedMoeFamily::Nvq ? w.nvq.kernel_format : w.nint.format_version);
                r.bits=static_cast<int>(w.family==MixedMoeFamily::Nvq ? w.nvq.sub_bits : w.nint.bits);
                r.group_size=static_cast<int>(w.family==MixedMoeFamily::Nvq ? w.nvq.gs : w.nint.gs);
                r.positions=job.count;r.width=job.rows[p].width;r.outputs=job.rows[p].outputs;
                r.work_ns=t.work_ns.load();r.callbacks=t.callbacks.load();r.rows=t.rows.load();
            }
            try {cache_->pipeline_cpu_profiles_.push_back(std::move(sample));}
            catch(const std::bad_alloc&) {
                cache_->pipeline_cpu_profile_enabled_=false;cache_->pipeline_cpu_profile_dropped_=true;
            }
        }
        cell.jobs.clear();
    }
    void trace_serve(const Cell& cell,const char* phase,bool force=false) const noexcept {
        const auto serves=cache_->stats_.pipeline_window_serves;
        if(!force && (!trace_serve_ || (cell.in_window && serves>=96 && serves%(48*64)>=48)))return;
        const auto* flags=static_cast<const volatile uint32_t*>(cell.flags.h<uint32_t>());
        std::fprintf(stderr,"moe_serve layer=%d tokens=%d window=%d serve=%llu window_serve=%llu phase=%s flags=%u,%u,%u,%u,%u,%u\n",
            sources_[0]->layer_id_,cell.tokens,int(cell.in_window),
            static_cast<unsigned long long>(cache_->stats_.pipeline_serves),
            static_cast<unsigned long long>(serves),phase,
            flags[0],flags[16],flags[32],flags[48],flags[64],flags[80]);
        std::fflush(stderr);
    }
    void serve(Cell& cell) {
        // Padding publishes invalid routes. A wholly inactive tile only opens
        // the graph gates; it does not plan, fetch, observe or count experts.
        const auto wait_started = std::chrono::steady_clock::now();
        wait_publication(cell);
        const auto waited_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count();
        cell.active_tokens = 0;
        for (int row = 0; row < cell.tokens; ++row)
            cell.active_tokens += cell.host_ids.h<int32_t>()[row * cell.routes] >= 0;
        if (cell.allow_padding && !cell.active_tokens) {
            cell.plan.groups.clear();
            cell.plan.kinds.assign(cell.entries, mfq::MoeDispatchKind::Inactive);
            std::memset(cell.kind.host, 0, cell.kind.bytes);
            if (cell.transfer_route_plan)
                std::memset(cell.transfer_route_plan->host, 0xff, cell.transfer_route_plan->bytes);
            for (int flag : {16, 32, 48, 80})
                publish(cell.flags.h<uint32_t>() + flag);
            return;
        }
        auto& stats=cache_->stats_;++stats.pipeline_serves;
        stats.pipeline_fused_transfer_copy_serves+=cell.fused_transfer_copy;
        stats.pipeline_two_stage_serves+=cell.two_stage;
        stats.pipeline_window_input_view_serves+=cell.window_input_views;
        stats.pipeline_phased_transfer_serves+=cell.phased_transfer;
        const auto route_started=stats.pipeline_route_wait_ns,plan_started=stats.pipeline_plan_ns;
        stats.pipeline_route_wait_ns += waited_ns;
        const auto fetch_before=stats.pipeline_fetch_ns,cpu_before=stats.pipeline_cpu_ns;
        const auto time=[](int64_t& total,const auto& fn) {
            struct Sample {
                int64_t& total;
                std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
                ~Sample(){total+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-started).count();}
            } sample{total};
            fn();
        };
        trace_serve(cell,"route_wait");
        time(stats.pipeline_route_wait_ns,[&]{wait_publication(cell);});
        // The forecast export precedes this route publication on the compute
        // stream. Consume it here, without an earlier standalone WDDM flush.
        auto forecast=std::move(cache_->ssd_pending_forecast_);
        cache_->ssd_pending_forecast_={};
        // Keep exported tensor owners until every host-ready flag is set.
        if(forecast)forecast();
        trace_serve(cell,"plan");
        time(stats.pipeline_plan_ns,[&]{plan(cell);});
        if(cell.router_hint && cell.tokens==1) {
            const auto& hint=*cell.router_hint;
            for(int i=0;i<hint.routes;++i) {
                const int predicted=hint.flags.h<int32_t>()[16+i];
                ++cache_->ssd_predicted_routes_;
                if(std::any_of(cell.plan.groups.begin(),cell.plan.groups.end(),
                        [predicted](const auto& group){return group.expert==predicted;}))++cache_->ssd_prediction_hits_;
            }
        }
        trace_serve(cell,"fetch");
        // The demand plan already deduplicates the window's token routes.
        // Parallel file reads share one immutable lease with CPU work and
        // Gate/Up/Down uploads, including when the LRU budget is exhausted.
        std::unique_ptr<MoeExpertCache::SsdExpertReads> ssd_reads;
        const auto ssd_read_before=stats.pipeline_ssd_read_ns;
        if(cache_->ssd_experts()) {
            std::vector<MoeExpertCache::SsdExpertRequest> requests;
            std::size_t gate_up_requests=0;
            for(int p=0;p<3;++p) {
                if(p==2)gate_up_requests=requests.size();
                for(const auto& group:cell.plan.groups)if(group.kind!=mfq::MoeDispatchKind::GpuResident) {
                    if((single_up() && p==1) || sources_[p]->quant_source_->gpu_resident(group.expert))continue;
                    if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                        if(cell.cached_resident && (cached_expert_fields_->h<mfq::cuda::MfeCachedExpert>()[group.expert].ready_mask&(uint64_t(1)<<p)))continue;
                        if(cell.transfer_cache && !cell.wire_fallback) {
                            const auto& source=*sources_[p];const int c=source.expert_to_cohort_[group.expert];
                            const auto* stage=cell.binding[p].stages[c];
                            if(stage && stage->book && stage->book->slot_for({source.id_,c,group.expert})>=0)continue;
                        }
                    }
                    requests.emplace_back(sources_[p]->quant_source_.get(),group.expert);
                }
            }
            if(cell.batch_decode)cache_->observe_ssd_prediction_demand(requests,
                {sources_[0]->quant_source_.get(),sources_[1]->quant_source_.get(),sources_[2]->quant_source_.get()});
            const auto* overlap=std::getenv("MFQ_MOE_SSD_GU_IO_OVERLAP");
            const auto* projection=std::getenv("MFQ_MOE_KV_PROJECTION_CACHE");
            const auto* lookup=std::getenv("MFQ_MOE_KV_LOOKUP_FIRST");
            const bool phased_reads=cell.direct_packet && cache_->kv_expert_loan_ &&
                gate_up_requests<requests.size() &&
                (!overlap || overlap[0]!='0') && (!projection || projection[0]=='1') && (!lookup || lookup[0]=='1') &&
                std::none_of(cell.plan.groups.begin(),cell.plan.groups.end(),[](const auto& group){return group.kind==mfq::MoeDispatchKind::Cpu;});
            time(stats.pipeline_ssd_read_ns,[&] {
                if(phased_reads)ssd_reads=cache_->begin_ssd_expert_reads(std::move(requests),gate_up_requests);
                else {
                    ssd_reads=cache_->prepare_ssd_expert_reads(std::move(requests));
                    cache_->execute_ssd_expert_reads(*ssd_reads);
                }
            });
        }
        // Forecasts publish IDs before this layer runs. Queue the
        // demanded reads first and hold their RAM leases before speculative
        // readers can claim I/O bandwidth or evict an unleased current hit.
        if(!cache_->ssd_pending_prediction_.empty()) {
            auto predicted=std::move(cache_->ssd_pending_prediction_);
            cache_->ssd_pending_prediction_.clear();
            const auto previous_jobs=cache_->ssd_prediction_jobs_;
            cache_->queue_ssd_prediction(std::move(predicted));
            if(!previous_jobs && cache_->ssd_prediction_jobs_)std::fprintf(stderr,
                "moe_ssd_forecast_demand_order scope=first_batch enabled=1 current_requests=%zu\n",
                ssd_reads?ssd_reads->requests.size():0);
        }
        const auto transferred=stats.ram_pcie_bytes,fetch_started=stats.pipeline_fetch_ns;
        if(cache_->pipeline_dma_profile_enabled_) {
            if(cell.dma_timing) {
                auto& timing=*cell.dma_timing;timing.started=false;timing.sample={};
                auto& sample=timing.sample;
                sample.layer=sources_[0]->layer_id_;sample.tokens=cell.tokens;sample.entries=cell.entries;
                sample.window=cell.in_window;sample.serve=stats.pipeline_serves;
                sample.route_wait_ns=stats.pipeline_route_wait_ns-route_started;
                sample.plan_ns=stats.pipeline_plan_ns-plan_started;
            } else ++cache_->pipeline_dma_profile_skipped_;
        }
        time(stats.pipeline_fetch_ns,[&] {
            if(cache_->pipeline_dma_profile_enabled_ && cell.dma_timing)
                cell.dma_timing->fetch_started=std::chrono::steady_clock::now();
            const auto read_before=stats.pipeline_ssd_read_ns;
            fetch(cell,ssd_reads.get());
            // Fetch includes CPU preparation, while file waits keep their
            // original separate counter even when reads run asynchronously.
            stats.pipeline_fetch_ns-=stats.pipeline_ssd_read_ns-read_before;
        });
        if(cell.in_window)stats.pipeline_window_ssd_read_ns+=stats.pipeline_ssd_read_ns-ssd_read_before;
        trace_serve(cell,"cpu");
        if(cache_->pipeline_dma_profile_enabled_ && cell.dma_timing) {
            auto& sample=cell.dma_timing->sample;
            sample.payload_bytes=stats.ram_pcie_bytes-transferred;
            sample.fetch_ns=stats.pipeline_fetch_ns-fetch_started;
        }
        if(stats.ram_pcie_bytes>transferred) {
            stage_bytes_+=stats.ram_pcie_bytes-transferred;stage_ns_+=stats.pipeline_fetch_ns-fetch_started;
        }
        std::size_t work=0;for(const auto& group:cell.plan.groups)
            if(group.kind==mfq::MoeDispatchKind::Cpu)work+=group.positions.size();
        const auto cpu_started=stats.pipeline_cpu_ns;
        if(cell.calibration_expert>=0) {
            const auto prepared=std::chrono::steady_clock::now();
            const auto found=std::find_if(cell.plan.groups.begin(),cell.plan.groups.end(),
                [&](const auto& group){return group.expert==cell.calibration_expert;});
            if(found==cell.plan.groups.end() || found->kind==mfq::MoeDispatchKind::Cpu)
                throw std::logic_error("background CPU calibration changed production dispatch");
            std::array<MoeHostExpertCache::Lease,3> leases;
            for(int p=0;p<3;++p)leases[p]=prepared_expert(ssd_reads.get(),p,found->expert);
            std::vector<float> input(found->positions.size()*width_);
            for(size_t row=0;row<found->positions.size();++row)
                std::memcpy(input.data()+row*width_,cell.x.h<float>()+
                    (found->positions[row]/cell.routes)*width_,width_*sizeof(float));
            const auto preparation_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-prepared).count();
            auto work=cpu_calibration_work(cpu_cost_key(*found),std::move(leases),std::move(input),
                width_,ff_,int(found->positions.size()),double(preparation_ns),combined_gate_up_,swiglu_limit_,up_only_);
            cache_->pipeline_cpu_calibration_.start(std::move(work));
        }
        time(stats.pipeline_cpu_ns,[&]{cpu(cell,ssd_reads.get());});
        trace_serve(cell,"complete");
        if(work){
            const auto elapsed=stats.pipeline_cpu_ns-cpu_started;
            cpu_ns_+=elapsed;cpu_work_+=work;
            if(cache_->pipeline_shared_cpu_cost_) {
                mfq::MoeCpuCostModel::Key key{};std::size_t groups=0;bool homogeneous=true;
                for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::Cpu) {
                    const auto candidate=cpu_cost_key(group);
                    if(!groups)key=candidate;else homogeneous&=candidate==key;
                    ++groups;
                }
                // Shared pool walltime cannot identify independent rates for
                // heterogeneous jobs; retain only unambiguous observations.
                if(homogeneous)cache_->pipeline_cpu_cost_.observe(key,double(elapsed),groups);
            }
        }
        if(cell.in_window) {
            ++stats.pipeline_window_serves;
            stats.pipeline_window_pcie_payload_bytes+=stats.ram_pcie_bytes-transferred;
            stats.pipeline_window_route_ns+=stats.pipeline_route_wait_ns-route_started;
            stats.pipeline_window_plan_ns+=stats.pipeline_plan_ns-plan_started;
            stats.pipeline_window_fetch_ns+=stats.pipeline_fetch_ns-fetch_before;
            stats.pipeline_window_cpu_ns+=stats.pipeline_cpu_ns-cpu_before;
        }
        if(cache_->ssd_experts() && cell.batch_decode && sources_[0]->layer_id_>=0)
            cache_->observe_ssd_read_wait(stats.pipeline_ssd_read_ns-ssd_read_before);
    }
    void observe_gpu(Cell& cell) {
        // Window finish runs after the complete GPU graph. Join at most its
        // final background sample before any RAM lease can be exchanged.
        if(cell.in_window)cache_->pipeline_cpu_calibration_.collect(cache_->pipeline_cpu_cost_,true);
        if(cell.skip_gpu_timing || !cell.active_tokens)return;
        float hot=0,transfer=0;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&hot,cell.timing.points[0],cell.timing.points[1]));
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&transfer,cell.timing.points[2],cell.timing.points[3]));
        const bool pure_gpu=!cell.two_stage || std::all_of(cell.plan.groups.begin(),cell.plan.groups.end(),
            [](const auto& group){return group.kind!=mfq::MoeDispatchKind::Cpu;});
        if(cell.batch_decode && pure_gpu){gpu_ns_+=(hot+transfer)*1e6;++gpu_samples_;}
        cache_->stats_.pipeline_gpu_ns+=static_cast<int64_t>((hot+transfer)*1e6);
        if(cell.in_window)cache_->stats_.pipeline_window_gpu_ns+=static_cast<int64_t>((hot+transfer)*1e6);
        if(cache_->pipeline_dma_profile_enabled_ && cell.dma_timing && cell.dma_timing->started) {
            auto& timing=*cell.dma_timing;auto& p=timing.sample;
            // The third event follows the stream flag write or callback fallback,
            // so it bounds notification completion.
            // Stream intervals include enqueue/queue gaps, not just copy-engine
            // execution. Synchronization and all extra events are opt-in only.
            MFQ_CUDA_CHECK(cudaEventSynchronize(timing.points[2]));
            const auto elapsed=[](cudaEvent_t begin,cudaEvent_t end) {
                float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
                return static_cast<std::int64_t>(std::llround(double(ms)*1e6));
            };
            p.copy_ns=elapsed(timing.points[0],timing.points[1]);
            p.notice_ns=elapsed(timing.points[1],timing.points[2]);
            p.hot_ns=static_cast<std::int64_t>(std::llround(double(hot)*1e6));
            p.copy_begin_after_hot_ns=elapsed(cell.timing.points[0],timing.points[0]);
            p.copy_end_after_hot_ns=elapsed(cell.timing.points[0],timing.points[1]);
            p.notice_end_after_hot_ns=elapsed(cell.timing.points[0],timing.points[2]);
            p.scatter_end_after_hot_ns=elapsed(cell.timing.points[0],cell.timing.points[2]);
            try {cache_->pipeline_dma_profiles_.push_back(p);}
            catch(const std::bad_alloc&) {
                cache_->pipeline_dma_profile_enabled_=false;cache_->pipeline_dma_profile_dropped_=true;
            }
            timing.started=false;
        }
    }
public:
    void prefetch_host(const std::vector<int32_t>& predictions,Mapped& flags) {
        struct Field {void* destination;tb::Tensor source;int64_t bytes,offset=0;};
        std::vector<Field> fields;
        const auto lease_begin=cache_->pipeline_dma_leases_.size();
        const auto slot_begin=cache_->pipeline_transfer_slots_.size();
        const auto key_begin=cache_->pipeline_prefetched_keys_.size();
        fields.reserve(predictions.size()*3*7);
        cache_->pipeline_dma_leases_.reserve(lease_begin+predictions.size()*3);
        cache_->pipeline_transfer_slots_.reserve(slot_begin+predictions.size()*3);
        cache_->pipeline_prefetched_keys_.reserve(key_begin+predictions.size()*3);
        struct Rollback {
            MoeExpertCache& cache;std::size_t leases,slots,keys;bool committed=false;
            ~Rollback() {
                if(committed)return;
                (void)cudaStreamSynchronize(cache.weight_stream_);
                for(auto i=slots;i<cache.pipeline_transfer_slots_.size();++i) {
                    const auto& held=cache.pipeline_transfer_slots_[i];
                    held.book->discard(held.key,held.slot,held.generation);
                }
                cache.pipeline_transfer_slots_.resize(slots);
                cache.pipeline_dma_leases_.resize(leases);
                cache.pipeline_prefetched_keys_.resize(keys);
            }
        } rollback{*cache_,lease_begin,slot_begin,key_begin};
        const int64_t capacity=cache_->pipeline_prefetch_host_stage_.numel();
        const int64_t header_bytes=(32+predictions.size()*3*7*24+15)&~int64_t(15);
        int64_t estimated=header_bytes;
        std::vector<int32_t> seen;
        for(const int e:predictions) {
            if(e<0 || e>=experts_)throw std::runtime_error("MFQ prefetch expert outside router geometry");
            if(std::find(seen.begin(),seen.end(),e)!=seen.end())continue;
            seen.push_back(e);bool copied=false;
            for(int p=0;p<3;++p) {
                if(single_up() && p==1)continue;
                auto& source=*sources_[p];const int c=source.expert_to_cohort_[e];
                auto& cohort=source.cohorts_[c];if(source.quant_source_->gpu_resident(e))continue;
                auto& arena=stage(p,cohort,moe_decode_transfer_slots(8,10),true);auto* book=arena.book.get();const mfq::MoeCacheKey key{source.id_,c,e};
                if(book->slot_for(key)>=0)continue;
                bool available=false;for(int slot=0;slot<book->capacity();++slot)available|=!book->inflight(slot);
                if(!available){++cache_->stats_.pipeline_prefetch_busy_skips;continue;}
                const auto size=static_cast<int64_t>(source.bytes_for_expert(e));
                if(size+int64_t(cohort.bytes_per_expert.size())*15>capacity-estimated)continue;
                auto lease=source.quant_source_->acquire_expert(e);auto values=moe_cache_fields(lease->weights);
                if(values.size()!=cohort.bytes_per_expert.size())throw std::logic_error("prefetch RAM field count changed");
                const auto held=book->acquire(key);
                cache_->pipeline_transfer_slots_.push_back({book,key,held.slot,held.generation});book->mark_inflight(held.slot);
                cache_->pipeline_prefetched_keys_.push_back(key);
                for(std::size_t f=0;f<values.size();++f) {
                    const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                    if(tensor_nbytes(values[f])!=bytes)throw std::logic_error("prefetch RAM field geometry changed");
                    fields.push_back({arena.field_pointer(held.slot,f,cohort.bytes_per_expert),std::move(values[f]),bytes});
                }
                estimated+=size+int64_t(values.size())*15;
                cache_->pipeline_dma_leases_.push_back(std::move(lease));copied=true;
            }
            cache_->stats_.pipeline_prefetch_experts+=copied;
        }
        if(fields.empty()) {
            publish(flags.h<uint32_t>()+8);publish(flags.h<uint32_t>()+4);
            rollback.committed=true;return;
        }
        auto* host=cache_->pipeline_prefetch_host_stage_.data_ptr<uint8_t>();
        auto* wire=cache_->pipeline_prefetch_wire_gpu_.data_ptr<uint8_t>();
        const auto ram_begin=cache_->pipeline_ram_complement_?reinterpret_cast<uintptr_t>(cache_->pipeline_ram_complement_->data_ptr()):uintptr_t(0);
        const auto ram_bytes=static_cast<uint64_t>(cache_->pipeline_ram_registered_bytes_);
        std::vector<std::size_t> direct,staged;
        for(std::size_t i=0;i<fields.size();++i) {
            const auto address=reinterpret_cast<uintptr_t>(fields[i].source.data_ptr());
            (address>=ram_begin && address-ram_begin<=ram_bytes && uint64_t(fields[i].bytes)<=ram_bytes-(address-ram_begin)?direct:staged).push_back(i);
        }
        std::sort(direct.begin(),direct.end(),[&](auto a,auto b){return reinterpret_cast<uintptr_t>(fields[a].source.data_ptr())<reinterpret_cast<uintptr_t>(fields[b].source.data_ptr());});
        struct Interval {const void* source;int64_t bytes,offset;};std::vector<Interval> intervals;
        int64_t offset=header_bytes,direct_bytes=0,staged_bytes=0;
        for(std::size_t first=0;first<direct.size();) {
            const auto begin=reinterpret_cast<uintptr_t>(fields[direct[first]].source.data_ptr());
            auto end=begin+fields[direct[first]].bytes;auto last=first+1;
            while(last<direct.size()) {
                const auto next=reinterpret_cast<uintptr_t>(fields[direct[last]].source.data_ptr());
                if(next<end || next-end>15 || ((next-begin)&15))break;
                end=next+fields[direct[last]].bytes;++last;
            }
            offset=(offset+15)&~int64_t(15);const auto count=static_cast<int64_t>(end-begin);
            if(count>capacity-offset)throw std::logic_error("prefetch wire interval capacity exceeded");
            for(auto i=first;i<last;++i) {
                auto& field=fields[direct[i]];field.offset=offset+static_cast<int64_t>(reinterpret_cast<uintptr_t>(field.source.data_ptr())-begin);
                direct_bytes+=field.bytes;
            }
            intervals.push_back({reinterpret_cast<const void*>(begin),count,offset});offset+=count;first=last;
        }
        const auto staged_begin=(offset+15)&~int64_t(15);offset=staged_begin;
        for(const auto i:staged) {
            offset=(offset+15)&~int64_t(15);auto& field=fields[i];
            if(field.bytes>capacity-offset)throw std::logic_error("prefetch staged field capacity exceeded");
            field.offset=offset;std::memcpy(host+offset,field.source.data_ptr(),field.bytes);offset+=field.bytes;staged_bytes+=field.bytes;
        }
        std::sort(fields.begin(),fields.end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
        auto* words=reinterpret_cast<uint64_t*>(host);words[0]=fields.size();words[1]=header_bytes;words[2]=offset;words[3]=0;
        int64_t payload=0;
        for(std::size_t i=0;i<fields.size();++i) {
            words[4+i*3]=reinterpret_cast<uint64_t>(fields[i].destination);words[5+i*3]=fields[i].offset;words[6+i*3]=fields[i].bytes;payload+=fields[i].bytes;
        }
        MFQ_CUDA_CHECK(cudaMemcpyAsync(wire,host,header_bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
        for(const auto& interval:intervals)MFQ_CUDA_CHECK(cudaMemcpyAsync(wire+interval.offset,interval.source,interval.bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
        if(!staged.empty())MFQ_CUDA_CHECK(cudaMemcpyAsync(wire+staged_begin,host+staged_begin,offset-staged_begin,cudaMemcpyHostToDevice,cache_->weight_stream_));
        MFQ_CUDA_CHECK(cudaLaunchHostFunc(cache_->weight_stream_,[](void* p){publish(static_cast<uint32_t*>(p));},flags.h<uint32_t>()+4));
        MFQ_CUDA_CHECK(cudaEventRecord(cache_->transfer_ready_,cache_->weight_stream_));cache_->transfer_ready_recorded_=true;
        cache_->stats_.pipeline_prefetch_bytes+=payload;cache_->stats_.ram_pcie_bytes+=payload;
        cache_->stats_.pipeline_direct_ram_bytes+=direct_bytes;cache_->stats_.pipeline_staged_ram_bytes+=staged_bytes;
        cache_->stats_.pipeline_window_dma_bytes+=offset;
        cache_->stats_.pipeline_window_dma_copies+=1+intervals.size()+!staged.empty();
        mfq::cuda::check_stream_progress(cache_->weight_stream_);
        rollback.committed=true;
    }
    mfq::cuda::DecodeWindow::Task prediction(const tb::Tensor& ids) {
        if((!mfq::cuda::DecodeWindow::recording() && !cache_->ssd_experts()) || ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 ||
            ids.size(1)>(cache_->ssd_experts()?int64_t(32)*experts_:2) ||
            ids.scalar_type()!=tb::kInt32 || !ids.is_cuda() ||
            (!cache_->ssd_experts() && (!cache_->pipeline_transfer_cache_ || cache_->pipeline_mapped_copy_)))
            throw std::invalid_argument("MFQ prefetch requires CUDA int32 candidates, DMA and transfer caching");
        if(cache_->ssd_experts()) {
            const auto* window=mfq::cuda::DecodeWindow::recording();
            const int count=int(ids.numel());
            auto& entry=prediction_cells_[window];
            if(!entry || entry->flags.bytes<64+std::size_t(count)*sizeof(int32_t))
                entry=std::make_shared<PredictionCell>(std::size_t(count));
            auto prediction=entry;auto* mapped=&prediction->flags;
            prediction->routes=count;
            if(!window)std::memset(mapped->host,0,mapped->bytes);
            mfq::cuda::moe_publish_routes(nullptr,ids.data_ptr<int32_t>(),nullptr,mapped->d<int32_t>()+16,
                mapped->d<uint32_t>(),0,count,mfq_current_cuda_stream());
            auto owner=shared_from_this();const auto stream=mfq_current_cuda_stream();
            const auto weak=weak_from_this();
            // Retain the exported GPU IDs until the current route's fence.
            // A deferred cache callback must not retain its owning pipeline.
            std::function<void()> serve=[weak,prediction,mapped,count,stream,retained_ids=ids] {
                    const auto owner=weak.lock();if(!owner)return;
                    const auto started=std::chrono::steady_clock::now();
                    mfq::cuda::wait_route_publication(mapped->h<uint32_t>(),stream);
                    owner->cache_->ssd_prediction_route_wait_ns_+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-started).count();
                    std::vector<int> candidates;std::vector<uint8_t> seen(owner->experts_,0);
                    candidates.reserve(std::min(count,owner->experts_));
                    for(int i=0;i<count;++i) {
                        const int expert=mapped->h<int32_t>()[16+i];
                        if(expert<0 || expert>=owner->experts_)throw std::out_of_range("SSD predicted expert is out of range");
                        if(!seen[expert]){seen[expert]=1;candidates.push_back(expert);}
                    }
                    std::vector<MoeExpertCache::SsdExpertRequest> requests;
                    auto& gpu_fields=owner->cache_->ssd_forecast_gpu_fields_;gpu_fields.clear();
                    if(owner->cache_->kv_expert_loan_ && owner->cached_expert_fields_)
                        for(const int expert:candidates) {
                            const auto mask=owner->cached_expert_fields_->h<mfq::cuda::MfeCachedExpert>()[expert].ready_mask;
                            for(int p=0;p<3;++p)if(!(owner->combined_gate_up_ && p==1) && (mask&(uint64_t(1)<<p)))
                                gpu_fields.push_back({owner->sources_[p]->id_,expert,-1});
                        }
                    // Gate/Up unlocks the next layer's computation. Its exact
                    // demand phase reads Down while Gate/Up uploads and runs;
                    // speculative Down images compete with that critical I/O.
                    for(int p=0;p<2;++p)for(const int expert:candidates) {
                        const auto mask=owner->cached_expert_fields_?
                            owner->cached_expert_fields_->h<mfq::cuda::MfeCachedExpert>()[expert].ready_mask:uint64_t(0);
                        if((owner->single_up() && p==1) || (mask&(uint64_t(1)<<p)))continue;
                        auto* source=owner->sources_[p]->quant_source_.get();
                        if(!source->gpu_resident(expert))requests.emplace_back(source,expert);
                    }
                    owner->cache_->ssd_pending_prediction_=std::move(requests);
                };
            if(!window)serve=[weak,ready=std::move(serve)]() mutable {
                if(const auto live=weak.lock())live->cache_->ssd_pending_forecast_=std::move(ready);
            };
            return {prediction.get(),[prediction]{std::memset(prediction->flags.host,0,prediction->flags.bytes);},
                std::move(serve),{},[prediction]{publish(prediction->flags.h<uint32_t>()+8);publish(prediction->flags.h<uint32_t>()+4);},
                [owner,window]{owner->prediction_cells_.erase(window);}};
        }
        constexpr int64_t capacity=16*1024*1024;
        if(!cache_->pipeline_prefetch_host_stage_.defined()) {
            cache_->pipeline_prefetch_wire_gpu_=tb::empty({capacity},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
            cache_->pipeline_prefetch_host_stage_=tb::empty({capacity},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        }
        for(int p=0;p<3;++p)for(auto& cohort:sources_[p]->cohorts_) {
            auto& arena=stage(p,cohort,moe_decode_transfer_slots(8,10),true);if(!arena.book)arena.book=std::make_unique<mfq::MoeCacheSlotBook>(arena.slots);
        }
        const auto* window=mfq::cuda::DecodeWindow::recording();
        auto& entry=prediction_cells_[window];if(!entry)entry=std::make_shared<PredictionCell>();
        const int count=static_cast<int>(ids.numel());auto prediction=entry;
        auto* mapped=&prediction->flags;
        mfq::cuda::moe_publish_routes(nullptr,ids.data_ptr<int32_t>(),nullptr,mapped->d<int32_t>()+16,mapped->d<uint32_t>(),0,count,mfq_current_cuda_stream());
        auto owner=shared_from_this();const auto stream=mfq_current_cuda_stream();
        return {prediction.get(),[prediction]{std::memset(prediction->flags.host,0,prediction->flags.bytes);},
            [owner,prediction,mapped,count,stream] {
                mfq::cuda::wait_route_publication(mapped->h<uint32_t>(),stream);
                owner->prefetch_host(std::vector<int32_t>(mapped->h<int32_t>()+16,mapped->h<int32_t>()+16+count),*mapped);
            },{}, [prediction] {
                publish(prediction->flags.h<uint32_t>()+8);publish(prediction->flags.h<uint32_t>()+4);
            },[owner,window] {
                (void)cudaStreamSynchronize(owner->cache_->weight_stream_);
                owner->prediction_cells_.erase(window);
            }};
    }
    void register_idle_reclaimer() {
        cache_->pipeline_decode_release_.push_back([owner=weak_from_this()] {
            if(auto pipeline=owner.lock()) {
                pipeline->cells_.clear();pipeline->prediction_cells_.clear();pipeline->window_routes_.clear();
                pipeline->cached_expert_fields_.reset();
            }
        });
        cache_->pipeline_decode_indirect_.push_back([owner=weak_from_this()] {
            const auto pipeline=owner.lock();if(!pipeline)return true;
            if(!pipeline->prediction_cells_.empty() || !pipeline->window_routes_.empty())return false;
            for(const auto& [key,cell]:pipeline->cells_) {
                if(std::get<2>(key) || !cell->graph_replay || !cell->wire_fallback ||
                        !cell->indirect_wire || !cell->host_cache_packet || !cell->cached_resident)return false;
            }
            return true;
        });
    }
    ~MoeFfnPipeline() {
        // A host error releases every graph wait. Drain both producers before
        // destroying mapped flags that a graph or DMA callback still uses.
        if(stream_)(void)cudaStreamSynchronize(stream_);
        if(cache_->weight_stream_)(void)cudaStreamSynchronize(cache_->weight_stream_);
    }
    MoeFfnPipeline(std::array<std::shared_ptr<MoeCachedSource>,3> sources,MoeFfnShared shared,MoeFfnDispatchObserver observer,
            MoeFfnSharedWeights shared_weights,double swiglu_limit)
        :sources_(std::move(sources)),cache_(sources_[0]->cache_),shared_(std::move(shared)),shared_weights_(std::move(shared_weights)),observer_(std::move(observer)),
        width_(sources_[0]->cpu_->neuron_len),ff_(sources_[2]->cpu_->neuron_len),experts_(sources_[0]->n_experts()),
        combined_gate_up_(sources_[0]==sources_[1] && sources_[0]->cpu_->out_per_expert!=sources_[2]->cpu_->neuron_len),
        up_only_(sources_[0]==sources_[1] && sources_[0]->cpu_->out_per_expert==sources_[2]->cpu_->neuron_len) {
        swiglu_limit_=swiglu_limit;
        const auto trace=mfq::cuda::runtime_options::trace_moe_serve();
        trace_serve_=trace && *trace;
        for(const auto& s:sources_)if(!s->quant_source_ || s->cache_!=cache_ || s->n_experts()!=experts_)
            throw std::invalid_argument("MFQ FFN sources must share one expert cache");
        const int gate_up_rows=ff_*(combined_gate_up_?2:1);
        if(sources_[0]->cpu_->out_per_expert!=gate_up_rows ||
            sources_[1]->cpu_->neuron_len!=width_ || sources_[1]->cpu_->out_per_expert!=gate_up_rows ||
            sources_[2]->cpu_->neuron_len!=ff_ || sources_[2]->cpu_->out_per_expert!=width_)
            throw std::invalid_argument("MFQ FFN projection geometry mismatch");
        const std::array<int,3> bundle{sources_[0]->id_,sources_[1]->id_,sources_[2]->id_};
        if((bundle[0]==bundle[1] && !single_up()) || bundle[0]==bundle[2] || bundle[1]==bundle[2])
            throw std::invalid_argument("MFQ FFN requires distinct projection sources");
        cpu_cost_keys_.resize(experts_);
        for(int e=0;e<experts_;++e)for(int p=0;p<3;++p) {
            const auto& source=*sources_[p];
            cpu_cost_keys_[e][p]=reinterpret_cast<std::uintptr_t>(
                source.cohorts_[source.expert_to_cohort_[e]].arena);
        }
        if(std::find(cache_->pipeline_bundles_.begin(),cache_->pipeline_bundles_.end(),bundle)==cache_->pipeline_bundles_.end()) {
            if(cache_->moe_residency_)throw std::logic_error("MFQ cannot add an expert bundle after scheduling starts");
            cache_->pipeline_bundles_.push_back(bundle);
        }
    }
    tb::Tensor forward(CudaExecutionContext& execution,const tb::Tensor& input,const tb::Tensor& ids,const tb::Tensor& weights) {
        if(failed_)throw std::runtime_error("MFQ FFN session requires recovery after its previous error");
        if(!cache_->pipeline_ready())throw std::runtime_error("MFQ FFN needs completed startup cache initialization");
        const auto current=mfq_current_cuda_stream();
        if(stream_ && stream_!=current)throw std::runtime_error("MFQ FFN session changed its CUDA stream");
        stream_=current;
        auto* window=mfq::cuda::DecodeWindow::recording();
        const int tokens=static_cast<int>(ids.size(0)),routes=static_cast<int>(ids.size(1));
        const bool masked_rows = execution.decode_row_mask.defined();
        // Concurrency rows are decode tokens from independent sequences. Keep
        // ordinary short prompts on their existing prefill arithmetic path.
        const bool batch_decode=tokens<=8 || (tokens<=32 && execution.continuous_batch_cache_serial);
        if(!window && batch_decode)release_moe_prefill_buffers(cache_);
        if(input.dim()!=2 || input.size(0)!=tokens || input.size(1)!=width_ || !input.is_cuda() ||
            input.scalar_type()!=tb::kFloat16 || ids.dim()!=2 || ids.scalar_type()!=tb::kInt32 || !ids.is_cuda() ||
            weights.sizes().vec()!=ids.sizes().vec())throw std::invalid_argument("MFQ FFN input/route geometry mismatch");
        // Keep temporary pages reusable across every prefill path. The existing
        // allocation limits still apply, and decode restores the old threshold.
        if(!window && !batch_decode)retain_prefill_pool(ids);
        const auto layer_prefill=mfq::cuda::runtime_options::moe_prefill_layer();
        if(!window && !batch_decode && !masked_rows && layer_prefill && *layer_prefill!=0 &&
                cache_->pipeline_prefill_layer_layout_ &&
                cache_->pipeline_ram_registered_bytes_>0 && execution.kl_mmq.mode==KlMmqMode::Default) {
            if(!cache_->moe_residency_)cache_->moe_residency_=std::make_unique<MoeResidencyManager>(cache_);
            cache_->moe_residency_->before_layer(sources_[0]->layer_id_);
            try {return whole_layer_prefill(execution,input,ids,weights);}
            catch(...) {handle_prefill_failure();throw;}
        }
        if(!cache_->moe_residency_)cache_->moe_residency_=std::make_unique<MoeResidencyManager>(cache_);
        if(!window)cache_->moe_residency_->before_layer(sources_[0]->layer_id_);
        const auto prefill_groups=mfq::cuda::runtime_options::moe_prefill_groups();
        if(!window && !batch_decode && !masked_rows && prefill_groups.value_or(cache_->ssd_experts()) &&
                (cache_->pipeline_ram_registered_bytes_>0 || cache_->ssd_experts()) && cache_->pipeline_dispatch_replay_.empty() &&
                execution.kl_mmq.mode==KlMmqMode::Default) {
            try {return streamed_prefill(execution,input,ids,weights);}
            catch(...) {handle_prefill_failure();throw;}
        }
        const auto tile=execution.continuous_batch_ffn_tile;
        const bool final_tile=tile+1==execution.continuous_batch_ffn_tiles;
        auto& cell=cells_[{tokens,routes,window,batch_decode,tile}];
        if(!cell) {
            const auto prefill_graph=mfq::cuda::runtime_options::moe_prefill_graph();
            const bool eager=!window && !batch_decode &&
                (cache_->ssd_experts() || (prefill_graph && *prefill_graph==0));
            cell=std::make_unique<Cell>(tokens,routes,width_,experts_,stream_,eager);
            initialize(*cell,execution,window!=nullptr);
        }
        cell->allow_padding = execution.decode_row_mask.defined();
        // A prefill may have returned the revocable expert cache while keeping
        // this graph. Its descriptors resolve the newly allocated pages.
        if(cell->wire_fallback)cache_->prepare_direct_transfer_cache();
        ensure_host_stage(cell->host_stage_bytes);
        if(cell->indirect_wire)bind_wire(*cell);
        struct WireLifetime {
            Cell& cell;
            ~WireLifetime(){if(cell.indirect_wire)cell.wire={};}
        } wire_lifetime{*cell};
        if(window) {
            if(cache_->ssd_experts())if(const auto found=prediction_cells_.find(window);found!=prediction_cells_.end())
                cell->router_hint=found->second;
            auto* captured=cell.get();
            const bool independent=window->independent_sequences();
            std::shared_ptr<WindowRoutes> grouped;
            if(execution.continuous_batch_ffn_tiles>1) {
                auto& group=window_routes_[window];
                if(!group)group=std::make_shared<WindowRoutes>();
                grouped=group;
            }
            window->enroll({captured,
                [this,captured,grouped,tile,owner=shared_from_this()] {
                    if(failed_)throw std::runtime_error("MFQ FFN session requires recovery");
                    if(!tile) {
                        cache_->moe_residency_->before_layer(sources_[0]->layer_id_);
                        if(grouped){grouped->ids.clear();grouped->tokens=0;}
                    }
                    std::memset(captured->flags.host,0,captured->flags.bytes);
                },
                [this,captured,grouped,final_tile,independent] {
                    serve(*captured);
                    if(grouped) {
                        grouped->ids.insert(grouped->ids.end(),captured->host_ids.h<int32_t>(),
                            captured->host_ids.h<int32_t>()+captured->entries);
                        grouped->tokens+=captured->active_tokens;
                    }
                    if(final_tile && cache_->moe_residency_->prepare_during_window()) {

                        const std::vector<int32_t> routed(captured->host_ids.h<int32_t>(),captured->host_ids.h<int32_t>()+captured->entries);
                        cache_->moe_residency_->after_layer(sources_[0]->layer_id_,
                            grouped?grouped->ids:routed,grouped?grouped->tokens:captured->active_tokens,
                            true,independent);

                    }
                },
                [this,captured,grouped,final_tile,independent] {

                    observe_gpu(*captured);

                    if(!final_tile)return;
                    cache_->pipeline_dma_leases_.clear();

                    if(cache_->moe_residency_->prepare_during_window())cache_->moe_residency_->finish_window(sources_[0]->layer_id_);
                    else {
                        cache_->record_compute_use();
                        const std::vector<int32_t> routed(captured->host_ids.h<int32_t>(),captured->host_ids.h<int32_t>()+captured->entries);
                        cache_->moe_residency_->after_layer(sources_[0]->layer_id_,
                            grouped?grouped->ids:routed,grouped?grouped->tokens:captured->active_tokens,
                            false,independent);
                    }

                },
                [this,captured,final_tile] {
                    publish(captured->flags.h<uint32_t>()+64);publish(captured->flags.h<uint32_t>()+80);for(int i=1;i<4;++i)publish(captured->flags.h<uint32_t>()+16*i);
                    if(final_tile)cache_->moe_residency_->finish_window(sources_[0]->layer_id_);
                },
                [this,window,final_tile,key=std::make_tuple(tokens,routes,static_cast<const void*>(window),batch_decode,tile)] {
                    (void)cudaStreamSynchronize(cache_->weight_stream_);cells_.erase(key);
                    if(final_tile)window_routes_.erase(window);
                },[captured] {captured->output={};}});
            cell->window_input_views=cell->two_stage && mfq::cuda::runtime_options::window_input_views() &&
                input.is_contiguous() && ids.is_contiguous() && weights.is_contiguous() && weights.is_cuda() &&
                weights.scalar_type()==tb::kFloat32 && ids.device()==input.device() && weights.device()==input.device();
            if(cell->window_input_views)launch_body(*cell,execution,&input,&ids,&weights);
            else {
                cell->input.copy_(input);cell->ids.copy_(ids);cell->weights.copy_(weights);
                launch_body(*cell,execution);
            }
            if(final_tile)cache_->moe_residency_->record_window_expert_fence(sources_[0]->layer_id_,stream_);
            return cell->output;
        }
        const auto* graph=cell->graphs->find(1,tokens);
        if(cell->pending) {
            if(cell->graph_replay) {
                if(!graph->wait_ms(60000))throw std::runtime_error("MFQ previous FFN graph did not complete");
            } else cell->eager_done->synchronize();
        }
        if(cell->pending)observe_gpu(*cell);
        cell->pending=false;std::memset(cell->flags.host,0,cell->flags.bytes);
        if(cell->graph_replay) {
            cell->input.copy_(input);cell->ids.copy_(ids);cell->weights.copy_(weights);
        } else {
            // Eager launches take the current tensors directly. Their storage
            // remains valid through the stream-ordered consumers; no graph
            // requires a fixed address between prefill chunks.
            cell->input=input;cell->ids=ids;cell->weights=weights;
        }
        std::string error;
        try {
            if(cell->graph_replay) {
                if(!graph->launch(stream_,error))throw std::runtime_error(error);
            } else {
                launch_body(*cell,execution);
                cell->eager_done->record(stream_);
            }
            cell->pending=true;
            serve(*cell);cache_->record_compute_use();
            std::vector<int32_t> routed(cell->host_ids.h<int32_t>(),cell->host_ids.h<int32_t>()+cell->entries);
            cache_->moe_residency_->after_layer(sources_[0]->layer_id_,routed,tokens,false,cell->allow_padding);
        }
        catch(...) {failed_=true;publish(cell->flags.h<uint32_t>()+64);publish(cell->flags.h<uint32_t>()+80);for(int i=1;i<4;++i)publish(cell->flags.h<uint32_t>()+16*i);throw;}
        if(!cell->graph_replay) {
            auto output=std::move(cell->output);
            cell->input={};cell->ids={};cell->weights={};
            return output;
        }
        return cell->output;
    }
    friend MoeFfnForward make_moe_ffn_pipeline(const std::vector<std::shared_ptr<MfeWeight>>&,
        const std::vector<std::shared_ptr<MfeWeight>>&,MoeFfnShared,MoeFfnDispatchObserver,MoeFfnSharedWeights,MoeFfnPrefetch*,double);
};

MoeFfnForward make_moe_ffn_pipeline(const std::vector<std::shared_ptr<MfeWeight>>& gate_up,
        const std::vector<std::shared_ptr<MfeWeight>>& down,MoeFfnShared shared,MoeFfnDispatchObserver observer,
        MoeFfnSharedWeights shared_weights,MoeFfnPrefetch* prefetch,double swiglu_limit) {
    const auto resident = [](const auto& projections) {
        return std::all_of(projections.begin(), projections.end(),
            [](const auto& weight) { return weight && !weight->cached_source; });
    };
    if (resident(gate_up) && resident(down)) {
        // Resident projections have no cold expert images to prefetch.
        if (prefetch) *prefetch = [](const tb::Tensor&) { return mfq::cuda::DecodeWindow::Task{}; };
        return {};
    }
    if((gate_up.size()!=1 && gate_up.size()!=2) || down.size()!=1)return {};
    const auto& up=gate_up.back();
    for(const auto& w:{gate_up[0],up,down[0]})if(!w || !w->cached_source)return {};
    // Combined predictor projections support short decode and the whole-layer
    // prefill image; backbone/tensor-parallel dispatch remains unchanged.
    if(gate_up.size()==1 && (gate_up[0]->cached_source->layer_id()>=0 || down[0]->neuron_len%2))return {};
    if(gate_up.size()==1) {
        for(const auto& w:{gate_up[0],down[0]})if(!w->cached_source->two_stage_compatible())return {};
        if(shared)for(const auto& w:shared_weights.projections)
            if(!w || !w->is_nint() || w->tensor_parallel() || w->nint.q8_zero)return {};
    }
    if(!gate_up[0]->cached_source->pipeline_enabled())return {};
    auto runtime=std::make_shared<MoeFfnPipeline>(std::array<std::shared_ptr<MoeCachedSource>,3>{
        gate_up[0]->cached_source,up->cached_source,down[0]->cached_source},std::move(shared),std::move(observer),std::move(shared_weights),swiglu_limit);
    runtime->register_idle_reclaimer();
    if(prefetch)*prefetch=[runtime](const tb::Tensor& ids){return runtime->prediction(ids);};
    return [runtime](CudaExecutionContext& execution,const tb::Tensor& input,const tb::Tensor& ids,const tb::Tensor& weights){
        return runtime->forward(execution,input,ids,weights);
    };
}
