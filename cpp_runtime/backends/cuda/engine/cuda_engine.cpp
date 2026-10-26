#include "cuda_runtime_config.h"
#include "cuda_execution.h"
#include "generation.h"
#include "prefill_recovery.h"
#include "storage/model_loader.h"
#include "storage/weight_loader.h"
#include "storage/moe_expert_cache.h"
#include "runtime/kv_offload.h"
#include "runtime/kv_expert_loan.h"
#include "mfq_cuda_resident_allocations.h"
#include "runtime/execution_options.h"
#include "cuda_batching.h"
#include "storage/text_session_cache.h"
#include "mtp_metrics.h"
#include "request_executor.h"
#include "generation_flow.h"
#include "mfq_weight_memory.h"

#include <cuda_runtime_api.h>
#include "mfq_cuda_kv_quantization.h"
#include <algorithm>
#include <limits>
#include <variant>
#include <set>

namespace mfq::cuda {
using namespace mfq::engine;
using namespace mfq::cuda::internal;

namespace engine_detail {
template <typename Model>
struct CudaEngineState {
    std::shared_ptr<WeightMemoryLedger> weight_memory;
    std::shared_ptr<CudaExecutionContext> execution;
    CudaRuntimeConfig config;
    Model model;
    RuntimeComponents<Model> components;
    Model& language;
    DecodeGraphCache graph;
    TextSessionCache cache;
    std::unique_ptr<ContinuousBatch<CudaBatchOperations<Model>>> batching;
    bool duplex_active = false;
    std::size_t resident_budget=0, weight_budget=0, initial_expert_budget=0, allocation_ceiling=0;
    std::shared_ptr<KvExpertLoan> graph_expert_loan;
    mfq_tensor_backend::Tensor graph_expert_storage;
    std::optional<std::size_t> graph_expert_region;
    struct WorkspaceExpertRegion {mfq_tensor_backend::Tensor storage;std::size_t region;};
    std::vector<WorkspaceExpertRegion> workspace_expert_regions;
    std::uint64_t workspace_expert_reclaimed_used_bytes=0;
    int graph_draft_depth=3;
    bool preparing_graph_depth=true;

    std::size_t workspace_expert_capacity() const {
        std::size_t bytes=0;for(const auto& part:workspace_expert_regions)bytes+=part.storage.nbytes();return bytes;
    }
    std::size_t workspace_expert_used() const {
        std::size_t bytes=0;
        if(graph_expert_loan)for(const auto& part:workspace_expert_regions)bytes+=graph_expert_loan->region_bytes(part.region);
        return bytes;
    }
    void release_graph_expert_space(std::size_t required_available=std::numeric_limits<std::size_t>::max()) {
        if(!graph_expert_region && workspace_expert_regions.empty())return;
        finish_moe_expert_exchanges(execution->moe_expert_cache);
        MFQ_CUDA_CHECK(cudaDeviceSynchronize());
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const bool all=required_available==std::numeric_limits<std::size_t>::max();
        while(!workspace_expert_regions.empty() && (all || context->available_memory()<required_available)) {
            const auto& part=workspace_expert_regions.back();
            const auto bytes=part.storage.nbytes();
            const auto used=graph_expert_loan->region_bytes(part.region);
            workspace_expert_reclaimed_used_bytes+=used;
            graph_expert_loan->remove(part.region);
            workspace_expert_regions.pop_back();
            context->reclaim_idle_memory();
            std::cerr<<"runtime_prefill_workspace_cache action=reclaim bytes="<<bytes<<" used_bytes="<<used<<'\n';
        }
        if(graph_expert_region && (all || context->available_memory()<required_available)) {
            const auto bytes=graph_expert_storage.nbytes();
            graph_expert_loan->remove(*graph_expert_region);
            graph_expert_region.reset();graph_expert_storage={};
            std::cerr<<"runtime_mtp_workspace_cache action=reclaim bytes="<<bytes<<'\n';
        }
        context->reclaim_idle_memory();
    }

    void restore_mtp_expert_space() {
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(preparing_graph_depth || graph_expert_region || !graph_expert_loan || !components.mtp)return;
            const auto maximum=qwen4_mtp_graph_reserve(language,components.mtp.get());
            const auto active=qwen4_mtp_graph_reserve(language,components.mtp.get(),graph_draft_depth);
            if(active>=maximum)return;
            auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
            const auto available=context->available_memory(),headroom=std::size_t(256)<<20;
            auto bytes=std::min(maximum-active,available>headroom?available-headroom:0);
            bytes=bytes/KvExpertLoan::page*KvExpertLoan::page;
            if(!bytes)return;
            // An ordinary resident tensor supplies revocable expert pages;
            // no decode graph captures an address into this allocation.
            ResidentAllocationScope resident;
            try {
                graph_expert_storage=mfq_tensor_backend::empty({int64_t(bytes)},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kUInt8));
            } catch(const std::exception& error) {
                const std::string message=error.what();
                if(message.find("VRAM limit")==std::string::npos && message.find("out of memory")==std::string::npos)throw;
                std::cerr<<"runtime_mtp_workspace_cache action=skip reason=memory_pressure\n";
                return;
            }
            graph_expert_region=graph_expert_loan->add(graph_expert_storage,int64_t(bytes),1,true,1,true,true);
            std::cerr<<"runtime_mtp_workspace_cache action=lend depth="<<graph_draft_depth<<" bytes="<<bytes<<'\n';
        }
    }

    void restore_graph_expert_space() {
        if(preparing_graph_depth)return;
        prepare_moe_decode_residency(execution->moe_expert_cache);
        mfq::cuda::default_context(execution->layer_placement.primary_device())->reclaim_idle_memory();
        restore_mtp_expert_space();
        if(!graph_expert_loan)return;
        bool ssd=false;std::size_t transfer=0,streamed=0;
        const auto expert_metrics=moe_expert_memory_metrics(execution->moe_expert_cache);
        for(const auto& [name,value]:expert_metrics) {
            if(name=="ssd_expert_enabled")ssd=value!=0;
            if(name=="expert_decode_stage_budget_bytes")transfer=std::size_t(value);
            if(name=="ram_expert_payload_bytes" || name=="ssd_expert_payload_bytes")streamed+=std::size_t(value);
        }
        if(!ssd)return;
        if(const auto* profile=std::getenv("MFQ_PROFILE_SSD_EXPERTS");profile && profile[0]=='1' && language.source) {
            const auto io=language.source->file_read_stats();
            std::cerr<<"runtime_model_io_stats boundary=prefill_end calls="<<io.calls
                <<" logical_bytes="<<io.logical_bytes<<" physical_bytes="<<io.physical_bytes
                <<" read_ms="<<double(io.read_nanoseconds)/1e6<<'\n';
            std::cerr<<"runtime_moe_pipeline_stats boundary=prefill_end";
            for(const auto& [name,value]:expert_metrics)
                if(name=="expert_route_wait_ms" || name=="expert_plan_ms" ||
                   name=="expert_host_fetch_ms" || name=="expert_logical_transfer_bytes" ||
                   name=="ssd_expert_read_wait_ms")std::cerr<<' '<<name<<'='<<value;
            std::cerr<<'\n';
        }
        // Spare device memory can hold revocable copies of streamed experts.
        // Primary arenas retain their configured budget. Prefill, KV growth
        // and online resident budgets return these chunks before allocation.
        const auto reserve=runtime_options::workspace_reserve_bytes();
        const auto decode=(std::size_t(1280)<<20)+transfer;
        const auto existing=workspace_expert_capacity();
        const auto target=std::max(streamed,reserve>decode?reserve-decode:0);
        if(existing>=target)return;
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto available=context->available_memory(),headroom=std::size_t(256)<<20;
        auto bytes=std::min(target-existing,available>headroom?available-headroom:0);
        bytes=bytes/KvExpertLoan::page*KvExpertLoan::page;
        if(!bytes)return;
        ResidentAllocationScope resident;
        // Small KV or prefix growth returns only the required chunks, keeping
        // the rest of the decoded expert working set available to MTP.
        while(bytes) {
            const auto chunk=std::min(bytes,std::size_t(256)<<20);
            mfq_tensor_backend::Tensor storage;
            try {
                storage=mfq_tensor_backend::empty({int64_t(chunk)},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kUInt8));
            }catch(const std::exception& error) {
                const std::string message=error.what();
                if(message.find("VRAM limit")==std::string::npos && message.find("out of memory")==std::string::npos)throw;
                break;
            }
            workspace_expert_regions.reserve(workspace_expert_regions.size()+1);
            const auto region=graph_expert_loan->add(storage,int64_t(chunk),1,true,1,true,true);
            workspace_expert_regions.push_back({std::move(storage),region});
            bytes-=chunk;
        }
        std::cerr<<"runtime_prefill_workspace_cache action=lend bytes="<<workspace_expert_capacity()-existing
            <<" capacity_bytes="<<workspace_expert_capacity()<<" reserve_bytes="<<reserve<<'\n';
    }

    void prepare_graph_depth(const InferenceRequest& input) {
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(!components.mtp)return;
            const auto* warmup=std::getenv("MFQ_QWEN4_WARMUP_GRAPHS");
            if(!language.reuse_decode_graphs || !execution->config.moe_pipeline || (warmup && warmup[0]=='0'))return;
            const auto& sampling=input.sampling;
            const bool wide=sampling.temperature<=0 || sampling.top_k==1 ||
                (sampling.top_k>0 && sampling.top_k<=64);
            const int requested=sampling.enable_mtp && wide
                ?std::clamp(int(sampling.mtp_max_draft_tokens),1,components.mtp->maximum_draft_depth()):1;
            // Short prefill shares the existing four-row graphs. Only draft
            // caps above three need additional verification storage.
            const int depth=std::max(3,requested);
            if(depth==graph_draft_depth && language.graph_registry)return;
            PrefillAllocationScope allocation_scope;
            const auto started=std::chrono::steady_clock::now();
            preparing_graph_depth=true;
            try {
                release_graph_expert_space();
                graph.invalidate();
                qwen4_release_decode_graphs(language,components.mtp.get());
                release_moe_decode_buffers(execution->moe_expert_cache);
                language.reset(1);components.mtp->reset(1);
                auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
                context->reclaim_idle_memory();
                reclaim_prefix_workspace((std::size_t(2)<<30)+qwen4_mtp_graph_reserve(language,components.mtp.get(),depth));
                qwen4_warmup_decode_graphs(language,components.mtp.get(),depth);
                graph_draft_depth=depth;
                preparing_graph_depth=false;
                restore_graph_expert_space();
            } catch(...) {
                graph_draft_depth=0;preparing_graph_depth=false;
                throw;
            }
            std::cerr<<"runtime_mtp_depth_plan requested_depth="<<requested<<" prepared_depth="<<depth
                <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<'\n';
        }
    }

    bool online_budget_supported() const {
        return execution->layer_placement.devices.size()<=1 && !execution->tensor_parallel.enabled()
            && !execution->expert_parallel.enabled();
    }

    void reclaim_prefix_workspace(std::size_t bytes) {
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto available=context->available_memory();
        if(available>=bytes)return;
        if(graph_expert_region || !workspace_expert_regions.empty()) {
            std::cerr<<"runtime_workspace_cache_reclaim reason=workspace requested_bytes="<<bytes
                <<" available_bytes="<<available<<'\n';
            release_graph_expert_space(bytes);
        }
        if(context->available_memory()>=bytes)return;
        std::size_t hot=0;
        for(const auto& [name,value]:cache.metrics())
            if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
        const auto shortage=bytes-context->available_memory();
        const auto released=cache.trim_hot(hot>shortage?hot-shortage:0);
        context->reclaim_idle_memory();
        if(released)std::cerr<<"runtime_workspace_reclaim requested_bytes="<<bytes
            <<" available_bytes="<<context->available_memory()<<" prefix_released_bytes="<<released<<'\n';
    }

    CudaEngineState(std::shared_ptr<CudaExecutionContext> owner, Model loaded,
            RuntimeComponents<Model> optional, CudaRuntimeConfig runtime_config,
            std::shared_ptr<WeightMemoryLedger> weights)
        : weight_memory(std::move(weights)), execution(std::move(owner)), config(std::move(runtime_config)),
          model(std::move(loaded)), components(std::move(optional)),
          language(components.language(model)), graph(language.max_position_embeddings()),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*language.source, language.max_position_embeddings(),
                  language.supports_paged_text_session_state(), config.prefix_cache),
                language.supports_text_session_state(), language.supports_text_session_state() ? 0 : 1,
                cuda_has_separate_memory()) {
        initial_expert_budget=moe_expert_cache_budget_bytes(execution->moe_expert_cache);
        if (execution->qsa_kv_store)
            execution->qsa_kv_store->set_ram_reservation([this](std::size_t bytes) {
                cache.reserve_ram(bytes);reclaim_moe_optional_host(execution->moe_expert_cache,bytes);
            });
        allocation_ceiling=mfq::cuda::default_context(execution->layer_placement.primary_device())->memory_stats().limit;
        language.reclaim_prefill_workspace=[this] {
            graph.invalidate();
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
                qwen4_release_decode_graphs(language,components.mtp.get());
            graph_draft_depth=0;
            release_graph_expert_space();
            cache.trim_hot(0);
        };
        if (config.continuous_batch.max_sequences) {
            if constexpr (std::is_same_v<Model, Qwen35CausalLm> || std::is_same_v<Model,Qwen4CausalLm>)
                batching = std::make_unique<ContinuousBatch<CudaBatchOperations<Model>>>(
                    config.generation.prefill_chunk_size, config.continuous_batch.prefill_token_budget,
                    language, *execution, config.continuous_batch, components.mtp.get(), BatchMemoryProvider{
                        [this] {
                            auto evictable = cache.device_hot_bytes();
                            auto& primary = evictable[execution->layer_placement.primary_device()];
                            primary = prefill_budget_add(primary, workspace_expert_capacity());
                            if (graph_expert_storage.defined())
                                primary = prefill_budget_add(primary, graph_expert_storage.nbytes());
                            return evictable;
                        },
                        [this](const KvAdmissionBytes& required) {
                            for (const auto& [device, bytes] : required) {
                                MfqCudaGuard guard(device);
                                auto context = mfq::cuda::default_context(device);
                                if (device == execution->layer_placement.primary_device())
                                    reclaim_prefix_workspace(bytes);
                                else if (context->available_memory() < bytes) {
                                    cache.trim_hot(0);
                                    context->reclaim_idle_memory();
                                }
                            }
                        }});
            else
                // Capacity is a serving hint. A model without per-request batch
                // state keeps the single-request path and reports capacity one
                // instead of failing the load for an unrelated reason.
                std::cerr<<"continuous_batching unavailable model="<<language.model_type()
                    <<" requested_capacity="<<config.continuous_batch.max_sequences<<'\n';
        }
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        if(execution->moe_expert_cache) {
            graph_expert_loan=execution->kv_memory?execution->kv_memory->expert_loan():nullptr;
            if(!graph_expert_loan && (components.mtp || moe_expert_cache_ssd_experts(execution->moe_expert_cache))) {
                graph_expert_loan=std::make_shared<KvExpertLoan>();
                set_moe_kv_expert_loan(execution->moe_expert_cache,graph_expert_loan);
            }
            language.restore_decode_workspace=[this]{restore_graph_expert_space();};
            set_moe_workspace_reservation(execution->moe_expert_cache,
                [this](std::size_t bytes) {reclaim_prefix_workspace(bytes);},
                [this](std::size_t bytes) {
                    cache.reserve_ram(bytes);reclaim_moe_optional_host(execution->moe_expert_cache,bytes);
                });
        }
        if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
            language.reserve_kv_workspace = [this](std::size_t bytes) {
                const auto device = execution->layer_placement.primary_device();
                auto context = mfq::cuda::default_context(device);
                if (context->available_memory() >= bytes) return;
                if(graph_expert_region || !workspace_expert_regions.empty()) {
                    std::cerr<<"runtime_workspace_cache_reclaim reason=kv_growth requested_bytes="<<bytes
                        <<" available_bytes="<<context->available_memory()<<'\n';
                    release_graph_expert_space(bytes);
                }
                if(context->available_memory()>=bytes)return;
                std::size_t hot = 0;
                for (const auto& [name,value] : cache.metrics())
                    if (name == "prefix_cache_hot_bytes") hot = std::size_t(value);
                const auto shortage = bytes - context->available_memory();
                cache.trim_hot(hot > shortage ? hot - shortage : 0);
                context->reclaim_idle_memory();
                if (context->available_memory() < bytes)
                    throw InferenceInputError(InferenceInputErrorCode::ContextLength,
                        "insufficient VRAM for context growth; reduce concurrent requests or prompt length");
            };
            if(execution->kv_memory)execution->kv_memory->set_reservation(language.reserve_kv_workspace);
        }
#endif
        if(const auto bytes=runtime_options::resident_budget_bytes()) {
            SessionCommand command;
            command.kind=SessionCommand::Kind::memory_budget;
            command.bytes=bytes;
            session(command);
        }
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            qwen4_warmup_decode_graphs(language,components.mtp.get());
            // Exercise the same draft/verify/commit loop as a real request.
            // Graph-only warmup cannot cover its history replay, sampling and
            // expert exchange boundaries. No synthetic policy/history survives.
            const auto* warmup=std::getenv("MFQ_QWEN4_WARMUP_MTP");
            if(language.graph_registry && components.mtp &&
                    language.max_position_embeddings()>=48 && (!warmup || warmup[0]!='0')) {
                struct CacheLearningScope {
                    std::shared_ptr<MoeExpertCache> cache;bool previous;
                    explicit CacheLearningScope(std::shared_ptr<MoeExpertCache> value):cache(std::move(value)),
                        previous(set_moe_reuse_learning(cache,false)) {}
                    ~CacheLearningScope(){set_moe_reuse_learning(cache,previous);}
                } learning(execution->moe_expert_cache);
                auto& predictor=*components.mtp;
                const auto saved_policy=predictor.policy_state;
                const auto start=std::chrono::steady_clock::now();
                InferenceRequest request;
                request.chat=false;
                request.sampling.temperature=0.7;
                request.sampling.top_p=0.8;
                request.sampling.top_k=20;
                request.sampling.seed=1234;
                request.sampling.max_tokens=static_cast<int32_t>(
                    std::min<int64_t>(64,language.max_position_embeddings()-16));
                request.sampling.mtp_max_draft_tokens=3;
                for(int i=0;i<16;++i)request.prompt.push_back((i*7919+17)%language.vocab_size());
                InferenceOutput output(request,nullptr,"cuda-mtp-warmup");
                auto scope=language.execution_scope();
                auto generation=run_mtp_generation(language,predictor,request,output,16);
                while(generation.next()) {}
                finish_moe_expert_exchanges(execution->moe_expert_cache);
                predictor.reset(1);
                language.reset(1);
                predictor.policy_state=saved_policy;
                predictor.last_stats={};
                predictor.last_cycles=predictor.last_accepted=predictor.last_rejected=0;
                MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
                std::cerr<<"runtime_qwen4_mtp_warmup tokens="<<output.result.completion_tokens
                    <<" ms="<<std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-start).count()<<'\n';
                release_moe_prefill_buffers(execution->moe_expert_cache);
                mfq::cuda::default_context(execution->layer_placement.primary_device())->reclaim_idle_memory();
            }
        }
        preparing_graph_depth=false;
        restore_graph_expert_space();
    }
    bool can_batch(const EngineRequest& request) const {
        const bool exclusive_mtp = request.input.sampling.enable_mtp && components.mtp &&
            !batch_mtp_supported(components.mtp.get(), request.input.sampling);
        if constexpr (std::is_same_v<Model,Qwen4CausalLm>)
            return batching && !request.input.media && request.input.cache_plan.session_id.empty() &&
                !exclusive_mtp;
        return batching && batch_compatible(request, cache.persistent_prefix_enabled(), exclusive_mtp);
    }
    // Report the rows this model can batch so the control plane never admits
    // more concurrent requests than the worker can execute.
    std::size_t batch_capacity() const {
        return batching ? std::max<std::size_t>(1, config.continuous_batch.max_sequences) : 1;
    }
    ~CudaEngineState() {
        language.restore_decode_workspace={};
        language.reclaim_prefill_workspace={};
        try {release_graph_expert_space();}
        catch(const std::exception& error) {std::cerr<<"runtime_mtp_workspace_cache cleanup_error="<<error.what()<<'\n';}
        if(execution->kv_memory)execution->kv_memory->set_reservation({});
        set_moe_workspace_reservation(execution->moe_expert_cache,{});
        if (execution->qsa_kv_store) execution->qsa_kv_store->set_ram_reservation({});
    }
    SessionResult session(const SessionCommand& command) {
        if(command.kind!=SessionCommand::Kind::memory_budget)return control_session(cache,command);
        if(!online_budget_supported())throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
        const auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto dense=weight_memory->resident_bytes();
        const auto minimum=moe_expert_cache_minimum_bytes(execution->moe_expert_cache);
        const auto requested_weights=command.weight_bytes.value_or(weight_budget);
        if(requested_weights && requested_weights<dense+minimum)
            throw std::invalid_argument("weight budget is below dense weights and minimum expert working set");
        auto limit=command.bytes ? allocation_ceiling ? std::min<std::size_t>(command.bytes,allocation_ceiling) : command.bytes : allocation_ceiling;
        if(limit) {
            std::size_t free=0,total=0;
            MFQ_CUDA_CHECK(cudaMemGetInfo(&free,&total));
            limit=std::min(limit,std::min(total,free+context->resident_memory_usage()));
        }
        if(limit && limit<dense+minimum)
            throw std::invalid_argument("resident memory budget is below non-offloadable weights");
        const auto previous_limit=context->memory_stats().limit;
        const auto previous_experts=moe_expert_cache_budget_bytes(execution->moe_expert_cache);
        const auto before=context->resident_memory_usage();
        if(batching)batching->release_idle_buffers();
        release_graph_expert_space();
        finish_moe_expert_exchanges(execution->moe_expert_cache);
        graph.invalidate();
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            qwen4_release_decode_graphs(language,components.mtp.get());
        if(components.mtp)components.mtp->reset(1);
        language.reset(1);
        release_moe_decode_buffers(execution->moe_expert_cache);
        context->reclaim_idle_memory();
        auto target=requested_weights ? requested_weights-dense : initial_expert_budget;
        std::size_t hot=0,expert_bytes=0;
        for(const auto& [name,value]:cache.metrics())if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
        for(const auto& [name,value]:moe_expert_memory_metrics(execution->moe_expert_cache))
            if(name=="resident_expert_cache_bytes")expert_bytes=std::size_t(value);
        auto used=context->resident_memory_usage();
        if(limit && used>limit) {
            cache.trim_hot(hot>used-limit ? hot-(used-limit) : 0);
            context->reclaim_idle_memory();used=context->resident_memory_usage();
        }
        const auto fixed=used>expert_bytes?used-expert_bytes:0;
        if(limit) {
            if(fixed>limit || minimum>limit-fixed)throw std::invalid_argument("resident memory budget is below live KV and workspace requirements");
            target=std::min(target,limit-fixed);
        }
        try {
            if(!limit || limit>previous_limit)context->set_memory_limit(limit);
            if(execution->moe_expert_cache)resize_moe_expert_cache(execution->moe_expert_cache,std::max(target,minimum));
            context->set_memory_limit(limit);
        } catch(...) {
            context->set_memory_limit(0);
            if(execution->moe_expert_cache)resize_moe_expert_cache(execution->moe_expert_cache,previous_experts);
            context->set_memory_limit(previous_limit);
            throw;
        }
        resident_budget=command.bytes;weight_budget=requested_weights;
        const auto after=context->resident_memory_usage();
        return {before>after?before-after:0,{}};
    }
    bool exclusive() const { return duplex_active; }
    bool mtp_available() const { return bool(components.mtp); }
    void execute(const std::vector<RequestId>& eligible) {
        if (batching) batching->step(eligible);
    }
    using Prepared = CudaPreparedPrompt;
    mfq::StepSequence<Prepared> prepare(InferenceRequest& input) {
        if (components.composite) return components.composite->prepare(input.prompt, *input.vision);
        if (components.grid_vision) return components.grid_vision->prepare(language, input.prompt, *input.vision);
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>)
            if(components.deepseek_vision)return components.deepseek_vision->prepare(language,input.prompt,*input.vision);
        if (components.radio_vision) return components.radio_vision->prepare(language, input.prompt, *input.vision);
        throw std::invalid_argument("model has no multimodal component");
    }
    auto advance_preparation(mfq::StepSequence<Prepared>& preparation) {
        mfq::cuda::PrefillAllocationScope allocation_scope;
        PrefillCudaTimer timer;
        auto step = preparation.next();
        MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        return std::pair{std::move(step), timer.elapsed_ms()};
    }
    void recover_preparation(const std::exception_ptr& failure,bool reclaimed) {
        if(!mfq::engine::prefill_resource_failure(failure))std::rethrow_exception(failure);
        internal::reclaim_prefill_memory(language);
        language.reset(1);
        if(components.mtp)components.mtp->reset(1);
        if(reclaimed)throw MfqResourceExhausted("vision prefill resource exhausted after workspace reclamation: "+
            mfq::engine::prefill_failure_message(failure));
        std::cerr<<"runtime_prefill_retry scope=vision action=reclaim\n";
    }
    Generation generate_text(InferenceRequest& input, InferenceOutput& output, std::optional<Prepared> prepared,
                             const RequestId& id, bool batched) {
        if(!batched) {
            try {prepare_graph_depth(input);}
            catch(...) {
                const auto failure=std::current_exception();
                if(!prefill_resource_failure(failure))std::rethrow_exception(failure);
                internal::reclaim_prefill_memory(language);
                language.reset(1);
                if(components.mtp)components.mtp->reset(1);
                std::cerr<<"runtime_prefill_retry scope=graph action=reclaim reason="
                    <<prefill_failure_message(failure)<<'\n';
            }
        }
        return internal::generate(language, graph, cache, config,
                                  input, output, batched ? nullptr : components.mtp.get(), std::move(prepared),
                                  batched ? batching.get() : nullptr, id);
    }
    Generation generate(const RequestId& id, ExecutionRequest& request) {
        if (request.batched) {
            graph.invalidate();
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
                // Exclusive MTP generation has ended before a batch starts.
                // Its graph consumers must retire before idle batch buffers.
                if (batching->empty()) {
                    qwen4_release_decode_graphs(language,components.mtp.get());
                    if (components.mtp) components.mtp->reset(1);
                }
            }
        } else if (batching) {
            batching->release_idle_buffers();
        }
        return generate_prepared(*this, request.input, request.output, id, request.batched);
    }
    template <class T> ControlResult control(T value) {
        if constexpr (std::is_same_v<T, RuntimeMetrics>) {
            const auto batch_metrics = batching ? batching->metrics() : Metrics{};
            const auto capabilities = components.state();
            Metrics metrics{
                {"mtp_graph_prepared_depth",double(graph_draft_depth)},
                {"mtp_flexible_expert_capacity_bytes",double(graph_expert_storage.defined()?graph_expert_storage.nbytes():0)},
                {"mtp_flexible_expert_used_bytes",double(graph_expert_loan && graph_expert_region?
                    graph_expert_loan->region_bytes(*graph_expert_region):0)},
                {"prefill_workspace_expert_capacity_bytes",double(workspace_expert_capacity())},
                {"prefill_workspace_expert_used_bytes",double(workspace_expert_used())},
                {"prefill_workspace_expert_reclaimed_used_bytes",double(workspace_expert_reclaimed_used_bytes)},
                {"kv_memory_supported",std::is_same_v<Model,Qwen4CausalLm>?1.0:0.0},
                {"kv_quantization_supported",(std::is_same_v<Model,Qwen4CausalLm>||std::is_same_v<Model,Qwen35CausalLm>)?1.0:0.0},
                {"kv_quantization_bits",(std::is_same_v<Model,Qwen4CausalLm>||std::is_same_v<Model,Qwen35CausalLm>)?cuda_kv_quantization().bits:0.0},
                {"vision_declared", double(capabilities.vision_declared)}, {"vision_supported", double(capabilities.vision_supported)},
                {"vision_available", double(capabilities.vision_available)}, {"mtp_declared", double(capabilities.mtp_declared)},
                {"mtp_supported", double(capabilities.mtp_supported)}, {"mtp_available", double(capabilities.mtp_available)}};
            auto experts = moe_expert_memory_metrics(execution->moe_expert_cache);
            metrics.insert(metrics.end(), experts.begin(), experts.end());
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
                auto phases=qwen4_decode_graph_profile_metrics(language,components.mtp.get());
                metrics.insert(metrics.end(),phases.begin(),phases.end());
            }
            if(language.source) {
                const auto io=language.source->file_read_stats();
                metrics.insert(metrics.end(),{{"model_io_direct",io.mode==mfq::FileReadMode::Direct?1.0:0.0},
                    {"model_io_read_calls",double(io.calls)},{"model_io_logical_bytes",double(io.logical_bytes)},
                    {"model_io_physical_bytes",double(io.physical_bytes)},{"model_io_read_ms",double(io.read_nanoseconds)/1e6},
                    {"model_io_staging_bytes",double(io.staging_bytes)},{"model_io_cached_staging_bytes",double(io.staging_cached_bytes)},
                    {"model_io_staging_allocations",double(io.staging_allocations)}});
            }
            if(execution->kv_memory) {
                auto kv=execution->kv_memory->metrics();
                metrics.insert(metrics.end(),kv.begin(),kv.end());
            }
            std::size_t resident_weights=weight_memory->resident_bytes();
            for(const auto& metric:experts)
                if(metric.first=="resident_expert_cache_bytes")resident_weights+=std::size_t(metric.second);
            const auto ple=weight_memory->ple_bytes();
            metrics.insert(metrics.end(),{{"resident_weight_bytes",double(resident_weights)},
                {"ssd_ple_enabled",ple>0?1.0:0.0},{"ssd_ple_payload_bytes",double(ple)}});
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                auto cache_memory=qwen4_cache_memory_metrics(language,components.mtp.get());
                for (auto& [name,value] : cache_memory)
                    for (const auto& [batch_name,batch_value] : batch_metrics) {
                        if (name == "kv_cache_bytes" && batch_name == "continuous_batching_request_kv_bytes")
                            value += batch_value;
                        if (name == "kv_cache_contexts" && batch_name == "continuous_batching_request_contexts")
                            value += batch_value;
                    }
                metrics.insert(metrics.end(),cache_memory.begin(),cache_memory.end());
            }
            const auto metric_bytes = [&](const std::string& key) {
                for(const auto& [name,value]:metrics)if(name==key)return std::size_t(value);
                return std::size_t(0);
            };
            std::set<int> devices(execution->layer_placement.devices.begin(),execution->layer_placement.devices.end());
            devices.insert(execution->tensor_parallel.devices.begin(),execution->tensor_parallel.devices.end());
            devices.insert(execution->expert_parallel.devices.begin(),execution->expert_parallel.devices.end());
            const auto primary=execution->layer_placement.primary_device();
            devices.insert(primary);
            auto prefix_bytes=cache.device_hot_bytes();
            std::size_t allocated=0,reserved=0,free_bytes=0,total_bytes=0,buffers=0;
            for(const int device:devices) {
                MfqCudaGuard guard(device);
                size_t free=0,total=0;
                MFQ_CUDA_CHECK(cudaMemGetInfo(&free,&total));
                const auto memory=mfq_cuda_memory_stats(device);
                const auto external=device==primary?metric_bytes("kv_memory_vmm_bytes"):0;
                const auto resident=std::max(memory.reserved_bytes,memory.allocated_bytes)+external;
                const auto stem="gpu_"+std::to_string(device)+"_";
                auto weights=weight_memory->resident_bytes(device);
                if(device==primary)weights+=metric_bytes("resident_expert_cache_bytes");
                const auto hot=prefix_bytes[device];
                auto live=metric_bytes(stem+"live_kv_bytes");
                if(device==primary)for(const auto& [name,value]:batch_metrics)
                    if(name=="continuous_batching_request_kv_bytes")live+=std::size_t(value);
                metrics.insert(metrics.end(),{{stem+"resident_bytes",double(resident)},
                    {stem+"resident_weight_bytes",double(weights)}, {stem+"prefix_cache_bytes",double(hot)},
                    {stem+"memory_total_bytes",double(total)}, {stem+"memory_available_bytes",double(free)}});
                if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                    const auto categorized=weights+live+hot;
                    const auto buffer=resident>categorized?resident-categorized:0;
                    metrics.emplace_back(stem+"resident_buffer_bytes",double(buffer));
                    metrics.emplace_back(stem+"kv_bytes",double(live+hot));
                    buffers+=buffer;
                }
                allocated+=memory.allocated_bytes+external;reserved+=memory.reserved_bytes+external;free_bytes+=free;total_bytes+=total;
            }
            metrics.insert(metrics.end(),{{"device_free_bytes",double(free_bytes)}, {"device_total_bytes",double(total_bytes)},
                {"cuda_allocated_bytes",double(allocated)}, {"cuda_reserved_bytes",double(reserved)}});
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>)metrics.emplace_back("resident_buffer_bytes",double(buffers));
            const auto expert_bytes=metric_bytes("resident_expert_cache_bytes");
            const auto minimum=moe_expert_cache_minimum_bytes(execution->moe_expert_cache);
            metrics.insert(metrics.end(),{{"resident_memory_dynamic_budget",online_budget_supported()?1.0:0.0},
                {"resident_weight_dynamic_budget",online_budget_supported()?1.0:0.0},
                {"resident_memory_budget_bytes",double(resident_budget)}, {"resident_weight_budget_bytes",double(weight_budget)},
                {"resident_memory_used_bytes",double(std::max(allocated,reserved))},
                {"resident_reclaimable_weight_bytes",double(expert_bytes>minimum?expert_bytes-minimum:0)}});
            metrics.emplace_back("qsa_kv_offload_supported",std::is_same_v<Model, Qwen4CausalLm> ? 1.0 : 0.0);
            if (execution->qsa_kv_store) {
                auto kv = execution->qsa_kv_store->metrics();
                metrics.insert(metrics.end(), kv.begin(), kv.end());
            } else metrics.insert(metrics.end(), {{"qsa_kv_offload_enabled", 0.0},
                {"qsa_kv_ram_enabled", 0.0}, {"qsa_kv_ram_bytes", 0.0}, {"qsa_kv_ssd_bytes", 0.0}});
            if (components.mtp) {
                mtp::append_generation_metrics(metrics, components.mtp->last_stats);
                metrics.emplace_back("mtp_predictor_graph_replays",double(components.mtp->graph_replay_count()));
            }
            if (batching) {
                metrics.insert(metrics.end(), batch_metrics.begin(), batch_metrics.end());
            }
            return metrics;
        } else {
            if (!components.composite) throw std::invalid_argument("model has no duplex component");
            if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) {
                components.composite->start(value); duplex_active = true;
            } else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) {
                return components.composite->step(value);
            } else { components.composite->stop(); duplex_active = false; }
            return std::monostate{};
        }
    }

};
using State = std::variant<
    std::unique_ptr<CudaEngineState<Qwen35CausalLm>>,
    std::unique_ptr<CudaEngineState<MiniCPMO45CausalLm>>,
    std::unique_ptr<CudaEngineState<MiniCPMOTtsCausalLm>>,
    std::unique_ptr<CudaEngineState<Gemma4CausalLm>>,
    std::unique_ptr<CudaEngineState<GlmDsaCausalLm>>,
    std::unique_ptr<CudaEngineState<Glm5CausalLm>>,
    std::unique_ptr<CudaEngineState<Qwen4CausalLm>>,
    std::unique_ptr<CudaEngineState<DeepseekV4CausalLm>>,
    std::unique_ptr<CudaEngineState<DeepseekV41CausalLm>>,
    std::unique_ptr<CudaEngineState<NemotronHCausalLm>>>;

struct CudaBackend {
    using Options = CudaEngineOptions;
    Options options;
    State state;
    template <class F> decltype(auto) visit(F&& run) {
        return std::visit([&](auto& state) -> decltype(auto) {
            if (!state) throw std::runtime_error("CUDA engine is unloaded");
            MfqCudaGuard guard(state->execution->layer_placement.primary_device());
            mfq_tensor_backend::NoGradGuard no_grad;
            return run(*state);
        }, state);
    }
    std::pair<EngineInfo, std::unique_ptr<TextProcessor>> load() {
        if (!options.context_size) options.context_size = 32768;
        auto execution = std::make_shared<CudaExecutionContext>();
        setup_cuda_load(options, *execution);
        execution->profiler.enabled = false;
        auto weight_memory=std::make_shared<WeightMemoryLedger>();
        const WeightMemoryLoadScope memory_load(*weight_memory);
        mfq_tensor_backend::NoGradGuard no_grad;
        auto config = resolve_cuda_runtime_config(options);
        const auto kv_options = runtime_options::qsa_kv_offload();
        execution->kv_memory=std::make_shared<KvMemoryManager>();
        if (kv_options.gpu_budget_bytes) {
            execution->kv_memory.reset();
            KvOffloadConfig kv;
            kv.gpu_budget_bytes = kv_options.gpu_budget_bytes;
            kv.ram_budget_bytes = cuda_has_separate_memory() ? kv_options.ram_budget_bytes : 0;
            kv.buffer_bytes = std::min<std::size_t>(kv.buffer_bytes, kv.gpu_budget_bytes / 4);
            kv.directory = kv_options.directory;
            execution->qsa_kv_store = std::make_shared<KvOffloadStore>(std::move(kv));
        }
        state = with_loaded_cuda_model(*execution, options, true,
            [&](auto& model, auto& components, auto, auto) -> State {
                using Model = std::decay_t<decltype(model)>;
                if constexpr (!std::is_same_v<Model, Qwen4CausalLm>)
                    if (execution->qsa_kv_store) throw std::invalid_argument("CUDA streamed KV requires a QSA model");
                return std::make_unique<CudaEngineState<Model>>(execution, std::move(model), std::move(components), config,weight_memory);
            });
        return visit([&](auto& state) {
            EngineInfo info;
            info.batch_capacity = state.batch_capacity();
            info.max_requests = info.batch_capacity;
            info.max_context = state.language.max_position_embeddings();
            info.vocab_size = static_cast<int32_t>(state.language.vocab_size());
            info.model_type = state.language.model_type();
            info.multimodal = state.components.grid_vision.has_value() ||
                state.components.deepseek_vision.has_value() ||
                state.components.radio_vision.has_value() || bool(state.components.composite);
            info.reload = true;
            const auto caps = cuda_runtime_capabilities(state.components.graph, state.components.plan,
                state.components.state(), info.multimodal, bool(state.components.composite));
            info.duplex = caps.full_duplex;
            info.capabilities = {state.components.graph.architecture, caps.text,
                caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
                caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
            auto text = TextProcessor::load(*state.language.source, options.tokenizer_model,
                info.vocab_size, info.model_type);
            info.stop_token_ids = model_stop_token_ids(
                load_model_config_json(*state.language.source, options.config_path), info.vocab_size);
            return std::pair{std::move(info), std::move(text)};
        });
    }
    bool exclusive() const {
        return std::visit([](const auto& state) { return state && state->exclusive(); }, state);
    }
    void unload() {
        state = std::unique_ptr<CudaEngineState<Qwen35CausalLm>>{};
    }
};
} // namespace engine_detail

std::unique_ptr<Engine> load_cuda_engine(CudaEngineOptions options) {
    return std::make_unique<EngineInstance<engine_detail::CudaBackend>>(std::move(options));
}
} // namespace mfq::cuda
