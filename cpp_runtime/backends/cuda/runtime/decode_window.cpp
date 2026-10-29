#include "decode_window.h"
#include "execution_options.h"
#include "mfq_cuda_context.h"
#include "mfq/runtime.h"
#include "mfq/cpu_expert_pool.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <exception>
#include <iostream>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace mfq::cuda {
namespace {
thread_local cudaStream_t flushed_window_stream=nullptr;
thread_local bool has_flushed_window=false;
struct WindowSubmission {
    cudaStream_t previous=flushed_window_stream;
    bool previously_flushed=has_flushed_window;
    ~WindowSubmission(){flushed_window_stream=previous;has_flushed_window=previously_flushed;}
};
void report_window_error(const char* operation,const char* phase) noexcept {
    try {
        std::rethrow_exception(std::current_exception());
    } catch(const std::exception& error) {
        std::fprintf(stderr,"decode_window_error operation=%s phase=%s error=%s\n",operation,phase,error.what());
    } catch(...) {
        std::fprintf(stderr,"decode_window_error operation=%s phase=%s error=unknown\n",operation,phase);
    }
    std::fflush(stderr);
}
}
void check_stream_progress(cudaStream_t stream) {
    const auto status=cudaStreamQuery(stream);
    if(status!=cudaSuccess && status!=cudaErrorNotReady)MFQ_NATIVE_CUDA_CHECK(status);
}
// Adapted from Strata's session_run_token (src/core/session.cpp).
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// MIT license: ../../../third_party/strata.LICENSE
// One WDDM flush submits the complete token graph. Layer waits keep their
// bounded fallback polling; standalone publications still flush immediately.
void check_route_submission(cudaStream_t stream) {
    if(!has_flushed_window || stream!=flushed_window_stream)check_stream_progress(stream);
}
void publish_mapped_flag(uint32_t* flag) noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *static_cast<volatile uint32_t*>(flag)=1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
void wait_route_publication(const uint32_t* flag,cudaStream_t stream,int timeout_ms) {
    using Clock=std::chrono::steady_clock;
    const auto start=Clock::now();auto flush=start;
    const auto poll_interval=std::chrono::microseconds(runtime_options::route_poll_interval_us());
    const auto* option=std::getenv("MFQ_MOE_ROUTE_SPIN");
    const bool spin=!option || option[0]!='0';
    uint32_t spins=0;
    if(*static_cast<const volatile uint32_t*>(flag)<1)check_route_submission(stream);
    while(*static_cast<const volatile uint32_t*>(flag)<1) {
        if(spin) {
            // Strata checks its mapped doorbell after pause and consults the
            // clock every 1024 spins (src/core/session.cpp, MIT license above).
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            _mm_pause();
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
            if((++spins&1023u)!=0)continue;
        }
        const auto now=Clock::now();
        if(now-flush>=poll_interval) {
            flush=now;const auto q=cudaStreamQuery(stream);
            if(q!=cudaSuccess && q!=cudaErrorNotReady)MFQ_NATIVE_CUDA_CHECK(q);
            if(q==cudaSuccess && *static_cast<const volatile uint32_t*>(flag)<1)
                throw std::runtime_error("MFQ graph finished before publishing its next layer");
        }
        if(now-start>std::chrono::milliseconds(timeout_ms))
            throw std::runtime_error("MFQ graph route publication stalled");
        if(!spin)std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}
namespace { thread_local DecodeWindow* active=nullptr; }
struct DecodeWindow::LayerPoint {
    int layer;
    int tile;
    std::string phase;
    cudaEvent_t event=nullptr;
    uint64_t samples=0;
    int64_t total_ns=0,min_ns=0,max_ns=0;
    LayerPoint(int l,int t,const char* p,bool clock):layer(l),tile(t),phase(p) {
        if(!clock)MFQ_NATIVE_CUDA_CHECK(cudaEventCreate(&event));
    }
    ~LayerPoint(){if(event)cudaEventDestroy(event);}
};
DecodeWindow* DecodeWindow::recording() noexcept { return active; }
struct DecodeWindow::ProfileClock {
    static constexpr std::size_t capacity=32768;
    uint64_t* device=nullptr;
    std::vector<uint64_t> host=std::vector<uint64_t>(capacity);
    ProfileClock(){MFQ_NATIVE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device),capacity*sizeof(uint64_t)));}
    ~ProfileClock(){if(device)(void)cudaFree(device);}
};
DecodeWindow::DecodeWindow(cudaStream_t stream,bool layer_profile,bool independent_sequences)
    :stream_(stream),graphs_(stream),layer_profile_(layer_profile),independent_sequences_(independent_sequences) {
    // External CUDA timing events split Windows graph submissions. Device
    // timestamps preserve the graph's submission and use the same clock as
    // Strata's verifier profiler. This storage exists only when tracing.
    const auto* option=std::getenv("MFQ_TRACE_LAYER_TIMINGS_CLOCK");
    if(layer_profile_ && (!option || option[0]!='0'))profile_clock_=std::make_unique<ProfileClock>();
}
void DecodeWindow::mark(int layer,const char* phase) {
    profile_layer_=layer;
    if(!layer_profile_)return;
    // Scope 2 keeps the MFE phase boundaries. Timing every small operation
    // measurably perturbs Windows graph replay; ordinary execution adds none.
    if(const auto* scope=std::getenv("MFQ_TRACE_LAYER_TIMINGS");scope &&
            (std::strcmp(scope,"2")==0 || std::strcmp(scope,"3")==0)) {
        constexpr const char* phases[]={"window_begin","window_end","hot_begin","hot_end",
            "dma_end","transfer_gate_up_end","cpu_wait_begin","transfer_end"};
        // Scope 3 separates the attention/residual work, route publication and
        // input preparation (including the mapped host-plan wait). Scope 2
        // retains its existing event count for historical comparisons.
        constexpr const char* boundaries[]={"attention_core_begin","attention_core_end",
            "ffn_pre_end","expert_begin","shared_end","ple_begin","ple_end","gate_up_end","down_end"};
        if(std::none_of(std::begin(phases),std::end(phases),
                [&](const char* allowed){return std::strcmp(phase,allowed)==0;}) &&
            (std::strcmp(scope,"3")!=0 || std::none_of(std::begin(boundaries),std::end(boundaries),
                [&](const char* allowed){return std::strcmp(phase,allowed)==0;})))return;
    }
    if(active!=this)throw std::logic_error("MFQ layer timing outside window recording");
    const auto key=std::make_tuple(layer,profile_tile_,std::string(phase));
    auto found=layer_point_index_.find(key);
    if(found==layer_point_index_.end()) {
        if(profile_clock_ && layer_points_.size()>=ProfileClock::capacity)
            throw std::runtime_error("GPU stage profile exceeds timestamp capacity");
        auto point=std::make_unique<LayerPoint>(layer,profile_tile_,phase,bool(profile_clock_));
        found=layer_point_index_.emplace(key,point.get()).first;layer_points_.push_back(std::move(point));
    }
    if(profile_clock_) {
        const auto index=std::find_if(layer_points_.begin(),layer_points_.end(),
            [&](const auto& point){return point.get()==found->second;})-layer_points_.begin();
        record_gpu_timestamp(profile_clock_->device,index,stream_);return;
    }
    cudaStreamCaptureStatus capture;
    MFQ_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream_,&capture));
    MFQ_NATIVE_CUDA_CHECK(cudaEventRecordWithFlags(found->second->event,stream_,
        capture==cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
}
void DecodeWindow::observe_layers() {
    if(profile_clock_ && !layer_points_.empty()) {
        MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(profile_clock_->host.data(),profile_clock_->device,
            layer_points_.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream_));
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));
        // A bad diagnostic sample must not cancel an otherwise completed
        // decode, or partly increment the phase totals for this replay.
        for(std::size_t i=1;i<layer_points_.size();++i) {
            const auto before=profile_clock_->host[i-1],after=profile_clock_->host[i];
            if(!before || after<before) {
                ++dropped_profiles_;
                std::cerr<<"native_layer_gpu_invalid role="<<profile_role_<<" rows="<<profile_rows_
                    <<" index="<<i<<" from_layer="<<layer_points_[i-1]->layer<<" layer="<<layer_points_[i]->layer
                    <<" from="<<layer_points_[i-1]->phase<<" to="<<layer_points_[i]->phase
                    <<" before="<<before<<" after="<<after<<std::endl;
                return;
            }
        }
    }
    for(std::size_t i=1;i<layer_points_.size();++i) {
        auto& current=*layer_points_[i];float ms=0;
        int64_t ns;
        if(profile_clock_) {
            const auto before=profile_clock_->host[i-1],after=profile_clock_->host[i];
            ns=static_cast<int64_t>(after-before);
        } else {
            MFQ_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,layer_points_[i-1]->event,current.event));
            ns=static_cast<int64_t>(ms*1e6);
        }
        current.total_ns+=ns;current.min_ns=current.samples ? std::min(current.min_ns,ns) : ns;
        current.max_ns=std::max(current.max_ns,ns);++current.samples;
    }
}
std::vector<DecodeWindow::GpuPhase> DecodeWindow::gpu_phases() const {
    std::vector<GpuPhase> phases;
    for(std::size_t i=1;i<layer_points_.size();++i) {
        const auto& previous=*layer_points_[i-1];const auto& current=*layer_points_[i];
        phases.push_back({previous.layer,current.layer,previous.phase,current.phase,
            current.samples,current.total_ns,current.min_ns,current.max_ns,previous.tile,current.tile});
    }
    return phases;
}
DecodeWindow::~DecodeWindow() {
    // Mapped flags and task owners must survive the last consumer, including
    // a graph released by a failed CPU or I/O phase.
    for(auto& task:tasks_)if(task.cancel)task.cancel();
    (void)cudaStreamSynchronize(stream_);
    graphs_.clear();
    for(auto& task:tasks_)if(task.cleanup)task.cleanup();
    if(replays_ && std::getenv("MFQ_TRACE_NATIVE_CUDA_GRAPH"))
        std::cerr<<"native_cuda_window tasks="<<tasks_.size()<<" replays="<<replays_
            <<" capture_ns="<<capture_ns_<<" replay_ns="<<replay_ns_
            <<" reset_ns="<<reset_ns_<<" launch_ns="<<launch_ns_<<" serve_ns="<<serve_ns_
            <<" wait_ns="<<wait_ns_<<" finish_ns="<<finish_ns_<<" profile_ns="<<profile_ns_
            <<" role="<<profile_role_<<" rows="<<profile_rows_
            <<" clock="<<(profile_clock_?1:0)<<" profile_dropped="<<dropped_profiles_<<std::endl;
    for(const auto& phase:gpu_phases())if(phase.samples)
        std::cerr<<"native_layer_gpu from_layer="<<phase.from_layer<<" layer="<<phase.layer
            <<" from="<<phase.from<<" to="<<phase.to<<" samples="<<phase.samples
            <<" total_ns="<<phase.total_ns<<" min_ns="<<phase.min_ns<<" max_ns="<<phase.max_ns
            <<" from_tile="<<phase.from_tile<<" tile="<<phase.tile
            <<" kind="<<(profile_kinds_.count(phase.layer)?profile_kinds_.at(phase.layer):"unknown")<<std::endl;
}
void DecodeWindow::enroll(Task task) {
    if(active!=this)throw std::logic_error("MFQ task outside window capture");
    if(std::any_of(tasks_.begin(),tasks_.end(),[&](const auto& t){return t.identity==task.identity;}))
        throw std::logic_error("one MFQ operator cell was recorded twice in a window");
    tasks_.push_back(std::move(task));
}
std::size_t DecodeWindow::nodes() const {
    std::size_t result=0;
    for(std::size_t i=0;i<segments();++i)if(const auto* graph=graphs_.find(i,1))result+=graph->nodes();
    return result;
}
void DecodeWindow::set_kernel_rows(std::size_t segment,cudaGraphNode_t node,unsigned rows) {
    if(!valid())throw std::logic_error("graph kernel update requires a captured window");
    const auto* graph=graphs_.find(segment,1);
    if(!graph)throw std::out_of_range("graph kernel update has no matching segment");
    graph->set_kernel_rows(node,rows);
}
void DecodeWindow::host_barrier(std::function<void()> serve) {
    if(active!=this)throw std::logic_error("host I/O barrier outside window capture");
    if(recording_segments_)graphs_.end(barriers_.size(),1);
    barriers_.push_back({tasks_.size(),std::move(serve)});
    if(recording_segments_)graphs_.begin(barriers_.size(),1);
}
void DecodeWindow::capture(const std::function<void()>& body,const std::function<void()>& restore,
        const std::function<void()>& release_output) {
    if(valid())return;
    const HostRegistrationBlock registration;
    if(active)throw std::logic_error("nested MFQ window capture");
    struct Recording {
        explicit Recording(DecodeWindow* window){active=window;}
        ~Recording(){active=nullptr;}
    } recording(this);
    const auto capture_started=std::chrono::steady_clock::now();
    const auto* trace_option=std::getenv("MFQ_TRACE_MOE_SERVE");
    const bool trace=trace_option && trace_option[0] && trace_option[0]!='0';
    const char* phase="prepare";
    const auto checkpoint=[&](const char* next) {
        phase=next;
        if(trace) {
            std::fprintf(stderr,"decode_window operation=capture phase=%s tasks=%zu\n",phase,tasks_.size());
            std::fflush(stderr);
        }
    };
    const auto release_warm_outputs=[&] {
        if(release_output)release_output();
        for(auto& task:tasks_)if(task.release_warm_output)task.release_warm_output();
    };
    // Preparation allocates every persistent cell and warms CUDA workspaces.
    // Warmup/capture may overwrite recurrent state, so both restore it.
    try {
        checkpoint("prepare");
        body();MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));restore();
        release_warm_outputs();
        tasks_.clear();barriers_.clear();
        checkpoint("warmup");
        // Some first-use operator outputs remain cached after the first pass.
        // Warm their steady allocation lifetimes before forbidding new storage
        // in capture, including a previously unused output projection.
        for(int pass=0;pass<2;++pass) {
            {int device=0;MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
                GraphWarmupScope warmup(default_context(device),stream_);body();}
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));restore();
            release_warm_outputs();
            if(!pass){tasks_.clear();barriers_.clear();}
        }
        const auto warm_tasks=tasks_.size(),warm_barriers=barriers_.size();tasks_.clear();barriers_.clear();
        checkpoint("record");
        try {
            recording_segments_=true;graphs_.begin(0,1);body();graphs_.end(barriers_.size(),1);
            recording_segments_=false;
        } catch(const std::exception& failure) {
            recording_segments_=false;graphs_.clear();
            const std::string error=failure.what();
            // Preserve quota errors so admission can discard this request
            // without marking the entire backend unhealthy.
            if(error.starts_with("CUDA process VRAM limit exceeded:") ||
                    error.starts_with("CUDA VRAM limit exceeded:") ||
                    error.starts_with("CUDA allocation failed:") ||
                    error.starts_with("CUDA pinned allocation failed:"))throw MfqResourceExhausted(error);
            throw std::runtime_error(error);
        }
        if(tasks_.size()!=warm_tasks || barriers_.size()!=warm_barriers)throw std::logic_error("MFQ window task order changed during capture");
        checkpoint("restore");
        restore();MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));
        capture_ns_=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-capture_started).count();
        checkpoint("complete");
    } catch(...) {
        recording_segments_=false;graphs_.clear();
        report_window_error("capture",phase);
        bool allocation_rejected=false;
        try {std::rethrow_exception(std::current_exception());}
        catch(const MfqResourceExhausted&) {allocation_rejected=true;}
        catch(const Error& error) {
            const std::string message=error.what();
            allocation_rejected=message.starts_with("CUDA process VRAM limit exceeded:") ||
                message.starts_with("CUDA VRAM limit exceeded:") ||
                message.starts_with("CUDA pinned allocation failed:");
        } catch(...) {}
        for(auto& task:tasks_)if(task.cancel)task.cancel();
        (void)cudaStreamSynchronize(stream_);
        if(allocation_rejected)(void)cudaGetLastError();
        restore();failed_=true;throw;
    }
}
void DecodeWindow::run() {
    if(failed_ || !valid())throw std::runtime_error("MFQ window is unavailable after a failed execution");
    const HostRegistrationBlock registration;
    const mfq::cpu::ScopedHostAffinity placement(runtime_options::decode_host_pin()
        ?mfq::cpu::planned_host_core():-1);
    const auto* trace_option=std::getenv("MFQ_TRACE_MOE_SERVE");
    const bool trace=trace_option && trace_option[0] && trace_option[0]!='0' && (replays_<2 || replays_%64==0);
    // Value 2 adds one summary per replay, including later slow tokens that
    // the sampled verbose trace would otherwise omit. Ordinary runs add no I/O.
    const bool trace_steps=trace_option && trace_option[0]=='2';
    const char* phase="reset";
    const auto checkpoint=[&](const char* next) {
        phase=next;
        if(trace) {
            std::fprintf(stderr,"decode_window operation=replay replay=%llu phase=%s tasks=%zu\n",
                static_cast<unsigned long long>(replays_),phase,tasks_.size());
            std::fflush(stderr);
        }
    };
    try {
        WindowSubmission submission;
        using Clock=std::chrono::steady_clock;
        const auto reset_before=reset_ns_,launch_before=launch_ns_,serve_before=serve_ns_,wait_before=wait_ns_,finish_before=finish_ns_;
        auto previous=Clock::now();const auto started=previous;
        const auto sample=[&](int64_t& total) {
            const auto now=Clock::now();
            total+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-previous).count();previous=now;
        };
        checkpoint("reset");
        for(auto& task:tasks_)task.reset();
        sample(reset_ns_);
        std::string error;
        std::size_t task_begin=0;
        for(std::size_t segment=0;segment<segments();++segment) {
            const auto* graph=graphs_.find(segment,1);
            checkpoint("launch");
            if(!graph->launch(stream_,error))throw std::runtime_error(error);
            checkpoint("submit");
            if(runtime_options::single_window_flush()) {
                check_stream_progress(stream_);
                flushed_window_stream=stream_;has_flushed_window=true;
            }
            sample(launch_ns_);
            checkpoint("serve");
            const auto task_end=segment<barriers_.size()?barriers_[segment].task_end:tasks_.size();
            for(;task_begin<task_end;++task_begin)tasks_[task_begin].serve();
            sample(serve_ns_);
            checkpoint("wait");
            if(!graph->wait_ms(60000))throw std::runtime_error("MFQ model window did not finish");
            sample(wait_ns_);
            if(segment<barriers_.size()) {
                checkpoint("host_io");barriers_[segment].serve();sample(serve_ns_);
            }
        }
        // Only after all GPU consumers finish may adaptation overwrite L1.
        checkpoint("finish");
        for(auto& task:tasks_)if(task.finish)task.finish();
        sample(finish_ns_);
        observe_layers();sample(profile_ns_);
        replay_ns_+=std::chrono::duration_cast<std::chrono::nanoseconds>(previous-started).count();++replays_;
        const auto elapsed=std::chrono::duration<double,std::milli>(previous-started).count();
        if(trace_steps) {
            double gpu_ms=0;
            if(profile_clock_ && !layer_points_.empty()) {
                const auto first=profile_clock_->host.front();
                const auto last=profile_clock_->host[layer_points_.size()-1];
                if(first && last>=first)gpu_ms=double(last-first)/1e6;
            }
            std::fprintf(stderr,"runtime_decode_window_step role=%s rows=%d replay=%llu ms=%.6f reset_ms=%.6f "
                "launch_ms=%.6f serve_ms=%.6f wait_ms=%.6f finish_ms=%.6f gpu_ms=%.6f\n",
                profile_role_.c_str(),profile_rows_,static_cast<unsigned long long>(replays_),elapsed,
                (reset_ns_-reset_before)/1e6,(launch_ns_-launch_before)/1e6,(serve_ns_-serve_before)/1e6,
                (wait_ns_-wait_before)/1e6,(finish_ns_-finish_before)/1e6,gpu_ms);
        }
        if(trace && replays_<=64 && elapsed>=45.0)
            std::cerr<<"runtime_decode_window_slow role="<<profile_role_<<" rows="<<profile_rows_
                <<" replay="<<replays_<<" ms="<<elapsed<<" reset_ms="<<(reset_ns_-reset_before)/1e6
                <<" launch_ms="<<(launch_ns_-launch_before)/1e6<<" serve_ms="<<(serve_ns_-serve_before)/1e6
                <<" wait_ms="<<(wait_ns_-wait_before)/1e6<<" finish_ms="<<(finish_ns_-finish_before)/1e6<<'\n';
        checkpoint("complete");
    } catch(...) {
        report_window_error("replay",phase);
        failed_=true;
        for(auto& task:tasks_)if(task.cancel)task.cancel();
        (void)cudaStreamSynchronize(stream_);throw;
    }
}
}
