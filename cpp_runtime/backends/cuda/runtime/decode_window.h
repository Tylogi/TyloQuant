#pragma once
#include "graph_registry.h"
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace mfq::cuda {
void wait_route_publication(const uint32_t* flag,cudaStream_t stream,int timeout_ms=60000);
void publish_mapped_flag(uint32_t* flag) noexcept;
void check_stream_progress(cudaStream_t stream);
void check_route_submission(cudaStream_t stream);
// Resident KV uses one model graph. Streamed KV adds boundaries at host I/O;
// expert operators within each segment keep their GPU body and host tasks in
// one submission. Disk reads run after a segment completes, without GPU spins.
class DecodeWindow {
public:
    struct Task {
        const void* identity;
        std::function<void()> reset, serve, finish, cancel;
        std::function<void()> cleanup;
        std::function<void()> release_warm_output;
    };
    struct GpuPhase {
        int from_layer,layer;
        std::string from,to;
        uint64_t samples;
        int64_t total_ns,min_ns,max_ns;
        int from_tile=-1,tile=-1;
    };
    explicit DecodeWindow(cudaStream_t stream,bool layer_profile=false,bool independent_sequences=false);
    bool independent_sequences() const noexcept {return independent_sequences_;}
    ~DecodeWindow();
    DecodeWindow(const DecodeWindow&)=delete;
    DecodeWindow& operator=(const DecodeWindow&)=delete;
    void capture(const std::function<void()>& body,
                 const std::function<void()>& restore,
                 const std::function<void()>& release_output = {});
    void enroll(Task task);
    // Finish the current GPU segment before running host I/O. During graph
    // preparation only the supplied placeholder buffers are consumed.
    void host_barrier(std::function<void()> serve);
    // Update the next segment at a completed host barrier, before its launch.
    // Zero disables a kernel while preserving the captured dependency edges.
    void set_kernel_rows(std::size_t segment,cudaGraphNode_t node,unsigned rows);
    template<class T,class Factory> std::shared_ptr<T> workspace(const void* identity,Factory create) {
        auto& owner=workspaces_[identity];if(!owner)owner=create();return std::static_pointer_cast<T>(owner);
    }
    void run();
    void mark(int layer,const char* phase);
    void mark_current(const char* phase) {mark(profile_layer_,phase);}
    int set_profile_tile(int tile) noexcept {const int previous=profile_tile_;profile_tile_=tile;return previous;}
    void set_profile_geometry(const char* role,int rows) {profile_role_=role;profile_rows_=rows;}
    void set_profile_layer_kind(int layer,const char* kind) {if(layer_profile_)profile_kinds_[layer]=kind;}
    std::vector<GpuPhase> gpu_phases() const;
    std::size_t nodes() const;
    std::size_t segments() const {return barriers_.size()+1;}
    bool valid() const { const auto* graph=graphs_.find(0,1);return graph && graph->valid() && !recording_segments_; }
    static DecodeWindow* recording() noexcept;
    bool prediction_enabled() const noexcept {return prediction_enabled_;}
    void set_prediction_enabled(bool enabled) noexcept {prediction_enabled_=enabled;}
private:
    cudaStream_t stream_;
    GraphRegistry graphs_;
    struct LayerPoint;
    struct ProfileClock;
    std::unique_ptr<ProfileClock> profile_clock_;
    std::vector<std::unique_ptr<LayerPoint>> layer_points_;
    std::map<std::tuple<int,int,std::string>,LayerPoint*> layer_point_index_;
    int profile_tile_=-1;
    int profile_layer_=-1;
    bool layer_profile_;
    bool independent_sequences_;
    bool prediction_enabled_=true;
    std::string profile_role_="unknown";
    int profile_rows_=1;
    std::map<int,std::string> profile_kinds_;
    void observe_layers();
    std::vector<Task> tasks_;
    std::map<const void*,std::shared_ptr<void>> workspaces_;
    struct Barrier {std::size_t task_end;std::function<void()> serve;};
    std::vector<Barrier> barriers_;
    bool recording_segments_=false;
    bool failed_=false;
    int64_t capture_ns_=0,replay_ns_=0,reset_ns_=0,launch_ns_=0,serve_ns_=0,wait_ns_=0,finish_ns_=0,profile_ns_=0;
    uint64_t replays_=0;
    uint64_t dropped_profiles_=0;
};
}
