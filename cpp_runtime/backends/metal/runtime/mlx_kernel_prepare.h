#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>
#include "mlx_sampling.h"

namespace mfq::metal {

class MlxKernelDevice {
public:
    explicit MlxKernelDevice(mlx::core::metal::Device& device) : device_(device) {}
    MTL::Library* get_library(const std::string& name,
        const mlx::core::CompileOptions& options, const std::function<std::string()>& source);
    MTL::Library* get_library(const std::string& name, const std::function<std::string()>& source) {
        return get_library(name, {}, source);
    }
    MTL::ComputePipelineState* get_kernel(const std::string& name, MTL::Library* library);
    MTL::ComputePipelineState* get_kernel(const std::string& name, MTL::Library* library,
        const std::string& key, const mlx::core::metal::MTLFCList& constants);
    MTL::ComputePipelineState* get_kernel(const std::string& name) { return device_.get_kernel(name); }
    const std::string& get_architecture() const { return device_.get_architecture(); }
    int get_architecture_gen() const { return device_.get_architecture_gen(); }
private:
    mlx::core::metal::Device& device_;
};

MlxKernelDevice& kernel_device(mlx::core::Device device);

class MlxPreparableKernel {
public:
    virtual ~MlxPreparableKernel() = default;
    virtual std::string preparation_key() const = 0;
    virtual void prepare_gpu() = 0;
};

std::string mlx_kernel_preparation_key(const mlx::core::Primitive& primitive);
void prepare_mlx_kernel_gpu(mlx::core::Primitive& primitive);

class MlxKernelPreparation {
public:
    MlxKernelPreparation(int context, int chunk_size);
    ~MlxKernelPreparation();
    MlxKernelPreparation(const MlxKernelPreparation&) = delete;
    MlxKernelPreparation& operator=(const MlxKernelPreparation&) = delete;
    static MlxKernelPreparation* current() noexcept;
    static bool collecting() noexcept;
    void collect(const std::function<void()>& build);
    void add(const mlx::core::array& output);
    void finish();
    void collect_sampling(int vocab, const MlxSamplingParams& defaults);
    void use_model_paths() noexcept { model_paths_ = true; }
    bool model_paths() const noexcept { return model_paths_; }
    const std::vector<int>& row_buckets() const noexcept { return rows_; }
    int context() const noexcept { return context_; }
    void set_routes(int routes) noexcept { routes_ = routes; }
    int routes() const noexcept { return routes_; }
    void set_activation_dtype(mlx::core::Dtype dtype) noexcept { activation_dtype_ = dtype; }
    mlx::core::Dtype activation_dtype() const noexcept { return activation_dtype_; }
    std::size_t size() const noexcept { return jobs_.size() + compiled_jobs_.size(); }
    const std::unordered_set<std::string>& unsupported() const noexcept { return unsupported_; }

private:
    static thread_local MlxKernelPreparation* active_;
    MlxKernelPreparation* previous_;
    bool collecting_ = false;
    bool model_paths_ = false;
    int routes_ = 0;
    int context_ = 0;
    mlx::core::Dtype activation_dtype_ = mlx::core::float16;
    std::vector<int> rows_;
    std::unordered_set<std::string> keys_;
    std::unordered_set<std::string> unsupported_;
    std::vector<std::function<void()>> jobs_;
    std::vector<std::function<void()>> compiled_jobs_;
};

} // namespace mfq::metal
