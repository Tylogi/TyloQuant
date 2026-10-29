#include "mlx_cached_kernel.h"
#include "mlx_kernel_prepare.h"
#include "mfq_mlx_utils_embedded.h"

#include <mlx/backend/metal/utils.h>
#include <mlx/fast_primitives.h>
#include <mlx/utils.h>
#include <mlx/version.h>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>

namespace mfq::metal {
namespace {
class RowMajorView final : public mlx::core::Primitive {
public:
    RowMajorView(const mlx::core::array& input, mlx::core::Stream stream)
        : Primitive(stream), copy_(mlx::core::contiguous(input, false, stream).primitive_ptr()) {}
    const char* name() const override { return "MlxRowMajorView"; }
    void eval_cpu(const std::vector<mlx::core::array>& inputs, std::vector<mlx::core::array>& outputs) override {
        if (inputs.front().flags().row_contiguous) outputs.front().copy_shared_buffer(inputs.front());
        else copy_->eval_cpu(inputs, outputs);
    }
    void eval_gpu(const std::vector<mlx::core::array>& inputs, std::vector<mlx::core::array>& outputs) override {
        if (inputs.front().flags().row_contiguous) outputs.front().copy_shared_buffer(inputs.front());
        else copy_->eval_gpu(inputs, outputs);
    }
private:
    std::shared_ptr<mlx::core::Primitive> copy_;
};
}

mlx::core::array mlx_row_contiguous_view(const mlx::core::array& input, mlx::core::Stream stream) {
    return mlx::core::array(input.shape(), input.dtype(),
        std::make_shared<RowMajorView>(input, stream), {input});
}

#if MLX_VERSION_NUMERIC >= 32001
namespace {
using namespace mlx::core;

using Arguments = std::vector<std::pair<std::string, fast::TemplateArg>>;
using Geometry = std::tuple<int, int, int>;

unsigned binding(const array& value) {
    return value.ndim() == 0 ? 0 : value.size() < 8 ? 1 : 2;
}

struct Description {
    Device device;
    std::vector<Dtype> inputs, outputs;
    std::vector<unsigned> bindings;
    Arguments arguments;
    std::string name, source, library_name;
    std::vector<std::tuple<bool, bool, bool>> metadata;
    bool contiguous;
    CompileOptions::Data options;
    std::mutex mutex;
    NS::SharedPtr<MTL::ComputePipelineState> owner;
    std::atomic<MTL::ComputePipelineState*> pipeline{nullptr};

    Description(Device device, const std::vector<array>& values,
        const std::vector<Dtype>& outputs, Arguments arguments,
        const fast::CustomKernel& prototype)
        : device(device), outputs(outputs), arguments(std::move(arguments)) {
        for (const auto& value : values) {
            inputs.push_back(value.dtype());
            bindings.push_back(binding(value));
        }
        auto state = prototype.state();
        name = std::move(std::get<0>(state));
        source = std::move(std::get<1>(state));
        metadata = std::move(std::get<4>(state));
        contiguous = std::get<5>(state);
        options = std::get<10>(state);
        std::ostringstream key;
        key << name << '_' << std::hex << std::hash<std::string>{}(source)
            << '_' << std::dec << options;
        library_name = key.str();
    }

    bool matches(Device target, const std::vector<array>& values,
        const std::vector<Dtype>& types, const Arguments& args) const {
        if (target != device || outputs != types || arguments != args) return false;
        for (std::size_t i = 0; i < values.size(); ++i)
            if (inputs[i] != values[i].dtype() || bindings[i] != binding(values[i])) return false;
        return true;
    }

    MTL::ComputePipelineState* prepare() {
        if (auto* ready = pipeline.load(std::memory_order_acquire)) return ready;
        std::lock_guard lock(mutex);
        if (auto* ready = pipeline.load(std::memory_order_relaxed)) return ready;
        auto& backend = mlx::core::metal::device(device);
        auto* library = backend.get_library(library_name, CompileOptions(options),
            [&] { return std::string(detail::kMlxUtilsSource) + source; });
        owner = NS::RetainPtr(backend.get_kernel(name, library));
        pipeline.store(owner.get(), std::memory_order_release);
        return owner.get();
    }
};

class CachedKernel final : public Primitive, public MlxPreparableKernel {
public:
    CachedKernel(Stream stream, std::shared_ptr<Description> description,
        Geometry grid, Geometry group, std::optional<float> initial)
        : Primitive(stream), description_(std::move(description)), grid_(grid),
            group_(group), initial_(initial) {
        if (initial_) initializer_ = full({1}, 0, float32, stream).primitive_ptr();
    }

    const char* name() const override { return "MlxCachedKernel"; }
    std::string preparation_key() const override {
        return description_->library_name + ':' + std::to_string(description_->device.index);
    }
    void prepare_gpu() override { (void)description_->prepare(); }
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("Metal kernels require a GPU stream");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto* pipeline = description_->prepare();
        std::vector<array> temporaries;
        for (auto& output : outputs) {
            if (initial_) {
                temporaries.emplace_back(*initial_, output.dtype());
                std::vector<array> filled{output};
                initializer_->eval_gpu({temporaries.back()}, filled);
                output = std::move(filled.front());
            } else output.set_data(allocator::malloc(output.nbytes()));
        }
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(pipeline);
        int slot = 0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& value = inputs[i];
            encoder.set_input_array(value, slot++);
            if (value.ndim() > 0) {
                const auto [shape, strides, dimensions] = description_->metadata[i];
                if (shape) encoder.set_vector_bytes(value.shape(), value.ndim(), slot++);
                if (strides) encoder.set_vector_bytes(value.strides(), value.ndim(), slot++);
                if (dimensions) encoder.set_bytes(value.ndim(), slot++);
            }
        }
        for (auto& output : outputs) encoder.set_output_array(output, slot++);
        const auto [tx, ty, tz] = group_;
        if (std::size_t(tx) * ty * tz > pipeline->maxTotalThreadsPerThreadgroup())
            throw std::invalid_argument("Metal threadgroup exceeds this kernel's thread limit");
        const auto [gx, gy, gz] = grid_;
        encoder.dispatch_threads(MTL::Size(gx, gy, gz),
            MTL::Size(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz)));
        encoder.add_temporaries(std::move(temporaries));
    }
private:
    std::shared_ptr<Description> description_;
    std::shared_ptr<Primitive> initializer_;
    Geometry grid_, group_;
    std::optional<float> initial_;
};

struct Factory {
    fast::CustomKernelFunction original;
    std::mutex mutex;
    std::vector<std::shared_ptr<Description>> descriptions;
};
}
#endif

mlx::core::fast::CustomKernelFunction cached_metal_kernel(
    const std::string& name, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const std::string& source,
    const std::string& header, bool contiguous, bool atomic,
    const mlx::core::CompileOptions& options) {
    auto original = mlx::core::fast::metal_kernel(name, inputs, outputs, source,
        header, contiguous, atomic, options);
#if MLX_VERSION_NUMERIC < 32001
    return original;
#else
    auto factory = std::make_shared<Factory>();
    factory->original = std::move(original);
    return [factory, input_count = inputs.size(), output_count = outputs.size()](
        const std::vector<array>& values, const std::vector<Shape>& shapes,
        const std::vector<Dtype>& types, Geometry grid, Geometry group,
        Arguments arguments, std::optional<float> initial, bool verbose, StreamOrDevice target) {
        if (values.size() != input_count || shapes.size() != output_count || types.size() != output_count)
            throw std::invalid_argument("Metal kernel input/output counts do not match its signature");
        const auto stream = std::holds_alternative<std::monostate>(target)
            ? default_stream(Device::gpu) : to_stream(target);
        if (stream.device != Device::gpu)
            throw std::invalid_argument("Metal kernels require a GPU stream");
        std::shared_ptr<Description> description;
        {
            std::lock_guard lock(factory->mutex);
            for (const auto& item : factory->descriptions) {
                if (item->matches(stream.device, values, types, arguments)) {
                    description = item;
                    break;
                }
            }
            if (!description) {
                auto prototype = factory->original(values, shapes, types, grid, group,
                    arguments, initial, false, stream);
                description = std::make_shared<Description>(stream.device, values, types,
                    std::move(arguments), static_cast<const fast::CustomKernel&>(prototype.front().primitive()));
                factory->descriptions.push_back(description);
            }
        }
        if (verbose) std::cout << description->source << '\n';
        std::vector<array> checked = values;
        if (description->contiguous)
            for (auto& value : checked)
                if (!value.is_available() || !value.flags().row_contiguous)
                    value = mlx_row_contiguous_view(value, stream);
        return array::make_arrays(shapes, types,
            std::make_shared<CachedKernel>(stream, std::move(description), grid, group, initial), std::move(checked));
    };
#endif
}

}
