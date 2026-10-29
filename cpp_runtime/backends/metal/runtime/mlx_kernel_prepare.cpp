#include "mlx_kernel_prepare.h"
#include "mfq_mlx_utils_embedded.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <mlx/backend/metal/device.h>
#include <mlx/fast_primitives.h>
#include <mlx/primitives.h>
#include <mlx/version.h>

namespace mfq::metal {
namespace {

mlx::core::array preparation_view_input(const mlx::core::array& input, std::string& key, int depth = 0) {
    key += '[';
    for (const int dimension : input.shape()) key += dimension == 1 ? '1' : 'n';
    key += ']';
    if (depth < 8 && input.has_primitive() && input.inputs().size() == 1 &&
        input.inputs().front().nbytes() <= (4u << 20)) {
        const auto primitive = input.primitive_ptr();
        const std::string_view name = primitive->name();
        if (name == "Transpose" || name == "Reshape" || name == "Slice" || name == "Broadcast") {
            key += name;
            if (name == "Transpose")
                for (const int axis : static_cast<const mlx::core::Transpose*>(primitive.get())->state())
                    key += ':' + std::to_string(axis);
            if (name == "Slice")
                for (const int stride : std::get<2>(static_cast<const mlx::core::Slice*>(primitive.get())->state()))
                    key += ':' + std::to_string(stride);
            auto source = preparation_view_input(input.inputs().front(), key, depth + 1);
            return mlx::core::array(input.shape(), input.dtype(), primitive, {std::move(source)});
        }
    }
    return mlx::core::zeros(input.shape(), input.dtype(), input.has_primitive()
        ? input.primitive().stream() : mlx::core::default_stream(mlx::core::Device::gpu));
}

}

std::string mlx_kernel_preparation_key(const mlx::core::Primitive& primitive) {
    if (const auto* native = dynamic_cast<const MlxPreparableKernel*>(&primitive))
        return native->preparation_key();
    if (std::string_view(primitive.name()) != "CustomKernel")
        throw std::invalid_argument("Metal kernel does not support preparation");
    const auto state = static_cast<const mlx::core::fast::CustomKernel&>(primitive).state();
    std::ostringstream key;
    key << std::get<0>(state);
#if MLX_VERSION_NUMERIC >= 32001
    key << '_' << std::hex << std::hash<std::string>{}(std::get<1>(state))
        << '_' << std::dec << std::get<10>(state);
#endif
    return key.str();
}

void prepare_mlx_kernel_gpu(mlx::core::Primitive& primitive) {
    if (auto* native = dynamic_cast<MlxPreparableKernel*>(&primitive)) {
        native->prepare_gpu();
        return;
    }
    const auto key = mlx_kernel_preparation_key(primitive);
    const auto state = static_cast<mlx::core::fast::CustomKernel&>(primitive).state();
    auto& device = mlx::core::metal::device(primitive.stream().device);
    auto* library = device.get_library(key, mlx::core::CompileOptions(std::get<10>(state)),
        [&] { return std::string(detail::kMlxUtilsSource) + std::get<1>(state); });
    (void)device.get_kernel(std::get<0>(state), library);
}

thread_local MlxKernelPreparation* MlxKernelPreparation::active_ = nullptr;

MlxKernelPreparation::MlxKernelPreparation(int context, int chunk_size)
    : previous_(active_), context_(context) {
    if (context < 1 || chunk_size < 1)
        throw std::invalid_argument("invalid Metal kernel preparation geometry");
    for (const int requested : {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 25, 31, 32, 33, 64, 65, 96, 128, chunk_size}) {
        const int rows = std::min(context, requested);
        if (std::find(rows_.begin(), rows_.end(), rows) == rows_.end()) rows_.push_back(rows);
    }
    active_ = this;
}

MlxKernelPreparation::~MlxKernelPreparation() { active_ = previous_; }

MlxKernelPreparation* MlxKernelPreparation::current() noexcept {
    return active_ && !active_->collecting_ ? active_ : nullptr;
}

bool MlxKernelPreparation::collecting() noexcept {
    return active_ && active_->collecting_;
}

void MlxKernelPreparation::collect(const std::function<void()>& build) {
    const bool previous = collecting_;
    collecting_ = true;
    try { build(); }
    catch (...) { collecting_ = previous; throw; }
    collecting_ = previous;
}

void MlxKernelPreparation::add(const mlx::core::array& output) {
    std::unordered_set<const mlx::core::Primitive*> visited;
    const auto visit = [&](const auto& self, const mlx::core::array& value) -> void {
        if (!value.has_primitive()) return;
        const auto primitive = value.primitive_ptr();
        if (!visited.insert(primitive.get()).second) return;
        if (dynamic_cast<MlxPreparableKernel*>(primitive.get()) || std::string_view(primitive->name()) == "CustomKernel") {
            if (keys_.insert("kernel:" + mlx_kernel_preparation_key(*primitive)).second)
                jobs_.push_back([primitive] { prepare_mlx_kernel_gpu(*primitive); });
        } else if (const auto* compiled = dynamic_cast<mlx::core::Compiled*>(primitive.get());
            compiled || std::string_view(primitive->name()) == "ScatterAxis" ||
                std::string_view(primitive->name()) == "Gather") {
            const auto outputs = value.outputs();
            const auto& inputs = value.inputs();
            std::vector<mlx::core::Shape> input_shapes, output_shapes;
            std::vector<mlx::core::Dtype> input_dtypes, output_dtypes;
            for (const auto& input : inputs) { input_shapes.push_back(input.shape()); input_dtypes.push_back(input.dtype()); }
            for (const auto& output : outputs) { output_shapes.push_back(output.shape()); output_dtypes.push_back(output.dtype()); }
            if (std::string_view(primitive->name()) == "ScatterAxis") {
                const int axis = static_cast<mlx::core::ScatterAxis*>(primitive.get())->state().second;
                for (auto& shape : input_shapes) shape.at(axis) = std::min(shape.at(axis), 8);
                for (auto& shape : output_shapes) shape.at(axis) = std::min(shape.at(axis), 8);
            }
            const auto bounded = [](const auto& shapes, const auto& dtypes) {
                std::size_t bytes = 0;
                for (std::size_t index = 0; index < shapes.size(); ++index) {
                    std::size_t size = dtypes[index].size();
                    for (const int dimension : shapes[index]) {
                        if (dimension && size > (4u << 20) / dimension) return false;
                        size *= dimension;
                    }
                    if (size > (4u << 20) - bytes) return false;
                    bytes += size;
                }
                return true;
            };
            std::string key = compiled ? "compiled:" + compiled->lib_name() : primitive->name();
            if (!compiled)
                for (const auto& input : inputs) key += ':' + std::to_string(static_cast<int>(input.dtype().val()));
            std::vector<mlx::core::array> view_inputs;
            if (compiled && bounded(input_shapes, input_dtypes))
                for (const auto& input : inputs) view_inputs.push_back(preparation_view_input(input, key));
            if (bounded(input_shapes, input_dtypes) && bounded(output_shapes, output_dtypes) && keys_.insert(key).second) {
                compiled_jobs_.push_back([primitive, input_shapes, input_dtypes, output_shapes, output_dtypes,
                        inputs = std::move(view_inputs)]() mutable {
                    if (inputs.empty()) for (std::size_t index = 0; index < input_shapes.size(); ++index)
                        inputs.push_back(mlx::core::zeros(input_shapes[index], input_dtypes[index], primitive->stream()));
                    auto outputs = mlx::core::array::make_arrays(output_shapes, output_dtypes, primitive, inputs);
                    mlx::core::eval(outputs);
                });
            }
        } else {
            const std::string name = primitive->name();
            if (name.find("Primitive") != std::string::npos) unsupported_.insert(name);
        }
        for (const auto& input : value.inputs()) self(self, input);
    };
    visit(visit, output);
}

void MlxKernelPreparation::collect_sampling(int vocab, const MlxSamplingParams& defaults) {
    collect([&] {
        for (const auto dtype : {mlx::core::float16, mlx::core::bfloat16, mlx::core::float32}) {
            for (int rows = 1; rows <= 6; ++rows) {
                const auto logits = mlx::core::zeros({rows, vocab}, dtype);
                const std::vector<float> random_values(rows, 0.5f);
                add(sample_greedy(logits));
                for (const auto& random : {mlx::core::zeros({rows}, mlx::core::float32),
                        mlx::core::array(random_values.begin(), mlx::core::Shape{rows})}) {
                    add(sample_softmax(logits, random));
                    add(sample(logits, defaults, random));
                    if (defaults.top_k > 0 && defaults.top_k <= 128)
                        add(sample_top_k_distribution(logits, random, 1.0, std::min(defaults.top_k, vocab)).sampled);
                }
                const auto counts = mlx::core::zeros({vocab}, mlx::core::int32);
                add(sample_apply_penalties(logits, counts, 0.1, 0.1, 1.05));
                for (const int count : {1, 8}) {
                    add(sample_token_counts_add(counts, mlx::core::zeros({count}, mlx::core::int32)));
                    const std::vector<std::int32_t> ids(count, 0);
                    add(sample_token_counts_add(counts, mlx::core::array(ids.begin(), mlx::core::Shape{count})));
                }
            }
        }
    });
}

void MlxKernelPreparation::finish() {
    struct Restore {
        MlxKernelPreparation*& active;
        MlxKernelPreparation* previous;
        ~Restore() { active = previous; }
    } restore{active_, previous_};
    const auto started = std::chrono::steady_clock::now();
    const auto parallel_total = jobs_.size();
    const auto total = parallel_total + compiled_jobs_.size();
    std::size_t completed = 0;
    const auto maximum_workers = std::max(1u, std::thread::hardware_concurrency());
    std::size_t workers = maximum_workers;
    if (const auto* value = std::getenv("MFQ_METAL_COMPILE_WORKERS")) {
        char* end = nullptr;
        const auto requested = std::strtoul(value, &end, 10);
        if (!*value || *end || requested < 1 || requested > maximum_workers)
            throw std::invalid_argument("MFQ_METAL_COMPILE_WORKERS must be in [1, "
                + std::to_string(maximum_workers) + "]");
        workers = requested;
    }
    workers = std::min(workers, std::max<std::size_t>(1, parallel_total));
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex progress;
    std::exception_ptr failure;
    const auto run = [&] {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        while (!failed.load()) {
            const auto index = next.fetch_add(1);
            if (index >= parallel_total) break;
            try { jobs_[index](); }
            catch (...) {
                std::lock_guard lock(progress);
                if (!failure) failure = std::current_exception();
                failed.store(true);
                break;
            }
            std::lock_guard lock(progress);
            ++completed;
            std::cerr << "mfq_load_progress stage=compiling completed=" << completed
                      << " total=" << total << std::endl;
        }
    };
    {
        std::vector<std::thread> threads;
        struct Join {
            std::vector<std::thread>& threads;
            ~Join() { for (auto& thread : threads) if (thread.joinable()) thread.join(); }
        } join{threads};
        try {
            for (std::size_t i = 1; i < workers; ++i) threads.emplace_back(run);
            run();
        } catch (...) {
            failed.store(true);
            throw;
        }
    }
    active_ = previous_;
    if (failure) { jobs_.clear(); compiled_jobs_.clear(); std::rethrow_exception(failure); }
    try {
        for (const auto& job : compiled_jobs_) {
            job();
            ++completed;
            std::cerr << "mfq_load_progress stage=compiling completed=" << completed
                      << " total=" << total << std::endl;
        }
    } catch (...) { jobs_.clear(); compiled_jobs_.clear(); throw; }
    std::cerr << "mfq_load_progress stage=compiling completed=" << completed
              << " total=" << total << std::endl;
    std::cerr << "mfq_kernel_prepare kernels=" << total << " workers=" << workers << " unsupported=" << unsupported_.size()
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
              << std::endl;
    jobs_.clear();
    compiled_jobs_.clear();
    active_ = previous_;
}

} // namespace mfq::metal
