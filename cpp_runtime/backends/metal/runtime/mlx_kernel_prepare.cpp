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
#include <mlx/version.h>

namespace mfq::metal {

thread_local MlxKernelPreparation* MlxKernelPreparation::active_ = nullptr;

MlxKernelPreparation::MlxKernelPreparation(int context, int chunk_size)
    : previous_(active_), context_(context) {
    if (context < 1 || chunk_size < 1)
        throw std::invalid_argument("invalid Metal kernel preparation geometry");
    for (const int requested : {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 25, 31, 32, 33, 64, 96, 128, chunk_size}) {
        const int rows = std::min(context, requested);
        if (std::find(rows_.begin(), rows_.end(), rows) == rows_.end()) rows_.push_back(rows);
    }
    active_ = this;
}

MlxKernelPreparation::~MlxKernelPreparation() { active_ = previous_; }

MlxKernelPreparation* MlxKernelPreparation::current() noexcept {
    return active_ && !active_->collecting_ ? active_ : nullptr;
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
        if (auto* native = dynamic_cast<MlxPreparableKernel*>(primitive.get())) {
            if (keys_.insert("native:" + native->preparation_key()).second)
                jobs_.push_back([primitive] {
                    dynamic_cast<MlxPreparableKernel&>(*primitive).prepare_gpu();
                });
        } else if (std::string_view(primitive->name()) == "CustomKernel") {
            const auto* custom = static_cast<mlx::core::fast::CustomKernel*>(primitive.get());
            const auto state = custom->state();
            const auto& name = std::get<0>(state);
            std::ostringstream library_key;
            library_key << name;
#if MLX_VERSION_NUMERIC >= 32001
            library_key << '_' << std::hex << std::hash<std::string>{}(std::get<1>(state))
                        << '_' << std::dec << std::get<10>(state);
#endif
            const auto library_name = library_key.str();
            if (keys_.insert("custom:" + library_name).second)
                jobs_.push_back([primitive, library_name] {
                    const auto state = static_cast<mlx::core::fast::CustomKernel&>(*primitive).state();
                    auto& device = mlx::core::metal::device(primitive->stream().device);
                    const auto& name = std::get<0>(state);
                    auto* library = device.get_library(library_name, mlx::core::CompileOptions(std::get<10>(state)),
                        [&] { return std::string(detail::kMlxUtilsSource) + std::get<1>(state); });
                    (void)device.get_kernel(name, library);
                });
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
    const auto total = jobs_.size();
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
    workers = std::min(workers, total);
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex progress;
    std::exception_ptr failure;
    const auto run = [&] {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        while (!failed.load()) {
            const auto index = next.fetch_add(1);
            if (index >= total) break;
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
    if (failure) { jobs_.clear(); std::rethrow_exception(failure); }
    std::cerr << "mfq_load_progress stage=compiling completed=" << completed
              << " total=" << total << std::endl;
    std::cerr << "mfq_kernel_prepare kernels=" << total << " workers=" << workers << " unsupported=" << unsupported_.size()
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
              << std::endl;
    jobs_.clear();
    active_ = previous_;
}

} // namespace mfq::metal
