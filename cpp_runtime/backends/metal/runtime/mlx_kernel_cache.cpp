#include "mlx_kernel_prepare.h"

#include <array>
#include <CommonCrypto/CommonDigest.h>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <mach-o/dyld.h>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>

namespace mfq::metal {
namespace {

template <typename T> struct Cache {
    std::shared_mutex mutex;
    std::unordered_map<std::string, std::shared_future<NS::SharedPtr<T>>> entries;
};

template <typename T, typename Build>
T* cached(Cache<T>& cache, const std::string& key, Build build) {
    std::shared_future<NS::SharedPtr<T>> future;
    {
        std::shared_lock lock(cache.mutex);
        if (const auto found = cache.entries.find(key); found != cache.entries.end())
            future = found->second;
    }
    if (future.valid()) return future.get().get();
    std::promise<NS::SharedPtr<T>> promise;
    {
        std::unique_lock lock(cache.mutex);
        const auto [entry, inserted] = cache.entries.try_emplace(key, promise.get_future().share());
        future = entry->second;
        if (!inserted) {
            lock.unlock();
            return future.get().get();
        }
    }
    try { promise.set_value(build()); }
    catch (...) {
        promise.set_exception(std::current_exception());
        std::unique_lock lock(cache.mutex);
        cache.entries.erase(key);
    }
    return future.get().get();
}

MTL::LanguageVersion language_version() {
    if (__builtin_available(macOS 27, iOS 27, tvOS 27, visionOS 27, *))
        return static_cast<MTL::LanguageVersion>((4 << 16) + 1);
    if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *))
        return MTL::LanguageVersion4_0;
    if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *))
        return MTL::LanguageVersion3_2;
    return MTL::LanguageVersion3_1;
}

MTL::Library* precompiled_library(MTL::Device* device, const std::string& source, int mode) {
    const auto fingerprint = source + '\0' + std::to_string(mode) + '\0'
        + std::to_string(static_cast<unsigned>(language_version()));
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(fingerprint.data(), static_cast<CC_LONG>(fingerprint.size()), digest);
    std::string filename;
    for (const auto byte : digest) {
        filename += "0123456789abcdef"[byte >> 4];
        filename += "0123456789abcdef"[byte & 15];
    }
    filename += ".metallib";
    static const auto roots = [] {
        std::vector<std::filesystem::path> paths;
        if (const auto* value = std::getenv("MFQ_METAL_LIBRARY_DIR")) paths.emplace_back(value);
        std::uint32_t size = 0;
        _NSGetExecutablePath(nullptr, &size);
        std::string executable(size, '\0');
        if (_NSGetExecutablePath(executable.data(), &size) == 0) {
            const auto parent = std::filesystem::weakly_canonical(executable.c_str()).parent_path();
            paths.push_back(parent / "lib" / "mfq-metal");
            paths.push_back(parent / ".." / "Resources" / "mfq-metal");
        }
        return paths;
    }();
    for (const auto& root : roots) {
        const auto path = root / filename;
        if (!std::filesystem::is_regular_file(path)) continue;
        NS::Error* error = nullptr;
        auto* library = device->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding), &error);
        if (!library) throw std::runtime_error("Cannot load bundled Metal library: " + path.string()
            + ": " + (error ? error->localizedDescription()->utf8String() : "invalid library"));
        return library;
    }
    return nullptr;
}

}

MlxKernelDevice& kernel_device(mlx::core::Device device) {
    static auto* cache = new MlxKernelDevice(mlx::core::metal::device(device));
    return *cache;
}

MTL::Library* MlxKernelDevice::get_library(const std::string& name,
    const mlx::core::CompileOptions& options, const std::function<std::string()>& source) {
    const auto mode = options.serialize();
    if (mode < 0 || mode > 2) throw std::invalid_argument("invalid Metal math mode");
    static auto* libraries = new std::array<Cache<MTL::Library>, 3>;
    return cached(libraries->at(mode), name, [&] {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        const auto text = source();
        if (auto* library = precompiled_library(device_.mtl_device(), text, mode))
            return NS::TransferPtr(library);
        auto config = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
        config->setLanguageVersion(language_version());
        if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
            const MTL::MathMode modes[]{MTL::MathModeSafe, MTL::MathModeRelaxed, MTL::MathModeFast};
            config->setMathMode(modes[mode]);
        } else {
            if (options.math_mode == mlx::core::MathMode::Relaxed)
                throw std::runtime_error("relaxed Metal math requires macOS 15");
            config->setFastMathEnabled(options.math_mode == mlx::core::MathMode::Fast);
        }
#ifndef NDEBUG
        if (config->languageVersion() >= MTL::LanguageVersion3_2) config->setEnableLogging(true);
#endif
        NS::Error* error = nullptr;
        auto library = NS::TransferPtr(device_.mtl_device()->newLibrary(
            NS::String::string(text.c_str(), NS::UTF8StringEncoding), config.get(), &error));
        if (!library) throw std::runtime_error("Metal library " + name + ": "
            + (error ? error->localizedDescription()->utf8String() : "compilation failed"));
        return library;
    });
}

MTL::ComputePipelineState* MlxKernelDevice::get_kernel(const std::string& name, MTL::Library* library) {
    return get_kernel(name, library, name, {});
}

MTL::ComputePipelineState* MlxKernelDevice::get_kernel(const std::string& name, MTL::Library* library,
    const std::string& specialization, const mlx::core::metal::MTLFCList& constants) {
    static auto* mutex = new std::mutex;
    static auto* pipelines = new std::unordered_map<MTL::Library*, std::unique_ptr<Cache<MTL::ComputePipelineState>>>;
    Cache<MTL::ComputePipelineState>* cache;
    {
        std::lock_guard lock(*mutex);
        auto& entry = (*pipelines)[library];
        if (!entry) entry = std::make_unique<Cache<MTL::ComputePipelineState>>();
        cache = entry.get();
    }
    const auto& key = specialization.empty() ? name : specialization;
    return cached(*cache, key, [&] {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        NS::SharedPtr<MTL::Function> function;
        NS::Error* error = nullptr;
        if (constants.empty() && key == name) {
            function = NS::TransferPtr(library->newFunction(
                NS::String::string(name.c_str(), NS::UTF8StringEncoding)));
        } else {
            auto values = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
            for (const auto& [value, type, index] : constants)
                values->setConstantValue(value, type, index);
            auto descriptor = NS::TransferPtr(MTL::FunctionDescriptor::alloc()->init());
            descriptor->setName(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
            descriptor->setSpecializedName(NS::String::string(key.c_str(), NS::UTF8StringEncoding));
            descriptor->setConstantValues(values.get());
            function = NS::TransferPtr(library->newFunction(descriptor.get(), &error));
        }
        if (!function) throw std::runtime_error("Metal function not found: " + name
            + (error ? ": " + std::string(error->localizedDescription()->utf8String()) : ""));
        error = nullptr;
        auto pipeline = NS::TransferPtr(device_.mtl_device()->newComputePipelineState(function.get(), &error));
        if (!pipeline) throw std::runtime_error("Metal pipeline " + name + ": "
            + (error ? error->localizedDescription()->utf8String() : "compilation failed"));
        return pipeline;
    });
}

} // namespace mfq::metal
