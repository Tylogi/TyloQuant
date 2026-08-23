#include "hf_safetensors_store.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

class AlignedBuffer {
public:
    explicit AlignedBuffer(std::size_t size) : size_(size) {
        void* pointer = nullptr;
        const auto result = ::posix_memalign(&pointer, 4096, size);
        if (result != 0) {
            throw std::bad_alloc();
        }
        data_.reset(static_cast<std::byte*>(pointer));
    }

    std::span<std::byte> span() noexcept {
        return {data_.get(), size_};
    }

private:
    struct Free {
        void operator()(std::byte* pointer) const noexcept {
            std::free(pointer);
        }
    };

    std::size_t size_ = 0;
    std::unique_ptr<std::byte, Free> data_;
};

struct Options {
    std::string model;
    std::size_t layer = 0;
    std::size_t workers = 8;
    std::size_t iterations = 3;
    std::size_t experts = 256;
    bool random = false;
};

std::size_t parse_size(const char* text, const char* option) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed);
        if (consumed != std::string(text).size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        throw std::runtime_error(
            std::string("invalid value for ") + option + ": " + text);
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&](const char* option) -> const char* {
            if (index + 1 >= argc) {
                throw std::runtime_error(
                    std::string("missing value for ") + option);
            }
            return argv[++index];
        };
        if (argument == "--model") {
            options.model = value("--model");
        } else if (argument == "--layer") {
            options.layer = parse_size(value("--layer"), "--layer");
        } else if (argument == "--workers") {
            options.workers = parse_size(value("--workers"), "--workers");
        } else if (argument == "--iterations") {
            options.iterations = parse_size(
                value("--iterations"), "--iterations");
        } else if (argument == "--experts") {
            options.experts = parse_size(value("--experts"), "--experts");
        } else if (argument == "--random") {
            options.random = true;
        } else if (argument == "--help") {
            std::cout
                << "Usage: mfq-metal-ssd-expert-benchmark --model DIR "
                   "[--layer N] [--workers N] [--iterations N] "
                   "[--experts N] [--random]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.model.empty()) {
        throw std::runtime_error("--model is required");
    }
    if (options.workers == 0 || options.iterations == 0 ||
        options.experts == 0 || options.experts > 256) {
        throw std::runtime_error(
            "workers/iterations must be positive and experts must be 1..256");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto index_begin = std::chrono::steady_clock::now();
        mfq::metal::DeepseekV4NativeExpertStore store(
            options.model,
            43,
            256);
        const auto index_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - index_begin).count();
        if (options.layer >= store.num_layers()) {
            throw std::runtime_error("layer is outside the checkpoint");
        }

        std::cout << "model=" << store.checkpoint().root() << '\n'
                  << "shards=" << store.checkpoint().shard_count()
                  << " tensors=" << store.checkpoint().tensor_count()
                  << " index_seconds=" << std::fixed << std::setprecision(3)
                  << index_seconds << '\n'
                  << "layer=" << options.layer
                  << " experts=" << options.experts
                  << " workers=" << options.workers
                  << " slot_mib="
                  << (static_cast<double>(store.slot_bytes()) / (1 << 20))
                  << " pattern=" << (options.random ? "random" : "sequential")
                  << '\n';

        std::vector<double> throughputs;
        throughputs.reserve(options.iterations);
        for (std::size_t iteration = 0;
             iteration < options.iterations;
             ++iteration) {
            std::vector<std::size_t> expert_ids(
                options.random ? store.num_experts() : options.experts);
            std::iota(expert_ids.begin(), expert_ids.end(), 0);
            if (options.random) {
                std::mt19937_64 generator(0x4d465100ULL + iteration);
                std::shuffle(expert_ids.begin(), expert_ids.end(), generator);
                expert_ids.resize(options.experts);
            }
            store.checkpoint().drop_file_cache();
            std::atomic<std::size_t> next{0};
            std::atomic<std::uint64_t> bytes{0};
            std::atomic<std::uint64_t> calls{0};
            std::exception_ptr failure;
            std::mutex failure_mutex;
            std::vector<std::thread> threads;
            threads.reserve(options.workers);
            const auto begin = std::chrono::steady_clock::now();
            for (std::size_t worker = 0;
                 worker < options.workers;
                 ++worker) {
                threads.emplace_back([&] {
                    try {
                        AlignedBuffer buffer(store.slot_bytes());
                        while (true) {
                            const auto item = next.fetch_add(1);
                            if (item >= expert_ids.size()) {
                                break;
                            }
                            const auto stats = store.load(
                                options.layer,
                                expert_ids[item],
                                buffer.span());
                            bytes.fetch_add(stats.bytes);
                            calls.fetch_add(stats.read_calls);
                        }
                    } catch (...) {
                        std::scoped_lock lock(failure_mutex);
                        if (!failure) {
                            failure = std::current_exception();
                        }
                    }
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }
            if (failure) {
                std::rethrow_exception(failure);
            }
            const auto seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count();
            const auto gbs = static_cast<double>(bytes.load()) / seconds / 1e9;
            throughputs.push_back(gbs);
            std::cout << "run=" << (iteration + 1)
                      << " seconds=" << std::setprecision(4) << seconds
                      << " bytes=" << bytes.load()
                      << " read_calls=" << calls.load()
                      << " GB/s=" << std::setprecision(3) << gbs << '\n';
        }
        const auto mean = std::accumulate(
            throughputs.begin(), throughputs.end(), 0.0) /
            static_cast<double>(throughputs.size());
        const auto [minimum, maximum] = std::minmax_element(
            throughputs.begin(), throughputs.end());
        std::cout << "mean_GB/s=" << std::setprecision(3) << mean
                  << " range_GB/s=" << *minimum << ".." << *maximum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mfq-metal-ssd-expert-benchmark: " << error.what() << '\n';
        return 1;
    }
}
