#include "runtime/batch_decode_graph.h"
#include "mfq_cuda_kv_hot_cache.h"
#include "batch_selected_attention.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace mfq::cuda;
namespace tb = mfq_tensor_backend;

static void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

static void check_exclusive_tiers(bool independent) {
    const auto options = tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32);
    const auto integers = options.dtype(tb::kInt64);
    BatchDecodeGraphCache graphs;
    MfqCudaStreamGuard guard(graphs.stream());
    std::vector<HotKvCache> caches;
    for (int request = 0; request < 4; ++request) {
        auto cache = HotKvCache::allocate(2, 8, request % 2 ? 5 : 3, 4, 6, options);
        const int cold = 8 - int(cache.keys.size(2));
        cache.set_cold_storage(tb::zeros({1, 2, cold, 4}, options), tb::zeros({1, 2, cold, 6}, options));
        auto keys = tb::full({1, 2, 6, 4}, 10 + request, options);
        auto values = tb::full({1, 2, 6, 6}, 110 + request, options);
        cache.append(keys, values, tb::arange(6, integers));
        caches.push_back(std::move(cache));
    }
    auto& graph = graphs.get({4, 8, 8, {}}, 4 + HotKvStateFields::Count, 0);
    auto& inputs = *graph.inputs;
    std::vector<tb::Tensor> outputs;
    for (const auto members : {std::vector<int>{0, 1, 2}, std::vector<int>{3, 2, 0}}) {
        const auto position = graphs.has_graphs() ? 7 : 6;
        inputs.update_rows({{20, position, members[0]}, {21, position, members[1]}, {22, position, members[2]}});
        int inactive = 0;
        while (std::find(members.begin(), members.end(), inactive) != members.end())
            ++inactive;
        auto& disabled = caches[inactive];
        const std::vector<tb::Tensor> disabled_state{disabled.keys, disabled.values, disabled.cold_keys,
            disabled.cold_values, disabled.mapping, disabled.cold_mapping, disabled.slot_ids,
            disabled.epochs, disabled.free_cold, disabled.free_gpu, disabled.cold_control, disabled.control};
        std::vector<tb::Tensor> before;
        for (const auto& tensor : disabled_state)
            before.push_back(tensor.cpu());
        for (int row = 0; row < 4; ++row) {
            auto& cache = caches[row < 3 ? members[row] : inactive];
            // The gather's logical extent is eight; each physical tier is smaller.
            inputs.state(row, 0, cache.mapping);
            inputs.state(row, 1, cache.mapping);
            cache.bind_decode_state(inputs, 4, row);
        }
        inputs.upload();
        const auto body = [&] {
            outputs.clear();
            if (independent) {
                auto keys = (tb::ones({4, 1, 2, 4}, options) *
                    inputs.ids().to(tb::kInt32).reshape({4, 1, 1, 1})).contiguous();
                auto values = (tb::ones({4, 1, 2, 6}, options) *
                    (inputs.ids().to(tb::kInt32) + 100).reshape({4, 1, 1, 1})).contiguous();
                batch_attention_store(inputs, 0, keys, values);
            }
            for (int row = 0; row < 4; ++row) {
                auto value = inputs.ids().narrow(0, row, 1).to(tb::kInt32).reshape({1, 1, 1, 1});
                auto keys = (tb::ones({1, 2, 1, 4}, options) * value).contiguous();
                auto values = (tb::ones({1, 2, 1, 6}, options) * (value + 100)).contiguous();
                auto positions = inputs.positions().narrow(0, row, 1);
                check(independent || caches.front().append_decode(
                    keys, values, positions, {}, {}, false, &inputs, 4, row),
                      "exclusive indirect append was rejected");
                outputs.push_back(batch_state_gather(inputs, 0, row, tb::arange(8, integers), 2, 4, options, 4));
                outputs.push_back(batch_state_gather(inputs, 1, row, tb::arange(8, integers), 2, 6, options, 4, true));
            }
        };
        if (!graph.valid) {
            graphs.prepare_pool({}, {});
            graph.window = std::make_unique<DecodeWindow>(graphs.stream().stream());
            graph.window->capture(body, [] {}, [&] { outputs.clear(); });
            graphs.captured(graph, 0);
        }
        graph.window->run();
        graphs.replayed(graph);
        for (std::size_t index = 0; index < disabled_state.size(); ++index) {
            auto after = disabled_state[index].cpu();
            check(std::memcmp(after.data_ptr(), before[index].data_ptr(), after.nbytes()) == 0,
                  "disabled row modified exclusive KV state");
        }
        for (int row = 0; row < 4; ++row) {
            for (int value = 0; value < 2; ++value) {
                auto actual = outputs[2 * row + value].cpu();
                for (int head = 0; head < 2; ++head) {
                    for (int token = 0; token < 8; ++token) {
                        int expected = 0;
                        if (row < 3) {
                            const auto request = members[row];
                            if (token < 6)
                                expected = 10 + request + value * 100;
                            else if (token == position)
                                expected = 20 + row + value * 100;
                            else if (token == 6 && position == 7 && request != 3)
                                expected = (request == 0 ? 20 : 22) + value * 100;
                        }
                        const int width = value ? 6 : 4;
                        for (int column = 0; column < width; ++column)
                            check(actual.data_ptr<int32_t>()[(head * 8 + token) * width + column] == expected,
                                  "exclusive gather read the wrong physical row");
                    }
                }
            }
        }
    }
    check(graphs.captures == 1 && graphs.bucket_reuses == 1, "exclusive tiers captured membership changes");
    for (const auto& cache : caches) {
        auto hot = cache.mapping.cpu(), cold = cache.cold_mapping.cpu();
        int hot_rows = 0, cold_rows = 0;
        for (int token = 0; token < 8; ++token) {
            hot_rows += hot.data_ptr<int32_t>()[token] >= 0;
            cold_rows += cold.data_ptr<int32_t>()[token] >= 0;
            check(hot.data_ptr<int32_t>()[token] < 0 || cold.data_ptr<int32_t>()[token] < 0,
                  "logical row occupies both KV tiers");
        }
        check(hot_rows == cache.keys.size(2) && cold_rows >= 1, "exclusive rows did not occupy both physical tiers");
    }
    std::cout << "batch_decode_exclusive_tiers PASS eviction=1 cold_reads=1 reorder=1 padding=1 captures=1"
              << " independent=" << independent << '\n';
}

int main() {
    try {
        int devices = 0;
        MFQ_CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices)
            return 77;
        check_exclusive_tiers(false);
        check_exclusive_tiers(true);
        for (int maximum : {1, 2, 3, 6, 16, 32}) {
            for (int rows = 1; rows <= maximum; ++rows) {
                const auto bucket = batch_decode_row_bucket(rows, maximum);
                check(bucket >= rows && bucket <= maximum, "invalid decode bucket ceiling");
                check(bucket == maximum || (bucket & (bucket - 1)) == 0, "invalid decode bucket shape");
            }
        }
        BatchDecodeGraphCache cache;
        MfqCudaStreamGuard guard(cache.stream());
        const auto options = tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
        std::vector<tb::Tensor> states, counts, snapshots;
        std::vector<int> expected(6, 0);
        for (int request = 0; request < 6; ++request) {
            states.push_back(tb::full({1, 64}, request, options));
            counts.push_back(tb::zeros({1}, options.dtype(tb::kInt32)));
        }
        ParallelConfig parallel;
        ModelParallelCollectiveRuntime collectives;
        cache.prepare_pool(parallel, collectives);
        const std::vector<std::vector<int>> schedule{
            {0, 1, 2}, {2, 0, 1, 3}, {3, 2, 0}, {4, 0, 2, 3}, {2, 4, 0},
            {0, 1}, {1}, {5, 4, 3}, {3, 5, 4, 0}};
        std::size_t largest_graph_reserved = 0;
        for (const auto& members : schedule) {
            const auto rows = batch_decode_row_bucket(members.size(), 6);
            auto& entry = cache.get({rows, 512, 256, {}}, 2, 0);
            auto& input = *entry.inputs;
            snapshots.clear();
            for (const auto& state : states)
                snapshots.push_back(state.cpu());
            for (int64_t row = 0; row < rows; ++row) {
                const bool active = row < int64_t(members.size());
                input.row(row, 1, 0, active ? members[row] : -1, active);
                input.state(row, 0, active ? states[members[row]] : states[5]);
                input.state(row, 1, active ? counts[members[row]] : counts[5]);
            }
            input.upload();
            const auto body = [&] {
                // Reuse a sizable temporary across every bucket capture.
                auto scratch = tb::empty({4, 1024 * 1024}, options);
                scratch.zero_();
                for (int64_t row = 0; row < rows; ++row) {
                    auto state = tb::empty_like(states[0]);
                    auto count = tb::empty_like(counts[0]);
                    batch_state_load(input, 0, row, state);
                    batch_state_load(input, 1, row, count);
                    state = state + 1;
                    count = count + 1;
                    batch_state_store(input, 0, row, state);
                    batch_state_store(input, 1, row, count);
                }
            };
            const auto restore = [&] {
                for (int request = 0; request < 6; ++request) {
                    states[request].copy_(snapshots[request]);
                    counts[request].fill_(expected[request]);
                }
            };
            const auto captures = cache.captures;
            if (!entry.valid) {
                entry.window = std::make_unique<DecodeWindow>(cache.stream().stream());
                entry.window->capture(body, restore);
                cache.captured(entry, 0);
            }
            entry.window->run();
            for (const auto member : members)
                ++expected[member];
            cache.replayed(entry);
            if (rows == 4 && captures)
                check(cache.captures == captures, "membership change captured a graph");
            for (int request = 0; request < 6; ++request) {
                const auto value = states[request].cpu();
                check(counts[request].item<int32_t>() == expected[request], "inactive token count changed");
                check(value.data_ptr<float>()[0] == request + expected[request], "request state changed incorrectly");
                if (std::find(members.begin(), members.end(), request) == members.end())
                    check(std::memcmp(value.data_ptr(), snapshots[request].data_ptr(), value.nbytes()) == 0,
                          "removed request state was modified");
            }
            if (!largest_graph_reserved)
                largest_graph_reserved = default_context()->pool_memory_stats().second;
        }
        check(cache.captures == 3 && cache.membership_captures == 0 && cache.bucket_reuses == 6,
              "decode graph reuse counters are incorrect");
        const auto buckets_reserved = default_context()->pool_memory_stats().second;
        check(buckets_reserved <= largest_graph_reserved + (std::size_t(2) << 20),
              "bucket graphs retained separate scratch pools");
        auto& input = *cache.get({4, 512, 256, {}}, 2, 0).inputs;
        input.update_rows({{1, 0, 0}, {1, 0, 1}, {1, 0, 2}});
        input.upload();
        CudaExecutionContext execution;
        std::vector<tb::Tensor> selected{
            tb::zeros({4, 2}, options.dtype(tb::kInt32)), tb::ones({4, 2}, options)};
        BatchDecodeMaskScope mask(execution, input.enabled());
        mask_batch_routes(execution, selected);
        auto routes = selected[0].cpu();
        check(routes.data_ptr<int32_t>()[6] == -1 && routes.data_ptr<int32_t>()[7] == -1,
              "padding selected an expert");
        execution.config.moe_route_stats_path = "padding-test";
        execution.config.moe_route_output_energy = true;
        record_moe_route_stats(execution, 0, selected[0], selected[1], tb::ones({8, 4}, options), 2);
        check(execution.moe_route_stats.at(0).counts.sum().item<double>() == 6,
              "padding contributed to route counts");
        check(execution.moe_route_stats.at(0).output_energy.sum().item<double>() == 24,
              "padding contributed to route energy statistics");
        std::cout << "batch_decode_graph PASS arrivals=1 departures=1 reorder=1 padding=1 state_bits=1"
                  << " captures=" << cache.captures << " bucket_reuses=" << cache.bucket_reuses << '\n';
        std::cout << "batch_decode_graph_pool largest_bytes=" << largest_graph_reserved
                  << " all_buckets_bytes=" << buckets_reserved << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
