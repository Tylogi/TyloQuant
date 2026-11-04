#pragma once

#include "models/qwen4_exp/batch_attention.h"

namespace {

QsaWeights batch_qsa_weights(int width, tb::ScalarType dtype, int heads = 4) {
    constexpr int hidden = 32, kv = 2, ih = 2, iw = 8;
    const auto projection = [&](int in, int out, int salt) -> Linear {
        auto weight = data({out, in}, salt).to(dtype);
        return [weight](CudaExecutionContext&, const Tensor& x) {
            return tb::matmul(x.to(weight.scalar_type()), weight.transpose(-1, -2));
        };
    };
    auto norm = data({width}, 1, 0), index_norm = data({iw}, 1, 0);
    return {projection(hidden, heads * 2 * width, 1), projection(hidden, kv * width, 2),
        projection(hidden, kv * width, 3), projection(heads * width, hidden, 4),
        projection(hidden, (ih + 1) * iw, 5), norm, norm, index_norm, index_norm};
}

std::unique_ptr<Qsa> batch_qsa_request(const QsaWeights& weights, const QsaConfig& config,
        const std::shared_ptr<RotaryEmbedding>& rotary, bool tiered, int cells = 16) {
    if (!tiered)
        return std::make_unique<Qsa>(weights, config, rotary);
    const auto quant = cuda_kv_quantization();
    const int kw = quant.enabled() ? cuda_kv_packed_width(config.width, quant.key_bits()) : config.width;
    const int vw = quant.enabled() ? cuda_kv_packed_width(config.width, quant.value_bits()) : config.width;
    KvMemoryOptions options;
    options.budget_policy = true;
    options.additional_request = true;
    options.gpu_budget = (config.maximum + (config.maximum + config.pool - 1) / config.pool) *
        config.index_width * 2 + cells * config.kv_heads * (kw + vw) * (quant.enabled() ? 4 : 2);
    auto memory = std::make_shared<KvMemoryManager>(options);
    auto request = std::make_unique<Qsa>(weights, config, rotary, true, true, nullptr, memory);
    memory->finalize();
    request->activate_memory();
    const auto metrics = memory->metrics();
    const auto resident = std::find_if(metrics.begin(), metrics.end(), [](const auto& metric) {
        return metric.first == "kv_memory_resident_cells";
    });
    MFQ_RUNTIME_CHECK(resident != metrics.end() && resident->second == cells,
        "QSA tier test did not allocate exclusive physical tiers");
    return request;
}

void qsa_independent_batch_case(bool tiered) {
    constexpr int hidden = 32, maximum = 512, pool = 4, budget = 16;
    int cases = 0;
    for (int rows : {2, 8, 16, 32}) {
        for (int width : {32, 256}) {
            for (auto dtype : {tb::kFloat16, tb::kFloat32}) {
                CudaExecutionContext execution;
                auto rotary = std::make_shared<RotaryEmbedding>(8, maximum, 10000.,
                    std::vector<int64_t>{2, 1, 1}, true);
                const auto weights = batch_qsa_weights(width, dtype);
                const QsaConfig config{4, 2, width, 2, 8, pool, budget, maximum, 1e-6};
                std::vector<std::unique_ptr<Qsa>> requests, oracles;
                std::unique_ptr<Qsa> cancelled;
                std::vector<Tensor> cancelled_before;
                std::vector<Tensor> histories;
                for (int row = 0; row < rows; ++row) {
                    requests.push_back(batch_qsa_request(weights, config, rotary, tiered));
                    oracles.push_back(std::make_unique<Qsa>(weights, config, rotary));
                    const int length = std::vector<int>{14, 15, 16, 17, 253, 254, 255}[row % 7];
                    auto history = tb::arange(length, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64));
                    auto input = data({1, length, hidden}, 100 + row, .4f).to(dtype);
                    requests.back()->forward(execution, input, history, history, true);
                    oracles.back()->forward(execution, input, history, history, true);
                    requests.back()->prepare_graph(history, length <= budget ? budget : 256);
                    histories.push_back(history);
                }
                BatchDecodeInputs inputs(rows, 8 + HotKvStateFields::Count, 0);
                auto hidden_input = data({rows, 1, hidden}, 29, .4f).to(dtype);
                Tensor output;
                std::map<int64_t, std::unique_ptr<DecodeWindow>> windows;
                const auto bind = [&](int step) {
                    for (int row = 0; row < rows; ++row) {
                        const int request = (row + step) % rows;
                        const bool enabled = step == 0 || row != rows - 1;
                        auto& state = *requests[request];
                        inputs.row(row, 0, state.position(), request, enabled);
                        // A disabled row deliberately retains live state pointers.
                        state.bind_decode_state(inputs, 4, row);
                    }
                    inputs.upload();
                };
                const auto window_for = [&](int64_t capacity) -> DecodeWindow& {
                    auto& window = windows[capacity];
                    if (!window) {
                        window = std::make_unique<DecodeWindow>(mfq_current_cuda_stream());
                        window->capture([&] {
                            auto positions = inputs.positions().reshape({1, rows, 1});
                            auto projected = requests[0]->project_inputs(execution, hidden_input);
                            auto attended = requests[0]->forward_independent_graph(execution, hidden_input,
                                positions, inputs, 4, capacity, std::move(projected));
                            output = requests[0]->project_output(execution, attended);
                        }, [] {}, [&] { output = {}; });
                    }
                    return *window;
                };
                for (int step = 0; step < 7; ++step) {
                    if (step == 3) {
                        cancelled = std::move(requests[0]);
                        for (const auto& tensor : cancelled->resident_cache_tensors())
                            if (tensor.defined())
                                cancelled_before.push_back(tensor.cpu());
                        requests[0] = batch_qsa_request(weights, config, rotary, tiered);
                        oracles[0] = std::make_unique<Qsa>(weights, config, rotary);
                        histories[0] = tb::arange(17, inputs.positions().options());
                        auto input = data({1, 17, hidden}, 315, .4f).to(dtype);
                        requests[0]->forward(execution, input, histories[0], histories[0], true);
                        oracles[0]->forward(execution, input, histories[0], histories[0], true);
                        requests[0]->prepare_graph(histories[0], 256);
                    }
                    // Grow allocations while retaining the captured row-table address.
                    for (int request = 0; request < rows; ++request) {
                        auto& state = *requests[request];
                        if (state.position() + 1 > state.decode_graph_capacity()) {
                            state.leave_graph();
                            state.prepare_graph(histories[request], state.position() < 256 ? 256 : maximum);
                        }
                    }
                    bind(step);
                    hidden_input.copy_(data({rows, 1, hidden}, 30 + step, .4f).to(dtype));
                    const bool dense = rows == 2 && std::all_of(requests.begin(), requests.end(),
                        [](const auto& request) { return request->position() < budget; });
                    auto& window = window_for(dense ? budget : maximum);
                    std::vector<std::vector<Tensor>> before(rows);
                    for (int request = 0; request < rows; ++request)
                        for (const auto& tensor : requests[request]->resident_cache_tensors())
                            if (tensor.defined())
                                before[request].push_back(tensor.cpu());
                    window.run();
                    if (cancelled) {
                        std::size_t index = 0;
                        for (const auto& tensor : cancelled->resident_cache_tensors()) {
                            if (!tensor.defined())
                                continue;
                            auto after = tensor.cpu();
                            const auto& before_cancel = cancelled_before[index++];
                            MFQ_RUNTIME_CHECK(after.nbytes() == before_cancel.nbytes() &&
                                std::memcmp(after.data_ptr(), before_cancel.data_ptr(), after.nbytes()) == 0,
                                "cancelled QSA request modified by reused graph");
                        }
                    }
                    for (int row = 0; row < rows; ++row) {
                        const int request = (row + step) % rows;
                        if (step && row == rows - 1) {
                            std::size_t index = 0;
                            for (const auto& tensor : requests[request]->resident_cache_tensors()) {
                                if (!tensor.defined())
                                    continue;
                                auto after = tensor.cpu();
                                const auto& prior = before[request][index++];
                                MFQ_RUNTIME_CHECK(after.nbytes() == prior.nbytes() &&
                                    std::memcmp(after.data_ptr(), prior.data_ptr(), after.nbytes()) == 0,
                                    "disabled independent QSA row modified cache");
                            }
                            continue;
                        }
                        auto& actual = *requests[request];
                        const auto position = actual.position();
                        auto current = tb::arange(position, position + 1, inputs.positions().options());
                        histories[request] = tb::cat({histories[request], current}, -1);
                        auto expected = oracles[request]->forward(execution, hidden_input.narrow(0, row, 1),
                            current, histories[request], true);
                        equal(output.narrow(0, row, 1), expected, "independent QSA versus serial");
                        actual.advance_graph(1);
                        if (step == 6) {
                            Qwen4BlockSessionState actual_state, expected_state;
                            std::size_t bytes = 0;
                            actual.capture_session(actual_state, actual.position(), bytes);
                            oracles[request]->capture_session(expected_state, actual.position(), bytes);
                            equal(actual_state.index, expected_state.index, "independent QSA index state");
                            const auto quant = cuda_kv_quantization();
                            const auto cache_values = [&](const Tensor& cache, bool value) {
                                if (!quant.enabled())
                                    return cache;
                                return cuda_kv_decode(cache.reshape({1, actual.position(), 2, cache.size(2) / 2}),
                                    width, value ? quant.value_bits() : quant.key_bits(), value).reshape(
                                        {1, actual.position(), 2 * width});
                            };
                            equal(cache_values(actual_state.keys, false), cache_values(expected_state.keys, false),
                                "independent QSA key state");
                            equal(cache_values(actual_state.values, true), cache_values(expected_state.values, true),
                                "independent QSA value state");
                        }
                        ++cases;
                    }
                }
            }
        }
    }
    std::cout << "QSA independent batch PASS cases=" << cases
              << " tiered=" << tiered << " rows=2/8/16/32 heterogeneous=1 disabled_state=1"
              << " cancelled=1 reorder=1 growth=1 dense_sparse=1\n";
}

void qsa_independent_batch_case() {
    qsa_independent_batch_case(false);
    qsa_independent_batch_case(true);
}

std::size_t qsa_kernel_nodes(cudaGraph_t graph) {
    std::size_t count = 0, kernels = 0;
    MFQ_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    MFQ_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
    for (auto node : nodes) {
        cudaGraphNodeType type;
        MFQ_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        kernels += type == cudaGraphNodeTypeKernel;
    }
    return kernels;
}

void qsa_independent_batch_benchmark() {
    const auto* option = std::getenv("MFQ_QSA_BATCH_TIERED");
    const bool tiered = option && option[0] != '0';
    constexpr int hidden = 32, maximum = 512, width = 256;
    CudaExecutionContext execution;
    auto rotary = std::make_shared<RotaryEmbedding>(8, maximum, 10000.);
    const auto weights = batch_qsa_weights(width, tb::kFloat16, 24);
    const QsaConfig config{24, 2, width, 2, 8, 4, 32, maximum, 1e-6};
    for (int rows : {1, 4, 8, 16, 32}) {
        Tensor reference;
        for (bool independent : {false, true}) {
            std::vector<std::unique_ptr<Qsa>> requests;
            BatchDecodeInputs inputs(rows, 4 + HotKvStateFields::Count, 0);
            for (int row = 0; row < rows; ++row) {
                requests.push_back(batch_qsa_request(weights, config, rotary, tiered));
                const int length = 128 + row % 4 * 16;
                auto history = tb::arange(length, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64));
                requests.back()->forward(execution, data({1, length, hidden}, 100 + row, .4f), history, history, true);
                requests.back()->prepare_graph(history, maximum);
                inputs.row(row, 0, length);
                requests.back()->bind_decode_state(inputs, 0, row);
            }
            inputs.upload();
            auto x = data({rows, 1, hidden}, 23, .4f);
            auto positions = inputs.positions().reshape({1, rows, 1});
            // Exclude shared input/output projections from the core launch count.
            auto projected = requests[0]->project_inputs(execution, x);
            Tensor output;
            cudaGraph_t native = nullptr;
            const auto compute = [&] {
                if (independent) {
                    output = requests[0]->forward_independent_graph(execution, x, positions,
                        inputs, 0, maximum, projected);
                } else {
                    std::vector<Tensor> parts;
                    for (int row = 0; row < rows; ++row) {
                        std::vector<Tensor> row_projection;
                        for (const auto& value : projected)
                            row_projection.push_back(value.narrow(0, row, 1));
                        auto position = inputs.positions().narrow(0, row, 1);
                        parts.push_back(requests[0]->forward_indirect_graph(execution, x.narrow(0, row, 1),
                            position, position, inputs, 0, row, maximum, std::move(row_projection), false));
                    }
                    output = tb::cat(parts, 0);
                }
                cudaStreamCaptureStatus status;
#if CUDART_VERSION >= 13000
                MFQ_CUDA_CHECK(cudaStreamGetCaptureInfo(mfq_current_cuda_stream(), &status,
                    nullptr, &native, nullptr, nullptr, nullptr));
#else
                MFQ_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(mfq_current_cuda_stream(), &status,
                    nullptr, &native, nullptr, nullptr));
#endif
            };
            {
                GraphWarmupScope warmup(default_context(), mfq_current_cuda_stream());
                compute();
            }
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            output = {};
            Graph graph;
            graph.capture_begin_shared();
            compute();
            graph.capture_end();
            constexpr int repeats = 100;
            cudaEvent_t begin, end;
            MFQ_CUDA_CHECK(cudaEventCreate(&begin));
            MFQ_CUDA_CHECK(cudaEventCreate(&end));
            for (int warm = 0; warm < 5; ++warm)
                graph.replay();
            if (independent)
                equal(output, reference, "QSA benchmark per-row reference");
            else
                reference = output.cpu();
            // One host submission keeps CPU launch gaps out of the GPU timer.
            // Child graphs run serially and retain the original graph's work.
            cudaGraph_t chain;
            cudaGraphExec_t replay;
            MFQ_CUDA_CHECK(cudaGraphCreate(&chain, 0));
            cudaGraphNode_t previous = nullptr;
            for (int repeat = 0; repeat < repeats; ++repeat) {
                cudaGraphNode_t child;
                MFQ_CUDA_CHECK(cudaGraphAddChildGraphNode(&child, chain, previous ? &previous : nullptr,
                    previous ? 1 : 0, native));
                previous = child;
            }
            MFQ_CUDA_CHECK(cudaGraphInstantiate(&replay, chain, 0));
            MFQ_CUDA_CHECK(cudaGraphLaunch(replay, mfq_current_cuda_stream()));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            MFQ_CUDA_CHECK(cudaEventRecord(begin, mfq_current_cuda_stream()));
            MFQ_CUDA_CHECK(cudaGraphLaunch(replay, mfq_current_cuda_stream()));
            MFQ_CUDA_CHECK(cudaEventRecord(end, mfq_current_cuda_stream()));
            MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float milliseconds = 0;
            MFQ_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
            equal(output, reference, "QSA benchmark chained replay output");
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));
            MFQ_CUDA_CHECK(cudaEventDestroy(end));
            MFQ_CUDA_CHECK(cudaGraphExecDestroy(replay));
            MFQ_CUDA_CHECK(cudaGraphDestroy(chain));
            std::cout << "qsa_independent_batch_bench rows=" << rows << " independent=" << independent
                      << " tiered=" << tiered << " kv_bits=" << cuda_kv_quantization().bits
                      << " kernels=" << qsa_kernel_nodes(native) << " nodes=" << graph.nodes()
                      << " graph_us=" << milliseconds * 1000 / repeats
                      << " heads=24 kv_heads=2 width=" << width << " context=128..176 capacity=" << maximum
                      << " replay_method=child_graph_chain repeats=" << repeats << '\n';
        }
    }
}

} // namespace
