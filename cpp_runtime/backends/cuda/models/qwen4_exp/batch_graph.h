#pragma once

// Model adaptation for the shared bucket cache. Included after request state
// is defined in ops.cpp; all request-dependent GPU reads use BatchDecodeInputs.
struct Qwen4BatchDecodeGraph::Impl {
    using Tensor = mfq_tensor_backend::Tensor;
    static constexpr int64_t fields_per_layer = 8 + HotKvStateFields::Count;
    struct Layer {
        std::vector<std::unique_ptr<qwen4_exp::Gdn>> gdn;
        std::vector<std::unique_ptr<qwen4_exp::Ple>> ple;
        std::vector<std::shared_ptr<RowStage>> embeddings;
        bool independent_gdn = false;
    };
    struct Prepared {
        std::vector<Layer> layers;
    };
    Qwen4CausalLm& model;
    BatchDecodeGraphCache cache;
    Event ready;
    Prepared prepared;
    explicit Impl(Qwen4CausalLm& owner) : model(owner), cache(true) {}
    ~Impl() { clear(); }
    void clear() {
        finish_moe_expert_exchanges(model.execution->moe_expert_cache);
        cache.clear();
        prepared.layers.clear();
    }
};

Qwen4BatchDecodeGraph::Qwen4BatchDecodeGraph(Qwen4CausalLm& model)
    : graph_(std::make_unique<Impl>(model)) {}
Qwen4BatchDecodeGraph::~Qwen4BatchDecodeGraph() = default;
void Qwen4BatchDecodeGraph::clear() { graph_->clear(); }
bool Qwen4BatchDecodeGraph::has_graphs() const { return graph_->cache.has_graphs(); }
int64_t Qwen4BatchDecodeGraph::captures() const { return graph_->cache.captures; }
int64_t Qwen4BatchDecodeGraph::replays() const { return graph_->cache.replays; }
int64_t Qwen4BatchDecodeGraph::membership_captures() const { return graph_->cache.membership_captures; }
int64_t Qwen4BatchDecodeGraph::bucket_reuses() const { return graph_->cache.bucket_reuses; }
double Qwen4BatchDecodeGraph::capture_ms() const { return graph_->cache.capture_ms; }

std::optional<mfq_tensor_backend::Tensor> Qwen4BatchDecodeGraph::run(
        const std::vector<Qwen4BatchRequestState*>& requests,
        const mfq_tensor_backend::Tensor& ids, const mfq_tensor_backend::Tensor& host_ids, bool greedy,
        int64_t maximum_rows) {
    namespace tb = mfq_tensor_backend;
    auto& owner = *graph_;
    auto& model = owner.model;
    auto& execution = *model.execution;
    if (execution.qsa_kv_store || !mfq_cuda_graph_capture_supported())
        return {};
    for (auto* request : requests) {
        for (const auto& layer : request->state_->layers) {
            if (layer.ple && !layer.ple->graph_supported())
                return {};
            if (layer.qsa && layer.qsa->has_kv_offload())
                return {};
        }
    }
    release_moe_prefill_buffers(execution.moe_expert_cache);
    const auto active = static_cast<int64_t>(requests.size());
    const auto batch = batch_decode_row_bucket(active, maximum_rows);
    int64_t required = 0;
    for (const auto* request : requests)
        required = std::max(required, request->position() + 1);
    MFQ_RUNTIME_CHECK(required <= model.max_position_embeddings(), "batch graph exceeds max_context");
    if (execution.kv_memory)
        execution.kv_memory->ensure(required);
    owner.ready.record(mfq_get_current_cuda_stream());
    MfqCudaStreamGuard stream_guard(owner.cache.stream());
    MFQ_CUDA_CHECK(cudaStreamWaitEvent(owner.cache.stream().stream(), owner.ready.get(), 0));

    std::size_t growth = 0;
    auto capacity = batch_decode_cache_capacity(required, batch, model.max_position_embeddings(),
                                                 runtime_options::continuous_batch_kv_initial_capacity());
    for (auto* request : requests) {
        for (auto& layer : request->state_->layers) {
            if (layer.gdn)
                growth += layer.gdn->request_cache_growth_bytes();
            if (layer.qsa && layer.qsa->decode_graph_capacity() < required) {
                growth += layer.qsa->graph_growth_bytes(capacity);
            }
        }
    }
    if (model.reserve_kv_workspace)
        model.reserve_kv_workspace(growth + (std::size_t(64) << 20));
    for (auto* request : requests) {
        for (auto& layer : request->state_->layers) {
            if (layer.gdn)
                layer.gdn->initialize_request_cache();
            if (layer.qsa) {
                if (layer.qsa->decode_graph_capacity() && layer.qsa->decode_graph_capacity() < required)
                    layer.qsa->leave_graph();
                layer.qsa->prepare_graph(request->state_->positions, capacity);
                capacity = std::max(capacity, layer.qsa->decode_graph_capacity());
            }
        }
    }
    const BatchDecodeGraphKey key{batch, decode_graph_bucket(required, model.max_position_embeddings()),
                                  capacity, {}};
    auto& graph = owner.cache.get(key, Impl::fields_per_layer * model.blocks.size(), ids.get_device());
    auto& input = *graph.inputs;
    auto& prepared = owner.prepared;
    std::vector<BatchDecodeRow> rows;
    for (int64_t row = 0; row < active; ++row)
        rows.push_back({host_ids.data_ptr<int64_t>()[row], requests[row]->position(), row});
    input.update_rows(rows);
    std::vector<std::pair<qwen4_exp::Ple*, qwen4_exp::NgramEmbedding::Context>> contexts;

    if (prepared.layers.empty()) {
        prepared.layers.resize(model.blocks.size());
        for (std::size_t i = 0; i < model.blocks.size(); ++i) {
            auto& layer = prepared.layers[i];
            auto& source = requests.front()->state_->layers[i];
            if (source.gdn) {
                std::vector<qwen4_exp::Gdn*> candidates(2, source.gdn.get());
                layer.independent_gdn = runtime_options::batch_attention_projections() &&
                    runtime_options::gdn_independent_batch() &&
                    qwen4_exp::Gdn::prepare_independent_batch(candidates).defined();
            }
        }
    }
    for (std::size_t i = 0; i < model.blocks.size(); ++i) {
        auto& layer = prepared.layers[i];
        auto& source = requests.front()->state_->layers[i];
        if (source.gdn && !layer.independent_gdn) {
            while (layer.gdn.size() < std::size_t(batch)) {
                auto working = std::make_unique<qwen4_exp::Gdn>(*source.gdn);
                working->restore_session(*source.gdn->graph_state()[0], *source.gdn->graph_state()[1]);
                working->initialize_request_cache();
                layer.gdn.push_back(std::move(working));
            }
        }
        if (source.ple) {
            while (layer.ple.size() < std::size_t(batch)) {
                auto working = std::make_unique<qwen4_exp::Ple>(*source.ple);
                working->restore_session(source.ple->conv_state(), {});
                layer.embeddings.push_back(working->graph_stage(1, ids.device()));
                layer.ple.push_back(std::move(working));
            }
        }
    }
    auto padding_id = tb::zeros({1, 1}, host_ids.options());
    for (int64_t row = 0; row < batch; ++row) {
        const bool enabled = row < active;
        const auto* request = enabled ? requests[row] : nullptr;
        for (std::size_t i = 0; i < model.blocks.size(); ++i) {
            const auto field = int64_t(Impl::fields_per_layer * i);
            auto& working = prepared.layers[i];
            if (enabled) {
                auto& layer = request->state_->layers[i];
                if (layer.gdn) {
                    const auto values = layer.gdn->indirect_state_tensors();
                    for (int offset = 0; offset < 3; ++offset)
                        input.state(row, field + offset, values[offset]);
                }
                if (layer.qsa)
                    layer.qsa->bind_decode_state(input, field + 4, row);
                if (layer.ple) {
                    input.state(row, field + 3, layer.ple->conv_state());
                    contexts.emplace_back(layer.ple.get(), layer.ple->session_context());
                    layer.ple->upload_graph(host_ids.narrow(0, row, 1), *working.embeddings[row]);
                }
            } else if (!working.ple.empty()) {
                working.ple[row]->upload_graph(padding_id, *working.embeddings[row]);
            }
        }
    }
    input.upload();
    const bool external_pool = prepare_moe_graph_pool(execution.moe_expert_cache) ||
        prepare_tq_graph_pool(execution.tq_expert_cache);
    owner.cache.prepare_pool(execution.tensor_parallel, execution.model_parallel_collectives, external_pool);
    BatchDecodeMaskScope mask_scope(execution, input.enabled());
    if (!graph.valid) {
        struct Saved { Tensor target, value; };
        std::vector<Saved> saved;
        for (auto* request : requests) {
            for (const auto& tensor : request->recurrent_state_tensors())
                if (tensor.defined())
                    saved.push_back({tensor, batch > 8 ? tensor.cpu() : tensor.clone()});
        }
        const auto restore = [&] {
            for (auto& value : saved)
                value.target.copy_(value.value);
        };
        graph.window = std::make_unique<DecodeWindow>(owner.cache.stream().stream(),
                                                       runtime_options::layer_timings(), true);
        graph.window->set_profile_geometry("batch_decode", static_cast<int>(batch));
        DecodeGraphBranchScope branch_scope(execution.decode_graph_serial_branches);
        const auto started = std::chrono::steady_clock::now();
        try {
            graph.window->capture([&] {
                auto hidden = model.adapter_prepare_hidden(model.embed_tokens(input.ids()), batch, 1);
                graph.window->mark(-1, "window_begin");
                std::vector<Tensor> prior_mix;
                const auto chain_setting = runtime_options::gr_chain_ffn_post();
                const bool chain_post = (!chain_setting || *chain_setting) && batch <= 8;
                for (std::size_t i = 0; i < model.blocks.size(); ++i) {
                    auto& block = static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
                    auto& working = prepared.layers[i];
                    const auto field = int64_t(Impl::fields_per_layer * i);
                    graph.window->mark(i, "block_begin");
                    graph.window->set_profile_layer_kind(i, block.gdn ? "gdn" : "qsa");
                    if (block.ple) {
                        std::vector<Tensor> positional;
                        for (int64_t row = 0; row < batch; ++row) {
                            auto& ple = *working.ple[row];
                            batch_state_load(input, field + 3, row, ple.conv_state());
                            working.embeddings[row]->decode();
                            auto rows = working.embeddings[row]->output().reshape({1, 1, model.hidden_size()});
                            auto value = hidden.narrow(0, row, 1);
                            positional.push_back(value + ple.forward_chunk(execution, value,
                                input.ids().narrow(0, row, 1), true, {}, rows));
                            batch_state_store(input, field + 3, row, ple.conv_state());
                        }
                        hidden = join_request_outputs(positional, 0);
                    }
                    auto mix = prior_mix.empty() ? block.attention_gr.pre(execution, hidden)
                        : block.attention_gr.pre_after(execution, hidden, prior_mix);
                    graph.window->mark(i, "attention_core_begin");
                    const bool independent_qsa = !block.gdn && batch_selected_attention_enabled() &&
                        requests.front()->state_->layers[i].qsa->independent_graph_supported();
                    const bool project_batch = independent_qsa || (runtime_options::batch_attention_projections() &&
                        (batch > 1 || (block.gdn && working.independent_gdn)));
                    auto projections = project_batch ? (block.gdn ? block.gdn->project_inputs(execution, mix[0])
                        : block.qsa->project_inputs(execution, mix[0])) : std::vector<Tensor>{};
                    Tensor attention;
                    if (independent_qsa) {
                        auto positions = (input.positions() + model.decode_position_delta).reshape({1, batch, 1});
                        attention = requests.front()->state_->layers[i].qsa->forward_independent_graph(
                            execution, mix[0], positions, input, field + 4, capacity, std::move(projections));
                    }
                    if (block.gdn && working.independent_gdn) {
                        if (!project_batch)
                            projections = block.gdn->project_inputs(execution, mix[0]);
                        auto addresses = input.states().narrow(0, field, 3).select(2, 0).transpose(0, 1).contiguous();
                        attention = requests.front()->state_->layers[i].gdn->forward_independent_batch(
                            mix[0], projections, addresses);
                    }
                    if (!attention.defined()) {
                        if (block.gdn && working.gdn.size() < std::size_t(batch)) {
                            auto& source = *requests.front()->state_->layers[i].gdn;
                            while (working.gdn.size() < std::size_t(batch)) {
                                auto fallback = std::make_unique<qwen4_exp::Gdn>(source);
                                fallback->restore_session(*source.graph_state()[0], *source.graph_state()[1]);
                                fallback->initialize_request_cache();
                                working.gdn.push_back(std::move(fallback));
                            }
                        }
                        std::vector<Tensor> attended;
                        for (int64_t row = 0; row < batch; ++row) {
                            std::vector<Tensor> projected;
                            for (const auto& value : projections)
                                projected.push_back(value.narrow(0, row, 1));
                            auto branch = mix[0].narrow(0, row, 1);
                            if (block.gdn) {
                                auto& gdn = *working.gdn[row];
                                auto values = gdn.indirect_state_tensors();
                                for (int offset = 0; offset < 3; ++offset)
                                    if (values[offset].defined())
                                        batch_state_load(input, field + offset, row, values[offset]);
                                attended.push_back(gdn.forward_chunk(execution, branch, true,
                                                                     std::move(projected), !project_batch));
                                values = gdn.indirect_state_tensors();
                                for (int offset = 0; offset < 3; ++offset)
                                    if (values[offset].defined())
                                        batch_state_store(input, field + offset, row, values[offset]);
                            } else {
                                auto position = input.positions().narrow(0, row, 1);
                                attended.push_back(requests.front()->state_->layers[i].qsa->forward_indirect_graph(
                                    execution, branch, position + model.decode_position_delta, position,
                                    input, field + 4, row, capacity, std::move(projected), !project_batch));
                            }
                        }
                        attention = join_request_outputs(attended, 0);
                    }
                    if (project_batch)
                        attention = block.gdn ? block.gdn->project_output(execution, attention)
                                              : block.qsa->project_output(execution, attention);
                    graph.window->mark(i, "attention_core_end");
                    auto ffn_mix = block.ffn_gr.pre_after(execution, std::move(attention), mix);
                    graph.window->mark(i, "ffn_pre_end");
                    auto branch = block.batch_ffn(execution, ffn_mix[0]);
                    bool defer = false;
                    if (chain_post && i + 1 < model.blocks.size()) {
                        auto& next = static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i + 1]);
                        defer = !next.ple && next.attention_gr.can_fuse(execution, ffn_mix[1]);
                    }
                    if (defer) {
                        hidden = std::move(branch);
                        prior_mix = std::move(ffn_mix);
                    } else {
                        hidden = block.ffn_gr.post(branch, ffn_mix);
                        prior_mix.clear();
                    }
                    graph.window->mark(i, "block_end");
                }
                auto finalized = model.finalize_hidden(std::move(hidden), batch, 1);
                graph.output = std::move(finalized);
                graph.window->mark(-1, "window_end");
            }, restore, [&] { graph.output = {}; });
            owner.cache.captured(graph, std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        } catch (...) {
            graph.window.reset();
            graph.output = {};
            restore();
            for (auto& [layer, context] : contexts)
                layer->restore_graph_context(std::move(context));
            throw;
        }
    }
    // Sampling mode belongs to the current group, preserving the fused greedy
    // vocabulary projection without capturing that choice in the window.
    const auto project_head = [&] {
        auto finalized = graph.output.narrow(0, 0, active);
        return greedy ? model.adapter_next_token(model.lm_head, model.last_hidden(std::move(finalized)))
                           .reshape({active, 1})
                      : model.logits_from_hidden(std::move(finalized)).reshape({active, -1});
    };
    try {
        owner.cache.prepare_output(active, greedy, [&] {
            // Price cold vocabulary workspace before committing recurrent state.
            graph.output.zero_();
            (void)project_head();
            MFQ_CUDA_CHECK(cudaStreamSynchronize(owner.cache.stream().stream()));
        });
    } catch (...) {
        for (auto& [layer, context] : contexts)
            layer->restore_graph_context(std::move(context));
        throw;
    }
    graph.window->run();
    owner.cache.replayed(graph);
    auto output = project_head();
    for (int64_t row = 0; row < active; ++row) {
        auto& state = *requests[row]->state_;
        for (auto& layer : state.layers)
            if (layer.qsa)
                layer.qsa->advance_graph(1);
        state.positions = append_decode_positions(state.positions, input.positions().narrow(0, row, 1),
                                                   state.position_storage, model.max_position_embeddings());
        ++state.cache_position;
    }
    return output;
}
