#pragma once

#include "mfq_cuda_moe_ops.h"
#include "moe_types.h"


struct MfeWeight {
    struct ExpertParallelShard {
        int device = 0;
        int64_t expert_begin = 0;
        int64_t expert_end = 0;
        std::shared_ptr<MfeWeight> weight;
    };

    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<MfePoolWeight> pools;
    int64_t mixed_weight_bytes = 0;
    bool partial_experts = false;
    bool unified_nint_projection = false;
    std::vector<ExpertParallelShard>
        expert_parallel_shards;
    std::function<void(const MoeRoutePlan &)> cache_prefetch;
    std::function<void(const MoeRoutePlan &)> cache_prefetch_begin;
    std::function<mfq_tensor_backend::Tensor(
        CudaExecutionContext&, mfq_tensor_backend::Tensor,
        const MoeRoutePlan&)> mixed_forward;
    std::function<mfq_tensor_backend::Tensor(
        CudaExecutionContext&, mfq_tensor_backend::Tensor,
        const MoeRoutePlan&)> mixed_prequantized_forward;
    std::function<mfq_tensor_backend::Tensor(
        CudaExecutionContext&, mfq_tensor_backend::Tensor,
        const MoeRoutePlan&, bool)> mixed_glu_output_forward;
    std::function<mfq_tensor_backend::Tensor(
        CudaExecutionContext&, mfq_tensor_backend::Tensor,
        const MoeRoutePlan&, bool)> mixed_glu_forward;
    std::function<mfq_tensor_backend::Tensor(
        CudaExecutionContext&, mfq_tensor_backend::Tensor,
        const MoeRoutePlan&, double)> mixed_clamped_swiglu_forward;
    std::vector<MoeActivationGeometry> activation_geometries;
    int activation_workspace_domain = 0;
    std::shared_ptr<MoeCachedSource> cached_source;
    mutable std::shared_ptr<std::unordered_map<
        MoeActivationKey, MoeActivationWorkspace, MoeActivationKeyHash>>
        activation_workspaces = std::make_shared<std::unordered_map<
            MoeActivationKey, MoeActivationWorkspace, MoeActivationKeyHash>>();

    MoeActivationWorkspace activation_workspace(
            mfq_tensor_backend::Tensor x, int input_rows, int groups, int gs,
            bool stable = true) const {
        // A producer may have already quantized this exact view for us.
        const MoeActivationKey exact{input_rows, groups, gs, x.get_device()};
        auto found = activation_workspaces->find(exact);
        if (found != activation_workspaces->end()) return found->second;
        stable = stable && input_rows <= MoeActivationWorkspace::stable_row_limit;
        const int capacity = MoeActivationWorkspace::capacity_rows(input_rows, stable);
        // Each negative key owns a permanent capacity bucket. Larger eager
        // prefill uses key zero and cannot replace a captured bucket address.
        const MoeActivationKey key{stable ? -capacity : 0, groups, gs, x.get_device()};
        auto it = activation_workspaces->find(key);
        if (it == activation_workspaces->end() || it->second.qx.size(0) < input_rows) {
            MoeActivationWorkspace workspace;
            workspace.qx = mfq_tensor_backend::empty(
                {capacity, groups * gs}, x.options().dtype(mfq_tensor_backend::kInt8));
            workspace.xscale = mfq_tensor_backend::empty(
                {capacity, groups}, x.options().dtype(mfq_tensor_backend::kFloat32));
            it = activation_workspaces->insert_or_assign(key, std::move(workspace)).first;
        }
        return {it->second.qx.narrow(0, 0, input_rows),
                it->second.xscale.narrow(0, 0, input_rows)};
    }

    bool expert_parallel() const {
        return !expert_parallel_shards.empty();
    }

    bool supports_projection_glu_epilogue() const {
        if (!expert_parallel()) {
            return unified_nint_projection || !pools.empty();
        }
        return std::all_of(
            expert_parallel_shards.begin(),
            expert_parallel_shards.end(),
            [](const ExpertParallelShard & shard) {
                return shard.weight &&
                    shard.weight->supports_projection_glu_epilogue();
            });
    }

    template <typename Forward>
    mfq_tensor_backend::Tensor forward_expert_parallel(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            Forward && forward) const {
        std::vector<mfq_tensor_backend::Tensor> outputs(
            expert_parallel_shards.size());
        for (size_t launch_position = 0;
             launch_position < expert_parallel_shards.size();
             ++launch_position) {
            const size_t index = model_parallel_launch_index(
                execution.config, launch_position, expert_parallel_shards.size());
            const auto & shard = expert_parallel_shards[index];
            if (!shard.weight) {
                throw std::runtime_error(
                    "expert-parallel MoE shard is missing");
            }
            MfqCudaGuard guard(shard.device);
            auto local_x = moe_tensor_to_device(
                execution.model_parallel_collectives, x, shard.device);
            const auto& local_route = moe_route_to_device(
                execution.model_parallel_collectives, route, shard.device);
            outputs[index] = forward(
                *shard.weight,
                local_x,
                local_route);
        }
        return reduce_model_parallel_outputs(
            execution, std::move(outputs));
    }

    void prefetch(const MoeRoutePlan & route) const {
        if (cache_prefetch) cache_prefetch(route);
    }

    void prefetch_begin(const MoeRoutePlan & route) const {
        if (cache_prefetch_begin) cache_prefetch_begin(route);
    }

    bool supports_prequantized_input() const {
        return !expert_parallel() &&
            (!pools.empty() || static_cast<bool>(mixed_prequantized_forward));
    }

    bool can_reuse_activation_for(
            const MfeWeight & consumer) const {
        if (!supports_prequantized_input() ||
                !consumer.supports_prequantized_input() ||
                neuron_len != consumer.neuron_len ||
                activation_workspace_domain == 0 ||
                activation_workspace_domain !=
                    consumer.activation_workspace_domain) {
            return false;
        }
        if (activation_geometries.empty() ||
                consumer.activation_geometries.empty()) {
            return false;
        }
        return std::all_of(
            consumer.activation_geometries.begin(),
            consumer.activation_geometries.end(),
            [&](const MoeActivationGeometry & required) {
                return std::find(
                    activation_geometries.begin(),
                    activation_geometries.end(),
                    required) != activation_geometries.end();
            });
    }

    mfq_tensor_backend::Tensor forward(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) const {
        return forward_impl(execution, x, route, false);
    }

    mfq_tensor_backend::Tensor forward_prequantized(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) const {
        return forward_impl(execution, x, route, true);
    }

    mfq_tensor_backend::Tensor forward_impl(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool input_prequantized,
            int epilogue_mode = 0) const {
        if (expert_parallel()) {
            if (input_prequantized) {
                throw std::runtime_error(
                    "prequantized MoE activation reuse is unavailable with expert parallelism");
            }
            return forward_expert_parallel(
                execution, x, route,
                [&execution, epilogue_mode](const MfeWeight & shard,
                   mfq_tensor_backend::Tensor local_x,
                   const MoeRoutePlan & local_route) {
                    return shard.forward_impl(
                        execution, local_x, local_route, false, epilogue_mode);
                });
        }
        if (epilogue_mode < 0 || epilogue_mode > 2 ||
                (epilogue_mode != 0 && (out_per_expert % 2) != 0)) {
            throw std::runtime_error("invalid MFE NINT projection epilogue");
        }
        if (input_prequantized && mixed_prequantized_forward) {
            if (epilogue_mode != 0) {
                throw std::runtime_error(
                    "mixed prequantized MFE GLU epilogue is unavailable");
            }
            return mixed_prequantized_forward(execution, x, route);
        }
        if (mixed_forward) {
            if (epilogue_mode != 0) {
                throw std::runtime_error(
                    "mixed MFE GLU epilogue must use its native dispatch");
            }
            return mixed_forward(execution, x, route);
        }
        if (!x.is_cuda() || !x.is_contiguous() || x.scalar_type() != mfq_tensor_backend::kFloat16 ||
            (x.dim() != 2 && x.dim() != 3) || x.size(-1) != neuron_len) {
            throw std::runtime_error("MFE input must be contiguous CUDA f16 with exact K");
        }
        int tokens = (int)route.ids.size(0);
        int routes = (int)route.ids.size(1);
        if (route.n_experts != n_experts || x.size(0) != tokens ||
            (x.dim() == 3 && x.size(1) != routes)) {
            throw std::runtime_error("MFE input and route shape mismatch");
        }
        int input_rows = x.dim() == 3 ? tokens * routes : tokens;
        const int result_width = epilogue_mode == 0
            ? out_per_expert
            : out_per_expert / 2;
        auto output = partial_experts
            ? mfq_tensor_backend::zeros(
                {tokens, routes, result_width},
                x.options().dtype(mfq_tensor_backend::kFloat16))
            : mfq_tensor_backend::empty(
                {tokens, routes, result_width},
                x.options().dtype(mfq_tensor_backend::kFloat16));
        std::unordered_set<MoeActivationKey, MoeActivationKeyHash> quantized;
        int nint_pool_phase = 0;
        const bool stable = input_rows <= 2048;
        const EagerActivationScratch<std::remove_reference_t<decltype(*activation_workspaces)>>
            eager_scratch{*activation_workspaces, !stable};
        for (const auto & pool : pools) {
            int groups = (int)pool.weight.ng;
            int gs = (int)pool.weight.gs;
            MoeActivationKey key{input_rows, groups, gs, x.get_device()};
            auto workspace = activation_workspace(x, input_rows, groups, gs, stable);
            bool input_quantized = input_prequantized ||
                quantized.find(key) != quantized.end();
            const int route_tile_m = select_nint_prefill_route_tile(
                route, tokens, routes, n_experts);
            mfe_nint_matmul_ws_cuda(
                pool.weight.q_packed, pool.weight.row_q_bits,
                pool.weight.row_q_bit_offsets, pool.weight.sub_scale,
                pool.weight.sub_min, pool.weight.neuron_scale,
                pool.weight.neuron_min, x, route.ids, pool.expert_local,
                n_experts, pool.local_experts, out_per_expert, gs,
                epilogue_mode, route.map_ready, input_quantized,
                output, workspace.qx,
                workspace.xscale, route.ids_dst, route.expert_bounds,
                nint_route_tile_bounds(route, route_tile_m),
                nint_route_tile_experts(route, route_tile_m),
                route_tile_m, nint_pool_phase++);
            if (input_quantized || route_tile_m == 8) {
                quantized.insert(key);
            }
        }
        return output;
    }

    mfq_tensor_backend::Tensor forward_glu_output(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x, const MoeRoutePlan & route, bool gelu) const {
        if (expert_parallel()) {
            return forward_expert_parallel(
                execution, x, route,
                [&execution, gelu](
                    const MfeWeight & shard,
                    mfq_tensor_backend::Tensor local_x,
                    const MoeRoutePlan & local_route) {
                    return shard.forward_glu_output(
                        execution, local_x, local_route, gelu);
                });
        }
        if (mixed_glu_output_forward) {
            return mixed_glu_output_forward(execution, x, route, gelu);
        }
        if (mixed_forward) {
            auto gate_up = mixed_forward(execution, x, route);
            return gelu
                ? moe_geglu_split_cuda(gate_up)
                : moe_swiglu_split_cuda(gate_up);
        }
        return forward_impl(execution, x, route, false, gelu ? 2 : 1);
    }

    mfq_tensor_backend::Tensor forward_glu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor gate_up, const MoeRoutePlan & route, bool gelu) const {
        if (expert_parallel()) {
            return forward_expert_parallel(
                execution, gate_up, route,
                [&execution, gelu](
                    const MfeWeight & shard,
                    mfq_tensor_backend::Tensor local_gate_up,
                    const MoeRoutePlan & local_route) {
                    return shard.forward_glu(
                        execution, local_gate_up,
                        local_route, gelu);
                });
        }
        if (mixed_glu_forward) {
            return mixed_glu_forward(execution, gate_up, route, gelu);
        }
        if (!gate_up.is_cuda() || !gate_up.is_contiguous() ||
                gate_up.scalar_type() != mfq_tensor_backend::kFloat16 ||
                gate_up.dim() != 3 || gate_up.size(2) != 2 * neuron_len) {
            throw std::runtime_error("fused MFE GLU input has an unsupported layout");
        }
        auto activation = gelu
            ? moe_geglu_split_cuda(gate_up)
            : moe_swiglu_split_cuda(gate_up);
        return forward(execution, activation, route);
    }

    mfq_tensor_backend::Tensor forward_swiglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan& route) const {
        return forward_glu(execution, gate_up, route, false);
    }

    bool supports_clamped_swiglu() const {
        if (expert_parallel()) {
            return std::all_of(
                expert_parallel_shards.begin(),
                expert_parallel_shards.end(),
                [](const ExpertParallelShard & shard) {
                    return shard.weight &&
                        shard.weight
                            ->supports_clamped_swiglu();
                });
        }
        return !pools.empty() ||
            static_cast<bool>(mixed_clamped_swiglu_forward);
    }

    mfq_tensor_backend::Tensor forward_clamped_swiglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) const {
        if (expert_parallel()) {
            return forward_expert_parallel(
                execution, gate_up, route,
                [&execution, limit](
                    const MfeWeight & shard,
                    mfq_tensor_backend::Tensor local_gate_up,
                    const MoeRoutePlan & local_route) {
                    return shard.forward_clamped_swiglu(
                        execution, local_gate_up,
                        local_route, limit);
                });
        }
        if (mixed_clamped_swiglu_forward) {
            return mixed_clamped_swiglu_forward(
                execution, gate_up, route, limit);
        }
        if (pools.empty() || !gate_up.is_cuda() ||
                !gate_up.is_contiguous() ||
                gate_up.scalar_type() != mfq_tensor_backend::kFloat16 ||
                gate_up.dim() != 3 ||
                gate_up.size(2) != 2 * neuron_len || limit <= 0.0) {
            throw std::runtime_error(
                "clamped SwiGLU is unavailable for this MFE tensor");
        }
        const int64_t width = gate_up.size(2) / 2;
        auto gate = mfq_tensor_backend::clamp_max(
            gate_up.slice(2, 0, width)
                .to(mfq_tensor_backend::kFloat32),
            limit);
        auto up = mfq_tensor_backend::clamp(
            gate_up.slice(2, width, 2 * width)
                .to(mfq_tensor_backend::kFloat32),
            -limit, limit);
        auto activation = (mfq_tensor_backend::silu(gate) * up)
            .to(mfq_tensor_backend::kFloat16).contiguous();
        return forward(execution, activation, route);
    }

    mfq_tensor_backend::Tensor forward_geglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan& route) const {
        return forward_glu(execution, gate_up, route, true);
    }
};


MoeRoutePlan build_moe_route_plan(
    mfq_tensor_backend::Tensor ids, int n_experts);

MfeWeight cpu_mixed_moe_metadata(
    const std::shared_ptr<MixedMoeRuntime>& runtime);

MfeWeight stage_cpu_mixed_moe(
    const std::shared_ptr<MixedMoeRuntime>& runtime,
    const CudaExecutionConfig& config = {});

bool prefetch_cached_moe_projection_bundle(
    const MfeWeight& gate,
    const MfeWeight& up,
    const MfeWeight& down,
    const MoeRoutePlan& route);

bool prefetch_cached_moe_projection_bundle(
    const MfeWeight& gate_up,
    const MfeWeight& down,
    const MoeRoutePlan& route);
