#include "ffn.h"
#include "models/common/gated_mlp.h"
#include "models/common/moe.h"

#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "storage/moe_expert_cache.h"
#include "storage/moe_ffn_pipeline.h"
#include "runtime/batch_decode_graph.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Tensor = mfq_tensor_backend::Tensor;
using ActivationKind = mfq::models::MlpActivation;

bool dense_linear_defined(const QuantLinear &linear) {
    return linear.out() > 0 && linear.neuron_len() > 0;
}

} // namespace

namespace {

Tensor dense_expert_activation(ActivationKind activation, Tensor gate, Tensor up, double limit) {
    if (activation == ActivationKind::relu_squared) {
        MFQ_RUNTIME_CHECK(!gate.defined(), "relu-squared experts have no gate projection");
        return relu_squared_cuda(up.contiguous());
    }
    MFQ_RUNTIME_CHECK(gate.defined(), "gated experts require a gate projection");
    if (activation == ActivationKind::gelu_gated)
        return gelu_mul_cuda(gate.contiguous(), up.contiguous());
    if (limit > 0.0) {
        auto clipped_gate = mfq_tensor_backend::clamp_max(
            gate.to(mfq_tensor_backend::kFloat32), limit);
        auto clipped_up = mfq_tensor_backend::clamp(
            up.to(mfq_tensor_backend::kFloat32), -limit, limit);
        return (mfq_tensor_backend::silu(clipped_gate) * clipped_up)
            .to(mfq_tensor_backend::kFloat16).contiguous();
    }
    return (up * mfq_tensor_backend::silu(gate)).contiguous();
}

} // namespace

void FFN::prepare_moe_pipeline(CudaExecutionContext& execution) const {
    if(moe_pipeline_initialized || !uses_moe_expert_cache() || expert_parallel_moe_compatible() ||
            cpu_moe_gate || cpu_moe_up || cpu_moe_gate_up || cpu_moe_down ||
            !execution.config.moe_route_stats_path.empty())return;
    std::vector<std::shared_ptr<MfeWeight>> gate_up;
    // An ungated bank publishes Up in both slots; the pipeline detects the
    // single-image layout from the projection geometry, not the model name.
    if(moe_ungated_experts)
        gate_up={std::make_shared<MfeWeight>(moe_up),std::make_shared<MfeWeight>(moe_up)};
    else if(moe_split_gate_up)gate_up={std::make_shared<MfeWeight>(moe_gate),std::make_shared<MfeWeight>(moe_up)};
    else gate_up={std::make_shared<MfeWeight>(moe_gate_up)};
    MoeFfnPrefetch prefetch;
    moe_pipeline_forward=make_moe_ffn_pipeline(gate_up,{std::make_shared<MfeWeight>(moe_down)},
        {},{},{},&prefetch,swiglu_limit);
    if(prefetch)moe_pipeline_prefetch=[prefetch=std::move(prefetch)](const auto& ids) {
        auto task=prefetch(ids);
        if(auto* window=mfq::cuda::DecodeWindow::recording())window->enroll(std::move(task));
        else task.serve();
    };
    moe_pipeline_initialized=true;
    if(moe_pipeline_forward)std::fprintf(stderr,"moe_generic_pipeline_dispatch layer=%d clipped_swiglu=%g\n",
        moe_layer,swiglu_limit);
}

void FFN::prefetch_moe(CudaExecutionContext& execution,const mfq_tensor_backend::Tensor& x,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids) const {
    if(!moe_pipeline_prefetch || !moe_expert_cache_ssd_router_prefetch(execution.moe_expert_cache) ||
            x.dim()<2 || x.numel()/x.size(-1)>32)return;
    namespace tb=mfq_tensor_backend;
    tb::Tensor ids;
    // Cover each row's leading candidates without reading low-confidence
    // tails just because the leading experts are already cached.
    constexpr int64_t candidates_per_row=4;
    if(moe_hash_ids.defined()) {
        if(!input_ids.has_value() || !input_ids.value().numel())return;
        ids=moe_hash_ids.index_select(0,input_ids.value().reshape({-1}).to(tb::kCUDA,tb::kInt64));
        ids=ids.narrow(1,0,std::min(candidates_per_row,ids.size(1))).to(tb::kInt32).contiguous();
    }else {
        auto row=x.reshape({-1,x.size(-1)}).to(tb::kFloat32);
        auto logits=moe_router_projection?moe_router_projection->forward(execution,row).contiguous():
            tb::matmul(row,moe_router.transpose(0,1)).contiguous();
        ids=moe_topk_cuda(logits,std::min({int64_t(moe_top_k),logits.size(-1),candidates_per_row}),moe_use_sigmoid,
            moe_use_sqrt_softplus,moe_normalize,moe_delayed_softmax,
            moe_router_bias.defined()?MfqOptional<tb::Tensor>(moe_router_bias):mfq_nullopt,
            1e-20,moe_router_scale).at(0).to(tb::kInt32).contiguous();
    }
    // Visit each row's highest-ranked candidate before lower ranks, so one
    // token cannot consume the entire speculative read budget for a batch.
    static std::atomic<bool> reported{false};
    if(!reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))
        std::fprintf(stderr,"moe_ssd_router_candidates scope=first_forecast rows=%lld candidates_per_row=%lld\n",
            static_cast<long long>(ids.size(0)),static_cast<long long>(ids.size(1)));
    moe_pipeline_prefetch(ids.transpose(0,1).contiguous().reshape({1,-1}));
}

bool FFN::uses_moe_expert_cache() const {
    if (!is_moe)
        return false;
    if (moe_split_gate_up) {
        return moe_gate.cached_source || moe_up.cached_source || moe_down.cached_source;
    }
    return moe_gate_up.cached_source || moe_down.cached_source;
}

bool FFN::tensor_parallel_dense_compatible() const {
    if (is_moe || gate_up.layers.size() != 2 || !down.tensor_parallel() ||
        down.tensor_parallel_axis != TensorParallelAxis::Input) {
        return false;
    }
    const auto &gate = gate_up.layers[0];
    const auto &up = gate_up.layers[1];
    if (!gate.tensor_parallel() || !up.tensor_parallel() ||
        gate.tensor_parallel_axis != TensorParallelAxis::Output ||
        up.tensor_parallel_axis != TensorParallelAxis::Output ||
        gate.tensor_parallel_shards.size() != up.tensor_parallel_shards.size() ||
        gate.tensor_parallel_shards.size() != down.tensor_parallel_shards.size()) {
        return false;
    }
    for (size_t index = 0; index < gate.tensor_parallel_shards.size(); ++index) {
        const auto &gate_shard = gate.tensor_parallel_shards[index];
        const auto &up_shard = up.tensor_parallel_shards[index];
        const auto &down_shard = down.tensor_parallel_shards[index];
        if (gate_shard.device != up_shard.device || gate_shard.device != down_shard.device ||
            gate_shard.output_begin != up_shard.output_begin ||
            gate_shard.output_end != up_shard.output_end ||
            gate_shard.output_begin != down_shard.input_begin ||
            gate_shard.output_end != down_shard.input_end) {
            return false;
        }
    }
    return true;
}

mfq_tensor_backend::Tensor FFN::forward_tensor_parallel_dense(CudaExecutionContext &execution,
                                                              mfq_tensor_backend::Tensor xh) const {
    auto shape = xh.sizes().vec();
    auto flat = xh.reshape({-1, xh.size(-1)});
    const size_t shard_count = down.tensor_parallel_shards.size();
    std::vector<mfq_tensor_backend::Tensor> partials(shard_count);
    for (size_t launch_position = 0; launch_position < shard_count; ++launch_position) {
        const size_t index =
            model_parallel_launch_index(execution.config, launch_position, shard_count);
        const auto &gate_shard = gate_up.layers[0].tensor_parallel_shards[index];
        const auto &up_shard = gate_up.layers[1].tensor_parallel_shards[index];
        const auto &down_shard = down.tensor_parallel_shards[index];
        MfqCudaGuard guard(gate_shard.device);
        auto local_x =
            tensor_to_cuda_device(execution.model_parallel_collectives, flat, gate_shard.device);
        using Tensor = mfq_tensor_backend::Tensor;
        auto unfused = [](const auto &...) { return std::optional<Tensor>{}; };
        partials[index] = mfq::models::gated_mlp(
            local_x, geglu, swiglu_limit, unfused, unfused,
            [&](Tensor input) {
                auto gate =
                    run_quant_linear_shard(execution.profiler, execution.kl_mmq, gate_shard, input);
                auto up =
                    run_quant_linear_shard(execution.profiler, execution.kl_mmq, up_shard, input);
                return std::array<Tensor, 2>{std::move(gate), std::move(up)};
            },
            [&](Tensor gate_output, Tensor up_output, auto activation_kind, double limit) {
                mfq_tensor_backend::Tensor activation;
                if (activation_kind == mfq::models::GatedActivation::gelu) {
                    activation = gelu_mul_cuda(gate_output.contiguous(), up_output.contiguous());
                } else if (limit > 0.0) {
                    auto clipped_gate = mfq_tensor_backend::clamp_max(
                        gate_output.to(mfq_tensor_backend::kFloat32), limit);
                    auto clipped_up = mfq_tensor_backend::clamp(
                        up_output.to(mfq_tensor_backend::kFloat32), -limit, limit);
                    activation = (mfq_tensor_backend::silu(clipped_gate) * clipped_up)
                                     .to(mfq_tensor_backend::kFloat16)
                                     .contiguous();
                } else {
                    activation = (mfq_tensor_backend::silu(gate_output) * up_output).contiguous();
                }
                return activation;
            },
            [&](Tensor activation) {
                return run_quant_linear_shard(execution.profiler, execution.kl_mmq, down_shard,
                                              activation);
            },
            unfused);
    }
    auto output = reduce_model_parallel_outputs(execution, std::move(partials));
    shape.back() = output.size(-1);
    return output.reshape(shape);
}

bool FFN::expert_parallel_moe_compatible() const {
    if (!is_moe || moe_split_gate_up ||
        moe_gate_up.expert_parallel_shards.size() != moe_down.expert_parallel_shards.size() ||
        moe_gate_up.expert_parallel_shards.empty()) {
        return false;
    }
    for (size_t index = 0; index < moe_gate_up.expert_parallel_shards.size(); ++index) {
        const auto &gate = moe_gate_up.expert_parallel_shards[index];
        const auto &down = moe_down.expert_parallel_shards[index];
        if (!gate.weight || !down.weight || gate.device != down.device ||
            gate.expert_begin != down.expert_begin || gate.expert_end != down.expert_end) {
            return false;
        }
    }
    return true;
}

mfq_tensor_backend::Tensor
FFN::forward_expert_parallel_moe(CudaExecutionContext &execution, mfq_tensor_backend::Tensor x,
                                 const MoeRoutePlan &route,
                                 mfq_tensor_backend::Tensor route_weights) const {
    const size_t shard_count = moe_gate_up.expert_parallel_shards.size();
    std::vector<mfq_tensor_backend::Tensor> routed_partials(shard_count);
    std::vector<mfq_tensor_backend::Tensor> down_partials;
    const bool collect_output_energy =
        !execution.config.moe_route_stats_path.empty() && execution.config.moe_route_output_energy;
    if (collect_output_energy) {
        down_partials.resize(shard_count);
    }
    for (size_t launch_position = 0; launch_position < shard_count; ++launch_position) {
        const size_t index =
            model_parallel_launch_index(execution.config, launch_position, shard_count);
        const auto &gate_shard = moe_gate_up.expert_parallel_shards[index];
        const auto &down_shard = moe_down.expert_parallel_shards[index];
        MfqCudaGuard guard(gate_shard.device);
        auto local_x =
            tensor_to_cuda_device(execution.model_parallel_collectives, x, gate_shard.device);
        auto local_weights = tensor_to_cuda_device(execution.model_parallel_collectives,
                                                   route_weights, gate_shard.device);
        const auto &local_route =
            moe_route_to_device(execution.model_parallel_collectives, route, gate_shard.device);
        using Tensor = mfq_tensor_backend::Tensor;
        Tensor gate_up_pair;
        auto down_pair = mfq::models::gated_mlp(
            local_x, false, swiglu_limit, [](const auto &...) { return std::optional<Tensor>{}; },
            [&](const Tensor &input, auto, double limit) -> std::optional<Tensor> {
                if (limit <= 0.0 && !execution.force_moe_materialized_swiglu &&
                    gate_shard.weight->supports_projection_glu_epilogue())
                    return gate_shard.weight->forward_glu_output(execution, input, local_route,
                                                                 false);
                return {};
            },
            [&](Tensor input) {
                gate_up_pair = gate_shard.weight->forward(execution, input, local_route);
                const auto width = gate_up_pair.size(-1) / 2;
                return std::array<Tensor, 2>{gate_up_pair.narrow(-1, 0, width),
                                             gate_up_pair.narrow(-1, width, width)};
            },
            [&](Tensor gate, Tensor up, auto, double limit) {
                if (limit <= 0.0)
                    return moe_swiglu_split_cuda(gate_up_pair);
                gate = mfq_tensor_backend::clamp_max(gate.to(mfq_tensor_backend::kFloat32), limit);
                up = mfq_tensor_backend::clamp(up.to(mfq_tensor_backend::kFloat32), -limit, limit);
                return (mfq_tensor_backend::silu(gate) * up)
                    .to(mfq_tensor_backend::kFloat16)
                    .contiguous();
            },
            [&](Tensor hidden) {
                return down_shard.weight->forward(execution, hidden, local_route);
            },
            [&](const Tensor &, const Tensor &, auto, double limit) -> std::optional<Tensor> {
                const bool allow_fusion =
                    !execution.force_moe_materialized_swiglu &&
                    execution.config.moe_swiglu_quant_fusion &&
                    (gate_up_pair.size(0) == 1 ||
                     (gate_up_pair.size(0) <= 4 && execution.config.moe_small_heterogeneous));
                if (!allow_fusion)
                    return {};
                if (limit <= 0.0)
                    return down_shard.weight->forward_swiglu(execution, gate_up_pair, local_route);
                if (down_shard.weight->supports_clamped_swiglu())
                    return down_shard.weight->forward_clamped_swiglu(execution, gate_up_pair,
                                                                     local_route, limit);
                return {};
            });
        if (collect_output_energy) {
            down_partials[index] = down_pair;
        }
        routed_partials[index] = moe_weighted_reduce_cuda(down_pair, local_weights);
    }
    if (!execution.config.moe_route_stats_path.empty()) {
        mfq_tensor_backend::Tensor complete_down;
        if (collect_output_energy) {
            complete_down = reduce_model_parallel_outputs(execution, std::move(down_partials));
        }
        record_moe_route_stats(execution, moe_layer, route.ids, route_weights, complete_down,
                               moe_gate_up.n_experts);
    }
    return reduce_model_parallel_outputs(execution, std::move(routed_partials));
}

mfq_tensor_backend::Tensor FFN::forward_dense_f32_down_kld(CudaExecutionContext &execution,
                                                           mfq_tensor_backend::Tensor xh) const {
    auto &profiler = execution.profiler;
    MFQ_RUNTIME_CHECK(!is_moe && !tensor_parallel_dense_compatible() &&
                          activation == ActivationKind::silu_gated && !geglu &&
                          swiglu_limit <= 0.0,
                      "FP32-output IN diagnostic requires a local dense SiLU FFN");
    auto parts = profiler.measure("ffn.gate_up", [&]() { return gate_up.forward(execution, xh); });
    return profiler.measure("ffn.down.fp32_output", [&]() {
        return down.forward_input_mul_f32_kld(execution, parts[1], parts[0], 2);
    });
}

mfq_tensor_backend::Tensor FFN::forward_impl(CudaExecutionContext &execution,
                                             mfq_tensor_backend::Tensor x,
                                             MfqOptional<mfq_tensor_backend::Tensor> input_ids,
                                             bool allow_important_neurons,
                                             MfqOptional<mfq_tensor_backend::Tensor> router_bias) const {
    auto &profiler = execution.profiler;
    mfq_tensor_backend::Tensor xh;
    if (x.scalar_type() == mfq_tensor_backend::kFloat16) {
        xh = x;
    } else {
        xh = profiler.measure("ffn.input_cast",
                              [&]() { return x.to(mfq_tensor_backend::kFloat16); });
    }
    if (allow_important_neurons && important_neurons) {
        if (is_moe) {
            throw std::runtime_error("important-neuron branches are only supported for dense FFNs");
        }
        const int64_t rows = xh.numel() / xh.size(-1);
        const bool use_f32_down = rows >= 16 && execution.kl_mmq.mode == KlMmqMode::Fp16 &&
                                  execution.config.diagnostic_in_f32_down;
        return mfq::models::additive_branches(
            [&](size_t index) {
                const auto &branch = index == 0 ? *this : *important_neurons;
                return use_f32_down ? branch.forward_dense_f32_down_kld(execution, xh)
                                    : branch.forward_impl(execution, xh, input_ids, false);
            },
            [&](auto &run_branch, auto &outputs) {
                return !use_f32_down &&
                       decode_branch_parallel_enabled(
                           execution.config, execution.decode_graph_serial_branches, rows) &&
                       execution.config.important_neuron_branch_parallel &&
                       important_neuron_executor->run(2, run_branch, outputs);
            },
            [&](auto low, auto high) {
                if (use_f32_down)
                    return profiler.measure("ffn.in.combine.fp32", [&] {
                        return (low + high).to(mfq_tensor_backend::kFloat16).contiguous();
                    });
                return profiler.measure("ffn.in.combine", [&] {
                    return acc_cuda(low.contiguous(), high.contiguous());
                });
            });
    }
    if (tensor_parallel_dense_compatible()) {
        return profiler.measure("ffn.tensor_parallel",
                                [&]() { return forward_tensor_parallel_dense(execution, xh); });
    }
    if (is_moe) {
        const bool tq_routed=bool(moe_tq);
        const int64_t rows = xh.numel() / xh.size(-1);
        if (execution.continuous_batch_cache_serial && uses_moe_expert_cache() && rows > 1) {
            // A bounded expert cache cannot safely admit the union of an
            // arbitrary request batch: a miss in that union otherwise
            // falls back to staging the complete projection. Keep the
            // outer request/KV/attention batch intact and execute only
            // each routed FFN row independently, so cache admission is
            // bounded by one token's top-k experts.
            auto flat = xh.reshape({rows, xh.size(-1)}).contiguous();
            std::vector<mfq_tensor_backend::Tensor> outputs;
            outputs.reserve(static_cast<size_t>(rows));
            for (int64_t row = 0; row < rows; ++row) {
                auto row_input = flat.narrow(0, row, 1).contiguous();
                if (input_ids.has_value()) {
                    MFQ_RUNTIME_CHECK(input_ids.value().numel() == rows,
                                      "continuous-batch MoE token ids do not match rows");
                    auto row_ids = input_ids.value().reshape({rows}).narrow(0, row, 1).contiguous();
                    auto bias=router_bias;
                    if(bias.has_value() && bias->dim()==2)bias=bias->narrow(0,row,1);
                    outputs.push_back(forward_impl(execution, row_input, row_ids, false,bias));
                } else {
                    auto bias=router_bias;
                    if(bias.has_value() && bias->dim()==2)bias=bias->narrow(0,row,1);
                    outputs.push_back(forward_impl(execution, row_input, mfq_nullopt, false,bias));
                }
            }
            return mfq_tensor_backend::cat(outputs, 0).reshape(xh.sizes());
        }
        if (!shared || moe_top_k <= 0 || (!moe_router.defined() && !moe_router_projection) ||
            (!moe_shared_ungated && !moe_shared_gate.defined())) {
            throw std::runtime_error("incomplete MoE FFN state");
        }
        auto xf = xh.reshape({-1, xh.size(-1)}).contiguous();
        auto xf32 = profiler.measure("moe.input_f32",
                                     [&]() { return xf.to(mfq_tensor_backend::kFloat32); });
        using Tensor = mfq_tensor_backend::Tensor;
        MoeRoutePlan route;
        Tensor shared_gate_logits;
        bool projection_bundle_prefetched = false;
        const bool expert_parallel = expert_parallel_moe_compatible();
        const bool use_pipeline=bool(moe_pipeline_forward) &&
            execution.kl_mmq.mode==KlMmqMode::Default && !execution.force_moe_materialized_swiglu;
        struct ExpertOutput {
            Tensor values;
            bool reduced;
        };
        // Start the tiny route readback before the shared expert.  The
        // cache consumes it after the shared branch has been enqueued,
        // then launches H2D on its separate stream.  The projection
        // forward below still performs the normal cache check and stream
        // wait, preserving the existing execution semantics.
        auto prefetch_projection_bundle = [&]() {
            if (use_pipeline || !execution.config.moe_projection_bundle_prefetch || cpu_moe_down ||
                execution.continuous_batch_cache_serial) {
                return false;
            }
            if (moe_ungated_experts) {
                if (cpu_moe_up || cpu_moe_down)
                    return false;
                return prefetch_cached_moe_projection_bundle(moe_up, moe_down, route);
            }
            if (moe_split_gate_up) {
                if (cpu_moe_gate || cpu_moe_up)
                    return false;
                return prefetch_cached_moe_projection_bundle(moe_gate, moe_up, moe_down, route);
            }
            if (cpu_moe_gate_up)
                return false;
            return prefetch_cached_moe_projection_bundle(moe_gate_up, moe_down, route);
        };

        auto shared_gate = [&] {
            if (!shared_gate_logits.defined())
                shared_gate_logits = profiler.measure("moe.shared_gate", [&] {
                    return mfq_tensor_backend::matmul(xf32, moe_shared_gate.transpose(0, 1))
                        .contiguous();
                });
            return shared_gate_logits;
        };
        return mfq::models::mixture_of_experts(
            [&] {
                return profiler.measure("moe.router", [&]() {
                    return moe_router_projection?moe_router_projection->forward(execution,xf32):
                        mfq_tensor_backend::matmul(xf32, moe_router.transpose(0, 1));
                });
            },
            [&](Tensor router_logits) {
                std::vector<mfq_tensor_backend::Tensor> selected;
                if (moe_hash_ids.defined()) {
                    if (!input_ids.has_value()) {
                        throw std::runtime_error("hash-routed MoE requires the current token ids");
                    }
                    selected = profiler.measure("moe.hash_route", [&]() {
                        auto ids = moe_hash_ids
                                       .index_select(0, input_ids.value().reshape({-1}).to(
                                                            mfq_tensor_backend::kCUDA,
                                                            mfq_tensor_backend::kInt64))
                                       .to(mfq_tensor_backend::kInt32)
                                       .contiguous();
                        auto weights = moe_sqrtsoftplus_weights_cuda(router_logits.contiguous(),
                                                                     ids, 1e-20, moe_router_scale);
                        return std::vector<mfq_tensor_backend::Tensor>{ids, weights};
                    });
                } else {
                    selected = profiler.measure("moe.topk", [&]() {
                        return moe_topk_cuda(
                            router_logits.contiguous(), moe_top_k, moe_use_sigmoid,
                            moe_use_sqrt_softplus, moe_normalize, moe_delayed_softmax,
                            router_bias.has_value()?router_bias:moe_router_bias.defined()
                                ? MfqOptional<mfq_tensor_backend::Tensor>(moe_router_bias)
                                : mfq_nullopt,
                            1e-20, moe_router_scale);
                    });
                }
                mfq::cuda::mask_batch_routes(execution, selected);
                if (!use_pipeline && !tq_routed) {
                    // Dense (source-precision) banks are not MFE cohorts, so
                    // their expert count comes from the bank itself.
                    const auto routed_experts =
                        (moe_dense_up.defined() || moe_dense_gate_up.defined())
                            ? static_cast<int>((moe_dense_up.defined() ? moe_dense_up : moe_dense_gate_up).size(0))
                            : (moe_ungated_experts
                                   ? moe_up.n_experts
                                   : (moe_split_gate_up ? moe_gate.n_experts
                                                        : moe_gate_up.n_experts));
                    route = profiler.measure("moe.route_map", [&]() {
                        return build_moe_route_plan(selected.at(0), routed_experts);
                    });
                }
                return selected;
            },
            [&](const auto &) {
                if (!expert_parallel && !use_pipeline && !tq_routed) {
                    if (moe_split_gate_up) {
                        if (execution.config.moe_delayed_route_readback) {
                            moe_gate.prefetch_begin(route);
                        } else {
                            projection_bundle_prefetched = prefetch_projection_bundle();
                            if (!projection_bundle_prefetched) {
                                moe_gate.prefetch(route);
                            }
                        }
                    } else {
                        if (execution.config.moe_delayed_route_readback) {
                            moe_gate_up.prefetch_begin(route);
                        } else {
                            projection_bundle_prefetched = prefetch_projection_bundle();
                            if (!projection_bundle_prefetched) {
                                moe_gate_up.prefetch(route);
                            }
                        }
                    }
                }
                auto shared_output = profiler.measure(
                    "moe.shared", [&]() { return shared->forward(execution, xf); });
                if (!moe_split_gate_up && !moe_shared_ungated) {
                    shared_gate_logits = profiler.measure("moe.shared_gate", [&]() {
                        return mfq_tensor_backend::matmul(xf32, moe_shared_gate.transpose(0, 1))
                            .contiguous();
                    });
                }
                if (!expert_parallel && !use_pipeline && !tq_routed && execution.config.moe_delayed_route_readback) {
                    projection_bundle_prefetched = prefetch_projection_bundle();
                    if (!projection_bundle_prefetched) {
                        if (moe_split_gate_up) {
                            moe_gate.prefetch(route);
                        } else {
                            moe_gate_up.prefetch(route);
                        }
                    }
                }
                return shared_output.reshape({xf.size(0), xf.size(1)})
                    .contiguous()
                    .to(mfq_tensor_backend::kFloat16);
            },
            [&](const auto &selected) -> ExpertOutput {
                const bool dense_routed =
                    moe_dense_up.defined() || moe_dense_down.defined() ||
                    moe_dense_gate.defined() || moe_dense_gate_up.defined();
                // Only routed experts run in the latent space; the router and
                // the shared expert keep reading the original hidden state.
                const bool latent_routed =
                    !dense_routed && dense_linear_defined(moe_latent_in);
                auto routed_input = xf;
                if (latent_routed)
                    routed_input = profiler.measure("moe.latent_down", [&] {
                        return moe_latent_in.forward(execution, xf);
                    });
                const auto finish_latent = [&](Tensor reduced) {
                    if (latent_routed)
                        return profiler.measure("moe.latent_up", [&] {
                            return moe_latent_out.forward(execution, reduced);
                        });
                    return reduced;
                };
                if(tq_routed) {
                    MFQ_RUNTIME_CHECK(moe_tq->up && moe_tq->down,
                        "TQ routed FFN requires matching Up and Down projections");
                    if(!moe_ungated_experts && activation==ActivationKind::silu_gated) {
                        auto reduced=profiler.measure("moe.tq",[&] {
                            return moe_tq->forward(routed_input,selected.at(0),selected.at(1),
                                static_cast<float>(swiglu_limit),&profiler);
                        });
                        return {finish_latent(std::move(reduced)),true};
                    }
                    Tensor gate,up;
                    if(moe_ungated_experts) {
                        MFQ_RUNTIME_CHECK(!moe_tq->gate,"ungated TQ experts cannot carry a Gate tensor");
                    }else {
                        MFQ_RUNTIME_CHECK(moe_tq->gate,"gated TQ experts require a Gate tensor");
                        gate=moe_tq->gate->forward(routed_input,selected.at(0));
                    }
                    up=moe_tq->up->forward(routed_input,selected.at(0));
                    auto hidden=dense_expert_activation(activation,std::move(gate),std::move(up),swiglu_limit);
                    auto pairs=moe_tq->down->forward(hidden,selected.at(0));
                    auto reduced=moe_weighted_reduce_cuda(pairs,selected.at(1));
                    return {finish_latent(std::move(reduced)),true};
                }
                if (dense_routed) {
                    if (moe_ungated_experts &&
                        (moe_dense_gate.defined() || moe_dense_gate_up.defined() || moe_gate_up.n_experts > 0 ||
                         moe_gate.n_experts > 0 || moe_up.n_experts > 0 ||
                         moe_gate_up.cached_source || moe_up.cached_source))
                        throw std::runtime_error(
                            "ungated routed MoE cannot mix a gate with a dense expert bank");
                    MFQ_RUNTIME_CHECK((moe_dense_up.defined() || moe_dense_gate_up.defined()) &&
                                      moe_dense_down.defined(),
                                      "dense routed MoE requires Up and Down banks");
                    auto expert_input = xf;
                    if (dense_linear_defined(moe_latent_in))
                        expert_input = profiler.measure("moe.latent_down", [&] {
                            return moe_latent_in.forward(execution, xf);
                        });
                    Tensor up;
                    Tensor gate;
                    if (moe_dense_gate_up.defined()) {
                        MFQ_RUNTIME_CHECK(!moe_dense_gate.defined() && !moe_dense_up.defined() &&
                                          moe_dense_gate_up.size(1) % 2 == 0,
                                          "dense combined Gate/Up cannot mix separate banks and needs even rows");
                        auto pair = profiler.measure("moe.dense_gate_up", [&] {
                            return dense_routed_projection(moe_dense_gate_up, expert_input, route);
                        });
                        const auto width = pair.size(-1) / 2;
                        gate = pair.narrow(-1, 0, width);
                        up = pair.narrow(-1, width, width);
                    } else {
                        up = profiler.measure("moe.dense_up", [&] {
                            return dense_routed_projection(moe_dense_up, expert_input, route);
                        });
                        if (!moe_ungated_experts) {
                            MFQ_RUNTIME_CHECK(moe_dense_gate.defined(),
                                              "gated dense routed MoE requires a Gate bank");
                            gate = profiler.measure("moe.dense_gate", [&] {
                                return dense_routed_projection(moe_dense_gate, expert_input, route);
                            });
                        }
                    }
                    auto hidden = profiler.measure("moe.dense_activation", [&] {
                        return dense_expert_activation(activation, std::move(gate),
                                                       std::move(up), swiglu_limit);
                    });
                    auto down_pair = profiler.measure("moe.dense_down", [&] {
                        return dense_routed_projection(moe_dense_down, hidden, route);
                    });
                    auto reduced = profiler.measure("moe.dense_reduce", [&] {
                        return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                    });
                    if (dense_linear_defined(moe_latent_out))
                        reduced = profiler.measure("moe.latent_up", [&] {
                            return moe_latent_out.forward(execution, reduced);
                        });
                    record_moe_route_stats(execution, moe_layer, selected.at(0), selected.at(1),
                                           down_pair, route.n_experts);
                    return {reduced, true};
                }
                if (moe_ungated_experts && !use_pipeline) {
                    // relu-squared experts store Up and Down only: one Up matmul,
                    // no gate projection, and the shared expert stays outside
                    // the routed latent space.
                    if (moe_gate.n_experts > 0 || moe_gate_up.n_experts > 0 ||
                        moe_gate.cached_source || moe_gate_up.cached_source)
                        throw std::runtime_error(
                            "ungated routed MoE cannot carry a gate projection");
                    MFQ_RUNTIME_CHECK(moe_up.n_experts > 0,
                                      "ungated routed MoE requires an Up bank");
                    if (!projection_bundle_prefetched) {
                        moe_up.prefetch(route);
                        moe_down.prefetch(route);
                    }
                    std::optional<MfeWeight> staged_up;
                    const MfeWeight *active_up = &moe_up;
                    if (cpu_moe_up) {
                        staged_up.emplace(profiler.measure("moe.cpu_offload_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_up, execution.config);
                        }));
                        active_up = &staged_up.value();
                    }
                    auto up = profiler.measure("moe.up", [&]() {
                        return active_up->forward(execution, routed_input, route);
                    });
                    auto hidden = profiler.measure("moe.relu_squared", [&]() {
                        return relu_squared_cuda(up.contiguous());
                    });
                    staged_up.reset();
                    std::optional<MfeWeight> staged_down;
                    const MfeWeight *active_down = &moe_down;
                    if (cpu_moe_down) {
                        staged_down.emplace(profiler.measure("moe.cpu_offload_down_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_down, execution.config);
                        }));
                        active_down = &staged_down.value();
                    }
                    auto down_pair = profiler.measure("moe.down", [&] {
                        return active_down->forward(execution, hidden, route);
                    });
                    record_moe_route_stats(execution, moe_layer, selected.at(0),
                                           selected.at(1), down_pair, moe_up.n_experts);
                    staged_down.reset();
                    if (latent_routed)
                        return {finish_latent(profiler.measure("moe.reduce", [&] {
                                    return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                                })),
                                true};
                    return {down_pair, false};
                }
                if(use_pipeline) {
                    auto result=moe_pipeline_forward(execution,routed_input,selected.at(0),
                                                     selected.at(1));
                    if(result.dim()==3)result=moe_weighted_reduce_cuda(result,selected.at(1));
                    return {finish_latent(result),true};
                }
                if (expert_parallel) {
                    auto result = profiler.measure("moe.expert_parallel", [&] {
                        return forward_expert_parallel_moe(execution, routed_input, route,
                                                           selected.at(1));
                    });
                    if (latent_routed && result.dim() == 3)
                        result = moe_weighted_reduce_cuda(result, selected.at(1));
                    return {finish_latent(result), true};
                }
                std::optional<MfeWeight> staged_gate_up;
                std::optional<MfeWeight> staged_gate;
                std::optional<MfeWeight> staged_up;
                const MfeWeight *active_gate_up = &moe_gate_up;
                const MfeWeight *active_gate = &moe_gate;
                const MfeWeight *active_up = &moe_up;
                mfq_tensor_backend::Tensor gate_up_pair;
                mfq_tensor_backend::Tensor projected_hidden;
                if (moe_split_gate_up) {
                    if (cpu_moe_gate) {
                        staged_gate.emplace(profiler.measure("moe.cpu_offload_gate_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_gate, execution.config);
                        }));
                        active_gate = &staged_gate.value();
                    }
                    if (cpu_moe_up) {
                        staged_up.emplace(profiler.measure("moe.cpu_offload_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_up, execution.config);
                        }));
                        active_up = &staged_up.value();
                    }
                    gate_up_pair = profiler.measure("moe.gate_up_split", [&]() {
                        const bool reuse_gate_activation =
                            execution.config.split_moe_activation_reuse &&
                            execution.kl_mmq.mode == KlMmqMode::Default && xf.size(0) <= 8 &&
                            active_gate->can_reuse_activation_for(*active_up);
                        auto gate = active_gate->forward(execution, xf, route);
                        if (!projection_bundle_prefetched) {
                            active_up->prefetch(route);
                        }
                        if (!moe_shared_ungated && !shared_gate_logits.defined()) {
                            shared_gate_logits = profiler.measure("moe.shared_gate", [&]() {
                                return mfq_tensor_backend::matmul(xf32,
                                                                  moe_shared_gate.transpose(0, 1))
                                    .contiguous();
                            });
                        }
                        auto up = reuse_gate_activation
                                      ? active_up->forward_prequantized(execution, xf, route)
                                      : active_up->forward(execution, xf, route);
                        return mfq_tensor_backend::cat({gate, up}, -1).contiguous();
                    });
                } else {
                    if (cpu_moe_gate_up) {
                        staged_gate_up.emplace(
                            profiler.measure("moe.cpu_offload_gate_up_h2d", [&]() {
                                return stage_cpu_mixed_moe(cpu_moe_gate_up, execution.config);
                            }));
                        active_gate_up = &staged_gate_up.value();
                    }
                    const bool fuse_projection_glu =
                        swiglu_limit <= 0.0 && !execution.force_moe_materialized_swiglu &&
                        active_gate_up->supports_projection_glu_epilogue();
                    if (fuse_projection_glu) {
                        projected_hidden = profiler.measure("moe.gate_up_swiglu", [&]() {
                            return active_gate_up->forward_glu_output(execution, xf, route, false);
                        });
                    } else {
                        gate_up_pair = profiler.measure("moe.gate_up", [&]() {
                            return active_gate_up->forward(execution, xf, route);
                        });
                    }
                }
                staged_gate_up.reset();
                staged_gate.reset();
                staged_up.reset();
                if (!projection_bundle_prefetched) {
                    moe_down.prefetch(route);
                }
                std::optional<MfeWeight> staged_down;
                const MfeWeight *active_down = &moe_down;
                if (cpu_moe_down) {
                    staged_down.emplace(profiler.measure("moe.cpu_offload_down_h2d", [&]() {
                        return stage_cpu_mixed_moe(cpu_moe_down, execution.config);
                    }));
                    active_down = &staged_down.value();
                }
                auto down_pair = mfq::models::gated_mlp(
                    xf, false, swiglu_limit,
                    [](const auto &...) { return std::optional<Tensor>{}; },
                    [&](const Tensor &, auto, double) -> std::optional<Tensor> {
                        if (projected_hidden.defined())
                            return projected_hidden;
                        return {};
                    },
                    [&](const Tensor &) {
                        const auto width = gate_up_pair.size(-1) / 2;
                        return std::array<Tensor, 2>{gate_up_pair.narrow(-1, 0, width),
                                                     gate_up_pair.narrow(-1, width, width)};
                    },
                    [&](const Tensor &, const Tensor &, auto, double limit) {
                        return profiler.measure("moe.swiglu", [&]() {
                            if (limit <= 0.0) {
                                return moe_swiglu_split_cuda(gate_up_pair);
                            }
                            const int64_t width = gate_up_pair.size(-1) / 2;
                            auto gate = mfq_tensor_backend::clamp_max(
                                gate_up_pair.slice(-1, 0, width).to(mfq_tensor_backend::kFloat32),
                                limit);
                            auto up =
                                mfq_tensor_backend::clamp(gate_up_pair.slice(-1, width, 2 * width)
                                                              .to(mfq_tensor_backend::kFloat32),
                                                          -limit, limit);
                            return (mfq_tensor_backend::silu(gate) * up)
                                .to(mfq_tensor_backend::kFloat16)
                                .contiguous();
                        });
                    },
                    [&](Tensor hidden) {
                        return profiler.measure("moe.down", [&] {
                            return active_down->forward(execution, hidden, route);
                        });
                    },
                    [&](const Tensor &, const Tensor &, auto,
                        double limit) -> std::optional<Tensor> {
                        const bool allow_fusion = !execution.force_moe_materialized_swiglu &&
                                                  execution.config.moe_swiglu_quant_fusion &&
                                                  (gate_up_pair.size(0) == 1 ||
                                                   (gate_up_pair.size(0) <= 4 &&
                                                    execution.config.moe_small_heterogeneous));
                        if (!allow_fusion)
                            return {};
                        if (limit <= 0.0)
                            return profiler.measure("moe.swiglu_down", [&] {
                                return active_down->forward_swiglu(execution, gate_up_pair, route);
                            });
                        if (active_down->supports_clamped_swiglu())
                            return profiler.measure("moe.swiglu_down", [&] {
                                return active_down->forward_clamped_swiglu(execution, gate_up_pair,
                                                                           route, limit);
                            });
                        return {};
                    });
                record_moe_route_stats(
                    execution, moe_layer, selected.at(0), selected.at(1), down_pair,
                    moe_split_gate_up ? moe_gate.n_experts : moe_gate_up.n_experts);
                staged_down.reset();
                return {down_pair, false};
            },
            [&](ExpertOutput experts, Tensor shared_half, const auto &selected) {
                return mfq::models::combine_experts(
                    experts.values, shared_half, experts.reduced, moe_shared_ungated,
                    [&](Tensor pairs, Tensor shared) -> std::optional<Tensor> {
                        if (execution.force_moe_unfused_reduce ||
                            !execution.config.moe_reduce_gate_fusion || moe_shared_ungated ||
                            pairs.size(0) > 8)
                            return {};
                        auto logits = shared_gate();
                        return profiler.measure("moe.reduce_combine", [&] {
                            return moe_weighted_reduce_shared_gate_cuda(pairs, selected.at(1),
                                                                        shared, logits);
                        });
                    },
                    [&](Tensor pairs) {
                        return profiler.measure("moe.reduce", [&] {
                            return moe_weighted_reduce_cuda(pairs, selected.at(1));
                        });
                    },
                    shared_gate,
                    [&](Tensor routed, Tensor shared) {
                        return profiler.measure("moe.combine", [&] {
                            // Dense BF16 expert banks may return a different storage
                            // type than the shared branch; align without changing it.
                            if (routed.scalar_type() != shared.scalar_type())
                                shared = shared.to(routed.scalar_type()).contiguous();
                            return acc_cuda(routed.contiguous(), shared);
                        });
                    },
                    [&](Tensor routed, Tensor shared, Tensor logits) {
                        return profiler.measure("moe.combine", [&] {
                            return moe_add_shared_gate_cuda(routed.contiguous(), shared, logits);
                        });
                    });
            });
    }
    using Tensor = mfq_tensor_backend::Tensor;
    using Activation = mfq::models::GatedActivation;
    if (activation == ActivationKind::relu_squared) {
        MFQ_RUNTIME_CHECK(!important_neurons && !tensor_parallel_dense_compatible(),
                          "relu-squared FFN does not support important neurons or tensor parallelism");
        auto parts = profiler.measure("ffn.up", [&]() { return gate_up.forward(execution, xh); });
        MFQ_RUNTIME_CHECK(parts.size() == 1,
                          "relu-squared FFN requires exactly one Up projection");
        auto hidden = profiler.measure("ffn.relu_squared", [&]() {
            return relu_squared_cuda(parts[0].contiguous());
        });
        return profiler.measure("ffn.down",
                                [&]() { return down.forward(execution, hidden); });
    }
    return mfq::models::gated_mlp(
        xh, geglu, swiglu_limit,
        [&](const Tensor &input, Activation activation, double limit) -> std::optional<Tensor> {
            if (activation != Activation::silu || limit > 0.0)
                return {};
            if (nvq_fusion_enabled(execution.config) && input.numel() / input.size(-1) == 1 &&
                gate_up.nvq_prefix2 && gate_up.layers.size() == 2 && gate_up.layers[0].is_nvq() &&
                gate_up.layers[1].is_nvq() && down.is_nvq() && gate_up.outs.size() == 2 &&
                gate_up.outs[0] == gate_up.outs[1] && gate_up.outs[0] == down.nvq.neuron_len &&
                (down.nvq.gs == 24 || down.nvq.gs == 28 || down.nvq.gs == 32)) {
                auto shape = input.sizes().vec();
                shape.back() = down.nvq.out;
                return nvq_ffn_swiglu_down(profiler, gate_up.layers[0].nvq, gate_up.layers[1].nvq,
                                           down.nvq, input.reshape({-1, input.size(-1)}))
                    .reshape(shape);
            }
            return {};
        },
        [&](const Tensor &input, Activation activation, double limit) -> std::optional<Tensor> {
            if(limit<=0.0 && activation==Activation::silu && gate_up.tq.defined() &&
                    gate_up.outs.size()==2 && gate_up.outs[0]==gate_up.outs[1])
                return profiler.measure("ffn.gate_up_swiglu",[&] {return gate_up.forward_swiglu(execution,input);});
            if (limit > 0.0 || !gate_up.nint_grouped || !gate_up.nint.split_w.empty())
                return {};
            const auto rows = input.numel() / input.size(-1);
            if (activation == Activation::gelu && execution.config.ffn_geglu_fusion && rows == 1)
                return profiler.measure("ffn.gate_up_geglu",
                                        [&] { return gate_up.forward_geglu(execution, input); });
            if (activation == Activation::silu && execution.config.ffn_swiglu_fusion && rows >= 1 &&
                rows <= 6 && gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1])
                return profiler.measure("ffn.gate_up_swiglu",
                                        [&] { return gate_up.forward_swiglu(execution, input); });
            return {};
        },
        [&](Tensor input) {
            return profiler.measure("ffn.gate_up",
                                    [&] { return gate_up.forward(execution, input); });
        },
        [&](Tensor gate, Tensor up, Activation activation, double limit) {
            if (activation == Activation::gelu)
                return profiler.measure(
                    "ffn.geglu", [&] { return gelu_mul_cuda(gate.contiguous(), up.contiguous()); });
            if (limit > 0.0)
                return profiler.measure("ffn.swiglu_clamped", [&] {
                    auto g =
                        mfq_tensor_backend::clamp_max(gate.to(mfq_tensor_backend::kFloat32), limit);
                    auto u = mfq_tensor_backend::clamp(up.to(mfq_tensor_backend::kFloat32), -limit,
                                                       limit);
                    return (mfq_tensor_backend::silu(g) * u)
                        .to(mfq_tensor_backend::kFloat16)
                        .contiguous();
                });
            return profiler.measure(
                "ffn.swiglu", [&] { return (up * mfq_tensor_backend::silu(gate)).contiguous(); });
        },
        [&](Tensor hidden) {
            return profiler.measure("ffn.down", [&] { return down.forward(execution, hidden); });
        },
        [&](const Tensor &gate, const Tensor &up, Activation activation,
            double limit) -> std::optional<Tensor> {
            if (activation != Activation::silu || limit > 0.0 || down.is_dense() || down.is_mxfp8())
                return {};
            return profiler.measure("ffn.down",
                                    [&] { return down.forward_input_mul(execution, up, gate, 2); });
        });
}

bool FFN::can_forward_fused_residual(const CudaExecutionConfig &config,
                                     const mfq_tensor_backend::Tensor &x,
                                     const mfq_tensor_backend::Tensor &residual) const {
    if (config.diagnostic_fp32_residual)
        return false;
    if (!nvq_fusion_enabled(config) || is_moe || geglu || swiglu_limit > 0.0 || important_neurons ||
        tensor_parallel_dense_compatible() || !x.is_cuda() || !residual.is_cuda() ||
        x.scalar_type() != mfq_tensor_backend::kFloat16 ||
        residual.scalar_type() != mfq_tensor_backend::kFloat16 || !x.is_contiguous() ||
        !residual.is_contiguous() || x.dim() < 1 || residual.dim() < 1 ||
        x.numel() / x.size(-1) != 1 || residual.numel() / residual.size(-1) != 1 ||
        gate_up.layers.size() != 2 || !gate_up.nvq_prefix2 || !gate_up.layers[0].is_nvq() ||
        !gate_up.layers[1].is_nvq() || !down.is_nvq() || gate_up.layers[0].tensor_parallel() ||
        gate_up.layers[1].tensor_parallel() || down.tensor_parallel() || gate_up.outs.size() != 2 ||
        gate_up.outs[0] != gate_up.outs[1] || gate_up.outs[0] != down.nvq.neuron_len ||
        x.size(-1) != gate_up.layers[0].nvq.neuron_len || residual.size(-1) != down.nvq.out) {
        return false;
    }
    return nvq_fused_residual_format(down.nvq.kernel_format);
}

mfq_tensor_backend::Tensor FFN::forward_fused_residual(CudaProfiler &profiler,
                                                       const CudaExecutionConfig &config,
                                                       mfq_tensor_backend::Tensor x,
                                                       mfq_tensor_backend::Tensor residual) const {
    MFQ_RUNTIME_CHECK(can_forward_fused_residual(config, x, residual),
                      "FFN fused residual requires a compatible single-token NVQ FFN");
    auto shape = x.sizes().vec();
    shape.back() = down.nvq.out;
    auto output =
        nvq_ffn_swiglu_down(profiler, gate_up.layers[0].nvq, gate_up.layers[1].nvq, down.nvq,
                            x.reshape({1, x.size(-1)}), residual.reshape({1, residual.size(-1)}));
    return output.reshape(shape);
}

mfq_tensor_backend::Tensor FFN::forward(CudaExecutionContext &execution,
                                        mfq_tensor_backend::Tensor x,
                                        MfqOptional<mfq_tensor_backend::Tensor> input_ids) const {
    return forward_impl(execution, std::move(x), input_ids, true);
}

mfq_tensor_backend::Tensor FFN::forward_with_router_bias(CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,mfq_tensor_backend::Tensor input_ids,
        const mfq_tensor_backend::Tensor& bias) const {
    MFQ_RUNTIME_CHECK(is_moe && !moe_hash_ids.defined() && bias.defined() &&
        ((bias.dim()==1 && bias.size(0)==(moe_router_projection?moe_router_projection->out():moe_router.size(0))) ||
         (bias.dim()==2 && bias.size(0)==x.numel()/x.size(-1) && bias.size(1)==(moe_router_projection?moe_router_projection->out():moe_router.size(0)))),
                      "MoE router correction bias shape mismatch");
    return forward_impl(execution,std::move(x),std::move(input_ids),true,bias);
}

void prepare_ffn_workspaces(CudaExecutionContext &execution, FFN &f) {
    if (execution.loading_cpu_layer)
        return;
    if (f.down.tensor_parallel())
        return;
    if (f.gate_up.nvq_prefix2 && f.gate_up.layers.size() == 2 && f.down.is_nvq() &&
        f.gate_up.outs.size() == 2 && f.gate_up.outs[0] == f.gate_up.outs[1] &&
        f.gate_up.outs[0] == f.down.nvq.neuron_len) {
        NvqWorkspace &ws = f.gate_up.layers[0].nvq.workspace(1);
        ws.swiglu_scratch = mfq_tensor_backend::empty({f.gate_up.outs[0]},
                                                      mfq_tensor_backend::TensorOptions()
                                                          .device(mfq_tensor_backend::kCUDA)
                                                          .dtype(mfq_tensor_backend::kFloat32));
        (void)f.down.nvq.workspace(1);
    }
    if (f.important_neurons) {
        prepare_ffn_workspaces(execution, *f.important_neurons);
    }
}
