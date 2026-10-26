#include "storage/weight_loader.h"
#include "model_checks.h"

#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm_dsa/ops.h"
#include "quant_linear.h"
#include "cuda_execution.h"
#include "storage/moe_expert_cache.h"
#include "mfq_cuda_moe_ops.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static double mfe_weight_bytes(const MfeWeight & weight) {
    double bytes = static_cast<double>(weight.mixed_weight_bytes);
    for (const auto & shard :
         weight.expert_parallel_shards) {
        if (shard.weight) {
            bytes += mfe_weight_bytes(
                *shard.weight);
        }
    }
    for (const auto & pool : weight.pools) {
        bytes += static_cast<double>(pool.weight.q_packed.numel()) * pool.weight.q_packed.element_size();
        bytes += static_cast<double>(pool.weight.row_q_bits.numel()) * pool.weight.row_q_bits.element_size();
        bytes += static_cast<double>(pool.weight.row_q_bit_offsets.numel()) * pool.weight.row_q_bit_offsets.element_size();
        bytes += static_cast<double>(pool.weight.sub_scale.numel()) * pool.weight.sub_scale.element_size();
        bytes += static_cast<double>(pool.weight.sub_min.numel()) * pool.weight.sub_min.element_size();
        bytes += static_cast<double>(pool.weight.neuron_scale.numel()) * pool.weight.neuron_scale.element_size();
        bytes += static_cast<double>(pool.weight.neuron_min.numel()) * pool.weight.neuron_min.element_size();
    }
    return bytes;
}

int run_expert_parallel_moe_check(
        CudaExecutionContext& execution,
        const std::string & model_path,
        const std::string & tensor_name,
        int tokens,
        int routes) {
    MFQ_RUNTIME_CHECK(
        moe_parallel_config(execution).enabled(),
        "--check-ep-moe requires --expert-parallel or --tensor-parallel");
    MFQ_RUNTIME_CHECK(
        tokens >= 1 && tokens <= 4096,
        "--check-ep-moe-tokens must be in [1, 4096]");
    MFQ_RUNTIME_CHECK(
        routes >= 1,
        "--check-ep-moe-routes must be positive");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    const std::string role =
        tensor_name.find("gate_up") != std::string::npos
        ? "gate_up" : "diagnostic";
    const ParallelConfig saved_tensor = execution.tensor_parallel;
    const ParallelConfig saved_expert = execution.expert_parallel;
    const ParallelConfig saved = moe_parallel_config(execution);
    execution.tensor_parallel = {};
    execution.expert_parallel = {};
    execution.expert_parallel.devices = {saved.primary_device()};
    MfeWeight full;
    try {
        full = load_mfe_gpu(execution,
            mfq, tensor_name, false, 0, role);
    } catch (...) {
        execution.tensor_parallel = saved_tensor;
        execution.expert_parallel = saved_expert;
        throw;
    }
    execution.tensor_parallel = saved_tensor;
    execution.expert_parallel = saved_expert;
    auto sharded = load_mfe_gpu(execution,
        mfq, tensor_name, false, 0, role);
    MFQ_RUNTIME_CHECK(
        sharded.expert_parallel(),
        "expert-parallel MoE diagnostic did not create shards");
    MFQ_RUNTIME_CHECK(
        full.n_experts == sharded.n_experts &&
        full.out_per_expert == sharded.out_per_expert &&
        full.neuron_len == sharded.neuron_len,
        "expert-parallel MoE metadata differs");
    MFQ_RUNTIME_CHECK(
        routes <= full.n_experts,
        "--check-ep-moe-routes exceeds the expert count");

    MfqCudaGuard primary_guard(
        saved.primary_device());
    const int64_t count =
        static_cast<int64_t>(tokens) *
        full.neuron_len;
    auto sequence = mfq_tensor_backend::arange(
        count,
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                saved.primary_device()))
            .dtype(mfq_tensor_backend::kFloat32));
    auto x = (
        (sequence.remainder(257) - 128.0) / 127.0 +
        0.03125 * mfq_tensor_backend::sin(sequence * 0.015625))
        .to(mfq_tensor_backend::kFloat16)
        .reshape({tokens, full.neuron_len})
        .contiguous();
    std::vector<int32_t> host_ids(
        static_cast<size_t>(tokens) *
        static_cast<size_t>(routes));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            host_ids[
                static_cast<size_t>(token) * routes +
                static_cast<size_t>(route)] =
                (token * routes + route * 3) %
                full.n_experts;
        }
    }
    auto ids = mfq_tensor_backend::from_blob(
        host_ids.data(), {tokens, routes},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone()
        .to(mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            saved.primary_device()))
        .contiguous();
    auto route =
        build_moe_route_plan(
            ids, full.n_experts);
    auto reference =
        full.forward(execution, x, route)
            .to(mfq_tensor_backend::kFloat32);
    auto test =
        sharded.forward(execution, x, route)
            .to(mfq_tensor_backend::kFloat32);
    mfq_cuda_synchronize();
    auto difference = test - reference;
    const double denominator =
        std::max(
            reference.norm().item<double>(),
            1.0e-30);
    const double relative =
        difference.norm().item<double>() /
        denominator;
    const double mean_abs =
        difference.abs().mean().item<double>();
    const double max_abs =
        difference.abs().max().item<double>();
    const int64_t differing =
        test.ne(reference).sum().item<int64_t>();
    std::cout
        << "expert_parallel_moe_check=1"
        << " tensor=" << tensor_name
        << " tokens=" << tokens
        << " routes=" << routes
        << " experts=" << full.n_experts
        << " logical_shape=["
        << full.out_per_expert << ','
        << full.neuron_len << ']'
        << " shards="
        << sharded.expert_parallel_shards.size()
        << " differing=" << differing
        << " relative=" << relative
        << " mean_abs=" << mean_abs
        << " max_abs=" << max_abs
        << '\n';
    if (!mfq_tensor_backend::isfinite(test).all().item<bool>() ||
        relative > 1.0e-6) {
        throw std::runtime_error(
            "expert-parallel MoE numerical check failed");
    }
    return 0;
}

int run_mfe_tensor_check(
        CudaExecutionContext& execution,
        const std::string & model_path,
        const std::string & tensor_name,
        int tokens,
        int routes,
        int reps,
        int split_width,
        bool routed_input,
        bool benchmark_only) {
    const int max_tokens = benchmark_only ? 131072 : 4096;
    if (tokens < 1 || tokens > max_tokens || routes < 1 || reps < 1 ||
            split_width < 0) {
        throw std::runtime_error("MFE tensor check dimensions are invalid");
    }
    if (benchmark_only && split_width != 0) {
        throw std::runtime_error(
            "MFE benchmark-only mode does not support split checks");
    }
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    auto weight = load_mfe_gpu(execution,
        mfq, tensor_name, true, 0, "diagnostic");
    if (moe_expert_cache_has_sources(execution.moe_expert_cache) &&
            !moe_expert_cache_finalized(execution.moe_expert_cache)) {
        finalize_moe_expert_cache(execution.moe_expert_cache);
    }
    if (routes > weight.n_experts) {
        throw std::runtime_error("MFE tensor check routes exceed expert count");
    }
    if (split_width > 0 &&
            (split_width >= weight.out_per_expert || weight.mixed_forward)) {
        throw std::runtime_error(
            "MFE merged/split check requires a resident NINT tensor "
            "and an interior split width");
    }
    const int64_t count =
        (int64_t)tokens * (routed_input ? routes : 1) * weight.neuron_len;
    const auto input_shape = routed_input
        ? std::vector<int64_t>{tokens, routes, weight.neuron_len}
        : std::vector<int64_t>{tokens, weight.neuron_len};
    mfq_tensor_backend::Tensor x;
    if (benchmark_only) {
        x = mfq_tensor_backend::zeros(
            input_shape,
            mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kFloat16));
    } else {
        auto sequence = mfq_tensor_backend::arange(
            count,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        x = (
            (sequence.remainder(257) - 128.0) / 127.0 +
            0.03125 * mfq_tensor_backend::sin(sequence * 0.015625))
            .to(mfq_tensor_backend::kFloat16)
            .reshape(input_shape)
            .contiguous();
    }
    std::vector<int32_t> host_ids((size_t)tokens * routes);
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            host_ids[(size_t)token * routes + route] =
                (token * routes + route * 3) % weight.n_experts;
        }
    }
    auto ids = mfq_tensor_backend::from_blob(
        host_ids.data(), {tokens, routes},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone().to(mfq_tensor_backend::kCUDA).contiguous();
    auto route = build_moe_route_plan(ids, weight.n_experts);
    mfq_tensor_backend::Tensor output;
    for (int warmup = 0; warmup < 5; ++warmup) {
        output = weight.forward(execution, x, route);
        if (warmup == 0) weight.prefetch(route);
    }
    mfq_cuda_synchronize();
    cudaEvent_t start, stop;
    MFQ_CUDA_CHECK(cudaEventCreate(&start));
    MFQ_CUDA_CHECK(cudaEventCreate(&stop));
    auto stream = mfq_get_current_cuda_stream().stream();
    MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int index = 0; index < reps; ++index) output = weight.forward(execution, x, route);
    MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(start));
    MFQ_CUDA_CHECK(cudaEventDestroy(stop));
    if (benchmark_only) {
        std::cout << std::fixed << std::setprecision(9)
                  << "mfe_tensor_benchmark"
                  << " tensor=" << tensor_name
                  << " tokens=" << tokens
                  << " routes=" << routes
                  << " experts=" << weight.n_experts
                  << " out=" << weight.out_per_expert
                  << " k=" << weight.neuron_len
                  << " routed_input=" << (routed_input ? 1 : 0)
                  << " mixed=" << (weight.mixed_forward ? 1 : 0)
                  << " weight_bytes=" << mfe_weight_bytes(weight)
                  << " cuda_ms=" << elapsed / reps
                  << '\n';
        return 0;
    }
    double dense_reference_rel = -1.0;
    double dense_reference_mean_abs = -1.0;
    double dense_reference_max_abs = -1.0;
    if (split_width == 0) {
        auto reference = mfe_dense_reference(
            mfq, tensor_name, x, host_ids,
            tokens, routes, routed_input);
        mfq_cuda_synchronize();
        auto actual_f32 =
            output.reshape({tokens * routes, weight.out_per_expert})
                .to(mfq_tensor_backend::kFloat32);
        auto reference_f32 = reference.to(mfq_tensor_backend::kFloat32);
        auto difference = actual_f32 - reference_f32;
        dense_reference_rel =
            (difference.norm() / reference_f32.norm()).item<double>();
        dense_reference_mean_abs =
            difference.abs().mean().item<double>();
        dense_reference_max_abs =
            difference.abs().max().item<double>();
    }
    auto output_f32 = output.to(mfq_tensor_backend::kFloat32);
    if (!mfq_tensor_backend::isfinite(output_f32).all().item<bool>()) {
        throw std::runtime_error("MFE tensor check produced a non-finite value");
    }
    const double checksum = output_f32.sum().item<double>();
    const double squared_checksum = output_f32.square().sum().item<double>();
    auto flat = output_f32.cpu().reshape({-1});
    std::cout << std::fixed << std::setprecision(9)
              << "mfe_tensor_check"
              << " tensor=" << tensor_name
              << " tokens=" << tokens
              << " routes=" << routes
              << " experts=" << weight.n_experts
              << " out=" << weight.out_per_expert
              << " k=" << weight.neuron_len
              << " routed_input=" << (routed_input ? 1 : 0)
              << " mixed=" << (weight.mixed_forward ? 1 : 0)
              << " weight_bytes=" << mfe_weight_bytes(weight)
              << " cuda_ms=" << elapsed / reps
              << " dense_reference_rel=" << dense_reference_rel
              << " dense_reference_mean_abs=" << dense_reference_mean_abs
              << " dense_reference_max_abs=" << dense_reference_max_abs
              << " checksum=" << checksum
              << " sqsum=" << squared_checksum
              << " values=";
    const int64_t shown = std::min<int64_t>(flat.numel(), 128);
    const float * values = flat.data_ptr<float>();
    for (int64_t index = 0; index < shown; ++index) {
        if (index) std::cout << ",";
        std::cout << values[index];
    }
    std::cout << "\n";

    if (split_width > 0) {
        auto run_segment = [&](int width, int row_offset) {
            return weight.forward(execution, x, route)
                .narrow(2, row_offset, width).contiguous();
        };
        mfq_tensor_backend::Tensor merged;
        mfq_tensor_backend::Tensor left;
        mfq_tensor_backend::Tensor right;
        for (int warmup = 0; warmup < 3; ++warmup) {
            merged = run_segment(weight.out_per_expert, 0);
            left = run_segment(split_width, 0);
            right = run_segment(
                weight.out_per_expert - split_width, split_width);
        }
        mfq_cuda_synchronize();
        auto split = mfq_tensor_backend::cat({left, right}, 2).contiguous();
        auto difference =
            merged.to(mfq_tensor_backend::kFloat32) - split.to(mfq_tensor_backend::kFloat32);
        const int64_t left_differing =
            merged.slice(2, 0, split_width).ne(left).sum().item<int64_t>();
        const int64_t right_differing =
            merged.slice(2, split_width, weight.out_per_expert)
                .ne(right).sum().item<int64_t>();
        const int64_t differing = left_differing + right_differing;
        const double merged_norm = merged.to(mfq_tensor_backend::kFloat32).norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "mfe_merged_split_check"
                  << " tensor=" << tensor_name
                  << " path=nint_matmul_kernel"
                  << " tokens=" << tokens
                  << " routes=" << routes
                  << " experts=" << weight.n_experts
                  << " bm=" << (tokens <= 128 ? 64 : (tokens <= 512 ? 32 : 64))
                  << " bn=64"
                  << " k=" << weight.neuron_len
                  << " full_width=" << weight.out_per_expert
                  << " split_width=" << split_width
                  << " right_width=" << weight.out_per_expert - split_width
                  << " weight_out_stride=" << weight.out_per_expert
                  << " left_row_offset=0"
                  << " right_row_offset=" << split_width
                  << " equal=" << (merged.equal(split) ? 1 : 0)
                  << " differing=" << differing
                  << " values=" << merged.numel()
                  << " left_differing=" << left_differing
                  << " right_differing=" << right_differing
                  << " rel_l2="
                  << (merged_norm == 0.0
                          ? 0.0
                          : difference.norm().item<double>() / merged_norm)
                  << " mean_abs=" << difference.abs().mean().item<double>()
                  << " max_abs=" << difference.abs().max().item<double>()
                  << "\n";
    }

    const char * warp_ab_env = std::getenv("MFQ_CHECK_NVQ_MOE_WARP_AB");
    if (warp_ab_env != nullptr && std::atoi(warp_ab_env) != 0) {
        const char * original_env = std::getenv("MFQ_NVQ_MOE_WARPS");
        const bool had_original_env = original_env != nullptr;
        const std::string original_value = had_original_env ? original_env : "";
        const char * original_exact_env =
            std::getenv("MFQ_NVQ_MOE_EXACT_REDUCTION");
        const bool had_original_exact_env = original_exact_env != nullptr;
        const std::string original_exact_value =
            had_original_exact_env ? original_exact_env : "";
        auto set_env = [](const char * name, const char * value) {
#ifdef _WIN32
            _putenv_s(name, value);
#else
            setenv(name, value, 1);
#endif
        };
        auto restore_env = [&](const char * name, bool existed, const std::string & value) {
#ifdef _WIN32
            _putenv_s(name, existed ? value.c_str() : "");
#else
            if (existed) {
                setenv(name, value.c_str(), 1);
            } else {
                unsetenv(name);
            }
#endif
        };
        set_env("MFQ_NVQ_MOE_WARPS", "0");
        set_env("MFQ_NVQ_MOE_EXACT_REDUCTION", "1");
        auto candidate = weight.forward(execution, x, route);
        set_env("MFQ_NVQ_MOE_EXACT_REDUCTION", "0");
        auto baseline = weight.forward(execution, x, route);
        mfq_cuda_synchronize();
        restore_env(
            "MFQ_NVQ_MOE_WARPS", had_original_env, original_value);
        restore_env(
            "MFQ_NVQ_MOE_EXACT_REDUCTION",
            had_original_exact_env, original_exact_value);
        auto candidate_f32 = candidate.to(mfq_tensor_backend::kFloat32);
        auto baseline_f32 = baseline.to(mfq_tensor_backend::kFloat32);
        auto diff = candidate_f32 - baseline_f32;
        const double baseline_norm = baseline_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "nvq_moe_warp_ab"
                  << " candidate_physical_warps=2"
                  << " baseline_warps=" << (weight.neuron_len >= 4096 ? 8 : 4)
                  << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                  << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                  << " rel_l2=" << diff.norm().item<double>() / baseline_norm
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    }

    const char * clamped_ab_env =
        std::getenv("MFQ_CHECK_MFE_CLAMPED_SWIGLU_AB");
    if (clamped_ab_env != nullptr && weight.supports_clamped_swiglu()) {
        const double limit = std::atof(clamped_ab_env);
        if (!std::isfinite(limit) || limit <= 0.0) {
            throw std::runtime_error(
                "MFQ_CHECK_MFE_CLAMPED_SWIGLU_AB must be positive");
        }
        const int64_t gate_up_count =
            static_cast<int64_t>(tokens) * routes * 2 * weight.neuron_len;
        auto gate_up_sequence = mfq_tensor_backend::arange(
            gate_up_count,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        auto gate_up = (
            12.0 * mfq_tensor_backend::sin(gate_up_sequence * 0.013671875) +
            0.5 * mfq_tensor_backend::cos(gate_up_sequence * 0.00390625))
            .to(mfq_tensor_backend::kFloat16)
            .reshape({tokens, routes, 2 * weight.neuron_len})
            .contiguous();
        auto run_candidate = [&]() {
            return weight.forward_clamped_swiglu(
                execution, gate_up, route, limit);
        };
        auto run_baseline = [&]() {
            const int64_t width = weight.neuron_len;
            auto gate = mfq_tensor_backend::clamp_max(
                gate_up.slice(-1, 0, width).to(mfq_tensor_backend::kFloat32), limit);
            auto up = mfq_tensor_backend::clamp(
                gate_up.slice(-1, width, 2 * width).to(mfq_tensor_backend::kFloat32),
                -limit, limit);
            auto hidden = (mfq_tensor_backend::silu(gate) * up)
                .to(mfq_tensor_backend::kFloat16).contiguous();
            return weight.forward(execution, hidden, route);
        };
        mfq_tensor_backend::Tensor candidate;
        mfq_tensor_backend::Tensor baseline;
        for (int warmup = 0; warmup < 5; ++warmup) {
            candidate = run_candidate();
            baseline = run_baseline();
        }
        mfq_cuda_synchronize();
        auto time_ms = [&](auto && fn) {
            cudaEvent_t begin, end;
            MFQ_CUDA_CHECK(cudaEventCreate(&begin));
            MFQ_CUDA_CHECK(cudaEventCreate(&end));
            auto cuda_stream = mfq_get_current_cuda_stream().stream();
            MFQ_CUDA_CHECK(cudaEventRecord(begin, cuda_stream));
            mfq_tensor_backend::Tensor value;
            for (int iteration = 0; iteration < reps; ++iteration) {
                value = fn();
            }
            MFQ_CUDA_CHECK(cudaEventRecord(end, cuda_stream));
            MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float elapsed_ms = 0.0f;
            MFQ_CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_ms, begin, end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));
            MFQ_CUDA_CHECK(cudaEventDestroy(end));
            return std::pair<double, mfq_tensor_backend::Tensor>(
                elapsed_ms / reps, std::move(value));
        };
        auto candidate_timing = time_ms(run_candidate);
        auto baseline_timing = time_ms(run_baseline);
        candidate = std::move(candidate_timing.second);
        baseline = std::move(baseline_timing.second);
        auto candidate_f32 = candidate.to(mfq_tensor_backend::kFloat32);
        auto baseline_f32 = baseline.to(mfq_tensor_backend::kFloat32);
        auto diff = candidate_f32 - baseline_f32;
        const double baseline_norm = baseline_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "mfe_clamped_swiglu_ab"
                  << " limit=" << limit
                  << " candidate_ms=" << candidate_timing.first
                  << " baseline_ms=" << baseline_timing.first
                  << " speedup=" << baseline_timing.first / candidate_timing.first
                  << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                  << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                  << " rel_l2=" << diff.norm().item<double>() / baseline_norm
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    }
    if (execution.moe_expert_cache) {
        print_moe_expert_cache_stats(execution.moe_expert_cache, std::cout);
    }
    return 0;
}
static int run_gemma_moe_check(
        CudaExecutionContext& execution,
        const mfq::ModelSource & mfq,
        const mfq::models::gemma4::Config & config,
        int layer,
        const std::vector<int64_t> & token_sizes,
    int reps) {
    const std::string prefix =
        "model.block." + std::to_string(layer) + ".mlp.experts.";
    auto gate_up = load_mfe_gpu(execution,
        mfq, prefix + "gate_up.weight",
        true, layer, "gate_up");
    auto down = load_mfe_gpu(execution,
        mfq, prefix + "down.weight",
        true, layer, "down");
    if (execution.moe_expert_cache &&
            !moe_expert_cache_finalized(execution.moe_expert_cache)) {
        finalize_moe_expert_cache(execution.moe_expert_cache);
    }
    const int routes = static_cast<int>(config.num_experts_per_tok);
    const int experts = static_cast<int>(config.num_experts);
    MFQ_RUNTIME_CHECK(routes > 0 && routes <= 8 && experts > routes,
        "Gemma MoE benchmark requires 1..8 routes and more experts than routes");
    const char * dense_reference_env =
        std::getenv("MFQ_CHECK_GEMMA_MOE_DENSE_REFERENCE");
    const bool dense_reference_enabled =
        dense_reference_env != nullptr && std::atoi(dense_reference_env) != 0;
    mfq_tensor_backend::Tensor dense_gate_up;
    mfq_tensor_backend::Tensor dense_down;
    if (dense_reference_enabled) {
        dense_gate_up = materialize_mfe_dense(
            mfq, prefix + "gate_up.weight");
        dense_down = materialize_mfe_dense(
            mfq, prefix + "down.weight");
    }
    std::cout << "gemma_moe_bench_config"
              << " layer=" << layer
              << " experts=" << experts
              << " top_k=" << routes
              << " hidden=" << gate_up.neuron_len
              << " intermediate=" << down.neuron_len
              << " gate_up_pools=" << gate_up.pools.size()
              << " down_pools=" << down.pools.size()
              << " routed_weight_bytes=" << std::fixed << std::setprecision(0)
              << mfe_weight_bytes(gate_up) + mfe_weight_bytes(down)
              << "\n";

    auto stream = mfq_get_current_cuda_stream().stream();
    auto time_ms = [&](auto && fn, int iterations) {
        mfq_tensor_backend::Tensor output;
        for (int warmup = 0; warmup < 5; ++warmup) output = fn();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < iterations; ++iteration) output = fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return std::pair<double, mfq_tensor_backend::Tensor>(elapsed / iterations, output);
    };

    auto topk_logits = mfq_tensor_backend::randn(
        {1, experts}, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
    auto selected = moe_topk_cuda(
        topk_logits, routes, false, false, false, true, mfq_nullopt, 1e-20, 1.0);
    auto reference = mfq_tensor_backend::topk(topk_logits, routes, 1, true, true);
    auto reference_weights = mfq_tensor_backend::softmax(std::get<0>(reference), 1);
    mfq_cuda_synchronize();
    auto topk_timing = time_ms([&]() {
        return moe_topk_cuda(
            topk_logits, routes, false, false, false, true,
            mfq_nullopt, 1e-20, 1.0).at(1);
    }, reps);
    std::cout << std::fixed << std::setprecision(6)
              << "gemma_topk_check"
              << " ids_equal="
              << (selected.at(0).equal(std::get<1>(reference).to(mfq_tensor_backend::kInt32)) ? 1 : 0)
              << " weights_max_abs="
              << (selected.at(1) - reference_weights).abs().max().item<float>()
              << " cuda_ms=" << topk_timing.first << "\n";

    mfq_tensor_backend::manual_seed(20260721 + layer);
    for (int64_t tokens : token_sizes) {
        auto x = mfq_tensor_backend::randn(
            {tokens, gate_up.neuron_len},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat16));
        std::vector<int32_t> host_ids(static_cast<size_t>(tokens) * routes);
        for (int64_t token = 0; token < tokens; ++token) {
            for (int route_index = 0; route_index < routes; ++route_index) {
                host_ids[static_cast<size_t>(token) * routes + route_index] =
                    static_cast<int32_t>(
                        (token * routes + route_index) % experts);
            }
        }
        auto ids = mfq_tensor_backend::from_blob(
            host_ids.data(), {tokens, routes}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::kCUDA).contiguous();
        auto weights = mfq_tensor_backend::full(
            {tokens, routes}, 1.0 / routes,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        auto forward_materialized = [&]() {
            auto route = build_moe_route_plan(ids, experts);
            const bool projection_bundle_prefetched =
                prefetch_cached_moe_projection_bundle(
                    gate_up, down, route);
            auto gate_pair = gate_up.forward(execution, x, route);
            auto hidden = moe_geglu_split_cuda(gate_pair);
            if (!projection_bundle_prefetched) down.prefetch(route);
            auto down_pair = down.forward(execution, hidden, route);
            return moe_weighted_reduce_cuda(down_pair, weights);
        };
        auto forward_gate_glu = [&]() {
            auto route = build_moe_route_plan(ids, experts);
            const bool projection_bundle_prefetched =
                prefetch_cached_moe_projection_bundle(
                    gate_up, down, route);
            auto hidden = gate_up.forward_glu_output(execution, x, route, true);
            if (!projection_bundle_prefetched) down.prefetch(route);
            auto down_pair = down.forward(execution, hidden, route);
            return moe_weighted_reduce_cuda(down_pair, weights);
        };
        const bool gate_glu_supported = tokens <= 4;
        auto forward = [&]() {
            return gate_glu_supported
                ? forward_gate_glu()
                : forward_materialized();
        };
        auto fused_check = forward();
        auto materialized_check = forward_materialized();
        auto gate_glu_check = gate_glu_supported
            ? forward_gate_glu()
            : fused_check;
        mfq_cuda_synchronize();
        auto fused_diff = (fused_check - materialized_check).abs().to(mfq_tensor_backend::kFloat32);
        auto fused_time = time_ms(forward, reps);
        auto materialized_time = time_ms(forward_materialized, reps);
        auto gate_glu_time = gate_glu_supported
            ? time_ms(forward_gate_glu, reps)
            : fused_time;
        auto gate_glu_diff = (gate_glu_check - materialized_check).abs().to(mfq_tensor_backend::kFloat32);
        std::cout << std::fixed << std::setprecision(6)
                  << "gemma_moe_geglu_quant_fusion"
                  << " tokens=" << tokens
                  << " equal=" << (fused_check.equal(materialized_check) ? 1 : 0)
                  << " max_abs=" << fused_diff.max().item<float>()
                  << " fused_ms=" << fused_time.first
                  << " materialized_ms=" << materialized_time.first
                  << " speedup=" << materialized_time.first / fused_time.first
                  << " gate_glu_supported=" << (gate_glu_supported ? 1 : 0)
                  << " gate_glu_equal=" << (gate_glu_check.equal(materialized_check) ? 1 : 0)
                  << " gate_glu_max_abs=" << gate_glu_diff.max().item<float>()
                  << " gate_glu_ms=" << gate_glu_time.first << "\n";

        if (tokens == 1) {
            auto stage_route = build_moe_route_plan(ids, experts);
            auto stage_hidden = gate_up.forward_glu_output(execution, x, stage_route, true);
            auto stage_down = down.forward(execution, stage_hidden, stage_route);
            mfq_cuda_synchronize();
            auto gate_stage = time_ms(
                [&]() { return gate_up.forward_glu_output(execution, x, stage_route, true); }, reps);
            auto down_stage = time_ms(
                [&]() { return down.forward(execution, stage_hidden, stage_route); }, reps);
            auto reduce_stage = time_ms(
                [&]() { return moe_weighted_reduce_cuda(stage_down, weights); }, reps);
            std::cout << "gemma_moe_stage"
                      << " gate_up_geglu_ms=" << gate_stage.first
                      << " down_ms=" << down_stage.first
                      << " reduce_ms=" << reduce_stage.first << "\n";
        }

        if (tokens != 1) {
            execution.force_moe_prefill_mma_off = false;
            auto mma = time_ms(forward, reps);
            auto mma_first = mma.second.clone();
            auto mma_repeat = forward().clone();
            mfq_cuda_synchronize();
            execution.force_moe_prefill_mma_off = true;
            auto baseline = time_ms(forward, reps);
            mfq_cuda_synchronize();
            execution.force_moe_prefill_mma_off = false;
            auto repeat_diff = (mma_repeat - mma_first).abs().to(mfq_tensor_backend::kFloat32);
            auto baseline_diff = (mma_first - baseline.second).abs().to(mfq_tensor_backend::kFloat32);
            std::cout << std::setprecision(6)
                      << "gemma_moe_prefill_ab"
                      << " tokens=" << tokens
                      << " mma_ms=" << mma.first
                      << " baseline_ms=" << baseline.first
                      << " speedup=" << baseline.first / mma.first
                      << " repeat_equal=" << (mma_repeat.equal(mma_first) ? 1 : 0)
                      << " repeat_max_abs=" << repeat_diff.max().item<float>()
                      << " baseline_rel="
                      << ((mma_first - baseline.second).to(mfq_tensor_backend::kFloat32).norm() /
                          baseline.second.to(mfq_tensor_backend::kFloat32).norm()).item<float>()
                      << " baseline_max_abs=" << baseline_diff.max().item<float>()
                      << "\n";
            if (dense_reference_enabled) {
                auto dense_route_forward = [&](const mfq_tensor_backend::Tensor & dense,
                                               const mfq_tensor_backend::Tensor & input) {
                    auto result = mfq_tensor_backend::empty(
                        {tokens, routes, dense.size(1)},
                        input.options().dtype(mfq_tensor_backend::kFloat16));
                    for (int route_index = 0; route_index < routes; ++route_index) {
                        const int expert = (route_index * 17) % experts;
                        auto selected = input.dim() == 3
                            ? input.select(1, route_index)
                            : input;
                        result.select(1, route_index).copy_(mfq_tensor_backend::matmul(
                            selected,
                            dense.index({expert}).transpose(0, 1)));
                    }
                    return result;
                };
                auto dense_gate_pair =
                    dense_route_forward(dense_gate_up, x);
                auto dense_hidden =
                    moe_geglu_split_cuda(dense_gate_pair);
                auto dense_down_pair =
                    dense_route_forward(dense_down, dense_hidden);
                auto dense_output =
                    moe_weighted_reduce_cuda(dense_down_pair, weights);
                auto dense_difference =
                    mma_first.to(mfq_tensor_backend::kFloat32) -
                    dense_output.to(mfq_tensor_backend::kFloat32);
                const double dense_norm =
                    dense_output.to(mfq_tensor_backend::kFloat32).norm().item<double>();
                std::cout << std::scientific << std::setprecision(9)
                          << "gemma_moe_dense_reference"
                          << " tokens=" << tokens
                          << " rel_l2="
                          << dense_difference.norm().item<double>() / dense_norm
                          << " mean_abs="
                          << dense_difference.abs().mean().item<double>()
                          << " max_abs="
                          << dense_difference.abs().max().item<double>()
                          << "\n";
            }
        }
    }
    if (execution.moe_expert_cache) {
        print_moe_expert_cache_stats(execution.moe_expert_cache, std::cout);
    }
    return 0;
}

int run_moe_check(
        CudaExecutionContext& execution,
        const std::string & model_path,
        const std::string & config_path,
        int layer,
        const std::vector<int64_t> & token_sizes,
        int reps) {
    if (layer < 0) throw std::runtime_error("--check-moe-layer must be nonnegative");
    if (reps < 1) throw std::runtime_error("--check-moe-reps must be positive");
    if (token_sizes.empty() || std::any_of(token_sizes.begin(), token_sizes.end(),
            [](int64_t value) { return value < 1 || value > 4096; })) {
        throw std::runtime_error("--check-moe-tokens values must be in [1, 4096]");
    }
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    mfq::cuda::validate_model_source(mfq);
    const auto graph = mfq.resolved_model_graph();
    const auto payload = mfq::cuda::load_model_config_json(
        mfq, config_path);
    FFN ffn;
    int64_t hidden_size = 0;
    if (graph.backbone == "gemma4") {
        const auto config = mfq::models::gemma4::Config::from_json(payload);
        if (layer >= config.num_hidden_layers) {
            throw std::runtime_error(
                "MoE benchmark layer is out of range");
        }
        return run_gemma_moe_check(
            execution, mfq, config, layer, token_sizes, reps);
    }
    if (graph.backbone == "glm_dsa") {
        const auto config = mfq::models::glm_dsa::Config::from_json(payload);
        if (layer >= config.num_hidden_layers) {
            throw std::runtime_error(
                "MoE benchmark layer is out of range");
        }
        mfq::cuda::glm_dsa::load_ffn(
            execution, mfq, config, layer, ffn);
        hidden_size = config.hidden_size;
    } else if (graph.backbone == "deepseek_v4") {
        const auto config =
            mfq::models::deepseek_v4::Config::from_json(payload);
        if (layer >= config.num_hidden_layers) {
            throw std::runtime_error(
                "MoE benchmark layer is out of range");
        }
        auto block = mfq::cuda::deepseek_v4::load_block(
            execution, mfq, config, layer, "deepseek_v4",
            std::make_shared<Dsv4SharedState>());
        ffn = std::move(static_cast<Dsv4Block&>(*block).ffn);
        hidden_size = config.hidden_size;
    } else if (graph.backbone == "deepseek_v41") {
        const auto config =
            mfq::models::deepseek_v41::Config::from_json(payload);
        if (layer >= config.n_layers) {
            throw std::runtime_error(
                "MoE benchmark layer is out of range");
        }
        ffn = mfq::cuda::deepseek_v41_runtime::load_moe_at(
            execution, mfq, config,
            "model.block." + std::to_string(layer) + ".mlp.",
            layer, config.top_k);
        hidden_size = config.hidden;
    } else {
        throw std::runtime_error(
            "selected CUDA backbone has no FFN MoE benchmark adapter");
    }
    if (!ffn.is_moe) {
        throw std::runtime_error(
            "selected layer does not contain MFE MoE weights");
    }
    if (execution.moe_expert_cache &&
            !moe_expert_cache_finalized(execution.moe_expert_cache)) {
        finalize_moe_expert_cache(execution.moe_expert_cache);
    }
    const double routed_weight_bytes =
        mfe_weight_bytes(ffn.moe_gate_up) + mfe_weight_bytes(ffn.moe_down);
    std::cout << "moe_bench_config"
              << " layer=" << layer
              << " experts=" << ffn.moe_gate_up.n_experts
              << " top_k=" << ffn.moe_top_k
              << " hidden=" << ffn.moe_gate_up.neuron_len
              << " intermediate=" << ffn.moe_down.neuron_len
              << " gate_up_pools=" << ffn.moe_gate_up.pools.size()
              << " down_pools=" << ffn.moe_down.pools.size()
              << " routed_weight_bytes=" << std::fixed << std::setprecision(0)
              << routed_weight_bytes << "\n";

    mfq_tensor_backend::manual_seed(20260720 + layer);
    mfq_tensor_backend::Tensor output;
    for (int64_t tokens : token_sizes) {
        auto x = mfq_tensor_backend::randn(
            {tokens, hidden_size},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat16));
        for (int warmup = 0; warmup < 10; ++warmup) output = ffn.forward(execution, x);
        mfq_cuda_synchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = mfq_get_current_cuda_stream().stream();
        auto wall_start = std::chrono::steady_clock::now();
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < reps; ++iteration) output = ffn.forward(execution, x);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        auto wall_stop = std::chrono::steady_clock::now();
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        const double cuda_ms = static_cast<double>(elapsed_ms) / reps;
        const double wall_ms =
            std::chrono::duration<double, std::milli>(wall_stop - wall_start).count() / reps;
        const double checksum = output.to(mfq_tensor_backend::kFloat32).sum().item<double>();
        if (!std::isfinite(checksum)) throw std::runtime_error("non-finite MoE benchmark output");
        std::cout << std::setprecision(6)
                  << "moe_bench_result"
                  << " layer=" << layer
                  << " tokens=" << tokens
                  << " reps=" << reps
                  << " cuda_ms=" << cuda_ms
                  << " wall_ms=" << wall_ms
                  << " tokens_per_second=" << (1000.0 * tokens / cuda_ms)
                  << " checksum=" << checksum << "\n";

        const char * prefill_ab_env = std::getenv("MFQ_CHECK_MOE_PREFILL_MMA_AB");
        if (prefill_ab_env != nullptr && std::atoi(prefill_ab_env) != 0 && tokens >= 9) {
            auto candidate = output.clone();
            execution.force_moe_prefill_mma_off = true;
            mfq_tensor_backend::Tensor baseline;
            try {
                baseline = ffn.forward(execution, x);
                mfq_cuda_synchronize();
            } catch (...) {
                execution.force_moe_prefill_mma_off = false;
                throw;
            }
            execution.force_moe_prefill_mma_off = false;
            auto diff = (candidate - baseline).to(mfq_tensor_backend::kFloat32);
            const double baseline_norm = baseline.to(mfq_tensor_backend::kFloat32).norm().item<double>();
            std::cout << "moe_prefill_mma_ab"
                      << " tokens=" << tokens
                      << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                      << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                      << " rel_l2=" << (diff.norm().item<double>() / baseline_norm)
                      << " mean_abs=" << diff.abs().mean().item<float>()
                      << " max_abs=" << diff.abs().max().item<float>()
                      << "\n";
        }

        const char * exact_env = std::getenv("MFQ_CHECK_MOE_POOL_EXACT");
        if (exact_env != nullptr && std::atoi(exact_env) != 0 && tokens <= 8) {
            auto candidate = output;
            execution.force_moe_pool_path = true;
            execution.force_moe_unfused_reduce = true;
            execution.force_moe_materialized_swiglu = true;
            mfq_tensor_backend::Tensor baseline;
            try {
                baseline = ffn.forward(execution, x);
                mfq_cuda_synchronize();
            } catch (...) {
                execution.force_moe_pool_path = false;
                execution.force_moe_unfused_reduce = false;
                execution.force_moe_materialized_swiglu = false;
                throw;
            }
            execution.force_moe_pool_path = false;
            execution.force_moe_unfused_reduce = false;
            execution.force_moe_materialized_swiglu = false;
            auto diff = (candidate - baseline).abs().to(mfq_tensor_backend::kFloat32);
            std::cout << "moe_exact_result"
                      << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                      << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                      << " max_abs=" << diff.max().item<float>()
                      << "\n";
        }

        execution.profiler.reset();
        execution.profiler.enabled = true;
        const int profile_reps = std::min(reps, 10);
        for (int iteration = 0; iteration < profile_reps; ++iteration) output = ffn.forward(execution, x);
        mfq_cuda_synchronize();
        execution.profiler.report(
            "moe_layer" + std::to_string(layer) + "_m" +
            std::to_string(tokens));
        execution.profiler.enabled = false;
        execution.profiler.reset();
    }
    if (execution.moe_expert_cache) {
        print_moe_expert_cache_stats(execution.moe_expert_cache, std::cout);
    }
    return 0;
}
