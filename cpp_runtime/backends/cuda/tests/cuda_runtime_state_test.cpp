#include "mfq_cuda_norm_ops.h"
#include "cuda_execution.h"
#include "cuda_runtime_config.h"
#include "cuda_sampling.h"
#include "diagnostics/generation_result.h"
#include "storage/text_session_cache.h"
#include "mfq_paged_prefix_cache.h"
#include "models/gemma4/causal_lm.h"
#include "models/minicpmo45/ops.h"
#include "models/minicpmo45/tts.h"
#include "models/qwen35/linear_attention.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/model.h"
#include "storage/session_state.h"
#include "moe.h"
#include "storage/mfe_expert_store.h"
#include "storage/moe_expert_cache.h"
#include "storage/weight_loader.h"
#include "runtime/kv_memory.h"
#include "runtime/kv_expert_loan.h"
#include "runtime/dma_copy_batch.h"
#include "mfe_cached_dispatch_test.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace mfq::cuda;
using namespace mfq::cuda::internal;

static void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void check_sampling_fallback() {
    namespace tb=mfq_tensor_backend;
    int checks=0;
    auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    for(auto [vocab,k]:{std::pair{129,65},std::pair{4096,257},std::pair{4096,1024},std::pair{248320,65}})
    for(int rows:{1,2})for(auto dtype:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})for(float p:{0.f,.8f,1.f}) {
        auto cpu=tb::empty({rows,vocab},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto logits=tb::zeros({rows,vocab},cpu.options().device(tb::kCUDA).dtype(dtype));
        auto random=tb::zeros({rows},logits.options().dtype(tb::kFloat32));
        tb::Tensor actual;
        const auto body=[&]{actual={};actual=sample_top_k_top_p_cuda(logits,random,.7,k,p);};
        {GraphWarmupScope warmup(context,stream);body();}
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));actual={};
        Graph graph;graph.capture_begin_shared();body();graph.capture_end();
        for(float u:{0.f,.137f,.731f,.99999994f}) {
            for(int i=0;i<rows*vocab;++i)cpu.data_ptr<float>()[i]=u==0.f?0.f:std::sin(float(i)*.037f)*5.f;
            logits.copy_(cpu.to(tb::kCUDA).to(dtype));random.fill_(u);graph.replay();
            auto sampled=actual.cpu(),values=logits.to(tb::kCPU,tb::kFloat32);
            for(int row=0;row<rows;++row) {
                std::vector<int> order(vocab);std::iota(order.begin(),order.end(),0);
                const auto* v=values.data_ptr<float>()+row*vocab;
                std::vector<float> scaled(vocab);
                for(int i=0;i<vocab;++i)scaled[i]=v[i]/.7f;
                std::partial_sort(order.begin(),order.begin()+k,order.end(),[&](int a,int b){
                    return scaled[a]>scaled[b] || (scaled[a]==scaled[b] && a<b);
                });
                std::vector<float> probabilities(k);float sum=0;
                for(int i=0;i<k;++i){probabilities[i]=std::exp(scaled[order[i]]-scaled[order[0]]);sum+=probabilities[i];}
                int keep=k;float keep_sum=sum;
                if(p>=0 && p<1) {
                    float cumulative=0;
                    for(int i=0;i<k;++i)if((cumulative+=probabilities[i])>=p*sum){keep=i+1;keep_sum=cumulative;break;}
                }
                float cdf=0;int expected=order[keep-1];
                for(int i=0;i<keep;++i)if((cdf+=probabilities[i])>=u*keep_sum){expected=order[i];break;}
                check(sampled.data_ptr<int64_t>()[row]==expected,"fallback top-k/top-p differs from CPU distribution");++checks;
            }
        }
    }
    context->end_graph_pool(stream);
    std::cout<<"Fallback sampling CPU/graph checks="<<checks<<" top-k65/257/1024, Half/BFloat16/Float, ties and top-p PASS\n";
}

static void check_sampling_storage() {
    using namespace mfq_tensor_backend;
    auto ids = tensor(std::vector<int64_t>{1, 2, 1}, TensorOptions().device(kCUDA).dtype(kInt64));
    auto storage = full({6}, 99, ids.options().dtype(kInt32));
    auto counts = SamplingOps::token_counts(ids, 6, storage);
    auto owned = SamplingOps::token_counts(ids, 6);
    check(counts.data_ptr() == storage.data_ptr(), "sampling replaced graph count storage");
    const auto expected = tensor(std::vector<int32_t>{0, 2, 1, 0, 0, 0});
    for (const auto& value : {counts, owned}) {
        auto host = value.cpu().contiguous();
        check(host.scalar_type() == kInt32 && host.nbytes() == expected.nbytes() &&
              std::memcmp(host.data_ptr(), expected.data_ptr(), host.nbytes()) == 0,
              "sampling counts were not reset from the prompt");
    }
    MfqSamplingParams params;
    params.temperature = 0.8; params.top_k = 4; params.top_p = 0.9;
    params.seed = 73; params.repetition_penalty = 1.2;
    Sampler first(params), second(params);
    check(first.ops().random_host().data_ptr() != second.ops().random_host().data_ptr(),
          "samplers share their host random buffer");
    auto logits = tensor(std::vector<float>{1, 2, 3, 4, 5, 6},
                         TensorOptions().device(kCUDA).dtype(kFloat32)).reshape({1, 6});
    for (int i = 0; i < 16; ++i)
        check(sample_logits(first, logits.clone(), counts).item<int64_t>() ==
              sample_logits(second, logits.clone(), owned).item<int64_t>(),
              "owned/reused sampling storage changed seeded output");
    (void)SamplingOps::token_counts(ids, 6, storage);
    check(owned.sum().item<int64_t>() == 3, "resetting one sampler changed another");
}

static void check_graph_warmup_state() {
    using namespace mfq_tensor_backend;
    Qwen35CausalLm model;
    auto block = std::make_unique<qwen35::LinearAttentionBlock>();
    auto* recurrent = block.get();
    const auto options = TensorOptions().device(kCUDA).dtype(kFloat32);
    recurrent->mixer.convolution = ones({2}, options);
    recurrent->mixer.recurrent = ones({3}, options);
    const auto* conv_address = recurrent->mixer.convolution.data_ptr();
    const auto* gdn_address = recurrent->mixer.recurrent.data_ptr();
    model.blocks.push_back(std::move(block));
    model.cache_pos = 5;
    for (bool fail : {false, true}) {
        MfqCudaGraph graph;
        int calls = 0;
        bool threw = false;
        try {
            prepare_decode_graph_memory(model, graph, [&] {
                ++calls;
                recurrent->mixer.convolution.zero_();
                recurrent->mixer.recurrent.zero_();
                if (fail) throw std::runtime_error("warmup failed");
            });
        } catch (const std::runtime_error&) { threw = true; }
        check(threw == fail && calls == (fail ? 1 : 2), "graph warmup execution changed");
        check(model.cache_pos == 5 && recurrent->mixer.convolution.data_ptr() == conv_address &&
                  recurrent->mixer.recurrent.data_ptr() == gdn_address,
              "graph warmup changed persistent state addresses");
        check(recurrent->mixer.convolution.sum().item<float>() == 2 &&
                  recurrent->mixer.recurrent.sum().item<float>() == 3,
              "graph warmup did not restore recurrent state");
    }
    recurrent->mixer.speculative_pending = true;
    bool rejected = false;
    try { (void)recurrent->graph_warmup_state(); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "graph warmup accepted unconfirmed recurrent state");
}

static void check_dense_loading() {
    using namespace mfq_tensor_backend;
    struct Source final : mfq::ModelSource {
        std::vector<uint8_t> bytes;
        std::vector<mfq::TensorMetadata> records{{"weight", "F32", "F32", std::nullopt, 0}};
        std::vector<std::filesystem::path> paths;
        std::unordered_map<std::string, std::string> meta;
        std::vector<std::string> asset_names;
        const std::vector<std::filesystem::path>& source_paths() const noexcept override { return paths; }
        std::string_view architecture() const noexcept override { return {}; }
        const std::unordered_map<std::string, std::string>& metadata() const noexcept override { return meta; }
        const std::vector<mfq::TensorMetadata>& tensors() const noexcept override { return records; }
        const mfq::TensorMetadata* find_tensor(std::string_view name) const noexcept override {
            return name == "weight" ? &records[0] : nullptr;
        }
        void read_range_into(std::string_view, uint64_t offset, std::byte* output, size_t size) const override {
            check(offset <= bytes.size() && size <= bytes.size() - offset, "dense fixture range");
            if (size) std::memcpy(output, bytes.data() + offset, size);
        }
        const std::vector<std::string>& assets() const noexcept override { return asset_names; }
        bool has_asset(std::string_view) const noexcept override { return false; }
        std::vector<std::byte> read_asset(std::string_view) const override { throw std::runtime_error("no assets"); }
        std::optional<mfq::ModelGraph> model_graph() const override { return std::nullopt; }
    } source;
    CudaExecutionContext execution;
    const uint32_t rank = 2;
    const int64_t shape[] = {2, 2};
    const size_t header = sizeof(rank) + sizeof(shape);
    auto same = [](Tensor value, const Tensor &expected) {
        value = value.cpu().contiguous();
        return value.scalar_type() == expected.scalar_type() && value.sizes() == expected.sizes() &&
               std::memcmp(value.data_ptr(), expected.data_ptr(), expected.nbytes()) == 0;
    };
    for (const auto &[name, dtype] : std::vector<std::pair<std::string, ScalarType>>{
             {"BF16", kBFloat16}, {"F16", kFloat16}, {"F32", kFloat32}, {"I64", kInt64}, {"I32", kInt32}}) {
        const auto expected = tensor(std::vector<float>{-3.25, 0, 1.125, 7}).reshape({2, 2}).to(dtype);
        source.records[0].dtype = name;
        source.bytes.resize(header + expected.nbytes());
        std::memcpy(source.bytes.data(), &rank, sizeof(rank));
        std::memcpy(source.bytes.data() + sizeof(rank), shape, sizeof(shape));
        std::memcpy(source.bytes.data() + header, expected.data_ptr(), expected.nbytes());
        source.records[0].nbytes = source.bytes.size();
        auto native = load_dense_native_gpu(execution, source, "weight");
        check(same(native, expected), "dense native dtype/value changed");
        const auto promoted = name == "BF16" || name == "F16" ? kFloat32 : dtype;
        for (bool cpu_layer : {false, true}) {
            execution.loading_cpu_layer = cpu_layer;
            auto value = load_dense_gpu(execution, source, "weight");
            check(value.is_cuda() != cpu_layer && value.scalar_type() == promoted &&
                      same(value, expected.to(promoted)), "dense load placement/promotion changed");
        }
    }
    const auto valid = source.bytes;
    auto rejects = [&] {
        source.records[0].nbytes = source.bytes.size();
        bool rejected = false;
        try { (void)load_dense_cpu(execution, source, "weight"); }
        catch (const std::exception&) { rejected = true; }
        check(rejected, "malformed dense tensor accepted");
    };
    for (size_t size : {size_t(0), size_t(3), header - 1, valid.size() - 1}) {
        source.bytes = valid;
        source.bytes.resize(size);
        rejects();
    }
    for (int64_t extent : {int64_t(-1), std::numeric_limits<int64_t>::max()}) {
        source.bytes = valid;
        std::memcpy(source.bytes.data() + sizeof(rank), &extent, sizeof(extent));
        rejects();
    }
}

static void check_cached_moe_binding() {
    using namespace mfq_tensor_backend;
    MfeCpu cpu;
    cpu.n_experts = 2;
    cpu.out_per_expert = 4;
    cpu.neuron_len = 32;
    MfeCpuPool pool;
    pool.dtype = "MXFP4";
    pool.expert_ids = {0, 1};
    pool.mxfp4 = {8, 32, std::vector<uint8_t>(128), std::vector<uint8_t>(8, 127)};
    for (size_t i = 0; i < pool.mxfp4.values.size(); ++i)
        pool.mxfp4.values[i] = static_cast<uint8_t>(i * 17);
    cpu.pools.push_back(pool);

    // The same weights exercise host-backed and exact-range cache registration.
    auto append = [](auto& bytes, uint64_t value, int width) {
        for (int i = 0; i < width; ++i) bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    };
    std::vector<uint8_t> payload{'M', 'X', 'T', '1', 1, 4, 0, 0};
    for (uint64_t value : {8, 32, 8, 16, 8, 1}) append(payload, value, 8);
    payload.insert(payload.end(), pool.mxfp4.values.begin(), pool.mxfp4.values.end());
    payload.insert(payload.end(), pool.mxfp4.scales.begin(), pool.mxfp4.scales.end());
    std::vector<uint8_t> blob{'N', 'I', 'M', '2'};
    for (uint64_t value : {2, 4, 32, 1, 2, 5}) append(blob, value, 4);
    append(blob, payload.size(), 8);
    append(blob, 0, 8);
    append(blob, 0, 4);
    append(blob, 1, 4);
    blob.insert(blob.end(), pool.dtype.begin(), pool.dtype.end());
    blob.insert(blob.end(), payload.begin(), payload.end());
    auto store = std::make_shared<MfeMxfp4ExpertStore>(MfqRecordRange{
        "experts", "MFE", {}, 0, blob.size(),
        [&blob](uint64_t offset, std::span<uint8_t> output) {
            check(offset <= blob.size() && output.size() <= blob.size() - offset,
                  "expert range is out of bounds");
            std::copy_n(blob.data() + offset, output.size(), output.data());
        }});
    CudaExecutionContext execution;
    const auto resident = to_gpu_mixed_moe(cpu, execution.config);
    auto options = TensorOptions().device(kCUDA);
    auto input = (arange(32, options.dtype(kFloat32)) * 0.01 - 0.1)
                     .to(kFloat16).reshape({1, 32});
    for (bool ranges : {false, true}) {
        execution.moe_expert_cache = make_moe_expert_cache(1 << 20, execution.config);
        auto& cache = execution.moe_expert_cache;
        std::weak_ptr<MoeExpertCache> lifetime = cache;
        auto runtime = ranges ? make_mxfp4_range_runtime(*store)
                              : make_mixed_moe_runtime(cpu, false);
        auto weight = cache_moe_weight(cache, "experts", runtime, 1, 0, "gate",
                                       ranges ? store : nullptr);
        check(moe_expert_cache_has_sources(cache), "MoE source was not registered");
        finalize_moe_expert_cache(cache);
        CudaExecutionContext other;
        other.moe_expert_cache = make_moe_expert_cache(1 << 20, other.config);
        execution.reset();
        check(!lifetime.expired(), "execution reset destroyed a live model cache");
        check(!moe_expert_cache_has_sources(other.moe_expert_cache),
              "Engine instances share registered MoE sources");
        for (int32_t expert : {0, 1, 0}) {
            auto ids = tensor(std::vector<int32_t>{expert}, options.dtype(kInt32)).reshape({1, 1});
            auto route = build_moe_route_plan(ids, 2);
            auto expected = resident.forward(execution, input, route).to(kFloat32);
            auto actual = weight.forward(execution, input, route).to(kFloat32);
            check((actual - expected).abs().max().item<float>() == 0,
                  "cache registration changed expert output");
        }
        auto retained_forward = weight.mixed_forward;
        weight = {};
        check(!lifetime.expired(), "copied projection lost its cache");
        auto ids = tensor(std::vector<int32_t>{1}, options.dtype(kInt32)).reshape({1, 1});
        auto route = build_moe_route_plan(ids, 2);
        auto actual = retained_forward(execution, input, route).to(kFloat32);
        auto expected = resident.forward(execution, input, route).to(kFloat32);
        check((actual - expected).abs().max().item<float>() == 0,
              "retained projection cannot use its cache");
        retained_forward = {};
        check(lifetime.expired(), "registered sources form a cache ownership cycle");
        check(bool(other.moe_expert_cache), "releasing one cache changed another Engine");
    }
}

static void check_gemma_composition() {
    using namespace mfq_tensor_backend;
    using mfq::models::gemma4::FfnNorm;
    auto fp32 = TensorOptions().dtype(kFloat32).device(kCUDA);
    auto values = arange(16, fp32).reshape({2, 8});
    auto dense = ((values - 8) * 0.09).to(kFloat16);
    auto routed = ((values + 2) * 0.13).to(kFloat16);
    auto residual = ((values - 3) * 0.05).to(kFloat16);
    auto weight = arange(8, fp32) * 0.02 + 1;
    auto dense_norm = weight, expert_norm = weight + 0.1, output_norm = weight + 0.2;
    auto scale = full({1}, 0.8, fp32).to(kFloat16);
    auto run = [&](bool fused) {
        return mfq::models::gemma4::feed_forward(
            true, true, fused, residual, [&] { return dense; }, [&] { return routed; },
            [&](Tensor x, FfnNorm role) {
                auto norm = role == FfnNorm::dense     ? dense_norm
                            : role == FfnNorm::experts ? expert_norm
                                                       : output_norm;
                return rms_norm_f16_cuda(x.contiguous(), norm, 1e-6, 0.0);
            },
            [](Tensor x, Tensor y) { return x + y; },
            [](Tensor x, Tensor y) { return acc_cuda(x.contiguous(), y.contiguous()); },
            [&](Tensor x) { return x * scale; },
            [&](Tensor d, Tensor e, Tensor r) {
                return gemma4_ffn_merge_f16_cuda(d, e, r, dense_norm, expert_norm, output_norm,
                                                 scale, 1e-6);
            });
    };
    auto expected = run(false).to(kFloat32);
    auto actual = run(true).to(kFloat32);
    check((actual - expected).abs().max().item<float>() < 3e-3F,
          "shared Gemma FFN composition differs from the fused kernel");
}

static void check_tts_sampling() {
    using namespace mfq_tensor_backend;
    MiniCPMO45TtsSamplingOps ops;
    auto float_options = TensorOptions().dtype(kFloat32).device(kCUDA);
    auto scores = tensor(std::vector<float>{-2, 4, 3, 1, 0, 20}, float_options).reshape({1, 6});
    auto token = tensor(std::vector<int64_t>{1}, TensorOptions().dtype(kInt64).device(kCUDA));
    std::vector<Tensor> history(16, token);
    auto penalized = ops.penalties(scores.clone(), history, 1.1).cpu();
    check(std::abs(penalized.data_ptr<float>()[1] - 4.0F / std::pow(1.1F, 16)) < 2e-6F,
          "TTS repeated-token penalty changed");
    auto top = ops.top_k(scores.clone(), 3).cpu();
    check(std::isinf(top.data_ptr<float>()[0]) && top.data_ptr<float>()[2] == 3,
          "TTS top-k filtering changed");
    mfq::models::minicpmo45::TtsSampling options{5, 1, 3, 0, 0.8, 0.0001, 1.1, 0.0};
    std::mt19937 rng(42);
    for (bool reference : {false, true}) {
        ops.evaluator_rng = reference ? &rng : nullptr;
        auto sample = [&](int64_t step) {
            return mfq::models::minicpmo45::sample_tts(ops, scores.clone(), options,
                                                       std::span<const Tensor>(history), step,
                                                       reference)
                .item<int64_t>();
        };
        check(sample(0) == 2, "TTS did not mask EOS or penalize history before selection");
        check(sample(1) == 5, "TTS did not release EOS after minimum length");
    }
}

static void check_ffn_branches() {
    using namespace mfq_tensor_backend;
    const auto options = TensorOptions().dtype(kFloat32).device(kCUDA);
    auto linear = [&](int64_t out, int64_t in, float scale) {
        QuantLinear result;
        result.kind = QuantLinearKind::Dense;
        result.dense =
            ((arange(out * in, options).reshape({out, in}) - out * in / 2) * scale).to(kFloat16);
        return result;
    };
    auto dense_ffn = [&](float scale) {
        FFN result;
        result.gate_up.layers = {linear(16, 8, scale), linear(16, 8, scale / 2)};
        result.gate_up.outs = {16, 16};
        result.down = linear(8, 16, scale / 3);
        return result;
    };
    auto input = ((arange(24, options).reshape({3, 8}) - 12) * 0.05).to(kFloat16);
    CudaExecutionContext execution;
    auto low = dense_ffn(0.01F);
    auto high = std::make_unique<FFN>(dense_ffn(0.015F));
    execution.config.decode_branch_parallel = true;
    auto decode = input.narrow(0, 0, 1);
    auto expected = acc_cuda(low.forward(execution, decode), high->forward(execution, decode));
    low.important_neurons = std::move(high);
    for (bool parallel : {false, true}) {
        execution.config.important_neuron_branch_parallel = parallel;
        auto actual = low.forward(execution, decode);
        if (parallel)
            check(low.important_neuron_executor->streams.size() == 2,
                  "IN parallel fixture did not use two streams");
        check((actual.to(kFloat32) - expected.to(kFloat32)).abs().max().item<float>() < 1e-3,
              "important-neuron branch composition changed");
    }
    // One physical shard exercises the TP operation binding on single-GPU hosts.
    auto shard = [](QuantLinear &weight, TensorParallelAxis axis) {
        QuantLinearShard piece;
        piece.kind = QuantLinearKind::Dense;
        piece.dense = weight.dense;
        piece.input_end = weight.dense.size(1);
        piece.output_end = weight.dense.size(0);
        weight.logical_out = piece.output_end;
        weight.logical_neuron_len = piece.input_end;
        weight.tensor_parallel_axis = axis;
        weight.tensor_parallel_shards = {std::move(piece)};
    };
    for (int activation = 0; activation < 3; ++activation) {
        auto local = dense_ffn(0.01F);
        auto parallel = dense_ffn(0.01F);
        local.geglu = parallel.geglu = activation == 1;
        local.swiglu_limit = parallel.swiglu_limit = activation == 2 ? 0.1 : 0.0;
        auto expected = local.forward(execution, input);
        for (auto &projection : parallel.gate_up.layers)
            shard(projection, TensorParallelAxis::Output);
        shard(parallel.down, TensorParallelAxis::Input);
        check(parallel.tensor_parallel_dense_compatible(), "TP fixture is not eligible");
        auto actual = parallel.forward(execution, input);
        check((actual.to(kFloat32) - expected.to(kFloat32)).abs().max().item<float>() < 1e-3,
              "tensor-parallel gated MLP differs from local composition");
    }
}

static void check_linear_execution() {
    Nint8ZeroCpu nint_cpu;
    nint_cpu.out = 2;
    nint_cpu.ng = 1;
    nint_cpu.neuron_len = 32;
    nint_cpu.shape = {2, 32};
    nint_cpu.q.assign(64, 1);
    nint_cpu.scale_h.assign(2, 0x3c00); // FP16 1
    QuantLinear nint;
    nint.kind = QuantLinearKind::Nint;
    nint.nint = to_gpu_nint8_zero(nint_cpu);

    NvqCpu nvq_cpu;
    nvq_cpu.format = 3; // D4, four values per codebook entry
    nvq_cpu.sub_bits = 4;
    nvq_cpu.gs = nvq_cpu.neuron_len = 24;
    nvq_cpu.out = 2;
    nvq_cpu.ng = 1;
    nvq_cpu.nvec = 6;
    nvq_cpu.nsign = 3;
    nvq_cpu.shape = {2, 24};
    nvq_cpu.indices_packed.assign(12, 0);
    nvq_cpu.aux_packed.assign(6, 0);
    nvq_cpu.sub_scale_packed.assign(1, 0xff);
    nvq_cpu.neuron_scale_h.assign(2, 0x3c00);
    nvq_cpu.codebook.assign(256 * 4, 1);
    QuantLinear nvq;
    nvq.kind = QuantLinearKind::Nvq;
    nvq.nvq = to_gpu_nvq(nvq_cpu);

    CudaExecutionContext owner, other;
    for (const auto* linear : {&nint, &nvq}) {
        const auto input = (mfq_tensor_backend::arange(
            16 * linear->neuron_len(),
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kFloat32)) * 0.013)
            .to(mfq_tensor_backend::kFloat16).reshape({2, 8, linear->neuron_len()});
        for (auto mode : {KlMmqMode::Default, KlMmqMode::Fp16, KlMmqMode::Nint8One}) {
            owner.kl_mmq.mode = mode;
            other.kl_mmq.mode = mode == KlMmqMode::Nint8One
                ? KlMmqMode::Fp16 : KlMmqMode::Nint8One;
            auto verify = [&](auto run) {
                auto expected = run(input.reshape({16, linear->neuron_len()}))
                    .reshape({2, 8, 2});
                const auto calls = owner.kl_mmq.activation_quantize_calls;
                const auto dense_calls = owner.kl_mmq.dense_calls;
                const auto actual = run(input);
                check(actual.scalar_type() == expected.scalar_type() &&
                          actual.equal(expected),
                      "linear output lost its shape/dtype");
                check(owner.kl_mmq.activation_quantize_calls == calls +
                          (mode == KlMmqMode::Nint8One ? 1 : 0) &&
                      owner.kl_mmq.dense_calls == dense_calls +
                          (mode != KlMmqMode::Default ? 1 : 0) &&
                      other.kl_mmq.activation_quantize_calls == 0 &&
                      other.kl_mmq.dense_calls == 0,
                      "linear diagnostics escaped the explicit execution");
                check(run(input).equal(expected),
                      "linear execution changed without hidden state");
            };
            verify([&](auto x) { return linear->forward(owner, x); });
            verify([&](auto x) { return linear->forward_bf16_output(owner, x); });
            for (int gate_mode : {1, 2}) {
                verify([&](auto x) { return linear->forward_input_mul(owner, x, x, gate_mode); });
                if (linear->is_nint() && mode == KlMmqMode::Fp16) {
                    verify([&](auto x) {
                        return linear->forward_input_mul_f32_kld(owner, x, x, gate_mode);
                    });
                }
            }
        }
    }
}

static void check_snapshots(CudaExecutionContext& execution, bool hybrid) {
    Qwen35CausalLm model;
    model.execution = &execution;
    auto full = std::make_unique<FullBlock>();
    full->cache = KVCache(1, 1, 8, 4);
    model.blocks.push_back(std::move(full));
    if (hybrid) {
        auto linear = std::make_unique<qwen35::LinearAttentionBlock>();
        auto& config = linear->qwen_config;
        config.linear_conv_kernel_dim = 2;
        config.linear_num_key_heads = config.linear_num_value_heads = 1;
        config.linear_key_head_dim = config.linear_value_head_dim = 2;
        const auto options = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32);
        linear->mixer.convolution = mfq_tensor_backend::zeros({1, 1, 6}, options);
        linear->mixer.recurrent = mfq_tensor_backend::zeros({1, 1, 2, 2}, options);
        model.blocks.push_back(std::move(linear));
    }
    model.cache_pos = 2;
    const auto text = model.capture_text_session_state({1, 2});
    check(text.kind() == (hybrid ? TextSessionStateKind::HybridAttention
                                : TextSessionStateKind::FullAttention),
          "wrong session codec");
    model.decode_position_delta = -1;
    auto image = model.capture_text_session_state({1, 2});
    image.input_key = "image";
    check(image.decode_position_delta == -1, "image position not captured");

    TextSessionCache memory({}, {});
    memory.store("text", text);
    memory.store("image", image);
    check(memory.restore_best(model, nullptr, "text", {1, 2, 3}, 2).tokens == 2,
          "text snapshot missed");
    check(model.decode_position_delta == 0, "image position leaked into text");
    check(memory.restore_best(model, nullptr, "image", {1, 2, 3}, 2, "image").tokens == 2,
          "image snapshot missed");
    check(model.decode_position_delta == -1, "image position not restored");
    if (hybrid) return;

    bool rejected = false;
    try {
        (void)encode_cuda_paged_session(image, 2, 0);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "paged codec silently discarded media positions");
    const auto decoded = decode_cuda_paged_session(
        encode_cuda_paged_session(text, 2, 0), text.tokens, 2);
    model.restore_text_session_state(decoded);
    check(model.decode_position_delta == 0, "paged text retained media position");

    const auto directory = std::filesystem::temp_directory_path() /
        ("mfq-runtime-state-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        mfq::cache::PagedPrefixCacheConfig config;
        config.cache_dir = directory;
        config.compatibility_key = "runtime-state-test";
        config.block_size_tokens = 2;
        auto paged = std::make_shared<mfq::cache::PagedPrefixCache>(config);
        TextSessionCache mixed({}, {}, paged);
        mixed.store("source", text);
        mixed.store("source", image);
        check(mixed.fork_session("source", "fork") == 2,
              "fork omitted a cache backend");
        check(mixed.close_session("source") == 2,
              "close omitted a cache backend");
        check(mixed.restore_best(model, nullptr, "fork", {1, 2, 3}, 2, "image").tokens == 2,
              "forked memory snapshot missing");
        check(mixed.close_session("fork") == 2, "fork ownership missing");
        // Prefix lookup is cross-session: close both owners before checking.
        check(mixed.restore_best(model, nullptr, "source", {1, 2, 3}, 2, "image").tokens == 0,
              "closed memory snapshot is still reusable");
        mixed.store("source", text);
        mixed.store("source", image);
        check(mixed.clear_live_sessions() == 2, "live clear omitted a backend");
        check(mixed.restore_best(model, nullptr, "source", {1, 2, 3}, 2, "image").tokens == 0,
              "live clear retained memory snapshot");
        mixed.store("source", text);
        mixed.store("source", image);
        check(mixed.clear() >= 1, "clear released nothing");
        check(mixed.restore_best(model, nullptr, "source", {1, 2, 3}, 2, "image").tokens == 0,
              "clear retained memory snapshot");
        check(mixed.restore_best(model, nullptr, "source", {1, 2, 3}, 2).tokens == 0,
              "clear retained paged snapshot");
    }
    {
        mfq::cache::PagedPrefixCacheConfig config;
        config.cache_dir = directory / "vram-ssd";
        config.compatibility_key = "vram-ssd-prefix-test";
        config.block_size_tokens = 2;
        config.max_hot_bytes = 0;
        auto paged = std::make_shared<mfq::cache::PagedPrefixCache>(config);
        mfq::engine::PrefixCacheConfig prefix;
        prefix.hot_bytes = text.bytes * 2;
        TextSessionCache cache({}, prefix, paged, true, 0, true);
        const auto metric = [&](const char* name) {
            for (const auto& entry : cache.metrics()) if (entry.first == name) return entry.second;
            throw std::runtime_error(std::string("missing prefix metric ") + name);
        };
        cache.store("", text);paged->flush();
        check(metric("prefix_cache_hot_bytes") == text.bytes && metric("prefix_cache_ram_bytes") == 0,
              "discrete prefix cache must retain native GPU snapshots without a RAM hot layer");
        check(cache.restore_best(model, nullptr, "", {1,2,3}, 2).tokens == 2 && metric("prefix_cache_hot_hits") == 1,
              "anonymous requests must reuse GPU-hot prefixes");
        check(cache.trim_hot(0) == text.bytes && metric("prefix_cache_hot_bytes") == 0,
              "GPU-hot prefix trim did not release its snapshot");
        check(cache.restore_best(model, nullptr, "", {1,2,3}, 2).tokens == 2,
              "SSD-cold prefix restore failed after GPU eviction");
        check(metric("prefix_cache_hot_bytes") == text.bytes && metric("prefix_cache_ram_bytes") == 0,
              "SSD prefix restore must promote directly into the GPU hot tier");
        check(cache.set_hot_limit(0) == text.bytes,
              "GPU prefix budget change did not release native snapshots");
        check(cache.restore_best(model, nullptr, "", {1,2,3}, 2).tokens == 2 && !metric("prefix_cache_hot_bytes") && !metric("prefix_cache_ram_bytes"),
              "zero GPU-hot budget must retain SSD reuse without RAM residency");
    }
    std::filesystem::remove_all(directory);
}

static void check_media_steps(mfq::engine::Engine& engine, const char* model_path) {
    using namespace mfq::engine;
    if (!engine.info().capabilities.image_input) return;
    const auto source = mfq::open_model_source(model_path);
    const auto config = mfq::models::qwen35::Config::from_source(*source);
    const auto& vision = config.grid_vision.value();
    const auto side = static_cast<int32_t>(vision.spatial_merge_size);
    EngineRequest image;
    image.id = "image";
    image.input.chat_input.emplace().preformatted_prompt =
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image."
        "<|im_end|>\n<|im_start|>assistant\n";
    image.input.sampling.temperature = 0;
    image.input.sampling.top_k = 1;
    image.input.sampling.max_tokens = 8;
    image.input.sampling.enable_mtp = false;
    auto& media = image.input.media.emplace();
    media.processor = MfqMultimodalProcessor::grid_vision;
    media.processor_name = mfq::kMfqGridVisionInputContract;
    media.vision_grid = media.image_grid = {1, side, side};
    media.vision_grid_shape = media.image_grid_shape = {1, 3};
    media.vision_types = {1};
    media.pixel_shape = {side * side, vision.patch_width()};
    media.pixel_values.assign(side * side * vision.patch_width(), 0.1F);
    const auto run = [&] {
        check(engine.admit(image) == Admission::accepted, "image admission");
        std::vector<int64_t> tokens;
        int terminals = 0, preparation_steps = 0;
        bool preparing = true;
        for (int tick = 0; tick < 128 && !terminals; ++tick) {
            auto result = engine.step({image.id});
            if (preparing && result.events.empty() && !result.advanced.empty()) ++preparation_steps;
            for (const auto& event : result.events) {
                if (auto* failure = std::get_if<Failed>(&event.data)) throw std::runtime_error(failure->message);
                if (std::holds_alternative<PrefillProgress>(event.data)) preparing = false;
                if (auto* delta = std::get_if<OutputDelta>(&event.data))
                    tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
                terminals += terminal(event.data);
            }
        }
        check(terminals == 1 && !tokens.empty(), "image request did not finish");
        return std::pair{tokens, preparation_steps};
    };
    const auto [reference, steps] = run();
    check(steps >= vision.depth + 2, "image encoder did not yield between layers");
    for (const auto stop : {int64_t(0), int64_t(2), 2 + vision.depth / 2, 2 + vision.depth}) {
        auto interrupted = image;
        interrupted.input.media->pixel_values[0] += static_cast<float>(stop + 1); // Bypass the image cache.
        check(engine.admit(std::move(interrupted)) == Admission::accepted, "image cancellation admission");
        for (int tick = 0; tick < stop; ++tick) {
            auto result = engine.step({image.id});
            check(result.events.empty() && result.advanced.size() == 1, "image preparation crossed its quantum");
        }
        engine.cancel(image.id);
        auto result = engine.step({});
        check(result.events.size() == 2 && std::holds_alternative<Cancelled>(result.events.back().data),
              "image cancellation did not release before its terminal");
        check(engine.step({}).events.empty(), "image emitted after cancellation");
        check(run().first == reference, "image cancellation changed subsequent output");
    }
    std::cout << "CUDA media step checks passed layers=" << vision.depth << " cancellation=4\n";
}

// Optional real-model check: pass a Qwen3.5 model directory and tokenizer GGUF.
static void check_batching(const char* model_path, const char* tokenizer) {
    CudaEngineOptions options;
    options.model_path = model_path;
    options.tokenizer_model = tokenizer;
    options.context_size = 256;
    options.continuous_batching = 2;
    options.prefill_chunk_size = 8;
    auto engine = load_cuda_engine(options);
    using namespace mfq::engine;
    MfqSamplingParams sampling;
    sampling.max_tokens = 32; sampling.temperature = 0; sampling.top_k = 1; sampling.enable_mtp = false;
    const auto reference = mfq::cuda::diagnostics::check_engine_steps(*engine, {101, 202, 303}, sampling);
    check(reference.size() == 32, "batched generation length");
    check(mfq::cuda::diagnostics::check_engine_steps(*engine, {101, 202, 303}, sampling) == reference,
          "batched repeat changed output");
    EngineRequest cancelled; cancelled.id = "cancel";
    cancelled.token_ids = {101, 202, 303}; cancelled.input.sampling = sampling;
    check(engine->admit(cancelled) == Admission::accepted, "cancel admission");
    engine->cancel("cancel");
    auto result = engine->step({});
    check(std::count_if(result.events.begin(), result.events.end(), [](const auto& event) {
              return terminal(event.data);
          }) == 1 && std::holds_alternative<Cancelled>(result.events.back().data),
          "cancel before prefill did not release");
    cancelled.token_ids.assign(32, 101);
    check(engine->admit(cancelled) == Admission::accepted, "prefill cancel admission");
    for (int tick = 0; tick < 8; ++tick) {
        result = engine->step({"cancel"});
        if (!result.events.empty()) break;
    }
    check(result.events.size() == 1 && std::holds_alternative<PrefillProgress>(result.events[0].data),
          "prefill did not yield after one chunk");
    engine->cancel("cancel");
    result = engine->step({});
    check(std::count_if(result.events.begin(), result.events.end(), [](const auto& event) {
              return terminal(event.data);
          }) == 1 && std::holds_alternative<Cancelled>(result.events.back().data),
          "cancel during prefill did not release");
    check(mfq::cuda::diagnostics::check_engine_steps(*engine, {101, 202, 303}, sampling) == reference,
          "cancel changed subsequent output");
    check_media_steps(*engine, model_path);
}

static void check_retained_recurrent_storage() {
    namespace tb=mfq_tensor_backend;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    qwen4_exp::GdnWeights weights;
    weights.norm=tb::ones({32},options);
    qwen4_exp::Gdn gdn(std::move(weights),1,1,32,4,1e-6,true);
    auto state=gdn.graph_state();
    *state[0]=tb::full({1,3,96},2.0,options);
    *state[1]=tb::full({1,1,32,32},3.0,options);
    const auto conv=state[0]->clone(),recurrent=state[1]->clone();
    const auto* conv_address=state[0]->data_ptr();
    const auto* recurrent_address=state[1]->data_ptr();
    gdn.reset(true);
    check(state[0]->sum().item<float>()==0 && state[1]->sum().item<float>()==0,
        "retained GDN storage leaked the previous request state");
    gdn.restore_session(conv,recurrent,true);
    check(state[0]->data_ptr()==conv_address && state[1]->data_ptr()==recurrent_address,
        "prefix restoration invalidated captured GDN pointers");
    check(state[0]->sum().item<float>()==576 && state[1]->sum().item<float>()==3072,
        "prefix restoration lost GDN state values");
    gdn.reset();
    check(!state[0]->defined() && !state[1]->defined(),"explicit reset retained GDN storage");
}

static void check_kv_memory() {
    namespace tb=mfq_tensor_backend;
    auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    KvMemoryOptions config;config.dynamic=true;
    KvMemoryManager manager(config);
    auto allocation=manager.add(32768,2,128,128,16,4,options);
    manager.finalize();
    const auto bytes=allocation->gpu_bytes();
    auto* address=allocation->keys.data_ptr();
    auto prefix=tb::full({1,2,16,128},3.0,options);
    allocation->keys.narrow(2,0,16).copy_(prefix);
    manager.ensure(24576);
    check(allocation->keys.data_ptr()==address,"dynamic KV moved its virtual address");
    check(allocation->gpu_bytes()>bytes,"dynamic KV failed to map growth pages");
    auto host=allocation->keys.narrow(2,0,16).to(tb::kCPU,tb::kFloat32).contiguous();
    for(int i=0;i<host.numel();++i)check(host.data_ptr<float>()[i]==3.0f,"dynamic KV growth corrupted existing values");
    auto child = manager.add(32768,2,128,128,16,4,options);
    const auto child_bytes = child->gpu_bytes();
    auto* child_address = child->keys.data_ptr();
    check(child_bytes == child->index_bytes(), "additional dynamic KV mapped unadmitted raw pages");
    check(child->growth_bytes(16) > 0 && manager.growth_bytes(16) == child->growth_bytes(16),
        "additional dynamic KV did not expose the full next mapping cost");
    manager.ensure(16);
    check(manager.growth_bytes(16) == 0, "mapped child KV was charged twice");
    check(child->keys.data_ptr() == child_address && child->gpu_bytes() > child_bytes,
        "additional dynamic KV failed to map pages at its stable address");
    child->keys.narrow(2,0,16).copy_(prefix);
    manager.ensure(24576);
    host = child->keys.narrow(2,0,16).to(tb::kCPU,tb::kFloat32).contiguous();
    for (int i = 0; i < host.numel(); ++i)
        check(host.data_ptr<float>()[i] == 3.0f, "additional dynamic KV growth corrupted existing values");
    KvMemoryOptions ram_config;
    ram_config.placement = "ram";
    KvMemoryManager ram_manager(ram_config);
    auto ram = ram_manager.add(128,1,32,32,32,2,options);
    ram_manager.finalize();
    const auto planned_gpu = ram_manager.additional_preallocated_gpu_bytes(128,1,64,64,32,2);
    check(planned_gpu && *planned_gpu == ram->index_bytes(), "RAM KV admission charged raw KV to VRAM");
    const auto ram_child = ram_manager.add(128,1,32,32,32,2,options);
    check(ram_child->gpu_bytes() == *planned_gpu, "RAM KV admission did not preserve placement");
    // Allocation admission must fail before creating any SSD fallback.
    KvMemoryOptions constrained;constrained.gpu_limit=8192;constrained.ram_limit_set=true;constrained.ram_limit=0;
    KvMemoryManager rejected(constrained);
    auto pending=rejected.add(128,1,32,32,32,2,options);
    bool failed=false;
    try {rejected.finalize();} catch(const std::runtime_error& e) {
        failed=std::string(e.what()).find("kv_memory_insufficient:")==0;
    }
    check(failed && !pending->ready,"automatic KV silently exceeded RAM/VRAM capacity");

    // A small resident window deliberately overflows the selected-row union.
    // Both GPU hits and direct RAM reads must agree with a GPU-only oracle.
    for(int width:{32,256})for(double bits:{0.,2.5,4.}) {
        KvMemoryOptions automatic;automatic.auto_stream_threshold=128;automatic.resident_cells=4;
        automatic.ram_limit_set=true;automatic.ram_limit=1<<20;
        KvMemoryManager tiered(automatic);
        const auto floating=options.dtype(tb::kFloat32);
        const int key_bits=int(std::floor(bits)),value_bits=int(std::ceil(bits));
        const int key_width=bits?cuda_kv_packed_width(width,key_bits):width;
        const int value_width=bits?cuda_kv_packed_width(width,value_bits):width;
        auto memory=tiered.add(128,1,key_width,value_width,8,4,options.dtype(bits?tb::kInt32:tb::kFloat16));
        tiered.finalize();
        check(memory->host && memory->hot.defined() && !memory->offload,
            "automatic RAM KV did not keep a graph-compatible resident window");
        check(memory->ram_bytes()==memory->raw_bytes() && memory->gpu_bytes()<memory->raw_bytes()+memory->index_bytes()+4096,
            "automatic RAM KV incorrectly charged the full raw cache to VRAM");
        auto raw=tb::arange(128*width,floating).remainder(23).reshape({1,1,128,width})/23.;
        auto encoded_keys=bits?cuda_kv_encode(raw,key_bits,false):raw.to(tb::kFloat16);
        auto encoded_values=bits?cuda_kv_encode(raw*.5,value_bits,true):(raw*.5).to(tb::kFloat16);
        memory->keys.copy_(encoded_keys);memory->values.copy_(encoded_values);
        auto query=tb::full({1,2,2,width},.125,floating);
        auto selected=tb::tensor(std::vector<int32_t>{1,2,3,4,127,2,1,5,6,126},options.dtype(tb::kInt32)).reshape({1,2,5});
        const auto check_published=[&](const std::vector<int>& rows) {
            const auto tags=memory->hot.mapping.cpu().contiguous();
            const auto slots=memory->hot.slot_ids.cpu().contiguous();
            for(int row:rows)check(tags.data_ptr<int32_t>()[row]>=0,
                "published KV row was not resident");
            for(const auto& pair:{std::pair{memory->keys,memory->hot.keys},
                                  std::pair{memory->values,memory->hot.values}}) {
                const auto raw=pair.first.cpu().contiguous(),hot=pair.second.cpu().contiguous();
                const auto bytes=raw.size(3)*raw.element_size();
                for(int row=0;row<128;++row) {
                    const int slot=tags.data_ptr<int32_t>()[row];if(slot<0)continue;
                    check(slot<4 && slots.data_ptr<int32_t>()[slot]==row,
                        "published KV has inconsistent row/slot ownership");
                    check(std::memcmp(static_cast<const char*>(raw.data_ptr())+row*bytes,
                        static_cast<const char*>(hot.data_ptr())+slot*bytes,bytes)==0,
                        "published KV retained stale encoded data");
                }
            }
        };
        memory->hot.publish(memory->keys,memory->values,tb::arange(4,options.dtype(tb::kInt64)));
        check_published({0,1,2,3});
        auto prefix=tb::tensor(std::vector<int32_t>{0,1,2,3},options.dtype(tb::kInt32)).reshape({1,1,4});
        memory->hot.prepare(memory->keys,memory->values,prefix,
            tb::tensor(std::vector<int64_t>{3},options.dtype(tb::kInt64)),128,true);
        check(memory->hot.control.cpu().contiguous().data_ptr<int64_t>()[2]==0,
            "resident short prefix still queued RAM reads");
        // One append larger than the cache must retain the final window.
        memory->hot.publish(memory->keys,memory->values,tb::arange(10,options.dtype(tb::kInt64)));
        check_published({6,7,8,9});
        const auto wrapped=memory->hot.mapping.cpu().contiguous();
        for(int row=0;row<6;++row)check(wrapped.data_ptr<int32_t>()[row]<0,
            "wrapped KV append retained an overwritten row tag");
        const auto compare=[&] {
            auto key=memory->keys.clone(),value=memory->values.clone();
            memory->hot.prepare(memory->keys,memory->values,selected);
            auto expected=bits?cuda_kv_attention(query,key,value,KvQuantization{double(bits)},selected):
                mfq_qwen4_exp::sparse_gqa_attention(query,key,value,selected);
            auto actual=bits?cuda_kv_attention(query,memory->keys,memory->values,KvQuantization{double(bits)},selected,
                {},0,{},0,{},{},0,memory->hot.view()):
                mfq_qwen4_exp::sparse_gqa_attention(query,memory->keys,memory->values,selected,memory->hot.view());
            auto a=actual.cpu().contiguous(),e=expected.cpu().contiguous();
            for(int64_t i=0;i<a.numel();++i)check(std::abs(a.data_ptr<float>()[i]-e.data_ptr<float>()[i])<1e-5,
                "resident/RAM selected attention changed the GPU-only result");
        };
        compare();compare();
        memory->hot.prepare(memory->keys,memory->values,
            tb::tensor(std::vector<int32_t>{1},options.dtype(tb::kInt32)).reshape({1,1,1}));
        check(memory->hot.mapping.cpu().contiguous().data_ptr<int32_t>()[1]>=0,
            "KV overwrite test did not begin with a resident old row");
        auto replacement=tb::full({1,1,1,width},.875,floating);
        memory->keys.narrow(2,1,1).copy_(bits?cuda_kv_encode(replacement,key_bits,false):replacement.to(tb::kFloat16));
        memory->values.narrow(2,1,1).copy_(bits?cuda_kv_encode(replacement,value_bits,true):replacement.to(tb::kFloat16));
        memory->hot.publish(memory->keys,memory->values,
            tb::tensor(std::vector<int64_t>{1},options.dtype(tb::kInt64)));
        check_published({1});
        // Rewind below the hot-window boundary after a long-prefix lookup.
        memory->hot.publish(memory->keys,memory->values,tb::arange(2,options.dtype(tb::kInt64)));
        check_published({0,1});
        compare();memory->hot.reset();compare();
        auto counters=memory->hot.control.cpu().contiguous();
        check(counters.data_ptr<int64_t>()[4]>0 && counters.data_ptr<int64_t>()[5]>0,
            "resident KV test failed to exercise cold fills and bounded overflow");
    }
    KvMemoryOptions no_ram;no_ram.auto_stream_threshold=128;no_ram.resident_cells=4;
    no_ram.ram_limit_set=true;no_ram.ram_limit=0;
    KvMemoryManager gpu_only(no_ram);
    auto resident=gpu_only.add(128,1,32,32,8,4,options);gpu_only.finalize();
    check(!resident->host && !resident->hot.defined(),"automatic KV ignored an unavailable RAM budget");
}

static void check_exclusive_kv() {
    namespace tb=mfq_tensor_backend;
    const auto options=tb::TensorOptions().device(tb::kCUDA);
    for(int width:{32,256})for(double bits:{0.,2.5,4.}) {
        const int maximum=128,heads=2,cells=5;
        const int kw=bits?cuda_kv_packed_width(width,int(std::floor(bits))):width;
        const int vw=bits?cuda_kv_packed_width(width,int(std::ceil(bits))):width;
        KvMemoryOptions policy;policy.budget_policy=true;policy.ram_limit_set=true;policy.ram_limit=1<<20;
        policy.gpu_budget=(maximum+maximum/4)*8*2+cells*heads*(kw+vw)*(bits?4:2);
        KvMemoryManager manager(policy);
        auto memory=manager.add(maximum,heads,kw,vw,8,4,options.dtype(bits?tb::kInt32:tb::kFloat16));manager.finalize();
        check(memory->hot.exclusive(),"budgeted KV did not use exclusive GPU/RAM tiers");
        check(memory->ram_bytes()==std::size_t(maximum-cells)*heads*(kw+vw)*(bits?4:2),"RAM KV duplicated GPU capacity");
        auto raw=tb::arange(heads*maximum*width,options.dtype(tb::kFloat32)).remainder(29).reshape({1,heads,maximum,width})/29.;
        auto k=bits?cuda_kv_encode(raw,int(std::floor(bits)),false):raw.to(tb::kFloat16);
        auto v=bits?cuda_kv_encode(raw*.5,int(std::ceil(bits)),true):(raw*.5).to(tb::kFloat16);
        auto positions=tb::arange(maximum,options.dtype(tb::kInt64));
        memory->hot.append(k.narrow(2,0,31),v.narrow(2,0,31),positions.narrow(0,0,31));
        memory->hot.append(k.narrow(2,31,97),v.narrow(2,31,97),positions.narrow(0,31,97));
        auto selected=tb::tensor(std::vector<int32_t>{1,2,3,4,127,2,1,5,6,126},options.dtype(tb::kInt32)).reshape({1,2,5});
        auto query=tb::full({1,4,2,width},.125,options.dtype(tb::kFloat32));
        const auto compare=[&] {
            memory->hot.prepare(memory->keys,memory->values,selected);
            auto expected=bits?cuda_kv_attention(query,k,v,KvQuantization{bits},selected):mfq_qwen4_exp::sparse_gqa_attention(query,k,v,selected);
            auto actual=bits?cuda_kv_attention(query,memory->keys,memory->values,KvQuantization{bits},selected,{},0,{},0,{},{},0,memory->hot.view()):
                mfq_qwen4_exp::sparse_gqa_attention(query,memory->keys,memory->values,selected,memory->hot.view());
            auto a=actual.cpu().contiguous(),e=expected.cpu().contiguous();
            for(int64_t i=0;i<a.numel();++i)check(std::abs(a.data_ptr<float>()[i]-e.data_ptr<float>()[i])<1e-5,"exclusive KV attention differs from GPU-only oracle");
            auto saved=memory->hot.gather(maximum);
            for(const auto& pair:{std::pair{saved.first,k},std::pair{saved.second,v}}) {
                auto found=pair.first.cpu().contiguous(),oracle=pair.second.cpu().contiguous();
                check(std::memcmp(found.data_ptr(),oracle.data_ptr(),found.nbytes())==0,"exclusive KV migration corrupted encoded rows");
            }
        };
        compare();compare();
        // Rewrites at full capacity must reuse free GPU slots as well as RAM
        // slots; a fixed ring destination would require an extra cold row.
        for(int row:{1,127,4,126,2,3}) {
            auto kr=k.narrow(2,row,1).contiguous(),vr=v.narrow(2,row,1).contiguous();
            memory->hot.append(kr,vr,positions.narrow(0,row,1));compare();
        }
        memory->hot.reset();memory->hot.append(k,v,positions);compare();
    }
}

static void check_decode_kv_append() {
    namespace tb=mfq_tensor_backend;
    auto stream=mfq_get_stream_from_pool();MfqCudaStreamGuard guard(stream);
    auto context=default_context();context->begin_graph_pool(stream.stream());
    constexpr int maximum=96,cells=32,heads=2,width=32,iw=8;
    const auto options=tb::TensorOptions().device(tb::kCUDA);
    for(auto type:{tb::kFloat16,tb::kFloat32,tb::kInt32})for(auto index_type:{tb::kFloat16,tb::kFloat32})
    for(bool cast:{false,true}) {
        if(cast && type!=tb::kFloat16)continue;
        auto hot=HotKvCache::allocate(heads,maximum,cells,width,width,options.dtype(type));
        hot.set_cold_storage(tb::empty({1,heads,maximum-cells,width},options.dtype(type)),
            tb::empty({1,heads,maximum-cells,width},options.dtype(type)));
        // Both KV and indexer inputs retain a larger row stride.
        auto k=tb::arange(heads*maximum*(width+8),options.dtype(tb::kFloat32)).remainder(71)
            .reshape({1,maximum,heads,width+8}).narrow(3,0,width).permute({0,2,1,3}).to(type);
        if(cast)k=k.to(tb::kFloat32)+.00031;
        auto v=(k.to(tb::kFloat32)*2).to(type);
        if(cast)v=v.to(tb::kFloat32)-.00039;
        auto raw=tb::arange(maximum*iw*3,options.dtype(tb::kFloat32)).remainder(53)
            .reshape({1,maximum,iw*3}).to(index_type).narrow(2,iw,iw);
        auto stored=tb::zeros({1,maximum,iw},options.dtype(index_type==tb::kFloat16?tb::kFloat32:tb::kFloat16));
        auto positions=tb::arange(maximum,options.dtype(tb::kInt64));
        int begin=0;
        for(int count:{1,2,3,4,5,6,11,32,32}) {
            const bool fused=hot.append_decode(k.narrow(2,begin,count),v.narrow(2,begin,count),positions.narrow(0,begin,count),
                raw.narrow(1,begin,count),stored);
            check(fused || count>8,"short KV append did not dispatch");
            if(!fused) {
                hot.append(k.narrow(2,begin,count).to(type),v.narrow(2,begin,count).to(type),positions.narrow(0,begin,count));
                stored.index_copy_(1,positions.narrow(0,begin,count),raw.narrow(1,begin,count).to(stored.scalar_type()));
            }
            begin+=count;
        }
        const auto compare=[&] {
            auto saved=hot.gather(maximum);
            for(const auto& pair:{std::pair{saved.first,k.to(type)},std::pair{saved.second,v.to(type)},std::pair{stored,raw.to(stored.scalar_type())}}) {
                auto a=pair.first.contiguous().cpu(),b=pair.second.contiguous().cpu();
                check(a.nbytes()==b.nbytes() && !std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),
                    "fused KV/index append changed stored rows");
            }
        };
        compare();
        // Replay the same graph at new positions, rewriting both RAM and VRAM
        // rows at full capacity; every cold allocation must reuse freed space.
        auto pos=positions.narrow(0,0,6).clone();
        auto key=k.narrow(2,0,6).contiguous(),value=v.narrow(2,0,6).contiguous();
        auto index=raw.narrow(1,0,6).contiguous();
        auto source_k=k.to(type).contiguous(),source_v=v.to(type).contiguous();
        DecodeWindow window(mfq_current_cuda_stream(),false);
        window.capture([&] {
            check(hot.append_decode(key,value,pos,index,stored,true),"captured KV append rejected");
            hot.prepare(source_k,source_v,pos.reshape({1,1,6}),pos,6,true,true);
        },[]{},[]{});
        for(int start:{0,90,31,12,60,1}) {
            pos.copy_(positions.narrow(0,start,6));key.copy_(k.narrow(2,start,6));
            value.copy_(v.narrow(2,start,6));index.copy_(raw.narrow(1,start,6));
            window.run();compare();
        }
    }
    context->end_graph_pool(stream.stream());
}

static void check_kv_expert_loan() {
    namespace tb=mfq_tensor_backend;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    auto arena=tb::zeros({512<<10},options),destination=tb::empty({128<<10},options);
    auto source=tb::full({128<<10},73,options.device(tb::kCPU));
    Stream copy(0);KvExpertLoan loan;loan.add(arena,128,4096,true);
    bool hit=false;
    check(loan.copy({0,1,2},source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit)
        && !hit && loan.bytes()==source.nbytes(),"unused KV storage did not accept an expert field");
    check(loan.copy({0,1,2},source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit) && hit,"extra expert cache missed a resident field");
    int uploads=0;
    const auto stage=[&]() -> const void* {++uploads;return source.data_ptr();};
    const auto* cached=loan.pin({0,2,2},nullptr,source.nbytes(),copy.get(),hit,nullptr,stage);
    check(cached && !hit && uploads==1,"pageable expert miss did not request its staged source");
    check(loan.pin({0,2,2},nullptr,source.nbytes(),copy.get(),hit,nullptr,stage)==cached && hit && uploads==1,
        "pageable expert hit staged its CPU weights again");
    loan.ensure(128);arena.zero_();
    check(!loan.pin({0,3,2},nullptr,source.nbytes(),copy.get(),hit,nullptr,stage) && uploads==1,
        "unavailable expert cache unnecessarily staged CPU weights");
    check(loan.bytes()==0 && !loan.copy({0,1,2},source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit),"KV growth did not reclaim expert pages");
    auto result=destination.cpu().contiguous();
    for(int64_t i=0;i<result.numel();++i)check(result.data_ptr<uint8_t>()[i]==73,"KV reclamation raced an expert transfer");
    KvOffloadConfig config;config.preallocate=true;config.ssd_enabled=false;
    config.buffer_bytes=256<<10;config.gpu_budget_bytes=(512<<10)+config.buffer_bytes;
    KvOffloadStore store(config);auto shared=std::make_shared<KvExpertLoan>();store.set_expert_loan(shared);
    check(shared->copy({1,1,2},source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit),"streamed KV arena did not lend unused storage");
    check(store.expert_loan_bytes()==source.nbytes(),"streamed KV loan accounting is incorrect");
    auto rows=tb::full({128,512},7,options.dtype(tb::kFloat16));auto block=store.write(rows);
    check(store.expert_loan_bytes()==0,"streamed KV growth did not reclaim its expert pages");
    auto recovered=store.read(block).to(tb::kCPU,tb::kFloat32).contiguous();
    for(int64_t i=0;i<recovered.numel();++i)check(recovered.data_ptr<float>()[i]==7,"streamed KV overwrote a live expert transfer");

    KvExpertLoan retired;auto retirement_arena=tb::zeros({512<<10},options);
    retired.add(retirement_arena,128,4096,true);
    for(const auto key:{KvExpertLoan::Key{1,2,0},{1,2,1},{2,2,0},{1,3,0}})
        check(retired.copy(key,source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit) && !hit,
            "retirement fixture did not fill all expert pages");
    retired.retire_expert(1,2,2);retired.retire_expert(1,2,2);
    check(retired.bytes()==4*source.nbytes(),"promotion reclaimed an in-flight expert address");
    // Reuse before the next publication cancels retirement for that field.
    check(retired.pin({1,2,1},source.data_ptr(),source.nbytes(),copy.get(),hit) && hit,
        "retired expert address vanished before its packet completed");
    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));
    retired.next_packet();
    check(retired.bytes()==3*source.nbytes() && retired.retired_bytes()==source.nbytes(),
        "promotion retirement reclaimed the wrong expert fields");
    for(const auto key:{KvExpertLoan::Key{1,2,1},{2,2,0},{1,3,0}})
        check(retired.pin(key,source.data_ptr(),source.nbytes(),copy.get(),hit) && hit,
            "retirement evicted an unrelated or reused field");
    check(retired.copy({3,4,0},source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit) && !hit,
        "promotion did not return reusable expert pages");
    retired.retire_expert(3,4,1);
    retired.ensure(128);retirement_arena.zero_();
    check(retired.bytes()==0,"KV growth did not revoke a retired expert field");
    result=destination.cpu().contiguous();
    for(int64_t i=0;i<result.numel();++i)check(result.data_ptr<uint8_t>()[i]==73,"retirement raced a pending expert copy");

    // Non-KV expert pages must be independently reclaimable for graph
    // widening, without dropping unrelated KV-backed entries or accounting
    // their bytes as occupied KV storage.
    KvExpertLoan flexible;
    flexible.add(tb::zeros({256<<10},options),64,4096,true);
    check(flexible.copy({10,1,0},source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit),
        "flexible fixture did not cache its KV-backed expert");
    const auto region=flexible.add(tb::zeros({256<<10},options),256<<10,1,true,1,true,true);
    check(flexible.copy({10,2,0},source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit),
        "flexible fixture did not fill the remaining KV pages");
    check(flexible.copy({11,3,0},source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit),
        "flexible expert allocation was not usable");
    check(flexible.external_bytes()==source.nbytes(),"non-KV expert pages were not accounted separately");
    const auto other_region=flexible.add(tb::zeros({256<<10},options),256<<10,1,true,1,true,true);
    for(int expert:{4,5})check(flexible.copy({11,expert,0},source.data_ptr(),destination.data_ptr(),
        source.nbytes(),copy.get(),hit),"second external expert allocation was not usable");
    check(flexible.region_bytes(region)==2*source.nbytes() &&
        flexible.region_bytes(other_region)==source.nbytes() && flexible.external_bytes()==3*source.nbytes(),
        "independent graph and transfer regions were combined in per-region accounting");
    flexible.remove(other_region);
    flexible.remove(region);
    check(flexible.external_bytes()==0 && flexible.bytes()==2*source.nbytes(),
        "reclaiming graph capacity erased unrelated KV experts");
    check(flexible.pin({10,1,0},source.data_ptr(),source.nbytes(),copy.get(),hit) && hit,
        "reclaiming graph capacity invalidated an unrelated expert address");
    result=destination.cpu().contiguous();
    for(int64_t i=0;i<result.numel();++i)check(result.data_ptr<uint8_t>()[i]==73,
        "graph capacity reclamation raced a pending expert upload");
    const auto reused=flexible.add(tb::zeros({256<<10},options),256<<10,1,true,1,true,true);
    check(reused==region,"repeated depth changes leaked expert-region descriptors");
    check(flexible.copy({11,4,0},source.data_ptr(),destination.data_ptr(),source.nbytes(),copy.get(),hit),
        "reclaimed graph capacity was not reusable by experts");
    check(flexible.external_bytes()==source.nbytes(),"reused graph capacity has incorrect accounting");
    flexible.remove(reused);
    {
        KvExpertLoan protected_cache;
        protected_cache.add(tb::zeros({256<<10},options),256<<10,1,true,1,true,true);
        for(int expert:{1,2})check(protected_cache.pin({31,expert,0},source.data_ptr(),source.nbytes(),copy.get(),hit),
            "exchange-read fixture did not fill its cache");
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));
        const auto* borrowed=protected_cache.find_for_exchange({31,1,0},source.nbytes());
        std::optional<KvExpertLoan::ExchangeReadGuard> guard(protected_cache.retain_exchange_reads());
        protected_cache.next_packet();protected_cache.retire_expert(31,1,1);protected_cache.next_packet();
        check(protected_cache.find_for_exchange({31,1,0},source.nbytes())==borrowed,
            "predictor packet reclaimed an exchange's retired source");
        check(!protected_cache.pin({31,3,0},source.data_ptr(),source.nbytes(),copy.get(),hit),
            "predictor pressure evicted an exchange's source instead of using its wire fallback");
        guard.reset();protected_cache.next_packet();
        check(!protected_cache.find_for_exchange({31,1,0},source.nbytes()) &&
            protected_cache.pin({31,3,0},source.data_ptr(),source.nbytes(),copy.get(),hit),
            "finished exchange kept blocking ordinary cache reclamation");
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));
        std::optional<KvExpertLoan::ExchangeReadGuard> growth(protected_cache.retain_exchange_reads());
        std::atomic<bool> released=false;
        std::thread releaser([hold=std::move(growth),&released]() mutable {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            released=true;hold.reset();
        });
        protected_cache.protect(0,256<<10);
        const bool respected=released.load();releaser.join();
        check(respected && !protected_cache.bytes(),"KV growth reclaimed an exchange source before it finished");
    }
    {
        KvExpertLoan scoped;
        scoped.add(tb::zeros({256<<10},options),256<<10,1,true,1,true,true);
        for(int expert:{1,2})check(scoped.pin({41,expert,0},source.data_ptr(),source.nbytes(),copy.get(),hit),
            "scoped exchange fixture could not fill its cache");
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));scoped.next_packet();
        std::optional<KvExpertLoan::ExchangeReadGuard> guard(scoped.retain_exchange_reads(false));
        const auto* borrowed=scoped.find_for_exchange({41,1,0},source.nbytes(),&*guard);
        check(borrowed && scoped.pin({41,3,0},source.data_ptr(),source.nbytes(),copy.get(),hit) &&
            !scoped.find_for_exchange({41,2,0},source.nbytes()) &&
            scoped.find_for_exchange({41,1,0},source.nbytes())==borrowed,
            "scoped exchange pinned unrelated entries or lost its borrowed source");
        // Nonadjacent repeated pins must balance, including retired entries.
        check(scoped.find_for_exchange({41,3,0},source.nbytes(),&*guard),"second scoped pin failed");
        check(scoped.find_for_exchange({41,1,0},source.nbytes(),&*guard)==borrowed,"repeated scoped pin changed address");
        scoped.retire_expert(41,1,1);scoped.next_packet();
        check(scoped.find_for_exchange({41,1,0},source.nbytes())==borrowed,
            "scoped retirement freed a borrowed source");
        guard.reset();scoped.next_packet();
        check(!scoped.find_for_exchange({41,1,0},source.nbytes()) &&
            scoped.pin({41,4,0},source.data_ptr(),source.nbytes(),copy.get(),hit),
            "released scoped pins kept their retired payloads alive");
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));
    }
}

static void check_kv_projection_loan() {
    namespace tb=mfq_tensor_backend;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    auto arena=tb::zeros({512<<10},options),source=tb::full({128<<10},73,options.device(tb::kCPU));
    auto destination=tb::empty({128<<10},options);
    Stream copy(0);KvExpertLoan loan;loan.add(arena,128,4096,true);
    const std::vector<int64_t> sizes={64<<10,17,4097,0};
    std::map<int64_t,const void*> visible;
    int publications=0,invalidations=0;
    loan.observe_projections(7,[&](int64_t expert,const void* pointer,const auto&) {
        if(pointer){visible[expert]=pointer;++publications;}
        else {visible.erase(expert);++invalidations;}
    });
    int uploads=0;bool hit=false;
    std::array<std::size_t,8> existing_offsets{};std::size_t existing_bytes=0;
    check(!loan.pin_existing_projection(7,11,sizes,existing_offsets,existing_bytes) && !loan.bytes(),
        "lookup-only miss allocated a projection");
    const auto fill=[&](void* pointer,const std::array<std::size_t,8>& offsets) {
        ++uploads;
        for(std::size_t f=0;f<sizes.size();++f)if(sizes[f])
            MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(pointer)+offsets[f],source.data_ptr(),
                std::size_t(sizes[f]),cudaMemcpyHostToDevice,copy.get()));
    };
    const auto* first=loan.pin_projection(7,11,sizes,copy.get(),hit,fill);
    check(first && !hit && uploads==1,"projection miss did not populate exactly once");
    check(visible.empty(),"new projection became GPU-visible before its reader boundary");
    for(std::size_t f=0;f<sizes.size();++f)if(sizes[f]) {
        const auto* field=loan.find_for_exchange({7,11,int64_t(f)},std::size_t(sizes[f]));
        check(field!=nullptr,"exchange could not find a compact projection field");
        MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(destination.data_ptr(),field,std::size_t(sizes[f]),cudaMemcpyDeviceToDevice,copy.get()));
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));
        auto value=destination.narrow(0,0,sizes[f]).cpu();
        for(int64_t i=0;i<sizes[f];++i)check(value.data_ptr<uint8_t>()[i]==73,"compact metadata bytes changed");
    }
    check(loan.pin_projection(7,11,sizes,copy.get(),hit,fill)==first && hit && uploads==1,
        "complete projection hit re-uploaded small metadata");
    check(loan.pin_existing_projection(7,11,sizes,existing_offsets,existing_bytes)==first &&
        existing_bytes==(64<<10)+17+4097 && uploads==1,"early projection hit changed bytes or staged RAM");
    for(std::size_t f=0;f<sizes.size();++f)if(sizes[f])
        check(static_cast<const uint8_t*>(first)+existing_offsets[f]==loan.find_for_exchange({7,11,int64_t(f)},sizes[f]),
            "early projection offsets differ from exchange lookup");
    bool rejected=false;auto invalid=sizes;invalid[1]=18;
    try{loan.pin_projection(7,11,invalid,copy.get(),hit,fill);}catch(const std::logic_error&){rejected=true;}
    check(rejected && uploads==1,"same-size projection layout mismatch was accepted");
    rejected=false;
    try{loan.pin_existing_projection(7,11,invalid,existing_offsets,existing_bytes);}catch(const std::logic_error&){rejected=true;}
    check(rejected,"early projection hit ignored changed field geometry");
    check(loan.pin_projection(7,12,sizes,copy.get(),hit,fill) && !hit,"second projection did not fit");
    check(loan.pin_projection(7,13,sizes,copy.get(),hit,fill) && !hit,"third projection did not fit");
    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));
    loan.next_packet();loan.protect_packet_fields({{7,11,-1}});
    check(publications==3 && visible.at(11)==first,"completed projections were not published at the route boundary");
    int late_observed=0;
    loan.observe_projections(7,[&](int64_t,const void* pointer,const auto&){late_observed+=pointer?1:-1;});
    check(late_observed==3,"new observer missed already published projections");
    auto observer_owner=std::make_shared<int>(0);
    int scoped_observed=0;
    loan.observe_projections(7,[&](int64_t,const void*,const auto&){++scoped_observed;},observer_owner);
    check(scoped_observed==3,"scoped observer missed published projections");
    observer_owner.reset();
    const std::vector<int64_t> large={192<<10};
    const auto large_fill=[&](void* pointer,const auto&) {++uploads;
        MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(pointer,41,192<<10,copy.get()));};
    check(loan.pin_projection(8,1,large,copy.get(),hit,large_fill) && !hit,"projection eviction did not recover contiguous pages");
    check(loan.pin_projection(7,11,sizes,copy.get(),hit,fill)==first && hit && uploads==4,
        "an earlier miss evicted a protected later projection");
    check(!loan.find_for_exchange({7,12,0},sizes[0]) && !loan.find_for_exchange({7,13,0},sizes[0]),
          "projection LRU did not evict unprotected entries");
    check(visible.size()==1 && invalidations==2 && late_observed==1,"evicted projections stayed GPU-visible");
    check(scoped_observed==3,"released graph metadata still received projection callbacks");
    loan.retire_expert(7,11,4);loan.retire_expert(7,11,4);
    check(visible.empty() && invalidations==3,"retirement did not invalidate GPU eligibility exactly once");
    check(loan.find_for_exchange({7,11,1},sizes[1])!=nullptr,"retirement freed a live compact field");
    check(loan.pin_existing_projection(7,11,sizes,existing_offsets,existing_bytes)==first,
          "early projection hit could not revive a deferred retirement");
    check(visible.empty(),"revived projection was republished before its reader boundary");
    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));loan.next_packet();
    check(visible.at(11)==first && publications==4,"revived projection did not become visible at the next boundary");
    loan.retire_expert(7,11,4);
    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));
    loan.next_packet();check(!loan.find_for_exchange({7,11,1},sizes[1]),"retired projection survived its packet");
    loan.ensure(128);arena.zero_();check(loan.bytes()==0,"KV growth did not reclaim compact projections");
    check(!loan.pin_existing_projection(8,1,large,existing_offsets,existing_bytes),"early lookup returned KV-reclaimed pages");
    check(!loan.pin_projection(7,11,sizes,copy.get(),hit,fill) && uploads==4,"failed projection allocation staged data");
    std::cout<<"KV compact projection bytes, metadata hits, geometry, packet protection, LRU, retirement and KV growth PASS\n";
}

static void check_reused_projection_observer() {
    namespace tb=mfq_tensor_backend;
    auto storage=tb::empty({4*int64_t(KvExpertLoan::page)},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
    KvExpertLoan loan;loan.set_reuse_priority(true);
    const auto region=loan.add(storage,4,KvExpertLoan::page,true);
    std::map<int64_t,bool> reusable;int publications=0;
    loan.observe_projections(3,[&](int64_t,const void* base,const auto&){if(base)++publications;});
    loan.observe_projections(3,[&](int64_t expert,const void* base,const auto&){reusable[expert]=base!=nullptr;},{},true);
    const std::vector<int64_t> sizes={int64_t(KvExpertLoan::page)};
    std::array<std::size_t,8> offsets{};std::size_t logical=0;bool hit=false;
    const auto stream=mfq_current_cuda_stream();
    for(int64_t expert=0;expert<4;++expert) {
        check(loan.pin_projection(3,expert,sizes,stream,hit,[&](void* base,const auto&) {
            MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(base,int(expert),KvExpertLoan::page,stream));
        }) && !hit,"reusable projection fixture failed to allocate");
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));loan.next_packet();
        check(!reusable.at(expert),"one-use GPU projection displaced a protected RAM image");
        check(loan.pin_existing_projection(3,expert,sizes,offsets,logical)!=nullptr && reusable.at(expert),
            "repeated GPU projection was not offered as a RAM eviction candidate");
        check(publications==expert+1,"reuse notifications changed ordinary projection publication");
    }
    check(!reusable.at(0),"GPU reuse quota demotion retained a RAM eviction hint");
    loan.retire_expert(3,3,1);
    check(!reusable.at(3),"retired GPU projection retained a RAM eviction hint");
    loan.set_reuse_priority(false);
    for(const auto& [expert,valid]:reusable)check(!valid,"disabled GPU protection retained an eviction hint");
    loan.remove(region);
    std::cout<<"GPU projection reuse eligibility, quota demotion and retirement observers PASS\n";
}

static void check_workspace_loan() {
    using namespace mfq_tensor_backend;
    const auto context=mfq::cuda::default_context();
    const auto stream=mfq_current_cuda_stream();
    auto arena=full({4096},7,TensorOptions().device(kCUDA).dtype(kUInt8));
    const auto address=arena.data_ptr();
    context->begin_workspace_loan({{arena.data_ptr(),arena.nbytes()}},stream);
    const auto charged=context->memory_stats().allocated;
    auto a=empty({1024},arena.options()),b=empty({1024},arena.options());
    check(a.data_ptr()==address && b.data_ptr()==static_cast<uint8_t*>(address)+1024,
          "prefill did not borrow expert storage");
    check(context->memory_stats().allocated==charged,"loaned bytes were charged twice");
    a.zero_();b.zero_();a={};b={};
    const auto touched=context->workspace_touched_regions();
    check(touched.size()==1 && touched[0].first==address && touched[0].second==2048,
          "prefill restore included unused expert storage or lost a freed slice");
    auto whole=empty({4096},arena.options());
    check(whole.data_ptr()==address,"workspace slices did not coalesce for the next phase");
    bool rejected=false;
    try{context->end_workspace_loan();}catch(const std::exception&){rejected=true;}
    check(rejected,"expert storage was returned with a live workspace tensor");
    whole={};
    {
        mfq::cuda::ResidentAllocationScope resident;
        auto state=empty({4096},arena.options());
        check(state.data_ptr()!=address,"persistent request state borrowed expert storage");
    }
    context->suspend_workspace_loan();
    auto detached=empty({4096},arena.options());
    check(detached.data_ptr()!=address,"escaped prefill output still aliases experts");
    context->end_workspace_loan();
    arena.copy_(full({4096},7,arena.options()));
    check(arena.data_ptr()==address && arena.cpu().data_ptr<uint8_t>()[2048]==7,
          "restoring expert bytes changed its address or contents");
}

static void check_graph_workspace_reuse() {
    namespace tb=mfq_tensor_backend;
    auto context=default_context();const auto stream=mfq_current_cuda_stream();
    const auto before=context->memory_stats().allocated;
    context->begin_graph_pool(stream);
    tb::Tensor retained;
    {
        GraphWarmupScope warmup(context,stream);
        auto large=tb::empty({65536},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
        const auto* address=large.data_ptr();large={};
        retained=tb::empty({49152},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
        check(retained.data_ptr()==address,"smaller graph shape did not reuse available scratch");
        retained.fill_(7);
    }
    context->end_graph_pool(stream);
    check(context->memory_stats().allocated==before+65536,"retained graph block lost its allocation capacity");
    check(retained.cpu().data_ptr<uint8_t>()[49151]==7,"retired graph pool invalidated a live tensor");
    retained={};context->reclaim_idle_memory();
    check(context->memory_stats().allocated==before,"oversized graph block was not fully released");
}

static void check_streamed_graph_segments() {
    namespace tb=mfq_tensor_backend;
    auto context=default_context();const auto stream=mfq_current_cuda_stream();
    context->begin_graph_pool(stream);
    {
        auto input=tb::ones({16},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32));
        auto first=tb::empty_like(input),second=tb::empty_like(input),output=tb::empty_like(input);
        int calls=0;bool fail=false;
        DecodeWindow window(stream);
        window.capture([&] {
            first.copy_(input*2);
            window.host_barrier([&] {
                ++calls;
                check(first.cpu().data_ptr<float>()[0]==input.cpu().data_ptr<float>()[0]*2,"host I/O ran before its GPU producer");
                if(fail)throw std::runtime_error("injected KV read failure");
                first.copy_(first+3);
            });
            second.copy_(first*4);
            window.host_barrier([&] {++calls;second.copy_(second+5);});
            output.copy_(second*6);
        },[]{});
        check(calls==0 && window.segments()==3,"capture executed host I/O or lost a boundary");
        for(int i=1;i<=3;++i) {
            input.fill_(i);window.run();
            auto host=output.cpu();
            for(int j=0;j<16;++j)check(host.data_ptr<float>()[j]==((i*2+3)*4+5)*6,"segmented graph replay changed its data dependency");
        }
        check(calls==6,"host I/O was duplicated during replay");
        fail=true;bool caught=false;
        try {window.run();} catch(const std::runtime_error& error) {caught=std::string(error.what())=="injected KV read failure";}
        check(caught && calls==7,"failed I/O ran a dependent segment");
    }
    context->end_graph_pool(stream);
    for(const auto dtype:{tb::kFloat16,tb::kInt32})for(bool preallocate_ram:{false,true}) {
        KvOffloadConfig config;config.preallocate=true;config.ram_preallocated=preallocate_ram;
        config.buffer_bytes=4096;config.gpu_budget_bytes=6144;config.ram_budget_bytes=1024;
        config.directory=(std::filesystem::current_path()/"streamed-kv-test").string();
        auto store=std::make_shared<KvOffloadStore>(config);KvOffloadSequence sequence(store,128,2,dtype);
        auto source=tb::arange(64*128,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64)).remainder(997).to(dtype).reshape({64,128});
        sequence.append(source);
        for(int planar:{0,32}) {
            auto workspace=sequence.gather_workspace(16,2,planar);
            for(int pass=0;pass<2;++pass) {
                std::vector<int64_t> selected{63,0,7,62,12,0,61};
                if(pass)std::reverse(selected.begin(),selected.end());
                sequence.gather_into(selected,*workspace);
                tb::Tensor actual;
                if(planar) {
                    auto flat=workspace->rows.reshape({-1});
                    auto keys=flat.narrow(0,0,16*64).reshape({2,16,32}).permute({1,0,2}).reshape({16,64});
                    auto values=flat.narrow(0,16*64,16*64).reshape({2,16,32}).permute({1,0,2}).reshape({16,64});
                    actual=tb::cat({keys,values},1).narrow(0,0,selected.size());
                } else actual=workspace->rows.narrow(0,0,selected.size());
                auto expected=source.index_select(0,tb::tensor(selected).to(tb::kCUDA)).cpu().contiguous();
                auto host=actual.cpu().contiguous();
                check(host.nbytes()==expected.nbytes() && !std::memcmp(host.data_ptr(),expected.data_ptr(),host.nbytes()),"fixed KV gather changed RAM/SSD/GPU rows");
            }
        }
        double reads=0,hits=0;for(const auto& [name,value]:store->metrics()) {
            if(name=="qsa_kv_ssd_reads")reads=value;if(name=="qsa_kv_gpu_hits")hits=value;
        }
        check(reads>0 && hits>0,"streamed gather test missed cold or hot rows");
    }
}

void check_gelu_approximations() {
    namespace tb=mfq_tensor_backend;
    std::vector<float> values(289);
    for(std::size_t i=0;i<values.size();++i)values[i]=float(i-144.0)/32;
    for(auto dtype:{tb::kFloat32,tb::kFloat16})for(bool strided:{false,true}) {
        auto input=tb::tensor(values,tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32))
            .reshape({17,17}).to(tb::kCUDA).to(dtype);
        if(strided)input=input.transpose(0,1);
        auto source=input.contiguous().to(tb::kFloat32).to(tb::kCPU);
        for(const std::string mode:{"none","tanh"}) {
            auto output=tb::gelu(input,mode).to(tb::kFloat32).to(tb::kCPU);
            for(int i=0;i<289;++i) {
                const double x=source.data_ptr<float>()[i];
                const auto reference=mode=="tanh"?
                    .5*x*(1+std::tanh(.7978845608028654*(x+.044715*x*x*x))):
                    .5*x*(1+std::erf(x/std::sqrt(2.0)));
                const auto tolerance=dtype==tb::kFloat32?2e-6:5e-4;
                check(std::abs(output.data_ptr<float>()[i]-reference)<=tolerance*std::max(1.0,std::abs(x)),
                    "GELU approximation differs from its scalar reference");
            }
        }
    }
}

void check_dma_copy_buckets() {
    using namespace mfq::cuda;
    namespace tb=mfq_tensor_backend;
    constexpr int rows=129,width=32;
    auto host=tb::empty({rows,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
    auto device=tb::empty({rows,width},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
    auto* source=host.data_ptr<uint8_t>();
    for(int i=0;i<rows*width;++i)source[i]=uint8_t(i%251);
    DmaCopyBatch batch(mfq_current_cuda_stream());
    batch.prepare(128,device.data_ptr(),source);
    for(int count:{1,17,65,128,3,73,2,0,129}) {
        device.fill_(255);
        std::vector<DmaCopy> copies;
        for(int i=0;i<count;++i)copies.push_back({device.data_ptr<uint8_t>()+i*width,source+i*width,width});
        check(batch.submit(copies)==(count>0 && count<=128),"DMA capacity dispatch mismatch");
        auto actual=device.cpu().contiguous();
        for(int i=0;i<rows*width;++i)check(actual.data_ptr<uint8_t>()[i]==(i<count*width?source[i]:255),
            "DMA capacity reused an inactive copy or lost an active copy");
    }
}

#include "kv_cache_bench.h"
#include "mapped_copy_bench.h"
#include "mfe_prepare_bench.h"
#include "mfe_transfer_bench.h"
#include "sampling_bench.h"
#include "batched_sampling_test.h"
#include "transfer_copy_bench.h"
#include "kv_speculative_lru_test.h"
int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    if(argc==2 && std::string(argv[1])=="--workspace-loan-check") {
        check_workspace_loan();check_graph_workspace_reuse();
        std::cout<<"CUDA workspace loan ranges and graph reuse checks passed\n";return 0;
    }
    if(argc==2 && std::string(argv[1])=="--kv-expert-loan-check") {
        const auto context=default_context();check_kv_expert_loan();check_kv_projection_loan();check_reused_projection_observer();return 0;
    }
    if(argc==2 && std::string(argv[1])=="--kv-speculative-lru-check") {
        const auto context=default_context();check_kv_speculative_lru();return 0;
    }
    if(argc==2 && std::string(argv[1])=="--kv-speculative-lru-bench") {
        const auto context=default_context();benchmark_kv_speculative_lru();return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--batched-sampling-bench") {
        benchmark_batched_sampling(argv[2]);
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--batched-sampling-cutoff-check") {
        check_batched_sampling_cutoffs();
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--batched-sampling-check") {
        check_batched_sampling();
        check_batched_sampling_cutoffs();
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--batched-sampling-grammar-check") {
        check_batched_sampling(argv[2]);
        return 0;
    }
    if(argc==2 && std::string(argv[1])=="--sampling-bench") {benchmark_sampling_keys();return 0;}
    if(argc==2 && std::string(argv[1])=="--conditional-compact-sampling-check") {check_conditional_compact_sampling();return 0;}
    if(argc==2 && std::string(argv[1])=="--sampling-fallback-check") {check_sampling_fallback();return 0;}
    if(argc==2 && std::string(argv[1])=="--sampling-check") {benchmark_sampling_keys(false);return 0;}
    if(argc==2 && std::string(argv[1])=="--transfer-copy-bench") {benchmark_transfer_copy();return 0;}
    if(argc==2 && std::string(argv[1])=="--transfer-copy-check") {benchmark_transfer_copy(false);return 0;}
    if(argc==2 && std::string(argv[1])=="--kv-cache-bench") {benchmark_decode_kv_cache();return 0;}
    if(argc==2 && std::string(argv[1])=="--route-export-bench") {benchmark_route_export();return 0;}
    if(argc==2 && std::string(argv[1])=="--mapped-copy-bench") {benchmark_mapped_copy();return 0;}
    if((argc==2 || argc==3) && std::string(argv[1])=="--mfe-prepare-bench") {
        benchmark_mfe_prepare();
        if(argc==3)mfe_expert_plan_test::verify_ffn(argv[2],true);
        return 0;
    }
    if(argc==2 && std::string(argv[1])=="--mfe-prepare-check") {benchmark_mfe_prepare(false);return 0;}
    if(argc==2 && std::string(argv[1])=="--mapped-copy-check") {benchmark_mapped_copy(false);return 0;}
    if(argc==2 && std::string(argv[1])=="--mapped-copy-pcie-bench") {benchmark_mapped_copy(true,true);return 0;}
    if(argc==2 && std::string(argv[1])=="--mfe-transfer-bench") {benchmark_mfe_transfer();return 0;}
    if(argc==2 && std::string(argv[1])=="--mfe-transfer-check") {benchmark_mfe_transfer(false);return 0;}
    if(argc==2 && std::string(argv[1])=="--kv-append-check") {
        check_exclusive_kv();check_decode_kv_append();
        std::cout<<"Exclusive short KV/index append and graph replay checks passed\n";return 0;
    }
    check_dma_copy_buckets();
    check_workspace_loan();
    check_kv_memory();
    check_exclusive_kv();
    check_decode_kv_append();
    check_kv_expert_loan();
    check_kv_projection_loan();
    check_reused_projection_observer();
    check_graph_workspace_reuse();
    check_gelu_approximations();
    check_streamed_graph_segments();
    check_retained_recurrent_storage();
    check_sampling_storage();
    check_linear_execution();
    check_graph_warmup_state();
    check_dense_loading();
    check_cached_moe_binding();
    check_gemma_composition();
    check_tts_sampling();
    check_ffn_branches();
    check_mfe_cached_dispatch_geometry();
    {
        CudaExecutionContext execution;
        check_snapshots(execution, false);
        check_snapshots(execution, true);
    }
    if (argc == 3) check_batching(argv[1], argv[2]);
    else check(argc == 1, "usage: mfq-cuda-runtime-state-test [model tokenizer]");
    std::cout << "CUDA runtime state checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
