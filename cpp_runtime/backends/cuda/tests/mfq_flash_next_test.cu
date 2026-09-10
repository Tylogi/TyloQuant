// Native ABI numerical-test bridge: tests/test_cuda_flash_next.py supplies the
// same NumPy oracle cases to this executable and the production Torch module.
#include "mfq/kernels/cuda/flash_next.h"
#include "flash_next/state.h"
#include "flash_next/qwen4.h"
#include "mfq_cuda_context.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

mfq_tensor_backend::Tensor attention_glm_mla_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor meta,
    double scale);
mfq_tensor_backend::Tensor attention_dsv4_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor mask,
    mfq_tensor_backend::Tensor sinks, mfq_tensor_backend::Tensor meta,
    double scale);

namespace {
using namespace mfq::cuda;
using Json = nlohmann::json;

void require_constant(
    const Tensor& value, float expected, float tolerance, const char * name) {
    const auto host = value.to(kFloat32).contiguous().cpu();
    float maximum_error = 0.0f;
    for (int64_t index = 0; index < host.numel(); ++index) {
        maximum_error = std::max(
            maximum_error,
            std::abs(host.data_ptr<float>()[index] - expected));
    }
    if (maximum_error > tolerance) {
        throw std::runtime_error(
            std::string(name) + " mismatch, max_abs=" +
            std::to_string(maximum_error));
    }
}

void require_qsa_gqa_means(
    const Tensor& value, float first, float second, float tolerance) {
    const auto host = value.to(kFloat32).contiguous().cpu();
    float maximum_error = 0.0f;
    for (int head = 0; head < 24; ++head) {
        const float expected = head < 12 ? first : second;
        for (int width = 0; width < 256; ++width) {
            maximum_error = std::max(
                maximum_error,
                std::abs(host.data_ptr<float>()[head * 256 + width] - expected));
        }
    }
    if (maximum_error > tolerance) {
        throw std::runtime_error(
            "QSA sparse GQA mismatch, max_abs=" +
            std::to_string(maximum_error));
    }
}

void check_shared_sparse_attention() {
    const auto float_cuda = TensorOptions{}.dtype(kFloat32).device(kCUDA);
    const auto half_cuda = TensorOptions{}.dtype(kFloat16).device(kCUDA);
    const auto int_cuda = TensorOptions{}.dtype(kInt32).device(kCUDA);
    constexpr int selected_count = 2048;
    std::vector<int32_t> selected(selected_count);
    float glm_expected = 0.0f;
    for (int index = 0; index < selected_count; ++index) {
        selected[index] = (index * 37 + 11) % 256;
        glm_expected += static_cast<float>(selected[index]);
    }
    glm_expected /= selected.size();
    const auto indices = tensor(selected)
        .reshape({1, 1, selected_count}).to(int_cuda);
    auto meta = empty({4 << 20}, float_cuda);

    const auto qsa_q = zeros({1, 24, 1, 256}, float_cuda);
    const auto qsa_k = zeros({1, 2, 256, 256}, half_cuda);
    const auto qsa_v = arange(512, float_cuda)
        .reshape({1, 2, 256, 1}).expand({1, 2, 256, 256})
        .contiguous().to(kFloat16);
    require_qsa_gqa_means(
        mfq_flash_next::qwen4_sparse_gqa_attention(
            qsa_q, qsa_k, qsa_v, indices),
        glm_expected, glm_expected + 256.0f, 6.0e-2f);

    const auto glm_q = zeros({1, 64, 1, 576}, float_cuda);
    const auto glm_kv = arange(256, float_cuda)
        .reshape({1, 256, 1}).expand({1, 256, 576}).contiguous().to(kFloat16);
    require_constant(
        attention_glm_mla_sparse_cuda(glm_q, glm_kv, indices, meta, 1.0),
        glm_expected, 2.0e-2f, "GLM sparse MLA");

    const auto dsv_q = zeros({1, 64, 1, 512}, float_cuda);
    const auto dsv_kv = arange(256, float_cuda)
        .reshape({1, 256, 1}).expand({1, 256, 512}).contiguous().to(kFloat16);
    std::vector<float> mask_values(selected_count, 0.0f);
    float dsv_sum = 0.0f;
    int dsv_count = 0;
    for (int index = 0; index < selected_count; ++index) {
        if (index % 5 == 0) {
            mask_values[index] = -std::numeric_limits<float>::infinity();
        } else {
            dsv_sum += static_cast<float>(selected[index]);
            ++dsv_count;
        }
    }
    const auto mask = tensor(mask_values)
        .reshape({1, 1, selected_count}).to(half_cuda);
    const auto sinks = zeros({64}, float_cuda);
    require_constant(
        attention_dsv4_sparse_cuda(
            dsv_q, dsv_kv, indices, mask, sinks, meta, 1.0),
        dsv_sum / static_cast<float>(dsv_count + 1),
        6.0e-2f, "DSV4 sparse attention");

    const auto all_invalid_mask = full(
        {1, 1, selected_count},
        -std::numeric_limits<float>::infinity(), half_cuda);
    const auto all_invalid_sinks = full(
        {64}, -std::numeric_limits<float>::infinity(), float_cuda);
    require_constant(
        attention_dsv4_sparse_cuda(
            dsv_q, dsv_kv, indices, all_invalid_mask,
            all_invalid_sinks, meta, 1.0),
        0.0f, 0.0f, "fully masked DSV4 sparse attention");
}

Tensor input(const Json& j) {
    if (j.is_null()) return {};
    const auto shape = j.at("shape").get<std::vector<int64_t>>();
    const auto dtype = j.value("dtype", "float32");
    auto host = empty(shape, TensorOptions{}.dtype(dtype == "int64" ? kInt64 : kFloat32));
    if (dtype == "int64") {
        const auto data = j.at("data").get<std::vector<int64_t>>();
        if (data.size() != static_cast<size_t>(host.numel())) throw std::runtime_error("fixture size mismatch");
        std::copy(data.begin(), data.end(), host.data_ptr<int64_t>());
    } else {
        const auto data = j.at("data").get<std::vector<float>>();
        if (data.size() != static_cast<size_t>(host.numel())) throw std::runtime_error("fixture size mismatch");
        std::copy(data.begin(), data.end(), host.data_ptr<float>());
    }
    auto value = host.to(Device{DeviceType::cuda, 0});
    if (dtype == "float16") value = value.to(kFloat16);
    else if (dtype == "bfloat16") value = value.to(kBFloat16);
    else if (dtype != "float32" && dtype != "int64") throw std::runtime_error("unknown fixture dtype");
    if (j.value("noncontiguous", false) && value.dim() >= 2)
        value = value.transpose(-1, -2).contiguous().transpose(-1, -2);
    return value;
}

std::vector<Tensor> run(const std::string& op, const std::vector<Tensor>& a, const Json& p) {
    using namespace mfq_flash_next;
    const auto optional = [&](size_t i) -> std::optional<Tensor> {
        return a.at(i).defined() ? std::optional<Tensor>(a.at(i)) : std::nullopt;
    };
    std::optional<double> scale;
    if (p.contains("scale") && !p.at("scale").is_null()) scale = p.at("scale").get<double>();
    if (op == "runtime_gdn") {
        const auto linear = [&](int index) -> mfq::flash_next::Linear {
            auto w=a.at(index);return [w](const Tensor& x) {return matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
        };
        mfq::flash_next::GdnWeights w{linear(1),linear(2),linear(3),linear(4),linear(5),a.at(6),a.at(7),a.at(8),a.at(9)};
        mfq::flash_next::Gdn block(std::move(w),p.at("key_heads"),p.at("value_heads"),p.at("width"),
            p.at("kernel"),p.value("eps",1e-6),p.value("silu_gate",false));
        std::vector<Tensor> out;
        for (const auto& step:p.at("steps")) {
            if (step.value("reset",false)) block.reset();
            else if (step.value("commit",false)) block.commit();
            else if (step.value("rollback",false)) block.rollback();
            else {
                out.push_back(block.forward(a.at(0).narrow(1,step.at("begin"),step.at("count")),step.value("cache",true),step.value("confirmed",0)));
                out.push_back(block.conv_state());out.push_back(block.recurrent_state());
            }
        }
        return out;
    }
    if (op == "runtime_ngram" || op == "runtime_ple") {
        const auto weights=a.at(op=="runtime_ngram"?1:2);
        std::vector<mfq::flash_next::Linear> shards;
        for (int64_t i=0;i<weights.size(0);++i) {
            auto w=weights.select(0,i);
            shards.push_back([w](const Tensor& ids) {
                auto shape=ids.sizes().vec();shape.push_back(w.size(1));
                return w.index_select(0,ids.reshape({-1}).to(kInt64)).reshape(shape);
            });
        }
        mfq::flash_next::NgramEmbedding embedding(std::move(shards),weights.size(1),weights.size(2),p.at("ngram"),
            p.at("heads_per_ngram"),p.at("eos"),p.at("multipliers").get<std::vector<int64_t>>(),
            p.at("offsets").get<std::vector<int64_t>>(),p.at("vocab").get<std::vector<int64_t>>());
        std::vector<Tensor> out;
        if (op=="runtime_ngram") {
            for (const auto& step:p.at("steps")) {
                if (step.value("reset",false)) embedding.reset();
                else {
                    Tensor ids;
                    out.push_back(embedding.forward(a.at(0).narrow(1,step.at("begin"),step.at("count")),step.value("cache",true),&ids));
                    out.push_back(ids);
                }
            }
        } else {
            const auto linear=[&](int i)->mfq::flash_next::Linear {
                auto w=a.at(i);return [w](const Tensor& x) {return matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
            };
            mfq::flash_next::PleWeights w{linear(3),linear(4),a.at(5),a.at(6),a.at(7),a.at(8)};
            mfq::flash_next::Ple block(std::move(embedding),std::move(w),p.at("hidden"),p.at("streams"),p.at("ngram"),p.value("eps",1e-6));
            for (const auto& step:p.at("steps")) {
                if (step.value("reset",false)) block.reset();
                else if (step.value("commit",false)) block.commit();
                else if (step.value("rollback",false)) block.rollback();
                else {
                    const int64_t begin=step.at("begin"),count=step.at("count");
                    out.push_back(block.forward(a.at(0).narrow(1,begin,count),a.at(1).narrow(1,begin,count),step.value("cache",true),step.value("confirmed",0)));
                    out.push_back(block.conv_state());
                }
            }
        }
        return out;
    }
    if (op == "runtime_rotary") {
        mfq::flash_next::Rotary rotary(p.at("rotary"),p.at("maximum"),p.value("base",1e7),
            p.value("sections",std::vector<int64_t>{}),p.value("interleaved",false));
        return {rotary.forward(a.at(0),a.at(1))};
    }
    if (op == "runtime_qsa") {
        const auto linear = [&](int index) -> mfq::flash_next::Linear {
            auto weight=a.at(index);
            return [weight](const Tensor& x) {return matmul(x.to(weight.scalar_type()),weight.transpose(-1,-2));};
        };
        auto rotary=std::make_shared<mfq::flash_next::Rotary>(p.at("rotary"),p.at("maximum"),p.value("base",1e7),
            p.value("sections",std::vector<int64_t>{}),p.value("interleaved",false));
        mfq::flash_next::QsaWeights weights{linear(1),linear(2),linear(3),linear(4),linear(5),a.at(6),a.at(7),a.at(8),a.at(9)};
        mfq::flash_next::QsaConfig config{p.at("heads"),p.at("kv_heads"),p.at("width"),p.at("index_heads"),
            p.at("index_width"),p.at("pool"),p.at("budget"),p.at("maximum"),p.value("eps",1e-6)};
        mfq::flash_next::Qsa block(std::move(weights),config,rotary);
        Tensor history;
        std::vector<Tensor> out;
        for (const auto& step:p.at("steps")) {
            if (step.value("reset",false)) {block.reset();history={};}
            else if (step.contains("truncate")) {
                block.truncate(step.at("truncate"));
                history=history.narrow(-1,0,step.at("truncate"));
            } else {
                auto pos=a.at(10).narrow(-1,step.at("begin"),step.at("count"));
                const bool cache=step.value("cache",true);
                auto full=cache && history.defined()?cat({history,pos},-1):pos;
                std::vector<Tensor> trace;
                out.push_back(block.forward(a.at(0).narrow(1,step.at("begin"),step.at("count")),pos,full,cache,
                    p.value("trace",false)?&trace:nullptr));
                out.insert(out.end(),trace.begin(),trace.end());
                if (cache) history=full;
            }
        }
        return out;
    }
    if (op == "runtime_select_pooled_blocks") return {mfq::flash_next::select_pooled_blocks(
        a.at(0), p.at("query_offset"), p.at("logical_length"), p.at("pool"), p.at("budget"), p.at("tail"))};
    if (op == "runtime_sequence_cache") {
        mfq::flash_next::SequenceCache cache(p.at("maximum"), a.at(0).size(-1));
        std::vector<Tensor> out;
        for (const auto& step : p.at("steps")) {
            if (step.contains("truncate")) cache.truncate(step.at("truncate"));
            else if (step.value("reset", false)) cache.reset();
            else out.push_back(cache.append(a.at(0).narrow(1, step.at("begin"), step.at("count"))).clone());
        }
        return out;
    }
    if (op == "runtime_kda") {
        const auto linear = [&](int index) -> mfq::flash_next::Linear {
            auto weight = a.at(index);
            return [weight](const Tensor& x) { return matmul(x.to(weight.scalar_type()), weight.transpose(-1,-2)); };
        };
        mfq::flash_next::KdaWeights weights{linear(1),linear(2),linear(3),linear(4),linear(5),linear(6),linear(7),
            a.at(8),a.at(9),a.at(10),a.at(11),a.at(12),a.at(13)};
        mfq::flash_next::Kda block(std::move(weights), p.at("heads"), p.at("width"), p.at("kernel"),
            p.value("lower_bound", -5.0), p.value("eps", 1e-5));
        std::vector<Tensor> out;
        for (const auto& step : p.at("steps")) {
            if (step.value("rollback", false)) block.rollback();
            else if (step.value("commit", false)) block.commit();
            else if (step.value("reset", false)) block.reset();
            else {
                out.push_back(block.forward(a.at(0).narrow(1, step.at("begin"), step.at("count")),
                    step.value("cache", true), step.value("confirmed", 0)));
                out.push_back(block.conv_state());
                out.push_back(block.recurrent_state());
            }
        }
        return out;
    }
    if (op == "qwen4_grouped_rms_norm") return {qwen4_grouped_rms_norm(a.at(0), a.at(1), p.at("group_size"), p.value("eps", 1e-6))};
    if (op == "qwen4_gated_residual_pre") return qwen4_gated_residual_pre(a.at(0), a.at(1), a.at(2), a.at(3), optional(4), p.at("hidden_size"), p.value("hc_count", 4), p.value("eps", 1e-6));
    if (op == "qwen4_gated_residual_post") return {qwen4_gated_residual_post(a.at(0), a.at(1), a.at(2), p.value("hc_count", 4))};
    if (op == "glm5_mhc_pre") return glm5_mhc_pre(a.at(0), a.at(1), a.at(2), a.at(3), p.value("sinkhorn_iterations", 20), p.value("hc_eps", 1e-6), p.value("rms_eps", 1e-5));
    if (op == "glm5_mhc_post") return {glm5_mhc_post(a.at(0), a.at(1), a.at(2), a.at(3))};
    if (op == "glm5_kda_forget_gate") return {glm5_kda_forget_gate(a.at(0), a.at(1), a.at(2), a.at(3), a.at(4), p.at("num_heads"), p.at("head_dim"), p.value("lower_bound", -5.0))};
    if (op == "qsa_block_scores") return {qsa_block_scores(a.at(0), a.at(1))};
    if (op == "glm5_kpool_scores") return {glm5_kpool_scores(a.at(0), a.at(1), a.at(2))};
    if (op == "glm5_kpool_states") return {glm5_kpool_states(a.at(0), a.at(1), a.at(2), p.value("pool_size", 4))};
    if (op == "qwen4_ple_dilated_conv_silu") return qwen4_ple_dilated_conv_silu(a.at(0), a.at(1), optional(2), p.at("dilation"));
    if (op == "qwen4_dense_gqa_attention") return {qwen4_dense_gqa_attention(a.at(0), a.at(1), a.at(2), p.at("query_offset"))};
    if (op == "qwen4_sparse_gqa_attention") return {qwen4_sparse_gqa_attention(a.at(0), a.at(1), a.at(2), a.at(3))};
    if (op == "glm5_dense_mla_attention") return {glm5_dense_mla_attention(a.at(0), a.at(1), p.at("query_offset"), scale)};
    if (op == "glm5_sparse_mla_attention") return {glm5_sparse_mla_attention(a.at(0), a.at(1), a.at(2), scale)};
    throw std::runtime_error("unknown Flash-Next operation");
}

Json output(const std::vector<Tensor>& tensors) {
    Json result = Json::array();
    for (const auto& value : tensors) {
        if (!value.defined()) { result.push_back(nullptr); continue; }
        auto host = value.to(kFloat32).contiguous().cpu();
        std::vector<float> data(host.numel());
        if (host.numel()) std::copy_n(host.data_ptr<float>(), host.numel(), data.data());
        result.push_back({{"shape", value.sizes().vec()}, {"data", data},
            {"dtype", value.scalar_type() == kFloat16 ? "float16" :
                value.scalar_type() == kBFloat16 ? "bfloat16" : "float32"}});
    }
    return result;
}
} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        const auto context = mfq::cuda::default_context(0);
        auto stream = mfq::cuda::stream_from_pool(false, 0);
        mfq::cuda::StreamGuard stream_guard(stream);
        if (argc == 1) {
            auto q = mfq::cuda::ones({1,1,2,4}, mfq::cuda::TensorOptions{}.device(mfq::cuda::kCUDA));
            auto k = mfq::cuda::ones({1,3,4}, q.options());
            auto got = mfq_flash_next::qsa_block_scores(q,k).cpu();
            for (int i = 0; i < 3; ++i)
                if (got.data_ptr<float>()[i] != 4.f) throw std::runtime_error("QSA smoke mismatch");
            check_shared_sparse_attention();
            std::cout << "Flash-Next and shared sparse-attention native smoke passed\n";
            return 0;
        }
        if (std::string(argv[1]) != "--json") throw std::runtime_error("expected --json");
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto request = Json::parse(line);
                std::vector<Tensor> tensors;
                for (const auto& item : request.at("inputs")) tensors.push_back(input(item));
                const auto op = request.at("op").get<std::string>();
                const auto params = request.value("params", Json::object());
                const auto execute = [&] { return run(op, tensors, params); };
                std::vector<Tensor> result;
                mfq::cuda::Graph graph;
                if (request.value("graph", false)) {
                    result = execute();
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq::cuda::current_stream(0).stream()));
                    graph.prepare_memory();
                    result = execute();
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq::cuda::current_stream(0).stream()));
                    result.clear();
                    graph.capture_begin();
                    result = execute();
                    graph.capture_end();
                    graph.replay();
                    graph.replay();
                    if (request.contains("replay_inputs")) {
                        const auto& updates = request.at("replay_inputs");
                        if (updates.size() != tensors.size()) throw std::runtime_error("replay fixture count mismatch");
                        for (size_t i = 0; i < tensors.size(); ++i) {
                            if (tensors[i].defined()) tensors[i].copy_(input(updates.at(i)));
                        }
                        graph.replay();
                    }
                } else result = execute();
                std::cout << Json({{"outputs", output(result)}}).dump() << std::endl;
            } catch (const std::exception& e) {
                std::cout << Json({{"error", e.what()}}).dump() << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Flash-Next native test failed: " << e.what() << '\n';
        return 1;
    }
}
