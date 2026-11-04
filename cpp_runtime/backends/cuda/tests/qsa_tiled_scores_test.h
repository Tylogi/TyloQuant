#pragma once

#include "qsa_scores.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// Modes: original small GEMMs, legacy large GEMMs, tiled default.
class QsaScoreSetting {
    const char* names_[2] = {"MFQ_QSA_PREFILL_TILED_SCORES", "MFQ_QSA_PREFILL_SCORE_GEMM"};
    std::string old_[2];
    static void set(const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        if (*value) setenv(name, value, 1);
        else unsetenv(name);
#endif
    }
public:
    explicit QsaScoreSetting(int mode) {
        for (int i = 0; i < 2; ++i)
            if (const auto* value = std::getenv(names_[i])) old_[i] = value;
        set(names_[0], mode == 2 ? "1" : "0");
        set(names_[1], mode == 1 ? "1" : "0");
    }
    ~QsaScoreSetting() {
        for (int i = 0; i < 2; ++i) set(names_[i], old_[i].c_str());
    }
};

Tensor qsa_score_data(std::vector<int64_t> shape, int salt, tb::ScalarType dtype) {
    int64_t count = 1;
    for (auto dim : shape) count *= dim;
    std::vector<float> values(count);
    for (int64_t i = 0; i < count; ++i) {
        uint32_t hash = uint32_t(i) + uint32_t(salt) * 7919u;
        hash ^= hash >> 16;
        hash *= 0x7feb352du;
        hash ^= hash >> 15;
        hash *= 0x846ca68bu;
        hash ^= hash >> 16;
        values[i] = (float(hash & 65535u) / 32768.f - 1.f) * .13f;
    }
    return tb::tensor(values).reshape(shape).to(dtype).to(tb::kCUDA);
}

void qsa_score_exact_indices(const Tensor& actual, const Tensor& expected, const char* label,
    bool unordered = false) {
    auto a = actual.to(tb::kInt64).contiguous().cpu();
    auto b = expected.to(tb::kInt64).contiguous().cpu();
    if (a.sizes() != b.sizes()) throw std::runtime_error(std::string(label) + " shape mismatch");
    if (unordered) {
        const auto count = a.size(-1);
        for (int64_t row = 0; row < a.numel() / count; ++row) {
            std::sort(a.data_ptr<int64_t>() + row * count, a.data_ptr<int64_t>() + (row + 1) * count);
            std::sort(b.data_ptr<int64_t>() + row * count, b.data_ptr<int64_t>() + (row + 1) * count);
        }
    }
    for (int64_t i = 0; i < a.numel(); ++i) {
        if (a.data_ptr<int64_t>()[i] != b.data_ptr<int64_t>()[i])
            throw std::runtime_error(std::string(label) + " differs at " + std::to_string(i) +
                " actual=" + std::to_string(a.data_ptr<int64_t>()[i]) +
                " expected=" + std::to_string(b.data_ptr<int64_t>()[i]));
    }
}

// FP32 dot products summed in different orders differ by up to about
// width * 2^-24 of the summed product magnitudes on each side. The error
// follows that magnitude, not the score, so cancelled scores legitimately
// differ by much more than their own relative tolerance.
Tensor qsa_score_tolerance(const Tensor& query, const Tensor& keys) {
    Tensor magnitude;
    {
        QsaScoreSetting setting(0);
        magnitude = qsa_block_scores(query.to(tb::kFloat32).abs(), keys.to(tb::kFloat32).abs());
    }
    return magnitude * (2.0 * double(query.size(-1)) / 16777216.0) + 1e-7;
}

float qsa_element_tolerance(const Tensor& tolerance, int64_t index, float value) {
    return tolerance.defined() ? tolerance.data_ptr<float>()[index] : 1e-7f + 1e-5f * std::abs(value);
}

void qsa_equal_within(const Tensor& actual, const Tensor& expected, const Tensor& tolerance, const char* label) {
    auto a = actual.to(tb::kFloat32).contiguous().cpu();
    auto b = expected.to(tb::kFloat32).contiguous().cpu();
    auto t = tolerance.to(tb::kFloat32).contiguous().cpu();
    if (a.sizes() != b.sizes() || a.sizes() != t.sizes()) throw std::runtime_error(std::string(label) + " shape mismatch");
    for (int64_t i = 0; i < a.numel(); ++i) {
        const float x = a.data_ptr<float>()[i], y = b.data_ptr<float>()[i], bound = t.data_ptr<float>()[i];
        if (!(std::abs(x - y) <= bound))
            throw std::runtime_error(std::string(label) + " differs at " + std::to_string(i) + " actual=" +
                std::to_string(x) + " expected=" + std::to_string(y) + " tolerance=" + std::to_string(bound));
    }
}

// A top-k may swap candidates whose reference scores lie within tolerance of
// the cutoff. Anything clearly above the cutoff must be selected and nothing
// clearly below it.
void qsa_score_selection_matches(const Tensor& scores, const Tensor& reference, int selected, const char* label,
    const Tensor& tolerance = {}) {
    auto chosen = std::get<1>(tb::topk(scores, selected, -1, true, true)).to(tb::kInt64).contiguous().cpu();
    auto expected = reference.to(tb::kFloat32).contiguous().cpu();
    auto bounds = tolerance.defined() ? tolerance.to(tb::kFloat32).contiguous().cpu() : Tensor{};
    const auto count = expected.size(-1), rows = expected.numel() / count;
    for (int64_t row = 0; row < rows; ++row) {
        const float* values = expected.data_ptr<float>() + row * count;
        std::vector<int64_t> order(count);
        for (int64_t index = 0; index < count; ++index) order[index] = index;
        std::nth_element(order.begin(), order.begin() + (selected - 1), order.end(),
            [&](int64_t x, int64_t y) { return values[x] > values[y]; });
        const auto cut = order[selected - 1];
        const float cutoff = values[cut];
        const float cutoff_bound = qsa_element_tolerance(bounds, row * count + cut, cutoff);
        std::vector<bool> picked(count, false);
        for (int64_t slot = 0; slot < selected; ++slot) {
            const auto index = chosen.data_ptr<int64_t>()[row * selected + slot];
            picked[index] = true;
            if (values[index] < cutoff - cutoff_bound - qsa_element_tolerance(bounds, row * count + index, values[index]))
                throw std::runtime_error(std::string(label) + " selected a score below the cutoff at row " +
                    std::to_string(row) + " index=" + std::to_string(index));
        }
        for (int64_t index = 0; index < count; ++index)
            if (!picked[index] &&
                values[index] > cutoff + cutoff_bound + qsa_element_tolerance(bounds, row * count + index, values[index]))
                throw std::runtime_error(std::string(label) + " missed a score above the cutoff at row " +
                    std::to_string(row) + " index=" + std::to_string(index));
    }
}

// Rows whose top-k block cutoff is closer than the score tolerance may select
// either candidate. Exact token and output comparisons use the other rows.
std::vector<bool> qsa_separated_rows(const Tensor& scores, const Tensor& tolerance, int64_t selected, int64_t pool) {
    auto s = scores.to(tb::kFloat32).contiguous().cpu();
    auto t = tolerance.to(tb::kFloat32).contiguous().cpu();
    const auto rows = s.size(1), pools = s.size(2);
    std::vector<bool> separated(s.size(0) * rows, true);
    for (int64_t index = 0; index < s.size(0) * rows; ++index) {
        const auto row = index % rows;
        std::vector<std::pair<float, float>> visible;
        for (int64_t block = 0; block < pools; ++block)
            if (block * pool + pool - 1 <= row)
                visible.push_back({s.data_ptr<float>()[index * pools + block], t.data_ptr<float>()[index * pools + block]});
        if (int64_t(visible.size()) <= selected) continue;
        std::partial_sort(visible.begin(), visible.begin() + selected + 1, visible.end(),
            [](const auto& x, const auto& y) { return x.first > y.first; });
        separated[index] = visible[selected - 1].first - visible[selected].first >
            visible[selected - 1].second + visible[selected].second;
    }
    return separated;
}

void qsa_check_score_selection(const Tensor& actual, const Tensor& expected, const Tensor& tolerance = {}) {
    if (actual.scalar_type() != tb::kFloat32) throw std::runtime_error("QSA scores must remain FP32");
    if (tolerance.defined()) qsa_equal_within(actual, expected, tolerance, "QSA tiled FP32 scores");
    else equal(actual, expected, "QSA tiled FP32 scores", 1e-7f, 1e-5f);
    const auto rows = actual.size(1), pools = actual.size(2);
    if (!pools) return;
    const auto options = actual.options().dtype(tb::kInt64);
    auto positions = tb::arange(rows, options).reshape({1, rows, 1});
    auto ends = (tb::arange(pools, options) * 4 + 3).reshape({1, 1, pools});
    auto visible = ends <= positions;
    auto masked = tb::where(visible, actual, tb::full_like(actual, -1e30));
    auto oracle = tb::where(visible, expected, tb::full_like(expected, -1e30));
    if (tolerance.defined()) qsa_equal_within(masked, oracle, tolerance, "QSA tiled causal visibility");
    else equal(masked, oracle, "QSA tiled causal visibility", 1e-7f, 1e-5f);
    for (int count : {1, 17, 64}) {
        const int selected = std::min<int64_t>(count, pools);
        // QSA requests sorted=false, so only the selected set matters.
        qsa_score_selection_matches(masked, oracle, selected, "QSA causal top-k", tolerance);
        qsa_score_selection_matches(actual, expected, selected, "QSA unmasked top-k", tolerance);
    }
}

void qsa_score_scalar_oracle(const Tensor& query, const Tensor& keys, const Tensor& scores) {
    auto q = query.to(tb::kFloat32).contiguous().cpu();
    auto k = keys.to(tb::kFloat32).contiguous().cpu();
    auto s = scores.contiguous().cpu();
    const auto rows = q.size(1), heads = q.size(2), width = q.size(3), pools = k.size(1);
    if (!pools) return;
    for (int64_t batch = 0; batch < q.size(0); ++batch)
    for (auto row : {int64_t(0), rows / 2, rows - 1})
    for (auto pool : {int64_t(0), pools / 2, pools - 1}) {
        double score = 0;
        for (int64_t head = 0; head < heads; ++head) {
            double dot = 0;
            for (int64_t d = 0; d < width; ++d)
                dot += double(q.data_ptr<float>()[((batch * rows + row) * heads + head) * width + d]) *
                    k.data_ptr<float>()[(batch * pools + pool) * width + d];
            score += std::max(0., dot);
        }
        score /= std::sqrt(double(width));
        if (std::abs(s.data_ptr<float>()[(batch * rows + row) * pools + pool] - score) >
            1e-7 + 1e-5 * std::abs(score))
            throw std::runtime_error("QSA tiled scalar oracle differs");
    }
}

void qsa_score_selected_attention(const Tensor& actual, const Tensor& expected, tb::ScalarType dtype) {
    constexpr int pool = 4, budget = 64;
    const auto batch = actual.size(0), rows = actual.size(1), length = actual.size(2) * pool + 3;
    const auto offset = length - rows;
    auto indices = select_pooled_blocks(actual, offset, length, pool, budget, true);
    auto reference = select_pooled_blocks(expected, offset, length, pool, budget, true);
    qsa_score_exact_indices(indices, reference, "QSA dtype selected token indices", true);
    qsa_score_exact_indices(indices.narrow(-1, budget, pool - 1),
        reference.narrow(-1, budget, pool - 1), "QSA dtype partial pool tail");
    auto query = qsa_score_data({batch, 4, rows, 32}, 61, dtype);
    auto key = qsa_score_data({batch, 2, length, 32}, 67, dtype);
    auto value = qsa_score_data({batch, 2, length, 32}, 71, dtype);
    equal(mfq_qwen4_exp::sparse_gqa_attention(query, key, value, indices),
        mfq_qwen4_exp::sparse_gqa_attention(query, key, value, reference), "QSA dtype final attention");
    auto selected = indices.contiguous().cpu();
    const auto count = indices.size(2);
    for (int64_t row = 0; row < batch * rows; ++row) for (int64_t i = 0; i < count; ++i)
        if (selected.data_ptr<int32_t>()[row * count + i] > offset + row % rows)
            throw std::runtime_error("QSA dtype attention selected a future token");
}

void qsa_score_attention_case() {
    int cases = 0;
    for (int rows : {129, 1025}) for (int index_heads : {3, 8}) {
        CudaExecutionContext execution;
        constexpr int hidden = 32, heads = 4, kv = 2, width = 32, iw = 128, pool = 4, budget = 64;
        auto projection = [](int in, int out, int salt) -> Linear {
            auto weight = qsa_score_data({out, in}, salt, tb::kFloat16);
            return [weight](CudaExecutionContext&, const Tensor& x) {
                return tb::matmul(x.to(weight.scalar_type()), weight.transpose(-1, -2));
            };
        };
        auto norm = tb::zeros({width}, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
        auto inorm = tb::zeros({iw}, norm.options());
        QsaWeights weights{projection(hidden, heads * 2 * width, 1), projection(hidden, kv * width, 2),
            projection(hidden, kv * width, 3), projection(heads * width, hidden, 4),
            projection(hidden, (index_heads + 1) * iw, 5), norm, norm, inorm, inorm};
        QsaConfig cfg{heads, kv, width, index_heads, iw, pool, budget, rows + 8, 1e-6};
        auto rotary = std::make_shared<RotaryEmbedding>(8, rows + 8, 10000.);
        Qsa actual(weights, cfg, rotary), oracle(weights, cfg, rotary);
        auto input = qsa_score_data({1, rows, hidden}, 97, tb::kFloat16);
        auto positions = tb::arange(rows, input.options().dtype(tb::kInt64));
        std::vector<Tensor> old_trace, new_trace;
        Tensor expected, result;
        {
            QsaScoreSetting setting(0);
            expected = oracle.forward(execution, input, positions, positions, false, &old_trace);
        }
        {
            QsaScoreSetting setting(2);
            result = actual.forward(execution, input, positions, positions, false, &new_trace);
        }
        if (old_trace.size() != 4 || new_trace.size() != 4)
            throw std::runtime_error("QSA score test did not exercise sparse attention");
        equal(new_trace[2], old_trace[2], "QSA model index queries", 0, 0);
        equal(new_trace[3], old_trace[3], "QSA model pooled keys", 0, 0);
        std::cout << "qsa_model_score_check rows=" << rows << " heads=" << index_heads
            << " query_dtype=" << int(new_trace[2].scalar_type())
            << " key_dtype=" << int(new_trace[3].scalar_type()) << " max_abs="
            << (new_trace[0] - old_trace[0]).abs().max().item<float>() << std::endl;
        for (auto dtype : {tb::kFloat32, tb::kBFloat16}) {
            auto query = old_trace[2].to(dtype);
            auto pooled = old_trace[3].to(dtype);
            Tensor reference, tiled;
            {
                QsaScoreSetting setting(0);
                reference = qsa_block_scores(query, pooled);
            }
            {
                QsaScoreSetting setting(2);
                tiled = qsa_block_scores(query, pooled);
            }
            std::cout << "qsa_normalized_dtype_score_check dtype=" << int(dtype) << " max_abs="
                << (tiled - reference).abs().max().item<float>() << std::endl;
            qsa_check_score_selection(tiled, reference, qsa_score_tolerance(query, pooled));
        }
        const auto tolerance = qsa_score_tolerance(old_trace[2], old_trace[3]);
        qsa_check_score_selection(new_trace[0], old_trace[0], tolerance);
        const auto separated = qsa_separated_rows(old_trace[0], tolerance, budget / pool, pool);
        int ties = 0;
        for (int row = 0; row < rows; ++row) {
            if (!separated[row]) {
                ++ties;
                continue;
            }
            qsa_score_exact_indices(new_trace[1].narrow(1, row, 1), old_trace[1].narrow(1, row, 1),
                "QSA model selected token indices", true);
            qsa_score_exact_indices(new_trace[1].narrow(1, row, 1).narrow(-1, budget, pool - 1),
                old_trace[1].narrow(1, row, 1).narrow(-1, budget, pool - 1), "QSA model partial pool tail");
            equal(result.narrow(1, row, 1), expected.narrow(1, row, 1), "QSA tiled final attention");
        }
        std::cout << "qsa_model_selection_check rows=" << rows << " near_tie_rows=" << ties << std::endl;
        auto selected = new_trace[1].cpu().contiguous();
        const int count = int(selected.size(2));
        for (int row = 0; row < rows; ++row) for (int i = 0; i < count; ++i)
            if (selected.data_ptr<int32_t>()[row * count + i] > row)
                throw std::runtime_error("QSA selected a future token");
        ++cases;
    }
    std::cout << "qsa_tiled_attention_cases=" << cases << " exact selection/partial pool/final output PASS\n";
}

void qsa_prefill_scores_case() {
    struct Geometry { int batch, rows, pools, heads, width; };
    const Geometry geometries[] = {{1, 33, 65, 1, 32}, {2, 129, 257, 2, 64},
        {1, 65, 129, 3, 37}, {1, 1025, 513, 4, 128}, {1, 4097, 257, 8, 128},
        {2, 33, 63, 16, 256}, {1, 33, 129, 32, 7}, {1, 16385, 129, 4, 128},
        {1, 33, 64, 4, 128}, {1, 33, 2048, 4, 128}, {1, 1024, 64, 4, 128},
        {1, 1024, 2048, 4, 128}, {1, 2048, 64, 4, 128}, {1, 2048, 2048, 4, 128},
        {1, 4097, 2049, 4, 128}, {1, 16385, 4097, 4, 128},
        {1, 7, 65, 4, 128}, {1, 33, 65, 33, 32}, {1, 33, 65, 4, 257}, {1, 33, 0, 4, 32}};
    int cases = 0;
    for (auto dtype : {tb::kFloat32, tb::kBFloat16, tb::kFloat16}) for (const auto& g : geometries) {
        auto query = qsa_score_data({g.batch, g.heads, g.rows, g.width}, 17, dtype).permute({0, 2, 1, 3});
        auto keys = qsa_score_data({g.batch, g.pools, g.width + 8}, 31, dtype).narrow(-1, 4, g.width);
        Tensor expected, actual;
        {
            QsaScoreSetting setting(0);
            expected = qsa_block_scores(query, keys);
        }
        {
            QsaScoreSetting setting(2);
            actual = qsa_block_scores(query, keys);
        }
        std::cout << "qsa_tiled_check rows=" << g.rows << " pools=" << g.pools << " heads=" << g.heads
            << " width=" << g.width << " dtype=" << int(dtype) << std::endl;
        qsa_check_score_selection(actual, expected, qsa_score_tolerance(query, keys));
        qsa_score_scalar_oracle(query, keys, actual);
        if (g.rows == 65 && g.heads == 3) qsa_score_selected_attention(actual, expected, dtype);
        ++cases;
        if (g.rows == 33 && g.heads == 1) {
            QsaScoreSetting setting(1);
            qsa_check_score_selection(qsa_block_scores(query, keys), expected, qsa_score_tolerance(query, keys));
            ++cases;
        }
    }
    for (auto qtype : {tb::kFloat32, tb::kBFloat16, tb::kFloat16})
    for (auto ktype : {tb::kFloat32, tb::kBFloat16, tb::kFloat16}) {
        auto query = qsa_score_data({2, 65, 3, 128}, 13, qtype);
        auto keys = qsa_score_data({2, 129, 128}, 29, ktype);
        Tensor expected;
        {
            QsaScoreSetting setting(0);
            expected = mfq_qwen4_exp::block_scores(query, keys);
        }
        QsaScoreSetting setting(2);
        qsa_check_score_selection(mfq_qwen4_exp::block_scores(query, keys), expected,
            qsa_score_tolerance(query, keys));
        ++cases;
    }
    // Exact duplicates and sub-BF16 FP32 differences straddle the top-k cutoff.
    for (auto dtype : {tb::kFloat32, tb::kBFloat16, tb::kFloat16}) {
        std::vector<float> q(65 * 3 * 32), k(129 * 32);
        for (int row = 0; row < 65; ++row) for (int head = 0; head < 3; ++head)
            q[(row * 3 + head) * 32] = head == 1 ? -1.f : 1.f;
        for (int pool = 0; pool < 129; ++pool) k[pool * 32] = 1.f + float(pool / 2) * 0.0001f;
        auto query = tb::tensor(q).reshape({1, 65, 3, 32}).to(dtype).to(tb::kCUDA);
        auto keys = tb::tensor(k).reshape({1, 129, 32}).to(dtype).to(tb::kCUDA);
        Tensor expected;
        {
            QsaScoreSetting setting(0);
            expected = qsa_block_scores(query, keys);
        }
        QsaScoreSetting setting(2);
        auto actual = qsa_block_scores(query, keys);
        qsa_check_score_selection(actual, expected);
        qsa_score_exact_indices(std::get<1>(tb::topk(actual, 64, -1, true, true)),
            std::get<1>(tb::topk(expected, 64, -1, true, true)), "QSA exact ties and near-cutoff order");
        ++cases;
    }
    auto query = qsa_score_data({1, 65, 3, 37}, 41, tb::kBFloat16);
    auto keys = qsa_score_data({1, 129, 37}, 43, tb::kBFloat16);
    Tensor output;
    Graph graph;
    {
        QsaScoreSetting setting(2);
        {
            GraphWarmupScope warmup(default_context(), mfq_current_cuda_stream());
            output = qsa_block_scores(query, keys);
        }
        output = {};
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        graph.capture_begin_shared();
        output = qsa_block_scores(query, keys);
        graph.capture_end();
    }
    for (int salt : {47, 53}) {
        query.copy_(qsa_score_data({1, 65, 3, 37}, salt, tb::kBFloat16));
        graph.replay();
        QsaScoreSetting setting(0);
        qsa_check_score_selection(output, qsa_block_scores(query, keys), qsa_score_tolerance(query, keys));
        ++cases;
    }
    qsa_score_attention_case();
    std::cout << "qsa_prefill_score_cases=" << cases
        << " F32/BF16/F16, strided/mixed, tails, fallback, graph, exact top-k PASS\n";
}

void qsa_prefill_score_benchmark(int rows, int mode, bool bf16) {
    if (rows != 1024 && rows != 4096 && rows != 16384) throw std::runtime_error("invalid score benchmark rows");
    if (mode < 0 || mode > 2) throw std::runtime_error("invalid score benchmark mode");
    QsaScoreSetting setting(mode);
    const auto dtype = bf16 ? tb::kBFloat16 : tb::kFloat32;
    auto query = qsa_score_data({1, 4, rows, 128}, 17, dtype).permute({0, 2, 1, 3});
    auto keys = qsa_score_data({1, rows / 4, 128}, 31, bf16 ? dtype : tb::kFloat16);
    const auto context = default_context();
    const auto before = context->memory_stats();
    auto output = qsa_block_scores(query, keys);
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    const auto after = context->memory_stats();
    if (after.peak <= before.peak) throw std::runtime_error("score workspace peak was obscured by setup");
    const auto peak_workspace = after.peak - before.allocated;
    for (int i = 0; i < 10; ++i) {
        output = {};
        output = qsa_block_scores(query, keys);
    }
    output = {};
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    cudaEvent_t begin, end;
    MFQ_CUDA_CHECK(cudaEventCreate(&begin));
    MFQ_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> samples;
    for (int sample = 0; sample < 5; ++sample) {
        MFQ_CUDA_CHECK(cudaEventRecord(begin, mfq_current_cuda_stream()));
        for (int i = 0; i < 20; ++i) {
            output = {};
            output = qsa_block_scores(query, keys);
        }
        MFQ_CUDA_CHECK(cudaEventRecord(end, mfq_current_cuda_stream()));
        MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        samples.push_back(ms / 20);
    }
    MFQ_CUDA_CHECK(cudaEventDestroy(begin));
    MFQ_CUDA_CHECK(cudaEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    std::cout << "qsa_prefill_score_bench rows=" << rows << " pools=" << rows / 4 << " heads=4 width=128 mode="
        << mode << " bf16=" << bf16 << " ms=" << samples[2] << " min_ms=" << samples.front()
        << " max_ms=" << samples.back() << " peak_workspace_bytes=" << peak_workspace
        << " output_bytes=" << output.nbytes() << " scope=tensor_allocations_including_output\n";
}
