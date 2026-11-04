#include "batch_mtp.h"

#include "mfq_cuda_sampling_ops.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <numeric>
#include <string_view>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
namespace policy = mfq::engine::mtp;
using Tensor = tb::Tensor;

// Batched MTP is opt-in while it runs eagerly: until draft and verify replay
// CUDA Graphs, an exclusive MTP request on the graph path can be faster.
bool batch_mtp_enabled() {
    const char* option = std::getenv("MFQ_CONTINUOUS_BATCH_MTP");
    return option && std::string_view(option) == "1";
}

bool batch_mtp_supported(const MtpModule* predictor, const MfqSamplingParams& sampling) {
    return batch_mtp_enabled() && predictor && predictor->supports_batch() &&
        (sampling.temperature <= 0 || sampling.top_k == 1);
}

int64_t batch_mtp_checkpoints(const MtpModule& predictor) {
    if (const char* option = std::getenv("MFQ_BATCH_MTP_CHECKPOINTS")) return std::max(0, std::atoi(option));
    return predictor.maximum_draft_depth();
}

BatchMtpRow::~BatchMtpRow() {
    // A result can be abandoned by cancellation or coroutine destruction.
    if (resolve) {
        try { resolve(0); } catch (...) {}
    }
}

void BatchMtpRow::discard() {
    if (!resolve) return;
    resolve(0);
    resolve = {};
}

std::unique_ptr<BatchMtpRow> make_batch_mtp(MtpModule* predictor, const MfqSamplingParams& sampling,
        bool gradual_warmup) {
    if (!sampling.enable_mtp || !batch_mtp_supported(predictor, sampling)) return {};
    auto row = std::make_unique<BatchMtpRow>();
    row->predictor = predictor->make_batch_state();
    const auto* batch_policy = std::getenv("MFQ_MTP_METAL_POLICY");
    row->depth = std::make_unique<policy::GenerationPolicy<policy::DsparkDepthController>>(
        std::min(predictor->maximum_draft_depth(),
            std::clamp(sampling.mtp_max_draft_tokens, 1, policy::kMaximumDraftDepth)),
        false, nullptr, gradual_warmup, gradual_warmup && (!batch_policy || batch_policy[0] != '0'));
    row->stats.standard_tokens = 1;
    row->stats.available = row->stats.used = true;
    return row;
}

void prime_batch_mtp(MtpModule& predictor, const MtpTarget& target, BatchMtpRow& row,
        Tensor raw, const Tensor& ids, int64_t offset, int64_t prompt) {
    if (!row.predictor) row.predictor = predictor.make_batch_state();
    const auto pairs = std::min(raw.size(1), prompt - offset - 1);
    if (pairs > 0) {
        (void)predictor.batch_step(target, {row.predictor.get()}, raw.narrow(1, 0, pairs).contiguous(),
            ids.narrow(1, offset + 1, pairs).contiguous());
    }
    row.history_hidden = raw.narrow(1, raw.size(1) - 1, 1).clone();
}

static Tensor device_ids(const std::vector<int64_t>& ids, int64_t rows, const Tensor& like) {
    return tb::tensor(ids, tb::TensorOptions().dtype(tb::kInt64)).reshape({rows, -1}).to(like.device());
}

static int32_t greedy_sample(BatchMtpInput input, Tensor logits, const Tensor& counts,
        const MfqTokenConstraintPtr& constraint = {}) {
    if (input.sampler->has_penalties()) logits = logits.clone();
    return sample_logits(*input.sampler, std::move(logits), counts, constraint).item<int64_t>();
}

std::vector<int64_t> run_batch_mtp(MtpModule& predictor, const MtpTarget& target,
        const std::vector<BatchMtpInput>& inputs, int64_t max_context, const BatchMtpVerify& verify,
        int64_t draft_tokens) {
    using Clock = std::chrono::steady_clock;
    namespace policy = mfq::engine::mtp;
    const auto started = Clock::now();
    auto cycle_ms = std::make_shared<double>(0.0);
    const auto batch = inputs.size();
    std::vector<int> depths(batch), widths(batch, 1);
    std::vector<std::vector<int32_t>> drafts(batch);
    std::vector<Tensor> counts(batch), chain(batch), sample(batch);
    std::vector<int64_t> history_boundary(batch);
    std::vector<bool> primed(batch, false);
    Tensor device_reference;
    std::map<int64_t, std::vector<std::size_t>> histories;
    int maximum = 0;
    // Uniform target width is bounded by the shortest context headroom. Rows
    // with a smaller requested depth use padding and commit only their prefix.
    int64_t headroom = policy::kMaximumDraftDepth;
    for (const auto& input : inputs)
        headroom = std::min(headroom, max_context - input.request->cache_length - 1);
    for (std::size_t row = 0; row < batch; ++row) {
        const auto& input = inputs[row];
        if (!input.mtp) continue;
        auto& mtp = *input.mtp;
        device_reference = mtp.history_hidden;
        MFQ_RUNTIME_CHECK(!mtp.resolve && mtp.predictor && !mtp.history_ids.empty(),
                          "batched MTP has an unresolved window or missing predictor history");
        MFQ_RUNTIME_CHECK(mtp.history_hidden.defined() &&
            mtp.history_hidden.size(1) == int64_t(mtp.history_ids.size()),
            "batched MTP hidden/token history lost alignment");
        const auto remaining = input.request->generation_limit - input.request->produced;
        depths[row] = std::max<int64_t>(0, std::min<int64_t>(
            mtp.depth->depth(), std::min<int64_t>(remaining - 1, headroom)));
    }
    // Shorten the deepest chains first; later rows give up a tied draft.
    if (draft_tokens >= 0) {
        for (auto total = std::accumulate(depths.begin(), depths.end(), int64_t(0)); total > draft_tokens; --total)
            --*std::max_element(depths.rbegin(), depths.rend());
    }
    for (std::size_t row = 0; row < batch; ++row) {
        const auto& input = inputs[row];
        if (!input.mtp) continue;
        widths[row] += depths[row];
        maximum = std::max(maximum, depths[row]);
        if (depths[row] > 0 || input.mtp->history_ids.size() >= 16)
            histories[input.mtp->history_ids.size()].push_back(row);
        counts[row] = input.counts.defined() ? input.counts.clone() : Tensor{};
    }
    for (const auto& [width, rows] : histories) {
        std::vector<MtpBatchState*> states;
        std::vector<Tensor> hidden;
        std::vector<int64_t> ids;
        for (const auto row : rows) {
            auto& mtp = *inputs[row].mtp;
            states.push_back(mtp.predictor.get());
            hidden.push_back(mtp.history_hidden);
            ids.insert(ids.end(), mtp.history_ids.begin(), mtp.history_ids.end());
        }
        auto joined = tb::cat(hidden, 0);
        auto head = predictor.batch_step(target, states, joined, device_ids(ids, rows.size(), joined));
        for (std::size_t index = 0; index < rows.size(); ++index) {
            const auto row = rows[index];
            sample[row] = head.sample_hidden.narrow(0, index, 1).narrow(1, width - 1, 1);
            chain[row] = head.chain_hidden.narrow(0, index, 1).narrow(1, width - 1, 1);
            history_boundary[row] = states[index]->position;
            primed[row] = true;
            inputs[row].mtp->history_ids.clear();
            inputs[row].mtp->history_hidden = {};
        }
    }
    for (int position = 0; position < maximum; ++position) {
        std::vector<std::size_t> active;
        std::vector<Tensor> hidden;
        for (std::size_t row = 0; row < batch; ++row) {
            if (depths[row] <= position) continue;
            active.push_back(row);
            hidden.push_back(sample[row]);
        }
        auto head_hidden = tb::cat(hidden, 0);
        if (!predictor.preserve_output_dtype()) head_hidden = head_hidden.to(tb::kFloat16);
        auto logits = target.logits(head_hidden.contiguous()).reshape({int64_t(active.size()), -1});
        Tensor greedy;
        if (std::all_of(active.begin(), active.end(), [&](auto row) { return !inputs[row].counts.defined(); }))
            greedy = sample_greedy_cuda(logits.contiguous()).to(tb::kCPU).contiguous();
        std::vector<MtpBatchState*> states;
        std::vector<Tensor> next_hidden;
        std::vector<int64_t> next_ids;
        std::vector<std::size_t> next_rows;
        for (std::size_t index = 0; index < active.size(); ++index) {
            const auto row = active[index];
            const int32_t token = greedy.defined() ? greedy.data_ptr<int64_t>()[index]
                : greedy_sample(inputs[row], logits.narrow(0, index, 1), counts[row]);
            drafts[row].push_back(token);
            if (counts[row].defined())
                sample_token_counts_add_cuda(counts[row], device_ids({token}, 1, logits).reshape({-1}));
            if (depths[row] <= position + 1) continue;
            states.push_back(inputs[row].mtp->predictor.get());
            next_hidden.push_back(chain[row]);
            next_ids.push_back(token);
            next_rows.push_back(row);
        }
        if (!states.empty()) {
            auto joined = tb::cat(next_hidden, 0);
            auto head = predictor.batch_step(target, states, joined, device_ids(next_ids, states.size(), joined));
            for (std::size_t index = 0; index < next_rows.size(); ++index) {
                sample[next_rows[index]] = head.sample_hidden.narrow(0, index, 1);
                chain[next_rows[index]] = head.chain_hidden.narrow(0, index, 1);
            }
        }
    }
    for (std::size_t row = 0; row < batch; ++row)
        if (primed[row]) inputs[row].mtp->predictor->trim(history_boundary[row]);

    std::vector<int64_t> ids(batch * (maximum + 1), 0);
    for (std::size_t row = 0; row < batch; ++row) {
        ids[row * (maximum + 1)] = inputs[row].request->pending_token;
        std::copy(drafts[row].begin(), drafts[row].end(), ids.begin() + row * (maximum + 1) + 1);
    }
    auto host = tb::tensor(ids, tb::TensorOptions().dtype(tb::kInt64)).reshape({int64_t(batch), maximum + 1});
    const auto verify_started = Clock::now();
    auto verified = verify(host, host.to(device_reference.device()), widths);
    Tensor greedy_targets;
    if (std::all_of(inputs.begin(), inputs.end(), [](const auto& input) {
            return input.sampler->greedy() && !input.counts.defined();
        })) {
        greedy_targets = sample_greedy_cuda(verified.logits.contiguous()
            .reshape({int64_t(batch) * (maximum + 1), -1})).to(tb::kCPU).contiguous();
    }
    std::vector<int64_t> tokens(batch);
    for (std::size_t row = 0; row < batch; ++row) {
        const auto input = inputs[row];
        auto targets = verified.logits.narrow(0, row, 1).reshape({maximum + 1, -1});
        if (!input.mtp) {
            tokens[row] = greedy_sample(input, targets.narrow(0, 0, 1), input.counts, input.request->token_constraint);
            verified.resolve[row](1);
            continue;
        }
        auto& mtp = *input.mtp;
        auto row_counts = input.counts.defined() ? input.counts.clone() : Tensor{};
        std::vector<int32_t> target_tokens;
        for (int position = 0; position <= depths[row]; ++position) {
            target_tokens.push_back(greedy_targets.defined()
                ? greedy_targets.data_ptr<int64_t>()[row * (maximum + 1) + position]
                : greedy_sample(input, targets.narrow(0, position, 1), row_counts));
            if (position < depths[row] && row_counts.defined())
                sample_token_counts_add_cuda(row_counts, device_ids({drafts[row][position]}, 1, targets).reshape({-1}));
        }
        auto result = drafts[row].empty() ? policy::ChainVerification{0, target_tokens.front(), true}
            : policy::verify_greedy(drafts[row], target_tokens);
        std::vector<int64_t> candidates(drafts[row].begin(), drafts[row].begin() + result.accepted_drafts);
        candidates.push_back(result.next_token);
        if (input.request->token_constraint) {
            auto cursor = input.request->token_constraint->clone();
            MFQ_RUNTIME_CHECK(cursor, "batched MTP grammar clone is incomplete");
            auto grammar_counts = input.counts.defined() ? input.counts.clone() : Tensor{};
            for (std::size_t position = 0; position < candidates.size(); ++position) {
                if (!cursor->allows(candidates[position])) {
                    candidates.resize(position + 1);
                    candidates.back() = greedy_sample(input, targets.narrow(0, position, 1), grammar_counts, cursor);
                    result.accepted_drafts = position;
                    break;
                }
                cursor->accept(candidates[position]);
                if (grammar_counts.defined())
                    sample_token_counts_add_cuda(grammar_counts,
                        device_ids({candidates[position]}, 1, targets).reshape({-1}));
            }
        }
        mtp.candidates = std::move(candidates);
        mtp.accepted = result.accepted_drafts;
        mtp.drafted = depths[row];
        tokens[row] = mtp.candidates.front();
        const auto start_position = input.request->cache_length;
        auto raw = verified.raw.narrow(0, row, 1).clone();
        mtp.resolve = [resolve = std::move(verified.resolve[row]), start_position, input, raw,
                cycle_ms, used = depths[row], accepted = int(result.accepted_drafts)](int emitted) {
            const auto resolve_started = Clock::now();
            resolve(emitted + 1);
            input.request->cache_length = start_position + emitted + 1;
            auto history = raw.narrow(1, 0, emitted + 1);
            input.mtp->history_hidden = input.mtp->history_hidden.defined()
                ? tb::cat({input.mtp->history_hidden, history}, 1) : history.clone();
            // Include commit work, excluding time spent waiting for a consumer.
            auto& row = *input.mtp;
            const double elapsed = *cycle_ms + std::chrono::duration<double, std::milli>(
                Clock::now() - resolve_started).count();
            if (row.depth->parked()) row.depth->observe_plain(elapsed);
            else row.depth->observe(used, accepted, elapsed);
            row.stats.selected_depth = row.depth->depth();
            row.stats.park_count = row.depth->parks();
            row.stats.reentry_probes = row.depth->reentries();
            for (int depth = 0; depth <= row.depth->maximum_depth(); ++depth)
                row.stats.measured_depth_ms[depth] = row.depth->measured_cycle_ms(depth).value_or(0.0);
        };
        ++mtp.stats.cycles;
        mtp.stats.head_ms += std::chrono::duration<double, std::milli>(verify_started - started).count();
        mtp.stats.target_ms += std::chrono::duration<double, std::milli>(Clock::now() - verify_started).count();
        if (depths[row] == 0) ++mtp.stats.standard_tokens;
        mtp.stats.drafted_tokens += depths[row];
        mtp.stats.accepted_tokens += result.accepted_drafts;
        ++mtp.stats.depth_cycles[depths[row]];
        for (int position = 0; position < depths[row]; ++position) ++mtp.stats.position_drafted[position];
        for (std::size_t position = 0; position < result.accepted_drafts; ++position)
            ++mtp.stats.position_accepted[position];
    }
    *cycle_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    return tokens;
}

void accept_batch_mtp_tokens(BatchMtpInput input, const mfq::engine::TokenOutput& delta) {
    MfqCudaGuard guard(input.mtp->history_hidden.get_device());
    for (const auto token : delta.token_ids) {
        if (input.counts.defined())
            sample_token_counts_add_cuda(input.counts,
                device_ids({token}, 1, input.mtp->history_hidden).reshape({-1}));
        if (input.request->token_constraint) input.request->token_constraint->accept(token);
    }
}

mfq::engine::TokenOutput publish_batch_mtp(BatchMtpInput input, mfq::engine::InferenceOutput& output) {
    auto& row = *input.mtp;
    MFQ_RUNTIME_CHECK(row.resolve && !row.candidates.empty(), "batched MTP has no verified result");
    auto delta = output.append(row.candidates);
    const int retained = std::min<int>(row.accepted, delta.token_ids.size());
    const auto started = std::chrono::steady_clock::now();
    row.resolve(retained);
    row.resolve = {};
    row.history_ids.insert(row.history_ids.end(), delta.token_ids.begin(), delta.token_ids.end());
    accept_batch_mtp_tokens(input, delta);
    row.stats.rollback_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    output.metrics.mtp = row.stats;
    return delta;
}

} // namespace mfq::cuda
