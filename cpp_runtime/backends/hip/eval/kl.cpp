#include "models/nemotron_h/ops.h"
#include "hip/hip_runtime.h"
#include "kl.h"

#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"
#include "storage/weight_loader.h"
#include "quant_linear.h"
#include "hip_execution.h"
#include "storage/moe_expert_cache.h"
#include "mfq/kernels/cuda/deepseek_v4_attention.h"
#include "mfq/kernels/cuda/deepseek_v4_hc.h"
#include "mfq/kernels/cuda/deepseek_v41.h"
#include "mfq_tensor_backend.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct KlEvalChunk {
    std::vector<int32_t> tokens;
    std::vector<float> target_log_probs;
    int target_start = 0;
    int score_count = 0;
    std::streamoff row_offset = 0;
};

static void validate_kl_execution_geometry(
        int64_t execution_n_batch,
        int64_t execution_n_ubatch,
        const KlReferenceContract & reference_contract,
        const char * execution_path) {
    if (reference_contract.n_batch == 0) return;
    if (execution_n_batch != reference_contract.n_batch ||
            execution_n_ubatch != reference_contract.n_ubatch) {
        std::ostringstream message;
        message << execution_path
                << " KL execution geometry n_batch=" << execution_n_batch
                << " n_ubatch=" << execution_n_ubatch
                << " does not match reference n_batch="
                << reference_contract.n_batch
                << " n_ubatch=" << reference_contract.n_ubatch;
        throw std::runtime_error(message.str());
    }
}

const char * kl_evaluator_name(KlEvaluator evaluator) {
    return evaluator == KlEvaluator::Legacy ? "legacy" : "optimized";
}

template <typename Model>
static int run_kl_eval(
        Model& model,
        const std::string & path,
        int max_chunks,
        KlEvaluator evaluator,
        int score_override,
        const KlReferenceContract & reference_contract) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open KL logits file: " + path);
    char magic[8];
    int32_t n_vocab = 0;
    f.read(magic, sizeof(magic));
    if (!f) {
        throw std::runtime_error("invalid KL logits header");
    }
    const std::string reference_format(magic, sizeof(magic));
    std::vector<KlEvalChunk> eval_chunks;
    if (reference_format == "_logits_") {
        uint32_t n_ctx = 0;
        int32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&n_ctx), sizeof(n_ctx));
        f.read(reinterpret_cast<char*>(&n_vocab), sizeof(n_vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || n_ctx < 2 || n_vocab <= 0 || n_chunks <= 0) {
            throw std::runtime_error("invalid legacy KL logits header");
        }
        std::vector<int32_t> tokens((size_t)n_ctx * n_chunks);
        f.read(
            reinterpret_cast<char*>(tokens.data()),
            (std::streamsize)(tokens.size() * sizeof(int32_t)));
        if (!f) throw std::runtime_error("truncated legacy KL token header");
        const int first = (int)n_ctx / 2;
        const int score_count = (int)n_ctx - 1 - first;
        eval_chunks.resize((size_t)n_chunks);
        const int32_t legacy_bos = tokens[0];
        int legacy_bos_replacements = 0;
        for (int ci = 0; ci < n_chunks; ++ci) {
            auto begin = tokens.begin() + (size_t)ci * n_ctx;
            eval_chunks[(size_t)ci].tokens.assign(begin, begin + n_ctx);
            // The reference perplexity evaluator temporarily replaces every Gemma context
            // chunk's first corpus token with BOS before evaluating it, then
            // serializes the original corpus tokens in this legacy header.
            // Qwen legacy references use the serialized token directly.
            if (model.metadata.gemma4 &&
                    eval_chunks[(size_t)ci].tokens[0] != legacy_bos) {
                eval_chunks[(size_t)ci].tokens[0] = legacy_bos;
                ++legacy_bos_replacements;
            }
            eval_chunks[(size_t)ci].target_start = first + 1;
            eval_chunks[(size_t)ci].score_count = score_count;
        }
        std::cout << "cpp_kl_legacy_chunk_bos="
                  << (model.metadata.gemma4 ? legacy_bos : -1)
                  << " replacements=" << legacy_bos_replacements
                  << " model_type=" << model.model_type() << "\n";
    } else if (reference_format == "_logit2_" || reference_format == "_logit3_") {
        const bool has_exact_target_log_probs = reference_format == "_logit3_";
        uint32_t vocab = 0;
        uint32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&vocab), sizeof(vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || vocab == 0 || vocab > (uint32_t)std::numeric_limits<int32_t>::max() ||
            n_chunks == 0 || n_chunks > (1u << 20)) {
            throw std::runtime_error("invalid trace KL logits header");
        }
        n_vocab = (int32_t)vocab;
        eval_chunks.resize((size_t)n_chunks);
        std::vector<uint32_t> token_counts(n_chunks);
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            uint32_t target_start = 0;
            uint32_t score_count = 0;
            f.read(reinterpret_cast<char*>(&token_counts[ci]), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&target_start), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&score_count), sizeof(uint32_t));
            if (!f || token_counts[ci] < 2 || target_start < 1 ||
                target_start >= token_counts[ci] || score_count < 1 ||
                score_count > token_counts[ci] - target_start) {
                throw std::runtime_error("invalid trace KL chunk descriptor");
            }
            eval_chunks[ci].target_start = (int)target_start;
            eval_chunks[ci].score_count = (int)score_count;
        }
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            auto & chunk = eval_chunks[ci];
            chunk.tokens.resize(token_counts[ci]);
            f.read(
                reinterpret_cast<char*>(chunk.tokens.data()),
                (std::streamsize)(chunk.tokens.size() * sizeof(int32_t)));
            if (!f) throw std::runtime_error("truncated trace KL token header");
        }
        if (has_exact_target_log_probs) {
            for (auto & chunk : eval_chunks) {
                chunk.target_log_probs.resize((size_t)chunk.score_count);
                f.read(
                    reinterpret_cast<char*>(chunk.target_log_probs.data()),
                    (std::streamsize)(chunk.target_log_probs.size() * sizeof(float)));
                if (!f) throw std::runtime_error("truncated trace KL target log probabilities");
                for (float value : chunk.target_log_probs) {
                    if (!std::isfinite(value) || value > 0.0f) {
                        throw std::runtime_error("invalid trace KL target log probability");
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("invalid KL logits magic");
    }
    const std::streamoff data_offset = f.tellg();
    const int nv = 2 * ((n_vocab + 1) / 2) + 4;
    std::streamoff row_offset = data_offset;
    for (auto & chunk : eval_chunks) {
        chunk.row_offset = row_offset;
        row_offset += (std::streamoff)chunk.score_count * nv * (std::streamoff)sizeof(uint16_t);
    }
    f.seekg(0, std::ios::end);
    if (!f || f.tellg() < row_offset) {
        throw std::runtime_error("truncated KL logits rows");
    }
    const int available_chunks = (int)eval_chunks.size();
    const int chunks = max_chunks < 0
        ? available_chunks
        : std::min(max_chunks, available_chunks);
    if (chunks <= 0) throw std::runtime_error("KL evaluation requires at least one chunk");
    for (int ci = 0; ci < chunks; ++ci) {
        if ((int64_t)eval_chunks[(size_t)ci].tokens.size() >
                model.max_position_embeddings()) {
            throw std::runtime_error(
                "KL reference exceeds model context capacity");
        }
    }
    const int64_t execution_n_batch =
        (int64_t)eval_chunks[0].tokens.size();
    if (reference_contract.n_batch != 0) {
        for (int ci = 0; ci < chunks; ++ci) {
            const int64_t execution_tokens =
                (int64_t)eval_chunks[(size_t)ci].tokens.size();
            validate_kl_execution_geometry(
                execution_tokens, execution_tokens,
                reference_contract, "single-sequence");
        }
    }
    constexpr int KL_BATCH = 8;
    double kld_sum = 0.0;
    double reverse_kld_sum = 0.0;
    double bf16_ce_sum = 0.0;
    double mfq_ce_sum = 0.0;
    int64_t same_top = 0;
    int64_t count = 0;
    const bool optimized = evaluator == KlEvaluator::Optimized;
    auto started = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_execution evaluator="
              << kl_evaluator_name(evaluator)
              << " graph=single_sequence"
              << " available_chunks=" << available_chunks
              << " selected_chunks=" << chunks
              << " execution_n_seq=1"
              << " execution_n_batch=" << execution_n_batch
              << " execution_n_ubatch=" << execution_n_batch
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << "\n";

    for (int ci = 0; ci < chunks; ++ci) {
        const auto & chunk = eval_chunks[(size_t)ci];
        const int stored_score_count = chunk.score_count;
        const int score_count = score_override < 0 ? stored_score_count : score_override;
        if (score_count > stored_score_count) {
            throw std::runtime_error(
                "--kl-score-count exceeds the stored chunk score count");
        }
        const int first = chunk.target_start - 1;
        model.reset(1);
        std::vector<int64_t> chunk_tokens(chunk.tokens.size());
        for (size_t j = 0; j < chunk.tokens.size(); ++j) {
            chunk_tokens[j] = chunk.tokens[j];
        }
        auto ids = mfq_tensor_backend::from_blob(chunk_tokens.data(), {1, (int64_t)chunk_tokens.size()},
                                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone().to(mfq_tensor_backend::kCUDA);
        mfq_tensor_backend::Tensor pred;
        if (optimized) {
            auto hidden = model.hidden_forward(ids);
            auto selected_hidden = hidden.index({
                0, Slice(first, first + score_count), Slice()
            }).contiguous();
            pred = model.logits_from_hidden(selected_hidden);
        } else if (score_count < stored_score_count) {
            (void)model.hidden_forward(ids.index({Slice(), Slice(0, first)}));
            auto logits = model.forward(ids.index({Slice(), Slice(first, first + score_count)}));
            pred = logits.index({0, Slice(), Slice()});
        } else {
            auto logits = model.forward(ids);
            pred = logits.index({0, Slice(first, first + score_count), Slice()});
        }
        if (pred.size(1) != n_vocab) throw std::runtime_error("KL vocab size mismatch");

        mfq_tensor_backend::Tensor optimized_kld_sum;
        mfq_tensor_backend::Tensor optimized_reverse_kld_sum;
        mfq_tensor_backend::Tensor optimized_bf16_ce_sum;
        mfq_tensor_backend::Tensor optimized_mfq_ce_sum;
        mfq_tensor_backend::Tensor optimized_same_top;
        if (optimized) {
            const auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
            optimized_kld_sum = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kFloat64));
            optimized_reverse_kld_sum = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kFloat64));
            optimized_bf16_ce_sum = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kFloat64));
            optimized_mfq_ce_sum = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kFloat64));
            optimized_same_top = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kInt64));
        }
        f.seekg(chunk.row_offset);
        for (int s = 0; s < score_count; s += KL_BATCH) {
            const int b = std::min(KL_BATCH, score_count - s);
            std::vector<uint16_t> rows((size_t)b * nv);
            f.read(reinterpret_cast<char*>(rows.data()), (std::streamsize)(rows.size() * sizeof(uint16_t)));
            if (!f) throw std::runtime_error("truncated KL logits data");

            std::vector<float> scales(b), mins(b);
            std::vector<int32_t> codes((size_t)b * n_vocab);
            for (int r = 0; r < b; ++r) {
                const uint16_t* row = rows.data() + (size_t)r * nv;
                std::memcpy(&scales[r], row + 0, sizeof(float));
                std::memcpy(&mins[r], row + 2, sizeof(float));
                for (int v = 0; v < n_vocab; ++v) codes[(size_t)r * n_vocab + v] = row[4 + v];
            }
            auto scale = mfq_tensor_backend::from_blob(scales.data(), {b}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone().to(mfq_tensor_backend::kCUDA);
            auto min_lp = mfq_tensor_backend::from_blob(mins.data(), {b}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone().to(mfq_tensor_backend::kCUDA);
            auto base_codes = mfq_tensor_backend::from_blob(
                codes.data(), {b, n_vocab}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto base_logp = base_codes.to(mfq_tensor_backend::kFloat32) * scale.unsqueeze(1) +
                min_lp.unsqueeze(1);
            auto q = pred.index({Slice(s, s + b), Slice()}).to(mfq_tensor_backend::kFloat32);
            auto lse = mfq_tensor_backend::logsumexp(q, -1);
            auto quant_logp = q - lse.unsqueeze(1);
            auto normalized_base_logp =
                base_logp - mfq_tensor_backend::logsumexp(base_logp, -1, true);
            auto p_base = mfq_tensor_backend::exp(base_logp).masked_fill(base_codes.eq(0), 0.0f);
            auto kld = (p_base * (base_logp - q + lse.unsqueeze(1))).sum(-1);
            auto reverse_kld =
                (mfq_tensor_backend::exp(quant_logp) * (quant_logp - normalized_base_logp)).sum(-1);
            if (optimized) {
                optimized_kld_sum.add_(kld.to(mfq_tensor_backend::kFloat64).sum());
                optimized_reverse_kld_sum.add_(
                    reverse_kld.to(mfq_tensor_backend::kFloat64).sum());
            } else {
                kld_sum += kld.to(mfq_tensor_backend::kFloat64).sum().item<double>();
                reverse_kld_sum +=
                    reverse_kld.to(mfq_tensor_backend::kFloat64).sum().item<double>();
            }
            std::vector<int64_t> target_ids(b);
            for (int r = 0; r < b; ++r) {
                target_ids[r] = chunk_tokens[(size_t)chunk.target_start + s + r];
            }
            auto target = mfq_tensor_backend::from_blob(target_ids.data(), {b, 1},
                                            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
                              .clone().to(mfq_tensor_backend::kCUDA);
            if (chunk.target_log_probs.empty()) {
                auto batch_bf16_ce =
                    -base_logp.gather(1, target).to(mfq_tensor_backend::kFloat64).sum();
                if (optimized) {
                    optimized_bf16_ce_sum.add_(batch_bf16_ce);
                } else {
                    bf16_ce_sum += batch_bf16_ce.item<double>();
                }
            } else {
                for (int r = 0; r < b; ++r) {
                    bf16_ce_sum -= chunk.target_log_probs[(size_t)s + r];
                }
            }
            auto batch_mfq_ce = (lse.unsqueeze(1) - q.gather(1, target))
                                     .to(mfq_tensor_backend::kFloat64).sum();
            auto batch_same_top =
                q.argmax(-1).eq(base_logp.argmax(-1)).sum();
            if (optimized) {
                optimized_mfq_ce_sum.add_(batch_mfq_ce);
                optimized_same_top.add_(batch_same_top);
            } else {
                mfq_ce_sum += batch_mfq_ce.item<double>();
                same_top += batch_same_top.item<int64_t>();
            }
            count += b;
        }
        if (optimized) {
            kld_sum += optimized_kld_sum.item<double>();
            reverse_kld_sum += optimized_reverse_kld_sum.item<double>();
            if (chunk.target_log_probs.empty()) {
                bf16_ce_sum += optimized_bf16_ce_sum.item<double>();
            }
            mfq_ce_sum += optimized_mfq_ce_sum.item<double>();
            same_top += optimized_same_top.item<int64_t>();
        }
        mfq_hip_synchronize();
        std::cout << "cpp_kl_chunk=" << (ci + 1)
                  << " mean=" << (kld_sum / (double)count)
                  << " mean_kld_q_ref=" << (reverse_kld_sum / (double)count)
                  << " same_top=" << ((double)same_top / (double)count) << "\n";
    }
    auto ended = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_result chunks=" << chunks
              << " scored_tokens=" << count
              << " sec=" << std::chrono::duration<double>(ended - started).count()
              << " kld=" << (kld_sum / (double)count)
              << " mean_kld_q_ref=" << (reverse_kld_sum / (double)count)
              << " bf16_ce=" << (bf16_ce_sum / (double)count)
              << " mfq_ce=" << (mfq_ce_sum / (double)count)
              << " bf16_ppl=" << std::exp(bf16_ce_sum / (double)count)
              << " mfq_ppl=" << std::exp(mfq_ce_sum / (double)count)
              << " kld_pct_bf16=" << (100.0 * kld_sum / bf16_ce_sum)
              << " same_top=" << ((double)same_top / (double)count)
              << " same_top_count=" << same_top
              << " reference_format="
              << (reference_format == "_logit3_" ? "trace_v3" :
                  (reference_format == "_logit2_" ? "trace_v2" : "legacy"))
              << " execution=" << kl_evaluator_name(evaluator)
              << " graph=single_sequence"
              << " execution_n_seq=1"
              << " execution_n_batch=" << execution_n_batch
              << " execution_n_ubatch=" << execution_n_batch
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << "\n";
    return 0;
}

struct StreamedKlInput {
    std::string reference_format;
    int32_t n_vocab = 0;
    int nv = 0;
    std::vector<KlEvalChunk> chunks;
};

static StreamedKlInput load_streamed_kl_input(
    const std::string & path,
    int max_chunks) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open KL logits file: " + path);
    char magic[8];
    f.read(magic, sizeof(magic));
    if (!f) throw std::runtime_error("invalid KL logits header");

    StreamedKlInput input;
    input.reference_format.assign(magic, sizeof(magic));
    if (input.reference_format == "_logits_") {
        uint32_t n_ctx = 0;
        int32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&n_ctx), sizeof(n_ctx));
        f.read(reinterpret_cast<char*>(&input.n_vocab), sizeof(input.n_vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || n_ctx < 2 || input.n_vocab <= 0 || n_chunks <= 0) {
            throw std::runtime_error("invalid legacy KL logits header");
        }
        std::vector<int32_t> tokens((size_t)n_ctx * n_chunks);
        f.read(
            reinterpret_cast<char*>(tokens.data()),
            (std::streamsize)(tokens.size() * sizeof(int32_t)));
        if (!f) throw std::runtime_error("truncated legacy KL token header");
        const int first = (int)n_ctx / 2;
        const int score_count = (int)n_ctx - 1 - first;
        input.chunks.resize((size_t)n_chunks);
        for (int ci = 0; ci < n_chunks; ++ci) {
            auto begin = tokens.begin() + (size_t)ci * n_ctx;
            input.chunks[(size_t)ci].tokens.assign(begin, begin + n_ctx);
            input.chunks[(size_t)ci].target_start = first + 1;
            input.chunks[(size_t)ci].score_count = score_count;
        }
    } else if (
        input.reference_format == "_logit2_" ||
        input.reference_format == "_logit3_") {
        const bool has_exact_target_log_probs =
            input.reference_format == "_logit3_";
        uint32_t vocab = 0;
        uint32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&vocab), sizeof(vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || vocab == 0 ||
            vocab > (uint32_t)std::numeric_limits<int32_t>::max() ||
            n_chunks == 0 || n_chunks > (1u << 20)) {
            throw std::runtime_error("invalid trace KL logits header");
        }
        input.n_vocab = (int32_t)vocab;
        input.chunks.resize((size_t)n_chunks);
        std::vector<uint32_t> token_counts(n_chunks);
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            uint32_t target_start = 0;
            uint32_t score_count = 0;
            f.read(reinterpret_cast<char*>(&token_counts[ci]), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&target_start), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&score_count), sizeof(uint32_t));
            if (!f || token_counts[ci] < 2 || target_start < 1 ||
                target_start >= token_counts[ci] || score_count < 1 ||
                score_count > token_counts[ci] - target_start) {
                throw std::runtime_error("invalid trace KL chunk descriptor");
            }
            input.chunks[ci].target_start = (int)target_start;
            input.chunks[ci].score_count = (int)score_count;
        }
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            auto & chunk = input.chunks[ci];
            chunk.tokens.resize(token_counts[ci]);
            f.read(
                reinterpret_cast<char*>(chunk.tokens.data()),
                (std::streamsize)(chunk.tokens.size() * sizeof(int32_t)));
            if (!f) throw std::runtime_error("truncated trace KL token header");
        }
        if (has_exact_target_log_probs) {
            for (auto & chunk : input.chunks) {
                chunk.target_log_probs.resize((size_t)chunk.score_count);
                f.read(
                    reinterpret_cast<char*>(chunk.target_log_probs.data()),
                    (std::streamsize)(
                        chunk.target_log_probs.size() * sizeof(float)));
                if (!f) {
                    throw std::runtime_error(
                        "truncated trace KL target log probabilities");
                }
                for (float value : chunk.target_log_probs) {
                    if (!std::isfinite(value) || value > 0.0f) {
                        throw std::runtime_error(
                            "invalid trace KL target log probability");
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("invalid KL logits magic");
    }

    const std::streamoff data_offset = f.tellg();
    input.nv = 2 * ((input.n_vocab + 1) / 2) + 4;
    std::streamoff row_offset = data_offset;
    for (auto & chunk : input.chunks) {
        chunk.row_offset = row_offset;
        row_offset += (std::streamoff)chunk.score_count * input.nv *
            (std::streamoff)sizeof(uint16_t);
    }
    f.seekg(0, std::ios::end);
    if (!f || f.tellg() < row_offset) {
        throw std::runtime_error("truncated KL logits rows");
    }
    const int available_chunks = (int)input.chunks.size();
    const int chunks = max_chunks < 0
        ? available_chunks
        : std::min(max_chunks, available_chunks);
    if (chunks <= 0) {
        throw std::runtime_error("KL evaluation requires at least one chunk");
    }
    input.chunks.resize((size_t)chunks);
    return input;
}

static mfq_tensor_backend::Tensor streamed_kl_ids(
    const StreamedKlInput & input,
    int begin,
    int count,
    int64_t n_ctx) {
    std::vector<int64_t> token_ids((size_t)count * n_ctx);
    for (int bi = 0; bi < count; ++bi) {
        const auto & source = input.chunks[(size_t)(begin + bi)].tokens;
        for (int64_t ti = 0; ti < n_ctx; ++ti) {
            token_ids[(size_t)bi * n_ctx + (size_t)ti] =
                source[(size_t)ti];
        }
    }
    return mfq_tensor_backend::from_blob(
        token_ids.data(), {count, n_ctx},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
        .clone().to(mfq_tensor_backend::kCUDA);
}

struct KlEvalMetrics {
    double kld_sum = 0.0;
    double reverse_kld_sum = 0.0;
    double reference_ce_sum = 0.0;
    double quant_ce_sum = 0.0;
    int64_t same_top = 0;
    int64_t count = 0;
};

static void accumulate_streamed_kl_chunk(
    std::ifstream & f,
    const StreamedKlInput & input,
    int chunk_index,
    mfq_tensor_backend::Tensor pred,
    int score_count,
    KlEvalMetrics & metrics) {
    constexpr int KL_BATCH = 8;
    const auto & chunk = input.chunks[(size_t)chunk_index];
    f.seekg(chunk.row_offset);
    for (int s = 0; s < score_count; s += KL_BATCH) {
        const int b = std::min(KL_BATCH, score_count - s);
        std::vector<uint16_t> rows((size_t)b * input.nv);
        f.read(
            reinterpret_cast<char*>(rows.data()),
            (std::streamsize)(rows.size() * sizeof(uint16_t)));
        if (!f) throw std::runtime_error("truncated KL logits data");

        std::vector<float> scales(b), mins(b);
        std::vector<int32_t> codes((size_t)b * input.n_vocab);
        for (int r = 0; r < b; ++r) {
            const uint16_t * row = rows.data() + (size_t)r * input.nv;
            std::memcpy(&scales[r], row + 0, sizeof(float));
            std::memcpy(&mins[r], row + 2, sizeof(float));
            for (int v = 0; v < input.n_vocab; ++v) {
                codes[(size_t)r * input.n_vocab + v] = row[4 + v];
            }
        }
        auto scale = mfq_tensor_backend::from_blob(
            scales.data(), {b},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
            .clone().to(mfq_tensor_backend::kCUDA);
        auto min_lp = mfq_tensor_backend::from_blob(
            mins.data(), {b},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
            .clone().to(mfq_tensor_backend::kCUDA);
        auto base_codes = mfq_tensor_backend::from_blob(
            codes.data(), {b, input.n_vocab},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::kCUDA);
        auto base_logp = base_codes.to(mfq_tensor_backend::kFloat32) *
            scale.unsqueeze(1) + min_lp.unsqueeze(1);
        auto q = pred.index({Slice(s, s + b), Slice()}).to(mfq_tensor_backend::kFloat32);
        auto lse = mfq_tensor_backend::logsumexp(q, -1);
        auto quant_logp = q - lse.unsqueeze(1);
        auto normalized_base_logp =
            base_logp - mfq_tensor_backend::logsumexp(base_logp, -1, true);
        auto p_base =
            mfq_tensor_backend::exp(base_logp).masked_fill(base_codes.eq(0), 0.0f);
        auto kld =
            (p_base * (base_logp - q + lse.unsqueeze(1))).sum(-1);
        auto reverse_kld =
            (mfq_tensor_backend::exp(quant_logp) *
             (quant_logp - normalized_base_logp)).sum(-1);
        metrics.kld_sum +=
            kld.to(mfq_tensor_backend::kFloat64).sum().item<double>();
        metrics.reverse_kld_sum +=
            reverse_kld.to(mfq_tensor_backend::kFloat64).sum().item<double>();

        std::vector<int64_t> target_ids(b);
        for (int r = 0; r < b; ++r) {
            target_ids[r] =
                chunk.tokens[(size_t)chunk.target_start + s + r];
        }
        auto target = mfq_tensor_backend::from_blob(
            target_ids.data(), {b, 1},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
            .clone().to(mfq_tensor_backend::kCUDA);
        if (chunk.target_log_probs.empty()) {
            metrics.reference_ce_sum +=
                -base_logp.gather(1, target)
                    .to(mfq_tensor_backend::kFloat64).sum().item<double>();
        } else {
            for (int r = 0; r < b; ++r) {
                metrics.reference_ce_sum -=
                    chunk.target_log_probs[(size_t)s + r];
            }
        }
        metrics.quant_ce_sum +=
            (lse.unsqueeze(1) - q.gather(1, target))
                .to(mfq_tensor_backend::kFloat64).sum().item<double>();
        metrics.same_top +=
            q.argmax(-1).eq(base_logp.argmax(-1)).sum().item<int64_t>();
        metrics.count += b;
    }
}

template <typename Model>
int run_kl_eval_batched(
    Model& model,
    const std::string & reference_path,
    int max_chunks,
    int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract & reference_contract) {
    auto input = load_streamed_kl_input(reference_path, max_chunks);
    const int chunks = (int)input.chunks.size();
    const int64_t n_ctx = (int64_t)input.chunks[0].tokens.size();
    const int64_t n_batch = requested_n_batch == 0
        ? (reference_contract.n_batch == 0
              ? n_ctx : reference_contract.n_batch)
        : requested_n_batch;
    if (n_ctx > model.max_position_embeddings()) {
        throw std::runtime_error(
            "KL reference exceeds model context capacity");
    }
    if (n_ctx <= 0 || n_batch < n_ctx || n_batch % n_ctx != 0) {
        throw std::runtime_error(
            "optimized KL requires --kl-n-batch to be at least n_ctx "
            "and exactly divisible by n_ctx");
    }
    const int kl_n_seq = std::max<int64_t>(
        1, n_batch / n_ctx);
    validate_kl_execution_geometry(
        n_batch, n_batch, reference_contract, "optimized");
    KlKvCacheCapacityScope kv_cache_capacity_scope(
        n_ctx, model.execution->kl_mmq);
    const int target_start = input.chunks[0].target_start;
    int score_count = input.chunks[0].score_count;
    for (const auto & chunk : input.chunks) {
        if ((int64_t)chunk.tokens.size() != n_ctx ||
            chunk.target_start != target_start ||
            chunk.score_count != score_count) {
            throw std::runtime_error(
                "optimized KL requires uniform chunk geometry");
        }
    }
    if (score_override >= 0) {
        score_count = score_override;
        if (score_count < 1 ||
            score_count > input.chunks[0].score_count) {
            throw std::runtime_error(
                "--kl-score-count exceeds the stored chunk score count");
        }
    }
    if (input.n_vocab != model.vocab_size()) {
        throw std::runtime_error("KL vocab size mismatch");
    }
    int bos_replacements = 0;
    int32_t legacy_bos = -1;
    if (input.reference_format == "_logits_" &&
        model.metadata.gemma4) {
        legacy_bos = input.chunks[0].tokens[0];
        for (auto & chunk : input.chunks) {
            if (chunk.tokens[0] != legacy_bos) {
                chunk.tokens[0] = legacy_bos;
                ++bos_replacements;
            }
        }
    }

    std::ifstream reference(reference_path, std::ios::binary);
    if (!reference) {
        throw std::runtime_error(
            "cannot reopen KL logits file: " + reference_path);
    }
    const int first = target_start - 1;
    KlEvalMetrics metrics;
    auto started = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_execution evaluator=optimized"
              << " graph=batched_contexts"
              << " n_batch=" << n_batch
              << " n_ctx=" << n_ctx
              << " n_seq=" << kl_n_seq
              << " score_count=" << score_count
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << " logits_start=" << first
              << " bos=" << legacy_bos
              << " bos_replacements=" << bos_replacements
              << "\n";

    for (int begin = 0; begin < chunks; begin += kl_n_seq) {
        const int n_seq_batch =
            std::min(kl_n_seq, chunks - begin);
        auto ids = streamed_kl_ids(
            input, begin, n_seq_batch, n_ctx);
        model.reset(n_seq_batch);
        auto hidden = model.hidden_forward(ids);
        for (int bi = 0; bi < n_seq_batch; ++bi) {
            auto selected_hidden = hidden.index({
                bi, Slice(first, first + score_count), Slice()
            }).contiguous();
            auto pred =
                model.logits_from_hidden(selected_hidden);
            accumulate_streamed_kl_chunk(
                reference, input, begin + bi, pred,
                score_count, metrics);
            std::cout << "cpp_kl_chunk=" << (begin + bi + 1)
                      << " mean="
                      << (metrics.kld_sum / (double)metrics.count)
                      << " mean_kld_q_ref="
                      << (metrics.reverse_kld_sum /
                          (double)metrics.count)
                      << " same_top="
                      << ((double)metrics.same_top /
                          (double)metrics.count)
                      << "\n";
        }
        mfq_hip_synchronize();
    }
    auto ended = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_result chunks=" << chunks
              << " scored_tokens=" << metrics.count
              << " sec="
              << std::chrono::duration<double>(
                     ended - started).count()
              << " kld="
              << (metrics.kld_sum / (double)metrics.count)
              << " mean_kld_q_ref="
              << (metrics.reverse_kld_sum / (double)metrics.count)
              << " bf16_ce="
              << (metrics.reference_ce_sum / (double)metrics.count)
              << " mfq_ce="
              << (metrics.quant_ce_sum / (double)metrics.count)
              << " bf16_ppl="
              << std::exp(
                     metrics.reference_ce_sum /
                     (double)metrics.count)
              << " mfq_ppl="
              << std::exp(
                     metrics.quant_ce_sum /
                     (double)metrics.count)
              << " kld_pct_bf16="
              << (100.0 * metrics.kld_sum /
                  metrics.reference_ce_sum)
              << " same_top="
              << ((double)metrics.same_top /
                  (double)metrics.count)
              << " same_top_count=" << metrics.same_top
              << " reference_format="
              << (input.reference_format == "_logit3_"
                      ? "trace_v3"
                      : (input.reference_format == "_logit2_"
                             ? "trace_v2" : "legacy"))
               << " execution=optimized"
               << " graph=batched_contexts"
               << " n_ctx=" << n_ctx
               << " n_batch=" << n_batch
               << " n_seq=" << kl_n_seq
               << " score_count=" << score_count
               << " score_count_override=" << score_override
               << " reference_n_batch=" << reference_contract.n_batch
               << " reference_n_ubatch=" << reference_contract.n_ubatch
               << "\n";
    const auto& execution = *model.execution;
    std::cout << "cpp_kl_mmq"
              << " mmq=" << kl_mmq_mode_name(execution.kl_mmq.mode)
              << " activation_quantize_calls="
              << execution.kl_mmq.activation_quantize_calls
              << " dense_calls=" << execution.kl_mmq.dense_calls
              << " moe_calls=" << execution.kl_mmq.moe_calls
              << " fallback_calls=" << execution.kl_mmq.fallback_calls
              << "\n";
    return 0;
}

template <typename Model>
int run_selected_kl_eval(
    Model& model,
    const std::string & reference_path,
    int max_chunks,
    KlEvaluator evaluator,
    int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract & reference_contract) {
    return evaluator == KlEvaluator::Optimized
        ? run_kl_eval_batched(
              model, reference_path, max_chunks,
              requested_n_batch, score_override,
              reference_contract)
        : run_kl_eval(
              model, reference_path, max_chunks, evaluator,
              score_override, reference_contract);
}

int run_kl_eval_streamed(
    CudaExecutionContext& execution,
    const std::string & model_path,
    const std::string & config_path,
    const std::string & reference_path,
    const std::string & logits_output_path,
    int max_chunks,
    int layer_group,
    int chunk_batch,
    int score_override,
    const KlReferenceContract & reference_contract) {
    if (layer_group < 1 || chunk_batch < 1) {
        throw std::runtime_error(
            "streamed KL layer group and chunk batch must be positive");
    }
    execution.moe_route_stats.clear();
    MfqDropFileCacheGuard drop_cache_guard(
        true, execution.drop_file_cache);
    auto input = load_streamed_kl_input(reference_path, max_chunks);
    const int chunks = (int)input.chunks.size();
    const int64_t n_ctx = (int64_t)input.chunks[0].tokens.size();
    const int64_t execution_n_seq = std::min(chunk_batch, chunks);
    const int64_t execution_n_batch = execution_n_seq * n_ctx;
    validate_kl_execution_geometry(
        execution_n_batch, execution_n_batch,
        reference_contract, "streamed");
    const int target_start = input.chunks[0].target_start;
    int score_count = input.chunks[0].score_count;
    for (const auto & chunk : input.chunks) {
        if ((int64_t)chunk.tokens.size() != n_ctx ||
            chunk.target_start != target_start ||
            chunk.score_count != score_count) {
            throw std::runtime_error(
                "streamed KL currently requires uniform chunk geometry");
        }
    }
    if (score_override >= 0) {
        score_count = score_override;
        if (score_count < 1 ||
            score_count > input.chunks[0].score_count) {
            throw std::runtime_error(
                "--kl-score-count exceeds the stored chunk score count");
        }
    }
    const int first = target_start - 1;

    std::ofstream logits_output;
    std::filesystem::path logits_final;
    std::filesystem::path logits_partial;
    if (!logits_output_path.empty()) {
        logits_final = std::filesystem::path(logits_output_path);
        logits_partial = logits_final;
        logits_partial += ".partial";
        if (std::filesystem::exists(logits_final) ||
                std::filesystem::exists(logits_partial)) {
            throw std::runtime_error(
                "refusing to overwrite saved KL logits: " +
                logits_output_path);
        }
        if (!logits_final.parent_path().empty()) {
            std::filesystem::create_directories(
                logits_final.parent_path());
        }
        logits_output.open(logits_partial, std::ios::binary);
        if (!logits_output) {
            throw std::runtime_error(
                "cannot create saved KL logits: " +
                logits_partial.string());
        }
        const char magic[8] = {'_', 'm', 'f', 'q', 'f', '1', '6', '_'};
        const int32_t header[5] = {
            chunks,
            static_cast<int32_t>(n_ctx),
            target_start,
            score_count,
            input.n_vocab,
        };
        logits_output.write(magic, sizeof(magic));
        logits_output.write(
            reinterpret_cast<const char *>(header), sizeof(header));
    }

    auto started = std::chrono::steady_clock::now();
    auto model = mfq::cuda::load_causal_lm<
        mfq::cuda::DeepseekV4CausalLm>(
            execution, model_path, config_path, n_ctx, false);
    if (chunk_batch > 16) {
        throw std::runtime_error(
            "DeepSeek V4 streamed KL chunk batch must not exceed 16");
    }
    if (model.vocab_size() != input.n_vocab) {
        throw std::runtime_error("KL vocab size mismatch");
    }
    const auto& mfq = *model.source;

    const int64_t hidden_bytes =
        (int64_t)chunks * n_ctx * model.hc_mult() *
        model.hidden_size() * (int64_t)sizeof(mfq_half);
    auto hidden_cpu = mfq_tensor_backend::empty(
        {chunks, n_ctx, model.hc_mult(), model.hidden_size()},
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCPU)
            .dtype(mfq_tensor_backend::kFloat16)
            .pinned_memory(true));
    std::cout << "cpp_kl_stream_begin chunks=" << chunks
              << " ctx=" << n_ctx
              << " score_count=" << score_count
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << " layer_group=" << layer_group
              << " chunk_batch=" << chunk_batch
              << " execution_n_seq=" << execution_n_seq
              << " execution_n_batch=" << execution_n_batch
              << " execution_n_ubatch=" << execution_n_batch
              << " hidden_bytes=" << hidden_bytes << "\n";

    for (int begin = 0; begin < chunks; begin += chunk_batch) {
        const int count = std::min(chunk_batch, chunks - begin);
        auto ids = streamed_kl_ids(input, begin, count, n_ctx);
        auto x = model.embed_forward(ids);
        x = x.unsqueeze(2)
            .expand({
                count, n_ctx, model.hc_mult(), model.hidden_size()})
            .contiguous();
        hidden_cpu.narrow(0, begin, count).copy_(x, true);
        mfq_hip_synchronize();
    }
    std::cout << "cpp_kl_stream_phase=embedding completed_chunks="
              << chunks << "\n";

    for (int layer_begin = 0;
        layer_begin < model.num_hidden_layers();
         layer_begin += layer_group) {
        const int layer_end = std::min(
            layer_begin + layer_group,
            static_cast<int>(model.num_hidden_layers()));
        auto group_started = std::chrono::steady_clock::now();
        auto dsv4_state = std::make_shared<Dsv4SharedState>();
        std::vector<std::unique_ptr<Block>> blocks;
        blocks.reserve((size_t)(layer_end - layer_begin));
        for (int layer = layer_begin; layer < layer_end; ++layer) {
            std::cerr << "stream loading layer " << layer << " "
                      << model.layer_type(layer) << std::endl;
            blocks.push_back(mfq::cuda::deepseek_v4::load_block(
                *model.execution, mfq, model.config, layer,
                std::string(model.layer_type(layer)), dsv4_state));
        }
        for (int begin = 0; begin < chunks; begin += chunk_batch) {
            const int count = std::min(chunk_batch, chunks - begin);
            auto ids = streamed_kl_ids(input, begin, count, n_ctx);
            auto pos = mfq_tensor_backend::arange(
                0, n_ctx,
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kInt64));
            auto x = hidden_cpu.narrow(0, begin, count).to(mfq_tensor_backend::kCUDA);
            for (auto & block : blocks) {
                block->reset(count);
                block->set_token_ids(ids);
                x = block->forward(
                    *model.execution, x, pos, 0, mfq_nullopt, model.rope);
            }
            hidden_cpu.narrow(0, begin, count).copy_(x, true);
            mfq_hip_synchronize();
        }
        blocks.clear();
        dsv4_state.reset();
        mfq_hip_empty_cache();
        auto group_ended = std::chrono::steady_clock::now();
        std::cout << "cpp_kl_stream_layers begin=" << layer_begin
                  << " end=" << layer_end
                  << " sec="
                  << std::chrono::duration<double>(
                         group_ended - group_started).count()
                  << "\n";
    }

    std::ifstream reference(reference_path, std::ios::binary);
    if (!reference) {
        throw std::runtime_error(
            "cannot reopen KL logits file: " + reference_path);
    }
    KlEvalMetrics metrics;
    for (int begin = 0; begin < chunks; begin += chunk_batch) {
        const int count = std::min(chunk_batch, chunks - begin);
        auto x = hidden_cpu.narrow(0, begin, count).to(mfq_tensor_backend::kCUDA);
        auto y = model.finalize_hidden(x, count, n_ctx);
        auto selected =
            y.index({Slice(), Slice(first, first + score_count), Slice()});
        auto logits = model.apply_final_logit_softcap(
            model.lm_head.forward(*model.execution, selected));
        if (logits_output.is_open()) {
            auto saved = logits.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat16)
                             .contiguous();
            logits_output.write(
                reinterpret_cast<const char *>(
                    saved.data_ptr<mfq_half>()),
                static_cast<std::streamsize>(
                    saved.numel() * sizeof(mfq_half)));
            if (!logits_output) {
                throw std::runtime_error(
                    "failed while writing saved KL logits: " +
                    logits_partial.string());
            }
        }
        for (int bi = 0; bi < count; ++bi) {
            accumulate_streamed_kl_chunk(
                reference, input, begin + bi,
                logits.index({bi, Slice(), Slice()}),
                score_count, metrics);
            std::cout << "cpp_kl_chunk=" << (begin + bi + 1)
                      << " mean="
                      << (metrics.kld_sum / (double)metrics.count)
                      << " mean_kld_q_ref="
                      << (metrics.reverse_kld_sum / (double)metrics.count)
                      << " same_top="
                      << ((double)metrics.same_top /
                          (double)metrics.count)
                      << "\n";
        }
        mfq_hip_synchronize();
    }
    auto ended = std::chrono::steady_clock::now();
    if (logits_output.is_open()) {
        logits_output.close();
        if (!logits_output) {
            throw std::runtime_error(
                "failed to finalize saved KL logits: " +
                logits_partial.string());
        }
        std::filesystem::rename(logits_partial, logits_final);
    }
    std::cout << "cpp_kl_result chunks=" << chunks
              << " scored_tokens=" << metrics.count
              << " sec="
              << std::chrono::duration<double>(ended - started).count()
              << " kld="
              << (metrics.kld_sum / (double)metrics.count)
              << " mean_kld_q_ref="
              << (metrics.reverse_kld_sum / (double)metrics.count)
              << " bf16_ce="
              << (metrics.reference_ce_sum / (double)metrics.count)
              << " mfq_ce="
              << (metrics.quant_ce_sum / (double)metrics.count)
              << " bf16_ppl="
              << std::exp(
                  metrics.reference_ce_sum / (double)metrics.count)
              << " mfq_ppl="
              << std::exp(metrics.quant_ce_sum / (double)metrics.count)
              << " kld_pct_bf16="
              << (100.0 * metrics.kld_sum / metrics.reference_ce_sum)
              << " same_top="
              << ((double)metrics.same_top / (double)metrics.count)
              << " reference_format="
              << (input.reference_format == "_logit3_"
                  ? "trace_v3"
                  : (input.reference_format == "_logit2_"
                     ? "trace_v2" : "legacy"))
              << " execution=streamed_layer_groups"
              << " graph=streamed_layer_groups"
              << " n_ctx=" << n_ctx
              << " execution_n_seq=" << execution_n_seq
              << " execution_n_batch=" << execution_n_batch
              << " execution_n_ubatch=" << execution_n_batch
              << " score_count=" << score_count
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << "\n";
    write_moe_route_stats(execution.config, execution.moe_route_stats);
    return 0;
}

#define MFQ_INSTANTIATE_KL(TYPE)                                           \
    template int run_kl_eval_batched<TYPE>(                               \
        TYPE&, const std::string&, int, std::int64_t, int,                 \
        const KlReferenceContract&);                                       \
    template int run_selected_kl_eval<TYPE>(                              \
        TYPE&, const std::string&, int, KlEvaluator, std::int64_t, int,    \
        const KlReferenceContract&)

MFQ_INSTANTIATE_KL(mfq::cuda::Qwen35CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::DeepseekV41CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::NemotronHCausalLm);

#undef MFQ_INSTANTIATE_KL
