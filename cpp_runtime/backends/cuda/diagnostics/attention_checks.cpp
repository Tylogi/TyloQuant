#include "model_checks.h"

#include "models/deepseek_v4/ops.h"
#include "mfq/kernels/cuda/deepseek_v4_attention.h"
#include "mfq/kernels/cuda/deepseek_v4_hc.h"
#include "mfq/kernels/cuda/deepseek_v41.h"
#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_norm_ops.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int run_attention_decode_check(int length, int reps, int D, bool sliding, int window) {
    if (length < 1 || length > 262144) {
        throw std::runtime_error("--check-attention-decode must be in [1, 262144]");
    }
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    constexpr int B = 1;
    constexpr int Hq = 16;
    constexpr int max_parts = 256;
    if (D != 256 && D != 512) {
        throw std::runtime_error("--check-attention-head-dim must be 256 or 512");
    }
    if (sliding && D != 256) {
        throw std::runtime_error("SWA decode check currently requires head_dim 256");
    }
    const int Hk = sliding ? 8 : 2;
    const int kv_tile = D == 512 ? 32 : 64;
    const int visible_len = sliding ? std::min(length, window) : length;
    const int max_seq = sliding ? window : (length + kv_tile - 1) / kv_tile * kv_tile;
    const int parts = std::min(max_parts, std::max(1, (length + 127) / 128));
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    mfq_tensor_backend::manual_seed(20260720);
    auto q = mfq_tensor_backend::randn({B, Hq, 1, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto k = mfq_tensor_backend::randn({B, Hk, max_seq, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto v = mfq_tensor_backend::randn({B, Hk, max_seq, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto seq_len = mfq_tensor_backend::tensor({length}, cuda.dtype(mfq_tensor_backend::kInt64));
    auto partial_o = mfq_tensor_backend::empty({B * Hq, max_parts, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto partial_m = mfq_tensor_backend::empty({B * Hq, max_parts}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto partial_l = mfq_tensor_backend::empty({B * Hq, max_parts}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto qh = q.to(mfq_tensor_backend::kFloat16);
    const int mask_stride = (visible_len + kv_tile - 1) / kv_tile * kv_tile;
    const int ntiles_kv = (visible_len + kv_tile - 1) / kv_tile;
    const int64_t max_blocks = B * Hk * ntiles_kv;
    const int64_t meta_float2 = max_blocks * 8 * (2 + D / 2);
    auto mask = mfq_tensor_backend::empty({B, mask_stride}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto kv_max = mfq_tensor_backend::empty({B}, cuda.dtype(mfq_tensor_backend::kInt32));
    auto meta = mfq_tensor_backend::empty({2 * meta_float2}, cuda.dtype(mfq_tensor_backend::kFloat32));
    const double scale = 1.0 / std::sqrt((double)D);
    auto run_ref = [&]() {
        if (sliding) {
            return attention_cache_swa_planned_cuda(
                qh, k, v, seq_len, scale, window, visible_len);
        }
        return parts > 1
            ? attention_cache_decode_split_cuda(
                  qh, k, v, seq_len, scale, partial_o, partial_m, partial_l, parts)
            : attention_cache_decode_cuda(qh, k, v, seq_len, scale);
    };
    auto run_test = [&]() {
        if (sliding) {
            return mfq_attention_mma256_swa_decode_cuda(
                q, k, v, seq_len, scale, visible_len, mask, kv_max, meta);
        }
        return D == 512
            ? mfq_attention_mma512_decode_cuda(
                q, k, v, seq_len, scale, length, mask, kv_max, meta)
            : mfq_attention_mma256_decode_cuda(
                q, k, v, seq_len, scale, length, mask, kv_max, meta);
    };
    mfq_tensor_backend::Tensor ref, test;
    for (int i = 0; i < 10; ++i) {
        ref = run_ref();
        test = run_test();
    }
    mfq_cuda_synchronize();
    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = mfq_get_current_cuda_stream().stream();
        cudaEventRecord(start, stream);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return elapsed / reps;
    };
    const float ref_ms = time_ms(run_ref);
    const float test_ms = time_ms(run_test);
    ref = run_ref().to(mfq_tensor_backend::kFloat32);
    test = run_test().permute({0, 2, 1, 3}).contiguous();
    mfq_cuda_synchronize();
    auto diff = (test - ref).abs();
    std::cout << "attention_decode_check mode=" << (sliding ? "swa" : "full")
              << " head_dim=" << D << " length=" << length
              << " parts=" << parts << " ref_ms=" << ref_ms
              << " test_ms=" << test_ms
              << " speedup=" << ref_ms / test_ms << "\n";
    std::cout << "attention_decode_rel="
              << ((test - ref).norm() / ref.norm()).item<float>() << "\n";
    std::cout << "attention_decode_mean_abs=" << diff.mean().item<float>() << "\n";
    std::cout << "attention_decode_max_abs=" << diff.max().item<float>() << "\n";
    std::cout << "attention_decode_test_finite="
              << (mfq_tensor_backend::isfinite(test).all().item<bool>() ? 1 : 0) << "\n";
    return 0;
}

int run_gemma4_swa_check(int reps) {
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    constexpr int B = 1;
    constexpr int Hq = 32;
    constexpr int Hk = 16;
    constexpr int D = 256;
    const double scale = 1.0 / std::sqrt((double)D);
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    mfq_tensor_backend::manual_seed(20260721);

    struct Shape {
        int tokens;
        int window;
    };
    const Shape shapes[] = {
        {1, 1}, {17, 5}, {33, 17}, {73, 32}, {256, 128}, {1024, 1024},
    };
    double worst_rel = 0.0;
    double worst_mean_abs = 0.0;
    double worst_max_abs = 0.0;
    for (const auto shape : shapes) {
        auto q = mfq_tensor_backend::randn({B, Hq, shape.tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto k = mfq_tensor_backend::randn({B, Hk, shape.tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto v = mfq_tensor_backend::randn({B, Hk, shape.tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ref = attention_swa_cuda(
            q, k.to(mfq_tensor_backend::kFloat32), v.to(mfq_tensor_backend::kFloat32), scale, shape.window);
        auto test = mfq_attention_mma256_swa_cuda(q, k, v, scale, shape.window)
            .permute({0, 2, 1, 3}).contiguous();
        mfq_cuda_synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const double mean_abs = diff.mean().item<double>();
        const double max_abs = diff.max().item<double>();
        const bool finite = mfq_tensor_backend::isfinite(test).all().item<bool>();
        worst_rel = std::max(worst_rel, rel);
        worst_mean_abs = std::max(worst_mean_abs, mean_abs);
        worst_max_abs = std::max(worst_max_abs, max_abs);
        std::cout << "gemma4_swa_check tokens=" << shape.tokens
                  << " window=" << shape.window
                  << " rel=" << rel
                  << " mean_abs=" << mean_abs
                  << " max_abs=" << max_abs
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("Gemma4 SWA FlashAttention numerical check failed");
        }
    }

    struct FullAttentionShape {
        int tokens;
        int kv_heads;
    };
    for (const auto shape : {
            FullAttentionShape{33, 4}, FullAttentionShape{256, 4},
            FullAttentionShape{65, 2}, FullAttentionShape{256, 2}}) {
        constexpr int full_hq = 16;
        const int tokens = shape.tokens;
        const int full_hk = shape.kv_heads;
        auto q = mfq_tensor_backend::randn({B, full_hq, tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto k = mfq_tensor_backend::randn({B, full_hk, tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto v = mfq_tensor_backend::randn({B, full_hk, tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ref = attention_cuda(
            q, k.to(mfq_tensor_backend::kFloat32), v.to(mfq_tensor_backend::kFloat32), scale, true);
        auto test = mfq_attention_mma256_cuda(q, k, v, scale)
            .permute({0, 2, 1, 3}).contiguous();
        mfq_cuda_synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = mfq_tensor_backend::isfinite(test).all().item<bool>();
        std::cout << "flash256_full_check tokens=" << tokens
                  << " gqa=" << full_hq / full_hk
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("full FlashAttention numerical regression failed");
        }
    }

    for (const int tokens : {65, 256}) {
        constexpr int full_hq = 16;
        constexpr int full_hk = 2;
        constexpr int full_d = 128;
        const double full_scale = 1.0 / std::sqrt((double)full_d);
        auto q = mfq_tensor_backend::randn(
            {B, full_hq, tokens, full_d},
            cuda.dtype(mfq_tensor_backend::kFloat32));
        auto k = mfq_tensor_backend::randn(
            {B, full_hk, tokens, full_d},
            cuda.dtype(mfq_tensor_backend::kFloat16));
        auto v = mfq_tensor_backend::randn(
            {B, full_hk, tokens, full_d},
            cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ref = attention_cuda(
            q, k.to(mfq_tensor_backend::kFloat32),
            v.to(mfq_tensor_backend::kFloat32), full_scale, true);
        auto test = mfq_attention_mma128_cuda(q, k, v, full_scale)
            .permute({0, 2, 1, 3}).contiguous();
        mfq_cuda_synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = mfq_tensor_backend::isfinite(test).all().item<bool>();
        std::cout << "flash128_full_check tokens=" << tokens
                  << " gqa=" << full_hq / full_hk
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error(
                "D128 full FlashAttention numerical regression failed");
        }
    }

    for (const int tokens : {256}) {
        constexpr int full_hq = 16;
        constexpr int full_hk = 2;
        constexpr int full_d = 512;
        auto q = mfq_tensor_backend::randn({B, full_hq, tokens, full_d}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto k = mfq_tensor_backend::randn({B, full_hk, tokens, full_d}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto v = mfq_tensor_backend::randn({B, full_hk, tokens, full_d}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ref = attention_cuda(
            q, k.to(mfq_tensor_backend::kFloat32), v.to(mfq_tensor_backend::kFloat32), 1.0, true);
        auto test = mfq_attention_mma512_cuda(q, k, v, 1.0)
            .permute({0, 2, 1, 3}).contiguous();
        mfq_cuda_synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = mfq_tensor_backend::isfinite(test).all().item<bool>();
        std::cout << "gemma4_flash512_check tokens=" << tokens
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("Gemma4 D512 FlashAttention numerical check failed");
        }
    }

    constexpr int bench_tokens = 256;
    constexpr int bench_window = 128;
    auto q = mfq_tensor_backend::randn({B, Hq, bench_tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto k = mfq_tensor_backend::randn({B, Hk, bench_tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto v = mfq_tensor_backend::randn({B, Hk, bench_tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto kf = k.to(mfq_tensor_backend::kFloat32);
    auto vf = v.to(mfq_tensor_backend::kFloat32);
    auto run_ref = [&]() { return attention_swa_cuda(q, kf, vf, scale, bench_window); };
    auto run_test = [&]() {
        return mfq_attention_mma256_swa_cuda(q, k, v, scale, bench_window);
    };
    for (int i = 0; i < 10; ++i) {
        run_ref();
        run_test();
    }
    mfq_cuda_synchronize();
    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    const float ref_ms = time_ms(run_ref);
    const float test_ms = time_ms(run_test);
    std::cout << "gemma4_swa_bench tokens=" << bench_tokens
              << " window=" << bench_window
              << " generic_ms=" << ref_ms
              << " mma_attention_ms=" << test_ms
              << " speedup=" << ref_ms / test_ms << "\n";

    std::cout << "gemma4_swa_worst_rel=" << worst_rel
              << " worst_mean_abs=" << worst_mean_abs
              << " worst_max_abs=" << worst_max_abs << "\n";
    return 0;
}

int run_glm_dsa_check(int reps) {
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    mfq_tensor_backend::NoGradGuard no_grad;
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    mfq_tensor_backend::manual_seed(20260722);

    for (const auto dtype : {mfq_tensor_backend::kFloat16, mfq_tensor_backend::kFloat32}) {
        constexpr int H = 64, T = 8192, D = 64;
        auto x = mfq_tensor_backend::ones({1, H, T, D}, cuda.dtype(dtype));
        auto pos = mfq_tensor_backend::arange(T, cuda.dtype(mfq_tensor_backend::kInt64));
        auto cosine = mfq_tensor_backend::zeros({T, D / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto sine = mfq_tensor_backend::ones_like(cosine);
        auto output = glm_interleaved_rope_cuda(x, pos, cosine, sine, D)
            .to(mfq_tensor_backend::kFloat32).cpu();
        const auto* values = output.data_ptr<float>();
        for (int64_t i = 0; i < output.numel(); ++i)
            if (values[i] != (i % 2 ? 1.f : -1.f))
                throw std::runtime_error("GLM interleaved RoPE incomplete 8192-token coverage at " +
                    std::to_string(i));
        std::cout << "glm_rope_coverage tokens=8192 elements=" << output.numel()
                  << " dtype=" << (dtype == mfq_tensor_backend::kFloat16 ? "FP16" : "FP32") << " PASS\n";
    }

    {
        constexpr int B = 2, H = 3, T = 5, D = 128, RD = 64;
        auto x = mfq_tensor_backend::randn({B, H, T, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto pos = mfq_tensor_backend::arange(7, 7 + T, cuda.dtype(mfq_tensor_backend::kInt64));
        auto freq = mfq_tensor_backend::pow(
            mfq_tensor_backend::full({RD / 2}, 8000000.0, cuda.dtype(mfq_tensor_backend::kFloat32)),
            -mfq_tensor_backend::arange(0, RD, 2, cuda.dtype(mfq_tensor_backend::kFloat32)) / (double)RD);
        auto angle = mfq_tensor_backend::arange(32, cuda.dtype(mfq_tensor_backend::kFloat32)).unsqueeze(1) * freq.unsqueeze(0);
        auto cos = mfq_tensor_backend::cos(angle).contiguous();
        auto sin = mfq_tensor_backend::sin(angle).contiguous();
        auto test = glm_interleaved_rope_cuda(x, pos, cos, sin, RD);
        auto paired = x.index({Slice(), Slice(), Slice(), Slice(0, RD)})
            .reshape({B, H, T, RD / 2, 2});
        auto c = cos.index_select(0, pos).reshape({1, 1, T, RD / 2});
        auto s = sin.index_select(0, pos).reshape({1, 1, T, RD / 2});
        auto rotated = mfq_tensor_backend::stack({
            paired.select(-1, 0) * c - paired.select(-1, 1) * s,
            paired.select(-1, 1) * c + paired.select(-1, 0) * s,
        }, -1).reshape({B, H, T, RD});
        auto ref = mfq_tensor_backend::cat({
            rotated, x.index({Slice(), Slice(), Slice(), Slice(RD, D)})}, -1);
        const double max_abs = (test - ref).abs().max().item<double>();
        std::cout << "glm_rope_max_abs=" << max_abs << "\n";
        if (max_abs > 2e-6) throw std::runtime_error("GLM interleaved RoPE numerical check failed");
    }

    {
        constexpr int ROWS = 17, D = 128;
        auto x = mfq_tensor_backend::randn({ROWS, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto weight = mfq_tensor_backend::randn({D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto bias = mfq_tensor_backend::randn({D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto test = glm_dsa_indexer_layer_norm_cuda(x, weight, bias, 1e-5);
        auto ref = mfq_tensor_backend::layer_norm(
            x.to(mfq_tensor_backend::kFloat32), {D}, weight, bias, 1e-5).to(mfq_tensor_backend::kFloat16);
        const double max_abs = (test.to(mfq_tensor_backend::kFloat32) - ref.to(mfq_tensor_backend::kFloat32))
            .abs().max().item<double>();
        std::cout << "glm_indexer_layer_norm_max_abs=" << max_abs << "\n";
        if (max_abs > 0.004) {
            throw std::runtime_error("GLM indexer LayerNorm numerical check failed");
        }
    }

    {
        constexpr int ROWS = 3, EXPERTS = 256, TOPK = 8;
        auto logits = mfq_tensor_backend::randn({ROWS, EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto bias = mfq_tensor_backend::randn({EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto selected = moe_topk_cuda(
            logits, TOPK, true, false, true, false, bias, 1e-20, 2.5);
        auto sigmoid = mfq_tensor_backend::sigmoid(logits);
        auto expected_topk = mfq_tensor_backend::topk(
            sigmoid + bias.unsqueeze(0), TOPK, 1, true, true);
        auto expected_ids = std::get<1>(expected_topk);
        auto expected_weights = sigmoid.gather(1, expected_ids);
        expected_weights = expected_weights /
            expected_weights.sum(1, true).clamp_min(1e-20) * 2.5;
        const bool ids_equal = selected[0].to(mfq_tensor_backend::kInt64).equal(expected_ids);
        const double max_abs = (selected[1] - expected_weights)
            .abs().max().item<double>();
        std::cout << "glm_moe_route_ids_equal=" << (ids_equal ? 1 : 0)
                  << " max_abs=" << max_abs << "\n";
        if (!ids_equal || max_abs > 2e-6) {
            throw std::runtime_error("GLM MoE routing numerical check failed");
        }
    }

    {
        constexpr int B = 1, M = 3, K = 2112, H = 32, D = 128;
        auto q = mfq_tensor_backend::randn({B, M, H, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto k = mfq_tensor_backend::randn({B, K, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto weights = mfq_tensor_backend::randn({B, M, H}, cuda.dtype(mfq_tensor_backend::kFloat32));
        const int offset = K - M;
        auto test = glm_dsa_indexer_scores_cuda(q, k, weights, offset, K);
        auto heads = mfq_tensor_backend::einsum(
            "bmhd,bkd->bmhk", {q.to(mfq_tensor_backend::kFloat32), k.to(mfq_tensor_backend::kFloat32)}) /
            std::sqrt(128.0);
        auto ref = (mfq_tensor_backend::relu(heads) * weights.unsqueeze(-1)).sum(2) / std::sqrt(32.0);
        auto key_pos = mfq_tensor_backend::arange(K, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({1, 1, K});
        auto query_pos = mfq_tensor_backend::arange(offset, offset + M, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({1, M, 1});
        ref = ref.masked_fill(key_pos > query_pos, -std::numeric_limits<float>::infinity());
        auto finite = mfq_tensor_backend::isfinite(ref);
        auto diff = mfq_tensor_backend::where(finite, (test - ref).abs(), mfq_tensor_backend::zeros_like(ref));
        const double rel = mfq_tensor_backend::where(finite, test - ref, mfq_tensor_backend::zeros_like(ref)).norm().item<double>() /
            std::max(mfq_tensor_backend::where(finite, ref, mfq_tensor_backend::zeros_like(ref)).norm().item<double>(), 1e-30);
        const double max_abs = diff.max().item<double>();
        std::cout << "glm_indexer_rel=" << rel << " max_abs=" << max_abs << "\n";
        if (!mfq_tensor_backend::isneginf(test.masked_select(~finite)).all().item<bool>() || rel > 0.003) {
            throw std::runtime_error("GLM indexer score numerical check failed");
        }
    }

    {
        constexpr int B = 1, M = 1, VISIBLE = 2112, PLANNED = 2176;
        constexpr int H = 32, D = 128;
        auto q = mfq_tensor_backend::randn({B, M, H, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto k = mfq_tensor_backend::randn({B, PLANNED, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto weights = mfq_tensor_backend::randn({B, M, H}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto seq_len = mfq_tensor_backend::tensor({VISIBLE}, cuda.dtype(mfq_tensor_backend::kInt64));
        auto test = glm_dsa_indexer_scores_decode_cuda(
            q, k, weights, seq_len, PLANNED);
        auto ref = (mfq_tensor_backend::relu(mfq_tensor_backend::einsum(
            "bmhd,bkd->bmhk",
            {q.to(mfq_tensor_backend::kFloat32), k.to(mfq_tensor_backend::kFloat32)}) / std::sqrt(128.0)) *
            weights.unsqueeze(-1)).sum(2) / std::sqrt(32.0);
        ref.index({Slice(), Slice(), Slice(VISIBLE, PLANNED)})
            .fill_(-std::numeric_limits<float>::infinity());
        auto finite = mfq_tensor_backend::isfinite(ref);
        const double rel = mfq_tensor_backend::where(
            finite, test - ref, mfq_tensor_backend::zeros_like(ref)).norm().item<double>() /
            std::max(mfq_tensor_backend::where(
                finite, ref, mfq_tensor_backend::zeros_like(ref)).norm().item<double>(), 1e-30);
        const bool future_inf = mfq_tensor_backend::isneginf(
            test.index({Slice(), Slice(), Slice(VISIBLE, PLANNED)})).all().item<bool>();
        std::cout << "glm_indexer_decode_rel=" << rel
                  << " future_inf=" << (future_inf ? 1 : 0) << "\n";
        if (rel > 0.003 || !future_inf) {
            throw std::runtime_error("GLM decode indexer numerical check failed");
        }
    }

    {
        constexpr int B = 1, H = 64, T = 33, DQ = 576, DV = 512;
        const double scale = 1.0 / std::sqrt(256.0);
        auto q = mfq_tensor_backend::randn({B, H, T, DQ}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto kv = mfq_tensor_backend::randn({B, 1, T, DQ}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto kv_cache = mfq_tensor_backend::zeros({B, 1, 64, DQ}, cuda.dtype(mfq_tensor_backend::kFloat16));
        kv_cache.index({Slice(), Slice(), Slice(0, T), Slice()}).copy_(kv);
        auto mask = mfq_tensor_backend::empty({T, 64}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto kv_max = mfq_tensor_backend::empty({B * ((T + 3) / 4)}, cuda.dtype(mfq_tensor_backend::kInt32));
        auto meta = mfq_tensor_backend::empty({8 * 1024 * 1024}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto test = attention_glm_mla576_cached_cuda(
            q, kv_cache, T, mask, kv_max, meta, scale);
        auto q_ref = q.to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
        auto k_ref = kv.to(mfq_tensor_backend::kFloat32).expand({B, H, T, DQ});
        auto scores = mfq_tensor_backend::matmul(q_ref, k_ref.transpose(-1, -2)) * scale;
        auto causal = mfq_tensor_backend::ones({T, T}, cuda.dtype(mfq_tensor_backend::kBool)).tril();
        scores = scores.masked_fill(~causal, -std::numeric_limits<float>::infinity());
        auto ref = mfq_tensor_backend::matmul(
            mfq_tensor_backend::softmax(scores, -1),
            k_ref.index({Slice(), Slice(), Slice(), Slice(0, DV)}))
            .permute({0, 2, 1, 3}).contiguous();
        const double rel = (test - ref).norm().item<double>() / ref.norm().item<double>();
        const double max_abs = (test - ref).abs().max().item<double>();
        std::cout << "glm_dense_mla_rel=" << rel << " max_abs=" << max_abs << "\n";
        if (!mfq_tensor_backend::isfinite(test).all().item<bool>() || rel > 0.02) {
            throw std::runtime_error("GLM dense MLA numerical check failed");
        }
    }

    {
        constexpr int B = 1, H = 64, VISIBLE = 33, PLANNED = 64;
        constexpr int DQ = 576, DV = 512;
        const double scale = 1.0 / std::sqrt(256.0);
        auto q = mfq_tensor_backend::randn({B, H, 1, DQ}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto kv = mfq_tensor_backend::randn({B, 1, PLANNED, DQ}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto seq_len = mfq_tensor_backend::tensor({VISIBLE}, cuda.dtype(mfq_tensor_backend::kInt64));
        auto mask = mfq_tensor_backend::empty({B, PLANNED}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto kv_max = mfq_tensor_backend::empty({B}, cuda.dtype(mfq_tensor_backend::kInt32));
        auto meta = mfq_tensor_backend::empty({8 * 1024 * 1024}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto test = attention_glm_mla576_decode_cuda(
            q, kv, seq_len, scale, PLANNED, mask, kv_max, meta);
        auto selected = kv.index({Slice(), Slice(), Slice(0, VISIBLE), Slice()})
            .to(mfq_tensor_backend::kFloat32).expand({B, H, VISIBLE, DQ});
        auto ref = mfq_tensor_backend::matmul(
            mfq_tensor_backend::softmax(mfq_tensor_backend::matmul(q, selected.transpose(-1, -2)) * scale, -1),
            selected.index({Slice(), Slice(), Slice(), Slice(0, DV)}))
            .permute({0, 2, 1, 3}).contiguous();
        const double rel = (test - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "glm_dense_decode_rel=" << rel << "\n";
        if (!mfq_tensor_backend::isfinite(test).all().item<bool>() || rel > 0.02) {
            throw std::runtime_error("GLM dense decode numerical check failed");
        }
    }

    constexpr int B = 1, H = 64, M = 3, K = 2112, TOPK = 2048, DQ = 576, DV = 512;
    const double scale = 1.0 / std::sqrt(256.0);
    auto q = mfq_tensor_backend::randn({B, H, M, DQ}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto kv = mfq_tensor_backend::randn({B, K, DQ}, cuda.dtype(mfq_tensor_backend::kFloat16));
    std::vector<mfq_tensor_backend::Tensor> rows;
    rows.reserve(M);
    for (int row = 0; row < M; ++row) {
        rows.push_back(mfq_tensor_backend::randperm(K - M + row + 1, cuda.dtype(mfq_tensor_backend::kInt64))
            .narrow(0, 0, TOPK).to(mfq_tensor_backend::kInt32));
    }
    auto indices = mfq_tensor_backend::stack(rows, 0).unsqueeze(0).contiguous();
    auto meta = mfq_tensor_backend::empty({8 * 1024 * 1024}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto run_sparse = [&]() {
        return attention_glm_mla_sparse_cuda(q, kv, indices, meta, scale);
    };
    auto test = run_sparse();
    std::vector<mfq_tensor_backend::Tensor> refs;
    refs.reserve(M);
    auto q_ref = q.to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
    for (int row = 0; row < M; ++row) {
        auto idx = indices.index({0, row}).to(mfq_tensor_backend::kInt64);
        auto selected = kv.index_select(1, idx).index({0}).to(mfq_tensor_backend::kFloat32);
        auto score = mfq_tensor_backend::matmul(q_ref.index({0, Slice(), row}), selected.transpose(0, 1)) * scale;
        refs.push_back(mfq_tensor_backend::matmul(
            mfq_tensor_backend::softmax(score, -1), selected.index({Slice(), Slice(0, DV)})));
    }
    auto ref = mfq_tensor_backend::stack(refs, 0).unsqueeze(0);
    mfq_cuda_synchronize();
    const double rel = (test - ref).norm().item<double>() / ref.norm().item<double>();
    const double max_abs = (test - ref).abs().max().item<double>();
    std::cout << "glm_sparse_mla_rel=" << rel << " max_abs=" << max_abs << "\n";
    if (!mfq_tensor_backend::isfinite(test).all().item<bool>() || rel > 0.02) {
        throw std::runtime_error("GLM sparse MLA numerical check failed");
    }

    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    for (int i = 0; i < 5; ++i) run_sparse();
    const float sparse_ms = time_ms(run_sparse);
    std::cout << "glm_sparse_mla_m=" << M << " k=" << K
              << " topk=" << TOPK << " cuda_ms=" << sparse_ms << "\n";
    return 0;
}

int run_dsv4_hc_check(int reps) {
    if (reps < 1) {
        throw std::runtime_error("--check-attention-reps must be positive");
    }
    mfq_tensor_backend::NoGradGuard no_grad;
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    constexpr int64_t hc = 4;
    constexpr int64_t hidden = 4096;
    constexpr int64_t mix_width = 24;
    constexpr double eps = 1e-6;

    auto x_sequence = mfq_tensor_backend::arange(
        hc * hidden, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto x = (
        (x_sequence.remainder(251) - 125.0) / 64.0 +
        0.125 * mfq_tensor_backend::sin(x_sequence * 0.015625))
        .to(mfq_tensor_backend::kFloat16)
        .reshape({1, 1, hc, hidden})
        .contiguous();
    auto function_sequence = mfq_tensor_backend::arange(
        mix_width * hc * hidden, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto function = (
        0.003 * mfq_tensor_backend::sin(function_sequence * 0.001953125) +
        0.001 * mfq_tensor_backend::cos(function_sequence * 0.0078125))
        .reshape({mix_width, hc * hidden})
        .contiguous();
    auto scale = (
        0.7 + 0.2 * mfq_tensor_backend::arange(3, cuda.dtype(mfq_tensor_backend::kFloat32)))
        .contiguous();
    auto base = (
        0.1 * mfq_tensor_backend::sin(
            mfq_tensor_backend::arange(
                mix_width, cuda.dtype(mfq_tensor_backend::kFloat32)) * 0.3125))
        .contiguous();
    auto flat = x.flatten(2).to(mfq_tensor_backend::kFloat32);
    auto inverse_rms = mfq_tensor_backend::rsqrt(
        flat.square().mean(-1, true) + eps);
    auto mixes = (
        mfq_tensor_backend::matmul(flat, function.transpose(0, 1)) * inverse_rms)
        .contiguous();

    auto reference_split = dsv4_hc_split_sinkhorn(
        mixes, scale, base, hc, 20, eps);
    auto reference_reduced = (
        reference_split.at(0).unsqueeze(-1) *
        flat.reshape({1, 1, hc, hidden}))
        .sum(2).to(mfq_tensor_backend::kFloat16).contiguous();
    auto candidate = dsv4_hc_pre_cuda(
        x, mixes, scale, base, 20, eps);

    auto direct = x.index(
        {Slice(), Slice(), 0, Slice()}).contiguous();
    auto reference_post = (
        reference_split.at(1).unsqueeze(-1) *
            direct.unsqueeze(-2) +
        (reference_split.at(2).unsqueeze(-1) *
            x.to(mfq_tensor_backend::kFloat32).unsqueeze(-2)).sum(2))
        .to(mfq_tensor_backend::kFloat16).contiguous();
    auto candidate_post = dsv4_hc_post_cuda(
        direct, x, candidate.at(1), candidate.at(2));
    mfq_cuda_synchronize();

    auto compare = [](const char * name,
                      mfq_tensor_backend::Tensor reference,
                      mfq_tensor_backend::Tensor value) {
        auto reference_f32 = reference.to(mfq_tensor_backend::kFloat32);
        auto value_f32 = value.to(mfq_tensor_backend::kFloat32);
        auto diff = value_f32 - reference_f32;
        const double norm = reference_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "dsv4_hc_ab tensor=" << name
                  << " equal=" << (value.equal(reference) ? 1 : 0)
                  << " differing="
                  << value.ne(reference).sum().item<int64_t>()
                  << " rel_l2="
                  << (norm == 0.0
                      ? 0.0
                      : diff.norm().item<double>() / norm)
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    };
    compare("reduced", reference_reduced, candidate.at(0));
    compare("post", reference_split.at(1), candidate.at(1));
    compare("combination", reference_split.at(2), candidate.at(2));
    compare("hc_post", reference_post, candidate_post);

    auto time_ms = [&](auto && fn) {
        for (int warmup = 0; warmup < 10; ++warmup) fn();
        mfq_cuda_synchronize();
        cudaEvent_t begin, end;
        MFQ_CUDA_CHECK(cudaEventCreate(&begin));
        MFQ_CUDA_CHECK(cudaEventCreate(&end));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(begin, stream));
        for (int iteration = 0; iteration < reps; ++iteration) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(end, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
        MFQ_CUDA_CHECK(cudaEventDestroy(begin));
        MFQ_CUDA_CHECK(cudaEventDestroy(end));
        return elapsed / reps;
    };
    const float reference_pre_ms = time_ms([&]() {
        auto split = dsv4_hc_split_sinkhorn(
            mixes, scale, base, hc, 20, eps);
        return (
            split.at(0).unsqueeze(-1) *
            flat.reshape({1, 1, hc, hidden}))
            .sum(2).to(mfq_tensor_backend::kFloat16).contiguous();
    });
    const float candidate_pre_ms = time_ms([&]() {
        return dsv4_hc_pre_cuda(
            x, mixes, scale, base, 20, eps);
    });
    const float reference_post_ms = time_ms([&]() {
        return (
            reference_split.at(1).unsqueeze(-1) *
                direct.unsqueeze(-2) +
            (reference_split.at(2).unsqueeze(-1) *
                x.to(mfq_tensor_backend::kFloat32).unsqueeze(-2)).sum(2))
            .to(mfq_tensor_backend::kFloat16).contiguous();
    });
    const float candidate_post_ms = time_ms([&]() {
        return dsv4_hc_post_cuda(
            direct, x, candidate.at(1), candidate.at(2));
    });
    std::cout << std::fixed << std::setprecision(9)
              << "dsv4_hc_timing"
              << " reference_pre_ms=" << reference_pre_ms
              << " candidate_pre_ms=" << candidate_pre_ms
              << " pre_speedup=" << reference_pre_ms / candidate_pre_ms
              << " reference_post_ms=" << reference_post_ms
              << " candidate_post_ms=" << candidate_post_ms
              << " post_speedup=" << reference_post_ms / candidate_post_ms
              << "\n";
    return 0;
}

int run_dsv4_attention_check(int reps) {
    if (reps < 1) {
        throw std::runtime_error("--check-attention-reps must be positive");
    }
    mfq_tensor_backend::NoGradGuard no_grad;
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    mfq_tensor_backend::manual_seed(20260723);

    {
        constexpr int EXPERTS = 256, TOPK = 6;
        auto logits = mfq_tensor_backend::randn({1, EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32)) * 3.0;
        auto bias = mfq_tensor_backend::randn({EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32)) * 0.05;
        auto selected = moe_topk_cuda(
            logits, TOPK, false, true, true, false, bias, 1e-20, 1.5);
        auto transformed = mfq_tensor_backend::sqrt(mfq_tensor_backend::where(
            logits > 20.0, logits, mfq_tensor_backend::log1p(mfq_tensor_backend::exp(logits))));
        auto expected_topk = mfq_tensor_backend::topk(
            transformed + bias.unsqueeze(0), TOPK, 1, true, true);
        auto expected_ids = std::get<1>(expected_topk);
        auto expected_weights = transformed.gather(1, expected_ids);
        expected_weights = expected_weights /
            expected_weights.sum(1, true).clamp_min(1e-20) * 1.5;
        const bool ids_equal =
            selected.at(0).to(mfq_tensor_backend::kInt64).equal(expected_ids);
        const double max_abs = (selected.at(1) - expected_weights)
            .abs().max().item<double>();

        auto hash_ids = mfq_tensor_backend::randint(
            0, EXPERTS, {7, TOPK}, cuda.dtype(mfq_tensor_backend::kInt32));
        auto hash_logits =
            mfq_tensor_backend::randn({7, EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32)) * 3.0;
        auto hash_weights = moe_sqrtsoftplus_weights_cuda(
            hash_logits, hash_ids, 1e-20, 1.5);
        auto hash_transformed = mfq_tensor_backend::sqrt(mfq_tensor_backend::where(
            hash_logits > 20.0, hash_logits,
            mfq_tensor_backend::log1p(mfq_tensor_backend::exp(hash_logits))));
        auto expected_hash = hash_transformed.gather(
            1, hash_ids.to(mfq_tensor_backend::kInt64));
        expected_hash = expected_hash /
            expected_hash.sum(1, true).clamp_min(1e-20) * 1.5;
        const double hash_max_abs = (hash_weights - expected_hash)
            .abs().max().item<double>();
        std::cout << "dsv4_moe_route_ids_equal=" << (ids_equal ? 1 : 0)
                  << " max_abs=" << max_abs
                  << " hash_max_abs=" << hash_max_abs << "\n";
        if (!ids_equal || max_abs > 2e-6 || hash_max_abs > 2e-6) {
            throw std::runtime_error(
                "DeepSeek V4 MoE routing numerical check failed");
        }
    }

    constexpr int B = 1;
    constexpr int D = 512;
    constexpr int RD = 64;
    constexpr double eps = 1e-6;
    auto norm = mfq_tensor_backend::randn({D}, cuda.dtype(mfq_tensor_backend::kFloat32));

    {
        constexpr int W = 2, R = 128;
        auto kv = mfq_tensor_backend::randn({B, W, R, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto gate = mfq_tensor_backend::randn({B, W, R, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ape = mfq_tensor_backend::randn({R, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto empty = mfq_tensor_backend::empty({0}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto positions = mfq_tensor_backend::arange(W, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({B, W});
        auto cos = mfq_tensor_backend::ones({W + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto sin = mfq_tensor_backend::zeros_like(cos);
        auto test = dsv4_compress_cuda(
            kv, gate, ape, norm, empty, empty, positions, cos, sin,
            R, false, 1, eps);
        auto score = gate.to(mfq_tensor_backend::kFloat32) +
            ape.reshape({1, 1, R, D});
        auto pooled = (kv.to(mfq_tensor_backend::kFloat32) *
            mfq_tensor_backend::softmax(score, 2)).sum(2);
        auto ref = pooled * mfq_tensor_backend::rsqrt(
            pooled.square().mean(-1, true) + eps) * norm;
        ref = ref.to(mfq_tensor_backend::kBFloat16).to(mfq_tensor_backend::kFloat32);
        auto ref_nope = ref.slice(-1, 0, D - RD)
            .reshape({-1, (D - RD) / 64, 64});
        auto fp8_scale = ref_nope.abs().amax(-1, true)
            .clamp_min(1e-4) / 448.0;
        ref_nope = (ref_nope / fp8_scale)
            .clamp(-448.0, 448.0)
            .to(mfq_float8_e4m3fn)
            .to(mfq_tensor_backend::kFloat32) * fp8_scale;
        ref.slice(-1, 0, D - RD).copy_(
            ref_nope.reshape({B, W, D - RD}));
        ref = ref.to(mfq_tensor_backend::kBFloat16)
            .to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
        const double rel =
            (test.to(mfq_tensor_backend::kFloat32) - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "dsv4_hca_compressor_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 ratio-128 compressor numerical check failed");
        }
    }

    {
        constexpr int W = 3, R = 4, OD = 2 * D;
        auto kv = mfq_tensor_backend::randn({B, W, R, OD}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto gate = mfq_tensor_backend::randn({B, W, R, OD}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto prev_kv = mfq_tensor_backend::randn({B, R, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto prev_gate = mfq_tensor_backend::randn({B, R, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ape = mfq_tensor_backend::randn({R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto positions = mfq_tensor_backend::arange(W, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({B, W});
        auto cos = mfq_tensor_backend::ones({W + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto sin = mfq_tensor_backend::zeros_like(cos);
        auto test = dsv4_compress_cuda(
            kv, gate, ape, norm, prev_kv, prev_gate, positions,
            cos, sin, R, true, 0, eps);
        std::vector<mfq_tensor_backend::Tensor> ref_rows;
        for (int w = 0; w < W; ++w) {
            auto left_kv = w == 0
                ? prev_kv
                : kv.index({Slice(), w - 1, Slice(), Slice(0, D)});
            auto left_gate = w == 0
                ? prev_gate
                : gate.index({Slice(), w - 1, Slice(), Slice(0, D)});
            auto right_kv =
                kv.index({Slice(), w, Slice(), Slice(D, OD)});
            auto right_gate =
                gate.index({Slice(), w, Slice(), Slice(D, OD)});
            auto values = mfq_tensor_backend::cat({left_kv, right_kv}, 1)
                .to(mfq_tensor_backend::kFloat32);
            auto score = mfq_tensor_backend::cat({
                left_gate.to(mfq_tensor_backend::kFloat32) +
                    ape.index({Slice(), Slice(0, D)}).unsqueeze(0),
                right_gate.to(mfq_tensor_backend::kFloat32) +
                    ape.index({Slice(), Slice(D, OD)}).unsqueeze(0)}, 1);
            auto pooled = (values * mfq_tensor_backend::softmax(score, 1)).sum(1);
            ref_rows.push_back(
                pooled * mfq_tensor_backend::rsqrt(
                    pooled.square().mean(-1, true) + eps) * norm);
        }
        auto ref = mfq_tensor_backend::stack(ref_rows, 1)
            .to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
        const double rel =
            (test.to(mfq_tensor_backend::kFloat32) - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "dsv4_csa_overlap_compressor_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 ratio-4 overlap compressor numerical check failed");
        }
    }

    {
        constexpr int T = 12, R = 4, W = T / R, OD = 2 * D;
        auto kv = mfq_tensor_backend::randn({B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto gate = mfq_tensor_backend::randn({B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto ape = mfq_tensor_backend::randn({R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto empty = mfq_tensor_backend::empty({0}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto positions = mfq_tensor_backend::arange(W, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({B, W});
        auto cos = mfq_tensor_backend::ones({W + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto sin = mfq_tensor_backend::zeros_like(cos);
        auto batch = dsv4_compress_cuda(
            kv.reshape({B, W, R, OD}).contiguous(),
            gate.reshape({B, W, R, OD}).contiguous(),
            ape, norm, empty, empty, positions, cos, sin,
            R, true, 1, eps);
        auto state_kv = mfq_tensor_backend::zeros(
            {B, R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto state_gate = mfq_tensor_backend::zeros_like(state_kv);
        auto prev_kv = mfq_tensor_backend::zeros(
            {B, R, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto prev_gate = mfq_tensor_backend::zeros_like(prev_kv);
        auto pool = mfq_tensor_backend::zeros(
            {B, W, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto seq_len = mfq_tensor_backend::zeros({B}, cuda.dtype(mfq_tensor_backend::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            dsv4_decode_pool_update_cuda(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                ape, norm, state_kv, state_gate, prev_kv, prev_gate,
                pool, seq_len, cos, sin, R, true, 1, eps);
        }
        const double rel =
            (pool.to(mfq_tensor_backend::kFloat32) - batch.to(mfq_tensor_backend::kFloat32))
                .norm().item<double>() /
            std::max(batch.to(mfq_tensor_backend::kFloat32).norm().item<double>(), 1e-30);
        std::cout << "dsv4_decode_pool_state_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 decode compressor state check failed");
        }
    }

    {
        constexpr int ID = 128, T = 12, R = 4, W = T / R;
        constexpr int OD = 2 * ID;
        auto index_norm = mfq_tensor_backend::randn(
            {ID}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto kv = mfq_tensor_backend::randn(
            {B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto gate = mfq_tensor_backend::randn(
            {B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto ape = mfq_tensor_backend::randn(
            {R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto empty = mfq_tensor_backend::empty({0}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto positions = mfq_tensor_backend::arange(
            W, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({B, W});
        auto cos = mfq_tensor_backend::ones(
            {W + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto sin = mfq_tensor_backend::zeros_like(cos);
        auto batch = dsv4_compress_cuda(
            kv.reshape({B, W, R, OD}).contiguous(),
            gate.reshape({B, W, R, OD}).contiguous(),
            ape, index_norm, empty, empty, positions, cos, sin,
            R, true, 2, eps);
        auto state_kv = mfq_tensor_backend::zeros(
            {B, R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto state_gate = mfq_tensor_backend::zeros_like(state_kv);
        auto prev_kv = mfq_tensor_backend::zeros(
            {B, R, ID}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto prev_gate = mfq_tensor_backend::zeros_like(prev_kv);
        auto pool = mfq_tensor_backend::zeros(
            {B, W, ID}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto seq_len = mfq_tensor_backend::zeros(
            {B}, cuda.dtype(mfq_tensor_backend::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            dsv4_decode_pool_update_cuda(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                ape, index_norm, state_kv, state_gate,
                prev_kv, prev_gate, pool, seq_len, cos, sin,
                R, true, 2, eps);
        }
        const double rel =
            (pool.to(mfq_tensor_backend::kFloat32) - batch.to(mfq_tensor_backend::kFloat32))
                .norm().item<double>() /
            std::max(
                batch.to(mfq_tensor_backend::kFloat32).norm().item<double>(),
                1e-30);
        std::cout << "dsv4_indexer_pool_state_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 indexer compressor state check failed");
        }
    }

    {
        constexpr int ID = 128, T = 14, R = 4, OD = 2 * ID;
        auto index_norm = mfq_tensor_backend::randn(
            {ID}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto kv = mfq_tensor_backend::randn(
            {B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto gate = mfq_tensor_backend::randn(
            {B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        Dsv4RopeTable rope;
        rope.cos = mfq_tensor_backend::ones(
            {T + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        rope.sin = mfq_tensor_backend::zeros_like(rope.cos);
        rope.negative_sin = -rope.sin;

        Dsv4PoolState batched;
        batched.ratio = R;
        batched.head_dim = ID;
        batched.overlap = true;
        batched.cache_quant_mode = 2;
        batched.ape = mfq_tensor_backend::randn(
            {R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        batched.norm = index_norm;
        batched.reset(B, T);

        Dsv4PoolState decoded;
        decoded.ratio = batched.ratio;
        decoded.head_dim = batched.head_dim;
        decoded.overlap = batched.overlap;
        decoded.cache_quant_mode = batched.cache_quant_mode;
        decoded.ape = batched.ape;
        decoded.norm = batched.norm;
        decoded.reset(B, T);

        const int64_t windows = batched.prefill(kv, gate, rope);
        auto seq_len = mfq_tensor_backend::zeros(
            {B}, cuda.dtype(mfq_tensor_backend::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            decoded.update(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                seq_len, rope);
        }
        const auto pool_width = T / R;
        auto reference_pool = decoded.pool.narrow(1, 0, pool_width);
        auto candidate_pool = batched.pool.narrow(1, 0, pool_width);
        const double pool_rel =
            (candidate_pool.to(mfq_tensor_backend::kFloat32) -
             reference_pool.to(mfq_tensor_backend::kFloat32)).norm().item<double>() /
            std::max(
                reference_pool.to(mfq_tensor_backend::kFloat32).norm().item<double>(),
                1e-30);
        const double state_rel =
            (batched.state_kv - decoded.state_kv)
                .norm().item<double>() /
            std::max(decoded.state_kv.norm().item<double>(), 1e-30);
        const double gate_rel =
            (batched.state_gate - decoded.state_gate)
                .norm().item<double>() /
            std::max(decoded.state_gate.norm().item<double>(), 1e-30);
        const double previous_rel =
            (batched.previous_kv - decoded.previous_kv)
                .norm().item<double>() /
            std::max(decoded.previous_kv.norm().item<double>(), 1e-30);
        const double previous_gate_rel =
            (batched.previous_gate - decoded.previous_gate)
                .norm().item<double>() /
            std::max(decoded.previous_gate.norm().item<double>(), 1e-30);
        std::cout << "dsv4_pool_prefill_windows=" << windows
                  << " pool_rel=" << pool_rel
                  << " state_rel=" << state_rel
                  << " gate_rel=" << gate_rel
                  << " previous_rel=" << previous_rel
                  << " previous_gate_rel=" << previous_gate_rel << "\n";
        if (windows != pool_width || pool_rel > 0.002 ||
            state_rel > 1e-7 || gate_rel > 1e-7 ||
            previous_rel > 1e-7 || previous_gate_rel > 1e-7) {
            throw std::runtime_error(
                "DeepSeek V4 batched compressor prefill state check failed");
        }
    }

    {
        constexpr int ROWS = 9, WIDTH = 128;
        auto input = mfq_tensor_backend::randn(
            {ROWS, WIDTH}, cuda.dtype(mfq_tensor_backend::kFloat16)) * 2.0;
        auto test = dsv4_fp4_sim_cuda(input.contiguous());
        auto grouped = input.to(mfq_tensor_backend::kFloat32)
            .reshape({ROWS, WIDTH / 32, 32});
        auto scale = mfq_tensor_backend::exp2(mfq_tensor_backend::ceil(mfq_tensor_backend::log2(
            grouped.abs().amax(-1, true)
                .clamp_min(6.0 * std::ldexp(1.0, -126)) / 6.0)));
        auto normalized = (grouped / scale).clamp(-6.0, 6.0);
        auto magnitude = normalized.abs();
        auto quantized = mfq_tensor_backend::where(
            magnitude <= 0.25, mfq_tensor_backend::zeros_like(magnitude),
            mfq_tensor_backend::where(
                magnitude < 0.75, mfq_tensor_backend::full_like(magnitude, 0.5),
                mfq_tensor_backend::where(
                    magnitude <= 1.25, mfq_tensor_backend::ones_like(magnitude),
                    mfq_tensor_backend::where(
                        magnitude < 1.75,
                        mfq_tensor_backend::full_like(magnitude, 1.5),
                        mfq_tensor_backend::where(
                            magnitude <= 2.5,
                            mfq_tensor_backend::full_like(magnitude, 2.0),
                            mfq_tensor_backend::where(
                                magnitude < 3.5,
                                mfq_tensor_backend::full_like(magnitude, 3.0),
                                mfq_tensor_backend::where(
                                    magnitude <= 5.0,
                                    mfq_tensor_backend::full_like(magnitude, 4.0),
                                    mfq_tensor_backend::full_like(
                                        magnitude, 6.0))))))));
        quantized = mfq_tensor_backend::where(
            normalized < 0, -quantized, quantized);
        auto reference = (quantized * scale)
            .reshape({ROWS, WIDTH}).to(mfq_tensor_backend::kFloat16);
        const double max_abs = (test - reference)
            .abs().max().item<double>();
        std::cout << "dsv4_fp4_sim_max_abs=" << max_abs << "\n";
        if (max_abs != 0.0) {
            throw std::runtime_error(
                "DeepSeek V4 FP4 activation simulation check failed");
        }
    }

    {
        constexpr int M = 3, K = 768, H = 64, ID = 128;
        auto q = mfq_tensor_backend::randn({B, M, H, ID}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto k = mfq_tensor_backend::randn({B, K, ID}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto weights = mfq_tensor_backend::randn({B, M, H}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto test = dsv4_indexer_scores_cuda(q, k, weights, 4096, 4);
        auto dot = mfq_tensor_backend::einsum(
            "bmhd,bkd->bmhk",
            {q.to(mfq_tensor_backend::kFloat32), k.to(mfq_tensor_backend::kFloat32)});
        auto ref = (mfq_tensor_backend::relu(dot) *
            weights.to(mfq_tensor_backend::kFloat32).unsqueeze(-1)).sum(2) /
            std::sqrt(static_cast<double>(H * ID));
        ref = ref.to(mfq_tensor_backend::kFloat16);
        const double rel =
            (test.to(mfq_tensor_backend::kFloat32) - ref.to(mfq_tensor_backend::kFloat32))
                .norm().item<double>() /
            std::max(ref.to(mfq_tensor_backend::kFloat32).norm().item<double>(), 1e-30);
        auto selected = dsv4_topk512_cuda(test);
        auto expected_topk = mfq_tensor_backend::topk(test, 512, -1, true, false);
        auto expected_scores = std::get<0>(expected_topk);
        auto expected = std::get<1>(expected_topk);
        auto selected_i64 = selected.to(mfq_tensor_backend::kInt64);
        auto selected_sorted = std::get<0>(
            mfq_tensor_backend::sort(selected_i64, -1));
        auto expected_sorted = std::get<0>(
            mfq_tensor_backend::sort(expected.to(mfq_tensor_backend::kInt64), -1));
        const bool topk_id_equal = selected_sorted.equal(expected_sorted);
        const bool topk_ids_valid =
            (selected_i64 >= 0).all().item<bool>() &&
            (selected_i64 < test.size(-1)).all().item<bool>();
        const bool topk_ids_unique =
            selected_sorted.slice(-1, 1, selected_sorted.size(-1))
                .ne(selected_sorted.slice(-1, 0, selected_sorted.size(-1) - 1))
                .all().item<bool>();
        bool topk_scores_equal = false;
        if (topk_ids_valid) {
            auto selected_scores = test.gather(-1, selected_i64);
            auto selected_score_sorted = std::get<0>(
                mfq_tensor_backend::sort(selected_scores, -1));
            auto expected_score_sorted = std::get<0>(
                mfq_tensor_backend::sort(expected_scores, -1));
            topk_scores_equal =
                selected_score_sorted.equal(expected_score_sorted);
        }
        std::cout << "dsv4_indexer_rel=" << rel
                  << " topk_id_set_equal=" << (topk_id_equal ? 1 : 0)
                  << " topk_score_multiset_equal="
                  << (topk_scores_equal ? 1 : 0)
                  << " topk_ids_valid=" << (topk_ids_valid ? 1 : 0)
                  << " topk_ids_unique=" << (topk_ids_unique ? 1 : 0)
                  << "\n";
        if (rel > 0.004 || !topk_scores_equal ||
            !topk_ids_valid || !topk_ids_unique) {
            throw std::runtime_error(
                "DeepSeek V4 indexer/top-k numerical check failed");
        }
    }

    constexpr int H = 64;
    constexpr int M = 3;
    constexpr int HISTORY = 127;
    constexpr int POOL = 800;
    constexpr int TOPK = 512;
    constexpr int WINDOW = 128;
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    auto q = mfq_tensor_backend::randn({B, H, M, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto raw = mfq_tensor_backend::randn({B, HISTORY + M, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto pooled = mfq_tensor_backend::randn({B, POOL, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto kv = mfq_tensor_backend::cat({raw, pooled}, 1).contiguous();
    std::vector<mfq_tensor_backend::Tensor> topk_rows;
    for (int row = 0; row < M; ++row) {
        topk_rows.push_back(
            mfq_tensor_backend::randperm(POOL, cuda.dtype(mfq_tensor_backend::kInt64))
                .narrow(0, 0, TOPK).to(mfq_tensor_backend::kInt32));
    }
    auto topk = mfq_tensor_backend::stack(topk_rows, 0).unsqueeze(0).contiguous();
    auto plan = dsv4_build_prefill_plan_cuda(
        topk, 4096, HISTORY, POOL, 4, WINDOW);
    {
        auto seq_len = mfq_tensor_backend::tensor({4097}, cuda.dtype(mfq_tensor_backend::kInt64));
        auto decode_plan = dsv4_build_decode_plan_cuda(
            topk.narrow(1, 0, 1).contiguous(),
            seq_len, POOL, 4, WINDOW);
        auto expected_local =
            mfq_tensor_backend::arange(4097 - WINDOW, 4097, cuda.dtype(mfq_tensor_backend::kInt64))
                .remainder(WINDOW);
        const bool local_equal = decode_plan[0]
            .index({0, 0, Slice(0, WINDOW)})
            .to(mfq_tensor_backend::kInt64).equal(expected_local);
        const bool pooled_equal = decode_plan[0]
            .index({0, 0, Slice(WINDOW, WINDOW + TOPK)})
            .equal(topk.index({0, 0}) + WINDOW);
        const bool mask_clear =
            (decode_plan[1] == 0).all().item<bool>();
        std::cout << "dsv4_decode_plan="
                  << (local_equal && pooled_equal && mask_clear ? 1 : 0)
                  << "\n";
        if (!local_equal || !pooled_equal || !mask_clear) {
            throw std::runtime_error(
                "DeepSeek V4 decode cache plan check failed");
        }
    }
    auto sinks = mfq_tensor_backend::randn({H}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto meta = mfq_tensor_backend::empty(
        {8 * 1024 * 1024}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto run_sparse = [&]() {
        return attention_dsv4_sparse_cuda(
            q, kv, plan[0], plan[1], sinks, meta, scale);
    };
    auto test = run_sparse();
    std::vector<mfq_tensor_backend::Tensor> ref_rows;
    auto q_ref = q.to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
    for (int row = 0; row < M; ++row) {
        auto idx = plan[0].index({0, row}).to(mfq_tensor_backend::kInt64);
        auto selected = kv.index_select(1, idx).index({0})
            .to(mfq_tensor_backend::kFloat32);
        auto score = mfq_tensor_backend::matmul(
            q_ref.index({0, Slice(), row}),
            selected.transpose(0, 1)) * scale;
        score = score + plan[1].index({0, row})
            .to(mfq_tensor_backend::kFloat32).unsqueeze(0);
        auto logits = mfq_tensor_backend::cat({score, sinks.unsqueeze(1)}, 1);
        auto probabilities = mfq_tensor_backend::softmax(logits, -1)
            .index({Slice(), Slice(0, score.size(1))});
        ref_rows.push_back(mfq_tensor_backend::matmul(probabilities, selected));
    }
    auto ref = mfq_tensor_backend::stack(ref_rows, 0).unsqueeze(0);
    mfq_cuda_synchronize();
    const double rel = (test - ref).norm().item<double>() /
        std::max(ref.norm().item<double>(), 1e-30);
    const double max_abs = (test - ref).abs().max().item<double>();
    std::cout << "dsv4_sparse_attention_rel=" << rel
              << " max_abs=" << max_abs << "\n";
    if (!mfq_tensor_backend::isfinite(test).all().item<bool>() || rel > 0.02) {
        throw std::runtime_error(
            "DeepSeek V4 sparse attention numerical check failed");
    }

    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        for (int i = 0; i < 5; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    std::cout << "dsv4_sparse_attention_m=" << M
              << " selected=" << plan[0].size(2)
              << " cuda_ms=" << time_ms(run_sparse) << "\n";
    return 0;
}
