#include "mlx_deepseek_v4_hc.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

constexpr int kBatch = 1;
constexpr int kTokens = 2;
constexpr int kConnections = 4;
constexpr int kHidden = 4096;
constexpr int kMixWidth = 24;
constexpr float kEps = 1e-6f;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

std::vector<float> evaluated_float(mlx::core::array value) {
    if (value.dtype() != mlx::core::float32) {
        value = mlx::core::astype(
            value,
            mlx::core::float32);
    }
    value.eval();
    return {
        value.data<float>(),
        value.data<float>() + value.size(),
    };
}

void require_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance,
    const std::string& label) {
    require(
        actual.size() == expected.size(),
        label + " size mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) >
            tolerance) {
            throw std::runtime_error(
                label + " mismatch at " +
                std::to_string(index) + ": actual=" +
                std::to_string(actual[index]) +
                " expected=" +
                std::to_string(expected[index]));
        }
    }
}

void require_float_bits_equal(
    mlx::core::array actual,
    mlx::core::array expected,
    const std::string& label) {
    require(actual.shape() == expected.shape(), label + " shape mismatch");
    require(
        actual.dtype() == mlx::core::float32 &&
            expected.dtype() == mlx::core::float32,
        label + " dtype mismatch");
    actual = mlx::core::contiguous(actual);
    expected = mlx::core::contiguous(expected);
    mlx::core::eval(actual, expected);
    const auto* actual_bits = actual.data<std::uint32_t>();
    const auto* expected_bits = expected.data<std::uint32_t>();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual_bits[index] != expected_bits[index]) {
            throw std::runtime_error(
                label + " mismatch at " + std::to_string(index) +
                ": actual_bits=" + std::to_string(actual_bits[index]) +
                " expected_bits=" + std::to_string(expected_bits[index]) +
                " actual=" + std::to_string(actual.data<float>()[index]) +
                " expected=" + std::to_string(expected.data<float>()[index]));
        }
    }
}

mlx::core::array slice_last(
    const mlx::core::array& input,
    int begin,
    int end) {
    auto start = mlx::core::Shape(input.ndim(), 0);
    auto stop = input.shape();
    start.back() = begin;
    stop.back() = end;
    return mlx::core::slice(input, std::move(start), std::move(stop));
}

mlx::core::array softmax_last(const mlx::core::array& input) {
    auto source = mlx::core::contiguous(input);
    auto maximum = mlx::core::max(source, -1, true);
    auto exponential = mlx::core::exp(source - maximum);
    return exponential / mlx::core::sum(exponential, -1, true);
}

struct GenericV41Metadata {
    mlx::core::array post;
    mlx::core::array combination;
    mlx::core::array pre;
};

GenericV41Metadata generic_v41_metadata(
    const mlx::core::array& mixes,
    const mlx::core::array& scale,
    const mlx::core::array& base) {
    using namespace mlx::core;
    auto pre = sigmoid(
        slice_last(mixes, 0, kConnections) *
            slice_last(scale, 0, 1) +
        slice_last(base, 0, kConnections)) +
        kEps;
    auto post = 2.0f * sigmoid(
        slice_last(mixes, kConnections, 2 * kConnections) *
            slice_last(scale, 1, 2) +
        slice_last(base, kConnections, 2 * kConnections));
    auto combination = softmax_last(
        reshape(
            slice_last(mixes, 2 * kConnections, kMixWidth),
            Shape{1, 1, kConnections, kConnections}) *
            slice_last(scale, 2, 3) +
        reshape(
            slice_last(base, 2 * kConnections, kMixWidth),
            Shape{kConnections, kConnections})) +
        kEps;
    combination = combination /
        (sum(combination, -2, true) + kEps);
    for (int iteration = 1; iteration < 20; ++iteration) {
        combination = combination /
            (sum(combination, -1, true) + kEps);
        combination = combination /
            (sum(combination, -2, true) + kEps);
    }
    return {
        std::move(post),
        std::move(combination),
        std::move(pre),
    };
}

std::array<float, 16> sinkhorn(
    const float* values) {
    std::array<float, 16> result{};
    for (int source = 0; source < kConnections; ++source) {
        float maximum = values[source * kConnections];
        for (int destination = 1;
             destination < kConnections;
             ++destination) {
            maximum = std::max(
                maximum,
                values[source * kConnections + destination]);
        }
        float denominator = kEps;
        for (int destination = 0;
             destination < kConnections;
             ++destination) {
            const float probability = std::exp(
                values[source * kConnections + destination]
                - maximum);
            result[source * kConnections + destination] =
                probability;
            denominator += probability;
        }
        for (int destination = 0;
             destination < kConnections;
             ++destination) {
            result[source * kConnections + destination] =
                result[source * kConnections + destination]
                    / denominator
                + kEps;
        }
    }
    for (int destination = 0;
         destination < kConnections;
         ++destination) {
        float denominator = kEps;
        for (int source = 0;
             source < kConnections;
             ++source) {
            denominator +=
                result[source * kConnections + destination];
        }
        for (int source = 0;
             source < kConnections;
             ++source) {
            result[source * kConnections + destination] /=
                denominator;
        }
    }
    for (int iteration = 1; iteration < 20; ++iteration) {
        for (int source = 0;
             source < kConnections;
             ++source) {
            float denominator = kEps;
            for (int destination = 0;
                 destination < kConnections;
                 ++destination) {
                denominator += result[
                    source * kConnections + destination];
            }
            for (int destination = 0;
                 destination < kConnections;
                 ++destination) {
                result[
                    source * kConnections + destination] /=
                    denominator;
            }
        }
        for (int destination = 0;
             destination < kConnections;
             ++destination) {
            float denominator = kEps;
            for (int source = 0;
                 source < kConnections;
                 ++source) {
                denominator += result[
                    source * kConnections + destination];
            }
            for (int source = 0;
                 source < kConnections;
                 ++source) {
                result[
                    source * kConnections + destination] /=
                    denominator;
            }
        }
    }
    return result;
}

void test_hc_pre_post() {
    constexpr int rows = kBatch * kTokens;
    std::vector<float> residual(
        rows * kConnections * kHidden);
    for (std::size_t index = 0;
         index < residual.size();
         ++index) {
        residual[index] =
            static_cast<float>(
                static_cast<int>(index % 29) - 14)
            * 0.03125f;
    }
    std::vector<float> mixes(rows * kMixWidth);
    for (std::size_t index = 0;
         index < mixes.size();
         ++index) {
        mixes[index] =
            static_cast<float>(
                static_cast<int>((index * 7) % 31) - 15)
            * 0.025f;
    }
    const std::vector<float> scale{
        0.9f,
        1.1f,
        0.7f,
    };
    std::vector<float> base(kMixWidth);
    for (int index = 0; index < kMixWidth; ++index) {
        base[index] =
            static_cast<float>((index % 9) - 4) * 0.015f;
    }

    const mlx::core::array residual_array(
        residual.begin(),
        mlx::core::Shape{
            kBatch,
            kTokens,
            kConnections,
            kHidden,
        });
    const mlx::core::array mixes_array(
        mixes.begin(),
        mlx::core::Shape{kBatch, kTokens, kMixWidth});
    const mlx::core::array scale_array(
        scale.begin(),
        mlx::core::Shape{3});
    const mlx::core::array base_array(
        base.begin(),
        mlx::core::Shape{kMixWidth});

    auto actual = mfq::metal::deepseek_v4_hc_pre(
        residual_array,
        mixes_array,
        scale_array,
        base_array,
        20,
        kEps);
    std::vector<float> expected_reduced(rows * kHidden);
    std::vector<float> expected_post(rows * kConnections);
    std::vector<float> expected_combination(
        rows * kConnections * kConnections);
    for (int row = 0; row < rows; ++row) {
        std::array<float, kConnections> pre{};
        for (int connection = 0;
             connection < kConnections;
             ++connection) {
            pre[connection] = sigmoid(
                mixes[row * kMixWidth + connection]
                    * scale[0]
                + base[connection]) + kEps;
            expected_post[row * kConnections + connection] =
                2.0f * sigmoid(
                    mixes[
                        row * kMixWidth
                        + kConnections + connection]
                        * scale[1]
                    + base[kConnections + connection]);
        }
        std::array<float, 16> affine{};
        for (int index = 0; index < 16; ++index) {
            affine[index] =
                mixes[row * kMixWidth + 8 + index]
                    * scale[2]
                + base[8 + index];
        }
        const auto combination = sinkhorn(affine.data());
        std::copy(
            combination.begin(),
            combination.end(),
            expected_combination.begin() + row * 16);
        for (int feature = 0; feature < kHidden; ++feature) {
            float value = 0.0f;
            for (int connection = 0;
                 connection < kConnections;
                 ++connection) {
                value += pre[connection] * residual[
                    (row * kConnections + connection)
                        * kHidden + feature];
            }
            expected_reduced[row * kHidden + feature] = value;
        }
    }
    require_close(
        evaluated_float(actual.reduced),
        expected_reduced,
        8e-4f,
        "HC pre reduced");
    require_close(
        evaluated_float(actual.post),
        expected_post,
        2e-4f,
        "HC pre post-gates");
    require_close(
        evaluated_float(actual.combination),
        expected_combination,
        3e-4f,
        "HC pre Sinkhorn");

    std::vector<float> norm(kHidden);
    for (int feature = 0; feature < kHidden; ++feature) {
        norm[feature] =
            0.9f + 0.01f * static_cast<float>(feature % 17);
    }
    const mlx::core::array norm_array(
        norm.begin(),
        mlx::core::Shape{kHidden});
    mfq::metal::MlxRmsNorm separate_norm(norm_array, kEps);
    auto expected_normalized = separate_norm(actual.reduced);
    auto fused_normalized = mfq::metal::deepseek_v4_hc_pre_norm(
        residual_array,
        mixes_array,
        scale_array,
        base_array,
        norm_array,
        20,
        kEps,
        kEps);
    require_close(
        evaluated_float(fused_normalized.reduced),
        evaluated_float(expected_normalized),
        1.5e-3f,
        "fused HC pre RMSNorm");
    require_close(
        evaluated_float(fused_normalized.post),
        evaluated_float(actual.post),
        2e-4f,
        "fused HC pre post-gates");
    require_close(
        evaluated_float(fused_normalized.combination),
        evaluated_float(actual.combination),
        3e-4f,
        "fused HC pre Sinkhorn");

    auto residual_half = mlx::core::astype(
        residual_array,
        mlx::core::float16);
    auto residual_flat = mlx::core::reshape(
        residual_half,
        mlx::core::Shape{
            kBatch,
            kTokens,
            kConnections * kHidden,
        });
    auto residual_float = mlx::core::astype(
        residual_flat,
        mlx::core::float32);
    auto residual_inverse = mlx::core::rsqrt(
        mlx::core::mean(
            residual_float * residual_float,
            -1,
            true) +
        kEps);
    auto unnormalized_mixes =
        mixes_array / residual_inverse;
    auto fully_fused = mfq::metal::deepseek_v4_hc_pre_norm(
        residual_array,
        unnormalized_mixes,
        scale_array,
        base_array,
        norm_array,
        20,
        kEps,
        kEps,
        true);
    require_close(
        evaluated_float(fully_fused.reduced),
        evaluated_float(expected_normalized),
        2e-3f,
        "fused HC input RMS and output RMSNorm");
    require_close(
        evaluated_float(fully_fused.post),
        evaluated_float(actual.post),
        3e-4f,
        "fused HC input RMS post-gates");
    require_close(
        evaluated_float(fully_fused.combination),
        evaluated_float(actual.combination),
        4e-4f,
        "fused HC input RMS Sinkhorn");

    std::vector<float> branch(rows * kHidden);
    for (std::size_t index = 0;
         index < branch.size();
         ++index) {
        branch[index] =
            static_cast<float>(
                static_cast<int>((index * 5) % 23) - 11)
            * 0.04f;
    }
    const mlx::core::array branch_array(
        branch.begin(),
        mlx::core::Shape{kBatch, kTokens, kHidden});
    auto expanded = mfq::metal::deepseek_v4_hc_post(
        branch_array,
        residual_array,
        actual.post,
        actual.combination);
    std::vector<float> expected_expanded(
        rows * kConnections * kHidden);
    for (int row = 0; row < rows; ++row) {
        for (int destination = 0;
             destination < kConnections;
             ++destination) {
            for (int feature = 0;
                 feature < kHidden;
                 ++feature) {
                float value =
                    expected_post[
                        row * kConnections + destination]
                    * branch[row * kHidden + feature];
                for (int source = 0;
                     source < kConnections;
                     ++source) {
                    value += expected_combination[
                        (row * kConnections + source)
                            * kConnections + destination]
                        * residual[
                            (row * kConnections + source)
                                * kHidden + feature];
                }
                expected_expanded[
                    (row * kConnections + destination)
                        * kHidden + feature] = value;
            }
        }
    }
    require_close(
        evaluated_float(std::move(expanded)),
        expected_expanded,
        1.5e-3f,
        "HC post");

    auto shared_array = branch_array * 0.25f;
    auto fused_sum = mfq::metal::deepseek_v4_hc_post_sum(
        branch_array,
        shared_array,
        residual_array,
        actual.post,
        actual.combination);
    auto separate_sum = mfq::metal::deepseek_v4_hc_post(
        branch_array + shared_array,
        residual_array,
        actual.post,
        actual.combination);
    require_close(
        evaluated_float(std::move(fused_sum)),
        evaluated_float(std::move(separate_sum)),
        1.5e-3f,
        "HC post fused branch sum");

    auto residual_bfloat = mlx::core::astype(
        residual_array,
        mlx::core::bfloat16);
    auto bfloat_pre = mfq::metal::deepseek_v4_hc_pre(
        residual_bfloat,
        mixes_array,
        scale_array,
        base_array,
        20,
        kEps);
    require(
        bfloat_pre.reduced.dtype() == mlx::core::bfloat16,
        "BF16 HC collapse did not preserve the activation dtype");
    require(
        bfloat_pre.packed_metadata.has_value(),
        "BF16 HC did not expose packed post metadata");
    require_close(
        evaluated_float(bfloat_pre.reduced),
        expected_reduced,
        8e-3f,
        "BF16 HC pre reduced");
    require_close(
        evaluated_float(bfloat_pre.post),
        expected_post,
        2e-4f,
        "BF16 HC pre post-gates");
    require_close(
        evaluated_float(bfloat_pre.combination),
        expected_combination,
        3e-4f,
        "BF16 HC pre Sinkhorn");

    auto bfloat_fused_norm =
        mfq::metal::deepseek_v4_hc_pre_norm(
            residual_bfloat,
            mixes_array,
            scale_array,
            base_array,
            norm_array,
            20,
            kEps,
            kEps);
    auto bfloat_separate_norm = separate_norm(
        bfloat_pre.reduced);
    require(
        bfloat_fused_norm.reduced.dtype() ==
            mlx::core::bfloat16,
        "BF16 fused HC/RMSNorm changed the activation dtype");
    require_close(
        evaluated_float(bfloat_fused_norm.reduced),
        evaluated_float(bfloat_separate_norm),
        1.6e-2f,
        "BF16 fused HC pre RMSNorm");

    auto bfloat_branch = mlx::core::astype(
        branch_array,
        mlx::core::bfloat16);
    auto bfloat_post = mfq::metal::deepseek_v4_hc_post(
        bfloat_branch,
        residual_bfloat,
        bfloat_pre.post,
        bfloat_pre.combination);
    auto bfloat_packed_post =
        mfq::metal::deepseek_v4_hc_post_packed(
            bfloat_branch,
            residual_bfloat,
            *bfloat_pre.packed_metadata);
    require(
        bfloat_packed_post.dtype() == mlx::core::bfloat16,
        "BF16 HC post changed the activation dtype");
    require_close(
        evaluated_float(bfloat_packed_post),
        evaluated_float(bfloat_post),
        0.0f,
        "BF16 packed HC post");
}

void test_invalid_shapes() {
    bool rejected = false;
    try {
        (void)mfq::metal::deepseek_v4_hc_pre(
            mlx::core::zeros(
                {1, 1, 3, 64},
                mlx::core::float16),
            mlx::core::zeros(
                {1, 1, 24},
                mlx::core::float32),
            mlx::core::zeros(
                {3},
                mlx::core::float32),
            mlx::core::zeros(
                {24},
                mlx::core::float32));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid HC connection count was accepted");
}

void test_v41_hidden_width() {
    constexpr int hidden = 5120;
    auto residual = mlx::core::zeros(
        {1, 1, kConnections, hidden},
        mlx::core::bfloat16);
    auto mixes = mlx::core::zeros(
        {1, 1, kMixWidth},
        mlx::core::float32);
    auto scale = mlx::core::ones({3}, mlx::core::float32);
    auto base = mlx::core::zeros({kMixWidth}, mlx::core::float32);
    auto norm = mlx::core::ones({hidden}, mlx::core::float32);
    auto pre = mfq::metal::deepseek_v4_hc_pre_norm(
        residual,
        mixes,
        scale,
        base,
        norm,
        20,
        kEps,
        kEps);
    require(
        pre.reduced.shape() == mlx::core::Shape{1, 1, hidden},
        "V4.1-width HC pre returned the wrong shape");
    require(
        pre.reduced.dtype() == mlx::core::bfloat16 &&
            pre.packed_metadata.has_value() && pre.pre.has_value(),
        "V4.1-width HC pre lost BF16 packed output");
    require_close(
        evaluated_float(*pre.pre),
        std::vector<float>(kConnections, 0.5f + kEps),
        2e-4f,
        "V4.1-width carried HC pre");
    auto branch = mlx::core::ones(
        {1, 1, hidden},
        mlx::core::bfloat16);
    auto post = mfq::metal::deepseek_v4_hc_post_packed(
        branch,
        residual,
        *pre.packed_metadata);
    require_close(
        evaluated_float(std::move(post)),
        std::vector<float>(
            static_cast<std::size_t>(kConnections) * hidden,
            1.0f),
        0.0f,
        "V4.1-width packed HC post");
}

void test_v41_post_matches_generic_graph() {
    constexpr int hidden = 5120;
    const auto residual_values = mlx::core::reshape(
        mlx::core::astype(
            mlx::core::arange(
                kConnections * hidden,
                mlx::core::float32) *
                    0.00037f -
                3.5f,
            mlx::core::bfloat16),
        mlx::core::Shape{1, 1, kConnections, hidden});
    const auto branch_values = mlx::core::reshape(
        mlx::core::astype(
            mlx::core::sin(
                mlx::core::arange(hidden, mlx::core::float32) *
                0.013f),
            mlx::core::bfloat16),
        mlx::core::Shape{1, 1, hidden});
    const auto shared_values = mlx::core::reshape(
        mlx::core::astype(
            mlx::core::cos(
                mlx::core::arange(hidden, mlx::core::float32) *
                0.017f) *
                0.25f,
            mlx::core::bfloat16),
        mlx::core::Shape{1, 1, hidden});
    const mlx::core::array post(
        {0.37f, 0.81f, 1.19f, 1.63f},
        mlx::core::Shape{1, 1, kConnections});
    const mlx::core::array combination(
        {
            0.13f, 0.17f, 0.29f, 0.41f,
            0.31f, 0.23f, 0.19f, 0.11f,
            0.07f, 0.43f, 0.37f, 0.13f,
            0.47f, 0.09f, 0.21f, 0.23f,
        },
        mlx::core::Shape{1, 1, kConnections, kConnections});
    const auto generic = [&](const mlx::core::array& branch) {
        auto mixed = mlx::core::sum(
            mlx::core::expand_dims(combination, -1) *
                mlx::core::expand_dims(residual_values, 3),
            2);
        return mlx::core::astype(
            mlx::core::expand_dims(post, -1) *
                    mlx::core::expand_dims(branch, 2) +
                mixed,
            mlx::core::bfloat16);
    };

    require_close(
        evaluated_float(mfq::metal::deepseek_v4_hc_post(
            branch_values,
            residual_values,
            post,
            combination)),
        evaluated_float(generic(branch_values)),
        0.0f,
        "V4.1 HC post generic equivalence");
    require_close(
        evaluated_float(mfq::metal::deepseek_v4_hc_post_sum(
            branch_values,
            shared_values,
            residual_values,
            post,
            combination)),
        evaluated_float(generic(branch_values + shared_values)),
        0.0f,
        "V4.1 HC post-sum generic equivalence");

    for (int seed = 1; seed <= 6; ++seed) {
        const float phase = static_cast<float>(seed) * 0.071f;
        const auto residual_case = mlx::core::reshape(
            mlx::core::astype(
                mlx::core::sin(
                    mlx::core::arange(
                        kConnections * hidden,
                        mlx::core::float32) *
                        (0.003f + 0.0002f * seed) +
                    phase) *
                    (1.0f + 0.13f * seed),
                mlx::core::bfloat16),
            mlx::core::Shape{1, 1, kConnections, hidden});
        const auto branch_case = mlx::core::reshape(
            mlx::core::astype(
                mlx::core::cos(
                    mlx::core::arange(hidden, mlx::core::float32) *
                        (0.007f + 0.0003f * seed) -
                    phase),
                mlx::core::bfloat16),
            mlx::core::Shape{1, 1, hidden});
        const auto shared_case = mlx::core::reshape(
            mlx::core::astype(
                mlx::core::sin(
                    mlx::core::arange(hidden, mlx::core::float32) *
                        (0.011f + 0.0001f * seed) +
                    2.0f * phase) *
                    0.31f,
                mlx::core::bfloat16),
            mlx::core::Shape{1, 1, hidden});
        const auto post_case = mlx::core::reshape(
            mlx::core::sin(
                mlx::core::arange(kConnections, mlx::core::float32) *
                    0.37f +
                phase) +
                1.1f,
            mlx::core::Shape{1, 1, kConnections});
        const auto combination_case = mlx::core::reshape(
            mlx::core::cos(
                mlx::core::arange(
                    kConnections * kConnections,
                    mlx::core::float32) *
                    (0.19f + 0.003f * seed) +
                phase) *
                0.47f,
            mlx::core::Shape{
                1, 1, kConnections, kConnections});
        const auto generic_case = [&](const mlx::core::array& branch) {
            auto mixed = mlx::core::sum(
                mlx::core::expand_dims(combination_case, -1) *
                    mlx::core::expand_dims(residual_case, 3),
                2);
            return mlx::core::astype(
                mlx::core::expand_dims(post_case, -1) *
                        mlx::core::expand_dims(branch, 2) +
                    mixed,
                mlx::core::bfloat16);
        };
        require_close(
            evaluated_float(mfq::metal::deepseek_v4_hc_post(
                branch_case,
                residual_case,
                post_case,
                combination_case)),
            evaluated_float(generic_case(branch_case)),
            0.0f,
            "V4.1 HC post randomized generic equivalence");
        require_close(
            evaluated_float(mfq::metal::deepseek_v4_hc_post_sum(
                branch_case,
                shared_case,
                residual_case,
                post_case,
                combination_case)),
            evaluated_float(generic_case(branch_case + shared_case)),
            0.0f,
            "V4.1 HC post-sum randomized generic equivalence");
    }

    if (std::getenv("MFQ_METAL_HC_BENCH") != nullptr) {
        const auto benchmark = [](auto&& make_output) {
            constexpr int width = 40;
            constexpr int repetitions = 20;
            const auto run_bundle = [&] {
                std::vector<mlx::core::array> pending;
                pending.reserve(width);
                for (int index = 0; index < width; ++index) {
                    pending.push_back(make_output());
                }
                mlx::core::eval(std::move(pending));
            };
            run_bundle();
            const auto started = std::chrono::steady_clock::now();
            for (int repetition = 0;
                 repetition < repetitions;
                 ++repetition) {
                run_bundle();
            }
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count() /
                static_cast<double>(repetitions * width);
        };
        std::cout
            << "HC post benchmark ms/call generic="
            << benchmark([&] { return generic(branch_values); })
            << " fused="
            << benchmark([&] {
                   return mfq::metal::deepseek_v4_hc_post(
                       branch_values,
                       residual_values,
                       post,
                       combination);
               })
            << " generic_sum="
            << benchmark([&] {
                   return generic(branch_values + shared_values);
               })
            << " fused_sum="
            << benchmark([&] {
                   return mfq::metal::deepseek_v4_hc_post_sum(
                       branch_values,
                       shared_values,
                       residual_values,
                       post,
                       combination);
               })
            << '\n';
    }
}

void test_v41_metadata_matches_generic_graph() {
    using namespace mlx::core;
    for (int test_case = 0; test_case < 14; ++test_case) {
        std::vector<float> mix_values(kMixWidth);
        std::vector<float> base_values(kMixWidth);
        for (int index = 0; index < kMixWidth; ++index) {
            if (test_case == 10) {
                mix_values[static_cast<std::size_t>(index)] = 0.0f;
                base_values[static_cast<std::size_t>(index)] = 0.0f;
            } else if (test_case > 10) {
                const float gain = test_case == 13 ? 4.75f : 0.73f;
                mix_values[static_cast<std::size_t>(index)] =
                    gain * static_cast<float>((index * 11 + 3) % 19 - 9);
                base_values[static_cast<std::size_t>(index)] =
                    0.037f * static_cast<float>((index * 5 + 1) % 13 - 6);
            } else {
                mix_values[static_cast<std::size_t>(index)] =
                    std::sin(
                        static_cast<float>(index * 7 + test_case * 3) *
                        (0.071f + 0.002f * test_case)) *
                        (0.35f + 0.17f * test_case) +
                    0.013f *
                        static_cast<float>((index + test_case) % 5 - 2);
                base_values[static_cast<std::size_t>(index)] =
                    std::cos(
                        static_cast<float>(index * 5 + test_case) * 0.043f) *
                    (0.19f + 0.01f * test_case);
            }
        }
        std::vector<float> scale_values{
            0.47f + 0.031f * test_case,
            0.83f - 0.017f * test_case,
            test_case == 12
                ? -1.37f
                : 1.11f + 0.023f * test_case,
        };
        const array mixes(
            mix_values.begin(),
            Shape{1, 1, kMixWidth});
        const array scale(scale_values.begin(), Shape{3});
        const array base(base_values.begin(), Shape{kMixWidth});
        auto expected = generic_v41_metadata(mixes, scale, base);
        auto actual = mfq::metal::deepseek_v41_hc_metadata_exact(
            mixes,
            scale,
            base,
            20,
            kEps);
        require_float_bits_equal(
            actual.pre,
            expected.pre,
            "V4.1 HC exact metadata pre case " +
                std::to_string(test_case));
        require_float_bits_equal(
            actual.post,
            expected.post,
            "V4.1 HC exact metadata post case " +
                std::to_string(test_case));
        require_float_bits_equal(
            actual.combination,
            expected.combination,
            "V4.1 HC exact metadata Sinkhorn case " +
                std::to_string(test_case));

        const char* benchmark_setting =
            std::getenv("MFQ_METAL_HC_METADATA_BENCH");
        if (test_case == 0 && benchmark_setting != nullptr &&
            std::string(benchmark_setting) != "0") {
            constexpr int width = 40;
            constexpr int repetitions = 20;
            const auto benchmark = [&](const auto& make_outputs) {
                const auto run_bundle = [&] {
                    std::vector<array> pending;
                    pending.reserve(width * 3);
                    for (int index = 0; index < width; ++index) {
                        auto values = make_outputs();
                        for (auto& value : values) {
                            pending.push_back(std::move(value));
                        }
                    }
                    eval(std::move(pending));
                };
                run_bundle();
                const auto started = std::chrono::steady_clock::now();
                for (int repetition = 0;
                     repetition < repetitions;
                     ++repetition) {
                    run_bundle();
                }
                return std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count() /
                    static_cast<double>(repetitions * width);
            };
            std::cout
                << "HC metadata benchmark ms/call generic="
                << benchmark([&] {
                       auto value = generic_v41_metadata(
                           mixes,
                           scale,
                           base);
                       return std::vector<array>{
                           std::move(value.post),
                           std::move(value.combination),
                           std::move(value.pre),
                       };
                   })
                << " fused="
                << benchmark([&] {
                       auto value =
                           mfq::metal::deepseek_v41_hc_metadata_exact(
                               mixes,
                               scale,
                               base,
                               20,
                               kEps);
                       return std::vector<array>{
                           std::move(value.post),
                           std::move(value.combination),
                           std::move(value.pre),
                       };
                   })
                << '\n';
        }
    }
}

void test_v41_collapse_norm_matches_generic_graph() {
    using namespace mlx::core;
    constexpr int hidden = 5120;
    constexpr int rows = 1;
    for (int test_case = 0; test_case < 6; ++test_case) {
        std::vector<float> residual_values(
            static_cast<std::size_t>(kConnections) * hidden);
        for (std::size_t index = 0;
             index < residual_values.size();
             ++index) {
            residual_values[index] =
                std::sin(static_cast<float>(index + 3 * test_case) * 0.013f) *
                    0.7f +
                std::cos(static_cast<float>(index * 7 + test_case) * 0.003f) *
                    0.2f;
        }
        std::vector<float> pre_values(kConnections);
        for (int connection = 0;
             connection < kConnections;
             ++connection) {
            pre_values[static_cast<std::size_t>(connection)] =
                0.15f + 0.07f * static_cast<float>(
                    connection + test_case % 3);
        }
        std::vector<float> norm_values(hidden);
        for (int feature = 0; feature < hidden; ++feature) {
            norm_values[static_cast<std::size_t>(feature)] =
                0.85f + 0.003f * static_cast<float>(
                    (feature * 5 + test_case * 11) % 97);
        }
        auto residual = astype(
            array(
                residual_values.begin(),
                Shape{rows, 1, kConnections, hidden}),
            bfloat16);
        const array pre(
            pre_values.begin(),
            Shape{rows, 1, kConnections});
        const array norm(norm_values.begin(), Shape{hidden});
        mfq::metal::MlxRmsNorm normalizer(norm, kEps);
        const auto generic = [&] {
            auto reduced = sum(
                expand_dims(pre, -1) * astype(residual, float32),
                2);
            reduced = astype(reduced, bfloat16);
            return normalizer(reduced);
        };
        auto expected = contiguous(generic());
        auto actual = contiguous(
            mfq::metal::deepseek_v41_hc_collapse_norm(
                residual,
                pre,
                norm,
                kEps));
        eval(expected, actual);
        const auto* expected_bits = expected.data<std::uint16_t>();
        const auto* actual_bits = actual.data<std::uint16_t>();
        for (std::size_t index = 0; index < actual.size(); ++index) {
            if (actual_bits[index] != expected_bits[index]) {
                throw std::runtime_error(
                    "V4.1 HC collapse/RMSNorm mismatch in case " +
                    std::to_string(test_case) + " at " +
                    std::to_string(index) + ": actual_bits=" +
                    std::to_string(actual_bits[index]) +
                    " expected_bits=" +
                    std::to_string(expected_bits[index]));
            }
        }

        const char* benchmark_setting =
            std::getenv("MFQ_METAL_HC_COLLAPSE_BENCH");
        if (test_case == 0 && benchmark_setting != nullptr &&
            std::string(benchmark_setting) != "0") {
            constexpr int width = 80;
            constexpr int repetitions = 20;
            const auto benchmark = [&](const auto& make_output) {
                const auto run_bundle = [&] {
                    std::vector<array> pending;
                    pending.reserve(width);
                    for (int index = 0; index < width; ++index) {
                        pending.push_back(make_output());
                    }
                    eval(std::move(pending));
                };
                run_bundle();
                const auto started = std::chrono::steady_clock::now();
                for (int repetition = 0;
                     repetition < repetitions;
                     ++repetition) {
                    run_bundle();
                }
                return std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count() /
                    static_cast<double>(repetitions * width);
            };
            std::cout
                << "HC collapse/RMSNorm benchmark ms/call generic="
                << benchmark(generic)
                << " fused="
                << benchmark([&] {
                       return mfq::metal::deepseek_v41_hc_collapse_norm(
                           residual,
                           pre,
                           norm,
                           kEps);
                   })
                << '\n';
        }
    }
}

} // namespace

int main() {
    try {
        test_hc_pre_post();
        test_v41_hidden_width();
        test_v41_post_matches_generic_graph();
        test_v41_metadata_matches_generic_graph();
        test_v41_collapse_norm_matches_generic_graph();
        test_invalid_shapes();
        std::cout
            << "MFQ C++ DeepSeek-V4 hyper-connection "
               "Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
