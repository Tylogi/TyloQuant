#include "mlx_deepseek_v4_hc.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <array>
#include <cmath>
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

} // namespace

int main() {
    try {
        test_hc_pre_post();
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
