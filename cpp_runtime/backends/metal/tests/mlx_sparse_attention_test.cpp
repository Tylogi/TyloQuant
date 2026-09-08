#include "mlx_sparse_attention.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

mlx::core::array patterned_half(
    std::size_t count,
    const mlx::core::Shape& shape,
    int multiplier,
    float scale) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<float>(
            static_cast<int>((index * multiplier) % 251) - 125) * scale;
    }
    return mlx::core::astype(
        mlx::core::array(
            values.begin(),
            shape,
            mlx::core::float32),
        mlx::core::float16);
}

void test_block_gqa_matches_expanded_reference() {
    using namespace mlx::core;
    constexpr int batch = 1;
    constexpr int heads = 24;
    constexpr int kv_heads = 2;
    constexpr int queries = 3;
    constexpr int keys = 20;
    constexpr int dimension = 256;
    constexpr int query_offset = keys - queries;
    constexpr int block_size = 4;
    constexpr int selected_count = 3;

    auto query = patterned_half(
        batch * heads * queries * dimension,
        Shape{batch, heads, queries, dimension},
        37,
        1.0f / 511.0f);
    auto key = patterned_half(
        batch * kv_heads * keys * dimension,
        Shape{batch, kv_heads, keys, dimension},
        53,
        1.0f / 487.0f);
    auto value = patterned_half(
        batch * kv_heads * keys * dimension,
        Shape{batch, kv_heads, keys, dimension},
        71,
        1.0f / 463.0f);

    const std::vector<std::int32_t> block_values{
        0, 2, 3,
        0, 2, 3,
        0, 2, 3,
    };
    const array blocks(
        block_values.begin(),
        Shape{batch, queries, selected_count},
        int32);

    constexpr int expanded = selected_count * block_size + block_size - 1;
    std::vector<std::int32_t> indices(
        batch * queries * expanded,
        -1);
    for (int token = 0; token < queries; ++token) {
        const int absolute = query_offset + token;
        const int complete = (absolute + 1) / block_size;
        const int valid_blocks = std::min(selected_count, complete);
        int output = token * expanded;
        for (int selected = 0; selected < valid_blocks; ++selected) {
            const int block = block_values[token * selected_count + selected];
            for (int offset = 0; offset < block_size; ++offset) {
                indices[output++] = block * block_size + offset;
            }
        }
        output = token * expanded + selected_count * block_size;
        for (int position = complete * block_size;
             position <= absolute && output < (token + 1) * expanded;
             ++position) {
            indices[output++] = position;
        }
    }
    const array expanded_indices(
        indices.begin(),
        Shape{batch, queries, expanded},
        int32);

    auto actual = mfq::metal::mlx_sparse_block_gqa_attention(
        query,
        key,
        value,
        blocks,
        query_offset,
        block_size);
    auto normalized = mfq::metal::mlx_sparse_block_gqa_attention(
        astype(query, float32),
        astype(key, float32),
        astype(value, float32),
        astype(blocks, int64),
        query_offset,
        block_size);
    actual = astype(actual, float32);
    normalized = astype(normalized, float32);
    auto query32 = astype(query, float32);
    auto key32 = astype(key, float32);
    auto value32 = astype(value, float32);
    eval(actual, normalized, query32, key32, value32, expanded_indices);

    std::vector<float> expected(actual.size(), 0.0f);
    const auto* q = query32.data<float>();
    const auto* k = key32.data<float>();
    const auto* v = value32.data<float>();
    for (int token = 0; token < queries; ++token) {
        for (int head = 0; head < heads; ++head) {
            const int kv_head = head / (heads / kv_heads);
            std::vector<float> scores(expanded, -INFINITY);
            float maximum_score = -INFINITY;
            for (int selected = 0; selected < expanded; ++selected) {
                const int row = indices[token * expanded + selected];
                if (row < 0 || row >= keys) continue;
                float dot = 0.0f;
                for (int d = 0; d < dimension; ++d) {
                    dot += q[(head * queries + token) * dimension + d]
                        * k[(kv_head * keys + row) * dimension + d];
                }
                scores[selected] = dot / std::sqrt(float(dimension));
                maximum_score = std::max(maximum_score, scores[selected]);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                if (!std::isfinite(score)) continue;
                score = std::exp(score - maximum_score);
                denominator += score;
            }
            for (int selected = 0; selected < expanded; ++selected) {
                const int row = indices[token * expanded + selected];
                if (row < 0 || row >= keys || scores[selected] <= 0.0f) continue;
                const float probability = scores[selected] / denominator;
                for (int d = 0; d < dimension; ++d) {
                    expected[(token * heads + head) * dimension + d] +=
                        probability
                        * v[(kv_head * keys + row) * dimension + d];
                }
            }
        }
    }

    float maximum = 0.0f;
    float normalization_maximum = 0.0f;
    double squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(
            actual.data<float>()[index] - expected[index]);
        maximum = std::max(maximum, error);
        squared += static_cast<double>(error) * error;
        normalization_maximum = std::max(
            normalization_maximum,
            std::fabs(
                actual.data<float>()[index]
                - normalized.data<float>()[index]));
    }
    const float rms = static_cast<float>(
        std::sqrt(squared / static_cast<double>(actual.size())));
    if (maximum > 8e-3f || rms > 1e-3f ||
        normalization_maximum != 0.0f) {
        throw std::runtime_error(
            "selected-block sparse GQA mismatch: max="
            + std::to_string(maximum)
            + " rms=" + std::to_string(rms)
            + " normalized=" + std::to_string(normalization_maximum));
    }
    std::cout << "selected-block sparse GQA max=" << maximum
              << " rms=" << rms << "\n";
}

} // namespace

int main() {
    try {
        mlx::core::set_default_device(mlx::core::Device::gpu);
        test_block_gqa_matches_expanded_reference();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
