#include "mlx_multimodal.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void require_values(
    const mlx::core::array& value,
    const std::vector<T>& expected,
    const std::string& message) {
    auto evaluated = value;
    evaluated.eval();
    require(evaluated.size() == expected.size(), message + ": size");
    const auto* actual = evaluated.data<T>();
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (actual[index] != expected[index]) {
            throw std::runtime_error(
                message + ": element " + std::to_string(index));
        }
    }
}

template <typename Callable>
void require_invalid(Callable&& callable, const std::string& message) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        using mlx::core::Shape;
        using mlx::core::array;
        using mfq::metal::MlxGridShape;

        const std::vector<float> token_embedding_values{
            0.0f, 1.0f,
            2.0f, 3.0f,
            4.0f, 5.0f,
            6.0f, 7.0f,
            8.0f, 9.0f,
        };
        const auto token_embeddings = array(
            token_embedding_values.begin(), Shape{1, 5, 2});
        const std::vector<float> replacement_values{
            20.0f, 21.0f,
            30.0f, 31.0f,
            40.0f, 41.0f,
        };
        const auto replacement = array(
            replacement_values.begin(),
            Shape{3, 2});
        const auto injected =
            mfq::metal::replace_multimodal_token_embeddings(
                token_embeddings,
                replacement,
                {10, 101, 101, 11, 102},
                {101, 102});
        require_values<float>(
            injected,
            {
                0.0f, 1.0f,
                20.0f, 21.0f,
                30.0f, 31.0f,
                6.0f, 7.0f,
                40.0f, 41.0f,
            },
            "placeholder injection");
        require_invalid(
            [&] {
                const std::vector<float> short_values{1.0f, 2.0f};
                (void)mfq::metal::replace_multimodal_token_embeddings(
                    token_embeddings,
                    array(short_values.begin(), Shape{1, 2}),
                    {10, 101, 101, 11, 102},
                    {101, 102});
            },
            "placeholder/replacement count mismatch was accepted");

        const auto image_positions = mfq::metal::build_grid_mrope_positions(
            {7, 101, 101, 101, 101, 8},
            101,
            102,
            2,
            {MlxGridShape{1, 4, 4}},
            {});
        require(image_positions.values.shape() == Shape{3, 6},
                "image position shape");
        require_values<std::int32_t>(
            image_positions.values,
            {
                0, 1, 1, 1, 1, 3,
                0, 1, 1, 2, 2, 3,
                0, 1, 2, 1, 2, 3,
            },
            "image MRoPE positions");
        require(image_positions.decode_delta == -2,
                "image MRoPE decode delta");

        const auto video_positions = mfq::metal::build_grid_mrope_positions(
            {9, 102, 10, 102, 11},
            101,
            102,
            2,
            {},
            {MlxGridShape{2, 2, 2}});
        require_values<std::int32_t>(
            video_positions.values,
            {
                0, 1, 2, 3, 4,
                0, 1, 2, 3, 4,
                0, 1, 2, 3, 4,
            },
            "video-frame MRoPE positions");
        require(video_positions.decode_delta == 0,
                "video-frame MRoPE decode delta");

        require_invalid(
            [] {
                (void)mfq::metal::build_grid_mrope_positions(
                    {101, 101, 101},
                    101,
                    102,
                    2,
                    {MlxGridShape{1, 4, 4}},
                    {});
            },
            "invalid grid/token geometry was accepted");

        std::cout << "mlx multimodal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
