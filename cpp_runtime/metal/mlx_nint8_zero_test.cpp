#include "mlx_nint8_zero.h"
#include "mlx_deepseek_v4_attention.h"
#include "mlx_tensor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

template <typename T>
void append(std::vector<std::uint8_t>& blob, T value) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(&value);
    blob.insert(blob.end(), bytes, bytes + sizeof(value));
}

struct Fixture {
    std::vector<std::uint8_t> blob;
    std::vector<std::int8_t> q;
    std::vector<float> scales;
};

template <typename T>
void write_scalar(std::ostream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_string(std::ostream& stream, const std::string& value) {
    write_scalar<std::uint32_t>(
        stream,
        static_cast<std::uint32_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

class TemporaryFile {
public:
    explicit TemporaryFile(std::filesystem::path path)
        : path_(std::move(path)) {}

    ~TemporaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

Fixture make_fixture() {
    constexpr std::int32_t output_size = 3;
    constexpr std::int32_t input_size = 64;
    constexpr std::uint32_t groups = 2;
    constexpr std::uint16_t scale_bits[] = {
        0x3800, 0x3c00,
        0x3400, 0x4000,
        0x3000, 0x3e00,
    };
    const std::vector<float> scales{
        0.5f, 1.0f,
        0.25f, 2.0f,
        0.125f, 1.5f,
    };

    std::vector<std::int8_t> q(
        static_cast<std::size_t>(output_size) * input_size);
    for (std::size_t index = 0; index < q.size(); ++index) {
        q[index] = static_cast<std::int8_t>(
            static_cast<int>((index * 7 + 3) % 31) - 15);
    }

    std::vector<std::uint8_t> blob{'N', 'I', '8', '0'};
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(blob, output_size);
    append<std::uint32_t>(blob, groups);
    for (std::size_t block = 0; block < scales.size(); ++block) {
        append<std::uint16_t>(blob, scale_bits[block]);
        const auto source = block * 32;
        const auto* bytes =
            reinterpret_cast<const std::uint8_t*>(q.data() + source);
        blob.insert(blob.end(), bytes, bytes + 32);
    }
    return {std::move(blob), std::move(q), scales};
}

Fixture make_grouped_fixture() {
    constexpr std::int32_t output_size = 6;
    constexpr std::int32_t input_size = 64;
    constexpr std::uint32_t groups = 2;
    const std::vector<float> scales(
        static_cast<std::size_t>(output_size) * groups,
        0.5f);

    std::vector<std::int8_t> q(
        static_cast<std::size_t>(output_size) * input_size);
    for (std::size_t index = 0; index < q.size(); ++index) {
        q[index] = static_cast<std::int8_t>(
            static_cast<int>((index * 11 + 5) % 29) - 14);
    }

    std::vector<std::uint8_t> blob{'N', 'I', '8', '0'};
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(blob, output_size);
    append<std::uint32_t>(blob, groups);
    for (std::size_t block = 0; block < scales.size(); ++block) {
        append<std::uint16_t>(blob, 0x3800);
        const auto source = block * 32;
        const auto* bytes =
            reinterpret_cast<const std::uint8_t*>(q.data() + source);
        blob.insert(blob.end(), bytes, bytes + 32);
    }
    return {std::move(blob), std::move(q), scales};
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    float tolerance = 2e-4f) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "NINT8-0 Metal result mismatch: actual=" +
            std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

float decoded(
    const Fixture& fixture,
    int output,
    int column) {
    return fixture.scales[
        static_cast<std::size_t>(output) * 2 + column / 32] *
        static_cast<float>(
            fixture.q[
                static_cast<std::size_t>(output) * 64 + column]);
}

void require_parse_error(std::vector<std::uint8_t> blob) {
    bool rejected = false;
    try {
        (void)mfq::metal::MlxNint8ZeroWeight::from_blob(blob);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "malformed NINT8-0 blob was accepted");
}

mfq::metal::MfqContainer make_container(
    const Fixture& fixture,
    TemporaryFile& file) {
    {
        std::ofstream stream(file.path(), std::ios::binary);
        require(
            static_cast<bool>(stream),
            "failed to create NINT8-0 MFQ test container");
        stream.write("MFQ1", 4);
        write_scalar<std::uint32_t>(stream, 2);
        write_string(stream, "unit-test");
        write_scalar<std::uint32_t>(stream, 0);
        write_scalar<std::uint32_t>(stream, 1);
        write_string(stream, "weight");
        write_string(stream, "NINT8-0");
        write_scalar<std::uint64_t>(
            stream,
            static_cast<std::uint64_t>(fixture.blob.size()));
        stream.write(
            reinterpret_cast<const char*>(fixture.blob.data()),
            static_cast<std::streamsize>(fixture.blob.size()));
        require(
            static_cast<bool>(stream),
            "failed to write NINT8-0 MFQ test container");
    }
    return mfq::metal::MfqContainer(file.path());
}

} // namespace

int main() {
    try {
        using namespace mlx::core;
        require(
            mfq::metal::is_nint8_zero_dtype("NINT8-0"),
            "real NINT8-0 dtype was rejected");
        for (const char* invalid : {
                 "NINT8", "NINT", "NINT8-1", "nint8-0", "",
             }) {
            require(
                !mfq::metal::is_nint8_zero_dtype(invalid),
                std::string("invalid NINT8-0 dtype was accepted: ") +
                    invalid);
        }

        const auto fixture = make_fixture();
        const auto weight =
            mfq::metal::MlxNint8ZeroWeight::from_blob(fixture.blob);
        require(weight.input_size() == 64, "NINT8-0 input size mismatch");
        require(weight.output_size() == 3, "NINT8-0 output size mismatch");
        require(weight.groups() == 2, "NINT8-0 group count mismatch");
        require(
            weight.packed_nbytes() == 3 * 64 + 3 * 2 * 2,
            "NINT8-0 packed byte count mismatch");

        std::vector<float> input_values(2 * 64);
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] =
                static_cast<float>(
                    static_cast<int>((index * 5 + 1) % 19) - 9) /
                8.0f;
        }
        const array input(input_values.begin(), Shape{2, 64});

        auto output = weight.matmul(input);
        output.eval();
        const auto* values = output.data<float>();
        for (int row = 0; row < 2; ++row) {
            for (int output_index = 0; output_index < 3; ++output_index) {
                float expected = 0.0f;
                for (int column = 0; column < 64; ++column) {
                    expected +=
                        input_values[row * 64 + column] *
                        decoded(fixture, output_index, column);
                }
                require_close(
                    values[row * 3 + output_index],
                    expected);
            }
        }

        auto output_f16 = astype(
            weight.matmul(astype(input, float16)),
            float32);
        output_f16.eval();
        const auto* values_f16 = output_f16.data<float>();
        for (int row = 0; row < 2; ++row) {
            for (int output_index = 0; output_index < 3; ++output_index) {
                float expected = 0.0f;
                for (int column = 0; column < 64; ++column) {
                    expected +=
                        input_values[row * 64 + column] *
                        decoded(fixture, output_index, column);
                }
                require_close(
                    values_f16[row * 3 + output_index],
                    expected,
                    0.05f);
            }
        }

        const auto grouped_fixture = make_grouped_fixture();
        const auto grouped_weight =
            mfq::metal::MlxNint8ZeroWeight::from_blob(
                grouped_fixture.blob);
        std::vector<float> grouped_input_values(2 * 2 * 64);
        for (std::size_t index = 0;
             index < grouped_input_values.size();
             ++index) {
            grouped_input_values[index] =
                static_cast<float>(
                    static_cast<int>((index * 3 + 2) % 23) - 11) /
                16.0f;
        }
        const array grouped_input(
            grouped_input_values.begin(),
            Shape{2, 2, 64});
        auto grouped_output = grouped_weight.grouped_row_matmul(
            grouped_input,
            2);
        grouped_output.eval();
        require(
            grouped_output.shape() == Shape({2, 2, 3}),
            "NINT8-0 grouped-row output shape mismatch");
        const auto* grouped_values = grouped_output.data<float>();
        for (int row = 0; row < 2; ++row) {
            for (int group = 0; group < 2; ++group) {
                for (int local_output = 0;
                     local_output < 3;
                     ++local_output) {
                    const int output_index = group * 3 + local_output;
                    float expected = 0.0f;
                    for (int column = 0; column < 64; ++column) {
                        expected +=
                            grouped_input_values[
                                (row * 2 + group) * 64 + column] *
                            grouped_fixture.scales[
                                output_index * 2 + column / 32] *
                            static_cast<float>(
                                grouped_fixture.q[
                                    output_index * 64 + column]);
                    }
                    require_close(
                        grouped_values[
                            (row * 2 + group) * 3 + local_output],
                        expected);
                }
            }
        }

        auto grouped_output_f16 = astype(
            grouped_weight.grouped_row_matmul(
                astype(grouped_input, float16),
                2),
            float32);
        grouped_output_f16.eval();
        const auto* grouped_values_f16 =
            grouped_output_f16.data<float>();
        for (int index = 0; index < 12; ++index) {
            require_close(
                grouped_values_f16[index],
                grouped_values[index],
                0.05f);
        }

        std::vector<float> rope_cos_values(2 * 4);
        std::vector<float> rope_sin_values(2 * 4);
        for (int token = 0; token < 2; ++token) {
            for (int pair = 0; pair < 4; ++pair) {
                const float angle =
                    0.07f * static_cast<float>(1 + token * 4 + pair);
                rope_cos_values[token * 4 + pair] = std::cos(angle);
                rope_sin_values[token * 4 + pair] = std::sin(angle);
            }
        }
        const array rope_cosine(
            rope_cos_values.begin(),
            Shape{2, 4});
        const array rope_sine(
            rope_sin_values.begin(),
            Shape{2, 4});
        auto grouped_input_half = astype(grouped_input, float16);
        auto attention_input = reshape(
            grouped_input_half,
            Shape{1, 2, 8, 16});
        auto rope_prefix = slice(
            attention_input,
            Shape{0, 0, 0, 0},
            Shape{1, 2, 8, 8});
        auto rope_tail = slice(
            attention_input,
            Shape{0, 0, 0, 8},
            Shape{1, 2, 8, 16});
        auto explicit_inverse_rope = reshape(
            concatenate(
                {
                    rope_prefix,
                    mfq::metal::deepseek_v4_rope_adjacent(
                        rope_tail,
                        rope_cosine,
                        rope_sine,
                        true),
                },
                -1),
            Shape{2, 2, 64});
        auto explicit_rope_output = astype(
            grouped_weight.grouped_row_matmul(
                explicit_inverse_rope,
                2),
            float32);
        auto fused_rope_output = astype(
            grouped_weight.grouped_row_matmul_inverse_rope(
                grouped_input_half,
                2,
                rope_cosine,
                rope_sine,
                16,
                8),
            float32);
        eval(explicit_rope_output, fused_rope_output);
        const auto* explicit_rope_values =
            explicit_rope_output.data<float>();
        const auto* fused_rope_values =
            fused_rope_output.data<float>();
        for (int index = 0; index < 12; ++index) {
            require_close(
                fused_rope_values[index],
                explicit_rope_values[index],
                0.01f);
        }

        std::vector<float> large_input_values(64 * 64);
        for (std::size_t index = 0;
             index < large_input_values.size();
             ++index) {
            large_input_values[index] =
                static_cast<float>(
                    static_cast<int>(index % 17) - 8) /
                16.0f;
        }
        const array large_input(
            large_input_values.begin(),
            Shape{64, 64});
        auto large_output = astype(
            weight.matmul(astype(large_input, float16)),
            float32);
        large_output.eval();
        const auto* large_values = large_output.data<float>();
        for (int row = 0; row < 64; ++row) {
            for (int output_index = 0;
                 output_index < 3;
                 ++output_index) {
                float expected = 0.0f;
                for (int column = 0; column < 64; ++column) {
                    expected +=
                        large_input_values[row * 64 + column] *
                        decoded(fixture, output_index, column);
                }
                require_close(
                    large_values[row * 3 + output_index],
                    expected,
                    0.1f);
            }
        }

        const array token_ids({2, 0}, Shape{2}, int32);
        auto embeddings =
            weight.embedding(token_ids, float32);
        embeddings.eval();
        const auto* embedded = embeddings.data<float>();
        for (int token = 0; token < 2; ++token) {
            const int source_row = token == 0 ? 2 : 0;
            for (int column = 0; column < 64; ++column) {
                require_close(
                    embedded[token * 64 + column],
                    decoded(fixture, source_row, column));
            }
        }

        auto embeddings_f16 = astype(
            weight.embedding(token_ids),
            float32);
        embeddings_f16.eval();
        const auto* embedded_f16 = embeddings_f16.data<float>();
        for (int token = 0; token < 2; ++token) {
            const int source_row = token == 0 ? 2 : 0;
            for (int column = 0; column < 64; ++column) {
                require_close(
                    embedded_f16[token * 64 + column],
                    decoded(fixture, source_row, column),
                    0.01f);
            }
        }

        TemporaryFile file(
            std::filesystem::temp_directory_path() /
            "mfq-metal-nint8-zero-runtime-test.mfq");
        const auto model = make_container(fixture, file);
        const auto linear =
            mfq::metal::MlxLinear::load(model, "weight");
        require(linear.packed(), "NINT8-0 linear was not marked packed");
        require(
            linear.input_size() == 64 && linear.output_size() == 3,
            "loaded NINT8-0 linear shape mismatch");
        auto loaded_output = linear(input);
        loaded_output.eval();
        const auto* loaded_values = loaded_output.data<float>();
        for (int row = 0; row < 2; ++row) {
            for (int output_index = 0; output_index < 3; ++output_index) {
                float expected = 0.0f;
                for (int column = 0; column < 64; ++column) {
                    expected +=
                        input_values[row * 64 + column] *
                        decoded(fixture, output_index, column);
                }
                require_close(
                    loaded_values[row * 3 + output_index],
                    expected);
            }
        }

        const auto embedding =
            mfq::metal::MlxEmbedding::load(model, "weight");
        require(
            embedding.vocabulary_size() == 3 &&
                embedding.hidden_size() == 64,
            "loaded NINT8-0 embedding shape mismatch");
        auto loaded_embeddings =
            embedding(token_ids, float32);
        loaded_embeddings.eval();
        const auto* loaded_embedded =
            loaded_embeddings.data<float>();
        for (int token = 0; token < 2; ++token) {
            const int source_row = token == 0 ? 2 : 0;
            for (int column = 0; column < 64; ++column) {
                require_close(
                    loaded_embedded[token * 64 + column],
                    decoded(fixture, source_row, column));
            }
        }

        auto tied_projection = embedding.project(input);
        tied_projection.eval();
        const auto* tied_projection_values =
            tied_projection.data<float>();
        for (int row = 0; row < 2; ++row) {
            for (int output_index = 0;
                 output_index < 3;
                 ++output_index) {
                float expected = 0.0f;
                for (int column = 0; column < 64; ++column) {
                    expected +=
                        input_values[row * 64 + column] *
                        decoded(
                            fixture,
                            output_index,
                            column);
                }
                require_close(
                    tied_projection_values[
                        row * 3 + output_index],
                    expected);
            }
        }

        mfq::metal::set_mlx_predequantize_fp16(true);
        const auto dense_linear =
            mfq::metal::MlxLinear::load(model, "weight");
        const auto dense_embedding =
            mfq::metal::MlxEmbedding::load(model, "weight");
        mfq::metal::set_mlx_predequantize_fp16(false);
        require(
            !dense_linear.packed() &&
                dense_linear.dense_weight_ref() != nullptr &&
                dense_linear.dense_weight_ref()->dtype() == float16,
            "NINT8-0 was not materialized as FP16");
        auto dense_output = astype(dense_linear(input), float32);
        auto dense_tied = astype(
            dense_embedding.project(input),
            float32);
        eval(dense_output, dense_tied);
        const auto* dense_output_values = dense_output.data<float>();
        const auto* dense_tied_values = dense_tied.data<float>();
        for (std::size_t index = 0;
             index < dense_output.size();
             ++index) {
            require_close(
                dense_output_values[index],
                loaded_values[index],
                0.05f);
            require_close(
                dense_tied_values[index],
                tied_projection_values[index],
                0.05f);
        }

        auto bad_magic = fixture.blob;
        bad_magic[0] = 'X';
        require_parse_error(std::move(bad_magic));
        auto truncated = fixture.blob;
        truncated.pop_back();
        require_parse_error(std::move(truncated));

        std::cout
            << "MFQ C++ NINT8-0 packed matmul/embedding/"
            << "tied-projection Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
