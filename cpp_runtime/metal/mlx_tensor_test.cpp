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
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    blob.insert(blob.end(), bytes, bytes + sizeof(value));
}

std::vector<std::uint8_t> dense_blob() {
    std::vector<std::uint8_t> result;
    append<std::uint32_t>(result, 2);
    append<std::int64_t>(result, 3);
    append<std::int64_t>(result, 2);
    for (const float value : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}) {
        append<float>(result, value);
    }
    return result;
}

std::vector<std::uint8_t> dense_bf16_blob() {
    std::vector<std::uint8_t> result;
    append<std::uint32_t>(result, 2);
    append<std::int64_t>(result, 3);
    append<std::int64_t>(result, 2);
    for (const std::uint16_t value : {
             std::uint16_t{0x3f80}, std::uint16_t{0x4000},
             std::uint16_t{0x4040}, std::uint16_t{0x4080},
             std::uint16_t{0x40a0}, std::uint16_t{0x40c0}}) {
        append<std::uint16_t>(result, value);
    }
    return result;
}

std::vector<std::uint8_t> pack_values(
    const std::vector<std::uint8_t>& values,
    int bits) {
    std::vector<std::uint8_t> result(
        (values.size() * static_cast<std::size_t>(bits) + 7) / 8,
        0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        for (int bit = 0; bit < bits; ++bit) {
            if ((values[index] >> bit) & 1u) {
                const auto target =
                    index * static_cast<std::size_t>(bits) +
                    static_cast<std::size_t>(bit);
                result[target / 8] |=
                    static_cast<std::uint8_t>(1u << (target & 7));
            }
        }
    }
    return result;
}

std::vector<std::uint8_t> nint_blob(int bits) {
    constexpr std::int32_t output_size = 2;
    constexpr std::int32_t group_size = 5;
    constexpr std::int32_t groups = 2;
    constexpr std::int32_t input_size = 9;
    const auto maximum = (1u << bits) - 1u;
    std::vector<std::uint8_t> quantized(
        output_size * groups * group_size);
    for (std::size_t index = 0; index < quantized.size(); ++index) {
        quantized[index] = static_cast<std::uint8_t>(
            (index * 3 + 1) % (maximum + 1));
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, static_cast<std::uint8_t>(bits));
    append<std::uint8_t>(blob, 1);
    append<std::int32_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(blob, output_size);
    append<std::uint32_t>(blob, groups);
    append<std::uint16_t>(blob, 0x3c00);
    append<std::uint16_t>(blob, 0x3c00);
    append<std::uint16_t>(blob, 0);
    append<std::uint16_t>(blob, 0);
    append<std::uint8_t>(blob, 0x0f);
    append<std::uint8_t>(blob, 0x00);
    const auto packed = pack_values(quantized, bits);
    blob.insert(blob.end(), packed.begin(), packed.end());
    return blob;
}

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

struct NintRecordFixture {
    std::string name;
    std::string dtype;
    std::vector<std::uint8_t> blob;
};

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

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_nint_record_dtypes() {
    std::vector<NintRecordFixture> records;
    records.reserve(9);
    records.push_back({"legacy", "NINT", nint_blob(4)});
    for (int bits = 1; bits <= 8; ++bits) {
        records.push_back({
            "weight" + std::to_string(bits),
            "NINT" + std::to_string(bits),
            nint_blob(bits),
        });
    }

    TemporaryFile file(
        std::filesystem::temp_directory_path() /
        "mfq-metal-tensor-nint-dtypes.mfq");
    {
        std::ofstream stream(file.path(), std::ios::binary);
        require(
            static_cast<bool>(stream),
            "failed to create NINT dtype test container");
        stream.write("MFQ1", 4);
        write_scalar<std::uint32_t>(stream, 2);
        write_string(stream, "unit-test");
        write_scalar<std::uint32_t>(stream, 0);
        write_scalar<std::uint32_t>(
            stream,
            static_cast<std::uint32_t>(records.size()));
        for (const auto& record : records) {
            write_string(stream, record.name);
            write_string(stream, record.dtype);
            write_scalar<std::uint64_t>(
                stream,
                static_cast<std::uint64_t>(record.blob.size()));
        }
        for (const auto& record : records) {
            stream.write(
                reinterpret_cast<const char*>(record.blob.data()),
                static_cast<std::streamsize>(record.blob.size()));
        }
        require(
            static_cast<bool>(stream),
            "failed to write NINT dtype test container");
    }

    const mfq::metal::MfqContainer model(file.path());
    for (const auto& record : records) {
        const auto linear =
            mfq::metal::MlxLinear::load(model, record.name);
        require(
            linear.packed(),
            record.dtype + " linear was not loaded as packed NINT");
        require(
            linear.input_size() == 9 && linear.output_size() == 2,
            record.dtype + " linear shape mismatch");

        const auto embedding =
            mfq::metal::MlxEmbedding::load(model, record.name);
        require(
            embedding.vocabulary_size() == 2 &&
                embedding.hidden_size() == 9,
            record.dtype + " embedding shape mismatch");

        const mlx::core::array projection_input(
            {
                1.0f, -1.0f, 2.0f,
                -2.0f, 3.0f, -3.0f,
                4.0f, -4.0f, 5.0f,
            },
            mlx::core::Shape{1, 9});
        auto linear_projection = linear(projection_input);
        auto tied_projection =
            embedding.project(projection_input);
        mlx::core::eval(
            linear_projection,
            tied_projection);
        const auto* linear_values =
            linear_projection.data<float>();
        const auto* tied_values =
            tied_projection.data<float>();
        for (std::size_t index = 0;
             index < tied_projection.size();
             ++index) {
            require(
                std::fabs(
                    tied_values[index] -
                    linear_values[index]) < 1e-4f,
                record.dtype +
                    " tied embedding projection mismatch");
        }
    }
}

void require_close(float actual, float expected) {
    if (std::fabs(actual - expected) > 1e-5f) {
        throw std::runtime_error(
            "dense MLX tensor mismatch: actual=" +
            std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

} // namespace

int main() {
    try {
        using namespace mlx::core;
        const auto dense =
            mfq::metal::load_dense_array("F32", dense_blob());
        const mfq::metal::MlxLinear linear(dense);
        const array input({2.0f, -1.0f}, Shape{1, 2});
        auto projected = linear(input);
        projected.eval();
        const auto* projected_values = projected.data<float>();
        require_close(projected_values[0], 0.0f);
        require_close(projected_values[1], 2.0f);
        require_close(projected_values[2], 4.0f);

        const mfq::metal::MlxEmbedding embedding(dense);
        const array ids({2, 0}, Shape{2}, int32);
        auto selected = embedding(ids, float32);
        selected.eval();
        const auto* selected_values = selected.data<float>();
        require_close(selected_values[0], 5.0f);
        require_close(selected_values[1], 6.0f);
        require_close(selected_values[2], 1.0f);
        require_close(selected_values[3], 2.0f);

        auto tied_projected = embedding.project(input);
        tied_projected.eval();
        const auto* tied_projected_values =
            tied_projected.data<float>();
        require_close(tied_projected_values[0], 0.0f);
        require_close(tied_projected_values[1], 2.0f);
        require_close(tied_projected_values[2], 4.0f);

        const auto dense_bf16 =
            mfq::metal::load_dense_array(
                "BF16", dense_bf16_blob());
        require(
            dense_bf16.dtype() == bfloat16,
            "BF16 dense tensor dtype mismatch");
        const mfq::metal::MlxLinear bf16_linear(dense_bf16);
        auto bf16_projected = astype(bf16_linear(input), float32);
        bf16_projected.eval();
        const auto* bf16_values = bf16_projected.data<float>();
        require_close(bf16_values[0], 0.0f);
        require_close(bf16_values[1], 2.0f);
        require_close(bf16_values[2], 4.0f);

        const array grouped_weight(
            {
                1.0f, 0.0f,
                0.0f, 1.0f,
                2.0f, 0.0f,
                0.0f, 2.0f,
            },
            Shape{4, 2});
        const mfq::metal::MlxLinear grouped_linear(
            grouped_weight);
        const array grouped_input(
            {
                3.0f, 4.0f,
                5.0f, 6.0f,
            },
            Shape{1, 2, 2});
        auto grouped_output =
            grouped_linear.grouped_row_matmul(
                grouped_input,
                2);
        grouped_output.eval();
        require(
            grouped_output.shape() == Shape({1, 2, 2}),
            "dense grouped-row output shape mismatch");
        const auto* grouped_values =
            grouped_output.data<float>();
        require_close(grouped_values[0], 3.0f);
        require_close(grouped_values[1], 4.0f);
        require_close(grouped_values[2], 10.0f);
        require_close(grouped_values[3], 12.0f);

        test_nint_record_dtypes();
        std::cout
            << "MFQ C++ dense and NINT1-NINT8 tensor/linear/"
            << "embedding/tied-projection/grouped-row tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
