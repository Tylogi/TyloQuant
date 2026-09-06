#include "mlx_grid_vision.h"
#include "mlx_tensor_schema.h"
#include "qwen35_model.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename T> void append(std::vector<std::uint8_t> &output, T value) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(&value);
  output.insert(output.end(), bytes, bytes + sizeof(value));
}

std::vector<std::uint8_t> dense_blob(const std::vector<std::int64_t> &shape,
                                     const std::vector<float> &values) {
  std::size_t count = 1;
  for (const auto dimension : shape) {
    count *= static_cast<std::size_t>(dimension);
  }
  require(count == values.size(), "dense fixture size mismatch");
  std::vector<std::uint8_t> output;
  append<std::uint32_t>(output, static_cast<std::uint32_t>(shape.size()));
  for (const auto dimension : shape)
    append<std::int64_t>(output, dimension);
  for (const auto value : values)
    append<float>(output, value);
  return output;
}

void write_string(std::ostream &output, const std::string &value) {
  const auto size = static_cast<std::uint32_t>(value.size());
  output.write(reinterpret_cast<const char *>(&size), sizeof(size));
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

struct DenseRecord {
  std::string name;
  std::vector<std::uint8_t> blob;
};

void write_fixture(const std::filesystem::path &path,
                   const std::vector<DenseRecord> &records) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(output), "failed to create grid-ViT fixture");
  output.write("MFQ1", 4);
  const std::uint32_t version = 2;
  output.write(reinterpret_cast<const char *>(&version), sizeof(version));
  write_string(output, "grid-vit-test");
  const std::uint32_t metadata_count = 0;
  output.write(reinterpret_cast<const char *>(&metadata_count),
               sizeof(metadata_count));
  const auto record_count = static_cast<std::uint32_t>(records.size());
  output.write(reinterpret_cast<const char *>(&record_count),
               sizeof(record_count));
  for (const auto &record : records) {
    write_string(output, record.name);
    write_string(output, "F32");
    const auto size = static_cast<std::uint64_t>(record.blob.size());
    output.write(reinterpret_cast<const char *>(&size), sizeof(size));
  }
  for (const auto &record : records) {
    output.write(reinterpret_cast<const char *>(record.blob.data()),
                 static_cast<std::streamsize>(record.blob.size()));
  }
  require(static_cast<bool>(output), "failed to finish grid-ViT fixture");
}

std::vector<float> zeros(std::size_t count) {
  return std::vector<float>(count, 0.0f);
}

std::vector<float> identity(int width) {
  std::vector<float> result(static_cast<std::size_t>(width * width), 0.0f);
  for (int index = 0; index < width; ++index) {
    result[static_cast<std::size_t>(index * width + index)] = 1.0f;
  }
  return result;
}

void test_canonical_schema() {
  using mfq::metal::GridVisionTensorRole;
  require(mfq::metal::grid_vision_canonical_name(
              GridVisionTensorRole::patch_weight) ==
              "vision.patch_embedding.weight",
          "canonical patch role mismatch");
  require(mfq::metal::grid_vision_canonical_name(
              GridVisionTensorRole::block_mlp_up_weight, 12) ==
              "vision.block.12.mlp.up.weight",
          "canonical block role mismatch");
  require(mfq::metal::grid_vision_canonical_name(
              GridVisionTensorRole::merger_mlp_down_bias) ==
              "vision.merger.mlp.down.bias",
          "canonical merger role mismatch");

}

void test_block_major_layout() {
  const std::vector<mfq::metal::GridThw> grids = {{2, 2, 4}};
  const auto layout = mfq::metal::make_grid_vision_layout(grids, 2);
  require(layout.patch_count == 16, "layout patch count mismatch");
  require(layout.segment_lengths == std::vector<std::int32_t>({8, 8}),
          "layout frame segmentation mismatch");
  const std::vector<std::int32_t> one_frame = {
      0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3,
  };
  require(std::vector<std::int32_t>(layout.positions.begin(),
                                    layout.positions.begin() +
                                        one_frame.size()) == one_frame,
          "layout is not block-major");
  require(std::vector<std::int32_t>(layout.positions.begin() + one_frame.size(),
                                    layout.positions.end()) == one_frame,
          "temporal layout did not repeat spatial positions");
}

void test_learned_position_interpolation() {
  const std::vector<mfq::metal::GridThw> grids = {{1, 2, 4}};
  const auto interpolation =
      mfq::metal::make_learned_position_interpolation(grids, 2, 3);
  require(interpolation.patch_count == 8 &&
              interpolation.indices.size() == 32 &&
              interpolation.weights.size() == 32,
          "interpolation shape mismatch");
  require(std::vector<std::int32_t>(interpolation.indices.begin(),
                                    interpolation.indices.begin() + 4) ==
              std::vector<std::int32_t>({0, 1, 3, 4}),
          "interpolation tap order mismatch");
  require(std::abs(interpolation.weights.at(0) - 1.0f) < 1e-7f,
          "interpolation corner weight mismatch");
  for (std::size_t patch = 0; patch < 8; ++patch) {
    float sum = 0.0f;
    for (std::size_t tap = 0; tap < 4; ++tap) {
      sum += interpolation.weights.at(patch * 4 + tap);
    }
    require(std::abs(sum - 1.0f) < 1e-6f,
            "interpolation weights do not sum to one");
  }
  require(std::abs(interpolation.weights.at(5) - 2.0f / 3.0f) < 1e-6f,
          "interpolation fractional column weight mismatch");
}

void test_config_validation() {
  mfq::metal::GridVisionConfig config;
  config.hidden_size = 1152;
  config.intermediate_size = 4304;
  config.depth = 27;
  config.num_heads = 16;
  config.in_channels = 3;
  config.patch_size = 16;
  config.temporal_patch_size = 2;
  config.spatial_merge_size = 2;
  config.out_hidden_size = 5120;
  config.num_position_embeddings = 2304;
  config.validate();
  require(config.head_dim() == 72, "config head dimension mismatch");
  require(config.patch_width() == 1536, "config patch width mismatch");
}

void test_small_canonical_tower() {
  using mfq::metal::GridVisionTensorRole;
  const auto name = [](GridVisionTensorRole role) {
    return mfq::metal::grid_vision_canonical_name(role, 0);
  };
  const std::vector<float> ones(4, 1.0f);
  const std::vector<float> zero4(4, 0.0f);
  const std::vector<DenseRecord> records = {
      {name(GridVisionTensorRole::patch_weight),
       dense_blob({4, 1, 1, 1, 1}, {1.0f, 2.0f, 3.0f, 4.0f})},
      {name(GridVisionTensorRole::patch_bias), dense_blob({4}, zero4)},
      {name(GridVisionTensorRole::position_weight),
       dense_blob({4, 4},
                  {10, 20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})},
      {name(GridVisionTensorRole::block_norm1_weight), dense_blob({4}, ones)},
      {name(GridVisionTensorRole::block_norm1_bias), dense_blob({4}, zero4)},
      {name(GridVisionTensorRole::block_attention_qkv_weight),
       dense_blob({12, 4}, zeros(48))},
      {name(GridVisionTensorRole::block_attention_output_weight),
       dense_blob({4, 4}, zeros(16))},
      {name(GridVisionTensorRole::block_norm2_weight), dense_blob({4}, ones)},
      {name(GridVisionTensorRole::block_norm2_bias), dense_blob({4}, zero4)},
      {name(GridVisionTensorRole::block_mlp_up_weight),
       dense_blob({4, 4}, zeros(16))},
      {name(GridVisionTensorRole::block_mlp_down_weight),
       dense_blob({4, 4}, identity(4))},
      {name(GridVisionTensorRole::merger_norm_weight), dense_blob({4}, ones)},
      {name(GridVisionTensorRole::merger_norm_bias), dense_blob({4}, zero4)},
      {name(GridVisionTensorRole::merger_mlp_up_weight),
       dense_blob({4, 4}, zeros(16))},
      {name(GridVisionTensorRole::merger_mlp_down_weight),
       dense_blob({4, 4}, identity(4))},
  };
  const auto path = std::filesystem::temp_directory_path() /
                    "mfq-grid-vision-canonical-test.mfq";
  write_fixture(path, records);
  try {
    mfq::metal::GridVisionConfig config;
    config.hidden_size = 4;
    config.intermediate_size = 4;
    config.depth = 1;
    config.num_heads = 1;
    config.in_channels = 1;
    config.patch_size = 1;
    config.temporal_patch_size = 1;
    config.spatial_merge_size = 1;
    config.out_hidden_size = 4;
    config.num_position_embeddings = 4;
    mfq::metal::MfqContainer model(path);
    auto tower = mfq::metal::MlxQwenVisionTower::load(model, config);
    mfq::metal::MlxGridMediaInput media;
    media.processor = "unit-grid";
    const std::vector<float> pixels = {1.0f};
    media.pixel_values =
        mlx::core::array(pixels.begin(), mlx::core::Shape{1, 1});
    media.media_grids = {{1, 1, 1}};
    auto output = tower.forward(media);
    mlx::core::eval(output.patch_features, output.merged_features);
    require(output.patch_features.shape() == mlx::core::Shape{1, 4},
            "canonical tower patch output shape mismatch");
    require(output.merged_features.shape() == mlx::core::Shape{1, 4},
            "canonical tower merger output shape mismatch");
    const auto *patches = output.patch_features.data<float>();
    for (int index = 0; index < 4; ++index) {
      require(std::abs(patches[index] - (11.0f * (index + 1))) < 1e-5f,
              "canonical tower patch/position result mismatch");
      require(std::abs(output.merged_features.data<float>()[index]) < 1e-6f,
              "canonical tower merger result mismatch");
    }
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

void test_real_legacy_qwen38(const std::filesystem::path &path) {
  mfq::metal::MfqContainer model(path);
  const auto config = mfq::metal::Qwen35Config::from_mfq(model);
  require(config.has_vision(), "real Qwen artifact has no vision config");
  // The container is the sole legacy-read boundary. Runtime components only
  // request canonical tensor names, even for an old artifact.
  auto tower = mfq::metal::MlxQwenVisionTower::load(model, *config.vision);
  const int patches = 4;
  const int patch_width = static_cast<int>(config.vision->patch_width());
  const auto values = zeros(static_cast<std::size_t>(patches * patch_width));
  mfq::metal::MlxGridMediaInput media;
  media.processor = "qwen-grid-v1";
  media.pixel_values =
      mlx::core::array(values.begin(), mlx::core::Shape{patches, patch_width});
  media.media_grids = {{1, 2, 2}};
  auto one_block_config = *config.vision;
  one_block_config.depth = 1;
  auto one_block =
      mfq::metal::MlxGridVisionEncoder::load(model, one_block_config)
          .encode(media);
  auto one_block_float = mlx::core::astype(one_block, mlx::core::float32);
  mlx::core::eval(one_block_float);
  double one_block_sum = 0.0;
  for (std::size_t index = 0; index < one_block_float.size(); ++index) {
    one_block_sum += one_block_float.data<float>()[index];
  }
  std::cout << "real Qwen one-block sum=" << one_block_sum << " first8=";
  for (int index = 0; index < 8; ++index) {
    if (index)
      std::cout << ',';
    std::cout << one_block_float.data<float>()[index];
  }
  std::cout << "\n";
  auto output = tower.forward(media);
  auto patch_float =
      mlx::core::astype(output.patch_features, mlx::core::float32);
  auto merged_float =
      mlx::core::astype(output.merged_features, mlx::core::float32);
  mlx::core::eval(patch_float, merged_float);
  require(output.patch_features.shape() == mlx::core::Shape{4, 1152},
          "real Qwen patch output shape mismatch");
  require(output.merged_features.shape() == mlx::core::Shape{1, 5120},
          "real Qwen merged output shape mismatch");
  const auto *merged = merged_float.data<float>();
  const auto *patch_values = patch_float.data<float>();
  double patch_sum = 0.0;
  for (std::size_t index = 0; index < patch_float.size(); ++index) {
    patch_sum += patch_values[index];
  }
  std::cout << "real Qwen patch sum=" << patch_sum << " first8=";
  for (int index = 0; index < 8; ++index) {
    if (index)
      std::cout << ',';
    std::cout << patch_values[index];
  }
  std::cout << "\n";
  double sum = 0.0;
  for (std::size_t index = 0; index < merged_float.size(); ++index) {
    require(std::isfinite(static_cast<float>(merged[index])),
            "real Qwen merged output is not finite");
    sum += merged[index];
  }
  std::cout << "real Qwen merged sum=" << sum << " first8=";
  for (int index = 0; index < 8; ++index) {
    if (index)
      std::cout << ',';
    std::cout << merged[index];
  }
  std::cout << "\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc <= 2, "usage: mlx_grid_vision_test [qwen-model.mfq]");
    test_canonical_schema();
    test_block_major_layout();
    test_learned_position_interpolation();
    test_config_validation();
    test_small_canonical_tower();
    if (argc == 2)
      test_real_legacy_qwen38(argv[1]);
    std::cout << "native canonical grid-ViT test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
