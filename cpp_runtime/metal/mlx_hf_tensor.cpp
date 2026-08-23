#include "mlx_hf_tensor.h"

#include <mlx/allocator.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;

Shape shape(const HfSafetensorRecord& record) {
    if (record.shape.empty() || record.shape.size() > 8) {
        throw std::runtime_error(
            "unsupported Safetensors rank: " + record.name);
    }
    Shape result;
    result.reserve(record.shape.size());
    for (const auto dimension : record.shape) {
        if (dimension <= 0 || dimension > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error(
                "Safetensors dimension exceeds MLX limits: " + record.name);
        }
        result.push_back(static_cast<std::int32_t>(dimension));
    }
    return result;
}

Dtype dense_dtype(const HfSafetensorRecord& record) {
    if (record.dtype == "BF16") {
        return mlx::core::bfloat16;
    }
    if (record.dtype == "F16") {
        return mlx::core::float16;
    }
    if (record.dtype == "F32") {
        return mlx::core::float32;
    }
    if (record.dtype == "I32") {
        return mlx::core::int32;
    }
    if (record.dtype == "I64") {
        return mlx::core::int64;
    }
    throw std::runtime_error(
        "Safetensors tensor is not a supported dense dtype: " + record.name);
}

std::string scale_name(const std::string& weight) {
    constexpr std::string_view suffix = ".weight";
    if (weight.size() < suffix.size() ||
        weight.compare(weight.size() - suffix.size(), suffix.size(), suffix) != 0) {
        throw std::runtime_error("native MX tensor is not named *.weight: " + weight);
    }
    return weight.substr(0, weight.size() - suffix.size()) + ".scale";
}

array allocate_and_read(
    const HfSafetensorStore& checkpoint,
    const HfSafetensorRecord& record,
    Dtype dtype,
    Shape tensor_shape) {
    auto result = array(
        mlx::core::allocator::malloc(record.nbytes),
        std::move(tensor_shape),
        dtype);
    checkpoint.read_tensor(
        record,
        std::span<std::byte>(
            reinterpret_cast<std::byte*>(result.data<std::uint8_t>()),
            record.nbytes));
    return result;
}

} // namespace

MlxHfTensorStore::MlxHfTensorStore(std::filesystem::path root)
    : checkpoint_(std::make_shared<HfSafetensorStore>(std::move(root))) {}

MlxHfTensorStore::MlxHfTensorStore(
    std::shared_ptr<HfSafetensorStore> checkpoint)
    : checkpoint_(std::move(checkpoint)) {
    if (!checkpoint_) {
        throw std::invalid_argument("HF tensor checkpoint cannot be null");
    }
}

const HfSafetensorStore& MlxHfTensorStore::checkpoint() const noexcept {
    return *checkpoint_;
}

std::shared_ptr<HfSafetensorStore>
MlxHfTensorStore::shared_checkpoint() const noexcept {
    return checkpoint_;
}

array MlxHfTensorStore::load_dense(const std::string& name) const {
    const auto& record = checkpoint_->tensor(name);
    return allocate_and_read(
        *checkpoint_,
        record,
        dense_dtype(record),
        shape(record));
}

MlxMxWeight MlxHfTensorStore::load_mx(const std::string& name) const {
    const auto& values_record = checkpoint_->tensor(name);
    const auto& scales_record = checkpoint_->tensor(scale_name(name));
    if (values_record.shard != scales_record.shard ||
        scales_record.dtype != "F8_E8M0" ||
        values_record.shape.size() != 2 || scales_record.shape.size() != 2) {
        throw std::runtime_error("invalid native MX Safetensors pair: " + name);
    }
    std::string dtype;
    int input = 0;
    if (values_record.dtype == "I8") {
        dtype = "MXFP4";
        input = static_cast<int>(values_record.shape[1] * 2);
    } else if (values_record.dtype == "F8_E4M3" ||
               values_record.dtype == "F8_E4M3FN") {
        dtype = "MXFP8";
        input = static_cast<int>(values_record.shape[1]);
    } else {
        throw std::runtime_error(
            "Safetensors tensor is not native MXFP4/MXFP8: " + name);
    }
    const auto output = static_cast<int>(values_record.shape[0]);
    auto values = allocate_and_read(
        *checkpoint_,
        values_record,
        mlx::core::uint8,
        shape(values_record));
    auto scales = allocate_and_read(
        *checkpoint_,
        scales_record,
        mlx::core::uint8,
        shape(scales_record));
    return MlxMxWeight::from_arrays(
        dtype,
        std::move(values),
        std::move(scales),
        input,
        output);
}

MlxLinear MlxHfTensorStore::load_linear(const std::string& name) const {
    const auto& record = checkpoint_->tensor(name);
    if (record.dtype == "I8" || record.dtype == "F8_E4M3" ||
        record.dtype == "F8_E4M3FN") {
        return MlxLinear(load_mx(name));
    }
    return MlxLinear(load_dense(name));
}

MlxEmbedding MlxHfTensorStore::load_embedding(const std::string& name) const {
    const auto& record = checkpoint_->tensor(name);
    if (record.dtype == "I8" || record.dtype == "F8_E4M3" ||
        record.dtype == "F8_E4M3FN") {
        return MlxEmbedding(load_mx(name));
    }
    return MlxEmbedding(load_dense(name));
}

} // namespace mfq::metal
