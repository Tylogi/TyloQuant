#include "mlx_deepseek_v41_mhc.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

array dense_array(const MfqContainer& model, const std::string& name) {
    const auto mapped = model.map_record(name);
    return load_dense_array(model.record(name).dtype, mapped.view());
}

} // namespace

MlxDeepseekV41Mhc MlxDeepseekV41Mhc::load(
    const MfqContainer& model,
    const DeepseekV41Config& config,
    const std::string& prefix,
    const std::string& norm_name) {
    return MlxDeepseekV41Mhc(
        config,
        MlxLinear::load(model, prefix + ".function"),
        dense_array(model, prefix + ".base"),
        dense_array(model, prefix + ".scale"),
        dense_array(model, norm_name));
}

MlxDeepseekV41Mhc::MlxDeepseekV41Mhc(
    DeepseekV41Config config,
    MlxLinear function,
    array base,
    array scale,
    array norm)
    : config_(std::move(config)),
      function_(std::move(function)),
      base_(mlx::core::reshape(
          mlx::core::astype(base, mlx::core::float32), Shape{24})),
      scale_(mlx::core::reshape(
          mlx::core::astype(scale, mlx::core::float32), Shape{3})),
      norm_(std::move(norm), static_cast<float>(config_.rms_eps)) {
    if (config_.hc_mult != 4 || function_.input_size() != 4 * config_.hidden ||
        function_.output_size() != 24) {
        throw std::runtime_error("DeepSeek-V4.1 Mega-mHC geometry disagrees");
    }
}

MlxDeepseekV41MhcResult MlxDeepseekV41Mhc::collapse(
    const array& residual,
    const array& previous_pre) const {
    if (residual.ndim() != 4 || residual.shape(0) <= 0 ||
        residual.shape(1) <= 0 || residual.shape(2) != 4 ||
        residual.shape(3) != config_.hidden ||
        previous_pre.shape() !=
            Shape{residual.shape(0), residual.shape(1), 4}) {
        throw std::invalid_argument("DeepSeek-V4.1 Mega-mHC input mismatch");
    }
    const int batch = residual.shape(0);
    const int tokens = residual.shape(1);
    auto flattened = mlx::core::reshape(
        mlx::core::astype(residual, mlx::core::float32),
        Shape{batch, tokens, static_cast<int>(4 * config_.hidden)});
    auto inverse = mlx::core::rsqrt(
        mlx::core::mean(flattened * flattened, -1, true) +
        static_cast<float>(config_.rms_eps));
    auto mixes = mlx::core::astype(
        function_(flattened * inverse), mlx::core::float32);
    auto pre_raw = mlx::core::slice(
        mixes, Shape{0, 0, 0}, Shape{batch, tokens, 4});
    auto pre_scale = mlx::core::slice(scale_, Shape{0}, Shape{1});
    auto pre_base = mlx::core::slice(base_, Shape{0}, Shape{4});
    auto next_pre = mlx::core::sigmoid(pre_raw * pre_scale + pre_base) +
        static_cast<float>(config_.hc_eps);

    auto expansion = deepseek_v4_hc_pre(
        residual,
        mixes,
        scale_,
        base_,
        static_cast<int>(config_.hc_sinkhorn_iters),
        static_cast<float>(config_.hc_eps));
    auto branch = mlx::core::sum(
        mlx::core::expand_dims(
            mlx::core::astype(previous_pre, mlx::core::float32), -1) *
            mlx::core::astype(residual, mlx::core::float32),
        2);
    branch = norm_(mlx::core::astype(branch, residual.dtype()));
    return {
        std::move(branch),
        std::move(next_pre),
        std::move(expansion),
    };
}

array MlxDeepseekV41Mhc::expand(
    const array& branch,
    const array& residual,
    const MlxDeepseekV4HcPreResult& expansion) const {
    if (expansion.packed_metadata.has_value()) {
        return deepseek_v4_hc_post_packed(
            branch, residual, *expansion.packed_metadata);
    }
    return deepseek_v4_hc_post(
        branch,
        residual,
        expansion.post,
        expansion.combination);
}

array MlxDeepseekV41Mhc::identity_pre(int batch, int tokens) {
    if (batch <= 0 || tokens <= 0) {
        throw std::invalid_argument("DeepSeek-V4.1 identity pre-mix is empty");
    }
    const array first({1.0f, 0.0f, 0.0f, 0.0f}, Shape{1, 1, 4});
    return mlx::core::broadcast_to(first, Shape{batch, tokens, 4});
}

} // namespace mfq::metal
