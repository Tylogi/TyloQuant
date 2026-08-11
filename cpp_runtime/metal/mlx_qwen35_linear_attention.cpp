#include "mlx_qwen35_linear_attention.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 ||
        value > static_cast<std::int64_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("invalid Qwen3.5 linear-attention ") + name);
    }
    return static_cast<int>(value);
}

array load_dense(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" &&
        record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "Qwen3.5 linear-attention tensor must be dense BF16/F16/F32: " +
            name);
    }
    return load_dense_array(record.dtype, model.read(name));
}

array load_dense_vector(
    const MfqContainer& model,
    const std::string& name) {
    auto result = load_dense(model, name);
    if (result.ndim() != 1) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention tensor must be a vector: " +
            name);
    }
    return result.dtype() == mlx::core::float32
        ? mlx::core::contiguous(result)
        : mlx::core::contiguous(
            mlx::core::astype(result, mlx::core::float32));
}

std::optional<array> load_optional_dense_vector(
    const MfqContainer& model,
    const std::optional<std::string>& name) {
    if (!name.has_value() || !model.contains(*name)) {
        return std::nullopt;
    }
    return load_dense_vector(model, *name);
}

std::optional<MlxLinear> load_optional_linear(
    const MfqContainer& model,
    const std::optional<std::string>& name) {
    if (!name.has_value() || !model.contains(*name)) {
        return std::nullopt;
    }
    return MlxLinear::load(model, *name);
}

bool is_conv_weight_shape(
    const array& weight,
    int channels,
    int kernel) {
    return
        (weight.ndim() == 2 &&
         ((weight.shape(0) == channels &&
           weight.shape(1) == kernel) ||
          (weight.shape(0) == kernel &&
           weight.shape(1) == channels))) ||
        (weight.ndim() == 3 &&
         weight.shape(0) == channels &&
         weight.shape(1) == 1 &&
         weight.shape(2) == kernel);
}

void validate_position_shape(
    const array& positions,
    int tokens) {
    const bool valid_rank =
        positions.ndim() == 1 ||
        (positions.ndim() == 2 &&
         (positions.shape(0) == 1 ||
          positions.shape(0) == 3));
    if (!valid_rank) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention positions must have "
            "[tokens], [1,tokens], or [3,tokens] shape");
    }
    if (positions.shape(-1) != tokens) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention position length must match "
            "the token count");
    }
}

} // namespace

MlxQwen35LinearAttentionBlock
MlxQwen35LinearAttentionBlock::load(
    const MfqContainer& model,
    const Qwen35Config& config,
    const Qwen35TensorNames& names,
    std::size_t layer_index) {
    if (layer_index >=
        static_cast<std::size_t>(config.num_hidden_layers)) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention layer index is out of range");
    }
    if (!config.layer_types.empty() &&
        config.layer_types.at(layer_index) != "linear_attention") {
        throw std::runtime_error(
            "Qwen3.5 layer is not a linear-attention layer");
    }

    const auto resolved = names.layer(layer_index);
    auto split_qk =
        load_optional_linear(model, resolved.linear_qk);
    auto split_value =
        load_optional_linear(model, resolved.linear_value);
    const bool split_input =
        split_qk.has_value() && split_value.has_value();
    if (split_qk.has_value() != split_value.has_value()) {
        throw std::runtime_error(
            "Qwen3.5 split linear-attention input is incomplete");
    }

    std::optional<MlxLinear> combined_qkv;
    if (!split_input) {
        combined_qkv =
            MlxLinear::load(model, resolved.linear_qkv);
    }

    auto convolution_weight =
        load_dense(model, resolved.linear_conv);
    auto convolution_bias =
        load_optional_dense_vector(
            model,
            resolved.linear_conv_bias);
    return MlxQwen35LinearAttentionBlock(
        config,
        names.linear_qkv == "blk.{i}.attn_qkv.weight",
        load_qwen35_rms_norm(
            model,
            resolved.attention_norm,
            config.rms_norm_eps,
            config.norm_weight_offset),
        std::move(combined_qkv),
        std::move(split_qk),
        std::move(split_value),
        MlxLinear::load(model, resolved.linear_z),
        MlxLinear::load(model, resolved.linear_alpha),
        MlxLinear::load(model, resolved.linear_beta),
        std::move(convolution_weight),
        std::move(convolution_bias),
        load_dense_vector(model, resolved.linear_dt_bias),
        load_dense_vector(model, resolved.linear_a),
        MlxRmsNorm(
            load_dense_vector(model, resolved.linear_norm),
            static_cast<float>(config.rms_norm_eps)),
        MlxLinear::load(model, resolved.linear_output),
        load_qwen35_rms_norm(
            model,
            resolved.ffn_norm,
            config.rms_norm_eps,
            config.norm_weight_offset),
        MlxQwen35DenseSwiGlu::load(model, resolved));
}

MlxQwen35LinearAttentionBlock::MlxQwen35LinearAttentionBlock(
    Qwen35Config config,
    bool gguf_layout,
    MlxRmsNorm attention_norm,
    std::optional<MlxLinear> combined_qkv,
    std::optional<MlxLinear> split_qk,
    std::optional<MlxLinear> split_value,
    MlxLinear z,
    MlxLinear alpha,
    MlxLinear beta,
    array convolution_weight,
    std::optional<array> convolution_bias,
    array dt_bias,
    array a,
    MlxRmsNorm linear_norm,
    MlxLinear output,
    MlxRmsNorm ffn_norm,
    MlxQwen35DenseSwiGlu ffn)
    : config_(std::move(config)),
      gguf_layout_(gguf_layout),
      split_input_(
          split_qk.has_value() && split_value.has_value()),
      attention_norm_(std::move(attention_norm)),
      combined_qkv_(std::move(combined_qkv)),
      split_qk_(std::move(split_qk)),
      split_value_(std::move(split_value)),
      z_(std::move(z)),
      alpha_(std::move(alpha)),
      beta_(std::move(beta)),
      convolution_weight_(
          mlx::core::contiguous(std::move(convolution_weight))),
      convolution_bias_(
          convolution_bias.has_value()
              ? std::optional<array>(
                    mlx::core::contiguous(
                        std::move(*convolution_bias)))
              : std::nullopt),
      dt_bias_(
          mlx::core::contiguous(
              mlx::core::astype(
                  std::move(dt_bias),
                  mlx::core::float32))),
      a_(
          mlx::core::contiguous(
              mlx::core::astype(
                  std::move(a),
                  mlx::core::float32))),
      linear_norm_(std::move(linear_norm)),
      output_(std::move(output)),
      ffn_norm_(std::move(ffn_norm)),
      ffn_(std::move(ffn)) {
    validate_components();

    // ``a`` is a model constant.  Keeping the log-to-decay conversion in
    // forward() rebuilt 48 Exp/Negative primitives for every decode token.
    // Materialize the converted value once and reuse it across the lifetime
    // of the block instead.
    if (config_.linear_a_is_log) {
        a_ = mlx::core::contiguous(
            -mlx::core::exp(a_));
        a_.eval();
    }

    const auto* alpha_weight = alpha_->dense_weight_ref();
    const auto* beta_weight = beta_->dense_weight_ref();
    if (alpha_weight != nullptr &&
        beta_weight != nullptr &&
        alpha_weight->dtype() == beta_weight->dtype()) {
        alpha_beta_.emplace(
            mlx::core::contiguous(
                mlx::core::concatenate(
                    {*alpha_weight, *beta_weight},
                    0)));
        alpha_.reset();
        beta_.reset();
    }
}

void MlxQwen35LinearAttentionBlock::
validate_components() const {
    const int hidden =
        checked_int(config_.hidden_size, "hidden_size");
    const int intermediate =
        checked_int(
            config_.intermediate_size,
            "intermediate_size");
    const int key_heads =
        checked_int(
            config_.linear_key_heads(),
            "linear key head count");
    const int value_heads =
        checked_int(
            config_.linear_value_heads(),
            "linear value head count");
    const int key_dimension =
        checked_int(
            config_.linear_key_head_dim,
            "linear key head dimension");
    const int value_dimension =
        checked_int(
            config_.linear_value_head_dim,
            "linear value head dimension");
    const int key_size =
        checked_int(
            config_.linear_key_size(),
            "linear key size");
    const int value_size =
        checked_int(
            config_.linear_value_size(),
            "linear value size");
    const int channels =
        checked_int(
            config_.linear_qkv_size(),
            "linear QKV size");
    const int kernel =
        checked_int(
            config_.linear_conv_kernel_dim,
            "linear convolution width");
    checked_int(
        config_.max_position_embeddings,
        "maximum position count");

    if (kernel < 2 ||
        key_dimension != value_dimension ||
        value_heads % key_heads != 0 ||
        !std::isfinite(config_.rms_norm_eps) ||
        config_.rms_norm_eps <= 0.0) {
        throw std::runtime_error(
            "invalid Qwen3.5 Gated DeltaNet configuration");
    }

    const bool has_combined = combined_qkv_.has_value();
    const bool has_split_qk = split_qk_.has_value();
    const bool has_split_value = split_value_.has_value();
    if (has_split_qk != has_split_value ||
        split_input_ != (has_split_qk && has_split_value) ||
        has_combined == split_input_) {
        throw std::runtime_error(
            "Qwen3.5 Gated DeltaNet requires exactly one "
            "combined or split input projection layout");
    }
    if (has_combined &&
        (combined_qkv_->input_size() != hidden ||
         combined_qkv_->output_size() != channels)) {
        throw std::runtime_error(
            "Qwen3.5 combined QKV projection size mismatch");
    }
    if (split_input_ &&
        (split_qk_->input_size() != hidden ||
         split_qk_->output_size() != 2 * key_size ||
         split_value_->input_size() != hidden ||
         split_value_->output_size() != value_size)) {
        throw std::runtime_error(
            "Qwen3.5 split QK/V projection size mismatch");
    }

    if (attention_norm_.width() != hidden ||
        ffn_norm_.width() != hidden ||
        z_.input_size() != hidden ||
        z_.output_size() != value_size ||
        alpha_->input_size() != hidden ||
        alpha_->output_size() != value_heads ||
        beta_->input_size() != hidden ||
        beta_->output_size() != value_heads ||
        dt_bias_.ndim() != 1 ||
        dt_bias_.shape(0) != value_heads ||
        a_.ndim() != 1 ||
        a_.shape(0) != value_heads ||
        linear_norm_.width() != value_dimension ||
        output_.input_size() != value_size ||
        output_.output_size() != hidden ||
        ffn_.input_size() != hidden ||
        ffn_.intermediate_size() != intermediate ||
        ffn_.output_size() != hidden) {
        throw std::runtime_error(
            "Qwen3.5 Gated DeltaNet component size mismatch");
    }
    if (!is_conv_weight_shape(
            convolution_weight_,
            channels,
            kernel) ||
        (convolution_bias_.has_value() &&
         (convolution_bias_->ndim() != 1 ||
          convolution_bias_->shape(0) != channels))) {
        throw std::runtime_error(
            "Qwen3.5 Gated DeltaNet convolution size mismatch");
    }
}

void MlxQwen35LinearAttentionBlock::reset_cache(int batch) {
    if (batch <= 0) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention cache batch must be positive");
    }
    const int channels =
        checked_int(
            config_.linear_qkv_size(),
            "linear QKV size");
    const int kernel =
        checked_int(
            config_.linear_conv_kernel_dim,
            "linear convolution width");
    const int value_heads =
        checked_int(
            config_.linear_value_heads(),
            "linear value head count");
    const int dimension =
        checked_int(
            config_.linear_value_head_dim,
            "linear value head dimension");
    if (!zero_convolution_state_ ||
        !zero_recurrent_state_ ||
        zero_cache_batch_ != batch) {
        zero_convolution_state_ = mlx::core::zeros(
            Shape{batch, kernel - 1, channels},
            mlx::core::float32);
        zero_recurrent_state_ = mlx::core::zeros(
            Shape{batch, value_heads, dimension, dimension},
            mlx::core::float32);
        zero_cache_batch_ = batch;
    }
    convolution_state_ = *zero_convolution_state_;
    recurrent_state_ = *zero_recurrent_state_;
    cache_position_ = 0;
    cache_batch_ = batch;
}

void MlxQwen35LinearAttentionBlock::materialize_cache() {
    if (convolution_state_ && recurrent_state_) {
        mlx::core::eval(
            *convolution_state_,
            *recurrent_state_);
    }
}

void MlxQwen35LinearAttentionBlock::clear_cache() noexcept {
    convolution_state_.reset();
    recurrent_state_.reset();
    zero_convolution_state_.reset();
    zero_recurrent_state_.reset();
    zero_cache_batch_ = 0;
    cache_position_ = 0;
    cache_batch_ = 0;
}

MlxQwen35LinearAttentionCacheSnapshot
MlxQwen35LinearAttentionBlock::snapshot_cache() const {
    if (!convolution_state_ || !recurrent_state_ ||
        cache_position_ <= 0 || cache_batch_ <= 0) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention cache is unavailable");
    }
    auto convolution = mlx::core::astype(
        *convolution_state_, convolution_state_->dtype(), true);
    auto recurrent = mlx::core::astype(
        *recurrent_state_, recurrent_state_->dtype(), true);
    mlx::core::eval(convolution, recurrent);
    return {
        std::move(convolution),
        std::move(recurrent),
        cache_position_,
        cache_batch_,
    };
}

void MlxQwen35LinearAttentionBlock::restore_cache(
    const MlxQwen35LinearAttentionCacheSnapshot& snapshot) {
    const int channels = static_cast<int>(config_.linear_qkv_size());
    const int kernel = static_cast<int>(config_.linear_conv_kernel_dim);
    const int value_heads = static_cast<int>(config_.linear_value_heads());
    const int dimension = static_cast<int>(config_.linear_value_head_dim);
    if (snapshot.batch <= 0 || snapshot.position <= 0 ||
        snapshot.position > config_.max_position_embeddings ||
        snapshot.convolution_state.shape() !=
            Shape{snapshot.batch, kernel - 1, channels} ||
        snapshot.recurrent_state.shape() !=
            Shape{snapshot.batch, value_heads, dimension, dimension} ||
        snapshot.convolution_state.dtype() != mlx::core::float32 ||
        snapshot.recurrent_state.dtype() != mlx::core::float32) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention cache snapshot topology mismatch");
    }
    auto convolution = mlx::core::astype(
        snapshot.convolution_state, mlx::core::float32, true);
    auto recurrent = mlx::core::astype(
        snapshot.recurrent_state, mlx::core::float32, true);
    mlx::core::eval(convolution, recurrent);
    convolution_state_ = std::move(convolution);
    recurrent_state_ = std::move(recurrent);
    cache_position_ = snapshot.position;
    cache_batch_ = snapshot.batch;
}

array MlxQwen35LinearAttentionBlock::forward(
    const array& input,
    bool use_cache) {
    if (input.ndim() != 3 ||
        input.shape(0) <= 0 ||
        input.shape(1) <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention input must have a non-empty "
            "[batch,tokens,hidden] shape");
    }
    if (use_cache && !convolution_state_) {
        reset_cache(input.shape(0));
    }
    return forward(
        input,
        use_cache ? cache_position_ : 0,
        use_cache);
}

array MlxQwen35LinearAttentionBlock::forward(
    const array& input,
    int position_offset,
    bool use_cache) {
    if (input.ndim() != 3 ||
        input.shape(0) <= 0 ||
        input.shape(1) <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention input must have a non-empty "
            "[batch,tokens,hidden] shape");
    }
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    if (position_offset < 0 ||
        position_offset >
            config_.max_position_embeddings - tokens) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention position range is invalid");
    }

    if (use_cache) {
        if (!convolution_state_ || !recurrent_state_) {
            if (position_offset != 0) {
                throw std::runtime_error(
                    "Qwen3.5 linear-attention cache must start "
                    "at position zero");
            }
            reset_cache(batch);
        }
        if (cache_batch_ != batch) {
            throw std::runtime_error(
                "Qwen3.5 linear-attention cache batch mismatch");
        }
        if (cache_position_ != position_offset) {
            throw std::runtime_error(
                "Qwen3.5 linear-attention cache append must "
                "be contiguous");
        }
    }

    const int key_size =
        static_cast<int>(config_.linear_key_size());
    const int value_size =
        static_cast<int>(config_.linear_value_size());
    const int channels =
        static_cast<int>(config_.linear_qkv_size());
    const int key_heads =
        static_cast<int>(config_.linear_key_heads());
    const int value_heads =
        static_cast<int>(config_.linear_value_heads());
    const int dimension =
        static_cast<int>(config_.linear_value_head_dim);
    const int kernel =
        static_cast<int>(config_.linear_conv_kernel_dim);
    const auto activation_dtype = input.dtype();

    const auto normalized = attention_norm_(input);
    array qk = normalized;
    array value_input = normalized;
    array z = normalized;
    if (split_input_) {
        qk = (*split_qk_)(normalized);
        value_input = (*split_value_)(normalized);
        z = z_(normalized);
    } else {
        const auto projected = (*combined_qkv_)(normalized);
        auto pieces = mlx::core::split(
            projected,
            Shape{2 * key_size},
            -1);
        qk = std::move(pieces.at(0));
        value_input = std::move(pieces.at(1));
        z = z_(normalized);
    }

    array alpha = normalized;
    array beta = normalized;
    if (alpha_beta_) {
        auto projected = (*alpha_beta_)(normalized);
        auto pieces = mlx::core::split(
            projected,
            Shape{value_heads},
            -1);
        alpha = mlx::core::reshape(
            mlx::core::astype(
                pieces.at(0),
                mlx::core::float32),
            Shape{batch, tokens, value_heads});
        beta = mlx::core::reshape(
            mlx::core::sigmoid(
                mlx::core::astype(
                    pieces.at(1),
                    mlx::core::float32)),
            Shape{batch, tokens, value_heads});
    } else {
        alpha = mlx::core::reshape(
            mlx::core::astype(
                (*alpha_)(normalized),
                mlx::core::float32),
            Shape{batch, tokens, value_heads});
        beta = mlx::core::reshape(
            mlx::core::sigmoid(
                mlx::core::astype(
                    (*beta_)(normalized),
                    mlx::core::float32)),
            Shape{batch, tokens, value_heads});
    }

    const auto gate_input =
        alpha +
        mlx::core::reshape(
            dt_bias_,
            Shape{1, 1, value_heads});
    const auto softplus =
        mlx::core::maximum(
            gate_input,
            array(0.0f)) +
        mlx::core::log1p(
            mlx::core::exp(
                -mlx::core::abs(gate_input)));
    const auto gate =
        softplus *
        mlx::core::reshape(a_, Shape{1, 1, value_heads});

    const auto zero_convolution_state =
        mlx::core::zeros(
            Shape{batch, kernel - 1, channels},
            mlx::core::float32);
    const auto& convolution_state = use_cache
        ? *convolution_state_
        : zero_convolution_state;
    auto convolved = linear_conv_qkv(
        convolution_state,
        qk,
        value_input,
        convolution_weight_,
        key_heads,
        value_heads,
        static_cast<int>(config_.linear_key_head_dim),
        dimension,
        convolution_bias_,
        static_cast<float>(config_.rms_norm_eps));
    auto recurrent = gated_delta_net(
        convolved.query,
        convolved.key,
        convolved.value,
        mlx::core::transpose(gate, {0, 2, 1}),
        mlx::core::transpose(beta, {0, 2, 1}),
        use_cache ? recurrent_state_ : std::nullopt,
        false,
        gguf_layout_);

    if (use_cache) {
        convolution_state_ = convolved.state;
        recurrent_state_ = recurrent.state;
        cache_position_ += tokens;
    }

    auto normalized_value = linear_norm_(recurrent.output);
    normalized_value = mlx::core::reshape(
        mlx::core::transpose(
            normalized_value,
            {0, 2, 1, 3}),
        Shape{batch, tokens, value_size});
    z = mlx::core::reshape(
        z,
        Shape{batch, tokens, value_size});
    const auto gated_value =
        normalized_value *
        (z * mlx::core::sigmoid(z));
    auto residual = input + output_(
        mlx::core::astype(
            gated_value,
            activation_dtype));
    residual = residual + ffn_(ffn_norm_(residual));
    return residual;
}

array MlxQwen35LinearAttentionBlock::forward(
    const array& input,
    const array& positions,
    bool use_cache) {
    if (input.ndim() != 3 ||
        input.shape(0) <= 0 ||
        input.shape(1) <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention input must have a non-empty "
            "[batch,tokens,hidden] shape");
    }
    validate_position_shape(positions, input.shape(1));
    if (use_cache && !convolution_state_) {
        reset_cache(input.shape(0));
    }
    return forward(
        input,
        positions,
        use_cache ? cache_position_ : 0,
        use_cache);
}

array MlxQwen35LinearAttentionBlock::forward(
    const array& input,
    const array& positions,
    int position_offset,
    bool use_cache) {
    if (input.ndim() != 3 ||
        input.shape(0) <= 0 ||
        input.shape(1) <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention input must have a non-empty "
            "[batch,tokens,hidden] shape");
    }
    validate_position_shape(positions, input.shape(1));
    return forward(input, position_offset, use_cache);
}

} // namespace mfq::metal
