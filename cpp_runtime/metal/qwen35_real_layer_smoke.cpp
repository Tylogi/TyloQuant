#include "mlx_qwen35_full_attention.h"
#include "mlx_qwen35_linear_attention.h"
#include "qwen35_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/memory.h>
#include <mlx/mlx.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MfqContainer;
using mfq::metal::MlxQwen35FullAttentionBlock;
using mfq::metal::MlxQwen35LinearAttentionBlock;
using mfq::metal::Qwen35Config;
using mfq::metal::Qwen35ResolvedLayerNames;
using mfq::metal::Qwen35TensorNames;

constexpr std::size_t kLinearLayer = 0;
constexpr std::size_t kFullLayer = 3;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_memory(const std::string& label) {
    std::cout
        << label
        << ": active=" << std::fixed << std::setprecision(1)
        << mib(mlx::core::get_active_memory()) << " MiB"
        << ", cache=" << mib(mlx::core::get_cache_memory()) << " MiB"
        << ", peak=" << mib(mlx::core::get_peak_memory()) << " MiB\n";
}

array token_input(int hidden, int token_index) {
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(hidden));
    for (int index = 0; index < hidden; ++index) {
        const int centered =
            (index * 17 + token_index * 11) % 127 - 63;
        values.push_back(
            static_cast<float>(centered) / 256.0f);
    }
    return mlx::core::astype(
        array(values.begin(), Shape{1, 1, hidden}),
        mlx::core::float16);
}

struct OutputSummary {
    float maximum_absolute = 0.0f;
    float l1 = 0.0f;
};

OutputSummary validate_output(
    const array& output,
    const array& input,
    int hidden,
    const std::string& label) {
    require(
        output.shape() == Shape{1, 1, hidden},
        label + " output shape mismatch");
    auto checked = mlx::core::astype(
        output,
        mlx::core::float32);
    auto checked_input = mlx::core::astype(
        input,
        mlx::core::float32);
    mlx::core::eval(checked, checked_input);

    const auto* values = checked.data<float>();
    const auto* input_values = checked_input.data<float>();
    OutputSummary summary;
    bool changed = false;
    for (int index = 0; index < hidden; ++index) {
        const float value = values[index];
        require(
            std::isfinite(value),
            label + " produced a non-finite value");
        summary.maximum_absolute =
            std::max(summary.maximum_absolute, std::fabs(value));
        summary.l1 += std::fabs(value);
        changed = changed ||
            std::fabs(value - input_values[index]) > 1e-7f;
    }
    require(
        summary.maximum_absolute > 1e-8f && summary.l1 > 1e-6f,
        label + " produced an all-zero output");
    require(
        changed,
        label + " did not change the residual");
    return summary;
}

void print_tensor(
    const MfqContainer& model,
    const std::string& name) {
    const auto metadata =
        mfq::metal::inspect_qwen35_tensor_metadata(model, name);
    std::cout << "  " << name << ": " << metadata.dtype << " [";
    for (std::size_t index = 0;
         index < metadata.shape.size();
         ++index) {
        if (index != 0) {
            std::cout << ",";
        }
        std::cout << metadata.shape[index];
    }
    std::cout << "]\n";
}

std::set<std::string> selected_dtypes(
    const MfqContainer& model,
    const Qwen35ResolvedLayerNames& linear,
    const Qwen35ResolvedLayerNames& full) {
    std::set<std::string> result;
    for (const auto* name : {
             &linear.linear_qkv,
             &linear.linear_z,
             &linear.linear_output,
             &linear.ffn_gate,
             &linear.ffn_up,
             &linear.ffn_down,
             &full.attention_query,
             &full.attention_key,
             &full.attention_value,
             &full.attention_output,
             &full.ffn_gate,
             &full.ffn_up,
             &full.ffn_down,
         }) {
        result.insert(model.record(*name).dtype);
    }
    return result;
}

void run_linear_layer(
    const MfqContainer& model,
    const Qwen35Config& config,
    const Qwen35TensorNames& names) {
    const auto resolved = names.layer(kLinearLayer);
    std::cout << "linear-attention layer " << kLinearLayer << ":\n";
    print_tensor(model, resolved.linear_qkv);
    print_tensor(model, resolved.linear_z);
    print_tensor(model, resolved.linear_output);
    print_tensor(model, resolved.linear_conv);
    print_tensor(model, resolved.ffn_gate);
    print_tensor(model, resolved.ffn_down);

    mlx::core::reset_peak_memory();
    print_memory("  before load");
    {
        auto block =
            MlxQwen35LinearAttentionBlock::load(
                model,
                config,
                names,
                kLinearLayer);
        require(
            block.uses_grouped_ffn(),
            "real linear-attention FFN missed grouped gate/up");
        require(
            block.uses_combined_alpha_beta(),
            "real linear-attention layer missed combined alpha/beta");
        const auto first_input =
            token_input(static_cast<int>(config.hidden_size), 0);
        const auto second_input =
            token_input(static_cast<int>(config.hidden_size), 1);
        auto prefill = block.forward(first_input, 0, true);
        const auto prefill_summary = validate_output(
            prefill,
            first_input,
            static_cast<int>(config.hidden_size),
            "linear-attention prefill");
        require(
            prefill.dtype() == mlx::core::float16,
            "real linear-attention block promoted FP16 output");
        auto decode = block.forward(second_input, 1, true);
        require(
            block.convolution_state().has_value() &&
                block.recurrent_state().has_value(),
            "linear-attention cache tensors are missing");
        require(
            block.convolution_state()->dtype() ==
                    mlx::core::float32 &&
                block.recurrent_state()->dtype() ==
                    mlx::core::float32,
            "real linear-attention cache/state must remain FP32");
        mlx::core::eval(
            decode,
            *block.convolution_state(),
            *block.recurrent_state());
        const auto decode_summary = validate_output(
            decode,
            second_input,
            static_cast<int>(config.hidden_size),
            "linear-attention decode");
        require(
            block.cache_position() == 2,
            "linear-attention cache position mismatch");
        require(
            block.convolution_state()->shape() ==
                Shape{
                    1,
                    static_cast<int>(
                        config.linear_conv_kernel_dim - 1),
                    static_cast<int>(config.linear_qkv_size()),
                },
            "linear-attention convolution cache shape mismatch");
        require(
            block.recurrent_state()->shape() ==
                Shape{
                    1,
                    static_cast<int>(config.linear_value_heads()),
                    static_cast<int>(config.linear_value_head_dim),
                    static_cast<int>(config.linear_value_head_dim),
                },
            "linear-attention recurrent cache shape mismatch");
        std::cout
            << "  prefill max|x|=" << prefill_summary.maximum_absolute
            << ", decode max|x|=" << decode_summary.maximum_absolute
            << ", cache_position=" << block.cache_position() << "\n";
        print_memory("  evaluated");
    }
    mlx::core::clear_cache();
    print_memory("  released");
}

void run_full_layer(
    const MfqContainer& model,
    const Qwen35Config& config,
    const Qwen35TensorNames& names) {
    const auto resolved = names.layer(kFullLayer);
    std::cout << "full-attention layer " << kFullLayer << ":\n";
    print_tensor(model, resolved.attention_query);
    print_tensor(model, resolved.attention_key);
    print_tensor(model, resolved.attention_value);
    print_tensor(model, resolved.attention_output);
    print_tensor(model, resolved.ffn_gate);
    print_tensor(model, resolved.ffn_down);

    mlx::core::reset_peak_memory();
    print_memory("  before load");
    {
        auto block =
            MlxQwen35FullAttentionBlock::load(
                model,
                config,
                names,
                kFullLayer);
        require(
            block.uses_grouped_qkv(),
            "real full-attention layer missed grouped QKV");
        require(
            block.uses_grouped_ffn(),
            "real full-attention FFN missed grouped gate/up");
        const auto first_input =
            token_input(static_cast<int>(config.hidden_size), 2);
        const auto second_input =
            token_input(static_cast<int>(config.hidden_size), 3);
        auto prefill = block.forward(first_input, 0, true);
        const auto prefill_summary = validate_output(
            prefill,
            first_input,
            static_cast<int>(config.hidden_size),
            "full-attention prefill");
        auto decode = block.forward(second_input, 1, true);
        const auto decode_summary = validate_output(
            decode,
            second_input,
            static_cast<int>(config.hidden_size),
            "full-attention decode");
        require(
            block.cache_position() == 2,
            "full-attention cache position mismatch");
        std::cout
            << "  prefill max|x|=" << prefill_summary.maximum_absolute
            << ", decode max|x|=" << decode_summary.maximum_absolute
            << ", cache_position=" << block.cache_position() << "\n";
        print_memory("  evaluated");
    }
    mlx::core::clear_cache();
    print_memory("  released");
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc == 2,
            "usage: qwen35_real_layer_smoke model.mfq");
#ifdef MFQ_MLX_METALLIB_DEFAULT
        mlx::core::metal::set_metallib_path(
            MFQ_MLX_METALLIB_DEFAULT);
#endif
        require(
            mlx::core::is_available(mlx::core::Device::gpu),
            "no Apple GPU is available");
        mlx::core::set_default_device(mlx::core::Device::gpu);

        const MfqContainer model(argv[1]);
        const auto names =
            Qwen35TensorNames::detect(model);
        const auto config =
            mfq::metal::adapt_qwen35_config_for_tensor_names(
                Qwen35Config::from_mfq(model),
                names);
        require(
            config.hidden_size == 5120,
            "smoke model hidden size is not 5120");
        require(
            config.layer_types.at(kLinearLayer) ==
                "linear_attention",
            "layer 0 is not linear attention");
        require(
            config.layer_types.at(kFullLayer) ==
                "full_attention",
            "layer 3 is not full attention");

        const auto linear = names.layer(kLinearLayer);
        const auto full = names.layer(kFullLayer);
        const auto dtypes =
            selected_dtypes(model, linear, full);
        for (const auto* required_dtype : {
                 "NINT4",
                 "NINT5",
                 "NINT6",
                 "NINT8-0",
             }) {
            require(
                dtypes.contains(required_dtype),
                std::string("selected layers do not cover ") +
                    required_dtype);
        }

        std::cout
            << "model=" << argv[1] << "\n"
            << "architecture=" << model.header().architecture
            << ", layers=" << config.num_hidden_layers
            << ", hidden=" << config.hidden_size
            << ", linear_qkv=" << config.linear_qkv_size()
            << ", attention=" << config.attention_size() << "\n";
        print_memory("initial");
        run_linear_layer(model, config, names);
        run_full_layer(model, config, names);
        mlx::core::clear_cache();
        print_memory("final");
        std::cout << "Qwen3.5 real packed layer smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
