#include "mfq_container.h"
#include "mlx_grouped_linear.h"
#include "mlx_legacy_tensor_compat.h"
#include "mlx_tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/stream.h>

namespace {

using Clock = std::chrono::steady_clock;
using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MfqContainer;
using mfq::metal::MlxGroupedLinear;
using mfq::metal::MlxGroupedLinearWeightRef;
using mfq::metal::MlxLinear;

using Projection = std::function<std::vector<array>(const array&)>;

int g_profile_rows = 1;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

array make_input(int width) {
    std::vector<float> values(
        static_cast<std::size_t>(g_profile_rows)
            * static_cast<std::size_t>(width));
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>((index * 17u + 11u) % 127u) / 256.0f
            - 0.24609375f;
    }
    return mlx::core::astype(
        array(values.begin(), Shape{1, g_profile_rows, width}),
        mlx::core::float16);
}

double elapsed_ms(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
}

std::vector<MlxGroupedLinearWeightRef> grouped_refs(
    const std::vector<const MlxLinear*>& linears) {
    std::vector<MlxGroupedLinearWeightRef> refs;
    refs.reserve(linears.size());
    for (const auto* linear : linears) {
        const auto ref = linear->grouped_weight_ref();
        require(ref.has_value(), "profile group contains a dense weight");
        refs.push_back(*ref);
    }
    return refs;
}

std::size_t packed_bytes(const MlxLinear& linear) {
    if (const auto ref = linear.grouped_weight_ref()) {
        return std::visit(
            [](const auto* weight) {
                return weight->packed_nbytes();
            },
            *ref);
    }
    if (const auto* dense = linear.dense_weight_ref()) {
        return dense->nbytes();
    }
    throw std::runtime_error(
        "profile linear has no accessible storage");
}

void profile(
    const std::string& label,
    int input_width,
    std::size_t packed_bytes,
    int dispatches,
    const Projection& projection,
    int warmup,
    int repetitions) {
    const auto input = make_input(input_width);
    auto outputs = projection(input);
    mlx::core::eval(outputs);
    for (int index = 0; index < warmup; ++index) {
        outputs = projection(input);
        mlx::core::eval(outputs);
    }
    mlx::core::synchronize();

    const auto started = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        outputs = projection(input);
        mlx::core::eval(outputs);
    }
    mlx::core::synchronize();
    const double mean_ms =
        elapsed_ms(started) / static_cast<double>(repetitions);
    const double gbps =
        static_cast<double>(packed_bytes) / (mean_ms * 1.0e6);

    std::cout
        << label << '\t'
        << packed_bytes << '\t'
        << dispatches << '\t'
        << std::fixed << std::setprecision(4) << mean_ms << '\t'
        << std::setprecision(1) << gbps << '\n';
}

void profile_full_qkv(
    const MfqContainer& model,
    int layer,
    const std::string& prefix,
    int warmup,
    int repetitions) {
    const auto block = "model.block." + std::to_string(layer);
    const std::vector<std::string> names{
        block + ".attention.query.weight",
        block + ".attention.key.weight",
        block + ".attention.value.weight",
    };
    auto q = MlxLinear::load(model, names[0]);
    auto k = MlxLinear::load(model, names[1]);
    auto v = MlxLinear::load(model, names[2]);
    const auto bytes =
        packed_bytes(q) + packed_bytes(k) + packed_bytes(v);
    const int width = q.input_size();

    profile(
        prefix + "_separate",
        width,
        bytes,
        3,
        [&](const array& input) {
            return std::vector<array>{q(input), k(input), v(input)};
        },
        warmup,
        repetitions);

    if (g_profile_rows <= MlxGroupedLinear::max_rows()) {
        MlxGroupedLinear grouped(grouped_refs({&q, &k, &v}));
        profile(
            prefix + "_grouped",
            width,
            grouped.packed_nbytes(),
            1,
            [&](const array& input) {
                return grouped(input);
            },
            warmup,
            repetitions);
    }
}

void profile_gate_up(
    const MfqContainer& model,
    int layer,
    const std::string& prefix,
    int warmup,
    int repetitions) {
    const auto block = "model.block." + std::to_string(layer);
    const std::vector<std::string> names{
        block + ".mlp.gate.weight",
        block + ".mlp.up.weight",
    };
    auto gate = MlxLinear::load(model, names[0]);
    auto up = MlxLinear::load(model, names[1]);
    const auto bytes = packed_bytes(gate) + packed_bytes(up);
    const int width = gate.input_size();

    profile(
        prefix + "_separate",
        width,
        bytes,
        3,
        [&](const array& input) {
            const auto gate_value = gate(input);
            const auto up_value = up(input);
            return std::vector<array>{
                gate_value * mlx::core::sigmoid(gate_value) * up_value,
            };
        },
        warmup,
        repetitions);

    if (g_profile_rows <= MlxGroupedLinear::max_rows()) {
        MlxGroupedLinear grouped(grouped_refs({&gate, &up}));
        profile(
            prefix + "_grouped",
            width,
            grouped.packed_nbytes(),
            2,
            [&](const array& input) {
                const auto values = grouped(input);
                return std::vector<array>{
                    values.at(0) *
                        mlx::core::sigmoid(values.at(0)) *
                        values.at(1),
                };
            },
            warmup,
            repetitions);
    }

    const auto* gate_nint = gate.nint_weight_ref();
    const auto* up_nint = up.nint_weight_ref();
    if (g_profile_rows == 1 &&
        gate_nint != nullptr &&
        up_nint != nullptr &&
        gate_nint->can_fuse_swiglu(*up_nint)) {
        const auto parity_input = make_input(width);
        const auto gate_value = gate(parity_input);
        const auto up_value = up(parity_input);
        auto reference = mlx::core::astype(
            gate_value *
                mlx::core::sigmoid(gate_value) *
                up_value,
            mlx::core::float32);
        auto fused = mlx::core::astype(
            gate_nint->swiglu(*up_nint, parity_input),
            mlx::core::float32);
        mlx::core::eval(reference, fused);
        const auto* expected = reference.data<float>();
        const auto* actual = fused.data<float>();
        double difference_squared = 0.0;
        double reference_squared = 0.0;
        float maximum_absolute = 0.0f;
        for (std::size_t index = 0; index < reference.size(); ++index) {
            const float difference = actual[index] - expected[index];
            maximum_absolute = std::max(
                maximum_absolute,
                std::fabs(difference));
            difference_squared +=
                static_cast<double>(difference) * difference;
            reference_squared +=
                static_cast<double>(expected[index]) * expected[index];
        }
        std::cerr
            << prefix << "_fused_parity"
            << "\tmax_abs=" << maximum_absolute
            << "\trel_l2="
            << std::sqrt(
                   difference_squared /
                   std::max(reference_squared, 1.0e-30))
            << '\n';
        profile(
            prefix + "_fused_nint4",
            width,
            bytes,
            1,
            [&](const array& input) {
                return std::vector<array>{
                    gate_nint->swiglu(*up_nint, input),
                };
            },
            warmup,
            repetitions);
    }
}

void profile_single(
    const MfqContainer& model,
    const std::string& label,
    const std::string& name,
    int warmup,
    int repetitions) {
    auto linear = MlxLinear::load(model, name);
    profile(
        label,
        linear.input_size(),
        packed_bytes(linear),
        1,
        [&](const array& input) {
            return std::vector<array>{linear(input)};
        },
        warmup,
        repetitions);
}

void compare_packed_and_transient(
    const MfqContainer& model,
    const std::string& label,
    const std::string& name) {
    auto linear = MlxLinear::load(model, name);
    const auto input = make_input(linear.input_size());
    ::setenv("MFQ_METAL_NINT_PREFILL_PACKED_MMQ", "1", 1);
    auto packed = mlx::core::astype(linear(input), mlx::core::float32);
    ::setenv("MFQ_METAL_NINT_PREFILL_PACKED_MMQ", "0", 1);
    auto transient = mlx::core::astype(linear(input), mlx::core::float32);
    ::unsetenv("MFQ_METAL_NINT_PREFILL_PACKED_MMQ");
    mlx::core::eval(packed, transient);

    const auto* expected = packed.data<float>();
    const auto* actual = transient.data<float>();
    double difference_squared = 0.0;
    double reference_squared = 0.0;
    float maximum_absolute = 0.0f;
    for (std::size_t index = 0; index < packed.size(); ++index) {
        const float difference = actual[index] - expected[index];
        maximum_absolute = std::max(maximum_absolute, std::fabs(difference));
        difference_squared += static_cast<double>(difference) * difference;
        reference_squared +=
            static_cast<double>(expected[index]) * expected[index];
    }
    std::cout
        << label
        << "\tmax_abs=" << maximum_absolute
        << "\trel_l2="
        << std::sqrt(
               difference_squared /
               std::max(reference_squared, 1.0e-30))
        << '\n';
}

void profile_nint_breakdown(
    const MfqContainer& model,
    const std::string& label,
    const std::string& name,
    int warmup,
    int repetitions) {
    auto linear = MlxLinear::load(model, name);
    const auto* weight = linear.nint_weight_ref();
    require(weight != nullptr, label + " is not a dense NINT tensor");
    auto decode_once = [&]() {
        auto dense = weight->dequantize(mlx::core::float16);
        dense.eval();
    };
    decode_once();
    for (int index = 0; index < warmup; ++index) {
        decode_once();
    }
    mlx::core::synchronize();
    const auto decode_started = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        decode_once();
    }
    mlx::core::synchronize();
    const double decode_ms =
        elapsed_ms(decode_started) / static_cast<double>(repetitions);

    auto dense = weight->dequantize(mlx::core::float16);
    dense.eval();
    profile(
        label + "_f16_matmul",
        linear.input_size(),
        dense.nbytes(),
        1,
        [&](const array& input) {
            return std::vector<array>{mlx::core::matmul(
                input, mlx::core::transpose(dense))};
        },
        warmup,
        repetitions);

    ::setenv("MFQ_METAL_NINT_PREFILL_PACKED_MMQ", "1", 1);
    profile(
        label + "_packed_nax",
        linear.input_size(),
        packed_bytes(linear),
        1,
        [&](const array& input) {
            return std::vector<array>{linear(input)};
        },
        warmup,
        repetitions);
    ::setenv("MFQ_METAL_NINT_PREFILL_PACKED_MMQ", "0", 1);
    profile(
        label + "_transient_f16",
        linear.input_size(),
        packed_bytes(linear),
        2,
        [&](const array& input) {
            return std::vector<array>{linear(input)};
        },
        warmup,
        repetitions);
    ::unsetenv("MFQ_METAL_NINT_PREFILL_PACKED_MMQ");

    std::cout
        << label << "_decode_to_f16\t"
        << packed_bytes(linear) << "\t1\t"
        << std::fixed << std::setprecision(4) << decode_ms << "\t"
        << std::setprecision(1)
        << static_cast<double>(dense.nbytes()) / (decode_ms * 1.0e6)
        << '\n';
}

void profile_linear_alpha_beta(
    const MfqContainer& model,
    int warmup,
    int repetitions) {
    auto alpha = MlxLinear::load(
        model,
        "model.block.0.linear_attention.alpha.weight");
    auto beta = MlxLinear::load(
        model,
        "model.block.0.linear_attention.beta.weight");
    const auto* alpha_weight = alpha.dense_weight_ref();
    const auto* beta_weight = beta.dense_weight_ref();
    if (alpha_weight == nullptr || beta_weight == nullptr) {
        if (g_profile_rows > MlxGroupedLinear::max_rows()) {
            profile(
                "linear_alpha_prefill",
                alpha.input_size(),
                packed_bytes(alpha),
                1,
                [&](const array& input) {
                    return std::vector<array>{alpha(input)};
                },
                warmup,
                repetitions);
            profile(
                "linear_beta_prefill",
                beta.input_size(),
                packed_bytes(beta),
                1,
                [&](const array& input) {
                    return std::vector<array>{beta(input)};
                },
                warmup,
                repetitions);
            return;
        }
        MlxGroupedLinear grouped(grouped_refs({&alpha, &beta}));
        profile(
            "linear_alpha_beta_grouped",
            alpha.input_size(),
            grouped.packed_nbytes(),
            1,
            [&](const array& input) {
                return grouped(input);
            },
            warmup,
            repetitions);
        return;
    }
    auto combined = mlx::core::contiguous(
        mlx::core::concatenate(
            {*alpha_weight, *beta_weight},
            0));
    combined.eval();
    const auto bytes = combined.nbytes();
    const int width = alpha.input_size();
    profile(
        "linear_alpha_beta_dense",
        width,
        bytes,
        1,
        [&](const array& input) {
            return std::vector<array>{
                mlx::core::matmul(
                    input,
                    mlx::core::transpose(combined)),
            };
        },
        warmup,
        repetitions);
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc >= 2,
            "usage: mfq-metal-qwen35-projection-profile "
            "MODEL.mfq [REPETITIONS]");
        const int repetitions =
            argc >= 3 ? std::stoi(argv[2]) : 40;
        g_profile_rows = argc >= 4 ? std::stoi(argv[3]) : 1;
        require(repetitions > 0, "repetitions must be positive");
        require(
            g_profile_rows > 0 && g_profile_rows <= 131072,
            "profile rows must be in [1, 131072]");
        constexpr int warmup = 4;

        MfqContainer model(argv[1]);
        mfq::metal::install_legacy_tensor_compatibility(model);
        std::cout
            << "case\tpacked_bytes\tdispatches\tms\tGB/s\n";
        if (argc >= 5 && std::string(argv[4]) == "parity") {
            const std::vector<std::pair<std::string, std::string>> cases{
                {"linear_qkv_l0", "model.block.0.linear_attention.qkv.weight"},
                {"linear_gate_l0", "model.block.0.linear_attention.gate.weight"},
                {"linear_output_l0", "model.block.0.linear_attention.output.weight"},
                {"full_query_l3", "model.block.3.attention.query.weight"},
                {"full_query_l63", "model.block.63.attention.query.weight"},
                {"ffn_gate_l0", "model.block.0.mlp.gate.weight"},
                {"ffn_gate_l50", "model.block.50.mlp.gate.weight"},
                {"ffn_down_l0", "model.block.0.mlp.down.weight"},
                {"ffn_down_l8", "model.block.8.mlp.down.weight"},
                {"ffn_down_l12", "model.block.12.mlp.down.weight"},
                {"lm_head", "model.output.weight"},
            };
            for (const auto& [label, name] : cases) {
                compare_packed_and_transient(model, label, name);
            }
            return 0;
        }
        if (argc >= 5 && std::string(argv[4]) == "breakdown") {
            profile_nint_breakdown(
                model,
                "ffn_gate_nint4_l0",
                "model.block.0.mlp.gate.weight",
                warmup,
                repetitions);
            profile_nint_breakdown(
                model,
                "ffn_down_nint6_l0",
                "model.block.0.mlp.down.weight",
                warmup,
                repetitions);
            return 0;
        }
        if (argc >= 5 && std::string(argv[4]) == "ffn") {
            profile_gate_up(
                model,
                0,
                "ffn_swiglu_nint4_l0",
                warmup,
                repetitions);
            profile_single(
                model,
                "ffn_down_nint6_l0",
                "model.block.0.mlp.down.weight",
                warmup,
                repetitions);
            return 0;
        }
        profile_single(
            model,
            "linear_qkv_nint6_l0",
            "model.block.0.linear_attention.qkv.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "linear_qkv_nint4_l10",
            "model.block.10.linear_attention.qkv.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "linear_z_nint5_l0",
            "model.block.0.linear_attention.gate.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "linear_out_nint8_l0",
            "model.block.0.linear_attention.output.weight",
            warmup,
            repetitions);
        profile_linear_alpha_beta(
            model,
            warmup,
            repetitions);
        profile_full_qkv(
            model,
            3,
            "full_q4k4v6_l3",
            warmup,
            repetitions);
        profile_full_qkv(
            model,
            15,
            "full_q4k4v5_l15",
            warmup,
            repetitions);
        profile_full_qkv(
            model,
            63,
            "full_q5k5v6_l63",
            warmup,
            repetitions);
        profile_single(
            model,
            "full_q_nint4_l3",
            "model.block.3.attention.query.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "full_q_nint5_l63",
            "model.block.63.attention.query.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "full_k_nint4_l3",
            "model.block.3.attention.key.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "full_k_nint5_l63",
            "model.block.63.attention.key.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "full_v_nint6_l3",
            "model.block.3.attention.value.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "full_v_nint5_l15",
            "model.block.15.attention.value.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "full_attn_out_nint4_l3",
            "model.block.3.attention.output.weight",
            warmup,
            repetitions);
        profile_gate_up(
            model,
            0,
            "ffn_swiglu_nint4_l0",
            warmup,
            repetitions);
        profile_gate_up(
            model,
            50,
            "ffn_swiglu_nint5_l50",
            warmup,
            repetitions);
        profile_single(
            model,
            "ffn_down_nint6_l0",
            "model.block.0.mlp.down.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "ffn_down_nint4_l8",
            "model.block.8.mlp.down.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "ffn_down_nint5_l12",
            "model.block.12.mlp.down.weight",
            warmup,
            repetitions);
        profile_single(
            model,
            "lm_head",
            "model.output.weight",
            warmup,
            repetitions);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3.5 projection profile failed: "
                  << error.what() << '\n';
        return 1;
    }
}
