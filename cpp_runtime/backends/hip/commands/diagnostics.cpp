#include "hip/hip_runtime.h"
#include "models/qwen4_exp/mtp.h"
#include "models/glm5_next/mtp.h"
#include "storage/weight_loader.h"
#include "cli.h"
#include "diagnostics/token_generation.h"
#include "storage/model_loader.h"
#include "diagnostics/backend_checks.h"
#include "diagnostics/model_checks.h"
#include "diagnostics/runtime_checks.h"
#include "diagnostics/flash_next_mtp.h"
#include "minicpmo45.h"
#include "hip_execution.h"
#include "storage/moe_expert_cache.h"
#include "mfq/kernels/cuda/deepseek_v41.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mfq::cuda {

struct DiagnosticsCommandOptions : CudaLoadOptions, TokenInputOptions {
    std::string check_linear, check_linear_cpu, check_linear_gate;
    std::string check_tp_linear, check_ep_moe;
    std::string check_tp_axis_arg = "output";
    std::string check_linear_group, check_q8_embedding;
    std::string check_gdn_input, check_gdn_output, check_gdn_state;
    std::string check_linear_conv_input, check_linear_conv_output;
    std::string check_mfe_tensor, check_dsv4_output_a;
    std::string prefill_sweep_arg;
    std::string check_moe_tokens = "1,2,4,8,16,32,64,128,256";
    std::string block_trace_reference, block_trace_output;
    std::string check_tokenizer_text =
        "MFQ tokenizer check: hello, world! <think>";
    std::string bench_qwen35_mtp;
    std::string kl_mmq_arg = "default";
    int bench_qwen35_mtp_reps = 3;
    int64_t minicpmo_eval_vision_batch_size = 16;
    int64_t block_trace_start = 0;
    int64_t block_trace_count = 0;
    int prefill_repeat = 0;
    int prefill_sweep_reps = 5;
    int check_linear_m = 1;
    int check_linear_reps = 200;
    int check_tp_m = 1;
    int check_ep_moe_tokens = 1;
    int check_ep_moe_routes = 2;
    int check_gdn_tokens = 512;
    int check_gdn_q_heads = 16;
    int check_gdn_v_heads = 32;
    int check_gdn_head_dim = 128;
    int check_linear_conv_tokens = 512;
    int check_linear_conv_q_heads = 16;
    int check_linear_conv_v_heads = 32;
    int check_linear_conv_key_dim = 128;
    int check_linear_conv_value_dim = 128;
    int check_linear_conv_kernel = 4;
    int check_gemma_geglu_layer = -1;
    int check_gemma_geglu_reps = 100;
    int check_moe_layer = -1;
    int check_moe_reps = 100;
    int check_mfe_tokens = 1;
    int check_mfe_routes = 2;
    int check_mfe_reps = 100;
    int check_mfe_split_width = 0;
    int check_dsv4_output_a_batch = 1;
    int check_dsv4_output_a_reps = 200;
    int check_attention_decode = 0;
    int check_attention_reps = 200;
    int check_attention_head_dim = 256;
    int check_attention_window = 4096;
    int compare_mma_decode_steps = 1;
    int compare_mma_decode_planned_len = 0;
    bool profile = false;
    bool ignore_eos = false;
    bool compare_ffn_transfer_phases = false;
    bool compare_router_lookahead = false;
    bool compare_rotary_fusion = false;
    bool compare_attention_grouping = false;
    bool compare_mhc_fusion = false;
    bool compare_mhc_shared_norm = false;
    bool compare_mhc_dense_fusion = false;
    bool compare_mfe_parallel_gu = false;
    bool compare_mfe_gpu_bundle = false;
    bool compare_mfe_shared_first = false;
    bool compare_mfe_early_gu = false;
    bool compare_cpu_transfer_budget = false;
    bool compare_cpu_background_calibration = false;
    bool compare_router_topk = false;
    int warmup_prefill = -1;
    std::string router_lookahead_audit;
    std::string decode_inputs_file;
    std::vector<int64_t> decode_inputs;
    bool check_backend_bf16_add = false;
    bool check_backend_argmax = false;
    bool compare_mma_attention = false;
    bool compare_decode_splitk = false;
    bool compare_mma_decode = false;
    bool compare_nvq_vec4 = false;
    bool check_gemma4_swa = false;
    bool check_glm_dsa = false;
    bool check_dsv4_attention = false;
    bool check_dsv4_hc = false;
    bool check_deepseek_v41 = false;
    bool check_text_session_state = false;
    bool check_qwen35_mtp = false;
    bool check_continuous_batching = false;
    bool check_engine_isolation = false;
    bool check_flash_next = false;
    bool check_flash_next_mtp = false;
    bool check_nemotron_h=false,check_nemotron_h_mtp=false,check_nemotron_h_vision=false;
    bool compare_dsv4_hc_ops = false;
    bool compare_dsv4_hc_model = false;
    bool check_attention_swa_decode = false;
    bool check_mfe_routed_input = false;
    bool check_mfe_benchmark_only = false;
    bool check_runtime_assets = false;
    bool check_mfq_container = false;
    bool minicpmo_eval_batch = false;

    int runtime_comparison_modes() const {
        return int(compare_ffn_transfer_phases) + int(compare_router_lookahead) +
               int(compare_rotary_fusion) + int(compare_attention_grouping) + int(compare_mhc_fusion) + int(compare_mhc_shared_norm) + int(compare_mhc_dense_fusion) + int(compare_mfe_parallel_gu) + int(compare_mfe_gpu_bundle) + int(compare_mfe_shared_first) + int(compare_mfe_early_gu) + int(compare_cpu_transfer_budget) + int(compare_cpu_background_calibration) +
               int(compare_router_topk);
    }
    const char* runtime_comparison_name() const {
        if (compare_router_topk) return "router_topk_comparison";
        if (compare_mhc_fusion) return "mhc_fusion_comparison";
        if (compare_mhc_shared_norm) return "mhc_shared_norm_comparison";
        if (compare_mhc_dense_fusion) return "mhc_dense_fusion_comparison";
        if (compare_mfe_parallel_gu) return "mfe_parallel_gu_comparison";
        if (compare_mfe_gpu_bundle) return "mfe_gpu_bundle_comparison";
        if (compare_mfe_shared_first) return "mfe_shared_first_comparison";
        if (compare_mfe_early_gu) return "mfe_early_gate_up_comparison";
        if (compare_cpu_background_calibration) return "cpu_background_calibration_comparison";
        if (compare_cpu_transfer_budget) return "cpu_transfer_budget_comparison";
        if (compare_attention_grouping) return "attention_grouping_comparison";
        if (compare_rotary_fusion) return "rotary_fusion_comparison";
        if (compare_router_lookahead) return "router_lookahead_comparison";
        return "ffn_transfer_comparison";
    }
};

} // namespace mfq::cuda

using namespace mfq::cuda::internal;

using namespace mfq::cuda::diagnostics;

namespace {
class ScopedDiagnosticEnvironment {
    std::string name_;
    std::string previous_;
    bool present_;
public:
    explicit ScopedDiagnosticEnvironment(const char* name) : name_(name) {
        const char* previous = std::getenv(name);
        present_ = previous != nullptr;
        if (previous) previous_ = previous;
    }
    void set(const char* value) {
#ifdef _WIN32
        const int result = _putenv_s(name_.c_str(), value);
#else
        const int result = setenv(name_.c_str(), value, 1);
#endif
        MFQ_RUNTIME_CHECK(result == 0,
            "cannot set diagnostic runtime option: ", name_);
    }
    ~ScopedDiagnosticEnvironment() {
#ifdef _WIN32
        (void)_putenv_s(name_.c_str(), present_ ? previous_.c_str() : "");
#else
        if (present_) (void)setenv(name_.c_str(), previous_.c_str(), 1);
        else (void)unsetenv(name_.c_str());
#endif
    }
};

struct DiagnosticsCommand : mfq::cuda::DiagnosticsCommandOptions {
    explicit DiagnosticsCommand(mfq::cuda::DiagnosticsCommandOptions options)
        : mfq::cuda::DiagnosticsCommandOptions(std::move(options)) {}
    int run() { return with_command_errors([&]() -> int {
        if (check_engine_isolation) {
            mfq::cuda::CudaEngineOptions options;
            static_cast<mfq::cuda::CudaLoadOptions&>(options) =
                static_cast<const mfq::cuda::CudaLoadOptions&>(*this);
            return run_cuda_engine_isolation_check(std::move(options));
        }
        CudaExecutionContext execution;
        // Operator-only checks do not require loading a model.
        if (check_backend_bf16_add) return run_backend_bf16_add_check(4096, 10000);
        if (check_backend_argmax) return run_backend_argmax_check(151748, 2000);
        if(runtime_comparison_modes()) {
            MFQ_RUNTIME_CHECK(gen>1 && !decode_inputs.empty() && !profile &&
                runtime_comparison_modes() == 1 && router_lookahead_audit.empty() &&
                !execution.config.moe_mapped_gather && execution.config.moe_preload_all &&
                execution.config.moe_two_stage_ffn && moe_gpu_cache_gb>0,
                "runtime comparison requires fixed decode inputs, complete expert preload, two-stage FFN and profiling disabled");
            MFQ_RUNTIME_CHECK(!compare_attention_grouping ||
                execution.config.diagnostic_nint_group,
                "attention-grouping comparison requires grouped NINT enabled");
            MFQ_RUNTIME_CHECK(!compare_mhc_fusion || (execution.config.gr_fused_projections &&
                execution.config.gr_native_projection_input),
                "residual fusion comparison requires native packed projections");
            execution.config.moe_residency_adapt=false;
            std::cout<<std::unitbuf;
        }
        setup_cuda_load(*this, execution);
        if(runtime_comparison_modes()) {
            std::cout<<runtime_comparison_name()<<"_setup\n";
            print_moe_expert_cache_stats(execution.moe_expert_cache,std::cout);
        }
        if (!check_linear_cpu.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-linear-cpu requires --model");
            }
            int gate_mode = 0;
            if (check_linear_gate == "sigmoid") gate_mode = 1;
            else if (check_linear_gate == "silu") gate_mode = 2;
            else if (!check_linear_gate.empty()) {
                throw std::runtime_error("--check-linear-gate must be sigmoid or silu");
            }
            return run_cpu_linear_check(
                execution, model_path, check_linear_cpu, check_linear_m,
                gate_mode, check_linear_reps);
        }
        if (!check_linear.empty()) {
            if (model_path.empty()) throw std::runtime_error("--check-linear requires --model");
            int gate_mode = 0;
            if (check_linear_gate == "sigmoid") gate_mode = 1;
            else if (check_linear_gate == "silu") gate_mode = 2;
            else if (!check_linear_gate.empty()) {
                throw std::runtime_error("--check-linear-gate must be sigmoid or silu");
            }
            return run_linear_check(
                execution, model_path, check_linear, check_linear_m, gate_mode,
                check_linear_reps);
        }
        if (!check_tp_linear.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--check-tp-linear requires --model");
            }
            const TensorParallelAxis axis =
                check_tp_axis_arg == "output"
                ? TensorParallelAxis::Output
                : check_tp_axis_arg == "input"
                ? TensorParallelAxis::Input
                : throw std::runtime_error(
                    "--check-tp-axis must be output or input");
            return run_tensor_parallel_linear_check(
                execution, model_path, check_tp_linear,
                axis, check_tp_m);
        }
        if (!check_ep_moe.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--check-ep-moe requires --model");
            }
            return run_expert_parallel_moe_check(
                execution, model_path, check_ep_moe,
                check_ep_moe_tokens,
                check_ep_moe_routes);
        }
        if (!check_linear_group.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-linear-group requires --model");
            }
            return run_linear_group_check(
                execution, model_path, check_linear_group, check_linear_m,
                check_linear_reps);
        }
        if (!check_q8_embedding.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-q8-embedding requires --model");
            }
            return run_q8_embedding_check(
                execution, model_path, check_q8_embedding);
        }
        if (!check_gdn_input.empty()) {
            if (check_gdn_output.empty() || check_gdn_state.empty()) {
                throw std::runtime_error(
                    "--check-gdn-input requires --check-gdn-output and --check-gdn-state");
            }
            return run_gdn_operator_check(
                check_gdn_input, check_gdn_output, check_gdn_state,
                check_gdn_tokens, check_gdn_q_heads,
                check_gdn_v_heads, check_gdn_head_dim);
        }
        if (!check_linear_conv_input.empty()) {
            if (check_linear_conv_output.empty()) {
                throw std::runtime_error(
                    "--check-linear-conv-input requires --check-linear-conv-output");
            }
            return run_linear_conv_operator_check(
                check_linear_conv_input, check_linear_conv_output,
                check_linear_conv_tokens, check_linear_conv_q_heads,
                check_linear_conv_v_heads, check_linear_conv_key_dim,
                check_linear_conv_value_dim, check_linear_conv_kernel,
                1.0e-6);
        }
        if (check_gemma_geglu_layer >= 0) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-gemma-geglu-layer requires --model");
            }
            return run_gemma_geglu_check(
                execution, model_path, check_gemma_geglu_layer,
                check_gemma_geglu_reps);
        }
        if (check_mfq_container) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--check-mfq-container requires --model");
            }
            auto model_source = mfq::open_model_source(model_path);
            const auto& mfq = *model_source;
            std::cout << "mfq_container_check=ok"
                      << " shards=" << mfq.source_paths().size()
                      << " tensors=" << mfq.tensors().size()
                      << " assets=" << mfq.assets().size()
                      << "\n";
            return 0;
        }
        if (check_runtime_assets) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--check-runtime-assets requires --model");
            }
            auto model_source = mfq::open_model_source(model_path);
            const auto& mfq = *model_source;
            const auto config = mfq::models::ModelConfig::from_json(
                mfq::cuda::load_model_config_json(mfq, config_path));
            if (!mfq.has_asset(mfq::kTokenizerGgufAsset)) {
                throw std::runtime_error(
                    "model source has no tokenizer GGUF");
            }
            const auto tokenizer_blob =
                read_asset(mfq, mfq::kTokenizerGgufAsset);
            const auto embedded =
                probe_mfq_tokenizer(tokenizer_blob, check_tokenizer_text);
            if (embedded.vocab_size != config.vocab_size) {
                throw std::runtime_error(
                    "embedded tokenizer/model vocabulary mismatch: tokenizer=" +
                    std::to_string(embedded.vocab_size) + " model=" +
                    std::to_string(config.vocab_size));
            }
            std::cout << "runtime_assets_check=ok"
                      << " config=embedded"
                      << " tokenizer=embedded"
                      << " vocab_size=" << embedded.vocab_size
                      << " chat_template="
                      << (!embedded.chat_template.empty() ? 1 : 0)
                      << " bos=" << embedded.bos_token
                      << " eos=" << embedded.eos_token
                      << " eot=" << embedded.eot_token
                      << " pad=" << embedded.pad_token
                      << " token_count=" << embedded.tokens.size()
                      << "\n";
            if (!tokenizer_model.empty()) {
                const auto external =
                    probe_mfq_tokenizer(tokenizer_model, check_tokenizer_text);
                if (external.vocab_size != embedded.vocab_size ||
                    external.bos_token != embedded.bos_token ||
                    external.eos_token != embedded.eos_token ||
                    external.eot_token != embedded.eot_token ||
                    external.pad_token != embedded.pad_token ||
                    external.chat_template != embedded.chat_template ||
                    external.tokens != embedded.tokens) {
                    throw std::runtime_error(
                        "embedded tokenizer differs from external GGUF");
                }
                std::cout << "runtime_assets_external_match=ok\n";
            }
            return 0;
        }
        if (check_moe_layer >= 0) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-moe-layer requires --model");
            }
            return run_moe_check(
                execution, model_path, config_path, check_moe_layer, parse_ids(check_moe_tokens), check_moe_reps);
        }
        if (!check_mfe_tensor.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-mfe-tensor requires --model");
            }
            KlMmqScope check_kl_mmq_scope(
                parse_kl_mmq_mode(kl_mmq_arg), execution.kl_mmq);
            const auto tensor_names =
                parse_tensor_names(check_mfe_tensor);
            if (execution.moe_expert_cache &&
                    tensor_names.size() != 1) {
                throw std::runtime_error(
                    "cached --check-mfe-tensor accepts one tensor "
                    "per invocation");
            }
            for (const auto & tensor_name : tensor_names) {
                const int result = run_mfe_tensor_check(
                    execution, model_path, tensor_name, check_mfe_tokens,
                    check_mfe_routes, check_mfe_reps,
                    check_mfe_split_width, check_mfe_routed_input,
                    check_mfe_benchmark_only);
                if (result != 0) return result;
            }
            return 0;
        }
        if (!check_dsv4_output_a.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-dsv4-output-a requires --model");
            }
            return run_dsv4_output_a_check(
                execution, model_path, check_dsv4_output_a,
                check_dsv4_output_a_batch, check_dsv4_output_a_reps);
        }
        if (check_attention_decode > 0) {
            return run_attention_decode_check(
                check_attention_decode, check_attention_reps,
                check_attention_head_dim, check_attention_swa_decode,
                check_attention_window);
        }
        if (check_gemma4_swa) {
            return run_gemma4_swa_check(check_attention_reps);
        }
        if (check_glm_dsa) {
            return run_glm_dsa_check(check_attention_reps);
        }
        if (check_dsv4_attention) {
            return run_dsv4_attention_check(check_attention_reps);
        }
        if (check_dsv4_hc) {
            return run_dsv4_hc_check(check_attention_reps);
        }
        if (check_deepseek_v41) {
            return mfq::cuda::deepseek_v41_runtime::run_self_check();
        }
        if (check_text_session_state) {
            return run_text_session_state_check();
        }
        if (minicpmo_eval_batch) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--minicpmo-eval-batch requires --model");
            }
            if (!ids_arg.empty() || !ids_file.empty()) {
                throw std::runtime_error(
                    "MiniCPM-o eval mode cannot be combined with "
                    "composite, duplex, token, transport, KL, or prefill modes");
            }
            if (minicpmo_eval_vision_batch_size <= 0) {
                throw std::runtime_error(
                    "--minicpmo-eval-vision-batch-size must be positive");
            }
            return mfq::cuda::minicpmo45::run_eval_batch(
                execution, model_path, config_path, context_size,
                minicpmo_eval_vision_batch_size);
        }
        if (model_path.empty()) {
            std::cerr << "missing model or execution mode (see --help)\n";
            return 2;
        }
        if (context_size < 0) throw std::runtime_error("--ctx-size must be positive");
        if (!bench_qwen35_mtp.empty()) {
            MFQ_RUNTIME_CHECK(bench_qwen35_mtp == "ordinary" || bench_qwen35_mtp == "mtp",
                "--bench-qwen35-mtp expects ordinary or mtp");
            MFQ_RUNTIME_CHECK(!check_qwen35_mtp && ids_arg.empty() && ids_file.empty() &&
                prefill_sweep_arg.empty(), "MTP benchmark cannot combine execution modes");
            if (context_size == 0) context_size = 512;
            MFQ_RUNTIME_CHECK(gen >= 2 && context_size >= static_cast<int64_t>(gen) + 17 &&
                bench_qwen35_mtp_reps > 0 && bench_qwen35_mtp_reps <= 100,
                "MTP benchmark requires gen>=2, reps1-100 and context>=gen+17");
            std::cout << std::unitbuf;
        }
        if(check_nemotron_h_vision && !context_size)context_size=128;
        if(check_nemotron_h_mtp) {
            if(!context_size)context_size=128;
            MFQ_RUNTIME_CHECK(context_size>=64,"Nemotron-H MTP requires --ctx-size >=64");
        }
        if (check_qwen35_mtp) {
            if (context_size == 0) context_size = 512;
            if (context_size < 64) throw std::runtime_error("MTP correctness gate requires --ctx-size >= 64");
            std::cout << std::unitbuf;
        }
        if (check_continuous_batching) {
            MFQ_RUNTIME_CHECK(
                !check_qwen35_mtp &&
                !check_flash_next && !check_flash_next_mtp &&
                bench_qwen35_mtp.empty() && ids_arg.empty() &&
                ids_file.empty() &&
                prefill_sweep_arg.empty(),
                "continuous batching check cannot combine execution modes");
            if (context_size == 0) context_size = 512;
            if (context_size < 32) {
                throw std::runtime_error(
                    "continuous batching check requires --ctx-size >= 32");
            }
            std::cout << std::unitbuf;
        }
        std::vector<int64_t> prefill_sweep_sizes;
        if (!prefill_sweep_arg.empty()) {
            prefill_sweep_sizes = parse_ids(prefill_sweep_arg);
            if (std::any_of(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end(),
                            [](int64_t value) { return value <= 0; })) {
                throw std::runtime_error("--prefill-sweep values must be positive");
            }
            if (context_size == 0) {
                context_size = *std::max_element(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end());
            }
        }
        if ((!block_trace_reference.empty() || !block_trace_output.empty()) &&
                context_size == 0) {
            context_size = (int64_t)(
                ids_file.empty() ? parse_ids(ids_arg) : load_ids_file(ids_file)).size();
        }
        execution.profiler.enabled = false;
        mfq_tensor_backend::NoGradGuard no_grad;
        return with_loaded_cuda_model(execution, *this,
            check_qwen35_mtp || check_flash_next_mtp || check_nemotron_h_mtp || check_nemotron_h_vision || !bench_qwen35_mtp.empty(),
            [&](auto& model, auto& runtime_components,
                    auto t0, auto t1) -> int {
        using Model = std::remove_cvref_t<decltype(model)>;
        if (check_continuous_batching) {
            if constexpr (std::is_same_v<Model, mfq::cuda::Qwen35CausalLm>) {
                return run_cuda_continuous_batching_check(model);
            }
            throw std::runtime_error(
                "continuous batching requires Qwen35CausalLm");
        }
        if(check_nemotron_h || check_nemotron_h_mtp || check_nemotron_h_vision) {
            if constexpr(std::is_same_v<Model,mfq::cuda::NemotronHCausalLm>) {
                if(check_nemotron_h_mtp) {
                    auto* predictor=dynamic_cast<mfq::cuda::nemotron_h::NemotronHMtp*>(runtime_components.mtp.get());
                    MFQ_RUNTIME_CHECK(predictor,"Nemotron-H diagnostic requires MTP weights");
                    return run_nemotron_h_mtp_check(model,*predictor);
                }
                if(check_nemotron_h_vision) {
                    MFQ_RUNTIME_CHECK(runtime_components.radio_vision,"Nemotron-H diagnostic requires RADIO weights");
                    return run_nemotron_h_vision_check(model,*runtime_components.radio_vision);
                }
                return run_nemotron_h_check(model);
            }
            throw std::runtime_error("Nemotron-H diagnostic requires a Nemotron-H causal LM");
        }
        if (check_flash_next) {
            if constexpr (
                    std::is_same_v<Model, mfq::cuda::Qwen4CausalLm> ||
                    std::is_same_v<Model, mfq::cuda::Glm5CausalLm>) {
                return run_flash_next_check(model);
            }
            throw std::runtime_error(
                "Flash-Next diagnostic requires a Flash-Next causal LM");
        }
        if (check_flash_next_mtp) {
            if constexpr (
                    std::is_same_v<Model, mfq::cuda::Qwen4CausalLm> ||
                    std::is_same_v<Model, mfq::cuda::Glm5CausalLm>) {
                using Predictor=std::conditional_t<
                    std::is_same_v<Model, mfq::cuda::Qwen4CausalLm>,
                    mfq::cuda::qwen4_exp::Qwen4ExpMtp,
                    mfq::cuda::glm5_next::Glm5NextMtp>;
                auto* predictor=dynamic_cast<Predictor*>(runtime_components.mtp.get());
                MFQ_RUNTIME_CHECK(predictor != nullptr,
                    "Flash-Next MTP diagnostic requires a loaded predictor component");
                return run_flash_next_mtp_check(model,*predictor);
            }
            throw std::runtime_error(
                "Flash-Next MTP diagnostic requires a Flash-Next causal LM");
        }
        if (check_qwen35_mtp) {
            if constexpr (std::is_same_v<Model, mfq::cuda::Qwen35CausalLm>) {
                auto* predictor = dynamic_cast<Qwen35Mtp*>(
                    runtime_components.mtp.get());
                MFQ_RUNTIME_CHECK(predictor != nullptr,
                    "--check-qwen35-mtp requires a supported model containing MTP weights");
                return run_qwen35_mtp_check(model, *predictor);
            }
            throw std::runtime_error(
                "--check-qwen35-mtp requires Qwen35CausalLm");
        }
        if (!bench_qwen35_mtp.empty()) {
            if constexpr (std::is_same_v<Model, mfq::cuda::Qwen35CausalLm>) {
                auto* predictor = dynamic_cast<Qwen35Mtp*>(
                    runtime_components.mtp.get());
                MFQ_RUNTIME_CHECK(predictor != nullptr,
                    "--bench-qwen35-mtp requires a supported model containing MTP weights");
                return run_qwen35_mtp_bench(model, *predictor,
                    bench_qwen35_mtp == "mtp", gen,
                    bench_qwen35_mtp_reps);
            }
            throw std::runtime_error(
                "--bench-qwen35-mtp requires Qwen35CausalLm");
        }
        if (!prefill_sweep_sizes.empty()) {
            const int status = run_prefill_sweep(
                model, prefill_sweep_sizes, prefill_sweep_reps);
            if (execution.moe_expert_cache) {
                print_moe_expert_cache_stats(
                    execution.moe_expert_cache, std::cout);
            }
            return status;
        }
        if(!router_lookahead_audit.empty()) {
            MFQ_RUNTIME_CHECK(!compare_ffn_transfer_phases && !profile && gen>1 && !decode_inputs.empty(),
                "router lookahead audit requires fixed single-token decode inputs and profiling disabled");
            if constexpr (std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
                qwen4_begin_router_lookahead_audit(model,gen-1);
            else throw std::runtime_error("router lookahead audit requires Qwen4 Flash-Next");
        }
        auto ids_vec = ids_file.empty() ? parse_ids(ids_arg) : load_ids_file(ids_file);
        auto ids = mfq_tensor_backend::tensor(ids_vec, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA)).unsqueeze(0);
        if (compare_dsv4_hc_ops) {
            g_dsv4_compare_hc_ops = true;
            g_dsv4_fused_hc = false;
            model.reset(1);
            (void)model.forward(ids);
            mfq_hip_synchronize();
            return 0;
        }
        if (compare_dsv4_hc_model) {
            return run_dsv4_hc_model_compare(model, ids);
        }
        if (!block_trace_reference.empty()) {
            return run_block_trace_compare(
                model, block_trace_reference, config_path, context_size, ids);
        }
        if (!block_trace_output.empty()) {
            return run_block_trace_dump(
                model, block_trace_output, ids,
                block_trace_start, block_trace_count);
        }
        if(runtime_comparison_modes()) {
            if constexpr (!std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
                throw std::runtime_error("runtime comparison requires Qwen4 Flash-Next");
            const char* comparison_name = runtime_comparison_name();
            std::cout<<comparison_name<<"_model_load_sec="
                <<std::chrono::duration<double>(t1-t0).count()<<"\n";
            const int rounds=warmup_prefill<0?1:warmup_prefill;
            const bool compare_dense_mhc=execution.config.gr_two_stage_dense_injection;
            ScopedDiagnosticEnvironment topk_cache("MFQ_MOE_TOPK_CACHE");
            ScopedDiagnosticEnvironment shared_norm("MFQ_GR_SHARE_NORM");
            ScopedDiagnosticEnvironment dense_fusion("MFQ_GR_DENSE_FUSION");
            ScopedDiagnosticEnvironment parallel_gu("MFQ_MFE_PARALLEL_GU");
            ScopedDiagnosticEnvironment gpu_bundle("MFQ_MFE_GPU_BUNDLE");
            ScopedDiagnosticEnvironment shared_first("MFQ_MFE_SHARED_FIRST");
            ScopedDiagnosticEnvironment early_gu("MFQ_MFE_EARLY_GU");
            ScopedDiagnosticEnvironment cpu_background("MFQ_MOE_CPU_BACKGROUND_CALIBRATION");
            MFQ_RUNTIME_CHECK(!(compare_mfe_parallel_gu || compare_mfe_gpu_bundle || compare_mfe_shared_first || compare_mfe_early_gu) || execution.config.moe_two_stage_ffn,
                "MFE parallel Gate/Up comparison requires two-stage FFN");
            MFQ_RUNTIME_CHECK(!compare_mhc_shared_norm || execution.config.gr_two_stage,
                "MHC norm comparison requires fused residual projection");
            MFQ_RUNTIME_CHECK(!compare_mhc_dense_fusion || (execution.config.gr_two_stage && execution.config.gr_two_stage_dense_injection),
                "MHC dense fusion comparison requires fused residual BF16 injection");
            for(int pass=0;pass<3;++pass) {
                const bool phased=pass==1;
                if (compare_cpu_background_calibration) cpu_background.set(phased ? "1" : "0");
                if (compare_router_topk) topk_cache.set(phased ? "1" : "0");
                if (compare_mhc_shared_norm) shared_norm.set(phased ? "1" : "0");
                if (compare_mhc_dense_fusion) {shared_norm.set("1");dense_fusion.set(phased ? "1" : "0");}
                if (compare_mfe_parallel_gu) parallel_gu.set(phased ? "1" : "0");
                if (compare_mfe_gpu_bundle) {
                    parallel_gu.set("1");gpu_bundle.set(phased ? "1" : "0");
                }
                if (compare_mfe_shared_first) {
                    parallel_gu.set("1");gpu_bundle.set("1");shared_first.set(phased ? "1" : "0");
                }
                if (compare_mfe_early_gu) {
                    parallel_gu.set("1");gpu_bundle.set("1");early_gu.set(phased ? "1" : "0");
                }
                // reset destroys the decoder graphs before each branch is captured.
                model.reset(1);mfq_hip_synchronize();
                if(compare_ffn_transfer_phases)execution.config.moe_ffn_transfer_phases=phased;
                if(compare_mhc_fusion) {
                    execution.config.gr_two_stage=phased;
                    execution.config.gr_two_stage_dense_injection=phased && compare_dense_mhc;
                }
                if constexpr (std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
                    if(compare_router_lookahead)model.router_lookahead=phased;
                    if(compare_rotary_fusion)qwen4_set_rotary_fusion(model,phased);
                    if(compare_attention_grouping) {
                        qwen4_set_rotary_fusion(model,true);
                        qwen4_set_attention_grouping(model,phased);
                    }
                    if(compare_mhc_fusion) {
                        qwen4_set_rotary_fusion(model,true);
                        qwen4_set_attention_grouping(model,true);
                    }
                }
                if(compare_cpu_background_calibration)
                    prepare_moe_cpu_calibration_comparison(execution.moe_expert_cache,pass>0);
                else if(compare_cpu_transfer_budget)
                    prepare_moe_cpu_budget_comparison(execution.moe_expert_cache,pass>0,phased);
                else prepare_moe_pipeline_comparison(execution.moe_expert_cache,pass>0);
                std::cout<<comparison_name<<"_pass="<<pass<<" phase="<<phased<<" begin=1\n";
                const auto warm_start=std::chrono::steady_clock::now();
                for(int round=0;round<rounds;++round) {
                    model.reset(1);(void)model.next_token(ids);mfq_hip_synchronize();
                }
                model.reset(1);mfq_hip_synchronize();execution.profiler.reset();
                std::cout<<"warmup_prefill_rounds="<<rounds<<"\n"
                    <<"warmup_prefill_tokens="<<rounds*ids_vec.size()<<"\n"
                    <<"warmup_prefill_sec="<<std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-warm_start).count()<<"\n";
                const auto pass_start=std::chrono::steady_clock::now();
                const int status=generate_diagnostic_tokens(execution,model,ids,gen,false,
                    pass_start,pass_start,decode_inputs);
                if(status)return status;
                const auto records=finish_moe_pipeline_comparison(execution.moe_expert_cache);
                if(compare_cpu_transfer_budget || compare_cpu_background_calibration || compare_mfe_shared_first || compare_mhc_dense_fusion || compare_mfe_early_gu)
                    print_moe_expert_cache_stats(execution.moe_expert_cache,std::cout);
                std::cout<<comparison_name<<"_pass="<<pass<<" phase="<<phased
                    <<" end=1 dispatch_records="<<records<<"\n";
            }
            return 0;
        }
        const int warmup_rounds = warmup_prefill < 0 ? (profile ? 1 : 0) : warmup_prefill;
        const auto warmup_start = std::chrono::steady_clock::now();
        for (int round = 0; round < warmup_rounds; ++round) {
            model.reset(1);
            (void)model.next_token(ids);
            mfq_hip_synchronize();
        }
        if (warmup_rounds) {
            model.reset(1);
            execution.profiler.reset();
        }
        std::cout << "warmup_prefill_rounds=" << warmup_rounds << "\n"
                  << "warmup_prefill_tokens=" << warmup_rounds * ids_vec.size() << "\n"
                  << "warmup_prefill_sec="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - warmup_start).count()
                  << "\n";
        if (profile) {
            execution.profiler.enabled = true;
        }
        if (compare_decode_splitk) {
            auto run = [&](bool split) {
                execution.config.attention_decode_split_k = split;
                model.reset(1);
                (void)model.hidden_forward(ids);
                const int64_t decode_len = model.cache_pos + 1;
                auto seq_len = mfq_tensor_backend::tensor({decode_len}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
                auto decode_id = ids.index({Slice(), -1}).reshape({1, 1});
                auto hidden = model.hidden_forward(decode_id, mfq_nullopt, seq_len);
                auto last = hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous();
                auto logits = model.lm_head.forward(execution, last).to(mfq_tensor_backend::kFloat32);
                mfq_hip_synchronize();
                return logits;
            };
            auto ref = run(false);
            auto test = run(true);
            auto ref_logp = mfq_tensor_backend::log_softmax(ref, -1);
            auto test_logp = mfq_tensor_backend::log_softmax(test, -1);
            auto kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1);
            auto diff = (ref - test).abs();
            std::cout << "decode_splitk_compare_kl=" << kl.template item<float>() << "\n";
            std::cout << "decode_splitk_compare_rel=" << ((test - ref).norm() / ref.norm()).template item<float>() << "\n";
            std::cout << "decode_splitk_compare_mean_abs=" << diff.mean().template item<float>() << "\n";
            std::cout << "decode_splitk_compare_max_abs=" << diff.max().template item<float>() << "\n";
            std::cout << "decode_splitk_compare_same_top="
                      << (ref.argmax(-1).eq(test.argmax(-1)).template item<bool>() ? 1 : 0) << "\n";
            return 0;
        }
        if (compare_mma_decode) {
            if (compare_mma_decode_steps < 1) {
                throw std::runtime_error("--compare-mma-decode-steps must be positive");
            }
            auto cuda_i64 = mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
            std::vector<mfq_tensor_backend::Tensor> reference_logits;
            std::vector<int64_t> teacher_tokens;
            reference_logits.reserve(compare_mma_decode_steps);
            teacher_tokens.reserve(compare_mma_decode_steps);
            execution.config.mma_attention_decode = false;
            model.reset(1);
            auto input = model.next_token(ids).reshape({1, 1});
            const int64_t initial_teacher_token = input.template item<int64_t>();
            for (int step = 0; step < compare_mma_decode_steps; ++step) {
                const int64_t decode_len = model.cache_pos + 1;
                auto seq_len = mfq_tensor_backend::tensor({decode_len}, cuda_i64);
                auto hidden = model.hidden_forward(input, mfq_nullopt, seq_len);
                auto last = hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous();
                auto logits = model.lm_head.forward(execution, last).to(mfq_tensor_backend::kFloat32);
                reference_logits.push_back(logits.clone());
                const int64_t next = logits.argmax(-1).template item<int64_t>();
                teacher_tokens.push_back(next);
                input = mfq_tensor_backend::tensor({next}, cuda_i64).reshape({1, 1});
            }
            execution.config.mma_attention_decode = true;
            const int64_t planned_kv_length =
                compare_mma_decode_planned_len > 0
                ? compare_mma_decode_planned_len
                : ids.size(1) + compare_mma_decode_steps;
            if (planned_kv_length > model.max_position_embeddings()) {
                throw std::runtime_error("decode comparison planned length exceeds context capacity");
            }
            model.reset(1);
            (void)model.next_token(ids);
            input = mfq_tensor_backend::tensor({initial_teacher_token}, cuda_i64).reshape({1, 1});
            double kl_sum = 0.0;
            double kl_max = 0.0;
            double abs_sum = 0.0;
            double delta_sq_sum = 0.0;
            double reference_sq_sum = 0.0;
            double max_abs = 0.0;
            int same_top = 0;
            int first_top_difference = -1;
            int64_t values = 0;
            for (int step = 0; step < compare_mma_decode_steps; ++step) {
                const int64_t decode_len = model.cache_pos + 1;
                auto seq_len = mfq_tensor_backend::tensor({decode_len}, cuda_i64);
                auto hidden = model.hidden_forward(
                    input, mfq_nullopt, seq_len, nullptr, mfq_nullopt,
                    nullptr, 0, {planned_kv_length, 0});
                auto last = hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous();
                auto test = model.lm_head.forward(execution, last).to(mfq_tensor_backend::kFloat32);
                const auto & ref = reference_logits[(size_t)step];
                auto ref_logp = mfq_tensor_backend::log_softmax(ref, -1);
                auto test_logp = mfq_tensor_backend::log_softmax(test, -1);
                const double kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1).template item<double>();
                auto delta = test - ref;
                kl_sum += kl;
                kl_max = std::max(kl_max, kl);
                abs_sum += delta.abs().sum().template item<double>();
                delta_sq_sum += delta.square().sum().template item<double>();
                reference_sq_sum += ref.square().sum().template item<double>();
                max_abs = std::max(max_abs, delta.abs().max().template item<double>());
                values += delta.numel();
                const bool top_equal = test.argmax(-1).eq(ref.argmax(-1)).template item<bool>();
                same_top += top_equal ? 1 : 0;
                if (!top_equal && first_top_difference < 0) first_top_difference = step;
                input = mfq_tensor_backend::tensor({teacher_tokens[(size_t)step]}, cuda_i64).reshape({1, 1});
            }
            mfq_hip_synchronize();
            std::cout << std::setprecision(10)
                      << "mma_decode_compare_steps=" << compare_mma_decode_steps << "\n"
                      << "mma_decode_compare_mean_kl=" << kl_sum / compare_mma_decode_steps << "\n"
                      << "mma_decode_compare_max_kl=" << kl_max << "\n"
                      << "mma_decode_compare_rel=" << std::sqrt(delta_sq_sum / reference_sq_sum) << "\n"
                      << "mma_decode_compare_mean_abs=" << abs_sum / values << "\n"
                      << "mma_decode_compare_max_abs=" << max_abs << "\n"
                      << "mma_decode_compare_same_top=" << same_top << "\n"
                      << "mma_decode_compare_first_top_difference=" << first_top_difference << "\n";
            return 0;
        }
        if (compare_nvq_vec4) {
            auto run = [&](bool) {
                model.reset(1);
                auto logits = model.last_logits(ids).to(mfq_tensor_backend::kFloat32);
                mfq_hip_synchronize();
                return logits;
            };
            auto ref = run(false);
            auto repeat = run(false);
            auto test = run(true);
            auto ref_logp = mfq_tensor_backend::log_softmax(ref, -1);
            auto repeat_logp = mfq_tensor_backend::log_softmax(repeat, -1);
            auto test_logp = mfq_tensor_backend::log_softmax(test, -1);
            auto repeat_kl = (ref_logp.exp() * (ref_logp - repeat_logp)).sum(-1);
            auto kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1);
            auto diff = (ref - test).abs();
            auto repeat_diff = (ref - repeat).abs();
            std::cout << "nvq_vec4_repeat_kl=" << repeat_kl.template item<float>() << "\n";
            std::cout << "nvq_vec4_repeat_max_abs=" << repeat_diff.max().template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_kl=" << kl.template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_rel="
                      << ((test - ref).norm() / ref.norm()).template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_mean_abs=" << diff.mean().template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_max_abs=" << diff.max().template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_same_top="
                      << (ref.argmax(-1).eq(test.argmax(-1)).template item<bool>() ? 1 : 0) << "\n";
            return 0;
        }
        if (prefill_repeat > 0) {
            const char * trace_env = std::getenv("MFQ_CHECK_PREFILL_REPEAT_TRACE");
            const bool trace_repeat = trace_env != nullptr && std::atoi(trace_env) != 0;
            std::vector<mfq_tensor_backend::Tensor> reference_trace;
            std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> reference_gemma_trace;
            for (int i = 0; i < prefill_repeat; ++i) {
                model.reset(1);
                auto run_t0 = std::chrono::steady_clock::now();
                std::vector<mfq_tensor_backend::Tensor> trace;
                std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> gemma_trace;
                mfq_tensor_backend::Tensor logits;
                if (trace_repeat) {
                    model.execution->gemma_trace_layer = 0;
                    model.execution->gemma_stage_trace = &gemma_trace;
                    auto hidden = model.hidden_forward(
                        ids, mfq_nullopt, mfq_nullopt, &trace);
                    model.execution->gemma_stage_trace = nullptr;
                    model.execution->gemma_trace_layer = -1;
                    auto last = hidden.index({Slice(), -1, Slice()})
                        .to(mfq_tensor_backend::kFloat16).contiguous();
                    logits = model.lm_head.forward(execution, last);
                } else {
                    logits = model.last_logits(ids);
                }
                mfq_hip_synchronize();
                auto run_t1 = std::chrono::steady_clock::now();
                std::cout << "prefill_repeat=" << (i + 1)
                          << " sec=" << std::chrono::duration<double>(run_t1 - run_t0).count()
                          << " top=" << logits.argmax(-1).template item<int64_t>() << "\n";
                if (trace_repeat) {
                    if (reference_trace.empty()) {
                        reference_trace = std::move(trace);
                        reference_gemma_trace = std::move(gemma_trace);
                    } else {
                        for (size_t stage = 0; stage < gemma_trace.size(); ++stage) {
                            auto diff = (gemma_trace[stage].second -
                                         reference_gemma_trace[stage].second).abs();
                            const float max_abs = diff.max().template item<float>();
                            if (max_abs != 0.0f) {
                                std::cout << "prefill_repeat_first_gemma_stage="
                                          << gemma_trace[stage].first
                                          << " max_abs=" << max_abs
                                          << " mean_abs=" << diff.mean().template item<float>() << "\n";
                                break;
                            }
                        }
                        for (size_t stage = 0; stage < trace.size(); ++stage) {
                            auto diff = (trace[stage] - reference_trace[stage]).abs();
                            const float max_abs = diff.max().template item<float>();
                            if (max_abs != 0.0f) {
                                std::cout << "prefill_repeat_first_difference=" << stage
                                          << " layer=" << static_cast<int64_t>(stage) - 1
                                          << " max_abs=" << max_abs
                                          << " mean_abs=" << diff.mean().template item<float>() << "\n";
                                break;
                            }
                        }
                    }
                }
            }
            return 0;
        }
        if (compare_mma_attention) {
            execution.config.mma_attention = false;
            execution.config.minicpm_bf16_flash128 = false;
            auto ref = model.last_logits(ids).to(mfq_tensor_backend::kFloat32);
            mfq_hip_synchronize();
            model.reset(1);
            execution.config.mma_attention = true;
            execution.config.minicpm_bf16_flash128 = true;
            auto test = model.last_logits(ids).to(mfq_tensor_backend::kFloat32);
            mfq_hip_synchronize();
            auto ref_logp = mfq_tensor_backend::log_softmax(ref, -1);
            auto test_logp = mfq_tensor_backend::log_softmax(test, -1);
            auto kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1);
            auto diff = (ref - test).abs();
            auto same_top = ref.argmax(-1).eq(test.argmax(-1));
            std::cout << "attention_compare_kl=" << kl.template item<float>() << "\n";
            std::cout << "attention_compare_max_logit_abs=" << diff.max().template item<float>() << "\n";
            std::cout << "attention_compare_mean_logit_abs=" << diff.mean().template item<float>() << "\n";
            std::cout << "attention_compare_same_top=" << (same_top.template item<bool>() ? 1 : 0) << "\n";
            return 0;
        }
        std::unordered_set<int64_t> stop_ids;
        if (!ignore_eos && tokenizer_model.empty() && decode_inputs.empty()) {
            stop_ids = mfq::engine::model_stop_token_ids(
                mfq::cuda::load_model_config_json(*model.source, config_path), model.vocab_size());
            if (stop_ids.empty()) std::cerr << "warning: no model EOS token IDs; generation is limited only by --gen\n";
        }
        const auto status=generate_diagnostic_tokens(
            execution, model, ids, gen, profile, t0, t1,decode_inputs,stop_ids);
        if(!status && !router_lookahead_audit.empty()) {
            if constexpr (std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
                qwen4_finish_router_lookahead_audit(model,router_lookahead_audit);
        }
        return status;
            });
    }); }
};

} // namespace

namespace mfq::cuda {
int execute_diagnostics_command(DiagnosticsCommandOptions options) {
    return DiagnosticsCommand(std::move(options)).run();
}
} // namespace mfq::cuda

namespace mfq::cuda::commands {
namespace {

void print_diagnostics_help() {
    std::cout
        << "MFQ CUDA diagnostics\n\n"
        << "Usage:\n"
        << "  mfq-diagnostics [--model MODEL] CHECK [OPTIONS]\n\n"
        << "Checks and benchmarks:\n"
        << "  --check-linear NAME             check one GPU linear\n"
        << "  --check-linear-cpu NAME         check one CPU linear\n"
        << "  --check-linear-group NAMES      check a tensor group\n"
        << "  --check-q8-embedding NAME       check a Q8 embedding\n"
        << "  --check-tp-linear NAME          check tensor-parallel linear\n"
        << "  --check-ep-moe NAME             check expert-parallel MoE\n"
        << "  --check-gdn-input PATH          check GDN operator fixtures\n"
        << "  --check-linear-conv-input PATH  check linear-conv fixtures\n"
        << "  --check-moe-layer N             check a model MoE layer\n"
        << "  --check-mfe-tensor NAME         check an MFE tensor\n"
        << "  --check-dsv4-output-a NAME      check DeepSeek output-A\n"
        << "  --check-attention-decode N      check decode attention\n"
        << "  --check-gemma4-swa              check Gemma4 SWA\n"
        << "  --check-glm-dsa                 check GLM DSA\n"
        << "  --check-dsv4-attention          check DeepSeek-V4 attention\n"
        << "  --check-dsv4-hc                 check DeepSeek-V4 HC\n"
        << "  --check-deepseek-v41            run DeepSeek-V4.1 self-check\n"
        << "  --check-text-session-state      check session state\n"
        << "  --check-qwen35-mtp              check Qwen3.5 MTP\n"
        << "  --check-flash-next              check Flash-Next\n"
        << "  --check-flash-next-mtp          check Flash-Next MTP\n"
        << "  --check-continuous-batching     check continuous batching\n"
        << "  --check-engine-isolation        check two concurrent CUDA engines\n"
        << "  --check-backend-bf16-add        check backend BF16 add\n"
        << "  --check-backend-argmax          check backend argmax\n"
        << "  --check-runtime-assets          validate embedded assets\n"
        << "  --check-mfq-container           validate the MFQ container\n"
        << "  --minicpmo-eval-batch           run MiniCPM-o batch evaluation\n"
        << "  --bench-qwen35-mtp MODE         benchmark ordinary or mtp mode\n"
        << "  --prefill-sweep LIST            benchmark prefill sizes\n"
        << "  --prefill-repeat N              repeat prefill N times\n"
        << "  --compare-block-trace PATH      compare a block trace\n"
        << "  --dump-block-trace PATH         write a block trace\n"
        << "  --compare-dsv4-hc-ops           compare DeepSeek HC operators\n"
        << "  --compare-dsv4-hc-model         compare DeepSeek HC model\n"
        << "  --compare-mma-attention         compare MMA attention\n"
        << "  --compare-decode-splitk         compare decode split-K\n"
        << "  --compare-mma-decode            compare MMA decode\n"
        << "  --compare-nvq-vec4              compare NVQ vec4\n"
        << "  --profile                       profile token generation\n\n"
        << "Common options:\n"
        << "  --model PATH                    model or split-model shard\n"
        << "  --config PATH                   external model config\n"
        << "  --ids LIST | --ids-file PATH    diagnostic token input\n"
        << "  --router-lookahead-audit PREFIX record actual FFN inputs and next-router logits; diagnostic only\n"
        << "  --compare-ffn-transfer-phases     one load, fixed primary tiers and CPU routes; baseline/phased/baseline\n"
        << "  --compare-router-lookahead       one load, fixed primary tiers and CPU routes; baseline/prefetch/baseline\n"
        << "  --compare-rotary-fusion          one load, fixed primary tiers and CPU routes; original/fused/original\n"
        << "  --compare-attention-grouping     one load, fixed primary tiers and CPU routes; separate/grouped/separate\n"
        << "  --compare-mhc-fusion             one load, fixed primary tiers and CPU routes; original/fused/original; honors MFQ_GR_TWO_STAGE_DENSE_INJECTION\n"
        << "  --compare-mhc-shared-norm        one load, fixed primary tiers and CPU routes; fused/shared-norm/fused\n"
        << "  --compare-mhc-dense-fusion       one load, fixed primary tiers and CPU routes; original/fused/original BF16 injection\n"
        << "  --compare-mfe-parallel-gu        one load, fixed primary tiers and CPU routes; serial/parallel/serial Gate-Up\n"
        << "  --compare-mfe-gpu-bundle         one load, fixed primary tiers and CPU routes; current/bundle/current FFN\n"
        << "  --compare-mfe-shared-first       one load, fixed primary tiers and CPU routes; off/on/off shared expert CTA order\n"
        << "  --compare-mfe-early-gate-up      one load, fixed primary tiers and CPU routes; off/on/off resident Gate-Up with cold Down\n"
        << "  --compare-cpu-background-calibration one load; synchronous/background/synchronous, fresh CPU costs\n"
        << "  --compare-cpu-transfer-budget    one load; original/conservative/original CPU policy; report natural route changes\n"
        << "  --compare-router-topk            one load, fixed primary tiers and CPU routes; original/cached/original\n"
        << "  --decode-inputs-file FILE        fixed decode input ids, raw int32, gen-1 entries\n"
        << "  --gen N                         generated tokens\n"
        << "  --ignore-eos                    fixed-length token diagnostics\n"
        << "  --warmup-prefill N              prefill warmups, 0-100; default profile=1, otherwise=0\n"
        << "  --ctx-size N                    context size; 0 selects a default\n"
        << "  -t, --threads N                 positive CPU thread count\n"
        << "  -ngl, --n-gpu-layers N          non-negative GPU layer count\n"
        << "  --cpu-offload-layers RANGES     explicit CPU-offloaded layers\n"
        << "  --moe-gpu-cache-gb N            bounded MoE GPU cache\n"
        << "  --tensor-parallel DEVICES       tensor-parallel devices\n"
        << "  --expert-parallel DEVICES       expert-parallel devices\n"
        << "  --layer-parallel DEVICES        layer-placement devices\n"
        << "  -h, --help                      show this help\n";
}

bool has_diagnostic_action(const DiagnosticsCommandOptions& value) {
    return !value.check_linear.empty() || !value.check_linear_cpu.empty() ||
        !value.check_tp_linear.empty() || !value.check_ep_moe.empty() ||
        !value.check_linear_group.empty() || !value.check_q8_embedding.empty() ||
        !value.check_gdn_input.empty() || !value.check_linear_conv_input.empty() ||
        !value.check_mfe_tensor.empty() || !value.check_dsv4_output_a.empty() ||
        value.check_moe_layer >= 0 || value.check_gemma_geglu_layer >= 0 ||
        value.check_attention_decode > 0 || value.check_backend_bf16_add ||
        value.check_backend_argmax || value.check_gemma4_swa || value.check_glm_dsa ||
        value.check_dsv4_attention || value.check_dsv4_hc ||
        value.check_deepseek_v41 || value.check_text_session_state ||
        value.check_qwen35_mtp || value.check_continuous_batching ||
        value.check_engine_isolation ||
        value.check_flash_next || value.check_flash_next_mtp ||
        value.check_nemotron_h || value.check_nemotron_h_mtp || value.check_nemotron_h_vision ||
        value.check_runtime_assets || value.check_mfq_container ||
        value.compare_dsv4_hc_ops || value.compare_dsv4_hc_model ||
        value.compare_mma_attention || value.compare_decode_splitk ||
        value.compare_mma_decode || value.compare_nvq_vec4 ||
        !value.block_trace_reference.empty() || !value.block_trace_output.empty() ||
        !value.bench_qwen35_mtp.empty() || !value.prefill_sweep_arg.empty() ||
        !value.ids_arg.empty() || !value.ids_file.empty() ||
        value.prefill_repeat > 0 || value.profile || value.minicpmo_eval_batch;
}

DiagnosticsCommandOptions parse_diagnostics(ArgCursor& args) {
    DiagnosticsCommandOptions result;
    while (!args.empty()) {
        const std::string_view option = args.next();
        if (option == "--help" || option == "-h") {
            print_diagnostics_help();
            throw HelpRequested{};
        }
        if (parse_cuda_load_option(option, args, result) ||
                parse_token_input_option(option, args, result)) {
            continue;
        }
        if (option == "--minicpmo-eval-batch") result.minicpmo_eval_batch = true;
        else if (option == "--minicpmo-eval-vision-batch-size") {
            result.minicpmo_eval_vision_batch_size =
                integer<int64_t>(args.value(option), option);
            if (result.minicpmo_eval_vision_batch_size <= 0) {
                usage_error("--minicpmo-eval-vision-batch-size must be positive");
            }
        }
        else if (option == "--check-linear") result.check_linear = args.value(option);
        else if (option == "--check-linear-cpu") result.check_linear_cpu = args.value(option);
        else if (option == "--check-tp-linear") result.check_tp_linear = args.value(option);
        else if (option == "--check-tp-axis") result.check_tp_axis_arg = args.value(option);
        else if (option == "--check-tp-m") result.check_tp_m = integer<int>(args.value(option), option);
        else if (option == "--check-ep-moe" || option == "--check-tp-moe") {
            result.check_ep_moe = args.value(option);
        }
        else if (option == "--check-ep-moe-tokens" ||
                option == "--check-tp-moe-tokens") {
            result.check_ep_moe_tokens = integer<int>(args.value(option), option);
        }
        else if (option == "--check-ep-moe-routes" ||
                option == "--check-tp-moe-routes") {
            result.check_ep_moe_routes = integer<int>(args.value(option), option);
        }
        else if (option == "--check-linear-group") result.check_linear_group = args.value(option);
        else if (option == "--check-q8-embedding") result.check_q8_embedding = args.value(option);
        else if (option == "--check-gdn-input") result.check_gdn_input = args.value(option);
        else if (option == "--check-gdn-output") result.check_gdn_output = args.value(option);
        else if (option == "--check-gdn-state") result.check_gdn_state = args.value(option);
        else if (option == "--check-gdn-tokens") result.check_gdn_tokens = integer<int>(args.value(option), option);
        else if (option == "--check-gdn-q-heads") result.check_gdn_q_heads = integer<int>(args.value(option), option);
        else if (option == "--check-gdn-v-heads") result.check_gdn_v_heads = integer<int>(args.value(option), option);
        else if (option == "--check-gdn-head-dim") result.check_gdn_head_dim = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-input") result.check_linear_conv_input = args.value(option);
        else if (option == "--check-linear-conv-output") result.check_linear_conv_output = args.value(option);
        else if (option == "--check-linear-conv-tokens") result.check_linear_conv_tokens = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-q-heads") result.check_linear_conv_q_heads = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-v-heads") result.check_linear_conv_v_heads = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-key-dim") result.check_linear_conv_key_dim = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-value-dim") result.check_linear_conv_value_dim = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-kernel") result.check_linear_conv_kernel = integer<int>(args.value(option), option);
        else if (option == "--check-linear-m") result.check_linear_m = integer<int>(args.value(option), option);
        else if (option == "--check-linear-reps") result.check_linear_reps = integer<int>(args.value(option), option);
        else if (option == "--check-linear-gate") result.check_linear_gate = args.value(option);
        else if (option == "--check-gemma-geglu-layer") result.check_gemma_geglu_layer = integer<int>(args.value(option), option);
        else if (option == "--check-gemma-geglu-reps") result.check_gemma_geglu_reps = integer<int>(args.value(option), option);
        else if (option == "--check-moe-layer") result.check_moe_layer = integer<int>(args.value(option), option);
        else if (option == "--check-moe-tokens") {
            result.check_moe_tokens = args.value(option);
            validate_integer_list(result.check_moe_tokens, option);
        }
        else if (option == "--check-moe-reps") result.check_moe_reps = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-tensor") result.check_mfe_tensor = args.value(option);
        else if (option == "--check-mfe-tokens") result.check_mfe_tokens = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-routes") result.check_mfe_routes = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-reps") result.check_mfe_reps = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-split-width") result.check_mfe_split_width = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-routed-input") result.check_mfe_routed_input = true;
        else if (option == "--check-mfe-benchmark-only") result.check_mfe_benchmark_only = true;
        else if (option == "--check-dsv4-output-a") result.check_dsv4_output_a = args.value(option);
        else if (option == "--check-dsv4-output-a-batch") result.check_dsv4_output_a_batch = integer<int>(args.value(option), option);
        else if (option == "--check-dsv4-output-a-reps") result.check_dsv4_output_a_reps = integer<int>(args.value(option), option);
        else if (option == "--check-attention-decode") result.check_attention_decode = integer<int>(args.value(option), option);
        else if (option == "--check-attention-reps") result.check_attention_reps = integer<int>(args.value(option), option);
        else if (option == "--check-attention-head-dim") result.check_attention_head_dim = integer<int>(args.value(option), option);
        else if (option == "--check-attention-window") result.check_attention_window = integer<int>(args.value(option), option);
        else if (option == "--check-attention-swa-decode") result.check_attention_swa_decode = true;
        else if (option == "--check-gemma4-swa") result.check_gemma4_swa = true;
        else if (option == "--check-glm-dsa") result.check_glm_dsa = true;
        else if (option == "--check-dsv4-attention") result.check_dsv4_attention = true;
        else if (option == "--check-dsv4-hc") result.check_dsv4_hc = true;
        else if (option == "--check-deepseek-v41") result.check_deepseek_v41 = true;
        else if (option == "--check-text-session-state") result.check_text_session_state = true;
        else if (option == "--check-qwen35-mtp") result.check_qwen35_mtp = true;
        else if(option=="--check-nemotron-h")result.check_nemotron_h=true;
        else if(option=="--check-nemotron-h-mtp")result.check_nemotron_h_mtp=true;
        else if(option=="--check-nemotron-h-vision")result.check_nemotron_h_vision=true;
        else if (option == "--check-flash-next") result.check_flash_next = true;
        else if (option == "--check-flash-next-mtp") result.check_flash_next_mtp = true;
        else if (option == "--bench-qwen35-mtp") result.bench_qwen35_mtp = args.value(option);
        else if (option == "--bench-qwen35-mtp-reps") result.bench_qwen35_mtp_reps = integer<int>(args.value(option), option);
        else if (option == "--compare-dsv4-hc-ops") result.compare_dsv4_hc_ops = true;
        else if (option == "--compare-dsv4-hc-model") result.compare_dsv4_hc_model = true;
        else if (option == "--kl-mmq") {
            result.kl_mmq_arg = args.value(option);
            try { (void)parse_kl_mmq_mode(result.kl_mmq_arg); }
            catch (const std::runtime_error& error) {
                usage_error(error.what());
            }
        }
        else if (option == "--compare-block-trace") result.block_trace_reference = args.value(option);
        else if (option == "--dump-block-trace") result.block_trace_output = args.value(option);
        else if (option == "--dump-block-trace-start") result.block_trace_start = integer<int64_t>(args.value(option), option);
        else if (option == "--dump-block-trace-count") result.block_trace_count = integer<int64_t>(args.value(option), option);
        else if (option == "--prefill-repeat") result.prefill_repeat = integer<int>(args.value(option), option);
        else if (option == "--prefill-sweep") {
            result.prefill_sweep_arg = args.value(option);
            validate_integer_list(result.prefill_sweep_arg, option);
        }
        else if (option == "--prefill-sweep-reps") result.prefill_sweep_reps = integer<int>(args.value(option), option);
        else if (option == "--check-continuous-batching") result.check_continuous_batching = true;
        else if (option == "--check-engine-isolation") result.check_engine_isolation = true;
        else if (option == "--check-runtime-assets") result.check_runtime_assets = true;
        else if (option == "--check-mfq-container") result.check_mfq_container = true;
        else if (option == "--check-tokenizer-text") result.check_tokenizer_text = args.value(option);
        else if (option == "--router-lookahead-audit") result.router_lookahead_audit = args.value(option);
        else if (option == "--compare-ffn-transfer-phases") result.compare_ffn_transfer_phases = true;
        else if (option == "--compare-router-lookahead") result.compare_router_lookahead = true;
        else if (option == "--compare-rotary-fusion") result.compare_rotary_fusion = true;
        else if (option == "--compare-attention-grouping") result.compare_attention_grouping = true;
        else if (option == "--compare-mhc-fusion") result.compare_mhc_fusion = true;
        else if (option == "--compare-mhc-shared-norm") result.compare_mhc_shared_norm = true;
        else if (option == "--compare-mhc-dense-fusion") result.compare_mhc_dense_fusion = true;
        else if (option == "--compare-mfe-parallel-gu") result.compare_mfe_parallel_gu = true;
        else if (option == "--compare-mfe-gpu-bundle") result.compare_mfe_gpu_bundle = true;
        else if (option == "--compare-mfe-shared-first") result.compare_mfe_shared_first = true;
        else if (option == "--compare-mfe-early-gate-up") result.compare_mfe_early_gu = true;
        else if (option == "--compare-cpu-background-calibration") result.compare_cpu_background_calibration = true;
        else if (option == "--compare-cpu-transfer-budget") result.compare_cpu_transfer_budget = true;
        else if (option == "--compare-router-topk") result.compare_router_topk = true;
        else if (option == "--decode-inputs-file") result.decode_inputs_file = args.value(option);
        else if (option == "--profile") result.profile = true;
        else if (option == "--ignore-eos") result.ignore_eos = true;
        else if (option == "--warmup-prefill") {
            result.warmup_prefill = integer<int>(args.value(option), option);
            if (result.warmup_prefill < 0 || result.warmup_prefill > 100)
                usage_error("--warmup-prefill requires a count from 0 to 100");
        }
        else if (option == "--check-backend-bf16-add") result.check_backend_bf16_add = true;
        else if (option == "--check-backend-argmax") result.check_backend_argmax = true;
        else if (option == "--compare-mma-attention") result.compare_mma_attention = true;
        else if (option == "--compare-decode-splitk") result.compare_decode_splitk = true;
        else if (option == "--compare-mma-decode") result.compare_mma_decode = true;
        else if (option == "--compare-mma-decode-steps") result.compare_mma_decode_steps = integer<int>(args.value(option), option);
        else if (option == "--compare-mma-decode-planned-len") result.compare_mma_decode_planned_len = integer<int>(args.value(option), option);
        else if (option == "--compare-nvq-vec4" || option == "--compare-niq-vec4") result.compare_nvq_vec4 = true;
        else usage_error("unknown option: " + std::string(option));
    }
    if (!result.decode_inputs_file.empty()) {
        result.decode_inputs = load_ids_file(result.decode_inputs_file);
        if (result.gen <= 1 || result.decode_inputs.size() != size_t(result.gen - 1))
            usage_error("--decode-inputs-file requires exactly gen-1 entries and gen greater than 1");
        if (std::any_of(result.decode_inputs.begin(),result.decode_inputs.end(),[](int64_t id){return id<0;}))
            usage_error("--decode-inputs-file requires nonnegative token ids");
    }
    if (!result.ids_arg.empty() && !result.ids_file.empty()) {
        usage_error("--ids and --ids-file are mutually exclusive");
    }
    if (!has_diagnostic_action(result)) {
        usage_error("mfq-diagnostics requires a check or benchmark");
    }
    return result;
}

} // namespace

int run_diagnostics(int argc, char** argv) {
    try {
        ArgCursor args(argc, argv);
        auto options = parse_diagnostics(args);
        return mfq::cuda::execute_diagnostics_command(std::move(options));
    } catch (const HelpRequested&) {
        return 0;
    } catch (const UsageError& error) {
        return print_usage_error(error);
    }
}

} // namespace mfq::cuda::commands
