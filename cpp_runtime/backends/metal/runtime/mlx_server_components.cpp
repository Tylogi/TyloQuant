#include "mlx_server_components.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <malloc/malloc.h>

namespace mfq::metal {
namespace {

mlx::core::Shape checked_shape(
    const std::vector<std::int64_t>& dimensions,
    const char* name) {
    mlx::core::Shape shape;
    shape.reserve(dimensions.size());
    for (const auto dimension : dimensions) {
        if (dimension <= 0 ||
            dimension > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                std::string(name) + " has an invalid MLX dimension");
        }
        shape.push_back(static_cast<int>(dimension));
    }
    return shape;
}

MlxSamplingParams metal_sampling(const MfqSamplingParams& sampling) {
    MlxSamplingParams result;
    result.temperature = sampling.temperature;
    result.top_k = sampling.top_k;
    result.top_p = sampling.top_p;
    result.presence_penalty = sampling.presence_penalty;
    result.frequency_penalty = sampling.frequency_penalty;
    result.repetition_penalty = sampling.repetition_penalty;
    result.enable_mtp = sampling.enable_mtp;
    result.seed = sampling.seed;
    return result;
}

const MfqGraphComponent* component_with_implementation(
    const MfqModelGraph* graph,
    std::string_view kind,
    std::string_view implementation) {
    if (graph == nullptr) return nullptr;
    const auto* component = graph->component(kind);
    return component != nullptr && component->implementation == implementation
        ? component : nullptr;
}

std::vector<MlxGridShape> grid_shapes(
    const std::vector<std::int32_t>& values) {
    if (values.size() % 3 != 0) {
        throw std::invalid_argument(
            "grid-Vision THW inventory is invalid");
    }
    std::vector<MlxGridShape> result;
    result.reserve(values.size() / 3);
    for (std::size_t index = 0; index < values.size(); index += 3) {
        result.push_back({
            values[index], values[index + 1], values[index + 2]});
    }
    return result;
}

} // namespace

MlxServerComponentCallbacks make_mlx_server_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxQwen35CausalLm>> runtime_holder,
    mlx::core::Stream runtime_stream) {
    MlxServerComponentCallbacks result;
    result.duplex.name = "metal";
    result.mtp_available =
        component_with_implementation(
            graph, "predictor", "next_token_prediction") != nullptr &&
        runtime_holder->has_value() &&
        runtime_holder->value().supports_mtp();
    if (component_with_implementation(
            graph, "vision", "grid_vit") == nullptr ||
        !runtime_holder->has_value() ||
        !runtime_holder->value().supports_multimodal()) {
        return result;
    }
    result.multimodal_generate =
        [runtime_mutex = std::move(runtime_mutex),
         runtime_holder = std::move(runtime_holder), runtime_stream](
            const std::vector<std::int64_t>& prompt,
            const MfqMultimodalInput& media,
            const MfqSamplingParams& sampling,
            const MfqTokenCallback& callback,
            const MfqPrefillCallback& on_prefill,
            const MfqTokenConstraintPtr& token_constraint) {
            std::lock_guard<std::mutex> lock(*runtime_mutex);
            if (!runtime_holder->has_value()) {
                throw std::runtime_error(
                    "grid-Vision runtime is unavailable after a failed reload");
            }
            auto& runtime = runtime_holder->value();
            if (media.processor != MfqMultimodalProcessor::grid_vision ||
                media.processor_name != runtime.multimodal_input_contract()) {
                throw std::invalid_argument(
                    "media payload disagrees with the model input contract");
            }
            const auto pixel_shape = checked_shape(
                media.pixel_shape, "pixel_values");
            if (pixel_shape.size() != 2 ||
                media.vision_grid.size() % 3 != 0 ||
                media.vision_types.size() != media.vision_grid.size() / 3) {
                throw std::invalid_argument(
                    "grid-Vision media geometry is invalid");
            }
            MlxGridMediaInput input;
            input.processor = media.processor_name;
            input.pixel_values = mlx::core::array(
                media.pixel_values.begin(), pixel_shape,
                mlx::core::float32);
            input.media_grids = grid_shapes(media.vision_grid);
            input.media_types = media.vision_types;
            input.image_grids = grid_shapes(media.image_grid);
            input.video_grids = grid_shapes(media.video_grid);

            mlx::core::set_default_device(mlx::core::Device::gpu);
            mlx::core::set_default_stream(runtime_stream);
            const auto component_started =
                std::chrono::steady_clock::now();
            auto prepared = runtime.prepare_multimodal_prompt(prompt, input);
            if (!prepared.embeddings) {
                throw std::runtime_error(
                    "multimodal component did not produce embeddings");
            }
            prepared.embeddings->eval();
            mlx::core::synchronize(runtime_stream);
            const auto component_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - component_started)
                    .count();

            std::function<void(std::size_t, double)> report_prefill;
            if (on_prefill) {
                report_prefill = [on_prefill, component_ms](
                    std::size_t tokens, double llm_ms) {
                    on_prefill(MfqPrefillTiming{
                        tokens, llm_ms, component_ms,
                        llm_ms + component_ms});
                };
            }
            return runtime.generate_prepared(
                prepared,
                metal_sampling(sampling),
                sampling.max_tokens,
                callback,
                report_prefill,
                token_constraint);
        };
    return result;
}

MlxServerComponentCallbacks make_mlx_server_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxMiniCPMO45Runtime>> runtime_holder,
    mlx::core::Stream runtime_stream) {
    MlxServerComponentCallbacks result;
    result.duplex.name = "metal";
    if (component_with_implementation(
            graph, "vision", "minicpmo45_vision") != nullptr) {
        result.multimodal_generate =
            [runtime_mutex, runtime_holder, runtime_stream](
                const std::vector<std::int64_t>& prompt,
                const MfqMultimodalInput& media,
                const MfqSamplingParams& sampling,
                const MfqTokenCallback& callback,
                const MfqPrefillCallback& on_prefill,
                const MfqTokenConstraintPtr& token_constraint) {
                std::lock_guard<std::mutex> lock(*runtime_mutex);
                if (!runtime_holder->has_value()) {
                    throw std::runtime_error(
                        "MiniCPM-o runtime is unavailable after a failed reload");
                }
                if (media.processor != MfqMultimodalProcessor::minicpmo) {
                    throw std::invalid_argument(
                        "MiniCPM-o received a foreign multimodal payload");
                }
                if (media.image_bounds.size() % 4 != 0 ||
                    media.audio_bounds.size() % 4 != 0) {
                    throw std::invalid_argument(
                        "MiniCPM-o media bounds are invalid");
                }
                mlx::core::set_default_device(mlx::core::Device::gpu);
                mlx::core::set_default_stream(runtime_stream);

                std::vector<std::int32_t> prompt_ids;
                prompt_ids.reserve(prompt.size());
                for (const auto token : prompt) {
                    if (token < 0 ||
                        token > std::numeric_limits<std::int32_t>::max()) {
                        throw std::invalid_argument(
                            "MiniCPM-o prompt token is out of range");
                    }
                    prompt_ids.push_back(static_cast<std::int32_t>(token));
                }
                MlxMiniCPMO45Inputs inputs{
                    mlx::core::array(
                        prompt_ids.begin(),
                        mlx::core::Shape{
                            1, static_cast<int>(prompt_ids.size())},
                        mlx::core::int32),
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                };
                if (!media.image_bounds.empty()) {
                    inputs.pixel_values = mlx::core::array(
                        media.pixel_values.begin(),
                        checked_shape(media.pixel_shape, "pixel_values"),
                        mlx::core::float32);
                    inputs.patch_mask = mlx::core::astype(
                        mlx::core::array(
                            media.patch_mask.begin(),
                            checked_shape(
                                media.patch_mask_shape, "patch_mask"),
                            mlx::core::uint8),
                        mlx::core::bool_);
                    inputs.target_sizes = mlx::core::array(
                        media.target_sizes.begin(),
                        checked_shape(
                            media.target_sizes_shape, "target_sizes"),
                        mlx::core::int32);
                    inputs.image_bounds = mlx::core::array(
                        media.image_bounds.begin(),
                        mlx::core::Shape{
                            static_cast<int>(media.image_bounds.size() / 4),
                            4},
                        mlx::core::int64);
                }
                if (!media.audio_bounds.empty()) {
                    inputs.audio_features = mlx::core::array(
                        media.audio_features.begin(),
                        checked_shape(
                            media.audio_features_shape, "audio_features"),
                        mlx::core::float32);
                    inputs.audio_lengths = mlx::core::array(
                        media.audio_lengths.begin(),
                        mlx::core::Shape{
                            static_cast<int>(media.audio_lengths.size())},
                        mlx::core::int64);
                    inputs.audio_bounds = mlx::core::array(
                        media.audio_bounds.begin(),
                        mlx::core::Shape{
                            static_cast<int>(media.audio_bounds.size() / 4),
                            4},
                        mlx::core::int64);
                }

                std::function<void(std::size_t, double, double, double)>
                    report_prefill;
                if (on_prefill) {
                    report_prefill = [on_prefill](
                        std::size_t tokens,
                        double llm_ms,
                        double multimodal_ms,
                        double model_ms) {
                        on_prefill(MfqPrefillTiming{
                            tokens, llm_ms, multimodal_ms, model_ms});
                    };
                }
                return runtime_holder->value().generate_multimodal(
                    inputs,
                    metal_sampling(sampling),
                    sampling.max_tokens,
                    callback,
                    report_prefill,
                    token_constraint);
            };
    }

    const bool duplex_declared =
        component_with_implementation(
            graph, "duplex", "minicpmo45_duplex") != nullptr ||
        (graph != nullptr && graph->has_capability("realtime_duplex") &&
         graph->has_component("audio_input") &&
         graph->has_component("audio_output"));
    if (!duplex_declared) {
        return result;
    }
    result.duplex.start =
        [runtime_mutex, runtime_holder, runtime_stream](
            const MfqDuplexSessionParams& parameters) {
            if (parameters.special_ids.size() != 15) {
                throw std::invalid_argument(
                    "MiniCPM-o duplex requires 15 special token IDs");
            }
            std::lock_guard<std::mutex> lock(*runtime_mutex);
            if (!runtime_holder->has_value()) {
                throw std::runtime_error(
                    "MiniCPM-o runtime is unavailable");
            }
            mlx::core::set_default_device(mlx::core::Device::gpu);
            mlx::core::set_default_stream(runtime_stream);
            auto& runtime = runtime_holder->value();
            runtime.reset();
            mlx::core::clear_cache();

            MlxMiniCPMO45DuplexConfig config;
            auto& ids = config.special_ids;
            ids.unit_start = parameters.special_ids[0];
            ids.unit_end = parameters.special_ids[1];
            ids.image_start = parameters.special_ids[2];
            ids.image_end = parameters.special_ids[3];
            ids.slice_start = parameters.special_ids[4];
            ids.slice_end = parameters.special_ids[5];
            ids.listen = parameters.special_ids[6];
            ids.speak = parameters.special_ids[7];
            ids.tts_bos = parameters.special_ids[8];
            ids.tts_eos = parameters.special_ids[9];
            ids.chunk_eos = parameters.special_ids[10];
            ids.chunk_tts_eos = parameters.special_ids[11];
            ids.turn_eos = parameters.special_ids[12];
            ids.tts_pad = parameters.special_ids[13];
            ids.audio_bos = parameters.special_ids[14];
            config.forbidden_ids = parameters.forbidden_ids;
            config.greedy = parameters.greedy;
            config.temperature = parameters.temperature;
            config.top_k = parameters.top_k;
            config.top_p = parameters.top_p;
            config.listen_probability_scale =
                parameters.listen_probability_scale;
            config.repetition_penalty = parameters.repetition_penalty;
            config.repetition_window = parameters.repetition_window;
            config.length_penalty = parameters.length_penalty;
            config.tts_temperature = parameters.tts_temperature;
            config.tts_repetition_penalty =
                parameters.tts_repetition_penalty;
            config.seed = parameters.seed;

            std::optional<mlx::core::array> system_prefix_ids;
            if (!parameters.system_prefix.empty()) {
                system_prefix_ids.emplace(
                    parameters.system_prefix.begin(),
                    mlx::core::Shape{
                        1,
                        static_cast<int>(parameters.system_prefix.size())},
                    mlx::core::int64);
            }
            std::optional<mlx::core::array> reference_features;
            if (!parameters.reference_audio_features.empty()) {
                if (parameters.reference_audio_frames <= 0 ||
                    parameters.reference_audio_features.size() !=
                        static_cast<std::size_t>(
                            parameters.reference_audio_frames) * 80) {
                    throw std::invalid_argument(
                        "MiniCPM-o reference Mel geometry is invalid");
                }
                reference_features.emplace(
                    parameters.reference_audio_features.begin(),
                    mlx::core::Shape{
                        1, 80, parameters.reference_audio_frames},
                    mlx::core::float32);
            }
            std::optional<mlx::core::array> system_suffix_ids;
            if (!parameters.system_suffix.empty()) {
                system_suffix_ids.emplace(
                    parameters.system_suffix.begin(),
                    mlx::core::Shape{
                        1,
                        static_cast<int>(parameters.system_suffix.size())},
                    mlx::core::int64);
            }
            runtime.prepare_duplex(
                config,
                system_prefix_ids,
                reference_features,
                system_suffix_ids);
            mlx::core::synchronize(runtime_stream);
        };
    result.duplex.step =
        [runtime_mutex, runtime_holder, runtime_stream](
            const MfqDuplexStepInput& input) {
            const bool has_audio = input.audio_frames > 0;
            const bool has_text = !input.text_tokens.empty();
            if (has_audio &&
                input.audio_features.size() !=
                    static_cast<std::size_t>(input.audio_frames) * 80) {
                throw std::invalid_argument(
                    "MiniCPM-o duplex Mel geometry is invalid");
            }
            if (!has_audio && !has_text) {
                throw std::invalid_argument(
                    "MiniCPM-o duplex step has no input");
            }
            std::lock_guard<std::mutex> lock(*runtime_mutex);
            if (!runtime_holder->has_value()) {
                throw std::runtime_error(
                    "MiniCPM-o runtime is unavailable");
            }
            mlx::core::set_default_device(mlx::core::Device::gpu);
            mlx::core::set_default_stream(runtime_stream);
            auto& runtime = runtime_holder->value();
            if (!runtime.duplex_prepared()) {
                throw std::runtime_error(
                    "MiniCPM-o duplex session is not prepared");
            }

            MlxMiniCPMO45DuplexInputs inputs;
            if (has_audio) {
                inputs.audio_features.emplace(
                    input.audio_features.begin(),
                    mlx::core::Shape{1, 80, input.audio_frames},
                    mlx::core::float32);
            }
            if (has_text) {
                inputs.text_ids.emplace(
                    input.text_tokens.begin(),
                    mlx::core::Shape{
                        1, static_cast<int>(input.text_tokens.size())},
                    mlx::core::int64);
            }
            inputs.audio_prefix_extra_frames =
                input.audio_prefix_extra_frames;
            inputs.audio_suffix_extra_frames =
                input.audio_suffix_extra_frames;
            inputs.max_new_speak_tokens = input.max_new_speak_tokens;
            inputs.force_listen = input.force_listen;
            inputs.force_speak = input.force_speak;

            const auto started = std::chrono::steady_clock::now();
            auto generated = runtime.duplex_step(inputs);
            generated.generated_ids.eval();
            generated.tts_codes.eval();
            mlx::core::synchronize(runtime_stream);

            MfqDuplexStepResult response;
            const auto* generated_ids =
                generated.generated_ids.data<std::int64_t>();
            response.generated_tokens.assign(
                generated_ids,
                generated_ids + generated.generated_ids.size());
            const auto* codes = generated.tts_codes.data<std::int32_t>();
            response.audio_tokens.assign(
                codes, codes + generated.tts_codes.size());
            response.is_listen = generated.is_listen;
            response.end_of_turn = generated.end_of_turn;
            response.tts_force_flush = generated.tts_force_flush;
            response.audio_chunk_index = generated.audio_chunk_index;
            response.language_cache_position =
                generated.language_cache_position;
            response.audio_cache_position = generated.audio_cache_position;
            response.tts_cache_position = generated.tts_cache_position;
            response.inference_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started)
                    .count();
            return response;
        };
    result.duplex.stop =
        [runtime_mutex = std::move(runtime_mutex),
         runtime_holder = std::move(runtime_holder), runtime_stream]() {
            std::lock_guard<std::mutex> lock(*runtime_mutex);
            if (!runtime_holder->has_value()) return;
            mlx::core::set_default_device(mlx::core::Device::gpu);
            mlx::core::set_default_stream(runtime_stream);
            runtime_holder->value().reset();
            mlx::core::synchronize(runtime_stream);
            mlx::core::clear_cache();
            malloc_zone_pressure_relief(nullptr, 0);
        };
    return result;
}

MlxServerComponentCallbacks make_mlx_server_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxDeepseekV4CausalLm>> runtime_holder,
    mlx::core::Stream runtime_stream) {
    MlxServerComponentCallbacks result;
    result.duplex.name = "metal";
    const bool legacy_hf = graph == nullptr;
    result.mtp_available =
        (legacy_hf || component_with_implementation(
            graph, "predictor", "dspark") != nullptr) &&
        runtime_holder->has_value() &&
        runtime_holder->value().supports_mtp();
    if ((!legacy_hf && component_with_implementation(
             graph, "vision", "deepseek_v4_vision") == nullptr) ||
        !runtime_holder->has_value() ||
        !runtime_holder->value().supports_multimodal()) {
        return result;
    }
    result.multimodal_generate =
        [runtime_mutex = std::move(runtime_mutex),
         runtime_holder = std::move(runtime_holder), runtime_stream](
            const std::vector<std::int64_t>& prompt,
            const MfqMultimodalInput& media,
            const MfqSamplingParams& sampling,
            const MfqTokenCallback& callback,
            const MfqPrefillCallback& on_prefill,
            const MfqTokenConstraintPtr& token_constraint) {
            std::lock_guard<std::mutex> lock(*runtime_mutex);
            if (!runtime_holder->has_value()) {
                throw std::runtime_error(
                    "DeepSeek-V4 runtime is unavailable after a failed reload");
            }
            if (media.processor != MfqMultimodalProcessor::deepseek_v4) {
                throw std::invalid_argument(
                    "DeepSeek-V4 received a foreign multimodal payload");
            }
            mlx::core::set_default_device(mlx::core::Device::gpu);
            mlx::core::set_default_stream(runtime_stream);
            const auto& config = runtime_holder->value().config();
            const auto pixel_shape = checked_shape(
                media.pixel_shape, "pixel_values");
            if (pixel_shape.size() != 3 ||
                media.image_bounds.size() % 4 != 0 ||
                media.image_permutation_offsets.size() !=
                    media.image_bounds.size() / 4 + 1 ||
                media.vision_grid.size() !=
                    media.image_bounds.size() / 4 * 4) {
                throw std::invalid_argument(
                    "DeepSeek-V4 media payload geometry is invalid");
            }
            const auto pixels = mlx::core::array(
                media.pixel_values.begin(), pixel_shape,
                mlx::core::float32);
            std::vector<MlxDeepseekV4ImageInput> images;
            const std::size_t count = media.image_bounds.size() / 4;
            images.reserve(count);
            for (std::size_t source = 0; source < count; ++source) {
                const int bound_source = static_cast<int>(
                    media.image_bounds[4 * source + 1]);
                const int begin = static_cast<int>(
                    media.image_bounds[4 * source + 2]);
                const int end = static_cast<int>(
                    media.image_bounds[4 * source + 3]);
                if (bound_source != static_cast<int>(source) ||
                    begin < 0 || end <= begin ||
                    end > static_cast<int>(prompt.size())) {
                    throw std::invalid_argument(
                        "DeepSeek-V4 image bound is invalid");
                }
                const int vit_h = media.vision_grid[4 * source];
                const int vit_w = media.vision_grid[4 * source + 1];
                const int active = vit_h * vit_w;
                auto image_pixels = mlx::core::slice(
                    pixels,
                    mlx::core::Shape{static_cast<int>(source), 0, 0},
                    mlx::core::Shape{
                        static_cast<int>(source + 1), active,
                        pixel_shape[2]});
                image_pixels = mlx::core::reshape(
                    image_pixels,
                    mlx::core::Shape{active, pixel_shape[2]});
                std::vector<std::int64_t> types;
                types.reserve(static_cast<std::size_t>(end - begin));
                for (int position = begin; position < end; ++position) {
                    const auto type =
                        prompt[static_cast<std::size_t>(position)] -
                        config.vocab;
                    if (type < 0 || type > 4) {
                        throw std::invalid_argument(
                            "DeepSeek-V4 image span contains a text token");
                    }
                    types.push_back(type);
                }
                const auto perm_begin =
                    media.image_permutation_offsets[source];
                const auto perm_end =
                    media.image_permutation_offsets[source + 1];
                if (perm_begin < 0 || perm_end < perm_begin ||
                    static_cast<std::size_t>(perm_end) >
                        media.image_permutation.size()) {
                    throw std::invalid_argument(
                        "DeepSeek-V4 image permutation offset is invalid");
                }
                std::vector<std::int32_t> permutation;
                permutation.reserve(
                    static_cast<std::size_t>(perm_end - perm_begin));
                for (auto index = perm_begin; index < perm_end; ++index) {
                    const auto value = media.image_permutation[
                        static_cast<std::size_t>(index)];
                    if (value < 0 ||
                        value > std::numeric_limits<std::int32_t>::max()) {
                        throw std::invalid_argument(
                            "DeepSeek-V4 image permutation is invalid");
                    }
                    permutation.push_back(static_cast<std::int32_t>(value));
                }
                images.push_back({
                    std::move(image_pixels),
                    vit_h,
                    vit_w,
                    begin,
                    end,
                    std::move(types),
                    std::move(permutation),
                });
            }
            std::function<void(std::size_t, double)> report_prefill;
            if (on_prefill) {
                report_prefill = [on_prefill](
                    std::size_t tokens, double model_ms) {
                    on_prefill(MfqPrefillTiming{
                        tokens, model_ms, 0.0, model_ms});
                };
            }
            return runtime_holder->value().generate_multimodal(
                prompt,
                images,
                metal_sampling(sampling),
                sampling.max_tokens,
                callback,
                std::nullopt,
                report_prefill,
                token_constraint);
        };
    return result;
}

} // namespace mfq::metal
