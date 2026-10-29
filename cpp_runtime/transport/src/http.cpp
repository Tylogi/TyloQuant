#include "mfq/fatal_diagnostics.h"
#include "common.h"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace {

using namespace mfq::transport_detail;

static json error_body(const std::string & message, const std::string & type, const std::string & param = {}) {
    return {
        {"error", {
            {"message", message},
            {"type", type},
            {"param", param.empty() ? json(nullptr) : json(param)},
            {"code", json(nullptr)},
        }},
    };
}

static void set_json(httplib::Response & res, const json & body, int status = 200) {
    res.status = status;
    res.set_content(body.dump(), "application/json; charset=utf-8");
}

static bool write_sse(httplib::DataSink & sink, const json & value) {
    const std::string event = "data: " + value.dump() + "\n\n";
    return sink.write(event.data(), event.size());
}

static bool authorized(const httplib::Request & req, httplib::Response & res, const std::string & api_key) {
    if (api_key.empty()) return true;
    const std::string expected = "Bearer " + api_key;
    if (req.get_header_value("Authorization") == expected) return true;
    res.set_header("WWW-Authenticate", "Bearer");
    set_json(res, error_body("invalid API key", "authentication_error"), 401);
    return false;
}

static json parse_body(const httplib::Request & req) {
    try {
        return json::parse(req.body);
    } catch (const json::parse_error & error) {
        throw ApiError(400, "invalid_request_error", std::string("invalid JSON: ") + error.what());
    }
}

static json stream_error_body(const ApiError & error) {
    // The stream has already committed HTTP 200, so the status code travels
    // inside the frame like the stdio transport's error frames.
    return {
        {"error", {
            {"code", error.type},
            {"message", error.what()},
            {"status_code", error.status},
            {"retryable", (error.status == 429 || error.status == 502 ||
                error.status == 503 || error.status == 504) &&
                error.type != "deadline_exceeded"},
        }},
    };
}

static void handle_api_error(httplib::Response & res, const ApiError & error) {
    set_json(res, error_body(error.what(), error.type, error.param), error.status);
}

static ApiError api_error(const mfq::engine::InferenceInputError & error) {
    const char * type = "invalid_request_error";
    if (error.code == mfq::engine::InferenceInputErrorCode::Unsupported) {
        type = "unsupported_parameter";
    } else if (error.code ==
            mfq::engine::InferenceInputErrorCode::ContextLength) {
        type = "context_length_exceeded";
    }
    return ApiError(400, type, error.what(), error.field);
}

int run_mfq_http_transport(
        const MfqHttpRuntimeTransportConfig & config,
        const MfqScheduler & scheduler) {
    const bool duplex = scheduler.info().duplex;
    if (config.port < 1 || config.port > 65535) {
        throw std::runtime_error("runtime transport port must be in [1, 65535]");
    }

    const MfqSamplingParams sampling_defaults =
        default_sampling_params(config);
    const json duplex_sampling_defaults =
        duplex_profile_json(config.runtime_profile.duplex);
    const json tts_sampling_defaults =
        tts_profile_json(config.runtime_profile.tts);
    const std::string duplex_backend_name =
        "cuda";
    const json chat_template_capabilities =
        chat_template_capabilities_json(
            scheduler.chat_template_capabilities());
    const auto model_capability_profile = config.model_capabilities
        ? *config.model_capabilities
        : architecture_capability_profile(config.model_type);
    const json model_capabilities =
        model_capability_profile_json(model_capability_profile);
    const bool vision_supported =
        model_capability_profile.image_input ||
        model_capability_profile.video_input;
    const bool vision_available =
        scheduler.supports_multimodal_generation() &&
        model_capability_profile.image_input;
    const bool video_available =
        scheduler.supports_multimodal_generation() &&
        model_capability_profile.video_input;

    httplib::Server server;
    RuntimeRequestMetrics request_metrics_store;
    std::atomic<int64_t> active_context{config.max_context};
    std::atomic<bool> reloading{false};
    std::mutex duplex_gate;
    std::string duplex_session_id;
    httplib::ws::WebSocket * duplex_socket = nullptr;
    bool duplex_backend_started = false;

    const auto duplex_is_active = [&]() {
        std::lock_guard<std::mutex> lock(duplex_gate);
        return !duplex_session_id.empty();
    };
    const auto stop_duplex_session = [&](const std::string & session_id,
                                         bool close_socket) {
        httplib::ws::WebSocket * socket = nullptr;
        bool stop_backend = false;
        {
            std::lock_guard<std::mutex> lock(duplex_gate);
            if (duplex_session_id.empty() ||
                duplex_session_id != session_id) {
                return false;
            }
            socket = duplex_socket;
            stop_backend = duplex_backend_started;
            duplex_session_id.clear();
            duplex_socket = nullptr;
            duplex_backend_started = false;
        }
        if (stop_backend) (void)scheduler.control(mfq::engine::StopDuplex{});
        if (close_socket && socket != nullptr && socket->is_open()) {
            socket->close(
                httplib::ws::CloseStatus::Normal, "session closed");
        }
        return true;
    };
    server.set_payload_max_length(
        (scheduler.supports_multimodal_generation() ? 512ULL : 16ULL) *
        1024ULL * 1024ULL);
    server.set_read_timeout(300, 0);
    server.set_write_timeout(300, 0);
    server.set_keep_alive_max_count(100);
    server.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
        {"X-Content-Type-Options", "nosniff"},
    });

    server.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    if (duplex) {
        server.WebSocket("/runtime/realtime", [&](const httplib::Request & req,
                                          httplib::ws::WebSocket & ws) {
            const std::string expected = "Bearer " + config.api_key;
            if (!config.api_key.empty() &&
                req.get_header_value("Authorization") != expected) {
                ws.close(
                    httplib::ws::CloseStatus::PolicyViolation,
                    "authentication failed");
                return;
            }

            std::string owned_session;
            std::unordered_set<int64_t> session_controls;
            const auto send_event = [&](json event) {
                event["server_send_ts"] =
                    std::chrono::duration<double>(
                        std::chrono::system_clock::now()
                            .time_since_epoch()).count();
                return ws.send(event.dump());
            };
            try {
                std::string message;
                while (ws.is_open()) {
                    const auto read_result = ws.read(message);
                    if (read_result == httplib::ws::ReadResult::Fail) break;
                    if (read_result != httplib::ws::ReadResult::Text) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "duplex backend accepts JSON text frames only");
                    }
                    json body;
                    try {
                        body = json::parse(message);
                    } catch (const json::parse_error & error) {
                        throw ApiError(
                            400, "invalid_request_error",
                            std::string("invalid duplex JSON: ") +
                                error.what());
                    }
                    if (!body.is_object() || !body.contains("type") ||
                        !body["type"].is_string()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "duplex message requires a string type");
                    }
                    const std::string type = body["type"].get<std::string>();

                    if (type == "session.init") {
                        if (!owned_session.empty()) {
                            throw ApiError(
                                409, "conflict",
                                "duplex session is already initialized");
                        }
                        const json payload = body.value(
                            "payload", json::object());
                        if (!payload.is_object()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "session.init payload must be an object");
                        }
                        const std::string mode = payload.value(
                            "mode", std::string("full_duplex"));
                        if (mode != "full_duplex") {
                            throw ApiError(
                                400, "unsupported_operation",
                                duplex_backend_name +
                                    " duplex backend supports full_duplex only");
                        }

                        owned_session = request_id("sess-");
                        {
                            std::lock_guard<std::mutex> lock(duplex_gate);
                            if (!duplex_session_id.empty()) {
                                owned_session.clear();
                                throw ApiError(
                                    409, "conflict",
                                    "the " + duplex_backend_name +
                                        " worker already owns a duplex session");
                            }
                            duplex_session_id = owned_session;
                            duplex_socket = &ws;
                        }

                        MfqDuplexSessionParams parameters;
                        const std::string system_prompt = payload.value(
                            "system_prompt",
                            config.runtime_profile.duplex.system_prompt.value_or(
                                "Streaming Omni Conversation."));
                        scheduler.prepare_duplex_session(
                            system_prompt, parameters);
                        if (payload.contains("reference_audio_features")) {
                            if (!payload["reference_audio_features"].is_string()) {
                                throw ApiError(
                                    400, "invalid_request_error",
                                    "reference_audio_features must be base64 float32 Mel data",
                                    "reference_audio_features");
                            }
                            parameters.reference_audio_frames =
                                static_cast<int32_t>(integer_field(
                                    payload, "reference_audio_frames", 0));
                            parameters.reference_audio_features =
                                decode_audio_features(
                                    payload["reference_audio_features"].get<std::string>(),
                                    parameters.reference_audio_frames);
                        }
                        session_controls = std::unordered_set<int64_t>(
                            parameters.special_ids.begin(),
                            parameters.special_ids.end());
                        const json generation = payload.value(
                            "config", json::object());
                        if (!generation.is_object()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "session config must be an object", "config");
                        }
                        const auto & duplex_defaults = config.runtime_profile.duplex;
                        parameters.greedy = generation.value(
                            "decode_mode", duplex_defaults.decode_mode.value_or("sampling")) ==
                            "greedy";
                        parameters.temperature = number_field(
                            generation, "temperature", duplex_defaults.temperature.value_or(0.7));
                        parameters.top_k = static_cast<int32_t>(integer_field(
                            generation, "top_k", duplex_defaults.top_k.value_or(100)));
                        parameters.top_p = number_field(
                            generation, "top_p", duplex_defaults.top_p.value_or(0.8));
                        parameters.listen_probability_scale = number_field(
                            generation, "listen_prob_scale", duplex_defaults.listen_prob_scale.value_or(1.0));
                        parameters.repetition_penalty = number_field(
                            generation, "text_repetition_penalty", duplex_defaults.text_repetition_penalty.value_or(1.05));
                        parameters.repetition_window =
                            static_cast<int32_t>(integer_field(
                                generation,
                                "text_repetition_window_size",
                                duplex_defaults.text_repetition_window_size.value_or(512)));
                        parameters.length_penalty = number_field(
                            generation, "length_penalty", duplex_defaults.length_penalty.value_or(1.0));
                        parameters.tts_temperature = number_field(
                            generation, "tts_temperature",
                            config.runtime_profile.tts.temperature.value_or(0.8));
                        parameters.tts_repetition_penalty = number_field(
                            generation, "tts_repetition_penalty",
                            config.runtime_profile.tts.repetition_penalty.value_or(1.05));
                        if (generation.contains("seed")) {
                            parameters.seed = static_cast<uint64_t>(
                                integer_field(generation, "seed", 0));
                        } else {
                            std::random_device random;
                            parameters.seed =
                                (static_cast<uint64_t>(random()) << 32) ^
                                static_cast<uint64_t>(random());
                        }

                        try {
                            (void)scheduler.control(parameters);
                            std::lock_guard<std::mutex> lock(duplex_gate);
                            if (duplex_session_id == owned_session) {
                                duplex_backend_started = true;
                            }
                        } catch (...) {
                            std::lock_guard<std::mutex> lock(duplex_gate);
                            if (duplex_session_id == owned_session) {
                                duplex_session_id.clear();
                                duplex_socket = nullptr;
                            }
                            owned_session.clear();
                            throw;
                        }
                        send_event({
                            {"type", "session.created"},
                            {"session_id", owned_session},
                            {"mode", "full_duplex"},
                            {"metrics", {{"backend", duplex_backend_name}}},
                        });
                        continue;
                    }

                    if (type != "input.append") {
                        throw ApiError(
                            400, "invalid_request_error",
                            "unsupported duplex message type: " + type);
                    }
                    if (owned_session.empty()) {
                        throw ApiError(
                            409, "conflict",
                            "session.init must precede input.append");
                    }
                    const json input = body.value("input", json::object());
                    if (!input.is_object()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "input must be an object", "input");
                    }
                    const bool has_audio = input.contains("audio_features");
                    const bool has_text = input.contains("text");
                    if (!has_audio && !has_text) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "input requires audio_features or text", "input");
                    }
                    MfqDuplexStepInput step;
                    if (has_audio) {
                        if (!input["audio_features"].is_string()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "input.audio_features must be base64 float32 Mel data",
                                "audio_features");
                        }
                        step.audio_frames = static_cast<int32_t>(integer_field(
                            input, "audio_frames", 0));
                        step.audio_features = decode_audio_features(
                            input["audio_features"].get<std::string>(),
                            step.audio_frames);
                        step.audio_prefix_extra_frames = integer_field(
                            input, "audio_prefix_extra_frames", 0);
                        step.audio_suffix_extra_frames = integer_field(
                            input, "audio_suffix_extra_frames", 0);
                    }
                    if (has_text) {
                        if (!input["text"].is_string() ||
                            input["text"].get_ref<const std::string&>().empty()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "input.text must be a non-empty string", "text");
                        }
                        scheduler.prepare_duplex_step(
                            input["text"].get<std::string>(), step);
                    }
                    step.max_new_speak_tokens = static_cast<int32_t>(
                        integer_field(
                            input,
                            "max_new_speak_tokens",
                            config.runtime_profile.duplex
                                .max_new_speak_tokens_per_chunk
                                .value_or(20)));
                    if (input.contains("force_listen") &&
                        !input["force_listen"].is_boolean()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_listen must be boolean", "force_listen");
                    }
                    step.force_listen = input.value("force_listen", false);
                    if (input.contains("force_speak") &&
                        !input["force_speak"].is_boolean()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_speak must be boolean", "force_speak");
                    }
                    step.force_speak = input.value("force_speak", false);
                    if (step.force_listen && step.force_speak) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_listen and force_speak are mutually exclusive");
                    }

                    const auto result = std::get<MfqDuplexStepResult>(scheduler.control(step));
                    const std::string response_id = request_id("resp-");
                    json metrics = {
                        {"backend", duplex_backend_name},
                        {"wall_clock_ms", result.inference_ms},
                        {"kv_cache_length", result.language_cache_position},
                        {"audio_cache_length", result.audio_cache_position},
                        {"tts_cache_length", result.tts_cache_position},
                        {"audio_chunk_index", result.audio_chunk_index},
                    };

                    const std::string text_delta = scheduler.decode_tokens(
                        result.generated_tokens, session_controls);
                    if (!text_delta.empty()) {
                        send_event({
                            {"type", "response.output.delta"},
                            {"kind", "text"},
                            {"text", text_delta},
                            {"session_id", owned_session},
                            {"response_id", response_id},
                            {"end_of_turn", result.end_of_turn},
                            {"metrics", metrics},
                        });
                    }
                    if (!result.audio_tokens.empty() ||
                        (result.end_of_turn && !result.is_listen)) {
                        send_event({
                            {"type", "response.output.delta"},
                            {"kind", "audio_tokens"},
                            {"audio_tokens", result.audio_tokens},
                            {"session_id", owned_session},
                            {"response_id", response_id},
                            {"end_of_turn", result.end_of_turn},
                            {"force_flush", result.tts_force_flush},
                            {"metrics", metrics},
                        });
                    }
                    if (result.is_listen) {
                        send_event({
                            {"type", "response.output.delta"},
                            {"kind", "listen"},
                            {"session_id", owned_session},
                            {"response_id", response_id},
                            {"metrics", metrics},
                        });
                    }
                    send_event({
                        {"type", "response.step.done"},
                        {"session_id", owned_session},
                        {"response_id", response_id},
                        {"end_of_turn", result.end_of_turn},
                        {"metrics", metrics},
                    });
                }
            } catch (const mfq::engine::InferenceInputError & error) {
                send_event({
                    {"type", "session.closed"},
                    {"session_id", owned_session},
                    {"reason", "invalid_request"},
                    {"diagnostic", {{"message", error.what()}}},
                });
                if (ws.is_open()) {
                    ws.close(
                        httplib::ws::CloseStatus::PolicyViolation,
                        "invalid duplex input");
                }
            } catch (const std::exception & error) {
                mfq::log_runtime_error("transport", error.what());
                send_event({
                    {"type", "session.closed"},
                    {"session_id", owned_session},
                    {"reason", "backend_error"},
                    {"diagnostic", {{"message", error.what()}}},
                });
                if (ws.is_open()) {
                    ws.close(
                        httplib::ws::CloseStatus::InternalError,
                        "duplex backend error");
                }
            }
            if (!owned_session.empty()) {
                stop_duplex_session(owned_session, false);
            }
        });

        server.Post(R"(/runtime/realtime/sessions/([A-Za-z0-9_-]+)/close)",
            [&](const httplib::Request & req, httplib::Response & res) {
                if (!authorized(req, res, config.api_key)) return;
                const std::string session_id = req.matches[1].str();
                if (!stop_duplex_session(session_id, true)) {
                    set_json(res, error_body(
                        "duplex session was not found", "not_found"), 404);
                    return;
                }
                set_json(res, {
                    {"ok", true},
                    {"session_id", session_id},
                    {"closed", true},
                });
            });
    }

    server.Get("/", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        set_json(res, {
            {"name", "MFQ C++ HTTP runtime transport"},
            {"model", config.model_name},
            {"endpoints", {
                "/runtime/generate", "/runtime/score", "/runtime/models",
                "/runtime/health", "/runtime/status", "/runtime/reload",
                "/runtime/realtime", "/runtime/cache/clear",
                "/runtime/cache/trim", "/runtime/cache/budget", "/runtime/cache/refresh", "/runtime/memory/budget", "/runtime/sessions/fork",
                "/runtime/sessions/{id}",
                "/runtime/sessions/{id}/cancel",
            }},
        });
    });

    const auto add_runtime_metrics = [&](json & value) {

        const auto snapshot = scheduler.snapshot();
        value["healthy"] = snapshot.status.healthy;
        const auto& metrics = snapshot.runtime;
        value["runtime_metrics_age_ms"] = std::chrono::duration<double, std::milli>(
            mfq::engine::Clock::now() - snapshot.captured).count();

        for (const auto& item : metrics) {
            value[item.first] = item.second;
        }
    };
    const auto add_session_metrics = [&](json & value) {

        for (const auto & item : scheduler.snapshot().session) {
            value[item.first] = item.second;
        }
    };

    server.Get("/runtime/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {
            {"status", reloading.load() ? "loading" : (scheduler.snapshot().status.healthy ? "ok" : "unhealthy")},
            {"model", config.model_name},
            {"model_type", config.model_type},
            {"probability_available", scheduler.info().probability},
            {"model_capabilities", model_capabilities},
            {"vision_supported", vision_supported},
            {"vision_available", vision_available},
            {"video_available", video_available},
            {"mtp_supported", model_capability_profile.mtp},
            {"mtp_available", false},
            {"vision_enabled_default", sampling_defaults.enable_vision},
            {"mtp_enabled_default", sampling_defaults.enable_mtp},
            {"max_context", active_context.load()},
            {"batch_capacity", scheduler.info().batch_capacity},
            {"duplex_available", static_cast<bool>(duplex)},
            {"duplex_active", duplex_is_active()},
            {"sampling_defaults", sampling_params_json(sampling_defaults)},
            {"duplex_sampling_defaults", duplex_sampling_defaults},
            {"tts_sampling_defaults", tts_sampling_defaults},
            {"runtime_profile_source", config.runtime_profile.source},
            {"chat_template_capabilities", chat_template_capabilities},
        };
        add_runtime_metrics(health);
        add_session_metrics(health);
        set_json(res, health, scheduler.snapshot().status.healthy ? 200 : 503);
    });

    server.Get("/runtime/realtime/capabilities", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        set_json(res, {
            {"available", static_cast<bool>(duplex)},
            {"modes", duplex ? json::array({"audio"}) : json::array()},
        });
    });

    server.Get("/runtime/status", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        json status = request_metrics_store.snapshot(
            config, active_context.load(), reloading.load());
        status["sampling_defaults"] = sampling_params_json(
            sampling_defaults);
        status["duplex_sampling_defaults"] = duplex_sampling_defaults;
        status["tts_sampling_defaults"] = tts_sampling_defaults;
        status["runtime_profile_source"] = config.runtime_profile.source;
        status["chat_template_capabilities"] =
            chat_template_capabilities;
        status["model_capabilities"] = model_capabilities;
        status["vision_supported"] = vision_supported;
        status["vision_available"] = vision_available;
        status["video_available"] = video_available;
        status["mtp_supported"] = model_capability_profile.mtp;
        status["mtp_available"] = false;
        status["vision_enabled_default"] = sampling_defaults.enable_vision;
        status["mtp_enabled_default"] = sampling_defaults.enable_mtp;
        status["batch_capacity"] = scheduler.info().batch_capacity;
        status["duplex_available"] = static_cast<bool>(duplex);
        status["duplex_active"] = duplex_is_active();
        add_runtime_metrics(status);
        add_session_metrics(status);
        set_json(res, status);
    });

    server.Post("/runtime/cache/clear", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;


        try {
            const size_t released = scheduler.session({mfq::engine::SessionCommand::Kind::clear}).count;
            json result = {
                {"status", "ok"},
                {"released_snapshots", released},
            };
            add_session_metrics(result);
            reloading.store(false);
            set_json(res, result);
        } catch (const std::exception & error) {
            mfq::log_runtime_error("transport", error.what());
            reloading.store(false);
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Post("/runtime/cache/refresh", [&](const httplib::Request&, httplib::Response& res) {
        try {
            json result = {{"status", "ok"},
                {"removed_blocks", scheduler.session({mfq::engine::SessionCommand::Kind::refresh}).count}};
            add_session_metrics(result);
            set_json(res, result);
        } catch (const std::exception& error) {
            mfq::log_runtime_error("transport", error.what());
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });
    server.Post("/runtime/memory/budget", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req, res, config.api_key)) return;
        try {
            const auto body = parse_body(req);
            const auto bytes = integer_field(body, "target_bytes", 0);
            if (bytes < 0) throw ApiError(400, "invalid_request_error", "target_bytes must be non-negative");
            mfq::engine::SessionCommand command{mfq::engine::SessionCommand::Kind::memory_budget, {}, {}, static_cast<uint64_t>(bytes)};
            if (body.contains("weight_target_bytes")) {
                const auto weight = integer_field(body, "weight_target_bytes", 0);
                if (weight < 0) throw ApiError(400, "invalid_request_error", "weight_target_bytes must be non-negative");
                command.weight_bytes = static_cast<uint64_t>(weight);
            }
            json result = {{"status", "ok"}, {"released_bytes", scheduler.session(command).count}};
            add_session_metrics(result);
            set_json(res, result);
        } catch (const ApiError& error) { handle_api_error(res, error); }
        catch (const MfqResourceExhausted& error) { set_json(res, error_body(error.what(), "resident_budget_exhausted"), 413); }
        catch (const std::exception& error) {
            mfq::log_runtime_error("transport", error.what());
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Post(R"(/runtime/cache/(trim|budget))", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;

        try {
            const json body = parse_body(req);
            if (!body.is_object()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "request body must be a JSON object");
            }
            std::uint64_t target_bytes = 0;
            if (body.contains("target_bytes")) {
                if (!body["target_bytes"].is_number_unsigned() &&
                    !(body["target_bytes"].is_number_integer() &&
                      body["target_bytes"].get<std::int64_t>() >= 0)) {
                    throw ApiError(
                        400, "invalid_request_error",
                        "target_bytes must be a non-negative integer",
                        "target_bytes");
                }
                target_bytes = body["target_bytes"].get<std::uint64_t>();
            }
            const auto kind = req.path == "/runtime/cache/budget"
                ? mfq::engine::SessionCommand::Kind::budget : mfq::engine::SessionCommand::Kind::trim;
            mfq::engine::SessionCommand command{kind, {}, {}, target_bytes};
            if (kind == mfq::engine::SessionCommand::Kind::budget && body.contains("disk_target_bytes")) {
                const auto disk = integer_field(body, "disk_target_bytes", 0);
                if (disk < 0) throw ApiError(400, "invalid_request_error", "disk_target_bytes must be non-negative");
                command.disk_bytes = static_cast<uint64_t>(disk);
            }
            if (kind == mfq::engine::SessionCommand::Kind::budget && body.contains("ram_target_bytes")) {
                const auto ram = integer_field(body, "ram_target_bytes", 0);
                if (ram < 0) throw ApiError(400, "invalid_request_error", "ram_target_bytes must be non-negative");
                command.ram_bytes = static_cast<uint64_t>(ram);
            }
            const auto released = scheduler.session(command).count;
            json result = {
                {"status", "ok"},
                {"released_bytes", released},
                {"target_bytes", target_bytes},
            };
            add_session_metrics(result);
            set_json(res, result);
        } catch (const ApiError & error) {
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            mfq::log_runtime_error("transport", error.what());
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Post(
        R"(/runtime/sessions/([A-Za-z0-9._:-]{1,128})/cancel)",
        [&] (const httplib::Request & req, httplib::Response & res) {
            if (!authorized(req, res, config.api_key)) return;
            const std::string session_id = req.matches[1].str();
            set_json(res, {
                {"status", "ok"},
                {"cancelled", scheduler.cancel_session(session_id)},
            });
        });

    server.Post("/runtime/sessions/fork", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;

        try {
            const json body = parse_body(req);
            const auto read_session_id = [&](const char * field) {
                if (!body.contains(field) || !body[field].is_string()) {
                    throw ApiError(
                        400, "invalid_request_error",
                        std::string(field) + " must be a string", field);
                }
                auto session_id = body[field].get<std::string>();
                if (!valid_mfq_session_id(session_id)) {
                    throw ApiError(
                        400, "invalid_request_error",
                        std::string(field) +
                            " must contain 1 to 128 safe identifier bytes",
                        field);
                }
                return session_id;
            };
            const std::string source_session_id =
                read_session_id("source_session_id");
            const std::string target_session_id =
                read_session_id("target_session_id");
            if (source_session_id == target_session_id) {
                throw ApiError(
                    400, "invalid_request_error",
                    "source and target sessions must differ",
                    "target_session_id");
            }
            const size_t copied = scheduler.session({mfq::engine::SessionCommand::Kind::fork, source_session_id, target_session_id}).count;
            set_json(res, {
                {"status", "ok"},
                {"copied_snapshots", copied},
            });
        } catch (const ApiError & error) {
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            mfq::log_runtime_error("transport", error.what());
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Delete(
        R"(/runtime/sessions/([A-Za-z0-9._:-]{1,128}))",
        [&] (const httplib::Request & req, httplib::Response & res) {
            if (!authorized(req, res, config.api_key)) return;

            try {
                const std::string session_id = req.matches[1].str();
                const size_t released = scheduler.session({mfq::engine::SessionCommand::Kind::close, session_id}).count;
                set_json(res, {
                    {"status", "ok"},
                    {"released_snapshots", released},
                });
            } catch (const std::exception & error) {
                mfq::log_runtime_error("transport", error.what());
                set_json(res, error_body(error.what(), "server_error"), 500);
            }
        });

    server.Post("/runtime/reload", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (!scheduler.supports_reload()) {
            set_json(res, error_body(
                "this runtime does not support model reload",
                "unsupported_operation"), 501);
            return;
        }

        const auto finish_reload = [&] {
            reloading.store(false);
        };
        try {
            const json body = parse_body(req);
            const int64_t context_size = integer_field(
                body, "context_size", active_context.load());
            const int64_t capacity = config.context_capacity > 0
                ? config.context_capacity
                : config.max_context;
            if (context_size < 1 ||
                (capacity > 0 && context_size > capacity)) {
                throw ApiError(
                    400, "invalid_request_error",
                    "context_size must be within the model context capacity",
                    "context_size");
            }
            reloading.store(true);
            const int64_t loaded_context = scheduler.reload(context_size);
            if (loaded_context < 1 ||
                (capacity > 0 && loaded_context > capacity)) {
                throw std::runtime_error(
                    "runtime reload returned an invalid context size");
            }
            active_context.store(loaded_context);
            finish_reload();
            set_json(res, {
                {"status", "ok"},
                {"model", config.model_name},
                {"max_context", loaded_context},
                {"context_capacity", capacity},
            });
        } catch (const ApiError & error) {
            finish_reload();
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            mfq::log_runtime_error("transport", error.what());
            finish_reload();
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Get("/runtime/models", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        set_json(res, {
            {"models", json::array({{
                {"name", config.model_name},
                {"type", config.model_type},
                {"capabilities", model_capabilities},
            }})},
        });
    });

    server.Post("/runtime/count_tokens", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req, res, config.api_key)) return;
        try {
            set_json(res, token_count_json(std::get<mfq::engine::TokenCount>(
                scheduler.control(parse_token_count_request(parse_body(req))))));
        } catch (const ApiError& error) { handle_api_error(res, error); }
        catch (const std::invalid_argument& error) { set_json(res, error_body(error.what(), "invalid_request_error"), 400); }
        catch (const std::exception& error) {
            mfq::log_runtime_error("transport", error.what());
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Post("/runtime/score", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req, res, config.api_key)) return;
        try {
            if (!scheduler.info().probability)
                throw ApiError(501, "probability_unavailable", "probability scoring is unavailable for this model");
            set_json(res, likelihood_result_json(std::get<mfq::engine::LikelihoodResult>(
                scheduler.control(parse_score_request(parse_body(req))))));
        } catch (const ApiError& error) { handle_api_error(res, error); }
        catch (const std::invalid_argument& error) { set_json(res, error_body(error.what(), "invalid_request_error"), 400); }
        catch (const std::exception& error) {
            mfq::log_runtime_error("transport", error.what());
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    auto runtime_generate_handler = [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        try {
            const json body = runtime_generate_body(parse_body(req));
            auto input = parse_input(body, true, sampling_defaults);
            const RequestSchedule schedule = parse_schedule(body);
            if (body.contains("mfq_multimodal")) {
                if (!input.sampling.enable_vision) {
                    throw ApiError(
                        400, "invalid_request_error",
                        "vision is disabled for this request; set enable_vision=true",
                        "enable_vision");
                }
                if (!scheduler.supports_multimodal_generation()) {
                    throw ApiError(
                        501, "unsupported_parameter",
                        "the loaded model has no native vision runtime",
                        "mfq_multimodal");
                }
                input.media = parse_mfq_vision(body["mfq_multimodal"]);
            }
            RequestWork work = std::move(input);
            const std::string id = request_id("run-");
            const int64_t created = unix_time_seconds();
            std::shared_ptr<ActiveRequest> active_request;
            active_request = std::make_shared<ActiveRequest>(request_metrics_store);
            auto completion = std::make_shared<CompletionStream>(scheduler, work, id, schedule);

            if (!work.stream) {
                auto next_progress = std::chrono::steady_clock::time_point{};
                while (completion->next()) {
                    const auto now = std::chrono::steady_clock::now();
                    if (now >= next_progress) {
                        active_request->progress(id, completion->progress());
                        next_progress = now + std::chrono::milliseconds(250);
                    }
                }
                auto result = std::move(completion->result);
                auto metrics = completion->metrics;
                const RequestMetricValues metric_values =
                    request_metric_values(result, metrics);
                log_request_metrics(
                    id, true, false, completion->prompt_tokens, work.sampling,
                    result, metric_values);
                active_request->complete(
                    id, true, false, completion->prompt_tokens, result, metric_values);
                auto performance =
                    request_metric_values_json(metric_values, work.sampling);
                add_request_runtime_metrics(performance, metrics);
                set_json(res, runtime_generation_result(
                    id, created, config.model_name, result,
                    usage_json(completion->prompt_tokens, result.completion_tokens),
                    std::move(performance)));
                return;
            }

            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider(
                "text/event-stream; charset=utf-8",
                [work = std::move(work), id, created, &scheduler,
                 &config, active_request, completion]
                (size_t offset, httplib::DataSink & sink) mutable -> bool {
                    if (offset != 0) {
                        sink.done();
                        return false;
                    }
                    try {
                        auto next_progress = std::chrono::steady_clock::time_point{};
                        while (auto diffs = completion->next()) {
                            const auto now = std::chrono::steady_clock::now();
                            if (now >= next_progress) {
                                active_request->progress(id, completion->progress());
                                next_progress = now + std::chrono::milliseconds(250);
                            }
                            for (const auto& diff : *diffs) {
                                json delta = chat_diff_json(diff);
                                if (delta.empty()) continue;
                                auto event = runtime_generation_event("delta", id, created, config.model_name);
                                event["delta"] = std::move(delta);
                                if (!write_sse(sink, event)) { completion->cancel(); return false; }
                            }
                        }
                        auto result = std::move(completion->result);
                        auto metrics = completion->metrics;
                        const RequestMetricValues metric_values =
                            request_metric_values(result, metrics);
                        log_request_metrics(
                            id, true, true, completion->prompt_tokens, work.sampling,
                            result, metric_values);
                        active_request->complete(
                            id, true, true, completion->prompt_tokens, result,
                            metric_values);
                        if (!result.client_connected) return false;
                        auto complete = runtime_generation_event(
                            "complete", id, created, config.model_name);
                        complete["finish_reason"] = result.finish_reason;
                        auto performance = request_metric_values_json(
                            metric_values, work.sampling);
                        add_request_runtime_metrics(performance, metrics);
                        complete["metrics"] = std::move(performance);
                        if (!write_sse(sink, complete)) return false;
                        if (work.include_usage) {
                            auto usage = runtime_generation_event(
                                "usage", id, created, config.model_name);
                            usage["usage"] = usage_json(
                                completion->prompt_tokens, result.completion_tokens);
                            if (!write_sse(sink, usage)) return false;
                        }
                        static constexpr char done[] = "data: [DONE]\n\n";
                        if (!sink.write(done, sizeof(done) - 1)) return false;
                    } catch (const ApiError & error) {
                        write_sse(sink, stream_error_body(error));
                    } catch (const std::exception & error) {
                        mfq::log_runtime_error("transport", error.what());
                        write_sse(sink, error_body(error.what(), "runtime_error"));
                    }
                    sink.done();
                    return false;
                });
        } catch (const ApiError & error) {
            handle_api_error(res, error);
        } catch (const mfq::engine::InferenceInputError & error) {
            handle_api_error(res, api_error(error));
        } catch (const std::exception & error) {
            mfq::log_runtime_error("transport", error.what());
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    };

    server.Post("/runtime/generate", runtime_generate_handler);

    server.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message = "unhandled server exception";
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception & error) {
            mfq::log_runtime_error("transport", error.what());
            message = error.what();
        }
        set_json(res, error_body(message, "server_error"), 500);
    });

    if (!server.bind_to_port(config.host, config.port)) {
        throw std::runtime_error(
            "failed to bind " + config.host + ":" +
            std::to_string(config.port));
    }
    const std::string endpoint = "http://" + config.host + ":" +
        std::to_string(config.port);
    std::cout << "MFQ HTTP runtime transport ready: " << endpoint
              << " model=" << config.model_name
              << " context=" << config.max_context
              << " vocab=" << scheduler.vocab_size() << std::endl;
    if (!server.listen_after_bind()) {
        throw std::runtime_error(
            "runtime transport stopped after binding " + config.host + ":" +
            std::to_string(config.port));
    }
    return 0;
}

class MfqHttpTransport final : public MfqTransport {
public:
    explicit MfqHttpTransport(MfqHttpRuntimeTransportConfig config)
        : config_(std::move(config)) {}

    int run(const MfqScheduler & scheduler) override {
        return run_mfq_http_transport(config_, scheduler);
    }

private:
    MfqHttpRuntimeTransportConfig config_;
};

} // namespace

std::unique_ptr<MfqTransport> make_mfq_http_transport(
        MfqHttpRuntimeTransportConfig config) {
    return std::make_unique<MfqHttpTransport>(std::move(config));
}
