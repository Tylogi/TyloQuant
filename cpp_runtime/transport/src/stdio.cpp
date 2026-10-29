#include "mfq/fatal_diagnostics.h"
#include "common.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

using namespace mfq::transport_detail;

std::mutex stdio_protocol_mutex;
int stdio_protocol_fd = -1;

static const char * input_error_type(
        mfq::engine::InferenceInputErrorCode code) {
    if (code == mfq::engine::InferenceInputErrorCode::Unsupported) {
        return "unsupported_parameter";
    }
    if (code == mfq::engine::InferenceInputErrorCode::ContextLength) {
        return "context_length_exceeded";
    }
    return "invalid_request_error";
}

static bool write_stdio_protocol(const json & frame) {
    const std::string line = frame.dump() + "\n";
    std::lock_guard<std::mutex> lock(stdio_protocol_mutex);
    if (stdio_protocol_fd < 0) return false;
    size_t offset = 0;
    while (offset < line.size()) {
#ifdef _WIN32
        const auto remaining = std::min<size_t>(
            line.size() - offset,
            static_cast<size_t>(std::numeric_limits<int>::max()));
        const int written = _write(
            stdio_protocol_fd, line.data() + offset,
            static_cast<unsigned int>(remaining));
#else
        const ssize_t written = ::write(
            stdio_protocol_fd, line.data() + offset,
            line.size() - offset);
#endif
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        offset += static_cast<size_t>(written);
    }
    return true;
}

class MfqStdioTransport final : public MfqTransport {
public:
    explicit MfqStdioTransport(MfqRuntimeTransportConfig config)
        : config_(std::move(config)) {}

    int run(const MfqScheduler & scheduler) override {
        const bool duplex = scheduler.info().duplex;
        if (stdio_protocol_fd < 0) {
            throw std::runtime_error(
                "prepare_mfq_stdio_transport must be called before model loading");
        }
        const MfqSamplingParams sampling_defaults =
            default_sampling_params(config_);
        const json duplex_sampling_defaults =
            duplex_profile_json(config_.runtime_profile.duplex);
        const json tts_sampling_defaults =
            tts_profile_json(config_.runtime_profile.tts);
        const json chat_template_capabilities =
            chat_template_capabilities_json(
                scheduler.chat_template_capabilities());
        const auto capability_profile = config_.model_capabilities
            ? *config_.model_capabilities
            : architecture_capability_profile(config_.model_type);
        const json model_capabilities =
            model_capability_profile_json(capability_profile);
        const bool vision_supported = capability_profile.image_input ||
            capability_profile.video_input;
        const bool vision_available =
            scheduler.supports_multimodal_generation() &&
            capability_profile.image_input;
        const bool video_available =
            scheduler.supports_multimodal_generation() &&
            capability_profile.video_input;

        RuntimeRequestMetrics request_metrics_store;
        std::atomic<int64_t> active_context{config_.max_context};
        std::atomic<bool> reloading{false};
        struct RealtimeState {
            std::string channel_id;
            std::string session_id;
            std::unordered_set<int64_t> control_tokens;
            bool backend_started = false;
        };
        std::mutex realtime_gate;
        RealtimeState realtime;
        const auto duplex_active = [&] {
            std::lock_guard<std::mutex> lock(realtime_gate);
            return !realtime.session_id.empty();
        };
        struct Task {
            std::thread thread;
            std::shared_ptr<std::atomic<bool>> done;
            std::shared_ptr<std::atomic<bool>> submitted;
            std::string id;
        };
        std::vector<Task> tasks;

        const auto send = [](json frame) {
            frame["v"] = 1;
            return write_stdio_protocol(frame);
        };
        const auto send_result = [&](const std::string & id, json data) {
            return send({
                {"id", id},
                {"type", "result"},
                {"data", std::move(data)},
            });
        };
        const auto send_event = [&](const std::string & id, json data) {
            return send({
                {"id", id},
                {"type", "event"},
                {"data", std::move(data)},
            });
        };
        const auto send_done = [&](const std::string & id) {
            return send({{"id", id}, {"type", "done"}});
        };
        const auto send_error = [&](
                const std::string & id, int status,
                const std::string & code, const std::string & message) {
            // An elapsed request deadline cannot be fixed by retrying the same
            // absolute bound, so it stays non-retryable despite HTTP 504.
            const bool retryable = (status == 429 || status == 502 ||
                status == 503 || status == 504) && code != "deadline_exceeded";
            return send({
                {"id", id},
                {"type", "error"},
                {"error", {
                    {"code", code},
                    {"message", message},
                    {"status_code", status},
                    {"retryable", retryable},
                }},
            });
        };
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
        const auto health = [&] {
            json value = {
                {"status", reloading.load() ? "loading" : (scheduler.snapshot().status.healthy ? "ok" : "unhealthy")},
                {"model", config_.model_name},
                {"model_type", config_.model_type},
                {"probability_available", scheduler.info().probability},
                {"model_capabilities", model_capabilities},
                {"vision_supported", vision_supported},
                {"vision_available", vision_available},
                {"video_available", video_available},
                {"mtp_supported", capability_profile.mtp},
                {"mtp_available", false},
                {"vision_enabled_default", sampling_defaults.enable_vision},
                {"mtp_enabled_default", sampling_defaults.enable_mtp},
                {"max_context", active_context.load()},
                {"batch_capacity", scheduler.info().batch_capacity},
                {"duplex_available", static_cast<bool>(duplex)},
                {"duplex_active", duplex_active()},
                {"sampling_defaults", sampling_params_json(sampling_defaults)},
                {"duplex_sampling_defaults", duplex_sampling_defaults},
                {"tts_sampling_defaults", tts_sampling_defaults},
                {"runtime_profile_source", config_.runtime_profile.source},
                {"chat_template_capabilities", chat_template_capabilities},
            };
            add_runtime_metrics(value);
            add_session_metrics(value);
            return value;
        };
        const auto status = [&] {
            json value = request_metrics_store.snapshot(
                config_, active_context.load(), reloading.load());
            value["sampling_defaults"] =
                sampling_params_json(sampling_defaults);
            value["duplex_sampling_defaults"] = duplex_sampling_defaults;
            value["tts_sampling_defaults"] = tts_sampling_defaults;
            value["runtime_profile_source"] = config_.runtime_profile.source;
            value["chat_template_capabilities"] =
                chat_template_capabilities;
            value["model_capabilities"] = model_capabilities;
            value["vision_supported"] = vision_supported;
            value["vision_available"] = vision_available;
            value["video_available"] = video_available;
            value["mtp_supported"] = capability_profile.mtp;
            value["mtp_available"] = false;
            value["vision_enabled_default"] =
                sampling_defaults.enable_vision;
            value["mtp_enabled_default"] = sampling_defaults.enable_mtp;
            value["batch_capacity"] = scheduler.info().batch_capacity;
            value["duplex_available"] = static_cast<bool>(duplex);
            value["duplex_active"] = duplex_active();
            add_runtime_metrics(value);
            add_session_metrics(value);
            return value;
        };
        const auto reap_tasks = [&](bool all) {
            for (auto item = tasks.begin(); item != tasks.end();) {
                if (all || item->done->load(std::memory_order_acquire)) {
                    if (item->thread.joinable()) item->thread.join();
                    item = tasks.erase(item);
                } else {
                    ++item;
                }
            }
        };
        const auto close_realtime = [&](const std::string & channel_id) {
            bool stop_backend = false;
            {
                std::lock_guard<std::mutex> lock(realtime_gate);
                if (realtime.channel_id.empty() ||
                    (!channel_id.empty() &&
                     realtime.channel_id != channel_id)) {
                    return false;
                }
                stop_backend = realtime.backend_started;
                realtime = {};
            }
            if (stop_backend) (void)scheduler.control(mfq::engine::StopDuplex{});
            return true;
        };
        const auto emit_realtime = [&](json event) {
            std::string channel_id;
            {
                std::lock_guard<std::mutex> lock(realtime_gate);
                channel_id = realtime.channel_id;
            }
            if (channel_id.empty()) return false;
            event["server_send_ts"] =
                std::chrono::duration<double>(
                    std::chrono::system_clock::now()
                        .time_since_epoch()).count();
            return send_event(channel_id, event.dump());
        };

        send({{"type", "ready"}});
        bool running = true;
        std::string line;
        while (running && std::getline(std::cin, line)) {
            reap_tasks(false);
            std::string id;
            try {
                if (line.size() > 64ULL * 1024ULL * 1024ULL) {
                    throw ApiError(
                        413, "request_too_large",
                        "stdio frame exceeds 64 MiB");
                }
                const json request = json::parse(line);
                if (!request.is_object() || request.value("v", 0) != 1) {
                    throw ApiError(
                        400, "backend_protocol_error",
                        "stdio request requires protocol version 1");
                }
                if (request.contains("id")) {
                    if (!request["id"].is_string()) {
                        throw ApiError(
                            400, "backend_protocol_error",
                            "stdio request id must be a string");
                    }
                    id = request["id"].get<std::string>();
                }
                if (!request.contains("op") || !request["op"].is_string()) {
                    throw ApiError(
                        400, "backend_protocol_error",
                        "stdio request requires a string op");
                }
                const std::string op = request["op"].get<std::string>();
                const json params = request.value("params", json::object());
                if (!params.is_object()) {
                    throw ApiError(
                        400, "backend_protocol_error",
                        "stdio params must be an object");
                }

                if (op == "shutdown") {
                    if (!id.empty()) send_result(id, {{"status", "ok"}});
                    running = false;
                    continue;
                }
                if (op == "request.cancel") {
                    const std::string target = params.value(
                        "target_id", std::string());
                    for(const auto& task:tasks) if(task.id==target)
                        task.submitted->wait(false,std::memory_order_acquire);
                    const bool cancelled = scheduler.cancel_request(target);
                    if (!id.empty()) {
                        send_result(id, {
                            {"status", "ok"},
                            {"cancelled", cancelled},
                        });
                    }
                    continue;
                }
                if (id.empty()) {
                    throw ApiError(
                        400, "backend_protocol_error",
                        "stdio operation requires an id");
                }
                if (op == "health") {
                    send_result(id, health());
                    continue;
                }
                if (op == "status") {
                    send_result(id, status());
                    continue;
                }
                if (op == "models") {
                    send_result(id, {
                        {"models", json::array({{
                            {"name", config_.model_name},
                            {"type", config_.model_type},
                            {"capabilities", model_capabilities},
                        }})},
                    });
                    continue;
                }
                if (op == "realtime.capabilities") {
                    send_result(id, {
                        {"available", static_cast<bool>(duplex)},
                        {"modes", duplex ? json::array({"audio"}) : json::array()},
                    });
                    continue;
                }
                if (op == "session.cancel") {
                    const std::string session_id = params.value(
                        "session_id", std::string());
                    const bool cancelled =
                        scheduler.cancel_session(session_id);
                    send_result(id, {
                        {"status", "ok"},
                        {"cancelled", cancelled},
                    });
                    continue;
                }
                if (op == "session.fork") {

                    const std::string source = params.value(
                        "source_session_id", std::string());
                    const std::string target = params.value(
                        "target_session_id", std::string());
                    if (!valid_mfq_session_id(source) ||
                        !valid_mfq_session_id(target) || source == target) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "source and target session IDs must be distinct safe identifiers");
                    }
                    send_result(id, {
                        {"status", "ok"},
                        {"copied_snapshots", scheduler.session({mfq::engine::SessionCommand::Kind::fork, source, target}).count},
                    });
                    continue;
                }
                if (op == "session.close") {

                    const std::string session_id = params.value(
                        "session_id", std::string());
                    if (!valid_mfq_session_id(session_id)) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "session_id must be a safe identifier");
                    }
                    send_result(id, {
                        {"status", "ok"},
                        {"released_snapshots", scheduler.session({mfq::engine::SessionCommand::Kind::close, session_id}).count},
                    });
                    continue;
                }
                if (op == "cache.clear") {


                    try {
                        json result = {
                            {"status", "ok"},
                            {"released_snapshots", scheduler.session({mfq::engine::SessionCommand::Kind::clear}).count},
                        };
                        add_session_metrics(result);
                        reloading.store(false);
                        send_result(id, std::move(result));
                    } catch (...) {
                        reloading.store(false);
                        throw;
                    }
                    continue;
                }
                if (op == "memory.budget") {
                    const auto bytes = integer_field(params, "target_bytes", 0);
                    if (bytes < 0) throw ApiError(400, "invalid_request_error", "target_bytes must be non-negative");
                    mfq::engine::SessionCommand command{mfq::engine::SessionCommand::Kind::memory_budget, {}, {}, static_cast<uint64_t>(bytes)};
                    if (params.contains("weight_target_bytes")) {
                        const auto weight = integer_field(params, "weight_target_bytes", 0);
                        if (weight < 0) throw ApiError(400, "invalid_request_error", "weight_target_bytes must be non-negative");
                        command.weight_bytes = static_cast<uint64_t>(weight);
                    }
                    auto result = json{{"status", "ok"}, {"released_bytes", scheduler.session(command).count}};
                    add_session_metrics(result);
                    send_result(id, std::move(result));
                    continue;
                }
                if (op == "cache.refresh") {
                    json result = {{"status", "ok"},
                        {"removed_blocks", scheduler.session({mfq::engine::SessionCommand::Kind::refresh}).count}};
                    add_session_metrics(result);
                    send_result(id, std::move(result));
                    continue;
                }
                if (op == "cache.trim" || op == "cache.budget") {

                    const int64_t target = integer_field(
                        params, "target_bytes", 0);
                    if (target < 0) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "target_bytes must be non-negative");
                    }
                    mfq::engine::SessionCommand command{op == "cache.trim" ? mfq::engine::SessionCommand::Kind::trim : mfq::engine::SessionCommand::Kind::budget, {}, {}, static_cast<uint64_t>(target)};
                    if (op == "cache.budget" && params.contains("disk_target_bytes")) {
                        const auto disk = integer_field(params, "disk_target_bytes", 0);
                        if (disk < 0) throw ApiError(400, "invalid_request_error", "disk_target_bytes must be non-negative");
                        command.disk_bytes = static_cast<uint64_t>(disk);
                    }
                    if (op == "cache.budget" && params.contains("ram_target_bytes")) {
                        const auto ram = integer_field(params, "ram_target_bytes", 0);
                        if (ram < 0) throw ApiError(400, "invalid_request_error", "ram_target_bytes must be non-negative");
                        command.ram_bytes = static_cast<uint64_t>(ram);
                    }
                    json result = {
                        {"status", "ok"},
                        {"released_bytes", scheduler.session(command).count},
                        {"target_bytes", target},
                    };
                    add_session_metrics(result);
                    send_result(id, std::move(result));
                    continue;
                }
                if (op == "reload") {
                    if (!scheduler.supports_reload()) {
                        throw ApiError(
                            501, "unsupported_operation",
                            "this runtime does not support model reload");
                    }

                    try {
                        const int64_t requested = integer_field(
                            params, "context_size", active_context.load());
                        const int64_t capacity = config_.context_capacity > 0
                            ? config_.context_capacity
                            : config_.max_context;
                        if (requested < 1 ||
                            (capacity > 0 && requested > capacity)) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "context_size exceeds model capacity");
                        }
                        reloading.store(true);
                        const int64_t loaded = scheduler.reload(requested);
                        if (loaded < 1 ||
                            (capacity > 0 && loaded > capacity)) {
                            throw std::runtime_error(
                                "runtime reload returned an invalid context size");
                        }
                        active_context.store(loaded);
                        reloading.store(false);
                        send_result(id, {
                            {"status", "ok"},
                            {"model", config_.model_name},
                            {"max_context", loaded},
                            {"context_capacity", capacity},
                        });
                    } catch (...) {
                        reloading.store(false);
                        throw;
                    }
                    continue;
                }
                if (op == "realtime.open") {
                    if (!duplex) {
                        throw ApiError(
                            501, "unsupported_operation",
                            "this runtime has no realtime backend");
                    }
                    if (params.value("mode", std::string("audio")) != "audio") {
                        throw ApiError(
                            400, "unsupported_operation",
                            "stdio realtime supports audio mode only");
                    }
                    {
                        std::lock_guard<std::mutex> lock(realtime_gate);
                        if (!realtime.channel_id.empty()) {
                            throw ApiError(
                                409, "conflict",
                                "a realtime channel is already open");
                        }
                        realtime.channel_id = id;
                    }
                    send_result(id, {{"status", "ok"}});
                    continue;
                }
                if (op == "realtime.close") {
                    const std::string target = params.value(
                        "target_id", std::string());
                    if (!close_realtime(target)) {
                        throw ApiError(
                            404, "not_found",
                            "realtime channel was not found");
                    }
                    send_done(target);
                    send_result(id, {{"status", "ok"}});
                    continue;
                }
                if (op == "realtime.send") {
                    const std::string target = params.value(
                        "target_id", std::string());
                    const std::string encoded = params.value(
                        "data", std::string());
                    {
                        std::lock_guard<std::mutex> lock(realtime_gate);
                        if (target.empty() || target != realtime.channel_id) {
                            throw ApiError(
                                404, "not_found",
                                "realtime channel was not found");
                        }
                    }
                    json body;
                    try {
                        body = json::parse(encoded);
                    } catch (const json::parse_error & error) {
                        throw ApiError(
                            400, "invalid_request_error",
                            std::string("invalid realtime JSON: ") +
                                error.what());
                    }
                    if (!body.is_object() || !body.contains("type") ||
                        !body["type"].is_string()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "realtime message requires a string type");
                    }
                    const std::string type = body["type"].get<std::string>();
                    if (type == "session.init") {
                        {
                            std::lock_guard<std::mutex> lock(realtime_gate);
                            if (!realtime.session_id.empty()) {
                                throw ApiError(
                                    409, "conflict",
                                    "realtime session is already initialized");
                            }
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
                                "realtime backend supports full_duplex only");
                        }
                        MfqDuplexSessionParams parameters;
                        const std::string system_prompt = payload.value(
                            "system_prompt",
                            config_.runtime_profile.duplex.system_prompt.value_or(
                                "Streaming Omni Conversation."));
                        scheduler.prepare_duplex_session(
                            system_prompt, parameters);
                        if (payload.contains("reference_audio_features")) {
                            if (!payload["reference_audio_features"].is_string()) {
                                throw ApiError(
                                    400, "invalid_request_error",
                                    "reference_audio_features must be base64 float32 data");
                            }
                            parameters.reference_audio_frames =
                                static_cast<int32_t>(integer_field(
                                    payload, "reference_audio_frames", 0));
                            parameters.reference_audio_features =
                                decode_audio_features(
                                    payload["reference_audio_features"]
                                        .get<std::string>(),
                                    parameters.reference_audio_frames);
                        }
                        const json generation_config = payload.value(
                            "config", json::object());
                        if (!generation_config.is_object()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "session config must be an object");
                        }
                        const auto & defaults = config_.runtime_profile.duplex;
                        parameters.greedy = generation_config.value(
                            "decode_mode",
                            defaults.decode_mode.value_or("sampling")) ==
                            "greedy";
                        parameters.temperature = number_field(
                            generation_config, "temperature",
                            defaults.temperature.value_or(0.7));
                        parameters.top_k = static_cast<int32_t>(integer_field(
                            generation_config, "top_k",
                            defaults.top_k.value_or(100)));
                        parameters.top_p = number_field(
                            generation_config, "top_p",
                            defaults.top_p.value_or(0.8));
                        parameters.listen_probability_scale = number_field(
                            generation_config, "listen_prob_scale",
                            defaults.listen_prob_scale.value_or(1.0));
                        parameters.repetition_penalty = number_field(
                            generation_config, "text_repetition_penalty",
                            defaults.text_repetition_penalty.value_or(1.05));
                        parameters.repetition_window =
                            static_cast<int32_t>(integer_field(
                                generation_config,
                                "text_repetition_window_size",
                                defaults.text_repetition_window_size.value_or(512)));
                        parameters.length_penalty = number_field(
                            generation_config, "length_penalty",
                            defaults.length_penalty.value_or(1.0));
                        parameters.tts_temperature = number_field(
                            generation_config, "tts_temperature",
                            config_.runtime_profile.tts.temperature.value_or(0.8));
                        parameters.tts_repetition_penalty = number_field(
                            generation_config, "tts_repetition_penalty",
                            config_.runtime_profile.tts.repetition_penalty.value_or(1.05));
                        if (generation_config.contains("seed")) {
                            parameters.seed = static_cast<uint64_t>(
                                integer_field(generation_config, "seed", 0));
                        } else {
                            std::random_device random;
                            parameters.seed =
                                (static_cast<uint64_t>(random()) << 32) ^
                                static_cast<uint64_t>(random());
                        }
                        const std::string session_id = request_id("sess-");
                        {
                            std::lock_guard<std::mutex> lock(realtime_gate);
                            realtime.session_id = session_id;
                            realtime.control_tokens =
                                std::unordered_set<int64_t>(
                                    parameters.special_ids.begin(),
                                    parameters.special_ids.end());
                        }
                        try {
                            (void)scheduler.control(parameters);
                            std::lock_guard<std::mutex> lock(realtime_gate);
                            realtime.backend_started = true;
                        } catch (...) {
                            std::lock_guard<std::mutex> lock(realtime_gate);
                            realtime.session_id.clear();
                            realtime.control_tokens.clear();
                            throw;
                        }
                        emit_realtime({
                            {"type", "session.created"},
                            {"session_id", session_id},
                            {"mode", "full_duplex"},
                            {"metrics", {{
                                "backend",
                                "cuda"
                            }}},
                        });
                        send_result(id, {{"status", "ok"}});
                        continue;
                    }
                    if (type != "input.append") {
                        throw ApiError(
                            400, "invalid_request_error",
                            "unsupported realtime message type: " + type);
                    }
                    std::string session_id;
                    std::unordered_set<int64_t> controls;
                    {
                        std::lock_guard<std::mutex> lock(realtime_gate);
                        session_id = realtime.session_id;
                        controls = realtime.control_tokens;
                    }
                    if (session_id.empty()) {
                        throw ApiError(
                            409, "conflict",
                            "session.init must precede input.append");
                    }
                    const json input = body.value("input", json::object());
                    if (!input.is_object()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "input must be an object");
                    }
                    const bool has_audio = input.contains("audio_features");
                    const bool has_text = input.contains("text");
                    if (!has_audio && !has_text) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "input requires audio_features or text");
                    }
                    MfqDuplexStepInput step;
                    if (has_audio) {
                        if (!input["audio_features"].is_string()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "input.audio_features must be base64 float32 data");
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
                                "input.text must be a non-empty string");
                        }
                        scheduler.prepare_duplex_step(
                            input["text"].get<std::string>(), step);
                    }
                    step.max_new_speak_tokens = static_cast<int32_t>(
                        integer_field(
                            input, "max_new_speak_tokens",
                            config_.runtime_profile.duplex
                                .max_new_speak_tokens_per_chunk.value_or(20)));
                    step.force_listen = input.value("force_listen", false);
                    step.force_speak = input.value("force_speak", false);
                    if (step.force_listen && step.force_speak) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_listen and force_speak are mutually exclusive");
                    }
                    const auto result = std::get<MfqDuplexStepResult>(scheduler.control(step));
                    const std::string response_id = request_id("resp-");
                    json metrics = {
                        {"backend", "cuda"},
                        {"wall_clock_ms", result.inference_ms},
                        {"kv_cache_length", result.language_cache_position},
                        {"audio_cache_length", result.audio_cache_position},
                        {"tts_cache_length", result.tts_cache_position},
                        {"audio_chunk_index", result.audio_chunk_index},
                    };
                    const std::string text_delta = scheduler.decode_tokens(
                        result.generated_tokens, controls);
                    if (!text_delta.empty()) {
                        emit_realtime({
                            {"type", "response.output.delta"},
                            {"kind", "text"},
                            {"text", text_delta},
                            {"session_id", session_id},
                            {"response_id", response_id},
                            {"end_of_turn", result.end_of_turn},
                            {"metrics", metrics},
                        });
                    }
                    if (!result.audio_tokens.empty() ||
                        (result.end_of_turn && !result.is_listen)) {
                        emit_realtime({
                            {"type", "response.output.delta"},
                            {"kind", "audio_tokens"},
                            {"audio_tokens", result.audio_tokens},
                            {"session_id", session_id},
                            {"response_id", response_id},
                            {"end_of_turn", result.end_of_turn},
                            {"force_flush", result.tts_force_flush},
                            {"metrics", metrics},
                        });
                    }
                    if (result.is_listen) {
                        emit_realtime({
                            {"type", "response.output.delta"},
                            {"kind", "listen"},
                            {"session_id", session_id},
                            {"response_id", response_id},
                            {"metrics", metrics},
                        });
                    }
                    emit_realtime({
                        {"type", "response.step.done"},
                        {"session_id", session_id},
                        {"response_id", response_id},
                        {"end_of_turn", result.end_of_turn},
                        {"metrics", metrics},
                    });
                    send_result(id, {{"status", "ok"}});
                    continue;
                }
                if (op == "count_tokens") {
                    send_result(id, token_count_json(std::get<mfq::engine::TokenCount>(
                        scheduler.control(parse_token_count_request(params)))));
                    continue;
                }
                if (op == "score") {
                    if (!scheduler.info().probability)
                        throw ApiError(501, "probability_unavailable", "probability scoring is unavailable for this model");
                    send_result(id, likelihood_result_json(std::get<mfq::engine::LikelihoodResult>(
                        scheduler.control(parse_score_request(params)))));
                    continue;
                }
                if (op != "generate") {
                    throw ApiError(
                        404, "unsupported_operation",
                        "unsupported stdio operation: " + op);
                }
                const json body = runtime_generate_body(params);
                auto input = parse_input(
                    body, true, sampling_defaults);
                const RequestSchedule schedule = parse_schedule(body);
                if (body.contains("mfq_multimodal")) {
                    if (!input.sampling.enable_vision) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "vision is disabled for this request");
                    }
                    if (!scheduler.supports_multimodal_generation()) {
                        throw ApiError(
                            501, "unsupported_parameter",
                            "the loaded model has no native vision runtime");
                    }
                    input.media = parse_mfq_vision(
                        body["mfq_multimodal"]);
                }
                RequestWork work = std::move(input);
                auto done = std::make_shared<std::atomic<bool>>(false);
                auto submitted = std::make_shared<std::atomic<bool>>(false);
                tasks.push_back({
                    std::thread([
                        &, id, work = std::move(work), schedule, done, submitted
                    ]() mutable {
                        try {
                            // Admission can wait for a GPU quantum. Keep the
                            // protocol reader available for concurrent arrivals.
                            auto completion = std::make_shared<CompletionStream>(
                                scheduler, work, id, schedule);
                            submitted->store(true,std::memory_order_release);
                            submitted->notify_all();
                            const std::string response_id =
                                request_id("run-");
                            const int64_t created = unix_time_seconds();
                            ActiveRequest active_request(request_metrics_store);
                            auto next_progress = std::chrono::steady_clock::time_point{};
                            while (auto diffs = completion->next()) {
                                const auto now = std::chrono::steady_clock::now();
                                if (now >= next_progress) {
                                    active_request.progress(response_id, completion->progress());
                                    next_progress = now + std::chrono::milliseconds(250);
                                }
                                if (!work.stream) continue;
                                for (const auto& diff : *diffs) {
                                    json delta = chat_diff_json(diff);
                                    if (delta.empty()) continue;
                                    auto event = runtime_generation_event("delta", response_id, created, config_.model_name);
                                    event["delta"] = std::move(delta);
                                    if (!send_event(id, std::move(event))) { completion->cancel(); break; }
                                }
                            }
                            auto result = std::move(completion->result);
                            auto metrics = completion->metrics;
                            const RequestMetricValues metric_values =
                                request_metric_values(result, metrics);
                            log_request_metrics(
                                response_id, true, work.stream,
                                completion->prompt_tokens, work.sampling,
                                result, metric_values);
                            active_request.complete(
                                response_id, true, work.stream,
                                completion->prompt_tokens, result, metric_values);
                            auto performance = request_metric_values_json(
                                metric_values, work.sampling);
                            add_request_runtime_metrics(performance, metrics);
                            if (work.stream) {
                                auto complete = runtime_generation_event(
                                    "complete", response_id, created,
                                    config_.model_name);
                                complete["finish_reason"] = result.finish_reason;
                                complete["metrics"] = std::move(performance);
                                send_event(id, std::move(complete));
                                if (work.include_usage) {
                                    auto usage = runtime_generation_event(
                                        "usage", response_id, created,
                                        config_.model_name);
                                    usage["usage"] = usage_json(
                                        completion->prompt_tokens,
                                        result.completion_tokens);
                                    send_event(id, std::move(usage));
                                }
                            } else {
                                send_event(id, runtime_generation_result(
                                    response_id, created, config_.model_name,
                                    result,
                                    usage_json(
                                        completion->prompt_tokens,
                                        result.completion_tokens),
                                    std::move(performance)));
                            }
                            send_done(id);
                        } catch (const ApiError & error) {
                            send_error(
                                id, error.status, error.type, error.what());
                        } catch (const mfq::engine::InferenceInputError & error) {
                            send_error(
                                id, 400, input_error_type(error.code),
                                error.what());
                        } catch (const std::invalid_argument& error) {
                            send_error(id,400,"invalid_request_error",error.what());
                        } catch (const std::exception & error) {
                            mfq::log_runtime_error("transport", error.what());
                            send_error(
                                id, 500, "server_error", error.what());
                        }
                        submitted->store(true,std::memory_order_release);
                        submitted->notify_all();
                        done->store(true, std::memory_order_release);
                    }),
                    done,
                    submitted,
                    id,
                });
            } catch (const ApiError & error) {
                send_error(id, error.status, error.type, error.what());
            } catch (const mfq::engine::InferenceInputError & error) {
                send_error(
                    id, 400, input_error_type(error.code), error.what());
            } catch (const MfqResourceExhausted & error) {
                send_error(id, 413, "resident_budget_exhausted", error.what());
            } catch (const json::exception & error) {
                send_error(id, 400, "backend_protocol_error", error.what());
            } catch (const std::invalid_argument& error) {
                send_error(id, 400, "invalid_request_error", error.what());
            } catch (const std::exception & error) {
                mfq::log_runtime_error("transport", error.what());
                send_error(id, 500, "server_error", error.what());
            }
        }

        for(const auto& task:tasks)task.submitted->wait(false,std::memory_order_acquire);
        scheduler.cancel_all();
        close_realtime("");
        reap_tasks(true);
        return 0;
    }

private:
    MfqRuntimeTransportConfig config_;
};

} // namespace

void prepare_mfq_stdio_transport() {
    std::lock_guard<std::mutex> lock(stdio_protocol_mutex);
    if (stdio_protocol_fd >= 0) return;
    std::cout.flush();
    std::fflush(stdout);
#ifdef _WIN32
    const HANDLE stdout_handle = reinterpret_cast<HANDLE>(
        _get_osfhandle(_fileno(stdout)));
    HANDLE protocol_handle = nullptr;
    if (stdout_handle == INVALID_HANDLE_VALUE ||
        !DuplicateHandle(
            GetCurrentProcess(), stdout_handle,
            GetCurrentProcess(), &protocol_handle,
            0, FALSE, DUPLICATE_SAME_ACCESS)) {
        throw std::runtime_error("cannot reserve stdout for MFQ stdio protocol");
    }
    stdio_protocol_fd = _open_osfhandle(
        reinterpret_cast<intptr_t>(protocol_handle), _O_WRONLY | _O_BINARY);
    if (stdio_protocol_fd < 0) {
        CloseHandle(protocol_handle);
        throw std::runtime_error("cannot reserve stdout for MFQ stdio protocol");
    }
    if (_dup2(_fileno(stderr), _fileno(stdout)) != 0) {
        _close(stdio_protocol_fd);
        stdio_protocol_fd = -1;
        throw std::runtime_error("cannot redirect runtime stdout to stderr");
    }
#else
    stdio_protocol_fd = ::dup(STDOUT_FILENO);
    if (stdio_protocol_fd < 0 ||
        ::fcntl(stdio_protocol_fd, F_SETFD, FD_CLOEXEC) < 0 ||
        ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        if (stdio_protocol_fd >= 0) ::close(stdio_protocol_fd);
        stdio_protocol_fd = -1;
        throw std::runtime_error("cannot reserve stdout for MFQ stdio protocol");
    }
    std::signal(SIGPIPE, SIG_IGN);
#endif
    std::cout.clear();
}

std::unique_ptr<MfqTransport> make_mfq_stdio_transport(
        MfqRuntimeTransportConfig config) {
    return std::make_unique<MfqStdioTransport>(std::move(config));
}
