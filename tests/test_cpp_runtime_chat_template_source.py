from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = (ROOT / "cpp_runtime" / "mfq_server.cpp").read_text(
    encoding="utf-8"
)
DECODE = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(
    encoding="utf-8"
)
CMAKE = (ROOT / "cpp_runtime" / "CMakeLists.txt").read_text(
    encoding="utf-8"
)
METAL_CMAKE = (
    ROOT / "cpp_runtime" / "metal" / "CMakeLists.txt"
).read_text(encoding="utf-8")
METAL_DECODE = (
    ROOT / "cpp_runtime" / "metal" / "mfq_decode_mlx.cpp"
).read_text(encoding="utf-8")
METAL_DSV4 = (
    ROOT / "cpp_runtime" / "metal" / "mlx_deepseek_v4_causal_lm.cpp"
).read_text(encoding="utf-8")
STUDIO_APP = (ROOT / "MFQStudio" / "src" / "App.tsx").read_text(
    encoding="utf-8"
)
STUDIO_API = (ROOT / "MFQStudio" / "src" / "api.ts").read_text(
    encoding="utf-8"
)
LLAMA_CHAT = (
    ROOT / "third_party" / "llama-runtime" / "common" / "chat.cpp"
).read_text(encoding="utf-8")
LLAMA_CHAT_H = (
    ROOT / "third_party" / "llama-runtime" / "common" / "chat.h"
).read_text(encoding="utf-8")
def _section(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def test_server_uses_native_gguf_jinja_template_and_common_parser() -> None:
    assert "common_chat_templates_apply" in SERVER
    assert "common_chat_msgs_parse_oaicompat" in SERVER
    assert "common_chat_parse" in SERVER
    assert "common_chat_msg_diff::compute_diffs" in SERVER
    assert "config.tokenizer_gguf.empty()" in SERVER
    assert "config.tokenizer_model.empty()" in SERVER
    assert "requires an embedded or external tokenizer GGUF" in SERVER
    assert "format_gemma4_chat_prompt" not in SERVER
    assert "format_dsv4_chat_prompt" not in SERVER


def test_server_enforces_complete_chat_template_tool_calls() -> None:
    assert "LlamaGrammarConstraint" in SERVER
    assert "make_token_constraint(tokenizer, chat_params)" in SERVER
    assert "work.token_constraint" in SERVER
    assert "if (partial)" in SERVER
    assert "parsed.tool_calls.clear()" in SERVER
    assert 'uses_tool_calls ? "tool_calls" : "function_calls"' in LLAMA_CHAT
    assert 'src.find("tool_calls") != std::string::npos' in LLAMA_CHAT
    assert "token_constraint->apply" in METAL_DSV4
    assert "token_constraint->accept" in METAL_DSV4
    assert "token_constraint," in METAL_DECODE
    assert "CUDA constrained sampler returned an invalid token" in DECODE
    assert "masked.to(logits.device())" in DECODE
    assert "!token_constraint &&" in DECODE


def test_native_server_cancels_active_session_generation_per_token() -> None:
    assert 'R"(/api/runtime/sessions/([A-Za-z0-9._:-]{1,128})/cancel)"' in SERVER
    assert "request_cancellations.cancel(session_id)" in SERVER
    assert "cancel_requested->load(std::memory_order_acquire)" in SERVER
    assert 'result.finish_reason = "cancelled"' in SERVER
    assert "!result.cancelled && !result.tool_calls.empty()" in SERVER
    assert "work.cache_plan.stable_prefix_tokens = 0;" in SERVER
    assert "work.cache_plan = {};" not in SERVER


def test_server_links_matching_llama_common_runtime() -> None:
    assert 'set(MFQ_LLAMA_SOURCE_DIR' in CMAKE
    assert 'add_subdirectory(' in CMAKE
    assert "mfq_llama_common" in CMAKE
    assert "common/chat.h" in CMAKE
    assert "MFQ_LLAMA_RUNTIME_DYLIBS" in METAL_CMAKE
    assert "BUILD_WITH_INSTALL_RPATH ON" in METAL_CMAKE


def test_server_rejects_external_runtime_assets() -> None:
    assert "MFQ server does not accept an external model config" in DECODE
    assert (
        "MFQ server requires embedded model config, tokenizer, "
        in DECODE
    )
    assert "server_config.tokenizer_model" not in DECODE


def test_studio_keeps_reasoning_separate_and_template_controlled() -> None:
    assert "reasoning: string;" in STUDIO_APP
    assert "include_reasoning_history: !effectiveSettings.excludeReasoning" in STUDIO_APP
    assert "effectiveSettings.enableThinking" in STUDIO_APP
    assert "effectiveSettings.excludeReasoning" in STUDIO_APP


def test_studio_exposes_template_gated_reasoning_effort() -> None:
    assert "chat_template_capabilities_json" in SERVER
    assert '{"chat_template_capabilities", chat_template_capabilities}' in SERVER
    assert 'chat_template.find("enable_thinking")' in SERVER
    assert 'runtime?.chat_template_capabilities?.thinking?.supported' in STUDIO_APP
    assert "runtime?.chat_template_capabilities?.reasoning_effort?.values" in STUDIO_APP
    assert "effectiveSettings.enableThinking && reasoningValues.length > 0" in STUDIO_APP
    assert "reasoning_effort: effectiveSettings.reasoningEffort || null" in STUDIO_APP


def test_native_server_does_not_bundle_or_mount_a_webui() -> None:
    assert "mfq-web-assets" not in CMAKE
    assert "Copying MFQ WebUI assets" not in METAL_CMAKE
    assert "--web-root" not in DECODE
    assert "--web-root" not in METAL_DECODE
    assert 'server.Get("/admin"' not in SERVER
    assert "set_mount_point" not in SERVER


def test_dsv4_server_uses_exact_stable_prefix_kv_reuse() -> None:
    assert "MfqPromptCachePlan" in SERVER
    assert 'model_type == "deepseek_v4"' in SERVER
    assert 'work.sampling.enable_thinking ? "<think>" : "</think>"' in SERVER
    assert "stable_prefix_tokens" in SERVER
    assert '{"prefill_tokens", values.prefill_tokens}' in SERVER


def test_studio_can_reload_model_with_a_new_context() -> None:
    assert "async function reloadRuntime()" in STUDIO_APP
    assert "api.reloadRuntime(contextSize)" in STUDIO_APP
    assert 'request("/api/v1/runtime/reload"' in STUDIO_API
    assert 'server.Post("/api/reload"' in SERVER
    assert "context_size must be within the model context capacity" in SERVER


def test_server_supports_structured_output_and_named_tool_choice() -> None:
    assert "static std::string request_json_schema(const json & body)" in SERVER
    assert 'body.contains("response_format")' in SERVER
    assert 'type == "json_object"' in SERVER
    assert 'type == "json_schema"' in SERVER
    assert "inputs.json_schema = request_json_schema(body);" in SERVER
    assert "tool_choice.is_object()" in SERVER
    assert "named tool_choice must select a function name" in SERVER
    assert "named tool_choice does not match any supplied tool" in SERVER
    assert "inputs.tools = {*selected};" in SERVER
    assert "inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;" in SERVER


def test_dsv4_template_preserves_message_extensions() -> None:
    assert "std::map<std::string, std::string>        extra_fields;" in LLAMA_CHAT_H
    assert 'for (const char * key : {"task", "tools", "response_format"})' in LLAMA_CHAT
    assert "msg.extra_fields[key] = message.at(key).dump();" in LLAMA_CHAT
    assert "for (const auto & [key, value] : extra_fields)" in LLAMA_CHAT
    assert "jmsg[key] = json::parse(value);" in LLAMA_CHAT
    assert "has_message_scoped_tools" in LLAMA_CHAT


def test_dsv4_template_uses_message_scoped_tools_and_schema() -> None:
    dsv4 = _section(
        LLAMA_CHAT,
        "static common_chat_params common_chat_params_init_deepseek_v3_2",
        "static common_chat_params common_chat_params_init_cohere2moe",
    )
    gpt_oss = _section(
        LLAMA_CHAT,
        "static common_chat_params common_chat_params_init_gpt_oss",
        "static common_chat_params common_chat_params_init_gemma4",
    )

    assert "json available_tools = inputs.tools.is_array()" in dsv4
    assert "std::set<std::string> available_tool_names;" in dsv4
    assert 'message.contains("tools")' in dsv4
    assert "available_tools.push_back(tool);" in dsv4
    assert 'message.contains("response_format")' in dsv4
    assert "normalize_response_format(message.at(\"response_format\"))" in dsv4
    assert 'type == "text"' in dsv4
    assert 'type == "json_object"' in dsv4
    assert 'type == "json_schema"' in dsv4
    assert 'response_format = response_format.at("schema");' in dsv4
    assert "foreach_function(available_tools" in dsv4
    assert 'p.schema(p.json(), "response-format-schema", response_schema)' in dsv4
    assert "auto schema = response_schema;" in dsv4
    assert "json available_tools = inputs.tools.is_array()" not in gpt_oss


def test_dsv4_disabled_thinking_consumes_close_marker() -> None:
    dsv4 = _section(
        LLAMA_CHAT,
        "static common_chat_params common_chat_params_init_deepseek_v3_2",
        "static common_chat_params common_chat_params_init_cohere2moe",
    )
    assert "p.optional(p.literal(THINK_END)) +" in dsv4
    assert "p.literal(THINK_START) +" in dsv4
    assert "p.until(THINK_END) +" in dsv4
    assert "p.literal(THINK_END));" in dsv4
