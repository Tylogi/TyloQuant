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
LLAMA_CHAT = (
    ROOT / "third_party" / "llama-runtime" / "common" / "chat.cpp"
).read_text(encoding="utf-8")
WEB_JS = (ROOT / "cpp_runtime" / "web" / "app.js").read_text(
    encoding="utf-8"
)
WEB_HTML = (ROOT / "cpp_runtime" / "web" / "index.html").read_text(
    encoding="utf-8"
)
WEB_CSS = (ROOT / "cpp_runtime" / "web" / "app.css").read_text(
    encoding="utf-8"
)


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


def test_webui_keeps_reasoning_separate_and_template_controlled() -> None:
    assert 'reasoning_format: "auto"' in WEB_JS
    assert "requestMessage.reasoning_content = message.reasoning" in WEB_JS
    assert "excludeReasoningFromContext: false" in WEB_JS
    assert "state.generatingMessage === message" in WEB_JS
    assert "reasoning.open = rememberedOpen ?? isGeneratingMessage" in WEB_JS
    assert 'id="setting-exclude-reasoning"' in WEB_HTML


def test_webui_exposes_template_gated_reasoning_effort() -> None:
    assert "chat_template_capabilities_json" in SERVER
    assert '{"chat_template_capabilities", chat_template_capabilities}' in SERVER
    assert 'id="reasoning-effort-control" hidden' in WEB_HTML
    assert 'id="reasoning-effort-select"' in WEB_HTML
    assert '<option value="high">高</option>' in WEB_HTML
    assert '<option value="max">最大</option>' in WEB_HTML
    assert "state.status?.chat_template_capabilities?.reasoning_effort" in WEB_JS
    assert "supportedReasoningEfforts.includes(state.settings.reasoningEffort)" in WEB_JS
    assert "chatTemplateKwargs.reasoning_effort = state.settings.reasoningEffort" in WEB_JS
    assert "control.hidden = !supported || !state.settings.enableThinking" in WEB_JS
    assert ".reasoning-effort-control[hidden]" in WEB_CSS


def test_webui_uses_the_bundled_server_origin() -> None:
    assert "return location.origin;" in WEB_JS
    assert "LEGACY_LOCAL_ENDPOINT" in WEB_JS
    assert 'streamState.finishReason === "length"' in WEB_JS


def test_dsv4_server_uses_exact_stable_prefix_kv_reuse() -> None:
    assert "MfqPromptCachePlan" in SERVER
    assert 'model_type == "deepseek_v4"' in SERVER
    assert 'request_enable_thinking(body) ? "<think>" : "</think>"' in SERVER
    assert "stable_prefix_tokens" in SERVER
    assert '{"prefill_tokens", values.prefill_tokens}' in SERVER


def test_webui_can_reload_model_with_a_new_context() -> None:
    assert 'id="setting-context-window"' in WEB_HTML
    assert 'id="reload-model"' in WEB_HTML
    assert 'fetchJson("/api/reload"' in WEB_JS
    assert 'server.Post("/api/reload"' in SERVER
    assert "context_size must be within the model context capacity" in SERVER
