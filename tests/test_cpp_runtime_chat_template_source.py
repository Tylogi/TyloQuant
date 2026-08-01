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
WEB_JS = (ROOT / "cpp_runtime" / "web" / "app.js").read_text(
    encoding="utf-8"
)
WEB_HTML = (ROOT / "cpp_runtime" / "web" / "index.html").read_text(
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


def test_server_links_matching_llama_common_runtime() -> None:
    assert "MFQ_LLAMA_COMMON_LIBRARY" in CMAKE
    assert "mfq_llama_common" in CMAKE
    assert "llama-common.dll" in CMAKE
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


def test_webui_uses_the_bundled_server_origin() -> None:
    assert "return location.origin;" in WEB_JS
    assert "LEGACY_LOCAL_ENDPOINT" in WEB_JS
    assert 'streamState.finishReason === "length"' in WEB_JS
