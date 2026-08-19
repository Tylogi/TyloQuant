from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "cpp_runtime" / "web"
HTML = (WEB / "index.html").read_text(encoding="utf-8")
JS = (WEB / "app.js").read_text(encoding="utf-8")
CSS = (WEB / "app.css").read_text(encoding="utf-8")
SERVER_HEADER = (ROOT / "cpp_runtime" / "mfq_server.h").read_text(
    encoding="utf-8"
)
SERVER = (ROOT / "cpp_runtime" / "mfq_server.cpp").read_text(encoding="utf-8")
RUNTIME = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(encoding="utf-8")


def test_markdown_and_latex_assets_are_local_and_loaded_before_app() -> None:
    assets = [
        "vendor/marked/marked.umd.js",
        "vendor/dompurify/purify.min.js",
        "vendor/katex/katex.min.css",
        "vendor/katex/katex.min.js",
        "vendor/katex/auto-render.min.js",
    ]
    for asset in assets:
        assert (WEB / asset).is_file()
        assert asset in HTML
    assert HTML.index("vendor/marked/marked.umd.js") < HTML.index("app.js")
    assert HTML.index("vendor/katex/auto-render.min.js") < HTML.index("app.js")


def test_rich_text_uses_sanitized_gfm_and_katex() -> None:
    assert "globalThis.marked.parse" in JS
    assert "globalThis.DOMPurify.sanitize" in JS
    assert "gfm: true" in JS
    assert "globalThis.renderMathInElement" in JS
    assert '{ left: "$$", right: "$$", display: true }' in JS
    assert '{ left: "$", right: "$", display: false }' in JS
    assert 'FORBID_TAGS: ["script", "style", "template"]' in JS
    assert "decorateCodeBlocks(container);" in JS
    assert ".message-content table" in CSS
    assert ".message-content .katex-display" in CSS


def test_prefill_speed_uses_cuda_events_around_only_the_first_model_eval() -> None:
    assert "using MfqPrefillCallback" in SERVER_HEADER
    assert "const MfqPrefillCallback & on_prefill" in RUNTIME
    assert "class ServerPrefillCudaTimer" in RUNTIME
    assert "cudaEventRecord(started_, stream_)" in RUNTIME
    assert "cudaEvent_t prefill_finished = nullptr" in RUNTIME
    assert RUNTIME.count(
        "prefill_finished, at::cuda::getCurrentCUDAStream()"
    ) == 2
    assert "prefill_timer.finished_event()" in RUNTIME
    assert "const int64_t token = next.item<int64_t>();" in RUNTIME
    assert "const double prefill_ms = prefill_timer.elapsed_ms();" in RUNTIME
    assert "on_prefill(MfqPrefillTiming{" in RUNTIME
    assert "prompt.size() - reused_tokens" in RUNTIME

    sample = RUNTIME.split("static torch::Tensor sample_server_token(", 1)[1]
    sample = sample.split("class ServerPrefillCudaTimer", 1)[0]
    logits = sample.index("auto logits = model.last_logits(ids)")
    finished = sample.index("cudaEventRecord(", logits)
    sampling = sample.index("sample_server_logits(", logits)
    assert logits < finished < sampling
    sampler = RUNTIME.split("static torch::Tensor sample_server_logits(", 1)[1]
    sampler = sampler.split("static torch::Tensor sample_server_token(", 1)[0]
    assert "sample_apply_penalties_cuda(" in sampler

    first = RUNTIME.split("auto sample_first_token = [&]()", 1)[1]
    first = first.split("const char * reprefill_env", 1)[0]
    assert first.index("ServerPrefillCudaTimer prefill_timer") < first.index(
        "auto next = sample_server_token("
    )
    assert first.index("const int64_t token = next.item<int64_t>();") < first.index(
        "prefill_timer.elapsed_ms()"
    )
    assert "1000.0 * metrics.prefill_tokens / metrics.prefill_ms" in SERVER
    assert '{"prefill_tps", values.prefill_tps}' in SERVER
    assert '{"prefill_ms", values.prefill_ms}' in SERVER


def test_monitor_metrics_are_neutral_and_chart_uses_blue() -> None:
    assert ".metric-tile::before" not in CSS
    assert "metric-tile accent-" not in HTML
    assert 'ctx.strokeStyle = "#3b72b9";' in JS
    assert 'ctx.strokeStyle = "#17836f";' not in JS
