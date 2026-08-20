from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "MFQStudio" / "web"
APP = (WEB / "src" / "App.tsx").read_text(encoding="utf-8")
API = (WEB / "src" / "api.ts").read_text(encoding="utf-8")
MARKDOWN = (WEB / "src" / "Markdown.tsx").read_text(encoding="utf-8")
CSS = (WEB / "src" / "styles.css").read_text(encoding="utf-8")
PACKAGE = (WEB / "package.json").read_text(encoding="utf-8")
SERVER_HEADER = (ROOT / "cpp_runtime" / "mfq_server.h").read_text(
    encoding="utf-8"
)
SERVER = (ROOT / "cpp_runtime" / "mfq_server.cpp").read_text(encoding="utf-8")
RUNTIME = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(encoding="utf-8")


def test_studio_bundles_markdown_sanitization_and_latex_dependencies() -> None:
    for dependency in ("dompurify", "katex", "marked"):
        assert f'"{dependency}"' in PACKAGE
    assert 'import "katex/dist/katex.min.css"' in (
        WEB / "src" / "main.tsx"
    ).read_text(encoding="utf-8")


def test_rich_text_uses_sanitized_gfm_and_katex() -> None:
    assert "marked.parse" in MARKDOWN
    assert "DOMPurify.sanitize" in MARKDOWN
    assert "gfm: true" in MARKDOWN
    assert "renderMathInElement" in MARKDOWN
    assert '{ left: "$$", right: "$$", display: true }' in MARKDOWN
    assert '{ left: "$", right: "$", display: false }' in MARKDOWN
    assert 'ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]' in MARKDOWN
    assert 'button.className = "code-copy"' in MARKDOWN
    assert ".rich-text table" in CSS


def test_prefill_speed_uses_cuda_events_around_only_the_first_model_eval() -> None:
    assert "using MfqPrefillCallback" in SERVER_HEADER
    assert "const MfqPrefillCallback & on_prefill" in RUNTIME
    assert "class ServerPrefillCudaTimer" in RUNTIME
    assert "cudaEventRecord(started_, stream_)" in RUNTIME
    assert "cudaEvent_t prefill_finished = nullptr" in RUNTIME
    assert RUNTIME.count(
        "prefill_finished, mfq_get_current_cuda_stream()"
    ) == 2
    assert "prefill_timer.finished_event()" in RUNTIME
    assert "const int64_t token = next.item<int64_t>();" in RUNTIME
    assert "const double prefill_ms = prefill_timer.elapsed_ms();" in RUNTIME
    assert "on_prefill(MfqPrefillTiming{" in RUNTIME
    assert "prompt.size() - reused_tokens,\n                prefill_ms,\n                0.0,\n                prefill_ms" in RUNTIME

    sample = RUNTIME.split(
        "static mfq_tensor_backend::Tensor sample_server_token(", 1
    )[1]
    sample = sample.split("class ServerPrefillCudaTimer", 1)[0]
    logits = sample.index("auto logits = model.last_logits(ids)")
    finished = sample.index("cudaEventRecord(", logits)
    sampling = sample.index("sample_server_logits(", logits)
    assert logits < finished < sampling
    sampler = RUNTIME.split(
        "static mfq_tensor_backend::Tensor sample_server_logits(", 1
    )[1]
    sampler = sampler.split(
        "static mfq_tensor_backend::Tensor sample_server_token(", 1
    )[0]
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


def test_studio_displays_native_multimodal_prefill_without_media_preparation() -> None:
    assert "model_prefill_ms?: number;" in API
    assert "complete_prefill_ms?: number;" in API
    assert "complete_prefill_tps?: number;" in API
    assert "function displayPrefillMetric" in APP
    assert "const nativeMilliseconds = Number(metrics.ttft_ms);" in APP
    assert "const modelMilliseconds = Number(metrics.model_prefill_ms);" in APP
    assert "tokensPerSecond: (tokens * 1000) / milliseconds" in APP
    assert "? (tokens * 1000) / nativeMilliseconds" in APP
    assert "preferPositiveMetric(last?.ttft_ms, last?.complete_prefill_ms)" in APP
    assert "displayPrefillMetric(response?.performance)" in APP
    assert "const lastPrefill = displayPrefillMetric(last);" in APP
    assert "formatNumber(lastPrefill.tokensPerSecond, 1)" in APP
    assert "formatNumber(lastPrefill.milliseconds, 1)" in APP
    assert "formatNumber(last?.prefill_tps, 1)" not in APP


def test_monitor_metrics_use_one_neutral_chart_style() -> None:
    assert ".metric-tile::before" not in CSS
    assert "metric-tile accent-" not in APP
    assert ".runtime-chart polyline" in CSS
