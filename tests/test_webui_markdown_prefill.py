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
METAL_TIMER = (ROOT / "cpp_runtime" / "metal" / "mlx_eval_timing.h").read_text(
    encoding="utf-8"
)
METAL_QWEN = (
    ROOT / "cpp_runtime" / "metal" / "mlx_qwen35_causal_lm.cpp"
).read_text(encoding="utf-8")
METAL_DSV4 = (
    ROOT / "cpp_runtime" / "metal" / "mlx_deepseek_v4_causal_lm.cpp"
).read_text(encoding="utf-8")


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
    assert "on_prefill(prompt.size(), prefill_ms);" in RUNTIME

    sample = RUNTIME.split("static torch::Tensor sample_server_token(", 1)[1]
    sample = sample.split("class ServerPrefillCudaTimer", 1)[0]
    logits = sample.index("auto logits = model.last_logits(ids)")
    finished = sample.index("cudaEventRecord(", logits)
    penalties = sample.index("sample_apply_penalties_cuda(", logits)
    assert logits < finished < penalties

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


def test_metal_prefill_speed_accumulates_only_explicit_mlx_evaluations() -> None:
    assert "class ScopedMlxEvaluationTiming" in METAL_TIMER
    assert "active_mlx_evaluation_ms" in METAL_TIMER
    assert "std::chrono::steady_clock::now()" in METAL_TIMER
    assert "CPU graph construction" in METAL_TIMER

    for source in (METAL_QWEN, METAL_DSV4):
        assert "prefill_started" not in source
        assert "prefill_evaluation_ms" in source
        assert "ScopedMlxEvaluationTiming timing" in source
        assert "detail::eval_with_timing(value);" in source
        assert "prefill_evaluation_ms);" in source

    cache_setup = METAL_QWEN.index("prepare_cache_for_prefill(", METAL_QWEN.index("std::int32_t MlxQwen35CausalLm::generate"))
    timed_eval = METAL_QWEN.index("ScopedMlxEvaluationTiming timing", cache_setup)
    assert cache_setup < timed_eval
    assert "Cache allocation/zeroing is request setup" in METAL_QWEN

    dsv4_generate = METAL_DSV4.index("std::int32_t MlxDeepseekV4CausalLm::generate")
    dsv4_reset = METAL_DSV4.index("reset_cache(1);", dsv4_generate)
    dsv4_timed_eval = METAL_DSV4.index("ScopedMlxEvaluationTiming timing", dsv4_reset)
    assert dsv4_reset < dsv4_timed_eval
    assert "State allocation/zeroing is request setup" in METAL_DSV4

    # DeepSeek eagerly materializes every layer and state. These evaluations
    # must contribute while the surrounding CPU graph construction does not.
    assert "detail::eval_with_timing(hidden_values);" in METAL_DSV4
    assert "detail::eval_with_timing(std::move(arrays));" in METAL_DSV4


def test_monitor_metrics_are_neutral_and_chart_uses_blue() -> None:
    assert ".metric-tile::before" not in CSS
    assert "metric-tile accent-" not in HTML
    assert 'ctx.strokeStyle = "#3b72b9";' in JS
    assert 'ctx.strokeStyle = "#17836f";' not in JS


def test_prefill_speed_is_visible_in_summary_and_request_details() -> None:
    for element_id in (
        "top-prefill-tps",
        "metric-prefill-tps",
        "last-prefill-tps",
        "last-prefill-ms",
    ):
        assert f'id="{element_id}"' in HTML
        assert f'refs["{element_id}"]' in JS
