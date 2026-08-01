from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "cpp_runtime" / "web" / "app.css").read_text(encoding="utf-8")
JS = (ROOT / "cpp_runtime" / "web" / "app.js").read_text(encoding="utf-8")


def test_chat_scroller_has_a_visible_stable_scrollbar() -> None:
    assert ".chat-view.is-visible" in CSS
    assert "overflow: hidden;" in CSS
    assert "scrollbar-gutter: stable;" in CSS
    assert "overflow-y: scroll;" in CSS
    assert "touch-action: pan-y;" in CSS
    assert ".message-scroller::-webkit-scrollbar" in CSS
    assert ".message-scroller::-webkit-scrollbar-thumb" in CSS


def test_streaming_render_respects_manual_scroll_position() -> None:
    assert "followOutput: true" in JS
    assert 'refs["message-scroller"].addEventListener("scroll"' in JS
    assert "distanceFromBottom" in JS
    assert "state.followOutput" in JS
    assert "previousScrollTop" in JS


def test_sending_a_message_resumes_tail_following() -> None:
    assert "state.followOutput = true;" in JS


def test_reasoning_panel_grows_without_an_inner_scrollbar() -> None:
    reasoning_content = CSS.split(".reasoning-content {", 1)[1].split("}", 1)[0]
    assert "max-height" not in reasoning_content
    assert "overflow-y" not in reasoning_content
    assert "overflow-wrap: anywhere;" in reasoning_content


def test_reasoning_open_state_survives_streaming_rerenders() -> None:
    assert "reasoningOpenState: new WeakMap()" in JS
    assert "const rememberedOpen = state.reasoningOpenState.get(message);" in JS
    assert "reasoning.open = rememberedOpen ?? isGeneratingMessage;" in JS
    assert 'summary.addEventListener("pointerdown"' in JS
    assert "reasoning.open = nextOpen;" in JS


def test_default_generation_limit_is_4096_tokens() -> None:
    assert "maxTokens: 4096," in JS
    assert (
        'numberInput(refs["setting-max-tokens"], 4096, 1, 8192)'
        in JS
    )
