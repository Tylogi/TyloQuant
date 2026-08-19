from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "MFQStudio" / "web" / "src" / "styles.css").read_text(
    encoding="utf-8"
)
APP = (ROOT / "MFQStudio" / "web" / "src" / "App.tsx").read_text(
    encoding="utf-8"
)


def test_chat_scroller_owns_vertical_overflow() -> None:
    assert ".chat-view" in CSS
    assert ".message-scroller { overflow-y: auto; }" in CSS
    assert ".message-list" in CSS


def test_streaming_render_follows_only_while_the_user_is_near_the_tail() -> None:
    assert "const messageScrollerRef = useRef<HTMLDivElement | null>(null)" in APP
    assert "const autoFollowOutputRef = useRef(true)" in APP
    assert "distanceFromBottom <= 8" in APP
    assert "if (!scroller || !autoFollowOutputRef.current) return" in APP
    assert "scroller.scrollTop = scroller.scrollHeight" in APP
    assert "onScroll={handleMessageScroll}" in APP
    assert "scrollIntoView" not in APP
    assert "currentVoiceMessages, liveVoice, live, busy" in APP


def test_sending_a_message_adds_an_optimistic_user_turn() -> None:
    assert "if (optimistic)" in APP
    assert 'role: "user"' in APP


def test_reasoning_panel_grows_without_an_inner_scrollbar() -> None:
    reasoning = CSS.split(".reasoning {", 1)[1].split("}", 1)[0]
    assert "max-height" not in reasoning
    assert "overflow-y" not in reasoning
    assert ".reasoning .rich-text" in CSS


def test_completed_and_live_reasoning_have_explicit_disclosure_state() -> None:
    assert '<details className="reasoning"><summary>' in APP
    assert '<details className="reasoning" open>' in APP


def test_default_generation_limit_is_4096_tokens() -> None:
    assert "maxTokens: 4096," in APP
    assert "max={65536}" in APP
