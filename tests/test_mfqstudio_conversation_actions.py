from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "MFQStudio" / "web" / "src" / "App.tsx").read_text(
    encoding="utf-8"
)


def test_edits_rewind_the_current_session() -> None:
    assert "async function saveEdit(message: Message)" in APP
    assert "const rewound = await api.rewindSession(" in APP
    assert "messages.slice(0, messageIndex)" in APP
    assert "await generate(rewound, text, false)" in APP
    assert "rewound.revision" in APP
    assert 'tr("保存", "Save")' in APP
    assert 'tr("保存到新分支", "Save as branch")' not in APP


def test_regeneration_rewinds_the_current_session() -> None:
    assert "async function regenerate(message: Message)" in APP
    assert "for (let cursor = index - 1; cursor >= 0; cursor -= 1)" in APP
    assert "api.rewindSession(" in APP
    assert "await generate(rewound, text, false)" in APP
