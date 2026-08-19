from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "MFQStudio" / "web" / "src" / "styles.css").read_text(
    encoding="utf-8"
)
APP = (ROOT / "MFQStudio" / "web" / "src" / "App.tsx").read_text(
    encoding="utf-8"
)


def test_conversation_titles_can_be_renamed_and_persisted() -> None:
    assert "const [renamingId, setRenamingId]" in APP
    assert "const [renameValue, setRenameValue]" in APP
    assert "async function saveRename(session: Session)" in APP
    assert "api.updateSession(session.id, { title: title || null })" in APP
    assert "setRenamingId(null)" in APP


def test_user_and_assistant_messages_can_be_edited() -> None:
    assert "async function saveEdit(message: Message)" in APP
    assert "setEditDraft({ messageId: message.id, ...parts })" in APP
    assert "const rewound = await api.rewindSession(" in APP
    assert "messages.slice(0, messageIndex)" in APP
    assert "await generate(rewound, parts, false)" in APP
    assert "rewound.revision" in APP
    assert 'tr("保存", "Save")' in APP
    assert 'tr("保存到新分支", "Save as branch")' not in APP
    assert ".message-editor" in CSS
    assert "message.role === \"assistant\" && <textarea" in APP


def test_regenerate_rewinds_to_the_preceding_user_message() -> None:
    assert "async function regenerate(message: Message)" in APP
    assert "for (let cursor = index - 1; cursor >= 0; cursor -= 1)" in APP
    assert "api.rewindSession(" in APP
    assert "user.id," in APP
    assert "await generate(rewound, user.parts, false)" in APP


def test_media_attachments_are_previewed_uploaded_and_sent_as_typed_parts() -> None:
    assert "const [attachments, setAttachments]" in APP
    assert "api.uploadMedia(attachment.file)" in APP
    assert 'accept={attachmentAccept}' in APP
    assert 'type: "video"' in APP
    assert "<MediaPartView" in APP
    assert ".attachment-tray" in CSS


def test_generation_tracks_the_exact_target_message() -> None:
    assert "setLive({ reasoning: \"\", text: \"\", tools: [] })" in APP
    assert "setLive(null)" in APP
    assert "setMessages(persisted)" in APP
    assert "messages.pop()" not in APP
