from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "cpp_runtime" / "web" / "app.css").read_text(encoding="utf-8")
HTML = (ROOT / "cpp_runtime" / "web" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "cpp_runtime" / "web" / "app.js").read_text(encoding="utf-8")


def test_conversation_titles_can_be_renamed_and_persisted() -> None:
    assert 'id="icon-edit"' in HTML
    assert "renamingConversationId" in JS
    assert 'input.className = "conversation-rename-input"' in JS
    assert "finishConversationRename(conversation, input.value)" in JS
    assert "conversation.title = title.slice(0, 80);" in JS
    assert "persistState();" in JS
    assert ".conversation-rename" in CSS


def test_user_and_assistant_messages_can_be_edited() -> None:
    assert "function startMessageEdit(message)" in JS
    assert 'messageActionButton("edit", "编辑"' in JS
    assert 'addField("消息", message.content' in JS
    assert 'addField("思考过程", parts.reasoning' in JS
    assert 'addField("回答", parts.content' in JS
    assert "message.content = content;" in JS
    assert "message.reasoning = reasoning;" in JS
    assert ".message-editor" in CSS
    assert 'if (message.role === "assistant") body.append(actions);' in JS
    assert (
        'if (actions && message.role === "user") article.append(actions);'
        in JS
    )


def test_reroll_uses_only_preceding_context_and_restores_on_failure() -> None:
    assert "async function rerollAssistant(conversation, assistant)" in JS
    assert "conversationRequestMessages(conversation, assistantIndex)" in JS
    assert "restoreOnFailure: snapshot" in JS
    assert "Object.assign(assistant, options.restoreOnFailure);" in JS
    assert 'messageActionButton("refresh", "重新生成"' in JS


def test_generation_tracks_the_exact_target_message() -> None:
    assert "state.generatingMessage = assistant;" in JS
    assert "state.generatingMessage === message" in JS
    assert "state.generatingMessage = null;" in JS
    assert "conversation.messages.pop()" not in JS
