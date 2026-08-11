from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import UUID

import pytest

from mfqd.models import (
    CreateSessionRequest,
    ErrorDetail,
    ForkSessionRequest,
    MessageRole,
    ResponseStatus,
    SessionMode,
    SessionState,
    TokenUsage,
    UpdateSessionRequest,
)
from mfqd.storage import (
    IdempotencyConflictError,
    MessageNotFoundError,
    ResponseInProgressError,
    RevisionConflictError,
    SessionNotFoundError,
    SessionStore,
)

SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
FORK_ID = UUID("22222222-2222-4222-8222-222222222222")
FIRST_MESSAGE_ID = UUID("33333333-3333-4333-8333-333333333333")
SECOND_MESSAGE_ID = UUID("44444444-4444-4444-8444-444444444444")
FORK_MESSAGE_ID = UUID("55555555-5555-4555-8555-555555555555")
REQUEST_ID = UUID("66666666-6666-4666-8666-666666666666")
RESPONSE_ID = UUID("77777777-7777-4777-8777-777777777777")
OUTPUT_MESSAGE_ID = UUID("88888888-8888-4888-8888-888888888888")
NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)


def make_store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "mfqd.sqlite3")


def test_session_persists_in_wal_mode(tmp_path) -> None:
    path = tmp_path / "mfqd.sqlite3"
    store = SessionStore(path)
    created = store.create_session(
        CreateSessionRequest(model="model-a", mode=SessionMode.FULL_DUPLEX),
        session_id=SESSION_ID,
        now=NOW,
    )
    reopened = SessionStore(path)
    assert reopened.journal_mode() == "wal"
    assert reopened.get_session(SESSION_ID) == created
    assert created.revision == 0


def test_session_mode_can_change_without_erasing_its_title(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_session(
        CreateSessionRequest(model="model-a", title="conversation"),
        session_id=SESSION_ID,
        now=NOW,
    )
    updated = store.update_session(
        SESSION_ID,
        UpdateSessionRequest(mode=SessionMode.FULL_DUPLEX),
        now=NOW,
    )
    assert updated.mode == SessionMode.FULL_DUPLEX
    assert updated.title == "conversation"


def test_append_is_revision_guarded_and_persistent(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_session(CreateSessionRequest(model="model-a"), session_id=SESSION_ID, now=NOW)
    session, message = store.append_message(
        SESSION_ID,
        0,
        MessageRole.USER,
        [{"type": "text", "text": "hello"}],
        message_id=FIRST_MESSAGE_ID,
        now=NOW,
    )
    assert session.revision == 1
    assert message.parent_id is None
    assert store.list_messages(SESSION_ID) == [message]
    with pytest.raises(RevisionConflictError) as caught:
        store.append_message(
            SESSION_ID,
            0,
            MessageRole.USER,
            [{"type": "text", "text": "stale"}],
        )
    assert (caught.value.expected, caught.value.actual) == (0, 1)


def test_fork_shares_prefix_then_diverges(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_session(CreateSessionRequest(model="model-a"), session_id=SESSION_ID, now=NOW)
    store.append_message(
        SESSION_ID,
        0,
        MessageRole.USER,
        [{"type": "text", "text": "question"}],
        message_id=FIRST_MESSAGE_ID,
        now=NOW,
    )
    store.append_message(
        SESSION_ID,
        1,
        MessageRole.ASSISTANT,
        [{"type": "text", "text": "answer"}],
        message_id=SECOND_MESSAGE_ID,
        now=NOW,
    )
    fork = store.fork_session(
        SESSION_ID,
        ForkSessionRequest(at_message_id=FIRST_MESSAGE_ID, title="branch"),
        target_session_id=FORK_ID,
        now=NOW,
    )
    assert fork.revision == 1
    assert [message.id for message in store.list_messages(FORK_ID)] == [FIRST_MESSAGE_ID]
    store.append_message(
        FORK_ID,
        1,
        MessageRole.USER,
        [{"type": "text", "text": "different"}],
        message_id=FORK_MESSAGE_ID,
        now=NOW,
    )
    assert [message.id for message in store.list_messages(SESSION_ID)] == [
        FIRST_MESSAGE_ID,
        SECOND_MESSAGE_ID,
    ]
    assert [message.id for message in store.list_messages(FORK_ID)] == [
        FIRST_MESSAGE_ID,
        FORK_MESSAGE_ID,
    ]


def test_fork_rejects_message_from_another_branch(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_session(CreateSessionRequest(model="model-a"), session_id=SESSION_ID, now=NOW)
    with pytest.raises(MessageNotFoundError):
        store.fork_session(
            SESSION_ID,
            ForkSessionRequest(at_message_id=FIRST_MESSAGE_ID),
            target_session_id=FORK_ID,
        )


def test_concurrent_append_accepts_exactly_one_revision(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_session(CreateSessionRequest(model="model-a"), session_id=SESSION_ID, now=NOW)

    def append(text: str) -> str:
        try:
            store.append_message(
                SESSION_ID,
                0,
                MessageRole.USER,
                [{"type": "text", "text": text}],
            )
        except RevisionConflictError:
            return "conflict"
        return "accepted"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(append, ("one", "two")))
    assert sorted(outcomes) == ["accepted", "conflict"]
    assert store.get_session(SESSION_ID).revision == 1
    assert len(store.list_messages(SESSION_ID)) == 1


def test_delete_preserves_shared_messages_until_last_session_is_removed(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_session(CreateSessionRequest(model="model-a"), session_id=SESSION_ID, now=NOW)
    store.append_message(
        SESSION_ID,
        0,
        MessageRole.USER,
        [{"type": "text", "text": "shared"}],
        message_id=FIRST_MESSAGE_ID,
        now=NOW,
    )
    store.fork_session(
        SESSION_ID,
        ForkSessionRequest(),
        target_session_id=FORK_ID,
        now=NOW,
    )
    store.delete_session(SESSION_ID)
    assert [message.id for message in store.list_messages(FORK_ID)] == [FIRST_MESSAGE_ID]
    with pytest.raises(SessionNotFoundError):
        store.get_session(SESSION_ID)
    store.delete_session(FORK_ID)
    with pytest.raises(SessionNotFoundError):
        store.list_messages(FORK_ID)


def test_response_lifecycle_is_persistent_and_idempotent(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_session(CreateSessionRequest(model="model-a"), session_id=SESSION_ID, now=NOW)
    started = store.begin_response(
        SESSION_ID,
        REQUEST_ID,
        RESPONSE_ID,
        "same-request",
        0,
        [{"type": "text", "text": "question"}],
        input_message_id=FIRST_MESSAGE_ID,
        now=NOW,
    )
    assert started.started
    assert started.response.status == ResponseStatus.RUNNING
    assert started.session.state == SessionState.PROCESSING
    assert started.session.revision == 1

    replay = store.begin_response(
        SESSION_ID,
        REQUEST_ID,
        UUID("99999999-9999-4999-8999-999999999999"),
        "same-request",
        0,
        [{"type": "text", "text": "question"}],
    )
    assert not replay.started
    assert replay.response.id == RESPONSE_ID
    with pytest.raises(IdempotencyConflictError):
        store.begin_response(
            SESSION_ID,
            REQUEST_ID,
            UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            "different-request",
            1,
            [{"type": "text", "text": "changed"}],
        )

    completed = store.complete_response(
        RESPONSE_ID,
        [{"type": "text", "text": "answer"}],
        "stop",
        TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        output_message_id=OUTPUT_MESSAGE_ID,
        now=NOW,
    )
    assert completed.status == ResponseStatus.COMPLETED
    assert completed.output[0].type == "text"
    assert completed.finish_reason == "stop"
    assert completed.usage is not None and completed.usage.total_tokens == 5
    session = store.get_session(SESSION_ID)
    assert session.state == SessionState.IDLE
    assert session.revision == 2
    assert [message.id for message in store.list_messages(SESSION_ID)] == [
        FIRST_MESSAGE_ID,
        OUTPUT_MESSAGE_ID,
    ]


def test_failed_response_records_error_and_allows_next_turn(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_session(CreateSessionRequest(model="model-a"), session_id=SESSION_ID, now=NOW)
    store.begin_response(
        SESSION_ID,
        REQUEST_ID,
        RESPONSE_ID,
        "request",
        0,
        [{"type": "text", "text": "question"}],
        now=NOW,
    )
    failed = store.terminate_response(
        RESPONSE_ID,
        ErrorDetail(code="backend_timeout", message="timed out", retryable=True),
        now=NOW,
    )
    assert failed.status == ResponseStatus.FAILED
    assert failed.error is not None and failed.error.retryable
    assert store.get_session(SESSION_ID).state == SessionState.ERROR
    session, _ = store.append_message(
        SESSION_ID,
        1,
        MessageRole.USER,
        [{"type": "text", "text": "continue"}],
    )
    assert session.revision == 2


def test_running_response_blocks_parallel_generation_and_deletion(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_session(CreateSessionRequest(model="model-a"), session_id=SESSION_ID, now=NOW)
    store.begin_response(
        SESSION_ID,
        REQUEST_ID,
        RESPONSE_ID,
        "request",
        0,
        [{"type": "text", "text": "question"}],
        now=NOW,
    )
    with pytest.raises(ResponseInProgressError):
        store.begin_response(
            SESSION_ID,
            UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            "parallel",
            1,
            [{"type": "text", "text": "parallel"}],
        )
    with pytest.raises(ResponseInProgressError):
        store.delete_session(SESSION_ID)
