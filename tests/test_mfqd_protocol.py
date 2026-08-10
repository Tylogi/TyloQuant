from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.resources import files
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from mfqd.api import create_contract_app
from mfqd.models import (
    PROTOCOL_VERSION,
    ContentPart,
    CreateResponseRequest,
    InputAudioDelta,
    RealtimeFrame,
    ResponseTextDelta,
    SessionState,
    SessionStateChanged,
)
from mfqd.openapi import REALTIME_EVENTS, build_openapi_schema, render_openapi

SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
RESPONSE_ID = UUID("22222222-2222-4222-8222-222222222222")
REQUEST_ID = UUID("33333333-3333-4333-8333-333333333333")


def test_content_parts_are_strictly_discriminated() -> None:
    adapter = TypeAdapter(ContentPart)
    part = adapter.validate_python({"type": "text", "text": "hello"})
    assert part.type == "text"
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "video", "text": "hello"})
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "text", "text": "hello", "unknown": True})


def test_response_request_requires_revision_and_request_identity() -> None:
    request = CreateResponseRequest(
        request_id=REQUEST_ID,
        expected_revision=7,
        input=[{"type": "text", "text": "hello"}],
    )
    assert request.expected_revision == 7
    assert request.request_id == REQUEST_ID
    with pytest.raises(ValidationError):
        CreateResponseRequest.model_validate(
            {"request_id": str(REQUEST_ID), "expected_revision": -1, "input": []}
        )


def test_realtime_frames_cover_audio_text_state_and_sequence() -> None:
    timestamp = datetime(2026, 8, 10, tzinfo=timezone.utc)
    frames = [
        RealtimeFrame(
            session_id=SESSION_ID,
            sequence=0,
            timestamp=timestamp,
            payload=InputAudioDelta(
                audio_sequence=0,
                timestamp_ms=0,
                data_base64="AAA=",
            ),
        ),
        RealtimeFrame(
            session_id=SESSION_ID,
            sequence=1,
            timestamp=timestamp,
            payload=ResponseTextDelta(response_id=RESPONSE_ID, delta="ok"),
        ),
        RealtimeFrame(
            session_id=SESSION_ID,
            sequence=2,
            timestamp=timestamp,
            payload=SessionStateChanged(state=SessionState.SPEAKING, revision=3),
        ),
    ]
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert all(frame.protocol_version == PROTOCOL_VERSION for frame in frames)


def test_openapi_contract_has_all_native_routes_and_realtime_extension() -> None:
    schema = build_openapi_schema()
    expected_paths = {
        "/api/v1/sessions",
        "/api/v1/sessions/{session_id}/messages",
        "/api/v1/sessions/{session_id}/responses",
        "/api/v1/sessions/{session_id}/fork",
        "/api/v1/sessions/{session_id}",
        "/api/v1/media",
        "/api/v1/models/load",
        "/api/v1/models/unload",
        "/api/v1/runtime/instances",
    }
    assert set(schema["paths"]) == expected_paths
    assert schema["x-mfqd-protocol-version"] == PROTOCOL_VERSION
    assert schema["x-mfqd-websocket"]["events"] == REALTIME_EVENTS
    assert any(route.path == "/api/v1/realtime" for route in create_contract_app().routes)


def test_checked_in_openapi_is_current() -> None:
    path = files("mfqd").joinpath("protocol", "openapi.json")
    assert path.read_text(encoding="utf-8") == render_openapi()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["openapi"].startswith("3.1.")


def test_openapi_references_resolve() -> None:
    schema = build_openapi_schema()
    references: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str):
                references.add(reference)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(schema)
    for reference in references:
        assert reference.startswith("#/components/schemas/")
        name = reference.rsplit("/", 1)[-1]
        assert name in schema["components"]["schemas"]


def test_runtime_proto_contains_versioned_envelope_and_shared_memory() -> None:
    source = files("mfqd").joinpath("protocol", "runtime.proto").read_text(encoding="utf-8")
    assert "package mfqd.v1;" in source
    assert "message RuntimeEnvelope" in source
    assert "message RuntimeIdentity" in source
    assert "message SharedMemoryRef" in source
    assert "message PushAudioRequest" in source
    assert "message CommitAudioRequest" in source
    assert "message ResponseReasoningDelta" in source
    assert "message ResponseToolCallDelta" in source
    assert "message ResponseCompleted" in source
    assert "message TokenUsage" in source
    assert "oneof frame" in source
    assert "oneof payload" in source
