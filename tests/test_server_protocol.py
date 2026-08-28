from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.resources import files
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from mfq.server.api import create_app, create_contract_app
from mfq.server.models import (
    PROTOCOL_VERSION,
    ContentPart,
    CreateResponseRequest,
    InputAudioDelta,
    RealtimeFrame,
    ResponseTextDelta,
    SessionState,
    SessionStateChanged,
)
from mfq.server.openapi import REALTIME_EVENTS, build_openapi_schema, render_openapi

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


def test_response_request_validates_tool_choice_and_json_schema() -> None:
    request = CreateResponseRequest.model_validate(
        {
            "request_id": str(REQUEST_ID),
            "expected_revision": 0,
            "input": [{"type": "text", "text": "lookup"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
        }
    )
    assert request.tools[0].function.name == "lookup"
    assert request.response_format.type == "json_schema"
    with pytest.raises(ValidationError):
        CreateResponseRequest.model_validate(
            {
                "request_id": str(REQUEST_ID),
                "expected_revision": 0,
                "input": [{"type": "text", "text": "lookup"}],
                "tools": [],
                "tool_choice": "required",
            }
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
        "/api/v1/sessions/import",
        "/api/v1/jobs",
        "/api/v1/jobs/kinds",
        "/api/v1/jobs/completed",
        "/api/v1/jobs/{job_id}",
        "/api/v1/jobs/{job_id}/cancel",
        "/api/v1/jobs/{job_id}/retry",
        "/api/v1/jobs/{job_id}/events",
        "/api/v1/jobs/{job_id}/events/stream",
        "/api/v1/presets",
        "/api/v1/presets/{preset_id}",
        "/api/v1/sessions/{session_id}/messages",
        "/api/v1/sessions/{session_id}/export",
        "/api/v1/sessions/{session_id}/responses",
        "/api/v1/sessions/{session_id}/responses/cancel",
        "/api/v1/sessions/{session_id}/fork",
        "/api/v1/sessions/{session_id}/rewind",
        "/api/v1/sessions/{session_id}",
        "/api/v1/media",
        "/api/v1/media/{media_id}",
        "/api/v1/documents",
        "/api/v1/documents/{media_id}",
        "/api/v1/datasets",
        "/api/v1/datasets/{dataset_id}",
        "/api/v1/evaluations",
        "/api/v1/evaluations/compare",
        "/api/v1/cluster/nodes",
        "/api/v1/cluster/nodes/{node_id}",
        "/api/v1/artifacts/lineage",
        "/api/v1/artifacts/remove",
        "/api/v1/auth/keys",
        "/api/v1/auth/keys/{key_id}/revoke",
        "/api/v1/auth/keys/{key_id}/rotate",
        "/api/v1/mcp/servers",
        "/api/v1/mcp/servers/{server_id}",
        "/api/v1/mcp/tools",
        "/api/v1/mcp/tools/call",
        "/api/v1/models",
        "/api/v1/models/directories",
        "/api/v1/models/directories/register",
        "/api/v1/models/load",
        "/api/v1/models/unload",
        "/api/v1/hub/models",
        "/api/v1/hub/models/{provider}/{owner}/{name}",
        "/api/v1/runtime/instances",
        "/api/v1/runtime/logs",
        "/api/v1/runtime/metrics",
        "/api/v1/runtime/capabilities",
        "/api/v1/runtime/status",
        "/api/v1/runtime/models",
        "/api/v1/runtime/realtime/capabilities",
        "/api/v1/components/voice-output",
        "/api/v1/components/voice-output/install",
        "/api/v1/components/voice-output/activate",
        "/api/v1/runtime/reload",
        "/api/v1/runtime/cache/clear",
        "/api/v1/runtime/profiles",
        "/api/v1/runtime/profiles/{profile_id}",
        "/api/v1/runtime/profiles/{profile_id}/load",
    }
    assert set(schema["paths"]) == expected_paths
    assert schema["x-mfq-protocol-version"] == PROTOCOL_VERSION
    assert schema["x-mfq-websocket"]["events"] == REALTIME_EVENTS
    assert any(route.path == "/api/v1/realtime" for route in create_contract_app().routes)
    assert any(route.path == "/api/v1/runtime/realtime" for route in create_contract_app().routes)


def test_checked_in_openapi_is_current() -> None:
    path = files("mfq.server").joinpath("protocol", "openapi.json")
    assert path.read_text(encoding="utf-8") == render_openapi()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["openapi"].startswith("3.1.")


def test_optional_api_auth_and_security_headers() -> None:
    import asyncio

    import httpx

    async def scenario() -> None:
        app = create_app(api_key="unit-key")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            denied = await client.get("/api/v1/sessions")
            assert denied.status_code == 401
            allowed = await client.get(
                "/api/v1/sessions", headers={"Authorization": "Bearer unit-key"}
            )
            assert allowed.status_code == 501
            assert allowed.headers["x-content-type-options"] == "nosniff"
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.headers["x-frame-options"] == "DENY"

    asyncio.run(scenario())


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
    source = files("mfq.server").joinpath("protocol", "runtime.proto").read_text(encoding="utf-8")
    assert "package mfq.server.v1;" in source
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
