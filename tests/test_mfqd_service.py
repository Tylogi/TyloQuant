from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest

from mfqd.api import create_app
from mfqd.backend import BackendDelta, BackendError, BackendToolCallDelta
from mfqd.capabilities import capabilities_for_architecture
from mfqd.models import (
    CreateResponseRequest,
    CreateSessionRequest,
    ResponseStatus,
    RuntimeCapabilitiesResource,
    SamplingParams,
    SessionState,
    TokenUsage,
)
from mfqd.service import MfqdService, ServiceError
from mfqd.storage import SessionStore

SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
REQUEST_ID = UUID("22222222-2222-4222-8222-222222222222")


class FakeBackend:
    def __init__(
        self,
        deltas: Sequence[BackendDelta] = (),
        *,
        error: BackendError | None = None,
    ) -> None:
        self.deltas = tuple(deltas)
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.forks: list[tuple[UUID, UUID]] = []
        self.closed_sessions: list[UUID] = []
        self.closed = False

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
    ) -> AsyncIterator[BackendDelta]:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "sampling": sampling,
                "session_id": session_id,
            }
        )
        if self.error is not None:
            raise self.error
        for delta in self.deltas:
            yield delta

    async def aclose(self) -> None:
        self.closed = True

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool:
        self.forks.append((source_session_id, target_session_id))
        return True

    async def close_session(self, session_id: UUID) -> bool:
        self.closed_sessions.append(session_id)
        return True

    async def capabilities(self) -> RuntimeCapabilitiesResource:
        return RuntimeCapabilitiesResource(
            model="model-a",
            model_type="minicpmo",
            model_capabilities=capabilities_for_architecture("minicpmo"),
            duplex_available=True,
        )


def make_service(path: Path, backend: FakeBackend) -> MfqdService:
    return MfqdService(SessionStore(path / "mfqd.sqlite3"), backend)


def completed_deltas() -> tuple[BackendDelta, ...]:
    return (
        BackendDelta(reasoning_delta="reason "),
        BackendDelta(content_delta="answer"),
        BackendDelta(
            tool_calls=(
                BackendToolCallDelta(
                    index=0,
                    call_id="call-1",
                    name="lookup",
                    arguments_delta='{"q":',
                ),
            )
        ),
        BackendDelta(
            tool_calls=(BackendToolCallDelta(index=0, arguments_delta='"mfq"}'),),
            finish_reason="tool_calls",
        ),
        BackendDelta(usage=TokenUsage(prompt_tokens=4, completion_tokens=5, total_tokens=9)),
    )


def test_service_collects_and_replays_a_persisted_response(tmp_path: Path) -> None:
    async def run() -> None:
        backend = FakeBackend(completed_deltas())
        service = make_service(tmp_path, backend)
        await asyncio.to_thread(
            service.store.create_session,
            CreateSessionRequest(model="model-a"),
            session_id=SESSION_ID,
        )
        request = CreateResponseRequest(
            request_id=REQUEST_ID,
            expected_revision=0,
            input=[{"type": "text", "text": "question"}],
            stream=False,
        )
        prepared = await service.prepare_response(SESSION_ID, request)
        completed = await service.collect_response(prepared)
        assert completed.status == ResponseStatus.COMPLETED
        assert [part.type for part in completed.output] == ["reasoning", "text", "tool_call"]
        assert completed.finish_reason == "tool_calls"
        assert completed.usage is not None and completed.usage.total_tokens == 9
        assert backend.calls[0]["messages"] == [{"role": "user", "content": "question"}]
        assert backend.calls[0]["session_id"] == SESSION_ID
        assert service.store.get_session(SESSION_ID).revision == 2

        replay_request = request.model_copy(update={"stream": True})
        replay = await service.prepare_response(SESSION_ID, replay_request)
        frames = "".join([frame async for frame in service.stream_response(replay)])
        payloads = [
            json.loads(line[6:])["payload"]
            for line in frames.splitlines()
            if line.startswith("data: ")
        ]
        assert [payload["type"] for payload in payloads] == [
            "response.reasoning.delta",
            "response.text.delta",
            "response.tool_call.delta",
            "response.completed",
            "session.state",
        ]
        assert len(backend.calls) == 1

    asyncio.run(run())


def test_service_failure_is_persisted_and_reported(tmp_path: Path) -> None:
    async def run() -> None:
        backend = FakeBackend(error=BackendError("backend_timeout", "timed out", retryable=True))
        service = make_service(tmp_path, backend)
        await asyncio.to_thread(
            service.store.create_session,
            CreateSessionRequest(model="model-a"),
            session_id=SESSION_ID,
        )
        prepared = await service.prepare_response(
            SESSION_ID,
            CreateResponseRequest(
                request_id=REQUEST_ID,
                expected_revision=0,
                input=[{"type": "text", "text": "question"}],
                stream=False,
            ),
        )
        with pytest.raises(ServiceError) as caught:
            await service.collect_response(prepared)
        assert caught.value.status_code == 502
        assert caught.value.detail.code == "backend_timeout"
        stored = service.store.get_response(prepared.begin.response.id)
        assert stored.status == ResponseStatus.FAILED
        assert service.store.get_session(SESSION_ID).state == SessionState.ERROR

    asyncio.run(run())


def test_executable_api_persists_nonstream_text_responses(tmp_path: Path) -> None:
    async def run() -> None:
        backend = FakeBackend(
            (
                BackendDelta(content_delta="hello"),
                BackendDelta(finish_reason="stop"),
            )
        )
        service = make_service(tmp_path, backend)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/v1/sessions",
                json={"model": "model-a", "mode": "text"},
            )
            assert created.status_code == 201
            session = created.json()
            listed = await client.get("/api/v1/sessions")
            assert listed.json()["data"][0]["id"] == session["id"]
            fetched = await client.get(f"/api/v1/sessions/{session['id']}")
            assert fetched.json()["revision"] == 0
            generated = await client.post(
                f"/api/v1/sessions/{session['id']}/responses",
                json={
                    "request_id": str(REQUEST_ID),
                    "expected_revision": 0,
                    "input": [{"type": "text", "text": "hi"}],
                    "stream": False,
                },
            )
            assert generated.status_code == 200
            assert generated.json()["output"] == [{"type": "text", "text": "hello"}]
            messages = await client.get(f"/api/v1/sessions/{session['id']}/messages")
            assert [message["role"] for message in messages.json()["data"]] == [
                "user",
                "assistant",
            ]
            runtimes = await client.get("/api/v1/runtime/instances")
            assert runtimes.json() == {"data": []}
            capabilities = await client.get("/api/v1/runtime/capabilities")
            assert capabilities.json()["model_capabilities"]["features"] == {
                "text": True,
                "image_input": True,
                "video_input": True,
                "audio_input": True,
                "audio_output": True,
                "full_duplex": True,
            }
            assert capabilities.json()["duplex_available"] is True
            invalid = await client.post("/api/v1/sessions", json={"model": ""})
            assert invalid.status_code == 422
            assert invalid.json()["error"]["code"] == "invalid_request"
            forked = await client.post(
                f"/api/v1/sessions/{session['id']}/fork",
                json={"title": "branch"},
            )
            assert forked.status_code == 201
            forked_session = forked.json()
            assert backend.forks == [
                (UUID(session["id"]), UUID(forked_session["id"]))
            ]
            deleted = await client.delete(f"/api/v1/sessions/{session['id']}")
            assert deleted.status_code == 204
            assert backend.closed_sessions == [UUID(session["id"])]
            deleted_fork = await client.delete(
                f"/api/v1/sessions/{forked_session['id']}"
            )
            assert deleted_fork.status_code == 204
            assert backend.closed_sessions == [
                UUID(session["id"]),
                UUID(forked_session["id"]),
            ]

    asyncio.run(run())


def test_executable_api_can_serve_a_built_web_root(tmp_path: Path) -> None:
    async def run() -> None:
        web_root = tmp_path / "web"
        web_root.mkdir()
        (web_root / "index.html").write_text("<h1>MFQ Web</h1>", encoding="utf-8")
        service = make_service(tmp_path, FakeBackend())
        transport = httpx.ASGITransport(app=create_app(service, web_root=web_root))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            root = await client.get("/")
            assert root.status_code == 200
            assert "MFQ Web" in root.text
            sessions = await client.get("/api/v1/sessions")
            assert sessions.status_code == 200
            assert sessions.json() == {"data": []}

    asyncio.run(run())
