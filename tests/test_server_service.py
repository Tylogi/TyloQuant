from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.backend import BackendDelta, BackendError, BackendToolCallDelta
from mfq.server.capabilities import capabilities_for_architecture
from mfq.server.models import (
    CreateResponseRequest,
    CreateSessionRequest,
    MessageRole,
    ResponseStatus,
    RewindSessionRequest,
    RuntimeCapabilitiesResource,
    SamplingParams,
    SessionState,
    TokenUsage,
)
from mfq.server.service import ServerService, ServiceError
from mfq.server.storage import SessionStore

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
        self.cancelled_sessions: list[UUID] = []
        self.cache_clears = 0
        self.closed = False

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        response_format: Any = None,
    ) -> AsyncIterator[BackendDelta]:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "sampling": sampling,
                "session_id": session_id,
                "tools": tools,
                "tool_choice": tool_choice,
                "response_format": response_format,
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

    async def cancel_response(self, session_id: UUID) -> bool:
        self.cancelled_sessions.append(session_id)
        return True

    async def capabilities(self) -> RuntimeCapabilitiesResource:
        return RuntimeCapabilitiesResource(
            model="model-a",
            model_type="minicpmo",
            model_capabilities=capabilities_for_architecture("minicpmo"),
            duplex_available=True,
        )

    async def runtime_status(self) -> dict[str, object]:
        return {"model": "model-a", "max_context": 8192, "active_requests": 0}

    async def runtime_models(self) -> dict[str, object]:
        return {"object": "list", "data": [{"id": "model-a"}]}

    async def realtime_capabilities(self) -> dict[str, object]:
        return {"available": True, "input_sample_rate": 16000}

    async def reload_runtime(self, context_size: int) -> dict[str, object]:
        return {"model": "model-a", "max_context": context_size}

    async def clear_runtime_cache(self) -> dict[str, object]:
        self.cache_clears += 1
        return {
            "status": "ok",
            "released_snapshots": 3,
            "prefix_cache_snapshots": 0,
            "prefix_cache_bytes": 0,
        }


def make_service(path: Path, backend: FakeBackend) -> ServerService:
    return ServerService(SessionStore(path / "mfq.server.sqlite3"), backend)


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


def test_explicit_cancel_stops_a_response_and_allows_immediate_edited_retry(
    tmp_path: Path,
) -> None:
    class BlockingBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.block = True
            self.started = asyncio.Event()

        async def stream(self, **kwargs: Any) -> AsyncIterator[BackendDelta]:
            self.calls.append(kwargs)
            if self.block:
                self.started.set()
                await asyncio.Event().wait()
            yield BackendDelta(content_delta="edited answer", finish_reason="stop")

    async def run() -> None:
        backend = BlockingBackend()
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
                input=[{"type": "text", "text": "original question"}],
                stream=True,
            ),
        )

        async def consume() -> None:
            async for _ in service.stream_response(prepared):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(backend.started.wait(), timeout=1)
        cancelled = await service.cancel_response(SESSION_ID)
        assert cancelled.status == ResponseStatus.CANCELLED
        with suppress(asyncio.CancelledError):
            await consumer
        assert backend.cancelled_sessions == [SESSION_ID]
        session = await service.get_session(SESSION_ID)
        assert session.state == SessionState.INTERRUPTED
        assert session.revision == 1

        original = (await service.list_messages(SESSION_ID)).data[0]
        rewound = await service.rewind_session(
            SESSION_ID,
            RewindSessionRequest(
                expected_revision=1,
                at_message_id=original.id,
                include_message=False,
            ),
        )
        backend.block = False
        retry = await service.prepare_response(
            SESSION_ID,
            CreateResponseRequest(
                request_id=UUID("33333333-3333-4333-8333-333333333333"),
                expected_revision=rewound.revision,
                input=[{"type": "text", "text": "edited question"}],
                stream=False,
            ),
        )
        completed = await service.collect_response(retry)
        assert completed.status == ResponseStatus.COMPLETED
        assert completed.output[0].text == "edited answer"

    asyncio.run(run())


def test_media_upload_retrieval_and_multimodal_forwarding(tmp_path: Path) -> None:
    async def run() -> None:
        backend = FakeBackend((BackendDelta(content_delta="seen", finish_reason="stop"),))
        service = make_service(tmp_path, backend)
        data = b"\x89PNG\r\nmedia"
        digest = hashlib.sha256(data).hexdigest()
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            uploaded = await client.post(
                "/api/v1/media",
                content=data,
                headers={"Content-Type": "image/png", "X-Content-SHA256": digest},
            )
            assert uploaded.status_code == 201
            media = uploaded.json()["media"]
            fetched = await client.get(f"/api/v1/media/{media['id']}")
            assert fetched.status_code == 200
            assert fetched.content == data
            assert fetched.headers["content-type"] == "image/png"

        await asyncio.to_thread(
            service.store.create_session,
            CreateSessionRequest(model="model-a"),
            session_id=SESSION_ID,
        )
        request = CreateResponseRequest(
            request_id=REQUEST_ID,
            expected_revision=0,
            input=[{"type": "image", "media": media, "width": 1, "height": 1}],
            stream=False,
        )
        prepared = await service.prepare_response(SESSION_ID, request)
        content = prepared.backend_messages[-1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        await service.collect_response(prepared)

    asyncio.run(run())


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


def test_service_applies_console_context_and_sampling_options(tmp_path: Path) -> None:
    async def run() -> None:
        backend = FakeBackend(
            (BackendDelta(content_delta="ok"), BackendDelta(finish_reason="stop"))
        )
        service = make_service(tmp_path, backend)
        await asyncio.to_thread(
            service.store.create_session,
            CreateSessionRequest(model="model-a"),
            session_id=SESSION_ID,
        )
        await asyncio.to_thread(
            service.store.append_message,
            SESSION_ID,
            0,
            MessageRole.USER,
            [{"type": "text", "text": "first"}],
        )
        await asyncio.to_thread(
            service.store.append_message,
            SESSION_ID,
            1,
            MessageRole.ASSISTANT,
            [
                {"type": "reasoning", "text": "private chain"},
                {"type": "text", "text": "first answer"},
            ],
        )
        request = CreateResponseRequest(
            request_id=REQUEST_ID,
            expected_revision=2,
            input=[{"type": "text", "text": "second question"}],
            sampling=SamplingParams(enable_thinking=False),
            system_prompt="Follow the test instruction.",
            include_reasoning_history=False,
            stream=False,
        )
        prepared = await service.prepare_response(SESSION_ID, request)
        await service.collect_response(prepared)
        assert backend.calls[0]["messages"] == [
            {"role": "system", "content": "Follow the test instruction."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
        assert backend.calls[0]["sampling"].enable_thinking is False
        assert service.store.get_session(SESSION_ID).title == "second question"

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
            health = await client.get("/health")
            assert health.json() == {
                "status": "ok",
                "service": "mfq-server",
                "protocol_version": "1.0",
            }
            preflight = await client.options(
                "/api/v1/sessions",
                headers={
                    "Origin": "tauri://localhost",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            assert preflight.status_code == 200
            assert preflight.headers["access-control-allow-origin"] == "tauri://localhost"
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
            renamed = await client.patch(
                f"/api/v1/sessions/{session['id']}",
                json={"title": "renamed"},
            )
            assert renamed.json()["title"] == "renamed"
            edited_branch = await client.post(
                f"/api/v1/sessions/{session['id']}/fork",
                json={
                    "at_message_id": messages.json()["data"][0]["id"],
                    "include_message": False,
                    "title": "edited",
                },
            )
            assert edited_branch.status_code == 201
            assert edited_branch.json()["revision"] == 0
            appended = await client.post(
                f"/api/v1/sessions/{edited_branch.json()['id']}/messages",
                json={
                    "expected_revision": 0,
                    "role": "user",
                    "parts": [{"type": "text", "text": "edited question"}],
                },
            )
            assert appended.status_code == 201
            assert appended.json()["session"]["revision"] == 1
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
            status = await client.get("/api/v1/runtime/status")
            assert status.json()["max_context"] == 8192
            models = await client.get("/api/v1/runtime/models")
            assert models.json()["data"] == [{"id": "model-a"}]
            realtime = await client.get("/api/v1/runtime/realtime/capabilities")
            assert realtime.json()["available"] is True
            reloaded = await client.post(
                "/api/v1/runtime/reload",
                json={"context_size": 16384},
            )
            assert reloaded.json()["max_context"] == 16384
            cleared = await client.post("/api/v1/runtime/cache/clear")
            assert cleared.status_code == 200
            assert cleared.json()["released_snapshots"] == 3
            assert backend.cache_clears == 1
            invalid = await client.post("/api/v1/sessions", json={"model": ""})
            assert invalid.status_code == 422
            assert invalid.json()["error"]["code"] == "invalid_request"
            forked = await client.post(
                f"/api/v1/sessions/{session['id']}/fork",
                json={"title": "branch"},
            )
            assert forked.status_code == 201
            forked_session = forked.json()
            assert backend.forks == [(UUID(session["id"]), UUID(forked_session["id"]))]
            rewound = await client.post(
                f"/api/v1/sessions/{session['id']}/rewind",
                json={
                    "expected_revision": 2,
                    "at_message_id": messages.json()["data"][0]["id"],
                    "include_message": False,
                },
            )
            assert rewound.status_code == 200
            assert rewound.json()["id"] == session["id"]
            assert rewound.json()["revision"] == 0
            assert (await client.get(f"/api/v1/sessions/{session['id']}/messages")).json() == {
                "data": []
            }
            assert (await client.get(f"/api/v1/sessions/{session['id']}/responses")).json() == {
                "data": []
            }
            assert backend.forks == [(UUID(session["id"]), UUID(forked_session["id"]))]
            deleted = await client.delete(f"/api/v1/sessions/{session['id']}")
            assert deleted.status_code == 204
            assert backend.closed_sessions == [UUID(session["id"])]
            deleted_fork = await client.delete(f"/api/v1/sessions/{forked_session['id']}")
            assert deleted_fork.status_code == 204
            assert backend.closed_sessions == [
                UUID(session["id"]),
                UUID(forked_session["id"]),
            ]
            deleted_edited = await client.delete(f"/api/v1/sessions/{edited_branch.json()['id']}")
            assert deleted_edited.status_code == 204

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
