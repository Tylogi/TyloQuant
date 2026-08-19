from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from uuid import UUID

import httpx

from mfq.server.api import create_app
from mfq.server.cluster import ClusterBackend
from mfq.server.service import ServerService
from mfq.server.storage import SessionStore
from tests.test_server_service import FakeBackend


def _remote_app():
    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"service": "mfq-server"})
        if path == "/api/v1/runtime/models":
            return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
        if path == "/api/v1/runtime/status":
            return httpx.Response(200, json={"total_requests": 7, "process_resident_bytes": 1024})
        if path == "/api/v1/media":
            assert request.headers["content-type"] == "image/png"
            assert request.content == b"image"
            return httpx.Response(
                201,
                json={
                    "media": {
                        "id": "66666666-6666-4666-8666-666666666666",
                        "sha256": hashlib.sha256(b"image").hexdigest(),
                        "mime_type": "image/png",
                        "byte_size": 5,
                    },
                    "created_at": "2026-08-12T00:00:00Z",
                },
            )
        if path == "/api/v1/sessions":
            return httpx.Response(
                201,
                json={
                    "id": "11111111-1111-4111-8111-111111111111",
                    "model": "remote-model",
                    "mode": "text",
                    "state": "idle",
                    "revision": 0,
                    "title": None,
                    "runtime_instance_id": None,
                    "created_at": "2026-08-12T00:00:00Z",
                    "updated_at": "2026-08-12T00:00:00Z",
                    "metadata": {},
                },
            )
        if path.endswith("/responses"):
            frames = [
                {
                    "protocol_version": "1.0",
                    "session_id": "11111111-1111-4111-8111-111111111111",
                    "sequence": 0,
                    "timestamp": "2026-08-12T00:00:00Z",
                    "payload": {
                        "type": "response.text.delta",
                        "response_id": "22222222-2222-4222-8222-222222222222",
                        "delta": "remote",
                    },
                },
                {
                    "protocol_version": "1.0",
                    "session_id": "11111111-1111-4111-8111-111111111111",
                    "sequence": 1,
                    "timestamp": "2026-08-12T00:00:00Z",
                    "payload": {
                        "type": "response.completed",
                        "response_id": "22222222-2222-4222-8222-222222222222",
                        "finish_reason": "stop",
                    },
                },
                {
                    "protocol_version": "1.0",
                    "session_id": "11111111-1111-4111-8111-111111111111",
                    "sequence": 2,
                    "timestamp": "2026-08-12T00:00:00Z",
                    "payload": {"type": "session.state", "state": "idle", "revision": 2},
                },
            ]
            content = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
            return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
        if path.endswith("/fork"):
            return httpx.Response(
                201,
                json={
                    "id": "44444444-4444-4444-8444-444444444444",
                    "model": "remote-model",
                    "mode": "text",
                    "state": "idle",
                    "revision": 2,
                    "title": None,
                    "runtime_instance_id": None,
                    "created_at": "2026-08-12T00:00:00Z",
                    "updated_at": "2026-08-12T00:00:00Z",
                    "metadata": {},
                },
            )
        if request.method == "DELETE" and "/api/v1/sessions/" in path:
            return httpx.Response(204)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_cluster_registers_probes_and_routes_matching_model(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        client = httpx.AsyncClient(transport=_remote_app())
        local = FakeBackend()
        cluster = ClusterBackend(local, store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "worker-a", "url": "http://worker-a:8090"},
            )
            assert created.status_code == 201
            assert created.json()["healthy"] is True
            assert created.json()["models"] == ["remote-model"]
            node_id = created.json()["id"]
            listed = await api.get("/api/v1/cluster/nodes?refresh=true")
            assert listed.json()["data"][0]["healthy"] is True
            assert listed.json()["data"][0]["metrics"]["total_requests"] == 7
            models = await cluster.runtime_models()
            assert any(item["id"] == "remote-model" for item in models["data"])

            chunks = []
            async for delta in cluster.stream(
                model="remote-model",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(b"image").decode("ascii")
                                },
                            }
                        ],
                    }
                ],
                sampling=__import__(
                    "mfq.server.models", fromlist=["SamplingParams"]
                ).SamplingParams(),
                session_id=UUID("33333333-3333-4333-8333-333333333333"),
            ):
                chunks.append(delta)
            assert "".join(item.content_delta for item in chunks) == "remote"
            assert chunks[-1].finish_reason == "stop"
            state = cluster._states[UUID(node_id)]
            tool_result = await cluster._message_parts(
                state.resource,
                {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                {},
            )
            assert tool_result == [
                {
                    "type": "tool_result",
                    "call_id": "call-1",
                    "result": "done",
                    "is_error": False,
                }
            ]
            tool_call = await cluster._message_parts(
                state.resource,
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "checking",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                        }
                    ],
                },
                {},
            )
            assert [part["type"] for part in tool_call] == ["reasoning", "tool_call"]
            assert tool_call[1]["arguments"] == {"q": "x"}
            status = await cluster.runtime_status()
            assert status["cluster_total_requests"] == 7
            assert status["cluster_process_resident_bytes"] == 1024
            forked_id = UUID("55555555-5555-4555-8555-555555555555")
            assert await cluster.fork_session(
                UUID("33333333-3333-4333-8333-333333333333"), forked_id
            )
            assert await cluster.close_session(forked_id)

            assert (await api.delete(f"/api/v1/cluster/nodes/{node_id}")).status_code == 204

        await client.aclose()

    asyncio.run(run())


def test_remote_node_configuration_never_persists_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REMOTE_NODE_TOKEN", "private-token")
    store = SessionStore(tmp_path / "mfq.server.sqlite3")
    node = store.create_remote_node(
        __import__(
            "mfq.server.models", fromlist=["CreateRemoteNodeRequest"]
        ).CreateRemoteNodeRequest(
            name="secure", url="https://worker.example", api_key_env="REMOTE_NODE_TOKEN"
        )
    )
    assert node.api_key_env == "REMOTE_NODE_TOKEN"
    assert b"private-token" not in (tmp_path / "mfq.server.sqlite3").read_bytes()
