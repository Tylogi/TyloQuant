from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import Any

import uvicorn
import websockets

from mfqd.api import create_app
from mfqd.backend import OpenAIChatBackend
from mfqd.service import MfqdService
from mfqd.storage import SessionStore


def test_runtime_realtime_websocket_is_forwarded_without_protocol_translation(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        received: list[dict[str, Any]] = []

        async def upstream(socket: Any) -> None:
            assert socket.request.path == "/v1/realtime?mode=audio"
            await socket.send(json.dumps({"type": "session.queue_done"}))
            received.append(json.loads(await socket.recv()))
            await socket.send(json.dumps({"type": "session.created", "session_id": "voice-1"}))
            received.append(json.loads(await socket.recv()))
            await socket.send(
                json.dumps(
                    {
                        "type": "response.output.delta",
                        "kind": "text",
                        "text": "hello",
                    }
                )
            )

        async with websockets.serve(upstream, "127.0.0.1", 0) as upstream_server:
            upstream_port = upstream_server.sockets[0].getsockname()[1]
            service = MfqdService(
                SessionStore(tmp_path / "mfqd.sqlite3"),
                OpenAIChatBackend(f"http://127.0.0.1:{upstream_port}"),
            )
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            server = uvicorn.Server(
                uvicorn.Config(
                    create_app(service),
                    host="127.0.0.1",
                    port=port,
                    log_level="error",
                )
            )
            task = asyncio.create_task(server.serve(sockets=[listener]))
            while not server.started:
                await asyncio.sleep(0.01)
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{port}/api/v1/runtime/realtime?mode=audio",
                    proxy=None,
                ) as client:
                    assert json.loads(await client.recv())["type"] == "session.queue_done"
                    await client.send(json.dumps({"type": "session.init", "payload": {}}))
                    assert json.loads(await client.recv())["type"] == "session.created"
                    await client.send(
                        json.dumps(
                            {
                                "type": "input.append",
                                "input": {"text": "hi"},
                            }
                        )
                    )
                    response = json.loads(await client.recv())
                    assert response["kind"] == "text"
                    assert response["text"] == "hello"
            finally:
                server.should_exit = True
                await task
            assert received == [
                {"type": "session.init", "payload": {}},
                {"type": "input.append", "input": {"text": "hi"}},
            ]

    asyncio.run(run())
