from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path

import httpx

from mfq.server.api import create_app
from mfq.server.service import ServerService
from mfq.server.storage import SessionStore
from tests.test_server_service import FakeBackend


def test_session_archive_round_trips_messages_media_and_documents(tmp_path: Path) -> None:
    async def run() -> None:
        service = ServerService(SessionStore(tmp_path / "mfq.server.sqlite3"), FakeBackend())
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            session = (
                await client.post(
                    "/api/v1/sessions",
                    json={"model": "model-a", "mode": "text", "title": "portable"},
                )
            ).json()
            raw = b"portable document"
            digest = hashlib.sha256(raw).hexdigest()
            media = (
                await client.post(
                    "/api/v1/media",
                    content=raw,
                    headers={"Content-Type": "text/plain", "X-Content-SHA256": digest},
                )
            ).json()["media"]
            document = (
                await client.post(
                    "/api/v1/documents",
                    json={"media_id": media["id"], "name": "note.txt"},
                )
            ).json()
            appended = await client.post(
                f"/api/v1/sessions/{session['id']}/messages",
                json={
                    "expected_revision": 0,
                    "role": "user",
                    "parts": [{"type": "document", "media": media, "name": "note.txt"}],
                },
            )
            assert appended.status_code == 201
            archive = (await client.get(f"/api/v1/sessions/{session['id']}/export")).json()
            assert archive["format"] == "mfq-session-v1"
            assert base64.b64decode(archive["media"][0]["data_base64"]) == raw
            assert archive["media"][0]["document"]["text"] == document["text"]

            imported = await client.post("/api/v1/sessions/import", json=archive)
            assert imported.status_code == 201, imported.text
            result = imported.json()
            assert result["session"]["id"] != session["id"]
            assert result["messages_imported"] == 1
            messages = (
                await client.get(f"/api/v1/sessions/{result['session']['id']}/messages")
            ).json()["data"]
            assert messages[0]["parts"][0]["media"]["sha256"] == digest

    asyncio.run(run())
