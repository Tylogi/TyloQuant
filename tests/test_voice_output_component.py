from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx

import mfq.server.components as components
from mfq.server.api import create_app
from mfq.server.components import ComponentFile, VoiceOutputComponent
from mfq.server.jobs import JobManager
from mfq.server.service import ServerService
from mfq.server.storage import SessionStore


class IdleBackend:
    async def aclose(self) -> None:
        return None


def test_voice_component_requires_the_pinned_verified_manifest(
    tmp_path: Path, monkeypatch,
) -> None:
    payload = b"token2wav"
    files = (
        ComponentFile(
            "assets/token2wav/test.bin",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        ),
    )
    monkeypatch.setattr(components, "VOICE_OUTPUT_FILES", files)
    monkeypatch.setattr(components, "VOICE_OUTPUT_TOTAL_BYTES", len(payload))
    component = VoiceOutputComponent(tmp_path)
    target = component.root / files[0].path
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    assert component.ready() is False

    component._manifest.write_text(
        json.dumps(
            {
                "component": components.VOICE_OUTPUT_COMPONENT_ID,
                "revision": components.VOICE_OUTPUT_REVISION,
            }
        ),
        encoding="utf-8",
    )
    assert component.ready() is True
    assert component.status()["installed_bytes"] == len(payload)


def test_voice_component_api_exposes_status_and_fixed_install_job(tmp_path: Path) -> None:
    class FakeComponent:
        def status(self):
            return {
                "id": components.VOICE_OUTPUT_COMPONENT_ID,
                "state": "missing",
                "ready": False,
                "installed_bytes": 0,
                "total_bytes": components.VOICE_OUTPUT_TOTAL_BYTES,
            }

    async def install(_context, _payload):
        return {"ready": True}

    async def scenario() -> None:
        store = SessionStore(tmp_path / "server.sqlite3")
        jobs = JobManager(store, {"component.voice_output.install": install})
        service = ServerService(
            store,
            IdleBackend(),
            jobs=jobs,
            voice_component=FakeComponent(),
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            status = await client.get("/api/v1/components/voice-output")
            assert status.status_code == 200
            assert status.json()["total_bytes"] == 1_231_783_384
            accepted = await client.post("/api/v1/components/voice-output/install")
            assert accepted.status_code == 202
            assert accepted.json()["operation_id"]
        await jobs.close()

    asyncio.run(scenario())
