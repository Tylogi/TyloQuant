from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import numpy as np

from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.server.api import create_app
from mfq.server.catalog import ModelCatalog
from mfq.server.service import ServerService
from mfq.server.storage import SCHEMA_VERSION, SessionStore
from tests.test_server_service import FakeBackend


def _model(path: Path) -> None:
    save(
        path,
        FileHeader(version=2, model_arch="qwen35"),
        {"weight": np.arange(16, dtype=np.float16).reshape(4, 4)},
    )


def test_runtime_profiles_persist_and_detect_artifact_drift(tmp_path: Path) -> None:
    async def run() -> None:
        model_path = tmp_path / "model.mfq"
        _model(model_path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = (await catalog.list()).data[0]
        database = tmp_path / "mfq.server.sqlite3"
        service = ServerService(
            SessionStore(database),
            FakeBackend(),
            catalog=catalog,
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/v1/runtime/profiles",
                json={
                    "name": "local",
                    "load": {
                        "model": artifact.name,
                        "context_size": 65536,
                        "prefill_chunk_size": 4096,
                        "prefix_cache_max_sessions": 8,
                        "prefix_cache_max_bytes": 1073741824,
                        "sampling_defaults": {
                            "temperature": 0.7,
                            "top_p": 0.8,
                            "repetition_penalty": 1.05,
                        },
                    },
                },
            )
            assert created.status_code == 201, created.text
            profile = created.json()
            assert profile["load"]["context_size"] == 65536
            assert profile["load"]["sampling_defaults"]["repetition_penalty"] == 1.05
            assert not profile["drifted"]

            listed = await client.get("/api/v1/runtime/profiles")
            assert listed.json()["data"][0]["name"] == "local"
        await service.aclose()

        reopened = ServerService(SessionStore(database), FakeBackend(), catalog=catalog)
        assert (await reopened.list_runtime_profiles()).data[0].name == "local"

        model_path.unlink()
        missing = await reopened.get_runtime_profile(profile["id"])
        assert missing.drifted
        assert missing.drift_reason
        await reopened.aclose()

    asyncio.run(run())


def test_runtime_profile_schema_nine_migrates(tmp_path: Path) -> None:
    database = tmp_path / "mfq.server.sqlite3"
    store = SessionStore(database)
    with store._connection() as connection:
        connection.execute("UPDATE schema_meta SET value = '9' WHERE key = 'schema_version'")
    migrated = SessionStore(database)
    with migrated._connection() as connection:
        version = connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()["value"]
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'runtime_profiles'"
        ).fetchone()
    assert version == str(SCHEMA_VERSION)
    assert table is not None
