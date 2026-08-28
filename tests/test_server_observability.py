from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx

import mfq.server.storage as storage_module
from mfq.server.api import create_app
from mfq.server.models import RuntimeLogLevel
from mfq.server.service import ServerService
from mfq.server.storage import SessionStore

INSTANCE_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


class StatusBackend:
    async def runtime_status(self):
        return {
            "instance_id": str(INSTANCE_ID),
            "model": "model-a",
            "total_requests": 3,
            "last_request": {"id": "request-a", "decode_tps": 24.5},
        }

    async def aclose(self):
        return None


def test_runtime_metrics_and_logs_persist_and_filter(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, StatusBackend())  # type: ignore[arg-type]
        store.append_runtime_log(
            RuntimeLogLevel.INFO,
            "loaded",
            instance_id=INSTANCE_ID,
            fields={"source": "test"},
            now=NOW,
        )
        store.append_runtime_log(RuntimeLogLevel.ERROR, "failed", now=NOW)
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                status = await client.get("/api/v1/runtime/status")
                assert status.status_code == 200
                repeated_status = await client.get("/api/v1/runtime/status")
                assert repeated_status.status_code == 200
                metrics = await client.get(
                    "/api/v1/runtime/metrics", params={"instance_id": str(INSTANCE_ID)}
                )
                assert metrics.status_code == 200
                assert len(metrics.json()["data"]) == 1
                assert metrics.json()["data"][0]["values"]["total_requests"] == 3
                assert metrics.json()["data"][0]["model"] == "model-a"

                logs = await client.get(
                    "/api/v1/runtime/logs",
                    params={"instance_id": str(INSTANCE_ID), "level": "info"},
                )
                assert logs.status_code == 200
                assert [entry["message"] for entry in logs.json()["data"]] == ["loaded"]
        finally:
            await service.aclose()

        reopened = SessionStore(tmp_path / "mfq.server.sqlite3")
        assert reopened.list_runtime_metrics()[0].values["last_request"]["id"] == "request-a"
        assert len(reopened.list_runtime_logs()) == 2

    asyncio.run(run())


def test_runtime_metric_history_has_a_bounded_retention_window(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(storage_module, "_MAX_RUNTIME_METRICS", 3)
    store = SessionStore(tmp_path / "mfq.server.sqlite3")
    for value in range(5):
        store.append_runtime_metric({"value": value})

    assert [entry.values["value"] for entry in store.list_runtime_metrics()] == [2, 3, 4]
