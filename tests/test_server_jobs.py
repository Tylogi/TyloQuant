from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.jobs import JobManager
from mfq.server.models import (
    CreateJobRequest,
    ErrorDetail,
    JobEventType,
    JobStatus,
)
from mfq.server.service import ServerService
from mfq.server.storage import InvalidJobStateError, JobNotFoundError, SessionStore

JOB_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


class IdleBackend:
    async def aclose(self) -> None:
        return None


def make_store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "mfq.server.sqlite3")


def test_job_storage_lifecycle_and_restart_recovery(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    created = store.create_job("test.work", {"value": 4}, job_id=JOB_ID, now=NOW)
    assert created.status == JobStatus.QUEUED
    assert created.progress == 0
    assert store.list_job_events(JOB_ID)[0].type == JobEventType.STATE

    running = store.claim_job(JOB_ID, now=NOW)
    assert running.status == JobStatus.RUNNING
    store.update_job_progress(JOB_ID, 0.4, message="working", now=NOW)
    store.update_job_progress(JOB_ID, 0.2, now=NOW)
    assert store.get_job(JOB_ID).progress == 0.4

    reopened = make_store(tmp_path)
    assert reopened.recover_interrupted_jobs(now=NOW) == [JOB_ID]
    interrupted = reopened.get_job(JOB_ID)
    assert interrupted.status == JobStatus.INTERRUPTED
    assert interrupted.error is not None and interrupted.error.retryable
    assert [event.sequence for event in reopened.list_job_events(JOB_ID)] == [1, 2, 3, 4, 5]


def test_only_terminal_job_records_can_be_deleted(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    active = store.create_job("test.active", {}, now=NOW)
    with pytest.raises(InvalidJobStateError):
        store.archive_job(active.id)

    completed = store.create_job("test.complete", {}, now=NOW)
    store.claim_job(completed.id, now=NOW)
    store.complete_job(completed.id, {"artifact": "workspace://kept.bin"}, now=NOW)
    store.record_artifact_lineage(
        artifact_uri="workspace://kept.bin",
        artifact_name="kept.bin",
        producer_job_id=completed.id,
        now=NOW,
    )
    store.archive_job(completed.id)
    with pytest.raises(JobNotFoundError):
        store.get_job(completed.id)
    assert store.list_job_events(completed.id)
    assert store.list_artifact_lineage()[0].artifact_uri == "workspace://kept.bin"
    assert store.get_job(active.id).status == JobStatus.QUEUED


def test_clear_completed_jobs_preserves_active_jobs(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    active = store.create_job("test.active", {}, now=NOW)
    completed = store.create_job("test.complete", {}, now=NOW)
    store.claim_job(completed.id, now=NOW)
    store.complete_job(completed.id, {}, now=NOW)
    cancelled = store.create_job("test.cancelled", {}, now=NOW)
    store.request_job_cancel(cancelled.id, now=NOW)

    assert store.archive_completed_jobs() == 2
    assert [job.id for job in store.list_jobs()] == [active.id]


def test_job_manager_completes_and_cancels_registered_handlers(tmp_path: Path) -> None:
    async def run() -> None:
        store = make_store(tmp_path)
        release = asyncio.Event()

        async def complete_handler(context, payload):
            await context.log("started")
            await context.progress(0.5, message="half")
            return {"doubled": payload["value"] * 2}

        async def blocking_handler(context, payload):
            del payload
            await release.wait()
            context.raise_if_cancelled()
            return {}

        manager = JobManager(
            store,
            {"test.complete": complete_handler, "test.block": blocking_handler},
        )
        completed = await manager.submit(
            CreateJobRequest(kind="test.complete", payload={"value": 6})
        )
        for _ in range(100):
            state = store.get_job(completed.id)
            if state.status == JobStatus.SUCCEEDED:
                break
            await asyncio.sleep(0.01)
        assert state.status == JobStatus.SUCCEEDED
        assert state.progress == 1
        assert state.result == {"doubled": 12}

        blocked = await manager.submit(CreateJobRequest(kind="test.block"))
        for _ in range(100):
            if store.get_job(blocked.id).status == JobStatus.RUNNING:
                break
            await asyncio.sleep(0.01)
        cancelling = await manager.cancel(blocked.id)
        assert cancelling.status == JobStatus.CANCELLING
        for _ in range(100):
            cancelled = store.get_job(blocked.id)
            if cancelled.status == JobStatus.CANCELLED:
                break
            await asyncio.sleep(0.01)
        assert cancelled.status == JobStatus.CANCELLED
        await manager.close()

    asyncio.run(run())


def test_job_api_streams_persisted_events_and_rejects_unknown_kinds(tmp_path: Path) -> None:
    async def run() -> None:
        store = make_store(tmp_path)

        async def handler(context, payload):
            await context.progress(0.25, message="quarter")
            await context.artifact(name="result", uri="artifact://result.json")
            return {"input": payload}

        jobs = JobManager(store, {"test.api": handler})
        service = ServerService(store, IdleBackend(), jobs=jobs)  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unknown = await client.post("/api/v1/jobs", json={"kind": "unknown"})
            assert unknown.status_code == 422
            response = await client.post(
                "/api/v1/jobs", json={"kind": "test.api", "payload": {"x": 1}}
            )
            assert response.status_code == 202
            job_id = response.json()["id"]
            for _ in range(100):
                current = await client.get(f"/api/v1/jobs/{job_id}")
                if current.json()["status"] == "succeeded":
                    break
                await asyncio.sleep(0.01)
            assert current.json()["result"] == {"input": {"x": 1}}
            retry = await client.post(f"/api/v1/jobs/{job_id}/retry")
            assert retry.status_code == 409
            events = await client.get(f"/api/v1/jobs/{job_id}/events")
            assert events.status_code == 200
            assert [event["sequence"] for event in events.json()["data"]] == list(
                range(1, len(events.json()["data"]) + 1)
            )
            lineage = await client.get(
                "/api/v1/artifacts/lineage",
                params={"artifact_uri": "artifact://result.json"},
            )
            assert lineage.status_code == 200
            assert lineage.json()["data"][0]["producer_job_id"] == job_id
            assert lineage.json()["data"][0]["parameters"] == {"x": 1}
            streamed = await client.get(f"/api/v1/jobs/{job_id}/events/stream")
            assert streamed.status_code == 200
            payloads = [
                json.loads(line[6:])
                for line in streamed.text.splitlines()
                if line.startswith("data: ")
            ]
            assert payloads[-1]["data"]["status"] == "succeeded"
            resumed = await client.get(
                f"/api/v1/jobs/{job_id}/events/stream",
                headers={"Last-Event-ID": str(payloads[-2]["sequence"])},
            )
            resumed_payloads = [
                json.loads(line[6:])
                for line in resumed.text.splitlines()
                if line.startswith("data: ")
            ]
            assert [event["sequence"] for event in resumed_payloads] == [payloads[-1]["sequence"]]
            deleted = await client.delete(f"/api/v1/jobs/{job_id}")
            assert deleted.status_code == 204
            assert (await client.get(f"/api/v1/jobs/{job_id}")).status_code == 404
            assert (await client.delete("/api/v1/jobs/completed")).status_code == 204
        await service.aclose()

    asyncio.run(run())


def test_queued_job_can_be_cancelled_without_execution(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_job("test.queue", {}, job_id=JOB_ID, now=NOW)
    cancelled = store.request_job_cancel(JOB_ID, now=NOW)
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.cancel_requested
    assert cancelled.completed_at == NOW
    assert store.request_job_cancel(JOB_ID) == cancelled


def test_failed_job_preserves_structured_error(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_job("test.failure", {}, job_id=JOB_ID, now=NOW)
    store.claim_job(JOB_ID, now=NOW)
    failed = store.fail_job(
        JOB_ID,
        ErrorDetail(code="bad_input", message="invalid", details={"field": "x"}),
        now=NOW,
    )
    assert failed.status == JobStatus.FAILED
    assert failed.error is not None and failed.error.details == {"field": "x"}
