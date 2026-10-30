from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.protocol.models import (
    CreateJobRequest,
    ErrorDetail,
    JobEventType,
    JobStatus,
)
from mfq.server.services.jobs import JobManager
from mfq.server.services.service import ServerService
from mfq.server.state.storage import InvalidJobStateError, JobNotFoundError, SessionStore

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


def test_download_retry_and_restart_preserve_task_identity_and_progress(tmp_path):
    async def run():
        store = make_store(tmp_path)
        original = store.create_job('download.modelscope', {'repo_id': 'owner/model'})
        store.claim_job(original.id)
        store.update_job_progress(original.id, .45, data={'downloaded_bytes': 450})
        store.interrupt_job(original.id)
        started = asyncio.Event()
        async def handler(context, payload):
            started.set()
            await asyncio.Event().wait()
        manager = JobManager(store, {'download.modelscope': handler})
        resumed = await manager.retry(original.id)
        assert resumed.id == original.id and resumed.progress == .45 and resumed.error is None
        await started.wait()
        again = await manager.retry(original.id)
        assert again.id == original.id and len(store.list_jobs()) == 1
        await manager.close()
        saved = store.get_job(original.id)
        assert saved.status == JobStatus.QUEUED and saved.progress == .45 and saved.error is None
        assert saved.progress_data['downloaded_bytes'] == 450
        started.clear()
        restarted = JobManager(store, {'download.modelscope': handler})
        await restarted.start()
        await started.wait()
        assert store.get_job(original.id).status == JobStatus.RUNNING
        assert len(store.list_jobs()) == 1
        await restarted.close()
    asyncio.run(run())


def test_crashed_download_returns_to_queue_without_resetting_progress(tmp_path):
    store = make_store(tmp_path)
    original = store.create_job('download.huggingface', {'repo_id': 'owner/model'})
    store.claim_job(original.id)
    store.update_job_progress(original.id, .42)
    store.recover_interrupted_jobs()
    saved = store.get_job(original.id)
    assert saved.status == JobStatus.QUEUED and saved.progress == .42 and saved.error is None


@pytest.mark.parametrize("download_kinds", [
    ("download.huggingface", "download.modelscope"),
    ("dataset.download", "dataset.download"),
])
def test_downloads_do_not_block_model_loading_or_runtime_controls(tmp_path: Path, download_kinds) -> None:
    async def run() -> None:
        store = make_store(tmp_path)
        started = {name: asyncio.Event() for name in ("download0", "download1", "load", "control")}

        async def handler(context, payload):
            started[payload["name"]].set()
            await asyncio.Event().wait()

        kinds = {*download_kinds, "model.load", "runtime.memory.configure"}
        manager = JobManager(store, {kind: handler for kind in kinds}, max_concurrency=2)
        try:
            downloads = [await manager.submit(CreateJobRequest(kind=kind, payload={"name": f"download{i}"}))
                for i, kind in enumerate(download_kinds)]
            await asyncio.wait_for(asyncio.gather(started["download0"].wait(), started["download1"].wait()), 5)
            load = await manager.submit(CreateJobRequest(kind="model.load", payload={"name": "load"}))
            control = await manager.submit(CreateJobRequest(kind="runtime.memory.configure", payload={"name": "control"}))
            await asyncio.wait_for(asyncio.gather(started["load"].wait(), started["control"].wait()), 5)
            assert all(store.get_job(job.id).status == JobStatus.RUNNING for job in [*downloads, load, control])
        finally:
            await manager.close()

    asyncio.run(run())


def test_background_work_does_not_block_runtime_or_downloads(tmp_path: Path) -> None:
    async def run() -> None:
        store = make_store(tmp_path)
        started = {kind: asyncio.Event() for kind in ("quantization.fit", "download.huggingface", "model.load")}

        async def handler(context, payload):
            started[payload["kind"]].set()
            await asyncio.Event().wait()

        manager = JobManager(store, {kind: handler for kind in started}, max_concurrency=1)
        try:
            for kind in started:
                await manager.submit(CreateJobRequest(kind=kind, payload={"kind": kind}))
                await asyncio.wait_for(started[kind].wait(), 5)
        finally:
            await manager.close()

    asyncio.run(run())


@pytest.mark.parametrize("kinds", [
    ("download.huggingface", "download.modelscope", "dataset.download"),
    ("model.load", "runtime.context.configure", "runtime.memory.configure"),
    ("quantization.fit", "evaluate.wikitext2", "benchmark.inference"),
])
def test_execution_lanes_keep_limits_and_release_cancelled_slots(tmp_path: Path, kinds) -> None:
    async def run() -> None:
        store = make_store(tmp_path)
        started = [asyncio.Event() for _ in kinds]

        async def handler(context, payload):
            started[payload["index"]].set()
            await asyncio.Event().wait()

        manager = JobManager(store, {kind: handler for kind in kinds}, max_concurrency=1)
        try:
            first = await manager.submit(CreateJobRequest(kind=kinds[0], payload={"index": 0}))
            await asyncio.wait_for(started[0].wait(), 5)
            second = await manager.submit(CreateJobRequest(kind=kinds[1], payload={"index": 1}))
            third = await manager.submit(CreateJobRequest(kind=kinds[2], payload={"index": 2}))
            await manager.cancel(third.id)
            assert store.get_job(second.id).status == JobStatus.QUEUED
            assert not started[1].is_set() and not started[2].is_set()
            await manager.cancel(first.id)
            await asyncio.wait_for(started[1].wait(), 5)
            assert store.get_job(second.id).status == JobStatus.RUNNING
            assert not started[2].is_set()
        finally:
            await manager.close()
        assert store.get_job(first.id).status == JobStatus.CANCELLED
        assert store.get_job(third.id).status == JobStatus.CANCELLED

    asyncio.run(run())


def test_restart_resumes_downloads_without_blocking_queued_model_load(tmp_path: Path) -> None:
    async def run() -> None:
        store = make_store(tmp_path)
        downloads = [store.create_job(kind, {}) for kind in ("download.huggingface", "download.modelscope")]
        for job in downloads:
            store.claim_job(job.id)
            store.update_job_progress(job.id, .4, data={"downloaded_bytes": 400})
        load = store.create_job("model.load", {})
        started = {job.id: asyncio.Event() for job in [*downloads, load]}

        async def handler(context, payload):
            started[context.job_id].set()
            await asyncio.Event().wait()

        manager = JobManager(store, {kind: handler for kind in ("download.huggingface", "download.modelscope", "model.load")})
        try:
            await manager.start()
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 5)
            for job in downloads:
                saved = store.get_job(job.id)
                assert saved.status == JobStatus.RUNNING and saved.progress == .4
                assert saved.progress_data["downloaded_bytes"] == 400
            assert store.get_job(load.id).status == JobStatus.RUNNING
        finally:
            await manager.close()
        assert len(store.list_jobs()) == 3
        assert all(store.get_job(job.id).status == JobStatus.QUEUED for job in downloads)

    asyncio.run(run())


def test_job_event_tail_keeps_latest_output_and_preserves_forward_replay(tmp_path: Path) -> None:
    async def run() -> None:
        store = make_store(tmp_path)
        original = store.create_job("test.work", {}, job_id=JOB_ID, now=NOW)
        last = None
        for index in range(1005):
            last = store.append_job_event(JOB_ID, JobEventType.LOG, message=f"output-{index}")
        assert store.get_job(JOB_ID).updated_at == original.updated_at
        service = ServerService(store, IdleBackend())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                head = await client.get(f"/api/v1/jobs/{JOB_ID}/events?limit=1000")
                assert head.status_code == 200 and head.json()["data"][0]["sequence"] == 1
                tail = await client.get(f"/api/v1/jobs/{JOB_ID}/events?limit=1000&tail=true")
                assert tail.status_code == 200
                entries = tail.json()["data"]
                assert len(entries) == 1000 and entries[-1]["message"] == "output-1004"
                assert [item["sequence"] for item in entries] == list(range(last.sequence - 999, last.sequence + 1))
                following = await client.get(f"/api/v1/jobs/{JOB_ID}/events?after={last.sequence - 1}&tail=true")
                assert [item["sequence"] for item in following.json()["data"]] == [last.sequence]
                assert (await client.get(f"/api/v1/jobs/{JOB_ID}/events?after={last.sequence}&tail=true")).json()["data"] == []
                assert (await client.get(f"/api/v1/jobs/{JOB_ID}/events?tail=invalid")).status_code == 422
                assert (await client.get(f"/api/v1/jobs/{JOB_ID}/events?after=-1&tail=true")).status_code == 422
        finally:
            await service.aclose()

    asyncio.run(run())


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


@pytest.mark.parametrize("status", [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED])
def test_terminal_job_event_stream_replays_all_pages_and_resumes_without_gaps(tmp_path: Path, status: JobStatus) -> None:
    async def run() -> None:
        store = make_store(tmp_path)
        store.create_job("test.replay", {}, job_id=JOB_ID, now=NOW)
        store.claim_job(JOB_ID, now=NOW)
        for index in range(405):
            store.append_job_event(JOB_ID, JobEventType.LOG, message=f"output-{index}")
        if status == JobStatus.SUCCEEDED:
            store.complete_job(JOB_ID, {})
        elif status == JobStatus.FAILED:
            store.fail_job(JOB_ID, ErrorDetail(code="test_failure", message="last error", retryable=False))
        elif status == JobStatus.CANCELLED:
            store.cancel_job(JOB_ID)
        else:
            store.interrupt_job(JOB_ID)
        expected = store.list_job_events(JOB_ID, limit=1000)
        service = ServerService(store, IdleBackend())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                for after, header in [(0, 0), (200, 300), (expected[-1].sequence, 0)]:
                    response = await client.get(f"/api/v1/jobs/{JOB_ID}/events/stream?after={after}",
                        headers={"Last-Event-ID": str(header)})
                    assert response.status_code == 200
                    payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                    assert [item["sequence"] for item in payloads] == [item.sequence for item in expected if item.sequence > max(after, header)]
                    if payloads:
                        assert payloads[-1]["data"]["status"] == status.value
        finally:
            await service.aclose()

    asyncio.run(run())


def test_job_event_stream_keeps_a_terminal_event_created_after_the_last_page_read(tmp_path: Path) -> None:
    async def run() -> None:
        store = make_store(tmp_path)
        store.create_job("test.replay", {}, job_id=JOB_ID, now=NOW)
        store.claim_job(JOB_ID, now=NOW)
        store.append_job_event(JOB_ID, JobEventType.LOG, message="last output")
        service = ServerService(store, IdleBackend())
        original = service.list_job_events
        completed = False

        async def complete_after_read(*args, **kwargs):
            nonlocal completed
            result = await original(*args, **kwargs)
            if not completed:
                completed = True
                store.complete_job(JOB_ID, {})
            return result

        service.list_job_events = complete_after_read
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                response = await client.get(f"/api/v1/jobs/{JOB_ID}/events/stream")
                assert response.status_code == 200
                payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                assert [item["sequence"] for item in payloads] == [1, 2, 3, 4]
                assert payloads[-1]["data"]["status"] == "succeeded"
        finally:
            await service.aclose()

    asyncio.run(run())


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


def test_cancel_during_claim_cannot_leave_a_job_cancelling(tmp_path, monkeypatch):
    async def run():
        store = make_store(tmp_path)
        claim = store.claim_job

        def claim_and_cancel(job_id):
            claim(job_id)
            return store.request_job_cancel(job_id)

        async def handler(context, payload):
            pytest.fail("a cancelled claim must not enter the handler")

        monkeypatch.setattr(store, "claim_job", claim_and_cancel)
        manager = JobManager(store, {"test.claim": handler})
        try:
            job = await manager.submit(CreateJobRequest(kind="test.claim"))
            task = manager._tasks[job.id]
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
            assert store.get_job(job.id).status == JobStatus.CANCELLED
        finally:
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
