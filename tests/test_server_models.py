from __future__ import annotations

import asyncio
import json
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.server.api import create_app
from mfq.server.backend import BackendDelta
from mfq.server.catalog import DiscoveredModel, DuplicateModelNameError, ModelCatalog
from mfq.server.models import JobStatus, RuntimeInstanceState, SamplingParams
from mfq.server.runtime_pool import ManagedRuntimePool, RuntimeConflictError, _ManagedRuntime
from mfq.server.service import ServerService
from mfq.server.storage import SessionStore
from mfq.tools.split_mfq import split_mfq


class IdleBackend:
    async def aclose(self) -> None:
        return None


def _model(path: Path, *, architecture: str = "test-model") -> None:
    save(
        path,
        FileHeader(version=2, model_arch=architecture),
        {
            "weight.0": np.arange(16, dtype=np.float16).reshape(4, 4),
            "weight.1": np.arange(16, dtype=np.float16).reshape(4, 4),
        },
    )


def _fake_runtime(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json
            from http.server import BaseHTTPRequestHandler, HTTPServer

            parser = argparse.ArgumentParser()
            parser.add_argument('--mfq')
            parser.add_argument('--server', action='store_true')
            parser.add_argument('--host')
            parser.add_argument('--port', type=int)
            parser.add_argument('--ctx-size', type=int)
            parser.add_argument('--prefill-chunk-size')
            parser.add_argument('--model-name')
            parser.add_argument('--moe-gpu-cache-gb')
            args = parser.parse_args()

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/health':
                        payload = {
                            'status': 'ok',
                            'model': args.model_name,
                            'model_type': 'qwen35',
                            'max_context': args.ctx_size,
                        }
                    elif self.path == '/api/status':
                        payload = {'status': 'ok', 'model': args.model_name}
                    elif self.path == '/v1/models':
                        payload = {'object': 'list', 'data': [{'id': args.model_name}]}
                    else:
                        self.send_response(404)
                        self.end_headers()
                        return
                    body = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def log_message(self, *args):
                    pass

            server = HTTPServer((args.host, args.port), Handler)
            server.serve_forever()
            """
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_catalog_validates_complete_and_incomplete_shards(tmp_path: Path) -> None:
    async def run() -> None:
        source = tmp_path / "source.mfq"
        _model(source)
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        shards = split_mfq(source, model_dir / "split.mfq", split_max_tensors=1)

        catalog = ModelCatalog([model_dir], cache_seconds=0)
        complete = await catalog.list()
        assert len(complete.data) == 1
        assert complete.data[0].name == "split"
        assert complete.data[0].complete
        assert complete.data[0].shard_count == 2
        assert complete.data[0].tensor_count == 2
        assert complete.data[0].dtypes == ["F16"]
        assert str(model_dir) not in complete.model_dump_json()

        shards[1].unlink()
        incomplete = await catalog.list(refresh=True)
        assert len(incomplete.data) == 1
        assert not incomplete.data[0].complete
        assert not incomplete.data[0].loadable
        assert "missing MFQ shard" in (incomplete.data[0].error or "")
        assert str(model_dir) not in incomplete.model_dump_json()

    asyncio.run(run())


def test_catalog_loads_registered_external_mfq_files(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "catalog"
        model_dir.mkdir()
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        external = external_dir / "portable.mfq"
        _model(external, architecture="qwen35")
        (model_dir / ".mfq-files.json").write_text(
            json.dumps({"version": 1, "files": [str(external)]}),
            encoding="utf-8",
        )

        catalog = ModelCatalog([model_dir], cache_seconds=0)
        artifacts = await catalog.list()

        assert [artifact.name for artifact in artifacts.data] == ["portable"]
        assert artifacts.data[0].architecture == "qwen35"
        assert artifacts.data[0].loadable

    asyncio.run(run())


def test_catalog_expands_registered_external_shard_set(tmp_path: Path) -> None:
    async def run() -> None:
        source = tmp_path / "source.mfq"
        _model(source, architecture="qwen35")
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        shards = split_mfq(source, external_dir / "portable.mfq", split_max_tensors=1)
        model_dir = tmp_path / "catalog"
        model_dir.mkdir()
        (model_dir / ".mfq-files.json").write_text(
            json.dumps({"version": 1, "files": [str(shards[1])]}),
            encoding="utf-8",
        )

        catalog = ModelCatalog([model_dir], cache_seconds=0)
        artifacts = await catalog.list()

        assert [artifact.name for artifact in artifacts.data] == ["portable"]
        assert artifacts.data[0].complete
        assert artifacts.data[0].loadable
        assert artifacts.data[0].shard_count == 2

    asyncio.run(run())


def test_common_server_browses_and_registers_external_model_directories(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        model_dir = tmp_path / "catalog"
        model_dir.mkdir()
        external_dir = tmp_path / "external-model"
        external_dir.mkdir()
        _model(external_dir / "portable.mfq", architecture="qwen35")
        native_picker_dir = tmp_path / "native-picker-model"
        native_picker_dir.mkdir()
        _model(native_picker_dir / "desktop.mfq", architecture="minicpmo45")
        catalog = ModelCatalog(
            [model_dir],
            cache_seconds=0,
            browse_roots=[external_dir],
        )
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            IdleBackend(),
            catalog=catalog,
        )
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                roots = await client.get("/api/v1/models/directories")
                assert roots.status_code == 200
                assert [entry["name"] for entry in roots.json()["data"]] == ["external-model"]
                directory_id = roots.json()["data"][0]["id"]

                listing = await client.get(
                    "/api/v1/models/directories",
                    params={"directory_id": directory_id},
                )
                assert listing.status_code == 200
                assert listing.json()["current_name"] == "external-model"
                assert listing.json()["current_path"] == str(external_dir.resolve())
                assert listing.json()["model_file_count"] == 1

                jumped = await client.get(
                    "/api/v1/models/directories",
                    params={"path": str(native_picker_dir)},
                )
                assert jumped.status_code == 200
                assert jumped.json()["current_name"] == "native-picker-model"
                assert jumped.json()["current_path"] == str(native_picker_dir.resolve())

                ambiguous = await client.get(
                    "/api/v1/models/directories",
                    params={"directory_id": directory_id, "path": str(external_dir)},
                )
                assert ambiguous.status_code == 400
                assert ambiguous.json()["error"]["code"] == "invalid_model_directory_source"

                registered = await client.post(
                    "/api/v1/models/directories/register",
                    json={"directory_id": directory_id},
                )
                assert registered.status_code == 200
                assert [artifact["name"] for artifact in registered.json()["data"]] == [
                    "portable"
                ]
                assert registered.json()["data"][0]["loadable"]

                native_registered = await client.post(
                    "/api/v1/models/directories/register",
                    json={"path": str(native_picker_dir)},
                )
                assert native_registered.status_code == 200
                assert [
                    artifact["name"] for artifact in native_registered.json()["data"]
                ] == ["desktop"]
                models = await client.get("/api/v1/models", params={"refresh": True})
                assert [artifact["name"] for artifact in models.json()["data"]] == [
                    "desktop",
                    "portable",
                ]
                index = json.loads((model_dir / ".mfq-files.json").read_text())
                assert index["version"] == 1
                assert index["files"] == sorted(
                    [
                        str(external_dir / "portable.mfq"),
                        str(native_picker_dir / "desktop.mfq"),
                    ]
                )
        finally:
            await service.aclose()

    asyncio.run(run())


def test_common_server_rejects_directories_without_mfq_models(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "catalog"
        model_dir.mkdir()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        catalog = ModelCatalog([model_dir], cache_seconds=0, browse_roots=[empty_dir])
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            IdleBackend(),
            catalog=catalog,
        )
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                directory_id = (await client.get("/api/v1/models/directories")).json()[
                    "data"
                ][0]["id"]
                response = await client.post(
                    "/api/v1/models/directories/register",
                    json={"directory_id": directory_id},
                )
                assert response.status_code == 400
                assert response.json()["error"]["code"] == "model_registration_failed"
                assert not (model_dir / ".mfq-files.json").exists()
        finally:
            await service.aclose()

    asyncio.run(run())


def test_empty_runtime_pool_reports_idle_state(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        pool = ManagedRuntimePool(ModelCatalog([model_dir]), tmp_path / "runtime")

        assert await pool.runtime_status() == {
            "runtime_state": "idle",
            "model": None,
            "active_requests": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "reloading": False,
        }
        assert await pool.realtime_capabilities() == {"available": False, "modes": []}

    asyncio.run(run())


def test_empty_runtime_pool_keeps_management_api_available(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = ManagedRuntimePool(catalog, tmp_path / "runtime")
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            pool,
            catalog=catalog,
            runtime_manager=pool,
        )
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                health = await client.get("/health")
                assert health.status_code == 200
                assert health.json()["service"] == "mfq-server"

                models = await client.get("/api/v1/models")
                assert models.status_code == 200
                assert models.json()["data"] == []

                runtime_models = await client.get("/api/v1/runtime/models")
                assert runtime_models.status_code == 200
                assert runtime_models.json() == {"object": "list", "data": []}

                status = await client.get("/api/v1/runtime/status")
                assert status.status_code == 200
                assert status.json()["runtime_state"] == "idle"
                assert status.json()["model"] is None

                realtime = await client.get("/api/v1/runtime/realtime/capabilities")
                assert realtime.status_code == 200
                assert realtime.json() == {"available": False, "modes": []}

                capabilities = await client.get("/api/v1/runtime/capabilities")
                assert capabilities.status_code == 503
                assert capabilities.json()["error"]["code"] == "model_not_loaded"
        finally:
            await service.aclose()

    asyncio.run(run())


def test_catalog_rejects_duplicate_mfq_file_stems(tmp_path: Path) -> None:
    async def run() -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        _model(first / "same-name.mfq", architecture="first")
        _model(second / "same-name.mfq", architecture="second")

        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        with pytest.raises(DuplicateModelNameError, match="duplicate catalog model name: same-name"):
            await catalog.list()

    asyncio.run(run())


def test_managed_runtime_loads_and_unloads_through_persistent_jobs(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "tiny.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = ManagedRuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            max_instances=1,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(
            store,
            pool,
            catalog=catalog,
            runtime_manager=pool,
        )
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                models = await client.get("/api/v1/models")
                assert models.status_code == 200
                artifact = models.json()["data"][0]
                assert str(model_dir) not in models.text

                accepted = await client.post(
                    "/api/v1/models/load",
                    json={"model": artifact["name"], "context_size": 4096},
                )
                assert accepted.status_code == 202
                job_id = accepted.json()["operation_id"]
                for _ in range(200):
                    job = (await client.get(f"/api/v1/jobs/{job_id}")).json()
                    if job["status"] in {"succeeded", "failed"}:
                        break
                    await asyncio.sleep(0.025)
                assert job["status"] == "succeeded", job
                instance_id = job["result"]["instance_id"]
                assert job["result"]["model_id"] == artifact["name"]
                assert job["result"]["artifact_id"] == artifact["id"]

                instances = await client.get("/api/v1/runtime/instances")
                assert instances.status_code == 200
                assert instances.json()["data"][0]["state"] == "ready"
                assert instances.json()["data"][0]["context_size"] == 4096

                duplicate = await client.post(
                    "/api/v1/models/load", json={"model": artifact["name"]}
                )
                duplicate_id = duplicate.json()["operation_id"]
                for _ in range(100):
                    duplicate_job = (await client.get(f"/api/v1/jobs/{duplicate_id}")).json()
                    if duplicate_job["status"] == "failed":
                        break
                    await asyncio.sleep(0.01)
                assert duplicate_job["error"]["code"] == "model_already_loaded"

                unloaded = await client.post(
                    "/api/v1/models/unload",
                    json={"instance_id": instance_id},
                )
                unload_id = unloaded.json()["operation_id"]
                for _ in range(200):
                    unload_job = (await client.get(f"/api/v1/jobs/{unload_id}")).json()
                    if unload_job["status"] in {"succeeded", "failed"}:
                        break
                    await asyncio.sleep(0.025)
                assert unload_job["status"] == "succeeded", unload_job
                assert (await client.get("/api/v1/runtime/instances")).json()["data"] == []
                assert store.get_job(unload_id).status == JobStatus.SUCCEEDED
        finally:
            await service.aclose()

    asyncio.run(run())


def test_concurrent_loads_reserve_the_catalog_name(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "tiny.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = ManagedRuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            max_instances=2,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, pool, catalog=catalog, runtime_manager=pool)
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            artifact = (await catalog.list()).data[0]
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                accepted = await asyncio.gather(
                    client.post("/api/v1/models/load", json={"model": artifact.name}),
                    client.post("/api/v1/models/load", json={"model": artifact.name}),
                )
                job_ids = [response.json()["operation_id"] for response in accepted]
                jobs = []
                for _ in range(200):
                    jobs = [
                        (await client.get(f"/api/v1/jobs/{job_id}")).json()
                        for job_id in job_ids
                    ]
                    if all(job["status"] in {"succeeded", "failed"} for job in jobs):
                        break
                    await asyncio.sleep(0.025)

                assert sorted(job["status"] for job in jobs) == ["failed", "succeeded"]
                failed = next(job for job in jobs if job["status"] == "failed")
                assert failed["error"]["code"] == "model_already_loaded"
                assert len((await pool.instances()).data) == 1
        finally:
            await service.aclose()

    asyncio.run(run())


def test_started_runtime_is_registered_in_the_instances_api(tmp_path: Path) -> None:
    async def run() -> None:
        model = tmp_path / "initial.mfq"
        _model(model, architecture="qwen35")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model)
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
        )
        pool = ManagedRuntimePool(catalog, tmp_path / "runtime", max_instances=1)
        instance_id = pool.register_started(
            artifact=artifact,
            process=process,
            backend=IdleBackend(),
            port=43123,
            context_size=8192,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(
            store,
            pool,
            catalog=catalog,
            runtime_manager=pool,
        )
        app = create_app(service)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/v1/runtime/instances")
                assert response.status_code == 200
                instances = response.json()["data"]
                assert len(instances) == 1
                assert instances[0]["model"] == "initial"
                assert instances[0]["state"] == "ready"
                assert instances[0]["context_size"] == 8192

                models = await client.get("/api/v1/runtime/models")
                assert models.status_code == 200
                visible = models.json()["data"]
                assert len(visible) == 1
                assert visible[0]["id"] == "initial"
                assert visible[0]["id"] != str(instances[0]["id"])
                assert visible[0]["instance_id"] == str(instance_id)
                assert visible[0]["instance_id"] == str(instances[0]["id"])
        assert process.poll() is not None

    asyncio.run(run())


def test_runtime_models_uses_catalog_name_when_no_alias_is_configured(tmp_path: Path) -> None:
    async def run() -> None:
        model = tmp_path / "catalog-name.mfq"
        _model(model, architecture="qwen35")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model)
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
        )
        pool = ManagedRuntimePool(catalog, tmp_path / "runtime", max_instances=1)
        instance_id = pool.register_started(
            artifact=artifact,
            process=process,
            backend=IdleBackend(),
            port=43123,
            context_size=8192,
        )
        duplicate = DiscoveredModel(
            resource=artifact.resource.model_copy(update={"id": "f" * 32}),
            path=tmp_path / "elsewhere" / "catalog-name.mfq",
        )
        try:
            with pytest.raises(
                RuntimeConflictError, match="model is already loaded: catalog-name"
            ):
                pool.register_started(
                    artifact=duplicate,
                    process=process,
                    backend=IdleBackend(),
                    port=43124,
                    context_size=8192,
                )
            models = await pool.runtime_models()
            assert models["data"] == [
                {
                    "id": "catalog-name",
                    "object": "model",
                    "model": "catalog-name",
                    "state": "ready",
                    "instance_id": str(instance_id),
                }
            ]
        finally:
            await pool.aclose()

    asyncio.run(run())


def test_started_runtime_monitor_reports_abnormal_exit(tmp_path: Path) -> None:
    async def run() -> None:
        model = tmp_path / "initial.mfq"
        _model(model, architecture="qwen35")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model)
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(0.05); raise SystemExit(137)",
            ],
            stdin=subprocess.DEVNULL,
        )
        pool = ManagedRuntimePool(catalog, tmp_path / "runtime", max_instances=1)
        pool.register_started(
            artifact=artifact,
            process=process,
            backend=IdleBackend(),
            port=43123,
            context_size=8192,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(
            store,
            pool,
            catalog=catalog,
            runtime_manager=pool,
        )
        app = create_app(service)
        async with app.router.lifespan_context(app):
            for _ in range(100):
                instances = await pool.instances()
                if instances.data[0].state == RuntimeInstanceState.FAILED:
                    break
                await asyncio.sleep(0.01)
            assert instances.data[0].state == RuntimeInstanceState.FAILED
            assert instances.data[0].error is not None
            assert instances.data[0].error.code == "runtime_exited"
            assert "137" in instances.data[0].error.message

    asyncio.run(run())


def test_managed_runtime_reports_and_bounds_queued_requests(tmp_path: Path) -> None:
    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        class BlockingBackend:
            async def stream(self, **options: object):
                seen_models.append(str(options["model"]))
                entered.set()
                await release.wait()
                yield BackendDelta(content_delta="ok", finish_reason="stop")

        seen_models: list[str] = []
        model = tmp_path / "tiny.mfq"
        _model(model)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve((await catalog.list()).data[0].id)
        instance = _ManagedRuntime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=BlockingBackend(),
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            request_slots=asyncio.Semaphore(1),
        )
        pool = ManagedRuntimePool(catalog, tmp_path / "runtime")
        pool._instances[instance.id] = instance
        assert await pool._select(artifact.resource.name, session_id=None) is instance
        assert await pool._select(artifact.resource.id, session_id=None) is None

        async def consume() -> None:
            async for _ in pool.stream(
                model=artifact.resource.name,
                messages=[{"role": "user", "content": "hello"}],
                sampling=SamplingParams(),
            ):
                pass

        first = asyncio.create_task(consume())
        await entered.wait()
        second = asyncio.create_task(consume())
        await asyncio.sleep(0)
        assert instance.active_requests == 1
        assert instance.queued_requests == 1
        listed = await pool.instances()
        assert listed.data[0].queued_requests == 1
        release.set()
        await asyncio.gather(first, second)
        assert seen_models == [artifact.resource.name, artifact.resource.name]
        assert instance.active_requests == 0
        assert instance.queued_requests == 0

    asyncio.run(run())
