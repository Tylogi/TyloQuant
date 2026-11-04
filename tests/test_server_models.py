from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import Float8E4M3Array, open_mmap, save
from mfq.server.api import create_app
from mfq.server.runtime.capabilities import capabilities_for_architecture
from mfq.server.state.catalog import (
    MODEL_FILE_INDEX,
    DiscoveredModel,
    DuplicateModelNameError,
    ModelCatalog,
)
from mfq.server.runtime.host_memory import HostMemorySnapshot
from mfq.server.services.jobs import JobExecutionError
from mfq.server.protocol.models import (
    ErrorDetail,
    JobStatus,
    ModelArtifactResource,
    ModelLoadRequest,
    RuntimeCapabilitiesResource,
    RuntimeInstanceState,
    SamplingParams,
)
from mfq.server.runtime.backend import BackendDelta, BackendError
from mfq.server.runtime.runtime_pool import (
    RuntimeConflictError,
    RuntimePool,
    _CachedLoadFailure,
    _RequestSlotGate,
    _Runtime,
)
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from mfq.tools.split_mfq import split_mfq
from tests.platform_helpers import launch_python_runtime


class IdleBackend:
    async def aclose(self) -> None:
        return None


class _TestJobContext:
    def __init__(self) -> None:
        self.callbacks = []

    def raise_if_cancelled(self) -> None:
        return None

    def add_cleanup(self, callback) -> None:
        self.callbacks.append(callback)

    async def progress(self, *_args, **_kwargs) -> None:
        return None

    async def log(self, *_args, **_kwargs) -> None:
        return None

    async def cleanup(self) -> None:
        for callback in reversed(self.callbacks):
            result = callback()
            if result is not None:
                await result


def test_empty_runtime_pool_reports_an_idle_server() -> None:
    async def run() -> None:
        pool = RuntimePool(ModelCatalog([]), "runtime", backend="metal")

        status = await pool.runtime_status()
        assert status["runtime_state"] == "idle"
        assert status["model"] is None
        assert status["active_requests"] == 0
        assert await pool.runtime_models() == {"object": "list", "data": []}
        assert await pool.realtime_capabilities() == {"available": False, "modes": []}

    asyncio.run(run())


def test_metal_runtime_pool_derives_a_safe_default_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.total_physical_memory",
        lambda: 128 << 30,
    )
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.metal_allocation_limit",
        lambda: 110 << 30,
    )

    automatic = RuntimePool(ModelCatalog([]), "runtime", backend="metal")
    disabled = RuntimePool(
        ModelCatalog([]),
        "runtime",
        backend="metal",
        automatic_memory_budget=False,
    )
    explicit = RuntimePool(
        ModelCatalog([]),
        "runtime",
        backend="metal",
        max_runtime_memory_bytes=100 << 30,
    )
    cuda = RuntimePool(ModelCatalog([]), "runtime", backend="cuda")

    assert automatic.max_runtime_memory_bytes == 120 << 30
    assert automatic.automatic_memory_budget is True
    assert disabled.max_runtime_memory_bytes is None
    assert disabled.automatic_memory_budget is False
    assert explicit.max_runtime_memory_bytes == 100 << 30
    assert explicit.automatic_memory_budget is False
    assert cuda.max_runtime_memory_bytes is None
    assert cuda.automatic_memory_budget is False


def test_automatic_memory_budget_tracks_current_reclaimable_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1 << 30
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.process_physical_footprint", lambda _: 0)
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.total_physical_memory",
        lambda: 128 * gib,
    )
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.metal_allocation_limit",
        lambda: 110 * gib,
    )
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.host_memory_snapshot",
        lambda: HostMemorySnapshot(
            total=128 * gib,
            free=4 * gib,
            active=80 * gib,
            inactive=4 * gib,
            wired=40 * gib,
        ),
    )

    automatic = RuntimePool(ModelCatalog([]), "runtime", backend="metal")
    automatic._load_bytes["resident"] = 30 * gib
    automatic._resident_process_usage_locked = lambda: (30 * gib, 0)
    explicit = RuntimePool(
        ModelCatalog([]),
        "runtime",
        backend="metal",
        max_runtime_memory_bytes=100 * gib,
    )
    explicit._load_bytes["resident"] = 30 * gib
    explicit._resident_process_usage_locked = lambda: (30 * gib, 0)

    assert automatic._effective_runtime_memory_budget_locked() == 50 * gib
    assert explicit._effective_runtime_memory_budget_locked() == 50 * gib
    assert HostMemorySnapshot(0, -1, 10, -1, 0).reclaimable(
        active_ratio=2.0
    ) == 10



def test_automatic_memory_pressure_uses_soft_and_hard_watermarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1 << 30
    reclaimable_gib = 12
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.total_physical_memory",
        lambda: 128 * gib,
    )
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.host_memory_snapshot",
        lambda: HostMemorySnapshot(
            total=128 * gib,
            free=reclaimable_gib * gib,
            active=0,
            inactive=0,
            wired=80 * gib,
        ),
    )
    pool = RuntimePool(ModelCatalog([]), "runtime", backend="metal")
    pool._load_bytes["resident"] = 40 * gib
    pool._resident_process_usage_locked = lambda: (40 * gib, 0)

    level, ratio, ceiling, committed = pool._runtime_memory_pressure_locked()
    assert level == "soft"
    assert ratio == pytest.approx(40 / 44)
    assert ceiling == 44 * gib
    assert committed == 40 * gib

    reclaimable_gib = 10
    level, ratio, ceiling, committed = pool._runtime_memory_pressure_locked()
    assert level == "hard"
    assert ratio == pytest.approx(40 / 42)
    assert ceiling == 42 * gib
    assert committed == 40 * gib


def test_load_pressure_reclaims_shared_host_cache_before_model_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        gib = 1 << 30
        reclaimed = False
        calls = 0

        def snapshot() -> HostMemorySnapshot:
            return HostMemorySnapshot(
                total=128 * gib,
                free=(40 if reclaimed else 2) * gib,
                active=8 * gib,
                inactive=2 * gib,
                wired=40 * gib,
            )

        def reclaim() -> int:
            nonlocal calls, reclaimed
            calls += 1
            reclaimed = True
            return 512 << 20

        monkeypatch.setattr(
            "mfq.server.runtime.runtime_pool.total_physical_memory",
            lambda: 128 * gib,
        )
        monkeypatch.setattr(
            "mfq.server.runtime.runtime_pool.host_memory_snapshot",
            snapshot,
        )
        pool = RuntimePool(
            ModelCatalog([]),
            "runtime",
            backend="metal",
            shared_cache_reclaimer=reclaim,
        )
        pool._load_bytes["resident"] = 30 * gib
        pool._resident_process_usage_locked = lambda: (30 * gib, 0)

        assert await pool._reclaim_shared_cache_for_budget(
            additional_bytes=10 * gib,
        ) == 512 << 20
        assert await pool._reclaim_shared_cache_for_budget(
            additional_bytes=10 * gib,
        ) == 0
        assert calls == 1
        assert pool._shared_cache_reclaims == 1
        assert pool._shared_cache_released_bytes == 512 << 20
        assert pool._shared_cache_reclaim_failures == 0

    asyncio.run(run())


def test_startup_models_use_the_managed_load_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        model = tmp_path / "startup.mfq"
        executable = tmp_path / "fake-runtime"
        _model(model, architecture="qwen35")
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model)
        request = ModelLoadRequest(
            model=artifact.resource.name,
            artifact_uri=f"mfq://{artifact.resource.id}",
            context_size=8192,
            prefill_chunk_size=333,
            prefix_cache_disk_bytes=1234,
        )
        pool = RuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            startup_loads=[request],
        )

        try:
            await pool.start()
            instances = (await pool.instances()).data
            assert len(instances) == 1
            assert instances[0].model == "startup"
            assert instances[0].state == RuntimeInstanceState.READY
            assert instances[0].context_size == 8192
            remembered = pool._load_requests["startup"]
            assert remembered.artifact_uri == f"mfq://{artifact.resource.id}"
            assert remembered.prefill_chunk_size == 333
            assert remembered.prefix_cache_disk_bytes == 1234
            assert pool._startup_loads == []
        finally:
            await pool.aclose()

    asyncio.run(run())


def test_startup_model_can_use_stdio_runtime_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        model = tmp_path / "stdio.mfq"
        executable = tmp_path / "fake-stdio-runtime"
        _model(model, architecture="qwen35")
        _fake_stdio_runtime(executable, monkeypatch)
        pool = RuntimePool(
            ModelCatalog([tmp_path], cache_seconds=0),
            executable,
            startup_timeout_seconds=5,
            startup_loads=[ModelLoadRequest(model="stdio", context_size=8192)],
            transport="stdio",
        )

        try:
            await pool.start()
            instances = (await pool.instances()).data
            assert len(instances) == 1
            assert instances[0].model == "stdio"
            assert instances[0].state == RuntimeInstanceState.READY
            assert instances[0].context_size == 8192
            assert instances[0].id in pool._instances
            assert pool._instances[instances[0].id].port == 0
        finally:
            await pool.aclose()

    asyncio.run(run())


def test_automatic_expert_residency_is_recomputed_for_later_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        model = tmp_path / "automatic-residency.mfq"
        executable = tmp_path / "fake-runtime"
        _model(model, architecture="qwen35")
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
        )

        def automatic_residency(
            _artifact: DiscoveredModel,
            request: ModelLoadRequest,
            *,
            memory_ceiling: int | None = None,
        ) -> ModelLoadRequest:
            del memory_ceiling
            return request.model_copy(update={"moe_gpu_cache_gb": 42.0})

        pool._apply_automatic_expert_residency = (  # type: ignore[method-assign]
            automatic_residency
        )
        context = _TestJobContext()
        try:
            await pool.load(
                context,  # type: ignore[arg-type]
                {"model": "automatic-residency"},
            )
            assert pool._load_requests["automatic-residency"].moe_gpu_cache_gb is None
        finally:
            await context.cleanup()
            await pool.aclose()

    asyncio.run(run())


def test_failed_startup_model_remains_retryable_without_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        model = tmp_path / "too-large.mfq"
        executable = tmp_path / "fake-runtime"
        _model(model, architecture="qwen35")
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model)
        request = ModelLoadRequest(
            model=artifact.resource.name,
            artifact_uri=f"mfq://{artifact.resource.id}",
        )
        pool = RuntimePool(
            catalog,
            executable,
            max_runtime_memory_bytes=1,
            startup_loads=[request],
        )

        try:
            with pytest.raises(JobExecutionError) as blocked:
                await pool.start()
            assert blocked.value.detail.code == "runtime_model_too_large"
            assert pool._instances == {}
            assert pool._startup_loads == [request]
        finally:
            await pool.aclose()

    asyncio.run(run())


def _model(path: Path, *, architecture: str = "test-model") -> None:
    save(
        path,
        FileHeader(version=2, model_arch=architecture),
        {
            "weight.0": np.arange(16, dtype=np.float16).reshape(4, 4),
            "weight.1": np.arange(16, dtype=np.float16).reshape(4, 4),
        },
    )


def _fake_stdio_runtime(path: Path, monkeypatch) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json
            import sys

            parser = argparse.ArgumentParser()
            parser.add_argument('--model')
            parser.add_argument('--transport', choices=('stdio', 'http'))
            parser.add_argument('--ctx-size', type=int)
            parser.add_argument('--model-name')
            args, _ = parser.parse_known_args()

            def send(frame):
                print(json.dumps({'v': 1, **frame}, separators=(',', ':')), flush=True)

            print('fake runtime log', file=sys.stderr, flush=True)
            send({'type': 'ready'})
            for line in sys.stdin:
                request = json.loads(line)
                request_id = request.get('id')
                op = request['op']
                if op == 'shutdown':
                    break
                if op == 'health':
                    data = {
                        'status': 'ok',
                        'model': args.model_name,
                        'model_type': 'qwen35',
                        'model_capabilities': {
                            'architecture_family': 'qwen3_5',
                            'source': 'fake-runtime',
                            'features': {'text': True, 'mtp': False},
                        },
                        'mtp_available': False,
                        'max_context': args.ctx_size,
                    }
                elif op == 'status':
                    data = {'status': 'ok', 'model': args.model_name}
                elif op == 'models':
                    data = {'models': [{'name': args.model_name, 'type': 'qwen35'}]}
                else:
                    send({
                        'id': request_id,
                        'type': 'error',
                        'error': {
                            'code': 'unsupported_operation',
                            'message': op,
                            'status_code': 501,
                            'retryable': False,
                        },
                    })
                    continue
                send({'id': request_id, 'type': 'result', 'data': data})
            """
        ),
        encoding="utf-8",
    )
    launch_python_runtime(path, monkeypatch)


def _fake_runtime(path: Path, monkeypatch) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json
            from http.server import BaseHTTPRequestHandler, HTTPServer

            parser = argparse.ArgumentParser()
            parser.add_argument('--model')
            parser.add_argument('--transport', choices=('stdio', 'http'))
            parser.add_argument('--host')
            parser.add_argument('--port', type=int)
            parser.add_argument('--ctx-size', type=int)
            parser.add_argument('--prefill-chunk-size')
            parser.add_argument('--model-name')
            parser.add_argument('--moe-gpu-cache-gb')
            args = parser.parse_args()

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/runtime/health':
                        has_mtp = args.model_name.endswith('-with-mtp')
                        payload = {
                            'status': 'ok',
                            'model': args.model_name,
                            'model_type': 'qwen35',
                            'model_capabilities': {
                                'architecture_family': 'qwen3_5',
                                'source': 'fake-runtime',
                                'features': {'text': True, 'mtp': has_mtp},
                            },
                            'mtp_available': has_mtp,
                            'max_context': args.ctx_size,
                        }
                    elif self.path == '/runtime/status':
                        payload = {'status': 'ok', 'model': args.model_name}
                    elif self.path == '/runtime/models':
                        payload = {'models': [{'name': args.model_name, 'type': 'qwen35'}]}
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
    launch_python_runtime(path, monkeypatch)


async def _wait_for_job(
    client: httpx.AsyncClient,
    operation_id: str,
    *,
    attempts: int = 300,
) -> dict[str, object]:
    job: dict[str, object] = {}
    for _ in range(attempts):
        job = (await client.get(f"/api/v1/jobs/{operation_id}")).json()
        if job.get("status") in {"succeeded", "failed", "cancelled"}:
            return job
        await asyncio.sleep(0.025)
    return job


@pytest.mark.parametrize("extensions", [(".mfq", ".mfq"), (".MFQ", ".MFQ"), (".mFq", ".MfQ")])
def test_catalog_validates_complete_and_incomplete_shards(tmp_path: Path, extensions) -> None:
    async def run() -> None:
        source = tmp_path / "source.mfq"
        _model(source)
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        shards = split_mfq(source, model_dir / "split.mfq", split_max_tensors=1)
        shards = [shard.rename(shard.with_suffix(extension))
            for shard, extension in zip(shards, extensions, strict=True)]

        catalog = ModelCatalog([model_dir], cache_seconds=0)
        complete = await catalog.list()
        assert len(complete.data) == 1
        assert complete.data[0].name == "split"
        assert complete.data[0].complete
        assert complete.data[0].shard_count == 2
        assert complete.data[0].tensor_count == 2
        assert complete.data[0].dtypes == ["F16"]
        with open_mmap(shards[0]) as store:
            assert complete.data[0].estimated_resident_weight_bytes == sum(record.nbytes for record in store.records.values())
        assert complete.data[0].ssd_ple_bytes == 0
        assert str(model_dir) not in complete.model_dump_json()

        shards[1].unlink()
        incomplete = await catalog.list(refresh=True)
        assert len(incomplete.data) == 1
        assert not incomplete.data[0].complete
        assert not incomplete.data[0].loadable
        assert incomplete.data[0].missing_shards == 1
        assert incomplete.data[0].estimated_resident_weight_bytes is None
        assert incomplete.data[0].ssd_ple_bytes is None
        assert incomplete.data[0].total_bytes == shards[0].stat().st_size
        assert catalog._immediate_model_count(model_dir) == 1
        assert "missing MFQ shard" in (incomplete.data[0].error or "")
        assert str(model_dir) not in incomplete.model_dump_json()

    asyncio.run(run())


def test_catalog_serves_stale_snapshot_during_one_background_refresh(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _model(tmp_path / "first.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=60)
        initial = await catalog.list()
        assert [item.name for item in initial.data] == ["first"]

        _model(tmp_path / "second.mfq")
        entered = threading.Event()
        release = threading.Event()
        original_scan = catalog._scan
        scan_count = 0

        def blocking_scan() -> dict[str, DiscoveredModel]:
            nonlocal scan_count
            scan_count += 1
            entered.set()
            assert release.wait(timeout=2)
            return original_scan()

        catalog._scan = blocking_scan  # type: ignore[method-assign]
        catalog._last_scan = 0.0
        catalog._last_scan_attempt = 0.0
        try:
            stale = await asyncio.wait_for(catalog.list(), timeout=0.1)
            await asyncio.sleep(0)
            assert entered.wait(timeout=1)
            another = await asyncio.wait_for(catalog.list(), timeout=0.1)
            assert [item.name for item in stale.data] == ["first"]
            assert [item.name for item in another.data] == ["first"]
            assert scan_count == 1
        finally:
            release.set()

        for _ in range(100):
            if catalog._refresh_task is None:
                break
            await asyncio.sleep(0.01)
        assert catalog._refresh_task is None
        refreshed = await catalog.list()
        assert [item.name for item in refreshed.data] == ["first", "second"]
        assert scan_count == 1

    asyncio.run(run())


def test_catalog_force_refresh_waits_for_a_post_background_scan(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _model(tmp_path / "first.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=60)
        await catalog.list()

        entered = threading.Event()
        release = threading.Event()
        original_scan = catalog._scan
        scan_count = 0

        def blocking_scan() -> dict[str, DiscoveredModel]:
            nonlocal scan_count
            scan_count += 1
            entered.set()
            assert release.wait(timeout=2)
            return original_scan()

        catalog._scan = blocking_scan  # type: ignore[method-assign]
        catalog._last_scan = 0.0
        catalog._last_scan_attempt = 0.0
        await catalog.list()
        await asyncio.sleep(0)
        assert entered.wait(timeout=1)

        forced = asyncio.create_task(catalog.list(refresh=True))
        await asyncio.sleep(0)
        assert not forced.done()
        release.set()

        result = await asyncio.wait_for(forced, timeout=2)
        assert [item.name for item in result.data] == ["first"]
        assert scan_count == 2

    asyncio.run(run())


def test_catalog_tracks_streamable_routed_expert_bytes(tmp_path: Path) -> None:
    async def run() -> None:
        dense = np.arange(16, dtype=np.float16).reshape(4, 4)
        experts = np.arange(128, dtype=np.float16).reshape(2, 8, 8)
        path = tmp_path / "moe.mfq"
        save(
            path,
            FileHeader(version=2, model_arch="qwen4_exp"),
            {
                "model.token_embedding.weight": dense,
                "model.block.0.mlp.experts.gate_up.weight": experts,
            },
        )

        artifact = await ModelCatalog([tmp_path], cache_seconds=0).resolve_path(path)
        with open_mmap(path) as store:
            expert_blob_bytes = store.records[
                "model.block.0.mlp.experts.gate_up.weight"
            ].nbytes
        assert artifact.routed_expert_bytes == expert_blob_bytes
        assert expert_blob_bytes > experts.nbytes
        assert artifact.resource.total_bytes > artifact.routed_expert_bytes

        request = ModelLoadRequest(
            model=artifact.resource.name,
            moe_gpu_cache_gb=1 / (1 << 30),
        )
        pool = RuntimePool(ModelCatalog([tmp_path]), tmp_path / "runtime")
        estimated = pool._estimated_load_bytes(artifact, request)
        assert estimated == (
            artifact.resource.total_bytes - artifact.routed_expert_bytes + 1
        )
        runtime = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=0,
            context_size=4096,
            reserved_bytes=estimated,
        )
        assert RuntimePool._committed_runtime_bytes(runtime) == estimated
        runtime.resident_bytes = max(1, estimated // 2)
        assert RuntimePool._committed_runtime_bytes(runtime) == estimated
        runtime.resident_bytes = estimated + 17
        assert RuntimePool._committed_runtime_bytes(runtime) == estimated + 17

    asyncio.run(run())


def test_native_hf_metal_auto_streaming_reserves_its_expert_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "native-hf"
    model.mkdir()
    artifact = DiscoveredModel(
        resource=ModelArtifactResource(
            id="1" * 32,
            name="native-hf",
            architecture="qwen4-exp-hf-full-mfq",
            format="hf",
            shard_count=1,
            total_bytes=10 << 30,
            tensor_count=1,
            record_count=2,
            complete=True,
            loadable=True,
            modified_at=datetime.now(timezone.utc),
        ),
        path=model,
        routed_expert_bytes=8 << 30,
    )
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.total_physical_memory",
        lambda: 6 << 30,
    )
    request = ModelLoadRequest(model="native-hf")

    metal = RuntimePool(ModelCatalog([tmp_path]), tmp_path / "runtime")
    cuda = RuntimePool(
        ModelCatalog([tmp_path]),
        tmp_path / "runtime",
        backend="cuda",
    )

    assert metal._estimated_load_bytes(artifact, request) == 6 << 30
    assert cuda._estimated_load_bytes(artifact, request) == 10 << 30


def test_oversized_mfq_moe_gets_an_automatic_metal_expert_budget(
    tmp_path: Path,
) -> None:
    model = tmp_path / "oversized.mfq"
    _model(model)
    artifact = DiscoveredModel(
        resource=ModelArtifactResource(
            id="2" * 32,
            name="oversized",
            architecture="qwen4-exp-mfq",
            format="mfq",
            shard_count=1,
            total_bytes=125 << 30,
            tensor_count=1,
            record_count=2,
            complete=True,
            loadable=True,
            modified_at=datetime.now(timezone.utc),
        ),
        path=model,
        routed_expert_bytes=110 << 30,
    )
    pool = RuntimePool(
        ModelCatalog([tmp_path]),
        tmp_path / "runtime",
        max_runtime_memory_bytes=96 << 30,
    )

    automatic = pool._apply_automatic_expert_residency(
        artifact,
        ModelLoadRequest(model="oversized"),
    )
    full_resident = pool._apply_automatic_expert_residency(
        artifact,
        ModelLoadRequest(model="oversized", moe_gpu_cache_gb=0),
    )

    assert automatic.moe_gpu_cache_gb == 77
    assert pool._estimated_load_bytes(artifact, automatic) == 92 << 30
    assert full_resident.moe_gpu_cache_gb == 0
    assert pool._estimated_load_bytes(artifact, full_resident) == 125 << 30

    pressure_limited = pool._apply_automatic_expert_residency(
        artifact,
        ModelLoadRequest(model="oversized"),
        memory_ceiling=60 << 30,
    )
    assert pressure_limited.moe_gpu_cache_gb == 42
    assert pool._estimated_load_bytes(artifact, pressure_limited) == 57 << 30


def test_native_hf_moe_uses_the_memory_budget(tmp_path: Path) -> None:
    model = tmp_path / "native-hf"
    model.mkdir()
    artifact = DiscoveredModel(
        resource=ModelArtifactResource(
            id="3" * 32,
            name="native-hf",
            architecture="deepseek_v4",
            format="hf",
            shard_count=16,
            total_bytes=160 << 30,
            tensor_count=1,
            record_count=1,
            complete=True,
            loadable=True,
            modified_at=datetime.now(timezone.utc),
        ),
        path=model,
        routed_expert_bytes=140 << 30,
    )

    roomy = RuntimePool(
        ModelCatalog([tmp_path]),
        tmp_path / "runtime",
        max_runtime_memory_bytes=506 << 30,
    )
    constrained = RuntimePool(
        ModelCatalog([tmp_path]),
        tmp_path / "runtime",
        max_runtime_memory_bytes=120 << 30,
    )

    full = roomy._apply_automatic_expert_residency(
        artifact,
        ModelLoadRequest(model="native-hf"),
    )
    bounded = constrained._apply_automatic_expert_residency(
        artifact,
        ModelLoadRequest(model="native-hf"),
    )

    assert full.moe_gpu_cache_gb == 0
    assert roomy._estimated_load_bytes(artifact, full) == 160 << 30
    assert bounded.moe_gpu_cache_gb == 96
    assert constrained._estimated_load_bytes(artifact, bounded) == 116 << 30


def test_native_hf_residency_excludes_independently_streamed_engram(
    tmp_path: Path,
) -> None:
    model = tmp_path / "native-v41"
    model.mkdir()
    artifact = DiscoveredModel(
        resource=ModelArtifactResource(
            id="5" * 32,
            name="native-v41",
            architecture="deepseek_v41-hf-full-mfq",
            format="hf",
            shard_count=16,
            total_bytes=160 << 30,
            tensor_count=1,
            record_count=1,
            complete=True,
            loadable=True,
            modified_at=datetime.now(timezone.utc),
        ),
        path=model,
        routed_expert_bytes=100 << 30,
        always_streamed_bytes=50 << 30,
    )
    pool = RuntimePool(
        ModelCatalog([tmp_path]),
        tmp_path / "runtime",
        max_runtime_memory_bytes=100 << 30,
    )

    automatic = pool._apply_automatic_expert_residency(
        artifact,
        ModelLoadRequest(model="native-v41"),
    )
    full_resident = pool._apply_automatic_expert_residency(
        artifact,
        ModelLoadRequest(model="native-v41", moe_gpu_cache_gb=0),
    )

    assert automatic.moe_gpu_cache_gb == 86
    assert pool._estimated_load_bytes(artifact, automatic) == 96 << 30
    assert pool._estimated_load_bytes(artifact, full_resident) == 110 << 30


def test_catalog_excludes_only_ple_table_payloads_from_resident_weights(tmp_path: Path) -> None:
    model = tmp_path / "ple.mfq"
    root = "model.block.0.position_embedding.ngram"
    tensors = {
        root + ".shard.0.weight": np.zeros((8, 16), dtype=np.uint8).view(Float8E4M3Array),
        root + ".shard.12.weight": np.zeros((8, 16), dtype=np.uint8).view(Float8E4M3Array),
        root + ".weight_scale": np.ones(1, dtype=np.float16),
        root + ".head_offsets": np.zeros(4, dtype=np.int64),
        "model.block.0.position_embedding.key.weight": np.zeros((16, 16), dtype=np.float16),
        "model.block.0.mlp.experts.gate.weight": np.zeros((2, 8, 16), dtype=np.float16),
    }
    save(model, FileHeader(model_arch="qwen4_exp", num_tensors=len(tensors)), tensors)
    artifact = ModelCatalog._inspect(tmp_path, model)
    with open_mmap(model) as store:
        ple_bytes = sum(store.records[root + f".shard.{i}.weight"].nbytes for i in (0, 12))
        expert_bytes = store.records["model.block.0.mlp.experts.gate.weight"].nbytes
        weight_bytes = sum(record.nbytes for record in store.records.values()) - ple_bytes
    assert artifact.always_streamed_bytes == ple_bytes
    assert artifact.routed_expert_bytes == expert_bytes
    assert artifact.resource.ssd_ple_bytes == ple_bytes
    assert artifact.resource.estimated_resident_weight_bytes == weight_bytes


def test_local_and_download_resident_estimates_use_the_same_tensor_metadata(tmp_path: Path) -> None:
    from mfq.server.services.hub_metadata import estimated_resident_weight_bytes, inspect_mfq_header

    model = tmp_path / "metadata.mfq"
    tensors = {
        "model.block.1.position_embedding.ngram.shard.0.weight": np.zeros((8, 16), dtype=np.float16),
        "model.weight": np.zeros((8, 16), dtype=np.float16),
        "__mfq_asset__/hf/tokenizer.json": b'{"extra": true}',
    }
    save(model, FileHeader(version=2, model_arch="qwen4_exp"), tensors)
    metadata = inspect_mfq_header(model.read_bytes(), model.stat().st_size)
    resource = ModelCatalog._inspect(tmp_path, model).resource
    assert resource.estimated_resident_weight_bytes == estimated_resident_weight_bytes(metadata.weight_bytes, metadata.weight_bytes_by_dtype)
    assert resource.estimated_resident_weight_bytes == metadata.weight_bytes == 276
    assert resource.ssd_ple_bytes == metadata.ssd_ple_bytes == 276


@pytest.mark.parametrize("format", ["mfq", "hf"])
def test_ple_residency_uses_remaining_budget_for_experts(tmp_path: Path, format: str) -> None:
    artifact = DiscoveredModel(
        resource=ModelArtifactResource(
            id="6" * 32, name="ple", architecture="qwen4_exp", format=format,
            shard_count=1, total_bytes=104 << 30, tensor_count=1, record_count=1,
            complete=True, loadable=True, modified_at=datetime.now(timezone.utc),
        ),
        path=tmp_path / "ple.mfq", routed_expert_bytes=68 << 30,
        always_streamed_bytes=30 << 30,
    )
    pool = RuntimePool(ModelCatalog([tmp_path]), tmp_path / "runtime",
                       max_runtime_memory_bytes=100 << 30)
    request = ModelLoadRequest(model="ple")
    full = pool._apply_automatic_expert_residency(artifact, request)
    assert full.moe_gpu_cache_gb in (None, 0)
    assert pool._estimated_load_bytes(artifact, full) == 74 << 30
    bounded = pool._apply_automatic_expert_residency(
        artifact, request, memory_ceiling=60 << 30,
    )
    # Six GiB of dense weights, three GiB of headroom; no full PLE reservation.
    assert bounded.moe_gpu_cache_gb == 51
    assert pool._estimated_load_bytes(artifact, bounded) == 57 << 30
    explicit = pool._apply_automatic_expert_residency(
        artifact, request.model_copy(update={"moe_gpu_cache_gb": 20}),
    )
    assert explicit.moe_gpu_cache_gb == 20


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


def test_catalog_refresh_drops_deleted_registered_files(tmp_path: Path) -> None:
    async def run() -> None:
        root = tmp_path / "catalog"
        root.mkdir()
        external = tmp_path / "portable.mfq"
        _model(external)
        index = root / MODEL_FILE_INDEX
        index.write_text(json.dumps({"version": 1, "files": [str(external)]}))
        catalog = ModelCatalog([root], cache_seconds=60)
        assert len((await catalog.list()).data) == 1
        external.unlink()
        assert (await catalog.list(refresh=True)).data == []
        assert (await ModelCatalog([root]).list()).data == []

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


@pytest.mark.parametrize("kind", ["single", "sharded", "incomplete"])
def test_model_file_directory_resolves_the_logical_asset(tmp_path: Path, kind: str) -> None:
    async def run() -> None:
        directory = tmp_path / "library" / "model-files"
        directory.mkdir(parents=True)
        source = tmp_path / "source.mfq"
        if kind == "single":
            _model(directory / "checkpoint.mfq")
        else:
            _model(source)
            shards = split_mfq(source, directory / "checkpoint.mfq", split_max_tensors=1)
            if kind == "incomplete":
                shards[-1].unlink()
        catalog = ModelCatalog([tmp_path / "library"])
        service = ServerService(SessionStore(tmp_path / "server.sqlite3"), IdleBackend(), catalog=catalog)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                artifacts = (await client.get("/api/v1/models")).json()["data"]
                assert len(artifacts) == 1
                response = await client.get(f'/api/v1/models/{artifacts[0]["id"]}/directory')
                assert response.status_code == 200
                assert response.json()["current_path"] == str(directory.resolve())
                assert response.json()["model_file_count"] == 1
                assert response.json()["files"] == [
                    {"name": item.name, "byte_size": item.stat().st_size}
                    for item in sorted(directory.glob("*.mfq"))
                ]
                if kind == "incomplete":
                    assert artifacts[0]["missing_shards"] == 1
                missing = await client.get(f'/api/v1/models/{"0" * 32}/directory')
                assert missing.status_code == 404
                assert missing.json()["error"]["code"] == "model_artifact_not_found"
        finally:
            await service.aclose()
    asyncio.run(run())


def test_hf_model_file_directory_uses_checkpoint_directory_not_its_parent(tmp_path: Path, monkeypatch) -> None:
    async def run() -> None:
        directory = tmp_path / "hf-model"
        directory.mkdir()
        catalog = ModelCatalog([tmp_path])
        async def get(_):
            return SimpleNamespace(path=directory, resource=SimpleNamespace(format="hf"))
        monkeypatch.setattr(catalog, "get", get)
        service = ServerService(SessionStore(tmp_path / "server.sqlite3"), IdleBackend(), catalog=catalog)
        try:
            listing = await service.model_artifact_directory("asset")
            assert listing.current_path == str(directory.resolve())
        finally:
            await service.aclose()
    asyncio.run(run())


def test_directory_browser_lists_mfq_files_without_decoding_or_registering(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "nested").mkdir()
    (directory / "b.MFQ").write_bytes(b"not-yet-complete")
    (directory / "a.mfq").write_bytes(b"partial")
    (directory / "ignored.txt").write_text("not a model")
    catalog = ModelCatalog([tmp_path])
    listing = catalog._browse_directories(None, directory)
    assert [item.name for item in listing.data] == ["nested"]
    assert [(item.name, item.byte_size) for item in listing.files] == [("a.mfq", 7), ("b.MFQ", 16)]
    assert not (tmp_path / MODEL_FILE_INDEX).exists()


@pytest.mark.parametrize("suffix", ["MFQ", "mfQ", "mFq"])
def test_catalog_discovers_and_registers_mfq_files_shown_by_the_browser(tmp_path: Path, suffix: str) -> None:
    async def run() -> None:
        directory = tmp_path / "external"
        directory.mkdir()
        _model(directory / f"portable.{suffix}", architecture="qwen35")
        direct = ModelCatalog([directory], cache_seconds=0)
        listing = direct._browse_directories(None, directory)
        assert listing.model_file_count == 1 and listing.files[0].name == f"portable.{suffix}"
        discovered = await direct.list()
        assert len(discovered.data) == 1 and discovered.data[0].name == "portable"
        assert discovered.data[0].loadable
        root = tmp_path / "catalog"
        root.mkdir()
        catalog = ModelCatalog([root], cache_seconds=0)
        registered = await catalog.register_directory(path=directory)
        assert len(registered.data) == 1 and registered.data[0].name == "portable"
        assert (await catalog.list()).data[0].loadable
    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, OSError("failed"), subprocess.TimeoutExpired("open", 5)])
@pytest.mark.parametrize("platform,display", [("darwin", "DISPLAY"), ("win32", "DISPLAY"),
    ("linux", "DISPLAY"), ("linux", "WAYLAND_DISPLAY")])
def test_file_manager_opens_only_a_browsed_directory_by_opaque_id(tmp_path: Path, monkeypatch, failure, platform, display) -> None:
    async def run() -> None:
        directory = tmp_path / "models ; spaces"
        directory.mkdir()
        catalog = ModelCatalog([tmp_path])
        service = ServerService(SessionStore(tmp_path / "server.sqlite3"), IdleBackend(), catalog=catalog)
        calls = []
        def open_directory(command, **kwargs):
            calls.append((command, kwargs))
            if failure is not None:
                raise failure
        monkeypatch.setattr("mfq.server.services.desktop.sys.platform", platform)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv(display, ":0")
        monkeypatch.setattr("mfq.server.services.desktop.shutil.which", lambda name: "/usr/bin/xdg-open")
        monkeypatch.setattr("mfq.server.services.desktop.os.startfile",
            lambda path, operation: open_directory([path, operation]), raising=False)
        monkeypatch.setattr("mfq.server.services.service.subprocess.run", open_directory)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                listing = (await client.get("/api/v1/models/directories", params={"path": str(directory)})).json()
                assert listing["can_open_directory"] is True
                assert listing["can_open_in_finder"] is (platform == "darwin")
                invalid = await client.post("/api/v1/models/directories/open", json={"directory_id": "0" * 32})
                assert invalid.status_code == 404 and not calls
                arbitrary = await client.post("/api/v1/models/directories/open", json={"path": str(directory)})
                assert arbitrary.status_code == 422 and not calls
                response = await client.post("/api/v1/models/directories/open", json={"directory_id": listing["current_id"]})
                assert response.status_code == (200 if failure is None else 503)
                if failure is None:
                    assert response.json() == {"opened": True}
                if platform == "win32":
                    assert calls[0][0] == [str(directory.resolve()), "explore"]
                else:
                    expected = (["/usr/bin/open", "-a", "Finder", str(directory.resolve())] if platform == "darwin"
                        else ["/usr/bin/xdg-open", str(directory.resolve())])
                    assert calls[0][0] == expected
                    assert calls[0][1]["timeout"] == 5 and calls[0][1]["check"] is True
                assert not calls[0][1].get("shell", False)
                assert not (tmp_path / MODEL_FILE_INDEX).exists()
        finally:
            await service.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("platform,display,opener", [
    ("linux", None, "/usr/bin/xdg-open"),
    ("linux", ":0", None),
    ("win32", None, None),
    ("freebsd", None, None),
])
def test_file_manager_is_disabled_without_a_desktop_opener(tmp_path: Path, monkeypatch, platform, display, opener) -> None:
    async def run() -> None:
        catalog = ModelCatalog([tmp_path])
        service = ServerService(SessionStore(tmp_path / "server.sqlite3"), IdleBackend(), catalog=catalog)
        monkeypatch.setattr("mfq.server.services.desktop.sys.platform", platform)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        if display:
            monkeypatch.setenv("DISPLAY", display)
        monkeypatch.delattr("mfq.server.services.desktop.os.startfile", raising=False)
        monkeypatch.setattr("mfq.server.services.desktop.shutil.which", lambda name: opener)
        calls = []
        monkeypatch.setattr("mfq.server.services.desktop.subprocess.run", lambda *args, **kwargs: calls.append(args))
        try:
            listing = await service.model_directories(path=str(tmp_path))
            assert not listing.can_open_directory
            assert not listing.can_open_in_finder
            from mfq.server.protocol.models import OpenModelDirectoryRequest
            from mfq.server.services.service import ServiceError
            with pytest.raises(ServiceError) as error:
                await service.open_model_directory(OpenModelDirectoryRequest(directory_id=listing.current_id))
            assert error.value.status_code == 501
            assert error.value.detail.code == "file_manager_unavailable"
            assert not calls
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


def test_empty_runtime_pool_reports_idle_state(tmp_path: Path, monkeypatch) -> None:
    from mfq.server.runtime import runtime_pool
    topology = {'memory_architecture': 'discrete', 'unified_memory': False,
                'host_memory_total_bytes': 64 << 30, 'device_memory_total_bytes': 24 << 30,
                'prefix_cache_hot_tier': 'vram', 'prefix_cache_cold_tier': 'ssd'}
    monkeypatch.setattr(runtime_pool, 'memory_topology', lambda: topology)
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        pool = RuntimePool(
            ModelCatalog([model_dir]),
            tmp_path / "runtime",
            automatic_memory_budget=False,
        )
        pool.store = SessionStore(tmp_path / "state.sqlite3")

        status = await pool.runtime_status()
        assert status.pop("prefix_cache_total_disk_max_bytes") > 0
        assert status == {
            **topology,
            "mtp_service_enabled": True,
            "prefix_cache_total_hot_bytes": 0,
            "prefix_cache_total_hot_max_bytes": 2 << 30,
            "prefix_cache_total_ram_bytes": 0,
            "prefix_cache_total_ram_max_bytes": 0,
            "prefix_cache_total_disk_bytes": 0,
            "prefix_cache_total_disk_blocks": 0,
            "prefix_cache_budget_failures": 0,
            "runtime_memory_budget_bytes": None,
            "runtime_memory_effective_budget_bytes": None,
            "runtime_memory_budget_mode": "disabled",
            "runtime_memory_committed_bytes": 0,
            "runtime_memory_headroom_bytes": None,
            "runtime_memory_pressure_level": "disabled",
            "runtime_memory_pressure_ratio": None,
            "runtime_memory_host_paging_pressure": False,
            "runtime_memory_shared_cache_reclaims": 0,
            "runtime_memory_shared_cache_released_bytes": 0,
            "runtime_memory_shared_cache_reclaim_failures": 0,
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
        pool = RuntimePool(catalog, tmp_path / "runtime")
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


def test_openai_models_advertises_complete_catalog_models_before_load(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "available.mfq", architecture="qwen35")
        source = tmp_path / "source.mfq"
        _model(source, architecture="qwen35")
        shards = split_mfq(source, model_dir / "incomplete.mfq", split_max_tensors=1)
        shards[1].unlink()

        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime")
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            pool,
            catalog=catalog,
            runtime_manager=pool,
        )
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                runtime_models = await client.get("/api/v1/runtime/models")
                assert runtime_models.json() == {"object": "list", "data": []}

                response = await client.get("/v1/models")
                assert response.status_code == 200
                assert response.json() == {
                    "object": "list",
                    "data": [
                        {
                            "id": "available",
                            "object": "model",
                            "created": 0,
                            "owned_by": "mfq",
                        }
                    ],
                }
        finally:
            await service.aclose()

    asyncio.run(run())


def test_openai_models_keeps_local_catalog_visible_when_runtime_listing_fails(
    tmp_path: Path,
) -> None:
    class UnavailableBackend(IdleBackend):
        async def runtime_models(self) -> dict[str, object]:
            raise BackendError("backend_unavailable", "runtime is offline", retryable=True)

    async def run() -> None:
        _model(tmp_path / "available.mfq", architecture="qwen35")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            UnavailableBackend(),
            catalog=catalog,
        )
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/v1/models")
                assert response.status_code == 200
                assert [item["id"] for item in response.json()["data"]] == ["available"]
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


def test_catalog_exact_path_resolves_before_duplicate_name_enumeration(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        selected = first / "same-name.mfq"
        _model(selected, architecture="selected")
        _model(second / "same-name.mfq", architecture="other")

        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(selected)

        assert artifact.path == selected
        assert artifact.resource.name == "same-name"
        assert artifact.resource.architecture == "selected"
        with pytest.raises(DuplicateModelNameError, match="duplicate catalog model name"):
            await catalog.list()

    asyncio.run(run())


def test_catalog_discovers_a_registered_external_mfq_without_exposing_its_path(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        catalog_root = tmp_path / "catalog"
        catalog_root.mkdir()
        external_root = tmp_path / "private-models"
        external_root.mkdir()
        model = external_root / "manual.mfq"
        _model(model)
        (catalog_root / MODEL_FILE_INDEX).write_text(
            json.dumps({"version": 1, "files": [str(model)]}),
            encoding="utf-8",
        )

        catalog = ModelCatalog([catalog_root], cache_seconds=0)
        result = await catalog.list()

        assert [item.name for item in result.data] == ["manual"]
        assert result.data[0].loadable
        assert str(external_root) not in result.model_dump_json()
        assert (await catalog.resolve("manual")).path == model

    asyncio.run(run())


def test_catalog_registered_shard_resolves_its_complete_external_family(tmp_path: Path) -> None:
    async def run() -> None:
        catalog_root = tmp_path / "catalog"
        catalog_root.mkdir()
        external_root = tmp_path / "private-models"
        external_root.mkdir()
        source = external_root / "source.mfq"
        _model(source)
        shards = split_mfq(source, external_root / "manual.mfq", split_max_tensors=1)
        source.unlink()
        (catalog_root / MODEL_FILE_INDEX).write_text(
            json.dumps({"version": 1, "files": [str(shards[1])]}),
            encoding="utf-8",
        )

        result = await ModelCatalog([catalog_root], cache_seconds=0).list()

        assert [item.name for item in result.data] == ["manual"]
        assert result.data[0].complete
        assert result.data[0].shard_count == 2

    asyncio.run(run())


def test_managed_runtime_retries_timed_out_startup_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mfq.server.runtime.backend import OpenAIChatBackend

    async def run() -> None:
        _model(tmp_path / "tiny.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime.py"
        _fake_runtime(executable, monkeypatch)
        pool = RuntimePool(
            ModelCatalog([tmp_path], cache_seconds=0), executable,
            startup_timeout_seconds=5, automatic_memory_budget=False,
        )
        pool.memory_policy.prefix_directory = str(tmp_path / "prefix")
        original_capabilities = OpenAIChatBackend.capabilities
        probes = 0
        cancelled = False

        async def delayed_capabilities(self, *args, **kwargs):
            nonlocal probes, cancelled
            probes += 1
            if probes == 1:
                try:
                    await asyncio.sleep(2)
                except asyncio.CancelledError:
                    cancelled = True
                    raise
            return await original_capabilities(self, *args, **kwargs)

        monkeypatch.setattr(OpenAIChatBackend, "capabilities", delayed_capabilities)
        context = _TestJobContext()
        try:
            result = await pool.load(context, {"model": "tiny", "context_size": 4096})
            instances = (await pool.instances()).data
            assert cancelled and probes >= 2
            assert len(instances) == 1
            assert instances[0].state == RuntimeInstanceState.READY
            assert str(instances[0].id) == result["instance_id"]
        finally:
            await context.cleanup()
            await pool.aclose()

    asyncio.run(run())


def test_managed_runtime_loads_and_unloads_through_persistent_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "tiny.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(
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
                idle = await client.get("/api/v1/runtime/status")
                assert idle.status_code == 200
                assert idle.json()["runtime_state"] == "idle"
                assert idle.json()["model"] is None
                realtime = await client.get("/api/v1/runtime/realtime/capabilities")
                assert realtime.status_code == 200
                assert realtime.json() == {"available": False, "modes": []}

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
                assert instances.json()["data"][0]["mtp_supported"] is False
                assert instances.json()["data"][0]["mtp_available"] is False

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
                assert (await client.get("/api/v1/runtime/status")).json()["runtime_state"] == "idle"
                assert store.get_job(unload_id).status == JobStatus.SUCCEEDED
        finally:
            await service.aclose()

    asyncio.run(run())


def test_managed_runtime_dispatches_qwen4_mfq_to_native_cpp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model = model_dir / "Qwen3.8-Flash-Next.mfq"
        _model(model, architecture="qwen4_exp-hf-mfq-nint-recipe")
        native_executable = tmp_path / "fake-native-runtime"
        _fake_runtime(native_executable, monkeypatch)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(
            catalog,
            native_executable,
            startup_timeout_seconds=5,
            max_instances=1,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, pool, catalog=catalog, runtime_manager=pool)
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                artifact = (await client.get("/api/v1/models")).json()["data"][0]
                accepted = await client.post(
                    "/api/v1/models/load",
                    json={"model": artifact["name"], "context_size": 8192},
                )
                assert accepted.status_code == 202
                job = await _wait_for_job(client, accepted.json()["operation_id"])
                assert job["status"] == "succeeded", job
                instance_id = job["result"]["instance_id"]
                unload = await client.post(
                    "/api/v1/models/unload",
                    json={"instance_id": instance_id},
                )
                unload_job = await _wait_for_job(client, unload.json()["operation_id"])
                assert unload_job["status"] == "succeeded", unload_job
        finally:
            await service.aclose()

    asyncio.run(run())


def test_managed_cuda_runtime_connects_explicit_request_concurrency(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        model = tmp_path / "tiny.mfq"
        _model(model, architecture="qwen35")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model)
        pool = RuntimePool(
            catalog,
            tmp_path / "mfq-runtime",
            backend="cuda",
            max_requests_per_instance=6,
        )

        command, environment = pool._launch_configuration(
            artifact,
            ModelLoadRequest(model="tiny", prefix_cache_max_sessions=4),
            port=43123,
        )

        assert command[command.index("--continuous-batching") + 1] == "6"
        assert environment["MFQ_RUNTIME_MAX_KV_SESSIONS"] == "4"
        assert "MFQ_SERVER_MAX_KV_SESSIONS" not in environment

    asyncio.run(run())


def test_managed_native_runtime_uses_selected_transport(tmp_path: Path) -> None:
    async def run() -> None:
        model = tmp_path / "tiny.mfq"
        _model(model, architecture="qwen35")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model)
        stdio = RuntimePool(
            catalog,
            tmp_path / "mfq-runtime",
            backend="cuda",
            transport="stdio",
        )
        http = RuntimePool(
            catalog,
            tmp_path / "mfq-runtime",
            backend="cuda",
            transport="http",
        )

        stdio_command, _ = stdio._launch_configuration(
            artifact,
            ModelLoadRequest(model="tiny"),
            port=0,
        )
        http_command, _ = http._launch_configuration(
            artifact,
            ModelLoadRequest(model="tiny"),
            port=43123,
        )

        assert stdio_command[stdio_command.index("--transport") + 1] == "stdio"
        assert "--host" not in stdio_command
        assert "--port" not in stdio_command
        assert http_command[http_command.index("--transport") + 1] == "http"
        assert http_command[http_command.index("--port") + 1] == "43123"

    asyncio.run(run())


def test_managed_native_runtime_leaves_default_prefill_chunk_to_model_autotune(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        model = tmp_path / "tiny.mfq"
        _model(model, architecture="qwen35")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model)
        pool = RuntimePool(catalog, tmp_path / "mfq-decode-metal", backend="metal")

        automatic, _environment = pool._launch_configuration(
            artifact,
            ModelLoadRequest(model="tiny"),
            port=43123,
        )
        explicit, _environment = pool._launch_configuration(
            artifact,
            ModelLoadRequest(model="tiny", prefill_chunk_size=4096),
            port=43124,
        )

        assert "--prefill-chunk-size" not in automatic
        assert explicit[explicit.index("--prefill-chunk-size") + 1] == "4096"

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_managed_native_runtime_leaves_continuous_batching_validation_to_worker(
    tmp_path: Path, backend: str,
) -> None:
    async def run() -> None:
        model = tmp_path / "tiny.mfq"
        _model(model, architecture="qwen4_exp")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model)
        pool = RuntimePool(
            catalog,
            tmp_path / "mfq-runtime",
            backend=backend,
            max_requests_per_instance=6,
        )
        request = ModelLoadRequest(model="tiny")

        dense, _environment = pool._launch_configuration(
            artifact,
            request,
            port=43123,
        )
        moe, _environment = pool._launch_configuration(
            DiscoveredModel(
                resource=artifact.resource,
                path=artifact.path,
                routed_expert_bytes=1,
            ),
            request,
            port=43124,
        )

        assert dense[dense.index("--continuous-batching") + 1] == "6"
        assert moe[moe.index("--continuous-batching") + 1] == "6"

        cached, _environment = pool._launch_configuration(
            DiscoveredModel(
                resource=artifact.resource,
                path=artifact.path,
                routed_expert_bytes=1,
            ),
            request.model_copy(update={"moe_gpu_cache_gb": 1.0}),
            port=43125,
        )
        assert cached[cached.index("--continuous-batching") + 1] == "6"

    asyncio.run(run())


def test_runtime_instances_keep_model_bound_mtp_capabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "plain.mfq", architecture="qwen35")
        _model(model_dir / "ready-with-mtp.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            max_instances=2,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, pool, catalog=catalog, runtime_manager=pool)
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                for model in ("plain", "ready-with-mtp"):
                    accepted = await client.post("/api/v1/models/load", json={"model": model})
                    loaded = await _wait_for_job(client, accepted.json()["operation_id"])
                    assert loaded["status"] == "succeeded", loaded

                instances = {
                    item["model"]: item
                    for item in (await client.get("/api/v1/runtime/instances")).json()["data"]
                }
                assert instances["plain"]["mtp_supported"] is False
                assert instances["plain"]["mtp_available"] is False
                assert instances["ready-with-mtp"]["mtp_supported"] is True
                assert instances["ready-with-mtp"]["mtp_available"] is True
        finally:
            await service.aclose()

    asyncio.run(run())


def test_concurrent_loads_reserve_the_catalog_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "tiny.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(
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
                assert failed["error"]["code"] in {
                    "model_already_loaded",
                    "model_already_loading",
                }
                assert len((await pool.instances()).data) == 1
        finally:
            await service.aclose()

    asyncio.run(run())


def test_failed_runtime_start_does_not_leave_a_stuck_pool_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "tiny.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        executable.write_text(
            "#!/usr/bin/env python3\nraise SystemExit(2)\n",
            encoding="utf-8",
        )
        launch_python_runtime(executable, monkeypatch)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            max_instances=1,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, pool, catalog=catalog, runtime_manager=pool)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                accepted = await client.post(
                    "/api/v1/models/load", json={"model": "tiny"}
                )
                failed = await _wait_for_job(client, accepted.json()["operation_id"])
                assert failed["status"] == "failed", failed
                for _ in range(100):
                    if not (await pool.instances()).data:
                        break
                    await asyncio.sleep(0.01)
                assert (await pool.instances()).data == []

                _fake_runtime(executable, monkeypatch)
                accepted = await client.post(
                    "/api/v1/models/load", json={"model": "tiny"}
                )
                loaded = await _wait_for_job(client, accepted.json()["operation_id"])
                assert loaded["status"] == "succeeded", loaded

    asyncio.run(run())


def test_stale_load_cleanup_cannot_release_a_new_load_reservation() -> None:
    pool = RuntimePool(ModelCatalog([]), "runtime")
    stale_event = asyncio.Event()
    current_event = asyncio.Event()
    pool._load_events["tiny"] = current_event
    pool._loading_model_names.add("tiny")
    pool._loading_artifact_ids["tiny"] = "artifact-current"
    pool._load_ports["tiny"] = 32123
    pool._load_bytes["tiny"] = 4096
    pool._reserved_ports.add(32123)

    pool._finish_model_load_locked("tiny", stale_event, None)

    assert stale_event.is_set()
    assert pool._load_events["tiny"] is current_event
    assert "tiny" in pool._loading_model_names
    assert pool._loading_artifact_ids["tiny"] == "artifact-current"
    assert pool._load_ports["tiny"] == 32123
    assert pool._load_bytes["tiny"] == 4096
    assert 32123 in pool._reserved_ports


def test_runtime_memory_accounting_uses_backend_device_metrics() -> None:
    observed = RuntimePool._observed_runtime_bytes(
        100,
        {
            "mlx_active_bytes": 80,
            "mlx_cache_bytes": 70,
            "cuda_allocated_bytes": 175,
            "cuda_reserved_bytes": 200,
        },
    )
    assert observed == 200
    assert RuntimePool._observed_runtime_bytes(100, {}) == 100
    assert RuntimePool._observed_runtime_bytes(None, {}) is None


def test_runtime_memory_enforcement_evicts_only_idle_unpinned_instances(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _model(tmp_path / "older.mfq")
        _model(tmp_path / "newer.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        older_artifact = await catalog.resolve("older")
        newer_artifact = await catalog.resolve("newer")
        older = _Runtime(
            id=uuid4(),
            artifact=older_artifact,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            resident_bytes=70,
            last_used_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        newer = _Runtime(
            id=uuid4(),
            artifact=newer_artifact,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            resident_bytes=70,
            last_used_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            max_instances=2,
            max_runtime_memory_bytes=100,
        )
        pool._instances = {older.id: older, newer.id: newer}

        async with pool._lock:
            victims = pool._claim_over_budget_instances_for_unload_locked()

        assert victims == [older]
        assert list(pool._instances) == [older.id, newer.id]
        assert older.state == RuntimeInstanceState.UNLOADING
        async with pool._lock:
            assert pool._claim_over_budget_instances_for_unload_locked(
                pending_releases=[older],
            ) == []
        assert newer.state == RuntimeInstanceState.READY

        newer.resident_bytes = 101
        newer.pinned = True
        async with pool._lock:
            assert pool._claim_over_budget_instances_for_unload_locked() == []
        assert list(pool._instances) == [older.id, newer.id]

    asyncio.run(run())


def test_runtime_memory_enforcement_rechecks_when_a_busy_runtime_drains(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        release = asyncio.Event()

        class BlockingBackend(IdleBackend):
            async def stream(self, **_options: object):
                await release.wait()
                yield BackendDelta(content_delta="ok", finish_reason="stop")

        _model(tmp_path / "busy.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve("busy")
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=BlockingBackend(),
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            resident_bytes=101,
            request_slots=_RequestSlotGate(1),
        )
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            max_runtime_memory_bytes=100,
            metric_interval_seconds=60,
        )
        pool._instances[instance.id] = instance
        stopped = asyncio.Event()

        async def stop_process(victim: _Runtime) -> None:
            assert victim is instance
            stopped.set()

        async def refresh_usage(_instance: _Runtime) -> None:
            return None

        pool._stop_process = stop_process  # type: ignore[method-assign]
        pool._refresh_instance_usage = refresh_usage  # type: ignore[method-assign]
        pool._idle_reaper_task = asyncio.create_task(pool._idle_reaper())

        async def consume() -> None:
            async for _ in pool.stream(
                model=artifact.resource.name,
                messages=[{"role": "user", "content": "hello"}],
                sampling=SamplingParams(),
            ):
                pass

        response = asyncio.create_task(consume())
        for _ in range(100):
            if instance.active_requests == 1:
                break
            await asyncio.sleep(0)
        assert instance.active_requests == 1
        assert not stopped.is_set()

        release.set()
        await response
        await asyncio.wait_for(stopped.wait(), timeout=1)
        assert instance.id not in pool._instances
        await pool.aclose()

    asyncio.run(run())


def test_runtime_control_lease_blocks_lru_eviction(tmp_path: Path) -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingBackend(IdleBackend):
            async def clear_runtime_cache(self) -> dict[str, object]:
                started.set()
                await release.wait()
                return {"cleared": True}

        _model(tmp_path / "model.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve("model")
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=BlockingBackend(),
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            last_used_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        pool = RuntimePool(catalog, tmp_path / "runtime", max_instances=1)
        pool._instances[instance.id] = instance
        clearing = asyncio.create_task(pool.clear_runtime_cache(instance.id))
        await asyncio.wait_for(started.wait(), timeout=1)

        async with pool._lock:
            assert instance.control_leases == 1
            assert pool._claim_lru_instance_for_unload_locked() is None

        release.set()
        assert await clearing == {"cleared": True}
        async with pool._lock:
            assert instance.control_leases == 0
            assert pool._claim_lru_instance_for_unload_locked() is instance

    asyncio.run(run())


@pytest.mark.parametrize("read_only", [False, True])
def test_runtime_control_lease_release_survives_caller_cancellation(
    tmp_path: Path,
    read_only: bool,
) -> None:
    async def run() -> None:
        _model(tmp_path / "model.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve("model")
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            control_leases=1,
            read_control_leases=int(read_only),
        )
        pool = RuntimePool(catalog, tmp_path / "runtime")
        pool._instances[instance.id] = instance

        await pool._lock.acquire()
        releasing = asyncio.create_task(pool._release_control_lease(instance, read_only=read_only))
        while not pool._lease_release_tasks:
            await asyncio.sleep(0)
        releasing.cancel()
        pool._lock.release()
        with pytest.raises(asyncio.CancelledError):
            await releasing
        await pool._drain_control_lease_releases()

        assert instance.control_leases == 0
        assert instance.read_control_leases == 0
        assert not pool._lease_release_tasks

    asyncio.run(run())


def test_runtime_memory_enforcement_trims_idle_hot_tiers_before_models(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        class TrimBackend(IdleBackend):
            def __init__(self) -> None:
                self.instance: _Runtime | None = None
                self.targets: list[int] = []

            async def trim_runtime_cache(self, target_bytes: int = 0) -> dict[str, object]:
                self.targets.append(target_bytes)
                assert self.instance is not None
                released = max(0, (self.instance.kv_bytes or 0) - target_bytes)
                self.instance.kv_bytes = target_bytes
                self.instance.resident_bytes = max(
                    0,
                    (self.instance.resident_bytes or 0) - released,
                )
                return {"released_bytes": released, "prefix_cache_hot_bytes": target_bytes}

            async def runtime_status(self) -> dict[str, object]:
                assert self.instance is not None
                return {"prefix_cache_hot_bytes": self.instance.kv_bytes or 0}

        _model(tmp_path / "older.mfq")
        _model(tmp_path / "newer.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        older_backend = TrimBackend()
        newer_backend = TrimBackend()
        older = _Runtime(
            id=uuid4(),
            artifact=await catalog.resolve("older"),
            process=SimpleNamespace(returncode=None),
            backend=older_backend,
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            resident_bytes=70,
            kv_bytes=30,
            last_used_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        newer = _Runtime(
            id=uuid4(),
            artifact=await catalog.resolve("newer"),
            process=SimpleNamespace(returncode=None),
            backend=newer_backend,
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            resident_bytes=70,
            kv_bytes=30,
            last_used_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        older_backend.instance = older
        newer_backend.instance = newer
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            max_instances=2,
            max_runtime_memory_bytes=100,
        )
        pool._instances = {older.id: older, newer.id: newer}

        await pool._trim_idle_prefix_caches_for_budget()
        async with pool._lock:
            victims = pool._claim_over_budget_instances_for_unload_locked()

        assert older_backend.targets == [0]
        assert newer_backend.targets == [20]
        assert older.resident_bytes == 40
        assert newer.resident_bytes == 60
        assert victims == []
        assert set(pool._instances) == {older.id, newer.id}

        older.resident_bytes = 70
        older.kv_bytes = 30
        older_backend.targets.clear()
        pool._instances = {older.id: older}
        await pool._trim_idle_prefix_caches_for_budget(
            additional_bytes=40,
            prospective_model="third",
        )
        assert older_backend.targets == [20]
        assert older.resident_bytes == 60

        older_backend.targets.clear()
        await pool._trim_idle_prefix_caches_for_budget(
            additional_bytes=70,
            prospective_model="older",
        )
        assert older_backend.targets == []

    asyncio.run(run())


def test_runtime_controls_target_the_requested_model_instance(tmp_path: Path) -> None:
    async def run() -> None:
        class ControlBackend:
            def __init__(self, name: str) -> None:
                self.name = name
                self.reloads: list[int] = []
                self.cache_clears = 0
                self.cache_trims: list[int] = []

            async def reload_runtime(self, context_size: int) -> dict[str, object]:
                self.reloads.append(context_size)
                return {"model": self.name, "max_context": context_size}

            async def clear_runtime_cache(self) -> dict[str, object]:
                self.cache_clears += 1
                return {"model": self.name, "released_snapshots": 1}

            async def trim_runtime_cache(self, target_bytes: int = 0) -> dict[str, object]:
                self.cache_trims.append(target_bytes)
                return {"model": self.name, "released_bytes": 2}

            async def capabilities(self) -> RuntimeCapabilitiesResource:
                return RuntimeCapabilitiesResource(
                    model=self.name,
                    model_type="qwen35",
                    model_capabilities=capabilities_for_architecture("qwen35"),
                )

            async def runtime_status(self) -> dict[str, object]:
                return {"model": self.name, "active_requests": 0, "context_capacity": 32768}

        _model(tmp_path / "first.mfq")
        _model(tmp_path / "second.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        first_artifact = await catalog.resolve("first")
        second_artifact = await catalog.resolve("second")
        first_backend = ControlBackend("first")
        second_backend = ControlBackend("second")
        first = _Runtime(
            id=uuid4(),
            artifact=first_artifact,
            process=SimpleNamespace(returncode=None),
            backend=first_backend,  # type: ignore[arg-type]
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
        )
        second = _Runtime(
            id=uuid4(),
            artifact=second_artifact,
            process=SimpleNamespace(returncode=None),
            backend=second_backend,  # type: ignore[arg-type]
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
        )
        pool = RuntimePool(catalog, tmp_path / "runtime", max_instances=2)
        pool._instances = {first.id: first, second.id: second}
        pool._last_instance_id = first.id
        pool._load_requests["second"] = ModelLoadRequest(model="second", context_size=4096)
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            pool,
            catalog=catalog,
            runtime_manager=pool,
        )

        reloaded = await pool.reload_runtime(8192, second.id)
        cleared = await pool.clear_runtime_cache(second.id)
        trimmed = await pool.trim_runtime_cache(4096, second.id)
        capabilities = await service.runtime_capabilities(second.id)
        status = await service.runtime_status(second.id)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            api_capabilities = await client.get(
                "/api/v1/runtime/capabilities",
                params={"instance_id": str(second.id)},
            )
            api_status = await client.get(
                "/api/v1/runtime/status",
                params={"instance_id": str(second.id)},
            )
            missing_status = await client.get(
                "/api/v1/runtime/status",
                params={"instance_id": str(uuid4())},
            )

        assert reloaded["model"] == "second"
        assert cleared["model"] == "second"
        assert trimmed["model"] == "second"
        assert capabilities.model == "second"
        assert status["model"] == "second"
        assert status["instance_id"] == str(second.id)
        assert api_capabilities.json()["model"] == "second"
        assert api_status.json()["model"] == "second"
        assert missing_status.status_code == 404
        assert missing_status.json()["error"]["code"] == "runtime_instance_not_found"
        assert first_backend.reloads == []
        assert first_backend.cache_clears == 0
        assert first_backend.cache_trims == []
        assert second_backend.reloads == [8192]
        assert second_backend.cache_clears == 1
        assert second_backend.cache_trims == [4096]
        assert second.context_size == 8192
        listed = (await pool.instances()).data
        assert len(listed) == 2
        assert all(item.state == RuntimeInstanceState.READY for item in listed)
        assert first.context_size == 4096
        assert second.context_capacity == 32768
        assert pool._load_requests["second"].context_size == 8192
        over_capacity = await pool.reload_runtime(65536, second.id)
        assert over_capacity['warning'] == "context_size_exceeded"
        assert over_capacity['max_context'] == 32768
        assert second_backend.reloads == [8192, 32768]
        with pytest.raises(BackendError) as missing:
            await pool.clear_runtime_cache(uuid4())
        assert missing.value.code == "runtime_instance_not_found"

    asyncio.run(run())


def test_unexpected_runtime_exit_is_contained_until_explicit_retry(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        class ExitedProcess:
            returncode = 9

            async def wait(self) -> int:
                return 9

        class ClosedBackend:
            closed = False

            async def aclose(self) -> None:
                self.closed = True

        _model(tmp_path / "unstable.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve("unstable")
        backend = ClosedBackend()
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=ExitedProcess(),  # type: ignore[arg-type]
            backend=backend,  # type: ignore[arg-type]
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
        )
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            load_failure_cooldown_seconds=60,
        )
        pool._instances[instance.id] = instance

        await pool._monitor(instance)

        assert instance.state == RuntimeInstanceState.FAILED
        assert backend.closed
        with pytest.raises(BackendError) as contained:
            await pool._ensure_model_loaded("unstable")
        assert contained.value.code == "runtime_exited"

    asyncio.run(run())


def test_stream_revives_after_unexpected_exit_within_cooldown(
    tmp_path: Path,
) -> None:
    class RevivedBackend:
        async def aclose(self) -> None:
            return None

        async def stream(self, **_options: object):
            yield BackendDelta(content_delta="revived", finish_reason="stop")

    async def run() -> None:
        _model(tmp_path / "unstable.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve("unstable")
        dead = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(),  # type: ignore[arg-type]
            backend=SimpleNamespace(),  # type: ignore[arg-type]
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.FAILED,
        )
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            load_failure_cooldown_seconds=60,
        )
        pool._instances[dead.id] = dead
        detail = ErrorDetail(
            code="runtime_exited",
            message="runtime process exited with status 9",
            retryable=True,
        )
        pool._load_errors["unstable"] = detail
        pool._load_failures["unstable"] = _CachedLoadFailure(
            artifact_id=artifact.resource.id,
            detail=detail,
            failed_at=time.monotonic(),
        )
        revived = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),  # type: ignore[arg-type]
            backend=RevivedBackend(),  # type: ignore[arg-type]
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            request_slots=_RequestSlotGate(1),
        )
        loads: list[str] = []

        async def fake_load(_context: object, request: object) -> None:
            model = request.get("model") if isinstance(request, dict) else None
            loads.append(str(model))
            pool._instances.pop(dead.id, None)
            pool._instances[revived.id] = revived
            pool._load_failures.pop("unstable", None)
            pool._load_errors.pop("unstable", None)

        pool.load = fake_load  # type: ignore[method-assign]

        deltas = [
            delta
            async for delta in pool.stream(
                model="unstable",
                messages=[{"role": "user", "content": "hello"}],
                sampling=SamplingParams(),
            )
        ]
        assert [delta.content_delta for delta in deltas] == ["revived"]
        assert loads == ["unstable"]

    asyncio.run(run())


def test_stream_revival_is_bounded_to_one_per_cooldown_window() -> None:
    pool = RuntimePool(
        ModelCatalog([], cache_seconds=0),
        "missing-runtime",
        load_failure_cooldown_seconds=60,
    )
    assert pool._begin_runtime_revival("unstable")
    assert not pool._begin_runtime_revival("unstable")

    unbounded = RuntimePool(
        ModelCatalog([], cache_seconds=0),
        "missing-runtime",
        load_failure_cooldown_seconds=0,
    )
    assert unbounded._begin_runtime_revival("unstable")
    assert unbounded._begin_runtime_revival("unstable")


def test_request_driven_load_waiters_receive_the_same_startup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "broken.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import time\n"
            "time.sleep(0.1)\n"
            "raise SystemExit(2)\n",
            encoding="utf-8",
        )
        launch_python_runtime(executable, monkeypatch)
        pool = RuntimePool(
            ModelCatalog([model_dir], cache_seconds=0),
            executable,
            startup_timeout_seconds=5,
            load_failure_cooldown_seconds=60,
        )
        starts = 0
        create_subprocess = asyncio.create_subprocess_exec

        async def counted_create_subprocess(*args: object, **kwargs: object):
            nonlocal starts
            starts += 1
            return await create_subprocess(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", counted_create_subprocess)

        async def consume() -> None:
            async for _delta in pool.stream(
                model="broken",
                messages=[{"role": "user", "content": "hello"}],
                sampling=SamplingParams(),
            ):
                pass

        try:
            results = await asyncio.gather(
                consume(),
                consume(),
                return_exceptions=True,
            )
            assert all(isinstance(item, BackendError) for item in results)
            assert {item.code for item in results if isinstance(item, BackendError)} == {
                "runtime_start_failed"
            }
            assert {
                item.status_code
                for item in results
                if isinstance(item, BackendError)
            } == {503}
            assert starts == 1
            with pytest.raises(BackendError) as cached:
                await consume()
            assert cached.value.code == "runtime_start_failed"
            assert starts == 1
            assert (await pool.instances()).data == []
        finally:
            await pool.aclose()

    asyncio.run(run())


def test_request_driven_model_loads_coalesce_and_restore_exact_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "first.mfq", architecture="qwen35")
        _model(model_dir / "second.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            max_instances=1,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, pool, catalog=catalog, runtime_manager=pool)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            activated = await asyncio.gather(
                pool._ensure_model_loaded("first"),
                pool._ensure_model_loaded("first"),
            )
            assert activated[0] is not None
            assert activated[0] is activated[1]
            assert [item.model for item in (await pool.instances()).data] == ["first"]

            first_id = (await pool.instances()).data[0].id
            unload_context = _TestJobContext()
            await pool.unload(
                unload_context,  # type: ignore[arg-type]
                {"instance_id": str(first_id)},
            )
            await unload_context.cleanup()
            load_context = _TestJobContext()
            await pool.load(
                load_context,  # type: ignore[arg-type]
                {
                    "model": "first",
                    "context_size": 8192,
                    "prefill_chunk_size": 333,
                    "prefix_cache_disk_bytes": 123456,
                    "prefix_cache_block_tokens": 64,
                    "pin": False,
                },
            )
            await load_context.cleanup()
            second_context = _TestJobContext()
            await pool.load(
                second_context,  # type: ignore[arg-type]
                {"model": "second"},
            )
            await second_context.cleanup()
            assert [item.model for item in (await pool.instances()).data] == ["second"]

            assert await pool._ensure_model_loaded("first")
            restored = (await pool.instances()).data
            assert [item.model for item in restored] == ["first"]
            assert restored[0].context_size == 8192
            saved = pool._load_requests["first"]
            assert saved.prefill_chunk_size == 333
            assert saved.prefix_cache_disk_bytes == 123456
            assert saved.prefix_cache_block_tokens == 64

    asyncio.run(run())


def test_named_load_replaces_an_idle_stale_artifact_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model = model_dir / "revision.mfq"
        _model(model, architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        pool = RuntimePool(
            ModelCatalog([model_dir], cache_seconds=0),
            executable,
            startup_timeout_seconds=5,
            max_instances=1,
        )

        try:
            first = await pool._ensure_model_loaded("revision")
            assert first is not None
            first_mtime = first.artifact.resource.modified_at
            stat_result = model.stat()
            os.utime(
                model,
                ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000_000),
            )

            first.active_requests = 1
            with pytest.raises(JobExecutionError) as busy:
                await pool.load(
                    _TestJobContext(),  # type: ignore[arg-type]
                    {"model": "revision"},
                )
            assert busy.value.detail.code == "runtime_revision_busy"
            assert first.process.returncode is None
            first.active_requests = 0

            context = _TestJobContext()
            await pool.load(
                context,  # type: ignore[arg-type]
                {"model": "revision"},
            )
            await context.cleanup()
            second = await pool._select("revision", session_id=None)

            assert second is not None
            assert second.id != first.id
            assert second.artifact.resource.modified_at > first_mtime
            assert first.process.returncode is not None
            instances = (await pool.instances()).data
            assert [instance.id for instance in instances] == [second.id]
        finally:
            await pool.aclose()

    asyncio.run(run())


def test_runtime_pool_evicts_idle_lru_but_preserves_pinned_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "older.mfq", architecture="qwen35")
        _model(model_dir / "newer.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            max_instances=1,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, pool, catalog=catalog, runtime_manager=pool)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                accepted = await client.post(
                    "/api/v1/models/load",
                    json={"model": "older", "pin": False},
                )
                first = await _wait_for_job(client, accepted.json()["operation_id"])
                assert first["status"] == "succeeded", first

                accepted = await client.post(
                    "/api/v1/models/load",
                    json={"model": "newer", "pin": True},
                )
                second = await _wait_for_job(client, accepted.json()["operation_id"])
                assert second["status"] == "succeeded", second
                listed = (await client.get("/api/v1/runtime/instances")).json()["data"]
                assert [item["model"] for item in listed] == ["newer"]
                assert listed[0]["pinned"] is True

                accepted = await client.post(
                    "/api/v1/models/load",
                    json={"model": "older"},
                )
                blocked = await _wait_for_job(client, accepted.json()["operation_id"])
                assert blocked["status"] == "failed", blocked
                assert blocked["error"]["code"] == "runtime_instance_limit"

    asyncio.run(run())


def test_runtime_memory_budget_evicts_idle_models_and_respects_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(root: Path, *, pinned: bool) -> None:
        root.mkdir()
        _model(root / "first.mfq", architecture="qwen35")
        _model(root / "second.mfq", architecture="qwen35")
        executable = root / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([root], cache_seconds=0)
        artifacts = (await catalog.list()).data
        budget = max(item.total_bytes for item in artifacts)
        pool = RuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            max_instances=2,
            max_runtime_memory_bytes=budget,
        )
        first_context = _TestJobContext()
        try:
            await pool.load(
                first_context,  # type: ignore[arg-type]
                {"model": "first", "pin": pinned},
            )
            await first_context.cleanup()
            second_context = _TestJobContext()
            if pinned:
                with pytest.raises(JobExecutionError) as blocked:
                    await pool.load(
                        second_context,  # type: ignore[arg-type]
                        {"model": "second"},
                    )
                assert blocked.value.detail.code == "runtime_memory_limit"
                assert [item.model for item in (await pool.instances()).data] == ["first"]
            else:
                await pool.load(
                    second_context,  # type: ignore[arg-type]
                    {"model": "second"},
                )
                await second_context.cleanup()
                assert [item.model for item in (await pool.instances()).data] == ["second"]
        finally:
            await pool.aclose()

    async def run() -> None:
        await scenario(tmp_path / "evictable", pinned=False)
        await scenario(tmp_path / "pinned", pinned=True)

    asyncio.run(run())


def test_runtime_memory_limit_without_instances_reports_available_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        _model(tmp_path / "small.mfq", architecture="qwen35")
        catalog = ModelCatalog([tmp_path])
        artifact = await catalog.resolve("small")
        size = artifact.resource.total_bytes
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            max_runtime_memory_bytes=size * 2,
        )
        monkeypatch.setattr(pool, "_effective_runtime_memory_budget_locked", lambda: size // 2)

        with pytest.raises(JobExecutionError) as blocked:
            await pool.load(
                _TestJobContext(),  # type: ignore[arg-type]
                {"model": "small"},
            )

        assert blocked.value.detail.code == "runtime_memory_limit"
        assert f"model needs {size:,} B" in blocked.value.detail.message
        assert f"remaining capacity under the runtime budget: {size // 2:,} B" in blocked.value.detail.message
        assert "pinned or busy" not in blocked.value.detail.message

    asyncio.run(run())


def test_failed_load_admission_does_not_partially_retire_lru_models(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _model(tmp_path / "idle.mfq", architecture="qwen35")
        _model(tmp_path / "pinned.mfq", architecture="qwen35")
        _model(tmp_path / "incoming.mfq", architecture="qwen35")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        idle_artifact = await catalog.resolve("idle")
        pinned_artifact = await catalog.resolve("pinned")
        incoming_artifact = await catalog.resolve("incoming")
        unit = incoming_artifact.resource.total_bytes
        idle = _Runtime(
            id=uuid4(),
            artifact=idle_artifact,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=1,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            resident_bytes=unit,
            last_used_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        pinned = _Runtime(
            id=uuid4(),
            artifact=pinned_artifact,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=2,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            resident_bytes=unit,
            last_used_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            pinned=True,
        )
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            max_instances=3,
            max_runtime_memory_bytes=unit + unit // 2,
        )
        pool._instances = {idle.id: idle, pinned.id: pinned}
        session_id = uuid4()
        pool._session_routes[session_id] = idle.id

        with pytest.raises(JobExecutionError) as blocked:
            await pool.load(
                _TestJobContext(),  # type: ignore[arg-type]
                {"model": "incoming"},
            )

        assert blocked.value.detail.code == "runtime_memory_limit"
        assert idle.state == RuntimeInstanceState.READY
        assert pinned.state == RuntimeInstanceState.READY
        assert pool._session_routes[session_id] == idle.id

    asyncio.run(run())


def test_runtime_pool_unloads_an_idle_ttl_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "ephemeral.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            max_instances=1,
            metric_interval_seconds=0.25,
            default_idle_ttl_seconds=0,
        )
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, pool, catalog=catalog, runtime_manager=pool)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                accepted = await client.post(
                    "/api/v1/models/load", json={"model": "ephemeral"},
                )
                loaded = await _wait_for_job(client, accepted.json()["operation_id"])
                assert loaded["status"] == "succeeded", loaded
                for _ in range(80):
                    listed = (await client.get("/api/v1/runtime/instances")).json()["data"]
                    logs = [item for item in store.list_runtime_logs() if item.fields.get("source") == "runtime.lifecycle"]
                    if not listed and logs:
                        break
                    await asyncio.sleep(0.025)
                assert listed == []
                assert len(logs) == 1 and logs[0].fields["model"] == "ephemeral"

    asyncio.run(run())


def test_runtime_pool_close_waits_for_an_inflight_idle_unload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "ephemeral.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        pool = RuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=5,
            max_instances=1,
            metric_interval_seconds=0.25,
        )
        stop_started = asyncio.Event()
        allow_stop = asyncio.Event()
        stop_was_cancelled = False
        original_stop_process = pool._stop_process

        async def controlled_stop(instance: object) -> None:
            nonlocal stop_was_cancelled
            stop_started.set()
            try:
                await allow_stop.wait()
            except asyncio.CancelledError:
                stop_was_cancelled = True
                raise
            await original_stop_process(instance)  # type: ignore[arg-type]

        pool._stop_process = controlled_stop  # type: ignore[method-assign]
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, pool, catalog=catalog, runtime_manager=pool)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                accepted = await client.post(
                    "/api/v1/models/load",
                    json={"model": "ephemeral", "idle_ttl_seconds": 0},
                )
                loaded = await _wait_for_job(client, accepted.json()["operation_id"])
                assert loaded["status"] == "succeeded", loaded
                await asyncio.wait_for(stop_started.wait(), timeout=2)

                closing = asyncio.create_task(pool.aclose())
                await asyncio.sleep(0)
                assert not closing.done()
                assert not stop_was_cancelled
                allow_stop.set()
                await asyncio.wait_for(closing, timeout=5)
                assert not stop_was_cancelled

    asyncio.run(run())


def test_runtime_stop_closes_backend_after_process_control_failure(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _model(tmp_path / "model.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve("model")

        class ClosingBackend(IdleBackend):
            closed = False

            async def aclose(self) -> None:
                self.closed = True

        class BrokenProcess:
            returncode = None

            def terminate(self) -> None:
                raise RuntimeError("terminate failed")

        backend = ClosingBackend()
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=BrokenProcess(),  # type: ignore[arg-type]
            backend=backend,
            port=0,
            context_size=4096,
        )
        pool = RuntimePool(catalog, tmp_path / "runtime")
        with pytest.raises(RuntimeError, match="terminate failed"):
            await pool._stop_process(instance)
        assert backend.closed

    asyncio.run(run())


def test_runtime_stop_uses_kill_after_terminate_failure(tmp_path: Path) -> None:
    async def run() -> None:
        _model(tmp_path / "model.mfq")
        artifact = await ModelCatalog(
            [tmp_path], cache_seconds=0
        ).resolve("model")

        class ClosingBackend(IdleBackend):
            closed = False

            async def aclose(self) -> None:
                self.closed = True

        class RecoverableProcess:
            returncode = None
            killed = False

            def terminate(self) -> None:
                raise RuntimeError("terminate failed")

            def kill(self) -> None:
                self.killed = True

            async def wait(self) -> int:
                assert self.killed
                self.returncode = -9
                return self.returncode

        process = RecoverableProcess()
        backend = ClosingBackend()
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=process,  # type: ignore[arg-type]
            backend=backend,  # type: ignore[arg-type]
            port=0,
            context_size=4096,
        )
        pool = RuntimePool(
            ModelCatalog([tmp_path], cache_seconds=0),
            tmp_path / "runtime",
        )

        await pool._stop_process(instance)

        assert process.killed
        assert process.returncode == -9
        assert backend.closed

    asyncio.run(run())


def test_runtime_retirement_retries_a_failed_process_automatically(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _model(tmp_path / "model.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve("model")
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            resident_bytes=123,
        )
        pool = RuntimePool(catalog, tmp_path / "runtime")
        pool._instances[instance.id] = instance
        session_id = uuid4()
        pool._session_routes[session_id] = instance.id
        pool._last_instance_id = instance.id
        attempts = 0

        async def stop_process(candidate: _Runtime) -> None:
            nonlocal attempts
            assert candidate is instance
            attempts += 1
            if attempts == 1:
                raise RuntimeError("process is still alive")

        pool._stop_process = stop_process  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="still alive"):
            await pool._retire_instance(instance)

        listed = (await pool.instances()).data
        assert len(listed) == 1
        assert listed[0].id == instance.id
        assert listed[0].state == RuntimeInstanceState.UNLOADING
        assert listed[0].error is not None
        assert listed[0].error.code == "runtime_unload_failed"
        assert pool._committed_pool_bytes_locked() == 123
        assert session_id not in pool._session_routes
        assert pool._last_instance_id is None

        for _ in range(100):
            if instance.id not in pool._instances:
                break
            await asyncio.sleep(0.01)

        assert attempts == 2
        assert instance.id not in pool._instances
        assert instance.retirement_retry_task is not None
        await instance.retirement_retry_task
        assert instance.retirement_retry_task.done()
        await pool.aclose()

    asyncio.run(run())


def test_runtime_pool_close_attempts_every_instance_before_reporting_error(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _model(tmp_path / "first.mfq")
        _model(tmp_path / "second.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        first_artifact = await catalog.resolve("first")
        second_artifact = await catalog.resolve("second")

        class ClosingBackend(IdleBackend):
            closed = False

            async def aclose(self) -> None:
                self.closed = True

        fallback = ClosingBackend()
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            fallback=fallback,
        )
        first = _Runtime(
            id=uuid4(),
            artifact=first_artifact,
            process=SimpleNamespace(returncode=0),
            backend=IdleBackend(),
            port=0,
            context_size=4096,
        )
        second = _Runtime(
            id=uuid4(),
            artifact=second_artifact,
            process=SimpleNamespace(returncode=0),
            backend=IdleBackend(),
            port=0,
            context_size=4096,
        )
        pool._instances = {first.id: first, second.id: second}
        attempts = []

        async def stop(instance: _Runtime) -> None:
            attempts.append(instance.id)
            if instance is first:
                raise RuntimeError("first stop failed")

        pool._stop_process = stop  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="first stop failed"):
            await pool.aclose()
        assert attempts == [first.id, second.id]
        assert fallback.closed
        assert pool._instances == {}

    asyncio.run(run())


def test_runtime_pool_close_retires_instances_concurrently(tmp_path: Path) -> None:
    async def run() -> None:
        _model(tmp_path / "first.mfq")
        _model(tmp_path / "second.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifacts = [
            await catalog.resolve("first"),
            await catalog.resolve("second"),
        ]
        pool = RuntimePool(catalog, tmp_path / "runtime")
        instances = [
            _Runtime(
                id=uuid4(),
                artifact=artifact,
                process=SimpleNamespace(returncode=0),
                backend=IdleBackend(),
                port=0,
                context_size=4096,
            )
            for artifact in artifacts
        ]
        pool._instances = {instance.id: instance for instance in instances}
        both_started = asyncio.Event()
        started: set[UUID] = set()

        async def stop(instance: _Runtime) -> None:
            started.add(instance.id)
            if len(started) == len(instances):
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)

        pool._stop_process = stop  # type: ignore[method-assign]
        await pool.aclose()

        assert started == {instance.id for instance in instances}
        assert pool._instances == {}

    asyncio.run(run())


def test_process_resident_bytes_reads_the_current_process() -> None:
    assert RuntimePool._process_resident_bytes(os.getpid()) > 0
    assert RuntimePool._process_resident_bytes(-1) is None


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
        pool = RuntimePool(catalog, tmp_path / "runtime", max_instances=1)
        instance_id = pool.register_started(
            artifact=artifact,
            process=process,
            backend=IdleBackend(),
            port=43123,
            load_request=ModelLoadRequest(
                model="initial",
                context_size=8192,
                prefill_chunk_size=333,
                moe_gpu_cache_gb=0,
                prefix_cache_disk_bytes=1234,
                idle_ttl_seconds=120,
                pin=True,
            ),
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
                assert instances[0]["idle_ttl_seconds"] == 120
                assert instances[0]["pinned"] is True
                assert instances[0]["resident_bytes"] > 0

                models = await client.get("/api/v1/runtime/models")
                assert models.status_code == 200
                visible = models.json()["data"]
                assert len(visible) == 1
                assert visible[0]["id"] == "initial"
                assert visible[0]["id"] != str(instances[0]["id"])
                assert visible[0]["instance_id"] == str(instance_id)
                assert visible[0]["instance_id"] == str(instances[0]["id"])
                remembered = pool._load_requests["initial"]
                assert remembered.prefill_chunk_size == 333
                assert remembered.moe_gpu_cache_gb == 0
                assert remembered.prefix_cache_disk_bytes == 1234
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
        pool = RuntimePool(catalog, tmp_path / "runtime", max_instances=1)
        instance_id = pool.register_started(
            artifact=artifact,
            process=process,
            backend=IdleBackend(),
            port=43123,
            load_request=ModelLoadRequest(model="catalog-name", context_size=8192),
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
                    load_request=ModelLoadRequest(
                        model="catalog-name",
                        context_size=8192,
                    ),
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
        pool = RuntimePool(catalog, tmp_path / "runtime", max_instances=1)
        instance_id = pool.register_started(
            artifact=artifact,
            process=process,
            backend=IdleBackend(),
            port=43123,
            load_request=ModelLoadRequest(model="initial", context_size=8192),
        )
        session_id = uuid4()
        pool._session_routes[session_id] = instance_id
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
            await pool._instances[instance_id].monitor_task
            logs = store.list_runtime_logs(instance_id=instance_id)
            assert len(logs) == 1 and logs[0].fields["model"] == "initial"
            assert session_id not in pool._session_routes
            assert await pool._select("initial", session_id=session_id) is None
            assert (await pool.runtime_status())["runtime_state"] == "idle"

    asyncio.run(run())


def test_runtime_selection_does_not_route_a_session_to_the_wrong_model(
    tmp_path: Path,
) -> None:
    class TrackingBackend(IdleBackend):
        def __init__(self) -> None:
            self.closed_sessions: list[object] = []

        async def close_session(self, session_id: object) -> bool:
            self.closed_sessions.append(session_id)
            return True

    async def run() -> None:
        first_path = tmp_path / "first.mfq"
        second_path = tmp_path / "second.mfq"
        _model(first_path)
        _model(second_path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        first_artifact = await catalog.resolve_path(first_path)
        second_artifact = await catalog.resolve_path(second_path)
        pool = RuntimePool(catalog, tmp_path / "runtime", max_instances=2)
        first_backend = TrackingBackend()
        first = _Runtime(
            id=uuid4(),
            artifact=first_artifact,
            process=SimpleNamespace(returncode=None),
            backend=first_backend,
            port=1,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            request_slots=_RequestSlotGate(1),
        )
        second = _Runtime(
            id=uuid4(),
            artifact=second_artifact,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=2,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            request_slots=_RequestSlotGate(1),
        )
        pool._instances = {first.id: first, second.id: second}
        session_id = uuid4()
        pool._session_routes[session_id] = first.id

        assert await pool._select(first_artifact.resource.id, session_id=session_id) is first
        assert pool._session_routes[session_id] == first.id
        assert first_backend.closed_sessions == []

        assert await pool._select(second_artifact.resource.id, session_id=session_id) is second
        assert session_id not in pool._session_routes
        assert first_backend.closed_sessions == [session_id]

        first.state = RuntimeInstanceState.FAILED
        pool._last_instance_id = first.id
        assert pool._current_instance_locked() is second

    asyncio.run(run())


def test_artifact_id_request_waits_on_the_canonical_model_load(
    tmp_path: Path,
) -> None:
    class StreamingBackend(IdleBackend):
        async def stream(self, **_options: object):
            yield BackendDelta(content_delta="ok", finish_reason="stop")

    async def run() -> None:
        model_path = tmp_path / "loading.mfq"
        _model(model_path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model_path)
        pool = RuntimePool(catalog, tmp_path / "runtime")
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=StreamingBackend(),
            port=1,
            context_size=4096,
            state=RuntimeInstanceState.LOADING,
            request_slots=_RequestSlotGate(1),
        )
        pool._instances[instance.id] = instance
        waited: list[str] = []

        async def wait_for_model_ready(model: str) -> bool:
            waited.append(model)
            instance.state = RuntimeInstanceState.READY
            return True

        pool._wait_for_model_ready = wait_for_model_ready  # type: ignore[method-assign]
        deltas = [
            delta
            async for delta in pool.stream(
                model=artifact.resource.id,
                messages=({"role": "user", "content": "hello"},),
                sampling=SamplingParams(),
            )
        ]

        assert waited == [artifact.resource.name]
        assert [delta.content_delta for delta in deltas] == ["ok"]

    asyncio.run(run())


def test_runtime_preflight_rejects_an_exited_worker_before_streaming(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        model_path = tmp_path / "exited.mfq"
        _model(model_path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model_path)
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=9),
            backend=IdleBackend(),
            port=1,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            request_slots=_RequestSlotGate(1),
        )
        pool = RuntimePool(catalog, tmp_path / "runtime")
        pool._instances[instance.id] = instance

        with pytest.raises(BackendError) as unavailable:
            await pool.preflight(
                model=artifact.resource.name,
                messages=({"role": "user", "content": "hello"},),
                sampling=SamplingParams(),
            )

        assert unavailable.value.code == "model_not_ready"
        assert unavailable.value.status_code == 503

    asyncio.run(run())


def test_runtime_sampling_defaults_apply_only_to_omitted_request_fields(
    tmp_path: Path,
) -> None:
    class SamplingBackend(IdleBackend):
        def __init__(self) -> None:
            self.sampling: list[SamplingParams] = []

        async def stream(self, **options: object):
            self.sampling.append(options["sampling"])  # type: ignore[arg-type]
            yield BackendDelta(content_delta="ok", finish_reason="stop")

    async def run() -> None:
        model_path = tmp_path / "defaults.mfq"
        _model(model_path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve_path(model_path)
        backend = SamplingBackend()
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=backend,
            port=1,
            context_size=4096,
            sampling_defaults=SamplingParams(enable_mtp=False, temperature=0.4),
            state=RuntimeInstanceState.READY,
            request_slots=_RequestSlotGate(1),
        )
        pool = RuntimePool(catalog, tmp_path / "runtime")
        pool._instances[instance.id] = instance

        async def consume(sampling: SamplingParams) -> None:
            async for _delta in pool.stream(
                model=artifact.resource.name,
                messages=({"role": "user", "content": "hello"},),
                sampling=sampling,
            ):
                pass

        await consume(SamplingParams())
        await consume(SamplingParams(enable_mtp=True))

        assert backend.sampling[0].enable_mtp is False
        assert backend.sampling[0].temperature == 0.4
        assert backend.sampling[1].enable_mtp is True
        assert backend.sampling[1].temperature == 0.4

    asyncio.run(run())


def test_request_driven_load_rejects_a_changed_artifact_behind_a_loaded_name(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        model = tmp_path / "shared.mfq"
        _model(model, architecture="qwen35")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        original = await catalog.resolve((await catalog.list()).data[0].id)
        instance = _Runtime(
            id=uuid4(),
            artifact=original,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=1,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            request_slots=_RequestSlotGate(1),
        )
        pool = RuntimePool(catalog, tmp_path / "runtime", max_instances=2)
        pool._instances[instance.id] = instance

        _model(model, architecture="minicpmo45")
        changed = (await catalog.list(refresh=True)).data[0]
        assert changed.id != original.resource.id
        with pytest.raises(BackendError) as conflict:
            await pool._ensure_model_loaded(changed.id)
        assert conflict.value.code == "model_name_conflict"
        assert conflict.value.status_code == 409

    asyncio.run(run())


def test_managed_runtime_reports_and_bounds_queued_requests(tmp_path: Path) -> None:
    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        class BlockingBackend:
            def __init__(self) -> None:
                self.cancelled: list[object] = []

            async def stream(self, **options: object):
                seen_models.append(str(options["model"]))
                entered.set()
                await release.wait()
                yield BackendDelta(content_delta="ok", finish_reason="stop")

            async def cancel_response(self, session_id: object) -> bool:
                self.cancelled.append(session_id)
                return True

        seen_models: list[str] = []
        model = tmp_path / "tiny.mfq"
        _model(model)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve((await catalog.list()).data[0].id)
        blocking_backend = BlockingBackend()
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=blocking_backend,
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            request_slots=_RequestSlotGate(1),
        )
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            max_queued_requests_per_instance=1,
        )
        pool._instances[instance.id] = instance
        assert await pool._select(artifact.resource.name, session_id=None) is instance
        assert await pool._select(artifact.resource.id, session_id=None) is instance

        async def consume(model: str = artifact.resource.name) -> None:
            async for _ in pool.stream(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                sampling=SamplingParams(),
                session_id=session_id,
            ):
                pass

        session_id = uuid4()
        first = asyncio.create_task(consume(artifact.resource.id))
        await entered.wait()
        assert await pool.cancel_response(session_id)
        assert blocking_backend.cancelled == [session_id]
        second = asyncio.create_task(consume())
        await asyncio.sleep(0)
        assert instance.active_requests == 1
        assert instance.queued_requests == 1
        listed = await pool.instances()
        assert listed.data[0].queued_requests == 1
        with pytest.raises(BackendError) as rejected:
            await consume()
        assert rejected.value.code == "runtime_queue_full"
        assert rejected.value.status_code == 429
        assert rejected.value.retryable
        assert instance.queued_requests == 1
        release.set()
        await asyncio.gather(first, second)
        assert seen_models == [artifact.resource.name, artifact.resource.name]
        assert instance.active_requests == 0
        assert instance.queued_requests == 0

    asyncio.run(run())


def test_managed_runtime_closes_backend_stream_when_consumer_stops(
    tmp_path: Path,
) -> None:
    class InterruptibleBackend(IdleBackend):
        def __init__(self) -> None:
            self.stream_closed = False

        async def stream(self, **_options: object):
            try:
                yield BackendDelta(content_delta="partial")
                await asyncio.Event().wait()
            finally:
                self.stream_closed = True

    async def run() -> None:
        model = tmp_path / "interruptible.mfq"
        _model(model)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve((await catalog.list()).data[0].id)
        backend = InterruptibleBackend()
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=backend,
            port=0,
            context_size=4096,
            state=RuntimeInstanceState.READY,
            request_slots=_RequestSlotGate(1),
        )
        pool = RuntimePool(catalog, tmp_path / "runtime")
        pool._instances[instance.id] = instance

        stream = pool.stream(
            model=artifact.resource.name,
            messages=[{"role": "user", "content": "hello"}],
            sampling=SamplingParams(),
        )
        assert (await anext(stream)).content_delta == "partial"
        assert instance.active_requests == 1
        await stream.aclose()

        assert backend.stream_closed
        assert instance.active_requests == 0
        assert instance.state == RuntimeInstanceState.READY
        assert instance.request_slots is not None
        assert instance.request_slots.available == 1

    asyncio.run(run())


def test_runtime_instance_policy_updates_without_reload(tmp_path: Path) -> None:
    async def run() -> None:
        model = tmp_path / "policy.mfq"
        _model(model)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve((await catalog.list()).data[0].id)
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=IdleBackend(),
            port=0,
            context_size=4096,
            idle_ttl_seconds=300,
            state=RuntimeInstanceState.READY,
            request_slots=_RequestSlotGate(1),
        )
        pool = RuntimePool(catalog, tmp_path / "runtime")
        pool._instances[instance.id] = instance
        pool._load_requests[artifact.resource.name] = ModelLoadRequest(
            model=artifact.resource.name,
            idle_ttl_seconds=300,
        )
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            pool,
            catalog=catalog,
            runtime_manager=pool,
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pinned = await client.patch(
                f"/api/v1/runtime/instances/{instance.id}",
                json={"pinned": True},
            )
            assert pinned.status_code == 200
            assert pinned.json()["pinned"] is True
            assert pinned.json()["idle_ttl_seconds"] == 300
            assert pool._load_requests[artifact.resource.name].pin is True

            unpinned = await client.patch(
                f"/api/v1/runtime/instances/{instance.id}",
                json={"pinned": False, "idle_ttl_seconds": None},
            )
            assert unpinned.status_code == 200
            assert unpinned.json()["pinned"] is False
            assert unpinned.json()["idle_ttl_seconds"] is None
            remembered = pool._load_requests[artifact.resource.name]
            assert remembered.pin is False
            assert remembered.idle_ttl_seconds is None

            assert (
                await client.patch(
                    f"/api/v1/runtime/instances/{instance.id}", json={}
                )
            ).status_code == 422
            assert (
                await client.patch(
                    f"/api/v1/runtime/instances/{instance.id}",
                    json={"pinned": None},
                )
            ).status_code == 422
            missing = await client.patch(
                f"/api/v1/runtime/instances/{uuid4()}",
                json={"pinned": True},
            )
            assert missing.status_code == 404
            assert missing.json()["error"]["code"] == "runtime_instance_not_found"

    asyncio.run(run())


def test_minicpmo_voice_component_activates_in_the_managed_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        import mfq.runtime.minicpmo45_realtime as realtime

        model = tmp_path / "voice.mfq"
        _model(model, architecture="minicpmo45")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve((await catalog.list()).data[0].id)

        class Component:
            root = tmp_path / "voice-component"

            @staticmethod
            def ready() -> bool:
                return True

        class Gateway:
            def __init__(self, backend_url, assets, *, token2wav_steps):
                self.arguments = (backend_url, assets, token2wav_steps)
                self.served = []

            async def capabilities(self):
                return {"available": True, "output": ["text", "audio"]}

            async def serve(self, client):
                self.served.append(client)

        monkeypatch.setattr(realtime, "RealtimeGateway", Gateway)
        monkeypatch.setattr(realtime, "_backend_token2wav_steps", lambda *_args: 10)
        backend = SimpleNamespace(base_url="http://127.0.0.1:43123")
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=SimpleNamespace(returncode=None),
            backend=backend,
            port=43123,
            context_size=4096,
            state=RuntimeInstanceState.READY,
        )
        pool = RuntimePool(
            catalog,
            tmp_path / "runtime",
            voice_component=Component(),
        )
        pool._instances[instance.id] = instance
        pool._last_instance_id = instance.id

        assert (await pool.enable_realtime())["active"] is True
        assert (await pool.realtime_capabilities())["available"] is True
        client = object()
        assert await pool.realtime_serve(client)
        assert instance.realtime_gateway.served == [client]

    asyncio.run(run())


def test_runtime_pool_drains_startup_background_tasks_on_close(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        started = asyncio.Event()
        finished = asyncio.Event()
        release = asyncio.Event()

        class Component:
            @staticmethod
            def ready() -> bool:
                return True

        pool = RuntimePool(
            ModelCatalog([tmp_path], cache_seconds=0),
            tmp_path / "runtime",
            voice_component=Component(),
        )

        async def enable_realtime(
            _instance_id: UUID | None = None,
        ) -> dict[str, object]:
            started.set()
            try:
                await release.wait()
                return {"active": True}
            finally:
                finished.set()

        pool.enable_realtime = enable_realtime  # type: ignore[method-assign]
        await pool.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        await pool.aclose()

        assert finished.is_set()
        assert not pool._background_tasks

    asyncio.run(run())
