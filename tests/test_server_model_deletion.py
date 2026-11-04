from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.server.api import create_app
from mfq.server.protocol.models import RuntimeInstanceState
from mfq.server.runtime.backend import BackendError
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.services.model_deletion import ModelDeletionError, delete_model_files, model_deletion_plan
from mfq.server.services.service import ServerService
from mfq.server.services.jobs import JobExecutionError
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore
from tests.platform_helpers import symlink_or_skip


def checkpoint(path: Path) -> None:
    save(path, FileHeader(version=2, model_arch="test"), {"weight": np.zeros((2, 2), dtype=np.float16)})


def test_delete_requires_confirmation_and_preserves_neighbours(tmp_path):
    async def run():
        target, other = tmp_path / "target.mfq", tmp_path / "other.mfq"
        checkpoint(target)
        checkpoint(other)
        catalog = ModelCatalog([tmp_path])
        pool = RuntimePool(catalog, tmp_path / "runtime")
        service = ServerService(SessionStore(tmp_path / "db.sqlite3"), pool, catalog=catalog, runtime_manager=pool)
        artifact = await catalog.resolve("target")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service=service)), base_url="http://test") as client:
            base = f"/api/v1/models/{artifact.resource.id}"
            plan = (await client.get(base + "/deletion")).json()
            assert plan["files"] == [str(target)]
            assert target.exists()
            assert (await client.post(base + "/delete", json={})).status_code == 422
            assert (await client.post(base + "/delete", json={"token": "0" * 64})).status_code == 409
            assert target.exists()
            deleted = await client.post(base + "/delete", json={"token": plan["token"]})
            assert deleted.status_code == 200, deleted.text
            assert deleted.json()["files"] == 1
            assert not target.exists()
            assert other.exists()
            assert [x["name"] for x in (await client.get("/api/v1/models")).json()["data"]] == ["other"]
            assert (await client.post(base + "/delete", json={"token": plan["token"]})).status_code == 404
    asyncio.run(run())


def test_incomplete_shards_are_one_delete_target(tmp_path):
    async def run():
        shards = [tmp_path / f"model-{n:05d}-of-00003.mfq" for n in (1, 3)]
        for shard in shards:
            checkpoint(shard)
        neighbour = tmp_path / "model-other.mfq"
        checkpoint(neighbour)
        artifact = await ModelCatalog([tmp_path]).resolve("model")
        plan = model_deletion_plan(artifact)
        assert set(plan.files) == {str(p) for p in shards}
        delete_model_files(artifact, plan.token)
        assert all(not p.exists() for p in shards)
        assert neighbour.exists()
    asyncio.run(run())


def test_changed_files_do_not_match_the_confirmation(tmp_path):
    async def run():
        target = tmp_path / "model.mfq"
        checkpoint(target)
        artifact = await ModelCatalog([tmp_path]).resolve("model")
        plan = model_deletion_plan(artifact)
        target.write_bytes(target.read_bytes() + b"changed")
        with pytest.raises(ModelDeletionError, match="model_delete_changed"):
            delete_model_files(artifact, plan.token)
        assert target.exists()
    asyncio.run(run())


def test_hf_delete_leaves_nonmodel_files_and_symlink_targets(tmp_path):
    async def run():
        folder = tmp_path / "hf"
        folder.mkdir()
        config = folder / "config.json"
        config.write_text(json.dumps({"model_type": "qwen3"}))
        shared = tmp_path / "shared-weights"
        shared.write_bytes(b"shared")
        weights = folder / "model.safetensors"
        symlink_or_skip(weights, shared)
        keep = folder / "notes.txt"
        keep.write_text("keep")
        artifact = await ModelCatalog([tmp_path]).resolve("hf")
        plan = model_deletion_plan(artifact)
        delete_model_files(artifact, plan.token)
        assert not config.exists() and not weights.is_symlink()
        assert shared.read_bytes() == b"shared"
        assert keep.read_text() == "keep"
    asyncio.run(run())


def test_hf_index_cannot_delete_outside_the_checkpoint(tmp_path):
    async def run():
        folder = tmp_path / "hf"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "../other.safetensors"}}))
        other = tmp_path / "other.safetensors"
        other.write_bytes(b"keep")
        artifact = await ModelCatalog([tmp_path]).resolve("hf")
        with pytest.raises(ModelDeletionError, match="model_delete_unsafe"):
            model_deletion_plan(artifact)
        assert other.read_bytes() == b"keep"
    asyncio.run(run())


@pytest.mark.parametrize("state", [RuntimeInstanceState.READY, RuntimeInstanceState.BUSY, RuntimeInstanceState.LOADING, RuntimeInstanceState.UNLOADING])
def test_loaded_models_are_protected(tmp_path, state):
    async def run():
        target = tmp_path / "model.mfq"
        checkpoint(target)
        catalog = ModelCatalog([tmp_path])
        artifact = await catalog.resolve("model")
        pool = RuntimePool(catalog, tmp_path / "runtime")
        pool._instances[uuid4()] = SimpleNamespace(artifact=artifact, state=state)
        with pytest.raises(BackendError, match="Unload the model"):
            async with pool.model_deletion_guard(artifact):
                pytest.fail("loaded model may not be deleted")
        assert target.exists()
    asyncio.run(run())


def test_loading_and_concurrent_deletions_are_protected(tmp_path):
    async def run():
        target = tmp_path / "model.mfq"
        checkpoint(target)
        catalog = ModelCatalog([tmp_path])
        artifact = await catalog.resolve("model")
        pool = RuntimePool(catalog, tmp_path / "runtime")
        pool._loading_model_names.add("model")
        with pytest.raises(BackendError):
            async with pool.model_deletion_guard(artifact):
                pytest.fail("loading model may not be deleted")
        pool._loading_model_names.clear()
        async with pool.model_deletion_guard(artifact):
            with pytest.raises(BackendError):
                async with pool.model_deletion_guard(artifact):
                    pytest.fail("duplicate deletion may not start")
            with pytest.raises(JobExecutionError) as blocked:
                await pool.load(SimpleNamespace(job_id=uuid4()), {"model": "model"})
            assert blocked.value.detail.code == "model_delete_busy"
        assert not pool._deleting_model_paths
    asyncio.run(run())


def test_active_download_blocks_deletion(tmp_path):
    async def run():
        target = tmp_path / "model.mfq"
        checkpoint(target)
        catalog = ModelCatalog([tmp_path])
        artifact = await catalog.resolve("model")
        pool = RuntimePool(catalog, tmp_path / "runtime")
        store = SessionStore(tmp_path / "db.sqlite3")
        store.create_job("download.modelscope", {"destination": "models"})
        tools = SimpleNamespace(download_request=lambda *_: (None, tmp_path))
        service = ServerService(store, pool, catalog=catalog, runtime_manager=pool, tool_handlers=tools)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service=service)), base_url="http://test") as client:
            base = f"/api/v1/models/{artifact.resource.id}"
            plan = (await client.get(base + "/deletion")).json()
            response = await client.post(base + "/delete", json={"token": plan["token"]})
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "model_delete_downloading"
            assert target.exists()
    asyncio.run(run())
