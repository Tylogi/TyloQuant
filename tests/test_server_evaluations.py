from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from mfq.server.api import create_app
from mfq.server.service import ServerService
from mfq.server.storage import SCHEMA_VERSION, SessionStore
from tests.test_server_service import FakeBackend


class WorkspaceTools:
    def __init__(self, root: Path) -> None:
        self.root = root

    def workspace_file_manifest(self, uri: str) -> dict[str, object]:
        import hashlib

        data = (self.root / uri.removeprefix("workspace://")).read_bytes()
        return {"sha256": hashlib.sha256(data).hexdigest(), "byte_size": len(data)}


def test_dataset_registry_and_matching_evaluation_comparison(tmp_path: Path) -> None:
    async def run() -> None:
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("reproducible corpus", encoding="utf-8")
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(
            store,
            FakeBackend(),
            tool_handlers=WorkspaceTools(tmp_path),
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/v1/datasets",
                json={
                    "name": "WikiText fixture",
                    "kind": "wikitext2",
                    "artifact_uri": "workspace://corpus.txt",
                    "source_uri": "huggingface://Salesforce/wikitext",
                    "revision": "fixture",
                },
            )
            assert created.status_code == 201
            dataset_id = created.json()["id"]
            assert (await client.get("/api/v1/datasets")).json()["data"][0]["byte_size"] == 19

            comparison_key = "a" * 64
            evaluation_ids = []
            for model, ppl in (("base", 10.0), ("quantized", 11.5)):
                job = store.create_job("evaluate.perplexity", {"model": model})
                evaluation = store.record_evaluation(
                    job_id=job.id,
                    kind="perplexity",
                    model_id=model,
                    metrics={"perplexity": ppl, "tokens_per_second": 100.0 / ppl},
                    parameters={"context_size": 512},
                    dataset_id=__import__("uuid").UUID(dataset_id),
                    dataset_manifest={"sha256": created.json()["sha256"]},
                    hardware_identity={"machine": "test"},
                    runtime_identity={"build": "test"},
                    comparison_key=comparison_key,
                )
                evaluation_ids.append(str(evaluation.id))

            compared = await client.post(
                "/api/v1/evaluations/compare",
                json={"evaluation_ids": evaluation_ids},
            )
            assert compared.status_code == 200
            assert compared.json()["rows"][1]["deltas"]["perplexity"] == 1.5
            assert compared.json()["rows"][1]["ratios"]["perplexity"] == 1.15

            other_job = store.create_job("benchmark.kernel", {})
            other = store.record_evaluation(
                job_id=other_job.id,
                kind="kernel_benchmark",
                model_id="base",
                metrics={"latency_ms": 1.0},
                parameters={},
                dataset_id=None,
                dataset_manifest={},
                hardware_identity={},
                runtime_identity={},
                comparison_key="b" * 64,
            )
            rejected = await client.post(
                "/api/v1/evaluations/compare",
                json={"evaluation_ids": [evaluation_ids[0], str(other.id)]},
            )
            assert rejected.status_code == 409
            assert rejected.json()["error"]["code"] == "evaluations_not_comparable"

            assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 204

    asyncio.run(run())


def test_evaluation_schema_migrates_from_previous_version(tmp_path: Path) -> None:
    database = tmp_path / "mfq.server.sqlite3"
    store = SessionStore(database)
    with store._connection() as connection:
        connection.execute("UPDATE schema_meta SET value = '12' WHERE key = 'schema_version'")
    migrated = SessionStore(database)
    with migrated._connection() as connection:
        assert connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()["value"] == str(SCHEMA_VERSION)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='evaluation_results'"
            ).fetchone()["name"]
            == "evaluation_results"
        )
