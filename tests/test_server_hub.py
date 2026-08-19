from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from mfq.server.api import create_app
from mfq.server.hub import HubError
from mfq.server.models import HubModelFile, HubModelInfo, HubModelSearchResult, HubModelSummary
from mfq.server.service import ServerService
from mfq.server.storage import SessionStore
from tests.test_server_service import FakeBackend


class FakeHub:
    async def search(self, provider, query, *, limit):
        return HubModelSearchResult(
            data=[HubModelSummary(provider=provider, repo_id=f"owner/{query}")][:limit]
        )

    async def info(self, provider, repo_id, revision):
        return HubModelInfo(
            provider=provider,
            repo_id=repo_id,
            revision=revision or "main",
            total_bytes=42,
            files=[HubModelFile(name="weight.mfq", byte_size=42)],
        )


def test_hub_search_and_info_are_normalized(tmp_path: Path) -> None:
    async def run() -> None:
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            FakeBackend(),
            hub_catalog=FakeHub(),
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            search = await client.get(
                "/api/v1/hub/models",
                params={"provider": "modelscope", "query": "model"},
            )
            assert search.status_code == 200
            assert search.json()["data"][0]["repo_id"] == "owner/model"
            info = await client.get(
                "/api/v1/hub/models/huggingface/owner/model",
                params={"revision": "dev"},
            )
            assert info.status_code == 200
            assert info.json()["files"][0]["byte_size"] == 42
            assert info.json()["revision"] == "dev"
        await service.aclose()

    asyncio.run(run())


def test_hub_failures_are_retryable_gateway_errors(tmp_path: Path) -> None:
    class BrokenHub(FakeHub):
        async def search(self, provider, query, *, limit):
            raise HubError("offline")

    async def run() -> None:
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            FakeBackend(),
            hub_catalog=BrokenHub(),
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/hub/models",
                params={"provider": "huggingface", "query": "model"},
            )
            assert response.status_code == 502
            assert response.json()["error"]["retryable"]
        await service.aclose()

    asyncio.run(run())
