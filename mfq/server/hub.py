"""Read-only model hub discovery for MFQ Server."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Literal

import httpx

from mfq.server.models import HubModelFile, HubModelInfo, HubModelSearchResult, HubModelSummary

HubProvider = Literal["huggingface", "modelscope"]


class HubError(RuntimeError):
    pass


class HubCatalog:
    async def search(
        self, provider: HubProvider, query: str, *, limit: int
    ) -> HubModelSearchResult:
        if provider == "huggingface":
            return await asyncio.to_thread(self._search_huggingface, query, limit)
        return await self._search_modelscope(query, limit)

    async def info(self, provider: HubProvider, repo_id: str, revision: str | None) -> HubModelInfo:
        if provider == "huggingface":
            return await asyncio.to_thread(self._info_huggingface, repo_id, revision)
        return await asyncio.to_thread(self._info_modelscope, repo_id, revision)

    @staticmethod
    def _search_huggingface(query: str, limit: int) -> HubModelSearchResult:
        try:
            from huggingface_hub import HfApi

            models = HfApi().list_models(
                search=query,
                sort="downloads",
                limit=limit,
                expand=["downloads", "likes", "lastModified"],
                token=os.environ.get("HF_TOKEN") or None,
            )
            return HubModelSearchResult(
                data=[
                    HubModelSummary(
                        provider="huggingface",
                        repo_id=item.id,
                        downloads=item.downloads or 0,
                        likes=item.likes or 0,
                        updated_at=item.last_modified,
                    )
                    for item in models
                ]
            )
        except Exception as error:
            raise HubError(str(error)) from error

    @staticmethod
    def _info_huggingface(repo_id: str, revision: str | None) -> HubModelInfo:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(
                repo_id,
                revision=revision,
                files_metadata=True,
                token=os.environ.get("HF_TOKEN") or None,
            )
            files = [
                HubModelFile(name=item.rfilename, byte_size=item.size or 0)
                for item in info.siblings or []
            ]
            return HubModelInfo(
                provider="huggingface",
                repo_id=info.id,
                revision=revision or info.sha or "main",
                downloads=info.downloads or 0,
                likes=info.likes or 0,
                total_bytes=sum(item.byte_size for item in files),
                updated_at=info.last_modified,
                files=files,
                tags=list(info.tags or []),
            )
        except Exception as error:
            raise HubError(str(error)) from error

    @staticmethod
    async def _search_modelscope(query: str, limit: int) -> HubModelSearchResult:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.put(
                    "https://modelscope.cn/api/v1/models/",
                    json={"PageSize": limit, "Name": query},
                )
                response.raise_for_status()
                payload = response.json().get("Data", {})
            values = payload.get("Models", payload.get("models", []))
            result = []
            for item in values:
                owner = item.get("Path") or ""
                name = item.get("Name") or ""
                repo_id = f"{owner}/{name}" if owner and name else name or owner
                if not repo_id:
                    continue
                result.append(
                    HubModelSummary(
                        provider="modelscope",
                        repo_id=repo_id,
                        downloads=int(item.get("Downloads") or 0),
                        likes=int(item.get("Likes") or item.get("Stars") or 0),
                        total_bytes=int(item.get("StorageSize") or 0),
                    )
                )
            return HubModelSearchResult(data=result[:limit])
        except Exception as error:
            raise HubError(str(error)) from error

    @staticmethod
    def _info_modelscope(repo_id: str, revision: str | None) -> HubModelInfo:
        try:
            from modelscope.hub.api import HubApi

            api = HubApi()
            model = api.get_model(repo_id, revision=revision)
            files = [
                HubModelFile(
                    name=item.get("Name") or item.get("Path") or "file",
                    byte_size=int(item.get("Size") or 0),
                )
                for item in api.get_model_files(repo_id, revision=revision) or []
            ]
            updated = model.get("LastUpdatedTime") if isinstance(model, dict) else None
            return HubModelInfo(
                provider="modelscope",
                repo_id=repo_id,
                revision=revision or "master",
                downloads=int(model.get("Downloads") or 0) if isinstance(model, dict) else 0,
                likes=int(model.get("Likes") or 0) if isinstance(model, dict) else 0,
                total_bytes=sum(item.byte_size for item in files),
                updated_at=datetime.fromisoformat(updated) if updated else None,
                files=files,
                tags=list(model.get("Tags") or []) if isinstance(model, dict) else [],
            )
        except Exception as error:
            raise HubError(str(error)) from error
