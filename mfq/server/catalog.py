"""MFQ model discovery without exposing host filesystem paths."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from mfq.formats.assets import is_asset_record
from mfq.formats.io import open_mmap
from mfq.formats.shards import matching_shard_paths, parse_shard_path
from mfq.server.models import ModelArtifactList, ModelArtifactResource

MODEL_FILE_INDEX = ".mfq-files.json"


class ModelArtifactNotFoundError(LookupError):
    pass


class DuplicateModelNameError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscoveredModel:
    resource: ModelArtifactResource
    path: Path


class ModelCatalog:
    """Discover complete MFQ containers under explicitly configured roots."""

    def __init__(self, roots: list[str | Path], *, cache_seconds: float = 5.0) -> None:
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots)
        self.cache_seconds = max(0.0, cache_seconds)
        self._models: dict[str, DiscoveredModel] = {}
        self._last_scan = float("-inf")
        self._lock = asyncio.Lock()

    async def list(self, *, refresh: bool = False) -> ModelArtifactList:
        models = await self._snapshot(refresh=refresh)
        return ModelArtifactList(
            data=sorted(
                (item.resource for item in models.values()),
                key=lambda item: (item.name.casefold(), item.id),
            )
        )

    async def get(self, model_id: str, *, refresh: bool = False) -> DiscoveredModel:
        models = await self._snapshot(refresh=refresh)
        result = models.get(model_id)
        if result is None and not refresh:
            models = await self._snapshot(refresh=True)
            result = models.get(model_id)
        if result is None:
            raise ModelArtifactNotFoundError(model_id)
        return result

    async def resolve(
        self,
        model: str,
        artifact_uri: str | None = None,
    ) -> DiscoveredModel:
        identifier = model
        if artifact_uri is not None:
            if not artifact_uri.startswith("mfq://"):
                raise ModelArtifactNotFoundError(
                    "only catalog-backed mfq:// artifact URIs are accepted"
                )
            identifier = artifact_uri.removeprefix("mfq://")
        models = await self._snapshot()
        exact = models.get(identifier)
        if exact is not None:
            return exact
        matches = [item for item in models.values() if item.resource.name == identifier]
        if not matches:
            models = await self._snapshot(refresh=True)
            exact = models.get(identifier)
            if exact is not None:
                return exact
            matches = [item for item in models.values() if item.resource.name == identifier]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ModelArtifactNotFoundError(f"ambiguous model name: {identifier}")
        raise ModelArtifactNotFoundError(identifier)

    async def resolve_path(self, path: str | Path) -> DiscoveredModel:
        """Resolve a configured model path to the catalog artifact that owns it."""

        target = self._canonical_model_path(Path(path).expanduser().resolve())
        for refresh in (False, True):
            models = await self._snapshot(refresh=refresh)
            matches = [
                item for item in models.values() if self._canonical_model_path(item.path) == target
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ModelArtifactNotFoundError(f"ambiguous model path: {path}")
        raise ModelArtifactNotFoundError(str(path))

    @staticmethod
    def _canonical_model_path(path: Path) -> Path:
        try:
            parsed = parse_shard_path(path)
        except ValueError:
            parsed = None
        return (parsed[0] if parsed is not None else path).resolve()

    async def _snapshot(self, *, refresh: bool = False) -> dict[str, DiscoveredModel]:
        async with self._lock:
            if refresh or monotonic() - self._last_scan >= self.cache_seconds:
                self._models = await asyncio.to_thread(self._scan)
                self._last_scan = monotonic()
            return dict(self._models)

    def _scan(self) -> dict[str, DiscoveredModel]:
        result: dict[str, DiscoveredModel] = {}
        model_paths: dict[str, Path] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            grouped: dict[Path, list[Path]] = {}
            paths = {*root.rglob("*.mfq"), *self._registered_paths(root)}
            for path in sorted(paths):
                try:
                    parsed = parse_shard_path(path)
                except ValueError:
                    parsed = None
                grouped.setdefault(parsed[0] if parsed is not None else path, []).append(path)
            for paths in grouped.values():
                path = next(
                    (
                        item
                        for item in paths
                        if (parsed := parse_shard_path(item)) is None or parsed[1] == 1
                    ),
                    paths[0],
                )
                privacy_root = root if path.is_relative_to(root) else path.parent
                discovered = self._inspect(privacy_root, path)
                canonical_path = self._canonical_model_path(discovered.path)
                previous_path = model_paths.get(discovered.resource.name)
                if previous_path is not None and previous_path != canonical_path:
                    raise DuplicateModelNameError(
                        f"duplicate catalog model name: {discovered.resource.name}"
                    )
                model_paths[discovered.resource.name] = canonical_path
                previous = result.get(discovered.resource.id)
                if previous is None or str(path) < str(previous.path):
                    result[discovered.resource.id] = discovered
        return result

    @staticmethod
    def _registered_paths(root: Path) -> set[Path]:
        index = root / MODEL_FILE_INDEX
        try:
            document = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return set()
        if not isinstance(document, dict) or document.get("version") != 1:
            return set()
        values = document.get("files")
        if not isinstance(values, list):
            return set()
        result: set[Path] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            path = Path(value).expanduser()
            if path.suffix.casefold() != ".mfq":
                continue
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                continue
            if resolved.is_file():
                try:
                    parsed = parse_shard_path(resolved)
                except ValueError:
                    continue
                if parsed is None:
                    result.add(resolved)
                else:
                    result.update(matching_shard_paths(parsed[0]))
        return result

    @staticmethod
    def _inspect(root: Path, path: Path) -> DiscoveredModel:
        try:
            parsed = parse_shard_path(path)
        except ValueError:
            parsed = None
        name = (parsed[0] if parsed is not None else path).stem
        relative = path.relative_to(root).as_posix()
        try:
            with open_mmap(path) as store:
                paths = tuple(store.paths)
                stats = tuple(item.stat() for item in paths)
                dtypes = sorted({record.dtype for record in store.records.values()})
                tensor_count = sum(
                    not is_asset_record(record.name) for record in store.records.values()
                )
                fingerprint = "\0".join(
                    [
                        store.header.model_arch,
                        name,
                        *(
                            f"{item.name}:{stat.st_size}"
                            for item, stat in zip(paths, stats, strict=True)
                        ),
                    ]
                )
                identifier = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
                resource = ModelArtifactResource(
                    id=identifier,
                    name=name,
                    architecture=store.header.model_arch or "unknown",
                    shard_count=len(paths),
                    total_bytes=sum(stat.st_size for stat in stats),
                    tensor_count=tensor_count,
                    record_count=len(store.records),
                    dtypes=dtypes,
                    complete=True,
                    loadable=True,
                    modified_at=datetime.fromtimestamp(
                        max(stat.st_mtime for stat in stats), timezone.utc
                    ),
                )
        except Exception as error:
            stat = path.stat()
            identifier = hashlib.sha256(f"{relative}\0{stat.st_size}".encode()).hexdigest()[:32]
            resource = ModelArtifactResource(
                id=identifier,
                name=name,
                architecture="unknown",
                shard_count=parsed[2] if parsed is not None else 1,
                total_bytes=stat.st_size,
                tensor_count=0,
                record_count=0,
                complete=False,
                loadable=False,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                error=str(error).replace(str(root), "<model-root>"),
            )
        return DiscoveredModel(resource=resource, path=path)
