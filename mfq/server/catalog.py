"""MFQ model discovery without exposing host filesystem paths."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from mfq.formats.assets import is_asset_record
from mfq.formats.io import open_mmap
from mfq.formats.shards import matching_shard_paths, parse_shard_path
from mfq.server.models import (
    ModelArtifactList,
    ModelArtifactResource,
    ModelDirectoryEntry,
    ModelDirectoryList,
)

MODEL_FILE_INDEX = ".mfq-files.json"


class ModelArtifactNotFoundError(LookupError):
    pass


class DuplicateModelNameError(RuntimeError):
    pass


class ModelDirectoryNotFoundError(LookupError):
    pass


class ModelRegistrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscoveredModel:
    resource: ModelArtifactResource
    path: Path


class ModelCatalog:
    """Discover complete MFQ containers and native HF checkpoints."""

    def __init__(
        self,
        roots: list[str | Path],
        *,
        cache_seconds: float = 5.0,
        browse_roots: list[str | Path] | None = None,
    ) -> None:
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots)
        self.browse_roots = (
            None
            if browse_roots is None
            else tuple(Path(root).expanduser().resolve() for root in browse_roots)
        )
        self.cache_seconds = max(0.0, cache_seconds)
        self._models: dict[str, DiscoveredModel] = {}
        self._last_scan = float("-inf")
        self._lock = asyncio.Lock()
        self._registration_lock = asyncio.Lock()
        self._directory_ids: dict[str, Path] = {}
        self._directory_secret = os.urandom(32)

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

    async def browse_directories(
        self,
        directory_id: str | None = None,
        *,
        path: str | Path | None = None,
    ) -> ModelDirectoryList:
        """List server-side directories by opaque identifier or explicit path."""

        return await asyncio.to_thread(self._browse_directories, directory_id, path)

    async def register_directory(
        self,
        *,
        directory_id: str | None = None,
        path: str | Path | None = None,
    ) -> ModelArtifactList:
        """Register every MFQ container below one server-side directory."""

        if (directory_id is None) == (path is None):
            raise ModelRegistrationError("exactly one directory source is required")
        if directory_id is not None:
            selected = self._directory_ids.get(directory_id)
            if selected is None:
                raise ModelDirectoryNotFoundError(directory_id)
        else:
            try:
                selected = Path(path or "").expanduser().resolve(strict=True)
            except OSError as error:
                raise ModelDirectoryNotFoundError(str(path)) from error
        if not selected.is_dir():
            raise ModelRegistrationError("selected model path is not a directory")
        if not self.roots:
            raise ModelRegistrationError("the server has no writable model catalog")

        async with self._registration_lock:
            registration_paths = await asyncio.to_thread(
                self._registration_paths,
                selected,
            )
            if not registration_paths:
                raise ModelRegistrationError("selected directory contains no supported models")
            index_path = self.roots[0] / MODEL_FILE_INDEX
            previous = index_path.read_bytes() if index_path.is_file() else None
            await asyncio.to_thread(self._write_registered_paths, index_path, registration_paths)
            try:
                models = await self._snapshot(refresh=True)
            except Exception:
                await asyncio.to_thread(self._restore_index, index_path, previous)
                await self._snapshot(refresh=True)
                raise

        selected_models = [
            item.resource
            for item in models.values()
            if item.path.resolve().is_relative_to(selected)
        ]
        return ModelArtifactList(
            data=sorted(selected_models, key=lambda item: (item.name.casefold(), item.id))
        )

    def _directory_id(self, path: Path) -> str:
        resolved = path.resolve()
        identifier = hashlib.sha256(
            self._directory_secret + os.fsencode(resolved)
        ).hexdigest()[:32]
        self._directory_ids[identifier] = resolved
        return identifier

    def _browse_roots(self) -> list[Path]:
        candidates = (
            [*self.roots, Path.home(), Path(Path.home().anchor or os.sep)]
            if self.browse_roots is None
            else list(self.browse_roots)
        )
        result: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_dir() or resolved in seen:
                continue
            seen.add(resolved)
            result.append(resolved)
        return result

    @staticmethod
    def _directory_name(path: Path) -> str:
        return path.name or path.anchor or str(path)

    @staticmethod
    def _immediate_model_count(path: Path) -> int:
        try:
            mfq_count = sum(
                item.is_file() and item.suffix.casefold() == ".mfq"
                for item in path.iterdir()
            )
            hf_count = int(ModelCatalog._is_hf_model_directory(path)) + sum(
                item.is_dir() and ModelCatalog._is_hf_model_directory(item)
                for item in path.iterdir()
            )
            return mfq_count + hf_count
        except OSError:
            return 0

    def _directory_entry(self, path: Path) -> ModelDirectoryEntry:
        return ModelDirectoryEntry(
            id=self._directory_id(path),
            name=self._directory_name(path),
            model_file_count=self._immediate_model_count(path),
        )

    def _browse_directories(
        self,
        directory_id: str | None,
        path: str | Path | None,
    ) -> ModelDirectoryList:
        if directory_id is None and path is None:
            return ModelDirectoryList(
                data=[self._directory_entry(path) for path in self._browse_roots()]
            )
        if path is not None:
            try:
                current = Path(path).expanduser().resolve(strict=True)
            except OSError as error:
                raise ModelDirectoryNotFoundError(str(path)) from error
        else:
            current = self._directory_ids.get(directory_id or "")
        if current is None or not current.is_dir():
            raise ModelDirectoryNotFoundError(str(path) if path is not None else directory_id)
        try:
            children = sorted(
                (item.resolve() for item in current.iterdir() if item.is_dir()),
                key=lambda item: (item.name.casefold(), item.name),
            )
        except OSError as error:
            raise ModelDirectoryNotFoundError(self._directory_name(current)) from error
        parent = current.parent if current.parent != current else None
        return ModelDirectoryList(
            current_id=self._directory_id(current),
            current_name=self._directory_name(current),
            current_path=str(current),
            parent_id=self._directory_id(parent) if parent is not None else None,
            model_file_count=self._immediate_model_count(current),
            data=[self._directory_entry(path) for path in children],
        )

    @staticmethod
    def _registration_paths(directory: Path) -> list[Path]:
        grouped: dict[Path, list[Path]] = {}
        try:
            candidates = sorted(directory.rglob("*.mfq"))
        except OSError as error:
            raise ModelRegistrationError("cannot scan selected model directory") from error
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                parsed = parse_shard_path(candidate)
            except ValueError as error:
                raise ModelRegistrationError(str(error)) from error
            grouped.setdefault(parsed[0] if parsed is not None else candidate, []).append(candidate)
        result: list[Path] = []
        for paths in grouped.values():
            result.append(
                next(
                    (
                        item
                        for item in paths
                        if (parsed := parse_shard_path(item)) is None or parsed[1] == 1
                    ),
                    paths[0],
                ).resolve()
            )
        hf_directories = {
            config.parent.resolve()
            for config in directory.rglob("config.json")
            if ModelCatalog._is_hf_model_directory(config.parent)
        }
        if ModelCatalog._is_hf_model_directory(directory):
            hf_directories.add(directory.resolve())
        result.extend(sorted(hf_directories))
        return result

    @staticmethod
    def _write_registered_paths(index: Path, paths: list[Path]) -> None:
        index.parent.mkdir(parents=True, exist_ok=True)
        document: dict[str, object] = {"version": 1, "files": []}
        if index.is_file():
            try:
                loaded = json.loads(index.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ModelRegistrationError("the model catalog index is invalid") from error
            if not isinstance(loaded, dict) or loaded.get("version") != 1:
                raise ModelRegistrationError("the model catalog index version is unsupported")
            document = loaded
        existing = document.get("files")
        files = {value for value in existing if isinstance(value, str)} if isinstance(existing, list) else set()
        files.update(str(path) for path in paths)
        payload = json.dumps(
            {"version": 1, "files": sorted(files)},
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8") + b"\n"
        temporary = index.with_name(f".{index.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(payload)
            temporary.chmod(0o600)
            os.replace(temporary, index)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _restore_index(index: Path, previous: bytes | None) -> None:
        if previous is None:
            index.unlink(missing_ok=True)
            return
        temporary = index.with_name(f".{index.name}.{os.getpid()}.restore")
        try:
            temporary.write_bytes(previous)
            temporary.chmod(0o600)
            os.replace(temporary, index)
        finally:
            temporary.unlink(missing_ok=True)

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
            paths.update(
                config.parent
                for config in root.rglob("config.json")
                if self._is_hf_model_directory(config.parent)
            )
            if self._is_hf_model_directory(root):
                paths.add(root)
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
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                continue
            if resolved.is_dir() and ModelCatalog._is_hf_model_directory(resolved):
                result.add(resolved)
                continue
            if resolved.suffix.casefold() != ".mfq":
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
        if path.is_dir():
            return ModelCatalog._inspect_hf(root, path)
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

    @staticmethod
    def _is_hf_model_directory(path: Path) -> bool:
        if not (path / "config.json").is_file():
            return False
        if (path / "model.safetensors.index.json").is_file():
            return True
        try:
            return any(item.is_file() for item in path.glob("*.safetensors"))
        except OSError:
            return False

    @staticmethod
    def _safetensors_header(path: Path) -> dict[str, object]:
        with path.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"truncated Safetensors header: {path.name}")
            size = struct.unpack("<Q", raw_size)[0]
            raw_header = stream.read(size)
        if len(raw_header) != size:
            raise ValueError(f"truncated Safetensors header: {path.name}")
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError(f"invalid Safetensors header: {path.name}")
        return header

    @staticmethod
    def _inspect_hf(root: Path, path: Path) -> DiscoveredModel:
        name = path.name
        relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else name
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            model_type = config.get("model_type") if isinstance(config, dict) else None
            if not isinstance(model_type, str) or not model_type:
                raise ValueError("HF config.json has no model_type")
            index_path = path / "model.safetensors.index.json"
            indexed_names: set[str] | None = None
            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("invalid Safetensors weight_map")
                if not all(isinstance(key, str) and isinstance(value, str)
                           for key, value in weight_map.items()):
                    raise ValueError("invalid Safetensors weight_map entry")
                indexed_names = set(weight_map)
                shard_paths = tuple(
                    path / item for item in sorted(set(weight_map.values()))
                )
            else:
                shard_paths = tuple(sorted(path.glob("*.safetensors")))
            if not shard_paths or any(not item.is_file() for item in shard_paths):
                raise ValueError("HF checkpoint is missing Safetensors shards")
            tensors: set[str] = set()
            dtypes: set[str] = set()
            for shard in shard_paths:
                for tensor_name, entry in ModelCatalog._safetensors_header(shard).items():
                    if tensor_name == "__metadata__":
                        continue
                    if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str):
                        raise ValueError(f"invalid Safetensors tensor: {tensor_name}")
                    if tensor_name in tensors:
                        raise ValueError(f"duplicate Safetensors tensor: {tensor_name}")
                    tensors.add(tensor_name)
                    dtypes.add(entry["dtype"])
            if indexed_names is not None and not indexed_names.issubset(tensors):
                raise ValueError("Safetensors index references missing tensors")
            stats = tuple(item.stat() for item in shard_paths)
            fingerprint = "\0".join(
                [model_type, name, *(f"{item.name}:{stat.st_size}" for item, stat in zip(
                    shard_paths, stats, strict=True
                ))]
            )
            identifier = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
            resource = ModelArtifactResource(
                id=identifier,
                name=name,
                architecture=f"{model_type}-hf-full-mfq",
                format="hf",
                shard_count=len(shard_paths),
                total_bytes=sum(stat.st_size for stat in stats),
                tensor_count=len(indexed_names if indexed_names is not None else tensors),
                record_count=len(indexed_names if indexed_names is not None else tensors) + 1,
                dtypes=sorted(dtypes),
                complete=True,
                loadable=True,
                modified_at=datetime.fromtimestamp(
                    max(stat.st_mtime for stat in stats), timezone.utc
                ),
            )
        except Exception as error:
            stat = path.stat()
            identifier = hashlib.sha256(f"{relative}\0{stat.st_mtime_ns}".encode()).hexdigest()[:32]
            resource = ModelArtifactResource(
                id=identifier,
                name=name,
                architecture="unknown",
                format="hf",
                shard_count=0,
                total_bytes=0,
                tensor_count=0,
                record_count=0,
                complete=False,
                loadable=False,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                error=str(error).replace(str(root), "<model-root>"),
            )
        return DiscoveredModel(resource=resource, path=path)
