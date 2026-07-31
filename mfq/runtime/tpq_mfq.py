"""TyloQuant store adapter for native TPQ payloads inside MFQ files."""

from __future__ import annotations

import mmap
import os
import struct
import threading
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from mfq.formats.tpq import CccpInt4Tensor, CccpPqSpec, CccpPqTensor
from mfq.formats.tpq import normalize_tpq_dtype
from mfq.formats.io import MMapTensorStore, load_mmap
from mfq.formats.moe import NintMoePool, NintMoeTensor

_CHAR_TO_TIER = {
    "x": "x",
    "w": "w",
    "v": "v",
    "V": "vv",
    "d": "drop",
}

_NINT_MOE_HEADER = struct.Struct("<4sIIII")
_NINT_MOE_POOL_V2_HEADER = struct.Struct("<IIQQ")
_CCCP_PQ_HEADER = struct.Struct("<4sBBBBiiII")


@dataclass
class _ManifestView:
    config: dict[str, Any]
    quant: dict[str, Any]
    expert_files: dict[int, str]
    tiers_per_layer: dict[int, str]
    vq_dims: dict[str, tuple[int, int]]
    int4_group: int
    zlib: bool = False


@dataclass(frozen=True)
class _DirectCccpPool:
    spec: CccpPqSpec
    codebook: np.ndarray
    indices_offset: int
    rows_per_expert: int
    columns: int
    expert_count: int
    source_index: int

    @property
    def indices_per_expert(self) -> int:
        return self.rows_per_expert * (
            self.columns // self.spec.vector_size
        )


@dataclass(frozen=True)
class NativeCCCPArtifact:
    """Validated native CCCP payloads stored in one MFQ file."""

    path: Path
    manifest: dict[str, Any]
    disk_bytes: int
    expert_bytes: int
    model_arch: str
    index_storage: dict[str, int]

    @classmethod
    def open(cls, path: str | Path) -> NativeCCCPArtifact:
        resolved = Path(path).resolve()
        header, store = load_mmap(resolved)
        try:
            manifest = header.extra.get(
                "tpq_manifest",
                header.extra.get("cccp_manifest"),
            )
            if header.extra.get("source_format") not in {
                "tpq-1",
                "cccp-1",
            } or not isinstance(manifest, dict):
                raise ValueError(
                    f"MFQ file has no native TPQ manifest: {resolved}"
                )
            expert_bytes = sum(
                record.nbytes
                for record in store.records.values()
                if record.dtype == "NINTM"
            )
            index_storage = {
                str(key): int(value)
                for key, value in dict(
                    header.extra.get(
                        "tpq_index_storage",
                        header.extra.get("cccp_index_storage", {}),
                    )
                ).items()
            }
            return cls(
                path=resolved,
                manifest=manifest,
                disk_bytes=sum(path.stat().st_size for path in store.paths),
                expert_bytes=expert_bytes,
                model_arch=header.model_arch,
                index_storage=index_storage,
            )
        finally:
            store.close()

    @property
    def architecture(self) -> str:
        config = self.manifest["config"]
        if (
            str(self.manifest.get("model_family", "")).lower() == "kimi_k3"
            or ("kda_layers" in config and "routed_hidden" in config)
        ):
            return "kimi_k3"
        return "deepseek_v4" if "hc_mult" in config else "glm"

    def summary(self) -> dict[str, Any]:
        config = self.manifest["config"]
        return {
            "format": "mfq-native-tpq.v1",
            "path": str(self.path),
            "disk_bytes": self.disk_bytes,
            "disk_gib": self.disk_bytes / (1 << 30),
            "architecture": self.model_arch,
            "layers": int(config["n_layers"]),
            "experts_per_layer": int(config["n_experts"]),
            "expert_records": 2 * len(self.manifest["expert_files"]),
            "index_storage": self.index_storage,
        }


class MfqCccpStore:
    """Expose native MFQ records through TyloQuant's CCCPStore interface."""

    _mfq_direct_host_pin = True

    def __init__(self, path: str | Path, tpq: ModuleType) -> None:
        self.path = Path(path).resolve()
        self.header, self._store = load_mmap(self.path, cache=False)
        source_format = self.header.extra.get("source_format")
        manifest = self.header.extra.get(
            "tpq_manifest",
            self.header.extra.get("cccp_manifest"),
        )
        if source_format not in {"tpq-1", "cccp-1"} or not isinstance(
            manifest, dict
        ):
            self.close()
            raise ValueError(f"MFQ file has no native TPQ manifest: {self.path}")
        config = dict(manifest["config"])
        quant = dict(manifest["quant"])
        tiers = {
            int(layer): str(value)
            for layer, value in manifest.get("tiers_per_layer", {}).items()
        }
        expert_layers = sorted(
            int(layer) for layer in manifest["expert_files"]
        )
        self.man = _ManifestView(
            config=config,
            quant=quant,
            expert_files={
                layer: self._gate_name(layer)
                for layer in expert_layers
            },
            tiers_per_layer=tiers,
            vq_dims={
                str(name): (int(value[0]), int(value[1]))
                for name, value in quant["vq"].items()
            },
            int4_group=int(quant.get("int4_group", 64)),
        )
        self.cfg = config
        self._tpq = tpq
        self._dense_names = tuple(
            name
            for name, record in self._store.records.items()
            if record.dtype != "NINTM"
        )
        self._layer_cache: dict[
            int, tuple[NintMoeTensor, NintMoeTensor]
        ] = {}
        self._layer_maps: dict[
            int,
            tuple[
                dict[int, tuple[NintMoePool, int]],
                dict[int, tuple[NintMoePool, int]],
            ],
        ] = {}
        self._layer_lock = threading.RLock()
        self._expert_signature_count_cache = None
        self.heat_ranks = None
        self._direct_maps = {
            layer: (
                self._parse_direct_projection(self._gate_name(layer)),
                self._parse_direct_projection(self._down_name(layer)),
            )
            for layer in expert_layers
        }

    @staticmethod
    def _gate_name(layer: int) -> str:
        return f"layers.{layer}.ffn.experts.gate_up.weight"

    @staticmethod
    def _down_name(layer: int) -> str:
        return f"layers.{layer}.ffn.experts.down.weight"

    @property
    def expert_bytes(self) -> int:
        return sum(
            self._store.records[self._gate_name(layer)].nbytes
            + self._store.records[self._down_name(layer)].nbytes
            for layer in self.man.expert_files
        )

    def close(self) -> None:
        store = getattr(self, "_store", None)
        if store is not None:
            self._direct_maps.clear()
            self._layer_cache.clear()
            self._layer_maps.clear()
            store.close()
            self._store = None

    def has_mtp(self) -> bool:
        return False

    def get_mtp(self, name: str):
        raise KeyError(f"native CCCP MFQ has no MTP tensor: {name}")

    def has(self, name: str) -> bool:
        return name in self._store.records

    def dense_names(self) -> list[str]:
        return list(self._dense_names)

    @staticmethod
    def _torch_array(value: np.ndarray) -> torch.Tensor:
        array = np.asarray(value)
        if not array.flags.writeable:
            array = array.copy()
        return torch.from_numpy(array)

    def get_raw(self, name: str) -> torch.Tensor:
        value = self._store[name]
        if isinstance(value, np.ndarray):
            return self._torch_array(value)
        if isinstance(value, CccpInt4Tensor):
            return self._torch_array(value.packed)
        raise TypeError(f"native CCCP record is not dense: {name}")

    def get_dense(self, name: str):
        value = self._store[name]
        if isinstance(value, CccpInt4Tensor):
            return self._tpq.kernels.Int4Weight(
                self._torch_array(value.packed),
                self._torch_array(value.scales),
                value.neuron_len,
                value.group_size,
            )
        if isinstance(value, np.ndarray):
            return self._torch_array(value).float()
        raise TypeError(f"native CCCP record is not dense: {name}")

    def _drop_records_file_cache(self, names: tuple[str, ...]) -> None:
        posix_fadvise = getattr(os, "posix_fadvise", None)
        dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
        if posix_fadvise is None or dontneed is None:
            return
        page_size = mmap.PAGESIZE
        mmap_dontneed = getattr(mmap, "MADV_DONTNEED", None)
        for name in names:
            record = self._store.records[name]
            file_obj = self._store.file_for(record)
            mm = self._store.mmap_for(record)
            fd = file_obj.fileno()
            if mmap_dontneed is not None and hasattr(mm, "madvise"):
                start = record.offset // page_size * page_size
                end = min(
                    mm.size(),
                    (
                        (record.offset + record.nbytes + page_size - 1)
                        // page_size
                    )
                    * page_size,
                )
                with suppress(OSError, ValueError):
                    mm.madvise(mmap_dontneed, start, end - start)
            with suppress(OSError):
                posix_fadvise(fd, record.offset, record.nbytes, dontneed)

    def drop_dense_file_cache(self) -> None:
        self._drop_records_file_cache(self._dense_names)

    def drop_expert_file_cache(self, layer: int) -> None:
        self._drop_records_file_cache(
            (self._gate_name(layer), self._down_name(layer))
        )

    def _parse_direct_projection(
        self,
        name: str,
    ) -> dict[int, tuple[_DirectCccpPool, int]]:
        record = self._store.records[name]
        if record.dtype != "NINTM":
            raise TypeError(f"native CCCP expert record is not NINTM: {name}")
        mm = self._store.mmap_for(record)
        start = int(record.offset)
        end = start + int(record.nbytes)
        if start + _NINT_MOE_HEADER.size > end:
            raise ValueError(f"truncated native CCCP expert header: {name}")
        magic, n_experts, rows_per_expert, columns, pool_count = (
            _NINT_MOE_HEADER.unpack_from(mm, start)
        )
        if magic != b"NIM2" or int(n_experts) != int(self.cfg["n_experts"]):
            raise ValueError(f"unsupported native CCCP expert container: {name}")
        offset = start + _NINT_MOE_HEADER.size
        result: dict[int, tuple[_DirectCccpPool, int]] = {}
        for _ in range(int(pool_count)):
            if offset + _NINT_MOE_POOL_V2_HEADER.size > end:
                raise ValueError(f"truncated native CCCP pool header: {name}")
            expert_count, dtype_nbytes, payload_nbytes, runtime_nbytes = (
                _NINT_MOE_POOL_V2_HEADER.unpack_from(mm, offset)
            )
            offset += _NINT_MOE_POOL_V2_HEADER.size
            ids_nbytes = int(expert_count) * np.dtype("<i4").itemsize
            ids_end = offset + ids_nbytes
            dtype_end = ids_end + int(dtype_nbytes)
            runtime_end = dtype_end + int(runtime_nbytes)
            payload_end = runtime_end + int(payload_nbytes)
            if (
                expert_count == 0
                or dtype_nbytes == 0
                or dtype_nbytes > 32
                or payload_end > end
            ):
                raise ValueError(f"invalid native CCCP pool metadata: {name}")
            expert_ids = np.frombuffer(
                mm,
                dtype="<i4",
                count=int(expert_count),
                offset=offset,
            ).copy()
            try:
                dtype = normalize_tpq_dtype(
                    bytes(mm[ids_end:dtype_end]).decode("ascii")
                )
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"native CCCP pool dtype is not ASCII: {name}"
                ) from exc
            if runtime_nbytes:
                raise ValueError(
                    f"native CCCP pool has unexpected runtime metadata: {name}"
                )
            payload_start = runtime_end
            if payload_start + _CCCP_PQ_HEADER.size > payload_end:
                raise ValueError(f"truncated native CCCP payload: {name}")
            (
                pq_magic,
                pq_version,
                _tier_id,
                vector_size,
                index_bits,
                axis,
                neuron_len,
                ndim,
                codebook_entries,
            ) = _CCCP_PQ_HEADER.unpack_from(mm, payload_start)
            tier_prefix = "TPQ-"
            if (
                pq_magic != b"CPQ1"
                or pq_version != 1
                or not dtype.startswith(tier_prefix)
                or axis != 0
                or ndim != 2
            ):
                raise ValueError(f"invalid native CCCP payload header: {name}")
            spec = CccpPqSpec(
                tier=dtype[len(tier_prefix) :].lower(),
                vector_size=int(vector_size),
                codebook_entries=int(codebook_entries),
            )
            if index_bits != spec.index_bits or index_bits not in {8, 16}:
                raise ValueError(
                    f"native CCCP indices are not source-byte-aligned: {name}"
                )
            payload_offset = payload_start + _CCCP_PQ_HEADER.size
            shape = tuple(
                int(value)
                for value in struct.unpack_from("<2q", mm, payload_offset)
            )
            payload_offset += 16
            rows = int(struct.unpack_from("<I", mm, payload_offset)[0])
            payload_offset += 4
            expected_shape = (
                int(expert_count) * int(rows_per_expert),
                int(columns),
            )
            if (
                shape != expected_shape
                or rows != shape[0]
                or int(neuron_len) != shape[1]
                or shape[1] % spec.vector_size
            ):
                raise ValueError(
                    f"native CCCP pool shape mismatch in {name}: "
                    f"{shape} != {expected_shape}"
                )
            codebook_count = spec.codebook_entries * spec.vector_size
            codebook_nbytes = codebook_count * np.dtype("<f4").itemsize
            codebook_end = payload_offset + codebook_nbytes
            if codebook_end > payload_end:
                raise ValueError(f"truncated native CCCP codebook: {name}")
            codebook = np.frombuffer(
                mm,
                dtype="<f4",
                count=codebook_count,
                offset=payload_offset,
            ).astype(np.float32, copy=True)
            codebook = codebook.reshape(
                spec.codebook_entries,
                spec.vector_size,
            )
            index_count = shape[0] * (shape[1] // spec.vector_size)
            expected_end = codebook_end + index_count * (index_bits // 8)
            if expected_end != payload_end:
                raise ValueError(
                    f"native CCCP index payload size mismatch: {name}"
                )
            pool = _DirectCccpPool(
                spec=spec,
                codebook=codebook,
                indices_offset=codebook_end,
                rows_per_expert=int(rows_per_expert),
                columns=int(columns),
                expert_count=int(expert_count),
                source_index=record.source_index,
            )
            for local, expert in enumerate(expert_ids):
                expert_id = int(expert)
                if expert_id in result:
                    raise ValueError(
                        f"duplicate native CCCP expert {expert_id}: {name}"
                    )
                result[expert_id] = (pool, local)
            offset = payload_end
        if offset != end:
            raise ValueError(
                f"invalid native CCCP expert tail: {name} "
                f"({end - offset} bytes)"
            )
        return result

    def _direct_indices(
        self,
        pool: _DirectCccpPool,
        local: int,
    ) -> np.ndarray:
        if local < 0 or local >= pool.expert_count:
            raise IndexError(f"native CCCP local expert is out of range: {local}")
        dtype = np.dtype(np.uint8 if pool.spec.index_bits == 8 else "<u2")
        count = pool.indices_per_expert
        offset = pool.indices_offset + local * count * dtype.itemsize
        return np.frombuffer(
            self._store._mmaps[pool.source_index],
            dtype=dtype,
            count=count,
            offset=offset,
        ).reshape(
            pool.rows_per_expert,
            pool.columns // pool.spec.vector_size,
        )

    @staticmethod
    def _expert_map(
        tensor: NintMoeTensor,
    ) -> dict[int, tuple[NintMoePool, int]]:
        result: dict[int, tuple[NintMoePool, int]] = {}
        for pool in tensor.pools:
            for local, expert in enumerate(
                np.asarray(pool.expert_ids, dtype=np.int32)
            ):
                result[int(expert)] = (pool, local)
        return result

    def _layer(
        self,
        layer: int,
    ) -> tuple[
        tuple[NintMoeTensor, NintMoeTensor],
        tuple[
            dict[int, tuple[NintMoePool, int]],
            dict[int, tuple[NintMoePool, int]],
        ],
    ]:
        layer = int(layer)
        with self._layer_lock:
            tensors = self._layer_cache.get(layer)
            maps = self._layer_maps.get(layer)
            if tensors is None:
                gate = self._store[self._gate_name(layer)]
                down = self._store[self._down_name(layer)]
                if not isinstance(gate, NintMoeTensor) or not isinstance(
                    down, NintMoeTensor
                ):
                    raise TypeError(f"native CCCP layer {layer} is not NINTM")
                tensors = (gate, down)
                maps = (
                    self._expert_map(gate),
                    self._expert_map(down),
                )
                self._layer_cache[layer] = tensors
                self._layer_maps[layer] = maps
            assert maps is not None
            return tensors, maps

    def expert_kind(self, layer: int, expert: int) -> str:
        assignments = self.man.tiers_per_layer.get(int(layer))
        if assignments is not None and int(expert) < len(assignments):
            try:
                return _CHAR_TO_TIER[assignments[int(expert)]]
            except KeyError as exc:
                raise ValueError(
                    f"invalid CCCP tier character {assignments[int(expert)]!r}"
                ) from exc
        maps = self._direct_maps[int(layer)]
        pair = maps[0].get(int(expert))
        if pair is None:
            return "drop"
        return pair[0].spec.tier

    def available_mask(self, layer: int) -> torch.Tensor:
        return torch.tensor(
            [
                self.expert_kind(layer, expert) != "drop"
                for expert in range(int(self.cfg["n_experts"]))
            ],
            dtype=torch.bool,
        )

    @staticmethod
    def _pool_tensor(
        pair: tuple[NintMoePool, int],
    ) -> tuple[CccpPqTensor, int]:
        pool, local = pair
        if not isinstance(pool.tensor, CccpPqTensor):
            raise TypeError("native CCCP pool is not a learned PQ tensor")
        return pool.tensor, int(local)

    def codebooks(
        self,
        layer: int,
        kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        maps = self._direct_maps[int(layer)]
        base = kind.rstrip("z")
        expert = next(
            (
                value
                for value in range(int(self.cfg["n_experts"]))
                if self.expert_kind(layer, value) == base
            ),
            None,
        )
        if expert is None:
            raise KeyError(f"CCCP layer {layer} has no {base} cohort")
        gate, _ = maps[0][expert]
        down, _ = maps[1][expert]
        return (
            self._torch_array(gate.codebook),
            self._torch_array(down.codebook),
        )

    def load_expert(self, layer: int, expert: int):
        maps = self._direct_maps[int(layer)]
        if expert not in maps[0] or expert not in maps[1]:
            raise KeyError(f"CCCP expert {layer}/{expert} is absent")
        gate, gate_local = maps[0][expert]
        down, down_local = maps[1][expert]
        if gate.spec.tier != down.spec.tier:
            raise ValueError(
                f"CCCP expert {layer}/{expert} has split projection tiers"
            )
        gate_indices = self._direct_indices(gate, gate_local)
        down_indices = self._direct_indices(down, down_local)
        weight_type = self._tpq.kernels.VQWeight
        return (
            weight_type(
                self._torch_array(gate_indices),
                self._torch_array(gate.codebook),
                int(self.cfg["hidden"]),
            ),
            weight_type(
                self._torch_array(down_indices),
                self._torch_array(down.codebook),
                int(self.cfg["moe_inter"]),
            ),
        )

    def expert_signature_counts(self):
        cached = self._expert_signature_count_cache
        if cached is not None:
            return cached.copy()
        signature_type = self._tpq.expert_slots.ExpertSignature
        counts = Counter()
        hidden = int(self.cfg["hidden"])
        intermediate = int(self.cfg["moe_inter"])
        for layer in self.man.expert_files:
            for expert in range(int(self.cfg["n_experts"])):
                kind = self.expert_kind(layer, expert)
                if kind == "drop":
                    continue
                vector_size, entries = self.man.vq_dims[kind]
                dtype = torch.uint16 if entries > 256 else torch.uint8
                counts[
                    signature_type(
                        (2 * intermediate, hidden // vector_size),
                        dtype,
                        (hidden, intermediate // vector_size),
                        dtype,
                    )
                ] += 1
        self._expert_signature_count_cache = counts
        return counts.copy()


def install_mfq_cccp_store(tpq: ModuleType) -> None:
    """Teach TyloQuant's DSV4 constructor to accept a native MFQ path."""

    from tpq import dsv4model, store

    current = dsv4model.CCCPStore
    if getattr(current, "_mfq_native_dispatch", False):
        return
    original = current

    class CCCPStoreDispatch:
        _mfq_native_dispatch = True

        def __new__(cls, root):
            path = Path(root)
            if path.is_file() and path.suffix.lower() == ".mfq":
                return MfqCccpStore(path, tpq)
            return original(root)

    dsv4model.CCCPStore = CCCPStoreDispatch
    store.CCCPStoreDispatch = CCCPStoreDispatch


def inspect_native_cccp(path: str | Path) -> dict[str, Any]:
    """Validate and summarize one native CCCP MFQ file."""

    return NativeCCCPArtifact.open(path).summary()


# Canonical public names for new callers.
NativeTPQArtifact = NativeCCCPArtifact
MfqTpqStore = MfqCccpStore
inspect_native_tpq = inspect_native_cccp
install_mfq_tpq_store = install_mfq_cccp_store


__all__ = [
    "MfqCccpStore",
    "MfqTpqStore",
    "NativeCCCPArtifact",
    "NativeTPQArtifact",
    "inspect_native_cccp",
    "inspect_native_tpq",
    "install_mfq_cccp_store",
    "install_mfq_tpq_store",
]
