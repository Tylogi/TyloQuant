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
from mfq.formats.tpq import normalize_tpq_dtype, unpack_tpq_indices
from mfq.formats.io import MMapTensorStore, load_mmap
from mfq.formats.moe import NintMoePool, NintMoeTensor
from mfq.formats.mx import MXFP8_DTYPE, MxTensor

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
    model_family: str
    projection_vq: bool
    projection_layout_by_layer: dict[int, dict[str, str]]
    projection_layout_by_expert: dict[
        int, tuple[dict[str, str], ...]
    ]
    projection_layout_specs: dict[str, dict[str, Any]]
    projection_codebook_group_sizes: dict[str, int]
    projection_codebook_group_counts: dict[str, int]
    index_packing: dict[str, str]
    no_expert_drop: bool
    routed_layers: int
    routed_experts_per_layer: int
    zlib: bool = False

    def tier_string(self, layer: int) -> str | None:
        return self.tiers_per_layer.get(int(layer))

    def projection_operator_capability(self, layer: int) -> dict[str, tuple]:
        if not self.projection_vq:
            return {}
        formats = {"u8": "p8", "u16": "p16"}
        formats.update({f"packed-u{bits}": f"p{bits}" for bits in range(8, 17)})
        def capability(
            layouts: dict[str, str], projection: str
        ) -> tuple[str, int, int]:
            layout = layouts[projection]
            dim, size = self.vq_dims[layout]
            packing = self.index_packing.get(layout)
            if packing is None:
                bits = int(size).bit_length() - 1
                if size <= 0 or 1 << bits != size:
                    raise ValueError(
                        f"L{layer} {projection} cannot infer packed width"
                    )
                packing = (
                    "u8" if bits == 8 else "u16" if bits == 16
                    else f"packed-u{bits}"
                )
            try:
                packed_format = formats[packing]
            except KeyError as exc:
                raise ValueError(
                    f"L{layer} {projection} has unsupported packing "
                    f"{packing!r}"
                ) from exc
            return packed_format, int(dim), int(size)

        layouts_by_expert = self.projection_layout_by_expert.get(int(layer))
        if layouts_by_expert is None:
            exact = tuple(
                capability(
                    self.projection_layout_by_layer[int(layer)], projection
                )
                for projection in ("gate", "up", "down")
            )
            return {
                "packed_formats": tuple(value[0] for value in exact),
                "code_dims": tuple(value[1] for value in exact),
                "codebook_sizes": tuple(value[2] for value in exact),
            }

        mixed = {
            capability(layouts, projection)
            for layouts in layouts_by_expert
            for projection in ("gate", "up", "down")
        }
        return {
            "packed_formats": tuple(sorted({value[0] for value in mixed})),
            "code_dims": tuple(sorted({value[1] for value in mixed})),
            "codebook_sizes": tuple(sorted({value[2] for value in mixed})),
        }

    def projection_layout(
        self,
        layer: int,
        projection: str,
        expert: int | None = None,
    ) -> str:
        layouts_by_expert = self.projection_layout_by_expert.get(int(layer))
        if layouts_by_expert is not None:
            if expert is None:
                raise ValueError(
                    f"L{layer} uses heterogeneous projection layouts; "
                    "expert_id is required"
                )
            if expert < 0 or expert >= len(layouts_by_expert):
                raise IndexError(f"TPQ expert is out of range: {expert}")
            return layouts_by_expert[int(expert)][str(projection)]
        return self.projection_layout_by_layer[int(layer)][str(projection)]


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
            "expert_records": (
                3 if self.manifest["quant"].get("method") == "projection-vq"
                else 2
            ) * len(self.manifest["expert_files"]),
            "index_storage": self.index_storage,
        }


class MfqCccpStore:
    """Expose native MFQ records through TyloQuant's CCCPStore interface."""

    _mfq_direct_host_pin = True

    def __init__(self, path: str | Path, tpq: ModuleType) -> None:
        self.path = Path(path).resolve()
        self.root = str(self.path.parent)
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
        routed_layers = (
            (manifest.get("routed_experts") or {}).get("layer_files") or {}
        )
        routed = manifest.get("routed_experts") or {}
        projection_vq = quant.get("method") == "projection-vq"
        projection_layout_by_expert = {}
        if projection_vq and routed_layers:
            projection_layout_by_layer = {
                int(layer): {
                    str(projection): str(layout)
                    for projection, layout in item["projection_layout"].items()
                }
                for layer, item in routed_layers.items()
            }
            projection_layout_specs = {
                str(name): dict(value)
                for name, value in (quant.get("projection_layouts") or {}).items()
            }
        elif projection_vq:
            projection_layout_specs = {
                str(name): dict(value)
                for name, value in (quant.get("layouts") or {}).items()
            }
            raw_layouts = quant.get("projection_layouts") or {}
            if raw_layouts:
                projection_layout_by_layer = {
                    int(layer): {
                        str(projection): str(layout)
                        for projection, layout in value.items()
                    }
                    for layer, value in raw_layouts.items()
                }
            else:
                heterogeneous = (
                    quant.get("heterogeneous_expert_tiering") or {}
                )
                precision_levels = (
                    heterogeneous.get("precision_levels") or {}
                )
                layer_levels = (
                    heterogeneous.get("layer_expert_levels") or {}
                )
                n_experts = int(config["n_experts"])
                projection_layout_by_layer = {}
                for raw_layer, raw_levels in layer_levels.items():
                    levels = tuple(str(value) for value in raw_levels)
                    if len(levels) != n_experts:
                        raise ValueError(
                            f"TPQ L{raw_layer} has {len(levels)} expert "
                            f"layouts, expected {n_experts}"
                        )
                    unknown = sorted(set(levels).difference(precision_levels))
                    if unknown:
                        raise ValueError(
                            f"TPQ L{raw_layer} uses unknown precision levels: "
                            f"{unknown[:8]}"
                        )
                    projection_layout_by_expert[int(raw_layer)] = tuple(
                        {
                            projection: str(
                                precision_levels[level][projection]
                            )
                            for projection in ("gate", "up", "down")
                        }
                        for level in levels
                    )
                missing = sorted(
                    set(expert_layers).difference(projection_layout_by_expert)
                )
                if missing:
                    raise ValueError(
                        "TPQ heterogeneous projection layouts are missing "
                        f"layers: {missing[:8]}"
                    )
        else:
            projection_layout_by_layer = {}
            projection_layout_specs = {}
        if projection_vq:
            vq_dims = {
                name: (int(value["dim"]), int(value["size"]))
                for name, value in projection_layout_specs.items()
            }
        else:
            vq_dims = {
                str(name): (int(value[0]), int(value[1]))
                for name, value in quant["vq"].items()
            }
        self.man = _ManifestView(
            config=config,
            quant=quant,
            expert_files={
                layer: self._gate_name(layer)
                for layer in expert_layers
            },
            tiers_per_layer=tiers,
            vq_dims=vq_dims,
            int4_group=int(quant.get("int4_group", 64)),
            model_family=str(manifest.get("model_family", "")),
            projection_vq=projection_vq,
            projection_layout_by_layer=projection_layout_by_layer,
            projection_layout_by_expert=projection_layout_by_expert,
            projection_layout_specs=projection_layout_specs,
            projection_codebook_group_sizes={
                name: int(value["group_size"])
                for name, value in projection_layout_specs.items()
                if value.get("group_size") is not None
            },
            projection_codebook_group_counts={
                name: int(value["groups"])
                for name, value in projection_layout_specs.items()
                if value.get("groups") is not None
            },
            index_packing={
                str(name): str(value)
                for name, value in (quant.get("index_packing") or {}).items()
            },
            no_expert_drop=bool(
                routed.get(
                    "no_expert_drop",
                    quant.get("no_expert_drop", False),
                )
            ),
            routed_layers=int(
                routed.get("layers", len(expert_layers))
            ),
            routed_experts_per_layer=int(
                routed.get(
                    "experts_per_layer",
                    config.get("n_experts", 0),
                )
            ),
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
        self._record_names = {
            layer: (
                tuple(
                    self._projection_name(layer, projection)
                    for projection in ("gate", "up", "down")
                )
                if projection_vq
                else (self._gate_name(layer), self._down_name(layer))
            )
            for layer in expert_layers
        }
        self._direct_maps = {
            layer: tuple(
                self._parse_direct_projection(name)
                for name in self._record_names[layer]
            )
            for layer in expert_layers
        }

    @staticmethod
    def _gate_name(layer: int) -> str:
        return f"layers.{layer}.ffn.experts.gate_up.weight"

    @staticmethod
    def _down_name(layer: int) -> str:
        return f"layers.{layer}.ffn.experts.down.weight"

    @staticmethod
    def _projection_name(layer: int, projection: str) -> str:
        return f"layers.{layer}.ffn.experts.{projection}.weight"

    @property
    def expert_bytes(self) -> int:
        return sum(
            sum(self._store.records[name].nbytes for name in names)
            for layer, names in self._record_names.items()
        )

    def packed_expert_residency(
        self,
        layer: int,
    ) -> tuple[int, tuple[int, ...], int]:
        """Return exact file, per-expert index, and auxiliary byte sizes."""
        layer = int(layer)
        names = self._record_names[layer]
        file_nbytes = sum(self._store.records[name].nbytes for name in names)
        projection_maps = self._direct_maps[layer]
        payloads: list[int] = []
        for expert in range(int(self.cfg["n_experts"])):
            payload = 0
            for projection in projection_maps:
                pool, _local = projection[expert]
                payload += (
                    pool.indices_per_expert * pool.spec.index_bits + 7
                ) // 8
            payloads.append(payload)
        payload_total = sum(payloads)
        return (
            int(file_nbytes),
            tuple(payloads),
            max(0, int(file_nbytes) - payload_total),
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

    def dense_nbytes(self, name: str) -> int:
        return int(self._store.records[str(name)].nbytes)

    def dense_resident_nbytes(self, name: str) -> int:
        # Native MFQ dense records are consumed in their packed representation.
        return self.dense_nbytes(name)

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
        if isinstance(value, MxTensor):
            return self._torch_array(value.values)
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
        if isinstance(value, MxTensor):
            if value.dtype != MXFP8_DTYPE:
                raise TypeError(
                    f"native TPQ dense tensor {name} uses unsupported "
                    f"MX dtype {value.dtype}"
                )
            raw = self._torch_array(value.values)
            scales = torch.pow(
                2.0,
                self._torch_array(value.scales).float() - 127.0,
            )
            return self._tpq.kernels.BlockFP8Weight(
                raw,
                scales,
                int(value.shape[1]),
                128,
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
        self._drop_records_file_cache(self._record_names[int(layer)])

    def release_dense_ram_blob(self) -> tuple[int, tuple[str, ...]]:
        """Native MFQ uses mmap and has no separate Dense RAM mirror."""
        return 0, ()

    def release_ram_blobs(self) -> None:
        """Keep the mmap store open for bounded expert-cache misses."""
        return None

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
                storage_bits=int(index_bits),
            )
            if index_bits != spec.index_bits or not 8 <= index_bits <= 16:
                raise ValueError(
                    f"native TPQ index width is unsupported: {name}"
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
            expected_end = codebook_end + (index_count * index_bits + 7) // 8
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
        count = pool.indices_per_expert
        nbytes = (count * pool.spec.index_bits + 7) // 8
        offset = pool.indices_offset + local * nbytes
        values, end = unpack_tpq_indices(
            self._store._mmaps[pool.source_index],
            offset,
            count,
            pool.spec.index_bits,
        )
        if end != offset + nbytes:
            raise ValueError("native TPQ expert index span is inconsistent")
        return values.reshape(
            pool.rows_per_expert,
            pool.columns // pool.spec.vector_size,
        )

    def _direct_raw(self, pool: _DirectCccpPool, local: int) -> np.ndarray:
        if local < 0 or local >= pool.expert_count:
            raise IndexError(f"native TPQ local expert is out of range: {local}")
        count = pool.indices_per_expert
        bits = pool.spec.index_bits
        total_bits = count * bits
        if total_bits % 8:
            raise ValueError("native TPQ expert payload is not byte aligned")
        nbytes = total_bits // 8
        return np.frombuffer(
            self._store._mmaps[pool.source_index],
            dtype=np.uint8,
            count=nbytes,
            offset=pool.indices_offset + local * nbytes,
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
        maps = self._direct_maps[int(layer)]
        if self.man.projection_vq:
            return (
                "projection-vq"
                if all(int(expert) in projection for projection in maps)
                else "drop"
            )
        assignments = self.man.tiers_per_layer.get(int(layer))
        if assignments is not None and int(expert) < len(assignments):
            try:
                return _CHAR_TO_TIER[assignments[int(expert)]]
            except KeyError as exc:
                raise ValueError(
                    f"invalid CCCP tier character {assignments[int(expert)]!r}"
                ) from exc
        pair = maps[0].get(int(expert))
        if pair is None:
            return "drop"
        return pair[0].spec.tier

    def available_mask(self, layer: int) -> torch.Tensor:
        assignments = self.man.tier_string(int(layer))
        if assignments is not None:
            if len(assignments) != int(self.cfg["n_experts"]):
                raise ValueError(f"TPQ L{layer} tier string has invalid length")
            return torch.tensor(
                [value.lower() != "d" for value in assignments],
                dtype=torch.bool,
            )
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

    def projection_codebooks(
        self,
        layer: int,
        eid: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.man.projection_vq:
            raise RuntimeError("native TPQ model is not projection-VQ")
        if eid is None:
            raise ValueError("projection-VQ grouped codebooks require expert_id")
        maps = self._direct_maps[int(layer)]
        result = []
        for projection in maps:
            try:
                pool, _local = projection[int(eid)]
            except KeyError as exc:
                raise KeyError(f"TPQ expert {layer}/{eid} is absent") from exc
            result.append(self._torch_array(pool.codebook))
        return tuple(result)

    def projection_codebook_variants(
        self,
        layer: int,
        eid: int | None = None,
    ) -> tuple[str, str, str]:
        if eid is None:
            raise ValueError("projection-VQ codebook variant requires expert_id")
        result = []
        for projection in ("gate", "up", "down"):
            layout = self.man.projection_layout(
                int(layer), projection, int(eid)
            )
            group_size = self.man.projection_codebook_group_sizes.get(layout)
            suffix = "" if group_size is None else f".g{int(eid) // group_size:03d}"
            result.append(f"L{layer}.{projection}.{layout}{suffix}")
        return tuple(result)

    def load_expert(self, layer: int, expert: int):
        if self.man.projection_vq:
            raise RuntimeError(
                "projection-VQ uses load_expert_packed with three projections"
            )
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

    def load_expert_packed(self, layer: int, expert: int):
        maps = self._direct_maps[int(layer)]
        if self.man.projection_vq:
            if len(maps) != 3 or not all(int(expert) in value for value in maps):
                raise KeyError(f"TPQ expert {layer}/{expert} is absent")
            weight_type = self._tpq.store.PackedVQWeight
            result = []
            for projection in maps:
                pool, local = projection[int(expert)]
                result.append(
                    weight_type(
                        self._torch_array(self._direct_raw(pool, local)),
                        self._torch_array(pool.codebook),
                        pool.rows_per_expert,
                        pool.columns,
                        pool.spec.index_bits,
                    )
                )
            return tuple(result)
        gate, down = maps
        weight_type = self._tpq.store.PackedVQWeight
        result = []
        for projection in (gate, down):
            pool, local = projection[int(expert)]
            result.append(
                weight_type(
                    self._torch_array(self._direct_raw(pool, local)),
                    self._torch_array(pool.codebook),
                    pool.rows_per_expert,
                    pool.columns,
                    pool.spec.index_bits,
                )
            )
        return tuple(result)

    def expert_signature_counts(self):
        if self.man.projection_vq:
            return Counter()
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

    current = getattr(
        dsv4model,
        "TPQStore",
        getattr(dsv4model, "CCCPStore", None),
    )
    if current is None:
        raise AttributeError("TPQ runtime exposes no store constructor")
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

    dsv4model.TPQStore = CCCPStoreDispatch
    dsv4model.CCCPStore = CCCPStoreDispatch
    store.TPQStoreDispatch = CCCPStoreDispatch
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
