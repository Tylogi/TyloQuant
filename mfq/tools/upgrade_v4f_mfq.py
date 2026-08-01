"""Upgrade selected V4F MFQ expert projections to a tested NINT tier."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np

from mfq.formats.header import FileHeader, MFQ_MAGIC
from mfq.formats.io import (
    _NINT_MOE_HDR,
    _NINT_MOE_MAGIC_V2,
    _NINT_MOE_POOL_V2_HDR,
    _pack_nint_moe_runtime,
    _pack_tensor,
    _u32,
    open_mmap,
)
from mfq.formats.moe import NintMoePool, NintMoeTensor, expert_tensor_family
from mfq.formats.nepq import NepqTensor
from mfq.formats.nvq import NvqJscTensor, NvqTensor
from mfq.quantize.nint_quant import NintTensor
from mfq.quantize.v4f_upgrade import (
    NINT_SPECS,
    PROJECTIONS,
    V4FUpgradePlan,
    allocate_v4f_marked_nint8_upgrade,
    allocate_v4f_nint4_upgrade,
    allocate_v4f_sensitivity_reallocation,
    allocation_document,
    load_upgrade,
    marked_allocation_document,
    sensitivity_reallocation_document,
    sha256_file,
)


_ROUTED_NAME = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<projection>gate_up|down)_exps\.weight$"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _pool_name(layer: int, projection: str, family: str) -> str:
    return f"layer{layer:02d}-{projection}-{family.lower()}.blob"


def _allocation_family(profile: str) -> str:
    match = re.match(r"^(NINT\d+)-", profile)
    return match.group(1) if match else profile


def _nintm_allocation_profiles(
    blob: bytes | memoryview,
) -> tuple[tuple[int, int, int], tuple[str, ...]]:
    view = memoryview(blob)
    if len(view) < _NINT_MOE_HDR.size:
        raise ValueError("truncated NINTM header")
    magic, n_experts, out_per_expert, neuron_len, pool_count = (
        _NINT_MOE_HDR.unpack_from(view)
    )
    if magic != _NINT_MOE_MAGIC_V2:
        raise ValueError(f"unsupported NINTM magic: {magic!r}")
    profiles = [""] * n_experts
    offset = _NINT_MOE_HDR.size
    for _ in range(pool_count):
        if offset + _NINT_MOE_POOL_V2_HDR.size > len(view):
            raise ValueError("truncated NINTM pool header")
        expert_count, dtype_nbytes, payload_nbytes, runtime_nbytes = (
            _NINT_MOE_POOL_V2_HDR.unpack_from(view, offset)
        )
        offset += _NINT_MOE_POOL_V2_HDR.size
        ids_nbytes = expert_count * np.dtype(np.int32).itemsize
        metadata_end = offset + ids_nbytes + dtype_nbytes
        pool_end = metadata_end + runtime_nbytes + payload_nbytes
        if pool_end > len(view):
            raise ValueError("truncated NINTM pool payload")
        expert_ids = tuple(
            int(value)
            for value in np.frombuffer(
                view,
                dtype=np.dtype("<i4"),
                count=expert_count,
                offset=offset,
            )
        )
        offset += ids_nbytes
        dtype = bytes(view[offset : offset + dtype_nbytes]).decode("ascii")
        offset = pool_end
        family = _allocation_family(dtype)
        for expert in expert_ids:
            if expert < 0 or expert >= n_experts:
                raise ValueError(f"NINTM pool contains invalid expert {expert}")
            if profiles[expert]:
                raise ValueError(f"NINTM expert {expert} belongs to multiple pools")
            profiles[expert] = family
    if offset != len(view):
        raise ValueError("NINTM blob contains trailing bytes")
    missing = [index for index, family in enumerate(profiles) if not family]
    if missing:
        raise ValueError(f"NINTM pools omit experts {missing[:16]}")
    return (
        (int(n_experts), int(out_per_expert), int(neuron_len)),
        tuple(profiles),
    )


def _selection(
    plan: V4FUpgradePlan,
    projection: str,
) -> dict[int, tuple[int, ...]]:
    return plan.selected(projection)


def command_plan(args) -> None:
    allocation = allocate_v4f_nint4_upgrade(
        args.base_allocation,
        args.reap_csv,
        target_bytes=args.target_bytes,
        container_reserve_bytes=args.container_reserve_bytes,
    )
    document = allocation_document(
        allocation,
        base_allocation_path=args.base_allocation,
        reap_csv=args.reap_csv,
        source_index_sha256=args.source_index_sha256,
    )
    output = Path(args.output).resolve()
    _write_json(output, document)
    print(json.dumps(document, ensure_ascii=False), flush=True)


def command_plan_marked_nint8(args) -> None:
    allocation = allocate_v4f_marked_nint8_upgrade(
        args.base_allocation,
        args.reap_csv,
        args.sensitivity_map,
        target_bytes=args.target_bytes,
        container_reserve_bytes=args.container_reserve_bytes,
    )
    document = marked_allocation_document(
        allocation,
        base_allocation_path=args.base_allocation,
        reap_csv=args.reap_csv,
        sensitivity_map=args.sensitivity_map,
        source_index_sha256=args.source_index_sha256,
    )
    output = Path(args.output).resolve()
    _write_json(output, document)
    print(json.dumps(document, ensure_ascii=False), flush=True)


def command_plan_sensitivity_reallocation(args) -> None:
    allocation = allocate_v4f_sensitivity_reallocation(
        args.base_mfq,
        args.reap_csv,
        args.sensitivity_map,
        target_bytes=args.target_bytes,
        container_reserve_bytes=args.container_reserve_bytes,
    )
    document = sensitivity_reallocation_document(
        allocation,
        base_mfq=args.base_mfq,
        reap_csv=args.reap_csv,
        sensitivity_map=args.sensitivity_map,
    )
    output = Path(args.output).resolve()
    _write_json(output, document)
    print(json.dumps(document, ensure_ascii=False), flush=True)


def _source_index_path(root: Path) -> Path:
    return root / "model.safetensors.index.json"


def command_quantize_selected(args) -> None:
    import torch

    from mfq.quantize.v4f_source import V4FCheckpoint
    from mfq.quantize.v4f_upgrade import _nint_blob_nbytes
    from mfq.tools.quantize_hf_to_mfq import (
        _ExpertPoolRowSource,
        _write_nint_axis0_blob,
    )

    source_root = Path(args.input).resolve()
    plan_path = Path(args.plan).resolve()
    plan = load_upgrade(plan_path)
    family = plan.upgrade_family
    if getattr(plan, "demoted_count", 0):
        raise ValueError(
            "streaming patch cannot reconstruct demoted expert payloads"
        )
    spec = NINT_SPECS[family]
    raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_index = _source_index_path(source_root)
    source_index_sha = sha256_file(source_index)
    if source_index_sha != raw_plan["source_index_sha256"]:
        raise ValueError(
            "source index differs from the source used for the base model: "
            f"{source_index_sha}"
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = V4FCheckpoint(source_root)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    selected_layers = (
        None
        if not args.layers
        else {int(value) for value in args.layers.split(",") if value}
    )
    jobs = [
        (projection, layer, experts)
        for projection in PROJECTIONS
        for layer, experts in sorted(_selection(plan, projection).items())
        if selected_layers is None or layer in selected_layers
    ]
    started = time.perf_counter()
    records = []
    for index, (projection, layer, experts) in enumerate(jobs, start=1):
        output = output_dir / _pool_name(layer, projection, family)
        sidecar = output.with_suffix(output.suffix + ".json")
        columns = 4096 if projection == "gate_up" else 2048
        expected = _nint_blob_nbytes(
            len(experts) * 4096,
            columns,
            spec,
        )
        contract = {
            "format": "mfq.v4f-selected-precision.v1",
            "plan_sha256": sha256_file(plan_path),
            "source_index_sha256": source_index_sha,
            "layer": layer,
            "projection": projection,
            "expert_ids": list(experts),
            "rows_per_expert": 4096,
            "columns": columns,
            "dtype": family,
            "expected_bytes": expected,
        }
        reused = False
        if output.is_file() or sidecar.is_file():
            if not output.is_file() or not sidecar.is_file():
                raise ValueError(f"incomplete existing selected pool: {output}")
            existing = json.loads(sidecar.read_text(encoding="utf-8"))
            if any(existing.get(key) != value for key, value in contract.items()):
                raise ValueError(f"selected pool contract mismatch: {output}")
            if output.stat().st_size != expected:
                raise ValueError(f"selected pool size mismatch: {output}")
            if sha256_file(output) != existing.get("sha256"):
                raise ValueError(f"selected pool checksum mismatch: {output}")
            reused = True
        item_started = time.perf_counter()
        if not reused:
            source = checkpoint.expert_source(layer, projection)
            pool_source = _ExpertPoolRowSource(
                source,
                source.shape,
                source.shape,
                experts,
            )
            temporary = output.with_suffix(output.suffix + ".partial")
            if temporary.exists():
                raise FileExistsError(f"partial selected pool exists: {temporary}")
            nbytes = _write_nint_axis0_blob(
                pool_source,
                (len(experts) * 4096, columns),
                spec,
                temporary,
                args.row_chunk,
                "cuda",
                args.device,
            )
            if nbytes != expected:
                raise RuntimeError(
                    f"selected {family} size mismatch: {nbytes} != {expected}"
                )
            os.replace(temporary, output)
            contract["sha256"] = sha256_file(output)
            _write_json(sidecar, contract)
            del pool_source, source
        else:
            contract["sha256"] = sha256_file(output)
        elapsed = time.perf_counter() - item_started
        total_elapsed = time.perf_counter() - started
        record = {
            **contract,
            "path": str(output),
            "status": "reused" if reused else "written",
            "seconds": elapsed,
            "peak_vram_mib": (
                torch.cuda.max_memory_reserved(args.device) / (1024 * 1024)
                if str(args.device).startswith("cuda")
                else 0.0
            ),
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(jobs),
                    "layer": layer,
                    "projection": projection,
                    "experts": len(experts),
                    "mb": expected / 1e6,
                    "status": record["status"],
                    "seconds": elapsed,
                    "peak_vram_mib": record["peak_vram_mib"],
                    "eta_seconds": total_elapsed / index * (len(jobs) - index),
                }
            ),
            flush=True,
        )
    manifest = {
        "format": "mfq.v4f-selected-precision-manifest.v1",
        "dtype": family,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "source": str(source_root),
        "source_index_sha256": source_index_sha,
        "records": records,
        "total_bytes": sum(int(record["expected_bytes"]) for record in records),
        "seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "manifest.json", manifest)


def _subset_first_axis(value: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value)[positions])


def _subset_flat_rows(
    value: np.ndarray,
    positions: np.ndarray,
    *,
    experts: int,
    rows_per_expert: int,
) -> np.ndarray:
    array = np.asarray(value)
    tail = array.shape[1:]
    reshaped = array.reshape(experts, rows_per_expert, *tail)
    return np.ascontiguousarray(reshaped[positions].reshape(-1, *tail))


def _subset_pool_tensor(
    tensor,
    positions: np.ndarray,
    *,
    pool_experts: int,
    rows_per_expert: int,
):
    if isinstance(tensor, NepqTensor):
        return NepqTensor(
            spec=tensor.spec,
            shape=(len(positions), tensor.out_per_expert, tensor.neuron_len),
            neuron_scale=_subset_first_axis(tensor.neuron_scale, positions),
            state=_subset_first_axis(tensor.state, positions),
            indices=_subset_first_axis(tensor.indices, positions),
            aux=(
                None
                if tensor.aux is None
                else _subset_first_axis(tensor.aux, positions)
            ),
            bank_ids=_subset_first_axis(tensor.bank_ids, positions),
            table_payloads=np.ascontiguousarray(tensor.table_payloads),
            rotation_block=tensor.rotation_block,
            rotation_seed=tensor.rotation_seed,
        )
    common = {
        "shape": (len(positions) * rows_per_expert, tensor.neuron_len),
        "axis": tensor.axis,
        "neuron_len": tensor.neuron_len,
    }
    if isinstance(tensor, NvqJscTensor):
        return NvqJscTensor(
            **common,
            neuron_scale=_subset_flat_rows(
                tensor.neuron_scale,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            scale_lut=np.ascontiguousarray(tensor.scale_lut),
            bank_for_state=np.ascontiguousarray(tensor.bank_for_state),
            state=_subset_flat_rows(
                tensor.state,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            indices=_subset_flat_rows(
                tensor.indices,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            signs=_subset_flat_rows(
                tensor.signs,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            codebooks=np.ascontiguousarray(tensor.codebooks),
            base_spec=tensor.base_spec,
        )
    if isinstance(tensor, NvqTensor):
        return NvqTensor(
            **common,
            spec=tensor.spec,
            neuron_scale=_subset_flat_rows(
                tensor.neuron_scale,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            sub_scale=_subset_flat_rows(
                tensor.sub_scale,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            indices=_subset_flat_rows(
                tensor.indices,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            signs=_subset_flat_rows(
                tensor.signs,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            codebook=(
                None
                if tensor.codebook is None
                else np.ascontiguousarray(tensor.codebook)
            ),
        )
    if isinstance(tensor, NintTensor):
        return NintTensor(
            **common,
            spec=tensor.spec,
            q=_subset_flat_rows(
                tensor.q,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            neuron_scale=_subset_flat_rows(
                tensor.neuron_scale,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            neuron_min=_subset_flat_rows(
                tensor.neuron_min,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            sub_scale=_subset_flat_rows(
                tensor.sub_scale,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
            sub_min=_subset_flat_rows(
                tensor.sub_min,
                positions,
                experts=pool_experts,
                rows_per_expert=rows_per_expert,
            ),
        )
    raise TypeError(f"unsupported V4F base pool type: {type(tensor)!r}")


def _write_upgraded_routed_stream(
    base: NintMoeTensor,
    selected_ids: tuple[int, ...],
    selected_path: Path,
    handle,
    selected_family: str = "NINT4",
) -> int:
    start = handle.tell()
    selected = set(selected_ids)
    if not selected:
        raise ValueError("upgraded routed blob requires selected experts")
    owners: set[int] = set()
    pools = []
    for pool in base.pools:
        ids = np.asarray(pool.expert_ids, dtype=np.int32).reshape(-1)
        keep_positions = np.asarray(
            [index for index, expert in enumerate(ids) if int(expert) not in selected],
            dtype=np.int64,
        )
        if not keep_positions.size:
            continue
        keep_ids = np.ascontiguousarray(ids[keep_positions], dtype=np.int32)
        subset = _subset_pool_tensor(
            pool.tensor,
            keep_positions,
            pool_experts=len(ids),
            rows_per_expert=base.out_per_expert,
        )
        pools.append((keep_ids, subset))
        owners.update(int(value) for value in keep_ids)
    owners.update(selected)
    if owners != set(range(base.n_experts)):
        raise ValueError("upgraded routed pools do not cover all experts")
    if selected_family not in NINT_SPECS:
        raise ValueError(f"unsupported selected precision: {selected_family}")
    if selected_path.stat().st_size <= 0:
        raise ValueError(
            f"empty selected {selected_family} payload: {selected_path}"
        )

    handle.write(
        _NINT_MOE_HDR.pack(
            _NINT_MOE_MAGIC_V2,
            base.n_experts,
            base.out_per_expert,
            base.neuron_len,
            len(pools) + 1,
        )
    )
    for expert_ids, tensor in pools:
        dtype, payload = _pack_tensor(tensor, allow_moe=False)
        runtime = _pack_nint_moe_runtime(tensor)
        dtype_bytes = dtype.encode("ascii")
        handle.write(
            _NINT_MOE_POOL_V2_HDR.pack(
                len(expert_ids),
                len(dtype_bytes),
                len(payload),
                len(runtime),
            )
        )
        handle.write(expert_ids.tobytes())
        handle.write(dtype_bytes)
        handle.write(runtime)
        handle.write(payload)
        del payload, tensor
    dtype_bytes = selected_family.encode("ascii")
    handle.write(
        _NINT_MOE_POOL_V2_HDR.pack(
            len(selected_ids),
            len(dtype_bytes),
            selected_path.stat().st_size,
            0,
        )
    )
    handle.write(np.asarray(selected_ids, dtype=np.int32).tobytes())
    handle.write(dtype_bytes)
    with selected_path.open("rb") as source:
        shutil.copyfileobj(source, handle, length=32 * 1024 * 1024)
    return handle.tell() - start


def _write_upgraded_routed_blob(
    base: NintMoeTensor,
    selected_ids: tuple[int, ...],
    selected_path: Path,
    output: Path,
    selected_family: str = "NINT4",
) -> int:
    with output.open("wb") as handle:
        return _write_upgraded_routed_stream(
            base,
            selected_ids,
            selected_path,
            handle,
            selected_family,
        )


def _write_header(handle, header: FileHeader, records: list[tuple[str, str, int]]) -> None:
    version = max(2, int(header.version))
    extra = dict(header.extra)
    handle.write(MFQ_MAGIC)
    handle.write(_u32(version))
    arch = header.model_arch.encode("utf-8")
    handle.write(_u32(len(arch)))
    handle.write(arch)
    handle.write(_u32(len(extra)))
    for key, value in extra.items():
        kb = str(key).encode("utf-8")
        vb = json.dumps(value).encode("utf-8")
        handle.write(_u32(len(kb)))
        handle.write(kb)
        handle.write(_u32(len(vb)))
        handle.write(vb)
    handle.write(_u32(len(records)))
    for name, dtype, nbytes in records:
        nb = name.encode("utf-8")
        db = dtype.encode("utf-8")
        handle.write(_u32(len(nb)))
        handle.write(nb)
        handle.write(_u32(len(db)))
        handle.write(db)
        handle.write(struct.pack("<Q", int(nbytes)))


def command_patch(args) -> None:
    base_path = Path(args.base_mfq).resolve()
    plan_path = Path(args.plan).resolve()
    selected_dir = Path(args.selected_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    plan = load_upgrade(plan_path)
    family = plan.upgrade_family
    raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (selected_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("selected precision manifest belongs to another plan")
    if manifest.get("dtype", "NINT4") != family:
        raise ValueError("selected precision manifest has the wrong dtype")

    started = time.perf_counter()
    with open_mmap(base_path) as store:
        if len(store.paths) != 1:
            raise ValueError(
                "upgrade_v4f_mfq requires a single-file base; "
                "apply the upgrade before splitting"
            )
        if store.header.extra.get("source_index_sha256") != raw_plan[
            "source_index_sha256"
        ]:
            raise ValueError("base MFQ and upgrade plan use different source weights")
        replacements: dict[
            str,
            tuple[tuple[int, ...], Path, int, int, str],
        ] = {}
        routed_sum = 0
        manifest_records = {
            (int(record["layer"]), str(record["projection"])): record
            for record in manifest.get("records", [])
        }
        for name, record in store.records.items():
            match = _ROUTED_NAME.match(name)
            if not match:
                continue
            layer = int(match.group("layer"))
            projection = match.group("projection")
            selected = _selection(plan, projection).get(layer, ())
            if not selected:
                routed_sum += record.nbytes
                continue
            selected_path = selected_dir / _pool_name(
                layer,
                projection,
                family,
            )
            if not selected_path.is_file():
                raise FileNotFoundError(
                    f"selected {family} payload is absent: {selected_path}"
                )
            manifest_record = manifest_records.get((layer, projection))
            if manifest_record is None:
                raise ValueError(
                    f"selected precision manifest has no record for {name}"
                )
            if tuple(manifest_record.get("expert_ids", ())) != tuple(selected):
                raise ValueError(
                    f"selected precision manifest has the wrong experts for {name}"
                )
            if int(manifest_record.get("expected_bytes", -1)) != (
                selected_path.stat().st_size
            ):
                raise ValueError(
                    f"selected precision manifest has the wrong size for {name}"
                )
            if sha256_file(selected_path) != manifest_record.get("sha256"):
                raise ValueError(
                    f"selected precision payload checksum differs for {name}"
                )
            expected_counts = dict(Counter(plan.families(projection, layer)))
            from mfq.quantize.v4f_plan import routed_family_blob_bytes

            expected = routed_family_blob_bytes(projection, expected_counts)
            replacements[name] = (
                tuple(selected),
                selected_path,
                expected,
                layer,
                projection,
            )
            routed_sum += expected
        if routed_sum != plan.routed_bytes:
            raise RuntimeError(
                f"patched routed accounting mismatch: {routed_sum} != "
                f"{plan.routed_bytes}"
            )

        extra = dict(store.header.extra)
        extra.update(
            {
                "upgrade_plan_sha256": sha256_file(plan_path),
                "target_bytes": plan.target_bytes,
                "estimated_blob_bytes": plan.estimated_blob_bytes,
                "expert_low_family": "NEPQ0-S",
                "expert_middle_family": "NVQ2J",
                "expert_high_family": family,
                "expert_allocation_objective": raw_plan[
                    "allocation_objective"
                ],
            }
        )
        header = FileHeader(
            version=2,
            model_arch=store.header.model_arch,
            num_tensors=len(store.records),
            extra=extra,
        )
        records = [
            (
                name,
                record.dtype,
                replacements[name][2]
                if name in replacements
                else record.nbytes,
            )
            for name, record in store.records.items()
        ]
        partial = output.with_suffix(output.suffix + ".partial")
        if partial.exists():
            raise FileExistsError(f"partial output already exists: {partial}")
        output.parent.mkdir(parents=True, exist_ok=True)
        required = sum(nbytes for _name, _dtype, nbytes in records) + 1_000_000
        free = shutil.disk_usage(output.parent).free
        if free < required:
            raise OSError(
                f"output filesystem has {free} free bytes; "
                f"streaming patch needs at least {required}"
            )
        with partial.open("wb") as handle, base_path.open("rb") as base_handle:
            _write_header(handle, header, records)
            total_records = len(records)
            for index, (name, _dtype, expected_nbytes) in enumerate(
                records,
                start=1,
            ):
                if name in replacements:
                    (
                        selected,
                        selected_path,
                        _expected,
                        layer,
                        projection,
                    ) = replacements[name]
                    base_tensor = store[name]
                    if not isinstance(base_tensor, NintMoeTensor):
                        raise TypeError(f"base routed tensor is not NINTM: {name}")
                    if hasattr(plan, "base_families"):
                        expected_base = plan.base_families(
                            projection, layer
                        )
                    else:
                        base_high = set(
                            plan.base_high(projection).get(layer, ())
                        )
                        expected_base = tuple(
                            "NVQ2J"
                            if expert in base_high
                            else "NEPQ0-S"
                            for expert in range(256)
                        )
                    actual_base = tuple(
                        _allocation_family(profile)
                        for profile in base_tensor.expert_profiles
                    )
                    if actual_base != expected_base:
                        raise ValueError(
                            f"base precision map differs from the plan for {name}"
                        )
                    written = _write_upgraded_routed_stream(
                        base_tensor,
                        selected,
                        selected_path,
                        handle,
                        family,
                    )
                    if written != expected_nbytes:
                        raise RuntimeError(
                            f"upgraded routed size mismatch for {name}: "
                            f"{written} != {expected_nbytes}"
                        )
                    del base_tensor
                    gc.collect()
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "total": total_records,
                                "name": name,
                                "selected_dtype": family,
                                "selected_experts": len(selected),
                                "mb": written / 1e6,
                                "status": "streamed",
                            }
                        ),
                        flush=True,
                    )
                else:
                    record = store.records[name]
                    base_handle.seek(record.offset)
                    remaining = record.nbytes
                    while remaining:
                        block = base_handle.read(min(32 * 1024 * 1024, remaining))
                        if not block:
                            raise EOFError(f"truncated base tensor blob: {name}")
                        handle.write(block)
                        remaining -= len(block)
                    if index % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "completed": index,
                                    "total": total_records,
                                    "name": name,
                                    "status": "copied",
                                }
                            ),
                            flush=True,
                        )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def command_verify(args) -> None:
    model = Path(args.mfq).resolve()
    plan = load_upgrade(args.plan)
    counts: dict[str, int] = {}
    checked = 0
    with open_mmap(model) as store:
        for name in store:
            match = _ROUTED_NAME.match(name)
            if not match:
                continue
            layer = int(match.group("layer"))
            projection = match.group("projection")
            record = store.records[name]
            if record.dtype != "NINTM":
                raise TypeError(f"routed tensor is not NINTM: {name}")
            blob = store.blob_view(record)
            try:
                shape, actual = _nintm_allocation_profiles(blob)
            finally:
                blob.release()
            expected_shape = (
                256,
                4096,
                4096 if projection == "gate_up" else 2048,
            )
            if shape != expected_shape:
                raise ValueError(f"routed shape mismatch: {name}: {shape}")
            expected = plan.families(projection, layer)
            if actual != expected:
                raise ValueError(f"precision map mismatch: {name}")
            for family in actual:
                counts[family] = counts.get(family, 0) + 1
            checked += 1
        if checked != 86:
            raise ValueError(f"expected 86 routed tensors, found {checked}")
    print(
        json.dumps(
            {
                "model": str(model),
                "bytes": model.stat().st_size,
                "sha256": sha256_file(model),
                "routed_tensors": checked,
                "projection_expert_counts": counts,
            }
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--base-allocation", required=True)
    plan.add_argument("--reap-csv", required=True)
    plan.add_argument("--source-index-sha256", required=True)
    plan.add_argument("--target-bytes", type=int, default=45_000_000_000)
    plan.add_argument("--container-reserve-bytes", type=int, default=4_000_000)
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=command_plan)

    marked = commands.add_parser("plan-marked-nint8")
    marked.add_argument("--base-allocation", required=True)
    marked.add_argument("--reap-csv", required=True)
    marked.add_argument("--sensitivity-map", required=True)
    marked.add_argument("--source-index-sha256", required=True)
    marked.add_argument("--target-bytes", type=int)
    marked.add_argument("--container-reserve-bytes", type=int, default=4_000_000)
    marked.add_argument("--output", required=True)
    marked.set_defaults(func=command_plan_marked_nint8)

    reallocate = commands.add_parser("plan-sensitivity-reallocation")
    reallocate.add_argument("--base-mfq", required=True)
    reallocate.add_argument("--reap-csv", required=True)
    reallocate.add_argument("--sensitivity-map", required=True)
    reallocate.add_argument("--target-bytes", type=int)
    reallocate.add_argument(
        "--container-reserve-bytes",
        type=int,
        default=4_000_000,
    )
    reallocate.add_argument("--output", required=True)
    reallocate.set_defaults(func=command_plan_sensitivity_reallocation)

    quantize = commands.add_parser("quantize-selected")
    quantize.add_argument("--input", required=True)
    quantize.add_argument("--plan", required=True)
    quantize.add_argument("--output-dir", required=True)
    quantize.add_argument("--device", default="cuda")
    quantize.add_argument("--row-chunk", type=int, default=256)
    quantize.add_argument("--layers", default="")
    quantize.set_defaults(func=command_quantize_selected)

    patch = commands.add_parser("patch")
    patch.add_argument("--base-mfq", required=True)
    patch.add_argument("--plan", required=True)
    patch.add_argument(
        "--selected-dir",
        "--nint4-dir",
        dest="selected_dir",
        required=True,
    )
    patch.add_argument(
        "--temp-dir",
        default="",
        help=argparse.SUPPRESS,
    )
    patch.add_argument("--output", required=True)
    patch.set_defaults(func=command_patch)

    verify = commands.add_parser("verify")
    verify.add_argument("--mfq", required=True)
    verify.add_argument("--plan", required=True)
    verify.set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
