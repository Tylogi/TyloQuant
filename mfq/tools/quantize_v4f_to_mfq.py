"""Production DeepSeek-V4-Flash MXFP -> EW MFQ conversion pipeline."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertPrecision,
    ExpertSelection,
    ExpertTensorSelection,
    load_scheme,
    save_scheme,
)
from mfq.formats.header import FileHeader
from mfq.formats.runtime_profile import (
    RUNTIME_SAMPLING_METADATA_KEY,
    architecture_profile,
)
from mfq.formats.nint import NintSpec
from mfq.formats.shards import (
    matching_shard_paths,
    parse_size,
    validate_split_limits,
    write_blob_record_shards,
)
from mfq.quantize.nepq_train import (
    NepqBankTrainConfig,
    train_nepq0_s_banks,
)
from mfq.quantize.nvq_jsc import (
    NvqJscConfig,
    jsc_tables_from_tensor,
    train_nvq_jsc,
)
from mfq.quantize.v4f_plan import (
    RecipeTensor,
    V4FEwAllocation,
    V4FTieredAllocation,
    allocate_v4f_ew,
    allocate_v4f_ew_nvq2j_nint4,
    load_recipe,
    read_gguf_header_recipe,
    recipe_source_map,
    recipe_target_dtype,
    routed_blob_bytes,
    routed_family_blob_bytes,
    save_recipe,
)
from mfq.quantize.v4f_source import V4FCheckpoint
from mfq.quantize.v4f_imatrix import (
    V4FExpertImportance,
    V4FImportanceMatrix,
)
from mfq.quantize.mxfp import read_dense_rows
from mfq.tools.quantize_hf_to_mfq import (
    BlobRecord,
    TensorPlan,
    _dense_blob_from_tensor,
    _plan_blob_nbytes,
    _spec_for_plan,
    _write_mixed_moe_axis0_blob,
    _write_nint_axis0_blob,
)


_ROTATION_BLOCK = 2048
_ROTATION_SEED = 18601311049
_DEFAULT_ADMM_ITERATIONS = 8
_DEFAULT_ADMM_RHO = 1.0
_DEFAULT_NINT = NintSpec(4, 24, 6)
_RECIPE_SPECS = {
    "NINT4": NintSpec(4, 24, 6),
    "NINT5": NintSpec(5, 28, 7),
    "NINT6": NintSpec(6, 24, 7),
    "NINT8": NintSpec(8, 48, 7),
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _json_array(value: object) -> np.ndarray:
    return np.frombuffer(_canonical(value), dtype=np.uint8).copy()


def _json_from_array(value: np.ndarray) -> object:
    return json.loads(np.asarray(value, dtype=np.uint8).tobytes())


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _write_json_idempotent(path: Path, document: object) -> None:
    encoded = json.dumps(document, ensure_ascii=False, indent=2)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if _canonical(current) != _canonical(document):
            raise ValueError(f"existing run artifact has a different contract: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _allocation_document(allocation: V4FEwAllocation, args) -> dict:
    return {
        "format": "mfq.v4f-ew-allocation.v1",
        "target_bytes": allocation.target_bytes,
        "nonexpert_bytes": allocation.nonexpert_bytes,
        "routed_bytes": allocation.routed_bytes,
        "estimated_blob_bytes": allocation.estimated_blob_bytes,
        "estimated_headroom_bytes": (
            allocation.target_bytes - allocation.estimated_blob_bytes
        ),
        "gate_up_high_count": sum(map(len, allocation.gate_up_high.values())),
        "down_high_count": sum(map(len, allocation.down_high.values())),
        "gate_up_high": {
            str(layer): list(experts)
            for layer, experts in allocation.gate_up_high.items()
        },
        "down_high": {
            str(layer): list(experts)
            for layer, experts in allocation.down_high.items()
        },
        "gate_up_energy_fraction": allocation.gate_up_energy_fraction,
        "down_energy_fraction": allocation.down_energy_fraction,
        "reap_csv": str(Path(args.reap_csv).resolve()),
        "reap_sha256": _sha256(args.reap_csv),
        "imatrix": str(Path(args.imatrix).resolve()),
        "imatrix_sha256": (
            args.imatrix_sha256
            if hasattr(args, "imatrix_sha256")
            else _sha256(args.imatrix)
        ),
        "expert_objective": "count_weighted_diagonal_imatrix",
        "low_family": "NEPQ0-S",
        "high_family": "NVQ2J",
        "rotation_block": _ROTATION_BLOCK,
        "rotation_seed": _ROTATION_SEED,
        "admm_iterations": args.admm_iterations,
        "admm_rho": args.admm_rho,
        "mtp_included": False,
    }


def _tiered_allocation_document(
    allocation: V4FTieredAllocation,
    args,
) -> dict:
    return {
        "format": "mfq.v4f-ew-allocation.v2",
        "profile": "nvq2j-nint4",
        "target_bytes": allocation.target_bytes,
        "nonexpert_bytes": allocation.nonexpert_bytes,
        "routed_bytes": allocation.routed_bytes,
        "estimated_blob_bytes": allocation.estimated_blob_bytes,
        "estimated_headroom_bytes": (
            allocation.target_bytes - allocation.estimated_blob_bytes
        ),
        "base_family": "NVQ2J",
        "upgrade_family": "NINT4",
        "gate_up_nint4_count": sum(
            map(len, allocation.gate_up_nint4.values())
        ),
        "down_nint4_count": sum(map(len, allocation.down_nint4.values())),
        "gate_up_nint4": {
            str(layer): list(experts)
            for layer, experts in allocation.gate_up_nint4.items()
        },
        "down_nint4": {
            str(layer): list(experts)
            for layer, experts in allocation.down_nint4.items()
        },
        "gate_up_energy_fraction": allocation.gate_up_energy_fraction,
        "down_energy_fraction": allocation.down_energy_fraction,
        "allocation_objective": "reap_output_energy_per_upgrade_byte",
        "reap_csv": str(Path(args.reap_csv).resolve()),
        "reap_sha256": _sha256(args.reap_csv),
        "imatrix": str(Path(args.imatrix).resolve()),
        "imatrix_sha256": _sha256(args.imatrix),
        "expert_objective": "count_weighted_diagonal_imatrix",
        "mtp_included": False,
    }


def _load_allocation(
    path: Path,
) -> V4FEwAllocation | V4FTieredAllocation:
    raw = json.loads(path.read_text(encoding="utf-8"))
    allocation_format = raw.get("format")
    common = {
        "target_bytes": int(raw["target_bytes"]),
        "nonexpert_bytes": int(raw["nonexpert_bytes"]),
        "routed_bytes": int(raw["routed_bytes"]),
        "estimated_blob_bytes": int(raw["estimated_blob_bytes"]),
        "gate_up_energy_fraction": float(raw["gate_up_energy_fraction"]),
        "down_energy_fraction": float(raw["down_energy_fraction"]),
    }
    if allocation_format == "mfq.v4f-ew-allocation.v1":
        return V4FEwAllocation(
            **common,
            gate_up_high={
                int(layer): tuple(int(value) for value in experts)
                for layer, experts in raw["gate_up_high"].items()
            },
            down_high={
                int(layer): tuple(int(value) for value in experts)
                for layer, experts in raw["down_high"].items()
            },
        )
    if (
        allocation_format == "mfq.v4f-ew-allocation.v2"
        and raw.get("profile") == "nvq2j-nint4"
    ):
        return V4FTieredAllocation(
            **common,
            gate_up_nint4={
                int(layer): tuple(int(value) for value in experts)
                for layer, experts in raw["gate_up_nint4"].items()
            },
            down_nint4={
                int(layer): tuple(int(value) for value in experts)
                for layer, experts in raw["down_nint4"].items()
            },
        )
    raise ValueError(f"unsupported V4F allocation: {path}")


def _artifact_name(layer: int, projection: str, family: str) -> str:
    label = family.lower().replace("-", "")
    return f"layer{layer:02d}-{projection}-{label}.npz"


def _precision(
    layer: int,
    projection: str,
    family: str,
    *,
    row_chunk: int,
    bank_chunk: int,
    admm_iterations: int,
    admm_rho: float,
) -> ExpertPrecision:
    artifact = f"artifacts/{_artifact_name(layer, projection, family)}"
    if family == "NEPQ0-S":
        options = (
            ("rotation_block", _ROTATION_BLOCK),
            ("rotation_seed", _ROTATION_SEED),
            ("anchor_multipliers", "1.0"),
            ("refine_steps", 1),
            ("row_chunk", row_chunk),
            ("bank_chunk", bank_chunk),
            ("admm_iterations", admm_iterations),
            ("admm_rho", admm_rho),
        )
    elif family == "NVQ2J":
        options = (
            ("banks", 4),
            ("assignment_refine_steps", 2),
            ("search_steps", 19),
            ("group_chunk", 1024),
        )
    else:
        raise ValueError(f"unsupported V4F family: {family}")
    return ExpertPrecision(
        family=family,
        artifact=artifact,
        options=options,
    )


def _selection(
    layer: int,
    projection: str,
    high_experts: tuple[int, ...],
    *,
    row_chunk: int,
    bank_chunk: int,
    admm_iterations: int,
    admm_rho: float,
) -> ExpertTensorSelection:
    high = set(high_experts)
    rows = 4096
    columns = 4096 if projection == "gate_up" else 2048
    low_precision = _precision(
        layer,
        projection,
        "NEPQ0-S",
        row_chunk=row_chunk,
        bank_chunk=bank_chunk,
        admm_iterations=admm_iterations,
        admm_rho=admm_rho,
    )
    high_precision = _precision(
        layer,
        projection,
        "NVQ2J",
        row_chunk=row_chunk,
        bank_chunk=bank_chunk,
        admm_iterations=admm_iterations,
        admm_rho=admm_rho,
    )
    low_average = routed_blob_bytes(projection, 256, 0) * 8 // 256
    high_average = routed_blob_bytes(projection, 0, 256) * 8 // 256
    experts = tuple(
        ExpertSelection(
            expert_id=expert,
            spec=None,
            precision=high_precision if expert in high else low_precision,
            storage_bits=high_average if expert in high else low_average,
            train_loss=0.0,
            validation_loss=0.0,
        )
        for expert in range(256)
    )
    return ExpertTensorSelection(
        name=f"blk.{layer}.ffn_{projection}_exps.weight",
        group=f"v4f.layer{layer}.{projection}",
        n_experts=256,
        rows_per_expert=rows,
        columns=columns,
        selections=experts,
    )


def _tiered_selection(
    layer: int,
    projection: str,
    nint4_experts: tuple[int, ...],
) -> ExpertTensorSelection:
    nint4 = set(nint4_experts)
    rows = 4096
    columns = 4096 if projection == "gate_up" else 2048
    nvq2j_precision = _precision(
        layer,
        projection,
        "NVQ2J",
        row_chunk=512,
        bank_chunk=8,
        admm_iterations=_DEFAULT_ADMM_ITERATIONS,
        admm_rho=_DEFAULT_ADMM_RHO,
    )
    nint4_precision = ExpertPrecision(
        family="NINT4",
        nint_spec=_RECIPE_SPECS["NINT4"],
    )
    nvq2j_average = (
        routed_family_blob_bytes(projection, {"NVQ2J": 256}) * 8 // 256
    )
    nint4_average = (
        routed_family_blob_bytes(projection, {"NINT4": 256}) * 8 // 256
    )
    experts = tuple(
        ExpertSelection(
            expert_id=expert,
            spec=(
                _RECIPE_SPECS["NINT4"]
                if expert in nint4
                else None
            ),
            precision=(
                nint4_precision
                if expert in nint4
                else nvq2j_precision
            ),
            storage_bits=(
                nint4_average if expert in nint4 else nvq2j_average
            ),
            train_loss=0.0,
            validation_loss=0.0,
        )
        for expert in range(256)
    )
    return ExpertTensorSelection(
        name=f"blk.{layer}.ffn_{projection}_exps.weight",
        group=f"v4f.layer{layer}.{projection}",
        n_experts=256,
        rows_per_expert=rows,
        columns=columns,
        selections=experts,
    )


def prepare(args) -> None:
    if args.admm_iterations <= 0:
        raise ValueError("V4F production NEPQ requires positive ADMM iterations")
    if not np.isfinite(args.admm_rho) or args.admm_rho <= 0:
        raise ValueError("V4F ADMM rho must be finite and positive")
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = run_dir / "recipe.json"
    if recipe_path.exists():
        recipe, _ = load_recipe(recipe_path)
    else:
        recipe, metadata = read_gguf_header_recipe(args.recipe_headers)
        save_recipe(recipe_path, recipe, metadata)
    checkpoint = V4FCheckpoint(args.input)
    imatrix = V4FImportanceMatrix.load(args.imatrix)
    source_map = recipe_source_map(checkpoint, recipe)
    if args.expert_profile == "nepq0s-nvq2j":
        allocation: V4FEwAllocation | V4FTieredAllocation = allocate_v4f_ew(
            recipe,
            args.reap_csv,
            target_bytes=args.target_bytes,
        )
        allocation_document = _allocation_document(allocation, args)
        target_profile = "v4f-ew40g-nepq0s-nvq2j"
    elif args.expert_profile == "nvq2j-nint4":
        allocation = allocate_v4f_ew_nvq2j_nint4(
            recipe,
            args.reap_csv,
            target_bytes=args.target_bytes,
        )
        allocation_document = _tiered_allocation_document(allocation, args)
        target_profile = "v4f-ew88g-nvq2j-nint4"
    else:
        raise ValueError(f"unsupported V4F expert profile: {args.expert_profile}")
    allocation_path = run_dir / "allocation.json"
    _write_json_idempotent(allocation_path, allocation_document)
    scheme_path = run_dir / "scheme.json"
    if not scheme_path.exists():
        selections = {}
        if isinstance(allocation, V4FEwAllocation):
            for layer in range(43):
                for projection, high_map in (
                    ("gate_up", allocation.gate_up_high),
                    ("down", allocation.down_high),
                ):
                    item = _selection(
                        layer,
                        projection,
                        high_map.get(layer, ()),
                        row_chunk=args.assignment_row_chunk,
                        bank_chunk=args.assignment_bank_chunk,
                        admm_iterations=args.admm_iterations,
                        admm_rho=args.admm_rho,
                    )
                    selections[item.name] = item
        else:
            for layer in range(43):
                for projection, nint4_map in (
                    ("gate_up", allocation.gate_up_nint4),
                    ("down", allocation.down_nint4),
                ):
                    item = _tiered_selection(
                        layer,
                        projection,
                        nint4_map.get(layer, ()),
                    )
                    selections[item.name] = item
        scheme = CalibrationScheme(
            path=None,
            target_profile=target_profile,
            target_storage_bits=allocation.routed_bytes * 8,
            selections={},
            expert_selections=selections,
            metadata={
                "source": str(Path(args.input).resolve()),
                "source_index_sha256": _sha256(
                    Path(args.input) / "model.safetensors.index.json"
                ),
                "recipe": str(recipe_path),
                "recipe_sha256": _sha256(recipe_path),
                "allocation": str(allocation_path),
                "allocation_sha256": _sha256(allocation_path),
                "imatrix": str(imatrix.matrix.path),
                "imatrix_sha256": _sha256(imatrix.matrix.path),
                "imatrix_datasets": list(imatrix.matrix.datasets),
                "imatrix_chunk_count": imatrix.matrix.chunk_count,
                "imatrix_chunk_size": imatrix.matrix.chunk_size,
                "expert_objective": "count_weighted_diagonal_imatrix",
                "expert_profile": args.expert_profile,
                "mtp_included": False,
            },
            candidate_table={},
        )
        save_scheme(scheme_path, scheme)
    else:
        load_scheme(scheme_path)
    summary = {
        "run_dir": str(run_dir),
        "recipe_tensors": len(recipe),
        "mapped_nonexpert_tensors": len(source_map),
        "estimated_blob_gb": allocation.estimated_blob_bytes / 1e9,
        "headroom_mb": (
            allocation.target_bytes - allocation.estimated_blob_bytes
        )
        / 1e6,
        "gate_up_energy_percent": 100 * allocation.gate_up_energy_fraction,
        "down_energy_percent": 100 * allocation.down_energy_fraction,
        "scheme": str(scheme_path),
    }
    if isinstance(allocation, V4FEwAllocation):
        summary["gate_up_high"] = sum(
            map(len, allocation.gate_up_high.values())
        )
        summary["down_high"] = sum(map(len, allocation.down_high.values()))
    else:
        summary["gate_up_nint4"] = sum(
            map(len, allocation.gate_up_nint4.values())
        )
        summary["down_nint4"] = sum(
            map(len, allocation.down_nint4.values())
        )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def _energy_map(path: str | Path, metric: str) -> dict[tuple[int, int], float]:
    result = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[(int(row["layer"]), int(row["expert_id"]))] = float(
                row[metric]
            )
    return result


def _artifact_contract(
    args,
    layer: int,
    projection: str,
    family: str,
    high_experts: tuple[int, ...],
    expert_importance: V4FExpertImportance,
) -> dict:
    source_root = Path(args.input).resolve()
    return {
        "format": "mfq.v4f-codebook-contract.v1",
        "source": str(source_root),
        "source_index_sha256": _sha256(
            source_root / "model.safetensors.index.json"
        ),
        "layer": layer,
        "projection": projection,
        "family": family,
        "sample_rows_per_expert": args.train_rows,
        "sample_seed": args.sample_seed,
        "high_experts": list(high_experts),
        "objective": "count_weighted_diagonal_imatrix",
        "imatrix": str(Path(args.imatrix).resolve()),
        "imatrix_sha256": _sha256(args.imatrix),
        "imatrix_entries": list(expert_importance.entry_names),
        "imatrix_count_min": int(expert_importance.counts.min()),
        "imatrix_count_max": int(expert_importance.counts.max()),
        "rotation_block": _ROTATION_BLOCK if family == "NEPQ0-S" else 0,
        "rotation_seed": _ROTATION_SEED if family == "NEPQ0-S" else 0,
        "nepq_train": dataclasses.asdict(
            NepqBankTrainConfig(
                iterations=args.bank_iterations,
                assignment_refine_steps=args.bank_refine_steps,
                kmeans_iterations=args.kmeans_iterations,
                expert_batch=args.expert_batch,
                seed=args.sample_seed,
            )
        ),
        "nvq2j_train": dataclasses.asdict(
            NvqJscConfig(
                banks=4,
                iterations=args.jsc_iterations,
                assignment_refine_steps=args.jsc_refine_steps,
            )
        ),
    }


def _artifact_matches(path: Path, contract: dict) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as payload:
        if "contract_json" not in payload:
            raise ValueError(f"artifact has no contract: {path}")
        current = _json_from_array(payload["contract_json"])
    if _canonical(current) != _canonical(contract):
        raise ValueError(f"artifact contract mismatch: {path}")
    return True


def _parse_layers(value: str) -> list[int]:
    if not value:
        return list(range(43))
    result: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            result.update(range(start, stop + 1))
        else:
            result.add(int(part))
    if any(layer < 0 or layer >= 43 for layer in result):
        raise ValueError("V4F layers must be in [0,42]")
    return sorted(result)


def _layer_source_shards(
    weight_map: dict[str, str],
    layer: int,
) -> tuple[str, ...]:
    prefix = f"layers.{layer}.ffn.experts."
    shards = {
        shard
        for name, shard in weight_map.items()
        if name.startswith(prefix)
        and (name.endswith(".weight") or name.endswith(".scale"))
    }
    if not shards:
        raise ValueError(f"V4F layer {layer} has no routed-expert source")
    return tuple(sorted(shards))


def _source_shards_ready(
    source_root: Path,
    shards: tuple[str, ...],
    expected_sizes: dict[str, int],
) -> bool:
    for shard in shards:
        if shard not in expected_sizes:
            raise ValueError(f"source manifest has no shard {shard}")
        path = source_root / shard
        if (
            not path.is_file()
            or path.stat().st_size != expected_sizes[shard]
        ):
            return False
    return True


def _atomic_state(path: Path, document: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _train_tiered_codebooks_as_shards_arrive(
    args,
    allocation: V4FTieredAllocation,
) -> None:
    if args.poll_seconds <= 0:
        raise ValueError("source shard poll interval must be positive")
    source_root = Path(args.input).resolve()
    manifest_path = Path(args.wait_source_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sizes = {
        str(name): int(size)
        for name, size in manifest.get("shards", {}).items()
    }
    if len(expected_sizes) != 46:
        raise ValueError("V4F source manifest must contain 46 shards")
    checkpoint = V4FCheckpoint(source_root)
    requested = _parse_layers(args.layers)
    required = {
        layer: _layer_source_shards(checkpoint.weight_map, layer)
        for layer in requested
    }
    pending = set(requested)
    artifact_dir = Path(args.run_dir).resolve() / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_path = (
        Path(args.run_dir).resolve()
        / "incremental_codebook_state.json"
    )
    completion_marker = (
        Path(args.source_complete_marker).resolve()
        if args.source_complete_marker
        else None
    )
    original_layers = args.layers
    while pending:
        ready = []
        for layer in sorted(pending):
            artifacts_present = all(
                (
                    artifact_dir
                    / _artifact_name(layer, projection, "NVQ2J")
                ).is_file()
                for projection in ("gate_up", "down")
            )
            if artifacts_present or _source_shards_ready(
                source_root, required[layer], expected_sizes
            ):
                ready.append(layer)
        _atomic_state(
            state_path,
            {
                "status": "training" if ready else "waiting_for_shards",
                "completed_layers": len(requested) - len(pending),
                "total_layers": len(requested),
                "ready_layers": ready,
                "pending_layers": sorted(pending),
                "updated_unix": time.time(),
            },
        )
        if ready:
            args.layers = ",".join(str(layer) for layer in ready)
            _train_tiered_codebooks(args, allocation)
            pending.difference_update(ready)
            continue
        if completion_marker is not None and completion_marker.is_file():
            missing = {
                layer: required[layer]
                for layer in sorted(pending)
                if not _source_shards_ready(
                    source_root, required[layer], expected_sizes
                )
            }
            if missing:
                raise RuntimeError(
                    "verified source download is missing routed shards: "
                    f"{missing}"
                )
        time.sleep(args.poll_seconds)
    args.layers = original_layers
    _atomic_state(
        state_path,
        {
            "status": "complete",
            "completed_layers": len(requested),
            "total_layers": len(requested),
            "pending_layers": [],
            "updated_unix": time.time(),
        },
    )


def _train_tiered_codebooks(
    args,
    allocation: V4FTieredAllocation,
) -> None:
    checkpoint = V4FCheckpoint(args.input)
    imatrix = V4FImportanceMatrix.load(args.imatrix)
    args.imatrix_sha256 = _sha256(args.imatrix)
    layers = _parse_layers(args.layers)
    artifact_dir = Path(args.run_dir).resolve() / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.run_dir).resolve() / "codebook_progress.jsonl"
    total = len(layers) * 2
    completed = 0
    started = time.perf_counter()
    for layer in layers:
        for projection, nint4_map in (
            ("gate_up", allocation.gate_up_nint4),
            ("down", allocation.down_nint4),
        ):
            nint4 = set(nint4_map.get(layer, ()))
            nvq2j_experts = tuple(
                expert for expert in range(256) if expert not in nint4
            )
            artifact_path = artifact_dir / _artifact_name(
                layer, projection, "NVQ2J"
            )
            expert_importance = imatrix.expert(layer, projection)
            contract = (
                None
                if not nvq2j_experts
                else _artifact_contract(
                    args,
                    layer,
                    projection,
                    "NVQ2J",
                    nvq2j_experts,
                    expert_importance,
                )
            )
            done = contract is None or _artifact_matches(
                artifact_path, contract
            )
            item_started = time.perf_counter()
            if not done:
                samples = checkpoint.expert_source(
                    layer, projection
                ).sample_experts(
                    args.train_rows,
                    seed=args.sample_seed,
                    device=args.device,
                )
                expert_ids = np.asarray(nvq2j_experts, dtype=np.int64)
                ids = torch.as_tensor(
                    expert_ids,
                    device=samples.device,
                    dtype=torch.int64,
                )
                selected = samples.index_select(0, ids).reshape(
                    -1, samples.shape[-1]
                )
                input_objective = torch.as_tensor(
                    expert_importance.count_weighted(expert_ids),
                    device=samples.device,
                    dtype=torch.float32,
                )
                importance = (
                    input_objective[:, None, :]
                    .expand(-1, args.train_rows, -1)
                    .reshape_as(selected)
                    .contiguous()
                )
                tensor, history = train_nvq_jsc(
                    selected,
                    importance=importance,
                    config=NvqJscConfig(
                        banks=4,
                        iterations=args.jsc_iterations,
                        assignment_refine_steps=args.jsc_refine_steps,
                    ),
                    device=args.device,
                )
                tables = jsc_tables_from_tensor(tensor)
                _atomic_npz(
                    artifact_path,
                    contract_json=_json_array(contract),
                    history_json=_json_array(
                        [dataclasses.asdict(item) for item in history]
                    ),
                    scale_lut=tables.scale_lut,
                    bank_for_state=tables.bank_for_state,
                    codebooks=tables.codebooks,
                )
                del samples, selected, importance, tensor
                torch.cuda.empty_cache()
            completed += 1
            elapsed = time.perf_counter() - item_started
            total_elapsed = time.perf_counter() - started
            event = {
                "completed": completed,
                "total": total,
                "layer": layer,
                "projection": projection,
                "nvq2j_experts": len(nvq2j_experts),
                "nint4_experts": len(nint4),
                "seconds": elapsed,
                "status": "reused" if done else "trained",
                "eta_seconds": total_elapsed / completed * (total - completed),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)


def train_codebooks(args) -> None:
    run_dir = Path(args.run_dir).resolve()
    allocation = _load_allocation(run_dir / "allocation.json")
    if isinstance(allocation, V4FTieredAllocation):
        if getattr(args, "wait_source_manifest", ""):
            _train_tiered_codebooks_as_shards_arrive(
                args, allocation
            )
        else:
            _train_tiered_codebooks(args, allocation)
        return
    if getattr(args, "wait_source_manifest", ""):
        raise ValueError(
            "incremental shard training requires a tiered allocation"
        )
    checkpoint = V4FCheckpoint(args.input)
    imatrix = V4FImportanceMatrix.load(args.imatrix)
    args.imatrix_sha256 = _sha256(args.imatrix)
    layers = _parse_layers(args.layers)
    artifact_dir = run_dir / "artifacts"
    progress_path = run_dir / "codebook_progress.jsonl"
    total = len(layers) * 2
    completed = 0
    started = time.perf_counter()
    for layer in layers:
        for projection, high_map in (
            ("gate_up", allocation.gate_up_high),
            ("down", allocation.down_high),
        ):
            high_experts = high_map.get(layer, ())
            expert_importance = imatrix.expert(layer, projection)
            low_path = artifact_dir / _artifact_name(
                layer, projection, "NEPQ0-S"
            )
            high_path = artifact_dir / _artifact_name(
                layer, projection, "NVQ2J"
            )
            low_contract = _artifact_contract(
                args,
                layer,
                projection,
                "NEPQ0-S",
                high_experts,
                expert_importance,
            )
            high_contract = _artifact_contract(
                args,
                layer,
                projection,
                "NVQ2J",
                high_experts,
                expert_importance,
            )
            low_done = _artifact_matches(low_path, low_contract)
            high_done = not high_experts or _artifact_matches(
                high_path, high_contract
            )
            item_started = time.perf_counter()
            if not low_done or not high_done:
                samples = checkpoint.expert_source(
                    layer, projection
                ).sample_experts(
                    args.train_rows,
                    seed=args.sample_seed,
                    device=args.device,
                )
                if not low_done:
                    trained = train_nepq0_s_banks(
                        samples,
                        importance=expert_importance.count_weighted(),
                        config=NepqBankTrainConfig(
                            iterations=args.bank_iterations,
                            assignment_refine_steps=args.bank_refine_steps,
                            kmeans_iterations=args.kmeans_iterations,
                            expert_batch=args.expert_batch,
                            seed=args.sample_seed,
                        ),
                        rotation_block=_ROTATION_BLOCK,
                        rotation_seed=_ROTATION_SEED,
                    )
                    _atomic_npz(
                        low_path,
                        contract_json=_json_array(low_contract),
                        table_payloads=trained.table_payloads,
                        weighted_nmse_percent=trained.weighted_nmse_percent,
                    )
                if not high_done:
                    expert_ids = np.asarray(high_experts, dtype=np.int64)
                    ids = torch.as_tensor(
                        expert_ids,
                        device=samples.device,
                        dtype=torch.int64,
                    )
                    selected = samples.index_select(0, ids).reshape(
                        -1, samples.shape[-1]
                    )
                    input_objective = torch.as_tensor(
                        expert_importance.count_weighted(expert_ids),
                        device=samples.device,
                        dtype=torch.float32,
                    )
                    importance = (
                        input_objective[:, None, :]
                        .expand(-1, args.train_rows, -1)
                        .reshape_as(selected)
                        .contiguous()
                    )
                    tensor, history = train_nvq_jsc(
                        selected,
                        importance=importance,
                        config=NvqJscConfig(
                            banks=4,
                            iterations=args.jsc_iterations,
                            assignment_refine_steps=args.jsc_refine_steps,
                        ),
                        device=args.device,
                    )
                    tables = jsc_tables_from_tensor(tensor)
                    _atomic_npz(
                        high_path,
                        contract_json=_json_array(high_contract),
                        history_json=_json_array(
                            [dataclasses.asdict(item) for item in history]
                        ),
                        scale_lut=tables.scale_lut,
                        bank_for_state=tables.bank_for_state,
                        codebooks=tables.codebooks,
                    )
                    del selected, importance, tensor
                del samples
                torch.cuda.empty_cache()
            completed += 1
            elapsed = time.perf_counter() - item_started
            total_elapsed = time.perf_counter() - started
            event = {
                "completed": completed,
                "total": total,
                "layer": layer,
                "projection": projection,
                "high_experts": len(high_experts),
                "seconds": elapsed,
                "status": "reused" if low_done and high_done else "trained",
                "eta_seconds": (
                    total_elapsed / completed * (total - completed)
                ),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)


def _plans(
    args,
    checkpoint: V4FCheckpoint,
    recipe: list[RecipeTensor],
    scheme: CalibrationScheme,
) -> list[TensorPlan]:
    source_map = recipe_source_map(checkpoint, recipe)
    plans: list[TensorPlan] = []
    for item in recipe:
        if item.name not in source_map:
            continue
        source = source_map[item.name]
        plans.append(
            TensorPlan(
                name=item.name,
                shard=checkpoint.shard_for(source),
                shape=item.shape,
                source_dtype=checkpoint.info(source).dtype,
                target_dtype=recipe_target_dtype(item.dtype),
                gguf_name=item.name,
                gguf_type=item.dtype,
                source_name=source,
            )
        )
    for layer in range(43):
        for projection in ("gate_up", "down"):
            name = f"blk.{layer}.ffn_{projection}_exps.weight"
            selection = scheme.require_expert(name)
            shape = (
                selection.n_experts,
                selection.rows_per_expert,
                selection.columns,
            )
            source = checkpoint.expert_source(layer, projection)
            plans.append(
                TensorPlan(
                    name=name,
                    shard=checkpoint.shard_for(
                        f"layers.{layer}.ffn.experts.0."
                        f"{'w1' if projection == 'gate_up' else 'w2'}.weight"
                    ),
                    shape=shape,
                    source_dtype="MXFP4",
                    target_dtype="NINTM",
                    expert_shape=shape,
                    expert_precisions=selection.precisions,
                    transform=f"v4f_{projection}",
                )
            )
            del source
    normal = [item for item in plans if item.target_dtype != "NINTM"]
    routed = [item for item in plans if item.target_dtype == "NINTM"]
    normal.sort(
        key=lambda item: (
            item.shard,
            checkpoint.info(item.source_name or "").data_start,
        )
    )
    routed.sort(key=lambda item: item.name)
    return normal + routed


def _read_dense_source(
    checkpoint: V4FCheckpoint,
    name: str,
    target_dtype: str,
) -> torch.Tensor:
    reader = checkpoint.reader_for(name)
    info = reader.info(name)
    value = read_dense_rows(
        reader,
        name,
        0,
        info.shape[0],
        device="cpu",
        dtype=None if target_dtype in {"I32", "I64"} else torch.float32,
    )
    return value.reshape(info.shape)


def convert(args) -> None:
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    split_max_size = int(getattr(args, "split_max_size", 0))
    split_max_tensors = int(getattr(args, "split_max_tensors", 0))
    validate_split_limits(split_max_size, split_max_tensors)
    if output.exists() or matching_shard_paths(output):
        raise FileExistsError(f"V4F MFQ output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe_path = Path(
        getattr(args, "recipe", "") or run_dir / "recipe.json"
    ).resolve()
    scheme_path = Path(
        getattr(args, "scheme", "") or run_dir / "scheme.json"
    ).resolve()
    recipe, _ = load_recipe(recipe_path)
    scheme = load_scheme(scheme_path)
    routed_families = {
        item.descriptor.family
        for selection in scheme.expert_selections.values()
        for item in selection.selections
    }
    tpq_mode = bool(routed_families) and routed_families <= {
        "TPQ-X",
        "TPQ-W",
        "TPQ-V",
        "TPQ-VV",
    }
    allocation = None
    allocation_document = None
    imatrix = None
    imatrix_sha256 = None
    if tpq_mode:
        if scheme.metadata.get("codebook_objective") != "euclidean_sse":
            raise ValueError("TPQ scheme is not an original Euclidean scheme")
    else:
        allocation = _load_allocation(run_dir / "allocation.json")
        allocation_document = json.loads(
            (run_dir / "allocation.json").read_text(encoding="utf-8")
        )
        imatrix_sha256 = _sha256(args.imatrix)
        if allocation_document.get("imatrix_sha256") != imatrix_sha256:
            raise ValueError("V4F conversion imatrix differs from the prepared run")
        imatrix = V4FImportanceMatrix.load(args.imatrix)
    checkpoint = V4FCheckpoint(args.input)
    plans = _plans(args, checkpoint, recipe, scheme)
    artifact_root = scheme.path.parent
    estimated = sum(
        _plan_blob_nbytes(item, _DEFAULT_NINT, artifact_root)
        for item in plans
    )
    if allocation is not None and estimated != allocation.estimated_blob_bytes:
        raise ValueError(
            f"V4F plan size changed: {estimated} != "
            f"{allocation.estimated_blob_bytes}"
        )
    temp_dir = (
        Path(args.temp_dir).resolve()
        if args.temp_dir
        else run_dir / "blobs"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "convert_progress.jsonl"
    records: list[BlobRecord] = []
    started = time.perf_counter()
    for index, item in enumerate(plans, start=1):
        blob_path = temp_dir / f"{index:05d}.blob"
        expected = _plan_blob_nbytes(item, _DEFAULT_NINT, artifact_root)
        if blob_path.is_file() and blob_path.stat().st_size == expected:
            records.append(
                BlobRecord(item.name, item.target_dtype, expected, blob_path)
            )
            status = "reused"
            elapsed = 0.0
        else:
            if blob_path.exists():
                raise ValueError(f"partial V4F blob has the wrong size: {blob_path}")
            partial_blob = blob_path.with_suffix(blob_path.suffix + ".partial")
            if partial_blob.exists():
                partial_blob.unlink()
            item_started = time.perf_counter()
            if item.target_dtype == "NINTM":
                if item.expert_shape is None or item.expert_precisions is None:
                    raise ValueError(f"invalid V4F routed plan: {item.name}")
                layer = int(item.name.split(".")[1])
                projection = (
                    "gate_up" if "gate_up" in item.name else "down"
                )
                source = checkpoint.expert_source(layer, projection)
                expert_importance = (
                    None
                    if imatrix is None
                    else imatrix.expert(layer, projection).values
                )
                nbytes = _write_mixed_moe_axis0_blob(
                    source,
                    item.expert_shape,
                    item.expert_shape,
                    item.expert_precisions,
                    partial_blob,
                    args.row_chunk,
                    "cuda",
                    args.device,
                    artifact_root,
                    importance=expert_importance,
                )
                del source
            elif item.target_dtype.startswith("NINT"):
                source = checkpoint.tensor_source(item.source_name or "")
                nbytes = _write_nint_axis0_blob(
                    source,
                    item.shape,
                    _spec_for_plan(item, _DEFAULT_NINT),
                    partial_blob,
                    args.row_chunk,
                    "cuda",
                    args.device,
                )
                del source
            else:
                source = _read_dense_source(
                    checkpoint,
                    item.source_name or "",
                    item.target_dtype,
                )
                nbytes = _dense_blob_from_tensor(
                    source, partial_blob, item.target_dtype
                )
                del source
            if nbytes != expected:
                raise RuntimeError(
                    f"V4F blob size mismatch for {item.name}: "
                    f"{nbytes} != {expected}"
                )
            os.replace(partial_blob, blob_path)
            records.append(
                BlobRecord(item.name, item.target_dtype, nbytes, blob_path)
            )
            status = "written"
            elapsed = time.perf_counter() - item_started
        total_elapsed = time.perf_counter() - started
        event = {
            "completed": index,
            "total": len(plans),
            "name": item.name,
            "dtype": item.target_dtype,
            "blob_mb": expected / 1e6,
            "seconds": elapsed,
            "status": status,
            "eta_seconds": total_elapsed / index * (len(plans) - index),
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    if tpq_mode:
        expert_metadata = {
            "expert_families": sorted(routed_families),
            "expert_allocation_objective": scheme.metadata.get(
                "allocator",
                "tpq-per-layer-routing-energy",
            ),
            "expert_codebook_objective": "euclidean_sse",
            "expert_audit_metric": scheme.metadata.get("audit_metric"),
            "expert_audit_upgrade_factor": scheme.metadata.get(
                "audit_upgrade_factor"
            ),
        }
    elif isinstance(allocation, V4FTieredAllocation):
        expert_metadata = {
            "expert_base_family": "NVQ2J",
            "expert_upgrade_family": "NINT4",
            "expert_allocation_objective": (
                "reap_output_energy_per_upgrade_byte"
            ),
        }
    else:
        expert_metadata = {
            "expert_low_family": "NEPQ0-S",
            "expert_high_family": "NVQ2J",
            "rotation_block": _ROTATION_BLOCK,
            "rotation_seed": _ROTATION_SEED,
        }
    extra = {
        "source": str(Path(args.input).resolve()),
        "source_index_sha256": _sha256(
            Path(args.input) / "model.safetensors.index.json"
        ),
        "recipe_sha256": _sha256(recipe_path),
        "scheme_sha256": _sha256(scheme_path),
        "target_bytes": (
            estimated if allocation is None else allocation.target_bytes
        ),
        "estimated_blob_bytes": estimated,
        "mtp_included": False,
        "expert_objective": (
            "euclidean_sse"
            if tpq_mode
            else "count_weighted_diagonal_imatrix"
        ),
        **expert_metadata,
    }
    if not tpq_mode:
        assert allocation is not None
        assert imatrix is not None
        assert imatrix_sha256 is not None
        extra.update(
            {
                "allocation_sha256": _sha256(run_dir / "allocation.json"),
                "imatrix": str(Path(args.imatrix).resolve()),
                "imatrix_sha256": imatrix_sha256,
                "imatrix_datasets": list(imatrix.matrix.datasets),
                "imatrix_chunk_count": imatrix.matrix.chunk_count,
                "imatrix_chunk_size": imatrix.matrix.chunk_size,
            }
        )
    runtime_profile = architecture_profile("deepseek_v4")
    if runtime_profile is not None:
        extra[RUNTIME_SAMPLING_METADATA_KEY] = runtime_profile
    header = FileHeader(
        version=2,
        model_arch=(
            "deepseek_v4-tpq-mfq"
            if tpq_mode
            else "deepseek_v4-ew-mfq"
        ),
        num_tensors=len(records),
        extra=extra,
    )
    outputs = write_blob_record_shards(
        output,
        header,
        records,
        split_max_size=split_max_size,
        split_max_tensors=split_max_tensors,
        consume_blobs=True,
    )
    output_bytes = sum(path.stat().st_size for path in outputs)
    print(
        json.dumps(
            {
                "output": str(outputs[0]),
                "outputs": [str(path) for path in outputs],
                "shard_count": len(outputs),
                "bytes": output_bytes,
                "gb": output_bytes / 1e9,
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reap-csv", required=True)
    parser.add_argument("--imatrix", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    _common(prepare_parser)
    prepare_parser.add_argument("--recipe-headers", nargs="+", required=True)
    prepare_parser.add_argument("--target-bytes", type=int, default=40_000_000_000)
    prepare_parser.add_argument(
        "--expert-profile",
        choices=("nepq0s-nvq2j", "nvq2j-nint4"),
        default="nepq0s-nvq2j",
    )
    prepare_parser.add_argument("--assignment-row-chunk", type=int, default=512)
    prepare_parser.add_argument("--assignment-bank-chunk", type=int, default=8)
    prepare_parser.add_argument(
        "--admm-iterations", type=int, default=_DEFAULT_ADMM_ITERATIONS
    )
    prepare_parser.add_argument(
        "--admm-rho", type=float, default=_DEFAULT_ADMM_RHO
    )
    prepare_parser.set_defaults(func=prepare)

    train_parser = subparsers.add_parser("train-codebooks")
    _common(train_parser)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--layers", default="")
    train_parser.add_argument("--train-rows", type=int, default=64)
    train_parser.add_argument("--sample-seed", type=int, default=20260723)
    train_parser.add_argument("--bank-iterations", type=int, default=3)
    train_parser.add_argument("--bank-refine-steps", type=int, default=1)
    train_parser.add_argument("--kmeans-iterations", type=int, default=6)
    train_parser.add_argument("--expert-batch", type=int, default=32)
    train_parser.add_argument("--jsc-iterations", type=int, default=4)
    train_parser.add_argument("--jsc-refine-steps", type=int, default=2)
    train_parser.add_argument("--wait-source-manifest", default="")
    train_parser.add_argument("--source-complete-marker", default="")
    train_parser.add_argument("--poll-seconds", type=int, default=30)
    train_parser.set_defaults(func=train_codebooks)

    convert_parser = subparsers.add_parser("convert")
    _common(convert_parser)
    convert_parser.add_argument("--output", required=True)
    convert_parser.add_argument("--temp-dir", default="")
    convert_parser.add_argument("--device", default="cuda")
    convert_parser.add_argument("--row-chunk", type=int, default=512)
    split = convert_parser.add_mutually_exclusive_group()
    split.add_argument(
        "--split-max-size",
        type=parse_size,
        default=0,
        metavar="N[M|G]",
    )
    split.add_argument("--split-max-tensors", type=int, default=0)
    convert_parser.set_defaults(func=convert)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
