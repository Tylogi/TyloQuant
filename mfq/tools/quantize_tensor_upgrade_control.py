"""Build the whole-tensor precision-upgrade control for IN experiments.

The control spends the same extra byte budget as an Important-Neuron model,
but only raises the precision of complete existing tensors.  Every unchanged
record is copied byte-for-byte from the baseline MFQ; selected tensors are
requantized directly from the BF16 GGUF.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mfq.formats.assets import ASSET_DTYPE, ASSET_PREFIX
from mfq.formats.header import FileHeader
from mfq.formats.io import open_mmap
from mfq.formats.shards import write_blob_record_shards
from mfq.quantize.imatrix import load_importance_matrix
from mfq.tools.quantize_gguf_to_mfq import (
    GgufRowSource,
    GgufTensorPlan,
    _NINT_SPECS,
    _build_plan,
    _estimate_blob_bytes,
    _load_gguf,
    _write_nint8_zero_axis0_blob,
)
from mfq.tools.quantize_hf_to_mfq import (
    BlobRecord,
    _write_nint_axis0_blob,
)
from mfq.tools.quantize_important_neurons import (
    FileSpanRecord,
    _atomic_json,
    _binding,
    _container_overhead,
    _file_identity,
    _importance_vector,
    _record_span,
)


_NEXT_DTYPE = {
    "NINT3": "NINT4",
    "NVQ3J": "NINT4",
    "NINT4": "NINT5",
    "NINT5": "NINT6",
    "NINT6": "NINT8-0",
}
_QUALITY_BITS = {
    "NINT3": 3.0,
    "NVQ3J": 3.35,
    "NINT4": 4.0,
    "NINT5": 5.0,
    "NINT6": 6.0,
    "NINT8-0": 8.0,
}
CONTROL_PADDING_ASSET_NAME = (
    ASSET_PREFIX + "tensor_precision_control.padding"
)


@dataclass(frozen=True)
class Candidate:
    item: GgufTensorPlan
    low_dtype: str
    high_dtype: str
    old_nbytes: int
    new_nbytes: int
    utility: float

    @property
    def cost(self) -> int:
        return self.new_nbytes - self.old_nbytes


def _fused_peer_name(name: str) -> str | None:
    replacements = (
        (r"^(.*\.ffn_)gate(\.weight)$", r"\1up\2"),
        (r"^(.*\.ffn_)up(\.weight)$", r"\1gate\2"),
        (r"^(.*\.)gate_proj(\.weight)$", r"\1up_proj\2"),
        (r"^(.*\.)up_proj(\.weight)$", r"\1gate_proj\2"),
        (r"^(.*\.attn_)q(\.weight)$", r"\1k\2"),
        (r"^(.*\.attn_)k(\.weight)$", r"\1q\2"),
        (r"^(.*\.)q_proj(\.weight)$", r"\1k_proj\2"),
        (r"^(.*\.)k_proj(\.weight)$", r"\1q_proj\2"),
    )
    for pattern, replacement in replacements:
        if re.fullmatch(pattern, name):
            return re.sub(pattern, replacement, name)
    return None


def _candidate_units(
    candidates: list[Candidate],
) -> list[tuple[Candidate, ...]]:
    by_name = {
        candidate.item.name: candidate for candidate in candidates
    }
    visited: set[str] = set()
    units: list[tuple[Candidate, ...]] = []
    for candidate in candidates:
        name = candidate.item.name
        if name in visited:
            continue
        peer_name = _fused_peer_name(name)
        if peer_name is None:
            units.append((candidate,))
            visited.add(name)
            continue
        peer = by_name.get(peer_name)
        if peer is None:
            visited.add(name)
            continue
        visited.update((name, peer_name))
        if (
            candidate.low_dtype != peer.low_dtype
            or candidate.high_dtype != peer.high_dtype
        ):
            raise ValueError(
                f"fused control candidates have different precision steps: "
                f"{name}, {peer_name}"
            )
        first, second = sorted(
            (candidate, peer), key=lambda item: item.item.name
        )
        units.append((first, second))
    return units


def _candidate_utility(
    item: GgufTensorPlan,
    low_dtype: str,
    high_dtype: str,
    importance: np.ndarray,
) -> float:
    baseline_error = 2.0 ** (-2.0 * _QUALITY_BITS[low_dtype])
    candidate_error = 2.0 ** (-2.0 * _QUALITY_BITS[high_dtype])
    rows = int(item.storage_shape[0])
    return (
        float(np.sum(importance, dtype=np.float64))
        * rows
        * (baseline_error - candidate_error)
    )


def _build_candidates(
    plan: list[GgufTensorPlan],
    store,
    imatrix,
) -> list[Candidate]:
    result: list[Candidate] = []
    for item in plan:
        record = store.records.get(item.name)
        if record is None:
            continue
        high_dtype = _NEXT_DTYPE.get(record.dtype)
        if high_dtype is None:
            continue
        try:
            _entry, importance = _importance_vector(
                imatrix, item
            )
        except KeyError:
            continue
        if high_dtype == "NINT8-0" and item.storage_shape[1] % 32:
            continue
        upgraded = replace(item, target_dtype=high_dtype)
        new_nbytes = _estimate_blob_bytes(upgraded)
        if new_nbytes <= record.nbytes:
            continue
        result.append(
            Candidate(
                item=item,
                low_dtype=record.dtype,
                high_dtype=high_dtype,
                old_nbytes=record.nbytes,
                new_nbytes=new_nbytes,
                utility=_candidate_utility(
                    item,
                    record.dtype,
                    high_dtype,
                    importance,
                ),
            )
        )
    if not result:
        raise ValueError("no imatrix-bound whole-tensor upgrades are available")
    return result


def _header(
    baseline: FileHeader,
    *,
    source: Path,
    recipe: Path,
    imatrix: Path,
    baseline_model: Path,
    target_bytes: int,
    selected: list[Candidate],
) -> FileHeader:
    extra = dict(baseline.extra)
    extra["tensor_precision_control"] = {
        "version": 1,
        "method": "whole_tensor_next_precision_knapsack",
        "source_bf16": str(source),
        "recipe": str(recipe),
        "imatrix": str(imatrix),
        "baseline_mfq": str(baseline_model),
        "target_bytes": target_bytes,
        "upgrades": [
            {
                "name": candidate.item.name,
                "from": candidate.low_dtype,
                "to": candidate.high_dtype,
                "cost_bytes": candidate.cost,
            }
            for candidate in selected
        ],
    }
    return FileHeader(
        version=max(2, int(baseline.version)),
        model_arch=baseline.model_arch,
        num_tensors=0,
        extra=extra,
    )


def _metadata(store, selected: dict[str, Candidate]):
    return [
        (
            record.name,
            (
                selected[record.name].high_dtype
                if record.name in selected
                else record.dtype
            ),
            (
                selected[record.name].new_nbytes
                if record.name in selected
                else record.nbytes
            ),
        )
        for record in store.records.values()
    ]


def _total_size(
    store,
    baseline_header: FileHeader,
    selected: list[Candidate],
    header_args: dict[str, Any],
) -> tuple[int, FileHeader]:
    selected_by_name = {
        candidate.item.name: candidate for candidate in selected
    }
    header = _header(
        baseline_header, selected=selected, **header_args
    )
    metadata = _metadata(store, selected_by_name)
    return (
        sum(nbytes for _, _, nbytes in metadata)
        + _container_overhead(header, metadata),
        header,
    )


def _select_candidates(
    store,
    baseline_header: FileHeader,
    candidates: list[Candidate],
    target_bytes: int,
    header_args: dict[str, Any],
    unit: int = 16 * 1024,
) -> tuple[list[Candidate], FileHeader, int, int]:
    units = _candidate_units(candidates)
    empty_total, _ = _total_size(
        store, baseline_header, [], header_args
    )
    budget = target_bytes - empty_total
    if budget <= 0:
        raise ValueError(
            f"target does not exceed baseline container size: {budget}"
        )
    capacity = budget // unit
    negative = -np.inf
    dp = np.full(capacity + 1, negative, dtype=np.float64)
    dp[0] = 0.0
    take = np.zeros(
        (len(units), capacity + 1), dtype=np.bool_
    )
    previous = np.full(
        (len(units), capacity + 1), -1, dtype=np.int32
    )
    for index, unit_candidates in enumerate(units):
        next_dp = dp.copy()
        previous[index, :] = np.arange(
            capacity + 1, dtype=np.int32
        )
        unit_cost = sum(
            candidate.cost for candidate in unit_candidates
        )
        unit_utility = sum(
            candidate.utility for candidate in unit_candidates
        )
        cost_units = math.ceil(unit_cost / unit)
        if cost_units <= capacity:
            source = dp[: capacity + 1 - cost_units]
            values = source + unit_utility
            target = next_dp[cost_units:]
            better = values > target
            positions = np.flatnonzero(better) + cost_units
            next_dp[positions] = values[better]
            take[index, positions] = True
            previous[index, positions] = positions - cost_units
        dp = next_dp

    cursor = int(np.nanargmax(dp))
    selected_unit_indices: set[int] = set()
    for index in range(len(units) - 1, -1, -1):
        if take[index, cursor]:
            selected_unit_indices.add(index)
        cursor = int(previous[index, cursor])
        if cursor < 0:
            raise RuntimeError("whole-tensor knapsack path is invalid")
    selected_units = [
        unit_candidates
        for index, unit_candidates in enumerate(units)
        if index in selected_unit_indices
    ]
    order = {
        candidate.item.name: index
        for index, candidate in enumerate(candidates)
    }

    def flatten(
        values: list[tuple[Candidate, ...]],
    ) -> list[Candidate]:
        return sorted(
            (
                candidate
                for unit_candidates in values
                for candidate in unit_candidates
            ),
            key=lambda candidate: order[candidate.item.name],
        )

    selected = flatten(selected_units)
    total, header = _total_size(
        store, baseline_header, selected, header_args
    )
    while True:
        remaining = target_bytes - total
        available = [
            unit_candidates
            for unit_candidates in units
            if unit_candidates not in selected_units
            and sum(
                candidate.cost for candidate in unit_candidates
            ) <= remaining
        ]
        if not available:
            break
        chosen = max(
            available,
            key=lambda unit_candidates: (
                sum(
                    candidate.utility
                    for candidate in unit_candidates
                )
                / sum(
                    candidate.cost
                    for candidate in unit_candidates
                )
            ),
        )
        trial_units = selected_units + [chosen]
        trial = flatten(trial_units)
        trial_total, trial_header = _total_size(
            store, baseline_header, trial, header_args
        )
        if trial_total > target_bytes:
            units = [
                unit_candidates
                for unit_candidates in units
                if unit_candidates is not chosen
            ]
            continue
        selected_units = trial_units
        selected, total, header = trial, trial_total, trial_header
    remaining = target_bytes - total
    padding_overhead = (
        4
        + len(CONTROL_PADDING_ASSET_NAME.encode("utf-8"))
        + 4
        + len(ASSET_DTYPE.encode("utf-8"))
        + 8
    )
    padding_nbytes = (
        remaining - padding_overhead
        if remaining >= padding_overhead
        else 0
    )
    exact_total = (
        total + padding_overhead + padding_nbytes
        if padding_nbytes
        else total
    )
    return selected, header, exact_total, padding_nbytes


def _quantize_candidate(
    source: GgufRowSource,
    candidate: Candidate,
    blob_path: Path,
    *,
    imatrix,
    row_chunk: int,
    device: str,
) -> int:
    entry_name, importance = _importance_vector(
        imatrix, candidate.item
    )
    binding = _binding(entry_name, importance)
    if candidate.high_dtype == "NINT8-0":
        return _write_nint8_zero_axis0_blob(
            source,
            candidate.item.storage_shape,
            blob_path,
            row_chunk,
        )
    return _write_nint_axis0_blob(
        source,
        candidate.item.storage_shape,
        _NINT_SPECS[candidate.high_dtype],
        blob_path,
        row_chunk,
        "cuda",
        device,
        importance_rows=binding.rows,
    )


def convert(args: argparse.Namespace) -> None:
    if args.stop_after < 0:
        raise ValueError("--stop-after must be nonnegative")
    source_path = Path(args.input_bf16_gguf).resolve()
    recipe_path = Path(args.recipe_gguf).resolve()
    baseline_path = Path(args.baseline_mfq).resolve()
    imatrix_path = Path(args.imatrix).resolve()
    output = Path(args.output).resolve()
    target_bytes = (
        int(args.target_bytes)
        if args.target_bytes
        else recipe_path.stat().st_size
    )
    if output.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    GGUFReader, dequantize = _load_gguf()
    source_reader = GGUFReader(str(source_path), "r")
    recipe_reader = GGUFReader(str(recipe_path), "r")
    source_tensors = {
        str(tensor.name): tensor for tensor in source_reader.tensors
    }
    plan = _build_plan(
        source_reader, recipe_reader, exclude_mtp=True
    )
    imatrix = load_importance_matrix(imatrix_path)

    with open_mmap(baseline_path) as store:
        candidates = _build_candidates(plan, store, imatrix)
        header_args = {
            "source": source_path,
            "recipe": recipe_path,
            "imatrix": imatrix_path,
            "baseline_model": baseline_path,
            "target_bytes": target_bytes,
        }
        selected, header, estimated_total, padding_nbytes = _select_candidates(
            store,
            store.header,
            candidates,
            target_bytes,
            header_args,
        )
        contract = {
            "format": "mfq.whole-tensor-upgrade-control.run.v1",
            "input_bf16_gguf": _file_identity(source_path),
            "recipe_gguf": _file_identity(recipe_path),
            "baseline_mfq": _file_identity(baseline_path),
            "imatrix": _file_identity(imatrix_path),
            "output": str(output),
            "candidate_tensors": len(candidates),
            "selected_tensors": len(selected),
            "target_bytes": target_bytes,
            "estimated_output_bytes": estimated_total,
            "estimated_gap_bytes": target_bytes - estimated_total,
            "padding_bytes": padding_nbytes,
            "upgrades": [
                {
                    "name": candidate.item.name,
                    "from": candidate.low_dtype,
                    "to": candidate.high_dtype,
                    "cost_bytes": candidate.cost,
                }
                for candidate in selected
            ],
            "device": args.device,
            "row_chunk": args.row_chunk,
        }
        print(json.dumps(contract, ensure_ascii=False), flush=True)
        if args.dry_run:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("whole-tensor control quantization requires CUDA")

        temporary_root = (
            output.parent / f".{output.name}.tensor-control"
        )
        blob_root = temporary_root / "blobs"
        if temporary_root.exists() and not args.resume:
            raise FileExistsError(
                f"control temporary directory exists; use --resume: "
                f"{temporary_root}"
            )
        blob_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(temporary_root / "contract.json", contract)
        selected_by_name = {
            candidate.item.name: candidate
            for candidate in selected
        }
        generated: dict[str, BlobRecord] = {}
        started = time.time()
        for index, candidate in enumerate(selected, start=1):
            blob_path = blob_root / f"{index:03d}.blob"
            t0 = time.time()
            if blob_path.is_file():
                nbytes = blob_path.stat().st_size
                if nbytes != candidate.new_nbytes:
                    raise ValueError(
                        f"resume blob size mismatch for "
                        f"{candidate.item.name}: "
                        f"{nbytes} != {candidate.new_nbytes}"
                    )
            else:
                source = GgufRowSource(
                    source_tensors[candidate.item.source_name],
                    candidate.item,
                    dequantize,
                )
                nbytes = _quantize_candidate(
                    source,
                    candidate,
                    blob_path,
                    imatrix=imatrix,
                    row_chunk=args.row_chunk,
                    device=args.device,
                )
                if nbytes != candidate.new_nbytes:
                    raise ValueError(
                        f"control blob estimate mismatch for "
                        f"{candidate.item.name}: "
                        f"{nbytes} != {candidate.new_nbytes}"
                    )
            generated[candidate.item.name] = BlobRecord(
                candidate.item.name,
                candidate.high_dtype,
                nbytes,
                blob_path,
            )
            state = {
                "status": "quantizing",
                "completed": index,
                "tensors": len(selected),
                "last_tensor": candidate.item.name,
                "last_dtype": candidate.high_dtype,
                "last_seconds": time.time() - t0,
                "elapsed_seconds": time.time() - started,
            }
            _atomic_json(temporary_root / "state.json", state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
            if args.stop_after and index >= args.stop_after:
                probe_state = {
                    **state,
                    "status": "probe_complete",
                    "stop_after": args.stop_after,
                }
                _atomic_json(
                    temporary_root / "state.json", probe_state
                )
                print(
                    json.dumps(probe_state, ensure_ascii=False),
                    flush=True,
                )
                return

        records: list[FileSpanRecord | BlobRecord] = []
        for record in store.records.values():
            replacement = generated.get(record.name)
            records.append(
                replacement
                if replacement is not None
                else _record_span(store, record)
            )
        if padding_nbytes:
            padding_path = temporary_root / "padding.bin"
            with padding_path.open("wb") as handle:
                handle.truncate(padding_nbytes)
            records.append(
                BlobRecord(
                    CONTROL_PADDING_ASSET_NAME,
                    ASSET_DTYPE,
                    padding_nbytes,
                    padding_path,
                )
            )
        outputs = write_blob_record_shards(
            output,
            header,
            records,
            overwrite=args.overwrite,
        )
        actual_size = sum(path.stat().st_size for path in outputs)
        if actual_size != estimated_total:
            raise ValueError(
                f"control output size differs from the plan: "
                f"{actual_size} != {estimated_total}"
            )
        final_state = {
            "status": "complete",
            "output": str(outputs[0]),
            "outputs": [str(path) for path in outputs],
            "output_bytes": actual_size,
            "target_bytes": target_bytes,
            "gap_bytes": target_bytes - actual_size,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(temporary_root / "state.json", final_state)
        print(json.dumps(final_state, ensure_ascii=False), flush=True)
    if not args.keep_temp:
        shutil.rmtree(temporary_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bf16-gguf", required=True)
    parser.add_argument("--recipe-gguf", required=True)
    parser.add_argument("--baseline-mfq", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=0,
        help="defaults to the recipe GGUF file size",
    )
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="quantize this many production blobs, persist them, and exit",
    )
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
