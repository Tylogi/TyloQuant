"""Recipe accounting and EW allocation for DeepSeek-V4-Flash."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mfq.formats.io import (
    _NINT_HDR,
    _NINT_MOE_HDR,
    _NINT_MOE_POOL_V2_HDR,
    _NINT_MOE_ROTATION_HDR,
)
from mfq.formats.nepq import (
    NEPQ0_S,
    _HEADER as _NEPQ_HEADER,
)
from mfq.formats.nint import NintSpec
from mfq.formats.nvq import NVQ2_E8, _HEADER as _NVQ_HEADER
from mfq.formats.mx import mx_header_bytes


_ROUTED_SUFFIXES = (
    ".ffn_gate_exps.weight",
    ".ffn_up_exps.weight",
    ".ffn_down_exps.weight",
)
_RECIPE_NINT = {
    "Q4_K": NintSpec(4, 24, 6),
    "Q5_K": NintSpec(5, 28, 7),
    "Q6_K": NintSpec(6, 24, 7),
    "Q8_0": NintSpec(8, 48, 7),
}
_ROUTED_NINT = {
    f"NINT{spec.bits}": spec
    for spec in _RECIPE_NINT.values()
}


def _nint_blob_nbytes(rows: int, columns: int, spec: NintSpec) -> int:
    groups = (columns + spec.groupsize - 1) // spec.groupsize
    header = _NINT_HDR.size + 4 + 2 * 8 + 8
    sub = (rows * groups * spec.sub_bits + 7) // 8
    q = (rows * groups * spec.groupsize * spec.bits + 7) // 8
    return int(header + rows * 4 + 2 * sub + q)


@dataclass(frozen=True)
class RecipeTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class V4FEwAllocation:
    target_bytes: int
    nonexpert_bytes: int
    routed_bytes: int
    estimated_blob_bytes: int
    gate_up_high: dict[int, tuple[int, ...]]
    down_high: dict[int, tuple[int, ...]]
    gate_up_energy_fraction: float
    down_energy_fraction: float

    @property
    def high_count(self) -> int:
        return sum(map(len, self.gate_up_high.values())) + sum(
            map(len, self.down_high.values())
        )


@dataclass(frozen=True)
class V4FTieredAllocation:
    """NVQ2J baseline with REAP-ranked NINT4 expert upgrades."""

    target_bytes: int
    nonexpert_bytes: int
    routed_bytes: int
    estimated_blob_bytes: int
    gate_up_nint4: dict[int, tuple[int, ...]]
    down_nint4: dict[int, tuple[int, ...]]
    gate_up_energy_fraction: float
    down_energy_fraction: float

    @property
    def nint4_count(self) -> int:
        return sum(map(len, self.gate_up_nint4.values())) + sum(
            map(len, self.down_nint4.values())
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_gguf_header_recipe(paths: list[str | Path]) -> tuple[list[RecipeTensor], dict]:
    gguf_py = (
        Path(__file__).resolve().parents[2]
        / "references"
        / "llamacpp"
        / "gguf-py"
    )
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))
    from gguf import GGMLQuantizationType, GGUFReader  # type: ignore

    class HeaderReader(GGUFReader):
        def _build_tensors(self, start_offs, fields) -> None:
            self.tensor_fields = fields

    tensors: dict[str, RecipeTensor] = {}
    identities = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        reader = HeaderReader(str(path), "r")
        identities.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        for field in reader.tensor_fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, _offset = field.parts
            name = bytes(name_data).decode("utf-8")
            item = RecipeTensor(
                name=name,
                dtype=GGMLQuantizationType(int(raw_dtype[0])).name,
                shape=tuple(reversed(tuple(int(value) for value in dims))),
            )
            if name in tensors and tensors[name] != item:
                raise ValueError(f"conflicting GGUF split tensor metadata: {name}")
            tensors[name] = item
    return (
        [tensors[name] for name in sorted(tensors)],
        {
            "format": "mfq.v4f-gguf-recipe.v1",
            "headers": identities,
            "tensor_count": len(tensors),
        },
    )


def save_recipe(
    path: str | Path,
    tensors: list[RecipeTensor],
    metadata: dict,
) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = dict(metadata)
    document["tensors"] = [
        {"name": item.name, "dtype": item.dtype, "shape": list(item.shape)}
        for item in tensors
    ]
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(output)


def load_recipe(path: str | Path) -> tuple[list[RecipeTensor], dict]:
    resolved = Path(path).resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if document.get("format") != "mfq.v4f-gguf-recipe.v1":
        raise ValueError(f"unsupported V4F recipe document: {resolved}")
    tensors = [
        RecipeTensor(
            name=str(raw["name"]),
            dtype=str(raw["dtype"]),
            shape=tuple(int(value) for value in raw["shape"]),
        )
        for raw in document["tensors"]
    ]
    if len({item.name for item in tensors}) != len(tensors):
        raise ValueError("V4F recipe contains duplicate tensor names")
    return tensors, document


def recipe_source_map(
    checkpoint: V4FCheckpoint,
    recipe: list[RecipeTensor],
) -> dict[str, str]:
    from mfq.quantize.v4f_source import v4f_source_to_gguf_name

    source_by_target: dict[str, str] = {}
    for source in checkpoint.weight_map:
        if source.endswith(".scale") or ".ffn.experts." in source:
            continue
        target = v4f_source_to_gguf_name(source)
        if target is None:
            continue
        if target in source_by_target:
            raise ValueError(f"multiple V4F sources map to {target}")
        source_by_target[target] = source
    expected = {
        item.name
        for item in recipe
        if not item.name.endswith(_ROUTED_SUFFIXES)
    }
    missing = sorted(expected - set(source_by_target))
    extra = sorted(set(source_by_target) - expected)
    if missing or extra:
        raise ValueError(
            "V4F recipe/source mapping mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    for item in recipe:
        if item.name not in expected:
            continue
        source_shape = checkpoint.info(source_by_target[item.name]).shape
        if source_shape != item.shape:
            raise ValueError(
                f"V4F shape mismatch for {item.name}: "
                f"source={source_shape}, recipe={item.shape}"
            )
    return source_by_target


def recipe_target_dtype(dtype: str) -> str:
    if dtype in _RECIPE_NINT:
        return f"NINT{_RECIPE_NINT[dtype].bits}"
    if dtype == "F32":
        return "F32"
    if dtype == "BF16":
        return "F16"
    if dtype == "I32":
        return "I32"
    raise ValueError(f"unsupported nonexpert V4F recipe dtype: {dtype}")


def nonexpert_blob_bytes(recipe: list[RecipeTensor]) -> int:
    total = 0
    for item in recipe:
        if item.name.endswith(_ROUTED_SUFFIXES):
            continue
        target = recipe_target_dtype(item.dtype)
        if target.startswith("NINT"):
            spec = _RECIPE_NINT[item.dtype]
            if len(item.shape) != 2:
                raise ValueError(f"NINT recipe tensor is not a matrix: {item}")
            total += _nint_blob_nbytes(item.shape[0], item.shape[1], spec)
        else:
            item_size = {"F16": 2, "F32": 4, "I32": 4}[target]
            total += 4 + 8 * len(item.shape) + int(np.prod(item.shape)) * item_size
    return int(total)


def routed_blob_bytes(
    projection: str,
    low_count: int,
    high_count: int,
) -> int:
    return routed_family_blob_bytes(
        projection,
        {"NEPQ0-S": low_count, "NVQ2J": high_count},
    )


def routed_family_blob_bytes(
    projection: str,
    family_counts: dict[str, int],
) -> int:
    """Return the exact NINTM blob bytes for one V4F routed projection."""

    if projection not in {"gate_up", "down"}:
        raise ValueError(f"unsupported V4F routed projection: {projection}")
    supported = {"NEPQ0-S", "NVQ2J", "MXFP4", *_ROUTED_NINT}
    unknown = set(family_counts) - supported
    if unknown:
        raise ValueError(f"unsupported V4F routed families: {sorted(unknown)}")
    normalized = {
        family: int(family_counts.get(family, 0))
        for family in sorted(supported)
    }
    if any(count < 0 for count in normalized.values()):
        raise ValueError("V4F routed family counts cannot be negative")
    if sum(normalized.values()) != 256:
        raise ValueError("V4F routed family counts must sum to 256")

    total = _NINT_MOE_HDR.size
    total += sum(
        routed_family_pool_bytes(projection, family, count)
        for family, count in normalized.items()
    )
    return int(total)


def routed_family_pool_bytes(
    projection: str,
    family: str,
    count: int,
) -> int:
    """Return one non-empty NINTM cohort's exact serialized bytes."""

    if projection not in {"gate_up", "down"}:
        raise ValueError(f"unsupported V4F routed projection: {projection}")
    supported = {"NEPQ0-S", "NVQ2J", "MXFP4", *_ROUTED_NINT}
    if family not in supported:
        raise ValueError(f"unsupported V4F routed family: {family}")
    count = int(count)
    if count < 0 or count > 256:
        raise ValueError("V4F routed family count must be in [0, 256]")
    if count == 0:
        return 0

    rows_per_expert = 4096
    columns = 4096 if projection == "gate_up" else 2048
    runtime_nbytes = 0
    if family == "NEPQ0-S":
        runtime_nbytes = _NINT_MOE_ROTATION_HDR.size + columns
        payload = _NEPQ_HEADER.size + NEPQ0_S.payload_nbytes(
            count,
            rows_per_expert,
            columns,
            bank_count=256,
        )
    elif family == "NVQ2J":
        rows = count * rows_per_expert
        table_nbytes = 64 + 4 * 256 * NVQ2_E8.vector_size
        payload = (
            _NVQ_HEADER.size
            + 2 * 8
            + 4
            + table_nbytes
            + NVQ2_E8.payload_nbytes(rows, columns)
        )
    elif family == "MXFP4":
        rows = count * rows_per_expert
        payload = len(
            mx_header_bytes(
                "MXFP4",
                (rows, columns),
                (rows, columns // 2),
                (rows, columns // 32),
            )
        ) + rows * (columns // 2 + columns // 32)
    else:
        payload = _nint_blob_nbytes(
            count * rows_per_expert,
            columns,
            _ROUTED_NINT[family],
        )
    return int(
        _NINT_MOE_POOL_V2_HDR.size
        + count * 4
        + len(family)
        + runtime_nbytes
        + payload
    )


def _read_reap(path: str | Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "layer": int(raw["layer"]),
                    "expert": int(raw["expert_id"]),
                    "gate_up_energy": float(raw["gate_up_energy"]),
                    "down_energy": float(raw["down_energy"]),
                }
            )
    owners = {(int(row["layer"]), int(row["expert"])) for row in rows}
    if len(rows) != 43 * 256 or len(owners) != len(rows):
        raise ValueError("V4F REAP table must contain exactly 43x256 experts")
    return rows


def allocate_v4f_ew(
    recipe: list[RecipeTensor],
    reap_csv: str | Path,
    *,
    target_bytes: int = 40_000_000_000,
    container_reserve_bytes: int = 4_000_000,
) -> V4FEwAllocation:
    nonexpert = nonexpert_blob_bytes(recipe)
    low_gate = routed_blob_bytes("gate_up", 256, 0)
    low_down = routed_blob_bytes("down", 256, 0)
    low_routed = 43 * (low_gate + low_down)
    available_upgrade = (
        target_bytes - container_reserve_bytes - nonexpert - low_routed
    )
    if available_upgrade < 0:
        raise ValueError(
            f"40G V4F baseline exceeds the budget by {-available_upgrade} bytes"
        )
    reap = _read_reap(reap_csv)

    def ranked(metric: str) -> list[dict[str, float | int]]:
        return sorted(
            (row for row in reap if int(row["layer"]) >= 3),
            key=lambda row: float(row[metric]),
            reverse=True,
        )

    gate_ranked = ranked("gate_up_energy")
    down_ranked = ranked("down_energy")
    gate_budget = available_upgrade * 2 // 3
    down_budget = available_upgrade - gate_budget

    def select(
        projection: str,
        candidates: list[dict[str, float | int]],
        budget: int,
    ) -> tuple[list[dict[str, float | int]], int]:
        chosen: list[dict[str, float | int]] = []
        layer_counts: dict[int, int] = {}
        spent = 0
        for row in candidates:
            layer = int(row["layer"])
            high_count = layer_counts.get(layer, 0)
            before = routed_blob_bytes(
                projection, 256 - high_count, high_count
            )
            after = routed_blob_bytes(
                projection, 255 - high_count, high_count + 1
            )
            delta = after - before
            if spent + delta > budget:
                continue
            chosen.append(row)
            layer_counts[layer] = high_count + 1
            spent += delta
        return chosen, spent

    gate, gate_spent = select("gate_up", gate_ranked, gate_budget)
    down, down_spent = select("down", down_ranked, down_budget)
    def by_layer(rows: list[dict[str, float | int]]) -> dict[int, tuple[int, ...]]:
        result: dict[int, list[int]] = {}
        for row in rows:
            result.setdefault(int(row["layer"]), []).append(int(row["expert"]))
        return {
            layer: tuple(sorted(experts))
            for layer, experts in sorted(result.items())
        }

    gate_total = sum(float(row["gate_up_energy"]) for row in reap)
    down_total = sum(float(row["down_energy"]) for row in reap)
    gate_energy = sum(float(row["gate_up_energy"]) for row in gate)
    down_energy = sum(float(row["down_energy"]) for row in down)
    routed = (
        43 * (low_gate + low_down) + gate_spent + down_spent
    )
    return V4FEwAllocation(
        target_bytes=int(target_bytes),
        nonexpert_bytes=nonexpert,
        routed_bytes=routed,
        estimated_blob_bytes=nonexpert + routed,
        gate_up_high=by_layer(gate),
        down_high=by_layer(down),
        gate_up_energy_fraction=gate_energy / gate_total,
        down_energy_fraction=down_energy / down_total,
    )


def allocate_v4f_ew_nvq2j_nint4(
    recipe: list[RecipeTensor],
    reap_csv: str | Path,
    *,
    target_bytes: int = 88_000_000_000,
    container_reserve_bytes: int = 4_000_000,
) -> V4FTieredAllocation:
    """Allocate NINT4 upgrades over an all-NVQ2J routed baseline."""

    nonexpert = nonexpert_blob_bytes(recipe)
    base_per_layer = (
        routed_family_blob_bytes("gate_up", {"NVQ2J": 256})
        + routed_family_blob_bytes("down", {"NVQ2J": 256})
    )
    base_routed = 43 * base_per_layer
    available_upgrade = (
        target_bytes - container_reserve_bytes - nonexpert - base_routed
    )
    if available_upgrade < 0:
        raise ValueError(
            "NVQ2J V4F baseline exceeds the target by "
            f"{-available_upgrade} bytes"
        )

    reap = _read_reap(reap_csv)
    selected: dict[str, dict[int, list[int]]] = {
        "gate_up": {},
        "down": {},
    }

    def steady_delta(projection: str) -> int:
        before = routed_family_blob_bytes(
            projection, {"NVQ2J": 255, "NINT4": 1}
        )
        after = routed_family_blob_bytes(
            projection, {"NVQ2J": 254, "NINT4": 2}
        )
        delta = after - before
        if delta <= 0:
            raise RuntimeError(
                f"NINT4 upgrade has nonpositive byte cost for {projection}"
            )
        return delta

    deltas = {
        projection: steady_delta(projection)
        for projection in ("gate_up", "down")
    }
    candidates: list[tuple[float, float, str, int, int]] = []
    for row in reap:
        layer = int(row["layer"])
        expert = int(row["expert"])
        for projection, metric in (
            ("gate_up", "gate_up_energy"),
            ("down", "down_energy"),
        ):
            energy = float(row[metric])
            candidates.append(
                (
                    energy / deltas[projection],
                    energy,
                    projection,
                    layer,
                    expert,
                )
            )
    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2],
            item[3],
            item[4],
        )
    )

    spent = 0
    for _density, _energy, projection, layer, expert in candidates:
        current = len(selected[projection].get(layer, ()))
        before = routed_family_blob_bytes(
            projection,
            {"NVQ2J": 256 - current, "NINT4": current},
        )
        after = routed_family_blob_bytes(
            projection,
            {"NVQ2J": 255 - current, "NINT4": current + 1},
        )
        delta = after - before
        if spent + delta > available_upgrade:
            continue
        selected[projection].setdefault(layer, []).append(expert)
        spent += delta

    def freeze(
        values: dict[int, list[int]],
    ) -> dict[int, tuple[int, ...]]:
        return {
            layer: tuple(sorted(experts))
            for layer, experts in sorted(values.items())
        }

    gate_up = freeze(selected["gate_up"])
    down = freeze(selected["down"])
    routed = 0
    for layer in range(43):
        gate_count = len(gate_up.get(layer, ()))
        down_count = len(down.get(layer, ()))
        routed += routed_family_blob_bytes(
            "gate_up",
            {"NVQ2J": 256 - gate_count, "NINT4": gate_count},
        )
        routed += routed_family_blob_bytes(
            "down",
            {"NVQ2J": 256 - down_count, "NINT4": down_count},
        )

    energy = {
        (int(row["layer"]), int(row["expert"])): row
        for row in reap
    }
    gate_total = sum(float(row["gate_up_energy"]) for row in reap)
    down_total = sum(float(row["down_energy"]) for row in reap)
    gate_selected = sum(
        float(energy[(layer, expert)]["gate_up_energy"])
        for layer, experts in gate_up.items()
        for expert in experts
    )
    down_selected = sum(
        float(energy[(layer, expert)]["down_energy"])
        for layer, experts in down.items()
        for expert in experts
    )
    return V4FTieredAllocation(
        target_bytes=int(target_bytes),
        nonexpert_bytes=nonexpert,
        routed_bytes=int(routed),
        estimated_blob_bytes=int(nonexpert + routed),
        gate_up_nint4=gate_up,
        down_nint4=down,
        gate_up_energy_fraction=gate_selected / gate_total,
        down_energy_fraction=down_selected / down_total,
    )


__all__ = [
    "RecipeTensor",
    "V4FEwAllocation",
    "V4FTieredAllocation",
    "allocate_v4f_ew",
    "allocate_v4f_ew_nvq2j_nint4",
    "load_recipe",
    "nonexpert_blob_bytes",
    "read_gguf_header_recipe",
    "recipe_source_map",
    "recipe_target_dtype",
    "routed_blob_bytes",
    "routed_family_blob_bytes",
    "routed_family_pool_bytes",
    "save_recipe",
]
