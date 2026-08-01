"""Plan a bounded NINT4 upgrade over an existing V4F EW allocation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mfq.formats.io import (
    _NINT_HDR,
    _NINT_MOE_HDR,
    _NINT_MOE_MAGIC_V2,
    _NINT_MOE_POOL_V2_HDR,
    open_mmap,
)
from mfq.formats.nint import NintSpec
from mfq.quantize.expert_sensitivity import (
    ExpertSensitivityMap,
    load_expert_sensitivity_map,
)
from mfq.quantize.v4f_plan import (
    routed_family_blob_bytes,
    routed_family_pool_bytes,
)


NINT4 = NintSpec(4, 24, 6)
NINT8 = NintSpec(8, 48, 7)
NINT_SPECS = {
    "NINT4": NINT4,
    "NINT8": NINT8,
}
PROJECTIONS = ("gate_up", "down")
_ROUTED_NAME = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<projection>gate_up|down)_exps\.weight$"
)


def _nint_blob_nbytes(rows: int, columns: int, spec: NintSpec) -> int:
    groups = (columns + spec.groupsize - 1) // spec.groupsize
    header = _NINT_HDR.size + 4 + 2 * 8 + 8
    sub = (rows * groups * spec.sub_bits + 7) // 8
    q = (rows * groups * spec.groupsize * spec.bits + 7) // 8
    return int(header + rows * 4 + 2 * sub + q)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(values: dict[int, list[int]]) -> dict[int, tuple[int, ...]]:
    return {
        int(layer): tuple(sorted(int(expert) for expert in experts))
        for layer, experts in sorted(values.items())
        if experts
    }


def _read_map(raw: dict, key: str) -> dict[int, tuple[int, ...]]:
    return {
        int(layer): tuple(sorted(int(expert) for expert in experts))
        for layer, experts in raw[key].items()
    }


@dataclass(frozen=True)
class V4FNint4Upgrade:
    target_bytes: int
    container_reserve_bytes: int
    nonexpert_bytes: int
    routed_bytes: int
    estimated_blob_bytes: int
    base_gate_up_high: dict[int, tuple[int, ...]]
    base_down_high: dict[int, tuple[int, ...]]
    gate_up_nint4: dict[int, tuple[int, ...]]
    down_nint4: dict[int, tuple[int, ...]]
    gate_up_energy_fraction: float
    down_energy_fraction: float
    nint4_payload_bytes: int

    @property
    def nint4_count(self) -> int:
        return sum(
            len(experts)
            for mapping in (self.gate_up_nint4, self.down_nint4)
            for experts in mapping.values()
        )

    @property
    def upgrade_family(self) -> str:
        return "NINT4"

    def base_high(self, projection: str) -> dict[int, tuple[int, ...]]:
        if projection == "gate_up":
            return self.base_gate_up_high
        if projection == "down":
            return self.base_down_high
        raise ValueError(f"unsupported projection: {projection}")

    def nint4(self, projection: str) -> dict[int, tuple[int, ...]]:
        if projection == "gate_up":
            return self.gate_up_nint4
        if projection == "down":
            return self.down_nint4
        raise ValueError(f"unsupported projection: {projection}")

    def selected(self, projection: str) -> dict[int, tuple[int, ...]]:
        return self.nint4(projection)

    def families(self, projection: str, layer: int) -> tuple[str, ...]:
        high = set(self.base_high(projection).get(int(layer), ()))
        nint4 = set(self.nint4(projection).get(int(layer), ()))
        return tuple(
            "NINT4"
            if expert in nint4
            else "NVQ2J"
            if expert in high
            else "NEPQ0-S"
            for expert in range(256)
        )


@dataclass(frozen=True)
class V4FMarkedNint8Upgrade:
    """Hard-protect marked experts while preserving every other assignment."""

    target_bytes: int
    container_reserve_bytes: int
    nonexpert_bytes: int
    routed_bytes: int
    estimated_blob_bytes: int
    base_gate_up_high: dict[int, tuple[int, ...]]
    base_down_high: dict[int, tuple[int, ...]]
    gate_up_nint8: dict[int, tuple[int, ...]]
    down_nint8: dict[int, tuple[int, ...]]
    gate_up_energy_fraction: float
    down_energy_fraction: float
    nint8_payload_bytes: int
    mark_counts: dict[str, int]

    @property
    def upgrade_family(self) -> str:
        return "NINT8"

    @property
    def nint8_count(self) -> int:
        return sum(
            len(experts)
            for mapping in (self.gate_up_nint8, self.down_nint8)
            for experts in mapping.values()
        )

    def base_high(self, projection: str) -> dict[int, tuple[int, ...]]:
        if projection == "gate_up":
            return self.base_gate_up_high
        if projection == "down":
            return self.base_down_high
        raise ValueError(f"unsupported projection: {projection}")

    def selected(self, projection: str) -> dict[int, tuple[int, ...]]:
        if projection == "gate_up":
            return self.gate_up_nint8
        if projection == "down":
            return self.down_nint8
        raise ValueError(f"unsupported projection: {projection}")

    def families(self, projection: str, layer: int) -> tuple[str, ...]:
        high = set(self.base_high(projection).get(int(layer), ()))
        nint8 = set(self.selected(projection).get(int(layer), ()))
        return tuple(
            "NINT8"
            if expert in nint8
            else "NVQ2J"
            if expert in high
            else "NEPQ0-S"
            for expert in range(256)
        )


@dataclass(frozen=True)
class V4FSensitivityReallocation:
    """Protect marked experts while preserving a mixed-precision V4F base."""

    target_bytes: int
    container_reserve_bytes: int
    base_file_bytes: int
    nonexpert_bytes: int
    baseline_routed_bytes: int
    protected_routed_bytes: int
    routed_bytes: int
    estimated_blob_bytes: int
    base_gate_up_nepq0s: dict[int, tuple[int, ...]]
    base_down_nepq0s: dict[int, tuple[int, ...]]
    base_gate_up_nint4: dict[int, tuple[int, ...]]
    base_down_nint4: dict[int, tuple[int, ...]]
    gate_up_nint4: dict[int, tuple[int, ...]]
    down_nint4: dict[int, tuple[int, ...]]
    gate_up_nint8: dict[int, tuple[int, ...]]
    down_nint8: dict[int, tuple[int, ...]]
    gate_up_demoted: dict[int, tuple[int, ...]]
    down_demoted: dict[int, tuple[int, ...]]
    baseline_gate_up_energy_fraction: float
    baseline_down_energy_fraction: float
    final_gate_up_energy_fraction: float
    final_down_energy_fraction: float
    protected_gate_up_energy_fraction: float
    protected_down_energy_fraction: float
    mark_counts: dict[str, int]
    base_allocation_sha256: str
    source_index_sha256: str

    @property
    def upgrade_family(self) -> str:
        return "NINT8"

    @property
    def nint8_count(self) -> int:
        return sum(
            len(experts)
            for mapping in (self.gate_up_nint8, self.down_nint8)
            for experts in mapping.values()
        )

    @property
    def demoted_count(self) -> int:
        return sum(
            len(experts)
            for mapping in (self.gate_up_demoted, self.down_demoted)
            for experts in mapping.values()
        )

    def selected(self, projection: str) -> dict[int, tuple[int, ...]]:
        if projection == "gate_up":
            return self.gate_up_nint8
        if projection == "down":
            return self.down_nint8
        raise ValueError(f"unsupported projection: {projection}")

    def base_families(self, projection: str, layer: int) -> tuple[str, ...]:
        if projection == "gate_up":
            nepq0s = set(self.base_gate_up_nepq0s.get(int(layer), ()))
            nint4 = set(self.base_gate_up_nint4.get(int(layer), ()))
        elif projection == "down":
            nepq0s = set(self.base_down_nepq0s.get(int(layer), ()))
            nint4 = set(self.base_down_nint4.get(int(layer), ()))
        else:
            raise ValueError(f"unsupported projection: {projection}")
        return tuple(
            "NINT4"
            if expert in nint4
            else "NEPQ0-S"
            if expert in nepq0s
            else "NVQ2J"
            for expert in range(256)
        )

    def families(self, projection: str, layer: int) -> tuple[str, ...]:
        if projection == "gate_up":
            nepq0s = set(self.base_gate_up_nepq0s.get(int(layer), ()))
            nint4 = set(self.gate_up_nint4.get(int(layer), ()))
            nint8 = set(self.gate_up_nint8.get(int(layer), ()))
        elif projection == "down":
            nepq0s = set(self.base_down_nepq0s.get(int(layer), ()))
            nint4 = set(self.down_nint4.get(int(layer), ()))
            nint8 = set(self.down_nint8.get(int(layer), ()))
        else:
            raise ValueError(f"unsupported projection: {projection}")
        return tuple(
            "NINT8"
            if expert in nint8
            else "NINT4"
            if expert in nint4
            else "NEPQ0-S"
            if expert in nepq0s
            else "NVQ2J"
            for expert in range(256)
        )


V4FUpgradePlan = (
    V4FNint4Upgrade
    | V4FMarkedNint8Upgrade
    | V4FSensitivityReallocation
)


def _read_reap(path: str | Path) -> dict[tuple[int, int], dict[str, float]]:
    result: dict[tuple[int, int], dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (int(raw["layer"]), int(raw["expert_id"]))
            if key in result:
                raise ValueError(f"duplicate REAP expert: {key}")
            result[key] = {
                "gate_up": float(raw["gate_up_energy"]),
                "down": float(raw["down_energy"]),
            }
    expected = {(layer, expert) for layer in range(43) for expert in range(256)}
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(
            f"REAP table coverage mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    return result


def _allocation_family(profile: str) -> str:
    match = re.match(r"^(NINT\d+)-", profile)
    return match.group(1) if match else profile


def _nintm_profiles(
    blob: bytes | memoryview,
) -> tuple[
    tuple[int, int, int],
    tuple[str, ...],
    tuple[tuple[str, int], ...],
]:
    view = memoryview(blob)
    if len(view) < _NINT_MOE_HDR.size:
        raise ValueError("truncated NINTM header")
    magic, n_experts, out_per_expert, neuron_len, pool_count = (
        _NINT_MOE_HDR.unpack_from(view)
    )
    if magic != _NINT_MOE_MAGIC_V2:
        raise ValueError(f"unsupported NINTM magic: {magic!r}")
    profiles = [""] * int(n_experts)
    pools: list[tuple[str, int]] = []
    offset = _NINT_MOE_HDR.size
    for _ in range(int(pool_count)):
        if offset + _NINT_MOE_POOL_V2_HDR.size > len(view):
            raise ValueError("truncated NINTM pool header")
        expert_count, dtype_nbytes, payload_nbytes, runtime_nbytes = (
            _NINT_MOE_POOL_V2_HDR.unpack_from(view, offset)
        )
        offset += _NINT_MOE_POOL_V2_HDR.size
        ids_nbytes = int(expert_count) * 4
        metadata_end = offset + ids_nbytes + int(dtype_nbytes)
        pool_end = metadata_end + int(runtime_nbytes) + int(payload_nbytes)
        if pool_end > len(view):
            raise ValueError("truncated NINTM pool payload")
        expert_ids = struct.unpack_from(
            f"<{int(expert_count)}i",
            view,
            offset,
        )
        offset += ids_nbytes
        dtype = bytes(view[offset : offset + int(dtype_nbytes)]).decode("ascii")
        family = _allocation_family(dtype)
        pools.append((family, int(expert_count)))
        offset = pool_end
        for expert in expert_ids:
            if expert < 0 or expert >= n_experts:
                raise ValueError(f"NINTM pool contains invalid expert {expert}")
            if profiles[expert]:
                raise ValueError(
                    f"NINTM expert {expert} belongs to multiple pools"
                )
            profiles[expert] = family
    if offset != len(view):
        raise ValueError("NINTM blob contains trailing bytes")
    if any(not family for family in profiles):
        raise ValueError("NINTM pools do not cover every expert")
    return (
        (int(n_experts), int(out_per_expert), int(neuron_len)),
        tuple(profiles),
        tuple(pools),
    )


def _read_v4f_mfq_profiles(
    path: str | Path,
) -> tuple[
    dict[tuple[str, int], tuple[str, ...]],
    dict[str, object],
]:
    model = Path(path).resolve()
    profiles: dict[tuple[str, int], tuple[str, ...]] = {}
    routed_bytes = 0
    canonical_routed_bytes = 0
    with open_mmap(model) as store:
        extra = dict(store.header.extra)
        for name, record in store.records.items():
            match = _ROUTED_NAME.match(name)
            if match is None:
                continue
            if record.dtype != "NINTM":
                raise TypeError(f"routed tensor is not NINTM: {name}")
            projection = match.group("projection")
            layer = int(match.group("layer"))
            blob = store.blob_view(record)
            try:
                shape, families, pools = _nintm_profiles(blob)
            finally:
                blob.release()
            expected_shape = (
                256,
                4096,
                4096 if projection == "gate_up" else 2048,
            )
            if shape != expected_shape:
                raise ValueError(
                    f"unexpected routed shape for {name}: {shape}"
                )
            parsed_nbytes = _NINT_MOE_HDR.size + sum(
                routed_family_pool_bytes(
                    projection,
                    family,
                    count,
                )
                for family, count in pools
            )
            if parsed_nbytes != record.nbytes:
                raise ValueError(
                    f"routed byte accounting mismatch for {name}: "
                    f"{parsed_nbytes} != {record.nbytes}"
                )
            family_counts = dict(Counter(families))
            canonical_routed_bytes += routed_family_blob_bytes(
                projection,
                family_counts,
            )
            profiles[(projection, layer)] = families
            routed_bytes += record.nbytes
        expected_keys = {
            (projection, layer)
            for projection in PROJECTIONS
            for layer in range(43)
        }
        if set(profiles) != expected_keys:
            missing = sorted(expected_keys - set(profiles))
            extra_keys = sorted(set(profiles) - expected_keys)
            raise ValueError(
                "V4F routed tensor coverage mismatch: "
                f"missing={missing[:8]}, extra={extra_keys[:8]}"
            )
    estimated_blob_bytes = int(extra["estimated_blob_bytes"])
    nonexpert_bytes = estimated_blob_bytes - routed_bytes
    if nonexpert_bytes < 0:
        raise ValueError("MFQ routed bytes exceed estimated blob bytes")
    metadata: dict[str, object] = {
        "file_bytes": model.stat().st_size,
        "estimated_blob_bytes": estimated_blob_bytes,
        "routed_bytes": routed_bytes,
        "canonical_routed_bytes": canonical_routed_bytes,
        "duplicate_pool_overhead_bytes": (
            routed_bytes - canonical_routed_bytes
        ),
        "nonexpert_bytes": nonexpert_bytes,
        "target_bytes": int(extra["target_bytes"]),
        "allocation_sha256": str(extra["allocation_sha256"]),
        "source_index_sha256": str(extra["source_index_sha256"]),
    }
    return profiles, metadata


def _profiles_by_layer(
    profiles: dict[tuple[str, int], list[str] | tuple[str, ...]],
    projection: str,
    family: str,
) -> dict[int, tuple[int, ...]]:
    return _freeze(
        {
            layer: [
                expert
                for expert, value in enumerate(profiles[(projection, layer)])
                if value == family
            ]
            for layer in range(43)
        }
    )


def _profile_routed_bytes(
    profiles: dict[tuple[str, int], list[str] | tuple[str, ...]],
) -> int:
    return sum(
        routed_family_blob_bytes(projection, dict(Counter(families)))
        for (projection, _layer), families in profiles.items()
    )


def _energy_fraction(
    profiles: dict[tuple[str, int], list[str] | tuple[str, ...]],
    energy: dict[tuple[int, int], dict[str, float]],
    projection: str,
    families: set[str],
) -> float:
    total = sum(
        energy[(layer, expert)][projection]
        for layer in range(43)
        for expert in range(256)
    )
    selected = sum(
        energy[(layer, expert)][projection]
        for layer in range(43)
        for expert, family in enumerate(profiles[(projection, layer)])
        if family in families
    )
    return selected / total


def _allocate_v4f_sensitivity_reallocation(
    base_profiles: dict[tuple[str, int], tuple[str, ...]],
    metadata: dict[str, object],
    energy: dict[tuple[int, int], dict[str, float]],
    marks: ExpertSensitivityMap,
    *,
    target_bytes: int | None,
    container_reserve_bytes: int,
) -> V4FSensitivityReallocation:
    allowed = {"NEPQ0-S", "NVQ2J", "NINT4"}
    actual = {
        family
        for families in base_profiles.values()
        for family in families
    }
    if not actual <= allowed:
        raise ValueError(
            "sensitivity reallocation requires a "
            "NEPQ0-S/NVQ2J/NINT4 base; "
            f"found {sorted(actual)}"
        )
    base = {
        key: list(families)
        for key, families in base_profiles.items()
    }
    current = {
        key: list(families)
        for key, families in base_profiles.items()
    }
    baseline_routed = _profile_routed_bytes(base)
    if baseline_routed != int(metadata["routed_bytes"]):
        raise ValueError("baseline routed byte accounting changed")

    protected = {
        layer: marks.experts(layer, "V")
        for layer in range(43)
        if marks.experts(layer, "V")
    }
    for (projection, layer), families in current.items():
        for expert in protected.get(layer, ()):
            families[expert] = "NINT8"
    protected_routed = _profile_routed_bytes(current)

    nonexpert_bytes = int(metadata["nonexpert_bytes"])
    resolved_target = (
        nonexpert_bytes + protected_routed + int(container_reserve_bytes)
        if target_bytes is None
        else int(target_bytes)
    )
    routed_limit = (
        resolved_target
        - int(container_reserve_bytes)
        - nonexpert_bytes
    )
    demoted: dict[str, dict[int, list[int]]] = {
        projection: {}
        for projection in PROJECTIONS
    }
    current_routed = protected_routed
    while current_routed > routed_limit:
        best = None
        for (projection, layer), families in current.items():
            before_counts = Counter(families)
            before_bytes = routed_family_blob_bytes(
                projection,
                dict(before_counts),
            )
            for expert, family in enumerate(families):
                if family != "NINT4" or marks.mark(layer, expert) != "w":
                    continue
                after_counts = dict(before_counts)
                after_counts["NINT4"] -= 1
                after_counts["NVQ2J"] = after_counts.get("NVQ2J", 0) + 1
                saving = before_bytes - routed_family_blob_bytes(
                    projection,
                    after_counts,
                )
                if saving <= 0:
                    raise RuntimeError("NINT4 demotion has nonpositive savings")
                value = energy[(layer, expert)][projection]
                rank = (
                    value / saving,
                    value,
                    -saving,
                    projection,
                    layer,
                    expert,
                )
                if best is None or rank < best[0]:
                    best = (
                        rank,
                        projection,
                        layer,
                        expert,
                        saving,
                    )
        if best is None:
            raise ValueError(
                "the target cannot be met without demoting a v/V expert"
            )
        _rank, projection, layer, expert, saving = best
        current[(projection, layer)][expert] = "NVQ2J"
        demoted[projection].setdefault(layer, []).append(expert)
        current_routed -= saving

    exact_routed = _profile_routed_bytes(current)
    if exact_routed != current_routed:
        raise RuntimeError("incremental routed accounting mismatch")
    for projection in PROJECTIONS:
        for layer, experts in demoted[projection].items():
            for expert in experts:
                if marks.mark(layer, expert) != "w":
                    raise RuntimeError("sensitive expert was demoted")

    protected_profiles = {
        key: tuple(
            "NINT8" if marks.mark(key[1], expert) == "V" else family
            for expert, family in enumerate(families)
        )
        for key, families in base_profiles.items()
    }
    return V4FSensitivityReallocation(
        target_bytes=resolved_target,
        container_reserve_bytes=int(container_reserve_bytes),
        base_file_bytes=int(metadata["file_bytes"]),
        nonexpert_bytes=nonexpert_bytes,
        baseline_routed_bytes=baseline_routed,
        protected_routed_bytes=protected_routed,
        routed_bytes=exact_routed,
        estimated_blob_bytes=nonexpert_bytes + exact_routed,
        base_gate_up_nepq0s=_profiles_by_layer(
            base, "gate_up", "NEPQ0-S"
        ),
        base_down_nepq0s=_profiles_by_layer(base, "down", "NEPQ0-S"),
        base_gate_up_nint4=_profiles_by_layer(base, "gate_up", "NINT4"),
        base_down_nint4=_profiles_by_layer(base, "down", "NINT4"),
        gate_up_nint4=_profiles_by_layer(current, "gate_up", "NINT4"),
        down_nint4=_profiles_by_layer(current, "down", "NINT4"),
        gate_up_nint8=_profiles_by_layer(current, "gate_up", "NINT8"),
        down_nint8=_profiles_by_layer(current, "down", "NINT8"),
        gate_up_demoted=_freeze(demoted["gate_up"]),
        down_demoted=_freeze(demoted["down"]),
        baseline_gate_up_energy_fraction=_energy_fraction(
            base,
            energy,
            "gate_up",
            {"NINT4"},
        ),
        baseline_down_energy_fraction=_energy_fraction(
            base,
            energy,
            "down",
            {"NINT4"},
        ),
        final_gate_up_energy_fraction=_energy_fraction(
            current,
            energy,
            "gate_up",
            {"NINT4", "NINT8"},
        ),
        final_down_energy_fraction=_energy_fraction(
            current,
            energy,
            "down",
            {"NINT4", "NINT8"},
        ),
        protected_gate_up_energy_fraction=_energy_fraction(
            protected_profiles,
            energy,
            "gate_up",
            {"NINT8"},
        ),
        protected_down_energy_fraction=_energy_fraction(
            protected_profiles,
            energy,
            "down",
            {"NINT8"},
        ),
        mark_counts={
            mark: marks.count(mark)
            for mark in ("V", "v", "w")
        },
        base_allocation_sha256=str(metadata["allocation_sha256"]),
        source_index_sha256=str(metadata["source_index_sha256"]),
    )


def allocate_v4f_sensitivity_reallocation(
    base_mfq: str | Path,
    reap_csv: str | Path,
    sensitivity_map: str | Path,
    *,
    target_bytes: int | None = None,
    container_reserve_bytes: int = 4_000_000,
) -> V4FSensitivityReallocation:
    """Protect uppercase-V experts at NINT8 over a mixed-precision base."""

    profiles, metadata = _read_v4f_mfq_profiles(base_mfq)
    marks = load_expert_sensitivity_map(
        sensitivity_map,
        expected_layers=43,
        expected_experts=256,
    )
    return _allocate_v4f_sensitivity_reallocation(
        profiles,
        metadata,
        _read_reap(reap_csv),
        marks,
        target_bytes=target_bytes,
        container_reserve_bytes=int(container_reserve_bytes),
    )


def allocate_v4f_marked_nint8_upgrade(
    base_allocation_path: str | Path,
    reap_csv: str | Path,
    sensitivity_map: str | Path,
    *,
    target_bytes: int | None = None,
    container_reserve_bytes: int = 4_000_000,
) -> V4FMarkedNint8Upgrade:
    """Upgrade every uppercase-V expert projection to NINT8.

    The existing NEPQ0-S/NVQ2J assignment remains unchanged for every
    non-protected expert.
    """

    base = json.loads(Path(base_allocation_path).read_text(encoding="utf-8"))
    if base.get("format") != "mfq.v4f-ew-allocation.v1":
        raise ValueError("the base allocation must be a V4F EW v1 document")
    marks = load_expert_sensitivity_map(
        sensitivity_map,
        expected_layers=43,
        expected_experts=256,
    )
    protected = _freeze(
        {
            layer: list(marks.experts(layer, "V"))
            for layer in range(43)
        }
    )
    if not protected:
        raise ValueError("the expert sensitivity map has no uppercase-V experts")
    base_high = {
        "gate_up": _read_map(base, "gate_up_high"),
        "down": _read_map(base, "down_high"),
    }

    base_routed = 0
    routed = 0
    for projection in PROJECTIONS:
        for layer in range(43):
            high = set(base_high[projection].get(layer, ()))
            selected = set(protected.get(layer, ()))
            base_routed += routed_family_blob_bytes(
                projection,
                {
                    "NEPQ0-S": 256 - len(high),
                    "NVQ2J": len(high),
                },
            )
            routed += routed_family_blob_bytes(
                projection,
                {
                    "NEPQ0-S": 256 - len(high | selected),
                    "NVQ2J": len(high - selected),
                    "NINT8": len(selected),
                },
            )
    recorded_routed = int(base["routed_bytes"])
    if base_routed != recorded_routed:
        raise ValueError(
            f"base routed accounting mismatch: computed={base_routed}, "
            f"recorded={recorded_routed}"
        )

    nonexpert = int(base["nonexpert_bytes"])
    estimated = nonexpert + routed
    minimum_target = estimated + int(container_reserve_bytes)
    resolved_target = minimum_target if target_bytes is None else int(target_bytes)
    if minimum_target > resolved_target:
        raise ValueError(
            "marked NINT8 protection exceeds the requested target by "
            f"{minimum_target - resolved_target} bytes"
        )

    energy = _read_reap(reap_csv)
    fractions: dict[str, float] = {}
    payload_bytes = 0
    for projection in PROJECTIONS:
        total = sum(
            energy[(layer, expert)][projection]
            for layer in range(43)
            for expert in range(256)
        )
        protected_energy = sum(
            energy[(layer, expert)][projection]
            for layer, experts in protected.items()
            for expert in experts
        )
        fractions[projection] = protected_energy / total
        columns = 4096 if projection == "gate_up" else 2048
        payload_bytes += sum(
            _nint_blob_nbytes(len(experts) * 4096, columns, NINT8)
            for experts in protected.values()
        )

    return V4FMarkedNint8Upgrade(
        target_bytes=resolved_target,
        container_reserve_bytes=int(container_reserve_bytes),
        nonexpert_bytes=nonexpert,
        routed_bytes=int(routed),
        estimated_blob_bytes=int(estimated),
        base_gate_up_high=base_high["gate_up"],
        base_down_high=base_high["down"],
        gate_up_nint8=protected,
        down_nint8=protected,
        gate_up_energy_fraction=fractions["gate_up"],
        down_energy_fraction=fractions["down"],
        nint8_payload_bytes=int(payload_bytes),
        mark_counts={
            mark: marks.count(mark)
            for mark in ("V", "v", "w")
        },
    )


def allocate_v4f_nint4_upgrade(
    base_allocation_path: str | Path,
    reap_csv: str | Path,
    *,
    target_bytes: int = 45_000_000_000,
    container_reserve_bytes: int = 4_000_000,
) -> V4FNint4Upgrade:
    """Greedily maximize observed projection energy per exact added byte.

    The existing NEPQ0-S/NVQ2J assignment is immutable.  The only allowed
    transition is from the assigned family to NINT4.
    """

    base = json.loads(Path(base_allocation_path).read_text(encoding="utf-8"))
    if base.get("format") != "mfq.v4f-ew-allocation.v1":
        raise ValueError("the base allocation must be a V4F EW v1 document")
    energy = _read_reap(reap_csv)
    base_high = {
        "gate_up": _read_map(base, "gate_up_high"),
        "down": _read_map(base, "down_high"),
    }
    counts: dict[tuple[str, int], dict[str, int]] = {}
    streams: dict[tuple[str, int, str], list[int]] = {}
    positions: dict[tuple[str, int, str], int] = {}
    routed = 0
    for projection in PROJECTIONS:
        for layer in range(43):
            high = set(base_high[projection].get(layer, ()))
            family_counts = {
                "NEPQ0-S": 256 - len(high),
                "NVQ2J": len(high),
                "NINT4": 0,
            }
            counts[(projection, layer)] = family_counts
            routed += routed_family_blob_bytes(projection, family_counts)
            for family, experts in (
                ("NVQ2J", high),
                ("NEPQ0-S", set(range(256)) - high),
            ):
                key = (projection, layer, family)
                streams[key] = sorted(
                    experts,
                    key=lambda expert: (
                        -energy[(layer, expert)][projection],
                        expert,
                    ),
                )
                positions[key] = 0
    recorded_routed = int(base["routed_bytes"])
    if routed != recorded_routed:
        raise ValueError(
            f"base routed accounting mismatch: computed={routed}, "
            f"recorded={recorded_routed}"
        )

    nonexpert = int(base["nonexpert_bytes"])
    routed_limit = int(target_bytes) - int(container_reserve_bytes) - nonexpert
    if routed > routed_limit:
        raise ValueError("base allocation already exceeds the requested target")
    selected: dict[str, dict[int, list[int]]] = {
        "gate_up": {},
        "down": {},
    }

    while True:
        best = None
        for key, experts in streams.items():
            index = positions[key]
            if index >= len(experts):
                continue
            projection, layer, source_family = key
            before_counts = counts[(projection, layer)]
            after_counts = dict(before_counts)
            after_counts[source_family] -= 1
            after_counts["NINT4"] += 1
            delta = (
                routed_family_blob_bytes(projection, after_counts)
                - routed_family_blob_bytes(projection, before_counts)
            )
            if delta <= 0:
                raise RuntimeError("NINT4 upgrade has nonpositive byte cost")
            if routed + delta > routed_limit:
                continue
            expert = experts[index]
            value = energy[(layer, expert)][projection]
            rank = (
                value / delta,
                value,
                -delta,
                0 if projection == "down" else 1,
                -layer,
                -expert,
            )
            if best is None or rank > best[0]:
                best = (
                    rank,
                    key,
                    expert,
                    delta,
                    after_counts,
                )
        if best is None:
            break
        _rank, key, expert, delta, after_counts = best
        projection, layer, _source_family = key
        positions[key] += 1
        counts[(projection, layer)] = after_counts
        routed += delta
        selected[projection].setdefault(layer, []).append(expert)

    frozen = {
        projection: _freeze(selected[projection])
        for projection in PROJECTIONS
    }
    fractions: dict[str, float] = {}
    payload_bytes = 0
    for projection in PROJECTIONS:
        total = sum(
            energy[(layer, expert)][projection]
            for layer in range(43)
            for expert in range(256)
        )
        kept = sum(
            energy[(layer, expert)][projection]
            for layer, experts in frozen[projection].items()
            for expert in experts
        )
        fractions[projection] = kept / total
        columns = 4096 if projection == "gate_up" else 2048
        payload_bytes += sum(
            _nint_blob_nbytes(len(experts) * 4096, columns, NINT4)
            for experts in frozen[projection].values()
        )

    return V4FNint4Upgrade(
        target_bytes=int(target_bytes),
        container_reserve_bytes=int(container_reserve_bytes),
        nonexpert_bytes=nonexpert,
        routed_bytes=int(routed),
        estimated_blob_bytes=int(nonexpert + routed),
        base_gate_up_high=base_high["gate_up"],
        base_down_high=base_high["down"],
        gate_up_nint4=frozen["gate_up"],
        down_nint4=frozen["down"],
        gate_up_energy_fraction=fractions["gate_up"],
        down_energy_fraction=fractions["down"],
        nint4_payload_bytes=int(payload_bytes),
    )


def allocation_document(
    allocation: V4FNint4Upgrade,
    *,
    base_allocation_path: str | Path,
    reap_csv: str | Path,
    source_index_sha256: str,
) -> dict:
    def mapping(values: dict[int, tuple[int, ...]]) -> dict[str, list[int]]:
        return {
            str(layer): list(experts)
            for layer, experts in sorted(values.items())
        }

    gate_from_nvq = sum(
        expert in set(allocation.base_gate_up_high.get(layer, ()))
        for layer, experts in allocation.gate_up_nint4.items()
        for expert in experts
    )
    down_from_nvq = sum(
        expert in set(allocation.base_down_high.get(layer, ()))
        for layer, experts in allocation.down_nint4.items()
        for expert in experts
    )
    gate_count = sum(map(len, allocation.gate_up_nint4.values()))
    down_count = sum(map(len, allocation.down_nint4.values()))
    return {
        "format": "mfq.v4f-nint4-upgrade.v1",
        "profile": "nepq0s-nvq2j-nint4",
        "target_bytes": allocation.target_bytes,
        "container_reserve_bytes": allocation.container_reserve_bytes,
        "nonexpert_bytes": allocation.nonexpert_bytes,
        "routed_bytes": allocation.routed_bytes,
        "estimated_blob_bytes": allocation.estimated_blob_bytes,
        "estimated_headroom_bytes": (
            allocation.target_bytes - allocation.estimated_blob_bytes
        ),
        "base_gate_up_high": mapping(allocation.base_gate_up_high),
        "base_down_high": mapping(allocation.base_down_high),
        "gate_up_nint4": mapping(allocation.gate_up_nint4),
        "down_nint4": mapping(allocation.down_nint4),
        "gate_up_nint4_count": gate_count,
        "down_nint4_count": down_count,
        "gate_up_from_nvq2j": gate_from_nvq,
        "gate_up_from_nepq0s": gate_count - gate_from_nvq,
        "down_from_nvq2j": down_from_nvq,
        "down_from_nepq0s": down_count - down_from_nvq,
        "gate_up_energy_fraction": allocation.gate_up_energy_fraction,
        "down_energy_fraction": allocation.down_energy_fraction,
        "nint4_payload_bytes": allocation.nint4_payload_bytes,
        "allocation_objective": "reap_projection_energy_per_exact_added_byte",
        "single_intervention": (
            "upgrade selected existing NEPQ0-S/NVQ2J expert projections "
            "to NINT4"
        ),
        "base_allocation": str(Path(base_allocation_path).resolve()),
        "base_allocation_sha256": sha256_file(base_allocation_path),
        "reap_csv": str(Path(reap_csv).resolve()),
        "reap_sha256": sha256_file(reap_csv),
        "source_index_sha256": str(source_index_sha256),
        "mtp_included": False,
    }


def marked_allocation_document(
    allocation: V4FMarkedNint8Upgrade,
    *,
    base_allocation_path: str | Path,
    reap_csv: str | Path,
    sensitivity_map: str | Path,
    source_index_sha256: str,
) -> dict:
    def mapping(values: dict[int, tuple[int, ...]]) -> dict[str, list[int]]:
        return {
            str(layer): list(experts)
            for layer, experts in sorted(values.items())
        }

    gate_from_nvq = sum(
        expert in set(allocation.base_gate_up_high.get(layer, ()))
        for layer, experts in allocation.gate_up_nint8.items()
        for expert in experts
    )
    down_from_nvq = sum(
        expert in set(allocation.base_down_high.get(layer, ()))
        for layer, experts in allocation.down_nint8.items()
        for expert in experts
    )
    gate_count = sum(map(len, allocation.gate_up_nint8.values()))
    down_count = sum(map(len, allocation.down_nint8.values()))
    return {
        "format": "mfq.v4f-marked-nint8-upgrade.v1",
        "profile": "nepq0s-nvq2j-nint8-protected",
        "target_bytes": allocation.target_bytes,
        "container_reserve_bytes": allocation.container_reserve_bytes,
        "nonexpert_bytes": allocation.nonexpert_bytes,
        "routed_bytes": allocation.routed_bytes,
        "estimated_blob_bytes": allocation.estimated_blob_bytes,
        "estimated_headroom_bytes": (
            allocation.target_bytes - allocation.estimated_blob_bytes
        ),
        "base_gate_up_high": mapping(allocation.base_gate_up_high),
        "base_down_high": mapping(allocation.base_down_high),
        "gate_up_nint8": mapping(allocation.gate_up_nint8),
        "down_nint8": mapping(allocation.down_nint8),
        "gate_up_nint8_count": gate_count,
        "down_nint8_count": down_count,
        "gate_up_from_nvq2j": gate_from_nvq,
        "gate_up_from_nepq0s": gate_count - gate_from_nvq,
        "down_from_nvq2j": down_from_nvq,
        "down_from_nepq0s": down_count - down_from_nvq,
        "gate_up_energy_fraction": allocation.gate_up_energy_fraction,
        "down_energy_fraction": allocation.down_energy_fraction,
        "nint8_payload_bytes": allocation.nint8_payload_bytes,
        "mark_counts": dict(allocation.mark_counts),
        "allocation_objective": "hard_uppercase_V_to_NINT8",
        "single_intervention": (
            "upgrade both expert projections marked uppercase V to NINT8; "
            "preserve every other NEPQ0-S/NVQ2J assignment"
        ),
        "base_allocation": str(Path(base_allocation_path).resolve()),
        "base_allocation_sha256": sha256_file(base_allocation_path),
        "reap_csv": str(Path(reap_csv).resolve()),
        "reap_sha256": sha256_file(reap_csv),
        "sensitivity_map": str(Path(sensitivity_map).resolve()),
        "sensitivity_map_sha256": sha256_file(sensitivity_map),
        "source_index_sha256": str(source_index_sha256),
        "mtp_included": False,
    }


def sensitivity_reallocation_document(
    allocation: V4FSensitivityReallocation,
    *,
    base_mfq: str | Path,
    reap_csv: str | Path,
    sensitivity_map: str | Path,
) -> dict:
    def mapping(values: dict[int, tuple[int, ...]]) -> dict[str, list[int]]:
        return {
            str(layer): list(experts)
            for layer, experts in sorted(values.items())
        }

    def count(values: dict[int, tuple[int, ...]]) -> int:
        return sum(map(len, values.values()))

    gate_nint4 = count(allocation.gate_up_nint4)
    down_nint4 = count(allocation.down_nint4)
    gate_nint8 = count(allocation.gate_up_nint8)
    down_nint8 = count(allocation.down_nint8)
    gate_demoted = count(allocation.gate_up_demoted)
    down_demoted = count(allocation.down_demoted)
    return {
        "format": "mfq.v4f-sensitivity-reallocation.v1",
        "profile": "nepq0s-nvq2j-nint4-nint8-V-protected",
        "target_bytes": allocation.target_bytes,
        "container_reserve_bytes": allocation.container_reserve_bytes,
        "base_file_bytes": allocation.base_file_bytes,
        "nonexpert_bytes": allocation.nonexpert_bytes,
        "baseline_routed_bytes": allocation.baseline_routed_bytes,
        "protected_routed_bytes_before_rebalance": (
            allocation.protected_routed_bytes
        ),
        "protected_added_bytes": (
            allocation.protected_routed_bytes
            - allocation.baseline_routed_bytes
        ),
        "rebalance_saved_bytes": (
            allocation.protected_routed_bytes - allocation.routed_bytes
        ),
        "routed_bytes": allocation.routed_bytes,
        "estimated_blob_bytes": allocation.estimated_blob_bytes,
        "estimated_headroom_bytes": (
            allocation.target_bytes - allocation.estimated_blob_bytes
        ),
        "base_gate_up_nint4": mapping(allocation.base_gate_up_nint4),
        "base_down_nint4": mapping(allocation.base_down_nint4),
        "base_gate_up_nepq0s": mapping(allocation.base_gate_up_nepq0s),
        "base_down_nepq0s": mapping(allocation.base_down_nepq0s),
        "gate_up_nint4": mapping(allocation.gate_up_nint4),
        "down_nint4": mapping(allocation.down_nint4),
        "gate_up_nint8": mapping(allocation.gate_up_nint8),
        "down_nint8": mapping(allocation.down_nint8),
        "gate_up_demoted": mapping(allocation.gate_up_demoted),
        "down_demoted": mapping(allocation.down_demoted),
        "base_gate_up_nint4_count": count(
            allocation.base_gate_up_nint4
        ),
        "base_down_nint4_count": count(allocation.base_down_nint4),
        "gate_up_nint4_count": gate_nint4,
        "down_nint4_count": down_nint4,
        "gate_up_nint8_count": gate_nint8,
        "down_nint8_count": down_nint8,
        "gate_up_nepq0s_count": sum(
            family == "NEPQ0-S"
            for layer in range(43)
            for family in allocation.families("gate_up", layer)
        ),
        "down_nepq0s_count": sum(
            family == "NEPQ0-S"
            for layer in range(43)
            for family in allocation.families("down", layer)
        ),
        "gate_up_nvq2j_count": sum(
            family == "NVQ2J"
            for layer in range(43)
            for family in allocation.families("gate_up", layer)
        ),
        "down_nvq2j_count": sum(
            family == "NVQ2J"
            for layer in range(43)
            for family in allocation.families("down", layer)
        ),
        "gate_up_demoted_count": gate_demoted,
        "down_demoted_count": down_demoted,
        "baseline_gate_up_energy_fraction": (
            allocation.baseline_gate_up_energy_fraction
        ),
        "baseline_down_energy_fraction": (
            allocation.baseline_down_energy_fraction
        ),
        "final_gate_up_energy_fraction": (
            allocation.final_gate_up_energy_fraction
        ),
        "final_down_energy_fraction": (
            allocation.final_down_energy_fraction
        ),
        "protected_gate_up_energy_fraction": (
            allocation.protected_gate_up_energy_fraction
        ),
        "protected_down_energy_fraction": (
            allocation.protected_down_energy_fraction
        ),
        "mark_counts": dict(allocation.mark_counts),
        "allocation_objective": (
            "hard_V_to_NINT8_then_minimum_REAP_loss_per_saved_byte"
        ),
        "constraints": [
            "uppercase V uses NINT8 for gate_up and down",
            "lowercase v is never demoted",
            "only w experts already using NINT4 may be demoted",
            "all remaining assignments preserve the base model",
        ],
        "base_mfq": str(Path(base_mfq).resolve()),
        "base_allocation_sha256": allocation.base_allocation_sha256,
        "reap_csv": str(Path(reap_csv).resolve()),
        "reap_sha256": sha256_file(reap_csv),
        "sensitivity_map": str(Path(sensitivity_map).resolve()),
        "sensitivity_map_sha256": sha256_file(sensitivity_map),
        "source_index_sha256": allocation.source_index_sha256,
        "mtp_included": False,
    }


def load_upgrade(path: str | Path) -> V4FUpgradePlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format") == "mfq.v4f-nint4-upgrade.v1":
        return V4FNint4Upgrade(
            target_bytes=int(raw["target_bytes"]),
            container_reserve_bytes=int(raw["container_reserve_bytes"]),
            nonexpert_bytes=int(raw["nonexpert_bytes"]),
            routed_bytes=int(raw["routed_bytes"]),
            estimated_blob_bytes=int(raw["estimated_blob_bytes"]),
            base_gate_up_high=_read_map(raw, "base_gate_up_high"),
            base_down_high=_read_map(raw, "base_down_high"),
            gate_up_nint4=_read_map(raw, "gate_up_nint4"),
            down_nint4=_read_map(raw, "down_nint4"),
            gate_up_energy_fraction=float(raw["gate_up_energy_fraction"]),
            down_energy_fraction=float(raw["down_energy_fraction"]),
            nint4_payload_bytes=int(raw["nint4_payload_bytes"]),
        )
    if raw.get("format") == "mfq.v4f-marked-nint8-upgrade.v1":
        return V4FMarkedNint8Upgrade(
            target_bytes=int(raw["target_bytes"]),
            container_reserve_bytes=int(raw["container_reserve_bytes"]),
            nonexpert_bytes=int(raw["nonexpert_bytes"]),
            routed_bytes=int(raw["routed_bytes"]),
            estimated_blob_bytes=int(raw["estimated_blob_bytes"]),
            base_gate_up_high=_read_map(raw, "base_gate_up_high"),
            base_down_high=_read_map(raw, "base_down_high"),
            gate_up_nint8=_read_map(raw, "gate_up_nint8"),
            down_nint8=_read_map(raw, "down_nint8"),
            gate_up_energy_fraction=float(raw["gate_up_energy_fraction"]),
            down_energy_fraction=float(raw["down_energy_fraction"]),
            nint8_payload_bytes=int(raw["nint8_payload_bytes"]),
            mark_counts={
                str(mark): int(count)
                for mark, count in raw["mark_counts"].items()
            },
        )
    if raw.get("format") == "mfq.v4f-sensitivity-reallocation.v1":
        return V4FSensitivityReallocation(
            target_bytes=int(raw["target_bytes"]),
            container_reserve_bytes=int(raw["container_reserve_bytes"]),
            base_file_bytes=int(raw["base_file_bytes"]),
            nonexpert_bytes=int(raw["nonexpert_bytes"]),
            baseline_routed_bytes=int(raw["baseline_routed_bytes"]),
            protected_routed_bytes=int(
                raw["protected_routed_bytes_before_rebalance"]
            ),
            routed_bytes=int(raw["routed_bytes"]),
            estimated_blob_bytes=int(raw["estimated_blob_bytes"]),
            base_gate_up_nepq0s=_read_map(raw, "base_gate_up_nepq0s"),
            base_down_nepq0s=_read_map(raw, "base_down_nepq0s"),
            base_gate_up_nint4=_read_map(raw, "base_gate_up_nint4"),
            base_down_nint4=_read_map(raw, "base_down_nint4"),
            gate_up_nint4=_read_map(raw, "gate_up_nint4"),
            down_nint4=_read_map(raw, "down_nint4"),
            gate_up_nint8=_read_map(raw, "gate_up_nint8"),
            down_nint8=_read_map(raw, "down_nint8"),
            gate_up_demoted=_read_map(raw, "gate_up_demoted"),
            down_demoted=_read_map(raw, "down_demoted"),
            baseline_gate_up_energy_fraction=float(
                raw["baseline_gate_up_energy_fraction"]
            ),
            baseline_down_energy_fraction=float(
                raw["baseline_down_energy_fraction"]
            ),
            final_gate_up_energy_fraction=float(
                raw["final_gate_up_energy_fraction"]
            ),
            final_down_energy_fraction=float(
                raw["final_down_energy_fraction"]
            ),
            protected_gate_up_energy_fraction=float(
                raw["protected_gate_up_energy_fraction"]
            ),
            protected_down_energy_fraction=float(
                raw["protected_down_energy_fraction"]
            ),
            mark_counts={
                str(mark): int(count)
                for mark, count in raw["mark_counts"].items()
            },
            base_allocation_sha256=str(raw["base_allocation_sha256"]),
            source_index_sha256=str(raw["source_index_sha256"]),
        )
    raise ValueError(f"unsupported V4F upgrade plan: {path}")


__all__ = [
    "NINT4",
    "NINT8",
    "NINT_SPECS",
    "PROJECTIONS",
    "V4FMarkedNint8Upgrade",
    "V4FNint4Upgrade",
    "V4FSensitivityReallocation",
    "V4FUpgradePlan",
    "allocate_v4f_marked_nint8_upgrade",
    "allocate_v4f_nint4_upgrade",
    "allocate_v4f_sensitivity_reallocation",
    "allocation_document",
    "load_upgrade",
    "marked_allocation_document",
    "sensitivity_reallocation_document",
    "sha256_file",
]
