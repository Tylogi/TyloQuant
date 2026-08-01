from __future__ import annotations

import csv
import json

import numpy as np

from mfq.formats.io import _pack_tensor, unpack_nint_moe
from mfq.formats.moe import NintMoePool, NintMoeTensor
from mfq.formats.nepq import NEPQ0_S
from mfq.formats.nint import NintSpec
from mfq.quantize.expert_sensitivity import load_expert_sensitivity_map
from mfq.quantize.v4f_plan import routed_family_blob_bytes
from mfq.quantize.v4f_upgrade import (
    _nintm_profiles,
    _allocate_v4f_sensitivity_reallocation,
    allocate_v4f_marked_nint8_upgrade,
    allocate_v4f_nint4_upgrade,
    load_upgrade,
    marked_allocation_document,
    sensitivity_reallocation_document,
)
from mfq.quantize.nint_quant import quantize
from mfq.tools.upgrade_v4f_mfq import (
    _allocation_family,
    _nintm_allocation_profiles,
    _subset_pool_tensor,
    _write_upgraded_routed_blob,
)
from tests.mixed_family_fixtures import make_flat_family
from tests.test_formats.test_nepq import _tensor as make_nepq


def _write_reap(path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("layer", "expert_id", "gate_up_energy", "down_energy"),
        )
        writer.writeheader()
        for layer in range(43):
            for expert in range(256):
                value = layer * 256 + expert + 1
                writer.writerow(
                    {
                        "layer": layer,
                        "expert_id": expert,
                        "gate_up_energy": value,
                        "down_energy": value * value,
                    }
                )


def _write_base(path) -> int:
    routed = 0
    gate = {}
    down = {}
    for layer in range(43):
        gate[str(layer)] = list(range(4))
        down[str(layer)] = list(range(8))
        routed += routed_family_blob_bytes(
            "gate_up", {"NEPQ0-S": 252, "NVQ2J": 4}
        )
        routed += routed_family_blob_bytes(
            "down", {"NEPQ0-S": 248, "NVQ2J": 8}
        )
    path.write_text(
        json.dumps(
            {
                "format": "mfq.v4f-ew-allocation.v1",
                "nonexpert_bytes": 0,
                "routed_bytes": routed,
                "estimated_blob_bytes": routed,
                "gate_up_high": gate,
                "down_high": down,
            }
        ),
        encoding="utf-8",
    )
    return routed


def _write_marks(path) -> None:
    rows = []
    protected = {4: {170}, 41: {84, 154}}
    for layer in range(43):
        values = ["w"] * 256
        for expert in protected.get(layer, ()):
            values[expert] = "V"
        if layer == 0:
            values[0] = "v"
        rows.append(f'  "{layer}": "{"".join(values)}",')
    path.write_text("\n".join(rows) + "\n},\n", encoding="utf-8")


def test_nintm_profiles_preserves_duplicate_family_pools() -> None:
    rng = np.random.default_rng(20260725)
    rows = 2
    width = 24
    first = quantize(
        rng.normal(size=(2 * rows, width)).astype(np.float32),
        NintSpec(4, 24, 6),
    )
    second = quantize(
        rng.normal(size=(2 * rows, width)).astype(np.float32),
        NintSpec(4, 24, 6),
    )
    tensor = NintMoeTensor(
        (4, rows, width),
        (
            NintMoePool(np.array([0, 1], dtype=np.int32), first),
            NintMoePool(np.array([2, 3], dtype=np.int32), second),
        ),
    )
    dtype, blob = _pack_tensor(tensor)
    assert dtype == "NINTM"
    shape, profiles, pools = _nintm_profiles(blob)
    assert shape == (4, rows, width)
    assert profiles == ("NINT4",) * 4
    assert pools == (("NINT4", 2), ("NINT4", 2))


def test_expert_sensitivity_map_validates_shape_and_case(tmp_path) -> None:
    path = tmp_path / "marks.txt"
    _write_marks(path)
    marks = load_expert_sensitivity_map(
        path,
        expected_layers=43,
        expected_experts=256,
    )
    assert marks.count("V") == 3
    assert marks.count("v") == 1
    assert marks.count("w") == 43 * 256 - 4
    assert marks.experts(41, "V") == (84, 154)


def test_marked_nint8_upgrade_protects_only_uppercase_v(tmp_path) -> None:
    reap = tmp_path / "reap.csv"
    base = tmp_path / "base.json"
    marks = tmp_path / "marks.txt"
    _write_reap(reap)
    _write_base(base)
    _write_marks(marks)

    allocation = allocate_v4f_marked_nint8_upgrade(base, reap, marks)
    assert allocation.nint8_count == 6
    assert allocation.gate_up_nint8 == {4: (170,), 41: (84, 154)}
    assert allocation.down_nint8 == allocation.gate_up_nint8
    assert allocation.mark_counts == {"V": 3, "v": 1, "w": 43 * 256 - 4}
    assert allocation.target_bytes == (
        allocation.estimated_blob_bytes + allocation.container_reserve_bytes
    )
    assert allocation.families("gate_up", 4)[170] == "NINT8"
    assert allocation.families("gate_up", 0)[0] != "NINT8"

    document = marked_allocation_document(
        allocation,
        base_allocation_path=base,
        reap_csv=reap,
        sensitivity_map=marks,
        source_index_sha256="source-index",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(document), encoding="utf-8")
    restored = load_upgrade(plan)
    assert restored == allocation


def test_v4f_nint4_upgrade_preserves_base_and_fills_budget(tmp_path) -> None:
    reap = tmp_path / "reap.csv"
    base = tmp_path / "base.json"
    _write_reap(reap)
    routed = _write_base(base)
    target = routed + 400_000_000
    first = allocate_v4f_nint4_upgrade(
        base,
        reap,
        target_bytes=target,
        container_reserve_bytes=4_000_000,
    )
    second = allocate_v4f_nint4_upgrade(
        base,
        reap,
        target_bytes=target,
        container_reserve_bytes=4_000_000,
    )
    assert first == second
    assert first.nint4_count > 0
    assert first.estimated_blob_bytes + 4_000_000 <= target
    for projection in ("gate_up", "down"):
        for layer in range(43):
            families = first.families(projection, layer)
            assert len(families) == 256
            assert set(families) <= {"NEPQ0-S", "NVQ2J", "NINT4"}
    assert 0 <= first.gate_up_energy_fraction <= 1
    assert 0 <= first.down_energy_fraction <= 1
    assert first.gate_up_energy_fraction + first.down_energy_fraction > 0


def test_sensitivity_upgrade_preserves_mixed_base_and_roundtrips(tmp_path) -> None:
    reap = tmp_path / "reap.csv"
    marks_path = tmp_path / "marks.txt"
    _write_reap(reap)
    _write_marks(marks_path)
    marks = load_expert_sensitivity_map(
        marks_path,
        expected_layers=43,
        expected_experts=256,
    )
    profiles = {}
    for projection in ("gate_up", "down"):
        for layer in range(43):
            families = ["NEPQ0-S"] * 256
            families[0] = "NVQ2J"
            families[1] = "NINT4"
            profiles[(projection, layer)] = tuple(families)
    routed = sum(
        routed_family_blob_bytes(
            projection,
            {"NEPQ0-S": 254, "NVQ2J": 1, "NINT4": 1},
        )
        for projection in ("gate_up", "down")
        for _layer in range(43)
    )
    energy = {
        (layer, expert): {
            "gate_up": float(layer * 256 + expert + 1),
            "down": float((layer * 256 + expert + 1) ** 2),
        }
        for layer in range(43)
        for expert in range(256)
    }
    allocation = _allocate_v4f_sensitivity_reallocation(
        profiles,
        {
            "file_bytes": routed + 123,
            "routed_bytes": routed,
            "nonexpert_bytes": 0,
            "allocation_sha256": "base-allocation",
            "source_index_sha256": "source-index",
        },
        energy,
        marks,
        target_bytes=None,
        container_reserve_bytes=4_000_000,
    )

    assert allocation.demoted_count == 0
    assert allocation.nint8_count == 6
    assert allocation.target_bytes == (
        allocation.estimated_blob_bytes + allocation.container_reserve_bytes
    )
    assert allocation.base_families("gate_up", 0)[:3] == (
        "NVQ2J",
        "NINT4",
        "NEPQ0-S",
    )
    assert allocation.families("gate_up", 4)[170] == "NINT8"
    assert allocation.families("gate_up", 0)[0] == "NVQ2J"
    assert allocation.families("gate_up", 0)[1] == "NINT4"
    assert allocation.families("gate_up", 0)[2] == "NEPQ0-S"

    document = sensitivity_reallocation_document(
        allocation,
        base_mfq=tmp_path / "base.mfq",
        reap_csv=reap,
        sensitivity_map=marks_path,
    )
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(document), encoding="utf-8")
    assert load_upgrade(plan) == allocation


def test_write_upgraded_routed_blob_replaces_only_selected_experts(tmp_path) -> None:
    rows = 3
    neuron_len = 96
    base_tensor = make_flat_family(
        "NVQ2J",
        rows=3 * rows,
        neuron_len=neuron_len,
    )
    base = NintMoeTensor(
        (3, rows, neuron_len),
        (
            NintMoePool(
                np.asarray([0, 1, 2], dtype=np.int32),
                base_tensor,
            ),
        ),
    )
    selected_tensor = quantize(
        np.arange(rows * neuron_len, dtype=np.float32).reshape(rows, neuron_len),
        NintSpec(4, 24, 6),
        axis=0,
    )
    dtype, selected_payload = _pack_tensor(selected_tensor, allow_moe=False)
    assert dtype == "NINT4"
    selected_path = tmp_path / "selected.nint4"
    selected_path.write_bytes(selected_payload)

    output = tmp_path / "upgraded.nintm"
    _write_upgraded_routed_blob(base, (1,), selected_path, output)
    restored = unpack_nint_moe(output.read_bytes())

    assert restored.expert_profiles == ("NVQ2J", "NINT4-24", "NVQ2J")
    shape, allocation_profiles = _nintm_allocation_profiles(output.read_bytes())
    assert shape == (3, rows, neuron_len)
    assert allocation_profiles == ("NVQ2J", "NINT4", "NVQ2J")
    retained = restored.pools[0]
    assert retained.expert_ids.tolist() == [0, 2]
    original_rows = np.asarray(base_tensor.indices).reshape(3, rows, -1)
    retained_rows = np.asarray(retained.tensor.indices).reshape(2, rows, -1)
    np.testing.assert_array_equal(retained_rows[0], original_rows[0])
    np.testing.assert_array_equal(retained_rows[1], original_rows[2])
    np.testing.assert_array_equal(
        np.asarray(restored.pools[1].tensor.q),
        np.asarray(selected_tensor.q),
    )


def test_write_upgraded_routed_blob_accepts_nint8(tmp_path) -> None:
    rows = 3
    neuron_len = 96
    base_tensor = make_flat_family(
        "NVQ2J",
        rows=2 * rows,
        neuron_len=neuron_len,
    )
    base = NintMoeTensor(
        (2, rows, neuron_len),
        (
            NintMoePool(
                np.asarray([0, 1], dtype=np.int32),
                base_tensor,
            ),
        ),
    )
    selected_tensor = quantize(
        np.arange(rows * neuron_len, dtype=np.float32).reshape(rows, neuron_len),
        NintSpec(8, 48, 7),
        axis=0,
    )
    dtype, selected_payload = _pack_tensor(selected_tensor, allow_moe=False)
    assert dtype == "NINT8"
    selected_path = tmp_path / "selected.nint8"
    selected_path.write_bytes(selected_payload)

    output = tmp_path / "upgraded.nintm"
    _write_upgraded_routed_blob(
        base,
        (1,),
        selected_path,
        output,
        "NINT8",
    )
    restored = unpack_nint_moe(output.read_bytes())
    assert restored.expert_profiles == ("NVQ2J", "NINT8-48")


def test_subset_nepq_pool_keeps_exact_selected_expert_payload() -> None:
    tensor = make_nepq(NEPQ0_S)
    subset = _subset_pool_tensor(
        tensor,
        np.asarray([1], dtype=np.int64),
        pool_experts=2,
        rows_per_expert=tensor.out_per_expert,
    )

    assert subset.shape == (1, tensor.out_per_expert, tensor.neuron_len)
    np.testing.assert_array_equal(subset.neuron_scale, tensor.neuron_scale[[1]])
    np.testing.assert_array_equal(subset.state, tensor.state[[1]])
    np.testing.assert_array_equal(subset.indices, tensor.indices[[1]])
    np.testing.assert_array_equal(subset.bank_ids, tensor.bank_ids[[1]])
    np.testing.assert_array_equal(subset.table_payloads, tensor.table_payloads)


def test_runtime_nint_profile_maps_to_allocation_family() -> None:
    assert _allocation_family("NINT4-24") == "NINT4"
    assert _allocation_family("NVQ2J") == "NVQ2J"
    assert _allocation_family("NEPQ0-S") == "NEPQ0-S"
