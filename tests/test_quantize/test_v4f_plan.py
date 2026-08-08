from __future__ import annotations

import csv
import pytest

from mfq.quantize.v4f_source import V4FExpertSource
from mfq.quantize.v4f_plan import (
    allocate_v4f_ew_nvq2j_nint4,
    routed_blob_bytes,
    routed_family_blob_bytes,
    routed_family_pool_bytes,
)
from mfq.tools.quantize_v4f_to_mfq import (
    _layer_source_shards,
    _source_shards_ready,
)


def _write_reap(path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "layer",
                "expert_id",
                "gate_up_energy",
                "down_energy",
            ),
        )
        writer.writeheader()
        for layer in range(43):
            for expert in range(256):
                rank = layer * 256 + expert + 1
                writer.writerow(
                    {
                        "layer": layer,
                        "expert_id": expert,
                        "gate_up_energy": float(rank),
                        "down_energy": float(rank * rank),
                    }
                )


def test_v4f_family_accounting_preserves_legacy_two_pool_sizes() -> None:
    for projection in ("gate_up", "down"):
        for low_count in (0, 1, 127, 255, 256):
            high_count = 256 - low_count
            assert routed_family_blob_bytes(
                projection,
                {"NEPQ0-S": low_count, "NVQ2J": high_count},
            ) == routed_blob_bytes(projection, low_count, high_count)


def test_v4f_native_mxfp4_pool_uses_exact_4_25_bpw_payload() -> None:
    for projection, columns in (("gate_up", 4096), ("down", 2048)):
        one = routed_family_pool_bytes(projection, "MXFP4", 1)
        two = routed_family_pool_bytes(projection, "MXFP4", 2)
        expert_payload = 4096 * (columns // 2 + columns // 32)
        # The first cohort carries one MX payload header; subsequent experts
        # add only the original packed value and E8M0 byte streams.
        assert two - one == expert_payload + 4
        assert expert_payload * 8 / (4096 * columns) == 4.25


def test_v4f_88g_allocator_uses_budget_and_is_deterministic(tmp_path) -> None:
    reap = tmp_path / "reap.csv"
    _write_reap(reap)
    base = 43 * (
        routed_family_blob_bytes("gate_up", {"NVQ2J": 256})
        + routed_family_blob_bytes("down", {"NVQ2J": 256})
    )
    reserve = 4_000_000
    target = base + reserve + 1_000_000_000

    first = allocate_v4f_ew_nvq2j_nint4(
        [], reap, target_bytes=target, container_reserve_bytes=reserve
    )
    second = allocate_v4f_ew_nvq2j_nint4(
        [], reap, target_bytes=target, container_reserve_bytes=reserve
    )

    assert first == second
    assert first.nint4_count > 0
    assert first.estimated_blob_bytes + reserve <= target
    minimum_upgrade = min(
        routed_family_blob_bytes(
            projection, {"NVQ2J": 254, "NINT4": 2}
        )
        - routed_family_blob_bytes(
            projection, {"NVQ2J": 255, "NINT4": 1}
        )
        for projection in ("gate_up", "down")
    )
    assert target - first.estimated_blob_bytes < reserve + minimum_upgrade
    assert 0.0 <= first.gate_up_energy_fraction <= 1.0
    assert 0.0 <= first.down_energy_fraction <= 1.0
    assert (
        first.gate_up_energy_fraction + first.down_energy_fraction
    ) > 0.0


def test_v4f_incremental_training_tracks_exact_layer_shards(
    tmp_path,
) -> None:
    weight_map = {
        "layers.7.ffn.experts.0.w1.weight": "model-08.safetensors",
        "layers.7.ffn.experts.0.w1.scale": "model-08.safetensors",
        "layers.7.ffn.experts.0.w2.weight": "model-09.safetensors",
        "layers.7.attn_q.weight": "model-07.safetensors",
        "layers.8.ffn.experts.0.w1.weight": "model-10.safetensors",
    }
    shards = _layer_source_shards(weight_map, 7)
    assert shards == (
        "model-08.safetensors",
        "model-09.safetensors",
    )

    expected = {
        "model-08.safetensors": 3,
        "model-09.safetensors": 4,
    }
    (tmp_path / "model-08.safetensors").write_bytes(b"abc")
    assert not _source_shards_ready(tmp_path, shards, expected)
    (tmp_path / "model-09.safetensors").write_bytes(b"wxyz")
    assert _source_shards_ready(tmp_path, shards, expected)
    (tmp_path / "model-09.safetensors").write_bytes(b"short")
    assert not _source_shards_ready(tmp_path, shards, expected)


def test_v4f_sample_start_rejects_oversized_or_empty_ranges() -> None:
    with pytest.raises(ValueError, match="within"):
        V4FExpertSource._sample_start(
            seed=1,
            layer=0,
            expert=0,
            part="w1",
            rows=2049,
            total_rows=2048,
        )
    with pytest.raises(ValueError, match="source row"):
        V4FExpertSource._sample_start(
            seed=1,
            layer=0,
            expert=0,
            part="w1",
            rows=1,
            total_rows=0,
        )
