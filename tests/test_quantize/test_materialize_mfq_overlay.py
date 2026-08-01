from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import (
    _NINT_MOE_HDR,
    _NINT_MOE_POOL_V2_HDR,
    _pack_tensor,
    _unpack_tensor,
    open_mmap,
    save,
)
from mfq.formats.moe import NintMoePool, NintMoeTensor
from mfq.formats.nepq import NEPQ0_S, NepqTensor
from mfq.formats.nint import NintSpec
from mfq.formats.npq0_s import pack_npq0_s_tables
from mfq.quantize.nint_quant import dequantize as dequantize_nint
from mfq.quantize.nint_quant import quantize as quantize_nint
from mfq.quantize.nvq_jsc import dequantize_nvq_jsc
from mfq.tools.materialize_mfq_overlay import (
    build_materialization_plan,
    read_mfq_index,
    validate_materialized_mfq,
)
from tests.mixed_family_fixtures import make_flat_family


def _pack_delta(
    *,
    n_experts: int,
    out_per_expert: int,
    neuron_len: int,
    pools: list[tuple[list[int], object]],
) -> bytes:
    parts = [
        _NINT_MOE_HDR.pack(
            b"NID2",
            n_experts,
            out_per_expert,
            neuron_len,
            len(pools),
        )
    ]
    for expert_ids, tensor in pools:
        dtype, payload = _pack_tensor(tensor, allow_moe=False)
        dtype_bytes = dtype.encode("ascii")
        parts.extend(
            [
                _NINT_MOE_POOL_V2_HDR.pack(len(expert_ids), len(dtype_bytes), len(payload), 0),
                struct.pack(f"<{len(expert_ids)}i", *expert_ids),
                dtype_bytes,
                payload,
            ]
        )
    return b"".join(parts)


def _write_raw_mfq(
    path: Path,
    *,
    arch: str,
    records: list[tuple[str, str, bytes]],
    extra: dict | None = None,
) -> None:
    extra = {} if extra is None else extra
    with path.open("wb") as handle:
        handle.write(b"MFQ1")
        handle.write(struct.pack("<I", 2))
        arch_bytes = arch.encode()
        handle.write(struct.pack("<I", len(arch_bytes)))
        handle.write(arch_bytes)
        handle.write(struct.pack("<I", len(extra)))
        for key, value in extra.items():
            key_bytes = key.encode()
            value_bytes = __import__("json").dumps(value).encode()
            handle.write(struct.pack("<I", len(key_bytes)))
            handle.write(key_bytes)
            handle.write(struct.pack("<I", len(value_bytes)))
            handle.write(value_bytes)
        handle.write(struct.pack("<I", len(records)))
        for name, dtype, payload in records:
            name_bytes = name.encode()
            dtype_bytes = dtype.encode()
            handle.write(struct.pack("<I", len(name_bytes)))
            handle.write(name_bytes)
            handle.write(struct.pack("<I", len(dtype_bytes)))
            handle.write(dtype_bytes)
            handle.write(struct.pack("<Q", len(payload)))
        for _, _, payload in records:
            handle.write(payload)


def _expert_values(tensor: NintMoeTensor, expert: int) -> np.ndarray:
    rows = tensor.out_per_expert
    for pool in tensor.pools:
        matches = np.flatnonzero(np.asarray(pool.expert_ids) == expert)
        if not matches.size:
            continue
        local = int(matches[0])
        values = (
            dequantize_nint(pool.tensor)
            if hasattr(pool.tensor, "q")
            else dequantize_nvq_jsc(pool.tensor)
        )
        return values[local * rows : (local + 1) * rows]
    raise AssertionError(f"missing expert {expert}")


@pytest.mark.parametrize(
    "family",
    ("NVQ2J", "NVQ2J-L", "NVQ2J-XL", "NVQ3J-L"),
)
def test_materialize_overlay_slices_nint_and_jsc_without_requantizing(
    tmp_path: Path,
    family: str,
) -> None:
    rng = np.random.default_rng(20260725)
    n_experts = 4
    rows = 2
    width = 96
    nint = quantize_nint(
        rng.normal(size=(2 * rows, width)).astype(np.float32),
        NintSpec(4, 24, 6),
    )
    nvq = make_flat_family(family, rows=2 * rows, neuron_len=width)
    base_tensor = NintMoeTensor(
        (n_experts, rows, width),
        (
            NintMoePool(np.array([0, 1], dtype=np.int32), nint),
            NintMoePool(np.array([2, 3], dtype=np.int32), nvq),
        ),
    )
    base_path = tmp_path / "base.mfq"
    save(
        base_path,
        FileHeader(
            version=2,
            model_arch="test",
            num_tensors=2,
            extra={
                "allocation_sha256": "base-allocation",
                "source_index_sha256": "source",
            },
        ),
        {
            "dense": np.arange(8, dtype=np.float16).reshape(2, 4),
            "experts": base_tensor,
        },
    )

    replacement_nint = quantize_nint(
        rng.normal(size=(rows, width)).astype(np.float32),
        NintSpec(4, 24, 6),
    )
    replacement_nvq = make_flat_family(
        family,
        rows=rows,
        neuron_len=width,
        seed=20260726,
    )
    delta = _pack_delta(
        n_experts=n_experts,
        out_per_expert=rows,
        neuron_len=width,
        pools=[([1], replacement_nvq), ([2], replacement_nint)],
    )
    overlay_path = tmp_path / "overlay.mfq"
    _write_raw_mfq(
        overlay_path,
        arch="overlay",
        records=[("experts", "NINTMD", delta)],
        extra={
            "base_allocation_sha256": "base-allocation",
            "source_index_sha256": "source",
        },
    )

    plan = build_materialization_plan(base_path, overlay_path)
    output_path = tmp_path / "materialized.mfq"
    sources = {
        "base": base_path.open("rb"),
        "overlay": overlay_path.open("rb"),
    }
    try:
        from mfq.tools.materialize_mfq_overlay import _stream_plan

        with output_path.open("wb") as output:
            _stream_plan(
                plan,
                sources,
                output,
                start_offset=0,
                length=None,
                chunk_bytes=1024,
                progress_bytes=0,
            )
    finally:
        for handle in sources.values():
            handle.close()

    assert output_path.stat().st_size == plan.total_bytes
    assert read_mfq_index(output_path).records[1].dtype == "NINTM"
    resumed_path = tmp_path / "resumed.mfq"
    split = 137
    with (
        resumed_path.open("wb") as output,
        base_path.open("rb") as base_handle,
        overlay_path.open("rb") as overlay_handle,
    ):
        _stream_plan(
            plan,
            {"base": base_handle, "overlay": overlay_handle},
            output,
            start_offset=0,
            length=split,
            chunk_bytes=31,
            progress_bytes=0,
        )
    with (
        resumed_path.open("ab") as output,
        base_path.open("rb") as base_handle,
        overlay_path.open("rb") as overlay_handle,
    ):
        _stream_plan(
            plan,
            {"base": base_handle, "overlay": overlay_handle},
            output,
            start_offset=split,
            length=None,
            chunk_bytes=31,
            progress_bytes=0,
        )
    assert resumed_path.read_bytes() == output_path.read_bytes()
    validation = validate_materialized_mfq(
        output_path,
        expected_bytes=plan.total_bytes,
        expected_family_expert_counts={"NINT4": 2, family: 2},
    )
    assert validation["status"] == "passed"
    assert validation["moe_records"] == 1
    with open_mmap(base_path) as base_store, open_mmap(output_path) as merged_store:
        original = base_store["experts"]
        merged = merged_store["experts"]
        assert isinstance(original, NintMoeTensor)
        assert isinstance(merged, NintMoeTensor)
        assert merged.expert_profiles == (
            "NINT4-24",
            family,
            "NINT4-24",
            family,
        )
        np.testing.assert_array_equal(_expert_values(merged, 0), _expert_values(original, 0))
        np.testing.assert_array_equal(_expert_values(merged, 3), _expert_values(original, 3))
        np.testing.assert_array_equal(
            _expert_values(merged, 1),
            dequantize_nvq_jsc(
                _unpack_tensor(
                    family,
                    _pack_tensor(replacement_nvq, allow_moe=False)[1],
                )
            ),
        )
        np.testing.assert_array_equal(
            _expert_values(merged, 2),
            dequantize_nint(
                _unpack_tensor(
                    "NINT4",
                    _pack_tensor(replacement_nint, allow_moe=False)[1],
                )
            ),
        )


def test_materialize_overlay_slices_nepq_without_requantizing(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260726)
    n_experts = 3
    rows = 2
    width = 96
    groups = 4
    vectors = 12
    supergroups = 1
    table_source = make_flat_family("NPQ0-S", rows=rows, neuron_len=width)
    table_payload = pack_npq0_s_tables(
        table_source.scale_lut,
        table_source.first_codebooks,
        table_source.second_codebooks,
    )
    nepq = NepqTensor(
        spec=NEPQ0_S,
        shape=(2, rows, width),
        neuron_scale=rng.uniform(0.005, 0.05, (2, rows)).astype(np.float32),
        state=rng.integers(0, 4, (2, rows, groups), dtype=np.uint8),
        indices=rng.integers(0, 64, (2, rows, vectors), dtype=np.uint8),
        aux=None,
        bank_ids=np.zeros((2, rows, supergroups), dtype=np.uint8),
        table_payloads=np.frombuffer(table_payload, dtype=np.uint8).copy()[None, :],
    )
    nint = quantize_nint(
        rng.normal(size=(rows, width)).astype(np.float32),
        NintSpec(4, 24, 6),
    )
    base_tensor = NintMoeTensor(
        (n_experts, rows, width),
        (
            NintMoePool(np.array([0, 1], dtype=np.int32), nepq),
            NintMoePool(np.array([2], dtype=np.int32), nint),
        ),
    )
    base_path = tmp_path / "base-nepq.mfq"
    save(
        base_path,
        FileHeader(version=2, model_arch="test", num_tensors=1),
        {"experts": base_tensor},
    )

    replacement = quantize_nint(
        rng.normal(size=(rows, width)).astype(np.float32),
        NintSpec(4, 24, 6),
    )
    overlay_path = tmp_path / "overlay-nepq.mfq"
    _write_raw_mfq(
        overlay_path,
        arch="overlay",
        records=[
            (
                "experts",
                "NINTMD",
                _pack_delta(
                    n_experts=n_experts,
                    out_per_expert=rows,
                    neuron_len=width,
                    pools=[([1], replacement)],
                ),
            )
        ],
    )

    plan = build_materialization_plan(base_path, overlay_path)
    output_path = tmp_path / "materialized-nepq.mfq"
    from mfq.tools.materialize_mfq_overlay import _stream_plan

    with (
        output_path.open("wb") as output,
        base_path.open("rb") as base_handle,
        overlay_path.open("rb") as overlay_handle,
    ):
        _stream_plan(
            plan,
            {"base": base_handle, "overlay": overlay_handle},
            output,
            start_offset=0,
            length=None,
            chunk_bytes=1024,
            progress_bytes=0,
        )
    validate_materialized_mfq(
        output_path,
        expected_bytes=plan.total_bytes,
        expected_family_expert_counts={"NEPQ0-S": 1, "NINT4": 2},
    )
    with open_mmap(base_path) as base_store, open_mmap(output_path) as merged_store:
        original = base_store["experts"]
        merged = merged_store["experts"]
        assert isinstance(original, NintMoeTensor)
        assert isinstance(merged, NintMoeTensor)
        assert merged.expert_profiles == ("NEPQ0-S", "NINT4-24", "NINT4-24")
        original_nepq = original.pools[0].tensor
        merged_nepq = merged.pools[0].tensor
        assert isinstance(original_nepq, NepqTensor)
        assert isinstance(merged_nepq, NepqTensor)
        np.testing.assert_array_equal(
            merged_nepq.table_payloads, original_nepq.table_payloads
        )
        np.testing.assert_array_equal(
            merged_nepq.neuron_scale[0], original_nepq.neuron_scale[0]
        )
        np.testing.assert_array_equal(merged_nepq.state[0], original_nepq.state[0])
        np.testing.assert_array_equal(merged_nepq.indices[0], original_nepq.indices[0])
        np.testing.assert_array_equal(merged_nepq.bank_ids[0], original_nepq.bank_ids[0])
