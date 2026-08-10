from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from mfq.calibration.artifact import ExpertPrecision
from mfq.calibration.artifact import load_scheme
from mfq.calibration.tpq import (
    allocate_tpq_tiers,
    build_tpq_expert_selection,
    load_tpq_tier_profile,
)
from mfq.formats.tpq import (
    TPQ_V,
    TPQ_VV,
    TPQ_W,
    TPQ_X,
    TpqPqTensor,
    TpqPqSpec,
    pack_tpq_indices,
    pack_tpq_int4,
    pack_tpq_pq,
    unpack_tpq_indices,
    unpack_tpq_int4,
    unpack_tpq_pq,
)
from mfq.formats.io import pack_nint_moe, unpack_nint_moe
from mfq.quantize.tpq import (
    TpqKmeansConfig,
    assign_tpq_codebook,
    tpq_reconstruction_sums,
    dequantize_tpq_int4,
    dequantize_tpq_pq,
    quantize_tpq_int4,
    quantize_tpq_pq_fixed,
    train_tpq_codebook,
    train_tpq_expert_codebook,
    train_tpq_pq,
)
from mfq.quantize.expert_nint import (
    dequantize_expertwise,
    quantize_expertwise,
)
from mfq.tools.quantize_hf_to_mfq import _write_flat_family_axis0_blob
from mfq.tools.prepare_tpq_scheme import prepare
from mfq.tools.train_tpq_v4f_codebooks import train as train_tpq_v4f


@pytest.mark.parametrize(
    ("entries", "expected_bits", "dtype"),
    ((80, 8, np.uint8), (4096, 16, np.uint16)),
)
def test_tpq_indices_follow_source_storage_dtype(
    entries: int,
    expected_bits: int,
    dtype,
) -> None:
    spec = TpqPqSpec("w", 4, entries)
    values = np.asarray([0, 1, entries - 1], dtype=dtype)
    payload = pack_tpq_indices(values, spec.index_bits)
    restored, offset = unpack_tpq_indices(
        payload,
        0,
        values.size,
        spec.index_bits,
    )
    assert spec.index_bits == expected_bits
    assert len(payload) == values.size * np.dtype(dtype).itemsize
    assert offset == len(payload)
    np.testing.assert_array_equal(restored, values)


@pytest.mark.parametrize("bits", range(8, 17))
def test_tpq_projection_indices_roundtrip_every_packed_width(bits: int) -> None:
    entries = 1 << min(bits, 9)
    values = np.arange(32, dtype=np.uint16).reshape(4, 8) % entries
    payload = pack_tpq_indices(values, bits)
    restored, offset = unpack_tpq_indices(payload, 0, values.size, bits)
    assert offset == len(payload) == (values.size * bits + 7) // 8
    np.testing.assert_array_equal(restored.reshape(values.shape), values)

    spec = TpqPqSpec("p", 2, entries, bits)
    tensor = TpqPqTensor(
        spec=spec,
        shape=(4, 16),
        axis=0,
        neuron_len=16,
        indices=values,
        codebook=np.arange(entries * 2, dtype=np.float32).reshape(entries, 2),
    )
    np.testing.assert_array_equal(
        dequantize_tpq_pq(tensor),
        tensor.codebook[values].reshape(tensor.shape),
    )
    decoded = unpack_tpq_pq(pack_tpq_pq(tensor))
    assert decoded.spec == spec
    decoded_payload = (
        decoded.indices.tobytes()
        if bits not in {8, 16}
        else pack_tpq_indices(decoded.indices, bits)
    )
    decoded_values, _ = unpack_tpq_indices(
        decoded_payload,
        0,
        values.size,
        bits,
    )
    np.testing.assert_array_equal(decoded_values.reshape(values.shape), values)
    np.testing.assert_array_equal(
        dequantize_tpq_pq(decoded),
        tensor.codebook[values].reshape(tensor.shape),
    )


def test_tpq_projection_packed_indices_reject_missing_codeword() -> None:
    spec = TpqPqSpec("p", 2, 300, 12)
    packed = np.full((4 * 8 * 12 + 7) // 8, 0xFF, dtype=np.uint8)
    with pytest.raises(ValueError, match="missing codeword"):
        TpqPqTensor(
            spec=spec,
            shape=(4, 16),
            axis=0,
            neuron_len=16,
            indices=packed,
            codebook=np.zeros((300, 2), dtype=np.float32),
        )


def test_tpq_pq_roundtrip_all_tiers() -> None:
    rng = np.random.default_rng(20260726)
    for spec in (TPQ_X, TPQ_W, TPQ_V, TPQ_VV):
        weight = rng.normal(size=(3, 24)).astype(np.float32)
        codebook = rng.normal(size=(spec.codebook_entries, spec.vector_size)).astype(np.float32)
        tensor = quantize_tpq_pq_fixed(
            weight,
            spec,
            codebook,
            device="cpu",
            distance_bytes=1 << 20,
        )
        payload = pack_tpq_pq(tensor)
        restored = unpack_tpq_pq(payload)
        assert len(payload) == tensor.payload_nbytes
        assert restored.spec == spec
        np.testing.assert_array_equal(restored.indices, tensor.indices)
        np.testing.assert_array_equal(restored.codebook, tensor.codebook)
        np.testing.assert_array_equal(
            dequantize_tpq_pq(restored),
            dequantize_tpq_pq(tensor),
        )


def test_tpq_pq_roundtrip_historical_k80_tier() -> None:
    rng = np.random.default_rng(80)
    spec = TpqPqSpec("w", 4, 80)
    weight = rng.normal(size=(3, 24)).astype(np.float32)
    codebook = rng.normal(size=(80, 4)).astype(np.float32)
    tensor = quantize_tpq_pq_fixed(weight, spec, codebook, device="cpu")
    restored = unpack_tpq_pq(pack_tpq_pq(tensor))
    assert restored.spec == spec
    np.testing.assert_array_equal(restored.indices, tensor.indices)


def test_tpq_int4_matches_reference_equations() -> None:
    rng = np.random.default_rng(19)
    weight = rng.normal(size=(7, 128)).astype(np.float32)
    tensor = quantize_tpq_int4(weight)
    restored = unpack_tpq_int4(pack_tpq_int4(tensor))
    np.testing.assert_array_equal(restored.packed, tensor.packed)
    np.testing.assert_array_equal(restored.scales, tensor.scales)
    np.testing.assert_array_equal(
        dequantize_tpq_int4(restored),
        dequantize_tpq_int4(tensor),
    )


def test_tpq_int4_accepts_zero_scale_from_fp16_underflow() -> None:
    weight = np.zeros((1, 64), dtype=np.float32)
    tensor = quantize_tpq_int4(weight)
    assert tensor.scales.tolist() == [[0.0]]
    restored = unpack_tpq_int4(pack_tpq_int4(tensor))
    np.testing.assert_array_equal(
        dequantize_tpq_int4(restored),
        weight,
    )


def test_tpq_assignment_uses_euclidean_objective() -> None:
    point = np.asarray([[4.0, 6.0]], dtype=np.float32)
    codebook = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    assigned = assign_tpq_codebook(point, codebook, device="cpu")
    assert assigned.tolist() == [0]


def test_tpq_production_apis_cannot_accept_imatrix_weights() -> None:
    for function in (
        assign_tpq_codebook,
        quantize_tpq_pq_fixed,
        train_tpq_codebook,
        train_tpq_expert_codebook,
        train_tpq_pq,
        train_tpq_v4f,
    ):
        parameters = inspect.signature(function).parameters
        assert "importance" not in parameters
        assert "input_importance" not in parameters
        assert "imatrix_path" not in parameters


def test_tpq_kmeans_trains_exact_point_codebook() -> None:
    rng = np.random.default_rng(5)
    points = rng.normal(size=(256, 8)).astype(np.float32)
    result = train_tpq_codebook(
        points,
        TPQ_X,
        config=TpqKmeansConfig(
            iterations=1,
            restarts=1,
            sample_points=256,
            seed=11,
            distance_bytes=1 << 20,
        ),
        device="cpu",
    )
    assert result.codebook.shape == (256, 8)
    assert result.sse < 1e-4


def test_tpq_audit_sums_match_materialized_reconstruction() -> None:
    rng = np.random.default_rng(44)
    weight = rng.normal(size=(5, 24)).astype(np.float32)
    codebook = rng.normal(size=(256, 8)).astype(np.float32)
    tensor = quantize_tpq_pq_fixed(
        weight,
        TPQ_X,
        codebook,
        device="cpu",
    )
    reconstruction = dequantize_tpq_pq(tensor)
    sse, signal = tpq_reconstruction_sums(
        weight,
        TPQ_X,
        codebook,
        device="cpu",
    )
    assert sse == pytest.approx(float(np.square(weight - reconstruction).sum()))
    assert signal == pytest.approx(float(np.square(weight).sum()))


def test_tpq_tiers_and_nintm_roundtrip() -> None:
    allocation = allocate_tpq_tiers(
        [70.0, 26.6, 3.2, 0.2],
        vv_share=0.5,
    )
    assert allocation.tiers == ("vv", "v", "w", "x")
    selection = build_tpq_expert_selection(
        name="blk.0.ffn_gate_up_exps.weight",
        group="gate_up",
        allocation=allocation,
        rows_per_expert=2,
        columns=24,
        artifacts={
            "vv": "vv.npz",
            "v": "v.npz",
            "w": "w.npz",
            "x": "x.npz",
        },
    )
    assert tuple(item.descriptor.family for item in selection.selections) == (
        "TPQ-VV",
        "TPQ-V",
        "TPQ-W",
        "TPQ-X",
    )

    rng = np.random.default_rng(81)
    weight = rng.normal(size=(4, 2, 24)).astype(np.float32)
    codebooks = {
        family: rng.normal(size=(spec.codebook_entries, spec.vector_size)).astype(np.float32)
        for family, spec in {
            "TPQ-X": TPQ_X,
            "TPQ-W": TPQ_W,
            "TPQ-V": TPQ_V,
            "TPQ-VV": TPQ_VV,
        }.items()
    }
    tensor = quantize_expertwise(
        weight,
        selection.precisions,
        artifacts=codebooks,
        device="cpu",
    )
    with_generic_importance = quantize_expertwise(
        weight,
        selection.precisions,
        artifacts=codebooks,
        importance=rng.uniform(0.1, 10.0, size=weight.shape).astype(np.float32),
        device="cpu",
    )
    assert with_generic_importance.expert_profiles == tensor.expert_profiles
    for plain_pool, calibrated_pool in zip(
        tensor.pools,
        with_generic_importance.pools,
    ):
        np.testing.assert_array_equal(
            plain_pool.tensor.indices,
            calibrated_pool.tensor.indices,
        )
    restored = unpack_nint_moe(pack_nint_moe(tensor))
    assert restored.expert_profiles == tensor.expert_profiles
    np.testing.assert_array_equal(
        dequantize_expertwise(restored),
        dequantize_expertwise(tensor),
    )


def test_tpq_streaming_writer_uses_native_payload(tmp_path: Path) -> None:
    rng = np.random.default_rng(92)
    weight = rng.normal(size=(6, 24)).astype(np.float32)
    codebook = rng.normal(size=(256, 8)).astype(np.float32)
    artifact = tmp_path / "tpq-x.npz"
    np.savez(artifact, family=np.asarray("TPQ-X"), codebook=codebook)
    precision = ExpertPrecision(
        family="TPQ-X",
        artifact=artifact.name,
        options=(("distance_bytes", 1 << 20),),
    )
    output = tmp_path / "tpq-x.blob"
    written = _write_flat_family_axis0_blob(
        weight,
        weight.shape,
        precision,
        output,
        row_chunk=2,
        quant_backend="cpu",
        device="cpu",
        artifact_root=tmp_path,
    )
    restored = unpack_tpq_pq(output.read_bytes())
    assert written == output.stat().st_size == restored.payload_nbytes
    expected = quantize_tpq_pq_fixed(
        weight,
        TPQ_X,
        codebook,
        device="cpu",
    )
    np.testing.assert_array_equal(restored.indices, expected.indices)


def test_prepare_tpq_scheme_is_consumable(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        '{"counts":{"0":{"0":70.0,"1":26.6,"2":3.2,"3":0.2}}}',
        encoding="utf-8",
    )
    output = tmp_path / "scheme.json"
    prepared = prepare(
        profile_path=profile,
        output_path=output,
        rows_gate_up=4,
        columns_gate_up=24,
        rows_down=3,
        columns_down=24,
        vv_share=0.5,
    )
    restored = load_scheme(output)
    assert restored.target_storage_bits == prepared.storage_bits
    assert set(restored.expert_selections) == {
        "blk.0.ffn_gate_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    }
    assert restored.require_expert("blk.0.ffn_gate_up_exps.weight").precisions[0].family == "TPQ-VV"


def test_prepare_tpq_scheme_accepts_fixed_tier_fragment(tmp_path: Path) -> None:
    profile = tmp_path / "tiers.json"
    profile.write_text(
        '"tiers_per_layer":{"0":"Vvwx"},',
        encoding="utf-8",
    )
    assert load_tpq_tier_profile(profile) == {0: ("vv", "v", "w", "x")}
    output = tmp_path / "scheme.json"
    prepare(
        profile_path=profile,
        output_path=output,
        rows_gate_up=4,
        columns_gate_up=24,
        rows_down=3,
        columns_down=24,
    )
    restored = load_scheme(output)
    assert restored.metadata["tier_source"] == "fixed_tiers_per_layer"
    assert tuple(
        precision.family
        for precision in restored.require_expert("blk.0.ffn_gate_up_exps.weight").precisions
    ) == ("TPQ-VV", "TPQ-V", "TPQ-W", "TPQ-X")
