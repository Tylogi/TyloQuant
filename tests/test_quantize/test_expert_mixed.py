from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertPrecision,
    ExpertSelection,
    ExpertTensorSelection,
    load_scheme,
    save_scheme,
)
from mfq.formats import io
from mfq.formats.moe import expert_tensor_family
from mfq.formats.mx import MxTensor, pack_mx
from mfq.formats.nepq import NEPQ0_L, NEPQ0_S, NEPQ1_L, NEPQ1_S
from mfq.formats.nint import NintSpec
from mfq.quantize.expert_nint import (
    dequantize_expertwise,
    quantize_expertwise,
    quantize_flat_cohort,
)
from mfq.quantize.nint_quant import quantize as quantize_nint
from mfq.quantize.npq0_l import Npq0LTables
from mfq.quantize.npq0_s import Npq0STables
from mfq.quantize.nvq_jsc import NvqJscTables
from mfq.tools.quantize_hf_to_mfq import (
    _ExpertPoolRowSource,
    _mixed_moe_blob_nbytes,
    _write_flat_family_axis0_blob,
    _write_mixed_moe_axis0_blob,
)
from tests.mixed_family_fixtures import FLAT_FAMILIES, make_flat_family
from tests.test_formats.test_nepq import _tensor as make_nepq


def _artifact_for_family(tmp_path, family: str) -> ExpertPrecision:
    tensor = make_flat_family(family, rows=6, neuron_len=96)
    path = tmp_path / f"{family}.npz"
    if family == "NPQ0-L":
        np.savez(
            path,
            scale_lut=tensor.scale_lut,
            first_codebooks=tensor.first_codebooks,
            second_codebooks=tensor.second_codebooks,
        )
    elif family == "NPQ0-S":
        np.savez(
            path,
            scale_lut=tensor.scale_lut,
            first_codebooks=tensor.first_codebooks,
            second_codebooks=tensor.second_codebooks,
        )
    elif family in {
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3J",
        "NVQ3J-L",
    }:
        np.savez(
            path,
            scale_lut=tensor.scale_lut,
            bank_for_state=tensor.bank_for_state,
            codebooks=tensor.codebooks,
        )
    else:
        return ExpertPrecision(family)
    return ExpertPrecision(family, artifact=path.name)


def test_mixed_expert_precision_scheme_roundtrip(tmp_path):
    name = "blk.0.ffn_down_exps.weight"
    precisions = (
        ExpertPrecision("NINT4", nint_spec=NintSpec(4, 24, 6)),
        ExpertPrecision("NVQ2J", artifact="tables/down-nvq2j.npz"),
        ExpertPrecision(
            "NEPQ0-S",
            artifact="tables/down-nepq0s.npz",
            options=(("rotation_block", 2048), ("rotation_seed", 17)),
        ),
    )
    selection = ExpertTensorSelection(
        name=name,
        group="blk.0.expert_down",
        n_experts=3,
        rows_per_expert=2,
        columns=96,
        selections=tuple(
            ExpertSelection(
                expert_id=index,
                spec=precision.nint_spec,
                precision=precision,
                storage_bits=1000 + index,
                train_loss=0.1 + index,
                validation_loss=0.2 + index,
            )
            for index, precision in enumerate(precisions)
        ),
    )
    scheme = CalibrationScheme(
        path=None,
        target_profile="mixed",
        target_storage_bits=selection.storage_bits,
        selections={},
        expert_selections={name: selection},
        metadata={},
        candidate_table={},
    )
    path = tmp_path / "scheme.json"
    save_scheme(path, scheme)
    restored = load_scheme(path)
    assert restored.require_expert(name).precisions == precisions
    assert '"format": "mfq.calibration-scheme.v3"' in path.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("family", FLAT_FAMILIES)
def test_streaming_flat_family_blob_roundtrip(tmp_path, family: str):
    rng = np.random.default_rng(20260723)
    shape = (2, 3, 96)
    weight = rng.normal(0, 0.05, shape).astype(np.float32)
    precision = _artifact_for_family(tmp_path, family)
    source = _ExpertPoolRowSource(weight, shape, shape, (0, 1))
    path = tmp_path / f"{family}.blob"
    nbytes = _write_flat_family_axis0_blob(
        source,
        (6, 96),
        precision,
        path,
        row_chunk=8,
        quant_backend="cpu",
        device="cpu",
        artifact_root=tmp_path,
    )
    tensor = io._unpack_tensor(family, path.read_bytes())
    assert nbytes == path.stat().st_size
    assert expert_tensor_family(tensor) == family
    assert tensor.shape == (6, 96)


def test_quantize_expertwise_mixes_flat_and_cross_expert_families():
    rng = np.random.default_rng(20260723)
    weight = rng.normal(0, 0.04, (4, 3, 104)).astype(np.float32)
    npq = make_flat_family("NPQ0-S", rows=3, neuron_len=104)
    jsc = make_flat_family("NVQ2J", rows=3, neuron_len=104)
    nepq = make_nepq(NEPQ0_S)
    profiles = (
        ExpertPrecision("NINT4", nint_spec=NintSpec(4, 24, 6)),
        ExpertPrecision("NPQ0-S"),
        ExpertPrecision("NVQ2J"),
        ExpertPrecision(
            "NEPQ0-S",
            options=(("rotation_block", 8), ("rotation_seed", 18601311049)),
        ),
    )
    artifacts = {
        profiles[1]: Npq0STables(
            npq.scale_lut, npq.first_codebooks, npq.second_codebooks
        ),
        profiles[2]: NvqJscTables(
            jsc.scale_lut,
            jsc.bank_for_state,
            jsc.codebooks,
            jsc.spec,
        ),
        profiles[3]: nepq.table_payloads,
    }
    tensor = quantize_expertwise(
        weight,
        profiles,
        artifacts=artifacts,
        device="cpu",
    )
    restored = dequantize_expertwise(tensor)
    assert tensor.expert_profiles == (
        "NINT4-24",
        "NPQ0-S",
        "NVQ2J",
        "NEPQ0-S",
    )
    assert restored.shape == weight.shape
    assert np.isfinite(restored).all()


def test_mixed_moe_writer_size_and_roundtrip(tmp_path):
    rng = np.random.default_rng(20260723)
    shape = (4, 3, 104)
    weight = rng.normal(0, 0.04, shape).astype(np.float32)
    npq = make_flat_family("NPQ0-S", rows=3, neuron_len=104)
    jsc = make_flat_family("NVQ2J", rows=3, neuron_len=104)
    nepq = make_nepq(NEPQ0_S)
    np.savez(
        tmp_path / "npq.npz",
        scale_lut=npq.scale_lut,
        first_codebooks=npq.first_codebooks,
        second_codebooks=npq.second_codebooks,
    )
    np.savez(
        tmp_path / "jsc.npz",
        scale_lut=jsc.scale_lut,
        bank_for_state=jsc.bank_for_state,
        codebooks=jsc.codebooks,
    )
    np.savez(tmp_path / "nepq.npz", table_payloads=nepq.table_payloads)
    precisions = (
        ExpertPrecision("NINT4", nint_spec=NintSpec(4, 24, 6)),
        ExpertPrecision("NPQ0-S", artifact="npq.npz"),
        ExpertPrecision("NVQ2J", artifact="jsc.npz"),
        ExpertPrecision(
            "NEPQ0-S",
            artifact="nepq.npz",
            options=(("rotation_block", 8), ("rotation_seed", 18601311049)),
        ),
    )
    path = tmp_path / "mixed.blob"
    nbytes = _write_mixed_moe_axis0_blob(
        weight,
        shape,
        shape,
        precisions,
        path,
        row_chunk=8,
        quant_backend="cpu",
        device="cpu",
        artifact_root=tmp_path,
    )
    assert nbytes == _mixed_moe_blob_nbytes(shape, precisions, tmp_path)
    tensor = io.unpack_nint_moe(path.read_bytes())
    assert tensor.expert_profiles == (
        "NINT4-24",
        "NPQ0-S",
        "NVQ2J",
        "NEPQ0-S",
    )
    assert io.pack_nint_moe(tensor) == path.read_bytes()


def test_mixed_moe_preserves_native_mxfp4_bytes(tmp_path):
    shape = (2, 2, 32)
    values = np.arange(2 * 2 * 16, dtype=np.uint8).reshape(2, 2, 16)
    scales = np.arange(2 * 2, dtype=np.uint8).reshape(2, 2, 1)

    class ExactMxSource:
        def write_mxfp4_expert_pool(self, expert_ids, path):
            ids = np.asarray(expert_ids, dtype=np.int64)
            tensor = MxTensor(
                "MXFP4",
                (len(ids) * 2, 32),
                np.ascontiguousarray(values[ids].reshape(-1, 16)),
                np.ascontiguousarray(scales[ids].reshape(-1, 1)),
            )
            payload = pack_mx(tensor)
            path.write_bytes(payload)
            return len(payload)

    precisions = (ExpertPrecision("MXFP4"),) * 2
    path = tmp_path / "mixed-mxfp4.blob"
    nbytes = _write_mixed_moe_axis0_blob(
        ExactMxSource(),
        shape,
        shape,
        precisions,
        path,
        row_chunk=2,
        quant_backend="cpu",
        device="cpu",
        artifact_root=tmp_path,
    )
    assert nbytes == _mixed_moe_blob_nbytes(shape, precisions, tmp_path)
    restored = io.unpack_nint_moe(path.read_bytes())
    assert restored.expert_profiles == ("MXFP4", "MXFP4")
    pool = restored.pools[0].tensor
    assert isinstance(pool, MxTensor)
    np.testing.assert_array_equal(pool.values, values.reshape(-1, 16))
    np.testing.assert_array_equal(pool.scales, scales.reshape(-1, 1))


@pytest.mark.parametrize(
    ("quant_backend", "device"),
    (("cpu", "cpu"), ("cuda", "cuda")),
)
def test_streaming_writer_builds_all_precision_families(
    tmp_path,
    quant_backend: str,
    device: str,
):
    if device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(20260723)
    rows = 3
    neuron_len = 104
    precisions = [
        ExpertPrecision("NINT2", nint_spec=NintSpec(2, 16, 5)),
        ExpertPrecision("NINT4", nint_spec=NintSpec(4, 24, 6)),
        ExpertPrecision("NINT5", nint_spec=NintSpec(5, 28, 7)),
        ExpertPrecision("NINT6", nint_spec=NintSpec(6, 24, 7)),
        ExpertPrecision("NINT8", nint_spec=NintSpec(8, 48, 7)),
    ]
    for family in FLAT_FAMILIES:
        precisions.append(_artifact_for_family(tmp_path, family))
    for spec in (NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L):
        table_path = tmp_path / f"{spec.label}.npz"
        np.savez(table_path, table_payloads=make_nepq(spec).table_payloads)
        precisions.append(
            ExpertPrecision(
                spec.label,
                artifact=table_path.name,
                options=(("rotation_block", 8), ("rotation_seed", 18601311049)),
            )
        )

    expert_precisions = tuple(precisions)
    shape = (len(expert_precisions), rows, neuron_len)
    weight = rng.normal(0, 0.04, shape).astype(np.float32)
    path = tmp_path / "all-families.blob"
    context = (
        pytest.warns(RuntimeWarning, match="NVQ1-S has no CUDA")
        if device == "cuda"
        else nullcontext()
    )
    with context:
        nbytes = _write_mixed_moe_axis0_blob(
            weight,
            shape,
            shape,
            expert_precisions,
            path,
            row_chunk=8,
            quant_backend=quant_backend,
            device=device,
            artifact_root=tmp_path,
        )
    restored = io.unpack_nint_moe(path.read_bytes())
    assert nbytes == path.stat().st_size
    assert restored.expert_profiles == (
        "NINT2-16",
        "NINT4-24",
        "NINT5-28",
        "NINT6-24",
        "NINT8-48",
        *FLAT_FAMILIES,
        "NEPQ0-S",
        "NEPQ0-L",
        "NEPQ1-S",
        "NEPQ1-L",
    )
    assert io.pack_nint_moe(restored) == path.read_bytes()


def test_mixed_nint_imatrix_changes_nint4_and_leaves_nint8_unchanged(tmp_path):
    rng = np.random.default_rng(20260725)
    rows = 5
    columns = 113
    shape = (2, rows, columns)
    weight = rng.normal(0, 0.08, size=shape).astype(np.float32)
    importance = np.stack(
        (
            np.geomspace(0.001, 1000.0, columns),
            np.geomspace(1000.0, 0.001, columns),
        )
    ).astype(np.float32)
    precisions = (
        ExpertPrecision("NINT4", nint_spec=NintSpec(4, 24, 6)),
        ExpertPrecision("NINT8", nint_spec=NintSpec(8, 48, 7)),
    )
    plain_path = tmp_path / "plain.blob"
    weighted_path = tmp_path / "weighted.blob"

    _write_mixed_moe_axis0_blob(
        weight,
        shape,
        shape,
        precisions,
        plain_path,
        row_chunk=8,
        quant_backend="cpu",
        device="cpu",
        artifact_root=tmp_path,
    )
    _write_mixed_moe_axis0_blob(
        weight,
        shape,
        shape,
        precisions,
        weighted_path,
        row_chunk=8,
        quant_backend="cpu",
        device="cpu",
        artifact_root=tmp_path,
        importance=importance,
    )
    plain = io.unpack_nint_moe(plain_path.read_bytes())
    weighted = io.unpack_nint_moe(weighted_path.read_bytes())

    assert io.pack_nint(plain.pools[0].tensor) != io.pack_nint(
        weighted.pools[0].tensor
    )
    assert io.pack_nint(plain.pools[1].tensor) == io.pack_nint(
        weighted.pools[1].tensor
    )


def test_flat_nint_cohort_forwards_imatrix_to_nint_solver():
    rng = np.random.default_rng(20260802)
    rows = rng.normal(0, 0.08, size=(7, 113)).astype(np.float32)
    importance = np.geomspace(0.001, 1000.0, rows.shape[1]).astype(
        np.float32
    )
    precision = ExpertPrecision(
        "NINT4", nint_spec=NintSpec(4, 24, 6)
    )

    cohort = quantize_flat_cohort(
        rows,
        precision,
        importance=importance,
        device="cpu",
    )
    direct = quantize_nint(
        rows,
        precision.nint_spec,
        axis=0,
        importance=importance,
    )

    for field in (
        "q",
        "neuron_scale",
        "neuron_min",
        "sub_scale",
        "sub_min",
    ):
        np.testing.assert_array_equal(
            getattr(cohort, field), getattr(direct, field)
        )


def test_flat_nint8_cohort_does_not_consume_imatrix():
    rng = np.random.default_rng(20260803)
    rows = rng.normal(0, 0.08, size=(7, 113)).astype(np.float32)
    importance = np.geomspace(0.001, 1000.0, rows.shape[1]).astype(
        np.float32
    )
    precision = ExpertPrecision(
        "NINT8", nint_spec=NintSpec(8, 48, 7)
    )
    plain = quantize_flat_cohort(rows, precision, device="cpu")
    weighted = quantize_flat_cohort(
        rows,
        precision,
        importance=importance,
        device="cpu",
    )
    for field in (
        "q",
        "neuron_scale",
        "neuron_min",
        "sub_scale",
        "sub_min",
    ):
        np.testing.assert_array_equal(
            getattr(plain, field), getattr(weighted, field)
        )
