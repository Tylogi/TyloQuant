import numpy as np
import pytest

from mfq.calibration.artifact import ExpertPrecision
from mfq.formats.nepq import NEPQ0_A, NEPQ0_S, NEPQ1_A, NEPQ1_S, pack_nepq
from mfq.quantize.expert_nint import resolve_precision_artifact
from mfq.quantize.nepq import NepqQuantConfig
from mfq.quantize.nepq_a import (
    NepqAArtifact,
    NepqAQuantConfig,
    quantize_nepq_a_fixed,
)
from tests.test_formats.test_nepq import _tables


@pytest.mark.parametrize("spec", [NEPQ0_A, NEPQ1_A])
def test_quantize_nepq_a_fixed_builds_residual_streams_on_cpu(spec):
    rng = np.random.default_rng(20260809 + spec.profile_id)
    weight = rng.normal(0.0, 0.2, size=(1, 2, 104)).astype(np.float32)
    dictionary = rng.normal(0.0, 0.05, size=(1024, 8)).astype(np.float32)
    base_spec = NEPQ0_S if spec is NEPQ0_A else NEPQ1_S
    tensor = quantize_nepq_a_fixed(
        weight,
        spec,
        NepqAArtifact(_tables(base_spec), dictionary),
        rotation_block=8,
        second_records=0 if spec is NEPQ1_A else None,
        config=NepqAQuantConfig(
            base=NepqQuantConfig(
                anchor_multipliers=(1.0,),
                refine_steps=0,
                row_chunk=2,
                bank_chunk=2,
            ),
            residual_block_chunk=32,
        ),
        device="cpu",
    )
    assert tensor.spec is spec
    assert tensor.residual_codebook.dtype == np.float16
    assert tensor.residual_first.shape == (
        tensor.n_experts,
        tensor.out_per_expert,
        tensor.residual_blocks_per_row,
    )
    if spec is NEPQ1_A:
        assert np.count_nonzero(tensor.residual_second_mask) == 0
        assert tensor.residual_second_records.size == 0
    assert len(pack_nepq(tensor)) == 36 + tensor.payload_nbytes


def test_nepq1_a_target_nbytes_is_exact():
    rng = np.random.default_rng(20260815)
    weight = rng.normal(0.0, 0.2, size=(1, 2, 104)).astype(np.float32)
    artifact = NepqAArtifact(
        _tables(NEPQ1_S),
        rng.normal(0.0, 0.05, size=(1024, 8)).astype(np.float32),
    )
    config = NepqAQuantConfig(
        base=NepqQuantConfig(
            anchor_multipliers=(1.0,),
            refine_steps=0,
            row_chunk=2,
            bank_chunk=2,
        ),
        residual_block_chunk=32,
    )
    zero = quantize_nepq_a_fixed(
        weight,
        NEPQ1_A,
        artifact,
        rotation_block=8,
        second_records=0,
        config=config,
        device="cpu",
    )
    target = len(pack_nepq(zero)) + 5
    tensor = quantize_nepq_a_fixed(
        weight,
        NEPQ1_A,
        artifact,
        rotation_block=8,
        target_nbytes=target,
        config=config,
        device="cpu",
    )
    assert len(pack_nepq(tensor)) == target


def test_nepq_a_npz_artifact_loads_base_tables_and_residual_dictionary(tmp_path):
    dictionary = np.zeros((1024, 8), dtype=np.float32)
    path = tmp_path / "nepq1-a.npz"
    np.savez(
        path,
        table_payloads=_tables(NEPQ1_S),
        residual_codebook=dictionary,
    )
    artifact = resolve_precision_artifact(
        ExpertPrecision("NEPQ1-A", artifact=path.name),
        artifact_root=tmp_path,
    )
    assert isinstance(artifact, NepqAArtifact)
    np.testing.assert_array_equal(artifact.table_payloads, _tables(NEPQ1_S))
    np.testing.assert_array_equal(artifact.residual_codebook, dictionary)
