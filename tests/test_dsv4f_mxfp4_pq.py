import torch

from bench.dsv4f_mxfp4_pq import (
    mxfp4_pq_rate,
    project_mxfp4_codebook,
    quantize_mxfp4_pq,
)
from mfq.formats.nvq import NVQ2_E8


def test_exact_rate_point_is_below_nvq2() -> None:
    rate = mxfp4_pq_rate(4096, 4096)
    budget = NVQ2_E8.payload_nbytes(4096, 4096)
    assert rate.entries == 16384
    assert rate.codebook_nbytes == 81_920
    assert rate.index_nbytes == 3_670_016
    assert rate.anchor_nbytes == 8_192
    assert rate.scale_nbytes == 525_312
    assert rate.payload_nbytes == 4_285_440
    assert budget - rate.payload_nbytes == 5_120
    assert rate.payload_bpw == 2.04345703125

    larger = mxfp4_pq_rate(
        4096,
        4096,
        entries=32768,
        index_bits=15,
        scale_bits=2,
    )
    assert larger.payload_nbytes == 4_279_296
    assert budget - larger.payload_nbytes == 11_264
    assert larger.payload_bpw == 2.04052734375


def test_eight_dimensional_mxfp4_codebook_projection() -> None:
    source = torch.tensor(
        [[0.0, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0]],
        dtype=torch.float32,
    )
    projected, packed, scales = project_mxfp4_codebook(
        source, return_storage=True
    )
    assert packed is not None and scales is not None
    assert packed.shape == (1, 4)
    assert scales.shape == (1, 1)
    torch.testing.assert_close(projected, source, rtol=0, atol=0)


def test_small_learned_mxfp4_pq_smoke() -> None:
    generator = torch.Generator().manual_seed(31)
    source = torch.randn(16, 32, generator=generator) * 0.02
    rate = mxfp4_pq_rate(
        16,
        32,
        entries=16,
        index_bits=4,
        scale_bits=3,
    )
    reconstruction, codebook, assignments, scales, metadata = quantize_mxfp4_pq(
        source,
        rate,
        device="cpu",
        seed=37,
        iterations=3,
        vector_chunk=32,
    )
    assert reconstruction.shape == source.shape
    assert codebook.shape == (16, 8)
    assert assignments.shape == (64,)
    assert scales.shape == (16, 2)
    assert torch.isfinite(reconstruction).all()
    assert metadata["used_codewords"] > 0
    assert metadata["physical_storage_roundtrip_verified"] is True
