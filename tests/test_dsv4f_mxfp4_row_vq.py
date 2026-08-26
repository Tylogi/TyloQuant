import numpy as np
import torch

from bench.dsv4f_mxfp4_row_vq import (
    match_row_vq_rate,
    mxfp4_matrix_nbytes,
    project_mxfp4,
    quantize_row_vq,
    row_vq_payload_nbytes,
)
from mfq.formats.nvq import NVQ2_E8
from mfq.quantize.mxfp import decode_mxfp4


def test_rate_match_4096_square_against_nvq2_e8() -> None:
    rows = columns = 4096
    budget = NVQ2_E8.payload_nbytes(rows, columns)
    matched = match_row_vq_rate(rows, columns, budget)
    assert matched.entries == 1969
    assert matched.index_bits == 11
    assert matched.payload_nbytes == 4_290_176
    assert matched.budget_nbytes - matched.payload_nbytes == 384
    assert row_vq_payload_nbytes(rows, columns, 1970) > budget


def test_mxfp4_projection_emits_decodable_physical_streams() -> None:
    values = torch.tensor(
        [
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 4,
            [0.0, -1.0, 2.0, -3.0, 4.0, -6.0, 1.5, -0.5] * 4,
        ],
        dtype=torch.float32,
    )
    projected, packed, scales = project_mxfp4(values, return_storage=True)
    assert packed is not None and scales is not None
    assert packed.numel() + scales.numel() == mxfp4_matrix_nbytes(2, 32)
    decoded = decode_mxfp4(
        packed.numpy(), scales.numpy(), device="cpu"
    )
    torch.testing.assert_close(projected, values, rtol=0, atol=0)
    torch.testing.assert_close(decoded, values, rtol=0, atol=0)


def test_mxfp4_projection_chooses_valid_nearest_values() -> None:
    source = torch.linspace(-7.0, 7.0, 64, dtype=torch.float32).reshape(2, 32)
    projected, packed, scales = project_mxfp4(source, return_storage=True)
    assert packed is not None and scales is not None
    decoded = decode_mxfp4(
        np.asarray(packed), np.asarray(scales), device="cpu"
    )
    torch.testing.assert_close(decoded, projected, rtol=0, atol=0)
    assert torch.isfinite(projected).all()


def test_nearest_pair_initialization_uses_all_rows_without_singletons() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(16, 32, generator=generator) * 0.02
    source, _, _ = project_mxfp4(source)
    reconstruction, centroids, metadata = quantize_row_vq(
        source,
        7,
        device="cpu",
        seed=11,
        iterations=3,
        initialization="nearest-pair",
        assignment_row_chunk=16,
        projection_row_chunk=16,
    )
    initialization = metadata["initialization_metadata"]
    assert reconstruction.shape == source.shape
    assert centroids.shape == (7, 32)
    assert initialization["partition_two_row_clusters"] == 6
    assert initialization["partition_four_row_clusters"] == 1
    assert metadata["singleton_clusters"] == 0
    assert metadata["empty_clusters"] == 0


def test_ward_initialization_cuts_to_exact_requested_codebook() -> None:
    generator = torch.Generator().manual_seed(19)
    source = torch.randn(16, 32, generator=generator) * 0.02
    source, _, _ = project_mxfp4(source)
    reconstruction, centroids, metadata = quantize_row_vq(
        source,
        7,
        device="cpu",
        seed=23,
        iterations=3,
        initialization="ward",
        assignment_row_chunk=16,
        projection_row_chunk=16,
    )
    initialization = metadata["initialization_metadata"]
    assert reconstruction.shape == source.shape
    assert centroids.shape == (7, 32)
    assert initialization["partition_cluster_min"] >= 1
    assert initialization["partition_empty_clusters"] == 0
    assert metadata["empty_clusters"] == 0
