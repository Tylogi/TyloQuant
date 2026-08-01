from __future__ import annotations

import numpy as np

from mfq.formats.header import FileHeader
from mfq.formats.io import open_mmap, save
from mfq.formats.mx import MxTensor
from mfq.quantize.mfq_source import FullPrecisionMfqCheckpoint


def test_mx_tensor_roundtrip_and_streamed_rows(tmp_path):
    path = tmp_path / "full.mfq"
    fp4_values = np.arange(4 * 48, dtype=np.uint8).reshape(4, 48)
    fp4_scales = np.full((4, 3), 127, dtype=np.uint8)
    fp8_values = np.arange(128 * 128, dtype=np.uint8).reshape(128, 128)
    fp8_scales = np.full((1, 1), 127, dtype=np.uint8)
    save(
        path,
        FileHeader(version=2, model_arch="test-full-mfq"),
        {
            "fp4.weight": MxTensor(
                "MXFP4", (4, 96), fp4_values, fp4_scales
            ),
            "fp8.weight": MxTensor(
                "MXFP8", (128, 128), fp8_values, fp8_scales
            ),
        },
    )

    with open_mmap(path) as store:
        assert store.records["fp4.weight"].dtype == "MXFP4"
        assert store.records["fp8.weight"].dtype == "MXFP8"
        assert store["fp4.weight"].shape == (4, 96)
        assert store["fp8.weight"].shape == (128, 128)

    with FullPrecisionMfqCheckpoint(path) as checkpoint:
        fp4 = checkpoint.tensor_source("fp4.weight")
        fp8 = checkpoint.tensor_source("fp8.weight")
        assert fp4.read_rows(1, 3, device="cpu").shape == (2, 96)
        assert fp8.read_rows(126, 128, device="cpu").shape == (2, 128)
        sampled = fp4.read_rows(
            np.array([3, 0, 2], dtype=np.int64), device="cpu"
        )
        assert sampled.shape == (3, 96)
