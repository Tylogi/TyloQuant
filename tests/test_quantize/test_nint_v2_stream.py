from __future__ import annotations

import numpy as np
import torch

from mfq.formats.io import unpack_nint
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import (
    allocate_row_sub_bits,
    dequantize,
    quantize,
)
from mfq.tools.quantize_hf_to_mfq import _write_nint_axis0_blob


def test_stream_writer_packs_mixed_neuron_metadata_across_chunk_boundaries(tmp_path):
    rng = np.random.default_rng(20260911)
    weight = rng.normal(0, 0.05, size=(6, 73)).astype(np.float32)
    neuron_importance = np.asarray([1.0, 2.0, 3.0, 10.0, 50.0, 100.0], dtype=np.float32)
    row_sub_bits = allocate_row_sub_bits(neuron_importance, 6)
    output = tmp_path / "mixed-nint.blob"

    _write_nint_axis0_blob(
        torch.from_numpy(weight),
        weight.shape,
        NintSpec(4, 24, 6),
        output,
        row_chunk=2,
        quant_backend="cpu",
        device="cpu",
        neuron_importance_rows=lambda start, end: neuron_importance[start:end],
    )

    restored = unpack_nint(output.read_bytes())
    expected = quantize(
        weight,
        NintSpec(4, 24, 6),
        row_sub_bits=row_sub_bits,
    )
    np.testing.assert_array_equal(restored.row_sub_bits, [5, 5, 5, 6, 7, 8])
    np.testing.assert_allclose(dequantize(restored), dequantize(expected), rtol=0, atol=0)
