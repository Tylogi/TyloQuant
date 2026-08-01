from __future__ import annotations

import numpy as np

from mfq.formats.npq0_l import Npq0LTensor
from mfq.formats.npq0_s import Npq0STensor
from mfq.formats.nvq import (
    D4_1024,
    D4_256,
    E8_1024,
    E8_256,
    E8_4096,
    NVQ2_E8,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_1024,
    NvqJscTensor,
    NvqTensor,
)
from mfq.formats.nvq1_l import NVQ1_L_T8_S3, Nvq1LTensor
from mfq.formats.nvq1_s import NVQ1_S, NVQ1_S_BOOTSTRAP_BANKS, Nvq1STensor
from mfq.quantize.npq0_l import dequantize_npq0_l
from mfq.quantize.npq0_s import dequantize_npq0_s
from mfq.quantize.nvq1_l_quant import dequantize as dequantize_nvq1_l
from mfq.quantize.nvq1_s_quant import dequantize as dequantize_nvq1_s
from mfq.quantize.nvq_jsc import dequantize_nvq_jsc
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq


FLAT_FAMILIES = (
    "NVQ1-L",
    "NVQ1-S",
    "NPQ0-L",
    "NPQ0-S",
    "NVQ2",
    "NVQ2J",
    "NVQ2J-L",
    "NVQ2J-XL",
    "NVQ3",
    "NVQ3J",
    "NVQ3J-L",
)


_JSC_FAMILIES = {
    "NVQ2J": (NVQ2_E8, E8_256),
    "NVQ2J-L": (NVQ2_E8_1024, E8_1024),
    "NVQ2J-XL": (NVQ2_E8_4096, E8_4096),
    "NVQ3J": (NVQ3_D4, D4_256),
    "NVQ3J-L": (NVQ3_D4_1024, D4_1024),
}


def make_flat_family(
    family: str,
    *,
    rows: int = 6,
    neuron_len: int = 96,
    seed: int = 20260723,
):
    rng = np.random.default_rng(seed + FLAT_FAMILIES.index(family))
    ng = (neuron_len + 23) // 24
    nvec8 = neuron_len // 8
    nvec4 = neuron_len // 4
    anchors = rng.uniform(0.005, 0.05, rows).astype(np.float32)
    shape = (rows, neuron_len)
    if family == "NVQ1-L":
        return Nvq1LTensor(
            NVQ1_L_T8_S3,
            shape,
            0,
            neuron_len,
            anchors,
            rng.integers(1, 8, (rows, ng), dtype=np.uint8),
            rng.integers(0, 2048, (rows, nvec8), dtype=np.uint16),
            rng.integers(0, 2, (rows, ng), dtype=np.uint8),
        )
    if family == "NVQ1-S":
        return Nvq1STensor(
            NVQ1_S,
            shape,
            0,
            neuron_len,
            anchors,
            rng.integers(1, 16, (rows, ng), dtype=np.uint8),
            rng.integers(0, 512, (rows, nvec8), dtype=np.uint16),
            rng.integers(0, 2, (rows, ng), dtype=np.uint8),
            NVQ1_S_BOOTSTRAP_BANKS,
        )
    if family == "NPQ0-L":
        return Npq0LTensor(
            shape,
            0,
            neuron_len,
            anchors,
            np.linspace(0.125, 1.0, 8, dtype=np.float32),
            rng.integers(0, 8, (rows, ng), dtype=np.uint8),
            rng.integers(0, 128, (rows, nvec8), dtype=np.uint8),
            rng.integers(-8, 9, (8, 8, 4), dtype=np.int16).astype(np.int8),
            rng.integers(-8, 9, (8, 16, 4), dtype=np.int16).astype(np.int8),
        )
    if family == "NPQ0-S":
        return Npq0STensor(
            shape,
            0,
            neuron_len,
            anchors,
            np.linspace(0.25, 1.0, 4, dtype=np.float32),
            rng.integers(0, 4, (rows, ng), dtype=np.uint8),
            rng.integers(0, 64, (rows, nvec8), dtype=np.uint8),
            rng.integers(-8, 9, (4, 8, 4), dtype=np.int16).astype(np.int8),
            rng.integers(-8, 9, (4, 8, 4), dtype=np.int16).astype(np.int8),
        )
    if family in {"NVQ2", "NVQ3"}:
        spec = NVQ2_E8 if family == "NVQ2" else NVQ3_D4
        nvec = nvec8 if family == "NVQ2" else nvec4
        return NvqTensor(
            spec,
            shape,
            0,
            neuron_len,
            anchors,
            rng.integers(1, 16, (rows, ng), dtype=np.uint8),
            rng.integers(0, 256, (rows, nvec), dtype=np.uint8),
            rng.integers(0, 128, (rows, nvec8), dtype=np.uint8),
        )
    spec, base = _JSC_FAMILIES[family]
    nvec = neuron_len // spec.vector_size
    codebooks = np.stack(
        (
            base.astype(np.int16) * 8,
            np.roll(base, 1, axis=0).astype(np.int16) * 8,
        )
    ).astype(np.int8)
    return NvqJscTensor(
        shape,
        0,
        neuron_len,
        anchors,
        np.arange(16, dtype=np.float32),
        np.arange(16, dtype=np.uint8) & 1,
        rng.integers(1, 16, (rows, ng), dtype=np.uint8),
        rng.integers(
            0,
            spec.codebook_entries,
            (rows, nvec),
            dtype=np.uint8 if spec.index_bits <= 8 else np.uint16,
        ),
        rng.integers(0, 128, (rows, nvec8), dtype=np.uint8),
        codebooks,
        base_spec=spec,
    )


def dequantize_flat_family(tensor) -> np.ndarray:
    if isinstance(tensor, NvqJscTensor):
        return dequantize_nvq_jsc(tensor)
    if isinstance(tensor, NvqTensor):
        return dequantize_nvq(tensor)
    if isinstance(tensor, Nvq1LTensor):
        return dequantize_nvq1_l(tensor)
    if isinstance(tensor, Nvq1STensor):
        return dequantize_nvq1_s(tensor)
    if isinstance(tensor, Npq0LTensor):
        return dequantize_npq0_l(tensor)
    if isinstance(tensor, Npq0STensor):
        return dequantize_npq0_s(tensor)
    raise TypeError(type(tensor))
