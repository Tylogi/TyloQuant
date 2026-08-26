from __future__ import annotations

import numpy as np
import torch

from bench.dsv4f_mxfp4_sq import (
    ALPHABETS,
    quantize_row_scale_strict_mxfp4_sq,
    quantize_strict_mxfp4_sq,
    strict_mxfp4_sq_rate,
    strict_mxfp4_sq_row_scale_rate,
)
from mfq.quantize.mxfp import decode_mxfp4


def test_strict_mxfp4_sq_rate_has_no_table_payload() -> None:
    rate = strict_mxfp4_sq_rate(4096, 4096)
    assert rate.symbol_nbytes == 4_194_304
    assert rate.scale_nbytes == 90_112
    assert rate.payload_nbytes == 4_284_416
    assert rate.payload_bpw == 2.04296875


def test_fixed_scalar_quantizer_roundtrips_as_native_mxfp4() -> None:
    packed = np.array(
        [
            [
                0x21,
                0x43,
                0x65,
                0x87,
                0x19,
                0x3B,
                0x5D,
                0x7F,
                0x12,
                0x34,
                0x56,
                0x78,
                0x91,
                0xB3,
                0xD5,
                0xF7,
            ]
        ],
        dtype=np.uint8,
    )
    source = decode_mxfp4(
        packed, np.array([[126]], dtype=np.uint8), device="cpu"
    )
    rate = strict_mxfp4_sq_rate(1, 32, scale_group_size=32)
    reconstruction, symbols, scale_raw, metadata = quantize_strict_mxfp4_sq(
        source,
        ALPHABETS["sym-1-3"],
        rate,
        device="cpu",
        group_chunk=1,
        exponent_radius=8,
    )
    assert reconstruction.shape == source.shape
    assert symbols.shape == source.shape
    assert int(symbols.max()) <= 3
    assert scale_raw.shape == (1, 1)
    assert metadata["learned_codebook"] is False
    assert metadata["physical_storage_roundtrip_verified"] is True
    assert metadata["final_values_are_native_block32_mxfp4"] is True


def test_row_base_one_bit_scale_layout_has_no_table() -> None:
    rate = strict_mxfp4_sq_row_scale_rate(4096, 4096)
    assert rate.symbol_nbytes == 4_194_304
    assert rate.row_base_nbytes == 4_096
    assert rate.row_delta_nbytes == 1_024
    assert rate.scale_selector_nbytes == 65_536
    assert rate.payload_nbytes == 4_264_960
    assert rate.payload_bpw == 2.03369140625


def test_row_base_one_bit_scale_roundtrips_as_native_mxfp4() -> None:
    packed = np.tile(
        np.array([[0x21, 0x43, 0x65, 0x87] * 4], dtype=np.uint8), (2, 1)
    )
    source = decode_mxfp4(
        packed,
        np.array([[124], [128]], dtype=np.uint8),
        device="cpu",
    )
    rate = strict_mxfp4_sq_row_scale_rate(2, 32)
    reconstruction, symbols, native_scale, metadata = (
        quantize_row_scale_strict_mxfp4_sq(
            source,
            ALPHABETS["sym-0p5-2"],
            rate,
            device="cpu",
            exponent_radius=8,
        )
    )
    assert reconstruction.shape == source.shape
    assert symbols.shape == source.shape
    assert native_scale.shape == (2, 1)
    assert metadata["learned_codebook"] is False
    assert metadata["physical_storage_roundtrip_verified"] is True


def test_ternary_layout_uses_only_fixed_zero_and_signed_level() -> None:
    source = torch.tensor(
        [[0.0] * 16 + [0.5, -0.5] * 8], dtype=torch.float32
    )
    rate = strict_mxfp4_sq_rate(1, 32, scale_group_size=32)
    reconstruction, symbols, _, metadata = quantize_strict_mxfp4_sq(
        source,
        ALPHABETS["ternary"],
        rate,
        device="cpu",
        group_chunk=1,
        exponent_radius=8,
    )
    assert set(symbols.unique().tolist()) <= {0, 1, 2}
    assert set(reconstruction.unique().tolist()) == {-0.5, 0.0, 0.5}
    assert metadata["alphabet_levels"] == [0.0, 1.0, -1.0, -0.0]
