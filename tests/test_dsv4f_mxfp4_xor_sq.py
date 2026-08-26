from __future__ import annotations

import numpy as np
import torch

from bench.dsv4f_mxfp4_adaptive_sq import FIXED16_PALETTE_IDS
from bench.dsv4f_mxfp4_xor_sq import (
    _candidate_xor_errors,
    _pack_two_bit_symbols,
    _quantize_blocks_for_tag,
    _unpack_two_bit_symbols,
    decode_xor_mxfp4_sq,
    quantize_xor_mxfp4_sq,
    xor_mxfp4_sq_rate,
)
from mfq.quantize.mxfp import decode_mxfp4


def _toy_native_mxfp4() -> tuple[np.ndarray, np.ndarray]:
    packed = np.asarray(
        [
            [0x21, 0x43, 0x65, 0x87, 0x19, 0x3B, 0x5D, 0x7F] * 4,
            [0xF7, 0xD5, 0xB3, 0x91, 0x78, 0x56, 0x34, 0x12] * 4,
        ],
        dtype=np.uint8,
    )
    scale = np.asarray([[121, 122], [120, 121]], dtype=np.uint8)
    return packed, scale


def test_xor_rate_has_no_explicit_selector() -> None:
    rate = xor_mxfp4_sq_rate(4096, 4096)
    assert rate.symbol_nbytes == 4_194_304
    assert rate.explicit_block_selector_nbytes == 0
    assert rate.state_scale_nbytes == 16_384
    assert rate.state_palette_nbytes == 8_192
    assert rate.payload_nbytes == 4_218_880
    assert rate.payload_bpw == 2.01171875
    assert rate.payload_nbytes < 4_290_560


def test_two_bit_symbol_pack_roundtrip() -> None:
    symbols = np.arange(64, dtype=np.uint8).reshape(2, 32) & 3
    assert np.array_equal(_unpack_two_bit_symbols(_pack_two_bit_symbols(symbols)), symbols)


def test_xor_constrained_cost_and_materialization_agree() -> None:
    source_nibbles = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7] * 4], dtype=np.uint8)
    source_scales = np.asarray([121], dtype=np.uint8)
    state_scales = np.asarray([121], dtype=np.uint8)
    state_palettes = np.asarray([FIXED16_PALETTE_IDS[0]], dtype=np.int16)
    errors = _candidate_xor_errors(
        source_nibbles,
        source_scales,
        state_scales,
        state_palettes,
    )
    scale = 2.0 ** (121 - 127)
    target = (
        np.asarray(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 4,
            dtype=np.float64,
        )[None, :]
        * scale
    )
    levels = np.asarray([-6.0, -3.0, 0.0, 3.0]) * scale
    for tag in range(4):
        symbols, measured = _quantize_blocks_for_tag(target, levels, tag)
        assert int(np.bitwise_xor.reduce(symbols, axis=1)[0]) == tag
        assert measured[0] == errors[tag, 0, 0]


def test_xor_sq_roundtrips_through_native_mxfp4() -> None:
    packed, scale = _toy_native_mxfp4()
    source = decode_mxfp4(packed, scale, device="cpu")
    reconstruction, encoding, metadata = quantize_xor_mxfp4_sq(
        packed,
        scale,
        exponent_radius=0,
        maximum_refinement_steps=4,
    )
    decoded, packed_mxfp4, native_scale, block_tags = decode_xor_mxfp4_sq(
        encoding.packed_symbols,
        encoding.state_scale_raw,
        encoding.packed_state_palettes,
        device="cpu",
    )
    assert torch.equal(decoded, reconstruction)
    assert np.array_equal(packed_mxfp4, encoding.packed_mxfp4)
    assert np.array_equal(native_scale, encoding.native_scale_raw)
    assert np.array_equal(block_tags, encoding.block_tags)
    assert np.array_equal(
        np.bitwise_xor.reduce(
            _unpack_two_bit_symbols(encoding.packed_symbols).reshape(2, 2, 32),
            axis=2,
        ),
        block_tags,
    )
    assert float((source - reconstruction).to(torch.float64).square().sum()) == (
        encoding.searched_sse
    )
    assert metadata["explicit_block_selector_bits"] == 0
    assert metadata["stored_fp16_scale_or_centroid"] is False
    assert metadata["physical_storage_roundtrip_verified"] is True
    assert metadata["final_values_are_native_block32_mxfp4"] is True
