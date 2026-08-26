from __future__ import annotations

import numpy as np
import torch

from bench.dsv4f_mxfp4_sq3 import (
    _pack_fixed_width,
    _pack_three_bit_symbols,
    _unpack_fixed_width,
    _unpack_three_bit_symbols,
    decode_mxfp4_sq3,
    mxfp4_sq3_rate,
    quantize_mxfp4_sq3,
)
from bench.dsv4f_mxfp4_sq3_eight import (
    decode_eight_mxfp4_sq3,
    eight_mxfp4_sq3_rate,
    quantize_eight_mxfp4_sq3,
)
from bench.dsv4f_mxfp4_sq3_hybrid import hybrid_mxfp4_sq3_rate
from mfq.formats.nvq import NVQ3_D4
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


def test_sq3_rates_stay_below_nvq3() -> None:
    nvq3_nbytes = NVQ3_D4.payload_nbytes(4096, 4096)
    full = mxfp4_sq3_rate(4096, 4096)
    hybrid = hybrid_mxfp4_sq3_rate(4096, 4096)
    eight = eight_mxfp4_sq3_rate(4096, 4096)
    assert nvq3_nbytes == 6_387_712
    assert full.payload_nbytes == 6_378_496
    assert full.payload_bpw == 3.04150390625
    assert hybrid.payload_nbytes == 6_383_616
    assert hybrid.payload_bpw == 3.0439453125
    assert eight.payload_nbytes == 6_385_665
    assert eight.payload_bpw == 3.044922351837158
    assert full.payload_nbytes < nvq3_nbytes
    assert hybrid.payload_nbytes < nvq3_nbytes
    assert eight.payload_nbytes < nvq3_nbytes


def test_fixed_width_and_three_bit_symbol_roundtrip() -> None:
    symbols = (np.arange(64, dtype=np.uint8) % 8).reshape(2, 32)
    packed_symbols = _pack_three_bit_symbols(symbols)
    assert packed_symbols.shape == (2, 12)
    assert np.array_equal(_unpack_three_bit_symbols(packed_symbols), symbols)

    values = np.asarray([[0, 6434], [4095, 17]], dtype=np.uint16)
    packed_values = _pack_fixed_width(values, 13)
    unpacked = _unpack_fixed_width(packed_values, 13, count=values.size)
    assert np.array_equal(unpacked, values.reshape(-1))


def test_two_state_sq3_roundtrips_through_native_mxfp4() -> None:
    packed, scale = _toy_native_mxfp4()
    source = decode_mxfp4(packed, scale, device="cpu")
    reconstruction, encoding, metadata = quantize_mxfp4_sq3(
        packed,
        scale,
        exponent_radius=0,
        maximum_refinement_steps=4,
        row_chunk=2,
    )
    decoded, packed_mxfp4, native_scale, selectors = decode_mxfp4_sq3(
        encoding.packed_symbols,
        encoding.packed_block_selectors,
        encoding.state_scale_raw,
        encoding.packed_state_palettes,
        device="cpu",
    )
    assert torch.equal(decoded, reconstruction)
    assert np.array_equal(packed_mxfp4, encoding.packed_mxfp4)
    assert np.array_equal(native_scale, encoding.native_scale_raw)
    assert np.array_equal(selectors, encoding.block_selectors)
    assert float((source - reconstruction).to(torch.float64).square().sum()) == (
        encoding.searched_sse
    )
    assert metadata["stored_fp16_scale_or_centroid"] is False
    assert metadata["physical_storage_roundtrip_verified"] is True
    assert metadata["final_values_are_native_block32_mxfp4"] is True


def test_eight_state_sq3_roundtrips_tags_and_native_mxfp4() -> None:
    packed, scale = _toy_native_mxfp4()
    source = decode_mxfp4(packed, scale, device="cpu")
    reconstruction, encoding, metadata = quantize_eight_mxfp4_sq3(
        packed,
        scale,
        maximum_refinement_steps=4,
    )
    decoded, packed_mxfp4, native_scale, block_tags = decode_eight_mxfp4_sq3(
        encoding.packed_symbols,
        encoding.packed_block_selectors,
        encoding.matrix_scale_base,
        encoding.packed_state_scales,
        encoding.packed_state_palettes,
        device="cpu",
    )
    symbols = _unpack_three_bit_symbols(encoding.packed_symbols).reshape(2, 2, 32)
    assert torch.equal(decoded, reconstruction)
    assert np.array_equal(packed_mxfp4, encoding.packed_mxfp4)
    assert np.array_equal(native_scale, encoding.native_scale_raw)
    assert np.array_equal(block_tags, encoding.block_tags)
    assert np.array_equal(
        np.bitwise_xor.reduce(symbols & 3, axis=2),
        block_tags & 3,
    )
    assert float((source - reconstruction).to(torch.float64).square().sum()) == (
        encoding.searched_sse
    )
    assert metadata["explicit_block_selector_bits"] == 1
    assert metadata["implicit_symbol_tag_bits"] == 2
    assert metadata["stored_fp16_scale_or_centroid"] is False
    assert metadata["physical_storage_roundtrip_verified"] is True
    assert metadata["final_values_are_native_block32_mxfp4"] is True
