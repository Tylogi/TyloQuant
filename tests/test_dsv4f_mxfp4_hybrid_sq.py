from __future__ import annotations

import numpy as np
import torch

from bench.dsv4f_mxfp4_hybrid_sq import (
    _pack_block_selectors,
    _unpack_block_selectors,
    decode_hybrid_mxfp4_sq,
    hybrid_mxfp4_sq_rate,
    quantize_hybrid_mxfp4_sq,
)
from bench.dsv4f_mxfp4_xor_sq import _unpack_two_bit_symbols
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


def test_hybrid_rate_stays_below_nvq2() -> None:
    rate = hybrid_mxfp4_sq_rate(4096, 4096)
    assert rate.symbol_nbytes == 4_194_304
    assert rate.block_selector_nbytes == 65_536
    assert rate.state_scale_nbytes == 16_384
    assert rate.state_palette_nbytes == 8_192
    assert rate.payload_nbytes == 4_284_416
    assert rate.payload_bpw == 2.04296875
    assert rate.payload_nbytes < 4_290_560


def test_one_bit_selector_pack_roundtrip() -> None:
    selectors = (np.arange(256).reshape(2, 128) % 3 == 0).astype(np.uint8)
    packed = _pack_block_selectors(selectors)
    assert packed.nbytes == 32
    assert np.array_equal(_unpack_block_selectors(packed, rows=2, blocks=128), selectors)


def test_hybrid_sq_roundtrips_through_native_mxfp4() -> None:
    packed, scale = _toy_native_mxfp4()
    source = decode_mxfp4(packed, scale, device="cpu")
    reconstruction, encoding, metadata = quantize_hybrid_mxfp4_sq(
        packed,
        scale,
        exponent_radius=0,
        maximum_refinement_steps=4,
    )
    decoded, packed_mxfp4, native_scale, block_tags = decode_hybrid_mxfp4_sq(
        encoding.packed_symbols,
        encoding.packed_block_selectors,
        encoding.state_scale_raw,
        encoding.packed_state_palettes,
        device="cpu",
    )
    symbols = _unpack_two_bit_symbols(encoding.packed_symbols).reshape(2, 2, 32)
    assert torch.equal(decoded, reconstruction)
    assert np.array_equal(packed_mxfp4, encoding.packed_mxfp4)
    assert np.array_equal(native_scale, encoding.native_scale_raw)
    assert np.array_equal(block_tags, encoding.block_tags)
    assert np.array_equal(
        np.bitwise_xor.reduce(symbols, axis=2) & 1,
        block_tags & 1,
    )
    assert float((source - reconstruction).to(torch.float64).square().sum()) == (
        encoding.searched_sse
    )
    assert metadata["explicit_block_selector_bits"] == 1
    assert metadata["implicit_symbol_tag_bits"] == 1
    assert metadata["stored_fp16_scale_or_centroid"] is False
    assert metadata["physical_storage_roundtrip_verified"] is True
    assert metadata["final_values_are_native_block32_mxfp4"] is True
