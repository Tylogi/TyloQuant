from __future__ import annotations

import numpy as np
import pytest
import torch

from bench.dsv4f_mxfp4_sq2_eight import (
    FIXED32_PALETTE_IDS,
    SCREENED32_PALETTE_IDS,
    decode_eight_mxfp4_sq2,
    eight_mxfp4_sq2_rate,
    quantize_eight_mxfp4_sq2,
)
from bench.dsv4f_mxfp4_xor_sq import _unpack_two_bit_symbols
from mfq.formats.nvq import NVQ2_E8
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


def test_eight_state_sq2_rate_stays_below_nvq2() -> None:
    fixed16 = eight_mxfp4_sq2_rate(4096, 4096, palette_id_bits=4)
    nvq2_nbytes = NVQ2_E8.payload_nbytes(4096, 4096)

    assert fixed16.symbol_nbytes == 4_194_304
    assert fixed16.block_selector_nbytes == 65_536
    assert fixed16.state_scale_nbytes == 8_192
    assert fixed16.state_palette_nbytes == 16_384
    assert fixed16.matrix_scale_base_nbytes == 1
    assert fixed16.payload_nbytes == 4_284_417
    assert fixed16.payload_bpw == 2.042969226837158
    assert nvq2_nbytes == 4_290_560
    assert nvq2_nbytes - fixed16.payload_nbytes == 6_143

    fixed32 = eight_mxfp4_sq2_rate(4096, 4096)
    assert len(SCREENED32_PALETTE_IDS) == 32
    assert FIXED32_PALETTE_IDS.tolist() == [
        120,
        127,
        187,
        504,
        512,
        518,
        547,
        548,
        558,
        562,
        592,
        767,
        806,
        833,
        966,
        967,
        192,
        1112,
        971,
        802,
        559,
        240,
        112,
        226,
        232,
        182,
        188,
        121,
        505,
        848,
        0,
        1,
    ]
    assert len(np.unique(FIXED32_PALETTE_IDS)) == 32
    assert fixed32.state_palette_nbytes == 20_480
    assert fixed32.payload_nbytes == 4_288_513
    assert fixed32.payload_bpw == 2.044922351837158
    assert nvq2_nbytes - fixed32.payload_nbytes == 2_047


def test_eight_state_sq2_roundtrips_tags_and_native_mxfp4() -> None:
    packed, scale = _toy_native_mxfp4()
    source = decode_mxfp4(packed, scale, device="cpu")
    reconstruction, encoding, metadata = quantize_eight_mxfp4_sq2(
        packed,
        scale,
        maximum_refinement_steps=4,
    )
    decoded, packed_mxfp4, native_scale, block_tags = decode_eight_mxfp4_sq2(
        encoding.packed_symbols,
        encoding.packed_block_selectors,
        encoding.matrix_scale_base,
        encoding.packed_state_scales,
        encoding.packed_state_palettes,
        device="cpu",
    )
    symbols = _unpack_two_bit_symbols(encoding.packed_symbols).reshape(2, 2, 32)

    assert torch.equal(decoded, reconstruction)
    assert np.array_equal(packed_mxfp4, encoding.packed_mxfp4)
    assert np.array_equal(native_scale, encoding.native_scale_raw)
    assert np.array_equal(block_tags, encoding.block_tags)
    assert np.array_equal(np.bitwise_xor.reduce(symbols, axis=2), block_tags & 3)
    assert float((source - reconstruction).to(torch.float64).square().sum()) == (
        encoding.searched_sse
    )
    assert metadata["explicit_block_selector_bits"] == 1
    assert metadata["implicit_symbol_tag_bits"] == 2
    assert metadata["stored_fp16_scale_or_centroid"] is False
    assert metadata["physical_storage_roundtrip_verified"] is True
    assert metadata["final_values_are_native_block32_mxfp4"] is True
    assert all(len(metadata[key]) == 64 for key in metadata if key.endswith("_sha256"))


def test_eight_state_sq2_rejects_scale_ranges_outside_two_bits() -> None:
    packed, scale = _toy_native_mxfp4()
    scale = scale.copy()
    scale[0, 0] = 119
    scale[1, 1] = 123

    with pytest.raises(ValueError, match="four-value matrix scale window"):
        quantize_eight_mxfp4_sq2(packed, scale, maximum_refinement_steps=1)
