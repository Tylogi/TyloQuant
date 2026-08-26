from __future__ import annotations

import numpy as np
import torch

from bench.dsv4f_mxfp4_adaptive_sq import (
    FIXED4_PALETTE_IDS,
    FIXED16_PALETTE_IDS,
    PALETTE_NIBBLES,
    PALETTE_VALUES,
    adaptive_mxfp4_sq_rate,
    quantize_adaptive_mxfp4_sq,
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


def test_adaptive_rate_stays_below_nvq2() -> None:
    fixed = adaptive_mxfp4_sq_rate(4096, 4096, palette_id_bits=2)
    assert fixed.symbol_nbytes == 4_194_304
    assert fixed.block_selector_nbytes == 65_536
    assert fixed.state_scale_nbytes == 8_192
    assert fixed.state_palette_nbytes == 2_048
    assert fixed.payload_nbytes == 4_270_080
    assert fixed.payload_bpw == 2.0361328125

    fixed16 = adaptive_mxfp4_sq_rate(4096, 4096, palette_id_bits=4)
    assert fixed16.state_palette_nbytes == 4_096
    assert fixed16.payload_nbytes == 4_272_128
    assert fixed16.payload_bpw == 2.037109375

    full = adaptive_mxfp4_sq_rate(4096, 4096, palette_id_bits=11)
    assert full.state_palette_nbytes == 11_264
    assert full.payload_nbytes == 4_279_296
    assert full.payload_bpw == 2.04052734375
    assert full.payload_nbytes < 4_290_560


def test_fixed_palette_catalog_is_native_e2m1() -> None:
    assert PALETTE_VALUES.shape == (1365, 4)
    assert PALETTE_NIBBLES.shape == (1365, 4)
    assert PALETTE_VALUES[FIXED4_PALETTE_IDS].tolist() == [
        [-4.0, -1.5, 1.0, 4.0],
        [-4.0, -1.0, 1.5, 4.0],
        [-3.0, -1.0, 0.5, 2.0],
        [-2.0, -0.5, 1.0, 3.0],
    ]
    assert len(FIXED16_PALETTE_IDS) == 16
    assert set(FIXED4_PALETTE_IDS.tolist()) <= set(FIXED16_PALETTE_IDS.tolist())


def test_fixed4_roundtrips_through_native_mxfp4() -> None:
    packed, scale = _toy_native_mxfp4()
    reconstruction, encoding, metadata = quantize_adaptive_mxfp4_sq(
        packed,
        scale,
        mode="fixed4",
        exponent_radius=2,
        row_chunk=1,
    )
    decoded = decode_mxfp4(
        encoding.packed_mxfp4,
        encoding.native_scale_raw,
        device="cpu",
    )
    assert torch.equal(decoded, reconstruction)
    assert encoding.symbols.shape == (2, 64)
    assert int(encoding.symbols.max()) <= 3
    assert encoding.block_selectors.shape == (2, 2)
    assert set(np.unique(encoding.state_palette_ids)) <= set(FIXED4_PALETTE_IDS.tolist())
    assert metadata["learned_vector_codebook"] is False
    assert metadata["stored_fp16_scale_or_centroid"] is False
    assert metadata["physical_storage_roundtrip_verified"] is True
    assert metadata["final_values_are_native_block32_mxfp4"] is True


def test_full_catalog_refinement_never_loses_to_fixed4_seed() -> None:
    packed, scale = _toy_native_mxfp4()
    source = decode_mxfp4(packed, scale, device="cpu")
    fixed, _, _ = quantize_adaptive_mxfp4_sq(
        packed,
        scale,
        mode="fixed16",
        exponent_radius=2,
        row_chunk=2,
    )
    full, encoding, metadata = quantize_adaptive_mxfp4_sq(
        packed,
        scale,
        mode="full",
        exponent_radius=2,
        maximum_refinement_steps=10,
        row_chunk=2,
    )
    fixed_sse = float((source - fixed).to(torch.float64).square().sum())
    full_sse = float((source - full).to(torch.float64).square().sum())
    assert full_sse <= fixed_sse
    assert int(encoding.state_palette_ids.max()) < 1365
    assert metadata["implicit_full_e2m1_palette_catalog"] is True
    assert metadata["optimizer_is_global_within_catalog"] is False
