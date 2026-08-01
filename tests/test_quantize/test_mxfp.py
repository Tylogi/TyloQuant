import json
import struct

import numpy as np
import torch

from mfq.quantize.mxfp import (
    RawSafeTensorFile,
    decode_e8m0,
    decode_mxfp4,
    decode_mxfp8,
    read_mxfp4_rows,
)


def _write_safetensor(path, tensors):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        raw = bytes(data)
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_decode_e8m0_bias_and_nan():
    raw = np.array([0, 126, 127, 128, 254, 255], dtype=np.uint8)
    got = decode_e8m0(raw, device="cpu").numpy()
    expected = np.array(
        [2.0**-127, 0.5, 1.0, 2.0, 2.0**127, np.nan],
        dtype=np.float32,
    )
    np.testing.assert_allclose(got[:5], expected[:5], rtol=0, atol=0)
    assert np.isnan(got[5])


def test_decode_mxfp4_low_nibble_first():
    packed = np.array([[0x10, 0x32, 0x98, 0xFE] * 4], dtype=np.uint8)
    scales = np.array([[128]], dtype=np.uint8)
    got = decode_mxfp4(packed, scales, device="cpu").numpy()
    values = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            0.0,
            -0.5,
            -4.0,
            -6.0,
        ]
        * 4,
        dtype=np.float32,
    )
    np.testing.assert_array_equal(got, values[None] * 2.0)


def test_decode_mxfp8_uses_128x128_scale_blocks():
    source = torch.tensor(
        [[1.0] * 128, [0.5] * 128], dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    encoded = source.view(torch.uint8)
    scales = np.array([[127], [129]], dtype=np.uint8)
    got = decode_mxfp8(
        encoded,
        scales,
        row_start=127,
        total_rows=256,
        device="cpu",
    ).numpy()
    np.testing.assert_array_equal(got[0], np.ones(128, dtype=np.float32))
    np.testing.assert_array_equal(got[1], np.full(128, 2.0, dtype=np.float32))


def test_raw_safetensor_row_reader_decodes_mxfp4(tmp_path):
    packed = np.arange(64, dtype=np.uint8).reshape(4, 16)
    scales = np.array([[127], [128], [126], [129]], dtype=np.uint8)
    path = tmp_path / "mxfp4.safetensors"
    _write_safetensor(
        path,
        {
            "weight": ("I8", packed.shape, packed.tobytes()),
            "scale": ("F8_E8M0", scales.shape, scales.tobytes()),
        },
    )
    shard = RawSafeTensorFile(path)
    got = read_mxfp4_rows(
        shard,
        "weight",
        "scale",
        1,
        3,
        device="cpu",
    )
    expected = decode_mxfp4(
        packed[1:3],
        scales[1:3],
        device="cpu",
    )
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
