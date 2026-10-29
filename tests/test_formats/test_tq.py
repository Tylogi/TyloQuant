from __future__ import annotations

import struct

import numpy as np
import pytest

from mfq.formats.tq import RELATIVE, TQTensor, fixed_codebook, pack_states, unpack_states


@pytest.mark.parametrize("bits", range(1, 9))
def test_ring_pack_and_wrap(bits):
    rng = np.random.default_rng(4817 + bits)
    packed = rng.integers(0, 256, (3, 32 * bits), dtype=np.uint8)
    states = unpack_states(packed, bits)
    np.testing.assert_array_equal(pack_states(states, bits), packed)
    # Consecutive states retain the preceding 16-k bits, including the tail.
    mask = (1 << (16 - bits)) - 1
    np.testing.assert_array_equal(states >> bits, np.roll(states, 1, axis=1) & mask)


def tensor_fixture(columns=False):
    rng = np.random.default_rng(123)
    bits = np.array([8, 1, 5, 2, 7, 3, 6, 4, 1], dtype=np.uint64)
    width, gs = 768, 512
    offsets = np.r_[np.uint64(0), np.cumsum(width * bits[:-1] // 32)]
    anchor = np.full(len(bits), .02, dtype=np.float16)
    desc = offsets | ((bits - 1) << np.uint64(32))
    desc |= anchor.view(np.uint16).astype(np.uint64) << np.uint64(40)
    packed = rng.integers(0, 256, int((width * bits // 8).sum()), dtype=np.uint8)
    # Nine rows * two partial subgroups exercises nibble indexing across rows.
    scales = np.linspace(.5, 1.5, width).astype(np.float16) if columns else np.arange(9, dtype=np.uint8) * 17
    return TQTensor(len(bits), width, gs, columns, desc, scales, packed)


@pytest.mark.parametrize("columns", [False, True])
def test_serialized_roundtrip_and_cpu_decode(tmp_path, columns):
    tensor = tensor_fixture(columns)
    path = tmp_path / "matrix.trellis"
    tensor.save(path)
    restored = TQTensor.load(path)
    assert restored.to_bytes() == tensor.to_bytes()
    assert path.stat().st_size == tensor.nbytes
    assert tensor.bpw == tensor.nbytes * 8 / (tensor.rows * tensor.width)
    decoded = tensor.dequantize(513)
    assert decoded.shape == (9, 513) and decoded.dtype == np.float16
    # Independent scalar extraction, deliberately crossing ring byte boundaries.
    lut = fixed_codebook()
    for row in range(tensor.rows):
        k = int(tensor.bits[row])
        offset = int(tensor.descriptors[row] & np.uint64(0xFFFFFFFF)) * 4
        for col in (0, 1, 127, 254, 255, 256, 511, 512):
            ring = tensor.packed[offset + (col // 256) * 32 * k:][:32 * k]
            raw = int.from_bytes(ring.tobytes(), "little")
            shift = (255 - col % 256) * k
            state = ((raw | (raw << (256 * k))) >> shift) & 65535
            if columns:
                relative = tensor.scales[col]
            else:
                group = row * tensor.groups + col // tensor.gs
                code = (int(tensor.scales[group // 2]) >> (4 * (group % 2))) & 15
                relative = RELATIVE[code]
            expected = np.float16(np.float32(lut[state]) * (np.float32(tensor.anchors[row]) * np.float32(relative)))
            assert decoded[row, col] == expected


def test_reject_bad_payloads():
    raw = tensor_fixture().to_bytes()
    for data in (raw[:31], raw[:-1], raw + b"x", b"BADMAGIC" + raw[8:]):
        with pytest.raises(ValueError):
            TQTensor.from_bytes(data)
    changed = bytearray(raw)
    changed[32] = 1
    with pytest.raises(ValueError, match="coefficients"):
        TQTensor.from_bytes(changed)
    changed = bytearray(raw)
    struct.pack_into("<Q", changed, 64, 1)
    with pytest.raises(ValueError, match="offsets"):
        TQTensor.from_bytes(changed)


def test_odd_subgroup_selector_count():
    # One GS512 subgroup per row, three selectors in two bytes.
    desc = np.array([0, 8, 16], dtype=np.uint64)
    desc |= np.uint64(0x3C00) << np.uint64(40)
    tensor = TQTensor(3, 256, 512, False, desc, np.array([0x10, 0x02]), np.zeros(96, dtype=np.uint8))
    decoded = TQTensor.from_bytes(tensor.to_bytes()).dequantize()
    np.testing.assert_array_equal(decoded[:, 0], (fixed_codebook()[0].astype(np.float32) * RELATIVE[:3].astype(np.float32)).astype(np.float16))


@pytest.mark.parametrize("bit", [*range(35, 40), *range(56, 64)])
def test_reject_reserved_row_descriptor_bits(bit):
    raw = bytearray(tensor_fixture().to_bytes())
    descriptor = struct.unpack_from("<Q", raw, 64)[0]
    struct.pack_into("<Q", raw, 64, descriptor | (1 << bit))
    with pytest.raises(ValueError, match="Reserved row descriptor"):
        TQTensor.from_bytes(raw)


def test_model_container_roundtrip(tmp_path):
    from mfq.formats.header import FileHeader
    from mfq.formats.io import open_mmap, save
    t = tensor_fixture()
    path = tmp_path / "tq.mfq"
    save(path, FileHeader(version=2, model_arch="tq_test",
        extra={"tq_shapes": {"projection": [t.rows, 513]}}), {"projection": t})
    with open_mmap(path) as model:
        assert model.records["projection"].dtype == "TQ"
        assert model["projection"].to_bytes() == t.to_bytes()
        assert model.header.extra["tq_shapes"]["projection"] == [t.rows, 513]
