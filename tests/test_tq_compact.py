import struct

import numpy as np
import pytest

from mfq.formats.tq import TQTensor
from mfq.formats.tq_rows import TQRows


def padded_tensor(logical):
    rng = np.random.default_rng(20261030)
    width = (logical + 255) // 256 * 256
    bits = np.arange(1, 9, dtype=np.uint64)
    lengths = width * bits // 32
    offsets = np.r_[np.uint64(0), np.cumsum(lengths[:-1], dtype=np.uint64)]
    descriptors = offsets | ((bits - 1) << np.uint64(32))
    descriptors |= np.uint64(np.float16(.03125).view(np.uint16)) << np.uint64(40)
    return TQTensor(8, width, 512, True, descriptors,
                    rng.uniform(.5, 1.5, width).astype(np.float16),
                    rng.integers(0, 256, int(lengths.sum()) * 4, dtype=np.uint8))


@pytest.mark.parametrize("logical", [64, 128, 192, 320, 384, 448, 576, 640, 704, 1088])
def test_compact_preserves_mixed_precision_states_and_rows(logical):
    source = padded_tensor(logical)
    compact = source.compact_tail(logical)
    expected = source.dequantize(logical)
    np.testing.assert_array_equal(compact.dequantize(), expected)
    np.testing.assert_array_equal(compact.bits, source.bits)
    np.testing.assert_array_equal(compact.anchors, source.anchors)
    for row in range(source.rows):
        np.testing.assert_array_equal(compact.states(row), source.states(row)[:logical])
    raw = compact.to_bytes()
    assert raw[:8] == b"RTRELL06"
    padding = source.width - logical
    assert source.nbytes - compact.nbytes == padding * int(source.bits.sum()) // 8 + padding * 2
    np.testing.assert_array_equal(TQTensor.from_bytes(raw).dequantize(), expected)
    reader = TQRows(len(raw), lambda begin, size: raw[begin:begin + size])
    np.testing.assert_array_equal(reader.slice_rows(2, 7).dequantize(), expected[2:7])
    np.testing.assert_array_equal(reader.gather_rows([7, 0, 3, 3]).dequantize(), expected[[7, 0, 3, 3]])


def test_compact_rejects_invalid_boundary_and_odd_word_rows():
    raw = bytearray(padded_tensor(640).compact_tail(640).to_bytes())
    at = 64 + 7 * 8
    descriptor = struct.unpack_from("<Q", raw, at)[0]
    struct.pack_into("<Q", raw, at, descriptor | (1 << 31))
    with pytest.raises(ValueError, match="boundary"):
        TQTensor.from_bytes(raw)
    with pytest.raises(ValueError, match="even word"):
        padded_tensor(32).compact_tail(32)


def test_compact_model_repack_keeps_source_and_logical_weights(tmp_path):
    from mfq.formats.header import FileHeader
    from mfq.formats.io import open_mmap, save
    from mfq.tools.compact_tq import repack_file

    tensor = padded_tensor(640)
    source, target = tmp_path / "source.mfq", tmp_path / "compact.tq"
    save(source, FileHeader(version=2, model_arch="tq_test",
        extra={"tq_shapes": {"projection": [tensor.rows, 640]}}), {"projection": tensor})
    original = source.read_bytes()
    result = repack_file(source, target)
    assert result["compacted_tensors"] == 1
    assert target.stat().st_size < source.stat().st_size
    assert source.read_bytes() == original
    with open_mmap(target) as model:
        np.testing.assert_array_equal(model["projection"].dequantize(), tensor.dequantize(640))
    with pytest.raises(FileExistsError):
        repack_file(source, target)
