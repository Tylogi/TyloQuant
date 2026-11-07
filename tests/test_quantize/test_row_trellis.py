import itertools
import struct

import numpy as np
import pytest

from mfq.quantize.row_trellis import (
    RowTrellisTensor, allocate_bits, assemble_rows, decode_cpu, fixed_nbytes,
)


def example_tensor(rows=65, columns=640, group_size=128, state_bits=16, columns_scale=True):
    rng = np.random.default_rng(417)
    qs = np.resize(np.arange(1, 9, dtype=np.uint8), rows)
    qs[:31] = 8
    buckets = {
        row: (
            rng.integers(0, 2**32, columns * int(q) // 32, dtype=np.uint32),
            np.float16(0.03125),
            rng.integers(0, 16, columns // group_size, dtype=np.uint8),
        )
        for row, q in enumerate(qs)
    }
    codes = rng.integers(0, 16, columns, dtype=np.uint8) if columns_scale else None
    return assemble_rows(
        buckets, qs, (rows, columns), group_size=group_size, state_bits=state_bits,
        column_codes=codes,
    )


@pytest.mark.parametrize("group_size,columns", [(128, 640), (256, 768)])
@pytest.mark.parametrize("state_bits", [10, 16])
@pytest.mark.parametrize("columns_scale", [False, True])
def test_container_roundtrip_and_accounting(group_size, columns, state_bits, columns_scale):
    tensor = example_tensor(columns=columns, group_size=group_size, state_bits=state_bits, columns_scale=columns_scale)
    blob = tensor.to_bytes()
    loaded = RowTrellisTensor.from_bytes(blob)
    assert loaded.to_bytes() == blob
    assert loaded.nbytes == len(blob)
    assert loaded.bpw == len(blob) * 8 / np.prod(tensor.shape)
    expected = fixed_nbytes(tensor.shape, group_size, columns_scale) + int(tensor.precisions.sum()) * columns // 8
    assert len(blob) == expected
    assert int.from_bytes(tensor.descriptors[31].tobytes(), "little") >> 15 == 248
    assert int.from_bytes(tensor.descriptors[32].tobytes(), "little") >> 15 == 0
    assert np.isfinite(decode_cpu(loaded)).all()
    assert not loaded.payload.flags.writeable


def test_budget_allocation_matches_exhaustive_search():
    rng = np.random.default_rng(51)
    errors = rng.uniform(0, 4, (4, 8))
    for budget in (4, 6, 12, 20, 32, 40):
        qs = allocate_bits(errors, budget)
        actual = errors[np.arange(4), qs - 1].sum()
        expected = min(
            sum(errors[row, q - 1] for row, q in enumerate(candidate))
            for candidate in itertools.product(range(1, 9), repeat=4) if sum(candidate) <= budget
        )
        assert actual == pytest.approx(expected)
        assert qs.sum() <= budget


def test_budget_is_not_forced_to_spend_unused_bits():
    errors = np.tile(np.arange(8), (3, 1))
    np.testing.assert_array_equal(allocate_bits(errors, 24), [1, 1, 1])


@pytest.mark.parametrize("errors,budget", [
    (np.zeros((3, 8)), 2), (np.zeros((0, 8)), 0), (np.zeros((3, 7)), 9),
    (np.full((3, 8), np.nan), 9), (-np.ones((3, 8)), 9),
])
def test_bad_budget_arguments(errors, budget):
    with pytest.raises(ValueError):
        allocate_bits(errors, budget)


@pytest.mark.parametrize("change", [
    lambda b: b[:16],
    lambda b: b[:-1],
    lambda b: b + bytes(4),
    lambda b: b"INVALID!" + b[8:],
    lambda b: b[:28] + struct.pack("<I", 2) + b[32:],
    lambda b: b[:8] + struct.pack("<I", 2**32 - 1) + b[12:],
    lambda b: b[:24] + struct.pack("<I", 8) + b[28:],
    lambda b: b[:16] + struct.pack("<I", 64) + b[20:],
    lambda b: b[:20] + struct.pack("<I", 15) + b[24:],
    lambda b: b[:34] + bytes([b[34] | 128]) + b[35:],
    lambda b: b[:37] + bytes([b[37] ^ 128]) + b[38:],
    lambda b: b[:227] + bytes([b[227] ^ 1]) + b[228:],
])
def test_malformed_container_rejected(change):
    with pytest.raises(ValueError):
        RowTrellisTensor.from_bytes(change(example_tensor().to_bytes()))


def test_invalid_column_codes_and_prefix():
    tensor = example_tensor()
    with pytest.raises(ValueError, match="4-bit"):
        RowTrellisTensor(
            tensor.columns, tensor.group_size, tensor.state_bits, tensor.descriptors,
            tensor.blocks, tensor.payload, np.full(tensor.columns, 16, np.uint8),
        )
    descriptors = tensor.descriptors.copy()
    descriptors[31, 2] ^= 1
    with pytest.raises(ValueError, match="prefix"):
        RowTrellisTensor(
            tensor.columns, tensor.group_size, tensor.state_bits, descriptors,
            tensor.blocks, tensor.payload,
        )


def test_container_is_little_endian():
    tensor = example_tensor()
    swapped = RowTrellisTensor(
        tensor.columns, tensor.group_size, tensor.state_bits, tensor.descriptors,
        tensor.blocks.astype(">u8"), tensor.payload.astype(">u4"), tensor.column_codes,
    )
    assert swapped.to_bytes() == tensor.to_bytes()
