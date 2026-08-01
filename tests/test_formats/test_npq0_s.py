from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.npq0_s import (
    NPQ0_S,
    NPQ0_S_TABLE_BYTES,
    Npq0STensor,
    pack_npq0_s,
    pack_npq0_s_tables,
    unpack_npq0_s,
    unpack_npq0_s_tables,
)


def _tensor(neuron_len: int) -> Npq0STensor:
    rng = np.random.default_rng(20260723 + neuron_len)
    out = 3
    ng = (neuron_len + 23) // 24
    nvec = neuron_len // 8
    return Npq0STensor(
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.random(out, dtype=np.float32),
        scale_lut=np.linspace(0.25, 1.0, 4, dtype=np.float32),
        state=rng.integers(0, 4, size=(out, ng), dtype=np.uint8),
        indices=rng.integers(0, 64, size=(out, nvec), dtype=np.uint8),
        first_codebooks=rng.integers(
            -127,
            128,
            size=(4, 8, 4),
            dtype=np.int16,
        ).astype(np.int8),
        second_codebooks=rng.integers(
            -127,
            128,
            size=(4, 8, 4),
            dtype=np.int16,
        ).astype(np.int8),
    )


def test_npq0_s_physical_bpw_is_below_one_bit() -> None:
    assert NPQ0_S_TABLE_BYTES == 320
    assert NPQ0_S.bpw(4096, out=4096) == 0.837554931640625
    assert NPQ0_S.bpw(5120, out=4096) == 0.8368408203125


def test_npq0_s_table_roundtrip() -> None:
    tensor = _tensor(24)
    payload = pack_npq0_s_tables(
        tensor.scale_lut,
        tensor.first_codebooks,
        tensor.second_codebooks,
    )
    assert len(payload) == NPQ0_S_TABLE_BYTES
    scale, first, second, consumed = unpack_npq0_s_tables(payload)
    assert consumed == len(payload)
    np.testing.assert_array_equal(scale, tensor.scale_lut.astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(first, tensor.first_codebooks)
    np.testing.assert_array_equal(second, tensor.second_codebooks)


def test_npq0_s_blob_roundtrip_without_k_padding() -> None:
    for neuron_len in (24, 32, 40):
        tensor = _tensor(neuron_len)
        payload = pack_npq0_s(tensor)
        assert payload[:4] == b"NPQS"
        restored = unpack_npq0_s(payload)
        assert restored.shape == tensor.shape
        assert restored.axis == tensor.axis
        assert restored.neuron_len == tensor.neuron_len
        assert restored.payload_nbytes == tensor.payload_nbytes
        np.testing.assert_array_equal(restored.state, tensor.state)
        np.testing.assert_array_equal(restored.indices, tensor.indices)
        np.testing.assert_array_equal(restored.first_codebooks, tensor.first_codebooks)
        np.testing.assert_array_equal(restored.second_codebooks, tensor.second_codebooks)
        np.testing.assert_array_equal(
            restored.neuron_scale,
            tensor.neuron_scale.astype(np.float16).astype(np.float32),
        )


def test_npq0_s_rejects_the_old_vq64_profile() -> None:
    tensor = _tensor(24)
    table = bytearray(
        pack_npq0_s_tables(
            tensor.scale_lut,
            tensor.first_codebooks,
            tensor.second_codebooks,
        )
    )
    table[0] = 1
    with pytest.raises(ValueError, match="unsupported NPQ0-S table profile"):
        unpack_npq0_s_tables(table)

    blob = bytearray(pack_npq0_s(tensor))
    blob[4] = 1
    with pytest.raises(ValueError, match="invalid or unsupported NPQ0-S header"):
        unpack_npq0_s(blob)
