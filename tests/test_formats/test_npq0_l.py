from __future__ import annotations

import numpy as np

from mfq.formats.npq0_l import (
    NPQ0_L,
    NPQ0_L_TABLE_BYTES,
    Npq0LTensor,
    pack_npq0_l,
    pack_npq0_l_tables,
    unpack_npq0_l,
    unpack_npq0_l_tables,
)


def _tensor(neuron_len: int) -> Npq0LTensor:
    rng = np.random.default_rng(20260722 + neuron_len)
    out = 3
    ng = (neuron_len + 23) // 24
    nvec = neuron_len // 8
    return Npq0LTensor(
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.random(out, dtype=np.float32),
        scale_lut=np.linspace(0.125, 1.0, 8, dtype=np.float32),
        state=rng.integers(0, 8, size=(out, ng), dtype=np.uint8),
        indices=rng.integers(0, 128, size=(out, nvec), dtype=np.uint8),
        first_codebooks=rng.integers(-127, 128, size=(8, 8, 4), dtype=np.int16).astype(np.int8),
        second_codebooks=rng.integers(
            -127,
            128,
            size=(8, 16, 4),
            dtype=np.int16,
        ).astype(np.int8),
    )


def test_npq0_l_physical_bpw_is_about_one_bit() -> None:
    assert NPQ0_L_TABLE_BYTES == 832
    assert NPQ0_L.bpw(4096, out=4096) == 1.004547119140625
    assert NPQ0_L.bpw(5120, out=4096) == 1.0038330078125
    assert NPQ0_L.bpw(11008, out=4096) == 1.0016919513081395


def test_npq0_l_table_roundtrip() -> None:
    tensor = _tensor(24)
    payload = pack_npq0_l_tables(
        tensor.scale_lut,
        tensor.first_codebooks,
        tensor.second_codebooks,
    )
    assert len(payload) == NPQ0_L_TABLE_BYTES
    scale, first, second, consumed = unpack_npq0_l_tables(payload)
    assert consumed == len(payload)
    np.testing.assert_array_equal(scale, tensor.scale_lut.astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(first, tensor.first_codebooks)
    np.testing.assert_array_equal(second, tensor.second_codebooks)


def test_npq0_l_blob_roundtrip_without_k_padding() -> None:
    for neuron_len in (24, 32, 40):
        tensor = _tensor(neuron_len)
        payload = pack_npq0_l(tensor)
        assert payload[:4] == b"NPQL"
        restored = unpack_npq0_l(payload)
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
