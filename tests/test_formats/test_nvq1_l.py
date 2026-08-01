from __future__ import annotations

import numpy as np

from mfq.formats.nvq1_l import (
    IQ1S_TERNARY_2048,
    NVQ1_L_T8_S3,
    NVQ1_L_T8_S4,
    Nvq1LTensor,
    pack_nvq1_l,
    unpack_nvq1_l,
)
from mfq.formats.ternary import (
    NEURON_TERNARY_S3,
    NeuronTernaryTensor,
    pack_neuron_ternary,
    pack_trits,
    unpack_neuron_ternary,
    unpack_trits,
)


def test_iq1s_grid_shape_alphabet_and_uniqueness():
    assert IQ1S_TERNARY_2048.shape == (2048, 8)
    assert set(np.unique(IQ1S_TERNARY_2048)) == {-1, 0, 1}
    assert np.unique(IQ1S_TERNARY_2048, axis=0).shape[0] == 2048


def test_nvq1_l_and_ternary_actual_bpw_at_qwen_width():
    assert NVQ1_L_T8_S3.bpw(5120, out=4096) == 1.5453125
    assert NVQ1_L_T8_S4.bpw(5120, out=4096) == 1.587109375
    assert NEURON_TERNARY_S3.bpw(5120, out=4096) == 1.728515625


def test_nvq1_l_blob_roundtrip():
    rng = np.random.default_rng(30)
    out, neuron_len = 3, 80
    tensor = Nvq1LTensor(
        spec=NVQ1_L_T8_S3,
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.random(out, dtype=np.float32),
        sub_scale=rng.integers(0, 8, size=(out, 4), dtype=np.uint8),
        indices=rng.integers(0, 2048, size=(out, 10), dtype=np.uint16),
        delta_sign=rng.integers(0, 2, size=(out, 4), dtype=np.uint8),
    )
    payload = pack_nvq1_l(tensor)
    assert payload[:4] == b"NQ1L"
    restored = unpack_nvq1_l(payload)
    assert restored.spec == tensor.spec
    assert restored.shape == tensor.shape
    np.testing.assert_array_equal(restored.sub_scale, tensor.sub_scale)
    np.testing.assert_array_equal(restored.indices, tensor.indices)
    np.testing.assert_array_equal(restored.delta_sign, tensor.delta_sign)
    np.testing.assert_array_equal(
        restored.neuron_scale,
        tensor.neuron_scale.astype(np.float16).astype(np.float32),
    )

def test_custom_nvq1_l_codebook_roundtrip():
    custom = np.roll(IQ1S_TERNARY_2048, 1, axis=0).copy()
    tensor = Nvq1LTensor(
        spec=NVQ1_L_T8_S3,
        shape=(1, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.ones(1, dtype=np.float32),
        sub_scale=np.ones((1, 1), dtype=np.uint8),
        indices=np.asarray([[0, 1, 2]], dtype=np.uint16),
        delta_sign=np.zeros((1, 1), dtype=np.uint8),
        codebook=custom,
    )
    restored = unpack_nvq1_l(pack_nvq1_l(tensor))
    np.testing.assert_array_equal(restored.codebook, custom)
    assert restored.payload_nbytes == NVQ1_L_T8_S3.payload_nbytes(1, 24) + 4096


def test_base3_trit_stream_and_tensor_roundtrip():
    rng = np.random.default_rng(31)
    symbols = rng.integers(0, 3, size=37, dtype=np.uint8)
    packed = pack_trits(symbols)
    restored, off = unpack_trits(packed, 0, symbols.size)
    assert off == len(packed)
    np.testing.assert_array_equal(restored, symbols)

    out, neuron_len = 2, 29
    tensor = NeuronTernaryTensor(
        spec=NEURON_TERNARY_S3,
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.random(out, dtype=np.float32),
        sub_scale=rng.integers(0, 8, size=(out, 2), dtype=np.uint8),
        trits=rng.integers(0, 3, size=(out, neuron_len), dtype=np.uint8),
    )
    decoded = unpack_neuron_ternary(pack_neuron_ternary(tensor))
    np.testing.assert_array_equal(decoded.sub_scale, tensor.sub_scale)
    np.testing.assert_array_equal(decoded.trits, tensor.trits)
