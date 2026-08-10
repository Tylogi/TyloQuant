from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.nvq import (
    D4_256,
    D4_512,
    D4_1024,
    E8_256,
    E8_1024,
    E8_4096,
    NVQ2_E8,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_512,
    NVQ3_D4_1024,
    NvqJscTensor,
    NvqSpec,
    NvqTensor,
    pack_nvq,
    unpack_nvq,
)


def test_llama_codebooks_have_expected_shape_and_alphabet():
    assert E8_256.shape == (256, 8)
    assert D4_256.shape == (256, 4)
    assert D4_512.shape == (512, 4)
    assert E8_1024.shape == (1024, 8)
    assert E8_4096.shape == (4096, 8)
    assert D4_1024.shape == (1024, 4)
    assert set(np.unique(E8_256)).issubset({1, 3, 5, 7})
    assert set(np.unique(D4_256)).issubset(set(range(1, 16, 2)))
    assert set(np.unique(D4_512)).issubset(set(range(1, 16, 2)))
    assert np.unique(E8_1024, axis=0).shape[0] == 1024
    assert np.unique(E8_4096, axis=0).shape[0] == 4096
    assert np.unique(D4_1024, axis=0).shape[0] == 1024
    np.testing.assert_array_equal(E8_1024[:256], E8_256)
    np.testing.assert_array_equal(E8_4096[:1024], E8_1024)
    np.testing.assert_array_equal(D4_1024[:512], D4_512)


def test_nvq_actual_bpw_at_qwen_width():
    assert NVQ2_E8.bpw(5120, out=4096) == 2.0453125
    assert NVQ3_D4.bpw(5120, out=4096) == 3.0453125
    assert NVQ3_D4_512.bpw(5120, out=4096) == 3.2953125
    assert NVQ2_E8_1024.bpw(5120, out=4096) == 2.2953125
    assert NVQ2_E8_4096.bpw(5120, out=4096) == 2.5453125
    assert NVQ3_D4_1024.bpw(5120, out=4096) == 3.5453125


def test_extended_nvq_jsc_blob_roundtrip():
    rng = np.random.default_rng(20260730)
    for spec, base in (
        (NVQ2_E8_1024, E8_1024),
        (NVQ2_E8_4096, E8_4096),
        (NVQ3_D4_1024, D4_1024),
    ):
        out, neuron_len = 2, 48
        ng = 2
        nvec = neuron_len // spec.vector_size
        nsign = neuron_len // 8
        tensor = NvqJscTensor(
            shape=(out, neuron_len),
            axis=0,
            neuron_len=neuron_len,
            neuron_scale=rng.uniform(0.001, 0.01, size=out).astype(np.float32),
            scale_lut=np.arange(16, dtype=np.float32),
            bank_for_state=np.zeros(16, dtype=np.uint8),
            state=rng.integers(0, 16, size=(out, ng), dtype=np.uint8),
            indices=rng.integers(
                0,
                spec.codebook_entries,
                size=(out, nvec),
                dtype=np.uint16,
            ),
            signs=rng.integers(0, 128, size=(out, nsign), dtype=np.uint8),
            codebooks=(base[None].astype(np.int16) * 8).astype(np.int8),
            base_spec=spec,
        )
        restored = unpack_nvq(pack_nvq(tensor))
        assert isinstance(restored, NvqJscTensor)
        assert restored.spec == spec
        np.testing.assert_array_equal(restored.state, tensor.state)
        np.testing.assert_array_equal(restored.indices, tensor.indices)
        np.testing.assert_array_equal(restored.signs, tensor.signs)
        np.testing.assert_array_equal(restored.codebooks, tensor.codebooks)


def test_nvq_blob_roundtrip():
    rng = np.random.default_rng(5)
    out, neuron_len = 3, 80
    ng = 4
    tensor = NvqTensor(
        spec=NVQ2_E8,
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.random(out, dtype=np.float32),
        sub_scale=rng.integers(0, 16, size=(out, ng), dtype=np.uint8),
        indices=rng.integers(0, 256, size=(out, 10), dtype=np.uint8),
        signs=rng.integers(0, 128, size=(out, 10), dtype=np.uint8),
    )
    restored = unpack_nvq(pack_nvq(tensor))
    assert restored.spec == tensor.spec
    assert restored.shape == tensor.shape
    np.testing.assert_array_equal(restored.sub_scale, tensor.sub_scale)
    np.testing.assert_array_equal(restored.indices, tensor.indices)
    np.testing.assert_array_equal(restored.signs, tensor.signs)
    np.testing.assert_array_equal(restored.neuron_scale, tensor.neuron_scale.astype(np.float16).astype(np.float32))


def test_nvq_rejects_fp16_anchor_overflow() -> None:
    tensor = NvqTensor(
        spec=NVQ2_E8,
        shape=(1, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([1e10], dtype=np.float32),
        sub_scale=np.asarray([[1]], dtype=np.uint8),
        indices=np.asarray([[1, 2, 3]], dtype=np.uint8),
        signs=np.asarray([[1, 2, 3]], dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="FP16"):
        pack_nvq(tensor)


def test_legacy_niq_blob_decodes_identically():
    tensor = NvqTensor(
        spec=NVQ2_E8,
        shape=(1, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.25], dtype=np.float32),
        sub_scale=np.asarray([[7]], dtype=np.uint8),
        indices=np.asarray([[1, 2, 3]], dtype=np.uint8),
        signs=np.asarray([[4, 5, 6]], dtype=np.uint8),
    )
    nvq_payload = pack_nvq(tensor)
    niq_payload = b"NIQ" + nvq_payload[3:]
    current = unpack_nvq(nvq_payload)
    legacy = unpack_nvq(niq_payload)
    np.testing.assert_array_equal(legacy.neuron_scale, current.neuron_scale)
    np.testing.assert_array_equal(legacy.sub_scale, current.sub_scale)
    np.testing.assert_array_equal(legacy.indices, current.indices)
    np.testing.assert_array_equal(legacy.signs, current.signs)


def test_custom_nvq_codebook_roundtrip():
    custom = np.roll(E8_256, 1, axis=0).copy()
    tensor = NvqTensor(
        spec=NVQ2_E8,
        shape=(1, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.ones(1, dtype=np.float32),
        sub_scale=np.ones((1, 1), dtype=np.uint8),
        indices=np.asarray([[0, 1, 2]], dtype=np.uint8),
        signs=np.zeros((1, 3), dtype=np.uint8),
        codebook=custom,
    )
    payload = pack_nvq(tensor)
    restored = unpack_nvq(payload)
    np.testing.assert_array_equal(restored.codebook, custom)
    assert restored.payload_nbytes == NVQ2_E8.payload_nbytes(1, 24) + 512


def test_index_parity_sign_mode_survives_blob_roundtrip():
    spec = NvqSpec("e8_256", groupsize=24, sub_bits=4, sign_mode="index_parity")
    tensor = NvqTensor(
        spec=spec,
        shape=(1, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.ones(1, dtype=np.float32),
        sub_scale=np.ones((1, 1), dtype=np.uint8),
        indices=np.asarray([[0, 128, 1]], dtype=np.uint8),
        signs=np.asarray([[0, 0, 0]], dtype=np.uint8),
    )
    restored = unpack_nvq(pack_nvq(tensor))
    assert restored.spec.sign_mode == "index_parity"
    np.testing.assert_array_equal(restored.indices, tensor.indices)


def test_nvq_jsc_blob_roundtrip_and_metadata_accounting():
    codebooks = np.stack(
        (E8_256.astype(np.int16) * 8, np.roll(E8_256, 1, axis=0).astype(np.int16) * 8)
    ).astype(np.int8)
    tensor = NvqJscTensor(
        shape=(3, 50),
        axis=0,
        neuron_len=50,
        neuron_scale=np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
        scale_lut=np.linspace(0.0, 15.0, 16, dtype=np.float32),
        bank_for_state=(np.arange(16, dtype=np.uint8) & 1),
        state=np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8),
        indices=np.arange(21, dtype=np.uint8).reshape(3, 7),
        signs=(np.arange(21, dtype=np.uint8).reshape(3, 7) * 3) & 0x7F,
        codebooks=codebooks,
    )
    payload = pack_nvq(tensor)
    restored = unpack_nvq(payload)
    assert isinstance(restored, NvqJscTensor)
    assert restored.shape == tensor.shape
    assert restored.axis == 0
    assert restored.payload_nbytes == NVQ2_E8.payload_nbytes(3, 50) + 64 + 2 * 256 * 8
    np.testing.assert_array_equal(restored.neuron_scale, tensor.neuron_scale.astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(restored.scale_lut, tensor.scale_lut.astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(restored.bank_for_state, tensor.bank_for_state)
    np.testing.assert_array_equal(restored.state, tensor.state)
    np.testing.assert_array_equal(restored.indices, tensor.indices)
    np.testing.assert_array_equal(restored.signs, tensor.signs)
    np.testing.assert_array_equal(restored.codebooks, tensor.codebooks)


def test_nvq3j_blob_roundtrip_and_metadata_accounting():
    codebooks = np.stack(
        (D4_256.astype(np.int16) * 8, np.roll(D4_256, 1, axis=0).astype(np.int16) * 8)
    ).astype(np.int8)
    tensor = NvqJscTensor(
        shape=(3, 50),
        axis=0,
        neuron_len=50,
        neuron_scale=np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
        scale_lut=np.linspace(0.0, 15.0, 16, dtype=np.float32),
        bank_for_state=(np.arange(16, dtype=np.uint8) & 1),
        state=np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8),
        indices=np.arange(39, dtype=np.uint8).reshape(3, 13),
        signs=(np.arange(21, dtype=np.uint8).reshape(3, 7) * 3) & 0x7F,
        codebooks=codebooks,
        base_spec=NVQ3_D4,
    )
    restored = unpack_nvq(pack_nvq(tensor))
    assert isinstance(restored, NvqJscTensor)
    assert restored.spec == NVQ3_D4
    assert restored.payload_nbytes == NVQ3_D4.payload_nbytes(3, 50) + 64 + 2 * 256 * 4
    np.testing.assert_array_equal(restored.state, tensor.state)
    np.testing.assert_array_equal(restored.indices, tensor.indices)
    np.testing.assert_array_equal(restored.signs, tensor.signs)
    np.testing.assert_array_equal(restored.codebooks, tensor.codebooks)


def test_nvq3j_512_blob_roundtrip_and_metadata_accounting():
    codebooks = np.stack(
        (
            D4_512.astype(np.int16) * 8,
            np.roll(D4_512, 1, axis=0).astype(np.int16) * 8,
        )
    ).astype(np.int8)
    tensor = NvqJscTensor(
        shape=(3, 50),
        axis=0,
        neuron_len=50,
        neuron_scale=np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
        scale_lut=np.linspace(0.0, 15.0, 16, dtype=np.float32),
        bank_for_state=(np.arange(16, dtype=np.uint8) & 1),
        state=np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8),
        indices=np.arange(39, dtype=np.uint16).reshape(3, 13) + 300,
        signs=(np.arange(21, dtype=np.uint8).reshape(3, 7) * 3) & 0x7F,
        codebooks=codebooks,
        base_spec=NVQ3_D4_512,
    )
    restored = unpack_nvq(pack_nvq(tensor))
    assert isinstance(restored, NvqJscTensor)
    assert restored.spec == NVQ3_D4_512
    assert (
        restored.payload_nbytes
        == NVQ3_D4_512.payload_nbytes(3, 50) + 64 + 2 * 512 * 4
    )
    np.testing.assert_array_equal(restored.state, tensor.state)
    np.testing.assert_array_equal(restored.indices, tensor.indices)
    np.testing.assert_array_equal(restored.signs, tensor.signs)
    np.testing.assert_array_equal(restored.codebooks, tensor.codebooks)
