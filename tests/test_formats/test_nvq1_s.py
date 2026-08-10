from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.nvq1_s import (
    NVQ1_S,
    NVQ1_S_BOOTSTRAP_512,
    NVQ1_S_BOOTSTRAP_BANKS,
    Nvq1STensor,
    pack_nvq1_s,
    pack_nvq1_s_banked_codebook,
    pack_nvq1_s_codebook,
    unpack_nvq1_s,
    unpack_nvq1_s_banked_codebook,
    unpack_nvq1_s_codebook,
)


def test_nvq1_s_physical_bpw_includes_anchor_and_tails():
    assert NVQ1_S.bpw(4096, out=4096) == 1.338623046875
    assert NVQ1_S.bpw(5120, out=4096) == 1.337890625
    assert NVQ1_S.bpw(11008, out=4096) == 1.3353015988372092


def test_nvq1_s_codebook_is_exactly_1024_bytes():
    payload = pack_nvq1_s_codebook(NVQ1_S_BOOTSTRAP_512)
    assert len(payload) == 1024
    np.testing.assert_array_equal(
        unpack_nvq1_s_codebook(payload),
        NVQ1_S_BOOTSTRAP_512,
    )
    banked = pack_nvq1_s_banked_codebook(NVQ1_S_BOOTSTRAP_BANKS)
    assert len(banked) == 2048
    np.testing.assert_array_equal(
        unpack_nvq1_s_banked_codebook(banked),
        NVQ1_S_BOOTSTRAP_BANKS,
    )


def test_nvq1_s_blob_roundtrip_for_both_tail_shapes():
    rng = np.random.default_rng(40)
    for neuron_len in (72, 80, 88):
        out = 3
        ng = (neuron_len + 23) // 24
        tensor = Nvq1STensor(
            spec=NVQ1_S,
            shape=(out, neuron_len),
            axis=0,
            neuron_len=neuron_len,
            neuron_scale=rng.random(out, dtype=np.float32),
            sub_scale=rng.integers(0, 16, size=(out, ng), dtype=np.uint8),
            indices=rng.integers(
                0,
                512,
                size=(out, neuron_len // 8),
                dtype=np.uint16,
            ),
            delta_sign=rng.integers(0, 2, size=(out, ng), dtype=np.uint8),
            codebook=NVQ1_S_BOOTSTRAP_BANKS,
        )
        payload = pack_nvq1_s(tensor)
        assert payload[:4] == b"NQ1S"
        restored = unpack_nvq1_s(payload)
        assert restored.spec == NVQ1_S
        assert restored.shape == tensor.shape
        np.testing.assert_array_equal(restored.sub_scale, tensor.sub_scale)
        np.testing.assert_array_equal(restored.indices, tensor.indices)
        np.testing.assert_array_equal(restored.delta_sign, tensor.delta_sign)
        np.testing.assert_array_equal(restored.codebook, NVQ1_S_BOOTSTRAP_BANKS)
        np.testing.assert_array_equal(
            restored.neuron_scale,
            tensor.neuron_scale.astype(np.float16).astype(np.float32),
        )


def test_nvq1_s_rejects_fp16_anchor_overflow() -> None:
    tensor = Nvq1STensor(
        spec=NVQ1_S,
        shape=(1, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([1e10], dtype=np.float32),
        sub_scale=np.ones((1, 1), dtype=np.uint8),
        indices=np.asarray([[0, 1, 2]], dtype=np.uint16),
        delta_sign=np.zeros((1, 1), dtype=np.uint8),
        codebook=NVQ1_S_BOOTSTRAP_BANKS,
    )
    with pytest.raises(ValueError, match="FP16"):
        pack_nvq1_s(tensor)
