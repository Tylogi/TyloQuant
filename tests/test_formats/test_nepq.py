import math

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.header import FileHeader
from mfq.formats.nepq import (
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_L,
    NEPQ1_S,
    NepqTensor,
    dequantize_nepq,
    pack_nepq,
    unpack_nepq,
)
from mfq.formats.npq0_l import Npq0LTensor, pack_npq0_l_tables, unpack_npq0_l_tables
from mfq.formats.npq0_s import Npq0STensor, pack_npq0_s_tables, unpack_npq0_s_tables
from mfq.formats.nvq1_l import (
    IQ1S_TERNARY_2048,
    NVQ1_L_T8_S3,
    Nvq1LTensor,
    pack_ternary_codebook,
    unpack_ternary_codebook,
)
from mfq.formats.nvq1_s import (
    NVQ1_S,
    NVQ1_S_SYNTHETIC_BANKS,
    Nvq1STensor,
    pack_nvq1_s_banked_codebook,
    unpack_nvq1_s_banked_codebook,
)
from mfq.quantize.npq0_l import dequantize_npq0_l
from mfq.quantize.npq0_s import dequantize_npq0_s
from mfq.quantize.nvq1_l_quant import dequantize as dequantize_nvq1_l
from mfq.quantize.nvq1_s_quant import dequantize as dequantize_nvq1_s


def _npq0_s_table(offset: int) -> bytes:
    scale = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    first = np.arange(4 * 8 * 4, dtype=np.int16).reshape(4, 8, 4)
    second = np.flip(first, axis=1).copy()
    first = ((first + offset) % 17 - 8).astype(np.int8)
    second = ((second + 2 * offset) % 19 - 9).astype(np.int8)
    return pack_npq0_s_tables(scale, first, second)


def _npq0_l_table(offset: int) -> bytes:
    scale = np.linspace(0.125, 1.0, 8, dtype=np.float32)
    first = np.arange(8 * 8 * 4, dtype=np.int16).reshape(8, 8, 4)
    second = np.arange(8 * 16 * 4, dtype=np.int16).reshape(8, 16, 4)
    first = ((first + offset) % 17 - 8).astype(np.int8)
    second = ((second + 2 * offset) % 19 - 9).astype(np.int8)
    return pack_npq0_l_tables(scale, first, second)


def _tables(spec):
    if spec is NEPQ0_S:
        return np.stack(
            [np.frombuffer(_npq0_s_table(i), dtype=np.uint8) for i in range(2)]
        )
    if spec is NEPQ0_L:
        return np.stack(
            [np.frombuffer(_npq0_l_table(i), dtype=np.uint8) for i in range(2)]
        )
    if spec is NEPQ1_S:
        return np.stack(
            [
                np.frombuffer(
                    pack_nvq1_s_banked_codebook(
                        np.roll(NVQ1_S_SYNTHETIC_BANKS, i, axis=1)
                    ),
                    dtype=np.uint8,
                )
                for i in range(2)
            ]
        )
    return np.stack(
        [
            np.frombuffer(
                pack_ternary_codebook(np.roll(IQ1S_TERNARY_2048, i, axis=0)),
                dtype=np.uint8,
            )
            for i in range(2)
        ]
    )


def _tensor(spec) -> NepqTensor:
    rng = np.random.default_rng(20260723 + spec.profile_id)
    shape = (2, 3, 104)
    ng = math.ceil(shape[2] / 24)
    nvec = shape[2] // 8
    nsuper = math.ceil(ng / 4)
    selected = (np.arange(shape[0] * shape[1]) % 2).reshape(shape[:2])
    bank_ids = np.repeat(selected[:, :, None], nsuper, axis=2).astype(np.uint8)
    return NepqTensor(
        spec=spec,
        shape=shape,
        neuron_scale=rng.uniform(0.01, 0.2, size=shape[:2]).astype(np.float32),
        state=rng.integers(
            0, 1 << spec.state_bits, size=shape[:2] + (ng,), dtype=np.uint8
        ),
        indices=rng.integers(
            0, 1 << spec.index_bits, size=shape[:2] + (nvec,), dtype=np.uint16
        ),
        aux=(
            rng.integers(0, 2, size=shape[:2] + (ng,), dtype=np.uint8)
            if spec.aux_bits
            else None
        ),
        bank_ids=bank_ids,
        table_payloads=_tables(spec),
        rotation_block=8,
        rotation_seed=18601311049,
    )


def _base_row(tensor: NepqTensor, expert: int, row: int):
    bank = int(tensor.bank_ids[expert, row, 0])
    payload = tensor.table_payloads[bank].tobytes()
    common = {
        "shape": (1, tensor.neuron_len),
        "axis": 0,
        "neuron_len": tensor.neuron_len,
        "neuron_scale": tensor.neuron_scale[expert, row : row + 1],
    }
    if tensor.spec is NEPQ0_S:
        scale, first, second, _ = unpack_npq0_s_tables(payload)
        base = Npq0STensor(
            **common,
            scale_lut=scale,
            state=tensor.state[expert, row : row + 1],
            indices=tensor.indices[expert, row : row + 1],
            first_codebooks=first,
            second_codebooks=second,
        )
        return dequantize_npq0_s(base)[0]
    if tensor.spec is NEPQ0_L:
        scale, first, second, _ = unpack_npq0_l_tables(payload)
        base = Npq0LTensor(
            **common,
            scale_lut=scale,
            state=tensor.state[expert, row : row + 1],
            indices=tensor.indices[expert, row : row + 1],
            first_codebooks=first,
            second_codebooks=second,
        )
        return dequantize_npq0_l(base)[0]
    if tensor.spec is NEPQ1_S:
        base = Nvq1STensor(
            spec=NVQ1_S,
            **common,
            sub_scale=tensor.state[expert, row : row + 1],
            indices=tensor.indices[expert, row : row + 1],
            delta_sign=tensor.aux[expert, row : row + 1],
            codebook=unpack_nvq1_s_banked_codebook(payload),
        )
        return dequantize_nvq1_s(base)[0]
    base = Nvq1LTensor(
        spec=NVQ1_L_T8_S3,
        **common,
        sub_scale=tensor.state[expert, row : row + 1],
        indices=tensor.indices[expert, row : row + 1],
        delta_sign=tensor.aux[expert, row : row + 1],
        codebook=unpack_ternary_codebook(payload),
    )
    return dequantize_nvq1_l(base)[0]


@pytest.mark.parametrize("spec", [NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L])
def test_nepq_roundtrip_and_reference_decode(spec):
    tensor = _tensor(spec)
    payload = pack_nepq(tensor)
    restored = unpack_nepq(payload)
    assert restored.spec is spec
    assert restored.shape == tensor.shape
    assert restored.rotation_block == 8
    assert restored.rotation_seed == 18601311049
    np.testing.assert_array_equal(restored.state, tensor.state)
    np.testing.assert_array_equal(restored.indices, tensor.indices)
    np.testing.assert_array_equal(restored.bank_ids, tensor.bank_ids)
    np.testing.assert_array_equal(restored.table_payloads, tensor.table_payloads)
    if spec.aux_bits:
        np.testing.assert_array_equal(restored.aux, tensor.aux)
    else:
        assert restored.aux is None
    np.testing.assert_array_equal(
        restored.neuron_scale, tensor.neuron_scale.astype(np.float16).astype(np.float32)
    )

    decoded = dequantize_nepq(restored)
    for expert in range(tensor.n_experts):
        for row in range(tensor.out_per_expert):
            np.testing.assert_allclose(
                decoded[expert, row], _base_row(restored, expert, row), rtol=0, atol=0
            )


def test_nepq0_s_physical_rate_matches_supergroup_layout():
    expected = 0.8361409505208334 + 8.0 / 96.0
    assert NEPQ0_S.bpw(
        6144, n_experts=256, out_per_expert=2048, bank_count=256
    ) == pytest.approx(expected, rel=0, abs=1e-15)


def test_nepq_rejects_out_of_pool_bank_id():
    tensor = _tensor(NEPQ0_S)
    tensor.bank_ids[0, 0, 0] = 2
    with pytest.raises(ValueError, match="bank ID exceeds"):
        pack_nepq(tensor)


def test_nepq_roundtrip_supports_full_uint8_bank_space():
    tensor = _tensor(NEPQ0_S)
    tensor.table_payloads = np.repeat(tensor.table_payloads[:1], 256, axis=0)
    tensor.bank_ids[0, 0, 0] = 0
    tensor.bank_ids[0, 0, 1] = 255
    restored = unpack_nepq(pack_nepq(tensor))
    assert restored.bank_count == 256
    assert int(restored.bank_ids[0, 0, 0]) == 0
    assert int(restored.bank_ids[0, 0, 1]) == 255


def test_nepq_file_and_mmap_roundtrip(tmp_path):
    tensor = _tensor(NEPQ0_S)
    canonical = unpack_nepq(pack_nepq(tensor))
    path = tmp_path / "cross-expert.mfq"
    io.save(
        path,
        FileHeader(model_arch="nepq-test", num_tensors=1),
        {"blk.0.ffn_gate_exps.weight": tensor},
    )
    _, loaded = io.load(path)
    restored = loaded["blk.0.ffn_gate_exps.weight"]
    np.testing.assert_array_equal(restored.bank_ids, tensor.bank_ids)
    _, store = io.load_mmap(path)
    try:
        assert store.records["blk.0.ffn_gate_exps.weight"].dtype == "NEPQ0-S"
        lazy = store["blk.0.ffn_gate_exps.weight"]
        np.testing.assert_allclose(
            dequantize_nepq(lazy), dequantize_nepq(canonical), rtol=0, atol=0
        )
    finally:
        store.close()
