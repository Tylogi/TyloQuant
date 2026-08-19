"""NINT serialization round-trip tests for formats.io."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.header import FileHeader
from mfq.formats.nint import NintSpec
from mfq.formats.npq0_l import Npq0LTensor
from mfq.formats.npq0_s import Npq0STensor
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
    NvqTensor,
)
from mfq.formats.nvq1_l import NVQ1_L_T8_S3, Nvq1LTensor
from mfq.formats.nvq1_s import NVQ1_S, NVQ1_S_BOOTSTRAP_BANKS, Nvq1STensor
from mfq.quantize import nint_quant
from mfq.quantize.expert_nint import dequantize_expertwise, quantize_expertwise


def _extended_jsc_tensor(spec, base, seed: int) -> NvqJscTensor:
    rng = np.random.default_rng(seed)
    rows, neuron_len = 2, 48
    return NvqJscTensor(
        shape=(rows, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.uniform(0.001, 0.01, rows).astype(np.float32),
        scale_lut=np.arange(16, dtype=np.float32),
        bank_for_state=np.zeros(16, dtype=np.uint8),
        state=rng.integers(0, 16, (rows, 2), dtype=np.uint8),
        indices=rng.integers(
            0,
            spec.codebook_entries,
            (rows, neuron_len // spec.vector_size),
            dtype=np.uint16,
        ),
        signs=rng.integers(0, 128, (rows, neuron_len // 8), dtype=np.uint8),
        codebooks=(base[None].astype(np.int16) * 8).astype(np.int8),
        base_spec=spec,
    )


def _mk(seed: int = 0, shape: tuple[int, ...] = (8, 96)) -> nint_quant.NintTensor:
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.05, size=shape).astype(np.float32)
    return nint_quant.quantize(W, NintSpec(4, 24, 6), axis=0)


def test_pack_roundtrip_fields():
    t = _mk()
    t2 = io.unpack_nint(io.pack_nint(t))
    assert t2.spec == t.spec
    assert t2.shape == t.shape and t2.axis == t.axis and t2.neuron_len == t.neuron_len
    np.testing.assert_array_equal(t2.q, t.q)
    np.testing.assert_array_equal(t2.sub_scale, t.sub_scale)
    np.testing.assert_array_equal(t2.sub_min, t.sub_min)
    np.testing.assert_allclose(t2.neuron_scale, t.neuron_scale)
    np.testing.assert_allclose(t2.neuron_min, t.neuron_min)


@pytest.mark.parametrize("bits", range(1, 8))
@pytest.mark.parametrize("count", (0, 1, 7, 8, 9, 31, 257))
def test_pack_bits_matches_little_endian_reference(bits: int, count: int) -> None:
    rng = np.random.default_rng(bits * 1000 + count)
    values = rng.integers(0, 1 << bits, count, dtype=np.uint8)
    bit_rows = np.unpackbits(values[:, None], axis=1, bitorder="little")[:, :bits]
    expected = np.packbits(bit_rows.reshape(-1), bitorder="little").tobytes()

    packed = io.pack_bits(values, bits)

    assert packed == expected


def test_pack_roundtrip_dequant():
    t = _mk(5)
    t2 = io.unpack_nint(io.pack_nint(t))
    np.testing.assert_allclose(nint_quant.dequantize(t2), nint_quant.dequantize(t))


def test_dense_bfloat16_roundtrip_preserves_raw_bits():
    bits = np.asarray(
        [[0x3F80, 0xC020], [0x0000, 0x7F80]],
        dtype="<u2",
    ).view(io.BFloat16Array)

    dtype, blob = io.pack_dense(bits)
    restored = io.unpack_dense(dtype, blob)

    assert dtype == "BF16"
    assert io.is_bfloat16_array(restored)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(bits))
    np.testing.assert_array_equal(
        io.bfloat16_to_float32(restored),
        np.asarray([[1.0, -2.5], [0.0, np.inf]], dtype=np.float32),
    )


def test_pack_nint_uses_bitpacked_payload():
    t = _mk(6, (128, 5120))
    blob = io.pack_nint(t)
    payload_bits = len(blob) * 8
    weight_bits = np.prod(t.shape)
    assert payload_bits / weight_bits < 4.55


@pytest.mark.parametrize(
    "spec",
    [
        NintSpec(3, 24, 5),
        NintSpec(5, 24, 6),
        NintSpec(6, 24, 6),
        NintSpec(8, 16, 8),
    ],
)
def test_pack_roundtrip_high_bit_nint(spec):
    rng = np.random.default_rng(12 + spec.bits)
    W = rng.normal(0, 0.05, size=(17, 257)).astype(np.float32)
    t = nint_quant.quantize(W, spec, axis=0)
    t2 = io.unpack_nint(io.pack_nint(t))
    assert t2.spec == spec
    np.testing.assert_array_equal(t2.q, t.q)
    np.testing.assert_array_equal(t2.sub_scale, t.sub_scale)
    np.testing.assert_array_equal(t2.sub_min, t.sub_min)
    np.testing.assert_allclose(nint_quant.dequantize(t2), nint_quant.dequantize(t))


def test_nintm_roundtrip_preserves_expert_profiles():
    rng = np.random.default_rng(20260719)
    weight = rng.normal(0, 0.05, size=(6, 5, 73)).astype(np.float32)
    specs = (
        NintSpec(4, 24, 6),
        NintSpec(6, 24, 6),
        NintSpec(4, 24, 6),
        NintSpec(8, 24, 8),
        NintSpec(5, 28, 6),
        NintSpec(6, 24, 6),
    )
    tensor = quantize_expertwise(weight, specs)
    restored = io.unpack_nint_moe(io.pack_nint_moe(tensor))
    assert restored.shape == tensor.shape
    assert restored.expert_profiles == tensor.expert_profiles
    assert len(restored.pools) == 4
    np.testing.assert_allclose(
        dequantize_expertwise(restored),
        dequantize_expertwise(tensor),
    )


def test_nintm_file_and_mmap_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(77)
    weight = rng.normal(0, 0.05, size=(4, 3, 48)).astype(np.float32)
    tensor = quantize_expertwise(
        weight,
        [NintSpec(4, 24, 6), NintSpec(6, 24, 6)] * 2,
    )
    path = tmp_path / "expert-wise.mfq"
    io.save(
        path,
        FileHeader(model_arch="moe-test", num_tensors=1),
        {"blk.0.ffn_gate_exps.weight": tensor},
    )
    _header, loaded = io.load(path)
    assert loaded["blk.0.ffn_gate_exps.weight"].expert_profiles == tensor.expert_profiles
    _header, store = io.load_mmap(path)
    try:
        assert store.records["blk.0.ffn_gate_exps.weight"].dtype == "NINTM"
        lazy = store["blk.0.ffn_gate_exps.weight"]
        np.testing.assert_allclose(
            dequantize_expertwise(lazy),
            dequantize_expertwise(tensor),
        )
    finally:
        store.close()


def test_file_roundtrip(tmp_path: Path):
    t0 = _mk(1, (8, 96))
    t1 = _mk(2, (16, 48))
    tensors = {"blk.0.gate": t0, "blk.1.up": t1}
    header = FileHeader(model_arch="test", num_tensors=2)
    path = tmp_path / "m.mfq"
    io.save(path, header, tensors)

    h2, t2 = io.load(path)
    assert h2.model_arch == "test" and h2.num_tensors == 2
    assert set(t2) == {"blk.0.gate", "blk.1.up"}
    np.testing.assert_allclose(nint_quant.dequantize(t2["blk.0.gate"]), nint_quant.dequantize(t0))
    np.testing.assert_allclose(nint_quant.dequantize(t2["blk.1.up"]), nint_quant.dequantize(t1))


def test_file_roundtrip_mixed_dense_and_extra(tmp_path: Path):
    t0 = _mk(3, (8, 32))
    norm = np.linspace(0.5, 1.5, 32, dtype=np.float32)
    tensors = {"token_embd.weight": t0, "output_norm.weight": norm}
    header = FileHeader(model_arch="tiny", num_tensors=2, extra={"hidden_size": "32"})
    path = tmp_path / "mixed.mfq"
    io.save(path, header, tensors)

    h2, t2 = io.load(path)
    assert h2.version == 2
    assert h2.extra["hidden_size"] == "32"
    np.testing.assert_allclose(
        nint_quant.dequantize(t2["token_embd.weight"]), nint_quant.dequantize(t0)
    )
    np.testing.assert_array_equal(t2["output_norm.weight"], norm)


def test_mmap_load_lazy_roundtrip(tmp_path: Path):
    t0 = _mk(7, (8, 96))
    norm = np.linspace(0.5, 1.5, 32, dtype=np.float32)
    path = tmp_path / "lazy.mfq"
    io.save(
        path,
        FileHeader(model_arch="lazy", num_tensors=2, extra={"mode": "mmap"}),
        {"blk.0.gate": t0, "output_norm.weight": norm},
    )

    header, store = io.load_mmap(path)
    try:
        assert header.model_arch == "lazy"
        assert header.extra["mode"] == "mmap"
        assert len(store) == 2
        assert set(store) == {"blk.0.gate", "output_norm.weight"}
        assert store.records["blk.0.gate"].nbytes < t0.q.nbytes
        t_lazy = store["blk.0.gate"]
        np.testing.assert_allclose(nint_quant.dequantize(t_lazy), nint_quant.dequantize(t0))
        np.testing.assert_array_equal(store["output_norm.weight"], norm)
    finally:
        store.close()


def test_file_and_mmap_roundtrip_nvq_tensors(tmp_path: Path):
    nvq1_l = Nvq1LTensor(
        spec=NVQ1_L_T8_S3,
        shape=(2, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.1, 0.2], dtype=np.float32),
        sub_scale=np.asarray([[3], [5]], dtype=np.uint8),
        indices=np.asarray([[1, 257, 1023], [2, 511, 2047]], dtype=np.uint16),
        delta_sign=np.asarray([[0], [1]], dtype=np.uint8),
    )
    nvq2 = NvqTensor(
        spec=NVQ2_E8,
        shape=(2, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.3, 0.4], dtype=np.float32),
        sub_scale=np.asarray([[7], [9]], dtype=np.uint8),
        indices=np.arange(6, dtype=np.uint8).reshape(2, 3),
        signs=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
    )
    nvq3 = NvqTensor(
        spec=NVQ3_D4,
        shape=(2, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.5, 0.6], dtype=np.float32),
        sub_scale=np.asarray([[11], [13]], dtype=np.uint8),
        indices=np.arange(12, dtype=np.uint8).reshape(2, 6),
        signs=np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.uint8),
    )
    nvq2j = NvqJscTensor(
        shape=(2, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.7, 0.8], dtype=np.float32),
        scale_lut=np.arange(16, dtype=np.float32),
        bank_for_state=np.zeros(16, dtype=np.uint8),
        state=np.asarray([[2], [4]], dtype=np.uint8),
        indices=np.asarray([[3, 2, 1], [6, 5, 4]], dtype=np.uint8),
        signs=np.asarray([[13, 14, 15], [16, 17, 18]], dtype=np.uint8),
        codebooks=(E8_256[None].astype(np.int16) * 8).astype(np.int8),
    )
    nvq3j = NvqJscTensor(
        shape=(2, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.9, 1.0], dtype=np.float32),
        scale_lut=np.arange(16, dtype=np.float32),
        bank_for_state=np.zeros(16, dtype=np.uint8),
        state=np.asarray([[3], [5]], dtype=np.uint8),
        indices=np.arange(12, dtype=np.uint8).reshape(2, 6),
        signs=np.asarray([[19, 20, 21], [22, 23, 24]], dtype=np.uint8),
        codebooks=(D4_256[None].astype(np.int16) * 8).astype(np.int8),
        base_spec=NVQ3_D4,
    )
    nvq3j_512 = NvqJscTensor(
        shape=(2, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([1.1, 1.2], dtype=np.float32),
        scale_lut=np.arange(16, dtype=np.float32),
        bank_for_state=np.zeros(16, dtype=np.uint8),
        state=np.asarray([[6], [7]], dtype=np.uint8),
        indices=np.asarray(
            [[0, 255, 256, 383, 511, 17], [511, 384, 257, 129, 1, 0]],
            dtype=np.uint16,
        ),
        signs=np.asarray([[25, 26, 27], [28, 29, 30]], dtype=np.uint8),
        codebooks=(D4_512[None].astype(np.int16) * 8).astype(np.int8),
        base_spec=NVQ3_D4_512,
    )
    path = tmp_path / "nvq.mfq"
    io.save(
        path,
        FileHeader(model_arch="nvq", num_tensors=6),
        {
            "w1": nvq1_l,
            "w2": nvq2,
            "w2j": nvq2j,
            "w3": nvq3,
            "w3j": nvq3j,
            "w3j512": nvq3j_512,
        },
    )

    _header, loaded = io.load(path)
    assert isinstance(loaded["w1"], Nvq1LTensor)
    assert isinstance(loaded["w2"], NvqTensor)
    assert isinstance(loaded["w2j"], NvqJscTensor)
    assert isinstance(loaded["w3"], NvqTensor)
    assert isinstance(loaded["w3j"], NvqJscTensor)
    assert isinstance(loaded["w3j512"], NvqJscTensor)
    assert loaded["w3j"].spec == NVQ3_D4
    assert loaded["w3j512"].spec == NVQ3_D4_512
    np.testing.assert_array_equal(loaded["w1"].indices, nvq1_l.indices)
    np.testing.assert_array_equal(loaded["w2"].signs, nvq2.signs)
    np.testing.assert_array_equal(loaded["w2j"].state, nvq2j.state)
    np.testing.assert_array_equal(loaded["w3"].sub_scale, nvq3.sub_scale)
    np.testing.assert_array_equal(loaded["w3j512"].indices, nvq3j_512.indices)

    _header, store = io.load_mmap(path)
    try:
        assert store.records["w1"].dtype == "NVQ1-L"
        assert store.records["w2"].dtype == "NVQ2"
        assert store.records["w2j"].dtype == "NVQ2J"
        assert store.records["w3"].dtype == "NVQ3"
        assert store.records["w3j"].dtype == "NVQ3J"
        assert store.records["w3j512"].dtype == "NVQ3J-512"
        np.testing.assert_array_equal(store["w1"].delta_sign, nvq1_l.delta_sign)
        np.testing.assert_array_equal(store["w2"].indices, nvq2.indices)
        np.testing.assert_array_equal(store["w2j"].codebooks, nvq2j.codebooks)
        np.testing.assert_array_equal(store["w3j"].codebooks, nvq3j.codebooks)
        np.testing.assert_array_equal(store["w3j512"].indices, nvq3j_512.indices)
    finally:
        store.close()

    legacy_path = tmp_path / "niq-legacy.mfq"
    io.save(
        legacy_path,
        FileHeader(model_arch="nvq", num_tensors=3),
        {"w2": nvq2, "w2j": nvq2j, "w3": nvq3},
    )
    legacy_path.write_bytes(legacy_path.read_bytes().replace(b"NVQ", b"NIQ"))
    _header, legacy_store = io.load_mmap(legacy_path)
    try:
        assert legacy_store.records["w2"].dtype == "NIQ2"
        assert legacy_store.records["w2j"].dtype == "NIQ2J"
        assert legacy_store.records["w3"].dtype == "NIQ3"
        np.testing.assert_array_equal(legacy_store["w2"].signs, nvq2.signs)
        np.testing.assert_array_equal(legacy_store["w2j"].state, nvq2j.state)
        np.testing.assert_array_equal(legacy_store["w3"].sub_scale, nvq3.sub_scale)
    finally:
        legacy_store.close()


def test_file_and_mmap_roundtrip_extended_nvq_jsc(tmp_path: Path) -> None:
    tensors = {
        "v2l": _extended_jsc_tensor(NVQ2_E8_1024, E8_1024, 13),
        "v2xl": _extended_jsc_tensor(NVQ2_E8_4096, E8_4096, 14),
        "v3l": _extended_jsc_tensor(NVQ3_D4_1024, D4_1024, 15),
    }
    path = tmp_path / "extended-nvq.mfq"
    io.save(
        path,
        FileHeader(model_arch="extended-nvq", num_tensors=len(tensors)),
        tensors,
    )
    _header, store = io.load_mmap(path)
    try:
        assert store.records["v2l"].dtype == "NVQ2J-L"
        assert store.records["v2xl"].dtype == "NVQ2J-XL"
        assert store.records["v3l"].dtype == "NVQ3J-L"
        for name, tensor in tensors.items():
            np.testing.assert_array_equal(store[name].indices, tensor.indices)
            np.testing.assert_array_equal(store[name].codebooks, tensor.codebooks)
    finally:
        store.close()


def test_extended_nvq_file_profile_requires_jsc_tensor(tmp_path: Path) -> None:
    source = _extended_jsc_tensor(NVQ2_E8_1024, E8_1024, 21)
    direct = NvqTensor(
        spec=source.spec,
        shape=source.shape,
        axis=source.axis,
        neuron_len=source.neuron_len,
        neuron_scale=source.neuron_scale,
        sub_scale=source.state,
        indices=source.indices,
        signs=source.signs,
    )
    with pytest.raises(ValueError, match="requires an NvqJscTensor"):
        io.save(
            tmp_path / "invalid-direct-extended-nvq.mfq",
            FileHeader(model_arch="extended-nvq", num_tensors=1),
            {"weight": direct},
        )


def test_file_and_mmap_roundtrip_npq0_l(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260722)
    tensor = Npq0LTensor(
        shape=(2, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.01, 0.02], dtype=np.float32),
        scale_lut=np.linspace(0.125, 1.0, 8, dtype=np.float32),
        state=np.asarray([[1], [7]], dtype=np.uint8),
        indices=np.asarray([[0, 63, 127], [3, 64, 99]], dtype=np.uint8),
        first_codebooks=rng.integers(-127, 128, size=(8, 8, 4), dtype=np.int16).astype(np.int8),
        second_codebooks=rng.integers(
            -127,
            128,
            size=(8, 16, 4),
            dtype=np.int16,
        ).astype(np.int8),
    )
    path = tmp_path / "nvq1_s-pq.mfq"
    io.save(
        path,
        FileHeader(model_arch="nvq1_s-pq", num_tensors=1),
        {"weight": tensor},
    )
    _header, loaded = io.load(path)
    assert isinstance(loaded["weight"], Npq0LTensor)
    np.testing.assert_array_equal(loaded["weight"].indices, tensor.indices)

    _header, store = io.load_mmap(path)
    try:
        assert store.records["weight"].dtype == "NPQ0-L"
        np.testing.assert_array_equal(store["weight"].state, tensor.state)
        np.testing.assert_array_equal(store["weight"].second_codebooks, tensor.second_codebooks)
    finally:
        store.close()


def test_file_and_mmap_roundtrip_nvq1_s_and_npq0_s(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260723)
    nvq1_s = Nvq1STensor(
        spec=NVQ1_S,
        shape=(2, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.01, 0.02], dtype=np.float32),
        sub_scale=np.asarray([[5], [11]], dtype=np.uint8),
        indices=np.asarray([[0, 255, 511], [17, 128, 300]], dtype=np.uint16),
        delta_sign=np.asarray([[0], [1]], dtype=np.uint8),
        codebook=NVQ1_S_BOOTSTRAP_BANKS,
    )
    npq0_s = Npq0STensor(
        shape=(2, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.003, 0.004], dtype=np.float32),
        scale_lut=np.linspace(0.25, 1.0, 4, dtype=np.float32),
        state=np.asarray([[1], [3]], dtype=np.uint8),
        indices=np.asarray([[0, 31, 63], [3, 32, 49]], dtype=np.uint8),
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
    path = tmp_path / "small-vq.mfq"
    io.save(
        path,
        FileHeader(model_arch="small-vq", num_tensors=2),
        {"nvq1_s": nvq1_s, "npq0_s": npq0_s},
    )
    _header, loaded = io.load(path)
    assert isinstance(loaded["nvq1_s"], Nvq1STensor)
    assert isinstance(loaded["npq0_s"], Npq0STensor)
    np.testing.assert_array_equal(loaded["nvq1_s"].codebook, nvq1_s.codebook)
    np.testing.assert_array_equal(
        loaded["npq0_s"].first_codebooks,
        npq0_s.first_codebooks,
    )
    np.testing.assert_array_equal(
        loaded["npq0_s"].second_codebooks,
        npq0_s.second_codebooks,
    )

    _header, store = io.load_mmap(path)
    try:
        assert store.records["nvq1_s"].dtype == "NVQ1-S"
        assert store.records["npq0_s"].dtype == "NPQ0-S"
        np.testing.assert_array_equal(store["nvq1_s"].indices, nvq1_s.indices)
        np.testing.assert_array_equal(store["npq0_s"].state, npq0_s.state)
    finally:
        store.close()


def test_load_rejects_bad_magic(tmp_path: Path):
    path = tmp_path / "bad.mfq"
    path.write_bytes(b"XXXX" + b"\x00" * 16)
    with pytest.raises(ValueError):
        io.load(path)
