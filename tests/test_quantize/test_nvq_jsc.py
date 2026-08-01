from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.formats.nvq import (
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_512,
)
from mfq.quantize.nvq_jsc import (
    NvqJscConfig,
    dequantize_nvq_jsc,
    initial_jsc_tables,
    initial_raw_codebooks,
    quantize_nvq_jsc_fixed,
    train_nvq_jsc,
)


def test_initial_raw_codebooks_are_deterministic_and_valid() -> None:
    config = NvqJscConfig(banks=4, raw_multiplier=8, seed=17)
    first = initial_raw_codebooks(config)
    second = initial_raw_codebooks(config)
    assert first.shape == (4, 256, 8)
    assert first.dtype == np.int8
    assert np.array_equal(first, second)
    assert int(first.min()) >= 0
    assert int(first.max()) <= 127
    assert not np.any(np.all(first == 0, axis=-1))
    assert not np.array_equal(first[0], first[1])


def test_fixed_jsc_quantizer_has_cpu_fallback_and_uses_shared_tables() -> None:
    weight = torch.linspace(-0.05, 0.06, 4 * 24, dtype=torch.float32).reshape(4, 24)
    tables = initial_jsc_tables(
        NvqJscConfig(banks=2, iterations=0, assignment_refine_steps=1, search_steps=3)
    )
    tensor = quantize_nvq_jsc_fixed(
        weight,
        tables,
        assignment_refine_steps=1,
        search_steps=3,
        group_chunk=16,
        device="cpu",
    )
    assert tensor.axis == 0
    assert tensor.state.shape == (4, 1)
    np.testing.assert_array_equal(tensor.scale_lut, tables.scale_lut)
    np.testing.assert_array_equal(tensor.bank_for_state, tables.bank_for_state)
    np.testing.assert_array_equal(tensor.codebooks, tables.codebooks)
    assert np.isfinite(dequantize_nvq_jsc(tensor)).all()


def test_fixed_nvq3j_quantizer_has_cpu_fallback() -> None:
    weight = torch.linspace(-0.05, 0.06, 4 * 24, dtype=torch.float32).reshape(4, 24)
    config = NvqJscConfig(
        banks=2,
        iterations=0,
        assignment_refine_steps=1,
        search_steps=3,
        spec=NVQ3_D4,
    )
    tables = initial_jsc_tables(config)
    tensor = quantize_nvq_jsc_fixed(
        weight,
        tables,
        assignment_refine_steps=1,
        search_steps=3,
        group_chunk=16,
        device="cpu",
    )
    assert tensor.spec == NVQ3_D4
    assert tensor.indices.shape == (4, 6)
    assert tensor.codebooks.shape == (2, 256, 4)
    assert np.isfinite(dequantize_nvq_jsc(tensor)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nvq3j_512_cuda_quantizer_uses_9bit_indices() -> None:
    generator = torch.Generator().manual_seed(20260803)
    weight = 0.04 * torch.randn((12, 50), generator=generator)
    importance = np.linspace(0.5, 1.5, 50, dtype=np.float32)
    tensor, history = train_nvq_jsc(
        weight,
        importance=importance,
        config=NvqJscConfig(
            banks=2,
            iterations=1,
            assignment_refine_steps=1,
            search_steps=5,
            learned_scale_lut=True,
            seed=23,
            spec=NVQ3_D4_512,
        ),
    )
    reconstruction = dequantize_nvq_jsc(tensor)
    assert tensor.spec == NVQ3_D4_512
    assert tensor.indices.dtype == np.uint16
    assert tensor.indices.shape == (12, 13)
    assert tensor.codebooks.shape == (2, 512, 4)
    assert int(tensor.indices.max()) > 255
    assert np.isfinite(reconstruction).all()
    assert min(item.weighted_sse for item in history) <= history[0].weighted_sse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("banks", [1, 2])
def test_nvq_jsc_cuda_roundtrip_and_payload(banks: int) -> None:
    generator = torch.Generator().manual_seed(20260717)
    weight = 0.04 * torch.randn((12, 50), generator=generator)
    importance = np.linspace(0.5, 1.5, 50, dtype=np.float32)
    tensor, history = train_nvq_jsc(
        weight,
        importance=importance,
        config=NvqJscConfig(
            banks=banks,
            iterations=1,
            assignment_refine_steps=1,
            search_steps=5,
            learned_scale_lut=True,
            seed=19,
        ),
    )
    reconstruction = dequantize_nvq_jsc(tensor)
    assert reconstruction.shape == tuple(weight.shape)
    assert np.isfinite(reconstruction).all()
    assert tensor.state.shape == (12, 3)
    assert tensor.indices.shape == (12, 7)
    assert tensor.signs.shape == (12, 7)
    assert tensor.codebooks.shape == (banks, 256, 8)
    assert tensor.payload_nbytes > 0
    assert len(history) == 2
    assert history[-1].weighted_nmse_percent > 0
    assert min(item.weighted_sse for item in history) <= history[0].weighted_sse
    assert int(tensor.bank_for_state[tensor.state].max()) < banks

    error = (
        (reconstruction - weight.numpy()) ** 2
        * importance.reshape(1, -1)
    ).sum()
    assert error == pytest.approx(
        min(item.weighted_sse for item in history),
        rel=2e-5,
        abs=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nvq_jsc_fixed_scale_lut_stays_linear() -> None:
    weight = torch.linspace(-0.08, 0.08, 8 * 48, dtype=torch.float32).reshape(8, 48)
    tensor, _history = train_nvq_jsc(
        weight,
        config=NvqJscConfig(
            banks=1,
            iterations=0,
            assignment_refine_steps=1,
            search_steps=5,
            learned_scale_lut=False,
        ),
    )
    assert np.array_equal(tensor.scale_lut, np.arange(16, dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nvq_jsc_float32_codebook_oracle() -> None:
    weight = torch.linspace(-0.05, 0.07, 6 * 48, dtype=torch.float32).reshape(6, 48)
    tensor, history = train_nvq_jsc(
        weight,
        config=NvqJscConfig(
            banks=1,
            iterations=1,
            assignment_refine_steps=1,
            search_steps=3,
            learned_scale_lut=True,
            codebook_storage="float32",
            group_chunk=32,
        ),
    )
    assert tensor.codebooks.dtype == np.float32
    assert np.isfinite(dequantize_nvq_jsc(tensor)).all()
    assert min(item.weighted_sse for item in history) <= history[0].weighted_sse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nvq2j_native_fixed_state_search_matches_exhaustive_reference() -> None:
    from mfq.quantize.cuda._ext import ext

    rng = np.random.default_rng(41)
    codebooks_np = rng.integers(
        1, 24, size=(4, 256, 8), dtype=np.int8
    )
    alpha_np = np.linspace(0.2, 1.7, 16, dtype=np.float32)
    bank_np = np.arange(16, dtype=np.uint8) % 4
    generator = torch.Generator(device="cuda")
    generator.manual_seed(43)
    value = torch.rand((2, 48), generator=generator, device="cuda")
    value[:, 40:] = 0
    objective_weight = 0.25 + 1.75 * torch.rand(
        (2, 48), generator=generator, device="cuda"
    )
    objective_weight[:, 40:] = 0
    objective_weight[:, 7::13] = 0
    anchor = torch.tensor([0.03125, 0.046875], device="cuda")
    alpha = torch.as_tensor(alpha_np, device="cuda")
    bank = torch.as_tensor(bank_np, device="cuda")
    codebooks = torch.as_tensor(
        codebooks_np, device="cuda"
    ).permute(0, 2, 1).contiguous()
    native_anchor, native_state, native_indices = ext().nvq2j_assign(
        value,
        objective_weight,
        anchor,
        alpha,
        bank,
        codebooks,
        40,
        1,
    )

    def assign(current_anchor):
        states = torch.empty((2, 2), device="cuda", dtype=torch.uint8)
        indices = torch.empty((2, 2, 3), device="cuda", dtype=torch.uint8)
        dense_codebooks = torch.as_tensor(codebooks_np, device="cuda").float()
        for row in range(2):
            for group in range(2):
                valid_group = min(24, 40 - group * 24)
                best_error = torch.inf
                best_state = 0
                best_vector_indices = None
                for state in range(16):
                    scale = current_anchor[row] * alpha[state]
                    selected = []
                    error = torch.zeros((), device="cuda")
                    for vector in range(3):
                        valid = min(8, max(0, valid_group - vector * 8))
                        if valid == 0:
                            selected.append(0)
                            continue
                        source = value[
                            row,
                            group * 24 + vector * 8 :
                            group * 24 + vector * 8 + valid,
                        ]
                        candidates = dense_codebooks[
                            int(bank_np[state]), :, :valid
                        ]
                        weights = objective_weight[
                            row,
                            group * 24 + vector * 8 :
                            group * 24 + vector * 8 + valid,
                        ]
                        distance = (
                            weights[None]
                            * (source[None] - scale * candidates).square()
                        ).sum(1)
                        index = int(distance.argmin())
                        selected.append(index)
                        error += distance[index]
                    if error < best_error:
                        best_error = error
                        best_state = state
                        best_vector_indices = selected
                states[row, group] = best_state
                indices[row, group] = torch.tensor(
                    best_vector_indices,
                    device="cuda",
                    dtype=torch.uint8,
                )
        return states, indices

    first_state, first_indices = assign(anchor)
    dense_codebooks = torch.as_tensor(codebooks_np, device="cuda").float()
    basis = torch.zeros_like(value)
    for row in range(2):
        for position in range(40):
            group = position // 24
            vector = (position % 24) // 8
            coordinate = position % 8
            state = int(first_state[row, group])
            entry = int(first_indices[row, group, vector])
            basis[row, position] = (
                alpha[state]
                * dense_codebooks[int(bank_np[state]), entry, coordinate]
            )
    numerator = (
        objective_weight[:, :40] * value[:, :40] * basis[:, :40]
    ).sum(1)
    denominator = (
        objective_weight[:, :40] * basis[:, :40].square()
    ).sum(1)
    fitted = (numerator / denominator).to(torch.float16).to(torch.float32)
    reference_state, reference_indices = assign(fitted)

    torch.testing.assert_close(native_anchor, fitted, rtol=0, atol=0)
    assert torch.equal(native_state, reference_state)
    assert torch.equal(native_indices, reference_indices)


@pytest.mark.parametrize("spec", [NVQ2_E8_1024, NVQ2_E8_4096])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_extended_nvq2j_native_assignment_matches_exhaustive_reference(
    spec,
) -> None:
    from mfq.quantize.cuda._ext import ext

    config = NvqJscConfig(
        banks=4,
        iterations=0,
        assignment_refine_steps=0,
        spec=spec,
    )
    tables = initial_jsc_tables(config)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(101 + spec.codebook_entries)
    value = torch.rand((1, 24), generator=generator, device="cuda")
    objective_weight = 0.2 + torch.rand(
        (1, 24), generator=generator, device="cuda"
    )
    anchor = torch.tensor([0.0078125], device="cuda")
    alpha = torch.as_tensor(tables.scale_lut, device="cuda")
    bank = torch.as_tensor(tables.bank_for_state, device="cuda")
    codebooks = torch.as_tensor(
        tables.codebooks, device="cuda"
    ).permute(0, 2, 1).contiguous()

    _, native_state, native_indices = ext().nvq2j_assign(
        value,
        objective_weight,
        anchor,
        alpha,
        bank,
        codebooks,
        24,
        0,
    )

    dense_codebooks = torch.as_tensor(
        tables.codebooks, device="cuda", dtype=torch.float32
    )
    state_errors = []
    state_indices = []
    for state in range(16):
        scale = anchor[0] * alpha[state]
        vector_errors = []
        vector_indices = []
        for vector in range(3):
            start = vector * 8
            residual = (
                value[0, start : start + 8]
                - scale * dense_codebooks[int(tables.bank_for_state[state])]
            )
            error = (
                objective_weight[0, start : start + 8]
                * residual.square()
            ).sum(1)
            index = error.argmin()
            vector_errors.append(error[index])
            vector_indices.append(index)
        state_errors.append(torch.stack(vector_errors).sum())
        state_indices.append(torch.stack(vector_indices))
    reference_state = torch.stack(state_errors).argmin()
    reference_indices = state_indices[int(reference_state)]

    assert int(native_state[0, 0]) == int(reference_state)
    assert torch.equal(
        native_indices[0, 0].to(torch.int64),
        reference_indices.to(torch.int64),
    )


@pytest.mark.parametrize("spec", [NVQ2_E8_1024, NVQ2_E8_4096])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_extended_nvq2j_fused_bank_search_matches_legacy_kernels(
    spec,
) -> None:
    from mfq.quantize.cuda._ext import ext

    config = NvqJscConfig(
        banks=4,
        iterations=0,
        assignment_refine_steps=0,
        search_steps=5,
        spec=spec,
    )
    codebooks = torch.as_tensor(
        initial_raw_codebooks(config), device="cuda"
    ).contiguous()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(151 + spec.codebook_entries)
    xgroup = torch.rand(
        (4, 24), generator=generator, device="cuda"
    )
    wgroup = 0.1 + torch.rand(
        (4, 24), generator=generator, device="cuda"
    )
    bank_qmax = codebooks.amax((1, 2)).to(torch.float32)
    fused_scale, fused_indices = ext().nvq2j_search_banks(
        xgroup,
        wgroup,
        codebooks,
        bank_qmax,
        2,
        17,
        config.search_steps,
    )
    legacy_scales = []
    legacy_indices = []
    for bank_id in range(4):
        scale, indices = ext().nvq_search(
            xgroup,
            wgroup,
            codebooks[bank_id],
            2,
            17,
            8,
            config.search_steps,
            float(bank_qmax[bank_id]),
        )
        legacy_scales.append(scale)
        legacy_indices.append(indices)
    legacy_scale = torch.stack(legacy_scales, dim=1)
    legacy_index = torch.stack(legacy_indices, dim=1)
    torch.testing.assert_close(fused_scale, legacy_scale, rtol=0, atol=0)
    assert torch.equal(
        fused_indices.to(torch.int64),
        legacy_index,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nvq2j_weighted_fixed_quantizer_uses_native_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mfq.quantize.cuda import _ext as ext_module

    native = ext_module.ext()
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class NativeProxy:
        def nvq2j_assign(self, value, objective_weight, *args):
            calls.append(
                (tuple(value.shape), tuple(objective_weight.shape))
            )
            return native.nvq2j_assign(
                value, objective_weight, *args
            )

    proxy = NativeProxy()
    monkeypatch.setattr(ext_module, "ext", lambda: proxy)
    generator = torch.Generator().manual_seed(47)
    weight = 0.04 * torch.randn((8, 50), generator=generator)
    importance = np.linspace(0.25, 1.75, 50, dtype=np.float32)
    importance[::11] = 0
    tables = initial_jsc_tables(
        NvqJscConfig(
            banks=4,
            iterations=0,
            assignment_refine_steps=1,
        )
    )
    tensor = quantize_nvq_jsc_fixed(
        weight,
        tables,
        importance=importance,
        assignment_refine_steps=1,
        device="cuda",
    )

    assert calls == [((8, 72), (8, 72))]
    assert np.isfinite(dequantize_nvq_jsc(tensor)).all()
