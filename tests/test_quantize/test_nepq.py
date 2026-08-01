from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.formats.nepq import (  # noqa: E402
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_L,
    NEPQ1_S,
    dequantize_nepq,
    pack_nepq,
    rotation_signs,
    unpack_nepq,
)
from mfq.formats.npq0_s import pack_npq0_s_tables  # noqa: E402
from mfq.quantize.nepq import (  # noqa: E402
    NepqQuantConfig,
    _decode_pool,
    _fwht_blocks,
    _project,
    _project_native_nepq0_s,
    quantize_nepq_fixed,
)
from tests.test_formats.test_nepq import _tensor  # noqa: E402


def _known_tensor(spec):
    tensor = unpack_nepq(pack_nepq(_tensor(spec)))
    for supergroup in range(tensor.bank_ids.shape[-1]):
        tensor.bank_ids[:, :, supergroup] = (
            np.arange(tensor.n_experts * tensor.out_per_expert).reshape(
                tensor.n_experts, tensor.out_per_expert
            )
            + supergroup
        ) % tensor.bank_count
    return tensor


@pytest.mark.parametrize("spec", [NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L])
def test_nepq_fixed_pool_quantizer_recovers_feasible_codes(spec):
    source = _known_tensor(spec)
    weight = dequantize_nepq(source)
    result = quantize_nepq_fixed(
        weight,
        spec,
        source.table_payloads,
        initial_anchor=source.neuron_scale,
        config=NepqQuantConfig(
            anchor_multipliers=(1.0,),
            refine_steps=0,
            row_chunk=2,
            bank_chunk=1,
        ),
    )
    reconstructed = dequantize_nepq(result)
    relative = np.linalg.norm(reconstructed - weight) / np.linalg.norm(weight)
    assert relative < 2e-6, f"{spec.label} relative error {relative}"


def test_nepq_rotated_importance_requires_admm():
    source = _known_tensor(NEPQ0_S)
    with pytest.raises(ValueError, match="ADMM"):
        quantize_nepq_fixed(
            dequantize_nepq(source),
            NEPQ0_S,
            source.table_payloads,
            importance=np.ones(source.neuron_len, dtype=np.float32),
            rotation_block=8,
            rotation_seed=18601311049,
            config=NepqQuantConfig(
                anchor_multipliers=(1.0,),
                refine_steps=0,
                row_chunk=2,
                bank_chunk=1,
            ),
        )


def test_nepq_hadamard_imatrix_admm_never_loses_initial_feasible_result():
    source = _known_tensor(NEPQ0_S)
    rng = np.random.default_rng(44)
    weight = dequantize_nepq(source) + rng.normal(
        0.0, 0.02, size=source.shape
    ).astype(np.float32)
    importance = np.ones(
        (source.n_experts, source.neuron_len), dtype=np.float32
    )
    importance[0, : source.neuron_len // 2] = 32.0
    importance[1, source.neuron_len // 2 :] = 32.0
    common = dict(
        anchor_multipliers=(1.0,),
        refine_steps=1,
        row_chunk=2,
        bank_chunk=2,
    )
    initial = quantize_nepq_fixed(
        weight,
        NEPQ0_S,
        source.table_payloads,
        rotation_block=8,
        rotation_seed=18601311049,
        config=NepqQuantConfig(**common),
    )
    candidate = quantize_nepq_fixed(
        weight,
        NEPQ0_S,
        source.table_payloads,
        importance=importance,
        rotation_block=8,
        rotation_seed=18601311049,
        config=NepqQuantConfig(
            **common,
            admm_iterations=2,
            admm_rho=1.0,
        ),
    )
    signs = torch.as_tensor(
        rotation_signs(source.neuron_len, 8, 18601311049),
        device="cuda",
        dtype=torch.float32,
    )
    rotated = _fwht_blocks(
        torch.as_tensor(weight, device="cuda") * signs, 8
    )

    def weighted_error(tensor):
        reconstruction = torch.as_tensor(
            dequantize_nepq(tensor), device="cuda"
        )
        residual = _fwht_blocks(rotated - reconstruction, 8)
        objective = torch.as_tensor(importance, device="cuda")[:, None, :]
        return float((objective * residual.square()).sum().item())

    initial_error = weighted_error(initial)
    candidate_error = weighted_error(candidate)
    assert candidate_error <= initial_error * (1.0 + 1e-6)


def test_nepq0_s_native_assignment_matches_reference():
    rng = np.random.default_rng(20260723)
    tables = []
    for _ in range(256):
        scale = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        first = rng.integers(-12, 13, size=(4, 8, 4), dtype=np.int8)
        second = rng.integers(-12, 13, size=(4, 8, 4), dtype=np.int8)
        tables.append(
            np.frombuffer(
                pack_npq0_s_tables(scale, first, second),
                dtype=np.uint8,
            )
        )
    pool = _decode_pool(NEPQ0_S, np.stack(tables), "cuda")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(31)
    value = torch.randn((3, 40), generator=generator, device="cuda")
    base_anchor = value.abs().amax(1) / pool.maximum_basis
    reference = _project(
        value,
        torch.ones_like(value),
        base_anchor,
        pool,
        NepqQuantConfig(
            anchor_multipliers=(1.0,),
            refine_steps=1,
            row_chunk=3,
            bank_chunk=32,
        ),
    )
    native = _project_native_nepq0_s(value, base_anchor, pool)
    torch.testing.assert_close(native.anchor, reference.anchor, rtol=0, atol=0)
    assert torch.equal(native.bank_ids, reference.bank_ids)
    assert torch.equal(native.state, reference.state)
    assert torch.equal(native.indices, reference.indices.to(torch.uint8))
