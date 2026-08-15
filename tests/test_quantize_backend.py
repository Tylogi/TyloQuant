from __future__ import annotations

import pytest
import torch

from mfq.quantize.backend import (
    QuantBackend,
    resolve_quant_backend,
    resolve_row_chunk,
)


def test_cpu_quant_backend_ignores_accelerator_device() -> None:
    assert resolve_quant_backend("cpu", "cuda:7") == QuantBackend(
        name="cpu", device="cpu"
    )


def test_row_chunk_uses_larger_metal_default() -> None:
    assert resolve_row_chunk(0, "metal") == 8192
    assert resolve_row_chunk(0, "cuda") == 1024
    assert resolve_row_chunk(2048, "metal") == 2048


def test_auto_quant_backend_prefers_metal_after_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("mfq.quantize.backend.metal_available", lambda: True)
    assert resolve_quant_backend("auto", "cuda") == QuantBackend(
        name="metal", device="mps"
    )


def test_explicit_unavailable_metal_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mfq.quantize.backend.metal_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        resolve_quant_backend("metal", "mps")


def test_tpq_default_device_prefers_metal_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mfq.quantize.tpq import _device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _device(None) == torch.device("mps")


def test_gguf_converter_enables_native_assignment_for_metal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mfq.tools.quantize_gguf_to_mfq import _quantize_nvq_chunk

    captured: dict[str, object] = {}
    sentinel = object()

    def fake_quantize_axis0(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "mfq.quantize.nvq_quant_torch.quantize_axis0",
        fake_quantize_axis0,
    )
    result = _quantize_nvq_chunk(
        torch.zeros((1, 24)),
        "NVQ2",
        "metal",
        "mps",
        64,
        0,
        (0.75,),
        1,
        None,
        None,
        3,
        True,
        True,
    )
    assert result is sentinel
    assert captured["nvq_native_assignment"] is True
    assert captured["nvq1_l_native_assignment"] is True


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
@pytest.mark.parametrize("entries,vector_size", [(256, 8), (4096, 4)])
def test_metal_tpq_assignment_matches_torch_reference(
    entries: int,
    vector_size: int,
) -> None:
    from mfq.quantize.tpq import _assign_device, _assign_device_torch

    generator = torch.Generator().manual_seed(entries + vector_size)
    points = torch.randn((19, vector_size), generator=generator).to("mps")
    codebook = torch.randn(
        (entries, vector_size), generator=generator
    ).to("mps")
    labels, errors = _assign_device(
        points, codebook, distance_bytes=1 << 20
    )
    reference_labels, reference_errors = _assign_device_torch(
        points, codebook, distance_bytes=1 << 20
    )
    assert torch.equal(labels, reference_labels)
    torch.testing.assert_close(errors, reference_errors, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_tpq_training_uses_native_assignment_end_to_end() -> None:
    from mfq.formats.tpq import TPQ_X
    from mfq.quantize.tpq import TpqKmeansConfig, train_tpq_codebook

    points = torch.randn(
        (264, TPQ_X.vector_size),
        generator=torch.Generator().manual_seed(20260814),
    )
    result = train_tpq_codebook(
        points,
        TPQ_X,
        config=TpqKmeansConfig(
            iterations=2,
            restarts=1,
            sample_points=264,
            distance_bytes=1 << 20,
        ),
        device="mps",
    )
    assert result.codebook.shape == (
        TPQ_X.codebook_entries,
        TPQ_X.vector_size,
    )
    assert result.sse >= 0.0
    assert len(result.history) == 2


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_tpq_int4_matches_cpu_encoding() -> None:
    import numpy as np

    from mfq.quantize.tpq import quantize_tpq_int4

    weight = np.random.default_rng(20260814).normal(
        0.0, 0.2, (7, 128)
    ).astype(np.float32)
    cpu = quantize_tpq_int4(weight)
    metal = quantize_tpq_int4(torch.from_numpy(weight).to("mps"))
    np.testing.assert_array_equal(metal.packed, cpu.packed)
    np.testing.assert_array_equal(metal.scales, cpu.scales)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nint8_zero_matches_cpu_encoding() -> None:
    import numpy as np

    from mfq.formats.nint8_zero import quantize_nint8_zero
    from mfq.quantize.metal.nint8_zero import quantize

    weight = np.random.default_rng(20260814).normal(
        0.0, 0.2, (7, 128)
    ).astype(np.float32)
    cpu = quantize_nint8_zero(weight)
    metal = quantize(torch.from_numpy(weight).to("mps"))
    np.testing.assert_array_equal(metal.q, cpu.q)
    np.testing.assert_array_equal(metal.scale, cpu.scale)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
@pytest.mark.parametrize("width", [40, 48])
def test_metal_nvq1_s_matches_cpu_encoding(width: int) -> None:
    import numpy as np

    from mfq.quantize.nvq1_s_quant import quantize as quantize_cpu
    from mfq.quantize.nvq1_s_quant_torch import quantize_axis0

    weight = np.random.default_rng(20260814 + width).normal(
        0.0, 0.05, (3, width)
    ).astype(np.float32)
    kwargs = {"anchor_multipliers": (0.75,), "refine_steps": 1}
    cpu = quantize_cpu(weight, **kwargs)
    metal = quantize_axis0(
        torch.from_numpy(weight), device="mps", **kwargs
    )
    np.testing.assert_array_equal(metal.neuron_scale, cpu.neuron_scale)
    np.testing.assert_array_equal(metal.sub_scale, cpu.sub_scale)
    np.testing.assert_array_equal(metal.delta_sign, cpu.delta_sign)
    np.testing.assert_array_equal(metal.indices, cpu.indices)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_hf_stream_routes_nvq1_s_to_metal() -> None:
    import numpy as np

    from mfq.calibration.artifact import ExpertPrecision
    from mfq.tools.quantize_hf_to_mfq import _quantize_flat_stream_chunk

    weight = torch.from_numpy(
        np.random.default_rng(20260814)
        .normal(0.0, 0.05, (3, 48))
        .astype(np.float32)
    )
    tensor = _quantize_flat_stream_chunk(
        weight,
        ExpertPrecision(
            "NVQ1-S",
            options=(
                ("anchor_multipliers", "0.75"),
                ("refine_steps", 0),
            ),
        ),
        None,
        quant_backend="metal",
        device="mps",
    )
    assert tensor.shape == (3, 48)
    assert tensor.indices.shape == (3, 6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nint_matches_cpu_reconstruction_quality() -> None:
    import numpy as np

    from mfq.formats.nint import NintSpec
    from mfq.quantize.nint_quant import dequantize, quantize
    from mfq.quantize.nint_quant_torch import quantize_axis0

    generator = torch.Generator().manual_seed(20260812)
    weight = torch.randn((32, 288), generator=generator)
    spec = NintSpec(4, 24, 6)
    cpu = quantize(weight.numpy(), spec, axis=0)
    metal = quantize_axis0(weight, spec, device="mps")
    cpu_rmse = np.sqrt(np.mean((dequantize(cpu) - weight.numpy()) ** 2))
    metal_rmse = np.sqrt(np.mean((dequantize(metal) - weight.numpy()) ** 2))
    assert metal_rmse <= cpu_rmse * 1.01


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nvq2j_matches_cpu_quality() -> None:
    import numpy as np

    from mfq.quantize.nvq_jsc import (
        NvqJscConfig,
        dequantize_nvq_jsc,
        initial_jsc_tables,
        quantize_nvq_jsc_fixed,
    )

    generator = torch.Generator().manual_seed(20260812)
    weight = torch.randn((32, 288), generator=generator)
    tables = initial_jsc_tables(NvqJscConfig(banks=4))
    cpu = quantize_nvq_jsc_fixed(weight, tables, device="cpu")
    metal = quantize_nvq_jsc_fixed(weight, tables, device="mps")
    source = weight.numpy()
    cpu_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(cpu) - source) ** 2)
    )
    metal_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(metal) - source) ** 2)
    )
    assert metal_rmse <= cpu_rmse * 1.01


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nvq3j_matches_cpu_quality() -> None:
    import numpy as np

    from mfq.formats.nvq import NVQ3_D4
    from mfq.quantize.nvq_jsc import (
        NvqJscConfig,
        dequantize_nvq_jsc,
        initial_jsc_tables,
        quantize_nvq_jsc_fixed,
    )

    generator = torch.Generator().manual_seed(20260814)
    weight = torch.randn((32, 288), generator=generator)
    tables = initial_jsc_tables(
        NvqJscConfig(banks=2, spec=NVQ3_D4)
    )
    cpu = quantize_nvq_jsc_fixed(weight, tables, device="cpu")
    metal = quantize_nvq_jsc_fixed(weight, tables, device="mps")
    source = weight.numpy()
    cpu_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(cpu) - source) ** 2)
    )
    metal_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(metal) - source) ** 2)
    )
    assert metal_rmse <= cpu_rmse * 1.01


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
@pytest.mark.parametrize("entries", [512, 1024])
def test_metal_large_nvq3j_matches_cpu_quality(entries: int) -> None:
    import numpy as np

    from mfq.formats.nvq import NVQ3_D4_512, NVQ3_D4_1024
    from mfq.quantize.nvq_jsc import (
        NvqJscConfig,
        dequantize_nvq_jsc,
        initial_jsc_tables,
        quantize_nvq_jsc_fixed,
    )

    spec = NVQ3_D4_512 if entries == 512 else NVQ3_D4_1024
    weight = torch.from_numpy(
        np.random.default_rng(20260814 + entries)
        .normal(0.0, 0.05, (3, 48))
        .astype(np.float32)
    )
    tables = initial_jsc_tables(NvqJscConfig(banks=2, spec=spec))
    cpu = quantize_nvq_jsc_fixed(
        weight, tables, device="cpu", assignment_refine_steps=1
    )
    metal = quantize_nvq_jsc_fixed(
        weight, tables, device="mps", assignment_refine_steps=1
    )
    source = weight.numpy()
    cpu_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(cpu) - source) ** 2)
    )
    metal_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(metal) - source) ** 2)
    )
    assert metal_rmse <= cpu_rmse * 1.01


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
@pytest.mark.parametrize("entries", [1024, 4096])
def test_metal_large_nvq2j_matches_cpu_quality(entries: int) -> None:
    import numpy as np

    from mfq.formats.nvq import NVQ2_E8_1024, NVQ2_E8_4096
    from mfq.quantize.nvq_jsc import (
        NvqJscConfig,
        dequantize_nvq_jsc,
        initial_jsc_tables,
        quantize_nvq_jsc_fixed,
    )

    spec = NVQ2_E8_1024 if entries == 1024 else NVQ2_E8_4096
    weight = torch.from_numpy(
        np.random.default_rng(20260814 + entries)
        .normal(0.0, 0.05, (3, 48))
        .astype(np.float32)
    )
    tables = initial_jsc_tables(NvqJscConfig(banks=4, spec=spec))
    cpu = quantize_nvq_jsc_fixed(
        weight, tables, device="cpu", assignment_refine_steps=1
    )
    metal = quantize_nvq_jsc_fixed(
        weight, tables, device="mps", assignment_refine_steps=1
    )
    source = weight.numpy()
    cpu_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(cpu) - source) ** 2)
    )
    metal_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(metal) - source) ** 2)
    )
    assert metal_rmse <= cpu_rmse * 1.01


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
@pytest.mark.parametrize("family", ["s", "l"])
def test_metal_npq_fixed_assignment_matches_cpu_quality(family: str) -> None:
    import numpy as np

    rng = np.random.default_rng(20260814)
    generator = torch.Generator().manual_seed(20260814)
    weight = 0.05 * torch.randn((16, 280), generator=generator)
    if family == "s":
        from mfq.quantize.npq0_s import (
            Npq0SConfig as Config,
        )
        from mfq.quantize.npq0_s import (
            Npq0STables as Tables,
        )
        from mfq.quantize.npq0_s import (
            dequantize_npq0_s as dequantize,
        )
        from mfq.quantize.npq0_s import (
            quantize_npq0_s_fixed as quantize_fixed,
        )

        states, second_entries = 4, 8
    else:
        from mfq.quantize.npq0_l import (
            Npq0LConfig as Config,
        )
        from mfq.quantize.npq0_l import (
            Npq0LTables as Tables,
        )
        from mfq.quantize.npq0_l import (
            dequantize_npq0_l as dequantize,
        )
        from mfq.quantize.npq0_l import (
            quantize_npq0_l_fixed as quantize_fixed,
        )

        states, second_entries = 8, 16
    tables = Tables(
        scale_lut=np.linspace(0.2, 1.5, states, dtype=np.float32),
        first_codebooks=rng.integers(
            -31, 32, (states, 8, 4), dtype=np.int8
        ),
        second_codebooks=rng.integers(
            -31, 32, (states, second_entries, 4), dtype=np.int8
        ),
    )
    config = Config(fixed_refine_steps=1, anchor_multipliers=(1.0,))
    cpu = quantize_fixed(weight, tables, config=config, device="cpu")
    metal = quantize_fixed(weight, tables, config=config, device="mps")
    source = weight.numpy()
    cpu_rmse = np.sqrt(np.mean((dequantize(cpu) - source) ** 2))
    metal_rmse = np.sqrt(np.mean((dequantize(metal) - source) ** 2))
    assert metal_rmse <= cpu_rmse * 1.01


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nepq0_s_256_bank_projection_matches_reference() -> None:
    import numpy as np

    from mfq.formats.nepq import NEPQ0_S
    from mfq.formats.npq0_s import pack_npq0_s_tables
    from mfq.quantize.nepq import (
        NepqQuantConfig,
        _decode_pool,
        _project,
        _project_native_nepq0_s,
    )

    rng = np.random.default_rng(20260814)
    tables = []
    for _ in range(256):
        scale = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        first = rng.integers(-12, 13, (4, 8, 4), dtype=np.int8)
        second = rng.integers(-12, 13, (4, 8, 4), dtype=np.int8)
        tables.append(
            np.frombuffer(
                pack_npq0_s_tables(scale, first, second), dtype=np.uint8
            )
        )
    pool = _decode_pool(NEPQ0_S, np.stack(tables), "mps")
    generator = torch.Generator().manual_seed(20260814)
    value = torch.randn((2, 40), generator=generator).to("mps")
    base_anchor = value.abs().amax(1) / pool.maximum_basis
    reference = _project(
        value,
        torch.ones_like(value),
        base_anchor,
        pool,
        NepqQuantConfig(
            anchor_multipliers=(1.0,),
            refine_steps=1,
            row_chunk=2,
            bank_chunk=32,
        ),
    )
    metal = _project_native_nepq0_s(value, base_anchor, pool)
    torch.testing.assert_close(metal.anchor, reference.anchor, rtol=0, atol=0)
    assert torch.equal(metal.bank_ids, reference.bank_ids)
    assert torch.equal(metal.state, reference.state)
    assert torch.equal(metal.indices, reference.indices.to(torch.uint8))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
@pytest.mark.parametrize("family", ["nvq2", "nvq3"])
def test_metal_native_nvq_search_matches_torch_path(family: str) -> None:
    import numpy as np

    from mfq.formats.nvq import NVQ2_E8, NVQ3_D4
    from mfq.quantize.nvq_quant_torch import quantize_axis0

    spec = NVQ2_E8 if family == "nvq2" else NVQ3_D4
    weight = torch.from_numpy(
        np.random.default_rng(20260814 + spec.vector_size)
        .normal(0, 0.05, size=(5, 53))
        .astype(np.float32)
    )
    kwargs = dict(device="mps", search_steps=5, group_chunk=64)
    native = quantize_axis0(
        weight, spec, nvq_native_assignment=True, **kwargs
    )
    reference = quantize_axis0(
        weight, spec, nvq_native_assignment=False, **kwargs
    )
    np.testing.assert_array_equal(native.neuron_scale, reference.neuron_scale)
    np.testing.assert_array_equal(native.sub_scale, reference.sub_scale)
    np.testing.assert_array_equal(native.indices, reference.indices)
    np.testing.assert_array_equal(native.signs, reference.signs)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_native_nvq1_l_assignment_matches_torch_path() -> None:
    import numpy as np

    from mfq.formats.nvq1_l import NVQ1_L_T8_S3
    from mfq.quantize.nvq_quant_torch import quantize_axis0

    weight = torch.from_numpy(
        np.random.default_rng(20260814)
        .normal(0, 0.05, size=(3, 48))
        .astype(np.float32)
    )
    kwargs = dict(
        device="mps",
        anchor_multipliers=(0.75,),
        refine_steps=1,
        group_chunk=64,
        nvq1_l_candidates=0,
    )
    native = quantize_axis0(
        weight, NVQ1_L_T8_S3, nvq1_l_native_assignment=True, **kwargs
    )
    reference = quantize_axis0(
        weight, NVQ1_L_T8_S3, nvq1_l_native_assignment=False, **kwargs
    )
    np.testing.assert_array_equal(native.neuron_scale, reference.neuron_scale)
    np.testing.assert_array_equal(native.sub_scale, reference.sub_scale)
    np.testing.assert_array_equal(native.delta_sign, reference.delta_sign)
    np.testing.assert_array_equal(native.indices, reference.indices)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nvq_jsc_training_search_matches_cpu_quality() -> None:
    import numpy as np

    from mfq.quantize.nvq_jsc import (
        NvqJscConfig,
        dequantize_nvq_jsc,
        train_nvq_jsc,
    )

    weight = torch.from_numpy(
        np.random.default_rng(20260814)
        .normal(0, 0.05, size=(8, 48))
        .astype(np.float32)
    )
    config = NvqJscConfig(
        banks=4,
        iterations=0,
        assignment_refine_steps=0,
        search_steps=3,
        group_chunk=32,
    )
    cpu, _ = train_nvq_jsc(weight, config=config, device="cpu")
    metal, _ = train_nvq_jsc(weight, config=config, device="mps")
    source = weight.numpy()
    cpu_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(cpu) - source) ** 2)
    )
    metal_rmse = np.sqrt(
        np.mean((dequantize_nvq_jsc(metal) - source) ** 2)
    )
    assert metal_rmse <= cpu_rmse * 1.01


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nepq_bank_training_assignment_matches_torch() -> None:
    from mfq.quantize.nepq_train import _assign, _assign_torch

    generator = torch.Generator().manual_seed(20260814)
    batches, rows, groups_per_row = 2, 3, 2
    groups = rows * groups_per_row
    value = torch.randn(
        (batches, groups, 24), generator=generator
    ).to("mps")
    objective = (
        0.25 + torch.rand((batches, groups, 24), generator=generator)
    ).to("mps")
    anchor = (
        0.01 + 0.1 * torch.rand((batches, rows), generator=generator)
    ).to("mps")
    scale_lut = torch.tensor(
        [[0.25, 0.5, 0.75, 1.0], [0.2, 0.45, 0.7, 1.0]],
        device="mps",
    )
    first = torch.randint(
        -12, 13, (batches, 4, 8, 4), generator=generator, dtype=torch.int8
    ).to("mps")
    second = torch.randint(
        -12, 13, (batches, 4, 8, 4), generator=generator, dtype=torch.int8
    ).to("mps")
    kwargs = dict(rows=rows, ng=groups_per_row)
    reference = _assign_torch(
        value, objective, anchor, scale_lut, first, second, **kwargs
    )
    metal = _assign(
        value, objective, anchor, scale_lut, first, second, **kwargs
    )
    assert torch.equal(metal[0], reference[0])
    assert torch.equal(metal[1], reference[1])
    assert torch.equal(metal[2], reference[2])
    torch.testing.assert_close(metal[3], reference[3], rtol=2e-6, atol=2e-5)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nepq_bank_training_end_to_end() -> None:
    import numpy as np

    from mfq.formats.npq0_s import NPQ0_S_TABLE_BYTES
    from mfq.quantize.nepq_train import (
        NepqBankTrainConfig,
        train_nepq0_s_banks,
    )

    generator = torch.Generator().manual_seed(20260814)
    samples = (
        0.05 * torch.randn((2, 8, 48), generator=generator)
    ).to("mps")
    trained = train_nepq0_s_banks(
        samples,
        config=NepqBankTrainConfig(
            iterations=1,
            assignment_refine_steps=1,
            kmeans_iterations=2,
            expert_batch=2,
        ),
    )
    assert trained.table_payloads.shape == (2, NPQ0_S_TABLE_BYTES)
    assert np.isfinite(trained.weighted_nmse_percent).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nepq_a_residual_search_matches_tensor_reference() -> None:
    from mfq.quantize.nepq_a import _best_records

    generator = torch.Generator().manual_seed(20260814)
    blocks = torch.randn((5, 16, 8), generator=generator).to("mps")
    dictionary = (
        0.1 * torch.randn((1024, 8), generator=generator)
    ).to("mps")
    dictionary_norm = dictionary.square().sum(1)
    valid = torch.tensor([16, 13, 9, 5, 1], device="mps")
    records, gains = _best_records(
        blocks,
        dictionary,
        dictionary_norm,
        valid,
        position_bits=4,
        block_chunk=5,
    )
    score = 2.0 * torch.matmul(
        blocks.reshape(-1, 8), dictionary.T
    ).reshape(5, 16, 1024)
    score.sub_(dictionary_norm.reshape(1, 1, -1))
    positions = torch.arange(16, device="mps").reshape(1, -1, 1)
    score.masked_fill_(positions >= valid.reshape(-1, 1, 1), -torch.inf)
    expected_gain, best = score.reshape(5, -1).max(1)
    expected_position = torch.div(best, 1024, rounding_mode="floor")
    expected_dictionary = best.remainder(1024)
    expected_record = expected_position | (expected_dictionary << 4)
    assert torch.equal(records, expected_record)
    torch.testing.assert_close(gains, expected_gain, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nepq_a_quantization_end_to_end() -> None:
    import numpy as np

    from mfq.formats.nepq import NEPQ0_A, NEPQ0_S, validate_nepq
    from mfq.quantize.nepq import NepqQuantConfig
    from mfq.quantize.nepq_a import (
        NepqAArtifact,
        NepqAQuantConfig,
        quantize_nepq_a_fixed,
    )
    from tests.test_formats.test_nepq import _tables

    rng = np.random.default_rng(20260814)
    weight = rng.normal(0.0, 0.2, size=(1, 2, 104)).astype(np.float32)
    artifact = NepqAArtifact(
        _tables(NEPQ0_S),
        rng.normal(0.0, 0.05, size=(1024, 8)).astype(np.float32),
    )
    tensor = quantize_nepq_a_fixed(
        weight,
        NEPQ0_A,
        artifact,
        rotation_block=8,
        config=NepqAQuantConfig(
            base=NepqQuantConfig(
                anchor_multipliers=(1.0,),
                refine_steps=0,
                row_chunk=2,
                bank_chunk=2,
            ),
            residual_block_chunk=32,
        ),
        device="mps",
    )
    validate_nepq(tensor)
    assert tensor.residual_first.shape[:2] == (1, 2)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
@pytest.mark.parametrize("family", ["s", "l"])
def test_metal_nepq1_pool_assignment_recovers_feasible_weight(
    family: str,
) -> None:
    import numpy as np

    from mfq.formats.nepq import (
        NEPQ1_L,
        NEPQ1_S,
        dequantize_nepq,
        pack_nepq,
        unpack_nepq,
    )
    from mfq.quantize.nepq import NepqQuantConfig, quantize_nepq_fixed
    from tests.test_formats.test_nepq import _tensor

    spec = NEPQ1_S if family == "s" else NEPQ1_L
    source = unpack_nepq(pack_nepq(_tensor(spec)))
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
        device="mps",
    )
    reconstruction = dequantize_nepq(result)
    relative = np.linalg.norm(reconstruction - weight) / np.linalg.norm(weight)
    assert relative < 2e-6


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
def test_metal_nepq1_a_quantization_end_to_end() -> None:
    import numpy as np

    from mfq.formats.nepq import NEPQ1_A, NEPQ1_S, validate_nepq
    from mfq.quantize.nepq import NepqQuantConfig
    from mfq.quantize.nepq_a import (
        NepqAArtifact,
        NepqAQuantConfig,
        quantize_nepq_a_fixed,
    )
    from tests.test_formats.test_nepq import _tables

    rng = np.random.default_rng(20260815)
    weight = rng.normal(0.0, 0.2, size=(1, 2, 104)).astype(np.float32)
    artifact = NepqAArtifact(
        _tables(NEPQ1_S),
        rng.normal(0.0, 0.05, size=(1024, 8)).astype(np.float32),
    )
    tensor = quantize_nepq_a_fixed(
        weight,
        NEPQ1_A,
        artifact,
        rotation_block=8,
        second_records=0,
        config=NepqAQuantConfig(
            base=NepqQuantConfig(
                anchor_multipliers=(1.0,),
                refine_steps=0,
                row_chunk=2,
                bank_chunk=2,
            ),
            residual_block_chunk=32,
        ),
        device="mps",
    )
    validate_nepq(tensor)
    assert tensor.residual_first.shape[:2] == (1, 2)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal/MPS is unavailable",
)
@pytest.mark.parametrize("family", ["s", "l"])
def test_metal_general_nepq0_pool_recovers_feasible_weight(
    family: str,
) -> None:
    import numpy as np

    from mfq.formats.nepq import (
        NEPQ0_L,
        NEPQ0_S,
        dequantize_nepq,
        pack_nepq,
        unpack_nepq,
    )
    from mfq.quantize.nepq import NepqQuantConfig, quantize_nepq_fixed
    from tests.test_formats.test_nepq import _tensor

    spec = NEPQ0_S if family == "s" else NEPQ0_L
    source = unpack_nepq(pack_nepq(_tensor(spec)))
    weight = dequantize_nepq(source)
    result = quantize_nepq_fixed(
        weight,
        spec,
        source.table_payloads,
        importance=np.ones(source.neuron_len, dtype=np.float32),
        initial_anchor=source.neuron_scale,
        config=NepqQuantConfig(
            anchor_multipliers=(1.0,),
            refine_steps=0,
            row_chunk=2,
            bank_chunk=1,
        ),
        device="mps",
    )
    reconstruction = dequantize_nepq(result)
    relative = np.linalg.norm(reconstruction - weight) / np.linalg.norm(weight)
    assert relative < 2e-6
