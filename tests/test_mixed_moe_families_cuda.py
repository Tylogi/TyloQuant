from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)
if shutil.which("cl") is None and shutil.which("cl.exe") is None:
    pytest.skip("MSVC cl unavailable", allow_module_level=True)

from mfq.formats import io  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.formats.nepq import (  # noqa: E402
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_L,
    NEPQ1_S,
    dequantize_nepq,
)
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.kernels.cuda._ext import ext  # noqa: E402
from mfq.kernels.cuda.moe import MoeRoutePlan, grouped_matmul, to_gpu  # noqa: E402
from mfq.quantize.nint_quant import quantize  # noqa: E402
from tests.mixed_family_fixtures import (  # noqa: E402
    FLAT_FAMILIES,
    dequantize_flat_family,
    make_flat_family,
)
from tests.test_formats.test_nepq import _tensor as make_nepq  # noqa: E402


def _ids(tokens: int) -> torch.Tensor:
    return torch.tensor(
        [[index % 2, (index + 1) % 2] for index in range(tokens)],
        device="cuda",
        dtype=torch.int32,
    )


def _selected_reference(
    dense: torch.Tensor,
    x: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty(
        ids.shape[0], ids.shape[1], dense.shape[1],
        device=x.device, dtype=torch.float32,
    )
    for token in range(ids.shape[0]):
        for route in range(ids.shape[1]):
            result[token, route] = x[token].float() @ dense[int(ids[token, route])].T
    return result


@pytest.mark.parametrize("family", FLAT_FAMILIES)
@pytest.mark.parametrize("tokens", (1, 4, 13))
def test_nintm_grouped_kernel_supports_flat_family(family: str, tokens: int):
    tensor = make_flat_family(family)
    container = NintMoeTensor(
        (2, 3, 96),
        (NintMoePool(np.arange(2, dtype=np.int32), tensor),),
    )
    weight = to_gpu(container)
    x = torch.randn(tokens, 96, device="cuda", dtype=torch.float16) * 0.1
    ids = _ids(tokens)
    actual = grouped_matmul(weight, x, MoeRoutePlan.build(ids, 2)).float()
    dense = torch.as_tensor(
        dequantize_flat_family(tensor).reshape(2, 3, 96),
        device="cuda",
    )
    expected = _selected_reference(dense, x, ids)
    relative = float((actual - expected).norm() / expected.norm())
    assert relative < 0.02, (family, tokens, relative)


@pytest.mark.parametrize("spec", (NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L))
@pytest.mark.parametrize("tokens", (1, 4, 13))
def test_nintm_grouped_kernel_supports_nepq_family(spec, tokens: int):
    tensor = make_nepq(spec)
    container = NintMoeTensor(
        tensor.shape,
        (NintMoePool(np.arange(2, dtype=np.int32), tensor),),
    )
    weight = to_gpu(container)
    packed = weight.pools[0].weight
    x = torch.randn(
        tokens, tensor.neuron_len, device="cuda", dtype=torch.float16
    ) * 0.1
    rotated = ext().nepq_hadamard_input_cuda(
        x, packed["rotation_signs"], packed["rotation_block"]
    )
    ids = _ids(tokens)
    actual = grouped_matmul(weight, x, MoeRoutePlan.build(ids, 2)).float()
    dense = torch.as_tensor(dequantize_nepq(tensor), device="cuda")
    expected = _selected_reference(dense, rotated, ids)
    relative = float((actual - expected).norm() / expected.norm())
    assert relative < 0.02, (spec.label, tokens, relative)


def _all_family_container() -> NintMoeTensor:
    rows = 3
    neuron_len = 104
    pools = []
    next_expert = 0
    rng = np.random.default_rng(20260723)
    for spec in (
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 24, 7),
        NintSpec(8, 48, 7),
    ):
        tensor = quantize(
            rng.normal(0, 0.04, (rows, neuron_len)).astype(np.float32),
            spec,
            axis=0,
        )
        pools.append(
            NintMoePool(np.asarray([next_expert], dtype=np.int32), tensor)
        )
        next_expert += 1
    for family in FLAT_FAMILIES:
        pools.append(
            NintMoePool(
                np.asarray([next_expert], dtype=np.int32),
                make_flat_family(family, rows=rows, neuron_len=neuron_len),
            )
        )
        next_expert += 1
    for spec in (NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L):
        tensor = make_nepq(spec)
        pools.append(
            NintMoePool(
                np.arange(next_expert, next_expert + tensor.n_experts, dtype=np.int32),
                tensor,
            )
        )
        next_expert += tensor.n_experts
    return NintMoeTensor(
        (next_expert, rows, neuron_len),
        tuple(pools),
    )


def test_cpp_runtime_matches_python_for_all_nintm_families(tmp_path):
    executable = (
        Path(__file__).resolve().parents[1]
        / "build"
        / "cpp_runtime"
        / "mfq-decode.exe"
    )
    if not executable.exists():
        pytest.skip("C++ runtime is not built")

    tensor_name = "all.families"
    tensor = _all_family_container()
    model_path = tmp_path / "all-families.mfq"
    io.save(
        model_path,
        FileHeader(version=2, model_arch="nintm-test"),
        {tensor_name: tensor},
    )
    _, stored_tensors = io.load(model_path)
    tensor = stored_tensors[tensor_name]

    tokens = 3
    routes = 8
    count = tokens * tensor.neuron_len
    sequence = torch.arange(count, device="cuda", dtype=torch.float32)
    x = (
        (sequence.remainder(257) - 128.0) / 127.0
        + 0.03125 * torch.sin(sequence * 0.015625)
    ).to(torch.float16).reshape(tokens, tensor.neuron_len).contiguous()
    ids = torch.as_tensor(
        [
            [
                (token * routes + route * 3) % tensor.n_experts
                for route in range(routes)
            ]
            for token in range(tokens)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    expected = grouped_matmul(
        to_gpu(tensor),
        x,
        MoeRoutePlan.build(ids, tensor.n_experts),
    ).float().cpu().reshape(-1)

    env = os.environ.copy()
    torch_lib = Path(torch.__path__[0]) / "lib"
    path_parts = [str(executable.parent), str(torch_lib)]
    if cuda_root := env.get("CUDA_PATH"):
        path_parts.append(str(Path(cuda_root) / "bin"))
    path_parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(path_parts)
    completed = subprocess.run(
        [
            str(executable),
            "--mfq",
            str(model_path),
            "--check-nintm-tensor",
            tensor_name,
            "--check-nintm-tokens",
            str(tokens),
            "--check-nintm-routes",
            str(routes),
            "--check-nintm-reps",
            "2",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=60,
    )
    line = next(
        value
        for value in completed.stdout.splitlines()
        if value.startswith("nintm_tensor_check ")
    )
    fields = dict(part.split("=", 1) for part in line.split()[1:])
    actual = torch.as_tensor(
        [float(value) for value in fields["values"].split(",")],
        dtype=torch.float32,
    )
    assert fields["mixed"] == "1"
    expected = expected[: actual.numel()]
    diagnostics = []
    for token in range(tokens):
        for route_index in range(routes):
            offset = (token * routes + route_index) * tensor.out_per_expert
            got = actual[offset : offset + tensor.out_per_expert]
            ref = expected[offset : offset + tensor.out_per_expert]
            max_abs = float((got - ref).abs().max())
            if max_abs > 2e-5:
                expert = int(ids[token, route_index])
                diagnostics.append(
                    (
                        token,
                        route_index,
                        expert,
                        tensor.expert_profiles[expert],
                        max_abs,
                        float((got - ref).norm() / ref.norm().clamp_min(1e-30)),
                    )
                )
    diagnostics.sort(key=lambda value: value[4], reverse=True)
    assert not diagnostics, diagnostics[:10]
