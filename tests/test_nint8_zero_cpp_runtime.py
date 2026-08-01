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

from mfq.formats import io  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.formats.nint8_zero import (  # noqa: E402
    Nint8ZeroTensor,
    dequantize_nint8_zero,
)
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.quantize.nint_quant import quantize  # noqa: E402


def _executable() -> Path:
    path = (
        Path(__file__).resolve().parents[1]
        / "build"
        / "cpp_runtime"
        / "mfq-decode.exe"
    )
    if not path.exists():
        pytest.skip("C++ runtime is not built")
    return path


def _runtime_env(executable: Path) -> dict[str, str]:
    env = os.environ.copy()
    torch_lib = Path(torch.__path__[0]) / "lib"
    path_parts = [str(executable.parent), str(torch_lib)]
    if cuda_root := env.get("CUDA_PATH"):
        path_parts.append(str(Path(cuda_root) / "bin"))
    path_parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(path_parts)
    return env


def _tensor(rows: int, neuron_len: int, seed: int) -> Nint8ZeroTensor:
    rng = np.random.default_rng(seed)
    groups = neuron_len // 32
    return Nint8ZeroTensor(
        shape=(rows, neuron_len),
        axis=0,
        scale=rng.uniform(0.0005, 0.01, (rows, groups)).astype(np.float16),
        q=rng.integers(
            -127, 128, (rows, groups, 32), dtype=np.int16
        ).astype(np.int8),
        neuron_len=neuron_len,
    )


@pytest.mark.parametrize("m", (1, 16, 65, 512))
def test_cpp_nint8_zero_linear_paths_match_materialized_weight(tmp_path, m):
    executable = _executable()
    tensor = _tensor(37, 96, 20260725)
    model_path = tmp_path / "nint8-zero-linear.mfq"
    io.save(
        model_path,
        FileHeader(version=2, model_arch="nint8-zero-test"),
        {"linear.weight": tensor},
    )
    completed = subprocess.run(
        [
            str(executable),
            "--mfq",
            str(model_path),
            "--check-linear",
            "linear.weight",
            "--check-linear-m",
            str(m),
        ],
        check=True,
        capture_output=True,
        env=_runtime_env(executable),
        text=True,
        timeout=60,
    )
    fields = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    assert fields["dtype"].startswith("NINT8")
    assert float(fields["production_rel"]) < 0.02


def test_cpp_tensor_overlay_replaces_base_record(tmp_path):
    executable = _executable()
    tensor = _tensor(37, 96, 20260727)
    rng = np.random.default_rng(20260728)
    base_tensor = quantize(
        rng.normal(0.0, 0.1, (37, 96)).astype(np.float32),
        NintSpec(4, 24, 6),
        axis=0,
    )
    base_path = tmp_path / "base.mfq"
    overlay_path = tmp_path / "overlay.mfq"
    io.save(
        base_path,
        FileHeader(version=2, model_arch="nint8-zero-test"),
        {"linear.weight": base_tensor},
    )
    io.save(
        overlay_path,
        FileHeader(version=2, model_arch="nint8-zero-test"),
        {"linear.weight": tensor},
    )
    env = _runtime_env(executable)
    env["MFQ_TENSOR_OVERLAY"] = str(overlay_path)
    completed = subprocess.run(
        [
            str(executable),
            "--mfq",
            str(base_path),
            "--check-linear",
            "linear.weight",
            "--check-linear-m",
            "1",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=60,
    )
    assert "dtype=NINT8-0" in completed.stdout


@pytest.mark.parametrize("tokens", (1, 13, 80, 512))
def test_cpp_nint8_zero_moe_paths_match_dense_reference(tmp_path, tokens):
    executable = _executable()
    experts, rows, neuron_len, routes = 4, 5, 96, 2
    tensor = _tensor(experts * rows, neuron_len, 20260726)
    container = NintMoeTensor(
        (experts, rows, neuron_len),
        (
            NintMoePool(
                np.arange(experts, dtype=np.int32),
                tensor,
            ),
        ),
    )
    model_path = tmp_path / "nint8-zero-moe.mfq"
    io.save(
        model_path,
        FileHeader(version=2, model_arch="nint8-zero-test"),
        {"experts.weight": container},
    )
    completed = subprocess.run(
        [
            str(executable),
            "--mfq",
            str(model_path),
            "--check-nintm-tensor",
            "experts.weight",
            "--check-nintm-tokens",
            str(tokens),
            "--check-nintm-routes",
            str(routes),
            "--check-nintm-reps",
            "2",
        ],
        check=True,
        capture_output=True,
        env=_runtime_env(executable),
        text=True,
        timeout=60,
    )
    line = next(
        value
        for value in completed.stdout.splitlines()
        if value.startswith("nintm_tensor_check ")
    )
    fields = dict(part.split("=", 1) for part in line.split()[1:])
    actual = torch.tensor(
        [float(value) for value in fields["values"].split(",")],
        dtype=torch.float32,
    )

    sequence = torch.arange(tokens * neuron_len, dtype=torch.float32)
    x = (
        (sequence.remainder(257) - 128.0) / 127.0
        + 0.03125 * torch.sin(sequence * 0.015625)
    ).to(torch.float16).reshape(tokens, neuron_len)
    dense = torch.from_numpy(
        dequantize_nint8_zero(tensor).reshape(
            experts, rows, neuron_len
        )
    )
    expected = torch.empty(tokens, routes, rows, dtype=torch.float32)
    for token in range(tokens):
        for route in range(routes):
            expert = (token * routes + route * 3) % experts
            expected[token, route] = x[token].float() @ dense[expert].T
    expected = expected.reshape(-1)[: actual.numel()]
    relative = float((actual - expected).norm() / expected.norm())
    assert relative < 0.02
    if tokens >= 256:
        assert float(fields["dense_reference_rel"]) == 0.0
