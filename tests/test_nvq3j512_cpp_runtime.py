from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.formats import io  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.nvq import (  # noqa: E402
    D4_512,
    D4_1024,
    E8_1024,
    E8_4096,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4_512,
    NVQ3_D4_1024,
    NvqJscTensor,
    NvqSpec,
)


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


def _tensor(rows: int, neuron_len: int, seed: int) -> NvqJscTensor:
    rng = np.random.default_rng(seed)
    groups = neuron_len // NVQ3_D4_512.groupsize
    vectors = neuron_len // NVQ3_D4_512.vector_size
    sign_groups = neuron_len // 8
    codebook = (D4_512.astype(np.int16) * 8).astype(np.int8)
    return NvqJscTensor(
        shape=(rows, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.uniform(0.002, 0.02, rows).astype(np.float32),
        scale_lut=np.linspace(0.45, 1.55, 16, dtype=np.float32),
        bank_for_state=(np.arange(16, dtype=np.uint8) & 1),
        state=rng.integers(0, 16, (rows, groups), dtype=np.uint8),
        indices=rng.integers(
            0,
            NVQ3_D4_512.codebook_entries,
            (rows, vectors),
            dtype=np.uint16,
        ),
        signs=rng.integers(0, 128, (rows, sign_groups), dtype=np.uint8),
        codebooks=np.stack((codebook, codebook), axis=0),
        base_spec=NVQ3_D4_512,
    )


def _extended_tensor(
    spec: NvqSpec,
    base: np.ndarray,
    rows: int,
    neuron_len: int,
    seed: int,
) -> NvqJscTensor:
    rng = np.random.default_rng(seed)
    groups = neuron_len // spec.groupsize
    vectors = neuron_len // spec.vector_size
    sign_groups = neuron_len // 8
    codebook = (base.astype(np.int16) * 8).astype(np.int8)
    return NvqJscTensor(
        shape=(rows, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.uniform(0.002, 0.02, rows).astype(np.float32),
        scale_lut=np.linspace(0.45, 1.55, 16, dtype=np.float32),
        bank_for_state=(np.arange(16, dtype=np.uint8) & 1),
        state=rng.integers(0, 16, (rows, groups), dtype=np.uint8),
        indices=rng.integers(
            0,
            spec.codebook_entries,
            (rows, vectors),
            dtype=np.uint16,
        ),
        signs=rng.integers(0, 128, (rows, sign_groups), dtype=np.uint8),
        codebooks=np.stack((codebook, codebook), axis=0),
        base_spec=spec,
    )


@pytest.mark.parametrize("m", (1, 16, 65))
def test_cpp_nvq3j512_linear_paths_match_materialized_weight(tmp_path, m):
    executable = _executable()
    tensor = _tensor(37, 96, 20260726)
    model_path = tmp_path / "nvq3j512-linear.mfq"
    io.save(
        model_path,
        FileHeader(version=2, model_arch="nvq3j512-test"),
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
    assert fields["dtype"].startswith("NVQ12 ")
    assert float(fields["production_rel"]) < 0.02


@pytest.mark.parametrize(
    ("spec", "base", "format_id"),
    (
        (NVQ2_E8_1024, E8_1024, 13),
        (NVQ2_E8_4096, E8_4096, 14),
        (NVQ3_D4_1024, D4_1024, 15),
    ),
)
@pytest.mark.parametrize("m", (1, 16, 65))
def test_cpp_extended_nvq_linear_paths_match_materialized_weight(
    tmp_path,
    spec,
    base,
    format_id,
    m,
):
    executable = _executable()
    tensor = _extended_tensor(spec, base, 37, 96, 20260730 + format_id)
    model_path = tmp_path / f"nvq{format_id}-linear.mfq"
    io.save(
        model_path,
        FileHeader(version=2, model_arch="nvq-extended-test"),
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
    assert fields["dtype"].startswith(f"NVQ{format_id} ")
    assert float(fields["production_rel"]) < 0.02
