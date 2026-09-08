from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.header import FileHeader
from mfq.formats.moe import NintMoePool, NintMoeTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize


@pytest.fixture(scope="module")
def model_parallel_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    binary_value = os.environ.get("MFQ_DECODE_NATIVE_TEST")
    if not binary_value:
        pytest.skip("MFQ_DECODE_NATIVE_TEST is required")
    binary = Path(binary_value)
    if not binary.is_file():
        pytest.skip(f"native CUDA runtime does not exist: {binary}")

    rng = np.random.default_rng(20260908)
    spec = NintSpec(4, 24, 6)
    neuron_len = 384
    output = 128
    experts = 8
    output_per_expert = 16
    dense = quantize(
        rng.normal(0.0, 0.04, (output, neuron_len)).astype(np.float32),
        spec,
        axis=0,
    )
    expert_weight = quantize(
        rng.normal(
            0.0,
            0.04,
            (experts * output_per_expert, neuron_len),
        ).astype(np.float32),
        spec,
        axis=0,
    )
    routed = NintMoeTensor(
        (experts, output_per_expert, neuron_len),
        (NintMoePool(np.arange(experts, dtype=np.int32), expert_weight),),
    )
    path = tmp_path_factory.mktemp("model-parallel") / "fixture.mfq"
    io.save(
        path,
        FileHeader(version=2, model_arch="model-parallel-test"),
        {"dense.weight": dense, "experts.weight": routed},
    )
    return binary, path


def _duplicate_rank_args(option: str, ranks: int) -> list[str]:
    return [
        option,
        ",".join("0" for _ in range(ranks)),
        option.replace("parallel", "split"),
        ",".join("1" for _ in range(ranks)),
        "--parallel-test-duplicates",
    ]


@pytest.mark.parametrize("ranks", (4, 8))
@pytest.mark.parametrize("axis", ("output", "input"))
def test_duplicate_rank_tensor_parallel_matches_single_device(
    model_parallel_fixture: tuple[Path, Path],
    ranks: int,
    axis: str,
) -> None:
    binary, model = model_parallel_fixture
    completed = subprocess.run(
        [
            str(binary),
            "--mfq",
            str(model),
            *_duplicate_rank_args("--tensor-parallel", ranks),
            "--check-tp-linear",
            "dense.weight",
            "--check-tp-axis",
            axis,
            "--check-tp-m",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "tensor_parallel_check=1" in completed.stdout
    assert f"shards={ranks}" in completed.stdout


@pytest.mark.parametrize("ranks", (4, 8))
@pytest.mark.parametrize("option", ("--expert-parallel", "--tensor-parallel"))
def test_duplicate_rank_expert_ownership_matches_single_device(
    model_parallel_fixture: tuple[Path, Path],
    ranks: int,
    option: str,
) -> None:
    binary, model = model_parallel_fixture
    completed = subprocess.run(
        [
            str(binary),
            "--mfq",
            str(model),
            *_duplicate_rank_args(option, ranks),
            "--check-ep-moe",
            "experts.weight",
            "--check-ep-moe-tokens",
            "4",
            "--check-ep-moe-routes",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "expert_parallel_moe_check=1" in completed.stdout
    assert f"shards={ranks}" in completed.stdout


def test_combined_tensor_and_weighted_expert_parallel(
    model_parallel_fixture: tuple[Path, Path],
) -> None:
    binary, model = model_parallel_fixture
    devices = "0,0,0,0"
    completed = subprocess.run(
        [
            str(binary),
            "--mfq",
            str(model),
            "--tensor-parallel",
            devices,
            "--tensor-split",
            "1,1,1,1",
            "--expert-parallel",
            devices,
            "--expert-split",
            "1,1,2,4",
            "--parallel-test-duplicates",
            "--check-ep-moe",
            "experts.weight",
            "--check-ep-moe-tokens",
            "4",
            "--check-ep-moe-routes",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "expert_parallel_moe_check=1" in completed.stdout
    assert "shards=4" in completed.stdout
