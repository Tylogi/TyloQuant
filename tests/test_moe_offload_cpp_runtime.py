from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.formats import io  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.formats.nepq import (  # noqa: E402
    NEPQ0_A,
    NEPQ0_S,
    NEPQ1_A,
    NEPQ1_L,
)
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.formats.nint8_zero import Nint8ZeroTensor  # noqa: E402
from mfq.quantize.nint_quant import quantize  # noqa: E402
from tests.mixed_family_fixtures import make_flat_family  # noqa: E402
from tests.test_formats.test_nepq import _tensor as make_nepq  # noqa: E402
from tests.test_formats.test_nepq_a import _a_tensor as make_nepq_a  # noqa: E402


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


def _nint8_zero(
    rows: int,
    neuron_len: int,
    seed: int,
) -> Nint8ZeroTensor:
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


def _aligned_nepq(spec):
    tensor = (
        make_nepq_a(spec)[0]
        if spec in (NEPQ0_A, NEPQ1_A)
        else make_nepq(spec)
    )
    rows = 8
    indices = np.arange(rows) % tensor.out_per_expert

    def select(value):
        if value is None:
            return None
        return np.take(value, indices, axis=1).copy()

    residual_second_mask = select(tensor.residual_second_mask)
    residual_second_records = None
    if residual_second_mask is not None:
        source_mask = np.asarray(tensor.residual_second_mask).reshape(-1)
        source_records = np.asarray(tensor.residual_second_records)
        source_dense = np.full(source_mask.size, -1, dtype=np.int32)
        source_dense[np.flatnonzero(source_mask)] = source_records
        source_dense = source_dense.reshape(tensor.residual_second_mask.shape)
        selected_dense = select(source_dense)
        residual_second_records = selected_dense[selected_dense >= 0].astype(
            np.uint16
        )
    return type(tensor)(
        spec=tensor.spec,
        shape=(tensor.n_experts, rows, tensor.neuron_len),
        neuron_scale=select(tensor.neuron_scale),
        state=select(tensor.state),
        indices=select(tensor.indices),
        aux=select(tensor.aux),
        bank_ids=select(tensor.bank_ids),
        table_payloads=tensor.table_payloads.copy(),
        rotation_block=tensor.rotation_block,
        rotation_seed=tensor.rotation_seed,
        residual_codebook=(
            None
            if tensor.residual_codebook is None
            else tensor.residual_codebook.copy()
        ),
        residual_first=select(tensor.residual_first),
        residual_second_mask=residual_second_mask,
        residual_second_records=residual_second_records,
        residual_padding_nbytes=tensor.residual_padding_nbytes,
    )


def _container(family: str) -> NintMoeTensor:
    experts = 4
    rows = 4
    neuron_len = 96
    if family == "NINT-MIXED":
        rng = np.random.default_rng(20260728)
        values = rng.normal(
            0.0, 0.05, (experts, rows, neuron_len)
        ).astype(np.float32)
        specs = (
            NintSpec(4, 24, 6),
            NintSpec(5, 28, 7),
            NintSpec(6, 24, 6),
            NintSpec(8, 48, 8),
        )
        return NintMoeTensor(
            (experts, rows, neuron_len),
            tuple(
                NintMoePool(
                    np.asarray([expert], dtype=np.int32),
                    quantize(values[expert], specs[expert], axis=0),
                )
                for expert in range(experts)
            ),
        )
    if family == "NINT4":
        rng = np.random.default_rng(20260728)
        tensor = quantize(
            rng.normal(
                0.0, 0.05, (experts * rows, neuron_len)
            ).astype(np.float32),
            NintSpec(4, 24, 6),
            axis=0,
        )
    elif family == "NINT8-0":
        tensor = _nint8_zero(
            experts * rows, neuron_len, 20260728
        )
    elif family in {
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3J",
        "NVQ3J-L",
        "NPQ0-S",
    }:
        tensor = make_flat_family(
            family,
            rows=experts * rows,
            neuron_len=neuron_len,
            seed=20260728,
        )
    elif family == "NEPQ0-S":
        tensor = _aligned_nepq(NEPQ0_S)
        return NintMoeTensor(
            tensor.shape,
            (
                NintMoePool(
                    np.arange(tensor.n_experts, dtype=np.int32),
                    tensor,
                ),
            ),
        )
    elif family == "NEPQ1-L":
        tensor = _aligned_nepq(NEPQ1_L)
        return NintMoeTensor(
            tensor.shape,
            (
                NintMoePool(
                    np.arange(tensor.n_experts, dtype=np.int32),
                    tensor,
                ),
            ),
        )
    elif family == "NEPQ0-A":
        tensor = _aligned_nepq(NEPQ0_A)
        return NintMoeTensor(
            tensor.shape,
            (
                NintMoePool(
                    np.arange(tensor.n_experts, dtype=np.int32),
                    tensor,
                ),
            ),
        )
    elif family == "NEPQ1-A":
        tensor = _aligned_nepq(NEPQ1_A)
        return NintMoeTensor(
            tensor.shape,
            (
                NintMoePool(
                    np.arange(tensor.n_experts, dtype=np.int32),
                    tensor,
                ),
            ),
        )
    else:
        raise ValueError(family)
    return NintMoeTensor(
        (experts, rows, neuron_len),
        (
            NintMoePool(
                np.arange(experts, dtype=np.int32),
                tensor,
            ),
        ),
    )


def _write_fixture(tmp_path: Path, family: str) -> Path:
    output = tmp_path / f"{family.lower()}.mfq"
    io.save(
        output,
        FileHeader(version=2, model_arch="moe-cache-test"),
        {"experts.weight": _container(family)},
    )
    return output


def _parse_fields(line: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in line.split()[1:])


def _run_check(
    model: Path,
    *,
    cached: bool,
    tokens: int = 1,
    mapped_gather: bool = False,
) -> tuple[dict[str, str], dict[str, str] | None]:
    executable = _executable()
    command = [
        str(executable),
        "--mfq",
        str(model),
        "--check-nintm-tensor",
        "experts.weight",
        "--check-nintm-tokens",
        str(tokens),
        "--check-nintm-routes",
        "2",
        "--check-nintm-reps",
        "3",
    ]
    if cached:
        command.extend(("--moe-gpu-cache-gb", "0.01"))
    environment = _runtime_env(executable)
    if mapped_gather:
        environment["MFQ_MOE_MAPPED_GATHER"] = "1"
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        env=environment,
        text=True,
        timeout=60,
    )
    result_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("nintm_tensor_check ")
    )
    stats_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("moe_cache_stats ")
    ]
    return (
        _parse_fields(result_line),
        _parse_fields(stats_lines[-1]) if stats_lines else None,
    )


def _run_profile_check(
    model: Path,
    profile: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    executable = _executable()
    completed = subprocess.run(
        [
            str(executable),
            "--mfq",
            str(model),
            "--check-nintm-tensor",
            "experts.weight",
            "--check-nintm-tokens",
            "1",
            "--check-nintm-routes",
            "2",
            "--check-nintm-reps",
            "3",
            "--moe-gpu-cache-gb",
            "0.01",
            "--moe-cache-profile",
            str(profile),
        ],
        check=True,
        capture_output=True,
        env=_runtime_env(executable),
        text=True,
        timeout=60,
    )
    prewarm_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("moe_cache_prewarm ")
    )
    stats_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("moe_cache_stats ")
    )
    return _parse_fields(prewarm_line), _parse_fields(stats_line)


@pytest.mark.parametrize(
    "family",
    (
        "NINT4",
        "NINT8-0",
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3J",
        "NVQ3J-L",
        "NPQ0-S",
        "NEPQ0-S",
        "NEPQ1-L",
        "NEPQ0-A",
        "NEPQ1-A",
    ),
)
def test_cached_nintm_is_bit_exact_to_resident(
    tmp_path: Path,
    family: str,
) -> None:
    model = _write_fixture(tmp_path, family)
    resident, resident_stats = _run_check(model, cached=False)
    cached, cache_stats = _run_check(model, cached=True)

    assert resident_stats is None
    assert cache_stats is not None
    assert cached["values"] == resident["values"]
    assert cached["checksum"] == resident["checksum"]
    assert cached["sqsum"] == resident["sqsum"]
    if family == "NINT4":
        assert cached["hetero"] == "1"
        assert int(cache_stats["hetero_dispatches"]) > 0
    assert int(cache_stats["demand_misses"]) > 0
    assert int(cache_stats["demand_hits"]) > 0
    assert int(cache_stats["prefetch_hits"]) > 0
    assert int(cache_stats["allocated_bytes"]) <= int(
        cache_stats["budget_bytes"]
    )


@pytest.mark.parametrize("family", ("NINT4", "NEPQ0-A", "NEPQ1-A"))
def test_cached_prefill_fallback_is_bit_exact_to_resident(
    tmp_path: Path,
    family: str,
) -> None:
    model = _write_fixture(tmp_path, family)
    resident, _ = _run_check(model, cached=False, tokens=13)
    cached, cache_stats = _run_check(model, cached=True, tokens=13)

    assert cached["values"] == resident["values"]
    assert cache_stats is not None
    assert int(cache_stats["full_projection_fallbacks"]) > 0


@pytest.mark.parametrize("family", ("NEPQ0-A", "NEPQ1-A"))
def test_cached_nepq_a_mapped_gather_is_bit_exact_to_resident(
    tmp_path: Path,
    family: str,
) -> None:
    model = _write_fixture(tmp_path, family)
    resident, _ = _run_check(model, cached=False)
    cached, cache_stats = _run_check(
        model,
        cached=True,
        mapped_gather=True,
    )

    assert cached["values"] == resident["values"]
    assert cached["checksum"] == resident["checksum"]
    assert cached["sqsum"] == resident["sqsum"]
    assert cache_stats is not None
    assert int(cache_stats["mapped_gather_bytes"]) > 0
    assert int(cache_stats["mapped_gather_submissions"]) > 0


def test_cached_prefill_mma_is_bit_exact_to_resident(
    tmp_path: Path,
) -> None:
    model = _write_fixture(tmp_path, "NINT-MIXED")
    resident, _ = _run_check(model, cached=False, tokens=256)
    cached, cache_stats = _run_check(model, cached=True, tokens=256)

    assert cached["values"] == resident["values"]
    assert cached["checksum"] == resident["checksum"]
    assert cached["sqsum"] == resident["sqsum"]
    assert cache_stats is not None
    assert int(cache_stats["full_projection_fallbacks"]) > 0


def test_generic_cache_rejects_legacy_layer_offload(
    tmp_path: Path,
) -> None:
    model = _write_fixture(tmp_path, "NINT4")
    executable = _executable()
    completed = subprocess.run(
        [
            str(executable),
            "--mfq",
            str(model),
            "--check-nintm-tensor",
            "experts.weight",
            "--moe-gpu-cache-gb",
            "0.01",
            "--cpu-offload-layers",
            "0",
        ],
        capture_output=True,
        env=_runtime_env(executable),
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "cannot be combined" in completed.stderr


def test_profile_prewarms_routes_without_polluting_runtime_counters(
    tmp_path: Path,
) -> None:
    model = _write_fixture(tmp_path, "NINT4")
    profile = tmp_path / "profile.json"
    profile.write_text(
        """
        {
          "version": 1,
          "layers": {
            "0": {"ranking": [0, 3]}
          }
        }
        """,
        encoding="utf-8",
    )

    prewarm, stats = _run_profile_check(model, profile)

    assert int(prewarm["expert_bundles"]) == 2
    assert int(prewarm["projection_entries"]) == 2
    assert int(prewarm["h2d_bytes"]) > 0
    assert float(prewarm["time_ms"]) >= 0.0
    assert int(stats["demand_misses"]) == 0
    assert int(stats["h2d_bytes"]) == 0
    assert int(stats["demand_hits"]) > 0


def test_profile_requires_gpu_cache(tmp_path: Path) -> None:
    model = _write_fixture(tmp_path, "NINT4")
    profile = tmp_path / "profile.json"
    profile.write_text(
        '{"version": 1, "layers": {}}',
        encoding="utf-8",
    )
    executable = _executable()
    completed = subprocess.run(
        [
            str(executable),
            "--mfq",
            str(model),
            "--check-nintm-tensor",
            "experts.weight",
            "--moe-cache-profile",
            str(profile),
        ],
        capture_output=True,
        env=_runtime_env(executable),
        text=True,
        timeout=60,
    )

    assert completed.returncode != 0
    assert "requires --moe-gpu-cache-gb" in completed.stderr
