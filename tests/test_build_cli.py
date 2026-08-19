from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfq.commands import build


def test_cli_parser_does_not_import_quantization_or_server_frameworks() -> None:
    script = """
import sys
from mfq.cli import _build_parser
_build_parser()
heavy = sorted(
    name for name in sys.modules
    if name == 'torch'
    or name.startswith('mfq.quantize')
    or name.startswith('mfq.calibration')
    or name in {'fastapi', 'uvicorn'}
)
if heavy:
    raise SystemExit('eager imports: ' + ', '.join(heavy))
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_mlx_namespace_package_root_is_detected(tmp_path: Path, monkeypatch) -> None:
    mlx_root = tmp_path / "mlx"
    (mlx_root / "lib").mkdir(parents=True)
    (mlx_root / "lib" / "mlx.metallib").write_bytes(b"metal")
    monkeypatch.setitem(
        build.sys.modules,
        "mlx",
        SimpleNamespace(__file__=None, __path__=[str(mlx_root)]),
    )

    assert build._mlx_root() == mlx_root


def test_detect_backend_selects_metal_on_apple_silicon(monkeypatch) -> None:
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build.platform, "machine", lambda: "arm64")

    assert build.detect_backend() == "metal"


def test_detect_backend_selects_cuda_when_nvcc_is_available(monkeypatch) -> None:
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(build, "_cuda_compiler", lambda: Path("/opt/cuda/bin/nvcc"))

    assert build.detect_backend() == "cuda"


def test_detect_backend_rejects_an_unsupported_host(monkeypatch) -> None:
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build.platform, "machine", lambda: "x86_64")

    with pytest.raises(build.BuildError, match="no supported inference accelerator"):
        build.detect_backend()


def test_build_plan_selects_only_the_metal_server_target_and_forwards_cmake_args(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "source"
    (root / "cpp_runtime").mkdir(parents=True)
    (root / "cpp_runtime" / "CMakeLists.txt").write_text("", encoding="utf-8")
    mlx = tmp_path / "mlx"
    monkeypatch.setattr(build, "repository_root", lambda: root)
    monkeypatch.setattr(build, "detect_backend", lambda _: "metal")
    monkeypatch.setattr(build, "_mlx_root", lambda: mlx)
    monkeypatch.setattr(
        build.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"cmake", "ninja"} else None,
    )

    plan = build.create_build_plan(
        backend="auto",
        build_dir=tmp_path / "output",
        jobs=3,
        cmake_args=["-DMFQ_EXPERIMENTAL_KERNEL=ON"],
    )

    assert plan.backend == "metal"
    assert plan.target == "mfq-decode-metal"
    assert "-DMFQ_BUILD_CPP_SERVER=ON" in plan.configure_command
    assert "-DMFQ_BUILD_METAL_RUNTIME=ON" in plan.configure_command
    assert "-DMFQ_EXPERIMENTAL_KERNEL=ON" in plan.configure_command
    assert plan.build_command[-2:] == ("-j", "3")


def test_double_dash_arguments_are_forwarded_without_the_separator(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build.add_parser(subparsers)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        build,
        "build_runtime",
        lambda **options: captured.update(options) or Path("runtime"),
    )

    args = parser.parse_args(["build", "--", "-DMFQ_CUDA_ARCHITECTURES=90"])
    assert args._impl(args) == 0
    assert captured["cmake_args"] == ["-DMFQ_CUDA_ARCHITECTURES=90"]


def test_custom_build_is_recorded_and_loaded_for_serve(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "source"
    (root / "cpp_runtime").mkdir(parents=True)
    (root / "cpp_runtime" / "CMakeLists.txt").write_text("", encoding="utf-8")
    mlx = tmp_path / "mlx"
    monkeypatch.setattr(build, "repository_root", lambda: root)
    monkeypatch.setattr(build, "detect_backend", lambda _: "metal")
    monkeypatch.setattr(build, "_mlx_root", lambda: mlx)
    monkeypatch.setattr(
        build.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"cmake", "ninja"} else None,
    )
    custom_build = tmp_path / "custom-native-output"
    executable = custom_build / "metal" / "mfq-decode-metal"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime")
    plan = build.create_build_plan(
        backend="metal",
        build_dir=custom_build,
        build_type="RelWithDebInfo",
        generator="Ninja",
        cmake_args=["-DMFQ_EXPERIMENTAL_KERNEL=ON"],
    )

    manifest = build.record_managed_build(plan, executable)
    managed = build.load_managed_build("metal")

    assert manifest == root / "build" / "mfq-runtime.json"
    assert managed is not None
    assert managed.executable == executable.resolve()
    assert managed.build_dir == custom_build.resolve()
    assert managed.build_type == "RelWithDebInfo"
    assert managed.generator == "Ninja"
    assert managed.cmake_args == ("-DMFQ_EXPERIMENTAL_KERNEL=ON",)
