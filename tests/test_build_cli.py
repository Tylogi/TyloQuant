from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfq.commands import build


ROOT = Path(__file__).parents[1]


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


def test_cuda_build_plan_is_native_and_does_not_import_or_configure_torch(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "source"
    (root / "cpp_runtime").mkdir(parents=True)
    (root / "cpp_runtime" / "CMakeLists.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(build, "repository_root", lambda: root)
    monkeypatch.setattr(build, "detect_backend", lambda _: "cuda")
    monkeypatch.setattr(
        build.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"cmake", "ninja"} else None,
    )

    plan = build.create_build_plan(
        backend="cuda",
        build_dir=tmp_path / "output",
        jobs=2,
    )

    command = " ".join(plan.configure_command)
    assert plan.target == "mfq-decode"
    assert "MFQ_BUILD_TORCH_REFERENCE_RUNTIME=OFF" in command
    assert "CMAKE_PREFIX_PATH" not in command
    assert "Python_EXECUTABLE" not in command
    source = (ROOT / "mfq" / "commands" / "build.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source


def test_default_cuda_cmake_target_has_no_python_or_libtorch_dependency() -> None:
    cmake = (ROOT / "cpp_runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
    native_start = cmake.index("add_executable(mfq-decode\n")
    reference_start = cmake.index("option(\n    MFQ_BUILD_TORCH_REFERENCE_RUNTIME")
    native_target = cmake[native_start:reference_start]

    assert "MFQ_NATIVE_CUDA_RUNTIME=1" in native_target
    assert "mfq-cuda-core" in native_target
    assert "mfq-cuda-native-kernels" in native_target
    assert "TORCH_LIBRARIES" not in native_target
    assert "Python::Python" not in native_target
    assert "find_package(Torch" not in native_target
    assert "find_package(Python" not in native_target

    assert "MFQ_BUILD_TORCH_REFERENCE_RUNTIME" in cmake
    assert "find_package(Torch REQUIRED)" in cmake[reference_start:]
    assert "add_executable(mfq-decode-torch" in cmake[reference_start:]


def test_native_cuda_runtime_compilation_units_do_not_include_torch() -> None:
    cmake = (ROOT / "cpp_runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
    source_block = cmake.split("set(MFQ_CUDA_KERNEL_SOURCES", 1)[1].split(")", 1)[0]
    sources = [
        ROOT / "cpp_runtime" / "mfq_decode.cpp",
        ROOT / "cpp_runtime" / "minicpmo45_runtime.inc",
        *(
            ROOT / "mfq" / "kernels" / "cuda" / name
            for name in re.findall(r"\.\./mfq/kernels/cuda/([^\s]+\.cu)", source_block)
        ),
    ]
    forbidden = ("<torch", "<ATen", "<c10", "torch::", "at::", "c10::")

    assert len(sources) > 20
    for source_path in sources:
        source = source_path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), source_path

    native_start = cmake.index("add_executable(mfq-decode\n")
    reference_start = cmake.index("option(\n    MFQ_BUILD_TORCH_REFERENCE_RUNTIME")
    assert "mfq_cuda.cpp" not in cmake[native_start:reference_start]


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
