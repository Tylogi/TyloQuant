"""Build the native MFQ inference runtime through CMake."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class BuildError(RuntimeError):
    """Raised when the native runtime cannot be configured or built."""


@dataclass(frozen=True)
class BuildPlan:
    backend: str
    source_dir: Path
    build_dir: Path
    build_type: str
    generator: str | None
    cmake_args: tuple[str, ...]
    target: str
    executable: Path
    configure_command: tuple[str, ...]
    build_command: tuple[str, ...]


@dataclass(frozen=True)
class ManagedBuild:
    backend: str
    source_dir: Path
    build_dir: Path
    build_type: str
    generator: str | None
    cmake_args: tuple[str, ...]
    target: str
    executable: Path


_MANIFEST_VERSION = 1


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "cpp_runtime" / "CMakeLists.txt").is_file():
        raise BuildError(
            "the native runtime sources are not installed; run mfq from an MFQ source checkout"
        )
    return root


def build_manifest_path() -> Path:
    """Return the CLI-owned native build manifest for this source checkout."""

    return repository_root() / "build" / "mfq-runtime.json"


def _empty_manifest() -> dict[str, object]:
    return {"version": _MANIFEST_VERSION, "builds": {}}


def _read_manifest_document(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _empty_manifest()
    if not isinstance(document, dict) or document.get("version") != _MANIFEST_VERSION:
        return _empty_manifest()
    if not isinstance(document.get("builds"), dict):
        return _empty_manifest()
    return document


def _write_manifest_document(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_managed_build(plan: BuildPlan, executable: Path) -> Path:
    """Persist the actual runtime produced by a successful ``mfq build``."""

    path = build_manifest_path()
    document = _read_manifest_document(path)
    builds = document["builds"]
    assert isinstance(builds, dict)
    builds[plan.backend] = {
        "system": platform.system(),
        "machine": platform.machine().lower(),
        "source_dir": str(plan.source_dir),
        "build_dir": str(plan.build_dir),
        "build_type": plan.build_type,
        "generator": plan.generator,
        "cmake_args": list(plan.cmake_args),
        "target": plan.target,
        "executable": str(executable.resolve()),
    }
    _write_manifest_document(path, document)
    return path


def load_managed_build(backend: str) -> ManagedBuild | None:
    """Load a build for this checkout and host, including a missing artifact's recipe."""

    root = repository_root()
    document = _read_manifest_document(build_manifest_path())
    builds = document["builds"]
    assert isinstance(builds, dict)
    record = builds.get(backend)
    if not isinstance(record, dict):
        return None
    if record.get("system") != platform.system():
        return None
    if record.get("machine") != platform.machine().lower():
        return None
    required_strings = ("source_dir", "build_dir", "build_type", "target", "executable")
    if any(not isinstance(record.get(name), str) for name in required_strings):
        return None
    generator = record.get("generator")
    if generator is not None and not isinstance(generator, str):
        return None
    raw_cmake_args = record.get("cmake_args", [])
    if not isinstance(raw_cmake_args, list) or not all(
        isinstance(value, str) for value in raw_cmake_args
    ):
        return None
    source_dir = Path(record["source_dir"]).expanduser().resolve()
    if source_dir != (root / "cpp_runtime").resolve():
        return None
    return ManagedBuild(
        backend=backend,
        source_dir=source_dir,
        build_dir=Path(record["build_dir"]).expanduser().resolve(),
        build_type=record["build_type"],
        generator=generator,
        cmake_args=tuple(raw_cmake_args),
        target=record["target"],
        executable=Path(record["executable"]).expanduser().resolve(),
    )


def _cuda_compiler() -> Path | None:
    configured = os.environ.get("CUDACXX")
    if configured and Path(configured).is_file():
        return Path(configured)
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if not root:
            continue
        candidate = Path(root) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        if candidate.is_file():
            return candidate
    return None


def detect_backend(requested: str = "auto") -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if requested == "metal":
        if system != "Darwin" or machine not in {"arm64", "aarch64"}:
            raise BuildError("the Metal runtime requires Apple silicon and macOS")
        return "metal"
    if requested == "cuda":
        if system not in {"Linux", "Windows"}:
            raise BuildError("the CUDA runtime is supported on Linux and Windows")
        if _cuda_compiler() is None:
            raise BuildError("CUDA was selected but nvcc was not found")
        return "cuda"
    if requested != "auto":
        raise BuildError(f"unsupported inference backend: {requested}")
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "metal"
    if system in {"Linux", "Windows"} and _cuda_compiler() is not None:
        return "cuda"
    raise BuildError(
        "no supported inference accelerator was detected; MFQ requires Apple silicon/Metal "
        "on macOS or an NVIDIA CUDA toolkit on Linux or Windows"
    )


def _mlx_root() -> Path:
    try:
        import mlx
    except ModuleNotFoundError as error:
        raise BuildError("Metal runtime compilation requires the 'metal' extra (mlx)") from error
    package_paths = tuple(getattr(mlx, "__path__", ()))
    module_file = getattr(mlx, "__file__", None)
    if package_paths:
        root = Path(package_paths[0]).resolve()
    elif module_file:
        root = Path(module_file).resolve().parent
    else:
        raise BuildError("the installed MLX package location could not be determined")
    required = root / "lib" / "mlx.metallib"
    if not required.is_file():
        raise BuildError(f"the installed MLX package has no native runtime assets: {required}")
    return root


def _torch_cmake_prefix() -> str:
    try:
        from torch.utils import cmake_prefix_path
    except (ImportError, ModuleNotFoundError) as error:
        raise BuildError("CUDA runtime compilation requires PyTorch") from error
    return str(cmake_prefix_path)


def runtime_executable(build_dir: Path, backend: str) -> Path:
    if backend == "metal":
        return build_dir / "metal" / "mfq-decode-metal"
    suffix = ".exe" if os.name == "nt" else ""
    return build_dir / f"mfq-decode{suffix}"


def _find_built_executable(plan: BuildPlan) -> Path | None:
    if plan.executable.is_file():
        return plan.executable
    name = plan.executable.name
    candidates = [
        path
        for path in plan.build_dir.rglob(name)
        if path.is_file() and "CMakeFiles" not in path.parts
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def create_build_plan(
    *,
    backend: str = "auto",
    build_dir: Path | None = None,
    build_type: str = "Release",
    jobs: int | None = None,
    generator: str | None = None,
    cmake_args: Sequence[str] = (),
) -> BuildPlan:
    root = repository_root()
    selected = detect_backend(backend)
    cmake = shutil.which("cmake")
    if not cmake:
        raise BuildError("cmake was not found; install CMake 3.26 or newer")
    output = (build_dir or root / "build" / "cpp_runtime").expanduser().resolve()
    source = root / "cpp_runtime"
    target = "mfq-decode-metal" if selected == "metal" else "mfq-decode"
    configure = [
        cmake,
        "-S",
        str(source),
        "-B",
        str(output),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DMFQ_BUILD_CPP_SERVER=ON",
    ]
    selected_generator = generator
    if selected_generator is None and shutil.which("ninja"):
        selected_generator = "Ninja"
    if selected_generator:
        configure.extend(["-G", selected_generator])
    if selected == "metal":
        configure.extend(
            [
                "-DMFQ_BUILD_METAL_RUNTIME=ON",
                f"-DMFQ_MLX_ROOT={_mlx_root()}",
            ]
        )
    else:
        configure.extend(
            [
                "-DMFQ_BUILD_METAL_RUNTIME=OFF",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DCMAKE_PREFIX_PATH={_torch_cmake_prefix()}",
            ]
        )
    configure.extend(str(value) for value in cmake_args)
    parallelism = jobs if jobs is not None else max(1, os.cpu_count() or 1)
    if parallelism < 1:
        raise BuildError("--jobs must be positive")
    build = [
        cmake,
        "--build",
        str(output),
        "--config",
        build_type,
        "--target",
        target,
        "-j",
        str(parallelism),
    ]
    return BuildPlan(
        backend=selected,
        source_dir=source,
        build_dir=output,
        build_type=build_type,
        generator=selected_generator,
        cmake_args=tuple(str(value) for value in cmake_args),
        target=target,
        executable=runtime_executable(output, selected),
        configure_command=tuple(configure),
        build_command=tuple(build),
    )


def execute_build(plan: BuildPlan, *, dry_run: bool = False) -> Path:
    print(f"MFQ native backend: {plan.backend}", flush=True)
    print("Configure:", subprocess.list2cmdline(plan.configure_command), flush=True)
    print("Build:", subprocess.list2cmdline(plan.build_command), flush=True)
    if dry_run:
        return plan.executable
    try:
        subprocess.run(plan.configure_command, check=True)
        subprocess.run(plan.build_command, check=True)
    except subprocess.CalledProcessError as error:
        raise BuildError(
            f"native runtime build failed with exit code {error.returncode}"
        ) from error
    executable = _find_built_executable(plan)
    if executable is None:
        raise BuildError(f"CMake completed without producing {plan.executable}")
    print(f"Native runtime ready: {executable}", flush=True)
    return executable


def build_runtime(
    *,
    backend: str = "auto",
    build_dir: Path | None = None,
    build_type: str = "Release",
    jobs: int | None = None,
    generator: str | None = None,
    cmake_args: Sequence[str] = (),
    dry_run: bool = False,
) -> Path:
    plan = create_build_plan(
        backend=backend,
        build_dir=build_dir,
        build_type=build_type,
        jobs=jobs,
        generator=generator,
        cmake_args=cmake_args,
    )
    executable = execute_build(plan, dry_run=dry_run)
    if not dry_run:
        manifest = record_managed_build(plan, executable)
        print(f"Build manifest: {manifest}", flush=True)
    return executable


def _run(args: argparse.Namespace) -> int:
    cmake_args = args.cmake_args
    if cmake_args[:1] == ["--"]:
        cmake_args = cmake_args[1:]
    build_runtime(
        backend=args.backend,
        build_dir=args.build_dir,
        build_type=args.build_type,
        jobs=args.jobs,
        generator=args.generator,
        cmake_args=cmake_args,
        dry_run=args.dry_run,
    )
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "build",
        help="compile the native inference runtime for this machine",
        description=(
            "Detect Metal or CUDA and compile the matching native runtime. "
            "Arguments after '--' are forwarded to the CMake configure command."
        ),
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "metal"), default="auto")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("-j", "--jobs", type=int)
    parser.add_argument("--generator")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("cmake_args", nargs=argparse.REMAINDER, metavar="-- CMAKE_ARG")
    parser.set_defaults(_impl=_run)
