from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfq.cli import _build_parser
from mfq.commands.serve import _prepare_web_root, _resolve_runtime_executable, _run, _select_backend
from mfq.server.native import NativeRuntime, native_runtime_environment


def test_server_imports_framework_without_loading_quantization_stack() -> None:
    script = """
import importlib
import sys

for name in (
    'mfq.server.api',
    'mfq.server.catalog',
    'mfq.commands.serve',
    'mfq.server.native',
    'mfq.server.runtime_pool',
    'mfq.server.service',
):
    importlib.import_module(name)

unexpected = sorted(
    name for name in sys.modules
    if name == 'torch'
    or name == 'mfq.quantize'
    or name.startswith('mfq.quantize.')
    or name == 'mfq.calibration'
    or name.startswith('mfq.calibration.')
)
if unexpected:
    raise SystemExit('eager imports: ' + ', '.join(unexpected))
if 'fastapi' not in sys.modules:
    raise SystemExit('FastAPI was not loaded by the server application')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_serve_exposes_public_host_and_port_options(tmp_path: Path) -> None:
    defaults = _build_parser().parse_args(["serve"])
    args = _build_parser().parse_args(
        [
            "serve",
            "--model",
            str(tmp_path / "model.mfq"),
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
        ]
    )

    assert defaults.host == "127.0.0.1"
    assert defaults.port == 8090
    assert defaults.model is None
    assert defaults.running_executable is None
    assert defaults.access_log is True
    assert args.host == "0.0.0.0"
    assert args.port == 9001


def test_serve_accepts_an_empty_initial_model_catalog() -> None:
    args = _build_parser().parse_args(["serve"])

    assert args.model is None


def test_serve_accepts_a_prebuilt_native_runtime(tmp_path: Path) -> None:
    executable = tmp_path / "mfq-decode-metal"
    executable.write_bytes(b"runtime")

    assert _resolve_runtime_executable("metal", executable) == executable.resolve()


def test_serve_uses_the_runtime_recorded_by_mfq_build(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "custom" / "metal" / "mfq-decode-metal"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime")
    managed = SimpleNamespace(executable=executable)
    monkeypatch.setattr("mfq.commands.serve.load_managed_build", lambda _: managed)
    monkeypatch.setattr(
        "mfq.commands.serve.build_runtime",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )

    assert _resolve_runtime_executable("metal") == executable


def test_serve_uses_an_explicit_prebuilt_runtime_without_managed_lookup(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "release" / "mfq-decode-metal"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime")
    monkeypatch.setattr(
        "mfq.commands.serve.load_managed_build",
        lambda _: (_ for _ in ()).throw(AssertionError("must not inspect managed builds")),
    )
    monkeypatch.setattr(
        "mfq.commands.serve.build_runtime",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    assert _resolve_runtime_executable("metal", executable) == executable.resolve()


def test_serve_rejects_a_missing_explicit_prebuilt_runtime(tmp_path: Path) -> None:
    missing = tmp_path / "mfq-decode-metal"

    with pytest.raises(FileNotFoundError, match="--running-executable"):
        _resolve_runtime_executable("metal", missing)


def test_prebuilt_cuda_runtime_does_not_require_a_local_compiler(monkeypatch) -> None:
    monkeypatch.setattr("mfq.commands.serve.platform.system", lambda: "Linux")
    monkeypatch.setattr("mfq.commands.serve.platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        "mfq.commands.serve.detect_backend",
        lambda _: (_ for _ in ()).throw(AssertionError("must not require nvcc")),
    )

    assert _select_backend("auto", Path("mfq-decode")) == "cuda"


def test_serve_rebuilds_a_missing_runtime_from_its_recorded_recipe(
    tmp_path: Path, monkeypatch
) -> None:
    managed = SimpleNamespace(
        executable=tmp_path / "custom" / "metal" / "mfq-decode-metal",
        build_dir=tmp_path / "custom",
        build_type="RelWithDebInfo",
        generator="Ninja",
        cmake_args=("-DMFQ_EXPERIMENTAL_KERNEL=ON",),
    )
    captured: dict[str, object] = {}
    rebuilt = tmp_path / "rebuilt-runtime"
    monkeypatch.setattr("mfq.commands.serve.load_managed_build", lambda _: managed)
    monkeypatch.setattr(
        "mfq.commands.serve.build_runtime",
        lambda **options: captured.update(options) or rebuilt,
    )

    assert _resolve_runtime_executable("metal") == rebuilt
    assert captured == {
        "backend": "metal",
        "build_dir": tmp_path / "custom",
        "build_type": "RelWithDebInfo",
        "generator": "Ninja",
        "cmake_args": ("-DMFQ_EXPERIMENTAL_KERNEL=ON",),
    }


def test_native_cuda_worker_is_private_and_uses_a_loopback_port(tmp_path: Path) -> None:
    runtime = NativeRuntime(
        executable=tmp_path / "mfq-decode",
        model=tmp_path / "model.mfq",
        model_name="model",
        backend="cuda",
        context_size=32768,
    )

    command = runtime.command(43123)

    assert command[:3] == [str(runtime.executable), "--mfq", str(runtime.model)]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "43123"
    assert command[command.index("--ctx-size") + 1] == "32768"
    assert "--prefill-chunk-size" not in command


def test_native_metal_worker_receives_prefill_chunk_size(tmp_path: Path) -> None:
    runtime = NativeRuntime(
        executable=tmp_path / "mfq-decode-metal",
        model=tmp_path / "model.mfq",
        model_name="model",
        backend="metal",
        prefill_chunk_size=4096,
    )

    command = runtime.command(43123)

    assert command[command.index("--prefill-chunk-size") + 1] == "4096"


def test_native_metal_worker_finds_release_resources(tmp_path: Path) -> None:
    executable = tmp_path / "sidecars" / "mfq-decode-metal"
    executable.parent.mkdir()
    executable.touch()
    resources = tmp_path / "Resources"
    resources.mkdir()
    metallib = resources / "mlx.metallib"
    metallib.touch()
    frameworks = tmp_path / "Frameworks"
    frameworks.mkdir()
    video_library = frameworks / "libmfq_avfoundation_video.dylib"
    video_library.touch()

    environment = native_runtime_environment(executable, "metal", {})

    assert environment["MFQ_MLX_METALLIB"] == str(metallib)
    assert environment["MFQ_AVFOUNDATION_VIDEO_LIBRARY"] == str(video_library)


def test_native_runtime_environment_preserves_explicit_resource_paths(tmp_path: Path) -> None:
    environment = native_runtime_environment(
        tmp_path / "mfq-decode-metal",
        "metal",
        {
            "MFQ_MLX_METALLIB": "/configured/mlx.metallib",
            "MFQ_AVFOUNDATION_VIDEO_LIBRARY": "/configured/video.dylib",
        },
    )

    assert environment == {
        "MFQ_MLX_METALLIB": "/configured/mlx.metallib",
        "MFQ_AVFOUNDATION_VIDEO_LIBRARY": "/configured/video.dylib",
    }


def test_serve_builds_web_ui_when_the_source_is_newer(tmp_path: Path, monkeypatch) -> None:
    web = tmp_path / "web"
    source = web / "src" / "App.tsx"
    source.parent.mkdir(parents=True)
    source.write_text("export {};", encoding="utf-8")
    (web / "package.json").write_text("{}", encoding="utf-8")
    (web / "package-lock.json").write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr("mfq.commands.serve._studio_dir", lambda: web)
    monkeypatch.setattr("mfq.commands.serve.shutil.which", lambda _: "/usr/bin/npm")

    def run(command, **_):
        calls.append(command)
        if command[-2:] == ["run", "build"]:
            (web / "dist").mkdir()
            (web / "dist" / "index.html").write_text("ready", encoding="utf-8")

    monkeypatch.setattr("mfq.commands.serve.subprocess.run", run)

    assert _prepare_web_root(None, disabled=False) == web / "dist"
    assert calls == [["/usr/bin/npm", "ci"], ["/usr/bin/npm", "run", "build"]]


def test_configured_web_root_must_contain_an_index(tmp_path: Path) -> None:
    web_root = tmp_path / "web"
    web_root.mkdir()

    with pytest.raises(FileNotFoundError, match="Web UI entry point does not exist"):
        _prepare_web_root(web_root, disabled=False)

    (web_root / "index.html").write_text("ready", encoding="utf-8")

    assert _prepare_web_root(web_root, disabled=False) == web_root


def test_environment_web_root_must_exist(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing-web"
    monkeypatch.setenv("MFQ_SERVER_WEB_ROOT", str(missing))

    with pytest.raises(FileNotFoundError, match="MFQ_SERVER_WEB_ROOT Web UI directory"):
        _prepare_web_root(None, disabled=False)


def test_serve_validates_web_ui_before_backend_build_or_model_load(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "model.mfq"
    model.write_bytes(b"model")
    missing_web_root = tmp_path / "missing-web"
    args = _build_parser().parse_args(
        [
            "serve",
            "--model",
            str(model),
            "--web-root",
            str(missing_web_root),
        ]
    )
    monkeypatch.setattr(
        "mfq.commands.serve.detect_backend",
        lambda _: (_ for _ in ()).throw(AssertionError("backend detection must not run")),
    )

    with pytest.raises(FileNotFoundError, match="--web-root Web UI directory"):
        _run(args)


def test_serve_can_disable_web_ui_build(monkeypatch) -> None:
    monkeypatch.setattr(
        "mfq.commands.serve._studio_dir",
        lambda: (_ for _ in ()).throw(AssertionError("should not inspect Studio")),
    )

    assert _prepare_web_root(None, disabled=True) is None


def test_serve_starts_without_loading_an_initial_model(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "mfq-decode-metal"
    executable.write_bytes(b"runtime")
    args = _build_parser().parse_args(
        [
            "serve",
            "--no-web-ui",
            "--backend",
            "metal",
            "--running-executable",
            str(executable),
            "--db",
            str(tmp_path / "server.sqlite3"),
            "--model-dir",
            str(tmp_path / "models"),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "mfq.server.native.NativeRuntime.start",
        lambda _: (_ for _ in ()).throw(AssertionError("must not start a model runtime")),
    )

    def run(app, **options):
        captured["app"] = app
        captured.update(options)

    monkeypatch.setattr("uvicorn.run", run)

    assert _run(args) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8090
    assert captured["access_log"] is True
