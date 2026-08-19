from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfq.cli import _build_parser
from mfq.server.cli import _prepare_web_root, _resolve_runtime_executable, _run
from mfq.server.native import NativeRuntime


def test_server_imports_framework_without_loading_quantization_stack() -> None:
    script = """
import importlib
import sys

for name in (
    'mfq.server.api',
    'mfq.server.catalog',
    'mfq.server.cli',
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
    defaults = _build_parser().parse_args(["serve", "--model", str(tmp_path / "model.mfq")])
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
    assert args.host == "0.0.0.0"
    assert args.port == 9001


def test_serve_uses_the_runtime_recorded_by_mfq_build(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "custom" / "metal" / "mfq-decode-metal"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime")
    managed = SimpleNamespace(executable=executable)
    monkeypatch.setattr("mfq.server.cli.load_managed_build", lambda _: managed)
    monkeypatch.setattr(
        "mfq.server.cli.build_runtime",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )

    assert _resolve_runtime_executable("metal") == executable


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
    monkeypatch.setattr("mfq.server.cli.load_managed_build", lambda _: managed)
    monkeypatch.setattr(
        "mfq.server.cli.build_runtime",
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


def test_serve_builds_web_ui_when_the_source_is_newer(tmp_path: Path, monkeypatch) -> None:
    web = tmp_path / "web"
    source = web / "src" / "App.tsx"
    source.parent.mkdir(parents=True)
    source.write_text("export {};", encoding="utf-8")
    (web / "package.json").write_text("{}", encoding="utf-8")
    (web / "package-lock.json").write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr("mfq.server.cli._studio_web_dir", lambda: web)
    monkeypatch.setattr("mfq.server.cli.shutil.which", lambda _: "/usr/bin/npm")

    def run(command, **_):
        calls.append(command)
        if command[-2:] == ["run", "build"]:
            (web / "dist").mkdir()
            (web / "dist" / "index.html").write_text("ready", encoding="utf-8")

    monkeypatch.setattr("mfq.server.cli.subprocess.run", run)

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
        "mfq.server.cli.detect_backend",
        lambda _: (_ for _ in ()).throw(AssertionError("backend detection must not run")),
    )

    with pytest.raises(FileNotFoundError, match="--web-root Web UI directory"):
        _run(args)


def test_serve_can_disable_web_ui_build(monkeypatch) -> None:
    monkeypatch.setattr(
        "mfq.server.cli._studio_web_dir",
        lambda: (_ for _ in ()).throw(AssertionError("should not inspect Studio")),
    )

    assert _prepare_web_root(None, disabled=True) is None
