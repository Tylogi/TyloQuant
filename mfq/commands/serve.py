"""Implementation of the public ``mfq serve`` command."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from mfq.commands.build import BuildError, build_runtime, detect_backend, load_managed_build


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("port must be at most 65535")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _console_script_dir(executable: str | Path) -> Path:
    return Path(executable).parent


def _environment_paths(name: str) -> list[Path]:
    return [Path(value) for value in os.environ.get(name, "").split(os.pathsep) if value]


def _studio_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "MFQStudio"


def _validate_web_root(path: Path, *, source: str) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{source} Web UI directory does not exist: {root}")
    index = root / "index.html"
    if not index.is_file():
        raise FileNotFoundError(f"{source} Web UI entry point does not exist: {index}")
    return root


def _prepare_web_root(configured: Path | None, *, disabled: bool) -> Path | None:
    if disabled:
        return None
    if configured is not None:
        return _validate_web_root(configured, source="--web-root")
    environment_root = os.environ.get("MFQ_SERVER_WEB_ROOT")
    if environment_root:
        return _validate_web_root(
            Path(environment_root),
            source="MFQ_SERVER_WEB_ROOT",
        )
    web_dir = _studio_dir()
    package = web_dir / "package.json"
    output = web_dir / "dist"
    index = output / "index.html"
    if not package.is_file():
        return None

    inputs = [
        *(path for path in web_dir.iterdir() if path.is_file()),
        *(path for path in (web_dir / "src").rglob("*") if path.is_file()),
    ]
    newest_input = max(
        (path.stat().st_mtime_ns for path in inputs if path.is_file()),
        default=0,
    )
    if index.is_file() and index.stat().st_mtime_ns >= newest_input:
        return output

    npm = shutil.which("npm")
    if npm is None:
        print(
            "Web UI source was found but npm is unavailable; serving the API only",
            file=sys.stderr,
        )
        return None
    print("Building MFQ Web UI...", flush=True)
    try:
        subprocess.run([npm, "ci"], cwd=web_dir, check=True)
        subprocess.run([npm, "run", "build"], cwd=web_dir, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"MFQ Web UI build failed with exit code {error.returncode}") from error
    if not index.is_file():
        raise RuntimeError(f"Web UI build completed without producing {index}")
    return output


def _resolve_runtime_executable(
    backend: str,
    configured: Path | None = None,
) -> Path:
    if configured is not None:
        executable = configured.expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(
                f"--running-executable does not exist or is not a file: {executable}"
            )
        print(f"Using configured native runtime: {executable}", flush=True)
        return executable
    managed = load_managed_build(backend)
    if managed is not None and managed.executable.is_file():
        print(f"Using managed native runtime: {managed.executable}", flush=True)
        return managed.executable
    if managed is not None:
        print(
            f"Managed native runtime is missing; rebuilding in {managed.build_dir}",
            flush=True,
        )
        return build_runtime(
            backend=backend,
            build_dir=managed.build_dir,
            build_type=managed.build_type,
            generator=managed.generator,
            cmake_args=managed.cmake_args,
        )
    return build_runtime(backend=backend)


def _select_backend(requested: str, running_executable: Path | None) -> str:
    if running_executable is None:
        return detect_backend(requested)
    system = platform.system()
    machine = platform.machine().lower()
    if requested == "auto":
        if system == "Darwin" and machine in {"arm64", "aarch64"}:
            return "metal"
        if system in {"Linux", "Windows"}:
            return "cuda"
        raise BuildError(f"no prebuilt MFQ runtime backend is supported on {system} {machine}")
    if requested == "metal":
        if system != "Darwin" or machine not in {"arm64", "aarch64"}:
            raise BuildError("the Metal runtime requires Apple silicon and macOS")
        return "metal"
    if requested == "cuda":
        if system not in {"Linux", "Windows"}:
            raise BuildError("the CUDA runtime is supported on Linux and Windows")
        return "cuda"
    raise BuildError(f"unsupported inference backend: {requested}")


def _avfoundation_video_library(executable: Path) -> Path | None:
    from mfq.server.native import find_native_runtime_resource

    configured = os.environ.get("MFQ_AVFOUNDATION_VIDEO_LIBRARY")
    candidate = (
        Path(configured).expanduser().resolve()
        if configured
        else find_native_runtime_resource(executable, "libmfq_avfoundation_video.dylib")
    )
    return candidate if candidate is not None and candidate.is_file() else None


def _run(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("mfq serve requires the 'daemon' optional dependency") from error

    from mfq.server.api import create_app
    from mfq.server.auth import ApiKeyManager
    from mfq.server.backend import OpenAIChatBackend
    from mfq.server.catalog import ModelCatalog
    from mfq.server.cluster import ClusterBackend
    from mfq.server.components import VoiceOutputComponent
    from mfq.server.jobs import JobManager
    from mfq.server.native import NativeRuntime
    from mfq.server.runtime_pool import ManagedRuntimePool
    from mfq.server.service import ServerService
    from mfq.server.storage import SessionStore
    from mfq.server.tool_jobs import ToolJobHandlers, ToolJobPaths

    model = args.model.expanduser().resolve() if args.model is not None else None
    if model is not None and not (model.is_file() or model.is_dir()):
        raise FileNotFoundError(model)
    web_root = _prepare_web_root(args.web_root, disabled=args.no_web_ui)
    selected_backend = _select_backend(args.backend, args.running_executable)
    executable = _resolve_runtime_executable(selected_backend, args.running_executable)
    configured_roots = [path.expanduser().resolve() for path in args.model_dir]
    if not configured_roots:
        configured_roots = _environment_paths("MFQ_SERVER_MODEL_DIRS")
    if not configured_roots:
        configured_roots = [args.work_dir.expanduser().resolve() / "models"]
    configured_roots = [path.expanduser().resolve() for path in configured_roots]
    if model is not None:
        model_catalog_root = model if model.is_dir() else model.parent
        if model_catalog_root not in configured_roots:
            configured_roots.append(model_catalog_root)
    configured_roots[0].mkdir(parents=True, exist_ok=True)
    catalog = ModelCatalog(configured_roots)
    voice_component = VoiceOutputComponent(args.work_dir.expanduser().resolve())
    runtime = None
    try:
        runtime_manager = ManagedRuntimePool(
            catalog,
            executable,
            startup_timeout_seconds=args.runtime_startup_timeout,
            max_instances=args.max_runtime_instances,
            max_requests_per_instance=args.max_requests_per_runtime,
            backend=selected_backend,
            voice_component=voice_component,
        )
        if model is not None:
            initial_artifact = asyncio.run(catalog.resolve_path(model))
            runtime = NativeRuntime(
                executable=executable,
                model=model,
                model_name=initial_artifact.resource.name,
                backend=selected_backend,
                context_size=args.context_size,
                prefill_chunk_size=args.prefill_chunk_size,
                startup_timeout=args.runtime_startup_timeout,
            )
            runtime.start()
            if runtime.process is None or runtime.port is None:
                raise RuntimeError("initial native runtime did not expose its process and port")
            runtime_manager.register_started(
                artifact=initial_artifact,
                process=runtime.process,
                backend=OpenAIChatBackend(
                    runtime.base_url,
                    local_tensor_files=True,
                    avfoundation_video_library=_avfoundation_video_library(executable),
                ),
                port=runtime.port,
                context_size=args.context_size,
            )
        store = SessionStore(args.db.expanduser().resolve())
        backend = ClusterBackend(runtime_manager, store)
        binary_dir = _console_script_dir(sys.executable)
        perplexity = executable.with_name("mfq-perplexity")
        handlers = ToolJobHandlers(
            catalog,
            ToolJobPaths(
                work_root=args.work_dir.expanduser().resolve(),
                python=Path(sys.executable),
                modelscope=(
                    (binary_dir / "modelscope") if (binary_dir / "modelscope").is_file() else None
                ),
                huggingface=(binary_dir / "hf") if (binary_dir / "hf").is_file() else None,
                runtime=executable,
                perplexity=perplexity if perplexity.is_file() else None,
                standalone_cli=bool(getattr(sys, "frozen", False)),
            ),
            voice_component=voice_component,
            activate_voice_output=runtime_manager.enable_realtime,
        )
        jobs = JobManager(store, handlers.handlers())
        service = ServerService(
            store,
            backend,
            jobs=jobs,
            catalog=catalog,
            runtime_manager=runtime_manager,
            tool_handlers=handlers,
            cluster=backend,
            voice_component=voice_component,
        )
        client_api_key = os.environ.get(args.api_key_env, "")
        api_keys = ApiKeyManager(store, client_api_key) if client_api_key else None
        public_url = f"http://{args.host}:{args.port}"
        print(f"MFQ Server ready: {public_url}")
        if web_root is None:
            print("Web UI assets were not found; serving the API only")
        uvicorn.run(
            create_app(service, web_root=web_root, api_keys=api_keys),
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )
    finally:
        if runtime is not None:
            runtime.stop()
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="start the MFQ inference server and optionally load a model",
        description=(
            "Resolve the native runtime for this machine and expose the MFQ API and Web UI. "
            "An initial MFQ model is optional; additional models can be loaded through the API."
        ),
    )
    parser.add_argument("--model", type=Path, help="optional MFQ model to load at startup")
    parser.add_argument(
        "--running-executable",
        type=Path,
        help="prebuilt native runtime executable; skips managed runtime lookup and compilation",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="public API bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=8090,
        help="public API bind port (default: 8090)",
    )
    parser.add_argument("--context-size", type=_nonnegative_int, default=0)
    parser.add_argument("--prefill-chunk-size", type=_positive_int, default=2048)
    parser.add_argument("--runtime-startup-timeout", type=_positive_float, default=1800.0)
    parser.add_argument("--db", type=Path, default=Path("mfq-server.sqlite3"))
    web = parser.add_mutually_exclusive_group()
    web.add_argument("--web-root", type=Path)
    web.add_argument("--no-web-ui", action="store_true")
    parser.add_argument("--model-dir", action="append", type=Path, default=[])
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--api-key-env", default="MFQ_SERVER_API_KEY")
    parser.add_argument("--max-runtime-instances", type=_positive_int, default=2)
    parser.add_argument("--max-requests-per-runtime", type=_positive_int, default=1)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--backend", choices=("auto", "cuda", "metal"), default="auto")
    parser.set_defaults(_impl=_run)
