"""Lifecycle management for the private native inference worker."""

from __future__ import annotations

import functools
import json
import math
import os
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mfq.formats.assets import (
    HF_TOKENIZER_CONFIG_ASSET,
    HF_TOKENIZER_JSON_ASSET,
    MODEL_CONFIG_ASSET,
    TOKENIZER_GGUF_ASSET,
)
from mfq.formats.io import inspect_mfe_header, open_mmap
from mfq.formats.shards import ShardFileVersions, shard_file_versions
from mfq.server.state.model_weights import weight_role
from mfq.server.runtime.hf_tokenizer import (
    ensure_hf_tokenizer_gguf,
    ensure_mfq_tokenizer_gguf,
    native_hf_asset_environment,
)


class NativeRuntimeError(RuntimeError):
    """Raised when the private native worker cannot be started."""


# Plain dense expert banks (for example an unquantized MTP head) are resident
# on the device and run on the dense routed path, outside the RAM expert
# pipeline, so they neither qualify nor disqualify a model for it.
_RESIDENT_DENSE_EXPERT_DTYPES = frozenset({"BF16", "F16", "F32"})


def supports_cuda_ram_expert_pipeline(model: Path) -> bool:
    """Inspect expert headers without decoding or copying their weight payloads."""
    try:
        path = Path(model).resolve()
        # Residency planning and launch configuration ask about the same
        # checkpoint several times per load; probe each shard set version once.
        return _cuda_ram_expert_pipeline_probe(str(path), shard_file_versions(path))
    except (OSError, ValueError):
        # Not cached: unreadable or inconsistent shard sets may be repaired.
        return False


@functools.lru_cache(maxsize=16)
def _cuda_ram_expert_pipeline_probe(model: str, versions: ShardFileVersions) -> bool:
    found = False
    try:
        with open_mmap(model) as store:
            for name, record in store.records.items():
                if weight_role(name) != "experts":
                    continue
                if record.dtype in _RESIDENT_DENSE_EXPERT_DTYPES:
                    continue
                if record.dtype == "NINT":
                    view = store.blob_view(record)
                    try:
                        if len(view) < 50 or struct.unpack_from("<I", view, 14)[0] != 3:
                            return False
                    finally:
                        view.release()
                    found = True
                    continue
                if record.dtype != "MFE":
                    return False
                view = store.blob_view(record)
                try:
                    _, pools = inspect_mfe_header(view)
                    if any(pool.dtype not in {"NINT", "NVQ", "NPQ", "MXFP4"} for pool in pools):
                        return False
                finally:
                    view.release()
                found = True
    except ValueError:
        return False
    return found


def append_native_prefill_chunk_override(
    command: list[str],
    prefill_chunk_size: int | str,
) -> None:
    """Leave Auto to the backend; forward every explicit limit, including 2048."""

    if prefill_chunk_size == "auto":
        return
    if type(prefill_chunk_size) is not int or not 1 <= prefill_chunk_size <= 8192:
        raise NativeRuntimeError("prefill chunk size must be auto or an integer in [1, 8192]")
    command.extend(["--prefill-chunk-size", str(prefill_chunk_size)])


# Capacity is a serving hint: the native worker reports the rows the loaded
# model can actually batch after its state-compatibility checks, and the
# control plane reconciles its admission limit with that report. Asking for a
# bounded capacity up front therefore cannot make an unsupported model fail to
# load; it only tells a batch-capable worker how many rows to prepare for.
DEFAULT_RUNTIME_CONCURRENCY = 8
MAX_RUNTIME_CONCURRENCY = 32
_BATCHING_BACKENDS = frozenset({"cuda", "hip", "metal"})


def native_request_capacity(
    *,
    backend: str,
    routed_expert_bytes: int,
    requested: int,
) -> int:
    """Return the control-plane admission limit requested from a backend worker."""

    if requested < 1:
        raise ValueError("requested runtime concurrency must be positive")
    # Routed expert storage does not determine whether the native scheduler can
    # batch requests.  The native worker owns model/state compatibility and
    # its MoE implementation handles both resident and cached experts.
    _ = routed_expert_bytes
    if backend not in _BATCHING_BACKENDS:
        return 1
    return min(requested, MAX_RUNTIME_CONCURRENCY)


def find_native_runtime_resource(executable: str | Path, name: str) -> Path | None:
    """Find a native resource in build, release, or application layouts."""

    executable_path = Path(executable).expanduser().resolve()
    directory = executable_path.parent
    candidates = (
        directory / name,
        directory / "lib" / name,
        directory.parent / "Resources" / name,
        directory.parent / "Frameworks" / name,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def native_runtime_environment(
    executable: str | Path,
    backend: str,
    base: Mapping[str, str] | None = None,
    model: str | Path | None = None,
) -> dict[str, str]:
    """Return source assets plus any backend-specific runtime resources."""

    environment = dict(os.environ if base is None else base)
    if model is not None:
        environment.update(native_hf_asset_environment(model))
    if backend != "metal":
        return environment
    if not environment.get("MFQ_MLX_METALLIB"):
        metallib = find_native_runtime_resource(executable, "mlx.metallib")
        if metallib is not None:
            environment["MFQ_MLX_METALLIB"] = str(metallib)
    if not environment.get("MFQ_AVFOUNDATION_VIDEO_LIBRARY"):
        video_library = find_native_runtime_resource(
            executable,
            "libmfq_avfoundation_video.dylib",
        )
        if video_library is not None:
            environment["MFQ_AVFOUNDATION_VIDEO_LIBRARY"] = str(video_library)
    return environment


def reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def native_tokenizer_arguments(model: str | Path, backend: str) -> list[str]:
    model_path = Path(model).expanduser().resolve()
    if model_path.is_dir():
        tokenizer = ensure_hf_tokenizer_gguf(model_path)
    elif model_path.is_file():
        with open_mmap(model_path) as store:
            has_embedded_tokenizer = (
                TOKENIZER_GGUF_ASSET in store.records
                or all(
                    name in store.records
                    for name in (
                        MODEL_CONFIG_ASSET,
                        HF_TOKENIZER_JSON_ASSET,
                        HF_TOKENIZER_CONFIG_ASSET,
                    )
                )
            )
        if not has_embedded_tokenizer:
            return []
        tokenizer = ensure_mfq_tokenizer_gguf(model_path)
    else:
        return []
    return ["--tokenizer", str(tokenizer)]


@dataclass
class NativeRuntime:
    executable: Path
    model: Path
    model_name: str
    backend: str
    context_size: int = 0
    prefill_chunk_size: int | str = "auto"
    continuous_batching: int = 0
    routed_expert_bytes: int = 0
    moe_gpu_cache_gb: float | None = None
    startup_timeout: float = 1800.0
    environment: Mapping[str, str] | None = None
    process: subprocess.Popen[bytes] | None = None
    port: int | None = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise NativeRuntimeError("native runtime has not been started")
        return f"http://127.0.0.1:{self.port}"

    def command(self, port: int) -> list[str]:
        if self.continuous_batching < 0:
            raise NativeRuntimeError("continuous batching must be non-negative")
        if self.continuous_batching > 0 and self.backend not in _BATCHING_BACKENDS:
            raise NativeRuntimeError(
                "continuous batching requires a CUDA, HIP or Metal worker"
            )
        if self.moe_gpu_cache_gb is not None and (
            not math.isfinite(self.moe_gpu_cache_gb) or self.moe_gpu_cache_gb < 0
        ):
            raise NativeRuntimeError("MoE expert cache size must be finite and non-negative")
        command = [
            str(self.executable),
            "--model",
            str(self.model),
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-name",
            self.model_name,
        ]
        append_native_prefill_chunk_override(command, self.prefill_chunk_size)
        if self.context_size > 0:
            command.extend(["--ctx-size", str(self.context_size)])
        request_capacity = native_request_capacity(
            backend=self.backend,
            routed_expert_bytes=self.routed_expert_bytes,
            requested=max(1, self.continuous_batching),
        )
        if self.continuous_batching > 0 and request_capacity > 1:
            command.extend(
                ["--continuous-batching", str(request_capacity)]
            )
        if self.moe_gpu_cache_gb is not None:
            command.extend(["--moe-gpu-cache-gb", str(self.moe_gpu_cache_gb)])
        command.extend(native_tokenizer_arguments(self.model, self.backend))
        return command

    def start(self) -> None:
        if self.process is not None:
            raise NativeRuntimeError("native runtime is already running")
        if not self.executable.is_file():
            raise NativeRuntimeError(f"native runtime executable does not exist: {self.executable}")
        if not (self.model.is_file() or self.model.is_dir()):
            raise NativeRuntimeError(f"model does not exist: {self.model}")
        self.port = reserve_loopback_port()
        command = self.command(self.port)
        print(
            "Starting native inference worker:",
            subprocess.list2cmdline(command),
            flush=True,
        )
        self.process = subprocess.Popen(
            command,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            stdin=subprocess.DEVNULL,
            env=native_runtime_environment(
                self.executable,
                self.backend,
                base={**os.environ, **dict(self.environment or {})},
                model=self.model,
            ),
        )
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        if self.process is None:
            raise NativeRuntimeError("native runtime has not been started")
        deadline = time.monotonic() + self.startup_timeout
        last_error = "worker did not accept connections"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            status = self.process.poll()
            if status is not None:
                self.process = None
                raise NativeRuntimeError(
                    f"native inference worker exited during startup with status {status}"
                )
            try:
                with opener.open(f"{self.base_url}/health", timeout=1.0) as response:
                    payload = json.loads(response.read())
                if isinstance(payload, dict) and payload.get("model"):
                    return
                last_error = "worker returned an invalid health response"
            except (OSError, ValueError, urllib.error.URLError) as error:
                last_error = str(error)
            time.sleep(0.25)
        self.stop()
        raise NativeRuntimeError(
            f"native inference worker did not become ready within {self.startup_timeout:g}s: "
            f"{last_error}"
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def __enter__(self) -> NativeRuntime:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
