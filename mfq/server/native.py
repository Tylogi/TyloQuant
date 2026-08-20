"""Lifecycle management for the private native inference worker."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class NativeRuntimeError(RuntimeError):
    """Raised when the private native worker cannot be started."""


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
) -> dict[str, str]:
    """Return a worker environment with relocatable Metal resources resolved."""

    environment = dict(os.environ if base is None else base)
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


@dataclass
class NativeRuntime:
    executable: Path
    model: Path
    model_name: str
    backend: str
    context_size: int = 0
    prefill_chunk_size: int = 2048
    startup_timeout: float = 1800.0
    process: subprocess.Popen[bytes] | None = None
    port: int | None = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise NativeRuntimeError("native runtime has not been started")
        return f"http://127.0.0.1:{self.port}"

    def command(self, port: int) -> list[str]:
        command = [
            str(self.executable),
            "--mfq",
            str(self.model),
            "--server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-name",
            self.model_name,
        ]
        if self.backend == "metal":
            command.extend(["--prefill-chunk-size", str(self.prefill_chunk_size)])
        if self.context_size > 0:
            command.extend(["--ctx-size", str(self.context_size)])
        return command

    def start(self) -> None:
        if self.process is not None:
            raise NativeRuntimeError("native runtime is already running")
        if not self.executable.is_file():
            raise NativeRuntimeError(f"native runtime executable does not exist: {self.executable}")
        if not self.model.is_file():
            raise NativeRuntimeError(f"MFQ model does not exist: {self.model}")
        self.port = reserve_loopback_port()
        command = self.command(self.port)
        print(
            "Starting native inference worker:",
            subprocess.list2cmdline(command),
            flush=True,
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            env=native_runtime_environment(self.executable, self.backend),
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
