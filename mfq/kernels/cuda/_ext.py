"""Loader for the unified mfq_cuda extension.

All CUDA ops (norm/acc/rope/attention/gated_delta_net/kv_cache/embedding/sampling/nint_matmul) are compiled into a
single extension, matching llama.cpp's single libggml-cuda (avoids one ~45s compile per op).
Cached under ``torch_extensions`` after the first build. Thin per-op wrappers pull their
function from here.

Requires nvcc + (on Windows) MSVC cl on PATH (source vcvars64 or use a Developer Prompt).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_DIR = os.path.dirname(__file__)
_REPOSITORY_ROOT = str(Path(_DIR).resolve().parents[2])
_SOURCES = [
    os.path.join(_DIR, "norm.cu"),
    os.path.join(_DIR, "acc.cu"),
    os.path.join(_DIR, "rope.cu"),
    os.path.join(_DIR, "attention.cu"),
    os.path.join(_DIR, "gated_delta_net.cu"),
    os.path.join(_DIR, "kv_cache.cu"),
    os.path.join(_DIR, "activation.cu"),
    os.path.join(_DIR, "embedding.cu"),
    os.path.join(_DIR, "sampling.cu"),
    os.path.join(_DIR, "ssm_conv.cu"),
    os.path.join(_DIR, "moe.cu"),
    os.path.join(_DIR, "mx_matmul.cu"),
    os.path.join(_DIR, "nint_matmul.cu"),
    os.path.join(_DIR, "nvq_matmul.cu"),
    os.path.join(_DIR, "nepq.cu"),
    os.path.join(_DIR, "nepq_residual.cu"),
    os.path.join(_DIR, "tpq_matmul.cu"),
    os.path.join(_DIR, "mfq_cuda.cpp"),
]
_module = None


def _ensure_msvc() -> None:
    if os.name != "nt" or shutil.which("cl"):
        return
    roots = tuple(
        Path(value) / "Microsoft Visual Studio"
        for name in ("ProgramFiles", "ProgramFiles(x86)")
        if (value := os.environ.get(name))
    )
    candidates = [
        path
        for root in roots
        if root.exists()
        for path in root.glob("*/*/VC/Auxiliary/Build/vcvars64.bat")
    ]
    if not candidates:
        raise RuntimeError("MSVC vcvars64.bat was not found")
    vcvars = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    command = f'cmd.exe /d /c "call ""{vcvars}"" >nul && set"'
    output = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", command],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            os.environ[key] = value
    if not shutil.which("cl"):
        raise RuntimeError(f"vcvars64 did not expose cl.exe: {vcvars}")


def ext():
    """Trigger the (first) build and return the cached extension module."""

    global _module
    if _module is None:
        _ensure_msvc()
        import torch  # noqa: F401  ensure torch imported (cpp_extension depends on it)
        from torch.utils.cpp_extension import load

        _module = load(
            name="mfq_cuda",
            sources=_SOURCES,
            extra_include_paths=[_REPOSITORY_ROOT],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_ldflags=["cublas.lib"] if os.name == "nt" else ["-lcublas"],
            verbose=False,
        )
    return _module
