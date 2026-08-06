"""Loader for the offline quantization CUDA extension."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


_DIR = Path(__file__).resolve().parent
_SOURCES = [
    str(_DIR / "nvq_quant_assign.cu"),
    str(_DIR / "nepq0_s_assign.cu"),
    str(_DIR / "npq0_s_assign.cu"),
    str(_DIR / "npq0_l_assign.cu"),
    str(_DIR / "nvq2j_assign.cu"),
    str(_DIR / "nvq3j_assign.cu"),
    str(_DIR / "nint_quant.cu"),
    str(_DIR / "nvq_quant_cuda.cpp"),
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
    global _module
    if _module is None:
        _ensure_msvc()
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load

        _module = load(
            name="mfq_nvq_quant_cuda",
            sources=_SOURCES,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _module
