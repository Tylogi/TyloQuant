# Self-contained Windows release

The Windows release packages the `mfq serve` CLI, MFQ Studio, and the native
CUDA `mfq-decode` sidecar in a single x64 NSIS installer. This serve-only build
does not include PyTorch or the training, calibration, quantization, TPQ, and
MiniCPM-o dependency groups. The sidecar is packaged with its CUDA, cuBLAS,
MSVC, and optional OpenMP runtime DLLs. End users do not need a separate Python,
CUDA Toolkit, or native build environment.

Build it from PowerShell 7 or newer on a 64-bit Windows machine:

```powershell
.\release\build_release_windows.ps1
```

The build host needs Visual Studio Build Tools with the C++ x64 workload, a
CUDA Toolkit, `uv`, CMake, Ninja, Rust, Node.js/npm, and NSIS. The script finds
the Visual Studio and CUDA installations, creates an isolated Python 3.12
serve environment, and produces the NSIS installer in `release\dist\`.

By default it targets NVIDIA compute capability 8.6 (Ampere). Override the
target or other release inputs when necessary:

```powershell
.\release\build_release_windows.ps1 -CudaArchitectures 89 -Jobs 2
```

The installer is intended for 64-bit Windows systems with a compatible NVIDIA
GPU and driver. The sidecar CUDA runtime DLLs are copied from the build
machine's CUDA Toolkit; the Toolkit is not required on the end-user machine.
