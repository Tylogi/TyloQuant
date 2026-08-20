# Self-contained Windows release

The Windows release packages the public `mfq` CLI and MFQ Studio in a single
x64 NSIS installer. It includes the CUDA `mfq-decode` sidecar and the CUDA,
PyTorch, Python, and MSVC runtime DLLs that it needs. No separate Python,
PyTorch, CUDA runtime, or native build is needed by an end user.

Build it from PowerShell 7 or newer on a 64-bit Windows machine:

```powershell
.\release\build_release_windows.ps1
```

The build host needs Visual Studio Build Tools with the C++ x64 workload, a
CUDA Toolkit, `uv`, CMake, Ninja, Rust, Node.js/npm, and NSIS. The script finds
the Visual Studio and CUDA installations, creates an isolated Python 3.12
release environment, installs the CUDA PyTorch wheel, and produces the NSIS
installer in `release\dist\`.

By default it targets NVIDIA compute capability 8.6 (Ampere). Override the
target or other release inputs when necessary:

```powershell
.\release\build_release_windows.ps1 -CudaArchitectures 89 -Jobs 2
```

The installer is intended for 64-bit Windows systems with a compatible NVIDIA
GPU and driver. CUDA components are bundled from the selected PyTorch wheel;
the CUDA Toolkit is only required on the build machine.
