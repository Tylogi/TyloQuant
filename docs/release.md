# Self-contained releases

MFQ desktop releases support macOS and Windows. Build scripts live in
`release/` and write artifacts to `release/dist/`.

The packages include the server, Python environment, native runtime, and
required runtime libraries. `mfq serve` or MFQ Studio manages the native
sidecars.

## macOS

The macOS release packages the public `mfq` CLI and MFQ Studio into an Apple
Silicon DMG. It includes:

- the unified `mfq` CLI with `serve`, `quantize`, `calibrate`, and related
  subcommands;
- private native sidecars such as `mfq-decode-metal` and `mfq-perplexity`;
- `libmlx`, `libjaccl`, `mlx.metallib`, and the AVFoundation video bridge;
- MFQ Studio and its production Web UI.

Build from a clean checkout on Apple Silicon:

```shell
cd release
./build_release_mac.sh
```

The script builds from the canonical `cpp_runtime` source tree, validates the
native runtime, CLI subcommands, Rust desktop app, Web UI, architectures, and
code signatures, then copies the DMG to `release/dist/`.

## Windows

The Windows release packages `mfq serve`, MFQ Studio, and the native CUDA
`mfq-decode` sidecar into a single x64 NSIS installer. This serve-only package
excludes PyTorch and the training, calibration, quantization, TPQ, and MiniCPM-o
dependency groups. The CUDA, cuBLAS, MSVC, and optional OpenMP runtime DLLs are
bundled with the sidecar, so end users do not need Python, the CUDA Toolkit, or a
native build environment.

Build from PowerShell 7 or newer on 64-bit Windows:

```powershell
cd release
.\build_release_windows.ps1
```

The build host needs Visual Studio Build Tools with the C++ x64 workload, a CUDA
Toolkit, `uv`, CMake, Ninja, Rust, Node.js/npm, and NSIS. By default the script
targets NVIDIA compute capability 8.6. Override release inputs when needed:

```powershell
.\build_release_windows.ps1 -CudaArchitectures 89 -Jobs 2
```

The installer supports 64-bit Windows with a compatible NVIDIA GPU and driver.
It bundles CUDA runtime DLLs from the build machine, so users do not need the
CUDA Toolkit.

## Runtime behavior

MFQ Studio starts `mfq serve` locally without a model. The catalog and native
file picker can load, unload, or replace `.mfq` models. Studio records external
paths without copying files; selecting one shard registers the full sibling
shard family.
