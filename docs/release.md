# Self-contained releases

MFQ provides self-contained desktop releases for macOS and Windows. Release build
scripts live in `release/`; run them from that directory. Build artifacts are
written to `release/dist/`.
End users should not need a separate server, Python environment, CUDA Toolkit,
quantizer CLI, inference CLI, llama.cpp checkout, MLX runtime, or shader library.
Native runtime binaries are packaged as private sidecars and are managed by
`mfq serve` or MFQ Studio.

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

The installer is intended for 64-bit Windows systems with a compatible NVIDIA
GPU and driver. CUDA runtime DLLs are copied from the build machine's CUDA
Toolkit; the Toolkit is not required on the end-user machine.

## Runtime behavior

MFQ Studio starts a local `mfq serve` instance without loading a model. Users can
select, load, unload, or replace `.mfq` models from the catalog or native file
picker. Studio registers external model paths in its private application-data
catalog without copying model files; selecting one shard registers the complete
sibling shard family.
