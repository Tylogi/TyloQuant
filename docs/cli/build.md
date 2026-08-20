# `mfq build`

`mfq build` compiles the native inference runtime for the current machine. It
detects Metal or CUDA, invokes CMake, and records the resulting executable so
that [`mfq serve`](serve.md) can find it without a binary path.

The `mfq` command must come from an MFQ source checkout containing
`cpp_runtime/CMakeLists.txt`. MFQ derives the repository root from the installed
Python package, not from the current working directory. In an editable
development environment, run the CLI through uv from that checkout.

## Quick start

Sync the dependency set for the current host once:

```shell
cd /path/to/MFQ
uv sync --extra metal
```

On a CUDA host, the native build itself needs no Python accelerator package.
Install the daemon extra only when serving through the HTTP API:

```shell
uv sync --extra daemon
```

Then let MFQ detect the backend and build it:

```shell
uv run mfq build
```

Automatic backend selection uses:

- Metal on Apple silicon macOS;
- CUDA on Linux or Windows when `nvcc` is available.

Detection selects a backend; it does not prove that every build dependency is
installed. The Metal and CUDA requirements are listed below.

The command stops with an explanation if neither backend is available. Use an
explicit backend when diagnosing detection or build problems:

```shell
uv run mfq build --backend metal
uv run mfq build --backend cuda
```

## Output and build manifest

The default CMake build directory is `<repo>/build/cpp_runtime`. The main
executable is normally:

| Backend | Executable |
| --- | --- |
| Metal | `<repo>/build/cpp_runtime/metal/mfq-decode-metal` |
| CUDA | `<repo>/build/cpp_runtime/mfq-decode` |
| CUDA on Windows | Usually `<repo>\build\cpp_runtime\mfq-decode.exe`; multi-configuration generators may add `Release\` |

CMake generators may place the executable in a configuration subdirectory.
MFQ searches the build tree after compilation and records the actual path,
rather than assuming one fixed layout.

A successful build writes `<repo>/build/mfq-runtime.json`. This location is
fixed relative to the source checkout even when `--build-dir` points elsewhere.
The manifest stores the backend, source directory, build directory, build type,
generator, forwarded CMake arguments, target, and the actual absolute
executable path. It is internal CLI state under the ignored `build/` directory;
users do not add the native executable to `PATH`.

To inspect a generator-dependent path, read the `executable` field for the
selected backend in the manifest:

```shell
uv run python -m json.tool build/mfq-runtime.json
```

`mfq serve` reads this manifest automatically. If the executable exists, it is
used directly. If it was deleted, MFQ rebuilds it in the recorded directory with
the recorded configuration.

## Custom build directory

Use `--build-dir` when the build tree belongs on another disk:

```shell
cd /path/to/MFQ
uv run mfq build --build-dir /mnt/fast-build/mfq
```

The manifest records the absolute build directory and executable path. A later
`mfq serve` command finds this custom build without another option:

```shell
uv run mfq serve --model /models/model.mfq
```

Both commands must use the `mfq` installation associated with the same source
checkout. The current working directory may change between them.

## Forward CMake configuration arguments

Arguments after `--` are passed to the CMake configure command. The separator
is required so the arguments are not parsed as `mfq build` options.

```shell
uv run mfq build --backend cuda -- \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DGGML_CCACHE=OFF
```

These arguments are saved in the build manifest and reused if MFQ must recreate
a missing executable.

## Command options

| Option | Meaning | Default |
| --- | --- | --- |
| `--backend {auto,cuda,metal}` | Select or detect the accelerator backend. | `auto` |
| `--build-dir PATH` | Set the CMake build directory. | `<repo>/build/cpp_runtime` |
| `--build-type TYPE` | Set `CMAKE_BUILD_TYPE` and the build configuration. | `Release` |
| `-j N`, `--jobs N` | Set parallel build jobs. | Host CPU count |
| `--generator NAME` | Select a CMake generator. | Ninja when available, otherwise CMake default |
| `--dry-run` | Print configure and build commands without executing them. | Off |
| `-- CMAKE_ARG...` | Forward remaining arguments to CMake configuration. | None |

`--dry-run` does not compile an executable or update the manifest.

## Backend requirements

### Metal

- Apple silicon macOS;
- CMake 3.26 or newer;
- a C++ toolchain;
- the `metal` optional dependency, which provides MLX and its native assets.

For a source checkout managed by uv:

```shell
uv sync --extra metal
uv run mfq build
```

### CUDA

- Linux or Windows;
- CMake 3.26 or newer;
- a C++/CUDA toolchain with `nvcc`;
- CUDA Toolkit 12 or newer, including `nvcc`, cuBLAS, and the CUDA runtime.

The default CUDA inference runtime is self-contained and does not require
Python, PyTorch, or LibTorch. Developers doing migration A/B validation may
explicitly configure `-DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=ON`; that optional
reference target has its own Python and LibTorch requirements and is not part
of the normal runtime build.

MFQ checks `CUDACXX`, `PATH`, `CUDA_HOME`, and `CUDA_PATH` when locating
`nvcc`.

## Troubleshooting

### No supported accelerator was detected

Confirm the host is Apple silicon, or make `nvcc` visible through one of the
CUDA locations above. `--backend` validates the requested backend; it does not
bypass platform and toolchain checks.

### CMake completed without producing the executable

Read the printed configure and build commands first. If a custom generator uses
an unexpected output layout, MFQ searches the build tree by executable name. A
remaining error usually means the requested target was not produced.

### Inspect the exact commands

```shell
uv run mfq build --dry-run --backend cuda -- \
  -DCMAKE_CUDA_ARCHITECTURES=90
```

After a successful build, continue with [`mfq serve`](serve.md).
