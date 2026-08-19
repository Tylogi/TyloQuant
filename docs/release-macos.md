# Self-contained macOS release

The macOS release packages one public `mfq` CLI and MFQ Studio. Users do not
install a separate server, quantizer, inference CLI, llama.cpp checkout, MLX
runtime, or Metal shader library. Native worker and evaluation executables are
private application sidecars.

Build an Apple Silicon DMG from a clean checkout:

```shell
./release/build_release_mac.sh
```

The build uses the canonical `cpp_runtime` source tree. It does not copy runtime
sources into the release directory. The resulting application contains:

- the unified `mfq` CLI, including `serve`, `quantize`, `calibrate`, and related
  subcommands;
- `mfq-decode-metal` and `mfq-perplexity` as private sidecars;
- `libmlx`, `libjaccl`, `mlx.metallib`, and the AVFoundation video bridge;
- MFQ Studio and its production Web UI.

MFQ Studio starts its local server without loading a model. A model can then be
selected from the native file picker, loaded, unloaded, or replaced from the
model catalog. Browser deployments use the same catalog and model-management
API. The desktop application stores registered external file paths only in its
private application-data directory, not in the repository.

The script creates an isolated release environment and validates the native
runtime, CLI subcommands, Rust desktop application, Web UI, architectures, and
code signatures before copying the DMG to `release/dist/`.
