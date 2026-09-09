# MFQ C++ runtime

The native runtime is organized by responsibility rather than checkpoint
family or source origin:

- `core/` — canonical model graph, backend-neutral policies, interfaces, and
  generated tables; temporary old-artifact name adapters live in
  `core/compat/`;
- `server/` — the backend-neutral HTTP/tokenizer server API and implementation;
- `components/` — focused integrated components (`ggml`, `tokenizer`, `http`,
  and `json`);
- `backends/cuda/` — CUDA headers, implementation, model adapters,
  applications, and backend tests;
- `backends/metal/` — Metal/MLX storage, runtime utilities, operators, model
  implementations, kernels, applications, tests, benchmarks, and diagnostics;
- `tests/` — backend-independent native tests;
- `cmake/` — backend-specific build orchestration.

The mandatory ownership and canonicalization rules are defined in the
repository [development rules](../CONTRIBUTING.md). In particular, reusable
state machines, cache lifecycle, sampling, dispatch, metrics, multimodal
pipelines, and MTP orchestration must never live in a model-architecture
directory.

`CMakeLists.txt` is the single entry point. Existing executable target names
(`mfq-decode`, `mfq-decode-metal`, and `mfq-perplexity`) and the established
Metal build output directory remain unchanged. Integrated upstream-derived
code retains its original licensing as documented in the repository `NOTICE`
and `LICENSES/` directory.
