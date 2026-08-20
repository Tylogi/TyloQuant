# Native CUDA runtime validation

MFQ's production CUDA inference executable is `mfq-decode`. It owns tensor
storage, CUDA streams and events, CUDA Graphs, and generic graph operations
without Python, PyTorch, ATen, or LibTorch. The packed MFQ CUDA kernels are the
same kernels used by the optional reference executable.

`mfq-decode-torch` is an opt-in migration target. It exists only for A/B
validation and is excluded from normal builds and packages.

## Build and dependency checks

Configure the production runtime on both Linux and Windows:

```shell
cmake -S cpp_runtime -B build/cuda-native \
  -DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=OFF \
  -DBUILD_TESTING=ON
cmake --build build/cuda-native --config Release -j
ctest --test-dir build/cuda-native -C Release --output-on-failure
```

The resulting executable must not import or link Python, Torch, ATen, c10, or
LibTorch. Check the final dependency table with `ldd` on Linux and
`dumpbin /DEPENDENTS` on Windows. CUDA runtime, CUDA driver, cuBLAS, the host C++
runtime, and optional NCCL are expected.

For an A/B build, configure a separate tree with LibTorch discoverable and
`-DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=ON`. This adds `mfq-decode-torch`; it does
not alter `mfq-decode`.

## Required A/B matrix

Run every row with fixed model files, prompts, seeds, cache limits, and sampling
parameters. Start each executable in a fresh process.

| Area | Required coverage |
| --- | --- |
| Architectures | dense causal LM, GQA, multimodal MiniCPM-o, and routed MoE |
| Formats | dense BF16/F16, NINT, NVQ/NPQ/NEPQ, TPQ, MXFP8, and MXFP4 where the architecture permits them |
| Shapes | single-token decode, short and long prefill, odd sizes, batched inputs, and GQA head broadcasting |
| State | empty cache, reused prefix cache, context rollover, session reset, and interrupted generation |
| Sampling | greedy, temperature, top-k, top-p, min-p, repetition/presence penalties, and fixed random seed |
| Execution | eager, CUDA Graph capture/replay, one and multiple CUDA streams, one GPU, and tensor parallel when NCCL is present |
| Media | text, image, video, audio, half duplex, and full duplex |

Capture raw logits from both executables at prefill and at every decode step.
Custom packed kernels should retain their existing numerical contract because
their device bodies are unchanged. Generic tensor reductions and cuBLAS calls
may use a different reduction tree than ATen, so acceptance is based on the
model's established logit tolerance plus identical greedy tokens, not an
unsupported claim of universal bit identity. Any token divergence must be
explained before the candidate branches are merged.

Also compare peak device memory, host memory, model-load time, prefill speed,
and decode speed. The native runtime is rejected if it introduces a material
regression or silently falls back to host computation.

## Packaging gate

Create clean Windows and Linux packages from the native build. Install each in
an environment that has no `torch` package and no LibTorch files, then verify:

1. empty-server startup and model discovery;
2. loading and unloading representative dense and MoE models;
3. OpenAI-compatible streaming and non-streaming generation;
4. multimodal processing through MFQd;
5. process shutdown without leaked workers; and
6. package size and dependency inventory.

The legacy MiniCPM-o diagnostic `.pt` filenames use MFQ's `MFQTNSR1` tensor
envelope in the native executable. Python pickle files created by `torch.save`
are intentionally handled only by `mfq-decode-torch`; production HTTP media
requests do not cross this file boundary.
