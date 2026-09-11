# MFQ documentation

MFQ combines high-quality, fine-grained mixed-precision quantization, optimized
CUDA/Metal C++ inference, and local serving. Start with the workflow that
matches what you want to do.

## Get started

- [Build the C++ runtime](cli/build.md)
- [Run MFQ Server and Studio](cli/serve.md)
- [Create a self-contained release](release.md)
- [Check model and backend support](runtime-support.md)

## Quantize and calibrate

- [Quantize HF, GGUF, or full-precision MFQ](cli/quantize.md)
- [Collect reusable calibration artifacts](cli/calibrate.md)
- [Solve an Expert-Wise precision budget](ew-joint-solver.md)
- [Embed runtime sampling profiles](runtime-sampling-profiles.md)

## Models and runtimes

- [Runtime support matrix](runtime-support.md)
- [MiniCPM-o 4.5 multimodal runtime](minicpmo45.md)
- [CUDA runtime validation](cuda-native-runtime-validation.md)
- [C++ runtime layout](../cpp_runtime/README.md)

## APIs

- [HTTP API](api/http.md)
- [WebSocket and realtime API](api/websocket.md)

## Develop MFQ

- [Architecture and contribution rules](../CONTRIBUTING.md)
- [Repository README](../README.md)

Runtime support is checkpoint- and backend-dependent. Use the support matrix
as the starting point, then validate the exact model artifact on its target
machine before deployment.
