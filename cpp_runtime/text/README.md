# MFQ text runtime

This is MFQ's minimal native text-processing runtime. It contains only the
components needed by `mfq-server`:

- GGUF metadata parsing;
- tokenizer and detokenizer implementations;
- grammar-constrained token filtering;
- Jinja chat-template rendering and tool-call parsing;
- the GGML CUDA MMA headers used by MFQ attention kernels.

It does not contain llama.cpp model inference, model architecture
implementations, command-line tools, servers, examples, or optional GGML
backends. The code is compiled directly as the MFQ targets `mfq-gguf` and
`mfq-text-runtime`. The tokenizer/chat public surface is
`include/mfq_text.h`; no upstream compatibility header or API is exposed.

Portions are adapted from `ggml-org/llama.cpp` commit
`25a1d63f4346b472e508c6dbd9ab2ed1d81ace2e` and remain under the MIT license.
See the repository `NOTICE` and `LICENSES/llama.cpp-MIT.txt`.
