# MFQ tokenizer component

This is MFQ's minimal native text-processing runtime. It contains only the
components needed by `mfq-server`:

- GGUF metadata parsing;
- tokenizer and detokenizer implementations;
- grammar-constrained token filtering;
- Jinja chat-template rendering and tool-call parsing;

It does not contain llama.cpp model inference, model architecture
implementations, command-line tools, servers, examples, or optional GGML
backends. GGML/GGUF support lives in the sibling `components/ggml` component.
This code is compiled as `mfq-tokenizer`; the old CMake target
`mfq-text-runtime` remains as a temporary alias. The stable tokenizer/chat C
ABI remains `include/mfq_text.h` and the `mfq_text_*` symbol family.

Portions are adapted from `ggml-org/llama.cpp` commit
`25a1d63f4346b472e508c6dbd9ab2ed1d81ace2e` and remain under the MIT license.
See the repository `NOTICE` and `LICENSES/llama.cpp-MIT.txt`.
