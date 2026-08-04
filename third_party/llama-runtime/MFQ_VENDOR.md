# Bundled llama.cpp runtime subset

This directory vendors the tokenizer, GGUF metadata, chat-template, GGML CPU,
and supporting sources from `ggml-org/llama.cpp` commit
`25a1d63f4346b472e508c6dbd9ab2ed1d81ace2e`.

MFQ builds these sources statically. No external llama.cpp checkout, build
directory, shared library, or runtime installation is required. The upstream
license is preserved in `LICENSE`.
