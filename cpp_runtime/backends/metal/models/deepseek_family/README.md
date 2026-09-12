# DeepSeek raw-HF family core

This directory is the shared, format-preserving execution core for the
original DeepSeek V4 Flash and V4.1 Hugging Face checkpoints. It is not the
owner of either architecture's public entry point.

Variant checks are allowed here only when they select a checkpoint-defined
tensor shape or graph schedule inside the shared loader. Architecture policy,
device tuning switches, Engram storage, and public dispatch belong in
`deepseek_v4/` or `deepseek_v41/`; reusable Metal primitives belong in
`backends/metal/ops/`.

Keeping one execution core is intentional: duplicating the roughly identical
raw-HF graph would let correctness and the measured M3 Ultra hot path drift.
The V4 compatibility headers and V4.1 raw-HF factory both enter this core
through architecture-owned façades.
