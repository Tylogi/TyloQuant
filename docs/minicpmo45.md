# MiniCPM-o 4.5 Runtime

MFQ supports the official `OpenBMB/MiniCPM-o-4_5` composite graph in the
Python/CUDA runtime. The official remote-code modules continue to own image and
audio preprocessing, SigLIP, the Resampler, Whisper, Qwen3, TTS, attention
masks, and streaming caches. MFQ replaces supported `Linear` and `Embedding`
weights with packed CUDA operators.

This path requires the official model directory at runtime. The directory
provides the tokenizer, processor, remote-code definitions, and Token2wav
assets. These assets are not copied into the MFQ weight file.

## Environment

Use the official dependency versions through the optional extra:

```bash
pip install -e ".[minicpmo45]"
```

The extra pins `transformers==4.51.0` and the official Torch/Torchaudio 2.3–2.8
range. TTS and streaming install `minicpmo-utils[all]>=1.0.5`.

## Convert

```bash
python -m mfq.tools.quantize_hf_to_mfq \
  --input /models/OpenBMB/MiniCPM-o-4_5 \
  --output /models/MiniCPM-o-4_5-NINT4.mfq \
  --bits 4 \
  --groupsize 24 \
  --sub-bits 6
```

The MiniCPM-o 4.5 policy keeps all six checkpoint roots: `llm`, `vpm`, `apm`,
`resampler`, `audio_projection_layer`, and `tts`. In the official checkpoint,
710 module matrices are quantized and 704 tensors retain their source BF16
representation. The following raw matrices stay dense because the official
graph accesses them directly:

- Whisper and SigLIP position embeddings
- Resampler query, output projection, and packed MultiheadAttention projections
- TTS weight-normalized output-head parameters

The converter rejects unsupported compact raw parameters during runtime load.

To reproduce the tensor precision allocation from an official language-model
GGUF while keeping the complete MiniCPM-o graph, pass it as a recipe:

```bash
python -m mfq.tools.quantize_hf_to_mfq \
  --input /models/OpenBMB/MiniCPM-o-4_5 \
  --output /models/MiniCPM-o-4_5-Q4KM-table.mfq \
  --recipe-gguf /models/MiniCPM-o-4_5-Q4_K_M.gguf
```

The official Q4_K_M GGUF contains only the Qwen3 language model. Every
`llm.*` tensor must map to the recipe exactly; a missing mapping stops the
conversion. Vision, audio, Resampler, audio-projector, and TTS tensors remain
at their source precision instead of being omitted from the MFQ file.

## Load

```python
from mfq.runtime import load_minicpmo45

runtime = load_minicpmo45(
    "/models/OpenBMB/MiniCPM-o-4_5",
    "/models/MiniCPM-o-4_5-NINT4.mfq",
    device="cuda:0",
)

model = runtime.model
print(runtime.load_report)
```

`runtime` delegates the official model methods, including `forward`, `chat`,
streaming, duplex, and TTS entry points. Call them with the same processor data
and arguments used by the official repository.

## Native C++ composite graph

`mfq-decode` has a CUDA-native tensor interface for SigLIP, the Resampler,
Whisper, the audio projector, Qwen3, and the TTS code decoder. It reads tensors
produced by the official processor from files sharing an input prefix:

The converter embeds the official NumPy-generated Resampler position table as
a versioned BF16 runtime asset. The C++ graph requires this asset so
cross-attention does not depend on platform-specific sin/cos rounding.

- `<prefix>.input_ids.pt` is required.
- Image input uses `pixel_values`, `patch_mask`, `target_sizes`, and
  `image_bounds` files. Bounds have rows `[batch, source, begin, end]`.
- Audio input uses `audio_features`, `audio_lengths`, and `audio_bounds` files
  with the same bound layout.
- TTS accepts `tts_inputs_embeds` directly, or derives the condition from the
  LLM hidden states selected by the two-element `tts_bound` tensor.

Each name above uses the `.pt` suffix. Run the graph with:

```bash
mfq-decode \
  --mfq /models/MiniCPM-o-4_5-Q4KM-table.mfq \
  --minicpmo-input-prefix /data/request \
  --minicpmo-output-prefix /data/result \
  --minicpmo-tts-steps 2
```

The output prefix receives component states, merged input embeddings, Qwen3
hidden states and logits, plus TTS code logits and generated codes when TTS is
enabled. The native graph produces S3 audio codes. Token2wav waveform rendering
still uses the assets supplied with the official model directory.

The HTTP server does not yet parse raw MiniCPM-o image and audio requests.
Quality and performance results require a calibrated checkpoint and
modality-specific reference evaluation; structural support alone does not
establish those results.
