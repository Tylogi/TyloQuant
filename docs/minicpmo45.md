# MiniCPM-o 4.5 Runtime

MFQ supports the official `OpenBMB/MiniCPM-o-4_5` composite graph in the
Python/CUDA runtime and in the native C++ CUDA and Apple MLX runtimes. The
Python path delegates graph ownership to the official remote-code modules and
replaces supported `Linear` and `Embedding` weights with packed operators. The
native paths implement SigLIP, the Resampler, Whisper, the Qwen3-8B language
backbone, TTS, attention masks, and streaming caches directly.

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

Apple realtime serving uses a separate media extra so the native model worker
does not inherit Python runtime dependencies:

```bash
pip install -e ".[metal,minicpmo45-realtime]"
```

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
Whisper, the audio projector, Qwen3, and the TTS code decoder. The same graph is
available on Apple platforms through `MlxMiniCPMO45Runtime`; the Metal HTTP
server loads the full visual graph for MiniCPM-o models.

MFQd is the shared processor boundary for HTTP image and video input. It ports
the official MiniCPM-o image processor exactly: optional 448-pixel slicing,
bicubic resize, `[0, 1]` conversion, mean/std normalization, 14-pixel patch
packing, contiguous patch masks, and 64-query placeholders. Video is decoded
at one frame per second with no per-frame slicing. MFQd sends the same versioned
`pixel_values`, `patch_mask`, and `target_sizes` package to CUDA and Metal. The
native server applies the model chat template, tokenizes it, derives
`image_bounds` from `<image>` and `<slice>` spans, and then injects Resampler
outputs into the Qwen3 prefill graph. This keeps tokenizer behavior native while
ensuring both device backends receive identical processor tensors.

The CUDA tensor interface reads tensors produced by the official processor
from files sharing an input prefix:

The converter embeds the official NumPy-generated Resampler position table as
a versioned BF16 runtime asset. The C++ graph requires this asset so
cross-attention does not depend on platform-specific sin/cos rounding.

- `<prefix>.input_ids.pt` is required.
- `<prefix>.position_ids.pt` and `<prefix>.attention_mask.pt` are optional.
  Supplying them preserves the official per-batch RoPE positions and padding
  mask; omitted position IDs use consecutive cache positions.
- Image input uses `pixel_values`, `patch_mask`, `target_sizes`, and
  `image_bounds` files. Bounds have rows `[batch, source, begin, end]`.
- Audio input uses `audio_features`, `audio_lengths`, and `audio_bounds` files
  with the same bound layout.
- TTS accepts `tts_inputs_embeds` directly, or derives the condition from the
  LLM hidden states selected by the two-element `tts_bound` tensor.

Each name above uses the `.pt` suffix. Run the graph with:

The suffix is retained for CLI compatibility, but the self-contained CUDA
runtime reads and writes MFQ's `MFQTNSR1` tensor envelope rather than a Python
pickle. Files created by `torch.save` remain available through the optional
`mfq-decode-torch` A/B runtime. Normal image, audio, and video requests through
MFQd do not use this diagnostic file interface.

```bash
mfq-decode \
  --mfq /models/MiniCPM-o-4_5-Q4KM-table.mfq \
  --minicpmo-input-prefix /data/request \
  --minicpmo-output-prefix /data/result \
  --minicpmo-tts-steps 2
```

The output prefix receives component states, merged input embeddings, Qwen3
hidden states and logits, plus TTS code logits and generated codes when TTS is
enabled. Qwen3 and TTS keep the official BF16 projection, SDPA, residual, and
KV-cache boundaries. TTS code generation uses the official default
temperature, top-p, top-k, 16-token repetition penalty, minimum length, and
multinomial sampling. The native graph produces S3 audio codes. Token2wav
waveform rendering still uses the assets supplied with the official model
directory.

## Native full-duplex session

The native CUDA runtime also accepts a sequence of processor-produced units in
one process. The Apple runtime exposes the equivalent state machine through
`prepare_duplex()` and `duplex_step()`. Both preserve the official streaming
Whisper, Qwen3, and TTS caches across units and follow the listen/speak,
chunk-end, and turn-end state transitions used by `MiniCPMODuplex`.

The session prefix contains:

- `<prefix>.special_ids.pt`: 15 token IDs in this order: unit start/end,
  image start/end, slice start/end, listen, speak, TTS BOS/EOS, chunk EOS,
  chunk TTS EOS, turn EOS, TTS pad, and audio BOS.
- `<prefix>.forbidden_ids.pt`: optional tokenizer `bad_token_ids`.
- `<prefix>.system_ids.pt`: optional tokenized system prompt.

Each zero-based unit uses `<prefix>.stepNNNN` and may contain image tensors
(`pixel_values`, `patch_mask`, `target_sizes`, and optional
`image_slice_counts`), `audio_features`, or `text_ids`. Streaming audio features
come from the official processor. Optional `audio_prefix_extra_frames`,
`audio_suffix_extra_frames`, and `force_listen` scalar tensors override their
official defaults. A `reset_session` scalar clears the Whisper, Qwen3, TTS,
sampling, and turn state before processing that unit.

```bash
mfq-decode \
  --mfq models/MiniCPM-o-4_5-Q4KM-table.mfq \
  --minicpmo-duplex-input-prefix fixtures/session \
  --minicpmo-duplex-output-prefix outputs/result \
  --minicpmo-duplex-steps 4
```

Every output unit contains decision logits, generated text token IDs, S3 TTS
codes, and a state tensor. Audio units also contain the pooled Whisper
embeddings. The six state values are listen, end-of-turn, Qwen cache length,
Whisper cache length, TTS cache length, and the official streaming time index.
`--minicpmo-duplex-greedy` and `--minicpmo-duplex-seed` provide deterministic
validation modes.

On Apple, `MlxMiniCPMO45DuplexResult` reports all three cache positions and the
streaming audio index after every unit. TTS only returns codes that have already
advanced the TTS KV cache, so the next unit can verify its expected position
exactly. `reset()` destroys the duplex state and clears all modality caches.

### Realtime service

Start the native CUDA or Apple worker. MiniCPM-o models expose their realtime
duplex capability automatically; it is not a server launch option:

```bash
CUDA_VISIBLE_DEVICES=0 mfq-decode \
  --mfq /models/MiniCPM-o-4_5.mfq \
  --server \
  --host 127.0.0.1 \
  --port 8081
```

On Apple Silicon, use the Metal executable with the same server options:

```bash
mfq-decode-metal \
  --mfq /models/MiniCPM-o-4_5.mfq \
  --server \
  --host 127.0.0.1 \
  --port 8081
```

Start the media gateway with the official `assets/token2wav` directory and
`system_ref_audio.wav` available below the assets root:

```bash
mfq-minicpmo-realtime \
  --backend http://127.0.0.1:8081 \
  --assets /models/OpenBMB/MiniCPM-o-4_5 \
  --host 127.0.0.1 \
  --port 8090
```

Open `http://127.0.0.1:8090/` for the standard MFQ WebUI. When the loaded model
advertises realtime audio capability, a small voice control and a playback
toggle appear in the existing chat composer. Text responses, when present, are
added to the current conversation. Audio responses are always produced and
delivered; the playback toggle only controls whether the browser plays them in
real time. The gateway keeps the normal chat, monitoring, settings, and
model-reload UI intact and proxies their HTTP API requests to the native worker.
Voice sessions use the current official MiniCPM-o Demo defaults: one 16 kHz
audio unit per second, three initial force-listen units, at most 20 generated
text tokens per unit, temperature 0.7, top-k 20, top-p 0.8, repetition penalty
1.05 over 512 tokens, and turn length penalty 1.05. The TTS decoder uses
temperature 0.8 and repetition penalty 1.05; Token2wav uses 10 flow steps and
the browser buffers 200 ms before playback. The WebUI uses the official
language-specific Chinese or English duplex prompt unless the user supplies a
custom system prompt. Generic chat sampling controls do not override these
voice defaults. The gateway selects CUDA for Token2wav when available, followed
by Apple MPS and CPU.

Its additional public media protocol is
`WS /v1/realtime?mode=audio`: clients send base64 float32 mono PCM at 16 kHz and
receive independent text deltas and base64 float32 mono PCM at 24 kHz. The
gateway uses the official exact streaming Mel geometry, including the 1030 ms
first window and two-frame CNN boundary context. It renders S3 codes with the
official Flow and HiFT weights on MPS, with CPU fallback when MPS is unavailable.
The first TTS chunk follows the official early-flush path; later chunks retain
three S3 lookahead codes and consume 25 codes at a time.

The current public media gateway is audio-only. Native callers can still use
the composite graph interfaces for image and mixed-modality requests. Quality
and performance results require a calibrated checkpoint and modality-specific
reference evaluation; structural support alone does not establish those
results.
