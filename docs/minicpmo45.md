# MiniCPM-o 4.5 Runtime

MFQ supports the official `OpenBMB/MiniCPM-o-4_5` composite graph in the
Python/CUDA runtime and in the native C++ CUDA and Apple MLX runtimes. The
Python path delegates graph ownership to the official remote-code modules and
replaces supported `Linear` and `Embedding` weights with packed operators. The
native paths implement SigLIP, the Resampler, Whisper, the Qwen3-8B language
backbone, TTS, attention masks, and streaming caches directly.

The runtime requires the official model directory for its tokenizer, processor,
remote-code definitions, and Token2wav assets. MFQ does not copy those assets
into the weight file.

## Environment

Install the model-specific extra:

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
uv run -m mfq.tools.quantize_hf_to_mfq \
  --input /models/OpenBMB/MiniCPM-o-4_5 \
  --output /models/MiniCPM-o-4_5-NINT4.mfq \
  --bits 4 \
  --groupsize 24 \
  --sub-bits 6
```

The MiniCPM-o 4.5 policy keeps all six checkpoint roots: `llm`, `vpm`, `apm`,
`resampler`, `audio_projection_layer`, and `tts`. In the official checkpoint,
710 module matrices are quantized and 704 tensors retain their source BF16
representation. These matrices stay dense because the official graph accesses
them directly:

- Whisper and SigLIP position embeddings
- Resampler query, output projection, and packed MultiheadAttention projections
- TTS weight-normalized output-head parameters

The converter rejects unsupported compact raw parameters during runtime load.

Pass a language-model GGUF as a recipe to reuse its tensor precisions while
keeping the full MiniCPM-o graph:

```bash
uv run -m mfq.tools.quantize_hf_to_mfq \
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

MFQd applies the MiniCPM-o image pipeline to HTTP input:

- optional 448-pixel slicing;
- bicubic resize and `[0, 1]` conversion;
- mean/std normalization;
- 14-pixel patch packing and contiguous patch masks;
- 64-query placeholders.

Video is decoded at one frame per second without per-frame slicing. CUDA and
Metal receive the same versioned `pixel_values`, `patch_mask`, and
`target_sizes`. The native server applies the chat template, tokenizes it,
derives `image_bounds` from `<image>` and `<slice>` spans, and injects Resampler
outputs into Qwen3 prefill.

The converter embeds the NumPy-generated Resampler position table as a
versioned BF16 runtime asset, avoiding platform-dependent sin/cos rounding in
cross-attention.

The CUDA diagnostic interface reads files with a shared prefix:

- `<prefix>.input_ids.pt` is required.
- `<prefix>.position_ids.pt` and `<prefix>.attention_mask.pt` are optional.
  Supplying them preserves per-batch RoPE positions and the padding mask;
  omitted position IDs use consecutive cache positions.
- Image input uses `pixel_values`, `patch_mask`, `target_sizes`, and
  `image_bounds` files. Bounds have rows `[batch, source, begin, end]`.
- Audio input uses `audio_features`, `audio_lengths`, and `audio_bounds` files
  with the same bound layout.
- TTS accepts `tts_inputs_embeds` directly, or derives the condition from LLM
  hidden states selected by the two-element `tts_bound` tensor.

The `.pt` suffix is kept for CLI compatibility. The self-contained CUDA runtime
uses MFQ's `MFQTNSR1` envelope, not Python pickle. Files written by `torch.save`
work only with the optional `mfq-decode-torch` A/B runtime. Normal MFQd image,
audio, and video requests do not use this diagnostic interface.

```bash
mfq-decode \
  --mfq /models/MiniCPM-o-4_5-Q4KM-table.mfq \
  --minicpmo-input-prefix /data/request \
  --minicpmo-output-prefix /data/result \
  --minicpmo-tts-steps 2
```

The output prefix contains component states, merged input embeddings, Qwen3
hidden states and logits, and—when enabled—TTS logits and generated codes.
Qwen3 and TTS keep the BF16 projection, SDPA, residual, and KV-cache boundaries.
TTS uses the model's default temperature, top-p, top-k, 16-token repetition
penalty, minimum length, and multinomial sampling. The graph produces S3 audio
codes; Token2wav renders them with assets from the official model directory.

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

Open `http://127.0.0.1:8090/`. Realtime-capable models add voice and playback
controls to the chat composer. Text is added to the conversation. Audio is
always generated; the playback toggle only controls browser playback. Other UI
pages proxy their API requests to the native worker.

Voice defaults match the MiniCPM-o demo:

- one 16 kHz audio unit per second;
- three initial force-listen units;
- at most 20 generated text tokens per unit;
- temperature 0.7, top-k 20, and top-p 0.8;
- repetition penalty 1.05 over 512 tokens;
- turn length penalty 1.05.

TTS uses temperature 0.8 and repetition penalty 1.05. Token2wav uses 10 flow
steps, and the browser buffers 200 ms before playback. The Web UI selects the
Chinese or English duplex prompt unless the user sets a system prompt. Generic
chat sampling options do not change these voice defaults. Token2wav prefers
CUDA, then Apple MPS, then CPU.

The media protocol is `WS /v1/realtime?mode=audio`. Clients send base64 float32
mono PCM at 16 kHz and receive separate text deltas and base64 float32 mono PCM
at 24 kHz.

Streaming Mel uses a 1030 ms first window and two frames of CNN boundary
context. Flow and HiFT render S3 codes on MPS, with CPU fallback. The first TTS
chunk uses early flush; later chunks keep three S3 lookahead codes and consume
25 codes at a time.

The current public media gateway is audio-only. Native callers can still use
the composite graph interfaces for image and mixed-modality requests. Quality
and performance results require a calibrated checkpoint and modality-specific
reference evaluation; structural support alone does not establish those
results.
