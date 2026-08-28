# Runtime sampling profiles

MFQ runtime sampling profiles are versioned, partial JSON objects. Missing
fields are not serialized, so runtimes can add defaults without rewriting model
files.

The effective value of each field is resolved independently, from highest to
lowest priority:

1. An explicit API request field.
2. A server profile passed with `--sampling-profile PROFILE.json`.
3. A model sidecar.
4. The `runtime.sampling.v1` metadata embedded in the MFQ header.
5. An exact model registry entry.
6. An architecture registry entry.
7. The runtime's generic default.

An unsharded `model.mfq` accepts `model.runtime.json` and
`model.mfq.runtime.json`.

A sharded model such as `model-00001-of-00008.mfq` accepts the family sidecar
`model.runtime.json` and an optional shard-specific sidecar. The shard-specific
file wins. Embedded global metadata lives in shard zero; both native loaders
find shard zero even when given another shard.

Example:

```json
{
  "schema": "mfq.runtime.sampling",
  "version": 1,
  "chat": {
    "temperature": 0.7,
    "top_k": 100,
    "top_p": 0.8,
    "repetition_penalty": 1.02
  },
  "duplex": {
    "system_prompt": "Streaming Omni Conversation.",
    "decode_mode": "sampling",
    "temperature": 0.7,
    "top_k": 100,
    "top_p": 0.8,
    "force_listen_count": 0
  },
  "tts": {
    "temperature": 0.8,
    "repetition_penalty": 1.05,
    "token2wav_steps": 10
  }
}
```

`mfq quantize` accepts `--sampling-profile`. HF conversions also discover
`generation_config.json` and `mfq-runtime.json` beside the checkpoint. A new
MFQ embeds only fields supplied by those files or by a verified model or
architecture registry entry. Unknown architectures with no supplied profile
receive no sampling metadata.

The server exposes the resolved chat, duplex, and TTS defaults plus their
winning source through `/health` and `/api/status`.
