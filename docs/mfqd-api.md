# MFQd native protocol

MFQd is the long-running service boundary for Web clients, MFQ Studio, SDKs,
and native MFQ runtime workers. Protocol version `1.0` defines the data model
used across these boundaries.

## Public client API

The checked-in OpenAPI document is `mfqd/protocol/openapi.json`. It defines:

- persistent session creation, response generation, forking, and deletion;
- content-addressed media upload;
- model load and unload operations;
- runtime instance inspection;
- the `/api/v1/realtime` WebSocket path and its event names.

Messages contain typed parts: text, reasoning, image, audio, transcript, tool
call, tool result, or generated audio. Image and audio parts refer to an
uploaded media object by identifier and exact SHA-256 digest. A response
request carries an idempotent request identifier and the expected session
revision so concurrent updates cannot silently overwrite one another.

Realtime frames carry a monotonically increasing sequence number. Full-duplex
PCM chunks include their own audio sequence and timestamp. The initial JSON
transport uses base64-encoded mono or multichannel `pcm_s16le` data.

## Runtime IPC

The versioned Protobuf source is `mfqd/protocol/runtime.proto`. MFQd exchanges
framed `RuntimeEnvelope` messages with a native worker over a Unix socket or a
Windows named pipe. Large tensors and audio blocks may use `SharedMemoryRef`;
the descriptor includes the byte range, scalar type, shape, strides, and an
exact SHA-256 digest.

`RuntimeIdentity` binds cache state to model, quantization, runtime build,
tokenizer, chat template, processor, RoPE parameters, and KV dtype. A worker
must reject state created under a different identity.

## Updating the contract

After changing the Python protocol models or routes, regenerate and verify the
OpenAPI artifact:

```bash
python -m mfqd.openapi mfqd/protocol/openapi.json
python -m mfqd.openapi --check mfqd/protocol/openapi.json
```

Breaking changes require a new protocol version and a new Protobuf package.

## Conversation storage

`mfqd.storage.SessionStore` provides the initial SQLite WAL conversation layer.
Messages are immutable records. Sessions reference an ordered message chain, so
forks share their existing prefix and append independent messages afterward.
Every append requires the current session revision; stale concurrent writes are
rejected before a message is inserted. SQLite transactions end before runtime
inference begins.

## Text gateway

The executable service keeps the existing C++ OpenAI-compatible endpoint on
the generation path and adds persistent sessions around it. Start the native
MFQ server, then run:

```bash
mfqd --backend-url http://127.0.0.1:8080 --db mfqd.sqlite3
```

MFQd listens on `127.0.0.1:8090` by default. The bearer token for a protected
backend is read from `MFQD_BACKEND_API_KEY`; it is not accepted as a command
line value. Text responses can be returned as JSON or streamed as typed SSE
frames. Reasoning deltas, tool calls, finish reasons, token usage, failures,
and session state transitions are persisted or forwarded without replacing
the backend sampler. MFQd attaches the native session identifier to backend
requests. C++ workers keep compact device-side attention-state snapshots for
exact token prefixes, so interleaved sessions and forks can prefill only the
appended suffix. Snapshots cover ordinary full-attention KV state, DeepSeek V4
local and compressed attention state, and GLM DSA MLA/index state. The cache
defaults to four sessions and a 2 GiB total budget;
`MFQ_SERVER_MAX_KV_SESSIONS` and `MFQ_SERVER_KV_SESSION_BYTES` override those
limits. Each session retains up to four immutable prompt snapshots so a fork
from an earlier turn can still reuse its matching prefix;
`MFQ_SERVER_MAX_KV_SNAPSHOTS_PER_SESSION` changes that limit. MFQd forwards
fork and delete operations to compatible native workers, and deleting a
session releases all of its runtime snapshots. Setting any cache limit to zero
disables the corresponding runtime session caching scope.

Media upload, realtime audio, and model load or unload endpoints currently
return an explicit `501` error. Runtime instance listing returns an empty list
until native worker registration is connected.

The React client lives in `clients/web`. Its development server proxies native
API requests to MFQd:

```bash
cd clients/web
npm install
npm run dev
```

For a same-origin local deployment, build the client and give its output to
MFQd:

```bash
npm run build
mfqd --web-root clients/web/dist
```
