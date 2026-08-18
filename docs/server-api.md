# MFQ Server native protocol

MFQ Server is the long-running service boundary for Web clients, MFQ Studio, SDKs,
and native MFQ runtime workers. Protocol version `1.0` defines the data model
used across these boundaries.

## Public client API

The checked-in OpenAPI document is `mfq/server/protocol/openapi.json`. It defines:

- persistent session creation, response generation, forking, and deletion;
- response history with exact sampling and performance metadata;
- content-addressed media upload and durable document extraction;
- model discovery, validation, load, unload, and managed runtime instances;
- persistent background jobs, logs, progress, cancellation, and artifacts;
- runtime metrics, logs, memory, queues, and instance inspection;
- persistent runtime profiles with artifact drift detection;
- artifact lineage with producing parameters, sources, and validation jobs;
- normalized Hugging Face and ModelScope discovery plus resumable download jobs;
- reproducible dataset registration and durable evaluation comparison;
- health-aware remote-node registration, inventory, routing, and aggregate metrics;
- portable session export/import with referenced media and documents;
- viewer, operator, and administrator API-key roles;
- structured output, model-visible tool definitions, and tool-result turns;
- MCP server registration, tool discovery, and confirmed tool execution;
- the `/api/v1/realtime` WebSocket path and its event names.

Messages contain typed parts: text, reasoning, image, video, audio, document,
transcript, tool call, tool result, or generated audio. Media parts refer to an
uploaded media object by identifier and exact SHA-256 digest. A response
request carries an idempotent request identifier and the expected session
revision so concurrent updates cannot silently overwrite one another.

Document originals are stored as immutable media. Extracted text is bounded,
cached in SQLite, and expanded by MFQ Server before inference. PDF extraction uses
the optional daemon dependency; plain text and DOCX extraction remain
self-contained.

Realtime frames carry a monotonically increasing sequence number. Full-duplex
PCM chunks include their own audio sequence and timestamp. The initial JSON
transport uses base64-encoded mono or multichannel `pcm_s16le` data.

## Runtime IPC

The versioned Protobuf source is `mfq/server/protocol/runtime.proto`. MFQ Server exchanges
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
python -m mfq.server.openapi mfq/server/protocol/openapi.json
python -m mfq.server.openapi --check mfq/server/protocol/openapi.json
```

Breaking changes require a new protocol version and a new Protobuf package.

## Conversation storage

`mfq.server.storage.SessionStore` provides the initial SQLite WAL conversation layer.
Messages are immutable records. Sessions reference an ordered message chain, so
forks share their existing prefix and append independent messages afterward.
Every append requires the current session revision; stale concurrent writes are
rejected before a message is inserted. SQLite transactions end before runtime
inference begins.

## Text gateway

`mfq serve` is the only public server entry point. It resolves the native
runtime from the build manifest written by `mfq build` and exposes the
persistent MFQ Server API. The server may remain idle with no model, or it can
start a requested model on a private loopback port. A missing native executable
is rebuilt from the recipe retained in the manifest:

```bash
mfq serve --host 127.0.0.1 --port 8090
```

MFQ Server listens on `127.0.0.1:8090` by default. Source users do not launch or
manage the native executable directly; packaged launchers may pass their bundled
worker through `--running-executable`. Public MFQ Server API authentication is
optional and reads its token from `MFQ_SERVER_API_KEY`; pass `--api-key-env` to
select another environment variable.
That value is the root credential for creating scoped subkeys. Subkeys may use
viewer, operator, or administrator roles and can add explicit inference,
model-management, job/artifact, or administrative scopes. They may expire and
can be revoked or rotated. MFQ Server stores only SHA-256 digests and a short display
prefix; a generated plaintext token is returned once. Studio keeps remote
credentials in the platform Keychain or Credential Manager rather than
conversation data, browser local storage, command-line arguments, or repository
files. Remote-node records likewise persist only the name of an environment
variable supplied by the service manager.
Text responses can be returned as JSON or streamed as typed SSE
frames. Reasoning deltas, tool calls, finish reasons, token usage, failures,
and session state transitions are persisted or forwarded without replacing
the backend sampler. The final response records backend-reported TTFT, prefill,
decode, token counts, finish reason, and the exact sampling settings. MFQ Server
attaches the native session identifier to backend
requests. C++ workers keep compact device-side attention-state snapshots for
exact token prefixes, so interleaved sessions and forks can prefill only the
appended suffix. Snapshots cover ordinary full-attention KV state, DeepSeek V4
local and compressed attention state, and GLM DSA MLA/index state. The cache
defaults to four sessions and a 2 GiB total budget;
`MFQ_SERVER_MAX_KV_SESSIONS` and `MFQ_SERVER_KV_SESSION_BYTES` override those
limits. Each session retains up to four immutable prompt snapshots so a fork
from an earlier turn can still reuse its matching prefix;
`MFQ_SERVER_MAX_KV_SNAPSHOTS_PER_SESSION` changes that limit. MFQ Server forwards
fork and delete operations to compatible native workers, and deleting a
session releases all of its runtime snapshots. CUDA and Metal workers expose
cache queries, hits, reused tokens, sessions, snapshots, retained tokens, bytes,
and configured limits through runtime status. MFQ Server forwards guarded cache clears
through `POST /api/v1/runtime/cache/clear`; the worker rejects the operation while
generation or a full-duplex session is active. Setting any cache limit to zero
disables the corresponding runtime session caching scope.

Media uploads are immutable, content-addressed blobs stored next to the MFQ Server
SQLite database. The client supplies the exact SHA-256 digest, and MFQ Server rejects
forged references before appending a message. Uploaded objects are available
from `/api/v1/media/{id}` for previews and are forwarded as OpenAI-compatible
multimodal content to capable backends. For MiniCPM-o 4.5, MFQ Server performs the
official image slicing, bicubic resize, normalization, patch packing, and video
frame sampling once on the CPU. It sends the resulting versioned tensor package
to either native CUDA or Metal through the internal `mfq_multimodal` request
field. The native server derives image bounds from the final chat-template
tokens and rejects mismatched placeholders or tensor geometry. Visual requests
do not reuse text-only prefix-cache entries.

Recorded audio follows the separate realtime audio gateway rather than the
vision tensor package. Managed runtimes are discovered from configured model
roots and launched by MFQ Server. Each instance has an explicit request capacity;
additional requests enter a cancellation-aware queue exposed by the runtime
API. Externally managed OpenAI-compatible runtimes remain available as a
fallback.

Runtime profiles persist model selection, device selection, context,
prefill chunking, expert offload budget, prefix-cache limits, duplex capability,
and sampling defaults. Profiles bind to the discovered artifact fingerprint and
modification timestamp. MFQ Server marks a profile drifted when that artifact is
replaced or disappears, and profile loading requires an explicit override after
drift. Profiles are managed through `/api/v1/runtime/profiles`.

## Jobs and model operations

Long-running operations are persistent typed jobs rather than shell commands
hidden behind the UI. MFQ Server currently exposes download, conversion,
quantization, imatrix calibration, validation, perplexity/KLD/logit generation,
and benchmark job types. Jobs retain state, structured progress, bounded logs,
output artifacts, and cancellation across daemon restarts. Model discovery and
validation feed the managed load/unload API used by Studio.

Published job artifacts also create durable lineage records. Each record keeps
the producing job and kind, its fully defaulted parameters, upstream URIs,
bounded public metadata, and later validation job identifiers. Clients query the
records through `GET /api/v1/artifacts/lineage` without receiving host paths.

Perplexity and kernel benchmark jobs also publish durable evaluation records.
Each result binds numeric metrics to a dataset manifest, hardware identity,
runtime identity, model, and canonical comparison key. Registered WikiText-2 or
custom datasets retain their exact SHA-256 and byte size. The comparison API
rejects results with different evaluation kinds or comparison keys instead of
presenting invalid deltas.

Model hub discovery normalizes search results, revisions, file lists, and byte
sizes from Hugging Face and ModelScope. Download jobs accept the inspected size,
preflight available disk space while preserving existing resumable files, verify
that no partial markers remain and the completed byte count is plausible, and can
be retried from their original parameters. Workspace artifacts have an explicit,
confined removal endpoint; model roots and arbitrary host paths cannot be deleted.

## Sessions and distributed nodes

`GET /api/v1/sessions/{id}/export` creates a versioned archive with immutable
messages, referenced media, and extracted document metadata. Import creates a
new session, verifies every media digest, remaps identifiers, and never overwrites
the source branch.

Administrators can register remote MFQ Server nodes under `/api/v1/cluster/nodes`.
MFQ Server probes health, native model inventory, and runtime status without using host
proxy settings, then routes a matching model to the healthy least-active node.
Remote session affinity, forks, and deletion are propagated. The local runtime
remains the fallback when no remote advertises the requested model. Node records
expose health and public runtime metrics; aggregate runtime status includes the
remote totals.

## Tools and MCP

Native response requests accept OpenAI-style `tools`, `tool_choice`, and
`response_format` fields. Tool results are appended with `input_role=tool` and
then continued through the same session branch.

MCP servers can use stdio or Streamable HTTP. Secrets are referenced by
environment-variable name and are never persisted as values. Servers are
disabled unless explicitly enabled, tool calls require `confirm=true`, HTTP
connections ignore host proxy settings, and request time, response size, and
audit logging are bounded. MFQ Server logs the server and tool identity but not tool
arguments.

The React and Tauri clients share the `MFQStudio` package. `mfq serve` builds the
web client automatically when its sources are newer than the local output. Its
development server can also proxy API requests to MFQ Server:

```bash
cd MFQStudio
npm install
npm run dev
```

For a same-origin local deployment, `mfq serve` builds and serves the client
automatically. It can start empty and load or switch models through the catalog.
A prebuilt client can still be selected explicitly:

```bash
npm run build
mfq serve --model path/to/model.mfq --web-root MFQStudio/dist
```

Use `--no-web-ui` for an API-only process.

The same package runs and bundles the desktop client through Tauri:

```bash
npm run tauri dev
npm run tauri build
```

In local mode, Studio starts `mfq serve` without an initial model, gives it a
private application-data model catalog, and then uses the public model load API.
If a platform runtime is bundled next to the application or in its resource
directory, Studio passes it through `--running-executable`. The Models and jobs
page can also open a native file picker and load an arbitrary local `.mfq` file.
Studio registers the selected path in its private catalog without copying the
model; selecting one shard registers the complete sibling shard family.
