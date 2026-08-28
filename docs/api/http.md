# MFQ Server HTTP API

MFQ Server provides a persistent HTTP API for MFQ Studio, scripts, and SDKs.
The machine-readable contract is
[`mfq/server/protocol/openapi.json`](../../mfq/server/protocol/openapi.json); it
defines all fields, constraints, and response schemas.

Default server address:

```text
http://127.0.0.1:8090
```

The versioned API root is `http://127.0.0.1:8090/api/v1`. The current OpenAPI
contract contains 57 HTTP paths and 75 operations. The unversioned `/health`
endpoint is not part of OpenAPI.

The running server exposes Swagger and the live schema at:

```text
http://127.0.0.1:8090/docs
http://127.0.0.1:8090/openapi.json
```

## 1. Common conventions

### 1.1 Authentication and authorization

When `MFQ_SERVER_API_KEY` (or the environment variable selected by
`--api-key-env`) is not configured, routes under `/api/` do not require
authentication. Once a root credential is configured, every API request must
send:

```http
Authorization: Bearer <token>
```

The root credential has administrator access. Child keys created through
`/api/v1/auth/keys` may have explicit scopes, a role, or both. Roles grant these
additional scopes:

| Role | Granted scopes |
|---|---|
| `viewer` | `inference` |
| `operator` | `inference`, `models`, `jobs` |
| `administrator` | `admin` |

| Route area | Required scope |
|---|---|
| Sessions, presets, media, documents, and MCP reads | `inference` |
| `GET` operations under `/models`, `/runtime`, and `/hub` | `inference` |
| Model or runtime mutations | `models` |
| Jobs, lineage, datasets, and evaluations | `jobs` |
| MCP mutations, key management, and cluster nodes | `admin` |

The `admin` scope permits every operation. `/health` is outside `/api/` and
never requires a bearer credential.

### 1.2 Types and strict validation

| Notation | Meaning |
|---|---|
| `UUID` | Standard UUID string; every `*_id` path parameter is a UUID. |
| `DateTime` | Timezone-aware ISO 8601 date and time. |
| `SHA-256` | 64 lowercase hexadecimal characters. |
| `Object` | JSON object; fields may be extended by the server unless a strict schema is named. |
| `String[]` | JSON array of strings. |

Request objects derived from `ProtocolModel` reject unknown fields. Free-form
objects such as `metadata` and `payload`, plus backend pass-through objects, are
exceptions. Missing fields, type errors, unknown fields, and range violations
return HTTP `422`.

### 1.3 Error responses

API errors use one envelope:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "request validation failed",
    "retryable": false,
    "details": {
      "errors": []
    }
  }
}
```

Declared error statuses include `400`, `401`, `403`, `404`, `409`, `413`,
`415`, `422`, `501`, `502`, and `503`; the actual status depends on the
operation and runtime state.

- `401` means the credential is missing, invalid, revoked, or expired. The
  response includes `WWW-Authenticate: Bearer`.
- `403` means a valid credential lacks the required scope.
- `409` commonly represents a session-revision conflict, idempotency conflict,
  or an already-running response.
- `501` means the configured server cannot provide a feature, such as API-key
  management or a realtime backend.

### 1.4 Lists, deletes, and asynchronous operations

List responses use `{"data": [...]}`. Successful deletion returns
`204 No Content`.

Model load/unload, runtime-profile load, and job creation or retry return
`202 Accepted`. Acceptance means the operation was queued; it does not mean the
operation has completed.

`OperationAccepted` has this shape:

| Field | Type | Description |
|---|---|---|
| `operation_id` | UUID | Background operation or job identifier. |
| `status` | String | Fixed at `accepted`. |

## 2. Endpoint reference

Request and response names below come from OpenAPI. A dash means there is no
JSON body or no fixed JSON response schema. OpenAPI contains the query
parameters and field constraints.

### 2.1 Authentication

| Method | Path | Request | Success |
|---|---|---|---|
| `GET` | `/api/v1/auth/keys` | — | `200 ApiKeyList` |
| `POST` | `/api/v1/auth/keys` | `CreateApiKeyRequest` | `201 ApiKeySecretResource` |
| `POST` | `/api/v1/auth/keys/{key_id}/revoke` | — | `200 ApiKeyResource` |
| `POST` | `/api/v1/auth/keys/{key_id}/rotate` | — | `200 ApiKeySecretResource` |

### 2.2 Sessions

| Method | Path | Request | Success |
|---|---|---|---|
| `GET` | `/api/v1/sessions` | — | `200 SessionList` |
| `POST` | `/api/v1/sessions` | `CreateSessionRequest` | `201 SessionResource` |
| `POST` | `/api/v1/sessions/import` | `SessionArchive` | `201 SessionImportResult` |
| `GET` | `/api/v1/sessions/{session_id}` | — | `200 SessionResource` |
| `PATCH` | `/api/v1/sessions/{session_id}` | `UpdateSessionRequest` | `200 SessionResource` |
| `DELETE` | `/api/v1/sessions/{session_id}` | — | `204` |
| `GET` | `/api/v1/sessions/{session_id}/export` | — | `200 SessionArchive` |
| `POST` | `/api/v1/sessions/{session_id}/fork` | `ForkSessionRequest` | `201 SessionResource` |
| `GET` | `/api/v1/sessions/{session_id}/messages` | — | `200 MessageList` |
| `POST` | `/api/v1/sessions/{session_id}/messages` | `AppendMessageRequest` | `201 AppendMessageResult` |
| `GET` | `/api/v1/sessions/{session_id}/responses` | — | `200 ResponseList` |
| `POST` | `/api/v1/sessions/{session_id}/responses` | `CreateResponseRequest` | `200 ResponseResource` or SSE |
| `POST` | `/api/v1/sessions/{session_id}/responses/cancel` | — | `200 ResponseResource` |
| `POST` | `/api/v1/sessions/{session_id}/rewind` | `RewindSessionRequest` | `200 SessionResource` |

### 2.3 Generation presets

| Method | Path | Request | Success |
|---|---|---|---|
| `GET` | `/api/v1/presets` | — | `200 GenerationPresetList` |
| `POST` | `/api/v1/presets` | `CreateGenerationPresetRequest` | `201 GenerationPresetResource` |
| `PUT` | `/api/v1/presets/{preset_id}` | `UpdateGenerationPresetRequest` | `200 GenerationPresetResource` |
| `DELETE` | `/api/v1/presets/{preset_id}` | — | `204` |

`PUT` replaces the complete preset; it is not a partial update.

### 2.4 Media and documents

| Method | Path | Request | Success |
|---|---|---|---|
| `POST` | `/api/v1/documents` | `CreateDocumentRequest` | `201 DocumentResource` |
| `GET` | `/api/v1/documents/{media_id}` | — | `200 DocumentResource` |
| `POST` | `/api/v1/media` | Raw bytes | `201 MediaResource` |
| `GET` | `/api/v1/media/{media_id}` | — | `200` raw bytes |

### 2.5 MCP

| Method | Path | Request | Success |
|---|---|---|---|
| `GET` | `/api/v1/mcp/servers` | — | `200 McpServerList` |
| `POST` | `/api/v1/mcp/servers` | `CreateMcpServerRequest` | `201 McpServerResource` |
| `PATCH` | `/api/v1/mcp/servers/{server_id}` | `UpdateMcpServerRequest` | `200 McpServerResource` |
| `DELETE` | `/api/v1/mcp/servers/{server_id}` | — | `204` |
| `GET` | `/api/v1/mcp/tools` | — | `200 McpToolList` |
| `POST` | `/api/v1/mcp/tools/call` | `McpToolCallRequest` | `200 McpToolCallResult` |

Tool calls require `confirm: true`. Server definitions support `stdio` and
`streamable_http`; secret header values are read from environment variables and
are not stored in the server definition.

### 2.6 Jobs

| Method | Path | Request | Success |
|---|---|---|---|
| `GET` | `/api/v1/jobs` | — | `200 JobList` |
| `POST` | `/api/v1/jobs` | `CreateJobRequest` | `202 JobResource` |
| `DELETE` | `/api/v1/jobs/completed` | — | `204` |
| `GET` | `/api/v1/jobs/kinds` | — | `200 JobKindList` |
| `GET` | `/api/v1/jobs/{job_id}` | — | `200 JobResource` |
| `DELETE` | `/api/v1/jobs/{job_id}` | — | `204` |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | — | `200 JobResource` |
| `GET` | `/api/v1/jobs/{job_id}/events` | — | `200 JobEventList` |
| `GET` | `/api/v1/jobs/{job_id}/events/stream` | — | `200` SSE |
| `POST` | `/api/v1/jobs/{job_id}/retry` | — | `202 JobResource` |

Call `GET /api/v1/jobs/kinds` before creating a job. The response describes the
job kinds registered by the current deployment and includes each kind's
payload JSON Schema. Do not assume every deployment registers the same tool
jobs.

### 2.7 Models, hubs, and artifacts

| Method | Path | Request | Success |
|---|---|---|---|
| `GET` | `/api/v1/artifacts/lineage` | — | `200 ArtifactLineageList` |
| `POST` | `/api/v1/artifacts/remove` | `DeleteWorkspaceArtifactRequest` | `200 Object` |
| `GET` | `/api/v1/hub/models` | — | `200 HubModelSearchResult` |
| `GET` | `/api/v1/hub/models/{provider}/{owner}/{name}` | — | `200 HubModelInfo` |
| `GET` | `/api/v1/models` | — | `200 ModelArtifactList` |
| `GET` | `/api/v1/models/directories` | — | `200 ModelDirectoryList` |
| `POST` | `/api/v1/models/directories/register` | `RegisterModelDirectoryRequest` | `200 ModelArtifactList` |
| `POST` | `/api/v1/models/load` | `ModelLoadRequest` | `202 OperationAccepted` |
| `POST` | `/api/v1/models/unload` | `ModelUnloadRequest` | `202 OperationAccepted` |

Hub providers are `huggingface` and `modelscope`. Artifact removal accepts only
restricted `workspace://` URIs; it does not delete arbitrary host paths or
model roots.

### 2.8 Datasets and evaluations

| Method | Path | Request | Success |
|---|---|---|---|
| `GET` | `/api/v1/datasets` | — | `200 DatasetList` |
| `POST` | `/api/v1/datasets` | `CreateDatasetRequest` | `201 DatasetResource` |
| `DELETE` | `/api/v1/datasets/{dataset_id}` | — | `204` |
| `GET` | `/api/v1/evaluations` | — | `200 EvaluationResultList` |
| `POST` | `/api/v1/evaluations/compare` | `CompareEvaluationsRequest` | `200 EvaluationComparisonResource` |

Only evaluations with compatible kinds and comparison keys can be compared.

### 2.9 Cluster nodes

| Method | Path | Request | Success |
|---|---|---|---|
| `GET` | `/api/v1/cluster/nodes` | — | `200 RemoteNodeList` |
| `POST` | `/api/v1/cluster/nodes` | `CreateRemoteNodeRequest` | `201 RemoteNodeResource` |
| `PUT` | `/api/v1/cluster/nodes/{node_id}` | `UpdateRemoteNodeRequest` | `200 RemoteNodeResource` |
| `DELETE` | `/api/v1/cluster/nodes/{node_id}` | — | `204` |

`PUT` replaces the complete node definition. Remote credentials are referenced
by environment-variable name and are not persisted as plaintext values.

### 2.10 Runtime

| Method | Path | Request | Success |
|---|---|---|---|
| `POST` | `/api/v1/runtime/cache/clear` | — | `200 Object` |
| `GET` | `/api/v1/runtime/capabilities` | — | `200 RuntimeCapabilitiesResource` |
| `GET` | `/api/v1/runtime/instances` | — | `200 RuntimeInstanceList` |
| `GET` | `/api/v1/runtime/logs` | — | `200 RuntimeLogList` |
| `GET` | `/api/v1/runtime/metrics` | — | `200 RuntimeMetricList` |
| `GET` | `/api/v1/runtime/models` | — | `200 Object` |
| `GET` | `/api/v1/runtime/profiles` | — | `200 RuntimeProfileList` |
| `POST` | `/api/v1/runtime/profiles` | `CreateRuntimeProfileRequest` | `201 RuntimeProfileResource` |
| `GET` | `/api/v1/runtime/profiles/{profile_id}` | — | `200 RuntimeProfileResource` |
| `PUT` | `/api/v1/runtime/profiles/{profile_id}` | `UpdateRuntimeProfileRequest` | `200 RuntimeProfileResource` |
| `DELETE` | `/api/v1/runtime/profiles/{profile_id}` | — | `204` |
| `POST` | `/api/v1/runtime/profiles/{profile_id}/load` | `RuntimeProfileLoadRequest` | `202 OperationAccepted` |
| `GET` | `/api/v1/runtime/realtime/capabilities` | — | `200 Object` |
| `POST` | `/api/v1/runtime/reload` | `RuntimeReloadRequest` | `200 Object` |
| `GET` | `/api/v1/runtime/status` | — | `200 Object` |

Runtime status, realtime capabilities, reload, cache-clear, and model-list
responses may contain backend-specific fields. Cache clearing can be rejected
while generation or a full-duplex session is active.

## 3. Session and response workflow

### 3.1 Create a session

Create a persistent session with `POST /api/v1/sessions`:

```json
{
  "model": "my-model"
}
```

`CreateSessionRequest`:

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `model` | String | Yes | — | Model identifier, 1–255 characters. |
| `mode` | String | No | `text` | `text`, `voice`, or `full_duplex`. |
| `title` | String or null | No | `null` | Optional title, at most 512 characters. |
| `metadata` | Object | No | `{}` | Application-defined metadata. |

Keep the returned `id` and `revision`. Mutations reject a stale
`expected_revision` with HTTP `409`.

### 3.2 Append, fork, rewind, export, and import

`AppendMessageRequest` requires `expected_revision`, a role (`system`, `user`,
`assistant`, or `tool`), and at least one `ContentPart`.

`ForkSessionRequest` may specify `at_message_id`, whether to include that
message, and an optional title. `RewindSessionRequest` requires the current
revision, a target message, and whether to retain the target.

Session export uses the `mfq-session-v1` archive format and includes the
session, messages, referenced media, and document metadata. Import validates
media digests, remaps identifiers, and creates a new session; it never
overwrites the source session.

### 3.3 Create a response

Send `POST /api/v1/sessions/{session_id}/responses` with:

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `request_id` | UUID | Yes | — | Client-generated idempotency key. |
| `expected_revision` | Integer | Yes | — | Current non-negative session revision. |
| `input` | `ContentPart[]` | Yes | — | At least one input part. |
| `input_role` | String | No | `user` | `user` or `tool`. |
| `sampling` | `SamplingParams` | No | Schema defaults | Sampling controls. |
| `system_prompt` | String or null | No | `null` | At most 32,768 characters. |
| `include_reasoning_history` | Boolean | No | `true` | Include prior reasoning in model context. |
| `tools` | `ToolDefinition[]` | No | `[]` | At most 128 function tools. |
| `tool_choice` | String or Object | No | `auto` | `auto`, `none`, `required`, or a named function. |
| `response_format` | Object | No | `{"type":"text"}` | Text, JSON object, or JSON Schema output. |
| `stream` | Boolean | No | `true` | Return SSE when true; JSON when false. |

A minimal authenticated streaming request is:

```shell
curl -N -X POST http://127.0.0.1:8090/api/v1/sessions/<session-id>/responses \
  -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  --data '{
    "request_id": "00000000-0000-0000-0000-000000000001",
    "expected_revision": 0,
    "input": [{"type": "text", "text": "Hello"}],
    "stream": true
  }'
```

For `input_role=tool`, `input` must contain exactly one `tool_result` part.
Normal user input must not contain a `tool_result`. Tool names must be unique,
and a named `tool_choice` must reference one of the supplied tools.

### 3.4 Idempotency and safe retries

`request_id` is scoped to one session. Its fingerprint covers the full request
except `stream`.

| Situation | Server behavior | Client action |
|---|---|---|
| Same ID and fingerprint; previous response completed | Returns the saved JSON response or reconstructs an SSE replay without running inference again. Original chunk boundaries are not guaranteed. | Retry safely with the same request. |
| Same ID but a different fingerprint | `409 idempotency_conflict`. | Restore the original body or create a new request ID. |
| Same ID while the original request is running | `409 response_in_progress` with `retryable: true`. | Back off or query response history; do not start a parallel duplicate. |
| Stale `expected_revision` | `409 revision_conflict` with expected and actual revisions in `details`. | Reload the session and retry against the new revision. |

For non-streaming requests, backend failure may return HTTP `502`. A streaming
request has already committed an HTTP `200`; later backend failure is emitted as
an SSE `error` followed by the final `session.state`, without a
`response.completed` event.

### 3.5 Sampling defaults

| Field | Default | Constraint |
|---|---|---|
| `max_tokens` | 4096 | At least 1. |
| `temperature` | 1.0 | At least 0. |
| `top_k` | 20 | At least 0. |
| `top_p` | 0.95 | Greater than 0 and at most 1. |
| `presence_penalty` | 0.0 | Range -2 to 2. |
| `frequency_penalty` | 0.0 | Range -2 to 2. |
| `repetition_penalty` | 1.0 | Greater than 0. |
| `seed` | `null` | Non-negative when set. |
| `enable_thinking` | `true` | Boolean. |
| `reasoning_effort` | `null` | 1–32 characters when set. |

## 4. Content, tools, and media

### 4.1 Content parts

`ContentPart` is discriminated by `type`:

| Type | Additional fields |
|---|---|
| `text`, `reasoning` | `text` String. |
| `image` | `media: MediaRef`; optional positive `width` and `height`. |
| `video` | `media`; optional positive `width`/`height` and non-negative `duration_ms`. |
| `audio`, `generated_audio` | `media`, positive `sample_rate_hz`, `channels` in 1–8, optional `duration_ms`. |
| `transcript` | `text`; optional `language`, `start_ms`, and `end_ms`. |
| `document` | `media` and `name`. |
| `tool_call` | `call_id`, `name`, and `arguments` Object. |
| `tool_result` | `call_id`, arbitrary `result`, and optional `is_error`. |

`MediaRef` contains `id`, `sha256`, `mime_type`, and non-negative `byte_size`.
Media references must exactly match a previously uploaded record and use the
correct MIME family. Document parts must reference an extracted document whose
media reference and name match the stored record.

`reasoning` and `tool_call` parts are valid only in assistant messages. A tool
message must contain exactly one `tool_result`; other roles cannot contain that
part.

### 4.2 Function tools and structured output

A function tool has this form:

```json
{
  "type": "function",
  "function": {
    "name": "lookup_weather",
    "description": "Optional description",
    "parameters": {"type": "object"}
  }
}
```

Function names match `[A-Za-z_][A-Za-z0-9_.-]{0,127}`. `response_format`
supports text, JSON object, or strict/non-strict JSON Schema output as defined
by the OpenAPI schemas.

### 4.3 Upload media

`POST /api/v1/media` accepts raw bytes, not JSON or multipart data. It requires:

| Location | Name | Description |
|---|---|---|
| Header | `Content-Type` | Actual MIME type of the raw body. |
| Header | `X-Content-SHA256` | Exact SHA-256 digest of the body. |
| Body | Raw bytes | Non-empty and at most 512 MiB. |

Allowed media include `image/*`, `audio/*`, `video/*`, `text/*`, JSON, PDF,
XML, YAML, octet stream, MFQ imatrix, and DOCX. Parameters such as `charset`
are removed during MIME normalization.

```shell
curl -X POST http://127.0.0.1:8090/api/v1/media \
  -H 'Content-Type: image/png' \
  -H "X-Content-SHA256: $(shasum -a 256 image.png | awk '{print $1}')" \
  --data-binary @image.png
```

`GET /api/v1/media/{media_id}` returns the original bytes and MIME type with an
SHA-256 `ETag`, `Cache-Control: private, immutable`, and
`X-Content-Type-Options: nosniff`.

Common upload errors are `empty_media` (`400`), `media_too_large` (`413`),
`unsupported_media_type` (`415`), and `media_digest_mismatch` (`422`).

## 5. Models, jobs, and runtime state

### 5.1 Model loading

Discover local artifacts with `GET /api/v1/models`. Register additional roots
through `/api/v1/models/directories/register`. A `ModelLoadRequest` selects the
model and may set device IDs, context size, prefill chunk size, MoE cache size,
prefix-cache limits, duplex mode, pinning, idle TTL, and sampling defaults.

```shell
curl -X POST http://127.0.0.1:8090/api/v1/models/load \
  -H 'Content-Type: application/json' \
  --data '{"model":"model","context_size":32768}'
```

The response is `202 OperationAccepted`. Track the returned operation through
the jobs API and inspect active instances through
`GET /api/v1/runtime/instances`.

### 5.2 Jobs

A `JobResource` moves through `queued`, `running`, `cancelling`, `succeeded`,
`failed`, `cancelled`, or `interrupted`. It includes progress, cancellation
state, optional result/error data, and timestamps.

Use `/events` for paged event history or `/events/stream` for SSE. Unknown job
kinds return `422 job_kind_unavailable`; invalid state transitions return
`409 job_state_conflict`.

### 5.3 Runtime profiles and drift

A runtime profile stores a named `ModelLoadRequest` together with the exact
artifact identity observed when the profile was created. Replaced or missing
artifacts mark the profile as drifted. Loading a drifted profile requires an
explicit `allow_drift: true`.

Runtime instances expose model, state, devices, active sessions, queued
requests, optional memory and context metrics, and identity hashes. Runtime
capabilities identify architecture family and text/image/video/audio/duplex
features.

## 6. Server-Sent Events

### 6.1 Response generation

When `stream=true`, response creation returns `Content-Type:
text/event-stream`. Each event uses a `RealtimeFrame` envelope:

```text
event: response.text.delta
id: 2
data: {"protocol_version":"1.0","session_id":"...","sequence":2,"timestamp":"...","payload":{...}}

```

The event name equals `payload.type`. The HTTP response stream currently emits:

| Event | Payload |
|---|---|
| `response.text.delta`, `response.reasoning.delta` | `response_id`, `delta`. |
| `response.tool_call.delta` | `response_id`, `index`, optional `call_id`/`name`, and `arguments_delta`. |
| `response.completed` | `response_id`, `finish_reason`, optional `usage` and `performance`. |
| `session.state` | `state`, `revision`. |
| `error` | Common `ErrorDetail`. |

Response SSE does not emit `response.audio.delta`, `response.interrupted`, or
`runtime.metrics`; those events belong to the realtime WebSocket protocol. See
the [WebSocket API](websocket.md).

Response SSE does not support `Last-Event-ID` continuation. If a connection is
interrupted, repeat the POST with the identical body and `request_id` to use the
completed-response replay semantics.

### 6.2 Job events

`GET /api/v1/jobs/{job_id}/events/stream` returns SSE. The non-negative `after`
query parameter defaults to 0. `Last-Event-ID` may also be supplied; the server
continues after the greater of the two values.

```text
id: <JobEventResource.sequence>
event: <JobEventResource.type>
data: <JobEventResource JSON>

```

The stream sends comment keep-alives while idle and ends after the job reaches
`succeeded`, `failed`, `cancelled`, or `interrupted`.

## 7. Health check

`GET /health` returns:

```json
{
  "status": "ok",
  "service": "mfq-server",
  "protocol_version": "1.0"
}
```

`/health` is not in OpenAPI and never requires authentication.

## 8. Contract maintenance

Regenerate and verify OpenAPI after changing routes or protocol models:

```shell
uv run -m mfq.server.openapi mfq/server/protocol/openapi.json
uv run -m mfq.server.openapi --check mfq/server/protocol/openapi.json
```

WebSockets are not represented as OpenAPI path operations. The
`x-mfq-websocket` extension records the legacy frame model; current handshake
and native realtime proxy behavior are documented in
[the WebSocket API](websocket.md).
