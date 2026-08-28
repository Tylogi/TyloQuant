# MFQ Server WebSocket API

MFQ Server has two WebSocket paths. `/api/v1/runtime/realtime` proxies native
realtime audio in both directions. The legacy `/api/v1/realtime` route remains
registered but rejects connections.

Default server address:

```text
ws://127.0.0.1:8090
```

OpenAPI does not list either path as an operation. Its `x-mfq-websocket`
extension records the legacy `/api/v1/realtime` frame schema and events.
`/api/v1/runtime/realtime` is an untyped proxy and follows the active native
backend's protocol. HTTP endpoints and SSE are documented in the
[HTTP API](http.md).

## 1. Connection and authentication

### 1.1 URLs

| URL | Status | Purpose |
|---|---|---|
| `ws://127.0.0.1:8090/api/v1/runtime/realtime?mode=audio` | Backend-dependent | Proxies text and binary frames bidirectionally between the client and the active native runtime. |
| `ws://127.0.0.1:8090/api/v1/realtime` | Currently unavailable | Closes with code 1013 after successful authentication; do not use it as a realtime client entry point. |

Use `wss` instead of `ws` when TLS terminates at the server or reverse proxy.

### 1.2 Authentication

When no API key is configured, neither path requires authentication. When API
keys are enabled, the client must provide a credential with the `inference` or
`admin` scope. Prefer the authorization header:

```http
Authorization: Bearer <token>
```

Browser clients that cannot set WebSocket headers may use the query parameter:

```text
ws://127.0.0.1:8090/api/v1/runtime/realtime?mode=audio&access_token=<token>
```

The server reads `access_token` only when no bearer token is present in the
header. Query strings are commonly recorded by logs and proxies, so prefer a
header or a short-lived credential.

Authentication failure closes the connection with:

| Close code | Reason |
|---|---|
| `1008` | `invalid API credential` |

### 1.3 Mode and connection failures

`/api/v1/runtime/realtime` accepts only `mode=audio`; omitting `mode` also
defaults to `audio`. Any other value is rejected before the connection is
accepted:

| Close code | Reason |
|---|---|
| `1008` | `audio mode is required` |

If no loaded backend can provide realtime transport, or if the proxy encounters
an unhandled error, the server attempts to send an error JSON object and then
closes with:

| Close code | Reason |
|---|---|
| `1011` | `realtime proxy failed` |

When the client disconnects normally or either proxy direction finishes first,
the server cancels the other direction and makes a best-effort close with code
`1000`.

## 2. Native realtime audio proxy

### 2.1 Proxy boundary

After authentication and connection setup, MFQ Server applies these rules:

1. Client text frames are forwarded unchanged to the upstream runtime.
2. Client binary frames are forwarded unchanged to the upstream runtime.
3. Upstream text and binary frames are forwarded unchanged to the client.
4. The HTTP server does not parse, resequence, re-encode, or validate frame
   payloads against a JSON Schema.

The client and native runtime must use the same realtime protocol. Do not send
HTTP `ResponseResource` objects or SSE records to this socket.

### 2.2 Normalized `RealtimeFrame`

The MFQ protocol model defines a normalized envelope for JSON realtime
messages. Clients using this format send and receive UTF-8 text frames:

```json
{
  "protocol_version": "1.0",
  "session_id": "00000000-0000-0000-0000-000000000000",
  "sequence": 0,
  "timestamp": "2026-08-18T12:00:00+00:00",
  "payload": {
    "type": "input_audio.commit",
    "last_audio_sequence": 0
  }
}
```

| Envelope field | Type | Required | Description |
|---|---|---:|---|
| `protocol_version` | String | No | Fixed at `1.0`; defaults to `1.0` when omitted. |
| `session_id` | UUID | Yes | MFQ session associated with the realtime connection. |
| `sequence` | Integer | Yes | Non-negative, monotonically increasing transport sequence. |
| `timestamp` | DateTime | Yes | Timezone-aware message timestamp. |
| `payload` | Object | Yes | Discriminated by `type`. |

The proxy does not enforce this envelope. If the upstream runtime uses another
frame format, the server forwards it unchanged and the client must follow that
format.

### 2.3 Client-to-server events

| `payload.type` | Fields | Meaning |
|---|---|---|
| `input_audio.delta` | `audio_sequence` (non-negative Integer), `timestamp_ms` (non-negative Integer), `encoding` (fixed `pcm_s16le`), `sample_rate_hz` (default 16000, at least 1), `channels` (default 1, range 1–8), and `data_base64` (Base64 bytes) | Appends one PCM S16LE audio chunk. |
| `input_audio.commit` | `last_audio_sequence` (non-negative Integer) | Marks the current input-audio segment complete. |

Send audio with contiguous `audio_sequence` values and monotonic
`timestamp_ms` values. `data_base64` contains Base64-encoded raw mono or
interleaved multichannel PCM S16LE bytes; the sample rate and channel count must
match the actual byte stream.

### 2.4 Server-to-client events

| `payload.type` | Fields | Meaning |
|---|---|---|
| `response.text.delta` | `response_id` UUID, `delta` String | Incremental text. |
| `response.reasoning.delta` | `response_id` UUID, `delta` String | Incremental reasoning content. |
| `response.tool_call.delta` | `response_id` UUID, non-negative `index`, optional `call_id`, `name`, and `arguments_delta` (default empty string) | Incremental tool call. |
| `response.audio.delta` | `response_id` UUID, `audio_sequence`, `timestamp_ms`, `encoding: pcm_s16le`, `sample_rate_hz`, `channels` (1–8), and `data_base64` | Generated PCM audio chunk. |
| `response.interrupted` | `response_id` UUID and `reason` | Interruption reason: `client_cancelled`, `new_input`, `session_closed`, or `runtime_error`. |
| `response.completed` | `response_id` UUID, `finish_reason`, optional `usage`, and optional `performance` | Completes the current response. |
| `session.state` | `state` and non-negative `revision` | Session-state transition. |
| `runtime.metrics` | `instance_id` UUID, `queue_depth`, `resident_bytes`, `kv_bytes`, optional `prefill_tokens_per_second`, and optional `decode_tokens_per_second` | Runtime resource and throughput data. |
| `error` | `error` (`ErrorDetail`) | Protocol or runtime error. |

`session.state` is one of `idle`, `listening`, `processing`, `speaking`,
`interrupted`, `reconnecting`, `error`, or `closed`.

`usage` contains non-negative `prompt_tokens`, `completion_tokens`, and
`total_tokens`. `performance` contains prefill-token count, TTFT,
prefill/decode/generation duration and throughput, and the effective `sampling`
parameters. `error` has the form:

```json
{
  "code": "error_code",
  "message": "human-readable message",
  "retryable": false,
  "details": {}
}
```

## 3. Unavailable legacy path

`ws://127.0.0.1:8090/api/v1/realtime` performs the authentication described
above and then immediately closes the socket:

| Close code | Reason |
|---|---|
| `1013` | `Realtime audio transport is not available` |

Code `1013` means the service is temporarily unavailable on that path. Clients
must not create a reconnect storm. Query
`GET /api/v1/runtime/realtime/capabilities` first, and use
`/api/v1/runtime/realtime?mode=audio` only when a compatible native realtime
backend is available.

## 4. Client implementation guidance

1. Create or restore an HTTP session first and retain its UUID; use the same
   `session_id` in realtime frames.
2. Prefer a bearer token in the header. Use a query token only as a fallback in
   restricted browser environments.
3. Maintain a local audio sequence and send `input_audio.commit` when an input
   speech segment ends.
4. End the current turn after `response.completed` or `response.interrupted`,
   but do not assume the WebSocket itself has closed.
5. Refresh or correct credentials after code `1008`. Back off after code `1011`
   and inspect the HTTP runtime-status endpoints before reconnecting.

Session creation, media upload, response SSE, and common errors are documented
in the [HTTP API](http.md).
