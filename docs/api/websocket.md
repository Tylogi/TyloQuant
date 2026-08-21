# 📖 MFQ Server WebSocket API 文档

MFQ Server 当前暴露两个 WebSocket 路径。用于原生实时音频的
<code>/api/v1/runtime/realtime</code> 是一个双向透明代理；历史的
<code>/api/v1/realtime</code> 已保留路由，但当前实现会明确拒绝连接。

默认服务地址：

~~~
ws://127.0.0.1:8090
~~~

这两个路径没有作为 OpenAPI <code>paths</code> 操作描述。不过，已提交的 OpenAPI 契约在
<code>x-mfq-websocket</code> 扩展中记录了旧
<code>/api/v1/realtime</code> 的 RealtimeFrame schema 和事件集合；可用的
<code>/api/v1/runtime/realtime</code> 是无类型透明代理，须以当前原生后端协议为准。
HTTP 接口、普通 SSE 生成流和作业 SSE 请参阅 <code>docs/api/http.md</code>。

## 1. 连接与认证

### 1.1 URL

| 路径 | 状态 | 用途 |
| :--- | :--- | :--- |
| <code>ws://127.0.0.1:8090/api/v1/runtime/realtime?mode=audio</code> | 可用（取决于后端） | 将浏览器或客户端的文本/二进制帧双向代理到当前原生运行时。 |
| <code>ws://127.0.0.1:8090/api/v1/realtime</code> | 当前不可用 | 认证成功后服务端以 1013 关闭。不可作为实时客户端入口。 |

生产环境使用 TLS 时，将 <code>ws</code> 替换为 <code>wss</code>。

### 1.2 认证

没有配置 API 密钥时，两个路径均无需认证。配置 API 密钥后，客户端须提供具有
<code>inference</code>（或 <code>admin</code>）scope 的凭据。优先使用 Header：

~~~
Authorization: Bearer <token>
~~~

无法设置 WebSocket Header 的浏览器客户端可使用查询参数：

~~~
ws://127.0.0.1:8090/api/v1/runtime/realtime?mode=audio&access_token=<token>
~~~

仅在 Header 未提供 Bearer token 时，服务端才读取 <code>access_token</code>。因为查询字符串
容易被日志和代理记录，应优先使用 Header 或短期凭据。

认证失败时服务端关闭连接：

| close code | reason |
| :--- | :--- |
| <code>1008</code> | <code>invalid API credential</code> |

### 1.3 mode 参数和连接失败

<code>/api/v1/runtime/realtime</code> 只接受 <code>mode=audio</code>；省略 mode 时默认值也是
<code>audio</code>。任何其他值都会在 accept 前关闭：

| close code | reason |
| :--- | :--- |
| <code>1008</code> | <code>audio mode is required</code> |

当前后端没有实时传输能力、没有已加载模型，或代理过程发生未处理错误时，服务端尝试向客户端
发送错误 JSON，然后以如下状态关闭：

| close code | reason |
| :--- | :--- |
| <code>1011</code> | <code>realtime proxy failed</code> |

客户端正常断开或任一代理方向先结束时，服务端会取消另一侧任务，并尽力以
<code>1000</code> 关闭。

## 2. 原生实时音频代理

### 2.1 行为边界

成功鉴权并连接后，MFQ Server 调用当前运行时的实时连接器并执行以下规则：

1. 客户端发送的 text WebSocket frame 原样转发给上游运行时。
2. 客户端发送的 binary WebSocket frame 原样转发给上游运行时。
3. 上游返回的 text 或 binary frame 也原样转发给客户端。
4. HTTP 服务端不解析、不补序号、不重编码，也不会校验帧的 JSON Schema。

因此，客户端和所选原生运行时必须使用相同的实时传输协议。请勿把 HTTP 的
<code>ResponseResource</code> JSON 或 SSE 数据直接写入该 socket。

### 2.2 规范化 RealtimeFrame

MFQ 协议模型定义了用于 JSON 实时消息的规范化 envelope。支持该格式的客户端应以
UTF-8 text frame 发送和接收：

~~~
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
~~~

| 外层字段 | 类型 | 必填 | 描述 |
| :--- | :--- | :---: | :--- |
| <code>protocol_version</code> | String | 否 | 固定 <code>1.0</code>；省略时默认为 <code>1.0</code>。 |
| <code>session_id</code> | UUID | **是** | 此实时会话对应的 MFQ session。 |
| <code>sequence</code> | Integer | **是** | 非负、单调递增的传输序号。 |
| <code>timestamp</code> | DateTime | **是** | 含时区的消息时间。 |
| <code>payload</code> | Object | **是** | 由 <code>type</code> 判别，见下节。 |

代理层本身不会强制上述 envelope；它描述的是 MFQ 原生实时协议的公共模型。若上游运行时
协商使用另一种帧格式，服务端仍会透明转发，客户端应以运行时实际协议为准。

### 2.3 客户端到服务端事件

| <code>payload.type</code> | 字段 | 说明 |
| :--- | :--- | :--- |
| <code>input_audio.delta</code> | <code>audio_sequence</code>（非负 Integer）、<code>timestamp_ms</code>（非负 Integer）、<code>encoding</code>（固定 <code>pcm_s16le</code>）、<code>sample_rate_hz</code>（默认 16000，至少 1）、<code>channels</code>（默认 1，1–8）、<code>data_base64</code>（Base64 bytes）。 | 追加一段 PCM S16LE 音频。 |
| <code>input_audio.commit</code> | <code>last_audio_sequence</code>（非负 Integer）。 | 表示当前输入音频段完成。 |

推荐以连续的 <code>audio_sequence</code> 和单调的 <code>timestamp_ms</code> 发送音频。
<code>data_base64</code> 是 mono 或多声道交错的 PCM S16LE 原始字节的 Base64 编码；
采样率和声道数必须与实际字节内容一致。

### 2.4 服务端到客户端事件

| <code>payload.type</code> | 字段 | 说明 |
| :--- | :--- | :--- |
| <code>response.text.delta</code> | <code>response_id</code> UUID、<code>delta</code> String。 | 增量文本。 |
| <code>response.reasoning.delta</code> | <code>response_id</code> UUID、<code>delta</code> String。 | 增量 reasoning 内容。 |
| <code>response.tool_call.delta</code> | <code>response_id</code> UUID、<code>index</code>（非负 Integer）、可选 <code>call_id</code>、<code>name</code>、<code>arguments_delta</code>（默认空字符串）。 | 增量工具调用。 |
| <code>response.audio.delta</code> | <code>response_id</code> UUID、<code>audio_sequence</code>、<code>timestamp_ms</code>、<code>encoding: pcm_s16le</code>、<code>sample_rate_hz</code>、<code>channels</code>（1–8）、<code>data_base64</code>。 | 生成的 PCM 音频块。 |
| <code>response.interrupted</code> | <code>response_id</code> UUID、<code>reason</code>。 | 中断原因：<code>client_cancelled</code>、<code>new_input</code>、<code>session_closed</code> 或 <code>runtime_error</code>。 |
| <code>response.completed</code> | <code>response_id</code> UUID、<code>finish_reason</code>、可选 <code>usage</code>、<code>performance</code>。 | 本轮响应完成。 |
| <code>session.state</code> | <code>state</code>、<code>revision</code>（非负 Integer）。 | 会话状态变化。 |
| <code>runtime.metrics</code> | <code>instance_id</code> UUID、<code>queue_depth</code>、<code>resident_bytes</code>、<code>kv_bytes</code>、可选 <code>prefill_tokens_per_second</code>/<code>decode_tokens_per_second</code>。 | 运行时资源和吞吐信息。 |
| <code>error</code> | <code>error</code>（ErrorDetail）。 | 协议或运行时错误。 |

<code>session.state</code> 的值为 <code>idle</code>、<code>listening</code>、
<code>processing</code>、<code>speaking</code>、<code>interrupted</code>、
<code>reconnecting</code>、<code>error</code> 或 <code>closed</code>。

<code>usage</code> 包含 <code>prompt_tokens</code>、<code>completion_tokens</code>、
<code>total_tokens</code>（均为非负）。<code>performance</code> 包含预填充 token 数、
TTFT、prefill/decode/generation 的毫秒数与 tokens/s，以及实际
<code>sampling</code> 参数。<code>error</code> 采用
<code>{"code": String, "message": String, "retryable": Boolean, "details": Object}</code>。

## 3. 当前不可用的旧路径

<code>ws://127.0.0.1:8090/api/v1/realtime</code> 会先执行第 1 节的认证。认证通过后，
实现立即关闭 socket：

| close code | reason |
| :--- | :--- |
| <code>1013</code> | <code>Realtime audio transport is not available</code> |

<code>1013</code> 表示服务暂时无法提供该路径的服务。客户端不得在该地址进行重连风暴；
应先检查 <code>GET /api/v1/runtime/realtime/capabilities</code>，在已有可用原生实时后端时
改连 <code>/api/v1/runtime/realtime?mode=audio</code>。

## 4. 客户端实现建议

1. 先建立或恢复 HTTP session，并保存它的 UUID；实时帧使用同一 <code>session_id</code>。
2. 使用 Header 中的 Bearer token；查询 token 只作为受限浏览器环境的后备方案。
3. 发送音频时维持本地 sequence，并在一个语音段结束后发送
   <code>input_audio.commit</code>。
4. 收到 <code>response.completed</code> 或 <code>response.interrupted</code> 后结束当前轮，
   但不要假定 WebSocket 已关闭。
5. 收到 <code>1008</code> 时刷新或修正凭据；<code>1011</code> 时按退避策略重试，并通过
   HTTP runtime 状态接口诊断原生后端。

HTTP 创建会话、媒体上传、会话响应 SSE 与错误模型的完整定义见
<code>docs/api/http.md</code>。
