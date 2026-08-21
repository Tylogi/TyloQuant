# 📖 MFQ Server HTTP API 文档

MFQ Server HTTP API 面向 MFQ Studio、脚本和 SDK。本文以当前服务端路由和已提交的
OpenAPI 契约为准；机器可读的完整契约位于
<code>mfq/server/protocol/openapi.json</code>。

默认服务地址：

~~~
http://127.0.0.1:8090
~~~

版本化 API 的根地址是 <code>http://127.0.0.1:8090/api/v1</code>。本文覆盖 53
个 HTTP 路径、70 个 HTTP 操作，以及不在 OpenAPI 中的健康检查。

## 1. 通用约定

### 1.1 认证和权限

未配置 <code>MFQ_SERVER_API_KEY</code>（或 <code>--api-key-env</code> 指定的环境变量）时，
<code>/api/</code> 下的接口不要求认证。配置后，每个请求均须带：

~~~
Authorization: Bearer <token>
~~~

根凭据拥有管理员权限。由 <code>/auth/keys</code> 创建的子密钥可以带显式 scope 或角色；
角色会额外授予下列 scope：

| 角色 | 授予的 scope |
| :--- | :--- |
| <code>viewer</code> | <code>inference</code> |
| <code>operator</code> | <code>inference</code>、<code>models</code>、<code>jobs</code> |
| <code>administrator</code> | <code>admin</code> |

| 接口范围 | 所需 scope |
| :--- | :--- |
| 会话、预设、媒体、文档，以及 MCP 的读取 | <code>inference</code> |
| <code>/models</code>、<code>/runtime</code>、<code>/hub</code> 的 GET | <code>inference</code> |
| 模型或运行时的非 GET 操作 | <code>models</code> |
| 作业、谱系、数据集和评测 | <code>jobs</code> |
| MCP 的写操作、密钥管理和集群节点 | <code>admin</code> |

<code>admin</code> 可以访问所有 scope。<code>/health</code> 不在 <code>/api/</code> 下，始终无需
Bearer 凭据。

### 1.2 类型、日期和严格校验

| 记法 | 含义 |
| :--- | :--- |
| <code>UUID</code> | 标准 UUID 字符串。所有路径中的 <code>*_id</code> 均为 UUID。 |
| <code>DateTime</code> | 含时区的 ISO 8601 日期时间。 |
| <code>SHA-256</code> | 64 个小写十六进制字符（256 bit 摘要的十六进制表示）。 |
| <code>Object</code> | JSON 对象；未特别说明的对象字段可由服务端扩展。 |
| <code>String[]</code> | JSON 字符串数组。 |

由 <code>ProtocolModel</code> 定义的*请求对象*拒绝未知字段；明确标为 Object 的字段
（如 <code>metadata</code>、<code>payload</code>）及后端透传 Object 响应例外。缺少必填字段、
类型错误或违反取值范围时返回 <code>422</code>。

### 1.3 通用错误响应

所有 API 路由都可返回下列结构。路由声明的错误状态码集合为
<code>400</code>、<code>401</code>、<code>403</code>、<code>404</code>、<code>409</code>、
<code>413</code>、<code>415</code>、<code>422</code>、<code>501</code>、<code>502</code> 和
<code>503</code>；实际出现的状态码取决于操作和运行时状态。

~~~
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
~~~

<code>401</code> 表示凭据缺失、无效、撤销或过期，并带有
<code>WWW-Authenticate: Bearer</code>；<code>403</code> 表示有效凭据缺少所需 scope。
<code>409</code> 常用于会话 revision 冲突或已有进行中的响应。<code>501</code> 表示当前配置的
服务不具备该能力，例如未启用密钥管理或实时后端。

### 1.4 结果包装与异步操作

列表响应统一采用 <code>{"data": [...]}</code>。删除成功返回 <code>204 No Content</code>。
模型加载、卸载、运行时 profile 加载、创建和重试作业会返回 <code>202 Accepted</code>；这只表示
操作已接受或已进入队列，不表示实际完成。

<code>OperationAccepted</code>：

| 字段 | 类型 | 描述 |
| :--- | :--- | :--- |
| <code>operation_id</code> | UUID | 对应后台操作或作业 ID。 |
| <code>status</code> | String | 固定为 <code>accepted</code>。 |

## 2. API 概览

### 2.1 服务、认证和会话

| 方法 | 地址 | 成功响应 | 用途 |
| :--- | :--- | :--- | :--- |
| <code>GET</code> | <code>/health</code> | 200 Object | 存活检查。 |
| <code>POST</code> | <code>/api/v1/auth/keys</code> | 201 ApiKeySecretResource | 创建受限 API 密钥。 |
| <code>GET</code> | <code>/api/v1/auth/keys</code> | 200 ApiKeyList | 列出 API 密钥。 |
| <code>POST</code> | <code>/api/v1/auth/keys/{key_id}/revoke</code> | 200 ApiKeyResource | 撤销密钥。 |
| <code>POST</code> | <code>/api/v1/auth/keys/{key_id}/rotate</code> | 200 ApiKeySecretResource | 轮换并只返回一次新明文。 |
| <code>POST</code> | <code>/api/v1/sessions</code> | 201 SessionResource | 创建持久化会话。 |
| <code>GET</code> | <code>/api/v1/sessions</code> | 200 SessionList | 分页列出会话。 |
| <code>POST</code> | <code>/api/v1/sessions/import</code> | 201 SessionImportResult | 导入可移植会话归档。 |
| <code>GET</code> | <code>/api/v1/sessions/{session_id}</code> | 200 SessionResource | 读取会话。 |
| <code>PATCH</code> | <code>/api/v1/sessions/{session_id}</code> | 200 SessionResource | 修改会话元信息。 |
| <code>DELETE</code> | <code>/api/v1/sessions/{session_id}</code> | 204 | 删除会话及其运行时状态。 |
| <code>GET</code> | <code>/api/v1/sessions/{session_id}/messages</code> | 200 MessageList | 读取消息链。 |
| <code>POST</code> | <code>/api/v1/sessions/{session_id}/messages</code> | 201 AppendMessageResult | 追加一条消息。 |
| <code>GET</code> | <code>/api/v1/sessions/{session_id}/export</code> | 200 SessionArchive | 导出会话、引用媒体和文档元数据。 |
| <code>GET</code> | <code>/api/v1/sessions/{session_id}/responses</code> | 200 ResponseList | 读取历史生成响应。 |
| <code>POST</code> | <code>/api/v1/sessions/{session_id}/responses</code> | 200 ResponseResource 或 SSE | 生成或流式生成响应。 |
| <code>POST</code> | <code>/api/v1/sessions/{session_id}/fork</code> | 201 SessionResource | 从消息位置派生分支。 |
| <code>POST</code> | <code>/api/v1/sessions/{session_id}/rewind</code> | 200 SessionResource | 回退会话消息链。 |

### 2.2 预设、媒体和 MCP

| 方法 | 地址 | 成功响应 | 用途 |
| :--- | :--- | :--- | :--- |
| <code>POST</code> / <code>GET</code> | <code>/api/v1/presets</code> | 201 GenerationPresetResource / 200 GenerationPresetList | 创建或列出生成预设。 |
| <code>PUT</code> / <code>DELETE</code> | <code>/api/v1/presets/{preset_id}</code> | 200 GenerationPresetResource / 204 | 整体更新或删除预设。 |
| <code>POST</code> | <code>/api/v1/media</code> | 201 MediaResource | 上传不可变原始媒体。 |
| <code>GET</code> | <code>/api/v1/media/{media_id}</code> | 200 二进制内容 | 取得媒体原始字节。 |
| <code>POST</code> | <code>/api/v1/documents</code> | 201 DocumentResource | 从已上传媒体提取文档文本。 |
| <code>GET</code> | <code>/api/v1/documents/{media_id}</code> | 200 DocumentResource | 读取已提取的文档。 |
| <code>POST</code> / <code>GET</code> | <code>/api/v1/mcp/servers</code> | 201 McpServerResource / 200 McpServerList | 注册或列出 MCP 服务器。 |
| <code>PATCH</code> / <code>DELETE</code> | <code>/api/v1/mcp/servers/{server_id}</code> | 200 McpServerResource / 204 | 启停或删除 MCP 服务器。 |
| <code>GET</code> | <code>/api/v1/mcp/tools</code> | 200 McpToolList | 刷新并列出可用 MCP 工具。 |
| <code>POST</code> | <code>/api/v1/mcp/tools/call</code> | 200 McpToolCallResult | 显式确认后调用 MCP 工具。 |

### 2.3 作业、模型、数据和集群

| 方法 | 地址 | 成功响应 | 用途 |
| :--- | :--- | :--- | :--- |
| <code>POST</code> / <code>GET</code> | <code>/api/v1/jobs</code> | 202 JobResource / 200 JobList | 创建或过滤后台作业。 |
| <code>GET</code> | <code>/api/v1/jobs/kinds</code> | 200 JobKindList | 获取当前注册作业种类及 payload JSON Schema。 |
| <code>GET</code> | <code>/api/v1/jobs/{job_id}</code> | 200 JobResource | 查询作业。 |
| <code>POST</code> | <code>/api/v1/jobs/{job_id}/cancel</code> | 200 JobResource | 请求取消作业。 |
| <code>POST</code> | <code>/api/v1/jobs/{job_id}/retry</code> | 202 JobResource | 按原 payload 重试作业。 |
| <code>GET</code> | <code>/api/v1/jobs/{job_id}/events</code> | 200 JobEventList | 增量读取作业事件。 |
| <code>GET</code> | <code>/api/v1/jobs/{job_id}/events/stream</code> | 200 SSE | 订阅作业事件。 |
| <code>GET</code> | <code>/api/v1/models</code> | 200 ModelArtifactList | 发现本地 MFQ 模型产物。 |
| <code>POST</code> | <code>/api/v1/models/load</code> | 202 OperationAccepted | 异步加载模型。 |
| <code>POST</code> | <code>/api/v1/models/unload</code> | 202 OperationAccepted | 异步卸载实例。 |
| <code>GET</code> | <code>/api/v1/hub/models</code> | 200 HubModelSearchResult | 搜索 Hugging Face 或 ModelScope。 |
| <code>GET</code> | <code>/api/v1/hub/models/{provider}/{owner}/{name}</code> | 200 HubModelInfo | 读取模型仓库详情。 |
| <code>GET</code> | <code>/api/v1/artifacts/lineage</code> | 200 ArtifactLineageList | 查询工作区产物谱系。 |
| <code>POST</code> | <code>/api/v1/artifacts/remove</code> | 200 Object | 删除受限工作区 URI 指向的产物。 |
| <code>POST</code> / <code>GET</code> | <code>/api/v1/datasets</code> | 201 DatasetResource / 200 DatasetList | 注册或列出评测数据集。 |
| <code>DELETE</code> | <code>/api/v1/datasets/{dataset_id}</code> | 204 | 删除数据集记录。 |
| <code>GET</code> | <code>/api/v1/evaluations</code> | 200 EvaluationResultList | 查询已记录评测。 |
| <code>POST</code> | <code>/api/v1/evaluations/compare</code> | 200 EvaluationComparisonResource | 对比可比评测。 |
| <code>POST</code> / <code>GET</code> | <code>/api/v1/cluster/nodes</code> | 201 RemoteNodeResource / 200 RemoteNodeList | 注册或列出远程 MFQ 节点。 |
| <code>PUT</code> / <code>DELETE</code> | <code>/api/v1/cluster/nodes/{node_id}</code> | 200 RemoteNodeResource / 204 | 整体更新或删除远程节点。 |

### 2.4 运行时

| 方法 | 地址 | 成功响应 | 用途 |
| :--- | :--- | :--- | :--- |
| <code>POST</code> / <code>GET</code> | <code>/api/v1/runtime/profiles</code> | 201 RuntimeProfileResource / 200 RuntimeProfileList | 创建或列出运行时 profile。 |
| <code>GET</code> / <code>PUT</code> / <code>DELETE</code> | <code>/api/v1/runtime/profiles/{profile_id}</code> | 200 RuntimeProfileResource / 204 | 读取、整体更新或删除 profile。 |
| <code>POST</code> | <code>/api/v1/runtime/profiles/{profile_id}/load</code> | 202 OperationAccepted | 按 profile 异步加载。 |
| <code>GET</code> | <code>/api/v1/runtime/instances</code> | 200 RuntimeInstanceList | 列出托管运行时实例。 |
| <code>GET</code> | <code>/api/v1/runtime/capabilities</code> | 200 RuntimeCapabilitiesResource | 获取当前模型能力。 |
| <code>GET</code> | <code>/api/v1/runtime/status</code> | 200 Object | 获取当前原生运行时状态快照。 |
| <code>GET</code> | <code>/api/v1/runtime/metrics</code> | 200 RuntimeMetricList | 查询持久化指标快照。 |
| <code>GET</code> | <code>/api/v1/runtime/logs</code> | 200 RuntimeLogList | 查询持久化日志。 |
| <code>GET</code> | <code>/api/v1/runtime/models</code> | 200 Object | 列出后端可用模型。 |
| <code>GET</code> | <code>/api/v1/runtime/realtime/capabilities</code> | 200 Object | 查询原生实时能力。 |
| <code>POST</code> | <code>/api/v1/runtime/reload</code> | 200 Object | 用新的 context size 重载当前运行时。 |
| <code>POST</code> | <code>/api/v1/runtime/cache/clear</code> | 200 Object | 清除当前运行时的前缀/KV 缓存。 |

## 3. 会话与响应

### 3.1 创建、读取和修改会话

**接口描述**：创建会话。<br>
**请求方式**：<code>POST</code><br>
**请求地址**：<code>/api/v1/sessions</code><br>
**成功响应**：<code>201 SessionResource</code>

| Body 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>model</code> | String | **是** | - | 模型标识，1–255 个字符。 |
| <code>mode</code> | String | 否 | <code>text</code> | <code>text</code>、<code>voice</code> 或 <code>full_duplex</code>。 |
| <code>title</code> | String / null | 否 | <code>null</code> | 标题，最多 512 个字符。 |
| <code>metadata</code> | Object | 否 | <code>{}</code> | 应用自定义元数据。 |

最小请求：

~~~
{
  "model": "my-model"
}
~~~

**列出会话**：<code>GET /api/v1/sessions</code>，查询参数 <code>limit</code> 为 1–200，
默认 50；<code>offset</code> 最小为 0，默认 0。返回 <code>SessionList</code>。

**读取、修改、删除会话**：

| 方法 | 地址 | 请求 Body | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>GET</code> | <code>/api/v1/sessions/{session_id}</code> | 无 | 200 SessionResource |
| <code>PATCH</code> | <code>/api/v1/sessions/{session_id}</code> | UpdateSessionRequest | 200 SessionResource |
| <code>DELETE</code> | <code>/api/v1/sessions/{session_id}</code> | 无 | 204 |

<code>UpdateSessionRequest</code> 必须至少提供一个字段：

| 字段 | 类型 | 限制 |
| :--- | :--- | :--- |
| <code>title</code> | String / null | 最多 512 个字符；显式传 <code>null</code> 可清空标题。 |
| <code>mode</code> | String | <code>text</code>、<code>voice</code> 或 <code>full_duplex</code>；不得为 <code>null</code>。 |
| <code>metadata</code> | Object | 不得为 <code>null</code>。 |

### 3.2 消息、分支、回退和归档

| 方法 | 地址 | 请求 | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>GET</code> | <code>/api/v1/sessions/{session_id}/messages</code> | 无 | 200 MessageList |
| <code>POST</code> | <code>/api/v1/sessions/{session_id}/messages</code> | AppendMessageRequest | 201 AppendMessageResult |
| <code>POST</code> | <code>/api/v1/sessions/{session_id}/fork</code> | ForkSessionRequest | 201 SessionResource |
| <code>POST</code> | <code>/api/v1/sessions/{session_id}/rewind</code> | RewindSessionRequest | 200 SessionResource |
| <code>GET</code> | <code>/api/v1/sessions/{session_id}/export</code> | 无 | 200 SessionArchive |
| <code>POST</code> | <code>/api/v1/sessions/import</code> | SessionArchive | 201 SessionImportResult |

<code>AppendMessageRequest</code>：

| 字段 | 类型 | 必填 | 描述 |
| :--- | :--- | :---: | :--- |
| <code>expected_revision</code> | Integer | **是** | 当前会话 revision，最小 0；过期值返回 <code>409</code>。 |
| <code>role</code> | String | **是** | <code>system</code>、<code>user</code>、<code>assistant</code> 或 <code>tool</code>。 |
| <code>parts</code> | ContentPart[] | **是** | 至少一个内容分片，定义见 3.4。 |

<code>ForkSessionRequest</code>：

| 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>at_message_id</code> | UUID / null | 否 | <code>null</code> | 分叉位置；省略时使用当前末尾。 |
| <code>include_message</code> | Boolean | 否 | <code>true</code> | 是否把指定消息留在新分支中。 |
| <code>title</code> | String / null | 否 | <code>null</code> | 新分支标题，最多 512 字符。 |

<code>RewindSessionRequest</code>：

| 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>expected_revision</code> | Integer | **是** | - | 当前 revision，最小 0。 |
| <code>at_message_id</code> | UUID | **是** | - | 回退目标消息。 |
| <code>include_message</code> | Boolean | 否 | <code>true</code> | 是否保留目标消息。 |

导出格式为 <code>SessionArchive</code>，其 <code>format</code> 固定为
<code>mfq-session-v1</code>；<code>session</code> 是 SessionResource，<code>messages</code>
是含 <code>role</code>、<code>parts</code>、<code>created_at</code> 的消息数组，
<code>media</code> 是可选的 <code>sha256</code>、<code>mime_type</code>、Base64
<code>data_base64</code> 和可选 <code>document</code> 数组。导入会创建新会话、验证每份媒体摘要并
重新映射标识，绝不会覆盖源会话。

### 3.3 创建生成响应

**接口描述**：在会话内追加输入并生成 assistant 响应；使用同一 <code>request_id</code> 可避免
网络重试造成重复生成。<br>
**请求方式**：<code>POST</code><br>
**请求地址**：<code>/api/v1/sessions/{session_id}/responses</code>

| Body 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>request_id</code> | UUID | **是** | - | 客户端生成的幂等请求 ID。 |
| <code>expected_revision</code> | Integer | **是** | - | 当前 session revision，最小 0。 |
| <code>input</code> | ContentPart[] | **是** | - | 至少一个输入分片。 |
| <code>input_role</code> | String | 否 | <code>user</code> | <code>user</code> 或 <code>tool</code>。 |
| <code>sampling</code> | SamplingParams | 否 | 见下表 | 采样参数。 |
| <code>system_prompt</code> | String / null | 否 | <code>null</code> | 最大 32768 个字符。 |
| <code>include_reasoning_history</code> | Boolean | 否 | <code>true</code> | 是否向模型提供历史 reasoning 内容。 |
| <code>tools</code> | ToolDefinition[] | 否 | <code>[]</code> | 最多 128 个函数工具。 |
| <code>tool_choice</code> | String / Object | 否 | <code>auto</code> | <code>auto</code>、<code>none</code>、<code>required</code>，或指定函数对象。 |
| <code>response_format</code> | Object | 否 | <code>{"type":"text"}</code> | 文本、JSON 对象或 JSON Schema 输出。 |
| <code>stream</code> | Boolean | 否 | <code>true</code> | 为真时返回 <code>text/event-stream</code>；否则返回 JSON。 |

<code>input_role=tool</code> 时，<code>input</code> 必须恰有一个
<code>tool_result</code> 分片；普通 user 输入不得包含 <code>tool_result</code>。
工具名称不得重复；指定函数的 <code>tool_choice</code> 必须对应 <code>tools</code> 中已有名称；
<code>required</code> 至少需要一个工具。

#### 幂等、冲突与安全重试

<code>request_id</code> 的幂等范围是同一个 <code>session_id</code>。服务端比较除
<code>stream</code> 外的完整请求指纹：

| 场景 | 服务端行为 | 客户端处理 |
| :--- | :--- | :--- |
| 相同 <code>request_id</code>、相同指纹、先前已完成 | <code>stream=false</code> 直接返回原 ResponseResource；<code>stream=true</code> 根据已保存的最终输出重新构造 reasoning/text/tool-call 增量事件和最终 <code>session.state</code>，不再次推理。原始 chunk 边界不保证保留。 | 可安全重试。 |
| 相同 <code>request_id</code>、不同指纹 | <code>409 idempotency_conflict</code>。 | 生成新的 request ID，或恢复原请求内容。 |
| 相同 <code>request_id</code>、原请求仍在执行 | <code>409 response_in_progress</code>，<code>retryable: true</code>。 | 使用退避重试或查询响应历史；不要并行再次生成。 |
| <code>expected_revision</code> 过期 | <code>409 revision_conflict</code>，<code>details</code> 含 expected/actual revision。 | 重新读取会话并基于新 revision 发起请求。 |

<code>stream=false</code> 时后端失败可返回 HTTP <code>502</code>。<code>stream=true</code> 的
实现会先发送 <code>session.state</code>，随后后端失败将以 SSE <code>error</code> 事件和最后的
<code>session.state</code> 结束；它不会再转换为 HTTP <code>502</code>，且不应期待
<code>response.completed</code>。

<code>SamplingParams</code>：

| 字段 | 类型 | 默认值 | 限制 |
| :--- | :--- | :--- | :--- |
| <code>max_tokens</code> | Integer | 4096 | 至少 1。 |
| <code>temperature</code> | Number | 1.0 | 不小于 0。 |
| <code>top_k</code> | Integer | 20 | 不小于 0。 |
| <code>top_p</code> | Number | 0.95 | 区间 (0, 1]。 |
| <code>presence_penalty</code> | Number | 0.0 | 区间 [-2, 2]。 |
| <code>frequency_penalty</code> | Number | 0.0 | 区间 [-2, 2]。 |
| <code>repetition_penalty</code> | Number | 1.0 | 大于 0。 |
| <code>seed</code> | Integer / null | <code>null</code> | 不小于 0。 |
| <code>enable_thinking</code> | Boolean | <code>true</code> | 是否启用模型 reasoning。 |
| <code>reasoning_effort</code> | String / null | <code>null</code> | 1–32 个字符。 |

非流式请求返回 <code>200 ResponseResource</code>。流式请求返回
<code>200 text/event-stream</code>，其帧格式和事件类型见本文件第 11 节。历史响应可由
<code>GET /api/v1/sessions/{session_id}/responses?limit=200</code> 读取，<code>limit</code>
范围为 1–1000，默认 200。

一个带认证的最小流式调用如下。SSE 连接本身不接受 <code>Last-Event-ID</code> 续传；如连接中断，
请用*相同*请求体和 <code>request_id</code> 再次 POST，以利用已完成响应的重放语义。

~~~
curl -N -X POST http://127.0.0.1:8090/api/v1/sessions/<session-id>/responses \
  -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  --data '{
    "request_id": "00000000-0000-0000-0000-000000000001",
    "expected_revision": 0,
    "input": [{"type": "text", "text": "你好"}],
    "stream": true
  }'
~~~

### 3.4 ContentPart、工具和结构化输出

<code>ContentPart</code> 由 <code>type</code> 判别：

| <code>type</code> | 附加字段 |
| :--- | :--- |
| <code>text</code>、<code>reasoning</code> | <code>text</code>: String。 |
| <code>image</code> | <code>media</code>: MediaRef；可选 <code>width</code>、<code>height</code>，均至少 1。 |
| <code>video</code> | <code>media</code>；可选 <code>width</code>、<code>height</code>（至少 1）、<code>duration_ms</code>（至少 0）。 |
| <code>audio</code>、<code>generated_audio</code> | <code>media</code>、<code>sample_rate_hz</code>（至少 1）、<code>channels</code>（1–8）、可选 <code>duration_ms</code>。 |
| <code>transcript</code> | <code>text</code>，可选 <code>language</code>（2–35 字符）、<code>start_ms</code>、<code>end_ms</code>（均至少 0）。 |
| <code>document</code> | <code>media</code>、<code>name</code>（1–512 字符）。 |
| <code>tool_call</code> | <code>call_id</code>、<code>name</code>（均 1–255 字符）、<code>arguments</code> Object，默认为 <code>{}</code>。 |
| <code>tool_result</code> | <code>call_id</code>（1–255 字符）、<code>result</code>（任意 JSON）、<code>is_error</code>（默认 <code>false</code>）。 |

<code>MediaRef</code> 固定包含 <code>id</code>、<code>sha256</code>、<code>mime_type</code> 和
非负 <code>byte_size</code>。

#### 角色与媒体引用校验

| 规则 | 违反时的错误 |
| :--- | :--- |
| <code>reasoning</code> 与 <code>tool_call</code> 只能位于 <code>assistant</code> 消息。 | <code>422 unsupported_reasoning_part</code> 或 <code>422 unsupported_tool_call_part</code>。 |
| <code>tool</code> 消息必须仅有一个 <code>tool_result</code>；非 tool 消息不得含 <code>tool_result</code>。 | <code>422 unsupported_tool_message</code> 或 <code>422 unsupported_tool_result_part</code>。 |
| image/video/audio/generated_audio 必须引用已上传、四字段完全匹配的 MediaRef，且 MIME 前缀分别为 <code>image/</code>、<code>video/</code>、<code>audio/</code>、<code>audio/</code>。 | <code>422 media_not_found</code>、<code>media_reference_mismatch</code> 或 <code>media_type_mismatch</code>。 |
| document 必须引用已提取的文档，且 MediaRef 和 <code>name</code> 与存储记录一致。 | <code>422 document_not_found</code> 或 <code>document_reference_mismatch</code>。 |

一个函数工具采用：

~~~
{
  "type": "function",
  "function": {
    "name": "lookup_weather",
    "description": "可选，最多 8192 个字符",
    "parameters": { "type": "object" }
  }
}
~~~

函数名必须匹配 <code>[A-Za-z_][A-Za-z0-9_.-]{0,127}</code>。
命名 <code>tool_choice</code> 为
<code>{"type":"function","function":{"name":"lookup_weather"}}</code>。
<code>response_format</code> 支持 <code>{"type":"text"}</code>、
<code>{"type":"json_object"}</code>，或
<code>{"type":"json_schema","json_schema":{"name":"result","schema":{},"strict":true}}</code>；
JSON Schema 名称使用同一函数名规则，<code>description</code> 可选。

## 4. 预设、媒体和文档

### 4.1 生成预设

| 方法 | 地址 | Body | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>POST</code> | <code>/api/v1/presets</code> | CreateGenerationPresetRequest | 201 GenerationPresetResource |
| <code>GET</code> | <code>/api/v1/presets</code> | 无 | 200 GenerationPresetList |
| <code>PUT</code> | <code>/api/v1/presets/{preset_id}</code> | UpdateGenerationPresetRequest | 200 GenerationPresetResource |
| <code>DELETE</code> | <code>/api/v1/presets/{preset_id}</code> | 无 | 204 |

创建和更新使用完全相同的字段，<code>PUT</code> 不是局部 patch：

| 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>name</code> | String | **是** | - | 1–64 字符。 |
| <code>model</code> | String / null | 否 | <code>null</code> | 1–255 字符。 |
| <code>mode</code> | String / null | 否 | <code>null</code> | 会话模式。 |
| <code>settings</code> | ResponseRequestSettings | **是** | - | 与生成请求相同的 <code>sampling</code>、<code>system_prompt</code>、<code>include_reasoning_history</code>、<code>input_role</code>、<code>tools</code>、<code>tool_choice</code>、<code>response_format</code>；其中 <code>sampling</code> 必填。 |
| <code>context_size</code> | Integer | 否 | 32768 | 至少 512。 |
| <code>metadata</code> | Object | 否 | <code>{}</code> | 自定义元数据。 |

### 4.2 媒体上传和读取

**上传接口**：<code>POST /api/v1/media</code><br>
**成功响应**：<code>201 MediaResource</code>

| 位置 | 参数 | 类型 | 必填 | 描述 |
| :--- | :--- | :--- | :---: | :--- |
| Header | <code>Content-Type</code> | String | **是** | 原始字节的实际 MIME 类型。允许 <code>image/</code>、<code>audio/</code>、<code>video/</code>、<code>text/</code> 前缀，以及 <code>application/json</code>、<code>application/pdf</code>、<code>application/xml</code>、<code>application/yaml</code>、<code>application/octet-stream</code>、<code>application/x-mfq-imatrix</code> 和 DOCX MIME 类型。参数（如 charset）会被忽略后规范化。 |
| Header | <code>X-Content-SHA256</code> | SHA-256 | **是** | Body 原始字节的精确摘要。 |
| Body | 原始 bytes | 与 Content-Type 相同 | **是** | 不使用 JSON 或 multipart；最大 536870912 bytes（512 MiB），且不能为空。 |

示例：

~~~
curl -X POST http://127.0.0.1:8090/api/v1/media \
  -H 'Content-Type: image/png' \
  -H "X-Content-SHA256: $(shasum -a 256 image.png | awk '{print $1}')" \
  --data-binary @image.png
~~~

响应中的 <code>MediaResource</code> 为
<code>{"media": MediaRef, "created_at": DateTime}</code>。
<code>GET /api/v1/media/{media_id}</code> 返回原始字节及其原 MIME 类型，带
<code>ETag</code>（SHA-256）、<code>Cache-Control: private, immutable</code> 和
<code>X-Content-Type-Options: nosniff</code>。

上传常见异常：

| HTTP 状态 | <code>error.code</code> | 触发条件 |
| :--- | :--- | :--- |
| 400 | <code>empty_media</code> | Body 没有字节。 |
| 413 | <code>media_too_large</code> | Body 超过 512 MiB。 |
| 415 | <code>unsupported_media_type</code> | 规范化后的 Content-Type 不在支持范围内。 |
| 422 | <code>media_digest_mismatch</code> | Header 中的 SHA-256 与上传字节不匹配。 |

### 4.3 文档提取

| 方法 | 地址 | Body / 参数 | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>POST</code> | <code>/api/v1/documents</code> | <code>media_id</code>（UUID，必填）、<code>name</code>（String，1–512 字符，必填） | 201 DocumentResource |
| <code>GET</code> | <code>/api/v1/documents/{media_id}</code> | 路径 UUID | 200 DocumentResource |

<code>DocumentResource</code> 包含 <code>media</code>、<code>name</code>、已提取的
<code>text</code>、可选 <code>page_count</code>（至少 1）、<code>extractor</code> 和
<code>created_at</code>。

## 5. MCP

### 5.1 MCP 服务管理

| 方法 | 地址 | Body | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>POST</code> | <code>/api/v1/mcp/servers</code> | CreateMcpServerRequest | 201 McpServerResource |
| <code>GET</code> | <code>/api/v1/mcp/servers</code> | 无 | 200 McpServerList |
| <code>PATCH</code> | <code>/api/v1/mcp/servers/{server_id}</code> | <code>{"enabled": Boolean}</code> | 200 McpServerResource |
| <code>DELETE</code> | <code>/api/v1/mcp/servers/{server_id}</code> | 无 | 204 |

<code>CreateMcpServerRequest</code>：

| 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>name</code> | String | **是** | - | <code>[A-Za-z_][A-Za-z0-9_.-]{0,63}</code>。 |
| <code>transport</code> | String | **是** | - | <code>stdio</code> 或 <code>streamable_http</code>。 |
| <code>enabled</code> | Boolean | 否 | <code>false</code> | 新服务器默认禁用。 |
| <code>url</code> | String / null | 条件必填 | <code>null</code> | <code>streamable_http</code> 时必须为 HTTP(S) URL，最多 2048 字符。 |
| <code>command</code> | String / null | 条件必填 | <code>null</code> | <code>stdio</code> 时必须提供，最多 1024 字符。 |
| <code>args</code> | String[] | 否 | <code>[]</code> | 最多 128 项；HTTP transport 时必须为空。 |
| <code>header_env</code> | Object | 否 | <code>{}</code> | HTTP header 到环境变量名的映射；stdio 时必须为空。不会存储秘密值。 |
| <code>timeout_seconds</code> | Number | 否 | 30.0 | 区间 (0, 300]。 |

### 5.2 发现和调用 MCP 工具

| 方法 | 地址 | Body | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>GET</code> | <code>/api/v1/mcp/tools</code> | 无 | 200 McpToolList |
| <code>POST</code> | <code>/api/v1/mcp/tools/call</code> | McpToolCallRequest | 200 McpToolCallResult |

<code>McpToolCallRequest</code> 包含必填 <code>name</code>（1–255 字符）、
可选 <code>arguments</code> Object（默认 <code>{}</code>）和 <code>confirm</code> Boolean
（默认 <code>false</code>）。服务端要求 <code>confirm=true</code> 才会执行调用。
工具列表每项有 <code>server_id</code>、<code>server</code>、<code>name</code>、
<code>qualified_name</code>、可选 <code>description</code>、<code>input_schema</code>；
无法连接的服务以 <code>errors</code> 映射单独报告。

MCP 常见异常：

| HTTP 状态 | <code>error.code</code> | 触发条件 |
| :--- | :--- | :--- |
| 409 | <code>mcp_server_conflict</code> | 创建同名或冲突的 MCP 服务器。 |
| 409 | <code>tool_confirmation_required</code> | 工具调用未明确传 <code>confirm: true</code>。 |
| 404 | <code>mcp_server_not_found</code> | 更新、删除或调用时目标服务器不存在或未启用。 |
| 404 | <code>mcp_tool_not_found</code> | 已连接服务器不提供指定工具。 |
| 422 | <code>invalid_tool_name</code> | <code>name</code> 不是 <code>server.tool</code> 形式。 |
| 502 | <code>mcp_tool_failed</code> | 连接、发现或执行 MCP 工具失败。 |

## 6. 作业、模型和产物

### 6.1 作业

**创建作业**：<code>POST /api/v1/jobs</code>，返回 <code>202 JobResource</code>。

| Body 字段 | 类型 | 必填 | 描述 |
| :--- | :--- | :---: | :--- |
| <code>kind</code> | String | **是** | 匹配 <code>[a-z][a-z0-9_.-]{0,63}</code> 的已注册作业种类。 |
| <code>payload</code> | Object | 否 | 种类专属参数，默认 <code>{}</code>。 |

先调用 <code>GET /api/v1/jobs/kinds</code> 获取*当前运行实例*可用的 <code>kind</code> 与其
<code>payload_schema</code>；不要假定所有部署注册了相同的工具作业。托管运行时至少会注册
<code>model.load</code> 和 <code>model.unload</code>。未知种类返回
<code>422 job_kind_unavailable</code>。

| 方法 | 地址 | 查询参数 | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>GET</code> | <code>/api/v1/jobs</code> | <code>status</code> 可选；<code>kind</code> 可选且匹配 kind 规则；<code>limit</code> 1–200，默认 50；<code>offset</code> 最小 0，默认 0。 | 200 JobList |
| <code>GET</code> | <code>/api/v1/jobs/{job_id}</code> | 无 | 200 JobResource |
| <code>POST</code> | <code>/api/v1/jobs/{job_id}/cancel</code> | 无 | 200 JobResource |
| <code>POST</code> | <code>/api/v1/jobs/{job_id}/retry</code> | 无 | 202 JobResource |
| <code>GET</code> | <code>/api/v1/jobs/{job_id}/events</code> | <code>after</code> 最小 0，默认 0；<code>limit</code> 1–1000，默认 200。 | 200 JobEventList |

<code>JobResource</code> 包含 <code>id</code>、<code>kind</code>、状态
<code>queued</code>/<code>running</code>/<code>cancelling</code>/<code>succeeded</code>/
<code>failed</code>/<code>cancelled</code>/<code>interrupted</code>、<code>payload</code>、
0–1 的 <code>progress</code>、<code>cancel_requested</code>、可选 <code>result</code>/
<code>error</code>，以及 <code>created_at</code>、<code>updated_at</code>、可选
<code>started_at</code>/<code>completed_at</code>。

作业常见异常：

| HTTP 状态 | <code>error.code</code> | 触发条件 |
| :--- | :--- | :--- |
| 404 | <code>job_not_found</code> | 查询、取消、重试或订阅未知作业。 |
| 409 | <code>job_state_conflict</code> | 当前作业状态不允许重试。 |
| 422 | <code>job_kind_unavailable</code> | <code>kind</code> 不在该服务实例的已注册种类中。 |

### 6.2 本地模型、模型仓库和工作区产物

| 方法 | 地址 | 参数 / Body | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>GET</code> | <code>/api/v1/models</code> | <code>refresh</code> Boolean，默认 <code>false</code>。 | 200 ModelArtifactList |
| <code>POST</code> | <code>/api/v1/models/load</code> | ModelLoadRequest | 202 OperationAccepted |
| <code>POST</code> | <code>/api/v1/models/unload</code> | <code>instance_id</code> UUID（必填）、<code>force</code> Boolean（默认 <code>false</code>）。 | 202 OperationAccepted |
| <code>GET</code> | <code>/api/v1/hub/models</code> | <code>provider</code> 必填：<code>huggingface</code> 或 <code>modelscope</code>；<code>query</code> 必填，1–255 字符；<code>limit</code> 1–100，默认 20。 | 200 HubModelSearchResult |
| <code>GET</code> | <code>/api/v1/hub/models/{provider}/{owner}/{name}</code> | 路径 provider、owner、name；<code>provider</code> 仅允许 <code>huggingface</code> 或 <code>modelscope</code>；可选 <code>revision</code>，最多 255 字符。 | 200 HubModelInfo |
| <code>GET</code> | <code>/api/v1/artifacts/lineage</code> | 可选 <code>artifact_uri</code>，最多 2048 字符；<code>limit</code> 1–1000，默认 200。 | 200 ArtifactLineageList |
| <code>POST</code> | <code>/api/v1/artifacts/remove</code> | <code>{"artifact_uri":"workspace://..."}</code>；URI 必须匹配工作区 URI 格式。 | 200 Object |

<code>ModelLoadRequest</code>：

| 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>model</code> | String | **是** | - | 1–255 字符的模型名。 |
| <code>artifact_uri</code> | String / null | 否 | <code>null</code> | 显式产物 URI，非空。 |
| <code>device_ids</code> | String[] | 否 | <code>[]</code> | 指定设备。 |
| <code>idle_ttl_seconds</code> | Integer / null | 否 | <code>null</code> | 空闲卸载时间，非负。 |
| <code>pin</code> | Boolean | 否 | <code>false</code> | 是否固定实例。 |
| <code>context_size</code> | Integer | 否 | 32768 | 至少 512。 |
| <code>prefill_chunk_size</code> | Integer | 否 | 2048 | 至少 1。 |
| <code>moe_gpu_cache_gb</code> | Number / null | 否 | <code>null</code> | 非负。 |
| <code>enable_duplex</code> | Boolean | 否 | <code>true</code> | 请求全双工能力。 |
| <code>prefix_cache_max_sessions</code> | Integer / null | 否 | <code>null</code> | 非负。 |
| <code>prefix_cache_max_snapshots_per_session</code> | Integer / null | 否 | <code>null</code> | 非负。 |
| <code>prefix_cache_max_bytes</code> | Integer / null | 否 | <code>null</code> | 非负。 |
| <code>sampling_defaults</code> | SamplingParams / null | 否 | <code>null</code> | 实例默认采样配置。 |

本地模型列表元素 <code>ModelArtifactResource</code> 含 <code>id</code>（32 位小写十六进制）、
<code>name</code>、<code>architecture</code>、固定 <code>format: "mfq"</code>、
<code>shard_count</code>、<code>total_bytes</code>、<code>tensor_count</code>、
<code>record_count</code>、<code>dtypes</code>、<code>complete</code>、<code>loadable</code>、
<code>modified_at</code> 和可选 <code>error</code>。Hub 搜索元素含 <code>provider</code>、
<code>repo_id</code>、<code>downloads</code>、<code>likes</code>、<code>total_bytes</code>、
可选 <code>updated_at</code>；详情额外有 <code>revision</code>、<code>files</code>
（<code>name</code>、<code>byte_size</code>）和 <code>tags</code>。

谱系元素包含 <code>id</code>、<code>artifact_uri</code>、<code>artifact_name</code>、
<code>producer_job_id</code>、<code>producer_kind</code>、<code>source_uris</code>、
<code>parameters</code>、<code>metadata</code>、<code>validation_job_ids</code> 和
<code>created_at</code>。删除接口仅接受受限 <code>workspace://</code> URI，不会删除任意
主机路径或模型根目录。

## 7. 数据集、评测和远程节点

### 7.1 数据集和评测

| 方法 | 地址 | 参数 / Body | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>POST</code> | <code>/api/v1/datasets</code> | CreateDatasetRequest | 201 DatasetResource |
| <code>GET</code> | <code>/api/v1/datasets</code> | 无 | 200 DatasetList |
| <code>DELETE</code> | <code>/api/v1/datasets/{dataset_id}</code> | 无 | 204 |
| <code>GET</code> | <code>/api/v1/evaluations</code> | 可选 <code>kind</code>：<code>perplexity</code> 或 <code>kernel_benchmark</code>；可选 <code>model_id</code>，最多 255 字符；<code>limit</code> 1–2000，默认 200。 | 200 EvaluationResultList |
| <code>POST</code> | <code>/api/v1/evaluations/compare</code> | <code>evaluation_ids</code>：2–16 个互异 UUID。 | 200 EvaluationComparisonResource |

<code>CreateDatasetRequest</code>：

| 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>name</code> | String | **是** | - | 1–128 字符。 |
| <code>kind</code> | String | 否 | <code>custom</code> | <code>wikitext2</code> 或 <code>custom</code>。 |
| <code>artifact_uri</code> | String | **是** | - | <code>workspace://</code> URI。 |
| <code>source_uri</code> | String / null | 否 | <code>null</code> | 最多 2048 字符。 |
| <code>revision</code> | String / null | 否 | <code>null</code> | 最多 255 字符。 |
| <code>metadata</code> | Object | 否 | <code>{}</code> | 自定义元数据。 |

<code>DatasetResource</code> 含 <code>id</code>、<code>name</code>、<code>kind</code>、
<code>artifact_uri</code>、<code>sha256</code>、<code>byte_size</code>、可选
<code>source_uri</code>/<code>revision</code>、<code>metadata</code>、<code>created_at</code>、
<code>updated_at</code>。评测列表元素包含 <code>id</code>、<code>job_id</code>、<code>kind</code>、
<code>model_id</code>、<code>metrics</code>、<code>parameters</code>、可选
<code>dataset_id</code>、<code>dataset_manifest</code>、<code>hardware_identity</code>、
<code>runtime_identity</code>、<code>comparison_key</code>（SHA-256）和 <code>created_at</code>。

对比成功时返回相同 <code>comparison_key</code>、<code>baseline_id</code>、参与指标名
<code>metrics</code> 和 <code>rows</code>；每个 row 含原始 <code>evaluation</code>、
<code>deltas</code> 和 <code>ratios</code>。不同评测 kind 或 comparison key 不能对比。

### 7.2 远程节点

| 方法 | 地址 | Body / 参数 | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>POST</code> | <code>/api/v1/cluster/nodes</code> | CreateRemoteNodeRequest | 201 RemoteNodeResource |
| <code>GET</code> | <code>/api/v1/cluster/nodes</code> | <code>refresh</code> Boolean，默认 <code>false</code>。 | 200 RemoteNodeList |
| <code>PUT</code> | <code>/api/v1/cluster/nodes/{node_id}</code> | UpdateRemoteNodeRequest | 200 RemoteNodeResource |
| <code>DELETE</code> | <code>/api/v1/cluster/nodes/{node_id}</code> | 无 | 204 |

创建和更新使用相同完整 body：

| 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>name</code> | String | **是** | - | <code>[A-Za-z_][A-Za-z0-9_.-]{0,63}</code>。 |
| <code>url</code> | String | **是** | - | HTTP 或 HTTPS URL，不能带用户信息。 |
| <code>api_key_env</code> | String / null | 否 | <code>null</code> | 远程凭据所在环境变量名；不会持久化值。 |
| <code>enabled</code> | Boolean | 否 | <code>true</code> | 是否允许路由到该节点。 |

响应还包含 <code>id</code>、<code>healthy</code>、<code>models</code>、
<code>active_requests</code>、<code>metrics</code>、可选 <code>last_checked_at</code>/
<code>error</code>、<code>created_at</code> 和 <code>updated_at</code>。

## 8. 运行时

### 8.1 Runtime profile

| 方法 | 地址 | Body | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>POST</code> | <code>/api/v1/runtime/profiles</code> | <code>name</code>（1–64 字符）和 <code>load</code>（ModelLoadRequest） | 201 RuntimeProfileResource |
| <code>GET</code> | <code>/api/v1/runtime/profiles</code> | 无 | 200 RuntimeProfileList |
| <code>GET</code> | <code>/api/v1/runtime/profiles/{profile_id}</code> | 无 | 200 RuntimeProfileResource |
| <code>PUT</code> | <code>/api/v1/runtime/profiles/{profile_id}</code> | 与 POST 完全相同的完整 body | 200 RuntimeProfileResource |
| <code>DELETE</code> | <code>/api/v1/runtime/profiles/{profile_id}</code> | 无 | 204 |
| <code>POST</code> | <code>/api/v1/runtime/profiles/{profile_id}/load</code> | <code>allow_drift</code> Boolean，默认 <code>false</code>。 | 202 OperationAccepted |

<code>RuntimeProfileResource</code> 在 <code>id</code>、<code>name</code>、<code>load</code> 外，
还包含 <code>artifact_id</code>（32 位小写十六进制）、<code>artifact_modified_at</code>、
<code>drifted</code>、可选 <code>drift_reason</code>、<code>created_at</code> 和
<code>updated_at</code>。检测到产物替换或丢失时 profile 会 drift；必须显式
<code>allow_drift=true</code> 才能加载。

### 8.2 实例、状态、指标和日志

| 方法 | 地址 | 查询 / Body | 成功响应 |
| :--- | :--- | :--- | :--- |
| <code>GET</code> | <code>/api/v1/runtime/instances</code> | 无 | 200 RuntimeInstanceList |
| <code>GET</code> | <code>/api/v1/runtime/capabilities</code> | 无 | 200 RuntimeCapabilitiesResource |
| <code>GET</code> | <code>/api/v1/runtime/status</code> | 无 | 200 Object |
| <code>GET</code> | <code>/api/v1/runtime/metrics</code> | 可选 <code>instance_id</code> UUID、<code>since</code> DateTime；<code>limit</code> 1–2000，默认 200。 | 200 RuntimeMetricList |
| <code>GET</code> | <code>/api/v1/runtime/logs</code> | 可选 <code>instance_id</code> UUID、<code>level</code>（<code>debug</code>/<code>info</code>/<code>warning</code>/<code>error</code>）；<code>after</code> 最小 0，默认 0；<code>limit</code> 1–2000，默认 200。 | 200 RuntimeLogList |
| <code>GET</code> | <code>/api/v1/runtime/models</code> | 无 | 200 Object |
| <code>GET</code> | <code>/api/v1/runtime/realtime/capabilities</code> | 无 | 200 Object |
| <code>POST</code> | <code>/api/v1/runtime/reload</code> | <code>{"context_size": Integer}</code>，至少 512。 | 200 Object |
| <code>POST</code> | <code>/api/v1/runtime/cache/clear</code> | 无 | 200 Object |

<code>RuntimeInstanceResource</code> 包含 <code>id</code>、<code>model</code>、状态
<code>loading</code>/<code>ready</code>/<code>busy</code>/<code>unloading</code>/<code>failed</code>、
<code>devices</code>、<code>active_sessions</code>、<code>queued_requests</code>、可选
<code>resident_bytes</code>/<code>kv_bytes</code>/<code>context_size</code>/<code>started_at</code>/
<code>last_used_at</code>/<code>identity</code>/<code>error</code>。
<code>identity</code> 含模型、tokenizer、chat template、RoPE 参数的 SHA-256，以及
<code>quantization</code>、<code>runtime_build</code>、<code>processor_version</code>、
<code>kv_dtype</code>。

<code>RuntimeCapabilitiesResource</code> 含 <code>model</code>、<code>model_type</code>、
<code>duplex_available</code> 和 <code>model_capabilities</code>。后者含
<code>architecture_family</code>、<code>source</code> 和 <code>features</code>；
features 的 <code>text</code> 默认 true，<code>image_input</code>、<code>video_input</code>、
<code>audio_input</code>、<code>audio_output</code>、<code>full_duplex</code> 默认 false。

指标元素为 <code>sequence</code>、可选 <code>instance_id</code>/<code>model</code>、
<code>values</code> Object 和 <code>captured_at</code>。日志元素为 <code>sequence</code>、
可选 <code>instance_id</code>、<code>level</code>、<code>message</code>、<code>fields</code>、
<code>created_at</code>。

<code>/runtime/status</code>、<code>/runtime/realtime/capabilities</code>、
<code>/runtime/reload</code> 和 <code>/runtime/cache/clear</code> 透传已连接原生后端的对象，
因此不承诺固定字段集。<code>/runtime/models</code> 在托管运行时至少使用
<code>{"object":"list","data":[...]}</code>，每个元素含 <code>id</code>、<code>object</code>
（<code>model</code>）、<code>model</code>、<code>state</code> 和 <code>instance_id</code>。
清除缓存时原生后端可以因正在生成或全双工会话活动而拒绝请求。

## 9. 返回模型速查

| 模型 | 顶层字段 |
| :--- | :--- |
| SessionResource | <code>id</code>、<code>model</code>、<code>mode</code>、<code>state</code>、<code>revision</code>、<code>title</code>、<code>runtime_instance_id</code>、<code>created_at</code>、<code>updated_at</code>、<code>metadata</code>。 |
| Message | <code>id</code>、<code>role</code>、<code>parts</code>、<code>parent_id</code>、<code>created_at</code>。 |
| AppendMessageResult | <code>session</code>（SessionResource）、<code>message</code>（Message）。 |
| ResponseResource | <code>id</code>、<code>request_id</code>、<code>session_id</code>、<code>status</code>、<code>output_message_id</code>、<code>output</code>、<code>finish_reason</code>、<code>usage</code>、<code>performance</code>、<code>settings</code>、<code>created_at</code>、<code>completed_at</code>、<code>error</code>。 |
| GenerationPresetResource | <code>id</code>、<code>name</code>、<code>model</code>、<code>mode</code>、<code>settings</code>、<code>context_size</code>、<code>metadata</code>、<code>created_at</code>、<code>updated_at</code>。 |
| ApiKeyResource | <code>id</code>、<code>name</code>、<code>prefix</code>、<code>scopes</code>、<code>role</code>、<code>expires_at</code>、<code>revoked_at</code>、<code>created_at</code>、<code>last_used_at</code>。 |
| ApiKeySecretResource | <code>key</code>（ApiKeyResource）、<code>token</code>；token 只在创建或轮换时出现一次。 |
| McpServerResource | <code>id</code>、<code>name</code>、<code>transport</code>、<code>enabled</code>、<code>url</code>、<code>command</code>、<code>args</code>、<code>header_env</code>、<code>timeout_seconds</code>、<code>created_at</code>、<code>updated_at</code>。 |
| McpToolCallResult | <code>server</code>、<code>name</code>、<code>content</code>、<code>structured_content</code>、<code>is_error</code>。 |
| JobEventResource | <code>job_id</code>、<code>sequence</code>、<code>type</code>（state/progress/log/artifact）、<code>level</code>、<code>message</code>、<code>progress</code>、<code>data</code>、<code>created_at</code>。 |
| RemoteNodeResource | 见第 7.2 节。 |
| RuntimeProfileResource | 见第 8.1 节。 |

所有 <code>*List</code> 和 <code>*SearchResult</code> 响应都以 <code>data</code> 数组包装；
<code>McpToolList</code> 另外有 <code>errors</code> Object。

## 10. 认证和密钥接口细节

**创建密钥**：<code>POST /api/v1/auth/keys</code>，成功返回
<code>201 ApiKeySecretResource</code>。

| Body 字段 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :---: | :--- | :--- |
| <code>name</code> | String | **是** | - | 1–64 字符。 |
| <code>scopes</code> | String[] | 否 | <code>[]</code> | 最多 4 个且不重复：<code>inference</code>、<code>models</code>、<code>jobs</code>、<code>admin</code>。 |
| <code>role</code> | String / null | 否 | <code>null</code> | <code>viewer</code>、<code>operator</code> 或 <code>administrator</code>。 |
| <code>expires_at</code> | DateTime / null | 否 | <code>null</code> | 到期时间。 |

<code>scopes</code> 和 <code>role</code> 至少提供一个；两者可以组合，最终权限为并集。
<code>GET /api/v1/auth/keys</code> 返回 ApiKeyList；
<code>POST /api/v1/auth/keys/{key_id}/revoke</code> 返回撤销后的 ApiKeyResource；
<code>POST /api/v1/auth/keys/{key_id}/rotate</code> 返回新的 ApiKeySecretResource。
密钥存储只保留哈希和展示前缀，调用方必须立即安全保存 <code>token</code>。

## 11. SSE 流

### 11.1 响应生成 SSE

<code>POST /api/v1/sessions/{session_id}/responses</code> 在 <code>stream=true</code>（默认）时使用
<code>Content-Type: text/event-stream</code>。每个事件使用：

~~~
event: response.text.delta
id: 2
data: {"protocol_version":"1.0","session_id":"...","sequence":2,"timestamp":"...","payload":{...}}

~~~

外层为 RealtimeFrame：<code>protocol_version</code> 固定 <code>1.0</code>，
<code>session_id</code> 为 UUID，<code>sequence</code> 从 0 单调递增，
<code>timestamp</code> 为 DateTime。<code>event</code> 等于 payload 的 <code>type</code>。

| 事件类型 | payload 字段 |
| :--- | :--- |
| <code>response.text.delta</code>、<code>response.reasoning.delta</code> | <code>response_id</code>、<code>delta</code>。 |
| <code>response.tool_call.delta</code> | <code>response_id</code>、<code>index</code>、可选 <code>call_id</code>/<code>name</code>、<code>arguments_delta</code>。 |
| <code>response.completed</code> | <code>response_id</code>、<code>finish_reason</code>、可选 <code>usage</code>/<code>performance</code>。 |
| <code>session.state</code> | <code>state</code>、<code>revision</code>。 |
| <code>error</code> | <code>error</code>（通用 ErrorDetail）。 |

<code>usage</code> 含非负 <code>prompt_tokens</code>、<code>completion_tokens</code>、
<code>total_tokens</code>。<code>performance</code> 含 <code>prefill_tokens</code>、
<code>ttft_ms</code>、<code>prefill_ms</code>、<code>prefill_tps</code>、
<code>decode_ms</code>、<code>decode_tps</code>、<code>generation_ms</code>、
<code>generation_tps</code> 和实际 <code>sampling</code>。

上表是此*HTTP SSE 路由当前实际产生*的事件集合。<code>response.audio.delta</code>、
<code>response.interrupted</code> 和 <code>runtime.metrics</code> 虽是通用实时协议模型的事件，
但不会由 <code>POST /sessions/{session_id}/responses</code> 的 SSE 实现发出；它们属于实时
WebSocket 协议，见 <code>docs/api/websocket.md</code>。

### 11.2 作业 SSE

<code>GET /api/v1/jobs/{job_id}/events/stream</code> 返回 <code>text/event-stream</code>。
<code>after</code> 查询参数最小为 0，默认 0；可选 Header
<code>Last-Event-ID</code>（最小 0）可用于断线恢复，服务端从两者中的较大值继续。

每个事件为：

~~~
id: <JobEventResource.sequence>
event: <JobEventResource.type>
data: <JobEventResource JSON>

~~~

无新事件时会发送注释 keep-alive；作业进入 <code>succeeded</code>、<code>failed</code>、
<code>cancelled</code> 或 <code>interrupted</code> 后流结束。

## 12. 健康检查

**请求方式**：<code>GET</code><br>
**请求地址**：<code>/health</code>

~~~
{
  "status": "ok",
  "service": "mfq-server",
  "protocol_version": "1.0"
}
~~~

## 13. 契约更新

服务端路由或协议模型变更后，重新生成并校验 OpenAPI：

~~~
python -m mfq.server.openapi mfq/server/protocol/openapi.json
python -m mfq.server.openapi --check mfq/server/protocol/openapi.json
~~~

WebSocket 没有作为 OpenAPI <code>paths</code> 操作描述；契约的
<code>x-mfq-websocket</code> 扩展记录了旧路径的帧模型。握手和当前实时音频透明代理行为详见
<code>docs/api/websocket.md</code>。
