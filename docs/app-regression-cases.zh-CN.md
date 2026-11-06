# MFQ Studio 正常链路回归用例（第一版）

## 目标与边界

以用户安装并启动的 MFQ Studio 及其随附服务为主要测试对象。第一版只覆盖一次正常使用：用固定源模型生成量化文件，加载后回答一个有明确答案的问题，再通过同一服务调用。先判定用户实际使用的功能结果是否正确，然后用**同一个 `.mfq` 文件**判定链路是否完整；不另加重复的内部检查。

**版本适用范围**：当前 Windows 便携版 `0.4.0+dev.867b54f2.modified` 是服务和推理发布包。其打包脚本明确不包含量化依赖，Studio 也未接入量化页面；`MFQStudio/e2e/quantization-disabled.spec.ts` 明确检查量化入口不存在。因此 APP-R02 是未来提供应用内量化的版本的预留用例，不能作为当前便携版的通过门槛。当前便携版先按“已有完整 MFQ → 登记 → 加载 → 对话 → 服务调用”执行应用回归；源模型量化另用项目正式提供的量化入口单独验证，并记录两者的版本与产物交接。

不把源码编译作为使用流程的前置条件。断网、强制关机和罕见边界留给后续有实际需求时增补。本文件是待执行的用例，不代表链路已经通过。

## 固定输入与环境

- 源模型：ModelScope `Qwen/Qwen3.5-0.8B`，修订版 `4d58a7b524cd33ed843d5125be8cd8f0a452d9bf`。下载后按 [模型锁定清单](test-models/modelscope-qwen3.5-0.8b.lock.json)核对文件、大小和 SHA-256；模型不完整时不开始测试。
- 应用：记录实际安装包版本、下载来源及所带服务版本。使用该安装包的 Studio 和服务，不能以开发环境中的服务替代。运行设备需满足应用对该模型的支持与内存要求。
- 输出：选择一个可写、空间充足的测试目录，给本次运行使用新的 `.mfq` 文件名。记录源模型目录、输出路径和应用显示的模型名称。测试前不预先登记该输出文件。
- 量化设置：初次执行使用界面默认的目标 `4 bpw`、所有与源模型兼容的候选、`DF-V1-AlphaQ` 配方和“不影响服务”策略，完整记录实际候选与参数。若该版本报告此组合无可行配方，先记录结果，再依据界面给出的可行范围确定并锁定后续回归参数；不能把调整后的结果冒充原组合通过。生成文本不作逐字比较。若所测安装包没有量化工作台或不支持该源模型，应记录为该版本**不适用**，改用具备该功能的正式安装包执行，不能用源码命令假装完成应用用例。

## 用例

### 最小正确性判定

APP-R01 只证明应用和服务已就绪。对生成的量化模型，在 APP-R03 发送一次固定的简单问题“只回答数字：2+3 等于多少？”，要求实际答案为 `5`；APP-R04 用同一模型验证服务接口也能正确回答。复用这两次结果判断推理功能，再核对产物是否来自 APP-R02，判断链路是否完整。当前软件包没有应用内量化入口，源码量化结果单独记录，不能计作 APP-R02 通过。项目已有张量级数值测试；常规应用回归不重复抽查内部权重。

### APP-R01：应用与随附服务就绪

**步骤**：启动 MFQ Studio；进入“概览”“模型”“服务”，读取服务状态和应用版本。

**通过标准**：工作区可正常打开；应用显示服务已连接或可由应用正常启动；“模型”与“服务”页面能读取状态。记录版本和服务地址。此处不要求本地已有模型。

### APP-R02：从源模型生成并登记整模

**前置**：APP-R01 通过；锁定源模型完整。

**步骤**：在应用的“量化工作台”选择源模型目录并点击“检查模型”；记录识别出的架构、张量数和源精度。选择兼容候选及目标 bpw，点击“生成配方”，等待配方任务完成；使用该配方设置新的整模输出路径，点击“注册量化任务”，等待任务完成。随后在“模型”页查找本次输出。

**通过标准**：源模型检查成功且识别信息与输入相符；配方任务和整模量化任务均显示“已完成”，没有失败状态；输出位置产生非空 `.mfq` 文件；应用自动登记该文件，并在“模型”页显示为完整、可加载的本地检查点。记录实际文件大小、SHA-256、任务编号及应用中的模型名称。只核对本次产物，不使用预制量化文件替代。

### APP-R03：加载本次产物并完成对话

**前置**：APP-R02 通过。

**步骤**：在“模型”页加载 APP-R02 自动登记的检查点，等待状态就绪；进入“对话”，选择该模型，发送“只回答数字：2+3 等于多少？”并等待生成结束。

**通过标准**：加载任务成功，模型状态显示已加载或就绪；生成正常结束，回复表达的数值为 `5`，界面没有模型加载或推理错误。保存模型名称、任务状态及回复截图或文字。

### APP-R04：通过应用随附服务调用同一模型

**前置**：APP-R03 通过；沿用应用启动的同一服务与已加载模型。

**步骤**：向该服务请求 `GET /v1/models`，从响应中取出本次模型的实际 ID；用该 ID 向 `POST /v1/chat/completions` 发送“只回答数字：2+3 等于多少？”，设置 `enable_thinking=false`、`max_tokens=64`、`temperature=0`。需要鉴权时使用应用当前服务的配置。

**通过标准**：两个请求均成功；模型列表包含 APP-R03 已加载的模型；生成响应中的模型标识与请求一致、请求正常完成，且回答表达的数值为 `5`。记录请求时间、HTTP 状态码、模型 ID 与去敏后的响应摘要。该用例验证应用随附服务的真实推理结果，不以模拟响应代替。

### APP-R05：把新生成的产物从其自身路径添加进应用并推理

**前置**：应用与服务已就绪（APP-R01）；已有一次可用的源码量化产物（APP-R02 不适用时的替代来源）。

**步骤**：在应用“模型”页添加/登记产物所在目录，等待扫描完成；确认新产物以独立条目出现；加载该条目并等待实例就绪；通过对话发送“只回答数字：2+3 等于多少？”；再用随附服务 `/v1/models` 与 `/v1/chat/completions` 对同一模型提问。

**通过标准**：登记后新产物出现在模型列表，条目显示为完整、可加载；加载任务成功、模型状态就绪；对话得到数值为 `5` 的可见回答并正常结束；服务接口返回同一模型标识、`finish_reason=stop`、回答 `5`。记录产物路径、大小、SHA-256、登记 ID、实例 ID、设备与耗时。该用例不要求应用内量化，覆盖“添加已有模型 → 加载 → 推理”这一当前版本可用流程。

### APP-R06：流式输出（SSE）

**前置**：APP-R01 通过，模型已加载且可对话（APP-R03/R04 的功能步骤）。

**步骤**：经 OpenAI 兼容接口以 `stream=true` 发固定问题，读取事件流；再经应用会话接口以 `stream=true` 发同一问题，读取事件流。

**通过标准**：HTTP 200 且 `Content-Type: text/event-stream`；收到文本增量事件与完成事件，`finish_reason=stop`；拼接后的回答匹配固定断言；会话流需出现 `session.state`，且不出现 `error` 事件。记录首个事件耗时与答案。

### APP-R07：多轮上下文、系统提示与历史

**前置**：同 APP-R06。

**步骤**：同一会话内 turn1 发“2+3”；turn2 发“把上一个答案再加 4”；turn3 带 system prompt 发“1+1”；读取该会话的消息列表。

**通过标准**：三问分别得到 `5`、`9`、`2`；第二问的 `total_tokens` 大于第一问（历史进入上下文）；消息列表同时包含 user 和 assistant 角色且数量与轮次一致。

### APP-R08：模型卸载、切换与重载

**前置**：已登记两个不同名的模型 A、B（可来自同一产物的不同副本，文件名不同）。

**步骤**：加载 A 对话；加载 B 对话；卸载 B；重新加载 A 对话。

**通过标准**：两次对话均返回所请求模型标识、回答匹配固定断言；B 的卸载任务成功且运行实例消失；A 重新加载后状态 `ready` 并仍能回答。

### APP-R09：加载与登记的错误路径

**前置**：同 APP-R06。

**步骤**：登记不存在的目录；登记含损坏 `.mfq` 的目录；登记含两个同名产物的目录；登记只有一半分片的目录；加载不存在的模型；对已登记模型指定不存在的 `mfq://` URI。

**通过标准**：不存在目录返回 404；损坏容器返回 400 或被标记为 incomplete；重名返回 409；缺分片条目标记 `loadable=false`（或 400）；未知模型/URI 的加载任务进入 `failed` 且带错误码。错误输入不得被当作可用模型。

## 执行与归档

已有 Python 服务测试、量化测试和 Studio 组件测试覆盖了各环节的部分行为；`MFQStudio/e2e/mockServer.ts` 对应用接口使用模拟服务。它们不能替代 R01–R09 中“同一次量化产物 → 正式应用 → 随附服务”的真实闭环，因此本用例只补这个缺口，不复制现有单元测试组合。

- 每次执行记录：日期、操作系统与显卡、应用及服务版本、源模型修订版和校验结果、量化设置、输出文件路径/大小/校验值、各用例结果及失败阶段。结果使用“通过 / 失败 / 不适用 / 未执行”，不能把“不适用”算作通过。
- R01–R09 是一条连续的人工回归链路。后续若自动化，应驱动正式应用及其随附服务，并保留同样的产物关联和可观察断言。
- 新功能在对应 PR 中附持续性用例；功能合入后，只将稳定、常用且仍受支持的场景归档到此回归集，避免扩大为大量重复的防御性测试。

## 2026-11-03 首轮执行记录

测试对象是桌面快捷方式指向的便携版启动文件；界面报告 `v0.4.0`、`开发版 · 867b54f2`，内部版本为 `0.4.0+dev.867b54f2.modified`。本轮没有重新核对源模型锁定清单，也没有从应用重新生成产物；仅使用此前已生成并登记的 `qwen35-0.8b-nint4.mfq`（484,845,334 字节）做加载与服务调用。

| 用例 | 结果 | 可观察证据 |
| --- | --- | --- |
| APP-R01 | 通过 | Studio 概览显示服务空闲、端点 `127.0.0.1:8090/v1`；随附服务 `/health` 返回 `status=ok`、`service=mfq-server`。 |
| APP-R02 | 当前版本不适用 | Windows 发布包按打包脚本排除量化组件，Studio 路由也没有接入量化页面；量化工作区候选集合为空与此一致。这不是本次测试发现的量化失败。 |
| APP-R03 | 界面对话步骤通过；整条链路未通过 | 已登记模型通过应用随附服务加载，实例状态为 `ready`。重测时在 Studio 新建会话、发送“请直接回答：你好。”，页面显示非空回复“你好！我是 Qwen3.5，非常乐意为你提供帮助。请问有什么我可以协助你的吗？”。因产物不是本轮 APP-R02 在应用中生成，本用例的连续链路前置条件仍未满足。 |
| APP-R04 | 部分验证 | `/v1/models` 返回该模型；真实 `/v1/chat/completions` 在 `enable_thinking=false`、`max_tokens=64` 时返回非空回答“你好！请问有什么我可以帮您的吗？”且 `finish_reason=stop`。因 APP-R02/03 未完整通过，整条回归链路仍未通过。 |

默认思考模式下，`max_tokens=32` 和 `128` 两次请求都在输出思考内容时达到长度上限，`message.content` 为空；这不满足本用例的“可见回答”断言。后续固定请求应显式关闭思考模式，或按该模式设置足够的生成上限，并记录实际参数。

2026-11-03 再核对：GitHub 当前最新发布为 `v0.4.0`，本机 `MFQ.Studio_0.4.0_x64-portable.zip` 的 SHA-256 与官方 `SHA256SUMS.txt` 一致（`6e3c7c3e29f1a8f38f27d1eddb50d63f1d20dcb6440de01609dd7b60bb5ed4a7`）。界面重测证实对话可用；先前未完成是测试操作未进入对话页，不是观察到的应用故障。

## 2026-11-03 源码量化补充验证

这是与 APP-R02 分开的源码测试，不改变当前 Windows 便携版“应用内量化不适用”的判定。测试使用主仓库提交 `f615cfc5cccf9670813733d2d78150a00fc00428` 的量化源码、既有 Python 量化环境，以及锁定修订版 `4d58a7b524cd33ed843d5125be8cd8f0a452d9bf` 的 Qwen3.5-0.8B 源模型。执行 `mfq quantize`，参数为 `--source-format hf --bits 4 --groupsize 24 --sub-bits 6 --backend cpu --device cpu --text-only`；输出到 `tmp/qwen35-0.8b-nint4-source-current.mfq`，未覆盖旧产物。

命令以退出码 0 完成，处理 335 个源张量，耗时 417.73 秒。新文件为 484,845,334 字节，SHA-256 为 `26ed9eee9af0f38232c404984f1b8caa2f31896735ab1b8b7d5b5a3fdee9a9d2`；源码 `load_mmap` 成功打开，读到 340 条记录。校验值与此前经 Studio 随附服务加载、在界面对话的旧产物完全一致，因此此次源码生成的文件与已验证的应用产物逐字节相同。这些证据先证明量化命令、文件生成及读取可用，不能单独证明数值正确。

此前额外做过一次内部权重抽查：两块量化权重的归一化均方根误差为 `0.067843`、`0.068373`，一块保留精度的权重与源值相同。这是历史诊断记录，**不列入常规应用回归用例**。固定问题的推理正确性尚未执行；已有“你好”回复只证明应用能生成可见文本，不能代替该检查。

## 2026-11-03 固定问题复测（界面与服务）

复测对象沿用首轮：桌面快捷方式指向的便携版（界面 `v0.4.0`、`开发版 · 867b54f2`，内部版本 `0.4.0+dev.867b54f2.modified`），随附服务 `127.0.0.1:8090`，已加载模型 `qwen35-0.8b-nint4`（实例 `5c9685c4-655a-41e5-93d9-fe4210cda44d`，状态 `ready`）。固定问题为“只回答数字：2+3 等于多少？”。本节补充并替代首轮中仅基于“你好”回复的记录；首轮记录的版本与产物来源判定不变。

| 步骤 | 实际结果 | 判定 |
| --- | --- | --- |
| 界面会话（APP-R03 功能步骤） | Studio 会话记录中存在该用户消息与助手回复（`2026-11-03T05:21:52Z` / `05:22:00Z`）；助手先输出推理（默认思考模式），可见回答为 `2+3 = 5`，响应正常结束。 | 界面推理正确性：**通过** |
| 服务调用（APP-R04 功能步骤） | `POST /v1/chat/completions`，固定 `enable_thinking=false`、`max_tokens=64`、`temperature=0`；HTTP 200，返回模型标识 `qwen35-0.8b-nint4`（与请求一致），`finish_reason=stop`，`message.content` 为 `5`。 | 服务推理正确性：**通过** |

证据来源：助手回复可在 `/api/v1/sessions/<会话ID>/messages` 中查到；服务侧请求记录见 `/api/v1/runtime/status`（`endpoint=chat`、`finish_reason=stop`、`failed_requests=0`）。两处使用同一已加载产物：484,845,334 字节，SHA-256 `26ed9eee9af0f38232c404984f1b8caa2f31896735ab1b8b7d5b5a3fdee9a9d2`，与本轮源码量化产物逐字节一致。

**链路判定**：仍不能记为“APP-R02 → APP-R03 → APP-R04 整链通过”。APP-R02 对当前便携版不适用（打包排除量化依赖、Studio 未接入量化入口）；本轮产物来自源码量化并已由应用登记/加载，因此可确认的是“源码量化产物 → 应用登记/加载 → 对话与服务推理”的交接可用。APP-R03、APP-R04 只记录各自功能步骤通过，不把有 APP-R02 前置条件的整项标为通过；完整“应用内量化 → 自动登记 → 推理”待正式包提供量化入口后再测。

### API 驱动复测（同日，无需 GUI）

为支持后续自动化执行，改用应用随附服务的会话接口复测同一功能路径（Studio 前端调用同一接口，非模拟服务）：

| 检查项 | 结果 |
| --- | --- |
| 服务健康 | `GET /health` 返回 `status=ok`、`service=mfq-server`。 |
| 运行时实例 | `qwen35-0.8b-nint4` 状态 `ready`，设备 `cuda:0`。 |
| 会话对话 | `POST /api/v1/sessions` 创建会话后，`POST /api/v1/sessions/{id}/responses` 发送固定问题：`status=completed`、`finish_reason=stop`，可见回答 `5`，耗时 435 ms。 |
| OpenAI 兼容 | `POST /v1/chat/completions`（`enable_thinking=false`、`max_tokens=64`、`temperature=0`）：`finish_reason=stop`，`message.content="5"`，耗时 385 ms。 |
| 产物元数据 | 应用识别：484,845,334 字节、335 张量、340 条记录、`complete=true`、`loadable=true`。 |
| 运行计数 | `total_requests=8`、`failed_requests=0`、`active_requests=0`。 |

结论：界面与服务两条推理判据在 API 驱动下同样通过；链路判定不变（仍受 APP-R02 不适用限制）。该复测不改变 APP-R03/APP-R04 的整项判定口径，也不替代以 GUI 观察“可见回答与正常结束”的原始证据。

## 2026-11-03 源码量化自测复跑（本轮）

环境：主仓库提交 `f615cfc5cccf9670813733d2d78150a00fc00428`；Python 3.13.14（`uv sync --extra train`，torch 2.13.0+cpu、safetensors 0.8.0）；CPU 量化后端。原 `.venv` 已损坏（缺 `pyvenv.cfg`、解释器无法启动），本次删除后按锁文件重建。

命令（与“源码量化补充验证”同组参数，输出到新文件、不覆盖旧产物）：

```sh
mfq quantize <源模型目录> tmp/qwen35-0.8b-nint4-source-rerun-20261103.mfq \
  --source-format hf --bits 4 --groupsize 24 --sub-bits 6 \
  --backend cpu --device cpu --text-only
```

| 项目 | 记录 |
| --- | --- |
| 开始/结束 | 2026-11-03 13:32:02 / 13:40:05（Asia/Shanghai） |
| 退出码 | `0`，状态 `ok` |
| 张量 | 335 个源张量全部处理；目标计数 `NINT4` 187 / `F32` 133 / `BF16` 15 |
| 耗时 | 475.46 秒（`elapsed_sec`） |
| 产物 | `tmp/qwen35-0.8b-nint4-source-rerun-20261103.mfq`，484,845,334 字节，单分片 |
| SHA-256 | `26ed9eee9af0f38232c404984f1b8caa2f31896735ab1b8b7d5b5a3fdee9a9d2`，与历史产物及 Studio 已加载产物逐字节一致 |
| 可读性 | 源码 `load_mmap` 打开成功：header v2、架构 `qwen3_5-hf-mfq-nint-recipe`、340 条记录、dtypes `BF16/BLOB/F32/NINT`，与应用登记元数据一致 |

观察到的非致命警告：构建原生 tokenizer GGUF 时，`C:\Users\30985\.cache\mfq\tokenizers\` 下临时文件写入被拒绝（Permission denied），命令回退后继续完成。因产物与已验证产物逐字节相同，判定对本次结果无影响；该缓存权限问题值得后续单独排查。

结论：源码量化链路（固定源模型 → 量化 → 文件生成 → 文件可读）在本轮通过。该结果只验证源码量化功能，不改变 APP-R02 对当前便携版“不适用”的判定，也不代表应用内量化已可用。该产物当时未在 Studio 重新登记；同日稍晚已通过 APP-R05 完成“新产物登记 → 加载 → 对话”，见下节。

## 2026-11-03 新产物直接登记、加载与对话（补齐“添加模型 → 推理”）

背景：应用内量化不适用（APP-R02），此前只验证过“旧产物已加载 → 对话/服务”；“把一次新生成的产物从它自己的路径添加进应用 → 加载 → 推理”这一步没有单独执行过。本轮用源码新生成的 `tmp/qwen35-0.8b-nint4-source-rerun-20261103.mfq`（484,845,334 字节，SHA-256 `26ed9eee...e9a9d2`）直接走通，并固化为自动化用例 APP-R05。

环境：Windows 11，便携版 `0.4.0+dev.867b54f2.modified`，随附服务 `127.0.0.1:8090`，RTX 4060（`cuda:0`），模型目录 `D:\project\TyloQuant\tmp`。

| 步骤 | 实际结果 | 判定 |
| --- | --- | --- |
| 登记目录 | `POST /api/v1/models/directories/register`（`path=D:\project\TyloQuant\tmp`）返回两个目录条目；新产物登记为 `qwen35-0.8b-nint4-source-rerun-20261103`（ID `d5f2f6a4...`，340 记录，`complete=true`、`loadable=true`）。 | 通过 |
| 加载 | `POST /api/v1/models/load`（`artifact_uri=mfq://d5f2f6a4...`）→ 任务 `succeeded`，实例 `38192768...`，耗时 2.0 秒，状态 `ready`、设备 `cuda:0`。 | 通过 |
| 界面/会话对话 | `POST /api/v1/sessions` + `responses`（Studio 前端同款路径）：`status=completed`、`finish_reason=stop`、回答 `5`，耗时 177 毫秒。 | 通过 |
| 服务对话 | `POST /v1/chat/completions`（`enable_thinking=false`、`max_tokens=64`、`temperature=0`）：模型标识一致、`finish_reason=stop`、回答 `5`，耗时 513 毫秒。 | 通过 |

复跑方式（运行器的 `app` stage，登记 → 加载 → 服务 → 会话一次完成）：

```sh
python tests/e2e/app_regression.py --stage app --register-dir tmp \
  --artifact-name qwen35-0.8b-nint4-source-rerun-20261103 \
  --stage service --stage session \
  --service-url http://127.0.0.1:8090 --require-service \
  --report tmp/e2e-app-full.json
```

结论：APP-R05 通过。在 APP-R02 暂不适用的前提下，当前版本“添加模型 → 加载 → 推理”的基本流程已测完整。APP-R03/APP-R04 的整项判定不变（其前置仍是应用内量化），但“已有产物 → 应用 → 对话/服务”的交接已由本条独立验证，不再依赖字节等价推断。注意：应用数据库会保留本次登记的 `tmp` 目录条目（`qwen35-0.8b-nint4-source-current` 与 `qwen35-0.8b-nint4-source-rerun-20261103`），需要时可在应用内移除。

## 2026-11-06 P0 增强执行记录（APP-R06–R09 + 断言增强）

背景：首版正确链路回归只覆盖“单模型、单问题、单轮、非流式”。本轮按 P0 补齐四类深度：流式、多轮与系统提示、加载/切换生命周期、负面边界，并加强 service/session 的断言（usage、性能、结构化输出、幂等、确定性）。

环境：Windows 11，便携版 `0.4.0+dev.867b54f2.modified`，服务 `127.0.0.1:8090`，RTX 4060（`cuda:0`）。

一条命令执行 9 个 stage（`r0`、`artifact`、`app`、`service`、`session`、`stream`、`multi_turn`、`unload_switch`、`negative`），14.6 秒、64 个检查点：

| 用例 | 实际结果 | 判定 |
| --- | --- | --- |
| APP-R06（流式） | OpenAI SSE：`text/event-stream`、8 个事件、`[DONE]`、`finish_reason=stop`、首个事件 157ms；会话 SSE：`response.text.delta`、`response.completed`、`session.state` 均出现，拼接回答 `5`。 | 通过 |
| APP-R07（多轮） | turn1=`5`、turn2=`9`（`total_tokens` 25→47）、system prompt=`2`、消息 6 条（3 user + 3 assistant）。 | 通过 |
| APP-R08（切换） | A 加载→对话 `5`；B 加载→对话 `5`；卸载 B（job `succeeded`、实例消失）；重载 A→对话 `5`。 | 通过 |
| APP-R09（负面） | 不存在目录 404；损坏容器 incomplete；重名 409；缺分片 `missing_shards=1`、`loadable=false`；未知模型/URI 的加载 job failed（`model_artifact_not_found`）。 | 通过 |
| 断言增强 | service/session `usage`、`ttft_ms`/`prefill_tps`、JSON 结构化输出 `{"answer": 5}`、同 `request_id` 幂等重放、temperature=0 内容可复现。 | 通过 |

复跑方式：

```sh
python tests/e2e/app_regression.py \
  --stage r0 --stage artifact --artifact <path/to/model.mfq> \
  --expect-records 340 --expect-arch qwen3_5-hf-mfq-nint-recipe \
  --stage app --register-dir tmp --artifact-name <catalog-model-name> \
  --stage service --stage session --stage stream --stage multi_turn \
  --model <model-id> --model-b <second-model-id> \
  --stage unload_switch --stage negative \
  --service-url http://127.0.0.1:8090 --require-service \
  --report tmp/e2e-p0-full.json
```

结论：APP-R06–R09 通过，P0 完成，全链一条命令可复跑。范围仍限定为“回归 × 正确链路”：异常场景（断网、异常关机等）按原设计后置，持续性测试仍按功能 PR 附带。运行器的 stage 列表与参数见 `tests/e2e/README.md`；`negative` stage 会向应用数据库登记少量带唯一名的临时目录，用于验证错误路径。
