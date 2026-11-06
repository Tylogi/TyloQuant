# MFQ 正确链路回归运行器（跨平台）

`tests/e2e/app_regression.py` 把"量化 → 产物 → 添加模型 → 加载 → 对话 → 服务"的正确链路做成可重复执行的最小运行器，不依赖 GUI，Windows / macOS / Linux 均可用。

## 分层

与 [回归测试集设计](../../docs/regression-test-plan.zh-CN.md) 的 R0–R3 对应：

| stage | 内容 | 环境要求 | 平台 |
| --- | --- | --- | --- |
| `r0` | 微型确定性 HF 模型 → `mfq quantize`（CPU）→ 容器读取 | Python + `train` 依赖；无需网络、无需 GPU、无需真实模型 | 全平台 |
| `artifact` | 指定 `.mfq`：大小、SHA-256、header、记录数、dtypes、内嵌资源 | Python + `mfq` | 全平台 |
| `app` | 把产物所在目录登记进应用目录 → 加载指定产物 → 等待运行时就绪 | 运行中的应用/服务；产物目录可访问 | 有应用/服务即可 |
| `service` | `/health`、`/v1/models`、`/v1/chat/completions`（固定问题）、usage、JSON 结构化输出 | 运行中的 MFQ 服务（应用随附或源码） | 有服务即可 |
| `session` | `/api/v1/sessions` + `responses`（Studio 前端同款路径），含 usage、性能、历史、幂等与 temperature=0 可复现 | 同上 | 同上 |
| `stream` | 两条流式路径：OpenAI 分块 SSE、Studio 会话 SSE（`response.text.delta`） | 同上；模型支持流式 | 同上 |
| `multi_turn` | 多轮上下文（上一答案参与下一问）、system prompt、会话历史持久化 | 同上 | 同上 |
| `unload_switch` | 加载 A → 加载 B → 用 B 对话 → 卸载 B → 重新加载 A → 再用 A 对话 | 需要两个已登记模型（`--model` 与 `--model-b`） | 同上 |
| `negative` | 错误路径：不存在的目录、损坏容器、目录重名、缺分片、未知模型、未知产物 URI | 同上；会用 CPU 量化一个微型 fixture | 同上 |

每个检查输出 `pass` / `fail` / `skip`；**skip 不计入通过**。退出码：`0` 无失败、`1` 有失败、`2` 配置错误。

## 审查者快速验证（3 步）

1. **先看唯一的产线代码改动（2 行）**：
   `git diff master -- cpp_runtime/backends/cuda/kernels/kv_hot_cache.cu`
   只补了 `batch_attention_store()` 调用 `append_exclusive_decode()` 时缺的 `DeviceInputError*` 参数。
2. **只看测试（无服务、无 GPU，约 3 秒）**：
   `uv sync --extra train --python 3.13`
   `python tests/e2e/app_regression.py --stage r0 --report /tmp/e2e-r0.json`
   期望：`r0.*` 5 项全 pass，退出码 0。
3. **Linux 全链（有 GPU + nvcc 时，约 30 秒）**：按「Linux 源码级服务复跑」章节三条命令执行，
   期望 9 stage / 64 checks 全 pass；无 GPU 时可只跑
   `--stage artifact --artifact <path> --expect-sha256 26ed9eee... --expect-records 340` 校验产物。

审查范围提示：产线代码仅 `kv_hot_cache.cu` 2 行（且为编译修复，不改内核语义）；其余全部是新增测试与文档；
不修改现有接口、不修改构建配置；484 MB 模型产物不入库，只提交校验清单。

## 运行

```sh
# 一次性准备（复现 Python 环境；见下方“环境可移植性”）
uv sync --extra train --python 3.13

# R0：任何机器，秒级完成
python tests/e2e/app_regression.py --stage r0 --report /tmp/e2e-r0.json

# 添加新产物 → 加载 → 对话 → 服务（应用已启动）
python tests/e2e/app_regression.py \
  --stage app --register-dir tmp --artifact-name <catalog-model-name> \
  --stage service --stage session --stage stream --stage multi_turn \
  --service-url http://127.0.0.1:8090 --require-service \
  --report /tmp/e2e-add-load.json

# P0 全量（正确链路 + 流式/多轮/切换/负面；需要两个已登记模型）
python tests/e2e/app_regression.py \
  --stage r0 --stage artifact --artifact <path/to/model.mfq> \
  --expect-records 340 --expect-arch qwen3_5-hf-mfq-nint-recipe \
  --stage app --register-dir tmp --artifact-name <catalog-model-name> \
  --stage service --stage session --stage stream --stage multi_turn \
  --model <model-id> --model-b <second-model-id> \
  --stage unload_switch --stage negative \
  --service-url http://127.0.0.1:8090 --require-service \
  --report /tmp/e2e-p0-full.json

# 产物 + 服务 + 会话（服务已启动、模型已登记并加载）
python tests/e2e/app_regression.py \
  --stage artifact --artifact <path/to/model.mfq> \
  --expect-records 340 --expect-arch qwen3_5-hf-mfq-nint-recipe \
  --stage service --service-url http://127.0.0.1:8090 --model <model-id> --require-service \
  --stage session \
  --report /tmp/e2e-app.json
```

所有参数也可写在 JSON 配置里（见 `app_regression.config.example.json`），用 `--config` 传入；命令行参数优先。

## 三问的答案：可移植性与版本演进

### 1. 能在另一台电脑上复现吗

- **`r0` / `artifact`：可以。** 前提是仓库同一提交 + `uv.lock`、显式指定 Python（本机曾出现 uv 误选 LibreOffice 内置 Python）、首次运行需要网络安装依赖。R0 使用微型 fixture，不下载任何模型。
- **`service` / `session`：需要同版本的安装包**（Windows x64 便携版或 macOS 包）且该机器有受支持的 GPU（运行时只有 CUDA/Metal），模型目录还需在本机登记（应用数据库是每机本地的）。
- 真实模型链路额外要求按 [模型锁定清单](../../docs/test-models/modelscope-qwen3.5-0.8b.lock.json) 下载并逐文件校验 Qwen3.5-0.8B。

### 2. 换到 Linux 还能进行吗

- **`r0` / `artifact`：可以。** 量化走 `--backend cpu --device cpu`；`uv.lock` 包含 manylinux x86_64/aarch64 的 torch 等 wheel。
- **`service` / `session`：不能直接沿用安装包路径。** `packaging/` 只有 macOS 与 Windows 发布脚本，没有 Linux 安装包；`cpp_runtime/backends/` 只有 `cuda` 与 `metal`，没有 CPU 推理后端。Linux + NVIDIA + `nvcc` 时可从源码 `mfq build --backend cuda` 并用 `mfq serve` 跑**源码级**服务用例，但这不等同于"安装包验收"；无 GPU 机器上应记为 `skip`（不适用），不能算通过。

### 3. 版本更新后还能作为回归用吗

可以，前提是遵守以下规则（运行器已按此实现）：

- **只用公开契约**：`docs/cli/quantize.md` 记录的 CLI 参数、`docs/api/http.md` 记录的 HTTP 接口；不依赖内部模块路径或私有实现。
- **断言稳定不变量**：退出码、文件非空、header 可读、记录数不少于张量数、内嵌资源存在、`finish_reason=stop`、答案匹配固定模式（默认"以 5 结尾"，同时接受 `5` 和 `2+3 = 5`）、`usage` 计数为正、流式事件顺序（delta → completed）、temperature=0/top_k=1 内容可复现、`request_id` 重放不产生新响应。
- **不跨版本断言字节哈希**：`--expect-sha256` 仅用于同机/同产物溯源；格式或运行时版本升级后字节会变，结构断言才长期有效。
- **能力探测决定适用性**：会话接口返回 404 时记为 `skip` 而不是失败；未来版本提供应用内量化入口时，APP-R02 自动从"不适用"变为可测，无需改代码。
- **记录版本四元组**：repo HEAD、mfq 版本、服务 `protocol_version`、模型 id、产物 SHA，失败时能直接定位版本组合。
- **预期值放配置**：`--expect-records`、`--expect-arch`、`--answer-pattern` 可随格式版本调整，不修改运行器。

## Linux 源码级服务复跑（可选）

没有 Linux 安装包、但有 NVIDIA GPU 和 `nvcc` 时，可以从源码构建运行时并跑完整 9 个 stage。
本仓库在 WSL2 Ubuntu 26.04 + CUDA 12.4 + RTX 4060 上实测通过（64/64）。

```sh
# 1) 构建 CUDA 运行时（RTX 4060 = sm_89；用 g++-13 作为 CUDA host compiler）
uv sync --extra train --python 3.13
mfq build --backend cuda -j 8 -- \
  -DMFQ_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DBUILD_TESTING=OFF

# 2) 启动服务（模型文件放本地盘；WSL 见下方 LD_LIBRARY_PATH 说明）
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH   # 仅 WSL 需要
mfq serve --model <path/to/model.mfq> --no-web-ui \
  --host 127.0.0.1 --port 8092 --data-dir <data-dir> --work-dir <work-dir>

# 3) 服务 /health 就绪后，跑完整 9 stage（首次运行会自动量化 tiny fixture）
python tests/e2e/app_regression.py \
  --stage r0 --stage artifact --artifact <path/to/model.mfq> \
  --expect-records 340 --expect-arch qwen3_5-hf-mfq-nint-recipe \
  --stage app --register-dir <model-dir> --artifact-name <catalog-name> \
  --stage service --stage session --stage stream --stage multi_turn \
  --model <catalog-name> --model-b <second-catalog-name> \
  --stage unload_switch --stage negative \
  --service-url http://127.0.0.1:8092 --require-service \
  --report /tmp/e2e-p0-linux.json
```

- 构建依赖：CMake ≥ 3.26、`nvcc`（12.4 / 13.3 均可）、g++-13；其他 GPU 按实际算力修改
  `-DMFQ_CUDA_ARCHITECTURES`（如 4090 = 89，H100 = 90）。
- 这是**源码级服务**验证，不等同于 Linux 安装包验收；无 GPU 环境应记为 `skip`。
- WSL 若安装过 `nvidia-cuda-toolkit`，其桌面驱动库会被二进制 RUNPATH 优先加载，导致
  `CUDA error: no CUDA-capable device is detected`；启动前设置 `LD_LIBRARY_PATH=/usr/lib/wsl/lib`
  即可（worker 会继承该环境变量）。
- 端口若与本机便携版 worker 冲突（默认 8090/8091），换空闲端口并同步 `--service-url`。

## 环境可移植性注意事项

- 显式指定解释器：`uv sync --extra train --python 3.13`（或 `--python <绝对路径>`），避免 uv 选中非标准 Python。
- 模型与产物路径通过参数传入，不使用绝对路径常量。
- 量化时若出现 `~/.cache/mfq/tokenizers/` 权限警告，属于缓存问题，不影响产物一致性；建议先确认缓存目录可写。
- 服务端口、模型 id、问题与答案模式全部可配置。

## 已知边界

- 应用内量化：当前便携版不可用；运行器不会把源码量化计作应用内量化。
- GUI 可见性：`app`/`service`/`session`/`stream` 通过应用自身的模型登记、加载接口与 Studio 同款会话接口驱动，覆盖"添加模型 → 加载 → 推理"；界面截图与布局断言仍需人工或 UI 自动化。
- `negative` stage 会向应用数据库登记几个临时目录（每次运行使用唯一名），用于验证错误路径；不需要时可手动移除。
- 性能与质量（KLD、吞吐）不在本运行器范围。
- 无 GPU 的环境只跑 `r0` / `artifact`，推理相关检查记为 `skip`。

## 实测记录

### 2026-11-06 P0 增强：流式、多轮、切换、负面 + 断言增强（Windows）

| 检查 | 结果 |
| --- | --- |
| 环境 | Windows 11，便携版 `0.4.0+dev.867b54f2.modified`，服务 `127.0.0.1:8090`，RTX 4060（`cuda:0`） |
| 全链命令 | `r0` + `artifact` + `app` + `service` + `session` + `stream` + `multi_turn` + `unload_switch` + `negative` 一条命令执行 |
| 结果 | **PASS**：14.6 秒，64 个检查点，0 fail / 0 skip |
| 流式 | OpenAI SSE：`text/event-stream`、8 个事件、收到 `[DONE]`、`finish_reason=stop`、首个事件 157ms；会话 SSE：`response.text.delta`、`response.completed`、`session.state` 均出现 |
| 多轮 | turn1=`5`、turn2=`9`（上下文生效，`total_tokens` 25→47）、system prompt 回答 `2`、历史消息 6 条（3 user + 3 assistant） |
| 切换 | A 加载→对话 `5`；B 加载→对话 `5`；B 卸载 job `succeeded` 且实例消失；A 重新加载→对话 `5` |
| 负面 | 不存在目录 404；损坏容器报 incomplete；目录重名 409；缺分片 `missing_shards=1`、`loadable=false`；未知模型/未知 URI 的加载 job 均 failed（`model_artifact_not_found`） |
| 断言增强 | `service`/`session` 校验 `usage`、`ttft_ms`/`prefill_tps`、JSON 结构化输出 `{"answer": 5}`、同 `request_id` 幂等重放、temperature=0 内容可复现 |

### 2026-11-03 新产物直接登记 → 加载 → 对话（`app` stage，Windows）

| 检查 | 结果 |
| --- | --- |
| 环境 | Windows 11，便携版 `0.4.0+dev.867b54f2.modified`，服务 `127.0.0.1:8090`，RTX 4060 |
| 登记 | `tmp` 目录登记成功；新产物得到 `qwen35-0.8b-nint4-source-rerun-20261103`（id `d5f2f6a4...`，484,845,334 字节，340 记录，`complete/loadable=true`） |
| 加载 | `POST /api/v1/models/load`（`artifact_uri=mfq://d5f2f6a4...`）→ job `succeeded`，实例 `38192768...`，2.0 秒，状态 `ready`、设备 `cuda:0` |
| 对话（OpenAI） | `/v1/chat/completions`：回答 `5`、`finish_reason=stop`，513ms |
| 对话（会话接口） | `/api/v1/sessions/.../responses`：`completed`、回答 `5`、177ms |
| 重复执行 | 再次运行时识别"模型已加载"并复用实例，全部 PASS |

### 2026-11-03 WSL2 Ubuntu（Linux 侧）验证

| 检查 | 结果 |
| --- | --- |
| 环境 | WSL2 Ubuntu，Linux 6.18.33.2-microsoft-standard-WSL2 x86_64，Python 3.13.18，CPU 版 PyTorch |
| `r0` 量化链路 | **PASS**：`mfq quantize` 1.6 秒，产物 2441 字节、4 条记录、2 个内嵌资源 |
| 跨系统一致性 | Linux 产物 SHA-256 `d0dea5dc...` 与 Windows 同 fixture 产物**逐字节相同** |
| `artifact` 真实产物 | **PASS**：读取 Windows 侧 484,845,334 字节容器，SHA、340 条记录、dtypes、内嵌 tokenizer 全部一致（经 `/mnt/d` 访问） |
| GPU | WSL 可直通 RTX 4060（`nvidia-smi` 可见），但 R0 不需要 GPU |

### 2026-11-06 WSL2 早期复跑（仅基础阶段）

| 检查 | 结果 |
| --- | --- |
| 环境 | WSL2 Ubuntu（内核 6.18.33.2），Python 3.13.18，CPU 版 PyTorch 2.15.0+cpu；当时尚未配置 `nvcc` |
| `r0` 量化链路 | **PASS**（2.53 秒）：产物 2441 字节、4 条记录、2 个内嵌资源 |
| `artifact` 真实产物 | **PASS**（2.81 秒）：经 `/mnt/d` 读取 484,845,334 字节容器，340 记录、`qwen3_5-hf-mfq-nint-recipe`、内嵌 tokenizer 全部通过；SHA-256 `26ed9eee...` |
| 跨系统一致性 | Linux 产物 SHA-256 `d0dea5dc...` 与 Windows 同 fixture 逐字节相同 |
| 服务类 stage | Linux 无推理运行时：`app` / `service` / `session` / `stream` / `multi_turn` / `unload_switch` / `negative` 全部记为 `skip`（服务不可达），整体 exit 0 |
| 未覆盖 | `service` 及之后的推理链路需要 Linux 运行时；本机 WSL 无 `nvcc`，未做源码级 CUDA 构建 |

注意：`~/mq-e2e` 是仓库子集快照（无 `.git`），报告里 `repo_head=None`；用 `cp`/rsync 从 D 盘同步最新的 `mfq/` 与 `tests/e2e/`。

注意事项：
- WSL 空闲重启后 `/tmp` 会被 systemd 清理，工作目录用 `$HOME` 下的路径更稳妥。
- Linux 侧安装 CPU 版 PyTorch 即可跑 R0；CUDA 相关依赖体积大，非必需。
- 服务类 stage 在服务不可达时记 `skip`、整体 exit 0；加 `--require-service` 才会视为失败。
- 源码级服务验证需要在 Linux 内自编 CUDA 运行时（`mfq build --backend cuda` + `mfq serve`），仍不等同安装包验收。

### 2026-11-06 最新 master 基线全链复跑（WSL2）

- 源码基线：上游 `master` 提交 `b06a0b73`；测试分支在该基线上重放测试提交。WSL 测试副本不含 `.git`，故 JSON 报告的 `repo_head` 为 `null`，以此处记录的提交为准。
- 原样构建在 `cpp_runtime/backends/cuda/kernels/kv_hot_cache.cu` 失败：`batch_attention_store()` 调用 `append_exclusive_decode()` 时少传最后的 `DeviceInputError*` 参数。测试分支补齐该一处参数后，CUDA 12.4 / GCC 13 的运行时构建成功。因此本次完整服务复跑验证的是 **`b06a0b73` 加此修复**，不能称为未修改的上游 `master` 已构建通过。
- 新服务使用刚构建的 `mfq-runtime`，在空数据目录中启动，不预加载模型；`app` 阶段将真实 `.mfq` 文件所在目录登记、首次加载到 `cuda:0`，然后执行对话。
- `r0`、`artifact`、`app`、`service`、`session`、`stream`、`multi_turn`、`unload_switch`、`negative` 共 9 阶段、64 个检查点：**64 pass / 0 fail / 0 skip**，耗时 33.51 秒。命令传入 `--require-service`，没有把服务不可达计为通过。
- 复跑报告保存在 `tmp/e2e-latest-master-full-20261106.json`（本机临时文件，不随 PR 提交）。本次是 Linux 源码级服务验证，Windows 便携版与图形界面的记录见前述条目。

## 模型与校验清单

测试用的派生 `.mfq` 产物体积大，不随仓库提交；其大小、SHA-256、记录数、dtypes 与内嵌资产固定在
[`docs/test-models/derived-artifacts.lock.json`](../../docs/test-models/derived-artifacts.lock.json)。
复跑前先校验产物，再执行 `artifact` stage：

```sh
sha256sum qwen35-0.8b-nint4-source-current.mfq qwen35-0.8b-nint4-source-rerun-20261103.mfq
# 期望两者均为 26ed9eee9af0f38232c404984f1b8caa2f31896735ab1b8b7d5b5a3fdee9a9d2

python tests/e2e/app_regression.py --stage artifact \
  --artifact <path>/qwen35-0.8b-nint4-source-rerun-20261103.mfq \
  --expect-sha256 26ed9eee9af0f38232c404984f1b8caa2f31896735ab1b8b7d5b5a3fdee9a9d2 \
  --expect-records 340 --expect-arch qwen3_5-hf-mfq-nint-recipe
```

源模型（HF 权重）的逐文件校验见
[`docs/test-models/modelscope-qwen3.5-0.8b.lock.json`](../../docs/test-models/modelscope-qwen3.5-0.8b.lock.json)。

## CI 矩阵建议

| 触发 | 平台 | stage |
| --- | --- | --- |
| 每个 PR | Linux / Windows（无 GPU） | `r0` |
| 定时或发布前 | Windows + NVIDIA、或 macOS Apple silicon | `artifact` + `app` + `service` + `session` + `stream` + `multi_turn` + `unload_switch` + `negative` |

无 GPU 的 runner 上必须显式 `skip`，并通过退出码区分"跳过"与"通过"。
