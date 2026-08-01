# 0xSero 公开资源索引

更新日期：2026-07-19

用途：跟踪 0xSero 发布的 MoE 专家观测、校准数据、REAP 模型与实现资料，优先记录能服务 MFQ Expert-Wise 精度分配的内容。

## 总入口

- [Hugging Face 主页](https://huggingface.co/0xSero)：当前 API 可见 73 个模型、21 个数据集。
- [全部 Collections](https://huggingface.co/0xSero/collections)：Qwen、GLM、DeepSeek、MiniMax、Nemotron、Gemma、Trinity 等模型族。
- [Observations & Calibration](https://huggingface.co/collections/0xSero/datasets-observations-and-calibration)：专家观测、校准集和训练数据的集中入口。
- [Qwen — REAP](https://huggingface.co/collections/0xSero/qwen-reap)：Qwen3.5、Qwen3.6 和 Qwen3-Coder 的 REAP/量化模型。
- [Proven REAPs](https://huggingface.co/collections/0xSero/proven-reaps)：已有公开使用量的 REAP 模型集合，可作端到端质量与速度基线。
- [GitHub](https://github.com/0xSero) / [Sybil Solutions](https://sybilsolutions.ai)
- [REAP 论文](https://arxiv.org/abs/2510.13999) / [上游代码](https://github.com/CerebrasResearch/reap) / [PR17](https://github.com/CerebrasResearch/reap/pull/17)
- [Qwen3.5 observer 分支](https://github.com/janmts/reap/tree/qwen3.5-support) / [observer.py](https://github.com/janmts/reap/blob/qwen3.5-support/src/reap/observer.py)

清单异常：Qwen3.6-35B-A3B 观测仓库的网页和文件树可以公开访问，但当前 `author=0xSero` 数据集 API 不返回它。自动同步时要保留这条手工索引。

## 价值分级

- **A**：已有逐层逐专家数据，可以直接形成 Expert-Wise 先验。
- **B**：数据丰富，需要先确认 manifest、抽取 schema 或选择性下载。
- **C**：校准、评估或实现参考，不能直接给出专家精度。
- **D**：占位仓库或缺少可复用数据。

## 优先资源

| 级别 | 资源 | 规模 | 内容 | 对 MFQ 的用途 |
|---|---|---:|---|---|
| A | [Qwen3.6-35B-A3B observations](https://huggingface.co/datasets/0xSero/qwen3.6-35b-a3b-reap-observations) | 79.5 MB | 40 层 × 256 专家、5,000 样本的 layerwise observer | 当前 Qwen3.6-35B-A3B Expert-Wise 配精度的首选公开先验 |
| A/B | [Nemotron Super artifacts](https://huggingface.co/datasets/0xSero/nemotron-super-reap-artifacts-draft) | 12.6 MB | rankings、heatmap、预算扫描、residency plan、压缩摘要 | 研究“专家重要性 + 显存驻留 + 总预算分配”的完整范例 |
| B | [Kimi-K2.6 observations v1](https://huggingface.co/datasets/0xSero/kimi-k2.6-reap-observations-v1) | 22.65 GB | 分层/分域观测、observer state、REAP 指标 | schema 完整，适合迁移 observer 和分域统计；应选择性下载 |
| B | [Kimi-K2.6 observations v4](https://huggingface.co/datasets/0xSero/kimi-k2.6-reap-observations-v4) | 72.2 MB | sample-level saliency，主文件约 64.4 MB | 测量不同样本、领域下专家排序的稳定性 |
| B | [Qwen3.5 layerwise observations](https://huggingface.co/datasets/0xSero/qwen35-reap-layerwise-observations) | 44.58 GB | 多模型、多分片、多次运行的 expert table/state/summary | Qwen 系列参考库；名称覆盖面很宽，使用前必须按 manifest 确认模型与 revision |
| B | [GLM-5 layerwise observations](https://huggingface.co/datasets/0xSero/glm5-layerwise-reap-observations) | 11.70 GB | observer state 和大量 group/block 指标 | 验证 Expert-Wise 分配对另一类大 MoE 的迁移性 |
| B | [MiniMax-M2.7 observations](https://huggingface.co/datasets/0xSero/minimax-m2.7-observations) | 4.09 GB | 全局及 category observer state、分组指标 | 比较全局统计与按领域统计的精度差异 |
| B | [DeepSeek-V4-Flash observations](https://huggingface.co/datasets/0xSero/deepseek-v4-flash-reap-observations-v1) | 3.03 GB | 约 3.01 GB 的 sample salience 及 smoke/full 产物 | sample-level 重要性研究；说明文档较少，先核对生成配置 |

## Qwen3.6-35B-A3B 观测详情

路径：`observations/qwen36-35b-a3b-5k-layerwise-v1/`

| 文件 | 大小 | 内容 |
|---|---:|---|
| `qwen36-35b-a3b-5k-layerwise-v1-expert-table.jsonl` | 1.51 MB | 10,240 行，即 40 个 MoE 层 × 256 个 routed experts |
| `qwen36-35b-a3b-5k-layerwise-v1-observer-state.pt` | 21.53 MB | 完整 observer 状态；文件为 PyTorch pickle，按外部可执行输入处理 |
| `qwen36-35b-a3b-5k-layerwise-v1-observer-summary.json` | 1.72 MB | 分层摘要 |
| `qwen36-35b-a3b-5k-layerwise-v1-manifest.json` | 小文件 | 模型、样本、token 和运行配置 |
| calibration JSONL | 11.06 MB | 本次观测使用的 5,000 条输入 |

已核对的统计：

- manifest 总 token 数为 1,998,561；observer 计入 1,984,839 个有效 token。
- 最大长度 8,192；共 272 个 packed sequences/batches。
- 数据覆盖 coding、reasoning、general、multilingual、communication、tool、structured 七类。
- 路由为 top-8；模型另有 `shared_expert_intermediate_size=512` 的 shared expert。shared expert 始终执行且不经过 router，因此专家表只统计 256 个 routed experts。MFQ 将 shared expert 视为普通 FFN 张量，不使用路由频率或 routed-expert EAN 先验。
- 每个专家提供 `expert_frequency`、`ean_sum`、`ean_mean`、`weighted_ean_sum`、`weighted_expert_frequency_sum`、`max_activations` 和 `reap`。
- 数据没有原始逐 token router logits、gate probability 或 activation tensor。
- 仓库未声明数据许可，公开结果用于发布模型或论文前需再次确认授权范围。

推荐先验：

```text
exposure[layer, expert] = weighted_ean_sum[layer, expert] / total_tokens[layer]
```

它保留路由频率、router 权重和专家输出范数。它只表示观测分布上的暴露量。精度分配还需要每个候选格式的量化误差 `D[layer, expert, precision]`，最终以 logits KL、NTP CE 或专家输出 MSE 验证。

已知陷阱：公开 state 中的 `pairwise_expert_frequency[i,j]` 等于两个专家边际频次之和，缺少真实共激活信息，禁止用于专家共驻留、专家配对或 grouped GEMM 调度。

### 本地 Expert-Wise 接入记录

2026-07-20 完成首个 Qwen3.6-35B-A3B REAP Expert-Wise MFQ。使用的本地源模型：

```text
HF BF16 Safetensors:
models/Qwen3.6-35B-A3B

BF16 GGUF 目录:
models/Qwen3.6-35B-A3B-BF16.gguf

BF16 GGUF 文件:
models/Qwen3.6-35B-A3B-BF16.gguf
```

匿名下载原仓库时使用其公开镜像
`mateowilliam/qwen3.6-35b-a3b-reap-observations@c0fbef14f8bedccc42048cfdf3edce0e58e35d70`；
该 revision 的 commit message 明确记录其复制自 0xSero 原仓库。expert table SHA256 为
`3c8c7ede96bf91b63c11089874badac799db93864bbbe013b9c4825a21eaf632`，外部 `.pt` pickle 未加载。

首轮 10,240 个 routed experts 的 profile 分布为 NINT4/5/6/8 =
`2317/5693/2213/17`。gate_up 与 down 对同一 expert 共用 profile；shared expert 不使用 REAP，按
UD Q4_K_M recipe 转换。实际 expert 存储预算为全 NINT5 基线的 `99.999343%`，REAP 加权权重
重建代理损失下降 `22.6743%`。该实验为高预算历史对照，产物已归档到：

成品：

```text
artifacts/Qwen3.6-35B-A3B-REAP-EW-NINT5.mfq
24,359,451,497 bytes
SHA256 AFDA301468AF8258F9DFA62A03B0760F6112EC4880A771F6ADC040CBC400079F
```

当前 UD 同 bpw 版本使用完整 41-block 配方。非 expert 张量按 UD 类型映射；主模型 40 层的
gate_up/down 独立做 REAP 分配；缺少 REAP 观测的 MTP expert 保持 UD 的 Q4 gate_up/Q6 down。

| 部分 | NINT4 | NINT5 | NINT6 | NINT8 |
|---|---:|---:|---:|---:|
| gate_up experts | 8,917 | 1,301 | 22 | 0 |
| down experts | 6,160 | 3,966 | 112 | 2 |

20,480 个独立选择组使用 expert 预算的 `99.999649%`。相对 UD 逐层映射基线，REAP 加权权重
NMSE 代理损失下降 `21.4169%`，同时 expert payload 减少 `2.441%`，用于抵消 NINT 与 GGUF 的
格式开销差异。2,851 个 expert 的 gate_up/down 最终使用不同 profile。

```text
当前产物:
artifacts/Qwen3.6-35B-A3B-REAP-EW-UD-BPW.mfq

大小: 21,712,333,861 bytes
bpw: 4.8921966122
UD bpw: 4.8922131224
SHA256: E1B53EB67BCA2A137D9F60365ECAE9224ABB4F157DE88FDBCFCC1649846665CB
```

文件比 UD 小 `73,275 bytes`，bpw 差 `0.00001651`。742 个 tensor records、19 个 MTP 张量和
82 个 NINTM 均已审计；实际使用的七种 gate/down profile 各抽一个 expert，与直接 CUDA 量化
逐字段一致。代理损失不代表 logits KL，端到端质量仍需独立评测。

## 其他观测与压缩资料

### Nemotron Super

- 模型：`NVIDIA-Nemotron-3-Super-120B-A12B-BF16`。
- 结构：88 blocks，其中 40 Mamba、40 MoE、8 attention；每个 MoE 层 512 experts，top-22。
- long lane 为 819,200 tokens，short lane 为 321,560 tokens，合并后 1,140,760 tokens。
- 资料含逐专家 rankings、frequency/delta heatmap、budget sweep、residency simulation 和 25%/50% REAP 压缩摘要。
- 许可受 NVIDIA Open Model License 约束。

### Kimi-K2.6

- v1 使用 REAP v1 与 structured-output 校准数据，最大长度 16K、batch 8、renormalized router、layerwise observer。
- v1 含 `expert_frequency`、EAN、weighted EAN、REAP 等字段；全仓 22.65 GB，不应直接下载全部文件。
- v4 的 `runs-v4/full/samples-saliency.jsonl` 约 64.4 MB，适合研究样本之间的专家重要性方差。

### Qwen3.5 / GLM / MiniMax / DeepSeek

- Qwen3.5 仓库包含 Qwen Coder Next、Qwen 122B 等不同运行；禁止仅凭仓库名混合统计。
- GLM-5 仓库有约 40 MB 的 `observer-state.pt`，其余大量文件可按需要取单层或单组。
- MiniMax-M2.7 同时提供全局和 category observer，适合检验领域条件化先验是否优于全局均值。
- DeepSeek-V4-Flash 以 sample salience 为主，当前公开元数据不足以直接复现实验配置。

## 校准与评估数据

| 级别 | 资源 | 内容 | 使用建议 |
|---|---|---|---|
| C | [REAP calibration data v1](https://huggingface.co/datasets/0xSero/reap-calibration-data-v1) | 原版 23,088 条、filtered v2 20,980 条，覆盖 function calling、agentic、cybersecurity、coding、reasoning、math、CUDA、terminal、long context、science | Apache-2.0；可作通用校准对照，模型自身生成的 trace 仍作为 MFQ 主训练数据 |
| C | [Structured outputs calibration v1](https://huggingface.co/datasets/0xSero/structured-outputs-calibration-v1) | 430 条 strict JSON、JSON schema、Mermaid 与 fenced Mermaid | MIT；用于增强工具调用和结构化输出覆盖 |
| C | [GLM-4.7 calibration 1360](https://huggingface.co/datasets/0xSero/glm47-calibration-1360) | GLM-4.7 定向校准集 | 参考其样本组织和领域配比 |
| C | [GLM-4.7 code/function calibration](https://huggingface.co/datasets/0xSero/glm47-reap-calibration-code-func) | code + function calling | 工具和代码专项对照 |
| C | [GLM-4.7 mixed calibration](https://huggingface.co/datasets/0xSero/glm47-reap-calibration-mix) | mixed-domain | 领域混合策略参考 |
| C | [GLM-4.7 calibration v2](https://huggingface.co/datasets/0xSero/glm47-reap-calibration-v2) / [v3](https://huggingface.co/datasets/0xSero/glm47-reap-calibration-v3) | 约 1.36K 条的迭代版本 | 比较版本差异后再采用 |
| C | [GLM Pile640 v2](https://huggingface.co/datasets/0xSero/glm-calibration-pile640-v2) | Pile 风格小型校准集 | 传统文本分布对照 |
| C | [GLM-5 REAP50 TuneComp SFT](https://huggingface.co/datasets/0xSero/glm5-reap50-tunecomp-sft) | REAP50 后续 SFT 数据 | 研究剪枝后恢复训练，不用于无训练量化基线 |
| C | [GLM vision SFT mix](https://huggingface.co/datasets/0xSero/glm-vision-sft-mix) | 视觉 SFT 混合数据 | 多模态扩展时再评估 |
| C | [MiniMax-M2.1 REAP observations](https://huggingface.co/datasets/0xSero/minimax-m2.1-reap-observations) | 96 组 REAP-20/30/40/50、温度与 prompt stress-test 结果 | 名称容易误导；它是评估结果集，缺少原始逐专家观测 |
| C | [Step-3.7 prune TerminalBench artifacts](https://huggingface.co/datasets/0xSero/step37-prune-terminal-bench-artifacts) | TerminalBench 轨迹与结果 | 剪枝模型端到端评估参考 |
| D | [Gemma MoE REAP dataset](https://huggingface.co/datasets/0xSero/gemma-moe-reap) | 当前只有 README 和 `.gitattributes` | 暂无可复用观测数据 |

另有 [MiniMax REAP observations](https://huggingface.co/datasets/0xSero/minimax-reap-observations)，使用前需核对它与 M2.7 专用仓库的模型、revision 和 schema。

## 模型基线

- [Qwen3.6-28B](https://huggingface.co/0xSero/Qwen3.6-28B)：由 Qwen3.6-35B-A3B 做 REAP 专家裁剪，可检查公开 expert ranking 的实际用途。
- [Qwen3.6-28B-GGUF](https://huggingface.co/0xSero/Qwen3.6-28B-GGUF) / [Qwen3.6-35B-GGUF](https://huggingface.co/0xSero/Qwen3.6-35B-GGUF)：端到端 GGUF 基线。
- [Qwen3.5-99B-GGUF](https://huggingface.co/0xSero/Qwen3.5-99B-GGUF)：REAP + GGUF 的成熟发布样例，可检查 recipe、tensor type 与模型卡评估方式。
- 其他模型统一从 [Collections](https://huggingface.co/0xSero/collections) 进入，避免记录很快失效的完整模型清单。

## MFQ 接入规则

1. 每份外部统计记录 `repo_id`、模型 revision、数据 revision、样本数、有效 token 数、最大长度、chat template、router 是否 renormalize、top-k 和 shared expert 规则。
2. 下载前读取 manifest 和仓库树；大仓只取 expert table、observer state、summary 与必要校准文本。
3. 外部 saliency 仅作为候选先验。每个专家每种精度的损失应在 MFQ 自己的 trace 上测量，并用独立验证集的 logits KL/CE 决定最终分配。
4. 分别保存全局、领域条件化和样本级统计，避免先求均值后丢失方差信息。
5. 未声明 license 的数据只用于内部研究；公开衍生物前重新检查仓库许可和上游模型条款。
6. 保存下载时的 commit hash，后续结果表必须能定位到同一份外部数据。
7. 优先读取 `expert-table.jsonl` 和 `observer-summary.json`。外部 `.pt` observer state 含 pickle，只在隔离环境中加载，禁止直接进入长期运行的量化或推理进程。
