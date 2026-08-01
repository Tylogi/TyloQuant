# MoE 公开 Observation 数据索引

更新日期：2026-07-23

本索引记录所有公开来源，不按发布者分组。只有包含逐层逐专家数值的文件才可直接用于
Expert-Wise 精度分配；剪枝模型、专家保留数量和评测结果只能作为旁证。

## DeepSeek-V4-Flash

### 可直接使用的数据

来源：

- [REAP25 10K Balanced](https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP25-REAPDataset10K-BalancedWithKO-DS4)，revision `654caec4cfe76876f40faca6f9ea01ec8a1952fa`
- [REAP50 10K Balanced](https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP50-REAPDataset10K-BalancedWithKO-DS4)，revision `7b86e8b7092de195f007029ec2cd4b9db3567aa8`

校准与结构：

| 项目 | 数值 |
|---|---:|
| 样本 | 10,000 |
| 英文 / 韩文 | 5,000 / 5,000 |
| 领域 | 8 |
| prompt token | 27,592,731 |
| routed-expert selections | 7,118,924,598 |
| context 上限 | 4,096 |
| seed | 42 |
| MoE 层 | 43 |
| routed experts / layer | 256 |
| active experts / token | 6 |
| 指标 | `activation_energy_sum2` |

CSV 共 11,008 行，即 `43 × 256`。层 0–2 为 hash routing 并保留全部专家，层 3–42
参与 REAP。字段包括路由次数、selection share、总 REAP、gate/up energy 和 down energy。
解析结果验证了：

```text
reap = gate_up_energy + down_energy
```

相对误差最大值为 `1.25e-9`。REAP25 与 REAP50 的 observation 数值完全一致，区别仅在
`kept`、`pruned` 和 `new_expert_id`。

本地文件：

```text
data/reap/deepseek-v4-flash/reap_dataset_10k_balanced_seed42_reap25_experts.csv
SHA256 7FE56B6501820582CC2E8735A0A95D4A4D9B1E20C9AD14166E1EF4D91F65BA0B

data/reap/deepseek-v4-flash/reap_dataset_10k_balanced_seed42_reap50_experts.csv
SHA256 13A37B5DEB5DC46D39B599D5588D8C54B2152C320A01262871660E3470A011F3
```

### 集中度

下表统计可裁剪的 40 层，数值为逐层中位数。Top-26 对应 10.16% 专家，Top-64 对应
25% 专家；每项都按该指标自身排序。

| 信号 | Top-26 覆盖 | Top-64 覆盖 | 有效专家数 | 覆盖 80% 所需专家 |
|---|---:|---:|---:|---:|
| 路由次数 | 34.96% | 58.62% | 168.4 | 120.5 |
| 总 REAP | 35.70% | 59.32% | 166.4 | 119.0 |
| gate/up energy | 34.98% | 58.61% | 168.4 | 120.5 |
| down energy | **57.42%** | **80.59%** | **84.5** | **62.5** |

路由次数与总 REAP 的逐层 Spearman 中位数为 `0.9979`。频率可近似总 REAP 排序；
它不能替代 projection-wise energy，因为 down 的集中度显著更高。

末三层尤其尖锐：

| 层 | Top-26 总 REAP | Top-64 总 REAP | Top-26 down | Top-64 down |
|---:|---:|---:|---:|---:|
| 40 | 60.36% | 75.16% | 96.75% | 98.57% |
| 41 | 69.93% | 81.26% | 97.50% | 98.95% |
| 42 | 88.78% | 93.67% | 97.97% | 99.38% |

EW 含义：

- Top-64 高精度只占 routed-expert 存储的 25%，却覆盖 58.62% 的实际路由和 80.59%
  的 down energy。
- gate/up 与 down 应独立分配精度。共用一个 expert profile 会丢掉 down 的集中度优势。
- DSV4 的整体路由集中度中等，后段 down projection 极适合 EW。

### 其他公开数据

[0xSero v1](https://huggingface.co/datasets/0xSero/deepseek-v4-flash-reap-observations-v1)
及其 [mateowilliam 镜像](https://huggingface.co/datasets/mateowilliam/deepseek-v4-flash-reap-observations-v1)
记录逐样本 Top-6 专家 ID 和名次。公开快照完成 9,315 / 24,576 样本，缺少 gate
probability、专家输出范数和 projection energy。它适合研究领域间路由变化，不能单独作为
完整的 REAP/EW 重要性。

## MiniMax-M3

### 当前公开程度

公开仓库中没有找到逐层逐专家 saliency 数组。已检查 Hugging Face 上约 100 个
MiniMax-M3 模型仓库、所有名称含 `reap` 的数据集，以及 JANGQ 的公开实现。

可验证的资料分为两类：

1. observer 配置与实现；
2. 剪枝后的专家数量和可从权重恢复的保留集合。

这些资料不能计算 Top-N saliency 覆盖率，也不能估算 EW 的平均使用 bpw。

### Observer 配置

[bullerwins/MiniMax-M3-REAP25-MXFP8](https://huggingface.co/bullerwins/MiniMax-M3-REAP25-MXFP8)
发布了 `reap_args.yaml`：

- 五类来源：Evol CodeAlpaca、xLAM function calling、Open-R1 code、SWE-smith tool
  trajectory、Spanish coding agent；
- `batch_size=1`，`batches_per_category=1024`；
- `model_max_length=2048`，seed 42；
- router weight 重新归一化；
- 输出文件名应为
  `observations_minimax-m3-mxfp8_reap25_len2048_bs1_seed_42.pt`。

该 `.pt` 没有发布，固定 revision 上直接请求返回 404。本地保存的配置：

```text
data/reap/minimax-m3/bullerwins-reap25-reap_args.yaml
SHA256 C6CC9979CF536EAB2454A3FAAF369C2FEBA3711502406CFE0CD429CECCCCFA54
```

JANGQ 的公开 `reap_profile.py` 定义：

```text
saliency[layer, expert] =
    sum over routed tokens of gate_weight * ||expert_output||_2
```

输出为 `saliency[L,E]`、`count[L,E]` 和 `layer_ids[L]`。MiniMax-M3 有 57 个 MoE
层、每层 128 个 routed experts、每 token 激活 4 个。

### 已发布保留规模

| 变体 | 每层保留 / 128 | 裁剪比例 |
|---|---:|---:|
| JANGQ REAP22 Coder | 100 | 21.88% |
| SparkArena / Bullerwins REAP25 | 96 | 25.00% |
| JANGQ REAP32 Coder | 87 | 32.03% |
| JANGQ REAP40 | 77 | 39.84% |
| Osaurus REAP45 | 70 | 45.31% |
| SparkArena REAP50 | 64 | 50.00% |

不同发布者使用的校准集和 observer 设置不同，不能把这些阈值拼成同一条 saliency
曲线。JANGQ/Osaurus 的 Coder 系列使用 Vera/GSM8K 与额外的 coding/math expert
保护规则，因此保留集合也不等价于纯粹按全局 saliency 截断。

## 使用规则

1. EW 输入优先使用逐 projection energy；只有频率时标记为 route-only prior。
2. 每个来源固定 repo revision、文件 SHA256、样本数、token 数、top-k、router
   renormalization 和 shared-expert 规则。
3. 不从剪枝后模型的质量表现反推 Top-N saliency 覆盖率。
4. 外部 observation 用于精度候选分配，最终仍以同一模型、独立验证集上的 logits KL
   或 NTP CE 验证。
