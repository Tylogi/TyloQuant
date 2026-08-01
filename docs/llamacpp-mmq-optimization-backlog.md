# llama.cpp 对位优化待办

本文只记录尚未进入生产路径的优化项和验证门槛。已合并功能以
`developer-architecture-and-operator-reference.md` 为准。

## 当前边界

- `.mfq` 张量结构保持不变，旧模型必须继续直接加载。
- 默认生产路径不启用 KLD 专用 NINT8-1 激活量化。
- NINT8-1 必须复现 llama.cpp 的 Q8_1 分块、scale、sum、输入与输出精度后，
  才能用于权重格式公平对照。
- Full-dequant + cuBLAS 仍可作为大 M 的生产路径；替代内核必须用真实张量证明更快。
- TP 和 Decode 多流不得依赖新的磁盘张量布局。

## 优先级

| 优先级 | 项目 | 当前状态 | 合并门槛 |
|---:|---|---|---|
| 1 | 大 M 压缩权重 MMQ | 待开发 | M=512/2048 的真实张量数值对齐，且快于 full-dequant + cuBLAS |
| 2 | 按设备与 M/N/K 动态选择 M tile | 待开发 | launch geometry 与 llama.cpp 对位，覆盖各量化格式 |
| 3 | Dequant workspace 环形复用 | 部分已有 | 峰值显存下降，连续层无同步与生命周期错误 |
| 4 | 运行时派生布局缓存 | 待开发 | 不改 `.mfq`，首次转换与后续复用分别计时 |
| 5 | Decode 独立投影多流 | 已合并 | 仅 M=1；串行、并行、CUDA Graph replay 数值一致 |
| 6 | 通用 Tensor Parallel | 已合并 | Dense 输出/输入切分、FFN 本地下投影、MoE 成对行均通过 |
| 7 | 模型专用融合 | 延后 | 针对具体模型单独设计，不进入通用分发 |
| 8 | 自动化性能回归 | 待补 | 同一矩阵、同一激活、同一精度和同一同步边界 |

## NINT8-1 公平对照

llama.cpp MMQ 激活格式的一个 block 覆盖 128 个值，并保存两组 64 值的
scale 与 sum。以下任一差异都会使 KLD 对照失效：

- 量化前把 FP32 激活提前舍入到 FP16；
- 只保存 int8 值或整行单一 scale；
- scale、sum、block 排列或尾部 padding 不一致；
- accumulator 或最终输出比 llama.cpp 更早写回 FP16；
- Dense 与 routed MoE 走了不同的激活量化定义。

修复完成前，命令行默认必须保持 NINT8-1 禁用。验证顺序为单矩阵逐元素、
前三个 KLD chunk、完整数据集；每一步同时对比普通生产路径和 llama.cpp。

## 大 M 对位几何

参考实现需要记录：

```text
grid.x = ceil(N / mmq_y)
grid.y = ceil(M / mmq_x)
shared activation = mmq_x * sizeof(block_q8_1_mmq)
shared weights = mmq_y * tile_stride
```

MFQ 候选实现必须报告 `M/N/K`、格式、group size、tile、split-K、SM 数、
共享内存、寄存器、occupancy、输入输出 dtype 和同步边界。只比较算术主体延迟
不能作为合并依据。

## 验证矩阵

| 维度 | 必测值 |
|---|---|
| M | 1、8、9、128、256、512、2048 |
| 权重 | NINT2/3/4/5/6、NINT8-0、NVQ/NPQ、NEPQ |
| 输出轴 | 单张量、QKV/Gate-Up 分组、LM head |
| MoE | 单专家、重复 expert id、全 expert、不同 top-k |
| 执行 | eager、并行 stream、CUDA Graph capture/replay |
| 基线 | 当前生产路径、逐张量 dense reference、llama.cpp |

性能报告至少包含中位数、P95、峰值显存和 H2D/D2D 字节；数值报告至少包含
max-abs、mean-abs、relative L2、same-top 与完整 KLD。

## 已合并部分的约束

Decode 多流只并行彼此独立的投影，父流记录 ready event，分支流完成后由父流
等待 completion event。输出 storage 必须登记父流，避免缓存分配器提前复用。

Tensor Parallel 直接切压缩权重：

- 输出切分在主卡按原顺序拼接；
- 输入切分在 FP32 中归并 partial，再还原输出 dtype；
- Dense Gate/Up 与 Down 使用同一边界，中间激活留在对应设备；
- Routed MoE Gate/Up 保持每个专家的 Gate 行、Up 行成对顺序；
- TP 与 routed expert GPU cache 当前互斥；
- 单设备 CUDA Graph 在 TP 模式下关闭。
