# llama.cpp 开发架构与算子参考（MFQ 对位版）

本文记录 MFQ 当前实际使用的 llama.cpp 对照实现。它同时是源码索引、算子表、
实验条件表和 MFQ 对位说明。修改基线 checkout、GGUF recipe、KLD reference、
CUDA MMQ/MMVQ、FlashAttention、MoE 或 batch/ubatch 设置时，必须同步更新本文。

<!-- LLAMA_REFERENCE_METADATA_BEGIN -->
- `LLAMA_REFERENCE_HEAD=4963a123c8f3a48f2fc0c8024252e131637ffa06`
- `LLAMA_REFERENCE_ORIGIN_MASTER=25a1d63`
- `LLAMA_MODEL_ARCH_COUNT=132`
- `LLAMA_MODEL_SOURCE_COUNT=136`
- `LLAMA_GGML_OP_COUNT=97`
- `LLAMA_GGML_TYPE_COUNT=34`
- `LLAMA_CUDA_DEFAULT_TU_COUNT=139`
<!-- LLAMA_REFERENCE_METADATA_END -->

最后核对日期：2026-07-29

## 1. 参考版本与审计边界

参考工作树由 `LLAMA_CPP_REFERENCE_DIR` 指定。审计时使用的分支为
`opt/padding-bankconflict`，HEAD 为 `4963a123`。这份 checkout 包含
`origin/master` 之后的 10 个本地提交：

1. `186a694`：mixed precision FlashAttention。
2. `18f5d4c`：Qwen3.5 mixed-KV 路径。
3. `25fe02d`：tiled mixed-KV FlashAttention。
4. `e50c61d`：移除 mixed per-head window 的额外 SWA cache。
5. `8fb941a`：per-head window 使用 grouped SWA cache。
6. `c36c19b`：mixed Q4/F16 attention prefill 优化。
7. `d3a62c2`：mixed attention KV split 调优。
8. `111df44`：mixed-KV runtime trace。
9. `e15479c`：记录 mixed-KV CUDA build artifact。
10. `4963a12`：mixed MMA shared-memory 行 padding，降低 bank conflict。

工作树还有未提交修改：

| 文件 | 当前改动 | 对实验的影响 |
|---|---|---|
| `ggml/src/ggml-cuda/fattn-mixed-vec.cuh` | `QWEN35_MIXED_TARGET_PB` 覆盖 parallel blocks | 改 mixed attention launch geometry |
| `tests/test-backend-ops.cpp` | Qwen3.5 IQ2_XXS 的 M=1…1024 perf shape | 只改测试 |
| `tools/gdn-trace/gdn-trace.cpp` | 精确 token、切片、结果与环境变量 | 改 trace 工件 |
| `tools/perplexity/perplexity.cpp` | KLD、BOS token、chunk offset、`trace_v3` | 直接决定 MFQ 对照数值 |
| `ggml/src/ggml-cuda/mmq.cu` | 当前只检测到行尾差异 | 没有可见算术差异 |

因此，本文中的“本机 llama.cpp 对照”指上述 checkout；“纯上游”指
`origin/master=25a1d63`。两者的 KLD 与 Qwen3.5 mixed-KV 能力不同。

## 2. 分层架构

| 层 | 主要位置 | 责任 |
|---|---|---|
| 公共 API | `include/llama.h` | model/context/batch/memory/sampler C API |
| 模型元数据 | `src/llama-arch.*`, `src/llama-hparams.*` | 架构枚举、GGUF key、超参数 |
| 模型注册 | `src/llama-model.cpp`, `src/models/` | 132 个架构映射、136 个模型源文件 |
| 模型加载 | `src/llama-model-loader.*`, `src/llama-model.cpp` | GGUF、split、mmap、mlock、tensor placement |
| 图构建 | `src/llama-graph.*`, `src/models/*.cpp` | Attention、FFN、MoE、recurrent、MTP |
| context/batch | `src/llama-context.cpp`, `src/llama-batch.*` | batch→ubatch、graph build、输出收集 |
| 模型状态 | `src/llama-memory.*`, `src/llama-kv-cache*` | KV、SWA、hybrid、DSV4、recurrent state |
| GGML IR | `ggml/include/ggml.h`, `ggml/src/ggml*.c` | tensor、97 个 op、34 个 active type |
| backend scheduler | `ggml/src/ggml-backend.cpp` | backend 分配、图切分、copy、allocator |
| CPU backend | `ggml/src/ggml-cpu/` | 全 op 参考、SIMD dot、repack、threadpool |
| CUDA backend | `ggml/src/ggml-cuda/` | MMVQ/MMQ/MMF/cuBLAS、FA、MoE、state |
| 量化 | `src/llama-quant.cpp`, `tools/quantize/` | recipe、tensor category、imatrix、GGUF 输出 |
| imatrix | `tools/imatrix/`, `common/imatrix-loader.*` | activation 二阶统计、MoE 分专家统计 |
| 评测 | `tools/perplexity/`, `tools/llama-bench/` | PPL、KLD、吞吐与延迟 |

顶层 CMake 依次组织 `ggml`、`src`、`common`、`tests`、`examples`、
`tools` 和 `app`。可动态注册的 backend 包括 CPU、BLAS、CUDA、HIP、
Metal、SYCL、Vulkan、CANN、MUSA、OpenCL、OpenVINO、WebGPU、RPC、
VirtGPU、Hexagon、zDNN 和 ZenDNN。最后一个 scheduler backend 必须是 CPU。

## 3. 与 MFQ 的逐项对应

| llama.cpp | MFQ | 对位边界 |
|---|---|---|
| GGUF metadata + tensor directory | `MFQ1` header + record directory | tensor 名、shape、dtype/format 必须逐项对应 |
| split GGUF 自动发现 | split MFQ 需 runtime/脚本显式支持 | llama.cpp 不要求先 merge split GGUF |
| `llm_arch` + `src/models/*.cpp` | native runtime 的 model config 与架构分支 | 模型语义必须对齐 |
| GGML graph | `cpp_runtime/mfq_decode.cpp` 手写执行图 | MFQ 当前不是通用动态图 scheduler |
| `GGML_OP_MUL_MAT` | MFQ dense NINT/NVQ/NPQ matmul | 比较时固定 M/N/K、activation dtype |
| `GGML_OP_MUL_MAT_ID` | MFQ routed expert grouped kernel | expert compaction、排序、scatter 必须对齐 |
| MMVQ | MFQ GEMV/small-M kernel | llama.cpp 阈值为 M≤8，MoE 还按格式/架构缩小 |
| MMQ | MFQ prefill MMQ | llama.cpp 在 Turing+ 对支持量化格式始终可选 |
| Q8_1 activation | MFQ `NINT8-1` KLD-only activation | 只用于公平对照时应复制完整分块语义 |
| FlashAttention | MFQ attention kernels | KV dtype、mask、GQA、head dim、ubatch 都是变量 |
| `ggml_backend_sched` | MFQ device dispatch + expert cache | llama.cpp 只复制当次使用专家，没有 LRU/prefetch |
| llama memory classes | MFQ KV/GDN/HC/cache 对象 | state layout 和清理边界必须相同 |
| `llama-imatrix` | MFQ imatrix/calibration | llama.cpp 统计输入通道 `sum(x²)` |
| `llama-quantize` recipe | MFQ scheme/format assignment | `Q4_K_M` 是混合 recipe，不等于全张量 Q4_K |
| `llama-perplexity` | MFQ historical/optimized KLD evaluator | token、BOS、score range、KLD 方向必须相同 |

## 4. GGUF、split 与 tensor placement

### 4.1 加载过程

`llama_model_loader` 先解析 GGUF metadata 和 tensor directory。split 文件名
采用 `<name>-00001-of-000NN.gguf`，loader 校验 split index、split count、
总 tensor 数，并保存每个 tensor 所属文件、文件内 offset 和大小。

split GGUF 可直接加载。手工 merge 只对不支持 split 的其他 runtime 有意义。

数据访问有三条路径：

1. mmap：平台支持时默认开启；tensor 可直接包装准确的 mmap 区间。
2. async upload：tensor 不可直接映射到目标 backend 时，经 pinned host memory 上传。
3. 普通读取：禁用 mmap 或 direct-I/O 条件不满足时使用。

`use_mmap`、direct I/O、mlock 和 backend buffer 能力共同决定路径。加载结束后，
不再使用的 mmap 前缀与后缀会解除映射。mlock 可作用于 mmap 区间或 host buffer。

### 4.2 GPU layer 的含义

`n_gpu_layers` 控制 tensor 放在哪种 backend buffer；它不直接生成另一套模型图。
scheduler 根据：

- 已分配的输出/view backend；
- weight buffer 所在 backend；
- backend 的 `supports_op`；
- backend 优先级；

确定每个 op 的执行设备。tensor override 可以把单个 tensor 强制放到指定 buffer type。
row-split buffer 只支持 `MUL_MAT`，不支持通用 `MUL_MAT_ID`。

## 5. GGML 图、scheduler 与内存生命周期

### 5.1 图执行

模型代码创建 GGML tensor 和 op，context 把 batch 切成 ubatch，为每个 ubatch：

1. memory context 选择/申请 KV 或 recurrent state 位置；
2. 构建模型图；
3. 写入 token、position、mask、RoPE 与 cache 索引；
4. scheduler 分配 backend；
5. graph allocator 复用或重新 reserve buffer；
6. backend split 依次 copy 输入并异步执行；
7. 根据输出请求复制 logits/embedding。

图 shape、backend placement 或 buffer layout 变化时，allocator 可能重新 reserve。
同步 API 最终等待全部 backend；async API 仍受 split 内部同步点约束。

### 5.2 backend 分配与 copy ring

`ggml_backend_sched` 的分配优先级是：

1. 预分配 destination；
2. view source；
3. weight buffer；
4. 从 GPU 节点向上下游扩展；
5. 其余节点放到首个支持该 op 的 backend；
6. 在 backend 边界插入 tensor copy。

parallel scheduler 使用 4 份 copy (`GGML_SCHED_MAX_COPIES`)；非 parallel 使用 1 份。
event 负责 copy buffer 的生命周期。当前 `compute_splits` 在每个 split 开始和
copy 结束后都调用 `ggml_backend_synchronize(split_backend)`，因此它不是完整的
跨层 copy/compute overlap scheduler。

### 5.3 llama.cpp 自带的 MoE CPU/GPU offload

当 host WEIGHTS 作为 split 首节点 `MUL_MAT_ID` 的 `src[0]` 时，scheduler：

1. 把 ids 读回 host；
2. 用 bitset 标记当前 ubatch 使用的 expert；
3. 把连续 expert id 合成一次 copy；
4. 每段末尾最多额外复制 512 bytes，保证 MMQ padding 有定义；
5. 只把这些 expert slice 写入 GPU copy tensor。

这里没有跨 token LRU、频率预热或下一层预取。ids readback、host/backend
synchronize 仍在关键路径。MFQ 通用异构推理的 cache、REAP 排名、频率预算和
prefetch 属于额外能力。

## 6. batch、ubatch、sequence 与状态

### 6.1 三个独立维度

| 参数 | 作用 |
|---|---|
| `n_ctx` | 单 sequence 的逻辑上下文长度 |
| `n_batch` | 单次 API batch 允许的 token 上限 |
| `n_ubatch` | 每次构图/执行的物理 token 上限 |
| `n_seq_max` | context 可同时管理的 sequence 数 |
| perplexity `n_seq` | `n_batch / chunk_n_ctx` 可并行评测的 chunk 数 |

causal context 中：

```text
n_batch  = min(n_ctx_total, params.n_batch)
n_ubatch = min(n_batch, params.n_ubatch == 0 ? n_batch : params.n_ubatch)
```

`n_ctx=2048, n_batch=2048, n_ubatch=512, n_seq=1` 会生成 4 个 ubatch。
量化 Linear 每次看到 `M=512`。它不等价于一个 `M=2048` 的 MMQ launch。

### 6.2 memory class 选择

| 架构/条件 | memory 实现 |
|---|---|
| embedding/diffusion 类 | 无 autoregressive memory |
| DeepSeek 3.2 | `llama_kv_cache_dsa` |
| 纯 recurrent | `llama_memory_recurrent`，state 为 F32 |
| Qwen3Next/Qwen3.5 hybrid | `llama_memory_hybrid` 或 `llama_memory_hybrid_iswa` |
| Qwen3.5 MTP | 普通 attention cache |
| Gemma4/Gemma3n | layer reuse + `llama_kv_cache_iswa` |
| DeepSeek4 | `llama_kv_cache_dsv4`，支持 raw/HCA/CSA/LID plan |
| 普通 decoder | `llama_kv_cache` |

`offload_kqv` 决定 K/Q/V 与 attention 中间量是否放到 device；
`kv_unified` 决定 sequence 是否共享统一 cache 管理。cache write/copy 是图中的显式 op，
不是隐式副作用。每个 PPL/KLD chunk 必须清空 model memory。

## 7. MFQ 重点模型图

### 7.1 Qwen3.5/Qwen3.6

源码：`src/models/qwen35.cpp`。

full-attention 层：

```text
input
  -> attn RMSNorm
  -> Q projection = Q + sigmoid gate
  -> K/V projection
  -> Q/K RMSNorm
  -> RoPE
  -> KV cache + attention
  -> sigmoid(Q gate) * attention output
  -> output projection
  -> residual
  -> post-attn RMSNorm
  -> dense SwiGLU
  -> residual
```

linear-attention/GDN 层：

```text
input
  -> RMSNorm
  -> wqkv + wqkv_gate
  -> beta sigmoid
  -> alpha + dt -> softplus * A
  -> causal convolution + SiLU
  -> Q/K L2 norm
  -> GATED_DELTA_NET recurrent state
  -> output norm gated by z
  -> output projection
  -> residual + dense SwiGLU
```

本机扩展允许每层/per-KV-head window 与 F16/Q4 mixed KV。环境变量
`QWEN35_MIXED_KV_WINDOW`、`QWEN35_PER_KV_WINDOWS`、
`QWEN35_PER_KV_WINDOWS_BY_LAYER` 和 `QWEN35_FORCE_FULL_ATTN_SWA`
会改变图或 cache plan。MTP tensor 由独立 `graph_mtp` 使用。

### 7.2 Gemma4-26B-A4B

源码：`src/models/gemma4.cpp`。

- SWA 与 full attention 按层交替，部分层共享 KV。
- Q/K 各自 norm，V 有 RMSNorm，再进入 RoPE/cache/attention。
- 某些 alternate attention 可以复用 K 作为 V。
- attention output 有 post norm，再做 residual。
- dense FFN 是 `FFN norm -> parallel GeGLU -> post norm -> residual`。
- expert block 同时计算 shared dense expert 与 routed MoE。
- router 输入来自 `RMS(attn_out) / sqrt(n_embd) * scale`。
- routed 与 shared 分支相加后做 post norm 和 residual。

因此 Gemma4 的 shared expert 不走 `MUL_MAT_ID`；它是普通 dense `MUL_MAT`。
分析 8-bit shared path 时必须检查 dense MMQ/cuBLAS 路由。

### 7.3 DeepSeek4

源码：`src/models/deepseek4.cpp`。

- Hyper-Connection state 通常有 4 个 stream。
- attention 子层：HC pre -> RMS -> MLA/DSA -> HC post。
- FFN 子层：HC pre -> RMS -> routed + shared expert -> HC post。
- MLA 包含 Q low-rank、Q RMS、noPE/RoPE split、KV low-rank。
- KV plan 可为 raw、HCA block compression、CSA overlapping compression 或 LID。
- attention 输出执行 inverse RoPE、grouped low-rank `wo_a`、`wo_b`。
- selected expert table/hash 层、sqrt-softplus gate 和 shared expert 都在模型图内。

### 7.4 GLM DSA

`src/models/models.h` 让 GLM DSA 复用 `llama_model_deepseek2::graph`。
它加载 MLA absorbed Q/KV 与 indexer tensor。DSA 先由 indexer 生成 top-k mask，
把未选位置设为 `-inf`，随后进入普通 masked attention。NextN tensor 在主图中
标记为可缺失/跳过。

## 8. GGML 算子全表

CPU switch 对 97 个 active op 全部有处理；其中 `NONE/RESHAPE/VIEW/PERMUTE/
TRANSPOSE` 是 metadata/no-op。CUDA dispatch 有 83 个入口。表中的“CUDA 是”
只表示存在 dispatch；真正执行还需通过 `ggml_backend_cuda_device_supports_op`
的 dtype、shape、stride 与 buffer 检查。

<!-- LLAMA_GGML_OPS_BEGIN -->
| GGML op | 类别 | CPU | CUDA |
|---|---|---:|---:|
| `GGML_OP_NONE` | metadata | 是 | 是 |
| `GGML_OP_DUP` | 布局/复制 | 是 | 是 |
| `GGML_OP_ADD` | 逐元素 | 是 | 是 |
| `GGML_OP_ADD_ID` | 专家 bias | 是 | 是 |
| `GGML_OP_ADD1` | 标量加 | 是 | 是 |
| `GGML_OP_ACC` | view 累加 | 是 | 是 |
| `GGML_OP_SUB` | 逐元素 | 是 | 是 |
| `GGML_OP_MUL` | 逐元素 | 是 | 是 |
| `GGML_OP_DIV` | 逐元素 | 是 | 是 |
| `GGML_OP_SQR` | 逐元素 | 是 | 是 |
| `GGML_OP_SQRT` | 逐元素 | 是 | 是 |
| `GGML_OP_LOG` | 逐元素 | 是 | 是 |
| `GGML_OP_SIN` | 逐元素 | 是 | 是 |
| `GGML_OP_COS` | 逐元素 | 是 | 是 |
| `GGML_OP_SUM` | 归约 | 是 | 是 |
| `GGML_OP_SUM_ROWS` | 行归约 | 是 | 是 |
| `GGML_OP_CUMSUM` | 前缀和 | 是 | 是 |
| `GGML_OP_MEAN` | 归约 | 是 | 是 |
| `GGML_OP_ARGMAX` | 归约 | 是 | 是 |
| `GGML_OP_COUNT_EQUAL` | 归约 | 是 | 是 |
| `GGML_OP_REPEAT` | 布局 | 是 | 是 |
| `GGML_OP_REPEAT_BACK` | 布局反向 | 是 | 是 |
| `GGML_OP_CONCAT` | 拼接 | 是 | 是 |
| `GGML_OP_SILU_BACK` | 激活反向 | 是 | 是 |
| `GGML_OP_NORM` | LayerNorm | 是 | 是 |
| `GGML_OP_RMS_NORM` | RMSNorm | 是 | 是 |
| `GGML_OP_RMS_NORM_BACK` | RMSNorm 反向 | 是 | 是 |
| `GGML_OP_GROUP_NORM` | GroupNorm | 是 | 是 |
| `GGML_OP_L2_NORM` | L2Norm | 是 | 是 |
| `GGML_OP_MUL_MAT` | 矩阵乘 | 是 | 是 |
| `GGML_OP_MUL_MAT_ID` | routed expert 矩阵乘 | 是 | 是 |
| `GGML_OP_OUT_PROD` | 外积 | 是 | 是 |
| `GGML_OP_SCALE` | 标量乘 | 是 | 是 |
| `GGML_OP_SET` | view 写入 | 是 | 是 |
| `GGML_OP_CPY` | dtype/布局复制 | 是 | 是 |
| `GGML_OP_CONT` | contiguous copy | 是 | 是 |
| `GGML_OP_RESHAPE` | metadata | 是 | 是 |
| `GGML_OP_VIEW` | metadata | 是 | 是 |
| `GGML_OP_PERMUTE` | metadata | 是 | 是 |
| `GGML_OP_TRANSPOSE` | metadata | 是 | 是 |
| `GGML_OP_GET_ROWS` | gather | 是 | 是 |
| `GGML_OP_GET_ROWS_BACK` | scatter-add | 是 | 是 |
| `GGML_OP_SET_ROWS` | scatter set | 是 | 是 |
| `GGML_OP_DIAG` | 对角矩阵 | 是 | 是 |
| `GGML_OP_DIAG_MASK_INF` | attention mask | 是 | 是 |
| `GGML_OP_DIAG_MASK_ZERO` | mask | 是 | 否 |
| `GGML_OP_SOFT_MAX` | softmax | 是 | 是 |
| `GGML_OP_SOFT_MAX_BACK` | softmax 反向 | 是 | 是 |
| `GGML_OP_ROPE` | RoPE | 是 | 是 |
| `GGML_OP_ROPE_BACK` | RoPE 反向 | 是 | 是 |
| `GGML_OP_CLAMP` | 逐元素 | 是 | 是 |
| `GGML_OP_CONV_TRANSPOSE_1D` | 卷积 | 是 | 是 |
| `GGML_OP_IM2COL` | 卷积变换 | 是 | 是 |
| `GGML_OP_IM2COL_BACK` | 卷积反向 | 是 | 否 |
| `GGML_OP_IM2COL_3D` | 3D 卷积变换 | 是 | 是 |
| `GGML_OP_COL2IM_1D` | 卷积变换 | 是 | 是 |
| `GGML_OP_CONV_2D` | 卷积 | 是 | 是 |
| `GGML_OP_CONV_3D` | 卷积 | 是 | 否 |
| `GGML_OP_CONV_2D_DW` | depthwise 卷积 | 是 | 是 |
| `GGML_OP_CONV_TRANSPOSE_2D` | 卷积 | 是 | 是 |
| `GGML_OP_POOL_1D` | pooling | 是 | 否 |
| `GGML_OP_POOL_2D` | pooling | 是 | 是 |
| `GGML_OP_POOL_2D_BACK` | pooling 反向 | 是 | 否 |
| `GGML_OP_UPSCALE` | 采样 | 是 | 是 |
| `GGML_OP_PAD` | padding | 是 | 是 |
| `GGML_OP_PAD_REFLECT_1D` | reflection padding | 是 | 是 |
| `GGML_OP_ROLL` | 循环移位 | 是 | 是 |
| `GGML_OP_ARANGE` | tensor 构造 | 是 | 是 |
| `GGML_OP_TIMESTEP_EMBEDDING` | diffusion embedding | 是 | 是 |
| `GGML_OP_ARGSORT` | 排序 | 是 | 是 |
| `GGML_OP_TOP_K` | top-k | 是 | 是 |
| `GGML_OP_LEAKY_RELU` | 激活 | 是 | 是 |
| `GGML_OP_TRI` | 三角 mask | 是 | 是 |
| `GGML_OP_FILL` | 填充 | 是 | 是 |
| `GGML_OP_FLASH_ATTN_EXT` | FlashAttention | 是 | 是 |
| `GGML_OP_FLASH_ATTN_BACK` | FlashAttention 反向 | 是 | 否 |
| `GGML_OP_SSM_CONV` | SSM convolution | 是 | 是 |
| `GGML_OP_SSM_SCAN` | SSM scan | 是 | 是 |
| `GGML_OP_WIN_PART` | 视觉窗口 | 是 | 否 |
| `GGML_OP_WIN_UNPART` | 视觉窗口 | 是 | 否 |
| `GGML_OP_GET_REL_POS` | 相对位置 | 是 | 否 |
| `GGML_OP_ADD_REL_POS` | 相对位置 | 是 | 否 |
| `GGML_OP_RWKV_WKV6` | recurrent | 是 | 是 |
| `GGML_OP_GATED_LINEAR_ATTN` | recurrent | 是 | 是 |
| `GGML_OP_RWKV_WKV7` | recurrent | 是 | 是 |
| `GGML_OP_SOLVE_TRI` | 三角方程 | 是 | 是 |
| `GGML_OP_GATED_DELTA_NET` | GDN recurrent | 是 | 是 |
| `GGML_OP_UNARY` | 激活集合 | 是 | 是 |
| `GGML_OP_MAP_CUSTOM1` | 自定义 CPU op | 是 | 否 |
| `GGML_OP_MAP_CUSTOM2` | 自定义 CPU op | 是 | 否 |
| `GGML_OP_MAP_CUSTOM3` | 自定义 CPU op | 是 | 否 |
| `GGML_OP_CUSTOM` | 自定义 CPU op | 是 | 否 |
| `GGML_OP_CROSS_ENTROPY_LOSS` | loss | 是 | 是 |
| `GGML_OP_CROSS_ENTROPY_LOSS_BACK` | loss 反向 | 是 | 是 |
| `GGML_OP_OPT_STEP_ADAMW` | optimizer | 是 | 是 |
| `GGML_OP_OPT_STEP_SGD` | optimizer | 是 | 是 |
| `GGML_OP_GLU` | gated activation | 是 | 是 |
<!-- LLAMA_GGML_OPS_END -->

### 8.1 CUDA 入口的主要限制

| op | CUDA `supports_op` 条件摘要 |
|---|---|
| unary | 输入必须 contiguous；支持 22 个 unary subtype |
| GLU | `src0` 必须 contiguous_1；支持 6 个 GLU subtype |
| MUL_MAT/ID | weight type 在支持表；F16 activation 只配 F16 weight；split 另有限制 |
| OUT_PROD | output/src0/src1 均 F32 |
| GET_ROWS | F16/F32/BF16/I32/Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 |
| SET_ROWS | output 为 F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL |
| CPY | 常用 F32/F16/BF16、I32 与部分 legacy quant 转换；同型 contiguous 可直拷 |
| binary ADD/SUB/MUL/DIV | 输入输出只能 F32/F16 |
| SSM_SCAN | Mamba2 state=128/256 且 head%16=0；Mamba state=16 等固定约束 |
| SSM_CONV | inner dimension % 128 = 0 |
| SOFT_MAX_BACK | `max_bias == 0` |
| ROPE | element stride 正确且 contiguous_2 |
| TOP_K/ARGSORT | 未启用 CUB 时行宽 ≤1024 |
| GATED_DELTA_NET | CUDA 可用；MUSA 当前禁用 |
| FLASH_ATTN_EXT | 由独立 kernel selector 检查完整 shape/type |

## 9. GGML 数据格式全表

bpw 是单个 block 的真实 payload 位数除以解量化元素数，不含 tensor/GGUF 元数据。
Q8_1 和 Q8_K 主要是 activation/dot-product 中间格式。

<!-- LLAMA_GGML_TYPES_BEGIN -->
| GGML type | block values | block bytes | bpw | CPU dot 的 activation | CUDA MMVQ | CUDA MMQ |
|---|---:|---:|---:|---|---:|---:|
| `GGML_TYPE_F32` | 1 | 4 | 32 | F32 | 否 | 否 |
| `GGML_TYPE_F16` | 1 | 2 | 16 | F16 | 否 | 否 |
| `GGML_TYPE_Q4_0` | 32 | 18 | 4.5 | Q8_0 | 是 | 是 |
| `GGML_TYPE_Q4_1` | 32 | 20 | 5.0 | Q8_1 | 是 | 是 |
| `GGML_TYPE_Q5_0` | 32 | 22 | 5.5 | Q8_0 | 是 | 是 |
| `GGML_TYPE_Q5_1` | 32 | 24 | 6.0 | Q8_1 | 是 | 是 |
| `GGML_TYPE_Q8_0` | 32 | 34 | 8.5 | Q8_0 | 是 | 是 |
| `GGML_TYPE_Q8_1` | 32 | 36 | 9.0 | 中间格式 | 否 | 否 |
| `GGML_TYPE_Q2_K` | 256 | 84 | 2.625 | Q8_K | 是 | 是 |
| `GGML_TYPE_Q3_K` | 256 | 110 | 3.4375 | Q8_K | 是 | 是 |
| `GGML_TYPE_Q4_K` | 256 | 144 | 4.5 | Q8_K | 是 | 是 |
| `GGML_TYPE_Q5_K` | 256 | 176 | 5.5 | Q8_K | 是 | 是 |
| `GGML_TYPE_Q6_K` | 256 | 210 | 6.5625 | Q8_K | 是 | 是 |
| `GGML_TYPE_Q8_K` | 256 | 292 | 9.125 | 中间格式 | 否 | 否 |
| `GGML_TYPE_IQ2_XXS` | 256 | 66 | 2.0625 | Q8_K | 是 | 是 |
| `GGML_TYPE_IQ2_XS` | 256 | 74 | 2.3125 | Q8_K | 是 | 是 |
| `GGML_TYPE_IQ3_XXS` | 256 | 98 | 3.0625 | Q8_K | 是 | 是 |
| `GGML_TYPE_IQ1_S` | 256 | 50 | 1.5625 | Q8_K | 是 | 是 |
| `GGML_TYPE_IQ4_NL` | 32 | 18 | 4.5 | Q8_0 | 是 | 是 |
| `GGML_TYPE_IQ3_S` | 256 | 110 | 3.4375 | Q8_K | 是 | 是 |
| `GGML_TYPE_IQ2_S` | 256 | 82 | 2.5625 | Q8_K | 是 | 是 |
| `GGML_TYPE_IQ4_XS` | 256 | 136 | 4.25 | Q8_K | 是 | 是 |
| `GGML_TYPE_I8` | 1 | 1 | 8 | 无通用 weight dot | 否 | 否 |
| `GGML_TYPE_I16` | 1 | 2 | 16 | 无通用 weight dot | 否 | 否 |
| `GGML_TYPE_I32` | 1 | 4 | 32 | 无通用 weight dot | 否 | 否 |
| `GGML_TYPE_I64` | 1 | 8 | 64 | 无通用 weight dot | 否 | 否 |
| `GGML_TYPE_F64` | 1 | 8 | 64 | 无通用 weight dot | 否 | 否 |
| `GGML_TYPE_IQ1_M` | 256 | 56 | 1.75 | Q8_K | 是 | 否 |
| `GGML_TYPE_BF16` | 1 | 2 | 16 | BF16 | 否 | 否 |
| `GGML_TYPE_TQ1_0` | 256 | 54 | 1.6875 | Q8_K | 否 | 否 |
| `GGML_TYPE_TQ2_0` | 256 | 66 | 2.0625 | Q8_K | 否 | 否 |
| `GGML_TYPE_MXFP4` | 32 | 17 | 4.25 | Q8_0 | 是 | 是 |
| `GGML_TYPE_NVFP4` | 64 | 36 | 4.5 | Q8_0 | 是 | 是 |
| `GGML_TYPE_Q1_0` | 128 | 18 | 1.125 | Q8_0 | 是 | 是 |
<!-- LLAMA_GGML_TYPES_END -->

### 9.1 Q、K、IQ 的含义

- `Q*_0/Q*_1`：32-value legacy scalar blocks；`_1` 保存 min/offset 信息。
- `Q*_K`：256-value super-block，子块 scale/min 也量化。
- `IQ*`：importance-aware nonlinear/vector codebook family；多种低 bit 格式要求 imatrix。
- `TQ*`：ternary family。
- `MXFP4/NVFP4`：浮点式 4-bit family；NVFP4 每 16 值有 UE4M3 scale。
- 文件 recipe 的平均 bpw 还包含高精度 tensor、alignment、metadata 和可能的 MTP。

## 10. CPU backend

### 10.1 通用 matmul

CPU `MUL_MAT` 从 `type_traits_cpu[weight_type]` 取得：

- `vec_dot`；
- activation 的 `vec_dot_type`；
- `from_float`；
- 一次 dot 可处理的行数。

若 activation 不是目标 dot type，先把 activation 转成 Q8_0、Q8_1、Q8_K、
F16、BF16 或 F32，再按 output rows 和 activation columns 分块调用 SIMD dot。
K-quant 和多数 IQ 使用 Q8_K；Q4_1/Q5_1 使用 Q8_1；Q4_0/Q5_0/Q8_0、
IQ4_NL、MXFP4、NVFP4 使用 Q8_0。

`MUL_MAT_ID` 仍按 expert id 选择 weight slice；CPU 线程池负责 row/task 分解。

### 10.2 优化层级

CPU 路径可能在通用 switch 前被 extra buffer traits 接管：

1. architecture-specific repack；
2. AMX；
3. KleidiAI；
4. llamafile SGEMM；
5. generic SIMD vec-dot。

repack 只改变内存排列与 microkernel 输入格式，不改变 GGUF 中的逻辑
`ggml_type`。当前 repack 覆盖 Q4_0、Q2_K、Q4_K、Q5_K、Q6_K、Q8_0、
IQ4_NL 和 MXFP4 的若干 4/8/16-row interleave。

BLAS 是独立 backend。它通常只在大 dense float matmul 上有优势，由 scheduler
选择；它不替代 CPU quant vec-dot。

## 11. CUDA dense matmul 分发

### 11.1 单 GPU 优先级

`ggml_cuda_mul_mat` 按以下顺序选择：

1. `mul_mat_vec_f`：F32/F16/BF16 的小 M custom vector kernel。
2. `mul_mat_f`：非量化 custom MMF。
3. `mul_mat_vec_q`：量化 weight 的 MMVQ。
4. `mul_mat_q`：量化 weight 的 MMQ。
5. batched cuBLAS：多 channel/sample 的 float matmul。
6. split-buffer 对应 wrapper。
7. dequant/convert + cuBLAS fallback。

量化 temporary view 若 padding 不能安全清零，MMVQ/MMQ 会被禁用并退到 cuBLAS。

### 11.2 MMVQ

MMVQ 的上限是 `MMVQ_MAX_BATCH_SIZE=8`。dense 条件是：

```text
weight quantized
activation F32
output F32
M <= 8
format/architecture selector returns true
```

activation 由 `quantize_row_q8_1_cuda` 转成 Q8_1。generic kernel 按 M 选择
每个 output row 的 warp 数：

| M | warps/output row |
|---:|---:|
| 1–4 | 4 |
| 5–8 | 2 |
| 其他可达路径 | 1 |

MoE MMVQ 使用专门的 `mul_mat_vec_q_moe`，最大 M 还会按 GPU 架构和 weight
format 缩小。Ada/Turing+ 的默认上限仍不超过 8。

### 11.3 MMQ 与 Q8_1 activation

MMQ 条件是 quantized weight、F32 activation/output、padding 安全且格式受支持。
在 Turing 或更新的 NVIDIA GPU 上，`turing_mma_available(cc)` 直接返回 true，
不再按 M 切回 cuBLAS。

因此本机 4090/Ada 上：

```text
M=512  -> MMQ
M=2048 -> MMQ
```

只要 weight format 属于表中的 21 种 MMQ 格式，并且没有 padding/view 例外。
`GGML_CUDA_FORCE_CUBLAS` 才会强制关闭。

MMQ activation 由 `quantize_mmq_q8_1_cuda` 转成 `block_q8_1_mmq`：

- K pad 到 `MATRIX_ROW_PADDING`；
- 一个 MMQ block 覆盖 128 个 int8 activation；
- 等价于 4 个 32-value Q8_1 子块；
- 每个子块保留 scale 与量化值和，用于带 offset weight 的 dot 修正。

只复制 int8 值或只保留一个全行 scale 都不等价于 llama.cpp Q8_1 MMQ。

### 11.4 MMQ launch geometry

NVIDIA Volta+：

```text
mmq_y = 128 output rows/block
block = (warp_size=32, nwarps=8, 1) = 256 threads
mmq_x candidates = 8,16,...,128 activation columns/block
```

selector 在 shared-memory 与 granularity 约束下，选择让
`ceil(M/mmq_x)` 最小的 `mmq_x`。Turing 的 `mmq_x>=48` granularity 为 16，
更小 tile 为 8。

普通 grid：

```text
grid.x = ceil(N / mmq_y)
grid.y = ceil(M / mmq_x)
grid.z = channels * samples
```

shared memory 包含 expert ids、quantized weight tile 和
`mmq_x * sizeof(block_q8_1_mmq)` activation tile。

Stream-K 在 NVIDIA Volta+ 和 AMD CDNA 可用。NVIDIA tile wave efficiency
达到 90% 时可直接按全部 tile 启动；其余情况通常按 SM 数启动。tile 数不能整除
block 数时使用 fixup buffer 和 `mul_mat_q_stream_k_fixup`。

## 12. CUDA routed expert 与融合

### 12.1 `MUL_MAT_ID`

CUDA 路由顺序：

1. quantized 且 M≤8、格式/架构允许：MoE MMVQ；
2. 支持 MMQ：grouped MMQ；
3. 支持 MMF：grouped float MMF；
4. fallback：host 端排序并逐 expert 调 matmul，含 stream synchronize。

grouped MMQ 的 `mm_ids_helper` 为每个 expert 建立连续 token 列表。id 使用：

```text
低 22 bit: token index
高 10 bit: token 内的 expert slot
```

top-k 有 2/4/6/8/16/32 的专门实例。输入按 expert compact 后执行 grouped
MMQ，输出通过 `ids_dst` scatter 到原 token/expert slot。

这里的“合并”是调度上的 compact/group，不改变每个 expert 的矩阵边界。
同一 weight、activation、accumulation dtype 下，逐 expert 与 grouped 的语义相同；
任何显著差异应查 padding、index、scatter、activation quantization 和累加顺序。

### 12.2 MoE top-k 图融合

CUDA 可以把：

```text
softmax/sigmoid
-> reshape
-> optional probability bias
-> argsort/top-k
-> view/get_rows
-> optional normalize/scale
```

融合为 `topk_moe_cuda`。约束包括：

- expert 数是 2 的幂且 ≤512，或恰好 576；
- logits/weights 为 F32，ids 为 I32；
- 4 rows/block；
- block `(warp_size, 4, 1)`；
- grid `ceil(n_rows/4)`。

`GGML_CUDA_DISABLE_FUSION=1` 禁用 CUDA graph pattern fusion。

### 12.3 FFN 形态

`build_moe_ffn` 支持：

- merged gate/up：一次 `MUL_MAT_ID`，随后 view 拆分；
- separate gate/up：两次 `MUL_MAT_ID`；
- gate activation；
- down `MUL_MAT_ID`；
- routed weight 加权归约；
- shared expert 分支。

比较 MFQ 与 llama.cpp 时，gate/up 是否物理拼接是 kernel geometry 变量。

## 13. FlashAttention 与基础 attention

### 13.1 selector

`ggml_cuda_get_best_fattn_kernel` 在 TILE、VEC、WMMA_F16、MMA_F16 中选择。
支持的 K head dimension：

```text
40, 64, 72, 80, 96, 112, 128, 192, 256, 320, 512, 576
```

特殊 V dimension：

- K=192 可配 V=128，并要求 GQA optimization 与 ratio%8=0；
- K=320 可配 V=256，并要求 ratio%32=0；
- K=576 可配 V=512；
- 其余常用路径 V=K。

GQA optimization 要求 ratio≥2、存在 mask、无 ALiBi max_bias、K length pad 到
`FATTN_KQ_STRIDE`，非量化 stride 还需 16 对齐。

### 13.2 format

当前构建 `GGML_CUDA_FA_ALL_QUANTS=OFF`，编译的 VEC type pair 是：

```text
F16/F16
Q4_0/Q4_0
Q8_0/Q8_0
BF16/BF16
```

F32 K/V 可在 TILE/WMMA/MMA 路径转换为 F16 workspace。开启
`FA_ALL_QUANTS` 才会加入 Q4_1、Q5_0、Q5_1 和更多 mixed pair。

VEC 主要用于 head≤256、head%64=0、head≠192 且 K padded 的形状。
Ada 上单 token 与低 M quant KV 常走 VEC；更大 M 通常进入 MMA F16。

### 13.3 本机 mixed-KV 扩展

`dst->op_params[4] != 0` 会绕过标准 selector，进入本地
`fattn-mixed-vec.cuh`。它处理 Qwen3.5 F16/Q4 mixed KV 和 per-head window。
`QWEN35_MIXED_TARGET_PB` 只属于本机扩展，不是纯上游接口。

## 14. recurrent、GDN 与状态算子

| op | CUDA geometry/约束 |
|---|---|
| `GATED_DELTA_NET` | state size 16/32/64/128；grid `(H,n_seq,ceil(Sv/4))`；block `(min(warp,Sv),4,1)` |
| `GATED_LINEAR_ATTN` | head size 64/128；每个 B×H 一个 block；每线程一个 head dim |
| `SSM_SCAN` | Mamba/Mamba2 两套固定 state/head 条件 |
| `SSM_CONV` | inner dimension 必须是 128 的倍数 |
| `RWKV_WKV6/7` | 独立 recurrent CUDA kernels |

GDN 每个 warp 持有一个 state/output column，state 分片放在寄存器中，token 维
顺序循环。支持 scalar decay `g` 和 KDA per-dimension decay，也可输出 recurrent
snapshot。源码仍标注缺少 chunked-prefill GDN kernel，因此长 prefill 的 token
并行度受限。

## 15. CUDA Graph、融合与多 GPU

### 15.1 CUDA Graph

llama.cpp 顶层默认 `GGML_CUDA_GRAPHS_DEFAULT=ON`。运行时要求：

- Ampere 或更新；
- 没有 split buffer；
- `MUL_MAT_ID` 必须落在可 capture 的 MMVQ 路径，较大 grouped 路径会同步；
- graph property 连续两次稳定后才 capture；
- shape/property 变化时 update executable 或重新 instantiate。

`GGML_CUDA_DISABLE_GRAPHS=1` 可禁用。

### 15.2 decode graph optimization

`GGML_CUDA_GRAPH_OPT=1` 只在单 GPU、CUDA Graph 已启用时生效。它寻找
`attn_norm` 后恰好 3 条 fan-out branch，分配到 auxiliary streams，再用 event join。
当前只考虑 `nrows<=1`，目标是 decode，不是通用 prefill overlap。

### 15.3 多 GPU

row split 主要用于 `MUL_MAT`。peer copy、VMM、NCCL 和 allreduce 由构建选项与
运行时环境控制。`MUL_MAT_ID` 不接受 CUDA split buffer；MoE 多 GPU 需要模型
placement 或更上层分工，不能套用 dense row split。

## 16. 量化 recipe 与 imatrix

### 16.1 CLI recipe

`llama-quantize` 支持 Q1_0、Q4_0、Q4_1、MXFP4_MOE、Q5_0、Q5_1、
IQ2_XXS/XS/S/M、IQ1_S/M、TQ1_0/TQ2_0、Q2_K/Q2_K_S、
IQ3_XXS/S/M/XS、Q3_K_S/M/L、IQ4_NL/XS、Q4_K_S/M、Q5_K_S/M、
Q6_K、Q8_0、F16、BF16、F32 和 COPY。

`Q3_K`、`Q4_K`、`Q5_K` 分别是 M recipe 的 alias。`--pure` 关闭混合
tensor type；`--tensor-type REGEX=TYPE` 可覆盖；`--keep-split` 保持 split 输出。

### 16.2 tensor sensitivity recipe

`llama_tensor_get_type_impl` 按 tensor category、model arch、GQA、expert count、
layer index 和 imatrix 是否存在选择实际格式。主要 category：

```text
output, token_embd, attn_qkv, attn_kv_b, attn_v, attn_k, attn_q,
attn_output, ffn_up, ffn_gate, ffn_down, other
```

`use_more_bits(i,n)` 覆盖：

```text
前 1/8 层
后 1/8 层
中间区域每 3 层中的第 3 层
```

output/tied embedding 通常升到 Q5_K/Q6_K/Q8_0。attention V、FFN down 和
attention output 有额外敏感性规则。因此“复制 Q4_K_M recipe”必须读取最终
逐 tensor type，不能只把所有 tensor 设为 Q4_K。

### 16.3 imatrix 采集

`llama-imatrix` 通过 scheduler eval callback 采集：

- `MUL_MAT_ID`：全部采集；
- `MUL_MAT`：M≥16、activation 为 F32，且 tensor name/选项允许；
- 每个输入通道累计 `sum(x²)` 和 count；
- MoE 只给被当前 token 选中的 expert 累计；
- MoE entry shape 是 `n_expert * K`，并保存 per-expert count。

量化 MoE tensor 时，每个 expert 单独调用 quantizer，并把 imatrix slice 移到
`imatrix + expert*K`。

强制要求 imatrix 的目标格式：

```text
IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M, IQ1_S
Q2_K 仅在 Q2_K_S recipe 中要求
```

token embedding 与 output weight 例外。较高 bit K-quant 也可以使用 imatrix
参与误差目标；“不强制”不表示“忽略”。

GGUF imatrix 保存 raw sums 与 counts；legacy `.dat` 保存经过旧 ncall 约定的值。
loader 合并文件时按格式恢复一致统计量。

## 17. perplexity 与 KLD

### 17.1 标准 PPL

无 `ppl_stride` 时：

1. 文本 tokenization；
2. 按 `chunk_n_ctx` 做不重叠 chunk；
3. 每个 chunk 首 token 按 vocab policy 替换为 BOS；
4. 每组 sequence 前清空 model memory；
5. 只请求后半段 logits；
6. score 数为 `chunk_n_ctx/2 - 1`；
7. target 是每个 logit 位置的下一个 token。

`chunk_n_ctx=2048` 时每块 score 1023 个 token。145 块对应 148335 个 scored
tokens。`n_batch` 可以让多个完整 chunk 并行；`n_ubatch` 再决定每次物理构图大小。

### 17.2 `_logits_` reference

旧格式：

```text
magic "_logits_"
uint32 n_ctx
int32 n_vocab
int32 n_chunk
token ids[n_ctx*n_chunk]
for each scored row:
    float scale
    float min_log_prob
    uint16 encoded_log_prob[n_vocab]
```

每行把低于 `max_logit-16` 的值压到编码 0。当前 dirty 修复把实际 BOS 替换后的
tokens 写入 reference，避免生成 reference 与复测 token 不同。

forward KLD 为 `KL(reference || quant)`；reverse KLD 为
`KL(quant || reference)`，累计字段是 `sum_reverse_kld`。forward 路径只累计非零 reference code；reverse
路径先对解码后的整行 reference log-prob 重新归一化。reference 本身仍是
uint16 且有 16-logit clipping，这一点属于协议误差。

### 17.3 `_logit3_` trace_v3

本机扩展格式：

```text
magic "_logit3_"
uint32 n_vocab
uint32 n_chunks
per chunk: uint32 token_count, target_start, score_count
all chunks: token ids
all chunks: exact float target_log_probs
all chunks: uint16 compressed distribution rows
```

评测时每个 chunk：

- token 数不得超过 context capacity；
- 清空 model memory；
- 按 `n_batch` 多次 decode；
- 请求 `[target_start-1, target_start-1+score_count)` 的 logits；
- target token 使用下一位置；
- 读取同样数量的 reference rows；
- 输出 forward KLD、reverse KLD、same-top、BF16 exact CE 和 quant CE。

`MFQ_PPL_CHUNK_OFFSET` 只作用于标准 PPL/reference 生成路径。`trace_v3` 的
chunk 选择由文件与 `n_chunks` 决定。

### 17.4 公平比较必须固定的字段

```text
reference checkout + build
model file hash
tokenizer/chat template/BOS policy
token ids or trace file hash
chunk_n_ctx
n_batch
n_ubatch
n_seq
chunk count and scored-token count
KV type and FlashAttention
weight recipe and MTP inclusion
KLD direction
CUDA_FORCE_MMQ/CUBLAS and graph/fusion switches
```

MFQ optimized evaluator 若一次把 `[1,2048]` 送入 Linear，会得到 M=2048；
llama.cpp `n_ubatch=512` 得到四次 M=512。两者 token 与 score 相同，kernel
geometry、activation quantization block 和累加顺序仍不同。

## 18. 构建与本机二进制

当前 `build-codex-cuda/CMakeCache.txt`：

```text
CMAKE_BUILD_TYPE=Release
BUILD_SHARED_LIBS=ON
GGML_NATIVE=ON
GGML_CPU_REPACK=ON
GGML_BLAS=OFF
GGML_CUDA=ON
GGML_CUDA_FA=ON
GGML_CUDA_FA_ALL_QUANTS=OFF
GGML_CUDA_FORCE_MMQ=OFF
GGML_CUDA_FORCE_CUBLAS=OFF
GGML_CUDA_GRAPHS=ON
GGML_CUDA_NCCL=ON
LLAMA_BUILD_TOOLS=ON
CMAKE_CUDA_ARCHITECTURES_NATIVE=86-real
```

`GGML_CUDA_NCCL=ON` 只表示请求查找 NCCL；该 build cache 中
`NCCL_INCLUDE_DIR` 与 `NCCL_LIBRARY` 都是 `NOTFOUND`，实际 binary 未编入 NCCL。

该 build 目录的 CUDA translation units：

```text
65 top-level .cu
12 fattn-tile instances
21 fattn-mma instances
21 mmq instances
16 mmf instances
4 default fattn-vec instances
= 139
```

CUDA 编译使用 `-use_fast_math -extended-lambda`。`GGML_NATIVE=ON` 时构建机检测到
sm86；当前 binary 因 NVIDIA minor-version compatibility 可在 sm89 运行，但它不含
sm89 专用重新编译带来的潜在调优。

主要可执行文件：

| 工具 | 用途 |
|---|---|
| `llama-perplexity` | PPL/KLD/reference |
| `llama-quantize` | GGUF 量化 |
| `llama-imatrix` | imatrix 采集 |
| `llama-bench` | pp/tg 性能 |
| `test-backend-ops` | backend 数值/性能 shape |
| `llama-gdn-trace` | 本机 GDN/trace 扩展 |

## 19. CUDA 源文件责任表

| 家族 | 主要文件 | 责任 |
|---|---|---|
| 总分发 | `ggml-cuda.cu`, `common.cuh` | op dispatch、supports、pool、stream、graph |
| quant GEMV | `mmvq.cu`, `mmvq.cuh`, `quantize.cu` | Q8_1 activation、dense/MoE MMVQ |
| quant GEMM | `mmq.cu`, `mmq.cuh`, `template-instances/mmq*` | Q8_1 MMQ、MMA/DP4A、Stream-K |
| float matmul | `mmf.cu`, `mmf.cuh`, `mmvf.cu`, `mmvf.cuh` | float GEMM/GEMV |
| expert ids | `mmid.cu`, `topk-moe.cu` | expert compact、bounds、top-k fusion |
| cuBLAS/convert | `convert.cu`, `cpy.cu` | dtype conversion、fallback workspace |
| FlashAttention | `fattn.cu`, `fattn-tile*`, `fattn-vec*`, `fattn-mma*`, `fattn-wmma-f16.cu` | selector 与各架构 kernel |
| 本机 mixed FA | `fattn-mixed-vec.cuh` | Qwen3.5 mixed F16/Q4 KV |
| normalization | `norm.cu`, `scale.cu` | norm/RMS/L2/group norm、scale |
| pointwise | `unary.cu`, `binbcast.cu`, `clamp.cu`, `softcap.cu` | activation 与 binary |
| attention basic | `rope.cu`, `softmax.cu`, `diagmask.cu` | RoPE、softmax、mask |
| recurrent | `gated_delta_net.cu`, `gla.cu`, `ssm-conv.cu`, `ssm-scan.cu`, `wkv.cu` | GDN/GLA/SSM/RWKV |
| gather/scatter | `getrows.cu`, `set-rows.cu`, `add-id.cu` | embedding/expert gather、scatter、bias |
| reduction/sort | `sum.cu`, `sumrows.cu`, `mean.cu`, `argmax.cu`, `argsort.cu`, `top-k.cu`, `cumsum.cu` | 归约与排序 |
| convolution | `im2col.cu`, `col2im-1d.cu`, `conv2d*.cu`, `pool2d.cu` | vision/audio/diffusion |
| optimizer | `cross-entropy-loss.cu`, `opt-step-adamw.cu`, `opt-step-sgd.cu` | training loss/step |
| multi-GPU | `allreduce.cu` | device reduction |

源码中有 134 个显式 `__global__` 定义，分布在 65 个 `.cu/.cuh` 文件。
template instance 数量远大于 kernel 源定义数量，二者不能混作“kernel 数”。

## 20. 运行时开关

### 20.1 CUDA/调度

| 环境变量 | 作用 |
|---|---|
| `GGML_CUDA_DISABLE_GRAPHS` | 禁用 CUDA Graph |
| `GGML_CUDA_GRAPH_OPT` | 开启 decode 三分支 stream 优化 |
| `GGML_CUDA_DISABLE_FUSION` | 禁用 CUDA graph pattern fusion |
| `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F` | 强制 cuBLAS 16F compute |
| `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` | 强制 cuBLAS 32F compute |
| `GGML_CUDA_NO_PINNED` | 禁用 pinned host buffer |
| `GGML_CUDA_REGISTER_HOST` | 注册 host memory |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | 开启 unified memory |
| `GGML_CUDA_P2P` | 控制 peer-to-peer |
| `GGML_CUDA_ALLREDUCE` | 控制 allreduce |
| `GGML_CUDA_PDL` | programmatic dependent launch |
| `GGML_OP_OFFLOAD_MIN_BATCH` | backend op offload 的最小 batch；CUDA 默认 32 |
| `GGML_SCHED_DEBUG` | scheduler placement debug |
| `GGML_SCHED_DEBUG_REALLOC` | graph reallocation 检查 |
| `GGML_CPU_DISABLE_FUSION` | 禁用 CPU fusion |

`GGML_CUDA_FORCE_MMQ` 和 `GGML_CUDA_FORCE_CUBLAS` 是 CMake 选项/编译宏，
不是当前源码读取的运行时环境变量。

### 20.2 模型/评测

| 环境变量 | 作用 |
|---|---|
| `MFQ_PPL_CHUNK_OFFSET` | 标准 PPL 从指定 chunk 开始 |
| `LLAMA_GRAPH_REUSE_DISABLE` | 禁用 graph reuse |
| `LLAMA_KV_CACHE_DEBUG` | KV cache debug |
| `LLAMA_GRAPH_RESULT_DEBUG` | graph result debug |
| `LLAMA_BATCH_DEBUG` | batch debug |
| `LLAMA_DSV4_COMPRESS_DEBUG` | DSV4 compression debug |
| `LLAMA_ATTN_ROT_DISABLE` | attention rotation 开关 |
| `QWEN35_MIXED_KV_WINDOW` | mixed KV window |
| `QWEN35_PER_KV_WINDOWS` | per-KV-head window |
| `QWEN35_PER_KV_WINDOWS_BY_LAYER` | per-layer/per-head window |
| `QWEN35_FORCE_FULL_ATTN_SWA` | Qwen3.5 SWA override |
| `QWEN35_MIXED_TARGET_PB` | 本机 mixed kernel parallel blocks |

## 21. 修改影响矩阵

| 修改目标 | 最少检查的源码 | 必须验证 |
|---|---|---|
| 新 GGML type | `ggml.h`, `ggml.c`, common block、CPU traits、CUDA traits | block bytes/bpw、roundtrip、dot、all backends |
| 新 quant recipe | `llama-quant.cpp`, quantize CLI | 最终逐 tensor type、imatrix、size、PPL/KLD |
| MMVQ/MMQ | `mmvq*`, `mmq*`, `quantize.cu`, dispatch | M=1/8/9/512/2048，全部 weight type |
| MoE grouped | `mmid.cu`, `mmq*`, `ggml-cuda.cu` | 全 expert、重复 id、top-k、scatter、padding |
| FlashAttention | `fattn*`, cache class、model graph | head dim、GQA、mask、KV type、SWA、M |
| GDN | `gated_delta_net.cu`, model graph、memory | state size、prefill/decode、state snapshot |
| backend offload | `ggml-backend.cpp`, buffer traits | copy bytes、同步、used expert、数值 |
| KLD | `perplexity.cpp`, token/reference producer | token、BOS、score offset、方向、scored count |
| model graph | 对应 `src/models/*.cpp` | FP reference logits、cache、residual/norm 顺序 |

## 22. 验证规范

数值验证至少覆盖：

```text
dense MUL_MAT: M=1,8,9,512,2048
MUL_MAT_ID: 1/多 token、重复 expert、全部 expert、边界 expert
weight type: recipe 中每一种实际 tensor type
activation: llama Q8_1 MMVQ、Q8_1 MMQ、float fallback
attention: full/SWA、GQA、mixed KV、head-dim 特殊形状
state: chunked prefill 与逐 token decode 最终 state
KLD: 前 3 块与全集；同一 reference hash
```

性能报告至少保存：

```text
GPU/driver/CUDA/build hash
M/N/K/E/top-k
weight type 与 activation type
实际 dispatch 路径
tile/block/grid/shared memory
warmup/repeat/synchronization
H2D/D2H 与 cache 冷热状态
```

本仓库的文档漂移测试：

```powershell
$env:LLAMA_CPP_REFERENCE_DIR = '<llama.cpp source directory>'
python -m pytest -q -p no:cacheprovider tests/test_llamacpp_operator_reference.py
```

## 23. 已知边界

1. 本机 KLD 修复和 `trace_v3` 尚未进入参考 checkout 的 Git commit。
2. 本机 mixed-KV 是 10 个本地提交构成的扩展。
3. 当前 binary 面向 sm86 构建，未重新生成 sm89 专用代码。
4. `_logits_` reference 使用 uint16 和 16-logit clipping，仍有协议近似。
5. scheduler 的 MoE expert copy 含同步和 ids readback，没有 LRU/prefetch。
6. GDN CUDA 仍缺少专门的 chunked-prefill kernel。
7. CUDA `supports_op=true` 只说明 shape/type 可接收，不能证明命中了目标 kernel。
