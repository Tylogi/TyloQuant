"""TPQ（TyloQuant PQ）—— CCCP 量化模型的开源推理框架。

读取 CCCP 流水线产出的 -cccp 模型（format "cccp-1"），双架构自动分派：
  - GLM（GlmMoeDsa）与 DeepSeek-V4（deepseek_v4），CPU / CUDA 自动适配
  - dense 权重 int4-g64 打包，按需反量化（CUDA 下一次反量化 bf16 常驻）
  - routed 专家以 VQ 索引态驻留/流式加载（两级 LRU），LUT 免反量化矩阵乘
  - 投机解码：GLM 走 MTP（layer 78），DSV4 走自带 DSpark 三层块并行草稿
  - 内存/显存自动适配：专家缓存预算按可用 RAM/显存自动计算，显存不足自动
    降档重试并回退 CPU（[tpq] 状态行可见）

启动方式：
    python -m tpq chat --model <模型目录> [--device cuda] [--spec N] [--think]
Windows 下也可双击 chat_dsv4.py 启动器（自动定位模型目录）。

设计约定：本包刻意不依赖 numpy —— Windows + Anaconda 下 numpy(MKL) 与 torch
会重复加载 OpenMP 运行库导致间歇性 native 崩溃；纯 torch 栈可完全规避。
"""
