# 通用算子注册层

本目录把“设备、量化格式、张量形状和数学能力”与具体模型解耦。Kimi、GLM 和
DeepSeek 通过同一注册表选择 CPU/CUDA VQ、MoE、Attention 和张量并行能力；
模型文件只保留架构数学差异。

- `spec.py`：设备、packed 格式、码本和算子能力描述。
- `registry.py`：注册、选择和能力查询。
- `api.py`：模型调用的稳定公共入口。
- `config.py`：注册层配置与环境开关。
- `cpu_backend.py`：CPU packed VQ/MoE 注册。
- `cuda_backend.py`：CUDA VQ/MoE/Attention 注册。
- `moe.py`：Top-K MoE 公共调度。
- `hidden.py`：跨 rank hidden 状态抽象。
- `tensor_parallel.py`：Column/Row-TP、collective 和固定地址 Graph。
- `profiling.py`：通用算子耗时探针。
- `selftest.py`：由 `tpq check --cuda-ops` 调用的公共 CUDA 数值验收。

约束：

1. 注册键描述数学能力，不使用模型名称作为算子分叉。
2. packed 8–16-bit 索引在磁盘、RAM、VRAM 中保持紧凑，不创建完整
   反量化矩阵；分组码本由专家 ID 选择，不能先展开成模型级统一 dtype。
3. owner 只能用于元数据，不能成为核心 hidden 或专家计算的数据 owner。
4. 新快路径必须保留正确性回退并增加无私有权重的数值测试。

三投影 packed MoE 的注册项覆盖 SiTU、SiLU/SwiGLU；具体模型只提交 projection
能力元组、激活名和 clamp 参数。H/C/S、front/tail 等层型属于配置数据，不能进入
算子注册名称。

DSV4 新增的是数学能力键 `cuda.route_topk.sqrtsoftplus.decode`、
`cuda.linear_route_topk.sqrtsoftplus.decode` 和
`cuda.attention.sliding_compressed_mqa.decode`。Kimi 既有的 SiTU、KDA、MLA、
Front44/Tail48 注册键不改名；`tpq check --cuda-ops` 会在一次测试中同时回归两类
布局，防止新增格式覆盖旧注册项。

Hyper-Connection CUDA 能力支持调用方固定输出缓冲：HC pre 复用 `y/post/comb`，HC post
复用互不别名的 hidden 结果；公共能力 `hyper_connection:post_moe` 还可把 FP32 routed、
BF16 shared 合并与 HC post 合成一次提交。注册名只有 `hyper_connection:pre_norm/post/post_moe`，
按 dtype、形状和数学能力选择，任何采用相同 H/C 数学的配置都能复用，不按模型目录名分叉。
`tpq check --cuda-ops` 的 `hyper_connection_workspace` 会同时验证公共注册选择、逐元素一致性和
输出地址复用。Kimi 不使用 H/C 数学，但它的 `residual_mix:attention`、TPHidden 固定 workspace、
packed MoE 固定输出遵循同一公共接口和“decode 不分配临时结果”的原则。

单卡 RAM+GPU 的动态路由通过公共能力
`cuda.packed_route_slots.fixed_metadata.decode` 完成。输入是 GPU 上的 Top-K
专家 ID 和固定槽目录，输出是调用方提供的指针/形状元数据与命中掩码；注册键不含
模型名。正常命中路径不调用 `.tolist()`，也不创建索引或反量化权重副本。目录未命中
时只回读 Top-K ID，再进入原有紧凑 RAM→VRAM 搬运回退。该算子和 FlashInfer MLA
动态 plan 都由 `python -m tpq check --cuda-ops` 做固定地址 CUDA Graph 验收；
MLA 同时逐字段核对单 CTA `1×78` 与双 CTA `2×39` 的官方调度。
