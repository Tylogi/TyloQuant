# TyloQuant MFQ C++ Runtime

`mfq-decode` 同时提供命令行推理与 OpenAI 兼容的本地 HTTP 服务。新 MFQ 文件把
模型配置和 tensor-free GGUF metadata 一并保存在单文件中；后者包含 tokenizer、
chat template、special-token ID 与模型 metadata。服务直接从 MFQ 内存记录初始化
llama.cpp vocab，不创建临时文件。

## macOS 原生 MLX/Metal runtime

Apple Silicon 使用独立的 C++20 + MLX C++ target，不依赖 Python runtime、
Torch、CUDA 或 NVCC。MLX wheel 必须提供 `include/`、`lib/libmlx.dylib`、
`lib/libjaccl.dylib` 和 `lib/mlx.metallib`：

```bash
MFQ_MLX_ROOT="$(python -c 'from pathlib import Path; import mlx; print(Path(mlx.__file__).resolve().parent)')"
cmake -S cpp_runtime -B build/cpp_runtime-metal -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMFQ_BUILD_METAL_RUNTIME=ON \
  -DMFQ_BUILD_CPP_SERVER=ON \
  -DMFQ_MLX_ROOT="$MFQ_MLX_ROOT" \
  -DMFQ_LLAMA_SOURCE_DIR="$MFQ_LLAMA_SOURCE_DIR" \
  -DMFQ_LLAMA_BUILD_DIR="$MFQ_LLAMA_BUILD_DIR"
cmake --build build/cpp_runtime-metal -j 8

build/cpp_runtime-metal/metal/mfq-decode-metal --self-test-metal
build/cpp_runtime-metal/metal/mfq-decode-metal \
  --mfq model.mfq --check-mfq-container
build/cpp_runtime-metal/metal/mfq-decode-metal \
  --mfq model.mfq --server --host 127.0.0.1 --port 8080

# Native DeepSeek-V4 TPQ runtime. The tokenizer may remain outside MFQ.
build/cpp_runtime-metal/metal/mfq-decode-metal \
  --mfq deepseek-v4.mfq \
  --server \
  --tokenizer-gguf tokenizer.gguf \
  --ctx-size 4096 \
  --expert-cache-gb 4
```

启用 C++ server 时，llama.cpp 仅负责内嵌或 `--tokenizer-gguf` 指定的 GGUF
tokenizer 与 chat template 解析；模型图和采样仍由原生 C++/MLX/Metal runtime
执行。构建会把 MLX、
llama/ggml dylib、`mlx.metallib` 和 WebUI 一并复制到可执行文件旁，产物使用
相对 RPATH，不依赖构建目录。

当前原生 target 已覆盖：

- dense F16/F32、NINT1–NINT8、NINT8-0；
- NVQ、NPQ、NEPQ 全部公开格式及 HSG1 rotation；
- TPQ-I4G64 与 TPQ-X/W/V/VV 各自合法的 8/12/14/16-bit index storage
  （读取器继续兼容旧 CCCP 标签）；
- GEMV、2–16 row MMQ、packed GEMM，以及大 M 临时反量化后的 MLX dense GEMM；
- NINT/NINT8/VQ/TPQ heterogeneous QKV/FFN grouped dispatch、TPQ O-LoRA
  grouped-row kernel 和 NINTM routed MoE；
- Qwen3.5/Qwen3.6 的 full-attention/Gated DeltaNet、KV/SSM cache、文本及
  三轴 mRoPE 坐标、sampling/generation；
- DeepSeek-V4 的 hyper-connection、ratio 0/4/128 compressor/indexer cache、
  sparse attention、hash/ordinary MoE、完整 causal-LM 与 generation；
- 自动按 MFQ architecture 选择模型图的 OpenAI 兼容 C++ server/WebUI。

DeepSeek-V4 的 TPQ expert 使用全模型共享的 packed-index LRU。NINTM 记录通过
`ifstream + seek + read` 按精确 byte range 读取，不使用 mmap，也不先复制整层
expert blob；每次最多以 16 行 route 为一个工作块。`--expert-cache-gb` 是缓存
expert index 的软预算，不是进程总驻留内存上限：当前 active working set 必须保留，
即使它本身超过预算。默认 4 GiB，设为 0 时只保留当前 active expert；共享 codebook
不计入该预算，并且每个 projection 只驻留一份。

尚未接入原生 C++ 整图的是视觉 encoder 本身；Qwen 的三轴 mRoPE 坐标接口已经存在，
但图片预处理和视觉 token 生成仍需独立实现。

## 构建

在 Visual Studio x64 Developer PowerShell 中运行：

```powershell
cmake -S cpp_runtime -B build/cpp_runtime -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_runtime -j 8
```

`MFQ_LLAMA_BUILD_DIR` 指向含 `src/llama.lib` 与 `bin/*.dll` 的 llama.cpp 构建目录。
`MFQ_LLAMA_SOURCE_DIR` 必须指向生成该构建的同一份 llama.cpp 源码，用于
`common/chat.h` 与 Jinja parser 头文件。

## 启动服务

```powershell
build/cpp_runtime/mfq-decode.exe `
  --mfq model.mfq `
  --server `
  --model-name mfq-qwen3.5-9b `
  --ctx-size 32768 `
  --host 127.0.0.1 `
  --port 8080
```

设置 `--api-key KEY` 或环境变量 `MFQ_API_KEY` 可启用 Bearer 认证。
服务模式默认把上下文限制为 32768，避免按模型声明的 262144 长度一次性分配 KV cache；
可通过 `--ctx-size` 调整，但不能超过模型配置上限。

`/v1/chat/completions` 使用 GGUF tokenizer 中的 Jinja chat template（可内嵌于
MFQ，也可来自 `--tokenizer-gguf`）。响应由 llama.cpp 的通用 chat parser 分成
`reasoning_content`、`content` 与 `tool_calls`，流式响应也保持这三个
通道独立。历史消息中的 `reasoning_content` 会原样交给模板；模板可自行丢弃旧思考，
也可通过 `chat_template_kwargs.preserve_reasoning=true` 请求保留。WebUI 默认采用模板
行为，并提供“排除历史思考”开关。

## 多 GPU 张量并行

`--tensor-parallel` 指定参与推理的 CUDA 设备，`--tensor-split` 指定各卡的
权重比例：

```powershell
build/cpp_runtime/mfq-decode.exe `
  --mfq model.mfq `
  --ids 2,106 `
  --gen 32 `
  --tensor-parallel 0,1 `
  --tensor-split 1,1
```

服务模式只接受包含内置模型配置、tokenizer 与 chat template 的 MFQ。可用下列工具
把已有权重重新封装成当前单文件格式，无需重新量化：

```powershell
python -m mfq.tools.pack_runtime_assets `
  --input legacy.mfq `
  --output self-contained.mfq `
  --config config.json `
  --tokenizer-gguf tokenizer.gguf

build/cpp_runtime/mfq-decode.exe `
  --mfq self-contained.mfq `
  --check-runtime-assets
```

最后一条命令检查内置 config、vocab、chat template 与 special-token ID。

## 分片 MFQ

量化器可通过 `--split-max-size 4G` 或 `--split-max-tensors 128` 直接输出
`model-00001-of-00004.mfq`。完整单文件也可流式切分：

```powershell
python -m mfq.tools.split_mfq `
  --input model.mfq `
  --output shards/model.mfq `
  --split-max-size 4G
```

`--mfq` 可传入组内任意一片。runtime 自动发现全组文件并直接读取各片的张量
offset，不合并临时文件，也不把完整权重读入主存。可用
`--check-mfq-container` 单独检查片号、缺片、重复记录和全局计数。

NINT、NINT8-0、NVQ/NPQ 的 Dense 权重会直接从压缩表示切分；NINTM routed
expert 的 NINT、NINT8-0、NVQ/NPQ、NEPQ cohort 也按输出行切分。Gate/Up
保持配对行顺序，Down 与 Attention 输出在主卡汇合。Dense/Shared FFN 会在各卡
完成 Gate、Up、激活和本地 Down 后，仅汇合 hidden-size partial。设备列表的第一张卡
负责路由、Attention、KV cache 和最终采样。`--tensor-split` 省略时等比分配。

张量并行与 `--moe-gpu-cache-gb` 当前不能同时启用。

## Important Neuron Dense FFN

IN 模型把 imatrix Top-K 中间神经元从普通 Dense FFN 记录中移出：原张量名保存
低精度 cold 分支，同名加 `.in_high` 保存高精度 hot 分支；
`__mfq_asset__/important_neurons.v1` 保存每层的神经元下标。Gate/Up 按输出行切分，
Down 按输入列切分，运行时计算两个独立 SwiGLU FFN 后逐元素相加。

M=1 decode 使用两个 CUDA stream 并行执行 hot/cold 分支。M>1 prefill 和 KLD
在父 stream 顺序执行，避免每层独立 stream 的 CUDA allocator 池累积；两条路径的
张量划分、精度与求和顺序保持一致。

## Routed MoE CPU/GPU 异构推理

`--moe-gpu-cache-gb N` 把 routed expert 的压缩权重保存在 CPU，并用至多
`N GiB` 显存建立专家粒度 GPU LRU 缓存。NINT、NINT8-0、NVQ/NPQ 与 NEPQ
cohort 共用该机制；Dense、共享专家、Attention、Embedding 和 LM head 继续常驻 GPU。

```powershell
build/cpp_runtime/mfq-decode.exe `
  --mfq moe-model.mfq `
  --ids 2,106 `
  --gen 32 `
  --moe-gpu-cache-gb 2
```

预算是压缩专家 arena 的硬上限。预算小于所有格式 cohort 的最小解码工作集时，
程序会在加载阶段报错。启用该选项后 CUDA Graph 自动停用；它不能与旧的
`--cpu-offload-layers` 同时使用。运行结束会输出命中、miss、eviction、H2D
字节数和实际 arena 分配量。

可选的 `--moe-cache-profile profile.json` 在模型加载阶段预热专家。JSON
描述的是来源无关的逐层专家排名或边缘频率；REAP、路由 trace 和手工统计都可以
生成它：

```json
{
  "version": 1,
  "metadata": {"source": "manual"},
  "layers": {
    "0": {"ranking": [17, 4, 91, 2]},
    "1": {"frequencies": {"8": 0.031, "43": 0.026}}
  }
}
```

排名模式按层分配预算；可跨层比较的频率会参与全局预算分配。字段和层都允许省略，
未提供数据的部分继续按需载入。Gate/Up/Down 以完整专家包预热，随后作为普通 LRU
条目参与淘汰。`moe_cache_prewarm` 单独报告加载时间和 H2D 字节，运行期
`moe_cache_stats` 从预热完成后重新计数。

浏览器打开 `http://127.0.0.1:8080/admin/` 可使用内置 WebUI。构建时页面资源会复制到
可执行文件同目录的 `web` 文件夹；也可以通过 `--web-root path` 指定资源目录。

## API

- `GET /health`
- `GET /api/status`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

```powershell
$body = @{
  model = "mfq-qwen3.5-9b"
  messages = @(@{ role = "user"; content = "你好，请介绍一下自己。" })
  max_tokens = 128
  temperature = 0.7
  top_p = 0.8
  top_k = 20
  stream = $false
  chat_template_kwargs = @{ enable_thinking = $false }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions `
  -Method Post -ContentType application/json -Body $body
```

服务支持 SSE、`stop`、`seed`、presence/frequency/repetition penalty，并校验上下文长度。
当前模型状态由单实例持有，请求会串行执行。图像、结构化 tool calls、`logprobs` 与 `min_p`
会返回明确的 `400` 错误。
