"""TPQ 生成引擎：tokenizer 封装 + 自回归生成循环（贪心 / top-p 采样）。

默认 EOS 取自 generation_config（GLM-5.2: [154820, 154827, 154829]），
<|user|>/<|observation|> 命中即停（对话模板的安全边界）。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .model import GLMModel
from .presets import load_manifest as _load_tpq_manifest

DEFAULT_EOS = [154820, 154827, 154829]
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _token_lcp(
    left: list[int] | None,
    right: list[int],
) -> int:
    """Return the exact token-ID longest common prefix."""
    if not left:
        return 0
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


@dataclass(frozen=True)
class KVPrefillStats:
    mode: str
    reason: str
    prompt_tokens: int
    baseline_tokens: int
    lcp_tokens: int
    replay_tokens: int
    suffix_tokens: int
    processed_tokens: int
    prefill_ms: float
    snapshot_bytes: int


@dataclass
class _DSV4Baseline:
    ids: list[int]
    snapshot: object


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _generation_open(
    generated: int,
    max_new: int | None,
    position: int,
    max_ctx: int | None,
) -> bool:
    """Whether another output token may be committed."""
    if max_new is not None and generated >= max_new:
        return False
    return max_ctx is None or position < max_ctx


def _make_model(
    model_dir: str,
    cache_gb: float,
    max_ctx: int,
    device: str,
    vram_cache_gb: float,
    tp_size: int = 1,
    extreme_fixed_gpu_bytes: int = 0,
):
    """按 TPQ 清单字段分派 Kimi、DeepSeek-V4 或 GLM。"""
    _root, manifest = _load_tpq_manifest(model_dir)
    cfg = manifest["config"]
    if (
        manifest.get("model_family") == "kimi_k3"
        or ("kda_layers" in cfg and "routed_hidden" in cfg)
    ):
        from .kimi_model import KimiK3TPQModel

        return KimiK3TPQModel(
            model_dir,
            cache_gb=cache_gb,
            max_ctx=max_ctx,
            device=device,
            vram_cache_gb=vram_cache_gb,
            tp_size=tp_size,
            extreme_fixed_gpu_bytes=extreme_fixed_gpu_bytes,
        ), "kimi_k3"
    if "hc_mult" in cfg or "compress_ratios" in cfg:
        # Reuse the canonical manifest parser: projection-VQ may be declared
        # per layer or by a heterogeneous per-expert precision map.
        from .store import Manifest

        projection_vq = Manifest(model_dir).projection_vq
        if tp_size != 1 and not projection_vq:
            raise ValueError(
                "--tp > 1 requires a projection-VQ DeepSeek-V4 archive"
            )
        from .dsv4model import DSV4TPQModel
        return DSV4TPQModel(model_dir, cache_gb=cache_gb, max_ctx=max_ctx,
                             device=device, vram_cache_gb=vram_cache_gb,
                             tp_size=tp_size,
                             extreme_fixed_gpu_bytes=(
                                 extreme_fixed_gpu_bytes
                             )), "dsv4"
    return GLMModel(model_dir, cache_gb=cache_gb, max_ctx=max_ctx,
                    device=device, vram_cache_gb=vram_cache_gb,
                    tp_size=tp_size), "glm"


def _dense_need_gb(model_dir: str, arch_hint: str, kv_gb: float) -> float:
    """按产物清单与 config 实际计算 dense 常驻需求（替代按架构硬编码）：
    DSV4 全 BF16 路径按 safetensors 头部精确计算展开常驻量；其他路径按
    dense.safetensors 实际大小 + head f32（vocab×hidden×4）+
    mtp/DSpark 附件 + 瞬时缓冲 1.5GB + KV。读取失败回退架构经验值。
    清单驱动使得任意档位产物（S/M/L）与任意显卡（16GB 起）都能正确自适应。"""
    fallback = (8.2 if arch_hint == "dsv4" else 13.5) + kv_gb
    try:
        _root, man = _load_tpq_manifest(model_dir)
        cfg = man["config"]
        if arch_hint == "kimi_k3":
            audit_name = man.get("dense_audit_file")
            if not audit_name:
                raise ValueError("Kimi dense audit is required")
            with open(
                os.path.join(model_dir, audit_name),
                "r",
                encoding="utf-8",
            ) as handle:
                audit = json.load(handle)
            language_bytes = sum(
                int(info.get("bytes", 0))
                for shard in audit.get("shards", {}).values()
                for name, info in shard.get("tensor_audit", {}).items()
                if name.startswith("language_model.")
            )
            router_fp32_extra = sum(
                int(info.get("bytes", 0))
                for shard in audit.get("shards", {}).values()
                for name, info in shard.get("tensor_audit", {}).items()
                if name.endswith(".block_sparse_moe.gate.weight")
            )
            if not language_bytes:
                raise ValueError("Kimi language dense bytes are missing")
            kda_state_gb = (
                len(cfg.get("kda_layers", []))
                * int(cfg["n_heads"])
                * int(cfg["head_dim"])
                * int(cfg["head_dim"])
                * 4
                / 2**30
            )
            return (
                language_bytes / 2**30
                + router_fp32_extra / 2**30
                + kda_state_gb
                + 1.5
                + kv_gb
            )
        dense_path = os.path.join(
            model_dir,
            man.get("dense_file", "dense.safetensors"),
        )
        dense_gb = os.path.getsize(dense_path) / 2**30
        head_gb = cfg["vocab"] * cfg["hidden"] * 4 / 2**30
        mtp_gb = 0.0
        dsv4_bf16_resident = False
        if arch_hint == "dsv4":
            from .capacity import dsv4_dense_runtime_bytes

            runtime_bytes = dsv4_dense_runtime_bytes(
                dense_path,
                os.environ.get("TPQ_DENSE_BF16"),
            )
            # The header calculation covers both compact and BF16-resident
            # modes and already includes lm_head exactly once.
            dense_gb = runtime_bytes / 2**30
            head_gb = 0.0
            dsv4_bf16_resident = str(
                os.environ.get("TPQ_DENSE_BF16", "")
            ).strip().lower() in {"1", "true", "all"}
        fn = man.get("dspark_file") or man.get("mtp_file")  # 清单指引（产物自包含）
        if fn and os.path.exists(os.path.join(model_dir, fn)):
            if man.get("dspark_file"):
                # DSpark：文件主体是草稿专家权重（驻 RAM/独立 LRU，不占 dense 显存），
                # dense 显存只需 stage bf16 ≈1.4GB + markov + VQ LRU ≈ 2.5GB
                if (
                    not dsv4_bf16_resident
                    or os.environ.get("TPQ_SPEC", "0") == "1"
                ):
                    mtp_gb = 2.5
            else:  # GLM MTP：dense 附件整体驻显存，按文件实际大小计
                mtp_gb = os.path.getsize(os.path.join(model_dir, fn)) / 2**30
        return dense_gb + head_gb + mtp_gb + 1.5 + kv_gb
    except Exception:
        return fallback


def _glm_startup_kv_gb(max_ctx: int, *, latent: bool) -> float:
    """Estimate only GLM's initial dynamic-KV working set.

    ``max_ctx`` is a logical admission ceiling; GLMModel grows KV tensors on
    demand instead of allocating that ceiling at startup.  Reserving the full
    model limit here would make the declared 1M context look like 90+ GiB of
    fixed VRAM and incorrectly force CUDA startup to CPU.  Expert arenas are
    physically shrunk later by the existing runtime VRAM monitor as KV grows.
    """
    logical_ctx = max(0, int(max_ctx))
    if latent:
        initial_ctx = min(logical_ctx, 4096)
        return 2.3 + 0.09 * initial_ctx / 1024
    initial_ctx = min(logical_ctx, 1024)
    return 5.0 * initial_ctx / 1024


def _safe_expert_budget(*, limit_bytes: int, allocated_bytes: int,
                        expert_bytes: int, requested_bytes: int,
                        reserve_bytes: int, min_bytes: int = 2**29) -> int:
    """Cap expert VRAM from actual fixed allocations, not model-size estimates."""
    fixed_bytes = max(0, int(allocated_bytes) - int(expert_bytes))
    room = max(0, int(limit_bytes) - fixed_bytes - int(reserve_bytes))
    return max(int(min_bytes), min(int(requested_bytes), room))


def _dense_file_paths(
    model_dir: str,
    manifest: dict,
) -> tuple[str, ...]:
    """Resolve manifest-declared Dense files without opening tensor bodies."""
    files = manifest.get("dense_files")
    if files is None:
        files = [manifest.get("dense_file", "dense.safetensors")]
    dense_root = str(
        (manifest.get("nonexpert") or {}).get("path", "dense")
    ).strip("/\\")
    resolved = []
    for filename in files:
        value = str(filename).replace("/", os.sep)
        direct = os.path.join(model_dir, value)
        nested = os.path.join(model_dir, dense_root, value)
        resolved.append(direct if os.path.exists(direct) else nested)
    return tuple(os.path.abspath(path) for path in resolved)


def _trim_process_heap() -> None:
    """Return large transient Dense read buffers to the host OS when possible."""
    if os.name != "posix":
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (OSError, TypeError):
        pass


class Engine:
    """GLM-5.2-CCCP 的生成引擎（CPU / CUDA，内存显存自动适配）。

    cache_gb / vram_cache_gb 传 None 时自动计算：
      RAM 预算  = 可用内存 − (运行时 2GB + f32 常驻 4.5GB + KV cache + 安全 3GB)
      VRAM 预算 = 空闲显存 − (dense 常驻 ≈13.5GB + KV cache + 安全 1GB)
    显存不足以常驻 dense 时自动回退 CPU 模式并提示。
    共享显存防线（WDDM）：初始化时按 空闲显存−预留(TPQ_VRAM_RESERVE_GB, 默认 1.25GB)
    给 torch 分配器设 per-process 硬上限，宁可 OOM 降档重试也不让驱动把显存页
    换到共享显存（带宽掉到 PCIe 级、整轮同步卡顿）。
    """

    def __init__(
        self,
        model_dir: str,
        cache_gb: float | None = None,
        max_ctx: int = 2048,
        quiet: bool = False,
        device: str = "cpu",
        vram_cache_gb: float | None = None,
        tp_size: int = 1,
        dense_residency: str = "auto",
        extreme_mode: bool | None = None,
    ):
        import psutil
        t0 = time.time()
        self.quiet = quiet
        requested_extreme = extreme_mode
        if extreme_mode is True:
            from .extreme import configure_extreme_environment

            configure_extreme_environment()
        self.extreme_mode = bool(
            extreme_mode is True
            or os.environ.get("TPQ_EXTREME_MODE", "0") != "0"
        )
        self.auto_extreme_decision = None
        self.extreme_strategy = "disabled"
        extreme_archive = None
        if tp_size <= 0:
            raise ValueError("tp_size must be positive")
        self.tp_size = int(tp_size)
        if self.extreme_mode and (self.tp_size != 1 or device != "cuda"):
            raise ValueError("极限模式要求单卡 device='cuda', tp_size=1")
        if self.extreme_mode:
            dense_residency = "gpu"
        dense_residency = str(dense_residency).strip().lower()
        if dense_residency not in {"auto", "gpu"}:
            raise ValueError("dense_residency must be 'auto' or 'gpu'")
        if dense_residency == "gpu" and device != "cuda":
            raise ValueError("dense_residency='gpu' requires device='cuda'")
        self.dense_residency = {
            "requested": dense_residency,
            "actual": "host",
            "host_mirror_bytes": 0,
        }
        ram_mirror = None
        self._vram_limit_bytes = 0
        self._vram_runtime_reserve_gb = 0.0
        # 架构判定（先读一次 TPQ 清单，供 RAM/VRAM 开销与模型分派共用）
        arch_hint = "glm"
        _manifest: dict = {}
        try:
            _root, _manifest = _load_tpq_manifest(model_dir)
        except (OSError, ValueError, KeyError, TypeError):
            _manifest = {}
        if _manifest:
            _cfg = _manifest["config"]
            if (
                _manifest.get("model_family") == "kimi_k3"
                or (
                    "kda_layers" in _cfg
                    and "routed_hidden" in _cfg
                )
            ):
                arch_hint = "kimi_k3"
            elif "hc_mult" in _cfg or "compress_ratios" in _cfg:
                arch_hint = "dsv4"
        if (
            not self.extreme_mode
            and requested_extreme is None
            and os.environ.get("TPQ_AUTO_EXTREME", "1") != "0"
        ):
            from .extreme import (
                configure_extreme_environment,
                detect_auto_extreme,
            )

            normal_reserve = float(
                os.environ.get("TPQ_RESIDENT_RESERVE_GB", "32")
            )
            self.auto_extreme_decision = detect_auto_extreme(
                model_dir,
                max_ctx=max_ctx,
                device=device,
                tp_size=self.tp_size,
                normal_ram_reserve_gib=normal_reserve,
                environment=os.environ,
            )
            if self.auto_extreme_decision.activate:
                configure_extreme_environment()
                self.extreme_mode = True
                dense_residency = "gpu"
                self.dense_residency["requested"] = "gpu"
                if not quiet:
                    decision = self.auto_extreme_decision
                    print(
                        "[tpq-auto] RAM 单侧安全容量不足，自动切换极限模式："
                        f"专家 {decision.expert_bytes / 2**30:.2f}GiB；"
                        f"转入 GPU {decision.spill_bytes / 2**30:.2f}GiB；"
                        f"GPU 专家余量 "
                        f"{decision.gpu_expert_capacity / 2**30:.2f}GiB",
                        flush=True,
                    )
        if self.extreme_mode:
            from .extreme import inspect_compact_projection_archive

            extreme_archive = inspect_compact_projection_archive(model_dir)
            self.extreme_strategy = "layered"
        # Tokenizer 是运行时硬依赖，必须在数百 GiB 权重加载之前验证并初始化。
        # 旧顺序在模型完整 preload 后才 import ``tokenizers``，一旦 Python
        # 环境缺包，会白白消耗数分钟加载时间和大量磁盘读。Kimi 继续使用其
        # 自身 tokenizer 适配，GLM/DeepSeek 使用标准 tokenizer.json。
        if arch_hint == "kimi_k3":
            from .kimi_tokenizer import KimiTokenizer

            prepared_tokenizer = KimiTokenizer(model_dir)
        else:
            from tokenizers import Tokenizer

            prepared_tokenizer = Tokenizer.from_file(
                os.path.join(model_dir, "tokenizer.json")
            )
        # RAM 开销按架构。普通 DSV4 允许 paged KV 按需增长；极限模式的
        # GPU-only 专家不可再收缩，所以必须在放置专家前为完整声明上下文预留。
        if arch_hint == "dsv4":
            if self.extreme_mode:
                from .capacity import dsv4_context_runtime_bytes

                kv_gb = (
                    dsv4_context_runtime_bytes(_cfg, max_ctx).total_bytes
                    / 2**30
                )
            else:
                kv_gb = 0.2
            ram_overhead = 2.0 + 2.1 + kv_gb + 3.0   # f32 2.1 + 安全 3（用户实测调优）
        else:
            # GLM：MLA 潜变量 KV（默认开）≈0.09MB/token + 吸收矩阵 2.3GB；
            # TPQ_LATENT_KV=0 回退逐头全量 K/V ≈5MB/token。KV 按需增长，
            # 启动预算不能把模型声明的逻辑上限当作已分配显存。
            kv_gb = _glm_startup_kv_gb(
                max_ctx,
                latent=os.environ.get("TPQ_LATENT_KV", "1") != "0",
            )
            ram_overhead = 2.0 + 4.5 + kv_gb + 6.0  # 安全余量 6GB
        if arch_hint == "kimi_k3":
            initial_ctx = min(max(0, int(max_ctx)), 4096)
            kv_gb = 0.5 + 0.027 * initial_ctx / 1024
            ram_overhead = 2.0 + kv_gb + 6.0
        if self.extreme_mode:
            from .extreme import effective_available_memory_bytes

            avail_ram = effective_available_memory_bytes() / 2**30
        else:
            avail_ram = psutil.virtual_memory().available / 2**30
        auto_ram = (
            max(0.0, avail_ram - 1.0)
            if self.extreme_mode
            else max(2.0, avail_ram - ram_overhead)
        )

        dev = device
        auto_vram = vram_cache_gb
        extreme_fixed_gpu_bytes = 0
        if device == "cuda":
            if not torch.cuda.is_available():
                if dense_residency == "gpu":
                    raise RuntimeError(
                        "Dense 要求 GPU 常驻，但当前 CUDA 不可用"
                    )
                print("[tpq] 无 CUDA，回退 CPU 模式", flush=True)
                dev = "cpu"
            else:
                if self.tp_size > torch.cuda.device_count():
                    raise RuntimeError(
                        f"tp={self.tp_size} but only "
                        f"{torch.cuda.device_count()} CUDA devices are visible"
                    )
                if (
                    self.tp_size > 1
                    and (
                        arch_hint == "glm"
                        or (
                            arch_hint == "kimi_k3"
                            and os.environ.get(
                                "TPQ_KIMI_TP_PACKED_HYBRID",
                                "0",
                            )
                            != "0"
                        )
                    )
                    and os.environ.get("TPQ_RAM_MIRROR", "0") == "1"
                ):
                    from .ramcache import ModelRamMirror

                    ram_mirror = ModelRamMirror(
                        model_dir,
                        exclude_paths=_dense_file_paths(
                            model_dir,
                            _manifest,
                        ),
                    )
                    ram_mirror.start()
                visible_ranks = min(
                    max(1, self.tp_size),
                    torch.cuda.device_count(),
                )
                rank_memory = []
                for rank in range(visible_ranks):
                    with torch.cuda.device(rank):
                        rank_memory.append(torch.cuda.mem_get_info(rank))
                free_v = min(item[0] for item in rank_memory) / 2**30
                total_v = min(item[1] for item in rank_memory) / 2**30
                # 共享显存防线：WDDM 下分配顶满物理显存会被驱动换页到内存（共享显存，
                # 带宽掉到 PCIe 级，整轮同步卡顿）。给本进程分配器设硬上限 =
                # 空闲显存 − 系统预留，宁可 OOM 报错也绝不触发共享显存换页。
                reserve_gb = float(os.environ.get("TPQ_VRAM_RESERVE_GB", "1.25"))
                extreme_vram_cap_gb = max(
                    0.0,
                    float(os.environ.get("TPQ_EXTREME_VRAM_CAP_GB", "0")),
                )
                planning_free_v = (
                    min(free_v, extreme_vram_cap_gb)
                    if extreme_vram_cap_gb > 0
                    else free_v
                )
                fractions = []
                limits = []
                for rank, (free_bytes, total_bytes) in enumerate(
                    rank_memory
                ):
                    process_available = free_bytes
                    minimum_fraction = 0.10
                    if extreme_vram_cap_gb > 0:
                        process_available = min(
                            process_available,
                            int(extreme_vram_cap_gb * 2**30),
                        )
                        minimum_fraction = 0.01
                    fraction = max(
                        minimum_fraction,
                        min(
                            0.99,
                            (
                                process_available
                                - int(reserve_gb * 2**30)
                            )
                            / total_bytes,
                        ),
                    )
                    torch.cuda.set_per_process_memory_fraction(
                        fraction,
                        rank,
                    )
                    fractions.append(fraction)
                    limits.append(int(fraction * total_bytes))
                frac = min(fractions)
                self._vram_limit_bytes = min(limits)
                if not quiet:
                    print(f"[tpq] 显存适配: 物理 {total_v:.1f}GB / 空闲 {free_v:.1f}GB → "
                          f"本进程上限 {frac * total_v:.1f}GB（预留 {reserve_gb:.2f}GB 防共享显存）",
                          flush=True)
                    if extreme_vram_cap_gb > 0:
                        print(
                            "[tpq-extreme] 测试显存硬上限："
                            f"{extreme_vram_cap_gb:.2f}GiB",
                            flush=True,
                        )
                # dense 常驻需求按架构：GLM ≈13.5GB（int4 9.2 + lm_head 3.8 + router 0.5），
                # DSV4 ≈10.5GB（dense 一次性反量化 bf16 常驻 ≈7.2 + head bf16 ≈1.1 + DSpark ≈2.2；
                # bf16 消除逐调用反量化，是 attn 段的关键提速；+ KV + 安全 2GB（悬崖余量）
                dense_need = _dense_need_gb(model_dir, arch_hint, kv_gb)
                if self.extreme_mode and extreme_archive is not None:
                    from .extreme import (
                        EXTREME_GPU_LOAD_WORKSPACE_GIB,
                        GIB,
                        choose_extreme_strategy,
                    )

                    dense_resident_need = max(
                        0.0,
                        dense_need - EXTREME_GPU_LOAD_WORKSPACE_GIB,
                    )

                    self.extreme_strategy = choose_extreme_strategy(
                        compact_expert_bytes=extreme_archive.expert_bytes,
                        fixed_gpu_bytes=int(dense_resident_need * GIB),
                        gpu_limit_bytes=self._vram_limit_bytes,
                    )
                    if self.extreme_strategy == "layered":
                        # The hybrid pool turns this estimate into a real CUDA
                        # allocation before placing any expert. Dense later
                        # replaces that allocation, so capacity is proven
                        # without keeping a second full weight copy.
                        extreme_fixed_gpu_bytes = int(
                            dense_resident_need * GIB
                        )
                    if self.extreme_strategy == "full-gpu":
                        # Reuse the same model-independent packed resident
                        # pool as profile=resident.  Extreme remains a capacity
                        # policy; it must not force RAM/H2D when the complete
                        # compact archive already fits one GPU.
                        os.environ["TPQ_PACKED_FULL_GPU"] = "1"
                    if not quiet:
                        print(
                            "[tpq-extreme] 公共紧凑归档："
                            f"{len(extreme_archive.layers)} 层 / "
                            f"{extreme_archive.expert_bytes / GIB:.2f}GiB；"
                            f"策略={self.extreme_strategy}",
                            flush=True,
                        )
                # 余量按架构：GLM 的 dense 常驻 + 吸收矩阵(2.1GB) + 瞬态反量化块贴近
                # 分配器硬上限，实测 1-2GB 余量仍会在 decode 中 OOM → GLM 3GB，DSV4 1GB
                #（GLM 专家本就走 RAM/磁盘流式，显存缓存价值低）
                margin = (
                    0.0
                    if self.extreme_mode
                    else (3.0 if arch_hint == "glm" else 1.0)
                )
                if planning_free_v < dense_need + margin:
                    if dense_residency == "gpu":
                        raise RuntimeError(
                            "Dense 要求 GPU 常驻，但空闲显存 "
                            f"{planning_free_v:.1f}GB < 需要 "
                            f"{dense_need + margin:.1f}GB"
                        )
                    print(f"[tpq] 显存不足（空闲 {planning_free_v:.1f}GB < 需要 {dense_need + margin:.1f}GB），"
                          f"回退 CPU 模式", flush=True)
                    dev = "cpu"
                elif vram_cache_gb is None:
                    # 显存余量 2GB（DSV4）：顶到 100% 会触发分配器 cudaFree+同步回收（悬崖 ×4）
                    # 极限模式把完整可用余量交给 packed pool；pool 会先用
                    # extreme_fixed_gpu_bytes 建立真实 CUDA 占位，再从剩余空间
                    # 放置专家。Dense 流式加载时直接复用占位块，不会重复扣除。
                    auto_vram = (
                        planning_free_v
                        if self.extreme_mode
                        else max(
                            0.5,
                            planning_free_v - dense_need - margin,
                        )
                    )
        if cache_gb is None:
            cache_gb = auto_ram
        if vram_cache_gb is None:
            vram_cache_gb = auto_vram if dev == "cuda" else 0.0
        if not quiet:
            if dev == "cuda" and self.tp_size > 1 and arch_hint == "glm":
                print(
                    f"[tpq] 内存适配: 可用RAM {avail_ram:.1f}GB；"
                    f"TP={self.tp_size} 优先全显存专家（运行期专家 RAM/H2D=0）；"
                    f"容量不足时自动回退 RAM {cache_gb:.1f}GB / "
                    f"主卡显存 {vram_cache_gb:.1f}GB",
                    flush=True,
                )
            else:
                print(
                    f"[tpq] 内存适配: 可用RAM {avail_ram:.1f}GB → "
                    f"专家缓存 {cache_gb:.1f}GB"
                    + (
                        f"；显存缓存 {vram_cache_gb:.1f}GB"
                        if dev == "cuda"
                        else ""
                    ),
                    flush=True,
                )

        if ram_mirror is not None and not ram_mirror.wait_and_activate():
            ram_mirror = None

        retry_vram_cache_gb = None
        try:
            self.model, self.arch = _make_model(
                model_dir,
                cache_gb=cache_gb,
                max_ctx=max_ctx,
                device=dev,
                vram_cache_gb=vram_cache_gb or 4.0,
                tp_size=self.tp_size,
                extreme_fixed_gpu_bytes=extreme_fixed_gpu_bytes,
            )
            self.model.preload()
        except torch.cuda.OutOfMemoryError as oom_error:
            if self.extreme_mode:
                raise RuntimeError(
                    "极限模式显存不足：Dense、GPU 专家层、Top-K staging 与当前"
                    f" max_ctx={max_ctx} 无法同时容纳。请降低 --max-ctx、"
                    "关闭其他显存进程或换用更小模型。底层分配："
                    f"{oom_error}"
                ) from oom_error
            if dev != "cuda" or (vram_cache_gb or 4.0) <= 1.0:
                raise
            # Leave the exception handler before retrying.  Its traceback keeps
            # preload frames alive; retrying inside this block used to retain
            # the failed model's dense weights and made the second attempt OOM
            # as well.
            self.model = None
            retry_vram_cache_gb = max(0.5, (vram_cache_gb or 4.0) / 2)
            print(f"[tpq] 显存触顶（硬上限保护），显存缓存降至 "
                  f"{retry_vram_cache_gb:.1f}GB 重试",
                  flush=True)
        if retry_vram_cache_gb is not None:
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            self.model, self.arch = _make_model(
                model_dir,
                cache_gb=cache_gb,
                max_ctx=max_ctx,
                device=dev,
                vram_cache_gb=retry_vram_cache_gb,
                tp_size=self.tp_size,
                extreme_fixed_gpu_bytes=0,
            )
            self.model.preload()
        full_resident = bool(
            getattr(getattr(self.model, "pool", None), "full_resident", False)
        )
        compact_cpu_resident = bool(
            dev == "cpu"
            and getattr(
                getattr(self.model, "pool", None),
                "compact_full_resident",
                False,
            )
        )
        if dev == "cuda":
            released, dense_paths = (
                self.model.store.release_dense_ram_blob()
            )
            mirror_released = (
                ram_mirror.release_paths(dense_paths)
                if ram_mirror is not None
                else 0
            )
            import gc as _dense_gc

            _dense_gc.collect()
            _trim_process_heap()
            self.dense_residency = {
                "requested": dense_residency,
                "actual": "gpu-only",
                "host_mirror_bytes": max(released, mirror_released),
            }
            if not quiet:
                print(
                    "[tpq] Dense 驻留：GPU-only；"
                    "CPU 仅保留启动期流式缓冲，运行期源镜像已释放"
                    + (
                        f" {max(released, mirror_released) / 2**30:.2f}GB"
                        if max(released, mirror_released)
                        else ""
                    ),
                    flush=True,
                )
        if (
            ram_mirror is not None
            and getattr(
                getattr(self.model, "pool", None),
                "retains_store_ram_blobs",
                False,
            )
        ):
            self._ram_mirror = ram_mirror
            ram_mirror = None
            if not quiet:
                print(
                    "[tpq] RAM 镜像直接作为 packed 专家常驻存储；"
                    "不建立第二份专家索引",
                    flush=True,
                )
        if ram_mirror is not None:
            self.model.store.release_ram_blobs()
            released = ram_mirror.release()
            import gc as _gc

            _gc.collect()
            if not quiet:
                print(
                    f"[tpq] RAM staging 已释放 "
                    f"{released / 2**30:.2f}GB；推理期不保留模型文件镜像",
                    flush=True,
                )
        if dev == "cuda" and not full_resident:
            default_runtime = 1.5 if arch_hint == "dsv4" else 3.0
            self._vram_runtime_reserve_gb = float(os.environ.get(
                "TPQ_VRAM_RUNTIME_GB", str(default_runtime)
            ))
            self._cap_expert_cache(
                self._vram_runtime_reserve_gb, "dense/运行时安全余量"
            )
        # 动态显存监测：滞回调节专家显存缓存预算（防其他进程抢占/碎片化/小显卡
        # 顶满物理显存触发共享显存换页）；TPQ_VRAM_WATCH=0 关闭
        if (
            dev == "cuda"
            and not full_resident
            and not self.extreme_mode
            and os.environ.get("TPQ_VRAM_WATCH", "1") != "0"
        ):
            _pool = getattr(self.model, "pool", None)
            if (
                _pool is not None
                and getattr(_pool, "supports_vram_watch", True)
            ):
                from .vramwatch import VramWatch
                self._vwatch = VramWatch(
                    _pool, max_budget=_pool.budget, quiet=quiet)
                self._vwatch.start()
        self.tok = prepared_tokenizer
        gc = os.path.join(model_dir, "generation_config.json")
        self.eos = DEFAULT_EOS
        if os.path.exists(gc):
            with open(gc, "r", encoding="utf-8") as f:
                e = json.load(f).get("eos_token_id", DEFAULT_EOS)
                self.eos = [e] if isinstance(e, int) else list(e)
        self.quiet = quiet
        self._cache_ids: list[int] | None = None   # KV 中已缓存的 token 前缀（多轮复用）
        self._cache_via_spec = False   # 缓存是否由投机路径建立（DSpark 环覆盖一致才可直接复用）
        self._kv_baseline: _DSV4Baseline | None = None
        self.last_kv_stats: KVPrefillStats | None = None
        self._kv_prefill_events = None
        if not quiet:
            if full_resident:
                pool = self.model.pool
                print(
                    f"[tpq] 模型加载完成（{time.time() - t0:.1f}s）："
                    f"TP={self.model.effective_tp_size} routed experts "
                    f"{pool.gpu_storage_bytes / 2**30:.2f}GB 全显存常驻，"
                    f"主机专家 {pool.host_expert_bytes / 2**30:.2f}GB",
                    flush=True,
                )
            elif compact_cpu_resident:
                pool = self.model.pool
                print(
                    f"[tpq] 模型加载完成（{time.time() - t0:.1f}s）："
                    f"CPU 专家执行镜像 {pool.host_expert_bytes / 2**30:.2f}GB "
                    f"全量常驻；cpu_compile="
                    f"{getattr(pool, 'cpu_compile_mode', 'off')}；"
                    f"expanded_index_bytes="
                    f"{getattr(pool, 'expanded_index_bytes', 0)}",
                    flush=True,
                )
            else:
                pool = getattr(self.model, "pool", None)
                extreme_detail = ""
                if self.extreme_mode and pool is not None:
                    ram_layers = len(
                        getattr(pool, "extreme_ram_layers", ())
                    )
                    gpu_layers = len(
                        getattr(pool, "extreme_gpu_layers", ())
                    )
                    ratio = float(
                        getattr(pool, "extreme_storage_ratio", 0.0)
                    )
                    extreme_detail = (
                        f"；极限常驻 RAM={ram_layers}层/GPU={gpu_layers}层"
                        f"/紧凑开销={ratio:.3f}x"
                    )
                print(
                    f"[tpq] 模型加载完成（{time.time() - t0:.1f}s）"
                    f"专家缓存预算 {cache_gb:.0f}GB"
                    f"{extreme_detail}",
                    flush=True,
                )

    def _cap_expert_cache(self, reserve_gb: float, reason: str) -> int | None:
        """Immediately enforce a cache ceiling within the allocator hard limit."""
        pool = getattr(getattr(self, "model", None), "pool", None)
        if (
            pool is None
            or getattr(pool, "full_resident", False)
            or getattr(pool, "fixed_extreme_residency", False)
            or getattr(pool, "manages_per_rank_budget", False)
            or not self._vram_limit_bytes
        ):
            return None
        allocated = torch.cuda.memory_allocated()
        expert_storage = getattr(pool, "gpu_storage_bytes", pool.bytes)
        new_budget = _safe_expert_budget(
            limit_bytes=self._vram_limit_bytes,
            allocated_bytes=allocated,
            expert_bytes=expert_storage,
            requested_bytes=pool.budget,
            reserve_bytes=int(reserve_gb * 2**30),
        )
        old_budget = pool.budget
        fixed_gb = max(0, allocated - expert_storage) / 2**30
        if new_budget < old_budget:
            arena_bytes = getattr(pool, "gpu_arena_bytes", 0)
            allocated_before = allocated
            resize = getattr(pool, "resize_gpu_arenas", None)
            resized = arena_bytes > new_budget and callable(resize)
            if resized:
                old_arena, new_arena = resize(new_budget)
            else:
                pool.trim_to(new_budget)
                old_arena = new_arena = arena_bytes
            torch.cuda.empty_cache()
            allocated_after = torch.cuda.memory_allocated()
            if (
                resized
                and old_arena > new_arena
                and allocated_after >= allocated_before
            ):
                raise RuntimeError(
                    "expert arena budget shrank without releasing CUDA allocations: "
                    f"arena {old_arena / 2**30:.2f}->{new_arena / 2**30:.2f} GiB, "
                    f"allocated {allocated_before / 2**30:.2f}->"
                    f"{allocated_after / 2**30:.2f} GiB"
                )
            if not self.quiet:
                detail = (
                    f"；arena {old_arena / 2**30:.1f}→{new_arena / 2**30:.1f}GB"
                    f"；allocated {allocated_before / 2**30:.1f}→"
                    f"{allocated_after / 2**30:.1f}GB"
                    if resized
                    else ""
                )
                print(
                    f"[tpq] 显存缓存安全封顶: {old_budget / 2**30:.1f}GB"
                    f" → {new_budget / 2**30:.1f}GB"
                    f"（常驻 {fixed_gb:.1f}GB + {reason} {reserve_gb:.1f}GB）",
                    f"{detail}",
                    flush=True,
                )
        watcher = getattr(self, "_vwatch", None)
        if watcher is not None:
            watcher.max_budget = min(watcher.max_budget, new_budget)
        return new_budget

    def _with_kv_capacity_retry(self, fn, *args, committed: int = 0, **kwargs):
        """Free expert VRAM and retry one transactional DSV4 page reservation."""
        from .dsv4cache import ContextCapacityError

        try:
            return fn(*args, **kwargs)
        except ContextCapacityError:
            self._cap_expert_cache(1.0, "KV cache 扩容")
            try:
                return fn(*args, **kwargs)
            except ContextCapacityError as final:
                final.committed = committed
                raise

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def new_decode_stream(self, *, skip_special_tokens: bool = False):
        """Create a stateful tokenizer stream for exact incremental decoding."""
        factory = getattr(self.tok, "new_decode_stream", None)
        if factory is not None:
            return factory(
                skip_special_tokens=skip_special_tokens,
            )
        from tokenizers.decoders import DecodeStream

        return DecodeStream(skip_special_tokens=skip_special_tokens)

    def reset(self) -> None:
        self.model.reset_kv()
        self._cache_ids = None
        self._cache_via_spec = False
        self._kv_baseline = None
        self.last_kv_stats = None
        self._kv_prefill_events = None
        dsp = getattr(self, "_dsp", None)
        if dsp is not None:
            dsp.reset()

    # ---- 多轮 KV 复用（省掉历史重 prefill）----
    def _kv_prefix_len(self, ids: list[int]) -> int:
        """上轮缓存的 token 序列（prompt+回复）仍是本轮 prompt 的严格前缀时返回其长度，
        调用方只需增量 prefill 后缀；否则 0（全量重置重跑）。
        逐 token id 精确比对，正确性不依赖 tokenizer decode→encode 往返稳定性；
        think 开启时思维链不回喂、前缀必然失配 → 自动回退全量（符合官方模板）。"""
        cached = getattr(self, "_cache_ids", None)
        if cached and len(cached) < len(ids) and ids[:len(cached)] == cached:
            return len(cached)
        return 0

    def _prefill_glm_suffix(
        self,
        ids: list[int],
        skip: int,
    ) -> torch.Tensor:
        """Prefill a short GLM prompt or exact-prefix chat suffix.

        Full-GPU CodeGEMM stores experts in its decode-optimized Psumbook
        layout.  The old multi-token path unpacked hundreds of routed experts
        per layer for a normal 30-50 token follow-up.  Replaying the already
        fused single-token path is substantially faster and also measures
        closer to the FP8 KLD baseline.  Keep RAM expert mode and large
        prefill batches unchanged: their cache/H2D reuse has different costs.
        """
        suffix = ids[skip:]
        pool = getattr(self.model, "pool", None)
        try:
            max_sequential = int(os.environ.get(
                "TPQ_GLM_SEQUENTIAL_PREFILL_MAX",
                "512",
            ))
        except ValueError:
            max_sequential = 0
        sequential = (
            0 < len(suffix) <= max_sequential
            and os.environ.get(
                "TPQ_GLM_SEQUENTIAL_PREFILL",
                "1",
            )
            != "0"
            and bool(getattr(pool, "full_resident", False))
        )
        if not sequential:
            return self.model.forward(suffix)
        if skip == 0:
            graph_target = max(
                0,
                int(
                    getattr(self.model, "cfg", {}).get(
                        "n_layers",
                        4,
                    )
                )
                - 4,
            )
            graph_needs_warmup = (
                graph_target > 0
                and os.environ.get(
                    "TPQ_ATTENTION_GRAPH",
                    "1",
                )
                != "0"
                and os.environ.get(
                    "TPQ_GLM_QB_SPLIT",
                    "1",
                )
                != "0"
                and not getattr(
                    self.model,
                    "_attention_graph_failed",
                    False,
                )
                and len(
                    getattr(
                        self.model,
                        "_attention_graphs",
                        {},
                    )
                )
                < graph_target
            )
            if graph_needs_warmup:
                # Capture with a sacrificial token before it can influence
                # the real prompt state, then retain only the stable graphs.
                self.model.forward(suffix[:1])
                torch.cuda.synchronize(self.model.device)
                self.reset()
        for token in suffix[:-1]:
            self.model.forward_hidden([token])
        return self.model.forward(suffix[-1:])

    def _prepare_glm_prompt(self, ids: list[int]) -> torch.Tensor:
        """Prepare GLM/Kimi prompts and expose exact-prefix reuse metrics.

        Kimi's KDA state is recurrent rather than a conventional prefix-cache
        object.  When the previous canonical token sequence is an exact
        prefix, retaining the live model state and evaluating only the suffix
        is the cache reuse operation.  Report that path through the same
        ``KVPrefillStats`` contract used by DSV4 without changing its math.
        """
        started = time.perf_counter()
        live = getattr(self, "_cache_ids", None)
        lcp = _token_lcp(live, ids)
        skip = self._kv_prefix_len(ids)
        if skip:
            mode = "exact-prefix"
            reason = (
                "live-kda-kv-prefix"
                if getattr(self, "arch", "glm") == "kimi_k3"
                else "live-prefix"
            )
        else:
            mode = "full-prefill"
            reason = (
                "no-live-prefix"
                if live
                else "empty-cache"
            )
            self.reset()
        logits = self._prefill_glm_suffix(ids, skip)
        stats = KVPrefillStats(
            mode=mode,
            reason=reason,
            prompt_tokens=len(ids),
            baseline_tokens=skip,
            lcp_tokens=lcp,
            replay_tokens=0,
            suffix_tokens=len(ids) - skip,
            processed_tokens=len(ids) - skip,
            prefill_ms=(time.perf_counter() - started) * 1000.0,
            snapshot_bytes=0,
        )
        self.last_kv_stats = stats
        if not getattr(self, "quiet", False):
            print(
                f"[KV] mode={stats.mode} reason={stats.reason} "
                f"baseline={stats.baseline_tokens} "
                f"lcp={stats.lcp_tokens} "
                f"suffix={stats.suffix_tokens} "
                f"prefill={stats.prefill_ms:.1f}ms",
                flush=True,
            )
        return logits

    @torch.no_grad()
    def _dsv4_prefill_suffix(self, ids: list[int], skip: int) -> tuple[torch.Tensor, torch.Tensor]:
        """DSV4 增量批量 prefill：复用 forward_verify 通道（快照不回滚，状态自然前进），
        64 token 分块（< sliding_window=128，避免环槽同批回绕）。
        返回 (末位 logits [vocab], 后缀全部位置 main_hidden [T, 3·hidden])。"""
        m = self.model
        mhs = []
        lg = None
        pos = skip
        for i in range(skip, len(ids), 64):
            chunk = ids[i:i + 64]
            lg2, mh2 = self._with_kv_capacity_retry(
                m.forward_verify, chunk, pos
            )
            m._spec = None            # 增量 prefill 不回滚，丢弃快照
            pos += len(chunk)
            mhs.append(mh2)
            lg = lg2
        m.pos = len(ids)
        return lg[-1], torch.cat(mhs, dim=0)

    @torch.no_grad()
    def _dsv4_prefill_range(
        self,
        ids: list[int],
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """Advance canonical DSV4 KV with the exact decode primitive.

        ``forward_verify`` is an experimental speculative batch path.  Its
        fused batch attention can diverge from sequential decode on the full
        model, so it must not build state later reused as canonical chat KV.
        """
        if not 0 < start < stop <= len(ids):
            raise ValueError(
                f"invalid DSV4 prefill range "
                f"{start}:{stop}/{len(ids)}"
            )
        model = self.model
        if model.pos != start:
            raise RuntimeError(
                f"DSV4 live position {model.pos} != range start {start}"
            )
        logits = None
        for position in range(start, stop):
            logits = self._with_kv_capacity_retry(
                model.forward,
                [ids[position]],
            )
            if model.pos != position + 1:
                raise RuntimeError(
                    f"DSV4 live position {model.pos} "
                    f"!= committed position {position + 1}"
                )
        assert logits is not None
        return logits

    def _save_dsv4_baseline(
        self,
        ids: list[int],
        baseline_len: int,
    ) -> int:
        snapshot = self.model.snapshot_kv()
        if snapshot.pos != baseline_len:
            raise RuntimeError(
                f"snapshot position {snapshot.pos} "
                f"!= baseline {baseline_len}"
            )
        self._kv_baseline = _DSV4Baseline(
            ids=list(ids[:baseline_len]),
            snapshot=snapshot,
        )
        return int(snapshot.nbytes)

    def _prefill_from_reset_to_boundary(
        self,
        ids: list[int],
        baseline_len: int,
    ) -> torch.Tensor:
        """Build canonical DSV4 state independently of request boundaries."""
        self.reset()
        logits = self._with_kv_capacity_retry(
            self.model.forward,
            [ids[0]],
        )
        if baseline_len > 1:
            logits = self._dsv4_prefill_range(
                ids,
                1,
                baseline_len,
            )
        return logits

    def _trace_kv_divergence(
        self,
        live: list[int] | None,
        reencoded: list[int],
        lcp: int,
        radius: int = 8,
    ) -> None:
        """Print the first token mismatch without touching model state."""
        if (
            not _env_enabled("TPQ_KV_TRACE")
            or not live
            or lcp >= len(live)
            or lcp >= len(reencoded)
        ):
            return

        start = max(0, lcp - radius)
        stop = lcp + radius + 1
        live_tokens = live[start:min(len(live), stop)]
        reencoded_tokens = reencoded[
            start:min(len(reencoded), stop)
        ]

        def token_piece(token_id: int) -> str | None:
            try:
                return self.tok.id_to_token(token_id)
            except Exception as error:
                return f"<id_to_token-error:{type(error).__name__}>"

        def decoded(
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
        ) -> str:
            try:
                return self.tok.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                )
            except Exception as error:
                return f"<decode-error:{type(error).__name__}>"

        false_text = {
            "live": decoded(
                live_tokens,
                skip_special_tokens=False,
            ),
            "reencoded": decoded(
                reencoded_tokens,
                skip_special_tokens=False,
            ),
        }
        true_text = {
            "live": decoded(
                live_tokens,
                skip_special_tokens=True,
            ),
            "reencoded": decoded(
                reencoded_tokens,
                skip_special_tokens=True,
            ),
        }
        lines = [
            (
                f"[KV-DIVERGE] pos={lcp} "
                f"window={start}:{stop} "
                f"live_len={len(live)} "
                f"reencoded_len={len(reencoded)}"
            ),
            f"live_id={live[lcp]}",
            f"reencoded_id={reencoded[lcp]}",
            f"live_tokens={json.dumps(live_tokens)}",
            f"reencoded_tokens={json.dumps(reencoded_tokens)}",
            (
                "live_piece="
                + json.dumps(
                    token_piece(live[lcp]),
                    ensure_ascii=False,
                )
            ),
            (
                "reencoded_piece="
                + json.dumps(
                    token_piece(reencoded[lcp]),
                    ensure_ascii=False,
                )
            ),
            (
                "skip_special_false_text="
                + json.dumps(false_text, ensure_ascii=False)
            ),
            (
                "skip_special_true_text="
                + json.dumps(true_text, ensure_ascii=False)
            ),
        ]
        print("\n".join(lines), flush=True)

    @torch.no_grad()
    def _prepare_dsv4_prompt(
        self,
        ids: list[int],
        baseline_len: int | None,
    ) -> torch.Tensor:
        """Prepare one prompt while retaining a pre-think rollback point."""
        started = time.perf_counter()
        cuda_events = None
        model_device = getattr(self.model, "device", None)
        if (
            model_device is not None
            and torch.device(model_device).type == "cuda"
        ):
            begin_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            begin_event.record()
            cuda_events = (begin_event, end_event)
        self._kv_prefill_events = None
        baseline_len = (
            len(ids) if baseline_len is None else baseline_len
        )
        if not 0 < baseline_len <= len(ids):
            raise ValueError(
                f"invalid DSV4 baseline_len={baseline_len} "
                f"for prompt length {len(ids)}"
            )

        live = getattr(self, "_cache_ids", None)
        baseline = getattr(self, "_kv_baseline", None)
        lcp = _token_lcp(live, ids)
        self._trace_kv_divergence(live, ids, lcp)
        strategy = "full"
        mode = "full-prefill"
        reason = "no-valid-baseline"
        baseline_tokens = 0
        replay_tokens = 0
        suffix_tokens = len(ids)
        processed_tokens = len(ids)
        start = 0

        if (
            live
            and lcp == len(live)
            and lcp <= baseline_len
            and lcp < len(ids)
            and self.model.pos == len(live)
        ):
            strategy = "live"
            mode = "lcp-replay"
            reason = "live-prefix"
            start = lcp
            baseline_tokens = lcp
            suffix_tokens = len(ids) - lcp
            processed_tokens = suffix_tokens
        elif (
            baseline is not None
            and getattr(baseline.snapshot, "pos", None)
            == len(baseline.ids)
            and lcp >= len(baseline.ids)
            and len(baseline.ids) <= baseline_len
            and lcp < len(ids)
        ):
            strategy = "rollback"
            mode = "lcp-replay"
            reason = "canonical-rollback"
            start = len(baseline.ids)
            baseline_tokens = start
            replay_tokens = lcp - start
            suffix_tokens = len(ids) - lcp
            processed_tokens = replay_tokens + suffix_tokens
        elif baseline is not None and lcp < len(baseline.ids):
            reason = "lcp-before-baseline"
        elif lcp == len(ids):
            reason = "no-new-suffix"

        def prepare_selected() -> tuple[torch.Tensor, int]:
            if strategy == "live":
                logits = (
                    self._dsv4_prefill_range(
                        ids, start, baseline_len
                    )
                    if start < baseline_len
                    else None
                )
            elif strategy == "rollback":
                self.model.restore_kv(baseline.snapshot)
                logits = (
                    self._dsv4_prefill_range(
                        ids, start, baseline_len
                    )
                    if start < baseline_len
                    else None
                )
            else:
                logits = self._prefill_from_reset_to_boundary(
                    ids,
                    baseline_len,
                )

            snapshot_bytes = self._save_dsv4_baseline(
                ids,
                baseline_len,
            )
            if baseline_len < len(ids):
                logits = self._dsv4_prefill_range(
                    ids,
                    baseline_len,
                    len(ids),
                )
            if logits is None:
                raise RuntimeError(
                    "DSV4 prompt preparation produced no logits"
                )
            return logits, snapshot_bytes

        try:
            logits, snapshot_bytes = prepare_selected()
        except Exception as preparation_error:
            original_strategy = strategy
            strategy = "full"
            try:
                logits, snapshot_bytes = prepare_selected()
            except Exception:
                raise preparation_error
            mode = "full-prefill"
            reason = (
                f"rollback-failed:"
                f"{type(preparation_error).__name__}"
                if original_strategy in ("live", "rollback")
                else
                f"full-prefill-retry:"
                f"{type(preparation_error).__name__}"
            )
            baseline_tokens = 0
            replay_tokens = 0
            suffix_tokens = len(ids)
            processed_tokens = len(ids)

        stats = KVPrefillStats(
            mode=mode,
            reason=reason,
            prompt_tokens=len(ids),
            baseline_tokens=baseline_tokens,
            lcp_tokens=lcp,
            replay_tokens=replay_tokens,
            suffix_tokens=suffix_tokens,
            processed_tokens=processed_tokens,
            prefill_ms=(
                time.perf_counter() - started
            ) * 1000.0,
            snapshot_bytes=snapshot_bytes,
        )
        if cuda_events is not None:
            cuda_events[1].record()
            self._kv_prefill_events = cuda_events
        self.last_kv_stats = stats
        if not getattr(self, "quiet", False):
            print(
                f"[KV] mode={stats.mode} reason={stats.reason} "
                f"baseline={stats.baseline_tokens} "
                f"lcp={stats.lcp_tokens} "
                f"replay={stats.replay_tokens} "
                f"suffix={stats.suffix_tokens} "
                f"prefill={stats.prefill_ms:.1f}ms",
                flush=True,
            )
        return logits

    def kv_prefill_cuda_ms(self) -> float | None:
        """Return completed CUDA-event preparation time without syncing."""
        events = getattr(self, "_kv_prefill_events", None)
        if events is None:
            return None
        begin_event, end_event = events
        if not end_event.query():
            return None
        return float(begin_event.elapsed_time(end_event))

    def _glm_device_greedy_window(
        self,
        *,
        temp: float,
        rep_penalty: float,
        no_repeat_ngram: int,
    ) -> int:
        """Return the safe GPU-token feedback window for greedy TP decode."""
        if (
            getattr(self, "arch", "glm") != "glm"
            or temp > 1e-6
            or rep_penalty != 1.0
            or no_repeat_ngram != 0
            or getattr(self.model, "device", torch.device("cpu")).type
            != "cuda"
            or getattr(self.model, "expert_parallel", None) is None
            or not getattr(
                getattr(self.model, "pool", None),
                "full_resident",
                False,
            )
        ):
            return 0
        try:
            window = max(
                0,
                int(os.environ.get("TPQ_GREEDY_DEVICE_WINDOW", "8")),
            )
        except ValueError:
            return 0
        if (
            window > 1
            and os.environ.get(
                "TPQ_FLASHINFER_MLA",
                "1",
            )
            != "0"
        ):
            from .fusedext import available as fused_available

            if (
                os.environ.get(
                    "TPQ_FLASHINFER_GPU_PLAN",
                    "1",
                )
                == "0"
                or not fused_available()
                or int(
                    getattr(
                        self.model,
                        "cfg",
                        {},
                    ).get("n_heads", 0)
                )
                != 64
            ):
                return 1
        return window

    def _generate_glm_device_greedy(
        self,
        *,
        ids: list[int],
        logits: torch.Tensor,
        out: list[int],
        max_new: int | None,
        max_ctx: int | None,
        window: int,
        callback,
        should_stop: Callable[[], bool] | None,
    ) -> list[int]:
        """Greedy decode with several GPU-resident token decisions per sync."""
        previous_static = os.environ.get("TPQ_STATIC_LM_OUTPUT")
        os.environ["TPQ_STATIC_LM_OUTPUT"] = "1"
        try:
            graph_target = max(
                0,
                int(
                    getattr(
                        self.model,
                        "cfg",
                        {},
                    ).get("n_layers", 4)
                )
                - 4,
            )
            graph_needs_capture = (
                os.environ.get(
                    "TPQ_ATTENTION_GRAPH",
                    "1",
                )
                != "0"
                and os.environ.get(
                    "TPQ_GLM_QB_SPLIT",
                    "1",
                )
                != "0"
                and not getattr(
                    self.model,
                    "_attention_graph_failed",
                    False,
                )
                and len(
                    getattr(
                        self.model,
                        "_attention_graphs",
                        {},
                    )
                )
                < graph_target
                and getattr(
                    self.model,
                    "_flashinfer_mla_state",
                    None,
                )
                is not None
            )
            if graph_needs_capture:
                # The first captured replay still shares temporary buffers
                # with graph construction.  Capture with one sacrificial
                # token, synchronize, then roll KV back and start generation
                # from the untouched prompt logits.  This is paid once per
                # model lifetime and keeps the first real device window exact.
                prompt_logits = logits.clone()
                capture_token = torch.argmax(
                    prompt_logits
                ).reshape(1)
                self.model.forward(capture_token)
                torch.cuda.synchronize(logits.device)
                # FlashInfer/Attention graph construction mutates fixed decode
                # workspaces beyond the single captured layer output.  A KV
                # truncation alone is insufficient; rebuild the prompt once
                # after capture while retaining the now-stable graph objects.
                self.reset()
                logits = self.model.forward(ids)

            while _generation_open(
                len(out),
                max_new,
                len(ids) + len(out),
                max_ctx,
            ):
                remaining = window
                attention_graph_warmup = (
                    os.environ.get(
                        "TPQ_ATTENTION_GRAPH",
                        "1",
                    )
                    != "0"
                    and os.environ.get(
                        "TPQ_GLM_QB_SPLIT",
                        "1",
                    )
                    != "0"
                    and not getattr(
                        self.model,
                        "_attention_graph_failed",
                        False,
                    )
                    and len(
                        getattr(
                            self.model,
                            "_attention_graphs",
                            {},
                        )
                    )
                    < graph_target
                )
                # Graph capture uses a side stream and must finish before the
                # next token reuses its fixed metadata/output buffers.  Only
                # the very first decode token is serialized; steady-state
                # generation immediately returns to the configured window.
                if attention_graph_warmup:
                    remaining = 1
                if max_new is not None:
                    remaining = min(remaining, max_new - len(out))
                if max_ctx is not None:
                    remaining = min(
                        remaining,
                        max_ctx - len(ids) - len(out),
                    )
                if remaining <= 0:
                    break

                base_position = self.model.pos
                device_tokens = torch.empty(
                    remaining,
                    dtype=torch.long,
                    device=logits.device,
                )
                for index in range(remaining):
                    torch.argmax(
                        logits,
                        out=device_tokens[index],
                    )
                    logits = self.model.forward(
                        device_tokens[index:index + 1]
                    )

                accepted = 0
                stop = False
                for next_token in device_tokens.cpu().tolist():
                    if next_token in self.eos:
                        stop = True
                        break
                    out.append(next_token)
                    accepted += 1
                    if callback:
                        callback(
                            next_token,
                            self.decode([next_token]),
                        )
                    if should_stop is not None and should_stop():
                        stop = True
                        break

                if accepted != remaining:
                    self.model.truncate_kv(
                        base_position + accepted
                    )
                if stop:
                    break
        finally:
            if previous_static is None:
                os.environ.pop("TPQ_STATIC_LM_OUTPUT", None)
            else:
                os.environ[
                    "TPQ_STATIC_LM_OUTPUT"
                ] = previous_static

        self._cache_ids = list(ids) + out
        self._cache_via_spec = False
        return out

    @torch.no_grad()
    def generate(self, ids: list[int], max_new: int | None = 128, temp: float = 0.0,
                 top_p: float = 1.0, callback=None, rep_penalty: float = 1.0,
                 no_repeat_ngram: int = 0,
                 should_stop: Callable[[], bool] | None = None,
                 kv_baseline_len: int | None = None) -> list[int]:
        """自回归生成。temp=0 贪心；callback(tok_id, 增量文本) 逐 token 回调。

        rep_penalty>1：对已出现 token 的 logits 施加重复惩罚（正除负乘），
        压制 PTQ 模型自由文本/长生成的复读循环（knee 档已知倾向）。
        no_repeat_ngram>0：禁止会复现已生成 n-gram 的候选 token。
        """
        out: list[int] = []
        mc = getattr(self.model, "max_ctx", None)
        if mc and len(ids) >= mc:
            print(f"[tpq] prompt 已达到 max_ctx={mc}，无法继续生成", flush=True)
            return out
        if max_new is not None and mc and len(ids) + max_new > mc:
            # 提前友好报错：越界会在 KV 压缩槽/rope 索引处抛 cryptic IndexError
            max_new = max(0, mc - len(ids))
            kv_hint = (
                "MLA latent KV 约 0.09MB/token"
                if getattr(self, "arch", "glm") == "glm"
                else "DSV4 使用环形窗+压缩槽"
            )
            print(f"[tpq] 警告：prompt {len(ids)} + max_new 超过 max_ctx={mc}，"
                  f"本次最多生成 {max_new} token"
                  f"（--max-ctx 可调大，{kv_hint}）",
                  flush=True)
            if max_new == 0:
                return out
        if getattr(self, "arch", "glm") == "dsv4":
            logits = self._prepare_dsv4_prompt(
                ids,
                kv_baseline_len,
            )
        else:
            logits = self._prepare_glm_prompt(ids)
        device_window = self._glm_device_greedy_window(
            temp=temp,
            rep_penalty=rep_penalty,
            no_repeat_ngram=no_repeat_ngram,
        )
        if device_window:
            return self._generate_glm_device_greedy(
                ids=ids,
                logits=logits,
                out=out,
                max_new=max_new,
                max_ctx=mc,
                window=device_window,
                callback=callback,
                should_stop=should_stop,
            )
        prev = list(ids)
        ngram_ban: dict[tuple, set] = {}
        while _generation_open(
            len(out), max_new, len(ids) + len(out), mc
        ):
            lg = logits
            if rep_penalty > 1.0 and prev:
                lg = logits.clone()
                seen = torch.tensor(sorted(set(prev)), device=lg.device)
                v = lg[seen]
                lg[seen] = torch.where(v > 0, v / rep_penalty, v * rep_penalty)
            if no_repeat_ngram > 0 and len(prev) >= no_repeat_ngram:
                if lg is logits:
                    lg = logits.clone()
                key = tuple(prev[-(no_repeat_ngram - 1):]) if no_repeat_ngram > 1 else ()
                for tok in ngram_ban.get(key, ()):  # 禁掉会复现 n-gram 的 token
                    lg[tok] = float("-inf")
            if temp <= 1e-6:
                nxt = int(lg.argmax().item())
            else:
                nxt = _sample_top_p(lg, temp, top_p)
            if nxt in self.eos:
                break
            out.append(nxt)
            if no_repeat_ngram > 0:
                seq = prev + [nxt]
                if len(seq) >= no_repeat_ngram:
                    k = tuple(seq[-no_repeat_ngram:-1]) if no_repeat_ngram > 1 else ()
                    ngram_ban.setdefault(k, set()).add(nxt)
            prev.append(nxt)
            if callback:
                callback(nxt, self.decode([nxt]))
            stop_requested = should_stop is not None and should_stop()
            if getattr(self, "arch", "glm") == "dsv4":
                logits = self._with_kv_capacity_retry(
                    self.model.forward, [nxt], committed=len(out)
                )
            else:
                logits = self.model.forward([nxt])
            if stop_requested:
                break
        self._cache_ids = list(ids) + out
        self._cache_via_spec = False   # 非投机路径不写 DSpark 环
        return out

    @torch.no_grad()
    def generate_speculative(
        self,
        ids: list[int],
        max_new: int | None = 128,
        k: int = 3,
        callback=None,
        should_stop: Callable[[], bool] | None = None,
        kv_baseline_len: int | None = None,
    ) -> list[int]:
        """MTP/DSpark 投机解码（贪心）。

        DSV4 默认严格回退 generate(temp=0)；仅显式设置
        TPQ_DSPARK_EXPERIMENTAL=1 才运行尚未数值等价的批量验证路径。
        GLM 仍走 MTP layer 78。
        """
        if getattr(self, "arch", "glm") == "kimi_k3":
            if (
                getattr(self.model, "device", torch.device("cpu")).type
                == "cpu"
                and hasattr(self.model, "forward_hidden_block_cpu")
                and hasattr(self.model, "snapshot_decode_state")
            ):
                return self._generate_kimi_prompt_lookup(
                    ids,
                    max_new=max_new,
                    k=k,
                    callback=callback,
                    should_stop=should_stop,
                )
            return self.generate(
                ids,
                max_new=max_new,
                temp=0.0,
                callback=callback,
                should_stop=should_stop,
            )
        if getattr(self, "arch", "glm") == "dsv4":
            if not _env_enabled("TPQ_DSPARK_EXPERIMENTAL"):
                self.spec_stats = {
                    "mode": "strict_fallback",
                    "rounds": 0,
                    "accepted": 0,
                    "drafted": 0,
                }
                if not getattr(
                    self, "_dspark_strict_notice_shown", False
                ):
                    print(
                        "[tpq] DSpark 严格模式：批量验证尚未与 "
                        "spec=0 数值等价，本次回退主模型贪心；"
                        "设置 TPQ_DSPARK_EXPERIMENTAL=1 "
                        "才启用实验路径",
                        flush=True,
                    )
                    self._dspark_strict_notice_shown = True
                generate_kwargs = {
                    "max_new": max_new,
                    "temp": 0.0,
                    "callback": callback,
                    "should_stop": should_stop,
                }
                if kv_baseline_len is not None:
                    generate_kwargs[
                        "kv_baseline_len"
                    ] = kv_baseline_len
                return self.generate(ids, **generate_kwargs)
            if not getattr(
                self, "_dspark_experimental_notice_shown", False
            ):
                print(
                    "[tpq] 警告：DSpark 实验模式不能保证与 "
                    "spec=0 token 等价",
                    flush=True,
                )
                self._dspark_experimental_notice_shown = True
            self._kv_baseline = None
            self.last_kv_stats = None
            mc = getattr(self.model, "max_ctx", None)
            if mc and len(ids) >= mc:
                print(f"[tpq] prompt 已达到 max_ctx={mc}，无法继续生成", flush=True)
                return []
            if max_new is not None and mc and len(ids) + max_new > mc:
                max_new = max(0, mc - len(ids))
                print(f"[tpq] 警告：超出 max_ctx={mc}，本次最多生成 {max_new} token",
                      flush=True)
            return self._generate_dspark(
                ids,
                max_new=max_new,
                k=k,
                callback=callback,
                should_stop=should_stop,
            )
        if getattr(self, "arch", "glm") != "glm":
            print("[tpq] 该架构投机解码未接入，回退贪心逐 token", flush=True)
            return self.generate(
                ids,
                max_new=max_new,
                temp=0.0,
                callback=callback,
                should_stop=should_stop,
            )
        from .mtp import MTPHead
        mc = getattr(self.model, "max_ctx", None)
        if mc and len(ids) >= mc:
            print(f"[tpq] prompt 已达到 max_ctx={mc}，无法继续生成", flush=True)
            return []
        if max_new is not None and mc and len(ids) + max_new > mc:
            max_new = max(0, mc - len(ids))
        self.reset()           # GLM-MTP 路径不支持增量 prefill，每轮全量重建
        mtp = MTPHead(self.model)
        mtp.reset()
        out: list[int] = []
        h_all = self.model.forward_hidden(ids)
        logits = self.model.logits_of(h_all[-1:]).squeeze(0)
        # MTP prefill 建立第 78 层上下文 KV；草稿首步用主模型 hidden（DeepSeek 流程），
        # 链式步才回喂 MTP 自身输出 h78
        mtp.prefill(h_all, ids)
        h_main_last = h_all[-1:]
        next_pos = len(ids)          # 下一个 MTP 步的 RoPE 位置
        next_t1 = int(logits.argmax())
        stats = {"rounds": 0, "accepted": 0, "drafted": 0}
        stop_requested = False
        while (
            _generation_open(len(out), max_new, len(ids) + len(out), mc)
            and next_t1 not in self.eos
            and not stop_requested
        ):
            t1 = next_t1
            out.append(t1)
            if callback:
                callback(t1, self.decode([t1]))
            stop_requested = should_stop is not None and should_stop()
            if stop_requested or not _generation_open(
                len(out), max_new, len(ids) + len(out), mc
            ):
                break
            # 1) 起草：首步输入 = (主模型 hidden, emb(t1))；其后回喂 h78
            kv0 = mtp.kv[0].shape[1] if mtp.kv is not None else 0
            h, drafts = h_main_last, []
            draft_count = k
            if max_new is not None:
                draft_count = min(draft_count, max_new - len(out))
            if mc is not None:
                draft_count = min(draft_count, mc - len(ids) - len(out))
            for j in range(max(0, draft_count)):
                h, lg = mtp.step(h, t1 if not drafts else drafts[-1], next_pos + j)
                drafts.append(int(lg.argmax()))
            stats["drafted"] += len(drafts)
            # 2) 主模型一次前向验证 [t1, d1..dk]
            pos0 = self.model.pos
            h2 = self.model.forward_hidden([t1] + drafts)
            lg2 = self.model.logits_of(h2)
            accepted = 0
            for i in range(len(drafts)):
                if not _generation_open(
                    len(out), max_new, len(ids) + len(out), mc
                ):
                    break
                if int(lg2[i].argmax()) == drafts[i] and drafts[i] not in self.eos:
                    accepted += 1
                    out.append(drafts[i])
                    if callback:
                        callback(drafts[i], self.decode([drafts[i]]))
                    if should_stop is not None and should_stop():
                        stop_requested = True
                        break
                else:
                    break
            stats["accepted"] += accepted
            stats["rounds"] += 1
            next_t1 = int(lg2[accepted].argmax())
            # 3) 主 KV 截断：被拒草稿不得留在上下文（只保留 t1 + 接受的前缀）
            keep = pos0 + 1 + accepted
            self.model.truncate_kv(keep)
            # 4) MTP 状态推进：KV 截断（保留 t1 + 接受前缀，t1 步恒有效）；
            #    下一轮首步的 hidden = 主模型在最后接受位的 hidden（h2[accepted]）
            L = kv0 + 1 + accepted
            mtp.kv = (mtp.kv[0][:, :L], mtp.kv[1][:, :L])
            h_main_last = h2[accepted:accepted + 1]
            next_pos += 1 + accepted
        self.spec_stats = stats
        return out

    @staticmethod
    def _prompt_lookup_draft(
        history: list[int],
        maximum: int,
        *,
        minimum_ngram: int = 3,
        maximum_ngram: int = 16,
    ) -> list[int]:
        """Copy a continuation after the longest previous suffix match."""
        if maximum <= 0 or len(history) <= minimum_ngram:
            return []
        upper = min(maximum_ngram, len(history) - 1)
        for width in range(upper, minimum_ngram - 1, -1):
            suffix = history[-width:]
            latest = len(history) - width - 1
            for start in range(latest, -1, -1):
                if history[start:start + width] != suffix:
                    continue
                draft = history[start + width:start + width + maximum]
                if draft:
                    return list(draft)
        return []

    @torch.no_grad()
    def _generate_kimi_prompt_lookup(
        self,
        ids: list[int],
        *,
        max_new: int | None,
        k: int,
        callback=None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[int]:
        """Lossless prompt-lookup decode using Kimi's CPU block verifier."""
        maximum_draft = max(1, min(int(k), 15))
        max_ctx = getattr(self.model, "max_ctx", None)
        if max_ctx and len(ids) >= max_ctx:
            return []
        if max_new is not None and max_ctx:
            max_new = min(max_new, max(0, max_ctx - len(ids)))
        logits = self._prepare_glm_prompt(ids)
        history = list(ids)
        out: list[int] = []
        stats = {
            "mode": "kimi_prompt_lookup_block",
            "rounds": 0,
            "block_rounds": 0,
            "fallback_rounds": 0,
            "drafted": 0,
            "accepted": 0,
            "replayed": 0,
        }
        stop = False
        while (
            not stop
            and _generation_open(
                len(out), max_new, len(ids) + len(out), max_ctx
            )
        ):
            first = int(logits.argmax().item())
            if first in self.eos:
                break
            out.append(first)
            history.append(first)
            if callback:
                callback(first, self.decode([first]))
            stop = should_stop is not None and should_stop()
            stats["rounds"] += 1
            room = maximum_draft
            if max_new is not None:
                room = min(room, max_new - len(out))
            if max_ctx is not None:
                room = min(room, max_ctx - len(ids) - len(out))
            if stop or room <= 0:
                logits = self.model.forward([first])
                stats["fallback_rounds"] += 1
                break
            drafts = self._prompt_lookup_draft(history, room)
            if not drafts:
                logits = self.model.forward([first])
                stats["fallback_rounds"] += 1
                continue
            snapshot = self.model.snapshot_decode_state()
            hidden = self.model.forward_hidden_block_cpu([first] + drafts)
            block_logits = self.model.logits_of(hidden)
            stats["block_rounds"] += 1
            stats["drafted"] += len(drafts)
            accepted = 0
            for index, draft in enumerate(drafts):
                if int(block_logits[index].argmax().item()) != draft:
                    break
                if draft in self.eos:
                    stop = True
                    break
                out.append(draft)
                history.append(draft)
                accepted += 1
                if callback:
                    callback(draft, self.decode([draft]))
                if should_stop is not None and should_stop():
                    stop = True
                    break
            stats["accepted"] += accepted
            fully_committed = accepted == len(drafts) and not stop
            if fully_committed:
                logits = block_logits[accepted]
                continue
            self.model.restore_decode_state(snapshot)
            committed = [first] + drafts[:accepted]
            canonical_hidden = self.model.forward_hidden(committed)
            logits = self.model.logits_of(canonical_hidden[-1:]).squeeze(0)
            stats["replayed"] += len(committed)
        self.spec_stats = stats
        self._cache_ids = list(ids) + out
        self._cache_via_spec = False
        return out

    @torch.no_grad()
    def _generate_dspark(
        self,
        ids: list[int],
        max_new: int | None = 128,
        k: int = 5,
        callback=None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[int]:
        """DSV4 DSpark 块并行投机解码（贪心验收，输出与 generate(temp=0) 一致）。

        每轮：1 次 DSpark 前向并行产出 block_size(=5) 个草稿 → 主模型 1 次批量
        前向验证 [t1, d1..dk] → 接受最长连续前缀（argmax 比对），首位不匹配的
        argmax 为奖励 token → 主 KV 按接受前缀截断（spec_commit），DSpark 环
        写入接受位置的 main_kv。
        """
        from .dspark import DSparkHead
        model = self.model
        dsp = getattr(self, "_dsp", None)
        if dsp is None:
            dspark_gb = float(os.environ.get("TPQ_DSPARK_VRAM_GB", "2.75"))
            self._cap_expert_cache(
                self._vram_runtime_reserve_gb + dspark_gb,
                "运行时+DSpark 余量",
            )
            dsp = self._dsp = DSparkHead(model)
        k = min(k, dsp.block_size)
        out: list[int] = []
        skip = self._kv_prefix_len(ids) if self._cache_via_spec else 0
        if skip:
            # 多轮 KV 复用：主模型只增量 prefill 新后缀，DSpark 环补写新位置的 main_kv
            lg_last, mh_suf = self._dsv4_prefill_suffix(ids, skip)
            dsp.update_kv(mh_suf, skip)
            t1 = int(lg_last.argmax())
            mh_last = mh_suf[-1]                    # 最末位置 main_hidden [3D]
        else:
            self.reset()
            dsp.reset()
            logits_last, mh = self._with_kv_capacity_retry(
                model.prefill_mh,
                torch.tensor([ids], device=model.device),
            )
            dsp.prefill_kv(mh[0])                # DSpark 环：positions 0..T-1
            t1 = int(logits_last[0].argmax())
            mh_last = mh[0, -1]                  # position p 的 main_hidden [3D]
        p = len(ids) - 1                         # 最末已处理位置
        stats = {"rounds": 0, "accepted": 0, "drafted": 0}
        mc = getattr(model, "max_ctx", None)
        stop_requested = False
        while (
            _generation_open(len(out), max_new, len(ids) + len(out), mc)
            and t1 not in self.eos
            and not stop_requested
        ):
            drafts = dsp.draft(t1, mh_last, p)   # 5 草稿；main_kv@p 写入各层环
            out.append(t1)
            if callback:
                callback(t1, self.decode([t1]))
            stop_requested = should_stop is not None and should_stop()
            draft_count = k
            if max_new is not None:
                draft_count = min(draft_count, max_new - len(out))
            if mc is not None:
                draft_count = min(draft_count, mc - len(ids) - len(out))
            if stop_requested:
                draft_count = 0
            block = [t1] + drafts[:max(0, draft_count)]
            pos0 = model.pos                      # = p+1
            lg2, mh2 = self._with_kv_capacity_retry(
                model.forward_verify,
                block,
                pos0,
                committed=len(out),
            )
            accepted = 0
            for i in range(max(0, draft_count)):
                if not _generation_open(
                    len(out), max_new, len(ids) + len(out), mc
                ):
                    break
                if int(lg2[i].argmax()) == drafts[i] and drafts[i] not in self.eos:
                    accepted += 1
                    out.append(drafts[i])
                    if callback:
                        callback(drafts[i], self.decode([drafts[i]]))
                    if should_stop is not None and should_stop():
                        stop_requested = True
                        break
                else:
                    break
            stats["accepted"] += accepted
            stats["drafted"] += k
            stats["rounds"] += 1
            next_t1 = int(lg2[accepted].argmax())
            keep = pos0 + 1 + accepted
            model.spec_commit(keep)               # 主 KV 按接受前缀截断
            dsp.update_kv(mh2[:accepted], pos0)   # 接受前缀入 DSpark 环（末位下轮 draft 写）
            mh_last = mh2[accepted]
            p = keep - 1
            t1 = next_t1
        self.spec_stats = stats
        self._cache_ids = list(ids) + out
        self._cache_via_spec = True    # DSpark 环已覆盖 prompt+回复全部位置
        return out


def _sample_top_p(logits: torch.Tensor, temp: float, top_p: float) -> int:
    """top-p（核）采样。"""
    probs = torch.softmax(logits.float() / max(temp, 1e-6), dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, 0)
    keep = (cum - sp) < top_p
    cand = si[keep]
    cp = sp[keep] / sp[keep].sum()
    return int(cand[torch.multinomial(cp, 1)].item())
