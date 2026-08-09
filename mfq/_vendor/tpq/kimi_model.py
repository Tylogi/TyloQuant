"""Kimi K3 text inference runtime for TPQ expert archives.

The first production path is one CUDA device with source-native BF16 dense
weights resident on GPU and routed experts supplied by the existing TPQ
RAM/VRAM cache.  It deliberately shares ``TPQStore`` and ``ExpertPool`` with
the established runtimes; model files remain read-only.
"""

from __future__ import annotations

import math
import os
import time
import gc
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .cconfig import KimiK3Config
from .grouped import activate_gate_up, moe_mlp_grouped_mixed
from .kernels import BlockFP8Weight, ProjectionGroup
from .kimi_ops import (
    attention_residual,
    gated_rmsnorm,
    kda_recurrent_step,
    rmsnorm,
    route_experts,
    short_conv_step,
)
from .precision import compute_dtype
from .store import TPQStore, ExpertPool, PackedCpuExpertPool


_ROOT = "language_model"


def _linear(value: torch.Tensor, weight) -> torch.Tensor:
    """Preserve the direct path's dtype through the public compact Linear."""
    from .ops import linear

    return linear(value, weight, output_dtype=value.dtype)


class KimiK3TPQModel:
    """Decode-correct Kimi K3 runtime with latent MLA and recurrent KDA."""

    def __init__(
        self,
        root: str,
        cache_gb: float = 16.0,
        max_ctx: int = 2048,
        device: str = "cpu",
        vram_cache_gb: float = 4.0,
        tp_size: int = 1,
        extreme_fixed_gpu_bytes: int = 0,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._cpu_threads = 0
        self._cpu_numa_interleaved = False
        self._cpu_block_major_weights = 0
        self._cpu_block_major_bytes = 0
        if self.device.type == "cpu":
            from .cpuext import (
                configure_cpu_threads,
                configure_numa_interleave,
            )

            self._cpu_threads = configure_cpu_threads()
            self._cpu_numa_interleaved = configure_numa_interleave()
        self.store = TPQStore(root)
        self.cfg = self.store.cfg
        self.config = KimiK3Config.from_json(self.cfg)
        from .ops import ModelOperatorConfig

        self.operator_config = ModelOperatorConfig.from_manifest(
            {
                "model_family": (
                    self.store.man.model_family or "kimi_k3"
                ),
                "config": self.cfg,
            }
        )
        self.max_ctx = int(max_ctx)
        self.tp_size = int(tp_size)
        if self.tp_size <= 0:
            raise ValueError("Kimi tp_size must be positive")
        if self.tp_size > 1:
            if self.device.type != "cuda":
                raise ValueError("Kimi multi-GPU pipeline requires CUDA")
            primary = (
                torch.cuda.current_device()
                if self.device.index is None
                else self.device.index
            )
            self.devices = tuple(
                torch.device("cuda", primary + rank)
                for rank in range(self.tp_size)
            )
            if self.devices[-1].index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Kimi tp={self.tp_size} requires cuda:{primary}.."
                    f"cuda:{self.devices[-1].index}"
                )
            from .kimi_experts import (
                PackedExpertPool,
                build_kimi_layer_plan,
            )

            self._pipeline_plan = build_kimi_layer_plan(
                self.store,
                self.tp_size,
            )
            tp_ram_offload = (
                os.environ.get(
                    "TPQ_KIMI_TP_PACKED_HYBRID",
                    "0",
                )
                != "0"
            )
            if tp_ram_offload:
                from .kimi_tp_hybrid import (
                    PackedTensorHybridPool,
                )

                self.pool = PackedTensorHybridPool(
                    self.store,
                    self.devices,
                    self._pipeline_plan,
                    vram_cache_gb,
                    ram_gb=cache_gb,
                )
            else:
                self.pool = PackedExpertPool(
                    self.store,
                    self.devices,
                    self._pipeline_plan,
                    parallelism=os.environ.get(
                        "TPQ_MOE_PARALLELISM",
                        "tensor",
                    ),
                    tensor_group_size=int(
                        os.environ.get("TPQ_MOE_TP_GROUP", "2")
                    ),
                )
        else:
            self.devices = (self.device,)
            self._pipeline_plan = None
            packed_cpu = (
                self.device.type == "cpu"
                and os.environ.get("TPQ_CPU_PACKED", "1") != "0"
            )
            packed_hybrid = (
                self.device.type == "cuda"
                and os.environ.get("TPQ_KIMI_PACKED_HYBRID", "1") != "0"
            )
            if packed_cpu:
                self.pool = PackedCpuExpertPool(
                    self.store,
                    cache_gb,
                )
            elif packed_hybrid:
                from .packed_hybrid import PackedHybridPool

                self.pool = PackedHybridPool(
                    self.store,
                    vram_cache_gb,
                    device=self.device,
                    ram_gb=cache_gb,
                    startup_gpu_reserve_bytes=extreme_fixed_gpu_bytes,
                )
            else:
                self.pool = ExpertPool(
                    self.store,
                    vram_cache_gb if self.device.type != "cpu" else cache_gb,
                    device=device,
                    ram_gb=cache_gb if self.device.type != "cpu" else 0.0,
                )
            if (
                self.device.type == "cuda"
                and os.environ.get(
                    "TPQ_SINGLE_GPU_LAYER_GRAPH",
                    "0",
                )
                != "0"
            ):
                from .kimi_experts import build_kimi_layer_plan

                # Reuse the public fixed-address TP graph builders at width
                # one.  The packed RAM pool remains unchanged; this plan only
                # describes dense/attention graph ownership and never turns
                # the single-GPU path into an owner/worker execution system.
                self._pipeline_plan = build_kimi_layer_plan(
                    self.store,
                    1,
                )
        self._weights: dict[str, object] = {}
        self._consumed_dense_names: set[str] = set()
        self._kda_input_proj: dict[int, object] = {}
        self._kda_gate_rank: dict[int, int] = {}
        self._mla_input_proj: dict[int, object] = {}
        self._dense_gate_up: dict[int, object] = {}
        self._shared_gate_up: dict[int, object] = {}
        self._moe_input_proj: dict[int, object] = {}
        self._cpu_latent_moe_layers: dict[int, object] = {}
        self._tp_dense_mlp = None
        self._tp_shared_mlp = None
        self._tp_kda = None
        self._tp_mla = None
        self._tp_routed_down = None
        self._tp_routed_up = None
        self._tp_router = None
        self._tp_route_down = None
        self._tp_moe_prelude = None
        self._tp_vocab = None
        self._tp_hidden_state_ready = False
        self._tp_token_hidden = None
        self._tp_block_residual = None
        self._tp_layer_prefix_hidden: dict[int, object] = {}
        self._tp_layer_output_hidden: dict[int, object] = {}
        self._tp_attention_mix: dict[int, tuple] = {}
        self._tp_mlp_mix: dict[int, tuple] = {}
        self._tp_final_mix: tuple | None = None
        self._tp_residual_workspaces: tuple[torch.Tensor, ...] = ()
        self._tp_moe_residual_hidden: dict[int, object] = {}
        self._tp_fixed_moe_prelude = None
        self._tp_attention_layer_graph = False
        self._tp_mlp_layer_graph = False
        self._tp_moe_owner_layer_graph = False
        self._tp_moe_all_rank_layer_graph = False
        self._tp_route_packed_graph = False
        self._tp_route_packed_plan = None
        self._tp_routed_finalize_graph = False
        self._tp_no_owner_moe_plans: dict[int, object] = {}
        self._tp_decode_layer_plans: dict[int, object] = {}
        self._tp_routed_latent_buffers: dict[int, torch.Tensor] = {}
        self._tp_routed_value_buffers: dict[int, torch.Tensor] = {}
        self._tp_routed_norm_buffers: dict[int, torch.Tensor] = {}
        self._tp_route_all_rank_buffers: dict[int, tuple] = {}
        self._tp_route_corrections: dict[int, tuple[torch.Tensor, ...]] = {}
        self._tp_route_masks: dict[int, tuple[torch.Tensor, ...]] = {}
        self._tp_routed_norm_hidden: dict[int, object] = {}
        self._tp_routed_norm_weights: dict[
            int, tuple[torch.Tensor, ...]
        ] = {}
        self._masks: dict[int, torch.Tensor] = {}
        self._route_buffers: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._absorbed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._kda_state: dict[int, torch.Tensor] = {}
        self._conv_state: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._mla_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._paged_latent_runners: dict[int, object] = {}
        self._paged_latent_prepared: dict[int, int] = {}
        self._paged_latent_unavailable: set[int] = set()
        self._kda_workspace: dict[int, torch.Tensor] = {}
        self._kda_output: dict[int, torch.Tensor] = {}
        self._residual_buffers: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._kda_fused_failed = False
        self._prev_ids: dict[int, list[int]] = {}
        self.last_layer_profile: list[dict[str, float | int]] = []
        self._cpu_substage_profile: dict[str, float] = {}
        self.last_cuda_profile: dict[str, object] = {}
        self._profile_enabled = False
        self._profile_pool_snapshot: dict[str, float | int] = {}
        self.tp_dataflow: dict[str, object] = {}
        self._tp_no_owner_moe_timing: dict[str, float] = {}
        self._tp_async_profile_active = False
        self._active_cuda_events: list[tuple] | None = None
        self.pos = 0
        self.effective_tp_size = self.tp_size

    # ---- weights / startup -------------------------------------------------
    def layer_device(self, layer: int) -> torch.device:
        if self._pipeline_plan is None:
            return self.device
        return self.devices[self._pipeline_plan.owner_by_layer[layer]]

    def _weight_device(self, name: str) -> torch.device:
        marker = ".model.layers."
        if marker in name:
            layer = int(name.split(marker, 1)[1].split(".", 1)[0])
            return self.layer_device(layer)
        if ".model.embed_tokens." in name:
            return self.devices[0]
        return self.devices[-1]

    def w(self, name: str):
        cached = self._weights.get(name)
        if cached is not None:
            return cached
        value = self.store.get_dense(name)
        if not isinstance(value, (torch.Tensor, BlockFP8Weight)):
            raise TypeError(
                f"Kimi dense tensor {name!r} has unsupported storage"
            )
        if (
            isinstance(value, BlockFP8Weight)
            and (
                "layernorm.weight" in name
                or name.endswith(".norm.weight")
                or name.endswith(".o_norm.weight")
            )
        ):
            # Norm vectors are tiny and consumed by fixed BF16 RMSNorm
            # kernels. This explicit materialization never applies to a
            # matrix projection.
            value = value.to(torch.bfloat16)
        if name.endswith(
            ".block_sparse_moe.gate.e_score_correction_bias"
        ):
            # The small correction vector is always evaluated in FP32.
            if isinstance(value, BlockFP8Weight):
                value = value.to(torch.float32)
            else:
                value = value.float()
        elif name.endswith(".block_sparse_moe.gate.weight"):
            # Kimi's published Router evaluates this source in FP32.
            value = (
                value.to(torch.float32)
                if isinstance(value, BlockFP8Weight)
                else value.float()
            )
        target = self._weight_device(name)
        if target.type == "cpu" and isinstance(value, BlockFP8Weight):
            if (
                os.environ.get("TPQ_CPU_BLOCK_MAJOR", "auto") != "0"
                and not name.endswith(".kv_b_proj.weight")
            ):
                converted = value.optimize_cpu_layout()
                if converted is not value:
                    value = converted
                    self._cpu_block_major_weights += 1
                    self._cpu_block_major_bytes += value.nbytes
        elif target.type != "cpu":
            value = value.to(target)
        self._weights[name] = value
        return value

    def _take_weight(self, name: str):
        """Load one source weight lazily and transfer ownership to an op."""
        value = self.w(name)
        self._weights.pop(name, None)
        self._consumed_dense_names.add(name)
        return value

    def _release_cuda_startup_cache(self, stage: str) -> None:
        """Release dead owner staging blocks between streamed TP stages."""
        if self._pipeline_plan is None:
            return
        gc.collect()
        released = []
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_reserved(device)
                torch.cuda.empty_cache()
                after = torch.cuda.memory_reserved(device)
                released.append(max(0, before - after))
        if any(released):
            print(
                f"[tpq-kimi] {stage} 后释放启动暂存："
                + "，".join(
                    f"cuda:{device.index}={value / 2**30:.2f}GiB"
                    for device, value in zip(self.devices, released)
                ),
                flush=True,
            )

    @staticmethod
    def _language_weight(name: str) -> bool:
        return name.startswith(f"{_ROOT}.")

    def preload(self) -> None:
        if self.device.type == "cpu":
            started = time.time()
            print(
                f"[tpq-kimi] CPU 推理线程：{self._cpu_threads}",
                flush=True,
            )
            if self._cpu_numa_interleaved:
                print(
                    "[tpq-kimi] CPU NUMA：后续权重与专家内存跨节点交错分配",
                    flush=True,
                )
            # 所有 CPU 模型统一使用同一组懒编译原生内核。启动阶段预编译，
            # 避免首个 token 混入编译耗时。
            from .cpuext import prebuild as prebuild_cpu

            prebuild_cpu()
            resident_all = self.pool.preload_all()
            if not resident_all:
                self.pool.preload_pinned()
            names = [
                name
                for name in self.store.dense_names()
                if self._language_weight(name)
            ]
            for index, name in enumerate(names, 1):
                self.w(name)
                if index % 160 == 0:
                    print(
                        f"[tpq-kimi] CPU 预载 dense {index}/{len(names)}",
                        flush=True,
                    )
            self._combine_dense_projections()
            self._prepare_cpu_latent_moe()
            print(
                "[tpq-kimi] CPU 公共 ProjectionGroup 完成："
                f"KDA={len(self._kda_input_proj)}，"
                f"MLA={len(self._mla_input_proj)}，"
                f"MoE={len(self._moe_input_proj)}；"
                f"总加载 {time.time() - started:.1f}s",
                flush=True,
            )
            if self._cpu_block_major_weights:
                print(
                    "[tpq-kimi] CPU 公共 block-major32 布局完成："
                    f"{self._cpu_block_major_weights} 个投影 / "
                    f"{self._cpu_block_major_bytes / 2**30:.2f} GiB；"
                    "紧凑FP8字节未展开",
                    flush=True,
                )
                decisions = BlockFP8Weight.cpu_layout_decisions()
                selected = sum(
                    bool(item["block_major32"])
                    for item in decisions.values()
                )
                print(
                    "[tpq-kimi] CPU block-FP8 自动选型："
                    f"{selected}/{len(decisions)} 种矩阵形状采用 "
                    "block-major32，其余保留 row-major",
                    flush=True,
                )
            return
        started = time.time()
        extreme_resident_all: bool | None = None
        if (
            self._pipeline_plan is None
            and getattr(self.pool, "startup_gpu_reserve_bytes", 0) > 0
        ):
            extreme_resident_all = self.pool.preload_all()
            if not extreme_resident_all:
                raise RuntimeError(
                    "极限模式要求全部紧凑专家固定驻留 RAM/VRAM"
                )
            self.pool.release_startup_gpu_reservation()
        if self._pipeline_plan is not None:
            allocate = getattr(self.pool, "allocate", None)
            if callable(allocate):
                allocate()
            if len(self.devices) == 1:
                placement = (
                    "单rank固定地址 Graph；Dense 流式落入 GPU，"
                    "packed 专家继续 RAM+VRAM"
                )
            else:
                placement = (
                    "按层流水线"
                    if self.pool.parallelism == "pipeline"
                    else (
                        "完整Dense权重仅按层暂存；运行权重已按维度分片到"
                        "全rank，packed专家同样跨全rank分片"
                    )
                )
            print(
                f"[tpq-kimi] {placement}："
                + "，".join(
                    f"cuda:{self.devices[rank].index}=L{start}-L{end - 1}"
                    f"/{self._pipeline_plan.bytes_by_rank[rank] / 2**30:.2f}GiB"
                    for rank, (start, end) in enumerate(
                        self._pipeline_plan.ranges
                    )
                ),
                flush=True,
            )
        names = [
            name for name in self.store.dense_names()
            if self._language_weight(name)
        ]
        if self._pipeline_plan is None:
            for index, name in enumerate(names, 1):
                self.w(name)
                if index % 160 == 0:
                    print(
                        f"[tpq-kimi] 预载 dense {index}/{len(names)}",
                        flush=True,
                    )
        else:
            print(
                "[tpq-kimi] Dense 流式建图：紧凑源权重读取后立即 TP 分片，"
                "不保留完整 BF16/FP8 owner 副本",
                flush=True,
            )
        if self._pipeline_plan is None:
            self._combine_dense_projections()
        self._prepare_tp_kda()
        self._release_cuda_startup_cache("KDA")
        self._prepare_tp_mla()
        self._release_cuda_startup_cache("MLA")
        self._prepare_tp_dense_mlp()
        self._release_cuda_startup_cache("Dense MLP")
        self._prepare_tp_moe_prelude()
        self._prepare_tp_shared_mlp()
        self._release_cuda_startup_cache("共享 MLP")
        self._prepare_tp_route_down()
        self._prepare_tp_routed_linear()
        self._prepare_tp_router()
        self._release_cuda_startup_cache("Router/MoE 投影")
        self._prepare_tp_hidden_state()
        self._assert_no_owner_tp_dataflow()
        self._prepare_tp_vocab()
        if self._pipeline_plan is not None:
            remaining = [
                name
                for name in names
                if (
                    name not in self._consumed_dense_names
                    and name not in self._weights
                )
            ]
            for index, name in enumerate(remaining, 1):
                self.w(name)
                if index % 160 == 0:
                    print(
                        f"[tpq-kimi] 预载剩余 dense "
                        f"{index}/{len(remaining)}",
                        flush=True,
                    )
        if self._tp_mla is None:
            for layer in self.config.full_attn_layers:
                self._absorbed_weights(layer)
            self._mla_buffers(layer, 1)
        if self._tp_kda is None:
            for layer in self.config.kda_layers:
                self._kda_buffers(layer)
        for layer in range(
            self.config.first_dense_layers,
            self.config.n_layers,
        ):
            self._mask(layer)
        allocated = [
            torch.cuda.memory_allocated(device) / 2**30
            for device in self.devices
        ]
        print(
            f"[tpq-kimi] 文本 dense 预载完成"
            f"（{time.time() - started:.1f}s，"
            + "，".join(
                f"cuda:{device.index}={value:.1f}GiB"
                for device, value in zip(self.devices, allocated)
            )
            + "）",
            flush=True,
        )
        if extreme_resident_all is not None:
            self.pool.verify_startup_gpu_reservation()
        if self._pipeline_plan is None:
            # Projection fusion temporarily holds both source and concatenated
            # BF16 tensors.  Their allocations are dead here, but PyTorch can
            # retain the released blocks in its CUDA cache.  The packed arena
            # deliberately uses driver-visible free memory as a hard safety
            # bound, so stale cached blocks would shrink a 26 GiB arena to only
            # a few GiB.  This is a startup-only synchronization and does not
            # enter the decode path.
            reserved_before = torch.cuda.memory_reserved(self.device)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            reserved_after = torch.cuda.memory_reserved(self.device)
            released = max(0, reserved_before - reserved_after)
            if released:
                print(
                    "[tpq-kimi] 释放预载临时 CUDA 缓存："
                    f"{released / 2**30:.1f}GiB",
                    flush=True,
                )
        if self._pipeline_plan is not None and hasattr(self.pool, "preload"):
            self.pool.preload()
            if (
                getattr(self.pool, "hidden_mode", False)
                and hasattr(self.pool, "compose_route_topk")
            ):
                from .ops import (
                    RoutePackedPlanSpec,
                    TensorParallelRoutePackedPlan,
                )

                route_layers = tuple(range(
                    self.config.first_dense_layers,
                    self.config.n_layers,
                ))
                self._tp_route_packed_plan = TensorParallelRoutePackedPlan(
                    self.devices,
                    RoutePackedPlanSpec(
                        scoring_func=self.config.scoring_func,
                        top_k=self.config.top_k,
                        normalize=self.config.norm_topk_prob,
                        scaling=self.config.routed_scaling,
                        n_group=self.config.n_group,
                        topk_group=self.config.topk_group,
                    ),
                    self.pool,
                    {
                        layer: self._tp_route_down.output_hidden(
                            layer
                        )[0]
                        for layer in route_layers
                    },
                    self._tp_route_corrections,
                    self._tp_route_masks,
                    self._tp_route_all_rank_buffers,
                    layers=route_layers,
                )
                self._tp_route_packed_graph = True
                self.tp_dataflow[
                    "route_packed_schedule"
                ] = "all_rank_topk_to_packed_parent_graph"
                if (
                    self.config.latent_moe_use_norm
                    and self._tp_routed_up is not None
                    and os.environ.get("TPQ_TP_LAYER_GRAPH", "0")
                    != "0"
                ):
                    for layer in range(
                        self.config.first_dense_layers,
                        self.config.n_layers,
                    ):
                        self._tp_routed_up.compose_normalize_prelude(
                            layer,
                            self.pool.output_hidden(layer),
                            self._tp_routed_norm_weights[layer],
                            self.config.rms_eps,
                        )
                    self._tp_routed_finalize_graph = True
                    self.tp_dataflow[
                        "routed_finalize_schedule"
                    ] = (
                        "all_rank_packed_to_rmsnorm_to_row_tp_parent_graph"
                    )
                    print(
                        "[tpq-kimi] 通用 packed输出→RMSNorm→"
                        "routed Up 全rank父图完成："
                        f"{self.config.n_layers - self.config.first_dense_layers}"
                        " 层",
                        flush=True,
                    )
                self._prepare_no_owner_moe_plans()
            return
        resident_all = (
            extreme_resident_all
            if extreme_resident_all is not None
            else self.pool.preload_all()
        )
        if resident_all:
            # Reserve the fixed VRAM arena before cudaHostRegister.  Very large
            # host mappings consume CUDA driver address-space resources and can
            # otherwise make the later arena allocation fail even though the
            # packed archive itself never enters VRAM.
            self.pool.build_gpu_arenas()
            self.pool.pin_host_resident()
        else:
            self.pool.preload_pinned()
            self.pool.build_gpu_arenas()

    def _assert_no_owner_tp_dataflow(self) -> None:
        """Reject transitional owner compute when formal no-owner TP is on."""
        if (
            self._pipeline_plan is None
            or len(self.devices) == 1
            or os.environ.get("TPQ_TP_HIDDEN_STATE", "0") == "0"
            or os.environ.get("TPQ_TP_NO_OWNER", "1") == "0"
        ):
            return
        executors = tuple(
            executor
            for executor in (
                self._tp_kda,
                self._tp_mla,
                self._tp_dense_mlp,
                self._tp_shared_mlp,
                self._tp_route_down,
                self._tp_routed_up,
            )
            if executor is not None
        )
        routed_up_bound = (
            self._tp_routed_up is not None
            and all(
                state.bound_input_addresses is not None
                for state in self._tp_routed_up.layers.values()
            )
        )
        if (
            not getattr(self.pool, "hidden_mode", False)
            or getattr(self.pool, "parallelism", None) != "tensor"
            or self._tp_route_down is None
            or self._tp_fixed_moe_prelude is not None
            or self._tp_moe_owner_layer_graph
            or not routed_up_bound
            or any(
                self._tp_executor_width(executor) != len(self.devices)
                for executor in executors
            )
            or any(
                getattr(
                    getattr(executor, "spec", None),
                    "capture_owner_dispatch",
                    False,
                )
                for executor in executors
            )
        ):
            raise RuntimeError(
                "TPQ_TP_NO_OWNER forbids owner/subgroup compute in the "
                "formal TP data flow"
            )
        self.tp_dataflow = {
            "hidden_layout": "all_rank_replicated",
            "hidden_owner": False,
            "route_owner": False,
            "attention_layout": "column_head_tp_to_row_tp",
            "dense_shared_layout": "column_tp_to_row_tp",
            "router_layout": "expert_row_column_tp_small_logits_reduce",
            "shared_router_schedule": (
                "rank_parallel_parent_graph"
                if self._tp_moe_all_rank_layer_graph
                else "independent_all_rank_graphs"
            ),
            "routed_up_input_layout": "bound_local_view_zero_copy",
            "packed_expert_parallelism": "tensor",
            "packed_expert_ranks": len(self.devices),
            "packed_expert_weight_layout": "all_rank_tensor_sharded",
            "dense_staging_layout": "layer_local_startup_only",
            "dense_runtime_layout": "all_layer_all_rank_tp_sharded",
            "owner_dataflow_ops": 0,
        }
        print(
            "[tpq-kimi] no-owner 真TP数据流已锁定："
            "Hidden/Router/packed专家输出均由全rank直接持有",
            flush=True,
        )

    def _prepare_no_owner_moe_plans(self) -> None:
        """Cache one generic host submission plan for every routed layer."""
        if os.environ.get("TPQ_TP_MOE_PLAN", "0") == "0":
            return
        if (
            not getattr(self.pool, "hidden_mode", False)
            or not self._tp_moe_all_rank_layer_graph
            or not self._tp_route_packed_graph
            or (
                self.config.latent_moe_use_norm
                and not self._tp_routed_finalize_graph
            )
            or self._tp_shared_mlp is None
            or self._tp_route_down is None
            or self._tp_routed_up is None
        ):
            raise RuntimeError(
                "TPQ_TP_MOE_PLAN requires the complete no-owner all-rank "
                "Graph chain"
            )
        from .ops import TensorParallelMoELayerPlan

        kda_layers = set(self.config.kda_layers)
        for layer in range(
            self.config.first_dense_layers,
            self.config.n_layers,
        ):
            attention_executor = (
                self._tp_kda
                if layer in kda_layers
                else self._tp_mla
            )
            if attention_executor is None:
                raise RuntimeError(
                    f"Attention TP executor is unavailable for layer {layer}"
                )
            self._tp_no_owner_moe_plans[layer] = (
                TensorParallelMoELayerPlan(
                    layer,
                    attention_executor.output_hidden(layer),
                    self._tp_layer_prefix_hidden[layer],
                    self._tp_shared_mlp,
                    self._tp_route_down,
                    self.pool,
                    self._tp_routed_up,
                )
            )
        self.tp_dataflow["moe_submission"] = (
            "one_host_call_fixed_all_rank_plan"
        )
        print(
            "[tpq-kimi] 通用全rank MoE执行计划完成："
            "shared/Router/TopK/packed/routed Up一次主机提交；"
            f"{len(self._tp_no_owner_moe_plans)} 层",
            flush=True,
        )
        self._prepare_no_owner_decode_layer_plans()

    def _prepare_no_owner_decode_layer_plans(self) -> None:
        """Compose captured Attention and MoE into one layer submission."""
        if os.environ.get("TPQ_TP_DECODE_LAYER_PLAN", "0") == "0":
            return
        if (
            len(self._tp_no_owner_moe_plans)
            != self.config.n_layers - self.config.first_dense_layers
            or not self._tp_attention_layer_graph
            or not self._tp_mlp_layer_graph
        ):
            raise RuntimeError(
                "TP decode layer plan requires complete Attention and MoE "
                "all-rank parent graphs"
            )
        from .ops import TensorParallelDecodeLayerPlan

        kda_layers = set(self.config.kda_layers)
        for layer, moe_plan in self._tp_no_owner_moe_plans.items():
            attention_executor = (
                self._tp_kda
                if layer in kda_layers
                else self._tp_mla
            )
            self._tp_decode_layer_plans[layer] = (
                TensorParallelDecodeLayerPlan(
                    layer,
                    attention_executor,
                    moe_plan,
                )
            )
        self.tp_dataflow["decode_layer_submission"] = (
            "one_host_call_attention_to_all_rank_moe"
        )
        print(
            "[tpq-kimi] 通用 Attention→MoE 整层执行计划完成："
            "全rank固定地址、无hidden owner；"
            f"{len(self._tp_decode_layer_plans)} 层",
            flush=True,
        )

    @staticmethod
    def _combine_projection_values(values):
        if all(isinstance(value, torch.Tensor) for value in values):
            return torch.cat(tuple(values), dim=0)
        return ProjectionGroup(values)

    def _load_kda_input_projection(self, layer: int) -> None:
        if layer in self._kda_input_proj:
            return
        prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
        names = [
            f"{prefix}.{projection}.weight"
            for projection in (
                "q_proj",
                "k_proj",
                "v_proj",
                "g_proj",
                "f_a_proj",
                "b_proj",
            )
        ]
        values = [self._take_weight(name) for name in names]
        self._kda_gate_rank[layer] = int(values[4].shape[0])
        self._kda_input_proj[layer] = self._combine_projection_values(
            values
        )
        norm_name = f"{prefix}.o_norm.weight"
        self._weights[norm_name] = self.w(norm_name).to(torch.bfloat16)

    def _load_mla_input_projection(self, layer: int) -> None:
        if layer in self._mla_input_proj:
            return
        prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
        names = [
            f"{prefix}.{projection}.weight"
            for projection in (
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "g_proj",
            )
        ]
        self._mla_input_proj[layer] = self._combine_projection_values(
            [self._take_weight(name) for name in names]
        )

    def _load_dense_gate_up(self, layer: int) -> None:
        if layer in self._dense_gate_up:
            return
        prefix = f"{_ROOT}.model.layers.{layer}.mlp"
        self._dense_gate_up[layer] = self._combine_projection_values(
            (
                self._take_weight(f"{prefix}.gate_proj.weight"),
                self._take_weight(f"{prefix}.up_proj.weight"),
            )
        )

    def _load_shared_gate_up(self, layer: int) -> None:
        if layer in self._shared_gate_up:
            return
        prefix = (
            f"{_ROOT}.model.layers.{layer}."
            "block_sparse_moe.shared_experts"
        )
        self._shared_gate_up[layer] = self._combine_projection_values(
            (
                self._take_weight(f"{prefix}.gate_proj.weight"),
                self._take_weight(f"{prefix}.up_proj.weight"),
            )
        )

    def _load_moe_input_projection(self, layer: int) -> None:
        """Group the block-FP8 projections fed by the same MoE input.

        The public ``ProjectionGroup`` backend preserves the three compact
        source tensors and emits one logical token-sized output.  This is a
        generic grouped GEMV capability; only the split sizes come from the
        model configuration.
        """
        if layer in self._moe_input_proj:
            return
        prefix = f"{_ROOT}.model.layers.{layer}.block_sparse_moe"
        shared = f"{prefix}.shared_experts"
        values = (
            self._take_weight(f"{shared}.gate_proj.weight"),
            self._take_weight(f"{shared}.up_proj.weight"),
            self._take_weight(f"{prefix}.routed_expert_down_proj.weight"),
        )
        # The full latent-MoE executor consumes the original compact sources;
        # keep their row origins instead of physically concatenating BF16.
        if self.device.type == "cpu":
            try:
                self._moe_input_proj[layer] = ProjectionGroup(values)
            except (AttributeError, TypeError, ValueError):
                # Lightweight store/test adapters may expose symbolic values.
                # Preserve their established combine hook; real compact CPU
                # weights stay in the zero-copy ProjectionGroup above.
                self._moe_input_proj[layer] = (
                    self._combine_projection_values(values)
                )
        else:
            self._moe_input_proj[layer] = (
                self._combine_projection_values(values)
            )

    def _prepare_cpu_latent_moe(self) -> None:
        """Build every format-driven full latent-MoE executor once."""
        if (
            self.device.type != "cpu"
            or os.environ.get("TPQ_CPU_FUSED_LATENT_MOE", "1") == "0"
            or not getattr(self.pool, "compact_full_resident", False)
            or not self.config.latent_moe_use_norm
            or self.config.n_group != 1
        ):
            return
        from .cpuext import make_packed_three_layer_cpu
        from .ops import create_latent_resident_moe_layer

        first = self.config.first_dense_layers
        for layer in range(first, self.config.n_layers):
            grouped = self._moe_input_proj.get(layer)
            if not isinstance(grouped, ProjectionGroup):
                continue
            entries = tuple(
                self.pool.pinned.get((layer, expert))
                for expert in range(self.config.n_experts)
            )
            if any(entry is None or len(entry) != 3 for entry in entries):
                continue
            prefix = f"{_ROOT}.model.layers.{layer}.block_sparse_moe"
            shared = f"{prefix}.shared_experts"
            router = self.w(f"{prefix}.gate.weight")
            executor = make_packed_three_layer_cpu(
                entries,
                force_mixed=True,
            )
            executor = create_latent_resident_moe_layer(
                executor,
                entries,
                (*grouped.weights, router),
                (
                    self.w(f"{shared}.down_proj.weight"),
                    self.w(f"{prefix}.routed_expert_up_proj.weight"),
                ),
                self.w(f"{prefix}.gate.e_score_correction_bias"),
                self._mask(layer),
                self.w(f"{prefix}.routed_expert_norm.weight"),
                activation=self.operator_config.expert_activation,
                scoring=self.config.scoring_func,
                top_k=self.config.top_k,
                normalize_route=self.config.norm_topk_prob,
                routed_scaling=self.config.routed_scaling,
                rms_eps=self.config.rms_eps,
                limit=0.0,
                beta=self.config.situ_beta,
                linear_beta=self.config.situ_linear_beta,
            )
            if executor is not None:
                self._cpu_latent_moe_layers[layer] = executor
        if self._cpu_latent_moe_layers:
            print(
                "[tpq-kimi] 公共 CPU latent-MoE 单团队完成："
                f"{len(self._cpu_latent_moe_layers)} 层；"
                "Input/Router→packed Top-K→Norm/Shared/Residual 固定地址",
                flush=True,
            )

    def _combine_dense_projections(self) -> None:
        """Eager compatibility path; TP uses per-layer streaming loaders."""
        for layer in self.config.kda_layers:
            self._load_kda_input_projection(layer)
        for layer in self.config.full_attn_layers:
            self._load_mla_input_projection(layer)
        for layer in range(self.config.first_dense_layers):
            self._load_dense_gate_up(layer)
        for layer in range(
            self.config.first_dense_layers,
            self.config.n_layers,
        ):
            if os.environ.get("TPQ_MOE_INPUT_GROUPED", "1") != "0":
                self._load_moe_input_projection(layer)
            else:
                self._load_shared_gate_up(layer)

    def _make_small_tp_executor(self, kind: str, spec):
        from .ops import create_tensor_parallel
        from .ops.tensor_parallel import OwnerGroupedTensorParallel

        def factory(devices):
            return create_tensor_parallel(kind, tuple(devices), spec)

        no_owner_hidden = (
            os.environ.get("TPQ_TP_HIDDEN_STATE", "0") != "0"
            and os.environ.get("TPQ_TP_NO_OWNER", "1") != "0"
        )
        group_size = (
            len(self.devices)
            if no_owner_hidden
            else min(
                len(self.devices),
                int(os.environ.get("TPQ_SMALL_OP_TP", "4")),
            )
        )
        while group_size > 1 and len(self.devices) % group_size:
            group_size -= 1
        if group_size == 1 and len(self.devices) == 1:
            return factory(self.devices)
        if group_size <= 1:
            raise ValueError("no usable TP subgroup for visible devices")
        if group_size == len(self.devices):
            return factory(self.devices)
        return OwnerGroupedTensorParallel(
            self.devices,
            group_size,
            factory,
        )

    @staticmethod
    def _tp_executor_width(executor) -> int:
        return int(
            getattr(
                executor,
                "group_size",
                len(executor.devices),
            )
        )

    def _attention_tp_input_buffer(
        self,
        layer: int,
    ) -> torch.Tensor | None:
        if os.environ.get("TPQ_TP_DIRECT_INPUT", "1") == "0":
            return None
        executor = (
            self._tp_kda
            if layer in set(self.config.kda_layers)
            else self._tp_mla
        )
        if executor is None:
            return None
        return executor.input_buffer(layer)

    def _mlp_tp_input_buffer(
        self,
        layer: int,
    ) -> torch.Tensor | None:
        if os.environ.get("TPQ_TP_DIRECT_INPUT", "1") == "0":
            return None
        if layer < self.config.first_dense_layers:
            executor = self._tp_dense_mlp
        elif self._tp_moe_prelude is None:
            executor = self._tp_shared_mlp
        else:
            executor = None
        if executor is None:
            return None
        return executor.input_buffer(layer)

    def _prepare_tp_shared_mlp(self) -> None:
        if (
            self._pipeline_plan is None
            or self._tp_moe_prelude is not None
            or os.environ.get(
                "TPQ_SHARED_MLP_TP",
                os.environ.get(
                    "TPQ_DENSE_TP",
                    os.environ.get("TPQ_KIMI_DENSE_TP", "1"),
                ),
            ) == "0"
        ):
            return
        from .ops.tensor_parallel import (
            GatedMLPSpec,
        )

        first_layer = self.config.first_dense_layers
        self._load_shared_gate_up(first_layer)
        first_weight = self._shared_gate_up[first_layer]
        intermediate = first_weight.shape[0] // 2
        spec = GatedMLPSpec(
            hidden_size=self.config.hidden,
            intermediate_size=intermediate,
            activation=self.operator_config.expert_activation,
            activation_beta=self.config.situ_beta,
            activation_linear_beta=self.config.situ_linear_beta,
        )
        executor = self._make_small_tp_executor(
            "gated_mlp",
            spec,
        )
        for layer in range(first_layer, self.config.n_layers):
            self._load_shared_gate_up(layer)
            prefix = (
                f"{_ROOT}.model.layers.{layer}."
                "block_sparse_moe.shared_experts"
            )
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._shared_gate_up.pop(layer),
                self._take_weight(f"{prefix}.down_proj.weight"),
            )
        executor.capture()
        self._tp_shared_mlp = executor
        print(
            "[tpq-kimi] 通用共享 MLP Row-TP Graph 完成："
            f"{self.config.n_layers - first_layer} 层×"
            f"TP{self._tp_executor_width(executor)}",
            flush=True,
        )

    def _prepare_tp_moe_prelude(self) -> None:
        """Fuse the three MoE branches which share the same hidden input."""
        if (
            self._pipeline_plan is None
            or os.environ.get("TPQ_MOE_PRELUDE_TP", "0") == "0"
        ):
            return
        from .ops.tensor_parallel import MoEPreludeSpec

        first_layer = self.config.first_dense_layers
        self._load_shared_gate_up(first_layer)
        shared_intermediate = (
            self._shared_gate_up[first_layer].shape[0] // 2
        )
        executor = self._make_small_tp_executor(
            "moe_prelude",
            MoEPreludeSpec(
                hidden_size=self.config.hidden,
                routed_hidden_size=self.config.routed_hidden,
                shared_intermediate_size=shared_intermediate,
                expert_count=self.config.n_experts,
                activation=self.operator_config.expert_activation,
                activation_beta=self.config.situ_beta,
                activation_linear_beta=self.config.situ_linear_beta,
            ),
        )
        for layer in range(first_layer, self.config.n_layers):
            self._load_shared_gate_up(layer)
            prefix = (
                f"{_ROOT}.model.layers.{layer}."
                "block_sparse_moe"
            )
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._take_weight(f"{prefix}.gate.weight"),
                self._take_weight(
                    f"{prefix}.routed_expert_down_proj.weight"
                ),
                self._shared_gate_up.pop(layer),
                self._take_weight(
                    f"{prefix}.shared_experts.down_proj.weight"
                ),
            )
        executor.capture()
        self._tp_moe_prelude = executor
        print(
            "[tpq-kimi] 通用 MoE 前导大图完成："
            "一次广播并行计算 Router / routed Down / 共享 MLP；"
            f"{self.config.n_layers - first_layer} 层×"
            f"TP{self._tp_executor_width(executor)}",
            flush=True,
        )

    def _prepare_tp_route_down(self) -> None:
        """Build a shared-input Router+routed-Down row-TP graph."""
        if (
            self._pipeline_plan is None
            or self._tp_moe_prelude is not None
            or os.environ.get(
                "TPQ_MOE_ROUTE_DOWN_TP",
                os.environ.get("TPQ_TP_HIDDEN_STATE", "0"),
            ) == "0"
        ):
            return
        from .ops.tensor_parallel import RouteDownSpec

        executor = self._make_small_tp_executor(
            "route_down",
            RouteDownSpec(
                hidden_size=self.config.hidden,
                routed_hidden_size=self.config.routed_hidden,
                expert_count=self.config.n_experts,
            ),
        )
        for layer in range(
            self.config.first_dense_layers,
            self.config.n_layers,
        ):
            prefix = (
                f"{_ROOT}.model.layers.{layer}."
                "block_sparse_moe"
            )
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._take_weight(f"{prefix}.gate.weight"),
                self._take_weight(
                    f"{prefix}.routed_expert_down_proj.weight"
                ),
            )
        if (
            os.environ.get("TPQ_TP_NO_OWNER", "1") != "0"
            and self._tp_shared_mlp is not None
        ):
            for layer in range(
                self.config.first_dense_layers,
                self.config.n_layers,
            ):
                executor.bind_input_hidden(
                    layer,
                    self._tp_shared_mlp.input_hidden(layer),
                )
        executor.capture()
        self._tp_route_down = executor
        print(
            "[tpq-kimi] 通用 Router Column-TP + routed Down Row-TP "
            "Graph 完成：直接读取全rank固定 hidden，规约小 logits 与 latent；"
            f"{self.config.n_layers - self.config.first_dense_layers} 层×"
            f"TP{self._tp_executor_width(executor)}",
            flush=True,
        )

    def _prepare_tp_dense_mlp(self) -> None:
        if (
            self._pipeline_plan is None
            or self.config.first_dense_layers <= 0
            or os.environ.get(
                "TPQ_FIRST_DENSE_TP",
                os.environ.get(
                    "TPQ_DENSE_TP",
                    os.environ.get("TPQ_KIMI_DENSE_TP", "1"),
                ),
            ) == "0"
        ):
            return
        from .ops.tensor_parallel import (
            GatedMLPSpec,
        )

        intermediate = self.config.inter_dense
        executor = self._make_small_tp_executor(
            "gated_mlp",
            GatedMLPSpec(
                hidden_size=self.config.hidden,
                intermediate_size=intermediate,
                activation=self.operator_config.expert_activation,
                activation_beta=self.config.situ_beta,
                activation_linear_beta=self.config.situ_linear_beta,
            ),
        )
        for layer in range(self.config.first_dense_layers):
            self._load_dense_gate_up(layer)
            prefix = f"{_ROOT}.model.layers.{layer}.mlp"
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._dense_gate_up.pop(layer),
                self._take_weight(f"{prefix}.down_proj.weight"),
            )
        executor.capture()
        self._tp_dense_mlp = executor
        print(
            "[tpq-kimi] 通用 Dense MLP Column/Row-TP Graph 完成："
            f"{self.config.first_dense_layers} 层×"
            f"TP{self._tp_executor_width(executor)}",
            flush=True,
        )

    def _prepare_tp_routed_linear(self) -> None:
        if (
            self._pipeline_plan is None
            or (
                os.environ.get(
                    "TPQ_ROUTED_PROJECTION_TP",
                    "0",
                ) == "0"
                and os.environ.get(
                    "TPQ_TP_HIDDEN_STATE",
                    "0",
                ) == "0"
            )
        ):
            return
        from .ops.tensor_parallel import (
            RowParallelLinearSpec,
        )
        from .ops import TPHidden

        down = None
        if (
            self._tp_moe_prelude is None
            and self._tp_route_down is None
            and os.environ.get("TPQ_ROUTED_DOWN_ROW_TP", "0")
            != "0"
        ):
            down = self._make_small_tp_executor(
                "row_linear",
                RowParallelLinearSpec(
                    in_features=self.config.hidden,
                    out_features=self.config.routed_hidden,
                ),
            )
        up = self._make_small_tp_executor(
            "row_linear",
            RowParallelLinearSpec(
                in_features=self.config.routed_hidden,
                out_features=self.config.hidden,
                capture_owner_dispatch=(
                    os.environ.get(
                        "TPQ_TP_HIDDEN_STATE",
                        "0",
                    )
                    != "0"
                    and os.environ.get(
                        "TPQ_TP_NO_OWNER",
                        "1",
                    )
                    == "0"
                ),
            ),
        )
        for layer in range(
            self.config.first_dense_layers,
            self.config.n_layers,
        ):
            prefix = (
                f"{_ROOT}.model.layers.{layer}."
                "block_sparse_moe"
            )
            owner = self._pipeline_plan.owner_by_layer[layer]
            if down is not None:
                down.add_layer(
                    layer,
                    owner,
                    self._take_weight(
                        f"{prefix}.routed_expert_down_proj.weight"
                    ),
                )
            up.add_layer(
                layer,
                owner,
                self._take_weight(
                    f"{prefix}.routed_expert_up_proj.weight"
                ),
            )
            if (
                self.config.latent_moe_use_norm
                and getattr(self.pool, "hidden_mode", False)
                and os.environ.get(
                    "TPQ_TP_HIDDEN_STATE",
                    "0",
                )
                != "0"
                and os.environ.get(
                    "TPQ_TP_NO_OWNER",
                    "1",
                )
                != "0"
            ):
                norm_hidden = TPHidden.empty(
                    self.devices,
                    (1, self.config.routed_hidden),
                    dtype=torch.bfloat16,
                )
                self._tp_routed_norm_hidden[layer] = norm_hidden
                up.bind_input_hidden(layer, norm_hidden)
        if down is not None:
            down.capture()
        up.capture()
        self._tp_routed_down = down
        self._tp_routed_up = up
        projection_label = (
            "routed Down/Up" if down is not None else "routed Up"
        )
        print(
            f"[tpq-kimi] 通用 {projection_label} Row-TP Graph 完成："
            f"{self.config.n_layers - self.config.first_dense_layers} 层×"
            f"TP{self._tp_executor_width(up)}",
            flush=True,
        )

    def _prepare_tp_router(self) -> None:
        if (
            self._pipeline_plan is None
            or self._tp_moe_prelude is not None
            or self._tp_route_down is not None
            or os.environ.get("TPQ_ROUTER_TP", "0") == "0"
        ):
            return
        from .ops.tensor_parallel import (
            RowParallelLinearSpec,
        )

        executor = self._make_small_tp_executor(
            "row_linear",
            RowParallelLinearSpec(
                in_features=self.config.hidden,
                out_features=self.config.n_experts,
                input_dtype=torch.bfloat16,
                weight_dtype=torch.float32,
            ),
        )
        for layer in range(
            self.config.first_dense_layers,
            self.config.n_layers,
        ):
            prefix = (
                f"{_ROOT}.model.layers.{layer}."
                "block_sparse_moe"
            )
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._take_weight(f"{prefix}.gate.weight"),
            )
        executor.capture()
        self._tp_router = executor
        print(
            "[tpq-kimi] 通用 Router Row-TP Graph 完成："
            f"{self.config.n_layers - self.config.first_dense_layers} 层×"
            f"TP{self._tp_executor_width(executor)}；"
            f"仅规约 {self.config.n_experts} 维 logits",
            flush=True,
        )

    def _prepare_tp_hidden_state(self) -> None:
        """Prepare fixed all-rank state using common hidden operators."""
        if (
            self._pipeline_plan is None
            or os.environ.get("TPQ_TP_HIDDEN_STATE", "0") == "0"
        ):
            return
        if os.environ.get("TPQ_TP_HIDDEN", "0") == "0":
            raise RuntimeError(
                "TPQ_TP_HIDDEN_STATE requires TPQ_TP_HIDDEN=1"
            )
        if (
            self._tp_kda is None
            or self._tp_mla is None
            or self._tp_dense_mlp is None
            or self._tp_shared_mlp is None
            or self._tp_routed_up is None
            or (
                getattr(self.pool, "hidden_mode", False)
                and self._tp_route_down is None
            )
        ):
            raise RuntimeError(
                "all-rank hidden state requires Attention, Dense, shared "
                "Router/Down and routed-Up TP operators"
            )
        from .ops import TPHidden, TPResidualBuffer

        hidden_shape = (1, self.config.hidden)
        self._tp_token_hidden = TPHidden.empty(
            self.devices,
            hidden_shape,
            dtype=torch.bfloat16,
        )
        residual_rows = (
            (self.config.n_layers - 1)
            // self.config.attn_res_block_size
            + 1
        )
        self._tp_block_residual = TPResidualBuffer.empty(
            self.devices,
            residual_rows,
            self.config.hidden,
        )
        self._tp_residual_workspaces = tuple(
            torch.empty(
                32,
                dtype=torch.float32,
                device=device,
            )
            for device in self.devices
        )
        for layer in range(
            self.config.first_dense_layers,
            self.config.n_layers,
        ):
            device = self.layer_device(layer)
            with torch.cuda.device(device):
                self._tp_routed_latent_buffers[layer] = torch.empty(
                    1,
                    self.config.routed_hidden,
                    dtype=torch.bfloat16,
                    device=device,
                )
                self._tp_routed_value_buffers[layer] = torch.empty_like(
                    self._tp_routed_latent_buffers[layer]
                )
                self._tp_routed_norm_buffers[layer] = torch.empty_like(
                    self._tp_routed_latent_buffers[layer]
                )
        if (
            not getattr(self.pool, "hidden_mode", False)
            and os.environ.get("TPQ_MOE_OWNER_GRAPH", "0") != "0"
        ):
            from .ops import FixedMoEPrelude, FixedMoEPreludeSpec

            prelude = FixedMoEPrelude(
                FixedMoEPreludeSpec(
                    hidden_size=self.config.hidden,
                    routed_hidden_size=self.config.routed_hidden,
                    expert_count=self.config.n_experts,
                    top_k=self.config.top_k,
                    scoring_func=self.config.scoring_func,
                    normalize=self.config.norm_topk_prob,
                    scaling=self.config.routed_scaling,
                    n_group=self.config.n_group,
                    topk_group=self.config.topk_group,
                )
            )
            for layer in range(
                self.config.first_dense_layers,
                self.config.n_layers,
            ):
                prefix = (
                    f"{_ROOT}.model.layers.{layer}."
                    "block_sparse_moe"
                )
                device = self.layer_device(layer)
                buffers = (
                    torch.empty(
                        1,
                        self.config.n_experts,
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.empty(
                        1,
                        self.config.top_k,
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.empty(
                        1,
                        self.config.top_k,
                        dtype=torch.long,
                        device=device,
                    ),
                )
                self._route_buffers[layer] = buffers
                shared_input = self._tp_shared_mlp.input_hidden(
                    layer
                )
                prelude.add_layer(
                    layer,
                    shared_input.on_device(device),
                    self.w(f"{prefix}.gate.weight"),
                    self.w(
                        f"{prefix}.gate.e_score_correction_bias"
                    ),
                    self._mask(layer),
                    self.w(
                        f"{prefix}.routed_expert_down_proj.weight"
                    ),
                    buffers,
                    self._tp_routed_latent_buffers[layer],
                )
            prelude.capture()
            self._tp_fixed_moe_prelude = prelude
            print(
                "[tpq-kimi] 通用 owner-local Router+routed Down "
                f"固定地址 Graph 完成："
                f"{self.config.n_layers - self.config.first_dense_layers} 层",
                flush=True,
            )

        def replicate(weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tuple(
                (
                    weight
                    if weight.device == device
                    else weight.to(device)
                ).contiguous()
                for device in self.devices
            )

        if getattr(self.pool, "hidden_mode", False):
            for layer in range(
                self.config.first_dense_layers,
                self.config.n_layers,
            ):
                prefix = (
                    f"{_ROOT}.model.layers.{layer}."
                    "block_sparse_moe"
                )
                weight_buffers = []
                index_buffers = []
                for device in self.devices:
                    with torch.cuda.device(device):
                        weight_buffers.append(
                            torch.empty(
                                1,
                                self.config.top_k,
                                dtype=torch.float32,
                                device=device,
                            )
                        )
                        index_buffers.append(
                            torch.empty(
                                1,
                                self.config.top_k,
                                dtype=torch.long,
                                device=device,
                            )
                        )
                self._tp_route_all_rank_buffers[layer] = (
                    tuple(weight_buffers),
                    tuple(index_buffers),
                )
                _router_hidden, latent_hidden = (
                    self._tp_route_down.output_hidden(layer)
                )
                self.pool.bind_hidden_inputs(
                    layer,
                    latent_hidden,
                    tuple(weight_buffers),
                    tuple(index_buffers),
                )
                self._tp_route_corrections[layer] = replicate(
                    self.w(
                        f"{prefix}.gate.e_score_correction_bias"
                    )
                )
                self._tp_route_masks[layer] = replicate(
                    self._mask(layer)
                )
                if self.config.latent_moe_use_norm:
                    if layer not in self._tp_routed_norm_hidden:
                        self._tp_routed_norm_hidden[layer] = (
                            TPHidden.empty(
                                self.devices,
                                (1, self.config.routed_hidden),
                                dtype=torch.bfloat16,
                            )
                        )
                    self._tp_routed_norm_weights[layer] = replicate(
                        self.w(
                            f"{prefix}.routed_expert_norm.weight"
                        )
                    )

        for layer in range(self.config.n_layers):
            prefix = f"{_ROOT}.model.layers.{layer}"
            self._tp_attention_mix[layer] = (
                replicate(
                    self.w(
                        f"{prefix}.self_attention_res_proj.weight"
                    )
                ),
                replicate(
                    self.w(
                        f"{prefix}.self_attention_res_norm.weight"
                    )
                ),
                replicate(
                    self.w(f"{prefix}.input_layernorm.weight")
                ),
            )
            self._tp_mlp_mix[layer] = (
                replicate(
                    self.w(f"{prefix}.mlp_res_proj.weight")
                ),
                replicate(
                    self.w(f"{prefix}.mlp_res_norm.weight")
                ),
                replicate(
                    self.w(
                        f"{prefix}.post_attention_layernorm.weight"
                    )
                ),
            )
            self._tp_layer_prefix_hidden[layer] = TPHidden.empty(
                self.devices,
                hidden_shape,
                dtype=torch.bfloat16,
            )
            if layer < self.config.first_dense_layers:
                self._tp_layer_output_hidden[layer] = TPHidden.empty(
                    self.devices,
                    hidden_shape,
                    dtype=torch.bfloat16,
                )
        self._tp_final_mix = (
            replicate(
                self.w(
                    f"{_ROOT}.model.output_attn_res_proj.weight"
                )
            ),
            replicate(
                self.w(
                    f"{_ROOT}.model.output_attn_res_norm.weight"
                )
            ),
            replicate(self.w(f"{_ROOT}.model.norm.weight")),
        )
        if os.environ.get("TPQ_TP_LAYER_GRAPH", "0") != "0":
            self._prepare_tp_attention_layer_graph()
            self._prepare_tp_mlp_layer_graph()
        self._tp_hidden_state_ready = True
        if len(self.devices) == 1:
            self.tp_dataflow.update(
                {
                    "hidden_layout": "single_rank_fixed_address",
                    "attention_schedule": (
                        "normalize_to_head_to_row_parent_graph"
                    ),
                    "mlp_schedule": (
                        "residual_to_shared_router_down_parent_graph"
                    ),
                    "packed_route_mapping": (
                        "cuda_fixed_slot_directory"
                    ),
                    "packed_moe_compute": "single_fused_cuda_kernel",
                    "ram_miss_fallback": "compact_staged_h2d",
                    "expanded_index_bytes": 0,
                }
            )
        print(
            "[tpq-kimi] 通用 TPHidden 跨层状态完成："
            f"{len(self.devices)} rank，固定 residual={residual_rows} 行",
            flush=True,
        )

    def _prepare_tp_attention_layer_graph(self) -> None:
        """Join rank-local normalization and Attention into one parent Graph."""
        if (
            self._tp_token_hidden is None
            or self._tp_block_residual is None
        ):
            raise RuntimeError("TP layer Graph requires fixed hidden state")
        hidden_source = self._tp_token_hidden
        block = self.config.attn_res_block_size
        kda_layers = set(self.config.kda_layers)
        for layer in range(self.config.n_layers):
            executor = (
                self._tp_kda
                if layer in kda_layers
                else self._tp_mla
            )
            target = executor.input_hidden(layer)
            plan = self._tp_attention_mix[layer]
            active_rows = (
                0
                if layer == 0
                else (layer - 1) // block + 1
            )
            executor.compose_normalize_prelude(
                layer,
                hidden_source,
                self._tp_block_residual,
                active_rows,
                self._tp_select(plan[0], target.devices),
                self._tp_select(plan[1], target.devices),
                self._tp_select(plan[2], target.devices),
                self._tp_select(
                    self._tp_residual_workspaces,
                    target.devices,
                ),
                self.config.rms_eps,
            )
            if layer < self.config.first_dense_layers:
                hidden_source = self._tp_layer_output_hidden[layer]
            else:
                hidden_source = self._tp_routed_up.output_hidden(layer)
        self._tp_attention_layer_graph = True
        print(
            "[tpq-kimi] 通用 Attention 前处理→Column/Row "
            f"父图完成：{self.config.n_layers} 层",
            flush=True,
        )

    def _prepare_tp_mlp_layer_graph(self) -> None:
        """Join prefix/residual preparation and gated MLP per rank."""
        if (
            self._tp_token_hidden is None
            or self._tp_block_residual is None
        ):
            raise RuntimeError("TP MLP layer Graph needs fixed hidden state")
        if (
            self._tp_executor_width(self._tp_dense_mlp)
            != len(self.devices)
            or self._tp_executor_width(self._tp_shared_mlp)
            != len(self.devices)
        ):
            print(
                "[tpq-kimi] MLP 父图暂不跨 TP 子组拼接；"
                "保留 Attention 父图和全 rank 正确路径",
                flush=True,
            )
            return
        hidden_source = self._tp_token_hidden
        block = self.config.attn_res_block_size
        kda_layers = set(self.config.kda_layers)
        for layer in range(self.config.n_layers):
            attention_executor = (
                self._tp_kda
                if layer in kda_layers
                else self._tp_mla
            )
            attention = attention_executor.output_hidden(layer)
            mlp_executor = (
                self._tp_dense_mlp
                if layer < self.config.first_dense_layers
                else self._tp_shared_mlp
            )
            target = mlp_executor.input_hidden(layer)
            plan = self._tp_mlp_mix[layer]
            mlp_executor.compose_mlp_prelude(
                layer,
                hidden_source,
                attention,
                self._tp_layer_prefix_hidden[layer],
                self._tp_block_residual,
                layer // block + 1,
                self._tp_select(plan[0], target.devices),
                self._tp_select(plan[1], target.devices),
                self._tp_select(plan[2], target.devices),
                self._tp_select(
                    self._tp_residual_workspaces,
                    target.devices,
                ),
                self.config.rms_eps,
                boundary=(layer % block == 0),
            )
            if layer < self.config.first_dense_layers:
                hidden_source = self._tp_layer_output_hidden[layer]
            else:
                hidden_source = self._tp_routed_up.output_hidden(layer)
        self._tp_mlp_layer_graph = True
        if (
            getattr(self.pool, "hidden_mode", False)
            and self._tp_route_down is not None
        ):
            for layer in range(
                self.config.first_dense_layers,
                self.config.n_layers,
            ):
                self._tp_shared_mlp.compose_rank_parallel_branch(
                    layer,
                    self._tp_route_down.retained_rank_graphs(layer),
                )
            self._tp_moe_all_rank_layer_graph = True
            print(
                "[tpq-kimi] 通用全rank MLP父图完成："
                "归一化后并行执行 shared MLP 与 Router/Down；"
                f"{self.config.n_layers - self.config.first_dense_layers} 层",
                flush=True,
            )
        if self._tp_fixed_moe_prelude is not None:
            for layer in range(
                self.config.first_dense_layers,
                self.config.n_layers,
            ):
                self._tp_shared_mlp.compose_owner_branch(
                    layer,
                    self._tp_fixed_moe_prelude.retained_graph(layer),
                )
            self._tp_moe_owner_layer_graph = True
            print(
                "[tpq-kimi] 通用 shared MLP ∥ owner Router/Down "
                f"并行分支父图完成："
                f"{self.config.n_layers - self.config.first_dense_layers} 层",
                flush=True,
            )
        print(
            "[tpq-kimi] 通用 Attention 后残差→MLP 输入→"
            f"gated MLP 父图完成：{self.config.n_layers} 层",
            flush=True,
        )

    def _prepare_tp_kda(self) -> None:
        if (
            self._pipeline_plan is None
            or os.environ.get(
                "TPQ_ATTENTION_TP",
                os.environ.get("TPQ_KIMI_ATTENTION_TP", "1"),
            ) == "0"
        ):
            return
        from .ops.tensor_parallel import KDASpec

        first_layer = self.config.kda_layers[0]
        self._load_kda_input_projection(first_layer)
        gate_rank = self._kda_gate_rank[first_layer]
        executor = self._make_small_tp_executor(
            "kda",
            KDASpec(
                hidden_size=self.config.hidden,
                heads=self.config.n_heads,
                head_dim=self.config.head_dim,
                gate_rank=gate_rank,
                rms_eps=self.config.rms_eps,
                gate_lower_bound=float(
                    self.cfg.get("gate_lower_bound", -5.0)
                ),
                conv_history=(
                    int(self.cfg.get("short_conv_kernel_size", 4))
                    - 1
                ),
            ),
        )
        for layer in self.config.kda_layers:
            self._load_kda_input_projection(layer)
            if self._kda_gate_rank[layer] != gate_rank:
                raise ValueError("KDA gate rank must be stable across layers")
            prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._kda_input_proj.pop(layer),
                self._take_weight(f"{prefix}.f_b_proj.weight"),
                tuple(
                    self._take_weight(
                        f"{prefix}.{name}_conv1d.weight"
                    )
                    for name in ("q", "k", "v")
                ),
                self._take_weight(f"{prefix}.A_log"),
                self._take_weight(f"{prefix}.dt_bias"),
                self._take_weight(f"{prefix}.o_norm.weight"),
                self._take_weight(f"{prefix}.o_proj.weight"),
            )
        executor.capture()
        self._tp_kda = executor
        print(
            "[tpq-kimi] 通用 KDA Head-TP Graph 完成："
            f"{len(self.config.kda_layers)} 层×"
            f"TP{self._tp_executor_width(executor)}",
            flush=True,
        )

    def _prepare_tp_mla(self) -> None:
        if (
            self._pipeline_plan is None
            or os.environ.get(
                "TPQ_ATTENTION_TP",
                os.environ.get("TPQ_KIMI_ATTENTION_TP", "1"),
            ) == "0"
            or os.environ.get("TPQ_MLA_TP", "1") == "0"
        ):
            return
        from .ops.tensor_parallel import MLASpec

        executor = self._make_small_tp_executor(
            "mla",
            MLASpec(
                hidden_size=self.config.hidden,
                heads=self.config.n_heads,
                q_lora_rank=self.config.q_lora_rank,
                kv_lora_rank=self.config.kv_lora_rank,
                qk_nope_head_dim=self.config.qk_nope_head_dim,
                qk_rope_head_dim=self.config.qk_rope_head_dim,
                v_head_dim=self.config.v_head_dim,
                max_ctx=self.max_ctx,
                rms_eps=self.config.rms_eps,
            ),
        )
        for layer in self.config.full_attn_layers:
            self._load_mla_input_projection(layer)
            prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
            key_absorb, value_absorb = self._absorbed_weights(layer)
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._mla_input_proj.pop(layer),
                self._take_weight(
                    f"{prefix}.q_a_layernorm.weight"
                ),
                self._take_weight(f"{prefix}.q_b_proj.weight"),
                self._take_weight(
                    f"{prefix}.kv_a_layernorm.weight"
                ),
                key_absorb,
                value_absorb,
                self._take_weight(f"{prefix}.o_proj.weight"),
            )
            self._absorbed.pop(layer, None)
        executor.capture()
        self._tp_mla = executor
        print(
            f"[tpq-kimi] MLA backend={executor.attention_backend}",
            flush=True,
        )
        print(
            "[tpq-kimi] 通用 MLA Head-TP Graph 完成："
            f"{len(self.config.full_attn_layers)} 层×"
            f"TP{self._tp_executor_width(executor)}",
            flush=True,
        )

    def _prepare_tp_vocab(self) -> None:
        """Shard the two large BF16 vocabulary matrices across all ranks."""
        if self._pipeline_plan is None or len(self.devices) == 1:
            return
        from .ops import TensorParallelVocab

        embedding_name = f"{_ROOT}.model.embed_tokens.weight"
        output_name = f"{_ROOT}.lm_head.weight"
        embedding = self.store.get_dense(embedding_name)
        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.dtype != torch.bfloat16
        ):
            raise ValueError("Kimi vocabulary TP requires BF16 embedding rows")
        offsets = TensorParallelVocab.offsets_for(
            int(embedding.shape[0]),
            len(self.devices),
        )

        def shard(weight):
            return tuple(
                weight[offsets[rank]:offsets[rank + 1]]
                .to(device)
                .contiguous()
                for rank, device in enumerate(self.devices)
            )

        embedding_shape = embedding.shape
        embedding_shards = shard(embedding)
        del embedding
        gc.collect()
        output = self.store.get_dense(output_name)
        if (
            not isinstance(output, torch.Tensor)
            or output.dtype != torch.bfloat16
            or output.shape != embedding_shape
        ):
            raise ValueError("Kimi vocabulary TP output rows do not match")
        output_shards = shard(output)
        del output
        self._tp_vocab = TensorParallelVocab(
            self.devices,
            embedding_shards,
            output_shards,
            offsets,
        )
        self._consumed_dense_names.update((embedding_name, output_name))
        self._release_cuda_startup_cache("Vocab TP")
        print(
            "[tpq-kimi] 通用 Vocab-TP 完成：Embedding/LM Head "
            f"按词表行分片到 TP{len(self.devices)}",
            flush=True,
        )

    # ---- persistent attention state ---------------------------------------
    def reset_kv(self) -> None:
        if self._tp_kda is not None:
            self._tp_kda.reset()
        if self._tp_mla is not None:
            self._tp_mla.reset()
        for state in self._kda_state.values():
            state.zero_()
        for states in self._conv_state.values():
            for state in states:
                state.zero_()
        # MLA cache rows beyond ``pos`` are never read; retain capacity and
        # simply rewind the logical length.
        self._paged_latent_prepared.clear()
        self._prev_ids.clear()
        self.pos = 0

    def reset(self) -> None:
        self.reset_kv()

    def _kda_buffers(
        self,
        layer: int,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        config = self.config
        device = self.layer_device(layer)
        state = self._kda_state.get(layer)
        if state is None:
            state = torch.zeros(
                config.n_heads,
                config.head_dim,
                config.head_dim,
                dtype=torch.float32,
                device=device,
            )
            self._kda_state[layer] = state
        conv = self._conv_state.get(layer)
        if conv is None:
            channels = config.n_heads * config.head_dim
            history = int(self.cfg.get("short_conv_kernel_size", 4)) - 1
            prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
            # ``_combine_dense_projections`` removes q_proj from ``_weights``
            # after folding it into the fused KDA input projection.  Asking
            # ``w`` for q_proj here would silently read and allocate a second
            # complete copy on single-GPU runs (all KDA layers together are
            # several GiB).  The fused tensor is the authoritative dtype.
            combined = self._kda_input_proj.get(layer)
            dtype = (
                combined.dtype
                if combined is not None
                else self.w(f"{prefix}.q_proj.weight").dtype
            )
            conv = tuple(
                torch.zeros(
                    channels,
                    history,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(3)
            )
            self._conv_state[layer] = conv
        return state, conv

    def _mla_buffers(
        self,
        layer: int,
        required: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        device = self.layer_device(layer)
        current = self._mla_cache.get(layer)
        if current is not None and current[0].shape[0] >= required:
            return current
        capacity = max(required, 256 if current is None else current[0].shape[0] * 2)
        prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
        # As above, do not reload the source projection after it has been
        # folded into the fused MLA input projection.
        combined = self._mla_input_proj.get(layer)
        dtype = (
            combined.dtype
            if combined is not None
            else self.w(f"{prefix}.kv_a_proj_with_mqa.weight").dtype
        )
        latent = torch.empty(
            capacity,
            config.kv_lora_rank,
            dtype=dtype,
            device=device,
        )
        rope = torch.empty(
            capacity,
            config.qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        if current is not None:
            length = min(self.pos, current[0].shape[0])
            latent[:length].copy_(current[0][:length])
            rope[:length].copy_(current[1][:length])
        self._mla_cache[layer] = (latent, rope)
        return latent, rope

    def _paged_latent_context(
        self,
        *,
        device: torch.device,
        length: int,
        query_nope: torch.Tensor,
        query_rope: torch.Tensor,
        latent_cache: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run the registered paged-latent operator when the backend exists."""
        if (
            device.type != "cuda"
            or latent_cache.dtype != torch.bfloat16
            or os.environ.get("TPQ_PAGED_LATENT_ATTENTION", "1") == "0"
        ):
            return None
        device_index = int(device.index or 0)
        if device_index in self._paged_latent_unavailable:
            return None
        from .ops import attention_step

        runner = self._paged_latent_runners.get(device_index)
        if runner is None:
            try:
                runner = attention_step(
                    "paged_latent_create",
                    "cuda",
                    device=device,
                    max_ctx=self.max_ctx,
                    heads=self.config.n_heads,
                    ckv_dim=self.config.kv_lora_rank,
                    kpe_dim=self.config.qk_rope_head_dim,
                    dtype=latent_cache.dtype,
                    qk_head_dim=(
                        self.config.qk_nope_head_dim
                        + self.config.qk_rope_head_dim
                    ),
                )
            except (ImportError, LookupError, RuntimeError):
                runner = None
            if runner is None:
                self._paged_latent_unavailable.add(device_index)
                return None
            self._paged_latent_runners[device_index] = runner
        if self._paged_latent_prepared.get(device_index) != length:
            try:
                prepared = attention_step(
                    "paged_latent_prepare",
                    "cuda",
                    runner=runner,
                    length=length,
                )
            except (LookupError, RuntimeError):
                prepared = False
            if not prepared:
                self._paged_latent_unavailable.add(device_index)
                return None
            self._paged_latent_prepared[device_index] = length
        page_size = int(runner.page_size)
        capacity = latent_cache.shape[0]
        if capacity % page_size:
            self._paged_latent_unavailable.add(device_index)
            return None
        try:
            return attention_step(
                "paged_latent_decode",
                "cuda",
                runner=runner,
                query_nope=query_nope,
                query_rope=query_rope,
                latent_cache=latent_cache.view(
                    -1,
                    page_size,
                    self.config.kv_lora_rank,
                ),
                rope_cache=rope_cache.view(
                    -1,
                    page_size,
                    self.config.qk_rope_head_dim,
                ),
            )
        except (LookupError, RuntimeError):
            self._paged_latent_unavailable.add(device_index)
            return None

    # ---- attention ---------------------------------------------------------
    def _kda_attention(
        self,
        value: torch.Tensor,
        layer: int,
        *,
        prepared: bool = False,
    ) -> torch.Tensor:
        if self._tp_kda is not None:
            if os.environ.get("TPQ_TP_HIDDEN", "0") != "0":
                hidden = self._tp_kda.input_hidden(layer)
                hidden.copy_from_owner(
                    value,
                    hidden.devices.index(value.device),
                )
                output = self._tp_kda.run_hidden(layer, hidden)
                return output.on_device(value.device)
            output = (
                self._tp_kda.run_prepared(layer)
                if prepared
                else self._tp_kda.run(layer, value)
            )
            return output.to(value.dtype)
        config = self.config
        device = self.layer_device(layer)
        prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
        combined_input = self._kda_input_proj.get(layer)
        if combined_input is None:
            query = _linear(
                value,
                self.w(f"{prefix}.q_proj.weight"),
            ).reshape(config.n_heads, config.head_dim)
            key = _linear(
                value,
                self.w(f"{prefix}.k_proj.weight"),
            ).reshape(config.n_heads, config.head_dim)
            val = _linear(
                value,
                self.w(f"{prefix}.v_proj.weight"),
            ).reshape(config.n_heads, config.head_dim)
            output_gate = _linear(
                value,
                self.w(f"{prefix}.g_proj.weight"),
            ).view(config.n_heads, config.head_dim)
            low_rank_gate = _linear(
                value,
                self.w(f"{prefix}.f_a_proj.weight"),
            )
            beta = _linear(
                value,
                self.w(f"{prefix}.b_proj.weight"),
            ).reshape(config.n_heads).float()
        else:
            projected = _linear(
                value,
                combined_input,
            ).split(
                (
                    config.n_heads * config.head_dim,
                    config.n_heads * config.head_dim,
                    config.n_heads * config.head_dim,
                    config.n_heads * config.head_dim,
                    self._kda_gate_rank[layer],
                    config.n_heads,
                ),
                dim=-1,
            )
            query, key, val, output_gate = (
                item.reshape(config.n_heads, config.head_dim)
                for item in projected[:4]
            )
            low_rank_gate = projected[4]
            beta = projected[5].reshape(config.n_heads).float()
        state, conv = self._kda_buffers(layer)
        conv_fused = False
        try:
            from .ops import attention_step

            conv_fused = attention_step(
                "short_conv3",
                device.type,
                query=query.reshape(-1),
                key=key.reshape(-1),
                value=val.reshape(-1),
                states=conv,
                weights=(
                    self.w(f"{prefix}.q_conv1d.weight"),
                    self.w(f"{prefix}.k_conv1d.weight"),
                    self.w(f"{prefix}.v_conv1d.weight"),
                ),
            )
        except (ImportError, RuntimeError, LookupError):
            conv_fused = False
        if not conv_fused:
            query, query_state = short_conv_step(
                query.reshape(-1),
                conv[0],
                self.w(f"{prefix}.q_conv1d.weight"),
            )
            key, key_state = short_conv_step(
                key.reshape(-1),
                conv[1],
                self.w(f"{prefix}.k_conv1d.weight"),
            )
            val, value_state = short_conv_step(
                val.reshape(-1),
                conv[2],
                self.w(f"{prefix}.v_conv1d.weight"),
            )
            self._conv_state[layer] = (
                query_state,
                key_state,
                value_state,
            )
        query = query.view(config.n_heads, config.head_dim)
        key = key.view(config.n_heads, config.head_dim)
        val = val.view(config.n_heads, config.head_dim)
        recurrent_gate = _linear(
            low_rank_gate,
            self.w(f"{prefix}.f_b_proj.weight"),
        ).view(config.n_heads, config.head_dim)

        output = None
        kda_output_norm_fused = False
        if not self._kda_fused_failed:
            try:
                from .ops import attention_step

                device_index = int(device.index or 0)
                workspace = self._kda_workspace.get(device_index)
                output_buffer = self._kda_output.get(device_index)
                if workspace is None:
                    workspace = torch.empty(
                        3 * config.n_heads * config.head_dim,
                        dtype=torch.float32,
                        device=device,
                    )
                    output_buffer = torch.empty(
                        config.n_heads,
                        config.head_dim,
                        dtype=(
                            query.dtype
                            if device.type == "cpu"
                            else compute_dtype(device)
                        ),
                        device=device,
                    )
                    self._kda_workspace[device_index] = workspace
                    self._kda_output[device_index] = output_buffer
                fused_norm_kwargs = (
                    {
                        "output_gate": output_gate,
                        "norm_weight": self.w(
                            f"{prefix}.o_norm.weight"
                        ),
                        "norm_eps": config.rms_eps,
                    }
                    if device.type == "cpu"
                    else {}
                )
                output = attention_step(
                    "kda_recurrent",
                    device.type,
                    query=query,
                    key=key,
                    value=val,
                    gate=recurrent_gate,
                    beta=beta,
                    a_log=self.w(f"{prefix}.A_log").float(),
                    dt_bias=self.w(f"{prefix}.dt_bias").float(),
                    state=state,
                    workspace=workspace,
                    output=output_buffer,
                    lower_bound=float(
                        self.cfg.get("gate_lower_bound", -5.0)
                    ),
                    **fused_norm_kwargs,
                )
                kda_output_norm_fused = (
                    output is not None and bool(fused_norm_kwargs)
                )
            except (ImportError, RuntimeError, LookupError):
                self._kda_fused_failed = True
                output = None
        if output is None:
            output = kda_recurrent_step(
                query,
                key,
                val,
                recurrent_gate,
                beta,
                self.w(f"{prefix}.A_log"),
                self.w(f"{prefix}.dt_bias"),
                state,
                lower_bound=float(
                    self.cfg.get("gate_lower_bound", -5.0)
                ),
            )

        normalized = output if kda_output_norm_fused else None
        if normalized is None:
            try:
                from .ops import attention_step

                normalized = attention_step(
                    "gated_rmsnorm",
                    device.type,
                    value=output,
                    gate=output_gate,
                    weight=self.w(f"{prefix}.o_norm.weight"),
                    output=output,
                    eps=config.rms_eps,
                )
            except (ImportError, RuntimeError, LookupError):
                normalized = None
        output = (
            normalized
            if normalized is not None
            else gated_rmsnorm(
                output,
                output_gate,
                self.w(f"{prefix}.o_norm.weight"),
                config.rms_eps,
            )
        )
        return _linear(
            output.reshape(1, -1),
            self.w(f"{prefix}.o_proj.weight"),
        )

    def _absorbed_weights(
        self,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._absorbed.get(layer)
        if cached is not None:
            return cached
        config = self.config
        prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
        source_name = f"{prefix}.kv_b_proj.weight"
        weight = self.w(source_name)
        if isinstance(weight, BlockFP8Weight):
            # MLA consumes smaller absorbed factors. Materialize this source
            # once and release it immediately; no full matrix remains resident.
            weight = weight.dequant_rows(
                0,
                weight.shape[0],
                torch.bfloat16,
            )
        weight = weight.view(
            config.n_heads,
            config.qk_nope_head_dim + config.v_head_dim,
            config.kv_lora_rank,
        )
        cached = (
            weight[:, : config.qk_nope_head_dim].contiguous(),
            weight[:, config.qk_nope_head_dim :].contiguous(),
        )
        self._absorbed[layer] = cached
        # The absorbed factors contain every value from KV-B and are the only
        # representation used by decode.  Releasing the original avoids
        # retaining a duplicate ~25 MiB tensor for each MLA layer.
        self._weights.pop(source_name, None)
        self._consumed_dense_names.add(source_name)
        return cached

    @staticmethod
    def _rmsnorm(
        value: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if value.device.type == "cuda":
            from .ops import rmsnorm as registered_rmsnorm

            result = registered_rmsnorm(
                value,
                weight,
                eps,
                output=output,
            )
            if result is not None:
                return result
        result = rmsnorm(value, weight, eps)
        if output is not None:
            output.copy_(result)
            return output
        return result

    def _attention_residual(
        self,
        prefix: torch.Tensor,
        residual: torch.Tensor,
        projection: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        post_norm_weight: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        residual_inverse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prefix.device.type == "cuda" and residual.shape[-2]:
            from .ops import residual_mix

            device_index = int(prefix.device.index or 0)
            buffers = self._residual_buffers.get(device_index)
            if buffers is None:
                buffers = (
                    torch.empty_like(prefix),
                    torch.empty_like(prefix),
                    torch.empty(
                        32,
                        dtype=torch.float32,
                        device=prefix.device,
                    ),
                )
                self._residual_buffers[device_index] = buffers
            if output is None:
                output = next(
                    buffer
                    for buffer in buffers[:2]
                    if buffer.data_ptr() != prefix.data_ptr()
                )
            elif output.data_ptr() == prefix.data_ptr():
                raise ValueError(
                    "attention residual output must not alias prefix"
                )
            fused = residual_mix(
                "attention",
                prefix,
                residual,
                projection,
                norm_weight,
                eps,
                output=output,
                post_norm_weight=post_norm_weight,
                workspace=buffers[2],
                residual_inverse=residual_inverse,
            )
            if fused is not None:
                return fused
        result = attention_residual(
            prefix,
            residual,
            projection,
            norm_weight,
            eps,
        )
        if post_norm_weight is not None:
            result = self._rmsnorm(
                result,
                post_norm_weight,
                eps,
                output=output,
            )
        elif output is not None:
            output.copy_(result)
            result = output
        return result

    def _mla_attention(
        self,
        value: torch.Tensor,
        layer: int,
        position: int,
        *,
        prepared: bool = False,
    ) -> torch.Tensor:
        if self._tp_mla is not None:
            if os.environ.get("TPQ_TP_HIDDEN", "0") != "0":
                hidden = self._tp_mla.input_hidden(layer)
                hidden.copy_from_owner(
                    value,
                    hidden.devices.index(value.device),
                )
                output = self._tp_mla.run_hidden(
                    layer,
                    hidden,
                    position,
                )
                return output.on_device(value.device)
            output = (
                self._tp_mla.run_prepared(layer, position)
                if prepared
                else self._tp_mla.run(layer, value, position)
            )
            return output.to(value.dtype)
        config = self.config
        prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
        combined_input = self._mla_input_proj.get(layer)
        if combined_input is None:
            query_source = _linear(
                value,
                self.w(f"{prefix}.q_a_proj.weight"),
            )
            compressed = _linear(
                value,
                self.w(f"{prefix}.kv_a_proj_with_mqa.weight"),
            )
            output_gate = _linear(
                value,
                self.w(f"{prefix}.g_proj.weight"),
            ).sigmoid()
        else:
            query_source, compressed, output_gate = _linear(
                value,
                combined_input,
            ).split(
                (
                    config.q_lora_rank,
                    config.kv_lora_rank + config.qk_rope_head_dim,
                    config.n_heads * config.v_head_dim,
                ),
                dim=-1,
            )
            output_gate = output_gate.sigmoid()
        query_residual = self._rmsnorm(
            query_source,
            self.w(f"{prefix}.q_a_layernorm.weight"),
            1e-6,
        )
        query = _linear(
            query_residual,
            self.w(f"{prefix}.q_b_proj.weight"),
        ).view(config.n_heads, -1)
        query_nope, query_rope = query.split(
            [config.qk_nope_head_dim, config.qk_rope_head_dim],
            dim=-1,
        )
        latent, key_rope = compressed.split(
            [config.kv_lora_rank, config.qk_rope_head_dim],
            dim=-1,
        )
        latent = self._rmsnorm(
            latent,
            self.w(f"{prefix}.kv_a_layernorm.weight"),
            1e-6,
        )
        latent_cache, rope_cache = self._mla_buffers(layer, position + 1)
        latent_cache[position].copy_(latent[0])
        rope_cache[position].copy_(key_rope[0])
        history_latent = latent_cache[: position + 1]
        history_rope = rope_cache[: position + 1]

        key_absorb, value_absorb = self._absorbed_weights(layer)
        absorbed_query = torch.bmm(
            query_nope[:, None, :],
            key_absorb,
        )
        paged_context = self._paged_latent_context(
            device=value.device,
            length=position + 1,
            query_nope=absorbed_query.transpose(0, 1),
            query_rope=query_rope.unsqueeze(0),
            latent_cache=latent_cache,
            rope_cache=rope_cache,
        )
        if paged_context is None:
            scores = (
                torch.matmul(absorbed_query, history_latent.t())
                + torch.matmul(
                    query_rope[:, None, :],
                    history_rope.t(),
                )
            ) * (1.0 / math.sqrt(
                config.qk_nope_head_dim + config.qk_rope_head_dim
            ))
            probabilities = scores.float().softmax(dim=-1).to(
                history_latent.dtype
            )
            context = torch.matmul(probabilities, history_latent)
        else:
            context = paged_context.transpose(0, 1)
        output = torch.bmm(
            context,
            value_absorb.transpose(1, 2),
        ).reshape(1, config.n_heads * config.v_head_dim)
        return _linear(
            output * output_gate,
            self.w(f"{prefix}.o_proj.weight"),
        )

    # ---- MLP / routing -----------------------------------------------------
    def _mask(self, layer: int) -> torch.Tensor:
        mask = self._masks.get(layer)
        if mask is None:
            mask = self.store.available_mask(layer).to(
                self.layer_device(layer)
            )
            self._masks[layer] = mask
        return mask

    def _dense_mlp(
        self,
        value: torch.Tensor,
        layer: int,
        *,
        prepared: bool = False,
    ) -> torch.Tensor:
        if self._tp_dense_mlp is not None:
            output = (
                self._tp_dense_mlp.run_prepared(layer)
                if prepared
                else self._tp_dense_mlp.run(layer, value)
            )
            return output.to(value.dtype)
        prefix = f"{_ROOT}.model.layers.{layer}.mlp"
        combined_gate_up = self._dense_gate_up.get(layer)
        if combined_gate_up is None:
            gate = _linear(
                value,
                self.w(f"{prefix}.gate_proj.weight"),
            )
            up = _linear(
                value,
                self.w(f"{prefix}.up_proj.weight"),
            )
        else:
            gate, up = _linear(
                value,
                combined_gate_up,
            ).chunk(2, dim=-1)
        activated = activate_gate_up(
            gate,
            up,
            activation=self.operator_config.expert_activation,
            situ_beta=self.config.situ_beta,
            situ_linear_beta=self.config.situ_linear_beta,
        )
        return _linear(
            activated,
            self.w(f"{prefix}.down_proj.weight"),
        )

    def _moe(
        self,
        value: torch.Tensor,
        layer: int,
        residual: torch.Tensor | None = None,
        *,
        prepared: bool = False,
    ) -> torch.Tensor:
        config = self.config
        device = self.layer_device(layer)
        latent_executor = self._cpu_latent_moe_layers.get(layer)
        if latent_executor is not None and residual is not None:
            fused_started = (
                time.perf_counter() if self._profile_enabled else None
            )
            fused = latent_executor.forward_latent_moe(value, residual)
            if fused.numel():
                self.pool.hits += config.top_k
                self._cpu_profile_finish("moe_latent_fused", fused_started)
                return fused if fused.dtype == value.dtype else fused.to(value.dtype)
        prefix = f"{_ROOT}.model.layers.{layer}.block_sparse_moe"
        prelude_logits = None
        prelude_latent = None
        prelude_shared = None
        prelude_shared_gate = None
        prelude_shared_up = None
        grouped_input = self._moe_input_proj.get(layer)
        grouped_started = (
            time.perf_counter()
            if self._profile_enabled and device.type == "cpu"
            else None
        )
        if grouped_input is not None:
            shared_intermediate = config.n_shared * config.moe_inter
            (
                prelude_shared_gate,
                prelude_shared_up,
                prelude_latent,
            ) = _linear(value, grouped_input).split(
                (
                    shared_intermediate,
                    shared_intermediate,
                    config.routed_hidden,
                ),
                dim=-1,
            )
        self._cpu_profile_finish("moe_input_projection", grouped_started)
        if self._tp_moe_prelude is not None:
            prelude_event = self._cuda_stage_start(
                layer,
                "moe_prelude",
                device,
            )
            (
                prelude_logits,
                prelude_latent,
                prelude_shared,
            ) = self._tp_moe_prelude.run(layer, value)
            self._cuda_stage_end(prelude_event)
        shared_pending = None
        shared_partials = None
        if (
            self._tp_moe_prelude is None
            and self._tp_shared_mlp is not None
        ):
            if (
                os.environ.get("TPQ_TP_HIDDEN", "0") != "0"
                and self._tp_routed_up is not None
                and residual is not None
            ):
                shared_hidden = self._tp_shared_mlp.input_hidden(
                    layer
                )
                shared_hidden.copy_from_owner(
                    value,
                    shared_hidden.devices.index(value.device),
                )
                shared_partials = (
                    self._tp_shared_mlp.launch_partials(
                        layer,
                        shared_hidden,
                    )
                )
            else:
                shared_pending = (
                    self._tp_shared_mlp.start_prepared(layer)
                    if prepared
                    else self._tp_shared_mlp.start(layer, value)
                )
        if (
            self._tp_moe_prelude is None
            and self._tp_route_down is not None
        ):
            route_down_event = self._cuda_stage_start(
                layer,
                "moe_route_down_tp",
                device,
            )
            prelude_logits, prelude_latent = (
                self._tp_route_down.run(layer, value)
            )
            self._cuda_stage_end(route_down_event)
        gate_weight = (
            None
            if (
                prelude_logits is not None
                or self._tp_router is not None
            )
            else self.w(f"{prefix}.gate.weight")
        )
        correction = self.w(
            f"{prefix}.gate.e_score_correction_bias"
        )
        available = self._mask(layer)

        def compute_route():
            route = None
            if prelude_logits is not None:
                try:
                    from .ops import route_topk

                    route = route_topk(
                        prelude_logits,
                        correction,
                        available,
                        scoring_func=config.scoring_func,
                        top_k=config.top_k,
                        normalize=config.norm_topk_prob,
                        scaling=config.routed_scaling,
                        n_group=config.n_group,
                        topk_group=config.topk_group,
                    )
                except (ImportError, RuntimeError, LookupError):
                    route = None
            if (
                route is None
                and config.scoring_func == "sigmoid"
                and config.norm_topk_prob
                and config.n_group == 1
            ):
                try:
                    from .ops import linear_route_topk, route_topk

                    buffers = self._route_buffers.get(layer)
                    if buffers is None:
                        buffers = (
                            torch.empty(
                                1,
                                config.n_experts,
                                dtype=torch.float32,
                                device=device,
                            ),
                            torch.empty(
                                1,
                                config.top_k,
                                dtype=torch.float32,
                                device=device,
                            ),
                            torch.empty(
                                1,
                                config.top_k,
                                dtype=torch.long,
                                device=device,
                            ),
                        )
                        self._route_buffers[layer] = buffers
                    if prelude_logits is not None:
                        route = route_topk(
                            prelude_logits,
                            correction,
                            available,
                            scoring_func=config.scoring_func,
                            top_k=config.top_k,
                            normalize=config.norm_topk_prob,
                            scaling=config.routed_scaling,
                            n_group=config.n_group,
                            topk_group=config.topk_group,
                            output_buffers=(buffers[1], buffers[2]),
                        )
                    elif self._tp_router is not None:
                        logits = self._tp_router.run(layer, value)
                        route = route_topk(
                            logits,
                            correction,
                            available,
                            scoring_func=config.scoring_func,
                            top_k=config.top_k,
                            normalize=config.norm_topk_prob,
                            scaling=config.routed_scaling,
                            n_group=config.n_group,
                            topk_group=config.topk_group,
                            output_buffers=(buffers[1], buffers[2]),
                        )
                    else:
                        if gate_weight is None:
                            raise RuntimeError(
                                "router weight is unavailable"
                            )
                        route = linear_route_topk(
                            value,
                            gate_weight,
                            correction,
                            available,
                            scoring_func=config.scoring_func,
                            top_k=config.top_k,
                            normalize=config.norm_topk_prob,
                            scaling=config.routed_scaling,
                            n_group=config.n_group,
                            topk_group=config.topk_group,
                            output_buffers=buffers,
                        )
                    if (
                        route is None
                        and prelude_logits is None
                        and self._tp_router is None
                    ):
                        if gate_weight is None:
                            raise RuntimeError(
                                "router weight is unavailable"
                            )
                        logits = _linear(value.float(), gate_weight)
                        route = route_topk(
                            logits,
                            correction,
                            available,
                            scoring_func=config.scoring_func,
                            top_k=config.top_k,
                            normalize=config.norm_topk_prob,
                            scaling=config.routed_scaling,
                            n_group=config.n_group,
                            topk_group=config.topk_group,
                            output_buffers=(buffers[1], buffers[2]),
                        )
                except (ImportError, RuntimeError):
                    route = None
            if route is None:
                if prelude_logits is not None:
                    raise RuntimeError(
                        "MoE prelude logits did not produce a route"
                    )
                if gate_weight is None:
                    raise RuntimeError(
                        "TP router did not produce a route"
                    )
                route = route_experts(
                    value,
                    gate_weight,
                    correction,
                    available,
                    top_k=config.top_k,
                    normalize=config.norm_topk_prob,
                    scaling=config.routed_scaling,
                    n_group=config.n_group,
                    topk_group=config.topk_group,
                )
            return route

        route_event = self._cuda_stage_start(
            layer,
            "moe_route",
            device,
        )
        route_started = (
            time.perf_counter()
            if self._profile_enabled and device.type == "cpu"
            else None
        )
        route = compute_route()
        self._cpu_profile_finish("moe_router", route_started)
        self._cuda_stage_end(route_event)
        down_event = self._cuda_stage_start(
            layer,
            "moe_routed_down",
            device,
        )
        if prelude_latent is not None:
            latent = prelude_latent.to(value.dtype)
        elif self._tp_routed_down is not None:
            latent = self._tp_routed_down.run(
                layer,
                value,
            ).to(value.dtype)
        else:
            latent = _linear(
                value,
                self.w(f"{prefix}.routed_expert_down_proj.weight"),
            )
        self._cuda_stage_end(down_event)
        weights, indices = route

        def compute_shared_output() -> torch.Tensor:
            cpu_started = (
                time.perf_counter()
                if self._profile_enabled and device.type == "cpu"
                else None
            )
            shared_event = self._cuda_stage_start(
                layer,
                "moe_shared",
                device,
            )
            shared_prefix = f"{prefix}.shared_experts"
            if prelude_shared is not None:
                shared_output = prelude_shared.to(value.dtype)
            elif shared_pending is not None:
                shared_output = self._tp_shared_mlp.finish(
                    layer,
                    shared_pending,
                ).to(value.dtype)
            else:
                combined_gate_up = self._shared_gate_up.get(layer)
                if (
                    prelude_shared_gate is not None
                    and prelude_shared_up is not None
                ):
                    shared_gate = prelude_shared_gate
                    shared_up = prelude_shared_up
                elif combined_gate_up is None:
                    shared_gate = _linear(
                        value,
                        self.w(f"{shared_prefix}.gate_proj.weight"),
                    )
                    shared_up = _linear(
                        value,
                        self.w(f"{shared_prefix}.up_proj.weight"),
                    )
                else:
                    shared_gate, shared_up = _linear(
                        value,
                        combined_gate_up,
                    ).chunk(2, dim=-1)
                shared_output = _linear(
                    activate_gate_up(
                        shared_gate,
                        shared_up,
                        activation=self.operator_config.expert_activation,
                        situ_beta=config.situ_beta,
                        situ_linear_beta=config.situ_linear_beta,
                    ),
                    self.w(f"{shared_prefix}.down_proj.weight"),
                )
            self._cuda_stage_end(shared_event)
            self._cpu_profile_finish("moe_shared", cpu_started)
            return shared_output

        overlap_shared = (
            device.type == "cuda"
            and os.environ.get("TPQ_KIMI_OVERLAP_SHARED", "1") != "0"
            and os.environ.get("TPQ_KIMI_LAYER_TIMING", "0") == "0"
            and hasattr(self.pool, "prepare_run")
            and hasattr(self.pool, "finish_run")
        )
        shared = None
        expert_event = self._cuda_stage_start(
            layer,
            "moe_packed_experts",
            device,
        )
        expert_started = (
            time.perf_counter()
            if self._profile_enabled and device.type == "cpu"
            else None
        )
        if (
            self._pipeline_plan is not None
            or getattr(self.pool, "device_routed", False)
        ):
            if overlap_shared:
                pending = self.pool.prepare_run(
                    layer,
                    latent,
                    indices[0],
                    weights[0],
                    activation=self.operator_config.expert_activation,
                    activation_beta=config.situ_beta,
                    activation_linear_beta=config.situ_linear_beta,
                )
                try:
                    shared = compute_shared_output()
                    routed = self.pool.finish_run(pending)
                except BaseException:
                    self.pool.cancel_run(pending)
                    raise
            else:
                routed = self.pool.run(
                    layer,
                    latent,
                    indices[0],
                    weights[0],
                    activation=self.operator_config.expert_activation,
                    activation_beta=config.situ_beta,
                    activation_linear_beta=config.situ_linear_beta,
                )
            routed = routed.view(
                1,
                config.routed_hidden,
            ).to(value.dtype)
            remember_route = os.environ.get(
                "TPQ_PREFETCH",
                "1" if getattr(self.pool, "prefetch_default", True) else "0",
            ) != "0"
            if (
                getattr(self.pool, "device_routed", False)
                and remember_route
            ):
                self._prev_ids[layer] = self.pool.last_expert_ids(layer)
        else:
            native = getattr(self.pool, "run_native", None)
            routed = (
                native(
                    layer,
                    latent,
                    indices[0],
                    weights[0],
                    activation=self.operator_config.expert_activation,
                    activation_beta=config.situ_beta,
                    activation_linear_beta=config.situ_linear_beta,
                )
                if native is not None
                else None
            )
            expert_ids = indices[0].tolist()
            self._prev_ids[layer] = expert_ids
            if routed is None:
                selected = self.pool.get_many(
                    [(layer, expert_id) for expert_id in expert_ids]
                )
                routed = moe_mlp_grouped_mixed(
                    latent,
                    [
                        selected[(layer, expert_id)]
                        for expert_id in expert_ids
                    ],
                    weights[0],
                    activation=self.operator_config.expert_activation,
                    situ_beta=config.situ_beta,
                    situ_linear_beta=config.situ_linear_beta,
                )
            routed = routed.view(1, config.routed_hidden).to(value.dtype)
        self._cuda_stage_end(expert_event)
        self._cpu_profile_finish("moe_packed_experts", expert_started)
        routed_up_event = self._cuda_stage_start(
            layer,
            "moe_routed_up",
            device,
        )
        routed_up_started = (
            time.perf_counter()
            if self._profile_enabled and device.type == "cpu"
            else None
        )
        if config.latent_moe_use_norm:
            routed = self._rmsnorm(
                routed,
                self.w(f"{prefix}.routed_expert_norm.weight"),
                config.rms_eps,
            )
        if self._tp_routed_up is not None:
            if shared_partials is not None and residual is not None:
                from .ops import TPHidden

                sharded = self._tp_routed_up.input_sharded(layer)
                sharded.copy_from_full(routed)
                residual_hidden = self._tp_moe_residual_hidden.get(
                    layer
                )
                if residual_hidden is None:
                    output_hidden = (
                        self._tp_routed_up.output_hidden(layer)
                    )
                    residual_hidden = TPHidden.empty(
                        output_hidden.devices,
                        tuple(residual.shape),
                        dtype=residual.dtype,
                    )
                    self._tp_moe_residual_hidden[layer] = (
                        residual_hidden
                    )
                residual_hidden.copy_from_owner(
                    residual,
                    residual_hidden.devices.index(residual.device),
                )
                routed = self._tp_routed_up.finalize_moe(
                    layer,
                    sharded,
                    shared_partials,
                    residual_hidden,
                ).on_device(device)
                self._cuda_stage_end(routed_up_event)
                return routed
            routed = self._tp_routed_up.run(
                layer,
                routed,
            ).to(value.dtype)
        else:
            routed = _linear(
                routed,
                self.w(f"{prefix}.routed_expert_up_proj.weight"),
            )
        self._cuda_stage_end(routed_up_event)
        self._cpu_profile_finish("moe_routed_up", routed_up_started)
        if shared is None:
            shared = compute_shared_output()
        if residual is not None and device.type == "cuda":
            from .ops import residual_add3

            combined = residual_add3(residual, routed, shared)
            if combined is not None:
                return combined
        expert_sum = routed + shared
        return (
            expert_sum
            if residual is None
            else residual + expert_sum
        )

    def _cpu_profile_finish(
        self,
        name: str,
        started: float | None,
    ) -> None:
        if started is not None:
            self._cpu_substage_profile[name] = (
                self._cpu_substage_profile.get(name, 0.0)
                + time.perf_counter()
                - started
            )

    def _cuda_stage_start(
        self,
        layer: int,
        stage: str,
        device: torch.device,
    ) -> torch.cuda.Event | None:
        records = self._active_cuda_events
        if records is None:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
        records.append(
            (layer, int(device.index or 0), stage, start, end)
        )
        return end

    @staticmethod
    def _cuda_stage_end(event: torch.cuda.Event | None) -> None:
        if event is not None:
            event.record()

    @staticmethod
    def _pool_profile_counters(pool) -> dict[str, float | int]:
        """Read monotonic cache counters without depending on one pool type."""
        return {
            "hits": int(getattr(pool, "hits", 0)),
            "misses": int(
                getattr(pool, "misses", getattr(pool, "miss", 0))
            ),
            "prefetch_hits": int(getattr(pool, "prefetch_hits", 0)),
            "uploaded_bytes": int(getattr(pool, "uploaded_bytes", 0)),
            "transfer_seconds": float(
                getattr(pool, "transfer_seconds", 0.0)
            ),
            "route_plan_hits": int(
                getattr(pool, "route_plan_hits", 0)
            ),
            "route_plan_misses": int(
                getattr(pool, "route_plan_misses", 0)
            ),
            "device_route_lookups": int(
                getattr(pool, "device_route_lookups", 0)
            ),
            "device_route_full_hits": int(
                getattr(pool, "device_route_full_hits", 0)
            ),
            "device_route_fallbacks": int(
                getattr(pool, "device_route_fallbacks", 0)
            ),
            "native_packed_hits": int(
                getattr(pool, "native_hits", 0)
            ),
            "native_packed_fallbacks": int(
                getattr(pool, "native_fallbacks", 0)
            ),
        }

    def start_profile(self) -> None:
        """Enable the official one-token Kimi CLI stage probe.

        Normal benchmark iterations stay untouched.  The probe token records
        CUDA events and wall-clock layer envelopes, then resolves them with a
        single final synchronization through :meth:`finish_profile`.
        """
        self.last_layer_profile = []
        self.last_cuda_profile = {}
        self._cpu_substage_profile = {}
        self._profile_pool_snapshot = self._pool_profile_counters(self.pool)
        if self.device.type == "cpu":
            from .cpuext import (
                reset_block_fp8_gemv_profile,
                reset_latent_moe_phase_profile,
                reset_packed_moe_phase_profile,
                reset_three_projection_phase_profile,
            )

            reset_block_fp8_gemv_profile()
            reset_latent_moe_phase_profile()
            reset_packed_moe_phase_profile()
            reset_three_projection_phase_profile()
        self._profile_enabled = True

    def finish_profile(self) -> dict[str, object]:
        """Finish and aggregate a Kimi probe in JSON-serializable form."""
        self._profile_enabled = False
        collect_transfer = getattr(
            self.pool,
            "collect_transfer_timing",
            None,
        )
        if callable(collect_transfer):
            collect_transfer(synchronize=True)
        elif self.device.type == "cuda":
            for device in self.devices:
                torch.cuda.synchronize(device)

        layers = [dict(item) for item in self.last_layer_profile]
        wall_totals_ms = {
            "device_transfer_ms": 0.0,
            "attention_ms": 0.0,
            "mlp_ms": 0.0,
            "expert_transfer_wait_ms": 0.0,
            "layer_envelope_ms": 0.0,
        }
        for item in layers:
            wall_totals_ms["device_transfer_ms"] += (
                float(item.get("transfer_seconds", 0.0)) * 1000.0
            )
            wall_totals_ms["attention_ms"] += (
                float(item.get("attention_seconds", 0.0)) * 1000.0
            )
            wall_totals_ms["mlp_ms"] += (
                float(item.get("mlp_seconds", 0.0)) * 1000.0
            )
            wall_totals_ms["expert_transfer_wait_ms"] += (
                float(item.get("expert_transfer_seconds", 0.0)) * 1000.0
            )
            wall_totals_ms["layer_envelope_ms"] += (
                float(item.get("layer_seconds", 0.0)) * 1000.0
            )
        top_layers = sorted(
            layers,
            key=lambda item: float(item.get("layer_seconds", 0.0)),
            reverse=True,
        )[:8]
        cuda_layer_totals: dict[int, dict[str, float | int]] = {}
        for item in self.last_cuda_profile.get("items", []):
            layer = int(item.get("layer", -1))
            if layer < 0:
                continue
            elapsed_ms = float(
                item.get("critical_ms", item.get("elapsed_ms", 0.0))
            )
            stage = str(item.get("stage", ""))
            current_layer = cuda_layer_totals.setdefault(
                layer,
                {
                    "layer": layer,
                    "attention_ms": 0.0,
                    "mlp_ms": 0.0,
                    "total_ms": 0.0,
                },
            )
            if stage.startswith("attention"):
                current_layer["attention_ms"] += elapsed_ms
            elif stage in ("dense", "moe"):
                current_layer["mlp_ms"] += elapsed_ms
            current_layer["total_ms"] += elapsed_ms
        cuda_top_layers = sorted(
            cuda_layer_totals.values(),
            key=lambda item: float(item["total_ms"]),
            reverse=True,
        )[:8]

        current = self._pool_profile_counters(self.pool)
        before = self._profile_pool_snapshot
        cache_delta: dict[str, float | int] = {}
        for name, value in current.items():
            previous = before.get(name, 0)
            cache_delta[name] = value - previous
        cache_delta["uploaded_gib"] = (
            float(cache_delta["uploaded_bytes"]) / 2**30
        )

        result: dict[str, object] = {
            "mode": (
                "tp_hidden_async_cuda_events"
                if self._tp_hidden_state_ready
                else "single_rank_cuda_events_and_layer_envelopes"
            ),
            "layer_count": len(layers) or len(cuda_layer_totals),
            "wall_totals_ms": wall_totals_ms,
            "top_layers": top_layers or cuda_top_layers,
            "cuda_top_layers": cuda_top_layers,
            "layers": layers,
            "cuda": dict(self.last_cuda_profile),
            "expert_cache_delta": cache_delta,
        }
        if self.device.type == "cpu":
            result["cpu_substages_ms"] = {
                name: seconds * 1000.0
                for name, seconds in self._cpu_substage_profile.items()
            }
            result["cpu_execution_image"] = {
                "compile_mode": str(
                    getattr(self.pool, "cpu_compile_mode", "off")
                ),
                "resident_bytes": int(
                    getattr(self.pool, "host_expert_bytes", 0)
                ),
                "source_index_bytes": int(
                    getattr(self.pool, "compiled_source_bytes", 0)
                ),
                "compiled_index_bytes": int(
                    getattr(self.pool, "compiled_index_bytes", 0)
                ),
                "expanded_index_bytes": int(
                    getattr(self.pool, "expanded_index_bytes", 0)
                ),
                "latent_moe_layers": len(
                    getattr(self, "_cpu_latent_moe_layers", {})
                ),
            }
        if self.device.type == "cpu":
            from .cpuext import (
                block_fp8_gemv_profile,
                latent_moe_phase_profile,
                packed_moe_phase_profile,
                three_projection_phase_profile,
            )

            result["block_fp8_gemv"] = block_fp8_gemv_profile()
            result["latent_resident_moe"] = latent_moe_phase_profile()
            result["block_fp8_layouts"] = (
                BlockFP8Weight.cpu_layout_decisions()
            )
            result["packed_moe"] = packed_moe_phase_profile()
            result["packed_three_projection"] = (
                three_projection_phase_profile()
            )
        mla_runners = getattr(
            getattr(self, "_tp_mla", None),
            "_paged_runners",
            None,
        )
        if mla_runners:
            result["mla_planner"] = {
                "gpu_plan_hits": sum(
                    int(getattr(runner, "gpu_plan_hits", 0))
                    for runner in mla_runners
                ),
                "gpu_plan_rejections": sum(
                    int(getattr(runner, "gpu_plan_rejections", 0))
                    for runner in mla_runners
                ),
                "cpu_plan_calls": sum(
                    int(getattr(runner, "cpu_plan_calls", 0))
                    for runner in mla_runners
                ),
                "layouts": [
                    int(runner._wrapper._plan_info[0])
                    for runner in mla_runners
                ],
            }
        self._profile_pool_snapshot = {}
        return result

    # ---- public model interface -------------------------------------------
    def embed(self, ids: list[int] | torch.Tensor) -> torch.Tensor:
        if self._tp_vocab is not None:
            return self._tp_vocab.embed(ids)
        weight = self.w(f"{_ROOT}.model.embed_tokens.weight")
        index = torch.as_tensor(ids, dtype=torch.long, device=weight.device)
        return F.embedding(
            index,
            weight,
        )

    def _tp_select(
        self,
        values: tuple[torch.Tensor, ...],
        devices: tuple[torch.device, ...],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            values[self.devices.index(device)] for device in devices
        )

    def _moe_hidden_no_owner(self, value, residual, layer: int):
        """True TP MoE: every rank owns state and computes every expert shard."""
        from .ops import TPHidden, route_topk

        trace_layer = int(
            os.environ.get("TPQ_TP_HIDDEN_TRACE_LAYER", "-1")
        )
        trace = layer == trace_layer
        trace_device = self.devices[0]
        moe_trace: dict[str, torch.Tensor] = {}
        profile = os.environ.get("TPQ_TP_HIDDEN_TIMING", "0") != "0"
        if profile:
            for timing_device in self.devices:
                torch.cuda.synchronize(timing_device)
            phase_started = time.perf_counter()

        def mark(name: str) -> None:
            nonlocal phase_started
            if not profile:
                return
            for timing_device in self.devices:
                torch.cuda.synchronize(timing_device)
            now = time.perf_counter()
            self._tp_no_owner_moe_timing[name] = (
                self._tp_no_owner_moe_timing.get(name, 0.0)
                + now
                - phase_started
            )
            phase_started = now

        if (
            self._tp_shared_mlp is None
            or self._tp_route_down is None
            or self._tp_routed_up is None
        ):
            raise RuntimeError("no-owner TP MoE operators are unavailable")
        fixed_plan = self._tp_no_owner_moe_plans.get(layer)
        if fixed_plan is not None and not trace and not profile:
            output = fixed_plan.launch(
                value.ready_events
                if self._tp_async_profile_active
                else None
            )
            self.pool.hits += self.config.top_k
            if os.environ.get("TPQ_ROUTE_HISTORY", "0") != "0":
                self._prev_ids[layer] = (
                    self._tp_route_all_rank_buffers[layer][1][0][0]
                    .tolist()
                )
            return output
        shared_partials = self._tp_shared_mlp.launch_partials(
            layer,
            value,
        )
        mark("shared")
        route_input = value
        if self._tp_mlp_layer_graph:
            normalized = self._tp_shared_mlp.input_hidden(layer)
            ready_by_device = {
                device: event
                for device, event in zip(
                    shared_partials.devices,
                    shared_partials.ready_events,
                )
            }
            route_input = TPHidden(
                normalized.devices,
                normalized.replicas,
                tuple(
                    ready_by_device[device]
                    for device in normalized.devices
                ),
            )
            residual = TPHidden(
                residual.devices,
                residual.replicas,
                tuple(
                    ready_by_device[device]
                    for device in residual.devices
                ),
            )
        if self._tp_moe_all_rank_layer_graph:
            logits, latent = (
                self._tp_route_down.reduce_hidden_from_events(
                    layer,
                    shared_partials.ready_events,
                )
            )
        else:
            logits, latent = self._tp_route_down.run_hidden(
                layer,
                route_input,
            )
        mark("router_down")
        if trace:
            moe_trace["value"] = (
                route_input.wait_on(trace_device).clone()
            )
            moe_trace["logits"] = (
                logits.wait_on(trace_device).clone()
            )
            moe_trace["latent"] = (
                latent.wait_on(trace_device).clone()
            )
        weight_buffers, index_buffers = (
            self._tp_route_all_rank_buffers[layer]
        )
        routes = [
            (weight_buffers[rank], index_buffers[rank])
            for rank in range(len(self.devices))
        ]
        if not self._tp_route_packed_graph:
            routes = []
            for rank, device in enumerate(self.devices):
                logits_rank = logits.devices.index(device)
                with torch.cuda.device(device):
                    torch.cuda.current_stream(device).wait_event(
                        logits.ready_events[logits_rank]
                    )
                    route = route_topk(
                        logits.replicas[logits_rank],
                        self._tp_route_corrections[layer][rank],
                        self._tp_route_masks[layer][rank],
                        scoring_func=self.config.scoring_func,
                        top_k=self.config.top_k,
                        normalize=self.config.norm_topk_prob,
                        scaling=self.config.routed_scaling,
                        n_group=self.config.n_group,
                        topk_group=self.config.topk_group,
                        output_buffers=(
                            weight_buffers[rank],
                            index_buffers[rank],
                        ),
                    )
                if route is None:
                    raise RuntimeError(
                        "no-owner TP route requires a registered local "
                        "Top-K operator"
                    )
                latent.ready_events[rank].record(
                    torch.cuda.current_stream(device)
                )
                routes.append(route)
        mark("topk")
        if trace and not self._tp_route_packed_graph:
            moe_trace["route_weights"] = routes[0][0].clone()
            moe_trace["route_indices"] = routes[0][1].clone()
        routed = self.pool.run_hidden(
            layer,
            latent,
            tuple(routes),
            activation=self.operator_config.expert_activation,
            activation_beta=self.config.situ_beta,
            activation_linear_beta=self.config.situ_linear_beta,
        )
        mark("packed")
        if trace:
            if self._tp_route_packed_graph:
                routed.wait_on(trace_device)
                moe_trace["route_weights"] = routes[0][0].clone()
                moe_trace["route_indices"] = routes[0][1].clone()
            moe_trace["packed"] = (
                routed.wait_on(trace_device).clone()
            )
        if (
            self.config.latent_moe_use_norm
            and self._tp_routed_finalize_graph
        ):
            routed_input = (
                self._tp_routed_up.composed_input_sharded(
                    layer,
                    routed,
                )
            )
        else:
            if self.config.latent_moe_use_norm:
                routed = routed.rmsnorm_to(
                    self._tp_routed_norm_weights[layer],
                    self.config.rms_eps,
                    self._tp_routed_norm_hidden[layer],
                )
            try:
                routed_input = (
                    self._tp_routed_up.bound_input_sharded(
                        layer,
                        routed,
                    )
                )
            except ValueError:
                routed_input = self._tp_routed_up.input_sharded(layer)
                routed_input.copy_from_replicated(routed)
        mark("routed_prepare")
        output = self._tp_routed_up.finalize_moe(
            layer,
            routed_input,
            shared_partials,
            residual,
        )
        mark("routed_up_finalize")
        if trace:
            routed_norm = (
                self._tp_routed_norm_hidden[layer]
                if self._tp_routed_finalize_graph
                and self.config.latent_moe_use_norm
                else routed
            )
            moe_trace["routed_norm"] = (
                routed_norm.wait_on(trace_device).clone()
            )
            moe_trace["output"] = (
                output.wait_on(trace_device).clone()
            )
            shared_sum = torch.zeros_like(
                moe_trace["output"],
                dtype=torch.float32,
            )
            for contribution in shared_partials.contributions:
                shared_sum.add_(
                    contribution.to(trace_device)
                )
            moe_trace["shared_sum"] = shared_sum
            self.last_tp_moe_trace = moe_trace
        if os.environ.get("TPQ_ROUTE_HISTORY", "0") != "0":
            self._prev_ids[layer] = routes[0][1][0].tolist()
        return output

    def _moe_hidden(self, value, residual, layer: int):
        """Run MoE from fixed replicated state with one hidden collective."""
        if getattr(self.pool, "hidden_mode", False):
            return self._moe_hidden_no_owner(value, residual, layer)
        from .ops import linear_route_topk

        if self._tp_shared_mlp is None or self._tp_routed_up is None:
            raise RuntimeError("TPHidden MoE operators are unavailable")
        device = self.layer_device(layer)
        shared_partials = self._tp_shared_mlp.launch_partials(
            layer,
            value,
        )
        if self._tp_mlp_layer_graph:
            from .ops import TPHidden

            normalized = self._tp_shared_mlp.input_hidden(layer)
            normalized_rank = normalized.devices.index(device)
            partial_rank = shared_partials.devices.index(device)
            with torch.cuda.device(device):
                torch.cuda.current_stream(device).wait_event(
                    shared_partials.ready_events[partial_rank]
                )
            owner_value = normalized.replicas[normalized_rank]
            owner_ready = shared_partials.ready_events[partial_rank]
            residual_events = {
                partial_device: event
                for partial_device, event in zip(
                    shared_partials.devices,
                    shared_partials.ready_events,
                )
            }
            residual = TPHidden(
                residual.devices,
                residual.replicas,
                tuple(
                    residual_events[residual_device]
                    for residual_device in residual.devices
                ),
            )
        else:
            owner_value = value.wait_on(device)
            if value.ready_events is None:
                raise RuntimeError(
                    "TPHidden MoE requires a ready input event"
                )
            owner_ready = value.ready_events[
                value.devices.index(device)
            ]
        if os.environ.get(
            "TPQ_TP_HIDDEN_SERIAL_SHARED",
            "0",
        ) != "0":
            for shared_device, shared_event in zip(
                shared_partials.devices,
                shared_partials.ready_events,
            ):
                with torch.cuda.device(shared_device):
                    torch.cuda.current_stream(
                        shared_device
                    ).wait_event(shared_event)
                    torch.cuda.synchronize(shared_device)
        prefix = (
            f"{_ROOT}.model.layers.{layer}.block_sparse_moe"
        )
        gate_weight = self.w(f"{prefix}.gate.weight")
        correction = self.w(
            f"{prefix}.gate.e_score_correction_bias"
        )
        available = self._mask(layer)
        buffers = self._route_buffers.get(layer)
        if buffers is None:
            buffers = (
                torch.empty(
                    1,
                    self.config.n_experts,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.empty(
                    1,
                    self.config.top_k,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.empty(
                    1,
                    self.config.top_k,
                    dtype=torch.long,
                    device=device,
                ),
            )
            self._route_buffers[layer] = buffers
        if self._tp_moe_owner_layer_graph:
            route, latent = self._tp_fixed_moe_prelude.result(layer)
        elif self._tp_fixed_moe_prelude is not None:
            route, latent = self._tp_fixed_moe_prelude.run(
                layer,
                owner_value,
                owner_ready,
            )
        else:
            # TPHidden deliberately has no permanent owner device.
            # Owner-local custom CUDA kernels must bind the layer owner;
            # relying on the process-global current device loses the true
            # producer-stream dependency after an owner boundary.
            with torch.cuda.device(device):
                route = linear_route_topk(
                    owner_value,
                    gate_weight,
                    correction,
                    available,
                    scoring_func=self.config.scoring_func,
                    top_k=self.config.top_k,
                    normalize=self.config.norm_topk_prob,
                    scaling=self.config.routed_scaling,
                    n_group=self.config.n_group,
                    topk_group=self.config.topk_group,
                    output_buffers=buffers,
                )
                if route is None:
                    route = route_experts(
                        owner_value,
                        gate_weight,
                        correction,
                        available,
                        top_k=self.config.top_k,
                        normalize=self.config.norm_topk_prob,
                        scaling=self.config.routed_scaling,
                        n_group=self.config.n_group,
                        topk_group=self.config.topk_group,
                    )
                latent = self._tp_routed_latent_buffers[layer]
                torch.mm(
                    owner_value,
                    self.w(
                        f"{prefix}.routed_expert_down_proj.weight"
                    ).t(),
                    out=latent,
                )
        trace_layer = int(
            os.environ.get("TPQ_TP_HIDDEN_TRACE_LAYER", "-1")
        )
        moe_trace: dict[str, torch.Tensor] = {}
        if layer == trace_layer:
            moe_trace["value"] = owner_value.clone()
            moe_trace["route_weights"] = route[0].clone()
            moe_trace["route_indices"] = route[1].clone()
        if layer == trace_layer:
            moe_trace["latent"] = latent.clone()
        weights, indices = route
        with torch.cuda.device(device):
            packed_result = self.pool.run(
                layer,
                latent,
                indices[0],
                weights[0],
                activation=self.operator_config.expert_activation,
                activation_beta=self.config.situ_beta,
                activation_linear_beta=self.config.situ_linear_beta,
            )
            routed = self._tp_routed_value_buffers[layer]
            routed.copy_(
                packed_result.view(1, self.config.routed_hidden)
            )
            if layer == trace_layer:
                moe_trace["packed"] = routed.clone()
            if self.config.latent_moe_use_norm:
                routed = self._rmsnorm(
                    routed,
                    self.w(
                        f"{prefix}.routed_expert_norm.weight"
                    ),
                    self.config.rms_eps,
                    output=self._tp_routed_norm_buffers[layer],
                )
        remember_route = os.environ.get(
            "TPQ_PREFETCH",
            "1" if getattr(self.pool, "prefetch_default", True) else "0",
        ) != "0"
        if getattr(self.pool, "device_routed", False) and remember_route:
            self._prev_ids[layer] = self.pool.last_expert_ids(layer)
        elif not getattr(self.pool, "device_routed", False):
            self._prev_ids[layer] = indices[0].tolist()
        if layer == trace_layer:
            moe_trace["routed_norm"] = routed.clone()
        output = self._tp_routed_up.finalize_moe_full(
            layer,
            routed,
            shared_partials,
            residual,
        )
        if layer == trace_layer:
            moe_trace["output"] = output.wait_on(device).clone()
            shared_sum = torch.zeros(
                shared_partials.shape,
                dtype=torch.float32,
                device=device,
            )
            for contribution in shared_partials.contributions:
                shared_sum.add_(contribution.to(device))
            moe_trace["shared_sum"] = shared_sum
            routed_sum = torch.zeros_like(shared_sum)
            for contribution in (
                self._tp_routed_up.last_partials(
                    layer
                ).contributions
            ):
                routed_sum.add_(contribution.to(device))
            moe_trace["routed_sum"] = routed_sum
            self.last_tp_moe_trace = moe_trace
        return output

    def _forward_token_hidden(self, token: int) -> torch.Tensor:
        """Decode while retaining the hidden state on every TP rank."""
        if (
            self._tp_token_hidden is None
            or self._tp_block_residual is None
            or self._tp_final_mix is None
        ):
            raise RuntimeError("TPHidden state was not prepared")
        hidden = self._tp_token_hidden
        embedding = self.embed([token])
        hidden.copy_from_owner(
            embedding,
            hidden.devices.index(embedding.device),
        )
        timing_enabled = (
            os.environ.get("TPQ_TP_HIDDEN_TIMING", "0") != "0"
        )
        cuda_event_profile = self._profile_enabled or (
            os.environ.get("TPQ_KIMI_CUDA_EVENTS", "0") != "0"
        )
        stage_profiler = None
        if cuda_event_profile:
            from .ops import TPHiddenStageProfiler

            stage_profiler = TPHiddenStageProfiler(True)
        self._tp_async_profile_active = cuda_event_profile
        timing_totals: dict[str, float] = {}

        def timing_mark(name: str, started: float) -> float:
            if not timing_enabled:
                return started
            for timing_device in self.devices:
                torch.cuda.synchronize(timing_device)
            now = time.perf_counter()
            timing_totals[name] = (
                timing_totals.get(name, 0.0) + now - started
            )
            return now

        if timing_enabled:
            self._tp_no_owner_moe_timing = {}
            for timing_device in self.devices:
                torch.cuda.synchronize(timing_device)
        residual = self._tp_block_residual
        residual.reset()
        position = self.pos
        kda_layers = set(self.config.kda_layers)
        trace_enabled = (
            os.environ.get("TPQ_TP_HIDDEN_TRACE", "0") != "0"
        )
        hidden_trace: list[torch.Tensor] = []
        trace_layer = int(
            os.environ.get("TPQ_TP_HIDDEN_TRACE_LAYER", "-1")
        )
        stage_trace: dict[str, torch.Tensor] = {}

        prefetch_default = (
            "1"
            if getattr(self.pool, "prefetch_default", True)
            else "0"
        )
        if (
            self._prev_ids
            and os.environ.get("TPQ_PREFETCH", prefetch_default) != "0"
        ):
            for layer, expert_ids in self._prev_ids.items():
                self.pool.prefetch(
                    [(layer, expert_id) for expert_id in expert_ids]
                )

        for layer in range(self.config.n_layers):
            timing_started = time.perf_counter()
            trace_device = self.layer_device(layer)
            if layer == trace_layer:
                stage_trace["hidden_in"] = (
                    hidden.wait_on(trace_device).clone()
                )
                stage_trace["residual_inverse_in"] = (
                    residual.inverses[
                        residual.devices.index(trace_device)
                    ][:residual.active_rows].clone()
                )
            attention_executor = (
                self._tp_kda
                if layer in kda_layers
                else self._tp_mla
            )
            attention_input = attention_executor.input_hidden(layer)
            attention_plan = self._tp_attention_mix[layer]
            if not self._tp_attention_layer_graph:
                if residual.active_rows:
                    hidden.residual_mix_to(
                        residual,
                        self._tp_select(
                            attention_plan[0],
                            attention_input.devices,
                        ),
                        self._tp_select(
                            attention_plan[1],
                            attention_input.devices,
                        ),
                        self.config.rms_eps,
                        attention_input,
                        post_norm_weights=self._tp_select(
                            attention_plan[2],
                            attention_input.devices,
                        ),
                        workspaces=self._tp_select(
                            self._tp_residual_workspaces,
                            attention_input.devices,
                        ),
                    )
                else:
                    hidden.rmsnorm_to(
                        self._tp_select(
                            attention_plan[2],
                            attention_input.devices,
                        ),
                        self.config.rms_eps,
                        attention_input,
                    )
            timing_started = timing_mark(
                "attention_prepare",
                timing_started,
            )
            if layer == trace_layer and not self._tp_attention_layer_graph:
                stage_trace["attention_input"] = (
                    attention_input.wait_on(trace_device).clone()
                )

            boundary = (
                layer % self.config.attn_res_block_size == 0
            )
            if boundary:
                residual.append(hidden)
            decode_plan = self._tp_decode_layer_plans.get(layer)
            if (
                decode_plan is not None
                and not trace_enabled
                and trace_layer < 0
                and not timing_enabled
                and stage_profiler is None
            ):
                hidden = decode_plan.launch(hidden, position)
                self.pool.hits += self.config.top_k
                if os.environ.get("TPQ_ROUTE_HISTORY", "0") != "0":
                    self._prev_ids[layer] = (
                        self._tp_route_all_rank_buffers[layer][1][0][0]
                        .tolist()
                    )
                if (
                    os.environ.get(
                        "TPQ_TP_HIDDEN_SYNC_LAYER",
                        "0",
                    )
                    != "0"
                ):
                    for tp_device in self.devices:
                        torch.cuda.synchronize(tp_device)
                continue
            attention_source = (
                hidden
                if self._tp_attention_layer_graph
                else attention_input
            )
            attention_profile = None
            if stage_profiler is not None:
                attention_source, attention_profile = (
                    stage_profiler.begin(
                        (
                            "attention_kda"
                            if layer in kda_layers
                            else "attention_mla"
                        ),
                        attention_source,
                        layer=layer,
                    )
                )
            attention = (
                attention_executor.run_hidden(
                    layer,
                    attention_source,
                )
                if layer in kda_layers
                else attention_executor.run_hidden(
                    layer,
                    attention_source,
                    position,
                )
            )
            if stage_profiler is not None:
                attention = stage_profiler.end(
                    attention_profile,
                    attention,
                )
            timing_started = timing_mark(
                "attention",
                timing_started,
            )
            if layer == trace_layer:
                if self._tp_attention_layer_graph:
                    stage_trace["attention_input"] = (
                        attention_input.wait_on(trace_device).clone()
                    )
                stage_trace["attention"] = (
                    attention.wait_on(trace_device).clone()
                )
            if self._tp_mlp_layer_graph:
                prefix_sum = self._tp_layer_prefix_hidden[layer]
            elif boundary:
                prefix_sum = attention
            else:
                prefix_sum = hidden.add_to(
                    attention,
                    self._tp_layer_prefix_hidden[layer],
                )
            timing_started = timing_mark(
                "attention_residual",
                timing_started,
            )
            if layer == trace_layer and not self._tp_mlp_layer_graph:
                stage_trace["prefix_sum"] = (
                    prefix_sum.wait_on(trace_device).clone()
                )

            mlp_executor = (
                self._tp_dense_mlp
                if layer < self.config.first_dense_layers
                else self._tp_shared_mlp
            )
            mlp_input = mlp_executor.input_hidden(layer)
            mlp_plan = self._tp_mlp_mix[layer]
            if not self._tp_mlp_layer_graph:
                prefix_sum.residual_mix_to(
                    residual,
                    self._tp_select(
                        mlp_plan[0],
                        mlp_input.devices,
                    ),
                    self._tp_select(
                        mlp_plan[1],
                        mlp_input.devices,
                    ),
                    self.config.rms_eps,
                    mlp_input,
                    post_norm_weights=self._tp_select(
                        mlp_plan[2],
                        mlp_input.devices,
                    ),
                    workspaces=self._tp_select(
                        self._tp_residual_workspaces,
                        mlp_input.devices,
                    ),
                )
            timing_started = timing_mark(
                "moe_prepare",
                timing_started,
            )
            if layer == trace_layer:
                if not self._tp_mlp_layer_graph:
                    stage_trace["mlp_input"] = (
                        mlp_input.wait_on(trace_device).clone()
                    )
                stage_trace["residual_inverse_out"] = (
                    residual.inverses[
                        residual.devices.index(trace_device)
                    ][:residual.active_rows].clone()
                )
            mlp_source = (
                attention
                if self._tp_mlp_layer_graph
                else mlp_input
            )
            mlp_profile = None
            if stage_profiler is not None:
                mlp_source, mlp_profile = stage_profiler.begin(
                    (
                        "dense"
                        if layer < self.config.first_dense_layers
                        else "moe"
                    ),
                    mlp_source,
                    layer=layer,
                )
            if layer < self.config.first_dense_layers:
                mlp = self._tp_dense_mlp.run_hidden(
                    layer,
                    mlp_source,
                )
                if self._tp_mlp_layer_graph:
                    from .ops import TPHidden

                    prefix_sum = TPHidden(
                        prefix_sum.devices,
                        prefix_sum.replicas,
                        mlp.ready_events,
                    )
                hidden = prefix_sum.add_to(
                    mlp,
                    self._tp_layer_output_hidden[layer],
                )
            else:
                hidden = self._moe_hidden(
                    mlp_source,
                    prefix_sum,
                    layer,
                )
            if stage_profiler is not None:
                hidden = stage_profiler.end(mlp_profile, hidden)
            timing_started = timing_mark(
                "dense_or_moe",
                timing_started,
            )
            if trace_enabled:
                hidden_trace.append(
                    hidden.wait_on(self.layer_device(layer)).clone()
                )
            if layer == trace_layer:
                if self._tp_mlp_layer_graph:
                    stage_trace["prefix_sum"] = (
                        prefix_sum.wait_on(trace_device).clone()
                    )
                    stage_trace["mlp_input"] = (
                        mlp_input.wait_on(trace_device).clone()
                    )
                stage_trace["hidden_out"] = (
                    hidden.wait_on(trace_device).clone()
                )
            if os.environ.get("TPQ_TP_HIDDEN_SYNC_LAYER", "0") != "0":
                for tp_device in self.devices:
                    torch.cuda.synchronize(tp_device)

        final_started = time.perf_counter()
        final_plan = self._tp_final_mix
        final_source = hidden
        final_profile = None
        if stage_profiler is not None:
            final_source, final_profile = stage_profiler.begin(
                "final_mix",
                hidden,
                layer=-1,
            )
        final_source.residual_mix_to(
            residual,
            final_plan[0],
            final_plan[1],
            self.config.rms_eps,
            self._tp_token_hidden,
            post_norm_weights=final_plan[2],
            workspaces=self._tp_residual_workspaces,
        )
        if stage_profiler is not None:
            stage_profiler.end(final_profile, self._tp_token_hidden)
        timing_mark("final_mix", final_started)
        self.pos += 1
        if trace_enabled:
            self.last_tp_hidden_trace = tuple(hidden_trace)
        if trace_layer >= 0:
            self.last_tp_hidden_stage_trace = stage_trace
        if timing_enabled:
            self.last_tp_hidden_timing = {
                name: {
                    "total_ms": seconds * 1000.0,
                    "ms_layer": (
                        seconds * 1000.0 / self.config.n_layers
                        if name != "final_mix"
                        else seconds * 1000.0
                    ),
                }
                for name, seconds in timing_totals.items()
            }
            if self._tp_no_owner_moe_timing:
                self.last_tp_hidden_timing["moe_detail"] = {
                    name: {
                        "total_ms": seconds * 1000.0,
                        "ms_layer": seconds * 1000.0 / (
                            self.config.n_layers
                            - self.config.first_dense_layers
                        ),
                    }
                    for name, seconds
                    in self._tp_no_owner_moe_timing.items()
                }
        self.last_layer_profile = []
        self.last_cuda_profile = (
            stage_profiler.result(self.devices)
            if stage_profiler is not None
            else {}
        )
        self._tp_async_profile_active = False
        return self._tp_token_hidden.wait_on(self.devices[-1])

    def _forward_token(self, token: int) -> torch.Tensor:
        if self._tp_hidden_state_ready:
            return self._forward_token_hidden(token)
        config = self.config
        profile = self._profile_enabled or (
            os.environ.get("TPQ_KIMI_LAYER_TIMING", "0") != "0"
        )
        cuda_event_profile = (
            (
                self._profile_enabled
                or os.environ.get("TPQ_KIMI_CUDA_EVENTS", "0") != "0"
            )
            and self.device.type == "cuda"
        )
        profile_print = (
            os.environ.get("TPQ_KIMI_LAYER_TIMING_PRINT", "0") != "0"
        )
        layer_profile: list[dict[str, float | int]] = []
        cuda_events: list[
            tuple[
                int,
                int,
                str,
                torch.cuda.Event,
                torch.cuda.Event,
            ]
        ] = []

        def event_pair(
            layer: int,
            device: torch.device,
            stage: str,
        ) -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
            if not cuda_event_profile:
                return None, None
            end = self._cuda_stage_start(layer, stage, device)
            return None, end

        def sync(device: torch.device) -> None:
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)

        hidden = self.embed([token])
        self._active_cuda_events = (
            cuda_events if cuda_event_profile else None
        )
        block_residual = hidden.new_empty(1, 0, config.hidden)
        residual_inverse_cache = (
            os.environ.get("TPQ_RESIDUAL_INVERSE_CACHE", "1") != "0"
        )
        block_residual_inverse = torch.empty(
            0,
            dtype=torch.float32,
            device=hidden.device,
        )
        position = self.pos
        kda_layers = set(config.kda_layers)

        prefetch_default = (
            "1"
            if getattr(self.pool, "prefetch_default", True)
            else "0"
        )
        if (
            self._prev_ids
            and os.environ.get("TPQ_PREFETCH", prefetch_default) != "0"
        ):
            for layer, expert_ids in self._prev_ids.items():
                self.pool.prefetch(
                    [(layer, expert_id) for expert_id in expert_ids]
                )

        for layer in range(config.n_layers):
            target = self.layer_device(layer)
            transfer_started = time.perf_counter() if profile else 0.0
            if hidden.device != target:
                hidden = hidden.to(target)
                block_residual = block_residual.to(target)
                block_residual_inverse = (
                    block_residual_inverse.to(target)
                )
                sync(target)
            transfer_seconds = (
                time.perf_counter() - transfer_started
                if profile
                else 0.0
            )
            context = (
                torch.cuda.device(target)
                if target.type == "cuda"
                else nullcontext()
            )
            with context:
                sync(target)
                layer_started = time.perf_counter() if profile else 0.0
                prefix = f"{_ROOT}.model.layers.{layer}"
                prefix_sum = hidden
                attention_input = None
                attention_buffer = self._attention_tp_input_buffer(
                    layer
                )
                if block_residual.shape[1]:
                    residual_event = self._cuda_stage_start(
                        layer,
                        "attention_residual_norm",
                        target,
                    )
                    attention_input = self._attention_residual(
                        prefix_sum,
                        block_residual,
                        self.w(
                            f"{prefix}.self_attention_res_proj.weight"
                        ),
                        self.w(
                            f"{prefix}.self_attention_res_norm.weight"
                        ),
                        config.rms_eps,
                        self.w(f"{prefix}.input_layernorm.weight"),
                        output=attention_buffer,
                        residual_inverse=(
                            block_residual_inverse
                            if residual_inverse_cache
                            else None
                        ),
                    )
                    self._cuda_stage_end(residual_event)
                if layer % config.attn_res_block_size == 0:
                    block_residual = torch.cat(
                        (block_residual, prefix_sum.unsqueeze(1)),
                        dim=1,
                    )
                    block_residual_inverse = torch.cat(
                        (
                            block_residual_inverse,
                            torch.zeros(
                                1,
                                dtype=torch.float32,
                                device=target,
                            ),
                        )
                    )
                    prefix_sum = None

                attention_started = (
                    time.perf_counter() if profile else 0.0
                )
                _attention_event_start, attention_event_end = event_pair(
                    layer,
                    target,
                    "attention",
                )
                if attention_input is None:
                    attention_input = self._rmsnorm(
                        hidden,
                        self.w(f"{prefix}.input_layernorm.weight"),
                        config.rms_eps,
                        output=attention_buffer,
                    )
                attention_prepared = (
                    attention_buffer is not None
                    and attention_input.data_ptr()
                    == attention_buffer.data_ptr()
                )
                attention = (
                    self._kda_attention(
                        attention_input,
                        layer,
                        prepared=attention_prepared,
                    )
                    if layer in kda_layers
                    else self._mla_attention(
                        attention_input,
                        layer,
                        position,
                        prepared=attention_prepared,
                    )
                )
                if attention_event_end is not None:
                    attention_event_end.record(
                        torch.cuda.current_stream(target)
                    )
                sync(target)
                attention_seconds = (
                    time.perf_counter() - attention_started
                    if profile
                    else 0.0
                )
                prefix_event = self._cuda_stage_start(
                    layer,
                    "attention_prefix_add",
                    target,
                )
                prefix_sum = (
                    attention
                    if prefix_sum is None
                    else prefix_sum + attention
                )
                self._cuda_stage_end(prefix_event)
                mlp_started = time.perf_counter() if profile else 0.0
                _mlp_event_start, mlp_event_end = event_pair(
                    layer,
                    target,
                    "mlp",
                )
                mlp_residual_event = self._cuda_stage_start(
                    layer,
                    "mlp_residual_norm",
                    target,
                )
                mlp_buffer = self._mlp_tp_input_buffer(layer)
                mlp_input = self._attention_residual(
                    prefix_sum,
                    block_residual,
                    self.w(f"{prefix}.mlp_res_proj.weight"),
                    self.w(f"{prefix}.mlp_res_norm.weight"),
                    config.rms_eps,
                    self.w(f"{prefix}.post_attention_layernorm.weight"),
                    output=mlp_buffer,
                    residual_inverse=(
                        block_residual_inverse
                        if residual_inverse_cache
                        else None
                    ),
                )
                self._cuda_stage_end(mlp_residual_event)
                mlp_prepared = (
                    mlp_buffer is not None
                    and mlp_input.data_ptr() == mlp_buffer.data_ptr()
                )
                expert_transfer_seconds = 0.0
                if layer < config.first_dense_layers:
                    mlp = self._dense_mlp(
                        mlp_input,
                        layer,
                        prepared=mlp_prepared,
                    )
                    hidden = prefix_sum + mlp
                else:
                    hidden = self._moe(
                        mlp_input,
                        layer,
                        residual=prefix_sum,
                        prepared=mlp_prepared,
                    )
                if layer >= config.first_dense_layers:
                    expert_transfer_seconds = float(
                        getattr(
                            self.pool,
                            "last_transfer_seconds",
                            0.0,
                        )
                    )
                if mlp_event_end is not None:
                    mlp_event_end.record(
                        torch.cuda.current_stream(target)
                    )
                sync(target)
                if profile:
                    item = {
                        "layer": layer,
                        "attention_kind": (
                            "kda" if layer in kda_layers else "mla"
                        ),
                        "mlp_kind": (
                            "dense"
                            if layer < config.first_dense_layers
                            else "moe"
                        ),
                        "device": int(target.index or 0)
                        if target.type == "cuda"
                        else -1,
                        "transfer_seconds": transfer_seconds,
                        "attention_seconds": attention_seconds,
                        "mlp_seconds": (
                            time.perf_counter() - mlp_started
                        ),
                        "expert_transfer_seconds": (
                            expert_transfer_seconds
                        ),
                        "layer_seconds": (
                            time.perf_counter() - layer_started
                        ),
                    }
                    layer_profile.append(item)
                    if profile_print:
                        print(
                            f"[tpq-kimi-profile] L{layer} "
                            f"attn={item['attention_seconds']:.4f}s "
                            f"mlp={item['mlp_seconds']:.4f}s "
                            f"total={item['layer_seconds']:.4f}s "
                            f"transfer={item['transfer_seconds']:.4f}s",
                            flush=True,
                        )

        target = self.devices[-1]
        if hidden.device != target:
            hidden = hidden.to(target)
            block_residual = block_residual.to(target)
            block_residual_inverse = block_residual_inverse.to(target)
        context = (
            torch.cuda.device(target)
            if target.type == "cuda"
            else nullcontext()
        )
        with context:
            final_residual_event = self._cuda_stage_start(
                -1,
                "final_residual",
                target,
            )
            hidden = self._attention_residual(
                hidden,
                block_residual,
                self.w(f"{_ROOT}.model.output_attn_res_proj.weight"),
                self.w(f"{_ROOT}.model.output_attn_res_norm.weight"),
                config.rms_eps,
                self.w(f"{_ROOT}.model.norm.weight"),
                residual_inverse=(
                    block_residual_inverse
                    if residual_inverse_cache
                    else None
                ),
            )
            self._cuda_stage_end(final_residual_event)
        self.pos += 1
        self.last_layer_profile = layer_profile
        if cuda_event_profile:
            for device in self.devices:
                torch.cuda.synchronize(device)
            event_items: list[dict[str, float | int | str]] = []
            totals = {
                "attention_ms": 0.0,
                "mlp_ms": 0.0,
                "kda_attention_ms": 0.0,
                "mla_attention_ms": 0.0,
            }
            for layer, device_index, stage, start, end in cuda_events:
                elapsed_ms = float(start.elapsed_time(end))
                event_items.append(
                    {
                        "layer": layer,
                        "device": device_index,
                        "stage": stage,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                totals.setdefault(f"{stage}_ms", 0.0)
                totals[f"{stage}_ms"] += elapsed_ms
                if stage == "attention":
                    attention_kind = (
                        "kda" if layer in kda_layers else "mla"
                    )
                    totals[f"{attention_kind}_attention_ms"] += elapsed_ms
            self.last_cuda_profile = {
                "items": event_items,
                "totals": totals,
            }
            self._active_cuda_events = None
        else:
            self.last_cuda_profile = {}
        return hidden

    def snapshot_decode_state(self) -> dict[str, object]:
        """Snapshot recurrent CPU state for one speculative verification.

        MLA storage is append-only and is rewound through ``pos``.  KDA and
        short-convolution state are mutable, so a fixed reusable snapshot is
        kept per model instead of allocating hundreds of MiB every round.
        """
        if self.device.type != "cpu" or self._tp_hidden_state_ready:
            raise RuntimeError("Kimi state snapshots currently require CPU TP1")
        saved = getattr(self, "_cpu_decode_snapshot", None)
        state_keys = tuple(sorted(self._kda_state))
        conv_keys = tuple(sorted(self._conv_state))
        valid = (
            isinstance(saved, dict)
            and saved.get("state_keys") == state_keys
            and saved.get("conv_keys") == conv_keys
        )
        if not valid:
            saved = {
                "state_keys": state_keys,
                "states": tuple(
                    self._kda_state[layer].clone() for layer in state_keys
                ),
                "conv_keys": conv_keys,
                "convs": tuple(
                    tuple(item.clone() for item in self._conv_state[layer])
                    for layer in conv_keys
                ),
            }
            self._cpu_decode_snapshot = saved
        else:
            for target, layer in zip(saved["states"], state_keys):
                target.copy_(self._kda_state[layer])
            for targets, layer in zip(saved["convs"], conv_keys):
                for target, source in zip(targets, self._conv_state[layer]):
                    target.copy_(source)
        saved["pos"] = int(self.pos)
        saved["prev_ids"] = {
            int(layer): list(ids) for layer, ids in self._prev_ids.items()
        }
        return saved

    def restore_decode_state(self, saved: dict[str, object]) -> None:
        """Restore a snapshot produced by :meth:`snapshot_decode_state`."""
        for source, layer in zip(saved["states"], saved["state_keys"]):
            self._kda_state[int(layer)].copy_(source)
        for sources, layer in zip(saved["convs"], saved["conv_keys"]):
            for target, source in zip(self._conv_state[int(layer)], sources):
                target.copy_(source)
        self.pos = int(saved["pos"])
        self._prev_ids = {
            int(layer): list(ids)
            for layer, ids in saved["prev_ids"].items()
        }
        self._paged_latent_prepared.clear()

    def _kda_attention_block_cpu(
        self,
        value: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """KDA prefill block: batch projections, ordered recurrent updates."""
        config = self.config
        prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
        combined_input = self._kda_input_proj.get(layer)
        if combined_input is None:
            return torch.cat(
                [self._kda_attention(row, layer) for row in value.split(1)],
                dim=0,
            )
        tokens = value.shape[0]
        projected = _linear(value, combined_input).split(
            (
                config.n_heads * config.head_dim,
                config.n_heads * config.head_dim,
                config.n_heads * config.head_dim,
                config.n_heads * config.head_dim,
                self._kda_gate_rank[layer],
                config.n_heads,
            ),
            dim=-1,
        )
        query, key, val, output_gate = (
            item.reshape(tokens, config.n_heads, config.head_dim)
            for item in projected[:4]
        )
        recurrent_gate = _linear(
            projected[4], self.w(f"{prefix}.f_b_proj.weight")
        ).view(tokens, config.n_heads, config.head_dim)
        beta = projected[5].reshape(tokens, config.n_heads).float()
        state, conv = self._kda_buffers(layer)
        normalized = torch.empty_like(query)
        from .ops import attention_step

        workspace = torch.empty(
            3 * config.n_heads * config.head_dim,
            dtype=torch.float32,
            device=value.device,
        )
        recurrent_output = torch.empty(
            config.n_heads,
            config.head_dim,
            dtype=query.dtype,
            device=value.device,
        )
        for token in range(tokens):
            q = query[token]
            k = key[token]
            v = val[token]
            conv_ok = False
            try:
                conv_ok = bool(attention_step(
                    "short_conv3",
                    "cpu",
                    query=q.reshape(-1),
                    key=k.reshape(-1),
                    value=v.reshape(-1),
                    states=conv,
                    weights=(
                        self.w(f"{prefix}.q_conv1d.weight"),
                        self.w(f"{prefix}.k_conv1d.weight"),
                        self.w(f"{prefix}.v_conv1d.weight"),
                    ),
                ))
            except (LookupError, RuntimeError):
                conv_ok = False
            if not conv_ok:
                q1, qs = short_conv_step(
                    q.reshape(-1), conv[0],
                    self.w(f"{prefix}.q_conv1d.weight"),
                )
                k1, ks = short_conv_step(
                    k.reshape(-1), conv[1],
                    self.w(f"{prefix}.k_conv1d.weight"),
                )
                v1, vs = short_conv_step(
                    v.reshape(-1), conv[2],
                    self.w(f"{prefix}.v_conv1d.weight"),
                )
                self._conv_state[layer] = conv = (qs, ks, vs)
                q.copy_(q1.view_as(q))
                k.copy_(k1.view_as(k))
                v.copy_(v1.view_as(v))
            output = None
            try:
                output = attention_step(
                    "kda_recurrent",
                    "cpu",
                    query=q,
                    key=k,
                    value=v,
                    gate=recurrent_gate[token],
                    beta=beta[token],
                    a_log=self.w(f"{prefix}.A_log").float(),
                    dt_bias=self.w(f"{prefix}.dt_bias").float(),
                    state=state,
                    workspace=workspace,
                    output=recurrent_output,
                    lower_bound=float(self.cfg.get("gate_lower_bound", -5.0)),
                    output_gate=output_gate[token],
                    norm_weight=self.w(f"{prefix}.o_norm.weight"),
                    norm_eps=config.rms_eps,
                )
            except (LookupError, RuntimeError):
                output = None
            if output is None:
                output = gated_rmsnorm(
                    kda_recurrent_step(
                        q,
                        k,
                        v,
                        recurrent_gate[token],
                        beta[token],
                        self.w(f"{prefix}.A_log"),
                        self.w(f"{prefix}.dt_bias"),
                        state,
                        lower_bound=float(
                            self.cfg.get("gate_lower_bound", -5.0)
                        ),
                    ),
                    output_gate[token],
                    self.w(f"{prefix}.o_norm.weight"),
                    config.rms_eps,
                )
            normalized[token].copy_(output)
        return _linear(
            normalized.reshape(tokens, -1),
            self.w(f"{prefix}.o_proj.weight"),
        )

    def _mla_attention_block_cpu(
        self,
        value: torch.Tensor,
        layer: int,
        position: int,
    ) -> torch.Tensor:
        """MLA block path with batched projections and ordered cache writes."""
        config = self.config
        prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
        combined_input = self._mla_input_proj.get(layer)
        if combined_input is None:
            return torch.cat(
                [
                    self._mla_attention(row, layer, position + token)
                    for token, row in enumerate(value.split(1))
                ],
                dim=0,
            )
        tokens = value.shape[0]
        query_source, compressed, output_gate = _linear(
            value, combined_input
        ).split(
            (
                config.q_lora_rank,
                config.kv_lora_rank + config.qk_rope_head_dim,
                config.n_heads * config.v_head_dim,
            ),
            dim=-1,
        )
        query = _linear(
            self._rmsnorm(
                query_source,
                self.w(f"{prefix}.q_a_layernorm.weight"),
                1e-6,
            ),
            self.w(f"{prefix}.q_b_proj.weight"),
        ).view(tokens, config.n_heads, -1)
        query_nope, query_rope = query.split(
            [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1
        )
        latent, key_rope = compressed.split(
            [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1
        )
        latent = self._rmsnorm(
            latent,
            self.w(f"{prefix}.kv_a_layernorm.weight"),
            1e-6,
        )
        latent_cache, rope_cache = self._mla_buffers(
            layer, position + tokens
        )
        key_absorb, value_absorb = self._absorbed_weights(layer)
        outputs = []
        scale = 1.0 / math.sqrt(
            config.qk_nope_head_dim + config.qk_rope_head_dim
        )
        for token in range(tokens):
            current = position + token
            latent_cache[current].copy_(latent[token])
            rope_cache[current].copy_(key_rope[token])
            history_latent = latent_cache[: current + 1]
            history_rope = rope_cache[: current + 1]
            absorbed_query = torch.bmm(
                query_nope[token, :, None, :], key_absorb
            )
            scores = (
                torch.matmul(absorbed_query, history_latent.t())
                + torch.matmul(
                    query_rope[token, :, None, :], history_rope.t()
                )
            ) * scale
            probabilities = scores.float().softmax(dim=-1).to(
                history_latent.dtype
            )
            context = torch.matmul(probabilities, history_latent)
            outputs.append(torch.bmm(
                context, value_absorb.transpose(1, 2)
            ).reshape(1, config.n_heads * config.v_head_dim))
        return _linear(
            torch.cat(outputs, dim=0) * output_gate.sigmoid(),
            self.w(f"{prefix}.o_proj.weight"),
        )

    def _moe_block_cpu(
        self,
        value: torch.Tensor,
        layer: int,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        """Batch dense MoE projections while experts stay compact Top-16."""
        config = self.config
        prefix = f"{_ROOT}.model.layers.{layer}.block_sparse_moe"
        grouped_input = self._moe_input_proj.get(layer)
        if grouped_input is None:
            return torch.cat(
                [
                    self._moe(
                        row,
                        layer,
                        residual=None if residual is None else residual[i:i + 1],
                    )
                    for i, row in enumerate(value.split(1))
                ],
                dim=0,
            )
        shared_intermediate = config.n_shared * config.moe_inter
        shared_gate, shared_up, latent = _linear(
            value, grouped_input
        ).split(
            (shared_intermediate, shared_intermediate, config.routed_hidden),
            dim=-1,
        )
        gate_weight = self.w(f"{prefix}.gate.weight")
        correction = self.w(f"{prefix}.gate.e_score_correction_bias")
        available = self._mask(layer)
        route = None
        if config.n_group == 1 and config.topk_group == 1:
            try:
                from .ops import route_topk

                route = route_topk(
                    F.linear(value.float(), gate_weight.float()),
                    correction,
                    available,
                    scoring_func=config.scoring_func,
                    top_k=config.top_k,
                    normalize=config.norm_topk_prob,
                    scaling=config.routed_scaling,
                    n_group=config.n_group,
                    topk_group=config.topk_group,
                )
            except (LookupError, RuntimeError):
                route = None
        if route is None:
            route = route_experts(
                value,
                gate_weight,
                correction,
                available,
                top_k=config.top_k,
                normalize=config.norm_topk_prob,
                scaling=config.routed_scaling,
                n_group=config.n_group,
                topk_group=config.topk_group,
            )
        weights, indices = route
        routed_rows = []
        native = getattr(self.pool, "run_native", None)
        for token in range(value.shape[0]):
            expert_ids = indices[token].tolist()
            self._prev_ids[layer] = expert_ids
            routed = (
                native(
                    layer,
                    latent[token:token + 1],
                    indices[token],
                    weights[token],
                    activation=self.operator_config.expert_activation,
                    activation_beta=config.situ_beta,
                    activation_linear_beta=config.situ_linear_beta,
                )
                if native is not None
                else None
            )
            if routed is None:
                selected = self.pool.get_many(
                    [(layer, expert_id) for expert_id in expert_ids]
                )
                routed = moe_mlp_grouped_mixed(
                    latent[token:token + 1],
                    [selected[(layer, expert_id)] for expert_id in expert_ids],
                    weights[token],
                    activation=self.operator_config.expert_activation,
                    situ_beta=config.situ_beta,
                    situ_linear_beta=config.situ_linear_beta,
                )
            routed_rows.append(routed.view(1, config.routed_hidden))
        routed = torch.cat(routed_rows, dim=0).to(value.dtype)
        if config.latent_moe_use_norm:
            routed = self._rmsnorm(
                routed,
                self.w(f"{prefix}.routed_expert_norm.weight"),
                config.rms_eps,
            )
        routed = _linear(
            routed, self.w(f"{prefix}.routed_expert_up_proj.weight")
        )
        shared_prefix = f"{prefix}.shared_experts"
        shared = _linear(
            activate_gate_up(
                shared_gate,
                shared_up,
                activation=self.operator_config.expert_activation,
                situ_beta=config.situ_beta,
                situ_linear_beta=config.situ_linear_beta,
            ),
            self.w(f"{shared_prefix}.down_proj.weight"),
        )
        result = routed + shared
        return result if residual is None else residual + result

    def forward_hidden_block_cpu(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Layer-major CPU block verification for 2..16 candidate tokens."""
        values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        if not 2 <= len(values) <= 16:
            raise ValueError("Kimi CPU block verification requires 2..16 tokens")
        if self.device.type != "cpu" or self._tp_hidden_state_ready:
            raise RuntimeError("Kimi CPU block verification requires CPU TP1")
        if self.pos + len(values) > self.max_ctx:
            raise RuntimeError(
                f"context exceeds max_ctx ({self.pos + len(values)} > {self.max_ctx})"
            )
        config = self.config
        tokens = len(values)
        hidden = self.embed(values)
        block_residual = hidden.new_empty(tokens, 0, config.hidden)
        position = self.pos
        kda_layers = set(config.kda_layers)
        for layer in range(config.n_layers):
            prefix = f"{_ROOT}.model.layers.{layer}"
            prefix_sum = hidden
            attention_input = None
            if block_residual.shape[1]:
                attention_input = self._attention_residual(
                    prefix_sum,
                    block_residual,
                    self.w(f"{prefix}.self_attention_res_proj.weight"),
                    self.w(f"{prefix}.self_attention_res_norm.weight"),
                    config.rms_eps,
                    self.w(f"{prefix}.input_layernorm.weight"),
                )
            if layer % config.attn_res_block_size == 0:
                block_residual = torch.cat(
                    (block_residual, prefix_sum.unsqueeze(1)), dim=1
                )
                prefix_sum = None
            if attention_input is None:
                attention_input = self._rmsnorm(
                    hidden,
                    self.w(f"{prefix}.input_layernorm.weight"),
                    config.rms_eps,
                )
            attention = (
                self._kda_attention_block_cpu(attention_input, layer)
                if layer in kda_layers
                else self._mla_attention_block_cpu(
                    attention_input, layer, position
                )
            )
            prefix_sum = attention if prefix_sum is None else prefix_sum + attention
            mlp_input = self._attention_residual(
                prefix_sum,
                block_residual,
                self.w(f"{prefix}.mlp_res_proj.weight"),
                self.w(f"{prefix}.mlp_res_norm.weight"),
                config.rms_eps,
                self.w(f"{prefix}.post_attention_layernorm.weight"),
            )
            if layer < config.first_dense_layers:
                hidden = prefix_sum + self._dense_mlp(mlp_input, layer)
            else:
                hidden = self._moe_block_cpu(mlp_input, layer, prefix_sum)
        hidden = self._attention_residual(
            hidden,
            block_residual,
            self.w(f"{_ROOT}.model.output_attn_res_proj.weight"),
            self.w(f"{_ROOT}.model.output_attn_res_norm.weight"),
            config.rms_eps,
            self.w(f"{_ROOT}.model.norm.weight"),
        )
        self.pos += tokens
        self.last_layer_profile = []
        self.last_cuda_profile = {}
        return hidden

    def forward_hidden(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        values = (
            ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        )
        if not values:
            raise ValueError("Kimi forward requires at least one token")
        if self.pos + len(values) > self.max_ctx:
            raise RuntimeError(
                f"上下文超限（{self.pos + len(values)} > {self.max_ctx}）"
            )
        return torch.cat(
            [self._forward_token(int(token)) for token in values],
            dim=0,
        )

    def logits_of(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._tp_vocab is not None:
            return self._tp_vocab.logits(hidden)
        weight = self.w(f"{_ROOT}.lm_head.weight")
        if (
            hidden.is_cuda
            and hidden.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
        ):
            return torch.mm(
                hidden,
                weight.t(),
                out_dtype=torch.float32,
            )
        return _linear(hidden.to(weight.dtype), weight).float()

    def forward(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(ids)
        return self.logits_of(hidden[-1:]).squeeze(0)


__all__ = ["KimiK3TPQModel"]
