"""Kimi K3 text inference runtime for CCCP expert archives.

The first production path is one CUDA device with source-native BF16 dense
weights resident on GPU and routed experts supplied by the existing TPQ
RAM/VRAM cache.  It deliberately shares ``CCCPStore`` and ``ExpertPool`` with
the established runtimes; model files remain read-only.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .cconfig import KimiK3Config
from .grouped import activate_gate_up, moe_mlp_grouped_mixed
from .kimi_ops import (
    attention_residual,
    gated_rmsnorm,
    kda_recurrent_step,
    rmsnorm,
    route_experts,
    short_conv_step,
)
from .precision import compute_dtype
from .store import CCCPStore, ExpertPool, PackedCpuExpertPool


_ROOT = "language_model"


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
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._cpu_threads = 0
        self._cpu_numa_interleaved = False
        if self.device.type == "cpu":
            from .cpuext import (
                configure_cpu_threads,
                configure_numa_interleave,
            )

            self._cpu_threads = configure_cpu_threads()
            self._cpu_numa_interleaved = configure_numa_interleave()
        self.store = CCCPStore(root)
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
                from .kimi_hybrid import KimiPackedHybridPool

                self.pool = KimiPackedHybridPool(
                    self.store,
                    vram_cache_gb,
                    device=self.device,
                    ram_gb=cache_gb,
                )
            else:
                self.pool = ExpertPool(
                    self.store,
                    vram_cache_gb if self.device.type != "cpu" else cache_gb,
                    device=device,
                    ram_gb=cache_gb if self.device.type != "cpu" else 0.0,
                )
        self._weights: dict[str, torch.Tensor] = {}
        self._kda_input_proj: dict[int, torch.Tensor] = {}
        self._kda_gate_rank: dict[int, int] = {}
        self._mla_input_proj: dict[int, torch.Tensor] = {}
        self._dense_gate_up: dict[int, torch.Tensor] = {}
        self._shared_gate_up: dict[int, torch.Tensor] = {}
        self._tp_dense_mlp = None
        self._tp_shared_mlp = None
        self._tp_kda = None
        self._tp_mla = None
        self._tp_routed_down = None
        self._tp_routed_up = None
        self._tp_router = None
        self._tp_route_down = None
        self._tp_moe_prelude = None
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
        self.last_cuda_profile: dict[str, object] = {}
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

    def w(self, name: str) -> torch.Tensor:
        cached = self._weights.get(name)
        if cached is not None:
            return cached
        value = self.store.get_dense(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Kimi dense tensor {name!r} is not source-native"
            )
        if (
            name.endswith(".block_sparse_moe.gate.weight")
            or name.endswith(
                ".block_sparse_moe.gate.e_score_correction_bias"
            )
        ):
            # Kimi's published router explicitly evaluates in FP32. Convert
            # once while loading instead of allocating a FP32 copy per token.
            value = value.float()
        target = self._weight_device(name)
        if target.type != "cpu":
            value = value.to(target)
        self._weights[name] = value
        return value

    @staticmethod
    def _language_weight(name: str) -> bool:
        return name.startswith(f"{_ROOT}.")

    def preload(self) -> None:
        if self.device.type == "cpu":
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
            return
        started = time.time()
        if self._pipeline_plan is not None:
            self.pool.allocate()
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
        for index, name in enumerate(names, 1):
            self.w(name)
            if index % 160 == 0:
                print(
                    f"[tpq-kimi] 预载 dense {index}/{len(names)}",
                    flush=True,
                )
        self._combine_dense_projections()
        self._prepare_tp_kda()
        self._prepare_tp_mla()
        self._prepare_tp_dense_mlp()
        self._prepare_tp_moe_prelude()
        self._prepare_tp_shared_mlp()
        self._prepare_tp_route_down()
        self._prepare_tp_routed_linear()
        self._prepare_tp_router()
        self._prepare_tp_hidden_state()
        self._assert_no_owner_tp_dataflow()
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
        if self._pipeline_plan is not None:
            self.pool.preload()
            if (
                getattr(self.pool, "hidden_mode", False)
                and hasattr(self.pool, "compose_route_topk")
            ):
                self.pool.compose_route_topk(
                    {
                        layer: self._tp_route_down.output_hidden(
                            layer
                        )[0]
                        for layer in range(
                            self.config.first_dense_layers,
                            self.config.n_layers,
                        )
                    },
                    self._tp_route_corrections,
                    self._tp_route_masks,
                    self._tp_route_all_rank_buffers,
                    scoring_func=self.config.scoring_func,
                    top_k=self.config.top_k,
                    normalize=self.config.norm_topk_prob,
                    scaling=self.config.routed_scaling,
                    n_group=self.config.n_group,
                    topk_group=self.config.topk_group,
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
        resident_all = self.pool.preload_all()
        if resident_all:
            self.pool.pin_host_resident()
        else:
            self.pool.preload_pinned()
        self.pool.build_gpu_arenas()

    def _assert_no_owner_tp_dataflow(self) -> None:
        """Reject transitional owner compute when formal no-owner TP is on."""
        if (
            self._pipeline_plan is None
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

    def _combine_dense_projections(self) -> None:
        """合并数学等价的 BF16 GEMV，减少 decode kernel 提交。"""
        for layer in self.config.kda_layers:
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
            values = [self._weights.pop(name) for name in names]
            self._kda_gate_rank[layer] = int(values[4].shape[0])
            self._kda_input_proj[layer] = torch.cat(values, dim=0)
            # The published tensor is FP32, while the reference operation
            # casts it to the activation dtype on every token.  Cache that
            # mathematically identical BF16 view once so the registered
            # gated-RMSNorm kernel is selected instead of a seven-kernel
            # eager fallback.
            norm_name = f"{prefix}.o_norm.weight"
            self._weights[norm_name] = self._weights[norm_name].to(
                torch.bfloat16
            )
        for layer in self.config.full_attn_layers:
            prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
            names = [
                f"{prefix}.{projection}.weight"
                for projection in (
                    "q_a_proj",
                    "kv_a_proj_with_mqa",
                    "g_proj",
                )
            ]
            values = [self._weights.pop(name) for name in names]
            self._mla_input_proj[layer] = torch.cat(values, dim=0)
        for layer in range(self.config.first_dense_layers):
            prefix = f"{_ROOT}.model.layers.{layer}.mlp"
            gate_name = f"{prefix}.gate_proj.weight"
            up_name = f"{prefix}.up_proj.weight"
            self._dense_gate_up[layer] = torch.cat(
                (
                    self._weights.pop(gate_name),
                    self._weights.pop(up_name),
                ),
                dim=0,
            )
        for layer in range(
            self.config.first_dense_layers,
            self.config.n_layers,
        ):
            prefix = (
                f"{_ROOT}.model.layers.{layer}."
                "block_sparse_moe.shared_experts"
            )
            gate_name = f"{prefix}.gate_proj.weight"
            up_name = f"{prefix}.up_proj.weight"
            self._shared_gate_up[layer] = torch.cat(
                (
                    self._weights.pop(gate_name),
                    self._weights.pop(up_name),
                ),
                dim=0,
            )

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
        if group_size <= 1:
            raise ValueError(
                "no usable small-op TP subgroup for visible devices"
            )
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
            prefix = (
                f"{_ROOT}.model.layers.{layer}."
                "block_sparse_moe.shared_experts"
            )
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._shared_gate_up.pop(layer),
                self._weights.pop(f"{prefix}.down_proj.weight"),
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
            prefix = (
                f"{_ROOT}.model.layers.{layer}."
                "block_sparse_moe"
            )
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._weights.pop(f"{prefix}.gate.weight"),
                self._weights.pop(
                    f"{prefix}.routed_expert_down_proj.weight"
                ),
                self._shared_gate_up.pop(layer),
                self._weights.pop(
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
                self._weights.pop(f"{prefix}.gate.weight"),
                self._weights.pop(
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

        first_weight = self._dense_gate_up[0]
        intermediate = first_weight.shape[0] // 2
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
            prefix = f"{_ROOT}.model.layers.{layer}.mlp"
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._dense_gate_up.pop(layer),
                self._weights.pop(f"{prefix}.down_proj.weight"),
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
                    self._weights.pop(
                        f"{prefix}.routed_expert_down_proj.weight"
                    ),
                )
            up.add_layer(
                layer,
                owner,
                self._weights.pop(
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
                self._weights.pop(f"{prefix}.gate.weight"),
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
            if self._kda_gate_rank[layer] != gate_rank:
                raise ValueError("KDA gate rank must be stable across layers")
            prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._kda_input_proj.pop(layer),
                self._weights.pop(f"{prefix}.f_b_proj.weight"),
                tuple(
                    self._weights.pop(
                        f"{prefix}.{name}_conv1d.weight"
                    )
                    for name in ("q", "k", "v")
                ),
                self._weights.pop(f"{prefix}.A_log"),
                self._weights.pop(f"{prefix}.dt_bias"),
                self._weights.pop(f"{prefix}.o_norm.weight"),
                self._weights.pop(f"{prefix}.o_proj.weight"),
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
            prefix = f"{_ROOT}.model.layers.{layer}.self_attn"
            key_absorb, value_absorb = self._absorbed_weights(layer)
            executor.add_layer(
                layer,
                self._pipeline_plan.owner_by_layer[layer],
                self._mla_input_proj.pop(layer),
                self._weights.pop(
                    f"{prefix}.q_a_layernorm.weight"
                ),
                self._weights.pop(f"{prefix}.q_b_proj.weight"),
                self._weights.pop(
                    f"{prefix}.kv_a_layernorm.weight"
                ),
                key_absorb,
                value_absorb,
                self._weights.pop(f"{prefix}.o_proj.weight"),
            )
            self._absorbed.pop(layer, None)
        executor.capture()
        self._tp_mla = executor
        print(
            "[tpq-kimi] 通用 MLA Head-TP Graph 完成："
            f"{len(self.config.full_attn_layers)} 层×"
            f"TP{self._tp_executor_width(executor)}",
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
            query = F.linear(
                value,
                self.w(f"{prefix}.q_proj.weight"),
            ).reshape(config.n_heads, config.head_dim)
            key = F.linear(
                value,
                self.w(f"{prefix}.k_proj.weight"),
            ).reshape(config.n_heads, config.head_dim)
            val = F.linear(
                value,
                self.w(f"{prefix}.v_proj.weight"),
            ).reshape(config.n_heads, config.head_dim)
            output_gate = F.linear(
                value,
                self.w(f"{prefix}.g_proj.weight"),
            ).view(config.n_heads, config.head_dim)
            low_rank_gate = F.linear(
                value,
                self.w(f"{prefix}.f_a_proj.weight"),
            )
            beta = F.linear(
                value,
                self.w(f"{prefix}.b_proj.weight"),
            ).reshape(config.n_heads).float()
        else:
            projected = F.linear(
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
        recurrent_gate = F.linear(
            low_rank_gate,
            self.w(f"{prefix}.f_b_proj.weight"),
        ).view(config.n_heads, config.head_dim)

        output = None
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

        normalized = None
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
        return F.linear(
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
        weight = self.w(source_name).view(
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
            query_source = F.linear(
                value,
                self.w(f"{prefix}.q_a_proj.weight"),
            )
            compressed = F.linear(
                value,
                self.w(f"{prefix}.kv_a_proj_with_mqa.weight"),
            )
            output_gate = F.linear(
                value,
                self.w(f"{prefix}.g_proj.weight"),
            ).sigmoid()
        else:
            query_source, compressed, output_gate = F.linear(
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
        query = F.linear(
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
        return F.linear(
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
            gate = F.linear(
                value,
                self.w(f"{prefix}.gate_proj.weight"),
            )
            up = F.linear(
                value,
                self.w(f"{prefix}.up_proj.weight"),
            )
        else:
            gate, up = F.linear(
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
        return F.linear(
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
        prefix = f"{_ROOT}.model.layers.{layer}.block_sparse_moe"
        prelude_logits = None
        prelude_latent = None
        prelude_shared = None
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
            if (
                device.type != "cpu"
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
                        logits = F.linear(value.float(), gate_weight)
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
        route = compute_route()
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
            latent = F.linear(
                value,
                self.w(f"{prefix}.routed_expert_down_proj.weight"),
            )
        self._cuda_stage_end(down_event)
        weights, indices = route
        expert_event = self._cuda_stage_start(
            layer,
            "moe_packed_experts",
            device,
        )
        if (
            self._pipeline_plan is not None
            or getattr(self.pool, "device_routed", False)
        ):
            routed = self.pool.run(
                layer,
                latent,
                indices[0],
                weights[0],
                activation=self.operator_config.expert_activation,
                activation_beta=config.situ_beta,
                activation_linear_beta=config.situ_linear_beta,
            ).view(1, config.routed_hidden).to(value.dtype)
            if getattr(self.pool, "device_routed", False):
                self._prev_ids[layer] = self.pool.last_expert_ids(layer)
        else:
            expert_ids = indices[0].tolist()
            self._prev_ids[layer] = expert_ids
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
            ).view(1, config.routed_hidden).to(value.dtype)
        self._cuda_stage_end(expert_event)
        routed_up_event = self._cuda_stage_start(
            layer,
            "moe_routed_up",
            device,
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
            routed = F.linear(
                routed,
                self.w(f"{prefix}.routed_expert_up_proj.weight"),
            )
        self._cuda_stage_end(routed_up_event)
        shared_event = self._cuda_stage_start(
            layer,
            "moe_shared",
            device,
        )
        shared_prefix = f"{prefix}.shared_experts"
        if prelude_shared is not None:
            shared = prelude_shared.to(value.dtype)
        elif shared_pending is not None:
            shared = self._tp_shared_mlp.finish(
                layer,
                shared_pending,
            ).to(value.dtype)
        else:
            combined_gate_up = self._shared_gate_up.get(layer)
            if combined_gate_up is None:
                shared_gate = F.linear(
                    value,
                    self.w(f"{shared_prefix}.gate_proj.weight"),
                )
                shared_up = F.linear(
                    value,
                    self.w(f"{shared_prefix}.up_proj.weight"),
                )
            else:
                shared_gate, shared_up = F.linear(
                    value,
                    combined_gate_up,
                ).chunk(2, dim=-1)
            shared = F.linear(
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

    # ---- public model interface -------------------------------------------
    def embed(self, ids: list[int] | torch.Tensor) -> torch.Tensor:
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
        if getattr(self.pool, "device_routed", False):
            self._prev_ids[layer] = self.pool.last_expert_ids(layer)
        else:
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
        cuda_event_profile = (
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
        profile = os.environ.get("TPQ_KIMI_LAYER_TIMING", "0") != "0"
        cuda_event_profile = (
            os.environ.get("TPQ_KIMI_CUDA_EVENTS", "0") != "0"
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
        return F.linear(hidden.to(weight.dtype), weight).float()

    def forward(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(ids)
        return self.logits_of(hidden[-1:]).squeeze(0)


__all__ = ["KimiK3TPQModel"]
