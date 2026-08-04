"""Packed full-GPU experts and contiguous layer placement.

Projection archives store mixed indices at their real 9/10/12/14-bit width.
Expanding them to uint16 would defeat compact residency. This module keeps
the byte-exact payload in one stable arena per
pipeline rank and publishes CUDA pointer metadata for direct packed GEMV.
The historical filename remains for Kimi import compatibility; the public
pool is also used by projection-VQ DeepSeek with a configuration-driven plan.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass

import torch


_ROOT = "language_model"


@dataclass(frozen=True)
class KimiLayerPlan:
    ranges: tuple[tuple[int, int], ...]
    owner_by_layer: tuple[int, ...]
    bytes_by_rank: tuple[int, ...]
    dense_bytes_by_rank: tuple[int, ...]
    expert_bytes_by_rank: tuple[int, ...]
    expert_payload_by_layer: tuple[int, ...]
    expert_payload_by_expert: tuple[tuple[int, ...], ...]
    expert_aux_by_layer: tuple[int, ...]

    @property
    def tp_size(self) -> int:
        return len(self.ranges)


def _contiguous_minimax(
    layer_bytes: list[int],
    ranks: int,
    first_extra: int,
    last_extra: int,
) -> list[tuple[int, int]]:
    """Exact contiguous minimax partition with endpoint-only tensors."""
    count = len(layer_bytes)
    if not 1 <= ranks <= count:
        raise ValueError(f"cannot split {count} layers over {ranks} ranks")
    prefix = [0]
    for value in layer_bytes:
        prefix.append(prefix[-1] + int(value))
    infinity = 1 << 100
    cost = [[infinity] * (count + 1) for _ in range(ranks + 1)]
    previous = [[-1] * (count + 1) for _ in range(ranks + 1)]
    cost[0][0] = 0
    for rank in range(1, ranks + 1):
        for end in range(rank, count + 1):
            endpoint = first_extra if rank == 1 else 0
            if rank == ranks and end == count:
                endpoint += last_extra
            for start in range(rank - 1, end):
                if cost[rank - 1][start] == infinity:
                    continue
                current = prefix[end] - prefix[start] + endpoint
                candidate = max(cost[rank - 1][start], current)
                if candidate < cost[rank][end]:
                    cost[rank][end] = candidate
                    previous[rank][end] = start
    ranges: list[tuple[int, int]] = []
    end = count
    for rank in range(ranks, 0, -1):
        start = previous[rank][end]
        if start < 0:
            raise RuntimeError("failed to construct Kimi layer partition")
        ranges.append((start, end))
        end = start
    ranges.reverse()
    return ranges


def build_kimi_layer_plan(store, tp_size: int) -> KimiLayerPlan:
    """Balance packed expert and dense payload while preserving layer order."""
    n_layers = int(store.cfg["n_layers"])
    dense_by_layer = [0] * n_layers
    first_extra = 0
    last_extra = 0
    marker = ".model.layers."
    for name in store.dense_names():
        if not name.startswith(f"{_ROOT}."):
            continue
        resident_size = getattr(
            store, "dense_resident_nbytes", store.dense_nbytes
        )
        size = resident_size(name)
        if marker in name:
            layer = int(name.split(marker, 1)[1].split(".", 1)[0])
            dense_by_layer[layer] += size
        elif ".model.embed_tokens." in name:
            first_extra += size
        else:
            last_extra += size

    expert_file_by_layer = [0] * n_layers
    expert_payload_by_layer = [0] * n_layers
    expert_payload_by_expert: list[tuple[int, ...]] = [
        tuple() for _ in range(n_layers)
    ]
    expert_aux_by_layer = [0] * n_layers
    for layer, filename in store.man.expert_files.items():
        if not 0 <= int(layer) < n_layers:
            continue
        native_residency = getattr(store, "packed_expert_residency", None)
        if callable(native_residency):
            file_nbytes, payloads, aux_nbytes = native_residency(layer)
            expert_file_by_layer[layer] = int(file_nbytes)
            expert_payload_by_expert[layer] = tuple(
                int(value) for value in payloads
            )
            expert_payload_by_layer[layer] = sum(payloads)
            expert_aux_by_layer[layer] = int(aux_nbytes)
            continue
        expert_file_by_layer[layer] = os.path.getsize(
            os.path.join(store.root, filename)
        )
        audit_name = store.man.expert_audit_files.get(layer)
        if audit_name is None:
            raise ValueError(
                f"packed residency requires expert audit for layer {layer}"
            )
        with open(
            os.path.join(store.root, audit_name),
            "r",
            encoding="utf-8",
        ) as handle:
            audit = json.load(handle)
        experts = audit.get("experts", {})
        maximum = max(
            (
                int(str(expert_id).lstrip("e"))
                for expert_id in experts
            ),
            default=-1,
        )
        n_experts = int(store.cfg.get("n_experts", maximum + 1))
        payloads = [0] * n_experts
        for expert_id, item in experts.items():
            index = int(str(expert_id).lstrip("e"))
            if "gu_bytes" in item or "down_bytes" in item:
                payloads[index] = (
                    int(item.get("gu_bytes", 0))
                    + int(item.get("down_bytes", 0))
                )
            else:
                payloads[index] = sum(
                    int(projection.get("packed_bytes", 0))
                    for projection in item.values()
                    if isinstance(projection, dict)
                )
        expert_payload_by_expert[layer] = tuple(payloads)
        expert_payload_by_layer[layer] = sum(payloads)
        expert_aux_by_layer[layer] = max(
            0,
            expert_file_by_layer[layer] - sum(payloads),
        )

    layer_bytes = [
        dense + expert
        for dense, expert in zip(dense_by_layer, expert_file_by_layer)
    ]
    ranges = _contiguous_minimax(
        layer_bytes,
        int(tp_size),
        first_extra,
        last_extra,
    )
    owner = [0] * n_layers
    bytes_by_rank = []
    dense_bytes_by_rank = []
    expert_bytes_by_rank = []
    for rank, (start, end) in enumerate(ranges):
        for layer in range(start, end):
            owner[layer] = rank
        dense = sum(dense_by_layer[start:end])
        expert = sum(expert_file_by_layer[start:end])
        total = dense + expert
        if rank == 0:
            dense += first_extra
            total += first_extra
        if rank == len(ranges) - 1:
            dense += last_extra
            total += last_extra
        bytes_by_rank.append(total)
        dense_bytes_by_rank.append(dense)
        expert_bytes_by_rank.append(expert)
    return KimiLayerPlan(
        ranges=tuple(ranges),
        owner_by_layer=tuple(owner),
        bytes_by_rank=tuple(bytes_by_rank),
        dense_bytes_by_rank=tuple(dense_bytes_by_rank),
        expert_bytes_by_rank=tuple(expert_bytes_by_rank),
        expert_payload_by_layer=tuple(expert_payload_by_layer),
        expert_payload_by_expert=tuple(expert_payload_by_expert),
        expert_aux_by_layer=tuple(expert_aux_by_layer),
    )


def build_primary_dense_packed_plan(store, tp_size: int) -> KimiLayerPlan:
    """Build a tensor-sharded expert plan for a primary-device dense graph.

    This is the transition plan used by architectures whose Attention/Dense
    graph is not yet represented as ``TPHidden``.  Every routed expert is
    still column/row sharded across all ranks and no rank stores a complete
    expert.  Only the comparatively small non-expert graph remains on rank 0.
    """
    base = build_kimi_layer_plan(store, int(tp_size))
    n_layers = int(store.cfg["n_layers"])
    dense_total = sum(
        int(
            getattr(
                store,
                "dense_resident_nbytes",
                store.dense_nbytes,
            )(name)
        )
        for name in store.dense_names()
    )
    ranges = ((0, n_layers),) + tuple(
        (n_layers, n_layers) for _ in range(int(tp_size) - 1)
    )
    dense_by_rank = (dense_total,) + (0,) * (int(tp_size) - 1)
    return KimiLayerPlan(
        ranges=ranges,
        owner_by_layer=(0,) * n_layers,
        bytes_by_rank=tuple(
            dense_by_rank[rank] + base.expert_bytes_by_rank[rank]
            for rank in range(int(tp_size))
        ),
        dense_bytes_by_rank=dense_by_rank,
        expert_bytes_by_rank=base.expert_bytes_by_rank,
        expert_payload_by_layer=base.expert_payload_by_layer,
        expert_payload_by_expert=base.expert_payload_by_expert,
        expert_aux_by_layer=base.expert_aux_by_layer,
    )


class PackedExpertPool:
    """配置驱动的全显存 packed 专家执行器。"""

    full_resident = True

    def __init__(
        self,
        store,
        devices: tuple[torch.device, ...],
        plan: KimiLayerPlan,
        *,
        parallelism: str = "pipeline",
        tensor_group_size: int | None = None,
    ):
        self.store = store
        self.devices = devices
        self.plan = plan
        if parallelism not in {
            "pipeline",
            "expert",
            "tensor",
            "hybrid",
        }:
            raise ValueError(
                f"unsupported packed parallelism {parallelism!r}"
            )
        self.parallelism = parallelism
        self.hidden_mode = (
            parallelism == "tensor"
            and os.environ.get("TPQ_TP_HIDDEN", "0") != "0"
            and os.environ.get("TPQ_TP_NO_OWNER", "1") != "0"
        )
        self._peer_access = all(
            left == right
            or torch.cuda.can_device_access_peer(
                int(devices[left].index),
                int(devices[right].index),
            )
            for left in range(len(devices))
            for right in range(len(devices))
        )
        if parallelism == "tensor":
            tensor_group_size = len(devices)
        elif parallelism == "hybrid":
            tensor_group_size = (
                2 if tensor_group_size is None else int(tensor_group_size)
            )
        else:
            tensor_group_size = 1
        if (
            tensor_group_size <= 0
            or len(devices) % tensor_group_size
            or (
                parallelism == "hybrid"
                and tensor_group_size == len(devices)
            )
        ):
            raise ValueError(
                "packed MoE tensor group must be a proper divisor "
                "of the device count"
            )
        self.tensor_group_size = tensor_group_size
        self.expert_group_count = len(devices) // tensor_group_size
        self.budget = sum(plan.expert_bytes_by_rank)
        self.hits = 0
        self.miss = 0
        self.active = False
        self._arenas: list[torch.Tensor] = []
        self._metadata: dict[int, tuple[torch.Tensor, ...]] = {}
        self._codebooks: dict[
            tuple[int, str, str, int], torch.Tensor
        ] = {}
        self._workspaces: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._streams: list[torch.cuda.Stream] = []
        self._source_events: list[torch.cuda.Event] = []
        self._done_events: list[list[torch.cuda.Event]] = []
        self._prefill_ready_events: list[list[torch.cuda.Event]] = []
        self._output_events: list[list[torch.cuda.Event]] = []
        self._routed_inputs: list[torch.Tensor] = []
        self._routed_ids: list[torch.Tensor] = []
        self._routed_weights: list[torch.Tensor] = []
        self._source_inputs: list[torch.Tensor] = []
        self._source_ids: list[torch.Tensor] = []
        self._source_weights: list[torch.Tensor] = []
        self._return_buffers: list[torch.Tensor] = []
        self._reduce_buffers: list[torch.Tensor] = []
        self._zero_buffers: list[torch.Tensor] = []
        self._graphs: dict[int, list[torch.cuda.CUDAGraph]] = {}
        self._graph_batches: dict[int, object] = {}
        self._graph_rank_order: dict[int, tuple[int, ...]] = {}
        self._prefill_graphs: dict[
            int, list[torch.cuda.CUDAGraph]
        ] = {}
        self._prefill_graph_batches: dict[int, object] = {}
        self._prefill_graph_rank_order: dict[int, tuple[int, ...]] = {}
        self._prefill_metadata_directories: dict[int, tuple] = {}
        self._prefill_selected_metadata: dict[int, tuple] = {}
        self._prefill_slot_ids: dict[int, tuple] = {}
        self._prefill_hit_masks: dict[int, tuple] = {}
        self._output_replicas: dict[int, list[torch.Tensor]] = {}
        self._route_graphs: dict[
            int, tuple[torch.cuda.CUDAGraph, ...]
        ] = {}
        self._bound_hidden_inputs: dict[int, tuple] = {}
        if self.parallelism == "pipeline":
            self._rank_payload_bytes = tuple(
                sum(
                    plan.expert_payload_by_layer[layer]
                    for layer in range(start, end)
                )
                for start, end in plan.ranges
            )
        elif self.parallelism == "expert":
            rank_bytes = [0] * len(devices)
            for layer, payloads in enumerate(
                plan.expert_payload_by_expert
            ):
                for expert_id, payload in enumerate(payloads):
                    rank_bytes[self.expert_owner(layer, expert_id)] += payload
            self._rank_payload_bytes = tuple(rank_bytes)
        elif self.parallelism == "hybrid":
            rank_bytes = [0] * len(devices)
            for layer, payloads in enumerate(
                plan.expert_payload_by_expert
            ):
                for expert_id, payload in enumerate(payloads):
                    if payload % self.tensor_group_size:
                        raise ValueError(
                            "packed expert payload is not group-TP divisible"
                        )
                    for rank in self.expert_ranks(layer, expert_id):
                        rank_bytes[rank] += (
                            payload // self.tensor_group_size
                        )
            self._rank_payload_bytes = tuple(rank_bytes)
        else:
            rank_bytes = [0] * len(devices)
            for payloads in plan.expert_payload_by_expert:
                for payload in payloads:
                    if payload % len(devices):
                        raise ValueError(
                            "packed expert payload is not TP divisible"
                        )
                    for rank in range(len(devices)):
                        rank_bytes[rank] += payload // len(devices)
            self._rank_payload_bytes = tuple(rank_bytes)

    def expert_owner(self, layer: int, expert_id: int) -> int:
        if self.parallelism == "pipeline":
            return self.plan.owner_by_layer[layer]
        if self.parallelism in {"tensor", "hybrid"}:
            raise RuntimeError("tensor-sharded experts have no single owner")
        return (int(layer) + int(expert_id)) % len(self.devices)

    def expert_ranks(
        self,
        layer: int,
        expert_id: int,
    ) -> range:
        """Return the contiguous TP group assigned to one routed expert."""
        if self.parallelism == "tensor":
            return range(len(self.devices))
        if self.parallelism != "hybrid":
            owner = self.expert_owner(layer, expert_id)
            return range(owner, owner + 1)
        group = (
            int(layer) + int(expert_id)
        ) % self.expert_group_count
        start = group * self.tensor_group_size
        return range(start, start + self.tensor_group_size)

    @property
    def gpu_storage_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self._arenas) + sum(
            tensor.nbytes for tensor in self._codebooks.values()
        )

    @property
    def gpu_arena_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self._arenas)

    @property
    def host_expert_bytes(self) -> int:
        return 0

    def output_hidden(self, layer: int):
        """Expose fixed all-rank packed outputs for parent-graph composition."""
        from .ops import TPHidden

        outputs = self._output_replicas.get(int(layer))
        if outputs is None:
            raise RuntimeError(
                f"packed MoE layer {layer} outputs are unavailable"
            )
        return TPHidden(
            self.devices,
            tuple(outputs),
            tuple(
                self._output_events[rank][int(layer)]
                for rank in range(len(self.devices))
            ),
        )

    def fixed_layer_plan(self, layer: int):
        """Expose immutable packed-TP scheduling metadata to a common plan."""
        layer = int(layer)
        graph_batch = self._graph_batches.get(layer)
        if graph_batch is None:
            raise RuntimeError(
                f"packed MoE layer {layer} graph is unavailable"
            )
        return (
            graph_batch,
            tuple(
                self._workspaces[rank][2]
                for rank in range(len(self.devices))
            ),
            self.output_hidden(layer),
        )

    def prefetch(self, _keys) -> None:
        return

    def bind_hidden_inputs(
        self,
        layer: int,
        value,
        weights: tuple[torch.Tensor, ...],
        indices: tuple[torch.Tensor, ...],
    ) -> None:
        """Bind fixed all-rank Router/Down outputs to packed expert graphs."""
        if (
            not self.hidden_mode
            or tuple(value.devices) != self.devices
            or value.ready_events is None
            or len(weights) != len(self.devices)
            or len(indices) != len(self.devices)
        ):
            raise ValueError("packed MoE fixed input layout mismatch")
        self._bound_hidden_inputs[int(layer)] = (
            tuple(value.replicas),
            tuple(item.reshape(-1) for item in weights),
            tuple(item.reshape(-1) for item in indices),
        )

    def allocate(self) -> None:
        """Reserve packed arenas before fragmented dense allocations begin."""
        if self._arenas:
            return
        reserve = float(os.environ.get("TPQ_VRAM_RUNTIME_GB", "3.0"))
        reserve_bytes = int(reserve * 2**30)
        details = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                free, _total = torch.cuda.mem_get_info(device)
            # ``bytes_by_rank`` includes dense tensors and FP32 codebooks.  Add
            # a conservative 512 MiB for KDA state, router FP32 promotion and
            # decode workspaces on each rank.
            if self.parallelism == "pipeline":
                required = self._rank_payload_bytes[rank]
            else:
                # Capacity is checked after the parent model has already
                # materialized Dense tensors.  Count only allocations that
                # this method will add: expert payload and local codebooks.
                required = (
                    self._rank_payload_bytes[rank]
                    + sum(self.plan.expert_aux_by_layer)
                )
            required += 512 * 2**20
            available = max(0, free - reserve_bytes)
            details.append(
                f"cuda:{device.index} 需{required / 2**30:.2f}GiB/"
                f"可用{available / 2**30:.2f}GiB"
            )
            if required > available:
                raise RuntimeError(
                    "packed TP 全显存容量不足：" + "，".join(details)
                )
        try:
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    self._arenas.append(
                        torch.empty(
                            self._rank_payload_bytes[rank],
                            dtype=torch.uint8,
                            device=device,
                        )
                    )
        except Exception:
            self._arenas.clear()
            gc.collect()
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            raise
        print(
            "[tpq-packed] packed 专家 arena 已分配："
            + "，".join(
                f"cuda:{device.index}={size / 2**30:.2f}GiB"
                for device, size in zip(
                    self.devices,
                    self._rank_payload_bytes,
                )
            ),
            flush=True,
        )

    def _device_codebook(
        self,
        rank: int,
        layer: int,
        tier: str,
        variant: str,
        projection: int,
        cb: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        key = (rank, tier, variant, projection)
        cached = self._codebooks.get(key)
        if cached is None:
            cached = cb.to(
                device=device,
                dtype=torch.bfloat16,
            ).contiguous()
            self._codebooks[key] = cached
        return cached

    @staticmethod
    def _tensor_shard_raw(
        weight,
        *,
        projection: int,
        rank: int,
        ranks: int,
        intermediate: int,
    ) -> tuple[torch.Tensor, int]:
        """Slice packed rows/columns without expanding their bit width."""
        row_bits = weight.blocks * weight.bits
        if row_bits % 8:
            raise ValueError("packed expert row is not byte aligned")
        row_bytes = row_bits // 8
        rows = weight.raw.view(weight.rows, row_bytes)
        local_intermediate = intermediate // ranks
        if projection == 0:
            start = rank * local_intermediate
            end = start + local_intermediate
            shard = torch.cat(
                (
                    rows[start:end],
                    rows[
                        intermediate + start:
                        intermediate + end
                    ],
                ),
                dim=0,
            ).contiguous()
            return shard.reshape(-1), weight.blocks
        if weight.blocks % ranks:
            raise ValueError("packed Down blocks are not TP divisible")
        local_blocks = weight.blocks // ranks
        start_bits = rank * local_blocks * weight.bits
        shard_bits = local_blocks * weight.bits
        if start_bits % 8 or shard_bits % 8:
            raise ValueError(
                "packed Down shard boundary is not byte aligned"
            )
        start_byte = start_bits // 8
        end_byte = start_byte + shard_bits // 8
        return (
            rows[:, start_byte:end_byte].contiguous().reshape(-1),
            local_blocks,
        )

    @staticmethod
    def _tensor_shard_projection_raw(
        weight,
        *,
        projection: int,
        rank: int,
        ranks: int,
        intermediate: int,
    ) -> tuple[torch.Tensor, int]:
        """按三投影数学维切分紧凑索引，不展开 p14。

        gate/up 是 Column-TP，沿输出行切；down 是 Row-TP，沿输入块切。
        """
        row_bits = weight.blocks * weight.bits
        if row_bits % 8:
            raise ValueError("packed expert row is not byte aligned")
        row_bytes = row_bits // 8
        rows = weight.raw.view(weight.rows, row_bytes)
        local_intermediate = intermediate // ranks
        if projection in (0, 1):
            start = rank * local_intermediate
            end = start + local_intermediate
            return (
                rows[start:end].contiguous().reshape(-1),
                weight.blocks,
            )
        if projection != 2:
            raise ValueError(f"invalid expert projection {projection}")
        if weight.blocks % ranks:
            raise ValueError("packed Down blocks are not TP divisible")
        local_blocks = weight.blocks // ranks
        start_bits = rank * local_blocks * weight.bits
        shard_bits = local_blocks * weight.bits
        if start_bits % 8 or shard_bits % 8:
            raise ValueError(
                "packed Down shard boundary is not byte aligned"
            )
        start_byte = start_bits // 8
        end_byte = start_byte + shard_bits // 8
        return (
            rows[:, start_byte:end_byte].contiguous().reshape(-1),
            local_blocks,
        )

    def preload(self) -> None:
        """Read each packed expert once and write it directly into its arena."""
        self.allocate()
        started = time.time()
        n_experts = int(self.store.cfg["n_experts"])
        top_k = int(self.store.cfg["top_k"])
        intermediate = int(self.store.cfg["moe_inter"])
        routed_hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        offsets = [0] * len(self.devices)
        loaded = 0
        for layer in sorted(self.store.man.expert_files):
            projection_vq = bool(self.store.man.projection_vq)
            metadata_by_rank = [
                torch.zeros(
                    15 if projection_vq else 10,
                    n_experts,
                    dtype=torch.long,
                )
                for _ in self.devices
            ]
            for expert_id in range(n_experts):
                tier = self.store.expert_kind(layer, expert_id)
                if tier == "drop":
                    continue
                base_tier = tier.rstrip("z")
                if projection_vq:
                    codebook_variants = (
                        self.store.projection_codebook_variants(
                            layer,
                            expert_id,
                        )
                    )
                    weights = self.store.load_expert_packed(
                        layer,
                        expert_id,
                    )
                    projection_weights = tuple(
                        zip((0, 5, 10), weights)
                    )
                else:
                    codebook_variants = self.store.codebook_variants(
                        layer,
                        base_tier,
                        expert_id,
                    )
                    gu, down = self.store.load_expert_packed(
                        layer,
                        expert_id,
                    )
                    projection_weights = ((0, gu), (5, down))
                target_ranks = self.expert_ranks(layer, expert_id)
                for rank in target_ranks:
                    device = self.devices[rank]
                    arena = self._arenas[rank]
                    metadata = metadata_by_rank[rank]
                    with torch.cuda.device(device):
                        for base, weight in projection_weights:
                            if self.parallelism in {"tensor", "hybrid"}:
                                group_rank = (
                                    rank
                                    if self.parallelism == "tensor"
                                    else rank % self.tensor_group_size
                                )
                                if projection_vq:
                                    raw, blocks = (
                                        self._tensor_shard_projection_raw(
                                            weight,
                                            projection=base // 5,
                                            rank=group_rank,
                                            ranks=self.tensor_group_size,
                                            intermediate=intermediate,
                                        )
                                    )
                                else:
                                    raw, blocks = self._tensor_shard_raw(
                                        weight,
                                        projection=base,
                                        rank=group_rank,
                                        ranks=self.tensor_group_size,
                                        intermediate=intermediate,
                                    )
                            else:
                                raw = weight.raw
                                blocks = weight.blocks
                            start = offsets[rank]
                            end = start + raw.numel()
                            if end > arena.numel():
                                raise RuntimeError(
                                    "packed arena overflow on "
                                    f"rank {rank}"
                                )
                            target = arena[start:end]
                            target.copy_(raw)
                            codebook = self._device_codebook(
                                rank,
                                layer,
                                base_tier,
                                codebook_variants[
                                    (
                                        base // 5
                                        if projection_vq
                                        else (0 if base == 0 else 1)
                                    )
                                ],
                                base,
                                weight.cb,
                                device,
                            )
                            metadata[base + 0, expert_id] = (
                                target.data_ptr()
                            )
                            metadata[base + 1, expert_id] = (
                                codebook.data_ptr()
                            )
                            metadata[base + 2, expert_id] = blocks
                            metadata[base + 3, expert_id] = weight.dim
                            metadata[base + 4, expert_id] = (
                                weight.dtype_tag
                            )
                            offsets[rank] = end
                loaded += 1
                if loaded % 2000 == 0:
                    print(
                        f"[tpq-packed] packed 专家写入 "
                        f"{loaded}",
                        flush=True,
                    )
            self._metadata[layer] = tuple(
                metadata.to(device)
                for metadata, device in zip(
                    metadata_by_rank,
                    self.devices,
                )
            )
        for rank, expected in enumerate(self._rank_payload_bytes):
            if offsets[rank] != expected:
                raise RuntimeError(
                    f"rank {rank} packed bytes mismatch: "
                    f"{offsets[rank]} != {expected}"
                )
        workspace_intermediate = (
            intermediate // self.tensor_group_size
            if self.parallelism in {"tensor", "hybrid"}
            else intermediate
        )
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                self._workspaces[rank] = (
                    torch.empty(
                        top_k,
                        2 * workspace_intermediate,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    torch.empty(
                        top_k,
                        routed_hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    torch.empty(
                        routed_hidden,
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                torch.cuda.synchronize(device)
        if self.parallelism in {"expert", "tensor", "hybrid"}:
            self._streams = [
                torch.cuda.Stream(device=device)
                for device in self.devices
            ]
            self._routed_inputs = [
                torch.empty(
                    1,
                    routed_hidden,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for device in self.devices
            ]
            self._routed_ids = [
                torch.empty(
                    top_k,
                    dtype=torch.long,
                    device=device,
                )
                for device in self.devices
            ]
            self._routed_weights = [
                torch.empty(
                    top_k,
                    dtype=torch.float32,
                    device=device,
                )
                for device in self.devices
            ]
            self._source_inputs = [
                torch.empty(
                    1,
                    routed_hidden,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for device in self.devices
            ]
            self._source_ids = [
                torch.empty(
                    top_k,
                    dtype=torch.long,
                    device=device,
                )
                for device in self.devices
            ]
            self._source_weights = [
                torch.empty(
                    top_k,
                    dtype=torch.float32,
                    device=device,
                )
                for device in self.devices
            ]
            self._source_events = [
                torch.cuda.Event()
                for _ in range(int(self.store.cfg["n_layers"]))
            ]
            self._done_events = [
                [
                    torch.cuda.Event()
                    for _ in range(int(self.store.cfg["n_layers"]))
                ]
                for _ in self.devices
            ]
            self._prefill_ready_events = [
                [
                    torch.cuda.Event()
                    for _ in range(int(self.store.cfg["n_layers"]))
                ]
                for _ in self.devices
            ]
            self._output_events = [
                [
                    torch.cuda.Event()
                    for _ in range(int(self.store.cfg["n_layers"]))
                ]
                for _ in self.devices
            ]
            for layer, event in enumerate(self._source_events):
                owner = self.plan.owner_by_layer[layer]
                with torch.cuda.device(self.devices[owner]):
                    event.cuda_event
            for rank, events in enumerate(self._done_events):
                with torch.cuda.device(self.devices[rank]):
                    for event in events:
                        event.cuda_event
                    for event in self._prefill_ready_events[rank]:
                        event.cuda_event
                    for event in self._output_events[rank]:
                        event.record(
                            torch.cuda.current_stream(self.devices[rank])
                        )
            for owner, (start, end) in enumerate(self.plan.ranges):
                device = self.devices[owner]
                with torch.cuda.device(device):
                    self._return_buffers.append(
                        torch.empty(
                            len(self.devices),
                            end - start,
                            routed_hidden,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    self._reduce_buffers.append(
                        torch.empty(
                            end - start,
                            routed_hidden,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    self._zero_buffers.append(
                        torch.zeros(
                            routed_hidden,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
            if os.environ.get(
                "TPQ_TP_GRAPH",
                os.environ.get("TPQ_KIMI_TP_GRAPH", "1"),
            ) != "0":
                self._prepare_expert_graphs()
        codebook_cache = getattr(self.store, "_cb_cache", None)
        if codebook_cache is not None:
            codebook_cache.clear()
        gc.collect()
        self.active = True
        print(
            f"[tpq-packed] packed 专家全显存完成：{loaded} 个，"
            f"{self.gpu_storage_bytes / 2**30:.2f}GiB，"
            f"{time.time() - started:.1f}s，运行期专家 H2D=0",
            flush=True,
        )

    def _prepare_expert_graphs(self) -> None:
        """Capture one fixed-buffer packed MoE graph per layer and rank."""
        from .fusedext import (
            expert_dispatch_pack_fused,
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .ops import packed_moe_topk

        if not self._streams:
            return
        started = time.time()
        top_k = int(self.store.cfg["top_k"])
        activation = str(
            self.store.cfg.get("activation", "situ")
        )
        activation_beta = float(
            self.store.cfg.get("situ_beta", 4.0)
        )
        linear_value = self.store.cfg.get("situ_linear_beta")
        activation_linear_beta = (
            0.0 if linear_value is None else float(linear_value)
        )
        activation_limit = float(
            self.store.cfg.get("swiglu_limit", 0.0)
        )
        for layer in sorted(self.store.man.expert_files):
            owner = self.plan.owner_by_layer[layer]
            owner_device = self.devices[owner]
            local_layer = layer - self.plan.ranges[owner][0]
            available = (
                self.store.available_mask(layer)
                .nonzero()
                .reshape(-1)[:top_k]
            )
            if available.numel() != top_k:
                raise RuntimeError(
                    f"layer {layer} has fewer than Top-K experts"
                )
            with torch.cuda.device(owner_device):
                self._source_inputs[owner].zero_()
                self._source_ids[owner].copy_(available)
                self._source_weights[owner].fill_(1.0 / top_k)
                torch.cuda.synchronize(owner_device)
            if self.hidden_mode:
                bound = self._bound_hidden_inputs.get(layer)
                if bound is None:
                    raise RuntimeError(
                        f"packed MoE layer {layer} has no fixed all-rank input"
                    )
                rank_inputs, rank_weights, rank_ids = bound
                for rank, device in enumerate(self.devices):
                    with torch.cuda.device(device):
                        rank_inputs[rank].zero_()
                        rank_ids[rank].copy_(
                            available.to(device)
                        )
                        rank_weights[rank].fill_(1.0 / top_k)
                        torch.cuda.synchronize(device)
            else:
                rank_inputs = tuple(self._routed_inputs)
                rank_weights = tuple(self._routed_weights)
                rank_ids = tuple(self._routed_ids)
            graphs: list[torch.cuda.CUDAGraph] = []
            rank_order = (
                tuple(range(len(self.devices)))
                if self.hidden_mode
                else (
                    owner,
                    *(
                        rank
                        for rank in range(len(self.devices))
                        if rank != owner
                    ),
                )
            )
            for rank in rank_order:
                device = self.devices[rank]
                stream = self._streams[rank]
                hidden, output, local_result = self._workspaces[rank]
                destination = (
                    None
                    if self.hidden_mode
                    else self._return_buffers[owner][rank, local_layer]
                )
                result = (
                    local_result
                    if self.hidden_mode or rank != owner
                    else destination
                )

                def launch_rank() -> None:
                    if (
                        not self.hidden_mode
                        and not expert_dispatch_pack_fused(
                            self._source_inputs[owner],
                            self._source_ids[owner],
                            self._source_weights[owner],
                            self._routed_inputs[rank],
                            self._routed_ids[rank],
                            self._routed_weights[rank],
                        )
                    ):
                        raise RuntimeError(
                            "packed MoE graph dispatch rejected fixed buffers"
                        )
                    packed_moe_topk(
                        rank_inputs[rank],
                        rank_ids[rank].reshape(-1),
                        rank_weights[rank].reshape(-1),
                        self._metadata[layer][rank],
                        activation=activation,
                        activation_beta=activation_beta,
                        activation_linear_beta=(
                            activation_linear_beta
                        ),
                        limit=activation_limit,
                        hidden_workspace=hidden,
                        output_workspace=output,
                        result=result,
                        grouped_prefix=-1,
                        **self.store.man.projection_operator_capability(
                            layer
                        ),
                    )
                    if (
                        not self.hidden_mode
                        and destination is not None
                        and
                        rank != owner
                        and not tp_peer_copy_fused(
                            local_result,
                            destination,
                        )
                    ):
                        raise RuntimeError(
                            "packed MoE local reduction dispatch "
                            "was rejected"
                        )

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    launch_rank()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph(
                        keep_graph=self.hidden_mode,
                    )
                    with torch.cuda.graph(graph, stream=stream):
                        launch_rank()
                    if self.hidden_mode:
                        graph.instantiate()
                    graphs.append(graph)
            for device in self.devices:
                torch.cuda.synchronize(device)
            ordered_streams = [
                self._streams[rank] for rank in rank_order
            ]
            ordered_events = [
                self._done_events[rank][layer]
                for rank in rank_order
            ]
            for rank in rank_order:
                device = self.devices[rank]
                stream = self._streams[rank]
                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    self._done_events[rank][layer].record(stream)
                    stream.synchronize()
            with torch.cuda.device(owner_device):
                self._source_events[layer].record(
                    torch.cuda.current_stream(owner_device)
                )
                torch.cuda.synchronize(owner_device)
                self._graphs[layer] = graphs
                self._graph_rank_order[layer] = rank_order
                self._graph_batches[layer] = make_tp_graph_launch_batch(
                    [int(self.devices[rank].index) for rank in rank_order],
                    graphs,
                    ordered_streams,
                    ordered_events,
                    self._source_events[layer],
                )
            if self.hidden_mode:
                self._output_replicas[layer] = []
                for rank, device in enumerate(self.devices):
                    with torch.cuda.device(device):
                        self._output_replicas[layer].append(
                            torch.empty(
                                1,
                                int(
                                    self.store.cfg.get(
                                        "routed_hidden",
                                        self.store.cfg["hidden"],
                                    )
                                ),
                                dtype=torch.bfloat16,
                                device=device,
                            )
                        )
        print(
            f"[tpq-packed] 通用 packed MoE TP Graph 完成："
            f"{len(self._graphs)} 层×{len(self.devices)} 卡，"
            f"{time.time() - started:.1f}s",
            flush=True,
        )

    def _prepare_prefill_expert_graph(
        self,
        layer: int,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float,
        activation_limit: float,
    ) -> None:
        """Capture owner-published graphs used by batched prefill.

        Hidden-mode graphs consume fixed all-rank decode buffers.  Prefill
        enters through :meth:`run` and publishes one routed row from the
        layer owner, so it needs a distinct graph set even when full-TP
        decode is enabled.
        """
        from .fusedext import (
            expert_dispatch_pack_fused,
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .ops import packed_moe_topk, packed_route_slots

        owner = self.plan.owner_by_layer[layer]
        owner_device = self.devices[owner]
        local_layer = layer - self.plan.ranges[owner][0]
        rank_order = (
            owner,
            *(rank for rank in range(len(self.devices)) if rank != owner),
        )
        top_k = int(self.store.cfg["top_k"])
        metadata_directories = tuple(
            metadata.t().contiguous()
            for metadata in self._metadata[layer]
        )
        selected_metadata = tuple(
            torch.empty(
                metadata.shape[0],
                top_k,
                dtype=torch.long,
                device=self.devices[rank],
            )
            for rank, metadata in enumerate(self._metadata[layer])
        )
        slot_ids = tuple(
            torch.arange(
                top_k,
                dtype=torch.long,
                device=device,
            )
            for device in self.devices
        )
        hit_masks = tuple(
            torch.empty(top_k, dtype=torch.bool, device=device)
            for device in self.devices
        )
        self._prefill_metadata_directories[layer] = metadata_directories
        self._prefill_selected_metadata[layer] = selected_metadata
        self._prefill_slot_ids[layer] = slot_ids
        self._prefill_hit_masks[layer] = hit_masks
        graphs: list[torch.cuda.CUDAGraph] = []
        for rank in rank_order:
            device = self.devices[rank]
            stream = self._streams[rank]
            hidden, output, local_result = self._workspaces[rank]
            with torch.cuda.device(device):
                self._routed_inputs[rank].zero_()
                self._routed_ids[rank].copy_(
                    self._source_ids[owner], non_blocking=False
                )
                self._routed_weights[rank].fill_(
                    0.0
                )
                self._routed_weights[rank].copy_(
                    self._source_weights[owner], non_blocking=False
                )

            def launch_rank(rank_index: int = rank) -> None:
                hidden.zero_()
                output.zero_()
                local_result.zero_()
                if not packed_route_slots(
                    self._routed_ids[rank_index],
                    metadata_directories[rank_index],
                    output=selected_metadata[rank_index],
                    hit_mask=hit_masks[rank_index],
                ):
                    raise RuntimeError(
                        "packed MoE prefill route-slot gather was rejected"
                    )
                packed_moe_topk(
                    self._routed_inputs[rank_index],
                    slot_ids[rank_index],
                    self._routed_weights[rank_index],
                    selected_metadata[rank_index],
                    activation=activation,
                    activation_beta=activation_beta,
                    activation_linear_beta=activation_linear_beta,
                    limit=activation_limit,
                    hidden_workspace=hidden,
                    output_workspace=output,
                    result=local_result,
                    grouped_prefix=-1,
                    **self.store.man.projection_operator_capability(layer),
                )

            with torch.cuda.device(device), torch.cuda.stream(stream):
                launch_rank()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, stream=stream):
                    launch_rank()
                graph.instantiate()
                graphs.append(graph)
        for device in self.devices:
            torch.cuda.synchronize(device)
        ordered_streams = [self._streams[rank] for rank in rank_order]
        ordered_events = [
            self._done_events[rank][layer] for rank in rank_order
        ]
        for rank in rank_order:
            stream = self._streams[rank]
            with (
                torch.cuda.device(self.devices[rank]),
                torch.cuda.stream(stream),
            ):
                self._done_events[rank][layer].record(stream)
                stream.synchronize()
        with torch.cuda.device(owner_device):
            self._source_events[layer].record(
                torch.cuda.current_stream(owner_device)
            )
            torch.cuda.synchronize(owner_device)
            self._prefill_graphs[layer] = graphs
            self._prefill_graph_rank_order[layer] = rank_order
            self._prefill_graph_batches[layer] = make_tp_graph_launch_batch(
                [int(self.devices[rank].index) for rank in rank_order],
                graphs,
                ordered_streams,
                ordered_events,
                self._source_events[layer],
            )

    def compose_route_topk(
        self,
        logits_by_layer: dict[int, object],
        corrections_by_layer: dict[int, tuple[torch.Tensor, ...]],
        masks_by_layer: dict[int, tuple[torch.Tensor, ...]],
        route_buffers_by_layer: dict[int, tuple],
        *,
        scoring_func: str,
        top_k: int,
        normalize: bool,
        scaling: float,
        n_group: int,
        topk_group: int,
        layers=None,
    ) -> None:
        """Compose registered Top-K and packed-expert graphs per TP rank.

        The Router/Down collective publishes fixed logits and latent replicas.
        Each rank then performs the same registered Top-K and computes its
        shard of every selected packed expert.  Only graph scheduling changes;
        packed indices and all-rank expert ownership are unchanged.
        """
        if not self.hidden_mode or not self._graphs:
            raise RuntimeError(
                "route/packed composition requires all-rank packed graphs"
            )
        from .fusedext import make_tp_graph_sequence_batch
        from .ops import route_topk

        selected_layers = (
            sorted(self._graphs)
            if layers is None
            else sorted({int(layer) for layer in layers})
        )
        unknown = set(selected_layers) - set(self._graphs)
        if unknown:
            raise ValueError(
                "route/packed composition references unknown layers: "
                + ",".join(str(layer) for layer in sorted(unknown))
            )
        for layer in selected_layers:
            logits = logits_by_layer[layer]
            corrections = corrections_by_layer[layer]
            masks = masks_by_layer[layer]
            weight_buffers, index_buffers = route_buffers_by_layer[layer]
            if (
                tuple(logits.devices) != self.devices
                or logits.ready_events is None
                or len(corrections) != len(self.devices)
                or len(masks) != len(self.devices)
            ):
                raise ValueError(
                    "route/packed fixed all-rank layout mismatch"
                )
            rank_order = self._graph_rank_order[layer]
            if tuple(rank_order) != tuple(range(len(self.devices))):
                raise RuntimeError(
                    "route/packed composition forbids owner-ordered graphs"
                )
            expert_by_rank = {
                rank: self._graphs[layer][ordered_rank]
                for ordered_rank, rank in enumerate(rank_order)
            }
            route_graphs = []
            for rank, device in enumerate(self.devices):
                stream = self._streams[rank]

                def launch_route(rank_index: int = rank) -> None:
                    route = route_topk(
                        logits.replicas[rank_index],
                        corrections[rank_index],
                        masks[rank_index],
                        scoring_func=scoring_func,
                        top_k=int(top_k),
                        normalize=bool(normalize),
                        scaling=float(scaling),
                        n_group=int(n_group),
                        topk_group=int(topk_group),
                        output_buffers=(
                            weight_buffers[rank_index],
                            index_buffers[rank_index],
                        ),
                    )
                    if route is None:
                        raise RuntimeError(
                            "registered route Top-K rejected graph inputs"
                        )

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    launch_route()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, stream=stream):
                        launch_route()
                    graph.instantiate()
                    stream.synchronize()
                route_graphs.append(graph)
            self._route_graphs[layer] = tuple(route_graphs)
            self._graph_batches[layer] = make_tp_graph_sequence_batch(
                [int(device.index) for device in self.devices],
                [
                    [
                        route_graphs[rank],
                        expert_by_rank[rank],
                    ]
                    for rank in range(len(self.devices))
                ],
                list(self._streams),
                [
                    self._done_events[rank][layer]
                    for rank in range(len(self.devices))
                ],
                self._source_events[layer],
            )
        print(
            "[tpq-packed] 通用 Route TopK→packed MoE 全rank父图完成："
            f"{len(selected_layers)} 层×{len(self.devices)} rank",
            flush=True,
        )

    def run_hidden(
        self,
        layer: int,
        value,
        routes: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
    ):
        """Run one tensor-sharded expert set and publish it on every rank.

        ``value`` and every route pair are rank-local replicas.  Each rank
        executes the same selected experts using its packed Column/Row shard;
        the only packed-MoE collective is the final all-rank Row reduction.
        """
        del activation, activation_beta, activation_linear_beta
        from .ops import TPHidden

        if (
            not self.active
            or not self.hidden_mode
            or len(routes) != len(self.devices)
            or value.ready_events is None
        ):
            raise RuntimeError(
                "packed MoE all-rank state is unavailable"
            )
        graph_batch = self._graph_batches.get(layer)
        outputs = self._output_replicas.get(layer)
        if graph_batch is None or outputs is None:
            raise RuntimeError(
                f"packed MoE layer {layer} graph is unavailable"
            )
        bound = self._bound_hidden_inputs.get(layer)
        if bound is None:
            raise RuntimeError(
                f"packed MoE layer {layer} fixed inputs are unavailable"
            )
        bound_inputs, bound_weights, bound_ids = bound
        for rank, device in enumerate(self.devices):
            weights, indices = routes[rank]
            weights = weights.reshape(-1)
            indices = indices.reshape(-1)
            if (
                weights.device != device
                or indices.device != device
                or weights.shape != bound_weights[rank].shape
                or indices.shape != bound_ids[rank].shape
                or weights.data_ptr() != bound_weights[rank].data_ptr()
                or indices.data_ptr() != bound_ids[rank].data_ptr()
                or value.replicas[rank].data_ptr()
                != bound_inputs[rank].data_ptr()
            ):
                raise ValueError(
                    "packed MoE route replica layout mismatch"
                )
        peer_access = all(
            left == right
            or torch.cuda.can_device_access_peer(
                int(self.devices[left].index),
                int(self.devices[right].index),
            )
            for left in range(len(self.devices))
            for right in range(len(self.devices))
        )
        if not peer_access:
            graph_batch.launch_from_events(
                [
                    value.ready_events[rank].cuda_event
                    for rank in range(len(self.devices))
                ]
            )
            contributions = [
                self._workspaces[rank][2]
                for rank in range(len(self.devices))
            ]
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    torch.cuda.current_stream(device).wait_event(
                        self._done_events[rank][layer]
                    )
            self.hits += int(routes[0][1].numel())
            return TPHidden(
                self.devices,
                tuple(outputs),
                tuple(
                    self._output_events[rank][layer]
                    for rank in range(len(self.devices))
                ),
            ).reduce_from(contributions)
        with torch.cuda.device(self.devices[0]):
            graph_batch.launch_all_rank_from_events(
                [
                    value.ready_events[rank].cuda_event
                    for rank in range(len(self.devices))
                ],
                [
                    self._workspaces[rank][2]
                    for rank in range(len(self.devices))
                ],
                outputs,
                [
                    self._output_events[rank][layer].cuda_event
                    for rank in range(len(self.devices))
                ],
            )
        self.hits += int(routes[0][1].numel())
        return TPHidden(
            self.devices,
            tuple(outputs),
            tuple(
                self._output_events[rank][layer]
                for rank in range(len(self.devices))
            ),
        )

    def run(
        self,
        layer: int,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> torch.Tensor:
        if not self.active:
            raise RuntimeError("packed experts are not ready")
        from .ops import packed_moe_topk

        if self.parallelism == "pipeline":
            rank = self.plan.owner_by_layer[layer]
            device = self.devices[rank]
            hidden, output, result = self._workspaces[rank]
            with torch.cuda.device(device):
                return packed_moe_topk(
                    value.to(torch.bfloat16),
                    route_ids.reshape(-1),
                    route_weights.reshape(-1),
                    self._metadata[layer][rank],
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=(
                        0.0
                        if activation_linear_beta is None
                        else float(activation_linear_beta)
                    ),
                    hidden_workspace=hidden,
                    output_workspace=output,
                    result=result,
                    grouped_prefix=-1,
                    **self.store.man.projection_operator_capability(
                        layer
                    ),
                    limit=float(limit),
                )

        owner = self.plan.owner_by_layer[layer]
        owner_device = self.devices[owner]
        if self.hidden_mode and layer not in self._prefill_graph_batches:
            from .fusedext import expert_dispatch_pack_fused

            # The prefill graph indirectly consumes the routed projection
            # metadata selected by the current expert IDs.  Capturing it at
            # pool construction time uses uninitialised route buffers and can
            # preserve an invalid first execution on multi-device CUDA graph
            # replay.  Publish the first real row before the one-time capture.
            with torch.cuda.device(owner_device):
                if not expert_dispatch_pack_fused(
                    value,
                    route_ids.reshape(-1),
                    route_weights.reshape(-1),
                    self._source_inputs[owner],
                    self._source_ids[owner],
                    self._source_weights[owner],
                ):
                    raise RuntimeError(
                        "packed MoE source publication was rejected"
                    )
                torch.cuda.current_stream(owner_device).synchronize()
            self._prepare_prefill_expert_graph(
                layer,
                activation=activation,
                activation_beta=float(activation_beta),
                activation_linear_beta=(
                    0.0
                    if activation_linear_beta is None
                    else float(activation_linear_beta)
                ),
                activation_limit=float(limit),
            )
        graph_batch = (
            self._prefill_graph_batches.get(layer)
            if self.hidden_mode
            else self._graph_batches.get(layer)
        )
        if graph_batch is not None:
            from .fusedext import expert_dispatch_pack_fused

            local_layer = layer - self.plan.ranges[owner][0]
            with torch.cuda.device(owner_device):
                dispatched = expert_dispatch_pack_fused(
                    value,
                    route_ids.reshape(-1),
                    route_weights.reshape(-1),
                    self._source_inputs[owner],
                    self._source_ids[owner],
                    self._source_weights[owner],
                )
                if not dispatched:
                    raise RuntimeError(
                        "packed MoE source publication was rejected"
                    )
                self._source_events[layer].record(
                    torch.cuda.current_stream(owner_device)
                )
                rank_order = (
                    self._prefill_graph_rank_order[layer]
                    if self.hidden_mode
                    else self._graph_rank_order[layer]
                )
                contributions = [
                    self._return_buffers[owner][rank, local_layer]
                    for rank in rank_order
                ]
                if self.hidden_mode:
                    source_ready = self._source_events[layer]
                    for rank, device in enumerate(self.devices):
                        stream = self._streams[rank]
                        with (
                            torch.cuda.device(device),
                            torch.cuda.stream(stream),
                        ):
                            stream.wait_event(source_ready)
                            self._routed_inputs[rank].copy_(
                                self._source_inputs[owner],
                                non_blocking=True,
                            )
                            self._routed_ids[rank].copy_(
                                self._source_ids[owner],
                                non_blocking=True,
                            )
                            self._routed_weights[rank].copy_(
                                self._source_weights[owner],
                                non_blocking=True,
                            )
                            self._prefill_ready_events[rank][layer].record(
                                stream
                            )
                    graph_batch.launch_from_events(
                        [
                            self._prefill_ready_events[rank][
                                layer
                            ].cuda_event
                            for rank in range(len(self.devices))
                        ]
                    )
                    reduced = self._reduce_buffers[owner][local_layer]
                    if self._peer_access:
                        for rank, device in enumerate(self.devices):
                            stream = self._streams[rank]
                            with (
                                torch.cuda.device(device),
                                torch.cuda.stream(stream),
                            ):
                                stream.wait_event(
                                    self._done_events[rank][layer]
                                )
                                self._return_buffers[owner][
                                    rank, local_layer
                                ].copy_(
                                    self._workspaces[rank][2],
                                    non_blocking=True,
                                )
                                self._done_events[rank][layer].record(stream)
                        owner_stream = torch.cuda.current_stream(owner_device)
                        for rank in range(len(self.devices)):
                            owner_stream.wait_event(
                                self._done_events[rank][layer]
                            )
                        reduced.copy_(contributions[0])
                        for rank in range(1, len(contributions)):
                            reduced.add_(contributions[rank])
                    else:
                        partials = [
                            self._workspaces[rank][2]
                            for rank in range(len(self.devices))
                        ]
                        if not torch.cuda.nccl.is_available(partials):
                            raise RuntimeError(
                                "packed MoE TP requires CUDA P2P or NCCL"
                            )
                        torch.cuda.nccl.all_reduce(
                            partials,
                            streams=self._streams,
                        )
                        for rank, device in enumerate(self.devices):
                            with torch.cuda.device(device):
                                self._done_events[rank][layer].record(
                                    self._streams[rank]
                                )
                        torch.cuda.current_stream(owner_device).wait_event(
                            self._done_events[owner][layer]
                        )
                        reduced.copy_(partials[owner])
                else:
                    reduced = graph_batch.launch_reduce(
                        contributions,
                        self._zero_buffers[owner],
                    )
            self.hits += route_ids.numel()
            return reduced

        source_ready = self._source_events[layer]
        source_ready.record(torch.cuda.current_stream(owner_device))
        local_layer = layer - self.plan.ranges[owner][0]
        for rank, device in enumerate(self.devices):
            stream = self._streams[rank]
            hidden, output, result = self._workspaces[rank]
            with (
                torch.cuda.device(device),
                torch.cuda.stream(stream),
            ):
                stream.wait_event(source_ready)
                self._routed_inputs[rank].copy_(
                    value,
                    non_blocking=True,
                )
                self._routed_ids[rank].copy_(
                    route_ids.reshape(-1),
                    non_blocking=True,
                )
                self._routed_weights[rank].copy_(
                    route_weights.reshape(-1),
                    non_blocking=True,
                )
                partial = packed_moe_topk(
                    self._routed_inputs[rank],
                    self._routed_ids[rank],
                    self._routed_weights[rank],
                    self._metadata[layer][rank],
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=(
                        0.0
                        if activation_linear_beta is None
                        else float(activation_linear_beta)
                    ),
                    hidden_workspace=hidden,
                    output_workspace=output,
                    result=result,
                    grouped_prefix=-1,
                    **self.store.man.projection_operator_capability(
                        layer
                    ),
                    limit=float(limit),
                )
                if self._peer_access:
                    self._return_buffers[owner][
                        rank, local_layer
                    ].copy_(
                        partial,
                        non_blocking=True,
                    )
                self._done_events[rank][layer].record(stream)
        self.hits += route_ids.numel()
        if not self._peer_access:
            partials = [
                self._workspaces[rank][2]
                for rank in range(len(self.devices))
            ]
            if not torch.cuda.nccl.is_available(partials):
                raise RuntimeError(
                    "packed MoE TP requires CUDA P2P or NCCL"
                )
            torch.cuda.nccl.all_reduce(
                partials,
                streams=self._streams,
            )
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    self._done_events[rank][layer].record(
                        self._streams[rank]
                    )
            with torch.cuda.device(owner_device):
                torch.cuda.current_stream(owner_device).wait_event(
                    self._done_events[owner][layer]
                )
                reduced = self._reduce_buffers[owner][local_layer]
                reduced.copy_(partials[owner])
            return reduced
        with torch.cuda.device(owner_device):
            owner_stream = torch.cuda.current_stream(owner_device)
            for rank in range(len(self.devices)):
                owner_stream.wait_event(
                    self._done_events[rank][layer]
                )
            contributions = self._return_buffers[owner][
                :, local_layer
            ]
            reduced = self._reduce_buffers[owner][local_layer]
            reduced.copy_(contributions[0])
            for rank in range(1, len(self.devices)):
                reduced.add_(contributions[rank])
        return reduced


KimiPackedExpertPool = PackedExpertPool


__all__ = [
    "KimiLayerPlan",
    "KimiPackedExpertPool",
    "PackedExpertPool",
    "build_kimi_layer_plan",
]
