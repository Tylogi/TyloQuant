"""Kimi K3 单卡紧凑专家 RAM+GPU 执行池。

与通用 ``ExpertPool`` 的主要区别：

* 12/14-bit 索引在 RAM 中保持 CCCP 原始打包格式，不展开为 uint16；
* 按专家签名预分配稳定 GPU 槽，换专家只覆盖槽内容；
* 上一个 token 的路由在后台预取，需求路径按层等待；
* Top-16 GU、SiTU、DOWN、路由加权直接使用 Kimi 融合 CUDA 核。

非专家 dense、注意力、共享专家和 KV 路径完全不变。
"""

from __future__ import annotations

import gc
import os
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import torch

from .expert_slots import SlotBook
from .store import CCCPStore, PackedVQWeight, PinnedStage


@dataclass(frozen=True)
class HostPackedWeight:
    raw: torch.Tensor
    cb: torch.Tensor
    rows: int
    cols: int
    blocks: int
    dim: int
    bits: int

    @classmethod
    def from_store(
        cls,
        weight: PackedVQWeight,
        codebook: torch.Tensor,
    ) -> "HostPackedWeight":
        return cls(
            raw=weight.raw,
            cb=codebook,
            rows=weight.rows,
            cols=weight.cols,
            blocks=weight.blocks,
            dim=weight.dim,
            bits=weight.bits,
        )

    @property
    def dtype_tag(self) -> int:
        return {8: 0, 16: 1, 12: 2, 14: 3}[self.bits]

    @property
    def nbytes(self) -> int:
        return self.raw.nbytes + self.cb.nbytes


@dataclass(frozen=True)
class DevicePackedWeight:
    raw: torch.Tensor
    cb: torch.Tensor
    rows: int
    cols: int
    blocks: int
    dim: int
    bits: int

    @property
    def dtype_tag(self) -> int:
        return {8: 0, 16: 1, 12: 2, 14: 3}[self.bits]

    @property
    def nbytes(self) -> int:
        return self.raw.nbytes + self.cb.nbytes


PackedExpert = tuple[HostPackedWeight, HostPackedWeight]
DeviceExpert = tuple[DevicePackedWeight, DevicePackedWeight]


@dataclass(frozen=True)
class PackedExpertSignature:
    gu_raw_bytes: int
    gu_cb_shape: tuple[int, int]
    gu_rows: int
    gu_cols: int
    gu_blocks: int
    gu_dim: int
    gu_bits: int
    down_raw_bytes: int
    down_cb_shape: tuple[int, int]
    down_rows: int
    down_cols: int
    down_blocks: int
    down_dim: int
    down_bits: int

    @classmethod
    def of(cls, expert: PackedExpert) -> "PackedExpertSignature":
        gu, down = expert
        return cls(
            gu.raw.numel(),
            tuple(gu.cb.shape),
            gu.rows,
            gu.cols,
            gu.blocks,
            gu.dim,
            gu.bits,
            down.raw.numel(),
            tuple(down.cb.shape),
            down.rows,
            down.cols,
            down.blocks,
            down.dim,
            down.bits,
        )

    @property
    def raw_slot_bytes(self) -> int:
        return self.gu_raw_bytes + self.down_raw_bytes

    @property
    def codebook_slot_bytes(self) -> int:
        gu_items = self.gu_cb_shape[0] * self.gu_cb_shape[1]
        down_items = self.down_cb_shape[0] * self.down_cb_shape[1]
        return (gu_items + down_items) * torch.bfloat16.itemsize

    @property
    def slot_bytes(self) -> int:
        return self.raw_slot_bytes + self.codebook_slot_bytes

    def storage_bytes(self, resident_codebooks: bool) -> int:
        return (
            self.raw_slot_bytes
            if resident_codebooks
            else self.slot_bytes
        )


def allocate_packed_slots(
    counts: dict[PackedExpertSignature, int],
    budget: int,
    minimum: int,
    weights: dict[PackedExpertSignature, float] | None = None,
    *,
    resident_codebooks: bool = False,
) -> dict[PackedExpertSignature, int]:
    """分配槽位，同时保证任一签名容得下完整 Top-K。

    ``weights`` 表示运行时路由流量，而不是模型中各档专家的静态数量。
    混合精度模型的高精度专家数量可能很少、调用却很频繁；若仍按静态数量
    分槽，该档位会在每个 token 内循环淘汰，显著放大 PCIe 传输。
    """
    if budget <= 0 or not counts:
        return {}
    minimums = {
        signature: min(count, max(1, int(minimum)))
        for signature, count in counts.items()
    }
    minimum_bytes = sum(
        signature.storage_bytes(resident_codebooks) * count
        for signature, count in minimums.items()
    )
    if minimum_bytes > budget:
        raise RuntimeError(
            "Kimi packed GPU cache is too small for one complete Top-K of "
            f"every tier: need {minimum_bytes / 2**30:.2f} GiB, "
            f"have {budget / 2**30:.2f} GiB"
        )

    usable_weights = None
    if weights is not None:
        usable_weights = {
            signature: max(0.0, float(weights.get(signature, 0.0)))
            for signature in counts
        }
        if not any(usable_weights.values()):
            usable_weights = None

    if usable_weights is None:
        total_bytes = sum(
            signature.storage_bytes(resident_codebooks) * count
            for signature, count in counts.items()
        )
        scale = min(1.0, budget / max(1, total_bytes))
        allocated = {
            signature: min(
                count,
                max(minimums[signature], int(count * scale)),
            )
            for signature, count in counts.items()
        }
    else:
        allocated = dict(minimums)

    def used() -> int:
        return sum(
            signature.storage_bytes(resident_codebooks) * count
            for signature, count in allocated.items()
        )

    while used() > budget:
        candidates = [
            signature
            for signature in counts
            if allocated[signature] > minimums[signature]
        ]
        if not candidates:
            raise RuntimeError("cannot fit minimum Kimi packed GPU slots")
        signature = max(
            candidates,
            key=lambda item: (
                item.storage_bytes(resident_codebooks)
                * allocated[item]
            ),
        )
        allocated[signature] -= 1

    while True:
        candidates = [
            signature
            for signature, total in counts.items()
            if allocated[signature] < total
            and (
                used() + signature.storage_bytes(resident_codebooks)
                <= budget
            )
        ]
        if not candidates:
            break
        target = usable_weights or counts
        signature = min(
            candidates,
            key=lambda item: (
                allocated[item] / max(float(target[item]), 1e-12),
                item.storage_bytes(resident_codebooks),
            ),
        )
        allocated[signature] += 1
    return allocated


class _PackedArena:
    def __init__(
        self,
        count: int,
        signature: PackedExpertSignature,
        device: torch.device,
        *,
        resident_codebooks: bool,
    ):
        self.signature = signature
        self.resident_codebooks = resident_codebooks
        self.book = SlotBook(count)
        self.gu_raw = torch.empty(
            count,
            signature.gu_raw_bytes,
            dtype=torch.uint8,
            device=device,
        )
        self.down_raw = torch.empty(
            count,
            signature.down_raw_bytes,
            dtype=torch.uint8,
            device=device,
        )
        self.gu_cb = None
        self.down_cb = None
        if not resident_codebooks:
            self.gu_cb = torch.empty(
                count,
                *signature.gu_cb_shape,
                dtype=torch.bfloat16,
                device=device,
            )
            self.down_cb = torch.empty(
                count,
                *signature.down_cb_shape,
                dtype=torch.bfloat16,
                device=device,
            )

    @property
    def nbytes(self) -> int:
        output = self.gu_raw.nbytes + self.down_raw.nbytes
        if self.gu_cb is not None:
            output += self.gu_cb.nbytes
        if self.down_cb is not None:
            output += self.down_cb.nbytes
        return output

    def lease(
        self,
        key: tuple[int, int],
        gu_codebook: torch.Tensor,
        down_codebook: torch.Tensor,
    ) -> tuple[object, DeviceExpert]:
        lease = self.book.acquire(key)
        slot = lease.slot
        signature = self.signature
        if not self.resident_codebooks:
            if self.gu_cb is None or self.down_cb is None:
                raise RuntimeError("Kimi slot codebook storage is missing")
            gu_codebook = self.gu_cb[slot]
            down_codebook = self.down_cb[slot]
        return lease, (
            DevicePackedWeight(
                self.gu_raw[slot],
                gu_codebook,
                signature.gu_rows,
                signature.gu_cols,
                signature.gu_blocks,
                signature.gu_dim,
                signature.gu_bits,
            ),
            DevicePackedWeight(
                self.down_raw[slot],
                down_codebook,
                signature.down_rows,
                signature.down_cols,
                signature.down_blocks,
                signature.down_dim,
                signature.down_bits,
            ),
        )


class _PackedArenas:
    def __init__(
        self,
        specs: dict[PackedExpertSignature, int],
        device: torch.device,
        *,
        resident_codebooks: bool,
    ):
        self.arenas = {
            signature: _PackedArena(
                count,
                signature,
                device,
                resident_codebooks=resident_codebooks,
            )
            for signature, count in specs.items()
            if count > 0
        }
        self.leases: dict[
            tuple[int, int],
            tuple[PackedExpertSignature, object],
        ] = {}

    @property
    def nbytes(self) -> int:
        return sum(arena.nbytes for arena in self.arenas.values())

    def touch(self, key: tuple[int, int]) -> None:
        item = self.leases.get(key)
        if item is not None:
            signature, _lease = item
            self.arenas[signature].book.touch(key)

    def protect(self, key: tuple[int, int]) -> bool:
        item = self.leases.get(key)
        if item is None:
            return False
        signature, _lease = item
        return self.arenas[signature].book.protect(key)

    def unprotect(self, key: tuple[int, int]) -> None:
        item = self.leases.get(key)
        if item is not None:
            signature, _lease = item
            self.arenas[signature].book.unprotect(key)

    @property
    def protected_count(self) -> int:
        return sum(
            arena.book.protected_count
            for arena in self.arenas.values()
        )

    def lease(
        self,
        key: tuple[int, int],
        expert: PackedExpert,
        device_codebooks: dict[int, torch.Tensor],
    ) -> tuple[tuple[int, int] | None, DeviceExpert]:
        signature = PackedExpertSignature.of(expert)
        arena = self.arenas[signature]
        gu_codebook = expert[0].cb
        down_codebook = expert[1].cb
        if arena.resident_codebooks:
            gu_codebook = device_codebooks[expert[0].cb.data_ptr()]
            down_codebook = device_codebooks[expert[1].cb.data_ptr()]
        lease, device_expert = arena.lease(
            key,
            gu_codebook,
            down_codebook,
        )
        replaced = lease.replaced
        if replaced is not None:
            self.leases.pop(replaced, None)
        self.leases[key] = (signature, lease)
        return replaced, device_expert


class KimiPackedHybridPool:
    """全量紧凑 RAM + 有界稳定 VRAM 的 Kimi Top-16 专家池。"""

    device_routed = True
    full_resident = False
    prefetch_default = False

    def __init__(
        self,
        store: CCCPStore,
        budget_gb: float,
        *,
        device: str | torch.device,
        ram_gb: float = 0.0,
    ):
        self.store = store
        self.device = torch.device(device)
        self.budget = int(float(budget_gb) * 2**30)
        self.ram_budget = int(float(ram_gb) * 2**30)
        self.pinned: dict[tuple[int, int], PackedExpert] = {}
        self.cache: OrderedDict[tuple[int, int], DeviceExpert] = OrderedDict()
        self.bytes = 0
        self.ram_bytes = 0
        self.hits = 0
        self.miss = 0
        self.prefetch_hits = 0
        self.uploaded_bytes = 0
        self.transfer_seconds = 0.0
        self.last_transfer_seconds = 0.0
        self._host_codebooks: dict[
            tuple[int, str, int | None, str],
            torch.Tensor,
        ] = {}
        self._device_codebooks: dict[int, torch.Tensor] = {}
        self._host_pinned_bytes = 0
        self._resident_codebooks = (
            os.environ.get("TPQ_KIMI_RESIDENT_CODEBOOKS", "1") != "0"
        )
        self._protect_previous = (
            os.environ.get("TPQ_KIMI_PROTECT_PREV", "0") != "0"
        )
        self._arenas: _PackedArenas | None = None
        self._stage = PinnedStage(self.device)
        self._lock = threading.RLock()
        self._transfer_lock = threading.RLock()
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tpq-kimi-packed-prefetch",
        )
        self._prefetch_futures: set[Future] = set()
        self._last_ids: dict[int, list[int]] = {}
        self._protected_by_layer: dict[
            int,
            tuple[tuple[int, int], ...],
        ] = {}
        self._route_ids: torch.Tensor | None = None
        self._ordered_weights: torch.Tensor | None = None
        self._metadata: torch.Tensor | None = None
        self._workspaces: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        self.slot_mix = self._read_slot_mix()
        self.arena_slots: dict[str, int] = {}

    @staticmethod
    def _read_slot_mix() -> dict[str, float] | None:
        value = os.environ.get("TPQ_KIMI_SLOT_MIX", "").strip()
        if not value or value.lower() in ("model", "static", "off", "0"):
            return None
        output: dict[str, float] = {}
        for item in value.split(","):
            name, separator, weight = item.partition("=")
            if not separator:
                raise ValueError(
                    "TPQ_KIMI_SLOT_MIX must use tier=weight entries"
                )
            name = name.strip().lower()
            if name not in {"x", "w", "v", "vv"}:
                raise ValueError(f"unknown Kimi slot tier: {name}")
            output[name] = max(0.0, float(weight))
        if not any(output.values()):
            raise ValueError("TPQ_KIMI_SLOT_MIX contains no positive weight")
        return output

    @staticmethod
    def _signature_tier(signature: PackedExpertSignature) -> str:
        if signature.gu_bits == 8 and signature.gu_dim == 4:
            return "v"
        if signature.gu_bits == 14 and signature.gu_dim == 8:
            return "w"
        if signature.gu_bits == 12 and signature.gu_dim == 4:
            return "vv"
        if signature.gu_bits == 12 and signature.gu_dim == 8:
            return "x"
        return f"p{signature.gu_bits}d{signature.gu_dim}"

    @property
    def host_expert_bytes(self) -> int:
        raw = sum(
            gu.raw.nbytes + down.raw.nbytes
            for gu, down in self.pinned.values()
        )
        codebooks = sum(cb.nbytes for cb in self._host_codebooks.values())
        return raw + codebooks

    @property
    def gpu_arena_bytes(self) -> int:
        return 0 if self._arenas is None else self._arenas.nbytes

    @property
    def gpu_storage_bytes(self) -> int:
        workspace = 0
        if self._workspaces is not None:
            workspace += sum(tensor.nbytes for tensor in self._workspaces)
        if self._metadata is not None:
            workspace += self._metadata.nbytes
        if self._route_ids is not None:
            workspace += self._route_ids.nbytes
        if self._ordered_weights is not None:
            workspace += self._ordered_weights.nbytes
        workspace += sum(
            codebook.nbytes
            for codebook in self._device_codebooks.values()
        )
        return self.gpu_arena_bytes + workspace

    @property
    def protected_experts(self) -> int:
        return (
            0
            if self._arenas is None
            else self._arenas.protected_count
        )

    def _host_codebook(
        self,
        key: tuple[int, str, int | None, str],
        cb: torch.Tensor,
    ) -> torch.Tensor:
        """Return one stable BF16 codebook for a semantic archive key.

        The store's FP32 codebook cache is populated concurrently during expert
        preload.  A losing duplicate tensor can be freed immediately and its
        ``data_ptr`` reused by another layer.  Pointer-only host keys therefore
        caused rare cross-layer codebook aliasing and nondeterministic logits.
        """
        with self._lock:
            value = self._host_codebooks.get(key)
            if value is None:
                value = cb.to(dtype=torch.bfloat16).contiguous()
                self._host_codebooks[key] = value
        return value

    def _load_one(self, layer: int, expert_id: int) -> PackedExpert:
        gu, down = self.store.load_expert_packed(layer, expert_id)
        tier = self.store.expert_kind(layer, expert_id).rstrip("z")
        keys = self.store._expert_keys[layer]
        gu_stem = f"cb.gu.{tier}"
        down_stem = (
            f"cb.down.{tier}"
            if f"cb.down.{tier}" in keys
            else f"cb.dn.{tier}"
        )
        dedicated = (
            expert_id
            if (
                f"{gu_stem}.e{expert_id}" in keys
                and f"{down_stem}.e{expert_id}" in keys
            )
            else None
        )
        return (
            HostPackedWeight(
                gu.raw,
                self._host_codebook(
                    (layer, tier, dedicated, "gu"),
                    gu.cb,
                ),
                gu.rows,
                gu.cols,
                gu.blocks,
                gu.dim,
                gu.bits,
            ),
            HostPackedWeight(
                down.raw,
                self._host_codebook(
                    (layer, tier, dedicated, "down"),
                    down.cb,
                ),
                down.rows,
                down.cols,
                down.blocks,
                down.dim,
                down.bits,
            ),
        )

    def preload_all(self, reserve_gb: float | None = None) -> bool:
        if self.pinned:
            return True
        if os.environ.get("TPQ_FULL_RESIDENT", "1") == "0":
            return False
        import psutil

        if reserve_gb is None:
            reserve_gb = float(
                os.environ.get("TPQ_RESIDENT_RESERVE_GB", "3.0")
            )
        expert_files = [
            os.path.join(self.store.root, filename)
            for filename in self.store.man.expert_files.values()
        ]
        stored_bytes = sum(
            os.path.getsize(path)
            for path in expert_files
            if os.path.exists(path)
        )
        available = psutil.virtual_memory().available
        if stored_bytes + int(reserve_gb * 2**30) > available:
            print(
                "[tpq-kimi] 紧凑专家无法全量常驻 RAM："
                f"文件 {stored_bytes / 2**30:.1f}GiB + "
                f"预留 {reserve_gb:.1f}GiB > "
                f"可用 {available / 2**30:.1f}GiB",
                flush=True,
            )
            return False

        n_experts = int(self.store.cfg["n_experts"])
        keys = [
            (layer, expert_id)
            for layer in sorted(self.store.man.expert_files)
            for expert_id in range(n_experts)
            if self.store.expert_kind(layer, expert_id) != "drop"
        ]
        workers = max(1, int(os.environ.get("TPQ_LOAD_WORKERS", "12")))
        started = time.perf_counter()
        print(
            f"[tpq-kimi] 紧凑专家常驻 RAM：{len(keys)} 个，"
            f"文件约 {stored_bytes / 2**30:.1f}GiB，workers={workers}",
            flush=True,
        )
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="tpq-kimi-packed-load",
        ) as executor:
            futures = {
                executor.submit(self._load_one, *key): key
                for key in keys
            }
            for index, future in enumerate(as_completed(futures), 1):
                self.pinned[futures[future]] = future.result()
                if index % 2000 == 0:
                    print(
                        f"[tpq-kimi] 紧凑专家常驻 "
                        f"{index}/{len(keys)}",
                        flush=True,
                    )
        # 所有运行时专家都只引用 BF16 码本；释放 store 的 FP32 中间副本。
        self.store._cb_cache.clear()
        gc.collect()
        self.ram_bytes = self.host_expert_bytes
        print(
            f"[tpq-kimi] 紧凑专家 RAM 常驻完成："
            f"{len(self.pinned)} 个 / {self.ram_bytes / 2**30:.1f}GiB，"
            f"{time.perf_counter() - started:.1f}s；运行期零磁盘读",
            flush=True,
        )
        return True

    def preload_pinned(self) -> None:
        if not self.preload_all():
            raise RuntimeError(
                "Kimi packed hybrid currently requires all experts in RAM"
            )

    def pin_host_resident(self, budget_gb: float | None = None) -> float:
        del budget_gb
        print(
            "[tpq-kimi] 紧凑 RAM 使用 pageable 索引与有界 pinned staging",
            flush=True,
        )
        return 0.0

    def _safe_budget(self) -> int:
        allocated = torch.cuda.memory_allocated(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        reserve = int(
            float(os.environ.get("TPQ_VRAM_RUNTIME_GB", "3.0")) * 2**30
        )
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device()
        try:
            fraction = torch.cuda.get_per_process_memory_fraction(index)
        except (AttributeError, RuntimeError):
            fraction = 1.0
        process_room = max(0, int(total * fraction) - allocated - reserve)
        device_room = max(0, free - reserve)
        return max(0, min(self.budget, process_room, device_room))

    def build_gpu_arenas(self) -> float:
        if self._arenas is not None:
            return self._arenas.nbytes / 2**30
        if not self.pinned:
            return 0.0
        safe_budget = self._safe_budget()
        if safe_budget <= 0:
            raise RuntimeError("Kimi packed hybrid has no safe GPU cache room")
        counts = Counter(
            PackedExpertSignature.of(expert)
            for expert in self.pinned.values()
        )
        host_codebooks = {
            weight.cb.data_ptr(): weight.cb
            for expert in self.pinned.values()
            for weight in expert
        }
        codebook_bytes = sum(
            codebook.nbytes
            for codebook in host_codebooks.values()
        ) if self._resident_codebooks else 0
        arena_budget = safe_budget - codebook_bytes
        if arena_budget <= 0:
            raise RuntimeError(
                "Kimi packed GPU cache cannot fit resident codebooks"
            )
        top_k = int(self.store.cfg["top_k"])
        weights = None
        if self.slot_mix is not None:
            tier_totals = Counter(
                self._signature_tier(signature)
                for signature, count in counts.items()
                for _ in range(count)
            )
            weights = {
                signature: (
                    self.slot_mix.get(
                        self._signature_tier(signature),
                        0.0,
                    )
                    * count
                    / max(
                        1,
                        tier_totals[self._signature_tier(signature)],
                    )
                )
                for signature, count in counts.items()
            }
        specs = allocate_packed_slots(
            counts,
            arena_budget,
            top_k,
            weights=weights,
            resident_codebooks=self._resident_codebooks,
        )
        self.budget = safe_budget
        self._arenas = _PackedArenas(
            specs,
            self.device,
            resident_codebooks=self._resident_codebooks,
        )
        if self._resident_codebooks:
            self._device_codebooks = {
                pointer: codebook.to(
                    device=self.device,
                    dtype=torch.bfloat16,
                    non_blocking=False,
                )
                for pointer, codebook in host_codebooks.items()
            }
        self.arena_slots = {}
        for signature, count in specs.items():
            tier = self._signature_tier(signature)
            self.arena_slots[tier] = (
                self.arena_slots.get(tier, 0) + count
            )

        intermediate = int(self.store.cfg["moe_inter"])
        hidden = int(self.store.cfg["routed_hidden"])
        self._route_ids = torch.arange(
            top_k,
            dtype=torch.long,
            device=self.device,
        )
        self._ordered_weights = torch.empty(
            top_k,
            dtype=torch.float32,
            device=self.device,
        )
        self._metadata = torch.empty(
            10,
            top_k,
            dtype=torch.long,
            device=self.device,
        )
        self._workspaces = (
            torch.empty(
                top_k,
                2 * intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.empty(
                top_k,
                hidden,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.empty(hidden, dtype=torch.float32, device=self.device),
        )
        self.bytes = self.gpu_storage_bytes
        detail = ", ".join(
            f"{tier}={count}"
            for tier, count in sorted(self.arena_slots.items())
        )
        print(
            f"[tpq-kimi] 紧凑专家固定显存槽："
            f"{sum(specs.values())} 个 / "
            f"{self.gpu_arena_bytes / 2**30:.2f}GiB（{detail}；"
            f"码本={'全局常驻' if self._resident_codebooks else '随槽复制'}）",
            flush=True,
        )
        return self.gpu_arena_bytes / 2**30

    def _copy_pairs(
        self,
        host: PackedExpert,
        device: DeviceExpert,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        pairs = [
            (host[0].raw, device[0].raw),
            (host[1].raw, device[1].raw),
        ]
        if not self._resident_codebooks:
            pairs.extend(
                (
                    (host[0].cb, device[0].cb),
                    (host[1].cb, device[1].cb),
                )
            )
        return pairs

    def _ensure_locked(
        self,
        keys: list[tuple[int, int]],
        *,
        prefetch: bool,
    ) -> dict[tuple[int, int], DeviceExpert]:
        if self._arenas is None:
            raise RuntimeError("Kimi packed GPU arenas are not initialized")
        output: dict[tuple[int, int], DeviceExpert] = {}
        missing: list[tuple[int, int]] = []
        with self._lock:
            for key in keys:
                value = self.cache.get(key)
                if value is None:
                    missing.append(key)
                    continue
                self.cache.move_to_end(key)
                self._arenas.touch(key)
                output[key] = value
                if prefetch:
                    self.prefetch_hits += 1
                else:
                    self.hits += 1
        if not missing:
            self.last_transfer_seconds = 0.0
            return output

        started = time.perf_counter()
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        staged: list[tuple[tuple[int, int], DeviceExpert]] = []
        with self._lock:
            for key in missing:
                # 可能已被前一个等待者装入。
                value = self.cache.get(key)
                if value is not None:
                    self.cache.move_to_end(key)
                    self._arenas.touch(key)
                    output[key] = value
                    continue
                host = self.pinned.get(key)
                if host is None:
                    raise KeyError(f"Kimi packed RAM expert missing: {key}")
                replaced, value = self._arenas.lease(
                    key,
                    host,
                    self._device_codebooks,
                )
                if replaced is not None:
                    self.cache.pop(replaced, None)
                pairs.extend(self._copy_pairs(host, value))
                staged.append((key, value))
        if pairs:
            self._stage.upload_batch(pairs)
            self._stage.last.synchronize()
            uploaded = sum(source.nbytes for source, _target in pairs)
            with self._lock:
                for key, value in staged:
                    self.cache[key] = value
                    output[key] = value
                    if not prefetch:
                        self.miss += 1
                self.uploaded_bytes += uploaded
        elapsed = time.perf_counter() - started
        self.last_transfer_seconds = elapsed
        self.transfer_seconds += elapsed
        return output

    def _ensure(
        self,
        keys: list[tuple[int, int]],
        *,
        prefetch: bool,
    ) -> dict[tuple[int, int], DeviceExpert]:
        with self._transfer_lock:
            return self._ensure_locked(keys, prefetch=prefetch)

    def prefetch(self, keys: list[tuple[int, int]]) -> None:
        if not keys or os.environ.get("TPQ_PREFETCH_STAGE", "1") == "0":
            return
        with self._lock:
            self._prefetch_futures = {
                future
                for future in self._prefetch_futures
                if not future.done()
            }
            # 模型在 token 开始时按层提交约 92 个请求；单线程执行保证
            # 槽位顺序，队列本身必须能容纳一整轮，否则只会预取浅层。
            if len(self._prefetch_futures) >= 128:
                return
            if all(key in self.cache for key in keys):
                return
            future = self._prefetch_executor.submit(
                self._ensure,
                list(keys),
                prefetch=True,
            )
            self._prefetch_futures.add(future)

    def get_many(
        self,
        keys: list[tuple[int, int]],
    ) -> dict[tuple[int, int], DeviceExpert]:
        return self._ensure(keys, prefetch=False)

    def last_expert_ids(self, layer: int) -> list[int]:
        return self._last_ids[layer]

    @staticmethod
    def _metadata_rows(experts: list[DeviceExpert]) -> list[list[int]]:
        return [
            [expert[0].raw.data_ptr() for expert in experts],
            [expert[0].cb.data_ptr() for expert in experts],
            [expert[0].blocks for expert in experts],
            [expert[0].dim for expert in experts],
            [expert[0].dtype_tag for expert in experts],
            [expert[1].raw.data_ptr() for expert in experts],
            [expert[1].cb.data_ptr() for expert in experts],
            [expert[1].blocks for expert in experts],
            [expert[1].dim for expert in experts],
            [expert[1].dtype_tag for expert in experts],
        ]

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
    ) -> torch.Tensor:
        if (
            self._workspaces is None
            or self._metadata is None
            or self._route_ids is None
            or self._ordered_weights is None
        ):
            raise RuntimeError("Kimi packed hybrid pool is not ready")
        # 这是 RAM 地址选择所需的唯一 GPU→CPU 路由同步；权重计算仍留在 GPU。
        expert_ids = [int(item) for item in route_ids.reshape(-1).tolist()]
        self._last_ids[layer] = expert_ids
        keys = [(layer, expert_id) for expert_id in expert_ids]

        # 在融合核排入默认流之前不允许后台预取复用槽位。释放锁后，
        # PinnedStage 的 wait_stream 会把后续覆盖排在本核之后。
        with self._transfer_lock:
            selected = self._ensure_locked(keys, prefetch=False)
            if self._protect_previous:
                old_keys = self._protected_by_layer.get(layer, ())
                current = set(keys)
                for key in old_keys:
                    if key not in current:
                        self._arenas.unprotect(key)
                for key in keys:
                    self._arenas.protect(key)
                self._protected_by_layer[layer] = tuple(keys)
            experts = [selected[key] for key in keys]
            metadata_cpu = torch.tensor(
                self._metadata_rows(experts),
                dtype=torch.long,
            )
            self._metadata[:, : len(experts)].copy_(
                metadata_cpu,
                non_blocking=False,
            )
            p12_positions = [
                position
                for position, expert in enumerate(experts)
                if (
                    expert[0].bits == 12
                    and expert[0].dim in (4, 8)
                    and expert[1].bits == 12
                    and expert[1].dim in (4, 8)
                )
            ]
            generic_positions = [
                position
                for position in range(len(experts))
                if position not in p12_positions
            ]
            order = torch.tensor(
                p12_positions + generic_positions,
                dtype=torch.long,
            )
            self._route_ids[: len(experts)].copy_(
                order,
                non_blocking=False,
            )
            torch.index_select(
                route_weights.reshape(-1).float().contiguous(),
                0,
                self._route_ids[: len(experts)],
                out=self._ordered_weights[: len(experts)],
            )
            hidden, output, result = self._workspaces
            from .ops import packed_moe_topk

            computed = packed_moe_topk(
                value.to(torch.bfloat16),
                self._route_ids[: len(experts)],
                self._ordered_weights[: len(experts)],
                self._metadata[:, : len(experts)],
                activation=activation,
                activation_beta=float(activation_beta),
                activation_linear_beta=(
                    0.0
                    if activation_linear_beta is None
                    else float(activation_linear_beta)
                ),
                hidden_workspace=hidden[: len(experts)],
                output_workspace=output[: len(experts)],
                result=result,
                grouped_prefix=len(p12_positions),
            )
            return computed

    def resize_gpu_arenas(
        self,
        budget: int,
        *,
        staging_timeout_s: float = 30.0,
    ) -> tuple[int, int]:
        del staging_timeout_s
        budget = max(0, int(budget))
        old = self.gpu_arena_bytes
        if budget >= old:
            self.budget = budget
            return old, old
        with self._transfer_lock:
            torch.cuda.synchronize(self.device)
            self.cache.clear()
            self._arenas = None
            self._device_codebooks = {}
            self._protected_by_layer.clear()
            self._workspaces = None
            self._metadata = None
            self._route_ids = None
            self._ordered_weights = None
            self.bytes = 0
            self.budget = budget
            gc.collect()
            torch.cuda.empty_cache()
            self.build_gpu_arenas()
        return old, self.gpu_arena_bytes

    def trim_to(self, budget: int) -> None:
        budget = max(0, int(budget))
        if self.gpu_arena_bytes > budget:
            self.resize_gpu_arenas(budget)
        else:
            self.budget = budget


__all__ = [
    "HostPackedWeight",
    "KimiPackedHybridPool",
    "PackedExpertSignature",
    "allocate_packed_slots",
]
