"""配置驱动的单卡紧凑专家 RAM+GPU 执行池。

与通用 ``ExpertPool`` 的主要区别：

* 9..15-bit 索引在 RAM 中保持 CCCP 原始打包格式，不展开为 uint16；
* 按专家签名预分配稳定 GPU 槽，换专家只覆盖槽内容；
* 上一个 token 的路由在后台预取，需求路径按层等待；
* Gate/Up、gated activation、Down、路由加权直接使用公共融合 CUDA 核。

非专家 dense、注意力、共享专家和 KV 路径完全不变。文件名保留用于导入兼容；
实现按 projection-VQ 清单和算子能力分派，同时服务 Kimi 与 DeepSeek-V4。
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
from .store import TPQStore, PackedVQWeight, PinnedStage


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
        return {
            8: 0, 16: 1, 12: 2, 14: 3, 10: 4, 9: 5,
            11: 6, 13: 7, 15: 8,
        }[
            self.bits
        ]

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
        return {
            8: 0, 16: 1, 12: 2, 14: 3, 10: 4, 9: 5,
            11: 6, 13: 7, 15: 8,
        }[
            self.bits
        ]

    @property
    def nbytes(self) -> int:
        return self.raw.nbytes + self.cb.nbytes


PackedExpert = tuple[HostPackedWeight, ...]
DeviceExpert = tuple[DevicePackedWeight, ...]


@dataclass
class PendingPackedRun:
    """A staged packed-MoE call whose arena slots remain exclusively leased."""

    layer: int
    value: torch.Tensor
    expert_count: int
    grouped_prefix: int
    activation: str
    activation_beta: float
    activation_linear_beta: float | None
    limit: float
    wait_for_stage: bool
    route_order: torch.Tensor | None = None
    ordered_weights: torch.Tensor | None = None
    metadata: torch.Tensor | None = None
    active: bool = True


@dataclass(frozen=True)
class PackedRoutePlan:
    """Device-resident metadata for an immediately reusable Top-K route."""

    expert_ids: tuple[int, ...]
    keys: tuple[tuple[int, int], ...]
    experts: tuple[DeviceExpert, ...]
    order: torch.Tensor
    metadata: torch.Tensor
    grouped_prefix: int
    identity_order: bool


@dataclass(frozen=True)
class PackedWeightSignature:
    raw_bytes: int
    cb_shape: tuple[int, int]
    rows: int
    cols: int
    blocks: int
    dim: int
    bits: int

    @classmethod
    def of(cls, weight: HostPackedWeight) -> "PackedWeightSignature":
        return cls(
            raw_bytes=weight.raw.numel(),
            cb_shape=tuple(weight.cb.shape),
            rows=weight.rows,
            cols=weight.cols,
            blocks=weight.blocks,
            dim=weight.dim,
            bits=weight.bits,
        )


@dataclass(frozen=True)
class PackedExpertSignature:
    """Shape-only cache key for either GU+Down or Gate+Up+Down experts."""

    weights: tuple[PackedWeightSignature, ...]

    @classmethod
    def of(cls, expert: PackedExpert) -> "PackedExpertSignature":
        if len(expert) not in (2, 3):
            raise ValueError(
                "packed expert must contain GU+Down or Gate+Up+Down"
            )
        return cls(tuple(PackedWeightSignature.of(weight) for weight in expert))

    @property
    def projection_count(self) -> int:
        return len(self.weights)

    @property
    def gu(self) -> PackedWeightSignature:
        return self.weights[0]

    @property
    def up(self) -> PackedWeightSignature | None:
        return self.weights[1] if len(self.weights) == 3 else None

    @property
    def down(self) -> PackedWeightSignature:
        return self.weights[-1]

    # Compatibility properties retain the public diagnostics/tests used by the
    # original two-projection archive while the backing key is projection-count
    # agnostic.
    gu_raw_bytes = property(lambda self: self.gu.raw_bytes)
    gu_cb_shape = property(lambda self: self.gu.cb_shape)
    gu_rows = property(lambda self: self.gu.rows)
    gu_cols = property(lambda self: self.gu.cols)
    gu_blocks = property(lambda self: self.gu.blocks)
    gu_dim = property(lambda self: self.gu.dim)
    gu_bits = property(lambda self: self.gu.bits)
    down_raw_bytes = property(lambda self: self.down.raw_bytes)
    down_cb_shape = property(lambda self: self.down.cb_shape)
    down_rows = property(lambda self: self.down.rows)
    down_cols = property(lambda self: self.down.cols)
    down_blocks = property(lambda self: self.down.blocks)
    down_dim = property(lambda self: self.down.dim)
    down_bits = property(lambda self: self.down.bits)

    @property
    def raw_slot_bytes(self) -> int:
        return sum(weight.raw_bytes for weight in self.weights)

    @property
    def codebook_slot_bytes(self) -> int:
        return sum(
            weight.cb_shape[0] * weight.cb_shape[1]
            for weight in self.weights
        ) * torch.bfloat16.itemsize

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
            "packed GPU cache is too small for one complete Top-K of "
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
            raise RuntimeError("cannot fit minimum packed GPU slots")
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
        self.raw = tuple(
            torch.empty(
                count,
                weight.raw_bytes,
                dtype=torch.uint8,
                device=device,
            )
            for weight in signature.weights
        )
        self.codebooks: tuple[torch.Tensor, ...] | None = None
        if not resident_codebooks:
            self.codebooks = tuple(
                torch.empty(
                    count,
                    *weight.cb_shape,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for weight in signature.weights
            )

    @property
    def nbytes(self) -> int:
        output = sum(tensor.nbytes for tensor in self.raw)
        if self.codebooks is not None:
            output += sum(tensor.nbytes for tensor in self.codebooks)
        return output

    def lease(
        self,
        key: tuple[int, int],
        codebooks: tuple[torch.Tensor, ...],
    ) -> tuple[object, DeviceExpert]:
        lease = self.book.acquire(key)
        slot = lease.slot
        signature = self.signature
        if not self.resident_codebooks:
            if self.codebooks is None:
                raise RuntimeError("packed slot codebook storage is missing")
            codebooks = tuple(
                storage[slot]
                for storage in self.codebooks
            )
        if len(codebooks) != signature.projection_count:
            raise ValueError("packed expert codebook count mismatch")
        return lease, tuple(
            DevicePackedWeight(
                self.raw[index][slot],
                codebooks[index],
                weight.rows,
                weight.cols,
                weight.blocks,
                weight.dim,
                weight.bits,
            )
            for index, weight in enumerate(signature.weights)
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
        codebooks = tuple(weight.cb for weight in expert)
        if arena.resident_codebooks:
            codebooks = tuple(
                device_codebooks[weight.cb.data_ptr()]
                for weight in expert
            )
        lease, device_expert = arena.lease(
            key,
            codebooks,
        )
        replaced = lease.replaced
        if replaced is not None:
            self.leases.pop(replaced, None)
        self.leases[key] = (signature, lease)
        return replaced, device_expert


class PackedHybridPool:
    """全量紧凑 RAM + 有界稳定 VRAM 的配置驱动 Top-K 专家池。"""

    device_routed = True
    full_resident = False
    prefetch_default = False

    def __init__(
        self,
        store: TPQStore,
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
            tuple[str, str, str],
            torch.Tensor,
        ] = {}
        self._device_codebooks: dict[int, torch.Tensor] = {}
        self._host_pinned_bytes = 0
        self._host_registrations: dict[int, int] = {}
        self._resident_codebooks = (
            os.environ.get("TPQ_KIMI_RESIDENT_CODEBOOKS", "1") != "0"
        )
        self._protect_previous = (
            os.environ.get("TPQ_KIMI_PROTECT_PREV", "0") != "0"
        )
        self._arenas: _PackedArenas | None = None
        self._stage = PinnedStage(self.device, measure=True)
        self._lock = threading.RLock()
        self._transfer_lock = threading.RLock()
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tpq-packed-prefetch",
        )
        self._prefetch_futures: set[Future] = set()
        self._last_ids: dict[int, list[int]] = {}
        self._protected_by_layer: dict[
            int,
            tuple[tuple[int, int], ...],
        ] = {}
        self._route_plans: dict[int, PackedRoutePlan] = {}
        self.route_plan_hits = 0
        self.route_plan_misses = 0
        self._route_ids: torch.Tensor | None = None
        self._route_order_identity = False
        self._ordered_weights: torch.Tensor | None = None
        self._metadata: torch.Tensor | None = None
        self._slot_directory: torch.Tensor | None = None
        self._slot_update_host: torch.Tensor | None = None
        self._route_hit_mask: torch.Tensor | None = None
        self._route_all_hit: torch.Tensor | None = None
        self._route_all_hit_host: torch.Tensor | None = None
        self._route_host_ids: torch.Tensor | None = None
        self._route_copy_done: torch.cuda.Event | None = None
        self.device_route_lookups = 0
        self.device_route_full_hits = 0
        self.device_route_fallbacks = 0
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
                raise ValueError(f"unknown packed slot tier: {name}")
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
            weight.raw.nbytes
            for expert in self.pinned.values()
            for weight in expert
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
        if self._slot_directory is not None:
            workspace += self._slot_directory.nbytes
        if self._route_hit_mask is not None:
            workspace += self._route_hit_mask.nbytes
        if self._route_all_hit is not None:
            workspace += self._route_all_hit.nbytes
        if self._route_ids is not None:
            workspace += self._route_ids.nbytes
        if self._ordered_weights is not None:
            workspace += self._ordered_weights.nbytes
        workspace += sum(
            plan.order.nbytes + plan.metadata.nbytes
            for plan in self._route_plans.values()
        )
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
        key: tuple[str, str, str],
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
        packed = self.store.load_expert_packed(layer, expert_id)
        if self.store.man.projection_vq:
            variants = self.store.projection_codebook_variants(
                layer,
                expert_id,
            )
            names = ("gate", "up", "down")
            if len(packed) != 3 or len(variants) != 3:
                raise ValueError(
                    f"L{layer} projection-VQ expert must have three weights"
                )
            return tuple(
                HostPackedWeight.from_store(
                    weight,
                    self._host_codebook(
                        ("projection-vq", variant, projection),
                        weight.cb,
                    ),
                )
                for projection, variant, weight in zip(
                    names,
                    variants,
                    packed,
                )
            )

        gu, down = packed
        tier = self.store.expert_kind(layer, expert_id).rstrip("z")
        codebook_variants = self.store.codebook_variants(
            layer,
            tier,
            expert_id,
        )
        return (
            HostPackedWeight(
                gu.raw,
                self._host_codebook(
                    (tier, codebook_variants[0], "gu"),
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
                    (tier, codebook_variants[1], "down"),
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
        native_expert_bytes = getattr(self.store, "expert_bytes", None)
        if native_expert_bytes is None:
            expert_files = [
                os.path.join(self.store.root, filename)
                for filename in self.store.man.expert_files.values()
            ]
            stored_bytes = sum(
                os.path.getsize(path)
                for path in expert_files
                if os.path.exists(path)
            )
        else:
            stored_bytes = int(native_expert_bytes)
        available = psutil.virtual_memory().available
        if stored_bytes + int(reserve_gb * 2**30) > available:
            print(
                "[tpq-packed] 紧凑专家无法全量常驻 RAM："
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
        if self.store.man.no_expert_drop:
            declared_layers = self.store.man.routed_layers
            if declared_layers and declared_layers != len(
                self.store.man.expert_files
            ):
                raise RuntimeError(
                    "projection-VQ CCCP 专家清单未收敛："
                    f"声明 {declared_layers} 层，"
                    f"实际只有 {len(self.store.man.expert_files)} 层"
                )
            declared_experts = (
                self.store.man.routed_experts_per_layer or n_experts
            )
            expected = len(self.store.man.expert_files) * declared_experts
            if declared_experts != n_experts or len(keys) != expected:
                present = set(keys)
                missing = [
                    f"L{layer}/e{expert_id}"
                    for layer in sorted(self.store.man.expert_files)
                    for expert_id in range(n_experts)
                    if (layer, expert_id) not in present
                ][:8]
                raise RuntimeError(
                    "projection-VQ no_expert_drop 清单与专家文件不一致："
                    f"期望 {expected}，实际 {len(keys)}"
                    + (f"，缺失示例 {', '.join(missing)}" if missing else "")
                )
        workers = max(1, int(os.environ.get("TPQ_LOAD_WORKERS", "12")))
        started = time.perf_counter()
        print(
            f"[tpq-packed] 紧凑专家常驻 RAM：{len(keys)} 个，"
            f"文件约 {stored_bytes / 2**30:.1f}GiB，workers={workers}",
            flush=True,
        )
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="tpq-packed-load",
        ) as executor:
            futures = {
                executor.submit(self._load_one, *key): key
                for key in keys
            }
            for index, future in enumerate(as_completed(futures), 1):
                self.pinned[futures[future]] = future.result()
                if index % 2000 == 0:
                    print(
                        f"[tpq-packed] 紧凑专家常驻 "
                        f"{index}/{len(keys)}",
                        flush=True,
                    )
        # 所有运行时专家都只引用 BF16 码本；释放 store 的 FP32 中间副本。
        codebook_cache = getattr(self.store, "_cb_cache", None)
        if codebook_cache is not None:
            codebook_cache.clear()
        gc.collect()
        self.ram_bytes = self.host_expert_bytes
        print(
            f"[tpq-packed] 紧凑专家 RAM 常驻完成："
            f"{len(self.pinned)} 个 / {self.ram_bytes / 2**30:.1f}GiB，"
            f"{time.perf_counter() - started:.1f}s；运行期零磁盘读",
            flush=True,
        )
        return True

    def preload_pinned(self) -> None:
        if not self.preload_all():
            raise RuntimeError(
                "packed hybrid currently requires all experts in RAM"
            )

    def pin_host_resident(self, budget_gb: float | None = None) -> float:
        """Register compact resident payloads in place for direct DMA.

        ``pin_memory()`` would allocate a second copy of a several-hundred-GiB
        expert archive. ``cudaHostRegister`` page-locks the existing bytearray
        storage instead: disk/RAM/VRAM all retain the original packed width and
        ``PinnedStage`` can bypass its small rotating bounce buffers.
        """
        if not self.pinned:
            return 0.0
        total = sum(
            weight.raw.nbytes
            for expert in self.pinned.values()
            for weight in expert
        )
        if budget_gb is None:
            raw = os.environ.get("TPQ_HOST_PIN_GB", "auto").strip().lower()
            if raw in ("", "auto"):
                import psutil

                available = psutil.virtual_memory().available
                reserve = max(64 * 2**30, total // 2)
                if available < reserve:
                    print(
                        "[tpq-packed] 紧凑专家原地锁页自动关闭："
                        f"可用RAM {available / 2**30:.1f}GiB < "
                        f"安全余量 {reserve / 2**30:.1f}GiB",
                        flush=True,
                    )
                    return 0.0
                # CUDA keeps mappings for registered host pages.  Registering
                # the entire archive can exhaust that driver resource before
                # ordinary device allocations report pressure.  Three times
                # physical VRAM stays below the observed mapping boundary while
                # still putting most packed experts on the direct-DMA path.
                device_bytes = torch.cuda.get_device_properties(
                    self.device
                ).total_memory
                multiplier = max(
                    0.0,
                    float(
                        os.environ.get(
                            "TPQ_HOST_PIN_VRAM_MULTIPLIER",
                            "3.0",
                        )
                    ),
                )
                driver_budget = int(device_bytes * multiplier)
                budget = min(total, driver_budget)
                print(
                    "[tpq-packed] 紧凑专家原地锁页自动预算："
                    f"{budget / 2**30:.1f}GiB / "
                    f"总量 {total / 2**30:.1f}GiB（VRAM×{multiplier:g}）",
                    flush=True,
                )
            else:
                budget = max(0, int(float(raw) * 2**30))
        else:
            budget = max(0, int(float(budget_gb) * 2**30))
        if budget <= self._host_pinned_bytes:
            return self._host_pinned_bytes / 2**30

        started = time.perf_counter()
        pinned_experts = 0
        registered_tensors = 0
        stop = False
        cudart = torch.cuda.cudart()
        # Cycle through every layer for the same expert id before moving on.
        # A partial budget therefore accelerates all layers instead of pinning
        # only a shallow contiguous prefix of the network.
        ordered_keys = sorted(
            self.pinned,
            key=lambda key: (key[1], key[0]),
        )
        for key in ordered_keys:
            for weight in self.pinned[key]:
                source = weight.raw
                pointer = source.data_ptr()
                if pointer in self._host_registrations or source.is_pinned():
                    continue
                if self._host_pinned_bytes + source.nbytes > budget:
                    stop = True
                    break
                error = cudart.cudaHostRegister(
                    pointer,
                    source.nbytes,
                    0,
                )
                error_code = getattr(error, "value", None)
                if error_code is None:
                    error_code = int(error)
                if error_code != 0:
                    try:
                        error_name = cudart.cudaGetErrorString(error)
                    except (AttributeError, RuntimeError, TypeError):
                        error_name = error
                    print(
                        "[tpq-packed] 紧凑专家原地锁页停止："
                        f"{self._host_pinned_bytes / 2**30:.1f}GiB，"
                        f"cudaHostRegister={error_name}",
                        flush=True,
                    )
                    stop = True
                    break
                self._host_registrations[pointer] = source.nbytes
                self._host_pinned_bytes += source.nbytes
                registered_tensors += 1
            pinned_experts += 1
            if pinned_experts % 2000 == 0:
                print(
                    "[tpq-packed] 紧凑专家原地锁页 "
                    f"{pinned_experts}/{len(self.pinned)} "
                    f"({self._host_pinned_bytes / 2**30:.1f}GiB)",
                    flush=True,
                )
            if stop:
                break
        print(
            "[tpq-packed] 紧凑专家原地锁页完成："
            f"{registered_tensors} 个 packed 张量 / "
            f"{self._host_pinned_bytes / 2**30:.1f}GiB / "
            f"{time.perf_counter() - started:.1f}s；直接异步 DMA",
            flush=True,
        )
        return self._host_pinned_bytes / 2**30

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
            raise RuntimeError("packed hybrid has no safe GPU cache room")
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
                "packed GPU cache cannot fit resident codebooks"
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

        # Projection-VQ manifests normalize Kimi and DeepSeek dimensions here.
        # Kimi has a separate routed latent width while DeepSeek routes the
        # model hidden state directly, so model-specific fields must not leak
        # into the shared packed arena.
        intermediate = int(self.store.cfg["moe_inter"])
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        self._route_ids = torch.arange(
            top_k,
            dtype=torch.long,
            device=self.device,
        )
        self._route_order_identity = True
        self._ordered_weights = torch.empty(
            top_k,
            dtype=torch.float32,
            device=self.device,
        )
        self._metadata = torch.empty(
            15 if self.store.man.projection_vq else 10,
            top_k,
            dtype=torch.long,
            device=self.device,
        )
        metadata_rows = int(self._metadata.shape[0])
        self._slot_directory = torch.zeros(
            int(self.store.cfg["n_layers"]),
            int(self.store.cfg["n_experts"]),
            metadata_rows,
            dtype=torch.long,
            device=self.device,
        )
        self._slot_update_host = torch.empty(
            metadata_rows,
            dtype=torch.long,
            pin_memory=True,
        )
        self._route_hit_mask = torch.empty(
            top_k,
            dtype=torch.bool,
            device=self.device,
        )
        self._route_all_hit = torch.empty(
            (),
            dtype=torch.bool,
            device=self.device,
        )
        self._route_all_hit_host = torch.empty(
            (),
            dtype=torch.bool,
            pin_memory=True,
        )
        self._route_host_ids = torch.empty(
            top_k,
            dtype=torch.long,
            pin_memory=True,
        )
        self._route_copy_done = torch.cuda.Event()
        self._metadata_host = torch.empty(
            self._metadata.shape,
            dtype=torch.long,
            pin_memory=True,
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
            f"[tpq-packed] 紧凑专家固定显存槽："
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
        if len(host) != len(device):
            raise ValueError("packed host/device projection count mismatch")
        pairs = [
            (source.raw, target.raw)
            for source, target in zip(host, device)
        ]
        if not self._resident_codebooks:
            pairs.extend(
                (source.cb, target.cb)
                for source, target in zip(host, device)
            )
        return pairs

    def _ensure_locked(
        self,
        keys: list[tuple[int, int]],
        *,
        prefetch: bool,
        defer_wait: bool = False,
    ) -> dict[tuple[int, int], DeviceExpert]:
        if self._arenas is None:
            raise RuntimeError("packed GPU arenas are not initialized")
        self.transfer_seconds = self._stage.collect_timing()
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
                    raise KeyError(f"packed RAM expert missing: {key}")
                replaced, value = self._arenas.lease(
                    key,
                    host,
                    self._device_codebooks,
                )
                if replaced is not None:
                    self.cache.pop(replaced, None)
                    self._set_slot_directory(replaced, None)
                pairs.extend(self._copy_pairs(host, value))
                staged.append((key, value))
        if pairs:
            self._stage.upload_batch(pairs)
            if not defer_wait:
                self._stage.last.synchronize()
            uploaded = sum(source.nbytes for source, _target in pairs)
            with self._lock:
                for key, value in staged:
                    self.cache[key] = value
                    output[key] = value
                    self._set_slot_directory(key, value)
                    if not prefetch:
                        self.miss += 1
                self.uploaded_bytes += uploaded
        elapsed = time.perf_counter() - started
        self.last_transfer_seconds = (
            0.0 if defer_wait and pairs else elapsed
        )
        self.transfer_seconds = self._stage.collect_timing()
        return output

    def collect_transfer_timing(self, *, synchronize: bool = False) -> float:
        self.transfer_seconds = self._stage.collect_timing(
            synchronize=synchronize,
        )
        return self.transfer_seconds

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
        if not experts:
            return []
        projection_count = len(experts[0])
        if projection_count not in (2, 3) or any(
            len(expert) != projection_count
            for expert in experts
        ):
            raise ValueError("inconsistent packed expert projection count")
        return [
            values
            for projection in range(projection_count)
            for values in (
                [
                    expert[projection].raw.data_ptr()
                    for expert in experts
                ],
                [
                    expert[projection].cb.data_ptr()
                    for expert in experts
                ],
                [
                    expert[projection].blocks
                    for expert in experts
                ],
                [
                    expert[projection].dim
                    for expert in experts
                ],
                [
                    expert[projection].dtype_tag
                    for expert in experts
                ],
            )
        ]

    def _copy_metadata(self, experts: list[DeviceExpert]) -> None:
        """Queue tiny route metadata without a per-layer host synchronization."""
        count = len(experts)
        rows = self._metadata_rows(experts)
        host = getattr(self, "_metadata_host", None)
        if host is None or host.shape[0] != len(rows) or host.shape[1] < count:
            host = torch.empty(len(rows), count, dtype=torch.long)
        host[:, :count].copy_(torch.tensor(rows, dtype=torch.long))
        self._metadata[:, :count].copy_(
            host[:, :count],
            non_blocking=bool(host.is_pinned()),
        )

    def _set_slot_directory(
        self,
        key: tuple[int, int],
        expert: DeviceExpert | None,
    ) -> None:
        """Publish or invalidate one stable slot in the CUDA directory."""
        if self._slot_directory is None:
            return
        layer, expert_id = key
        target = self._slot_directory[layer, expert_id]
        if expert is None:
            target.zero_()
            return
        rows = self._metadata_rows([expert])
        values = [row[0] for row in rows]
        host = self._slot_update_host
        if host is None or host.numel() != len(values):
            host = torch.empty(len(values), dtype=torch.long)
        host.copy_(torch.tensor(values, dtype=torch.long))
        target.copy_(host, non_blocking=bool(host.is_pinned()))

    def _device_route_metadata(
        self,
        layer: int,
        route_ids: torch.Tensor,
    ) -> tuple[bool, list[int] | None]:
        """Gather route metadata on CUDA and copy IDs only for a miss path.

        The fixed metadata directory is the source of truth for resident-slot
        hits.  One small pinned result is synchronized after the CUDA gather;
        Python no longer performs ``Tensor.tolist()`` or per-expert dictionary
        mapping on the normal all-hit path.
        """
        if (
            getattr(self, "_slot_directory", None) is None
            or getattr(self, "_metadata", None) is None
            or getattr(self, "_route_hit_mask", None) is None
            or getattr(self, "_route_all_hit", None) is None
            or getattr(self, "_route_all_hit_host", None) is None
            or getattr(self, "_route_host_ids", None) is None
            or getattr(self, "_route_copy_done", None) is None
        ):
            return False, None
        flat_ids = route_ids.reshape(-1)
        count = int(flat_ids.numel())
        from .ops import packed_route_slots

        if not packed_route_slots(
            flat_ids,
            self._slot_directory[int(layer)],
            output=self._metadata[:, :count],
            hit_mask=self._route_hit_mask[:count],
        ):
            return False, None
        torch.all(
            self._route_hit_mask[:count],
            out=self._route_all_hit,
        )
        self._route_all_hit_host.copy_(
            self._route_all_hit,
            non_blocking=True,
        )
        # Copy the tiny Top-K vector in the same synchronization window.  It
        # is consumed only if CUDA reports at least one non-resident expert.
        self._route_host_ids[:count].copy_(flat_ids, non_blocking=True)
        self._route_copy_done.record(torch.cuda.current_stream(self.device))
        self._route_copy_done.synchronize()
        self.device_route_lookups += 1
        if bool(self._route_all_hit_host):
            self.device_route_full_hits += 1
            self.hits += count
            self.last_transfer_seconds = 0.0
            if os.environ.get("TPQ_PREFETCH", "0") != "0":
                self._last_ids[int(layer)] = [
                    int(self._route_host_ids[index])
                    for index in range(count)
                ]
            return True, None
        self.device_route_fallbacks += 1
        return False, [
            int(self._route_host_ids[index])
            for index in range(count)
        ]

    def _host_route_ids(self, route_ids: torch.Tensor) -> list[int]:
        """Compatibility fallback without ``Tensor.tolist()``."""
        flat_ids = route_ids.reshape(-1)
        count = int(flat_ids.numel())
        if (
            flat_ids.is_cuda
            and getattr(self, "_route_host_ids", None) is not None
            and getattr(self, "_route_copy_done", None) is not None
        ):
            self._route_host_ids[:count].copy_(flat_ids, non_blocking=True)
            self._route_copy_done.record(
                torch.cuda.current_stream(self.device)
            )
            self._route_copy_done.synchronize()
            source = self._route_host_ids
        else:
            source = flat_ids.detach().cpu()
        return [int(source[index]) for index in range(count)]

    def _set_route_order(
        self,
        p12_positions: list[int],
        generic_positions: list[int],
    ) -> bool:
        """Update packed dispatch order only when it is not already identity.

        Projection-VQ archives use three weights per expert and therefore do
        not enter the legacy two-weight p12 prefix.  Their order is the fixed
        ``arange(top_k)`` allocated with the arena; re-uploading that same
        128-byte tensor once per layer introduced a needless CUDA host sync.
        """
        order_values = p12_positions + generic_positions
        identity = all(
            position == value
            for position, value in enumerate(order_values)
        )
        if not identity or not getattr(
            self,
            "_route_order_identity",
            False,
        ):
            order = torch.tensor(order_values, dtype=torch.long)
            self._route_ids[: len(order_values)].copy_(
                order,
                non_blocking=False,
            )
        self._route_order_identity = identity
        return identity

    def prepare_run(
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
    ) -> PendingPackedRun:
        """Start expert DMA while retaining exclusive ownership of its slots.

        Independent default-stream work may be enqueued before ``finish_run``.
        The packed kernel itself waits on the copy-stream event, so the arena is
        never read early and a prefetch thread cannot replace a leased slot.
        """
        if (
            self._workspaces is None
            or self._metadata is None
            or self._route_ids is None
            or self._ordered_weights is None
        ):
            raise RuntimeError("packed hybrid pool is not ready")
        self._transfer_lock.acquire()
        try:
            device_hit, expert_ids = self._device_route_metadata(
                layer,
                route_ids,
            )
            if device_hit:
                count = int(route_ids.numel())
                return PendingPackedRun(
                    layer=layer,
                    value=value,
                    expert_count=count,
                    grouped_prefix=-1,
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=activation_linear_beta,
                    limit=float(limit),
                    wait_for_stage=False,
                    route_order=self._route_ids[:count],
                    ordered_weights=(
                        route_weights.reshape(-1).float().contiguous()
                    ),
                    metadata=self._metadata[:, :count],
                )
            if expert_ids is None:
                expert_ids = self._host_route_ids(route_ids)
            plan = self._find_route_plan(layer, expert_ids)
            if plan is not None:
                if plan.identity_order:
                    ordered_weights = (
                        route_weights.reshape(-1).float().contiguous()
                    )
                else:
                    torch.index_select(
                        route_weights.reshape(-1).float().contiguous(),
                        0,
                        plan.order,
                        out=self._ordered_weights[: len(expert_ids)],
                    )
                    ordered_weights = self._ordered_weights[
                        : len(expert_ids)
                    ]
                return PendingPackedRun(
                    layer=layer,
                    value=value,
                    expert_count=len(expert_ids),
                    grouped_prefix=plan.grouped_prefix,
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=activation_linear_beta,
                    limit=float(limit),
                    wait_for_stage=False,
                    route_order=plan.order,
                    ordered_weights=ordered_weights,
                    metadata=plan.metadata,
                )
            self._last_ids[layer] = expert_ids
            keys = [(layer, expert_id) for expert_id in expert_ids]
            uploaded_before = self.uploaded_bytes
            selected = self._ensure_locked(
                keys,
                prefetch=False,
                defer_wait=True,
            )
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
            self._copy_metadata(experts)
            p12_positions = [
                position
                for position, expert in enumerate(experts)
                if (
                    len(expert) == 2
                    and expert[0].bits == 12
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
            identity_order = self._set_route_order(
                p12_positions,
                generic_positions,
            )
            if identity_order:
                ordered_weights = (
                    route_weights.reshape(-1).float().contiguous()
                )
            else:
                torch.index_select(
                    route_weights.reshape(-1).float().contiguous(),
                    0,
                    self._route_ids[: len(experts)],
                    out=self._ordered_weights[: len(experts)],
                )
                ordered_weights = self._ordered_weights[: len(experts)]
            self._save_route_plan(
                layer,
                expert_ids,
                keys,
                experts,
                len(p12_positions),
                identity_order,
            )
            return PendingPackedRun(
                layer=layer,
                value=value,
                expert_count=len(experts),
                grouped_prefix=len(p12_positions),
                activation=activation,
                activation_beta=float(activation_beta),
                activation_linear_beta=activation_linear_beta,
                limit=float(limit),
                wait_for_stage=self.uploaded_bytes > uploaded_before,
                route_order=self._route_ids[: len(experts)],
                ordered_weights=ordered_weights,
                metadata=self._metadata[:, : len(experts)],
            )
        except BaseException:
            self._transfer_lock.release()
            raise

    def finish_run(self, pending: PendingPackedRun) -> torch.Tensor:
        if not pending.active:
            raise RuntimeError("packed MoE pending call is no longer active")
        try:
            if pending.wait_for_stage:
                self._stage.wait()
            hidden, output, result = self._workspaces
            from .ops import packed_moe_topk

            route_order = (
                self._route_ids[: pending.expert_count]
                if pending.route_order is None
                else pending.route_order
            )
            ordered_weights = (
                self._ordered_weights[: pending.expert_count]
                if pending.ordered_weights is None
                else pending.ordered_weights
            )
            metadata = (
                self._metadata[:, : pending.expert_count]
                if pending.metadata is None
                else pending.metadata
            )

            return packed_moe_topk(
                pending.value.to(torch.bfloat16),
                route_order,
                ordered_weights,
                metadata,
                activation=pending.activation,
                activation_beta=pending.activation_beta,
                activation_linear_beta=(
                    0.0
                    if pending.activation_linear_beta is None
                    else float(pending.activation_linear_beta)
                ),
                limit=pending.limit,
                hidden_workspace=hidden[: pending.expert_count],
                output_workspace=output[: pending.expert_count],
                result=result,
                grouped_prefix=pending.grouped_prefix,
                **self.store.man.projection_operator_capability(
                    pending.layer
                ),
            )
        finally:
            pending.active = False
            self._transfer_lock.release()

    def cancel_run(self, pending: PendingPackedRun) -> None:
        if pending.active:
            pending.active = False
            self._transfer_lock.release()

    def _find_route_plan(
        self,
        layer: int,
        expert_ids: list[int],
    ) -> PackedRoutePlan | None:
        """Reuse device pointer metadata while its arena leases stay valid."""
        if os.environ.get("TPQ_KIMI_ROUTE_PLAN", "1") == "0":
            return None
        plan = self._route_plans.get(int(layer))
        if plan is None or tuple(expert_ids) != plan.expert_ids:
            self.route_plan_misses += 1
            return None
        with self._lock:
            valid = all(
                self.cache.get(key) is expert
                for key, expert in zip(plan.keys, plan.experts)
            )
            if valid:
                for key in plan.keys:
                    self.cache.move_to_end(key)
                    self._arenas.touch(key)
                    self.hits += 1
        if not valid:
            self.route_plan_misses += 1
            return None
        self._last_ids[int(layer)] = list(plan.expert_ids)
        self.last_transfer_seconds = 0.0
        self.route_plan_hits += 1
        return plan

    def _reuse_route_plan(
        self,
        layer: int,
        value: torch.Tensor,
        expert_ids: list[int],
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> torch.Tensor | None:
        plan = self._find_route_plan(layer, expert_ids)
        if plan is None:
            return None
        if plan.identity_order:
            ordered_weights = route_weights.reshape(-1).float().contiguous()
        else:
            torch.index_select(
                route_weights.reshape(-1).float().contiguous(),
                0,
                plan.order,
                out=self._ordered_weights[: len(plan.expert_ids)],
            )
            ordered_weights = self._ordered_weights[: len(plan.expert_ids)]
        hidden, output, result = self._workspaces
        from .ops import packed_moe_topk

        return packed_moe_topk(
            value.to(torch.bfloat16),
            plan.order,
            ordered_weights,
            plan.metadata,
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=(
                0.0
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
            limit=float(limit),
            hidden_workspace=hidden[: len(plan.expert_ids)],
            output_workspace=output[: len(plan.expert_ids)],
            result=result,
            grouped_prefix=plan.grouped_prefix,
            **self.store.man.projection_operator_capability(layer),
        )

    def _save_route_plan(
        self,
        layer: int,
        expert_ids: list[int],
        keys: list[tuple[int, int]],
        experts: list[DeviceExpert],
        grouped_prefix: int,
        identity_order: bool,
    ) -> None:
        if os.environ.get("TPQ_KIMI_ROUTE_PLAN", "1") == "0":
            return
        count = len(expert_ids)
        previous = self._route_plans.get(int(layer))
        if previous is None or previous.order.numel() != count:
            saved_order = torch.empty(
                count,
                dtype=torch.long,
                device=self.device,
            )
            saved_metadata = torch.empty(
                self._metadata.shape[0],
                count,
                dtype=torch.long,
                device=self.device,
            )
        else:
            saved_order = previous.order
            saved_metadata = previous.metadata
        saved_order.copy_(
            self._route_ids[:count],
            non_blocking=False,
        )
        saved_metadata.copy_(
            self._metadata[:, :count],
            non_blocking=False,
        )
        self._route_plans[int(layer)] = PackedRoutePlan(
            expert_ids=tuple(expert_ids),
            keys=tuple(keys),
            experts=tuple(experts),
            order=saved_order,
            metadata=saved_metadata,
            grouped_prefix=int(grouped_prefix),
            identity_order=bool(identity_order),
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
        if (
            self._workspaces is None
            or self._metadata is None
            or self._route_ids is None
            or self._ordered_weights is None
        ):
            raise RuntimeError("packed hybrid pool is not ready")
        # 这是 RAM 地址选择所需的唯一 GPU→CPU 路由同步；权重计算仍留在 GPU。
        # 在融合核排入默认流之前不允许后台预取复用槽位。释放锁后，
        # PinnedStage 的 wait_stream 会把后续覆盖排在本核之后。
        with self._transfer_lock:
            device_hit, expert_ids = self._device_route_metadata(
                layer,
                route_ids,
            )
            if device_hit:
                count = int(route_ids.numel())
                hidden, output, result = self._workspaces
                from .ops import packed_moe_topk

                return packed_moe_topk(
                    value.to(torch.bfloat16),
                    self._route_ids[:count],
                    route_weights.reshape(-1).float().contiguous(),
                    self._metadata[:, :count],
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=(
                        0.0
                        if activation_linear_beta is None
                        else float(activation_linear_beta)
                    ),
                    limit=float(limit),
                    hidden_workspace=hidden[:count],
                    output_workspace=output[:count],
                    result=result,
                    grouped_prefix=-1,
                    **self.store.man.projection_operator_capability(layer),
                )
            if expert_ids is None:
                expert_ids = self._host_route_ids(route_ids)
            reused = self._reuse_route_plan(
                layer,
                value,
                expert_ids,
                route_weights,
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=float(limit),
            )
            if reused is not None:
                return reused
            self._last_ids[layer] = expert_ids
            keys = [(layer, expert_id) for expert_id in expert_ids]
            async_stage = (
                os.environ.get("TPQ_KIMI_ASYNC_STAGE", "1") != "0"
                and os.environ.get("TPQ_KIMI_LAYER_TIMING", "0") == "0"
            )
            selected = self._ensure_locked(
                keys,
                prefetch=False,
                defer_wait=async_stage,
            )
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
            self._copy_metadata(experts)
            p12_positions = [
                position
                for position, expert in enumerate(experts)
                if (
                    len(expert) == 2
                    and
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
            identity_order = self._set_route_order(
                p12_positions,
                generic_positions,
            )
            if identity_order:
                ordered_weights = (
                    route_weights.reshape(-1).float().contiguous()
                )
            else:
                torch.index_select(
                    route_weights.reshape(-1).float().contiguous(),
                    0,
                    self._route_ids[: len(experts)],
                    out=self._ordered_weights[: len(experts)],
                )
                ordered_weights = self._ordered_weights[: len(experts)]
            self._save_route_plan(
                layer,
                expert_ids,
                keys,
                experts,
                len(p12_positions),
                identity_order,
            )
            # 大块 expert DMA 在 copy stream 继续进行时，CPU 已完成指针元数据
            # 和路由顺序构造；仅在融合核真正读取槽位前建立 GPU 事件依赖。
            # 这消除每层 cudaEventSynchronize 后的主机唤醒空洞，不改变槽位
            # 生命周期，也不把 packed 索引展开成中间矩阵。
            if async_stage:
                self._stage.wait()
            hidden, output, result = self._workspaces
            from .ops import packed_moe_topk

            computed = packed_moe_topk(
                value.to(torch.bfloat16),
                self._route_ids[: len(experts)],
                ordered_weights,
                self._metadata[:, : len(experts)],
                activation=activation,
                activation_beta=float(activation_beta),
                activation_linear_beta=(
                    0.0
                    if activation_linear_beta is None
                    else float(activation_linear_beta)
                ),
                limit=float(limit),
                hidden_workspace=hidden[: len(experts)],
                output_workspace=output[: len(experts)],
                result=result,
                grouped_prefix=len(p12_positions),
                **self.store.man.projection_operator_capability(
                    layer
                ),
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
            self._route_plans.clear()
            self._workspaces = None
            self._metadata = None
            self._slot_directory = None
            self._slot_update_host = None
            self._route_hit_mask = None
            self._route_all_hit = None
            self._route_all_hit_host = None
            self._route_host_ids = None
            self._route_copy_done = None
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


# The implementation is format-driven.  Retain the historical import name so
# existing Kimi deployments do not need to change at once.
KimiPackedHybridPool = PackedHybridPool


__all__ = [
    "HostPackedWeight",
    "KimiPackedHybridPool",
    "PackedHybridPool",
    "PackedExpertSignature",
    "PackedWeightSignature",
    "allocate_packed_slots",
]
