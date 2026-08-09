"""Configuration-driven shared single-device compact-expert RAM+GPU execution pool.

Key differences from the generic ``ExpertPool``:

* 9..15-bit indices retain their original TPQ packed format in RAM instead of expanding to uint16;
* stable GPU slots are preallocated by expert signature, and replacing an expert only overwrites slot contents;
* routes from the previous token are prefetched in the background, and the demand path waits by layer;
* Gate/Up, gated activation, Down, and routing weights directly use shared fused CUDA kernels.

Non-expert dense, attention, shared-expert, and KV paths remain unchanged. The implementation dispatches by
the projection-VQ manifest and operator capabilities and serves both Kimi and DeepSeek-V4.
"""

from __future__ import annotations

import gc
import os
import threading
import time
import ctypes
from collections.abc import Mapping
from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import torch

from .expert_slots import SlotBook
from .extreme import EXTREME_RAM_LOAD_WORKSPACE_GIB
from .store import TPQStore, PackedVQWeight, PinnedStage


def _release_host_allocator() -> None:
    """Return dead loader arenas to the OS before enforcing the RAM floor."""

    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        # Windows and non-glibc allocators do not expose malloc_trim. The
        # capacity check still uses their real MemAvailable after GC.
        pass


def _clear_codebook_cache(store) -> None:
    """Release an optional source-store codebook cache."""

    cache = getattr(store, "_cb_cache", None)
    if cache is not None:
        cache.clear()


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
    minimum: int | Mapping[PackedExpertSignature, int],
    weights: dict[PackedExpertSignature, float] | None = None,
    *,
    resident_codebooks: bool = False,
) -> dict[PackedExpertSignature, int]:
    """Allocate slots while ensuring that every signature can hold a complete Top-K.

    ``weights`` represents runtime routing traffic, not the static count of experts in each model tier.
    Mixed-precision models may have few high-precision experts that are called frequently. Allocating slots
    by static count would repeatedly evict that tier within every token and substantially amplify PCIe traffic.
    """
    if budget <= 0 or not counts:
        return {}
    if isinstance(minimum, Mapping):
        minimums = {
            signature: min(
                count,
                max(1, int(minimum.get(signature, 1))),
            )
            for signature, count in counts.items()
        }
    else:
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
        raw_storage: tuple[torch.Tensor, ...] | None = None,
        codebook_storage: tuple[torch.Tensor, ...] | None = None,
    ):
        self.signature = signature
        self.resident_codebooks = resident_codebooks
        self.book = SlotBook(count)
        self.raw = raw_storage or tuple(
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
            self.codebooks = codebook_storage or tuple(
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
        entries = tuple(
            (signature, count)
            for signature, count in specs.items()
            if count > 0
        )
        # Hundreds of heterogeneous signatures used to create three CUDA
        # allocations each.  Their allocator blocks and fragmentation were
        # not represented by logical tensor.nbytes and could consume hundreds
        # of MiB beyond the capacity plan.  One public byte slab (plus one
        # optional BF16 codebook slab) keeps all layouts compact while every
        # signature still exposes the same contiguous tensor views.
        raw_elements = sum(
            count * weight.raw_bytes
            for signature, count in entries
            for weight in signature.weights
        )
        self._raw_storage = torch.empty(
            raw_elements,
            dtype=torch.uint8,
            device=device,
        )
        codebook_elements = (
            0
            if resident_codebooks
            else sum(
                count * weight.cb_shape[0] * weight.cb_shape[1]
                for signature, count in entries
                for weight in signature.weights
            )
        )
        self._codebook_storage = (
            None
            if resident_codebooks
            else torch.empty(
                codebook_elements,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        raw_offset = 0
        codebook_offset = 0
        arenas: dict[PackedExpertSignature, _PackedArena] = {}
        for signature, count in entries:
            raw_views: list[torch.Tensor] = []
            codebook_views: list[torch.Tensor] = []
            for weight in signature.weights:
                raw_count = count * weight.raw_bytes
                raw_views.append(
                    self._raw_storage[
                        raw_offset:raw_offset + raw_count
                    ].view(count, weight.raw_bytes)
                )
                raw_offset += raw_count
                if self._codebook_storage is not None:
                    codebook_count = (
                        count
                        * weight.cb_shape[0]
                        * weight.cb_shape[1]
                    )
                    codebook_views.append(
                        self._codebook_storage[
                            codebook_offset:
                            codebook_offset + codebook_count
                        ].view(count, *weight.cb_shape)
                    )
                    codebook_offset += codebook_count
            arenas[signature] = _PackedArena(
                count,
                signature,
                device,
                resident_codebooks=resident_codebooks,
                raw_storage=tuple(raw_views),
                codebook_storage=(
                    tuple(codebook_views)
                    if codebook_views
                    else None
                ),
            )
        self.arenas = arenas
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
    """Configuration-driven Top-K expert pool with complete compact RAM residency and bounded stable VRAM."""

    device_routed = True
    full_resident = False
    prefetch_default = False
    # Keep p8-p16 indices in their original packed layout on disk, in RAM, and in VRAM.
    expanded_index_bytes = 0

    def __init__(
        self,
        store: TPQStore,
        budget_gb: float,
        *,
        device: str | torch.device,
        ram_gb: float = 0.0,
        startup_gpu_reserve_bytes: int = 0,
    ):
        self.store = store
        self.device = torch.device(device)
        self.budget = int(float(budget_gb) * 2**30)
        self.ram_budget = int(float(ram_gb) * 2**30)
        self.startup_gpu_reserve_bytes = max(
            0,
            int(startup_gpu_reserve_bytes),
        )
        self._startup_gpu_reservation: torch.Tensor | None = None
        self._startup_gpu_after_experts = 0
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
        self.route_counts: Counter[tuple[int, int]] = Counter()
        self._workspaces: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        self.slot_mix = self._read_slot_mix()
        self.arena_slots: dict[str, int] = {}
        self.fixed_extreme_residency = False
        self.supports_vram_watch = True
        self.extreme_ram_layers: tuple[int, ...] = ()
        self.extreme_gpu_layers: tuple[int, ...] = ()
        self.extreme_mixed_layers: tuple[int, ...] = ()
        self.extreme_placement_mode = "layer"
        self.extreme_score_source = "none"
        self.extreme_gpu_expert_count = 0
        self.extreme_storage_ratio = 0.0
        self._extreme_specs: dict[PackedExpertSignature, int] | None = None
        self._extreme_gpu_keys: set[tuple[int, int]] = set()
        self.extreme_stage_slots = 0
        self.extreme_route_working_set = 0
        self.extreme_route_history_resident = False

    def _reserve_startup_gpu_capacity(self) -> None:
        """Physically reserve Dense/runtime capacity before expert placement.

        A bookkeeping-only subtraction can still overcommit because CUDA
        fragmentation and the per-process limit are only known at allocation
        time. This byte tensor proves the reservation is physically available.
        It is released into PyTorch's reusable cache immediately before Dense
        is streamed to the device.
        """

        if (
            self.startup_gpu_reserve_bytes <= 0
            or self._startup_gpu_reservation is not None
        ):
            return
        self._startup_gpu_reservation = torch.empty(
            self.startup_gpu_reserve_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        print(
            "[tpq-extreme] Dense/上下文显存物理预留完成："
            f"{self.startup_gpu_reserve_bytes / 2**30:.2f}GiB；"
            "随后先放置紧凑专家",
            flush=True,
        )

    def release_startup_gpu_reservation(self, *, dense_next: bool = True) -> int:
        """Release the placeholder while keeping its CUDA block reusable."""

        if self._startup_gpu_reservation is None:
            return 0
        released = self._startup_gpu_reservation.nbytes
        self._startup_gpu_reservation = None
        gc.collect()
        self._startup_gpu_after_experts = torch.cuda.memory_allocated(
            self.device
        )
        if dense_next:
            message = (
                "[tpq-extreme] 专家放置完成，释放显存占位并开始流式加载 "
                f"Dense：{released / 2**30:.2f}GiB（分配器块直接复用）"
            )
        else:
            message = (
                "[tpq-extreme] 专家规划/加载失败，已释放 Dense 显存占位："
                f"{released / 2**30:.2f}GiB"
            )
        print(message, flush=True)
        return released

    def verify_startup_gpu_reservation(self) -> None:
        """Reject an underestimated fixed allocation before inference."""

        if self.startup_gpu_reserve_bytes <= 0:
            return
        actual = max(
            0,
            torch.cuda.memory_allocated(self.device)
            - self._startup_gpu_after_experts,
        )
        if actual > self.startup_gpu_reserve_bytes:
            raise RuntimeError(
                "极限模式固定显存估算不足：Dense/上下文实际增加 "
                f"{actual / 2**30:.2f}GiB > 预留 "
                f"{self.startup_gpu_reserve_bytes / 2**30:.2f}GiB；"
                "拒绝带着不可靠容量规划继续推理。"
            )
        print(
            "[tpq-extreme] Dense 显存替换校验通过："
            f"实际={actual / 2**30:.2f}GiB / "
            f"预留={self.startup_gpu_reserve_bytes / 2**30:.2f}GiB；"
            "无完整模型副本",
            flush=True,
        )

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

    @staticmethod
    def _packed_width(format_name: str) -> int:
        value = str(format_name).lower()
        if not value.startswith("p") or not value[1:].isdigit():
            raise ValueError(f"极限模式不支持 packed 格式 {format_name!r}")
        bits = int(value[1:])
        if not 8 <= bits <= 16:
            raise ValueError(f"极限模式 packed 位宽非法: {bits}")
        return bits

    def _extreme_signature(
        self,
        layer: int,
        expert_id: int,
    ) -> PackedExpertSignature:
        """Read only shared codebooks and manifest metadata, without reading the expert-index payload."""

        if not self.store.man.projection_vq:
            raise RuntimeError(
                "极限模式要求三投影 packed VQ；旧专家格式可能展开索引"
            )
        # Capacity and arena signatures are per expert.  Heterogeneous layers
        # return a layout union when expert_id is omitted, which is correct for
        # registry discovery but must never be used as one expert's Gate/Up/
        # Down tuple.
        capability = self.store.man.projection_operator_capability(
            layer,
            expert_id,
        )
        formats = capability["packed_formats"]
        dims = capability["code_dims"]
        codebooks = self.store.projection_codebooks(layer, expert_id)
        variants = self.store.projection_codebook_variants(
            layer,
            expert_id,
        )
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        intermediate = int(self.store.cfg["moe_inter"])
        shapes = (
            (intermediate, hidden),
            (intermediate, hidden),
            (hidden, intermediate),
        )
        weights = []
        for projection, format_name, dim, codebook, variant, shape in zip(
            ("gate", "up", "down"),
            formats,
            dims,
            codebooks,
            variants,
            shapes,
        ):
            bits = self._packed_width(format_name)
            rows, cols = shape
            blocks = cols // int(dim)
            payload_bits = rows * blocks * bits
            if payload_bits % 8:
                raise RuntimeError(
                    f"L{layer} e{expert_id} {projection} 不是整字节 packed payload"
                )
            host_codebook = self._host_codebook(
                ("projection-vq", variant, projection),
                codebook,
            )
            weights.append(
                PackedWeightSignature(
                    raw_bytes=payload_bits // 8,
                    cb_shape=tuple(host_codebook.shape),
                    rows=rows,
                    cols=cols,
                    blocks=blocks,
                    dim=int(dim),
                    bits=bits,
                )
            )
        return PackedExpertSignature(tuple(weights))

    def _preload_extreme(self, reserve_gb: float) -> bool:
        """Distribute complete layers across RAM/VRAM so expert files are not read at runtime."""

        if self.device.type != "cuda":
            raise RuntimeError("极限模式要求单卡 CUDA packed 专家池")
        if not self.store.man.projection_vq:
            raise RuntimeError(
                "极限模式拒绝旧专家格式：其索引加载后可能大于磁盘 payload"
            )
        import psutil

        layers = tuple(sorted(int(x) for x in self.store.man.expert_files))
        n_experts = int(self.store.cfg["n_experts"])
        signatures_by_layer: dict[int, tuple[PackedExpertSignature, ...]] = {}
        layer_bytes: dict[int, int] = {}
        keys_by_layer: dict[int, tuple[tuple[int, int], ...]] = {}
        print(
            "[tpq-extreme] 读取码本与 packing 元数据，规划整层 RAM/VRAM 放置…",
            flush=True,
        )
        for layer in layers:
            keys = tuple(
                (layer, expert_id)
                for expert_id in range(n_experts)
                if self.store.expert_kind(layer, expert_id) != "drop"
            )
            signatures = tuple(
                self._extreme_signature(*key)
                for key in keys
            )
            keys_by_layer[layer] = keys
            signatures_by_layer[layer] = signatures
            layer_bytes[layer] = sum(
                signature.raw_slot_bytes for signature in signatures
            )
        # projection_codebooks() uses FP32 as an intermediate read representation. At runtime, retain only the shared
        # BF16 codebook referenced by HostPackedWeight to avoid keeping two resident copies.
        _clear_codebook_cache(self.store)
        gc.collect()

        from .extreme import (
            GIB,
            effective_available_memory_bytes,
            load_expert_residency_scores,
            plan_extreme_expert_placement,
            plan_extreme_layer_placement,
        )

        available = effective_available_memory_bytes()
        ram_available = min(available, self.ram_budget + int(reserve_gb * GIB))
        safe_gpu = self._safe_budget()
        codebook_bytes = sum(
            value.nbytes for value in self._host_codebooks.values()
        )
        # mmap-backed packed rows and the two loader futures briefly coexist
        # while one expert is materialized. Keep one bounded 0.5 GiB loader
        # workspace in addition to the user's 1 GiB system reserve. A 5%
        # model-sized default silently held back several GiB on a 64 GiB host
        # and contradicted extreme mode's documented "fill until 1 GiB" rule;
        # 256 MiB was insufficient before releasing glibc's dead loader arenas
        # on a real 64 GiB host; the final physical 1 GiB check remains hard.
        configured_loader_workspace = os.environ.get(
            "TPQ_EXTREME_LOAD_WORKSPACE_GB",
            "",
        ).strip()
        loader_workspace = (
            int(float(configured_loader_workspace) * GIB)
            if configured_loader_workspace
            else int(EXTREME_RAM_LOAD_WORKSPACE_GIB * GIB)
        )
        if loader_workspace < 256 * 2**20:
            raise RuntimeError(
                "极限模式加载工作区至少需要 0.25 GiB"
            )
        print(
            "[tpq-extreme] RAM规划："
            f"payload={sum(layer_bytes.values()) / GIB:.2f}GiB；"
            f"码本={codebook_bytes / GIB:.2f}GiB；"
            f"加载工作区={loader_workspace / GIB:.2f}GiB；"
            f"系统预留={reserve_gb:.2f}GiB",
            flush=True,
        )
        signature_by_key = {
            key: signature
            for layer in layers
            for key, signature in zip(
                keys_by_layer[layer],
                signatures_by_layer[layer],
            )
        }
        size_by_key = {
            key: signature.raw_slot_bytes
            for key, signature in signature_by_key.items()
        }
        top_k = int(self.store.cfg["top_k"])
        # Capacity order is fixed: Dense/runtime reservation (already held),
        # shared codebooks, one executable Top-K for every packed signature,
        # then GPU-only experts. The old order filled GPU experts first and
        # discovered too late that the model could not execute a RAM layer.
        minimum_stage_by_signature: dict[PackedExpertSignature, int] = {}
        for layer in layers:
            layer_counts = Counter(signatures_by_layer[layer])
            for signature, count in layer_counts.items():
                minimum_stage_by_signature[signature] = max(
                    minimum_stage_by_signature.get(signature, 0),
                    min(top_k, count),
                )
        minimum_stage_bytes = sum(
            signature.storage_bytes(True) * count
            for signature, count in minimum_stage_by_signature.items()
        )
        gpu_expert_budget = max(
            0,
            safe_gpu - codebook_bytes - minimum_stage_bytes,
        )
        print(
            "[tpq-extreme] 显存规划顺序："
            f"固定Dense占位→码本 {codebook_bytes / GIB:.2f}GiB→"
            f"最小Top-K {minimum_stage_bytes / GIB:.2f}GiB→"
            f"GPU专家上限 {gpu_expert_budget / GIB:.2f}GiB",
            flush=True,
        )
        requested_placement = os.environ.get(
            "TPQ_EXTREME_PLACEMENT",
            "auto",
        ).strip().lower()
        if requested_placement not in {"auto", "layer", "precision"}:
            raise RuntimeError(
                "TPQ_EXTREME_PLACEMENT 只接受 auto/layer/precision"
            )
        precision_placement = (
            requested_placement == "precision"
            or (
                requested_placement == "auto"
                and self.store.man.heterogeneous_projection_vq
            )
        )
        if precision_placement:
            # The packed bytes assigned by the quantizer are a self-contained
            # precision/importance signal: all routed experts have identical
            # logical matrix shapes, so a larger compact payload represents a
            # larger bit budget.  This remains manifest-driven and introduces
            # no model-name or tier-name branch.
            score_file = os.environ.get(
                "TPQ_EXTREME_SCORE_FILE",
                "",
            ).strip()
            if score_file:
                precision_scores = load_expert_residency_scores(score_file)
                missing_scores = size_by_key.keys() - precision_scores.keys()
                extra_scores = precision_scores.keys() - size_by_key.keys()
                if missing_scores or extra_scores:
                    raise RuntimeError(
                        "专家常驻分数必须与归档一一覆盖："
                        f"缺少={len(missing_scores)}，多余={len(extra_scores)}"
                    )
                placement_groups = None
                self.extreme_score_source = "route-mass"
            else:
                precision_scores = {
                    key: float(size)
                    for key, size in size_by_key.items()
                }
                placement_groups = {
                    key: key[0] for key in size_by_key
                }
                self.extreme_score_source = "packed-bit-budget"
            try:
                placement = plan_extreme_expert_placement(
                    size_by_key,
                    precision_scores,
                    placement_groups=placement_groups,
                    available_ram_bytes=ram_available,
                    ram_reserve_bytes=int(reserve_gb * GIB),
                    # Host codebooks are already materialized above and are
                    # therefore already reflected in ``available``. Only the
                    # not-yet-allocated loader workspace is subtracted here.
                    fixed_ram_bytes=loader_workspace,
                    gpu_expert_bytes=gpu_expert_budget,
                )
            except RuntimeError:
                if requested_placement != "auto":
                    raise
                try:
                    # Per-layer fairness is a performance preference, not a
                    # capacity invariant. On a genuinely tight machine keep
                    # the largest/highest-bit experts in GPU globally; every
                    # layer remains executable through the reserved staging.
                    placement = plan_extreme_expert_placement(
                        size_by_key,
                        precision_scores,
                        placement_groups=None,
                        available_ram_bytes=ram_available,
                        ram_reserve_bytes=int(reserve_gb * GIB),
                        fixed_ram_bytes=loader_workspace,
                        gpu_expert_bytes=gpu_expert_budget,
                    )
                    self.extreme_score_source = (
                        "packed-bit-budget-capacity"
                    )
                    print(
                        "[tpq-extreme] 分层均衡精度放置超出容量，改用全局 "
                        "bit-budget 放置；最小 Top-K 仍完整保留",
                        flush=True,
                    )
                except RuntimeError as capacity_error:
                    precision_placement = False
                    print(
                        "[tpq-extreme] 全局精度放置仍无法满足容量："
                        f"{capacity_error}；最后尝试整层放置",
                        flush=True,
                    )
        if precision_placement:
            # Selection order follows score rank.  Loading order follows the
            # archive so codebook metadata is reused and startup does not
            # thrash one layer's temporary handles per expert.
            gpu_keys = tuple(sorted(placement.gpu_keys))
            ram_keys = tuple(placement.ram_keys)
            gpu_key_set = set(gpu_keys)
            ram_key_set = set(ram_keys)
            self.extreme_gpu_layers = tuple(
                layer
                for layer in layers
                if all(key in gpu_key_set for key in keys_by_layer[layer])
            )
            self.extreme_ram_layers = tuple(
                layer
                for layer in layers
                if any(key in ram_key_set for key in keys_by_layer[layer])
            )
            self.extreme_mixed_layers = tuple(
                layer
                for layer in layers
                if any(key in gpu_key_set for key in keys_by_layer[layer])
                and any(key in ram_key_set for key in keys_by_layer[layer])
            )
            self.extreme_placement_mode = "precision"
        if not precision_placement:
            placement = plan_extreme_layer_placement(
                layer_bytes,
                available_ram_bytes=ram_available,
                # Host codebooks are already reflected in current available
                # RAM. Keep one bounded *additional* loader workspace so
                # cgroup MemoryMax is not reached by materialization futures.
                ram_reserve_bytes=int(reserve_gb * GIB),
                fixed_ram_bytes=loader_workspace,
                gpu_expert_bytes=gpu_expert_budget,
            )
            self.extreme_ram_layers = placement.ram_layers
            self.extreme_gpu_layers = placement.gpu_layers
            self.extreme_mixed_layers = ()
            gpu_keys = tuple(
                key
                for layer in self.extreme_gpu_layers
                for key in keys_by_layer[layer]
            )
            ram_keys = tuple(
                key
                for layer in self.extreme_ram_layers
                for key in keys_by_layer[layer]
            )
            gpu_key_set = set(gpu_keys)
            ram_key_set = set(ram_keys)
            self.extreme_placement_mode = "layer"
            self.extreme_score_source = "none"
        self.extreme_gpu_expert_count = len(gpu_keys)

        resident_counts: Counter[PackedExpertSignature] = Counter(
            signature_by_key[key] for key in gpu_keys
        )
        ram_counts: Counter[PackedExpertSignature] = Counter(
            signature_by_key[key] for key in ram_keys
        )
        compact_baseline = sum(layer_bytes.values()) + codebook_bytes
        max_ratio = float(os.environ.get("TPQ_EXTREME_MAX_OVERHEAD", "1.10"))
        resident_bytes = sum(
            signature.storage_bytes(True) * count
            for signature, count in resident_counts.items()
        )
        # Protected GPU layers never participate in the LRU; the remaining safe VRAM is used for hot-expert slots across tokens.
        # Repeated staging remains constrained by the residency multiplier; do not trade memory for speed by copying the full model into VRAM again.
        duplicate_limit = max(
            0,
            int((max_ratio - 1.0) * compact_baseline)
            - codebook_bytes
            - 64 * 2**20,
        )
        stage_budget = min(
            max(0, safe_gpu - codebook_bytes - resident_bytes),
            duplicate_limit,
        )
        # A heterogeneous archive can contain hundreds of signatures while a
        # single layer has only one or two experts of most signatures.  The
        # arena only needs to hold the maximum number of same-signature
        # experts that one routed Top-K can request in one layer, not Top-K
        # slots for every signature in the whole model.
        minimum_by_signature: dict[PackedExpertSignature, int] = {}
        for layer in self.extreme_ram_layers:
            layer_counts = Counter(
                signature_by_key[key]
                for key in keys_by_layer[layer]
                if key in ram_key_set
            )
            for signature, count in layer_counts.items():
                minimum_by_signature[signature] = max(
                    minimum_by_signature.get(signature, 1),
                    min(top_k, count),
                )
        stage_counts = (
            allocate_packed_slots(
                dict(ram_counts),
                stage_budget,
                minimum_by_signature,
                resident_codebooks=True,
            )
            if ram_counts
            else {}
        )
        self.extreme_stage_slots = sum(stage_counts.values())
        self.extreme_route_working_set = sum(
            min(
                top_k,
                sum(key in ram_key_set for key in keys_by_layer[layer]),
            )
            for layer in self.extreme_ram_layers
        )
        # With one packed signature, every slot can be reused by any RAM layer. If a full Top-K round fits,
        # protect each layer's actual previous-token routes so the global LRU cannot evict slots needed by deeper layers
        # while progressing from shallow layers. Mixed-signature models continue to use the shared prefetch path
        # to avoid overcommitting based on an incorrect tier ratio.
        self.extreme_route_history_resident = (
            len(ram_counts) == 1
            and self.extreme_stage_slots >= self.extreme_route_working_set
        )
        if self.extreme_route_history_resident:
            self._protect_previous = True
        specs = dict(resident_counts)
        for signature, count in stage_counts.items():
            specs[signature] = specs.get(signature, 0) + count
        arena_bytes = sum(
            signature.storage_bytes(True) * count
            for signature, count in specs.items()
        )
        if arena_bytes + codebook_bytes > safe_gpu:
            raise RuntimeError(
                "极限模式显存不足：GPU 专家整层 + RAM Top-K staging + "
                f"共享码本需要 {(arena_bytes + codebook_bytes) / GIB:.2f} GiB，"
                f"安全可用仅 {safe_gpu / GIB:.2f} GiB。请降低 --max-ctx、"
                "关闭其他显存进程或换用更小模型。"
            )
        self._extreme_specs = specs
        self.build_gpu_arenas()

        gpu_total = len(gpu_keys)
        loaded_gpu = 0
        previous_layer = None
        for key in gpu_keys:
            if previous_layer is not None and key[0] != previous_layer:
                _clear_codebook_cache(self.store)
                gc.collect()
            host = self._load_one(*key)
            self.pinned[key] = host
            self._ensure_locked([key], prefetch=True)
            if self._arenas is None or not self._arenas.protect(key):
                raise RuntimeError(
                    f"极限模式无法保护 GPU-only 专家 L{key[0]}/e{key[1]}"
                )
            self._extreme_gpu_keys.add(key)
            self.pinned.pop(key, None)
            loaded_gpu += 1
            previous_layer = key[0]
            if loaded_gpu % 256 == 0:
                print(
                    f"[tpq-extreme] GPU-only 专家 {loaded_gpu}/{gpu_total}",
                    flush=True,
                )
        _clear_codebook_cache(self.store)
        gc.collect()

        workers = max(1, int(os.environ.get("TPQ_LOAD_WORKERS", "12")))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="tpq-extreme-ram",
        ) as executor:
            futures = {
                executor.submit(self._load_one, *key): key
                for key in ram_keys
            }
            for index, future in enumerate(as_completed(futures), 1):
                self.pinned[futures[future]] = future.result()
                if index % 2000 == 0:
                    print(
                        f"[tpq-extreme] RAM 专家 {index}/{len(ram_keys)}",
                        flush=True,
                    )
        _clear_codebook_cache(self.store)
        # Host codebooks for GPU-only layers are no longer needed; device codebooks are held independently.
        retained_codebook_ptrs = {
            weight.cb.data_ptr()
            for expert in self.pinned.values()
            for weight in expert
        }
        self._host_codebooks = {
            key: value
            for key, value in self._host_codebooks.items()
            if value.data_ptr() in retained_codebook_ptrs
        }
        _release_host_allocator()
        self.ram_bytes = self.host_expert_bytes
        remaining_ram = effective_available_memory_bytes()
        required_reserve = int(reserve_gb * GIB)
        if remaining_ram < required_reserve:
            raise RuntimeError(
                "极限模式 RAM 预留未满足：加载后可用 "
                f"{remaining_ram / GIB:.2f} GiB < 要求 {reserve_gb:.2f} GiB。"
                "请关闭其他进程或换用更小模型。"
            )
        actual = self.host_expert_bytes + self.gpu_storage_bytes
        self.extreme_storage_ratio = actual / max(1, compact_baseline)
        if self.extreme_storage_ratio > max_ratio:
            raise RuntimeError(
                "极限模式常驻放大超过限制："
                f"{self.extreme_storage_ratio:.3f}x > {max_ratio:.3f}x；"
                "拒绝用隐式副本换取表面容量。"
            )
        self.fixed_extreme_residency = True
        self.supports_vram_watch = False
        self.layer_prefetch_only = True
        self.bytes = self.gpu_storage_bytes
        print(
            "[tpq-extreme] 紧凑常驻完成："
            f"策略={self.extreme_placement_mode}；"
            f"热度来源={self.extreme_score_source}；"
            f"RAM参与层={list(self.extreme_ram_layers)} "
            f"{self.host_expert_bytes / GIB:.2f}GiB；"
            f"GPU整层={list(self.extreme_gpu_layers)}；"
            f"混合层={list(self.extreme_mixed_layers)}；"
            f"GPU-only专家={self.extreme_gpu_expert_count} / "
            f"专家与staging {self.gpu_storage_bytes / GIB:.2f}GiB；"
            f"RAM热槽={self.extreme_stage_slots}/"
            f"一轮路由={self.extreme_route_working_set}；"
            f"常驻/紧凑基准={self.extreme_storage_ratio:.3f}x；"
            "expanded_index_bytes=0，运行期专家磁盘读取=0",
            flush=True,
        )
        return True

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
        if os.environ.get("TPQ_EXTREME_MODE", "0") != "0":
            self._reserve_startup_gpu_capacity()
            try:
                return self._preload_extreme(float(reserve_gb))
            except BaseException:
                # A failed startup must not strand a multi-GiB placeholder in
                # a long-lived API worker or a retrying Engine process.
                self.release_startup_gpu_reservation(dense_next=False)
                raise
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
                    "projection-VQ TPQ 专家清单未收敛："
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
        # All runtime experts reference only BF16 codebooks; release the store's intermediate FP32 copies.
        _clear_codebook_cache(self.store)
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
                if not getattr(self, "fixed_extreme_residency", False):
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
                extreme_cap_gb = max(
                    0.0,
                    float(
                        os.environ.get(
                            "TPQ_EXTREME_VRAM_CAP_GB",
                            "0",
                        )
                    ),
                )
                if extreme_cap_gb > 0:
                    device_bytes = min(
                        device_bytes,
                        int(extreme_cap_gb * 2**30),
                    )
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
        if not self.pinned and self._extreme_specs is None:
            return 0.0
        safe_budget = self._safe_budget()
        if safe_budget <= 0:
            raise RuntimeError("packed hybrid has no safe GPU cache room")
        counts = (
            Counter(self._extreme_specs)
            if self._extreme_specs is not None
            else Counter(
                PackedExpertSignature.of(expert)
                for expert in self.pinned.values()
            )
        )
        host_codebooks = {
            codebook.data_ptr(): codebook
            for codebook in self._host_codebooks.values()
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
        specs = (
            dict(self._extreme_specs)
            if self._extreme_specs is not None
            else allocate_packed_slots(
                counts,
                arena_budget,
                top_k,
                weights=weights,
                resident_codebooks=self._resident_codebooks,
            )
        )
        required = sum(
            signature.storage_bytes(self._resident_codebooks) * count
            for signature, count in specs.items()
        )
        if required > arena_budget:
            raise RuntimeError(
                "极限模式固定专家槽超过安全显存预算："
                f"{required / 2**30:.2f} > {arena_budget / 2**30:.2f} GiB"
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
                # A preceding waiter may already have loaded it.
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
            # At the start of a token, the model submits about 92 requests by layer. Single-threaded execution preserves
            # slot order, and the queue itself must hold a full round or it will prefetch only shallow layers.
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

    def _release_stale_route_protection(
        self,
        layer: int,
        keys: list[tuple[int, int]],
    ) -> None:
        """Release only this layer's stale route before leasing replacements."""

        if not self._protect_previous or self._arenas is None:
            return
        current = {
            key for key in keys if key not in self._extreme_gpu_keys
        }
        with self._lock:
            for key in self._protected_by_layer.get(int(layer), ()):
                if key not in current:
                    self._arenas.unprotect(key)

    def _commit_route_protection(
        self,
        layer: int,
        keys: list[tuple[int, int]],
    ) -> None:
        """Protect one layer's current route after every key is resident."""

        if not self._protect_previous or self._arenas is None:
            return
        route_keys = [
            key for key in keys if key not in self._extreme_gpu_keys
        ]
        if not route_keys:
            return
        with self._lock:
            for key in route_keys:
                if not self._arenas.protect(key):
                    raise RuntimeError(
                        "cannot protect resident packed route slot "
                        f"L{key[0]}/e{key[1]}"
                    )
            self._protected_by_layer[int(layer)] = tuple(route_keys)

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
        if os.environ.get("TPQ_DEVICE_ROUTE_METADATA", "1") == "0":
            return False, None
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
        host_ids = [
            int(self._route_host_ids[index])
            for index in range(count)
        ]
        self._record_route_ids(layer, host_ids)
        if bool(self._route_all_hit_host):
            self.device_route_full_hits += 1
            self.hits += count
            self.last_transfer_seconds = 0.0
            if (
                self._protect_previous
                or os.environ.get("TPQ_PREFETCH", "0") != "0"
            ):
                self._last_ids[int(layer)] = list(host_ids)
            return True, None
        self.device_route_fallbacks += 1
        return False, host_ids

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

    def _record_route_ids(self, layer: int, expert_ids: list[int]) -> None:
        counts = getattr(self, "route_counts", None)
        if counts is None:
            counts = self.route_counts = Counter()
        counts.update((int(layer), int(expert_id)) for expert_id in expert_ids)

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
                if self._protect_previous:
                    self._commit_route_protection(
                        layer,
                        [
                            (int(layer), expert_id)
                            for expert_id in self._last_ids[int(layer)]
                        ],
                    )
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
                self._record_route_ids(layer, expert_ids)
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
            self._release_stale_route_protection(layer, keys)
            uploaded_before = self.uploaded_bytes
            selected = self._ensure_locked(
                keys,
                prefetch=False,
                defer_wait=True,
            )
            self._commit_route_protection(layer, keys)
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
        # This is the only GPU-to-CPU routing synchronization needed for RAM address selection; weight computation stays on the GPU.
        # Background prefetch cannot reuse slots until the fused kernel is queued on the default stream. After releasing the lock,
        # PinnedStage.wait_stream schedules subsequent overwrites after this kernel.
        with self._transfer_lock:
            device_hit, expert_ids = self._device_route_metadata(
                layer,
                route_ids,
            )
            if device_hit:
                count = int(route_ids.numel())
                if self._protect_previous:
                    self._commit_route_protection(
                        layer,
                        [
                            (int(layer), expert_id)
                            for expert_id in self._last_ids[int(layer)]
                        ],
                    )
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
                self._record_route_ids(layer, expert_ids)
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
            # Release previous-round slots no longer needed by this layer before leasing slots for new routes.
            # The old order leased before unprotecting; when protected slots exactly filled the cache, this caused a false
            # capacity error claiming no slot was available even though the cache could hold one routing round.
            self._release_stale_route_protection(layer, keys)
            async_stage = (
                os.environ.get("TPQ_KIMI_ASYNC_STAGE", "1") != "0"
                and os.environ.get("TPQ_KIMI_LAYER_TIMING", "0") == "0"
            )
            selected = self._ensure_locked(
                keys,
                prefetch=False,
                defer_wait=async_stage,
            )
            self._commit_route_protection(layer, keys)
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
            # While large expert DMA continues on the copy stream, the CPU finishes pointer metadata and route ordering.
            # Establish the GPU event dependency only before the fused kernel actually reads the slots.
            # This removes the host wake-up gap after each layer's cudaEventSynchronize without changing slot lifetimes
            # or expanding packed indices into an intermediate matrix.
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
        if getattr(self, "fixed_extreme_residency", False):
            raise RuntimeError(
                "极限模式 GPU-only 专家不可驱逐；请降低上下文后重新启动"
            )
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
        if (
            getattr(self, "fixed_extreme_residency", False)
            and int(budget) < self.gpu_storage_bytes
        ):
            raise RuntimeError(
                "极限模式没有可收缩专家副本；请降低上下文后重新启动"
            )
        budget = max(0, int(budget))
        if self.gpu_arena_bytes > budget:
            self.resize_gpu_arenas(budget)
        else:
            self.budget = budget


__all__ = [
    "HostPackedWeight",
    "PackedHybridPool",
    "PackedExpertSignature",
    "PackedWeightSignature",
    "allocate_packed_slots",
]
