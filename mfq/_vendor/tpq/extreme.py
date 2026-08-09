"""Shared configuration and capacity planning for single-device extreme RAM+VRAM residency mode."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Hashable, Mapping


GIB = 2**30
EXTREME_GPU_LOAD_WORKSPACE_GIB = 1.5
EXTREME_RAM_LOAD_WORKSPACE_GIB = 0.5
EXTREME_RAM_RESERVE_GIB = 1.0


def effective_available_memory_bytes(
    *,
    system_available_bytes: int | None = None,
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    cgroup_file: str | os.PathLike[str] = "/proc/self/cgroup",
) -> int:
    """Return the smaller available-memory value under physical-machine and cgroup constraints.

    psutil correctly reports native low-memory machines. Containers, systemd services, and restricted desktop
    environments must also subtract cgroup v2 ``memory.current``; otherwise extreme mode plans against total host
    memory until terminated by the OOM killer.
    """

    if system_available_bytes is None:
        import psutil

        system_available_bytes = int(psutil.virtual_memory().available)
    available = max(0, int(system_available_bytes))
    try:
        cgroup_path = next(
            line.split("::", 1)[1].strip()
            for line in Path(cgroup_file).read_text(
                encoding="utf-8",
            ).splitlines()
            if "::" in line
        )
        root = Path(cgroup_root) / cgroup_path.lstrip("/")
        maximum_text = (root / "memory.max").read_text(
            encoding="ascii",
        ).strip()
        if maximum_text != "max":
            maximum = int(maximum_text)
            current = int(
                (root / "memory.current").read_text(
                    encoding="ascii",
                ).strip()
            )
            # cgroup memory.current includes filesystem page cache.  Model
            # loading creates a large inactive file cache that the kernel can
            # reclaim before charging packed expert tensors.  Treating it as
            # anonymous residency makes an otherwise identical capacity plan
            # depend on whether the file was read recently.
            inactive_file = 0
            try:
                memory_stat = dict(
                    line.split(maxsplit=1)
                    for line in (root / "memory.stat").read_text(
                        encoding="ascii",
                    ).splitlines()
                    if " " in line
                )
                inactive_file = max(
                    0,
                    int(memory_stat.get("inactive_file", "0")),
                )
            except (OSError, ValueError):
                pass
            cgroup_available = min(
                maximum,
                max(0, maximum - current) + inactive_file,
            )
            available = min(available, cgroup_available)
    except (OSError, StopIteration, ValueError):
        pass
    return available


@dataclass(frozen=True)
class ExtremeLayerPlacement:
    """Contiguous-layer placement result; all layers after the RAM prefix enter VRAM."""

    ram_layers: tuple[int, ...]
    gpu_layers: tuple[int, ...]
    ram_bytes: int
    gpu_bytes: int
    ram_capacity: int
    gpu_capacity: int


@dataclass(frozen=True)
class ExtremeExpertPlacement:
    """Capacity-safe expert placement ranked by a model-provided precision signal."""

    ram_keys: tuple[Hashable, ...]
    gpu_keys: tuple[Hashable, ...]
    ram_bytes: int
    gpu_bytes: int
    ram_capacity: int
    gpu_capacity: int


@dataclass(frozen=True)
class CompactArchiveCapacity:
    """Model-independent compact three-projection archive capacity and shared operator signature."""

    expert_bytes: int
    layers: tuple[int, ...]
    packed_formats: tuple[str, ...]
    code_dims: tuple[int, ...]
    codebook_sizes: tuple[int, ...]


@dataclass(frozen=True)
class AutoExtremeDecision:
    """Configuration-driven automatic single-device placement decision."""

    activate: bool
    mode: str
    reason: str
    expert_bytes: int = 0
    available_ram_bytes: int = 0
    normal_ram_capacity: int = 0
    extreme_ram_capacity: int = 0
    gpu_expert_capacity: int = 0
    spill_bytes: int = 0


def plan_auto_extreme(
    *,
    compact_expert_bytes: int,
    available_ram_bytes: int,
    free_gpu_bytes: int,
    fixed_gpu_bytes: int,
    normal_ram_reserve_bytes: int = 32 * GIB,
    extreme_ram_reserve_bytes: int = GIB,
    load_workspace_bytes: int = int(EXTREME_RAM_LOAD_WORKSPACE_GIB * GIB),
    gpu_reserve_bytes: int = 512 * 2**20,
) -> AutoExtremeDecision:
    """Choose automatically among all-VRAM, normal RAM, and extreme RAM+VRAM placement.

    All inputs come from the manifest, actual file bytes, and current machine capacity, so model names are not used.
    Prefer resident mode when compact experts and fixed weights fit entirely on one device. Otherwise enable extreme mode
    only when normal safe RAM capacity is insufficient but the compact-expert overflow fits the fixed VRAM margin.
    When both sides are insufficient, retain the normal path so the upper layer can provide a clear capacity diagnosis.
    """

    expert_bytes = max(0, int(compact_expert_bytes))
    available_ram = max(0, int(available_ram_bytes))
    normal_ram_capacity = max(
        0,
        available_ram - max(0, int(normal_ram_reserve_bytes)),
    )
    extreme_ram_capacity = max(
        0,
        available_ram
        - max(0, int(extreme_ram_reserve_bytes))
        - max(0, int(load_workspace_bytes)),
    )
    gpu_expert_capacity = max(
        0,
        int(free_gpu_bytes)
        - max(0, int(fixed_gpu_bytes))
        - max(0, int(gpu_reserve_bytes)),
    )
    spill_bytes = max(0, expert_bytes - extreme_ram_capacity)
    common = dict(
        expert_bytes=expert_bytes,
        available_ram_bytes=available_ram,
        normal_ram_capacity=normal_ram_capacity,
        extreme_ram_capacity=extreme_ram_capacity,
        gpu_expert_capacity=gpu_expert_capacity,
        spill_bytes=spill_bytes,
    )
    if expert_bytes <= gpu_expert_capacity:
        return AutoExtremeDecision(
            activate=False,
            mode="resident",
            reason="固定权重与紧凑专家可完整进入单卡显存",
            **common,
        )
    if expert_bytes <= normal_ram_capacity:
        return AutoExtremeDecision(
            activate=False,
            mode="ram",
            reason="普通 RAM 安全容量足够",
            **common,
        )
    if spill_bytes <= gpu_expert_capacity:
        return AutoExtremeDecision(
            activate=True,
            mode="extreme",
            reason="RAM 单侧不足，但 RAM+VRAM 可容纳紧凑归档",
            **common,
        )
    return AutoExtremeDecision(
        activate=False,
        mode="insufficient",
        reason="RAM+VRAM 扣除固定权重和工作区后仍不足",
        **common,
    )


def detect_auto_extreme(
    model_dir: str | os.PathLike[str],
    *,
    max_ctx: int,
    device: str,
    tp_size: int,
    normal_ram_reserve_gib: float = 32.0,
    environment: Mapping[str, str] | None = None,
) -> AutoExtremeDecision:
    """Read the shared manifest and current hardware to produce an automatic extreme-mode decision."""

    if str(device) != "cuda" or int(tp_size) != 1:
        return AutoExtremeDecision(
            False,
            "disabled",
            "自动极限模式只适用于单卡 CUDA",
        )
    try:
        archive = inspect_compact_projection_archive(model_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return AutoExtremeDecision(
            False,
            "unsupported",
            str(exc),
        )

    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return AutoExtremeDecision(False, "disabled", "CUDA 不可用")
        free_gpu, _total_gpu = torch.cuda.mem_get_info(0)
    except (ImportError, RuntimeError) as exc:
        return AutoExtremeDecision(False, "disabled", f"显存探测失败：{exc}")

    from .check import fixed_vram_gib
    from .presets import detect_architecture, load_manifest

    root, manifest = load_manifest(model_dir)
    architecture = detect_architecture(manifest)
    effective_environment = dict(environment or {})
    fixed_gpu = int(
        max(
            0.0,
            fixed_vram_gib(
                root,
                manifest,
                architecture,
                int(max_ctx),
                effective_environment,
            ) - EXTREME_GPU_LOAD_WORKSPACE_GIB,
        )
        * GIB
    )
    cap_gib = max(
        0.0,
        float(os.environ.get("TPQ_EXTREME_VRAM_CAP_GB", "0") or 0),
    )
    if cap_gib:
        free_gpu = min(int(free_gpu), int(cap_gib * GIB))
    loader_workspace = int(
        max(
            EXTREME_RAM_LOAD_WORKSPACE_GIB,
            float(
                effective_environment.get(
                    "TPQ_EXTREME_LOAD_WORKSPACE_GB",
                    EXTREME_RAM_LOAD_WORKSPACE_GIB,
                )
            ),
        )
        * GIB
    )
    return plan_auto_extreme(
        compact_expert_bytes=archive.expert_bytes,
        available_ram_bytes=effective_available_memory_bytes(),
        free_gpu_bytes=int(free_gpu),
        fixed_gpu_bytes=fixed_gpu,
        normal_ram_reserve_bytes=int(normal_ram_reserve_gib * GIB),
        load_workspace_bytes=loader_workspace,
    )


def inspect_compact_projection_archive(
    model_dir: str | os.PathLike[str],
) -> CompactArchiveCapacity:
    """Audit an in-place unpackable projection-VQ archive through the shared manifest.

    This does not identify model names or require the legacy ``projection_layouts`` field. :class:`Manifest`
    first normalizes per-layer layouts and per-expert heterogeneous tiers, then resolves them to shared operator
    capability keys. Expert capacity is calculated from actual archive files, including codebooks and safetensors
    metadata, making VRAM fast-path decisions more conservative than counting index payload alone.
    """

    from .store import Manifest

    root = Path(model_dir)
    manifest = Manifest(str(root))
    if not manifest.projection_vq:
        raise RuntimeError(
            "极限模式只接受可由公共 packed 算子直接计算的三投影 TPQ "
            "归档；当前模型会展开专家索引。"
        )
    formats: set[str] = set()
    dimensions: set[int] = set()
    codebooks: set[int] = set()
    layers = tuple(sorted(int(layer) for layer in manifest.expert_files))
    for layer in layers:
        capability = manifest.projection_operator_capability(layer)
        formats.update(str(value) for value in capability["packed_formats"])
        dimensions.update(int(value) for value in capability["code_dims"])
        codebooks.update(int(value) for value in capability["codebook_sizes"])
    unsupported = sorted(
        value
        for value in formats
        if not value.startswith("p")
        or not value.removeprefix("p").isdigit()
        or not 8 <= int(value.removeprefix("p")) <= 16
    )
    if unsupported:
        raise RuntimeError(
            "极限模式不支持以下专家索引格式：" + ", ".join(unsupported)
        )
    files = {root / str(name) for name in manifest.expert_files.values()}
    missing = sorted(str(path) for path in files if not path.is_file())
    if missing:
        raise RuntimeError("极限模式缺少专家文件：" + ", ".join(missing))
    return CompactArchiveCapacity(
        expert_bytes=sum(path.stat().st_size for path in files),
        layers=layers,
        packed_formats=tuple(sorted(formats)),
        code_dims=tuple(sorted(dimensions)),
        codebook_sizes=tuple(sorted(codebooks)),
    )


def choose_extreme_strategy(
    *,
    compact_expert_bytes: int,
    fixed_gpu_bytes: int,
    gpu_limit_bytes: int,
) -> str:
    """Reuse the shared resident fast path directly when VRAM can hold the complete compact model."""

    required = int(compact_expert_bytes) + int(fixed_gpu_bytes)
    return "full-gpu" if required <= int(gpu_limit_bytes) else "layered"


def plan_extreme_layer_placement(
    layer_bytes: Mapping[int, int],
    *,
    available_ram_bytes: int,
    gpu_expert_bytes: int,
    ram_reserve_bytes: int = GIB,
    fixed_ram_bytes: int = 0,
    fixed_gpu_bytes: int = 0,
) -> ExtremeLayerPlacement:
    """Place the largest contiguous layer prefix in RAM and all remaining complete layers in VRAM.

    Planning uses only compact expert payload bytes. Callers must first subtract shared codebooks and loading workspace
    through ``fixed_ram_bytes`` and subtract dense weights, KV, GPU workspace, and staging through ``fixed_gpu_bytes``.
    Splitting half a layer's experts across devices and falling back to runtime disk reads are both forbidden.
    """

    ordered = tuple(sorted((int(k), int(v)) for k, v in layer_bytes.items()))
    if any(size < 0 for _layer, size in ordered):
        raise ValueError("极限模式层字节数不能为负")
    ram_capacity = max(
        0,
        int(available_ram_bytes)
        - int(ram_reserve_bytes)
        - int(fixed_ram_bytes),
    )
    gpu_capacity = max(0, int(gpu_expert_bytes) - int(fixed_gpu_bytes))
    ram_layers: list[int] = []
    gpu_layers: list[int] = []
    ram_used = 0
    overflow = False
    for layer, size in ordered:
        if not overflow and ram_used + size <= ram_capacity:
            ram_layers.append(layer)
            ram_used += size
        else:
            overflow = True
            gpu_layers.append(layer)
    gpu_used = sum(dict(ordered)[layer] for layer in gpu_layers)
    if gpu_used > gpu_capacity:
        raise RuntimeError(
            "极限模式容量不足：RAM 保留 "
            f"{ram_reserve_bytes / GIB:.2f} GiB 后只能放 "
            f"{len(ram_layers)}/{len(ordered)} 层，剩余 "
            f"{gpu_used / GIB:.2f} GiB 专家需要 VRAM，但可用仅 "
            f"{gpu_capacity / GIB:.2f} GiB。请降低上下文、关闭其他进程，"
            "或换用更小模型。"
        )
    return ExtremeLayerPlacement(
        ram_layers=tuple(ram_layers),
        gpu_layers=tuple(gpu_layers),
        ram_bytes=ram_used,
        gpu_bytes=gpu_used,
        ram_capacity=ram_capacity,
        gpu_capacity=gpu_capacity,
    )


def plan_extreme_expert_placement(
    expert_bytes: Mapping[Hashable, int],
    precision_scores: Mapping[Hashable, float],
    *,
    placement_groups: Mapping[Hashable, Hashable] | None = None,
    available_ram_bytes: int,
    gpu_expert_bytes: int,
    ram_reserve_bytes: int = GIB,
    fixed_ram_bytes: int = 0,
    fixed_gpu_bytes: int = 0,
) -> ExtremeExpertPlacement:
    """Keep precision-budgeted experts on GPU while satisfying RAM capacity.

    TPQ quantizers may assign more packed bits to frequently routed or more
    sensitive experts.  The score is deliberately supplied by the manifest
    adapter: this common planner neither recognizes model names nor assumes a
    particular set of bit widths.
    """

    sizes = {key: int(value) for key, value in expert_bytes.items()}
    if any(value < 0 for value in sizes.values()):
        raise ValueError("extreme expert bytes cannot be negative")
    ram_capacity = max(
        0,
        int(available_ram_bytes)
        - int(ram_reserve_bytes)
        - int(fixed_ram_bytes),
    )
    gpu_capacity = max(0, int(gpu_expert_bytes) - int(fixed_gpu_bytes))
    ram_used = sum(sizes.values())
    if placement_groups is None:
        ranked = sorted(
            sizes,
            key=lambda key: (
                float(precision_scores.get(key, 0.0)),
                sizes[key],
                str(key),
            ),
            reverse=True,
        )
    else:
        missing = sizes.keys() - placement_groups.keys()
        if missing:
            raise ValueError(
                "extreme placement groups do not cover every expert"
            )
        grouped: dict[Hashable, list[Hashable]] = {}
        for key in sizes:
            grouped.setdefault(placement_groups[key], []).append(key)
        ranked_rows: list[tuple[int, float, int, str, Hashable]] = []
        for keys in grouped.values():
            keys.sort(
                key=lambda key: (
                    float(precision_scores.get(key, 0.0)),
                    sizes[key],
                    str(key),
                ),
                reverse=True,
            )
            maximum = max(
                (float(precision_scores.get(key, 0.0)) for key in keys),
                default=0.0,
            )
            for rank, key in enumerate(keys):
                relative_score = (
                    float(precision_scores.get(key, 0.0)) / maximum
                    if maximum > 0
                    else 0.0
                )
                # Rank is primary: each group contributes its hottest expert
                # before any group contributes its second hottest.  TPQ's
                # fixed-per-layer budgets are only comparable within a layer,
                # while every routed layer runs once per token.
                ranked_rows.append(
                    (-rank, relative_score, sizes[key], str(key), key)
                )
        ranked_rows.sort(reverse=True)
        ranked = [row[-1] for row in ranked_rows]
    gpu_keys: list[Hashable] = []
    for key in ranked:
        if ram_used <= ram_capacity:
            break
        gpu_keys.append(key)
        ram_used -= sizes[key]
    if ram_used > ram_capacity:
        raise RuntimeError("extreme mode cannot satisfy RAM expert capacity")
    gpu_set = set(gpu_keys)
    gpu_used = sum(sizes[key] for key in gpu_keys)
    if gpu_used > gpu_capacity:
        raise RuntimeError(
            "extreme mode precision-weighted GPU experts need "
            f"{gpu_used / GIB:.2f} GiB, but only "
            f"{gpu_capacity / GIB:.2f} GiB is available"
        )
    return ExtremeExpertPlacement(
        ram_keys=tuple(key for key in sizes if key not in gpu_set),
        gpu_keys=tuple(gpu_keys),
        ram_bytes=ram_used,
        gpu_bytes=gpu_used,
        ram_capacity=ram_capacity,
        gpu_capacity=gpu_capacity,
    )


def load_expert_residency_scores(
    path: str | os.PathLike[str],
) -> dict[tuple[int, int], float]:
    """Load portable expert hotness scores without recognizing a model.

    Quantizers can emit either the compact TPQ score schema or TPQ's existing
    expert-preference audit.  Both describe layer/expert coordinates and a
    non-negative score; runtime placement remains independent of architecture.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    format_name = str(payload.get("format", ""))
    output: dict[tuple[int, int], float] = {}
    if format_name == "tpq-expert-residency-scores-v1":
        for coordinate, value in (payload.get("scores") or {}).items():
            layer_text, separator, expert_text = str(coordinate).partition(":")
            if not separator:
                raise ValueError(
                    "expert residency score keys must use layer:expert"
                )
            output[(int(layer_text), int(expert_text))] = float(value)
    elif format_name == "tpq-expert-projection-preference-map-v1":
        for layer_text, layer_data in (payload.get("layers") or {}).items():
            for expert in layer_data.get("experts", ()):
                output[(int(layer_text), int(expert["expert"]))] = float(
                    expert["route_mass"]
                )
    else:
        raise ValueError(
            "unsupported expert residency score format: " + format_name
        )
    if not output:
        raise ValueError("expert residency score file contains no experts")
    if any(not math.isfinite(value) or value < 0 for value in output.values()):
        raise ValueError("expert residency scores must be finite and non-negative")
    return output


def configure_extreme_environment() -> None:
    """Apply unified extreme mode; the launcher may still override the VRAM reserve through explicit CLI arguments."""

    os.environ["TPQ_EXTREME_MODE"] = "1"
    os.environ["TPQ_FULL_RESIDENT"] = "1"
    os.environ["TPQ_RAM_MIRROR"] = "0"
    os.environ["TPQ_RAM_RESERVE_GB"] = str(EXTREME_RAM_RESERVE_GIB)
    os.environ["TPQ_RESIDENT_RESERVE_GB"] = str(EXTREME_RAM_RESERVE_GIB)
    # Extreme mode already reduces free RAM to the 1 GiB safety threshold. Mlocking tens of GiB here would force
    # the kernel to swap out Python/file pages, adding no capacity while substantially slowing startup and decode.
    # Keep the normal small pinned-staging area; users with additional RAM can still override it explicitly.
    os.environ.setdefault("TPQ_HOST_PIN_GB", "0")
    # Keep only a few materialized expert temporaries at once so 12 large concurrent experts do not push the
    # 1 GiB system margin into swap. This conservative concurrency has little effect on sequential SSD throughput.
    os.environ.setdefault("TPQ_LOAD_WORKERS", "2")
    os.environ.setdefault("TPQ_VRAM_RESERVE_GB", "0.25")
    # Dense weights, KV, and shared-operator workspaces are physically allocated before the packed arena.
    # Only a small temporary margin for the decode hot path is needed here. Reserving another 1 GiB as in normal mode
    # would cost a 16 GiB card about 128 expert slots, leaving it just short of one 40-layer Top-8 routing round
    # and causing cyclic LRU thrashing across tokens.
    os.environ.setdefault("TPQ_VRAM_RUNTIME_GB", "0.25")
    os.environ["TPQ_VRAM_WATCH"] = "0"


def extreme_enabled() -> bool:
    return os.environ.get("TPQ_EXTREME_MODE", "0") != "0"


__all__ = [
    "AutoExtremeDecision",
    "CompactArchiveCapacity",
    "EXTREME_GPU_LOAD_WORKSPACE_GIB",
    "EXTREME_RAM_LOAD_WORKSPACE_GIB",
    "EXTREME_RAM_RESERVE_GIB",
    "ExtremeLayerPlacement",
    "choose_extreme_strategy",
    "configure_extreme_environment",
    "detect_auto_extreme",
    "effective_available_memory_bytes",
    "extreme_enabled",
    "inspect_compact_projection_archive",
    "plan_auto_extreme",
    "plan_extreme_layer_placement",
]
