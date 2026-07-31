"""发布版静态检查与容量规划。

该命令不加载模型权重，也不编译 CUDA 扩展；适合在正式启动前检查模型文件、
RAM/VRAM 余量、GPU 可见性、P2P 和 GLM TP2–TP8 容量。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .presets import (
    choose_ep_layout,
    detect_architecture,
    load_arch_config,
    load_manifest,
    resolve_preset,
)


GIB = 2**30


def _runtime_layers(manifest: dict[str, Any]) -> list[int]:
    config = manifest["config"]
    available = {int(layer) for layer in manifest.get("expert_files", {})}
    configured = config.get("moe_layers")
    if configured is not None:
        return sorted(int(layer) for layer in configured if int(layer) in available)
    limit = int(config.get("n_layers", 0))
    return sorted(layer for layer in available if not limit or layer < limit)


def _default_expert_kind(manifest: dict[str, Any]) -> str:
    kinds = manifest.get("quant", {}).get("vq", {})
    configured = str(manifest.get("quant", {}).get("expert", ""))
    for kind in kinds:
        if configured.startswith(kind):
            return kind
    if kinds:
        return next(iter(kinds))
    raise ValueError("cccp.json 的 quant.vq 为空")


def _expert_kind(
    manifest: dict[str, Any],
    layer: int,
    expert_id: int,
) -> str:
    tiers = manifest.get("tiers_per_layer", {})
    value = tiers.get(str(layer), tiers.get(layer))
    if value is not None:
        if expert_id >= len(value):
            raise ValueError(f"layer {layer} 的 tiers_per_layer 长度不足")
        return str(value[expert_id])
    return _default_expert_kind(manifest)


def _slot_bytes(manifest: dict[str, Any], kind: str) -> int:
    if kind == "d":
        return 0
    base = kind.rstrip("z")
    dimensions = manifest["quant"]["vq"].get(base)
    if dimensions is None:
        raise ValueError(f"未知专家量化档：{kind}")
    vector_dim, codebook_size = (int(value) for value in dimensions)
    config = manifest["config"]
    hidden = int(config["hidden"])
    intermediate = int(config["moe_inter"])
    item_size = 2 if codebook_size > 256 else 1
    return (
        2 * intermediate * (hidden // vector_dim)
        + hidden * (intermediate // vector_dim)
    ) * item_size


def expert_plan_bytes(
    manifest: dict[str, Any],
    tp: int,
    layout: str | None = None,
) -> tuple[str, tuple[int, ...]]:
    if tp < 1:
        raise ValueError("tp 必须为正整数")
    if tp == 1:
        layout = "expert"
    elif layout is None:
        layout = choose_ep_layout(manifest, tp)
    if layout not in {"expert", "tensor"}:
        raise ValueError(f"未知专家布局：{layout}")

    config = manifest["config"]
    n_experts = int(config["n_experts"])
    totals = [0] * tp
    for layer in _runtime_layers(manifest):
        for expert_id in range(n_experts):
            size = _slot_bytes(
                manifest,
                _expert_kind(manifest, layer, expert_id),
            )
            if not size:
                continue
            if layout == "tensor":
                if size % tp:
                    raise ValueError(
                        f"专家槽 {size} bytes 不能按 tp={tp} 等分"
                    )
                for rank in range(tp):
                    totals[rank] += size // tp
            else:
                rank = min(tp - 1, expert_id * tp // n_experts)
                totals[rank] += size
    return layout, tuple(totals)


def _required_files(root: Path, manifest: dict[str, Any]) -> list[Path]:
    names = {
        "cccp.json",
        "tokenizer.json",
        *manifest.get("expert_files", {}).values(),
    }
    dense_files = manifest.get("dense_files")
    if dense_files:
        names.update(str(name) for name in dense_files)
    else:
        names.add(manifest.get("dense_file", "dense.safetensors"))
    for key in ("dense_audit_file",):
        value = manifest.get(key)
        if value:
            names.add(value)
    names.update(
        str(name)
        for name in manifest.get("expert_audit_files", {}).values()
    )
    for key in ("mtp_file", "dspark_file"):
        value = manifest.get(key)
        if value:
            names.add(value)
    if manifest.get("mtp_file"):
        layer = int(manifest["config"]["n_layers"])
        names.add(f"experts.L{layer:02d}.safetensors")
    return sorted((root / str(name) for name in names), key=lambda path: path.name)


def resident_expert_bytes(
    root: Path,
    manifest: dict[str, Any],
) -> int:
    """RAM profile resident bytes, including GLM's optional MTP expert layer."""
    if detect_architecture(manifest) == "kimi_k3":
        total = 0
        for name in manifest.get("expert_audit_files", {}).values():
            with (root / str(name)).open("r", encoding="utf-8") as handle:
                audit = json.load(handle)
            total += sum(
                int(item.get("gu_bytes", 0))
                + int(item.get("down_bytes", 0))
                for item in audit.get("experts", {}).values()
            )
        return total
    total = expert_plan_bytes(manifest, 1)[1][0]
    if manifest.get("mtp_file"):
        layer = int(manifest["config"]["n_layers"])
        extra = root / f"experts.L{layer:02d}.safetensors"
        if extra.is_file() and layer not in _runtime_layers(manifest):
            count = int(manifest["config"]["n_experts"])
            total += count * _slot_bytes(
                manifest,
                _default_expert_kind(manifest),
            )
    return total


def _initial_kv_gib(architecture: str, max_ctx: int) -> float:
    if architecture == "dsv4":
        return 0.2
    initial = min(max(0, int(max_ctx)), 4096)
    return 2.3 + 0.09 * initial / 1024


def fixed_vram_gib(
    root: Path,
    manifest: dict[str, Any],
    architecture: str,
    max_ctx: int,
    environment: dict[str, str] | None = None,
) -> float:
    config = manifest["config"]
    if architecture == "kimi_k3":
        dense = 0.0
        audit_name = manifest.get("dense_audit_file")
        if audit_name and (root / str(audit_name)).is_file():
            with (root / str(audit_name)).open(
                "r",
                encoding="utf-8",
            ) as handle:
                audit = json.load(handle)
            dense = sum(
                int(info.get("bytes", 0))
                for shard in audit.get("shards", {}).values()
                for name, info in shard.get("tensor_audit", {}).items()
                if name.startswith("language_model.")
            ) / GIB
        if not dense:
            dense = sum(
                (root / str(name)).stat().st_size
                for name in manifest.get("dense_files", [])
                if (root / str(name)).is_file()
            ) / GIB
        return dense + 3.0 + _initial_kv_gib(architecture, max_ctx)
    dense_path = root / manifest.get("dense_file", "dense.safetensors")
    dense = dense_path.stat().st_size / GIB
    head = int(config["vocab"]) * int(config["hidden"]) * 4 / GIB
    attachment = 0.0
    if architecture == "glm":
        name = manifest.get("mtp_file")
        if name and (root / name).is_file():
            attachment = (root / name).stat().st_size / GIB
    else:
        from .capacity import dsv4_dense_runtime_bytes

        setting = (environment or {}).get("TPQ_DENSE_BF16")
        runtime_bytes = dsv4_dense_runtime_bytes(dense_path, setting)
        if runtime_bytes is not None:
            dense = runtime_bytes / GIB
            head = 0.0
        else:
            name = manifest.get("dspark_file")
            if name and (root / name).is_file():
                attachment = 2.5
    return dense + head + attachment + 1.5 + _initial_kv_gib(
        architecture,
        max_ctx,
    )


def _gpu_inventory() -> tuple[list[dict[str, Any]], list[list[bool]]]:
    try:
        import torch
    except Exception:
        return [], []
    if not torch.cuda.is_available():
        return [], []
    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        prop = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": prop.name,
                "free_gib": free / GIB,
                "total_gib": total / GIB,
                "capability": f"{prop.major}.{prop.minor}",
            }
        )
    peer = [
        [
            True if left == right else bool(torch.cuda.can_device_access_peer(left, right))
            for right in range(len(devices))
        ]
        for left in range(len(devices))
    ]
    return devices, peer


def _memory_status() -> tuple[float, float]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return memory.available / GIB, memory.total / GIB
    except Exception:
        return 0.0, 0.0


def _fmt(values: tuple[int, ...]) -> str:
    return "/".join(f"{value / GIB:.1f}" for value in values)


def _print_matrix(
    manifest: dict[str, Any],
    fixed_gib: float,
    reserve_vram_gib: float,
) -> None:
    if detect_architecture(manifest) == "kimi_k3":
        print(
            "\nKimi K3 使用 dense 按层放置 + packed MoE 二维并行；"
            "精确容量在模型装载前按审计文件计算。"
        )
        return
    if detect_architecture(manifest) != "glm":
        print("\nTP2–TP8：DeepSeek-V4 当前没有多卡执行路径。")
        return
    targets = (
        ("RTX 5090 32GB", 31.4),
        ("H20 96GB", 96.0),
        ("H20-3e 140GB", 139.8),
    )
    print("\nGLM 多卡容量矩阵（静态规划，不等于实机推理验收）：")
    print("TP  布局    专家/卡GiB                 最低单卡GiB  5090  H20-96  H20-3e")
    for tp in range(2, 9):
        layout, per_rank = expert_plan_bytes(manifest, tp)
        required = tuple(
            value / GIB
            + reserve_vram_gib
            + (fixed_gib if rank == 0 else 0.0)
            for rank, value in enumerate(per_rank)
        )
        minimum = max(required)
        fits = [
            "是" if minimum <= capacity else "否"
            for _name, capacity in targets
        ]
        print(
            f"{tp:<3} {layout:<7} {_fmt(per_rank):<27} "
            f"{minimum:>10.1f}  {fits[0]:>4}  {fits[1]:>6}  {fits[2]:>7}"
        )


def _self_test() -> None:
    glm = {
        "format": "cccp-1",
        "config": {
            "n_layers": 2,
            "hidden": 64,
            "n_experts": 4,
            "moe_inter": 32,
            "moe_layers": [0, 1],
        },
        "quant": {"expert": "v256", "vq": {"v": [4, 256]}},
        "expert_files": {"0": "a", "1": "b"},
        "tiers_per_layer": {"0": "vvvv", "1": "vdvv"},
    }
    dsv4 = {
        **glm,
        "config": {**glm["config"], "hc_mult": 4},
    }
    kimi = {
        **glm,
        "model_family": "kimi_k3",
        "config": {
            **glm["config"],
            "kda_layers": [0],
            "routed_hidden": 64,
        },
    }
    assert detect_architecture(glm) == "glm"
    assert detect_architecture(dsv4) == "dsv4"
    assert detect_architecture(kimi) == "kimi_k3"
    assert load_arch_config("glm")["supports_parallel"] is True
    assert load_arch_config("dsv4")["supports_parallel"] is False
    assert load_arch_config("kimi_k3")["supports_parallel"] is True
    assert choose_ep_layout(kimi, 8) == "tensor"
    total = sum(expert_plan_bytes(glm, 1)[1])
    for tp in range(2, 9):
        layout, planned = expert_plan_bytes(glm, tp)
        assert sum(planned) == total
        expected = "tensor" if 32 % tp == 0 else "expert"
        assert layout == expected
    print("[tpq-check] 基础测试通过：架构识别、配置加载、TP2–TP8 专家规划")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TPQ 发布版环境与容量检查")
    parser.add_argument("--model", help="CCCP 模型目录")
    parser.add_argument(
        "--profile",
        choices=("auto", "ram", "parallel"),
        default="auto",
    )
    parser.add_argument("--tp", type=int)
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES，例如 0,1")
    parser.add_argument("--max-ctx", type=int)
    parser.add_argument("--ram-reserve-gb", type=float)
    parser.add_argument("--vram-reserve-gb", type=float)
    parser.add_argument("--matrix", action="store_true", help="输出 GLM TP2–TP8 容量矩阵")
    parser.add_argument("--self-test", action="store_true", help="运行无模型基础测试")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.self_test:
        _self_test()
        if not args.model:
            return
    if not args.model:
        raise SystemExit("需要 --model，或单独使用 --self-test")
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    try:
        preset = resolve_preset(args.model, profile=args.profile, tp=args.tp)
        root, manifest = load_manifest(args.model)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[tpq-check] 模型配置失败：{exc}") from exc

    max_ctx = int(
        preset.defaults.get("max_ctx", 4096)
        if args.max_ctx is None
        else args.max_ctx
    )
    reserve_ram = float(
        preset.environment.get("TPQ_RESIDENT_RESERVE_GB", 32)
        if args.ram_reserve_gb is None
        else args.ram_reserve_gb
    )
    reserve_vram = float(
        preset.environment.get("TPQ_VRAM_RUNTIME_GB", 3)
        if args.vram_reserve_gb is None
        else args.vram_reserve_gb
    )

    missing = [path for path in _required_files(root, manifest) if not path.is_file()]
    if missing:
        print("[失败] 模型文件缺失：")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)

    expert_gib = resident_expert_bytes(root, manifest) / GIB
    fixed_gib = fixed_vram_gib(
        root,
        manifest,
        preset.architecture,
        max_ctx,
        preset.environment,
    )
    available_ram, total_ram = _memory_status()
    ram_need = expert_gib + reserve_ram + 12.0

    print(
        f"[通过] 模型={root.name}；架构={preset.architecture}；"
        f"profile={preset.profile}；tp={preset.tp}"
    )
    print(
        f"[通过] 文件={len(_required_files(root, manifest))}；"
        f"专家 arena={expert_gib:.1f}GiB；固定显存估算={fixed_gib:.1f}GiB"
    )
    if available_ram:
        ram_ok = available_ram >= ram_need
        print(
            f"[{'通过' if ram_ok else '警告'}] RAM 可用/总量="
            f"{available_ram:.1f}/{total_ram:.1f}GiB；"
            f"全量专家建议至少={ram_need:.1f}GiB"
        )

    devices, peer = _gpu_inventory()
    if not devices:
        print("[警告] 没有可用 CUDA；只能检查文件与配置")
    else:
        for device in devices:
            print(
                f"[GPU] cuda:{device['index']} {device['name']} "
                f"CC={device['capability']} "
                f"空闲/总量={device['free_gib']:.1f}/{device['total_gib']:.1f}GiB"
            )
        if preset.tp > len(devices):
            raise SystemExit(
                f"[失败] tp={preset.tp}，但只可见 {len(devices)} 张 CUDA 卡"
            )
        if preset.tp > 1:
            missing_peer = [
                f"0->{rank}"
                for rank in range(1, preset.tp)
                if not (peer[0][rank] and peer[rank][0])
            ]
            print(
                "[通过] 主卡双向 P2P 可用"
                if not missing_peer
                else "[警告] 缺少双向 P2P：" + ",".join(missing_peer)
            )

        if preset.profile == "parallel":
            layout, per_rank = expert_plan_bytes(manifest, preset.tp)
            required = [
                value / GIB
                + reserve_vram
                + (fixed_gib if rank == 0 else 0.0)
                for rank, value in enumerate(per_rank)
            ]
        else:
            layout = "-"
            required = [fixed_gib + reserve_vram]
        enough = all(
            devices[rank]["free_gib"] >= need
            for rank, need in enumerate(required)
        )
        print(
            f"[{'通过' if enough else '警告'}] 当前配置 layout={layout}；"
            "每卡所需约="
            + "/".join(f"{value:.1f}" for value in required)
            + "GiB"
        )

    if args.matrix:
        _print_matrix(manifest, fixed_gib, reserve_vram)


if __name__ == "__main__":
    main()
