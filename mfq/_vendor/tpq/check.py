"""发布版静态检查与容量规划。

该命令不加载模型权重，也不编译 CUDA 扩展；适合在正式启动前检查模型文件、
RAM/VRAM 余量、GPU 可见性、P2P 和 GLM TP2–TP8 容量。
"""

from __future__ import annotations

import argparse
import json
import math
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

_DSV4_SPECIAL_TOKENS = {
    "bos": "<｜begin▁of▁sentence｜>",
    "eos": "<｜end▁of▁sentence｜>",
    "think_start": "<think>",
    "think_end": "</think>",
}


def dsv4_attention_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the architecture policy that controls long DSV4 attention.

    Ratio-4 layers emit one compressed key every four source tokens.  The
    Indexer is unnecessary while every compressed key fits in ``index_topk``;
    immediately after that boundary it selects the best fixed-size set.
    """
    config = manifest["config"]
    ratios = tuple(int(value) for value in config.get("compress_ratios", ()))
    index_topk = int(config.get("index_topk", 512))
    indexed_ratio = 4 if 4 in ratios else None
    return {
        "sliding_window": int(config.get("sliding_window", 0)),
        "compress_ratios": tuple(sorted(set(ratios))),
        "index_topk": index_topk,
        "indexer_after_tokens": (
            indexed_ratio * index_topk
            if indexed_ratio is not None
            else None
        ),
    }


def _dsv4_token_protocol(
    root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Read special-token IDs without importing or constructing a tokenizer."""
    tokenizer_path = root / "tokenizer.json"
    with tokenizer_path.open("r", encoding="utf-8") as handle:
        tokenizer = json.load(handle)
    ids: dict[str, int] = {}
    by_content: dict[str, int] = {}
    for item in tokenizer.get("added_tokens", ()):
        content = item.get("content")
        token_id = item.get("id")
        if isinstance(content, str) and isinstance(token_id, int):
            by_content[content] = token_id
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if isinstance(vocab, dict):
        for content, token_id in vocab.items():
            if isinstance(content, str) and isinstance(token_id, int):
                by_content.setdefault(content, token_id)
    errors: list[str] = []
    for name, content in _DSV4_SPECIAL_TOKENS.items():
        token_id = by_content.get(content)
        if token_id is None:
            errors.append(f"tokenizer 缺少 {name}={content}")
        else:
            ids[name] = token_id
    configured_eos = manifest["config"].get("eos_token_id", ())
    if isinstance(configured_eos, int):
        configured_eos = (configured_eos,)
    if "eos" in ids and ids["eos"] not in {
        int(value) for value in configured_eos
    }:
        errors.append(
            f"eos token id={ids['eos']} 不在 config.eos_token_id="
            f"{list(configured_eos)}"
        )
    return ids, tuple(errors)


def _expert_files(manifest: dict[str, Any]) -> dict[str, str]:
    direct = manifest.get("expert_files")
    if direct:
        return {str(layer): str(path) for layer, path in direct.items()}
    return {
        str(layer): str(item["path"])
        for layer, item in (
            (manifest.get("routed_experts") or {}).get("layer_files") or {}
        ).items()
    }


def _runtime_layers(manifest: dict[str, Any]) -> list[int]:
    config = manifest["config"]
    available = {int(layer) for layer in _expert_files(manifest)}
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
    if manifest.get("quant", {}).get("method") == "projection-vq":
        return "projection-vq"
    tiers = manifest.get("tiers_per_layer", {})
    value = tiers.get(str(layer), tiers.get(layer))
    if value is not None:
        if expert_id >= len(value):
            raise ValueError(f"layer {layer} 的 tiers_per_layer 长度不足")
        return str(value[expert_id])
    return _default_expert_kind(manifest)


def _slot_bytes(
    manifest: dict[str, Any],
    kind: str,
    layer: int | None = None,
) -> int:
    if kind == "d":
        return 0
    quant = manifest["quant"]
    if quant.get("method") == "projection-vq":
        if layer is None:
            raise ValueError("projection-VQ capacity requires a layer")
        layouts = quant["projection_layouts"][str(layer)]
        config = manifest["config"]
        hidden = int(config["hidden"])
        intermediate = int(config["moe_inter"])
        shapes = {
            "gate": (intermediate, hidden),
            "up": (intermediate, hidden),
            "down": (hidden, intermediate),
        }
        total = 0
        for projection, layout in layouts.items():
            spec = quant["layouts"][layout]
            rows, columns = shapes[projection]
            indices = rows * (columns // int(spec["dim"]))
            packing = str(quant["index_packing"][layout])
            bits = int(
                packing.removeprefix("packed-u").removeprefix("u")
            )
            total += (indices * bits + 7) // 8
        return total
    base = kind.rstrip("z")
    # 单字符层配额需要同时区分 v 与 vv；历史 CCCP 清单以大写
    # ``V`` 表示 vv。容量规划只应规范化存储档位，不应依赖模型架构。
    if base == "V" and "vv" in manifest["quant"]["vq"]:
        base = "vv"
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
                layer,
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
    expert_files = _expert_files(manifest)
    names = {
        "tpq.json" if (root / "tpq.json").is_file() else "cccp.json",
        *expert_files.values(),
    }
    tokenizer_files = manifest.get("tokenizer_files")
    if tokenizer_files:
        names.update(str(name) for name in tokenizer_files)
    else:
        names.add("tokenizer.json")
    names.update(str(name) for name in manifest.get("model_files", ()))
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
    for item in manifest.get("layer_audit", {}).values():
        if isinstance(item, dict) and item.get("audit_path"):
            names.add(str(item["audit_path"]))
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
                layer,
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


def kimi_parallel_rank_bytes(
    root: Path,
    manifest: dict[str, Any],
    tp: int,
) -> tuple[int, ...]:
    """Estimate Kimi's real all-rank weight residency from its audits.

    Routed experts remain packed and are tensor-sharded.  Mixed dense weights
    are expanded to their logical source dtype once at startup; column/row TP
    tensors are sharded while norms, biases and other metadata are replicated.
    This deliberately does not reuse the GLM slot formula, which assumes
    expanded expert indices and can overstate Kimi's TP requirement by several
    hundred GiB.
    """
    if tp < 1:
        raise ValueError("tp 必须为正整数")
    audit_name = manifest.get("dense_audit_file")
    if not audit_name:
        raise ValueError("Kimi parallel 容量检查需要 dense_audit_file")
    with (root / str(audit_name)).open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    entries = audit.get("entries", {})
    if not entries:
        raise ValueError("Kimi dense audit 缺少 entries")

    rank_bytes = [0] * tp
    dtype_bytes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
        "I16": 2,
        "I32": 4,
        "I64": 8,
    }
    for info in entries.values():
        source_bytes = int(info.get("source_bytes", 0))
        if not source_bytes:
            shape = info.get("logical_shape", ())
            source_bytes = math.prod(int(value) for value in shape)
            source_bytes *= dtype_bytes.get(
                str(info.get("source_dtype", "BF16")).upper(),
                2,
            )
        if str(info.get("tp_axis", "replicated")) in {"column", "row"}:
            quotient, remainder = divmod(source_bytes, tp)
            for rank in range(tp):
                rank_bytes[rank] += quotient + (rank < remainder)
        else:
            for rank in range(tp):
                rank_bytes[rank] += source_bytes

    expert_bytes = sum(
        (root / str(name)).stat().st_size
        for name in _expert_files(manifest).values()
    )
    quotient, remainder = divmod(expert_bytes, tp)
    for rank in range(tp):
        rank_bytes[rank] += quotient + (rank < remainder)
    return tuple(rank_bytes)


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
        print(
            "\nDeepSeek-V4 使用 Head/Dense/Router/packed-MoE 全层真 TP；"
            "精确逐卡容量由本次模型审计输出。"
        )
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
        "format": "tpq-1",
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
    parser.add_argument("--model", help="TPQ 模型目录")
    parser.add_argument(
        "--profile",
        choices=("auto", "ram", "resident", "parallel"),
        default="auto",
    )
    parser.add_argument("--tp", type=int)
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES，例如 0,1")
    parser.add_argument("--max-ctx", type=int)
    parser.add_argument("--ram-reserve-gb", type=float)
    parser.add_argument("--vram-reserve-gb", type=float)
    parser.add_argument("--matrix", action="store_true", help="输出 GLM TP2–TP8 容量矩阵")
    parser.add_argument("--self-test", action="store_true", help="运行无模型基础测试")
    parser.add_argument(
        "--cuda-ops",
        action="store_true",
        help="在恰好一张可见 GPU 上运行公共 packed CUDA 数值测试",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.self_test:
        _self_test()
        if not args.model and not args.cuda_ops:
            return
    if args.cuda_ops:
        from .ops.selftest import projection_cuda_selftest

        report = projection_cuda_selftest()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["all_passed"]:
            raise SystemExit(1)
        if not args.model:
            return
    if not args.model:
        raise SystemExit(
            "需要 --model，或单独使用 --self-test/--cuda-ops"
        )

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

    if preset.architecture == "dsv4":
        token_ids, token_errors = _dsv4_token_protocol(root, manifest)
        if token_errors:
            for error in token_errors:
                print(f"[失败] DSV4 对话协议：{error}")
            raise SystemExit(1)
        policy = dsv4_attention_policy(manifest)
        print(
            "[通过] DSV4 对话协议："
            f"BOS={token_ids['bos']}；EOS={token_ids['eos']}；"
            f"<think>={token_ids['think_start']}；"
            f"</think>={token_ids['think_end']}；"
            "reasoning=chat/high/max"
        )
        boundary = policy["indexer_after_tokens"]
        print(
            "[通过] DSV4 注意力："
            f"raw-window={policy['sliding_window']}；"
            f"compress-ratios={list(policy['compress_ratios'])}；"
            f"Indexer Top-{policy['index_topk']} "
            + (
                f"在位置>{boundary}启用"
                if boundary is not None
                else "未配置 ratio-4 压缩层"
            )
        )
        if manifest.get("quant", {}).get("method") == "projection-vq":
            # Run exactly the same capability normalization used by model
            # construction so unsupported packed widths fail here, before
            # a multi-GiB weight load starts.
            from .store import Manifest

            model_manifest = Manifest(str(root))
            capabilities = {
                tuple(
                    model_manifest.projection_operator_capability(layer)[
                        "packed_formats"
                    ]
                )
                for layer in model_manifest.expert_files
            }
            formats = sorted(
                {item for capability in capabilities for item in capability},
                key=lambda item: int(item.removeprefix("p")),
            )
            print(
                "[通过] DSV4 公共 packed 能力："
                f"formats={formats}；"
                f"projection-layouts={len(capabilities)}"
            )

    expert_gib = resident_expert_bytes(root, manifest) / GIB
    fixed_gib = fixed_vram_gib(
        root,
        manifest,
        preset.architecture,
        max_ctx,
        preset.environment,
    )
    available_ram, total_ram = _memory_status()
    ram_need = (
        reserve_ram + 12.0
        if preset.profile == "resident"
        else expert_gib + reserve_ram + 12.0
    )

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

        if preset.profile == "resident":
            layout = "resident"
            _layout, per_rank = expert_plan_bytes(manifest, 1)
            required = [
                per_rank[0] / GIB + reserve_vram + fixed_gib
            ]
        elif preset.profile == "parallel":
            if preset.architecture == "kimi_k3":
                layout = "tensor"
                per_rank = kimi_parallel_rank_bytes(
                    root,
                    manifest,
                    preset.tp,
                )
                runtime_gib = 3.0 + _initial_kv_gib(
                    preset.architecture,
                    max_ctx,
                )
                required = [
                    value / GIB + runtime_gib + reserve_vram
                    for value in per_rank
                ]
            else:
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
