"""统一启动入口：自动识别模型、加载专属预设，再进入聊天或 API 服务。"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .presets import apply_preset_environment, resolve_preset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TPQ 通用启动器（自动识别 GLM / DeepSeek-V4）",
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("chat", "serve"),
        default="chat",
        help="chat=CLI 对话，serve=OpenAI 兼容 API",
    )
    parser.add_argument("--model", required=True, help="CCCP 模型目录")
    parser.add_argument(
        "--profile",
        choices=("auto", "ram", "parallel"),
        default="auto",
        help="auto 根据模型和 --tp 选择；DeepSeek 当前仅 ram",
    )
    parser.add_argument("--tp", type=int, help="GLM 专家并行卡数")
    parser.add_argument(
        "--gpus",
        help="设置 CUDA_VISIBLE_DEVICES，例如 0 或 0,1,2",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="覆盖模型专属配置中的计算设备",
    )
    parser.add_argument("--max-ctx", type=int)
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--cache-gb", type=float, help="主机专家缓存预算")
    parser.add_argument("--vram-gb", type=float, help="主卡专家显存缓存预算")
    parser.add_argument(
        "--ram-reserve-gb",
        type=float,
        help="至少留给系统/运行时的 RAM；同时控制镜像与全量常驻判定",
    )
    parser.add_argument(
        "--vram-reserve-gb",
        type=float,
        help="每卡至少保留的显存",
    )
    parser.add_argument("--pin-gb", type=float, help="RAM 模式锁页热专家预算")
    parser.add_argument("--spec", type=int)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--think", action="store_true", help="CLI 开启 Think")
    parser.add_argument("--prompt", help="CLI 单轮提示词")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--served-model-name")
    parser.add_argument(
        "--reasoning",
        choices=("chat", "high", "max"),
        help="API 默认推理模式；chat 表示关闭 Think",
    )
    parser.add_argument("--max-queue", type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--metrics-jsonl")
    parser.add_argument("--cors-allow-origin", action="append", default=[])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出识别结果和最终配置，不加载模型",
    )
    return parser


def _value(args: argparse.Namespace, preset: Any, name: str) -> Any:
    value = getattr(args, name)
    return preset.defaults.get(name) if value is None else value


def _configured(
    args: argparse.Namespace,
    preset: Any,
    argument: str,
    config_key: str,
) -> Any:
    value = getattr(args, argument)
    return preset.defaults.get(config_key) if value is None else value


def _apply_environment(
    args: argparse.Namespace,
    preset: Any,
) -> None:
    if args.gpus:
        devices = [part.strip() for part in args.gpus.split(",") if part.strip()]
        if len(devices) < preset.tp:
            raise ValueError(
                f"--gpus 只给出 {len(devices)} 张卡，但 tp={preset.tp}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    apply_preset_environment(preset)

    if args.ram_reserve_gb is not None:
        value = str(args.ram_reserve_gb)
        os.environ["TPQ_RAM_RESERVE_GB"] = value
        os.environ["TPQ_RESIDENT_RESERVE_GB"] = value
    if args.vram_reserve_gb is not None:
        value = str(args.vram_reserve_gb)
        os.environ["TPQ_VRAM_RESERVE_GB"] = value
        os.environ["TPQ_VRAM_RUNTIME_GB"] = value
    if args.pin_gb is not None:
        os.environ["TPQ_PIN_GB"] = str(args.pin_gb)


def _summary(args: argparse.Namespace, preset: Any) -> None:
    device = _value(args, preset, "device")
    max_ctx = _value(args, preset, "max_ctx")
    spec = _value(args, preset, "spec")
    layout = preset.ep_layout or "-"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "系统默认")
    print(
        "[tpq-launch] "
        f"模型={preset.model_dir.name}；架构={preset.architecture}；"
        f"profile={preset.profile}；device={device}；tp={preset.tp}；"
        f"layout={layout}；max_ctx={max_ctx}；spec={spec}；"
        f"CUDA_VISIBLE_DEVICES={visible}",
        flush=True,
    )
    print(
        "[tpq-launch] 内存预留："
        f"RAM={os.environ.get('TPQ_RESIDENT_RESERVE_GB', 'auto')}GB；"
        f"VRAM={os.environ.get('TPQ_VRAM_RUNTIME_GB', 'auto')}GB；"
        f"锁页={os.environ.get('TPQ_PIN_GB', '0')}GB",
        flush=True,
    )


def _chat_argv(args: argparse.Namespace, preset: Any) -> list[str]:
    result = [
        "--model",
        str(preset.model_dir),
        "--device",
        str(_value(args, preset, "device")),
        "--tp",
        str(preset.tp),
        "--max-ctx",
        str(_value(args, preset, "max_ctx")),
        "--spec",
        str(_value(args, preset, "spec")),
        "--temp",
        str(_configured(args, preset, "temp", "temperature")),
        "--top-p",
        str(_value(args, preset, "top_p")),
    ]
    max_new = _value(args, preset, "max_new")
    if max_new is None or int(max_new) <= 0:
        result.append("--no-max-new")
    else:
        result.extend(("--max-new", str(max_new)))
    if args.cache_gb is not None:
        result.extend(("--cache-gb", str(args.cache_gb)))
    if args.vram_gb is not None:
        result.extend(("--vram-gb", str(args.vram_gb)))
    if args.think:
        result.append("--think")
    if args.prompt is not None:
        result.extend(("--prompt", args.prompt))
    return result


def _serve_argv(args: argparse.Namespace, preset: Any) -> list[str]:
    result = [
        "--model",
        str(preset.model_dir),
        "--device",
        str(_value(args, preset, "device")),
        "--tp",
        str(preset.tp),
        "--max-ctx",
        str(_value(args, preset, "max_ctx")),
        "--spec",
        str(_value(args, preset, "spec")),
        "--host",
        str(_value(args, preset, "host")),
        "--port",
        str(_value(args, preset, "port")),
        "--default-reasoning",
        str(_value(args, preset, "reasoning")),
        "--max-queue",
        str(_value(args, preset, "max_queue")),
    ]
    if args.cache_gb is not None:
        result.extend(("--cache-gb", str(args.cache_gb)))
    if args.vram_gb is not None:
        result.extend(("--vram-gb", str(args.vram_gb)))
    if args.served_model_name:
        result.extend(("--served-model-name", args.served_model_name))
    if args.api_key:
        result.extend(("--api-key", args.api_key))
    if args.metrics_jsonl:
        result.extend(("--metrics-jsonl", args.metrics_jsonl))
    for origin in args.cors_allow_origin:
        result.extend(("--cors-allow-origin", origin))
    return result


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        preset = resolve_preset(
            args.model,
            profile=args.profile,
            tp=args.tp,
        )
        _apply_environment(args, preset)
    except (OSError, ValueError) as exc:
        print(f"[tpq-launch] 配置错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if _value(args, preset, "device") == "cpu" and preset.tp > 1:
        raise SystemExit("[tpq-launch] CPU 模式不能使用 tp > 1")

    _summary(args, preset)
    if args.dry_run:
        return

    if args.action == "serve":
        from .serve import main as serve_main

        serve_main(_serve_argv(args, preset))
    else:
        from .chat import main as chat_main

        chat_main(_chat_argv(args, preset))


if __name__ == "__main__":
    main()
