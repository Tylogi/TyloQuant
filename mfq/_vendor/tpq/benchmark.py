"""可复核的 TPQ 单请求 decode 基准。

该入口只测模型加载完成后的自回归 decode；模型加载、prefill 和 warmup 单独记录，
不混入 token/s。生成固定步数且忽略 EOS，避免不同量化模型提前结束导致样本长度
不一致。结果可打印并保存为 JSON。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

from .presets import resolve_preset


def _source_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"git_commit": commit, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("TPQ_SOURCE_COMMIT")
        return {
            "git_commit": commit,
            "git_dirty": None if commit is None else False,
        }


def _cpu_hardware(torch: Any) -> dict[str, Any]:
    name = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    cpuinfo_text = ""
    instruction_sets: list[str] = []
    if cpuinfo.is_file():
        cpuinfo_text = cpuinfo.read_text(
            encoding="utf-8", errors="replace"
        )
        for line in cpuinfo_text.splitlines():
            if line.lower().startswith("model name"):
                name = line.split(":", 1)[-1].strip()
            if line.lower().startswith(("flags", "features")):
                flags = set(line.split(":", 1)[-1].strip().split())
                instruction_sets = [
                    feature
                    for feature in (
                        "avx2",
                        "avx512f",
                        "avx512bw",
                        "avx512vbmi",
                        "avx512_vnni",
                    )
                    if feature in flags
                ]
            if name and instruction_sets:
                break
    logical = os.cpu_count()
    physical = None
    memory_gib = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        memory_gib = psutil.virtual_memory().total / 2**30
    except ImportError:
        pass
    if physical is None and cpuinfo_text:
        packages_and_cores: set[tuple[str, str]] = set()
        for block in cpuinfo_text.split("\n\n"):
            fields = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            if "physical id" in fields and "core id" in fields:
                packages_and_cores.add(
                    (fields["physical id"], fields["core id"])
                )
        physical = len(packages_and_cores) or None
    numa_nodes = None
    numa_online = Path("/sys/devices/system/node/online")
    if numa_online.is_file():
        numa_nodes = numa_online.read_text(encoding="ascii").strip()
    return {
        "name": name,
        "architecture": platform.machine(),
        "physical_cores": physical,
        "logical_cpus": logical,
        "inference_threads": torch.get_num_threads(),
        "numa_nodes_online": numa_nodes,
        "memory_gib": memory_gib,
        "instruction_sets": instruction_sets,
        "torch": torch.__version__,
    }


def _process_memory() -> dict[str, float | None]:
    """返回当前进程的常驻内存与历史峰值，单位 GiB。"""
    rss_gib = None
    peak_rss_gib = None
    try:
        import psutil

        rss_gib = psutil.Process().memory_info().rss / 2**30
    except ImportError:
        pass
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("VmHWM:"):
                peak_rss_gib = int(line.split()[1]) / 2**20
                break
    return {
        "rss_gib": rss_gib,
        "peak_rss_gib": peak_rss_gib,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="测量 TPQ 稳态单请求 decode token/s",
    )
    parser.add_argument("--model", required=True, help="CCCP 模型目录")
    parser.add_argument(
        "--profile",
        choices=("auto", "ram", "parallel"),
        default="auto",
    )
    parser.add_argument("--tp", type=int)
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES，例如 7 或 0,1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--prompt", default="请用中文简要介绍量化推理。")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--cache-gb", type=float)
    parser.add_argument("--vram-gb", type=float)
    parser.add_argument("--json", help="保存完整结果的 JSON 路径")
    return parser


def _device_steps(model: Any, logits: Any, steps: int, window: int):
    """GLM 贪心 decode：token 选择留在 GPU，按窗口回收结果。"""
    import torch

    output: list[int] = []
    for begin in range(0, steps, window):
        count = min(window, steps - begin)
        tokens = torch.empty(count, dtype=torch.long, device=logits.device)
        for index in range(count):
            torch.argmax(logits, out=tokens[index])
            logits = model.forward(tokens[index:index + 1])
        output.extend(tokens.cpu().tolist())
    return logits, output


def _host_steps(model: Any, logits: Any, steps: int):
    """DeepSeek decode：与当前生产生成循环一致，每步在主机取得 token id。"""
    output: list[int] = []
    for _ in range(steps):
        token = int(logits.argmax().item())
        output.append(token)
        logits = model.forward([token])
    return logits, output


def _steps(
    architecture: str,
    model: Any,
    logits: Any,
    count: int,
    window: int,
):
    if architecture == "glm":
        return _device_steps(model, logits, count, window)
    return _host_steps(model, logits, count)


def _apply_preset_environment(args: argparse.Namespace, preset: Any) -> None:
    if args.gpus:
        devices = [part.strip() for part in args.gpus.split(",") if part.strip()]
        if len(devices) < preset.tp:
            raise ValueError(
                f"--gpus 只给出 {len(devices)} 张卡，但 tp={preset.tp}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    for key, value in preset.environment.items():
        os.environ.setdefault(key, value)
    if preset.ep_layout is not None:
        os.environ.setdefault("TPQ_EP_LAYOUT", preset.ep_layout)
    # 固定输出缓冲可减少 GLM decode 中不必要的临时分配。
    os.environ.setdefault("TPQ_STATIC_LM_OUTPUT", "1")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.warmup < 1 or args.steps < 1 or args.repeat < 1:
        raise SystemExit("--warmup、--steps、--repeat 必须大于 0")
    if args.window < 1:
        raise SystemExit("--window 必须大于 0")

    preset = resolve_preset(args.model, profile=args.profile, tp=args.tp)
    _apply_preset_environment(args, preset)

    # CUDA_VISIBLE_DEVICES 必须在首次导入 torch/Engine 前设置。
    import torch

    from .engine import Engine

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")

    required = args.warmup + args.steps * args.repeat
    load_started = time.perf_counter()
    engine = Engine(
        str(preset.model_dir),
        cache_gb=args.cache_gb,
        max_ctx=args.max_ctx,
        device=args.device,
        vram_cache_gb=args.vram_gb,
        tp_size=preset.tp,
    )
    load_seconds = time.perf_counter() - load_started
    actual_device = torch.device(
        getattr(engine.model, "device", args.device)
    ).type
    if actual_device != args.device:
        raise SystemExit(
            f"请求 device={args.device}，实际回退到 {actual_device}；"
            "拒绝生成会误标硬件的基准结果"
        )
    effective_tp = int(getattr(engine.model, "effective_tp_size", 1))
    if preset.tp > 1 and effective_tp != preset.tp:
        raise SystemExit(
            f"请求 tp={preset.tp}，实际 effective_tp={effective_tp}；"
            "拒绝把 RAM 回退标成多卡性能"
        )
    from .chat_adapters import (
        ChatMessage,
        ChatOptions,
        adapter_for_arch,
    )

    options = ChatOptions(
        thinking_mode="chat",
        reasoning_effort=None,
        temperature=0.0,
        top_p=1.0,
        max_new=required,
    )
    prompt_plan = adapter_for_arch(preset.architecture).prepare(
        engine,
        [ChatMessage(role="user", content=args.prompt)],
        options,
        None,
    )
    prompt_ids = prompt_plan.input_ids
    if not prompt_ids:
        raise SystemExit("prompt 编码为空")
    if len(prompt_ids) + required + 1 > args.max_ctx:
        raise SystemExit(
            f"prompt({len(prompt_ids)}) + warmup/测量({required}) "
            f"超过 max_ctx={args.max_ctx}"
        )

    engine.reset()
    prefill_started = time.perf_counter()
    logits = engine.model.forward(prompt_ids)
    if args.device == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_started) * 1000.0

    logits, warmup_tokens = _steps(
        preset.architecture,
        engine.model,
        logits,
        args.warmup,
        args.window,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()

    runs: list[dict[str, Any]] = []
    all_tokens: list[int] = []
    for repeat_index in range(args.repeat):
        measured_position = int(getattr(engine.model, "pos", 0))
        cuda_events = None
        if args.device == "cuda":
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            cuda_events = (begin, end)
        wall_started = time.perf_counter()
        logits, tokens = _steps(
            preset.architecture,
            engine.model,
            logits,
            args.steps,
            args.window,
        )
        if cuda_events is not None:
            cuda_events[1].record()
            cuda_events[1].synchronize()
            cuda_ms = float(cuda_events[0].elapsed_time(cuda_events[1]))
        else:
            cuda_ms = None
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        all_tokens.extend(tokens)
        run = {
            "repeat": repeat_index + 1,
            "measured_position": measured_position,
            "steps": args.steps,
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "throughput_tok_s": args.steps / (wall_ms / 1000.0),
            "allocated_vram_gib": (
                torch.cuda.memory_allocated() / 2**30
                if args.device == "cuda"
                else None
            ),
            "tokens": tokens,
        }
        runs.append(run)
        print(
            f"repeat={repeat_index + 1} position={measured_position} "
            f"throughput={run['throughput_tok_s']:.3f} token/s",
            flush=True,
        )

    throughputs = [float(run["throughput_tok_s"]) for run in runs]
    hardware: dict[str, Any]
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        hardware = {
            "name": props.name,
            "compute_capability": (
                f"{props.major}.{props.minor}"
            ),
            "sm": f"sm_{props.major}{props.minor}",
            "vram_gib": props.total_memory / 2**30,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_count": torch.cuda.device_count(),
        }
    else:
        hardware = _cpu_hardware(torch)
    result = {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": _source_state(),
        "model": str(Path(args.model).resolve()),
        "architecture": preset.architecture,
        "profile": preset.profile,
        "tp": preset.tp,
        "effective_tp": effective_tp,
        "ep_layout": preset.ep_layout,
        "device": args.device,
        "hardware": hardware,
        "process_memory": _process_memory(),
        "environment": {
            key: os.environ[key]
            for key in (
                "TPQ_COMPUTE_DTYPE",
                "TPQ_DENSE_BF16",
                "TPQ_FUSED",
                "TPQ_PAGED_KV_FUSED",
                "TPQ_LATENT_KV",
                "TPQ_RAM_MIRROR",
                "TPQ_RESIDENT_RESERVE_GB",
                "TPQ_RAM_RESERVE_GB",
                "TPQ_VRAM_RESERVE_GB",
                "TPQ_VRAM_RUNTIME_GB",
                "TPQ_HOST_PIN_GB",
                "TPQ_EP_LAYOUT",
                "TPQ_CPU_THREADS",
                "TPQ_CPU_NUMA",
                "TPQ_CPU_FUSED",
                "TPQ_CPU_ATTN_MANY",
                "TPQ_CPU_QKV_POST",
                "TPQ_CPU_DN_BLOCK",
                "TPQ_CPU_VQ_INT8",
            )
            if key in os.environ
        },
        "load_seconds": load_seconds,
        "prompt": args.prompt,
        "prompt_mode": "production_chat_adapter",
        "prompt_tokens": len(prompt_ids),
        "prefill_ms": prefill_ms,
        "warmup_steps": args.warmup,
        "warmup_tokens": warmup_tokens,
        "steps_per_repeat": args.steps,
        "repeat": args.repeat,
        "throughput_tok_s_median": statistics.median(throughputs),
        "throughput_tok_s_min": min(throughputs),
        "throughput_tok_s_max": max(throughputs),
        "decoded_measured_text": engine.decode(all_tokens),
        "runs": runs,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
