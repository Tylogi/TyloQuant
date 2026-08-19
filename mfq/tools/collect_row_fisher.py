"""Collect output-row diagonal Fisher statistics for Qwen3.5 linear tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mfq.quantize.row_importance import save_row_importance
from mfq.tools.quantize_gguf_to_mfq import _build_plan, _load_gguf


@dataclass(frozen=True)
class Target:
    tensor_name: str
    module_name: str
    rows: int
    transform: str = "identity"


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "size": stat.st_size, "sha256": digest.hexdigest()}


def _module_binding(name: str, rows: int) -> Target:
    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or parts[3] != "weight":
        raise ValueError(f"unsupported Qwen3.5 row-Fisher tensor name: {name}")
    layer = int(parts[1])
    stem = parts[2]
    prefix = f"model.layers.{layer}"
    mlp = {
        "ffn_down": "down_proj",
        "ffn_gate": "gate_proj",
        "ffn_up": "up_proj",
    }
    if stem in mlp:
        return Target(name, f"{prefix}.mlp.{mlp[stem]}", rows)
    attention = {
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_output": "o_proj",
    }
    if stem in attention:
        return Target(name, f"{prefix}.self_attn.{attention[stem]}", rows)
    if stem == "attn_qkv":
        return Target(name, f"{prefix}.linear_attn.in_proj_qkv", rows, "linear_qkv")
    if stem == "attn_gate":
        return Target(name, f"{prefix}.linear_attn.in_proj_z", rows, "linear_v_rows")
    if stem == "ssm_out":
        return Target(name, f"{prefix}.linear_attn.out_proj", rows)
    raise ValueError(f"unsupported Qwen3.5 row-Fisher tensor: {name}")


def _load_targets(source_gguf: Path, recipe_gguf: Path) -> list[Target]:
    GGUFReader, _dequantize = _load_gguf()
    source = GGUFReader(str(source_gguf), "r")
    recipe = GGUFReader(str(recipe_gguf), "r")
    plan = _build_plan(source, recipe)
    targets = [
        _module_binding(item.name, int(item.storage_shape[0]))
        for item in plan
        if item.target_dtype in {
            "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
            "NVQ3J", "NVQ3J-512", "NVQ3J-L",
        }
    ]
    if len(targets) != len({item.tensor_name for item in targets}):
        raise ValueError("duplicate NVQ-JSC tensor target")
    if len(targets) != len({item.module_name for item in targets}):
        raise ValueError("multiple NVQ-JSC tensors map to one HF module")
    return targets


def _reorder_v_rows(value: np.ndarray, num_k_heads: int, num_v_heads: int, head_dim: int) -> np.ndarray:
    if num_v_heads % num_k_heads:
        raise ValueError("value heads must be divisible by key heads")
    ratio = num_v_heads // num_k_heads
    expected = num_v_heads * head_dim
    if value.size != expected:
        raise ValueError(f"V-row vector has {value.size} entries, expected {expected}")
    return np.ascontiguousarray(
        value.reshape(num_k_heads, ratio, head_dim).transpose(1, 0, 2).reshape(-1),
        dtype=np.float32,
    )


def _transform_rows(value: np.ndarray, target: Target, config: Any) -> np.ndarray:
    if target.transform == "identity":
        return np.ascontiguousarray(value, dtype=np.float32)
    num_k_heads = int(config.linear_num_key_heads)
    num_v_heads = int(config.linear_num_value_heads)
    head_dim = int(config.linear_value_head_dim)
    if target.transform == "linear_v_rows":
        return _reorder_v_rows(value, num_k_heads, num_v_heads, head_dim)
    if target.transform == "linear_qkv":
        key_rows = int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
        value_rows = num_v_heads * head_dim
        if value.size != key_rows * 2 + value_rows:
            raise ValueError(
                f"linear QKV row vector has {value.size} entries, "
                f"expected {key_rows * 2 + value_rows}"
            )
        return np.concatenate(
            [
                value[: 2 * key_rows],
                _reorder_v_rows(value[2 * key_rows :], num_k_heads, num_v_heads, head_dim),
            ]
        ).astype(np.float32, copy=False)
    raise ValueError(f"unsupported row transform: {target.transform}")


def _sample_sequences(
    parquet_path: Path,
    tokenizer: Any,
    samples: int,
    sequence_length: int,
    seed: int,
) -> list[list[int]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("collect_row_fisher requires pyarrow") from exc
    table = pq.read_table(parquet_path, columns=["text"])
    texts = [str(item) for item in table.column("text").to_pylist() if str(item).strip()]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    required = samples * sequence_length
    tokens: list[int] = []
    eos = tokenizer.eos_token_id
    for index in order:
        tokens.extend(tokenizer.encode(texts[int(index)], add_special_tokens=False))
        if eos is not None:
            tokens.append(int(eos))
        if len(tokens) >= required:
            break
    if len(tokens) < required:
        raise ValueError(f"calibration corpus produced {len(tokens)} tokens, need {required}")
    return [tokens[i * sequence_length : (i + 1) * sequence_length] for i in range(samples)]


class _Collector:
    def __init__(self, targets: list[Target]) -> None:
        self.targets = {item.module_name: item for item in targets}
        self.sums = [
            {item.tensor_name: torch.zeros(item.rows, device="cuda", dtype=torch.float64) for item in targets},
            {item.tensor_name: torch.zeros(item.rows, device="cuda", dtype=torch.float64) for item in targets},
        ]
        self.current_split = 0
        self.handles: list[Any] = []

    def install(self, model: torch.nn.Module) -> None:
        modules = dict(model.named_modules())
        missing = sorted(set(self.targets) - set(modules))
        if missing:
            raise ValueError(f"HF model is missing row-Fisher modules: {missing[:8]}")
        for module_name, target in self.targets.items():
            module = modules[module_name]
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"row-Fisher target is not Linear: {module_name}")
            if module.out_features != target.rows:
                raise ValueError(
                    f"row mismatch for {target.tensor_name}: module={module.out_features}, plan={target.rows}"
                )

            def forward_hook(_module, _inputs, output, *, _target=target):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"non-tensor output from {_target.module_name}")
                output.register_hook(lambda grad, _name=_target.tensor_name: self._add(_name, grad))

            self.handles.append(module.register_forward_hook(forward_hook))

    def _add(self, name: str, gradient: torch.Tensor) -> None:
        reduce = tuple(range(gradient.ndim - 1))
        value = gradient.detach().to(torch.float32).square().sum(dim=reduce, dtype=torch.float64)
        self.sums[self.current_split][name].add_(value)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    positive = value[value > 0]
    if not positive.size:
        raise ValueError("row Fisher entry is entirely zero")
    floor = max(float(np.mean(positive)) * 1e-8, np.finfo(np.float32).tiny)
    value = np.maximum(value, floor)
    value /= float(value.mean())
    return np.ascontiguousarray(value, dtype=np.float32)


def _log_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.log(np.maximum(left, np.finfo(np.float32).tiny))
    y = np.log(np.maximum(right, np.finfo(np.float32).tiny))
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 1.0
    return float(np.corrcoef(x, y)[0, 1])


def collect(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("collect_row_fisher requires transformers") from exc

    model_path = Path(args.model).resolve()
    source_gguf = Path(args.source_gguf).resolve()
    recipe_gguf = Path(args.recipe_gguf).resolve()
    calibration = Path(args.calibration).resolve()
    output = Path(args.output).resolve()
    targets = _load_targets(source_gguf, recipe_gguf)
    print(json.dumps({"event": "targets", "count": len(targets)}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    sequences = _sample_sequences(
        calibration, tokenizer, args.samples, args.sequence_length, args.seed
    )
    print(
        json.dumps(
            {
                "event": "calibration",
                "samples": len(sequences),
                "sequence_length": args.sequence_length,
                "predicted_tokens": len(sequences) * (args.sequence_length - 1),
            }
        ),
        flush=True,
    )
    load_started = time.time()
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        attn_implementation=args.attention,
        output_loading_info=True,
    )
    missing = [name for name in loading_info["missing_keys"] if not name.startswith("mtp.")]
    unexpected = [
        name
        for name in loading_info["unexpected_keys"]
        if not name.startswith(("mtp.", "model.visual."))
    ]
    if missing or unexpected or loading_info["mismatched_keys"]:
        raise RuntimeError(
            f"checkpoint load mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"mismatched={loading_info['mismatched_keys'][:8]}"
        )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    embedding_handle = model.model.embed_tokens.register_forward_hook(
        lambda _module, _inputs, value: value.detach().requires_grad_(True)
    )
    collector = _Collector(targets)
    collector.install(model)
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "seconds": round(time.time() - load_started, 3),
                "cuda_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
            }
        ),
        flush=True,
    )
    started = time.time()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 1)
    try:
        for index, token_ids in enumerate(sequences):
            collector.current_split = index & 1
            input_ids = torch.tensor([token_ids], device="cuda", dtype=torch.long)
            logits = model(input_ids=input_ids, use_cache=False).logits
            losses = F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                input_ids[:, 1:].reshape(-1),
                reduction="none",
            )
            signs = torch.empty(losses.shape, device="cuda", dtype=torch.float32)
            signs.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
            probe = (losses * signs).sum() / math.sqrt(float(losses.numel()))
            probe.backward()
            del input_ids, logits, losses, signs, probe
            if (index + 1) % args.progress_every == 0 or index + 1 == len(sequences):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "done": index + 1,
                            "total": len(sequences),
                            "seconds": round(time.time() - started, 3),
                            "cuda_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                            "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                        }
                    ),
                    flush=True,
                )
    finally:
        collector.close()
        embedding_handle.remove()

    entries: dict[str, np.ndarray] = {}
    correlations: dict[str, float] = {}
    target_by_name = {item.tensor_name: item for item in targets}
    for name in sorted(target_by_name):
        halves = [
            _normalize(collector.sums[split][name].cpu().numpy()) for split in range(2)
        ]
        combined = _normalize(
            collector.sums[0][name].cpu().numpy() + collector.sums[1][name].cpu().numpy()
        )
        target = target_by_name[name]
        entries[name] = _transform_rows(combined, target, model.config)
        correlations[name] = _log_correlation(halves[0], halves[1])
    correlation_values = np.asarray(list(correlations.values()), dtype=np.float64)
    metadata = {
        "objective": "rademacher_diagonal_fisher_of_next_token_cross_entropy",
        "normalization": "mean_one_per_tensor",
        "model": str(model_path),
        "source_gguf": str(source_gguf),
        "recipe_gguf": str(recipe_gguf),
        "calibration": _file_identity(calibration),
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "predicted_tokens": args.samples * (args.sequence_length - 1),
        "seed": args.seed,
        "attention": args.attention,
        "target_count": len(targets),
        "split_log_correlation": {
            "min": float(correlation_values.min()),
            "median": float(np.median(correlation_values)),
            "mean": float(correlation_values.mean()),
            "p10": float(np.quantile(correlation_values, 0.1)),
        },
        "per_tensor_split_log_correlation": correlations,
    }
    save_row_importance(output, entries, metadata)
    print(
        json.dumps(
            {
                "event": "done",
                "output": str(output),
                "bytes": output.stat().st_size,
                "seconds": round(time.time() - started, 3),
                "split_log_correlation": metadata["split_log_correlation"],
            }
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-gguf", required=True)
    parser.add_argument("--recipe-gguf", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 2 or args.sequence_length < 2 or args.progress_every <= 0:
        raise ValueError("samples must be >= 2, sequence length >= 2, and progress interval > 0")
    collect(args)


if __name__ == "__main__":
    main()
