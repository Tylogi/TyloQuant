"""Train model-, family-, or tensor-scoped NVQ2 E8 codebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from mfq.quantize.nvq_codebook import (
    NvqCodebookTrainingConfig,
    NvqTrainingMatrix,
    save_nvq_codebook_artifact,
    train_nvq2_codebook_set,
)


_LINEAR_SUFFIXES = (
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.out_proj.weight",
)


def _load_weight_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index}")
    return dict(json.loads(index.read_text(encoding="utf-8"))["weight_map"])


def _select_rows(out: int, count: int, seed: int, name: str) -> np.ndarray:
    if count <= 0 or count >= out:
        return np.arange(out, dtype=np.int64)
    name_seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed ^ name_seed)
    return np.sort(rng.choice(out, size=count, replace=False))


def _load_rows(
    root: Path,
    weight_map: dict[str, str],
    name: str,
    *,
    rows_per_tensor: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("model codebook training requires torch and safetensors") from exc

    shard = root / weight_map[name]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        sliced = handle.get_slice(name)
        shape = sliced.get_shape()
        if len(shape) != 2:
            raise ValueError(f"{name} is not a matrix: {shape}")
        rows = _select_rows(shape[0], rows_per_tensor, seed, name)
        if rows.size == shape[0]:
            tensor = handle.get_tensor(name)
        else:
            runs: list[tuple[int, int]] = []
            start = previous = int(rows[0])
            for raw in rows[1:]:
                value = int(raw)
                if value != previous + 1:
                    runs.append((start, previous + 1))
                    start = value
                previous = value
            runs.append((start, previous + 1))
            tensor = torch.cat([sliced[start:stop] for start, stop in runs], dim=0)
        tensor = tensor.to(torch.float32).contiguous()
    result = tensor.numpy().copy()
    del tensor
    return result, rows


def _selected_tensor_names(
    weight_map: dict[str, str],
    explicit: list[str] | None,
    includes: list[str] | None,
    excludes: list[str] | None,
) -> list[str]:
    if explicit:
        missing = [name for name in explicit if name not in weight_map]
        if missing:
            raise KeyError(f"tensors not found in model index: {missing}")
        names = list(dict.fromkeys(explicit))
    elif includes:
        patterns = [re.compile(pattern) for pattern in includes]
        names = [name for name in weight_map if any(pattern.search(name) for pattern in patterns)]
    else:
        names = [name for name in weight_map if name.endswith(_LINEAR_SUFFIXES)]
    if excludes:
        patterns = [re.compile(pattern) for pattern in excludes]
        names = [name for name in names if not any(pattern.search(name) for pattern in patterns)]
    if not names:
        raise ValueError("tensor selection is empty")
    return sorted(names)


def _importance_map(values: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--importance expects TENSOR=PATH.npy")
        name, path = value.split("=", 1)
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("model", "family", "tensor"), default="model")
    parser.add_argument("--sign-mode", choices=("even", "index_parity"), default="even")
    parser.add_argument("--tensor", action="append", dest="tensors")
    parser.add_argument("--include-regex", action="append")
    parser.add_argument("--exclude-regex", action="append")
    parser.add_argument("--importance", action="append", help="TENSOR=PATH.npy")
    parser.add_argument("--rows-per-tensor", type=int, default=64, help="0 means all rows")
    parser.add_argument("--max-tensors", type=int, default=0, help="deterministic debug cap")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--search-steps", type=int, default=19)
    parser.add_argument("--scale-refine-steps", type=int, default=3)
    parser.add_argument("--group-chunk", type=int, default=512)
    parser.add_argument("--projection-candidates", type=int, default=48)
    parser.add_argument("--reseed-pool-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite codebook artifact: {output}")
    weight_map = _load_weight_map(root)
    names = _selected_tensor_names(
        weight_map,
        args.tensors,
        args.include_regex,
        args.exclude_regex,
    )
    if args.max_tensors > 0:
        names = names[: args.max_tensors]
    importance_paths = _importance_map(args.importance)
    unknown_importance = sorted(set(importance_paths) - set(names))
    if unknown_importance:
        raise ValueError(f"importance supplied for unselected tensors: {unknown_importance}")

    config = NvqCodebookTrainingConfig(
        scope=args.scope,
        sign_mode=args.sign_mode,
        iterations=args.iterations,
        search_steps=args.search_steps,
        scale_refine_steps=args.scale_refine_steps,
        group_chunk=args.group_chunk,
        projection_candidates=args.projection_candidates,
        reseed_pool_size=args.reseed_pool_size,
        seed=args.seed,
    )
    contract = {
        "path": "offline NVQ2 codebook training",
        "runtime_speed_claim": False,
        "source_model": str(root),
        "output": str(output),
        "tensor_count": len(names),
        "tensors": names,
        "rows_per_tensor": args.rows_per_tensor,
        "importance": {name: str(path.resolve()) for name, path in importance_paths.items()},
        "config": config.__dict__,
    }
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)

    matrices: list[NvqTrainingMatrix] = []
    source_rows: dict[str, list[int]] = {}
    total_bytes = 0
    for index, name in enumerate(names, 1):
        weight, rows = _load_rows(
            root,
            weight_map,
            name,
            rows_per_tensor=args.rows_per_tensor,
            seed=args.seed,
        )
        importance = None
        if name in importance_paths:
            importance = np.load(importance_paths[name]).astype(np.float32)
        matrices.append(NvqTrainingMatrix(name, weight, importance))
        source_rows[name] = [int(value) for value in rows]
        total_bytes += weight.nbytes + (importance.nbytes if importance is not None else 0)
        print(
            f"loaded {index}/{len(names)} {name} shape={weight.shape} "
            f"sample_rss_payload={total_bytes / (1 << 20):.1f} MiB",
            flush=True,
        )

    def progress(event: dict[str, object]) -> None:
        print(
            f"[{event['scope_key']}] iteration={event['iteration']} "
            f"SNR={float(event['snr_db']):.6f} dB "
            f"NMSE={float(event['nmse_percent']):.6f}% "
            f"used_codes={event['used_codes']}",
            flush=True,
        )

    artifact = train_nvq2_codebook_set(
        matrices,
        config,
        source_model=str(root),
        source_rows=source_rows,
        progress=progress,
    )
    save_nvq_codebook_artifact(artifact, output, overwrite=args.overwrite)
    print(f"artifact={output}", flush=True)


if __name__ == "__main__":
    main()
