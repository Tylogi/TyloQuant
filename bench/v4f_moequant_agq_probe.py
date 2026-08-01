#!/usr/bin/env python3
"""A/B-test MoEQuant AGQ on real DeepSeek-V4-Flash experts.

The probe keeps calibration tokens, routed experts, source weights, Q4 format,
and model volume fixed.  It changes only the diagonal calibration objective:

* count:    sum(x**2), the ordinary routed-token imatrix objective;
* affinity: sum(c*x**2), MoEQuant's affinity-guided objective.

The official checkpoint already stores routed experts in FP4.  This experiment
therefore tests Q4 re-quantization into TyloQuant NINT4, not BF16-to-Q4 quality.
Full-model KLD requires all checkpoint shards and is deliberately out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from mfq._vendor.tpq.cconfig import DSV4Config
from mfq._vendor.tpq.dsv4 import (
    DSV4Checkpoint,
    RopeCache,
    attn_prefill,
    block_forward,
    gate_route,
    hc_post,
    hc_pre,
    rmsnorm,
)
from mfq.formats.nint import NintSpec
from mfq.quantize.moequant import (
    ExpertAffinityAccumulator,
    diagonal_second_moment,
)
from mfq.quantize.nint_quant import dequantize
from mfq.quantize.nint_quant_torch import quantize_axis0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--token-ids",
        type=Path,
        required=True,
        help="Little-endian int32 token IDs from the model tokenizer.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-sequences", type=int, default=4)
    parser.add_argument("--test-sequences", type=int, default=4)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--min-routes", type=int, default=4)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=24)
    parser.add_argument("--sub-bits", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fused-imatrix-kernels",
        action="store_true",
        help="Use TyloQuant's compiled CUDA imatrix search kernels.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _snr(signal: float, error: float) -> float:
    return 10.0 * math.log10(signal / max(error, np.finfo(np.float64).tiny))


def _load_tokens(options: argparse.Namespace, vocab: int) -> torch.Tensor:
    sequences = options.train_sequences + options.test_sequences
    needed = sequences * options.sequence_length
    values = np.fromfile(options.token_ids, dtype="<i4", count=needed)
    if values.size != needed:
        raise ValueError(f"token file has {values.size} IDs, need {needed}")
    if int(values.min()) < 0 or int(values.max()) >= vocab:
        raise ValueError("token file contains IDs outside the V4F vocabulary")
    return torch.from_numpy(values.astype(np.int64).reshape(sequences, -1))


def _layer_state(
    config: DSV4Config,
    *,
    batch: int,
    sequence_length: int,
    ratio: int,
    device: str,
) -> dict[str, torch.Tensor]:
    compressed = sequence_length // ratio + 1 if ratio else 0
    state = {
        "kv": torch.zeros(
            batch,
            config.sliding_window + compressed,
            config.head_dim,
            dtype=torch.float32,
            device=device,
        ),
        "win_pos": torch.full(
            (batch, config.sliding_window),
            -1,
            dtype=torch.int64,
            device=device,
        ),
    }
    if ratio:
        compressor_width = 2 if ratio == 4 else 1
        state["ckv"] = torch.zeros(
            batch,
            compressor_width * ratio,
            compressor_width * config.head_dim,
            dtype=torch.float32,
            device=device,
        )
        state["cscore"] = torch.full(
            (
                batch,
                compressor_width * ratio,
                compressor_width * config.head_dim,
            ),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
    return state


@torch.inference_mode()
def _target_ffn_inputs(
    checkpoint: DSV4Checkpoint,
    config: DSV4Config,
    tokens: torch.Tensor,
    target_layer: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stream the real prefix and stop immediately before one layer's experts."""

    ids = tokens.to(device=device, dtype=torch.int64)
    embedding = checkpoint.embed()
    hidden = embedding[ids].unsqueeze(2).repeat(1, 1, config.hc_mult, 1)
    del embedding
    sequence_length = int(tokens.shape[1])
    base_cache = RopeCache(
        config.qk_rope_head_dim,
        sequence_length,
        config.rope_theta,
        None,
    )
    compressed_cache = RopeCache(
        config.qk_rope_head_dim,
        sequence_length,
        config.compress_rope_theta,
        config.rope_scaling or None,
    )
    for cache in (base_cache, compressed_cache):
        cache.cos = cache.cos.to(device)
        cache.sin = cache.sin.to(device)
    ratios = (list(config.compress_ratios) + [0] * config.n_layers)[: config.n_layers]
    batch = int(tokens.shape[0])

    for layer in range(target_layer):
        layer_started = time.perf_counter()
        ratio = ratios[layer]
        state = _layer_state(
            config,
            batch=batch,
            sequence_length=sequence_length,
            ratio=ratio,
            device=device,
        )
        hidden = block_forward(
            hidden,
            checkpoint.layer(layer),
            state,
            config,
            compressed_cache if ratio else base_cache,
            ratio,
            ids,
            0,
            checkpoint.expert,
            layer,
        )
        print(
            json.dumps(
                {
                    "prefix_layer": layer,
                    "ratio": ratio,
                    "seconds": time.perf_counter() - layer_started,
                }
            ),
            flush=True,
        )
        del state
        torch.cuda.empty_cache()

    weights = checkpoint.layer(target_layer)
    ratio = ratios[target_layer]
    state = _layer_state(
        config,
        batch=batch,
        sequence_length=sequence_length,
        ratio=ratio,
        device=device,
    )

    residual = hidden
    value, post, combine = hc_pre(
        hidden,
        weights["hc_attn_fn"],
        weights["hc_attn_scale"],
        weights["hc_attn_base"],
        config,
    )
    value = rmsnorm(value, weights["attn_norm"], config.rms_eps)
    attention = attn_prefill(
        value,
        weights,
        state,
        config,
        compressed_cache if ratio else base_cache,
        ratio=ratio,
    )
    hidden = hc_post(attention, residual, post, combine)
    value, _post, _combine = hc_pre(
        hidden,
        weights["hc_ffn_fn"],
        weights["hc_ffn_scale"],
        weights["hc_ffn_base"],
        config,
    )
    value = rmsnorm(value, weights["ffn_norm"], config.rms_eps).float()
    flat = value.reshape(-1, config.hidden)
    affinity, expert_ids = gate_route(flat, weights, config, ids.reshape(-1))
    return flat.cpu(), affinity.cpu(), expert_ids.cpu()


def _selected_routes(
    inputs: torch.Tensor,
    affinities: torch.Tensor,
    expert_ids: torch.Tensor,
    expert: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_rows, slots = (expert_ids == expert).nonzero(as_tuple=True)
    return (
        inputs.index_select(0, token_rows).to(device=device, dtype=torch.float32),
        affinities[token_rows, slots].to(device=device, dtype=torch.float32),
    )


def _intermediate(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    gate_value = functional.linear(inputs, gate)
    up_value = functional.linear(inputs, up)
    if limit > 0:
        gate_value = gate_value.clamp(max=limit)
        up_value = up_value.clamp(min=-limit, max=limit)
    return functional.silu(gate_value) * up_value


def _expert_output(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    return functional.linear(_intermediate(inputs, gate, up, limit), down)


def _quantized_weight(
    weight: torch.Tensor,
    metric: np.ndarray,
    spec: NintSpec,
    device: str,
    fused: bool,
) -> torch.Tensor:
    quantized = quantize_axis0(
        weight,
        spec,
        device=device,
        importance=metric,
        use_cuda_imatrix_kernels=fused,
    )
    reconstruction = dequantize(quantized)
    return torch.as_tensor(reconstruction, device=device, dtype=torch.float32)


def _candidate_errors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    affinity: torch.Tensor,
) -> dict[str, np.ndarray]:
    error = (reference - candidate).square().sum(dim=1).double()
    signal = reference.square().sum(dim=1).double()
    c = affinity.double()
    return {
        "unweighted_error": error.cpu().numpy(),
        "unweighted_signal": signal.cpu().numpy(),
        "affinity_error": (c * error).cpu().numpy(),
        "affinity_signal": (c * signal).cpu().numpy(),
        "routed_error": (c.square() * error).cpu().numpy(),
        "routed_signal": (c.square() * signal).cpu().numpy(),
    }


def _metric_report(
    count: dict[str, np.ndarray],
    agq: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, float]:
    count_error = float(count[f"{prefix}_error"].sum(dtype=np.float64))
    agq_error = float(agq[f"{prefix}_error"].sum(dtype=np.float64))
    signal = float(count[f"{prefix}_signal"].sum(dtype=np.float64))
    return {
        "signal": signal,
        "count_error": count_error,
        "agq_error": agq_error,
        "count_snr_db": _snr(signal, count_error),
        "agq_snr_db": _snr(signal, agq_error),
        "agq_delta_snr_db": _snr(signal, agq_error) - _snr(signal, count_error),
        "agq_error_reduction_percent": 100.0 * (count_error - agq_error) / count_error,
    }


def _bootstrap(
    count_error: np.ndarray,
    agq_error: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float] | None:
    if iterations <= 0 or count_error.size < 2:
        return None
    rng = np.random.default_rng(seed)
    reductions = np.empty(iterations, dtype=np.float64)
    size = int(count_error.size)
    for iteration in range(iterations):
        selected = rng.integers(0, size, size=size)
        baseline = float(count_error[selected].sum(dtype=np.float64))
        candidate = float(agq_error[selected].sum(dtype=np.float64))
        reductions[iteration] = 100.0 * (baseline - candidate) / baseline
    low, high = np.quantile(reductions, [0.025, 0.975])
    return {
        "iterations": iterations,
        "unit": "heldout routed expert sample",
        "mean_error_reduction_percent": float(reductions.mean()),
        "ci95_low_percent": float(low),
        "ci95_high_percent": float(high),
        "probability_agq_better": float(np.mean(reductions > 0)),
    }


def _concat(rows: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate([row[key] for row in rows])


def main() -> int:
    options = parse_args()
    if options.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the V4F MoEQuant probe requires CUDA")
    for label in ("sequence_length", "train_sequences", "test_sequences", "experts"):
        if int(getattr(options, label)) <= 0:
            raise ValueError(f"{label} must be positive")
    if options.output.exists():
        raise FileExistsError(options.output)

    started = time.perf_counter()
    torch.set_default_device(options.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = DSV4Config.from_hf(str(options.model))
    if options.layer < 0 or options.layer >= config.n_layers:
        raise ValueError(f"layer must be in [0,{config.n_layers - 1}]")
    checkpoint = DSV4Checkpoint(str(options.model), device=options.device, cache_layers=1)
    tokens = _load_tokens(options, config.vocab)
    inputs, affinities, expert_ids = _target_ffn_inputs(
        checkpoint, config, tokens, options.layer, options.device
    )
    split = options.train_sequences * options.sequence_length
    train_inputs, test_inputs = inputs[:split], inputs[split:]
    train_affinity, test_affinity = affinities[:split], affinities[split:]
    train_ids, test_ids = expert_ids[:split], expert_ids[split:]

    gate_accumulator = ExpertAffinityAccumulator.create(config.n_experts, config.hidden)
    for slot in range(config.top_k):
        gate_accumulator.update(
            train_inputs.numpy(),
            train_ids[:, slot].numpy(),
            train_affinity[:, slot].numpy(),
        )
    train_counts = torch.bincount(train_ids.reshape(-1), minlength=config.n_experts)
    test_counts = torch.bincount(test_ids.reshape(-1), minlength=config.n_experts)
    eligible = [
        expert
        for expert in range(config.n_experts)
        if min(int(train_counts[expert]), int(test_counts[expert])) >= options.min_routes
    ]
    eligible.sort(
        key=lambda expert: (
            min(int(train_counts[expert]), int(test_counts[expert])),
            int(train_counts[expert]) + int(test_counts[expert]),
        ),
        reverse=True,
    )
    selected_experts = eligible[: options.experts]
    if len(selected_experts) < options.experts:
        raise RuntimeError(
            f"only {len(selected_experts)} experts have at least {options.min_routes} "
            "train and heldout routes"
        )

    artifact_dir = options.artifact_dir or options.output.parent / (options.output.stem + "-stats")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gate_artifact = artifact_dir / f"layer{options.layer}-gate-up-agq.npz"
    gate_accumulator.save(
        gate_artifact,
        metadata={
            "layer": options.layer,
            "projection": "gate_up",
            "train_tokens": int(train_inputs.shape[0]),
            "top_k": config.top_k,
            "source": str(options.model),
        },
    )

    spec = NintSpec(
        bits=options.bits,
        groupsize=options.group_size,
        sub_bits=options.sub_bits,
    )
    per_expert: list[dict[str, Any]] = []
    count_errors: list[dict[str, np.ndarray]] = []
    agq_errors: list[dict[str, np.ndarray]] = []
    down_accumulator = ExpertAffinityAccumulator.create(config.n_experts, config.moe_inter)

    for expert in selected_experts:
        expert_started = time.perf_counter()
        train_x, train_c = _selected_routes(
            train_inputs, train_affinity, train_ids, expert, options.device
        )
        test_x, test_c = _selected_routes(
            test_inputs, test_affinity, test_ids, expert, options.device
        )
        gate_weight, up_weight, down_weight = checkpoint.expert(options.layer, expert)
        train_hidden = _intermediate(
            train_x, gate_weight, up_weight, config.swiglu_limit
        )
        down_accumulator.update(
            train_hidden.cpu().numpy(),
            np.full(int(train_hidden.shape[0]), expert, dtype=np.int64),
            train_c.cpu().numpy(),
        )
        count_input_metric = diagonal_second_moment(train_x.cpu().numpy())
        agq_input_metric = diagonal_second_moment(
            train_x.cpu().numpy(), train_c.cpu().numpy()
        )
        count_down_metric = diagonal_second_moment(train_hidden.cpu().numpy())
        agq_down_metric = diagonal_second_moment(
            train_hidden.cpu().numpy(), train_c.cpu().numpy()
        )
        reference = _expert_output(
            test_x, gate_weight, up_weight, down_weight, config.swiglu_limit
        )

        count_gate = _quantized_weight(
            gate_weight,
            count_input_metric,
            spec,
            options.device,
            options.fused_imatrix_kernels,
        )
        count_up = _quantized_weight(
            up_weight,
            count_input_metric,
            spec,
            options.device,
            options.fused_imatrix_kernels,
        )
        count_down = _quantized_weight(
            down_weight,
            count_down_metric,
            spec,
            options.device,
            options.fused_imatrix_kernels,
        )
        count_output = _expert_output(
            test_x, count_gate, count_up, count_down, config.swiglu_limit
        )
        count_row = _candidate_errors(reference, count_output, test_c)
        del count_gate, count_up, count_down, count_output

        agq_gate = _quantized_weight(
            gate_weight,
            agq_input_metric,
            spec,
            options.device,
            options.fused_imatrix_kernels,
        )
        agq_up = _quantized_weight(
            up_weight,
            agq_input_metric,
            spec,
            options.device,
            options.fused_imatrix_kernels,
        )
        agq_down = _quantized_weight(
            down_weight,
            agq_down_metric,
            spec,
            options.device,
            options.fused_imatrix_kernels,
        )
        agq_output = _expert_output(
            test_x, agq_gate, agq_up, agq_down, config.swiglu_limit
        )
        agq_row = _candidate_errors(reference, agq_output, test_c)
        count_errors.append(count_row)
        agq_errors.append(agq_row)
        result = {
            "expert": expert,
            "train_routes": int(train_x.shape[0]),
            "heldout_routes": int(test_x.shape[0]),
            "mean_train_affinity": float(train_c.mean()),
            "mean_heldout_affinity": float(test_c.mean()),
            "unweighted": _metric_report(count_row, agq_row, "unweighted"),
            "paper_affinity_weighted": _metric_report(count_row, agq_row, "affinity"),
            "routed_contribution": _metric_report(count_row, agq_row, "routed"),
            "seconds": time.perf_counter() - expert_started,
        }
        per_expert.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del (
            train_x,
            train_c,
            test_x,
            test_c,
            train_hidden,
            gate_weight,
            up_weight,
            down_weight,
            reference,
            agq_gate,
            agq_up,
            agq_down,
            agq_output,
        )
        torch.cuda.empty_cache()

    down_artifact = artifact_dir / f"layer{options.layer}-down-agq.npz"
    down_accumulator.save(
        down_artifact,
        metadata={
            "layer": options.layer,
            "projection": "down",
            "selected_experts": selected_experts,
            "source": str(options.model),
        },
    )
    aggregate: dict[str, Any] = {}
    for name in ("unweighted", "affinity", "routed"):
        aggregate[name] = _metric_report(
            {key: _concat(count_errors, key) for key in count_errors[0]},
            {key: _concat(agq_errors, key) for key in agq_errors[0]},
            name,
        )
    aggregate["routed_bootstrap"] = _bootstrap(
        _concat(count_errors, "routed_error"),
        _concat(agq_errors, "routed_error"),
        options.bootstrap,
        options.seed,
    )
    report = {
        "format": "mfq.v4f-moequant-agq-probe.v1",
        "status": "completed",
        "scope": (
            f"real layer-{options.layer} weights and streamed-prefix activations; "
            "selected routed experts"
        ),
        "source_precision": "official FP4 checkpoint",
        "target_quantization": {
            "format": spec.profile_label,
            "bits": spec.bits,
            "group_size": spec.groupsize,
            "sub_bits": spec.sub_bits,
            "gate_up_bpw": spec.bpw(config.hidden),
            "down_bpw": spec.bpw(config.moe_inter),
            "same_volume_between_arms": True,
        },
        "arms": {
            "count": "sum(x^2) per routed expert",
            "agq": "sum(c*x^2) per routed expert (MoEQuant AGQ)",
        },
        "model": str(options.model.resolve()),
        "model_config_sha256": _sha256(options.model / "config.json"),
        "token_ids": str(options.token_ids.resolve()),
        "token_ids_sha256": _sha256(options.token_ids),
        "layer": options.layer,
        "prefix_layers": options.layer,
        "hash_routed_layer": options.layer < config.n_hash_layers,
        "sequence_length": options.sequence_length,
        "train_sequences": options.train_sequences,
        "test_sequences": options.test_sequences,
        "train_tokens": int(train_inputs.shape[0]),
        "heldout_tokens": int(test_inputs.shape[0]),
        "selected_experts": selected_experts,
        "artifacts": {
            "gate_up": str(gate_artifact.resolve()),
            "down": str(down_artifact.resolve()),
        },
        "aggregate": aggregate,
        "experts": per_expert,
        "environment": {
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "fused_imatrix_kernels": options.fused_imatrix_kernels,
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "This is an expert-output distortion probe, not full-model KLD.",
            (
                "The target layer uses token-ID hash routing."
                if options.layer < config.n_hash_layers
                else "The target layer uses score-based dynamic routing."
            ),
            "The source routed weights are official FP4, so this measures Q4 re-quantization.",
            "Bootstrap units are routed expert samples and are exploratory, not independent tokens.",
        ],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
