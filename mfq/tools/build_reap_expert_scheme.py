"""Build an Expert-Wise NINT scheme from REAP observations and BF16 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open

from mfq.calibration.artifact import save_scheme
from mfq.calibration.nint_profiles import NINT_EXPERT_PROFILES
from mfq.calibration.reap_expertwise import (
    ExpertProfileEvaluation,
    allocate_expert_profiles,
    evaluation_from_document,
    load_reap_expert_table,
)
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant_torch import make_qkx2_torch


@dataclass(frozen=True)
class ExpertModelLayout:
    model_type: str
    layers: int
    experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    gate_template: str
    down_template: str


class _Bfloat16MemmapTensor:
    def __init__(
        self,
        path: Path,
        *,
        byte_offset: int,
        shape: tuple[int, ...],
    ) -> None:
        self.shape = shape
        self._array = np.memmap(
            path,
            mode="c",
            dtype="<u2",
            offset=byte_offset,
            shape=shape,
        )

    def __getitem__(self, index) -> torch.Tensor:
        values = np.array(self._array[index], copy=True)
        return torch.from_numpy(values).view(torch.bfloat16)

    def close(self) -> None:
        mapping = getattr(self._array, "_mmap", None)
        if mapping is not None:
            mapping.close()


@contextmanager
def open_bfloat16_memmap_tensor(
    path: str | Path,
    tensor_name: str,
) -> Iterator[_Bfloat16MemmapTensor]:
    source = Path(path)
    with source.open("rb") as stream:
        header_bytes = stream.read(8)
        if len(header_bytes) != 8:
            raise ValueError(f"invalid safetensors header: {source}")
        header_length = struct.unpack("<Q", header_bytes)[0]
        header = json.loads(stream.read(header_length))
    metadata = header.get(tensor_name)
    if not isinstance(metadata, dict):
        raise KeyError(f"{source} has no tensor {tensor_name}")
    if metadata.get("dtype") != "BF16":
        raise ValueError(f"tensor {tensor_name} is not BF16")
    shape = tuple(int(value) for value in metadata["shape"])
    start, end = (int(value) for value in metadata["data_offsets"])
    if end - start != int(np.prod(shape, dtype=np.int64)) * 2:
        raise ValueError(f"tensor {tensor_name} has inconsistent payload size")
    view = _Bfloat16MemmapTensor(
        source,
        byte_offset=8 + header_length + start,
        shape=shape,
    )
    try:
        yield view
    finally:
        view.close()


def resolve_profiles(names: tuple[str, ...] | list[str]) -> dict[str, NintSpec]:
    if not names:
        return dict(NINT_EXPERT_PROFILES)
    if len(set(names)) != len(names):
        raise ValueError("duplicate profile names are forbidden")
    unknown = [name for name in names if name not in NINT_EXPERT_PROFILES]
    if unknown:
        raise ValueError(f"unknown expert profiles: {', '.join(unknown)}")
    return {name: NINT_EXPERT_PROFILES[name] for name in names}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _quantization_sse_by_row(
    weight: torch.Tensor,
    spec: NintSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact quantizer reconstruction SSE and signal for every row."""

    if weight.dim() != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point [rows, columns] tensor")
    original = weight.to(dtype=torch.float32).contiguous()
    rows, neuron_len = (int(original.shape[0]), int(original.shape[1]))
    gs = int(spec.groupsize)
    nmax = int(spec.nmax)
    levels = float((1 << int(spec.sub_bits)) - 1)
    pad = (-neuron_len) % gs
    padded = torch.nn.functional.pad(original, (0, pad)) if pad else original
    groups = padded.reshape(rows, -1, gs)

    search_weight = torch.sqrt((groups * groups).sum(dim=-1) / float(gs)).unsqueeze(-1)
    search_weight = search_weight + groups.abs()
    if pad:
        search_weight[:, -1, gs - pad :] = 0.0
    scale, zero_point = make_qkx2_torch(groups, search_weight, nmax=nmax)
    minimum = -zero_point
    neuron_scale = scale.amax(dim=-1)
    neuron_minimum = minimum.amax(dim=-1)
    stored_scale = torch.where(
        neuron_scale > 0,
        (neuron_scale / levels).to(torch.float16).to(torch.float32),
        torch.zeros_like(neuron_scale),
    )
    stored_minimum = torch.where(
        neuron_minimum > 0,
        (neuron_minimum / levels).to(torch.float16).to(torch.float32),
        torch.zeros_like(neuron_minimum),
    )
    scale_denominator = torch.where(
        neuron_scale > 0, neuron_scale, torch.ones_like(neuron_scale)
    )
    minimum_denominator = torch.where(
        neuron_minimum > 0, neuron_minimum, torch.ones_like(neuron_minimum)
    )
    sub_scale = torch.clamp(
        torch.round(levels * scale / scale_denominator.unsqueeze(-1)), 0, levels
    )
    sub_minimum = torch.clamp(
        torch.round(levels * minimum / minimum_denominator.unsqueeze(-1)), 0, levels
    )
    effective_scale = stored_scale.unsqueeze(-1) * sub_scale
    effective_minimum = stored_minimum.unsqueeze(-1) * sub_minimum
    scale_safe = torch.where(
        effective_scale > 0, effective_scale, torch.ones_like(effective_scale)
    )
    quantized = torch.clamp(
        torch.round(
            (groups + effective_minimum.unsqueeze(-1)) / scale_safe.unsqueeze(-1)
        ),
        0,
        nmax,
    )
    reconstruction = (
        effective_scale.unsqueeze(-1) * quantized - effective_minimum.unsqueeze(-1)
    ).reshape(rows, -1)[:, :neuron_len]
    error = reconstruction - original
    return (error * error).sum(dim=1), (original * original).sum(dim=1)


def _evaluate_expert_tensor(
    source,
    shape: tuple[int, int, int],
    profiles: dict[str, NintSpec],
    *,
    device: torch.device,
    expert_batch: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    experts, rows_per_expert, columns = shape
    profile_sse = {name: np.zeros(experts, dtype=np.float64) for name in profiles}
    signal = np.zeros(experts, dtype=np.float64)
    for start in range(0, experts, expert_batch):
        end = min(start + expert_batch, experts)
        host = source[start:end]
        # The safetensors slice is an mmap-backed CPU view.  Keep the H2D copy
        # synchronous so the next slice cannot release its storage while CUDA
        # is still reading from it.
        weight = host.reshape((end - start) * rows_per_expert, columns).to(
            device=device, dtype=torch.float32, non_blocking=False
        )
        local_signal = (weight * weight).sum(dim=1).reshape(end - start, rows_per_expert)
        signal[start:end] = local_signal.sum(dim=1).cpu().numpy().astype(np.float64)
        for profile, spec in profiles.items():
            row_sse, _row_signal = _quantization_sse_by_row(weight, spec)
            profile_sse[profile][start:end] = (
                row_sse.reshape(end - start, rows_per_expert)
                .sum(dim=1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        del host, weight, local_signal
    return profile_sse, signal


def _load_index(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    document = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid safetensor index: {index_path}")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def _tensor_shape(root: Path, weight_map: dict[str, str], name: str) -> tuple[int, ...]:
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"BF16 checkpoint has no tensor {name}")
    with safe_open(str(root / shard), framework="pt", device="cpu") as stream:
        tensor = stream.get_slice(name)
        if str(tensor.get_dtype()) != "BF16":
            raise ValueError(f"tensor {name} is {tensor.get_dtype()}, expected BF16")
        return tuple(int(value) for value in tensor.get_shape())


def _expert_layout(root: Path, weight_map: dict[str, str]) -> ExpertModelLayout:
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    layers = int(text["num_hidden_layers"])
    experts = int(text["num_experts"])
    top_k = int(text.get("top_k_experts", text.get("num_experts_per_tok", 0)))
    hidden = int(text["hidden_size"])
    intermediate = int(text["moe_intermediate_size"])
    candidates = (
        (
            "model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
            "model.language_model.layers.{layer}.mlp.experts.down_proj",
        ),
        (
            "model.language_model.layers.{layer}.experts.gate_up_proj",
            "model.language_model.layers.{layer}.experts.down_proj",
        ),
    )
    for gate_template, down_template in candidates:
        if gate_template.format(layer=0) in weight_map and down_template.format(layer=0) in weight_map:
            return ExpertModelLayout(
                model_type=str(config.get("model_type", text.get("model_type", "unknown"))),
                layers=layers,
                experts=experts,
                top_k=top_k,
                hidden_size=hidden,
                intermediate_size=intermediate,
                gate_template=gate_template,
                down_template=down_template,
            )
    raise ValueError("cannot identify routed-expert tensor names from the checkpoint")


def _layer_documents(
    root: Path,
    weight_map: dict[str, str],
    observations,
    layer: int,
    profiles: dict[str, NintSpec],
    layout: ExpertModelLayout,
    *,
    device: torch.device,
    expert_batch: int,
) -> list[dict[str, Any]]:
    gate_name = layout.gate_template.format(layer=layer)
    down_name = layout.down_template.format(layer=layer)
    gate_shape = _tensor_shape(root, weight_map, gate_name)
    down_shape = _tensor_shape(root, weight_map, down_name)
    expected_gate = (layout.experts, 2 * layout.intermediate_size, layout.hidden_size)
    expected_down = (layout.experts, layout.hidden_size, layout.intermediate_size)
    if gate_shape != expected_gate or down_shape != expected_down:
        raise ValueError(
            f"layer {layer} expert shape mismatch: gate={gate_shape}, down={down_shape}"
        )

    # torch-backed safetensors slices can fault at offsets above 2 GiB on
    # Windows. Read the BF16 payload through a NumPy memmap and copy each
    # expert batch synchronously to CUDA.
    with open_bfloat16_memmap_tensor(
        root / weight_map[gate_name], gate_name
    ) as gate_source:
        gate_sse, gate_signal = _evaluate_expert_tensor(
            gate_source,
            gate_shape,
            profiles,
            device=device,
            expert_batch=expert_batch,
        )

    with open_bfloat16_memmap_tensor(
        root / weight_map[down_name], down_name
    ) as down_source:
        down_sse, down_signal = _evaluate_expert_tensor(
            down_source,
            down_shape,
            profiles,
            device=device,
            expert_batch=expert_batch,
        )

    rows: list[dict[str, Any]] = []
    for expert in range(gate_shape[0]):
        observation = observations[(layer, expert)]
        for profile, spec in profiles.items():
            item = ExpertProfileEvaluation(
                layer=layer,
                expert=expert,
                profile=profile,
                spec=spec,
                gate_name=gate_name,
                down_name=down_name,
                gate_rows=gate_shape[1],
                gate_columns=gate_shape[2],
                down_rows=down_shape[1],
                down_columns=down_shape[2],
                exposure=observation.exposure,
                normalized_exposure=observation.normalized_exposure,
                gate_sse=float(gate_sse[profile][expert]),
                gate_signal=float(gate_signal[expert]),
                down_sse=float(down_sse[profile][expert]),
                down_signal=float(down_signal[expert]),
            )
            rows.append(item.as_document())
    return rows


def _atomic_json(path: Path, document: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> None:
    root = Path(args.input).resolve()
    reap_table = Path(args.reap_table).resolve()
    work_dir = Path(args.work_dir).resolve()
    scheme_path = Path(args.output_scheme).resolve()
    candidate_path = Path(args.candidate_table).resolve()
    report_path = Path(args.report).resolve()
    for path in (scheme_path, candidate_path, report_path):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = work_dir / "layers"
    layer_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("production REAP expert evaluation requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    weight_map = _load_index(root)
    layout = _expert_layout(root, weight_map)
    observations = load_reap_expert_table(
        reap_table,
        expected_layers=layout.layers,
        expected_experts=layout.experts,
        expected_top_k=layout.top_k,
    )
    profiles = resolve_profiles(args.profiles)
    started = time.time()
    all_documents: list[dict[str, Any]] = []
    for layer in range(layout.layers):
        layer_path = layer_dir / f"layer-{layer:03d}.json"
        if layer_path.exists():
            if not args.resume:
                raise FileExistsError(f"layer artifact already exists: {layer_path}")
            rows = json.loads(layer_path.read_text(encoding="utf-8"))
            if len(rows) != layout.experts * len(profiles):
                raise ValueError(f"invalid resumed layer artifact: {layer_path}")
            status = "reused"
        else:
            t0 = time.time()
            rows = _layer_documents(
                root,
                weight_map,
                observations,
                layer,
                profiles,
                layout,
                device=device,
                expert_batch=args.expert_batch,
            )
            _atomic_json(layer_path, rows)
            status = "computed"
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "status": status,
                        "rows": len(rows),
                        "sec": round(time.time() - t0, 3),
                        "elapsed_sec": round(time.time() - started, 3),
                    }
                ),
                flush=True,
            )
        if status == "reused":
            print(json.dumps({"layer": layer, "status": status}), flush=True)
        all_documents.extend(rows)

    evaluations = [evaluation_from_document(row) for row in all_documents]
    scheme, allocation_report = allocate_expert_profiles(
        evaluations,
        target_profile=args.target_profile,
        metadata={
            "source_model": str(root),
            "source_index_sha256": _sha256(root / "model.safetensors.index.json"),
            "reap_table": str(reap_table),
            "reap_table_sha256": _sha256(reap_table),
            "reap_effective_tokens": max(
                observation.total_tokens for observation in observations.values()
            ),
            "reap_top_k": layout.top_k,
            "model_type": layout.model_type,
            "layers": layout.layers,
            "experts_per_layer": layout.experts,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
    )
    candidate_tmp = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    with candidate_tmp.open("w", encoding="utf-8") as stream:
        for row in all_documents:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(candidate_tmp, candidate_path)
    save_scheme(scheme_path, scheme)
    allocation_report.update(
        {
            "input": str(root),
            "reap_table": str(reap_table),
            "reap_table_sha256": _sha256(reap_table),
            "candidate_table": str(candidate_path),
            "candidate_count": len(evaluations),
            "scheme": str(scheme_path),
            "scheme_storage_bits": scheme.storage_bits,
            "scheme_bpw": scheme.bpw,
            "elapsed_sec": time.time() - started,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        }
    )
    _atomic_json(report_path, allocation_report)
    print(json.dumps({"status": "ok", **allocation_report}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="BF16 safetensor model directory")
    parser.add_argument("--reap-table", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-scheme", required=True)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(NINT_EXPERT_PROFILES),
        choices=tuple(NINT_EXPERT_PROFILES),
    )
    parser.add_argument(
        "--target-profile",
        choices=tuple(NINT_EXPERT_PROFILES),
        default="NINT5",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expert-batch", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.expert_batch <= 0:
        parser.error("--expert-batch must be positive")
    if args.target_profile not in args.profiles:
        parser.error("--target-profile must be included in --profiles")
    build(args)


if __name__ == "__main__":
    main()
