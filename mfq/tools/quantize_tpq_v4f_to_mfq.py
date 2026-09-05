"""Quantize DeepSeek-V4-Flash with TPQ's native weight-only recipe."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch

from mfq.calibration.artifact import load_scheme
from mfq.formats.header import FileHeader
from mfq.formats.runtime_profile import (
    RUNTIME_SAMPLING_METADATA_KEY,
    architecture_profile,
)
from mfq.formats.shards import (
    matching_shard_paths,
    parse_size,
    validate_split_limits,
    write_blob_record_shards,
)
from mfq.formats.tpq import (
    TPQ_PQ_SPECS,
    pack_tpq_int4_prefix,
    tpq_int4_payload_nbytes,
)
from mfq.quantize.mxfp import read_dense_rows, read_mxfp8_rows
from mfq.quantize.tpq import quantize_tpq_int4
from mfq.quantize.v4f_source import V4FCheckpoint
from mfq.tools.quantize_hf_to_mfq import (
    BlobRecord,
    _mixed_moe_blob_nbytes,
    _write_mixed_moe_axis0_blob,
)

_TIER_CHARACTER = {
    "TPQ-X": "x",
    "TPQ-W": "w",
    "TPQ-V": "v",
    "TPQ-VV": "V",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tpq_config(root: Path) -> dict:
    source = json.loads((root / "config.json").read_text(encoding="utf-8"))
    eos = source.get("eos_token_id", [])
    if isinstance(eos, int):
        eos = [eos]
    n_layers = int(source["num_hidden_layers"])
    compress_ratios = [int(value) for value in (source.get("compress_ratios") or [])]
    target_layers = [int(value) for value in (source.get("dspark_target_layer_ids") or [])]
    structural_counts = {
        count
        for count in (
            max(0, len(compress_ratios) - n_layers),
            len(target_layers),
            int(source.get("n_mtp_layers", 0) or 0),
        )
        if count > 0
    }
    if len(structural_counts) > 1:
        raise ValueError(
            f"DeepSeek DSpark config has inconsistent stage counts: {sorted(structural_counts)}"
        )
    n_mtp_layers = (
        next(iter(structural_counts))
        if structural_counts
        else (
            int(source.get("num_nextn_predict_layers", 0) or 0)
            if int(source.get("dspark_block_size", 0) or 0) > 0
            else 0
        )
    )
    return {
        "n_layers": n_layers,
        "hidden": int(source["hidden_size"]),
        "n_experts": int(source["n_routed_experts"]),
        "top_k": int(source["num_experts_per_tok"]),
        "moe_inter": int(source["moe_intermediate_size"]),
        "n_shared": int(source.get("n_shared_experts", 0)),
        "n_heads": int(source["num_attention_heads"]),
        "head_dim": int(source.get("head_dim", 512)),
        "q_lora_rank": int(source["q_lora_rank"]),
        "o_lora_rank": int(source.get("o_lora_rank", 0)),
        "o_groups": int(source.get("o_groups", 1)),
        "kv_dim": int(source.get("kv_dim", 512)),
        "qk_rope_head_dim": int(source["qk_rope_head_dim"]),
        "n_kv_heads": int(source.get("num_key_value_heads", 1)),
        "vocab": int(source["vocab_size"]),
        "rms_eps": float(source.get("rms_norm_eps", 1e-6)),
        "scoring_func": str(source.get("scoring_func", "sqrtsoftplus")),
        "norm_topk_prob": bool(source.get("norm_topk_prob", True)),
        "routed_scaling": float(source.get("routed_scaling_factor", 1.0)),
        "swiglu_limit": float(source.get("swiglu_limit", 0.0)),
        "n_hash_layers": int(source.get("num_hash_layers", 0)),
        "sliding_window": int(source.get("sliding_window", 0)),
        "index_topk": int(source.get("index_topk", 0)),
        "rope_theta": float(source.get("rope_theta", 10000.0)),
        "rope_scaling": dict(source.get("rope_scaling") or {}),
        "eos_token_id": [int(value) for value in eos],
        "index_n_heads": int(source.get("index_n_heads", 64)),
        "index_head_dim": int(source.get("index_head_dim", 128)),
        "max_position_embeddings": int(source.get("max_position_embeddings", 0)),
        # Older 0731 text checkpoint configs report one next-token layer even
        # though their tensors contain the three-stage DSpark head.  The
        # compress-ratio tail and target list provide the structural count;
        # newer Vision configs expose all three stages directly.
        "n_mtp_layers": n_mtp_layers,
        "dspark_block_size": int(source.get("dspark_block_size", 0) or 0),
        "dspark_noise_token_id": int(source.get("dspark_noise_token_id", 0) or 0),
        "dspark_target_layer_ids": target_layers,
        "dspark_markov_rank": int(source.get("dspark_markov_rank", 256) or 256),
        "hc_mult": int(source.get("hc_mult", 4)),
        "hc_eps": float(source.get("hc_eps", 1e-6)),
        "hc_sinkhorn_iters": int(source.get("hc_sinkhorn_iters", 20)),
        "compress_rope_theta": float(source.get("compress_rope_theta", 160000.0)),
        "compress_ratios": compress_ratios,
        "vision_n_layers": int(source.get("vision_n_layers", 0) or 0),
        "vision_dim": int(source.get("vision_dim", 1024) or 1024),
        "vision_n_heads": int(source.get("vision_n_heads", 16) or 16),
        "vision_inter_dim": int(source.get("vision_inter_dim", 2816) or 2816),
        "vision_patch_size": int(source.get("vision_patch_size", 14) or 14),
        "vision_rope_theta": float(source.get("vision_rope_theta", 10000.0) or 10000.0),
        "vision_downsample_ratio": int(source.get("vision_downsample_ratio", 3) or 3),
        "vision_max_n_token": int(source.get("vision_max_n_token", 384) or 384),
        "vision_min_pixels": int(source.get("vision_min_pixels", 147456) or 147456),
        "vision_max_wh_ratio": int(source.get("vision_max_wh_ratio", 8) or 8),
    }


def _is_dense_source(name: str) -> bool:
    return (
        not name.startswith("mtp.") and ".ffn.experts." not in name and not name.endswith(".scale")
    )


def _is_dspark_dense_source(name: str) -> bool:
    return name.startswith("mtp.") and ".ffn.experts." not in name and not name.endswith(".scale")


def _scale_name(checkpoint: V4FCheckpoint, name: str) -> str | None:
    candidate = (
        name.removesuffix("weight") + "scale" if name.endswith("weight") else name + ".scale"
    )
    return candidate if candidate in checkpoint.weight_map else None


def _read_rows(
    checkpoint: V4FCheckpoint,
    name: str,
    start: int,
    stop: int,
    *,
    device: str,
) -> torch.Tensor:
    reader = checkpoint.reader_for(name)
    scale_name = _scale_name(checkpoint, name)
    if scale_name is not None:
        if checkpoint.shard_for(scale_name) != checkpoint.shard_for(name):
            raise ValueError(f"MXFP8 weight and scale are split: {name}")
        return read_mxfp8_rows(
            reader,
            name,
            scale_name,
            start,
            stop,
            device=device,
        )
    return read_dense_rows(
        reader,
        name,
        start,
        stop,
        device=device,
        dtype=torch.float32,
    )


def _f32_blob_nbytes(shape: tuple[int, ...]) -> int:
    return 4 + 8 * len(shape) + math.prod(shape) * 4


def _write_f32_blob(
    checkpoint: V4FCheckpoint,
    name: str,
    path: Path,
    *,
    row_chunk: int,
    device: str,
) -> int:
    shape = tuple(int(value) for value in checkpoint.info(name).shape)
    if not shape:
        raise ValueError(f"TPQ dense scalar is unsupported: {name}")
    with path.open("wb") as output:
        output.write(struct.pack("<I", len(shape)))
        output.write(struct.pack(f"<{len(shape)}q", *shape))
        for start in range(0, shape[0], row_chunk):
            end = min(start + row_chunk, shape[0])
            rows = _read_rows(
                checkpoint,
                name,
                start,
                end,
                device=device,
            )
            output.write(
                np.ascontiguousarray(
                    rows.float().cpu().numpy(),
                    dtype="<f4",
                ).tobytes()
            )
            del rows
    return path.stat().st_size


def _write_int4_blob(
    checkpoint: V4FCheckpoint,
    name: str,
    path: Path,
    *,
    row_chunk: int,
    device: str,
    group_size: int = 64,
) -> int:
    shape = tuple(int(value) for value in checkpoint.info(name).shape)
    if len(shape) != 2:
        raise ValueError(f"TPQ int4 source is not a matrix: {name}")
    rows, columns = shape
    expected = tpq_int4_payload_nbytes(shape, group_size)
    prefix = pack_tpq_int4_prefix(shape, group_size)
    packed_row_bytes = columns // 2
    scale_row_bytes = columns // group_size * 2
    packed_offset = len(prefix)
    scale_offset = packed_offset + rows * packed_row_bytes
    with path.open("wb+") as output:
        output.write(prefix)
        output.truncate(expected)
        for start in range(0, rows, row_chunk):
            end = min(start + row_chunk, rows)
            values = _read_rows(
                checkpoint,
                name,
                start,
                end,
                device=device,
            )
            tensor = quantize_tpq_int4(values, group_size=group_size)
            output.seek(packed_offset + start * packed_row_bytes)
            output.write(tensor.packed.tobytes())
            output.seek(scale_offset + start * scale_row_bytes)
            output.write(tensor.scales.astype("<f2", copy=False).tobytes())
            del values, tensor
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"TPQ int4 blob size mismatch: {actual} != {expected}")
    return actual


def _expert_precisions(selection) -> tuple:
    by_expert = {int(item.expert_id): item.descriptor for item in selection.selections}
    if sorted(by_expert) != list(range(selection.n_experts)):
        raise ValueError(f"incomplete TPQ expert selection: {selection.name}")
    return tuple(by_expert[expert] for expert in range(selection.n_experts))


def _dspark_expert_plan(
    config: dict,
) -> tuple[tuple[int, str, str, str], ...]:
    stages = int(config.get("n_mtp_layers", 0) or 0)
    targets = [int(value) for value in (config.get("dspark_target_layer_ids") or [])]
    if len(targets) != stages:
        raise ValueError("DeepSeek DSpark target-layer count does not match its stages")
    return tuple(
        (
            stage,
            projection,
            f"mtp.{stage}.ffn.experts.{projection}.weight",
            f"blk.{target}.ffn_{projection}_exps.weight",
        )
        for stage, target in enumerate(targets)
        for projection in ("gate_up", "down")
    )


def _tier_strings(scheme, n_layers: int, n_experts: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for layer in range(n_layers):
        name = f"blk.{layer}.ffn_gate_up_exps.weight"
        try:
            selection = scheme.expert_selections[name]
        except KeyError as exc:
            raise ValueError(f"TPQ scheme is missing layer {layer}") from exc
        families = _expert_precisions(selection)
        if len(families) != n_experts:
            raise ValueError(f"TPQ layer {layer} expert count differs")
        try:
            result[str(layer)] = "".join(_TIER_CHARACTER[item.family] for item in families)
        except KeyError as exc:
            raise ValueError(f"TPQ layer {layer} contains a non-TPQ precision") from exc
    return result


def _manifest(
    root: Path,
    scheme,
    config: dict,
    *,
    include_dspark: bool,
    source_dspark_layers: int,
) -> dict:
    tiers = _tier_strings(
        scheme,
        int(config["n_layers"]),
        int(config["n_experts"]),
    )
    tokenizer_files = [
        name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
            "special_tokens_map.json",
        )
        if (root / name).is_file()
    ]
    present_tiers = sorted({character for encoded in tiers.values() for character in encoded})
    return {
        "format": "tpq-1",
        "config": config,
        "quant": {
            "expert": "fixed-tiers-per-layer",
            "tiers": [],
            "kinds": present_tiers,
            "zlib": False,
            "dense": "int4-g64",
            "dense_bits": 4,
            "int4_group": 64,
            "vq": {
                spec.tier: [spec.vector_size, spec.codebook_entries]
                for spec in TPQ_PQ_SPECS.values()
            },
            "profile": Path(str(scheme.metadata.get("profile", "tiers_per_layer"))).name,
        },
        "tiers_per_layer": tiers,
        "dense_file": "native-mfq",
        "expert_files": {str(layer): "native-mfq" for layer in range(int(config["n_layers"]))},
        "skipped": ([] if include_dspark or source_dspark_layers == 0 else ["dspark(mtp.*)"]),
        "tokenizer_files": tokenizer_files,
    }


def convert(
    *,
    input_root: str | Path,
    scheme_path: str | Path,
    output_path: str | Path,
    work_dir: str | Path,
    temp_dir: str | Path | None = None,
    device: str = "cuda",
    row_chunk: int = 512,
    overwrite: bool = False,
    split_max_size: int = 0,
    split_max_tensors: int = 0,
    include_dspark: bool = True,
) -> Path:
    """Run TPQ's original dense/expert quantization into MFQ output."""

    if row_chunk <= 0:
        raise ValueError("TPQ row chunk must be positive")
    validate_split_limits(split_max_size, split_max_tensors)
    root = Path(input_root).resolve()
    scheme_file = Path(scheme_path).resolve()
    output = Path(output_path).resolve()
    run_dir = Path(work_dir).resolve()
    blobs = Path(temp_dir).resolve() if temp_dir else run_dir / "blobs"
    run_dir.mkdir(parents=True, exist_ok=True)
    blobs.mkdir(parents=True, exist_ok=True)
    if (output.exists() or matching_shard_paths(output)) and not overwrite:
        raise FileExistsError(f"MFQ output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    scheme = load_scheme(scheme_file)
    if scheme.metadata.get("codebook_objective") != "euclidean_sse":
        raise ValueError("TPQ quantization requires Euclidean weight-only codebooks")
    if scheme.metadata.get("codebook_calibration_data") != "none":
        raise ValueError("TPQ quantization cannot consume an imatrix")
    checkpoint = V4FCheckpoint(root)
    config = _tpq_config(root)
    source_dspark_layers = int(config.get("n_mtp_layers", 0))
    if include_dspark and source_dspark_layers > 0:
        missing_roots = [
            f"mtp.{stage}.attn.wq_a.weight"
            for stage in range(source_dspark_layers)
            if f"mtp.{stage}.attn.wq_a.weight" not in checkpoint.weight_map
        ]
        if missing_roots:
            raise ValueError(
                "DeepSeek config advertises DSpark but checkpoint is missing: "
                + ", ".join(missing_roots)
            )
    if not include_dspark:
        config = dict(config)
        config["n_mtp_layers"] = 0
        config["dspark_block_size"] = 0
        config["dspark_target_layer_ids"] = []
        config["compress_ratios"] = list(config["compress_ratios"])[: int(config["n_layers"])]
    manifest = _manifest(
        root,
        scheme,
        config,
        include_dspark=include_dspark,
        source_dspark_layers=source_dspark_layers,
    )
    artifact_root = scheme_file.parent
    progress_path = run_dir / "convert_progress.jsonl"
    records: list[BlobRecord] = []
    started = time.perf_counter()
    index = 0

    dense_names = sorted(name for name in checkpoint.weight_map if _is_dense_source(name))
    for name in dense_names:
        index += 1
        info = checkpoint.info(name)
        shape = tuple(int(value) for value in info.shape)
        use_int4 = len(shape) == 2 and math.prod(shape) >= 65_536 and shape[1] % 64 == 0
        dtype = "TPQ-I4G64" if use_int4 else "F32"
        expected = tpq_int4_payload_nbytes(shape, 64) if use_int4 else _f32_blob_nbytes(shape)
        blob = blobs / f"{index:05d}.blob"
        item_started = time.perf_counter()
        if blob.is_file() and blob.stat().st_size == expected:
            status = "reused"
        else:
            if blob.exists():
                raise ValueError(f"TPQ blob has the wrong size: {blob}")
            temporary = blob.with_suffix(".blob.partial")
            temporary.unlink(missing_ok=True)
            actual = (
                _write_int4_blob(
                    checkpoint,
                    name,
                    temporary,
                    row_chunk=row_chunk,
                    device=device,
                )
                if use_int4
                else _write_f32_blob(
                    checkpoint,
                    name,
                    temporary,
                    row_chunk=row_chunk,
                    device=device,
                )
            )
            if actual != expected:
                raise RuntimeError(
                    f"TPQ dense blob size mismatch for {name}: {actual} != {expected}"
                )
            os.replace(temporary, blob)
            status = "written"
        records.append(BlobRecord(name, dtype, expected, blob))
        event = {
            "completed": index,
            "name": name,
            "dtype": dtype,
            "blob_bytes": expected,
            "status": status,
            "seconds": time.perf_counter() - item_started,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    for layer in range(int(config["n_layers"])):
        for projection, target_name in (
            (
                "gate_up",
                f"layers.{layer}.ffn.experts.gate_up.weight",
            ),
            ("down", f"layers.{layer}.ffn.experts.down.weight"),
        ):
            index += 1
            selection_name = f"blk.{layer}.ffn_{projection}_exps.weight"
            selection = scheme.expert_selections[selection_name]
            precisions = _expert_precisions(selection)
            source = checkpoint.expert_source(layer, projection)
            expected = _mixed_moe_blob_nbytes(
                source.shape,
                precisions,
                artifact_root,
            )
            blob = blobs / f"{index:05d}.blob"
            item_started = time.perf_counter()
            if blob.is_file() and blob.stat().st_size == expected:
                status = "reused"
            else:
                if blob.exists():
                    raise ValueError(f"TPQ blob has the wrong size: {blob}")
                temporary = blob.with_suffix(".blob.partial")
                temporary.unlink(missing_ok=True)
                actual = _write_mixed_moe_axis0_blob(
                    source,
                    source.shape,
                    source.shape,
                    precisions,
                    temporary,
                    row_chunk,
                    "cuda",
                    device,
                    artifact_root,
                    importance=None,
                )
                if actual != expected:
                    raise RuntimeError(
                        f"TPQ expert blob size mismatch for {target_name}: {actual} != {expected}"
                    )
                os.replace(temporary, blob)
                status = "written"
            records.append(BlobRecord(target_name, "NINTM", expected, blob))
            event = {
                "completed": index,
                "name": target_name,
                "dtype": "NINTM",
                "blob_bytes": expected,
                "status": status,
                "seconds": time.perf_counter() - item_started,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            del source

    dspark_tensor_count = 0
    if include_dspark and source_dspark_layers > 0:
        # Append DSpark after every legacy/base tensor so an interrupted run
        # can safely reuse the old ordinal blob cache.
        for name in sorted(
            value for value in checkpoint.weight_map if _is_dspark_dense_source(value)
        ):
            index += 1
            info = checkpoint.info(name)
            shape = tuple(int(value) for value in info.shape)
            use_int4 = len(shape) == 2 and math.prod(shape) >= 65_536 and shape[1] % 64 == 0
            dtype = "TPQ-I4G64" if use_int4 else "F32"
            expected = tpq_int4_payload_nbytes(shape, 64) if use_int4 else _f32_blob_nbytes(shape)
            blob = blobs / f"{index:05d}.blob"
            item_started = time.perf_counter()
            if blob.is_file() and blob.stat().st_size == expected:
                status = "reused"
            else:
                if blob.exists():
                    raise ValueError(f"TPQ blob has the wrong size: {blob}")
                temporary = blob.with_suffix(".blob.partial")
                temporary.unlink(missing_ok=True)
                actual = (
                    _write_int4_blob(
                        checkpoint,
                        name,
                        temporary,
                        row_chunk=row_chunk,
                        device=device,
                    )
                    if use_int4
                    else _write_f32_blob(
                        checkpoint,
                        name,
                        temporary,
                        row_chunk=row_chunk,
                        device=device,
                    )
                )
                if actual != expected:
                    raise RuntimeError(
                        f"TPQ dense blob size mismatch for {name}: {actual} != {expected}"
                    )
                os.replace(temporary, blob)
                status = "written"
            records.append(BlobRecord(name, dtype, expected, blob))
            dspark_tensor_count += 1
            event = {
                "completed": index,
                "name": name,
                "dtype": dtype,
                "blob_bytes": expected,
                "status": status,
                "seconds": time.perf_counter() - item_started,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)

        for stage, projection, target_name, selection_name in _dspark_expert_plan(config):
            index += 1
            try:
                selection = scheme.expert_selections[selection_name]
            except KeyError as exc:
                raise ValueError(
                    "TPQ scheme cannot supply DSpark stage "
                    f"{stage}; missing analogue {selection_name}"
                ) from exc
            precisions = _expert_precisions(selection)
            source = checkpoint.expert_source(
                stage,
                projection,
                namespace="mtp",
            )
            expected = _mixed_moe_blob_nbytes(
                source.shape,
                precisions,
                artifact_root,
            )
            blob = blobs / f"{index:05d}.blob"
            item_started = time.perf_counter()
            if blob.is_file() and blob.stat().st_size == expected:
                status = "reused"
            else:
                if blob.exists():
                    raise ValueError(f"TPQ blob has the wrong size: {blob}")
                temporary = blob.with_suffix(".blob.partial")
                temporary.unlink(missing_ok=True)
                actual = _write_mixed_moe_axis0_blob(
                    source,
                    source.shape,
                    source.shape,
                    precisions,
                    temporary,
                    row_chunk,
                    "cuda",
                    device,
                    artifact_root,
                    importance=None,
                )
                if actual != expected:
                    raise RuntimeError(
                        "TPQ DSpark expert blob size mismatch for "
                        f"{target_name}: {actual} != {expected}"
                    )
                os.replace(temporary, blob)
                status = "written"
            records.append(BlobRecord(target_name, "NINTM", expected, blob))
            dspark_tensor_count += 1
            event = {
                "completed": index,
                "name": target_name,
                "dtype": "NINTM",
                "blob_bytes": expected,
                "status": status,
                "seconds": time.perf_counter() - item_started,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            del source

    runtime_profile = architecture_profile("deepseek_v4")
    header = FileHeader(
        version=2,
        model_arch="deepseek_v4-tpq-mfq",
        num_tensors=len(records),
        extra={
            **(
                {RUNTIME_SAMPLING_METADATA_KEY: runtime_profile}
                if runtime_profile is not None
                else {}
            ),
            "source_format": "tpq-1",
            "source_index_sha256": _sha256(root / "model.safetensors.index.json"),
            "scheme_sha256": _sha256(scheme_file),
            "tpq_manifest": manifest,
            "tpq_index_storage": {spec.tier: spec.index_bits for spec in TPQ_PQ_SPECS.values()},
            "mtp_included": bool(dspark_tensor_count),
            "mtp": {
                "architecture": "deepseek_v4_dspark",
                "included": bool(dspark_tensor_count),
                "hidden_layers": (source_dspark_layers if dspark_tensor_count else 0),
                "tensor_count": dspark_tensor_count,
            },
        },
    )
    outputs = write_blob_record_shards(
        output,
        header,
        records,
        split_max_size=split_max_size,
        split_max_tensors=split_max_tensors,
        overwrite=overwrite,
        consume_blobs=True,
    )
    if len(outputs) > 1 and output.exists() and overwrite:
        output.unlink()
    output_bytes = sum(path.stat().st_size for path in outputs)
    print(
        json.dumps(
            {
                "output": str(outputs[0]),
                "outputs": [str(path) for path in outputs],
                "shard_count": len(outputs),
                "bytes": output_bytes,
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    return outputs[0]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--temp-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-chunk", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-dspark",
        action="store_true",
        help="omit the DeepSeek DSpark MoE speculative head",
    )
    split = parser.add_mutually_exclusive_group()
    split.add_argument(
        "--split-max-size",
        type=parse_size,
        default=0,
        metavar="N[M|G]",
    )
    split.add_argument("--split-max-tensors", type=int, default=0)
    args = parser.parse_args()
    convert(
        input_root=args.input,
        scheme_path=args.scheme,
        output_path=args.output,
        work_dir=args.work_dir,
        temp_dir=args.temp_dir,
        device=args.device,
        row_chunk=args.row_chunk,
        overwrite=args.overwrite,
        split_max_size=args.split_max_size,
        split_max_tensors=args.split_max_tensors,
        include_dspark=not args.no_dspark,
    )


if __name__ == "__main__":
    main()
