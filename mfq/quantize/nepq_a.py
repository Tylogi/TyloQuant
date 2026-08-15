"""Fixed-table NEPQ-A quantization with a sparse additive residual."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from mfq.formats.nepq import (
    NEPQ0_A,
    NEPQ1_A,
    NepqSpec,
    NepqTensor,
    _decoded_tables,
    dequantize_nepq_rows,
    nepq_base_spec,
    nepq_spec,
    pack_nepq,
    rotation_signs,
    validate_nepq,
)
from mfq.quantize.nepq import (
    NepqQuantConfig,
    _fwht_blocks,
    _slice_rows,
    _source_shape,
    quantize_nepq_fixed,
)


@dataclass(frozen=True)
class NepqAArtifact:
    """Frozen base-table pool and projection-global residual dictionary."""

    table_payloads: np.ndarray
    residual_codebook: np.ndarray


@dataclass(frozen=True)
class NepqAQuantConfig:
    base: NepqQuantConfig = field(default_factory=NepqQuantConfig)
    residual_row_chunk: int = 128
    residual_block_chunk: int = 1024

    def __post_init__(self) -> None:
        if self.residual_row_chunk <= 0 or self.residual_block_chunk <= 0:
            raise ValueError("NEPQ-A residual chunk sizes must be positive")


def _residual_dictionary(value) -> np.ndarray:
    dictionary = np.asarray(value, dtype=np.float32)
    if dictionary.shape != (1024, 8):
        raise ValueError(
            f"NEPQ-A residual dictionary must have shape (1024,8), got {dictionary.shape}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        stored = dictionary.astype(np.float16)
    if not np.isfinite(stored).all():
        raise ValueError("NEPQ-A residual dictionary must be finite")
    if not np.any(np.all(stored == 0, axis=1)):
        zero_id = int(np.argmin(np.square(stored.astype(np.float32)).sum(axis=1)))
        stored[zero_id] = 0
    return np.ascontiguousarray(stored)


def _best_records(
    blocks: torch.Tensor,
    dictionary: torch.Tensor,
    dictionary_norm: torch.Tensor,
    valid_vectors: torch.Tensor,
    position_bits: int,
    block_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total, block_vectors, _ = blocks.shape
    if blocks.device.type == "mps":
        from mfq.quantize.metal.nepq_a import best_records

        return best_records(
            blocks,
            dictionary,
            dictionary_norm,
            valid_vectors,
            position_bits,
        )
    records = torch.empty(total, device=blocks.device, dtype=torch.int64)
    gains = torch.empty(total, device=blocks.device, dtype=torch.float32)
    positions = torch.arange(block_vectors, device=blocks.device).reshape(1, -1, 1)
    for start in range(0, total, block_chunk):
        stop = min(start + block_chunk, total)
        value = blocks[start:stop]
        score = 2.0 * torch.matmul(value.reshape(-1, value.shape[-1]), dictionary.T).reshape(
            stop - start, block_vectors, -1
        )
        score.sub_(dictionary_norm.reshape(1, 1, -1))
        score.masked_fill_(positions >= valid_vectors[start:stop].reshape(-1, 1, 1), -torch.inf)
        flat_score = score.reshape(stop - start, -1)
        best_gain, best = flat_score.max(1)
        position = torch.div(best, dictionary.shape[0], rounding_mode="floor")
        dictionary_id = best.remainder(dictionary.shape[0])
        records[start:stop] = position | (dictionary_id << position_bits)
        gains[start:stop] = best_gain
    return records, gains


def _residual_records(
    residual: torch.Tensor,
    spec: NepqSpec,
    dictionary: torch.Tensor,
    dictionary_norm: torch.Tensor,
    block_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, width = residual.shape
    vectors = width // spec.vector_size
    blocks_per_row = math.ceil(vectors / spec.residual_block_vectors)
    padded_vectors = blocks_per_row * spec.residual_block_vectors
    value = residual.reshape(rows, vectors, spec.vector_size)
    if padded_vectors != vectors:
        value = torch.nn.functional.pad(value, (0, 0, 0, padded_vectors - vectors))
    blocks = value.reshape(-1, spec.residual_block_vectors, spec.vector_size)
    block_in_row = torch.arange(blocks_per_row, device=residual.device).repeat(rows)
    valid = torch.clamp(
        vectors - block_in_row * spec.residual_block_vectors,
        min=1,
        max=spec.residual_block_vectors,
    )
    first, _ = _best_records(
        blocks,
        dictionary,
        dictionary_norm,
        valid,
        spec.residual_position_bits,
        block_chunk,
    )
    position_mask = (1 << spec.residual_position_bits) - 1
    first_position = first & position_mask
    first_dictionary = first >> spec.residual_position_bits
    block_ids = torch.arange(blocks.shape[0], device=residual.device)
    blocks[block_ids, first_position] -= dictionary[first_dictionary]
    if spec.residual_second:
        second, second_gain = _best_records(
            blocks,
            dictionary,
            dictionary_norm,
            valid,
            spec.residual_position_bits,
            block_chunk,
        )
    else:
        second = torch.empty(0, device=residual.device, dtype=torch.int64)
        second_gain = torch.empty(0, device=residual.device, dtype=torch.float32)
    return first, second, second_gain


def _second_capacity(
    spec: NepqSpec,
    base_nbytes: int,
    block_count: int,
    target_nbytes: int,
) -> tuple[int, int]:
    fixed = (
        base_nbytes
        + 64
        + 1024 * spec.vector_size * 2
        + math.ceil(block_count * spec.residual_record_bits / 8)
        + math.ceil(block_count / 8)
    )
    remaining = int(target_nbytes) - fixed
    if remaining < 0:
        raise ValueError(
            f"{spec.label} target contains {target_nbytes} bytes but needs at least {fixed}"
        )
    capacity = min(block_count, remaining * 8 // spec.residual_record_bits)
    return capacity, fixed


@torch.inference_mode()
def quantize_nepq_a_fixed(
    weight,
    spec: str | int | NepqSpec,
    artifact: NepqAArtifact,
    *,
    importance=None,
    initial_anchor=None,
    rotation_block: int = 2048,
    rotation_seed: int = 0,
    second_records: int | None = None,
    target_nbytes: int | None = None,
    target_bpw: float | None = None,
    config: NepqAQuantConfig | None = None,
    device: str | torch.device = "cuda",
    progress: Callable[[int, int], None] | None = None,
) -> NepqTensor:
    """Quantize one expert cohort using the fixed NEPQ-A artifact."""

    spec = nepq_spec(spec)
    if spec not in (NEPQ0_A, NEPQ1_A):
        raise ValueError(f"{spec.label} is not an NEPQ-A profile")
    if not isinstance(artifact, NepqAArtifact):
        raise TypeError("NEPQ-A requires a NepqAArtifact")
    if not rotation_block:
        raise ValueError(f"{spec.label} requires a Hadamard rotation")
    config = NepqAQuantConfig() if config is None else config
    shape = _source_shape(weight)
    rows = shape[0] * shape[1]
    width = shape[2]
    if rotation_block & (rotation_block - 1) or width % rotation_block:
        raise ValueError("NEPQ-A rotation block must be a power of two dividing K")
    dictionary_np = _residual_dictionary(artifact.residual_codebook)
    base = quantize_nepq_fixed(
        weight,
        nepq_base_spec(spec),
        artifact.table_payloads,
        importance=importance,
        initial_anchor=initial_anchor,
        rotation_block=rotation_block,
        rotation_seed=rotation_seed,
        config=config.base,
        device=device,
    )
    base.neuron_scale = np.ascontiguousarray(
        np.asarray(base.neuron_scale, dtype=np.float16).astype(np.float32)
    )
    base_nbytes = len(pack_nepq(base))
    blocks_per_row = math.ceil((width // spec.vector_size) / spec.residual_block_vectors)
    block_count = rows * blocks_per_row

    if spec is NEPQ0_A:
        if any(value is not None for value in (second_records, target_nbytes, target_bpw)):
            raise ValueError("NEPQ0-A has a fixed one-record residual rate")
        second_limit = 0
        fixed_nbytes = 0
    else:
        choices = sum(value is not None for value in (second_records, target_nbytes, target_bpw))
        if choices > 1:
            raise ValueError("provide at most one of second_records, target_nbytes, or target_bpw")
        if choices == 0:
            target_bpw = 1.5625
        if target_bpw is not None:
            if not np.isfinite(target_bpw) or target_bpw <= 0:
                raise ValueError("NEPQ1-A target_bpw must be finite and positive")
            target_nbytes = int(round(float(target_bpw) * int(np.prod(shape)) / 8.0))
        if target_nbytes is not None:
            second_limit, fixed_nbytes = _second_capacity(
                spec, base_nbytes, block_count, int(target_nbytes)
            )
        else:
            second_limit = int(second_records)
            if second_limit < 0 or second_limit > block_count:
                raise ValueError(f"NEPQ1-A second_records must be in [0,{block_count}]")
            fixed_nbytes = 0

    signs = torch.as_tensor(
        rotation_signs(width, rotation_block, rotation_seed),
        device=device,
        dtype=torch.float32,
    )
    dictionary = torch.as_tensor(
        dictionary_np.astype(np.float32), device=device, dtype=torch.float32
    )
    dictionary_norm = dictionary.square().sum(1)
    decoded_tables = _decoded_tables(base)
    first_all = np.empty((rows, blocks_per_row), dtype=np.uint16)
    second_all = np.empty((rows, blocks_per_row), dtype=np.uint16) if spec.residual_second else None
    gain_all = np.empty((rows, blocks_per_row), dtype=np.float32) if spec.residual_second else None
    for start in range(0, rows, config.residual_row_chunk):
        stop = min(start + config.residual_row_chunk, rows)
        value = _slice_rows(weight, start, stop, width, device)
        value = _fwht_blocks(value * signs, rotation_block)
        reconstruction = torch.as_tensor(
            dequantize_nepq_rows(
                base,
                start,
                stop,
                validate=False,
                decoded_tables=decoded_tables,
            ),
            device=device,
            dtype=torch.float32,
        )
        first, second, second_gain = _residual_records(
            value - reconstruction,
            spec,
            dictionary,
            dictionary_norm,
            config.residual_block_chunk,
        )
        first_all[start:stop] = first.reshape(stop - start, blocks_per_row).cpu().numpy()
        if second_all is not None and gain_all is not None:
            second_all[start:stop] = second.reshape(stop - start, blocks_per_row).cpu().numpy()
            gain_all[start:stop] = second_gain.reshape(stop - start, blocks_per_row).cpu().numpy()
        if progress is not None:
            progress(stop, rows)

    second_mask = None
    compact_second = None
    padding_nbytes = 0
    if spec.residual_second:
        assert second_all is not None and gain_all is not None
        flat_gain = gain_all.reshape(-1)
        selected_count = int(second_limit)
        selected = (
            np.argpartition(flat_gain, -selected_count)[-selected_count:]
            if 0 < selected_count < flat_gain.size
            else np.arange(flat_gain.size, dtype=np.int64)[:selected_count]
        )
        second_mask = np.zeros(block_count, dtype=np.uint8)
        second_mask[selected] = 1
        flat_second = second_all.reshape(-1)
        compact_second = flat_second[np.flatnonzero(second_mask)].astype(np.uint16, copy=False)
        if target_nbytes is not None:
            used = fixed_nbytes + math.ceil(compact_second.size * spec.residual_record_bits / 8)
            padding_nbytes = int(target_nbytes) - used
            if padding_nbytes < 0:
                raise AssertionError("NEPQ1-A residual budget accounting underflow")
        second_mask = second_mask.reshape(shape[0], shape[1], blocks_per_row)

    base.spec = spec
    base.residual_codebook = dictionary_np
    base.residual_first = first_all.reshape(shape[0], shape[1], blocks_per_row)
    base.residual_second_mask = second_mask
    base.residual_second_records = compact_second
    base.residual_padding_nbytes = padding_nbytes
    validate_nepq(base)
    if target_nbytes is not None and len(pack_nepq(base)) != int(target_nbytes):
        raise AssertionError("NEPQ1-A serialized size does not match its target")
    return base


__all__ = [
    "NepqAArtifact",
    "NepqAQuantConfig",
    "quantize_nepq_a_fixed",
]
