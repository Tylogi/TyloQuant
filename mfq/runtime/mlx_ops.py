"""Reusable MLX Transformer runtime primitives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.kernels.metal.ops import residual_rms_norm, rope, rope_tables


class MlxRMSNorm:
    """RMSNorm module with optional Qwen3.5-style weight offset."""

    def __init__(
        self,
        weight: mx.array | np.ndarray,
        eps: float = 1e-6,
        *,
        weight_offset: float = 0.0,
    ) -> None:
        scale = weight if isinstance(weight, mx.array) else mx.array(weight)
        if scale.ndim != 1:
            raise ValueError(f"RMSNorm weight must be one-dimensional, got {scale.shape}")
        self.weight = mx.contiguous(scale.astype(mx.float32))
        self.eps = float(eps)
        self.weight_offset = float(weight_offset)
        self._effective_weight = (
            self.weight
            if self.weight_offset == 0.0
            else mx.contiguous(self.weight + self.weight_offset)
        )
        mx.eval(self._effective_weight)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        source = x if isinstance(x, mx.array) else mx.array(x)
        if source.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            source = source.astype(mx.float16)
        normalized = mx.fast.rms_norm(source, self._effective_weight, self.eps)
        return normalized if normalized.dtype == source.dtype else normalized.astype(source.dtype)

    def add_and_forward(
        self,
        residual: mx.array | np.ndarray,
        update: mx.array | np.ndarray,
        *,
        normalized_dtype: mx.Dtype | None = None,
    ) -> tuple[mx.array, mx.array]:
        return residual_rms_norm(
            residual,
            update,
            self.weight,
            self.eps,
            weight_offset=self.weight_offset,
            normalized_dtype=normalized_dtype,
        )

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


class MlxRoPE:
    """Cached rotate-half RoPE module for full, partial, or multimodal RoPE."""

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        *,
        base: float = 1_000_000.0,
        sections: Sequence[int] | None = None,
        frequency_dim: int | None = None,
        active_pairs: int | None = None,
        mrope_interleaved: bool = False,
    ) -> None:
        self.rotary_dim = int(rotary_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.base = float(base)
        self.sections = None if sections is None else tuple(int(value) for value in sections)
        self.frequency_dim = None if frequency_dim is None else int(frequency_dim)
        self.active_pairs = None if active_pairs is None else int(active_pairs)
        self.mrope_interleaved = bool(mrope_interleaved)
        rope_tables(
            self.base,
            self.rotary_dim,
            self.max_position_embeddings,
            frequency_dim=self.frequency_dim,
            active_pairs=self.active_pairs,
        )

    def forward(
        self,
        x: mx.array | np.ndarray,
        positions: mx.array | np.ndarray,
        *,
        sequence_axis: int = -2,
    ) -> mx.array:
        return rope(
            x,
            positions,
            base=self.base,
            rotary_dim=self.rotary_dim,
            sections=self.sections,
            table_len=self.max_position_embeddings,
            sequence_axis=sequence_axis,
            frequency_dim=self.frequency_dim,
            active_pairs=self.active_pairs,
            mrope_interleaved=self.mrope_interleaved,
        )

    def forward_contiguous(
        self,
        x: mx.array | np.ndarray,
        offset: int,
        *,
        sequence_axis: int = -2,
    ) -> mx.array:
        """Apply the MLX fused RoPE path for ordinary contiguous text positions."""

        if int(offset) < 0:
            raise ValueError("RoPE offset cannot be negative")
        source = x if isinstance(x, mx.array) else mx.array(x)
        if source.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            source = source.astype(mx.float16)
        axis = int(sequence_axis) % source.ndim
        if axis == source.ndim - 1:
            raise ValueError("RoPE sequence axis cannot be the final feature axis")
        moved = axis != source.ndim - 2
        canonical = mx.moveaxis(source, axis, -2) if moved else source
        if self.frequency_dim is not None or self.active_pairs is not None:
            tokens = int(canonical.shape[-2])
            positions = mx.arange(offset, offset + tokens, dtype=mx.int32)
            output = self.forward(canonical, positions)
        else:
            output = mx.fast.rope(
                canonical,
                self.rotary_dim,
                traditional=False,
                base=self.base,
                scale=1.0,
                offset=int(offset),
            )
        return mx.moveaxis(output, -2, axis) if moved else output

    def __call__(
        self,
        x: mx.array | np.ndarray,
        positions: mx.array | np.ndarray,
    ) -> mx.array:
        return self.forward(x, positions)


__all__ = ["MlxRMSNorm", "MlxRoPE"]
