"""MLX attention and KV-cache primitives for Apple silicon."""

from __future__ import annotations

import math

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


def _floating(value: mx.array | np.ndarray, dtype: mx.Dtype | None = None) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if dtype is not None:
        return mx.contiguous(result.astype(dtype))
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return mx.contiguous(result)


def attention(
    q: mx.array | np.ndarray,
    k: mx.array | np.ndarray,
    v: mx.array | np.ndarray,
    *,
    causal: bool = True,
    scale: float | None = None,
    mask: mx.array | np.ndarray | None = None,
) -> mx.array:
    """Run MHA, GQA, or MQA for ``[B,H,T,D]`` inputs."""

    query = _floating(q)
    key = _floating(k, query.dtype)
    value = _floating(v, query.dtype)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention inputs must have [B,H,T,D] shape")
    if key.shape != value.shape:
        raise ValueError(f"attention key/value shapes differ: {key.shape} and {value.shape}")
    if (
        int(query.shape[0]) != int(key.shape[0])
        or int(query.shape[-1]) != int(key.shape[-1])
        or int(query.shape[1]) % int(key.shape[1])
    ):
        raise ValueError("attention batch/head dimensions are incompatible")
    if scale is None:
        scale = 1.0 / math.sqrt(int(query.shape[-1]))
    if mask is not None and causal:
        raise ValueError("pass either causal=True or an explicit attention mask")
    selected_mask: str | mx.array | None
    if mask is not None:
        selected_mask = mask if isinstance(mask, mx.array) else mx.array(mask)
        if selected_mask.dtype != mx.bool_:
            selected_mask = selected_mask.astype(query.dtype)
    else:
        selected_mask = "causal" if causal else None
    return mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=float(scale),
        mask=selected_mask,
    )


class MlxKVCache:
    """Dynamically growing FP16/FP32 cache with contiguous or indexed writes."""

    def __init__(
        self,
        batch: int,
        heads: int,
        max_seq: int,
        head_dim: int,
        *,
        initial_capacity: int = 16,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if min(batch, heads, max_seq, head_dim) <= 0:
            raise ValueError("KV cache dimensions must be positive")
        self.batch = int(batch)
        self.heads = int(heads)
        self.max_seq = int(max_seq)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        capacity = min(max(int(initial_capacity), 0), self.max_seq)
        shape = (self.batch, self.heads, capacity, self.head_dim)
        self.k = mx.zeros(shape, dtype=dtype)
        self.v = mx.zeros(shape, dtype=dtype)
        self.pos = 0

    @property
    def capacity(self) -> int:
        return int(self.k.shape[2])

    def reset(self) -> None:
        self.pos = 0

    def _ensure_capacity(self, required: int) -> None:
        required = int(required)
        if required <= self.capacity:
            return
        if required > self.max_seq:
            raise ValueError(f"KV cache position {required} exceeds max_seq {self.max_seq}")
        capacity = max(1, self.capacity)
        while capacity < required:
            capacity *= 2
        capacity = min(capacity, self.max_seq)
        shape = (self.batch, self.heads, capacity, self.head_dim)
        next_k = mx.zeros(shape, dtype=self.dtype)
        next_v = mx.zeros(shape, dtype=self.dtype)
        if self.capacity:
            start = mx.array([0, 0, 0, 0], dtype=mx.int32)
            next_k = mx.slice_update(next_k, self.k, start_indices=start, axes=(0, 1, 2, 3))
            next_v = mx.slice_update(next_v, self.v, start_indices=start, axes=(0, 1, 2, 3))
        self.k = next_k
        self.v = next_v

    def append(
        self,
        k: mx.array | np.ndarray,
        v: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
    ) -> tuple[mx.array, mx.array]:
        key = _floating(k, self.dtype)
        value = _floating(v, self.dtype)
        expected_prefix = (self.batch, self.heads)
        if (
            key.ndim != 4
            or key.shape != value.shape
            or tuple(int(item) for item in key.shape[:2]) != expected_prefix
            or int(key.shape[3]) != self.head_dim
        ):
            raise ValueError("KV append expects matching [batch,heads,tokens,head_dim] arrays")
        tokens = int(key.shape[2])
        if positions is None:
            indices = np.arange(self.pos, self.pos + tokens, dtype=np.int32)
        else:
            raw = positions if isinstance(positions, mx.array) else mx.array(positions)
            mx.eval(raw)
            indices = np.asarray(raw, dtype=np.int32).reshape(-1)
            if indices.size != tokens:
                raise ValueError("KV cache positions must contain one index per token")
            if np.any(indices < 0):
                raise ValueError("KV cache positions cannot be negative")
        new_pos = max(self.pos, int(indices.max(initial=-1)) + 1)
        self._ensure_capacity(new_pos)

        contiguous = bool(
            tokens == 0 or np.array_equal(indices, np.arange(indices[0], indices[0] + tokens))
        )
        if contiguous and tokens:
            start = mx.array([0, 0, int(indices[0]), 0], dtype=mx.int32)
            self.k = mx.slice_update(self.k, key, start_indices=start, axes=(0, 1, 2, 3))
            self.v = mx.slice_update(self.v, value, start_indices=start, axes=(0, 1, 2, 3))
        else:
            for token, position in enumerate(indices.tolist()):
                start = mx.array([0, 0, position, 0], dtype=mx.int32)
                self.k = mx.slice_update(
                    self.k,
                    key[:, :, token : token + 1, :],
                    start_indices=start,
                    axes=(0, 1, 2, 3),
                )
                self.v = mx.slice_update(
                    self.v,
                    value[:, :, token : token + 1, :],
                    start_indices=start,
                    axes=(0, 1, 2, 3),
                )
        self.pos = new_pos
        return self.view()

    def view(self, upto: int | None = None) -> tuple[mx.array, mx.array]:
        end = self.pos if upto is None else int(upto)
        if not 0 <= end <= self.pos:
            raise ValueError(f"KV cache view endpoint {end} is outside [0,{self.pos}]")
        return self.k[:, :, :end, :], self.v[:, :, :end, :]


class MlxSlidingWindowKVCache:
    """Fixed-size circular KV cache returned in chronological order."""

    def __init__(
        self,
        batch: int,
        heads: int,
        window: int,
        head_dim: int,
        *,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if min(batch, heads, window, head_dim) <= 0:
            raise ValueError("sliding-window cache dimensions must be positive")
        self.batch = int(batch)
        self.heads = int(heads)
        self.window = int(window)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        shape = (self.batch, self.heads, self.window, self.head_dim)
        self.k = mx.zeros(shape, dtype=dtype)
        self.v = mx.zeros(shape, dtype=dtype)
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def append(
        self,
        k: mx.array | np.ndarray,
        v: mx.array | np.ndarray,
    ) -> tuple[mx.array, mx.array]:
        key = _floating(k, self.dtype)
        value = _floating(v, self.dtype)
        if (
            key.ndim != 4
            or key.shape != value.shape
            or tuple(int(item) for item in key.shape[:2]) != (self.batch, self.heads)
            or int(key.shape[3]) != self.head_dim
        ):
            raise ValueError(
                "sliding KV append expects matching [batch,heads,tokens,head_dim] arrays"
            )
        for token in range(int(key.shape[2])):
            slot = (self.pos + token) % self.window
            start = mx.array([0, 0, slot, 0], dtype=mx.int32)
            self.k = mx.slice_update(
                self.k,
                key[:, :, token : token + 1, :],
                start_indices=start,
                axes=(0, 1, 2, 3),
            )
            self.v = mx.slice_update(
                self.v,
                value[:, :, token : token + 1, :],
                start_indices=start,
                axes=(0, 1, 2, 3),
            )
        self.pos += int(key.shape[2])
        return self.view()

    def view(self) -> tuple[mx.array, mx.array]:
        count = min(self.pos, self.window)
        if self.pos <= self.window:
            return self.k[:, :, :count, :], self.v[:, :, :count, :]
        start = self.pos % self.window
        return (
            mx.concatenate((self.k[:, :, start:, :], self.k[:, :, :start, :]), axis=2),
            mx.concatenate((self.v[:, :, start:, :], self.v[:, :, :start, :]), axis=2),
        )


def sliding_window_attention(
    q: mx.array | np.ndarray,
    k: mx.array | np.ndarray,
    v: mx.array | np.ndarray,
    window: int,
    *,
    scale: float | None = None,
) -> mx.array:
    """Run lower-right causal attention over at most the newest ``window`` keys."""

    if int(window) <= 0:
        raise ValueError("attention window must be positive")
    key = _floating(k)
    value = _floating(v, key.dtype)
    if int(key.shape[2]) > int(window):
        key = key[:, :, -int(window) :, :]
        value = value[:, :, -int(window) :, :]
    return attention(q, key, value, causal=True, scale=scale)


__all__ = [
    "MlxKVCache",
    "MlxSlidingWindowKVCache",
    "attention",
    "sliding_window_attention",
]
