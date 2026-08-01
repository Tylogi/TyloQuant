"""KV cache writer and small runtime cache helper."""

from __future__ import annotations

import torch

from mfq.kernels.cuda._ext import ext


def kv_cache_write(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write ``k/v [B,H,T,D]`` into ``cache [B,H,max_seq,D]`` at ``positions``."""

    return tuple(
        ext().kv_cache_write_cuda(
            k_cache,
            v_cache,
            k.contiguous().to(k_cache.dtype),
            v.contiguous().to(v_cache.dtype),
            positions.contiguous().to(device=k.device, dtype=torch.int64),
        )
    )


def kv_cache_write_ring(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append contiguous absolute positions to a circular cache."""

    return tuple(
        ext().kv_cache_write_ring_cuda(
            k_cache,
            v_cache,
            k.contiguous().to(k_cache.dtype),
            v.contiguous().to(v_cache.dtype),
            int(position_start),
        )
    )


class KVCache:
    """Dynamically growing f16/f32 KV cache using the CUDA writer."""

    def __init__(
        self,
        batch: int,
        heads: int,
        max_seq: int,
        head_dim: int,
        *,
        device: str | torch.device = "cuda",
        initial_capacity: int = 16,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.batch = int(batch)
        self.heads = int(heads)
        self.max_seq = int(max_seq)
        self.head_dim = int(head_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        cap = min(max(int(initial_capacity), 0), self.max_seq)
        self.k = torch.empty((self.batch, self.heads, cap, self.head_dim), device=self.device, dtype=self.dtype)
        self.v = torch.empty_like(self.k)
        self.pos = 0

    @property
    def capacity(self) -> int:
        return int(self.k.size(2))

    def _ensure_capacity(self, required: int) -> None:
        required = int(required)
        if required <= self.capacity:
            return
        if required > self.max_seq:
            raise ValueError(f"KV cache position {required} exceeds max_seq {self.max_seq}")
        new_cap = max(1, self.capacity)
        while new_cap < required:
            new_cap *= 2
        new_cap = min(new_cap, self.max_seq)
        k_new = torch.empty(
            (self.batch, self.heads, new_cap, self.head_dim), device=self.device, dtype=self.dtype
        )
        v_new = torch.empty_like(k_new)
        if self.capacity:
            k_new[:, :, : self.capacity, :].copy_(self.k)
            v_new[:, :, : self.capacity, :].copy_(self.v)
        self.k = k_new
        self.v = v_new

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = int(k.size(2))
        if positions is None:
            positions = torch.arange(self.pos, self.pos + T, device=k.device, dtype=torch.int64)
            new_pos = self.pos + T
        else:
            positions = positions.to(device=k.device, dtype=torch.int64).contiguous()
            new_pos = max(self.pos, int(positions.max().item()) + 1)
        self._ensure_capacity(new_pos)
        self.pos = new_pos
        kv_cache_write(self.k, self.v, k, v, positions)
        return self.view()

    def view(self, upto: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.pos if upto is None else int(upto)
        return self.k[:, :, :end, :], self.v[:, :, :end, :]


class SlidingWindowKVCache:
    """Fixed-size circular cache for standard causal SWA."""

    def __init__(
        self,
        batch: int,
        heads: int,
        window: int,
        head_dim: int,
        *,
        ubatch_capacity: int = 1,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if window <= 0 or ubatch_capacity <= 0:
            raise ValueError("window and ubatch_capacity must be positive")
        self.window = int(window)
        self.ubatch_capacity = int(ubatch_capacity)
        self.capacity = ((self.window + self.ubatch_capacity + 255) // 256) * 256
        self.k = torch.empty((batch, heads, self.capacity, head_dim), device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.pos = 0
        self.seq_len = torch.zeros(1, device=device, dtype=torch.int64)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = int(k.size(2))
        if tokens > self.ubatch_capacity:
            raise ValueError(
                f"append has {tokens} tokens, exceeding ubatch_capacity={self.ubatch_capacity}"
            )
        kv_cache_write_ring(self.k, self.v, k, v, self.pos)
        self.pos += tokens
        self.seq_len.fill_(self.pos)
        return self.k, self.v
