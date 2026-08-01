"""Apple-silicon attention and KV-cache tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.runtime.mlx_attention import (  # noqa: E402
    MlxKVCache,
    MlxSlidingWindowKVCache,
    attention,
    sliding_window_attention,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _reference_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    causal: bool,
) -> np.ndarray:
    repeat = q.shape[1] // k.shape[1]
    key = np.repeat(k, repeat, axis=1)
    value = np.repeat(v, repeat, axis=1)
    scores = np.einsum("bhtd,bhsd->bhts", q, key) / math.sqrt(q.shape[-1])
    if causal:
        offset = k.shape[2] - q.shape[2]
        query = np.arange(q.shape[2])[:, None]
        keys = np.arange(k.shape[2])[None, :]
        scores = np.where(keys <= query + offset, scores, -np.inf)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.einsum("bhts,bhsd->bhtd", probs, value)


@pytest.mark.parametrize("causal", [False, True])
def test_mlx_attention_matches_numpy_gqa(causal: bool):
    rng = np.random.default_rng(10 + causal)
    q = rng.normal(size=(2, 4, 3, 8)).astype(np.float16)
    k = rng.normal(size=(2, 2, 5, 8)).astype(np.float16)
    v = rng.normal(size=(2, 2, 5, 8)).astype(np.float16)
    actual = _array(attention(q, k, v, causal=causal))
    expected = _reference_attention(
        q.astype(np.float32),
        k.astype(np.float32),
        v.astype(np.float32),
        causal=causal,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_mlx_kv_cache_grows_and_supports_indexed_writes():
    cache = MlxKVCache(1, 2, 8, 4, initial_capacity=1, dtype=mx.float32)
    first_k = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    first_v = first_k + 100
    k, v = cache.append(first_k, first_v)
    np.testing.assert_array_equal(_array(k), first_k)
    np.testing.assert_array_equal(_array(v), first_v)
    assert cache.capacity == 4

    update_k = np.full((1, 2, 2, 4), -3, dtype=np.float32)
    update_v = np.full((1, 2, 2, 4), 7, dtype=np.float32)
    k, v = cache.append(update_k, update_v, np.array([4, 3], dtype=np.int32))
    expected_k = np.zeros((1, 2, 5, 4), dtype=np.float32)
    expected_v = np.zeros((1, 2, 5, 4), dtype=np.float32)
    expected_k[:, :, :3] = first_k
    expected_v[:, :, :3] = first_v
    expected_k[:, :, [4, 3]] = update_k
    expected_v[:, :, [4, 3]] = update_v
    np.testing.assert_array_equal(_array(k), expected_k)
    np.testing.assert_array_equal(_array(v), expected_v)


def test_sliding_window_cache_restores_chronological_order():
    cache = MlxSlidingWindowKVCache(1, 1, 4, 2, dtype=mx.float32)
    values = np.arange(12, dtype=np.float32).reshape(1, 1, 6, 2)
    cache.append(values[:, :, :3], values[:, :, :3] + 100)
    k, v = cache.append(values[:, :, 3:], values[:, :, 3:] + 100)
    np.testing.assert_array_equal(_array(k), values[:, :, -4:])
    np.testing.assert_array_equal(_array(v), values[:, :, -4:] + 100)


def test_sliding_window_attention_matches_explicit_slice():
    rng = np.random.default_rng(30)
    q = rng.normal(size=(1, 4, 2, 8)).astype(np.float16)
    k = rng.normal(size=(1, 2, 7, 8)).astype(np.float16)
    v = rng.normal(size=(1, 2, 7, 8)).astype(np.float16)
    actual = _array(sliding_window_attention(q, k, v, 4))
    expected = _array(attention(q, k[:, :, -4:], v[:, :, -4:], causal=True))
    np.testing.assert_array_equal(actual, expected)
