"""Numerical tests for DeepSeek-V4 Metal kernels."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.kernels.metal.deepseek_v4 import (  # noqa: E402
    attention_dsv4_sparse,
    dsv4_build_decode_plan,
    dsv4_build_prefill_plan,
    dsv4_compress,
    dsv4_decode_pool_step,
    dsv4_decode_pool_update,
    dsv4_fp4_sim,
    dsv4_indexer_scores,
    dsv4_indexer_scores_decode,
    dsv4_topk512,
)
from mfq.kernels.metal.deepseek_v4_hc import (  # noqa: E402
    dsv4_hc_post,
    dsv4_hc_pre,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _pow2_ceil(value: float) -> float:
    return float(2.0 ** math.ceil(math.log2(value)))


def _fp4(value: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(value / scale, -6.0, 6.0)
    magnitude = np.abs(normalized)
    quantized = np.select(
        [
            magnitude <= 0.25,
            magnitude < 0.75,
            magnitude <= 1.25,
            magnitude < 1.75,
            magnitude <= 2.5,
            magnitude < 3.5,
            magnitude <= 5.0,
        ],
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        default=6.0,
    )
    return np.copysign(quantized * scale, normalized)


def test_dsv4_fp4_sim_matches_group_reference():
    rng = np.random.default_rng(801)
    source = rng.normal(size=(3, 64)).astype(np.float16)
    actual = _array(dsv4_fp4_sim(source))
    expected = np.empty_like(source)
    for row in range(3):
        for start in (0, 32):
            group = source[row, start : start + 32].astype(np.float32)
            scale = _pow2_ceil(max(float(np.max(np.abs(group))) / 6.0, 2**-126))
            expected[row, start : start + 32] = _fp4(group, scale)
    np.testing.assert_array_equal(actual, expected)


def _bf16_round(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)


def test_dsv4_compress_quant0_matches_numpy():
    rng = np.random.default_rng(805)
    kv = rng.normal(size=(1, 2, 4, 128)).astype(np.float16)
    gate = rng.normal(size=kv.shape).astype(np.float16)
    ape = rng.normal(size=(4, 128)).astype(np.float32)
    norm = rng.normal(size=(128,)).astype(np.float32)
    positions = np.array([2, 4], dtype=np.int32)
    angles = rng.normal(size=(6, 32)).astype(np.float32)
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)
    actual = _array(
        dsv4_compress(
            kv,
            gate,
            ape,
            norm,
            None,
            None,
            positions,
            cosine,
            sine,
            4,
            False,
            0,
            1e-6,
        )
    )
    scores = gate.astype(np.float32) + ape[None, None]
    scores -= scores.max(axis=2, keepdims=True)
    weights = np.exp(scores)
    values = np.sum(weights * kv.astype(np.float32), axis=2) / np.sum(weights, axis=2)
    inverse = 1.0 / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + 1e-6)
    expected = _bf16_round(values * inverse * norm)
    for row, position in enumerate(positions):
        for pair in range(32):
            first = expected[0, row, 64 + pair * 2]
            second = expected[0, row, 64 + pair * 2 + 1]
            expected[0, row, 64 + pair * 2] = _bf16_round(
                first * cosine[position, pair] - second * sine[position, pair]
            )
            expected[0, row, 64 + pair * 2 + 1] = _bf16_round(
                second * cosine[position, pair] + first * sine[position, pair]
            )
    expected = _bf16_round(expected).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_dsv4_compress_overlap_fp4_and_fp8_paths_compile():
    rng = np.random.default_rng(806)
    ratio = 2
    angles = rng.normal(size=(4, 32)).astype(np.float32)
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)
    kv128 = rng.normal(size=(1, 1, ratio, 256)).astype(np.float16)
    gate128 = rng.normal(size=kv128.shape).astype(np.float16)
    output128 = _array(
        dsv4_compress(
            kv128,
            gate128,
            rng.normal(size=(ratio, 256)).astype(np.float32),
            np.ones((128,), dtype=np.float32),
            rng.normal(size=(1, ratio, 128)).astype(np.float16),
            rng.normal(size=(1, ratio, 128)).astype(np.float16),
            np.array([1], dtype=np.int32),
            cosine,
            sine,
            ratio,
            True,
            2,
        )
    )
    assert output128.shape == (1, 1, 128)
    assert np.isfinite(output128).all()

    kv512 = rng.normal(size=(1, 1, 1, 512)).astype(np.float16)
    output512 = _array(
        dsv4_compress(
            kv512,
            np.zeros_like(kv512),
            np.zeros((1, 512), dtype=np.float32),
            np.ones((512,), dtype=np.float32),
            None,
            None,
            np.array([0], dtype=np.int32),
            cosine,
            sine,
            1,
            False,
            1,
        )
    )
    assert output512.shape == (1, 1, 512)
    assert np.isfinite(output512).all()


def test_dsv4_decode_pool_update_returns_functional_state():
    rng = np.random.default_rng(807)
    ratio = 2
    dimension = 128
    state_kv = rng.normal(size=(1, ratio, dimension)).astype(np.float16)
    state_gate = rng.normal(size=(1, ratio, dimension)).astype(np.float16)
    token_kv = rng.normal(size=(1, 1, dimension)).astype(np.float16)
    token_gate = rng.normal(size=(1, 1, dimension)).astype(np.float16)
    ape = rng.normal(size=(ratio, dimension)).astype(np.float32)
    norm = rng.normal(size=(dimension,)).astype(np.float32)
    pool = rng.normal(size=(1, 4, dimension)).astype(np.float16)
    angles = rng.normal(size=(4, 32)).astype(np.float32)
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)

    first = dsv4_decode_pool_update(
        token_kv,
        token_gate,
        ape,
        norm,
        state_kv,
        state_gate,
        None,
        None,
        pool,
        np.array([1], dtype=np.int32),
        cosine,
        sine,
        ratio,
        False,
    )
    expected_state_kv = state_kv.copy()
    expected_state_gate = state_gate.copy()
    expected_state_kv[:, 0] = token_kv[:, 0]
    expected_state_gate[:, 0] = token_gate[:, 0]
    np.testing.assert_array_equal(_array(first.state_kv), expected_state_kv)
    np.testing.assert_array_equal(_array(first.state_gate), expected_state_gate)
    np.testing.assert_array_equal(_array(first.pool), pool)
    assert first.prev_kv is None
    assert first.prev_gate is None

    second = dsv4_decode_pool_update(
        token_kv,
        token_gate,
        ape,
        norm,
        first.state_kv,
        first.state_gate,
        None,
        None,
        first.pool,
        np.array([2], dtype=np.int32),
        cosine,
        sine,
        ratio,
        False,
    )
    expected_state_kv[:, 1] = token_kv[:, 0]
    expected_state_gate[:, 1] = token_gate[:, 0]
    expected_row = _array(
        dsv4_compress(
            expected_state_kv[:, None],
            expected_state_gate[:, None],
            ape,
            norm,
            None,
            None,
            np.array([0], dtype=np.int32),
            cosine,
            sine,
            ratio,
            False,
        )
    )
    expected_pool = pool.copy()
    expected_pool[:, 0] = expected_row[:, 0]
    np.testing.assert_array_equal(_array(second.state_kv), expected_state_kv)
    np.testing.assert_array_equal(_array(second.state_gate), expected_state_gate)
    np.testing.assert_allclose(_array(second.pool), expected_pool, rtol=2e-3, atol=2e-3)


def test_dsv4_decode_pool_step_emits_only_boundary_delta():
    rng = np.random.default_rng(817)
    ratio = 2
    dimension = 128
    state_kv = rng.normal(size=(1, ratio, dimension)).astype(np.float16)
    state_gate = rng.normal(size=(1, ratio, dimension)).astype(np.float16)
    token_kv = rng.normal(size=(1, 1, dimension)).astype(np.float16)
    token_gate = rng.normal(size=(1, 1, dimension)).astype(np.float16)
    ape = rng.normal(size=(ratio, dimension)).astype(np.float32)
    norm = rng.normal(size=(dimension,)).astype(np.float32)
    angles = rng.normal(size=(4, 32)).astype(np.float32)
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)

    first = dsv4_decode_pool_step(
        token_kv,
        token_gate,
        ape,
        norm,
        state_kv,
        state_gate,
        None,
        None,
        np.array([1], dtype=np.int32),
        cosine,
        sine,
        ratio,
        False,
    )
    np.testing.assert_array_equal(_array(first.emit_rows), np.array([-1]))
    np.testing.assert_array_equal(
        _array(first.emitted), np.zeros((1, 1, dimension), dtype=np.float16)
    )

    second = dsv4_decode_pool_step(
        token_kv,
        token_gate,
        ape,
        norm,
        first.state_kv,
        first.state_gate,
        None,
        None,
        np.array([2], dtype=np.int32),
        cosine,
        sine,
        ratio,
        False,
    )
    np.testing.assert_array_equal(_array(second.emit_rows), np.array([0]))
    assert second.emitted.shape == (1, 1, dimension)
    assert np.isfinite(_array(second.emitted)).all()


def test_dsv4_indexer_scores_matches_numpy_and_visibility():
    rng = np.random.default_rng(802)
    q = rng.normal(size=(1, 3, 64, 128)).astype(np.float16)
    k = rng.normal(size=(1, 70, 128)).astype(np.float16)
    weights = rng.normal(size=(1, 3, 64)).astype(np.float16)
    actual = _array(dsv4_indexer_scores(q, k, weights, 140, 2))
    dots = np.einsum(
        "bmhd,bkd->bmhk",
        q.astype(np.float32),
        k.astype(np.float32),
    )
    expected = np.einsum(
        "bmhk,bmh->bmk",
        np.maximum(dots, 0.0),
        weights.astype(np.float32),
    ) / math.sqrt(128.0 * 64.0)
    for query in range(3):
        visible = min(70, (140 + query + 1) // 2)
        expected[:, query, visible:] = -np.inf
    np.testing.assert_allclose(actual, expected, rtol=4e-3, atol=4e-3)


def test_dsv4_decode_indexer_stream_matches_numpy():
    rng = np.random.default_rng(818)
    q = rng.normal(size=(2, 1, 64, 128)).astype(np.float16)
    k = rng.normal(size=(2, 513, 128)).astype(np.float16)
    weights = rng.normal(size=(2, 1, 64)).astype(np.float16)
    actual = _array(dsv4_indexer_scores_decode(q, k, weights, 900, 2))
    dots = np.einsum(
        "bmhd,bkd->bmhk",
        q.astype(np.float32),
        k.astype(np.float32),
    )
    expected = np.einsum(
        "bmhk,bmh->bmk",
        np.maximum(dots, 0.0),
        weights.astype(np.float32),
    ) / math.sqrt(128.0 * 64.0)
    expected[:, :, 450:] = -np.inf
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)
    auto = _array(dsv4_indexer_scores(q, k, weights, 900, 2))
    np.testing.assert_array_equal(auto, actual)


def test_dsv4_topk512_matches_exact_monotonic_set():
    scores = np.arange(700, dtype=np.float16).reshape(1, 1, 700)
    actual = np.sort(_array(dsv4_topk512(scores))[0, 0])
    np.testing.assert_array_equal(actual, np.arange(188, 700, dtype=np.int32))


def test_dsv4_topk512_pads_short_rows_like_cuda():
    scores = np.arange(17, dtype=np.float16).reshape(1, 1, 17)
    actual = _array(dsv4_topk512(scores))[0, 0]
    np.testing.assert_array_equal(actual[:17], np.arange(17, dtype=np.int32))
    np.testing.assert_array_equal(actual[17:], np.zeros(495, dtype=np.int32))


def test_dsv4_topk512_deterministic_ties_choose_lowest_indices():
    scores = np.ones((1, 1, 700), dtype=np.float16)
    first = _array(dsv4_topk512(scores, deterministic=True))[0, 0]
    second = _array(dsv4_topk512(scores, deterministic=True))[0, 0]
    np.testing.assert_array_equal(first, np.arange(512, dtype=np.int32))
    np.testing.assert_array_equal(second, first)
    bucketed = _array(dsv4_topk512(scores, deterministic=False))[0, 0]
    assert len(np.unique(bucketed)) == 512


def test_dsv4_prefill_plan_matches_cuda_layout():
    topk = np.array([[[0, 3, 8], [1, 4, -1]]], dtype=np.int32)
    indices, mask = dsv4_build_prefill_plan(
        topk,
        query_offset=12,
        local_history=3,
        pool_len=6,
        ratio=2,
        window=4,
    )
    actual_indices = _array(indices)
    actual_mask = _array(mask)
    selected = 32
    expected_indices = np.zeros((1, 2, selected), dtype=np.int32)
    expected_mask = np.full((1, 2, selected), -np.inf, dtype=np.float16)
    for query in range(2):
        local_end = 3 + query + 1
        local_count = min(4, local_end)
        for slot in range(local_count):
            expected_indices[0, query, slot] = local_end - local_count + slot
            expected_mask[0, query, slot] = 0
        visible = min(6, (12 + query + 1) // 2)
        for offset, pooled in enumerate(topk[0, query]):
            if 0 <= pooled < visible:
                expected_indices[0, query, 4 + offset] = 5 + pooled
                expected_mask[0, query, 4 + offset] = 0
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(actual_mask, expected_mask)


def test_dsv4_decode_plan_matches_circular_layout():
    topk = np.array([[[0, 2]], [[1, 5]]], dtype=np.int32)
    lengths = np.array([3, 11], dtype=np.int32)
    indices, mask = dsv4_build_decode_plan(topk, lengths, 4, 2, 5)
    actual_indices = _array(indices)
    actual_mask = _array(mask)
    expected_indices = np.zeros_like(actual_indices)
    expected_mask = np.full_like(actual_mask, -np.inf)
    for batch, length in enumerate(lengths):
        local_count = min(int(length), 5)
        for slot in range(local_count):
            expected_indices[batch, 0, slot] = (length - local_count + slot) % 5
            expected_mask[batch, 0, slot] = 0
        visible = min(int(length) // 2, 4)
        for offset, pooled in enumerate(topk[batch, 0]):
            if 0 <= pooled < visible:
                expected_indices[batch, 0, 5 + offset] = 5 + pooled
                expected_mask[batch, 0, 5 + offset] = 0
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(actual_mask, expected_mask)


def test_dsv4_sparse_attention_matches_numpy_with_sinks():
    rng = np.random.default_rng(803)
    q = rng.normal(size=(1, 64, 2, 512)).astype(np.float32)
    kv = rng.normal(size=(1, 40, 512)).astype(np.float16)
    indices = np.stack(
        [
            rng.choice(40, size=32, replace=False),
            rng.choice(40, size=32, replace=False),
        ],
        axis=0,
    )[None].astype(np.int32)
    mask = np.zeros((1, 2, 32), dtype=np.float16)
    mask[:, :, -5:] = -np.inf
    sinks = rng.normal(size=(64,)).astype(np.float32)
    scale = 0.06
    actual = _array(attention_dsv4_sparse(q, kv, indices, mask, sinks, scale=scale))
    expected = np.empty((1, 2, 64, 512), dtype=np.float32)
    for query in range(2):
        selected = kv[0, indices[0, query]].astype(np.float32)
        scores = q[0, :, query] @ selected.T * scale
        scores[:, -5:] = -np.inf
        maximum = np.maximum(np.max(scores, axis=-1), sinks)
        exponentials = np.exp(scores - maximum[:, None])
        denominator = np.exp(sinks - maximum) + exponentials.sum(axis=-1)
        expected[0, query] = exponentials @ selected / denominator[:, None]
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)


def test_dsv4_sparse_attention_decode_four_head_groups_match_numpy():
    rng = np.random.default_rng(819)
    q = rng.normal(size=(1, 64, 1, 512)).astype(np.float32)
    kv = rng.normal(size=(1, 96, 512)).astype(np.float16)
    indices = rng.choice(96, size=(1, 1, 64), replace=False).astype(np.int32)
    mask = np.zeros((1, 1, 64), dtype=np.float16)
    mask[:, :, -7:] = -np.inf
    sinks = rng.normal(size=(64,)).astype(np.float32)
    scale = 0.05
    actual = _array(attention_dsv4_sparse(q, kv, indices, mask, sinks, scale=scale))
    selected = kv[0, indices[0, 0]].astype(np.float32)
    scores = q[0, :, 0] @ selected.T * scale
    scores[:, -7:] = -np.inf
    maximum = np.maximum(np.max(scores, axis=-1), sinks)
    exponentials = np.exp(scores - maximum[:, None])
    denominator = np.exp(sinks - maximum) + exponentials.sum(axis=-1)
    expected = exponentials @ selected / denominator[:, None]
    np.testing.assert_allclose(
        actual[0, 0],
        expected,
        rtol=5e-3,
        atol=5e-3,
    )


def test_dsv4_sparse_attention_prefill_mma_crossover_matches_numpy():
    rng = np.random.default_rng(820)
    queries = 32
    q = rng.normal(size=(1, 64, queries, 512)).astype(np.float16)
    kv = rng.normal(size=(1, 48, 512)).astype(np.float16)
    one_selection = rng.choice(48, size=32, replace=False).astype(np.int32)
    indices = np.broadcast_to(
        one_selection[None, None],
        (1, queries, 32),
    ).copy()
    mask = np.zeros((1, queries, 32), dtype=np.float16)
    mask[:, :, -3:] = -np.inf
    sinks = rng.normal(size=(64,)).astype(np.float32)
    scale = 0.04
    actual = _array(attention_dsv4_sparse(q, kv, indices, mask, sinks, scale=scale))
    selected = kv[0, one_selection].astype(np.float32)
    for query_index, head in ((0, 0), (0, 63), (31, 0), (31, 63)):
        scores = q[0, head, query_index].astype(np.float32) @ selected.T * scale
        scores[-3:] = -np.inf
        maximum = max(float(np.max(scores)), float(sinks[head]))
        exponentials = np.exp(scores - maximum)
        denominator = np.exp(sinks[head] - maximum) + exponentials.sum()
        expected = exponentials @ selected / denominator
        np.testing.assert_allclose(
            actual[0, query_index, head],
            expected,
            rtol=6e-3,
            atol=6e-3,
        )


def _hc_reference(
    x: np.ndarray,
    mixes: np.ndarray,
    scale: np.ndarray,
    base: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = 1.0 / (1.0 + np.exp(-(mixes[..., :4] * scale[0] + base[:4]))) + eps
    post = 2.0 / (1.0 + np.exp(-(mixes[..., 4:8] * scale[1] + base[4:8])))
    combination = mixes[..., 8:].reshape(*mixes.shape[:2], 4, 4) * scale[2] + base[8:].reshape(4, 4)
    maximum = combination.max(axis=-1, keepdims=True)
    combination = np.exp(combination - maximum)
    combination = combination / combination.sum(axis=-1, keepdims=True) + eps
    combination /= combination.sum(axis=-2, keepdims=True) + eps
    for _ in range(1, 20):
        combination /= combination.sum(axis=-1, keepdims=True) + eps
        combination /= combination.sum(axis=-2, keepdims=True) + eps
    reduced = np.einsum("bts,btsd->btd", pre, x.astype(np.float32))
    return reduced, post, combination


def test_dsv4_hc_pre_and_post_match_numpy():
    rng = np.random.default_rng(804)
    x = rng.normal(size=(1, 2, 4, 4096)).astype(np.float16)
    mixes = rng.normal(size=(1, 2, 24)).astype(np.float32)
    scale = rng.normal(size=(3,)).astype(np.float32)
    base = rng.normal(size=(24,)).astype(np.float32)
    eps = 1e-6
    reduced, post, combination = dsv4_hc_pre(x, mixes, scale, base, 20, eps)
    expected_reduced, expected_post, expected_combination = _hc_reference(
        x, mixes, scale, base, eps
    )
    np.testing.assert_allclose(_array(reduced), expected_reduced, rtol=4e-3, atol=4e-3)
    np.testing.assert_allclose(_array(post), expected_post, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        _array(combination),
        expected_combination,
        rtol=2e-5,
        atol=2e-5,
    )

    transformed = rng.normal(size=(1, 2, 4096)).astype(np.float16)
    actual = _array(dsv4_hc_post(transformed, x, post, combination))
    expected = expected_post[..., :, None] * transformed.astype(np.float32)[
        ..., None, :
    ] + np.einsum("btsd,btsc->btcd", x.astype(np.float32), expected_combination)
    np.testing.assert_allclose(actual, expected, rtol=4e-3, atol=4e-3)
