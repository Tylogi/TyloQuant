from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mfq.kernels.metal.flash_next import (  # noqa: E402
    glm5_dense_mla_attention,
    glm5_kda_forget_gate,
    glm5_kpool_scores,
    glm5_kpool_states,
    glm5_mhc_post,
    glm5_mhc_pre,
    glm5_sparse_mla_attention,
    qsa_block_scores,
    qwen4_dense_gqa_attention,
    qwen4_gated_residual_post,
    qwen4_gated_residual_pre,
    qwen4_ple_dilated_conv_silu,
    qwen4_sparse_gqa_attention,
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _softmax(value: np.ndarray, axis: int) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    output = np.exp(shifted)
    return output / np.sum(output, axis=axis, keepdims=True)


def test_qwen4_gated_residual_matches_reference() -> None:
    rng = np.random.default_rng(53)
    streams, hidden, lowrank = 4, 8, 5
    width = streams * hidden
    value = rng.normal(size=(1, 3, width)).astype(np.float32)
    norm = rng.normal(scale=0.1, size=(width,)).astype(np.float32)
    down = rng.normal(scale=0.1, size=(lowrank, width)).astype(np.float32)
    up = rng.normal(scale=0.1, size=(width, lowrank)).astype(np.float32)
    inject = rng.normal(scale=0.1, size=(streams, width)).astype(np.float32)

    mixed, residual, gates = qwen4_gated_residual_pre(
        mx.array(value),
        mx.array(norm),
        mx.array(down),
        mx.array(up),
        mx.array(inject),
        hidden_size=hidden,
        hc_count=streams,
    )
    mx.eval(mixed, gates)

    grouped = value.reshape(1, 3, streams, hidden)
    normalized = grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-6)
    normalized = normalized.reshape(value.shape) * (1.0 + norm)
    low = normalized @ down.T / streams
    low = low * _sigmoid(low)
    mixing = _sigmoid(low @ up.T).reshape(1, 3, streams, hidden)
    expected_mixed = np.mean(mixing * normalized.reshape(1, 3, streams, hidden), axis=-2)
    expected_gates = 2.0 * _sigmoid(normalized @ inject.T / streams)
    np.testing.assert_allclose(np.asarray(mixed), expected_mixed, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(gates), expected_gates, rtol=2e-4, atol=2e-4)

    branch = rng.normal(size=(1, 3, hidden)).astype(np.float32)
    output = qwen4_gated_residual_post(mx.array(branch), residual, gates, hc_count=streams)
    mx.eval(output)
    expected = value + (branch[..., None, :] * expected_gates[..., :, None]).reshape(value.shape)
    np.testing.assert_allclose(np.asarray(output), expected, rtol=2e-4, atol=2e-4)


def test_glm5_mhc_matches_reference_sinkhorn() -> None:
    rng = np.random.default_rng(54)
    streams, hidden = 4, 8
    mix_width = (2 + streams) * streams
    value = rng.normal(size=(1, 2, streams, hidden)).astype(np.float32)
    function = rng.normal(scale=0.08, size=(mix_width, streams * hidden)).astype(np.float32)
    base = rng.normal(scale=0.05, size=(mix_width,)).astype(np.float32)
    scale = np.asarray([0.8, 0.9, 1.1], dtype=np.float32)

    post, combination, collapsed = glm5_mhc_pre(
        mx.array(value),
        mx.array(function),
        mx.array(base),
        mx.array(scale),
        sinkhorn_iterations=5,
    )
    mx.eval(post, combination, collapsed)

    flat = value.reshape(1, 2, -1)
    flat = flat / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + 1e-5)
    logits = flat @ function.T
    pre_raw, post_raw, comb_raw = np.split(logits, [streams, 2 * streams], axis=-1)
    pre = _sigmoid(pre_raw * scale[0] + base[:streams]) + 1e-6
    expected_post = 2 * _sigmoid(post_raw * scale[1] + base[streams : 2 * streams])
    expected_combination = (
        _softmax(
            comb_raw.reshape(1, 2, streams, streams) * scale[2]
            + base[2 * streams :].reshape(streams, streams),
            -1,
        )
        + 1e-6
    )
    expected_combination /= expected_combination.sum(axis=-2, keepdims=True) + 1e-6
    for _ in range(4):
        expected_combination /= expected_combination.sum(axis=-1, keepdims=True) + 1e-6
        expected_combination /= expected_combination.sum(axis=-2, keepdims=True) + 1e-6
    expected_collapsed = np.sum(pre[..., None] * value, axis=-2)
    np.testing.assert_allclose(np.asarray(post), expected_post, rtol=7e-4, atol=7e-4)
    np.testing.assert_allclose(np.asarray(combination), expected_combination, rtol=7e-4, atol=7e-4)
    np.testing.assert_allclose(np.asarray(collapsed), expected_collapsed, rtol=7e-4, atol=7e-4)

    branch = rng.normal(size=(1, 2, hidden)).astype(np.float32)
    expanded = glm5_mhc_post(mx.array(branch), mx.array(value), post, combination)
    mx.eval(expanded)
    expected_expanded = expected_post[..., None] * branch[..., None, :] + np.matmul(
        np.swapaxes(expected_combination, -1, -2), value
    )
    np.testing.assert_allclose(np.asarray(expanded), expected_expanded, rtol=8e-4, atol=8e-4)


def test_glm5_kda_forget_gate_matches_safe_lower_bound() -> None:
    rng = np.random.default_rng(55)
    hidden, heads, dimension = 12, 3, 4
    value = rng.normal(size=(2, 5, hidden)).astype(np.float32)
    f_a = rng.normal(scale=0.2, size=(dimension, hidden)).astype(np.float32)
    f_b = rng.normal(scale=0.2, size=(heads * dimension, dimension)).astype(np.float32)
    bias = rng.normal(scale=0.1, size=(heads * dimension,)).astype(np.float32)
    a_log = rng.normal(scale=0.1, size=(heads,)).astype(np.float32)
    actual = glm5_kda_forget_gate(
        mx.array(value),
        mx.array(f_a),
        mx.array(f_b),
        mx.array(bias),
        mx.array(a_log),
        num_heads=heads,
        head_dim=dimension,
    )
    mx.eval(actual)
    gate = ((value @ f_a.T) @ f_b.T + bias).reshape(2, 5, heads, dimension)
    expected = -5.0 * _sigmoid(np.exp(a_log)[None, None, :, None] * gate)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-3, atol=2e-3)


def test_qsa_scores_and_glm_kpool_match_reference() -> None:
    rng = np.random.default_rng(56)
    query = rng.normal(size=(2, 3, 4, 8)).astype(np.float32)
    pooled = rng.normal(size=(2, 5, 8)).astype(np.float32)
    scores = qsa_block_scores(mx.array(query), mx.array(pooled))
    mx.eval(scores)
    expected_scores = np.maximum(np.einsum("bthd,bpd->bthp", query, pooled), 0.0).sum(
        axis=2
    ) / math.sqrt(8)
    np.testing.assert_allclose(np.asarray(scores), expected_scores, rtol=2e-3, atol=4e-3)

    keys = rng.normal(size=(2, 10, 8)).astype(np.float32)
    gates = rng.normal(size=(2, 10, 8)).astype(np.float32)
    ape = rng.normal(scale=0.1, size=(4, 8)).astype(np.float32)
    actual_pool = glm5_kpool_states(mx.array(keys), mx.array(gates), mx.array(ape), pool_size=4)
    mx.eval(actual_pool)
    grouped_keys = keys[:, :8].reshape(2, 2, 4, 8)
    logits = gates[:, :8].reshape(2, 2, 4, 8) + ape[None, None]
    probabilities = _softmax(logits, axis=2)
    expected_pool = np.sum(probabilities * grouped_keys, axis=2)
    np.testing.assert_allclose(np.asarray(actual_pool), expected_pool, rtol=2e-3, atol=3e-3)

    weights = rng.normal(size=(2, 3, 4)).astype(np.float32)
    actual_weighted = glm5_kpool_scores(mx.array(query), mx.array(pooled), mx.array(weights))
    mx.eval(actual_weighted)
    expected_weighted = np.sum(
        np.maximum(np.einsum("bthd,bpd->bthp", query, pooled), 0.0)
        / math.sqrt(8)
        * weights[..., None]
        / math.sqrt(4),
        axis=2,
    )
    np.testing.assert_allclose(np.asarray(actual_weighted), expected_weighted, rtol=2e-3, atol=4e-3)


def test_glm5_dense_and_sparse_mla_match_reference() -> None:
    rng = np.random.default_rng(57)
    batch, heads, tokens, cache_length, dimension = 1, 3, 2, 5, 128
    query = rng.normal(scale=0.2, size=(batch, heads, tokens, dimension)).astype(np.float32)
    cache = rng.normal(scale=0.2, size=(batch, cache_length, dimension)).astype(np.float32)
    offset = cache_length - tokens
    scale = 1.0 / math.sqrt(64)
    dense = glm5_dense_mla_attention(
        mx.array(query),
        mx.array(cache),
        query_offset=offset,
        scale=scale,
    )
    mx.eval(dense)

    dense_reference = np.empty((batch, tokens, heads, dimension), dtype=np.float32)
    for token in range(tokens):
        visible = offset + token + 1
        scores = np.einsum("bhd,bkd->bhk", query[:, :, token], cache[:, :visible]) * scale
        probabilities = _softmax(scores, axis=-1)
        dense_reference[:, token] = np.einsum("bhk,bkd->bhd", probabilities, cache[:, :visible])
    np.testing.assert_allclose(np.asarray(dense), dense_reference, rtol=4e-3, atol=4e-3)

    indices = np.asarray([[[0, 2, 3, -1], [1, 2, 4, -1]]], dtype=np.int32)
    sparse = glm5_sparse_mla_attention(
        mx.array(query),
        mx.array(cache),
        mx.array(indices),
        scale=scale,
    )
    mx.eval(sparse)
    sparse_reference = np.empty_like(dense_reference)
    for token in range(tokens):
        selected = indices[0, token]
        selected = selected[selected >= 0]
        selected_cache = cache[:, selected]
        scores = np.einsum("bhd,bkd->bhk", query[:, :, token], selected_cache) * scale
        probabilities = _softmax(scores, axis=-1)
        sparse_reference[:, token] = np.einsum("bhk,bkd->bhd", probabilities, selected_cache)
    np.testing.assert_allclose(np.asarray(sparse), sparse_reference, rtol=5e-3, atol=5e-3)


def test_qwen4_dense_and_sparse_gqa_match_reference() -> None:
    rng = np.random.default_rng(58)
    batch, heads, kv_heads, tokens, cache_length, dimension = 1, 4, 2, 2, 5, 128
    query = rng.normal(scale=0.2, size=(batch, heads, tokens, dimension)).astype(np.float32)
    key = rng.normal(scale=0.2, size=(batch, kv_heads, cache_length, dimension)).astype(np.float32)
    value = rng.normal(scale=0.2, size=key.shape).astype(np.float32)
    offset = cache_length - tokens
    dense = qwen4_dense_gqa_attention(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        query_offset=offset,
    )
    mx.eval(dense)

    repeated_key = np.repeat(key, heads // kv_heads, axis=1)
    repeated_value = np.repeat(value, heads // kv_heads, axis=1)
    dense_reference = np.empty((batch, tokens, heads, dimension), dtype=np.float32)
    for token in range(tokens):
        visible = offset + token + 1
        scores = np.einsum(
            "bhd,bhkd->bhk",
            query[:, :, token],
            repeated_key[:, :, :visible],
        ) / math.sqrt(dimension)
        probabilities = _softmax(scores, axis=-1)
        dense_reference[:, token] = np.einsum(
            "bhk,bhkd->bhd",
            probabilities,
            repeated_value[:, :, :visible],
        )
    np.testing.assert_allclose(np.asarray(dense), dense_reference, rtol=4e-3, atol=4e-3)

    indices = np.asarray([[[0, 2, 3, -1], [1, 2, 4, -1]]], dtype=np.int32)
    sparse = qwen4_sparse_gqa_attention(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        mx.array(indices),
    )
    mx.eval(sparse)
    sparse_reference = np.empty_like(dense_reference)
    for token in range(tokens):
        selected = indices[0, token]
        selected = selected[selected >= 0]
        scores = np.einsum(
            "bhd,bhkd->bhk",
            query[:, :, token],
            repeated_key[:, :, selected],
        ) / math.sqrt(dimension)
        probabilities = _softmax(scores, axis=-1)
        sparse_reference[:, token] = np.einsum(
            "bhk,bhkd->bhd",
            probabilities,
            repeated_value[:, :, selected],
        )
    np.testing.assert_allclose(np.asarray(sparse), sparse_reference, rtol=5e-3, atol=5e-3)


def test_sparse_attention_accepts_published_flash_next_head_widths() -> None:
    """Exercise both custom kernels at the dimensions used by the real checkpoints."""

    rng = np.random.default_rng(5801)
    selected = np.asarray([[[0, 2, 4, 6]]], dtype=np.int32)

    qwen_query = rng.normal(scale=0.1, size=(1, 24, 1, 256)).astype(np.float32)
    qwen_key = rng.normal(scale=0.1, size=(1, 2, 8, 256)).astype(np.float32)
    qwen_value = rng.normal(scale=0.1, size=qwen_key.shape).astype(np.float32)
    qwen_actual = qwen4_sparse_gqa_attention(
        mx.array(qwen_query),
        mx.array(qwen_key),
        mx.array(qwen_value),
        mx.array(selected),
    )
    mx.eval(qwen_actual)
    repeated_key = np.repeat(qwen_key, 12, axis=1)[:, :, selected[0, 0]]
    repeated_value = np.repeat(qwen_value, 12, axis=1)[:, :, selected[0, 0]]
    qwen_scores = np.einsum(
        "bhd,bhkd->bhk",
        qwen_query[:, :, 0],
        repeated_key,
    ) / math.sqrt(256)
    qwen_expected = np.einsum(
        "bhk,bhkd->bhd",
        _softmax(qwen_scores, axis=-1),
        repeated_value,
    )[:, None]
    np.testing.assert_allclose(
        np.asarray(qwen_actual),
        qwen_expected,
        rtol=6e-3,
        atol=6e-3,
    )

    glm_query = rng.normal(scale=0.1, size=(1, 64, 1, 512)).astype(np.float32)
    glm_cache = rng.normal(scale=0.1, size=(1, 8, 512)).astype(np.float32)
    glm_scale = 1.0 / math.sqrt(256)
    glm_actual = glm5_sparse_mla_attention(
        mx.array(glm_query),
        mx.array(glm_cache),
        mx.array(selected),
        scale=glm_scale,
    )
    mx.eval(glm_actual)
    selected_cache = glm_cache[:, selected[0, 0]]
    glm_scores = np.einsum(
        "bhd,bkd->bhk",
        glm_query[:, :, 0],
        selected_cache,
    ) * glm_scale
    glm_expected = np.einsum(
        "bhk,bkd->bhd",
        _softmax(glm_scores, axis=-1),
        selected_cache,
    )[:, None]
    np.testing.assert_allclose(
        np.asarray(glm_actual),
        glm_expected,
        rtol=6e-3,
        atol=6e-3,
    )


def test_qwen4_ple_dilated_convolution_is_cache_equivalent() -> None:
    rng = np.random.default_rng(59)
    value = rng.normal(size=(2, 7, 11)).astype(np.float32)
    weight = rng.normal(scale=0.2, size=(11, 1, 4)).astype(np.float32)
    full, _ = qwen4_ple_dilated_conv_silu(
        mx.array(value),
        mx.array(weight),
        None,
        dilation=3,
    )
    first, state = qwen4_ple_dilated_conv_silu(
        mx.array(value[:, :3]),
        mx.array(weight),
        None,
        dilation=3,
    )
    second, _ = qwen4_ple_dilated_conv_silu(
        mx.array(value[:, 3:]),
        mx.array(weight),
        state,
        dilation=3,
    )
    chunked = mx.concatenate((first, second), axis=1)
    mx.eval(full, chunked)
    np.testing.assert_allclose(np.asarray(chunked), np.asarray(full), rtol=1e-5, atol=1e-5)
