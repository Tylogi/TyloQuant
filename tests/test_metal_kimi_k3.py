"""Apple-silicon tests for TPQ2/Kimi-K3 Metal decode primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.kernels.metal.kimi_k3 import (  # noqa: E402
    kimi_attention_residual,
    kimi_gated_rmsnorm,
    kimi_kda_recurrent,
    kimi_route_experts,
    kimi_short_conv3,
    situ_mul,
    situ_split,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _random(seed: int, shape: tuple[int, ...], scale: float = 0.2) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size=shape).astype(np.float32)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def test_kimi_short_conv3_matches_reference_and_converts_state_dtype():
    channels, history = 37, 3
    values = tuple(_random(10 + stream, (channels,)).astype(np.float16) for stream in range(3))
    states = tuple(_random(20 + stream, (channels, history)) for stream in range(3))
    weights = tuple(_random(30 + stream, (channels, 1, history + 1)) for stream in range(3))

    query, key, value, next_states = kimi_short_conv3(
        values[0],
        values[1],
        values[2],
        states,
        weights,
    )
    for actual, next_state, source, state, weight in zip(
        (query, key, value),
        next_states,
        values,
        states,
        weights,
        strict=True,
    ):
        window = np.concatenate((state, source[:, None]), axis=-1)
        conv = np.sum(window * weight[:, 0], axis=-1)
        expected = conv * _sigmoid(conv)
        np.testing.assert_allclose(
            _array(actual),
            expected.astype(np.float16),
            rtol=2e-3,
            atol=2e-3,
        )
        np.testing.assert_array_equal(
            _array(next_state),
            window[:, 1:].astype(np.float16),
        )


def _kda_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gate: np.ndarray,
    beta: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    state: np.ndarray,
    lower_bound: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = query.shape[-1]
    query_f = query / np.maximum(
        np.sqrt(np.sum(query * query, axis=-1, keepdims=True)),
        1.0e-6,
    )
    key_f = key / np.maximum(
        np.sqrt(np.sum(key * key, axis=-1, keepdims=True)),
        1.0e-6,
    )
    gate_f = gate + dt_bias.reshape(gate.shape)
    a = np.exp(a_log[:, None])
    if lower_bound is None:
        softplus = np.maximum(gate_f, 0.0) + np.log1p(np.exp(-np.abs(gate_f)))
        log_decay = -a * softplus
    else:
        log_decay = lower_bound * _sigmoid(a * gate_f)
    current = state * np.exp(log_decay[:, None, :])
    prediction = (current @ key_f[..., None])[..., 0]
    delta = (value - prediction) * _sigmoid(beta)[:, None]
    current += delta[..., None] * key_f[:, None, :]
    output = (current @ query_f[..., None])[..., 0] / math.sqrt(dimension)
    return output, current


@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_kimi_kda_recurrent_matches_reference(lower_bound: float | None):
    heads, dimension = 3, 32
    query = _random(101, (heads, dimension)).astype(np.float16)
    key = _random(102, (heads, dimension)).astype(np.float16)
    value = _random(103, (heads, dimension)).astype(np.float16)
    gate = _random(104, (heads, dimension)).astype(np.float16)
    beta = _random(105, (heads,))
    a_log = _random(106, (heads,))
    dt_bias = _random(107, (heads * dimension,))
    state = _random(108, (heads, dimension, dimension), scale=0.02)
    expected, expected_state = _kda_reference(
        query.astype(np.float32),
        key.astype(np.float32),
        value.astype(np.float32),
        gate.astype(np.float32),
        beta,
        a_log,
        dt_bias,
        state,
        lower_bound,
    )
    actual, actual_state = kimi_kda_recurrent(
        query,
        key,
        value,
        gate,
        beta,
        a_log,
        dt_bias,
        state,
        lower_bound=lower_bound,
    )
    np.testing.assert_allclose(
        _array(actual),
        expected.astype(np.float16),
        rtol=4e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        _array(actual_state),
        expected_state,
        rtol=4e-4,
        atol=2e-5,
    )


@pytest.mark.parametrize("linear_beta", [None, 1.7])
def test_situ_mul_and_split_match_reference(linear_beta: float | None):
    gate = _random(201, (2, 37)).astype(np.float16)
    up = _random(202, gate.shape).astype(np.float16)
    beta = 2.3
    activated = beta * np.tanh(gate.astype(np.float32) / beta) * _sigmoid(gate.astype(np.float32))
    bounded_up = up.astype(np.float32)
    if linear_beta is not None:
        bounded_up = linear_beta * np.tanh(bounded_up / linear_beta)
    expected = activated * bounded_up

    actual = situ_mul(gate, up, beta=beta, linear_beta=linear_beta)
    packed = np.concatenate((gate, up), axis=-1)
    split_actual = situ_split(
        packed,
        beta=beta,
        linear_beta=linear_beta,
    )
    np.testing.assert_allclose(
        _array(actual),
        expected.astype(np.float16),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_array_equal(_array(actual), _array(split_actual))


def test_kimi_gated_rmsnorm_matches_reference():
    value = _random(301, (3, 65)).astype(np.float16)
    gate = _random(302, value.shape).astype(np.float16)
    weight = _random(303, (value.shape[-1],), scale=0.5)
    eps = 1.0e-5
    work = value.astype(np.float32)
    expected = (
        work
        / np.sqrt(np.mean(work * work, axis=-1, keepdims=True) + eps)
        * weight
        * _sigmoid(gate.astype(np.float32))
    )
    actual = kimi_gated_rmsnorm(value, gate, weight, eps)
    np.testing.assert_allclose(
        _array(actual),
        expected.astype(np.float16),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.parametrize("post_norm", [False, True])
def test_kimi_attention_residual_matches_reference(post_norm: bool):
    batch, rows, width = 2, 3, 65
    prefix = _random(401, (batch, width)).astype(np.float16)
    residual = _random(402, (batch, rows, width)).astype(np.float16)
    projection = _random(403, (width,))
    norm_weight = _random(404, (width,), scale=0.5)
    post_weight = _random(405, (width,), scale=0.5) if post_norm else None
    eps = 1.0e-5

    values = np.concatenate((residual, prefix[:, None]), axis=1).astype(np.float32)
    normalized = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + eps)
    scores = np.sum(
        normalized * projection[None, None] * norm_weight[None, None],
        axis=-1,
    )
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    expected = np.sum(probabilities[..., None] * values, axis=1)
    if post_weight is not None:
        expected = expected.astype(np.float16).astype(np.float32)
        expected = (
            expected
            / np.sqrt(np.mean(expected * expected, axis=-1, keepdims=True) + eps)
            * post_weight
        )
    actual = kimi_attention_residual(
        prefix,
        residual,
        projection,
        norm_weight,
        eps,
        post_norm_weight=post_weight,
    )
    np.testing.assert_allclose(
        _array(actual),
        expected.astype(np.float16),
        rtol=4e-3,
        atol=2e-3,
    )


def test_kimi_route_experts_masks_before_selection():
    logits = np.array(
        [[9.0, 8.0, 0.5, 0.4, 0.3, 0.2]],
        dtype=np.float16,
    )
    correction = np.array([0.0, 0.0, 0.8, 0.7, 0.0, 0.0], dtype=np.float32)
    available = np.array([False, False, True, True, True, True])
    weights, ids = kimi_route_experts(
        logits,
        correction,
        available,
        top_k=2,
        normalize=True,
        scaling=1.5,
    )
    actual_ids = _array(ids)
    assert set(actual_ids[0].tolist()) == {2, 3}
    selected = _sigmoid(logits.astype(np.float32))[0, actual_ids[0]]
    expected_weights = selected / selected.sum() * 1.5
    np.testing.assert_allclose(
        _array(weights)[0],
        expected_weights,
        rtol=2e-5,
        atol=2e-5,
    )


def test_kimi_route_experts_group_mask_matches_reference():
    logits = _random(501, (2, 12))
    correction = _random(502, (12,), scale=0.05)
    available = np.array(
        [True, True, False, True, True, False, True, True, True, False, True, True]
    )
    weights, ids = kimi_route_experts(
        logits,
        correction,
        available,
        top_k=3,
        normalize=False,
        scaling=1.25,
        n_group=4,
        topk_group=2,
    )

    scores = _sigmoid(logits)
    choice = np.where(available, scores + correction, -np.inf)
    grouped = choice.reshape(2, 4, 3)
    best_two = np.sort(grouped, axis=-1)[..., -2:]
    group_scores = np.sum(best_two, axis=-1)
    selected_groups = np.argsort(group_scores, axis=-1)[..., -2:]
    expected_choice = np.full_like(grouped, -np.inf)
    for row in range(2):
        expected_choice[row, selected_groups[row]] = grouped[row, selected_groups[row]]
    expected_ids = np.argsort(expected_choice.reshape(2, 12), axis=-1)[..., -3:][..., ::-1]
    expected_weights = np.take_along_axis(scores, expected_ids, axis=-1) * 1.25
    np.testing.assert_array_equal(_array(ids), expected_ids.astype(np.int32))
    np.testing.assert_allclose(
        _array(weights),
        expected_weights,
        rtol=2e-5,
        atol=2e-5,
    )
