"""Apple-silicon tests for GDN and SSM linear-attention kernels."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.kernels.metal.linear_attention import (  # noqa: E402
    gated_delta_net,
    linear_conv_qkv,
    ssm_conv_silu,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _random(seed: int, shape: tuple[int, ...], scale: float = 0.2) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size=shape).astype(np.float32)


def _gdn_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    state: np.ndarray | None = None,
    *,
    tiled_heads: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    batch, query_heads, tokens, dimension = q.shape
    value_heads = v.shape[1]
    result = np.empty_like(v)
    current = (
        np.zeros((batch, value_heads, dimension, dimension), dtype=np.float32)
        if state is None
        else state.copy()
    )
    for token in range(tokens):
        for value_head in range(value_heads):
            query_head = (
                value_head % query_heads
                if tiled_heads
                else value_head // (value_heads // query_heads)
            )
            if g.ndim == 4:
                current[:, value_head] *= np.exp(g[:, value_head, token, :, None])
            else:
                current[:, value_head] *= np.exp(g[:, value_head, token, None, None])
            key = k[:, query_head, token]
            query = q[:, query_head, token]
            projected = (np.swapaxes(current[:, value_head], -1, -2) @ key[..., None])[..., 0]
            delta = (v[:, value_head, token] - projected) * beta[:, value_head, token, None]
            current[:, value_head] += key[..., None] * delta[:, None, :]
            result[:, value_head, token] = (
                np.swapaxes(current[:, value_head], -1, -2) @ query[..., None]
            )[..., 0] * (dimension**-0.5)
    return result, current


@pytest.mark.parametrize("dimension", [32, 64, 128])
@pytest.mark.parametrize("kda", [False, True])
def test_gdn_matches_reference(dimension: int, kda: bool):
    batch, query_heads, value_heads, tokens = 1, 2, 4, 4
    q = _random(10 + dimension, (batch, query_heads, tokens, dimension))
    k = _random(20 + dimension, q.shape)
    v = _random(30 + dimension, (batch, value_heads, tokens, dimension))
    gate_shape = (batch, value_heads, tokens, dimension) if kda else (batch, value_heads, tokens)
    gate = -np.abs(_random(40 + dimension, gate_shape, scale=0.1))
    beta = 1.0 / (1.0 + np.exp(-_random(50 + dimension, (batch, value_heads, tokens))))
    expected, expected_state = _gdn_reference(q, k, v, gate, beta)
    actual, actual_state = gated_delta_net(q, k, v, gate, beta)
    np.testing.assert_allclose(_array(actual), expected, rtol=8e-5, atol=8e-6)
    np.testing.assert_allclose(
        _array(actual_state),
        expected_state,
        rtol=8e-5,
        atol=8e-6,
    )


def test_gdn_state_chunking_transpose_and_tiled_heads():
    batch, query_heads, value_heads, tokens, dimension = 1, 2, 4, 5, 32
    q = _random(101, (batch, query_heads, tokens, dimension))
    k = _random(102, q.shape)
    v = _random(103, (batch, value_heads, tokens, dimension))
    gate = -np.abs(_random(104, (batch, value_heads, tokens), scale=0.1))
    beta = 1.0 / (1.0 + np.exp(-_random(105, (batch, value_heads, tokens))))
    initial = _random(106, (batch, value_heads, dimension, dimension), scale=0.02)
    expected, expected_state = _gdn_reference(
        q,
        k,
        v,
        gate,
        beta,
        initial,
        tiled_heads=True,
    )
    first_out, first_state = gated_delta_net(
        q[:, :, :2],
        k[:, :, :2],
        v[:, :, :2],
        gate[:, :, :2],
        beta[:, :, :2],
        np.swapaxes(initial, -1, -2),
        transposed_state=True,
        tiled_heads=True,
    )
    second_out, final_state = gated_delta_net(
        q[:, :, 2:],
        k[:, :, 2:],
        v[:, :, 2:],
        gate[:, :, 2:],
        beta[:, :, 2:],
        first_state,
        transposed_state=True,
        tiled_heads=True,
    )
    actual = np.concatenate((_array(first_out), _array(second_out)), axis=2)
    np.testing.assert_allclose(actual, expected, rtol=8e-5, atol=8e-6)
    np.testing.assert_allclose(
        np.swapaxes(_array(final_state), -1, -2),
        expected_state,
        rtol=8e-5,
        atol=8e-6,
    )


def _ssm_reference(
    conv_input: np.ndarray,
    weight: np.ndarray,
    tokens: int,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    channels = conv_input.shape[-1]
    if weight.ndim == 3:
        packed = weight[:, 0, :]
    elif weight.shape[0] == channels:
        packed = weight
    else:
        packed = weight.T
    kernel = packed.shape[1]
    output = np.empty((conv_input.shape[0], tokens, channels), dtype=np.float32)
    for token in range(tokens):
        values = np.sum(
            conv_input[:, token : token + kernel] * packed.T[None, :, :],
            axis=1,
        )
        if bias is not None:
            values += bias
        output[:, token] = values / (1.0 + np.exp(-values))
    return output


@pytest.mark.parametrize("layout", ["c1k", "ck", "kc"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_ssm_conv_silu_layouts(layout: str, with_bias: bool):
    batch, tokens, channels, kernel = 2, 6, 37, 4
    source = _random(201, (batch, tokens + kernel - 1, channels))
    packed = _random(202, (channels, kernel))
    weight = packed[:, None, :] if layout == "c1k" else (packed if layout == "ck" else packed.T)
    bias = _random(203, (channels,)) if with_bias else None
    actual = _array(ssm_conv_silu(source, weight, tokens, bias))
    expected = _ssm_reference(source, weight, tokens, bias)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_fused_linear_conv_qkv_matches_composed_reference(tokens: int, dtype):
    batch, kernel = 2, 4
    key_heads, value_heads = 2, 4
    key_dim, value_dim = 37, 19
    qk_width = 2 * key_heads * key_dim
    v_width = value_heads * value_dim
    channels = qk_width + v_width
    state = _random(301, (batch, kernel - 1, channels))
    qk = _random(302, (batch, tokens, qk_width)).astype(dtype)
    v = _random(303, (batch, tokens, v_width)).astype(dtype)
    weight = _random(304, (kernel, channels))
    bias = _random(305, (channels,))

    q_actual, k_actual, v_actual, state_actual = linear_conv_qkv(
        state,
        qk,
        v,
        weight,
        num_key_heads=key_heads,
        num_value_heads=value_heads,
        key_head_dim=key_dim,
        value_head_dim=value_dim,
        bias=bias,
        eps=1e-5,
    )
    source = np.concatenate((state, np.concatenate((qk, v), axis=-1)), axis=1)
    convolved = _ssm_reference(source, weight, tokens, bias)
    q_ref = convolved[:, :, : key_heads * key_dim].reshape(batch, tokens, key_heads, key_dim)
    k_ref = convolved[:, :, key_heads * key_dim : 2 * key_heads * key_dim].reshape(
        batch, tokens, key_heads, key_dim
    )
    v_ref = convolved[:, :, qk_width:].reshape(batch, tokens, value_heads, value_dim)
    q_ref /= np.maximum(
        np.sqrt(np.sum(q_ref * q_ref, axis=-1, keepdims=True)),
        1e-5,
    )
    k_ref /= np.maximum(
        np.sqrt(np.sum(k_ref * k_ref, axis=-1, keepdims=True)),
        1e-5,
    )
    q_ref = np.transpose(q_ref, (0, 2, 1, 3))
    k_ref = np.transpose(k_ref, (0, 2, 1, 3))
    v_ref = np.transpose(v_ref, (0, 2, 1, 3))
    new_state = source[:, -(kernel - 1) :]
    tolerance = 6e-5 if dtype == np.float16 else 5e-6
    np.testing.assert_allclose(_array(q_actual), q_ref, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(_array(k_actual), k_ref, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(_array(v_actual), v_ref, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(_array(state_actual), new_state, rtol=0, atol=0)
