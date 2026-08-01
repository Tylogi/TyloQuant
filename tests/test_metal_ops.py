"""Apple-silicon tests for reusable Transformer Metal kernels."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.kernels.metal.ops import (  # noqa: E402
    gelu_mul,
    hadamard_mul,
    l2_norm,
    residual_add,
    residual_rms_norm,
    rms_norm,
    rope,
    rope_tables,
    silu_mul,
)
from mfq.runtime import MlxRMSNorm, MlxRoPE  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _random(seed: int, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.5, size=shape).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_elementwise_transformer_kernels(dtype):
    gate = _random(1, (2, 3, 97), dtype)
    up = _random(2, (2, 3, 97), dtype)
    gate32 = gate.astype(np.float32)
    up32 = up.astype(np.float32)
    tolerance = 2e-3 if dtype == np.float16 else 2e-6

    np.testing.assert_allclose(
        _array(residual_add(gate, up)),
        gate32 + up32,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        _array(hadamard_mul(gate, up)),
        gate32 * up32,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        _array(silu_mul(gate, up)),
        (gate32 / (1.0 + np.exp(-gate32))) * up32,
        rtol=tolerance,
        atol=tolerance,
    )
    gelu = (
        0.5
        * gate32
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gate32 + 0.044715 * gate32**3)))
        * up32
    )
    np.testing.assert_allclose(
        _array(gelu_mul(gate, up)),
        gelu,
        rtol=tolerance,
        atol=tolerance,
    )


def test_elementwise_promotes_fp16_and_fp32_to_fp32():
    left = _random(3, (4, 65), np.float16)
    right = _random(4, (4, 65), np.float32)
    actual = residual_add(left, right)
    assert actual.dtype == mx.float32
    np.testing.assert_allclose(
        _array(actual),
        left.astype(np.float32) + right,
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.parametrize("width", [16, 96, 4096])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_rms_norm_matches_numpy(width: int, dtype):
    source = _random(10 + width, (2, 3, width), dtype)
    weight = _random(20 + width, (width,))
    eps = 1e-5
    actual = _array(rms_norm(source, weight, eps))
    source32 = source.astype(np.float32)
    expected = (
        source32
        * (1.0 / np.sqrt(np.mean(source32 * source32, axis=-1, keepdims=True) + eps))
        * weight
    )
    tolerance = 2e-3 if dtype == np.float16 else 3e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_rms_norm_weight_offset_matches_qwen_semantics():
    source = _random(31, (5, 128))
    weight = _random(32, (128,))
    actual = _array(rms_norm(source, weight, 1e-6, weight_offset=1.0))
    inverse = 1.0 / np.sqrt(np.mean(source * source, axis=-1, keepdims=True) + 1e-6)
    expected = source * inverse * (weight + 1.0)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_l2_norm_matches_numpy(dtype):
    source = _random(41, (2, 7, 257), dtype)
    source32 = source.astype(np.float32)
    actual = _array(l2_norm(source, eps=1e-5))
    expected = source32 / np.maximum(
        np.sqrt(np.sum(source32 * source32, axis=-1, keepdims=True)),
        1e-5,
    )
    tolerance = 1e-3 if dtype == np.float16 else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_fused_residual_rms_norm_matches_separate_reference(dtype):
    residual = _random(51, (2, 4, 192), dtype)
    update = _random(52, (2, 4, 192), dtype)
    weight = _random(53, (192,))
    actual_sum, actual_norm = residual_rms_norm(
        residual,
        update,
        weight,
        1e-5,
        weight_offset=1.0,
        normalized_dtype=mx.float32,
    )
    stored = (residual.astype(np.float32) + update.astype(np.float32)).astype(dtype)
    stored32 = stored.astype(np.float32)
    inverse = 1.0 / np.sqrt(np.mean(stored32 * stored32, axis=-1, keepdims=True) + 1e-5)
    expected_norm = stored32 * inverse * (weight + 1.0)
    np.testing.assert_array_equal(_array(actual_sum), stored)
    np.testing.assert_allclose(_array(actual_norm), expected_norm, rtol=4e-6, atol=4e-6)


def _rope_reference(
    source: np.ndarray,
    positions: np.ndarray,
    base: float,
    rotary_dim: int,
    sections: tuple[int, int, int] | None = None,
    *,
    frequency_dim: int | None = None,
    active_pairs: int | None = None,
) -> np.ndarray:
    output = source.copy()
    half = rotary_dim // 2
    denominator = rotary_dim if frequency_dim is None else frequency_dim
    frequencies = base ** (-2.0 * np.arange(half, dtype=np.float32) / denominator)
    if active_pairs is not None:
        frequencies[active_pairs:] = 0.0
    for index in np.ndindex(source.shape[:-2]):
        for token in range(source.shape[-2]):
            for pair in range(half):
                axis = 0
                if sections is not None:
                    axis = 0 if pair < sections[0] else (1 if pair < sum(sections[:2]) else 2)
                    if positions.ndim == 1 or axis >= positions.shape[0]:
                        axis = 0
                position = positions[token] if positions.ndim == 1 else positions[axis, token]
                angle = float(position) * frequencies[pair]
                cosine, sine = np.cos(angle), np.sin(angle)
                first = float(source[index + (token, pair)])
                second = float(source[index + (token, pair + half)])
                output[index + (token, pair)] = first * cosine - second * sine
                output[index + (token, pair + half)] = second * cosine + first * sine
    return output


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_partial_rope_matches_rotate_half_reference(dtype):
    source = _random(61, (2, 3, 5, 24), dtype)
    positions = np.asarray([0, 1, 4, 9, 15], dtype=np.int32)
    actual = _array(
        rope(
            source,
            positions,
            base=10_000.0,
            rotary_dim=16,
            table_len=32,
        )
    )
    expected = _rope_reference(source, positions, 10_000.0, 16)
    tolerance = 2e-3 if dtype == np.float16 else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(actual[..., 16:], source[..., 16:])


def test_mrope_sections_match_reference():
    source = _random(71, (2, 2, 4, 12))
    positions = np.asarray(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=np.int32,
    )
    sections = (2, 2, 2)
    actual = _array(
        rope(
            source,
            positions,
            base=1_000.0,
            rotary_dim=12,
            sections=sections,
            table_len=16,
        )
    )
    expected = _rope_reference(source, positions, 1_000.0, 12, sections)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_rope_active_pairs_match_gemma4_partial_layout():
    source = _random(76, (2, 3, 5, 32))
    positions = np.asarray([0, 1, 4, 7, 11], dtype=np.int32)
    actual = _array(
        rope(
            source,
            positions,
            base=100_000.0,
            rotary_dim=32,
            table_len=16,
            frequency_dim=32,
            active_pairs=6,
        )
    )
    expected = _rope_reference(
        source,
        positions,
        100_000.0,
        32,
        frequency_dim=32,
        active_pairs=6,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_rope_accepts_non_penultimate_sequence_axis():
    source = _random(81, (2, 5, 3, 16))
    positions = np.arange(5, dtype=np.int32)
    actual = _array(
        rope(
            source,
            positions,
            base=10_000.0,
            table_len=8,
            sequence_axis=1,
        )
    )
    canonical = np.moveaxis(source, 1, -2)
    expected = np.moveaxis(
        _rope_reference(canonical, positions, 10_000.0, 16),
        -2,
        1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_runtime_norm_and_rope_modules():
    source = _random(91, (2, 3, 4, 32), np.float16)
    positions = np.arange(4, dtype=np.int32)
    norm = MlxRMSNorm(np.ones(32, dtype=np.float32), eps=1e-5)
    rotary = MlxRoPE(32, 16, base=10_000.0)
    normalized = norm(source)
    actual = _array(rotary(normalized, positions))
    assert actual.shape == source.shape
    assert np.isfinite(actual).all()

    residual, fused = norm.add_and_forward(source, source, normalized_dtype=mx.float32)
    mx.eval(residual, fused)
    assert residual.dtype == mx.float16
    assert fused.dtype == mx.float32


def test_rope_table_cache_reuses_arrays():
    first = rope_tables(10_000.0, 32, 128)
    second = rope_tables(10_000.0, 32, 128)
    assert first[0] is second[0]
    assert first[1] is second[1]
