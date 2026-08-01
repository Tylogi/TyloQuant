"""Numerical tests for GPU-resident Metal sampling."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.kernels.metal.sampling import (  # noqa: E402
    sample,
    sample_apply_penalties,
    sample_greedy,
    sample_softmax,
    sample_token_counts_add,
    sample_top_k_top_p,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _categorical_reference(
    logits: np.ndarray,
    random: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    result = []
    for row, uniform in zip(logits, random, strict=True):
        scores = row.astype(np.float32) / temperature
        probabilities = np.exp(scores - np.max(scores))
        target = float(uniform) * float(probabilities.sum())
        cumulative = 0.0
        selected = len(row) - 1
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if cumulative >= target:
                selected = index
                break
        result.append(selected)
    return np.asarray(result, dtype=np.int32)


def _top_reference(
    logits: np.ndarray,
    random: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> np.ndarray:
    result = []
    for row, uniform in zip(logits, random, strict=True):
        scores = row.astype(np.float32) / temperature
        order = np.lexsort((np.arange(row.size), -scores))
        if top_k > 0:
            order = order[:top_k]
        selected_scores = scores[order]
        probabilities = np.exp(selected_scores - selected_scores[0])
        keep = probabilities.size
        keep_sum = float(probabilities.sum())
        if top_p < 1.0:
            cutoff = top_p * keep_sum
            cumulative = 0.0
            for rank, probability in enumerate(probabilities):
                cumulative += float(probability)
                if cumulative >= cutoff:
                    keep = rank + 1
                    keep_sum = cumulative
                    break
        target = float(uniform) * keep_sum
        cumulative = 0.0
        chosen = int(order[keep - 1])
        for rank in range(keep):
            cumulative += float(probabilities[rank])
            if cumulative >= target:
                chosen = int(order[rank])
                break
        result.append(chosen)
    return np.asarray(result, dtype=np.int32)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_sample_greedy_matches_stable_argmax(dtype):
    logits = np.asarray(
        [
            [1.0, 4.0, 4.0, -2.0, np.nan],
            [-3.0, -1.0, 2.0, 1.0, 0.0],
        ],
        dtype=dtype,
    )
    actual = _array(sample_greedy(logits))
    np.testing.assert_array_equal(actual, np.asarray([1, 2], dtype=np.int32))


def test_sample_softmax_matches_uniform_cdf():
    rng = np.random.default_rng(801)
    logits = rng.normal(size=(3, 257)).astype(np.float32)
    random = np.asarray([0.07, 0.41, 0.91], dtype=np.float32)
    actual = _array(
        sample_softmax(
            logits,
            random,
            temperature=0.75,
        )
    )
    expected = _categorical_reference(
        logits,
        random,
        temperature=0.75,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("top_k", [7, 64, 80])
def test_sample_top_k_top_p_matches_sorted_reference(top_k: int):
    rng = np.random.default_rng(810 + top_k)
    logits = rng.normal(size=(2, 193)).astype(np.float32)
    random = np.asarray([0.13, 0.77], dtype=np.float32)
    actual = _array(
        sample_top_k_top_p(
            logits,
            random,
            temperature=0.9,
            top_k=top_k,
            top_p=0.82,
        )
    )
    expected = _top_reference(
        logits,
        random,
        temperature=0.9,
        top_k=top_k,
        top_p=0.82,
    )
    np.testing.assert_array_equal(actual, expected)


def test_sample_supports_global_top_p_without_top_k():
    rng = np.random.default_rng(901)
    logits = rng.normal(size=(2, 129)).astype(np.float16)
    random = np.asarray([0.23, 0.64], dtype=np.float32)
    actual = _array(
        sample(
            logits,
            temperature=1.1,
            top_k=0,
            top_p=0.73,
            random=random,
        )
    )
    expected = _top_reference(
        logits,
        random,
        temperature=1.1,
        top_k=0,
        top_p=0.73,
    )
    np.testing.assert_array_equal(actual, expected)


def test_sampling_counts_and_penalties_match_numpy():
    counts = np.asarray([1, 0, 2, 0, 0, 1], dtype=np.int32)
    tokens = np.asarray([[1, 2, 2], [5, -1, 9]], dtype=np.int32)
    actual_counts = _array(sample_token_counts_add(counts, tokens))
    expected_counts = np.asarray([1, 1, 4, 0, 0, 2], dtype=np.int32)
    np.testing.assert_array_equal(actual_counts, expected_counts)

    logits = np.asarray(
        [[2.0, -3.0, 1.5, 0.0, -1.0, 4.0]],
        dtype=np.float16,
    )
    actual = _array(
        sample_apply_penalties(
            logits,
            actual_counts,
            presence_penalty=0.2,
            frequency_penalty=0.1,
            repetition_penalty=1.25,
        )
    )
    expected = logits.astype(np.float32).copy()
    for token, count in enumerate(expected_counts):
        if count:
            expected[:, token] = np.where(
                expected[:, token] < 0.0,
                expected[:, token] * 1.25,
                expected[:, token] / 1.25,
            )
            expected[:, token] -= 0.2 + 0.1 * count
    np.testing.assert_allclose(
        actual,
        expected.astype(np.float16),
        rtol=0,
        atol=0,
    )
