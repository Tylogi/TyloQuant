"""Numerical tests for GLM DSA and sparse MLA Metal kernels."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.kernels.metal.glm_dsa import (  # noqa: E402
    attention_glm_mla_dense,
    attention_glm_mla_sparse,
    glm_dsa_cache_write,
    glm_dsa_indexer_layer_norm,
    glm_dsa_indexer_scores,
    glm_dsa_indexer_scores_decode,
    glm_interleaved_rope,
)
from mfq.runtime.mlx_glm_dsa import MlxGlmDsaConfig  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def test_glm_interleaved_rope_matches_numpy():
    rng = np.random.default_rng(701)
    source = rng.normal(size=(2, 3, 4, 12)).astype(np.float16)
    positions = np.array([0, 3, 5, 7], dtype=np.int32)
    angles = rng.normal(size=(8, 6)).astype(np.float32)
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)
    actual = _array(glm_interleaved_rope(source, positions, cosine, sine, rotary_dim=8))
    expected = source.astype(np.float32).copy()
    for token, position in enumerate(positions):
        for pair in range(4):
            first = source[:, :, token, pair * 2].astype(np.float32)
            second = source[:, :, token, pair * 2 + 1].astype(np.float32)
            expected[:, :, token, pair * 2] = (
                first * cosine[position, pair] - second * sine[position, pair]
            )
            expected[:, :, token, pair * 2 + 1] = (
                second * cosine[position, pair] + first * sine[position, pair]
            )
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_glm_indexer_layer_norm_matches_numpy():
    rng = np.random.default_rng(702)
    source = rng.normal(size=(2, 3, 128)).astype(np.float16)
    weight = rng.normal(size=(128,)).astype(np.float32)
    bias = rng.normal(size=(128,)).astype(np.float32)
    actual = _array(glm_dsa_indexer_layer_norm(source, weight, bias, 1e-5))
    values = source.astype(np.float32)
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.mean(values * values, axis=-1, keepdims=True) - mean * mean
    expected = (values - mean) / np.sqrt(np.maximum(variance, 0.0) + 1e-5)
    expected = expected * weight + bias
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_glm_cache_write_preserves_unselected_rows():
    cache = np.arange(2 * 7 * 5, dtype=np.float16).reshape(2, 7, 5)
    updates = -np.arange(2 * 3 * 5, dtype=np.float16).reshape(2, 3, 5)
    positions = np.array([5, -1, 2], dtype=np.int32)
    actual = _array(glm_dsa_cache_write(cache, updates, positions))
    expected = cache.copy()
    expected[:, 5] = updates[:, 0]
    expected[:, 2] = updates[:, 2]
    np.testing.assert_array_equal(actual, expected)


def _indexer_reference(
    q: np.ndarray,
    k: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    dots = np.einsum(
        "bmhd,bkd->bmhk",
        q.astype(np.float32),
        k.astype(np.float32),
    )
    dots = np.maximum(dots / math.sqrt(128.0), 0.0)
    return np.einsum("bmhk,bmh->bmk", dots, weights) / math.sqrt(32.0)


def test_glm_indexer_scores_prefill_matches_numpy():
    rng = np.random.default_rng(703)
    q = rng.normal(size=(2, 3, 32, 128)).astype(np.float16)
    k = rng.normal(size=(2, 70, 128)).astype(np.float16)
    weights = rng.normal(size=(2, 3, 32)).astype(np.float32)
    actual = _array(glm_dsa_indexer_scores(q, k, weights, 67, 70))
    expected = _indexer_reference(q, k, weights)
    for query in range(3):
        expected[:, query, 68 + query :] = -np.inf
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_glm_indexer_scores_decode_uses_per_batch_lengths():
    rng = np.random.default_rng(704)
    q = rng.normal(size=(2, 1, 32, 128)).astype(np.float16)
    k = rng.normal(size=(2, 80, 128)).astype(np.float16)
    weights = rng.normal(size=(2, 1, 32)).astype(np.float32)
    lengths = np.array([25, 59], dtype=np.int32)
    actual = _array(glm_dsa_indexer_scores_decode(q, k, weights, lengths, 64))
    expected = _indexer_reference(q, k[:, :64], weights)
    expected[0, 0, 25:] = -np.inf
    expected[1, 0, 59:] = -np.inf
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_glm_sparse_mla_matches_selected_numpy_attention():
    rng = np.random.default_rng(705)
    q = rng.normal(size=(1, 64, 2, 576)).astype(np.float32)
    kv = rng.normal(size=(1, 48, 576)).astype(np.float16)
    indices = np.stack(
        [
            rng.choice(48, size=32, replace=False),
            rng.choice(48, size=32, replace=False),
        ],
        axis=0,
    )[None].astype(np.int32)
    scale = 0.07
    actual = _array(attention_glm_mla_sparse(q, kv, indices, scale=scale))
    expected = np.empty((1, 2, 64, 512), dtype=np.float32)
    for query in range(2):
        selected = kv[0, indices[0, query]].astype(np.float32)
        scores = q[0, :, query] @ selected.T * scale
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        expected[0, query] = probabilities @ selected[:, :512]
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)


def test_glm_dense_mla_matches_numpy_causal_attention():
    rng = np.random.default_rng(706)
    query = rng.normal(size=(1, 64, 3, 576)).astype(np.float32)
    cache = rng.normal(size=(1, 7, 576)).astype(np.float16)
    scale = 0.04
    actual = _array(
        attention_glm_mla_dense(
            query,
            cache,
            logical_len=7,
            scale=scale,
        )
    )
    expected = np.empty((1, 3, 64, 512), dtype=np.float32)
    offset = 7 - 3
    for token in range(3):
        visible = cache[0, : offset + token + 1].astype(np.float32)
        scores = query[0, :, token] @ visible.T * scale
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        expected[0, token] = probabilities @ visible[:, :512]
    np.testing.assert_allclose(actual, expected, rtol=4e-3, atol=4e-3)


def _glm_config(**updates):
    values = {
        "model_type": "glm_moe_dsa",
        "vocab_size": 128,
        "hidden_size": 6144,
        "intermediate_size": 16384,
        "num_hidden_layers": 2,
        "max_position_embeddings": 8192,
        "q_lora_rank": 2048,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "indexer_types": ["full", "shared"],
        "mlp_layer_types": ["dense", "sparse"],
        "attention_bias": False,
        "rope_interleave": True,
        "indexer_rope_interleave": True,
        "hidden_act": "silu",
        "n_group": 1,
        "topk_group": 1,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "eos_token_id": [1, 2],
    }
    values.update(updates)
    return values


def test_glm_dsa_config_normalizes_schedules_and_fixed_dimensions():
    config = MlxGlmDsaConfig.from_hf_config(_glm_config())
    assert config.indexer_types == ("full", "shared")
    assert config.mlp_layer_types == ("dense", "sparse")
    assert config.shared_expert_intermediate_size == 2048
    assert config.eos_token_id == (1, 2)


def test_glm_dsa_config_rejects_shared_indexer_before_full():
    with pytest.raises(ValueError, match="indexer schedule"):
        MlxGlmDsaConfig.from_hf_config(_glm_config(indexer_types=["shared", "full"]))


def test_glm_dsa_config_rejects_non_equivalent_kernel_dimensions():
    with pytest.raises(ValueError, match="architecture dimensions"):
        MlxGlmDsaConfig.from_hf_config(_glm_config(index_n_heads=16))
