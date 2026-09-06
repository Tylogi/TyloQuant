from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats.io import BFloat16Array  # noqa: E402
from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.quantize.nint_quant import quantize  # noqa: E402
from mfq.runtime.mlx_glm5_next import (  # noqa: E402
    MlxGlm5NextKda,
    MlxGlm5NextMtp,
    MlxGlm5NextSparseAttention,
)
from mfq.runtime.mlx_linear import MlxNintModel  # noqa: E402


def _random(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    scale: float = 0.05,
) -> np.ndarray:
    return rng.normal(scale=scale, size=shape).astype(np.float32)


def _bfloat16(values: np.ndarray) -> BFloat16Array:
    """Store NumPy F32 values as the tagged BF16 representation used by MFQ."""

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return (bits >> 16).astype(np.uint16).view(BFloat16Array)


def _expert(weight: np.ndarray) -> NintMoeTensor:
    experts, output, width = (int(item) for item in weight.shape)
    packed = quantize(
        weight.reshape(experts * output, width),
        NintSpec(4, 8, 6),
    )
    return NintMoeTensor(
        shape=(experts, output, width),
        pools=(
            NintMoePool(
                expert_ids=np.arange(experts, dtype=np.int32),
                tensor=packed,
            ),
        ),
    )


def test_glm5_kda_cached_chunks_match_full_sequence() -> None:
    rng = np.random.default_rng(5301)
    hidden = dimension = 128
    heads = 1
    width = heads * dimension
    prefix = "layer.self_attn"
    tensors: dict[str, np.ndarray] = {}
    for projection in ("q", "k", "v"):
        tensors[f"{prefix}.{projection}_proj.weight"] = _random(rng, (width, hidden))
        conv = np.zeros((width, 1, 4), dtype=np.float32)
        conv[:, 0, -1] = 0.8
        conv[:, 0, -2] = 0.1
        tensors[f"{prefix}.{projection}_conv1d.weight"] = conv
    tensors[prefix + ".f_a_proj.weight"] = _random(rng, (dimension, hidden))
    tensors[prefix + ".f_b_proj.weight"] = _random(rng, (width, dimension))
    tensors[prefix + ".dt_bias"] = _random(rng, (width,), 0.02)
    tensors[prefix + ".A_log"] = _random(rng, (heads,), 0.02)
    tensors[prefix + ".b_proj.weight"] = _random(rng, (heads, hidden))
    tensors[prefix + ".g_a_proj.weight"] = _random(rng, (dimension, hidden))
    tensors[prefix + ".g_b_proj.weight"] = _random(rng, (width, dimension))
    tensors[prefix + ".o_norm.weight"] = np.ones((dimension,), dtype=np.float32)
    tensors[prefix + ".o_proj.weight"] = _random(rng, (hidden, width))
    config = SimpleNamespace(
        kda_num_heads=heads,
        kda_head_dim=dimension,
        kda_conv_kernel_size=4,
        kda_gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
    )
    source = mx.array(_random(rng, (1, 5, hidden), 0.2)).astype(mx.float16)

    full = MlxGlm5NextKda(MlxNintModel(tensors), config, prefix)
    expected = full(source, use_cache=False)

    cached = MlxGlm5NextKda(MlxNintModel(tensors), config, prefix)
    cached.reset_cache(1)
    first = cached(source[:, :2], use_cache=True)
    second = cached(source[:, 2:], use_cache=True)
    actual = mx.concatenate((first, second), axis=1)
    mx.eval(expected, actual)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=7e-3,
        atol=7e-3,
    )

    reference = MlxGlm5NextKda(MlxNintModel(tensors), config, prefix)
    reference.reset_cache(1)
    reference(source[:, :2], use_cache=True)
    expected_confirmed = reference(source[:, 2:3], use_cache=True)
    expected_after_reject = reference(source[:, 4:5], use_cache=True)

    speculative = MlxGlm5NextKda(MlxNintModel(tensors), config, prefix)
    speculative.reset_cache(1)
    speculative(source[:, :2], use_cache=True)
    verified = speculative(
        source[:, 2:4],
        use_cache=True,
        n_confirmed=1,
    )
    mx.eval(verified)
    speculative.rollback_speculative_cache()
    actual_after_reject = speculative(source[:, 4:5], use_cache=True)
    mx.eval(expected_confirmed, expected_after_reject, actual_after_reject)

    np.testing.assert_allclose(
        np.asarray(verified[:, :1]),
        np.asarray(expected_confirmed),
        rtol=7e-3,
        atol=7e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_after_reject),
        np.asarray(expected_after_reject),
        rtol=7e-3,
        atol=7e-3,
    )


def test_glm5_absorbed_sparse_mla_cached_chunks_match_full_sequence() -> None:
    rng = np.random.default_rng(5302)
    hidden = rank = nope = value_dim = index_dim = 128
    heads = index_heads = 2
    prefix = "layer.self_attn"
    tensors: dict[str, np.ndarray | NintMoeTensor] = {
        prefix + ".q_a_proj.weight": _random(rng, (rank, hidden)),
        prefix + ".kv_a_proj_with_mqa.weight": _random(rng, (rank, hidden)),
        prefix + ".q_a_layernorm.weight": np.ones((rank,), dtype=np.float32),
        prefix + ".kv_a_layernorm.weight": np.ones((rank,), dtype=np.float32),
        prefix + ".q_b_proj.weight": _random(rng, (heads * nope, rank)),
        prefix + ".o_proj.weight": _random(rng, (hidden, heads * value_dim)),
        prefix + ".indexer.wq_b.weight": _random(rng, (index_heads * index_dim, rank)),
        prefix + ".indexer.wk.weight": _random(rng, (index_dim, hidden)),
        prefix + ".indexer.weights_proj.weight": _random(rng, (index_heads, hidden)),
        prefix + ".indexer.k_norm.weight": np.ones((index_dim,), dtype=np.float32),
        prefix + ".indexer.k_norm.bias": np.zeros((index_dim,), dtype=np.float32),
        prefix + ".indexer.index_kpool_compress_gate": _random(rng, (index_dim, hidden)),
        prefix + ".indexer.index_kpool_compress_ape": _random(rng, (2, index_dim)),
        prefix + ".embed_q": _expert(_random(rng, (heads, rank, nope))),
        prefix + ".unembed_out": _expert(_random(rng, (heads, value_dim, rank))),
    }
    config = SimpleNamespace(
        hidden_size=hidden,
        q_lora_rank=rank,
        kv_lora_rank=rank,
        num_attention_heads=heads,
        qk_nope_head_dim=nope,
        v_head_dim=value_dim,
        index_n_heads=index_heads,
        index_head_dim=index_dim,
        index_topk=4,
        index_kpool=2,
        index_kpool_always_select_tail=True,
        rms_norm_eps=1e-5,
    )
    source = mx.array(_random(rng, (1, 6, hidden), 0.2)).astype(mx.float16)

    full = MlxGlm5NextSparseAttention(MlxNintModel(tensors), config, prefix, 16)
    expected = full(source, use_cache=False)

    cached = MlxGlm5NextSparseAttention(MlxNintModel(tensors), config, prefix, 16)
    cached.reset_cache(1)
    first = cached(source[:, :2], use_cache=True)
    second = cached(source[:, 2:], use_cache=True)
    actual = mx.concatenate((first, second), axis=1)
    mx.eval(expected, actual)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=2e-2,
        atol=2e-2,
    )


def test_glm5_mtp_cached_chunks_match_full_appended_layer() -> None:
    rng = np.random.default_rng(5303)
    hidden = rank = nope = value_dim = index_dim = 128
    heads = index_heads = experts = 2
    intermediate = 64
    prefix = "model.language_model.layers.1"
    tensors: dict[str, np.ndarray | NintMoeTensor] = {
        "model.language_model.embed_tokens.weight": _bfloat16(
            _random(rng, (32, hidden))
        ),
        prefix + ".enorm.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".hnorm.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".eh_proj.weight": _bfloat16(
            _random(rng, (hidden, 2 * hidden))
        ),
        prefix + ".input_layernorm.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".post_attention_layernorm.weight": np.ones(
            (hidden,), dtype=np.float32
        ),
        prefix + ".shared_head.norm.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".self_attn.q_a_proj.weight": _random(rng, (rank, hidden)),
        prefix + ".self_attn.kv_a_proj_with_mqa.weight": _random(
            rng, (rank, hidden)
        ),
        prefix + ".self_attn.q_a_layernorm.weight": np.ones(
            (rank,), dtype=np.float32
        ),
        prefix + ".self_attn.kv_a_layernorm.weight": np.ones(
            (rank,), dtype=np.float32
        ),
        prefix + ".self_attn.q_b_proj.weight": _random(
            rng, (heads * nope, rank)
        ),
        prefix + ".self_attn.o_proj.weight": _random(
            rng, (hidden, heads * value_dim)
        ),
        prefix + ".self_attn.indexer.wq_b.weight": _random(
            rng, (index_heads * index_dim, rank)
        ),
        prefix + ".self_attn.indexer.wk.weight": _random(rng, (index_dim, hidden)),
        prefix + ".self_attn.indexer.weights_proj.weight": _random(
            rng, (index_heads, hidden)
        ),
        prefix + ".self_attn.indexer.k_norm.weight": np.ones(
            (index_dim,), dtype=np.float32
        ),
        prefix + ".self_attn.indexer.k_norm.bias": np.zeros(
            (index_dim,), dtype=np.float32
        ),
        prefix + ".self_attn.indexer.index_kpool_compress_gate": _random(
            rng, (index_dim, hidden)
        ),
        prefix + ".self_attn.indexer.index_kpool_compress_ape": _random(
            rng, (2, index_dim)
        ),
        prefix + ".self_attn.embed_q": _expert(
            _random(rng, (heads, rank, nope))
        ),
        prefix + ".self_attn.unembed_out": _expert(
            _random(rng, (heads, value_dim, rank))
        ),
        prefix + ".mlp.experts.gate_up_proj": _expert(
            _random(rng, (experts, 2 * intermediate, hidden))
        ),
        prefix + ".mlp.experts.down_proj": _expert(
            _random(rng, (experts, hidden, intermediate))
        ),
        prefix + ".mlp.gate.weight": _random(rng, (experts, hidden)),
        prefix + ".mlp.gate.e_score_correction_bias": _random(rng, (experts,)),
        prefix + ".mlp.shared_experts.gate_proj.weight": _random(
            rng, (intermediate, hidden)
        ),
        prefix + ".mlp.shared_experts.up_proj.weight": _random(
            rng, (intermediate, hidden)
        ),
        prefix + ".mlp.shared_experts.down_proj.weight": _random(
            rng, (hidden, intermediate)
        ),
    }
    config = SimpleNamespace(
        num_nextn_predict_layers=1,
        num_hidden_layers=1,
        max_position_embeddings=16,
        hidden_size=hidden,
        tie_word_embeddings=True,
        rms_norm_eps=1e-5,
        q_lora_rank=rank,
        kv_lora_rank=rank,
        num_attention_heads=heads,
        qk_nope_head_dim=nope,
        v_head_dim=value_dim,
        index_n_heads=index_heads,
        index_head_dim=index_dim,
        index_topk=4,
        index_kpool=2,
        index_kpool_always_select_tail=True,
        num_experts=experts,
        num_experts_per_tok=1,
        moe_intermediate_size=intermediate,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        swiglu_limit=7.0,
    )
    ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    previous = mx.array(_random(rng, (1, 6, hidden), 0.2)).astype(mx.float16)

    full = MlxGlm5NextMtp(MlxNintModel(tensors), config, max_context=16)
    expected = full(ids, previous, use_cache=False)
    layer = full.layers[0]
    embeddings = full.embedding(ids)
    masked = mx.where(
        mx.arange(6, dtype=mx.int32)[None, :, None] == 0,
        mx.zeros_like(embeddings),
        embeddings,
    )
    fused = layer.eh_projection(
        mx.concatenate((layer.enorm(masked), layer.hnorm(previous)), axis=-1)
    )
    residual = fused
    attention = layer.attention(layer.input_norm(fused), use_cache=False)
    residual = (residual + attention).astype(fused.dtype)
    feed_forward = layer.ffn(layer.post_attention_norm(residual))
    reference = layer.shared_head_norm((residual + feed_forward).astype(fused.dtype))
    mx.eval(expected, reference)
    np.testing.assert_allclose(
        np.asarray(expected),
        np.asarray(reference),
        rtol=3e-2,
        atol=3e-2,
    )

    cached = MlxGlm5NextMtp(MlxNintModel(tensors), config, max_context=16)
    cached.reset_cache(1)
    first = cached(ids[:, :2], previous[:, :2], use_cache=True)
    second = cached(ids[:, 2:], previous[:, 2:], use_cache=True)
    actual = mx.concatenate((first, second), axis=1)
    mx.eval(expected, actual)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=3e-2,
        atol=3e-2,
    )
