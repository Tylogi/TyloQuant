from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.architectures.tensor_schema import map_source_tensor_name  # noqa: E402
from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.quantize.nint_quant import quantize  # noqa: E402
from mfq.runtime.mlx_linear import MlxNintModel  # noqa: E402
from mfq.runtime.mlx_qwen4_exp import (  # noqa: E402
    MlxQwen4ExpGdn,
    MlxQwen4ExpMtp,
    MlxQwen4ExpNgramEmbedding,
    MlxQwen4ExpPle,
    MlxQwen4ExpQsa,
)


def _random(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    scale: float = 0.05,
) -> np.ndarray:
    return rng.normal(scale=scale, size=shape).astype(np.float32)


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


def _canonical_tensors(tensors: dict, *, text_layers: int = 1) -> dict:
    config = {
        "model_type": "qwen4_exp",
        "num_hidden_layers": text_layers,
        "mtp_num_hidden_layers": 1,
    }
    mapped = {}
    for name, value in tensors.items():
        resolved = map_source_tensor_name(name, config, require_registered=True)
        assert resolved is not None
        mapped[resolved.canonical_name] = value
    return tensors.__class__(mapped)


def test_qwen4_gdn_cached_chunks_match_full_sequence() -> None:
    rng = np.random.default_rng(3801)
    hidden = dimension = 128
    key_heads, value_heads = 1, 2
    key_width = key_heads * dimension
    value_width = value_heads * dimension
    conv_width = 2 * key_width + value_width
    prefix = "model.language_model.layers.0.linear_attn"
    tensors: dict[str, np.ndarray] = {
        prefix + ".in_proj_qkv.weight": _random(rng, (conv_width, hidden)),
        prefix + ".in_proj_z.weight": _random(rng, (value_width, hidden)),
        prefix + ".in_proj_a.weight": _random(rng, (value_heads, hidden)),
        prefix + ".in_proj_b.weight": _random(rng, (value_heads, hidden)),
        prefix + ".dt_bias": _random(rng, (value_heads,), 0.02),
        prefix + ".A_log": _random(rng, (value_heads,), 0.02),
        prefix + ".norm.weight": np.ones((dimension,), dtype=np.float32),
        prefix + ".out_proj.weight": _random(rng, (hidden, value_width)),
    }
    conv = np.zeros((conv_width, 1, 4), dtype=np.float32)
    conv[:, 0, -1] = 0.8
    conv[:, 0, -2] = 0.1
    tensors[prefix + ".conv1d.weight"] = conv
    tensors = _canonical_tensors(tensors)
    prefix = "model.block.0.linear_attention"
    config = SimpleNamespace(
        linear_num_key_heads=key_heads,
        linear_num_value_heads=value_heads,
        linear_key_head_dim=dimension,
        linear_value_head_dim=dimension,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        output_gate_type="silu",
    )
    source = mx.array(_random(rng, (1, 6, hidden), 0.2)).astype(mx.float16)

    full = MlxQwen4ExpGdn(MlxNintModel(tensors), config, prefix)
    expected = full(source, use_cache=False)

    cached = MlxQwen4ExpGdn(MlxNintModel(tensors), config, prefix)
    cached.reset_cache(1)
    first = cached(source[:, :2], use_cache=True)
    second = cached(source[:, 2:], use_cache=True)
    actual = mx.concatenate((first, second), axis=1)
    mx.eval(expected, actual)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=8e-3,
        atol=8e-3,
    )

    reference = MlxQwen4ExpGdn(MlxNintModel(tensors), config, prefix)
    reference.reset_cache(1)
    reference(source[:, :2], use_cache=True)
    expected_confirmed = reference(source[:, 2:3], use_cache=True)
    expected_after_reject = reference(source[:, 4:5], use_cache=True)

    speculative = MlxQwen4ExpGdn(MlxNintModel(tensors), config, prefix)
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
        rtol=8e-3,
        atol=8e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_after_reject),
        np.asarray(expected_after_reject),
        rtol=8e-3,
        atol=8e-3,
    )


def _reference_ngram_ids(
    input_ids: np.ndarray,
    *,
    eos: int,
    multipliers: tuple[int, ...],
    vocab_sizes: tuple[int, ...],
    offsets: tuple[int, ...],
) -> np.ndarray:
    mask64 = (1 << 64) - 1
    context = len(multipliers) - 1
    history = np.concatenate(
        (np.full((input_ids.shape[0], context), eos, dtype=np.int64), input_ids),
        axis=1,
    )
    output = np.empty((*input_ids.shape, context), dtype=np.int64)
    for batch in range(input_ids.shape[0]):
        segment_start = 0
        shifted = []
        for shift in range(len(multipliers)):
            values = []
            segment_start = 0
            for token, token_id in enumerate(history[batch]):
                source = token - shift
                values.append(int(history[batch, source]) if source >= segment_start else eos)
                if token_id == eos:
                    segment_start = token + 1
            shifted.append(values)
        for token in range(context, history.shape[1]):
            for ngram in range(2, len(multipliers) + 1):
                mixed = 0
                for position in range(ngram):
                    product = shifted[position][token] * multipliers[position]
                    mixed ^= product & mask64
                signed = mixed if mixed < (1 << 63) else mixed - (1 << 64)
                head = ngram - 2
                output[batch, token - context, head] = signed % vocab_sizes[head] + offsets[head]
    return output


def test_qwen4_sharded_ple_hash_matches_reference_and_cache() -> None:
    class CountingDict(dict[str, np.ndarray]):
        def __init__(self, values):
            super().__init__(values)
            self.reads: dict[str, int] = {}

        def __getitem__(self, key):
            self.reads[key] = self.reads.get(key, 0) + 1
            return super().__getitem__(key)

    prefix = "model.language_model.layers.0.ple"
    embedding_prefix = prefix + ".ple_embedding.ngram_embedding"
    global_embedding = np.arange(16 * 2, dtype=np.float32).reshape(16, 2)
    tensors = CountingDict({
        embedding_prefix + ".shard_0.weight": global_embedding[:8],
        embedding_prefix + ".shard_1.weight": global_embedding[8:],
        prefix + ".ple_embedding.layer_multipliers": np.asarray([3, 5, 7], dtype=np.int64),
        prefix + ".ple_embedding.ngram_heads_offsets": np.asarray([0, 8], dtype=np.int64),
        prefix + ".ple_embedding.ngram_heads_vocab_sizes": np.asarray([5, 7], dtype=np.int64),
    })
    tensors = _canonical_tensors(tensors)
    prefix = "model.block.0.position_embedding"
    embedding_prefix = prefix + ".ngram.shard"
    config = SimpleNamespace(
        split_ngram_parts=2,
        ngram_size=3,
        heads_per_ngram=1,
        ple_embed_dim=4,
        eos_token_ids=(99,),
    )
    ids = np.asarray([[1, 2, 99, 3, 4]], dtype=np.int32)
    embedding = MlxQwen4ExpNgramEmbedding(MlxNintModel(tensors), config, prefix)
    assert tensors.reads[embedding_prefix + ".0.weight"] == 1
    assert tensors.reads[embedding_prefix + ".1.weight"] == 1

    expected_ids = _reference_ngram_ids(
        ids,
        eos=99,
        multipliers=(3, 5, 7),
        vocab_sizes=(5, 7),
        offsets=(0, 8),
    )
    actual_ids = embedding._ids(ids, use_cache=False)
    np.testing.assert_array_equal(actual_ids, expected_ids)

    expected_values = global_embedding[expected_ids].reshape(1, 5, 4)
    actual_values = embedding(mx.array(ids), use_cache=False)
    mx.eval(actual_values)
    np.testing.assert_array_equal(np.asarray(actual_values), expected_values)

    embedding.reset_cache(1)
    first = embedding(mx.array(ids[:, :2]), use_cache=True)
    second = embedding(mx.array(ids[:, 2:]), use_cache=True)
    cached_values = mx.concatenate((first, second), axis=1)
    mx.eval(cached_values)
    np.testing.assert_array_equal(np.asarray(cached_values), expected_values)


def test_qwen4_ple_reject_rollback_restores_convolution_and_ngram_history() -> None:
    rng = np.random.default_rng(3804)
    prefix = "model.language_model.layers.0.ple"
    embedding_prefix = prefix + ".ple_embedding.ngram_embedding"
    hidden, streams = 8, 2
    width = hidden * streams
    tensors = {
        embedding_prefix + ".shard_0.weight": _random(rng, (8, 2)),
        embedding_prefix + ".shard_1.weight": _random(rng, (8, 2)),
        prefix + ".ple_embedding.layer_multipliers": np.asarray([3, 5, 7], dtype=np.int64),
        prefix + ".ple_embedding.ngram_heads_offsets": np.asarray([0, 8], dtype=np.int64),
        prefix + ".ple_embedding.ngram_heads_vocab_sizes": np.asarray([5, 7], dtype=np.int64),
        prefix + ".key_proj.weight": _random(rng, (width, 4)),
        prefix + ".value_proj.weight": _random(rng, (hidden, 4)),
        prefix + ".norm_key.weight": np.zeros((width,), dtype=np.float32),
        prefix + ".norm_query.weight": np.zeros((width,), dtype=np.float32),
        prefix + ".norm_conv.weight": np.zeros((width,), dtype=np.float32),
        prefix + ".conv1d.weight": _random(rng, (width, 1, 3)),
    }
    tensors = _canonical_tensors(tensors)
    prefix = "model.block.0.position_embedding"
    config = SimpleNamespace(
        split_ngram_parts=2,
        ngram_size=3,
        heads_per_ngram=1,
        ple_embed_dim=4,
        eos_token_ids=(15,),
        hidden_size=hidden,
        hc_count=streams,
        rms_norm_eps=1e-6,
    )
    ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    source = mx.array(_random(rng, (1, 5, width), 0.2)).astype(mx.float16)

    reference = MlxQwen4ExpPle(MlxNintModel(tensors), config, prefix)
    reference.reset_cache(1)
    reference(source[:, :2], ids[:, :2], use_cache=True)
    expected_confirmed = reference(source[:, 2:3], ids[:, 2:3], use_cache=True)
    expected_after_reject = reference(source[:, 4:5], ids[:, 4:5], use_cache=True)

    speculative = MlxQwen4ExpPle(MlxNintModel(tensors), config, prefix)
    speculative.reset_cache(1)
    speculative(source[:, :2], ids[:, :2], use_cache=True)
    verified = speculative(
        source[:, 2:4],
        ids[:, 2:4],
        use_cache=True,
        n_confirmed=1,
    )
    mx.eval(verified)
    speculative.rollback_speculative_cache()
    actual_after_reject = speculative(
        source[:, 4:5],
        ids[:, 4:5],
        use_cache=True,
    )
    mx.eval(expected_confirmed, expected_after_reject, actual_after_reject)

    np.testing.assert_allclose(
        np.asarray(verified[:, :1]),
        np.asarray(expected_confirmed),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_after_reject),
        np.asarray(expected_after_reject),
        rtol=1e-3,
        atol=1e-3,
    )


def test_qwen4_qsa_cached_chunks_match_full_sparse_sequence() -> None:
    rng = np.random.default_rng(3802)
    hidden = dimension = 128
    heads, kv_heads, index_heads = 2, 1, 2
    prefix = "model.language_model.layers.0.self_attn"
    tensors = {
        prefix + ".q_proj.weight": _random(rng, (heads * 2 * dimension, hidden)),
        prefix + ".k_proj.weight": _random(rng, (kv_heads * dimension, hidden)),
        prefix + ".v_proj.weight": _random(rng, (kv_heads * dimension, hidden)),
        prefix + ".o_proj.weight": _random(rng, (hidden, heads * dimension)),
        prefix + ".q_norm.weight": np.zeros((dimension,), dtype=np.float32),
        prefix + ".k_norm.weight": np.zeros((dimension,), dtype=np.float32),
        prefix + ".indexer.index_qk_proj.weight": _random(
            rng, ((index_heads + 1) * dimension, hidden)
        ),
        prefix + ".indexer.q_layernorm.weight": np.zeros((dimension,), dtype=np.float32),
        prefix + ".indexer.k_layernorm.weight": np.zeros((dimension,), dtype=np.float32),
    }
    tensors = _canonical_tensors(tensors)
    prefix = "model.block.0.attention"
    config = SimpleNamespace(
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=dimension,
        rotary_dim=64,
        rope_theta=10_000.0,
        rope_sections=(11, 11, 10),
        mrope_interleaved=True,
        rms_norm_eps=1e-6,
        indexer_n_heads=index_heads,
        indexer_kv_heads=1,
        indexer_head_dim=dimension,
        indexer_budget=4,
        indexer_compress_ratio=2,
    )
    source = mx.array(_random(rng, (1, 6, hidden), 0.2)).astype(mx.float16)
    positions = mx.arange(6, dtype=mx.int32)

    full = MlxQwen4ExpQsa(MlxNintModel(tensors), config, prefix, 16)
    expected = full(
        source,
        positions,
        positions,
        use_cache=False,
    )

    cached = MlxQwen4ExpQsa(MlxNintModel(tensors), config, prefix, 16)
    cached.reset_cache(1)
    first = cached(
        source[:, :2],
        positions[:2],
        positions[:2],
        use_cache=True,
    )
    second = cached(
        source[:, 2:],
        positions[2:],
        positions,
        use_cache=True,
    )
    actual = mx.concatenate((first, second), axis=1)
    mx.eval(expected, actual)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1.5e-2,
        atol=1.5e-2,
    )


def test_qwen4_mtp_cached_chunks_match_full_multistream_sequence() -> None:
    rng = np.random.default_rng(3803)
    hidden = dimension = 128
    streams, heads, kv_heads, experts, intermediate = 2, 2, 1, 2, 64
    width = streams * hidden
    prefix = "mtp.layers.0"
    tensors: dict[str, np.ndarray | NintMoeTensor] = {
        "model.language_model.embed_tokens.weight": _random(rng, (32, hidden)),
        "mtp.pre_fc_norm_embedding.weight": np.zeros((hidden,), dtype=np.float32),
        "mtp.pre_fc_norm_hidden.weight": np.zeros((width,), dtype=np.float32),
        "mtp.fc_embedding.weight": _random(rng, (hidden, hidden)),
        "mtp.fc_hidden.weight": _random(rng, (hidden, hidden)),
        prefix + ".self_attn.q_proj.weight": _random(
            rng, (heads * 2 * dimension, hidden)
        ),
        prefix + ".self_attn.k_proj.weight": _random(
            rng, (kv_heads * dimension, hidden)
        ),
        prefix + ".self_attn.v_proj.weight": _random(
            rng, (kv_heads * dimension, hidden)
        ),
        prefix + ".self_attn.o_proj.weight": _random(
            rng, (hidden, heads * dimension)
        ),
        prefix + ".self_attn.q_norm.weight": np.zeros((dimension,), dtype=np.float32),
        prefix + ".self_attn.k_norm.weight": np.zeros((dimension,), dtype=np.float32),
        prefix + ".self_attn.indexer.index_qk_proj.weight": _random(
            rng, ((heads + 1) * dimension, hidden)
        ),
        prefix + ".self_attn.indexer.q_layernorm.weight": np.zeros(
            (dimension,), dtype=np.float32
        ),
        prefix + ".self_attn.indexer.k_layernorm.weight": np.zeros(
            (dimension,), dtype=np.float32
        ),
        prefix + ".mlp.experts.gate_up_proj": _expert(
            _random(rng, (experts, 2 * intermediate, hidden))
        ),
        prefix + ".mlp.experts.down_proj": _expert(
            _random(rng, (experts, hidden, intermediate))
        ),
        prefix + ".mlp.gate.weight": _random(rng, (experts, hidden)),
        prefix + ".mlp.shared_expert.gate_proj.weight": _random(
            rng, (intermediate, hidden)
        ),
        prefix + ".mlp.shared_expert.up_proj.weight": _random(
            rng, (intermediate, hidden)
        ),
        prefix + ".mlp.shared_expert.down_proj.weight": _random(
            rng, (hidden, intermediate)
        ),
        prefix + ".mlp.shared_expert_gate.weight": _random(rng, (1, hidden)),
    }
    for site in ("attn_hyper_connection", "mlp_hyper_connection"):
        tensors[prefix + f".{site}.hc_norm.weight"] = np.zeros(
            (width,), dtype=np.float32
        )
        tensors[prefix + f".{site}.input_mix_weight_down.weight"] = _random(
            rng, (16, width)
        )
        tensors[prefix + f".{site}.input_mix_weight_up.weight"] = _random(
            rng, (width, 16)
        )
        tensors[prefix + f".{site}.block_inject_weight.weight"] = _random(
            rng, (streams, width)
        )
    final = "mtp.hyper_connection_mixer"
    tensors[final + ".hc_norm.weight"] = np.zeros((width,), dtype=np.float32)
    tensors[final + ".input_mix_weight_down.weight"] = _random(rng, (16, width))
    tensors[final + ".input_mix_weight_up.weight"] = _random(rng, (width, 16))
    tensors = _canonical_tensors(tensors)
    config = SimpleNamespace(
        mtp_num_hidden_layers=1,
        max_position_embeddings=16,
        hidden_size=hidden,
        hc_count=streams,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=dimension,
        rotary_dim=64,
        rope_theta=10_000.0,
        rope_sections=(11, 11, 10),
        mrope_interleaved=True,
        indexer_n_heads=heads,
        indexer_kv_heads=1,
        indexer_head_dim=dimension,
        indexer_budget=4,
        indexer_compress_ratio=2,
        num_experts=experts,
        num_experts_per_tok=1,
        moe_intermediate_size=intermediate,
        norm_topk_prob=True,
    )
    ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    previous = mx.array(_random(rng, (1, 6, width), 0.2)).astype(mx.float16)

    full = MlxQwen4ExpMtp(MlxNintModel(tensors), config, max_context=16)
    expected_sample, expected_multi = full(ids, previous, use_cache=False)

    cached = MlxQwen4ExpMtp(MlxNintModel(tensors), config, max_context=16)
    cached.reset_cache(1)
    first_sample, first_multi = cached(ids[:, :2], previous[:, :2], use_cache=True)
    second_sample, second_multi = cached(ids[:, 2:], previous[:, 2:], use_cache=True)
    actual_sample = mx.concatenate((first_sample, second_sample), axis=1)
    actual_multi = mx.concatenate((first_multi, second_multi), axis=1)
    mx.eval(expected_sample, expected_multi, actual_sample, actual_multi)

    np.testing.assert_allclose(
        np.asarray(actual_sample),
        np.asarray(expected_sample),
        rtol=3e-2,
        atol=3e-2,
    )
    np.testing.assert_allclose(
        np.asarray(actual_multi),
        np.asarray(expected_multi),
        rtol=3e-2,
        atol=3e-2,
    )
