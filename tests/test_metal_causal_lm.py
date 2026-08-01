"""End-to-end MLX causal-LM tests."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats import io  # noqa: E402
from mfq.formats.assets import MODEL_CONFIG_ASSET, model_config_asset  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.quantize.nint_quant import quantize  # noqa: E402
from mfq.runtime.mlx_causal_lm import (  # noqa: E402
    MlxCausalLM,
    MlxCausalLMConfig,
    MlxQwen35LinearAttentionBlock,
)
from mfq.tools.split_mfq import split_mfq  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _model() -> MlxCausalLM:
    rng = np.random.default_rng(20260728)
    config = MlxCausalLMConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_head_dim=4,
        rope_base=10_000.0,
    )
    spec = NintSpec(4, 24, 6)

    def weight(out: int, width: int):
        dense = rng.normal(0, 0.08, size=(out, width)).astype(np.float32)
        return quantize(dense, spec)

    tensors = {
        "token_embd.weight": weight(config.vocab_size, config.hidden_size),
        "blk.0.attn_norm.weight": np.ones(config.hidden_size, dtype=np.float32),
        "blk.0.attn_q.weight": weight(config.attention_size, config.hidden_size),
        "blk.0.attn_k.weight": weight(config.kv_size, config.hidden_size),
        "blk.0.attn_v.weight": weight(config.kv_size, config.hidden_size),
        "blk.0.attn_output.weight": weight(config.hidden_size, config.attention_size),
        "blk.0.ffn_norm.weight": np.ones(config.hidden_size, dtype=np.float32),
        "blk.0.ffn_gate.weight": weight(config.intermediate_size, config.hidden_size),
        "blk.0.ffn_up.weight": weight(config.intermediate_size, config.hidden_size),
        "blk.0.ffn_down.weight": weight(config.hidden_size, config.intermediate_size),
        "output_norm.weight": np.ones(config.hidden_size, dtype=np.float32),
        "output.weight": weight(config.vocab_size, config.hidden_size),
    }
    return MlxCausalLM(tensors, config)


def _linear_attention_model() -> MlxCausalLM:
    rng = np.random.default_rng(20260729)
    config = MlxCausalLMConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=64,
        attention_head_dim=32,
        layer_types=("linear_attention",),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
    )
    spec = NintSpec(4, 24, 6)

    def weight(out: int, width: int):
        dense = rng.normal(0, 0.04, size=(out, width)).astype(np.float32)
        return quantize(dense, spec)

    tensors = {
        "token_embd.weight": weight(config.vocab_size, config.hidden_size),
        "blk.0.attn_norm.weight": np.ones(config.hidden_size, dtype=np.float32),
        "blk.0.ssm_qkv.weight": weight(96, config.hidden_size),
        "blk.0.ssm_z.weight": weight(32, config.hidden_size),
        "blk.0.ssm_alpha.weight": weight(1, config.hidden_size),
        "blk.0.ssm_beta.weight": weight(1, config.hidden_size),
        "blk.0.ssm_conv1d.weight": rng.normal(
            0,
            0.05,
            size=(96, 1, 4),
        ).astype(np.float32),
        "blk.0.ssm_dt.bias": np.zeros(1, dtype=np.float32),
        "blk.0.ssm_a": np.full(1, -1.0, dtype=np.float32),
        "blk.0.ssm_norm.weight": np.ones(32, dtype=np.float32),
        "blk.0.ssm_out.weight": weight(config.hidden_size, 32),
        "blk.0.ffn_norm.weight": np.ones(config.hidden_size, dtype=np.float32),
        "blk.0.ffn_gate.weight": weight(config.intermediate_size, config.hidden_size),
        "blk.0.ffn_up.weight": weight(config.intermediate_size, config.hidden_size),
        "blk.0.ffn_down.weight": weight(config.hidden_size, config.intermediate_size),
        "output_norm.weight": np.ones(config.hidden_size, dtype=np.float32),
        "output.weight": weight(config.vocab_size, config.hidden_size),
    }
    return MlxCausalLM(tensors, config)


def test_mlx_causal_lm_prefill_shape_and_finite_logits():
    model = _model()
    assert model.layers[0].qkv.uses_grouped_kernel
    assert model.layers[0].ffn.gate_up.uses_grouped_kernel
    logits = _array(model(np.asarray([[1, 7, 3]], dtype=np.int32)))
    assert logits.shape == (1, 3, 32)
    assert np.isfinite(logits).all()


def test_mlx_causal_lm_cached_decode_matches_causal_prefill():
    ids = np.asarray([[2, 5, 8, 4]], dtype=np.int32)
    prefill_model = _model()
    expected = _array(prefill_model(ids, use_cache=False))

    decode_model = _model()
    pieces = []
    for token in range(ids.shape[1]):
        pieces.append(_array(decode_model(ids[:, token : token + 1], use_cache=True)))
    actual = np.concatenate(pieces, axis=1)
    np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-3)
    assert decode_model.layers[0].cache is not None
    assert decode_model.layers[0].cache.pos == ids.shape[1]

    decode_model.reset_cache(1)
    assert decode_model.layers[0].cache is not None
    assert decode_model.layers[0].cache.pos == 0


def test_mlx_causal_lm_greedy_generate():
    model = _model()
    prompt = np.asarray([[1, 2, 3]], dtype=np.int32)
    generated = _array(model.generate(prompt, 3))
    assert generated.shape == (1, 6)
    np.testing.assert_array_equal(generated[:, :3], prompt)
    assert np.all((generated[:, 3:] >= 0) & (generated[:, 3:] < 32))


def test_mlx_qwen35_linear_attention_cached_decode_matches_prefill():
    ids = np.asarray([[3, 7, 1, 9, 5]], dtype=np.int32)
    prefill_model = _linear_attention_model()
    expected = _array(prefill_model(ids, use_cache=False))

    decode_model = _linear_attention_model()
    assert isinstance(
        decode_model.layers[0],
        MlxQwen35LinearAttentionBlock,
    )
    pieces = [
        _array(decode_model(ids[:, token : token + 1], use_cache=True))
        for token in range(ids.shape[1])
    ]
    actual = np.concatenate(pieces, axis=1)
    np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-3)
    assert decode_model.layers[0].cache_pos == ids.shape[1]

    decode_model.reset_cache(1)
    assert decode_model.layers[0].cache_pos == 0


def test_mlx_qwen35_linear_attention_preserves_fp16_activation_dtype():
    model = _linear_attention_model()
    layer = model.layers[0]
    assert isinstance(layer, MlxQwen35LinearAttentionBlock)
    ids = mx.array([[3]], dtype=mx.int32)
    hidden = model.embedding(ids)
    output = layer(
        hidden,
        mx.array([0], dtype=mx.int32),
        use_cache=True,
    )
    mx.eval(output, layer.conv_state, layer.gdn_state)

    assert hidden.dtype == mx.float16
    assert output.dtype == mx.float16
    assert layer.conv_state is not None
    assert layer.conv_state.dtype == mx.float32
    assert layer.gdn_state is not None
    assert layer.gdn_state.dtype == mx.float32


def test_mlx_qwen35_config_adapter_selects_hybrid_layers():
    config = MlxCausalLMConfig.from_qwen35_hf_config(
        {
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 1024,
            "head_dim": 32,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
        }
    )
    assert config.layer_types == ("linear_attention", "full_attention")
    assert config.norm_weight_offset == 1.0
    assert config.linear_key_head_dim == 32


def test_mlx_causal_lm_loads_embedded_config_from_any_mfq_shard(
    tmp_path,
):
    source_model = _model()
    config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "head_dim": 4,
        "rope_parameters": {"rope_theta": 10_000.0},
        "layer_types": ["full_attention"],
    }
    config_asset = model_config_asset(config)
    source = tmp_path / "causal-assets.mfq"
    tensors = dict(source_model.model.tensors)
    tensors["blk.0.post_attention_norm.weight"] = tensors.pop(
        "blk.0.ffn_norm.weight"
    )
    io.save(
        source,
        FileHeader(version=2, model_arch="qwen35", num_tensors=0),
        {
            **tensors,
            MODEL_CONFIG_ASSET: config_asset.data,
        },
    )
    shards = split_mfq(
        source,
        tmp_path / "causal-sharded.mfq",
        split_max_tensors=3,
    )
    assert len(shards) > 1
    with MlxCausalLM.from_mfq(shards[-1]) as model:
        assert model.config.hidden_size == 16
        assert model.config.norm_weight_offset == 0.0
        assert model.config.linear_a_is_log is False
        logits = _array(model(np.asarray([[1, 2, 3]], dtype=np.int32)))
        assert logits.shape == (1, 3, 32)
        assert np.isfinite(logits).all()
