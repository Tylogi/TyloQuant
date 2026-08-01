"""Apple-silicon tests for Gemma4 fused Metal kernels and runtime."""

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
from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.kernels.metal.gemma4 import (  # noqa: E402
    gemma4_attn_residual_pre_norms,
    gemma4_ffn_merge,
)
from mfq.kernels.metal.ops import residual_add, rms_norm  # noqa: E402
from mfq.quantize.nint_quant import quantize  # noqa: E402
from mfq.runtime.mlx_gemma4 import (  # noqa: E402
    MlxGemma4,
    MlxGemma4Config,
)
from mfq.tools.split_mfq import split_mfq  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


@pytest.mark.parametrize("rows,width", [(1, 96), (7, 2816)])
def test_gemma4_fused_pre_norms_match_materialized_path(rows: int, width: int):
    rng = np.random.default_rng(101 + rows)
    eps = 1e-6
    residual = rng.normal(0.0, 0.4, (rows, width)).astype(np.float16)
    attention = rng.normal(0.0, 0.4, (rows, width)).astype(np.float16)
    weights = tuple(rng.normal(1.0, 0.1, width).astype(np.float32) for _ in range(4))

    attention_post = rms_norm(attention, weights[0], eps)
    residual_ref = residual_add(residual, attention_post)
    dense_ref = rms_norm(residual_ref, weights[1], eps)
    router_ref = rms_norm(
        residual_ref.astype(mx.float32),
        weights[2],
        eps,
    )
    moe_ref = rms_norm(residual_ref, weights[3], eps)

    actual = gemma4_attn_residual_pre_norms(
        residual,
        attention,
        *weights,
        eps,
    )
    expected = (residual_ref, dense_ref, router_ref, moe_ref)
    for result, reference in zip(actual, expected, strict=True):
        reference_array = _array(reference)
        # The fused reduction uses a different Metal scheduling boundary than
        # four separately dispatched norms; allow at most a few FP16 ULPs.
        tolerance = 4e-3 if reference_array.dtype == np.float16 else 3e-3
        np.testing.assert_allclose(
            _array(result),
            reference_array,
            rtol=tolerance,
            atol=tolerance,
        )
    assert actual[2].dtype == mx.float32


@pytest.mark.parametrize("rows,width", [(1, 128), (9, 2816)])
def test_gemma4_fused_ffn_merge_matches_materialized_path(rows: int, width: int):
    rng = np.random.default_rng(211 + rows)
    eps = 1e-6
    dense = rng.normal(0.0, 0.3, (rows, width)).astype(np.float16)
    moe = rng.normal(0.0, 0.3, (rows, width)).astype(np.float16)
    residual = rng.normal(0.0, 0.3, (rows, width)).astype(np.float16)
    weights = tuple(rng.normal(1.0, 0.1, width).astype(np.float32) for _ in range(3))
    layer_scale = np.asarray([0.9375], dtype=np.float16)

    dense_post = rms_norm(dense, weights[0], eps)
    moe_post = rms_norm(moe, weights[1], eps)
    combined = residual_add(dense_post, moe_post)
    post = rms_norm(combined, weights[2], eps)
    residual_sum = residual_add(residual, post)
    expected = residual_sum * float(layer_scale[0])

    actual = gemma4_ffn_merge(
        dense,
        moe,
        residual,
        *weights,
        layer_scale,
        eps,
    )
    assert actual.dtype == mx.float16
    np.testing.assert_allclose(
        _array(actual),
        _array(expected),
        rtol=2e-3,
        atol=4e-3,
    )


def test_gemma4_fused_kernels_validate_shapes():
    activation = np.ones((2, 32), dtype=np.float16)
    weight = np.ones(32, dtype=np.float32)
    with pytest.raises(ValueError, match="shapes must match"):
        gemma4_attn_residual_pre_norms(
            activation,
            activation[:, :-1],
            weight,
            weight,
            weight,
            weight,
        )
    with pytest.raises(ValueError, match="must be scalar"):
        gemma4_ffn_merge(
            activation,
            activation,
            activation,
            weight,
            weight,
            weight,
            np.ones(2, dtype=np.float32),
        )


def _gemma4_model() -> MlxGemma4:
    rng = np.random.default_rng(404)
    config = MlxGemma4Config(
        vocab_size=40,
        hidden_size=32,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
        layer_types=("sliding_attention", "full_attention"),
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        full_rope_base=100_000.0,
        sliding_rope_base=10_000.0,
        full_partial_rotary_factor=0.5,
        attention_k_eq_v=True,
        tie_word_embeddings=True,
        final_logit_softcapping=4.0,
        eos_token_id=(2,),
    )
    spec = NintSpec(4, 24, 6)

    def weight(out: int, width: int):
        value = rng.normal(0.0, 0.04, (out, width)).astype(np.float32)
        return quantize(value, spec)

    def experts(out: int, width: int) -> NintMoeTensor:
        dense = rng.normal(
            0.0,
            0.035,
            (config.num_experts, out, width),
        ).astype(np.float32)
        packed = quantize(
            dense.reshape((config.num_experts * out, width)),
            spec,
        )
        return NintMoeTensor(
            dense.shape,
            (
                NintMoePool(
                    np.arange(config.num_experts, dtype=np.int32),
                    packed,
                ),
            ),
        )

    tensors = {
        "model.language_model.embed_tokens.weight": weight(
            config.vocab_size,
            config.hidden_size,
        ),
        "model.language_model.norm.weight": np.ones(
            config.hidden_size,
            dtype=np.float32,
        ),
    }
    for layer, layer_type in enumerate(config.layer_types):
        prefix = f"model.language_model.layers.{layer}"
        attention_prefix = f"{prefix}.self_attn"
        head_dim = config.head_dim if layer_type == "sliding_attention" else config.global_head_dim
        kv_heads = (
            config.num_key_value_heads
            if layer_type == "sliding_attention"
            else config.num_global_key_value_heads
        )
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": np.ones(
                    config.hidden_size,
                    dtype=np.float32,
                ),
                f"{prefix}.post_attention_layernorm.weight": np.ones(
                    config.hidden_size,
                    dtype=np.float32,
                ),
                f"{attention_prefix}.q_norm.weight": np.ones(
                    head_dim,
                    dtype=np.float32,
                ),
                f"{attention_prefix}.k_norm.weight": np.ones(
                    head_dim,
                    dtype=np.float32,
                ),
                f"{attention_prefix}.q_proj.weight": weight(
                    config.num_attention_heads * head_dim,
                    config.hidden_size,
                ),
                f"{attention_prefix}.k_proj.weight": weight(
                    kv_heads * head_dim,
                    config.hidden_size,
                ),
                f"{attention_prefix}.o_proj.weight": weight(
                    config.hidden_size,
                    config.num_attention_heads * head_dim,
                ),
                f"{prefix}.pre_feedforward_layernorm.weight": np.ones(
                    config.hidden_size,
                    dtype=np.float32,
                ),
                f"{prefix}.post_feedforward_layernorm.weight": np.ones(
                    config.hidden_size,
                    dtype=np.float32,
                ),
                f"{prefix}.post_feedforward_layernorm_1.weight": np.ones(
                    config.hidden_size,
                    dtype=np.float32,
                ),
                f"{prefix}.pre_feedforward_layernorm_2.weight": np.ones(
                    config.hidden_size,
                    dtype=np.float32,
                ),
                f"{prefix}.post_feedforward_layernorm_2.weight": np.ones(
                    config.hidden_size,
                    dtype=np.float32,
                ),
                f"{prefix}.layer_scalar": np.asarray(
                    [0.97],
                    dtype=np.float32,
                ),
                f"{prefix}.mlp.gate_proj.weight": weight(
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.up_proj.weight": weight(
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.down_proj.weight": weight(
                    config.hidden_size,
                    config.intermediate_size,
                ),
                f"{prefix}.experts.gate_up_proj": experts(
                    2 * config.moe_intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.experts.down_proj": experts(
                    config.hidden_size,
                    config.moe_intermediate_size,
                ),
                f"{prefix}.router.proj.weight": rng.normal(
                    0.0,
                    0.03,
                    (config.num_experts, config.hidden_size),
                ).astype(np.float32),
                f"{prefix}.router.scale": np.ones(
                    config.hidden_size,
                    dtype=np.float32,
                ),
                f"{prefix}.router.per_expert_scale": np.linspace(
                    0.9,
                    1.1,
                    config.num_experts,
                    dtype=np.float32,
                ),
            }
        )
        if layer_type == "sliding_attention" or not config.attention_k_eq_v:
            tensors[f"{attention_prefix}.v_proj.weight"] = weight(
                kv_heads * head_dim,
                config.hidden_size,
            )
    return MlxGemma4(tensors, config)


def test_gemma4_full_graph_prefill_decode_and_generation():
    model = _gemma4_model()
    prompt = np.asarray([[1, 5, 7, 9, 11, 13]], dtype=np.int32)
    uncached = _array(model(prompt))
    cached = _array(model.prefill(prompt))
    assert cached.shape == (1, prompt.shape[1], model.config.vocab_size)
    np.testing.assert_allclose(cached, uncached, rtol=8e-3, atol=8e-3)
    assert model.cache_position == prompt.shape[1]
    for layer in model.layers:
        assert layer.cache_position == prompt.shape[1]

    decoded = _array(model.decode(np.asarray([[17]], dtype=np.int32)))
    assert decoded.shape == (1, 1, model.config.vocab_size)
    assert np.isfinite(decoded).all()
    assert np.max(np.abs(decoded)) <= model.config.final_logit_softcapping

    generated = _array(
        model.generate(
            prompt[:, :3],
            3,
            temperature=0.0,
            eos_token_id=(),
        )
    )
    assert generated.shape == (1, 6)


def test_gemma4_config_normalizes_nested_hf_values():
    config = MlxGemma4Config.from_hf_config(
        {
            "text_config": {
                "vocab_size": 128,
                "hidden_size": 72,
                "intermediate_size": 144,
                "num_hidden_layers": 2,
                "num_attention_heads": 6,
                "num_key_value_heads": 2,
                "num_global_key_value_heads": 3,
                "head_dim": 12,
                "global_head_dim": 16,
                "max_position_embeddings": 4096,
                "sliding_window": 512,
                "layer_types": [
                    "sliding_attention",
                    "full_attention",
                ],
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 96,
                "full_attention": {
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {"rope_theta": 10_000.0},
                "attention_k_eq_v": True,
                "final_logit_softcapping": 30.0,
                "eos_token_id": [1, 2],
            }
        }
    )
    assert config.num_global_key_value_heads == 3
    assert config.full_partial_rotary_factor == 0.5
    assert config.attention_k_eq_v
    assert config.eos_token_id == (1, 2)
    assert config.embed_scale == _bf16_reference(np.sqrt(72.0))


def test_gemma4_loads_embedded_config_from_any_mfq_shard(tmp_path):
    source_model = _gemma4_model()
    config = source_model.config
    config_json = {
        "model_type": "gemma4_text",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "num_global_key_value_heads": config.num_global_key_value_heads,
        "head_dim": config.head_dim,
        "global_head_dim": config.global_head_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "sliding_window": config.sliding_window,
        "layer_types": list(config.layer_types),
        "num_experts": config.num_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "moe_intermediate_size": config.moe_intermediate_size,
        "full_attention": {
            "rope_theta": config.full_rope_base,
            "partial_rotary_factor": config.full_partial_rotary_factor,
        },
        "sliding_attention": {
            "rope_theta": config.sliding_rope_base,
        },
        "rms_norm_eps": config.rms_norm_eps,
        "attention_k_eq_v": config.attention_k_eq_v,
        "tie_word_embeddings": config.tie_word_embeddings,
        "final_logit_softcapping": config.final_logit_softcapping,
        "eos_token_id": list(config.eos_token_id),
    }
    asset = model_config_asset(config_json)
    source = tmp_path / "gemma4-assets.mfq"
    io.save(
        source,
        FileHeader(version=2, model_arch="gemma4", num_tensors=0),
        {
            **source_model.model.tensors,
            MODEL_CONFIG_ASSET: asset.data,
        },
    )
    shards = split_mfq(
        source,
        tmp_path / "gemma4-sharded.mfq",
        split_max_tensors=10,
    )
    assert len(shards) > 1
    with MlxGemma4.from_mfq(shards[-1]) as loaded:
        assert loaded.config.layer_types == config.layer_types
        assert loaded.config.attention_k_eq_v
        logits = _array(loaded(np.asarray([[1, 3, 5]], dtype=np.int32)))
        assert logits.shape == (1, 3, config.vocab_size)
        assert np.isfinite(logits).all()


def _bf16_reference(value: float) -> float:
    bits = np.asarray([value], dtype=np.float32).view(np.uint32)
    bits += np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return float((bits & np.uint32(0xFFFF0000)).view(np.float32)[0])
