"""End-to-end decode-graph tests for Kimi-K3 on MLX/Metal."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats.tpq import TPQ_X, TpqPqTensor  # noqa: E402
from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.runtime.mlx_kimi_k3 import (  # noqa: E402
    MlxKimiK3,
    MlxKimiK3Config,
)
from mfq.runtime.mlx_moe import MlxRoutedSiTUFFN  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _random(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    scale: float = 0.08,
) -> np.ndarray:
    return rng.normal(0.0, scale, size=shape).astype(np.float16)


def _config(*, kda: bool) -> MlxKimiK3Config:
    return MlxKimiK3Config(
        n_layers=1,
        hidden=64,
        routed_hidden=32,
        n_experts=4,
        top_k=2,
        moe_inter=16,
        n_shared=1,
        inter_dense=48,
        first_dense_layers=1,
        vocab=23,
        rms_eps=1.0e-5,
        routed_scaling=1.0,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=4.0,
        latent_moe_use_norm=False,
        n_heads=2,
        head_dim=32,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=6,
        q_lora_rank=16,
        max_position_embeddings=64,
        attn_res_block_size=1,
        kda_layers=(0,) if kda else (),
        short_conv_kernel_size=4,
    )


def _base_tensors(
    rng: np.random.Generator,
    config: MlxKimiK3Config,
) -> dict[str, np.ndarray]:
    hidden = config.hidden
    layer = "language_model.model.layers.0"
    tensors = {
        "language_model.model.embed_tokens.weight": _random(
            rng,
            (config.vocab, hidden),
            0.2,
        ),
        "language_model.model.norm.weight": np.ones(
            (hidden,),
            dtype=np.float32,
        ),
        "language_model.model.output_attn_res_proj.weight": _random(
            rng,
            (hidden,),
            0.04,
        ).astype(np.float32),
        "language_model.model.output_attn_res_norm.weight": np.ones(
            (hidden,),
            dtype=np.float32,
        ),
        "language_model.lm_head.weight": _random(
            rng,
            (config.vocab, hidden),
        ),
        f"{layer}.input_layernorm.weight": np.ones(
            (hidden,),
            dtype=np.float32,
        ),
        f"{layer}.post_attention_layernorm.weight": np.ones(
            (hidden,),
            dtype=np.float32,
        ),
        f"{layer}.self_attention_res_proj.weight": _random(
            rng,
            (hidden,),
            0.04,
        ).astype(np.float32),
        f"{layer}.self_attention_res_norm.weight": np.ones(
            (hidden,),
            dtype=np.float32,
        ),
        f"{layer}.mlp_res_proj.weight": _random(
            rng,
            (hidden,),
            0.04,
        ).astype(np.float32),
        f"{layer}.mlp_res_norm.weight": np.ones(
            (hidden,),
            dtype=np.float32,
        ),
        f"{layer}.mlp.gate_proj.weight": _random(
            rng,
            (config.inter_dense, hidden),
        ),
        f"{layer}.mlp.up_proj.weight": _random(
            rng,
            (config.inter_dense, hidden),
        ),
        f"{layer}.mlp.down_proj.weight": _random(
            rng,
            (hidden, config.inter_dense),
        ),
    }
    return tensors


def _kda_tensors(
    rng: np.random.Generator,
    config: MlxKimiK3Config,
) -> dict[str, np.ndarray]:
    prefix = "language_model.model.layers.0.self_attn"
    width = config.n_heads * config.head_dim
    rank = 9
    result = {
        f"{prefix}.{name}_proj.weight": _random(
            rng,
            (width, config.hidden),
        )
        for name in ("q", "k", "v", "g")
    }
    result.update(
        {
            f"{prefix}.f_a_proj.weight": _random(
                rng,
                (rank, config.hidden),
            ),
            f"{prefix}.b_proj.weight": _random(
                rng,
                (config.n_heads, config.hidden),
            ),
            f"{prefix}.f_b_proj.weight": _random(
                rng,
                (width, rank),
            ),
            f"{prefix}.A_log": _random(
                rng,
                (config.n_heads,),
            ).astype(np.float32),
            f"{prefix}.dt_bias": _random(
                rng,
                (width,),
            ).astype(np.float32),
            f"{prefix}.o_norm.weight": np.ones(
                (config.head_dim,),
                dtype=np.float32,
            ),
            f"{prefix}.o_proj.weight": _random(
                rng,
                (config.hidden, width),
            ),
        }
    )
    for name in ("q", "k", "v"):
        result[f"{prefix}.{name}_conv1d.weight"] = _random(
            rng,
            (width, 1, config.short_conv_kernel_size),
        ).astype(np.float32)
    return result


def _mla_tensors(
    rng: np.random.Generator,
    config: MlxKimiK3Config,
) -> dict[str, np.ndarray]:
    prefix = "language_model.model.layers.0.self_attn"
    query_width = config.n_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)
    value_width = config.n_heads * config.v_head_dim
    return {
        f"{prefix}.q_a_proj.weight": _random(
            rng,
            (config.q_lora_rank, config.hidden),
        ),
        f"{prefix}.q_a_layernorm.weight": np.ones(
            (config.q_lora_rank,),
            dtype=np.float32,
        ),
        f"{prefix}.q_b_proj.weight": _random(
            rng,
            (query_width, config.q_lora_rank),
        ),
        f"{prefix}.kv_a_proj_with_mqa.weight": _random(
            rng,
            (
                config.kv_lora_rank + config.qk_rope_head_dim,
                config.hidden,
            ),
        ),
        f"{prefix}.kv_a_layernorm.weight": np.ones(
            (config.kv_lora_rank,),
            dtype=np.float32,
        ),
        f"{prefix}.kv_b_proj.weight": _random(
            rng,
            (
                config.n_heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            ),
        ),
        f"{prefix}.g_proj.weight": _random(
            rng,
            (value_width, config.hidden),
        ),
        f"{prefix}.o_proj.weight": _random(
            rng,
            (config.hidden, value_width),
        ),
    }


@pytest.mark.parametrize("kda", [True, False])
def test_kimi_graph_batched_prefill_matches_cached_token_steps(kda: bool):
    rng = np.random.default_rng(100 + kda)
    config = _config(kda=kda)
    tensors = _base_tensors(rng, config)
    tensors.update(_kda_tensors(rng, config) if kda else _mla_tensors(rng, config))
    model = MlxKimiK3(tensors, config)
    ids = np.array([[2, 5, 7]], dtype=np.int32)
    expected = _array(model(ids, use_cache=False))

    model.reset_cache(1)
    pieces = [
        _array(model(ids[:, token : token + 1], use_cache=True)) for token in range(ids.shape[1])
    ]
    actual = np.concatenate(pieces, axis=1)
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)
    assert model.position == ids.shape[1]
    if not kda:
        assert model.layers[0].attention.latent_cache.shape[1] == ids.shape[1]


def _pq_pool(
    rng: np.random.Generator,
    expert_ids: np.ndarray,
    rows_per_expert: int,
    columns: int,
) -> tuple[NintMoePool, np.ndarray]:
    rows = int(expert_ids.size) * rows_per_expert
    codebook = rng.normal(
        0.0,
        0.15,
        size=(TPQ_X.codebook_entries, TPQ_X.vector_size),
    ).astype(np.float32)
    indices = rng.integers(
        0,
        TPQ_X.codebook_entries,
        size=(rows, columns // TPQ_X.vector_size),
        dtype=np.uint8,
    )
    tensor = TpqPqTensor(
        spec=TPQ_X,
        shape=(rows, columns),
        axis=0,
        neuron_len=columns,
        indices=indices,
        codebook=codebook,
    )
    return (
        NintMoePool(expert_ids=expert_ids, tensor=tensor),
        codebook[indices].reshape(rows, columns),
    )


def test_kimi_routed_situ_executes_tpq_experts_directly():
    rng = np.random.default_rng(401)
    experts, routed, intermediate = 4, 16, 8
    ids_a = np.array([0, 2], dtype=np.int32)
    ids_b = np.array([1, 3], dtype=np.int32)
    gu_a, gu_dense_a = _pq_pool(
        rng,
        ids_a,
        2 * intermediate,
        routed,
    )
    gu_b, gu_dense_b = _pq_pool(
        rng,
        ids_b,
        2 * intermediate,
        routed,
    )
    down_a, down_dense_a = _pq_pool(
        rng,
        ids_a,
        routed,
        intermediate,
    )
    down_b, down_dense_b = _pq_pool(
        rng,
        ids_b,
        routed,
        intermediate,
    )
    gate_up = NintMoeTensor(
        shape=(experts, 2 * intermediate, routed),
        pools=(gu_a, gu_b),
    )
    down = NintMoeTensor(
        shape=(experts, routed, intermediate),
        pools=(down_a, down_b),
    )
    module = MlxRoutedSiTUFFN(
        gate_up,
        down,
        beta=4.0,
        linear_beta=4.0,
    )
    assert module.gate_up.uses_grouped_kernel
    assert module.down.uses_grouped_kernel
    selected = np.array([[3, 0], [2, 1]], dtype=np.int32)
    weights = np.array([[0.6, 0.4], [0.7, 0.3]], dtype=np.float32)
    value = rng.normal(0.0, 0.2, size=(2, routed)).astype(np.float16)

    gu_by_expert: dict[int, np.ndarray] = {}
    down_by_expert: dict[int, np.ndarray] = {}
    for pool_ids, gu_dense, down_dense in (
        (ids_a, gu_dense_a, down_dense_a),
        (ids_b, gu_dense_b, down_dense_b),
    ):
        for local, expert in enumerate(pool_ids.tolist()):
            gu_by_expert[expert] = gu_dense[
                local * 2 * intermediate : (local + 1) * 2 * intermediate
            ]
            down_by_expert[expert] = down_dense[local * routed : (local + 1) * routed]
    expected = np.zeros((2, routed), dtype=np.float32)
    for token in range(2):
        for route in range(2):
            expert = int(selected[token, route])
            gate_up_value = value[token].astype(np.float32) @ gu_by_expert[expert].T
            gate, up = np.split(gate_up_value, 2)
            hidden = 4.0 * np.tanh(gate / 4.0) / (1.0 + np.exp(-gate)) * (4.0 * np.tanh(up / 4.0))
            expected[token] += (hidden @ down_by_expert[expert].T) * weights[token, route]
    actual = module(value, selected, weights)
    np.testing.assert_allclose(
        _array(actual),
        expected.astype(np.float16),
        rtol=8e-3,
        atol=4e-3,
    )
