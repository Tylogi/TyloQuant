"""End-to-end DeepSeek-V4 native-MFQ graph and cache tests."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats import io  # noqa: E402
from mfq.formats.tpq import TPQ_V, TpqPqTensor  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.quantize.nint_quant import quantize  # noqa: E402
from mfq.runtime import load_tpq_model  # noqa: E402
from mfq.runtime.mlx_deepseek_v4 import (  # noqa: E402
    MlxDeepseekV4,
    MlxDeepseekV4Config,
    MlxDeepseekV4Names,
    MlxDeepseekV4PoolState,
)
from mfq.tools.split_mfq import split_mfq  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _config(*, ratios=(0,)) -> dict:
    return {
        "n_layers": len(ratios),
        "hidden": 8,
        "n_experts": 2,
        "top_k": 1,
        "moe_inter": 4,
        "n_shared": 1,
        "n_heads": 2,
        "head_dim": 4,
        "q_lora_rank": 4,
        "o_lora_rank": 2,
        "o_groups": 2,
        "kv_dim": 4,
        "qk_rope_head_dim": 2,
        "n_kv_heads": 1,
        "vocab": 17,
        "rms_eps": 1e-6,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling": 1.5,
        "swiglu_limit": 10.0,
        "n_hash_layers": 0,
        "sliding_window": 4,
        "rope_theta": 10_000.0,
        "rope_scaling": {},
        "eos_token_id": [2],
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "max_position_embeddings": 32,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "compress_rope_theta": 160_000.0,
        "compress_ratios": list(ratios),
    }


def _dense(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    *,
    scale: float = 0.02,
) -> np.ndarray:
    return rng.normal(0.0, scale, size=shape).astype(np.float16)


def _expert(
    rng: np.random.Generator,
    *,
    experts: int,
    output: int,
    width: int,
) -> NintMoeTensor:
    tensor = quantize(
        rng.normal(
            0.0,
            0.03,
            size=(experts * output, width),
        ).astype(np.float32),
        NintSpec(4, 8, 6),
    )
    return NintMoeTensor(
        shape=(experts, output, width),
        pools=(
            NintMoePool(
                expert_ids=np.arange(experts, dtype=np.int32),
                tensor=tensor,
            ),
        ),
    )


def _tpq_expert(
    rng: np.random.Generator,
    *,
    experts: int,
    output: int,
    width: int,
) -> NintMoeTensor:
    rows = experts * output
    tensor = TpqPqTensor(
        spec=TPQ_V,
        shape=(rows, width),
        axis=0,
        neuron_len=width,
        indices=rng.integers(
            0,
            TPQ_V.codebook_entries,
            size=(rows, width // TPQ_V.vector_size),
            dtype=np.uint8,
        ),
        codebook=rng.normal(
            0.0,
            0.03,
            size=(TPQ_V.codebook_entries, TPQ_V.vector_size),
        ).astype(np.float32),
    )
    return NintMoeTensor(
        shape=(experts, output, width),
        pools=(
            NintMoePool(
                expert_ids=np.arange(experts, dtype=np.int32),
                tensor=tensor,
            ),
        ),
    )


def _write_tiny_model(
    path,
    *,
    ratio: int = 0,
    streamed_experts: bool = False,
) -> None:
    rng = np.random.default_rng(20260840)
    config = _config(ratios=(ratio,))
    hidden = config["hidden"]
    heads = config["n_heads"]
    head_dim = config["head_dim"]
    groups = config["o_groups"]
    o_rank = config["o_lora_rank"]
    q_rank = config["q_lora_rank"]
    intermediate = config["moe_inter"]
    tensors: dict[str, object] = {
        "embed.weight": _dense(rng, (config["vocab"], hidden)),
        "norm.weight": np.ones((hidden,), dtype=np.float16),
        "head.weight": _dense(rng, (config["vocab"], hidden)),
        "hc_head_fn": _dense(rng, (4, 4 * hidden)),
        "hc_head_base": np.zeros((4,), dtype=np.float32),
        "hc_head_scale": np.ones((1,), dtype=np.float32),
    }
    prefix = "layers.0"
    expert_factory = _tpq_expert if streamed_experts else _expert
    tensors.update(
        {
            f"{prefix}.attn.wq_a.weight": _dense(rng, (q_rank, hidden)),
            f"{prefix}.attn.q_norm.weight": np.ones((q_rank,), dtype=np.float16),
            f"{prefix}.attn.wq_b.weight": _dense(rng, (heads * head_dim, q_rank)),
            f"{prefix}.attn.wkv.weight": _dense(rng, (head_dim, hidden)),
            f"{prefix}.attn.kv_norm.weight": np.ones((head_dim,), dtype=np.float16),
            f"{prefix}.attn.attn_sink": np.zeros((heads,), dtype=np.float32),
            f"{prefix}.attn.wo_a.weight": _dense(
                rng,
                (groups * o_rank, heads * head_dim // groups),
            ),
            f"{prefix}.attn.wo_b.weight": _dense(rng, (hidden, groups * o_rank)),
            f"{prefix}.attn_norm.weight": np.ones((hidden,), dtype=np.float16),
            f"{prefix}.ffn_norm.weight": np.ones((hidden,), dtype=np.float16),
            f"{prefix}.ffn.gate.weight": _dense(rng, (config["n_experts"], hidden)),
            f"{prefix}.ffn.gate.bias": np.zeros((config["n_experts"],), dtype=np.float32),
            f"{prefix}.ffn.shared_experts.w1.weight": _dense(rng, (intermediate, hidden)),
            f"{prefix}.ffn.shared_experts.w3.weight": _dense(rng, (intermediate, hidden)),
            f"{prefix}.ffn.shared_experts.w2.weight": _dense(rng, (hidden, intermediate)),
            f"{prefix}.hc_attn_fn": _dense(rng, (24, 4 * hidden)),
            f"{prefix}.hc_attn_base": np.zeros((24,), dtype=np.float32),
            f"{prefix}.hc_attn_scale": np.ones((3,), dtype=np.float32),
            f"{prefix}.hc_ffn_fn": _dense(rng, (24, 4 * hidden)),
            f"{prefix}.hc_ffn_base": np.zeros((24,), dtype=np.float32),
            f"{prefix}.hc_ffn_scale": np.ones((3,), dtype=np.float32),
            f"{prefix}.ffn.experts.gate_up.weight": expert_factory(
                rng,
                experts=config["n_experts"],
                output=2 * intermediate,
                width=hidden,
            ),
            f"{prefix}.ffn.experts.down.weight": expert_factory(
                rng,
                experts=config["n_experts"],
                output=hidden,
                width=intermediate,
            ),
        }
    )
    if ratio:
        compressor_width = head_dim * (2 if ratio == 4 else 1)
        tensors.update(
            {
                f"{prefix}.attn.compressor.wkv.weight": _dense(rng, (compressor_width, hidden)),
                f"{prefix}.attn.compressor.wgate.weight": _dense(rng, (compressor_width, hidden)),
                f"{prefix}.attn.compressor.ape": _dense(rng, (ratio, compressor_width)).astype(
                    np.float32
                ),
                f"{prefix}.attn.compressor.norm.weight": np.ones((head_dim,), dtype=np.float16),
            }
        )
    if ratio == 4:
        index_heads = config["index_n_heads"]
        index_dim = config["index_head_dim"]
        index_compressor_width = 2 * index_dim
        tensors.update(
            {
                f"{prefix}.attn.indexer.wq_b.weight": _dense(
                    rng, (index_heads * index_dim, q_rank)
                ),
                f"{prefix}.attn.indexer.weights_proj.weight": _dense(rng, (index_heads, hidden)),
                f"{prefix}.attn.indexer.compressor.wkv.weight": _dense(
                    rng, (index_compressor_width, hidden)
                ),
                f"{prefix}.attn.indexer.compressor.wgate.weight": _dense(
                    rng, (index_compressor_width, hidden)
                ),
                f"{prefix}.attn.indexer.compressor.ape": _dense(
                    rng, (ratio, index_compressor_width)
                ).astype(np.float32),
                f"{prefix}.attn.indexer.compressor.norm.weight": np.ones(
                    (index_dim,), dtype=np.float16
                ),
            }
        )
    manifest = {
        "format": "tpq-1",
        "config": config,
        "quant": {},
        "expert_files": {"0": "experts.L0.safetensors"},
        "tiers_per_layer": {"0": "xx"},
    }
    io.save(
        path,
        FileHeader(
            version=2,
            model_arch="deepseek_v4-tpq-mfq",
            num_tensors=len(tensors),
            extra={
                "source_format": "tpq-1",
                "tpq_manifest": manifest,
            },
        ),
        tensors,
    )


def _write_official_shape_model(path, *, ratio: int = 0) -> None:
    rng = np.random.default_rng(20260842)
    config = _config(ratios=(ratio,))
    config.update(
        {
            "hidden": 4096,
            "n_heads": 64,
            "head_dim": 512,
            "q_lora_rank": 4,
            "o_lora_rank": 1,
            "o_groups": 8,
            "kv_dim": 512,
            "qk_rope_head_dim": 64,
            "moe_inter": 8,
            "sliding_window": 4,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
        }
    )
    hidden = config["hidden"]
    intermediate = config["moe_inter"]
    prefix = "layers.0"
    tensors: dict[str, object] = {
        "embed.weight": _dense(rng, (config["vocab"], hidden), scale=0.005),
        "norm.weight": np.ones((hidden,), dtype=np.float16),
        "head.weight": _dense(rng, (config["vocab"], hidden), scale=0.005),
        "hc_head_fn": _dense(rng, (4, 4 * hidden), scale=0.005),
        "hc_head_base": np.zeros((4,), dtype=np.float32),
        "hc_head_scale": np.ones((1,), dtype=np.float32),
        f"{prefix}.attn.wq_a.weight": _dense(rng, (config["q_lora_rank"], hidden), scale=0.005),
        f"{prefix}.attn.q_norm.weight": np.ones((config["q_lora_rank"],), dtype=np.float16),
        f"{prefix}.attn.wq_b.weight": _dense(
            rng,
            (
                config["n_heads"] * config["head_dim"],
                config["q_lora_rank"],
            ),
            scale=0.005,
        ),
        f"{prefix}.attn.wkv.weight": _dense(rng, (config["head_dim"], hidden), scale=0.005),
        f"{prefix}.attn.kv_norm.weight": np.ones((config["head_dim"],), dtype=np.float16),
        f"{prefix}.attn.attn_sink": np.zeros((config["n_heads"],), dtype=np.float32),
        f"{prefix}.attn.wo_a.weight": _dense(
            rng,
            (
                config["o_groups"] * config["o_lora_rank"],
                config["n_heads"] * config["head_dim"] // config["o_groups"],
            ),
            scale=0.005,
        ),
        f"{prefix}.attn.wo_b.weight": _dense(
            rng,
            (
                hidden,
                config["o_groups"] * config["o_lora_rank"],
            ),
            scale=0.005,
        ),
        f"{prefix}.attn_norm.weight": np.ones((hidden,), dtype=np.float16),
        f"{prefix}.ffn_norm.weight": np.ones((hidden,), dtype=np.float16),
        f"{prefix}.ffn.gate.weight": _dense(rng, (config["n_experts"], hidden), scale=0.005),
        f"{prefix}.ffn.gate.bias": np.zeros((config["n_experts"],), dtype=np.float32),
        f"{prefix}.ffn.shared_experts.w1.weight": _dense(rng, (intermediate, hidden), scale=0.005),
        f"{prefix}.ffn.shared_experts.w3.weight": _dense(rng, (intermediate, hidden), scale=0.005),
        f"{prefix}.ffn.shared_experts.w2.weight": _dense(rng, (hidden, intermediate), scale=0.005),
        f"{prefix}.hc_attn_fn": _dense(rng, (24, 4 * hidden), scale=0.005),
        f"{prefix}.hc_attn_base": np.zeros((24,), dtype=np.float32),
        f"{prefix}.hc_attn_scale": np.ones((3,), dtype=np.float32),
        f"{prefix}.hc_ffn_fn": _dense(rng, (24, 4 * hidden), scale=0.005),
        f"{prefix}.hc_ffn_base": np.zeros((24,), dtype=np.float32),
        f"{prefix}.hc_ffn_scale": np.ones((3,), dtype=np.float32),
        f"{prefix}.ffn.experts.gate_up.weight": _expert(
            rng,
            experts=config["n_experts"],
            output=2 * intermediate,
            width=hidden,
        ),
        f"{prefix}.ffn.experts.down.weight": _expert(
            rng,
            experts=config["n_experts"],
            output=hidden,
            width=intermediate,
        ),
    }
    if ratio:
        compressor_width = config["head_dim"] * (2 if ratio == 4 else 1)
        tensors.update(
            {
                f"{prefix}.attn.compressor.wkv.weight": _dense(
                    rng, (compressor_width, hidden), scale=0.005
                ),
                f"{prefix}.attn.compressor.wgate.weight": _dense(
                    rng, (compressor_width, hidden), scale=0.005
                ),
                f"{prefix}.attn.compressor.ape": _dense(
                    rng, (ratio, compressor_width), scale=0.005
                ).astype(np.float32),
                f"{prefix}.attn.compressor.norm.weight": np.ones(
                    (config["head_dim"],), dtype=np.float16
                ),
            }
        )
    if ratio == 4:
        index_width = 2 * config["index_head_dim"]
        tensors.update(
            {
                f"{prefix}.attn.indexer.wq_b.weight": _dense(
                    rng,
                    (
                        config["index_n_heads"] * config["index_head_dim"],
                        config["q_lora_rank"],
                    ),
                    scale=0.005,
                ),
                f"{prefix}.attn.indexer.weights_proj.weight": _dense(
                    rng, (config["index_n_heads"], hidden), scale=0.005
                ),
                f"{prefix}.attn.indexer.compressor.wkv.weight": _dense(
                    rng, (index_width, hidden), scale=0.005
                ),
                f"{prefix}.attn.indexer.compressor.wgate.weight": _dense(
                    rng, (index_width, hidden), scale=0.005
                ),
                f"{prefix}.attn.indexer.compressor.ape": _dense(
                    rng, (ratio, index_width), scale=0.005
                ).astype(np.float32),
                f"{prefix}.attn.indexer.compressor.norm.weight": np.ones(
                    (config["index_head_dim"],), dtype=np.float16
                ),
            }
        )
    manifest = {
        "format": "tpq-1",
        "config": config,
        "quant": {},
        "expert_files": {"0": "experts.L0.safetensors"},
        "tiers_per_layer": {"0": "xx"},
    }
    io.save(
        path,
        FileHeader(
            version=2,
            model_arch="deepseek_v4-tpq-mfq",
            num_tensors=len(tensors),
            extra={
                "source_format": "tpq-1",
                "tpq_manifest": manifest,
            },
        ),
        tensors,
    )


def test_dsv4_config_and_names_validate_ratios():
    config = MlxDeepseekV4Config.from_manifest(_config(ratios=(0, 4)))
    assert config.compress_ratios == (0, 4)
    required = MlxDeepseekV4Names.required(config)
    assert "layers.1.attn.compressor.wkv.weight" in required
    assert "layers.1.attn.indexer.wq_b.weight" in required

    invalid = _config()
    invalid["compress_ratios"] = [4, 128]
    with pytest.raises(ValueError, match="one entry per layer"):
        MlxDeepseekV4Config.from_manifest(invalid)


def test_dsv4_ratio4_pool_retains_overlap_history():
    state = MlxDeepseekV4PoolState.allocate(
        ratio=4,
        head_dim=128,
        overlap=True,
        batch=1,
        max_context=16,
    )
    rng = np.random.default_rng(20260841)
    ape = np.zeros((4, 256), dtype=np.float32)
    norm = np.ones((128,), dtype=np.float32)
    angles = np.zeros((16, 32), dtype=np.float32)
    for length in range(1, 9):
        kv = rng.normal(size=(1, 1, 256)).astype(np.float16)
        gate = rng.normal(size=(1, 1, 256)).astype(np.float16)
        state.update(
            mx.array(kv),
            mx.array(gate),
            mx.array(ape),
            mx.array(norm),
            length=length,
            cosine=mx.array(np.cos(angles)),
            sine=mx.array(np.sin(angles)),
            quant_mode=0,
            eps=1e-6,
        )
    mx.eval(*state.arrays())
    assert state.pool_len == 2
    assert state.remainder == 0
    assert state.prev_kv is not None
    assert np.isfinite(_array(state.pool[:, :2])).all()
    assert np.any(_array(state.prev_kv) != 0)


def test_dsv4_native_mfq_prefill_decode_and_generate(tmp_path):
    path = tmp_path / "tiny-dsv4.mfq"
    _write_tiny_model(path)
    ids = np.array([[1, 3, 5, 7, 9]], dtype=np.int32)
    next_id = np.array([[4]], dtype=np.int32)

    with MlxDeepseekV4.from_mfq(path, max_context=12) as model:
        prompt_logits = model.prefill(ids, chunk_size=3)
        assert prompt_logits.shape == (1, 5, 17)
        decoded = _array(model.decode(next_id))
        assert model.position == 6

        replay = _array(
            model.forward(
                np.concatenate((ids, next_id), axis=1),
                use_cache=False,
            )
        )
        np.testing.assert_allclose(
            decoded[:, -1],
            replay[:, -1],
            rtol=8e-3,
            atol=8e-3,
        )

        generated = _array(model.generate(ids, 2, top_p=1.0))
        assert generated.shape == (1, 7)
        assert np.array_equal(generated[:, :5], ids)
        assert model.position == 6


def test_dsv4_load_tpq_model_dispatches_to_metal_runtime(tmp_path):
    path = tmp_path / "tiny-dsv4-entrypoint.mfq"
    _write_tiny_model(path)
    with load_tpq_model(path, device="metal", max_ctx=7) as model:
        assert isinstance(model, MlxDeepseekV4)
        assert model.max_context == 7


def test_dsv4_native_tpq_experts_use_bounded_mmap_residency(tmp_path):
    path = tmp_path / "tiny-dsv4-streamed-experts.mfq"
    _write_tiny_model(path, streamed_experts=True)
    ids = np.array([[1, 3, 5]], dtype=np.int32)

    with MlxDeepseekV4.from_mfq(
        path,
        max_context=8,
        expert_cache_gb=0.001,
    ) as model:
        streamed_logits = _array(model.prefill(ids))
        assert np.isfinite(streamed_logits).all()
        assert model.expert_residency is not None
        assert model.expert_residency.cache_nbytes > 0
        layer = model.layers[0]
        assert layer is not None
        assert layer.moe.residency is model.expert_residency
        assert layer.moe.gate_up is None
        assert layer.moe.down is None
    with MlxDeepseekV4.from_mfq(
        path,
        mmap=False,
        max_context=8,
    ) as model:
        resident_logits = _array(model.prefill(ids))
        assert model.expert_residency is None
    np.testing.assert_allclose(
        streamed_logits,
        resident_logits,
        rtol=4e-3,
        atol=4e-3,
    )


def test_dsv4_streamed_experts_load_from_nonprimary_mfq_shards(tmp_path):
    path = tmp_path / "tiny-dsv4-shard-source.mfq"
    _write_tiny_model(path, streamed_experts=True)
    shards = split_mfq(
        path,
        tmp_path / "tiny-dsv4-sharded.mfq",
        split_max_tensors=4,
    )
    assert len(shards) > 1
    expert_names = {
        "layers.0.ffn.experts.gate_up.weight",
        "layers.0.ffn.experts.down.weight",
    }
    with io.open_mmap(shards[-1]) as store:
        assert any(store.records[name].source_index > 0 for name in expert_names)

    ids = np.array([[1, 3, 5]], dtype=np.int32)
    with MlxDeepseekV4.from_mfq(
        shards[-1],
        max_context=8,
        expert_cache_gb=0.001,
    ) as model:
        logits = _array(model.prefill(ids))
        assert np.isfinite(logits).all()
        assert model.expert_residency is not None
        assert model.expert_residency.cache_nbytes > 0


def test_dsv4_ratio4_chunked_prefill_matches_one_chunk(tmp_path):
    path = tmp_path / "tiny-dsv4-ratio4.mfq"
    _write_tiny_model(path, ratio=4)
    ids = np.array([[1, 3, 5, 7, 9, 11, 13, 15, 2]], dtype=np.int32)

    with MlxDeepseekV4.from_mfq(path, max_context=12) as model:
        one_chunk = _array(model.prefill(ids, chunk_size=32))
        chunked = _array(model.prefill(ids, chunk_size=3))
        np.testing.assert_allclose(
            chunked,
            one_chunk,
            rtol=1e-2,
            atol=1e-2,
        )
        assert model.states is not None
        state = model.states[0]
        assert state.main is not None and state.indexer is not None
        assert state.main.pool_len == state.indexer.pool_len == 2
        assert state.main.remainder == state.indexer.remainder == 1


def test_dsv4_official_shape_fast_kernels_match_token_replay(tmp_path):
    path = tmp_path / "official-shape-dsv4.mfq"
    _write_official_shape_model(path)
    ids = np.array([[1, 3]], dtype=np.int32)

    with MlxDeepseekV4.from_mfq(path, max_context=4) as model:
        assert model.config.fast_attention
        assert model.config.fast_hyper_connections
        full = _array(model.prefill(ids))
        model.prefill(ids[:, :1])
        replay = _array(model.decode(ids[:, 1:]))
        np.testing.assert_allclose(
            replay[:, -1],
            full[:, -1],
            rtol=1e-2,
            atol=1e-2,
        )


def test_dsv4_official_ratio4_compressor_fast_path(tmp_path):
    path = tmp_path / "official-shape-ratio4-dsv4.mfq"
    _write_official_shape_model(path, ratio=4)
    ids = np.array([[1, 3, 5, 7, 9]], dtype=np.int32)

    with MlxDeepseekV4.from_mfq(path, max_context=8) as model:
        assert model.config.fast_indexer
        one_chunk = _array(model.prefill(ids, chunk_size=8))
        chunked = _array(model.prefill(ids, chunk_size=3))
        np.testing.assert_allclose(
            chunked,
            one_chunk,
            rtol=1e-2,
            atol=1e-2,
        )
        assert model.states is not None
        state = model.states[0]
        assert state.main is not None and state.indexer is not None
        assert state.main.pool_len == state.indexer.pool_len == 1
        assert state.main.remainder == state.indexer.remainder == 1


def test_dsv4_capacity_fails_before_cache_mutation(tmp_path):
    path = tmp_path / "tiny-dsv4-capacity.mfq"
    _write_tiny_model(path)
    with MlxDeepseekV4.from_mfq(path, max_context=5) as model:
        model.prefill(np.array([[1, 2, 3, 4]], dtype=np.int32))
        before = model.position
        assert model.states is not None
        local = _array(model.states[0].local).copy()
        with pytest.raises(ValueError, match="max_context"):
            model.decode(np.array([[5, 6]], dtype=np.int32))
        assert model.position == before
        np.testing.assert_array_equal(_array(model.states[0].local), local)
