"""TorchNintCausalLM tiny end-to-end runtime tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
requires_cuda_runtime = pytest.mark.skipif(
    not torch.cuda.is_available()
    or (shutil.which("cl") is None and shutil.which("cl.exe") is None),
    reason="CUDA runtime tests require a CUDA device and MSVC cl",
)

from mfq.formats import io  # noqa: E402
from mfq.formats.assets import MODEL_CONFIG_ASSET  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.quantize import nint_quant  # noqa: E402
from mfq.runtime.causal_lm import (  # noqa: E402
    TorchNintCausalLM,
    TorchNintCausalLMConfig,
    TorchNintCausalLMNames,
)


def _qt(rng: np.random.Generator, shape: tuple[int, int], scale: float = 0.05):
    w = rng.normal(0, scale, size=shape).astype(np.float32)
    return nint_quant.quantize(w, NintSpec(4, 16, 6), axis=0)


def _tiny_tensors(seed: int = 0):
    rng = np.random.default_rng(seed)
    cfg = TorchNintCausalLMConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rope_base=10_000.0,
        rms_norm_eps=1e-5,
    )
    tensors = {
        "model.token_embedding.weight": _qt(rng, (cfg.vocab_size, cfg.hidden_size)),
        "model.block.0.attention.norm.weight": np.ones(cfg.hidden_size, dtype=np.float32),
        "model.block.0.attention.query.weight": _qt(rng, (cfg.hidden_size, cfg.hidden_size)),
        "model.block.0.attention.key.weight": _qt(rng, (cfg.kv_size, cfg.hidden_size)),
        "model.block.0.attention.value.weight": _qt(rng, (cfg.kv_size, cfg.hidden_size)),
        "model.block.0.attention.output.weight": _qt(rng, (cfg.hidden_size, cfg.hidden_size)),
        "model.block.0.mlp.norm.weight": np.ones(cfg.hidden_size, dtype=np.float32),
        "model.block.0.mlp.gate.weight": _qt(rng, (cfg.intermediate_size, cfg.hidden_size)),
        "model.block.0.mlp.up.weight": _qt(rng, (cfg.intermediate_size, cfg.hidden_size)),
        "model.block.0.mlp.down.weight": _qt(rng, (cfg.hidden_size, cfg.intermediate_size)),
        "model.output_norm.weight": np.ones(cfg.hidden_size, dtype=np.float32),
        "model.output.weight": _qt(rng, (cfg.vocab_size, cfg.hidden_size)),
    }
    return cfg, tensors


def _tiny_qwen35_tensors(seed: int = 10):
    rng = np.random.default_rng(seed)
    cfg = TorchNintCausalLMConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        rope_base=10_000.0,
        rms_norm_eps=1e-5,
        layer_types=("linear_attention", "full_attention"),
        qwen35_attn_q_gate=True,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_a_is_log=True,
        norm_weight_offset=1.0,
    )
    tensors = {
        "model.token_embedding.weight": _qt(rng, (cfg.vocab_size, cfg.hidden_size)),
        "model.output_norm.weight": np.ones(cfg.hidden_size, dtype=np.float32),
        "model.output.weight": _qt(rng, (cfg.vocab_size, cfg.hidden_size)),
    }
    for i in range(cfg.num_hidden_layers):
        tensors[f"model.block.{i}.attention.norm.weight"] = np.ones(
            cfg.hidden_size, dtype=np.float32
        )
        tensors[f"model.block.{i}.mlp.norm.weight"] = np.ones(
            cfg.hidden_size, dtype=np.float32
        )
        tensors[f"model.block.{i}.mlp.gate.weight"] = _qt(
            rng, (cfg.intermediate_size, cfg.hidden_size)
        )
        tensors[f"model.block.{i}.mlp.up.weight"] = _qt(
            rng, (cfg.intermediate_size, cfg.hidden_size)
        )
        tensors[f"model.block.{i}.mlp.down.weight"] = _qt(
            rng, (cfg.hidden_size, cfg.intermediate_size)
        )
    tensors.update({
        "model.block.0.linear_attention.qkv.weight": _qt(rng, (128, cfg.hidden_size)),
        "model.block.0.linear_attention.gate.weight": _qt(rng, (64, cfg.hidden_size)),
        "model.block.0.linear_attention.alpha.weight": _qt(rng, (2, cfg.hidden_size)),
        "model.block.0.linear_attention.beta.weight": _qt(rng, (2, cfg.hidden_size)),
        "model.block.0.linear_attention.conv.weight": rng.normal(
            0, 0.05, size=(128, 1, 4)
        ).astype(np.float32),
        "model.block.0.linear_attention.dt_bias": rng.normal(
            0, 0.05, size=(2,)
        ).astype(np.float32),
        "model.block.0.linear_attention.a": rng.normal(
            -1.0, 0.05, size=(2,)
        ).astype(np.float32),
        "model.block.0.linear_attention.norm.weight": np.ones(32, dtype=np.float32),
        "model.block.0.linear_attention.output.weight": _qt(rng, (cfg.hidden_size, 64)),
        "model.block.1.attention.query.weight": _qt(
            rng, (cfg.hidden_size * 2, cfg.hidden_size)
        ),
        "model.block.1.attention.key.weight": _qt(rng, (cfg.kv_size, cfg.hidden_size)),
        "model.block.1.attention.value.weight": _qt(rng, (cfg.kv_size, cfg.hidden_size)),
        "model.block.1.attention.output.weight": _qt(
            rng, (cfg.hidden_size, cfg.hidden_size)
        ),
        "model.block.1.attention.query_norm.weight": np.ones(
            cfg.head_dim, dtype=np.float32
        ),
        "model.block.1.attention.key_norm.weight": np.ones(
            cfg.head_dim, dtype=np.float32
        ),
    })
    return cfg, tensors


@requires_cuda_runtime
def test_torch_nint_causal_lm_prefill_decode_matches_full(tmp_path: Path):
    cfg, tensors = _tiny_tensors()
    path = tmp_path / "tiny.mfq"
    io.save(path, FileHeader(model_arch="tiny", num_tensors=len(tensors)), tensors)
    model = TorchNintCausalLM.from_mfq(path, cfg, device="cuda")

    ids = torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.int64)
    full = model(ids, use_cache=False)
    model.reset_cache(1)
    _ = model(ids[:, :3], use_cache=True)
    dec = model(ids[:, 3:], use_cache=True)
    torch.testing.assert_close(dec[:, -1, :], full[:, -1, :], atol=5e-3, rtol=5e-3)


@requires_cuda_runtime
def test_torch_nint_causal_lm_generate_shape(tmp_path: Path):
    cfg, tensors = _tiny_tensors(1)
    path = tmp_path / "tiny.mfq"
    io.save(path, FileHeader(model_arch="tiny", num_tensors=len(tensors)), tensors)
    model = TorchNintCausalLM.from_mfq(path, cfg, device="cuda")

    ids = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int64)
    out = model.generate(ids, max_new_tokens=3, temperature=0.0)
    assert tuple(out.shape) == (1, 6)
    assert int(out.min().item()) >= 0
    assert int(out.max().item()) < cfg.vocab_size


@requires_cuda_runtime
def test_torch_nint_causal_lm_mmap_prefill(tmp_path: Path):
    cfg, tensors = _tiny_tensors(2)
    path = tmp_path / "tiny-mmap.mfq"
    io.save(path, FileHeader(model_arch="tiny", num_tensors=len(tensors)), tensors)
    model = TorchNintCausalLM.from_mfq(path, cfg, device="cuda", mmap=True)

    ids = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int64)
    logits = model(ids, use_cache=False)
    assert tuple(logits.shape) == (1, 3, cfg.vocab_size)
    model.tensors.close()


@requires_cuda_runtime
def test_torch_nint_qwen35_mixed_layers_prefill_decode_matches_full(tmp_path: Path):
    cfg, tensors = _tiny_qwen35_tensors()
    path = tmp_path / "tiny-qwen35.mfq"
    io.save(path, FileHeader(model_arch="qwen35-tiny", num_tensors=len(tensors)), tensors)
    model = TorchNintCausalLM.from_mfq(path, cfg, device="cuda")

    ids = torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.int64)
    full = model(ids, use_cache=False)
    model.reset_cache(1)
    _ = model(ids[:, :3], use_cache=True)
    dec = model(ids[:, 3:], use_cache=True)
    torch.testing.assert_close(dec[:, -1, :], full[:, -1, :], atol=8e-3, rtol=8e-3)


@pytest.mark.parametrize("mmap", [False, True])
def test_qwen35_legacy_gguf_is_canonicalized_at_the_file_boundary(
    tmp_path: Path,
    mmap: bool,
):
    cfg, canonical = _tiny_qwen35_tensors()
    tensors = {
        "token_embd.weight": canonical.pop("model.token_embedding.weight"),
        "output_norm.weight": canonical.pop("model.output_norm.weight"),
        "output.weight": canonical.pop("model.output.weight"),
        "blk.0.attn_qkv.weight": canonical.pop(
            "model.block.0.linear_attention.qkv.weight"
        ),
        "blk.0.attn_gate.weight": canonical.pop(
            "model.block.0.linear_attention.gate.weight"
        ),
        "blk.0.ssm_alpha.weight": canonical.pop(
            "model.block.0.linear_attention.alpha.weight"
        ),
        "blk.0.ssm_beta.weight": canonical.pop(
            "model.block.0.linear_attention.beta.weight"
        ),
        "blk.0.ssm_conv1d.weight": canonical.pop(
            "model.block.0.linear_attention.conv.weight"
        ),
        "blk.0.ssm_dt.bias": canonical.pop("model.block.0.linear_attention.dt_bias"),
        "blk.0.ssm_a": canonical.pop("model.block.0.linear_attention.a"),
        "blk.0.ssm_norm.weight": canonical.pop(
            "model.block.0.linear_attention.norm.weight"
        ),
        "blk.0.ssm_out.weight": canonical.pop(
            "model.block.0.linear_attention.output.weight"
        ),
    }
    for i in range(cfg.num_hidden_layers):
        tensors[f"blk.{i}.attn_norm.weight"] = canonical.pop(
            f"model.block.{i}.attention.norm.weight"
        )
        tensors[f"blk.{i}.post_attention_norm.weight"] = canonical.pop(
            f"model.block.{i}.mlp.norm.weight"
        )
        tensors[f"blk.{i}.ffn_gate.weight"] = canonical.pop(
            f"model.block.{i}.mlp.gate.weight"
        )
        tensors[f"blk.{i}.ffn_up.weight"] = canonical.pop(
            f"model.block.{i}.mlp.up.weight"
        )
        tensors[f"blk.{i}.ffn_down.weight"] = canonical.pop(
            f"model.block.{i}.mlp.down.weight"
        )
    tensors.update(
        {
            "blk.1.attn_q.weight": canonical.pop("model.block.1.attention.query.weight"),
            "blk.1.attn_k.weight": canonical.pop("model.block.1.attention.key.weight"),
            "blk.1.attn_v.weight": canonical.pop("model.block.1.attention.value.weight"),
            "blk.1.attn_output.weight": canonical.pop(
                "model.block.1.attention.output.weight"
            ),
            "blk.1.attn_q_norm.weight": canonical.pop(
                "model.block.1.attention.query_norm.weight"
            ),
            "blk.1.attn_k_norm.weight": canonical.pop(
                "model.block.1.attention.key_norm.weight"
            ),
        }
    )
    assert not canonical
    tensors[MODEL_CONFIG_ASSET] = json.dumps(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": cfg.num_hidden_layers,
                "mtp_num_hidden_layers": 0,
            },
        }
    ).encode()
    path = tmp_path / "tiny-qwen35-legacy-gguf.mfq"
    io.save(path, FileHeader(model_arch="qwen35", num_tensors=len(tensors)), tensors)

    class BoundaryProbe(TorchNintCausalLM):
        def __init__(self, tensors, config, device="cuda"):
            self.tensors = tensors
            self.config = config
            self.device = device

    model = BoundaryProbe.from_mfq(path, cfg, device="cpu", mmap=mmap)

    assert model.config.norm_weight_offset == 0.0
    assert model.config.linear_a_is_log is False
    assert model.config.linear_attention_tiled_heads is True
    assert "model.block.0.linear_attention.qkv.weight" in model.tensors
    assert "blk.0.attn_qkv.weight" not in model.tensors
    if mmap:
        model.tensors.close()


def test_qwen35_hf_config_shape_fields():
    cfg = TorchNintCausalLMConfig.from_qwen35_hf_config({
        "text_config": {
            "vocab_size": 248320,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "attn_output_gate": True,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            * 16,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "rope_parameters": {"rope_theta": 10000000},
        }
    })
    assert cfg.head_dim == 256
    assert cfg.attention_size == 6144
    assert cfg.kv_size == 1024
    assert cfg.linear_num_value_heads * cfg.linear_value_head_dim == 6144
    assert cfg.norm_weight_offset == 1.0


def test_minicpmo45_hf_config_uses_the_shared_canonical_names():
    cfg = TorchNintCausalLMConfig.from_minicpmo45_hf_config(
        {
            "model_type": "minicpmo",
            "version": "4.5",
            "vocab_size": 151748,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
        }
    )
    names = TorchNintCausalLMNames()

    assert cfg.head_dim == 128
    assert cfg.kv_size == 1024
    assert cfg.layer_types == ("full_attention",) * 36
    assert cfg.norm_weight_offset == 0.0
    assert names.token_embd == "model.token_embedding.weight"
    assert names.output == "model.output.weight"


@requires_cuda_runtime
def test_torch_nint_causal_lm_accepts_modality_inputs_embeds():
    cfg, tensors = _tiny_tensors(23)
    model = TorchNintCausalLM(tensors, cfg, device="cuda")
    ids = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int64)
    embeds = model.embed(ids)

    from_ids = model(ids, use_cache=False)
    from_embeds = model(inputs_embeds=embeds, use_cache=False)

    torch.testing.assert_close(from_embeds, from_ids, atol=0, rtol=0)
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=embeds, use_cache=False)
