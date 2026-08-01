"""TorchNintCausalLM tiny end-to-end runtime tests."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA 不可用", allow_module_level=True)
if shutil.which("cl") is None and shutil.which("cl.exe") is None:
    pytest.skip("MSVC cl 不在 PATH", allow_module_level=True)

from mfq.formats import io  # noqa: E402
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
        "token_embd.weight": _qt(rng, (cfg.vocab_size, cfg.hidden_size)),
        "blk.0.attn_norm.weight": np.ones(cfg.hidden_size, dtype=np.float32),
        "blk.0.attn_q.weight": _qt(rng, (cfg.hidden_size, cfg.hidden_size)),
        "blk.0.attn_k.weight": _qt(rng, (cfg.kv_size, cfg.hidden_size)),
        "blk.0.attn_v.weight": _qt(rng, (cfg.kv_size, cfg.hidden_size)),
        "blk.0.attn_output.weight": _qt(rng, (cfg.hidden_size, cfg.hidden_size)),
        "blk.0.ffn_norm.weight": np.ones(cfg.hidden_size, dtype=np.float32),
        "blk.0.ffn_gate.weight": _qt(rng, (cfg.intermediate_size, cfg.hidden_size)),
        "blk.0.ffn_up.weight": _qt(rng, (cfg.intermediate_size, cfg.hidden_size)),
        "blk.0.ffn_down.weight": _qt(rng, (cfg.hidden_size, cfg.intermediate_size)),
        "output_norm.weight": np.ones(cfg.hidden_size, dtype=np.float32),
        "output.weight": _qt(rng, (cfg.vocab_size, cfg.hidden_size)),
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
    )
    tensors = {
        "token_embd.weight": _qt(rng, (cfg.vocab_size, cfg.hidden_size)),
        "output_norm.weight": np.ones(cfg.hidden_size, dtype=np.float32),
        "output.weight": _qt(rng, (cfg.vocab_size, cfg.hidden_size)),
    }
    for i in range(cfg.num_hidden_layers):
        tensors[f"blk.{i}.attn_norm.weight"] = np.ones(cfg.hidden_size, dtype=np.float32)
        tensors[f"blk.{i}.ffn_norm.weight"] = np.ones(cfg.hidden_size, dtype=np.float32)
        tensors[f"blk.{i}.ffn_gate.weight"] = _qt(rng, (cfg.intermediate_size, cfg.hidden_size))
        tensors[f"blk.{i}.ffn_up.weight"] = _qt(rng, (cfg.intermediate_size, cfg.hidden_size))
        tensors[f"blk.{i}.ffn_down.weight"] = _qt(rng, (cfg.hidden_size, cfg.intermediate_size))
    tensors.update({
        "blk.0.ssm_qkv.weight": _qt(rng, (128, cfg.hidden_size)),
        "blk.0.ssm_z.weight": _qt(rng, (64, cfg.hidden_size)),
        "blk.0.ssm_alpha.weight": _qt(rng, (2, cfg.hidden_size)),
        "blk.0.ssm_beta.weight": _qt(rng, (2, cfg.hidden_size)),
        "blk.0.ssm_conv1d.weight": rng.normal(0, 0.05, size=(128, 1, 4)).astype(np.float32),
        "blk.0.ssm_dt.bias": rng.normal(0, 0.05, size=(2,)).astype(np.float32),
        "blk.0.ssm_a": rng.normal(-1.0, 0.05, size=(2,)).astype(np.float32),
        "blk.0.ssm_norm.weight": np.ones(32, dtype=np.float32),
        "blk.0.ssm_out.weight": _qt(rng, (cfg.hidden_size, 64)),
        "blk.1.attn_q.weight": _qt(rng, (cfg.hidden_size * 2, cfg.hidden_size)),
        "blk.1.attn_k.weight": _qt(rng, (cfg.kv_size, cfg.hidden_size)),
        "blk.1.attn_v.weight": _qt(rng, (cfg.kv_size, cfg.hidden_size)),
        "blk.1.attn_output.weight": _qt(rng, (cfg.hidden_size, cfg.hidden_size)),
        "blk.1.attn_q_norm.weight": np.ones(cfg.head_dim, dtype=np.float32),
        "blk.1.attn_k_norm.weight": np.ones(cfg.head_dim, dtype=np.float32),
    })
    return cfg, tensors


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


def test_torch_nint_causal_lm_mmap_prefill(tmp_path: Path):
    cfg, tensors = _tiny_tensors(2)
    path = tmp_path / "tiny-mmap.mfq"
    io.save(path, FileHeader(model_arch="tiny", num_tensors=len(tensors)), tensors)
    model = TorchNintCausalLM.from_mfq(path, cfg, device="cuda", mmap=True)

    ids = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int64)
    logits = model(ids, use_cache=False)
    assert tuple(logits.shape) == (1, 3, cfg.vocab_size)
    model.tensors.close()


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


def test_qwen35_gguf_names_select_converted_weight_semantics():
    cfg, tensors = _tiny_qwen35_tensors()
    names = TorchNintCausalLMNames.qwen35_gguf()
    tensors["blk.0.attn_qkv.weight"] = tensors.pop("blk.0.ssm_qkv.weight")
    tensors["blk.0.attn_gate.weight"] = tensors.pop("blk.0.ssm_z.weight")
    for i in range(cfg.num_hidden_layers):
        tensors[f"blk.{i}.post_attention_norm.weight"] = tensors.pop(f"blk.{i}.ffn_norm.weight")

    model = TorchNintCausalLM(tensors, cfg, names=names, device="cuda")

    assert model.config.norm_weight_offset == 0.0
    assert model.config.linear_a_is_log is False
    assert model.blocks[0].gguf_layout is True


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
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16,
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
