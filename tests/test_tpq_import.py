from __future__ import annotations

import importlib
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from mfq.formats.tpq import TpqInt4Tensor
from mfq.formats.io import load_mmap
from mfq.formats.moe import NintMoeTensor
from mfq.quantize.expert_nint import dequantize_expertwise
from mfq.runtime.tpq import load_tpq_package, open_tpq_artifact
from mfq.runtime.tpq_mfq import MfqTpqStore, install_mfq_tpq_store
from mfq.tools.import_tpq_to_mfq import convert
from mfq.tools.split_mfq import split_mfq


def _cccp_artifact(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    root.mkdir()
    rng = np.random.default_rng(20260726)
    packed = rng.integers(0, 256, size=(3, 64), dtype=np.uint8)
    scales = np.full((3, 2), 0.125, dtype=np.float16)
    save_file(
        {
            "dense.weight": torch.from_numpy(packed),
            "dense.weight.qs": torch.from_numpy(scales),
            "norm.weight": torch.arange(8, dtype=torch.float32),
        },
        str(root / "dense.safetensors"),
    )

    tier_specs = {
        "x": (8, 256, np.uint8),
        "w": (4, 80, np.uint8),
        "v": (4, 256, np.uint8),
        "vv": (4, 4096, np.uint16),
    }
    tensors: dict[str, torch.Tensor] = {}
    expected: dict[str, np.ndarray] = {}
    for tier, (width, entries, dtype) in tier_specs.items():
        for tag, rows, columns in (("gu", 16, 8), ("dn", 8, 8)):
            codebook = rng.normal(size=(entries, width)).astype(np.float32)
            indices = rng.integers(
                0,
                entries,
                size=(rows, columns // width),
                dtype=dtype,
            )
            tensors[f"cb.{tag}.{tier}"] = torch.from_numpy(codebook)
            expert = ("x", "w", "v", "vv").index(tier)
            key = f"e{expert}.{tag}{tier}"
            if tier == "w" and tag == "gu":
                compressed = np.frombuffer(
                    zlib.compress(indices.tobytes()),
                    dtype=np.uint8,
                ).copy()
                tensors[key + "z"] = torch.from_numpy(compressed)
            else:
                tensors[key] = torch.from_numpy(indices)
            expected[f"{tag}.{expert}"] = codebook[
                indices.reshape(-1)
            ].reshape(rows, columns)
    save_file(tensors, str(root / "experts.L00.safetensors"))
    manifest = {
        "format": "cccp-1",
        "config": {
            "n_layers": 1,
            "n_experts": 4,
            "top_k": 2,
            "hidden": 8,
            "moe_inter": 8,
            "hc_mult": 4,
        },
        "quant": {
            "dense": "int4-g64",
            "int4_group": 64,
            "vq": {
                tier: [width, entries]
                for tier, (width, entries, _dtype) in tier_specs.items()
            },
        },
        "tiers_per_layer": {"0": "xwvV"},
        "dense_file": "dense.safetensors",
        "expert_files": {"0": "experts.L00.safetensors"},
        "tokenizer_files": [],
    }
    (root / "cccp.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, expected


def test_import_cccp_directory_to_native_mfq(tmp_path: Path) -> None:
    source, expected = _cccp_artifact(tmp_path / "cccp")
    output = convert(source, tmp_path / "model.mfq", row_chunk=2)
    header, store = load_mmap(output)
    try:
        assert header.model_arch == "deepseek_v4-tpq-mfq"
        assert header.extra["source_format"] == "cccp-1"
        assert store.records["dense.weight"].dtype == "TPQ-I4G64"
        dense = store["dense.weight"]
        assert isinstance(dense, TpqInt4Tensor)
        assert dense.shape == (3, 128)

        gate = store["layers.0.ffn.experts.gate_up.weight"]
        down = store["layers.0.ffn.experts.down.weight"]
        assert isinstance(gate, NintMoeTensor)
        assert isinstance(down, NintMoeTensor)
        assert gate.expert_profiles == (
            "TPQ-X",
            "TPQ-W",
            "TPQ-V",
            "TPQ-VV",
        )
        gate_values = dequantize_expertwise(gate)
        down_values = dequantize_expertwise(down)
        for expert in range(4):
            np.testing.assert_array_equal(
                gate_values[expert], expected[f"gu.{expert}"]
            )
            np.testing.assert_array_equal(
                down_values[expert], expected[f"dn.{expert}"]
            )
    finally:
        store.close()


def test_import_kimi_tpq2_sharded_dense_and_compact_experts(
    tmp_path: Path,
) -> None:
    source, _expected = _cccp_artifact(tmp_path / "kimi")
    dense_root = source / "dense"
    dense_root.mkdir()
    (source / "dense.safetensors").rename(dense_root / "part-1.safetensors")
    save_file(
        {"extra.weight": torch.arange(5, dtype=torch.bfloat16)},
        str(dense_root / "part-2.safetensors"),
    )
    (source / "experts.L00.safetensors").rename(
        source / "experts.L01.safetensors"
    )
    expert_shard = source / "experts.L01.safetensors"
    expert_tensors = load_file(str(expert_shard))
    save_file(
        {
            name: tensor
            for name, tensor in expert_tensors.items()
            if not name.startswith("e3.")
        },
        str(expert_shard),
    )
    manifest = json.loads((source / "cccp.json").read_text(encoding="utf-8"))
    manifest["model_family"] = "kimi_k3"
    manifest["config"] = {
        "n_layers": 2,
        "first_dense_layers": 1,
        "n_experts": 4,
        "top_k": 2,
        "hidden": 12,
        "routed_hidden": 8,
        "moe_inter": 8,
        "kda_layers": [0],
    }
    manifest.pop("dense_file")
    manifest["dense_files"] = [
        "part-1.safetensors",
        "part-2.safetensors",
    ]
    manifest["nonexpert"] = {"path": "dense"}
    manifest["expert_files"] = {"1": "experts.L01.safetensors"}
    manifest["tiers_per_layer"] = {"1": "xwvd"}
    (source / "cccp.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    output = convert(source, tmp_path / "kimi.mfq", row_chunk=2)
    header, store = load_mmap(output)
    try:
        assert header.model_arch == "kimi_k3-tpq-mfq"
        assert "extra.weight" in store.records
        assert store.records["extra.weight"].dtype == "F16"
        gate = store["layers.1.ffn.experts.gate_up.weight"]
        down = store["layers.1.ffn.experts.down.weight"]
        assert isinstance(gate, NintMoeTensor)
        assert isinstance(down, NintMoeTensor)
        assert gate.shape == (4, 16, 8)
        assert down.shape == (4, 8, 8)
        assert gate.expert_profiles[-1] == "TPQ-X"
        np.testing.assert_array_equal(
            dequantize_expertwise(gate)[-1],
            np.zeros((16, 8), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            dequantize_expertwise(down)[-1],
            np.zeros((8, 8), dtype=np.float32),
        )
        assert header.extra["tpq_manifest"]["tiers_per_layer"] == {
            "1": "xwvd"
        }
        assert open_tpq_artifact(output).architecture == "kimi_k3"
    finally:
        store.close()


def test_native_cccp_mfq_exposes_tpq_store_interface(tmp_path: Path) -> None:
    source, expected = _cccp_artifact(tmp_path / "cccp")
    output = convert(source, tmp_path / "model.mfq", row_chunk=2)
    shards = split_mfq(
        output,
        tmp_path / "model-split.mfq",
        split_max_tensors=1,
    )
    entry = shards[-1]
    artifact = open_tpq_artifact(entry)
    assert artifact.summary()["format"] == "mfq-native-tpq.v1"

    tpq = load_tpq_package()
    importlib.import_module("tpq.kernels")
    importlib.import_module("tpq.expert_slots")
    store = MfqTpqStore(entry, tpq)
    try:
        dense = store.get_dense("dense.weight")
        assert dense.shape == torch.Size([3, 128])
        gate, down = store.load_expert(1 - 1, 1)
        np.testing.assert_array_equal(
            gate.dequant().numpy(), expected["gu.1"]
        )
        np.testing.assert_array_equal(
            down.dequant().numpy(), expected["dn.1"]
        )
        assert store._layer_cache == {}
        assert store._layer_maps == {}
        assert sum(store.expert_signature_counts().values()) == 4
        assert store.available_mask(0).tolist() == [True, True, True, True]
    finally:
        store.close()

    install_mfq_tpq_store(tpq)
    from tpq import dsv4model

    dispatched = dsv4model.CCCPStore(str(entry))
    try:
        assert isinstance(dispatched, MfqTpqStore)
    finally:
        dispatched.close()
