from __future__ import annotations

import importlib
import json
import struct
from dataclasses import dataclass
from types import SimpleNamespace
import zlib
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from mfq.formats.tpq import TpqInt4Tensor, pack_tpq_indices
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


def _projection_tpq_artifact(root: Path) -> tuple[Path, dict[tuple[str, int], bytes]]:
    root.mkdir()
    save_file(
        {"norm.weight": torch.arange(8, dtype=torch.float32)},
        str(root / "dense.safetensors"),
    )
    layouts = {
        "g9": {"dim": 2, "size": 512, "group_size": 1, "groups": 2},
        "u10": {"dim": 4, "size": 1024},
        "d11": {"dim": 2, "size": 2048, "group_size": 2, "groups": 1},
    }
    projection_layout = {"gate": "g9", "up": "u10", "down": "d11"}
    shapes = {"gate": (4, 8), "up": (4, 8), "down": (8, 4)}
    bits = {"gate": 9, "up": 10, "down": 11}
    rng = np.random.default_rng(1200)
    tensors: dict[str, torch.Tensor] = {}
    expected: dict[tuple[str, int], bytes] = {}
    for projection, layout in projection_layout.items():
        spec = layouts[layout]
        codebooks = 2 if spec.get("group_size") == 1 else 1
        for group in range(codebooks):
            suffix = f".g{group:03d}" if "group_size" in spec else ""
            tensors[f"cb.{projection}.{layout}{suffix}"] = torch.from_numpy(
                rng.normal(size=(spec["size"], spec["dim"])).astype(np.float32)
            )
        rows, columns = shapes[projection]
        for expert in range(2):
            indices = rng.integers(
                0,
                spec["size"],
                size=(rows, columns // spec["dim"]),
                dtype=np.uint16,
            )
            raw = pack_tpq_indices(indices, bits[projection])
            tensors[f"e{expert}.{projection}.{layout}"] = torch.from_numpy(
                np.frombuffer(raw, dtype=np.uint8).copy()
            )
            expected[(projection, expert)] = raw
    save_file(tensors, str(root / "experts.L00.safetensors"))
    manifest = {
        "format": "tpq-1",
        "model_family": "deepseek_v4",
        "config": {
            "n_layers": 1,
            "n_experts": 2,
            "top_k": 1,
            "hidden": 8,
            "moe_inter": 4,
            "hc_mult": 4,
        },
        "quant": {
            "method": "projection-vq",
            "layouts": layouts,
            "projection_layouts": {"0": projection_layout},
            "index_packing": {
                "g9": "packed-u9",
                "u10": "packed-u10",
                "d11": "packed-u11",
            },
        },
        "dense_file": "dense.safetensors",
        "expert_files": {"0": "experts.L00.safetensors"},
        "tokenizer_files": [],
    }
    (root / "tpq.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, expected


def test_import_projection_tpq_preserves_three_packed_projections(
    tmp_path: Path,
) -> None:
    source, expected = _projection_tpq_artifact(tmp_path / "projection")
    output = convert(source, tmp_path / "projection.mfq")
    package = load_tpq_package()
    install_mfq_tpq_store(package)
    store = MfqTpqStore(output, package)
    try:
        assert store.man.projection_vq
        assert store.man.projection_operator_capability(0) == {
            "packed_formats": ("p9", "p10", "p11"),
            "code_dims": (2, 4, 2),
            "codebook_sizes": (512, 1024, 2048),
        }
        assert store.man.projection_layouts(0, 1) == {
            "gate": "g9",
            "up": "u10",
            "down": "d11",
        }
        assert store.man.projection_operator_capabilities(0) == (
            store.man.projection_operator_capability(0),
        )
        for expert in range(2):
            weights = store.load_expert_packed(0, expert)
            assert len(weights) == 3
            for projection, weight in zip(("gate", "up", "down"), weights):
                assert bytes(weight.raw.numpy()) == expected[(projection, expert)]
                assert pack_tpq_indices(
                    weight.unpack().numpy(), weight.bits
                ) == expected[(projection, expert)]
        header, mapped = load_mmap(output)
        try:
            assert header.extra["source_format"] == "tpq-1"
            assert "cccp_manifest" not in header.extra
            assert set(mapped.records).issuperset(
                {
                    "layers.0.ffn.experts.gate.weight",
                    "layers.0.ffn.experts.up.weight",
                    "layers.0.ffn.experts.down.weight",
                }
            )
            gate = mapped["layers.0.ffn.experts.gate.weight"]
            assert isinstance(gate, NintMoeTensor)
            assert {pool.tensor.spec.tier for pool in gate.pools} == {"p"}
            assert {pool.tensor.spec.index_bits for pool in gate.pools} == {9}
            del gate
        finally:
            mapped.close()
    finally:
        store.close()


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
    # Clone and release the source views before replacing the file: Windows
    # rejects overwriting a safetensors file with a live mapped section.
    filtered_experts = {
        name: tensor.clone()
        for name, tensor in expert_tensors.items()
        if not name.startswith("e3.")
    }
    del expert_tensors
    save_file(filtered_experts, str(expert_shard))
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
    from tpq import dsv4model, kimi_model

    dispatched = dsv4model.CCCPStore(str(entry))
    try:
        assert isinstance(dispatched, MfqTpqStore)
    finally:
        dispatched.close()
    kimi_dispatched = kimi_model.TPQStore(str(entry))
    try:
        assert isinstance(kimi_dispatched, MfqTpqStore)
    finally:
        kimi_dispatched.close()


def test_tpq4_native_lifecycle_is_not_replaced_by_legacy_patch() -> None:
    from mfq.runtime.tpq_residency_patch import apply_tpq_residency_patch

    package = load_tpq_package()
    dsv4model = importlib.import_module("tpq.dsv4model")
    native_preload = dsv4model.DSV4TPQModel.preload
    assert hasattr(
        dsv4model.DSV4TPQModel,
        "_prepare_tp_packed_finalizer",
    )
    apply_tpq_residency_patch()
    assert dsv4model.DSV4TPQModel.preload is native_preload


def test_native_tpq_mfq_reads_legacy_source_exact_fp8_pair(
    tmp_path: Path,
) -> None:
    from mfq.formats.header import FileHeader
    from mfq.formats.shards import write_blob_record_shards

    @dataclass(frozen=True)
    class BlobRecord:
        name: str
        dtype: str
        path: Path
        nbytes: int
        offset: int = 0

    @dataclass(frozen=True)
    class BlockFP8Weight:
        q: torch.Tensor
        s: torch.Tensor
        columns: int
        block: int

    def dense_blob(path: Path, value: np.ndarray) -> BlobRecord:
        payload = (
            struct.pack("<I", value.ndim)
            + struct.pack(f"<{value.ndim}q", *value.shape)
            + value.tobytes()
        )
        path.write_bytes(payload)
        return BlobRecord("", "", path, len(payload))

    raw = np.arange(128 * 256, dtype=np.uint8).reshape(128, 256)
    scales = np.array([[127, 128]], dtype=np.uint8)
    raw_record = dense_blob(tmp_path / "raw.bin", raw)
    scale_record = dense_blob(tmp_path / "scale.bin", scales)
    records = [
        BlobRecord(
            "dense.weight",
            "F8_E4M3",
            raw_record.path,
            raw_record.nbytes,
        ),
        BlobRecord(
            "dense.scale",
            "F8_E8M0",
            scale_record.path,
            scale_record.nbytes,
        ),
    ]
    output = tmp_path / "legacy-fp8.mfq"
    manifest = {
        "format": "cccp-1",
        "config": {"n_experts": 0},
        "quant": {"vq": {"x": [8, 256]}},
        "expert_files": {},
    }
    write_blob_record_shards(
        output,
        FileHeader(
            version=2,
            model_arch="deepseek_v4-tpq-mfq",
            num_tensors=2,
            extra={
                "source_format": "cccp-1",
                "cccp_manifest": manifest,
            },
        ),
        records,
    )
    package = SimpleNamespace(
        kernels=SimpleNamespace(BlockFP8Weight=BlockFP8Weight)
    )
    store = MfqTpqStore(output, package)
    try:
        assert store.dense_names() == ["dense.weight"]
        value = store.get_dense("dense.weight")
        assert torch.equal(value.q, torch.from_numpy(raw))
        torch.testing.assert_close(value.s, torch.tensor([[1.0, 2.0]]))
        assert value.columns == 256
        assert value.block == 128
        assert torch.equal(
            store.get_raw("dense.scale"), torch.from_numpy(scales)
        )
    finally:
        store.close()


def test_import_tpq4_routed_heterogeneous_projection_manifest(
    tmp_path: Path,
) -> None:
    source, _expected = _projection_tpq_artifact(tmp_path / "heterogeneous")
    manifest_path = source / "tpq.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layouts = manifest["quant"].pop("projection_layouts")["0"]
    manifest["quant"]["heterogeneous_expert_tiering"] = {
        "precision_levels": {"low": layouts, "high": layouts},
        "layer_expert_levels": {"0": ["low", "high"]},
    }
    manifest["routed_experts"] = {
        "layers": 1,
        "experts_per_layer": 2,
        "no_expert_drop": True,
        "layer_files": {
            "0": {"path": manifest["expert_files"]["0"]},
        },
    }
    manifest.pop("expert_files")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output = convert(source, tmp_path / "heterogeneous.mfq")
    package = load_tpq_package()
    store = MfqTpqStore(output, package)
    try:
        assert store.man.heterogeneous_projection_vq
        assert store.man.projection_layouts(0, 0) == layouts
        assert store.man.projection_layouts(0, 1) == layouts
        capabilities = store.man.projection_operator_capabilities(0)
        assert len(capabilities) == 1
        assert capabilities[0]["packed_formats"] == ("p9", "p10", "p11")
    finally:
        store.close()
