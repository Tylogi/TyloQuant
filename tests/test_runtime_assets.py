from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mfq.formats import io
from mfq.formats.assets import (
    ASSET_MANIFEST_KEY,
    MINICPMO45_RESAMPLER_POS_EMBED_ASSET,
    MODEL_CONFIG_ASSET,
    RuntimeAsset,
    TOKENIZER_GGUF_ASSET,
    gguf_metadata_asset,
    minicpmo45_resampler_pos_embed_asset,
    model_config_asset,
    runtime_asset_manifest,
)
from mfq.formats.header import FileHeader
from mfq.tools import pack_runtime_assets as pack_assets_module


def test_blob_records_roundtrip_and_mmap(tmp_path: Path) -> None:
    path = tmp_path / "assets.mfq"
    config = model_config_asset({"model_type": "tiny", "vocab_size": 32})
    tokenizer = b"GGUF tokenizer metadata"
    header = FileHeader(
        version=2,
        model_arch="tiny",
        extra={
            ASSET_MANIFEST_KEY: runtime_asset_manifest(
                [config]
            )
        },
    )
    io.save(
        path,
        header,
        {
            "weight": np.arange(8, dtype=np.float16).reshape(2, 4),
            MODEL_CONFIG_ASSET: config.data,
            TOKENIZER_GGUF_ASSET: tokenizer,
        },
    )

    loaded_header, tensors = io.load(path)
    assert tensors[MODEL_CONFIG_ASSET] == config.data
    assert tensors[TOKENIZER_GGUF_ASSET] == tokenizer
    assert json.loads(tensors[MODEL_CONFIG_ASSET])["vocab_size"] == 32
    assert loaded_header.extra[ASSET_MANIFEST_KEY]["version"] == 1

    with io.open_mmap(path) as store:
        assert store.records[MODEL_CONFIG_ASSET].dtype == "BLOB"
        assert store[MODEL_CONFIG_ASSET] == config.data
        np.testing.assert_array_equal(
            store["weight"], np.arange(8, dtype=np.float16).reshape(2, 4)
        )


def test_gguf_metadata_asset_removes_tensor_table() -> None:
    raw = bytearray(b"GGUF")
    raw.extend(struct.pack("<IQQ", 3, 7, 2))
    raw.extend(b"a" * 40)
    fields = {
        "general.architecture": SimpleNamespace(
            offset=24, parts=[np.zeros(10, dtype=np.uint8)]
        ),
        "tokenizer.ggml.tokens": SimpleNamespace(
            offset=34, parts=[np.zeros(30, dtype=np.uint8)]
        ),
    }
    reader = SimpleNamespace(
        fields=fields,
        data=np.frombuffer(raw, dtype=np.uint8),
        byte_order="I",
    )

    asset = gguf_metadata_asset(reader)

    assert asset.name == TOKENIZER_GGUF_ASSET
    assert struct.unpack_from("<Q", asset.data, 8)[0] == 0
    assert struct.unpack_from("<Q", asset.data, 16)[0] == 2
    assert len(asset.data) == 64


def test_minicpmo45_resampler_position_asset_matches_official_numpy_bf16() -> None:
    asset = minicpmo45_resampler_pos_embed_asset(
        max_size=(2, 3), embed_dim=8
    )
    magic, height, width, embed_dim = struct.unpack_from("<8sIII", asset.data)
    assert magic == b"MFQRSPB1"
    assert (height, width, embed_dim) == (2, 3, 8)
    assert asset.name == MINICPMO45_RESAMPLER_POS_EMBED_ASSET

    grid_h = np.arange(height, dtype=np.float32)
    grid_w = np.arange(width, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)

    def official_embed(position: np.ndarray) -> np.ndarray:
        omega = np.arange(embed_dim // 4, dtype=np.float32)
        omega /= embed_dim / 4.0
        omega = 1.0 / 10000**omega
        phase = np.einsum("hw,d->hwd", position, omega)
        return np.concatenate([np.sin(phase), np.cos(phase)], axis=-1)

    official = np.concatenate(
        [official_embed(grid[0]), official_embed(grid[1])], axis=-1
    )
    expected = (
        torch.from_numpy(official)
        .to(torch.bfloat16)
        .view(torch.uint16)
        .numpy()
    )
    actual = np.frombuffer(asset.data, dtype="<u2", offset=20).reshape(
        height, width, embed_dim
    )
    np.testing.assert_array_equal(actual, expected)


def test_pack_runtime_assets_adds_minicpmo45_position_asset(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mfq"
    output = tmp_path / "output.mfq"
    config = tmp_path / "config.json"
    tokenizer = tmp_path / "tokenizer.gguf"
    io.save(
        source,
        FileHeader(version=2, model_arch="minicpmo"),
        {"weight": np.arange(8, dtype=np.float16)},
    )
    config.write_text(
        json.dumps({"model_type": "minicpmo", "version": "4.5"}),
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"unused by fake reader")

    class FakeReader:
        def __init__(self, _path: Path) -> None:
            raw = bytearray(b"GGUF")
            raw.extend(struct.pack("<IQQ", 3, 0, 1))
            raw.extend(b"x" * 16)
            self.fields = {
                "general.architecture": SimpleNamespace(
                    offset=24, parts=[np.zeros(16, dtype=np.uint8)]
                )
            }
            self.data = np.frombuffer(raw, dtype=np.uint8)
            self.byte_order = "I"

    position = RuntimeAsset(
        MINICPMO45_RESAMPLER_POS_EMBED_ASSET,
        "application/octet-stream",
        b"exact-position",
    )
    monkeypatch.setattr(pack_assets_module, "_load_gguf_reader", lambda: FakeReader)
    monkeypatch.setattr(
        pack_assets_module,
        "minicpmo45_resampler_pos_embed_asset",
        lambda: position,
    )

    pack_assets_module.pack_runtime_assets(
        source,
        output,
        config_path=config,
        tokenizer_gguf=tokenizer,
    )

    with io.open_mmap(output) as store:
        assert store[MINICPMO45_RESAMPLER_POS_EMBED_ASSET] == b"exact-position"
        assert store["weight"].tolist() == list(range(8))
