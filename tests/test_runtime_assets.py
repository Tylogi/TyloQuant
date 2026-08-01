from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mfq.formats import io
from mfq.formats.assets import (
    ASSET_MANIFEST_KEY,
    MODEL_CONFIG_ASSET,
    TOKENIZER_GGUF_ASSET,
    gguf_metadata_asset,
    model_config_asset,
    runtime_asset_manifest,
)
from mfq.formats.header import FileHeader


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
