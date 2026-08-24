from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import pytest
from gguf import GGUFReader

from mfq.server.catalog import ModelCatalog
from mfq.server.hf_tokenizer import (
    ensure_hf_tokenizer_gguf,
    native_hf_asset_environment,
)
from mfq.server.native import native_tokenizer_arguments


def _hf_fixture(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"vocab_size": 6}}),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 3, "eos_token_id": 4, "pad_token_id": 3}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<bos>",
                "chat_template": "{{ messages[0].content }}",
                "add_bos_token": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "ab": 2},
                    "merges": [["a", "b"]],
                },
                "added_tokens": [
                    {"id": 3, "content": "<bos>", "special": True},
                    {"id": 4, "content": "<eos>", "special": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    header = json.dumps(
        {
            "weight": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        },
        separators=(",", ":"),
    ).encode()
    with (root / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.write(b"\x00\x00")


def test_catalog_discovers_native_hf_checkpoint(tmp_path: Path) -> None:
    model = tmp_path / "Qwen-Test"
    _hf_fixture(model)
    catalog = ModelCatalog([tmp_path], cache_seconds=0)
    artifacts = asyncio.run(catalog.list())

    assert len(artifacts.data) == 1
    assert artifacts.data[0].name == "Qwen-Test"
    assert artifacts.data[0].format == "hf"
    assert artifacts.data[0].architecture == "qwen3_5-hf-full-mfq"
    assert artifacts.data[0].tensor_count == 1
    assert artifacts.data[0].dtypes == ["BF16"]
    assert artifacts.data[0].complete
    assert artifacts.data[0].loadable


def test_hf_tokenizer_cache_is_reusable_and_runtime_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "Qwen-Test"
    cache = tmp_path / "cache"
    _hf_fixture(model)

    tokenizer = ensure_hf_tokenizer_gguf(model, cache)
    assert ensure_hf_tokenizer_gguf(model, cache) == tokenizer
    reader = GGUFReader(tokenizer, "r")
    assert reader.get_field("tokenizer.ggml.model").contents() == "gpt2"
    assert reader.get_field("tokenizer.ggml.pre").contents() == "qwen35"
    assert reader.get_field("tokenizer.ggml.tokens").contents() == [
        "a", "b", "ab", "<bos>", "<eos>", "[PAD5]"
    ]
    assert reader.get_field("tokenizer.ggml.bos_token_id").contents() == 3
    assert reader.get_field("tokenizer.ggml.eos_token_id").contents() == 4

    monkeypatch.setenv("MFQ_SERVER_TOKENIZER_CACHE_DIR", str(cache))
    arguments = native_tokenizer_arguments(model)
    assert arguments[0] == "--tokenizer-gguf"
    assert Path(arguments[1]).is_file()


def test_minicpmo_native_runtime_materializes_exact_resampler_asset(
    tmp_path: Path,
) -> None:
    model = tmp_path / "MiniCPM-Test"
    _hf_fixture(model)
    (model / "config.json").write_text(
        json.dumps({"model_type": "minicpmo", "vocab_size": 6}),
        encoding="utf-8",
    )
    environment = native_hf_asset_environment(model, tmp_path / "assets")
    asset = Path(environment["MFQ_MINICPMO45_RESAMPLER_POSITION_ASSET"])
    assert asset.is_file()
    assert asset.read_bytes()[:20] == b"MFQRSPB1" + struct.pack("<III", 70, 70, 4096)
