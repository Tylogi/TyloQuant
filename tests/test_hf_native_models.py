from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFReader

from mfq.formats.assets import (
    HF_GENERATION_CONFIG_ASSET,
    HF_TOKENIZER_CONFIG_ASSET,
    HF_TOKENIZER_JSON_ASSET,
    MODEL_CONFIG_ASSET,
)
from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.server.catalog import ModelCatalog, native_hf_model_type_supported
from mfq.server.hf_tokenizer import (
    DEEPSEEK_V41_CHAT_TEMPLATE,
    ensure_hf_tokenizer_gguf,
    ensure_mfq_tokenizer_gguf,
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
                    {
                        "id": 3,
                        "content": "<bos>",
                        "special": True,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                    },
                    {
                        "id": 4,
                        "content": "<eos>",
                        "special": True,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                    },
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


@pytest.mark.parametrize("model_type", ["qwen4_exp", "glm5_next"])
def test_catalog_recognizes_flash_next_hf_as_conversion_source_only(
    tmp_path: Path,
    model_type: str,
) -> None:
    model = tmp_path / model_type
    _hf_fixture(model)
    (model / "config.json").write_text(
        json.dumps({"model_type": model_type}),
        encoding="utf-8",
    )

    artifact = asyncio.run(ModelCatalog([tmp_path], cache_seconds=0).list()).data[0]

    assert artifact.complete
    assert artifact.format == "hf"
    assert artifact.architecture == f"{model_type}-hf-full-mfq"
    assert not artifact.loadable
    assert "convert it to MFQ" in (artifact.error or "")


def test_catalog_prefers_same_name_mfq_over_its_hf_source(tmp_path: Path) -> None:
    source = tmp_path / "same-name"
    _hf_fixture(source)
    converted = tmp_path / "same-name.mfq"
    save(
        converted,
        FileHeader(version=2, model_arch="qwen4_exp-test"),
        {"weight": np.ones((1,), dtype=np.float16)},
    )

    artifacts = asyncio.run(ModelCatalog([tmp_path], cache_seconds=0).list())

    assert len(artifacts.data) == 1
    assert artifacts.data[0].name == "same-name"
    assert artifacts.data[0].format == "mfq"
    assert artifacts.data[0].loadable


def test_native_hf_support_matches_cpp_dispatch_families() -> None:
    assert native_hf_model_type_supported("deepseek_v4_vision")
    assert native_hf_model_type_supported("deepseek_v41")
    assert native_hf_model_type_supported("minicpmo")
    assert native_hf_model_type_supported("qwen3_8")
    assert not native_hf_model_type_supported("qwen4_exp")
    assert not native_hf_model_type_supported("glm5_next")


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
        "a",
        "b",
        "ab",
        "<bos>",
        "<eos>",
        "[PAD5]",
    ]
    assert reader.get_field("tokenizer.ggml.bos_token_id").contents() == 3
    assert reader.get_field("tokenizer.ggml.eos_token_id").contents() == 4

    monkeypatch.setenv("MFQ_SERVER_TOKENIZER_CACHE_DIR", str(cache))
    arguments = native_tokenizer_arguments(model)
    assert arguments[0] == "--tokenizer-gguf"
    assert Path(arguments[1]).is_file()


def test_mfq_embedded_hf_tokenizer_cache_is_reusable_and_runtime_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "Qwen-Source"
    cache = tmp_path / "cache"
    _hf_fixture(source)
    model = tmp_path / "Qwen-Converted.mfq"
    save(
        model,
        FileHeader(version=2, model_arch="qwen4_exp-hf-mfq-nint-recipe"),
        {
            "weight": np.ones((1,), dtype=np.float16),
            MODEL_CONFIG_ASSET: (source / "config.json").read_bytes(),
            HF_TOKENIZER_JSON_ASSET: (source / "tokenizer.json").read_bytes(),
            HF_TOKENIZER_CONFIG_ASSET: (
                source / "tokenizer_config.json"
            ).read_bytes(),
            HF_GENERATION_CONFIG_ASSET: (
                source / "generation_config.json"
            ).read_bytes(),
        },
    )

    tokenizer = ensure_mfq_tokenizer_gguf(model, cache)
    assert ensure_mfq_tokenizer_gguf(model, cache) == tokenizer
    reader = GGUFReader(tokenizer, "r")
    assert reader.get_field("tokenizer.ggml.pre").contents() == "qwen35"
    assert reader.get_field("tokenizer.ggml.tokens").contents() == [
        "a",
        "b",
        "ab",
        "<bos>",
        "<eos>",
        "[PAD5]",
    ]

    monkeypatch.setenv("MFQ_SERVER_TOKENIZER_CACHE_DIR", str(cache))
    arguments = native_tokenizer_arguments(model)
    assert arguments == ["--tokenizer-gguf", str(tokenizer)]


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


def test_deepseek_v41_gets_chat_template_and_engram_token_map(
    tmp_path: Path,
) -> None:
    model = tmp_path / "DeepSeek-V41-Test"
    _hf_fixture(model)
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "bos_token_id": 3,
                "eos_token_id": 4,
                "text_config": {
                    "model_type": "deepseek_v41_text",
                    "vocab_size": 5,
                    "engram_layer_ids": [1],
                    "engram_num_embeddings": [36],
                    "engram_max_ngram_size": 3,
                    "engram_vocab_size": 5,
                    "engram_n_heads": 2,
                    "engram_compressed_vocab_size": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    tokenizer_config = json.loads((model / "tokenizer_config.json").read_text())
    tokenizer_config.pop("chat_template")
    (model / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config), encoding="utf-8"
    )

    tokenizer = ensure_hf_tokenizer_gguf(model, tmp_path / "tokenizers")
    reader = GGUFReader(tokenizer, "r")
    assert reader.get_field("tokenizer.chat_template").contents() == (
        DEEPSEEK_V41_CHAT_TEMPLATE
    )

    environment = native_hf_asset_environment(model, tmp_path / "assets")
    asset = Path(environment["MFQ_DEEPSEEK_V41_ENGRAM_TOKEN_MAP"])
    payload = asset.read_bytes()
    assert payload[:28] == struct.pack("<4sIIIIII", b"D41T", 2, 5, 5, 1, 3, 2)
    # header + token map + layer IDs + multipliers + prime bucket sizes
    assert len(payload) == 28 + 5 * 4 + 4 + 3 * 8 + 4 * 4
