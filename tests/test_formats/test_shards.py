from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.io import load, open_mmap, save
from mfq.formats.shards import (
    SPLIT_COUNT_KEY,
    SPLIT_NO_KEY,
    SPLIT_RECORDS_COUNT_KEY,
    SPLIT_TENSORS_COUNT_KEY,
    format_shard_path,
    parse_shard_path,
    parse_size,
)
from mfq.tools.split_mfq import split_mfq


def _source_model(path: Path) -> None:
    save(
        path,
        FileHeader(
            version=2,
            model_arch="test-shards",
            extra={"model": "fixture", "nested": {"value": 3}},
        ),
        {
            "weight.0": np.arange(12, dtype=np.float16).reshape(3, 4),
            "weight.1": np.arange(20, dtype=np.float32).reshape(4, 5),
            "weight.2": np.arange(6, dtype=np.int32).reshape(2, 3),
            MODEL_CONFIG_ASSET: b'{"model_type":"fixture"}',
        },
    )


def test_shard_name_round_trip(tmp_path: Path) -> None:
    base = tmp_path / "model.v1.mfq"
    shard = format_shard_path(base, 2, 17)
    assert shard.name == "model.v1-00002-of-00017.mfq"
    assert parse_shard_path(shard) == (base, 2, 17)
    assert parse_size("4G") == 4_000_000_000
    assert parse_size("512m") == 512_000_000


def test_offline_split_loads_from_any_shard_without_changing_records(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.mfq"
    _source_model(source)
    with open_mmap(source) as original:
        raw = {
            name: bytes(original.blob_view(record))
            for name, record in original.records.items()
        }

    shards = split_mfq(
        source,
        tmp_path / "split.mfq",
        split_max_tensors=1,
    )
    assert [path.name for path in shards] == [
        "split-00001-of-00003.mfq",
        "split-00002-of-00003.mfq",
        "split-00003-of-00003.mfq",
    ]

    for entry in (shards[0], shards[1], shards[-1]):
        with open_mmap(entry) as store:
            assert store.path == shards[0]
            assert store.paths == shards
            assert store.header.model_arch == "test-shards"
            assert store.header.extra["model"] == "fixture"
            assert store.header.extra[SPLIT_NO_KEY] == 0
            assert store.header.extra[SPLIT_COUNT_KEY] == 3
            assert store.header.extra[SPLIT_TENSORS_COUNT_KEY] == 3
            assert store.header.extra[SPLIT_RECORDS_COUNT_KEY] == 4
            assert set(store) == set(raw)
            for name, record in store.records.items():
                assert bytes(store.blob_view(record)) == raw[name]

    header, tensors = load(shards[-1])
    assert header.num_tensors == 4
    np.testing.assert_array_equal(
        tensors["weight.1"], np.arange(20, dtype=np.float32).reshape(4, 5)
    )
    assert tensors[MODEL_CONFIG_ASSET] == b'{"model_type":"fixture"}'


def test_missing_shard_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "model.mfq"
    _source_model(source)
    shards = split_mfq(
        source,
        tmp_path / "split.mfq",
        split_max_tensors=1,
    )
    shards[1].unlink()
    with pytest.raises(FileNotFoundError, match="missing MFQ shard"):
        open_mmap(shards[0])


def test_split_size_keeps_an_oversize_tensor_whole(tmp_path: Path) -> None:
    source = tmp_path / "model.mfq"
    _source_model(source)
    shards = split_mfq(
        source,
        tmp_path / "size.mfq",
        split_max_size=30,
    )
    with open_mmap(shards[-1]) as store:
        assert len(store) == 4
        assert len(shards) == 3


def test_split_can_use_the_source_base_name_and_keeps_the_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.mfq"
    _source_model(source)
    shards = split_mfq(source, source, split_max_tensors=99)
    assert source.is_file()
    assert [path.name for path in shards] == ["model-00001-of-00001.mfq"]
    with open_mmap(shards[0]) as store:
        assert store.header.extra[SPLIT_NO_KEY] == 0
        assert store.header.extra[SPLIT_COUNT_KEY] == 1
        assert len(store) == 4
