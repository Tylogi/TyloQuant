from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mfq.compat.legacy_tensor_names import LegacyCanonicalTensorView
from mfq.formats import io
from mfq.formats.header import FileHeader
from mfq.formats.nint import NintSpec
from mfq.quantize import nint_quant
from mfq.tools.split_mfq import split_mfq


@pytest.mark.parametrize(
    "spec",
    (
        NintSpec(2, 16, 5),
        NintSpec(3, 24, 6),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 6),
        NintSpec(6, 24, 6),
        NintSpec(8, 24, 6),
    ),
)
def test_mmap_nint_embedding_reads_only_selected_rows(
    tmp_path: Path,
    spec: NintSpec,
) -> None:
    rng = np.random.default_rng(4040 + spec.bits)
    source = rng.normal(size=(19, 53)).astype(np.float32)
    tensor = nint_quant.quantize(source, spec)
    path = tmp_path / f"nint{spec.bits}.mfq"
    io.save(
        path,
        FileHeader(model_arch="mmap-row-test", num_tensors=1),
        {"embedding.weight": tensor},
    )

    with io.open_mmap(path) as store:
        store.read_blob = lambda _name: (_ for _ in ()).throw(
            AssertionError("row lookup copied the complete tensor blob")
        )
        reader = store.embedding_reader("embedding.weight")
        requested = np.asarray([[18, 3, 18], [0, 7, 3]], dtype=np.int32)
        actual = reader.read_rows(requested)
        expected = nint_quant.dequantize(tensor)[requested.reshape(-1)]

        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        assert reader.shape == source.shape
        assert reader.last_rows_read == 4
        assert reader.last_logical_bytes < store.records["embedding.weight"].nbytes
        assert not store._cache


def test_mmap_mixed_sub_bits_embedding_reads_cohort_metadata_in_place(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(60911)
    source = rng.normal(size=(9, 53)).astype(np.float32)
    row_sub_bits = np.asarray([5, 6, 7, 5, 7, 6, 5, 7, 6], dtype=np.uint8)
    tensor = nint_quant.quantize(
        source,
        NintSpec(4, 24, 6),
        row_sub_bits=row_sub_bits,
    )
    path = tmp_path / "nint-v2.mfq"
    io.save(
        path,
        FileHeader(model_arch="mmap-row-test", num_tensors=1),
        {"embedding.weight": tensor},
    )

    with io.open_mmap(path) as store:
        store.read_blob = lambda _name: (_ for _ in ()).throw(
            AssertionError("mixed-sub-bit row lookup copied the complete blob")
        )
        reader = store.embedding_reader("embedding.weight")
        requested = np.asarray([8, 1, 7, 0, 8], dtype=np.int64)

        np.testing.assert_allclose(
            reader.read_rows(requested),
            nint_quant.dequantize(tensor)[requested],
            rtol=0,
            atol=0,
        )
        assert reader.last_rows_read == 4
        assert reader.last_logical_bytes < store.records["embedding.weight"].nbytes


def test_mmap_mixed_q_and_sub_bits_embedding_reads_cohorts_in_place(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(60912)
    source = rng.normal(size=(10, 53)).astype(np.float32)
    row_q_bits = np.asarray([2, 3, 4, 5, 6] * 2, dtype=np.uint8)
    row_sub_bits = np.asarray([5, 5, 6, 6, 8] * 2, dtype=np.uint8)
    tensor = nint_quant.quantize(
        source,
        NintSpec(4, 24, 6),
        row_q_bits=row_q_bits,
        row_sub_bits=row_sub_bits,
    )
    path = tmp_path / "nint-v2.mfq"
    io.save(
        path,
        FileHeader(model_arch="mmap-row-test", num_tensors=1),
        {"embedding.weight": tensor},
    )

    with io.open_mmap(path) as store:
        assert store.records["embedding.weight"].dtype == "NINTv2"
        store.read_blob = lambda _name: (_ for _ in ()).throw(
            AssertionError("mixed-q row lookup copied the complete blob")
        )
        reader = store.embedding_reader("embedding.weight")
        requested = np.asarray([9, 1, 8, 0, 9], dtype=np.int64)

        np.testing.assert_allclose(
            reader.read_rows(requested),
            nint_quant.dequantize(tensor)[requested],
            rtol=0,
            atol=0,
        )
        assert reader.last_rows_read == 4
        assert reader.last_logical_bytes < store.records["embedding.weight"].nbytes


@pytest.mark.parametrize("dtype", ("F16", "F32", "BF16"))
def test_mmap_dense_embedding_preserves_stored_values(
    tmp_path: Path,
    dtype: str,
) -> None:
    source = (np.arange(77, dtype=np.float32).reshape(11, 7) - 31) / 13
    if dtype == "F16":
        stored = source.astype(np.float16)
        expected = stored
    elif dtype == "F32":
        stored = source
        expected = source
    else:
        bits = (source.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
        stored = bits.view(io.BFloat16Array)
        expected = io.bfloat16_to_float32(stored)
    path = tmp_path / f"dense-{dtype.lower()}.mfq"
    io.save(
        path,
        FileHeader(model_arch="mmap-row-test", num_tensors=1),
        {"embedding.weight": stored},
    )

    with io.open_mmap(path) as store:
        reader = store.embedding_reader("embedding.weight")
        rows = np.asarray([10, 1, 10, 4], dtype=np.int64)
        actual = reader.read_rows(rows)

        np.testing.assert_array_equal(actual, expected[rows])
        assert reader.dtype == dtype
        assert reader.last_rows_read == 3
        assert reader.last_logical_bytes == 3 * 7 * stored.dtype.itemsize


def test_mmap_float8_embedding_decodes_only_selected_rows(tmp_path: Path) -> None:
    raw = np.asarray(
        [
            [0x00, 0x38, 0xB8, 0x30, 0x7E],
            [0x40, 0xC0, 0x28, 0xA8, 0x01],
            [0x48, 0xC8, 0x20, 0xA0, 0x02],
        ],
        dtype=np.uint8,
    ).view(io.Float8E4M3Array)
    path = tmp_path / "float8.mfq"
    io.save(
        path,
        FileHeader(model_arch="mmap-row-test", num_tensors=1),
        {"embedding.weight": raw},
    )

    with io.open_mmap(path) as store:
        loaded = store["embedding.weight"]
        assert io.is_float8_e4m3_array(loaded)
        reader = store.embedding_reader("embedding.weight")
        actual = reader.read_rows(np.asarray([0, 2, 0], dtype=np.int32))

        expected_rows = np.asarray(
            [
                [0.0, 1.0, -1.0, 0.5, 448.0],
                [4.0, -4.0, 0.125, -0.125, 0.00390625],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, expected_rows[[0, 1, 0]])
        assert reader.last_rows_read == 2
        assert reader.last_logical_bytes == 10


def test_mmap_embedding_uses_the_owning_shard_through_a_legacy_alias(
    tmp_path: Path,
) -> None:
    source = np.arange(117, dtype=np.float16).reshape(13, 9)
    unsplit = tmp_path / "source.mfq"
    io.save(
        unsplit,
        FileHeader(model_arch="mmap-row-test", num_tensors=2),
        {
            "padding.weight": np.ones((2, 2), dtype=np.float16),
            "legacy.embedding.weight": source,
        },
    )
    shards = split_mfq(
        unsplit,
        tmp_path / "sharded.mfq",
        split_max_tensors=1,
    )

    with io.open_mmap(shards[0]) as store:
        record = store.records["legacy.embedding.weight"]
        assert record.source_index == 1
        view = LegacyCanonicalTensorView(
            store,
            {"legacy.embedding.weight": "embedding.weight"},
        )
        store.read_blob = lambda _name: (_ for _ in ()).throw(
            AssertionError("aliased row lookup copied the complete tensor blob")
        )
        reader = view.embedding_reader("embedding.weight")
        rows = np.asarray([12, 0, 6, 12], dtype=np.int32)

        np.testing.assert_array_equal(reader.read_rows(rows), source[rows])
        assert reader.last_rows_read == 3
        assert not store._cache
