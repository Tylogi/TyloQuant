from __future__ import annotations

from dataclasses import replace

import numpy as np

from mfq.formats.header import FileHeader
from mfq.formats.io import open_mmap, save
from mfq.formats.nvq import E8_4096, NVQ2_E8_4096, NvqJscTensor
from mfq.tools.optimize_layout import optimize_layouts


def _tensor(storage_layout: str) -> NvqJscTensor:
    return NvqJscTensor(
        shape=(2, 50),
        axis=0,
        neuron_len=50,
        neuron_scale=np.asarray([0.25, 0.5], dtype=np.float32),
        scale_lut=np.arange(16, dtype=np.float32),
        bank_for_state=np.zeros(16, dtype=np.uint8),
        state=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        indices=np.arange(14, dtype=np.uint16).reshape(2, 7),
        signs=(np.arange(14, dtype=np.uint8).reshape(2, 7) * 5) & 0x7F,
        codebooks=E8_4096[None].astype(np.int8),
        base_spec=NVQ2_E8_4096,
        storage_layout=storage_layout,
    )


def test_optimize_layout_rewrites_only_legacy_nvq2j_xl(tmp_path):
    source = tmp_path / "source.mfq"
    output = tmp_path / "optimized.mfq"
    legacy = _tensor("streams")
    group64 = replace(legacy, storage_layout="group64")
    dense = np.arange(12, dtype=np.float16).reshape(3, 4)
    save(
        source,
        FileHeader(version=2, model_arch="test", extra={"sample": True}),
        {"legacy": legacy, "group64": group64, "dense": dense},
    )

    result = optimize_layouts(source, output)
    assert result["changed_tensors"] == 1
    with open_mmap(source) as before, open_mmap(output) as after:
        assert after.header.extra["sample"] is True
        assert after.header.extra["nvq2j_xl.storage_layout"] == "group64-v1"
        assert after.records["legacy"].nbytes > before.records["legacy"].nbytes
        assert after.read_blob("group64") == before.read_blob("group64")
        assert after.read_blob("dense") == before.read_blob("dense")
        for name in ("legacy", "group64"):
            restored = after[name]
            assert isinstance(restored, NvqJscTensor)
            assert restored.storage_layout == "group64"
            np.testing.assert_array_equal(restored.indices, legacy.indices)
            np.testing.assert_array_equal(restored.signs, legacy.signs)
