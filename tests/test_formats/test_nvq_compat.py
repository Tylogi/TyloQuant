from __future__ import annotations

import numpy as np

from mfq.formats.niq import NIQ2_E8, NiqSpec, NiqTensor, pack_niq, unpack_niq
from mfq.formats.nvq import NVQ2_E8, NvqSpec, NvqTensor, pack_nvq


def test_legacy_python_names_alias_canonical_nvq_api():
    assert NiqSpec is NvqSpec
    assert NiqTensor is NvqTensor
    assert NIQ2_E8 is NVQ2_E8


def test_legacy_pack_api_emits_canonical_nvq_wire_format():
    tensor = NiqTensor(
        spec=NIQ2_E8,
        shape=(1, 24),
        axis=0,
        neuron_len=24,
        neuron_scale=np.asarray([0.25], dtype=np.float32),
        sub_scale=np.asarray([[5]], dtype=np.uint8),
        indices=np.asarray([[1, 2, 3]], dtype=np.uint8),
        signs=np.asarray([[4, 5, 6]], dtype=np.uint8),
    )
    legacy_api_payload = pack_niq(tensor)
    assert legacy_api_payload == pack_nvq(tensor)
    assert legacy_api_payload[:3] == b"NVQ"
    restored = unpack_niq(legacy_api_payload)
    np.testing.assert_array_equal(restored.indices, tensor.indices)
