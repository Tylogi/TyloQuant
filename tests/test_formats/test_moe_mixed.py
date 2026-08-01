from __future__ import annotations

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.moe import NintMoePool, NintMoeTensor
from mfq.formats.nepq import NEPQ0_L, NEPQ0_S, NEPQ1_L, NEPQ1_S
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from tests.mixed_family_fixtures import FLAT_FAMILIES, make_flat_family
from tests.test_formats.test_nepq import _tensor as make_nepq


@pytest.mark.parametrize("family", FLAT_FAMILIES)
def test_nintm_v2_roundtrips_every_flat_compact_family(family: str):
    tensor = make_flat_family(family)
    container = NintMoeTensor(
        (2, 3, 96),
        (NintMoePool(np.arange(2, dtype=np.int32), tensor),),
    )
    blob = io.pack_nint_moe(container)
    restored = io.unpack_nint_moe(blob)
    assert blob[:4] == b"NIM2"
    assert restored.expert_profiles == (family, family)
    assert io.pack_nint_moe(restored) == blob


@pytest.mark.parametrize(
    "spec",
    (NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L),
)
def test_nintm_v2_roundtrips_every_cross_expert_family(spec):
    tensor = make_nepq(spec)
    container = NintMoeTensor(
        tensor.shape,
        (NintMoePool(np.arange(2, dtype=np.int32), tensor),),
    )
    blob = io.pack_nint_moe(container)
    restored = io.unpack_nint_moe(blob)
    assert restored.expert_profiles == (spec.label, spec.label)
    assert io.pack_nint_moe(restored) == blob


@pytest.mark.parametrize(
    "spec",
    (
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 6),
        NintSpec(6, 24, 6),
        NintSpec(8, 24, 8),
    ),
)
def test_nintm_v2_roundtrips_every_nint_family(spec):
    rng = np.random.default_rng(100 + spec.bits)
    values = rng.normal(0, 0.04, (6, 96)).astype(np.float32)
    tensor = quantize(values, spec, axis=0)
    container = NintMoeTensor(
        (2, 3, 96),
        (NintMoePool(np.arange(2, dtype=np.int32), tensor),),
    )
    restored = io.unpack_nint_moe(io.pack_nint_moe(container))
    assert restored.expert_profiles == (spec.profile_label, spec.profile_label)
