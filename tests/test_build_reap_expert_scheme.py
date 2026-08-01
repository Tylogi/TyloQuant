from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from mfq.tools.build_reap_expert_scheme import (
    open_bfloat16_memmap_tensor,
    resolve_profiles,
)


def test_resolve_profiles_can_select_only_missing_low_bit_candidates():
    profiles = resolve_profiles(("NINT2", "NINT3"))

    assert tuple(profiles) == ("NINT2", "NINT3")
    assert profiles["NINT2"].bits == 2
    assert profiles["NINT3"].bits == 3


def test_resolve_profiles_rejects_duplicates_and_unknown_names():
    with pytest.raises(ValueError, match="duplicate"):
        resolve_profiles(("NINT2", "NINT2"))
    with pytest.raises(ValueError, match="unknown"):
        resolve_profiles(("NINT1",))


def test_bfloat16_memmap_tensor_reads_exact_slices(tmp_path):
    tensor = torch.arange(48, dtype=torch.float32).reshape(4, 3, 4).to(
        torch.bfloat16
    )
    path = tmp_path / "weights.safetensors"
    save_file({"experts": tensor}, path)

    with open_bfloat16_memmap_tensor(path, "experts") as source:
        actual = source[1:3]

    torch.testing.assert_close(actual, tensor[1:3], rtol=0, atol=0)
