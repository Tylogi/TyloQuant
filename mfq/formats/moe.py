"""Expert-wise mixed-family compact tensor containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from mfq.formats.tpq import TpqPqTensor
from mfq.formats.nepq import NepqTensor
from mfq.formats.nint8_zero import Nint8ZeroTensor
from mfq.formats.npq0_l import Npq0LTensor
from mfq.formats.npq0_s import Npq0STensor
from mfq.formats.nvq import NvqJscTensor, NvqTensor
from mfq.formats.nvq1_l import Nvq1LTensor
from mfq.formats.nvq1_s import Nvq1STensor
from mfq.quantize.nint_quant import NintTensor

ExpertPoolTensor: TypeAlias = (
    NintTensor
    | Nint8ZeroTensor
    | NvqTensor
    | NvqJscTensor
    | Npq0LTensor
    | Npq0STensor
    | Nvq1LTensor
    | Nvq1STensor
    | NepqTensor
    | TpqPqTensor
)


def expert_tensor_family(tensor: ExpertPoolTensor) -> str:
    """Return the public precision-family label for one cohort tensor."""

    if isinstance(tensor, Nint8ZeroTensor):
        return "NINT8-0"
    if isinstance(tensor, NintTensor):
        return tensor.spec.profile_label
    if isinstance(tensor, NepqTensor):
        return tensor.spec.label
    if isinstance(tensor, TpqPqTensor):
        return tensor.spec.label
    if isinstance(tensor, Nvq1LTensor):
        return "NVQ1-L"
    if isinstance(tensor, Nvq1STensor):
        return "NVQ1-S"
    if isinstance(tensor, Npq0LTensor):
        return "NPQ0-L"
    if isinstance(tensor, Npq0STensor):
        return "NPQ0-S"
    if isinstance(tensor, NvqJscTensor):
        return {
            "e8_256": "NVQ2J",
            "e8_1024": "NVQ2J-L",
            "e8_4096": "NVQ2J-XL",
            "d4_256": "NVQ3J",
            "d4_512": "NVQ3J-512",
            "d4_1024": "NVQ3J-L",
        }[tensor.spec.codebook]
    if isinstance(tensor, NvqTensor):
        family = {
            "e8_256": "NVQ2",
            "d4_256": "NVQ3",
        }.get(tensor.spec.codebook)
        if family is None:
            raise ValueError(
                f"{tensor.spec.codebook} requires an NvqJscTensor expert profile"
            )
        return family
    raise TypeError(f"unsupported NINTM cohort tensor: {type(tensor)!r}")


@dataclass(frozen=True)
class NintMoePool:
    """One homogeneous precision cohort and its global expert IDs."""

    expert_ids: np.ndarray
    tensor: ExpertPoolTensor


@dataclass(frozen=True)
class NintMoeTensor:
    """One logical ``[experts, out, in]`` tensor with per-expert precision."""

    shape: tuple[int, int, int]
    pools: tuple[NintMoePool, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(int(value) <= 0 for value in self.shape):
            raise ValueError("NINTM shape must be [experts, out, in]")
        n_experts, out_per_expert, neuron_len = (int(value) for value in self.shape)
        if not self.pools:
            raise ValueError("NINTM must contain at least one precision pool")
        owners = np.full(n_experts, -1, dtype=np.int32)
        for pool_index, pool in enumerate(self.pools):
            expert_ids = np.ascontiguousarray(pool.expert_ids, dtype=np.int32).reshape(-1)
            if expert_ids.size == 0:
                raise ValueError("NINTM precision pools cannot be empty")
            if np.any(expert_ids < 0) or np.any(expert_ids >= n_experts):
                raise ValueError(f"NINTM pool {pool_index} contains an invalid expert id")
            if np.unique(expert_ids).size != expert_ids.size:
                raise ValueError(f"NINTM pool {pool_index} repeats an expert id")
            if np.any(owners[expert_ids] >= 0):
                raise ValueError("an expert belongs to multiple NINTM pools")
            owners[expert_ids] = pool_index

            tensor = pool.tensor
            expert_tensor_family(tensor)
            if isinstance(tensor, NepqTensor):
                expected_shape = (expert_ids.size, out_per_expert, neuron_len)
                valid = tuple(tensor.shape) == expected_shape
            else:
                expected_shape = (expert_ids.size * out_per_expert, neuron_len)
                valid = tuple(tensor.shape) == expected_shape and tensor.axis == 0
            if not valid:
                raise ValueError(
                    f"NINTM pool {pool_index} tensor shape {tensor.shape} must be {expected_shape}"
                )
        missing = np.flatnonzero(owners < 0)
        if missing.size:
            raise ValueError(f"NINTM pools do not cover experts {missing[:16].tolist()}")

    @property
    def n_experts(self) -> int:
        return int(self.shape[0])

    @property
    def out_per_expert(self) -> int:
        return int(self.shape[1])

    @property
    def neuron_len(self) -> int:
        return int(self.shape[2])

    @property
    def expert_profiles(self) -> tuple[str, ...]:
        result = [""] * self.n_experts
        for pool in self.pools:
            profile = expert_tensor_family(pool.tensor)
            for expert in np.asarray(pool.expert_ids).reshape(-1):
                result[int(expert)] = profile
        return tuple(result)
