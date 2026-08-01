"""Expert-wise mixed-family quantization for the NINTM container."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mfq.calibration.artifact import ExpertPrecision, nint_expert_precision
from mfq.formats.tpq import CccpPqTensor
from mfq.formats.tpq import (
    TPQ_PQ_SPECS_BY_LABEL,
    legacy_cccp_dtype,
    normalize_tpq_dtype,
)
from mfq.formats.moe import NintMoePool, NintMoeTensor
from mfq.formats.nepq import NepqTensor
from mfq.formats.nint import NintSpec
from mfq.formats.npq0_l import (
    Npq0LTensor,
    pack_npq0_l_tables,
    unpack_npq0_l_tables,
)
from mfq.formats.npq0_s import Npq0STensor, pack_npq0_s_tables
from mfq.formats.nvq import (
    NVQ2_E8,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_512,
    NVQ3_D4_1024,
    NvqJscTensor,
    NvqTensor,
    unpack_jsc_metadata,
)
from mfq.formats.nvq1_l import (
    NVQ1_L_T8_S3,
    Nvq1LTensor,
    pack_ternary_codebook,
)
from mfq.formats.nvq1_s import Nvq1STensor, pack_nvq1_s_banked_codebook
from mfq.quantize.nepq import NepqQuantConfig, quantize_nepq_fixed
from mfq.quantize.tpq import (
    CccpKmeansConfig,
    dequantize_cccp_pq,
    quantize_cccp_pq_fixed,
    train_cccp_pq,
)
from mfq.quantize.nint_quant import NintTensor
from mfq.quantize.nint_quant import dequantize as dequantize_nint
from mfq.quantize.nint_quant import quantize as quantize_nint
from mfq.quantize.npq0_l import (
    Npq0LConfig,
    Npq0LTables,
    dequantize_npq0_l,
    quantize_npq0_l_fixed,
    train_npq0_l,
)
from mfq.quantize.npq0_s import (
    Npq0SConfig,
    Npq0STables,
    dequantize_npq0_s,
    quantize_npq0_s_fixed,
    train_npq0_s,
)
from mfq.quantize.nvq1_l_quant import dequantize as dequantize_nvq1_l
from mfq.quantize.nvq1_l_quant import quantize as quantize_nvq1_l
from mfq.quantize.nvq1_s_quant import dequantize as dequantize_nvq1_s
from mfq.quantize.nvq1_s_quant import quantize as quantize_nvq1_s
from mfq.quantize.nvq_jsc import (
    NvqJscConfig,
    NvqJscTables,
    dequantize_nvq_jsc,
    quantize_nvq_jsc_fixed,
    train_nvq_jsc,
)
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq
from mfq.quantize.nvq_quant import quantize as quantize_nvq

ExpertProfile = NintSpec | ExpertPrecision
ArtifactMap = Mapping[ExpertPrecision | str, object]

_NVQ_SPECS = {
    "NVQ2": NVQ2_E8,
    "NVQ2J": NVQ2_E8,
    "NVQ2J-L": NVQ2_E8_1024,
    "NVQ2J-XL": NVQ2_E8_4096,
    "NVQ3": NVQ3_D4,
    "NVQ3J": NVQ3_D4,
    "NVQ3J-512": NVQ3_D4_512,
    "NVQ3J-L": NVQ3_D4_1024,
}


def _normalize_profile(value: ExpertProfile) -> ExpertPrecision:
    if isinstance(value, NintSpec):
        return nint_expert_precision(value)
    if isinstance(value, ExpertPrecision):
        return value
    raise TypeError(f"unsupported expert precision descriptor: {type(value)!r}")


def _ordered_profiles(
    profiles: Sequence[ExpertProfile] | Mapping[int, ExpertProfile],
    n_experts: int,
) -> tuple[ExpertPrecision, ...]:
    if isinstance(profiles, Mapping):
        missing = sorted(set(range(n_experts)) - set(int(key) for key in profiles))
        if missing:
            raise ValueError(f"missing precision descriptors for experts {missing[:16]}")
        values = tuple(profiles[expert] for expert in range(n_experts))
    else:
        values = tuple(profiles)
        if len(values) != n_experts:
            raise ValueError(
                f"received {len(values)} precision descriptors for {n_experts} experts"
            )
    return tuple(_normalize_profile(value) for value in values)


def _artifact_path(
    precision: ExpertPrecision,
    artifact_root: str | Path | None,
) -> Path | None:
    if precision.artifact is None:
        return None
    path = Path(precision.artifact)
    if not path.is_absolute():
        if artifact_root is None:
            raise ValueError(
                f"{precision.family} artifact is relative but no artifact root was supplied"
            )
        path = Path(artifact_root) / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{precision.family} quantizer artifact does not exist: {resolved}"
        )
    return resolved


def _artifact_value(
    precision: ExpertPrecision,
    artifacts: ArtifactMap | None,
    artifact_root: str | Path | None,
) -> object | None:
    if artifacts is not None:
        if precision in artifacts:
            return artifacts[precision]
        if precision.family in artifacts:
            return artifacts[precision.family]
        legacy_family = legacy_cccp_dtype(precision.family)
        if legacy_family in artifacts:
            return artifacts[legacy_family]
        if precision.artifact is not None and precision.artifact in artifacts:
            return artifacts[precision.artifact]
    path = _artifact_path(precision, artifact_root)
    return None if path is None else _load_precision_artifact(precision, path)


def resolve_precision_artifact(
    precision: ExpertPrecision,
    *,
    artifacts: ArtifactMap | None = None,
    artifact_root: str | Path | None = None,
) -> object | None:
    """Resolve one in-memory or on-disk cohort quantizer artifact."""

    return _artifact_value(precision, artifacts, artifact_root)


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).copy() for name in payload.files}


def _nepq_tables_from_arrays(
    precision: ExpertPrecision,
    arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    family = precision.family
    direct = arrays.get("table_payloads")
    if direct is not None:
        return np.ascontiguousarray(direct, dtype=np.uint8)
    if family == "NEPQ0-S":
        pack = pack_npq0_s_tables
    elif family == "NEPQ0-L":
        pack = pack_npq0_l_tables
    elif family == "NEPQ1-S":
        codebooks = np.asarray(arrays["codebooks"])
        return np.stack(
            [
                np.frombuffer(pack_nvq1_s_banked_codebook(table), dtype=np.uint8)
                for table in codebooks
            ]
        )
    elif family == "NEPQ1-L":
        codebooks = np.asarray(arrays["codebooks"])
        return np.stack(
            [
                np.frombuffer(pack_ternary_codebook(table), dtype=np.uint8)
                for table in codebooks
            ]
        )
    else:
        raise ValueError(f"{family} is not a NEPQ family")
    scale = np.asarray(arrays["scale_lut"])
    first = np.asarray(arrays["first_codebooks"])
    second = np.asarray(arrays["second_codebooks"])
    if scale.ndim != 2 or first.shape[0] != scale.shape[0] or second.shape[0] != scale.shape[0]:
        raise ValueError(f"{family} artifact contains inconsistent bank arrays")
    return np.stack(
        [
            np.frombuffer(pack(scale[index], first[index], second[index]), dtype=np.uint8)
            for index in range(scale.shape[0])
        ]
    )


def _load_precision_artifact(
    precision: ExpertPrecision,
    path: Path,
) -> object:
    family = precision.family
    if path.suffix.lower() == ".npz":
        arrays = _npz_arrays(path)
        if family in TPQ_PQ_SPECS_BY_LABEL:
            if "codebook" not in arrays:
                raise ValueError(f"{family} artifact lacks a codebook: {path}")
            stored_family = arrays.get("family")
            if (
                stored_family is not None
                and normalize_tpq_dtype(str(stored_family.item())) != family
            ):
                raise ValueError(
                    f"{family} artifact declares {stored_family.item()!r}: {path}"
                )
            return np.ascontiguousarray(arrays["codebook"], dtype=np.float32)
        if family == "NPQ0-L":
            return Npq0LTables(
                arrays["scale_lut"],
                arrays["first_codebooks"],
                arrays["second_codebooks"],
            )
        if family == "NPQ0-S":
            return Npq0STables(
                arrays["scale_lut"],
                arrays["first_codebooks"],
                arrays["second_codebooks"],
            )
        if family in {
            "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
            "NVQ3J", "NVQ3J-512", "NVQ3J-L",
        }:
            return NvqJscTables(
                arrays["scale_lut"],
                arrays["bank_for_state"],
                arrays["codebooks"],
                _NVQ_SPECS[family],
            )
        if family.startswith("NEPQ"):
            return _nepq_tables_from_arrays(precision, arrays)
        if "codebook" in arrays:
            return arrays["codebook"]
        raise ValueError(f"{family} artifact lacks a recognized payload: {path}")

    if path.suffix.lower() != ".json":
        raise ValueError(f"unsupported quantizer artifact extension: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = base64.b64decode(document["tables_b64"], validate=True)
    if family in {
        "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
        "NVQ3J", "NVQ3J-512", "NVQ3J-L",
    }:
        scale, bank, codebooks, consumed = unpack_jsc_metadata(
            raw,
            vector_size=_NVQ_SPECS[family].vector_size,
            codebook_entries=_NVQ_SPECS[family].codebook_entries,
        )
        if consumed != len(raw):
            raise ValueError(f"{family} artifact has an invalid table tail: {path}")
        return NvqJscTables(scale, bank, codebooks, _NVQ_SPECS[family])
    if family == "NPQ0-L":
        scale, first, second, consumed = unpack_npq0_l_tables(raw)
        if consumed != len(raw):
            raise ValueError(f"NPQ0-L artifact has an invalid table tail: {path}")
        return Npq0LTables(scale, first, second)
    raise ValueError(f"{family} does not support JSON table artifacts")


def _cohort_importance(
    importance: np.ndarray | torch.Tensor | None,
    expert_ids: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray | torch.Tensor | None:
    if importance is None:
        return None
    value_shape = tuple(int(value) for value in importance.shape)
    if value_shape == (shape[2],):
        return importance
    if value_shape != shape:
        raise ValueError(f"expert importance must have shape [{shape[2]}] or {shape}")
    if isinstance(importance, torch.Tensor):
        index = torch.as_tensor(
            expert_ids, device=importance.device, dtype=torch.int64
        )
        return importance.index_select(0, index)
    return importance[expert_ids]


def _numpy_importance(
    importance: np.ndarray | torch.Tensor | None,
) -> np.ndarray | None:
    if importance is None:
        return None
    if isinstance(importance, torch.Tensor):
        return importance.detach().cpu().numpy()
    return np.asarray(importance)


def _option_int(precision: ExpertPrecision, name: str, default: int) -> int:
    value = int(precision.option(name, default))
    return value


def _quantize_flat_cohort(
    rows: np.ndarray,
    precision: ExpertPrecision,
    artifact: object | None,
    importance: np.ndarray | torch.Tensor | None,
    device: str | torch.device,
):
    family = precision.family
    if family.startswith("NINT"):
        assert precision.nint_spec is not None
        if artifact is not None:
            raise ValueError(f"{family} does not consume a quantizer artifact")
        return quantize_nint(
            rows,
            precision.nint_spec,
            axis=0,
            importance=(
                _numpy_importance(importance)
                if precision.nint_spec.bits in {2, 3, 4, 5, 6}
                else None
            ),
        )
    if family in {"NVQ2", "NVQ3"}:
        codebook = None if artifact is None else np.asarray(artifact)
        return quantize_nvq(
            rows,
            _NVQ_SPECS[family],
            axis=0,
            importance=_numpy_importance(importance),
            search_steps=_option_int(precision, "search_steps", 19),
            group_chunk=_option_int(precision, "group_chunk", 1024),
            codebook=codebook,
        )
    if family in TPQ_PQ_SPECS_BY_LABEL:
        spec = TPQ_PQ_SPECS_BY_LABEL[family]
        config = CccpKmeansConfig(
            iterations=_option_int(precision, "iterations", 12),
            restarts=_option_int(precision, "restarts", 2),
            sample_points=_option_int(precision, "sample_points", 100_000),
            seed=_option_int(precision, "seed", 0),
            distance_bytes=_option_int(
                precision, "distance_bytes", 1 << 30
            ),
        )
        weight = torch.as_tensor(rows, dtype=torch.float32)
        if artifact is None:
            tensor, _ = train_cccp_pq(
                weight,
                spec,
                config=config,
                device=device,
            )
            return tensor
        return quantize_cccp_pq_fixed(
            weight,
            spec,
            np.asarray(artifact, dtype=np.float32),
            device=device,
            distance_bytes=config.distance_bytes,
        )
    if family == "NVQ1-L":
        codebook = None if artifact is None else np.asarray(artifact)
        return quantize_nvq1_l(
            rows,
            NVQ1_L_T8_S3,
            axis=0,
            importance=_numpy_importance(importance),
            refine_steps=_option_int(precision, "refine_steps", 2),
            group_chunk=_option_int(precision, "group_chunk", 64),
            codebook=codebook,
        )
    if family == "NVQ1-S":
        kwargs: dict[str, Any] = {}
        if artifact is not None:
            kwargs["codebook"] = np.asarray(artifact)
        return quantize_nvq1_s(
            rows,
            axis=0,
            importance=_numpy_importance(importance),
            refine_steps=_option_int(precision, "refine_steps", 2),
            group_chunk=_option_int(precision, "group_chunk", 64),
            **kwargs,
        )

    weight = torch.as_tensor(rows, dtype=torch.float32)
    if family in {
        "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
        "NVQ3J", "NVQ3J-512", "NVQ3J-L",
    }:
        spec = _NVQ_SPECS[family]
        config = NvqJscConfig(
            banks=_option_int(precision, "banks", 4),
            iterations=_option_int(precision, "iterations", 4),
            assignment_refine_steps=_option_int(
                precision, "assignment_refine_steps", 2
            ),
            search_steps=_option_int(precision, "search_steps", 19),
            group_chunk=_option_int(precision, "group_chunk", 1024),
            spec=spec,
        )
        if artifact is None:
            tensor, _ = train_nvq_jsc(
                weight, importance=importance, config=config, device=device
            )
            return tensor
        if not isinstance(artifact, NvqJscTables):
            raise TypeError(f"{family} artifact must contain NvqJscTables")
        return quantize_nvq_jsc_fixed(
            weight,
            artifact,
            importance=importance,
            assignment_refine_steps=config.assignment_refine_steps,
            search_steps=config.search_steps,
            group_chunk=config.group_chunk,
            device=device,
        )
    if family == "NPQ0-L":
        config = Npq0LConfig(
            iterations=_option_int(precision, "iterations", 4),
            assignment_refine_steps=_option_int(
                precision, "assignment_refine_steps", 2
            ),
            fixed_refine_steps=_option_int(precision, "fixed_refine_steps", 3),
            group_chunk=_option_int(precision, "group_chunk", 512),
        )
        if artifact is None:
            tensor, _ = train_npq0_l(
                weight, importance=importance, config=config, device=device
            )
            return tensor
        if not isinstance(artifact, Npq0LTables):
            raise TypeError("NPQ0-L artifact must contain Npq0LTables")
        return quantize_npq0_l_fixed(
            weight, artifact, importance=importance, config=config, device=device
        )
    if family == "NPQ0-S":
        config = Npq0SConfig(
            iterations=_option_int(precision, "iterations", 4),
            assignment_refine_steps=_option_int(
                precision, "assignment_refine_steps", 2
            ),
            fixed_refine_steps=_option_int(precision, "fixed_refine_steps", 3),
            group_chunk=_option_int(precision, "group_chunk", 256),
        )
        if artifact is None:
            tensor, _ = train_npq0_s(
                weight, importance=importance, config=config, device=device
            )
            return tensor
        if not isinstance(artifact, Npq0STables):
            raise TypeError("NPQ0-S artifact must contain Npq0STables")
        return quantize_npq0_s_fixed(
            weight, artifact, importance=importance, config=config, device=device
        )
    raise ValueError(f"{family} is not a flat expert precision")


def quantize_flat_cohort(
    rows: np.ndarray,
    precision: ExpertPrecision,
    *,
    artifact: object | None = None,
    importance: np.ndarray | torch.Tensor | None = None,
    device: str | torch.device = "cuda",
):
    """Quantize one flattened homogeneous expert cohort."""

    return _quantize_flat_cohort(
        np.ascontiguousarray(rows, dtype=np.float32),
        precision,
        artifact,
        importance,
        device,
    )


def quantize_expertwise(
    weight: np.ndarray | torch.Tensor,
    profiles: Sequence[ExpertProfile] | Mapping[int, ExpertProfile],
    *,
    artifacts: ArtifactMap | None = None,
    artifact_root: str | Path | None = None,
    importance: np.ndarray | torch.Tensor | None = None,
    device: str | torch.device = "cuda",
) -> NintMoeTensor:
    """Quantize ``[experts,out,K]`` into native family cohorts."""

    values = (
        weight.detach().cpu().numpy()
        if isinstance(weight, torch.Tensor)
        else np.asarray(weight)
    )
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("expert-wise quantization expects [experts, out, in]")
    shape = tuple(int(value) for value in values.shape)
    n_experts, out_per_expert, neuron_len = shape
    expert_profiles = _ordered_profiles(profiles, n_experts)
    cohorts: dict[ExpertPrecision, list[int]] = {}
    for expert, precision in enumerate(expert_profiles):
        cohorts.setdefault(precision, []).append(expert)

    pools: list[NintMoePool] = []
    for precision, expert_ids_list in cohorts.items():
        expert_ids = np.asarray(expert_ids_list, dtype=np.int32)
        artifact = _artifact_value(precision, artifacts, artifact_root)
        cohort_importance = _cohort_importance(
            importance, expert_ids, shape
        )
        if precision.family.startswith("NEPQ"):
            if artifact is None:
                raise ValueError(
                    f"{precision.family} requires a frozen cross-expert table artifact"
                )
            tensor = quantize_nepq_fixed(
                np.ascontiguousarray(values[expert_ids]),
                precision.family,
                artifact,
                importance=cohort_importance,
                rotation_block=_option_int(precision, "rotation_block", 0),
                rotation_seed=_option_int(precision, "rotation_seed", 0),
                config=NepqQuantConfig(
                    refine_steps=_option_int(precision, "refine_steps", 2),
                    row_chunk=_option_int(precision, "row_chunk", 8),
                    bank_chunk=_option_int(precision, "bank_chunk", 8),
                ),
                device=device,
            )
        else:
            rows = np.ascontiguousarray(values[expert_ids]).reshape(
                expert_ids.size * out_per_expert, neuron_len
            )
            flat_importance = cohort_importance
            if flat_importance is not None and tuple(flat_importance.shape) == (
                expert_ids.size,
                out_per_expert,
                neuron_len,
            ):
                flat_importance = flat_importance.reshape(rows.shape)
            tensor = _quantize_flat_cohort(
                rows,
                precision,
                artifact,
                flat_importance,
                device,
            )
        pools.append(NintMoePool(expert_ids=expert_ids, tensor=tensor))
    return NintMoeTensor(shape=shape, pools=tuple(pools))


def _dequantize_pool(tensor: object) -> np.ndarray:
    if isinstance(tensor, NintTensor):
        return dequantize_nint(tensor)
    if isinstance(tensor, NepqTensor):
        from mfq.formats.nepq import dequantize_nepq

        return dequantize_nepq(tensor)
    if isinstance(tensor, CccpPqTensor):
        return dequantize_cccp_pq(tensor)
    if isinstance(tensor, NvqJscTensor):
        return dequantize_nvq_jsc(tensor)
    if isinstance(tensor, NvqTensor):
        return dequantize_nvq(tensor)
    if isinstance(tensor, Nvq1LTensor):
        return dequantize_nvq1_l(tensor)
    if isinstance(tensor, Nvq1STensor):
        return dequantize_nvq1_s(tensor)
    if isinstance(tensor, Npq0LTensor):
        return dequantize_npq0_l(tensor)
    if isinstance(tensor, Npq0STensor):
        return dequantize_npq0_s(tensor)
    raise TypeError(f"unsupported NINTM cohort tensor: {type(tensor)!r}")


def dequantize_expertwise(tensor: NintMoeTensor) -> np.ndarray:
    """Restore any NINTM family combination to ``float32 [experts,out,K]``."""

    result = np.empty(tensor.shape, dtype=np.float32)
    for pool in tensor.pools:
        expert_ids = np.asarray(pool.expert_ids, dtype=np.int32)
        rows = _dequantize_pool(pool.tensor).reshape(
            expert_ids.size, tensor.out_per_expert, tensor.neuron_len
        )
        result[expert_ids] = rows
    return result


__all__ = [
    "ArtifactMap",
    "ExpertProfile",
    "dequantize_expertwise",
    "quantize_flat_cohort",
    "quantize_expertwise",
    "resolve_precision_artifact",
]
