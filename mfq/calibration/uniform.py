"""Calibration-free importance matrices from the HF converter's metadata plans."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mfq.architectures.hf_config import load_hf_model_config, text_section
from mfq.calibration.artifact import CalibrationScheme
from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix, save_importance_matrix


def uniform_imatrix(model_path: str | Path, output: str | Path) -> ImportanceMatrix:
    """Give every planned matrix equal input importance without reading weights."""
    from mfq.tools import quantize_hf_to_mfq as q

    root = Path(model_path).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"imatrix already exists: {output_path}")
    config = load_hf_model_config(root)
    inventory = q._hf_source_inventory(root)
    entries: dict[str, ImportanceEntry] = {}
    # Schemes/recipes can split a fused source into different logical matrices.
    # Cover both layouts using the same inventory, including auxiliary scopes.
    empty = CalibrationScheme(None, "uniform", 0, {}, {}, {})
    for scheme in (None, empty):
        plans = q._plan(
            root, False, None, "F32", scheme,
            quantize_vision=True, quantize_mtp=True, quantize_ple=True,
            source_inventory=inventory, source_config=config,
        )
        for item in plans:
            if len(item.shape) not in (2, 3) or item.source_dtype in q._HF_INTEGER_DTYPES:
                continue
            shape, _ = q._hf_imatrix_shapes(item)
            matrices = int(shape[0]) if len(shape) == 3 else 1
            width = int(shape[-1])
            name = q._hf_imatrix_names(item)[0]
            previous = entries.get(name)
            if previous is not None:
                if (previous.matrices, previous.width) != (matrices, width):
                    raise ValueError(f"conflicting uniform imatrix shapes for {name}: {item.name}")
                continue
            entries[name] = ImportanceEntry(
                np.ones((matrices, width), dtype=np.float32),
                np.zeros(matrices, dtype=np.int64),
            )
    return save_importance_matrix(
        output_path,
        entries,
        metadata={
            "objective": "uniform_calibration_free",
            "model_type": str(text_section(config).get("model_type", "")),
            "tokens": 0,
            "split": "none",
        },
    )


__all__ = ["uniform_imatrix"]
