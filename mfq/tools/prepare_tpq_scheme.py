"""Build an MFQ TPQ scheme from scores or fixed per-expert tiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mfq.calibration.artifact import CalibrationScheme, save_scheme
from mfq.calibration.tpq import (
    CccpTierAllocation,
    allocate_cccp_tiers,
    build_cccp_expert_selection,
    load_cccp_score_profile,
    load_cccp_tier_profile,
)


def prepare(
    *,
    profile_path: str | Path,
    output_path: str | Path,
    field: str = "counts",
    rows_gate_up: int = 4096,
    columns_gate_up: int = 4096,
    rows_down: int = 4096,
    columns_down: int = 2048,
    v_coverage: float = 0.965,
    w_coverage: float = 0.997,
    vv_share: float = 0.25,
) -> CalibrationScheme:
    """Create per-layer CCCP selections from scores or ``tiers_per_layer``."""

    fixed_tiers = load_cccp_tier_profile(profile_path)
    scores = (
        {}
        if fixed_tiers is not None
        else load_cccp_score_profile(profile_path, field=field)
    )
    layers = fixed_tiers if fixed_tiers is not None else scores
    expert_selections = {}
    candidate_table: dict[str, list[dict]] = {}
    for layer, layer_values in sorted(layers.items()):
        if fixed_tiers is None:
            layer_scores = scores[layer]
            allocation = allocate_cccp_tiers(
                layer_scores,
                v_coverage=v_coverage,
                w_coverage=w_coverage,
                vv_share=vv_share,
            )
        else:
            tiers = tuple(layer_values)
            layer_scores = np.ones(len(tiers), dtype=np.float64)
            allocation = CccpTierAllocation(
                tiers=tiers,
                scores=tuple(float(value) for value in layer_scores),
                boundaries=(
                    sum(tier in {"vv", "v"} for tier in tiers),
                    sum(tier in {"vv", "v", "w"} for tier in tiers),
                ),
                score_mass=tuple(
                    (
                        tier,
                        tiers.count(tier) / len(tiers),
                    )
                    for tier in ("vv", "v", "w", "x")
                ),
            )
        for projection, rows, columns in (
            ("gate_up", rows_gate_up, columns_gate_up),
            ("down", rows_down, columns_down),
        ):
            name = f"blk.{layer}.ffn_{projection}_exps.weight"
            artifacts = {
                tier: (
                    f"artifacts/layer{layer:02d}-{projection}-"
                    f"tpq-{tier}.npz"
                )
                for tier in set(allocation.tiers)
            }
            expert_selections[name] = build_cccp_expert_selection(
                name=name,
                group=projection,
                allocation=allocation,
                rows_per_expert=rows,
                columns=columns,
                artifacts=artifacts,
            )
            candidate_table[name] = [
                {
                    "expert_id": expert,
                    "tier": tier,
                    **(
                        {"source_tier": tier}
                        if fixed_tiers is not None
                        else {"score": allocation.scores[expert]}
                    ),
                }
                for expert, tier in enumerate(allocation.tiers)
            ]
    scheme = CalibrationScheme(
        path=Path(output_path).resolve(),
        target_profile="TPQ-EW",
        target_storage_bits=0,
        selections={},
        metadata={
            "allocator": "tpq-per-layer-routing-energy",
            "codebook_objective": "euclidean_sse",
            "codebook_calibration_data": "none",
            "tier_source": (
                "fixed_tiers_per_layer"
                if fixed_tiers is not None
                else "routing_scores"
            ),
            "profile": str(Path(profile_path).resolve()),
            "score_field": field,
            "v_coverage": v_coverage,
            "w_coverage": w_coverage,
            "vv_share": vv_share,
            "layers": len(layers),
        },
        candidate_table=candidate_table,
        expert_selections=expert_selections,
    )
    scheme = CalibrationScheme(
        path=scheme.path,
        target_profile=scheme.target_profile,
        target_storage_bits=scheme.storage_bits,
        selections=scheme.selections,
        metadata=scheme.metadata,
        candidate_table=scheme.candidate_table,
        expert_selections=scheme.expert_selections,
    )
    save_scheme(output_path, scheme)
    return scheme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--field", default="counts")
    parser.add_argument("--rows-gate-up", type=int, default=4096)
    parser.add_argument("--columns-gate-up", type=int, default=4096)
    parser.add_argument("--rows-down", type=int, default=4096)
    parser.add_argument("--columns-down", type=int, default=2048)
    parser.add_argument("--v-coverage", type=float, default=0.965)
    parser.add_argument("--w-coverage", type=float, default=0.997)
    parser.add_argument("--vv-share", type=float, default=0.25)
    args = parser.parse_args()
    scheme = prepare(
        profile_path=args.profile,
        output_path=args.output,
        field=args.field,
        rows_gate_up=args.rows_gate_up,
        columns_gate_up=args.columns_gate_up,
        rows_down=args.rows_down,
        columns_down=args.columns_down,
        v_coverage=args.v_coverage,
        w_coverage=args.w_coverage,
        vv_share=args.vv_share,
    )
    print(
        json.dumps(
            {
                "output": str(scheme.path),
                "expert_tensors": len(scheme.expert_selections),
                "storage_bits": scheme.storage_bits,
                "bpw": scheme.bpw,
            }
        )
    )


if __name__ == "__main__":
    main()
