"""Allocate an independent gate/down REAP scheme from cached expert candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mfq.calibration.artifact import save_scheme
from mfq.calibration.reap_expertwise import (
    allocate_expert_profiles,
    allocate_independent_expert_profiles,
    evaluation_from_document,
)
from mfq.tools.quantize_hf_to_mfq import _load_gguf_recipe


_RECIPE_PROFILE = {
    "Q3_K": "NINT3",
    "Q4_K": "NINT4",
    "Q5_0": "NINT5",
    "Q5_1": "NINT5",
    "Q5_K": "NINT5",
    "Q6_K": "NINT6",
    "Q8_0": "NINT8",
    "Q8_K": "NINT8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_evaluations(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return [
            evaluation_from_document(json.loads(line))
            for line in stream
            if line.strip()
        ]


def _ud_baseline_profiles(recipe_path: Path, evaluations) -> dict[str, str]:
    recipe = _load_gguf_recipe(recipe_path)
    profiles: dict[str, str] = {}
    groups = sorted({(item.layer, item.expert) for item in evaluations})
    tensor_names = {
        (item.layer, "gate"): item.gate_name
        for item in evaluations
    }
    tensor_names.update({
        (item.layer, "down"): item.down_name
        for item in evaluations
    })
    from mfq.tools.quantize_hf_to_mfq import _hf_to_gguf_name

    for layer, expert in groups:
        for kind in ("gate", "down"):
            gguf_name = _hf_to_gguf_name(tensor_names[(layer, kind)])
            if gguf_name is None:
                raise ValueError(f"cannot map layer {layer} {kind} expert tensor to GGUF")
            gguf_type = recipe.get(gguf_name)
            if gguf_type not in _RECIPE_PROFILE:
                raise ValueError(
                    f"UD recipe tensor {gguf_name} has unsupported type {gguf_type!r}"
                )
            profile = _RECIPE_PROFILE[gguf_type]
            profiles[f"layer.{layer}.expert.{expert}.{kind}"] = profile
    return profiles


def allocate_scheme(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate_table).resolve()
    recipe_path = Path(args.recipe_gguf).resolve()
    scheme_path = Path(args.output_scheme).resolve()
    report_path = Path(args.report).resolve()
    for path in (scheme_path, report_path):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
    evaluations = _load_evaluations(candidate_path)
    candidate_sha256 = _sha256(candidate_path)
    recipe_sha256 = _sha256(recipe_path)
    metadata = {
        "candidate_table": str(candidate_path),
        "candidate_table_sha256": candidate_sha256,
        "baseline_recipe": str(recipe_path),
        "baseline_recipe_sha256": recipe_sha256,
    }
    if args.coupling == "coupled":
        scheme, report = allocate_expert_profiles(
            evaluations,
            target_profile=args.coupled_reference_profile,
            target_storage_bits=args.target_storage_bits,
            target_label=args.target_label,
            metadata=metadata,
        )
    else:
        baseline_profiles = _ud_baseline_profiles(recipe_path, evaluations)
        scheme, report = allocate_independent_expert_profiles(
            evaluations,
            target_storage_bits=args.target_storage_bits,
            baseline_profiles=baseline_profiles,
            target_label=args.target_label,
            metadata=metadata,
        )
    save_scheme(scheme_path, scheme)
    report.update(
        {
            "candidate_table": str(candidate_path),
            "candidate_table_sha256": candidate_sha256,
            "candidate_count": len(evaluations),
            "baseline_recipe": str(recipe_path),
            "baseline_recipe_sha256": recipe_sha256,
            "scheme": str(scheme_path),
            "scheme_storage_bits": scheme.storage_bits,
            "scheme_bpw": scheme.bpw,
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(report_path)
    print(json.dumps({"status": "ok", **report}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--recipe-gguf", required=True)
    parser.add_argument("--target-storage-bits", required=True, type=int)
    parser.add_argument("--target-label", default="UD_BPW")
    parser.add_argument(
        "--coupling",
        choices=("independent", "coupled"),
        default="independent",
    )
    parser.add_argument(
        "--coupled-reference-profile",
        choices=("NINT4", "NINT5", "NINT6", "NINT8"),
        default="NINT5",
    )
    parser.add_argument("--output-scheme", required=True)
    parser.add_argument("--report", required=True)
    allocate_scheme(parser.parse_args())


if __name__ == "__main__":
    main()
