"""Extract exact routed-layer payload budgets and the non-routed UD recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


_ROUTED = re.compile(
    r"blk\.(?P<layer>\d+)\.ffn_(?P<kind>down|gate_up)_exps\.weight"
)


def logical_tensor_elements(shape: Iterable[int]) -> int:
    return int(np.prod(tuple(shape), dtype=np.int64))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_ud_layer_budget_document(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_layers: int,
    expected_experts: int,
    expected_top_k: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if expected_layers <= 0 or expected_experts <= 0 or expected_top_k <= 0:
        raise ValueError("expected model dimensions must be positive")
    layer_bits: dict[int, int] = defaultdict(int)
    layer_elements: dict[int, int] = defaultdict(int)
    layer_types: dict[int, dict[str, str]] = defaultdict(dict)
    non_routed_recipe: dict[str, str] = {}
    for record in records:
        name = str(record["name"])
        tensor_type = str(record["tensor_type"])
        payload_bytes = int(record["payload_bytes"])
        elements = int(record["elements"])
        if payload_bytes <= 0 or elements <= 0:
            raise ValueError(f"tensor {name} has invalid size")
        match = _ROUTED.fullmatch(name)
        if match is None:
            non_routed_recipe[name] = tensor_type
            continue
        layer = int(match.group("layer"))
        kind = match.group("kind")
        if not 0 <= layer < expected_layers:
            raise ValueError(f"routed tensor {name} is outside expected layers")
        if kind in layer_types[layer]:
            raise ValueError(f"duplicate routed tensor for layer {layer}/{kind}")
        layer_types[layer][kind] = tensor_type
        layer_bits[layer] += payload_bytes * 8
        layer_elements[layer] += elements

    layers: dict[str, Any] = {}
    for layer in range(expected_layers):
        if set(layer_types.get(layer, {})) != {"gate_up", "down"}:
            raise ValueError(
                f"layer {layer} must contain both routed tensors"
            )
        bits = layer_bits[layer]
        elements = layer_elements[layer]
        layers[str(layer)] = {
            "target_storage_bits": bits,
            "routed_elements": elements,
            "effective_bpw": bits / elements,
            "source_types": dict(sorted(layer_types[layer].items())),
        }
    return {
        "format": "mfq.ud-layer-budgets.v1",
        "expected_layers": expected_layers,
        "expected_experts": expected_experts,
        "expected_top_k": expected_top_k,
        "source": dict(source),
        "layers": layers,
        "non_routed_recipe": dict(sorted(non_routed_recipe.items())),
    }


def extract_ud_layer_budget(
    *,
    gguf_path: str | Path,
    output_path: str | Path,
    expected_layers: int = 30,
    expected_experts: int = 128,
    expected_top_k: int = 8,
) -> dict[str, Any]:
    from gguf import GGUFReader

    source = Path(gguf_path).resolve()
    output = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    reader = GGUFReader(str(source), "r")
    records = [
        {
            "name": str(tensor.name),
            "tensor_type": str(tensor.tensor_type.name),
            "payload_bytes": int(tensor.data.nbytes),
            "elements": logical_tensor_elements(tensor.shape),
        }
        for tensor in reader.tensors
    ]
    document = build_ud_layer_budget_document(
        records,
        expected_layers=expected_layers,
        expected_experts=expected_experts,
        expected_top_k=expected_top_k,
        source={
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-layers", type=int, default=30)
    parser.add_argument("--expected-experts", type=int, default=128)
    parser.add_argument("--expected-top-k", type=int, default=8)
    args = parser.parse_args()
    document = extract_ud_layer_budget(
        gguf_path=args.gguf,
        output_path=args.output,
        expected_layers=args.expected_layers,
        expected_experts=args.expected_experts,
        expected_top_k=args.expected_top_k,
    )
    print(
        json.dumps(
            {
                "format": document["format"],
                "source": document["source"],
                "layers": len(document["layers"]),
                "non_routed_tensors": len(document["non_routed_recipe"]),
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
