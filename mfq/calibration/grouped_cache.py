"""Content-addressed caches for grouped expert quantization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mfq.formats.nint import NintSpec


def grouped_expert_cache_layout(
    model_root: str | Path,
    index,
    source_name: str,
    spec: NintSpec,
    cache_dir: str | Path,
    *,
    imatrix_sha256: str = "",
) -> tuple[Path, dict[str, object]]:
    """Return a stable cache location for one grouped expert tensor."""

    root = Path(model_root).resolve()
    cache = Path(cache_dir).resolve()
    shard = root / index.weight_map[source_name]
    stat = shard.stat()
    shape = index.shape(source_name)
    metadata: dict[str, object] = {
        "format": "mfq.moe-grouped-nint-cache.v3",
        "model": str(root),
        "source_name": source_name,
        "bits": spec.bits,
        "groupsize": spec.groupsize,
        "sub_bits": spec.sub_bits,
        "shape": list(shape),
        "axis": 0,
        "neuron_len": shape[2],
        "source_shard_bytes": stat.st_size,
        "source_shard_mtime_ns": stat.st_mtime_ns,
        **({"imatrix_sha256": imatrix_sha256} if imatrix_sha256 else {}),
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return cache / "grouped-v3" / digest, metadata


__all__ = ["grouped_expert_cache_layout"]
