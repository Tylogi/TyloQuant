"""Runtime assets embedded in an MFQ file.

Assets use ordinary MFQ records with a reserved name prefix and ``BLOB`` dtype.
This keeps the version-2 file table readable by older C++ runtimes: they see
the records but ignore them unless explicitly requested.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ASSET_PREFIX = "__mfq_asset__/"
MODEL_CONFIG_ASSET = ASSET_PREFIX + "model_config.json"
TOKENIZER_GGUF_ASSET = ASSET_PREFIX + "tokenizer.gguf"
MINICPMO45_RESAMPLER_POS_EMBED_ASSET = (
    ASSET_PREFIX + "minicpmo45-resampler-pos-embed-v1.bf16"
)
ASSET_DTYPE = "BLOB"
ASSET_MANIFEST_KEY = "runtime_assets"

_MINICPMO45_RESAMPLER_MAGIC = b"MFQRSPB1"
_MINICPMO45_RESAMPLER_HEADER = struct.Struct("<8sIII")


@dataclass(frozen=True)
class RuntimeAsset:
    name: str
    media_type: str
    data: bytes

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "record": self.name,
            "media_type": self.media_type,
            "bytes": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }


def is_asset_record(name: str) -> bool:
    return name.startswith(ASSET_PREFIX)


def model_config_asset(config: dict[str, Any] | bytes | str) -> RuntimeAsset:
    if isinstance(config, dict):
        data = json.dumps(
            config, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    elif isinstance(config, str):
        data = config.encode("utf-8")
    else:
        data = bytes(config)
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("model config must be a JSON object")
    return RuntimeAsset(MODEL_CONFIG_ASSET, "application/json", data)


def minicpmo45_resampler_pos_embed_asset(
    *,
    max_size: tuple[int, int] = (70, 70),
    embed_dim: int = 4096,
) -> RuntimeAsset:
    """Serialize the official non-persistent NumPy 2D position cache.

    MiniCPM-o 4.5 constructs this cache with NumPy and omits it from the
    checkpoint.  Saving its BF16 values avoids platform-dependent sin/cos
    differences when another runtime reconstructs the graph constant.
    """

    height, width = (int(value) for value in max_size)
    if height <= 0 or width <= 0:
        raise ValueError("MiniCPM-o Resampler maximum size must be positive")
    if embed_dim <= 0 or embed_dim % 4:
        raise ValueError("MiniCPM-o Resampler embed_dim must be divisible by four")

    grid_h = np.arange(height, dtype=np.float32)
    grid_w = np.arange(width, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)

    def embed(position: np.ndarray) -> np.ndarray:
        half = embed_dim // 4
        omega = np.arange(half, dtype=np.float32)
        omega /= float(half)
        omega = 1.0 / 10000**omega
        phase = np.einsum("hw,d->hwd", position, omega)
        return np.concatenate([np.sin(phase), np.cos(phase)], axis=-1)

    values = np.ascontiguousarray(
        np.concatenate([embed(grid[0]), embed(grid[1])], axis=-1),
        dtype=np.float32,
    )
    bits = values.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bf16 = ((bits + rounding) >> np.uint32(16)).astype("<u2", copy=False)
    header = _MINICPMO45_RESAMPLER_HEADER.pack(
        _MINICPMO45_RESAMPLER_MAGIC,
        height,
        width,
        embed_dim,
    )
    return RuntimeAsset(
        MINICPMO45_RESAMPLER_POS_EMBED_ASSET,
        "application/vnd.mfq.minicpmo45-resampler-pos-embed+bfloat16",
        header + bf16.tobytes(order="C"),
    )


def gguf_metadata_asset(reader: Any) -> RuntimeAsset:
    """Copy all GGUF key/value metadata into a tensor-free GGUF blob.

    The GGUF key/value byte stream is preserved exactly. Only ``tensor_count``
    is changed to zero and the tensor-info/data sections are omitted.
    """

    fields = [
        field
        for name, field in reader.fields.items()
        if not str(name).startswith("GGUF.")
    ]
    if not fields:
        raise ValueError("GGUF contains no metadata fields")
    kv_end = max(
        int(field.offset) + sum(int(part.nbytes) for part in field.parts)
        for field in fields
    )
    if kv_end <= 24:
        raise ValueError(f"invalid GGUF metadata boundary: {kv_end}")
    blob = bytearray(memoryview(reader.data)[:kv_end])
    endian = ">" if getattr(reader, "byte_order", "I") == "S" else "<"
    struct.pack_into(endian + "Q", blob, 8, 0)
    return RuntimeAsset(
        TOKENIZER_GGUF_ASSET,
        "application/vnd.gguf",
        bytes(blob),
    )


def runtime_asset_manifest(
    assets: list[RuntimeAsset] | tuple[RuntimeAsset, ...],
) -> dict[str, Any]:
    return {
        "version": 1,
        "assets": {
            asset.name.removeprefix(ASSET_PREFIX): asset.manifest_entry()
            for asset in assets
        },
    }


def discover_model_config(
    source: str | Path,
    explicit: str | Path | None = None,
) -> Path | None:
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"model config does not exist: {path}")
        return path
    source_path = Path(source).resolve()
    candidate = source_path / "config.json" if source_path.is_dir() else source_path.parent / "config.json"
    return candidate if candidate.is_file() else None
