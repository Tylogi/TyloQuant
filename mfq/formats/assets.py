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
MODEL_GRAPH_ASSET = ASSET_PREFIX + "model_graph.json"
TOKENIZER_GGUF_ASSET = ASSET_PREFIX + "tokenizer.gguf"
HF_TOKENIZER_JSON_ASSET = ASSET_PREFIX + "hf/tokenizer.json"
HF_TOKENIZER_CONFIG_ASSET = ASSET_PREFIX + "hf/tokenizer_config.json"
HF_CHAT_TEMPLATE_ASSET = ASSET_PREFIX + "hf/chat_template.jinja"
HF_GENERATION_CONFIG_ASSET = ASSET_PREFIX + "hf/generation_config.json"
MINICPMO45_RESAMPLER_POS_EMBED_ASSET = ASSET_PREFIX + "minicpmo45-resampler-pos-embed-v1.bf16"
DEEPSEEK_V41_ENGRAM_ASSET = ASSET_PREFIX + "deepseek-v41-engram-v1.bin"
ASSET_DTYPE = "BLOB"
ASSET_MANIFEST_KEY = "runtime_assets"

_MINICPMO45_RESAMPLER_MAGIC = b"MFQRSPB1"
_MINICPMO45_RESAMPLER_HEADER = struct.Struct("<8sIII")
_DEEPSEEK_V41_ENGRAM_MAGIC = b"MFQENGR1"
_DEEPSEEK_V41_ENGRAM_HEADER = struct.Struct("<8sIIIII")


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
        data = json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    elif isinstance(config, str):
        data = config.encode("utf-8")
    else:
        data = bytes(config)
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("model config must be a JSON object")
    return RuntimeAsset(MODEL_CONFIG_ASSET, "application/json", data)


def model_graph_asset(graph: dict[str, Any] | bytes | str) -> RuntimeAsset:
    """Serialize the canonical, source-name-free runtime graph contract."""

    if isinstance(graph, dict):
        data = json.dumps(
            graph,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    elif isinstance(graph, str):
        data = graph.encode("utf-8")
    else:
        data = bytes(graph)
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("model graph must be a JSON object")
    if int(parsed.get("schema_version", 0) or 0) <= 0:
        raise ValueError("model graph must declare a positive schema_version")
    naming = parsed.get("canonical_naming")
    if not isinstance(naming, dict) or not naming.get("namespace"):
        raise ValueError("model graph must declare canonical_naming.namespace")
    return RuntimeAsset(MODEL_GRAPH_ASSET, "application/vnd.mfq.model-graph+json", data)


def hf_runtime_assets(source: str | Path) -> tuple[RuntimeAsset, ...]:
    """Embed dependency-light HF tokenizer and generation metadata.

    The native C++ runtime consumes the compact tokenizer-only GGUF. New
    architecture workers use the original Tokenizers JSON and chat template,
    so a converted model remains self-contained after leaving its HF source
    directory. Only metadata/tokenizer files are copied; executable remote
    model code is neither embedded nor trusted.
    """

    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        return ()
    specifications = (
        (
            "tokenizer.json",
            HF_TOKENIZER_JSON_ASSET,
            "application/vnd.huggingface.tokenizer+json",
            True,
        ),
        (
            "tokenizer_config.json",
            HF_TOKENIZER_CONFIG_ASSET,
            "application/json",
            True,
        ),
        (
            "chat_template.jinja",
            HF_CHAT_TEMPLATE_ASSET,
            "text/x-jinja2; charset=utf-8",
            False,
        ),
        (
            "generation_config.json",
            HF_GENERATION_CONFIG_ASSET,
            "application/json",
            False,
        ),
    )
    assets: list[RuntimeAsset] = []
    for filename, record, media_type, required in specifications:
        path = root / filename
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"HF runtime asset is missing: {path}")
            continue
        data = path.read_bytes()
        if not data:
            raise ValueError(f"HF runtime asset is empty: {path}")
        if filename.endswith(".json"):
            json.loads(data)
        else:
            data.decode("utf-8")
        assets.append(RuntimeAsset(record, media_type, data))
    return tuple(assets)


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


def deepseek_v41_engram_asset(
    tokenizer_path: str | Path,
    text_config: dict[str, Any],
) -> RuntimeAsset:
    """Build the tokenizer-dependent Engram hash contract once at conversion."""

    try:
        from tokenizers import Regex, Tokenizer, normalizers
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "DeepSeek-V4.1 conversion requires tokenizers to build its Engram asset"
        ) from error

    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path).resolve()))
    vocabulary = tokenizer.get_vocab_size(with_added_tokens=True)
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    token_map = np.empty(vocabulary, dtype="<i4")
    normalized_ids: dict[str, int] = {}
    for token_id in range(vocabulary):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in decoded:
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(decoded)
            key = normalized if normalized else decoded
        if key is None:
            raise ValueError(f"DeepSeek-V4.1 tokenizer has no token {token_id}")
        compressed = normalized_ids.setdefault(key, len(normalized_ids))
        token_map[token_id] = compressed

    expected_compressed = int(text_config["engram_compressed_vocab_size"])
    if len(normalized_ids) != expected_compressed:
        raise ValueError(
            "DeepSeek-V4.1 Engram compressed vocabulary differs: "
            f"{len(normalized_ids)} != {expected_compressed}"
        )
    layer_ids = np.asarray(text_config["engram_layer_ids"], dtype="<i4")
    table_rows = np.asarray(text_config["engram_num_embeddings"], dtype="<i8")
    max_ngram = int(text_config["engram_max_ngram_size"])
    heads = int(text_config["engram_n_heads"])
    base = int(text_config["engram_vocab_size"])
    if (
        layer_ids.ndim != 1
        or table_rows.shape != layer_ids.shape
        or max_ngram < 2
        or heads <= 0
        or base <= 0
    ):
        raise ValueError("DeepSeek-V4.1 Engram config geometry is invalid")

    def is_prime(value: int) -> bool:
        if value < 2 or value % 2 == 0:
            return value == 2
        divisor = 3
        while divisor * divisor <= value:
            if value % divisor == 0:
                return False
            divisor += 2
        return True

    seen: set[int] = set()
    primes = np.empty((len(layer_ids), max_ngram - 1, heads), dtype="<i8")
    for layer in range(len(layer_ids)):
        current = base - 1
        for order in range(max_ngram - 1):
            for head in range(heads):
                current += 1
                while not is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                primes[layer, order, head] = current
    flat_primes = primes.reshape(len(layer_ids), -1)
    offsets = np.zeros_like(flat_primes)
    if flat_primes.shape[1] > 1:
        offsets[:, 1:] = np.cumsum(flat_primes[:, :-1], axis=1)
    if not np.array_equal(offsets[:, -1] + flat_primes[:, -1], table_rows):
        raise ValueError("DeepSeek-V4.1 Engram table rows differ from hash buckets")

    multiplier_bound = max(1, (np.iinfo(np.int64).max // expected_compressed) // 2)
    multipliers = np.stack(
        [
            np.random.default_rng(10007 * int(layer)).integers(
                0,
                multiplier_bound,
                size=max_ngram,
                dtype=np.int64,
            )
            * 2
            + 1
            for layer in layer_ids
        ]
    ).astype("<i8", copy=False)
    pad_token_id = int(text_config["engram_pad_token_id"])
    if pad_token_id < 0 or pad_token_id >= vocabulary:
        raise ValueError("DeepSeek-V4.1 Engram pad token is outside the tokenizer")

    header = _DEEPSEEK_V41_ENGRAM_HEADER.pack(
        _DEEPSEEK_V41_ENGRAM_MAGIC,
        vocabulary,
        expected_compressed,
        len(layer_ids),
        max_ngram,
        heads,
    )
    data = b"".join(
        (
            header,
            struct.pack("<i", int(token_map[pad_token_id])),
            layer_ids.tobytes(order="C"),
            table_rows.tobytes(order="C"),
            primes.tobytes(order="C"),
            offsets.tobytes(order="C"),
            multipliers.tobytes(order="C"),
            token_map.tobytes(order="C"),
        )
    )
    return RuntimeAsset(
        DEEPSEEK_V41_ENGRAM_ASSET,
        "application/vnd.mfq.deepseek-v41-engram+binary",
        data,
    )


def gguf_metadata_asset(reader: Any) -> RuntimeAsset:
    """Copy all GGUF key/value metadata into a tensor-free GGUF blob.

    The GGUF key/value byte stream is preserved exactly. Only ``tensor_count``
    is changed to zero and the tensor-info/data sections are omitted.
    """

    fields = [field for name, field in reader.fields.items() if not str(name).startswith("GGUF.")]
    if not fields:
        raise ValueError("GGUF contains no metadata fields")
    kv_end = max(
        int(field.offset) + sum(int(part.nbytes) for part in field.parts) for field in fields
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
            asset.name.removeprefix(ASSET_PREFIX): asset.manifest_entry() for asset in assets
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
    candidate = (
        source_path / "config.json" if source_path.is_dir() else source_path.parent / "config.json"
    )
    return candidate if candidate.is_file() else None
