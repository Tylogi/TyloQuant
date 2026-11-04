"""Shared image decode cache and the processor-facing contract."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import secrets
import struct
import tempfile
import threading
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic.dataclasses import dataclass

_IMAGE_DECODE_CACHE_MAX_BYTES = 512 * 1024 * 1024
_image_decode_cache: OrderedDict[str, Any] = OrderedDict()
_image_decode_cache_bytes = 0
_image_decode_cache_generation = 0
_image_decode_cache_lock = threading.Lock()


def _decoded_pixel_bytes(image: Any) -> int:
    width, height = image.size
    return width * height * 4 + height * struct.calcsize("P")


def clear_image_decode_cache() -> int:
    """Drop shared decoded-image references and return accounted bytes."""

    global _image_decode_cache_bytes, _image_decode_cache_generation
    with _image_decode_cache_lock:
        released = _image_decode_cache_bytes
        _image_decode_cache.clear()
        _image_decode_cache_bytes = 0
        _image_decode_cache_generation += 1
        return released


def _decode_image_cached(data: bytes) -> Any:
    from PIL import Image

    key = hashlib.sha256(data).hexdigest()
    with _image_decode_cache_lock:
        generation = _image_decode_cache_generation
        hit = _image_decode_cache.get(key)
        if hit is not None:
            _image_decode_cache.move_to_end(key)
            return hit

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
    except Exception as error:
        raise VisionProcessingError(f"unable to decode image: {error}") from error

    size = _decoded_pixel_bytes(rgb)
    if size > _IMAGE_DECODE_CACHE_MAX_BYTES:
        return rgb

    global _image_decode_cache_bytes
    with _image_decode_cache_lock:
        if generation != _image_decode_cache_generation:
            return rgb
        duplicate = _image_decode_cache.pop(key, None)
        if duplicate is not None:
            _image_decode_cache_bytes -= _decoded_pixel_bytes(duplicate)
        while (
            _image_decode_cache
            and _image_decode_cache_bytes + size > _IMAGE_DECODE_CACHE_MAX_BYTES
        ):
            _, evicted = _image_decode_cache.popitem(last=False)
            _image_decode_cache_bytes -= _decoded_pixel_bytes(evicted)
        _image_decode_cache[key] = rgb
        _image_decode_cache_bytes += size
    return rgb


class VisionProcessingError(ValueError):
    pass


def _decode_data_url(url: str, expected_prefix: str) -> bytes:
    if not url.startswith("data:") or ";base64," not in url:
        raise VisionProcessingError("native vision input requires a base64 data URL")
    header, encoded = url.split(",", 1)
    mime_type = header[5:].split(";", 1)[0].lower()
    if not mime_type.startswith(expected_prefix):
        raise VisionProcessingError(f"expected {expected_prefix} media, got {mime_type}")
    if not encoded:
        raise VisionProcessingError("media data URL is empty")
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise VisionProcessingError("media data URL contains invalid base64") from error
    if expected_prefix == "image/":
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as image:
                detected = Image.MIME.get(image.format)
        except Exception as error:
            raise VisionProcessingError(f"unable to decode image: {error}") from error
        if detected and mime_type != detected:
            raise VisionProcessingError(f"image MIME type {mime_type} does not match payload {detected}")
    return data


def validate_image_messages(messages: Any) -> None:
    """Reject malformed images before an HTTP streaming response commits."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                spec = part.get("image_url")
                if not isinstance(spec, dict) or not isinstance(spec.get("url"), str):
                    raise VisionProcessingError("image_url content is missing its URL")
                _decode_image(_decode_data_url(spec["url"], "image/"))


def _decode_image(data: bytes) -> Any:
    return _decode_image_cached(data)


def _packed_tensor(value: np.ndarray, dtype: str) -> np.ndarray:
    little_endian = {
        "float32": "<f4",
        "int32": "<i4",
        "int64": "<i8",
        "uint8": "u1",
    }[dtype]
    return np.ascontiguousarray(value, dtype=little_endian)


def _tensor(value: np.ndarray, dtype: str) -> dict[str, Any]:
    packed = _packed_tensor(value, dtype)
    return {
        "dtype": dtype,
        "shape": list(packed.shape),
        "data_base64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _binary_tensors(
    values: list[tuple[str, np.ndarray, str]],
) -> tuple[dict[str, Any], Path]:
    magic = b"MFQMM01\0"
    token = secrets.token_bytes(32)
    descriptor, raw_path = tempfile.mkstemp(
        prefix="mfq-multimodal-",
        suffix=".bin",
    )
    path = Path(raw_path)
    tensors: dict[str, Any] = {
        "version": 1,
        "binary_file": {
            "path": str(path),
            "token": token.hex(),
        },
    }
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        elif os.name != "nt":
            os.chmod(path, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(magic)
            stream.write(token)
            stream.write(bytes(64 - len(magic) - len(token)))
            for name, value, dtype in values:
                packed = _packed_tensor(value, dtype)
                offset = stream.tell()
                padding = (-offset) % 64
                if padding:
                    stream.write(bytes(padding))
                    offset += padding
                raw = memoryview(packed).cast("B")
                stream.write(raw)
                tensors[name] = {
                    "dtype": dtype,
                    "shape": list(packed.shape),
                    "data_offset": offset,
                    "data_length": packed.nbytes,
                }
            tensors["binary_file"]["size"] = stream.tell()
        return tensors, path
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ProcessedVisionRequest:
    messages: list[dict[str, Any]]
    tensors: dict[str, Any]
    source_count: int
    frame_count: int
    cleanup_paths: tuple[Path, ...] = ()


class MultimodalProcessor(Protocol):
    """Model-family media adapter consumed by :class:`OpenAIChatBackend`."""

    def prepare_openai_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        use_binary_file: bool = False,
    ) -> ProcessedVisionRequest | None: ...
