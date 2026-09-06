"""Private OpenAI-compatible MLX worker for Flash-Next MFQ models.

The desktop server normally delegates inference to the C++ sidecar.  Qwen3.8
Flash-Next and GLM-5.3 Flash have substantially different graphs from the
older Qwen/GLM families, while their verified implementations currently live
in MFQ's Python MLX runtime.  This worker makes those implementations usable by
the same process pool and HTTP adapter without claiming that an HF source
checkpoint can be executed before it has been converted to MFQ.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import math
import mmap
import os
import stat
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError("the Flash-Next worker requires MFQ's daemon extra") from exc

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError("the Flash-Next worker requires MFQ's Metal extra") from exc

from mfq.architectures.flash_next import (
    Glm5NextConfig,
    Qwen4ExpConfig,
    parse_flash_next_config,
)
from mfq.architectures.tensor_schema import GRID_VISION_INPUT_CONTRACT
from mfq.formats.assets import (
    HF_CHAT_TEMPLATE_ASSET,
    HF_GENERATION_CONFIG_ASSET,
    HF_TOKENIZER_CONFIG_ASSET,
    HF_TOKENIZER_JSON_ASSET,
    MODEL_CONFIG_ASSET,
)
from mfq.formats.io import MMapTensorStore, open_mmap
from mfq.kernels.metal.sampling import (
    sample,
    sample_apply_penalties,
    sample_token_counts_add,
)
from mfq.runtime.mlx_causal_lm import (
    MlxCausalLM,
    MlxCausalLMConfig,
    MlxQwen35Mtp,
)
from mfq.runtime.mlx_flash_next_vision import (
    MlxGlm5NextVision,
    MlxQwen4ExpVision,
    MlxQwenVision,
    inject_vision_embeddings,
    qwen4_multimodal_positions,
)
from mfq.runtime.mlx_glm5_next import MlxGlm5Next, MlxGlm5NextMtp
from mfq.runtime.mlx_qwen4_exp import MlxQwen4Exp, MlxQwen4ExpMtp

_GENERATION_STREAM = mx.new_thread_unsafe_stream(mx.default_device())

_SPECIAL_TOKEN_FIELDS = (
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
)


class FlashNextWorkerError(RuntimeError):
    """A request or self-contained model asset is invalid."""


@dataclass(frozen=True)
class _WorkerFamilyRegistration:
    family: str
    aliases: tuple[str, ...]
    config_parser: Callable[[Mapping[str, Any]], Any]
    config_type: type[Any]
    model_type: type[Any]
    vision_type: type[Any] | None = None
    mtp_type: type[Any] | None = None

    def parse_config(self, payload: Mapping[str, Any]) -> Any:
        config = self.config_parser(payload)
        if not isinstance(config, self.config_type):  # pragma: no cover - invariant
            raise TypeError(f"{self.family} registry returned a foreign config")
        return config

    def load(
        self,
        path: Path,
        config: Any,
        max_context: int,
    ) -> tuple[Any, Any | None, Any | None]:
        """Load a family through one architecture-neutral component path."""

        model = self.model_type.from_mfq(path, config, max_context=max_context)
        try:
            vision = (
                None
                if self.vision_type is None
                else self.vision_type.load_if_present(model, config)
            )
            mtp = (
                None
                if self.mtp_type is None
                else self.mtp_type.load_if_present(
                    model,
                    config,
                    max_context=max_context,
                )
            )
        except Exception:
            model.close()
            raise
        return model, vision, mtp

    def vision_supported(self, config: Any) -> bool:
        return self.vision_type is not None and getattr(config, "vision", None) is not None

    @property
    def mtp_supported(self) -> bool:
        return self.mtp_type is not None


def _normalized_model_type(payload: Mapping[str, Any]) -> str:
    text = payload.get("text_config")
    text_type = text.get("model_type") if isinstance(text, Mapping) else None
    return str(text_type or payload.get("model_type", "")).strip().lower().replace("-", "_")


def _parse_qwen35_config(payload: Mapping[str, Any]) -> MlxCausalLMConfig:
    return MlxCausalLMConfig.from_qwen35_hf_config(dict(payload))


_WORKER_FAMILY_REGISTRY = (
    _WorkerFamilyRegistration(
        family="qwen3_5",
        aliases=("qwen3_5", "qwen3_5_text", "qwen35"),
        config_parser=_parse_qwen35_config,
        config_type=MlxCausalLMConfig,
        model_type=MlxCausalLM,
        vision_type=MlxQwenVision,
        mtp_type=MlxQwen35Mtp,
    ),
    _WorkerFamilyRegistration(
        family="qwen4_exp",
        aliases=("qwen4_exp", "qwen4_exp_text"),
        config_parser=parse_flash_next_config,
        config_type=Qwen4ExpConfig,
        model_type=MlxQwen4Exp,
        vision_type=MlxQwen4ExpVision,
        mtp_type=MlxQwen4ExpMtp,
    ),
    _WorkerFamilyRegistration(
        family="glm5_next",
        aliases=("glm5_next", "glm5_next_text"),
        config_parser=parse_flash_next_config,
        config_type=Glm5NextConfig,
        model_type=MlxGlm5Next,
        vision_type=MlxGlm5NextVision,
        mtp_type=MlxGlm5NextMtp,
    ),
)

_WORKER_FAMILY_BY_ALIAS = {
    alias: registration
    for registration in _WORKER_FAMILY_REGISTRY
    for alias in registration.aliases
}
_WORKER_FAMILY_BY_NAME = {
    registration.family: registration for registration in _WORKER_FAMILY_REGISTRY
}


def _worker_family_for_payload(payload: Mapping[str, Any]) -> _WorkerFamilyRegistration:
    identity = _normalized_model_type(payload)
    try:
        return _WORKER_FAMILY_BY_ALIAS[identity]
    except KeyError as error:
        raise ValueError(f"unsupported MLX worker architecture: {identity!r}") from error


def _worker_family_for_config(config: Any) -> _WorkerFamilyRegistration:
    family = str(getattr(config, "family", ""))
    try:
        return _WORKER_FAMILY_BY_NAME[family]
    except KeyError as error:
        raise ValueError(f"unregistered MLX worker family: {family!r}") from error


def _parse_worker_config(payload: Mapping[str, Any]):
    registration = _worker_family_for_payload(payload)
    return registration.parse_config(payload)


def _is_qwen_worker_config(config: object) -> bool:
    return isinstance(config, (Qwen4ExpConfig, MlxCausalLMConfig))


def _request_integer(value: Any, name: str) -> int:
    """Parse a JSON integer without silently truncating floats or accepting booleans."""

    if isinstance(value, bool):
        raise FlashNextWorkerError(f"{name} must be an integer")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise FlashNextWorkerError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FlashNextWorkerError(f"{name} must be an integer") from error


def _request_float(value: Any, name: str) -> float:
    """Parse a JSON number while keeping conversion failures in the 400 path."""

    if isinstance(value, bool):
        raise FlashNextWorkerError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FlashNextWorkerError(f"{name} must be a number") from error


def _request_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise FlashNextWorkerError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class _PreparedRequest:
    request_id: str
    session_id: str | None
    input_ids: tuple[int, ...]
    prompt_ends_in_thinking: bool
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    seed: int | None
    sampling_payload: dict[str, Any]
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    enable_vision: bool = True
    enable_mtp: bool = True
    multimodal: _MultimodalInput | None = None
    positions: np.ndarray | None = None
    position_delta: int | None = None


@dataclass(frozen=True)
class _GenerationSummary:
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    content: str
    reasoning: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _PromptPrefill:
    logits: mx.array
    reused_tokens: int
    computed_tokens: int
    elapsed_ms: float
    multimodal_ms: float = 0.0
    llm_ms: float | None = None


@dataclass
class _MtpDecodeStats:
    cycles: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    target_ms: float = 0.0
    head_ms: float = 0.0
    rollback_ms: float = 0.0


@dataclass(frozen=True)
class _MultimodalInput:
    pixel_values: np.ndarray
    image_grid_thw: np.ndarray
    video_grid_thw: np.ndarray | None = None
    vision_grid_thw: np.ndarray | None = None
    vision_types: np.ndarray | None = None
    binary_owner: Any | None = None


_MULTIMODAL_DTYPES = {
    "float32": np.dtype("<f4"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "uint8": np.dtype("u1"),
}
_MULTIMODAL_MAGIC = b"MFQMM01\0"
_MULTIMODAL_HEADER_BYTES = 64
_MAX_MULTIMODAL_BYTES = 2 * 1024 * 1024 * 1024


def _sampling_log_probs(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> mx.array:
    """Return the exact distribution used by :func:`sample` for acceptance."""

    values = logits.astype(mx.float32)
    if values.ndim == 1:
        values = values[None]
    values = mx.where(mx.isnan(values), -mx.inf, values)
    if temperature <= 0.0 or int(top_k) == 1:
        return values - mx.logsumexp(values, axis=-1, keepdims=True)
    scores = values / float(temperature)
    vocab = int(scores.shape[-1])
    limit = vocab if int(top_k) <= 0 else min(int(top_k), vocab)
    if limit < vocab or float(top_p) < 1.0:
        order = mx.argsort(scores, axis=-1)[:, ::-1]
        ranked = mx.take_along_axis(scores, order[:, :limit], axis=-1)
        weights = mx.exp(ranked - ranked[:, :1])
        cutoff = weights.sum(axis=-1, keepdims=True) * float(top_p)
        keep_count = mx.sum(
            mx.cumsum(weights, axis=-1) - weights < cutoff,
            axis=-1,
            keepdims=True,
        )
        ranks = mx.argsort(order, axis=-1)
        scores = mx.where(ranks < keep_count, scores, -mx.inf)
    return scores - mx.logsumexp(scores, axis=-1, keepdims=True)


def _residual_sample(target_log_probs: mx.array, draft_log_probs: mx.array) -> mx.array:
    target = mx.exp(target_log_probs)
    residual = mx.maximum(target - mx.exp(draft_log_probs), 0.0)
    total = residual.sum(axis=-1, keepdims=True)
    distribution = mx.where(total > 0.0, residual, target)
    return mx.random.categorical(mx.log(distribution))


class _ReasoningParser:
    """Hide protocol tags and separate reasoning from visible answer text."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, starts_in_reasoning: bool) -> None:
        self.reasoning = bool(starts_in_reasoning)
        self.buffer = ""
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []

    @staticmethod
    def _protected_suffix(value: str, marker: str) -> int:
        return max(
            (
                width
                for width in range(1, min(len(value), len(marker) - 1) + 1)
                if value.endswith(marker[:width])
            ),
            default=0,
        )

    def _append(self, kind: str, text: str) -> tuple[tuple[str, str], ...]:
        if not text:
            return ()
        if kind == "reasoning":
            self.reasoning_parts.append(text)
        else:
            self.content_parts.append(text)
        return ((kind, text),)

    def feed(self, text: str) -> tuple[tuple[str, str], ...]:
        if not text:
            return ()
        self.buffer += text
        emitted: list[tuple[str, str]] = []
        while self.buffer:
            marker = self._CLOSE if self.reasoning else self._OPEN
            position = self.buffer.find(marker)
            if position >= 0:
                kind = "reasoning" if self.reasoning else "content"
                emitted.extend(self._append(kind, self.buffer[:position]))
                self.buffer = self.buffer[position + len(marker) :]
                self.reasoning = not self.reasoning
                if not self.reasoning:
                    self.buffer = self.buffer.lstrip("\r\n")
                continue
            protected = self._protected_suffix(self.buffer, marker)
            visible = self.buffer[:-protected] if protected else self.buffer
            self.buffer = self.buffer[-protected:] if protected else ""
            kind = "reasoning" if self.reasoning else "content"
            emitted.extend(self._append(kind, visible))
            break
        return tuple(emitted)

    def finish(self) -> tuple[tuple[str, str], ...]:
        kind = "reasoning" if self.reasoning else "content"
        value, self.buffer = self.buffer, ""
        return self._append(kind, value)

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    @property
    def reasoning_text(self) -> str:
        return "".join(self.reasoning_parts)


def _json_object(data: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FlashNextWorkerError(f"invalid embedded {name}: {error}") from error
    if not isinstance(value, dict):
        raise FlashNextWorkerError(f"embedded {name} must be an object")
    return value


def _multimodal_binary(payload: Mapping[str, Any]) -> mmap.mmap | None:
    specification = payload.get("binary_file")
    if specification is None:
        return None
    if not isinstance(specification, Mapping):
        raise FlashNextWorkerError("mfq_multimodal.binary_file must be an object")
    raw_path = specification.get("path")
    raw_token = specification.get("token")
    raw_size = specification.get("size")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise FlashNextWorkerError("multimodal binary path must be absolute")
    if not isinstance(raw_token, str) or len(raw_token) != 64:
        raise FlashNextWorkerError("multimodal binary token must contain 32 bytes")
    try:
        token = bytes.fromhex(raw_token)
    except ValueError as error:
        raise FlashNextWorkerError("multimodal binary token is not hexadecimal") from error
    if (
        not isinstance(raw_size, int)
        or not _MULTIMODAL_HEADER_BYTES <= raw_size <= _MAX_MULTIMODAL_BYTES
    ):
        raise FlashNextWorkerError("multimodal binary size is outside the safe range")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(raw_path, flags)
    except OSError as error:
        raise FlashNextWorkerError(f"unable to open multimodal binary file: {error}") from error
    mapping: mmap.mmap | None = None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != raw_size:
            raise FlashNextWorkerError("multimodal binary file identity is invalid")
        mapping = mmap.mmap(descriptor, raw_size, access=mmap.ACCESS_READ)
    except BaseException:
        if mapping is not None:
            mapping.close()
        raise
    finally:
        os.close(descriptor)
    if mapping[:8] != _MULTIMODAL_MAGIC or mapping[8:40] != token:
        mapping.close()
        raise FlashNextWorkerError("multimodal binary header authentication failed")
    return mapping


def _multimodal_tensor(
    payload: Mapping[str, Any],
    name: str,
    binary: mmap.mmap | None,
) -> np.ndarray:
    specification = payload.get(name)
    if not isinstance(specification, Mapping):
        raise FlashNextWorkerError(f"mfq_multimodal.{name} must be an object")
    dtype_name = specification.get("dtype")
    if not isinstance(dtype_name, str) or dtype_name not in _MULTIMODAL_DTYPES:
        raise FlashNextWorkerError(f"mfq_multimodal.{name} has an unsupported dtype")
    raw_shape = specification.get("shape")
    if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
        raise FlashNextWorkerError(f"mfq_multimodal.{name}.shape must be an array")
    shape: list[int] = []
    for dimension in raw_shape:
        if not isinstance(dimension, int) or dimension <= 0:
            raise FlashNextWorkerError(
                f"mfq_multimodal.{name}.shape dimensions must be positive integers"
            )
        shape.append(dimension)
    if not shape:
        raise FlashNextWorkerError(f"mfq_multimodal.{name}.shape cannot be empty")
    dtype = _MULTIMODAL_DTYPES[dtype_name]
    elements = math.prod(shape)
    expected_bytes = elements * dtype.itemsize
    if expected_bytes > _MAX_MULTIMODAL_BYTES:
        raise FlashNextWorkerError(f"mfq_multimodal.{name} exceeds the safe size limit")
    encoded = specification.get("data_base64")
    has_encoded = isinstance(encoded, str)
    has_range = "data_offset" in specification or "data_length" in specification
    if has_encoded == has_range:
        raise FlashNextWorkerError(f"mfq_multimodal.{name} must use exactly one tensor transport")
    if has_encoded:
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise FlashNextWorkerError(f"mfq_multimodal.{name}.data_base64 is invalid") from error
    else:
        if binary is None:
            raise FlashNextWorkerError(f"mfq_multimodal.{name} refers to a missing binary file")
        offset = specification.get("data_offset")
        length = specification.get("data_length")
        if not isinstance(offset, int) or not isinstance(length, int):
            raise FlashNextWorkerError(f"mfq_multimodal.{name} binary range must use integers")
        if (
            offset < _MULTIMODAL_HEADER_BYTES
            or length != expected_bytes
            or offset > len(binary)
            or length > len(binary) - offset
        ):
            raise FlashNextWorkerError(f"mfq_multimodal.{name} binary range is invalid")
        raw = memoryview(binary)[offset : offset + length]
    if len(raw) != expected_bytes:
        raise FlashNextWorkerError(f"mfq_multimodal.{name} byte length disagrees with its shape")
    return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape))


def _parse_multimodal_images(
    payload: Any,
    config: Qwen4ExpConfig | Glm5NextConfig | MlxCausalLMConfig,
) -> _MultimodalInput:
    if not isinstance(payload, Mapping):
        raise FlashNextWorkerError("mfq_multimodal must be an object")
    if (
        payload.get("version") != 3
        or payload.get("processor") != GRID_VISION_INPUT_CONTRACT
    ):
        raise FlashNextWorkerError(
            "mfq_multimodal does not match the grid-vision input contract"
        )
    if config.vision is None or config.image_token_id is None:
        raise FlashNextWorkerError("the loaded Flash-Next config has no vision contract")
    binary = _multimodal_binary(payload)
    pixel_values = _multimodal_tensor(payload, "pixel_values", binary)

    def optional_tensor(name: str) -> np.ndarray | None:
        if name not in payload:
            return None
        return _multimodal_tensor(payload, name, binary)

    image_grid_thw = optional_tensor("image_grid_thw")
    video_grid_thw = optional_tensor("video_grid_thw")
    vision_grid_thw = optional_tensor("vision_grid_thw")
    vision_types = optional_tensor("vision_types")
    expected_width = (
        config.vision.in_channels
        * config.vision.temporal_patch_size
        * config.vision.patch_size
        * config.vision.patch_size
    )
    if pixel_values.ndim != 2 or pixel_values.shape[1] != expected_width:
        raise FlashNextWorkerError("Flash-Next pixel_values have incompatible patch width")
    if not np.isfinite(pixel_values).all():
        raise FlashNextWorkerError("Flash-Next pixel_values contain a non-finite value")

    def checked_grid(value: np.ndarray | None, name: str) -> np.ndarray:
        if value is None:
            return np.empty((0, 3), dtype=np.int64)
        if value.ndim != 2 or value.shape[1] != 3:
            raise FlashNextWorkerError(f"Flash-Next {name} must have [items,3] shape")
        return np.asarray(value, dtype=np.int64)

    images = checked_grid(image_grid_thw, "image_grid_thw")
    videos = checked_grid(video_grid_thw, "video_grid_thw")
    if vision_grid_thw is None:
        if not len(images) or len(videos):
            raise FlashNextWorkerError(
                "legacy Flash-Next media payload must contain image_grid_thw only"
            )
        grid = images
        types = np.ones((len(grid),), dtype=np.int64)
    else:
        grid = checked_grid(vision_grid_thw, "vision_grid_thw")
        if vision_types is None or vision_types.ndim != 1 or len(vision_types) != len(grid):
            raise FlashNextWorkerError(
                "Flash-Next vision_types must have one entry per vision grid"
            )
        types = np.asarray(vision_types, dtype=np.int64)
        if np.any((types != 1) & (types != 2)):
            raise FlashNextWorkerError("Flash-Next vision_types must use image=1 or video=2")
        if not np.array_equal(grid[types == 1], images):
            raise FlashNextWorkerError(
                "Flash-Next image grids disagree with the combined media order"
            )
        if not np.array_equal(grid[types == 2], videos):
            raise FlashNextWorkerError(
                "Flash-Next video grids disagree with the combined media order"
            )
    merge = config.vision.spatial_merge_size
    if np.any(grid <= 0) or np.any(grid[:, 1:] % merge):
        raise FlashNextWorkerError("Flash-Next media grid is incompatible with merge geometry")
    if len(images) and np.any(images[:, 0] != 1):
        raise FlashNextWorkerError("Flash-Next image grids must have temporal size one")
    if sum(math.prod(int(value) for value in row) for row in grid) != pixel_values.shape[0]:
        raise FlashNextWorkerError("Flash-Next pixel count disagrees with media grids")
    if len(videos):
        if _is_qwen_worker_config(config):
            if config.video_token_id is None:
                raise FlashNextWorkerError("Qwen4-Exp config has no video token")
        elif config.video_start_token_id is None or config.video_end_token_id is None:
            raise FlashNextWorkerError("GLM-5-Next config has no video span tokens")
    return _MultimodalInput(
        pixel_values=np.ascontiguousarray(pixel_values, dtype=np.float32),
        image_grid_thw=np.ascontiguousarray(images, dtype=np.int32),
        video_grid_thw=np.ascontiguousarray(videos, dtype=np.int32),
        vision_grid_thw=np.ascontiguousarray(grid, dtype=np.int32),
        vision_types=np.ascontiguousarray(types, dtype=np.int32),
        binary_owner=binary,
    )


def _asset_or_sibling(
    store: MMapTensorStore,
    model_path: Path,
    record: str,
    filename: str,
    *,
    required: bool,
) -> bytes | None:
    if record in store.records:
        value = store[record]
        if not isinstance(value, bytes):
            raise FlashNextWorkerError(f"runtime asset {record!r} is not a BLOB")
        return value
    sibling = model_path.parent / filename
    if sibling.is_file():
        return sibling.read_bytes()
    if required:
        raise FlashNextWorkerError(
            f"converted Flash-Next model lacks {record!r}; reconvert it with a "
            "current MFQ build or place the original tokenizer files beside the MFQ shards"
        )
    return None


def load_flash_next_tokenizer(
    store: MMapTensorStore,
    model_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Construct a generic fast tokenizer without importing remote model code."""

    try:
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
    except ModuleNotFoundError as error:  # pragma: no cover - release dependency
        raise FlashNextWorkerError(
            "the Flash-Next worker requires tokenizers and transformers"
        ) from error

    path = Path(model_path).expanduser().resolve()
    raw_tokenizer = _asset_or_sibling(
        store,
        path,
        HF_TOKENIZER_JSON_ASSET,
        "tokenizer.json",
        required=True,
    )
    raw_config = _asset_or_sibling(
        store,
        path,
        HF_TOKENIZER_CONFIG_ASSET,
        "tokenizer_config.json",
        required=True,
    )
    assert raw_tokenizer is not None and raw_config is not None
    tokenizer_config = _json_object(raw_config, "tokenizer_config.json")
    try:
        backend = Tokenizer.from_str(raw_tokenizer.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise FlashNextWorkerError(f"invalid embedded tokenizer.json: {error}") from error
    special_tokens = {
        field: tokenizer_config[field]
        for field in _SPECIAL_TOKEN_FIELDS
        if tokenizer_config.get(field) is not None
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        **special_tokens,
    )
    raw_template = tokenizer_config.get("chat_template")
    if isinstance(raw_template, str) and raw_template:
        template = raw_template
    else:
        template_asset = _asset_or_sibling(
            store,
            path,
            HF_CHAT_TEMPLATE_ASSET,
            "chat_template.jinja",
            required=True,
        )
        assert template_asset is not None
        template = template_asset.decode("utf-8")
    tokenizer.chat_template = template
    generation_data = _asset_or_sibling(
        store,
        path,
        HF_GENERATION_CONFIG_ASSET,
        "generation_config.json",
        required=False,
    )
    generation_config = (
        {} if generation_data is None else _json_object(generation_data, "generation_config.json")
    )
    return tokenizer, generation_config


class FlashNextTextWorker:
    """Own one converted MLX model and serialize generation on it."""

    def __init__(
        self,
        model: MlxQwen4Exp | MlxGlm5Next | MlxCausalLM,
        tokenizer: Any,
        *,
        model_name: str,
        model_type: str,
        generation_config: Mapping[str, Any] | None = None,
        vision: MlxQwen4ExpVision | MlxQwenVision | MlxGlm5NextVision | None = None,
        mtp: MlxQwen35Mtp | MlxQwen4ExpMtp | MlxGlm5NextMtp | None = None,
        vision_supported: bool | None = None,
        mtp_supported: bool | None = None,
        prefill_chunk_size: int = 2_048,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_type = model_type
        self.generation_config = dict(generation_config or {})
        self.vision = vision
        self.mtp = mtp
        registration = next(
            (
                item
                for item in _WORKER_FAMILY_REGISTRY
                if item.family == model_type
            ),
            None,
        )
        self.vision_supported = bool(
            vision_supported
            if vision_supported is not None
            else registration is not None and vision is not None
        )
        self.mtp_supported = bool(
            mtp_supported
            if mtp_supported is not None
            else registration is not None and registration.mtp_supported
        )
        self.prefill_chunk_size = max(1, int(prefill_chunk_size))
        self._generation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._cached_input_ids: tuple[int, ...] = ()
        self._cached_logits: mx.array | None = None
        self._cache_sessions: set[str] = set()
        self.prefix_cache_queries = 0
        self.prefix_cache_hits = 0
        self.prefix_cache_hit_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.mtp_requests = 0
        self.mtp_cycles = 0
        self.mtp_drafted_tokens = 0
        self.mtp_accepted_tokens = 0
        self.last_request: dict[str, Any] | None = None

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        *,
        model_name: str,
        max_context: int,
        prefill_chunk_size: int = 2_048,
    ) -> FlashNextTextWorker:
        model_path = Path(path).expanduser().resolve()
        with open_mmap(model_path) as store:
            if MODEL_CONFIG_ASSET not in store.records:
                raise FlashNextWorkerError("MFQ has no embedded model_config.json")
            payload = store[MODEL_CONFIG_ASSET]
            if not isinstance(payload, bytes):
                raise FlashNextWorkerError("embedded model config is not a BLOB")
            config = _parse_worker_config(_json_object(payload, "model_config.json"))
            tokenizer, generation_config = load_flash_next_tokenizer(store, model_path)
        registration = _worker_family_for_config(config)
        selected_context = int(max_context)
        if selected_context <= 0:
            selected_context = min(32_768, config.max_position_embeddings)
        selected_context = min(selected_context, config.max_position_embeddings)
        with mx.stream(_GENERATION_STREAM):
            model, vision, mtp = registration.load(model_path, config, selected_context)
        return cls(
            model,
            tokenizer,
            model_name=model_name,
            model_type=config.family,
            generation_config=generation_config,
            vision=vision,
            mtp=mtp,
            vision_supported=registration.vision_supported(config),
            mtp_supported=registration.mtp_supported,
            prefill_chunk_size=prefill_chunk_size,
        )

    def close(self) -> None:
        self.model.close()

    def _prefill_prompt(self, prepared: _PreparedRequest) -> _PromptPrefill:
        """Reuse the one physical model cache only for an exact live-session prefix."""

        prompt = prepared.input_ids
        with self._state_lock:
            self.prefix_cache_queries += 1
            cached = self._cached_input_ids
            cached_logits = self._cached_logits
            session_matches = (
                prepared.session_id is not None and prepared.session_id in self._cache_sessions
            )
        model_position = getattr(self.model, "position", len(cached))
        position_matches = int(model_position) == len(cached)
        can_reuse = (
            prepared.multimodal is None
            and session_matches
            and bool(cached)
            and cached_logits is not None
            and position_matches
            and len(cached) <= len(prompt)
            and prompt[: len(cached)] == cached
        )
        started = time.perf_counter()
        if prepared.multimodal is not None:
            if self.vision is None:
                raise FlashNextWorkerError(
                    "the loaded MFQ artifact does not contain a Flash-Next vision tower"
                )
            with self._state_lock:
                self._cached_input_ids = ()
                self._cached_logits = None
                self._cache_sessions.clear()
            self.model.reset_cache(1)
            vision_started = time.perf_counter()
            vision_grid = (
                prepared.multimodal.vision_grid_thw
                if prepared.multimodal.vision_grid_thw is not None
                else prepared.multimodal.image_grid_thw
            )
            _vision_hidden, vision_embeddings = self.vision(
                prepared.multimodal.pixel_values,
                vision_grid,
            )
            mx.eval(vision_embeddings)
            multimodal_ms = (time.perf_counter() - vision_started) * 1000.0
            input_ids = np.asarray(prompt, dtype=np.int32)[None]
            token_embeddings = self.model.embedding(mx.array(input_ids))
            image_token_id = self.model.config.image_token_id
            assert image_token_id is not None
            vision_token_ids: int | tuple[int, ...] = image_token_id
            video_grids = prepared.multimodal.video_grid_thw
            if (
                _is_qwen_worker_config(self.model.config)
                and video_grids is not None
                and len(video_grids)
            ):
                video_token_id = self.model.config.video_token_id
                assert video_token_id is not None
                vision_token_ids = (image_token_id, video_token_id)
            embeddings = inject_vision_embeddings(
                token_embeddings,
                input_ids,
                vision_embeddings,
                vision_token_ids,
            )
            llm_started = time.perf_counter()
            logits: mx.array | None = None
            for start in range(0, len(prompt), self.prefill_chunk_size):
                end = min(start + self.prefill_chunk_size, len(prompt))
                chunk_embeddings = embeddings[:, start:end]
                if _is_qwen_worker_config(self.model.config):
                    chunk_positions = (
                        None
                        if prepared.positions is None
                        else prepared.positions[..., start:end]
                    )
                    logits = self.model.forward_embeddings(
                        chunk_embeddings,
                        input_ids[:, start:end],
                        chunk_positions,
                        use_cache=True,
                    )
                else:
                    logits = self.model.forward_embeddings(
                        chunk_embeddings,
                        use_cache=True,
                    )
                mx.eval(logits)
            assert logits is not None
            llm_ms = (time.perf_counter() - llm_started) * 1000.0
            return _PromptPrefill(
                logits=logits,
                reused_tokens=0,
                computed_tokens=len(prompt),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                multimodal_ms=multimodal_ms,
                llm_ms=llm_ms,
            )
        if can_reuse:
            reused = len(cached)
            suffix = prompt[reused:]
            logits = cached_logits
            if suffix:
                suffix_ids = np.asarray(suffix, dtype=np.int32)[None]
                logits = self.model.forward(suffix_ids, use_cache=True)
            mx.eval(logits)
            with self._state_lock:
                self._cached_input_ids = prompt
                self._cached_logits = logits
                self.prefix_cache_hits += 1
                self.prefix_cache_hit_tokens += reused
            return _PromptPrefill(
                logits=logits,
                reused_tokens=reused,
                computed_tokens=len(suffix),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                llm_ms=(time.perf_counter() - started) * 1000.0,
            )

        input_ids = np.asarray(prompt, dtype=np.int32)[None]
        logits = self.model.prefill(input_ids)
        mx.eval(logits)
        with self._state_lock:
            self._cached_input_ids = prompt
            self._cached_logits = logits
            self._cache_sessions = (
                {prepared.session_id} if prepared.session_id is not None else set()
            )
        return _PromptPrefill(
            logits=logits,
            reused_tokens=0,
            computed_tokens=len(prompt),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            llm_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _remember_decoded_token(
        self,
        prepared: _PreparedRequest,
        token: int,
        logits: mx.array,
    ) -> None:
        with self._state_lock:
            if prepared.session_id is None or prepared.session_id not in self._cache_sessions:
                return
            self._cached_input_ids += (int(token),)
            self._cached_logits = logits

    def _decode_token(self, prepared: _PreparedRequest, token: int) -> mx.array:
        input_ids = np.asarray([[token]], dtype=np.int32)
        if prepared.position_delta is None:
            logits = self.model.decode(input_ids)
        else:
            position = int(self.model.position) + prepared.position_delta
            positions = np.full((3, 1, 1), position, dtype=np.int32)
            logits = self.model.forward(input_ids, positions, use_cache=True)
        return logits

    @staticmethod
    def _initial_penalty_counts(
        prepared: _PreparedRequest,
        vocab_size: int,
    ) -> mx.array | None:
        if (
            prepared.presence_penalty == 0.0
            and prepared.frequency_penalty == 0.0
            and prepared.repetition_penalty == 1.0
        ):
            return None
        counts = mx.zeros((int(vocab_size),), dtype=mx.int32)
        return sample_token_counts_add(counts, np.asarray(prepared.input_ids, dtype=np.int32))

    @staticmethod
    def _penalized_logits(
        prepared: _PreparedRequest,
        logits: mx.array,
        counts: mx.array | None,
    ) -> mx.array:
        if counts is None:
            return logits
        return sample_apply_penalties(
            logits,
            counts,
            presence_penalty=prepared.presence_penalty,
            frequency_penalty=prepared.frequency_penalty,
            repetition_penalty=prepared.repetition_penalty,
        )

    @staticmethod
    def _count_token(counts: mx.array | None, token: int) -> mx.array | None:
        if counts is None:
            return None
        return sample_token_counts_add(counts, np.asarray([int(token)], dtype=np.int32))

    def _forward_tokens_with_hidden(
        self,
        prepared: _PreparedRequest,
        tokens: Sequence[int],
        *,
        n_confirmed: int = 0,
    ) -> tuple[mx.array, mx.array]:
        input_ids = np.asarray([tuple(int(token) for token in tokens)], dtype=np.int32)
        if _is_qwen_worker_config(self.model.config):
            positions: np.ndarray | None = None
            if prepared.position_delta is not None:
                start = int(self.model.position) + prepared.position_delta
                positions = np.broadcast_to(
                    np.arange(start, start + len(tokens), dtype=np.int32)[None, None],
                    (3, 1, len(tokens)),
                ).copy()
            return self.model.forward_with_hidden(
                input_ids,
                positions,
                use_cache=True,
                n_confirmed=n_confirmed,
            )
        return self.model.forward_with_hidden(
            input_ids,
            use_cache=True,
            n_confirmed=n_confirmed,
        )

    def _mtp_draft(
        self,
        prepared: _PreparedRequest,
        previous_hidden: mx.array,
        next_main_token: int,
        counts: mx.array | None,
    ) -> tuple[int, mx.array]:
        if self.mtp is None:  # pragma: no cover - caller contract
            raise RuntimeError("Flash-Next MTP head is not loaded")
        input_ids = np.asarray([[int(next_main_token)]], dtype=np.int32)
        mtp_hidden = self.mtp.forward(
            input_ids,
            previous_hidden,
            use_cache=True,
        )
        if isinstance(mtp_hidden, tuple):
            mtp_hidden = mtp_hidden[0]
        logits = self._penalized_logits(
            prepared,
            self.mtp.compute_logits(mtp_hidden)[:, -1],
            counts,
        )
        log_probs = _sampling_log_probs(
            logits,
            temperature=prepared.temperature,
            top_k=prepared.top_k,
            top_p=prepared.top_p,
        )
        token_array = sample(
            logits,
            temperature=prepared.temperature,
            top_k=prepared.top_k,
            top_p=prepared.top_p,
        )
        mx.eval(token_array, log_probs)
        return int(np.asarray(token_array).reshape(-1)[0]), log_probs

    def _generate_with_mtp(
        self,
        prepared: _PreparedRequest,
        initial_logits: mx.array,
        cancellation: threading.Event,
        eos: set[int],
        emit_token: Callable[[int], None],
    ) -> tuple[list[int], str, float | None, _MtpDecodeStats]:
        """Run exact depth-one draft/verify with recurrent-state rollback."""

        assert self.mtp is not None
        generated: list[int] = []
        finish_reason = "stop"
        first_token_at: float | None = None
        stats = _MtpDecodeStats()
        greedy = prepared.temperature <= 0.0 or prepared.top_k == 1
        counts = self._initial_penalty_counts(
            prepared,
            int(initial_logits.shape[-1]),
        )

        with self._state_lock:
            self._cached_input_ids = ()
            self._cached_logits = None
            self._cache_sessions.clear()
        self.mtp.reset_cache(1)

        def draw_adjusted(logits: mx.array) -> int:
            value = sample(
                logits,
                temperature=prepared.temperature,
                top_k=prepared.top_k,
                top_p=prepared.top_p,
            )
            mx.eval(value)
            return int(np.asarray(value).reshape(-1)[0])

        def draw(logits: mx.array) -> int:
            return draw_adjusted(self._penalized_logits(prepared, logits, counts))

        def append(token: int) -> bool:
            nonlocal counts, finish_reason
            if token in eos:
                finish_reason = "stop"
                return False
            generated.append(int(token))
            counts = self._count_token(counts, token)
            emit_token(int(token))
            if len(generated) >= prepared.max_tokens:
                finish_reason = "length"
                return False
            return True

        main_token = draw(initial_logits[:, -1])
        first_token_at = time.perf_counter()
        if not append(main_token) or cancellation.is_set():
            return generated, finish_reason, first_token_at, stats

        target_started = time.perf_counter()
        next_logits, previous_hidden = self._forward_tokens_with_hidden(
            prepared,
            (main_token,),
        )
        mx.eval(next_logits, previous_hidden)
        stats.target_ms += (time.perf_counter() - target_started) * 1000.0
        next_main = draw(next_logits[:, -1])
        if not append(next_main) or cancellation.is_set():
            return generated, finish_reason, first_token_at, stats

        while not cancellation.is_set() and len(generated) < prepared.max_tokens:
            head_started = time.perf_counter()
            draft, draft_log_probs = self._mtp_draft(
                prepared,
                previous_hidden[:, -1:],
                next_main,
                counts,
            )
            stats.head_ms += (time.perf_counter() - head_started) * 1000.0
            stats.cycles += 1
            stats.drafted_tokens += 1

            target_started = time.perf_counter()
            verify_logits, verify_hidden = self._forward_tokens_with_hidden(
                prepared,
                (next_main, draft),
                n_confirmed=1,
            )
            mx.eval(verify_logits, verify_hidden)
            stats.target_ms += (time.perf_counter() - target_started) * 1000.0

            target_logits = self._penalized_logits(
                prepared,
                verify_logits[:, 0],
                counts,
            )
            target_log_probs = _sampling_log_probs(
                target_logits,
                temperature=prepared.temperature,
                top_k=prepared.top_k,
                top_p=prepared.top_p,
            )
            if greedy:
                correction = draw_adjusted(target_logits)
                accepted = correction == draft
            else:
                mx.eval(target_log_probs)
                log_ratio = float(
                    np.asarray(target_log_probs[0, draft] - draft_log_probs[0, draft])
                )
                accepted = log_ratio >= 0.0 or float(
                    np.asarray(mx.random.uniform(shape=()))
                ) < math.exp(log_ratio)
                correction = draft
                if not accepted:
                    correction_array = _residual_sample(
                        target_log_probs,
                        draft_log_probs,
                    )
                    mx.eval(correction_array)
                    correction = int(np.asarray(correction_array).reshape(-1)[0])

            if accepted:
                self.model.commit_speculative_cache()
                stats.accepted_tokens += 1
                if not append(draft) or cancellation.is_set():
                    break
                bonus = draw(verify_logits[:, 1])
                if not append(bonus) or cancellation.is_set():
                    break
                previous_hidden = verify_hidden[:, 1:2]
                next_main = bonus
                continue

            rollback_started = time.perf_counter()
            self.model.rollback_speculative_cache()
            stats.rollback_ms += (time.perf_counter() - rollback_started) * 1000.0
            if not append(correction) or cancellation.is_set():
                break
            previous_hidden = verify_hidden[:, 0:1]
            next_main = correction

        if cancellation.is_set():
            finish_reason = "stop"
        return generated, finish_reason, first_token_at, stats

    def fork_session(self, source_session_id: str, target_session_id: str) -> int:
        if not source_session_id or not target_session_id:
            raise FlashNextWorkerError("source and target session IDs must be non-empty")
        if source_session_id == target_session_id:
            raise FlashNextWorkerError("source and target sessions must differ")
        with self._state_lock:
            if source_session_id not in self._cache_sessions or not self._cached_input_ids:
                return 0
            self._cache_sessions.add(target_session_id)
            return 1

    def close_session(self, session_id: str) -> tuple[int, bool]:
        cancelled = self.cancel(session_id)
        with self._state_lock:
            released = int(session_id in self._cache_sessions)
            self._cache_sessions.discard(session_id)
        return released, cancelled

    def clear_cache(self) -> bool:
        if not self._generation_lock.acquire(blocking=False):
            return False
        try:
            with mx.stream(_GENERATION_STREAM):
                self.model.reset_cache()
                if self.mtp is not None:
                    self.mtp.reset_cache()
                mx.clear_cache()
            with self._state_lock:
                self._cached_input_ids = ()
                self._cached_logits = None
                self._cache_sessions.clear()
            return True
        finally:
            self._generation_lock.release()

    def prepare(self, request: Mapping[str, Any]) -> _PreparedRequest:
        requested_model = request.get("model")
        if requested_model != self.model_name:
            raise FlashNextWorkerError(f"model not found: {requested_model!r}")
        messages = request.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
            raise FlashNextWorkerError("messages must be a non-empty array")
        normalized: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise FlashNextWorkerError(f"messages[{index}] must be an object")
            role = message.get("role")
            content = message.get("content", "")
            if not isinstance(role, str) or role not in {
                "system",
                "developer",
                "user",
                "assistant",
                "tool",
            }:
                raise FlashNextWorkerError(f"messages[{index}] has an unsupported role")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise FlashNextWorkerError(
                    "Flash-Next text worker does not yet accept multimodal message parts"
                )
            item = dict(message)
            item["role"] = role
            item["content"] = content
            normalized.append(item)
        raw_template = request.get("chat_template_kwargs")
        template_options = dict(raw_template) if isinstance(raw_template, Mapping) else {}
        if request.get("enable_thinking") is not None:
            template_options["enable_thinking"] = bool(request["enable_thinking"])
        if request.get("reasoning_effort") is not None:
            template_options["reasoning_effort"] = str(request["reasoning_effort"])
        tools = request.get("tools")
        if tools is not None:
            if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
                raise FlashNextWorkerError("tools must be an array")
            template_options["tools"] = list(tools)
        try:
            rendered = self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
                **template_options,
            )
        except Exception as error:
            raise FlashNextWorkerError(f"chat template failed: {error}") from error
        if not isinstance(rendered, str) or not rendered:
            raise FlashNextWorkerError("chat template produced an empty prompt")
        encoded = self.tokenizer.encode(rendered, add_special_tokens=False)
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        input_ids = tuple(int(value) for value in encoded)
        if not input_ids:
            raise FlashNextWorkerError("tokenizer produced an empty prompt")
        multimodal: _MultimodalInput | None = None
        positions: np.ndarray | None = None
        position_delta: int | None = None
        enable_vision = _request_boolean(
            request.get("enable_vision", True),
            "enable_vision",
        )
        enable_mtp = _request_boolean(
            request.get("enable_mtp", True),
            "enable_mtp",
        )
        if request.get("mfq_multimodal") is not None:
            if not enable_vision:
                raise FlashNextWorkerError(
                    "vision is disabled for this request; set enable_vision=true"
                )
            if self.vision is None:
                raise FlashNextWorkerError(
                    "the loaded MFQ artifact does not contain a Flash-Next vision tower"
                )
            multimodal = _parse_multimodal_images(
                request["mfq_multimodal"],
                self.model.config,
            )
            merge = self.model.config.vision.spatial_merge_size
            image_tokens = sum(
                math.prod(int(value) for value in row) // (merge * merge)
                for row in multimodal.image_grid_thw
            )
            video_grids = multimodal.video_grid_thw
            video_tokens = sum(
                math.prod(int(value) for value in row) // (merge * merge)
                for row in (() if video_grids is None else video_grids)
            )
            image_token_id = self.model.config.image_token_id
            assert image_token_id is not None
            if _is_qwen_worker_config(self.model.config):
                ids = np.asarray(input_ids, dtype=np.int32)[None]
                video_token_id = self.model.config.video_token_id
                actual_images = int(np.count_nonzero(ids == image_token_id))
                actual_videos = (
                    0
                    if video_token_id is None
                    else int(np.count_nonzero(ids == video_token_id))
                )
                if actual_images != image_tokens or actual_videos != video_tokens:
                    raise FlashNextWorkerError(
                        "rendered Qwen image/video placeholders disagree with vision embeddings"
                    )
                modality_types = np.zeros_like(ids, dtype=np.int32)
                modality_types[ids == image_token_id] = 1
                if video_token_id is not None:
                    modality_types[ids == video_token_id] = 2
                positions, deltas = qwen4_multimodal_positions(
                    ids,
                    modality_types,
                    spatial_merge_size=merge,
                    image_grid_thw=(
                        multimodal.image_grid_thw if len(multimodal.image_grid_thw) else None
                    ),
                    video_grid_thw=(
                        video_grids if video_grids is not None and len(video_grids) else None
                    ),
                )
                position_delta = int(deltas[0, 0])
            else:
                ids = np.asarray(input_ids, dtype=np.int32)
                video_start = self.model.config.video_start_token_id
                video_end = self.model.config.video_end_token_id
                if video_start is None or video_end is None:
                    inside_video = np.zeros(ids.shape, dtype=bool)
                else:
                    inside_video = np.cumsum(ids == video_start) > np.cumsum(ids == video_end)
                placeholders = ids == image_token_id
                actual_images = int(np.count_nonzero(placeholders & ~inside_video))
                actual_videos = int(np.count_nonzero(placeholders & inside_video))
                if actual_images != image_tokens or actual_videos != video_tokens:
                    raise FlashNextWorkerError(
                        "rendered GLM image/video placeholders disagree with vision embeddings"
                    )
        raw_max_tokens = (
            request["max_tokens"]
            if "max_tokens" in request
            else self.generation_config.get("max_new_tokens", 256) or 256
        )
        max_tokens = _request_integer(raw_max_tokens, "max_tokens")
        if max_tokens <= 0:
            raise FlashNextWorkerError("max_tokens must be positive")
        remaining = self.model.max_context - len(input_ids)
        if remaining <= 0:
            raise FlashNextWorkerError(
                f"prompt has {len(input_ids)} tokens and exceeds context {self.model.max_context}"
            )
        max_tokens = min(max_tokens, remaining)
        temperature = _request_float(request.get("temperature", 1.0), "temperature")
        top_k = _request_integer(request.get("top_k", 0), "top_k")
        top_p = _request_float(request.get("top_p", 1.0), "top_p")
        if not math.isfinite(temperature) or temperature < 0.0:
            raise FlashNextWorkerError("temperature must be finite and non-negative")
        if not 0 <= top_k <= 1024:
            raise FlashNextWorkerError("top_k must be in [0,1024]")
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise FlashNextWorkerError("top_p must be in (0,1]")
        presence_penalty = _request_float(
            request.get("presence_penalty", 0.0),
            "presence_penalty",
        )
        frequency_penalty = _request_float(
            request.get("frequency_penalty", 0.0),
            "frequency_penalty",
        )
        repetition_penalty = _request_float(
            request.get("repetition_penalty", 1.0),
            "repetition_penalty",
        )
        if not math.isfinite(presence_penalty) or not -2.0 <= presence_penalty <= 2.0:
            raise FlashNextWorkerError("presence_penalty must be finite and in [-2,2]")
        if not math.isfinite(frequency_penalty) or not -2.0 <= frequency_penalty <= 2.0:
            raise FlashNextWorkerError("frequency_penalty must be finite and in [-2,2]")
        if not math.isfinite(repetition_penalty) or repetition_penalty <= 0.0:
            raise FlashNextWorkerError("repetition_penalty must be finite and positive")
        raw_seed = request.get("seed")
        seed = None if raw_seed is None else _request_integer(raw_seed, "seed")
        if seed is not None and seed < 0:
            raise FlashNextWorkerError("seed must be non-negative")
        session_id = request.get("mfq_session_id")
        if session_id is not None:
            session_id = str(session_id)
        return _PreparedRequest(
            request_id=f"chatcmpl-{uuid.uuid4().hex}",
            session_id=session_id,
            input_ids=input_ids,
            prompt_ends_in_thinking=rendered.rstrip().endswith("<think>"),
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            enable_vision=enable_vision,
            enable_mtp=enable_mtp,
            sampling_payload={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "repetition_penalty": repetition_penalty,
                "seed": seed,
                "enable_thinking": bool(template_options.get("enable_thinking", True)),
                "enable_vision": enable_vision,
                "enable_mtp": enable_mtp,
                "reasoning_effort": template_options.get("reasoning_effort"),
            },
            multimodal=multimodal,
            positions=positions,
            position_delta=position_delta,
        )

    def cancel(self, session_id: str) -> bool:
        with self._state_lock:
            event = self._cancel_events.get(session_id)
        if event is None:
            return False
        event.set()
        return True

    def _register_cancel(self, prepared: _PreparedRequest) -> threading.Event:
        event = threading.Event()
        if prepared.session_id is not None:
            with self._state_lock:
                if prepared.session_id in self._cancel_events:
                    raise FlashNextWorkerError("the session already has an active generation")
                self._cancel_events[prepared.session_id] = event
        return event

    def _unregister_cancel(self, prepared: _PreparedRequest) -> None:
        if prepared.session_id is None:
            return
        with self._state_lock:
            self._cancel_events.pop(prepared.session_id, None)

    def generate(
        self,
        prepared: _PreparedRequest,
        emit: Callable[[dict[str, Any]], None],
    ) -> _GenerationSummary:
        cancellation = self._register_cancel(prepared)
        queued_at = time.perf_counter()
        started = queued_at
        queue_ms = 0.0
        parser = _ReasoningParser(prepared.prompt_ends_in_thinking)
        generated: list[int] = []
        prefill_ms = 0.0
        multimodal_ms = 0.0
        model_prefill_ms = 0.0
        reused_prompt_tokens = 0
        computed_prompt_tokens = len(prepared.input_ids)
        first_token_at: float | None = None
        finish_reason = "stop"
        mtp_stats: _MtpDecodeStats | None = None
        try:
            with self._generation_lock, mx.stream(_GENERATION_STREAM):
                started = time.perf_counter()
                queue_ms = max(0.0, (started - queued_at) * 1000.0)
                if cancellation.is_set():
                    finish_reason = "stop"
                else:
                    if prepared.seed is not None:
                        mx.random.seed(prepared.seed)
                    prefill = self._prefill_prompt(prepared)
                    logits = prefill.logits
                    model_prefill_ms = prefill.elapsed_ms
                    multimodal_ms = prefill.multimodal_ms
                    prefill_ms = prefill.elapsed_ms if prefill.llm_ms is None else prefill.llm_ms
                    reused_prompt_tokens = prefill.reused_tokens
                    computed_prompt_tokens = prefill.computed_tokens
                    from tokenizers.decoders import DecodeStream

                    decode_stream = DecodeStream(skip_special_tokens=False)
                    eos = set(int(value) for value in self.model.config.eos_token_ids)
                    tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
                    if tokenizer_eos is not None:
                        eos.add(int(tokenizer_eos))

                    def emit_token(token: int) -> None:
                        piece = decode_stream.step(
                            self.tokenizer.backend_tokenizer,
                            token,
                        )
                        if piece:
                            for kind, text in parser.feed(piece):
                                emit({kind: text})

                    if (
                        self.mtp is not None
                        and prepared.enable_mtp
                        and prepared.max_tokens >= 3
                    ):
                        generated, finish_reason, first_token_at, mtp_stats = (
                            self._generate_with_mtp(
                                prepared,
                                logits,
                                cancellation,
                                eos,
                                emit_token,
                            )
                        )
                    else:
                        counts = self._initial_penalty_counts(
                            prepared,
                            int(logits.shape[-1]),
                        )
                        for step in range(prepared.max_tokens):
                            if cancellation.is_set():
                                finish_reason = "stop"
                                break
                            sampling_logits = self._penalized_logits(
                                prepared,
                                logits[:, -1],
                                counts,
                            )
                            next_id = sample(
                                sampling_logits,
                                temperature=prepared.temperature,
                                top_k=prepared.top_k,
                                top_p=prepared.top_p,
                            )
                            mx.eval(next_id)
                            token = int(np.asarray(next_id).reshape(-1)[0])
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            if token in eos:
                                finish_reason = "stop"
                                break
                            generated.append(token)
                            counts = self._count_token(counts, token)
                            emit_token(token)
                            if step + 1 >= prepared.max_tokens:
                                finish_reason = "length"
                                break
                            logits = self._decode_token(prepared, token)
                            self._remember_decoded_token(prepared, token, logits)
                    for kind, text in parser.finish():
                        emit({kind: text})
            ended = time.perf_counter()
            total_ms = max(0.0, (ended - started) * 1000.0)
            ttft_ms = (
                prefill_ms
                if first_token_at is None
                else max(0.0, (first_token_at - started) * 1000.0)
            )
            decode_ms = max(0.0, total_ms - model_prefill_ms)
            completion = len(generated)
            metrics = {
                "prefill_tokens": len(prepared.input_ids),
                "ttft_ms": ttft_ms,
                "prefill_ms": prefill_ms,
                "prefill_tps": (
                    1000.0 * len(prepared.input_ids) / prefill_ms if prefill_ms > 0.0 else 0.0
                ),
                "multimodal_ms": multimodal_ms,
                "model_prefill_ms": model_prefill_ms,
                "processor_ms": 0.0,
                "queue_ms": queue_ms,
                "complete_prefill_ms": ttft_ms,
                "complete_prefill_tps": (
                    1000.0 * len(prepared.input_ids) / ttft_ms if ttft_ms > 0.0 else 0.0
                ),
                "decode_ms": decode_ms,
                "decode_tps": 1000.0 * completion / decode_ms if decode_ms > 0.0 else 0.0,
                "generation_ms": total_ms,
                "complete_generation_ms": total_ms,
                "generation_tps": 1000.0 * completion / total_ms if total_ms > 0.0 else 0.0,
                "mtp_available": self.mtp is not None,
                "mtp_used": mtp_stats is not None,
                "mtp_cycles": 0 if mtp_stats is None else mtp_stats.cycles,
                "mtp_drafted_tokens": (
                    0 if mtp_stats is None else mtp_stats.drafted_tokens
                ),
                "mtp_accepted_tokens": (
                    0 if mtp_stats is None else mtp_stats.accepted_tokens
                ),
                "mtp_acceptance_rate": (
                    0.0
                    if mtp_stats is None or not mtp_stats.drafted_tokens
                    else mtp_stats.accepted_tokens / mtp_stats.drafted_tokens
                ),
                "mtp_target_ms": 0.0 if mtp_stats is None else mtp_stats.target_ms,
                "mtp_head_ms": 0.0 if mtp_stats is None else mtp_stats.head_ms,
                "mtp_rollback_ms": 0.0 if mtp_stats is None else mtp_stats.rollback_ms,
                "sampling": prepared.sampling_payload,
            }
            summary = _GenerationSummary(
                request_id=prepared.request_id,
                prompt_tokens=len(prepared.input_ids),
                completion_tokens=completion,
                finish_reason=finish_reason,
                content=parser.content,
                reasoning=parser.reasoning_text,
                metrics=metrics,
            )
            with self._state_lock:
                self.total_requests += 1
                self.total_prompt_tokens += summary.prompt_tokens
                self.total_completion_tokens += summary.completion_tokens
                if mtp_stats is not None:
                    self.mtp_requests += 1
                    self.mtp_cycles += mtp_stats.cycles
                    self.mtp_drafted_tokens += mtp_stats.drafted_tokens
                    self.mtp_accepted_tokens += mtp_stats.accepted_tokens
                self.last_request = {
                    "id": summary.request_id,
                    "prompt_tokens": summary.prompt_tokens,
                    "completion_tokens": summary.completion_tokens,
                    "finish_reason": summary.finish_reason,
                    "prefix_cache_hit_tokens": reused_prompt_tokens,
                    "computed_prefill_tokens": computed_prompt_tokens,
                    **summary.metrics,
                }
            return summary
        except BaseException:
            with self._state_lock:
                self._cached_input_ids = ()
                self._cached_logits = None
                self._cache_sessions.clear()
                self.total_requests += 1
                self.failed_requests += 1
            raise
        finally:
            self._unregister_cancel(prepared)

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            values = {
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "mtp_requests": self.mtp_requests,
                "mtp_cycles": self.mtp_cycles,
                "mtp_drafted_tokens": self.mtp_drafted_tokens,
                "mtp_accepted_tokens": self.mtp_accepted_tokens,
                "mtp_acceptance_rate": (
                    self.mtp_accepted_tokens / self.mtp_drafted_tokens
                    if self.mtp_drafted_tokens
                    else 0.0
                ),
                "last_request": self.last_request,
                "active_requests": len(self._cancel_events),
                "prefix_cache_queries": self.prefix_cache_queries,
                "prefix_cache_hits": self.prefix_cache_hits,
                "prefix_cache_hit_tokens": self.prefix_cache_hit_tokens,
                "prefix_cache_sessions": len(self._cache_sessions),
                "prefix_cache_snapshots": int(
                    bool(self._cache_sessions and self._cached_input_ids)
                ),
                "prefix_cache_tokens": (len(self._cached_input_ids) if self._cache_sessions else 0),
                "prefix_cache_mode": "single_device_hot_prefix",
            }
        return {
            "status": "ok",
            "ready": True,
            "model": self.model_name,
            "model_type": self.model_type,
            "vision_available": self.vision is not None,
            "video_available": self.vision is not None,
            "vision_supported": self.vision_supported,
            "mtp_available": self.mtp is not None,
            "mtp_supported": self.mtp_supported,
            "vision_enabled_default": True,
            "mtp_enabled_default": True,
            "max_context": self.model.max_context,
            "context_capacity": self.model.max_context,
            "prefill_chunk_size": self.prefill_chunk_size,
            "busy": self._generation_lock.locked(),
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_cache_bytes": int(mx.get_cache_memory()),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            **values,
        }


def _chunk(
    prepared: _PreparedRequest,
    delta: Mapping[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": prepared.request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "",
        "choices": [
            {
                "index": 0,
                "delta": dict(delta),
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def create_app(worker: FlashNextTextWorker) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.worker = worker

    @app.get("/health")
    @app.get("/api/status")
    def health() -> dict[str, Any]:
        status = worker.status()
        image_available = getattr(worker, "vision", None) is not None
        vision_supported = bool(
            getattr(worker, "vision_supported", image_available)
        )
        status["model_capabilities"] = {
            "architecture_family": worker.model_type,
            "source": f"architecture-registry:{worker.model_type}",
            "features": {
                "text": True,
                "image_input": vision_supported,
                "video_input": vision_supported,
                "audio_input": False,
                "audio_output": False,
                "full_duplex": False,
                "mtp": bool(
                    getattr(
                        worker,
                        "mtp_supported",
                        getattr(worker, "mtp", None) is not None,
                    )
                ),
            },
        }
        status["duplex_available"] = False
        return status

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": worker.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "mfq",
                }
            ],
        }

    @app.get("/realtime/capabilities")
    def realtime_capabilities() -> dict[str, Any]:
        return {"available": False, "modes": []}

    @app.post("/api/runtime/sessions/fork")
    async def fork_session(request: Request) -> Any:
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse(status_code=400, content={"error": "body must be an object"})
        source = payload.get("source_session_id")
        target = payload.get("target_session_id")
        if not isinstance(source, str) or not isinstance(target, str):
            return JSONResponse(
                status_code=400,
                content={"error": "source_session_id and target_session_id must be strings"},
            )
        try:
            copied = worker.fork_session(source, target)
        except FlashNextWorkerError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})
        return {"status": "ok", "copied_snapshots": copied}

    @app.delete("/api/runtime/sessions/{session_id}")
    def close_session(session_id: str) -> dict[str, Any]:
        released, cancelled = worker.close_session(session_id)
        return {
            "status": "ok",
            "released_snapshots": released,
            "cancelled": cancelled,
        }

    @app.post("/api/runtime/sessions/{session_id}/cancel")
    def cancel_session(session_id: str) -> dict[str, Any]:
        return {"cancelled": worker.cancel(session_id)}

    @app.post("/api/runtime/cache/clear")
    def clear_cache() -> dict[str, Any]:
        if not worker.clear_cache():
            return {"cleared": False, "reason": "runtime_busy"}
        return {"cleared": True}

    @app.post("/api/reload")
    async def reload_runtime(request: Request) -> Any:
        payload = await request.json()
        context_size = int(payload.get("context_size", 0)) if isinstance(payload, dict) else 0
        if context_size in {0, worker.model.max_context}:
            return worker.status()
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "reload_requires_process_restart",
                    "message": (
                        "Flash-Next context changes require unloading and reloading the model"
                    ),
                }
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            try:
                payload = await request.json()
            except (UnicodeDecodeError, ValueError) as error:
                raise FlashNextWorkerError("request body must be valid JSON") from error
            if not isinstance(payload, dict):
                raise FlashNextWorkerError("request body must be an object")
            prepared = worker.prepare(payload)
        except FlashNextWorkerError as error:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": str(error),
                        "type": "invalid_request_error",
                        "code": "invalid_request",
                    }
                },
            )

        if payload.get("stream") is not True:
            deltas: list[dict[str, Any]] = []
            try:
                summary = await asyncio.to_thread(worker.generate, prepared, deltas.append)
            except Exception as error:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": str(error),
                            "type": "server_error",
                            "code": "generation_failed",
                        }
                    },
                )
            message: dict[str, Any] = {"role": "assistant", "content": summary.content}
            if summary.reasoning:
                message["reasoning_content"] = summary.reasoning
            return {
                "id": summary.request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": worker.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "logprobs": None,
                        "finish_reason": summary.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": summary.prompt_tokens,
                    "completion_tokens": summary.completion_tokens,
                    "total_tokens": summary.prompt_tokens + summary.completion_tokens,
                },
                "mfq_metrics": summary.metrics,
            }

        loop = asyncio.get_running_loop()
        events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def emit_delta(delta: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(events.put_nowait, ("delta", delta))

        def run_generation() -> None:
            try:
                summary = worker.generate(prepared, emit_delta)
            except BaseException as error:
                loop.call_soon_threadsafe(events.put_nowait, ("error", error))
            else:
                loop.call_soon_threadsafe(events.put_nowait, ("done", summary))

        task = asyncio.create_task(asyncio.to_thread(run_generation))

        async def stream() -> AsyncIterator[str]:
            try:
                ready = _chunk(prepared, {"role": "assistant"})
                ready["model"] = worker.model_name
                yield f"data: {json.dumps(ready, ensure_ascii=False)}\n\n"
                while True:
                    kind, value = await events.get()
                    if kind == "delta":
                        delta = (
                            {"reasoning_content": value["reasoning"]}
                            if "reasoning" in value
                            else {"content": value.get("content", "")}
                        )
                        chunk = _chunk(prepared, delta)
                        chunk["model"] = worker.model_name
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        continue
                    if kind == "error":
                        error = {
                            "id": prepared.request_id,
                            "error": {
                                "message": str(value),
                                "type": "server_error",
                                "code": "generation_failed",
                            },
                        }
                        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
                        break
                    if kind != "done" or not isinstance(value, _GenerationSummary):
                        break
                    finished = _chunk(prepared, {}, finish_reason=value.finish_reason)
                    finished["model"] = worker.model_name
                    yield f"data: {json.dumps(finished, ensure_ascii=False)}\n\n"
                    usage = {
                        "id": value.request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": worker.model_name,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": value.prompt_tokens,
                            "completion_tokens": value.completion_tokens,
                            "total_tokens": value.prompt_tokens + value.completion_tokens,
                        },
                        "mfq_metrics": value.metrics,
                    }
                    yield f"data: {json.dumps(usage, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
            finally:
                if not task.done() and prepared.session_id is not None:
                    worker.cancel(prepared.session_id)
                await task

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def run_worker(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ModuleNotFoundError as error:  # pragma: no cover - release dependency
        raise FlashNextWorkerError("the Flash-Next worker requires uvicorn") from error
    worker = FlashNextTextWorker.from_mfq(
        args.mfq,
        model_name=args.model_name,
        max_context=args.context_size,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    print(
        json.dumps(
            {
                "event": "flash_next_runtime_ready",
                "model": worker.model_name,
                "model_type": worker.model_type,
                "max_context": worker.model.max_context,
            }
        ),
        flush=True,
    )
    try:
        uvicorn.run(
            create_app(worker),
            host=args.host,
            port=args.port,
            log_level="warning",
            access_log=False,
        )
    finally:
        worker.close()
    return 0


__all__ = [
    "FlashNextTextWorker",
    "FlashNextWorkerError",
    "create_app",
    "load_flash_next_tokenizer",
    "run_worker",
]
