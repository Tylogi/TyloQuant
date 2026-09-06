"""Shared MLX vision runtime for Qwen3.8-Flash-Next and GLM-5.3-Flash."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.architectures.flash_next import (
    Glm5NextConfig,
    Qwen4ExpConfig,
    VisionTowerConfig,
)
from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.io import MfqTensor
from mfq.runtime.mlx_attention import attention
from mfq.runtime.mlx_linear import MlxNintModel, mlx_dense_array
from mfq.runtime.mlx_ops import MlxRMSNorm


def _dense_array(
    model: MlxNintModel,
    name: str,
    *,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the vision tower")
    value = model.tensors[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"vision tensor {name!r} must be dense")
    return mx.contiguous(mlx_dense_array(value, dtype=dtype))


def _dense_vector(model: MlxNintModel, name: str) -> mx.array:
    value = _dense_array(model, name, dtype=mx.float32)
    if value.ndim != 1:
        raise TypeError(f"vision tensor {name!r} must be a vector")
    return value


class _VisionLinear:
    def __init__(self, model: MlxNintModel, prefix: str) -> None:
        self.linear = model.linear(prefix + ".weight")
        bias_name = prefix + ".bias"
        self.bias = _dense_vector(model, bias_name) if bias_name in model.tensors else None

    def __call__(self, value: mx.array) -> mx.array:
        output = self.linear(value)
        return output if self.bias is None else output + self.bias.astype(output.dtype)


class _LayerNorm:
    def __init__(
        self,
        model: MlxNintModel,
        prefix: str,
        *,
        eps: float,
    ) -> None:
        self.weight = _dense_vector(model, prefix + ".weight")
        self.bias = _dense_vector(model, prefix + ".bias")
        self.eps = float(eps)

    def __call__(self, value: mx.array) -> mx.array:
        dtype = value.dtype
        source = value.astype(mx.float32)
        mean = mx.mean(source, axis=-1, keepdims=True)
        centered = source - mean
        variance = mx.mean(centered * centered, axis=-1, keepdims=True)
        normalized = centered * mx.rsqrt(variance + self.eps)
        return (normalized * self.weight + self.bias).astype(dtype)


def _gelu_tanh(value: mx.array) -> mx.array:
    source = value.astype(mx.float32)
    coefficient = math.sqrt(2.0 / math.pi)
    result = (
        0.5 * source * (1.0 + mx.tanh(coefficient * (source + 0.044715 * source * source * source)))
    )
    return result.astype(value.dtype)


def _gelu_exact(value: mx.array) -> mx.array:
    source = value.astype(mx.float32)
    result = 0.5 * source * (1.0 + mx.erf(source / math.sqrt(2.0)))
    return result.astype(value.dtype)


def _grid_array(
    grid_thw: mx.array | np.ndarray | Sequence[Sequence[int]],
    *,
    merge: int,
) -> np.ndarray:
    if isinstance(grid_thw, mx.array):
        mx.eval(grid_thw)
    grid = np.asarray(grid_thw, dtype=np.int64)
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
        raise ValueError("vision grid_thw must have [items,3] shape")
    if np.any(grid <= 0):
        raise ValueError("vision grid dimensions must be positive")
    if np.any(grid[:, 1:] % int(merge)):
        raise ValueError("vision grid height/width must divide spatial_merge_size")
    return np.ascontiguousarray(grid)


def _spatial_positions(height: int, width: int, merge: int) -> np.ndarray:
    rows, columns = np.meshgrid(
        np.arange(height, dtype=np.int32),
        np.arange(width, dtype=np.int32),
        indexing="ij",
    )
    shape = (height // merge, merge, width // merge, merge)
    rows = rows.reshape(shape).swapaxes(1, 2).reshape(-1)
    columns = columns.reshape(shape).swapaxes(1, 2).reshape(-1)
    return np.stack((rows, columns), axis=-1)


def vision_layout(
    grid_thw: mx.array | np.ndarray | Sequence[Sequence[int]],
    spatial_merge_size: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return block-major axial positions and per-frame attention lengths."""

    merge = int(spatial_merge_size)
    grid = _grid_array(grid_thw, merge=merge)
    positions: list[np.ndarray] = []
    lengths: list[int] = []
    for temporal, height, width in grid:
        spatial = _spatial_positions(int(height), int(width), merge)
        positions.append(np.tile(spatial, (int(temporal), 1)))
        lengths.extend([int(height * width)] * int(temporal))
    return np.ascontiguousarray(np.concatenate(positions)), tuple(lengths)


def qwen4_multimodal_positions(
    input_ids: mx.array | np.ndarray,
    modality_types: mx.array | np.ndarray,
    *,
    spatial_merge_size: int,
    image_grid_thw: mx.array | np.ndarray | Sequence[Sequence[int]] | None = None,
    video_grid_thw: mx.array | np.ndarray | Sequence[Sequence[int]] | None = None,
    attention_mask: mx.array | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Qwen4-Exp's exact 3-axis text/image/video MRoPE positions.

    Modality IDs follow the official contract: text=0, image=1, video=2.
    Videos are split into one grid per frame because the prompt places each
    frame between its own timestamp/vision sentinels.
    """

    if isinstance(input_ids, mx.array):
        mx.eval(input_ids)
    if isinstance(modality_types, mx.array):
        mx.eval(modality_types)
    ids = np.asarray(input_ids)
    modalities = np.asarray(modality_types)
    if ids.ndim == 1:
        ids = ids[None]
    if modalities.ndim == 1:
        modalities = modalities[None]
    if ids.ndim != 2 or modalities.shape != ids.shape:
        raise ValueError("Qwen4-Exp IDs/modality types must have matching [B,T] shape")
    if np.any((modalities < 0) | (modalities > 2)):
        raise ValueError("Qwen4-Exp modality types must be text=0, image=1, or video=2")
    merge = int(spatial_merge_size)
    if merge <= 0:
        raise ValueError("Qwen4-Exp spatial_merge_size must be positive")

    def optional_grid(
        value: mx.array | np.ndarray | Sequence[Sequence[int]] | None,
    ) -> np.ndarray:
        if value is None:
            return np.empty((0, 3), dtype=np.int64)
        return _grid_array(value, merge=merge)

    images = optional_grid(image_grid_thw)
    videos = optional_grid(video_grid_thw)
    video_frames = (
        np.concatenate(
            tuple(
                np.tile(np.asarray([[1, height, width]], dtype=np.int64), (temporal, 1))
                for temporal, height, width in videos
            ),
            axis=0,
        )
        if len(videos)
        else np.empty((0, 3), dtype=np.int64)
    )
    image_index = 0
    video_index = 0
    mask = None
    if attention_mask is not None:
        if isinstance(attention_mask, mx.array):
            mx.eval(attention_mask)
        mask = np.asarray(attention_mask, dtype=bool)
        if mask.ndim == 1:
            mask = mask[None]
        if mask.shape != ids.shape:
            raise ValueError("Qwen4-Exp attention mask must match input IDs")

    positions = np.zeros((3, ids.shape[0], ids.shape[1]), dtype=np.int32)
    deltas = np.empty((ids.shape[0], 1), dtype=np.int32)
    for batch in range(ids.shape[0]):
        valid = np.ones(ids.shape[1], dtype=bool) if mask is None else mask[batch]
        current_modalities = modalities[batch, valid]
        output: list[np.ndarray] = []
        current_position = 0
        begin = 0
        while begin < len(current_modalities):
            modality = int(current_modalities[begin])
            end = begin + 1
            while end < len(current_modalities) and int(current_modalities[end]) == modality:
                end += 1
            group_tokens = end - begin
            if modality == 0:
                output.append(
                    np.broadcast_to(
                        np.arange(group_tokens, dtype=np.int32)[None] + current_position,
                        (3, group_tokens),
                    ).copy()
                )
                current_position += group_tokens
            else:
                if modality == 1:
                    if image_index >= len(images):
                        raise ValueError("Qwen4-Exp prompt contains more image spans than grids")
                    temporal, height, width = (int(value) for value in images[image_index])
                    image_index += 1
                else:
                    if video_index >= len(video_frames):
                        raise ValueError(
                            "Qwen4-Exp prompt contains more video-frame spans than grids"
                        )
                    temporal, height, width = (int(value) for value in video_frames[video_index])
                    video_index += 1
                grid_height = height // merge
                grid_width = width // merge
                temporal_ids, row_ids, column_ids = np.meshgrid(
                    np.arange(temporal, dtype=np.int32),
                    np.arange(grid_height, dtype=np.int32),
                    np.arange(grid_width, dtype=np.int32),
                    indexing="ij",
                )
                vision_positions = np.stack(
                    (temporal_ids, row_ids, column_ids),
                    axis=0,
                ).reshape(3, -1)
                vision_positions += current_position
                if vision_positions.shape[1] != group_tokens:
                    raise ValueError("Qwen4-Exp vision placeholder length disagrees with its grid")
                output.append(vision_positions)
                current_position += max(grid_height, grid_width)
            begin = end
        combined = np.concatenate(output, axis=1) if output else np.empty((3, 0), dtype=np.int32)
        positions[:, batch, valid] = combined
        deltas[batch, 0] = int(combined.max()) + 1 - int(valid.sum()) if combined.size else 0
    if image_index != len(images) or video_index != len(video_frames):
        raise ValueError("Qwen4-Exp multimodal grids contain unused image/video entries")
    return positions, deltas


def _qwen_interpolation(
    grid: np.ndarray,
    *,
    side: int,
    merge: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for temporal, height_raw, width_raw in grid:
        height, width = int(height_raw), int(width_raw)
        positions = _spatial_positions(height, width, merge)
        row_source = positions[:, 0].astype(np.float32) * (side - 1) / max(height - 1, 1)
        col_source = positions[:, 1].astype(np.float32) * (side - 1) / max(width - 1, 1)
        row_floor = np.floor(row_source).astype(np.int32)
        col_floor = np.floor(col_source).astype(np.int32)
        row_taps = np.stack((row_floor, row_floor + 1), axis=-1)
        col_taps = np.stack((col_floor, col_floor + 1), axis=-1)
        row_weights = np.stack((row_floor + 1 - row_source, row_source - row_floor), axis=-1)
        col_weights = np.stack((col_floor + 1 - col_source, col_source - col_floor), axis=-1)
        row_taps = np.clip(row_taps, 0, side - 1)
        col_taps = np.clip(col_taps, 0, side - 1)
        image_indices = (row_taps[:, :, None] * side + col_taps[:, None, :]).reshape(-1, 4)
        image_weights = (row_weights[:, :, None] * col_weights[:, None, :]).reshape(-1, 4)
        indices.append(np.tile(image_indices, (int(temporal), 1)))
        weights.append(np.tile(image_weights, (int(temporal), 1)))
    return (
        np.ascontiguousarray(np.concatenate(indices), dtype=np.int32),
        np.ascontiguousarray(np.concatenate(weights), dtype=np.float32),
    )


def _apply_axial_rope(
    query: mx.array,
    key: mx.array,
    positions: np.ndarray,
    *,
    theta: float,
) -> tuple[mx.array, mx.array]:
    dimension = int(query.shape[-1])
    spatial_dimension = dimension // 2
    inverse = 1.0 / (
        float(theta) ** (np.arange(0, spatial_dimension, 2, dtype=np.float32) / spatial_dimension)
    )
    frequency = positions.astype(np.float32)[..., None] * inverse[None, None, :]
    angles = np.concatenate((frequency[:, 0], frequency[:, 1]), axis=-1)
    angles = np.concatenate((angles, angles), axis=-1)
    cosine = mx.cos(mx.array(angles))[..., None, :]
    sine = mx.sin(mx.array(angles))[..., None, :]

    def rotate(value: mx.array) -> mx.array:
        source_dtype = value.dtype
        source = value.astype(mx.float32)
        first, second = mx.split(source, 2, axis=-1)
        rotated = mx.concatenate((-second, first), axis=-1)
        return (source * cosine + rotated * sine).astype(source_dtype)

    return rotate(query), rotate(key)


def _packed_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    lengths: tuple[int, ...],
) -> mx.array:
    outputs: list[mx.array] = []
    offset = 0
    for length in lengths:
        end = offset + int(length)
        q = mx.transpose(query[offset:end], (1, 0, 2))[None]
        k = mx.transpose(key[offset:end], (1, 0, 2))[None]
        v = mx.transpose(value[offset:end], (1, 0, 2))[None]
        attended = attention(q, k, v, causal=False)
        outputs.append(mx.transpose(attended[0], (1, 0, 2)))
        offset = end
    if offset != int(query.shape[0]):
        raise ValueError("vision attention segment lengths do not cover all patches")
    return mx.concatenate(outputs, axis=0)


class _VisionAttention:
    def __init__(
        self,
        model: MlxNintModel,
        config: VisionTowerConfig,
        prefix: str,
        *,
        normalize_qk: bool,
    ) -> None:
        self.config = config
        self.qkv = _VisionLinear(model, prefix + ".qkv")
        self.output = _VisionLinear(model, prefix + ".output")
        dimension = config.hidden_size // config.num_heads
        self.q_norm = (
            MlxRMSNorm(
                _dense_vector(model, prefix + ".query_norm.weight"),
                config.rms_norm_eps,
            )
            if normalize_qk
            else None
        )
        self.k_norm = (
            MlxRMSNorm(
                _dense_vector(model, prefix + ".key_norm.weight"),
                config.rms_norm_eps,
            )
            if normalize_qk
            else None
        )
        self.head_dim = dimension

    def __call__(
        self,
        hidden_states: mx.array,
        positions: np.ndarray,
        lengths: tuple[int, ...],
    ) -> mx.array:
        tokens = int(hidden_states.shape[0])
        qkv = self.qkv(hidden_states).reshape(
            tokens,
            3,
            self.config.num_heads,
            self.head_dim,
        )
        query, key, value = (qkv[:, index] for index in range(3))
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query, key = _apply_axial_rope(
            query,
            key,
            positions,
            theta=self.config.rope_theta,
        )
        output = _packed_attention(query, key, value, lengths)
        return self.output(output.reshape(tokens, self.config.hidden_size))


class _QwenVisionBlock:
    def __init__(self, model: MlxNintModel, config: VisionTowerConfig, prefix: str) -> None:
        self.norm1 = _LayerNorm(model, prefix + ".norm1", eps=1e-6)
        self.attention = _VisionAttention(
            model,
            config,
            prefix + ".attention",
            normalize_qk=False,
        )
        self.norm2 = _LayerNorm(model, prefix + ".norm2", eps=1e-6)
        self.fc1 = _VisionLinear(model, prefix + ".mlp.up")
        self.fc2 = _VisionLinear(model, prefix + ".mlp.down")

    def __call__(
        self,
        hidden_states: mx.array,
        positions: np.ndarray,
        lengths: tuple[int, ...],
    ) -> mx.array:
        hidden_states = hidden_states + self.attention(
            self.norm1(hidden_states),
            positions,
            lengths,
        )
        return hidden_states + self.fc2(_gelu_tanh(self.fc1(self.norm2(hidden_states))))


class _GlmVisionBlock:
    def __init__(self, model: MlxNintModel, config: VisionTowerConfig, prefix: str) -> None:
        self.config = config
        self.norm1 = MlxRMSNorm(
            _dense_vector(model, prefix + ".norm1.weight"),
            config.rms_norm_eps,
        )
        self.attention = _VisionAttention(
            model,
            config,
            prefix + ".attention",
            normalize_qk=True,
        )
        self.norm2 = MlxRMSNorm(
            _dense_vector(model, prefix + ".norm2.weight"),
            config.rms_norm_eps,
        )
        self.gate = _VisionLinear(model, prefix + ".mlp.gate")
        self.up = _VisionLinear(model, prefix + ".mlp.up")
        self.down = _VisionLinear(model, prefix + ".mlp.down")

    def __call__(
        self,
        hidden_states: mx.array,
        positions: np.ndarray,
        lengths: tuple[int, ...],
    ) -> mx.array:
        hidden_states = hidden_states + self.attention(
            self.norm1(hidden_states),
            positions,
            lengths,
        )
        normalized = self.norm2(hidden_states)
        gate = mx.minimum(self.gate(normalized), self.config.swiglu_limit)
        up = mx.clip(self.up(normalized), -self.config.swiglu_limit, self.config.swiglu_limit)
        return hidden_states + self.down((gate * mx.sigmoid(gate)) * up)


class _PatchEmbedding:
    def __init__(self, model: MlxNintModel, config: VisionTowerConfig, prefix: str) -> None:
        self.config = config
        self.weight = _dense_array(model, prefix + ".weight")
        self.bias = _dense_vector(model, prefix + ".bias")
        expected = (
            config.hidden_size,
            config.in_channels,
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        if tuple(self.weight.shape) != expected:
            raise ValueError("vision patch-embedding weight shape disagrees with config")

    def __call__(self, pixel_values: mx.array | np.ndarray, patches: int) -> mx.array:
        value = pixel_values if isinstance(pixel_values, mx.array) else mx.array(pixel_values)
        if value.ndim == 5:
            value = value.reshape(int(value.shape[0]), -1)
        expected_width = (
            self.config.in_channels
            * self.config.temporal_patch_size
            * self.config.patch_size
            * self.config.patch_size
        )
        if tuple(value.shape) != (patches, expected_width):
            raise ValueError(
                "vision pixel_values must contain one flattened CxTxHxW patch per grid token"
            )
        weight = self.weight.reshape(self.config.hidden_size, expected_width)
        output = mx.matmul(value.astype(weight.dtype), mx.swapaxes(weight, -1, -2))
        return output + self.bias.astype(output.dtype)


class MlxQwenVision:
    """Shared Qwen vision tower with learned positions and patch merger."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: Qwen4ExpConfig | VisionTowerConfig,
        *,
        prefix: str = "vision",
    ) -> None:
        vision = config if isinstance(config, VisionTowerConfig) else config.vision
        if vision is None:
            raise ValueError("Qwen config does not contain a vision tower")
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = vision
        self.patch_embedding = _PatchEmbedding(
            self.model,
            self.config,
            prefix + ".patch_embedding",
        )
        self.position_embedding = _dense_array(self.model, prefix + ".position_embedding.weight")
        positions = self.config.num_position_embeddings
        if positions is None or int(math.isqrt(positions)) ** 2 != positions:
            raise ValueError("Qwen4-Exp learned vision position table must be square")
        self.position_side = int(math.isqrt(positions))
        self.blocks = tuple(
            _QwenVisionBlock(
                self.model,
                self.config,
                f"{prefix}.block.{index}",
            )
            for index in range(self.config.depth)
        )
        self.merger_norm = _LayerNorm(self.model, prefix + ".merger.norm", eps=1e-6)
        self.merger_fc1 = _VisionLinear(self.model, prefix + ".merger.mlp.up")
        self.merger_fc2 = _VisionLinear(self.model, prefix + ".merger.mlp.down")

    @classmethod
    def load_if_present(
        cls,
        causal_lm: object,
        config: object,
        *,
        prefix: str = "vision",
    ) -> MlxQwenVision | None:
        """Attach the shared Qwen vision tower when the loaded artifact has one."""

        model = causal_lm.model
        tensors = model.tensors
        has_config = getattr(config, "vision", None) is not None
        has_any_weight = any(name.startswith(prefix + ".") for name in tensors)
        probe = prefix + ".patch_embedding.weight"
        if not has_config or probe not in tensors:
            if has_any_weight:
                raise ValueError("Qwen MFQ contains an incomplete or unconfigured vision tower")
            return None
        return cls(model, config, prefix=prefix)

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: Qwen4ExpConfig | Mapping[str, object] | None = None,
        *,
        prefix: str = "vision",
        mmap: bool = True,
    ) -> MlxQwen4ExpVision:
        """Load the Qwen vision tower from a converted MFQ model."""

        model = MlxNintModel.from_mfq(path, mmap=mmap)
        try:
            selected = config
            if selected is None:
                if MODEL_CONFIG_ASSET not in model.tensors:
                    raise ValueError(
                        "MFQ has no embedded model config; pass Qwen4ExpConfig explicitly"
                    )
                payload = model.tensors[MODEL_CONFIG_ASSET]
                if not isinstance(payload, bytes):
                    raise TypeError("embedded MFQ model config must be a BLOB record")
                selected = json.loads(payload)
            normalized = (
                selected
                if isinstance(selected, Qwen4ExpConfig)
                else Qwen4ExpConfig.from_hf_config(selected)
            )
            return cls(model, normalized, prefix=prefix)
        except BaseException:
            model.close()
            raise

    def forward(
        self,
        pixel_values: mx.array | np.ndarray,
        grid_thw: mx.array | np.ndarray | Sequence[Sequence[int]],
    ) -> tuple[mx.array, mx.array]:
        grid = _grid_array(grid_thw, merge=self.config.spatial_merge_size)
        positions, lengths = vision_layout(grid, self.config.spatial_merge_size)
        patches = int(sum(lengths))
        hidden = self.patch_embedding(pixel_values, patches)
        interpolation_ids, interpolation_weights = _qwen_interpolation(
            grid,
            side=self.position_side,
            merge=self.config.spatial_merge_size,
        )
        learned = mx.sum(
            self.position_embedding[mx.array(interpolation_ids)]
            * mx.array(interpolation_weights)[..., None],
            axis=1,
        )
        hidden = hidden + learned.astype(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, positions, lengths)
        unit = self.config.spatial_merge_size**2
        normalized = self.merger_norm(hidden).reshape(-1, unit * self.config.hidden_size)
        merged = self.merger_fc2(_gelu_exact(self.merger_fc1(normalized)))
        return hidden, merged

    def __call__(
        self,
        pixel_values: mx.array | np.ndarray,
        grid_thw: mx.array | np.ndarray | Sequence[Sequence[int]],
    ) -> tuple[mx.array, mx.array]:
        return self.forward(pixel_values, grid_thw)

    def close(self) -> None:
        self.model.close()


MlxQwen4ExpVision = MlxQwenVision


class MlxGlm5NextVision:
    """GLM-5.3-Flash vision tower, downsampler, and bounded-SwiGLU merger."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: Glm5NextConfig,
        *,
        prefix: str = "vision",
    ) -> None:
        if config.vision is None:
            raise ValueError("GLM-5-Next config does not contain a vision tower")
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config.vision
        self.patch_embedding = _PatchEmbedding(
            self.model,
            self.config,
            prefix + ".patch_embedding",
        )
        self.blocks = tuple(
            _GlmVisionBlock(
                self.model,
                self.config,
                f"{prefix}.block.{index}",
            )
            for index in range(self.config.depth)
        )
        self.post_norm = MlxRMSNorm(
            _dense_vector(self.model, prefix + ".output_norm.weight"),
            self.config.rms_norm_eps,
        )
        self.downsample_weight = _dense_array(self.model, prefix + ".downsample.weight")
        self.downsample_bias = _dense_vector(self.model, prefix + ".downsample.bias")
        expected_downsample = (
            self.config.out_hidden_size,
            self.config.hidden_size,
            self.config.spatial_merge_size,
            self.config.spatial_merge_size,
        )
        if tuple(self.downsample_weight.shape) != expected_downsample:
            raise ValueError("GLM-5-Next vision downsampler shape disagrees with config")
        self.merger_projection = _VisionLinear(self.model, prefix + ".merger.projection")
        self.merger_norm = _LayerNorm(
            self.model,
            prefix + ".merger.norm",
            eps=1e-5,
        )
        self.merger_gate = _VisionLinear(self.model, prefix + ".merger.mlp.gate")
        self.merger_up = _VisionLinear(self.model, prefix + ".merger.mlp.up")
        self.merger_down = _VisionLinear(self.model, prefix + ".merger.mlp.down")

    @classmethod
    def load_if_present(
        cls,
        causal_lm: object,
        config: Glm5NextConfig,
        *,
        prefix: str = "vision",
    ) -> MlxGlm5NextVision | None:
        """Attach the GLM vision tower when the loaded artifact has one."""

        model = causal_lm.model
        tensors = model.tensors
        has_config = config.vision is not None
        has_any_weight = any(name.startswith(prefix + ".") for name in tensors)
        probe = prefix + ".patch_embedding.weight"
        if not has_config or probe not in tensors:
            if has_any_weight:
                raise ValueError("GLM MFQ contains an incomplete or unconfigured vision tower")
            return None
        return cls(model, config, prefix=prefix)

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: Glm5NextConfig | Mapping[str, object] | None = None,
        *,
        prefix: str = "vision",
        mmap: bool = True,
    ) -> MlxGlm5NextVision:
        """Load the GLM vision tower from a converted MFQ model."""

        model = MlxNintModel.from_mfq(path, mmap=mmap)
        try:
            selected = config
            if selected is None:
                if MODEL_CONFIG_ASSET not in model.tensors:
                    raise ValueError(
                        "MFQ has no embedded model config; pass Glm5NextConfig explicitly"
                    )
                payload = model.tensors[MODEL_CONFIG_ASSET]
                if not isinstance(payload, bytes):
                    raise TypeError("embedded MFQ model config must be a BLOB record")
                selected = json.loads(payload)
            normalized = (
                selected
                if isinstance(selected, Glm5NextConfig)
                else Glm5NextConfig.from_hf_config(selected)
            )
            return cls(model, normalized, prefix=prefix)
        except BaseException:
            model.close()
            raise

    def forward(
        self,
        pixel_values: mx.array | np.ndarray,
        grid_thw: mx.array | np.ndarray | Sequence[Sequence[int]],
    ) -> tuple[mx.array, mx.array]:
        grid = _grid_array(grid_thw, merge=self.config.spatial_merge_size)
        positions, lengths = vision_layout(grid, self.config.spatial_merge_size)
        patches = int(sum(lengths))
        hidden = self.patch_embedding(pixel_values, patches)
        for block in self.blocks:
            hidden = block(hidden, positions, lengths)
        hidden = self.post_norm(hidden)
        merge = self.config.spatial_merge_size
        grouped = mx.transpose(
            hidden.reshape(-1, merge, merge, self.config.hidden_size),
            (0, 3, 1, 2),
        ).reshape(-1, self.config.hidden_size * merge * merge)
        weight = self.downsample_weight.reshape(self.config.out_hidden_size, -1)
        downsampled = mx.matmul(grouped.astype(weight.dtype), mx.swapaxes(weight, -1, -2))
        downsampled = downsampled + self.downsample_bias.astype(downsampled.dtype)
        projected = self.merger_projection(downsampled)
        projected = _gelu_exact(self.merger_norm(projected))
        gate = mx.minimum(self.merger_gate(projected), self.config.swiglu_limit)
        up = mx.clip(
            self.merger_up(projected),
            -self.config.swiglu_limit,
            self.config.swiglu_limit,
        )
        merged = self.merger_down((gate * mx.sigmoid(gate)) * up)
        return downsampled, merged

    def __call__(
        self,
        pixel_values: mx.array | np.ndarray,
        grid_thw: mx.array | np.ndarray | Sequence[Sequence[int]],
    ) -> tuple[mx.array, mx.array]:
        return self.forward(pixel_values, grid_thw)

    def close(self) -> None:
        self.model.close()


def inject_vision_embeddings(
    token_embeddings: mx.array | np.ndarray,
    input_ids: mx.array | np.ndarray,
    vision_embeddings: mx.array | np.ndarray,
    vision_token_ids: int | Sequence[int],
) -> mx.array:
    """Replace image/video placeholder embeddings in flattened prompt order."""

    embeddings = (
        token_embeddings if isinstance(token_embeddings, mx.array) else mx.array(token_embeddings)
    )
    ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
    vision = (
        vision_embeddings
        if isinstance(vision_embeddings, mx.array)
        else mx.array(vision_embeddings)
    )
    if ids.ndim == 1:
        ids = ids[None]
    if embeddings.ndim != 3 or tuple(embeddings.shape[:2]) != tuple(ids.shape):
        raise ValueError("token embeddings and input IDs have incompatible shapes")
    if vision.ndim != 2 or int(vision.shape[-1]) != int(embeddings.shape[-1]):
        raise ValueError("vision embeddings have an incompatible width")
    selected_ids = (
        (int(vision_token_ids),)
        if isinstance(vision_token_ids, int)
        else tuple(int(value) for value in vision_token_ids)
    )
    if not selected_ids:
        raise ValueError("at least one vision placeholder token ID is required")
    mx.eval(ids)
    flat_ids = np.asarray(ids).reshape(-1)
    positions = np.flatnonzero(np.isin(flat_ids, selected_ids))
    if positions.size != int(vision.shape[0]):
        raise ValueError(
            "vision embedding count does not match image/video placeholder token count"
        )
    replacement = np.zeros(flat_ids.shape, dtype=np.int32)
    replacement[positions] = np.arange(positions.size, dtype=np.int32)
    mask = np.zeros(flat_ids.shape, dtype=np.bool_)
    mask[positions] = True
    flat = embeddings.reshape(-1, int(embeddings.shape[-1]))
    gathered = vision[mx.array(replacement)]
    output = mx.where(mx.array(mask)[:, None], gathered.astype(flat.dtype), flat)
    return output.reshape(embeddings.shape)


__all__ = [
    "MlxGlm5NextVision",
    "MlxQwen4ExpVision",
    "MlxQwenVision",
    "inject_vision_embeddings",
    "qwen4_multimodal_positions",
    "vision_layout",
]
